"""Recorder schema v1 runtime lifecycle around the legacy segment writer.

``EpisodeRecorder`` remains the writer for compatible metadata and telemetry
JSONL files.  This module owns only run/segment identity, durable lifecycle
artifacts, lightweight events, recovery reports, and the rosbag child process.
It is ROS-independent and never participates in control or FSM decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from threading import RLock
from typing import Any, Callable, Mapping, Optional, Sequence
from uuid import uuid4

import yaml

from .competition_context import CompetitionContext, task_set_fingerprint
from .fsm import InstructionParser
from .interface_contract import interface_contract_path
from .interfaces import (
    FSMStatus,
    FinalAction,
    TaskSpec,
)
from .recorder import EpisodeRecorder
from .recorder_contract import (
    load_recorder_contract,
    recorder_contract_path,
    recorder_contract_sha256,
)
from .recording_contracts import (
    ActionPairingConfig,
    strict_action_dispatch_from_json,
    strict_final_action_from_json,
)


_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")
_JSONL_NAMES = (
    "final_actions.jsonl",
    "action_dispatches.jsonl",
    "action_frames.jsonl",
    "action_pairing_issues.jsonl",
    "fsm_status.jsonl",
    "competition_contexts.jsonl",
)
_EXPECTED_ROSBAG_QOS = {
    "/tf": {
        "reliability": "best_effort",
        "durability": "volatile",
        "history": "keep_last",
        "depth": 100,
    },
    "/tf_static": {
        "reliability": "reliable",
        "durability": "transient_local",
        "history": "keep_last",
        "depth": 1,
    },
}

PROVENANCE_ENV = {
    "project_commit": "TEAM_SORTING_PROJECT_COMMIT",
    "project_branch": "TEAM_SORTING_PROJECT_BRANCH",
    "dirty_worktree": "TEAM_SORTING_DIRTY_WORKTREE",
    "official_server_image_id": "TEAM_SORTING_OFFICIAL_SERVER_IMAGE_ID",
    "official_client_image_id": "TEAM_SORTING_OFFICIAL_CLIENT_IMAGE_ID",
    "docker_image_digest": "TEAM_SORTING_DOCKER_IMAGE_DIGEST",
    "container_identity": "TEAM_SORTING_CONTAINER_IDENTITY",
}


class RecorderRuntimeState(str, Enum):
    NEW = "NEW"
    BOOTSTRAP_ACTIVE = "BOOTSTRAP_ACTIVE"
    RUN_ACTIVE = "RUN_ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class RecorderRuntimeConfig:
    root_dir: Path
    record_rosbag: bool
    rosbag_topics: tuple[str, ...]
    recovery_scan_enabled: bool
    bag_sigint_timeout_sec: float
    bag_terminate_timeout_sec: float
    bag_kill_timeout_sec: float
    observe_only: bool
    official_publish_enabled: bool
    rosbag_qos_overrides_path: Optional[Path] = None
    bag_startup_timeout_sec: float = 10.0
    bag_startup_poll_interval_sec: float = 0.02
    config_path: Optional[Path] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_dir", Path(self.root_dir).expanduser())
        object.__setattr__(self, "rosbag_topics", tuple(self.rosbag_topics))
        if type(self.record_rosbag) is not bool or type(self.recovery_scan_enabled) is not bool:
            raise ValueError("Recorder runtime布尔配置必须是严格bool")
        if type(self.observe_only) is not bool or type(self.official_publish_enabled) is not bool:
            raise ValueError("Recorder控制事实必须是严格bool")
        if self.observe_only and self.official_publish_enabled:
            raise ValueError(
                "observe_only=true时effective official publish gate必须为false"
            )
        for name in (
            "bag_sigint_timeout_sec",
            "bag_terminate_timeout_sec",
            "bag_kill_timeout_sec",
            "bag_startup_timeout_sec",
            "bag_startup_poll_interval_sec",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name}必须是正数且不能是bool")
            object.__setattr__(self, name, float(value))
        if self.config_path is not None:
            object.__setattr__(self, "config_path", Path(self.config_path))
        qos_path = self.rosbag_qos_overrides_path
        if self.record_rosbag and qos_path is None:
            raise ValueError("record_rosbag=true时必须显式提供rosbag QoS override文件")
        if qos_path is not None:
            object.__setattr__(
                self,
                "rosbag_qos_overrides_path",
                validate_rosbag_qos_overrides_path(Path(qos_path)),
            )


@dataclass
class _ActiveSegment:
    segment_id: str
    kind: str
    parent_run_id: Optional[str]
    directory: Path
    event_path: Path
    recorder: EpisodeRecorder
    data: dict[str, Any]
    event_sequence: int = 0
    process: Optional[Any] = None
    bag_failure: str = ""
    bag_ready: bool = False
    bag_startup_failed: bool = False
    shutdown_event_written: bool = False
    recorder_finished: bool = False
    recorder_finished_event_written: bool = False
    complete_created: bool = False
    accepting_writes: bool = True
    closing_started: bool = False
    finalize_run_requested: bool = False
    process_ending_requested: bool = False
    close_reason: Optional[str] = None
    close_clean: Optional[bool] = None
    close_ros_ns: Optional[int] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_prefix(path: Path, byte_end_offset: int) -> str:
    digest = hashlib.sha256()
    remaining = byte_end_offset
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(
                    f"共享事件流短于声明prefix：path={path}, offset={byte_end_offset}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_error(
    kind: str,
    path: Path,
    segment_id: Optional[str],
    run_id: Optional[str],
    exc: BaseException,
) -> RuntimeError:
    return RuntimeError(
        f"Recorder artifact写入失败：type={kind}, path={path}, "
        f"segment={segment_id}, run={run_id}, cause={type(exc).__name__}: {exc}"
    )


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    artifact_type: str,
    segment_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """Durably replace one JSON object and fsync its parent directory."""

    temporary_fd: Optional[int] = None
    temporary_path: Optional[Path] = None
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        temporary_fd, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temporary_path = Path(raw_path)
        with os.fdopen(temporary_fd, "wb") as stream:
            temporary_fd = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    except (OSError, TypeError, ValueError) as exc:
        raise _artifact_error(artifact_type, path, segment_id, run_id, exc) from exc
    finally:
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def exclusive_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    artifact_type: str,
) -> None:
    """Durably create one JSON object without following or replacing a target."""

    descriptor: Optional[int] = None
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except (OSError, TypeError, ValueError) as exc:
        raise _artifact_error(artifact_type, path, None, None, exc) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def create_marker(
    path: Path,
    payload: Mapping[str, Any],
    *,
    segment_id: str,
    run_id: Optional[str],
) -> None:
    """Exclusively create and fsync a lifecycle marker and its directory."""

    descriptor: Optional[int] = None
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _fsync_directory(path.parent)
    except (OSError, TypeError, ValueError) as exc:
        raise _artifact_error("marker", path, segment_id, run_id, exc) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def remove_marker(path: Path, *, segment_id: str, run_id: Optional[str]) -> None:
    try:
        path.unlink()
        _fsync_directory(path.parent)
    except OSError as exc:
        raise _artifact_error("marker_remove", path, segment_id, run_id, exc) from exc


def append_jsonl(
    path: Path,
    payload: Mapping[str, Any],
    *,
    artifact_type: str,
    segment_id: Optional[str],
    run_id: Optional[str],
    allowed_root: Optional[Path] = None,
) -> None:
    """Append one strict JSON line without following the final path symlink."""

    parent_descriptor: Optional[int] = None
    descriptor: Optional[int] = None
    try:
        line = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        if "\n" in line or "\r" in line:
            raise ValueError("JSONL编码结果不得包含物理换行")
        encoded = (line + "\n").encode("utf-8")
        parent = path.parent
        if allowed_root is not None:
            root = allowed_root.resolve()
            if parent.is_symlink():
                raise ValueError(f"JSONL父目录不得是symlink：{parent}")
            _resolved_under(root, parent, "JSONL parent")
            _resolved_under(root, path, "JSONL path")
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(parent, directory_flags)
        try:
            existing = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError(f"JSONL目标必须是普通文件：{path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, 0o640, dir_fd=parent_descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"JSONL目标必须是普通文件：{path}")
        with os.fdopen(descriptor, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise _artifact_error(artifact_type, path, segment_id, run_id, exc) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if parent_descriptor is not None:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    """Decode strict JSON, rejecting duplicate keys and non-finite constants."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}包含重复JSON key：{key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label}包含非有限JSON常量：{value}")

    def strict_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{label}包含非有限JSON数字：{value}")
        return parsed

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_float=strict_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label}不是严格JSON对象：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}顶层必须是对象")
    return payload


def _read_regular_bytes_no_follow(path: Path, allowed_root: Path) -> bytes:
    """Read a regular file through a checked parent dirfd and O_NOFOLLOW."""

    parent = path.parent
    root = allowed_root.resolve()
    if parent.is_symlink():
        raise ValueError(f"持久化文件父目录不得是symlink：{parent}")
    _resolved_under(root, parent, "persistent file parent")
    _resolved_under(root, path, "persistent file path")
    parent_descriptor: Optional[int] = None
    descriptor: Optional[int] = None
    try:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(parent, directory_flags)
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"持久化文件必须是普通文件：{path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _unknown(source: str, reason: str) -> dict[str, Any]:
    return {"value": None, "status": "unavailable", "reason": reason, "source": source}


def _available(value: Any, source: str) -> dict[str, Any]:
    return {"value": value, "status": "available", "reason": None, "source": source}


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}必须是非空字符串")
    if value in {".", ".."}:
        raise ValueError(f"{label}不能是'.'或'..'")
    if "\x00" in value or "/" in value or "\\" in value:
        raise ValueError(f"{label}不能包含路径分隔符、反斜杠或NUL")
    if _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{label}只能包含字母、数字、点、下划线和连字符")
    return value


def validate_rosbag_qos_overrides_path(path: Path) -> Path:
    """Validate the dedicated TF rosbag QoS file without following its final entry."""

    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("rosbag QoS override路径必须是绝对路径，不能依赖cwd")
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise ValueError(f"rosbag QoS override文件缺失或不可访问：{candidate}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"rosbag QoS override路径不得是符号链接：{candidate}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"rosbag QoS override路径必须是普通文件：{candidate}")
    if metadata.st_size <= 0:
        raise ValueError(f"rosbag QoS override文件不能为空：{candidate}")

    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if key in result:
                raise ValueError(f"rosbag QoS override包含重复key：{key!r}")
            result[key] = loader.construct_object(value_node, deep=True)
        return result

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        text = candidate.read_text(encoding="utf-8")
        payload = yaml.load(text, Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"rosbag QoS override文件不可读或YAML非法：{candidate}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("rosbag QoS override顶层必须是映射")
    if set(payload) != set(_EXPECTED_ROSBAG_QOS):
        raise ValueError(
            "rosbag QoS override必须且只能定义/tf与/tf_static"
        )
    for topic, expected in _EXPECTED_ROSBAG_QOS.items():
        profile = payload[topic]
        if not isinstance(profile, Mapping) or set(profile) != set(expected):
            raise ValueError(f"rosbag QoS override {topic} 字段不完整或包含未知字段")
        for field_name, expected_value in expected.items():
            actual = profile[field_name]
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(
                    f"rosbag QoS override {topic}.{field_name}必须为{expected_value!r}，"
                    f"实际={actual!r}"
                )
    return candidate.resolve(strict=True)


def resolve_rosbag_qos_overrides_path(
    config_path: Path, configured_path: object
) -> Path:
    """Resolve one package resource beside the active config, never against cwd."""

    if not isinstance(configured_path, str):
        raise ValueError("recorder.rosbag.qos_overrides_path必须是字符串")
    resource_name = _safe_component(
        configured_path.strip(), "recorder.rosbag.qos_overrides_path"
    )
    config_file = Path(config_path).resolve(strict=True)
    return validate_rosbag_qos_overrides_path(config_file.parent / resource_name)


def _resolved_under(root: Path, path: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label}解析后逃逸dataset_root：{resolved}") from exc
    return resolved


class RecorderRuntimeManager:
    """Run-aware lifecycle manager that delegates raw files to EpisodeRecorder."""

    def __init__(
        self,
        config: RecorderRuntimeConfig,
        pairing_config: ActionPairingConfig,
        *,
        process_factory: Callable[[Sequence[str]], Any] = subprocess.Popen,
        environment: Optional[Mapping[str, str]] = None,
        wall_now: Callable[[], str] = _utc_now,
        monotonic_now_ns: Callable[[], int] = time.monotonic_ns,
        steady_now: Callable[[], float] = time.monotonic,
        startup_wait: Callable[[float], None] = time.sleep,
        bag_ready_probe: Optional[Callable[[Path], bool]] = None,
        pid: Optional[int] = None,
    ) -> None:
        if not isinstance(config, RecorderRuntimeConfig):
            raise ValueError("config必须是RecorderRuntimeConfig")
        if not isinstance(pairing_config, ActionPairingConfig):
            raise ValueError("pairing_config必须是ActionPairingConfig")
        # Loading validates that runtime code is bound to the frozen v1 contract.
        self.contract = load_recorder_contract()
        self.config = config
        self.pairing_config = pairing_config
        self._process_factory = process_factory
        source_environment = os.environ if environment is None else environment
        allowed_environment = {
            *PROVENANCE_ENV.values(),
            "ROS_DOMAIN_ID",
            "RMW_IMPLEMENTATION",
        }
        self._environment = {
            key: source_environment[key]
            for key in allowed_environment
            if key in source_environment
        }
        self._wall_now = wall_now
        self._monotonic_now_ns = monotonic_now_ns
        self._steady_now = steady_now
        self._startup_wait = startup_wait
        self._bag_ready_probe = bag_ready_probe or self._default_bag_ready_probe
        self._pid = os.getpid() if pid is None else pid
        if type(self._pid) is not int or self._pid <= 0:
            raise ValueError("pid必须是正整数且不能是bool")
        self.state = RecorderRuntimeState.NEW
        self._active: Optional[_ActiveSegment] = None
        self._last_valid_context: Optional[CompetitionContext] = None
        self._last_finished_run_id: Optional[str] = None
        self._last_instruction_signature: Optional[str] = None
        self._recovery_reports: list[dict[str, Any]] = []
        self._blocked_run_ids: set[str] = set()
        self._process_start_wall_utc = self._wall_now()
        self._node_start_ros_ns: Optional[int] = None
        self._lock = RLock()

    @property
    def current_recorder(self) -> EpisodeRecorder:
        if self._active is None:
            raise RuntimeError("Recorder runtime当前没有活动segment")
        return self._active.recorder

    @property
    def current_segment_id(self) -> Optional[str]:
        return None if self._active is None else self._active.segment_id

    @property
    def current_segment_dir(self) -> Optional[Path]:
        return None if self._active is None else self._active.directory

    @property
    def action_pairing_enabled(self) -> bool:
        return self._active is not None and self._active.recorder.action_pairing_enabled

    @property
    def metadata(self) -> Any:
        return None if self._active is None else self._active.recorder.metadata

    def start(self, now_ros_ns: int) -> Path:
        with self._lock:
            if self.state is not RecorderRuntimeState.NEW:
                raise RuntimeError(f"Recorder runtime只能从NEW启动，当前={self.state.value}")
            self._require_ns(now_ros_ns, "now_ros_ns")
            self._node_start_ros_ns = now_ros_ns
            root = self._prepare_root()
            if self.config.recovery_scan_enabled:
                self._recovery_reports = self._scan_recovery(root)
            try:
                segment = self._open_segment("bootstrap", None, now_ros_ns)
                self.state = RecorderRuntimeState.BOOTSTRAP_ACTIVE
                self._write_event(
                    "recorder_started",
                    now_ros_ns,
                    self._monotonic_now_ns(),
                    payload={
                        "implementation": "recorder_schema_v1_runtime_b2",
                        "runtime_provenance": self._runtime_provenance(),
                    },
                    source={"kind": "RecorderRuntimeManager", "derived": True},
                    source_event_ids=(),
                )
                self._write_pending_bag_started(now_ros_ns)
                for report in self._recovery_reports:
                    if not report.get("_newly_created", False):
                        continue
                    self._write_event(
                        "unclean_shutdown_detected",
                        now_ros_ns,
                        self._monotonic_now_ns(),
                        payload={
                            "recovery_report_id": report["report_id"],
                            "recovery_report": str(
                                Path("recovery") / f"{report['report_id']}.json"
                            ),
                            "issue_types": report["issue_types"],
                            "recovery_action": "none",
                        },
                        source={"kind": "RecoveryScanner", "derived": True},
                        source_event_ids=(),
                    )
                return segment.directory
            except Exception:
                # A partially opened segment must never remain writable after a
                # failed start.  Close what can be closed and leave durable
                # evidence if even that cleanup fails.
                if self._active is not None:
                    self.state = RecorderRuntimeState.BOOTSTRAP_ACTIVE
                    try:
                        self._close_active(
                            now_ros_ns,
                            "startup_failed",
                            clean=False,
                            process_ending=True,
                        )
                    except Exception:
                        pass
                self.state = RecorderRuntimeState.FAILED
                raise

    def record_final_action_payload(
        self, raw_payload: object, receive_ros_ns: int, receive_monotonic_ns: int
    ) -> tuple[str, ...]:
        with self._lock:
            active = self._require_active_for_write()
            path = active.directory / "final_actions.jsonl"
            before = path.stat().st_size if path.exists() else 0
            issues = active.recorder.ingest_final_action_payload(
                raw_payload, receive_ros_ns, receive_monotonic_ns
            )
            self._touch_ros(receive_ros_ns)
            after = path.stat().st_size if path.exists() else 0
            if after > before:
                try:
                    action = strict_final_action_from_json(raw_payload)  # type: ignore[arg-type]
                    self._safe_event(
                        "action_selected",
                        receive_ros_ns,
                        receive_monotonic_ns,
                        payload={
                            "sequence": action.sequence,
                            "generated_timestamp_ns": action.timestamp_ns,
                            "artifact": "final_actions.jsonl",
                            "actual_publish_claimed": False,
                        },
                        source={"topic": "/team/final_action", "derived": False},
                    )
                except ValueError:
                    pass
            self._record_pairing_issues(issues, receive_ros_ns, receive_monotonic_ns)
            return issues

    def record_final_action(self, action: FinalAction) -> None:
        """Compatibility path when strict asynchronous pairing is disabled."""

        with self._lock:
            active = self._require_active_for_write()
            active.recorder.record_final_action(action)
            self._touch_ros(action.timestamp_ns)
            self._safe_event(
                "action_selected",
                action.timestamp_ns,
                self._monotonic_now_ns(),
                payload={
                    "sequence": action.sequence,
                    "generated_timestamp_ns": action.timestamp_ns,
                    "artifact": "final_actions.jsonl",
                    "actual_publish_claimed": False,
                },
                source={"topic": "/team/final_action", "derived": False},
            )

    def record_action_dispatch_payload(
        self, raw_payload: object, receive_ros_ns: int, receive_monotonic_ns: int
    ) -> tuple[str, ...]:
        with self._lock:
            active = self._require_active_for_write()
            path = active.directory / "action_dispatches.jsonl"
            before = path.stat().st_size if path.exists() else 0
            issues = active.recorder.ingest_action_dispatch_payload(
                raw_payload, receive_ros_ns, receive_monotonic_ns
            )
            self._touch_ros(receive_ros_ns)
            after = path.stat().st_size if path.exists() else 0
            if after > before:
                try:
                    dispatch = strict_action_dispatch_from_json(raw_payload)  # type: ignore[arg-type]
                    common = {
                        "sequence": dispatch.sequence,
                        "generated_timestamp_ns": dispatch.timestamp_ns,
                        "artifact": "action_dispatches.jsonl",
                        "dispatch_mode": dispatch.dispatch_mode.value,
                        "controller_accepted": None,
                        "execution_confirmed": None,
                    }
                    if dispatch.publish_attempted:
                        self._safe_event(
                            "dispatch_attempted",
                            receive_ros_ns,
                            receive_monotonic_ns,
                            payload={**common, "attempted_groups": list(dispatch.attempted_groups)},
                            source={"topic": "/team/action_dispatch", "derived": False},
                        )
                        outcome = (
                            "dispatch_succeeded"
                            if dispatch.publisher_call_succeeded is True
                            else "dispatch_failed"
                        )
                        self._safe_event(
                            outcome,
                            receive_ros_ns,
                            receive_monotonic_ns,
                            payload={
                                **common,
                                "publisher_call_succeeded": dispatch.publisher_call_succeeded,
                                "publisher_failure_reason": dispatch.failure_reason,
                                "dispatched_mask": list(dispatch.dispatched_mask),
                                "publisher_success_scope": "local_call_only",
                            },
                            source={"topic": "/team/action_dispatch", "derived": False},
                        )
                except ValueError:
                    pass
            self._record_pairing_issues(issues, receive_ros_ns, receive_monotonic_ns)
            return issues

    def prune_action_pairs(
        self, receive_ros_ns: int, receive_monotonic_ns: int
    ) -> tuple[str, ...]:
        with self._lock:
            active = self._require_active_for_write()
            issues = active.recorder.prune_action_pairs(
                receive_ros_ns, receive_monotonic_ns
            )
            self._record_pairing_issues(issues, receive_ros_ns, receive_monotonic_ns)
            return issues

    def record_fsm_status(self, status: FSMStatus) -> None:
        with self._lock:
            self._require_active_for_write().recorder.record_fsm_status(status)
            self._touch_ros(status.timestamp_ns)

    def record_instruction(
        self, raw_text: str, receive_ros_ns: int, parser: InstructionParser
    ) -> Optional[TaskSpec]:
        with self._lock:
            recorder = self._require_active_for_write().recorder
            task = recorder.record_instruction(raw_text, receive_ros_ns, parser)
            self._touch_ros(receive_ros_ns)
            try:
                signature = "tasks:" + task_set_fingerprint(
                    recorder.metadata.parsed_tasks
                )
            except ValueError:
                signature = "raw:" + hashlib.sha256(
                    raw_text.encode("utf-8")
                ).hexdigest()
            if signature != self._last_instruction_signature:
                self._last_instruction_signature = signature
                self._safe_event(
                    "instruction_updated",
                    receive_ros_ns,
                    self._monotonic_now_ns(),
                    payload={
                        "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                        "parsed_task_count": len(recorder.metadata.parsed_tasks),
                        "parse_failure": recorder.metadata.instruction_parse_failure or None,
                    },
                    source={"topic": "/material/instruction", "derived": False},
                )
            return task

    def record_referee_message(
        self, topic: str, raw_value: str | int, receive_ros_ns: int
    ) -> None:
        with self._lock:
            self._require_active_for_write().recorder.record_referee_message(
                topic, raw_value, receive_ros_ns
            )
            self._touch_ros(receive_ros_ns)

    def record_competition_context_payload(
        self, raw_payload: object, receive_ros_ns: int, receive_monotonic_ns: int
    ) -> None:
        with self._lock:
            self._require_active_for_write()
            if not isinstance(raw_payload, str):
                self._record_invalid_context(
                    raw_payload,
                    "competition_context_payload_not_string",
                    receive_ros_ns,
                    receive_monotonic_ns,
                )
                return
            try:
                context = CompetitionContext.from_json(raw_payload)
            except ValueError as exc:
                self._record_invalid_context(
                    raw_payload,
                    f"competition_context_parse_error:{exc}",
                    receive_ros_ns,
                    receive_monotonic_ns,
                )
                return
            self.record_competition_context(
                context, receive_ros_ns, receive_monotonic_ns
            )

    def record_competition_context(
        self,
        context: CompetitionContext,
        receive_ros_ns: int,
        receive_monotonic_ns: Optional[int] = None,
    ) -> None:
        with self._lock:
            if not isinstance(context, CompetitionContext):
                raise ValueError("context必须是CompetitionContext")
            monotonic_ns = (
                self._monotonic_now_ns()
                if receive_monotonic_ns is None
                else receive_monotonic_ns
            )
            active = self._require_active_for_write()
            if not context.valid:
                self._record_context_diagnostic(
                    context,
                    context.failure_reason,
                    receive_ros_ns,
                    monotonic_ns,
                )
                return
            try:
                _safe_component(context.run_id, "CompetitionContext.run_id")
                _resolved_under(
                    self.config.root_dir,
                    self.config.root_dir / "runs" / context.run_id,
                    "run_id path",
                )
            except ValueError as exc:
                self._record_context_diagnostic(
                    context,
                    f"unsafe_run_id:{exc}",
                    receive_ros_ns,
                    monotonic_ns,
                )
                return

            if active.kind == "bootstrap":
                if context.finished:
                    try:
                        self._record_valid_context_fact(
                            context, receive_ros_ns, monotonic_ns
                        )
                    except Exception:
                        self._converge_context_failure(
                            receive_ros_ns, "bootstrap_context_commit_failed"
                        )
                        raise
                    return
                try:
                    self._validate_run_binding_available(context)
                except (ValueError, RuntimeError) as exc:
                    self._record_context_diagnostic(
                        context,
                        f"run_binding_failed:{type(exc).__name__}:{exc}",
                        receive_ros_ns,
                        monotonic_ns,
                    )
                    return
                try:
                    source_event_id = self._record_valid_context_fact(
                        context, receive_ros_ns, monotonic_ns
                    )
                except Exception:
                    self._converge_context_failure(
                        receive_ros_ns, "bootstrap_context_commit_failed"
                    )
                    raise
                try:
                    self._close_active(receive_ros_ns, "run_bound", clean=True)
                except Exception:
                    self._mark_failed_nonwritable()
                    raise
                self._open_run_segment(context, receive_ros_ns, source_event_id)
                return

            assert active.parent_run_id is not None
            if context.run_id != active.parent_run_id:
                try:
                    self._validate_run_binding_available(context)
                except (ValueError, RuntimeError) as exc:
                    self._record_context_diagnostic(
                        context,
                        f"run_binding_failed:{type(exc).__name__}:{exc}",
                        receive_ros_ns,
                        monotonic_ns,
                    )
                    return
                try:
                    boundary_event_id = self._write_event(
                        "run_changed",
                        receive_ros_ns,
                        monotonic_ns,
                        payload={
                            "previous_run_id": active.parent_run_id,
                            "new_run_id": context.run_id,
                            "new_context_sha256": hashlib.sha256(
                                context.to_json().encode("utf-8")
                            ).hexdigest(),
                            "task_set_fingerprint_merge_performed": False,
                            "rollover_phase": "requested",
                            "target_committed": False,
                        },
                        source={"kind": "RecorderContextBinding", "derived": True},
                        source_event_ids=(),
                    )
                except Exception:
                    self._converge_context_failure(
                        receive_ros_ns, "run_change_boundary_commit_failed"
                    )
                    raise
                try:
                    self._close_active(
                        receive_ros_ns, "run_changed", clean=True, finalize_run=True
                    )
                except Exception:
                    self._mark_failed_nonwritable()
                    raise
                self._open_run_segment(
                    context, receive_ros_ns, boundary_event_id
                )
                return

            if (
                self._last_valid_context is not None
                and context.task_set_fingerprint
                != self._last_valid_context.task_set_fingerprint
            ):
                self._record_context_diagnostic(
                    context,
                    "task_set_fingerprint_changed_within_run",
                    receive_ros_ns,
                    monotonic_ns,
                )
                return

            previous = self._last_valid_context
            try:
                source_event_id = self._record_valid_context_fact(
                    context, receive_ros_ns, monotonic_ns
                )
            except Exception:
                self._converge_context_failure(
                    receive_ros_ns, "same_run_context_commit_failed"
                )
                raise
            self._last_valid_context = context
            self._emit_transitions(previous, context, source_event_id, receive_ros_ns, monotonic_ns)
            self._observe_context(context)
            if context.finished and self._last_finished_run_id != context.run_id:
                self._last_finished_run_id = context.run_id
                try:
                    self._close_active(
                        receive_ros_ns, "official_finished", clean=True, finalize_run=True
                    )
                except Exception:
                    self._mark_failed_nonwritable()
                    raise
                self._last_valid_context = None
                try:
                    self._open_segment("bootstrap", None, receive_ros_ns)
                    self.state = RecorderRuntimeState.BOOTSTRAP_ACTIVE
                    self._write_event(
                        "recorder_started",
                        receive_ros_ns,
                        self._monotonic_now_ns(),
                        payload={"reason": "waiting_after_finished"},
                        source={"kind": "RecorderRuntimeManager", "derived": True},
                        source_event_ids=(),
                    )
                    self._write_pending_bag_started(receive_ros_ns)
                except Exception:
                    self._converge_context_failure(
                        receive_ros_ns, "post_finished_bootstrap_open_failed"
                    )
                    raise

    def _record_valid_context_fact(
        self,
        context: CompetitionContext,
        receive_ros_ns: int,
        receive_monotonic_ns: int,
    ) -> str:
        active = self._require_active_for_write()
        active.recorder.record_competition_context(context)
        active.data["context_valid_count"] += 1
        self._touch_ros(context.referee_timestamp_ns or receive_ros_ns)
        return self._write_event(
            "competition_context_updated",
            receive_ros_ns,
            receive_monotonic_ns,
            payload={"context": context.to_dict()},
            source={"topic": "/team/competition_context", "derived": False},
            source_event_ids=(),
            context_override=context,
        )

    def _record_context_diagnostic(
        self,
        context: CompetitionContext,
        reason: str,
        receive_ros_ns: int,
        receive_monotonic_ns: int,
    ) -> None:
        active = self._require_active_for_write()
        active.data["context_invalid_count"] += 1
        self._safe_event(
            "competition_context_updated",
            receive_ros_ns,
            receive_monotonic_ns,
            payload={
                "raw": context.to_dict(),
                "structured_context_persisted": False,
            },
            source={"topic": "/team/competition_context", "derived": False},
            validity="invalid",
            invalid_reasons=(reason,),
        )

    def monitor_bag(self, now_ros_ns: int) -> Optional[str]:
        with self._lock:
            if self._active is None or self._active.process is None:
                return None
            process = self._active.process
            exit_code = process.poll()
            if exit_code is None:
                return None
            self._active.process = None
            self._active.recorder.mark_rosbag_finished(int(exit_code))
            self._active.data["bag_exit_code"] = int(exit_code)
            self._active.data["warning_counters"]["bag_early_exit"] = (
                self._active.data["warning_counters"].get("bag_early_exit", 0) + 1
            )
            self._active.bag_failure = (
                f"ros2 bag record在segment运行期间提前退出，exit_code={exit_code}"
            )
            self._safe_event(
                "bag_stopped",
                now_ros_ns,
                self._monotonic_now_ns(),
                payload={
                    "exit_code": int(exit_code),
                    "normal": False,
                    "automatic_rollover": False,
                },
                source={"kind": "rosbag_process", "derived": False},
            )
            return self._active.bag_failure

    def close(self, now_ros_ns: int, reason: str = "node_shutdown") -> None:
        with self._lock:
            if self.state is RecorderRuntimeState.CLOSED:
                return
            if self.state is RecorderRuntimeState.NEW and self._active is None:
                self.state = RecorderRuntimeState.CLOSED
                return
            if self.state is RecorderRuntimeState.FAILED and self._active is None:
                self.state = RecorderRuntimeState.CLOSED
                return
            if self.state is RecorderRuntimeState.CLOSING:
                return
            self._require_ns(now_ros_ns, "now_ros_ns")
            self.state = RecorderRuntimeState.CLOSING
            try:
                if self._active is not None:
                    self._close_active(
                        now_ros_ns, reason, clean=True, process_ending=True
                    )
                self.state = RecorderRuntimeState.CLOSED
            except Exception:
                self.state = RecorderRuntimeState.FAILED
                raise

    def _prepare_root(self) -> Path:
        root = self.config.root_dir
        try:
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise NotADirectoryError(root)
            resolved = root.resolve()
            for role in ("bootstrap", "runs", "recovery"):
                path = resolved / role
                if path.is_symlink():
                    raise RuntimeError(
                        f"dataset role目录不得是symlink：role={role}, path={path}"
                    )
                path.mkdir(exist_ok=True)
                if not path.is_dir():
                    raise NotADirectoryError(path)
                _resolved_under(resolved, path, f"dataset role {role}")
            _fsync_directory(resolved)
            return resolved
        except OSError as exc:
            raise _artifact_error("dataset_root", root, None, None, exc) from exc

    def _open_segment(
        self, kind: str, parent_run_id: Optional[str], now_ros_ns: int
    ) -> _ActiveSegment:
        if kind not in {"bootstrap", "run_bound"}:
            raise ValueError("segment_kind必须是bootstrap或run_bound")
        if (kind == "bootstrap") != (parent_run_id is None):
            raise ValueError("bootstrap parent必须为null且run_bound parent必须非空")
        root = self.config.root_dir.resolve()
        if kind == "bootstrap":
            parent = root / "bootstrap"
            segment_sequence = 0
            event_path: Optional[Path] = None
        else:
            assert parent_run_id is not None
            run_dir = root / "runs" / parent_run_id
            parent = run_dir / "segments"
            self._validate_existing_run_integrity(
                parent_run_id,
                require_appendable=True,
                compare_current_runtime_facts=True,
            )
            manifest = self._read_json(run_dir / "manifest.json", "run_manifest")
            self._validate_run_manifest_identity(
                manifest, expected_run_id=parent_run_id
            )
            self._validate_run_events_path(run_dir)
            segment_sequence = len(manifest["recorder_segment_ids"])
            event_path = run_dir / "events.jsonl"
        segment_id = self._new_id("segment")
        directory = _resolved_under(root, parent / segment_id, "segment path")
        try:
            directory.mkdir()
            _fsync_directory(parent)
        except OSError as exc:
            raise _artifact_error("segment_directory", directory, segment_id, parent_run_id, exc) from exc
        marker_payload = {
            "schema_name": "team_sorting.recorder.marker",
            "schema_version": 1,
            "marker": "ACTIVE",
            "recorder_segment_id": segment_id,
            "parent_run_id": parent_run_id,
            "created_wall_utc": self._wall_now(),
        }
        create_marker(
            directory / "ACTIVE",
            marker_payload,
            segment_id=segment_id,
            run_id=parent_run_id,
        )
        data = self._initial_segment_data(
            segment_id, kind, parent_run_id, segment_sequence, now_ros_ns
        )
        self._validate_contract_record("recorder_segment_schema", data)
        atomic_write_json(
            directory / "segment.json",
            data,
            artifact_type="segment_manifest",
            segment_id=segment_id,
            run_id=parent_run_id,
        )
        if parent_run_id is not None:
            self._append_segment_to_run_manifest(parent_run_id, segment_id)
        try:
            recorder = EpisodeRecorder(parent, self.pairing_config)
            recorder.start(
                segment_id,
                now_ros_ns,
                "Recorder schema v1原始segment；不是Official Attempt或Training Episode",
                precreated_lifecycle_directory=True,
            )
            active = _ActiveSegment(
                segment_id=segment_id,
                kind=kind,
                parent_run_id=parent_run_id,
                directory=directory,
                event_path=(
                    directory / "events.jsonl" if event_path is None else event_path
                ),
                recorder=recorder,
                data=data,
            )
            self._active = active
            self.state = (
                RecorderRuntimeState.BOOTSTRAP_ACTIVE
                if kind == "bootstrap"
                else RecorderRuntimeState.RUN_ACTIVE
            )
            try:
                self._start_bag(active)
            except Exception:
                try:
                    self._close_active(
                        now_ros_ns, "segment_open_failed", clean=False
                    )
                finally:
                    self.state = RecorderRuntimeState.FAILED
                raise
            return active
        except Exception:
            self.state = RecorderRuntimeState.FAILED
            raise

    def _open_run_segment(
        self, context: CompetitionContext, now_ros_ns: int, source_event_id: str
    ) -> None:
        try:
            self._ensure_run_manifest(context, now_ros_ns)
            active = self._open_segment("run_bound", context.run_id, now_ros_ns)
            self.state = RecorderRuntimeState.RUN_ACTIVE
            self._write_pending_bag_started(now_ros_ns)
            active.recorder.record_competition_context(context)
            active.data["context_valid_count"] += 1
            self._write_event(
                "competition_context_updated",
                now_ros_ns,
                self._monotonic_now_ns(),
                payload={"context": context.to_dict(), "copied_at_run_boundary": True},
                source={"topic": "/team/competition_context", "derived": False},
                source_event_ids=(),
                context_override=context,
            )
            self._write_event(
                "run_bound",
                now_ros_ns,
                self._monotonic_now_ns(),
                payload={
                    "run_id": context.run_id,
                    "task_set_fingerprint": context.task_set_fingerprint,
                    "bootstrap_history_bound": False,
                    "binding_phase": "committed",
                    "committed": True,
                },
                source={"kind": "RecorderContextBinding", "derived": True},
                source_event_ids=(source_event_id,),
                context_override=context,
            )
        except Exception:
            self._converge_context_failure(now_ros_ns, "run_context_commit_failed")
            raise
        self._last_valid_context = context
        self._observe_context(context)

    def _append_segment_to_run_manifest(
        self, run_id: str, segment_id: str
    ) -> None:
        run_id = _safe_component(run_id, "run_id")
        segment_id = _safe_component(segment_id, "recorder_segment_id")
        run_dir = self.config.root_dir.resolve() / "runs" / run_id
        self._validate_run_events_path(run_dir)
        path = run_dir / "manifest.json"
        manifest = self._read_json(path, "run_manifest")
        self._validate_run_manifest_identity(manifest, expected_run_id=run_id)
        ids = manifest["recorder_segment_ids"]
        if segment_id in ids:
            raise RuntimeError("run manifest拒绝重复segment ID")
        ids.append(segment_id)
        self._validate_run_manifest_identity(manifest, expected_run_id=run_id)
        atomic_write_json(
            path,
            manifest,
            artifact_type="run_manifest",
            segment_id=segment_id,
            run_id=run_id,
        )

    def _close_active(
        self,
        now_ros_ns: int,
        reason: str,
        *,
        clean: bool,
        finalize_run: bool = False,
        process_ending: bool = False,
    ) -> None:
        active = self._active
        if active is None:
            return
        if active.close_reason is None:
            active.close_reason = reason
            active.close_clean = clean
            active.close_ros_ns = now_ros_ns
        active.finalize_run_requested = active.finalize_run_requested or finalize_run
        active.process_ending_requested = (
            active.process_ending_requested or process_ending
        )
        reason = active.close_reason
        clean = bool(active.close_clean)
        assert active.close_ros_ns is not None
        now_ros_ns = active.close_ros_ns
        finalize_run = active.finalize_run_requested
        process_ending = active.process_ending_requested
        active.accepting_writes = False
        active.closing_started = True
        if not active.shutdown_event_written:
            active.shutdown_event_written = self._safe_event(
                "shutdown_requested",
                now_ros_ns,
                self._monotonic_now_ns(),
                payload={"reason": reason},
                source={"kind": "RecorderRuntimeManager", "derived": True},
            ) is not None
        pairing_error: Optional[Exception] = None
        try:
            issues = active.recorder.close_action_pairing(
                now_ros_ns, self._monotonic_now_ns()
            )
            self._record_pairing_issues(
                issues, now_ros_ns, self._monotonic_now_ns()
            )
        except Exception as exc:  # bag ownership must still be discharged
            pairing_error = exc
            active.data["warning_counters"]["pairing_close_failure"] = 1
        self._stop_bag(active, now_ros_ns)
        self._validate_bag_completion(active)
        if pairing_error is not None:
            raise RuntimeError(
                f"关闭action pairer失败；rosbag已执行有界停止：{pairing_error}"
            ) from pairing_error
        if not active.recorder_finished:
            metadata = active.recorder.metadata
            active.recorder.finish(now_ros_ns)
            self._touch_ros(now_ros_ns)
            if metadata is not None:
                active.data["message_counters"] = dict(metadata.topic_counts)
            if process_ending:
                active.data["process_end_wall_utc"] = self._wall_now()
                active.data["node_end_ros_ns"] = now_ros_ns
            active.data["clean_shutdown"] = clean
            active.data["shutdown_reason"] = reason
            active.recorder_finished = True
        if not active.recorder_finished_event_written:
            active.recorder_finished_event_written = self._safe_event(
                "recorder_finished",
                now_ros_ns,
                self._monotonic_now_ns(),
                payload={"reason": reason, "clean_shutdown": clean},
                source={"kind": "RecorderRuntimeManager", "derived": True},
            ) is not None
        self._fsync_event_stream(active)
        active.data["jsonl_artifacts"] = self._artifact_inventory(active)
        active.data["marker_state"] = "complete"
        for internal_key in tuple(active.data):
            if internal_key.startswith("_"):
                active.data.pop(internal_key)
        self._validate_contract_record("recorder_segment_schema", active.data)
        atomic_write_json(
            active.directory / "segment.json",
            active.data,
            artifact_type="segment_manifest",
            segment_id=active.segment_id,
            run_id=active.parent_run_id,
        )
        if finalize_run and active.parent_run_id is not None:
            self._finalize_run_manifest(active.parent_run_id, now_ros_ns, reason)
        if not active.complete_created:
            create_marker(
                active.directory / "COMPLETE",
                {
                    "schema_name": "team_sorting.recorder.marker",
                    "schema_version": 1,
                    "marker": "COMPLETE",
                    "recorder_segment_id": active.segment_id,
                    "parent_run_id": active.parent_run_id,
                    "completed_wall_utc": self._wall_now(),
                },
                segment_id=active.segment_id,
                run_id=active.parent_run_id,
            )
            active.complete_created = True
        if (active.directory / "ACTIVE").exists():
            remove_marker(
                active.directory / "ACTIVE",
                segment_id=active.segment_id,
                run_id=active.parent_run_id,
            )
        self._active = None

    def _start_bag(self, active: _ActiveSegment) -> None:
        active.data["_pending_bag_started"] = None
        if not self.config.record_rosbag:
            return
        command = active.recorder.build_rosbag_command(
            self.config.rosbag_topics,
            qos_overrides_path=self.config.rosbag_qos_overrides_path,
        )
        try:
            process = self._process_factory(command)
            active.process = process
            active.recorder.mark_rosbag_started(command[4])
            active.data["bag_path"] = "rosbag"
            exit_code = process.poll()
            if exit_code is not None:
                active.recorder.mark_rosbag_finished(int(exit_code))
                active.data["bag_exit_code"] = int(exit_code)
                active.process = None
                raise RuntimeError(f"ros2 bag record启动后立即退出，退出码={exit_code}")
            active.data["_pending_bag_started"] = {
                "bag_path": "rosbag",
                "topics": list(self.config.rosbag_topics),
            }
        except Exception as exc:
            active.data["warning_counters"]["bag_start_failure"] = 1
            active.bag_failure = f"bag_start_failure:{type(exc).__name__}:{exc}"
            raise

    def _write_pending_bag_started(self, now_ros_ns: int) -> None:
        active = self._active
        if active is None:
            return
        pending = active.data.get("_pending_bag_started")
        if pending is not None:
            self._await_bag_ready(active)
            self._write_event(
                "bag_started",
                now_ros_ns,
                self._monotonic_now_ns(),
                payload=pending,
                source={"kind": "rosbag_process", "derived": False},
                source_event_ids=(),
            )
            active.data.pop("_pending_bag_started", None)

    @staticmethod
    def _default_bag_ready_probe(output_path: Path) -> bool:
        """Treat rosbag as ready only after its storage writer creates a file."""

        try:
            metadata = os.lstat(output_path)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                return False
            return any(
                not entry.is_symlink() and entry.is_file()
                for entry in output_path.iterdir()
                if entry.name != "metadata.yaml"
            )
        except OSError:
            return False

    def _await_bag_ready(self, active: _ActiveSegment) -> None:
        if not self.config.record_rosbag or active.bag_ready:
            return
        process = active.process
        if process is None:
            raise RuntimeError("rosbag ready确认失败：子进程不存在")
        output_path = active.directory / "rosbag"
        deadline = self._steady_now() + self.config.bag_startup_timeout_sec
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                active.process = None
                active.recorder.mark_rosbag_finished(int(exit_code))
                active.data["bag_exit_code"] = int(exit_code)
                active.data["warning_counters"]["bag_start_failure"] = 1
                active.bag_failure = (
                    "ros2 bag record在ready确认前退出，"
                    f"exit_code={int(exit_code)}"
                )
                raise RuntimeError(active.bag_failure)
            if self._bag_ready_probe(output_path):
                active.bag_ready = True
                return
            remaining = deadline - self._steady_now()
            if remaining <= 0:
                active.bag_startup_failed = True
                active.data["warning_counters"]["bag_startup_timeout"] = 1
                active.bag_failure = (
                    "ros2 bag record启动超时：未观察到已初始化的存储文件，"
                    f"timeout_sec={self.config.bag_startup_timeout_sec}"
                )
                raise RuntimeError(active.bag_failure)
            self._startup_wait(
                min(self.config.bag_startup_poll_interval_sec, remaining)
            )

    def _stop_bag(self, active: _ActiveSegment, now_ros_ns: int) -> None:
        process = active.process
        if process is None:
            return
        exit_code = process.poll()
        escalation = "none"
        if exit_code is not None:
            active.data["warning_counters"]["bag_early_exit"] = (
                active.data["warning_counters"].get("bag_early_exit", 0) + 1
            )
            active.bag_failure = (
                "ros2 bag record在停止请求前已经退出，"
                f"exit_code={int(exit_code)}"
            )
        if exit_code is None:
            if not active.bag_ready and not active.bag_startup_failed:
                self._await_bag_ready(active)
            if active.bag_startup_failed:
                process.terminate()
                escalation = "terminate_startup_failed"
            else:
                process.send_signal(signal.SIGINT)
                escalation = "sigint"
            try:
                exit_code = process.wait(
                    timeout=(
                        self.config.bag_terminate_timeout_sec
                        if active.bag_startup_failed
                        else self.config.bag_sigint_timeout_sec
                    )
                )
            except subprocess.TimeoutExpired:
                if not active.bag_startup_failed:
                    process.terminate()
                    escalation = "terminate"
                    active.data["warning_counters"]["bag_sigint_timeout"] = 1
                try:
                    exit_code = process.wait(
                        timeout=self.config.bag_terminate_timeout_sec
                    )
                except subprocess.TimeoutExpired:
                    process.kill()
                    escalation = "kill"
                    active.data["warning_counters"]["bag_terminate_timeout"] = 1
                    exit_code = process.wait(timeout=self.config.bag_kill_timeout_sec)
        active.process = None
        active.recorder.mark_rosbag_finished(int(exit_code))
        active.data["bag_exit_code"] = int(exit_code)
        if int(exit_code) != 0:
            active.data["warning_counters"]["bag_nonzero_exit"] = 1
        self._safe_event(
            "bag_stopped",
            now_ros_ns,
            self._monotonic_now_ns(),
            payload={
                "exit_code": int(exit_code),
                "normal": int(exit_code) == 0,
                "escalation": escalation,
                "automatic_rollover": False,
            },
            source={"kind": "rosbag_process", "derived": False},
        )

    def _validate_bag_completion(self, active: _ActiveSegment) -> None:
        if not self.config.record_rosbag:
            return
        exit_code = active.data.get("bag_exit_code")
        bag_path = active.directory / "rosbag"
        metadata_path = bag_path / "metadata.yaml"
        bag_directory_valid = False
        metadata_valid = False
        try:
            bag_metadata = os.lstat(bag_path)
            bag_directory_valid = stat.S_ISDIR(
                bag_metadata.st_mode
            ) and not stat.S_ISLNK(bag_metadata.st_mode)
            if bag_directory_valid:
                metadata = os.lstat(metadata_path)
                metadata_valid = (
                    stat.S_ISREG(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                    and metadata.st_size > 0
                )
        except OSError:
            pass
        failures: list[str] = []
        if type(exit_code) is not int or exit_code != 0:
            failures.append(f"bag_exit_code必须为0，实际={exit_code!r}")
        if not bag_directory_valid:
            failures.append(f"rosbag目录缺失或不是安全普通目录：{bag_path}")
        if not metadata_valid:
            failures.append(f"metadata.yaml缺失或不是安全普通文件：{metadata_path}")
        if active.bag_failure:
            failures.append(f"rosbag生命周期已记录真实失败：{active.bag_failure}")
        if failures:
            active.data["warning_counters"]["bag_completion_invalid"] = 1
            active.bag_failure = "；".join(failures)
            raise RuntimeError(
                "rosbag完整性检查失败，保留ACTIVE且拒绝创建COMPLETE："
                + active.bag_failure
            )

    def _initial_segment_data(
        self,
        segment_id: str,
        kind: str,
        parent_run_id: Optional[str],
        sequence: int,
        now_ros_ns: int,
    ) -> dict[str, Any]:
        if self._node_start_ros_ns is None:
            raise RuntimeError("Recorder node_start_ros_ns尚未初始化")
        return {
            "schema_name": "team_sorting.recorder.segment",
            "schema_version": 1,
            "recorder_segment_id": segment_id,
            "parent_run_id": parent_run_id,
            "segment_kind": kind,
            "segment_sequence": sequence,
            "process_start_wall_utc": self._process_start_wall_utc,
            "process_end_wall_utc": None,
            "node_start_ros_ns": self._node_start_ros_ns,
            "node_end_ros_ns": None,
            "first_ros_timestamp_ns": None,
            "last_ros_timestamp_ns": None,
            "pid": self._pid,
            "container_identity": self._injected_envelope("container_identity"),
            "clean_shutdown": None,
            "shutdown_reason": None,
            "bag_path": None,
            "bag_storage_identifier": _unknown(
                "rosbag metadata", "not_available_until_bag_metadata_is_inspected"
            ),
            "bag_exit_code": None,
            "jsonl_artifacts": [],
            "message_counters": {},
            "dropped_counters": {},
            "pairing_counters": {},
            "warning_counters": {},
            "observed_task_ids": [],
            "observed_settled_attempt_counts": [],
            "context_valid_count": 0,
            "context_invalid_count": 0,
            "marker_state": "active",
        }

    def _ensure_run_manifest(
        self, context: CompetitionContext, now_ros_ns: int
    ) -> dict[str, Any]:
        run_id = _safe_component(context.run_id, "run_id")
        if run_id in self._blocked_run_ids:
            raise RuntimeError(
                f"run_id={run_id}存在ACTIVE且无COMPLETE的旧segment，拒绝并发或静默竞争"
            )
        root = self.config.root_dir.resolve()
        run_dir = _resolved_under(root, root / "runs" / run_id, "run directory")
        manifest_path = run_dir / "manifest.json"
        if run_dir.exists():
            if (
                not run_dir.is_dir()
                or run_dir.is_symlink()
                or not manifest_path.is_file()
                or manifest_path.is_symlink()
            ):
                raise RuntimeError(f"run目录存在但manifest缺失或类型错误：{run_dir}")
            self._validate_existing_run_integrity(
                run_id,
                context=context,
                require_appendable=True,
                compare_current_runtime_facts=True,
            )
            self._validate_run_events_path(run_dir)
            manifest = self._read_json(manifest_path, "run_manifest")
            self._validate_run_manifest_identity(
                manifest,
                expected_run_id=run_id,
                expected_fingerprint=context.task_set_fingerprint,
            )
            if manifest.get("end_ros_ns") is not None:
                raise RuntimeError("现有run manifest已经逻辑结束，拒绝追加segment")
            segments = run_dir / "segments"
            if segments.is_symlink():
                raise RuntimeError(f"run segments目录不得是symlink：{segments}")
            segments.mkdir(exist_ok=True)
            return manifest
        try:
            run_dir.mkdir()
            (run_dir / "segments").mkdir()
            _fsync_directory(run_dir.parent)
        except OSError as exc:
            raise _artifact_error("run_directory", run_dir, None, run_id, exc) from exc
        manifest = self._new_run_manifest(context, now_ros_ns)
        self._validate_run_manifest_identity(
            manifest,
            expected_run_id=run_id,
            expected_fingerprint=context.task_set_fingerprint,
        )
        atomic_write_json(
            manifest_path,
            manifest,
            artifact_type="run_manifest",
            run_id=run_id,
        )
        return manifest

    def _validate_run_binding_available(self, context: CompetitionContext) -> None:
        """Check existing run identity before the current segment is sealed."""

        run_id = _safe_component(context.run_id, "run_id")
        if run_id in self._blocked_run_ids:
            raise RuntimeError(
                f"run_id={run_id}存在ACTIVE且无COMPLETE的旧segment，拒绝并发或静默竞争"
            )
        root = self.config.root_dir.resolve()
        run_dir = _resolved_under(root, root / "runs" / run_id, "run directory")
        if not run_dir.exists():
            return
        manifest_path = run_dir / "manifest.json"
        if (
            not run_dir.is_dir()
            or run_dir.is_symlink()
            or not manifest_path.is_file()
            or manifest_path.is_symlink()
        ):
            raise RuntimeError(f"run目录存在但manifest缺失或类型错误：{run_dir}")
        self._validate_existing_run_integrity(
            run_id,
            context=context,
            require_appendable=True,
            compare_current_runtime_facts=True,
        )
        self._validate_run_events_path(run_dir)
        manifest = self._read_json(manifest_path, "run_manifest")
        self._validate_run_manifest_identity(
            manifest,
            expected_run_id=run_id,
            expected_fingerprint=context.task_set_fingerprint,
        )
        if manifest.get("end_ros_ns") is not None:
            raise RuntimeError("现有run manifest已经逻辑结束，拒绝追加segment")

    def _validate_run_events_path(self, run_dir: Path) -> None:
        """Reject any existing run event target that is not an in-root regular file."""

        root = self.config.root_dir.resolve()
        _resolved_under(root, run_dir, "run events parent")
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise RuntimeError(f"run events父目录类型错误：{run_dir}")
        event_path = run_dir / "events.jsonl"
        if not os.path.lexists(event_path):
            return
        try:
            metadata = os.lstat(event_path)
        except OSError as exc:
            raise _artifact_error("run_events", event_path, None, run_dir.name, exc) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"run events不得是symlink：{event_path}")
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"run events必须是普通文件：{event_path}")
        _resolved_under(root, event_path, "run events path")

    def _validate_run_manifest_identity(
        self,
        manifest: Mapping[str, Any],
        *,
        expected_run_id: Optional[str] = None,
        expected_fingerprint: Optional[str] = None,
    ) -> None:
        """Validate persisted run identity beyond the generic frozen field shapes."""

        self._validate_contract_record("run_manifest_schema", manifest)
        run_id = _safe_component(manifest.get("run_id"), "run manifest run_id")
        if expected_run_id is not None and run_id != expected_run_id:
            raise RuntimeError("现有run manifest身份冲突：run_id不一致")
        if (
            expected_fingerprint is not None
            and manifest.get("task_set_fingerprint") != expected_fingerprint
        ):
            raise RuntimeError(
                "现有run manifest身份冲突：task_set_fingerprint不一致"
            )
        raw_ids = manifest.get("recorder_segment_ids")
        if not isinstance(raw_ids, list):
            raise RuntimeError("run manifest recorder_segment_ids必须是list")
        seen: set[str] = set()
        for index, raw_segment_id in enumerate(raw_ids):
            try:
                segment_id = _safe_component(
                    raw_segment_id, f"recorder_segment_ids[{index}]"
                )
            except ValueError as exc:
                raise RuntimeError(f"run manifest包含非法segment ID：{exc}") from exc
            if segment_id in seen:
                raise RuntimeError(
                    f"run manifest recorder_segment_ids包含重复值：{segment_id}"
                )
            seen.add(segment_id)

    def _validate_existing_run_integrity(
        self,
        run_id: str,
        *,
        context: Optional[CompetitionContext] = None,
        require_appendable: bool,
        compare_current_runtime_facts: bool,
    ) -> dict[str, Any]:
        report = self._inspect_existing_run_integrity(
            run_id,
            context=context,
            require_appendable=require_appendable,
            compare_current_runtime_facts=compare_current_runtime_facts,
        )
        if report["issue_types"]:
            terminal_note = (
                "；目标Run已逻辑结束或终态字段不一致"
                if "run_manifest_terminal_state_invalid" in report["issue_types"]
                else ""
            )
            suffix = (
                "; provenance change requires audited continuation, not implemented in B2"
                if "run_manifest_provenance_mismatch" in report["issue_types"]
                else ""
            )
            marker_state = (
                "active_and_complete"
                if "active_and_complete_both_present" in report["issue_types"]
                else "active_without_complete"
                if "active_without_complete" in report["issue_types"]
                else "integrity_invalid"
            )
            marker_note = (
                "；ACTIVE与COMPLETE同时存在"
                if marker_state == "active_and_complete"
                else "；ACTIVE存在但COMPLETE缺失"
                if marker_state == "active_without_complete"
                else ""
            )
            marker_read_note = (
                "；ACTIVE marker无法严格读取"
                if "active_marker_invalid" in report["issue_types"]
                else "；COMPLETE marker无法严格读取"
                if "complete_marker_invalid" in report["issue_types"]
                else ""
            )
            manifest_error = report[
                "immutable_contract_runtime_comparison"
            ].get("manifest_error", "")
            identity_note = (
                f"；{manifest_error}"
                if "task_set_fingerprint" in manifest_error
                else ""
            )
            raise RuntimeError(
                "目标Run完整性检查失败，拒绝追加："
                f"target_run_id={run_id}, "
                f"conflicting_segment={report.get('source_segment_identity')}, "
                f"marker_state={marker_state}, "
                f"refusal_reason={report['issue_types']}"
                f"{marker_note}{marker_read_note}{identity_note}"
                f"{terminal_note}{suffix}"
            )
        return report

    def _inspect_existing_run_integrity(
        self,
        run_id: str,
        *,
        context: Optional[CompetitionContext] = None,
        require_appendable: bool,
        compare_current_runtime_facts: bool,
    ) -> dict[str, Any]:
        run_id = _safe_component(run_id, "run integrity run_id")
        root = self.config.root_dir.resolve()
        run_dir = _resolved_under(root, root / "runs" / run_id, "run integrity")
        manifest_path = run_dir / "manifest.json"
        segments_dir = run_dir / "segments"
        events_path = run_dir / "events.jsonl"
        intrinsic_issues: list[str] = []
        append_compatibility_issues: list[str] = []
        # The remainder of the scanner records only persisted-data defects in
        # this list.  Current-process compatibility is collected separately.
        issues = intrinsic_issues
        affected_segment_paths: list[str] = []
        manifest: Optional[dict[str, Any]] = None
        manifest_status = "missing"
        manifest_ids: list[str] = []
        immutable_comparison: dict[str, Any] = {}

        try:
            manifest = self._read_json(manifest_path, "run_manifest")
            manifest_status = "readable"
            self._validate_run_manifest_identity(
                manifest,
                expected_run_id=run_id,
                expected_fingerprint=None,
            )
            manifest_ids = list(manifest["recorder_segment_ids"])
            if (
                context is not None
                and manifest.get("task_set_fingerprint")
                != context.task_set_fingerprint
            ):
                append_compatibility_issues.append(
                    "run_manifest_identity_mismatch"
                )
                immutable_comparison["manifest_error"] = (
                    "现有run manifest身份冲突：task_set_fingerprint不一致"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            manifest_status = "invalid" if os.path.lexists(manifest_path) else "missing"
            manifest_error = str(exc)
            issues.append(
                "run_manifest_identity_mismatch"
                if "现有run manifest身份冲突" in manifest_error
                else "run_manifest_schema_invalid"
            )
            immutable_comparison["manifest_error"] = manifest_error
            raw_ids = None if manifest is None else manifest.get("recorder_segment_ids")
            if (
                isinstance(raw_ids, list)
                and all(isinstance(item, str) for item in raw_ids)
                and len(raw_ids) == len(set(raw_ids))
            ):
                try:
                    manifest_ids = [
                        _safe_component(item, "recorder_segment_ids")
                        for item in raw_ids
                    ]
                except ValueError:
                    manifest_ids = []

        current_facts, _warnings = self._current_run_immutable_facts()
        if manifest is not None:
            contract_fields = (
                "schema_name",
                "schema_version",
                "recorder_schema_sha256",
                "interface_schema_name",
                "interface_schema_version",
                "interface_schema_sha256",
                "run_identity_scope",
            )
            contract_mismatches = {
                field: {"manifest": manifest.get(field), "current": current_facts[field]}
                for field in contract_fields
                if manifest.get(field) != current_facts[field]
            }
            immutable_comparison["contract"] = contract_mismatches
            if contract_mismatches:
                issues.append("run_manifest_contract_hash_mismatch")

            if compare_current_runtime_facts:
                gate_fields = ("observe_only", "official_publish_enabled")
                gate_mismatches = {
                    field: {
                        "manifest": manifest.get(field),
                        "current": current_facts[field],
                    }
                    for field in gate_fields
                    if manifest.get(field) != current_facts[field]
                }
                immutable_comparison["runtime_gate"] = gate_mismatches
                if gate_mismatches:
                    append_compatibility_issues.append(
                        "run_manifest_runtime_gate_mismatch"
                    )

                config_mismatch = (
                    manifest.get("config_sha256") != current_facts["config_sha256"]
                )
                immutable_comparison["config_sha256"] = {
                    "matches": not config_mismatch,
                    "manifest": manifest.get("config_sha256"),
                    "current": current_facts["config_sha256"],
                }
                if config_mismatch:
                    append_compatibility_issues.append(
                        "run_manifest_config_mismatch"
                    )

                runtime_fields = (
                    "ros_domain_id",
                    "rmw_implementation",
                    "official_server_image_id",
                    "official_client_image_id",
                )
                runtime_mismatches = {
                    field: {
                        "manifest": manifest.get(field),
                        "current": current_facts[field],
                    }
                    for field in runtime_fields
                    if manifest.get(field) != current_facts[field]
                }
                immutable_comparison["runtime_facts"] = runtime_mismatches
                if runtime_mismatches:
                    append_compatibility_issues.append(
                        "run_manifest_runtime_fact_mismatch"
                    )

                provenance_fields = (
                    "project_commit",
                    "project_branch",
                    "dirty_worktree",
                )
                provenance_mismatches = {
                    field: {
                        "manifest": manifest.get(field),
                        "current": current_facts[field],
                    }
                    for field in provenance_fields
                    if manifest.get(field) != current_facts[field]
                }
                immutable_comparison["provenance"] = provenance_mismatches
                if provenance_mismatches:
                    append_compatibility_issues.append(
                        "run_manifest_provenance_mismatch"
                    )
            else:
                immutable_comparison["current_runtime_facts_compared"] = False

            terminal_fields = (
                manifest.get("end_ros_ns"),
                manifest.get("end_wall_utc"),
                manifest.get("clean_shutdown"),
                manifest.get("shutdown_reason"),
                manifest.get("recovery_required"),
            )
            unfinished = all(value is None for value in terminal_fields)
            finished = (
                type(terminal_fields[0]) is int
                and isinstance(terminal_fields[1], str)
                and bool(terminal_fields[1])
                and type(terminal_fields[2]) is bool
                and isinstance(terminal_fields[3], str)
                and bool(terminal_fields[3])
                and type(terminal_fields[4]) is bool
            )
            immutable_comparison["terminal_state"] = {
                "unfinished": unfinished,
                "finished": finished,
            }
            if not (unfinished or finished):
                issues.append("run_manifest_terminal_state_invalid")
            elif require_appendable and not unfinished:
                append_compatibility_issues.append(
                    "run_manifest_terminal_state_invalid"
                )

        disk_ids: list[str] = []
        if segments_dir.is_symlink() or not segments_dir.is_dir():
            issues.append("run_segments_directory_invalid")
        else:
            for entry in sorted(segments_dir.iterdir(), key=lambda item: item.name):
                if entry.is_symlink():
                    issues.append("run_manifest_unlisted_segment")
                    affected_segment_paths.append(str(entry))
                    continue
                if entry.is_dir():
                    disk_ids.append(entry.name)

        manifest_set = set(manifest_ids)
        disk_set = set(disk_ids)
        missing_ids = sorted(manifest_set - disk_set)
        unlisted_ids = sorted(disk_set - manifest_set)
        if missing_ids:
            issues.append("run_manifest_segment_missing")
        if unlisted_ids:
            issues.append("run_manifest_unlisted_segment")

        sequence_map: dict[str, Any] = {}
        segment_manifests: dict[str, dict[str, Any]] = {}
        segment_reports: dict[str, dict[str, Any]] = {}
        for segment_id in disk_ids:
            segment_dir = segments_dir / segment_id
            local_issues: list[str] = []
            active_path = segment_dir / "ACTIVE"
            complete_path = segment_dir / "COMPLETE"
            active_exists = os.path.lexists(active_path)
            complete_exists = os.path.lexists(complete_path)
            if active_exists and not complete_exists:
                local_issues.append("active_without_complete")
            if active_exists and complete_exists:
                local_issues.append("active_and_complete_both_present")
            if not complete_exists:
                local_issues.append("complete_marker_missing")
            self._inspect_marker(
                active_path, "ACTIVE", segment_id, run_id, local_issues
            )
            complete_marker = self._inspect_marker(
                complete_path, "COMPLETE", segment_id, run_id, local_issues
            )
            complete_valid = complete_marker is not None and not any(
                issue.startswith("complete_marker_") for issue in local_issues
            )
            data, _status, diagnostics = self._inspect_segment_manifest(
                segment_dir, run_id, complete_valid, local_issues
            )
            if data is not None:
                segment_manifests[segment_id] = data
                sequence_map[segment_id] = data.get("segment_sequence")
            else:
                sequence_map[segment_id] = None
            integrity_issues = [
                issue
                for issue in local_issues
                if issue != "segment_unclean_shutdown"
            ]
            if diagnostics["segment_manifest_identity_mismatch"] or integrity_issues:
                issues.extend(integrity_issues)
                affected_segment_paths.append(str(segment_dir))
            # Keep all pre-existing per-Segment recovery checks (local JSONL,
            # rosbag clues, identity diagnostics) inside the single aggregated
            # Run report instead of silently dropping them.
            segment_report = self._inspect_segment(segment_dir, run_id)
            if segment_report is not None:
                segment_reports[segment_id] = segment_report
                issues.extend(segment_report["issue_types"])
                affected_segment_paths.append(str(segment_dir))

        valid_sequences = [
            value for value in sequence_map.values() if type(value) is int and value >= 0
        ]
        if len(valid_sequences) != len(set(valid_sequences)):
            issues.append("run_manifest_segment_sequence_duplicate")
        if sorted(valid_sequences) != list(range(len(disk_ids))):
            issues.append("run_manifest_segment_sequence_gap")
        if any(
            sequence_map.get(segment_id) != index
            for index, segment_id in enumerate(manifest_ids)
            if segment_id in sequence_map
        ):
            issues.append("run_manifest_segment_order_mismatch")

        events_exists = os.path.lexists(events_path)
        events_raw: Optional[bytes] = None
        events_result: Optional[dict[str, Any]] = None
        if manifest_ids and not events_exists:
            issues.append("run_events_missing")
        elif events_exists:
            try:
                metadata = os.lstat(events_path)
                if stat.S_ISLNK(metadata.st_mode):
                    issues.append("run_events_symlink_rejected")
                    issues.append("run_events_path_escape")
                elif not stat.S_ISREG(metadata.st_mode):
                    issues.append("run_events_not_regular")
                else:
                    events_raw = _read_regular_bytes_no_follow(events_path, root)
                    events_result = self._inspect_jsonl_bytes(
                        events_path.name, events_raw
                    )
                    events_result.update(
                        {
                            "shared_append_only": True,
                            "scope": "run_shared_append_only_events",
                        }
                    )
                    event_issue_mapping = {
                        "jsonl_trailing_incomplete": "run_events_trailing_incomplete",
                        "jsonl_trailing_json_invalid": "run_events_trailing_json_invalid",
                        "jsonl_middle_corruption": "run_events_middle_corruption",
                    }
                    issues.extend(
                        event_issue_mapping[item]
                        for item in events_result["issue_types"]
                    )
                    # Preserve the generic JSONL diagnostics consumed by existing
                    # recovery tooling while adding the Run-specific taxonomy.
                    issues.extend(events_result["issue_types"])
            except (OSError, RuntimeError, ValueError):
                issues.append("run_events_read_rejected")

        prefix_results: list[dict[str, Any]] = []
        pending_prefixes: list[tuple[str, int, str]] = []
        for segment_id in manifest_ids:
            data = segment_manifests.get(segment_id)
            if data is None:
                continue
            artifacts = data.get("jsonl_artifacts")
            if not isinstance(artifacts, list):
                issues.append("run_events_integrity_record_invalid")
                continue
            candidates = [
                item
                for item in artifacts
                if isinstance(item, dict)
                and (
                    item.get("path") == "../../events.jsonl"
                    or item.get("shared_append_only") is True
                )
            ]
            if not candidates:
                issues.append("run_events_integrity_record_missing")
                prefix_results.append({"segment_id": segment_id, "status": "missing"})
                continue
            if len(candidates) != 1:
                issues.append("run_events_integrity_record_invalid")
                prefix_results.append({"segment_id": segment_id, "status": "duplicate"})
                continue
            record = candidates[0]
            offset = record.get("byte_end_offset")
            prefix_hash = record.get("sha256_prefix")
            valid_record = (
                record.get("path") == "../../events.jsonl"
                and record.get("shared_append_only") is True
                and type(offset) is int
                and offset >= 0
                and isinstance(prefix_hash, str)
                and re.fullmatch(r"[0-9a-f]{64}", prefix_hash) is not None
            )
            if not valid_record:
                issues.append("run_events_integrity_record_invalid")
                prefix_results.append({"segment_id": segment_id, "status": "invalid"})
                continue
            assert isinstance(offset, int) and isinstance(prefix_hash, str)
            if events_raw is not None and offset > len(events_raw):
                issues.append("run_events_prefix_truncated")
                prefix_results.append(
                    {"segment_id": segment_id, "status": "truncated", "offset": offset}
                )
                continue
            pending_prefixes.append((segment_id, offset, prefix_hash))

        prefix_hashes: dict[int, str] = {}
        if events_raw is not None:
            digest = hashlib.sha256()
            cursor = 0
            for offset in sorted({item[1] for item in pending_prefixes}):
                digest.update(events_raw[cursor:offset])
                cursor = offset
                prefix_hashes[offset] = digest.copy().hexdigest()
        for segment_id, offset, expected_hash in pending_prefixes:
            actual_hash = prefix_hashes.get(offset)
            status = "valid" if actual_hash == expected_hash else "hash_mismatch"
            if status != "valid":
                issues.append("run_events_prefix_hash_mismatch")
            prefix_results.append(
                {
                    "segment_id": segment_id,
                    "status": status,
                    "offset": offset,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                }
            )

        unique_affected = sorted(set(affected_segment_paths))
        unique_intrinsic_issues = sorted(set(intrinsic_issues))
        unique_append_issues = sorted(set(append_compatibility_issues))
        unique_issues = sorted(
            set(unique_intrinsic_issues).union(unique_append_issues)
        )
        sole_segment_report = (
            segment_reports.get(Path(unique_affected[0]).name)
            if len(unique_affected) == 1
            else None
        )
        event_only = bool(unique_issues) and all(
            issue.startswith(("run_events_", "jsonl_")) for issue in unique_issues
        )
        source_hashes: dict[str, str] = {}
        if not {
            "run_events_symlink_rejected",
            "run_events_path_escape",
        }.intersection(unique_issues):
            try:
                manifest_metadata = os.lstat(manifest_path)
                if stat.S_ISREG(manifest_metadata.st_mode):
                    manifest_raw = _read_regular_bytes_no_follow(manifest_path, root)
                    source_hashes["manifest.json"] = hashlib.sha256(
                        manifest_raw
                    ).hexdigest()
            except OSError:
                pass
            if events_raw is not None:
                source_hashes["events.jsonl"] = hashlib.sha256(
                    events_raw
                ).hexdigest()
            for segment_id, segment_report in sorted(segment_reports.items()):
                for name, digest in sorted(
                    segment_report["source_artifact_hashes"].items()
                ):
                    source_hashes[f"segments/{segment_id}/{name}"] = digest
        return {
            **({} if sole_segment_report is None else sole_segment_report),
            "source_scope": "run_shared_events" if event_only else "run_integrity",
            "source_parent_run_id": run_id,
            "source_segment_path": (
                unique_affected[0] if len(unique_affected) == 1 else None
            ),
            "source_segment_identity": (
                Path(unique_affected[0]).name if len(unique_affected) == 1 else None
            ),
            "run_id": run_id,
            "manifest_status": manifest_status,
            "immutable_contract_runtime_comparison": immutable_comparison,
            "manifest_segment_ids": manifest_ids,
            "disk_segment_ids": disk_ids,
            "missing_segment_ids": missing_ids,
            "unlisted_segment_ids": unlisted_ids,
            "segment_sequence_map": sequence_map,
            "events_exists": events_exists,
            "events_strict_jsonl": events_result,
            "events_file_sha256": (
                None if events_raw is None else hashlib.sha256(events_raw).hexdigest()
            ),
            "segment_prefix_results": prefix_results,
            "intrinsic_integrity_issues": unique_intrinsic_issues,
            "append_compatibility_issues": unique_append_issues,
            "issue_types": unique_issues,
            "active_marker_present": None,
            "complete_marker_present": None,
            "segment_manifest_status": (
                "aggregated"
                if sole_segment_report is None
                else sole_segment_report["segment_manifest_status"]
            ),
            "jsonl": (
                ([] if events_result is None else [events_result])
                + [
                    item
                    for report in segment_reports.values()
                    for item in report["jsonl"]
                ]
            ),
            "bag_clues": sorted(
                {
                    clue
                    for report in segment_reports.values()
                    for clue in report["bag_clues"]
                }
            ),
            "source_artifact_hashes": source_hashes,
        }

    def _new_run_manifest(
        self, context: CompetitionContext, now_ros_ns: int
    ) -> dict[str, Any]:
        facts, warnings = self._current_run_immutable_facts()
        return {
            **facts,
            "run_id": context.run_id,
            "task_set_fingerprint": context.task_set_fingerprint,
            "start_ros_ns": now_ros_ns,
            "end_ros_ns": None,
            "start_wall_utc": self._wall_now(),
            "end_wall_utc": None,
            "recorder_segment_ids": [],
            "clean_shutdown": None,
            "shutdown_reason": None,
            "recovery_required": None,
            "provenance_warnings": warnings,
        }

    def _current_run_immutable_facts(
        self,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        warnings: list[dict[str, Any]] = []
        project_commit = self._injected_envelope("project_commit")
        project_branch = self._injected_envelope("project_branch")
        dirty = self._dirty_envelope(warnings)
        config_hash = self._config_hash_envelope(warnings)
        ros_domain = self._ros_domain_envelope(warnings)
        rmw = self._environment.get("RMW_IMPLEMENTATION", "").strip()
        rmw_envelope = (
            _available(rmw, "auto_detected")
            if rmw
            else _unknown("auto_detected", "RMW_IMPLEMENTATION_unset")
        )
        facts = {
            "schema_name": "team_sorting.recorder.run_manifest",
            "schema_version": 1,
            "recorder_schema_sha256": recorder_contract_sha256(),
            "interface_schema_name": "team_sorting.interface",
            "interface_schema_version": 1,
            "interface_schema_sha256": _sha256_file(interface_contract_path()),
            "run_identity_scope": "team_local",
            "project_commit": project_commit,
            "project_branch": project_branch,
            "dirty_worktree": dirty,
            "config_sha256": config_hash,
            "official_server_image_id": self._injected_envelope(
                "official_server_image_id"
            ),
            "official_client_image_id": self._injected_envelope(
                "official_client_image_id"
            ),
            "ros_domain_id": ros_domain,
            "rmw_implementation": rmw_envelope,
            "observe_only": self.config.observe_only,
            "official_publish_enabled": self.config.official_publish_enabled,
        }
        return facts, warnings

    def _finalize_run_manifest(self, run_id: str, now_ros_ns: int, reason: str) -> None:
        run_id = _safe_component(run_id, "run_id")
        run_dir = self.config.root_dir.resolve() / "runs" / run_id
        self._validate_run_events_path(run_dir)
        path = run_dir / "manifest.json"
        manifest = self._read_json(path, "run_manifest")
        self._validate_run_manifest_identity(manifest, expected_run_id=run_id)
        if manifest["end_ros_ns"] is not None:
            return
        manifest["end_ros_ns"] = now_ros_ns
        manifest["end_wall_utc"] = self._wall_now()
        manifest["clean_shutdown"] = True
        manifest["shutdown_reason"] = reason
        manifest["recovery_required"] = False
        self._validate_run_manifest_identity(manifest, expected_run_id=run_id)
        atomic_write_json(
            path,
            manifest,
            artifact_type="run_manifest",
            run_id=run_id,
        )

    def _write_event(
        self,
        event_type: str,
        receive_ros_ns: int,
        receive_monotonic_ns: int,
        *,
        payload: Mapping[str, Any],
        source: Mapping[str, Any],
        source_event_ids: Sequence[str],
        validity: str = "valid",
        invalid_reasons: Sequence[str] = (),
        context_override: Optional[CompetitionContext] = None,
    ) -> str:
        active = self._require_active_for_write(allow_closing=True)
        event_id = f"{active.segment_id}:{active.event_sequence:020d}"
        context = (
            context_override
            if context_override is not None
            and active.kind == "run_bound"
            and context_override.run_id == active.parent_run_id
            else self._last_valid_context
            if active.kind == "run_bound"
            and self._last_valid_context is not None
            and self._last_valid_context.run_id == active.parent_run_id
            else None
        )
        task_id = None if context is None else context.current_task_id
        settled = None if context is None else context.current_attempt_count
        local_key = (
            None
            if context is None or task_id is None
            else [context.run_id, task_id, settled]
        )
        event = {
            "schema_name": "team_sorting.recorder.event",
            "schema_version": 1,
            "event_id": event_id,
            "event_type": event_type,
            "event_timestamp_ns": receive_ros_ns,
            "receive_timestamp_ns": receive_ros_ns,
            "receive_monotonic_ns": receive_monotonic_ns,
            "monotonic_scope": "process_local",
            "run_id": (
                active.parent_run_id if active.kind == "run_bound" else None
            ),
            "recorder_segment_id": active.segment_id,
            "task_id": task_id,
            "settled_attempt_count": settled,
            "local_attempt_key": local_key,
            "payload": dict(payload),
            "source": dict(source),
            "validity": validity,
            "invalid_reasons": list(invalid_reasons),
            "source_event_ids": list(source_event_ids),
        }
        self._validate_contract_record("event_schema", event)
        append_jsonl(
            active.event_path,
            event,
            artifact_type="events_jsonl",
            segment_id=active.segment_id,
            run_id=active.parent_run_id,
            allowed_root=self.config.root_dir,
        )
        active.event_sequence += 1
        self._touch_ros(receive_ros_ns)
        return event_id

    def _mark_failed_nonwritable(self) -> None:
        if self._active is not None:
            self._active.accepting_writes = False
        self.state = RecorderRuntimeState.FAILED

    def _converge_context_failure(self, now_ros_ns: int, reason: str) -> None:
        """Best-effort seal after a required context identity commit fails."""

        self._mark_failed_nonwritable()
        if self._active is None:
            return
        try:
            self.state = RecorderRuntimeState.CLOSING
            self._close_active(now_ros_ns, reason, clean=False)
        except Exception:
            pass
        finally:
            self.state = RecorderRuntimeState.FAILED

    def _safe_event(
        self,
        event_type: str,
        receive_ros_ns: int,
        receive_monotonic_ns: int,
        *,
        payload: Mapping[str, Any],
        source: Mapping[str, Any],
        validity: str = "valid",
        invalid_reasons: Sequence[str] = (),
        source_event_ids: Sequence[str] = (),
    ) -> Optional[str]:
        try:
            return self._write_event(
                event_type,
                receive_ros_ns,
                receive_monotonic_ns,
                payload=payload,
                source=source,
                source_event_ids=source_event_ids,
                validity=validity,
                invalid_reasons=invalid_reasons,
            )
        except RuntimeError:
            if self._active is not None:
                counters = self._active.data["warning_counters"]
                counters["event_write_failure"] = counters.get("event_write_failure", 0) + 1
            return None

    def _record_pairing_issues(
        self, issues: Sequence[str], receive_ros_ns: int, receive_monotonic_ns: int
    ) -> None:
        if self._active is None:
            return
        for issue in issues:
            counters = self._active.data["pairing_counters"]
            counters[issue] = counters.get(issue, 0) + 1
            self._safe_event(
                "pairing_issue",
                receive_ros_ns,
                receive_monotonic_ns,
                payload={
                    "issue_type": issue,
                    "artifact": "action_pairing_issues.jsonl",
                },
                source={"kind": "ActionDispatchPairer", "derived": True},
            )

    def _record_invalid_context(
        self,
        raw_payload: object,
        reason: str,
        receive_ros_ns: int,
        receive_monotonic_ns: int,
    ) -> None:
        active = self._require_active_for_write()
        active.data["context_invalid_count"] += 1
        raw = raw_payload if isinstance(raw_payload, str) else repr(raw_payload)
        preview = raw[:4096]
        self._safe_event(
            "competition_context_updated",
            receive_ros_ns,
            receive_monotonic_ns,
            payload={
                "raw_payload_preview": preview,
                "raw_payload_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "preview_truncated": len(preview) != len(raw),
            },
            source={"topic": "/team/competition_context", "derived": False},
            validity="invalid",
            invalid_reasons=(reason,),
        )

    def _emit_transitions(
        self,
        previous: Optional[CompetitionContext],
        current: CompetitionContext,
        source_event_id: str,
        now_ros_ns: int,
        monotonic_ns: int,
    ) -> None:
        if previous is None or previous.run_id != current.run_id:
            return
        if previous.current_task_id != current.current_task_id:
            self._safe_event(
                "task_transition",
                now_ros_ns,
                monotonic_ns,
                payload={
                    "previous_task_id": previous.current_task_id,
                    "current_task_id": current.current_task_id,
                    "physical_reset_claimed": False,
                },
                source={"kind": "valid_context_comparison", "derived": True},
                source_event_ids=(source_event_id,),
            )
        if previous.current_attempt_count != current.current_attempt_count:
            self._safe_event(
                "attempt_transition",
                now_ros_ns,
                monotonic_ns,
                payload={
                    "previous_settled_attempt_count": previous.current_attempt_count,
                    "current_settled_attempt_count": current.current_attempt_count,
                    "official_attempt_start_claimed": False,
                },
                source={"kind": "valid_context_comparison", "derived": True},
                source_event_ids=(source_event_id,),
            )

    def _observe_context(self, context: CompetitionContext) -> None:
        if self._active is None:
            return
        if context.current_task_id is not None:
            tasks = self._active.data["observed_task_ids"]
            if context.current_task_id not in tasks:
                tasks.append(context.current_task_id)
        attempts = self._active.data["observed_settled_attempt_counts"]
        if context.current_attempt_count not in attempts:
            attempts.append(context.current_attempt_count)

    def _touch_ros(self, timestamp_ns: int) -> None:
        self._require_ns(timestamp_ns, "timestamp_ns")
        if self._active is None:
            return
        first = self._active.data["first_ros_timestamp_ns"]
        last = self._active.data["last_ros_timestamp_ns"]
        self._active.data["first_ros_timestamp_ns"] = (
            timestamp_ns if first is None else min(first, timestamp_ns)
        )
        self._active.data["last_ros_timestamp_ns"] = (
            timestamp_ns if last is None else max(last, timestamp_ns)
        )

    def _artifact_inventory(self, active: _ActiveSegment) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        candidates = [active.directory / name for name in _JSONL_NAMES]
        if active.event_path.parent == active.directory:
            candidates.append(active.event_path)
        for path in candidates:
            if not path.is_symlink() and path.is_file():
                artifacts.append(
                    {
                        "path": str(path.relative_to(active.directory)),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
        if active.event_path.parent != active.directory and os.path.lexists(active.event_path):
            self._validate_run_events_path(active.event_path.parent)
            event_bytes = _read_regular_bytes_no_follow(
                active.event_path, self.config.root_dir.resolve()
            )
            byte_end_offset = len(event_bytes)
            artifacts.append(
                {
                    "path": str(Path("../../events.jsonl")),
                    "shared_append_only": True,
                    "byte_end_offset": byte_end_offset,
                    "sha256_prefix": hashlib.sha256(event_bytes).hexdigest(),
                }
            )
        return artifacts

    def _fsync_event_stream(self, active: _ActiveSegment) -> None:
        if not os.path.lexists(active.event_path):
            return
        if active.event_path.parent != active.directory:
            self._validate_run_events_path(active.event_path.parent)
        parent_descriptor: Optional[int] = None
        descriptor: Optional[int] = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            parent_descriptor = os.open(active.event_path.parent, flags)
            descriptor = os.open(
                active.event_path.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError(
                    f"event stream必须是普通文件：{active.event_path}"
                )
            os.fsync(descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if parent_descriptor is not None:
                os.close(parent_descriptor)

    def _scan_recovery(self, root: Path) -> list[dict[str, Any]]:
        candidates: list[Path] = []
        run_ids: list[str] = []
        bootstrap = root / "bootstrap"
        if bootstrap.is_dir() and not bootstrap.is_symlink():
            candidates.extend(
                path
                for path in bootstrap.iterdir()
                if not path.is_symlink() and path.is_dir()
            )
        runs = root / "runs"
        if runs.is_dir() and not runs.is_symlink():
            for run_dir in runs.iterdir():
                if not run_dir.is_symlink() and run_dir.is_dir():
                    _resolved_under(root, run_dir, "recovery run source")
                    _safe_component(run_dir.name, "recovery path run_id")
                    run_ids.append(run_dir.name)
        reports: list[dict[str, Any]] = []
        for segment_dir in candidates:
            _resolved_under(root, segment_dir, "recovery source")
            report = self._inspect_segment(segment_dir, None)
            if report is None:
                continue
            report["intrinsic_integrity_issues"] = report["issue_types"]
            report["append_compatibility_issues"] = []
            reports.append(self._persist_recovery_finding(root, report))
        for run_id in sorted(run_ids):
            report = self._inspect_existing_run_integrity(
                run_id,
                require_appendable=False,
                compare_current_runtime_facts=False,
            )
            if not report["intrinsic_integrity_issues"]:
                continue
            reports.append(self._persist_recovery_finding(root, report))
            self._blocked_run_ids.add(run_id)
        return reports

    def _recovery_finding_material(
        self, root: Path, report: Mapping[str, Any]
    ) -> dict[str, Any]:
        source_path = report.get("source_segment_path")
        source_path_identity: Optional[str] = None
        if isinstance(source_path, str):
            try:
                source_path_identity = str(Path(source_path).resolve().relative_to(root))
            except ValueError as exc:
                raise RuntimeError(
                    f"Recovery来源路径逃逸dataset_root：{source_path}"
                ) from exc
        jsonl = report.get("jsonl")
        stable_jsonl = sorted(
            (item for item in jsonl if isinstance(item, dict)),
            key=lambda item: str(item.get("path")),
        ) if isinstance(jsonl, list) else []
        prefixes = report.get("segment_prefix_results")
        stable_prefixes = sorted(
            (item for item in prefixes if isinstance(item, dict)),
            key=lambda item: (
                str(item.get("segment_id")),
                int(item.get("offset", -1))
                if type(item.get("offset")) is int
                else -1,
            ),
        ) if isinstance(prefixes, list) else []
        intrinsic = report.get("intrinsic_integrity_issues", report.get("issue_types", []))
        return {
            "recovery_schema_name": "team_sorting.recorder.recovery_report",
            "recovery_schema_version": 1,
            "source_scope": report.get("source_scope", "segment_integrity"),
            "run_id": report.get("run_id", report.get("source_parent_run_id")),
            "source_segment_identity": report.get("source_segment_identity"),
            "source_segment_path_identity": source_path_identity,
            "issue_types": sorted(set(intrinsic)) if isinstance(intrinsic, list) else [],
            "source_artifact_hashes": report.get("source_artifact_hashes", {}),
            "manifest_status": report.get("manifest_status"),
            "segment_manifest_status": report.get("segment_manifest_status"),
            "active_marker_present": report.get("active_marker_present"),
            "complete_marker_present": report.get("complete_marker_present"),
            "events_file_sha256": report.get("events_file_sha256"),
            "segment_prefix_results": stable_prefixes,
            "jsonl": stable_jsonl,
            "missing_segment_ids": report.get("missing_segment_ids", []),
            "unlisted_segment_ids": report.get("unlisted_segment_ids", []),
            "segment_sequence_map": report.get("segment_sequence_map", {}),
        }

    def _recovery_finding_fingerprint(
        self, root: Path, report: Mapping[str, Any]
    ) -> str:
        encoded = json.dumps(
            self._recovery_finding_material(root, report),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _persist_recovery_finding(
        self, root: Path, report: Mapping[str, Any]
    ) -> dict[str, Any]:
        fingerprint = self._recovery_finding_fingerprint(root, report)
        report_id = f"recovery_{fingerprint}"
        recovery_dir = _resolved_under(
            root, root / "recovery", "recovery report directory"
        )
        output = recovery_dir / f"{report_id}.json"
        if os.path.lexists(output):
            metadata = os.lstat(output)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(
                    f"既有Recovery报告必须是非symlink普通文件：{output}"
                )
            try:
                existing = self._read_json(output, "recovery_report")
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"既有Recovery报告无法严格读取：{output}"
                ) from exc
            if (
                existing.get("schema_name")
                != "team_sorting.recorder.recovery_report"
                or existing.get("schema_version") != 1
                or existing.get("report_id") != report_id
                or existing.get("finding_fingerprint") != fingerprint
                or self._recovery_finding_fingerprint(root, existing) != fingerprint
            ):
                raise RuntimeError(
                    f"既有Recovery报告与确定性finding fingerprint不一致：{output}"
                )
            return {**existing, "_newly_created": False}

        first_detected_at = self._wall_now()
        persisted = {
            **report,
            "report_id": report_id,
            "finding_fingerprint": fingerprint,
            "schema_name": "team_sorting.recorder.recovery_report",
            "schema_version": 1,
            "detected_at": first_detected_at,
            "first_detected_at": first_detected_at,
            "recovery_action": "none",
            "requires_manual_or_offline_recovery": True,
        }
        exclusive_write_json(
            output, persisted, artifact_type="recovery_report"
        )
        return {**persisted, "_newly_created": True}

    def _inspect_segment(
        self, segment_dir: Path, path_parent_run_id: Optional[str]
    ) -> Optional[dict[str, Any]]:
        active_path = segment_dir / "ACTIVE"
        complete_path = segment_dir / "COMPLETE"
        active_exists = os.path.lexists(active_path)
        complete_exists = os.path.lexists(complete_path)
        issue_types: list[str] = []
        if active_path.is_symlink() or complete_path.is_symlink():
            issue_types.append("marker_symlink_rejected")
        if active_exists and not complete_exists:
            issue_types.append("active_without_complete")
        if active_exists and complete_exists:
            issue_types.append("active_and_complete_both_present")
        active_marker = self._inspect_marker(
            active_path,
            "ACTIVE",
            segment_dir.name,
            path_parent_run_id,
            issue_types,
        )
        complete_marker = self._inspect_marker(
            complete_path,
            "COMPLETE",
            segment_dir.name,
            path_parent_run_id,
            issue_types,
        )
        marker_parent_run_id = (
            None if active_marker is None else active_marker.get("parent_run_id")
        )
        complete_marker_parent_run_id = (
            None if complete_marker is None else complete_marker.get("parent_run_id")
        )
        if active_marker is not None and complete_marker is not None:
            active_identity = (
                active_marker.get("recorder_segment_id"),
                active_marker.get("parent_run_id"),
            )
            complete_identity = (
                complete_marker.get("recorder_segment_id"),
                complete_marker.get("parent_run_id"),
            )
            if active_identity != complete_identity:
                issue_types.append("marker_identity_mismatch")
        complete_marker_valid = complete_marker is not None and not any(
            issue.startswith("complete_marker_") for issue in issue_types
        )
        segment_data, segment_status, segment_diagnostics = (
            self._inspect_segment_manifest(
                segment_dir,
                path_parent_run_id,
                complete_marker_valid,
                issue_types,
            )
        )
        jsonl_reports: list[dict[str, Any]] = []
        for path in sorted(segment_dir.glob("*.jsonl")):
            if path.is_symlink() or not path.is_file():
                issue_types.append("jsonl_symlink_rejected")
                continue
            _resolved_under(segment_dir, path, "segment jsonl source")
            item = self._inspect_jsonl(path)
            jsonl_reports.append(item)
            issue_types.extend(item["issue_types"])
        bag_clues: list[str] = []
        bag_path = None if segment_data is None else segment_data.get("bag_path")
        if isinstance(bag_path, str):
            bag_dir = segment_dir / bag_path
            if bag_dir.is_symlink():
                bag_clues.append("bag_directory_symlink_rejected")
            elif not bag_dir.is_dir():
                bag_clues.append("bag_directory_missing")
            elif not (bag_dir / "metadata.yaml").is_file():
                bag_clues.append("bag_metadata_missing")
        issue_types.extend(bag_clues)
        manifest_parent_run_id = (
            None if segment_data is None else segment_data.get("parent_run_id")
        )
        identity_values = [
            value
            for value in (
                marker_parent_run_id,
                complete_marker_parent_run_id,
                manifest_parent_run_id,
            )
            if value is not None
        ]
        identity_mismatch = any(
            value != path_parent_run_id for value in identity_values
        )
        identity_mismatch = (
            identity_mismatch
            or segment_diagnostics["segment_manifest_identity_mismatch"]
        )
        marker_identity_issue = any(
            issue in issue_types
            for issue in (
                "active_marker_identity_mismatch",
                "complete_marker_identity_mismatch",
            )
        )
        if marker_identity_issue:
            issue_types.append("marker_identity_mismatch")
            identity_mismatch = True
        if identity_mismatch:
            issue_types.append("identity_mismatch")
        if not issue_types:
            return None
        source_hashes = {
            path.name: _sha256_file(path)
            for path in sorted(segment_dir.iterdir())
            if not path.is_symlink() and path.is_file()
        }
        return {
            "source_segment_path": str(segment_dir),
            "source_segment_identity": segment_dir.name,
            "source_parent_run_id": path_parent_run_id,
            "parent_run_identity_source": "canonical_path",
            "path_parent_run_id": path_parent_run_id,
            "marker_parent_run_id": marker_parent_run_id,
            "complete_marker_parent_run_id": complete_marker_parent_run_id,
            "manifest_parent_run_id": manifest_parent_run_id,
            "identity_mismatch": identity_mismatch,
            "parent_run_identity_mismatch": identity_mismatch,
            "issue_types": sorted(set(issue_types)),
            "active_marker_present": active_exists,
            "complete_marker_present": complete_exists,
            "segment_manifest_status": segment_status,
            "jsonl": jsonl_reports,
            "bag_clues": bag_clues,
            "source_artifact_hashes": source_hashes,
            **segment_diagnostics,
        }

    def _inspect_segment_manifest(
        self,
        segment_dir: Path,
        expected_parent_run_id: Optional[str],
        complete_marker_valid: bool,
        issue_types: list[str],
    ) -> tuple[Optional[dict[str, Any]], str, dict[str, Any]]:
        path = segment_dir / "segment.json"
        expected_kind = (
            "bootstrap" if expected_parent_run_id is None else "run_bound"
        )
        diagnostics: dict[str, Any] = {
            "path_segment_id": segment_dir.name,
            "manifest_segment_id": None,
            "path_segment_kind": expected_kind,
            "manifest_segment_kind": None,
            "expected_parent_run_id": expected_parent_run_id,
            "manifest_parent_run_id": None,
            "segment_manifest_identity_mismatch": False,
        }
        if path.is_symlink() or not path.is_file():
            issue_types.extend(
                ["segment_manifest_missing", "segment_manifest_schema_invalid"]
            )
            return None, "missing", diagnostics
        try:
            loaded = self._read_json(path, "segment_manifest")
        except (OSError, RuntimeError, ValueError):
            issue_types.extend(
                ["segment_manifest_invalid", "segment_manifest_schema_invalid"]
            )
            return None, "invalid", diagnostics

        diagnostics["manifest_segment_id"] = loaded.get("recorder_segment_id")
        diagnostics["manifest_segment_kind"] = loaded.get("segment_kind")
        diagnostics["manifest_parent_run_id"] = loaded.get("parent_run_id")
        try:
            self._validate_contract_record("recorder_segment_schema", loaded)
        except RuntimeError:
            issue_types.append("segment_manifest_schema_invalid")
        if (
            loaded.get("schema_name") != "team_sorting.recorder.segment"
            or type(loaded.get("schema_version")) is not int
            or loaded.get("schema_version") != 1
        ):
            issue_types.append("segment_manifest_schema_invalid")

        raw_segment_id = loaded.get("recorder_segment_id")
        try:
            safe_segment_id = _safe_component(
                raw_segment_id, "segment manifest recorder_segment_id"
            )
        except ValueError:
            safe_segment_id = None
        if safe_segment_id != segment_dir.name:
            issue_types.append("segment_manifest_identity_mismatch")

        if loaded.get("segment_kind") != expected_kind:
            issue_types.append("segment_manifest_kind_mismatch")
        if loaded.get("parent_run_id") != expected_parent_run_id:
            issue_types.append("segment_manifest_parent_mismatch")
        sequence = loaded.get("segment_sequence")
        if type(sequence) is not int or sequence < 0:
            issue_types.append("segment_manifest_sequence_invalid")

        marker_state = loaded.get("marker_state")
        clean_shutdown = loaded.get("clean_shutdown")
        shutdown_reason = loaded.get("shutdown_reason")
        terminal_fact_inconsistent = (
            type(clean_shutdown) is bool
            and (
                marker_state != "complete"
                or not isinstance(shutdown_reason, str)
                or not shutdown_reason
            )
        ) or (
            marker_state == "complete" and type(clean_shutdown) is not bool
        )
        if complete_marker_valid and (
            marker_state != "complete"
            or type(clean_shutdown) is not bool
            or not isinstance(shutdown_reason, str)
            or not shutdown_reason
        ):
            terminal_fact_inconsistent = True
        if terminal_fact_inconsistent:
            issue_types.append("segment_manifest_terminal_state_invalid")
        if clean_shutdown is None or marker_state != "complete":
            issue_types.append("segment_manifest_not_finished")
        elif clean_shutdown is False:
            issue_types.append("segment_unclean_shutdown")

        identity_mismatch = any(
            issue in issue_types
            for issue in (
                "segment_manifest_identity_mismatch",
                "segment_manifest_kind_mismatch",
                "segment_manifest_parent_mismatch",
            )
        )
        diagnostics["segment_manifest_identity_mismatch"] = identity_mismatch
        return loaded, "readable", diagnostics

    def _inspect_marker(
        self,
        path: Path,
        expected_marker: str,
        expected_segment_id: str,
        expected_parent_run_id: Optional[str],
        issue_types: list[str],
    ) -> Optional[dict[str, Any]]:
        if not os.path.lexists(path):
            return None
        prefix = expected_marker.lower()
        try:
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("marker必须是非symlink普通文件")
            marker = _strict_json_object(
                _read_regular_bytes_no_follow(
                    path, self.config.root_dir.resolve()
                ),
                f"{expected_marker} marker",
            )
        except (OSError, RuntimeError, ValueError):
            issue_types.append(f"{prefix}_marker_invalid")
            return None
        if (
            marker.get("schema_name") != "team_sorting.recorder.marker"
            or type(marker.get("schema_version")) is not int
            or marker.get("schema_version") != 1
            or marker.get("marker") != expected_marker
        ):
            issue_types.append(f"{prefix}_marker_invalid")
        raw_segment_id = marker.get("recorder_segment_id")
        raw_parent_run_id = marker.get("parent_run_id")
        identity_type_valid = isinstance(raw_segment_id, str) and (
            raw_parent_run_id is None or isinstance(raw_parent_run_id, str)
        )
        if not identity_type_valid:
            issue_types.append(f"{prefix}_marker_invalid")
        if (
            raw_segment_id != expected_segment_id
            or raw_parent_run_id != expected_parent_run_id
        ):
            issue_types.append(f"{prefix}_marker_identity_mismatch")
        return marker

    @staticmethod
    def _inspect_jsonl(path: Path) -> dict[str, Any]:
        return RecorderRuntimeManager._inspect_jsonl_bytes(path.name, path.read_bytes())

    @staticmethod
    def _inspect_jsonl_bytes(name: str, data: bytes) -> dict[str, Any]:
        lines = data.splitlines(keepends=True)
        offset = 0
        valid_end = 0
        middle: list[int] = []
        issues: list[str] = []
        trailing = b""
        for index, line in enumerate(lines):
            complete = line.endswith(b"\n")
            content = line[:-1] if complete else line
            if content.endswith(b"\r"):
                content = content[:-1]
            is_last = index == len(lines) - 1
            try:
                _strict_json_object(content, f"{name} JSONL record")
                valid_json = True
            except ValueError:
                valid_json = False
            if is_last and (not complete or not valid_json):
                trailing = line
                issues.append(
                    "jsonl_trailing_incomplete"
                    if not complete
                    else "jsonl_trailing_json_invalid"
                )
            elif not valid_json:
                middle.append(offset)
                issues.append("jsonl_middle_corruption")
            elif not middle and not trailing:
                valid_end = offset + len(line)
            offset += len(line)
        return {
            "path": name,
            "file_size": len(data),
            "valid_byte_end_offset": valid_end,
            "trailing_fragment_sha256": (
                None if not trailing else hashlib.sha256(trailing).hexdigest()
            ),
            "middle_corruption_offsets": middle,
            "issue_types": sorted(set(issues)),
        }

    def _runtime_provenance(self) -> dict[str, Any]:
        try:
            package_version = importlib_metadata.version("team_sorting")
        except importlib_metadata.PackageNotFoundError:
            package_version = "source_tree_uninstalled"
        return {
            "python_version": sys.version.split()[0],
            "package_version": package_version,
            "recorder_schema_sha256": recorder_contract_sha256(),
            "interface_schema_sha256": _sha256_file(interface_contract_path()),
            "docker_image_digest": self._injected_envelope("docker_image_digest"),
            "hostname_collected": False,
        }

    def _injected_envelope(self, key: str) -> dict[str, Any]:
        variable = PROVENANCE_ENV[key]
        value = self._environment.get(variable, "").strip()
        return (
            _available(value, "launcher_injected")
            if value
            else _unknown("launcher_injected", "not_injected_by_launcher")
        )

    def _dirty_envelope(self, warnings: list[dict[str, Any]]) -> dict[str, Any]:
        variable = PROVENANCE_ENV["dirty_worktree"]
        raw = self._environment.get(variable, "").strip().lower()
        if raw in {"true", "1"}:
            return _available(True, "launcher_injected")
        if raw in {"false", "0"}:
            return _available(False, "launcher_injected")
        reason = "not_injected_by_launcher" if not raw else "invalid_strict_boolean"
        if raw:
            warnings.append({"field": "dirty_worktree", "reason": reason})
        return _unknown("launcher_injected", reason)

    def _config_hash_envelope(self, warnings: list[dict[str, Any]]) -> dict[str, Any]:
        path = self.config.config_path
        if path is not None and path.is_file():
            return _available(_sha256_file(path), "auto_detected")
        warnings.append({"field": "config_sha256", "reason": "config_path_unavailable"})
        return _unknown("auto_detected", "config_path_unavailable")

    def _ros_domain_envelope(self, warnings: list[dict[str, Any]]) -> dict[str, Any]:
        raw = self._environment.get("ROS_DOMAIN_ID", "").strip()
        if not raw:
            return _unknown("auto_detected", "ROS_DOMAIN_ID_unset")
        try:
            value = int(raw)
        except ValueError:
            warnings.append({"field": "ros_domain_id", "reason": "invalid_integer"})
            return _unknown("auto_detected", "invalid_ROS_DOMAIN_ID_integer")
        return _available(value, "auto_detected")

    def _read_json(self, path: Path, artifact_type: str) -> dict[str, Any]:
        try:
            raw = _read_regular_bytes_no_follow(
                path, self.config.root_dir.resolve()
            )
            payload = _strict_json_object(raw, artifact_type)
        except (OSError, RuntimeError, ValueError) as exc:
            raise _artifact_error(artifact_type, path, None, None, exc) from exc
        return payload

    def _validate_contract_record(
        self, section_name: str, payload: Mapping[str, Any]
    ) -> None:
        """Validate required v1 field shape without embedding a second schema."""

        fields = self.contract[section_name]["fields"]
        for name, descriptor in fields.items():
            if descriptor["required"] and name not in payload:
                raise RuntimeError(f"{section_name}缺少冻结契约字段：{name}")
            if name not in payload:
                continue
            value = payload[name]
            if value is None:
                if not descriptor["nullable"]:
                    raise RuntimeError(f"{section_name}.{name}不得为null")
                continue
            kind = descriptor["type"]
            valid = True
            if kind in {"string", "sha256", "string_or_null"}:
                valid = isinstance(value, str)
            elif kind in {"integer", "integer_or_null"}:
                valid = type(value) is int
            elif kind in {"boolean", "boolean_or_null"}:
                valid = type(value) is bool
            elif kind.startswith("array<"):
                valid = isinstance(value, list)
                if valid:
                    item_kind = kind[6:-1]
                    if item_kind == "string":
                        valid = all(isinstance(item, str) for item in value)
                    elif item_kind == "integer":
                        valid = all(type(item) is int for item in value)
                    elif item_kind == "object":
                        valid = all(isinstance(item, dict) for item in value)
            elif kind == "tuple_or_null":
                valid = isinstance(value, list)
            elif kind == "map<string,integer>":
                valid = isinstance(value, dict) and all(
                    isinstance(key, str) and type(item) is int
                    for key, item in value.items()
                )
            elif kind == "object" or kind.startswith("unknown_value<"):
                valid = isinstance(value, dict)
            if not valid:
                raise RuntimeError(
                    f"{section_name}.{name}不符合冻结契约类型：{kind}"
                )
            allowed = descriptor.get("allowed_values")
            if allowed is not None and value not in allowed:
                raise RuntimeError(
                    f"{section_name}.{name}不在冻结契约允许值中：{value!r}"
                )
            if kind == "sha256" and re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise RuntimeError(f"{section_name}.{name}不是SHA-256")
            minimum = descriptor.get("minimum")
            if minimum is not None and isinstance(value, (int, float)) and value < minimum:
                raise RuntimeError(
                    f"{section_name}.{name}小于冻结契约minimum={minimum}"
                )

    def _require_active_for_write(self, *, allow_closing: bool = False) -> _ActiveSegment:
        allowed = {
            RecorderRuntimeState.BOOTSTRAP_ACTIVE,
            RecorderRuntimeState.RUN_ACTIVE,
        }
        if allow_closing:
            allowed.add(RecorderRuntimeState.CLOSING)
        if self._active is None or self.state not in allowed:
            raise RuntimeError(f"Recorder runtime没有可写segment，state={self.state.value}")
        assert self._active is not None
        if not allow_closing and (
            not self._active.accepting_writes
            or self._active.closing_started
            or self._active.recorder_finished
            or self._active.complete_created
        ):
            raise RuntimeError(
                f"Recorder runtime segment已进入关闭阶段，state={self.state.value}"
            )
        return self._active

    @staticmethod
    def _require_ns(value: object, label: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{label}必须是非负整数且不能是bool")
        return value

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex}"
