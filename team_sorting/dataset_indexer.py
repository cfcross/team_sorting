"""Read-only Dataset Indexer and core QC v1.

Raw artifacts are opened only for bounded reads.  All writes are confined to
``derived/indexer_v1`` and published by a same-filesystem directory rename.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import yaml

from .competition_context import CompetitionContext
from .data_tf_policy_contract import data_tf_policy_contract_path, load_data_tf_policy_contract
from .dataset_index_contract import (
    dataset_index_contract_path,
    dataset_index_contract_sha256,
    load_dataset_index_contract,
)
from .dataset_qc import PERCEPTION_TOPICS, aggregate_eligibility, finding, not_evaluated, segment_eligibility
from .interface_contract import interface_contract_path
from .recorder_contract import recorder_contract_path, load_recorder_contract
from .recording_contracts import strict_final_action_from_json, strict_action_dispatch_from_json


INDEXER_NAME = "team_sorting_dataset_index"
INDEXER_VERSION = "1.0.0"
IMPLEMENTATION_IDENTITY = "team_sorting.dataset_indexer:b3c_core_v1"
OUTPUT_NAMES = ("dataset_index.jsonl", "segment_qc.json", "run_qc.json")
_DEFERRED_CHECKS = (
    "tf_static_required_edge_missing", "tf_graph_disconnected",
    "rgb_depth_alignment_failed", "joint_state_dimension_invalid", "odom_frame_invalid",
    "selected_dispatched_mismatch", "execution_feedback_missing",
)


class DatasetIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexBuildResult:
    build_id: str
    output_directory: Path | None
    reused: bool
    segment_count: int
    run_count: int
    finding_count: int
    raw_immutability_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_id": self.build_id,
            "output_directory": None if self.output_directory is None else str(self.output_directory),
            "reused": self.reused,
            "segment_count": self.segment_count,
            "run_count": self.run_count,
            "finding_count": self.finding_count,
            "raw_immutability_verified": self.raw_immutability_verified,
        }


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetIndexError(f"JSON包含重复key：{key!r}")
        result[key] = value
    return result


def _json_constant(value: str) -> None:
    raise DatasetIndexError(f"JSON包含非法常量：{value}")


def _safe_regular(path: Path, root: Path, limit: int, label: str) -> Path:
    if path.is_absolute():
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise DatasetIndexError(f"{label}路径逃逸dataset root：{path}") from exc
    candidate = path if path.is_absolute() else root / path
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise DatasetIndexError(f"{label}缺失：{candidate}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise DatasetIndexError(f"{label}不得是符号链接：{candidate}")
    if not stat.S_ISREG(metadata.st_mode):
        raise DatasetIndexError(f"{label}必须是普通文件：{candidate}")
    if metadata.st_size > limit:
        raise DatasetIndexError(f"{label}超过大小上限：{metadata.st_size}>{limit}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DatasetIndexError(f"{label}解析后逃逸dataset root：{resolved}") from exc
    return resolved


def _read_json(path: Path, root: Path, limit: int, label: str) -> dict[str, Any]:
    selected = _safe_regular(path, root, limit, label)
    try:
        value = json.loads(selected.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs, parse_constant=_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetIndexError(f"{label}严格JSON读取失败：{selected}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetIndexError(f"{label}顶层必须是object：{selected}")
    return value


def _read_jsonl(path: Path, root: Path, limit: int) -> tuple[list[dict[str, Any]], bool]:
    selected = _safe_regular(path, root, limit, "JSONL")
    try:
        raw = selected.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise DatasetIndexError(f"JSONL不可读：{selected}: {exc}") from exc
    records: list[dict[str, Any]] = []
    trailing_partial = False
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            raise DatasetIndexError(f"JSONL包含空记录：{selected}:{index + 1}")
        try:
            value = json.loads(line, object_pairs_hook=_strict_pairs, parse_constant=_json_constant)
        except (json.JSONDecodeError, DatasetIndexError) as exc:
            if index == len(lines) - 1 and not raw.endswith(b"\n"):
                trailing_partial = True
                break
            raise DatasetIndexError(f"JSONL中间损坏：{selected}:{index + 1}: {exc}") from exc
        if not isinstance(value, dict):
            raise DatasetIndexError(f"JSONL记录顶层必须是object：{selected}:{index + 1}")
        records.append(value)
    return records, trailing_partial


def _snapshot_raw(root: Path, limits: Mapping[str, int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root)
        if relative_dir.parts and relative_dir.parts[0] == "derived":
            directories[:] = []
            continue
        for name in tuple(directories):
            child = current_path / name
            if child.is_symlink():
                raise DatasetIndexError(f"unexpected_symlink：{child.relative_to(root).as_posix()}")
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            suffix = path.suffix.lower()
            limit = limits["sqlite"] if suffix == ".db3" else limits["jsonl"] if suffix == ".jsonl" else limits["yaml"] if suffix in {".yaml", ".yml"} else limits["json"]
            safe = _safe_regular(path, root, int(limit), "source artifact")
            metadata = os.lstat(safe)
            records.append({"relative_path": relative, "sha256": _sha_file(safe), "size": metadata.st_size, "mtime_ns": metadata.st_mtime_ns})
    return sorted(records, key=lambda item: item["relative_path"])


def _yaml_metadata(path: Path, root: Path, limit: int) -> dict[str, Any]:
    selected = _safe_regular(path, root, limit, "rosbag metadata")
    if selected.stat().st_size == 0:
        raise DatasetIndexError("rosbag metadata为空")
    try:
        class UniqueLoader(yaml.SafeLoader):
            pass
        def construct_mapping(loader, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in result:
                    raise yaml.YAMLError(f"duplicate YAML key: {key!r}")
                result[key] = loader.construct_object(value_node, deep=deep)
            return result
        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
        value = yaml.load(selected.read_text(encoding="utf-8"), Loader=UniqueLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DatasetIndexError(f"rosbag metadata YAML非法：{exc}") from exc
    if not isinstance(value, dict) or set(value) != {"rosbag2_bagfile_information"}:
        raise DatasetIndexError("rosbag metadata顶层必须只包含rosbag2_bagfile_information")
    info = value["rosbag2_bagfile_information"]
    if not isinstance(info, dict) or not isinstance(info.get("storage_identifier"), str) or not info["storage_identifier"]:
        raise DatasetIndexError("rosbag metadata storage_identifier非法")
    duration = info.get("duration")
    if not isinstance(duration, dict) or type(duration.get("nanoseconds")) is not int or duration["nanoseconds"] < 0:
        raise DatasetIndexError("rosbag metadata duration.nanoseconds非法")
    topics = info.get("topics_with_message_count")
    if not isinstance(topics, list):
        raise DatasetIndexError("rosbag metadata topics_with_message_count非法")
    names = []
    total = 0
    for item in topics:
        topic = item.get("topic_metadata") if isinstance(item, dict) else None
        if not isinstance(topic, dict) or any(not isinstance(topic.get(key), str) or not topic[key] for key in ("name", "type", "serialization_format")) or type(item.get("message_count")) is not int or item["message_count"] < 0:
            raise DatasetIndexError("rosbag metadata topic记录非法")
        names.append(topic["name"]); total += item["message_count"]
    if len(names) != len(set(names)):
        raise DatasetIndexError("rosbag metadata topic name重复")
    if "message_count" in info and (type(info["message_count"]) is not int or info["message_count"] != total):
        raise DatasetIndexError("rosbag metadata总message_count不一致")
    return value


def _validate_schema_record(section: Mapping[str, Any], payload: Mapping[str, Any], label: str) -> None:
    for name, descriptor in section["fields"].items():
        if descriptor["required"] and name not in payload:
            raise DatasetIndexError(f"{label}缺少冻结字段：{name}")
        if name not in payload:
            continue
        value = payload[name]
        if value is None:
            if not descriptor["nullable"]: raise DatasetIndexError(f"{label}.{name}不得为null")
            continue
        kind = descriptor["type"]
        if kind in {"string", "string_or_null", "sha256"}: valid = isinstance(value, str)
        elif kind in {"integer", "integer_or_null"}: valid = type(value) is int
        elif kind in {"boolean", "boolean_or_null"}: valid = type(value) is bool
        elif kind.startswith("array<"):
            valid = isinstance(value, list)
            subtype = kind[6:-1]
            if valid and subtype == "string": valid = all(isinstance(x, str) for x in value)
            if valid and subtype == "integer": valid = all(type(x) is int for x in value)
            if valid and subtype == "object": valid = all(isinstance(x, dict) for x in value)
        elif kind == "map<string,integer>": valid = isinstance(value, dict) and all(isinstance(k, str) and type(v) is int for k, v in value.items())
        elif kind.startswith("unknown_value<") or kind == "object": valid = isinstance(value, dict)
        else: valid = True
        if not valid: raise DatasetIndexError(f"{label}.{name}类型非法：{kind}")
        if "allowed_values" in descriptor and value not in descriptor["allowed_values"]: raise DatasetIndexError(f"{label}.{name}值非法")
        if kind == "sha256" and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)): raise DatasetIndexError(f"{label}.{name}不是SHA256")
        if "minimum" in descriptor and value < descriptor["minimum"]: raise DatasetIndexError(f"{label}.{name}小于minimum")


def _metadata_counts(metadata: Mapping[str, Any]) -> tuple[str | None, dict[str, int], int | None]:
    info = metadata.get("rosbag2_bagfile_information", metadata)
    if not isinstance(info, Mapping):
        return None, {}, None
    storage = info.get("storage_identifier") if isinstance(info.get("storage_identifier"), str) else None
    duration = info.get("duration")
    duration_ns = duration.get("nanoseconds") if isinstance(duration, Mapping) and type(duration.get("nanoseconds")) is int else None
    counts: dict[str, int] = {}
    values = info.get("topics_with_message_count", [])
    if isinstance(values, list):
        for item in values:
            if not isinstance(item, Mapping):
                continue
            topic = item.get("topic_metadata")
            name = topic.get("name") if isinstance(topic, Mapping) else None
            count = item.get("message_count")
            if isinstance(name, str) and type(count) is int:
                counts[name] = count
    return storage, counts, duration_ns


def _sqlite_topics(files: Sequence[Path], root: Path, limit: int) -> tuple[list[dict[str, Any]], bool, int]:
    aggregate: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
    regression = False
    total_bytes = 0
    previous_topic_last: dict[str, int] = {}
    for path in files:
        safe = _safe_regular(path, root, limit, "rosbag SQLite")
        uri = f"file:{quote(safe.as_posix(), safe='/')}?mode=ro&immutable=1"
        try:
            connection = sqlite3.connect(uri, uri=True)
            connection.enable_load_extension(False)
            connection.execute("PRAGMA query_only=ON")
            connection.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action in {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE, sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_DROP_TABLE} else sqlite3.SQLITE_OK)
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise DatasetIndexError(f"SQLite integrity_check失败：{safe}: {integrity}")
            rows = connection.execute("SELECT id,name,type,serialization_format,offered_qos_profiles FROM topics ORDER BY id").fetchall()
            for topic_id, name, message_type, serialization, offered in rows:
                stats = connection.execute("SELECT COUNT(*),MIN(timestamp),MAX(timestamp),COALESCE(SUM(LENGTH(data)),0) FROM messages WHERE topic_id=?", (topic_id,)).fetchone()
                previous: int | None = None
                for (timestamp,) in connection.execute("SELECT timestamp FROM messages WHERE topic_id=? ORDER BY id", (topic_id,)):
                    if previous is not None and timestamp < previous:
                        regression = True
                    previous = timestamp
                key = (str(name), str(message_type), str(serialization), bool(offered))
                item = aggregate.setdefault(key, {"name": str(name), "type": str(message_type), "count": 0, "serialization_format": str(serialization), "offered_qos_profiles_present": bool(offered), "first_timestamp_ns": None, "last_timestamp_ns": None, "payload_bytes": 0})
                count, first, last, payload_bytes = stats
                if first is not None and name in previous_topic_last and int(first) < previous_topic_last[str(name)]:
                    regression = True
                if last is not None:
                    previous_topic_last[str(name)] = int(last)
                item["count"] += int(count)
                item["payload_bytes"] += int(payload_bytes)
                total_bytes += int(payload_bytes)
                if first is not None:
                    item["first_timestamp_ns"] = int(first) if item["first_timestamp_ns"] is None else min(item["first_timestamp_ns"], int(first))
                    item["last_timestamp_ns"] = int(last) if item["last_timestamp_ns"] is None else max(item["last_timestamp_ns"], int(last))
        except sqlite3.DatabaseError as exc:
            raise DatasetIndexError(f"SQLite只读分析失败：{safe}: {exc}") from exc
        finally:
            if "connection" in locals():
                connection.close()
                del connection
    return sorted(aggregate.values(), key=lambda item: item["name"]), regression, total_bytes


class DatasetIndexer:
    def __init__(self, dataset_root: str | Path, qc_config: Mapping[str, Any] | None = None, output_root: str | Path | None = None) -> None:
        requested = Path(dataset_root)
        try:
            metadata = os.lstat(requested)
        except OSError as exc:
            raise DatasetIndexError(f"dataset_root不可访问：{requested}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise DatasetIndexError("dataset_root必须是真实目录且不能是symlink")
        self.root = requested.resolve(strict=True)
        self.contract = load_dataset_index_contract()
        self.recorder_contract = load_recorder_contract()
        self.policy = load_data_tf_policy_contract()
        default_required = [item["name"] for item in self.policy["topic_policy"]["current_raw_baseline"]] + ["/tf"]
        raw_qc = dict(qc_config or {"required_topics": default_required, "required_static_edges": []})
        if set(raw_qc) != {"required_topics", "required_static_edges"}:
            raise DatasetIndexError("QC配置字段必须精确为required_topics和required_static_edges")
        required_topics = raw_qc["required_topics"]
        known_topics = {item["name"] for item in (*self.policy["topic_policy"]["current_raw_baseline"], *self.policy["topic_policy"]["b3_target_topics"])} | {"/tf"}
        if not isinstance(required_topics, list) or any(not isinstance(item, str) or not item.startswith("/") or item == "/" for item in required_topics) or len(required_topics) != len(set(required_topics)) or not set(required_topics) <= known_topics:
            raise DatasetIndexError("QC required_topics必须是已知、唯一、非空ROS绝对topic数组")
        edges = raw_qc["required_static_edges"]
        if not isinstance(edges, list) or edges:
            raise DatasetIndexError("B3C core尚未实现required_static_edges，仅允许空数组")
        self.qc_config = raw_qc
        self.qc_sha = _sha_bytes(_canonical(self.qc_config))
        default_output = self.root / "derived/indexer_v1"
        selected = default_output if output_root is None else Path(output_root)
        if not selected.is_absolute():
            selected = self.root / selected
        resolved_parent = selected.parent.resolve(strict=True) if selected.parent.exists() else selected.parent
        selected = resolved_parent / selected.name
        try:
            selected.relative_to(default_output)
        except ValueError as exc:
            if selected != default_output:
                raise DatasetIndexError("output_root必须位于dataset_root/derived/indexer_v1内") from exc
        self.output_root = selected
        self.limits = self.contract["safety"]["size_limits_bytes"]
        expected = {item["name"]: item["message_type"] for item in (*self.policy["topic_policy"]["current_raw_baseline"], *self.policy["topic_policy"]["b3_target_topics"])}
        self.expected_types = expected

    def _segments(self) -> list[Path]:
        values: list[Path] = []
        bootstrap = self.root / "bootstrap"
        if bootstrap.is_dir() and not bootstrap.is_symlink():
            values.extend(path for path in bootstrap.iterdir() if path.is_dir() and not path.is_symlink())
        runs = self.root / "runs"
        if runs.is_dir() and not runs.is_symlink():
            for run in runs.iterdir():
                segments = run / "segments"
                if run.is_dir() and not run.is_symlink() and segments.is_dir() and not segments.is_symlink():
                    values.extend(path for path in segments.iterdir() if path.is_dir() and not path.is_symlink())
        return sorted(values, key=lambda path: path.relative_to(self.root).as_posix())

    def _segment(self, path: Path, build_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        relative = path.relative_to(self.root).as_posix()
        findings: list[dict[str, Any]] = []
        manifest_path = path / "segment.json"
        manifest: dict[str, Any] = {}
        try:
            manifest = _read_json(manifest_path, self.root, self.limits["json"], "segment manifest")
            _validate_schema_record(self.recorder_contract["recorder_segment_schema"], manifest, "segment manifest")
            if manifest.get("schema_name") != "team_sorting.recorder.segment" or type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
                raise DatasetIndexError("segment manifest schema identity非法")
        except DatasetIndexError as exc:
            findings.append(finding("segment_manifest_invalid", "fatal", "segment_manifest", relative + "/segment.json", str(exc), blocking_use_cases=("diagnostic", "perception", "formal_bc")))
        segment_id = manifest.get("recorder_segment_id", path.name)
        kind = manifest.get("segment_kind", "unknown")
        parent = manifest.get("parent_run_id")
        if segment_id != path.name:
            findings.append(finding("segment_identity_mismatch", "fatal", "segment_manifest", relative, "segment ID与目录名不一致", evidence={"manifest": segment_id, "directory": path.name}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
        relative_parts = PurePosixPath(relative).parts
        physical_run_id = relative_parts[1] if len(relative_parts) >= 4 and relative_parts[0] == "runs" and relative_parts[2] == "segments" else None
        if physical_run_id is not None and parent != physical_run_id:
            findings.append(finding("segment_parent_run_id_mismatch", "fatal", "segment_manifest", relative, "Segment parent_run_id与物理Run目录不一致", evidence={"parent_run_id": parent, "directory_run_id": physical_run_id}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
        def marker_valid(name: str) -> bool:
            marker_path = path / name
            if not marker_path.exists(): return False
            try:
                marker = _read_json(marker_path, self.root, self.limits["json"], f"{name} marker")
            except DatasetIndexError as exc:
                findings.append(finding("marker_invalid", "fatal", "marker", relative + "/" + name, str(exc), blocking_use_cases=("diagnostic", "perception", "formal_bc"))); return False
            required_time = "created_wall_utc" if name == "ACTIVE" else "completed_wall_utc"
            if marker.get("schema_name") != "team_sorting.recorder.marker" or type(marker.get("schema_version")) is not int or marker.get("schema_version") != 1 or marker.get("marker") != name or not isinstance(marker.get(required_time), str) or not marker[required_time]:
                findings.append(finding("marker_invalid", "fatal", "marker", relative + "/" + name, "marker结构非法", blocking_use_cases=("diagnostic", "perception", "formal_bc"))); return False
            if marker.get("recorder_segment_id") != segment_id or marker.get("parent_run_id") != parent:
                findings.append(finding("marker_identity_mismatch", "fatal", "marker", relative + "/" + name, "marker身份与Segment不一致", blocking_use_cases=("diagnostic", "perception", "formal_bc"))); return False
            return True
        active = marker_valid("ACTIVE")
        complete = marker_valid("COMPLETE")
        if active:
            findings.append(finding("active_marker_remaining", "error", "marker", relative + "/ACTIVE", "ACTIVE marker仍存在", blocking_use_cases=("perception", "formal_bc")))
        if not complete:
            findings.append(finding("complete_marker_missing", "error", "marker", relative + "/COMPLETE", "COMPLETE marker缺失", blocking_use_cases=("perception", "formal_bc")))
        if (complete and (manifest.get("marker_state") != "complete" or manifest.get("clean_shutdown") is not True or not isinstance(manifest.get("shutdown_reason"), str) or not manifest.get("shutdown_reason"))) or (not complete and manifest.get("marker_state") == "complete"):
            findings.append(finding("segment_manifest_invalid", "fatal", "segment_manifest", relative, "Segment终态与Marker不一致", blocking_use_cases=("diagnostic", "perception", "formal_bc")))
        bag_required = manifest.get("bag_path") is not None
        bag_reference = manifest.get("bag_path")
        bag_dir: Path | None = None
        if bag_reference is not None:
            unsafe_reference = not isinstance(bag_reference, str) or PurePosixPath(bag_reference).is_absolute() or ".." in PurePosixPath(bag_reference).parts
            if unsafe_reference:
                findings.append(finding("source_path_escape", "fatal", "segment_manifest", relative, "bag_path不是安全Segment相对路径", evidence={"bag_path": bag_reference}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            else:
                candidate = path / bag_reference
                if candidate.is_symlink() or not candidate.is_dir():
                    findings.append(finding("bag_path_invalid", "fatal", "segment_manifest", relative, "bag_path目录缺失或不安全", evidence={"bag_path": bag_reference}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
                else: bag_dir = candidate
        elif (path / "rosbag").exists():
            findings.append(finding("bag_path_invalid", "fatal", "segment_manifest", relative, "manifest bag_path=null但物理bag目录存在", blocking_use_cases=("diagnostic", "perception", "formal_bc")))
        storage: str | None = None
        duration: int | None = None
        metadata_counts: dict[str, int] = {}
        topics: list[dict[str, Any]] = []
        bag_files: list[dict[str, Any]] = []
        regression = False
        if bag_required and bag_dir is not None:
            metadata_path = bag_dir / "metadata.yaml"
            if not metadata_path.exists():
                findings.append(finding("rosbag_metadata_missing", "error", "rosbag", relative + "/rosbag/metadata.yaml", "rosbag metadata缺失", blocking_use_cases=("perception", "formal_bc")))
            else:
                try:
                    metadata = _yaml_metadata(metadata_path, self.root, self.limits["yaml"])
                    storage, metadata_counts, duration = _metadata_counts(metadata)
                    relative_files = metadata["rosbag2_bagfile_information"].get("relative_file_paths")
                    if relative_files is not None:
                        if not isinstance(relative_files, list) or any(not isinstance(item, str) or PurePosixPath(item).is_absolute() or ".." in PurePosixPath(item).parts for item in relative_files):
                            raise DatasetIndexError("rosbag metadata relative_file_paths非法")
                        actual_files = sorted(item.name for item in bag_dir.glob("*.db3"))
                        if sorted(relative_files) != actual_files:
                            raise DatasetIndexError("rosbag metadata relative_file_paths与实际DB不一致")
                except DatasetIndexError as exc:
                    findings.append(finding("rosbag_metadata_invalid", "error", "rosbag", metadata_path.relative_to(self.root).as_posix(), str(exc), blocking_use_cases=("perception", "formal_bc")))
            db_files = sorted(bag_dir.glob("*.db3"))
            try:
                topics, regression, _ = _sqlite_topics(db_files, self.root, self.limits["sqlite"])
                bag_files = [{"relative_path": item.relative_to(self.root).as_posix(), "sha256": _sha_file(item), "size": item.stat().st_size} for item in db_files]
            except DatasetIndexError as exc:
                findings.append(finding("sqlite_integrity_failed", "fatal", "rosbag", relative + "/rosbag", str(exc), blocking_use_cases=("diagnostic", "perception", "formal_bc")))
        exit_code = manifest.get("bag_exit_code")
        if bag_required and exit_code != 0:
            findings.append(finding("rosbag_exit_nonzero", "error", "segment_manifest", relative, "rosbag exit code不是0", evidence={"exit_code": exit_code}, blocking_use_cases=("perception", "formal_bc")))
        topic_map = {item["name"]: item for item in topics}
        for name in self.qc_config.get("required_topics", []):
            if name not in topic_map:
                code = "camera_info_missing" if name == "/head_camera/color/camera_info" else "tf_dynamic_missing" if name == "/tf" else "required_topic_missing"
                blockers = ("formal_bc",) if code == "tf_dynamic_missing" else (("perception", "formal_bc") if name in PERCEPTION_TOPICS else ("formal_bc",))
                findings.append(finding(code, "error", "rosbag", relative, f"required topic缺失：{name}", evidence={"topic": name}, blocking_use_cases=blockers))
            elif topic_map[name]["count"] == 0:
                findings.append(finding("empty_required_topic", "error", "rosbag", relative, f"required topic消息数为0：{name}", evidence={"topic": name}, blocking_use_cases=("perception", "formal_bc")))
        for name, item in topic_map.items():
            expected = self.expected_types.get(name)
            if expected and item["type"] != expected:
                findings.append(finding("topic_type_mismatch", "error", "rosbag", relative, f"topic类型不匹配：{name}", evidence={"actual": item["type"], "expected": expected}, blocking_use_cases=("perception", "formal_bc")))
            if name in metadata_counts and metadata_counts[name] != item["count"]:
                findings.append(finding("metadata_sqlite_count_mismatch", "warning", "rosbag", relative, f"metadata与SQLite计数不一致：{name}", evidence={"metadata": metadata_counts[name], "sqlite": item["count"]}))
            if item["count"] and (item["first_timestamp_ns"] is None or item["last_timestamp_ns"] is None or item["first_timestamp_ns"] > item["last_timestamp_ns"]):
                findings.append(finding("message_timestamp_span_invalid", "error", "rosbag", relative, f"topic时间跨度非法：{name}", blocking_use_cases=("perception", "formal_bc")))
        if regression:
            findings.append(finding("timestamp_regression", "error", "rosbag", relative, "SQLite消息按写入ID出现时间戳倒退", blocking_use_cases=("perception", "formal_bc")))
        contexts: list[dict[str, Any]] = []
        context_path = path / "competition_contexts.jsonl"
        if context_path.exists():
            try:
                contexts, partial = _read_jsonl(context_path, self.root, self.limits["jsonl"])
                if partial:
                    findings.append(finding("json_invalid", "warning", "jsonl", relative + "/competition_contexts.jsonl", "仅末尾不完整JSONL记录被隔离"))
            except DatasetIndexError as exc:
                findings.append(finding("jsonl_midstream_corruption", "fatal", "jsonl", relative + "/competition_contexts.jsonl", str(exc), blocking_use_cases=("diagnostic", "perception", "formal_bc")))
        parsed_contexts: list[CompetitionContext] = []
        observe_only = None
        if parent:
            run_manifest_path = self.root / "runs" / str(parent) / "manifest.json"
            if run_manifest_path.exists():
                try:
                    observe_only = _read_json(run_manifest_path, self.root, self.limits["json"], "run manifest").get("observe_only")
                except DatasetIndexError:
                    pass
        structurally_invalid = 0
        for item in contexts:
            try:
                parsed_contexts.append(CompetitionContext.from_json(_canonical(item).decode("utf-8")))
            except ValueError:
                structurally_invalid += 1
        if not parsed_contexts:
            findings.append(finding("competition_context_missing", "error", "jsonl", relative, "CompetitionContext缺失", blocking_use_cases=("perception", "formal_bc")))
        invalid_contexts = [item for item in parsed_contexts if not item.valid]
        if invalid_contexts or structurally_invalid:
            findings.append(finding("competition_context_invalid", "error", "jsonl", relative, "存在无效CompetitionContext", evidence={"semantic_invalid_count": len(invalid_contexts), "structural_invalid_count": structurally_invalid}, blocking_use_cases=("perception", "formal_bc")))
        run_ids = sorted({item.run_id for item in parsed_contexts})
        fingerprints = sorted({item.task_set_fingerprint for item in parsed_contexts if item.valid})
        if len(run_ids) > 1:
            findings.append(finding("run_id_mixed", "fatal", "jsonl", relative, "同一Segment混入多个run_id", evidence={"run_ids": run_ids}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
        if kind == "run_bound" and run_ids and run_ids != [parent]:
            findings.append(finding("segment_context_identity_mismatch", "fatal", "jsonl", relative, "Context run_id与parent_run_id不一致", evidence={"run_ids": run_ids, "parent_run_id": parent}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
        parsed_jsonl_counts: dict[str, int] = {}
        parsed_jsonl_records: dict[str, list[dict[str, Any]]] = {}
        for jsonl_path in sorted(path.glob("*.jsonl")):
            if jsonl_path.name == "competition_contexts.jsonl":
                parsed_jsonl_counts[jsonl_path.name] = len(contexts)
                continue
            try:
                records, partial = _read_jsonl(jsonl_path, self.root, self.limits["jsonl"])
                parsed_jsonl_counts[jsonl_path.name] = len(records)
                parsed_jsonl_records[jsonl_path.name] = records
                if partial:
                    findings.append(finding("json_invalid", "warning", "jsonl", jsonl_path.relative_to(self.root).as_posix(), "仅末尾不完整JSONL记录被隔离"))
            except DatasetIndexError as exc:
                findings.append(finding("jsonl_midstream_corruption", "fatal", "jsonl", jsonl_path.relative_to(self.root).as_posix(), str(exc), blocking_use_cases=("diagnostic", "perception", "formal_bc")))
                parsed_jsonl_counts[jsonl_path.name] = 0
                parsed_jsonl_records[jsonl_path.name] = []
        selected = False
        dispatched = False
        action_summary = {"selected_action_record_count": 0, "selected_action_valid_record_count": 0, "selected_action_invalid_record_count": 0,
                          "selected_action_record_present": False,
                          "action_dispatch_record_count": 0, "action_dispatch_valid_record_count": 0, "action_dispatch_invalid_record_count": 0,
                          "action_dispatch_record_present": False, "publish_attempted_record_count": 0, "exact_payload_record_count": 0,
                          "exact_dispatched_action_present": False, "publisher_call_succeeded_record_count": 0}
        for name, parser in (("final_actions.jsonl", strict_final_action_from_json), ("action_dispatches.jsonl", strict_action_dispatch_from_json)):
            records = parsed_jsonl_records.get(name, [])
            valid = 0
            errors: list[str] = []
            for record in records:
                try:
                    parsed = parser(_canonical(record).decode("utf-8")); valid += 1
                    if name == "action_dispatches.jsonl":
                        payload = any(x is True for x in parsed.dispatched_mask) and any(x is not None for x in parsed.dispatched_action)
                        attempted = parsed.publish_attempted and bool(parsed.attempted_groups) and payload
                        if getattr(parsed, "publish_attempted", False): action_summary["publish_attempted_record_count"] += 1
                        if payload: action_summary["exact_payload_record_count"] += 1
                        if attempted and getattr(parsed, "publisher_call_succeeded", None) is True and observe_only is not True:
                            dispatched = True; action_summary["exact_dispatched_action_present"] = True
                        if getattr(parsed, "publisher_call_succeeded", None) is True: action_summary["publisher_call_succeeded_record_count"] += 1
                except (ValueError, TypeError) as exc:
                    errors.append(str(exc)[:512])
            if name == "final_actions.jsonl":
                invalid = len(records) - valid
                action_summary.update(selected_action_record_count=len(records), selected_action_valid_record_count=valid, selected_action_invalid_record_count=invalid, selected_action_record_present=bool(records)); selected = valid > 0
                if invalid: findings.append(finding("selected_action_invalid", "error", "jsonl", relative + "/" + name, "FinalAction严格语义校验失败", evidence={"invalid_record_count": invalid, "error_summaries": errors[:8], "summaries_truncated": len(errors) > 8}, blocking_use_cases=("formal_bc",)))
            else:
                invalid = len(records) - valid
                action_summary.update(action_dispatch_record_count=len(records), action_dispatch_valid_record_count=valid, action_dispatch_invalid_record_count=invalid, action_dispatch_record_present=bool(records))
                if invalid: findings.append(finding("action_dispatch_invalid", "error", "jsonl", relative + "/" + name, "ActionDispatchRecord严格语义校验失败", evidence={"invalid_record_count": invalid, "error_summaries": errors[:8], "summaries_truncated": len(errors) > 8}, blocking_use_cases=("formal_bc",)))
        if not selected:
            findings.append(finding("selected_action_missing", "warning", "jsonl", relative, "selected action记录缺失", blocking_use_cases=("formal_bc",)))
        if not dispatched:
            findings.append(finding("dispatched_action_missing", "error", "jsonl", relative, "dispatched action记录缺失", blocking_use_cases=("formal_bc",)))
        for code in _DEFERRED_CHECKS:
            findings.append(not_evaluated(code, "derived_qc", relative, "B3C core未实现该检查，不能视为pass"))
        if observe_only is True:
            findings.append(finding("observe_only_not_formal_bc", "error", "run_manifest", relative, "observe_only数据不得用于formal_bc", blocking_use_cases=("formal_bc",)))
        eligibility = segment_eligibility(findings, topic_map, context_valid=bool(parsed_contexts) and not invalid_contexts and structurally_invalid == 0 and len(run_ids) <= 1, observe_only=observe_only, dispatched_action_present=dispatched, complete=complete and not active and (not bag_required or exit_code == 0))
        index = {
            "schema_name": "team_sorting.dataset_index.segment", "schema_version": 1, "build_id": build_id,
            "source_segment_path": relative, "segment_id": segment_id, "segment_kind": kind, "parent_run_id": parent,
            "segment_sequence": manifest.get("segment_sequence"),
            "marker_state": {"active": active, "complete": complete},
            "segment_manifest_sha256": _sha_file(manifest_path) if manifest_path.is_file() and not manifest_path.is_symlink() else None,
            "rosbag_required": bag_required, "rosbag_storage_identifier": storage, "rosbag_duration_ns": duration,
            "rosbag_message_count": sum(item["count"] for item in topics), "rosbag_files": bag_files, "topics": topics,
            "competition_context_summary": {"count": len(contexts), "valid_count": len(parsed_contexts) - len(invalid_contexts), "run_ids": run_ids, "task_set_fingerprints": fingerprints},
            "action_summary": {**action_summary, "selected_action_present": selected, "dispatched_action_present": dispatched, "execution_feedback_evaluation": "not_evaluated"},
            "qc_reference": "segment_qc.json", "eligibility": eligibility,
        }
        qc = {"source_segment_path": relative, "segment_id": segment_id, "findings": findings, "eligibility": eligibility}
        return index, qc

    def _recovery_reports(self, segment_indexes: Sequence[Mapping[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        recovery_dir = self.root / "recovery"
        if not recovery_dir.exists():
            return []
        if recovery_dir.is_symlink() or not recovery_dir.is_dir():
            raise DatasetIndexError("recovery必须是真实目录且不能是symlink")
        known = {item["source_segment_path"] for item in segment_indexes}
        known_parents = {item["parent_run_id"] for item in segment_indexes if item.get("parent_run_id")}
        reports: list[tuple[str, dict[str, Any]]] = []
        for report_path in sorted(recovery_dir.iterdir(), key=lambda item: item.name):
            if report_path.suffix != ".json":
                continue
            relative_report = report_path.relative_to(self.root).as_posix()
            report = _read_json(report_path, self.root, self.limits["json"], "recovery report")
            source_path = report.get("source_segment_path")
            parent = report.get("source_parent_run_id")
            if source_path is not None:
                if not isinstance(source_path, str) or not source_path:
                    raise DatasetIndexError(f"Recovery source_segment_path非法：{relative_report}")
                pure = PurePosixPath(source_path)
                if pure.is_absolute() or ".." in pure.parts or source_path not in known:
                    raise DatasetIndexError(f"Recovery引用不存在或不安全的Segment：{relative_report}: {source_path}")
            if source_path is None and (not isinstance(parent, str) or not parent):
                raise DatasetIndexError(f"Recovery报告无法关联Segment或Run：{relative_report}")
            if source_path is None and parent not in known_parents:
                raise DatasetIndexError(f"Recovery报告引用不存在的Run：{relative_report}: {parent}")
            if source_path is not None:
                index = next(item for item in segment_indexes if item["source_segment_path"] == source_path)
                actual_parent = index.get("parent_run_id")
                if parent is not None and parent != actual_parent:
                    raise DatasetIndexError(f"Recovery Segment与parent run关联冲突：{relative_report}")
            reports.append((relative_report, report))
        return reports

    def _runs(self, segment_indexes: Sequence[Mapping[str, Any]], segment_qc: Sequence[Mapping[str, Any]], recovery_reports: Sequence[tuple[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
        qc_by_path = {item["source_segment_path"]: item for item in segment_qc}
        results: list[dict[str, Any]] = []
        runs = self.root / "runs"
        if not runs.is_dir():
            return results
        bootstrap_ids = [item["segment_id"] for item in segment_indexes if item["segment_kind"] == "bootstrap"]
        for run_dir in sorted((p for p in runs.iterdir() if p.is_dir() and not p.is_symlink()), key=lambda p: p.name):
            relative = run_dir.relative_to(self.root).as_posix()
            findings: list[dict[str, Any]] = []
            manifest_path = run_dir / "manifest.json"
            manifest: dict[str, Any] = {}
            if not manifest_path.exists():
                findings.append(finding("run_manifest_missing", "fatal", "run_manifest", relative, "run manifest缺失", blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            else:
                try:
                    manifest = _read_json(manifest_path, self.root, self.limits["json"], "run manifest")
                    _validate_schema_record(self.recorder_contract["run_manifest_schema"], manifest, "run manifest")
                except DatasetIndexError as exc:
                    findings.append(finding("run_manifest_invalid", "fatal", "run_manifest", relative, str(exc), blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            run_id = manifest.get("run_id", run_dir.name)
            if run_id != run_dir.name:
                findings.append(finding("run_identity_mismatch", "fatal", "run_manifest", relative, "run_id与目录名不一致", blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            segment_prefix = f"runs/{run_dir.name}/segments/"
            segments = sorted((item for item in segment_indexes if item["source_segment_path"].startswith(segment_prefix)), key=lambda item: (item["source_segment_path"], item["segment_id"]))
            manifest_ids = manifest.get("recorder_segment_ids", [])
            if not isinstance(manifest_ids, list) or any(not isinstance(item, str) or not item for item in manifest_ids):
                findings.append(finding("run_manifest_invalid", "fatal", "run_manifest", relative, "recorder_segment_ids必须是字符串数组", blocking_use_cases=("diagnostic", "perception", "formal_bc"))); manifest_ids = []
            if len(manifest_ids) != len(set(manifest_ids)):
                findings.append(finding("duplicate_segment_id", "fatal", "run_manifest", relative, "manifest包含重复Segment ID", blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            actual_ids = [item["segment_id"] for item in segments]
            if len(actual_ids) != len(set(actual_ids)):
                findings.append(finding("duplicate_segment_id", "fatal", "segment_manifest", relative, "物理Segment包含重复ID", blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            missing = [item for item in manifest_ids if item not in actual_ids]
            unlisted = [item for item in actual_ids if item not in manifest_ids]
            if missing: findings.append(finding("run_manifest_segment_missing", "fatal", "run_manifest", relative, "manifest引用不存在Segment", evidence={"segment_ids": missing}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            if unlisted: findings.append(finding("run_manifest_unlisted_segment", "fatal", "run_manifest", relative, "物理Segment未列入manifest", evidence={"segment_ids": unlisted}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            sequences = [item.get("segment_sequence") for item in segments]
            if any(type(value) is not int for value in sequences) or len(sequences) != len(set(sequences)) or sorted(sequences) != list(range(len(sequences))):
                findings.append(finding("segment_sequence_invalid", "fatal", "segment_manifest", relative, "Segment sequence必须唯一连续且从0开始", evidence={"sequences": sequences}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            physical_order = [item["segment_id"] for item in sorted(segments, key=lambda value: value.get("segment_sequence") if type(value.get("segment_sequence")) is int else -1)]
            if not missing and not unlisted and manifest_ids != physical_order:
                findings.append(finding("run_manifest_segment_order_mismatch", "fatal", "run_manifest", relative, "manifest Segment顺序不一致", blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            for item in segments:
                if item["parent_run_id"] != run_id:
                    findings.append(finding("segment_parent_run_id_mismatch", "fatal", "segment_manifest", item["source_segment_path"], "Segment parent_run_id混写", blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            context_run_ids = sorted({value for item in segments for value in item["competition_context_summary"]["run_ids"]})
            if context_run_ids and context_run_ids != [run_id]:
                findings.append(finding("run_id_mixed", "fatal", "jsonl", relative, "Run内部CompetitionContext身份混写", evidence={"run_ids": context_run_ids}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            context_fingerprints = sorted({value for item in segments for value in item["competition_context_summary"]["task_set_fingerprints"]})
            if context_fingerprints and context_fingerprints != [manifest.get("task_set_fingerprint")]:
                findings.append(finding("task_set_fingerprint_mismatch", "fatal", "jsonl", relative, "Context task_set_fingerprint与Run manifest不一致", evidence={"context": context_fingerprints, "manifest": manifest.get("task_set_fingerprint")}, blocking_use_cases=("diagnostic", "perception", "formal_bc")))
            if manifest.get("recovery_required") is True:
                findings.append(finding("recovery_required_by_manifest", "error", "run_manifest", relative, "Run manifest要求Recovery", blocking_use_cases=("perception", "formal_bc")))
            complete = type(manifest.get("end_ros_ns")) is int and manifest.get("end_ros_ns", -1) >= 0 and isinstance(manifest.get("end_wall_utc"), str) and bool(manifest.get("end_wall_utc")) and manifest.get("clean_shutdown") is True and isinstance(manifest.get("shutdown_reason"), str) and bool(manifest.get("shutdown_reason")) and manifest.get("recovery_required") is False
            incomplete = [] if complete else ["run_manifest_end_fields_incomplete"]
            if not complete:
                findings.append(finding("run_end_incomplete", "warning", "run_manifest", relative, "Run未记录完整结束字段"))
            recovery = [path for path, report in recovery_reports if report.get("source_parent_run_id") == run_id or any(item["source_segment_path"] == report.get("source_segment_path") for item in segments)]
            if recovery:
                findings.append(finding("recovery_finding_present", "error", "recovery", relative, "Run关联Recovery finding", evidence={"reports": recovery}, blocking_use_cases=("perception", "formal_bc")))
            segment_values = [item["eligibility"] for item in segments]
            eligibility = aggregate_eligibility(segment_values, run_complete=complete, findings=findings)
            results.append({
                "run_id": run_id, "task_set_fingerprint": manifest.get("task_set_fingerprint"),
                "run_manifest_sha256": _sha_file(manifest_path) if manifest_path.is_file() and not manifest_path.is_symlink() else None,
                "segments": [item["segment_id"] for item in segments], "bootstrap_association": {"available_segment_ids": bootstrap_ids},
                "run_end": {key: manifest.get(key) for key in ("end_ros_ns", "end_wall_utc", "clean_shutdown", "shutdown_reason", "recovery_required")},
                "task_attempt_summary": {"task_set_fingerprint": manifest.get("task_set_fingerprint"), "evaluation": "context_summary_only"},
                "recovery": {"reports": recovery}, "findings": findings,
                "segment_findings": {item["source_segment_path"]: qc_by_path[item["source_segment_path"]]["findings"] for item in segments},
                "eligibility": eligibility, "incomplete_run_reasons": incomplete,
                "source_artifacts": [relative + "/manifest.json", *[item["source_segment_path"] + "/segment.json" for item in segments]],
            })
        return results

    def build(self, *, check_only: bool = False) -> IndexBuildResult:
        before = _snapshot_raw(self.root, self.limits)
        dependencies = {
            "interface": {"path": "config/contracts/interface_v1.json", "sha256": _sha_file(interface_contract_path())},
            "recorder": {"path": "config/contracts/recorder_schema_v1.json", "sha256": _sha_file(recorder_contract_path())},
            "data_tf_policy": {"path": "config/contracts/data_tf_policy_v1.json", "sha256": _sha_file(data_tf_policy_contract_path())},
            "dataset_index": {"path": "config/contracts/dataset_index_v1.json", "sha256": dataset_index_contract_sha256()},
        }
        source_identity = [{"relative_path": item["relative_path"], "sha256": item["sha256"]} for item in before]
        build_material = {"schema_name": "team_sorting.dataset_index", "schema_version": 1, "indexer_name": INDEXER_NAME, "indexer_version": INDEXER_VERSION, "implementation_identity": IMPLEMENTATION_IDENTITY, "qc_config_sha256": self.qc_sha, "source_artifacts": source_identity, "dependency_contracts": dependencies}
        build_id = _sha_bytes(_canonical(build_material))
        indexes: list[dict[str, Any]] = []
        segment_qc: list[dict[str, Any]] = []
        for segment in self._segments():
            index, qc = self._segment(segment, build_id)
            indexes.append(index)
            segment_qc.append(qc)
        recovery_reports = self._recovery_reports(indexes)
        qc_by_path = {item["source_segment_path"]: item for item in segment_qc}
        index_by_path = {item["source_segment_path"]: item for item in indexes}
        for report_path, report in recovery_reports:
            source_path = report.get("source_segment_path")
            if source_path is None:
                continue
            item = finding("recovery_finding_present", "error", "recovery", report_path, "Segment关联Recovery finding", evidence={"report": report_path}, blocking_use_cases=("perception", "formal_bc"))
            qc_by_path[source_path]["findings"].append(item)
            for use_case in item["blocking_use_cases"]:
                qc_by_path[source_path]["eligibility"][use_case] = "ineligible"
                index_by_path[source_path]["eligibility"][use_case] = "ineligible"
        runs = self._runs(indexes, segment_qc, recovery_reports)
        after_analysis = _snapshot_raw(self.root, self.limits)
        if before != after_analysis:
            raise DatasetIndexError("raw树在只读分析期间发生变化")
        finding_count = sum(len(item["findings"]) for item in segment_qc) + sum(len(item["findings"]) for item in runs)
        if check_only:
            return IndexBuildResult(build_id, None, False, len(indexes), len(runs), finding_count, True)
        dataset_bytes = b"".join(_canonical(item) + b"\n" for item in indexes)
        segment_bytes = _canonical({"schema_name": "team_sorting.dataset_index.segment_qc", "schema_version": 1, "build_id": build_id, "segments": segment_qc}) + b"\n"
        run_bytes = _canonical({"schema_name": "team_sorting.dataset_index.run_qc", "schema_version": 1, "build_id": build_id, "runs": runs}) + b"\n"
        payloads = {"dataset_index.jsonl": dataset_bytes, "segment_qc.json": segment_bytes, "run_qc.json": run_bytes}
        expected_manifest = {
            "schema_name": "team_sorting.dataset_index.build", "schema_version": 1, "build_id": build_id,
            "build_status": "complete", "indexer_name": INDEXER_NAME, "indexer_version": INDEXER_VERSION,
            "implementation_identity": IMPLEMENTATION_IDENTITY,
            "source_dataset_identity": {"dataset_directory_name": self.root.name}, "source_artifact_count": len(before),
            "source_artifacts": source_identity, "source_root_fingerprint": _sha_bytes(_canonical(source_identity)),
            "qc_config_sha256": self.qc_sha, "dependency_contracts": dependencies,
            "outputs": {name: {"sha256": _sha_bytes(data), "size": len(data)} for name, data in payloads.items()},
            "raw_immutability_verified": True, "reused_existing_build": False,
            "determinism": {"semantic": True, "byte_identical_new_build": False, "generated_at_utc_excluded_from_build_id": True},
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.output_root.is_symlink():
            raise DatasetIndexError("derived/indexer_v1不得是symlink")
        final = self.output_root / build_id
        if final.exists():
            self._validate_existing(final, expected_manifest)
            if before != _snapshot_raw(self.root, self.limits):
                raise DatasetIndexError("raw树在既有build复用校验期间发生变化")
            return IndexBuildResult(build_id, final, True, len(indexes), len(runs), finding_count, True)
        temporary = Path(tempfile.mkdtemp(prefix=".index-build-", dir=self.output_root))
        published_new_build = False
        try:
            for name, data in payloads.items():
                with (temporary / name).open("xb") as stream:
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())
            index_build = {**expected_manifest, "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
            with (temporary / "index_build.json").open("xb") as stream:
                stream.write(_canonical(index_build) + b"\n"); stream.flush(); os.fsync(stream.fileno())
            descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if before != _snapshot_raw(self.root, self.limits):
                raise DatasetIndexError("raw树在原子发布前发生变化")
            os.rename(temporary, final)
            published_new_build = True
            parent_fd = os.open(self.output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            after_publish = _snapshot_raw(self.root, self.limits)
            if before != after_publish:
                raise DatasetIndexError("raw树在derived发布期间发生变化")
        except Exception as primary:
            try:
                if published_new_build:
                    self._remove_published(final, build_id)
                else:
                    self._remove_temporary(temporary)
            except Exception as cleanup_exc:
                raise DatasetIndexError(f"build失败：{primary}; cleanup失败：{cleanup_exc}") from primary
            raise
        return IndexBuildResult(build_id, final, False, len(indexes), len(runs), finding_count, True)

    def _validate_existing(self, directory: Path, expected_manifest: Mapping[str, Any]) -> None:
        if directory.is_symlink() or not directory.is_dir():
            raise DatasetIndexError("既有build路径不安全")
        expected_names = {"index_build.json", *OUTPUT_NAMES}
        actual_names: set[str] = set()
        for entry in os.scandir(directory):
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise DatasetIndexError(f"既有build包含非普通文件：{entry.name}")
            actual_names.add(entry.name)
        if actual_names != expected_names:
            raise DatasetIndexError(f"既有build文件集合不匹配：{sorted(actual_names)}")
        manifest = _read_json(directory / "index_build.json", self.root, self.limits["json"], "index build")
        generated = manifest.get("generated_at_utc")
        if not isinstance(generated, str) or not generated:
            raise DatasetIndexError("既有build generated_at_utc非法")
        comparable = {key: value for key, value in manifest.items() if key != "generated_at_utc"}
        if comparable != dict(expected_manifest):
            raise DatasetIndexError("既有build provenance与当前运行材料不一致")
        outputs = manifest["outputs"]
        for name in OUTPUT_NAMES:
            path = directory / name
            expected = outputs[name]
            safe = _safe_regular(path, self.root, self.limits["jsonl"] if name.endswith("jsonl") else self.limits["json"], "existing output")
            if _sha_file(safe) != expected.get("sha256") or safe.stat().st_size != expected.get("size"):
                raise DatasetIndexError(f"既有build内容不一致，拒绝覆盖：{name}")

    def _remove_temporary(self, directory: Path) -> None:
        if directory.parent != self.output_root or not directory.name.startswith(".index-build-") or directory.is_symlink():
            raise DatasetIndexError("拒绝清理未经验证的临时目录")
        if not directory.exists():
            return
        for current, directories, files in os.walk(directory, topdown=False, followlinks=False):
            for name in files:
                path = Path(current) / name
                if path.is_symlink():
                    raise DatasetIndexError("临时目录出现symlink，拒绝清理")
                path.unlink()
            for name in directories:
                (Path(current) / name).rmdir()
        directory.rmdir()

    def _remove_published(self, directory: Path, build_id: str) -> None:
        if directory.parent != self.output_root or directory.name != build_id or directory.is_symlink() or not directory.is_dir():
            raise DatasetIndexError("拒绝清理未经验证的已发布目录")
        manifest = directory / "index_build.json"
        if not manifest.is_file() or _read_json(manifest, self.root, self.limits["json"], "index build").get("build_id") != build_id:
            raise DatasetIndexError("已发布目录身份校验失败")
        for current, directories, files in os.walk(directory, topdown=False, followlinks=False):
            for name in files:
                p = Path(current) / name
                if p.is_symlink(): raise DatasetIndexError("published output symlink")
                p.unlink()
            for name in directories:
                p = Path(current) / name
                if p.is_symlink(): raise DatasetIndexError("published subdirectory symlink")
                p.rmdir()
        directory.rmdir()


def _load_qc_config(path: str | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    selected = Path(path)
    try:
        metadata = os.lstat(selected)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise DatasetIndexError("QC配置必须是非symlink普通文件")
        if metadata.st_size > 67_108_864:
            raise DatasetIndexError("QC配置超过64 MiB上限")
        value = json.loads(selected.read_text(encoding="utf-8"), object_pairs_hook=_strict_pairs, parse_constant=_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetIndexError(f"QC配置必须是严格JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise DatasetIndexError("QC配置顶层必须是object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=INDEXER_NAME, description="Read-only Dataset Index/QC v1")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--qc-config")
    parser.add_argument("--output-root")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json-summary", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = DatasetIndexer(args.dataset_root, _load_qc_config(args.qc_config), args.output_root).build(check_only=args.check_only)
    except (DatasetIndexError, ValueError, TypeError, OSError) as exc:
        if args.json_summary:
            print(json.dumps({"status": "indexer_failed", "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        else:
            print(f"indexer failed: {exc}", file=os.sys.stderr)
        return 2
    summary = result.to_dict()
    summary["status"] = "checked" if args.check_only else "reused" if result.reused else "built"
    if args.json_summary:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"{summary['status']}: build_id={result.build_id} findings={result.finding_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
