"""Recorder schema v1 B2 runtime lifecycle pure-Python regressions."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from team_sorting.competition_context import CompetitionContext
from team_sorting.action_mux import ActionMux
from team_sorting.interfaces import (
    ActionDispatchRecord,
    BaseCommand,
    DispatchGroupRecord,
    DispatchMode,
    Float64MultiArrayExactPayload,
    FSMStatus,
    FinalAction,
    GlobalPhase,
    LocalPhase,
    RobotJointState,
    TaskSpec,
    TwistExactPayload,
    action_dispatch_to_json,
    final_action_to_json,
)
from team_sorting.recorder import EpisodeRecorder
from team_sorting.recorder_contract import load_recorder_contract
from team_sorting.interface_contract import interface_contract_path
from team_sorting.recorder_contract import recorder_contract_sha256
from team_sorting.data_tf_policy_contract import load_data_tf_policy_contract
from team_sorting.recorder_runtime import (
    RecorderRuntimeConfig,
    RecorderRuntimeManager,
    RecorderRuntimeState,
    _safe_component,
    append_jsonl,
    atomic_write_json,
    create_marker,
    resolve_rosbag_qos_overrides_path,
    validate_rosbag_qos_overrides_path,
)
from team_sorting.recording_contracts import ActionPairingConfig
from team_sorting.ros_nodes import (
    _build_action_dispatch_record,
    _create_recorder_node,
    _official_publish_enabled,
    _validated_control_config,
)


ROOT = Path(__file__).resolve().parents[1]
QOS_PATH = ROOT / "config" / "rosbag_qos_overrides.yaml"
DEFAULT_ROSBAG_TOPICS = tuple(
    yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))[
        "recorder"
    ]["rosbag_topics"]
)


class _RecorderNodeBase:
    def __init__(self, name: str) -> None:
        self.enabled = False

    def declare_parameter(self, name: str, value: object) -> None:
        self.enabled = value

    def get_parameter(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(value=self.enabled)

    def get_clock(self) -> SimpleNamespace:
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1))

    def create_subscription(self, *args: Any) -> object:
        return object()

    def create_timer(self, *args: Any) -> object:
        return SimpleNamespace(cancel=lambda: None)

    def get_logger(self) -> SimpleNamespace:
        return SimpleNamespace(error=lambda message: None, warning=lambda message: None)

    def destroy_node(self) -> None:
        return None


def _recorder_node_config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    config["recorder"]["enabled"] = True
    config["recorder"]["root_dir"] = str(root)
    return config


def _recorder_ros() -> SimpleNamespace:
    return SimpleNamespace(Node=_RecorderNodeBase, String=object, Int32=object)


def _pairing(enabled: bool = False) -> ActionPairingConfig:
    return ActionPairingConfig(
        enabled=enabled,
        max_pending_per_side=8,
        max_completed_sequences=8,
        max_wait_ns=100,
        prune_period_sec=0.5,
        raw_payload_preview_chars=64,
    )


def _config(root: Path, **overrides: Any) -> RecorderRuntimeConfig:
    values: dict[str, Any] = {
        "root_dir": root,
        "record_rosbag": False,
        "rosbag_topics": DEFAULT_ROSBAG_TOPICS,
        "recovery_scan_enabled": True,
        "bag_sigint_timeout_sec": 1.0,
        "bag_terminate_timeout_sec": 1.0,
        "bag_kill_timeout_sec": 1.0,
        "observe_only": True,
        "official_publish_enabled": False,
        "rosbag_qos_overrides_path": QOS_PATH,
    }
    values.update(overrides)
    return RecorderRuntimeConfig(**values)


def _task(task_id: int = 1, timestamp_ns: int = 1) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        instruction=f"task-{task_id}",
        target_kind="box",
        target_body=f"box_{task_id}",
        target_color="pink",
        place_type="shelf_point",
        place_world_xyz=(1.0, 2.0, 3.0),
        place_frame_id="world",
        place_radius=0.2,
        timestamp_ns=timestamp_ns,
    )


def _context(
    run_id: str = "run-a",
    *,
    task_id: int = 1,
    attempt: int = 0,
    finished: bool = False,
    valid: bool = True,
    timestamp_ns: int = 10,
    fingerprint: str = "fingerprint-a",
) -> CompetitionContext:
    return CompetitionContext(
        schema_name="team_sorting.competition_context",
        schema_version=1,
        run_id=run_id,
        task_set_fingerprint=fingerprint if valid else "",
        current_task_id=None if finished else task_id,
        current_attempt_count=attempt,
        elapsed_sim_s=1.0,
        score=0,
        best_scores=(0, 0, 0),
        current_step="-",
        finished=finished,
        active_task=None if finished else _task(task_id, timestamp_ns),
        instruction_timestamp_ns=timestamp_ns,
        referee_timestamp_ns=timestamp_ns,
        valid=valid,
        failure_reason="" if valid else "referee_unavailable",
    )


def _manager(root: Path, **config_overrides: Any) -> RecorderRuntimeManager:
    return RecorderRuntimeManager(
        _config(root, **config_overrides),
        _pairing(),
        wall_now=lambda: "2026-01-02T03:04:05Z",
        monotonic_now_ns=iter(range(1000, 100000)).__next__,
        pid=123,
        environment={},
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _context_run_ids(segment: Path) -> set[str]:
    path = segment / "competition_contexts.jsonl"
    if not path.exists():
        return set()
    return {
        json.loads(line)["run_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
    }


def _create_closed_unfinished_run(root: Path) -> tuple[Path, Path]:
    manager = _manager(root)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    segment = manager.current_segment_dir
    manager.close(20)
    assert segment is not None
    return segment, root / "runs" / "run-a" / "events.jsonl"


def _action_and_dispatch(
    *, publish_attempted: bool, publish_succeeded: bool | None
) -> tuple[FinalAction, ActionDispatchRecord]:
    now = 100
    joints = RobotJointState((0.0,) * 17, (0.0,) * 17, (0.0,) * 17, now)
    status = FSMStatus(
        1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", now
    )
    action, decision = ActionMux().compose_with_decision(
        BaseCommand(0.0, 0.0, now, now + 10), None, joints, status, now
    )
    specs = (
        ("base", "/cmd_vel", "geometry_msgs/msg/Twist", 0, 2),
        ("spine", "/spine_forward_position_controller/commands", "std_msgs/msg/Float64MultiArray", 2, 3),
        ("head", "/head_forward_position_controller/commands", "std_msgs/msg/Float64MultiArray", 3, 5),
        ("left_arm", "/left_arm_forward_position_controller/commands", "std_msgs/msg/Float64MultiArray", 5, 12),
        ("right_arm", "/right_arm_forward_position_controller/commands", "std_msgs/msg/Float64MultiArray", 12, 19),
    )
    records: list[DispatchGroupRecord] = []
    for group, topic, message_type, start, stop in specs:
        attempted = publish_attempted and (
            publish_succeeded is True or group == "base"
        )
        payload = None
        if attempted:
            payload = (
                TwistExactPayload(
                    (action.values[0], 0.0, 0.0),
                    (0.0, 0.0, action.values[1]),
                )
                if group == "base"
                else Float64MultiArrayExactPayload(action.values[start:stop])
            )
        records.append(
            DispatchGroupRecord(
                group,
                topic,
                message_type,
                attempted,
                publish_succeeded if attempted else None,
                payload,
                "" if publish_succeeded is not False or not attempted else "publisher_failure",
            )
        )
    groups = tuple(records)
    dispatch = _build_action_dispatch_record(
        action,
        decision,
        publish_enabled=publish_attempted,
        publisher_created=publish_attempted,
        dispatch_mode=DispatchMode.FULL if publish_attempted else DispatchMode.NONE,
        group_records=groups,
        failure_reason=(
            "" if publish_succeeded is True else "publisher_failure" if publish_attempted else "observe_only"
        ),
    )
    return action, dispatch


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("record_rosbag", 1),
        ("record_rosbag", "false"),
        ("recovery_scan_enabled", 0),
        ("observe_only", 1),
        ("official_publish_enabled", None),
        ("bag_sigint_timeout_sec", 0),
        ("bag_sigint_timeout_sec", -1),
        ("bag_sigint_timeout_sec", True),
        ("bag_terminate_timeout_sec", 0.0),
        ("bag_terminate_timeout_sec", "1"),
        ("bag_kill_timeout_sec", -0.5),
        ("bag_kill_timeout_sec", False),
        ("bag_startup_timeout_sec", 0),
        ("bag_startup_timeout_sec", True),
        ("bag_startup_poll_interval_sec", 0.0),
        ("bag_startup_poll_interval_sec", "0.1"),
    ],
)
def test_runtime_config_fails_closed(field: str, value: object, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _config(tmp_path, **{field: value})


def test_runtime_config_rejects_observe_only_with_effective_publish_enabled(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="effective official publish gate"):
        _config(tmp_path, observe_only=True, official_publish_enabled=True)


def test_default_recorder_topics_include_unique_tf_topics_matching_policy() -> None:
    assert len(DEFAULT_ROSBAG_TOPICS) == len(set(DEFAULT_ROSBAG_TOPICS)) == 15
    assert DEFAULT_ROSBAG_TOPICS[-2:] == ("/tf", "/tf_static")
    policy = load_data_tf_policy_contract()
    entries = (
        *policy["topic_policy"]["current_raw_baseline"],
        *policy["topic_policy"]["b3_target_topics"],
    )
    types = {entry["name"]: entry["message_type"] for entry in entries}
    assert types["/tf"] == "tf2_msgs/msg/TFMessage"
    assert types["/tf_static"] == "tf2_msgs/msg/TFMessage"


def test_source_qos_resource_resolution_is_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = resolve_rosbag_qos_overrides_path(
        ROOT / "config" / "config.yaml", "rosbag_qos_overrides.yaml"
    )
    assert resolved == QOS_PATH.resolve()


def test_installed_prefix_qos_resource_resolution(tmp_path: Path) -> None:
    config_dir = tmp_path / "prefix/local/share/team_sorting/config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text("recorder: {}\n", encoding="utf-8")
    qos_path = config_dir / "rosbag_qos_overrides.yaml"
    qos_path.write_bytes(QOS_PATH.read_bytes())
    assert resolve_rosbag_qos_overrides_path(
        config_path, "rosbag_qos_overrides.yaml"
    ) == qos_path.resolve()


def test_setup_installs_rosbag_qos_resource() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"config/rosbag_qos_overrides.yaml"' in setup_text
    assert setup_text.count('"share/" + package_name + "/config"') == 1


def test_recording_requires_explicit_qos_resource(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="必须显式提供"):
        _config(tmp_path, record_rosbag=True, rosbag_qos_overrides_path=None)


@pytest.mark.parametrize("kind", ["missing", "empty", "directory", "symlink"])
def test_qos_resource_unsafe_or_missing_file_fails_closed(
    kind: str, tmp_path: Path
) -> None:
    path = tmp_path / "qos.yaml"
    if kind == "empty":
        path.write_bytes(b"")
    elif kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.yaml"
        target.write_bytes(QOS_PATH.read_bytes())
        path.symlink_to(target)
    with pytest.raises(ValueError):
        validate_rosbag_qos_overrides_path(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda p: p["/tf"].__setitem__("reliability", "reliable"), "/tf.reliability"),
        (lambda p: p["/tf_static"].__setitem__("durability", "volatile"), "/tf_static.durability"),
        (lambda p: p["/tf"].__setitem__("history", "keep_all"), "/tf.history"),
        (lambda p: p["/tf"].__setitem__("depth", True), "/tf.depth"),
        (lambda p: p.__setitem__("/extra", {}), "只能定义"),
    ],
)
def test_qos_resource_wrong_schema_or_policy_fails_closed(
    mutation: Any, match: str, tmp_path: Path
) -> None:
    payload = yaml.safe_load(QOS_PATH.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "qos.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        validate_rosbag_qos_overrides_path(path)


def test_qos_resource_malformed_yaml_and_duplicate_keys_fail_closed(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("/tf: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML非法"):
        validate_rosbag_qos_overrides_path(malformed)
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(QOS_PATH.read_text(encoding="utf-8") + "\n/tf: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复key"):
        validate_rosbag_qos_overrides_path(duplicate)


def test_qos_resource_unreadable_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "qos.yaml"
    path.write_bytes(QOS_PATH.read_bytes())
    original = Path.read_text

    def deny(candidate: Path, *args: Any, **kwargs: Any) -> str:
        if candidate == path:
            raise PermissionError("denied")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)
    with pytest.raises(ValueError, match="不可读"):
        validate_rosbag_qos_overrides_path(path)


@pytest.mark.parametrize(
    ("observe_only", "enable_official_publish", "expected"),
    [
        (True, True, False),
        (False, False, False),
        (False, True, True),
    ],
)
def test_run_manifest_records_effective_official_publish_gate(
    observe_only: bool,
    enable_official_publish: bool,
    expected: bool,
    tmp_path: Path,
) -> None:
    source = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    source["control"].update(
        {
            "observe_only": observe_only,
            "enable_official_publish": enable_official_publish,
            "simulation_only": True,
        }
    )
    control = _validated_control_config(source)
    effective = _official_publish_enabled(control)
    assert effective is expected
    manager = RecorderRuntimeManager(
        _config(
            tmp_path,
            observe_only=observe_only,
            official_publish_enabled=effective,
        ),
        _pairing(),
    )
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert manifest["official_publish_enabled"] is expected


def test_default_run_manifest_records_official_publish_disabled(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert manifest["observe_only"] is True
    assert manifest["official_publish_enabled"] is False


@pytest.mark.parametrize(
    ("observe_only", "enable_official_publish", "expected"),
    [(True, True, False), (False, False, False), (False, True, True)],
)
def test_recorder_node_uses_team_client_effective_publish_gate(
    observe_only: bool,
    enable_official_publish: bool,
    expected: bool,
    tmp_path: Path,
) -> None:
    class Node:
        def __init__(self, name: str) -> None:
            self.enabled = False

        def declare_parameter(self, name: str, value: object) -> None:
            self.enabled = value

        def get_parameter(self, name: str) -> SimpleNamespace:
            return SimpleNamespace(value=self.enabled)

        def get_clock(self) -> SimpleNamespace:
            return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1))

        def create_subscription(self, *args: Any) -> object:
            return object()

        def create_timer(self, *args: Any) -> object:
            return SimpleNamespace(cancel=lambda: None)

        def get_logger(self) -> SimpleNamespace:
            return SimpleNamespace(error=lambda message: None, warning=lambda message: None)

        def destroy_node(self) -> None:
            return None

    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    config["recorder"]["enabled"] = True
    config["recorder"]["record_rosbag"] = False
    config["recorder"]["root_dir"] = str(tmp_path)
    config["control"].update(
        {
            "observe_only": observe_only,
            "enable_official_publish": enable_official_publish,
            "simulation_only": True,
        }
    )
    ros = SimpleNamespace(Node=Node, String=object, Int32=object)
    node = _create_recorder_node(ros)(config, ros)
    assert node._runtime.config.official_publish_enabled is expected
    node._runtime.record_competition_context(_context(), 10, 11)
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert manifest["official_publish_enabled"] is expected
    node._runtime.close(20)


def test_recorder_node_without_rosbag_does_not_require_rosbag_section(
    tmp_path: Path,
) -> None:
    config = _recorder_node_config(tmp_path)
    config["recorder"]["record_rosbag"] = False
    config["recorder"].pop("rosbag")
    ros = _recorder_ros()

    node = _create_recorder_node(ros)(config, ros)

    assert node._runtime.config.record_rosbag is False
    assert node._runtime.config.rosbag_qos_overrides_path is None
    node.destroy_node()


def test_recorder_node_without_rosbag_never_resolves_missing_qos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _recorder_node_config(tmp_path)
    config["recorder"]["record_rosbag"] = False
    config["recorder"]["rosbag"]["qos_overrides_path"] = "missing.yaml"

    def forbidden(*args: Any, **kwargs: Any) -> Path:
        raise AssertionError("record_rosbag=false不得解析QoS文件")

    monkeypatch.setattr("team_sorting.ros_nodes.resolve_rosbag_qos_overrides_path", forbidden)
    ros = _recorder_ros()
    node = _create_recorder_node(ros)(config, ros)
    assert node._runtime.config.rosbag_qos_overrides_path is None
    node.destroy_node()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda recorder: recorder.pop("rosbag"), "recorder.rosbag配置必须是映射"),
        (lambda recorder: recorder["rosbag"].pop("qos_overrides_path"), "必须是字符串"),
        (
            lambda recorder: recorder["rosbag"].__setitem__(
                "qos_overrides_path", "missing.yaml"
            ),
            "缺失或不可访问",
        ),
    ],
)
def test_recorder_node_with_rosbag_rejects_missing_qos_configuration(
    mutation: Any, match: str, tmp_path: Path
) -> None:
    config = _recorder_node_config(tmp_path)
    config["recorder"]["record_rosbag"] = True
    mutation(config["recorder"])
    ros = _recorder_ros()
    with pytest.raises((RuntimeError, ValueError), match=match):
        _create_recorder_node(ros)(config, ros)


def test_recorder_node_resolves_qos_beside_copied_active_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "installed/share/team_sorting/config"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.yaml"
    qos_path = config_dir / "rosbag_qos_overrides.yaml"
    config_path.write_bytes((ROOT / "config" / "config.yaml").read_bytes())
    qos_path.write_bytes(QOS_PATH.read_bytes())
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["recorder"]["enabled"] = True
    config["recorder"]["root_dir"] = str(tmp_path / "dataset")
    captured: list[RecorderRuntimeConfig] = []

    class Runtime:
        def __init__(self, runtime_config: RecorderRuntimeConfig, *args: Any, **kwargs: Any) -> None:
            self.config = runtime_config
            captured.append(runtime_config)

        def start(self, now_ros_ns: int) -> Path:
            return tmp_path / "segment"

        def close(self, now_ros_ns: int, reason: str = "node_shutdown") -> None:
            return None

    monkeypatch.setenv("TEAM_SORTING_CONFIG", str(config_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("team_sorting.ros_nodes.RecorderRuntimeManager", Runtime)
    ros = _recorder_ros()
    node = _create_recorder_node(ros)(config, ros)
    assert captured[0].record_rosbag is True
    assert captured[0].rosbag_qos_overrides_path == qos_path.resolve()
    assert captured[0].config_path == config_path
    node.destroy_node()


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "a/b", "a\\b", "a\x00b", "space id", "中文", "a:b", "a?b"],
)
def test_unsafe_path_components_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        _safe_component(value, "run_id")


@pytest.mark.parametrize(
    "value", ["run", "run-1", "run_1", "run.1", "A9", "0", "a-b_c.d"]
)
def test_safe_path_components_are_accepted(value: str) -> None:
    assert _safe_component(value, "run_id") == value


def test_start_creates_bootstrap_layout_and_active_marker(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    segment = manager.start(10)
    assert manager.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert segment.parent == tmp_path / "bootstrap"
    assert (segment / "ACTIVE").is_file()
    assert not (segment / "COMPLETE").exists()
    assert (tmp_path / "runs").is_dir()
    assert (tmp_path / "recovery").is_dir()
    data = _json(segment / "segment.json")
    assert data["segment_kind"] == "bootstrap"
    assert data["parent_run_id"] is None


def test_clean_close_writes_complete_before_removing_active(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    segment = manager.start(10)
    manager.close(20)
    assert manager.state is RecorderRuntimeState.CLOSED
    assert (segment / "COMPLETE").is_file()
    assert not (segment / "ACTIVE").exists()
    data = _json(segment / "segment.json")
    assert data["marker_state"] == "complete"
    assert data["clean_shutdown"] is True
    assert data["shutdown_reason"] == "node_shutdown"
    assert data["node_end_ros_ns"] == 20
    assert data["bag_storage_identifier"]["status"] == "unavailable"
    assert data["bag_storage_identifier"]["value"] is None


def test_close_is_idempotent_and_start_is_single_use(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    with pytest.raises(RuntimeError, match="NEW"):
        manager.start(2)
    manager.close(3)
    manager.close(4)


def test_first_valid_context_seals_bootstrap_and_opens_run(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    bootstrap = manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    run_segment = manager.current_segment_dir
    assert manager.state is RecorderRuntimeState.RUN_ACTIVE
    assert run_segment is not None and run_segment.parent == tmp_path / "runs" / "run-a" / "segments"
    assert (bootstrap / "COMPLETE").is_file()
    assert not (bootstrap / "ACTIVE").exists()
    assert (run_segment / "ACTIVE").is_file()
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert manifest["recorder_segment_ids"] == [run_segment.name]
    assert manifest["end_ros_ns"] is None
    assert _context_run_ids(bootstrap) == {"run-a"}
    assert _context_run_ids(run_segment) == {"run-a"}


def test_bootstrap_history_is_not_moved_or_retroactively_bound(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    bootstrap = manager.start(1)
    manager.record_referee_message("/referee/score", 0, 2)
    manager.record_competition_context(_context(), 10, 11)
    assert bootstrap.exists()
    assert _json(bootstrap / "segment.json")["parent_run_id"] is None
    assert not (manager.current_segment_dir / "referee_messages.jsonl").exists()  # type: ignore[operator]


def test_same_run_context_does_not_roll_segment(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    segment_id = manager.current_segment_id
    manager.record_competition_context(_context(timestamp_ns=20), 20, 21)
    assert manager.current_segment_id == segment_id
    assert len(list((tmp_path / "runs" / "run-a" / "segments").iterdir())) == 1
    event_types = [
        event["event_type"]
        for event in _events(tmp_path / "runs" / "run-a" / "events.jsonl")
    ]
    assert "task_transition" not in event_types
    assert "attempt_transition" not in event_types


def test_task_and_attempt_changes_are_events_not_segment_boundaries(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    segment_id = manager.current_segment_id
    manager.record_competition_context(_context(task_id=2, attempt=1), 20, 21)
    assert manager.current_segment_id == segment_id
    event_types = [item["event_type"] for item in _events(tmp_path / "runs" / "run-a" / "events.jsonl")]
    assert "task_transition" in event_types
    assert "attempt_transition" in event_types


def test_invalid_context_does_not_replace_last_legal_context(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    manager.record_competition_context(_context(valid=False), 12, 13)
    manager.record_competition_context(_context(task_id=2), 20, 21)
    events = _events(tmp_path / "runs" / "run-a" / "events.jsonl")
    invalid = [item for item in events if item["validity"] == "invalid"]
    transition = next(item for item in events if item["event_type"] == "task_transition")
    assert invalid and invalid[-1]["invalid_reasons"] == ["referee_unavailable"]
    assert transition["payload"]["previous_task_id"] == 1


def test_same_run_fingerprint_change_is_invalid_and_does_not_transition(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    segment_id = manager.current_segment_id
    manager.record_competition_context(
        _context(task_id=2, fingerprint="conflicting"), 20, 21
    )
    assert manager.current_segment_id == segment_id
    events = _events(tmp_path / "runs" / "run-a" / "events.jsonl")
    assert events[-1]["validity"] == "invalid"
    assert events[-1]["invalid_reasons"] == [
        "task_set_fingerprint_changed_within_run"
    ]
    assert not any(event["event_type"] == "task_transition" for event in events)
    assert _context_run_ids(manager.current_segment_dir) == {"run-a"}  # type: ignore[arg-type]
    assert len(
        (manager.current_segment_dir / "competition_contexts.jsonl")  # type: ignore[operator]
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 1


def test_malformed_context_is_preserved_as_invalid_event(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context_payload("{broken", 10, 11)
    events = _events(manager.current_segment_dir / "events.jsonl")  # type: ignore[operator]
    assert events[-1]["event_type"] == "competition_context_updated"
    assert events[-1]["validity"] == "invalid"
    assert events[-1]["payload"]["raw_payload_preview"] == "{broken"
    assert len(events[-1]["payload"]["raw_payload_sha256"]) == 64


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "a\\b", "with space"])
def test_unsafe_run_id_never_creates_run_directory(run_id: str, tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(run_id=run_id), 10, 11)
    assert manager.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert list((tmp_path / "runs").iterdir()) == []


def test_unsafe_target_run_does_not_pollute_current_run_structured_context(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    run_segment = manager.current_segment_dir
    manager.record_competition_context(_context(run_id="../escape"), 20, 21)
    assert manager.current_segment_dir == run_segment
    assert _context_run_ids(run_segment) == {"run-a"}  # type: ignore[arg-type]
    event = _events(tmp_path / "runs" / "run-a" / "events.jsonl")[-1]
    assert event["validity"] == "invalid"


def test_ended_target_run_is_diagnostic_and_does_not_pollute_current_run(
    tmp_path: Path,
) -> None:
    owner = _manager(tmp_path)
    owner.start(1)
    owner.record_competition_context(
        _context(run_id="run-b", fingerprint="fingerprint-b"), 10, 11
    )
    owner.record_competition_context(
        _context(run_id="run-b", fingerprint="fingerprint-b", finished=True),
        20,
        21,
    )
    owner.close(22)

    manager = _manager(tmp_path)
    manager.start(30)
    manager.record_competition_context(_context(), 40, 41)
    run_a_segment = manager.current_segment_dir
    manager.record_competition_context(
        _context(run_id="run-b", fingerprint="fingerprint-b"), 50, 51
    )
    assert manager.current_segment_dir == run_a_segment
    assert _context_run_ids(run_a_segment) == {"run-a"}  # type: ignore[arg-type]
    event = _events(tmp_path / "runs" / "run-a" / "events.jsonl")[-1]
    assert event["validity"] == "invalid"
    assert "逻辑结束" in event["invalid_reasons"][0]


def test_run_change_finalizes_old_run_and_opens_new_run(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    old_segment = manager.current_segment_dir
    manager.record_competition_context(
        _context(run_id="run-b", fingerprint="fingerprint-b"), 20, 21
    )
    assert manager.current_segment_dir != old_segment
    assert manager.current_segment_dir.parent == tmp_path / "runs" / "run-b" / "segments"  # type: ignore[union-attr]
    old_manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert old_manifest["end_ros_ns"] == 20
    assert old_manifest["shutdown_reason"] == "run_changed"
    assert (old_segment / "COMPLETE").exists()  # type: ignore[operator]
    assert _context_run_ids(old_segment) == {"run-a"}  # type: ignore[arg-type]
    assert _context_run_ids(manager.current_segment_dir) == {"run-b"}  # type: ignore[arg-type]


def test_same_fingerprint_different_run_ids_are_not_merged(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    manager.record_competition_context(_context(run_id="run-b"), 20, 21)
    assert (tmp_path / "runs" / "run-a" / "manifest.json").exists()
    assert (tmp_path / "runs" / "run-b" / "manifest.json").exists()
    assert manager.current_segment_dir.parent == tmp_path / "runs" / "run-b" / "segments"  # type: ignore[union-attr]
    run_a_segment = next((tmp_path / "runs" / "run-a" / "segments").iterdir())
    assert _context_run_ids(run_a_segment) == {"run-a"}
    assert _context_run_ids(manager.current_segment_dir) == {"run-b"}  # type: ignore[arg-type]


def test_finished_context_finalizes_run_and_returns_to_bootstrap(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    run_segment = manager.current_segment_dir
    manager.record_competition_context(_context(finished=True), 20, 21)
    assert manager.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert manager.current_segment_dir.parent == tmp_path / "bootstrap"  # type: ignore[union-attr]
    assert (run_segment / "COMPLETE").exists()  # type: ignore[operator]
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert manifest["shutdown_reason"] == "official_finished"
    assert manifest["clean_shutdown"] is True


def test_repeated_finished_context_does_not_end_run_twice(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    manager.record_competition_context(_context(finished=True), 20, 21)
    bootstrap_id = manager.current_segment_id
    manifest_before = (tmp_path / "runs" / "run-a" / "manifest.json").read_bytes()
    manager.record_competition_context(_context(finished=True), 30, 31)
    assert manager.current_segment_id == bootstrap_id
    assert (tmp_path / "runs" / "run-a" / "manifest.json").read_bytes() == manifest_before


def test_finished_context_before_binding_does_not_create_run(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    segment_id = manager.current_segment_id
    manager.record_competition_context(_context(finished=True), 20, 21)
    assert manager.current_segment_id == segment_id
    assert list((tmp_path / "runs").iterdir()) == []


@pytest.mark.parametrize(
    "field",
    list(load_recorder_contract()["run_manifest_schema"]["fields"]),
)
def test_run_manifest_contains_every_frozen_required_field(field: str, tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    assert field in _json(tmp_path / "runs" / "run-a" / "manifest.json")


@pytest.mark.parametrize(
    "field",
    list(load_recorder_contract()["recorder_segment_schema"]["fields"]),
)
def test_segment_manifest_contains_every_frozen_required_field(field: str, tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    segment = manager.start(1)
    manager.close(2)
    assert field in _json(segment / "segment.json")


@pytest.mark.parametrize(
    "field", list(load_recorder_contract()["event_schema"]["fields"])
)
def test_event_contains_every_frozen_required_field(field: str, tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    segment = manager.start(1)
    assert field in _events(segment / "events.jsonl")[0]


def test_event_ids_are_unique_and_process_local(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    segment = manager.start(1)
    manager.record_referee_message("/referee/score", 0, 2)
    manager.close(3)
    events = _events(segment / "events.jsonl")
    ids = [event["event_id"] for event in events]
    assert len(ids) == len(set(ids))
    assert all(event["monotonic_scope"] == "process_local" for event in events)


def test_instruction_event_tracks_semantics_not_json_formatting(tmp_path: Path) -> None:
    class Parser:
        def parse(self, raw: str, timestamp_ns: int) -> tuple[TaskSpec, ...]:
            return tuple(_task(task_id, timestamp_ns) for task_id in (1, 2, 3))

    manager = _manager(tmp_path)
    segment = manager.start(1)
    manager.record_instruction("compact", 2, Parser())  # type: ignore[arg-type]
    manager.record_instruction("same semantics, different bytes", 3, Parser())  # type: ignore[arg-type]
    events = _events(segment / "events.jsonl")
    assert [event["event_type"] for event in events].count("instruction_updated") == 1


def test_final_action_event_is_written_after_compatible_jsonl(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    segment = manager.start(1)
    action, _ = _action_and_dispatch(
        publish_attempted=False, publish_succeeded=None
    )
    manager.record_final_action(action)
    assert json.loads((segment / "final_actions.jsonl").read_text(encoding="utf-8")) == json.loads(
        final_action_to_json(action)
    )
    event = _events(segment / "events.jsonl")[-1]
    assert event["event_type"] == "action_selected"
    assert event["payload"]["sequence"] == action.sequence
    assert event["payload"]["actual_publish_claimed"] is False
    assert "values" not in event["payload"]


@pytest.mark.parametrize(
    ("publish_succeeded", "expected"),
    [(True, "dispatch_succeeded"), (False, "dispatch_failed")],
)
def test_dispatch_events_preserve_local_publisher_scope(
    publish_succeeded: bool, expected: str, tmp_path: Path
) -> None:
    manager = RecorderRuntimeManager(_config(tmp_path), _pairing(True))
    segment = manager.start(1)
    _, dispatch = _action_and_dispatch(
        publish_attempted=True, publish_succeeded=publish_succeeded
    )
    manager.record_action_dispatch_payload(action_dispatch_to_json(dispatch), 101, 102)
    events = _events(segment / "events.jsonl")
    assert events[-2]["event_type"] == "dispatch_attempted"
    assert events[-1]["event_type"] == expected
    assert events[-1]["payload"]["publisher_success_scope"] == "local_call_only"
    assert events[-1]["payload"]["controller_accepted"] is None
    assert events[-1]["payload"]["execution_confirmed"] is None


def test_event_failure_does_not_rollback_raw_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RecorderRuntimeManager(_config(tmp_path), _pairing(True))
    segment = manager.start(1)
    action, _ = _action_and_dispatch(
        publish_attempted=False, publish_succeeded=None
    )
    original = append_jsonl

    def fail_events(path: Path, payload: Any, **kwargs: Any) -> None:
        if path.name == "events.jsonl":
            raise RuntimeError("injected event failure")
        original(path, payload, **kwargs)

    monkeypatch.setattr("team_sorting.recorder_runtime.append_jsonl", fail_events)
    assert manager.record_final_action_payload(final_action_to_json(action), 101, 102) == ()
    assert (segment / "final_actions.jsonl").is_file()
    assert _json(segment / "segment.json")["warning_counters"] == {}
    assert manager._active is not None  # runtime remains available to the ROS callback
    assert manager._active.data["warning_counters"]["event_write_failure"] == 1


def test_manifest_provenance_uses_injected_and_unknown_envelopes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manager = RecorderRuntimeManager(
        config,
        _pairing(),
        environment={
            "TEAM_SORTING_PROJECT_COMMIT": "abc123",
            "TEAM_SORTING_PROJECT_BRANCH": "stage2.2/contracts-recorder-v1",
            "TEAM_SORTING_DIRTY_WORKTREE": "false",
            "ROS_DOMAIN_ID": "7",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        },
    )
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert manifest["project_commit"]["value"] == "abc123"
    assert manifest["dirty_worktree"]["value"] is False
    assert manifest["ros_domain_id"]["value"] == 7
    assert manifest["official_server_image_id"]["status"] == "unavailable"
    assert "docker_image_digest" not in manifest


def test_manifest_contract_and_config_hashes_are_exact(tmp_path: Path) -> None:
    config_file = tmp_path / "runtime.yaml"
    config_file.write_bytes(b"recorder:\n  enabled: true\n")
    dataset = tmp_path / "dataset"
    manager = RecorderRuntimeManager(
        replace(_config(dataset), config_path=config_file), _pairing()
    )
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    manifest = _json(dataset / "runs" / "run-a" / "manifest.json")
    assert manifest["recorder_schema_sha256"] == recorder_contract_sha256()
    assert manifest["interface_schema_sha256"] == hashlib.sha256(
        interface_contract_path().read_bytes()
    ).hexdigest()
    assert manifest["config_sha256"]["value"] == hashlib.sha256(
        config_file.read_bytes()
    ).hexdigest()


def test_segment_parent_directory_is_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    real = __import__("team_sorting.recorder_runtime", fromlist=["_fsync_directory"])._fsync_directory

    def observe(path: Path) -> None:
        seen.append(path)
        real(path)

    monkeypatch.setattr("team_sorting.recorder_runtime._fsync_directory", observe)
    _manager(tmp_path).start(1)
    assert tmp_path / "bootstrap" in seen


def test_segment_directory_collision_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collision = tmp_path / "bootstrap" / "segment_fixed"
    collision.parent.mkdir(parents=True)
    collision.mkdir()
    sentinel = collision / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "_new_id", lambda prefix: "segment_fixed")
    with pytest.raises(RuntimeError, match="segment_directory"):
        manager.start(1)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_atomic_json_replaces_strict_utf8_object(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"value": "中文"}, artifact_type="test")
    atomic_write_json(path, {"value": 2}, artifact_type="test")
    assert _json(path) == {"value": 2}
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_atomic_json_and_jsonl_reject_nonfinite(value: float, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="artifact写入失败"):
        atomic_write_json(tmp_path / "x.json", {"value": value}, artifact_type="test")
    with pytest.raises(RuntimeError, match="artifact写入失败"):
        append_jsonl(
            tmp_path / "x.jsonl",
            {"value": value},
            artifact_type="test",
            segment_id=None,
            run_id=None,
        )


def test_marker_creation_is_exclusive(tmp_path: Path) -> None:
    marker = tmp_path / "ACTIVE"
    create_marker(marker, {"marker": "ACTIVE"}, segment_id="s", run_id=None)
    with pytest.raises(RuntimeError, match="marker"):
        create_marker(marker, {"marker": "ACTIVE"}, segment_id="s", run_id=None)


def test_precreated_episode_directory_accepts_only_lifecycle_files(tmp_path: Path) -> None:
    episode_dir = tmp_path / "segment"
    episode_dir.mkdir()
    (episode_dir / "ACTIVE").write_text("{}\n", encoding="utf-8")
    (episode_dir / "segment.json").write_text("{}\n", encoding="utf-8")
    recorder = EpisodeRecorder(tmp_path, _pairing())
    assert recorder.start(
        "segment", 1, "test boundary", precreated_lifecycle_directory=True
    ) == episode_dir
    recorder.finish(2)


@pytest.mark.parametrize("unexpected", ["metadata.json", "events.jsonl", "rosbag"])
def test_precreated_episode_directory_rejects_unexpected_content(
    unexpected: str, tmp_path: Path
) -> None:
    episode_dir = tmp_path / "segment"
    episode_dir.mkdir()
    (episode_dir / "ACTIVE").write_text("{}\n", encoding="utf-8")
    path = episode_dir / unexpected
    path.mkdir() if unexpected == "rosbag" else path.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected"):
        EpisodeRecorder(tmp_path, _pairing()).start(
            "segment", 1, "test boundary", precreated_lifecycle_directory=True
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'{"a":1}\n', []),
        (b'{"a":1}', ["jsonl_trailing_incomplete"]),
        (b'{broken\n', ["jsonl_trailing_json_invalid"]),
        (b'{broken\n{"a":1}\n', ["jsonl_middle_corruption"]),
        (b'{"a":1}\npartial', ["jsonl_trailing_incomplete"]),
        (b'\xff\n{"a":1}\n', ["jsonl_middle_corruption"]),
    ],
)
def test_jsonl_damage_classification(raw: bytes, expected: list[str], tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(raw)
    report = RecorderRuntimeManager._inspect_jsonl(path)
    assert report["issue_types"] == expected
    assert report["file_size"] == len(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'{"outer":{"items":[1,true,null,{"x":"ok"}]}}\n', []),
        (b'{"x":NaN}\n{"ok":true}\n', ["jsonl_middle_corruption"]),
        (b'{"x":NaN}\n', ["jsonl_trailing_json_invalid"]),
        (b'{"x":Infinity}\n', ["jsonl_trailing_json_invalid"]),
        (b'{"x":-Infinity}\n', ["jsonl_trailing_json_invalid"]),
        (b'{"x":1e9999}\n', ["jsonl_trailing_json_invalid"]),
        (b'{"x":1,"x":2}\n', ["jsonl_trailing_json_invalid"]),
        (b'[]\n', ["jsonl_trailing_json_invalid"]),
        (b'"text"\n', ["jsonl_trailing_json_invalid"]),
        (b'1\n', ["jsonl_trailing_json_invalid"]),
        (b'null\n', ["jsonl_trailing_json_invalid"]),
    ],
)
def test_jsonl_recovery_uses_strict_object_rules(
    raw: bytes, expected: list[str], tmp_path: Path
) -> None:
    path = tmp_path / "strict.jsonl"
    path.write_bytes(raw)
    before = path.read_bytes()
    report = RecorderRuntimeManager._inspect_jsonl(path)
    assert report["issue_types"] == expected
    assert path.read_bytes() == before


def test_run_events_nan_middle_line_is_reported_and_blocks_run(tmp_path: Path) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    raw = b'{"x":NaN}\n{"ok":true}\n'
    events.write_bytes(raw)
    recovered = _manager(tmp_path)
    recovered.start(30)
    report = next(
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_scope") == "run_shared_events"
    )
    assert "jsonl_middle_corruption" in report["issue_types"]
    assert "run-a" in recovered._blocked_run_ids
    assert events.read_bytes() == raw


def test_segment_jsonl_duplicate_key_is_reported_without_source_mutation(
    tmp_path: Path,
) -> None:
    segment, _events_path = _create_closed_unfinished_run(tmp_path)
    artifact = segment / "custom.jsonl"
    raw = b'{"x":1,"x":2}\n{"ok":true}\n'
    artifact.write_bytes(raw)
    recovered = _manager(tmp_path)
    recovered.start(30)
    report = next(
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    )
    item = next(entry for entry in report["jsonl"] if entry["path"] == "custom.jsonl")
    assert "jsonl_middle_corruption" in item["issue_types"]
    assert "run-a" in recovered._blocked_run_ids
    assert artifact.read_bytes() == raw


def test_recovery_scan_writes_report_without_mutating_source(tmp_path: Path) -> None:
    first = _manager(tmp_path)
    segment = first.start(1)
    before = {path.name: path.read_bytes() for path in segment.iterdir() if path.is_file()}
    second = _manager(tmp_path)
    new_segment = second.start(2)
    reports = list((tmp_path / "recovery").glob("*.json"))
    assert len(reports) == 1
    report = _json(reports[0])
    assert "active_without_complete" in report["issue_types"]
    assert report["recovery_action"] == "none"
    assert report["requires_manual_or_offline_recovery"] is True
    assert {path.name: path.read_bytes() for path in segment.iterdir() if path.is_file()} == before
    assert any(event["event_type"] == "unclean_shutdown_detected" for event in _events(new_segment / "events.jsonl"))


def test_completed_segment_is_not_reported_by_recovery(tmp_path: Path) -> None:
    first = _manager(tmp_path)
    first.start(1)
    first.close(2)
    second = _manager(tmp_path)
    second.start(3)
    assert list((tmp_path / "recovery").glob("*.json")) == []


@pytest.mark.parametrize(
    ("damage", "issue"),
    [
        ("tail", "jsonl_trailing_incomplete"),
        ("middle", "jsonl_middle_corruption"),
    ],
)
def test_recovery_detects_shared_run_events_damage_once(
    damage: str, issue: str, tmp_path: Path
) -> None:
    first = _manager(tmp_path)
    first.start(1)
    first.record_competition_context(_context(), 10, 11)
    first.close(20)
    events = tmp_path / "runs" / "run-a" / "events.jsonl"
    original = events.read_bytes()
    events.write_bytes(
        original + b"partial"
        if damage == "tail"
        else b"{broken\n" + original
    )

    second = _manager(tmp_path)
    second.start(30)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_scope") == "run_shared_events"
    ]
    assert len(reports) == 1
    assert reports[0]["source_parent_run_id"] == "run-a"
    assert issue in reports[0]["issue_types"]
    item = reports[0]["jsonl"][0]
    assert item["shared_append_only"] is True
    assert item["scope"] == "run_shared_append_only_events"
    bootstrap = second.current_segment_dir
    second.record_competition_context(_context(), 40, 41)
    assert second.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert _context_run_ids(bootstrap) == set()  # type: ignore[arg-type]
    assert _events(bootstrap / "events.jsonl")[-1]["validity"] == "invalid"  # type: ignore[operator]


def test_shared_run_events_checked_once_with_multiple_segments(tmp_path: Path) -> None:
    first = _manager(tmp_path)
    first.start(1)
    first.record_competition_context(_context(), 10, 11)
    first.close(20)
    second = _manager(tmp_path)
    second.start(30)
    second.record_competition_context(_context(), 40, 41)
    second.close(50)
    events = tmp_path / "runs" / "run-a" / "events.jsonl"
    events.write_bytes(events.read_bytes() + b"partial")

    third = _manager(tmp_path)
    third.start(60)
    shared_reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_scope") == "run_shared_events"
    ]
    assert len(shared_reports) == 1


def test_run_events_external_symlink_is_reported_blocked_and_never_followed(
    tmp_path: Path,
) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside_events.jsonl"
    outside.write_bytes(b'{"external":true}\n')
    original = outside.read_bytes()
    events.unlink()
    try:
        events.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support file symlinks")

    recovered = _manager(tmp_path)
    recovered.start(30)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_scope") == "run_shared_events"
    ]
    assert len(reports) == 1
    assert "run_events_symlink_rejected" in reports[0]["issue_types"]
    assert "run_events_path_escape" in reports[0]["issue_types"]
    assert reports[0]["source_artifact_hashes"] == {}
    assert "run-a" in recovered._blocked_run_ids
    bootstrap = recovered.current_segment_dir
    recovered.record_competition_context(_context(), 40, 41)
    assert recovered.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert _events(bootstrap / "events.jsonl")[-1]["validity"] == "invalid"  # type: ignore[operator]
    assert outside.read_bytes() == original


def test_live_run_event_append_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    events = tmp_path / "runs" / "run-a" / "events.jsonl"
    outside = tmp_path.parent / f"{tmp_path.name}_live_outside.jsonl"
    outside.write_bytes(b"unchanged\n")
    original = outside.read_bytes()
    events.unlink()
    try:
        events.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support file symlinks")
    with pytest.raises(RuntimeError, match="artifact写入失败"):
        manager.record_competition_context(_context(attempt=1), 20, 21)
    assert manager.state is RecorderRuntimeState.FAILED
    assert outside.read_bytes() == original


def test_existing_run_events_symlink_is_rejected_even_without_recovery_scan(
    tmp_path: Path,
) -> None:
    first_segment, events = _create_closed_unfinished_run(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_binding_outside.jsonl"
    outside.write_bytes(b"unchanged\n")
    original = outside.read_bytes()
    events.unlink()
    try:
        events.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support file symlinks")
    manager = _manager(tmp_path, recovery_scan_enabled=False)
    manager.start(30)
    bootstrap = manager.current_segment_dir
    manager.record_competition_context(_context(), 40, 41)
    assert manager.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert _events(bootstrap / "events.jsonl")[-1]["validity"] == "invalid"  # type: ignore[operator]
    assert {path.name for path in first_segment.parent.iterdir()} == {first_segment.name}
    assert outside.read_bytes() == original


def test_run_events_directory_is_reported_and_blocks_binding(tmp_path: Path) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    events.unlink()
    events.mkdir()
    recovered = _manager(tmp_path)
    recovered.start(30)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_scope") == "run_shared_events"
    ]
    assert len(reports) == 1
    assert "run_events_not_regular" in reports[0]["issue_types"]
    assert "run-a" in recovered._blocked_run_ids


def test_run_events_fifo_is_reported_without_blocking_scan_process(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("platform does not support FIFO")
    _segment, events = _create_closed_unfinished_run(tmp_path)
    events.unlink()
    try:
        os.mkfifo(events)
    except OSError:
        pytest.skip("platform cannot create FIFO")
    recovered = _manager(tmp_path)
    recovered.start(30)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_scope") == "run_shared_events"
    ]
    assert len(reports) == 1
    assert "run_events_not_regular" in reports[0]["issue_types"]
    assert "run-a" in recovered._blocked_run_ids


def test_existing_regular_run_events_remains_appendable(tmp_path: Path) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    before = events.read_bytes()
    manager = _manager(tmp_path)
    manager.start(30)
    manager.record_competition_context(_context(), 40, 41)
    assert events.is_file() and not events.is_symlink()
    assert events.read_bytes().startswith(before)
    assert len(events.read_bytes()) > len(before)


def test_recovery_role_symlink_escape_is_rejected(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    (tmp_path / "bootstrap").mkdir()
    (tmp_path / "runs").mkdir()
    try:
        (tmp_path / "recovery").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support directory symlinks")
    with pytest.raises(RuntimeError, match="不得是symlink"):
        _manager(tmp_path).start(1)
    assert list(outside.iterdir()) == []


def test_recovery_blocks_reuse_of_unclean_run_id(tmp_path: Path) -> None:
    first = _manager(tmp_path)
    first.start(1)
    first.record_competition_context(_context(), 10, 11)
    second = _manager(tmp_path)
    second.start(20)
    bootstrap = second.current_segment_dir
    second.record_competition_context(_context(), 30, 31)
    assert second.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert _context_run_ids(bootstrap) == set()  # type: ignore[arg-type]
    event = _events(bootstrap / "events.jsonl")[-1]  # type: ignore[operator]
    assert event["validity"] == "invalid"
    assert "拒绝并发" in event["invalid_reasons"][0]


def test_recovery_disabled_rejects_target_run_active_without_second_segment(
    tmp_path: Path,
) -> None:
    crashed = _manager(tmp_path)
    crashed.start(1)
    crashed.record_competition_context(_context(), 10, 11)
    manifest_path = tmp_path / "runs" / "run-a" / "manifest.json"
    manifest_before = _json(manifest_path)
    segment_names = {
        path.name for path in (tmp_path / "runs" / "run-a" / "segments").iterdir()
    }

    contender = _manager(tmp_path, recovery_scan_enabled=False)
    contender.start(20)
    bootstrap = contender.current_segment_dir
    contender.record_competition_context(_context(), 30, 31)
    assert contender.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    event = _events(bootstrap / "events.jsonl")[-1]  # type: ignore[operator]
    assert event["validity"] == "invalid"
    reason = event["invalid_reasons"][0]
    assert "target_run_id=run-a" in reason
    assert "conflicting_segment=" in reason
    assert "marker_state=active_without_complete" in reason
    assert _json(manifest_path)["recorder_segment_ids"] == manifest_before["recorder_segment_ids"]
    assert {
        path.name for path in (tmp_path / "runs" / "run-a" / "segments").iterdir()
    } == segment_names
    assert list((tmp_path / "recovery").glob("*.json")) == []


def test_recovery_disabled_allows_append_after_normal_complete_segment(
    tmp_path: Path,
) -> None:
    first_segment, _events_path = _create_closed_unfinished_run(tmp_path)
    contender = _manager(tmp_path, recovery_scan_enabled=False)
    contender.start(30)
    contender.record_competition_context(_context(), 40, 41)
    assert contender.state is RecorderRuntimeState.RUN_ACTIVE
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert manifest["recorder_segment_ids"][0] == first_segment.name
    assert len(manifest["recorder_segment_ids"]) == 2


def test_recovery_disabled_allows_binding_new_run(tmp_path: Path) -> None:
    manager = _manager(tmp_path, recovery_scan_enabled=False)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    assert manager.state is RecorderRuntimeState.RUN_ACTIVE
    assert len(_json(tmp_path / "runs" / "run-a" / "manifest.json")["recorder_segment_ids"]) == 1


def test_recovery_disabled_rejects_active_and_complete_target_segment(
    tmp_path: Path,
) -> None:
    segment, _events_path = _create_closed_unfinished_run(tmp_path)
    complete = _json(segment / "COMPLETE")
    active = {**complete, "marker": "ACTIVE", "created_wall_utc": "2026-01-02T03:04:05Z"}
    active.pop("completed_wall_utc", None)
    create_marker(segment / "ACTIVE", active, segment_id=segment.name, run_id="run-a")
    contender = _manager(tmp_path, recovery_scan_enabled=False)
    contender.start(30)
    contender.record_competition_context(_context(), 40, 41)
    assert contender.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    reason = _events(contender.current_segment_dir / "events.jsonl")[-1]["invalid_reasons"][0]  # type: ignore[operator]
    assert "marker_state=active_and_complete" in reason
    assert "ACTIVE与COMPLETE同时存在" in reason


def test_recovery_disabled_rejects_damaged_target_marker(tmp_path: Path) -> None:
    crashed = _manager(tmp_path)
    crashed.start(1)
    crashed.record_competition_context(_context(), 10, 11)
    segment = crashed.current_segment_dir
    assert segment is not None
    (segment / "ACTIVE").write_text("{broken\n", encoding="utf-8")
    contender = _manager(tmp_path, recovery_scan_enabled=False)
    contender.start(20)
    contender.record_competition_context(_context(), 30, 31)
    assert contender.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    reason = _events(contender.current_segment_dir / "events.jsonl")[-1]["invalid_reasons"][0]  # type: ignore[operator]
    assert "ACTIVE marker无法严格读取" in reason


def test_recovery_disabled_target_check_does_not_scan_other_run(tmp_path: Path) -> None:
    crashed = _manager(tmp_path)
    crashed.start(1)
    crashed.record_competition_context(_context(), 10, 11)
    contender = _manager(tmp_path, recovery_scan_enabled=False)
    contender.start(20)
    contender.record_competition_context(
        _context(run_id="run-b", fingerprint="fingerprint-b"), 30, 31
    )
    assert contender.state is RecorderRuntimeState.RUN_ACTIVE
    assert (tmp_path / "runs" / "run-b" / "manifest.json").exists()
    assert len(list((tmp_path / "runs" / "run-a" / "segments").iterdir())) == 1


@pytest.mark.parametrize(
    ("damage", "expected_issue"),
    [
        ("missing_manifest", "segment_manifest_missing"),
        ("invalid_manifest", "segment_manifest_invalid"),
        ("marker_parent_mismatch", "identity_mismatch"),
        ("manifest_parent_mismatch", "identity_mismatch"),
    ],
)
def test_recovery_uses_canonical_run_path_for_orphan_identity_and_blocking(
    damage: str,
    expected_issue: str,
    tmp_path: Path,
) -> None:
    crashed = _manager(tmp_path)
    crashed.start(1)
    crashed.record_competition_context(_context(), 10, 11)
    segment = crashed.current_segment_dir
    assert segment is not None
    if damage == "missing_manifest":
        (segment / "segment.json").unlink()
    elif damage == "invalid_manifest":
        (segment / "segment.json").write_text("{broken\n", encoding="utf-8")
    elif damage == "marker_parent_mismatch":
        marker = _json(segment / "ACTIVE")
        marker["parent_run_id"] = "run-wrong"
        atomic_write_json(segment / "ACTIVE", marker, artifact_type="test")
    else:
        manifest = _json(segment / "segment.json")
        manifest["parent_run_id"] = "run-wrong"
        atomic_write_json(segment / "segment.json", manifest, artifact_type="test")

    recovered = _manager(tmp_path)
    recovered.start(20)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    ]
    assert len(reports) == 1
    report = reports[0]
    assert report["source_parent_run_id"] == "run-a"
    assert report["parent_run_identity_source"] == "canonical_path"
    assert expected_issue in report["issue_types"]
    bootstrap = recovered.current_segment_dir
    recovered.record_competition_context(_context(), 30, 31)
    assert recovered.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    event = _events(bootstrap / "events.jsonl")[-1]  # type: ignore[operator]
    assert event["validity"] == "invalid"
    assert "拒绝并发" in event["invalid_reasons"][0]


def test_bootstrap_marker_parent_claim_is_reported_but_does_not_block_run(
    tmp_path: Path,
) -> None:
    crashed = _manager(tmp_path)
    segment = crashed.start(1)
    marker = _json(segment / "ACTIVE")
    marker["parent_run_id"] = "run-a"
    atomic_write_json(segment / "ACTIVE", marker, artifact_type="test")

    recovered = _manager(tmp_path)
    recovered.start(20)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    ]
    assert len(reports) == 1
    assert reports[0]["source_parent_run_id"] is None
    assert reports[0]["parent_run_identity_mismatch"] is True
    assert "active_marker_identity_mismatch" in reports[0]["issue_types"]
    assert "run-a" not in recovered._blocked_run_ids
    recovered.record_competition_context(_context(), 30, 31)
    assert recovered.state is RecorderRuntimeState.RUN_ACTIVE
    assert recovered.current_segment_dir is not None
    assert recovered.current_segment_dir.parent == tmp_path / "runs" / "run-a" / "segments"


def test_bootstrap_missing_segment_manifest_reports_without_blocking_run(
    tmp_path: Path,
) -> None:
    crashed = _manager(tmp_path)
    segment = crashed.start(1)
    (segment / "segment.json").unlink()
    recovered = _manager(tmp_path)
    recovered.start(20)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    ]
    assert len(reports) == 1
    assert reports[0]["source_parent_run_id"] is None
    assert "segment_manifest_missing" in reports[0]["issue_types"]
    recovered.record_competition_context(_context(), 30, 31)
    assert recovered.state is RecorderRuntimeState.RUN_ACTIVE


@pytest.mark.parametrize(
    ("damage", "expected_issue"),
    [
        ("invalid_json", "complete_marker_invalid"),
        ("wrong_marker_field", "complete_marker_invalid"),
        ("wrong_segment_id", "complete_marker_identity_mismatch"),
        ("wrong_parent_run", "complete_marker_identity_mismatch"),
    ],
)
def test_invalid_complete_marker_is_reported_and_blocks_canonical_run(
    damage: str,
    expected_issue: str,
    tmp_path: Path,
) -> None:
    segment, _events_path = _create_closed_unfinished_run(tmp_path)
    complete = segment / "COMPLETE"
    if damage == "invalid_json":
        complete.write_text("{broken\n", encoding="utf-8")
    else:
        payload = _json(complete)
        if damage == "wrong_marker_field":
            payload["marker"] = "ACTIVE"
        elif damage == "wrong_segment_id":
            payload["recorder_segment_id"] = "segment_wrong"
        else:
            payload["parent_run_id"] = "run-wrong"
        atomic_write_json(complete, payload, artifact_type="test")
    recovered = _manager(tmp_path)
    recovered.start(30)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    ]
    assert len(reports) == 1
    assert expected_issue in reports[0]["issue_types"]
    assert "run-a" in recovered._blocked_run_ids


def test_active_marker_segment_identity_mismatch_is_reported_and_blocks_run(
    tmp_path: Path,
) -> None:
    crashed = _manager(tmp_path)
    crashed.start(1)
    crashed.record_competition_context(_context(), 10, 11)
    segment = crashed.current_segment_dir
    assert segment is not None
    marker = _json(segment / "ACTIVE")
    marker["recorder_segment_id"] = "segment_wrong"
    atomic_write_json(segment / "ACTIVE", marker, artifact_type="test")
    recovered = _manager(tmp_path)
    recovered.start(30)
    report = next(
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    )
    assert "active_marker_identity_mismatch" in report["issue_types"]
    assert "marker_identity_mismatch" in report["issue_types"]
    assert "run-a" in recovered._blocked_run_ids


def test_active_and_complete_markers_together_are_explicitly_reported(
    tmp_path: Path,
) -> None:
    segment, _events_path = _create_closed_unfinished_run(tmp_path)
    complete = _json(segment / "COMPLETE")
    active = {
        **complete,
        "marker": "ACTIVE",
        "created_wall_utc": "2026-01-02T03:04:05Z",
    }
    active.pop("completed_wall_utc", None)
    create_marker(
        segment / "ACTIVE", active, segment_id=segment.name, run_id="run-a"
    )
    recovered = _manager(tmp_path)
    recovered.start(30)
    report = next(
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    )
    assert "active_and_complete_both_present" in report["issue_types"]
    assert "run-a" in recovered._blocked_run_ids


@pytest.mark.parametrize(
    ("field", "value", "expected_issue"),
    [
        ("schema_name", "bad", "segment_manifest_schema_invalid"),
        ("schema_version", 2, "segment_manifest_schema_invalid"),
        ("schema_version", True, "segment_manifest_schema_invalid"),
        ("recorder_segment_id", "segment_evil", "segment_manifest_identity_mismatch"),
        ("segment_kind", "bootstrap", "segment_manifest_kind_mismatch"),
        ("parent_run_id", "run-wrong", "segment_manifest_parent_mismatch"),
        ("segment_sequence", -1, "segment_manifest_sequence_invalid"),
        ("segment_sequence", True, "segment_manifest_sequence_invalid"),
        ("segment_sequence", "0", "segment_manifest_sequence_invalid"),
        ("marker_state", "active", "segment_manifest_terminal_state_invalid"),
        ("clean_shutdown", None, "segment_manifest_terminal_state_invalid"),
        ("shutdown_reason", None, "segment_manifest_terminal_state_invalid"),
    ],
)
def test_run_segment_manifest_identity_damage_is_reported_and_blocks_run(
    field: str,
    value: object,
    expected_issue: str,
    tmp_path: Path,
) -> None:
    segment, _events_path = _create_closed_unfinished_run(tmp_path)
    manifest_path = segment / "segment.json"
    manifest = _json(manifest_path)
    manifest[field] = value
    atomic_write_json(manifest_path, manifest, artifact_type="test")
    damaged = manifest_path.read_bytes()
    recovered = _manager(tmp_path)
    recovered.start(30)
    report = next(
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    )
    assert expected_issue in report["issue_types"]
    assert report["source_segment_identity"] == segment.name
    assert report["path_segment_id"] == segment.name
    assert report["manifest_segment_id"] == manifest.get("recorder_segment_id")
    assert report["path_segment_kind"] == "run_bound"
    assert report["manifest_segment_kind"] == manifest.get("segment_kind")
    assert report["expected_parent_run_id"] == "run-a"
    assert report["manifest_parent_run_id"] == manifest.get("parent_run_id")
    assert "run-a" in recovered._blocked_run_ids
    assert manifest_path.read_bytes() == damaged


def test_segment_manifest_missing_frozen_field_is_schema_invalid(tmp_path: Path) -> None:
    segment, _events_path = _create_closed_unfinished_run(tmp_path)
    manifest_path = segment / "segment.json"
    manifest = _json(manifest_path)
    manifest.pop("pid")
    atomic_write_json(manifest_path, manifest, artifact_type="test")
    damaged = manifest_path.read_bytes()
    recovered = _manager(tmp_path)
    recovered.start(30)
    report = next(
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    )
    assert "segment_manifest_schema_invalid" in report["issue_types"]
    assert "run-a" in recovered._blocked_run_ids
    assert manifest_path.read_bytes() == damaged


def test_bootstrap_manifest_kind_damage_reports_without_blocking_fake_run(
    tmp_path: Path,
) -> None:
    owner = _manager(tmp_path)
    segment = owner.start(1)
    owner.close(2)
    manifest_path = segment / "segment.json"
    manifest = _json(manifest_path)
    manifest["segment_kind"] = "run_bound"
    atomic_write_json(manifest_path, manifest, artifact_type="test")
    damaged = manifest_path.read_bytes()
    recovered = _manager(tmp_path)
    recovered.start(30)
    report = next(
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    )
    assert "segment_manifest_kind_mismatch" in report["issue_types"]
    assert report["source_segment_identity"] == segment.name
    assert report["source_parent_run_id"] is None
    assert "run-fake" not in recovered._blocked_run_ids
    recovered.record_competition_context(
        _context(run_id="run-fake", fingerprint="fingerprint-fake"), 40, 41
    )
    assert recovered.state is RecorderRuntimeState.RUN_ACTIVE
    assert manifest_path.read_bytes() == damaged


class _FakeProcess:
    def __init__(self, waits: list[object], initial_poll: object = None) -> None:
        self.waits = list(waits)
        self.initial_poll = initial_poll
        self.signals: list[object] = []
        self.output_path: Path | None = None

    def bind(self, command: Any) -> "_FakeProcess":
        self.output_path = Path(command[4])
        if self.initial_poll is None:
            self.output_path.mkdir(parents=True, exist_ok=True)
            (self.output_path / "rosbag_0.db3").write_bytes(b"sqlite")
        return self

    def poll(self) -> object:
        return self.initial_poll

    def send_signal(self, value: object) -> None:
        self.signals.append(value)

    def terminate(self) -> None:
        self.signals.append("terminate")

    def kill(self) -> None:
        self.signals.append("kill")

    def wait(self, timeout: float) -> int:
        value = self.waits.pop(0)
        if value == "timeout":
            raise subprocess.TimeoutExpired("bag", timeout)
        result = int(value)
        if result == 0 and self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)
            (self.output_path / "metadata.yaml").write_text(
                "rosbag2_bagfile_information: {}\n", encoding="utf-8"
            )
        return result


def _bag_manager(tmp_path: Path, process: _FakeProcess) -> RecorderRuntimeManager:
    return RecorderRuntimeManager(
        _config(tmp_path, record_rosbag=True),
        _pairing(),
        process_factory=lambda command: process.bind(command),
        wall_now=lambda: "2026-01-02T03:04:05Z",
        monotonic_now_ns=iter(range(1000, 100000)).__next__,
        pid=123,
        environment={},
    )


@pytest.mark.parametrize(
    ("waits", "expected"),
    [
        ([0], [signal.SIGINT]),
        (["timeout", 0], [signal.SIGINT, "terminate"]),
        (["timeout", "timeout", 0], [signal.SIGINT, "terminate", "kill"]),
    ],
)
def test_bag_shutdown_escalation(
    waits: list[object], expected: list[object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    process = _FakeProcess(waits)
    manager = _bag_manager(tmp_path, process)
    segment = manager.start(1)
    manager.close(2)
    assert process.signals == expected
    metadata = _json(segment / "metadata.json")
    assert metadata["rosbag_exit_code"] == 0
    events = _events(segment / "events.jsonl")
    assert [event["event_type"] for event in events].count("bag_started") == 1
    assert [event["event_type"] for event in events].count("bag_stopped") == 1


def test_tf_static_zero_messages_does_not_block_ready_or_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    commands: list[tuple[str, ...]] = []

    class Process(_FakeProcess):
        def wait(self, timeout: float) -> int:
            result = super().wait(timeout)
            assert self.output_path is not None
            (self.output_path / "metadata.yaml").write_text(
                "rosbag2_bagfile_information:\n"
                "  topics_with_message_count:\n"
                "    - topic_metadata: {name: /tf, type: tf2_msgs/msg/TFMessage}\n"
                "      message_count: 24\n"
                "    - topic_metadata: {name: /tf_static, type: tf2_msgs/msg/TFMessage}\n"
                "      message_count: 0\n",
                encoding="utf-8",
            )
            return result

    process = Process([0])

    def factory(command: Any) -> _FakeProcess:
        commands.append(tuple(command))
        return process.bind(command)

    manager = RecorderRuntimeManager(
        _config(tmp_path, record_rosbag=True),
        _pairing(),
        process_factory=factory,
    )
    segment = manager.start(1)
    manager.close(2)
    command = commands[0]
    qos_index = command.index("--qos-profile-overrides-path")
    assert command[qos_index + 1] == str(QOS_PATH.resolve())
    assert command[qos_index + 2 :] == DEFAULT_ROSBAG_TOPICS
    assert (segment / "COMPLETE").is_file()
    assert not (segment / "ACTIVE").exists()
    metadata = (segment / "rosbag" / "metadata.yaml").read_text(encoding="utf-8")
    assert "name: /tf_static" in metadata
    assert "message_count: 0" in metadata


def test_bag_immediate_exit_fails_start_and_leaves_active_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    process = _FakeProcess([], initial_poll=2)
    manager = _bag_manager(tmp_path, process)
    with pytest.raises(RuntimeError, match="rosbag完整性检查失败"):
        manager.start(1)
    assert manager.state is RecorderRuntimeState.FAILED
    segments = list((tmp_path / "bootstrap").iterdir())
    assert len(segments) == 1
    assert not (segments[0] / "COMPLETE").exists()
    assert (segments[0] / "ACTIVE").exists()
    assert _json(segments[0] / "segment.json")["clean_shutdown"] is None
    assert _json(segments[0] / "metadata.json")["rosbag_exit_code"] == 2


def test_pairer_close_failure_still_stops_owned_bag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    process = _FakeProcess([0])
    manager = _bag_manager(tmp_path, process)
    manager.start(1)

    def fail_pairer(self: EpisodeRecorder, *args: Any, **kwargs: Any) -> tuple[str, ...]:
        raise RuntimeError("injected pairing close failure")

    monkeypatch.setattr(EpisodeRecorder, "close_action_pairing", fail_pairer)
    with pytest.raises(RuntimeError, match="rosbag已执行有界停止"):
        manager.close(2)
    assert process.signals == [signal.SIGINT]
    assert manager.state is RecorderRuntimeState.FAILED


def test_bag_early_exit_is_reported_without_automatic_rollover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    process = _FakeProcess([], initial_poll=None)
    manager = _bag_manager(tmp_path, process)
    manager.start(1)
    segment_id = manager.current_segment_id
    process.initial_poll = 9
    failure = manager.monitor_bag(2)
    assert "提前退出" in failure
    assert manager.current_segment_id == segment_id
    assert manager.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE


def test_rollover_stops_old_bag_before_starting_unique_new_bag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    order: list[str] = []
    commands: list[tuple[str, ...]] = []

    class Process(_FakeProcess):
        def __init__(self, label: str) -> None:
            super().__init__([0])
            self.label = label

        def send_signal(self, value: object) -> None:
            order.append(f"stop:{self.label}")
            super().send_signal(value)

    def factory(command: Any) -> Process:
        frozen = tuple(command)
        commands.append(frozen)
        label = str(len(commands))
        order.append(f"start:{label}")
        return Process(label).bind(command)  # type: ignore[return-value]

    manager = RecorderRuntimeManager(
        _config(tmp_path, record_rosbag=True), _pairing(), process_factory=factory
    )
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    assert order[:3] == ["start:1", "stop:1", "start:2"]
    assert len(commands) == 2
    assert commands[0][4] != commands[1][4]
    assert "/tf" in commands[0]
    assert "/tf_static" in commands[0]
    qos_index = commands[0].index("--qos-profile-overrides-path")
    assert commands[0][qos_index + 1] == str(QOS_PATH.resolve())
    assert "/team/action_dispatch" not in commands[0]


def test_fast_context_rollover_waits_for_bootstrap_bag_ready_and_completes_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    processes: list[_FakeProcess] = []
    probes = iter((False, False, True, True))
    waits: list[float] = []
    steady = iter((0.0, 0.1, 0.2, 1.0)).__next__

    def factory(command: Any) -> _FakeProcess:
        process = _FakeProcess([0]).bind(command)
        processes.append(process)
        return process

    manager = RecorderRuntimeManager(
        replace(
            _config(tmp_path, record_rosbag=True),
            bag_startup_timeout_sec=1.0,
            bag_startup_poll_interval_sec=0.25,
        ),
        _pairing(),
        process_factory=factory,
        bag_ready_probe=lambda _path: next(probes),
        steady_now=steady,
        startup_wait=waits.append,
    )
    bootstrap = manager.start(1)
    assert waits == [0.25, 0.25]
    assert manager._active is not None and manager._active.bag_ready is True

    manager.record_competition_context(_context(), 2, 3)
    assert manager.state is RecorderRuntimeState.RUN_ACTIVE
    assert len(processes) == 2
    assert processes[0].signals == [signal.SIGINT]
    assert (bootstrap / "COMPLETE").is_file()
    assert not (bootstrap / "ACTIVE").exists()
    assert (bootstrap / "rosbag" / "metadata.yaml").is_file()
    assert _json(bootstrap / "metadata.json")["rosbag_exit_code"] == 0

    run_segment = manager.current_segment_dir
    assert run_segment is not None
    manager.close(4)
    assert (run_segment / "COMPLETE").is_file()
    assert not (run_segment / "ACTIVE").exists()
    assert (run_segment / "rosbag" / "metadata.yaml").is_file()
    assert _json(run_segment / "metadata.json")["rosbag_exit_code"] == 0


@pytest.mark.parametrize("failure", ["nonzero_exit", "metadata_missing"])
def test_rosbag_completion_failure_never_creates_complete_marker(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")

    class Process(_FakeProcess):
        def wait(self, timeout: float) -> int:
            result = super().wait(timeout)
            if failure == "metadata_missing" and self.output_path is not None:
                (self.output_path / "metadata.yaml").unlink()
            return result

    process = Process([-2 if failure == "nonzero_exit" else 0])
    manager = RecorderRuntimeManager(
        _config(tmp_path, record_rosbag=True),
        _pairing(),
        process_factory=lambda command: process.bind(command),
    )
    segment = manager.start(1)
    with pytest.raises(RuntimeError, match="rosbag完整性检查失败"):
        manager.close(2)
    assert manager.state is RecorderRuntimeState.FAILED
    assert (segment / "ACTIVE").is_file()
    assert not (segment / "COMPLETE").exists()
    assert _json(segment / "metadata.json")["rosbag_exit_code"] == (
        -2 if failure == "nonzero_exit" else 0
    )


def test_rosbag_exit_during_ready_handshake_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")

    class Process(_FakeProcess):
        def __init__(self) -> None:
            super().__init__([])
            self.polls = iter((None, None, 7))

        def poll(self) -> object:
            return next(self.polls)

    process = Process()
    manager = RecorderRuntimeManager(
        replace(
            _config(tmp_path, record_rosbag=True),
            bag_startup_timeout_sec=1.0,
            bag_startup_poll_interval_sec=0.1,
        ),
        _pairing(),
        process_factory=lambda command: process.bind(command),
        bag_ready_probe=lambda _path: False,
        steady_now=iter((0.0, 0.1)).__next__,
        startup_wait=lambda _seconds: None,
    )
    with pytest.raises(RuntimeError, match="ready确认前退出"):
        manager.start(1)
    segment = next((tmp_path / "bootstrap").iterdir())
    assert manager.state is RecorderRuntimeState.FAILED
    assert (segment / "ACTIVE").is_file()
    assert not (segment / "COMPLETE").exists()
    assert _json(segment / "metadata.json")["rosbag_exit_code"] == 7


def _failing_nth_bag_factory(fail_at: int) -> tuple[Any, list[_FakeProcess]]:
    processes: list[_FakeProcess] = []

    def factory(command: Any) -> _FakeProcess:
        index = len(processes) + 1
        process = _FakeProcess(
            [0] if index != fail_at else [],
            initial_poll=2 if index == fail_at else None,
        )
        processes.append(process)
        return process.bind(command)

    return factory, processes


def test_first_run_bag_failure_is_sealed_tracked_and_runtime_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    factory, _processes = _failing_nth_bag_factory(2)
    manager = RecorderRuntimeManager(
        _config(tmp_path, record_rosbag=True), _pairing(), process_factory=factory
    )
    manager.start(1)
    with pytest.raises(RuntimeError, match="rosbag完整性检查失败"):
        manager.record_competition_context(_context(), 10, 11)
    assert manager.state is RecorderRuntimeState.FAILED
    assert manager.current_segment_dir is not None
    with pytest.raises(RuntimeError, match="没有可写segment"):
        manager.record_referee_message("/referee/score", 0, 12)
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert len(manifest["recorder_segment_ids"]) == 1
    failed = tmp_path / "runs" / "run-a" / "segments" / manifest["recorder_segment_ids"][0]
    assert not (failed / "COMPLETE").exists()
    assert (failed / "ACTIVE").exists()
    segment = _json(failed / "segment.json")
    assert segment["clean_shutdown"] is None
    assert segment["shutdown_reason"] is None
    with pytest.raises(RuntimeError, match="rosbag完整性检查失败"):
        manager.close(13)
    assert manager.state is RecorderRuntimeState.FAILED


def test_bootstrap_context_raw_failure_seals_and_disables_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    segment = manager.start(1)

    def fail_context(self: EpisodeRecorder, context: CompetitionContext) -> None:
        raise RuntimeError("injected bootstrap raw failure")

    monkeypatch.setattr(EpisodeRecorder, "record_competition_context", fail_context)
    with pytest.raises(RuntimeError, match="bootstrap raw failure"):
        manager.record_competition_context(_context(), 10, 11)
    assert manager.state is RecorderRuntimeState.FAILED
    assert manager.current_segment_dir is None
    assert (segment / "COMPLETE").exists()
    with pytest.raises(RuntimeError, match="没有可写segment"):
        manager.record_referee_message("/referee/score", 0, 12)


def test_bootstrap_context_event_failure_cannot_claim_run_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    segment = manager.start(1)
    original = append_jsonl

    def fail_context_event(path: Path, payload: Any, **kwargs: Any) -> None:
        if payload.get("event_type") == "competition_context_updated":
            raise RuntimeError("injected bootstrap context event failure")
        original(path, payload, **kwargs)

    monkeypatch.setattr("team_sorting.recorder_runtime.append_jsonl", fail_context_event)
    with pytest.raises(RuntimeError, match="context event failure"):
        manager.record_competition_context(_context(), 10, 11)
    assert manager.state is RecorderRuntimeState.FAILED
    assert "run_bound" not in {
        event["event_type"] for event in _events(segment / "events.jsonl")
    }
    assert list((tmp_path / "runs").iterdir()) == []


@pytest.mark.parametrize(
    "failed_step",
    ["run_bound_event", "run_context_raw", "run_context_event"],
)
def test_first_run_identity_commit_failure_is_sealed_and_unwritable(
    failed_step: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    original_append = append_jsonl
    original_record = EpisodeRecorder.record_competition_context

    def fail_event(path: Path, payload: Any, **kwargs: Any) -> None:
        active = manager._active
        event_type = payload.get("event_type")
        should_fail = (
            active is not None
            and active.kind == "run_bound"
            and (
                (failed_step == "run_bound_event" and event_type == "run_bound")
                or (
                    failed_step == "run_context_event"
                    and event_type == "competition_context_updated"
                )
            )
        )
        if should_fail:
            raise RuntimeError(f"injected {failed_step}")
        original_append(path, payload, **kwargs)

    def fail_raw(self: EpisodeRecorder, context: CompetitionContext) -> None:
        if failed_step == "run_context_raw" and manager._active is not None and manager._active.kind == "run_bound":
            raise RuntimeError("injected run_context_raw")
        original_record(self, context)

    monkeypatch.setattr("team_sorting.recorder_runtime.append_jsonl", fail_event)
    monkeypatch.setattr(EpisodeRecorder, "record_competition_context", fail_raw)
    with pytest.raises(RuntimeError, match="injected"):
        manager.record_competition_context(_context(), 10, 11)
    assert manager.state is RecorderRuntimeState.FAILED
    assert manager._last_valid_context is None
    with pytest.raises(RuntimeError, match="没有可写segment"):
        manager.record_referee_message("/referee/score", 0, 12)
    run_segment = next((tmp_path / "runs" / "run-a" / "segments").iterdir())
    assert (run_segment / "COMPLETE").exists()
    assert not (run_segment / "ACTIVE").exists()
    assert _json(run_segment / "segment.json")["clean_shutdown"] is False


def test_first_run_bag_started_event_failure_stops_bag_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    processes: list[_FakeProcess] = []

    def factory(command: Any) -> _FakeProcess:
        process = _FakeProcess([0])
        processes.append(process)
        return process.bind(command)

    manager = RecorderRuntimeManager(
        _config(tmp_path, record_rosbag=True), _pairing(), process_factory=factory
    )
    manager.start(1)
    original = append_jsonl

    def fail_run_bag_event(path: Path, payload: Any, **kwargs: Any) -> None:
        if payload.get("event_type") == "bag_started":
            raise RuntimeError("injected run bag_started event failure")
        original(path, payload, **kwargs)

    monkeypatch.setattr("team_sorting.recorder_runtime.append_jsonl", fail_run_bag_event)
    with pytest.raises(RuntimeError, match="bag_started event failure"):
        manager.record_competition_context(_context(), 10, 11)
    assert manager.state is RecorderRuntimeState.FAILED
    assert len(processes) == 2
    assert processes[1].signals == [signal.SIGINT]
    with pytest.raises(RuntimeError, match="没有可写segment"):
        manager.record_referee_message("/referee/score", 0, 12)


def test_first_run_metadata_failure_preserves_append_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    original = EpisodeRecorder._write_metadata
    failed = False

    def fail_first_run_metadata(self: EpisodeRecorder, *args: Any, **kwargs: Any) -> Path:
        nonlocal failed
        if manager._active is not None and manager._active.kind == "run_bound" and not failed:
            failed = True
            raise RuntimeError("injected first-run metadata failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EpisodeRecorder, "_write_metadata", fail_first_run_metadata)
    with pytest.raises(RuntimeError, match="first-run metadata failure"):
        manager.record_competition_context(_context(), 10, 11)
    assert manager.state is RecorderRuntimeState.FAILED
    assert manager._last_valid_context is None
    segment = next((tmp_path / "runs" / "run-a" / "segments").iterdir())
    assert len((segment / "competition_contexts.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert _json(segment / "segment.json")["clean_shutdown"] is False
    recovered = _manager(tmp_path)
    recovered.start(20)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    ]
    assert len(reports) == 1
    assert "segment_unclean_shutdown" in reports[0]["issue_types"]


def test_same_run_raw_failure_does_not_advance_last_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(task_id=1), 10, 11)
    original = EpisodeRecorder.record_competition_context

    def fail_task_two(self: EpisodeRecorder, context: CompetitionContext) -> None:
        if context.current_task_id == 2:
            raise RuntimeError("injected same-run raw failure")
        original(self, context)

    monkeypatch.setattr(EpisodeRecorder, "record_competition_context", fail_task_two)
    with pytest.raises(RuntimeError, match="same-run raw failure"):
        manager.record_competition_context(_context(task_id=2), 20, 21)
    assert manager.state is RecorderRuntimeState.FAILED
    assert manager._last_valid_context is not None
    assert manager._last_valid_context.current_task_id == 1
    events = _events(tmp_path / "runs" / "run-a" / "events.jsonl")
    assert not any(event["event_type"] == "task_transition" for event in events)


def test_same_run_metadata_failure_after_raw_append_keeps_previous_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(task_id=1), 10, 11)
    segment = manager.current_segment_dir
    original = EpisodeRecorder._write_metadata
    failed = False

    def fail_once(self: EpisodeRecorder, *args: Any, **kwargs: Any) -> Path:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected metadata commit failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EpisodeRecorder, "_write_metadata", fail_once)
    with pytest.raises(RuntimeError, match="metadata commit failure"):
        manager.record_competition_context(_context(task_id=2), 20, 21)
    assert manager._last_valid_context is not None
    assert manager._last_valid_context.current_task_id == 1
    contexts = [
        json.loads(line)
        for line in (segment / "competition_contexts.jsonl").read_text(encoding="utf-8").splitlines()  # type: ignore[operator]
    ]
    assert contexts[-1]["current_task_id"] == 2
    assert manager.state is RecorderRuntimeState.FAILED
    recovered = _manager(tmp_path)
    recovered.start(30)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    ]
    assert len(reports) == 1
    assert "segment_unclean_shutdown" in reports[0]["issue_types"]


def test_run_change_boundary_event_failure_seals_old_run_without_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    original = append_jsonl

    def fail_run_changed(path: Path, payload: Any, **kwargs: Any) -> None:
        if payload.get("event_type") == "run_changed":
            raise RuntimeError("injected run_changed failure")
        original(path, payload, **kwargs)

    monkeypatch.setattr("team_sorting.recorder_runtime.append_jsonl", fail_run_changed)
    with pytest.raises(RuntimeError, match="run_changed failure"):
        manager.record_competition_context(
            _context(run_id="run-b", fingerprint="fingerprint-b"), 20, 21
        )
    assert manager.state is RecorderRuntimeState.FAILED
    assert not (tmp_path / "runs" / "run-b").exists()
    with pytest.raises(RuntimeError, match="没有可写segment"):
        manager.record_referee_message("/referee/score", 0, 22)


@pytest.mark.parametrize("boundary", ["bootstrap_bind", "run_change", "finished"])
def test_context_rollover_close_failure_never_opens_target_segment(
    boundary: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    if boundary != "bootstrap_bind":
        manager.record_competition_context(_context(), 10, 11)
    before_bootstrap = set((tmp_path / "bootstrap").iterdir())

    def fail_pairing(self: EpisodeRecorder, *args: Any, **kwargs: Any) -> tuple[str, ...]:
        raise RuntimeError(f"injected {boundary} close failure")

    monkeypatch.setattr(EpisodeRecorder, "close_action_pairing", fail_pairing)
    target = (
        _context()
        if boundary == "bootstrap_bind"
        else _context(run_id="run-b", fingerprint="fingerprint-b")
        if boundary == "run_change"
        else _context(finished=True)
    )
    with pytest.raises(RuntimeError, match="有界停止"):
        manager.record_competition_context(target, 20, 21)
    assert manager.state is RecorderRuntimeState.FAILED
    assert manager._active is not None and not manager._active.accepting_writes
    if boundary == "bootstrap_bind":
        assert not (tmp_path / "runs" / "run-a").exists()
    elif boundary == "run_change":
        assert not (tmp_path / "runs" / "run-b").exists()
    else:
        assert set((tmp_path / "bootstrap").iterdir()) == before_bootstrap


def test_run_finalize_failure_precedes_complete_and_close_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    segment = manager.current_segment_dir
    real_finalize = manager._finalize_run_manifest

    def fail_finalize(run_id: str, now_ros_ns: int, reason: str) -> None:
        raise RuntimeError("injected run finalize failure")

    monkeypatch.setattr(manager, "_finalize_run_manifest", fail_finalize)
    with pytest.raises(RuntimeError, match="run finalize failure"):
        manager.record_competition_context(_context(finished=True), 20, 21)
    assert manager.state is RecorderRuntimeState.FAILED
    assert (segment / "ACTIVE").exists()  # type: ignore[operator]
    assert not (segment / "COMPLETE").exists()  # type: ignore[operator]
    recovery_probe = _manager(tmp_path)
    recovery_probe.start(22)
    reports = [
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_segment_path") == str(segment)
    ]
    assert len(reports) == 1
    assert reports[0]["source_parent_run_id"] == "run-a"
    probe_bootstrap = recovery_probe.current_segment_dir
    recovery_probe.record_competition_context(_context(), 22, 23)
    assert recovery_probe.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert _events(probe_bootstrap / "events.jsonl")[-1]["validity"] == "invalid"  # type: ignore[operator]
    recovery_probe.close(22)
    with pytest.raises(RuntimeError, match="没有可写|关闭阶段"):
        manager.record_referee_message("/referee/score", 0, 22)
    monkeypatch.setattr(manager, "_finalize_run_manifest", real_finalize)
    manager.close(23)
    assert manager.state is RecorderRuntimeState.CLOSED
    assert (segment / "COMPLETE").exists()  # type: ignore[operator]
    assert not (segment / "ACTIVE").exists()  # type: ignore[operator]
    assert _json(tmp_path / "runs" / "run-a" / "manifest.json")["end_ros_ns"] == 20


@pytest.mark.parametrize("flag", ["recorder_finished", "complete_created"])
def test_business_write_rejects_segment_with_close_progress_flag(
    flag: str, tmp_path: Path
) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    assert manager._active is not None
    setattr(manager._active, flag, True)
    with pytest.raises(RuntimeError, match="关闭阶段"):
        manager.record_referee_message("/referee/score", 0, 2)


def test_new_run_bag_failure_never_leaves_half_active_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    factory, _processes = _failing_nth_bag_factory(3)
    manager = RecorderRuntimeManager(
        _config(tmp_path, record_rosbag=True), _pairing(), process_factory=factory
    )
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    with pytest.raises(RuntimeError, match="rosbag完整性检查失败"):
        manager.record_competition_context(
            _context(run_id="run-b", fingerprint="fingerprint-b"), 20, 21
        )
    assert manager.state is RecorderRuntimeState.FAILED
    assert manager.current_segment_dir is not None
    manifest = _json(tmp_path / "runs" / "run-b" / "manifest.json")
    failed = tmp_path / "runs" / "run-b" / "segments" / manifest["recorder_segment_ids"][0]
    assert not (failed / "COMPLETE").exists()
    assert (failed / "ACTIVE").exists()
    assert _json(failed / "segment.json")["clean_shutdown"] is None


def test_post_finished_bootstrap_bag_failure_enters_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/usr/bin/ros2")
    factory, _processes = _failing_nth_bag_factory(3)
    manager = RecorderRuntimeManager(
        _config(tmp_path, record_rosbag=True), _pairing(), process_factory=factory
    )
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    with pytest.raises(RuntimeError, match="rosbag完整性检查失败"):
        manager.record_competition_context(_context(finished=True), 20, 21)
    assert manager.state is RecorderRuntimeState.FAILED
    assert manager.current_segment_dir is not None
    failed_bootstrap = max(
        (tmp_path / "bootstrap").iterdir(), key=lambda path: path.stat().st_mtime_ns
    )
    assert not (failed_bootstrap / "COMPLETE").exists()
    assert (failed_bootstrap / "ACTIVE").exists()
    assert _json(failed_bootstrap / "segment.json")["clean_shutdown"] is None


def test_existing_run_manifest_fingerprint_conflict_fails_closed(tmp_path: Path) -> None:
    first = _manager(tmp_path)
    first.start(1)
    first.record_competition_context(_context(), 10, 11)
    first.close(20)
    manifest_path = tmp_path / "runs" / "run-a" / "manifest.json"
    manifest = _json(manifest_path)
    manifest["end_ros_ns"] = None
    manifest["end_wall_utc"] = None
    manifest["clean_shutdown"] = None
    manifest["shutdown_reason"] = None
    manifest["recovery_required"] = None
    atomic_write_json(manifest_path, manifest, artifact_type="test")
    second = _manager(tmp_path, recovery_scan_enabled=False)
    second.start(30)
    bootstrap = second.current_segment_dir
    second.record_competition_context(
        _context(fingerprint="different"), 40, 41
    )
    assert second.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert _context_run_ids(bootstrap) == set()  # type: ignore[arg-type]
    event = _events(bootstrap / "events.jsonl")[-1]  # type: ignore[operator]
    assert event["validity"] == "invalid"
    assert "fingerprint" in event["invalid_reasons"][0]


def test_clean_restart_appends_segment_to_same_unfinished_run(tmp_path: Path) -> None:
    first = _manager(tmp_path)
    first.start(1)
    first.record_competition_context(_context(), 10, 11)
    first_segment = first.current_segment_id
    first.close(20)
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert manifest["end_ros_ns"] is None
    second = _manager(tmp_path)
    second.start(30)
    second.record_competition_context(_context(), 40, 41)
    second_segment = second.current_segment_id
    assert second_segment != first_segment
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert manifest["recorder_segment_ids"] == [first_segment, second_segment]


@pytest.mark.parametrize(
    "segment_ids",
    [
        ["segment_one", "segment_one"],
        ["segment_one", 2],
        ["segment_one", "../segment_two"],
        ["segment_one", ""],
    ],
)
def test_invalid_persisted_segment_ids_fail_closed_without_manifest_repair(
    segment_ids: list[object], tmp_path: Path
) -> None:
    first_segment, _events_path = _create_closed_unfinished_run(tmp_path)
    manifest_path = tmp_path / "runs" / "run-a" / "manifest.json"
    manifest = _json(manifest_path)
    manifest["recorder_segment_ids"] = segment_ids
    atomic_write_json(manifest_path, manifest, artifact_type="test")
    damaged = manifest_path.read_bytes()
    before_segments = {path.name for path in first_segment.parent.iterdir()}

    second = _manager(tmp_path, recovery_scan_enabled=False)
    second.start(30)
    bootstrap = second.current_segment_dir
    second.record_competition_context(_context(), 40, 41)
    assert second.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert _events(bootstrap / "events.jsonl")[-1]["validity"] == "invalid"  # type: ignore[operator]
    assert manifest_path.read_bytes() == damaged
    assert {path.name for path in first_segment.parent.iterdir()} == before_segments


def test_duplicate_json_keys_in_run_manifest_are_rejected_without_repair(
    tmp_path: Path,
) -> None:
    first_segment, _events_path = _create_closed_unfinished_run(tmp_path)
    manifest_path = tmp_path / "runs" / "run-a" / "manifest.json"
    manifest = _json(manifest_path)
    manifest.pop("recorder_segment_ids")
    raw = (
        json.dumps(manifest, ensure_ascii=False)[:-1]
        + ',"recorder_segment_ids":["'
        + first_segment.name
        + '"],"recorder_segment_ids":[]}'
    )
    manifest_path.write_text(raw, encoding="utf-8")
    damaged = manifest_path.read_bytes()
    second = _manager(tmp_path, recovery_scan_enabled=False)
    second.start(30)
    with pytest.raises(RuntimeError, match="重复JSON key"):
        second._read_json(manifest_path, "run_manifest")
    second.record_competition_context(_context(), 40, 41)
    assert second.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert manifest_path.read_bytes() == damaged
    assert {path.name for path in first_segment.parent.iterdir()} == {first_segment.name}


@pytest.mark.parametrize("constant", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_constant_in_run_manifest_is_rejected(
    constant: float, tmp_path: Path
) -> None:
    first_segment, _events_path = _create_closed_unfinished_run(tmp_path)
    manifest_path = tmp_path / "runs" / "run-a" / "manifest.json"
    manifest = _json(manifest_path)
    manifest["start_ros_ns"] = constant
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    damaged = manifest_path.read_bytes()
    second = _manager(tmp_path, recovery_scan_enabled=False)
    second.start(30)
    with pytest.raises(RuntimeError, match="非有限JSON常量"):
        second._read_json(manifest_path, "run_manifest")
    second.record_competition_context(_context(), 40, 41)
    assert second.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert manifest_path.read_bytes() == damaged
    assert {path.name for path in first_segment.parent.iterdir()} == {first_segment.name}


def test_node_start_time_is_process_constant_and_rollover_does_not_end_node(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    first_bootstrap = manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    run_a = manager.current_segment_dir
    manager.record_competition_context(
        _context(run_id="run-b", fingerprint="fingerprint-b"), 20, 21
    )
    run_b = manager.current_segment_dir
    manifests = [_json(path / "segment.json") for path in (first_bootstrap, run_a, run_b)]  # type: ignore[arg-type]
    assert {item["node_start_ros_ns"] for item in manifests} == {1}
    assert manifests[0]["node_end_ros_ns"] is None
    assert manifests[1]["node_end_ros_ns"] is None


def test_shared_run_events_prefix_remains_verifiable_after_second_segment(
    tmp_path: Path,
) -> None:
    first = _manager(tmp_path)
    first.start(1)
    first.record_competition_context(_context(), 10, 11)
    first_segment = first.current_segment_dir
    first.close(20)
    inventory = _json(first_segment / "segment.json")["jsonl_artifacts"]  # type: ignore[operator]
    shared = next(item for item in inventory if item.get("shared_append_only"))
    assert "sha256" not in shared

    second = _manager(tmp_path)
    second.start(30)
    second.record_competition_context(_context(), 40, 41)
    second.close(50)
    events_path = tmp_path / "runs" / "run-a" / "events.jsonl"
    data = events_path.read_bytes()
    end = shared["byte_end_offset"]
    assert len(data) > end
    assert hashlib.sha256(data[:end]).hexdigest() == shared["sha256_prefix"]


def test_legacy_flat_directory_is_not_scanned_or_modified(tmp_path: Path) -> None:
    legacy = tmp_path / "episode_legacy"
    legacy.mkdir()
    source = legacy / "metadata.json"
    source.write_text('{"legacy":true}\n', encoding="utf-8")
    before = source.read_bytes()
    manager = _manager(tmp_path)
    manager.start(1)
    assert source.read_bytes() == before
    assert list((tmp_path / "recovery").glob("*.json")) == []


def test_runtime_creates_neither_samples_nor_training_episode_layout(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    manager.close(20)
    assert list(tmp_path.rglob("samples.jsonl")) == []
    assert not (tmp_path / "episodes").exists()


def test_recorder_node_destroy_retries_runtime_without_redestroying_parent(
    tmp_path: Path,
) -> None:
    class Logger:
        def error(self, message: str) -> None:
            pass

        def warning(self, message: str) -> None:
            pass

    class Timer:
        def cancel(self) -> None:
            pass

    class Node:
        def __init__(self, name: str) -> None:
            self.destroy_count = 0

        def declare_parameter(self, name: str, value: object) -> None:
            self.enabled = value

        def get_parameter(self, name: str) -> SimpleNamespace:
            return SimpleNamespace(value=self.enabled)

        def get_clock(self) -> SimpleNamespace:
            return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100))

        def create_subscription(self, *args: Any) -> object:
            return object()

        def create_timer(self, *args: Any) -> Timer:
            return Timer()

        def get_logger(self) -> Logger:
            return Logger()

        def destroy_node(self) -> str:
            self.destroy_count += 1
            return "destroyed"

    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    config["recorder"]["enabled"] = True
    config["recorder"]["record_rosbag"] = False
    config["recorder"]["root_dir"] = str(tmp_path)
    ros = SimpleNamespace(Node=Node, String=object, Int32=object)
    node = _create_recorder_node(ros)(config, ros)
    real_close = node._runtime.close
    close_calls = 0

    def flaky_close(*args: Any, **kwargs: Any) -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("injected close failure")
        real_close(*args, **kwargs)

    node._runtime.close = flaky_close
    with pytest.raises(RuntimeError, match="injected close failure"):
        node.destroy_node()
    assert node.destroy_count == 1
    assert close_calls == 1
    assert node.destroy_node() is None
    assert node._runtime.state is RecorderRuntimeState.CLOSED
    assert node.destroy_count == 1
    assert close_calls == 2
    assert node.destroy_node() is None
    assert close_calls == 2


def test_runtime_does_not_collect_hostname_or_complete_environment(tmp_path: Path) -> None:
    manager = RecorderRuntimeManager(
        _config(tmp_path),
        _pairing(),
        environment={"SECRET_TOKEN": "must-not-appear", "HOSTNAME": "must-not-appear"},
    )
    segment = manager.start(1)
    text = (segment / "events.jsonl").read_text(encoding="utf-8")
    assert "SECRET_TOKEN" not in text
    assert "must-not-appear" not in text
    assert _events(segment / "events.jsonl")[0]["payload"]["runtime_provenance"]["hostname_collected"] is False


def test_no_runtime_artifact_is_named_episode(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.start(1)
    manager.record_competition_context(_context(), 10, 11)
    manager.close(20)
    assert all("episode" not in path.name.lower() for path in tmp_path.rglob("*"))


def _persist_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_json(path, data, artifact_type="test")


def _target_bind_reason(root: Path, **config_overrides: Any) -> tuple[str, Path]:
    manager = _manager(root, recovery_scan_enabled=False, **config_overrides)
    manager.start(100)
    bootstrap = manager.current_segment_dir
    assert bootstrap is not None
    before = {
        path.name for path in (root / "runs" / "run-a" / "segments").iterdir()
    }
    manager.record_competition_context(_context(), 110, 111)
    after = {
        path.name for path in (root / "runs" / "run-a" / "segments").iterdir()
    }
    assert manager.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert before == after
    event = _events(bootstrap / "events.jsonl")[-1]
    assert event["validity"] == "invalid"
    return event["invalid_reasons"][0], bootstrap


@pytest.mark.parametrize(
    ("field", "issue"),
    [
        ("recorder_schema_sha256", "run_manifest_contract_hash_mismatch"),
        ("interface_schema_sha256", "run_manifest_contract_hash_mismatch"),
        ("observe_only", "run_manifest_runtime_gate_mismatch"),
        ("official_publish_enabled", "run_manifest_runtime_gate_mismatch"),
    ],
)
def test_existing_run_rejects_immutable_scalar_mismatch(
    field: str, issue: str, tmp_path: Path
) -> None:
    _create_closed_unfinished_run(tmp_path)
    path = tmp_path / "runs" / "run-a" / "manifest.json"
    manifest = _json(path)
    if field.endswith("sha256"):
        manifest[field] = "0" * 64
    else:
        manifest[field] = not manifest[field]
    _persist_json(path, manifest)
    damaged = path.read_bytes()
    reason, _bootstrap = _target_bind_reason(tmp_path)
    assert issue in reason
    assert path.read_bytes() == damaged


@pytest.mark.parametrize(
    ("field", "issue"),
    [
        ("config_sha256", "run_manifest_config_mismatch"),
        ("ros_domain_id", "run_manifest_runtime_fact_mismatch"),
        ("rmw_implementation", "run_manifest_runtime_fact_mismatch"),
        ("official_server_image_id", "run_manifest_runtime_fact_mismatch"),
        ("official_client_image_id", "run_manifest_runtime_fact_mismatch"),
        ("project_commit", "run_manifest_provenance_mismatch"),
    ],
)
def test_existing_run_rejects_immutable_envelope_mismatch(
    field: str, issue: str, tmp_path: Path
) -> None:
    _create_closed_unfinished_run(tmp_path)
    path = tmp_path / "runs" / "run-a" / "manifest.json"
    manifest = _json(path)
    envelope = dict(manifest[field])
    envelope["reason"] = f"{envelope.get('reason', '')}_tampered"
    manifest[field] = envelope
    _persist_json(path, manifest)
    reason, _bootstrap = _target_bind_reason(tmp_path)
    assert issue in reason
    if field == "project_commit":
        assert "audited continuation" in reason


def test_existing_run_with_identical_immutable_facts_appends(tmp_path: Path) -> None:
    _create_closed_unfinished_run(tmp_path)
    manager = _manager(tmp_path, recovery_scan_enabled=False)
    manager.start(100)
    manager.record_competition_context(_context(), 110, 111)
    assert manager.state is RecorderRuntimeState.RUN_ACTIVE
    assert len(_json(tmp_path / "runs" / "run-a" / "manifest.json")["recorder_segment_ids"]) == 2


def _make_three_closed_segments(root: Path) -> list[Path]:
    _create_closed_unfinished_run(root)
    for base in (100, 200):
        manager = _manager(root, recovery_scan_enabled=False)
        manager.start(base)
        manager.record_competition_context(_context(), base + 10, base + 11)
        manager.close(base + 20)
    manifest = _json(root / "runs" / "run-a" / "manifest.json")
    return [
        root / "runs" / "run-a" / "segments" / segment_id
        for segment_id in manifest["recorder_segment_ids"]
    ]


@pytest.mark.parametrize(
    ("damage", "issue"),
    [
        ("missing", "run_manifest_segment_missing"),
        ("unlisted", "run_manifest_unlisted_segment"),
        ("order", "run_manifest_segment_order_mismatch"),
        ("duplicate_manifest_id", "run_manifest_schema_invalid"),
        ("duplicate_sequence", "run_manifest_segment_sequence_duplicate"),
        ("sequence_gap", "run_manifest_segment_sequence_gap"),
    ],
)
def test_existing_run_rejects_segment_topology_damage(
    damage: str, issue: str, tmp_path: Path
) -> None:
    segments = _make_three_closed_segments(tmp_path)
    manifest_path = tmp_path / "runs" / "run-a" / "manifest.json"
    manifest = _json(manifest_path)
    if damage == "missing":
        manifest["recorder_segment_ids"].append("segment_missing")
    elif damage == "unlisted":
        manifest["recorder_segment_ids"].pop()
    elif damage == "order":
        manifest["recorder_segment_ids"][0:2] = reversed(
            manifest["recorder_segment_ids"][0:2]
        )
    elif damage == "duplicate_manifest_id":
        manifest["recorder_segment_ids"][1] = manifest["recorder_segment_ids"][0]
    else:
        segment_manifest_path = segments[1] / "segment.json"
        segment_manifest = _json(segment_manifest_path)
        segment_manifest["segment_sequence"] = 0 if damage == "duplicate_sequence" else 4
        _persist_json(segment_manifest_path, segment_manifest)
    _persist_json(manifest_path, manifest)
    manifest_bytes = manifest_path.read_bytes()
    reason, _bootstrap = _target_bind_reason(tmp_path)
    assert issue in reason
    assert manifest_path.read_bytes() == manifest_bytes


def test_consistent_three_segment_topology_allows_sequence_three(tmp_path: Path) -> None:
    _make_three_closed_segments(tmp_path)
    manager = _manager(tmp_path, recovery_scan_enabled=False)
    manager.start(400)
    manager.record_competition_context(_context(), 410, 411)
    assert manager.state is RecorderRuntimeState.RUN_ACTIVE
    assert _json(manager.current_segment_dir / "segment.json")["segment_sequence"] == 3  # type: ignore[operator]


@pytest.mark.parametrize(
    ("damage", "issue"),
    [
        ("tail", "run_events_trailing_incomplete"),
        ("middle", "run_events_middle_corruption"),
        ("duplicate_key", "run_events_trailing_json_invalid"),
        ("nan", "run_events_trailing_json_invalid"),
    ],
)
def test_target_binding_strictly_rejects_corrupt_shared_events_with_scan_disabled(
    damage: str, issue: str, tmp_path: Path
) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    original = events.read_bytes()
    damaged = {
        "tail": original + b"partial",
        "middle": b"{broken\n" + original,
        "duplicate_key": original + b'{"x":1,"x":2}\n',
        "nan": original + b'{"x":NaN}\n',
    }[damage]
    events.write_bytes(damaged)
    reason, _bootstrap = _target_bind_reason(tmp_path)
    assert issue in reason
    assert events.read_bytes() == damaged


@pytest.mark.parametrize("recovery_scan_enabled", [True, False])
def test_missing_shared_events_is_never_recreated(
    recovery_scan_enabled: bool, tmp_path: Path
) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    events.unlink()
    manager = _manager(tmp_path, recovery_scan_enabled=recovery_scan_enabled)
    manager.start(100)
    manager.record_competition_context(_context(), 110, 111)
    assert manager.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert not events.exists()
    if recovery_scan_enabled:
        reports = [_json(path) for path in (tmp_path / "recovery").glob("*.json")]
        assert len(reports) == 1
        assert reports[0]["source_parent_run_id"] == "run-a"
        assert "run_events_missing" in reports[0]["issue_types"]
        assert "run-a" in manager._blocked_run_ids


def test_valid_shared_events_and_prefix_allow_append(tmp_path: Path) -> None:
    _create_closed_unfinished_run(tmp_path)
    reason_manager = _manager(tmp_path, recovery_scan_enabled=False)
    reason_manager.start(100)
    reason_manager.record_competition_context(_context(), 110, 111)
    assert reason_manager.state is RecorderRuntimeState.RUN_ACTIVE


@pytest.mark.parametrize(
    ("damage", "issue"),
    [
        ("hash_format", "run_events_integrity_record_invalid"),
        ("wrong_path", "run_events_integrity_record_invalid"),
        ("truncated", "run_events_prefix_truncated"),
        ("missing", "run_events_integrity_record_missing"),
        ("duplicate", "run_events_integrity_record_invalid"),
    ],
)
def test_existing_run_rejects_invalid_shared_events_prefix_record(
    damage: str, issue: str, tmp_path: Path
) -> None:
    segment, events = _create_closed_unfinished_run(tmp_path)
    segment_path = segment / "segment.json"
    manifest = _json(segment_path)
    records = manifest["jsonl_artifacts"]
    shared = next(item for item in records if item.get("shared_append_only") is True)
    if damage == "hash_format":
        shared["sha256_prefix"] = "ABC"
    elif damage == "wrong_path":
        shared["path"] = "../../other.jsonl"
    elif damage == "truncated":
        shared["byte_end_offset"] = events.stat().st_size + 1
    elif damage == "missing":
        manifest["jsonl_artifacts"] = [item for item in records if item is not shared]
    else:
        manifest["jsonl_artifacts"].append(dict(shared))
    _persist_json(segment_path, manifest)
    segment_bytes = segment_path.read_bytes()
    events_bytes = events.read_bytes()
    reason, _bootstrap = _target_bind_reason(tmp_path)
    assert issue in reason
    assert segment_path.read_bytes() == segment_bytes
    assert events.read_bytes() == events_bytes


def test_shared_events_prefix_tamper_blocks_run_without_mutation(tmp_path: Path) -> None:
    segment, events = _create_closed_unfinished_run(tmp_path)
    raw = events.read_bytes()
    assert b'"validity":"valid"' in raw
    damaged = raw.replace(b'"validity":"valid"', b'"validity":"vAlid"', 1)
    events.write_bytes(damaged)
    segment_bytes = (segment / "segment.json").read_bytes()
    recovered = _manager(tmp_path)
    recovered.start(100)
    report = next(
        _json(path)
        for path in (tmp_path / "recovery").glob("*.json")
        if _json(path).get("source_parent_run_id") == "run-a"
    )
    assert "run_events_prefix_hash_mismatch" in report["issue_types"]
    assert "run-a" in recovered._blocked_run_ids
    assert events.read_bytes() == damaged
    assert (segment / "segment.json").read_bytes() == segment_bytes


@pytest.mark.parametrize("field", ["end_ros_ns", "clean_shutdown", "recovery_required"])
def test_partial_or_ended_run_terminal_state_rejects_append(
    field: str, tmp_path: Path
) -> None:
    _create_closed_unfinished_run(tmp_path)
    path = tmp_path / "runs" / "run-a" / "manifest.json"
    manifest = _json(path)
    manifest[field] = 123 if field == "end_ros_ns" else False
    _persist_json(path, manifest)
    reason, _bootstrap = _target_bind_reason(tmp_path)
    assert "run_manifest_terminal_state_invalid" in reason


def test_all_null_run_terminal_fields_allow_append(tmp_path: Path) -> None:
    _create_closed_unfinished_run(tmp_path)
    manifest = _json(tmp_path / "runs" / "run-a" / "manifest.json")
    assert all(
        manifest[field] is None
        for field in (
            "end_ros_ns",
            "end_wall_utc",
            "clean_shutdown",
            "shutdown_reason",
            "recovery_required",
        )
    )
    manager = _manager(tmp_path, recovery_scan_enabled=False)
    manager.start(100)
    manager.record_competition_context(_context(), 110, 111)
    assert manager.state is RecorderRuntimeState.RUN_ACTIVE


def _manager_with_environment(
    root: Path,
    environment: dict[str, str],
    **config_overrides: Any,
) -> RecorderRuntimeManager:
    return RecorderRuntimeManager(
        _config(root, **config_overrides),
        _pairing(),
        wall_now=lambda: "2026-01-02T03:04:05Z",
        monotonic_now_ns=iter(range(1000, 100000)).__next__,
        pid=123,
        environment=environment,
    )


def _create_finished_run(
    manager: RecorderRuntimeManager, *, run_id: str = "run-a", base: int = 1
) -> None:
    fingerprint = f"fingerprint-{run_id[-1]}"
    manager.start(base)
    manager.record_competition_context(
        _context(run_id=run_id, fingerprint=fingerprint), base + 10, base + 11
    )
    manager.record_competition_context(
        _context(run_id=run_id, fingerprint=fingerprint, finished=True),
        base + 20,
        base + 21,
    )
    manager.close(base + 30)


@pytest.mark.parametrize(
    "difference",
    ["observe_only", "official_publish_enabled", "config_sha256", "project_commit"],
)
def test_clean_historical_run_runtime_difference_is_not_recovery_damage(
    difference: str, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    config_a = tmp_path / "config-a.yaml"
    config_b = tmp_path / "config-b.yaml"
    config_a.write_text("recorder: A\n", encoding="utf-8")
    config_b.write_text("recorder: B\n", encoding="utf-8")
    environment_a = {"TEAM_SORTING_PROJECT_COMMIT": "commit-a"}
    environment_b = {"TEAM_SORTING_PROJECT_COMMIT": "commit-a"}
    owner_overrides: dict[str, Any] = {}
    current_overrides: dict[str, Any] = {}
    if difference in {"observe_only", "official_publish_enabled"}:
        current_overrides.update(observe_only=False, official_publish_enabled=True)
    elif difference == "config_sha256":
        owner_overrides["config_path"] = config_a
        current_overrides["config_path"] = config_b
    else:
        environment_b["TEAM_SORTING_PROJECT_COMMIT"] = "commit-b"
    _create_finished_run(
        _manager_with_environment(dataset, environment_a, **owner_overrides)
    )

    current = _manager_with_environment(
        dataset, environment_b, **current_overrides
    )
    bootstrap = current.start(100)
    assert list((dataset / "recovery").glob("*.json")) == []
    assert not any(
        event["event_type"] == "unclean_shutdown_detected"
        for event in _events(bootstrap / "events.jsonl")
    )


def test_unfinished_historical_run_difference_only_rejects_actual_append(
    tmp_path: Path,
) -> None:
    segment, _events_path = _create_closed_unfinished_run(tmp_path)
    manifest_path = tmp_path / "runs" / "run-a" / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    segment_names = {path.name for path in segment.parent.iterdir()}
    current = _manager(
        tmp_path,
        observe_only=False,
        official_publish_enabled=True,
    )
    bootstrap = current.start(100)
    assert list((tmp_path / "recovery").glob("*.json")) == []
    assert not any(
        event["event_type"] == "unclean_shutdown_detected"
        for event in _events(bootstrap / "events.jsonl")
    )

    current.record_competition_context(_context(), 110, 111)
    assert current.state is RecorderRuntimeState.BOOTSTRAP_ACTIVE
    assert "run_manifest_runtime_gate_mismatch" in _events(
        bootstrap / "events.jsonl"
    )[-1]["invalid_reasons"][0]
    assert manifest_path.read_bytes() == manifest_bytes
    assert {path.name for path in segment.parent.iterdir()} == segment_names


def _tamper_valid_run_event(events: Path, before: bytes = b'"validity":"valid"') -> None:
    raw = events.read_bytes()
    assert before in raw
    replacement = (
        b'"validity":"vAlid"'
        if before == b'"validity":"valid"'
        else b'"validity":"vaLid"'
    )
    events.write_bytes(raw.replace(before, replacement, 1))


def _unclean_detection_count(segment: Path) -> int:
    return sum(
        event["event_type"] == "unclean_shutdown_detected"
        for event in _events(segment / "events.jsonl")
    )


def test_intrinsic_prefix_damage_blocks_even_with_current_gate_difference(
    tmp_path: Path,
) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    _tamper_valid_run_event(events)
    current = _manager(
        tmp_path,
        observe_only=False,
        official_publish_enabled=True,
    )
    bootstrap = current.start(100)
    reports = [_json(path) for path in (tmp_path / "recovery").glob("*.json")]
    assert len(reports) == 1
    assert "run_events_prefix_hash_mismatch" in reports[0][
        "intrinsic_integrity_issues"
    ]
    assert reports[0]["append_compatibility_issues"] == []
    assert "run-a" in current._blocked_run_ids
    assert _unclean_detection_count(bootstrap) == 1


def test_identical_recovery_finding_reuses_report_and_suppresses_event(
    tmp_path: Path,
) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    _tamper_valid_run_event(events)
    first = _manager(tmp_path)
    first_bootstrap = first.start(100)
    first_paths = list((tmp_path / "recovery").glob("*.json"))
    assert len(first_paths) == 1
    first_report = _json(first_paths[0])
    first_bytes = first_paths[0].read_bytes()
    assert first_report["report_id"] == f"recovery_{first_report['finding_fingerprint']}"
    assert _unclean_detection_count(first_bootstrap) == 1
    assert "run-a" in first._blocked_run_ids
    first.close(120)

    second = _manager(tmp_path)
    second_bootstrap = second.start(200)
    second_paths = list((tmp_path / "recovery").glob("*.json"))
    assert second_paths == first_paths
    assert second_paths[0].read_bytes() == first_bytes
    assert _json(second_paths[0])["finding_fingerprint"] == first_report[
        "finding_fingerprint"
    ]
    assert _unclean_detection_count(second_bootstrap) == 0
    assert "run-a" in second._blocked_run_ids


def test_changed_corrupt_source_creates_new_finding_and_event(tmp_path: Path) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    _tamper_valid_run_event(events)
    first = _manager(tmp_path)
    first.start(100)
    first.close(120)
    first_paths = set((tmp_path / "recovery").glob("*.json"))
    assert len(first_paths) == 1

    _tamper_valid_run_event(events, b'"validity":"vAlid"')
    second = _manager(tmp_path)
    bootstrap = second.start(200)
    second_paths = set((tmp_path / "recovery").glob("*.json"))
    assert len(second_paths) == 2
    assert first_paths < second_paths
    assert len({_json(path)["finding_fingerprint"] for path in second_paths}) == 2
    assert _unclean_detection_count(bootstrap) == 1


def test_changed_jsonl_issue_type_creates_new_finding(tmp_path: Path) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    original = events.read_bytes()
    events.write_bytes(original + b"partial")
    first = _manager(tmp_path)
    first.start(100)
    first.close(120)
    assert len(list((tmp_path / "recovery").glob("*.json"))) == 1

    events.write_bytes(b"{broken\n" + original)
    second = _manager(tmp_path)
    bootstrap = second.start(200)
    reports = [_json(path) for path in (tmp_path / "recovery").glob("*.json")]
    assert len(reports) == 2
    assert {
        "run_events_trailing_incomplete",
        "run_events_middle_corruption",
    }.issubset({issue for report in reports for issue in report["issue_types"]})
    assert _unclean_detection_count(bootstrap) == 1


def test_same_issue_in_two_runs_has_distinct_finding_fingerprints(
    tmp_path: Path,
) -> None:
    first_owner = _manager(tmp_path)
    first_owner.start(1)
    first_owner.record_competition_context(_context(), 10, 11)
    first_owner.close(20)
    second_owner = _manager(tmp_path)
    second_owner.start(30)
    second_owner.record_competition_context(
        _context(run_id="run-b", fingerprint="fingerprint-b"), 40, 41
    )
    second_owner.close(50)
    _tamper_valid_run_event(tmp_path / "runs" / "run-a" / "events.jsonl")
    _tamper_valid_run_event(tmp_path / "runs" / "run-b" / "events.jsonl")

    recovered = _manager(tmp_path)
    recovered.start(100)
    reports = [_json(path) for path in (tmp_path / "recovery").glob("*.json")]
    assert len(reports) == 2
    assert {report["run_id"] for report in reports} == {"run-a", "run-b"}
    assert len({report["finding_fingerprint"] for report in reports}) == 2
    assert recovered._blocked_run_ids == {"run-a", "run-b"}


def test_bootstrap_and_run_finding_identity_produce_distinct_fingerprints(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    root = manager._prepare_root()
    common = {
        "issue_types": ["active_without_complete"],
        "intrinsic_integrity_issues": ["active_without_complete"],
        "append_compatibility_issues": [],
        "source_artifact_hashes": {"ACTIVE": "0" * 64},
        "segment_manifest_status": "readable",
        "active_marker_present": True,
        "complete_marker_present": False,
        "jsonl": [],
    }
    bootstrap = {
        **common,
        "source_scope": "segment_integrity",
        "source_parent_run_id": None,
        "source_segment_identity": "segment-same",
        "source_segment_path": str(root / "bootstrap" / "segment-same"),
    }
    run_bound = {
        **common,
        "source_scope": "run_integrity",
        "run_id": "run-a",
        "source_parent_run_id": "run-a",
        "source_segment_identity": "segment-same",
        "source_segment_path": str(
            root / "runs" / "run-a" / "segments" / "segment-same"
        ),
    }
    assert manager._recovery_finding_fingerprint(
        root, bootstrap
    ) != manager._recovery_finding_fingerprint(root, run_bound)


@pytest.mark.parametrize("damage", ["symlink", "invalid_json", "fingerprint"])
def test_existing_deterministic_recovery_report_damage_fails_closed(
    damage: str, tmp_path: Path
) -> None:
    _segment, events = _create_closed_unfinished_run(tmp_path)
    _tamper_valid_run_event(events)
    first = _manager(tmp_path)
    first.start(100)
    first.close(120)
    report_path = next((tmp_path / "recovery").glob("*.json"))
    if damage == "symlink":
        outside = tmp_path.parent / f"{tmp_path.name}-outside-report.json"
        outside.write_text('{"outside":true}\n', encoding="utf-8")
        original = outside.read_bytes()
        report_path.unlink()
        try:
            report_path.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("platform does not support file symlinks")
    elif damage == "invalid_json":
        report_path.write_text("{broken\n", encoding="utf-8")
    else:
        report = _json(report_path)
        report["finding_fingerprint"] = "0" * 64
        _persist_json(report_path, report)

    with pytest.raises(RuntimeError, match="Recovery报告"):
        _manager(tmp_path).start(200)
    if damage == "symlink":
        assert outside.read_bytes() == original
