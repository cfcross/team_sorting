"""Recorder Action Dispatch Pairing V1 的纯 Python 严格回归。"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

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
    TwistExactPayload,
    action_dispatch_to_json,
    final_action_to_json,
)
from team_sorting.recording_contracts import (
    FRAME_SCHEMA_NAME,
    FRAME_SCHEMA_VERSION,
    ISSUE_SCHEMA_NAME,
    ISSUE_SCHEMA_VERSION,
    ISSUE_TYPES,
    MAX_ISSUE_DETAIL_CHARS,
    ActionDispatchPairer,
    ActionPairingConfig,
    ActionPairingIssue,
    RecordedActionFrame,
    action_pairing_issue_from_json,
    action_pairing_issue_to_json,
    recorded_action_frame_from_json,
    recorded_action_frame_to_json,
)
from team_sorting.recorder import EpisodeRecorder
from team_sorting.ros_nodes import _build_action_dispatch_record, _create_recorder_node


NOW = 10_000
GROUPS = (
    ("base", "/cmd_vel", "geometry_msgs/msg/Twist"),
    (
        "spine",
        "/spine_forward_position_controller/commands",
        "std_msgs/msg/Float64MultiArray",
    ),
    (
        "head",
        "/head_forward_position_controller/commands",
        "std_msgs/msg/Float64MultiArray",
    ),
    (
        "left_arm",
        "/left_arm_forward_position_controller/commands",
        "std_msgs/msg/Float64MultiArray",
    ),
    (
        "right_arm",
        "/right_arm_forward_position_controller/commands",
        "std_msgs/msg/Float64MultiArray",
    ),
)


def _config(**overrides: object) -> ActionPairingConfig:
    values: dict[str, object] = {
        "enabled": True,
        "max_pending_per_side": 2,
        "max_completed_sequences": 2,
        "max_wait_ns": 100,
        "prune_period_sec": 0.5,
        "raw_payload_preview_chars": 32,
    }
    values.update(overrides)
    return ActionPairingConfig(**values)  # type: ignore[arg-type]


def _action_and_dispatch(
    sequence: int = 1,
    timestamp_ns: int = NOW,
) -> tuple[FinalAction, ActionDispatchRecord]:
    joints = RobotJointState((0.0,) * 17, (0.0,) * 17, (0.0,) * 17, timestamp_ns)
    status = FSMStatus(
        1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", timestamp_ns
    )
    action, decision = ActionMux().compose_with_decision(
        BaseCommand(0.1, 0.2, timestamp_ns, timestamp_ns + 100),
        None,
        joints,
        status,
        timestamp_ns,
    )
    action = replace(action, sequence=sequence, timestamp_ns=timestamp_ns)
    decision = replace(
        decision,
        sequence=sequence,
        final_action_sequence=sequence,
        timestamp_ns=timestamp_ns,
    )
    empty = tuple(
        DispatchGroupRecord(group, topic, message_type, False, None, None)
        for group, topic, message_type in GROUPS
    )
    dispatch = _build_action_dispatch_record(
        action,
        decision,
        publish_enabled=False,
        publisher_created=False,
        dispatch_mode=DispatchMode.NONE,
        group_records=empty,
        failure_reason="observe_only",
    )
    return action, dispatch


def _frame(**overrides: object) -> RecordedActionFrame:
    action, dispatch = _action_and_dispatch()
    values: dict[str, object] = {
        "schema_name": FRAME_SCHEMA_NAME,
        "schema_version": FRAME_SCHEMA_VERSION,
        "sequence": 1,
        "timestamp_ns": NOW,
        "recorder_timestamp_ns": NOW + 1,
        "final_action_received_monotonic_ns": 10,
        "dispatch_received_monotonic_ns": 20,
        "pairing_completed_monotonic_ns": 20,
        "arrival_order": "final_action_first",
        "final_action": action,
        "action_dispatch": dispatch,
        "pairing_status": "paired",
        "controller_accepted": None,
        "execution_confirmed": None,
        "limitations": ("not execution confirmation",),
    }
    values.update(overrides)
    return RecordedActionFrame(**values)  # type: ignore[arg-type]


def _issue(issue_type: str = "missing_action_dispatch", **overrides: object) -> ActionPairingIssue:
    values: dict[str, object] = {
        "schema_name": ISSUE_SCHEMA_NAME,
        "schema_version": ISSUE_SCHEMA_VERSION,
        "issue_type": issue_type,
        "sequence": 1,
        "recorder_timestamp_ns": 20,
        "received_monotonic_ns": 10,
        "side": "final_action",
        "detail": "missing",
        "raw_payload_preview": "{}",
        "existing_digest": "a" * 64,
        "incoming_digest": None,
        "final_action_present": True,
        "dispatch_present": False,
    }
    values.update(overrides)
    return ActionPairingIssue(**values)  # type: ignore[arg-type]


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fail_first_append_to(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    real_append = EpisodeRecorder._append_line
    failed = False

    def _append(path: Path, line: str) -> None:
        nonlocal failed
        if path.name == filename and not failed:
            failed = True
            raise RuntimeError(f"injected append failure: {filename}")
        real_append(path, line)

    monkeypatch.setattr(EpisodeRecorder, "_append_line", staticmethod(_append))


def test_frame_schema_strict_roundtrip_and_structured_children() -> None:
    frame = _frame()
    restored = recorded_action_frame_from_json(recorded_action_frame_to_json(frame))
    assert restored == frame
    payload = json.loads(recorded_action_frame_to_json(frame))
    assert payload["schema_name"] == FRAME_SCHEMA_NAME
    assert payload["schema_version"] == 1
    assert isinstance(payload["final_action"], dict)
    assert isinstance(payload["action_dispatch"], dict)


@pytest.mark.parametrize("field", ["sequence", "timestamp_ns"])
def test_frame_rejects_bool_integer_fields(field: str) -> None:
    with pytest.raises(ValueError):
        _frame(**{field: True})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_frame_json_rejects_nonfinite_or_bool_nested_action(value: object) -> None:
    payload = json.loads(recorded_action_frame_to_json(_frame()))
    payload["final_action"]["action"][0] = value
    raw = json.dumps(payload, allow_nan=True)
    with pytest.raises(ValueError):
        recorded_action_frame_from_json(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2), ("controller_accepted", True), ("execution_confirmed", False)],
)
def test_frame_rejects_unknown_version_or_confirmation(field: str, value: object) -> None:
    payload = json.loads(recorded_action_frame_to_json(_frame()))
    payload[field] = value
    with pytest.raises(ValueError):
        recorded_action_frame_from_json(json.dumps(payload))


def test_frame_rejects_unknown_field() -> None:
    payload = json.loads(recorded_action_frame_to_json(_frame()))
    payload["training_eligible"] = True
    with pytest.raises(ValueError, match="unknown"):
        recorded_action_frame_from_json(json.dumps(payload))


def test_frame_rejects_sequence_four_layer_mismatch() -> None:
    action, _dispatch = _action_and_dispatch(sequence=2)
    with pytest.raises(ValueError, match="sequence"):
        _frame(final_action=action)


def test_frame_rejects_timestamp_three_layer_mismatch() -> None:
    action, _dispatch = _action_and_dispatch(timestamp_ns=NOW + 1)
    with pytest.raises(ValueError, match="timestamp"):
        _frame(final_action=action)


@pytest.mark.parametrize("issue_type", sorted(ISSUE_TYPES))
def test_every_controlled_issue_type_roundtrips(issue_type: str) -> None:
    issue = _issue(issue_type)
    assert action_pairing_issue_from_json(action_pairing_issue_to_json(issue)) == issue


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("issue_type", "other"),
        ("sequence", True),
        ("final_action_present", 1),
        ("existing_digest", "short"),
    ],
)
def test_issue_rejects_invalid_contract_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _issue(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", 1),
        ("max_pending_per_side", True),
        ("max_completed_sequences", 0),
        ("max_wait_ns", -1),
        ("prune_period_sec", float("nan")),
        ("raw_payload_preview_chars", False),
    ],
)
def test_pairing_config_strict_validation(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _config(**{field: value})


def test_pairing_config_mapping_rejects_unknown_and_missing_fields() -> None:
    raw = _config().__dict__.copy()
    raw["unknown"] = 1
    with pytest.raises(ValueError, match="unknown"):
        ActionPairingConfig.from_mapping(raw)
    del raw["unknown"]
    del raw["max_wait_ns"]
    with pytest.raises(ValueError, match="missing"):
        ActionPairingConfig.from_mapping(raw)


@pytest.mark.parametrize(
    ("first", "expected_order"),
    [("final", "final_action_first"), ("dispatch", "dispatch_first")],
)
def test_pairer_supports_both_arrival_orders(first: str, expected_order: str) -> None:
    action, dispatch = _action_and_dispatch()
    pairer = ActionDispatchPairer(_config())
    if first == "final":
        first_outcome = pairer.add_final_action(action, final_action_to_json(action), 1, 10)
        outcome = pairer.add_dispatch(dispatch, action_dispatch_to_json(dispatch), 2, 20)
    else:
        first_outcome = pairer.add_dispatch(dispatch, action_dispatch_to_json(dispatch), 1, 10)
        outcome = pairer.add_final_action(action, final_action_to_json(action), 2, 20)
    assert first_outcome.frames == ()
    assert outcome.frames[0].arrival_order == expected_order
    assert outcome.frames[0].action_dispatch.dispatch_mode is DispatchMode.NONE


def test_timestamp_mismatch_never_forms_frame() -> None:
    action, _dispatch = _action_and_dispatch(timestamp_ns=NOW)
    _other_action, dispatch = _action_and_dispatch(timestamp_ns=NOW + 1)
    dispatch = replace(
        dispatch,
        sequence=action.sequence,
        final_action_sequence=action.sequence,
        decision=replace(
            dispatch.decision,
            sequence=action.sequence,
            final_action_sequence=action.sequence,
        ),
    )
    pairer = ActionDispatchPairer(_config())
    pairer.add_final_action(action, final_action_to_json(action), 1, 10)
    outcome = pairer.add_dispatch(dispatch, action_dispatch_to_json(dispatch), 2, 20)
    assert outcome.frames == ()
    assert outcome.issues[0].issue_type == "timestamp_mismatch"


def test_identical_and_conflicting_pending_duplicates_preserve_first() -> None:
    action, dispatch = _action_and_dispatch()
    pairer = ActionDispatchPairer(_config())
    pairer.add_final_action(action, final_action_to_json(action), 1, 10)
    identical = pairer.add_final_action(action, final_action_to_json(action), 2, 11)
    changed = replace(action, values=(0.2, *action.values[1:]))
    conflict = pairer.add_final_action(changed, final_action_to_json(changed), 3, 12)
    paired = pairer.add_dispatch(dispatch, action_dispatch_to_json(dispatch), 4, 13)
    assert identical.persist_raw is conflict.persist_raw is False
    assert identical.issues[0].issue_type == "duplicate_identical_final_action"
    assert conflict.issues[0].issue_type == "duplicate_conflicting_final_action"
    assert paired.frames[0].final_action == action


def test_dispatch_duplicate_and_late_duplicate_do_not_make_second_frame() -> None:
    action, dispatch = _action_and_dispatch()
    pairer = ActionDispatchPairer(_config())
    pairer.add_dispatch(dispatch, action_dispatch_to_json(dispatch), 1, 10)
    duplicate = pairer.add_dispatch(dispatch, action_dispatch_to_json(dispatch), 2, 11)
    paired = pairer.add_final_action(action, final_action_to_json(action), 3, 12)
    late = pairer.add_dispatch(dispatch, action_dispatch_to_json(dispatch), 4, 13)
    assert duplicate.issues[0].issue_type == "duplicate_identical_dispatch"
    assert len(paired.frames) == 1
    assert late.frames == ()
    assert late.issues[0].issue_type == "late_duplicate_dispatch"


def test_conflicting_dispatch_does_not_replace_first_candidate() -> None:
    action, dispatch = _action_and_dispatch()
    pairer = ActionDispatchPairer(_config())
    pairer.add_dispatch(dispatch, action_dispatch_to_json(dispatch), 1, 10)
    conflict = replace(dispatch, failure_reason="different diagnostic")
    duplicate = pairer.add_dispatch(conflict, action_dispatch_to_json(conflict), 2, 11)
    paired = pairer.add_final_action(action, final_action_to_json(action), 3, 12)
    assert duplicate.issues[0].issue_type == "duplicate_conflicting_dispatch"
    assert duplicate.issues[0].existing_digest != duplicate.issues[0].incoming_digest
    assert paired.frames[0].action_dispatch == dispatch


def test_head_only_shadow_payload_is_preserved_in_frame() -> None:
    action, none_dispatch = _action_and_dispatch()
    records = list(none_dispatch.group_records)
    records[2] = replace(
        records[2],
        attempted=True,
        succeeded=True,
        exact_payload=Float64MultiArrayExactPayload((action.values[3], -0.4)),
    )
    dispatch = _build_action_dispatch_record(
        action,
        none_dispatch.decision,
        publish_enabled=True,
        publisher_created=True,
        dispatch_mode=DispatchMode.HEAD_ONLY,
        group_records=tuple(records),
    )
    pairer = ActionDispatchPairer(_config())
    pairer.add_final_action(action, final_action_to_json(action), 1, 10)
    frame = pairer.add_dispatch(dispatch, action_dispatch_to_json(dispatch), 2, 20).frames[0]
    payload = frame.action_dispatch.group_records[2].exact_payload
    assert isinstance(payload, Float64MultiArrayExactPayload)
    assert payload.data == (action.values[3], -0.4)


def test_full_partial_failure_is_preserved_in_frame() -> None:
    action, none_dispatch = _action_and_dispatch()
    records = list(none_dispatch.group_records)
    records[0] = replace(
        records[0],
        attempted=True,
        succeeded=True,
        exact_payload=TwistExactPayload(
            (action.values[0], 0.0, 0.0), (0.0, 0.0, action.values[1])
        ),
    )
    records[1] = replace(
        records[1],
        attempted=True,
        succeeded=False,
        exact_payload=Float64MultiArrayExactPayload((action.values[2],)),
        failure_reason="injected failure",
    )
    dispatch = _build_action_dispatch_record(
        action,
        none_dispatch.decision,
        publish_enabled=True,
        publisher_created=True,
        dispatch_mode=DispatchMode.FULL,
        group_records=tuple(records),
        failure_reason="spine publish failed",
    )
    pairer = ActionDispatchPairer(_config())
    pairer.add_dispatch(dispatch, action_dispatch_to_json(dispatch), 1, 10)
    frame = pairer.add_final_action(action, final_action_to_json(action), 2, 20).frames[0]
    assert frame.action_dispatch.publisher_call_succeeded is False
    assert frame.action_dispatch.failed_groups == ("spine",)
    assert frame.controller_accepted is frame.execution_confirmed is None


def test_capacity_evicts_oldest_receive_monotonic_not_sequence() -> None:
    pairer = ActionDispatchPairer(_config(max_pending_per_side=2))
    for sequence, mono in ((9, 30), (2, 10), (5, 20)):
        action, _ = _action_and_dispatch(sequence=sequence)
        outcome = pairer.add_final_action(action, final_action_to_json(action), mono, mono)
    assert outcome.issues[0].issue_type == "pending_capacity_eviction"
    assert outcome.issues[0].sequence == 2
    assert set(pairer.pending_final_actions) == {5, 9}


def test_prune_hits_exact_wait_boundary_without_new_message() -> None:
    action, _ = _action_and_dispatch()
    pairer = ActionDispatchPairer(_config(max_wait_ns=100))
    pairer.add_final_action(action, final_action_to_json(action), 1, 50)
    assert pairer.prune(2, 149).issues == ()
    outcome = pairer.prune(3, 150)
    assert outcome.issues[0].issue_type == "pending_age_timeout"
    assert "action_dispatch" in outcome.issues[0].detail


def test_shutdown_orphans_all_pending_in_deterministic_order_and_is_idempotent() -> None:
    pairer = ActionDispatchPairer(_config())
    action2, _ = _action_and_dispatch(sequence=2)
    _, dispatch1 = _action_and_dispatch(sequence=1)
    pairer.add_final_action(action2, final_action_to_json(action2), 1, 20)
    pairer.add_dispatch(dispatch1, action_dispatch_to_json(dispatch1), 1, 10)
    outcome = pairer.close(2, 30)
    assert [issue.sequence for issue in outcome.issues] == [1, 2]
    assert all(issue.issue_type == "shutdown_orphan" for issue in outcome.issues)
    assert pairer.close(3, 40) == replace(outcome, issues=())


def test_completed_history_is_bounded_lru() -> None:
    pairer = ActionDispatchPairer(_config(max_completed_sequences=2))
    for sequence in (1, 2, 3):
        action, dispatch = _action_and_dispatch(sequence=sequence)
        pairer.add_final_action(action, final_action_to_json(action), sequence, sequence * 10)
        pairer.add_dispatch(
            dispatch, action_dispatch_to_json(dispatch), sequence, sequence * 10 + 1
        )
    assert pairer.completed_sequences == (2, 3)


def test_invalid_payload_preview_is_bounded_and_digest_is_stable() -> None:
    pairer = ActionDispatchPairer(_config(raw_payload_preview_chars=5))
    first = pairer.invalid_payload("final_action", "abcdefgh", "bad", 1, 2).issues[0]
    second = pairer.invalid_payload("final_action", "abcdefgh", "bad", 1, 2).issues[0]
    assert first.raw_payload_preview == "abcde"
    assert first.incoming_digest == second.incoming_digest


def test_recorder_writes_four_layers_and_keeps_final_action_wire_compatible(
    tmp_path: Path,
) -> None:
    recorder = EpisodeRecorder(tmp_path, _config())
    episode = recorder.start("pair", 0, "test")
    action, dispatch = _action_and_dispatch()
    original_final = final_action_to_json(action)
    recorder.ingest_final_action_payload(original_final, 1, 10)
    recorder.ingest_action_dispatch_payload(action_dispatch_to_json(dispatch), 2, 20)
    recorder.finish(3)
    assert (episode / "final_actions.jsonl").read_text(encoding="utf-8") == original_final + "\n"
    assert len(_json_lines(episode / "action_dispatches.jsonl")) == 1
    assert len(_json_lines(episode / "action_frames.jsonl")) == 1
    assert _json_lines(episode / "action_pairing_issues.jsonl") == []


def test_recorder_invalid_and_duplicate_messages_only_write_issues(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path, _config())
    episode = recorder.start("issues", 0, "test")
    action, dispatch = _action_and_dispatch()
    recorder.ingest_final_action_payload("not-json", 1, 1)
    recorder.ingest_action_dispatch_payload("{}", 2, 2)
    recorder.ingest_final_action_payload(final_action_to_json(action), 3, 3)
    recorder.ingest_final_action_payload(final_action_to_json(action), 4, 4)
    recorder.ingest_action_dispatch_payload(action_dispatch_to_json(dispatch), 5, 5)
    recorder.ingest_action_dispatch_payload(action_dispatch_to_json(dispatch), 6, 6)
    recorder.finish(7)
    types = [row["issue_type"] for row in _json_lines(episode / "action_pairing_issues.jsonl")]
    assert types == [
        "invalid_final_action_json",
        "invalid_action_dispatch_json",
        "duplicate_identical_final_action",
        "late_duplicate_dispatch",
    ]
    assert len(_json_lines(episode / "final_actions.jsonl")) == 1
    assert len(_json_lines(episode / "action_dispatches.jsonl")) == 1
    assert len(_json_lines(episode / "action_frames.jsonl")) == 1


def test_recorder_rejects_duplicate_dispatch_json_keys(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path, _config())
    episode = recorder.start("duplicate_json_key", 0, "test")
    _action, dispatch = _action_and_dispatch()
    raw = action_dispatch_to_json(dispatch).replace(
        '"schema_version":1,', '"schema_version":1,"schema_version":1,', 1
    )
    assert recorder.ingest_action_dispatch_payload(raw, 1, 1) == (
        "invalid_action_dispatch_json",
    )
    recorder.finish(2)
    assert not (episode / "action_dispatches.jsonl").exists()


def test_recorder_finish_flushes_shutdown_orphan_and_repeated_close_is_safe(
    tmp_path: Path,
) -> None:
    recorder = EpisodeRecorder(tmp_path, _config())
    episode = recorder.start("orphan", 0, "test")
    action, _ = _action_and_dispatch()
    recorder.ingest_final_action_payload(final_action_to_json(action), 1, 1)
    assert recorder.close_action_pairing(2, 2) == ("shutdown_orphan",)
    assert recorder.close_action_pairing(3, 3) == ()
    recorder.finish(4)
    assert _json_lines(episode / "action_pairing_issues.jsonl")[0]["issue_type"] == "shutdown_orphan"


def test_recorder_can_start_a_new_episode_with_fresh_pairing_state(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path, _config())
    action, dispatch = _action_and_dispatch()
    first = recorder.start("first", 0, "test")
    recorder.ingest_final_action_payload(final_action_to_json(action), 1, 1)
    recorder.finish(2)
    second = recorder.start("second", 3, "test")
    recorder.ingest_final_action_payload(final_action_to_json(action), 4, 4)
    recorder.ingest_action_dispatch_payload(action_dispatch_to_json(dispatch), 5, 5)
    recorder.finish(6)
    assert len(_json_lines(first / "action_pairing_issues.jsonl")) == 1
    assert len(_json_lines(second / "action_frames.jsonl")) == 1


def test_pairing_disabled_preserves_old_recorder_and_creates_no_new_files(
    tmp_path: Path,
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    episode = recorder.start("disabled", 0, "test")
    action, _ = _action_and_dispatch()
    recorder.record_final_action(action)
    recorder.finish(1)
    assert (episode / "final_actions.jsonl").exists()
    assert not (episode / "action_dispatches.jsonl").exists()
    assert not (episode / "action_frames.jsonl").exists()
    assert not (episode / "action_pairing_issues.jsonl").exists()


class _Logger:
    def error(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass


class _Timer:
    def __init__(self, callback: object) -> None:
        self.callback = callback
        self.cancel_count = 0

    def cancel(self) -> None:
        self.cancel_count += 1


class _Node:
    def __init__(self, _name: str) -> None:
        self.subscriptions: dict[str, object] = {}
        self.timers: list[_Timer] = []
        self.destroy_count = 0

    def declare_parameter(self, _name: str, value: object) -> None:
        self.enabled = value

    def get_parameter(self, _name: str) -> SimpleNamespace:
        return SimpleNamespace(value=self.enabled)

    def get_clock(self) -> SimpleNamespace:
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=NOW))

    def create_subscription(
        self, _kind: object, topic: str, callback: object, _depth: int
    ) -> object:
        self.subscriptions[topic] = callback
        return object()

    def create_timer(self, _period: float, callback: object) -> _Timer:
        timer = _Timer(callback)
        self.timers.append(timer)
        return timer

    def get_logger(self) -> _Logger:
        return _Logger()

    def destroy_node(self) -> str:
        self.destroy_count += 1
        return "destroyed"


def _node_config(tmp_path: Path, enabled: bool = True) -> dict[str, Any]:
    loaded = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    loaded["recorder"]["enabled"] = enabled
    loaded["recorder"]["record_rosbag"] = False
    loaded["recorder"]["root_dir"] = str(tmp_path)
    return loaded


def test_recorder_node_subscribes_configured_dispatch_and_has_independent_timer(
    tmp_path: Path,
) -> None:
    ros = SimpleNamespace(Node=_Node, String=object, Int32=object)
    node = _create_recorder_node(ros)(_node_config(tmp_path), ros)
    assert "/team/final_action" in node.subscriptions
    assert "/team/action_dispatch" in node.subscriptions
    assert len(node.timers) == 1
    timer = node.timers[0]
    assert node.destroy_node() == "destroyed"
    assert timer.cancel_count == 1
    assert node.destroy_node() is None


def test_recorder_node_disabled_creates_no_directory_subscription_or_timer(
    tmp_path: Path,
) -> None:
    ros = SimpleNamespace(Node=_Node, String=object, Int32=object)
    with pytest.raises(RuntimeError, match="未启用"):
        _create_recorder_node(ros)(_node_config(tmp_path, enabled=False), ros)
    assert list(tmp_path.iterdir()) == []


def test_config_keeps_action_dispatch_out_of_rosbag_and_enables_pairing() -> None:
    loaded = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    assert loaded["recorder"]["action_pairing"]["enabled"] is True
    assert "/team/action_dispatch" not in loaded["recorder"]["rosbag_topics"]


def test_pairing_module_has_no_control_dependencies_or_eight_to_nineteen_mapping() -> None:
    source = Path("team_sorting/recording_contracts.py").read_text(encoding="utf-8")
    assert "OfficialCommandPublisher" not in source
    assert "ActionMux(" not in source
    assert "rclpy" not in source
    assert "8→19" not in source and "8->19" not in source


def test_final_raw_append_failure_retries_without_becoming_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path, _config())
    episode = recorder.start("final_raw_retry", 0, "test")
    action, _ = _action_and_dispatch()
    raw = final_action_to_json(action)
    _fail_first_append_to(monkeypatch, "final_actions.jsonl")
    with pytest.raises(RuntimeError, match="injected append failure"):
        recorder.ingest_final_action_payload(raw, 1, 1)
    assert recorder.ingest_final_action_payload(raw, 2, 2) == ()
    recorder.finish(3)
    assert len(_json_lines(episode / "final_actions.jsonl")) == 1
    assert [
        item["issue_type"]
        for item in _json_lines(episode / "action_pairing_issues.jsonl")
    ] == ["shutdown_orphan"]


def test_dispatch_raw_append_failure_retries_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path, _config())
    episode = recorder.start("dispatch_raw_retry", 0, "test")
    _action, dispatch = _action_and_dispatch()
    raw = action_dispatch_to_json(dispatch)
    _fail_first_append_to(monkeypatch, "action_dispatches.jsonl")
    with pytest.raises(RuntimeError, match="injected append failure"):
        recorder.ingest_action_dispatch_payload(raw, 1, 1)
    assert recorder.ingest_action_dispatch_payload(raw, 2, 2) == ()
    recorder.finish(3)
    assert len(_json_lines(episode / "action_dispatches.jsonl")) == 1


def test_frame_append_failure_resumes_without_repeating_dispatch_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path, _config())
    episode = recorder.start("frame_retry", 0, "test")
    action, dispatch = _action_and_dispatch()
    recorder.ingest_final_action_payload(final_action_to_json(action), 1, 10)
    _fail_first_append_to(monkeypatch, "action_frames.jsonl")
    dispatch_raw = action_dispatch_to_json(dispatch)
    with pytest.raises(RuntimeError, match="injected append failure"):
        recorder.ingest_action_dispatch_payload(dispatch_raw, 2, 20)
    assert recorder._action_pairer is not None
    assert recorder._action_pairer.completed_sequences == ()
    assert recorder.ingest_action_dispatch_payload(dispatch_raw, 3, 30) == ()
    assert recorder._action_pairer.completed_sequences == (1,)
    recorder.finish(4)
    assert len(_json_lines(episode / "action_dispatches.jsonl")) == 1
    assert len(_json_lines(episode / "action_frames.jsonl")) == 1
    assert not (episode / "action_pairing_issues.jsonl").exists()


def test_conflict_issue_append_failure_preserves_first_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path, _config())
    episode = recorder.start("conflict_retry", 0, "test")
    action, dispatch = _action_and_dispatch()
    recorder.ingest_final_action_payload(final_action_to_json(action), 1, 10)
    changed = replace(action, values=(0.2, *action.values[1:]))
    changed_raw = final_action_to_json(changed)
    _fail_first_append_to(monkeypatch, "action_pairing_issues.jsonl")
    with pytest.raises(RuntimeError, match="injected append failure"):
        recorder.ingest_final_action_payload(changed_raw, 2, 20)
    assert recorder.ingest_final_action_payload(changed_raw, 3, 30) == (
        "duplicate_conflicting_final_action",
    )
    recorder.ingest_action_dispatch_payload(action_dispatch_to_json(dispatch), 4, 40)
    recorder.finish(5)
    assert len(_json_lines(episode / "final_actions.jsonl")) == 1
    assert len(_json_lines(episode / "action_frames.jsonl")) == 1
    issues = _json_lines(episode / "action_pairing_issues.jsonl")
    assert [item["issue_type"] for item in issues] == [
        "duplicate_conflicting_final_action"
    ]


def test_capacity_issue_failure_keeps_pending_until_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path, _config(max_pending_per_side=1))
    episode = recorder.start("capacity_retry", 0, "test")
    action1, _ = _action_and_dispatch(sequence=1)
    action2, _ = _action_and_dispatch(sequence=2)
    recorder.ingest_final_action_payload(final_action_to_json(action1), 1, 10)
    _fail_first_append_to(monkeypatch, "action_pairing_issues.jsonl")
    with pytest.raises(RuntimeError, match="injected append failure"):
        recorder.ingest_final_action_payload(final_action_to_json(action2), 2, 20)
    assert recorder._action_pairer is not None
    assert set(recorder._action_pairer.pending_final_actions) == {1, 2}
    assert recorder.ingest_final_action_payload(final_action_to_json(action2), 3, 30) == (
        "pending_capacity_eviction",
    )
    assert set(recorder._action_pairer.pending_final_actions) == {2}
    recorder.finish(4)
    assert len(_json_lines(episode / "final_actions.jsonl")) == 2
    issue_types = [
        item["issue_type"]
        for item in _json_lines(episode / "action_pairing_issues.jsonl")
    ]
    assert issue_types == ["pending_capacity_eviction", "shutdown_orphan"]


def test_prune_issue_failure_does_not_delete_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path, _config(max_wait_ns=10))
    episode = recorder.start("prune_retry", 0, "test")
    action, _ = _action_and_dispatch()
    recorder.ingest_final_action_payload(final_action_to_json(action), 1, 1)
    _fail_first_append_to(monkeypatch, "action_pairing_issues.jsonl")
    with pytest.raises(RuntimeError, match="injected append failure"):
        recorder.prune_action_pairs(2, 11)
    assert recorder._action_pairer is not None
    assert 1 in recorder._action_pairer.pending_final_actions
    assert recorder.prune_action_pairs(3, 12) == ("pending_age_timeout",)
    assert recorder._action_pairer.pending_final_actions == {}
    recorder.finish(4)
    assert [
        item["issue_type"]
        for item in _json_lines(episode / "action_pairing_issues.jsonl")
    ] == ["pending_age_timeout"]


def test_shutdown_issue_failure_is_retryable_and_close_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path, _config())
    episode = recorder.start("shutdown_retry", 0, "test")
    action, _ = _action_and_dispatch()
    recorder.ingest_final_action_payload(final_action_to_json(action), 1, 1)
    _fail_first_append_to(monkeypatch, "action_pairing_issues.jsonl")
    with pytest.raises(RuntimeError, match="injected append failure"):
        recorder.close_action_pairing(2, 2)
    assert recorder._action_pairer is not None
    assert 1 in recorder._action_pairer.pending_final_actions
    assert recorder.close_action_pairing(3, 3) == ("shutdown_orphan",)
    assert recorder.close_action_pairing(4, 4) == ()
    recorder.finish(5)
    assert [
        item["issue_type"]
        for item in _json_lines(episode / "action_pairing_issues.jsonl")
    ] == ["shutdown_orphan"]


def test_terminal_ledger_blocks_replay_after_recent_digest_lru_eviction(
    tmp_path: Path,
) -> None:
    recorder = EpisodeRecorder(tmp_path, _config(max_completed_sequences=1))
    episode = recorder.start("terminal_ledger", 0, "test")
    messages: dict[int, tuple[str, str]] = {}
    for sequence in (1, 2):
        action, dispatch = _action_and_dispatch(sequence=sequence)
        messages[sequence] = (
            final_action_to_json(action),
            action_dispatch_to_json(dispatch),
        )
        recorder.ingest_final_action_payload(messages[sequence][0], sequence, sequence * 10)
        recorder.ingest_action_dispatch_payload(
            messages[sequence][1], sequence, sequence * 10 + 1
        )
    assert recorder._action_pairer is not None
    assert recorder._action_pairer.completed_sequences == (2,)
    assert recorder._action_pairer.terminal_sequence_ranges == ((1, 2),)
    assert recorder.ingest_final_action_payload(messages[1][0], 3, 30) == (
        "late_duplicate_final_action",
    )
    assert recorder.ingest_action_dispatch_payload(messages[1][1], 4, 40) == (
        "late_duplicate_dispatch",
    )
    recorder.finish(5)
    assert len(_json_lines(episode / "action_frames.jsonl")) == 2
    issues = _json_lines(episode / "action_pairing_issues.jsonl")
    assert len(issues) == 2
    assert all("近期digest已淘汰" in item["detail"] for item in issues)


def test_huge_unknown_field_error_has_bounded_preview_and_detail_and_roundtrips(
    tmp_path: Path,
) -> None:
    recorder = EpisodeRecorder(
        tmp_path, _config(raw_payload_preview_chars=128)
    )
    episode = recorder.start("bounded_error", 0, "test")
    payload = {f"unknown_{index:05d}_" + "x" * 100: index for index in range(5_000)}
    raw = json.dumps(payload)
    assert recorder.ingest_final_action_payload(raw, 1, 1) == (
        "invalid_final_action_json",
    )
    recorder.finish(2)
    issue_payload = _json_lines(episode / "action_pairing_issues.jsonl")[0]
    assert len(issue_payload["raw_payload_preview"]) == 128
    assert len(issue_payload["detail"]) <= MAX_ISSUE_DETAIL_CHARS
    assert "unknown_count=5000" in issue_payload["detail"]
    assert "unknown_truncated=true" in issue_payload["detail"]
    issue = action_pairing_issue_from_json(json.dumps(issue_payload))
    assert action_pairing_issue_from_json(action_pairing_issue_to_json(issue)) == issue


def test_issue_contract_rejects_detail_over_hard_limit() -> None:
    with pytest.raises(ValueError, match="detail"):
        _issue(detail="x" * (MAX_ISSUE_DETAIL_CHARS + 1))
