"""ActionMux Decision V1 与精确 publisher dispatch 遥测回归。"""

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
    CandidateDisposition,
    DispatchMode,
    FSMStatus,
    FinalAction,
    Float64MultiArrayExactPayload,
    GlobalPhase,
    LocalPhase,
    ManipulationCommand,
    RobotJointState,
    TwistExactPayload,
    action_dispatch_from_json,
    action_dispatch_to_json,
)
from team_sorting.ros_nodes import (
    OfficialCommandPublisher,
    OfficialPublishError,
    _build_action_dispatch_record,
)


NOW = 10_000
OFFICIAL_TOPICS = {
    "cmd_vel": "/cmd_vel",
    "slide": "/spine_forward_position_controller/commands",
    "head": "/head_forward_position_controller/commands",
    "left_arm": "/left_arm_forward_position_controller/commands",
    "right_arm": "/right_arm_forward_position_controller/commands",
}


def _joints(position: tuple[float, ...] = (0.0,) * 17, *, valid: bool = True) -> RobotJointState:
    return RobotJointState(
        position=position,
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=NOW,
        valid=valid,
        failure_reason="bad_feedback" if not valid else "",
    )


def _status(phase: GlobalPhase = GlobalPhase.SEARCH_TARGET) -> FSMStatus:
    return FSMStatus(1, phase, LocalPhase.IDLE, 0, False, "", NOW)


def _base(*, valid: bool = True, expiry: int = NOW + 100, v: float = 0.0, w: float = 0.0) -> BaseCommand:
    return BaseCommand(v, w, NOW, expiry, valid, "invalid_base" if not valid else "")


def _manipulation(
    indices: tuple[int, ...],
    *,
    values: dict[int, float] | None = None,
    valid: bool = True,
    expiry: int = NOW + 100,
) -> ManipulationCommand:
    target = [0.0] * 17
    for index, value in (values or {}).items():
        target[index] = value
    return ManipulationCommand(
        tuple(target),
        tuple(index in indices for index in range(17)),
        LocalPhase.IDLE,
        NOW,
        expiry,
        valid,
        "invalid_manipulation" if not valid else "",
    )


def test_explicit_zero_base_is_requested_and_commanded_but_absent_is_not() -> None:
    _action, decision = ActionMux().compose_with_decision(
        _base(), None, _joints(), _status(), NOW
    )
    assert decision.requested_mask[:2] == (True, True)
    assert decision.commanded_mask[:2] == (True, True)
    assert decision.base_disposition is CandidateDisposition.ACCEPTED
    assert not any(decision.commanded_mask[2:])

    _action, absent = ActionMux().compose_with_decision(
        None, None, _joints(), _status(), NOW
    )
    assert absent.requested_mask[:2] == (False, False)
    assert absent.commanded_mask[:2] == (False, False)
    assert absent.base_disposition is CandidateDisposition.ABSENT
    assert absent.base_source == "none"


def test_manipulation_mask_is_preserved_without_turning_feedback_into_commands() -> None:
    positions = list(_joints().position)
    positions[4] = 0.25
    command = _manipulation((1,), values={1: 0.1})
    action, decision = ActionMux().compose_with_decision(
        None, command, _joints(tuple(positions)), _status(), NOW
    )

    assert decision.requested_mask[2:] == command.controlled_mask
    assert decision.commanded_mask[3] is True
    assert sum(decision.commanded_mask) == 1
    assert action.values[6] == 0.25
    assert decision.commanded_mask[6] is False


@pytest.mark.parametrize(
    ("base", "manipulation", "base_disposition", "manipulation_disposition"),
    [
        (
            _base(valid=False),
            _manipulation((1,), valid=False),
            CandidateDisposition.REJECTED_INVALID,
            CandidateDisposition.REJECTED_INVALID,
        ),
        (
            _base(expiry=NOW),
            _manipulation((1,), expiry=NOW),
            CandidateDisposition.REJECTED_STALE,
            CandidateDisposition.REJECTED_STALE,
        ),
    ],
)
def test_invalid_and_stale_candidates_are_rejected_without_commanded_dimensions(
    base: BaseCommand,
    manipulation: ManipulationCommand,
    base_disposition: CandidateDisposition,
    manipulation_disposition: CandidateDisposition,
) -> None:
    _action, decision = ActionMux().compose_with_decision(
        base, manipulation, _joints(), _status(), NOW
    )
    assert not any(decision.commanded_mask)
    assert decision.base_disposition is base_disposition
    assert decision.manipulation_disposition is manipulation_disposition
    assert decision.safety_override_mask[:2] == (True, True)
    assert decision.safety_override_mask[3] is True


def test_fsm_stop_overrides_all_output_dimensions_and_candidate_acceptance() -> None:
    _action, decision = ActionMux().compose_with_decision(
        _base(v=0.1),
        _manipulation((1,), values={1: 0.1}),
        _joints(),
        _status(GlobalPhase.SAFE_HOLD),
        NOW,
    )
    assert not any(decision.commanded_mask)
    assert all(decision.safety_override_mask)
    assert decision.base_disposition is CandidateDisposition.SAFETY_OVERRIDDEN
    assert decision.manipulation_disposition is CandidateDisposition.SAFETY_OVERRIDDEN


def test_clipped_mask_is_per_dimension_and_command_remains_accepted() -> None:
    action, decision = ActionMux().compose_with_decision(
        _base(v=2.0, w=0.1),
        _manipulation((3,), values={3: 9.0}),
        _joints(),
        _status(),
        NOW,
    )
    assert decision.commanded_mask[0:2] == (True, True)
    assert decision.commanded_mask[5] is True
    assert decision.clipped_mask[0] is True
    assert decision.clipped_mask[1] is False
    assert decision.clipped_mask[5] is True
    assert sum(decision.clipped_mask) == 2
    assert action.clipped is True


def test_compose_remains_compatible_with_compose_with_decision() -> None:
    args = (_base(v=0.1), _manipulation((1,), values={1: 0.1}), _joints(), _status(), NOW)
    old_action = ActionMux().compose(*args)
    new_action, decision = ActionMux().compose_with_decision(*args)
    assert old_action == new_action
    assert decision.final_action_sequence == new_action.sequence
    assert len(decision.requested_mask) == len(decision.commanded_mask) == 19
    assert len(decision.clipped_mask) == len(decision.safety_override_mask) == 19


@pytest.mark.parametrize(
    "mutation",
    [
        {"base_source": "none"},
        {"base_disposition": CandidateDisposition.ABSENT},
        {"base_disposition": CandidateDisposition.REJECTED_INVALID},
        {"commanded_mask": (False,) * 19},
    ],
)
def test_present_base_candidate_rejects_inconsistent_source_disposition_or_masks(
    mutation: dict[str, object],
) -> None:
    _action, decision = ActionMux().compose_with_decision(
        _base(), None, _joints(), _status(), NOW
    )
    with pytest.raises(ValueError):
        replace(decision, **mutation)


def test_absent_base_candidate_rejects_accepted_disposition() -> None:
    _action, decision = ActionMux().compose_with_decision(
        None, None, _joints(), _status(), NOW
    )
    with pytest.raises(ValueError, match="base_disposition"):
        replace(decision, base_disposition=CandidateDisposition.ACCEPTED)


@pytest.mark.parametrize(
    "mutation",
    [
        {"manipulation_source": "none"},
        {"manipulation_disposition": CandidateDisposition.ABSENT},
        {"manipulation_disposition": CandidateDisposition.REJECTED_STALE},
        {"commanded_mask": (False,) * 19},
    ],
)
def test_present_manipulation_candidate_rejects_inconsistent_contract(
    mutation: dict[str, object],
) -> None:
    _action, decision = ActionMux().compose_with_decision(
        None,
        _manipulation((1,), values={1: 0.1}),
        _joints(),
        _status(),
        NOW,
    )
    with pytest.raises(ValueError):
        replace(decision, **mutation)


def test_partially_accepted_requires_a_strict_nonempty_subset_of_requested() -> None:
    _action, decision = ActionMux().compose_with_decision(
        None,
        _manipulation((1, 2), values={1: 0.1, 2: 0.1}),
        _joints(),
        _status(),
        NOW,
    )
    partial_mask = list(decision.commanded_mask)
    partial_mask[4] = False
    partial = replace(
        decision,
        commanded_mask=tuple(partial_mask),
        manipulation_disposition=CandidateDisposition.PARTIALLY_ACCEPTED,
    )
    assert partial.commanded_mask[3:5] == (True, False)
    with pytest.raises(ValueError, match="partially_accepted"):
        replace(
            decision,
            commanded_mask=(False,) * 19,
            manipulation_disposition=CandidateDisposition.PARTIALLY_ACCEPTED,
        )


class _Vector:
    def __init__(self) -> None:
        self.x = self.y = self.z = 0.0


class _Twist:
    def __init__(self) -> None:
        self.linear = _Vector()
        self.angular = _Vector()


class _Array:
    def __init__(self) -> None:
        self.data: Any = []


class _Publisher:
    def __init__(self, topic: str, node: "_Node") -> None:
        self.topic = topic
        self.node = node
        self.messages: list[Any] = []

    def publish(self, message: Any) -> None:
        self.node.calls.append(self.topic)
        if self.node.fail_topic == self.topic:
            raise RuntimeError("injected_failure")
        self.messages.append(message)


class _Node:
    def __init__(self) -> None:
        self.publishers: dict[str, _Publisher] = {}
        self.calls: list[str] = []
        self.fail_topic: str | None = None

    def create_publisher(self, _type: object, topic: str, _depth: int) -> _Publisher:
        publisher = _Publisher(topic, self)
        self.publishers[topic] = publisher
        return publisher

    def count_publishers(self, topic: str) -> int:
        return int(topic in self.publishers)


def _publisher(*, tracking: bool = False) -> tuple[OfficialCommandPublisher, _Node]:
    node = _Node()
    config = {
        "enabled": tracking,
        "fresh_reset_confirmed": tracking,
        "initial_yaw_target": 0.0,
        "initial_pitch_target": 0.0,
        "require_exclusive_writer": True,
        "yaw_lower": -0.5,
        "yaw_upper": 0.5,
        "pitch_lower": -1.18,
        "pitch_upper": 0.16,
    }
    ros = SimpleNamespace(Twist=_Twist, Float64MultiArray=_Array)
    return OfficialCommandPublisher(node, OFFICIAL_TOPICS, ros, config), node


def _active_action_and_decision() -> tuple[FinalAction, Any]:
    return ActionMux().compose_with_decision(_base(v=0.1, w=0.2), None, _joints(), _status(), NOW)


def test_head_only_records_shadow_pitch_not_jointstate_pitch() -> None:
    publisher, node = _publisher(tracking=True)
    positions = list(_joints().position)
    positions[2] = 0.123
    action, decision = ActionMux().compose_with_decision(
        None,
        _manipulation((1,), values={1: 0.1}),
        _joints(tuple(positions)),
        _status(),
        NOW,
        manipulation_source="external_candidate",
    )

    groups = publisher.publish_head_with_trace(action)
    record = _build_action_dispatch_record(
        action,
        decision,
        publish_enabled=True,
        publisher_created=True,
        dispatch_mode=DispatchMode.HEAD_ONLY,
        group_records=groups,
    )

    assert node.calls == [OFFICIAL_TOPICS["head"]]
    assert action.values[3:5] == (0.1, 0.123)
    assert record.dispatched_action[3:5] == (0.1, 0.0)
    assert record.dispatched_mask == (False, False, False, True, True, *([False] * 14))
    assert record.attempted_groups == record.successful_groups == ("head",)
    assert record.controller_accepted is record.execution_confirmed is None


def test_full_trace_has_exact_five_group_mapping_and_complete_twist() -> None:
    publisher, node = _publisher()
    action, decision = _active_action_and_decision()
    groups = publisher.publish_with_trace(action)
    record = _build_action_dispatch_record(
        action,
        decision,
        publish_enabled=True,
        publisher_created=True,
        dispatch_mode=DispatchMode.FULL,
        group_records=groups,
    )

    assert node.calls == list(OFFICIAL_TOPICS.values())
    assert record.dispatched_action == action.values
    assert all(record.dispatched_mask)
    assert record.attempted_groups == (
        "base", "spine", "head", "left_arm", "right_arm"
    )
    base_payload = groups[0].exact_payload
    assert isinstance(base_payload, TwistExactPayload)
    assert base_payload.linear_xyz == (action.values[0], 0.0, 0.0)
    assert base_payload.angular_xyz == (0.0, 0.0, action.values[1])
    assert all(
        isinstance(group.exact_payload, Float64MultiArrayExactPayload)
        for group in groups[1:]
    )


def test_original_publish_and_publish_head_wrappers_remain_compatible() -> None:
    full_publisher, full_node = _publisher()
    action, _decision = _active_action_and_decision()
    assert full_publisher.publish(action) is None
    assert full_node.calls == list(OFFICIAL_TOPICS.values())

    head_publisher, head_node = _publisher(tracking=True)
    assert head_publisher.publish_head(action) is None
    assert head_node.calls == [OFFICIAL_TOPICS["head"]]


def test_full_middle_failure_preserves_success_failure_and_unattempted_groups() -> None:
    publisher, node = _publisher()
    node.fail_topic = OFFICIAL_TOPICS["head"]
    action, decision = _active_action_and_decision()

    with pytest.raises(OfficialPublishError) as captured:
        publisher.publish_with_trace(action)
    groups = captured.value.group_records
    record = _build_action_dispatch_record(
        action,
        decision,
        publish_enabled=True,
        publisher_created=True,
        dispatch_mode=DispatchMode.FULL,
        group_records=groups,
        failure_reason=str(captured.value),
    )

    assert record.successful_groups == ("base", "spine")
    assert record.failed_groups == ("head",)
    assert record.attempted_groups == ("base", "spine", "head")
    assert record.publisher_call_succeeded is False
    assert record.dispatched_mask[:5] == (True,) * 5
    assert record.dispatched_action[5:] == (None,) * 14
    assert groups[3].attempted is groups[4].attempted is False
    assert groups[3].exact_payload is groups[4].exact_payload is None


def _unattempted(group: Any) -> Any:
    return replace(
        group,
        attempted=False,
        succeeded=None,
        exact_payload=None,
        failure_reason="",
    )


def _failed(group: Any, reason: str) -> Any:
    return replace(group, succeeded=False, failure_reason=reason)


def test_full_dispatch_rejects_non_prefix_attempt_and_failure_order_conflicts() -> None:
    publisher, _node = _publisher()
    action, decision = _active_action_and_decision()
    successful = publisher.publish_with_trace(action)

    skipped_base = (
        _unattempted(successful[0]),
        successful[1],
        *tuple(_unattempted(group) for group in successful[2:]),
    )
    with pytest.raises(ValueError, match="连续前缀"):
        _build_action_dispatch_record(
            action,
            decision,
            publish_enabled=True,
            publisher_created=True,
            dispatch_mode=DispatchMode.FULL,
            group_records=skipped_base,
        )

    continued_after_failure = (
        _failed(successful[0], "base_failed"),
        successful[1],
        *tuple(_unattempted(group) for group in successful[2:]),
    )
    with pytest.raises(ValueError, match="最后一个"):
        _build_action_dispatch_record(
            action,
            decision,
            publish_enabled=True,
            publisher_created=True,
            dispatch_mode=DispatchMode.FULL,
            group_records=continued_after_failure,
        )

    two_failures = (
        _failed(successful[0], "base_failed"),
        _failed(successful[1], "spine_failed"),
        *tuple(_unattempted(group) for group in successful[2:]),
    )
    with pytest.raises(ValueError, match="最多允许一个"):
        _build_action_dispatch_record(
            action,
            decision,
            publish_enabled=True,
            publisher_created=True,
            dispatch_mode=DispatchMode.FULL,
            group_records=two_failures,
        )


def test_full_dispatch_allows_empty_attempt_prefix() -> None:
    publisher, _node = _publisher()
    action, decision = _active_action_and_decision()
    empty = tuple(
        _unattempted(group) for group in publisher.publish_with_trace(action)
    )
    record = _build_action_dispatch_record(
        action,
        decision,
        publish_enabled=True,
        publisher_created=True,
        dispatch_mode=DispatchMode.FULL,
        group_records=empty,
        failure_reason="preflight_failed_before_publisher_call",
    )
    assert record.attempted_groups == ()
    assert record.publisher_call_succeeded is None
    assert record.dispatched_action == (None,) * 19


def test_dispatch_json_is_strict_and_round_trips_without_execution_claims() -> None:
    publisher, _node = _publisher(tracking=True)
    action, decision = _active_action_and_decision()
    groups = publisher.publish_with_trace(action)
    record = _build_action_dispatch_record(
        action,
        decision,
        publish_enabled=True,
        publisher_created=True,
        dispatch_mode=DispatchMode.FULL,
        group_records=groups,
    )
    raw = action_dispatch_to_json(record)
    assert action_dispatch_from_json(raw) == record
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw)["controller_accepted"] is None
    assert json.loads(raw)["execution_confirmed"] is None


def test_dispatch_json_rejects_decision_timestamp_mismatch() -> None:
    publisher, _node = _publisher()
    action, decision = _active_action_and_decision()
    record = _build_action_dispatch_record(
        action,
        decision,
        publish_enabled=True,
        publisher_created=True,
        dispatch_mode=DispatchMode.FULL,
        group_records=publisher.publish_with_trace(action),
    )
    payload = json.loads(action_dispatch_to_json(record))
    payload["decision"]["timestamp_ns"] += 1
    with pytest.raises(ValueError, match="timestamp_ns"):
        action_dispatch_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "mutation",
    [
        "nan",
        "bool",
        "payload_bool",
        "dimension",
        "version",
        "decision_version",
        "unknown",
    ],
)
def test_dispatch_json_rejects_nonfinite_bool_dimension_version_and_unknown_fields(
    mutation: str,
) -> None:
    publisher, _node = _publisher()
    action, decision = _active_action_and_decision()
    record = _build_action_dispatch_record(
        action,
        decision,
        publish_enabled=True,
        publisher_created=True,
        dispatch_mode=DispatchMode.FULL,
        group_records=publisher.publish_with_trace(action),
    )
    payload = json.loads(action_dispatch_to_json(record))
    if mutation == "nan":
        payload["dispatched_action"][0] = float("nan")
    elif mutation == "bool":
        payload["dispatched_action"][0] = True
    elif mutation == "payload_bool":
        payload["group_records"][0]["exact_payload"]["linear"]["x"] = True
    elif mutation == "dimension":
        payload["dispatched_action"].pop()
    elif mutation == "version":
        payload["schema_version"] = 2
    elif mutation == "decision_version":
        payload["decision"]["schema_version"] = 2
    else:
        payload["unknown"] = "field"
    with pytest.raises(ValueError):
        action_dispatch_from_json(json.dumps(payload, allow_nan=True))


def test_config_has_internal_dispatch_topic_and_no_extra_official_publishers() -> None:
    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    assert config["topics"]["action_dispatch"] == "/team/action_dispatch"
    publisher, node = _publisher()
    assert publisher is not None
    assert set(node.publishers) == set(OFFICIAL_TOPICS.values())
    assert len(node.publishers) == 5


def test_control_sources_contain_no_native_pi05_or_eight_to_nineteen_mapping() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / "team_sorting" / name).read_text(encoding="utf-8")
        for name in ("interfaces.py", "action_mux.py", "ros_nodes.py")
    )
    assert "pi05_droid" not in source
    assert "8→19" not in source
    assert "8->19" not in source
