"""Stage-2A external Candidate consumer safety and integration regressions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from team_sorting.action_mux import ActionMux
from team_sorting.external_candidate import (
    ExternalCandidateConfig,
    ExternalCandidateConsumer,
    ExternalCandidateDecoder,
    instruction_fingerprint,
)
from team_sorting.interfaces import (
    FSMStatus,
    GlobalPhase,
    JOINT_NAMES,
    LocalPhase,
    ManipulationCommand,
    RobotJointState,
)
from team_sorting.ros_nodes import (
    _create_team_client_node,
    _external_candidate_subscription_enabled,
)


NOW = 10_000_000_000
RAW_INSTRUCTION = '{"instruction":"move","task":1}'


def _config(**overrides: object) -> ExternalCandidateConfig:
    values: dict[str, object] = {
        "enabled": True,
        "enable_actuation": True,
        "simulation_publish_enabled": True,
        "expected_episode_id": "episode_0001",
        "allow_generation_binding": True,
    }
    values.update(overrides)
    return ExternalCandidateConfig(**values)


def _identity(raw: str = RAW_INSTRUCTION, task_id: int = 1) -> str:
    return f"episode_0001:{task_id}:{instruction_fingerprint(raw)}"


def _payload(**overrides: object) -> dict[str, object]:
    values = [0.01 * index for index in range(17)]
    values[1] = 0.105
    payload: dict[str, object] = {
        "schema_version": 1,
        "timestamp_ns": NOW - 10_000_000,
        "request_id": "request-1",
        "generation": "opaque-run:1",
        "task_identity": _identity(),
        "values": [0.0, 0.0, *values],
        "controlled_mask": [False, False, False, True, *([False] * 15)],
        "valid": True,
        "valid_until_ns": NOW + 100_000_000,
        "source": "fixed_safe_candidate",
        "mode": "fixed_head_yaw",
        "failure_reason": "",
        "published_to_robot": False,
    }
    payload.update(overrides)
    return payload


def _raw(**overrides: object) -> str:
    return json.dumps(_payload(**overrides), allow_nan=True)


def _joints(
    *, timestamp_ns: int = NOW - 10_000_000, valid: bool = True, head_yaw: float = 0.1
) -> RobotJointState:
    positions = [0.01 * index for index in range(17)]
    positions[1] = head_yaw
    return RobotJointState(
        position=tuple(positions),
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=timestamp_ns,
        valid=valid,
        failure_reason="invalid" if not valid else "",
    )


def _status(phase: GlobalPhase = GlobalPhase.SEARCH_TARGET) -> FSMStatus:
    return FSMStatus(1, phase, LocalPhase.IDLE, 0, False, "", NOW)


def _consumer(config: ExternalCandidateConfig | None = None) -> ExternalCandidateConsumer:
    consumer = ExternalCandidateConsumer(config or _config())
    consumer.update_instruction(RAW_INSTRUCTION, 1, NOW - 20_000_000)
    return consumer


def _accept(consumer: ExternalCandidateConsumer, **overrides: object):
    return consumer.receive(_raw(**overrides), NOW)


def _take(
    consumer: ExternalCandidateConsumer,
    *,
    now_ns: int = NOW,
    joints: RobotJointState | None = None,
    dt: object = 0.05,
    existing: ManipulationCommand | None = None,
):
    return consumer.take(
        now_ns=now_ns,
        actual_joints=joints or _joints(),
        fsm_status=_status(),
        actual_dt_s=dt,
        existing_command=existing,
    )


def test_default_disabled_does_not_request_subscription() -> None:
    config = ExternalCandidateConfig()
    assert not config.enabled
    assert not _external_candidate_subscription_enabled(config)


def test_default_team_client_does_not_create_external_subscription() -> None:
    class Publisher:
        def publish(self, _message: object) -> None:
            return None

    class Logger:
        def info(self, _message: str) -> None:
            return None

        warning = info
        error = info

    class Node:
        def __init__(self, _name: str) -> None:
            self.subscriptions: list[str] = []

        def create_publisher(self, *_args: object) -> Publisher:
            return Publisher()

        def create_subscription(
            self, _type: object, topic: str, _callback: object, _depth: int
        ) -> object:
            self.subscriptions.append(topic)
            return object()

        def create_timer(self, *_args: object) -> SimpleNamespace:
            return SimpleNamespace(cancel=lambda: None)

        def get_logger(self) -> Logger:
            return Logger()

    class Message:
        def __init__(self) -> None:
            self.data: object = ""

    class Twist:
        def __init__(self) -> None:
            self.linear = SimpleNamespace(x=0.0)
            self.angular = SimpleNamespace(z=0.0)

    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    ros = SimpleNamespace(
        Node=Node,
        String=Message,
        Twist=Twist,
        Float64MultiArray=Message,
        Odometry=object,
        JointState=object,
        Detection3DArray=object,
    )
    node = _create_team_client_node(ros)(config, ros)
    assert "/mmk2_pi05_adapter/safe_candidate" not in node.subscriptions


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"enable_actuation": False}, "external_candidate_actuation_disabled"),
        ({"simulation_publish_enabled": False}, "simulation_publish_disabled"),
    ],
)
def test_execution_gates_reject_before_pending(overrides: dict[str, object], reason: str) -> None:
    decision = _accept(_consumer(_config(**overrides)))
    assert not decision.accepted
    assert decision.failure_reason == reason


def test_simulation_only_false_is_rejected() -> None:
    with pytest.raises(ValueError, match="simulation_only"):
        ExternalCandidateConfig(simulation_only=False)


def test_decoder_accepts_exact_thirteen_field_schema() -> None:
    candidate = ExternalCandidateDecoder(_config()).decode(_raw())
    assert len(_payload()) == 13
    assert candidate.values[3] == pytest.approx(0.105)


@pytest.mark.parametrize("change", ["unknown", "missing"])
def test_decoder_rejects_field_set_mismatch(change: str) -> None:
    payload = _payload()
    if change == "unknown":
        payload["extra"] = 1
    else:
        payload.pop("mode")
    with pytest.raises(ValueError, match="fields_mismatch"):
        ExternalCandidateDecoder(_config()).decode(json.dumps(payload))


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"schema_version": 2}, "unsupported_schema_version"),
        ({"valid": False}, "candidate_must_be_valid"),
        ({"published_to_robot": True}, "published_to_robot"),
        ({"source": "pi05_shadow"}, "source_mismatch"),
        ({"mode": "disabled"}, "mode_mismatch"),
        ({"failure_reason": "bad"}, "failure_reason"),
    ],
)
def test_decoder_rejects_invalid_contract_values(
    overrides: dict[str, object], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        ExternalCandidateDecoder(_config()).decode(_raw(**overrides))


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf")])
def test_decoder_rejects_bool_and_non_finite_values(value: object) -> None:
    values = list(_payload()["values"])
    values[3] = value
    with pytest.raises(ValueError, match="real_not_bool|non_finite|finite"):
        ExternalCandidateDecoder(_config()).decode(_raw(values=values))


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("values", [0.0] * 18, "19_items"),
        ("controlled_mask", [False] * 18, "19_items"),
        ("controlled_mask", [False, False, False, 1, *([False] * 15)], "strict_bool"),
    ],
)
def test_decoder_rejects_bad_vector_contract(field: str, value: object, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        ExternalCandidateDecoder(_config()).decode(_raw(**{field: value}))


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"values": [0.1, 0.0, *([0.0] * 17)]}, "base_values"),
        ({"controlled_mask": [True, False, False, True, *([False] * 15)]}, "index_3"),
        ({"controlled_mask": [False, False, True, False, *([False] * 15)]}, "index_3"),
    ],
)
def test_decoder_rejects_base_or_wrong_controlled_axis(
    overrides: dict[str, object], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        ExternalCandidateDecoder(_config()).decode(_raw(**overrides))


def test_candidate_converts_to_existing_seventeen_dimensional_command_once() -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    decision = _take(consumer)
    assert decision.accepted
    assert isinstance(decision.command, ManipulationCommand)
    assert len(decision.command.joint_target) == 17
    assert decision.command.controlled_mask == (False, True, *([False] * 15))
    assert decision.command.joint_target == tuple(_payload()["values"][2:])
    assert _take(consumer).failure_reason == "no_pending_candidate"


def test_task_identity_fingerprint_matches_adapter_algorithm() -> None:
    left = '{"task":1,"instruction":"移动"}'
    right = '{ "instruction": "移动", "task": 1 }'
    assert instruction_fingerprint(left) == instruction_fingerprint(right)
    assert len(instruction_fingerprint(left)) == 16


def test_expected_episode_id_missing_rejects() -> None:
    consumer = ExternalCandidateConsumer(_config(expected_episode_id=""))
    consumer.update_instruction(RAW_INSTRUCTION, 1, NOW - 1)
    decision = consumer.receive(_raw(), NOW)
    assert decision.failure_reason == "expected_episode_id_missing"


def test_task_identity_mismatch_rejects() -> None:
    decision = _accept(_consumer(), task_identity="episode_0001:9:wrong")
    assert decision.failure_reason == "task_identity_mismatch"


def test_generation_requires_explicit_binding_authorization() -> None:
    decision = _accept(_consumer(_config(allow_generation_binding=False)))
    assert decision.failure_reason == "generation_not_authorized"


def test_generation_binds_once_then_change_is_rejected() -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    assert consumer.bound_generation == "opaque-run:1"
    assert _take(consumer).accepted
    decision = _accept(consumer, request_id="request-2", generation="opaque-run:2")
    assert decision.failure_reason == "generation_changed"


def test_task_change_clears_generation_requests_and_pending() -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    assert consumer.pending and consumer.request_id_count == 1
    changed = consumer.update_instruction('{"instruction":"other","task":2}', 2, NOW + 1)
    assert changed
    assert not consumer.pending
    assert not consumer.bound_generation
    assert consumer.request_id_count == 0


def test_duplicate_request_id_cannot_be_consumed_twice() -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    assert _take(consumer).accepted
    decision = _accept(consumer)
    assert decision.failure_reason == "duplicate_request_id"


def test_request_id_cache_is_bounded() -> None:
    consumer = _consumer(_config(request_id_cache_size=2))
    for index in range(3):
        assert _accept(consumer, request_id=f"request-{index}").accepted
        _take(consumer)
    assert consumer.request_id_count == 2


def test_candidate_future_timestamp_rejects_as_clock_mismatch() -> None:
    assert _accept(_consumer(), timestamp_ns=NOW + 1).failure_reason == "candidate_clock_mismatch"


def test_candidate_expiry_equality_is_rejected() -> None:
    decision = _accept(_consumer(), timestamp_ns=NOW - 1, valid_until_ns=NOW)
    assert decision.failure_reason == "candidate_expired"


def test_consumer_ttl_is_tightened_never_extended() -> None:
    consumer = _consumer(_config(candidate_ttl_ms=25.0))
    decision = _accept(consumer, valid_until_ns=NOW + 1_000_000_000)
    assert decision.accepted
    assert decision.consumer_valid_until_ns == NOW + 25_000_000


@pytest.mark.parametrize(
    ("received_ns", "reason"),
    [
        (NOW + 1, "instruction_future_timestamp"),
        (NOW - 1_500_000_001, "instruction_stale"),
    ],
)
def test_instruction_time_is_independently_checked(received_ns: int, reason: str) -> None:
    consumer = ExternalCandidateConsumer(_config())
    consumer.update_instruction(RAW_INSTRUCTION, 1, received_ns)
    assert consumer.receive(_raw(), NOW).failure_reason == reason


def test_repeated_instruction_refreshes_liveness_without_identity_change() -> None:
    consumer = _consumer()
    identity = consumer.current_task_identity
    assert not consumer.update_instruction(RAW_INSTRUCTION, 1, NOW)
    assert consumer.current_task_identity == identity
    assert consumer.instruction_received_ns == NOW


@pytest.mark.parametrize(
    ("joints", "reason"),
    [
        (_joints(valid=False), "joint_state_invalid"),
        (_joints(timestamp_ns=NOW - 250_000_001), "joint_state_stale"),
        (_joints(timestamp_ns=NOW + 1), "joint_state_future_timestamp"),
    ],
)
def test_joint_state_safety_rejections(joints: RobotJointState, reason: str) -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    assert _take(consumer, joints=joints).failure_reason == reason


def test_delta_limit_rejects_without_action_mux_clipping() -> None:
    consumer = _consumer()
    values = list(_payload()["values"])
    values[3] = 0.2
    assert _accept(consumer, values=values).accepted
    decision = _take(consumer)
    assert decision.failure_reason == "joint_delta_exceeds_limit"
    assert decision.command is None


def test_velocity_limit_uses_actual_control_period() -> None:
    consumer = _consumer(_config(max_joint_delta_rad=1.0, max_joint_velocity_rad_s=0.2))
    assert _accept(consumer).accepted
    decision = _take(consumer, dt=0.01)
    assert decision.allowed_delta == pytest.approx(0.002)
    assert decision.failure_reason == "joint_delta_exceeds_limit"


def test_remaining_ttl_tightens_effective_dt() -> None:
    consumer = _consumer(_config(max_joint_delta_rad=1.0, max_joint_velocity_rad_s=0.2))
    values = list(_payload()["values"])
    values[3] = 0.1005
    assert _accept(consumer, values=values, valid_until_ns=NOW + 1_000_000).accepted
    decision = _take(consumer)
    assert decision.allowed_delta == pytest.approx(0.0002)
    assert decision.failure_reason == "joint_delta_exceeds_limit"


def test_provisional_head_yaw_limit_rejects_before_action_mux() -> None:
    consumer = _consumer(_config(provisional_head_yaw_upper_rad=0.11))
    values = list(_payload()["values"])
    values[3] = 0.12
    assert _accept(consumer, values=values).accepted
    decision = _take(consumer, joints=_joints(head_yaw=0.115))
    assert decision.failure_reason == "provisional_head_yaw_limit"
    assert decision.command is None


def test_second_pending_is_rejected_without_overwrite() -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    decision = _accept(consumer, request_id="request-2")
    assert decision.failure_reason == "pending_candidate_exists"
    assert _take(consumer).candidate.request_id == "request-1"


def test_parse_failure_does_not_overwrite_pending() -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    assert not consumer.receive("not-json", NOW + 1).accepted
    assert _take(consumer).candidate.request_id == "request-1"


def test_watchdog_clears_expired_pending() -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    decision = consumer.watchdog(NOW + 100_000_000)
    assert decision.failure_reason == "candidate_expired"
    assert not consumer.pending


def test_publisher_timeout_clears_without_replay() -> None:
    consumer = _consumer(_config(candidate_ttl_ms=1000.0, watchdog_timeout_ms=300.0))
    assert _accept(consumer, valid_until_ns=NOW + 1_000_000_000).accepted
    decision = consumer.watchdog(NOW + 300_000_001)
    assert decision.failure_reason == "candidate_publisher_timeout"
    assert _take(consumer, now_ns=NOW + 300_000_001).command is None


def test_shutdown_clears_pending_and_rejects_future_input() -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    consumer.shutdown(NOW + 1)
    assert not consumer.pending
    assert _accept(consumer, request_id="request-2").failure_reason == "consumer_shutdown"


def test_existing_controlled_manipulation_command_conflicts() -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    existing = ManipulationCommand(
        joint_target=_joints().position,
        controlled_mask=(True, *([False] * 16)),
        local_phase=LocalPhase.IDLE,
        timestamp_ns=NOW,
        valid_until_ns=NOW + 1,
    )
    assert _take(consumer, existing=existing).failure_reason == "manipulation_command_conflict"


@pytest.mark.parametrize("phase", [GlobalPhase.SAFE_HOLD, GlobalPhase.FAILED])
def test_action_mux_stop_phases_still_override_external_command(phase: GlobalPhase) -> None:
    consumer = _consumer()
    assert _accept(consumer).accepted
    decision = consumer.take(
        now_ns=NOW,
        actual_joints=_joints(),
        fsm_status=_status(phase),
        actual_dt_s=0.05,
        existing_command=None,
    )
    assert decision.accepted
    action = ActionMux().compose(None, decision.command, _joints(), _status(phase), NOW)
    assert action.values[3] == pytest.approx(_joints().position[1])


def test_default_configuration_preserves_existing_action_mux_hold() -> None:
    consumer = ExternalCandidateConsumer(ExternalCandidateConfig())
    decision = _take(consumer)
    assert not decision.accepted
    action = ActionMux().compose(None, None, _joints(), _status(), NOW)
    assert action.values[2:] == _joints().position


def test_external_module_has_no_official_publishers_or_native_8d_mapping() -> None:
    source = Path("team_sorting/external_candidate.py").read_text(encoding="utf-8")
    assert "create_publisher" not in source
    assert "OfficialCommandPublisher" not in source
    assert "/cmd_vel" not in source
    assert "8D" not in source


def test_audit_payload_keeps_adapter_published_flag_out_of_authorization() -> None:
    consumer = _consumer()
    decision = _accept(consumer)
    payload = decision.audit_dict()
    assert decision.accepted
    assert payload["official_publish_attempted"] is False
    assert payload["official_publish_success"] is False
