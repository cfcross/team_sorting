"""Stage 2.2 Interface v1 construction-invariant regressions."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from team_sorting.interfaces import (
    JOINT_NAMES,
    BaseCommand,
    BaseState,
    FSMStatus,
    FinalAction,
    GlobalPhase,
    GraspVerification,
    IKResult,
    LocalPhase,
    ManipulationCommand,
    ManipulationStatus,
    NavGoal,
    NavigationStatus,
    RobotJointState,
    final_action_from_json,
    final_action_to_json,
    fsm_status_from_json,
    fsm_status_to_json,
)


NOW = 1_000


def _base() -> BaseState:
    return BaseState(
        (1.0, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0.25,
        (0.1, 0.0, 0.0), (0.0, 0.0, -0.1), "odom", NOW,
    )


def _joints() -> RobotJointState:
    return RobotJointState((0.0,) * 17, (0.0,) * 17, (0.0,) * 17, NOW)


def _goal() -> NavGoal:
    return NavGoal("pick-1", "pick", (1.0, 2.0, 0.5), "odom", 0.05, 0.1, 2_000)


def _base_command() -> BaseCommand:
    return BaseCommand(0.1, -0.2, NOW, 2_000)


def _navigation_status() -> NavigationStatus:
    return NavigationStatus("pick-1", "moving", 1.0, -0.2, False, "", NOW)


def _ik() -> IKResult:
    return IKResult(0.1, (0.0,) * 6, None, True)


def _manipulation_command() -> ManipulationCommand:
    return ManipulationCommand(
        (0.0,) * 17, (True,) * 17, LocalPhase.APPROACH, NOW, 2_000
    )


def _manipulation_status() -> ManipulationStatus:
    return ManipulationStatus(
        LocalPhase.APPROACH, "RUNNING", 0.5, 0.1, False, "尚未完成", NOW
    )


@pytest.mark.parametrize(
    "state",
    (
        "MOTION_COMPLETED_PICK",
        "MOTION_COMPLETED_PLACE_VERIFICATION_PENDING",
    ),
)
def test_manipulation_status_motion_completion_requires_success_true(
    state: str,
) -> None:
    completed = ManipulationStatus(
        LocalPhase.IDLE, state, 1.0, 0.0, True,
        "运动完成但业务验证保持独立", NOW,
    )
    assert completed.success is True

    with pytest.raises(ValueError, match="运动学完成态"):
        replace(completed, success=False)


def test_manipulation_status_noncompletion_rejects_success_true() -> None:
    with pytest.raises(ValueError, match="运动学完成态"):
        replace(_manipulation_status(), success=True)


def _fsm_status() -> FSMStatus:
    return FSMStatus(
        1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW
    )


def _final_action() -> FinalAction:
    return FinalAction(
        (0.0,) * 19, 0, NOW, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE,
        True, False, "",
    )


def _grasp_verification(**overrides: object) -> GraspVerification:
    values: dict[str, object] = {
        "is_grasped": True,
        "confidence": 0.9,
        "visual_evidence": "v",
        "effort_evidence": "e",
        "success": True,
        "failure_reason": "",
        "timestamp_ns": 100,
    }
    values.update(overrides)
    return GraspVerification(**values)  # type: ignore[arg-type]


def test_grasp_verification_valid_construct() -> None:
    verification = _grasp_verification()
    assert verification.timestamp_ns == 100


@pytest.mark.parametrize("value", (1, "yes", 0))
def test_grasp_verification_is_grasped_rejects_non_strict_bool(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="is_grasped"):
        _grasp_verification(is_grasped=value)


@pytest.mark.parametrize("value", (1, "true"))
def test_grasp_verification_success_rejects_non_strict_bool(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="success"):
        _grasp_verification(success=value)


def test_grasp_verification_confidence_rejects_bool() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _grasp_verification(confidence=True)


@pytest.mark.parametrize("value", ("0.9", None))
def test_grasp_verification_confidence_rejects_non_real(value: object) -> None:
    with pytest.raises(ValueError, match="confidence"):
        _grasp_verification(confidence=value)


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_grasp_verification_confidence_rejects_nonfinite(value: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        _grasp_verification(confidence=value)


@pytest.mark.parametrize("value", (1.5, -0.1))
def test_grasp_verification_confidence_rejects_out_of_range(value: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        _grasp_verification(confidence=value)


@pytest.mark.parametrize("value", (123, None))
def test_grasp_verification_visual_evidence_rejects_non_string(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="visual_evidence"):
        _grasp_verification(visual_evidence=value)


def test_grasp_verification_effort_evidence_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="effort_evidence"):
        _grasp_verification(effort_evidence=None)


@pytest.mark.parametrize("value", (None, 123))
def test_grasp_verification_failure_reason_rejects_non_string(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        _grasp_verification(failure_reason=value)


def test_grasp_verification_timestamp_rejects_bool() -> None:
    with pytest.raises(ValueError, match="timestamp_ns"):
        _grasp_verification(timestamp_ns=True)


def test_grasp_verification_timestamp_rejects_float() -> None:
    with pytest.raises(ValueError, match="timestamp_ns"):
        _grasp_verification(timestamp_ns=1.0)


def test_grasp_verification_timestamp_rejects_negative() -> None:
    with pytest.raises(ValueError, match="timestamp_ns"):
        _grasp_verification(timestamp_ns=-1)


def test_all_target_types_accept_existing_production_shapes() -> None:
    values = (
        _base(), _joints(), _goal(), _base_command(), _navigation_status(),
        _ik(), _manipulation_command(), _manipulation_status(), _fsm_status(),
        _final_action(),
    )
    assert len(values) == 10
    assert all(value is not None for value in values)


@pytest.mark.parametrize(
    "factory",
    (
        lambda value: replace(_base(), timestamp_ns=value),
        lambda value: replace(_joints(), timestamp_ns=value),
        lambda value: replace(_goal(), deadline_ns=value),
        lambda value: replace(_base_command(), timestamp_ns=value),
        lambda value: replace(_manipulation_command(), timestamp_ns=value),
        lambda value: replace(_manipulation_status(), timestamp_ns=value),
        lambda value: replace(_fsm_status(), timestamp_ns=value),
        lambda value: replace(_final_action(), timestamp_ns=value),
    ),
)
@pytest.mark.parametrize("value", (True, -1))
def test_bool_and_negative_timestamps_are_rejected(factory, value: object) -> None:
    with pytest.raises(ValueError):
        factory(value)


def test_bool_cannot_masquerade_as_sequence_or_count() -> None:
    with pytest.raises(ValueError, match="sequence"):
        replace(_final_action(), sequence=True)
    with pytest.raises(ValueError, match="retry_count"):
        replace(_fsm_status(), retry_count=True)
    with pytest.raises(ValueError, match="task_id"):
        replace(_fsm_status(), task_id=True)


@pytest.mark.parametrize("field", ("position_xyz", "linear_velocity_xyz", "angular_velocity_xyz"))
@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_base_state_vectors_must_be_finite(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_base(), **{field: (bad, 0.0, 0.0)})


def test_base_state_frame_quaternion_yaw_and_valid_are_strict() -> None:
    with pytest.raises(ValueError, match="frame_id"):
        replace(_base(), frame_id="   ")
    with pytest.raises(ValueError, match="四元数范数"):
        replace(_base(), orientation_xyzw=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="yaw"):
        replace(_base(), yaw=float("nan"))
    with pytest.raises(ValueError, match="valid"):
        replace(_base(), valid=1)


@pytest.mark.parametrize("field", ("position", "velocity", "effort"))
def test_robot_joint_state_rejects_wrong_width_and_nonfinite_values(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_joints(), **{field: (0.0,) * 16})
    bad = [0.0] * 17
    bad[8] = float("nan")
    with pytest.raises(ValueError, match=field):
        replace(_joints(), **{field: tuple(bad)})


def test_robot_joint_state_requires_exact_joint_order() -> None:
    reversed_names = tuple(reversed(JOINT_NAMES))
    with pytest.raises(ValueError, match="joint_names"):
        replace(_joints(), joint_names=reversed_names)


def test_robot_joint_state_zero_filled_velocity_and_effort_remain_legal() -> None:
    state = RobotJointState((0.0,) * 17, (0.0,) * 17, (0.0,) * 17, NOW)
    assert state.velocity == state.effort == (0.0,) * 17


def test_nav_goal_rejects_nonfinite_pose_negative_tolerance_and_empty_frame() -> None:
    with pytest.raises(ValueError, match="pose_xyyaw"):
        replace(_goal(), pose_xyyaw=(float("inf"), 0.0, 0.0))
    with pytest.raises(ValueError, match="position_tolerance"):
        replace(_goal(), position_tolerance=-0.01)
    with pytest.raises(ValueError, match="yaw_tolerance"):
        replace(_goal(), yaw_tolerance=-0.01)
    with pytest.raises(ValueError, match="frame_id"):
        replace(_goal(), frame_id="")


@pytest.mark.parametrize("field", ("v", "w"))
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), True))
def test_base_command_rejects_nonfinite_and_bool_speeds(field: str, bad: object) -> None:
    with pytest.raises(ValueError, match=f"BaseCommand.{field}"):
        replace(_base_command(), **{field: bad})


def test_command_expiry_cannot_precede_generation_time() -> None:
    with pytest.raises(ValueError, match="valid_until_ns"):
        replace(_base_command(), valid_until_ns=NOW - 1)
    with pytest.raises(ValueError, match="valid_until_ns"):
        replace(_manipulation_command(), valid_until_ns=NOW - 1)


def test_navigation_status_rejects_unknown_state_and_invalid_errors() -> None:
    with pytest.raises(ValueError, match="state"):
        replace(_navigation_status(), state="teleported")
    with pytest.raises(ValueError, match="distance_error"):
        replace(_navigation_status(), distance_error=-0.1)
    with pytest.raises(ValueError, match="yaw_error"):
        replace(_navigation_status(), yaw_error=float("nan"))


def test_navigation_status_success_and_failure_state_are_consistent() -> None:
    arrived = NavigationStatus("pick-1", "arrived", 0.0, 0.0, True, "", NOW)
    assert arrived.success
    with pytest.raises(ValueError, match="success"):
        replace(_navigation_status(), success=True)
    with pytest.raises(ValueError, match="failure_reason"):
        NavigationStatus("pick-1", "failed", 0.0, 0.0, False, "", NOW)


def test_ik_result_success_requires_one_complete_six_axis_payload() -> None:
    with pytest.raises(ValueError, match="至少一侧"):
        IKResult(0.1, None, None, True)
    with pytest.raises(ValueError, match="6 项"):
        IKResult(0.1, (0.0,) * 5, None, True)
    with pytest.raises(ValueError, match="有限"):
        IKResult(0.1, (0.0,) * 5 + (float("nan"),), None, True)


def test_ik_result_failure_cannot_carry_targets_or_omit_reason() -> None:
    with pytest.raises(ValueError, match="不得携带"):
        IKResult(0.1, (0.0,) * 6, None, False, "失败")
    with pytest.raises(ValueError, match="failure_reason"):
        IKResult(0.1, None, None, False, "")


def test_manipulation_command_requires_strict_17d_values_and_bool_mask() -> None:
    with pytest.raises(ValueError, match="17 项"):
        replace(_manipulation_command(), joint_target=(0.0,) * 16)
    with pytest.raises(ValueError, match="有限"):
        replace(
            _manipulation_command(),
            joint_target=(0.0,) * 16 + (float("inf"),),
        )
    with pytest.raises(ValueError, match="严格 bool"):
        replace(_manipulation_command(), controlled_mask=(1,) + (False,) * 16)


def test_manipulation_command_rejects_wrong_phase_valid_and_empty_failure() -> None:
    with pytest.raises(ValueError, match="local_phase"):
        replace(_manipulation_command(), local_phase="APPROACH")
    with pytest.raises(ValueError, match="valid"):
        replace(_manipulation_command(), valid=1)
    with pytest.raises(ValueError, match="failure_reason"):
        replace(_manipulation_command(), valid=False, failure_reason="")


def test_manipulation_status_rejects_unknown_state_and_invalid_progress() -> None:
    with pytest.raises(ValueError, match="state"):
        replace(_manipulation_status(), state="SUCCEEDED")
    for progress in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="progress"):
            replace(_manipulation_status(), progress=progress)


def test_manipulation_status_preserves_positive_infinity_unknown_error_sentinel() -> None:
    status = replace(_manipulation_status(), max_joint_error=float("inf"))
    assert math.isinf(status.max_joint_error) and status.max_joint_error > 0
    for error in (float("nan"), float("-inf"), -0.1, True):
        with pytest.raises(ValueError, match="max_joint_error"):
            replace(_manipulation_status(), max_joint_error=error)


def test_fsm_status_rejects_non_enum_phases_negative_retry_and_false_done() -> None:
    with pytest.raises(ValueError, match="global_phase"):
        replace(_fsm_status(), global_phase="SEARCH_TARGET")
    with pytest.raises(ValueError, match="local_phase"):
        replace(_fsm_status(), local_phase="IDLE")
    with pytest.raises(ValueError, match="retry_count"):
        replace(_fsm_status(), retry_count=-1)
    with pytest.raises(ValueError, match="success"):
        replace(_fsm_status(), global_phase=GlobalPhase.DONE, success=False)


def test_final_action_rejects_width_nonfinite_sequence_timestamp_and_bool_fields() -> None:
    with pytest.raises(ValueError, match="19 项"):
        replace(_final_action(), values=(0.0,) * 18)
    with pytest.raises(ValueError, match="有限"):
        replace(_final_action(), values=(0.0,) * 18 + (float("nan"),))
    with pytest.raises(ValueError, match="sequence"):
        replace(_final_action(), sequence=True)
    with pytest.raises(ValueError, match="timestamp_ns"):
        replace(_final_action(), timestamp_ns=-1)
    with pytest.raises(ValueError, match="valid"):
        replace(_final_action(), valid=1)
    with pytest.raises(ValueError, match="clipped"):
        replace(_final_action(), clipped=0)


def test_invalid_final_action_requires_reason() -> None:
    with pytest.raises(ValueError, match="failure_reason"):
        replace(_final_action(), valid=False, failure_reason="")
    invalid = replace(_final_action(), valid=False, failure_reason="反馈无效")
    assert not invalid.valid


def test_final_action_normal_json_serialization_is_unchanged() -> None:
    action = _final_action()
    raw = final_action_to_json(action)
    assert raw == (
        '{"schema_version":1,"sequence":0,"timestamp_ns":1000,'
        '"action":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,'
        '0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],'
        '"global_phase":"SEARCH_TARGET","local_phase":"IDLE",'
        '"valid":true,"clipped":false,"failure_reason":""}'
    )
    assert final_action_from_json(raw) == action


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema_version", True), ("sequence", True), ("timestamp_ns", True),
     ("valid", 1), ("clipped", 0), ("failure_reason", 7)),
)
def test_final_action_json_cannot_coerce_invalid_scalar_types(
    field: str, value: object,
) -> None:
    payload = json.loads(final_action_to_json(_final_action()))
    payload[field] = value
    with pytest.raises(ValueError, match="FinalAction JSON 无效"):
        final_action_from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema_version", True), ("task_id", True), ("retry_count", True),
     ("success", 0), ("timestamp_ns", True), ("failure_reason", 7)),
)
def test_fsm_status_json_cannot_coerce_invalid_scalar_types(
    field: str, value: object,
) -> None:
    payload = json.loads(fsm_status_to_json(_fsm_status()))
    payload[field] = value
    with pytest.raises(ValueError, match="FSMStatus JSON 无效"):
        fsm_status_from_json(json.dumps(payload))
