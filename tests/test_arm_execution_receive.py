"""ArmExecutionController 抓取验证接收骨架的专项回归测试。"""

import pytest

from team_sorting.arm_execution import ArmExecutionController
from team_sorting.interfaces import (
    ArmExecutionConfig,
    ArmMotionPhase,
    GlobalPhase,
    GraspVerification,
    JointTrajectory,
    JointWaypoint,
    LocalPhase,
    RobotJointState,
)


_JOINT_POSITION = (
    0.10, 0.05, -0.20, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
    0.40, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, 0.45,
)
_CONTROLLED_MASK = (
    True, False, False, True, True, True, True, True, True, True,
    True, True, True, True, True, True, True,
)


def _verification(**overrides: object) -> GraspVerification:
    values: dict[str, object] = {
        "is_grasped": True,
        "confidence": 0.9,
        "visual_evidence": "目标随夹爪稳定移动",
        "effort_evidence": "夹爪 effort 在预期范围",
        "success": True,
        "failure_reason": "",
        "timestamp_ns": 123_456,
    }
    values.update(overrides)
    return GraspVerification(**values)  # type: ignore[arg-type]


def _damaged_verification(**overrides: object) -> GraspVerification:
    verification = object.__new__(GraspVerification)
    values: dict[str, object] = {
        "is_grasped": True,
        "confidence": 0.9,
        "visual_evidence": "视觉证据",
        "effort_evidence": "effort证据",
        "success": True,
        "failure_reason": "",
        "timestamp_ns": 123_456,
    }
    values.update(overrides)
    for name, value in values.items():
        object.__setattr__(verification, name, value)
    return verification


def _execution_controller() -> ArmExecutionController:
    return ArmExecutionController(
        ArmExecutionConfig(
            feedback_max_age_ns=1_000_000_000,
            trajectory_max_age_ns=1_000_000_000,
            command_ttl_ns=100_000_000,
            max_control_period_ns=2_000_000_000,
            waypoint_timeout_margin_ns=1_000_000_000,
            total_timeout_margin_ns=2_000_000_000,
            max_slide_velocity_m_s=0.2,
            max_arm_velocity_rad_s=1.0,
            max_gripper_velocity_per_s=0.5,
            slide_tolerance_m=0.005,
            arm_tolerance_rad=0.01,
            gripper_tolerance=0.02,
            settle_cycles=1,
            initial_slide_error_limit_m=0.05,
            initial_arm_error_limit_rad=0.2,
            initial_gripper_error_limit=0.2,
        )
    )


def _pick_trajectory() -> JointTrajectory:
    phases = (
        ArmMotionPhase.PREGRASP,
        ArmMotionPhase.GRASP,
        ArmMotionPhase.LIFT,
        ArmMotionPhase.RETREAT,
    )
    waypoints = tuple(
        JointWaypoint(phase, float(index), _JOINT_POSITION, _CONTROLLED_MASK)
        for index, phase in enumerate(phases)
    )
    return JointTrajectory(
        trajectory_id="verification-receive-test",
        task_id=1,
        target_body="box",
        execution_phase=GlobalPhase.EXECUTE_PICK,
        waypoints=waypoints,
        timestamp_ns=2_000,
    )


def _actual_joints(timestamp_ns: int) -> RobotJointState:
    return RobotJointState(
        position=_JOINT_POSITION,
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=timestamp_ns,
    )


def test_accept_grasp_verification_only_caches_and_records_timestamp() -> None:
    controller = ArmExecutionController()
    controller.local_phase = LocalPhase.VERIFY
    verification = _verification()

    result = controller.accept_grasp_verification(verification)

    assert result is None
    assert controller._cached_verification is verification
    assert controller._verification_received_ns == verification.timestamp_ns
    assert controller.local_phase is LocalPhase.VERIFY


def test_accept_grasp_verification_replaces_cache_and_reset_clears_it() -> None:
    controller = ArmExecutionController()
    first = _verification(timestamp_ns=100)
    latest = _verification(timestamp_ns=200, success=False, failure_reason="证据不足")

    controller.accept_grasp_verification(first)
    controller.accept_grasp_verification(latest)

    assert controller._cached_verification is latest
    assert controller._verification_received_ns == 200

    controller.reset()

    assert controller._cached_verification is None
    assert controller._verification_received_ns is None


def test_cached_verification_does_not_unlock_or_advance_verify() -> None:
    controller = _execution_controller()
    assert controller.start_trajectory(_pick_trajectory()).state == "LOADED"

    for index in range(3):
        now_ns = index * 1_000_000_000 + 2_000
        _, status = controller.step(_actual_joints(now_ns), now_ns)

    assert status.state == "VERIFICATION_PENDING"
    assert controller.local_phase is LocalPhase.VERIFY
    lift_waypoint_index = controller._waypoint_index

    verification = _verification(timestamp_ns=now_ns)
    controller.accept_grasp_verification(verification)
    next_ns = now_ns + 1_000_000_000
    _, repeated = controller.step(_actual_joints(next_ns), next_ns)

    assert repeated.state == "VERIFICATION_PENDING"
    assert repeated.success is False
    assert controller.local_phase is LocalPhase.VERIFY
    assert controller._waypoint_index == lift_waypoint_index
    assert controller._cached_verification is verification


@pytest.mark.parametrize(
    "verification",
    [
        object(),
        _damaged_verification(timestamp_ns=True),
        _damaged_verification(timestamp_ns=-1),
        _damaged_verification(timestamp_ns=1.0),
        _damaged_verification(success=1),
        _damaged_verification(failure_reason=None),
    ],
)
def test_accept_grasp_verification_rejects_invalid_values_transactionally(
    verification: object,
) -> None:
    controller = ArmExecutionController()
    accepted = _verification(timestamp_ns=99)
    controller.accept_grasp_verification(accepted)

    with pytest.raises(ValueError):
        controller.accept_grasp_verification(verification)  # type: ignore[arg-type]

    assert controller._cached_verification is accepted
    assert controller._verification_received_ns == 99


@pytest.mark.parametrize(
    "missing_field",
    [
        "is_grasped",
        "confidence",
        "visual_evidence",
        "effort_evidence",
        "success",
        "failure_reason",
        "timestamp_ns",
    ],
)
def test_accept_grasp_verification_rejects_incomplete_instance(
    missing_field: str,
) -> None:
    verification = _damaged_verification()
    object.__delattr__(verification, missing_field)
    controller = ArmExecutionController()

    with pytest.raises(ValueError, match="字段不完整"):
        controller.accept_grasp_verification(verification)

    assert controller._cached_verification is None
    assert controller._verification_received_ns is None
