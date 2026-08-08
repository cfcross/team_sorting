"""ArmExecutionController 抓取验证生命周期与运动完成闭环回归测试。"""

from fractions import Fraction

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


def _config(**overrides: object) -> ArmExecutionConfig:
    values: dict[str, object] = {
        "feedback_max_age_ns": 1_000_000_000,
        "trajectory_max_age_ns": 10_000_000_000,
        "command_ttl_ns": 100_000_000,
        "max_control_period_ns": 2_000_000_000,
        "verification_timeout_ns": 500_000_000,
        "waypoint_timeout_margin_ns": 1_000_000_000,
        "total_timeout_margin_ns": 2_000_000_000,
        "max_slide_velocity_m_s": 0.2,
        "max_arm_velocity_rad_s": 1.0,
        "max_gripper_velocity_per_s": 0.5,
        "slide_tolerance_m": 0.005,
        "arm_tolerance_rad": 0.01,
        "gripper_tolerance": 0.02,
        "settle_cycles": 2,
        "initial_slide_error_limit_m": 0.05,
        "initial_arm_error_limit_rad": 0.2,
        "initial_gripper_error_limit": 0.2,
    }
    values.update(overrides)
    return ArmExecutionConfig(**values)  # type: ignore[arg-type]


def _controller(**overrides: object) -> ArmExecutionController:
    return ArmExecutionController(_config(**overrides))


def _trajectory(
    execution_phase: GlobalPhase = GlobalPhase.EXECUTE_PICK,
    *,
    trajectory_id: str = "verification-closure-test",
    timestamp_ns: int = 2_000,
) -> JointTrajectory:
    phases = (
        (
            ArmMotionPhase.PREGRASP,
            ArmMotionPhase.GRASP,
            ArmMotionPhase.LIFT,
            ArmMotionPhase.RETREAT,
        )
        if execution_phase is GlobalPhase.EXECUTE_PICK
        else (
            ArmMotionPhase.PREPLACE,
            ArmMotionPhase.LOWER,
            ArmMotionPhase.RELEASE,
            ArmMotionPhase.POST_RELEASE_RETREAT,
        )
    )
    return JointTrajectory(
        trajectory_id=trajectory_id,
        task_id=1,
        target_body="box",
        execution_phase=execution_phase,
        waypoints=tuple(
            JointWaypoint(phase, float(index), _JOINT_POSITION, _CONTROLLED_MASK)
            for index, phase in enumerate(phases)
        ),
        timestamp_ns=timestamp_ns,
    )


def _actual(timestamp_ns: int) -> RobotJointState:
    return RobotJointState(
        position=_JOINT_POSITION,
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=timestamp_ns,
    )


def _verification(timestamp_ns: int, **overrides: object) -> GraspVerification:
    values: dict[str, object] = {
        "is_grasped": True,
        "confidence": 0.9,
        "visual_evidence": "目标随夹爪稳定移动",
        "effort_evidence": "夹爪effort在预期范围",
        "success": True,
        "failure_reason": "",
        "timestamp_ns": timestamp_ns,
    }
    values.update(overrides)
    return GraspVerification(**values)  # type: ignore[arg-type]


def _damaged_verification(
    base_timestamp_ns: int, **overrides: object
) -> GraspVerification:
    verification = object.__new__(GraspVerification)
    values: dict[str, object] = {
        "is_grasped": True,
        "confidence": 0.9,
        "visual_evidence": "视觉证据",
        "effort_evidence": "effort证据",
        "success": True,
        "failure_reason": "",
        "timestamp_ns": base_timestamp_ns,
    }
    values.update(overrides)
    for name, value in values.items():
        object.__setattr__(verification, name, value)
    return verification


def _settle_waypoint(
    controller: ArmExecutionController, target_ns: int
) -> tuple[object, object]:
    result = controller.step(_actual(target_ns), target_ns)
    return controller.step(_actual(target_ns + 1), target_ns + 1)


def _advance_pick_to_verify(
    controller: ArmExecutionController,
    *,
    start_ns: int = 2_000,
    trajectory_id: str = "verification-closure-test",
) -> tuple[JointTrajectory, int]:
    trajectory = _trajectory(
        trajectory_id=trajectory_id,
        timestamp_ns=start_ns,
    )
    assert controller.start_trajectory(trajectory).state == "LOADED"
    for index in range(3):
        _, status = _settle_waypoint(
            controller, start_ns + index * 1_000_000_000
        )
    assert status.state == "VERIFICATION_PENDING"
    assert status.local_phase is LocalPhase.VERIFY
    pending_ns = start_ns + 2_000_000_001
    return trajectory, pending_ns


def _unlock_retreat(
    controller: ArmExecutionController, pending_ns: int
) -> int:
    unlock_ns = pending_ns + 100_000_000
    command, status = controller.step(_actual(unlock_ns), unlock_ns)
    assert command.valid is True
    assert status.state == "RUNNING"
    assert status.local_phase is LocalPhase.RETREAT
    return unlock_ns


def _finish_retreat(
    controller: ArmExecutionController, unlock_ns: int
) -> tuple[object, object]:
    # 解锁帧不能计作RETREAT反馈；随后两个不同时间戳帧才满足settle_cycles=2。
    controller.step(_actual(unlock_ns + 1_000_000_000), unlock_ns + 1_000_000_000)
    return controller.step(
        _actual(unlock_ns + 1_000_000_001), unlock_ns + 1_000_000_001
    )


def test_positive_pick_verification_executes_retreat_and_reports_motion_complete() -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)
    accepted = _verification(pending_ns + 10)

    controller.accept_grasp_verification(accepted)
    # 内容相同且时间相同的重复消息幂等，不能替换首次接收对象。
    controller.accept_grasp_verification(_verification(pending_ns + 10))
    unlock_ns = _unlock_retreat(controller, pending_ns)
    command, completed = _finish_retreat(controller, unlock_ns)

    assert command.valid is False
    assert completed.state == "MOTION_COMPLETED_PICK"
    assert completed.success is True
    assert controller.latest_grasp_verification is accepted
    assert controller.latest_grasp_verification.is_grasped is True


def test_explicit_not_grasped_is_preserved_and_still_retreats_safely() -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)
    not_grasped = _verification(
        pending_ns + 10,
        is_grasped=False,
        confidence=0.95,
        visual_evidence="物体未随夹爪移动",
    )

    controller.accept_grasp_verification(not_grasped)
    unlock_ns = _unlock_retreat(controller, pending_ns)
    _, completed = _finish_retreat(controller, unlock_ns)

    assert completed.state == "MOTION_COMPLETED_PICK"
    assert completed.success is True  # 仅表示撤离轨迹完成，不是抓取业务成功。
    assert controller.latest_grasp_verification is not_grasped
    assert controller.latest_grasp_verification.success is True
    assert controller.latest_grasp_verification.is_grasped is False


def test_insufficient_evidence_waits_and_newer_completed_evidence_can_unlock() -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)
    insufficient = _verification(
        pending_ns + 10,
        is_grasped=False,
        confidence=0.2,
        success=False,
        failure_reason="当前证据不足",
    )

    controller.accept_grasp_verification(insufficient)
    _, waiting = controller.step(
        _actual(pending_ns + 100_000_000), pending_ns + 100_000_000
    )
    assert waiting.state == "VERIFICATION_PENDING"
    assert waiting.success is False
    assert "证据不足" in waiting.failure_reason
    assert controller.latest_grasp_verification is None

    completed_evidence = _verification(pending_ns + 200_000_000)
    controller.accept_grasp_verification(completed_evidence)
    unlock_ns = pending_ns + 300_000_000
    _, retreating = controller.step(_actual(unlock_ns), unlock_ns)
    assert retreating.state == "RUNNING"
    assert retreating.local_phase is LocalPhase.RETREAT
    assert controller.latest_grasp_verification is completed_evidence


def test_insufficient_evidence_times_out_fail_closed_and_clears_pending_context() -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)
    controller.accept_grasp_verification(
        _verification(
            pending_ns + 10,
            success=False,
            is_grasped=False,
            failure_reason="图像暂不可用",
        )
    )
    controller.step(_actual(pending_ns + 100_000_000), pending_ns + 100_000_000)

    timeout_ns = pending_ns + controller._config.verification_timeout_ns + 1
    command, failed = controller.step(_actual(timeout_ns), timeout_ns)

    assert command.valid is False
    assert failed.state == "FAILED"
    assert failed.success is False
    assert "verification_timeout_ns" in failed.failure_reason
    assert controller.latest_grasp_verification is None
    assert controller._cached_verification is None
    assert controller._verification_pending_since_ns is None


def test_place_trajectory_reaches_motion_complete_without_claiming_place_verified() -> None:
    controller = _controller()
    start_ns = 2_000
    assert controller.start_trajectory(
        _trajectory(GlobalPhase.EXECUTE_PLACE, timestamp_ns=start_ns)
    ).state == "LOADED"

    for index in range(4):
        command, status = _settle_waypoint(
            controller, start_ns + index * 1_000_000_000
        )

    assert command.valid is False
    assert status.state == "MOTION_COMPLETED_PLACE_VERIFICATION_PENDING"
    assert status.success is True
    assert "外部验证" in status.failure_reason
    assert controller.latest_grasp_verification is None


def test_accept_rejects_without_active_trajectory() -> None:
    with pytest.raises(ValueError, match="有效活动轨迹"):
        _controller().accept_grasp_verification(_verification(10))


def test_accept_rejects_place_trajectory_and_pick_before_lift() -> None:
    place = _controller()
    place.start_trajectory(_trajectory(GlobalPhase.EXECUTE_PLACE))
    with pytest.raises(ValueError, match="EXECUTE_PICK"):
        place.accept_grasp_verification(_verification(2_000))

    pick = _controller()
    pick.start_trajectory(_trajectory())
    with pytest.raises(ValueError, match="LIFT"):
        pick.accept_grasp_verification(_verification(2_000))


def test_accept_rejects_non_verify_and_timestamp_before_pending() -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)

    with pytest.raises(ValueError, match="早于进入VERIFY"):
        controller.accept_grasp_verification(_verification(pending_ns - 1))

    controller.local_phase = LocalPhase.TEST_LIFT
    with pytest.raises(ValueError, match="不是VERIFY"):
        controller.accept_grasp_verification(_verification(pending_ns + 1))


def test_accept_rejects_verification_timestamp_outside_waiting_window() -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)

    with pytest.raises(ValueError, match="verification_timeout_ns"):
        controller.accept_grasp_verification(
            _verification(pending_ns + controller._config.verification_timeout_ns + 1)
        )


def test_old_trajectory_verification_is_rejected_after_new_trajectory() -> None:
    controller = _controller(trajectory_max_age_ns=20_000_000_000)
    _, old_pending_ns = _advance_pick_to_verify(controller)
    old = _verification(old_pending_ns + 10)

    _, new_pending_ns = _advance_pick_to_verify(
        controller,
        start_ns=10_000_000_000,
        trajectory_id="replacement",
    )
    assert new_pending_ns > old.timestamp_ns
    with pytest.raises(ValueError, match="早于进入VERIFY"):
        controller.accept_grasp_verification(old)


def test_reset_rejects_reuse_and_new_trajectory_clears_published_result() -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)
    accepted = _verification(pending_ns + 10)
    controller.accept_grasp_verification(accepted)
    _unlock_retreat(controller, pending_ns)
    assert controller.latest_grasp_verification is accepted

    assert controller.start_trajectory(
        _trajectory(trajectory_id="new-trajectory")
    ).state == "LOADED"
    assert controller.latest_grasp_verification is None

    controller.reset()
    assert controller.latest_grasp_verification is None
    with pytest.raises(ValueError, match="有效活动轨迹"):
        controller.accept_grasp_verification(accepted)


def test_conflicting_same_timestamp_is_rejected_transactionally() -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)
    accepted = _verification(pending_ns + 10)
    controller.accept_grasp_verification(accepted)

    with pytest.raises(ValueError, match="同一timestamp_ns"):
        controller.accept_grasp_verification(
            _verification(pending_ns + 10, is_grasped=False)
        )

    _unlock_retreat(controller, pending_ns)
    assert controller.latest_grasp_verification is accepted


def test_older_timestamp_than_cached_verification_is_rejected() -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)
    accepted = _verification(pending_ns + 200)
    controller.accept_grasp_verification(accepted)

    with pytest.raises(ValueError, match="timestamp_ns"):
        controller.accept_grasp_verification(_verification(pending_ns + 100))

    controller.step(_actual(pending_ns + 1_000), pending_ns + 1_000)
    assert controller.latest_grasp_verification is accepted


def test_accept_supports_non_builtin_finite_real_confidence_in_active_window() -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)
    verification = _verification(pending_ns + 10, confidence=Fraction(1, 2))

    controller.accept_grasp_verification(verification)
    _unlock_retreat(controller, pending_ns)

    assert controller.latest_grasp_verification is verification


@pytest.mark.parametrize(
    "verification",
    [
        object(),
        _damaged_verification(1, timestamp_ns=True),
        _damaged_verification(1, timestamp_ns=-1),
        _damaged_verification(1, timestamp_ns=1.0),
        _damaged_verification(1, success=1),
        _damaged_verification(1, failure_reason=None),
        _damaged_verification(1, is_grasped="yes"),
        _damaged_verification(1, confidence=True),
        _damaged_verification(1, confidence="0.9"),
        _damaged_verification(1, confidence=float("nan")),
        _damaged_verification(1, confidence=float("inf")),
        _damaged_verification(1, confidence=-0.01),
        _damaged_verification(1, confidence=1.01),
        _damaged_verification(1, visual_evidence=None),
        _damaged_verification(1, effort_evidence=123),
    ],
)
def test_invalid_values_do_not_overwrite_already_accepted_verification(
    verification: object,
) -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)
    accepted = _verification(pending_ns + 10)
    controller.accept_grasp_verification(accepted)

    with pytest.raises(ValueError):
        controller.accept_grasp_verification(verification)  # type: ignore[arg-type]

    controller.step(_actual(pending_ns + 100_000_000), pending_ns + 100_000_000)
    assert controller.latest_grasp_verification is accepted


@pytest.mark.parametrize(
    "missing_field",
    (
        "is_grasped",
        "confidence",
        "visual_evidence",
        "effort_evidence",
        "success",
        "failure_reason",
        "timestamp_ns",
    ),
)
def test_incomplete_verification_is_rejected_without_overwriting_cache(
    missing_field: str,
) -> None:
    controller = _controller()
    _, pending_ns = _advance_pick_to_verify(controller)
    accepted = _verification(pending_ns + 10)
    controller.accept_grasp_verification(accepted)
    damaged = _damaged_verification(pending_ns + 20)
    object.__delattr__(damaged, missing_field)

    with pytest.raises(ValueError, match="字段不完整"):
        controller.accept_grasp_verification(damaged)

    controller.step(_actual(pending_ns + 100_000_000), pending_ns + 100_000_000)
    assert controller.latest_grasp_verification is accepted
