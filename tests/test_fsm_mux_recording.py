"""FSM、控制安全、数据记录与机械臂执行骨架的纯 Python 回归测试。

本文件覆盖 ``interfaces.py`` 中的 19 维动作顺序和 JSON 往返、
``InstructionParser``、``GlobalFSM``、``ActionMux``、``EpisodeRecorder``，以及
``ArmExecutionController`` 已实现的入口校验和安全保持骨架。主要负责人是 FSM、系统、
Recorder 和机械臂2负责人。本文件以单元测试和安全回归测试为主，也包含少量把公共接口
串起来的轻量集成测试；不启动 ROS2 节点、官方 Server 或真实 ``ros2 bag``，因此不需要
官方 Docker 环境。

测试使用 pytest 提供的 ``tmp_path`` 临时目录和 ``monkeypatch`` 临时替换外部依赖，
并用 ``SimpleNamespace`` 构造损坏输入。fake 用可控对象代替真实依赖；本文件主要用
``monkeypatch`` 临时替换原子写、路径探测和时间来源，而 mock 通常还可记录调用。
pytest 会在单个测试结束后恢复这些替换。以 ``test_`` 开头的函数是测试用例，``assert``
表示必须满足的条件，``pytest.raises`` 表示预期必须抛出异常，``parametrize`` 会用
多组输入重复检查同一规则。fixture（如 ``tmp_path`` 和 ``monkeypatch``）由 pytest
注入，测试函数不会参与机器人正式运行。

测试通过可以证明纯 Python 下的任务解析、FSM 转换、19 维仲裁与安全语义、Recorder
路径和事务语义、机械臂执行入口校验符合当前约定；不能证明真实 ROS 话题通信、rosbag
录制、机械臂实际执行、三个任务端到端完成或最终比赛得分。

单文件运行：
``python3 -m pytest -q tests/test_fsm_mux_recording.py -p no:cacheprovider``

全套运行：
``PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider``
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import team_sorting.recorder as recorder_module
from team_sorting.action_mux import ActionMux, ActionMuxConfig
from team_sorting.arm_execution import ArmExecutionController
from team_sorting.fsm import FSMEvent, GlobalFSM, InstructionParser
from team_sorting.interfaces import (
    ACTION_NAMES,
    ArmExecutionConfig,
    ArmMotionPhase,
    BaseCommand,
    FSMStatus,
    GlobalPhase,
    JOINT_NAMES,
    JointTrajectory,
    JointWaypoint,
    LocalPhase,
    ManipulationCommand,
    RobotJointState,
    final_action_from_json,
    final_action_to_json,
)
from team_sorting.recorder import EpisodeRecorder


# 共享构造器和故障注入
def _fail_next_metadata_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    """让下一次metadata原子替换失败，之后自动恢复真实写入。"""

    real_replace = recorder_module.os.replace
    should_fail = True

    def _replace(source: str | Path, target: str | Path) -> None:
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise OSError("模拟metadata原子替换失败")
        real_replace(source, target)

    monkeypatch.setattr(recorder_module.os, "replace", _replace)


def _joints() -> RobotJointState:
    return RobotJointState(
        position=(0.10, 0.05, -0.20, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                  0.40, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, 0.45),
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=1_000,
    )


def _actual_joints(
    position: tuple[float, ...] | None = None,
    timestamp_ns: int = 1_000,
    *,
    valid: bool = True,
    failure_reason: str = "",
) -> RobotJointState:
    base = _joints()
    return RobotJointState(
        position=base.position if position is None else position,
        velocity=base.velocity,
        effort=base.effort,
        timestamp_ns=timestamp_ns,
        valid=valid,
        failure_reason=failure_reason,
    )


def _execution_config(**overrides: object) -> ArmExecutionConfig:
    values: dict[str, object] = {
        "feedback_max_age_ns": 1_000_000_000,
        "trajectory_max_age_ns": 1_000_000_000,
        "command_ttl_ns": 100_000_000,
        "max_control_period_ns": 200_000_000,
        "verification_timeout_ns": 1_000_000_000,
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


def _execution_controller(**overrides: object) -> ArmExecutionController:
    return ArmExecutionController(_execution_config(**overrides))


def _status(
    phase: GlobalPhase = GlobalPhase.SEARCH_TARGET,
    failure_reason: str = "",
) -> FSMStatus:
    return FSMStatus(
        task_id=1,
        global_phase=phase,
        local_phase=LocalPhase.APPROACH,
        retry_count=0,
        success=phase is GlobalPhase.DONE,
        failure_reason=failure_reason,
        timestamp_ns=1_000,
    )


def _official_task_data(**overrides: object) -> dict[str, object]:
    task: dict[str, object] = {
        "task_id": 1,
        "instruction": "把粉色箱体放到桌面目标点",
        "target_kind": "cuboid_box",
        "target_body": "box_pink",
        "target_color": "pink",
        "place_type": "table_point",
        "place_world": [1.0, 2.0, 0.8],
        "place_radius": 0.1,
    }
    task.update(overrides)
    return task

def _loaded_fsm(
    max_pick_retries: int = 1,
    # ————————————————————————————————
    # 【Codex修改-17：测试辅助构造器接收超时策略】
    # 防止测试辅助层忽略显式超时配置，从而让边界测试实际运行在默认无超时模式。
    phase_timeouts_ns: dict[GlobalPhase, int] | None = None,
    # ————————————————————————————————
) -> GlobalFSM:
    raw = json.dumps(_official_task_data())
    task = InstructionParser().parse(raw, 100)[0]
    fsm = GlobalFSM(
        max_pick_retries=max_pick_retries,
        # ————————————————————————————————
        # 【Codex修改-18：测试辅助构造器传递超时策略】
        # 防止调用GlobalFSM时漏传上方策略，确保测试观察到真实配置行为。
        phase_timeouts_ns=phase_timeouts_ns,
        # ————————————————————————————————
    )
    assert fsm.handle_event(FSMEvent.SYSTEM_READY, 110)
    assert fsm.submit_task(task, 120)
    return fsm


def _advance(fsm: GlobalFSM, *events: FSMEvent, start_ns: int = 1_000) -> None:
    for timestamp_ns, event in enumerate(events, start=start_ns):
        assert fsm.handle_event(event, timestamp_ns)


def _advance_to_verify_place(fsm: GlobalFSM, *, start_ns: int = 1_000) -> None:
    _advance(
        fsm,
        FSMEvent.TARGET_FOUND,
        FSMEvent.PICK_NAV_REACHED,
        FSMEvent.TARGET_REFINED,
        FSMEvent.PICK_PLAN_READY,
        FSMEvent.PICK_EXECUTED,
        FSMEvent.PICK_VERIFIED,
        FSMEvent.PLACE_NAV_REACHED,
        FSMEvent.PLACE_PLAN_READY,
        FSMEvent.PLACE_EXECUTED,
        start_ns=start_ns,
    )
    assert fsm.phase is GlobalPhase.VERIFY_PLACE


def _execution_waypoint(
    time_from_start_s: float = 0.0,
    controlled_mask: tuple[bool, ...] = (
        True, False, False, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True,
    ),
    phase: ArmMotionPhase = ArmMotionPhase.PREGRASP,
    joint_position: tuple[float, ...] | None = None,
) -> JointWaypoint:
    return JointWaypoint(
        phase,
        time_from_start_s,
        _joints().position if joint_position is None else joint_position,
        controlled_mask,
    )


def _damaged_execution_waypoint(
    *,
    phase: object = ArmMotionPhase.PREGRASP,
    time_from_start_s: object = 0.0,
    joint_position: object | None = None,
    controlled_mask: object | None = None,
) -> JointWaypoint:
    waypoint = object.__new__(JointWaypoint)
    object.__setattr__(waypoint, "phase", phase)
    object.__setattr__(waypoint, "time_from_start_s", time_from_start_s)
    object.__setattr__(
        waypoint,
        "joint_position",
        _joints().position if joint_position is None else joint_position,
    )
    object.__setattr__(
        waypoint,
        "controlled_mask",
        (True, False, False) + (True,) * 14
        if controlled_mask is None
        else controlled_mask,
    )
    return waypoint


def _execution_trajectory(
    trajectory_id: str = "trajectory-1",
    waypoints: tuple[object, ...] | None = None,
    *,
    valid: bool = True,
    failure_reason: str = "",
    execution_phase: object = GlobalPhase.EXECUTE_PICK,
    target_body: str = "box",
) -> JointTrajectory:
    selected_waypoints = (
        (
            _execution_waypoint(0.0, phase=ArmMotionPhase.PREGRASP),
            _execution_waypoint(1.0, phase=ArmMotionPhase.GRASP),
            _execution_waypoint(2.0, phase=ArmMotionPhase.LIFT),
            _execution_waypoint(3.0, phase=ArmMotionPhase.RETREAT),
        )
        if waypoints is None
        else waypoints
    )
    # 机械臂执行边界必须防御来自反序列化/进程边界的损坏对象；这里有意绕过
    # frozen dataclass构造校验，保留既有ArmExecutionController防御测试。
    trajectory = object.__new__(JointTrajectory)
    for name, value in (
        ("trajectory_id", trajectory_id),
        ("task_id", 1),
        ("target_body", target_body),
        ("execution_phase", execution_phase),
        ("waypoints", selected_waypoints),
        ("timestamp_ns", 2_000),
        ("valid", valid),
        ("failure_reason", failure_reason),
    ):
        object.__setattr__(trajectory, name, value)
    return trajectory


# FinalAction与19维顺序
def test_19d_order_json_roundtrip_and_expired_commands() -> None:
    joints = _joints()
    status = _status()
    mux = ActionMux()

    target = tuple(value + 0.01 for value in joints.position)
    active = ManipulationCommand(
        joint_target=target,
        controlled_mask=(True,) * 17,
        local_phase=LocalPhase.MOVE_PREGRASP,
        timestamp_ns=1_000,
        valid_until_ns=2_000,
    )
    action = mux.compose(BaseCommand(0.12, -0.25, 1_000, 2_000), active, joints, status, 1_500)
    assert len(ACTION_NAMES) == len(action.values) == 19
    assert action.values[:2] == pytest.approx((0.12, -0.25))
    assert action.values[2:] == pytest.approx(target)
    assert ACTION_NAMES[2:5] == ("slide", "head_yaw", "head_pitch")
    assert ACTION_NAMES[5:12] == (
        "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
        "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6", "left_gripper",
    )
    assert final_action_from_json(final_action_to_json(action)) == action

    expired = mux.compose(
        BaseCommand(0.20, 0.30, 1_000, 1_100),
        ManipulationCommand(
            joint_target=(0.0,) * 17,
            controlled_mask=(True,) * 17,
            local_phase=LocalPhase.APPROACH,
            timestamp_ns=1_000,
            valid_until_ns=1_100,
        ),
        joints,
        status,
        1_500,
    )
    assert expired.values[:2] == (0.0, 0.0)
    assert expired.values[2:] == joints.position
    assert "过期" in expired.failure_reason


# ActionMux安全仲裁
def test_controlled_mask_only_overrides_true_positions() -> None:
    joints = _joints()
    target = tuple(0.0 for _ in range(17))
    mask = tuple(index in {3, 10} for index in range(17))
    command = ManipulationCommand(
        joint_target=target,
        controlled_mask=mask,
        local_phase=LocalPhase.APPROACH,
        timestamp_ns=1_000,
        valid_until_ns=2_000,
    )
    action = ActionMux().compose(
        BaseCommand(0.10, -0.20, 1_000, 2_000),
        command,
        joints,
        _status(),
        1_500,
    )

    assert len(action.values) == 19
    for index, controlled in enumerate(mask):
        expected = target[index] if controlled else joints.position[index]
        assert action.values[index + 2] == expected


def test_uncontrolled_actual_joint_outside_bounds_is_not_silently_clipped() -> None:
    position = list(_joints().position)
    position[0] = 0.90
    joints = RobotJointState(
        position=tuple(position),
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=1_000,
    )
    command = ManipulationCommand(
        joint_target=(0.0,) * 17,
        controlled_mask=(False,) * 17,
        local_phase=LocalPhase.APPROACH,
        timestamp_ns=1_000,
        valid_until_ns=2_000,
    )
    action = ActionMux().compose(BaseCommand(0.0, 0.0, 1_000, 2_000), command, joints, _status(), 1_500)

    assert action.values[2] == 0.90
    assert not action.valid
    assert not action.clipped
    assert "实际关节第 0 项超出配置边界" in action.failure_reason
    assert "保持实际反馈值" in action.failure_reason


@pytest.mark.parametrize(
    "phase",
    (
        GlobalPhase.WAIT_READY,
        GlobalPhase.LOAD_TASK,
        GlobalPhase.DONE,
        GlobalPhase.SAFE_HOLD,
        GlobalPhase.FAILED,
    ),
)
def test_stop_phases_zero_base_and_hold_actual_joints(phase: GlobalPhase) -> None:
    joints = _joints()
    target = tuple(value + 0.02 for value in joints.position)
    command = ManipulationCommand(
        joint_target=target,
        controlled_mask=(True,) * 17,
        local_phase=LocalPhase.APPROACH,
        timestamp_ns=1_000,
        valid_until_ns=9_000,
    )
    action = ActionMux().compose(
        BaseCommand(0.20, -0.30, 1_000, 9_000),
        command,
        joints,
        _status(phase),
        2_000,
    )

    assert action.values[:2] == (0.0, 0.0)
    assert action.values[2:] == joints.position
    assert action.global_phase is phase
    assert action.local_phase is LocalPhase.APPROACH
    assert phase.value in action.failure_reason
    assert "覆盖" in action.failure_reason


def test_invalid_actual_joints_ignore_candidates_and_preserve_reason() -> None:
    source = _joints()
    joints = RobotJointState(
        position=source.position,
        velocity=source.velocity,
        effort=source.effort,
        timestamp_ns=source.timestamp_ns,
        valid=False,
        failure_reason="关节反馈时间过期",
    )
    target = tuple(value + 0.02 for value in joints.position)
    command = ManipulationCommand(
        joint_target=target,
        controlled_mask=(True,) * 17,
        local_phase=LocalPhase.APPROACH,
        timestamp_ns=1_000,
        valid_until_ns=9_000,
    )
    action = ActionMux().compose(
        BaseCommand(0.20, -0.30, 1_000, 9_000),
        command,
        joints,
        _status(),
        2_000,
    )

    assert action.values[:2] == (0.0, 0.0)
    assert action.values[2:] == joints.position
    assert not action.valid
    assert "关节反馈时间过期" in action.failure_reason
    assert "忽略普通底盘和机械臂候选命令" in action.failure_reason


def test_failure_reasons_are_merged_without_losing_actual_feedback_reason() -> None:
    position = list(_joints().position)
    position[0] = 0.90
    joints = RobotJointState(
        position=tuple(position),
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=1_000,
        failure_reason="关节反馈边界待核对",
    )
    command = ManipulationCommand(
        joint_target=joints.position,
        controlled_mask=(False,) * 17,
        local_phase=LocalPhase.APPROACH,
        timestamp_ns=1_000,
        valid_until_ns=2_000,
        valid=False,
        failure_reason="机械臂轨迹无效",
    )
    action = ActionMux().compose(None, command, joints, _status(), 1_500)

    assert not action.valid
    assert action.values[:2] == (0.0, 0.0)
    assert action.values[2] == 0.90
    assert "关节反馈边界待核对" in action.failure_reason
    assert "无底盘候选命令" in action.failure_reason
    assert "机械臂轨迹无效" in action.failure_reason
    assert "实际关节第 0 项超出配置边界" in action.failure_reason


@pytest.mark.parametrize(
    ("v", "w"),
    (
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (0.0, float("-inf")),
        (True, 0.0),
        (0.0, False),
        ("错误速度", 0.0),
    ),
)
def test_invalid_base_speed_safely_falls_back_to_zero(v: object, w: object) -> None:
    joints = _joints()
    command = ManipulationCommand(
        joint_target=joints.position,
        controlled_mask=(False,) * 17,
        local_phase=LocalPhase.APPROACH,
        timestamp_ns=1_000,
        valid_until_ns=2_000,
    )
    # Interface v1构造边界已经拒绝这些值；这里显式绕过dataclass，只保留ActionMux面对
    # 内存损坏/不可信旧对象时仍安全降级的防御纵深回归。
    malformed = object.__new__(BaseCommand)
    object.__setattr__(malformed, "v", v)
    object.__setattr__(malformed, "w", w)
    object.__setattr__(malformed, "timestamp_ns", 1_000)
    object.__setattr__(malformed, "valid_until_ns", 2_000)
    object.__setattr__(malformed, "valid", True)
    object.__setattr__(malformed, "failure_reason", "")
    action = ActionMux().compose(
        malformed,
        command,
        joints,
        _status(),
        1_500,
    )

    assert action.values[:2] == (0.0, 0.0)
    assert action.values[2:] == joints.position
    assert "底盘候选速度" in action.failure_reason


def test_ttl_equal_to_now_is_expired_for_base_and_manipulation() -> None:
    joints = _joints()
    target = tuple(value + 0.02 for value in joints.position)
    command = ManipulationCommand(
        joint_target=target,
        controlled_mask=(True,) * 17,
        local_phase=LocalPhase.APPROACH,
        timestamp_ns=1_000,
        valid_until_ns=2_000,
    )
    action = ActionMux().compose(
        BaseCommand(0.20, -0.30, 1_000, 2_000),
        command,
        joints,
        _status(),
        2_000,
    )

    assert action.values[:2] == (0.0, 0.0)
    assert action.values[2:] == joints.position
    assert action.failure_reason.count("已过期") == 2


def test_controlled_joint_is_clipped_and_sequence_increments() -> None:
    joints = _joints()
    target = list(joints.position)
    target[0] = 1.50
    mask = tuple(index == 0 for index in range(17))
    command = ManipulationCommand(
        joint_target=tuple(target),
        controlled_mask=mask,
        local_phase=LocalPhase.APPROACH,
        timestamp_ns=1_000,
        valid_until_ns=9_000,
    )
    mux = ActionMux()
    first = mux.compose(BaseCommand(0.0, 0.0, 1_000, 9_000), command, joints, _status(), 2_000)
    second = mux.compose(BaseCommand(0.0, 0.0, 1_000, 9_000), command, joints, _status(), 2_001)

    assert first.values[2] == mux.config.joint_upper[0]
    assert first.values[3:] == joints.position[1:]
    assert first.clipped
    assert len(first.values) == len(second.values) == 19
    assert (first.sequence, second.sequence) == (1, 2)


def test_action_mux_config_rejects_bool_and_string_limits() -> None:
    defaults = ActionMuxConfig.conservative_defaults()
    lower_with_bool = list(defaults.joint_lower)
    lower_with_bool[0] = True
    upper_with_string = list(defaults.joint_upper)
    upper_with_string[0] = "0.87"
    invalid_configs = (
        (True, defaults.max_abs_base_w, defaults.joint_lower, defaults.joint_upper),
        ("0.25", defaults.max_abs_base_w, defaults.joint_lower, defaults.joint_upper),
        (float("nan"), defaults.max_abs_base_w, defaults.joint_lower, defaults.joint_upper),
        (defaults.max_abs_base_v, float("inf"), defaults.joint_lower, defaults.joint_upper),
        (defaults.max_abs_base_v, defaults.max_abs_base_w, tuple(lower_with_bool), defaults.joint_upper),
        (defaults.max_abs_base_v, defaults.max_abs_base_w, defaults.joint_lower, tuple(upper_with_string)),
    )
    for max_v, max_w, lower, upper in invalid_configs:
        with pytest.raises(ValueError):
            ActionMuxConfig(max_v, max_w, lower, upper)  # type: ignore[arg-type]


# InstructionParser（含一条基础FSM接线回归）
def test_instruction_parser_and_basic_fsm_transitions() -> None:
    raw = json.dumps(
        {
            "tasks": [
                {
                    "task_id": 7,
                        "instruction": "把粉色箱体放到目标位置",
                        "target_kind": "cuboid_box",
                        "target_body": "box",
                    "target_color": "pink",
                        "place_type": "table_point",
                    "place_world": [1.0, 2.0, 0.8],
                    "place_radius": 0.1,
                }
            ]
        },
        ensure_ascii=False,
    )
    task = InstructionParser().parse(raw, 100)[0]
    assert task.place_frame_id == "world"
    assert task.task_id == 7
    assert task.place_world_xyz == (1.0, 2.0, 0.8)

    fsm = GlobalFSM(max_pick_retries=1)
    assert fsm.handle_event(FSMEvent.SYSTEM_READY, 110)
    assert fsm.submit_task(task, 120)
    assert fsm.status(120).global_phase is GlobalPhase.SEARCH_TARGET
    assert fsm.handle_event(FSMEvent.TARGET_FOUND, 130)
    assert fsm.status(130).global_phase is GlobalPhase.NAV_TO_PICK
    assert not fsm.handle_event(FSMEvent.PICK_EXECUTED, 140)


# GlobalFSM
def test_done_and_failed_are_terminal_except_reset() -> None:
    done_fsm = _loaded_fsm()
    _advance(
        done_fsm,
        FSMEvent.TARGET_FOUND,
        FSMEvent.PICK_NAV_REACHED,
        FSMEvent.TARGET_REFINED,
        FSMEvent.PICK_PLAN_READY,
        FSMEvent.PICK_EXECUTED,
        FSMEvent.PICK_VERIFIED,
        FSMEvent.PLACE_NAV_REACHED,
        FSMEvent.PLACE_PLAN_READY,
        FSMEvent.PLACE_EXECUTED,
        FSMEvent.PLACE_VERIFIED,
        FSMEvent.RETURN_REACHED,
    )
    assert done_fsm.phase is GlobalPhase.DONE
    # ————————————————————————————————
    # 【Codex修改-19：DONE终止态保护】
    # 防止迟到失败或安全事件改写已经完成的任务结果。
    assert done_fsm.status(2_000).success is True
    assert done_fsm.status(2_000).task_id == 1
    # ————————————————————————————————
    assert not done_fsm.handle_event(FSMEvent.FAILURE, 2_000, "迟到的失败事件")
    # ————————————————————————————————
    # 【Codex修改-20：DONE拒绝安全控制事件】
    # 防止迟到的安全暂停或恢复事件让已完成任务重新进入活动状态。
    assert not done_fsm.handle_event(FSMEvent.SAFETY_HOLD, 2_001, "迟到的安全事件")
    assert not done_fsm.handle_event(FSMEvent.SAFETY_RECOVERED, 2_002)
    # ————————————————————————————————
    assert done_fsm.phase is GlobalPhase.DONE
    assert done_fsm.failure_reason == ""
    assert done_fsm.handle_event(FSMEvent.RESET, 2_003)
    assert done_fsm.phase is GlobalPhase.WAIT_READY

    failed_fsm = GlobalFSM()
    assert failed_fsm.handle_event(FSMEvent.FAILURE, 3_000, "启动失败")
    assert failed_fsm.phase is GlobalPhase.FAILED
    assert not failed_fsm.handle_event(FSMEvent.SYSTEM_READY, 3_001)
    assert not failed_fsm.handle_event(FSMEvent.FAILURE, 3_002, "迟到的第二个失败")
    # ————————————————————————————————
    # 【Codex修改-21：FAILED终止态保护】
    # 防止普通业务、安全恢复或重复失败让FAILED终态重新进入活动流程。
    assert not failed_fsm.handle_event(FSMEvent.SAFETY_HOLD, 3_003, "迟到的安全事件")
    assert not failed_fsm.handle_event(FSMEvent.SAFETY_RECOVERED, 3_004)
    # ————————————————————————————————
    assert failed_fsm.phase is GlobalPhase.FAILED
    assert failed_fsm.failure_reason == "启动失败"
    assert failed_fsm.handle_event(FSMEvent.RESET, 3_005)
    assert failed_fsm.phase is GlobalPhase.WAIT_READY


# ————————————————————————————————
# 【Codex修改-22：恢复抓取后的快照使用合法时间】
# 防止测试用早于DONE转换的时间读取状态，同时继续保护旧抓取失败原因被成功验证清除。
# ————————————————————————————————
def test_recovered_pick_failure_reason_is_cleared() -> None:
    fsm = _loaded_fsm(max_pick_retries=1)
    _advance(
        fsm,
        FSMEvent.TARGET_FOUND,
        FSMEvent.PICK_NAV_REACHED,
        FSMEvent.TARGET_REFINED,
        FSMEvent.PICK_PLAN_READY,
        FSMEvent.PICK_EXECUTED,
    )
    assert fsm.handle_event(FSMEvent.PICK_FAILED, 2_000, "第一次抓取验证失败")
    assert fsm.failure_reason == "第一次抓取验证失败"
    _advance(
        fsm,
        FSMEvent.TARGET_REFINED,
        FSMEvent.PICK_PLAN_READY,
        FSMEvent.PICK_EXECUTED,
        FSMEvent.PICK_VERIFIED,
        start_ns=2_100,
    )
    assert fsm.phase is GlobalPhase.NAV_TO_PLACE
    assert fsm.failure_reason == ""
    _advance(
        fsm,
        FSMEvent.PLACE_NAV_REACHED,
        FSMEvent.PLACE_PLAN_READY,
        FSMEvent.PLACE_EXECUTED,
        FSMEvent.PLACE_VERIFIED,
        FSMEvent.RETURN_REACHED,
        start_ns=3_000,
    )
    assert fsm.phase is GlobalPhase.DONE
    assert fsm.status(3_004).failure_reason == ""


def test_exhausted_pick_retry_reason_survives_return_to_failed() -> None:
    fsm = _loaded_fsm(max_pick_retries=1)
    _advance(
        fsm,
        FSMEvent.TARGET_FOUND,
        FSMEvent.PICK_NAV_REACHED,
        FSMEvent.TARGET_REFINED,
        FSMEvent.PICK_PLAN_READY,
        FSMEvent.PICK_EXECUTED,
    )
    assert fsm.handle_event(FSMEvent.PICK_FAILED, 2_000, "第一次抓取验证失败")
    _advance(
        fsm,
        FSMEvent.TARGET_REFINED,
        FSMEvent.PICK_PLAN_READY,
        FSMEvent.PICK_EXECUTED,
        start_ns=2_100,
    )
    assert fsm.handle_event(FSMEvent.PICK_FAILED, 2_200, "重试耗尽")
    assert fsm.phase is GlobalPhase.RETURN_END
    assert fsm.failure_reason == "重试耗尽"
    assert fsm.handle_event(FSMEvent.RETURN_REACHED, 2_300)
    assert fsm.phase is GlobalPhase.FAILED
    assert fsm.status(2_300).failure_reason == "重试耗尽"


def test_place_failure_uses_existing_unrecoverable_failure_event() -> None:
    assert "PLACE_FAILED" not in FSMEvent.__members__
    fsm = _loaded_fsm()
    _advance_to_verify_place(fsm, start_ns=200)

    assert fsm.handle_event(FSMEvent.FAILURE, 300, "放置验证失败")
    assert fsm.phase is GlobalPhase.RETURN_END
    assert fsm.failure_reason == "放置验证失败"
    assert fsm.handle_event(FSMEvent.RETURN_REACHED, 301)
    assert fsm.phase is GlobalPhase.FAILED


# 【Codex修改-23：状态读取无副作用】
# 防止phase_elapsed_ns或status读取推进FSM、触发超时或刷新阶段进入时间。
# ————————————————————————————————
def test_fsm_phase_time_and_status_are_side_effect_free() -> None:
    fsm = GlobalFSM()
    assert fsm.phase_entered_ns is None
    assert fsm.phase_elapsed_ns(50) is None
    assert fsm.handle_event(FSMEvent.SYSTEM_READY, 100)
    assert fsm.phase_entered_ns == 100
    assert fsm.phase_elapsed_ns(125) == 25
    with pytest.raises(ValueError, match="早于"):
        fsm.phase_elapsed_ns(99)

    before = fsm.status(150)
    after = fsm.status(250)
    assert before.global_phase is after.global_phase is GlobalPhase.LOAD_TASK
    assert before.retry_count == after.retry_count == 0
    assert fsm.phase_entered_ns == 100

    assert not fsm.handle_event(FSMEvent.TARGET_FOUND, 300)
    assert fsm.phase_entered_ns == 100


def test_fsm_legal_transitions_update_phase_entry_time_and_local_is_telemetry() -> None:
    fsm = _loaded_fsm()
    assert fsm.phase_entered_ns == 120
    fsm.set_local_phase(LocalPhase.APPROACH)
    assert fsm.phase is GlobalPhase.SEARCH_TARGET
    assert fsm.phase_entered_ns == 120
    assert fsm.status(125).local_phase is LocalPhase.APPROACH

    assert fsm.handle_event(FSMEvent.TARGET_FOUND, 130)
    assert fsm.phase is GlobalPhase.NAV_TO_PICK
    assert fsm.phase_entered_ns == 130


@pytest.mark.parametrize("place_type", ["world_point", "world", "relative", "point"])
def test_instruction_parser_rejects_non_official_place_type(place_type: str) -> None:
    raw = json.dumps(_official_task_data(place_type=place_type))
    with pytest.raises(ValueError, match="place_type"):
        InstructionParser().parse(raw, 100)


def test_fsm_rejects_negative_and_stale_mutating_timestamps() -> None:
    fsm = _loaded_fsm()
    with pytest.raises(ValueError, match="timestamp_ns"):
        fsm.handle_event(FSMEvent.TARGET_FOUND, -1)
    assert fsm.phase is GlobalPhase.SEARCH_TARGET
    assert fsm.phase_entered_ns == 120

    assert fsm.handle_event(FSMEvent.TARGET_FOUND, 200)
    assert not fsm.handle_event(FSMEvent.PICK_NAV_REACHED, 199)
    assert fsm.phase is GlobalPhase.NAV_TO_PICK
    assert fsm.phase_entered_ns == 200

    task = fsm.task
    assert task is not None
    assert not fsm.submit_task(task, 199)
    with pytest.raises(ValueError, match="timestamp_ns"):
        fsm.submit_task(task, -1)


# ————————————————————————————————
# 【Codex修改-24：恢复使用真实时间且拒绝暂停前旧事件】
# 防止恢复后伪造阶段进入时间，或让暂停前产生的迟到业务回执推进已恢复阶段。
# ————————————————————————————————
def test_late_events_cannot_cross_safe_hold_recovery_or_reset_boundary() -> None:
    fsm = _loaded_fsm()
    assert fsm.handle_event(FSMEvent.SAFETY_HOLD, 200, "急停输入短暂失效")
    assert fsm.handle_event(FSMEvent.SAFETY_RECOVERED, 300)
    assert fsm.phase is GlobalPhase.SEARCH_TARGET
    assert fsm.phase_entered_ns == 300
    assert fsm._phase_elapsed_offset_ns == 80

    # SAFE_HOLD前产生、恢复后才送达的旧回执不能推进恢复后的阶段。
    assert not fsm.handle_event(FSMEvent.TARGET_FOUND, 250)
    assert fsm.phase is GlobalPhase.SEARCH_TARGET
    assert fsm.phase_entered_ns == 300

    assert fsm.handle_event(FSMEvent.RESET, 400)
    assert not fsm.handle_event(FSMEvent.SYSTEM_READY, 399)
    assert fsm.phase is GlobalPhase.WAIT_READY
    assert fsm.phase_entered_ns == 400


def test_no_timeout_policy_preserves_compatibility() -> None:
    fsm = _loaded_fsm()
    assert not fsm.check_timeout(10**15)
    assert fsm.phase is GlobalPhase.SEARCH_TARGET
    assert fsm.phase_entered_ns == 120


# ————————————————————————————————
# 配置的阶段超时边界直接进入 FAILED，不自动恢复或成功。
# ————————————————————————————————
def test_configured_timeout_uses_exact_boundary_and_enters_failed() -> None:
    fsm = _loaded_fsm(phase_timeouts_ns={GlobalPhase.SEARCH_TARGET: 10})
    assert not fsm.check_timeout(129)
    assert fsm.phase is GlobalPhase.SEARCH_TARGET

    assert fsm.check_timeout(130)
    status = fsm.status(130)
    assert status.global_phase is GlobalPhase.FAILED
    assert status.success is False
    assert "SEARCH_TARGET阶段超时" in status.failure_reason
    assert "实际活动时间10ns" in status.failure_reason
    assert "配置限制10ns" in status.failure_reason
    assert not fsm.handle_event(FSMEvent.SAFETY_RECOVERED, 131)


def test_external_safe_hold_saves_elapsed_time_before_pause() -> None:
    fsm = _loaded_fsm(phase_timeouts_ns={GlobalPhase.SEARCH_TARGET: 100})
    assert fsm.handle_event(FSMEvent.SAFETY_HOLD, 150, "外部安全监控暂停")

    assert fsm.phase is GlobalPhase.SAFE_HOLD
    assert fsm._interrupted_phase is GlobalPhase.SEARCH_TARGET
    assert fsm._interrupted_elapsed_ns == 30
    assert fsm.phase_entered_ns == 150
    assert fsm._phase_elapsed_offset_ns == 0


def test_safe_hold_recovery_preserves_elapsed_timeout_budget() -> None:
    fsm = _loaded_fsm(phase_timeouts_ns={GlobalPhase.SEARCH_TARGET: 100})
    assert fsm.phase_elapsed_ns(150) == 30
    assert fsm.handle_event(FSMEvent.SAFETY_HOLD, 150, "短暂安全暂停")
    assert fsm.handle_event(FSMEvent.SAFETY_RECOVERED, 1_000)

    assert fsm.phase is GlobalPhase.SEARCH_TARGET
    assert fsm.phase_entered_ns == 1_000
    assert fsm._phase_elapsed_offset_ns == 30
    assert fsm.phase_elapsed_ns(1_000) == 30
    assert not fsm.check_timeout(1_069)
    assert fsm.check_timeout(1_070)
    assert fsm.phase is GlobalPhase.FAILED


def test_multiple_safe_holds_accumulate_only_active_elapsed_time() -> None:
    fsm = _loaded_fsm(phase_timeouts_ns={GlobalPhase.SEARCH_TARGET: 100})
    assert fsm.handle_event(FSMEvent.SAFETY_HOLD, 150, "第一次暂停")
    assert fsm.handle_event(FSMEvent.SAFETY_RECOVERED, 1_000)
    assert fsm.phase_elapsed_ns(1_010) == 40
    assert fsm.handle_event(FSMEvent.SAFETY_HOLD, 1_010, "第二次暂停")
    assert fsm._interrupted_elapsed_ns == 40
    assert fsm.handle_event(FSMEvent.SAFETY_RECOVERED, 2_000)

    assert fsm.phase_elapsed_ns(2_000) == 40
    assert not fsm.check_timeout(2_059)
    assert fsm.check_timeout(2_060)
    assert fsm.phase is GlobalPhase.FAILED



def test_safe_hold_preserves_task_retry_and_recovers_without_skipping() -> None:
    fsm = _loaded_fsm(max_pick_retries=2)
    _advance(
        fsm,
        FSMEvent.TARGET_FOUND,
        FSMEvent.PICK_NAV_REACHED,
        FSMEvent.TARGET_REFINED,
        FSMEvent.PICK_PLAN_READY,
        FSMEvent.PICK_EXECUTED,
        start_ns=200,
    )
    assert fsm.handle_event(FSMEvent.PICK_FAILED, 300, "一次可恢复抓取失败")
    task = fsm.task
    assert task is not None
    assert fsm.retry_count == 1
    assert fsm.phase is GlobalPhase.REFINE_TARGET

    assert fsm.handle_event(FSMEvent.SAFETY_HOLD, 310, "传感器短暂失效")
    status = fsm.status(311)
    assert status.task_id == task.task_id
    assert status.retry_count == 1
    assert status.success is False
    assert status.failure_reason == (
        "原失败：一次可恢复抓取失败；安全暂停：传感器短暂失效"
    )
    assert fsm.task is task
    assert fsm.failure_reason == "一次可恢复抓取失败"
    assert not fsm.handle_event(FSMEvent.TARGET_REFINED, 312)
    assert not fsm.handle_event(FSMEvent.RETURN_REACHED, 313)
    assert fsm.phase is GlobalPhase.SAFE_HOLD

    assert fsm.handle_event(FSMEvent.SAFETY_RECOVERED, 320)
    assert fsm.phase is GlobalPhase.REFINE_TARGET
    assert fsm.phase_entered_ns == 320
    assert fsm._phase_elapsed_offset_ns == 10
    assert fsm.phase_elapsed_ns(320) == 10
    assert fsm.failure_reason == "一次可恢复抓取失败"


def test_safe_hold_rejects_wrong_recovery_and_never_jumps_to_done() -> None:
    fsm = _loaded_fsm()
    assert not fsm.handle_event(FSMEvent.SAFETY_RECOVERED, 150)
    assert fsm.phase is GlobalPhase.SEARCH_TARGET
    assert fsm.handle_event(FSMEvent.SAFETY_HOLD, 160)
    assert not fsm.handle_event(FSMEvent.SAFETY_HOLD, 161)
    assert not fsm.handle_event(FSMEvent.RETURN_REACHED, 162)
    assert fsm.phase is GlobalPhase.SAFE_HOLD
    assert fsm.status(162).success is False


# ————————————————————————————————
# 【Codex修改-31：SAFE_HOLD不可恢复失败直接终止】
# 防止安全条件尚未恢复时进入RETURN_END，并保护任务、重试和一次性组合的诊断原因。
# ————————————————————————————————
def test_safe_hold_failure_enters_failed_and_preserves_diagnostics() -> None:
    fsm = _loaded_fsm(max_pick_retries=2)
    _advance(
        fsm,
        FSMEvent.TARGET_FOUND,
        FSMEvent.PICK_NAV_REACHED,
        FSMEvent.TARGET_REFINED,
        FSMEvent.PICK_PLAN_READY,
        FSMEvent.PICK_EXECUTED,
        start_ns=200,
    )
    assert fsm.handle_event(FSMEvent.PICK_FAILED, 300, "一次可恢复抓取失败")
    task = fsm.task
    assert task is not None
    assert fsm.retry_count == 1
    assert fsm.handle_event(FSMEvent.SAFETY_HOLD, 310, "安全监控暂停")

    assert fsm.handle_event(FSMEvent.FAILURE, 320, "安全条件不可恢复")

    assert fsm.phase is GlobalPhase.FAILED
    assert fsm.phase is not GlobalPhase.RETURN_END
    assert fsm.task is task
    assert fsm.retry_count == 1
    assert fsm.failure_reason == (
        "原失败：一次可恢复抓取失败；安全暂停失败：安全条件不可恢复"
    )
    assert fsm._fail_after_return is False
    assert fsm._interrupted_phase is None
    assert fsm._interrupted_elapsed_ns is None
    assert fsm._phase_elapsed_offset_ns == 0
    assert fsm._safe_hold_reason == ""
    assert not fsm.handle_event(FSMEvent.FAILURE, 321, "重复失败")
    assert fsm.failure_reason == (
        "原失败：一次可恢复抓取失败；安全暂停失败：安全条件不可恢复"
    )


# ————————————————————————————————
# 【Codex修改-32：SAFE_HOLD空失败原因优先保留暂停原因】
# 防止空reason覆盖安全监控提供的唯一可诊断原因。
# ————————————————————————————————
def test_safe_hold_failure_without_reason_uses_safe_hold_reason() -> None:
    fsm = _loaded_fsm()
    assert fsm.handle_event(FSMEvent.SAFETY_HOLD, 150, "急停无法复位")

    assert fsm.handle_event(FSMEvent.FAILURE, 160, "")

    assert fsm.phase is GlobalPhase.FAILED
    assert fsm.failure_reason == "急停无法复位"


def test_failure_without_task_and_with_task_keep_required_return_semantics() -> None:
    without_task = GlobalFSM()
    assert without_task.handle_event(FSMEvent.FAILURE, 10)
    assert without_task.phase is GlobalPhase.FAILED
    assert without_task.failure_reason == "业务模块报告失败"
    assert without_task.phase_entered_ns == 10

    with_task = _loaded_fsm()
    assert with_task.handle_event(FSMEvent.FAILURE, 200, "导航不可恢复")
    assert with_task.phase is GlobalPhase.RETURN_END
    assert with_task.failure_reason == "导航不可恢复"
    assert with_task.handle_event(FSMEvent.RETURN_REACHED, 210)
    assert with_task.phase is GlobalPhase.FAILED
    assert with_task.failure_reason == "导航不可恢复"


# ————————————————————————————————
# 【Codex修改-33：失败返区再次失败立即终止并组合根因】
# 防止返区导航失败后永久等待RETURN_REACHED，同时确保不会重新进入或刷新RETURN_END。
# ————————————————————————————————
def test_return_end_failure_enters_failed_and_combines_root_reason() -> None:
    fsm = _loaded_fsm()
    assert fsm.handle_event(FSMEvent.FAILURE, 200, "导航不可恢复")
    assert fsm.phase is GlobalPhase.RETURN_END
    assert fsm.phase_entered_ns == 200

    assert fsm.handle_event(FSMEvent.FAILURE, 240, "返区期间底盘异常")
    assert fsm.phase is GlobalPhase.FAILED
    assert fsm.failure_reason == "导航不可恢复；返区失败：返区期间底盘异常"
    assert fsm.phase_entered_ns == 240
    assert fsm._last_transition_ns == 240
    assert not fsm.handle_event(FSMEvent.RETURN_REACHED, 300)


# ————————————————————————————————
# 【Codex修改-34：返区失败原因只组合一次】
# 防止FAILED后的重复失败继续追加字符串并造成failure_reason无界增长。
# ————————————————————————————————
def test_return_end_repeated_failure_reason_stays_bounded_and_deduplicated() -> None:
    fsm = _loaded_fsm()
    assert fsm.handle_event(FSMEvent.FAILURE, 200, "最初导航失败")
    assert fsm.handle_event(FSMEvent.FAILURE, 201, "返区底盘故障")
    original_reason = "最初导航失败；返区失败：返区底盘故障"
    assert fsm.failure_reason == original_reason

    for index in range(100):
        repeated_reason = "最初导航失败" if index % 2 == 0 else f"返区错误{index}"
        assert not fsm.handle_event(
            FSMEvent.FAILURE,
            201 + index,
            repeated_reason,
        )

    assert fsm.failure_reason == original_reason
    assert len(fsm.failure_reason) == len(original_reason)
    assert fsm.phase is GlobalPhase.FAILED
    assert fsm.phase_entered_ns == 201
    assert fsm._last_transition_ns == 201


# ————————————————————————————————
# 【Codex修改-35：正常返区首次失败成为最终根因】
# 防止正常完成后的返区故障被忽略，并验证终止后不会再次追加后续错误。
# ————————————————————————————————
def test_first_failure_during_normal_return_becomes_root_reason() -> None:
    fsm = _loaded_fsm()
    _advance(
        fsm,
        FSMEvent.TARGET_FOUND,
        FSMEvent.PICK_NAV_REACHED,
        FSMEvent.TARGET_REFINED,
        FSMEvent.PICK_PLAN_READY,
        FSMEvent.PICK_EXECUTED,
        FSMEvent.PICK_VERIFIED,
        FSMEvent.PLACE_NAV_REACHED,
        FSMEvent.PLACE_PLAN_READY,
        FSMEvent.PLACE_EXECUTED,
        FSMEvent.PLACE_VERIFIED,
    )
    assert fsm.phase is GlobalPhase.RETURN_END
    assert fsm.handle_event(FSMEvent.FAILURE, 2_000, "正常返区时底盘故障")
    assert fsm.failure_reason == "正常返区时底盘故障"
    assert fsm.phase is GlobalPhase.FAILED
    assert fsm.phase_entered_ns == 2_000
    assert fsm._last_transition_ns == 2_000

    assert not fsm.handle_event(FSMEvent.FAILURE, 2_001, "后续返区错误")
    assert fsm.failure_reason == "正常返区时底盘故障"


# ————————————————————————————————
# 【Codex修改-36：RESET允许仿真时钟回退】
# 防止迟到事件检查阻止显式新生命周期在较小仿真时间上重建时间基准。
# ————————————————————————————————
def test_reset_accepts_clock_rollback_and_rebuilds_time_baseline() -> None:
    fsm = _loaded_fsm()
    assert fsm.handle_event(FSMEvent.TARGET_FOUND, 200)

    assert fsm.handle_event(FSMEvent.RESET, 0)
    assert fsm.phase is GlobalPhase.WAIT_READY
    assert fsm.phase_entered_ns == 0
    assert fsm._last_transition_ns == 0
    assert fsm._phase_elapsed_offset_ns == 0
    assert fsm.handle_event(FSMEvent.SYSTEM_READY, 0)
    assert fsm.phase is GlobalPhase.LOAD_TASK


# ————————————————————————————————
# 【Codex修改-37：RESET清理活动时间和安全暂停上下文】
# 防止新生命周期继承旧阶段偏移、暂停原因、被中断阶段或重试诊断状态。
# ————————————————————————————————
def test_reset_clears_all_fsm_runtime_state() -> None:
    fsm = _loaded_fsm(max_pick_retries=1)
    _advance(
        fsm,
        FSMEvent.TARGET_FOUND,
        FSMEvent.PICK_NAV_REACHED,
        FSMEvent.TARGET_REFINED,
        FSMEvent.PICK_PLAN_READY,
        FSMEvent.PICK_EXECUTED,
        start_ns=200,
    )
    assert fsm.handle_event(FSMEvent.PICK_FAILED, 300, "可恢复失败")
    fsm.set_local_phase(LocalPhase.FAILED)
    assert fsm.handle_event(FSMEvent.SAFETY_HOLD, 310, "安全暂停")
    assert fsm.handle_event(FSMEvent.RESET, 400)

    status = fsm.status(401)
    assert status.task_id == -1
    assert status.global_phase is GlobalPhase.WAIT_READY
    assert status.local_phase is LocalPhase.IDLE
    assert status.retry_count == 0
    assert status.success is False
    assert status.failure_reason == ""
    assert fsm.task is None
    assert fsm.failure_reason == ""
    assert fsm.phase_entered_ns == 400
    assert fsm._fail_after_return is False
    assert fsm._interrupted_phase is None
    assert fsm._interrupted_elapsed_ns is None
    assert fsm._phase_elapsed_offset_ns == 0
    assert fsm._safe_hold_reason == ""


def test_wait_ready_rejects_configured_phase_timeout() -> None:
    with pytest.raises(ValueError, match="WAIT_READY.*不能配置阶段超时"):
        GlobalFSM(phase_timeouts_ns={GlobalPhase.WAIT_READY: 1})


@pytest.mark.parametrize(
    "policy",
    (
        {GlobalPhase.DONE: 1},
        {GlobalPhase.FAILED: 1},
        {GlobalPhase.WAIT_READY: 1},
        {GlobalPhase.SAFE_HOLD: 1},
        {GlobalPhase.SEARCH_TARGET: 0},
        {GlobalPhase.SEARCH_TARGET: True},
    ),
)
def test_fsm_rejects_invalid_timeout_policies(
    policy: dict[GlobalPhase, object],
) -> None:
    with pytest.raises(ValueError, match="超时"):
        GlobalFSM(phase_timeouts_ns=policy)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "policy",
    (
        [],
        [(GlobalPhase.SEARCH_TARGET, 10)],
        "SEARCH_TARGET=10",
        SimpleNamespace(SEARCH_TARGET=10),
    ),
)
def test_fsm_rejects_non_mapping_timeout_policy(policy: object) -> None:
    with pytest.raises(ValueError, match="Mapping"):
        GlobalFSM(phase_timeouts_ns=policy)  # type: ignore[arg-type]


def test_fsm_timeout_policy_is_copied_and_read_only() -> None:
    source = {GlobalPhase.SEARCH_TARGET: 10}
    fsm = GlobalFSM(phase_timeouts_ns=source)

    source[GlobalPhase.SEARCH_TARGET] = 1
    source[GlobalPhase.NAV_TO_PICK] = 20
    assert fsm.phase_timeouts_ns == {GlobalPhase.SEARCH_TARGET: 10}

    with pytest.raises(TypeError):
        fsm.phase_timeouts_ns[GlobalPhase.SEARCH_TARGET] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        fsm.phase_timeouts_ns = {}  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, 1.0, "1", -1])
def test_fsm_rejects_invalid_internal_retry_limits(value: object) -> None:
    with pytest.raises(ValueError, match="max_pick_retries"):
        GlobalFSM(max_pick_retries=value)  # type: ignore[arg-type]


def test_fsm_constructor_keeps_existing_calls_compatible() -> None:
    assert GlobalFSM(max_pick_retries=2).max_pick_retries == 2
    assert GlobalFSM(2, {GlobalPhase.SEARCH_TARGET: 10}).phase_timeouts_ns == {
        GlobalPhase.SEARCH_TARGET: 10
    }
    with pytest.raises(TypeError):
        GlobalFSM(1, None, 1)  # type: ignore[misc]


def test_fsm_requires_public_contract_types() -> None:
    fsm = GlobalFSM()
    with pytest.raises(TypeError, match="FSMEvent"):
        fsm.handle_event("SYSTEM_READY", 1)  # type: ignore[arg-type]
    assert fsm.phase is GlobalPhase.WAIT_READY

    assert fsm.handle_event(FSMEvent.SYSTEM_READY, 2)
    with pytest.raises(TypeError, match="TaskSpec"):
        fsm.submit_task(SimpleNamespace(valid=True), 3)  # type: ignore[arg-type]
    assert fsm.phase is GlobalPhase.LOAD_TASK


@pytest.mark.parametrize(
    "reason",
    (None, True, 1, 1.0, b"failure", ["失败"], SimpleNamespace(text="失败")),
)
def test_handle_event_rejects_non_string_reason(reason: object) -> None:
    fsm = GlobalFSM()

    with pytest.raises(TypeError, match="reason.*str"):
        fsm.handle_event(
            FSMEvent.SYSTEM_READY,
            1,
            reason,  # type: ignore[arg-type]
        )

    assert fsm.phase is GlobalPhase.WAIT_READY
    assert fsm.phase_entered_ns is None


def test_handle_event_empty_reason_keeps_default_failure_reason() -> None:
    fsm = GlobalFSM()

    assert fsm.handle_event(FSMEvent.FAILURE, 1, "")

    assert fsm.phase is GlobalPhase.FAILED
    assert fsm.failure_reason == "业务模块报告失败"


# ————————————————————————————————
# 【Codex修改-38：所有FSM读取入口严格校验时间类型】
# 防止bool、负数、浮点数或字符串通过status或phase_elapsed_ns形成含糊时间语义。
# ————————————————————————————————
@pytest.mark.parametrize("timestamp_ns", [True, -1, 1.0, "1"])
def test_parser_and_status_reject_invalid_timestamps(timestamp_ns: object) -> None:
    raw = json.dumps(
        {
            "task": 1,
            "target_body": "box_pink",
            "place_type": "shelf_point",
            "place_world": [-2.68, 0.778, 1.156],
            "place_radius": 0.24,
        }
    )
    with pytest.raises(ValueError, match="timestamp_ns"):
        InstructionParser().parse(raw, timestamp_ns)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timestamp_ns"):
        GlobalFSM().status(timestamp_ns)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timestamp_ns"):
        GlobalFSM().phase_elapsed_ns(timestamp_ns)  # type: ignore[arg-type]


# ————————————————————————————————
# 【Codex修改-39：状态与活动时间读取拒绝倒序时间】
# 防止旧时间生成倒序快照或负活动时间，并验证异常与相同时间读取都没有状态副作用。
# ————————————————————————————————
def test_fsm_time_reads_reject_stale_and_allow_equal_timestamp() -> None:
    fsm = _loaded_fsm()
    assert fsm.handle_event(FSMEvent.TARGET_FOUND, 200)
    before = (
        fsm.phase,
        fsm.phase_entered_ns,
        fsm._phase_elapsed_offset_ns,
        fsm._last_transition_ns,
        fsm.failure_reason,
    )

    with pytest.raises(ValueError, match="最近合法状态转换时间"):
        fsm.status(199)
    with pytest.raises(ValueError, match="最近合法状态转换时间"):
        fsm.phase_elapsed_ns(199)

    assert fsm.status(200).global_phase is GlobalPhase.NAV_TO_PICK
    assert fsm.phase_elapsed_ns(200) == 0
    assert (
        fsm.phase,
        fsm.phase_entered_ns,
        fsm._phase_elapsed_offset_ns,
        fsm._last_transition_ns,
        fsm.failure_reason,
    ) == before


def test_fsm_accepts_equal_timestamps_without_skipping_event_order() -> None:
    raw = json.dumps(_official_task_data())
    task = InstructionParser().parse(raw, 100)[0]
    fsm = GlobalFSM()

    assert fsm.handle_event(FSMEvent.SYSTEM_READY, 100)
    assert fsm.submit_task(task, 100)
    assert fsm.handle_event(FSMEvent.TARGET_FOUND, 100)
    assert fsm.phase is GlobalPhase.NAV_TO_PICK
    assert fsm.phase_entered_ns == 100

    assert not fsm.handle_event(FSMEvent.PICK_EXECUTED, 100)
    assert fsm.phase is GlobalPhase.NAV_TO_PICK
    assert fsm.phase_entered_ns == 100
    assert fsm.handle_event(FSMEvent.PICK_NAV_REACHED, 100)
    assert fsm.phase is GlobalPhase.REFINE_TARGET
    assert fsm.phase_entered_ns == 100


def test_fsm_repeated_task_submission_does_not_reset_active_task() -> None:
    raw = json.dumps(
        [
            {
                "task": 1,
                "instruction": "把粉色箱体放到货架点",
                "target_kind": "cuboid_box",
                "target_body": "box_pink",
                "target_color": "pink",
                "place_type": "shelf_point",
                "place_world": [-2.68, 0.778, 1.156],
                "place_radius": 0.24,
            },
            {
                "task": 2,
                "instruction": "把棕色箱体放到桌面点",
                "target_kind": "cuboid_box",
                "target_body": "box_brown",
                "target_color": "brown",
                "place_type": "table_point",
                "place_world": [-1.0, 2.2, 0.834],
                "place_radius": 0.28,
            },
            {
                "task": 3,
                "instruction": "把黄色箱体放到包装箱左侧",
                "target_kind": "cuboid_box",
                "target_body": "box_yellow",
                "target_color": "yellow",
                "ref_prop": "packaging_box",
                "ref_prop_body": "packaging_box",
                "direction": "left",
                "place_type": "shelf_prop_side",
                "place_world": [-2.68, 0.54, 0.498],
                "place_radius": 0.24,
            },
        ]
    )
    parser = InstructionParser()
    first_broadcast = parser.parse(raw, 100)
    repeated_broadcast = parser.parse(raw, 500_000_100)
    assert len(first_broadcast) == len(repeated_broadcast) == 3
    assert (
        repeated_broadcast[0].timestamp_ns - first_broadcast[0].timestamp_ns
        == 500_000_000
    )

    fsm = GlobalFSM()
    assert fsm.handle_event(FSMEvent.SYSTEM_READY, 110)
    assert fsm.submit_task(first_broadcast[0], 120)
    assert fsm.handle_event(FSMEvent.TARGET_FOUND, 130)
    phase_entered_ns = fsm.phase_entered_ns

    # 这里只验证FSM层：执行中再次提交任务不会重载，不覆盖ROS回调或去重逻辑。
    assert not fsm.submit_task(repeated_broadcast[0], 500_000_100)
    assert fsm.phase is GlobalPhase.NAV_TO_PICK
    assert fsm.phase_entered_ns == phase_entered_ns
    assert fsm.task is first_broadcast[0]


def test_instruction_parser_rejects_negative_place_radius() -> None:
    raw = json.dumps(
        _official_task_data(
            task=1,
            place_type="shelf_point",
            place_world=[-2.68, 0.778, 1.156],
            place_radius=-0.01,
        )
    )
    with pytest.raises(ValueError, match="place_radius"):
        InstructionParser().parse(raw, 100)
# ————————————————————————————————


# InstructionParser补充校验
@pytest.mark.parametrize(
    "invalid_fields",
    (
        {"task_id": 1.5},
        {"task_id": True},
        {"place_world": [1.0, True, 0.8]},
        {"place_radius": True},
    ),
)
def test_instruction_parser_rejects_non_strict_numeric_values(
    invalid_fields: dict[str, object],
) -> None:
    task_data = _official_task_data()
    task_data.update(invalid_fields)
    with pytest.raises(ValueError):
        InstructionParser().parse(json.dumps(task_data), 100)


def test_instruction_parser_keeps_valid_integer_and_coordinates() -> None:
    raw = json.dumps(
        [
            {
                "task_id": 7,
                "instruction": "搬运粉色箱体",
                "target_kind": "cuboid_box",
                "target_body": "box",
                "target_color": "pink",
                "place_type": "table_point",
                "place_world": [1, 2.0, 0.8],
                "place_radius": 0.1,
            },
            {
                "task_id": "8",
                "instruction": "搬运棕色箱体",
                "target_kind": "cuboid_box",
                "target_body": "box",
                "target_color": "brown",
                "place_type": "table_point",
                "place_world": {"x": 0.5, "y": -0.2, "z": 0.7},
                "place_radius": 0.1,
            },
        ]
    )
    tasks = InstructionParser().parse(raw, 100)
    assert tuple(task.task_id for task in tasks) == (7, 8)
    assert tasks[0].place_world_xyz == (1.0, 2.0, 0.8)
    assert tasks[1].place_world_xyz == (0.5, -0.2, 0.7)


def test_official_instruction_array_preserves_task_and_new_fields() -> None:
    raw = json.dumps(
        [
            {
                "task": 1,
                "instruction": "把黄色箱体放到参考架旁边",
                "target_kind": "material",
                "target_body": "box",
                "target_color": "yellow",
                "place_type": "shelf_prop_side",
                "place_world": [0.5, -0.2, 0.7],
                "place_radius": 0.15,
                "ref_prop": "shelf",
                "ref_prop_body": "middle_board",
                "direction": "left",
            }
        ],
        ensure_ascii=False,
    )
    task = InstructionParser().parse(raw, 200)[0]
    assert task.task_id == 1
    assert task.task_id != 0
    assert task.target_kind == "material"
    assert task.ref_prop_body == "middle_board"


# ActionMux安全仲裁补充
def test_safe_hold_overrides_fresh_nonzero_commands() -> None:
    joints = _joints()
    status = FSMStatus(
        task_id=1,
        global_phase=GlobalPhase.SAFE_HOLD,
        local_phase=LocalPhase.APPROACH,
        retry_count=0,
        success=False,
        failure_reason="传感器保护",
        timestamp_ns=1_000,
    )
    manipulation = ManipulationCommand(
        joint_target=(0.0,) * 17,
        controlled_mask=(True,) * 17,
        local_phase=LocalPhase.APPROACH,
        timestamp_ns=1_000,
        valid_until_ns=9_000,
    )
    action = ActionMux().compose(
        BaseCommand(0.20, -0.30, 1_000, 9_000),
        manipulation,
        joints,
        status,
        2_000,
    )
    assert action.values[:2] == (0.0, 0.0)
    assert action.values[2:] == joints.position
    assert "SAFE_HOLD" in action.failure_reason
    assert "覆盖" in action.failure_reason
    assert "传感器保护" in action.failure_reason


# Recorder指令、裁判和基础记录
def test_instruction_raw_and_task_are_written_to_metadata(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    episode_dir = recorder.start("episode_instruction", 100, "测试边界")
    raw = json.dumps(
        [
            {
                "task": 3,
                "instruction": "记录任务",
                "target_kind": "material",
                "target_body": "box",
                "target_color": "brown",
                "place_type": "table_point",
                "place_world": [1.0, 0.0, 0.6],
                "place_radius": 0.1,
            }
        ],
        ensure_ascii=False,
    )
    task = recorder.record_instruction(raw, 110, InstructionParser())
    assert task is not None and task.task_id == 3
    assert recorder.metadata is not None
    assert recorder.metadata.instruction_raw == raw
    assert recorder.metadata.task == task
    assert recorder.metadata.instruction_parse_failure == ""
    metadata_path = recorder.finish(120)
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert persisted["instruction_raw"] == raw
    assert persisted["task"]["task_id"] == 3
    assert metadata_path.parent == episode_dir

    invalid_recorder = EpisodeRecorder(tmp_path / "invalid")
    invalid_recorder.start("episode_invalid", 200, "测试边界")
    assert invalid_recorder.record_instruction("不是JSON", 210, InstructionParser()) is None
    assert invalid_recorder.metadata is not None
    assert invalid_recorder.metadata.instruction_raw == "不是JSON"
    assert invalid_recorder.metadata.task is None
    assert "合法 JSON" in invalid_recorder.metadata.instruction_parse_failure


def test_plain_text_and_integer_referee_messages_are_preserved(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("episode_referee", 100, "测试边界")
    recorder.record_referee_message("/referee/taskinfo", "READY", 110)
    recorder.record_referee_message("/referee/gameinfo", '{"state":"RUNNING"}', 120)
    recorder.record_referee_message("/referee/score", 42, 130)

    assert recorder.metadata is not None
    plain, parsed, score = recorder.metadata.referee_messages
    assert plain["raw"] == "READY" and "parsed" not in plain
    assert parsed["raw"] == '{"state":"RUNNING"}'
    assert parsed["parsed"] == {"state": "RUNNING"}
    assert score["raw"] == 42 and isinstance(score["raw"], int)


# rosbag命令和状态（配置话题）
def test_rosbag_command_contains_all_configured_topics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    topics = tuple(config["recorder"]["rosbag_topics"])
    monkeypatch.setattr("team_sorting.recorder.shutil.which", lambda name: "/opt/ros2")

    recorder = EpisodeRecorder(tmp_path)
    recorder.start("episode_bag", 100, "测试边界")
    command = recorder.build_rosbag_command(topics)
    assert command[:4] == ("ros2", "bag", "record", "-o")
    assert set(command[5:]) == set(topics)
    assert Path(command[4]) == tmp_path / "episode_bag" / "rosbag"
    recorder.mark_rosbag_started(command[4])
    recorder.mark_rosbag_finished(0)
    assert recorder.metadata is not None
    assert recorder.metadata.rosbag_started is True
    assert recorder.metadata.rosbag_output == command[4]
    assert recorder.metadata.rosbag_exit_code == 0


# EpisodeRecorder路径与生命周期
def test_recorder_check_root_creates_and_probes_writable_directory(tmp_path: Path) -> None:
    root = tmp_path / "new_root"

    resolved = EpisodeRecorder(root).check_root()

    assert resolved == root.resolve()
    assert root.is_dir()
    assert not tuple(root.glob(".team_sorting_write_probe_*"))


def test_recorder_check_root_rejects_regular_file(tmp_path: Path) -> None:
    root = tmp_path / "not_a_directory"
    root.write_text("用户文件", encoding="utf-8")

    with pytest.raises(RuntimeError, match="无法创建数据目录|不是目录"):
        EpisodeRecorder(root).check_root()

    assert root.read_text(encoding="utf-8") == "用户文件"


def test_recorder_check_root_reports_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "unwritable"

    def _fail_probe(*args: object, **kwargs: object) -> tuple[int, str]:
        raise PermissionError("模拟目录不可写")

    monkeypatch.setattr(recorder_module.tempfile, "mkstemp", _fail_probe)

    with pytest.raises(RuntimeError, match="不可可靠写入"):
        EpisodeRecorder(root).check_root()

    assert root.is_dir()
    assert not tuple(root.glob(".team_sorting_write_probe_*"))


def test_recorder_start_write_failure_rolls_back_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    _fail_next_metadata_replace(monkeypatch)

    with pytest.raises(RuntimeError, match="原子写入"):
        recorder.start("retry_start", 10, "测试边界")

    assert recorder.metadata is None
    assert recorder.episode_dir is None
    assert not (tmp_path / "retry_start").exists()

    episode_dir = recorder.start("retry_start", 10, "测试边界")
    assert episode_dir == (tmp_path / "retry_start").resolve()
    assert recorder.metadata is not None
    assert (episode_dir / "metadata.json").is_file()


def test_recorder_start_failure_never_removes_preexisting_user_directory(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "user_episode"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("用户数据", encoding="utf-8")
    recorder = EpisodeRecorder(tmp_path)

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        recorder.start("user_episode", 10, "测试边界")

    assert sentinel.read_text(encoding="utf-8") == "用户数据"


def test_recorder_accepts_safe_path_components(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)

    episode_dir = recorder.start("episode.Safe_1-2", 0, "测试边界")
    metadata_path = recorder.finish(1)

    assert episode_dir.parent == tmp_path.resolve()
    assert metadata_path.parent == episode_dir
    assert EpisodeRecorder.make_episode_id("session.Safe_1-2").startswith(
        "session.Safe_1-2_"
    )


@pytest.mark.parametrize("episode_id", [".", "..", "nested/name", "nested\\name"])
def test_recorder_rejects_unsafe_episode_id(tmp_path: Path, episode_id: str) -> None:
    with pytest.raises(ValueError):
        EpisodeRecorder(tmp_path).start(episode_id, 0, "测试边界")


@pytest.mark.parametrize("prefix", [".", "..", "nested/name", "nested\\name"])
def test_recorder_rejects_unsafe_episode_prefix(prefix: str) -> None:
    with pytest.raises(ValueError):
        EpisodeRecorder.make_episode_id(prefix)


def test_recorder_rejects_duplicate_episode_directory(tmp_path: Path) -> None:
    first = EpisodeRecorder(tmp_path)
    first.start("same_episode", 10, "测试边界")
    first.finish(20)

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        EpisodeRecorder(tmp_path).start("same_episode", 30, "测试边界")


@pytest.mark.parametrize(
    "started_at_ns",
    [True, -1, 1.0, "1", float("nan"), float("inf")],
)
def test_recorder_rejects_invalid_start_timestamp(
    tmp_path: Path, started_at_ns: object
) -> None:
    with pytest.raises(ValueError, match="started_at_ns"):
        EpisodeRecorder(tmp_path).start(
            "episode", started_at_ns, "测试边界"  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("boundary_policy", ["", "   ", None, 1])
def test_recorder_requires_nonempty_boundary_policy(
    tmp_path: Path, boundary_policy: object
) -> None:
    with pytest.raises(ValueError, match="boundary_policy"):
        EpisodeRecorder(tmp_path).start(
            "episode", 0, boundary_policy  # type: ignore[arg-type]
        )


# Recorder指令、裁判和JSONL补充
def test_recorder_empty_instruction_result_preserves_raw_text(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("empty_tasks", 0, "测试边界")
    parser = SimpleNamespace(parse=lambda raw, timestamp_ns: [])

    result = recorder.record_instruction("[]", 10, parser)

    assert result is None
    assert recorder.metadata is not None
    assert recorder.metadata.instruction_raw == "[]"
    assert recorder.metadata.task is None
    assert recorder.metadata.parsed_tasks == ()
    assert "解析结果为空" in recorder.metadata.instruction_parse_failure


def test_recorder_instruction_write_failure_rolls_back_and_retries_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("instruction_retry", 0, "测试边界")
    raw = json.dumps(_official_task_data(task_id=7))
    _fail_next_metadata_replace(monkeypatch)

    with pytest.raises(RuntimeError, match="原子写入"):
        recorder.record_instruction(raw, 10, InstructionParser())

    assert recorder.metadata is not None
    assert recorder.metadata.instruction_raw is None
    assert recorder.metadata.task is None
    assert recorder.metadata.parsed_tasks == ()
    assert recorder.metadata.instruction_parse_failure == ""
    assert recorder.metadata.topic_counts == {}
    persisted = json.loads(
        (tmp_path / "instruction_retry" / "metadata.json").read_text(encoding="utf-8")
    )
    assert persisted["instruction_raw"] is None
    assert persisted["parsed_tasks"] == []
    assert persisted["topic_counts"] == {}

    task = recorder.record_instruction(raw, 10, InstructionParser())
    assert task is not None and task.task_id == 7
    assert recorder.metadata.topic_counts == {"/material/instruction": 1}


def test_recorder_preserves_all_parsed_tasks(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("multiple_tasks", 0, "测试边界")
    raw = json.dumps(
        [
            {
                "task_id": 1,
                "instruction": "搬运粉色物体",
                "target_kind": "material",
                "target_body": "box",
                "target_color": "pink",
                "place_type": "table_point",
                "place_world": [1.0, 2.0, 0.8],
                "place_radius": 0.1,
            },
            {
                "task_id": 2,
                "instruction": "搬运棕色物体",
                "target_kind": "material",
                "target_body": "box",
                "target_color": "brown",
                "place_type": "table_point",
                "place_world": [2.0, 1.0, 0.9],
                "place_radius": 0.1,
            },
        ],
        ensure_ascii=False,
    )

    returned = recorder.record_instruction(raw, 10, InstructionParser())
    metadata_path = recorder.finish(20)
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert returned is not None and returned.task_id == 1
    assert persisted["task"]["task_id"] == 1
    assert [task["task_id"] for task in persisted["parsed_tasks"]] == [1, 2]


def test_recorder_rejects_non_string_instruction_without_coercion(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("instruction_type", 0, "测试边界")

    with pytest.raises(ValueError, match="instruction_raw"):
        recorder.record_instruction(123, 10, InstructionParser())  # type: ignore[arg-type]

    assert recorder.metadata is not None
    assert recorder.metadata.instruction_raw is None


@pytest.mark.parametrize("timestamp_ns", [True, -1, 1.0, "10", float("nan"), float("inf")])
@pytest.mark.parametrize("operation", ["instruction", "referee", "final_result"])
def test_recorder_rejects_invalid_event_timestamps(
    tmp_path: Path, timestamp_ns: object, operation: str
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("timestamps", 0, "测试边界")

    with pytest.raises(ValueError, match="timestamp_ns"):
        if operation == "instruction":
            recorder.record_instruction(
                "{}", timestamp_ns, InstructionParser()  # type: ignore[arg-type]
            )
        elif operation == "referee":
            recorder.record_referee_message(
                "/referee/taskinfo", "READY", timestamp_ns  # type: ignore[arg-type]
            )
        else:
            recorder.set_final_result('{"success":true}', timestamp_ns)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "score",
    [True, 1.0, "1", float("nan"), float("inf")],
)
def test_recorder_score_requires_strict_integer(tmp_path: Path, score: object) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("score", 0, "测试边界")

    with pytest.raises(ValueError, match="score"):
        recorder.record_referee_message(
            "/referee/score", score, 10  # type: ignore[arg-type]
        )


def test_recorder_referee_write_failure_rolls_back_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("referee_retry", 0, "测试边界")
    _fail_next_metadata_replace(monkeypatch)

    with pytest.raises(RuntimeError, match="原子写入"):
        recorder.record_referee_message("/referee/taskinfo", "READY", 10)

    assert recorder.metadata is not None
    assert recorder.metadata.referee_messages == []
    assert recorder.metadata.topic_counts == {}
    persisted = json.loads(
        (tmp_path / "referee_retry" / "metadata.json").read_text(encoding="utf-8")
    )
    assert persisted["referee_messages"] == []
    assert persisted["topic_counts"] == {}

    recorder.record_referee_message("/referee/taskinfo", "READY", 10)
    assert len(recorder.metadata.referee_messages) == 1
    assert recorder.metadata.topic_counts == {"/referee/taskinfo": 1}


# rosbag命令和状态
@pytest.mark.parametrize("output_path", ["", "   "])
def test_recorder_rejects_empty_rosbag_started_path(
    tmp_path: Path, output_path: str
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("bag_path", 0, "测试边界")

    with pytest.raises(ValueError, match="输出路径"):
        recorder.mark_rosbag_started(output_path)


@pytest.mark.parametrize(
    "exit_code",
    [True, 0.0, "0", float("nan"), float("inf")],
)
def test_recorder_exit_code_requires_strict_integer(
    tmp_path: Path, exit_code: object
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("exit_code", 0, "测试边界")
    recorder.mark_rosbag_started(tmp_path / "exit_code" / "rosbag")

    with pytest.raises(ValueError, match="rosbag_exit_code"):
        recorder.mark_rosbag_finished(exit_code)  # type: ignore[arg-type]


def test_recorder_rejects_exit_code_before_start_and_duplicate_write(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("bag_lifecycle", 0, "测试边界")

    with pytest.raises(RuntimeError, match="尚未成功启动"):
        recorder.mark_rosbag_finished(0)

    recorder.mark_rosbag_started(tmp_path / "bag_lifecycle" / "rosbag")
    recorder.mark_rosbag_finished(0)
    with pytest.raises(RuntimeError, match="拒绝重复覆盖"):
        recorder.mark_rosbag_finished(1)


def test_recorder_rosbag_start_write_failure_rolls_back_and_rejects_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("bag_start_retry", 0, "测试边界")
    output = tmp_path / "bag_start_retry" / "rosbag"
    _fail_next_metadata_replace(monkeypatch)

    with pytest.raises(RuntimeError, match="原子写入"):
        recorder.mark_rosbag_started(output)

    assert recorder.metadata is not None
    assert recorder.metadata.rosbag_started is False
    assert recorder.metadata.rosbag_output is None
    assert recorder.metadata.rosbag_exit_code is None
    persisted = json.loads(
        (tmp_path / "bag_start_retry" / "metadata.json").read_text(encoding="utf-8")
    )
    assert persisted["rosbag_started"] is False
    assert persisted["rosbag_output"] is None
    assert persisted["rosbag_exit_code"] is None

    recorder.mark_rosbag_started(output)
    with pytest.raises(RuntimeError, match="已经启动过"):
        recorder.mark_rosbag_started(output)


def test_recorder_rosbag_finish_write_failure_can_retry_same_nonzero_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("bag_finish_retry", 0, "测试边界")
    output = tmp_path / "bag_finish_retry" / "rosbag"
    recorder.mark_rosbag_started(output)
    _fail_next_metadata_replace(monkeypatch)

    with pytest.raises(RuntimeError, match="原子写入"):
        recorder.mark_rosbag_finished(7)

    assert recorder.metadata is not None
    assert recorder.metadata.rosbag_started is True
    assert recorder.metadata.rosbag_output == str(output)
    assert recorder.metadata.rosbag_exit_code is None
    persisted = json.loads(
        (tmp_path / "bag_finish_retry" / "metadata.json").read_text(encoding="utf-8")
    )
    assert persisted["rosbag_started"] is True
    assert persisted["rosbag_exit_code"] is None

    recorder.mark_rosbag_finished(7)
    assert recorder.metadata.rosbag_exit_code == 7
    with pytest.raises(RuntimeError, match="拒绝重复覆盖"):
        recorder.mark_rosbag_finished(7)
    with pytest.raises(RuntimeError, match="已经启动过"):
        recorder.mark_rosbag_started(output)


# Recorder原子写入与失败回滚
@pytest.mark.parametrize("ended_at_ns", [True, 1.0, "10", float("nan"), float("inf")])
def test_recorder_finish_requires_strict_timestamp(
    tmp_path: Path, ended_at_ns: object
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("finish_type", 0, "测试边界")

    with pytest.raises(ValueError, match="ended_at_ns"):
        recorder.finish(ended_at_ns)  # type: ignore[arg-type]

    assert recorder.metadata is not None


def test_recorder_finish_rejects_time_before_start_and_stays_active(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("finish_order", 100, "测试边界")

    with pytest.raises(ValueError, match="不能早于"):
        recorder.finish(99)

    assert recorder.metadata is not None
    assert recorder.episode_dir is not None


# 以下用例会观察原子替换的私有写入细节，属于内部可靠性回归测试。
# 这些断言可随Recorder内部重构同步调整；新测试仍应优先通过公开方法验证行为。
def test_recorder_metadata_uses_same_directory_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    episode_dir = recorder.start("atomic", 0, "测试边界")
    replacements: list[tuple[Path, Path]] = []
    real_replace = recorder_module.os.replace

    def _record_replace(source: str | Path, target: str | Path) -> None:
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(recorder_module.os, "replace", _record_replace)
    recorder.record_referee_message("/referee/taskinfo", "READY", 10)

    source, target = replacements[-1]
    assert source.parent == target.parent == episode_dir
    assert target.name == "metadata.json"
    assert not source.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["referee_messages"]


def test_recorder_metadata_serialization_failure_preserves_old_file(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    episode_dir = recorder.start("serialization", 0, "测试边界")
    metadata_path = episode_dir / "metadata.json"
    original = metadata_path.read_bytes()
    assert recorder.metadata is not None
    recorder.metadata.final_result = {"not_json": object()}

    with pytest.raises(RuntimeError, match="原子写入"):
        recorder._write_metadata()

    assert metadata_path.read_bytes() == original
    assert not tuple(episode_dir.glob(".metadata_*.tmp"))
    assert recorder.metadata is not None


def test_recorder_finish_write_failure_keeps_episode_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    episode_dir = recorder.start("finish_atomic", 0, "测试边界")
    original = (episode_dir / "metadata.json").read_bytes()

    _fail_next_metadata_replace(monkeypatch)
    with pytest.raises(RuntimeError, match="原子写入"):
        recorder.finish(10)

    assert recorder.metadata is not None
    assert recorder.metadata.ended_at_ns is None
    assert recorder.episode_dir == episode_dir
    assert (episode_dir / "metadata.json").read_bytes() == original
    assert not tuple(episode_dir.glob(".metadata_*.tmp"))

    metadata_path = recorder.finish(10)
    assert recorder.metadata is None
    assert recorder.episode_dir is None
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["ended_at_ns"] == 10


def test_recorder_final_result_write_failure_preserves_old_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("result_retry", 0, "测试边界")
    recorder.set_final_result('{"success":false}', 10)
    assert recorder.metadata is not None
    old_result = recorder.metadata.final_result
    _fail_next_metadata_replace(monkeypatch)

    with pytest.raises(RuntimeError, match="原子写入"):
        recorder.set_final_result('{"success":true}', 20)

    assert recorder.metadata.final_result == old_result
    persisted = json.loads(
        (tmp_path / "result_retry" / "metadata.json").read_text(encoding="utf-8")
    )
    assert persisted["final_result"] == old_result

    recorder.set_final_result('{"success":true}', 20)
    assert recorder.metadata.final_result == {
        "timestamp_ns": 20,
        "payload": {"success": True},
    }


# EpisodeRecorder路径与生命周期补充
def test_recorder_episode_ids_do_not_repeat_at_same_utc_microsecond(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_datetime = recorder_module.datetime
    fixed = real_datetime(2026, 7, 23, 8, 9, 10, 123456, tzinfo=recorder_module.timezone.utc)

    class _FixedDateTime:
        @classmethod
        def now(cls, timezone_value: object) -> object:
            return fixed

    monkeypatch.setattr(recorder_module, "datetime", _FixedDateTime)
    first = EpisodeRecorder.make_episode_id("episode")
    second = EpisodeRecorder.make_episode_id("episode")

    assert first != second
    assert first.rsplit("_", 1)[0] == second.rsplit("_", 1)[0]
    assert "/" not in first and ":" not in first and " " not in first


# rosbag命令和状态补充
def test_recorder_started_bag_without_exit_code_remains_unconfirmed(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("unknown_exit", 0, "测试边界")
    recorder.mark_rosbag_started(tmp_path / "unknown_exit" / "rosbag")

    metadata_path = recorder.finish(10)
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert persisted["rosbag_started"] is True
    assert persisted["rosbag_exit_code"] is None


@pytest.mark.parametrize("topics", [[], "/camera", b"/camera"])
def test_recorder_rejects_empty_or_scalar_rosbag_topics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, topics: object
) -> None:
    monkeypatch.setattr(recorder_module.shutil, "which", lambda name: "/opt/ros2")
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("topics_shape", 0, "测试边界")

    with pytest.raises(ValueError, match="topics|至少"):
        recorder.build_rosbag_command(topics)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "topic",
    ["", "-a", "--all", "relative", "/has space", "/has\ttab", "/nul\x00topic"],
)
def test_recorder_rejects_unsafe_rosbag_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, topic: str
) -> None:
    monkeypatch.setattr(recorder_module.shutil, "which", lambda name: "/opt/ros2")
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("topic_values", 0, "测试边界")

    with pytest.raises(ValueError):
        recorder.build_rosbag_command((topic,))


def test_recorder_rejects_duplicate_rosbag_topics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recorder_module.shutil, "which", lambda name: "/opt/ros2")
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("duplicate_topics", 0, "测试边界")

    with pytest.raises(ValueError, match="重复"):
        recorder.build_rosbag_command(("/camera", "/camera"))


@pytest.mark.parametrize("output_name", [".", "..", "nested/name", "nested\\name"])
def test_recorder_rejects_unsafe_rosbag_output_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_name: str
) -> None:
    monkeypatch.setattr(recorder_module.shutil, "which", lambda name: "/opt/ros2")
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("bag_output", 0, "测试边界")

    with pytest.raises(ValueError):
        recorder.build_rosbag_command(("/camera",), output_name=output_name)


def test_recorder_rejects_existing_rosbag_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recorder_module.shutil, "which", lambda name: "/opt/ros2")
    recorder = EpisodeRecorder(tmp_path)
    episode_dir = recorder.start("existing_bag", 0, "测试边界")
    (episode_dir / "rosbag").mkdir()

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        recorder.build_rosbag_command(("/camera",))


def test_recorder_rosbag_command_preserves_topic_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recorder_module.shutil, "which", lambda name: "/opt/ros2")
    recorder = EpisodeRecorder(tmp_path)
    episode_dir = recorder.start("ordered_bag", 0, "测试边界")
    topics = ("/instruction", "/camera/rgb", "/joint_states")

    command = recorder.build_rosbag_command(topics, output_name="bag.Safe_1")

    assert command == (
        "ros2",
        "bag",
        "record",
        "-o",
        str(episode_dir / "bag.Safe_1"),
        *topics,
    )


# Recorder指令、裁判和JSONL补充
def test_recorder_writes_unique_final_action_and_fsm_jsonl(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    episode_dir = recorder.start("jsonl", 0, "测试边界")
    joints = _joints()
    status = _status()
    action = ActionMux().compose(None, None, joints, status, 1_500)

    recorder.record_final_action(action)
    recorder.record_fsm_status(status)
    metadata_path = recorder.finish(2_000)

    action_line = (episode_dir / "final_actions.jsonl").read_text(encoding="utf-8").splitlines()
    status_line = (episode_dir / "fsm_status.jsonl").read_text(encoding="utf-8").splitlines()
    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert action_line == [final_action_to_json(action)]
    assert json.loads(status_line[0])["global_phase"] == status.global_phase.value
    assert persisted["topic_counts"]["/team/final_action"] == 1
    assert persisted["topic_counts"]["/team/fsm_status"] == 1


def test_recorder_finish_allows_new_episode_with_new_id(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(tmp_path)
    recorder.start("first", 0, "节点生命周期")
    recorder.note_topic("/camera")
    first_metadata = recorder.finish(10)

    second_dir = recorder.start("second", 20, "节点生命周期")

    assert first_metadata.exists()
    assert second_dir.name == "second"
    assert recorder.metadata is not None
    assert recorder.metadata.topic_counts == {}


# ArmExecutionController基础框架
# 以下少量用例直接观察或布置执行器私有运行状态，用于保护装载、拒绝和reset的原子清理。
# 它们属于内部回归测试，内部状态重构时可以同步调整；新增测试优先使用公开接口。
def test_arm_execution_initial_state_is_idle_without_trajectory() -> None:
    controller = ArmExecutionController()

    assert controller.local_phase is LocalPhase.IDLE
    assert controller._trajectory is None
    assert controller._waypoint_index == 0
    assert controller._trajectory_started_ns is None


def test_arm_execution_hold_uses_actual_feedback_and_short_ttl() -> None:
    controller = ArmExecutionController()
    joints = _joints()

    command = controller.create_hold_command(joints, timestamp_ns=1_000, valid_for_ns=200)

    assert command.valid is True
    assert command.joint_target == joints.position
    assert command.joint_target != (0.0,) * 17
    assert command.controlled_mask == (True,) * 17
    assert command.timestamp_ns == 1_000
    assert command.valid_until_ns == 1_200


def test_arm_execution_invalid_actual_feedback_does_not_fake_hold() -> None:
    actual = _joints()
    invalid = RobotJointState(
        position=actual.position,
        velocity=actual.velocity,
        effort=actual.effort,
        timestamp_ns=actual.timestamp_ns,
        valid=False,
        failure_reason="JointState已过期",
    )

    command = ArmExecutionController().create_hold_command(invalid, 1_000, 200)

    assert command.valid is False
    assert command.joint_target == invalid.position
    assert command.controlled_mask == (False,) * 17
    assert "JointState已过期" in command.failure_reason


@pytest.mark.parametrize(
    "timestamp_ns",
    [True, -1, 1.0, "1000", float("nan"), float("inf")],
)
def test_arm_execution_hold_rejects_invalid_timestamp(timestamp_ns: object) -> None:
    with pytest.raises(ValueError, match="timestamp_ns"):
        ArmExecutionController().create_hold_command(_joints(), timestamp_ns, 200)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "valid_for_ns",
    [True, 0, -1, 1.0, "200", float("nan"), float("inf")],
)
def test_arm_execution_hold_rejects_invalid_ttl(valid_for_ns: object) -> None:
    with pytest.raises(ValueError, match="valid_for_ns"):
        ArmExecutionController().create_hold_command(_joints(), 1_000, valid_for_ns)  # type: ignore[arg-type]


def test_arm_execution_rejects_empty_or_explicitly_invalid_trajectory() -> None:
    controller = _execution_controller()

    empty = controller.start_trajectory(_execution_trajectory(waypoints=()))
    invalid = controller.start_trajectory(
        _execution_trajectory(valid=False, failure_reason="规划器报告无解")
    )

    assert empty.state == "REJECTED" and empty.success is False
    assert empty.local_phase is LocalPhase.FAILED
    assert invalid.state == "REJECTED" and invalid.success is False
    assert "规划器报告无解" in invalid.failure_reason


def test_public_trajectory_requires_independent_arm_motion_phase() -> None:
    waypoints = (
        _execution_waypoint(0.0, phase=ArmMotionPhase.PREGRASP),
        _execution_waypoint(1.0, phase=ArmMotionPhase.GRASP),
        _execution_waypoint(2.0, phase=ArmMotionPhase.LIFT),
        _execution_waypoint(3.0, phase=ArmMotionPhase.RETREAT),
    )
    trajectory = JointTrajectory(
        trajectory_id="pick-1",
        task_id=1,
        target_body="box",
        execution_phase=GlobalPhase.EXECUTE_PICK,
        waypoints=waypoints,
        timestamp_ns=2_000,
    )
    assert waypoints[0].phase is ArmMotionPhase.PREGRASP
    assert trajectory.execution_phase is GlobalPhase.EXECUTE_PICK
    with pytest.raises(ValueError, match="ArmMotionPhase"):
        JointWaypoint(
            LocalPhase.MOVE_PREGRASP, 0.0, _joints().position, (True,) * 17
        )  # type: ignore[arg-type]


def test_joint_waypoint_rejects_all_false_controlled_mask() -> None:
    with pytest.raises(ValueError, match="至少一项必须为 True"):
        JointWaypoint(
            ArmMotionPhase.PREGRASP, 0.0, _joints().position, (False,) * 17
        )


@pytest.mark.parametrize(("index", "value"), [(9, -0.01), (16, 1.01)])
def test_joint_waypoint_rejects_controlled_gripper_target_outside_official_range(
    index: int, value: float,
) -> None:
    target = list(_joints().position)
    target[index] = value
    with pytest.raises(ValueError, match="夹爪路点目标必须位于"):
        _execution_waypoint(joint_position=tuple(target))


def _waypoints_for_phases(
    phases: tuple[ArmMotionPhase, ...],
) -> tuple[JointWaypoint, ...]:
    return tuple(
        _execution_waypoint(float(index), phase=phase)
        for index, phase in enumerate(phases)
    )


def test_joint_trajectory_rejects_plain_string_execution_phase() -> None:
    with pytest.raises(ValueError, match="严格使用 GlobalPhase"):
        JointTrajectory(
            "pick", 1, "box", "EXECUTE_PICK", (), 2_000, False, "失败"
        )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("execution_phase", "phases", "unexpected"),
    [
        (
            GlobalPhase.EXECUTE_PICK,
            (
                ArmMotionPhase.PREGRASP,
                ArmMotionPhase.GRASP,
                ArmMotionPhase.LIFT,
                ArmMotionPhase.RELEASE,
            ),
            "RELEASE",
        ),
        (
            GlobalPhase.EXECUTE_PLACE,
            (
                ArmMotionPhase.PREPLACE,
                ArmMotionPhase.LOWER,
                ArmMotionPhase.GRASP,
                ArmMotionPhase.RELEASE,
                ArmMotionPhase.POST_RELEASE_RETREAT,
            ),
            "GRASP",
        ),
    ],
)
def test_joint_trajectory_rejects_cross_operation_phase(
    execution_phase: GlobalPhase,
    phases: tuple[ArmMotionPhase, ...],
    unexpected: str,
) -> None:
    with pytest.raises(ValueError, match=unexpected):
        JointTrajectory(
            "trajectory", 1, "box", execution_phase,
            _waypoints_for_phases(phases), 2_000,
        )


def test_joint_trajectory_rejects_phase_backtracking_and_missing_phase() -> None:
    with pytest.raises(ValueError, match="不得倒退"):
        JointTrajectory(
            "pick", 1, "box", GlobalPhase.EXECUTE_PICK,
            _waypoints_for_phases(
                (
                    ArmMotionPhase.PREGRASP,
                    ArmMotionPhase.GRASP,
                    ArmMotionPhase.PREGRASP,
                    ArmMotionPhase.LIFT,
                    ArmMotionPhase.RETREAT,
                )
            ),
            2_000,
        )
    with pytest.raises(ValueError, match="缺少必要阶段"):
        JointTrajectory(
            "pick", 1, "box", GlobalPhase.EXECUTE_PICK,
            _waypoints_for_phases(
                (ArmMotionPhase.PREGRASP, ArmMotionPhase.GRASP, ArmMotionPhase.LIFT)
            ),
            2_000,
        )


def test_joint_trajectory_allows_multiple_waypoints_per_phase() -> None:
    phases = tuple(
        phase
        for phase in (
            ArmMotionPhase.PREGRASP,
            ArmMotionPhase.GRASP,
            ArmMotionPhase.LIFT,
            ArmMotionPhase.RETREAT,
        )
        for _ in range(2)
    )
    trajectory = JointTrajectory(
        "pick", 1, "box", GlobalPhase.EXECUTE_PICK,
        _waypoints_for_phases(phases), 2_000,
    )
    assert len(trajectory.waypoints) == 8


def test_joint_trajectory_requires_one_controlled_mask_for_all_waypoints() -> None:
    default_mask = _execution_waypoint().controlled_mask
    changed_mask = (False,) + default_mask[1:]
    with pytest.raises(ValueError, match="完全一致"):
        JointTrajectory(
            "pick-mask-change", 1, "box", GlobalPhase.EXECUTE_PICK,
            (
                _execution_waypoint(0.0, default_mask, ArmMotionPhase.PREGRASP),
                _execution_waypoint(1.0, changed_mask, ArmMotionPhase.GRASP),
                _execution_waypoint(2.0, default_mask, ArmMotionPhase.LIFT),
                _execution_waypoint(3.0, default_mask, ArmMotionPhase.RETREAT),
            ),
            2_000,
        )


def test_arm_execution_rejects_mask_changed_after_trajectory_construction() -> None:
    trajectory = _execution_trajectory()
    changed_mask = (False,) + trajectory.waypoints[1].controlled_mask[1:]
    object.__setattr__(trajectory.waypoints[1], "controlled_mask", changed_mask)
    status = _execution_controller().start_trajectory(trajectory)
    assert status.state == "REJECTED"
    assert "完全一致" in status.failure_reason


def test_pick_and_place_trajectories_with_uniform_masks_remain_valid() -> None:
    pick = JointTrajectory(
        "uniform-pick", 1, "box", GlobalPhase.EXECUTE_PICK,
        _waypoints_for_phases((
            ArmMotionPhase.PREGRASP, ArmMotionPhase.GRASP,
            ArmMotionPhase.LIFT, ArmMotionPhase.RETREAT,
        )),
        2_000,
    )
    place = JointTrajectory(
        "uniform-place", 1, "box", GlobalPhase.EXECUTE_PLACE,
        _waypoints_for_phases((
            ArmMotionPhase.PREPLACE, ArmMotionPhase.LOWER,
            ArmMotionPhase.RELEASE, ArmMotionPhase.POST_RELEASE_RETREAT,
        )),
        2_000,
    )
    assert _execution_controller().start_trajectory(pick).state == "LOADED"
    assert _execution_controller().start_trajectory(place).state == "LOADED"


def test_arm_execution_loads_complete_pick_and_place_trajectories() -> None:
    pick = _execution_trajectory()
    place = _execution_trajectory(
        execution_phase=GlobalPhase.EXECUTE_PLACE,
        waypoints=_waypoints_for_phases(
            (
                ArmMotionPhase.PREPLACE,
                ArmMotionPhase.LOWER,
                ArmMotionPhase.RELEASE,
                ArmMotionPhase.POST_RELEASE_RETREAT,
            )
        ),
    )
    assert _execution_controller().start_trajectory(pick).state == "LOADED"
    assert _execution_controller().start_trajectory(place).state == "LOADED"


def test_arm_execution_rejects_damaged_phase_contract() -> None:
    string_phase = _execution_trajectory(execution_phase="EXECUTE_PICK")
    mixed = _execution_trajectory(
        waypoints=_waypoints_for_phases(
            (
                ArmMotionPhase.PREGRASP,
                ArmMotionPhase.GRASP,
                ArmMotionPhase.LIFT,
                ArmMotionPhase.RELEASE,
            )
        )
    )
    backtracking = _execution_trajectory(
        waypoints=_waypoints_for_phases(
            (
                ArmMotionPhase.PREGRASP,
                ArmMotionPhase.GRASP,
                ArmMotionPhase.PREGRASP,
                ArmMotionPhase.LIFT,
                ArmMotionPhase.RETREAT,
            )
        )
    )
    missing = _execution_trajectory(
        waypoints=_waypoints_for_phases(
            (ArmMotionPhase.PREGRASP, ArmMotionPhase.GRASP, ArmMotionPhase.LIFT)
        )
    )
    assert "GlobalPhase" in _execution_controller().start_trajectory(string_phase).failure_reason
    assert "RELEASE" in _execution_controller().start_trajectory(mixed).failure_reason
    assert "不得倒退" in _execution_controller().start_trajectory(backtracking).failure_reason
    assert "缺少必要阶段" in _execution_controller().start_trajectory(missing).failure_reason


def test_arm_execution_rejects_damaged_all_false_waypoint() -> None:
    phases = (
        ArmMotionPhase.PREGRASP,
        ArmMotionPhase.GRASP,
        ArmMotionPhase.LIFT,
        ArmMotionPhase.RETREAT,
    )
    waypoints = tuple(
        _damaged_execution_waypoint(
            phase=phase,
            time_from_start_s=float(index),
            joint_position=_joints().position,
            controlled_mask=(False,) * 17,
        )
        for index, phase in enumerate(phases)
    )
    status = _execution_controller().start_trajectory(
        _execution_trajectory(waypoints=waypoints)
    )
    assert status.state == "REJECTED"
    assert "至少一项必须为True" in status.failure_reason


def test_arm_execution_rejects_duck_typed_waypoint_with_legal_fields() -> None:
    real_waypoints = _execution_trajectory().waypoints
    duck_waypoints = tuple(
        SimpleNamespace(
            phase=waypoint.phase,
            time_from_start_s=waypoint.time_from_start_s,
            joint_position=waypoint.joint_position,
            controlled_mask=waypoint.controlled_mask,
        )
        for waypoint in real_waypoints
    )
    status = _execution_controller().start_trajectory(
        _execution_trajectory(waypoints=duck_waypoints)
    )
    assert status.state == "REJECTED"
    assert "JointWaypoint实例" in status.failure_reason


def test_arm_execution_accepts_real_waypoints_and_rejects_damaged_real_waypoint() -> None:
    controller = _execution_controller()
    assert controller.start_trajectory(_execution_trajectory()).state == "LOADED"

    damaged = _execution_waypoint()
    object.__setattr__(damaged, "joint_position", (0.0,) * 16 + (float("nan"),))
    status = controller.start_trajectory(
        _execution_trajectory(waypoints=(damaged,) + _execution_trajectory().waypoints[1:])
    )
    assert status.state == "REJECTED"
    assert "joint_position" in status.failure_reason


@pytest.mark.parametrize("bad_valid", ["yes", 0])
def test_arm_execution_rejects_non_bool_valid_without_exception(
    bad_valid: object,
) -> None:
    damaged = object.__new__(JointTrajectory)
    object.__setattr__(damaged, "valid", bad_valid)
    object.__setattr__(damaged, "timestamp_ns", 9_000)
    status = _execution_controller().start_trajectory(damaged)
    assert status.state == "REJECTED"
    assert "valid必须是严格bool" in status.failure_reason
    assert status.timestamp_ns == 9_000


def test_arm_execution_uses_zero_timestamp_for_damaged_timestamp() -> None:
    damaged = object.__new__(JointTrajectory)
    object.__setattr__(damaged, "valid", True)
    object.__setattr__(damaged, "timestamp_ns", "100")
    status = _execution_controller().start_trajectory(damaged)
    assert status.state == "REJECTED"
    assert "timestamp_ns" in status.failure_reason
    assert status.timestamp_ns == 0


def test_arm_execution_rejects_missing_trajectory_attributes_without_exception() -> None:
    damaged = object.__new__(JointTrajectory)
    object.__setattr__(damaged, "valid", True)
    object.__setattr__(damaged, "timestamp_ns", 10)
    status = _execution_controller().start_trajectory(damaged)
    assert status.state == "REJECTED"
    assert "task_id" in status.failure_reason
    assert status.timestamp_ns == 10


def test_arm_execution_rejects_non_trajectory_instance_without_exception() -> None:
    status = _execution_controller().start_trajectory(SimpleNamespace())  # type: ignore[arg-type]
    assert status.state == "REJECTED"
    assert "JointTrajectory实例" in status.failure_reason
    assert status.timestamp_ns == 0


def test_invalid_joint_trajectory_does_not_require_fake_target_identity() -> None:
    invalid = JointTrajectory(
        "", 1, "", GlobalPhase.EXECUTE_PICK, (), 2_000, False, "任务无效"
    )
    assert not invalid.valid and invalid.target_body == ""
    with pytest.raises(ValueError, match="waypoints 必须为空"):
        JointTrajectory(
            "", 1, "", GlobalPhase.EXECUTE_PICK,
            (_execution_waypoint(),), 2_000, False, "规划失败",
        )
    with pytest.raises(ValueError, match="target_body"):
        JointTrajectory(
            "pick", 1, "", GlobalPhase.EXECUTE_PICK,
            _waypoints_for_phases(
                (
                    ArmMotionPhase.PREGRASP,
                    ArmMotionPhase.GRASP,
                    ArmMotionPhase.LIFT,
                    ArmMotionPhase.RETREAT,
                )
            ),
            2_000,
        )

    damaged = object.__new__(JointTrajectory)
    object.__setattr__(damaged, "valid", False)
    object.__setattr__(damaged, "failure_reason", "上游任务无效")
    object.__setattr__(damaged, "timestamp_ns", 2_000)
    object.__setattr__(damaged, "task_id", 1)
    object.__setattr__(damaged, "execution_phase", GlobalPhase.EXECUTE_PICK)
    status = _execution_controller().start_trajectory(damaged)
    assert status.state == "REJECTED"
    assert status.failure_reason == "上游任务无效"


def test_arm_execution_rejects_empty_trajectory_id() -> None:
    status = _execution_controller().start_trajectory(_execution_trajectory("  "))

    assert status.state == "REJECTED"
    assert "trajectory_id" in status.failure_reason


@pytest.mark.parametrize("time_s", [-0.1, True, float("nan"), float("inf")])
def test_arm_execution_rejects_invalid_waypoint_time(time_s: object) -> None:
    waypoint = _damaged_execution_waypoint(
        phase=ArmMotionPhase.PREGRASP,
        time_from_start_s=time_s,
        joint_position=_joints().position,
    )

    status = _execution_controller().start_trajectory(
        _execution_trajectory(waypoints=(waypoint,))
    )

    assert status.state == "REJECTED"
    assert "time_from_start_s" in status.failure_reason


@pytest.mark.parametrize(
    "times",
    [(0.0, 0.0), (1.0, 0.5)],
)
def test_arm_execution_requires_strictly_increasing_waypoint_times(
    times: tuple[float, float],
) -> None:
    trajectory = _execution_trajectory(
        waypoints=tuple(_execution_waypoint(time_s) for time_s in times)
    )

    status = _execution_controller().start_trajectory(trajectory)

    assert status.state == "REJECTED"
    assert "严格递增" in status.failure_reason


@pytest.mark.parametrize(
    "joint_position",
    [
        (0.0,) * 16,
        (0.0,) * 16 + (True,),
        (0.0,) * 16 + ("bad",),
        (0.0,) * 16 + (float("nan"),),
        (0.0,) * 16 + (float("inf"),),
    ],
)
def test_arm_execution_rechecks_waypoint_joint_positions(
    joint_position: tuple[object, ...],
) -> None:
    waypoint = _damaged_execution_waypoint(
        phase=ArmMotionPhase.PREGRASP,
        time_from_start_s=0.0,
        joint_position=joint_position,
    )

    status = _execution_controller().start_trajectory(
        _execution_trajectory(waypoints=(waypoint,))
    )

    assert status.state == "REJECTED"
    assert "joint_position" in status.failure_reason


@pytest.mark.parametrize(
    "mask",
    [(True,) * 16, (True,) * 16 + (1,), (True,) * 16 + ("yes",)],
)
def test_arm_execution_requires_exact_boolean_controlled_mask(
    mask: tuple[object, ...],
) -> None:
    waypoint = _damaged_execution_waypoint(
        phase=ArmMotionPhase.PREGRASP,
        time_from_start_s=0.0,
        joint_position=_joints().position,
        controlled_mask=mask,
    )

    status = _execution_controller().start_trajectory(
        _execution_trajectory(waypoints=(waypoint,))
    )

    assert status.state == "REJECTED"
    assert "controlled_mask" in status.failure_reason


def test_arm_execution_valid_load_is_neutral_and_not_started() -> None:
    controller = _execution_controller()
    trajectory = _execution_trajectory()

    status = controller.start_trajectory(trajectory)

    assert status.state == "LOADED"
    assert status.success is False
    assert status.local_phase is controller.local_phase is LocalPhase.IDLE
    assert controller._trajectory is trajectory
    assert controller._trajectory_started_ns is None
    assert controller._waypoint_index == 0


def test_arm_execution_rejection_atomically_clears_old_runtime_state() -> None:
    controller = _execution_controller()
    assert controller.start_trajectory(_execution_trajectory()).state == "LOADED"
    controller._waypoint_index = 3
    controller._trajectory_started_ns = 9_000
    controller._stable_cycle_count = 4
    controller._cached_verification = object()

    status = controller.start_trajectory(_execution_trajectory(waypoints=()))

    assert status.state == "REJECTED"
    assert status.local_phase is controller.local_phase is LocalPhase.FAILED
    assert controller._trajectory is None
    assert controller._waypoint_index == 0
    assert controller._trajectory_started_ns is None
    assert controller._stable_cycle_count == 0
    assert controller._cached_verification is None


def test_arm_execution_valid_load_clears_previous_runtime_state() -> None:
    controller = _execution_controller()
    controller.local_phase = LocalPhase.RETREAT
    controller._waypoint_index = 2
    controller._trajectory_started_ns = 5_000
    controller._stable_cycle_count = 3
    controller._cached_verification = object()
    trajectory = _execution_trajectory("replacement")

    status = controller.start_trajectory(trajectory)

    assert status.state == "LOADED"
    assert status.local_phase is controller.local_phase is LocalPhase.IDLE
    assert controller._trajectory is trajectory
    assert controller._waypoint_index == 0
    assert controller._trajectory_started_ns is None
    assert controller._stable_cycle_count == 0
    assert controller._cached_verification is None


def test_arm_execution_reset_only_clears_memory_state() -> None:
    controller = _execution_controller()
    controller.start_trajectory(_execution_trajectory())
    controller._waypoint_index = 1
    controller._trajectory_started_ns = 3_000

    result = controller.reset()

    assert result is None
    assert controller.local_phase is LocalPhase.IDLE
    assert controller._trajectory is None
    assert controller._waypoint_index == 0
    assert controller._trajectory_started_ns is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feedback_max_age_ns", True),
        ("trajectory_max_age_ns", 0),
        ("command_ttl_ns", 1.5),
        ("settle_cycles", False),
        ("max_slide_velocity_m_s", float("nan")),
        ("max_arm_velocity_rad_s", 0.0),
        ("max_gripper_velocity_per_s", 0.0),
        ("gripper_tolerance", 1.01),
        ("initial_gripper_error_limit", 1.01),
    ],
)
def test_arm_execution_config_rejects_missing_or_invalid_values(
    field: str, value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        _execution_config(**{field: value})


def test_arm_execution_config_has_no_hidden_defaults() -> None:
    with pytest.raises(TypeError):
        ArmExecutionConfig()  # type: ignore[call-arg]
    status = ArmExecutionController().start_trajectory(_execution_trajectory())
    assert status.state == "REJECTED"
    assert "Config未注入" in status.failure_reason


@pytest.mark.parametrize(
    "value", [0.0, -1.0, float("nan"), float("inf")],
)
def test_arm_execution_gripper_velocity_rejects_only_nonpositive_or_nonfinite(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="max_gripper_velocity_per_s"):
        _execution_config(max_gripper_velocity_per_s=value)


def test_arm_execution_gripper_velocity_above_one_is_allowed() -> None:
    config = _execution_config(max_gripper_velocity_per_s=2.0)
    assert config.max_gripper_velocity_per_s == 2.0


def test_arm_execution_yaml_is_explicitly_unconfigured_and_disabled() -> None:
    data = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    expected_fields = set(ArmExecutionConfig.__dataclass_fields__)
    assert set(data["arm_execution"]) == expected_fields
    assert all(data["arm_execution"][field] is None for field in expected_fields)


def test_arm_execution_no_trajectory_returns_invalid_feedback_hold() -> None:
    controller = _execution_controller()
    joints = _actual_joints(timestamp_ns=2_000)
    command, status = controller.step(joints, 2_000)
    assert command.valid is False
    assert command.controlled_mask == (False,) * 17
    assert command.joint_target == joints.position
    assert status.state == "NO_TRAJECTORY"


def test_arm_execution_idle_long_gap_does_not_apply_active_control_period() -> None:
    controller = _execution_controller(max_control_period_ns=10)
    first_command, first = controller.step(
        _actual_joints(timestamp_ns=2_000), 2_000
    )
    second_command, second = controller.step(
        _actual_joints(timestamp_ns=1_000_002_000), 1_000_002_000
    )
    assert first.state == second.state == "NO_TRAJECTORY"
    assert first_command.valid is second_command.valid is False
    assert second_command.controlled_mask == (False,) * 17


def test_arm_execution_idle_history_does_not_pollute_new_trajectory_period() -> None:
    controller = _execution_controller(max_control_period_ns=10)
    controller.step(_actual_joints(timestamp_ns=100_000_000), 100_000_000)
    trajectory = _execution_trajectory()
    assert controller.start_trajectory(trajectory).state == "LOADED"
    _, status = controller.step(
        _actual_joints(timestamp_ns=100_000_001), 100_000_001
    )
    assert status.state == "RUNNING"


def test_arm_execution_first_step_uses_actual_start_and_configured_ttl() -> None:
    start = _joints().position
    goal = list(start)
    goal[3] += 0.1
    trajectory = _execution_trajectory(
        waypoints=(
            _execution_waypoint(1.0, joint_position=tuple(goal)),
            _execution_waypoint(2.0, phase=ArmMotionPhase.GRASP, joint_position=tuple(goal)),
            _execution_waypoint(3.0, phase=ArmMotionPhase.LIFT, joint_position=tuple(goal)),
            _execution_waypoint(4.0, phase=ArmMotionPhase.RETREAT, joint_position=tuple(goal)),
        )
    )
    controller = _execution_controller(command_ttl_ns=123_456_789)
    assert controller.start_trajectory(trajectory).state == "LOADED"
    command, status = controller.step(_actual_joints(start, 2_000), 2_000)
    assert command.joint_target == start
    assert command.valid_until_ns == 123_458_789
    assert status.state == "RUNNING"
    assert controller._trajectory_started_ns == 2_000


def test_arm_execution_nonzero_first_waypoint_interpolates_over_its_own_time() -> None:
    start = _joints().position
    goal = list(start)
    goal[3] += 0.1
    waypoints = (
        _execution_waypoint(1.0, joint_position=tuple(goal)),
        _execution_waypoint(2.0, phase=ArmMotionPhase.GRASP, joint_position=tuple(goal)),
        _execution_waypoint(3.0, phase=ArmMotionPhase.LIFT, joint_position=tuple(goal)),
        _execution_waypoint(4.0, phase=ArmMotionPhase.RETREAT, joint_position=tuple(goal)),
    )
    controller = _execution_controller(max_control_period_ns=1_000_000_000)
    controller.start_trajectory(_execution_trajectory(waypoints=waypoints))
    controller.step(_actual_joints(start, 2_000), 2_000)
    command, _ = controller.step(_actual_joints(start, 500_002_000), 500_002_000)
    assert command.joint_target[3] == pytest.approx(start[3] + 0.05)


def test_arm_execution_zero_time_far_first_waypoint_is_rejected_without_jump() -> None:
    goal = list(_joints().position)
    goal[3] += 0.5
    waypoints = (
        _execution_waypoint(0.0, joint_position=tuple(goal)),
        _execution_waypoint(1.0, phase=ArmMotionPhase.GRASP, joint_position=tuple(goal)),
        _execution_waypoint(2.0, phase=ArmMotionPhase.LIFT, joint_position=tuple(goal)),
        _execution_waypoint(3.0, phase=ArmMotionPhase.RETREAT, joint_position=tuple(goal)),
    )
    controller = _execution_controller(initial_arm_error_limit_rad=0.1)
    controller.start_trajectory(_execution_trajectory(waypoints=waypoints))
    command, status = controller.step(_actual_joints(timestamp_ns=2_000), 2_000)
    assert command.valid is False and command.joint_target == _joints().position
    assert status.state == "FAILED"
    assert "initial_arm_error_limit_rad" in status.failure_reason


def test_arm_execution_mask_false_uses_latest_actual_and_head_is_never_controlled() -> None:
    mask = (True, False, False) + (False,) * 14
    start = _joints().position
    goal = list(start)
    goal[0] += 0.02
    waypoints = (
        _execution_waypoint(1.0, mask, joint_position=tuple(goal)),
        _execution_waypoint(2.0, mask, ArmMotionPhase.GRASP, tuple(goal)),
        _execution_waypoint(3.0, mask, ArmMotionPhase.LIFT, tuple(goal)),
        _execution_waypoint(4.0, mask, ArmMotionPhase.RETREAT, tuple(goal)),
    )
    controller = _execution_controller(max_control_period_ns=1_000_000_000)
    controller.start_trajectory(_execution_trajectory(waypoints=waypoints))
    controller.step(_actual_joints(start, 2_000), 2_000)
    changed = list(start)
    changed[1] = 0.2
    changed[3] = 0.3
    command, _ = controller.step(_actual_joints(tuple(changed), 500_002_000), 500_002_000)
    assert command.controlled_mask[1:3] == (False, False)
    assert command.joint_target[1] == 0.2
    assert command.joint_target[3] == 0.3


def test_arm_execution_applies_separate_slide_arm_and_gripper_velocity_limits() -> None:
    start = _joints().position
    goal = list(start)
    goal[0] += 0.1
    goal[3] += 0.1
    goal[9] += 0.1
    waypoints = (
        _execution_waypoint(1.0, joint_position=tuple(goal)),
        _execution_waypoint(2.0, phase=ArmMotionPhase.GRASP, joint_position=tuple(goal)),
        _execution_waypoint(3.0, phase=ArmMotionPhase.LIFT, joint_position=tuple(goal)),
        _execution_waypoint(4.0, phase=ArmMotionPhase.RETREAT, joint_position=tuple(goal)),
    )
    controller = _execution_controller(
        max_control_period_ns=1_000_000_000,
        max_slide_velocity_m_s=0.01,
        max_arm_velocity_rad_s=0.02,
        max_gripper_velocity_per_s=0.03,
    )
    controller.start_trajectory(_execution_trajectory(waypoints=waypoints))
    controller.step(_actual_joints(start, 2_000), 2_000)
    command, _ = controller.step(_actual_joints(start, 500_002_000), 500_002_000)
    assert command.joint_target[0] == pytest.approx(start[0] + 0.005)
    assert command.joint_target[3] == pytest.approx(start[3] + 0.01)
    assert command.joint_target[9] == pytest.approx(start[9] + 0.015)


def test_arm_execution_valid_candidates_keep_grippers_inside_official_range() -> None:
    start = _joints().position
    goal = list(start)
    goal[9] = 0.0
    goal[16] = 1.0
    waypoints = tuple(
        _execution_waypoint(float(index + 1), phase=phase, joint_position=tuple(goal))
        for index, phase in enumerate((
            ArmMotionPhase.PREGRASP, ArmMotionPhase.GRASP,
            ArmMotionPhase.LIFT, ArmMotionPhase.RETREAT,
        ))
    )
    controller = _execution_controller(
        max_control_period_ns=1_000_000_000,
        max_gripper_velocity_per_s=2.0,
    )
    controller.start_trajectory(_execution_trajectory(waypoints=waypoints))
    controller.step(_actual_joints(start, 2_000), 2_000)
    command, _ = controller.step(
        _actual_joints(start, 500_002_000), 500_002_000
    )
    assert command.valid is True
    assert 0.0 <= command.joint_target[9] <= 1.0
    assert 0.0 <= command.joint_target[16] <= 1.0


@pytest.mark.parametrize("bad_time", [True, -1, 1.0, "2000"])
def test_arm_execution_step_rejects_bad_call_time_without_exception(bad_time: object) -> None:
    command, status = _execution_controller().step(_joints(), bad_time)  # type: ignore[arg-type]
    assert command.valid is False and status.state == "FAILED"


def test_arm_execution_rejects_future_stale_invalid_and_damaged_feedback() -> None:
    cases: tuple[object, ...] = (
        _actual_joints(timestamp_ns=3_000),
        _actual_joints(timestamp_ns=0),
        _actual_joints(timestamp_ns=2_000, valid=False, failure_reason="bad"),
        SimpleNamespace(valid=True, position=(0.0,) * 17, timestamp_ns="2"),
    )
    for actual in cases:
        controller = _execution_controller(feedback_max_age_ns=100)
        _, status = controller.step(actual, 2_000)  # type: ignore[arg-type]
        assert status.state == "FAILED"


def test_arm_execution_rejects_damaged_joint_name_identity_before_execution() -> None:
    reversed_names = _actual_joints(timestamp_ns=2_000)
    object.__setattr__(reversed_names, "joint_names", tuple(reversed(JOINT_NAMES)))
    controller = _execution_controller()
    controller.start_trajectory(_execution_trajectory())
    command, status = controller.step(reversed_names, 2_000)
    assert command.valid is False
    assert status.state == "FAILED"
    assert "JOINT_NAMES" in status.failure_reason

    missing_names = _actual_joints(timestamp_ns=2_000)
    object.__delattr__(missing_names, "joint_names")
    controller = _execution_controller()
    controller.start_trajectory(_execution_trajectory())
    _, missing_status = controller.step(missing_names, 2_000)
    assert missing_status.state == "FAILED"
    assert "JOINT_NAMES" in missing_status.failure_reason


def test_arm_execution_accepts_exact_joint_names_and_gripper_boundaries() -> None:
    position = list(_joints().position)
    position[9] = 0.0
    position[16] = 1.0
    controller = _execution_controller()
    command, status = controller.step(
        _actual_joints(tuple(position), timestamp_ns=2_000), 2_000
    )
    assert status.state == "NO_TRAJECTORY"
    assert command.valid is False
    assert tuple(JOINT_NAMES) == _actual_joints().joint_names


@pytest.mark.parametrize(("index", "value"), [(9, -0.01), (16, 1.01)])
def test_arm_execution_rejects_actual_gripper_outside_official_range(
    index: int, value: float,
) -> None:
    position = list(_joints().position)
    position[index] = value
    controller = _execution_controller(
        max_gripper_velocity_per_s=100.0,
        initial_gripper_error_limit=1.0,
    )
    controller.start_trajectory(_execution_trajectory())
    command, status = controller.step(
        _actual_joints(tuple(position), timestamp_ns=2_000), 2_000
    )
    assert command.valid is False
    assert status.state == "FAILED"
    assert "夹爪反馈必须位于[0,1]" in status.failure_reason


def test_arm_execution_hold_rejects_joint_identity_and_gripper_range() -> None:
    bad_names = _actual_joints()
    object.__setattr__(bad_names, "joint_names", tuple(reversed(JOINT_NAMES)))
    bad_name_command = _execution_controller().create_hold_command(bad_names, 1_000, 100)
    assert bad_name_command.valid is False
    assert bad_name_command.controlled_mask == (False,) * 17

    position = list(_joints().position)
    position[9] = -2.0
    bad_gripper_command = _execution_controller().create_hold_command(
        _actual_joints(tuple(position)), 1_000, 100
    )
    assert bad_gripper_command.valid is False
    assert bad_gripper_command.controlled_mask == (False,) * 17


def test_arm_execution_rejects_damaged_waypoint_gripper_target_at_entry() -> None:
    trajectory = _execution_trajectory()
    damaged_position = list(trajectory.waypoints[1].joint_position)
    damaged_position[16] = 3.0
    object.__setattr__(trajectory.waypoints[1], "joint_position", tuple(damaged_position))
    status = _execution_controller().start_trajectory(trajectory)
    assert status.state == "REJECTED"
    assert "右夹爪目标必须位于[0,1]" in status.failure_reason


def test_arm_execution_rejects_trajectory_damaged_after_loading_without_exception() -> None:
    controller = _execution_controller()
    trajectory = _execution_trajectory()
    controller.start_trajectory(trajectory)
    object.__setattr__(trajectory, "waypoints", None)
    command, status = controller.step(_actual_joints(timestamp_ns=2_000), 2_000)
    assert command.valid is False
    assert status.state == "FAILED"
    assert "运行期校验失败" in status.failure_reason


def test_arm_execution_rejects_time_reversal_and_excessive_control_period() -> None:
    idle = _execution_controller(max_control_period_ns=200_000_000)
    idle.step(_actual_joints(timestamp_ns=2_000), 2_000)
    _, rollback = idle.step(_actual_joints(timestamp_ns=1_999), 1_999)
    assert rollback.state == "FAILED"
    assert "倒退" in rollback.failure_reason

    active = _execution_controller(max_control_period_ns=200_000_000)
    active.start_trajectory(_execution_trajectory())
    active.step(_actual_joints(timestamp_ns=2_000), 2_000)
    _, timeout = active.step(
        _actual_joints(timestamp_ns=202_001_000), 202_001_000
    )
    assert timeout.state == "FAILED"
    assert "控制周期" in timeout.failure_reason


def test_arm_execution_checks_trajectory_freshness_only_before_start() -> None:
    stale = _execution_controller(trajectory_max_age_ns=10)
    stale.start_trajectory(_execution_trajectory())
    _, stale_status = stale.step(_actual_joints(timestamp_ns=2_011), 2_011)
    assert stale_status.state == "FAILED" and "过期" in stale_status.failure_reason

    running = _execution_controller(
        trajectory_max_age_ns=10,
        feedback_max_age_ns=10_000_000_000,
        max_control_period_ns=10_000_000_000,
    )
    running.start_trajectory(_execution_trajectory())
    running.step(_actual_joints(timestamp_ns=2_005), 2_005)
    _, status = running.step(_actual_joints(timestamp_ns=3_000_002_005), 3_000_002_005)
    assert "轨迹在首次执行前已过期" not in status.failure_reason


def test_arm_execution_does_not_advance_on_time_without_feedback_arrival() -> None:
    start = _joints().position
    goal = list(start)
    goal[3] += 0.1
    waypoints = (
        _execution_waypoint(0.5, joint_position=tuple(goal)),
        _execution_waypoint(1.0, phase=ArmMotionPhase.GRASP, joint_position=tuple(goal)),
        _execution_waypoint(1.5, phase=ArmMotionPhase.LIFT, joint_position=tuple(goal)),
        _execution_waypoint(2.0, phase=ArmMotionPhase.RETREAT, joint_position=tuple(goal)),
    )
    controller = _execution_controller(max_control_period_ns=1_000_000_000)
    controller.start_trajectory(_execution_trajectory(waypoints=waypoints))
    controller.step(_actual_joints(start, 2_000), 2_000)
    controller.step(_actual_joints(start, 600_002_000), 600_002_000)
    assert controller._waypoint_index == 0


def test_arm_execution_requires_consecutive_settle_cycles_and_group_tolerances() -> None:
    controller = _execution_controller(settle_cycles=2)
    controller.start_trajectory(_execution_trajectory())
    joints = _actual_joints(timestamp_ns=2_000)
    controller.step(joints, 2_000)
    assert controller._waypoint_index == 0
    controller.step(_actual_joints(timestamp_ns=3_000), 3_000)
    assert controller._waypoint_index == 1


def test_arm_execution_duplicate_feedback_cannot_fake_settle_cycles() -> None:
    controller = _execution_controller(settle_cycles=2)
    controller.start_trajectory(_execution_trajectory())
    controller.step(_actual_joints(timestamp_ns=2_000), 2_000)
    controller.step(_actual_joints(timestamp_ns=2_000), 3_000)
    assert controller._stable_cycle_count == 1
    assert controller._waypoint_index == 0

    controller.step(_actual_joints(timestamp_ns=3_000), 4_000)
    assert controller._waypoint_index == 1


def test_arm_execution_feedback_timestamp_rollback_fails_closed() -> None:
    controller = _execution_controller()
    controller.start_trajectory(_execution_trajectory())
    controller.step(_actual_joints(timestamp_ns=2_000), 2_000)
    command, status = controller.step(_actual_joints(timestamp_ns=1_999), 3_000)
    assert command.valid is False
    assert status.state == "FAILED"
    assert status.failure_reason == "actual_joints反馈时间倒退"


def test_arm_execution_reset_allows_lower_feedback_timestamp_in_new_episode() -> None:
    controller = _execution_controller()
    controller.start_trajectory(_execution_trajectory())
    controller.step(_actual_joints(timestamp_ns=5_000), 5_000)
    controller.reset()
    command, status = controller.step(_actual_joints(timestamp_ns=1_000), 1_000)
    assert status.state == "NO_TRAJECTORY"
    assert command.joint_target == _joints().position
    assert controller._last_feedback_timestamp_ns == 1_000


def test_arm_execution_new_trajectory_does_not_reuse_feedback_identity() -> None:
    controller = _execution_controller()
    controller.start_trajectory(_execution_trajectory())
    controller.step(_actual_joints(timestamp_ns=5_000), 5_000)
    replacement = _execution_trajectory("replacement")
    assert controller.start_trajectory(replacement).state == "LOADED"
    _, status = controller.step(_actual_joints(timestamp_ns=2_500), 2_500)
    assert status.state == "RUNNING"
    assert controller._last_feedback_timestamp_ns == 2_500


def test_arm_execution_waypoint_and_total_timeouts_fail_closed() -> None:
    waypoint = _execution_controller(
        max_control_period_ns=2_000_000_000,
        waypoint_timeout_margin_ns=100_000_000,
    )
    waypoint.start_trajectory(_execution_trajectory())
    waypoint.step(_actual_joints(timestamp_ns=2_000), 2_000)
    _, status = waypoint.step(_actual_joints(timestamp_ns=1_100_002_001), 1_100_002_001)
    assert status.state == "FAILED" and "waypoint" in status.failure_reason

    total = _execution_controller(
        max_control_period_ns=5_000_000_000,
        waypoint_timeout_margin_ns=10_000_000_000,
        total_timeout_margin_ns=100_000_000,
    )
    total.start_trajectory(_execution_trajectory())
    total.step(_actual_joints(timestamp_ns=2_000), 2_000)
    _, status = total.step(_actual_joints(timestamp_ns=3_100_002_001), 3_100_002_001)
    assert status.state == "FAILED" and "total" in status.failure_reason


def test_arm_execution_pick_stops_at_verify_with_valid_hold_candidate() -> None:
    controller = _execution_controller(settle_cycles=1, max_control_period_ns=2_000_000_000)
    trajectory = _execution_trajectory()
    controller.start_trajectory(trajectory)
    expected = (
        LocalPhase.HUG_OPEN,
        LocalPhase.HUG_CLOSE,
        LocalPhase.VERIFY,
        LocalPhase.VERIFY,
    )
    for index, phase in enumerate(expected):
        now = index * 1_000_000_000 + 2_000
        command, status = controller.step(_actual_joints(timestamp_ns=now), now)
        assert status.local_phase is phase
    lift_index = 2
    lift = trajectory.waypoints[lift_index]
    assert controller._waypoint_index == lift_index
    assert command.valid is True
    assert command.valid_until_ns == now + controller._config.command_ttl_ns
    assert command.controlled_mask == lift.controlled_mask
    assert command.joint_target[9] == lift.joint_position[9]
    assert command.joint_target[16] == lift.joint_position[16]
    assert status.state == "VERIFICATION_PENDING"
    assert status.success is False
    assert controller._cached_verification is None


def test_arm_execution_multiple_pregrasp_waypoints_delay_hug_open_until_last() -> None:
    phases = (
        ArmMotionPhase.PREGRASP, ArmMotionPhase.PREGRASP,
        ArmMotionPhase.GRASP, ArmMotionPhase.LIFT, ArmMotionPhase.RETREAT,
    )
    controller = _execution_controller(settle_cycles=1, max_control_period_ns=2_000_000_000)
    controller.start_trajectory(
        _execution_trajectory(waypoints=_waypoints_for_phases(phases))
    )
    _, first = controller.step(_actual_joints(timestamp_ns=2_000), 2_000)
    assert first.local_phase is LocalPhase.MOVE_PREGRASP
    assert controller._waypoint_index == 1
    _, second = controller.step(
        _actual_joints(timestamp_ns=1_000_002_000), 1_000_002_000
    )
    assert second.local_phase is LocalPhase.HUG_OPEN
    assert controller._waypoint_index == 2


def test_arm_execution_multiple_lift_waypoints_verify_only_after_last() -> None:
    phases = (
        ArmMotionPhase.PREGRASP, ArmMotionPhase.GRASP,
        ArmMotionPhase.LIFT, ArmMotionPhase.LIFT, ArmMotionPhase.RETREAT,
    )
    controller = _execution_controller(settle_cycles=1, max_control_period_ns=2_000_000_000)
    controller.start_trajectory(
        _execution_trajectory(waypoints=_waypoints_for_phases(phases))
    )
    statuses = []
    for index in range(4):
        now = index * 1_000_000_000 + 2_000
        _, status = controller.step(_actual_joints(timestamp_ns=now), now)
        statuses.append(status)
    assert statuses[2].local_phase is LocalPhase.TEST_LIFT
    assert statuses[2].state == "RUNNING"
    assert statuses[3].local_phase is LocalPhase.VERIFY
    assert statuses[3].state == "VERIFICATION_PENDING"
    assert controller._waypoint_index == 3


def test_arm_execution_multiple_retreat_phase_mapping_marks_only_last_as_transport() -> None:
    phases = (
        ArmMotionPhase.PREGRASP, ArmMotionPhase.GRASP, ArmMotionPhase.LIFT,
        ArmMotionPhase.RETREAT, ArmMotionPhase.RETREAT,
    )
    controller = _execution_controller()
    controller.start_trajectory(
        _execution_trajectory(waypoints=_waypoints_for_phases(phases))
    )
    controller._waypoint_index = 3
    assert controller._phase_for_current(True) is LocalPhase.RETREAT
    controller._waypoint_index = 4
    assert controller._phase_for_current(True) is LocalPhase.TRANSPORT_HOLD


def test_arm_execution_same_phase_waypoints_still_require_settle_cycles() -> None:
    phases = (
        ArmMotionPhase.PREGRASP, ArmMotionPhase.PREGRASP,
        ArmMotionPhase.GRASP, ArmMotionPhase.LIFT, ArmMotionPhase.RETREAT,
    )
    controller = _execution_controller(settle_cycles=2)
    controller.start_trajectory(
        _execution_trajectory(waypoints=_waypoints_for_phases(phases))
    )
    controller.step(_actual_joints(timestamp_ns=2_000), 2_000)
    assert controller._waypoint_index == 0
    _, reached = controller.step(_actual_joints(timestamp_ns=3_000), 3_000)
    assert reached.local_phase is LocalPhase.MOVE_PREGRASP
    assert controller._waypoint_index == 1


def test_arm_execution_verify_wait_without_completed_evidence_times_out() -> None:
    controller = _execution_controller(
        settle_cycles=1,
        max_control_period_ns=2_000_000_000,
        total_timeout_margin_ns=100_000_000,
    )
    controller.start_trajectory(_execution_trajectory())
    for index in range(3):
        now = index * 1_000_000_000 + 2_000
        controller.step(_actual_joints(timestamp_ns=now), now)
    assert controller.local_phase is LocalPhase.VERIFY
    assert controller._waypoint_index == 2

    command, status = controller.step(
        _actual_joints(timestamp_ns=3_100_002_001), 3_100_002_001
    )
    assert command.valid is False
    assert status.state == "FAILED"
    assert "verification_timeout_ns" in status.failure_reason
    assert controller._cached_verification is None


def test_arm_execution_two_grasp_waypoints_map_approach_then_hug_close() -> None:
    phases = (
        ArmMotionPhase.PREGRASP, ArmMotionPhase.GRASP, ArmMotionPhase.GRASP,
        ArmMotionPhase.LIFT, ArmMotionPhase.RETREAT,
    )
    controller = _execution_controller(settle_cycles=1, max_control_period_ns=2_000_000_000)
    controller.start_trajectory(_execution_trajectory(waypoints=_waypoints_for_phases(phases)))
    observed = []
    for index in range(3):
        now = index * 1_000_000_000 + 2_000
        _, status = controller.step(_actual_joints(timestamp_ns=now), now)
        observed.append(status.local_phase)
    assert observed[1:] == [LocalPhase.APPROACH, LocalPhase.HUG_CLOSE]


def test_arm_execution_place_phase_mapping_and_completion() -> None:
    phases = (
        ArmMotionPhase.PREPLACE, ArmMotionPhase.LOWER,
        ArmMotionPhase.RELEASE, ArmMotionPhase.POST_RELEASE_RETREAT,
    )
    controller = _execution_controller(settle_cycles=1, max_control_period_ns=2_000_000_000)
    controller.start_trajectory(_execution_trajectory(
        execution_phase=GlobalPhase.EXECUTE_PLACE,
        waypoints=_waypoints_for_phases(phases),
    ))
    observed = []
    for index in range(4):
        now = index * 1_000_000_000 + 2_000
        command, status = controller.step(_actual_joints(timestamp_ns=now), now)
        observed.append(status.local_phase)
    assert observed[:3] == [LocalPhase.MOVE_PREPLACE, LocalPhase.LOWER_OBJECT, LocalPhase.RELEASE]
    assert observed[-1] is LocalPhase.IDLE
    assert status.state == "MOTION_COMPLETED_PLACE_VERIFICATION_PENDING"
    assert status.success is True
    assert "物体位置与裁判语义仍待外部验证" in status.failure_reason
    assert command.valid is False


def test_arm_execution_completed_terminal_survives_long_gap_and_bad_feedback() -> None:
    phases = (
        ArmMotionPhase.PREPLACE, ArmMotionPhase.LOWER,
        ArmMotionPhase.RELEASE, ArmMotionPhase.POST_RELEASE_RETREAT,
    )
    controller = _execution_controller(settle_cycles=1, max_control_period_ns=2_000_000_000)
    controller.start_trajectory(_execution_trajectory(
        execution_phase=GlobalPhase.EXECUTE_PLACE,
        waypoints=_waypoints_for_phases(phases),
    ))
    for index in range(4):
        now = index * 1_000_000_000 + 2_000
        _, terminal = controller.step(_actual_joints(timestamp_ns=now), now)
    original_timestamp = terminal.timestamp_ns
    original_error = terminal.max_joint_error

    command, repeated = controller.step(
        SimpleNamespace(valid=True), 100_000_000_000  # type: ignore[arg-type]
    )
    assert command.valid is False and command.controlled_mask == (False,) * 17
    assert repeated is terminal
    assert repeated.timestamp_ns == original_timestamp
    assert repeated.max_joint_error == original_error
    assert repeated.state == "MOTION_COMPLETED_PLACE_VERIFICATION_PENDING"
    assert repeated.success is True

    invalid_time_command, same_terminal = controller.step(
        SimpleNamespace(), "bad"  # type: ignore[arg-type]
    )
    assert invalid_time_command.valid is False
    assert same_terminal is terminal

    rollback_command, rollback_terminal = controller.step(
        SimpleNamespace(), 50_000_000_000  # type: ignore[arg-type]
    )
    assert rollback_command.valid is False
    assert rollback_terminal is terminal


def test_arm_execution_failed_terminal_preserves_original_evidence() -> None:
    controller = _execution_controller()
    controller.start_trajectory(_execution_trajectory())
    _, terminal = controller.step(
        _actual_joints(timestamp_ns=2_000, valid=False, failure_reason="原始反馈故障"),
        2_000,
    )
    assert terminal.state == "FAILED"
    command, repeated = controller.step(SimpleNamespace(), 99_000_000_000)  # type: ignore[arg-type]
    assert command.valid is False
    assert repeated is terminal
    assert repeated.failure_reason == terminal.failure_reason
    assert repeated.timestamp_ns == 2_000


def test_arm_execution_reset_clears_new_runtime_fields() -> None:
    controller = _execution_controller()
    controller.start_trajectory(_execution_trajectory())
    controller.step(_actual_joints(timestamp_ns=2_000), 2_000)
    controller.reset()
    assert controller._last_command is None
    assert controller._last_step_ns is None
    assert controller._last_feedback_timestamp_ns is None
    assert controller._initial_position is None
    assert controller._terminal_status is None
