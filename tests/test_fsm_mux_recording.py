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
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import team_sorting.recorder as recorder_module
from team_sorting.action_mux import ActionMux, ActionMuxConfig
from team_sorting.arm_execution import ArmExecutionConfig, ArmExecutionController
from team_sorting.fsm import FSMEvent, GlobalFSM, InstructionParser
from team_sorting.interfaces import (
    ACTION_NAMES,
    BaseCommand,
    FSMStatus,
    GlobalPhase,
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


def _loaded_fsm(max_pick_retries: int = 1) -> GlobalFSM:
    raw = json.dumps(
        {
            "task_id": 1,
            "target_body": "box",
            "place_type": "world_point",
            "place_world": [1.0, 2.0, 0.8],
        }
    )
    task = InstructionParser().parse(raw, 100)[0]
    fsm = GlobalFSM(max_pick_retries=max_pick_retries)
    assert fsm.handle_event(FSMEvent.SYSTEM_READY, 110)
    assert fsm.submit_task(task, 120)
    return fsm


def _advance(fsm: GlobalFSM, *events: FSMEvent, start_ns: int = 1_000) -> None:
    for timestamp_ns, event in enumerate(events, start=start_ns):
        assert fsm.handle_event(event, timestamp_ns)


def _execution_waypoint(
    time_from_start_s: float = 0.0,
    controlled_mask: tuple[bool, ...] = (True,) * 17,
) -> JointWaypoint:
    return JointWaypoint(time_from_start_s, _joints().position, controlled_mask)


def _execution_trajectory(
    trajectory_id: str = "trajectory-1",
    waypoints: tuple[object, ...] | None = None,
    *,
    valid: bool = True,
    failure_reason: str = "",
) -> JointTrajectory:
    selected_waypoints = (_execution_waypoint(),) if waypoints is None else waypoints
    return JointTrajectory(
        trajectory_id=trajectory_id,
        waypoints=selected_waypoints,  # type: ignore[arg-type]
        timestamp_ns=2_000,
        valid=valid,
        failure_reason=failure_reason,
    )


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
    action = ActionMux().compose(
        BaseCommand(v, w, 1_000, 2_000),  # type: ignore[arg-type]
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
                    "target_body": "box",
                    "target_color": "pink",
                    "place_type": "world_point",
                    "place_world": [1.0, 2.0, 0.8],
                    "place_radius": 0.1,
                }
            ]
        },
        ensure_ascii=False,
    )
    task = InstructionParser().parse(raw, 100)[0]
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
    assert not done_fsm.handle_event(FSMEvent.FAILURE, 2_000, "迟到的失败事件")
    assert done_fsm.phase is GlobalPhase.DONE
    assert done_fsm.failure_reason == ""
    assert done_fsm.handle_event(FSMEvent.RESET, 2_001)
    assert done_fsm.phase is GlobalPhase.WAIT_READY

    failed_fsm = GlobalFSM()
    assert failed_fsm.handle_event(FSMEvent.FAILURE, 3_000, "启动失败")
    assert failed_fsm.phase is GlobalPhase.FAILED
    assert not failed_fsm.handle_event(FSMEvent.SYSTEM_READY, 3_001)
    assert not failed_fsm.handle_event(FSMEvent.FAILURE, 3_002, "迟到的第二个失败")
    assert failed_fsm.phase is GlobalPhase.FAILED
    assert failed_fsm.failure_reason == "启动失败"
    assert failed_fsm.handle_event(FSMEvent.RESET, 3_003)
    assert failed_fsm.phase is GlobalPhase.WAIT_READY


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
    assert fsm.status(3_000).failure_reason == ""


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
    task_data = {
        "task_id": 1,
        "target_body": "box",
        "place_type": "world_point",
        "place_world": [1.0, 2.0, 0.8],
        "place_radius": 0.1,
    }
    task_data.update(invalid_fields)
    with pytest.raises(ValueError):
        InstructionParser().parse(json.dumps(task_data), 100)


def test_instruction_parser_keeps_valid_integer_and_coordinates() -> None:
    raw = json.dumps(
        [
            {
                "task_id": 7,
                "target_body": "box",
                "place_type": "world_point",
                "place_world": [1, 2.0, 0.8],
                "place_radius": 0.1,
            },
            {
                "task_id": "8",
                "target_body": "box",
                "place_type": "world_point",
                "place_world": {"x": 0.5, "y": -0.2, "z": 0.7},
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
                "place_type": "relative",
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
                "place_type": "world",
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
    assert command[4].endswith("episode_bag/rosbag")
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
    raw = json.dumps(
        {
            "task_id": 7,
            "target_body": "box",
            "place_type": "world_point",
            "place_world": [1.0, 2.0, 0.8],
        }
    )
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
                "place_type": "world",
                "place_world": [1.0, 2.0, 0.8],
            },
            {
                "task_id": 2,
                "instruction": "搬运棕色物体",
                "target_kind": "material",
                "target_body": "box",
                "target_color": "brown",
                "place_type": "world",
                "place_world": [2.0, 1.0, 0.9],
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
    controller = ArmExecutionController()

    empty = controller.start_trajectory(_execution_trajectory(waypoints=()))
    invalid = controller.start_trajectory(
        _execution_trajectory(valid=False, failure_reason="规划器报告无解")
    )

    assert empty.state == "REJECTED" and empty.success is False
    assert empty.local_phase is LocalPhase.FAILED
    assert invalid.state == "REJECTED" and invalid.success is False
    assert "规划器报告无解" in invalid.failure_reason


def test_arm_execution_rejects_empty_trajectory_id() -> None:
    status = ArmExecutionController().start_trajectory(_execution_trajectory("  "))

    assert status.state == "REJECTED"
    assert "trajectory_id" in status.failure_reason


@pytest.mark.parametrize("time_s", [-0.1, True, float("nan"), float("inf")])
def test_arm_execution_rejects_invalid_waypoint_time(time_s: object) -> None:
    waypoint = SimpleNamespace(
        time_from_start_s=time_s,
        joint_position=_joints().position,
        controlled_mask=(True,) * 17,
    )

    status = ArmExecutionController().start_trajectory(
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

    status = ArmExecutionController().start_trajectory(trajectory)

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
    waypoint = SimpleNamespace(
        time_from_start_s=0.0,
        joint_position=joint_position,
        controlled_mask=(True,) * 17,
    )

    status = ArmExecutionController().start_trajectory(
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
    waypoint = SimpleNamespace(
        time_from_start_s=0.0,
        joint_position=_joints().position,
        controlled_mask=mask,
    )

    status = ArmExecutionController().start_trajectory(
        _execution_trajectory(waypoints=(waypoint,))
    )

    assert status.state == "REJECTED"
    assert "controlled_mask" in status.failure_reason


def test_arm_execution_valid_load_is_neutral_and_not_started() -> None:
    controller = ArmExecutionController()
    trajectory = _execution_trajectory(
        waypoints=(_execution_waypoint(0.0), _execution_waypoint(1.0))
    )

    status = controller.start_trajectory(trajectory)

    assert status.state == "LOADED"
    assert status.success is False
    assert status.local_phase is controller.local_phase is LocalPhase.IDLE
    assert controller._trajectory is trajectory
    assert controller._trajectory_started_ns is None
    assert controller._waypoint_index == 0


def test_arm_execution_rejection_atomically_clears_old_runtime_state() -> None:
    controller = ArmExecutionController()
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
    controller = ArmExecutionController()
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
    controller = ArmExecutionController()
    controller.start_trajectory(_execution_trajectory())
    controller._waypoint_index = 1
    controller._trajectory_started_ns = 3_000

    result = controller.reset()

    assert result is None
    assert controller.local_phase is LocalPhase.IDLE
    assert controller._trajectory is None
    assert controller._waypoint_index == 0
    assert controller._trajectory_started_ns is None


# ============================================================================
# ArmExecutionController.step() 完整测试（含所有组长要求的新增用例）
# ============================================================================


def _test_config(**overrides: object) -> ArmExecutionConfig:
    """构造带保守安全参数的测试配置，可通过 overrides 覆盖任意字段。"""
    kwargs = dict(
        joint_tolerance_17=(0.02,) * 17,
        max_joint_velocity_17=(10.0,) * 17,
        settle_cycles=2,
        total_timeout_ns=30_000_000_000,
        command_ttl_ns=100_000_000,
        feedback_max_age_ns=None,
        trajectory_max_age_ns=None,
    )
    kwargs.update(overrides)
    return ArmExecutionConfig(**kwargs)


def _step_joints(**overrides: object) -> RobotJointState:
    """与 _joints() 不同的实际反馈。"""
    pos = (0.20, 0.10, -0.30, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
           0.50, -0.10, -0.20, -0.30, -0.40, -0.50, -0.60, 0.55)
    kwargs = dict(position=pos, velocity=(0.0,)*17, effort=(0.0,)*17, timestamp_ns=10_000)
    kwargs.update(overrides)
    return RobotJointState(**kwargs)


def _step_waypoint(
    time_from_start_s: float = 0.0,
    position: tuple[float, ...] | None = None,
    controlled_mask: tuple[bool, ...] = (True,) * 17,
) -> SimpleNamespace:
    return SimpleNamespace(
        time_from_start_s=time_from_start_s,
        joint_position=position if position is not None else _joints().position,
        controlled_mask=controlled_mask,
    )


def _step_trajectory(
    waypoints: tuple[object, ...],
    trajectory_id: str = "test-trajectory",
    timestamp_ns: int = 10_000,
) -> JointTrajectory:
    return JointTrajectory(
        trajectory_id=trajectory_id,
        waypoints=waypoints,
        timestamp_ns=timestamp_ns,
        valid=True,
    )


# ============================================================================
# A. 输入和状态（6项）
# ============================================================================


def test_step_no_loaded_trajectory_returns_idle() -> None:
    controller = ArmExecutionController()
    cmd, status = controller.step(_joints(), 2_000)
    assert cmd.valid is False
    assert status.state == "IDLE"
    assert status.success is False
    assert controller.local_phase is LocalPhase.IDLE


def test_step_invalid_actual_joints_enters_failed() -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_execution_trajectory())
    invalid = RobotJointState(
        position=_joints().position, velocity=(0.0,)*17, effort=(0.0,)*17,
        timestamp_ns=10_000, valid=False, failure_reason="JointState过期",
    )
    cmd, status = controller.step(invalid, 2_000)
    assert cmd.valid is False
    assert "JointState过期" in cmd.failure_reason
    assert controller.local_phase is LocalPhase.FAILED


@pytest.mark.parametrize("timestamp_ns", [True, -1, 1.0, "2000", float("nan"), float("inf")])
def test_step_rejects_invalid_timestamp(timestamp_ns: object) -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_execution_trajectory())
    with pytest.raises(ValueError, match="timestamp_ns"):
        controller.step(_joints(), timestamp_ns)


def test_step_rejects_timestamp_not_strictly_increasing() -> None:
    """时间单调性：timestamp_ns 必须严格递增，等于也算非法。"""
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(2.0))))
    c1, _ = controller.step(_joints(), 2_000)
    assert c1.valid is True
    # 相等 → 非法倒退
    c2, s2 = controller.step(_joints(), 2_000)
    assert c2.valid is False
    assert "非严格递增" in s2.failure_reason
    assert controller.local_phase is LocalPhase.FAILED


def test_step_failed_state_blocks_subsequent_calls() -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_execution_trajectory())
    controller.local_phase = LocalPhase.FAILED
    cmd, status = controller.step(_joints(), 2_000)
    assert cmd.valid is False
    assert "FAILED" in status.failure_reason


def test_step_after_reset_without_new_trajectory_returns_idle() -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_execution_trajectory())
    controller.reset()
    cmd, status = controller.step(_joints(), 2_000)
    assert status.state == "IDLE"
    assert "尚未装载有效轨迹" in status.failure_reason


# ============================================================================
# B. 时间与插值（8项）
# ============================================================================


def test_step_at_first_waypoint_emits_that_position() -> None:
    wp = _step_waypoint(0.0)
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((wp, _step_waypoint(2.0))))
    cmd, _ = controller.step(_joints(), 2_000)
    assert cmd.valid is True
    assert cmd.joint_target == wp.joint_position


def test_step_between_waypoints_interpolates_linearly() -> None:
    base = _joints().position
    p0, p2 = base, tuple(v + 0.02 for v in base)
    controller = ArmExecutionController(_test_config(max_joint_velocity_17=None))
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, p0), _step_waypoint(2.0, p2))))
    controller.step(_joints(), 0)
    cmd, _ = controller.step(_joints(), 1_000_000_000)
    for i in range(17):
        assert cmd.joint_target[i] == pytest.approx(p0[i] + 0.5 * (p2[i] - p0[i]))


def test_step_exactly_on_middle_waypoint() -> None:
    base = _joints().position
    p0, p1, p2 = base, tuple(v + 0.01 for v in base), tuple(v + 0.02 for v in base)
    controller = ArmExecutionController(_test_config(max_joint_velocity_17=None))
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0,p0), _step_waypoint(1.0,p1), _step_waypoint(2.0,p2))))
    controller.step(_joints(), 0)
    cmd, _ = controller.step(_joints(), 1_000_000_000)
    assert cmd.joint_target == p1


def test_step_trajectory_completes_with_config() -> None:
    """配置了容差和稳定数 → 到位后 success=True + COMPLETED。"""
    target = _joints().position
    actual = RobotJointState(position=target, velocity=(0.0,)*17, effort=(0.0,)*17, timestamp_ns=10_000)
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, target), _step_waypoint(0.1, target))))
    controller.step(actual, 0)
    controller.step(actual, 200_000_000)  # dt=0.2s > 最后路点0.1s, settle=1
    cmd, status = controller.step(actual, 300_000_000)  # settle=2
    assert status.success is True
    assert status.state == "COMPLETED"


def test_step_beyond_last_waypoint_holds_last_target() -> None:
    base = _joints().position
    p_last = tuple(v + 0.02 for v in base)
    controller = ArmExecutionController(_test_config(max_joint_velocity_17=None))
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, base), _step_waypoint(1.0, p_last))))
    controller.step(_joints(), 0)
    cmd, _ = controller.step(_joints(), 7_000_000_000)
    assert cmd.joint_target == p_last


def test_step_multi_segment_selects_correct_interval() -> None:
    base = _joints().position
    p0, p1, p2, p3 = base, tuple(v+0.004 for v in base), tuple(v+0.008 for v in base), tuple(v+0.012 for v in base)
    controller = ArmExecutionController(_test_config(max_joint_velocity_17=None))
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0,p0),_step_waypoint(1.0,p1),_step_waypoint(2.0,p2),_step_waypoint(3.0,p3))))
    controller.step(_joints(), 0)
    cmd, _ = controller.step(_joints(), 2_500_000_000)
    for i in range(17):
        assert cmd.joint_target[i] == pytest.approx(p2[i] + 0.5 * (p3[i] - p2[i]))


def test_step_result_is_exactly_17_dimensions() -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(1.0))))
    cmd, _ = controller.step(_joints(), 2_000)
    assert len(cmd.joint_target) == 17
    assert len(cmd.controlled_mask) == 17


def test_step_no_nan_or_inf_in_target() -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(1.0))))
    cmd, _ = controller.step(_joints(), 2_000)
    assert all(math.isfinite(v) for v in cmd.joint_target)


# ============================================================================
# C. controlled_mask（5项） + mask 切换拒绝
# ============================================================================


def test_step_all_true_mask_controls_all_joints() -> None:
    target_pos = tuple(float(i) / 10.0 for i in range(17))
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, target_pos, (True,)*17),)))
    cmd, _ = controller.step(_joints(), 2_000)
    assert cmd.controlled_mask == (True,) * 17
    assert cmd.joint_target == target_pos


def test_step_all_false_mask_keeps_all_actual_positions() -> None:
    actual = _joints()
    wp = _step_waypoint(0.0, tuple(9.9 for _ in range(17)), (False,) * 17)
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((wp,)))
    cmd, _ = controller.step(actual, 2_000)
    assert cmd.joint_target == actual.position


def test_step_partial_mask_mixes_interpolated_and_actual() -> None:
    actual = _joints()
    mask = tuple(i == 0 for i in range(17))
    wp_target = tuple(0.5 for _ in range(17))
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, wp_target, mask),)))
    cmd, _ = controller.step(actual, 2_000)
    assert cmd.joint_target[0] == 0.5
    for i in range(1, 17):
        assert cmd.joint_target[i] == actual.position[i]


def test_step_uncontrolled_joints_are_not_zeroed() -> None:
    actual = _joints()
    assert actual.position[9] != 0.0
    mask = tuple(i != 9 for i in range(17))
    wp_target = tuple(0.0 for _ in range(17))
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, wp_target, mask),)))
    cmd, _ = controller.step(actual, 2_000)
    assert cmd.joint_target[9] == actual.position[9]
    assert cmd.joint_target[9] != 0.0


def test_step_mask_change_rejected() -> None:
    """相邻路点mask不一致 → FAILED。"""
    actual = _joints()
    mask_a = tuple(i == 0 for i in range(17))
    mask_b = tuple(i == 1 for i in range(17))
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((
        _step_waypoint(0.0, actual.position, mask_a),
        _step_waypoint(2.0, actual.position, mask_b),
    )))
    cmd, status = controller.step(actual, 2_000)
    assert cmd.valid is False
    assert "mask" in status.failure_reason.lower()


# ============================================================================
# D. 新鲜度检查（3项新增）
# ============================================================================


def test_step_stale_feedback_rejected() -> None:
    """actual_joints 时间戳太旧 → FAILED。"""
    actual = RobotJointState(position=_joints().position, velocity=(0.0,)*17, effort=(0.0,)*17,
                              timestamp_ns=0, valid=True)
    controller = ArmExecutionController(_test_config(feedback_max_age_ns=5_000))
    controller.start_trajectory(_execution_trajectory())
    cmd, status = controller.step(actual, 10_000)
    assert cmd.valid is False
    assert "过期" in status.failure_reason


def test_step_future_feedback_rejected() -> None:
    """actual_joints 时间戳来自未来 → FAILED。"""
    actual = RobotJointState(position=_joints().position, velocity=(0.0,)*17, effort=(0.0,)*17,
                              timestamp_ns=20_000, valid=True)
    controller = ArmExecutionController(_test_config(feedback_max_age_ns=5_000))
    controller.start_trajectory(_execution_trajectory())
    cmd, status = controller.step(actual, 10_000)
    assert cmd.valid is False
    assert "未来" in status.failure_reason


def test_step_stale_trajectory_rejected() -> None:
    """trajectory.timestamp_ns 太旧 → FAILED。"""
    controller = ArmExecutionController(_test_config(trajectory_max_age_ns=3_000))
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0),), timestamp_ns=5_000))
    cmd, status = controller.step(_joints(), 10_000)
    assert cmd.valid is False
    assert "过期" in status.failure_reason


# ============================================================================
# E. 首路点保护（1项新增）
# ============================================================================


def test_step_late_first_waypoint_uses_actual_position() -> None:
    """首路点 t>0 → 首次step用实际位置代替跳变。"""
    base = _joints().position
    far_target = tuple(v + 5.0 for v in base)
    controller = ArmExecutionController(_test_config(max_joint_velocity_17=None))
    controller.start_trajectory(_step_trajectory((_step_waypoint(1.0, far_target), _step_waypoint(3.0, far_target))))
    cmd, _ = controller.step(_joints(), 2_000)
    # 首路点 t=1s > 0 → 应以实际位置为隐式 t=0 路点
    assert cmd.joint_target == base


# ============================================================================
# F. COMPLETED 终态（1项新增）
# ============================================================================


def test_step_repeated_completed_returns_safe_hold() -> None:
    """首次 COMPLETED 后重复调用 → valid=False 安全保持。"""
    target = _joints().position
    actual = RobotJointState(position=target, velocity=(0.0,)*17, effort=(0.0,)*17, timestamp_ns=10_000)
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, target), _step_waypoint(0.1, target))))
    controller.step(actual, 0)
    controller.step(actual, 200_000_000)
    c1, s1 = controller.step(actual, 300_000_000)
    assert s1.success is True
    # 重复调用
    c2, s2 = controller.step(actual, 400_000_000)
    assert c2.valid is False
    assert s2.state == "COMPLETED"
    assert s2.success  # 终态保持 success=True 但不再产生新有效命令


# ============================================================================
# G. fail closed 验证（无配置永不 success）
# ============================================================================


def test_step_no_config_never_completes() -> None:
    """无 ArmExecutionConfig（安全参数全 None）→ fail closed，永不 COMPLETED。"""
    target = _joints().position
    actual = RobotJointState(position=target, velocity=(0.0,)*17, effort=(0.0,)*17, timestamp_ns=10_000)
    controller = ArmExecutionController()
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, target), _step_waypoint(0.1, target))))
    controller.step(actual, 0)
    cmd, status = controller.step(actual, 200_000_000)
    assert status.success is False
    assert "容差" in status.failure_reason


# ============================================================================
# H. 实际反馈闭环（4项：到位 + 稳定）
# ============================================================================


def test_step_time_ended_but_joints_not_arrived_is_not_success() -> None:
    """轨迹时间结束但实际关节超差 → success=False。"""
    target = _joints().position
    far_away = tuple(v + 0.5 for v in target)
    far_joints = RobotJointState(position=far_away, velocity=(0.0,)*17, effort=(0.0,)*17, timestamp_ns=10_000)
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, target), _step_waypoint(0.2, target))))
    controller.step(_joints(), 2_000)
    _, status = controller.step(far_joints, 4_000)
    assert status.success is False


def test_step_within_tolerance_but_insufficient_settle_cycles() -> None:
    """到位但稳定周期不足(settle=2, 只稳了1) → success=False。"""
    target = _joints().position
    actual = RobotJointState(position=target, velocity=(0.0,)*17, effort=(0.0,)*17, timestamp_ns=10_000)
    controller = ArmExecutionController(_test_config(settle_cycles=5))
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, target), _step_waypoint(0.1, target))))
    controller.step(actual, 2_000)
    _, status = controller.step(actual, 4_000)  # 仅1个稳定周期
    assert status.success is False


def test_step_settle_cycles_reached_reports_success() -> None:
    """连续稳定周期达标 → success=True。"""
    target = _joints().position
    actual = RobotJointState(position=target, velocity=(0.0,)*17, effort=(0.0,)*17, timestamp_ns=10_000)
    controller = ArmExecutionController(_test_config(settle_cycles=2))
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, target), _step_waypoint(0.1, target))))
    controller.step(actual, 0)
    controller.step(actual, 200_000_000)
    _, status = controller.step(actual, 300_000_000)
    assert status.success is True


def test_step_one_cycle_out_of_tolerance_resets_counter() -> None:
    """中间一周期超差 → 稳定计数清零。"""
    target = _joints().position
    far = tuple(v + 0.5 for v in target)
    actual_ok = RobotJointState(position=target, velocity=(0.0,)*17, effort=(0.0,)*17, timestamp_ns=10_000)
    actual_bad = RobotJointState(position=far, velocity=(0.0,)*17, effort=(0.0,)*17, timestamp_ns=10_000)
    controller = ArmExecutionController(_test_config(settle_cycles=3))
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, target), _step_waypoint(5.0, target))))
    controller.step(actual_ok, 0)
    controller.step(actual_ok, 100_000_000)
    controller.step(actual_bad, 200_000_000)
    assert controller._stable_cycle_count == 0


def test_step_max_joint_error_comes_from_actual_feedback() -> None:
    target = _joints().position
    offset = tuple(target[i] + (0.05 if i == 0 else 0.0) for i in range(17))
    actual = RobotJointState(position=offset, velocity=(0.0,)*17, effort=(0.0,)*17, timestamp_ns=10_000)
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, target),)))
    _, status = controller.step(actual, 2_000)
    assert status.max_joint_error == pytest.approx(0.05)


def test_step_max_joint_error_only_counts_controlled_joints() -> None:
    actual = _step_joints()
    target = _joints().position
    mask = tuple(i == 0 for i in range(17))
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0, target, mask),)))
    _, status = controller.step(actual, 2_000)
    assert status.max_joint_error == pytest.approx(abs(target[0] - actual.position[0]))


# ============================================================================
# I. 超时和失败（4项）
# ============================================================================


def test_step_total_timeout_enters_failure() -> None:
    controller = ArmExecutionController(_test_config(total_timeout_ns=500))
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(10.0))))
    controller.step(_joints(), 2_000)
    cmd, status = controller.step(_joints(), 3_000)
    assert cmd.valid is False
    assert "超时" in status.failure_reason


def test_step_after_failed_blocks_old_trajectory() -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(5.0))))
    controller.step(_joints(), 10_000)
    controller.step(_joints(), 5_000)  # 时间倒退触发 FAILED
    cmd, _ = controller.step(_joints(), 6_000)
    assert cmd.valid is False


def test_step_failure_command_does_not_use_all_zeros() -> None:
    actual = _joints()
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(10.0))))
    controller.step(actual, 10_000)
    controller.step(actual, 5_000)  # 倒退
    cmd, _ = controller.step(actual, 6_000)
    assert cmd.valid is False
    assert cmd.joint_target == actual.position
    assert cmd.joint_target != (0.0,) * 17


def test_step_failure_reason_is_readable() -> None:
    controller = ArmExecutionController(_test_config())
    _, status = controller.step(_joints(), 2_000)
    assert len(status.failure_reason) > 0
    assert "轨迹" in status.failure_reason


# ============================================================================
# J. 生命周期（4项）
# ============================================================================


def test_step_new_trajectory_clears_previous_progress() -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(1.0))))
    controller.step(_joints(), 500_000_000)
    assert controller._trajectory_started_ns is not None
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(2.0)), trajectory_id="second"))
    assert controller._trajectory_started_ns is None
    assert controller._waypoint_index == 0
    assert controller._stable_cycle_count == 0


def test_step_rejected_trajectory_clears_old_execution_context() -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(1.0))))
    controller.step(_joints(), 500_000_000)
    controller.start_trajectory(_execution_trajectory(waypoints=()))
    assert controller._trajectory is None
    assert controller._trajectory_started_ns is None


def test_step_reset_clears_all_state() -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(1.0))))
    controller.step(_joints(), 500_000_000)
    controller.reset()
    assert controller._trajectory is None
    assert controller._trajectory_started_ns is None
    assert controller._waypoint_index == 0
    assert controller._stable_cycle_count == 0
    assert controller._cached_verification is None
    assert controller._last_step_ns is None
    assert controller._completed is False
    assert controller.local_phase is LocalPhase.IDLE


def test_step_new_trajectory_first_step_records_fresh_start_time() -> None:
    controller = ArmExecutionController(_test_config())
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(1.0))))
    controller.step(_joints(), 5_000_000_000)
    assert controller._trajectory_started_ns == 5_000_000_000
    controller.start_trajectory(_step_trajectory((_step_waypoint(0.0), _step_waypoint(2.0)), trajectory_id="second"))
    controller.step(_joints(), 7_000_000_000)
    assert controller._trajectory_started_ns == 7_000_000_000

