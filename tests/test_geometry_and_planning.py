"""几何、三维感知、导航和机械臂规划的纯 Python 回归测试。

覆盖模块：``interfaces.py`` 的相关数据契约、``navigation.py`` 的基础几何与导航骨架、
``perception_3d.py`` 的反投影/深度采样/MMK2FK 适配，以及 ``arm_planning.py`` 的
MMK2Kdl 适配与规划骨架；同时静态检查少量 config/launch 约定。主要维护者分别是视觉2、
底盘2、机械臂1和系统负责人。本文件包含单元测试、安全回归测试和轻量静态集成测试，
不启动 ROS、MuJoCo 或真实机械臂，也不需要官方 Docker。

测试使用 ``_RecordingFK``、``_RecordingKDL`` 等 fake 代替官方 FK/KDL，使用
``patch`` mock 延迟导入并记录调用；这些替身只验证团队适配层如何传参，不代表官方
求解结果正确。pytest 会执行以 ``test_`` 开头的函数；``assert`` 表示条件必须成立，
``pytest.raises`` 表示该输入必须明确失败，``parametrize`` 用多组输入重复检查同一规则。
``tmp_path`` 和 ``monkeypatch`` 是 pytest 提供的 fixture，测试函数不会进入机器人运行。

测试通过能够证明纯 Python 校验、几何公式、fake 依赖下的适配映射以及当前配置静态
约定符合预期；不能证明真实相机同步、官方 MuJoCo 坐标完全正确、真实 IK 可达、机械臂
能够执行、ROS 控制链有效、三个任务端到端完成或最终比赛得分。

单文件运行：
``python3 -m pytest -q tests/test_geometry_and_planning.py -p no:cacheprovider``

全套运行：
``PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider``
"""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from team_sorting.arm_planning import ArmPlanner, OfficialKDLAdapter, _pose_to_matrix
from team_sorting.interfaces import (
    BaseState,
    CameraIntrinsics,
    Detection2D,
    DepthFrame,
    NavGoal,
    ObjectEstimate3D,
    Pose3D,
    RobotJointState,
    SlotType,
    TaskSpec,
)
from team_sorting.navigation import (
    Bounds3D,
    NavigationConfig,
    NavigationController,
    classify_slot_type,
    distance_xy,
    wrap_to_pi,
)
from team_sorting.perception_2d import OfficialYoloAdapter
from team_sorting.perception_3d import (
    CameraTransformProvider,
    Perception3DEstimator,
    median_depth_m,
    project_pixel_to_camera,
)
from team_sorting.ros_nodes import _perception_pipeline_from_config


# ---------------------------------------------------------------------------
# 共享构造器、fake依赖与测试隔离
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个用例使用独立的sys.path副本，避免self_check搜索路径泄漏到后续测试。"""

    monkeypatch.setattr(sys, "path", list(sys.path))


# 本文件少量访问 ``_searched``、``_fk``、``_solver``、``_ik_adapter`` 和私有几何
# 函数，用于保护路径诊断、失败清理、依赖注入与原子几何规则。这些是内部回归测试，
# 可能随内部重构同步调整；新增测试应优先使用公开接口，不继续无边界扩张白盒访问。


def _actual_joints() -> RobotJointState:
    return RobotJointState(
        position=(0.1, 0.0, -0.2, 0.0, 0.1, -0.1, 0.2, -0.2, 0.3, 0.5,
                  0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.5),
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=100,
    )


def _intrinsics(
    k: tuple[float, ...] = (
        500.0, 0.0, 320.0,
        0.0, 500.0, 240.0,
        0.0, 0.0, 1.0,
    ),
    *,
    frame_id: str = "camera_optical_frame",
    timestamp_ns: int = 100,
    valid: bool = True,
) -> CameraIntrinsics:
    return CameraIntrinsics(
        k=k,
        width=640,
        height=480,
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        valid=valid,
        failure_reason="内参未就绪" if not valid else "",
    )


def _depth(image: object, unit_scale_m: float = 0.001) -> DepthFrame:
    return DepthFrame(
        image=image,
        unit_scale_m=unit_scale_m,
        frame_id="camera_optical_frame",
        timestamp_ns=100,
    )


def _base(
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    *,
    valid: bool = True,
) -> BaseState:
    return BaseState(
        position_xyz=(1.0, 2.0, 3.0),
        orientation_xyzw=orientation_xyzw,
        yaw=0.0,
        linear_velocity_xyz=(0.0, 0.0, 0.0),
        angular_velocity_xyz=(0.0, 0.0, 0.0),
        frame_id="odom",
        timestamp_ns=100,
        valid=valid,
        failure_reason="Odom无效" if not valid else "",
    )


class _RecordingFK:
    """不加载MuJoCo，只记录适配器传入MMK2FK的值。"""

    def __init__(
        self,
        camera_position: object = (0.0, 0.0, 0.0),
        camera_quaternion_wxyz: object = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.camera_position = camera_position
        self.camera_quaternion_wxyz = camera_quaternion_wxyz
        self.calls: dict[str, object] = {}

    def set_base_pose(self, position: object, orientation: object) -> None:
        self.calls["base"] = (tuple(position), tuple(orientation))

    def set_slide_joint(self, value: object) -> None:
        self.calls["slide"] = value

    def set_head_joints(self, values: object) -> None:
        self.calls["head"] = tuple(values)

    def set_left_arm_joints(self, values: object) -> None:
        self.calls["left"] = tuple(values)

    def set_right_arm_joints(self, values: object) -> None:
        self.calls["right"] = tuple(values)

    def get_head_camera_pose(self) -> tuple[object, object]:
        return self.camera_position, self.camera_quaternion_wxyz


def _fake_fk_module(captured: dict[str, object]) -> SimpleNamespace:
    class _LoadedFK:
        def __init__(self, mjcf_path: str) -> None:
            captured["load_path"] = mjcf_path
            captured["xml"] = Path(mjcf_path).read_text(encoding="utf-8")

    return SimpleNamespace(MMK2FK=_LoadedFK)


class _RecordingKDL:
    """不用MuJoCo或解析IK，只记录薄适配器交给官方求解器的数据。"""

    def __init__(
        self,
        *,
        solutions: object = None,
        left_fk: object = None,
        right_fk: object = None,
        slide_limits: object = (-0.04, 0.87),
    ) -> None:
        self.solutions = solutions
        self.left_fk = np.eye(4) if left_fk is None else left_fk
        self.right_fk = np.eye(4) if right_fk is None else right_fk
        self.spine = SimpleNamespace(joint_limits=slide_limits)
        self.forward_input: tuple[float, ...] | None = None
        self.inverse_call: dict[str, object] | None = None

    def forward_kinematics(self, values: object) -> tuple[object, object]:
        self.forward_input = tuple(float(value) for value in values)
        return self.left_fk, self.right_fk

    def inverse_kinematics(self, **kwargs: object) -> object:
        self.inverse_call = kwargs
        return self.solutions


class _InjectedIKAdapter:
    """验证ArmPlanner只保存依赖，不在构造时自检或创建官方求解器。"""

    def __init__(self) -> None:
        self.self_check_called = False

    def self_check(self) -> None:
        self.self_check_called = True
        raise AssertionError("ArmPlanner构造时不应调用self_check")

    def solve_ik(self, *args: object, **kwargs: object) -> object:
        return args, kwargs


def _fake_kdl_module(solver: _RecordingKDL) -> SimpleNamespace:
    return SimpleNamespace(MMK2Kdl=lambda: solver)


def _planning_joints(*, slide: float = 0.1, valid: bool = True) -> RobotJointState:
    return RobotJointState(
        position=(slide, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=100,
        valid=valid,
        failure_reason="JointState无效" if not valid else "",
    )


def _arm_target(frame_id: str = "footprint") -> Pose3D:
    return Pose3D((0.4, 0.2, 0.8), (0.0, 0.0, 0.0, 1.0), frame_id)


# ---------------------------------------------------------------------------
# interfaces与基础几何
# ---------------------------------------------------------------------------


def test_interfaces_and_geometry() -> None:
    joints = _actual_joints()
    base = BaseState(
        position_xyz=(1.0, 2.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        yaw=0.0,
        linear_velocity_xyz=(0.0, 0.0, 0.0),
        angular_velocity_xyz=(0.0, 0.0, 0.0),
        frame_id="odom",
        timestamp_ns=100,
    )
    estimate = ObjectEstimate3D("pink", (1.2, 2.1, 0.8), 0.9, "odom", 100)
    assert joints.valid and base.valid and estimate.slot_type is SlotType.UNKNOWN
    assert wrap_to_pi(3.0 * math.pi) == pytest.approx(-math.pi)
    assert distance_xy((0.0, 0.0, 9.0), (3.0, 4.0, -9.0)) == pytest.approx(5.0)

    table = Bounds3D(0.0, 2.0, 0.0, 3.0, 0.5, 1.0)
    shelf = Bounds3D(3.0, 4.0, 0.0, 1.0, 0.0, 2.0)
    assert classify_slot_type(estimate.position_xyz, table, shelf) is SlotType.TABLE
    assert classify_slot_type((9.0, 9.0, 9.0), table, shelf) is SlotType.UNKNOWN


# ---------------------------------------------------------------------------
# navigation纯函数
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (math.pi, -math.pi),
        (-math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
        (-3.0 * math.pi, -math.pi),
        (1.0e-12, 1.0e-12),
        (-1.0e-12, -1.0e-12),
    ],
)
def test_wrap_to_pi_boundaries(angle: float, expected: float) -> None:
    assert wrap_to_pi(angle) == pytest.approx(expected)


@pytest.mark.parametrize("angle", [True, False, "1.0", None, math.nan, math.inf, -math.inf])
def test_wrap_to_pi_rejects_non_real_or_non_finite_values(angle: object) -> None:
    with pytest.raises(ValueError, match="角度必须是真实有限数"):
        wrap_to_pi(angle)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((0.0, 0.0), (0.0, 0.0), 0.0),
        ((0.0, 0.0), (3.0, 4.0), 5.0),
        ((0.0, 0.0, 99.0), (3.0, 4.0, -99.0), 5.0),
    ],
)
def test_distance_xy_uses_first_two_finite_coordinates(
    first: object, second: object, expected: float
) -> None:
    assert distance_xy(first, second) == pytest.approx(expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ((), (0.0, 0.0)),
        ((0.0,), (0.0, 0.0)),
        (None, (0.0, 0.0)),
        (1.0, (0.0, 0.0)),
        ("01", (0.0, 0.0)),
        ((True, 0.0), (0.0, 0.0)),
        (("0", 0.0), (0.0, 0.0)),
        ((math.nan, 0.0), (0.0, 0.0)),
        ((0.0, math.inf), (0.0, 0.0)),
    ],
)
def test_distance_xy_rejects_invalid_sequences(first: object, second: object) -> None:
    with pytest.raises(ValueError):
        distance_xy(first, second)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bounds",
    [
        Bounds3D(math.nan, 1.0, 0.0, 1.0, 0.0, 1.0),
        Bounds3D(0.0, math.inf, 0.0, 1.0, 0.0, 1.0),
        Bounds3D(True, 1.0, 0.0, 1.0, 0.0, 1.0),
        Bounds3D("0", 1.0, 0.0, 1.0, 0.0, 1.0),  # type: ignore[arg-type]
        Bounds3D(2.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    ],
)
def test_bounds3d_invalid_boundaries_return_false(bounds: Bounds3D) -> None:
    assert not bounds.contains((0.5, 0.5, 0.5))


@pytest.mark.parametrize(
    "point",
    [
        None,
        (0.5, 0.5),
        (0.5, 0.5, 0.5, 0.5),
        (True, 0.5, 0.5),
        ("0.5", 0.5, 0.5),
        (math.nan, 0.5, 0.5),
        (0.5, math.inf, 0.5),
    ],
)
def test_bounds3d_invalid_points_return_false(point: object) -> None:
    bounds = Bounds3D(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    assert not bounds.contains(point)  # type: ignore[arg-type]


def test_slot_classification_boundaries_invalid_points_and_overlap_policy() -> None:
    table = Bounds3D(0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
    shelf = Bounds3D(1.0, 3.0, 1.0, 3.0, 0.0, 2.0)
    assert classify_slot_type((0.5, 0.5, 0.5), table, shelf) is SlotType.TABLE
    assert classify_slot_type((2.5, 2.5, 0.5), table, shelf) is SlotType.SHELF
    assert classify_slot_type((9.0, 9.0, 9.0), table, shelf) is SlotType.UNKNOWN
    assert classify_slot_type((math.nan, 1.0, 1.0), table, shelf) is SlotType.UNKNOWN
    assert classify_slot_type((1.5, 1.5, 1.0), table, shelf) is SlotType.TABLE


# ---------------------------------------------------------------------------
# NavigationController站位和闭环控制
# ---------------------------------------------------------------------------

def _nav_base(
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    *,
    frame: str = "odom",
    stamp: int = 1_000_000_000,
    valid: bool = True,
) -> BaseState:
    return BaseState(
        (x, y, 0.0),
        (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
        yaw,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        frame,
        stamp,
        valid,
    )


def _nav_task(place: tuple[float, float, float] = (2.0, 0.0, 0.5)) -> TaskSpec:
    return TaskSpec(1, "move box", "box", "box_body", "pink", "point", place, 0.1)


def _nav_goal(
    x: float,
    y: float,
    yaw: float,
    *,
    deadline: int = 2_000_000_000,
    frame: str = "odom",
    valid: bool = True,
) -> NavGoal:
    return NavGoal("goal", "pick", (x, y, yaw), frame, 0.05, 0.1, deadline, valid)


def test_navigation_pick_goal_stands_off_and_faces_object() -> None:
    controller = NavigationController()
    target = ObjectEstimate3D("pink", (2.0, 0.0, 0.8), 0.9, "odom", 1_000_000_000)
    goal = controller.build_pick_goal(_nav_task(), target, _nav_base(), 1_000_000_000)
    assert goal.pose_xyyaw[:2] != pytest.approx(target.position_xyz[:2])
    assert distance_xy(goal.pose_xyyaw, target.position_xyz) == pytest.approx(0.6)
    assert goal.pose_xyyaw[2] == pytest.approx(
        math.atan2(
            target.position_xyz[1] - goal.pose_xyyaw[1],
            target.position_xyz[0] - goal.pose_xyyaw[0],
        )
    )


@pytest.mark.parametrize("bad_x", [math.nan, math.inf])
def test_navigation_pick_goal_rejects_invalid_coordinates_and_frame(bad_x: float) -> None:
    controller = NavigationController()
    with pytest.raises(ValueError):
        controller.build_pick_goal(
            _nav_task(),
            ObjectEstimate3D("pink", (bad_x, 0.0, 0.8), 0.9, "odom", 1_000_000_000),
            _nav_base(),
            1_000_000_000,
        )
    with pytest.raises(ValueError, match="frame"):
        controller.build_pick_goal(
            _nav_task(),
            ObjectEstimate3D("pink", (2.0, 0.0, 0.8), 0.9, "base_link", 1_000_000_000),
            _nav_base(),
            1_000_000_000,
        )


def test_navigation_place_goal_requires_approved_world_to_planning_transform() -> None:
    controller = NavigationController()
    task = _nav_task()
    with pytest.raises(NotImplementedError, match="world→planning"):
        controller.build_place_goal(task, _nav_base(frame="odom"), 1_000_000_000)
    with pytest.raises(NotImplementedError, match="world→planning"):
        controller.build_place_goal(
            task, _nav_base(frame="world"), 1_000_000_000
        )
    with pytest.raises(ValueError):
        controller.build_place_goal(
            _nav_task((math.nan, 0.0, 0.5)),
            _nav_base(frame="world"),
            1_000_000_000,
        )


def test_navigation_goal_generation_rejects_stale_base_and_target_at_boundary() -> None:
    config = NavigationConfig(odom_max_age_ns=100, target_max_age_ns=100)
    controller = NavigationController(config)
    task = _nav_task()
    target_at_boundary = ObjectEstimate3D(
        "pink", (2.0, 0.0, 0.8), 0.9, "odom", 900
    )
    controller.build_pick_goal(
        task, target_at_boundary, _nav_base(stamp=900), 1_000
    )
    with pytest.raises(ValueError, match="Odom 已过期"):
        controller.build_pick_goal(
            task, target_at_boundary, _nav_base(stamp=899), 1_000
        )
    with pytest.raises(ValueError, match="ObjectEstimate3D 已过期"):
        controller.build_pick_goal(
            task,
            ObjectEstimate3D("pink", (2.0, 0.0, 0.8), 0.9, "odom", 899),
            _nav_base(stamp=900),
            1_000,
        )


def test_navigation_config_is_injected_and_validated() -> None:
    config = NavigationConfig(max_abs_v_mps=0.1, max_abs_w_radps=0.2)
    command, _ = NavigationController(config).update(
        _nav_base(), _nav_goal(1.0, 1.0, 0.0), 1_000_000_000
    )
    assert abs(command.v) <= 0.1
    assert abs(command.w) <= 0.2
    with pytest.raises(ValueError, match="standoff_m"):
        NavigationConfig(standoff_m=math.nan)


@pytest.mark.parametrize(
    ("goal", "expected_w_sign"),
    [
        (_nav_goal(1.0, 0.0, 0.0), 0),
        (_nav_goal(1.0, 1.0, 0.0), 1),
        (_nav_goal(1.0, -1.0, 0.0), -1),
    ],
)
def test_navigation_update_steers_toward_goal(
    goal: NavGoal, expected_w_sign: int
) -> None:
    command, status = NavigationController().update(
        _nav_base(), goal, 1_000_000_000
    )
    assert not status.success
    assert command.v >= 0.0
    assert (command.w > 0) - (command.w < 0) == expected_w_sign
    assert abs(command.v) <= 0.25
    assert abs(command.w) <= 0.5


def test_navigation_update_wraps_heading_and_gates_large_error() -> None:
    base = _nav_base(yaw=math.pi - 0.05)
    goal = _nav_goal(-1.0, -0.01, -math.pi + 0.05)
    command, _ = NavigationController().update(base, goal, 1_000_000_000)
    assert abs(command.w) < 0.2
    side_command, _ = NavigationController().update(
        _nav_base(), _nav_goal(0.0, 1.0, 0.0), 1_000_000_000
    )
    assert side_command.v == 0.0
    assert side_command.w > 0.0


def test_navigation_update_slows_near_goal_and_requires_final_yaw() -> None:
    controller = NavigationController()
    far, far_status = controller.update(
        _nav_base(), _nav_goal(1.0, 0.0, 0.0), 1_000_000_000
    )
    near, near_status = controller.update(
        _nav_base(), _nav_goal(0.1, 0.0, 0.0), 1_000_000_000
    )
    assert 0.0 < near.v < far.v
    assert not far_status.success and not near_status.success
    align, align_status = controller.update(
        _nav_base(), _nav_goal(0.01, 0.0, 0.5), 1_000_000_000
    )
    assert align.v == 0.0 and align.w > 0.0
    assert not align_status.success
    arrived, arrived_status = controller.update(
        _nav_base(), _nav_goal(0.01, 0.0, 0.05), 1_000_000_000
    )
    assert (arrived.v, arrived.w) == (0.0, 0.0)
    assert arrived_status.success


def test_navigation_update_safely_stops_on_timeout_invalid_and_stale_odom() -> None:
    controller = NavigationController()
    cases = (
        (_nav_base(), _nav_goal(1.0, 0.0, 0.0, deadline=999_999_999)),
        (_nav_base(valid=False), _nav_goal(1.0, 0.0, 0.0)),
        (_nav_base(stamp=800_000_000), _nav_goal(1.0, 0.0, 0.0)),
        (_nav_base(), _nav_goal(math.nan, 0.0, 0.0)),
        (_nav_base(), _nav_goal(1.0, 0.0, 0.0, frame="world")),
    )
    for base, goal in cases:
        command, status = controller.update(base, goal, 1_000_000_000)
        assert (command.v, command.w) == (0.0, 0.0)
        assert not status.success
        assert status.failure_reason


def test_navigation_deadline_is_exclusive_and_stops_at_equal_timestamp() -> None:
    command, status = NavigationController().update(
        _nav_base(),
        _nav_goal(1.0, 0.0, 0.0, deadline=1_000_000_000),
        1_000_000_000,
    )
    assert (command.v, command.w) == (0.0, 0.0)
    assert not command.valid
    assert status.state == "timeout"
    assert not status.success


def test_navigation_fails_closed_when_finite_arithmetic_overflows() -> None:
    controller = NavigationController()
    with pytest.raises(ValueError):
        controller.build_pick_goal(
            _nav_task(),
            ObjectEstimate3D(
                "pink", (1e308, 0.0, 0.8), 0.9, "odom", 1_000_000_000
            ),
            _nav_base(x=-1e308),
            1_000_000_000,
        )

    command, status = controller.update(
        _nav_base(x=-1e308),
        _nav_goal(1e308, 0.0, 0.0),
        1_000_000_000,
    )
    assert (command.v, command.w) == (0.0, 0.0)
    assert not command.valid
    assert not status.success
    assert status.failure_reason
    assert math.isfinite(status.distance_error)
    assert math.isfinite(status.yaw_error)


# ---------------------------------------------------------------------------
# 延迟依赖导入烟雾测试
# ---------------------------------------------------------------------------

# 其中plan_grasp异常是临时防伪成功约束。ArmPlanner实现时，应在同一提交中把本测试
# 收窄为“构造与延迟导入不加载官方依赖”，具体规划行为由末尾ArmPlanner分区验证。
def test_projection_and_delayed_adapters_import_without_official_environment() -> None:
    intrinsics = _intrinsics()
    assert project_pixel_to_camera(320.0, 240.0, 2.0, intrinsics) == pytest.approx(
        (0.0, 0.0, 2.0)
    )

    adapters = (OfficialYoloAdapter(), CameraTransformProvider(), OfficialKDLAdapter())
    assert all(adapter is not None for adapter in adapters)
    with pytest.raises(NotImplementedError, match="机械臂1负责人"):
        ArmPlanner(OfficialKDLAdapter()).plan_grasp(
            ObjectEstimate3D("yellow", (0.4, 0.0, 0.8), 0.8, "base_link", 10),
            _actual_joints(),
        )


# ---------------------------------------------------------------------------
# config与launch约定
# ---------------------------------------------------------------------------


def _project_config() -> dict[str, object]:
    """读取仓库配置；这些静态检查不会启动ROS或验证官方参数。"""

    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    return config


def _strict_finite_number(value: object) -> float:
    """配置数值不得利用Python中bool是int子类的特性蒙混过关。"""

    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    number = float(value)
    assert math.isfinite(number)
    return number


def test_config_uses_official_odom_and_aligned_depth_topics() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["topics"]["odom"] == "/slamware_ros_sdk_server_node/odom"
    assert config["topics"]["depth"] == "/head_camera/aligned_depth_to_color/image_raw"
    assert config["topics"]["rgb"] == "/head_camera/color/image_raw"
    assert config["topics"]["camera_info"] == "/head_camera/color/camera_info"
    assert config["perception"]["depth_unit_scale_m"] == pytest.approx(0.001)
    launch_text = (config_path.parent.parent / "launch" / "team.launch.xml").read_text(
        encoding="utf-8"
    )
    assert '<arg name="record_data" default="false"' in launch_text
    assert 'if="$(var record_data)"' in launch_text


def test_config_has_required_mapping_sections_and_absolute_topics() -> None:
    config = _project_config()
    required_sections = {
        "official",
        "frames",
        "topics",
        "timing",
        "perception",
        "fsm",
        "slot_bounds",
        "action_mux",
        "recorder",
        "joint_aliases",
    }
    assert required_sections <= config.keys()
    assert all(isinstance(config[name], dict) for name in required_sections)

    topics = config["topics"]
    assert isinstance(topics, dict)
    ordinary_topics = {name: topic for name, topic in topics.items() if name != "official_commands"}
    assert ordinary_topics
    assert all(
        isinstance(topic, str) and bool(topic) and topic.startswith("/")
        for topic in ordinary_topics.values()
    )

    official_commands = topics["official_commands"]
    assert isinstance(official_commands, dict)
    assert set(official_commands) == {"cmd_vel", "slide", "head", "left_arm", "right_arm"}
    assert all(
        isinstance(topic, str) and bool(topic) and topic.startswith("/")
        for topic in official_commands.values()
    )


def test_config_numeric_types_finiteness_and_ranges() -> None:
    config = _project_config()
    timing = config["timing"]
    perception = config["perception"]
    fsm = config["fsm"]
    action_mux = config["action_mux"]
    assert isinstance(timing, dict)
    assert isinstance(perception, dict)
    assert isinstance(fsm, dict)
    assert isinstance(action_mux, dict)

    assert _strict_finite_number(timing["control_rate_hz"]) > 0.0
    assert _strict_finite_number(timing["command_ttl_s"]) > 0.0
    assert _strict_finite_number(timing["state_max_delta_s"]) >= 0.0

    confidence = _strict_finite_number(perception["confidence_threshold"])
    assert 0.0 <= confidence <= 1.0
    queue_size = perception["sync_queue_size"]
    assert isinstance(queue_size, int) and not isinstance(queue_size, bool) and queue_size > 0
    assert _strict_finite_number(perception["sync_slop_s"]) > 0.0
    assert _strict_finite_number(perception["depth_unit_scale_m"]) > 0.0
    stabilizer = perception["stabilizer_2d"]
    estimator = perception["estimator_3d"]
    assert isinstance(stabilizer, dict)
    assert isinstance(estimator, dict)
    assert set(stabilizer) == {
        "iou_match_threshold",
        "min_confirmed_hits",
        "max_missed_frames",
        "bbox_smoothing_alpha",
        "confidence_smoothing_alpha",
    }
    assert set(estimator) == {
        "depth_radius_px",
        "ema_alpha",
        "converge_frames",
        "max_track_age_s",
        "max_position_jump_m",
        "object_dimensions_m",
    }
    assert isinstance(estimator["depth_radius_px"], int)
    assert isinstance(estimator["converge_frames"], int)
    assert _strict_finite_number(estimator["ema_alpha"]) > 0.0
    assert _strict_finite_number(estimator["max_track_age_s"]) > 0.0
    assert _strict_finite_number(estimator["max_position_jump_m"]) > 0.0
    dimensions = estimator["object_dimensions_m"]
    assert isinstance(dimensions, dict)
    assert set(dimensions) == set(OfficialYoloAdapter.CLASS_NAMES)
    for values in dimensions.values():
        assert isinstance(values, list) and len(values) == 3
        assert tuple(_strict_finite_number(value) for value in values) == pytest.approx(
            (0.24, 0.16, 0.19)
        )

    retry_count = fsm["max_pick_retries"]
    assert isinstance(retry_count, int) and not isinstance(retry_count, bool) and retry_count >= 0
    assert _strict_finite_number(action_mux["max_abs_base_v"]) >= 0.0
    assert _strict_finite_number(action_mux["max_abs_base_w"]) >= 0.0


def test_config_bounds_and_action_limits_have_safe_static_shapes() -> None:
    """只锁定当前团队安全占位和17维顺序，不证明官方区域或限位已验证。"""

    config = _project_config()
    slot_bounds = config["slot_bounds"]
    action_mux = config["action_mux"]
    assert isinstance(slot_bounds, dict)
    assert isinstance(action_mux, dict)

    for name in ("table", "shelf"):
        values = slot_bounds[name]
        assert isinstance(values, list) and len(values) == 6
        numbers = tuple(_strict_finite_number(value) for value in values)
        # 当前每个轴都故意min>max，让未确认区域安全地分类为UNKNOWN。
        assert all(numbers[index] > numbers[index + 1] for index in (0, 2, 4))

    lower = action_mux["joint_lower"]
    upper = action_mux["joint_upper"]
    assert isinstance(lower, list) and isinstance(upper, list)
    assert len(lower) == len(upper) == 17
    lower_numbers = tuple(_strict_finite_number(value) for value in lower)
    upper_numbers = tuple(_strict_finite_number(value) for value in upper)
    assert all(low <= high for low, high in zip(lower_numbers, upper_numbers))


def test_config_recorder_aliases_and_current_frame_defaults() -> None:
    """当前值是仓库约定；测试通过不等于已用官方Docker核对frame、话题或关节名。"""

    config = _project_config()
    frames = config["frames"]
    recorder = config["recorder"]
    aliases = config["joint_aliases"]
    assert isinstance(frames, dict)
    assert isinstance(recorder, dict)
    assert isinstance(aliases, dict)

    assert frames["planning"] == "odom"
    assert recorder["enabled"] is False
    rosbag_topics = recorder["rosbag_topics"]
    assert isinstance(rosbag_topics, list) and rosbag_topics
    assert all(
        isinstance(topic, str) and bool(topic) and topic.startswith("/")
        for topic in rosbag_topics
    )
    assert len(rosbag_topics) == len(set(rosbag_topics))
    assert all(isinstance(source, str) and isinstance(target, str) for source, target in aliases.items())


# ---------------------------------------------------------------------------
# 深度采样与像素反投影
# ---------------------------------------------------------------------------


def test_mono16_depth_millimeters_are_converted_to_meters() -> None:
    depth = _depth(np.full((5, 5), 1200, dtype=np.uint16))
    assert median_depth_m(depth, (1.0, 1.0, 3.0, 3.0), radius_px=1) == pytest.approx(1.2)


def test_projection_of_non_principal_pixel() -> None:
    assert project_pixel_to_camera(370.0, 190.0, 2.0, _intrinsics()) == pytest.approx(
        (0.2, -0.2, 2.0)
    )


@pytest.mark.parametrize(
    ("u", "v", "depth_m"),
    [
        (True, 240.0, 1.0),
        (320.0, False, 1.0),
        (320.0, 240.0, True),
        (float("nan"), 240.0, 1.0),
        (320.0, float("inf"), 1.0),
        (320.0, 240.0, 0.0),
        (320.0, 240.0, -1.0),
        (320.0, 240.0, float("nan")),
    ],
)
def test_projection_rejects_invalid_pixel_or_depth(
    u: object, v: object, depth_m: object
) -> None:
    with pytest.raises(ValueError):
        project_pixel_to_camera(u, v, depth_m, _intrinsics())  # type: ignore[arg-type]


def test_projection_rejects_zero_or_non_finite_focal_length() -> None:
    zero_fx = _intrinsics(k=(0.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="焦距"):
        project_pixel_to_camera(320.0, 240.0, 1.0, zero_fx)

    # CameraIntrinsics自身会拒绝NaN；这里模拟受损外部对象，验证辅助函数仍会防御。
    damaged = SimpleNamespace(
        valid=True,
        failure_reason="",
        k=(float("nan"), 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="NaN"):
        project_pixel_to_camera(320.0, 240.0, 1.0, damaged)  # type: ignore[arg-type]


@pytest.mark.parametrize("radius", [-1, True, 1.5])
def test_median_depth_rejects_invalid_radius(radius: object) -> None:
    with pytest.raises(ValueError, match="radius_px"):
        median_depth_m(
            _depth(np.ones((5, 5))),
            (1.0, 1.0, 3.0, 3.0),
            radius_px=radius,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "bbox",
    [
        (float("nan"), 1.0, 3.0, 3.0),
        (1.0, float("inf"), 3.0, 3.0),
        (3.0, 1.0, 1.0, 3.0),
        (1.0, 3.0, 3.0, 1.0),
        (1.0, 1.0, 1.0, 3.0),
        (True, 1.0, 3.0, 3.0),
    ],
)
def test_median_depth_rejects_invalid_bbox(bbox: object) -> None:
    with pytest.raises(ValueError):
        median_depth_m(_depth(np.ones((5, 5))), bbox)  # type: ignore[arg-type]


def test_median_depth_rejects_bbox_fully_outside_image() -> None:
    with pytest.raises(ValueError, match="范围之外"):
        median_depth_m(_depth(np.ones((5, 5))), (20.0, 20.0, 30.0, 30.0))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((5, 5), dtype=float),
        np.full((5, 5), float("nan")),
        np.full((5, 5), float("inf")),
        np.full((5, 5), -1.0),
    ],
)
def test_median_depth_rejects_window_without_valid_depth(image: object) -> None:
    with pytest.raises(ValueError, match="没有有效深度"):
        median_depth_m(_depth(image), (1.0, 1.0, 3.0, 3.0), radius_px=1)


@pytest.mark.parametrize("image", [np.ones(5), np.ones((5, 5, 1))])
def test_median_depth_requires_strictly_2d_image(image: object) -> None:
    with pytest.raises(ValueError, match="严格二维"):
        median_depth_m(_depth(image), (1.0, 1.0, 3.0, 3.0))


@pytest.mark.parametrize("scale", [0.0, -0.001, True, float("nan"), float("inf")])
def test_median_depth_rejects_invalid_unit_scale(scale: object) -> None:
    with pytest.raises(ValueError, match="unit_scale_m"):
        median_depth_m(
            _depth(np.ones((5, 5)), unit_scale_m=scale),  # type: ignore[arg-type]
            (1.0, 1.0, 3.0, 3.0),
        )


# ---------------------------------------------------------------------------
# CameraTransformProvider
# ---------------------------------------------------------------------------


def test_camera_transform_constructor_rejects_empty_names() -> None:
    with pytest.raises(ValueError, match="module_name"):
        CameraTransformProvider(module_name="")
    with pytest.raises(ValueError, match="output_frame"):
        CameraTransformProvider(output_frame="  ")


def test_self_check_resolves_examples_material_sorting_mjcf(tmp_path: Path) -> None:
    source = (
        tmp_path
        / "examples"
        / "material_sorting"
        / "mjcf"
        / "material_competition.xml"
    )
    source.parent.mkdir(parents=True)
    source.write_text("<mujoco/>", encoding="utf-8")
    captured: dict[str, object] = {}
    provider = CameraTransformProvider(official_root=str(tmp_path), module_name="fake_fk")
    with patch(
        "team_sorting.perception_3d.importlib.import_module",
        return_value=_fake_fk_module(captured),
    ):
        provider.self_check()

    assert captured["load_path"] == str(source)
    assert str(source) in provider._searched
    assert provider._fk is not None


def test_self_check_uses_explicit_team_sorting_mjcf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "explicit.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.setenv("TEAM_SORTING_MJCF", str(source))
    captured: dict[str, object] = {}
    provider = CameraTransformProvider(module_name="fake_fk")
    with patch(
        "team_sorting.perception_3d.importlib.import_module",
        return_value=_fake_fk_module(captured),
    ):
        provider.self_check()

    assert captured["load_path"] == str(source)
    assert provider._searched[-1] == str(source)


def test_self_check_rejects_empty_and_missing_mjcf(tmp_path: Path) -> None:
    empty = tmp_path / "empty.xml"
    empty.write_text("", encoding="utf-8")
    fake_module = _fake_fk_module({})
    for source in (empty, tmp_path / "missing.xml"):
        provider = CameraTransformProvider(mjcf_path=str(source), module_name="fake_fk")
        with patch(
            "team_sorting.perception_3d.importlib.import_module",
            return_value=fake_module,
        ), pytest.raises(RuntimeError, match="找不到"):
            provider.self_check()


def test_prepare_mjcf_replaces_repo_root_and_cleans_temp_file(tmp_path: Path) -> None:
    task_dir = tmp_path / "examples" / "material_sorting"
    source = task_dir / "mjcf" / "material_competition.xml"
    source.parent.mkdir(parents=True)
    source.write_text('<include file="__REPO_ROOT__/assets/robot.xml"/>', encoding="utf-8")
    captured: dict[str, object] = {}
    provider = CameraTransformProvider(mjcf_path=str(source), module_name="fake_fk")
    with patch(
        "team_sorting.perception_3d.importlib.import_module",
        return_value=_fake_fk_module(captured),
    ):
        provider.self_check()

    assert "__REPO_ROOT__" not in str(captured["xml"])
    normalized_xml = str(captured["xml"]).replace("\\", "/")
    assert (task_dir / "assets" / "robot.xml").as_posix() in normalized_xml
    assert not Path(str(captured["load_path"])).exists()


def test_failed_recheck_clears_previous_fk(tmp_path: Path) -> None:
    source = tmp_path / "material_competition.xml"
    source.write_text("<mujoco/>", encoding="utf-8")
    provider = CameraTransformProvider(mjcf_path=str(source), module_name="fake_fk")
    with patch(
        "team_sorting.perception_3d.importlib.import_module",
        return_value=_fake_fk_module({}),
    ):
        provider.self_check()
    assert provider._fk is not None

    with patch(
        "team_sorting.perception_3d.importlib.import_module",
        side_effect=ImportError("fake missing"),
    ), pytest.raises(RuntimeError, match="无法导入"):
        provider.self_check()
    assert provider._fk is None


def test_camera_transform_reorders_quaternion_and_slices_17_joints() -> None:
    provider = CameraTransformProvider(output_frame="odom")
    fake_fk = _RecordingFK()
    provider._fk = fake_fk
    half = math.sqrt(0.5)
    provider.camera_to_output((0.0, 0.0, 0.0), _base((0.0, 0.0, half, half)), _actual_joints())

    assert fake_fk.calls["base"] == ((1.0, 2.0, 3.0), (half, 0.0, 0.0, half))
    positions = _actual_joints().position
    assert fake_fk.calls["slide"] == positions[0]
    assert fake_fk.calls["head"] == positions[1:3]
    assert fake_fk.calls["left"] == positions[3:9]
    assert fake_fk.calls["right"] == positions[10:16]


def test_camera_transform_identity_quaternion_rotates_and_translates() -> None:
    provider = CameraTransformProvider(output_frame="odom")
    provider._fk = _RecordingFK((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    assert provider.camera_to_output((0.5, -0.5, 1.0), _base(), _actual_joints()) == pytest.approx(
        (1.5, 1.5, 4.0)
    )


def test_camera_transform_known_90_degree_rotation() -> None:
    provider = CameraTransformProvider(output_frame="odom")
    half = math.sqrt(0.5)
    provider._fk = _RecordingFK((0.0, 0.0, 0.0), (half, 0.0, 0.0, half))
    assert provider.camera_to_output((1.0, 0.0, 0.0), _base(), _actual_joints()) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-12
    )


def test_camera_transform_rejects_zero_camera_quaternion() -> None:
    provider = CameraTransformProvider()
    provider._fk = _RecordingFK((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    with pytest.raises(RuntimeError, match="范数为零"):
        provider.camera_to_output((1.0, 0.0, 0.0), _base(), _actual_joints())


@pytest.mark.parametrize(
    ("position", "quaternion"),
    [
        ((float("nan"), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, float("inf"), 0.0, 0.0)),
    ],
)
def test_camera_transform_rejects_non_finite_fk_output(
    position: object, quaternion: object
) -> None:
    provider = CameraTransformProvider()
    provider._fk = _RecordingFK(position, quaternion)
    with pytest.raises(RuntimeError, match="位姿无效"):
        provider.camera_to_output((1.0, 0.0, 0.0), _base(), _actual_joints())


def test_camera_transform_rejects_invalid_input_state_and_point() -> None:
    provider = CameraTransformProvider()
    provider._fk = _RecordingFK()
    with pytest.raises(ValueError, match="底盘状态无效"):
        provider.camera_to_output((0.0, 0.0, 1.0), _base(valid=False), _actual_joints())
    invalid_joints = RobotJointState(
        position=_actual_joints().position,
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=100,
        valid=False,
        failure_reason="JointState过期",
    )
    with pytest.raises(ValueError, match="实际关节状态无效"):
        provider.camera_to_output((0.0, 0.0, 1.0), _base(), invalid_joints)
    with pytest.raises(ValueError, match="camera_point_xyz"):
        provider.camera_to_output((True, 0.0, 1.0), _base(), _actual_joints())


# ---------------------------------------------------------------------------
# Perception3DEstimator三维中心与多帧滤波回归
# ---------------------------------------------------------------------------


class _OffsetTransformProvider(CameraTransformProvider):
    """不加载MMK2FK，只给相机点增加固定的米制输出frame偏移。"""

    def __init__(
        self,
        offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        output_frame: str = "odom",
    ) -> None:
        super().__init__(output_frame=output_frame)
        self.offset_xyz = offset_xyz

    def camera_to_output(
        self,
        camera_point_xyz: tuple[float, float, float],
        base: BaseState,
        joints: RobotJointState,
    ) -> tuple[float, float, float]:
        del base, joints
        return tuple(
            camera_point_xyz[index] + self.offset_xyz[index]
            for index in range(3)
        )


def _estimator_depth(
    value_mm: float,
    timestamp_ns: int,
    *,
    image: object | None = None,
    frame_id: str = "camera_optical_frame",
) -> DepthFrame:
    """构造与默认内参尺寸一致的毫米深度帧。"""

    depth_image = (
        np.full((480, 640), value_mm, dtype=float)
        if image is None
        else image
    )
    return DepthFrame(
        image=depth_image,
        unit_scale_m=0.001,
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
    )


def _estimator_detection(
    class_id: str = "pink",
    bbox_xyxy: tuple[float, float, float, float] = (
        319.0,
        239.0,
        321.0,
        241.0,
    ),
    confidence: float = 0.9,
    timestamp_ns: int = 100,
    *,
    frame_id: str = "camera_optical_frame",
    track_id: int | None = None,
    valid: bool = True,
    failure_reason: str = "",
) -> Detection2D:
    return Detection2D(
        class_id=class_id,
        bbox_xyxy=bbox_xyxy,
        confidence=confidence,
        timestamp_ns=timestamp_ns,
        valid=valid,
        failure_reason=failure_reason,
        frame_id=frame_id,
        track_id=track_id,
    )


def test_perception_3d_single_frame_world_coordinate_and_timestamp() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider((1.0, 2.0, 3.0)),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    detection = _estimator_detection(timestamp_ns=100)

    result = estimator.estimate(
        (detection,),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert result.valid
    assert result.position_xyz == pytest.approx((1.0, 2.0, 4.1))
    assert result.frame_id == "odom"
    assert result.timestamp_ns == 100
    assert result.failure_reason == ""
    assert estimator._tracks == {}


def test_perception_3d_depth_failure_is_invalid_without_stopping_batch() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={
            "pink": (0.1, 0.1, 0.2),
            "yellow": (0.1, 0.1, 0.2),
        },
    )
    partly_valid_depth = np.zeros((480, 640), dtype=float)
    partly_valid_depth[236:245, 336:345] = 1000.0
    detections = (
        _estimator_detection(track_id=1),
        _estimator_detection(
            "yellow",
            (339.0, 239.0, 341.0, 241.0),
            0.8,
            track_id=2,
        ),
    )

    results = estimator.estimate(
        detections,
        _estimator_depth(0.0, 100, image=partly_valid_depth),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )

    assert len(results) == 2
    assert not results[0].valid
    assert "深度" in results[0].failure_reason
    assert results[0].confidence == 0.0
    assert results[1].valid
    assert results[1].position_xyz[2] == pytest.approx(1.1)


def test_perception_3d_overflowing_detection_isolated_from_valid_batch_item() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={
            "pink": (0.1, 0.1, 0.2),
            "yellow": (0.1, 0.1, 0.2),
        },
    )
    huge = 10**400
    bad = _estimator_detection(
        bbox_xyxy=(huge, 0.0, huge + 1, 10.0),
        track_id=1,
    )
    good = _estimator_detection("yellow", track_id=2)

    bad_result, good_result = estimator.estimate(
        (bad, good),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )

    assert not bad_result.valid
    assert "有限浮点数" in bad_result.failure_reason
    assert good_result.valid
    with pytest.raises(ValueError, match="有限浮点数"):
        Perception3DEstimator(
            _OffsetTransformProvider(),
            ema_alpha=huge,
        )


def test_perception_3d_requires_compensation_before_claiming_valid_center() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    known = _estimator_detection(
        "pink", (419.0, 289.0, 421.0, 291.0), track_id=1
    )
    unknown = _estimator_detection(
        "brown", (419.0, 289.0, 421.0, 291.0), track_id=2
    )

    known_result, unknown_result = estimator.estimate(
        (known, unknown),
        _estimator_depth(2000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )

    assert known_result.valid
    assert known_result.position_xyz == pytest.approx((0.42, 0.21, 2.1))
    assert not unknown_result.valid
    assert unknown_result.position_xyz == (0.0, 0.0, 0.0)
    assert unknown_result.confidence == 0.0
    assert "中心补偿失败" in unknown_result.failure_reason
    assert "可见表面" in unknown_result.failure_reason


def test_perception_3d_multiframe_ema_and_confidence_converge() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        ema_alpha=0.5,
        converge_frames=4,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    outputs = [
        estimator.estimate(
            (
                _estimator_detection(
                    confidence=0.8,
                    timestamp_ns=timestamp_ns,
                    track_id=7,
                ),
            ),
            _estimator_depth(value_mm, timestamp_ns),
            _intrinsics(timestamp_ns=timestamp_ns),
            _base(),
            _actual_joints(),
        )[0]
        for value_mm, timestamp_ns in (
            (1000.0, 100),
            (2000.0, 101),
            (2000.0, 102),
            (2000.0, 103),
        )
    ]

    assert [result.position_xyz[2] for result in outputs] == pytest.approx(
        [1.1, 1.6, 1.85, 1.975]
    )
    assert [result.confidence for result in outputs] == pytest.approx(
        [0.2, 0.4, 0.6, 0.8]
    )


def test_perception_3d_confidence_uses_valid_depth_fraction_and_stays_in_range() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        depth_radius_px=1,
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    image = np.zeros((480, 640), dtype=float)
    image[239:242, 319:322] = (
        (1000.0, 1000.0, 0.0),
        (1000.0, 1000.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    detection = _estimator_detection(confidence=1.0, track_id=8)

    result = estimator.estimate(
        (detection,),
        _estimator_depth(0.0, 100, image=image),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert result.valid
    assert result.confidence == pytest.approx(4.0 / 9.0)
    assert 0.0 <= result.confidence <= 1.0


def test_perception_3d_stable_track_id_jitter_still_converges() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        ema_alpha=0.25,
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    centers = (318.0, 322.0, 319.0, 321.0)
    outputs = []
    for index, center_x in enumerate(centers):
        timestamp_ns = 100 + index
        detection = _estimator_detection(
            bbox_xyxy=(
                center_x - 1.0,
                239.0,
                center_x + 1.0,
                241.0,
            ),
            timestamp_ns=timestamp_ns,
            track_id=9,
        )
        outputs.append(
            estimator.estimate(
                (detection,),
                _estimator_depth(1000.0, timestamp_ns),
                _intrinsics(timestamp_ns=timestamp_ns),
                _base(),
                _actual_joints(),
            )[0]
        )

    assert len(estimator._tracks) == 1
    assert abs(outputs[-1].position_xyz[0]) < abs(outputs[0].position_xyz[0])
    assert max(abs(result.position_xyz[0]) for result in outputs) <= 0.0045


def test_perception_3d_untracked_inputs_never_share_persistent_ema() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        ema_alpha=0.5,
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    first = estimator.estimate(
        (_estimator_detection(timestamp_ns=100),),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]
    second = estimator.estimate(
        (_estimator_detection(timestamp_ns=101),),
        _estimator_depth(2000.0, 101),
        _intrinsics(timestamp_ns=101),
        _base(),
        _actual_joints(),
    )[0]

    assert first.position_xyz[2] == pytest.approx(1.1)
    assert second.position_xyz[2] == pytest.approx(2.1)
    assert estimator._tracks == {}


def test_perception_3d_rejects_duplicate_and_out_of_order_track_updates() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        ema_alpha=0.5,
        converge_frames=5,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    detection = _estimator_detection(track_id=10)

    first = estimator.estimate(
        (detection,),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]
    duplicate = estimator.estimate(
        (detection,),
        _estimator_depth(3000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]
    out_of_order = estimator.estimate(
        (detection,),
        _estimator_depth(4000.0, 99),
        _intrinsics(timestamp_ns=99),
        _base(),
        _actual_joints(),
    )[0]
    assert estimator.estimate(
        (),
        _estimator_depth(0.0, 101),
        _intrinsics(timestamp_ns=101),
        _base(),
        _actual_joints(),
    ) == ()
    resumed = estimator.estimate(
        (_estimator_detection(timestamp_ns=102, track_id=10),),
        _estimator_depth(2000.0, 102),
        _intrinsics(timestamp_ns=102),
        _base(),
        _actual_joints(),
    )[0]

    assert first.position_xyz[2] == pytest.approx(1.1)
    for stale in (duplicate, out_of_order):
        assert not stale.valid
        assert stale.position_xyz == (0.0, 0.0, 0.0)
        assert stale.confidence == 0.0
        assert "陈旧" in stale.failure_reason
    assert resumed.position_xyz[2] == pytest.approx(1.6)
    assert resumed.confidence > first.confidence


def test_perception_3d_reset_tracks_restarts_convergence() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=5,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    detection = _estimator_detection(track_id=11)
    first = estimator.estimate(
        (detection,),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]
    estimator.reset_tracks()
    restarted = estimator.estimate(
        (_estimator_detection(timestamp_ns=101, track_id=11),),
        _estimator_depth(2000.0, 101),
        _intrinsics(timestamp_ns=101),
        _base(),
        _actual_joints(),
    )[0]

    assert restarted.position_xyz[2] == pytest.approx(2.1)
    assert restarted.confidence == pytest.approx(first.confidence)


def test_perception_3d_same_region_same_class_targets_keep_distinct_tracks() -> None:
    """两个旧40px网格同桶目标必须保持一对一身份，不能复用第一条EMA。"""

    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        ema_alpha=0.5,
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    image = np.zeros((480, 640), dtype=float)
    image[236:245, 296:305] = 1000.0
    image[236:245, 316:325] = 2000.0
    detections = (
        _estimator_detection(
            bbox_xyxy=(299.0, 239.0, 301.0, 241.0),
            track_id=21,
        ),
        _estimator_detection(
            bbox_xyxy=(319.0, 239.0, 321.0, 241.0),
            track_id=22,
        ),
    )

    first, second = estimator.estimate(
        detections,
        _estimator_depth(0.0, 100, image=image),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )

    assert first.valid and second.valid
    assert first.position_xyz[2] == pytest.approx(1.1)
    assert second.position_xyz[2] == pytest.approx(2.1)
    assert set(estimator._tracks) == {"stable:21", "stable:22"}


def test_perception_3d_crossing_tracks_follow_stable_id_not_image_position() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        ema_alpha=0.25,
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    first_frame = (
        _estimator_detection(
            bbox_xyxy=(279.0, 239.0, 281.0, 241.0),
            track_id=31,
        ),
        _estimator_detection(
            bbox_xyxy=(359.0, 239.0, 361.0, 241.0),
            track_id=32,
        ),
    )
    first_results = estimator.estimate(
        first_frame,
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )
    # 输入顺序也反转：track 32移动到左边，track 31移动到右边。
    crossed = (
        _estimator_detection(
            bbox_xyxy=(279.0, 239.0, 281.0, 241.0),
            timestamp_ns=101,
            track_id=32,
        ),
        _estimator_detection(
            bbox_xyxy=(359.0, 239.0, 361.0, 241.0),
            timestamp_ns=101,
            track_id=31,
        ),
    )
    crossed_results = estimator.estimate(
        crossed,
        _estimator_depth(1000.0, 101),
        _intrinsics(timestamp_ns=101),
        _base(),
        _actual_joints(),
    )

    assert first_results[0].position_xyz[0] < 0.0
    assert first_results[1].position_xyz[0] > 0.0
    # EMA仍跟随ID历史：当前位于左侧的track 32仍偏右，track 31反之。
    assert crossed_results[0].position_xyz[0] > 0.0
    assert crossed_results[1].position_xyz[0] < 0.0


def test_perception_3d_track_recovers_after_short_occlusion() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        ema_alpha=0.5,
        converge_frames=4,
        max_track_age_s=1.0,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    first = estimator.estimate(
        (_estimator_detection(track_id=41),),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]
    assert estimator.estimate(
        (),
        _estimator_depth(0.0, 101),
        _intrinsics(timestamp_ns=101),
        _base(),
        _actual_joints(),
    ) == ()
    recovered = estimator.estimate(
        (_estimator_detection(timestamp_ns=102, track_id=41),),
        _estimator_depth(1200.0, 102),
        _intrinsics(timestamp_ns=102),
        _base(),
        _actual_joints(),
    )[0]

    assert first.valid and recovered.valid
    assert recovered.position_xyz[2] == pytest.approx(1.2)
    assert recovered.confidence == pytest.approx(first.confidence * 2.0)
    assert len(estimator._tracks) == 1


def test_perception_3d_large_jump_is_invalid_and_does_not_pollute_tracks() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        ema_alpha=0.5,
        converge_frames=1,
        max_position_jump_m=0.5,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    first = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(299.0, 239.0, 301.0, 241.0),
                track_id=51,
            ),
        ),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]
    jump_image = np.zeros((480, 640), dtype=float)
    jump_image[236:245, 296:305] = 4000.0
    jump_image[236:245, 336:345] = 1500.0
    jumped, neighbor = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(299.0, 239.0, 301.0, 241.0),
                timestamp_ns=101,
                track_id=51,
            ),
            _estimator_detection(
                bbox_xyxy=(339.0, 239.0, 341.0, 241.0),
                timestamp_ns=101,
                track_id=52,
            ),
        ),
        _estimator_depth(0.0, 101, image=jump_image),
        _intrinsics(timestamp_ns=101),
        _base(),
        _actual_joints(),
    )
    resumed = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(299.0, 239.0, 301.0, 241.0),
                timestamp_ns=102,
                track_id=51,
            ),
        ),
        _estimator_depth(1100.0, 102),
        _intrinsics(timestamp_ns=102),
        _base(),
        _actual_joints(),
    )[0]

    assert first.valid
    assert not jumped.valid
    assert jumped.position_xyz == (0.0, 0.0, 0.0)
    assert "跳变超限" in jumped.failure_reason
    assert neighbor.valid
    assert resumed.valid
    assert resumed.position_xyz[2] == pytest.approx(1.15)


@pytest.mark.parametrize(
    ("detection_delta_ns", "camera_delta_ns", "reason"),
    (
        (20_000_000, 0, "Detection2D/DepthFrame"),
        (0, 20_000_000, "DepthFrame/CameraInfo"),
    ),
)
def test_perception_3d_rejects_detection_depth_or_camera_time_mismatch(
    detection_delta_ns: int,
    camera_delta_ns: int,
    reason: str,
) -> None:
    timestamp_ns = 1_000_000_000
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        max_input_skew_s=0.01,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    result = estimator.estimate(
        (
            _estimator_detection(
                timestamp_ns=timestamp_ns + detection_delta_ns,
                track_id=61,
            ),
        ),
        _estimator_depth(1000.0, timestamp_ns),
        _intrinsics(timestamp_ns=timestamp_ns + camera_delta_ns),
        _base(),
        _actual_joints(),
    )[0]

    assert not result.valid
    assert result.position_xyz == (0.0, 0.0, 0.0)
    assert result.confidence == 0.0
    assert reason in result.failure_reason
    assert "时间差" in result.failure_reason


def test_perception_3d_rejects_pairwise_detection_camera_skew() -> None:
    """两端各自贴Depth窗口边界时，Detection/Camera仍不能跨越两倍窗口。"""

    depth_timestamp_ns = 1_000_000_000
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        max_input_skew_s=0.01,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    result = estimator.estimate(
        (
            _estimator_detection(
                timestamp_ns=depth_timestamp_ns + 10_000_000,
                track_id=62,
            ),
        ),
        _estimator_depth(1000.0, depth_timestamp_ns),
        _intrinsics(timestamp_ns=depth_timestamp_ns - 10_000_000),
        _base(),
        _actual_joints(),
    )[0]

    assert not result.valid
    assert "Detection2D/CameraInfo" in result.failure_reason
    assert "时间差" in result.failure_reason


def test_perception_3d_signed_negative_skew_is_bounded_and_window_is_nonzero() -> None:
    timestamp_ns = 1_000_000_000
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        max_input_skew_s=0.01,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    # Depth-Detection为负5ms，但绝对值在10ms窗口内，边界方向不能被错误忽略。
    within_window = estimator.estimate(
        (
            _estimator_detection(
                timestamp_ns=timestamp_ns + 5_000_000,
                track_id=62,
            ),
        ),
        _estimator_depth(1000.0, timestamp_ns),
        _intrinsics(timestamp_ns=timestamp_ns),
        _base(),
        _actual_joints(),
    )[0]
    outside_estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        max_input_skew_s=0.01,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    outside_window = outside_estimator.estimate(
        (
            _estimator_detection(
                timestamp_ns=timestamp_ns + 10_000_001,
                track_id=62,
            ),
        ),
        _estimator_depth(1000.0, timestamp_ns),
        _intrinsics(timestamp_ns=timestamp_ns),
        _base(),
        _actual_joints(),
    )[0]

    assert within_window.valid
    assert not outside_window.valid
    assert "时间差" in outside_window.failure_reason
    for invalid_window in (0.0, -0.01):
        with pytest.raises(ValueError, match="非零正"):
            Perception3DEstimator(
                _OffsetTransformProvider(),
                max_input_skew_s=invalid_window,
            )


@pytest.mark.parametrize(
    "negative_source",
    ("detection", "depth", "camera_info"),
)
def test_perception_3d_rejects_negative_sensor_timestamps(
    negative_source: str,
) -> None:
    detection_timestamp_ns = -1 if negative_source == "detection" else 100
    depth_timestamp_ns = -1 if negative_source == "depth" else 100
    camera_timestamp_ns = -1 if negative_source == "camera_info" else 100
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    result = estimator.estimate(
        (
            _estimator_detection(
                timestamp_ns=detection_timestamp_ns,
                track_id=64,
            ),
        ),
        _estimator_depth(1000.0, depth_timestamp_ns),
        _intrinsics(timestamp_ns=camera_timestamp_ns),
        _base(),
        _actual_joints(),
    )[0]

    assert not result.valid
    assert result.timestamp_ns >= 0
    assert result.position_xyz == (0.0, 0.0, 0.0)
    assert result.confidence == 0.0
    assert "非负整数纳秒" in result.failure_reason


@pytest.mark.parametrize(
    ("detection_frame", "camera_frame", "reason"),
    (
        ("other_camera", "camera_optical_frame", "Detection2D/DepthFrame"),
        ("camera_optical_frame", "other_camera", "DepthFrame/CameraInfo"),
    ),
)
def test_perception_3d_rejects_three_way_frame_mismatch(
    detection_frame: str,
    camera_frame: str,
    reason: str,
) -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    result = estimator.estimate(
        (
            _estimator_detection(
                frame_id=detection_frame,
                track_id=63,
            ),
        ),
        _estimator_depth(1000.0, 100),
        _intrinsics(frame_id=camera_frame, timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert not result.valid
    assert reason in result.failure_reason
    assert "frame不一致" in result.failure_reason


def test_real_perception_node_config_produces_all_three_valid_classes() -> None:
    config = _project_config()
    stabilizer, estimator = _perception_pipeline_from_config(
        config,  # type: ignore[arg-type]
        _OffsetTransformProvider(),
    )
    timestamp_ns = 1_000_000_000
    first_detections = tuple(
        _estimator_detection(
            class_id,
            (299.0 + 20.0 * index, 239.0, 301.0 + 20.0 * index, 241.0),
            timestamp_ns=timestamp_ns,
            track_id=70 + index,
        )
        for index, class_id in enumerate(OfficialYoloAdapter.CLASS_NAMES)
    )
    assert stabilizer.update(
        first_detections,
        frame_timestamp_ns=timestamp_ns,
        frame_id="camera_optical_frame",
    ) == ()
    current_timestamp_ns = timestamp_ns + 1
    current_detections = tuple(
        _estimator_detection(
            detection.class_id,
            detection.bbox_xyxy,
            timestamp_ns=current_timestamp_ns,
        )
        for detection in first_detections
    )
    stable_detections = stabilizer.update(
        current_detections,
        frame_timestamp_ns=current_timestamp_ns,
        frame_id="camera_optical_frame",
    )
    results = estimator.estimate(
        stable_detections,
        _estimator_depth(1000.0, current_timestamp_ns),
        _intrinsics(timestamp_ns=current_timestamp_ns),
        _base(),
        _actual_joints(),
    )

    assert len(stable_detections) == 3
    assert all(detection.track_id is not None for detection in stable_detections)
    assert [result.class_id for result in results] == list(
        OfficialYoloAdapter.CLASS_NAMES
    )
    assert all(result.valid for result in results)
    assert [result.position_xyz[2] for result in results] == pytest.approx(
        (1.095, 1.095, 1.095)
    )


# ---------------------------------------------------------------------------
# OfficialKDLAdapter
# ---------------------------------------------------------------------------


def test_kdl_adapter_constructor_does_not_import_official_dependencies() -> None:
    with patch("team_sorting.arm_planning.importlib.import_module") as import_module:
        adapter = OfficialKDLAdapter()
    assert adapter._solver is None
    import_module.assert_not_called()


@pytest.mark.parametrize("module_name", ["", "   ", None, True])
def test_kdl_adapter_rejects_empty_module_name(module_name: object) -> None:
    with pytest.raises(ValueError, match="module_name必须是非空字符串"):
        OfficialKDLAdapter(module_name=module_name)  # type: ignore[arg-type]


def test_kdl_self_check_searches_formal_examples_directory(tmp_path: Path) -> None:
    examples = tmp_path / "examples" / "material_sorting"
    examples.mkdir(parents=True)
    solver = _RecordingKDL()
    adapter = OfficialKDLAdapter(official_root=str(tmp_path), module_name="fake_kdl")
    with patch(
        "team_sorting.arm_planning.importlib.import_module",
        return_value=_fake_kdl_module(solver),
    ):
        adapter.self_check()

    assert adapter._searched == [
        str(tmp_path),
        str(tmp_path / "material_sorting"),
        str(examples),
    ]
    assert adapter._solver is solver
    assert solver.forward_input == (0.0,) * 13


def test_kdl_self_check_accepts_root_already_at_examples_material_sorting(
    tmp_path: Path,
) -> None:
    examples = tmp_path / "examples" / "material_sorting"
    examples.mkdir(parents=True)
    adapter = OfficialKDLAdapter(official_root=str(examples), module_name="fake_kdl")
    with patch(
        "team_sorting.arm_planning.importlib.import_module",
        return_value=_fake_kdl_module(_RecordingKDL()),
    ):
        adapter.self_check()
    assert adapter._searched[0] == str(examples)
    assert adapter._solver is not None


def test_kdl_self_check_reports_missing_module_and_search_paths(tmp_path: Path) -> None:
    adapter = OfficialKDLAdapter(official_root=str(tmp_path), module_name="missing_kdl")
    with patch(
        "team_sorting.arm_planning.importlib.import_module",
        side_effect=ImportError("fake missing"),
    ), pytest.raises(RuntimeError, match="无法导入官方 MMK2Kdl") as exc_info:
        adapter.self_check()
    expected_search_path = str(tmp_path / "examples" / "material_sorting")
    assert repr(expected_search_path) in str(exc_info.value)
    assert adapter._solver is None


def test_kdl_self_check_rejects_module_without_mmk2kdl() -> None:
    adapter = OfficialKDLAdapter(module_name="fake_kdl")
    with patch(
        "team_sorting.arm_planning.importlib.import_module",
        return_value=SimpleNamespace(),
    ), pytest.raises(RuntimeError, match="不存在 MMK2Kdl"):
        adapter.self_check()
    assert adapter._solver is None


@pytest.mark.parametrize(
    ("left_fk", "right_fk", "reason"),
    [
        (np.eye(3), np.eye(4), "不是两个 4×4 矩阵"),
        (np.eye(4), np.full((4, 4), math.nan), "包含 NaN/Inf"),
        (np.full((4, 4), math.inf), np.eye(4), "包含 NaN/Inf"),
    ],
)
def test_kdl_self_check_rejects_invalid_fk_results(
    left_fk: object, right_fk: object, reason: str
) -> None:
    adapter = OfficialKDLAdapter(module_name="fake_kdl")
    solver = _RecordingKDL(left_fk=left_fk, right_fk=right_fk)
    with patch(
        "team_sorting.arm_planning.importlib.import_module",
        return_value=_fake_kdl_module(solver),
    ), pytest.raises(RuntimeError, match=reason):
        adapter.self_check()
    assert adapter._solver is None


def test_kdl_failed_recheck_clears_old_solver_and_search_record(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    adapter = OfficialKDLAdapter(official_root=str(first_root), module_name="fake_kdl")
    solver = _RecordingKDL()
    with patch(
        "team_sorting.arm_planning.importlib.import_module",
        return_value=_fake_kdl_module(solver),
    ):
        adapter.self_check()
    assert adapter._solver is solver

    adapter.official_root = str(second_root)
    with patch(
        "team_sorting.arm_planning.importlib.import_module",
        side_effect=ImportError("second check failed"),
    ), pytest.raises(RuntimeError, match="无法导入"):
        adapter.self_check()
    assert adapter._solver is None
    assert all(str(first_root) not in path for path in adapter._searched)
    assert adapter._searched[0] == str(second_root)


def test_kdl_solve_rejects_empty_targets_and_invalid_joint_feedback() -> None:
    adapter = OfficialKDLAdapter()
    adapter._solver = _RecordingKDL()
    empty = adapter.solve_ik(_planning_joints())
    assert not empty.success and "不能同时为空" in empty.failure_reason

    invalid = adapter.solve_ik(_planning_joints(valid=False), left_target=_arm_target())
    assert not invalid.success
    assert invalid.target_slide == pytest.approx(0.1)
    assert "JointState无效" in invalid.failure_reason


@pytest.mark.parametrize(
    ("position", "valid"),
    [
        ((), False),
        ((0.1,) * 16, True),
        (None, False),
        (("0.1", *((0.0,) * 16)), True),
        ((True, *((0.0,) * 16)), False),
        ((math.nan, *((0.0,) * 16)), True),
        ((math.inf, *((0.0,) * 16)), False),
        ((-math.inf, *((0.0,) * 16)), True),
    ],
)
def test_kdl_invalid_actual_position_always_returns_failed_result(
    position: object, valid: bool
) -> None:
    solver = _RecordingKDL(solutions=((0.1, 1, 2, 3, 4, 5, 6),))
    adapter = OfficialKDLAdapter()
    adapter._solver = solver
    damaged_state = SimpleNamespace(
        position=position,
        valid=valid,
        failure_reason="损坏的JointState",
    )

    result = adapter.solve_ik(damaged_state, left_target=_arm_target())

    assert not result.success
    assert result.left_joint_target is None
    assert result.right_joint_target is None
    assert "实际关节位置无效" in result.failure_reason
    assert solver.inverse_call is None


def test_kdl_legal_actual_position_but_invalid_state_returns_failed_result() -> None:
    adapter = OfficialKDLAdapter()
    adapter._solver = _RecordingKDL(solutions=((0.1, 1, 2, 3, 4, 5, 6),))
    state = SimpleNamespace(
        position=_planning_joints().position,
        valid=False,
        failure_reason="JointState尚未可信",
    )
    result = adapter.solve_ik(state, left_target=_arm_target())
    assert not result.success
    assert result.target_slide == pytest.approx(0.1)
    assert result.failure_reason == "JointState尚未可信"


def test_kdl_legal_actual_position_and_valid_state_reaches_solver() -> None:
    solver = _RecordingKDL(solutions=((0.1, 20, 21, 22, 23, 24, 25),))
    adapter = OfficialKDLAdapter()
    adapter._solver = solver
    state = SimpleNamespace(
        position=_planning_joints().position,
        valid=True,
        failure_reason="",
    )
    result = adapter.solve_ik(state, left_target=_arm_target())
    assert result.success
    assert result.left_joint_target == pytest.approx((20, 21, 22, 23, 24, 25))
    assert solver.inverse_call is not None


@pytest.mark.parametrize("frame_id", ["base_link", "base_footprint", "world", "odom", ""])
def test_kdl_solve_rejects_unconverted_or_unconfirmed_frames(frame_id: str) -> None:
    adapter = OfficialKDLAdapter()
    adapter._solver = _RecordingKDL(solutions=((0.1, 1, 2, 3, 4, 5, 6),))
    result = adapter.solve_ik(_planning_joints(), left_target=_arm_target(frame_id))
    assert not result.success
    assert "footprint" in result.failure_reason


def test_kdl_left_arm_uses_seven_dimensional_reference_and_actual_slide() -> None:
    solver = _RecordingKDL(solutions=((0.1, 20, 21, 22, 23, 24, 25),))
    adapter = OfficialKDLAdapter()
    adapter._solver = solver
    result = adapter.solve_ik(_planning_joints(), left_target=_arm_target())

    assert result.success
    assert result.target_slide == pytest.approx(0.1)
    assert result.left_joint_target == pytest.approx((20, 21, 22, 23, 24, 25))
    assert result.right_joint_target is None
    assert tuple(solver.inverse_call["ref_pos"]) == pytest.approx((0.1, 3, 4, 5, 6, 7, 8))
    assert solver.inverse_call["target_height"] == pytest.approx(0.1)
    assert solver.inverse_call["T_left"].shape == (4, 4)
    assert solver.inverse_call["T_right"] is None


def test_kdl_right_arm_uses_correct_seven_dimensional_slice() -> None:
    solver = _RecordingKDL(solutions=((0.2, 30, 31, 32, 33, 34, 35),))
    adapter = OfficialKDLAdapter()
    adapter._solver = solver
    result = adapter.solve_ik(
        _planning_joints(), right_target=_arm_target(), target_slide=0.2
    )

    assert result.success
    assert result.left_joint_target is None
    assert result.right_joint_target == pytest.approx((30, 31, 32, 33, 34, 35))
    assert tuple(solver.inverse_call["ref_pos"]) == pytest.approx((0.1, 10, 11, 12, 13, 14, 15))
    assert solver.inverse_call["target_height"] == pytest.approx(0.2)
    assert solver.inverse_call["T_left"] is None
    assert solver.inverse_call["T_right"].shape == (4, 4)


def test_kdl_both_arms_use_thirteen_dimensions_without_head_or_grippers() -> None:
    solution = (0.3, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35)
    solver = _RecordingKDL(solutions=(solution,))
    adapter = OfficialKDLAdapter()
    adapter._solver = solver
    result = adapter.solve_ik(
        _planning_joints(),
        left_target=_arm_target(),
        right_target=_arm_target(),
        target_slide=0.3,
    )

    assert result.success
    assert result.left_joint_target == pytest.approx(solution[1:7])
    assert result.right_joint_target == pytest.approx(solution[7:13])
    assert tuple(solver.inverse_call["ref_pos"]) == pytest.approx(
        (0.1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15)
    )
    assert 1.0 not in tuple(solver.inverse_call["ref_pos"])[1:]
    assert 2.0 not in tuple(solver.inverse_call["ref_pos"])[1:]
    assert 9.0 not in tuple(solver.inverse_call["ref_pos"])[1:]
    assert 16.0 not in tuple(solver.inverse_call["ref_pos"])[1:]


@pytest.mark.parametrize("target_slide", [True, False, "0.1", math.nan, math.inf, -math.inf])
def test_kdl_rejects_invalid_target_slide(target_slide: object) -> None:
    solver = _RecordingKDL(solutions=((0.1, 1, 2, 3, 4, 5, 6),))
    adapter = OfficialKDLAdapter()
    adapter._solver = solver
    result = adapter.solve_ik(
        _planning_joints(),
        left_target=_arm_target(),
        target_slide=target_slide,  # type: ignore[arg-type]
    )
    assert not result.success
    assert "target_slide" in result.failure_reason
    assert solver.inverse_call is None


def test_kdl_rejects_target_slide_outside_official_solver_limits() -> None:
    solver = _RecordingKDL(solutions=((1.0, 1, 2, 3, 4, 5, 6),))
    adapter = OfficialKDLAdapter()
    adapter._solver = solver
    result = adapter.solve_ik(
        _planning_joints(), left_target=_arm_target(), target_slide=1.0
    )
    assert not result.success
    assert "spine.joint_limits" in result.failure_reason
    assert solver.inverse_call is None


@pytest.mark.parametrize(
    ("solutions", "reason"),
    [
        (None, "未找到合法关节解"),
        ((), "空关节解"),
        (42, "不可迭代"),
        (((0.1, 1, 2, 3, 4, 5),), "长度不是 7"),
        ((((0.1, 1, 2, 3, 4, 5, 6),),), "必须是一维向量"),
        (((0.1, 1, 2, math.nan, 4, 5, 6),), "NaN或Inf"),
        (((0.1, 1, 2, math.inf, 4, 5, 6),), "NaN或Inf"),
        (((0.1, 1, 2, True, 4, 5, 6),), "不能使用bool"),
        (((0.2, 1, 2, 3, 4, 5, 6),), "与target_slide"),
    ],
)
def test_kdl_rejects_malformed_or_inconsistent_solutions(
    solutions: object, reason: str
) -> None:
    adapter = OfficialKDLAdapter()
    adapter._solver = _RecordingKDL(solutions=solutions)
    result = adapter.solve_ik(_planning_joints(), left_target=_arm_target())
    assert not result.success
    assert reason in result.failure_reason
    assert result.left_joint_target is None
    assert result.right_joint_target is None


# ---------------------------------------------------------------------------
# Pose3D转矩阵
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position",
    [
        (0.1, 0.2),
        (0.1, 0.2, 0.3, 0.4),
        (True, 0.2, 0.3),
        ("0.1", 0.2, 0.3),
        (math.nan, 0.2, 0.3),
        (math.inf, 0.2, 0.3),
    ],
)
def test_pose_to_matrix_rejects_invalid_position(position: object) -> None:
    pose = Pose3D(position, (0.0, 0.0, 0.0, 1.0), "footprint")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="position_xyz"):
        _pose_to_matrix(pose, np)


@pytest.mark.parametrize(
    "orientation",
    [
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0, 2.0),
        (False, 0.0, 0.0, 1.0),
        ("0", 0.0, 0.0, 1.0),
        (math.nan, 0.0, 0.0, 1.0),
        (math.inf, 0.0, 0.0, 1.0),
    ],
)
def test_pose_to_matrix_rejects_invalid_orientation(orientation: object) -> None:
    pose = Pose3D((0.1, 0.2, 0.3), orientation, "footprint")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="orientation_xyzw"):
        _pose_to_matrix(pose, np)


def test_pose_to_matrix_rejects_zero_norm_quaternion() -> None:
    pose = Pose3D((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 0.0), "footprint")
    with pytest.raises(ValueError, match="范数为零"):
        _pose_to_matrix(pose, np)


def test_pose_to_matrix_normalizes_identity_quaternion_and_translation() -> None:
    pose = Pose3D((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 2.0), "footprint")
    matrix = _pose_to_matrix(pose, np)
    assert matrix.shape == (4, 4)
    assert np.isfinite(matrix).all()
    assert matrix[:3, :3] == pytest.approx(np.eye(3))
    assert matrix[:3, 3] == pytest.approx((0.1, -0.2, 0.3))
    assert matrix[3, :] == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_pose_to_matrix_known_ninety_degree_rotation() -> None:
    half = math.sqrt(0.5)
    pose = Pose3D((0.0, 0.0, 0.0), (0.0, 0.0, half, half), "footprint")
    matrix = _pose_to_matrix(pose, np)
    np.testing.assert_allclose(
        matrix[:3, :3],
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# ArmPlanner临时未实现约束
# ---------------------------------------------------------------------------

# 当前作用：防止抓放Pose和轨迹尚未实现时返回零路点、空轨迹或伪成功。
# TODO(arm-planner-implementation)：plan_grasp/plan_place真正实现时，必须在同一提交中
# 替换为抓取Pose、多阶段IK、17维路点、不可达失败和放置上下文测试；生产方法仍未实现
# 时不得删除本组。
def test_arm_planner_methods_remain_explicitly_unimplemented() -> None:
    fake_adapter = _InjectedIKAdapter()
    planner = ArmPlanner(fake_adapter)  # type: ignore[arg-type]
    assert planner._ik_adapter is fake_adapter
    assert not fake_adapter.self_check_called
    joints = _planning_joints()
    target = ObjectEstimate3D("pink", (1.0, 2.0, 0.5), 0.9, "odom", 100)
    task = TaskSpec(
        task_id=1,
        instruction="move pink box",
        target_kind="box",
        target_body="box_body",
        target_color="pink",
        place_type="point",
        place_world_xyz=(2.0, 3.0, 0.5),
        place_radius=0.1,
    )
    with pytest.raises(NotImplementedError, match="机械臂1负责人"):
        planner.plan_grasp(target, joints)
    with pytest.raises(NotImplementedError, match="机械臂1负责人"):
        planner.plan_place(task, joints)


def test_arm_planner_rejects_dependency_without_solve_ik() -> None:
    with pytest.raises(TypeError, match="solve_ik"):
        ArmPlanner(object())  # type: ignore[arg-type]
