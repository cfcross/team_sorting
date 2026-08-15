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
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

import team_sorting.perception_3d as perception_3d_module
import team_sorting.arm_planning as arm_planning_module
from team_sorting.arm_planning import ArmPlanner, OfficialKDLAdapter, _pose_to_matrix
from team_sorting.interfaces import (
    ArmPlanningConfig,
    ArmMotionPhase,
    BaseState,
    CameraIntrinsics,
    Detection2D,
    DepthFrame,
    GraspContext,
    GraspTarget,
    GlobalPhase,
    IKResult,
    NavGoal,
    ObjectEstimate3D,
    PlaceTarget,
    Pose3D,
    RobotJointState,
    RigidTransform3D,
    SlotType,
    TaskSpec,
)
from team_sorting.navigation import (
    Bounds3D,
    NavigationConfig,
    NavigationController,
    _StaticAABB,
    _StaticObstacle,
    _StaticRouteConfig,
    _StaticRouteNavigator,
    _inflate_aabb,
    _plan_static_aabb_route,
    _point_intersects_aabb,
    _segment_intersects_aabb,
    _segment_intersects_any_aabb,
    classify_slot_type,
    distance_xy,
    wrap_to_pi,
)
from team_sorting.perception_2d import OfficialYoloAdapter
from team_sorting.perception_3d import (
    CameraTransformProvider,
    Perception3DEstimator,
    VisualObservationVerifier,
    _HeadCameraPose,
    median_depth_m,
    project_pixel_to_camera,
)
from team_sorting.ros_nodes import (
    _arm_planning_config_from_config,
    _base_state_from_odom,
    _estimates_from_vision,
    _estimates_to_vision,
    _navigation_config_from_config,
    _navigation_safety_enabled_from_config,
    _perception_pipeline_from_config,
    _source_slot_config_from_config,
    _select_target_estimate,
    _validate_vision_schema,
)


def _odom_message(
    *,
    frame_id: str = "/odom",
    sec: object = 12,
    nanosec: object = 345,
    position: tuple[object, object, object] = (1.25, -2.5, 0.75),
    quaternion: tuple[object, object, object, object] = (0.0, 0.0, 0.0, 1.0),
    linear: tuple[object, object, object] = (-0.4, 0.3, 0.2),
    angular: tuple[object, object, object] = (0.1, -0.2, 0.3),
) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id
        ),
        child_frame_id="base_footprint",
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(
                    x=position[0], y=position[1], z=position[2]
                ),
                orientation=SimpleNamespace(
                    x=quaternion[0],
                    y=quaternion[1],
                    z=quaternion[2],
                    w=quaternion[3],
                ),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=linear[0], y=linear[1], z=linear[2]),
                angular=SimpleNamespace(
                    x=angular[0], y=angular[1], z=angular[2]
                ),
            )
        ),
    )


def _visual_verifier() -> VisualObservationVerifier:
    return VisualObservationVerifier(
        minimum_lift_delta_m=0.03,
        max_horizontal_drift_m=0.02,
        max_observation_gap_s=0.5,
        minimum_observation_confidence=0.5,
        required_frame_id="odom",
        min_stationary_observations=3,
        max_stationary_spread_m=0.01,
    )


def _object_observation(
    position_xyz: tuple[float, float, float],
    timestamp_ns: int,
    *,
    object_id: str = "pink:track-7",
    valid: bool = True,
    confidence: float = 0.8,
    frame_id: str = "odom",
) -> ObjectEstimate3D:
    return ObjectEstimate3D(
        class_id="pink",
        position_xyz=position_xyz,
        confidence=confidence,
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        valid=valid,
        failure_reason="深度不可用" if not valid else "",
        object_id=object_id,
    )


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
    frame_id: str = "odom",
) -> BaseState:
    return BaseState(
        position_xyz=(1.0, 2.0, 3.0),
        orientation_xyzw=orientation_xyzw,
        yaw=0.0,
        linear_velocity_xyz=(0.0, 0.0, 0.0),
        angular_velocity_xyz=(0.0, 0.0, 0.0),
        frame_id=frame_id,
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


def test_odom_adapter_maps_all_base_state_fields() -> None:
    state = _base_state_from_odom(
        _odom_message(quaternion=(0.0, 0.0, 0.0, 2.0))
    )
    assert state.timestamp_ns == 12_000_000_345
    assert state.frame_id == "odom"
    assert state.position_xyz == pytest.approx((1.25, -2.5, 0.75))
    assert state.orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert state.linear_velocity_xyz == pytest.approx((-0.4, 0.3, 0.2))
    assert state.angular_velocity_xyz == pytest.approx((0.1, -0.2, 0.3))
    assert state.yaw == pytest.approx(0.0)
    assert state.valid is True
    assert state.failure_reason == ""


@pytest.mark.parametrize(
    "yaw",
    (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi - 1.0e-7, -math.pi + 1.0e-7),
)
def test_odom_adapter_converts_unit_quaternion_to_expected_yaw(yaw: float) -> None:
    quaternion = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    state = _base_state_from_odom(_odom_message(quaternion=quaternion))
    assert state.yaw == pytest.approx(yaw, abs=1.0e-9)


@pytest.mark.parametrize("frame_id", ("/odom", "odom"))
def test_odom_adapter_normalizes_supported_odom_frame(frame_id: str) -> None:
    assert _base_state_from_odom(_odom_message(frame_id=frame_id)).frame_id == "odom"


def test_odom_adapter_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="frame_id规范化后不能为空"):
        _base_state_from_odom(_odom_message(frame_id=""))


def test_non_odom_frame_is_preserved_then_navigation_fails_closed() -> None:
    state = _base_state_from_odom(_odom_message(frame_id="map"))
    assert state.frame_id == "map"
    command, status = NavigationController().update(
        state,
        _nav_goal(1.0, 0.0, 0.0),
        state.timestamp_ns,
    )
    assert not command.valid
    assert (command.v, command.w) == (0.0, 0.0)
    assert status.state == "failed"
    assert "frame" in status.failure_reason


@pytest.mark.parametrize(
    ("kwargs", "failure"),
    (
        ({"position": (math.nan, 0.0, 0.0)}, "位置包含 NaN/Inf"),
        ({"position": (math.inf, 0.0, 0.0)}, "位置包含 NaN/Inf"),
        ({"quaternion": (math.nan, 0.0, 0.0, 1.0)}, "四元数包含 NaN/Inf"),
        ({"quaternion": (0.0, 0.0, math.inf, 1.0)}, "四元数包含 NaN/Inf"),
        ({"quaternion": (0.0, 0.0, 0.0, 0.0)}, "四元数范数为零"),
        ({"sec": -1}, "时间戳范围无效"),
        ({"nanosec": 1_000_000_000}, "时间戳范围无效"),
        ({"sec": True}, "时间戳格式无效"),
    ),
)
def test_odom_adapter_rejects_nonfinite_quaternion_and_invalid_time(
    kwargs: dict[str, object], failure: str
) -> None:
    with pytest.raises(ValueError, match=failure):
        _base_state_from_odom(_odom_message(**kwargs))  # type: ignore[arg-type]


def test_odom_adapter_rejects_missing_required_fields() -> None:
    message = _odom_message()
    del message.pose.pose.position
    with pytest.raises(ValueError, match="Odom 字段格式无效"):
        _base_state_from_odom(message)


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


def test_static_aabb_is_private_frozen_and_normalizes_numeric_values() -> None:
    bounds = _StaticAABB(-2, -1.5, 3, 4.5)

    assert (bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y) == (
        -2.0,
        -1.5,
        3.0,
        4.5,
    )
    assert all(
        type(value) is float
        for value in (bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y)
    )
    with pytest.raises(AttributeError):
        bounds.min_x = -3.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "failure"),
    [
        ((True, 0.0, 1.0, 1.0), "min_x"),
        ((np.float64(0.0), 0.0, 1.0, 1.0), "min_x"),
        ((10**400, 0.0, 1.0, 1.0), "min_x"),
        ((0.0, False, 1.0, 1.0), "min_y"),
        ((0.0, 0.0, "1.0", 1.0), "max_x"),
        ((0.0, 0.0, 1.0, None), "max_y"),
        ((math.nan, 0.0, 1.0, 1.0), "min_x"),
        ((0.0, -math.inf, 1.0, 1.0), "min_y"),
        ((0.0, 0.0, math.inf, 1.0), "max_x"),
        ((0.0, 0.0, 1.0, math.nan), "max_y"),
        ((1.0, 0.0, 1.0, 1.0), "min_x必须小于max_x"),
        ((2.0, 0.0, 1.0, 1.0), "min_x必须小于max_x"),
        ((0.0, 1.0, 1.0, 1.0), "min_y必须小于max_y"),
        ((0.0, 2.0, 1.0, 1.0), "min_y必须小于max_y"),
    ],
)
def test_static_aabb_rejects_invalid_or_degenerate_bounds(
    values: tuple[object, object, object, object], failure: str
) -> None:
    with pytest.raises(ValueError, match=failure):
        _StaticAABB(*values)  # type: ignore[arg-type]


def test_inflate_aabb_expands_every_closed_boundary_without_hidden_constant() -> None:
    original = _StaticAABB(-2.875, 0.37, -2.465, 1.186)

    unchanged = _inflate_aabb(original, 0)
    inflated = _inflate_aabb(original, 0.5154080786132004)

    assert unchanged == original
    assert unchanged is not original
    assert inflated == _StaticAABB(
        -3.3904080786132003,
        -0.14540807861320038,
        -1.9495919213867996,
        1.7014080786132002,
    )
    assert original == _StaticAABB(-2.875, 0.37, -2.465, 1.186)


@pytest.mark.parametrize(
    "inflation",
    [
        True,
        False,
        np.float64(0.05),
        "0.05",
        None,
        10**400,
        -1.0e-12,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_inflate_aabb_rejects_invalid_inflation(inflation: object) -> None:
    with pytest.raises(ValueError, match="inflation_m必须是非负有限数"):
        _inflate_aabb(_StaticAABB(0.0, 0.0, 1.0, 1.0), inflation)  # type: ignore[arg-type]


def test_inflate_aabb_rejects_wrong_boundary_type_and_numeric_overflow() -> None:
    with pytest.raises(ValueError, match="bounds必须是_StaticAABB"):
        _inflate_aabb((0.0, 0.0, 1.0, 1.0), 0.1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="当前浮点精度"):
        _inflate_aabb(_StaticAABB(-1.0e308, -1.0, 1.0e308, 1.0), 1.0e308)


@pytest.mark.parametrize(
    ("bounds", "inflation"),
    [
        (_StaticAABB(-1.0, -1.0, 1.0, 1.0), 5e-324),
        (_StaticAABB(-1.0e308, -1.0, 1.0, 1.0), 1.0),
    ],
)
def test_inflate_aabb_rejects_unrepresentable_boundary_expansion(
    bounds: _StaticAABB, inflation: float
) -> None:
    with pytest.raises(ValueError, match="当前浮点精度"):
        _inflate_aabb(bounds, inflation)


@pytest.mark.parametrize(
    "point",
    [
        (0.5, 0.5),
        (0.0, 0.5),
        (1.0, 0.5),
        (0.5, 0.0),
        (0.5, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
    ],
)
def test_point_intersects_aabb_uses_closed_boundaries(
    point: tuple[float, float]
) -> None:
    assert _point_intersects_aabb(point, _StaticAABB(0.0, 0.0, 1.0, 1.0))


@pytest.mark.parametrize(
    "point",
    [
        (-5e-324, 0.5),
        (1.0000000000000002, 0.5),
        (0.5, -5e-324),
        (0.5, 1.0000000000000002),
    ],
)
def test_point_intersects_aabb_rejects_points_immediately_outside(
    point: tuple[float, float]
) -> None:
    assert not _point_intersects_aabb(point, _StaticAABB(0.0, 0.0, 1.0, 1.0))


@pytest.mark.parametrize(
    "point",
    [
        None,
        1.0,
        "01",
        (),
        (0.0,),
        (0.0, 0.0, 0.0),
        (True, 0.0),
        (np.float64(0.0), 0.0),
        ("0.0", 0.0),
        (10**400, 0.0),
        (math.nan, 0.0),
        (0.0, math.inf),
    ],
)
def test_point_intersects_aabb_rejects_invalid_exact_xy(point: object) -> None:
    with pytest.raises(ValueError, match="point_xy"):
        _point_intersects_aabb(point, _StaticAABB(0.0, 0.0, 1.0, 1.0))  # type: ignore[arg-type]


def test_point_intersects_aabb_requires_private_bounds() -> None:
    with pytest.raises(ValueError, match="bounds必须是_StaticAABB"):
        _point_intersects_aabb((0.5, 0.5), (0.0, 0.0, 1.0, 1.0))  # type: ignore[arg-type]


def test_point_intersects_aabb_accepts_builtin_list_without_mutating_it() -> None:
    point = [0.25, 0.75]

    assert _point_intersects_aabb(point, _StaticAABB(0.0, 0.0, 1.0, 1.0))
    assert point == [0.25, 0.75]


def test_point_intersects_aabb_rejects_non_builtin_xy_containers_stably() -> None:
    class _TupleSubclass(tuple):
        pass

    class _ListSubclass(list):
        pass

    class _CustomSequence(Sequence[float]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> float:
            return (0.25, 0.75)[index]

    class _CustomMapping(Mapping[int, float]):
        def __iter__(self):
            return iter((0, 1))

        def __len__(self) -> int:
            return 2

        def __getitem__(self, key: int) -> float:
            return (0.25, 0.75)[key]

    class _BadLength:
        def __len__(self) -> int:
            raise RuntimeError("不得调用自定义len")

    class _BadIndex:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, _index: int) -> float:
            raise RuntimeError("不得调用自定义getitem")

    bounds = _StaticAABB(0.0, 0.0, 1.0, 1.0)
    for point in (
        b"01",
        {0: 0.25, 1: 0.75},
        (value for value in (0.25, 0.75)),
        {0.25, 0.75},
        _CustomSequence(),
        _CustomMapping(),
        _TupleSubclass((0.25, 0.75)),
        _ListSubclass((0.25, 0.75)),
        _BadLength(),
        _BadIndex(),
    ):
        with pytest.raises(ValueError, match="point_xy"):
            _point_intersects_aabb(point, bounds)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ((-1.0, 0.5), (2.0, 0.5)),
        ((0.5, -1.0), (0.5, 2.0)),
        ((-1.0, -1.0), (2.0, 2.0)),
        ((0.25, 0.25), (0.75, 0.75)),
        ((-1.0, 0.0), (2.0, 0.0)),
        ((-1.0, -1.0), (0.0, 0.0)),
        ((0.5, 0.5), (0.5, 0.5)),
    ],
)
def test_segment_intersects_aabb_detects_crossing_inside_tangent_and_degenerate(
    start: tuple[float, float], end: tuple[float, float]
) -> None:
    bounds = _StaticAABB(0.0, 0.0, 1.0, 1.0)
    assert _segment_intersects_aabb(start, end, bounds)
    assert _segment_intersects_aabb(end, start, bounds)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ((-1.0, 1.0000000000000002), (2.0, 1.0000000000000002)),
        ((-1.0, 2.0), (2.0, 2.0)),
        ((-1.0, -1.0), (-0.5, -0.5)),
        ((2.0, 0.0), (2.0, 1.0)),
        ((-1.0, -1.0), (-1.0, -1.0)),
    ],
)
def test_segment_intersects_aabb_rejects_parallel_near_miss_and_degenerate_outside(
    start: tuple[float, float], end: tuple[float, float]
) -> None:
    bounds = _StaticAABB(0.0, 0.0, 1.0, 1.0)
    assert not _segment_intersects_aabb(start, end, bounds)
    assert not _segment_intersects_aabb(end, start, bounds)


def test_segment_intersects_aabb_validates_both_points_and_finite_arithmetic() -> None:
    bounds = _StaticAABB(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="start_xy"):
        _segment_intersects_aabb((True, 0.0), (1.0, 1.0), bounds)
    with pytest.raises(ValueError, match="end_xy"):
        _segment_intersects_aabb((0.0, 0.0), (1.0, math.inf), bounds)
    with pytest.raises(ValueError, match="当前浮点精度"):
        _segment_intersects_aabb((-1.0e308, 0.5), (1.0e308, 0.5), bounds)


def test_segment_intersects_any_aabb_handles_empty_order_and_invalid_members() -> None:
    segment = ((-1.0, 0.5), (2.0, 0.5))
    miss = _StaticAABB(4.0, 4.0, 5.0, 5.0)
    hit = _StaticAABB(0.0, 0.0, 1.0, 1.0)

    assert not _segment_intersects_any_aabb(*segment, ())
    assert _segment_intersects_any_aabb(*segment, (miss, hit))
    assert _segment_intersects_any_aabb(*segment, (hit, miss))
    with pytest.raises(ValueError, match="obstacles必须是_StaticAABB序列"):
        _segment_intersects_any_aabb(*segment, (miss, object()))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="obstacles必须是_StaticAABB序列"):
        _segment_intersects_any_aabb(*segment, (object(), hit))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="obstacles必须是_StaticAABB序列"):
        _segment_intersects_any_aabb(*segment, "not-obstacles")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start_xy"):
        _segment_intersects_any_aabb((True, 0.5), segment[1], ())


def test_segment_intersects_any_aabb_rejects_invalid_member_after_hit() -> None:
    segment = ((-1.0, 0.5), (2.0, 0.5))
    hit = _StaticAABB(0.0, 0.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="obstacles必须是_StaticAABB序列"):
        _segment_intersects_any_aabb(*segment, (hit, object()))  # type: ignore[arg-type]


def test_plan_static_aabb_route_returns_goal_for_empty_direct_route() -> None:
    assert _plan_static_aabb_route((0.0, 0.0), (1.0, 1.0), (), 0.1, 0) == (
        (1.0, 1.0),
    )


def test_plan_static_aabb_route_keeps_safe_direct_and_equal_goal_routes_minimal() -> None:
    obstacle = _StaticAABB(3.0, 3.0, 4.0, 4.0)

    assert _plan_static_aabb_route((0.0, 0.0), (1.0, 1.0), (obstacle,), 0.1, 3) == (
        (1.0, 1.0),
    )
    assert _plan_static_aabb_route((0.5, 0.5), (0.5, 0.5), (), 0.1, 0) == (
        (0.5, 0.5),
    )
    occupied = (_StaticAABB(0.0, 0.0, 1.0, 1.0),)
    assert _plan_static_aabb_route((0.5, 0.5), (0.5, 0.5), occupied, 0.1, 2) is None


def test_plan_static_aabb_route_selects_safe_lexicographic_shortest_detour() -> None:
    start = (-1.0, 0.5)
    goal = (2.0, 0.5)
    obstacles = (_StaticAABB(0.0, 0.0, 1.0, 1.0),)

    route = _plan_static_aabb_route(start, goal, obstacles, 0.1, 2)

    assert route == ((-0.1, -0.1), (1.1, -0.1), goal)
    assert route[-1] == goal
    assert len(route[:-1]) == 2
    assert all(
        not _point_intersects_aabb(waypoint, obstacles[0])
        for waypoint in route[:-1]
    )
    previous = start
    for point in route:
        assert not _segment_intersects_any_aabb(previous, point, obstacles)
        previous = point


def test_plan_static_aabb_route_is_repeatable_and_obstacle_order_independent() -> None:
    start = (-2.0, 0.0)
    goal = (4.0, 0.0)
    obstacles = (
        _StaticAABB(0.0, -1.0, 1.0, 1.0),
        _StaticAABB(2.0, -2.0, 3.0, 0.5),
    )

    expected = _plan_static_aabb_route(start, goal, obstacles, 0.1, 2)

    assert expected is not None
    assert len(expected[:-1]) == 2
    assert _plan_static_aabb_route(start, goal, obstacles, 0.1, 2) == expected
    assert _plan_static_aabb_route(start, goal, tuple(reversed(obstacles)), 0.1, 2) == expected
    assert _plan_static_aabb_route(start, goal, (*obstacles, obstacles[0]), 0.1, 2) == expected


def test_plan_static_aabb_route_large_valid_graph_is_bounded_and_deterministic() -> None:
    start = (-2.0, 0.0)
    goal = (2.0, 0.0)
    obstacles = (_StaticAABB(0.0, -0.5, 0.5, 0.5),) + tuple(
        _StaticAABB(10.0 * index, 10.0, 10.0 * index + 1.0, 11.0)
        for index in range(1, 8)
    )

    started = time.monotonic()
    route = _plan_static_aabb_route(start, goal, obstacles, 0.1, 24)
    elapsed_s = time.monotonic() - started

    assert elapsed_s < 2.0
    assert route is not None
    assert route[-1] == goal
    assert len(route[:-1]) <= 24
    assert len(route) == len(tuple(dict.fromkeys(route)))
    previous = start
    for point in route:
        assert not _segment_intersects_any_aabb(previous, point, obstacles)
        previous = point
    assert _plan_static_aabb_route(start, goal, obstacles, 0.1, 24) == route
    assert _plan_static_aabb_route(start, goal, tuple(reversed(obstacles)), 0.1, 24) == route


def test_plan_static_aabb_route_caps_mathematical_limit_at_candidate_count() -> None:
    start = (-2.0, 0.0)
    goal = (2.0, 0.0)
    obstacles = (_StaticAABB(0.0, -0.5, 0.5, 0.5),) + tuple(
        _StaticAABB(10.0 * index, 10.0, 10.0 * index + 1.0, 11.0)
        for index in range(1, 8)
    )
    candidate_count = 4 * len(obstacles)

    assert _plan_static_aabb_route(start, goal, obstacles, 0.1, 1000) == (
        _plan_static_aabb_route(start, goal, obstacles, 0.1, candidate_count)
    )


def test_plan_static_aabb_route_enforces_intermediate_waypoint_limit() -> None:
    start = (-1.0, 0.5)
    goal = (2.0, 0.5)
    obstacles = (_StaticAABB(0.0, 0.0, 1.0, 1.0),)

    assert _plan_static_aabb_route(start, goal, obstacles, 0.1, 0) is None
    assert _plan_static_aabb_route(start, goal, obstacles, 0.1, 1) is None
    assert _plan_static_aabb_route(start, goal, obstacles, 0.1, 2) is not None
    assert _plan_static_aabb_route(start, (-0.5, 0.5), obstacles, 0.1, 0) == (
        (-0.5, 0.5),
    )


@pytest.mark.parametrize("limit", [True, False, -1, 1.0, "1", None, np.int64(1)])
def test_plan_static_aabb_route_rejects_invalid_waypoint_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="max_intermediate_waypoints"):
        _plan_static_aabb_route((0.0, 0.0), (1.0, 1.0), (), 0.1, limit)


@pytest.mark.parametrize(
    "margin",
    [0, -0.1, True, False, math.nan, math.inf, -math.inf, "0.1", None, np.float64(0.1)],
)
def test_plan_static_aabb_route_rejects_invalid_margin(margin: object) -> None:
    with pytest.raises(ValueError, match="waypoint_margin_m"):
        _plan_static_aabb_route((0.0, 0.0), (1.0, 1.0), (), margin, 0)


@pytest.mark.parametrize(
    ("bounds", "margin"),
    [
        (_StaticAABB(-1.0, -1.0, 1.0, 1.0), 5e-324),
        (_StaticAABB(-1.0e308, -1.0, 1.0, 1.0), 1.0),
        (_StaticAABB(-1.0e308, -1.0, 1.0e308, 1.0), 1.0e308),
    ],
)
def test_plan_static_aabb_route_rejects_unrepresentable_candidates(
    bounds: _StaticAABB, margin: float
) -> None:
    with pytest.raises(ValueError, match="当前浮点精度"):
        _plan_static_aabb_route((0.0, 10.0), (1.0, 10.0), (bounds,), margin, 0)


def test_plan_static_aabb_route_accepts_representable_large_candidates() -> None:
    obstacle = _StaticAABB(1.0e12, 100.0, 1.0e12 + 0.5, 101.0)

    assert _plan_static_aabb_route((0.0, 0.0), (1.0, 0.0), (obstacle,), 0.5, 0) == (
        (1.0, 0.0),
    )


@pytest.mark.parametrize(
    "obstacles",
    [
        [],
        set(),
        {},
        (item for item in ()),
        "",
        tuple(),
    ],
)
def test_plan_static_aabb_route_requires_builtin_obstacle_tuple(
    obstacles: object,
) -> None:
    if type(obstacles) is tuple:
        class _TupleSubclass(tuple):
            pass

        obstacles = _TupleSubclass(obstacles)
    with pytest.raises(ValueError, match="obstacles"):
        _plan_static_aabb_route((0.0, 0.0), (1.0, 1.0), obstacles, 0.1, 0)


def test_plan_static_aabb_route_prevalidates_every_obstacle_and_input_in_order() -> None:
    hit = _StaticAABB(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="start_xy"):
        _plan_static_aabb_route((True, 0.0), (math.nan, 0.0), (object(),), 0.0, -1)
    with pytest.raises(ValueError, match="goal_xy"):
        _plan_static_aabb_route((0.0, 0.0), (math.nan, 0.0), (object(),), 0.0, -1)
    with pytest.raises(ValueError, match="obstacles成员"):
        _plan_static_aabb_route((-1.0, 0.5), (2.0, 0.5), (hit, object()), 0.1, 2)
    with pytest.raises(ValueError, match="obstacles成员"):
        _plan_static_aabb_route((-1.0, 0.5), (2.0, 0.5), (object(), hit), 0.1, 2)
    with pytest.raises(ValueError, match="waypoint_margin_m"):
        _plan_static_aabb_route((0.0, 0.0), (1.0, 1.0), (), 0.0, -1)
    with pytest.raises(ValueError, match="max_intermediate_waypoints"):
        _plan_static_aabb_route((0.0, 0.0), (1.0, 1.0), (), 0.1, -1)


@pytest.mark.parametrize(
    ("start", "goal"),
    [
        ((math.nan, 0.0), (1.0, 1.0)),
        ((0.0, 0.0), (math.inf, 1.0)),
        ({0: 0.0, 1: 0.0}, (1.0, 1.0)),
        ((0.0, 0.0), [1.0]),
    ],
)
def test_plan_static_aabb_route_rejects_invalid_xy_inputs(
    start: object, goal: object
) -> None:
    with pytest.raises(ValueError):
        _plan_static_aabb_route(start, goal, (), 0.1, 0)


def test_plan_static_aabb_route_rejects_occupied_start_or_goal_without_moving_them() -> None:
    obstacle = (_StaticAABB(0.0, 0.0, 1.0, 1.0),)
    outside = (-1.0, 0.5)
    for occupied in ((0.5, 0.5), (0.0, 0.5), (1.0, 1.0)):
        assert _plan_static_aabb_route(occupied, outside, obstacle, 0.1, 3) is None
        assert _plan_static_aabb_route(outside, occupied, obstacle, 0.1, 3) is None


def test_plan_static_aabb_route_rejects_corner_touch_edge_following_and_tangent() -> None:
    obstacle = (_StaticAABB(0.0, 0.0, 1.0, 1.0),)

    assert _plan_static_aabb_route((-1.0, -1.0), (0.0, 0.0), obstacle, 0.1, 0) is None
    assert _plan_static_aabb_route((-1.0, 0.0), (2.0, 0.0), obstacle, 0.1, 0) is None
    assert _plan_static_aabb_route((-1.0, 1.0), (2.0, 1.0), obstacle, 0.1, 0) is None
    assert _plan_static_aabb_route((-1.0, -0.1), (2.0, -0.1), obstacle, 0.1, 0) == (
        (2.0, -0.1),
    )


def test_plan_static_aabb_route_returns_none_for_closed_static_enclosure() -> None:
    obstacles = (
        _StaticAABB(-2.0, -2.0, -1.0, 2.0),
        _StaticAABB(1.0, -2.0, 2.0, 2.0),
        _StaticAABB(-1.0, -2.0, 1.0, -1.0),
        _StaticAABB(-1.0, 1.0, 1.0, 2.0),
    )

    assert _plan_static_aabb_route((0.0, 0.0), (3.0, 0.0), obstacles, 0.1, 8) is None


@pytest.mark.parametrize(
    ("start", "goal"),
    [
        ((-1.0e308, 0.0), (1.0e308, 0.0)),
        ((0.0, 0.0), (1.3e308, 1.3e308)),
    ],
)
def test_plan_static_aabb_route_rejects_unrepresentable_route_distance(
    start: tuple[float, float], goal: tuple[float, float]
) -> None:
    with pytest.raises(ValueError, match="当前浮点精度"):
        _plan_static_aabb_route(start, goal, (), 0.1, 0)


def test_plan_static_aabb_route_rejects_unrepresentable_accumulated_length() -> None:
    obstacle = (_StaticAABB(-1.0, -8.0e307, 1.0, 8.0e307),)

    with pytest.raises(ValueError, match="路线.*(无法表示|严格累计)"):
        _plan_static_aabb_route(
            (-8.0e307, 0.0),
            (8.0e307, 0.0),
            obstacle,
            1.0e292,
            2,
        )


@pytest.mark.parametrize("waypoint_limit", [2, 4])
def test_plan_static_aabb_route_rejects_positive_length_accumulation_cancellation(
    waypoint_limit: int,
) -> None:
    obstacles = (_StaticAABB(0.0, -0.01, 0.01, 0.01),)

    with pytest.raises(ValueError, match="路线.*当前浮点精度.*累计"):
        _plan_static_aabb_route(
            (-1.0e16, 0.0),
            (1.0, 0.0),
            obstacles,
            0.01,
            waypoint_limit,
        )


def test_plan_static_aabb_route_normal_detour_has_strictly_increasing_length() -> None:
    start = (-1.0, 0.5)
    obstacles = (_StaticAABB(0.0, 0.0, 1.0, 1.0),)
    route = _plan_static_aabb_route(start, (2.0, 0.5), obstacles, 0.1, 2)

    assert route is not None
    total = 0.0
    previous = start
    for point in route:
        segment_length = math.hypot(point[0] - previous[0], point[1] - previous[1])
        updated_total = total + segment_length
        assert segment_length > 0.0
        assert updated_total > total
        total = updated_total
        previous = point


def test_plan_static_aabb_route_does_not_modify_input_tuples() -> None:
    start = (-1.0, 0.5)
    goal = (2.0, 0.5)
    obstacles = (_StaticAABB(0.0, 0.0, 1.0, 1.0),)
    snapshot = (start, goal, obstacles)

    assert _plan_static_aabb_route(start, goal, obstacles, 0.1, 2) is not None
    assert (start, goal, obstacles) == snapshot


@pytest.fixture
def official_inflated_static_obstacles() -> tuple[_StaticAABB, ...]:
    inflation = 0.5154080786132004
    raw_obstacles = (
        _StaticAABB(-2.96, -0.40, -2.90, 2.90),
        _StaticAABB(0.40, -0.40, 0.46, 2.90),
        _StaticAABB(-2.93, -0.43, 0.43, -0.37),
        _StaticAABB(-2.93, 2.87, 0.43, 2.93),
        _StaticAABB(-2.875, 0.370, -2.465, 1.186),
        _StaticAABB(-1.370, 1.915, 0.290, 2.715),
    )
    return tuple(_inflate_aabb(bounds, inflation) for bounds in raw_obstacles)


TEST_ONLY_WAYPOINT_MARGIN_M = 0.01


def test_official_static_aabb_fixture_rejects_known_invalid_stands(
    official_inflated_static_obstacles: tuple[_StaticAABB, ...],
) -> None:
    obstacles = official_inflated_static_obstacles

    assert not any(
        _point_intersects_aabb((-0.70, 0.55), bounds) for bounds in obstacles
    )
    for stand in (
        (-0.8926687370800100, 1.6096780539400557),
        (-0.3603468208092485, 1.6277456647398847),
        (-0.5946292898982682, 1.7024921417376913),
        (-2.0341434247175547, 0.7076086532826956),
    ):
        assert any(_point_intersects_aabb(stand, bounds) for bounds in obstacles)


def test_official_static_aabb_fixture_checks_segments_across_all_obstacles(
    official_inflated_static_obstacles: tuple[_StaticAABB, ...],
) -> None:
    start = (-0.70, 0.55)

    assert not _segment_intersects_any_aabb(
        start, (-0.80, 0.55), official_inflated_static_obstacles
    )
    assert _segment_intersects_any_aabb(
        start,
        (-2.0341434247175547, 0.7076086532826956),
        official_inflated_static_obstacles,
    )


def test_official_static_route_fixture_keeps_return_route_and_rejects_invalid_goals(
    official_inflated_static_obstacles: tuple[_StaticAABB, ...],
) -> None:
    start = (-0.70, 0.55)

    assert _plan_static_aabb_route(
        start,
        (-0.80, 0.55),
        official_inflated_static_obstacles,
        TEST_ONLY_WAYPOINT_MARGIN_M,
        0,
    ) == ((-0.80, 0.55),)
    for invalid_goal in (
        (-0.8926687370800100, 1.6096780539400557),
        (-0.3603468208092485, 1.6277456647398847),
        (-0.5946292898982682, 1.7024921417376913),
        (-2.0341434247175547, 0.7076086532826956),
    ):
        assert (
            _plan_static_aabb_route(
                start,
                invalid_goal,
                official_inflated_static_obstacles,
                TEST_ONLY_WAYPOINT_MARGIN_M,
                8,
            )
            is None
        )


def test_official_static_route_fixture_records_narrow_gap_without_traversability_claim(
    official_inflated_static_obstacles: tuple[_StaticAABB, ...],
) -> None:
    west_wall, east_wall, south_wall, north_wall, shelf, table = (
        official_inflated_static_obstacles
    )
    del west_wall, east_wall, south_wall, north_wall

    assert len(official_inflated_static_obstacles) == 6
    assert table.min_x - shelf.max_x == pytest.approx(0.0641838427735992)


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
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> BaseState:
    return BaseState(
        (x, y, 0.0),
        (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
        yaw,
        linear_velocity,
        angular_velocity,
        frame,
        stamp,
        valid,
    )


def _nav_task(place: tuple[float, float, float] = (2.0, 0.0, 0.5)) -> TaskSpec:
    return TaskSpec(
        task_id=1,
        instruction="move box",
        target_kind="box",
        target_body="box_body",
        target_color="pink",
        place_type="table_point",
        place_world_xyz=place,
        place_frame_id="world",
        place_radius=0.1,
    )


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


def _unchecked_nav_goal(x: object, y: float, yaw: float) -> NavGoal:
    """构造边界之外的损坏对象，仅用于验证导航控制器的防御纵深。"""

    goal = object.__new__(NavGoal)
    for name, value in (
        ("goal_id", "goal"), ("goal_type", "pick"),
        ("pose_xyyaw", (x, y, yaw)), ("frame_id", "odom"),
        ("position_tolerance", 0.05), ("yaw_tolerance", 0.1),
        ("deadline_ns", 2_000_000_000), ("valid", True), ("failure_reason", ""),
    ):
        object.__setattr__(goal, name, value)
    return goal


def _overlapping_waypoint_navigator(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_StaticRouteNavigator, NavGoal]:
    """构造容差区重叠路线；官方fixture的固定间距无法覆盖该生命周期边界。"""

    route = ((0.0, 0.0), (0.04, 0.0), (0.06, 0.0))
    monkeypatch.setattr(
        "team_sorting.navigation._plan_static_aabb_route",
        lambda *_args: route,
    )
    navigator = _StaticRouteNavigator(
        NavigationController(),
        _StaticRouteConfig(
            frame_id="odom",
            inflation_radius_m=0.1,
            waypoint_margin_m=0.01,
            max_intermediate_waypoints=8,
            static_obstacles=(
                _StaticObstacle("far", _StaticAABB(10.0, 10.0, 11.0, 11.0)),
            ),
        ),
    )
    final_goal = NavGoal(
        "overlap-final",
        "pick",
        (0.06, 0.0, 0.0),
        "odom",
        0.05,
        0.1,
        31_000_000_000,
    )
    navigator.plan_route(_nav_base(x=-1.0), final_goal, 1_000_000_000)
    return navigator, final_goal


def test_static_route_consumes_each_odom_timestamp_for_at_most_one_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    navigator, final_goal = _overlapping_waypoint_navigator(monkeypatch)
    first_sample = _nav_base(x=0.02, stamp=1_000_000_000)

    _, first = navigator.update(first_sample, navigator.current_goal, 1_000_000_000)
    assert navigator.current_index == 1
    assert navigator._last_progress_odom_timestamp_ns == 1_000_000_000
    assert first.state == "moving" and not first.success
    assert not navigator.waiting_for_new_odom

    _, repeated = navigator.update(
        first_sample, navigator.current_goal, 1_000_000_001
    )
    assert navigator.current_index == 1
    assert navigator._last_progress_odom_timestamp_ns == 1_000_000_000
    assert repeated.state == "moving" and not repeated.success
    assert navigator.waiting_for_new_odom

    newer_sample = replace(first_sample, timestamp_ns=1_000_000_001)
    _, advanced = navigator.update(
        newer_sample, navigator.current_goal, 1_000_000_001
    )
    assert navigator.current_index == 2
    assert navigator.current_goal is final_goal
    assert navigator._last_progress_odom_timestamp_ns == 1_000_000_001
    assert advanced.state == "moving" and not advanced.success
    assert not navigator.waiting_for_new_odom


def test_static_route_repeated_odom_cannot_complete_final_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    navigator, final_goal = _overlapping_waypoint_navigator(monkeypatch)
    sample = _nav_base(x=0.02, stamp=1_000_000_000)
    navigator.update(sample, navigator.current_goal, 1_000_000_000)
    sample = replace(sample, timestamp_ns=1_000_000_001)
    navigator.update(sample, navigator.current_goal, 1_000_000_001)

    _, repeated = navigator.update(sample, final_goal, 1_000_000_002)
    assert navigator.current_index == 2
    assert repeated.state == "moving" and not repeated.success

    statuses = []
    for timestamp_ns in (1_000_000_002, 1_000_000_003, 1_000_000_004):
        fresh = _nav_base(x=0.06, stamp=timestamp_ns)
        _, status = navigator.update(fresh, final_goal, timestamp_ns)
        statuses.append(status)
    assert [status.success for status in statuses] == [False, False, True]
    assert navigator._last_progress_odom_timestamp_ns == 1_000_000_004


def test_static_route_rejects_odom_older_than_consumed_progress_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    navigator, _ = _overlapping_waypoint_navigator(monkeypatch)
    sample = _nav_base(x=0.02, stamp=1_000_000_000)
    navigator.update(sample, navigator.current_goal, 1_000_000_000)

    with pytest.raises(ValueError, match="Odom.*倒退"):
        navigator.update(
            replace(sample, timestamp_ns=999_999_999),
            navigator.current_goal,
            1_000_000_001,
        )
    assert navigator.current_index == 1
    assert navigator._last_progress_odom_timestamp_ns == 1_000_000_000
    assert not navigator.waiting_for_new_odom


def test_static_route_reset_and_new_plan_clear_consumed_odom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    navigator, final_goal = _overlapping_waypoint_navigator(monkeypatch)
    sample = _nav_base(x=0.02, stamp=1_000_000_000)
    navigator.update(sample, navigator.current_goal, 1_000_000_000)
    assert navigator._last_progress_odom_timestamp_ns == 1_000_000_000

    navigator.reset()
    assert navigator._last_progress_odom_timestamp_ns is None
    assert not navigator.waiting_for_new_odom
    navigator.plan_route(_nav_base(x=-1.0), final_goal, 1_000_000_010)
    assert navigator._last_progress_odom_timestamp_ns is None
    assert not navigator.waiting_for_new_odom


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


def test_navigation_pick_goal_rejects_mismatched_class_and_coincident_base() -> None:
    controller = NavigationController()
    with pytest.raises(ValueError, match="class_id"):
        controller.build_pick_goal(
            _nav_task(),
            ObjectEstimate3D("yellow", (2.0, 0.0, 0.8), 0.9, "odom", 1_000_000_000),
            _nav_base(),
            1_000_000_000,
        )
    with pytest.raises(ValueError, match="重合"):
        controller.build_pick_goal(
            _nav_task(),
            ObjectEstimate3D("pink", (0.0, 0.0, 0.8), 0.9, "odom", 1_000_000_000),
            _nav_base(),
            1_000_000_000,
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


def test_navigation_place_goal_uses_f1_alignment_and_stands_off() -> None:
    controller = NavigationController()
    task = _nav_task()
    goal = controller.build_place_goal(task, _nav_base(frame="odom"), 1_000_000_000)
    assert goal.goal_type == "place"
    assert "early_simulation_standoff_strategy" in goal.goal_id
    assert goal.frame_id == "odom"
    assert goal.pose_xyyaw[:2] != pytest.approx(task.place_world_xyz[:2])
    assert distance_xy(goal.pose_xyyaw, task.place_world_xyz) == pytest.approx(0.6)
    assert goal.pose_xyyaw[2] == pytest.approx(
        math.atan2(
            task.place_world_xyz[1] - goal.pose_xyyaw[1],
            task.place_world_xyz[0] - goal.pose_xyyaw[0],
        )
    )


def test_navigation_place_goal_rejects_wrong_frames_and_invalid_coordinates() -> None:
    controller = NavigationController()
    with pytest.raises(ValueError, match='BaseState.frame_id.*"odom"'):
        controller.build_place_goal(
            _nav_task(), _nav_base(frame="world"), 1_000_000_000
        )
    with pytest.raises(ValueError):
        controller.build_place_goal(
            _nav_task((math.nan, 0.0, 0.5)),
            _nav_base(),
            1_000_000_000,
        )


def test_navigation_return_goal_is_fixed_f5_pose_without_standoff() -> None:
    config = NavigationConfig(goal_timeout_ns=123_456)
    goal = NavigationController(config).build_return_goal(
        _nav_base(x=8.0, y=-3.0, yaw=-1.0), 1_000_000_000
    )
    assert goal.goal_type == "return"
    assert goal.pose_xyyaw == (-0.70, 0.55, math.pi / 2.0)
    assert goal.frame_id == "odom"
    assert goal.position_tolerance == config.position_tolerance_m
    assert goal.yaw_tolerance == config.yaw_tolerance_rad
    assert goal.deadline_ns == 1_000_123_456
    assert "return" in goal.goal_id and "1000000000" in goal.goal_id


def test_navigation_return_goal_rejects_invalid_stale_or_wrong_frame_odom() -> None:
    controller = NavigationController(NavigationConfig(odom_max_age_ns=100))
    for base in (
        _nav_base(valid=False, stamp=1_000),
        _nav_base(stamp=899),
        _nav_base(frame="world", stamp=1_000),
    ):
        with pytest.raises(ValueError):
            controller.build_return_goal(base, 1_000)


def test_navigation_all_goal_types_share_the_same_odom_update_path() -> None:
    controller = NavigationController()
    base = _nav_base()
    task = _nav_task()
    goals = (
        controller.build_pick_goal(
            task,
            ObjectEstimate3D("pink", (2.0, 0.0, 0.8), 0.9, "odom", 1_000_000_000),
            base,
            1_000_000_000,
        ),
        controller.build_place_goal(task, base, 1_000_000_000),
        controller.build_return_goal(base, 1_000_000_000),
    )
    for goal in goals:
        command, status = controller.update(base, goal, 1_000_000_000)
        assert command.valid
        assert command.valid_until_ns == 1_200_000_000
        assert status.goal_id == goal.goal_id
        assert status.state in {"aligning_to_goal", "moving"}
        assert not status.success


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
    arrival_goal = _nav_goal(0.01, 0.0, 0.05)
    for offset_ns in (0, 50_000_000):
        pending, pending_status = controller.update(
            _nav_base(stamp=1_000_000_000 + offset_ns),
            arrival_goal,
            1_000_000_000 + offset_ns,
        )
        assert (pending.v, pending.w) == (0.0, 0.0)
        assert not pending_status.success
    arrived, arrived_status = controller.update(
        _nav_base(stamp=1_100_000_000), arrival_goal, 1_100_000_000
    )
    assert (arrived.v, arrived.w) == (0.0, 0.0)
    assert arrived_status.success


def test_navigation_requires_low_actual_speed_and_distinct_consecutive_odom() -> None:
    controller = NavigationController(
        NavigationConfig(
            max_settled_linear_speed_mps=0.02,
            max_settled_angular_speed_radps=0.03,
            settled_required_cycles=2,
        )
    )
    goal = _nav_goal(0.01, 0.0, 0.05)
    for fast_base in (
        _nav_base(linear_velocity=(0.021, 0.0, 0.0)),
        _nav_base(angular_velocity=(0.0, 0.0, 0.031)),
    ):
        fast, fast_status = controller.update(fast_base, goal, 1_000_000_000)
        assert (fast.v, fast.w) == (0.0, 0.0)
        assert fast.valid and not fast_status.success

    first, first_status = controller.update(_nav_base(), goal, 1_000_000_000)
    repeated, repeated_status = controller.update(_nav_base(), goal, 1_000_000_000)
    assert (first.v, first.w) == (0.0, 0.0)
    assert not first_status.success and not repeated_status.success
    arrived, arrived_status = controller.update(
        _nav_base(stamp=1_050_000_000), goal, 1_050_000_000
    )
    assert (arrived.v, arrived.w) == (0.0, 0.0)
    assert arrived_status.success


def test_navigation_resets_settled_confirmation_after_leaving_tolerance() -> None:
    controller = NavigationController(NavigationConfig(settled_required_cycles=2))
    goal = _nav_goal(0.01, 0.0, 0.0)
    _, first = controller.update(_nav_base(), goal, 1_000_000_000)
    assert not first.success
    moving, outside = controller.update(
        _nav_base(x=-0.2, stamp=1_050_000_000), goal, 1_050_000_000
    )
    assert moving.v > 0.0 and not outside.success
    _, restarted = controller.update(
        _nav_base(stamp=1_100_000_000), goal, 1_100_000_000
    )
    assert not restarted.success
    _, arrived = controller.update(
        _nav_base(stamp=1_150_000_000), goal, 1_150_000_000
    )
    assert arrived.success


def test_navigation_goal_lifecycles_do_not_share_settled_samples() -> None:
    controller = NavigationController(NavigationConfig(settled_required_cycles=2))
    first_goal = _nav_goal(0.01, 0.0, 0.0)
    second_goal = NavGoal(
        "goal-2", "place", (0.01, 0.0, 0.0), "odom", 0.05, 0.1, 2_000_000_000
    )
    controller.update(_nav_base(), first_goal, 1_000_000_000)
    _, first_second_goal_status = controller.update(
        _nav_base(stamp=1_050_000_000), second_goal, 1_050_000_000
    )
    assert not first_second_goal_status.success
    _, arrived = controller.update(
        _nav_base(stamp=1_100_000_000), second_goal, 1_100_000_000
    )
    assert arrived.success


def test_navigation_arrival_latch_holds_zero_until_goal_changes() -> None:
    controller = NavigationController(NavigationConfig(settled_required_cycles=1))
    goal = _nav_goal(0.01, 0.0, 0.0)
    command, arrived = controller.update(_nav_base(), goal, 1_000_000_000)
    assert arrived.success and (command.v, command.w) == (0.0, 0.0)

    held, displaced = controller.update(
        _nav_base(x=-0.2, stamp=1_050_000_000), goal, 1_050_000_000
    )
    assert (held.v, held.w) == (0.0, 0.0)
    assert held.valid and not displaced.success
    assert "等待 FSM 阶段切换" in displaced.failure_reason

    replacement = NavGoal(
        "replacement", "return", (1.0, 0.0, 0.0), "odom", 0.05, 0.1, 2_000_000_000
    )
    moving, replacement_status = controller.update(
        _nav_base(x=-0.2, stamp=1_100_000_000), replacement, 1_100_000_000
    )
    assert moving.v > 0.0 and not replacement_status.success


def test_navigation_abnormal_settled_odom_interval_fails_closed() -> None:
    controller = NavigationController(
        NavigationConfig(settled_required_cycles=2, settled_max_odom_gap_ns=10)
    )
    goal = _nav_goal(0.01, 0.0, 0.0)
    controller.update(_nav_base(stamp=1_000), goal, 1_000)
    command, status = controller.update(_nav_base(stamp=1_011), goal, 1_011)
    assert (command.v, command.w) == (0.0, 0.0)
    assert not command.valid
    assert status.state == "failed" and "时间间隔异常" in status.failure_reason


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_settled_linear_speed_mps": -0.01},
        {"max_settled_angular_speed_radps": math.nan},
        {"settled_required_cycles": 0},
        {"settled_required_cycles": True},
        {"settled_max_odom_gap_ns": 0},
    ],
)
def test_navigation_rejects_invalid_settling_configuration(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        NavigationConfig(**kwargs)  # type: ignore[arg-type]


def test_navigation_update_safely_stops_on_invalid_and_stale_odom() -> None:
    controller = NavigationController()
    cases = (
        (_nav_base(valid=False), _nav_goal(1.0, 0.0, 0.0)),
        (_nav_base(stamp=800_000_000), _nav_goal(1.0, 0.0, 0.0)),
        (_nav_base(), _unchecked_nav_goal(math.nan, 0.0, 0.0)),
        (_nav_base(), _nav_goal(1.0, 0.0, 0.0, frame="world")),
    )
    for base, goal in cases:
        command, status = controller.update(base, goal, 1_000_000_000)
        assert (command.v, command.w) == (0.0, 0.0)
        assert not status.success
        assert status.failure_reason


def test_navigation_deadline_metadata_does_not_consume_paused_budget() -> None:
    controller = NavigationController(NavigationConfig(goal_timeout_ns=100))
    goal = _nav_goal(1.0, 0.0, 0.0, deadline=1)
    command, status = controller.update(_nav_base(stamp=1_000), goal, 1_000)
    controller.record_control_result(10_000, False)
    command, status = controller.update(_nav_base(stamp=10_000), goal, 10_000)
    assert command.valid
    assert status.state != "timeout"


def test_navigation_control_budget_starts_paused_and_resumes_without_backfill() -> None:
    controller = NavigationController(NavigationConfig(goal_timeout_ns=100))
    goal = _nav_goal(1.0, 0.0, 0.0, deadline=1)
    controller.update(_nav_base(stamp=1_000), goal, 1_000)
    controller.record_control_result(10_000, False)
    controller.record_control_result(20_000, True)
    _, resumed = controller.update(_nav_base(stamp=20_000), goal, 20_000)
    assert resumed.state != "timeout"
    controller.record_control_result(20_099, True)
    _, below = controller.update(_nav_base(stamp=20_099), goal, 20_099)
    assert below.state != "timeout"
    controller.record_control_result(20_100, True)
    command, timed_out = controller.update(_nav_base(stamp=20_100), goal, 20_100)
    assert not command.valid
    assert timed_out.state == "timeout"


def test_navigation_control_budget_only_accumulates_true_intervals() -> None:
    controller = NavigationController(NavigationConfig(goal_timeout_ns=100))
    goal = _nav_goal(1.0, 0.0, 0.0)
    controller.update(_nav_base(stamp=1_000), goal, 1_000)
    for timestamp, granted in (
        (1_010, True), (1_040, True), (1_050, False),
        (5_000, False), (6_000, True), (6_059, True),
    ):
        controller.record_control_result(timestamp, granted)
    _, below = controller.update(_nav_base(stamp=6_059), goal, 6_059)
    assert below.state != "timeout"
    controller.record_control_result(6_060, True)
    _, timed_out = controller.update(_nav_base(stamp=6_060), goal, 6_060)
    assert timed_out.state == "timeout"


def test_navigation_control_budget_matches_production_update_record_order() -> None:
    controller = NavigationController(NavigationConfig(goal_timeout_ns=100))
    goal = _nav_goal(1.0, 0.0, 0.0)

    _, at_n = controller.update(_nav_base(stamp=1_000), goal, 1_000)
    assert at_n.state != "timeout"
    controller.record_control_result(1_000, True)

    _, at_n1 = controller.update(_nav_base(stamp=1_040), goal, 1_040)
    assert at_n1.state != "timeout"
    controller.record_control_result(1_040, False)

    _, paused = controller.update(_nav_base(stamp=9_000), goal, 9_000)
    assert paused.state != "timeout"
    controller.record_control_result(9_000, True)

    command, at_n3 = controller.update(_nav_base(stamp=9_060), goal, 9_060)
    assert not command.valid
    assert at_n3.state == "timeout"


def test_navigation_same_timestamp_feedback_is_idempotent() -> None:
    controller = NavigationController(NavigationConfig(goal_timeout_ns=50))
    goal = _nav_goal(1.0, 0.0, 0.0)
    controller.update(_nav_base(stamp=1_000), goal, 1_000)
    controller.record_control_result(1_000, False)
    controller.record_control_result(1_000, True)

    _, before = controller.update(_nav_base(stamp=1_049), goal, 1_049)
    assert before.state != "timeout"
    controller.record_control_result(1_049, True)
    _, boundary = controller.update(_nav_base(stamp=1_050), goal, 1_050)
    assert boundary.state == "timeout"


def test_navigation_revoke_without_timestamp_is_idempotent_and_excludes_unknown_time() -> None:
    controller = NavigationController(NavigationConfig(goal_timeout_ns=100))
    goal = _nav_goal(1.0, 0.0, 0.0)
    controller.update(_nav_base(stamp=1_000), goal, 1_000)
    controller.record_control_result(1_000, True)
    elapsed_before = controller._execution_elapsed_ns
    checkpoint_before = controller._clock_checkpoint_ns

    controller._revoke_control_without_timestamp()
    controller._revoke_control_without_timestamp()

    assert controller._control_granted_since_checkpoint is False
    assert controller._execution_elapsed_ns == elapsed_before
    assert controller._clock_checkpoint_ns == checkpoint_before
    controller.record_control_result(9_000, False)
    assert controller._execution_elapsed_ns == elapsed_before


def test_navigation_new_goal_and_reset_do_not_leak_control_budget() -> None:
    controller = NavigationController(NavigationConfig(goal_timeout_ns=100))
    first = _nav_goal(1.0, 0.0, 0.0)
    second = NavGoal("second", "place", (1.0, 0.0, 0.0), "odom", 0.05, 0.1, 2)
    controller.update(_nav_base(stamp=1_000), first, 1_000)
    controller.record_control_result(1_000, True)
    controller.record_control_result(1_099, True)
    _, fresh = controller.update(_nav_base(stamp=1_100), second, 1_100)
    assert fresh.state != "timeout"
    controller.record_control_result(1_200, True)
    _, still_fresh = controller.update(_nav_base(stamp=1_200), second, 1_200)
    assert still_fresh.state != "timeout"
    controller.reset()
    controller.update(_nav_base(stamp=2_000), first, 2_000)
    controller.record_control_result(3_000, False)
    _, reset_status = controller.update(_nav_base(stamp=3_000), first, 3_000)
    assert reset_status.state != "timeout"


def test_navigation_invalid_odom_terminates_goal_and_replacement_resets_budget() -> None:
    controller = NavigationController(NavigationConfig(goal_timeout_ns=100))
    first = _nav_goal(1.0, 0.0, 0.0)
    controller.update(_nav_base(stamp=1_000), first, 1_000)
    controller.record_control_result(1_000, True)
    command, failed = controller.update(
        _nav_base(stamp=1_010, valid=False), first, 1_010
    )
    assert not command.valid and failed.state == "failed"
    controller.record_control_result(2_000, True)
    assert controller._control_granted_since_checkpoint is False

    replacement = NavGoal(
        "after-odom-failure", "pick", (1.0, 0.0, 0.0), "odom",
        0.05, 0.1, 2,
    )
    _, fresh = controller.update(_nav_base(stamp=2_000), replacement, 2_000)
    assert fresh.state != "timeout"


@pytest.mark.parametrize("timestamp", [True, -1, 1.0, "1"])
def test_navigation_control_feedback_invalid_time_fails_closed(timestamp: object) -> None:
    controller = NavigationController(NavigationConfig(goal_timeout_ns=100))
    goal = _nav_goal(1.0, 0.0, 0.0)
    controller.update(_nav_base(stamp=1_000), goal, 1_000)
    controller.record_control_result(1_000, True)
    with pytest.raises(ValueError):
        controller.record_control_result(timestamp, True)  # type: ignore[arg-type]
    controller.record_control_result(2_000, False)
    _, status = controller.update(_nav_base(stamp=2_000), goal, 2_000)
    assert status.state != "timeout"


def test_navigation_control_feedback_rejects_bad_flag_and_time_regression() -> None:
    controller = NavigationController(NavigationConfig(goal_timeout_ns=100))
    goal = _nav_goal(1.0, 0.0, 0.0)
    controller.update(_nav_base(stamp=1_000), goal, 1_000)
    controller.record_control_result(1_000, True)
    for timestamp, granted in ((999, True), (1_001, 1)):
        with pytest.raises(ValueError):
            controller.record_control_result(timestamp, granted)  # type: ignore[arg-type]
    controller.record_control_result(2_000, False)
    _, status = controller.update(_nav_base(stamp=2_000), goal, 2_000)
    assert status.state != "timeout"


@pytest.mark.parametrize("timestamp", [True, -1, 1.0])
def test_navigation_update_fails_closed_for_invalid_cycle_timestamps(
    timestamp: object,
) -> None:
    command, status = NavigationController().update(
        _nav_base(), _nav_goal(1.0, 0.0, 0.0), timestamp  # type: ignore[arg-type]
    )
    assert (command.v, command.w) == (0.0, 0.0)
    assert not command.valid
    assert command.valid_until_ns == 200_000_000
    assert status.state == "failed" and not status.success
    assert status.failure_reason


def test_navigation_update_rejects_odom_timestamp_from_the_future() -> None:
    command, status = NavigationController().update(
        _nav_base(stamp=1_000_000_001),
        _nav_goal(1.0, 0.0, 0.0),
        1_000_000_000,
    )
    assert (command.v, command.w) == (0.0, 0.0)
    assert not command.valid
    assert status.state == "failed" and "晚于" in status.failure_reason


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

def test_projection_and_delayed_adapters_import_without_official_environment() -> None:
    intrinsics = _intrinsics()
    assert project_pixel_to_camera(320.0, 240.0, 2.0, intrinsics) == pytest.approx(
        (0.0, 0.0, 2.0)
    )

    adapters = (OfficialYoloAdapter(), CameraTransformProvider(), OfficialKDLAdapter())
    assert all(adapter is not None for adapter in adapters)
    planner = ArmPlanner(adapters[-1], ArmPlanningConfig())
    assert planner._ik_adapter is adapters[-1]


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
        "source_slots",
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
    navigation = config["navigation"]
    assert isinstance(timing, dict)
    assert isinstance(perception, dict)
    assert isinstance(fsm, dict)
    assert isinstance(action_mux, dict)
    assert isinstance(navigation, dict)

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
        "reassociation_enabled",
        "reassociation_max_age_s",
        "reassociation_max_distance_m",
        "reassociation_size_tolerance_ratio",
        "reassociation_ambiguity_ratio",
        "max_identity_tracks",
        "object_dimensions_m",
        "object_local_size_xyz_m",
        "pose_refinement",
    }
    assert isinstance(estimator["depth_radius_px"], int)
    assert isinstance(estimator["converge_frames"], int)
    assert _strict_finite_number(estimator["ema_alpha"]) > 0.0
    assert _strict_finite_number(estimator["max_track_age_s"]) > 0.0
    assert _strict_finite_number(estimator["max_position_jump_m"]) > 0.0
    assert estimator["reassociation_enabled"] is True
    assert _strict_finite_number(estimator["reassociation_max_age_s"]) > 0.0
    assert _strict_finite_number(estimator["reassociation_max_distance_m"]) > 0.0
    reassociation_size_tolerance = _strict_finite_number(
        estimator["reassociation_size_tolerance_ratio"]
    )
    assert 0.0 <= reassociation_size_tolerance <= 1.0
    assert _strict_finite_number(estimator["reassociation_ambiguity_ratio"]) > 1.0
    max_identity_tracks = estimator["max_identity_tracks"]
    assert (
        isinstance(max_identity_tracks, int)
        and not isinstance(max_identity_tracks, bool)
        and max_identity_tracks > 0
    )
    dimensions = estimator["object_dimensions_m"]
    assert isinstance(dimensions, dict)
    assert set(dimensions) == set(OfficialYoloAdapter.CLASS_NAMES)
    for values in dimensions.values():
        assert isinstance(values, list) and len(values) == 3
        assert tuple(_strict_finite_number(value) for value in values) == pytest.approx(
            (0.24, 0.16, 0.19)
        )
    local_sizes = estimator["object_local_size_xyz_m"]
    assert isinstance(local_sizes, dict)
    assert set(local_sizes) == set(OfficialYoloAdapter.CLASS_NAMES)
    for values in local_sizes.values():
        assert isinstance(values, list) and len(values) == 3
        assert tuple(_strict_finite_number(value) for value in values) == pytest.approx(
            (0.24, 0.16, 0.19)
        )
    pose = estimator["pose_refinement"]
    assert isinstance(pose, dict)
    assert set(pose) == {
        "enabled",
        "min_points",
        "required_frames",
        "depth_band_m",
        "max_position_delta_m",
        "max_angular_delta_rad",
        "max_extent_error_ratio",
    }
    assert pose["enabled"] is False
    assert isinstance(pose["min_points"], int) and pose["min_points"] > 0
    assert isinstance(pose["required_frames"], int) and pose["required_frames"] > 0
    for name in (
        "depth_band_m",
        "max_position_delta_m",
        "max_angular_delta_rad",
        "max_extent_error_ratio",
    ):
        assert _strict_finite_number(pose[name]) > 0.0

    retry_count = fsm["max_pick_retries"]
    assert isinstance(retry_count, int) and not isinstance(retry_count, bool) and retry_count >= 0
    assert _strict_finite_number(action_mux["max_abs_base_v"]) >= 0.0
    assert _strict_finite_number(action_mux["max_abs_base_w"]) >= 0.0
    assert set(navigation) == {field.name for field in fields(NavigationConfig)}
    parsed_navigation = _navigation_config_from_config(config)
    assert parsed_navigation.max_abs_v_mps == pytest.approx(0.25)
    assert parsed_navigation.max_abs_w_radps == pytest.approx(0.5)
    assert parsed_navigation.settled_required_cycles == 3


def test_navigation_config_mapping_rejects_missing_and_unknown_fields() -> None:
    config = _project_config()
    missing = {**config, "navigation": dict(config["navigation"])}
    missing["navigation"].pop("command_ttl_ns")
    with pytest.raises(ValueError, match="command_ttl_ns"):
        _navigation_config_from_config(missing)

    unknown = {**config, "navigation": dict(config["navigation"])}
    unknown["navigation"]["hidden_default"] = 1
    with pytest.raises(ValueError, match="hidden_default"):
        _navigation_config_from_config(unknown)


def test_navigation_safety_is_explicitly_disabled_by_default() -> None:
    assert _navigation_safety_enabled_from_config(_project_config()) is False


@pytest.mark.parametrize("enabled", [0, 1, "false", None])
def test_navigation_safety_enabled_requires_strict_bool(enabled: object) -> None:
    config = _project_config()
    config["navigation_safety"]["enabled"] = enabled

    with pytest.raises(ValueError, match="enabled.*bool"):
        _navigation_safety_enabled_from_config(config)


@pytest.mark.parametrize(
    "invalid_case",
    ("missing_field", "missing_pose", "wrong_classes", "bool_value"),
)
def test_perception_config_reader_rejects_invalid_local_size_schema(
    invalid_case: str,
) -> None:
    config = _project_config()
    perception = config["perception"]
    assert isinstance(perception, dict)
    estimator = perception["estimator_3d"]
    assert isinstance(estimator, dict)

    if invalid_case == "missing_field":
        del estimator["object_local_size_xyz_m"]
    elif invalid_case == "missing_pose":
        del estimator["pose_refinement"]
    else:
        local_sizes = estimator["object_local_size_xyz_m"]
        assert isinstance(local_sizes, dict)
        if invalid_case == "wrong_classes":
            del local_sizes["brown"]
            local_sizes["red"] = [0.24, 0.16, 0.19]
        else:
            local_sizes["pink"] = [0.24, True, 0.19]

    with pytest.raises(RuntimeError) as exc_info:
        _perception_pipeline_from_config(
            config,  # type: ignore[arg-type]
            CameraTransformProvider(),
        )

    message = str(exc_info.value)
    assert (
        "pose_refinement" if invalid_case == "missing_pose"
        else "object_local_size_xyz_m"
    ) in message
    if invalid_case == "wrong_classes":
        assert "brown" in message and "red" in message
    if invalid_case == "bool_value":
        assert "pink" in message and "bool" in message


def test_config_source_slots_and_action_limits_have_frozen_official_values() -> None:
    """冻结当前官方offline场景的source-slot事实和17维动作限位。"""

    config = _project_config()
    source_slots = config["source_slots"]
    action_mux = config["action_mux"]
    assert isinstance(source_slots, dict)
    assert isinstance(action_mux, dict)
    frozen = _source_slot_config_from_config(config)
    assert frozen.frame_id == "odom"
    assert frozen.match_tolerance_m == pytest.approx(0.05)
    assert frozen.task_source_slots == {1: "table_side", 2: "shelf", 3: "table_top"}
    assert frozen.centers_by_slot == {
        "table_side": ((-1.0, 2.2, 0.834), (-0.18, 2.2, 0.834)),
        "table_top": ((-0.54, 2.3, 1.004),),
        "shelf": ((-2.63, 0.778, 0.508), (-2.63, 0.778, 0.837), (-2.63, 0.778, 1.166)),
    }
    assert frozen.yaw_by_slot["table_side"] == pytest.approx(0.0)
    assert frozen.yaw_by_slot["table_top"] == pytest.approx(math.pi / 2.0)
    assert frozen.yaw_by_slot["shelf"] == pytest.approx(math.pi / 2.0)
    assert config["perception"]["estimator_3d"]["pose_refinement"]["enabled"] is False
    lower = action_mux["joint_lower"]
    upper = action_mux["joint_upper"]
    assert isinstance(lower, list) and isinstance(upper, list)
    assert len(lower) == len(upper) == 17
    lower_numbers = tuple(_strict_finite_number(value) for value in lower)
    upper_numbers = tuple(_strict_finite_number(value) for value in upper)
    assert all(low <= high for low, high in zip(lower_numbers, upper_numbers))


def _source_slot_task(task_id: int, color: str = "pink") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        instruction="official source-slot task",
        target_kind="cuboid_box",
        target_body=f"box_{color}",
        target_color=color,
        place_type="table_point",
        place_world_xyz=(-1.0, 2.2, 0.834),
        place_frame_id="world",
        place_radius=0.28,
        timestamp_ns=100,
    )


def _source_slot_estimate(
    position: tuple[float, float, float],
    *,
    orientation: tuple[float, float, float, float] | None = None,
) -> ObjectEstimate3D:
    return ObjectEstimate3D(
        "pink", position, 0.9, "odom", 100, object_id="stable:source",
        orientation_xyzw=orientation, size_xyz_m=(0.24, 0.16, 0.19),
    )


def test_source_slot_selection_is_task_aware_and_resolves_only_unknown_orientation() -> None:
    slots = _source_slot_config_from_config(_project_config())

    task1 = _source_slot_task(1)
    assert _select_target_estimate(
        (_source_slot_estimate((-1.0, 2.2, 0.834)),), task1, 120, 100,
        source_slots=slots, require_source_slot=True, resolve_unknown_orientation=True,
    ).orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert _select_target_estimate(
        (_source_slot_estimate((-0.18, 2.2, 0.834)),), task1, 120, 100,
        source_slots=slots, require_source_slot=True,
    ) is not None
    assert _select_target_estimate(
        (_source_slot_estimate((-2.63, 0.778, 0.837)),), task1, 120, 100,
        source_slots=slots, require_source_slot=True,
    ) is None

    task2 = _source_slot_task(2)
    selected2 = _select_target_estimate(
        (_source_slot_estimate((-2.63, 0.778, 1.166)),), task2, 120, 100,
        source_slots=slots, require_source_slot=True, resolve_unknown_orientation=True,
    )
    assert selected2 is not None
    assert selected2.orientation_xyzw == pytest.approx(
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    )
    assert _select_target_estimate(
        (_source_slot_estimate((-1.0, 2.2, 0.834)),), task2, 120, 100,
        source_slots=slots, require_source_slot=True,
    ) is None

    task3 = _source_slot_task(3)
    selected3 = _select_target_estimate(
        (_source_slot_estimate((-0.54, 2.3, 1.004)),), task3, 120, 100,
        source_slots=slots, require_source_slot=True, resolve_unknown_orientation=True,
    )
    assert selected3 is not None
    assert selected3.orientation_xyzw == pytest.approx(
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    )
    assert _select_target_estimate(
        (_source_slot_estimate((-0.18, 2.2, 0.834)),), task3, 120, 100,
        source_slots=slots, require_source_slot=True,
    ) is None

    at_limit = _source_slot_estimate((-0.951, 2.2, 0.834))
    beyond_limit = _source_slot_estimate((-0.949, 2.2, 0.834))
    assert _select_target_estimate(
        (at_limit,), task1, 120, 100, source_slots=slots, require_source_slot=True,
    ) is not None
    assert _select_target_estimate(
        (beyond_limit,), task1, 120, 100, source_slots=slots, require_source_slot=True,
    ) is None

    observed = _source_slot_estimate(
        (-1.0, 2.2, 0.834), orientation=(0.0, 0.0, 0.2, math.sqrt(0.96))
    )
    assert _select_target_estimate(
        (observed,), task1, 120, 100,
        source_slots=slots, require_source_slot=True, resolve_unknown_orientation=True,
    ).orientation_xyzw == observed.orientation_xyzw


def test_source_slot_gate_is_not_a_global_post_grasp_object_gate() -> None:
    slots = _source_slot_config_from_config(_project_config())
    moved_after_grasp = _source_slot_estimate((-2.68, 0.54, 0.498))
    assert _select_target_estimate(
        (moved_after_grasp,), _source_slot_task(3), 120, 100,
    ) is moved_after_grasp
    assert _select_target_estimate(
        (moved_after_grasp,), _source_slot_task(3), 120, 100,
        source_slots=slots, require_source_slot=True,
    ) is None

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
    provider = CameraTransformProvider(output_frame="odom")
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
    provider = CameraTransformProvider(output_frame="odom")
    provider._fk = _RecordingFK(position, quaternion)
    with pytest.raises(RuntimeError, match="位姿无效"):
        provider.camera_to_output((1.0, 0.0, 0.0), _base(), _actual_joints())


def test_camera_transform_rejects_invalid_input_state_and_point() -> None:
    provider = CameraTransformProvider(output_frame="odom")
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
        pose = self.compute_head_camera_pose(base, joints)
        return self.transform_camera_point(camera_point_xyz, pose)

    def compute_head_camera_pose(
        self,
        base: BaseState,
        joints: RobotJointState,
    ) -> _HeadCameraPose:
        if not base.valid:
            raise ValueError(f"底盘状态无效，不能计算相机外参：{base.failure_reason}")
        if not joints.valid:
            raise ValueError(
                f"实际关节状态无效，不能计算相机外参：{joints.failure_reason}"
            )
        if base.frame_id != self.output_frame:
            raise ValueError(
                f"BaseState.frame_id ({base.frame_id!r}) 与 "
                f"CameraTransformProvider.output_frame ({self.output_frame!r}) 不一致"
            )
        return _HeadCameraPose(
            self.offset_xyz,
            (1.0, 0.0, 0.0, 0.0),
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
    assert result.size_xyz_m is None
    assert "heuristic center approximation" in result.failure_reason
    assert estimator._tracks == {}


def test_perception_dimensions_only_compensate_center_not_local_size_axes() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.11, 0.22, 0.33)},
    )
    result = estimator.estimate(
        (_estimator_detection(track_id=9),),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]
    assert result.valid
    assert result.position_xyz[2] == pytest.approx(1.165)
    assert result.size_xyz_m is None
    assert estimator._dims["pink"] == (0.11, 0.22, 0.33)


def test_perception_independent_local_size_is_published_without_axis_swapping() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.11, 0.22, 0.33)},
        object_local_size_xyz_m={"pink": (0.24, 0.16, 0.19)},
    )
    result = estimator.estimate(
        (_estimator_detection(track_id=10),),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert result.valid
    # 中心补偿仍只取旧映射的相机视线第三项；局部尺寸来自独立映射并保持XYZ顺序。
    assert result.position_xyz[2] == pytest.approx(1.165)
    assert result.size_xyz_m == pytest.approx((0.24, 0.16, 0.19))
    assert result.orientation_xyzw is None
    assert estimator._dims["pink"] == (0.11, 0.22, 0.33)
    assert estimator._local_sizes["pink"] == (0.24, 0.16, 0.19)


def test_perception_point_cloud_pose_requires_stable_multiframe_refine() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        # SEARCH 的旧视线启发式深度故意与已确认局部尺寸不同，证明成功后中心换源。
        object_dimensions_m={"pink": (0.24, 0.16, 0.33)},
        object_local_size_xyz_m={"pink": (0.24, 0.16, 0.19)},
        pose_refinement_enabled=True,
        pose_min_points=64,
        pose_required_frames=3,
        pose_depth_band_m=0.05,
        pose_max_position_delta_m=0.01,
        pose_max_angular_delta_rad=0.1,
        pose_max_extent_error_ratio=0.1,
    )
    results = []
    for timestamp_ns in (100, 101, 102):
        results.append(
            estimator.estimate(
                (
                    _estimator_detection(
                        bbox_xyxy=(260.0, 200.0, 380.0, 280.0),
                        timestamp_ns=timestamp_ns,
                        track_id=10,
                    ),
                ),
                _estimator_depth(1000.0, timestamp_ns),
                _intrinsics(timestamp_ns=timestamp_ns),
                _base(),
                _actual_joints(),
            )[0]
        )

    assert results[0].orientation_xyzw is None
    assert results[1].orientation_xyzw is None
    assert results[2].orientation_xyzw is not None
    # 点云平面位于相机 z=1.0；SEARCH 用旧启发式深度0.33得到1.165，
    # REFINE 则用局部深度0.19拟合出中心1.095，明确证明 position 已换源。
    assert results[0].position_xyz[2] == pytest.approx(1.165)
    assert results[1].position_xyz[2] == pytest.approx(1.165)
    assert results[2].position_xyz[2] == pytest.approx(1.095, abs=1e-6)
    assert 2.0 * math.acos(abs(results[2].orientation_xyzw[3])) < 0.01
    assert "pose converged" in results[2].failure_reason


@pytest.mark.parametrize("failure", ("insufficient", "bad_depth", "size_mismatch"))
def test_perception_point_cloud_refine_failures_keep_orientation_unknown(
    failure: str,
) -> None:
    local_size = (
        (0.60, 0.50, 0.40)
        if failure == "size_mismatch"
        else (0.24, 0.16, 0.19)
    )
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.24, 0.16, 0.33)},
        object_local_size_xyz_m={"pink": local_size},
        pose_refinement_enabled=True,
        pose_min_points=100_000 if failure == "insufficient" else 64,
        pose_required_frames=1,
        pose_max_extent_error_ratio=0.1,
    )
    image = np.full((480, 640), 1000.0)
    if failure == "bad_depth":
        image[:] = np.nan
    result = estimator.estimate(
        (_estimator_detection(track_id=10),),
        _estimator_depth(1000.0, 100, image=image),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert result.orientation_xyzw is None
    if failure != "bad_depth":
        assert result.valid
        assert result.position_xyz[2] == pytest.approx(1.165)


@pytest.mark.parametrize("unstable_component", ("center", "orientation"))
def test_perception_point_cloud_unstable_pose_resets_multiframe_refine(
    unstable_component: str,
) -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.24, 0.16, 0.33)},
        object_local_size_xyz_m={"pink": (0.24, 0.16, 0.19)},
        pose_refinement_enabled=True,
        pose_required_frames=2,
        pose_max_position_delta_m=0.01,
    )
    stable = ((0.0, 0.0, 1.095), (0.0, 0.0, 0.0, 1.0))
    unstable = (
        ((0.0, 0.0, 1.195), stable[1])
        if unstable_component == "center"
        else (stable[0], (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    )
    candidates = iter((stable, unstable, stable))
    estimator._point_cloud_pose_candidate = (  # type: ignore[method-assign]
        lambda *args: next(candidates)
    )

    results = [
        estimator.estimate(
            (_estimator_detection(timestamp_ns=timestamp_ns, track_id=10),),
            _estimator_depth(1000.0, timestamp_ns),
            _intrinsics(timestamp_ns=timestamp_ns),
            _base(),
            _actual_joints(),
        )[0]
        for timestamp_ns in (100, 101, 102)
    ]

    assert all(result.orientation_xyzw is None for result in results)
    assert all(result.position_xyz[2] == pytest.approx(1.165) for result in results)


def test_cuboid_point_cloud_fit_recovers_rotated_axes_up_to_box_symmetry() -> None:
    size = np.asarray((0.24, 0.16, 0.19))
    coordinates = [
        np.linspace(-extent / 2.0, extent / 2.0, 9) for extent in size
    ]
    points = []
    for axis in range(3):
        others = [index for index in range(3) if index != axis]
        for sign in (-1.0, 1.0):
            for first in coordinates[others[0]]:
                for second in coordinates[others[1]]:
                    point = np.zeros(3)
                    point[axis] = sign * size[axis] / 2.0
                    point[others[0]] = first
                    point[others[1]] = second
                    points.append(point)
    yaw, pitch, roll = 0.55, -0.30, 0.20
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    expected_rotation = np.asarray(
        (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        )
    )
    rotated = np.asarray(points) @ expected_rotation.T + np.asarray((1.0, 2.0, 3.0))
    quaternion = perception_3d_module._fit_cuboid_orientation_xyzw(
        rotated, tuple(size), 0.1, np
    )

    assert quaternion is not None
    x, y, z, w = quaternion
    actual_rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    assert np.abs(actual_rotation.T @ expected_rotation) == pytest.approx(
        np.eye(3), abs=1e-6
    )


@pytest.mark.parametrize(
    "local_sizes",
    (
        {" ": (0.24, 0.16, 0.19)},
        {" pink ": (0.24, 0.16, 0.19)},
        {"pink": (0.24, 0.16)},
        {"pink": (0.24, 0.16, 0.0)},
        {"pink": (0.24, -0.16, 0.19)},
        {"pink": (0.24, float("nan"), 0.19)},
        {"pink": (0.24, float("inf"), 0.19)},
        {"pink": (0.24, True, 0.19)},
        {"pink": (0.24, "0.16", 0.19)},
    ),
)
def test_perception_rejects_invalid_independent_local_sizes(
    local_sizes: dict[str, tuple[object, ...]],
) -> None:
    with pytest.raises(ValueError, match="object_local_size_xyz_m"):
        Perception3DEstimator(
            _OffsetTransformProvider(),
            object_local_size_xyz_m=local_sizes,  # type: ignore[arg-type]
        )


def test_perception_routes_distinct_local_sizes_by_class_id() -> None:
    expected = {
        "pink": (0.21, 0.11, 0.31),
        "yellow": (0.42, 0.22, 0.62),
    }
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={
            "pink": (0.1, 0.1, 0.2),
            "yellow": (0.1, 0.1, 0.2),
        },
        object_local_size_xyz_m=expected,
    )
    results = estimator.estimate(
        (
            _estimator_detection("pink", track_id=11),
            _estimator_detection("yellow", track_id=12),
        ),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )

    assert {result.class_id: result.size_xyz_m for result in results} == expected


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
        [0.1, 0.2, 0.3, 0.4]
    )
    assert len({result.object_id for result in outputs}) == 1
    assert outputs[0].object_id == "odom-object:0"


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
    assert result.confidence == pytest.approx(2.0 / 9.0)
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
    assert first.object_id is None and second.object_id is None
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
    assert first.object_id != second.object_id
    assert first.object_id == "odom-object:0"
    assert second.object_id == "odom-object:1"


def test_perception_3d_order_reversal_keeps_moderately_moving_tracks_distinct() -> None:
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
    # 输入顺序反转，但每个ID只作小幅移动，身份应保持而不依赖输入顺序。
    crossed = (
        _estimator_detection(
            bbox_xyxy=(349.0, 239.0, 351.0, 241.0),
            timestamp_ns=101,
            track_id=32,
        ),
        _estimator_detection(
            bbox_xyxy=(289.0, 239.0, 291.0, 241.0),
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
    assert crossed_results[0].valid and crossed_results[1].valid
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
    assert all(
        isinstance(result.object_id, str) and bool(result.object_id)
        for result in results
    )
    assert all(result.frame_id == "odom" for result in results)
    assert all(result.orientation_xyzw is None for result in results)
    assert all(
        result.size_xyz_m == pytest.approx((0.24, 0.16, 0.19))
        for result in results
    )
    assert [result.position_xyz[2] for result in results] == pytest.approx(
        (1.095, 1.095, 1.095)
    )


def test_median_depth_m_samples_only_inside_bbox() -> None:
    image = np.full((24, 24), 5000.0, dtype=float)
    image[10:12, 10:12] = 1000.0

    depth_m = median_depth_m(
        _depth(image),
        (10.0, 10.0, 12.0, 12.0),
        radius_px=4,
    )

    assert depth_m == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bbox",
    (
        (10.0, 10.0, 10.0, 12.0),
        (10.1, 10.1, 10.9, 10.9),
        (30.0, 30.0, 32.0, 32.0),
    ),
)
def test_median_depth_m_rejects_bbox_smaller_than_one_pixel(
    bbox: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValueError, match="bbox"):
        median_depth_m(_depth(np.ones((24, 24))), bbox, radius_px=4)


def test_perception_3d_small_target_avoids_background_in_estimation() -> None:
    image = np.full((480, 640), 5000.0, dtype=float)
    image[239:241, 319:321] = 1000.0
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.002)},
    )

    result = estimator.estimate(
        (_estimator_detection(track_id=81),),
        _estimator_depth(0.0, 100, image=image),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert result.valid
    assert result.position_xyz[2] == pytest.approx(1.001)
    assert result.position_xyz[2] < 2.0


def test_perception_3d_partial_occlusion_at_bbox_edge() -> None:
    image = np.zeros((480, 640), dtype=float)
    image[239:243, 321:323] = 1000.0
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )

    result = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(319.0, 239.0, 323.0, 243.0),
                track_id=82,
            ),
        ),
        _estimator_depth(0.0, 100, image=image),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert result.valid
    assert result.position_xyz[2] == pytest.approx(1.1)


def test_perception_3d_detects_track_id_swap_on_crossing() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        ema_alpha=0.5,
        converge_frames=1,
        max_position_jump_m=1.0,
        ambiguity_ratio=2.0,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    first = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(279.0, 239.0, 283.0, 243.0),
                track_id=31,
            ),
            _estimator_detection(
                bbox_xyxy=(357.0, 239.0, 361.0, 243.0),
                track_id=32,
            ),
        ),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )
    old_ids = tuple(result.object_id for result in first)

    swapped = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(357.0, 239.0, 361.0, 243.0),
                timestamp_ns=101,
                track_id=31,
            ),
            _estimator_detection(
                bbox_xyxy=(279.0, 239.0, 283.0, 243.0),
                timestamp_ns=101,
                track_id=32,
            ),
        ),
        _estimator_depth(1000.0, 101),
        _intrinsics(timestamp_ns=101),
        _base(),
        _actual_joints(),
    )

    assert first[0].position_xyz[0] < 0.0 < first[1].position_xyz[0]
    assert all(not result.valid for result in swapped)
    assert all(
        "ID交换" in result.failure_reason or "一致性校验失败" in result.failure_reason
        for result in swapped
    )
    resumed = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(279.0, 239.0, 283.0, 243.0),
                timestamp_ns=102,
                track_id=31,
            ),
            _estimator_detection(
                bbox_xyxy=(357.0, 239.0, 361.0, 243.0),
                timestamp_ns=102,
                track_id=32,
            ),
        ),
        _estimator_depth(1000.0, 102),
        _intrinsics(timestamp_ns=102),
        _base(),
        _actual_joints(),
    )
    assert tuple(result.object_id for result in resumed) == old_ids


def test_perception_3d_occlusion_recovery_keeps_track() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        ema_alpha=0.5,
        converge_frames=4,
        max_track_age_s=1.0,
        max_position_jump_m=0.5,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    first = estimator.estimate(
        (_estimator_detection(track_id=41),),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]
    for timestamp_ns in (101, 102):
        assert estimator.estimate(
            (),
            _estimator_depth(0.0, timestamp_ns),
            _intrinsics(timestamp_ns=timestamp_ns),
            _base(),
            _actual_joints(),
        ) == ()
    recovered = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(320.0, 239.0, 322.0, 241.0),
                timestamp_ns=103,
                track_id=41,
            ),
        ),
        _estimator_depth(1100.0, 103),
        _intrinsics(timestamp_ns=103),
        _base(),
        _actual_joints(),
    )[0]

    assert first.valid and recovered.valid
    assert recovered.position_xyz[2] == pytest.approx(1.15)
    assert "ID交换" not in recovered.failure_reason
    assert recovered.object_id == first.object_id
    assert recovered.confidence == pytest.approx(first.confidence * 2.0)


def test_perception_3d_complete_overlap_is_fail_closed() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )

    results = estimator.estimate(
        (
            _estimator_detection(track_id=51),
            _estimator_detection(
                bbox_xyxy=(319.5, 239.0, 321.5, 241.0),
                track_id=52,
            ),
        ),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )

    assert all(not result.valid for result in results)
    assert all("无法可靠区分身份" in result.failure_reason for result in results)
    assert estimator._tracks == {}


def test_perception_3d_duplicate_track_id_fails_both_detections_closed() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    results = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(279.0, 239.0, 283.0, 243.0),
                track_id=7,
            ),
            _estimator_detection(
                bbox_xyxy=(357.0, 239.0, 361.0, 243.0),
                track_id=7,
            ),
        ),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )

    assert len(results) == 2
    assert all(not result.valid for result in results)
    assert all("轨迹ID重复" in result.failure_reason for result in results)
    assert estimator._tracks == {}


def _reassociation_estimator(**overrides: object) -> Perception3DEstimator:
    parameters: dict[str, object] = {
        "converge_frames": 1,
        "max_track_age_s": 0.5,
        "object_dimensions_m": {
            "pink": (0.1, 0.1, 0.2),
            "yellow": (0.1, 0.1, 0.2),
        },
        "object_local_size_xyz_m": {
            "pink": (0.1, 0.1, 0.2),
            "yellow": (0.1, 0.1, 0.2),
        },
        "reassociation_enabled": True,
        "reassociation_max_age_s": 10.0,
        "reassociation_max_distance_m": 0.3,
        "reassociation_size_tolerance_ratio": 0.25,
        "reassociation_ambiguity_ratio": 1.2,
        "max_identity_tracks": 16,
    }
    parameters.update(overrides)
    return Perception3DEstimator(
        _OffsetTransformProvider(),
        **parameters,  # type: ignore[arg-type]
    )


def test_perception_3d_changed_2d_id_recovers_stable_3d_identity() -> None:
    estimator = _reassociation_estimator()
    first = estimator.estimate(
        (_estimator_detection(timestamp_ns=1_000_000_000, track_id=1),),
        _estimator_depth(1000.0, 1_000_000_000),
        _intrinsics(timestamp_ns=1_000_000_000),
        _base(),
        _actual_joints(),
    )[0]
    assert estimator.estimate(
        (),
        _estimator_depth(0.0, 2_000_000_000),
        _intrinsics(timestamp_ns=2_000_000_000),
        _base(),
        _actual_joints(),
    ) == ()
    recovered = estimator.estimate(
        (_estimator_detection(timestamp_ns=3_000_000_000, track_id=99),),
        _estimator_depth(1200.0, 3_000_000_000),
        _intrinsics(timestamp_ns=3_000_000_000),
        _base(),
        _actual_joints(),
    )[0]

    assert first.valid and recovered.valid
    assert recovered.object_id == first.object_id
    assert recovered.object_id != "stable:99"
    # 新片段必须从当前帧中心重建，不能把旧EMA混入1.3m结果。
    assert recovered.position_xyz[2] == pytest.approx(1.3)
    assert recovered.confidence == pytest.approx(first.confidence)


def test_perception_3d_reassociation_restarts_point_cloud_pose_convergence() -> None:
    estimator = _reassociation_estimator(
        pose_refinement_enabled=True,
        pose_required_frames=2,
    )
    candidate = ((0.0, 0.0, 1.1), (0.0, 0.0, 0.0, 1.0))
    estimator._point_cloud_pose_candidate = lambda *args: candidate  # type: ignore[method-assign]
    first = estimator.estimate(
        (_estimator_detection(timestamp_ns=1_000_000_000, track_id=1),),
        _estimator_depth(1000.0, 1_000_000_000),
        _intrinsics(timestamp_ns=1_000_000_000),
        _base(),
        _actual_joints(),
    )[0]
    converged = estimator.estimate(
        (_estimator_detection(timestamp_ns=1_100_000_000, track_id=1),),
        _estimator_depth(1000.0, 1_100_000_000),
        _intrinsics(timestamp_ns=1_100_000_000),
        _base(),
        _actual_joints(),
    )[0]
    estimator.estimate(
        (),
        _estimator_depth(0.0, 2_000_000_000),
        _intrinsics(timestamp_ns=2_000_000_000),
        _base(),
        _actual_joints(),
    )
    recovered = estimator.estimate(
        (_estimator_detection(timestamp_ns=3_000_000_000, track_id=2),),
        _estimator_depth(1000.0, 3_000_000_000),
        _intrinsics(timestamp_ns=3_000_000_000),
        _base(),
        _actual_joints(),
    )[0]

    assert first.orientation_xyzw is None
    assert converged.orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert recovered.object_id == first.object_id
    assert recovered.orientation_xyzw is None
    assert "pending (1/2)" in recovered.failure_reason


def test_perception_3d_reassociates_two_separated_objects_one_to_one_order_independent() -> None:
    def run(reverse: bool) -> dict[float, str | None]:
        estimator = _reassociation_estimator()
        first = (
            _estimator_detection(
                bbox_xyxy=(279.0, 239.0, 283.0, 243.0),
                timestamp_ns=1_000_000_000,
                track_id=1,
            ),
            _estimator_detection(
                bbox_xyxy=(357.0, 239.0, 361.0, 243.0),
                timestamp_ns=1_000_000_000,
                track_id=2,
            ),
        )
        estimator.estimate(
            tuple(reversed(first)) if reverse else first,
            _estimator_depth(1000.0, 1_000_000_000),
            _intrinsics(timestamp_ns=1_000_000_000),
            _base(),
            _actual_joints(),
        )
        estimator.estimate(
            (),
            _estimator_depth(0.0, 2_000_000_000),
            _intrinsics(timestamp_ns=2_000_000_000),
            _base(),
            _actual_joints(),
        )
        current = (
            _estimator_detection(
                bbox_xyxy=(357.0, 239.0, 361.0, 243.0),
                timestamp_ns=3_000_000_000,
                track_id=102,
            ),
            _estimator_detection(
                bbox_xyxy=(279.0, 239.0, 283.0, 243.0),
                timestamp_ns=3_000_000_000,
                track_id=101,
            ),
        )
        supplied = tuple(reversed(current)) if reverse else current
        results = estimator.estimate(
            supplied,
            _estimator_depth(1000.0, 3_000_000_000),
            _intrinsics(timestamp_ns=3_000_000_000),
            _base(),
            _actual_joints(),
        )
        assert all(result.valid for result in results)
        return {
            round(result.position_xyz[0], 6): result.object_id
            for result in results
        }

    assert run(False) == run(True)


def test_perception_3d_reassociation_ambiguous_histories_fail_closed() -> None:
    estimator = _reassociation_estimator()
    estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(299.0, 239.0, 303.0, 243.0),
                timestamp_ns=1_000_000_000,
                track_id=1,
            ),
            _estimator_detection(
                bbox_xyxy=(337.0, 239.0, 341.0, 243.0),
                timestamp_ns=1_000_000_000,
                track_id=2,
            ),
        ),
        _estimator_depth(1000.0, 1_000_000_000),
        _intrinsics(timestamp_ns=1_000_000_000),
        _base(),
        _actual_joints(),
    )
    estimator.estimate(
        (),
        _estimator_depth(0.0, 2_000_000_000),
        _intrinsics(timestamp_ns=2_000_000_000),
        _base(),
        _actual_joints(),
    )
    ambiguous = estimator.estimate(
        (_estimator_detection(timestamp_ns=3_000_000_000, track_id=3),),
        _estimator_depth(1000.0, 3_000_000_000),
        _intrinsics(timestamp_ns=3_000_000_000),
        _base(),
        _actual_joints(),
    )[0]

    assert not ambiguous.valid
    assert ambiguous.object_id is None
    assert "候选歧义" in ambiguous.failure_reason


def test_perception_3d_two_observations_competing_for_history_fail_closed() -> None:
    estimator = _reassociation_estimator()
    first = estimator.estimate(
        (_estimator_detection(timestamp_ns=1_000_000_000, track_id=1),),
        _estimator_depth(1000.0, 1_000_000_000),
        _intrinsics(timestamp_ns=1_000_000_000),
        _base(),
        _actual_joints(),
    )[0]
    estimator.estimate(
        (),
        _estimator_depth(0.0, 2_000_000_000),
        _intrinsics(timestamp_ns=2_000_000_000),
        _base(),
        _actual_joints(),
    )
    results = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(317.0, 239.0, 319.0, 241.0),
                timestamp_ns=3_000_000_000,
                track_id=2,
            ),
            _estimator_detection(
                bbox_xyxy=(321.0, 239.0, 323.0, 241.0),
                timestamp_ns=3_000_000_000,
                track_id=3,
            ),
        ),
        _estimator_depth(1000.0, 3_000_000_000),
        _intrinsics(timestamp_ns=3_000_000_000),
        _base(),
        _actual_joints(),
    )

    assert first.valid
    assert all(not result.valid for result in results)
    assert all("多个当前观测竞争" in result.failure_reason for result in results)


@pytest.mark.parametrize("gate", ("class", "distance", "age", "size"))
def test_perception_3d_reassociation_gates_create_new_identity(gate: str) -> None:
    estimator = _reassociation_estimator(
        reassociation_max_age_s=2.0 if gate == "age" else 10.0,
        reassociation_max_distance_m=0.1 if gate == "distance" else 0.3,
    )
    first = estimator.estimate(
        (_estimator_detection(timestamp_ns=1_000_000_000, track_id=1),),
        _estimator_depth(1000.0, 1_000_000_000),
        _intrinsics(timestamp_ns=1_000_000_000),
        _base(),
        _actual_joints(),
    )[0]
    estimator.estimate(
        (),
        _estimator_depth(0.0, 2_000_000_000),
        _intrinsics(timestamp_ns=2_000_000_000),
        _base(),
        _actual_joints(),
    )
    if gate == "size":
        # 模拟后续经批准的运行时尺寸来源发生明显变化；只观察公开输出身份。
        estimator._local_sizes["pink"] = (0.2, 0.2, 0.4)
    timestamp_ns = 4_000_000_000 if gate == "age" else 3_000_000_000
    current_class = "yellow" if gate == "class" else "pink"
    depth_mm = 1500.0 if gate == "distance" else 1000.0
    current = estimator.estimate(
        (
            _estimator_detection(
                current_class, timestamp_ns=timestamp_ns, track_id=2
            ),
        ),
        _estimator_depth(depth_mm, timestamp_ns),
        _intrinsics(timestamp_ns=timestamp_ns),
        _base(),
        _actual_joints(),
    )[0]

    assert first.valid and current.valid
    assert current.object_id != first.object_id


def test_perception_3d_stale_frame_does_not_change_identity_history() -> None:
    estimator = _reassociation_estimator()
    first = estimator.estimate(
        (_estimator_detection(timestamp_ns=2_000_000_000, track_id=1),),
        _estimator_depth(1000.0, 2_000_000_000),
        _intrinsics(timestamp_ns=2_000_000_000),
        _base(),
        _actual_joints(),
    )[0]
    stale = estimator.estimate(
        (_estimator_detection(timestamp_ns=1_000_000_000, track_id=2),),
        _estimator_depth(1000.0, 1_000_000_000),
        _intrinsics(timestamp_ns=1_000_000_000),
        _base(),
        _actual_joints(),
    )[0]
    resumed = estimator.estimate(
        (_estimator_detection(timestamp_ns=3_000_000_000, track_id=1),),
        _estimator_depth(1000.0, 3_000_000_000),
        _intrinsics(timestamp_ns=3_000_000_000),
        _base(),
        _actual_joints(),
    )[0]

    assert not stale.valid and "陈旧感知帧" in stale.failure_reason
    assert resumed.object_id == first.object_id


def test_perception_3d_identity_capacity_fails_without_evicting_current_frame() -> None:
    estimator = _reassociation_estimator(max_identity_tracks=1)
    results = estimator.estimate(
        (
            _estimator_detection(
                bbox_xyxy=(279.0, 239.0, 283.0, 243.0), track_id=1
            ),
            _estimator_detection(
                bbox_xyxy=(357.0, 239.0, 361.0, 243.0), track_id=2
            ),
        ),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )

    assert sum(result.valid for result in results) == 1
    assert any("缓存已满" in result.failure_reason for result in results)


def test_perception_3d_identity_capacity_evicts_oldest_noncurrent_history() -> None:
    estimator = _reassociation_estimator(
        max_identity_tracks=2,
        reassociation_max_distance_m=0.1,
    )
    first = estimator.estimate(
        (_estimator_detection(timestamp_ns=1_000_000_000, track_id=1),),
        _estimator_depth(1000.0, 1_000_000_000),
        _intrinsics(timestamp_ns=1_000_000_000),
        _base(),
        _actual_joints(),
    )[0]
    second = estimator.estimate(
        (
            _estimator_detection(
                "yellow", timestamp_ns=1_100_000_000, track_id=2
            ),
        ),
        _estimator_depth(1000.0, 1_100_000_000),
        _intrinsics(timestamp_ns=1_100_000_000),
        _base(),
        _actual_joints(),
    )[0]
    estimator.estimate(
        (),
        _estimator_depth(0.0, 2_000_000_000),
        _intrinsics(timestamp_ns=2_000_000_000),
        _base(),
        _actual_joints(),
    )
    third = estimator.estimate(
        (_estimator_detection(timestamp_ns=3_000_000_000, track_id=3),),
        _estimator_depth(1500.0, 3_000_000_000),
        _intrinsics(timestamp_ns=3_000_000_000),
        _base(),
        _actual_joints(),
    )[0]
    old_location_again = estimator.estimate(
        (_estimator_detection(timestamp_ns=4_000_000_000, track_id=4),),
        _estimator_depth(1000.0, 4_000_000_000),
        _intrinsics(timestamp_ns=4_000_000_000),
        _base(),
        _actual_joints(),
    )[0]

    assert first.valid and second.valid and third.valid and old_location_again.valid
    assert third.object_id not in {first.object_id, second.object_id}
    assert old_location_again.object_id != first.object_id


def test_perception_3d_reset_starts_new_identity_lifecycle_and_smaller_time() -> None:
    estimator = _reassociation_estimator()
    first = estimator.estimate(
        (_estimator_detection(timestamp_ns=5_000_000_000, track_id=7),),
        _estimator_depth(1000.0, 5_000_000_000),
        _intrinsics(timestamp_ns=5_000_000_000),
        _base(),
        _actual_joints(),
    )[0]
    estimator.reset_tracks()
    restarted = estimator.estimate(
        (_estimator_detection(timestamp_ns=100, track_id=8),),
        _estimator_depth(1200.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert first.valid and restarted.valid
    assert restarted.object_id == "odom-object:0"
    assert restarted.position_xyz[2] == pytest.approx(1.3)


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("reassociation_enabled", 1),
        ("reassociation_max_age_s", float("nan")),
        ("reassociation_max_distance_m", True),
        ("reassociation_size_tolerance_ratio", 1.1),
        ("reassociation_ambiguity_ratio", 1.0),
        ("max_identity_tracks", True),
        ("max_identity_tracks", 0),
    ),
)
def test_perception_3d_rejects_invalid_reassociation_parameters(
    keyword: str, value: object
) -> None:
    with pytest.raises(ValueError, match=keyword):
        Perception3DEstimator(
            _OffsetTransformProvider(),
            **{keyword: value},
        )


def test_perception_3d_rejects_reassociation_window_shorter_than_active_window() -> None:
    with pytest.raises(ValueError, match="reassociation_max_age_s"):
        Perception3DEstimator(
            _OffsetTransformProvider(),
            reassociation_enabled=True,
            max_track_age_s=1.0,
            reassociation_max_age_s=0.1,
        )


def test_camera_transform_rejects_frame_id_mismatch() -> None:
    provider = CameraTransformProvider(output_frame="odom")
    provider._fk = _RecordingFK()

    with pytest.raises(ValueError, match="frame_id.*map.*odom|frame_id.*odom.*map"):
        provider.camera_to_output(
            (0.0, 0.0, 1.0),
            _base(frame_id="map"),
            _actual_joints(),
        )
    with pytest.raises(ValueError, match="frame_id.*为空"):
        provider.camera_to_output(
            (0.0, 0.0, 1.0),
            _base(frame_id=""),
            _actual_joints(),
        )


def test_perception_3d_estimate_rejects_base_frame_mismatch() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(output_frame="odom"),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )

    result = estimator.estimate(
        (_estimator_detection(track_id=61),),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(frame_id="map"),
        _actual_joints(),
    )[0]

    assert not result.valid
    assert "frame_id" in result.failure_reason
    assert "map" in result.failure_reason and "odom" in result.failure_reason


def test_perception_3d_degraded_compensation_reduces_confidence() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        heuristic_center_reliability=0.5,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )

    result = estimator.estimate(
        (_estimator_detection(confidence=0.8, track_id=71),),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert result.valid
    assert result.confidence == pytest.approx(0.8 * 1.0 * 1.0 * 0.5)
    assert "heuristic center approximation" in result.failure_reason


def test_perception_3d_strict_compensation_rejects_heuristic_center() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        center_compensation_mode="strict",
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )

    result = estimator.estimate(
        (_estimator_detection(track_id=72),),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert not result.valid
    assert "not validated" in result.failure_reason


@pytest.mark.parametrize("mode", ("degraded", "strict"))
def test_perception_3d_unknown_dimension_still_invalid(mode: str) -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        center_compensation_mode=mode,
        object_dimensions_m={"yellow": (0.1, 0.1, 0.2)},
    )

    result = estimator.estimate(
        (_estimator_detection("pink", track_id=73),),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert not result.valid
    assert "中心补偿失败" in result.failure_reason


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("ambiguity_ratio", 1.0),
        ("ambiguity_ratio", True),
        ("center_compensation_mode", "fallback"),
        ("heuristic_center_reliability", 0.0),
        ("heuristic_center_reliability", True),
    ),
)
def test_perception_3d_rejects_invalid_new_safety_parameters(
    keyword: str, value: object
) -> None:
    with pytest.raises(ValueError, match=keyword):
        Perception3DEstimator(
            _OffsetTransformProvider(),
            **{keyword: value},
        )


class _CountingOffsetTransformProvider(_OffsetTransformProvider):
    """记录每帧 FK 位姿入口与廉价逐点变换的调用次数。"""

    def __init__(
        self,
        offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        output_frame: str = "odom",
    ) -> None:
        super().__init__(offset_xyz, output_frame)
        self.pose_calls = 0
        self.point_calls = 0

    def compute_head_camera_pose(
        self,
        base: BaseState,
        joints: RobotJointState,
    ) -> _HeadCameraPose:
        self.pose_calls += 1
        return super().compute_head_camera_pose(base, joints)

    def transform_camera_point(
        self,
        camera_point_xyz: tuple[float, float, float],
        pose: _HeadCameraPose,
    ) -> tuple[float, float, float]:
        self.point_calls += 1
        return super().transform_camera_point(camera_point_xyz, pose)


class _LegacyCameraToOutputProvider(CameraTransformProvider):
    """模拟只实现优化前公共变换入口的既有 Provider。"""

    def __init__(self, offset_xyz: tuple[float, float, float]) -> None:
        super().__init__(output_frame="odom")
        self.offset_xyz = offset_xyz
        self.calls = 0

    def camera_to_output(
        self,
        camera_point_xyz: tuple[float, float, float],
        base: BaseState,
        joints: RobotJointState,
    ) -> tuple[float, float, float]:
        del joints
        if base.frame_id != self.output_frame:
            raise ValueError("BaseState.frame_id 与 output_frame 不一致")
        self.calls += 1
        return tuple(
            camera_point_xyz[index] + self.offset_xyz[index]
            for index in range(3)
        )


class _UnconvertibleDepthImage:
    def __array__(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise TypeError("测试图像禁止转换")


def test_perception_3d_legacy_camera_to_output_override_remains_compatible() -> None:
    provider = _LegacyCameraToOutputProvider((1.0, 2.0, 3.0))
    estimator = Perception3DEstimator(
        provider,
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )

    result = estimator.estimate(
        (_estimator_detection(track_id=160),),
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert result.valid
    assert result.position_xyz == pytest.approx((1.0, 2.0, 4.1))
    assert provider.calls == 2


def test_median_depth_m_validates_metadata_before_array_conversion() -> None:
    bad_image = _UnconvertibleDepthImage()
    invalid_depth = DepthFrame(
        bad_image,
        0.001,
        "camera_optical_frame",
        100,
        valid=False,
        failure_reason="传感器无效",
    )
    valid_metadata = DepthFrame(
        bad_image,
        0.001,
        "camera_optical_frame",
        100,
    )
    invalid_scale = DepthFrame(
        bad_image,
        -0.001,
        "camera_optical_frame",
        100,
    )

    with pytest.raises(ValueError, match="深度帧无效：传感器无效"):
        median_depth_m(invalid_depth, (0.0, 0.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="radius_px"):
        median_depth_m(valid_metadata, (0.0, 0.0, 2.0, 2.0), -1)
    with pytest.raises(ValueError, match="x1>x0"):
        median_depth_m(valid_metadata, (1.0, 1.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="unit_scale_m"):
        median_depth_m(invalid_scale, (0.0, 0.0, 2.0, 2.0))


def test_perception_3d_validates_depth_scale_before_array_conversion() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    depth = DepthFrame(
        _UnconvertibleDepthImage(),
        -0.001,
        "camera_optical_frame",
        100,
    )

    result = estimator.estimate(
        (_estimator_detection(track_id=161),),
        depth,
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )[0]

    assert not result.valid
    assert "depth.unit_scale_m" in result.failure_reason
    assert "depth.image" not in result.failure_reason


def test_perception_3d_computes_head_camera_pose_once_for_five_targets() -> None:
    provider = _CountingOffsetTransformProvider()
    estimator = Perception3DEstimator(
        provider,
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    detections = tuple(
        _estimator_detection(
            bbox_xyxy=(
                258.0 + 30.0 * index,
                238.0,
                262.0 + 30.0 * index,
                242.0,
            ),
            track_id=100 + index,
        )
        for index in range(5)
    )

    results = estimator.estimate(
        detections,
        _estimator_depth(1000.0, 100),
        _intrinsics(timestamp_ns=100),
        _base(),
        _actual_joints(),
    )

    assert len(results) == 5
    assert all(result.valid for result in results)
    assert provider.pose_calls == 1
    # 每个目标分别变换一次表面点和一次补偿中心，但二者都不再触发 FK。
    assert provider.point_calls == 10


def test_perception_3d_shared_pose_preserves_legacy_per_detection_results() -> None:
    offset = (1.0, 2.0, 3.0)
    optimized_provider = _CountingOffsetTransformProvider(offset)
    legacy_provider = _OffsetTransformProvider(offset)
    estimator = Perception3DEstimator(
        optimized_provider,
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    detections = tuple(
        _estimator_detection(
            bbox_xyxy=(center_x - 2.0, 238.0, center_x + 2.0, 242.0),
            confidence=0.8,
            track_id=120 + index,
        )
        for index, center_x in enumerate((280.0, 320.0, 360.0))
    )
    base = _base()
    joints = _actual_joints()
    intrinsics = _intrinsics(timestamp_ns=100)
    expected_positions = []
    for detection in detections:
        x0, y0, x1, y1 = detection.bbox_xyxy
        surface = project_pixel_to_camera(
            (x0 + x1) / 2.0,
            (y0 + y1) / 2.0,
            1.0,
            intrinsics,
        )
        scale = 1.1 / surface[2]
        compensated = (
            surface[0] * scale,
            surface[1] * scale,
            1.1,
        )
        expected_positions.append(
            legacy_provider.camera_to_output(compensated, base, joints)
        )

    results = estimator.estimate(
        detections,
        _estimator_depth(1000.0, 100),
        intrinsics,
        base,
        joints,
    )

    assert optimized_provider.pose_calls == 1
    for result, expected_position in zip(results, expected_positions):
        assert result.position_xyz == pytest.approx(expected_position)
        assert result.confidence == pytest.approx(0.8 * 1.0 * 1.0 * 0.5)
        assert result.valid
        assert (
            result.failure_reason
            == "heuristic center approximation (surface-to-center not validated)"
        )


def test_perception_3d_samples_each_detection_depth_window_once() -> None:
    estimator = Perception3DEstimator(
        _OffsetTransformProvider(),
        converge_frames=1,
        object_dimensions_m={"pink": (0.1, 0.1, 0.2)},
    )
    detections = tuple(
        _estimator_detection(
            bbox_xyxy=(
                278.0 + 24.0 * index,
                238.0,
                282.0 + 24.0 * index,
                242.0,
            ),
            track_id=140 + index,
        )
        for index in range(4)
    )

    with patch.object(
        perception_3d_module,
        "_depth_window_statistics",
        wraps=perception_3d_module._depth_window_statistics,
    ) as statistics:
        results = estimator.estimate(
            detections,
            _estimator_depth(1000.0, 100),
            _intrinsics(timestamp_ns=100),
            _base(),
            _actual_joints(),
        )

    assert all(result.valid for result in results)
    assert statistics.call_count == len(detections)


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
    if not frame_id:
        with pytest.raises(ValueError, match="frame_id"):
            _arm_target(frame_id)
        return
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


def _damaged_pose(
    position: object = (0.1, 0.2, 0.3),
    orientation: object = (0.0, 0.0, 0.0, 1.0),
    frame_id: object = "footprint",
) -> Pose3D:
    pose = object.__new__(Pose3D)
    object.__setattr__(pose, "position_xyz", position)
    object.__setattr__(pose, "orientation_xyzw", orientation)
    object.__setattr__(pose, "frame_id", frame_id)
    return pose


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
    pose = _damaged_pose(position=position)
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
    pose = _damaged_pose(orientation=orientation)
    with pytest.raises(ValueError, match="orientation_xyzw"):
        _pose_to_matrix(pose, np)


def test_pose_to_matrix_rejects_zero_norm_quaternion() -> None:
    pose = _damaged_pose(orientation=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="范数为零"):
        _pose_to_matrix(pose, np)


def test_pose_to_matrix_normalizes_identity_quaternion_and_translation() -> None:
    pose = Pose3D((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 2.0), "footprint")
    assert pose.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    matrix = _pose_to_matrix(pose, np)
    assert matrix.shape == (4, 4)
    assert np.isfinite(matrix).all()
    assert matrix[:3, :3] == pytest.approx(np.eye(3))
    assert matrix[:3, 3] == pytest.approx((0.1, -0.2, 0.3))
    assert matrix[3, :] == pytest.approx((0.0, 0.0, 0.0, 1.0))


@pytest.mark.parametrize("position", [(math.nan, 0, 0), (True, 0, 0)])
def test_pose3d_constructor_rejects_invalid_position(position: object) -> None:
    with pytest.raises(ValueError, match="position_xyz"):
        Pose3D(position, (0, 0, 0, 1), "footprint")  # type: ignore[arg-type]


def test_strict_vectors_reject_numeric_strings_and_normalize_integers() -> None:
    with pytest.raises(ValueError, match="numbers.Real"):
        Pose3D(("1", "2", "3"), (0, 0, 0, 1), "footprint")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numbers.Real"):
        Pose3D((1, 2, 3), ("0", "0", "0", "1"), "footprint")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numbers.Real"):
        RigidTransform3D(
            "camera", "world", ("1", "2", "3"), (0, 0, 0, 1), 1, True
        )  # type: ignore[arg-type]
    pose = Pose3D((1, 2, 3), (0, 0, 0, 2), "footprint")
    assert pose.position_xyz == (1.0, 2.0, 3.0)
    assert pose.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_pose3d_constructor_rejects_zero_quaternion_and_empty_frame() -> None:
    with pytest.raises(ValueError, match="范数"):
        Pose3D((0, 0, 0), (0, 0, 0, 0), "footprint")
    with pytest.raises(ValueError, match="frame_id"):
        Pose3D((0, 0, 0), (0, 0, 0, 1), "  ")


def test_valid_targets_reject_damaged_pose_objects() -> None:
    damaged = _damaged_pose(position=(math.nan, 0.0, 0.0))
    with pytest.raises(ValueError, match="损坏"):
        GraspTarget(
            damaged, damaged, damaged, damaged, damaged, damaged, damaged, damaged,
            None, 0.8,
        )
    with pytest.raises(ValueError, match="损坏"):
        PlaceTarget(
            damaged, damaged, damaged, damaged, damaged, damaged, damaged, 0.0
        )


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


class _VisionStamp:
    def __init__(self) -> None:
        self.sec = 0
        self.nanosec = 0


class _VisionHeader:
    def __init__(self) -> None:
        self.frame_id = ""
        self.stamp = _VisionStamp()


class _VisionVector3:
    def __init__(self) -> None:
        self.x = self.y = self.z = 0.0


class _VisionQuaternion(_VisionVector3):
    def __init__(self) -> None:
        super().__init__()
        # geometry_msgs/Quaternion 的真实 ROS 默认构造值为单位四元数。
        self.w = 1.0


class _VisionPose:
    def __init__(self) -> None:
        self.position = _VisionVector3()
        self.orientation = _VisionQuaternion()


class _VisionHypothesis:
    def __init__(self) -> None:
        self.class_id = ""
        self.score = 0.0


class _VisionResult:
    def __init__(self) -> None:
        self.hypothesis = _VisionHypothesis()
        self.pose = SimpleNamespace(pose=_VisionPose())


class _VisionDetection:
    def __init__(self) -> None:
        self.header = _VisionHeader()
        self.id = ""
        self.bbox = SimpleNamespace(center=_VisionPose(), size=_VisionVector3())
        self.results: list[object] = []


class _VisionArray:
    def __init__(self) -> None:
        self.header = _VisionHeader()
        self.detections: list[object] = []


def _vision_types() -> SimpleNamespace:
    return SimpleNamespace(
        Detection3DArray=_VisionArray,
        Detection3D=_VisionDetection,
        ObjectHypothesisWithPose=_VisionResult,
    )


def _gripper_config_fields() -> dict[str, object]:
    return {
        "max_slide_waypoint_delta_m": 0.1,
        "max_arm_waypoint_delta_rad": 0.2,
        "max_gripper_waypoint_delta": 0.1,
        "left_gripper_min": 0.0,
        "left_gripper_max": 1.0,
        "right_gripper_min": 0.0,
        "right_gripper_max": 1.0,
        "left_gripper_open": 0.9,
        "left_gripper_closed": 0.2,
        "right_gripper_open": 0.9,
        "right_gripper_closed": 0.2,
        "gripper_verified_in_official_environment": True,
    }


def test_task_spec_valid_and_invalid_contracts_are_distinct() -> None:
    invalid = TaskSpec(
        task_id=3,
        instruction="无法解析",
        target_kind="",
        target_body="",
        target_color="",
        valid=False,
        failure_reason="缺少官方放置字段",
    )
    assert invalid.place_world_xyz is None and invalid.place_frame_id == ""

    with pytest.raises(ValueError, match="target_kind"):
        replace(invalid, valid=True, failure_reason="")
    with pytest.raises(ValueError, match="place_frame_id"):
        TaskSpec(3, "x", "box", "box", "pink", "table_point", (1, 2, 3), "", 0.1)


@pytest.mark.parametrize(
    "field_name", ["instruction", "target_kind", "target_body", "target_color"]
)
def test_valid_task_spec_requires_all_official_identity_fields(field_name: str) -> None:
    values = {
        "task_id": 1,
        "instruction": "把粉色箱体放到桌面点",
        "target_kind": "cuboid_box",
        "target_body": "box_pink",
        "target_color": "pink",
        "place_type": "table_point",
        "place_world_xyz": (1.0, 2.0, 0.8),
        "place_frame_id": "world",
        "place_radius": 0.1,
    }
    values[field_name] = ""
    with pytest.raises(ValueError, match=field_name):
        TaskSpec(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("place_type", ["shelf_point", "table_point"])
def test_task_spec_accepts_only_official_point_types(place_type: str) -> None:
    task = TaskSpec(
        1, "move", "box", "box", "pink", place_type,
        (1.0, 2.0, 3.0), "world", 0.1,
    )
    assert task.place_type == place_type
    with pytest.raises(ValueError, match="不得携带"):
        replace(task, direction="left")


def test_visual_verifier_confirms_same_target_moves_up_during_test_lift() -> None:
    result = _visual_verifier().verify_test_lift(
        _object_observation((1.0, 2.0, 0.50), 100_000_000),
        _object_observation((1.01, 2.0, 0.54), 200_000_000),
        timestamp_ns=210_000_000,
    )

    assert result.success
    assert result.is_grasped
    assert result.confidence == pytest.approx(0.8)
    assert "竖直位移=0.0400m" in result.visual_evidence
    assert "仅包含视觉证据" in result.effort_evidence


def test_visual_verifier_completed_negative_judgment_is_not_process_failure() -> None:
    result = _visual_verifier().verify_test_lift(
        _object_observation((1.0, 2.0, 0.50), 100_000_000),
        _object_observation((1.0, 2.0, 0.51), 200_000_000),
        timestamp_ns=210_000_000,
    )

    assert result.success
    assert not result.is_grasped
    assert result.failure_reason == ""


@pytest.mark.parametrize(
    ("before", "after", "now_ns", "reason"),
    (
        (
            _object_observation((0.0, 0.0, 0.0), 100, object_id="pink:1"),
            _object_observation((0.0, 0.0, 0.1), 200, object_id="pink:2"),
            300,
            "身份",
        ),
        (
            _object_observation((0.0, 0.0, 0.0), 200),
            _object_observation((0.0, 0.0, 0.1), 100),
            300,
            "时间戳",
        ),
        (
            _object_observation((0.0, 0.0, 0.0), 100, valid=False),
            _object_observation((0.0, 0.0, 0.1), 200),
            300,
            "无效",
        ),
        (
            _object_observation((0.0, 0.0, 0.0), 100_000_000),
            _object_observation((0.0, 0.0, 0.1), 200_000_000),
            800_000_001,
            "过期",
        ),
    ),
)
def test_visual_verifier_fails_closed_on_unusable_lift_evidence(
    before: ObjectEstimate3D,
    after: ObjectEstimate3D,
    now_ns: int,
    reason: str,
) -> None:
    result = _visual_verifier().verify_test_lift(
        before, after, timestamp_ns=now_ns
    )

    assert not result.success
    assert not result.is_grasped
    assert result.confidence == 0.0
    assert reason in result.failure_reason


@pytest.mark.parametrize(
    ("observation_kwargs", "reason"),
    (
        ({"confidence": 0.0}, "置信度"),
        ({"frame_id": "camera_color_optical_frame"}, "frame必须为odom"),
    ),
)
def test_visual_verifier_fails_closed_on_low_confidence_or_wrong_frame_lift(
    observation_kwargs: dict[str, object], reason: str
) -> None:
    result = _visual_verifier().verify_test_lift(
        _object_observation((0.0, 0.0, 0.0), 100, **observation_kwargs),
        _object_observation((0.0, 0.0, 0.1), 200),
        timestamp_ns=300,
    )

    assert not result.success
    assert not result.is_grasped
    assert reason in result.failure_reason


def test_visual_verifier_returns_latest_stable_post_motion_fact() -> None:
    observations = (
        _object_observation((1.000, 2.000, 0.500), 100_000_000),
        _object_observation((1.004, 2.001, 0.499), 150_000_000),
        _object_observation((1.002, 1.999, 0.501), 200_000_000),
    )

    result = _visual_verifier().stable_post_motion_observation(
        observations, timestamp_ns=210_000_000
    )

    assert result is observations[-1]
    assert result.valid


@pytest.mark.parametrize(
    ("observation_kwargs", "reason"),
    (
        ({"confidence": 0.0}, "置信度"),
        ({"frame_id": "camera_color_optical_frame"}, "frame必须为odom"),
    ),
)
def test_visual_verifier_fails_closed_on_low_confidence_or_wrong_frame_sequence(
    observation_kwargs: dict[str, object], reason: str
) -> None:
    observations = tuple(
        _object_observation((0.0, 0.0, 0.0), timestamp, **observation_kwargs)
        for timestamp in (100, 200, 300)
    )

    result = _visual_verifier().stable_post_motion_observation(
        observations, timestamp_ns=400
    )

    assert not result.valid
    assert reason in result.failure_reason


def test_visual_verifier_uses_maximum_pairwise_stationary_distance() -> None:
    observations = (
        _object_observation((-0.006, 0.0, 0.0), 100),
        _object_observation((0.0, 0.0, 0.0), 200),
        _object_observation((0.006, 0.0, 0.0), 300),
    )

    result = _visual_verifier().stable_post_motion_observation(
        observations, timestamp_ns=400
    )

    assert not result.valid
    assert "最大两两距离=0.0120m" in result.failure_reason


@pytest.mark.parametrize(
    ("maximum_distance_m", "expected_valid"),
    ((0.01, True), (0.010001, False)),
)
def test_visual_verifier_maximum_pairwise_distance_boundary(
    maximum_distance_m: float, expected_valid: bool
) -> None:
    observations = (
        _object_observation((0.0, 0.0, 0.0), 100),
        _object_observation((maximum_distance_m / 2.0, 0.0, 0.0), 200),
        _object_observation((maximum_distance_m, 0.0, 0.0), 300),
    )

    result = _visual_verifier().stable_post_motion_observation(
        observations, timestamp_ns=400
    )

    assert result.valid is expected_valid


@pytest.mark.parametrize(
    "observations",
    (
        (
            _object_observation((0.0, 0.0, 0.0), 100),
            _object_observation((0.05, 0.0, 0.0), 200),
            _object_observation((0.10, 0.0, 0.0), 300),
        ),
        (
            _object_observation((0.0, 0.0, 0.0), 100, object_id="pink:1"),
            _object_observation((0.0, 0.0, 0.0), 200, object_id="pink:2"),
            _object_observation((0.0, 0.0, 0.0), 300, object_id="pink:2"),
        ),
    ),
)
def test_visual_verifier_rejects_unstable_or_mismatched_post_motion_observations(
    observations: tuple[ObjectEstimate3D, ...],
) -> None:
    result = _visual_verifier().stable_post_motion_observation(
        observations, timestamp_ns=400
    )

    assert not result.valid
    assert result.failure_reason


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_lift_delta_m", 0.0),
        ("max_horizontal_drift_m", -0.1),
        ("max_observation_gap_s", True),
        ("minimum_observation_confidence", 0.0),
        ("required_frame_id", "camera_color_optical_frame"),
        ("min_stationary_observations", 1),
        ("max_stationary_spread_m", -0.1),
    ),
)
def test_visual_verifier_rejects_unsafe_parameters(field: str, value: object) -> None:
    kwargs = {
        "minimum_lift_delta_m": 0.03,
        "max_horizontal_drift_m": 0.02,
        "max_observation_gap_s": 0.5,
        "minimum_observation_confidence": 0.5,
        "required_frame_id": "odom",
        "min_stationary_observations": 3,
        "max_stationary_spread_m": 0.01,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        VisualObservationVerifier(**kwargs)


def test_object_estimate_optional_perception_facts_remain_optional() -> None:
    estimate = ObjectEstimate3D("pink", (1, 2, 3), 0.8, "odom", 4)
    assert estimate.object_id is None
    assert estimate.orientation_xyzw is None
    assert estimate.size_xyz_m is None
    assert not hasattr(estimate, "target_body")


def test_rigid_transform_normalizes_and_enforces_same_frame_identity() -> None:
    transform = RigidTransform3D(
        "camera", "world", (1, 2, 3), (0, 0, 0, 2), 10, True
    )
    assert transform.rotation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert RigidTransform3D(
        "world", "world", (0, 0, 0), (0, 0, 0, -2), 10, True
    ).rotation_xyzw == (0.0, 0.0, 0.0, -1.0)
    with pytest.raises(ValueError, match="同frame"):
        RigidTransform3D("world", "world", (0.01, 0, 0), (0, 0, 0, 1), 10, True)
    invalid = RigidTransform3D("camera", "world", None, None, 10, False, "TF缺失")
    assert invalid.translation_xyz is invalid.rotation_xyzw is None


def test_rigid_transform_normalizes_frame_whitespace_before_identity_check() -> None:
    transform = RigidTransform3D(
        " world ", " world ", (0, 0, 0), (0, 0, 0, 2), 10, True
    )
    assert transform.source_frame == transform.target_frame == "world"
    slash = RigidTransform3D(
        " /odom ", " footprint ", (0, 0, 0), (0, 0, 0, 1), 10, True
    )
    assert slash.source_frame == "/odom"
    assert slash.target_frame == "footprint"
    with pytest.raises(ValueError, match="同frame"):
        RigidTransform3D(
            " world ", "world", (0.01, 0, 0), (0, 0, 0, 1), 10, True
        )


def test_arm_planning_config_validates_grasp_without_place_calibration() -> None:
    config = ArmPlanningConfig(
        min_object_confidence=0.7,
        transform_max_age_ns=100,
        object_estimate_max_age_ns=100,
        joint_state_max_age_ns=100,
        planned_context_max_age_ns=100,
        pregrasp_distance_m=0.1,
        grasp_contact_offset_m=0.0,
        lift_distance_m=0.1,
        retreat_distance_m=0.1,
        pregrasp_duration_s=1.0,
        grasp_duration_s=1.0,
        lift_duration_s=1.0,
        retreat_duration_s=1.0,
        **_gripper_config_fields(),
    )
    config.validate_for_grasp()
    with pytest.raises(ValueError, match="confirmed_context_max_age_ns"):
        config.validate_for_place()


def test_arm_planning_config_validates_place_without_grasp_calibration() -> None:
    config = ArmPlanningConfig(
        transform_max_age_ns=100,
        joint_state_max_age_ns=100,
        confirmed_context_max_age_ns=100,
        preplace_height_m=0.1,
        release_offset_m=0.0,
        post_release_retreat_distance_m=0.1,
        settle_time_s=0.0,
        preplace_duration_s=1.0,
        lower_duration_s=1.0,
        release_duration_s=1.0,
        post_release_retreat_duration_s=1.0,
        **_gripper_config_fields(),
    )
    config.validate_for_place()
    with pytest.raises(ValueError, match="min_object_confidence"):
        config.validate_for_grasp()


def test_arm_planning_config_yaml_is_calibrated() -> None:
    config = yaml.safe_load((Path(__file__).parents[1] / "config/config.yaml").read_text())
    enabled, planning = _arm_planning_config_from_config(config)
    arm = config["arm_planning"]
    execution = config["arm_execution"]
    assert enabled is False
    assert isinstance(planning, ArmPlanningConfig)
    assert planning.gripper_verified_in_official_environment is True
    assert planning.left_gripper_min == planning.right_gripper_min == 0.0
    assert planning.left_gripper_max == planning.right_gripper_max == 1.0
    assert planning.left_gripper_open == planning.right_gripper_open == 1.0
    assert planning.left_gripper_closed == planning.right_gripper_closed == 0.1
    assert all(
        value is None
        for key, value in arm.items()
        if key not in {
            "enabled",
            "gripper_verified_in_official_environment",
            "left_gripper_min",
            "left_gripper_max",
            "right_gripper_min",
            "right_gripper_max",
            "left_gripper_open",
            "left_gripper_closed",
            "right_gripper_open",
            "right_gripper_closed",
        }
    )
    with pytest.raises(ValueError):
        planning.validate_for_grasp()
    with pytest.raises(ValueError):
        planning.validate_for_place()
    assert {
        key: execution[key]
        for key in (
            "feedback_max_age_ns",
            "trajectory_max_age_ns",
            "command_ttl_ns",
            "waypoint_timeout_margin_ns",
            "total_timeout_margin_ns",
            "initial_slide_error_limit_m",
            "initial_arm_error_limit_rad",
            "initial_gripper_error_limit",
        )
    } == {
        "feedback_max_age_ns": 131_175_150,
        "trajectory_max_age_ns": 76_103_892,
        "command_ttl_ns": 74_720_790,
        "waypoint_timeout_margin_ns": 2_737_458_938,
        "total_timeout_margin_ns": 9_075_055_791,
        "initial_slide_error_limit_m": 0.16,
        "initial_arm_error_limit_rad": 0.24,
        "initial_gripper_error_limit": 0.20,
    }
    assert execution["verification_timeout_ns"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"left_gripper_min": -0.1, "left_gripper_max": 1.0},
        {"right_gripper_min": 0.0, "right_gripper_max": 1.1},
        {"left_gripper_min": -1.0, "left_gripper_max": 1.0},
        {"right_gripper_min": 0.0, "right_gripper_max": 2.0},
    ],
)
def test_arm_planning_config_rejects_gripper_range_outside_official_ctrlrange(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match=r"ctrlrange \[0, 1\]"):
        ArmPlanningConfig(**overrides)


def test_arm_planning_config_accepts_official_gripper_ctrlrange() -> None:
    config = ArmPlanningConfig(
        left_gripper_min=0.0,
        left_gripper_max=1.0,
        right_gripper_min=0.0,
        right_gripper_max=1.0,
    )
    assert config.left_gripper_min == config.right_gripper_min == 0.0
    assert config.left_gripper_max == config.right_gripper_max == 1.0


@pytest.mark.parametrize("enabled", [0, 1, "false"])
def test_arm_planning_config_reader_rejects_non_bool_enabled(enabled: object) -> None:
    config = yaml.safe_load((Path(__file__).parents[1] / "config/config.yaml").read_text())
    config["arm_planning"]["enabled"] = enabled
    with pytest.raises(ValueError, match="enabled"):
        _arm_planning_config_from_config(config)


def test_arm_planning_config_reader_rejects_bad_section_unknown_and_value() -> None:
    config = yaml.safe_load((Path(__file__).parents[1] / "config/config.yaml").read_text())
    with pytest.raises(ValueError, match="config 必须是 Mapping"):
        _arm_planning_config_from_config([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Mapping"):
        _arm_planning_config_from_config({"arm_planning": []})

    config["arm_planning"]["typo_distance"] = None
    with pytest.raises(ValueError, match="未知字段"):
        _arm_planning_config_from_config(config)
    del config["arm_planning"]["typo_distance"]

    config["arm_planning"]["pregrasp_distance_m"] = -0.1
    with pytest.raises(ValueError, match="pregrasp_distance_m"):
        _arm_planning_config_from_config(config)


def test_arm_planning_config_reader_rejects_missing_explicit_null_field() -> None:
    config = yaml.safe_load((Path(__file__).parents[1] / "config/config.yaml").read_text())
    del config["arm_planning"]["pregrasp_distance_m"]
    with pytest.raises(ValueError, match="缺少显式字段"):
        _arm_planning_config_from_config(config)


def test_arm_planning_enabled_does_not_fake_operation_calibration() -> None:
    config = yaml.safe_load((Path(__file__).parents[1] / "config/config.yaml").read_text())
    config["arm_planning"]["enabled"] = True
    enabled, planning = _arm_planning_config_from_config(config)
    assert enabled is True
    with pytest.raises(ValueError, match="min_object_confidence"):
        planning.validate_for_grasp()
    with pytest.raises(ValueError, match="transform_max_age_ns"):
        planning.validate_for_place()


def test_grasp_context_confirmation_does_not_change_planned_relations() -> None:
    left = RigidTransform3D("left_gripper", "object-7", (0.1, 0, 0), (0, 0, 0, 1), 20, True)
    right = RigidTransform3D("right_gripper", "object-7", (-0.1, 0, 0), (0, 0, 0, 1), 20, True)
    planned = GraspContext(
        1, " box_body ", " pink ", " stable:7 ", " object-7 ", (0.24, 0.16, 0.19),
        left, right, (0, 0, 0, 1), 20, 21, None, False, True,
    )
    assert (
        planned.target_body,
        planned.target_class_id,
        planned.object_id,
        planned.object_frame,
    ) == ("box_body", "pink", "stable:7", "object-7")
    confirmed = replace(planned, confirmed=True, confirmed_at_ns=30)
    assert confirmed.object_from_left_gripper is left
    assert confirmed.object_from_right_gripper is right


def test_ros_unknown_orientation_and_size_round_trip_as_none() -> None:
    ros = _vision_types()
    _validate_vision_schema(ros)
    estimate = ObjectEstimate3D(
        "pink", (1, 2, 3), 0.9, "odom", 123,
        object_id="stable:7", orientation_xyzw=None, size_xyz_m=None,
    )
    message = _estimates_to_vision((estimate,), ros, _VisionStamp())
    detection = message.detections[0]
    pose = detection.results[0].pose.pose
    assert (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w) == (0, 0, 0, 0)
    assert tuple(vars(detection.bbox.size).values()) == (0, 0, 0)
    assert tuple(vars(detection.bbox.center.position).values()) == (1, 2, 3)
    assert tuple(vars(detection.bbox.center.orientation).values()) == (0, 0, 0, 0)
    decoded = _estimates_from_vision(message)[0]
    assert decoded.orientation_xyzw is None and decoded.size_xyz_m is None
    assert decoded.object_id == "stable:7"


def test_ros_known_local_size_with_unknown_orientation_round_trip() -> None:
    estimate = ObjectEstimate3D(
        "pink", (1, 2, 3), 0.9, "odom", 123,
        object_id="stable:8", orientation_xyzw=None,
        size_xyz_m=(0.24, 0.16, 0.19),
    )
    message = _estimates_to_vision((estimate,), _vision_types(), _VisionStamp())
    detection = message.detections[0]
    result_pose = detection.results[0].pose.pose
    bbox_pose = detection.bbox.center

    assert tuple(vars(bbox_pose.position).values()) == tuple(
        vars(result_pose.position).values()
    ) == (1, 2, 3)
    assert tuple(vars(bbox_pose.orientation).values()) == tuple(
        vars(result_pose.orientation).values()
    ) == (0, 0, 0, 0)
    assert tuple(vars(detection.bbox.size).values()) == pytest.approx(
        (0.24, 0.16, 0.19)
    )
    decoded = _estimates_from_vision(message)[0]
    assert decoded.orientation_xyzw is None
    assert decoded.size_xyz_m == pytest.approx((0.24, 0.16, 0.19))
    assert decoded.object_id == "stable:8"


def test_ros_observed_orientation_and_size_round_trip_normalized() -> None:
    estimate = ObjectEstimate3D(
        "pink", (1, 2, 3), 0.9, "odom", 123,
        orientation_xyzw=(0, 0, 0, 2), size_xyz_m=(0.24, 0.16, 0.19),
    )
    decoded = _estimates_from_vision(
        _estimates_to_vision((estimate,), _vision_types(), _VisionStamp())
    )[0]
    assert decoded.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert decoded.size_xyz_m == pytest.approx((0.24, 0.16, 0.19))


def test_ros_nonzero_size_rejects_inconsistent_bbox_center() -> None:
    estimate = ObjectEstimate3D(
        "pink", (1, 2, 3), 0.9, "odom", 123,
        size_xyz_m=(0.24, 0.16, 0.19),
    )
    message = _estimates_to_vision((estimate,), _vision_types(), _VisionStamp())
    message.detections[0].bbox.center.position.x = 99.0

    with pytest.raises(ValueError, match="center.position"):
        _estimates_from_vision(message)


def test_ros_nonzero_size_rejects_identity_bbox_for_unknown_orientation() -> None:
    estimate = ObjectEstimate3D(
        "pink", (1, 2, 3), 0.9, "odom", 123,
        size_xyz_m=(0.24, 0.16, 0.19),
    )
    message = _estimates_to_vision((estimate,), _vision_types(), _VisionStamp())
    message.detections[0].bbox.center.orientation.w = 1.0

    with pytest.raises(ValueError, match="同时使用零四元数"):
        _estimates_from_vision(message)


def test_ros_nonzero_size_rejects_inconsistent_known_bbox_orientation() -> None:
    estimate = ObjectEstimate3D(
        "pink", (1, 2, 3), 0.9, "odom", 123,
        orientation_xyzw=(0, 0, 0, 1),
        size_xyz_m=(0.24, 0.16, 0.19),
    )
    message = _estimates_to_vision((estimate,), _vision_types(), _VisionStamp())
    bbox_orientation = message.detections[0].bbox.center.orientation
    bbox_orientation.z, bbox_orientation.w = 1.0, 0.0

    with pytest.raises(ValueError, match="姿态与结果姿态不一致"):
        _estimates_from_vision(message)


def test_ros_nonzero_size_accepts_equivalent_negated_bbox_quaternion() -> None:
    estimate = ObjectEstimate3D(
        "pink", (1, 2, 3), 0.9, "odom", 123,
        orientation_xyzw=(0, 0, 0, 1),
        size_xyz_m=(0.24, 0.16, 0.19),
    )
    message = _estimates_to_vision((estimate,), _vision_types(), _VisionStamp())
    bbox_orientation = message.detections[0].bbox.center.orientation
    bbox_orientation.x = -bbox_orientation.x
    bbox_orientation.y = -bbox_orientation.y
    bbox_orientation.z = -bbox_orientation.z
    bbox_orientation.w = -bbox_orientation.w

    decoded = _estimates_from_vision(message)[0]
    assert decoded.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert decoded.size_xyz_m == pytest.approx((0.24, 0.16, 0.19))


def test_ros_rejects_partial_zero_size_sentinel() -> None:
    message = _estimates_to_vision(
        (ObjectEstimate3D("pink", (1, 2, 3), 0.9, "odom", 123),),
        _vision_types(),
        _VisionStamp(),
    )
    message.detections[0].bbox.size.x = 0.2
    with pytest.raises(ValueError, match="全零或三轴均为正数"):
        _estimates_from_vision(message)

# ---------------------------------------------------------------------------
# ArmPlanner纯Python抓放规划
# ---------------------------------------------------------------------------

# 下列参数是“测试专用已验证配置，用于验证算法结构，不代表官方环境真实标定值”。
def _verified_planner_config(**overrides: object) -> ArmPlanningConfig:
    values: dict[str, object] = {
        "min_object_confidence": 0.7,
        "transform_max_age_ns": 1_000,
        "object_estimate_max_age_ns": 1_000,
        "joint_state_max_age_ns": 1_000,
        "planned_context_max_age_ns": 1_000,
        "confirmed_context_max_age_ns": 1_000,
        "pregrasp_distance_m": 0.10,
        "grasp_contact_offset_m": 0.02,
        "lift_distance_m": 0.15,
        "retreat_distance_m": 0.20,
        "preplace_height_m": 0.20,
        "release_offset_m": 0.05,
        "post_release_retreat_distance_m": 0.25,
        "settle_time_s": 0.30,
        "max_slide_waypoint_delta_m": 0.50,
        "max_arm_waypoint_delta_rad": 0.50,
        "max_gripper_waypoint_delta": 1.0,
        "pregrasp_duration_s": 1.0,
        "grasp_duration_s": 2.0,
        "lift_duration_s": 3.0,
        "retreat_duration_s": 4.0,
        "preplace_duration_s": 1.5,
        "lower_duration_s": 2.5,
        "release_duration_s": 0.5,
        "post_release_retreat_duration_s": 3.5,
        "left_gripper_min": 0.0,
        "left_gripper_max": 1.0,
        "right_gripper_min": 0.0,
        "right_gripper_max": 1.0,
        "left_gripper_open": 0.8,
        "left_gripper_closed": 0.2,
        "right_gripper_open": 0.8,
        "right_gripper_closed": 0.2,
        "gripper_verified_in_official_environment": True,
    }
    values.update(overrides)
    return ArmPlanningConfig(**values)  # type: ignore[arg-type]


def _planner_task(
    *, color: str = "pink", place_type: str = "table_point",
    place_world_xyz: tuple[float, float, float] = (2.0, 0.5, 0.4),
) -> TaskSpec:
    side_fields = (
        {"ref_prop": "shelf", "ref_prop_body": "shelf_body", "direction": "left"}
        if place_type == "shelf_prop_side" else {}
    )
    return TaskSpec(
        task_id=7,
        instruction=f"move {color} box",
        target_kind="box",
        target_body=f"box_{color}",
        target_color=color,
        place_type=place_type,
        place_world_xyz=place_world_xyz,
        place_frame_id="world",
        place_radius=0.1,
        timestamp_ns=90,
        **side_fields,
    )


def _planner_target(
    *, color: str = "pink", position: tuple[float, float, float] = (1.0, 0.0, 0.5),
    orientation: tuple[float, float, float, float] | None = (0.0, 0.0, 0.0, 1.0),
    size: tuple[float, float, float] | None = (0.40, 0.20, 0.30),
    object_id: str | None = "track-3", timestamp_ns: int = 100,
    confidence: float = 0.9, valid: bool = True,
) -> ObjectEstimate3D:
    return ObjectEstimate3D(
        color, position, confidence, "camera", timestamp_ns,
        valid=valid, failure_reason="感知无效" if not valid else "",
        object_id=object_id, orientation_xyzw=orientation, size_xyz_m=size,
    )


def _planner_actual_joints(
    *, timestamp_ns: int = 100, valid: bool = True,
    left_gripper: float = 0.8, right_gripper: float = 0.8,
) -> RobotJointState:
    position = (
        0.1, 0.25, -0.25, *(0.0,) * 6, left_gripper,
        *(0.0,) * 6, right_gripper,
    )
    return RobotJointState(
        position=position,
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=timestamp_ns,
        valid=valid,
        failure_reason="关节反馈无效" if not valid else "",
    )


def _target_transforms(
    *, footprint_timestamp: int = 100, world_timestamp: int = 100,
) -> tuple[RigidTransform3D, RigidTransform3D]:
    return (
        RigidTransform3D("camera", "footprint", (0, 0, 0), (0, 0, 0, 1),
                         footprint_timestamp, True),
        RigidTransform3D("camera", "world", (4, 2, 0), (0, 0, 0, 1),
                         world_timestamp, True),
    )


class _FakePlannerKDL:
    """只记录规划器调用；确定性解不是官方标定或可达性结论。"""

    def __init__(
        self, *, fail_at: int | None = None, half_at: int | None = None,
        raise_at: int | None = None, arm_values: tuple[float, ...] | None = None,
    ) -> None:
        self.fail_at = fail_at
        self.half_at = half_at
        self.raise_at = raise_at
        self.arm_values = arm_values
        self.calls: list[dict[str, object]] = []
        self.self_check_called = False

    def self_check(self) -> None:
        self.self_check_called = True
        raise AssertionError("ArmPlanner构造或规划不得调用self_check")

    def solve_ik(self, **kwargs: object) -> IKResult:
        index = len(self.calls)
        self.calls.append(dict(kwargs))
        if index == self.raise_at:
            raise RuntimeError("fake KDL依赖异常")
        slide = float(kwargs["target_slide"])
        if index == self.fail_at:
            return IKResult(slide, None, None, False, f"阶段{index}无解")
        value = 0.01 * (index + 1)
        left = self.arm_values if self.arm_values is not None else (value,) * 6
        right = self.arm_values if self.arm_values is not None else (-value,) * 6
        if index == self.half_at:
            right = None
        return IKResult(slide, left, right, True)


class _SlideSearchFakeKDL(_FakePlannerKDL):
    """只模拟官方限位和特定高度可达；不代表真实机器人可达性。"""

    def __init__(
        self,
        *,
        success_slide: float | None,
        limits: tuple[float, float] = (-0.04, 0.60),
    ) -> None:
        super().__init__()
        self.success_slide = success_slide
        self.limits = limits

    def slide_limits(self) -> tuple[float, float]:
        return self.limits

    def solve_ik(self, **kwargs: object) -> IKResult:
        self.calls.append(dict(kwargs))
        slide = float(kwargs["target_slide"])
        if self.success_slide is None or not math.isclose(
            slide, self.success_slide, rel_tol=0.0, abs_tol=1e-12
        ):
            return IKResult(slide, None, None, False, f"slide={slide}无解")
        return IKResult(slide, (0.01,) * 6, (-0.01,) * 6, True)


def _run_grasp(
    adapter: _FakePlannerKDL | None = None,
    *, config: ArmPlanningConfig | None = None, task: TaskSpec | None = None,
    target: ObjectEstimate3D | None = None, joints: RobotJointState | None = None,
    transforms: tuple[RigidTransform3D, RigidTransform3D] | None = None,
    now_ns: int = 110,
) -> tuple[GraspTarget, object, _FakePlannerKDL]:
    fake = adapter or _FakePlannerKDL()
    footprint, world = transforms or _target_transforms()
    result = ArmPlanner(fake, config or _verified_planner_config()).plan_grasp(  # type: ignore[arg-type]
        task or _planner_task(), target or _planner_target(), footprint, world,
        joints or _planner_actual_joints(), now_ns,
    )
    return result[0], result[1], fake


def _confirmed_context() -> GraspContext:
    target, trajectory, _ = _run_grasp()
    assert target.valid and trajectory.valid and target.grasp_context is not None
    return replace(target.grasp_context, confirmed=True, confirmed_at_ns=120)


def _run_place(
    adapter: _FakePlannerKDL | None = None,
    *, config: ArmPlanningConfig | None = None, task: TaskSpec | None = None,
    context: GraspContext | None = None, joints: RobotJointState | None = None,
    transform: RigidTransform3D | None = None, now_ns: int = 130,
) -> tuple[PlaceTarget, object, _FakePlannerKDL]:
    fake = adapter or _FakePlannerKDL()
    world_to_footprint = transform or RigidTransform3D(
        "world", "footprint", (-1, 0, 0), (0, 0, 0, 1), 125, True
    )
    result = ArmPlanner(fake, config or _verified_planner_config()).plan_place(  # type: ignore[arg-type]
        task or _planner_task(), world_to_footprint, context or _confirmed_context(),
        joints or _planner_actual_joints(timestamp_ns=125), now_ns,
    )
    return result[0], result[1], fake


def test_arm_planner_constructor_only_saves_injected_dependencies() -> None:
    fake = _FakePlannerKDL()
    config = _verified_planner_config()
    planner = ArmPlanner(fake, config)  # type: ignore[arg-type]
    assert planner._ik_adapter is fake and planner._config is config
    assert not fake.self_check_called and fake.calls == []


def test_default_uncalibrated_config_fails_closed_for_both_operations() -> None:
    fake = _FakePlannerKDL()
    planner = ArmPlanner(fake, ArmPlanningConfig())  # type: ignore[arg-type]
    footprint, world = _target_transforms()
    grasp, pick = planner.plan_grasp(
        _planner_task(), _planner_target(), footprint, world,
        _planner_actual_joints(), 110,
    )
    place, place_trajectory = planner.plan_place(
        _planner_task(), RigidTransform3D("world", "footprint", (0, 0, 0),
                                         (0, 0, 0, 1), 100, True),
        _confirmed_context(), _planner_actual_joints(), 130,
    )
    assert not grasp.valid and not pick.valid and pick.waypoints == ()
    assert not place.valid and not place_trajectory.valid and place_trajectory.waypoints == ()
    assert fake.calls == []


def test_grasp_rejects_missing_perception_facts_and_identity_mismatch() -> None:
    cases = (
        (replace(_planner_task(), valid=False, failure_reason="任务无效"), _planner_target(), "任务无效"),
        (_planner_task(), _planner_target(valid=False), "感知无效"),
        (_planner_task(), _planner_target(color="yellow"), "不匹配"),
        (_planner_task(), _planner_target(object_id=None), "object_id"),
        (_planner_task(), _planner_target(orientation=None), "orientation"),
        (_planner_task(), _planner_target(size=None), "size_xyz"),
        (_planner_task(), _planner_target(confidence=0.6), "confidence"),
    )
    for task, target, reason in cases:
        grasp, trajectory, fake = _run_grasp(task=task, target=target)
        assert not grasp.valid and not trajectory.valid and trajectory.waypoints == ()
        assert grasp.failure_reason == trajectory.failure_reason
        assert reason in grasp.failure_reason and fake.calls == []


def test_grasp_rejects_wrong_transform_directions_without_inversion() -> None:
    wrong_footprint = RigidTransform3D(
        "footprint", "camera", (0, 0, 0), (0, 0, 0, 1), 100, True
    )
    wrong_world = RigidTransform3D(
        "world", "camera", (0, 0, 0), (0, 0, 0, 1), 100, True
    )
    for transforms, expected in (
        ((wrong_footprint, _target_transforms()[1]), "camera→footprint"),
        ((_target_transforms()[0], wrong_world), "camera→world"),
    ):
        grasp, trajectory, fake = _run_grasp(transforms=transforms)
        assert not grasp.valid and not trajectory.valid and expected in grasp.failure_reason
        assert fake.calls == []


def test_grasp_rejects_stale_future_and_invalid_time_inputs() -> None:
    config = _verified_planner_config(
        transform_max_age_ns=10, object_estimate_max_age_ns=10,
        joint_state_max_age_ns=10,
    )
    cases = (
        {"transforms": _target_transforms(footprint_timestamp=80), "now_ns": 110},
        {"transforms": _target_transforms(world_timestamp=120), "now_ns": 110},
        {"target": _planner_target(timestamp_ns=80), "now_ns": 110},
        {"target": _planner_target(timestamp_ns=120), "now_ns": 110},
        {"joints": _planner_actual_joints(timestamp_ns=80), "now_ns": 110},
        {"joints": _planner_actual_joints(timestamp_ns=120), "now_ns": 110},
        {"joints": _planner_actual_joints(valid=False), "now_ns": 110},
        {"task": replace(_planner_task(), timestamp_ns=120), "now_ns": 110},
        {"now_ns": True},
        {"now_ns": -1},
    )
    for kwargs in cases:
        grasp, trajectory, fake = _run_grasp(config=config, **kwargs)  # type: ignore[arg-type]
        assert not grasp.valid and not trajectory.valid and trajectory.waypoints == ()
        assert grasp.failure_reason == trajectory.failure_reason and fake.calls == []


def test_grasp_transforms_are_bound_to_object_observation_time() -> None:
    config = _verified_planner_config(transform_max_age_ns=10)
    identity = (0.0, 0.0, 0.0, 1.0)
    cases = (
        (
            _planner_target(timestamp_ns=100),
            (
                RigidTransform3D("camera", "footprint", (0, 0, 0), identity, 0, True),
                RigidTransform3D("camera", "world", (0, 0, 0), identity, 100, True),
            ),
            110,
            "过期",
        ),
        (
            _planner_target(timestamp_ns=100),
                (
                    RigidTransform3D("camera", "footprint", (0, 0, 0), identity, 111, True),
                    RigidTransform3D("camera", "world", (0, 0, 0), identity, 101, True),
            ),
            111,
            "target.timestamp_ns时间不匹配",
        ),
        (
            _planner_target(timestamp_ns=100),
                (
                    RigidTransform3D("camera", "footprint", (0, 0, 0), identity, 101, True),
                RigidTransform3D("camera", "world", (0, 0, 0), identity, 111, True),
            ),
            111,
            "target.timestamp_ns时间不匹配",
        ),
        # 两个变换对now都很新，但都与物体观测相差15ns。
        (
            _planner_target(timestamp_ns=100),
            (
                RigidTransform3D("camera", "footprint", (0, 0, 0), identity, 115, True),
                RigidTransform3D("camera", "world", (0, 0, 0), identity, 115, True),
            ),
            115,
            "target.timestamp_ns时间不匹配",
        ),
    )
    for target, transforms, now_ns, expected in cases:
        grasp, trajectory, fake = _run_grasp(
            config=config, target=target, transforms=transforms, now_ns=now_ns
        )
        assert not grasp.valid and not trajectory.valid and trajectory.waypoints == ()
        assert expected in grasp.failure_reason
        assert fake.calls == []


def test_grasp_transform_observation_time_boundary_is_allowed() -> None:
    config = _verified_planner_config(transform_max_age_ns=10)
    target = _planner_target(timestamp_ns=110)
    transforms = _target_transforms(footprint_timestamp=100, world_timestamp=100)
    grasp, trajectory, fake = _run_grasp(
        config=config, target=target, transforms=transforms, now_ns=110
    )
    assert grasp.valid and trajectory.valid and len(fake.calls) == 4


def test_grasp_rejects_actual_grippers_outside_official_ranges_before_ik() -> None:
    config = _verified_planner_config(max_gripper_waypoint_delta=100.0)
    for joints, expected in (
        (_planner_actual_joints(left_gripper=-0.01), "left gripper"),
        (_planner_actual_joints(right_gripper=1.01), "right gripper"),
    ):
        grasp, trajectory, fake = _run_grasp(config=config, joints=joints)
        assert not grasp.valid and not trajectory.valid and trajectory.waypoints == ()
        assert expected in grasp.failure_reason and fake.calls == []


def test_actual_gripper_range_boundaries_are_allowed_for_grasp_and_place() -> None:
    grasp_joints = _planner_actual_joints(left_gripper=0.0, right_gripper=1.0)
    grasp, pick, grasp_fake = _run_grasp(joints=grasp_joints)
    place_joints = _planner_actual_joints(
        timestamp_ns=125, left_gripper=0.0, right_gripper=1.0
    )
    place, place_trajectory, place_fake = _run_place(joints=place_joints)
    assert grasp.valid and pick.valid and len(grasp_fake.calls) == 4
    assert place.valid and place_trajectory.valid and len(place_fake.calls) == 3


def test_grasp_geometry_uses_object_sides_preplace_lift_and_radial_retreat() -> None:
    grasp, trajectory, _ = _run_grasp()
    assert grasp.valid and trajectory.valid
    assert grasp.left_grasp is not None and grasp.right_grasp is not None
    assert grasp.left_pregrasp is not None and grasp.right_pregrasp is not None
    assert grasp.left_lift is not None and grasp.right_lift is not None
    assert grasp.left_retreat is not None and grasp.right_retreat is not None
    center = (1.0, 0.0, 0.5)
    assert grasp.left_grasp.position_xyz != center != grasp.right_grasp.position_xyz
    assert grasp.left_grasp.position_xyz == pytest.approx((1.0, 0.12, 0.5))
    assert grasp.right_grasp.position_xyz == pytest.approx((1.0, -0.12, 0.5))
    assert grasp.left_pregrasp.position_xyz == pytest.approx((1.0, 0.22, 0.5))
    assert grasp.right_pregrasp.position_xyz == pytest.approx((1.0, -0.22, 0.5))
    assert grasp.left_lift.position_xyz == pytest.approx((1.0, 0.12, 0.65))
    assert grasp.right_lift.position_xyz == pytest.approx((1.0, -0.12, 0.65))
    assert grasp.left_retreat.position_xyz == pytest.approx((0.8, 0.12, 0.65))
    assert grasp.right_retreat.position_xyz == pytest.approx((0.8, -0.12, 0.65))


@pytest.mark.parametrize("yaw", [0.2, 0.7, 1.4, 2.3, -1.1])
def test_grasp_axis_is_deterministic_for_random_yaw(yaw: float) -> None:
    quaternion = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    first, _, _ = _run_grasp(target=_planner_target(orientation=quaternion))
    second, _, _ = _run_grasp(target=_planner_target(orientation=quaternion))
    assert first.valid and second.valid
    assert first.left_grasp == second.left_grasp
    assert first.right_grasp == second.right_grasp
    assert first.left_grasp is not None and first.right_grasp is not None
    side = np.subtract(first.left_grasp.position_xyz, first.right_grasp.position_xyz)
    assert np.linalg.norm(side[:2]) > 0.0 and side[2] == pytest.approx(0.0)


def test_grasp_strategy_does_not_depend_on_color_or_task_id() -> None:
    pink, _, _ = _run_grasp(task=_planner_task(color="pink"), target=_planner_target(color="pink"))
    yellow_task = replace(_planner_task(color="yellow"), task_id=99)
    yellow, _, _ = _run_grasp(task=yellow_task, target=_planner_target(color="yellow"))
    assert pink.left_grasp == yellow.left_grasp
    assert pink.right_grasp == yellow.right_grasp


def test_grasp_axis_tie_uses_smaller_dimension_then_local_y() -> None:
    yaw = math.pi / 4.0
    quaternion = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    small_x, _, _ = _run_grasp(
        target=_planner_target(orientation=quaternion, size=(0.1, 0.3, 0.2))
    )
    equal, _, _ = _run_grasp(
        target=_planner_target(orientation=quaternion, size=(0.2, 0.2, 0.2))
    )
    assert small_x.left_grasp is not None and equal.left_grasp is not None
    assert small_x.left_grasp.position_xyz != equal.left_grasp.position_xyz


def test_grasp_rejects_degenerate_center_and_nonvertical_object() -> None:
    tilted = (math.sin(0.2), 0.0, 0.0, math.cos(0.2))
    for target, expected in (
        (_planner_target(position=(0.0, 0.0, 0.5)), "退化"),
        (_planner_target(orientation=tilted), "竖直向上"),
        (_planner_target(orientation=(1.0, 0.0, 0.0, 0.0)), "竖直向上"),
    ):
        grasp, trajectory, fake = _run_grasp(target=target)
        assert not grasp.valid and not trajectory.valid and expected in grasp.failure_reason
        assert fake.calls == []


def test_grasp_rejects_world_orientation_tilt_created_by_transform_before_ik() -> None:
    half = math.sqrt(0.5)
    footprint, _ = _target_transforms()
    tilted_world = RigidTransform3D(
        "camera", "world", (4, 2, 0), (half, 0, 0, half), 100, True
    )
    grasp, trajectory, fake = _run_grasp(transforms=(footprint, tilted_world))
    assert not grasp.valid and not trajectory.valid and trajectory.waypoints == ()
    assert "world物体姿态" in grasp.failure_reason and fake.calls == []


def test_grasp_tool_orientations_are_finite_unit_footprint_poses() -> None:
    grasp, _, _ = _run_grasp()
    poses = (
        grasp.left_pregrasp, grasp.right_pregrasp, grasp.left_grasp, grasp.right_grasp,
        grasp.left_lift, grasp.right_lift, grasp.left_retreat, grasp.right_retreat,
    )
    for pose in poses:
        assert pose is not None and pose.frame_id == "footprint"
        assert all(math.isfinite(value) for value in pose.orientation_xyzw)
        assert math.hypot(*pose.orientation_xyzw) == pytest.approx(1.0)
    assert grasp.left_grasp is not None and grasp.right_grasp is not None
    assert arm_planning_module._rotate_vector(
        grasp.left_grasp.orientation_xyzw, (1.0, 0.0, 0.0)
    ) == pytest.approx((1.0, 0.0, 0.0))
    assert arm_planning_module._rotate_vector(
        grasp.left_grasp.orientation_xyzw, (0.0, 0.0, 1.0)
    ) == pytest.approx((0.0, 1.0, 0.0))
    assert arm_planning_module._rotate_vector(
        grasp.right_grasp.orientation_xyzw, (0.0, 0.0, 1.0)
    ) == pytest.approx((0.0, -1.0, 0.0))


def test_grasp_calls_four_unique_ik_stages_with_original_feedback_and_slide() -> None:
    joints = _planner_actual_joints()
    grasp, trajectory, fake = _run_grasp(joints=joints)
    assert grasp.valid and trajectory.valid and len(fake.calls) == 4
    assert all(call["actual_joints"] is joints for call in fake.calls)
    assert all(call["target_slide"] == joints.position[0] for call in fake.calls)
    assert [call["left_target"] for call in fake.calls] == [
        grasp.left_pregrasp, grasp.left_grasp, grasp.left_lift, grasp.left_retreat,
    ]
    assert [call["right_target"] for call in fake.calls] == [
        grasp.right_pregrasp, grasp.right_grasp, grasp.right_lift, grasp.right_retreat,
    ]


def test_grasp_searches_nearby_official_slide_after_current_height_fails() -> None:
    fake = _SlideSearchFakeKDL(success_slide=0.12)
    grasp, trajectory, _ = _run_grasp(fake)
    slides = [float(call["target_slide"]) for call in fake.calls]

    assert grasp.valid and trajectory.valid
    assert slides == pytest.approx([0.1, 0.12, 0.12, 0.12, 0.12])
    assert all(fake.limits[0] <= slide <= fake.limits[1] for slide in slides)
    assert all(
        waypoint.joint_position[0] == pytest.approx(0.12)
        for waypoint in trajectory.waypoints
    )


def test_grasp_slide_search_fails_atomically_when_all_official_candidates_fail() -> None:
    fake = _SlideSearchFakeKDL(success_slide=None)
    grasp, trajectory, _ = _run_grasp(fake)
    slides = [float(call["target_slide"]) for call in fake.calls]

    assert not grasp.valid and not trajectory.valid and trajectory.waypoints == ()
    assert grasp.failure_reason == trajectory.failure_reason
    assert "所有官方slide候选" in grasp.failure_reason
    assert slides[0] == pytest.approx(0.1)
    assert len(slides) > 1
    assert all(fake.limits[0] <= slide <= fake.limits[1] for slide in slides)


@pytest.mark.parametrize(("mode", "index"), [("fail_at", 2), ("half_at", 1), ("raise_at", 3)])
def test_grasp_any_ik_failure_or_half_solution_fails_atomically(mode: str, index: int) -> None:
    adapter = _FakePlannerKDL(**{mode: index})
    grasp, trajectory, _ = _run_grasp(adapter)
    assert not grasp.valid and grasp.grasp_context is None
    assert not trajectory.valid and trajectory.waypoints == ()
    assert grasp.failure_reason == trajectory.failure_reason


@pytest.mark.parametrize("returned_slide", [0.2, float("nan")])
def test_grasp_rejects_inconsistent_or_nonfinite_ik_slide(returned_slide: float) -> None:
    class _BadSlideKDL(_FakePlannerKDL):
        def solve_ik(self, **kwargs: object) -> IKResult:
            result = super().solve_ik(**kwargs)
            return replace(result, target_slide=returned_slide)

    grasp, trajectory, _ = _run_grasp(_BadSlideKDL())
    assert not grasp.valid and not trajectory.valid and trajectory.waypoints == ()
    assert "slide" in grasp.failure_reason


def test_grasp_trajectory_maps_13_to_17_and_preserves_head() -> None:
    joints = _planner_actual_joints()
    _, trajectory, _ = _run_grasp(joints=joints)
    assert trajectory.valid and trajectory.execution_phase is GlobalPhase.EXECUTE_PICK
    assert trajectory.trajectory_id == "pick-7-box_pink-track-3-110"
    assert len(trajectory.waypoints) == 5
    assert [waypoint.phase for waypoint in trajectory.waypoints] == [
        ArmMotionPhase.PREGRASP, ArmMotionPhase.GRASP, ArmMotionPhase.GRASP,
        ArmMotionPhase.LIFT, ArmMotionPhase.RETREAT,
    ]
    assert [waypoint.time_from_start_s for waypoint in trajectory.waypoints] == [1, 2, 3, 6, 10]
    assert [waypoint.joint_position[9] for waypoint in trajectory.waypoints] == [0.8, 0.8, 0.2, 0.2, 0.2]
    assert [waypoint.joint_position[16] for waypoint in trajectory.waypoints] == [0.8, 0.8, 0.2, 0.2, 0.2]
    for waypoint in trajectory.waypoints:
        assert waypoint.joint_position[0] == joints.position[0]
        assert waypoint.joint_position[1:3] == joints.position[1:3]
        assert waypoint.controlled_mask == (True, False, False, *(True,) * 14)
        assert len(waypoint.joint_position[3:9]) == len(waypoint.joint_position[10:16]) == 6
    assert trajectory.waypoints[0].joint_position[3:9] == pytest.approx((0.01,) * 6)
    assert trajectory.waypoints[0].joint_position[10:16] == pytest.approx((-0.01,) * 6)


def test_grasp_continuity_checks_first_arm_and_gripper_transitions() -> None:
    first_arm, first_trajectory, _ = _run_grasp(
        _FakePlannerKDL(arm_values=(0.2,) * 6),
        config=_verified_planner_config(max_arm_waypoint_delta_rad=0.1),
    )
    gripper, gripper_trajectory, _ = _run_grasp(
        config=_verified_planner_config(max_gripper_waypoint_delta=0.3)
    )
    assert not first_arm.valid and not first_trajectory.valid and "arm" in first_arm.failure_reason
    assert not gripper.valid and not gripper_trajectory.valid and "gripper" in gripper.failure_reason


def test_continuity_helper_rejects_adjacent_slide_and_arm_deltas_by_unit() -> None:
    actual = _planner_actual_joints().position
    config = _verified_planner_config(
        max_slide_waypoint_delta_m=0.05, max_arm_waypoint_delta_rad=0.05
    )
    mask = (True, False, False, *(True,) * 14)
    first = arm_planning_module.JointWaypoint(
        ArmMotionPhase.PREGRASP, 1.0, actual, mask
    )
    slide_position = list(actual)
    slide_position[0] += 0.1
    slide = arm_planning_module.JointWaypoint(
        ArmMotionPhase.GRASP, 2.0, tuple(slide_position), mask
    )
    with pytest.raises(ValueError, match="slide"):
        arm_planning_module._validate_waypoint_continuity(actual, (first, slide), config)
    arm_position = list(actual)
    arm_position[3] += 0.1
    arm = arm_planning_module.JointWaypoint(
        ArmMotionPhase.GRASP, 2.0, tuple(arm_position), mask
    )
    with pytest.raises(ValueError, match="arm"):
        arm_planning_module._validate_waypoint_continuity(actual, (first, arm), config)


def test_planned_grasp_context_identity_orientation_and_relative_transforms() -> None:
    grasp, trajectory, _ = _run_grasp()
    context = grasp.grasp_context
    assert trajectory.valid and context is not None and context.valid
    assert not context.confirmed and context.confirmed_at_ns is None
    assert (context.task_id, context.target_body, context.target_class_id,
            context.object_id, context.object_frame) == (
                7, "box_pink", "pink", "track-3", "object/track-3"
            )
    assert context.object_size_xyz_m == (0.4, 0.2, 0.3)
    assert context.object_orientation_world_xyzw_at_grasp == (0.0, 0.0, 0.0, 1.0)
    assert context.orientation_observed_at_ns == 100 and context.planned_at_ns == 110
    object_pose = Pose3D((1.0, 0.0, 0.5), (0, 0, 0, 1), "footprint")
    assert context.object_from_left_gripper is not None
    assert context.object_from_right_gripper is not None
    reconstructed_left = arm_planning_module._compose_pose_with_transform(
        object_pose, context.object_from_left_gripper
    )
    reconstructed_right = arm_planning_module._compose_pose_with_transform(
        object_pose, context.object_from_right_gripper
    )
    assert reconstructed_left == grasp.left_grasp
    assert reconstructed_right == grasp.right_grasp


def test_grasp_context_saves_composed_world_object_orientation() -> None:
    target_yaw = 0.3
    transform_yaw = 0.4
    target_q = (0.0, 0.0, math.sin(target_yaw / 2), math.cos(target_yaw / 2))
    transform_q = (
        0.0, 0.0, math.sin(transform_yaw / 2), math.cos(transform_yaw / 2)
    )
    footprint, _ = _target_transforms()
    world = RigidTransform3D(
        "camera", "world", (4, 2, 0), transform_q, 100, True
    )
    grasp, trajectory, _ = _run_grasp(
        target=_planner_target(orientation=target_q), transforms=(footprint, world)
    )
    assert grasp.valid and trajectory.valid and grasp.grasp_context is not None
    expected = (
        0.0, 0.0,
        math.sin((target_yaw + transform_yaw) / 2),
        math.cos((target_yaw + transform_yaw) / 2),
    )
    assert grasp.grasp_context.object_orientation_world_xyzw_at_grasp == pytest.approx(expected)


def test_planner_does_not_store_grasp_context_across_calls() -> None:
    fake = _FakePlannerKDL()
    planner = ArmPlanner(fake, _verified_planner_config())  # type: ignore[arg-type]
    footprint, world = _target_transforms()
    first, _ = planner.plan_grasp(
        _planner_task(), _planner_target(object_id="first"), footprint, world,
        _planner_actual_joints(), 110,
    )
    second, _ = planner.plan_grasp(
        _planner_task(), _planner_target(object_id="second"), footprint, world,
        _planner_actual_joints(), 111,
    )
    assert first.grasp_context is not None and second.grasp_context is not None
    assert first.grasp_context.object_id == "first"
    assert second.grasp_context.object_id == "second"
    assert not any("context" in name for name in vars(planner))


def test_place_rejects_unconfirmed_mismatched_stale_and_wrong_frame_context() -> None:
    confirmed = _confirmed_context()
    cases = (
        (replace(confirmed, confirmed=False, confirmed_at_ns=None), None, "confirmed"),
        (replace(confirmed, task_id=999), None, "身份不匹配"),
        (replace(confirmed, target_class_id="yellow"), None, "target_class_id"),
        (confirmed, _verified_planner_config(confirmed_context_max_age_ns=5), "过期"),
    )
    for context, config, expected in cases:
        place, trajectory, fake = _run_place(
            context=context, config=config, now_ns=130
        )
        assert not place.valid and not trajectory.valid and expected in place.failure_reason
        assert place.failure_reason == trajectory.failure_reason and fake.calls == []
    wrong = RigidTransform3D("footprint", "world", (0, 0, 0), (0, 0, 0, 1), 125, True)
    place, trajectory, fake = _run_place(transform=wrong)
    assert not place.valid and not trajectory.valid and "world→footprint" in place.failure_reason
    assert fake.calls == []


def test_place_rejects_task_without_world_goal_and_future_task() -> None:
    invalid = TaskSpec(7, "bad", "", "", "", valid=False, failure_reason="缺少place_world_xyz")
    for task in (invalid, replace(_planner_task(), timestamp_ns=140)):
        place, trajectory, fake = _run_place(task=task, now_ns=130)
        assert not place.valid and not trajectory.valid and trajectory.waypoints == ()
        assert fake.calls == []


def test_place_rejects_stale_or_future_transform_joints_and_context() -> None:
    config = _verified_planner_config(
        transform_max_age_ns=10, joint_state_max_age_ns=10,
        confirmed_context_max_age_ns=10,
    )
    cases = (
        {"transform": RigidTransform3D("world", "footprint", (0, 0, 0),
                                       (0, 0, 0, 1), 100, True)},
        {"transform": RigidTransform3D("world", "footprint", (0, 0, 0),
                                       (0, 0, 0, 1), 140, True)},
        {"joints": _planner_actual_joints(timestamp_ns=100)},
        {"joints": _planner_actual_joints(timestamp_ns=140)},
        {"context": _confirmed_context(), "now_ns": 140},
        {"context": replace(_confirmed_context(), confirmed_at_ns=140)},
    )
    for kwargs in cases:
        place, trajectory, fake = _run_place(config=config, **kwargs)
        assert not place.valid and not trajectory.valid and trajectory.waypoints == ()
        assert place.failure_reason == trajectory.failure_reason and fake.calls == []


def test_place_rejects_actual_gripper_out_of_range_before_ik() -> None:
    config = _verified_planner_config(max_gripper_waypoint_delta=100.0)
    for joints in (
        _planner_actual_joints(timestamp_ns=125, left_gripper=-0.01),
        _planner_actual_joints(timestamp_ns=125, right_gripper=1.01),
    ):
        place, trajectory, fake = _run_place(config=config, joints=joints)
        assert not place.valid and not trajectory.valid and trajectory.waypoints == ()
        assert "gripper" in place.failure_reason and fake.calls == []


def test_place_rejects_damaged_context_transform_direction() -> None:
    context = _confirmed_context()
    damaged = object.__new__(GraspContext)
    for name in context.__dataclass_fields__:
        object.__setattr__(damaged, name, getattr(context, name))
    object.__setattr__(
        damaged,
        "object_from_left_gripper",
        RigidTransform3D(
            context.object_frame, "left_gripper", (0, 0, 0),
            (0, 0, 0, 1), 110, True,
        ),
    )
    place, trajectory, fake = _run_place(context=damaged)
    assert not place.valid and not trajectory.valid
    assert "left_gripper→object/track-3" in place.failure_reason
    assert fake.calls == []


@pytest.mark.parametrize("place_type", ["shelf_point", "table_point", "shelf_prop_side"])
def test_all_official_place_types_use_same_exact_center_pipeline(place_type: str) -> None:
    place, trajectory, _ = _run_place(task=_planner_task(place_type=place_type))
    assert place.valid and trajectory.valid and place.object_goal_pose is not None
    assert place.object_goal_pose.position_xyz == pytest.approx((1.0, 0.5, 0.4))


def test_place_geometry_preserves_goal_orientation_offsets_and_common_retreat() -> None:
    context = _confirmed_context()
    place, trajectory, _ = _run_place(context=context)
    assert place.valid and trajectory.valid and place.object_goal_pose is not None
    assert place.left_preplace is not None and place.right_preplace is not None
    assert place.left_release is not None and place.right_release is not None
    assert place.left_post_release_retreat is not None
    assert place.right_post_release_retreat is not None
    assert place.object_goal_pose.position_xyz == pytest.approx((1.0, 0.5, 0.4))
    assert place.object_goal_pose.orientation_xyzw == context.object_orientation_world_xyzw_at_grasp
    assert place.left_preplace.position_xyz[2] - place.left_release.position_xyz[2] == pytest.approx(0.2)
    assert place.right_preplace.position_xyz[2] - place.right_release.position_xyz[2] == pytest.approx(0.2)
    radial = np.asarray((1.0, 0.5, 0.0))
    radial /= np.linalg.norm(radial)
    assert np.subtract(
        place.left_post_release_retreat.position_xyz, place.left_release.position_xyz
    ) == pytest.approx(-radial * 0.25)
    assert np.subtract(
        place.right_post_release_retreat.position_xyz, place.right_release.position_xyz
    ) == pytest.approx(-radial * 0.25)
    # 最终目标不含release_offset；release夹爪由抬高0.05m的物体Pose派生。
    assert place.left_release.position_xyz[2] == pytest.approx(0.45)
    assert place.right_release.position_xyz[2] == pytest.approx(0.45)


def test_place_converts_world_object_orientation_into_footprint() -> None:
    yaw = -0.4
    rotation = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    transform = RigidTransform3D(
        "world", "footprint", (-1, 0, 0), rotation, 125, True
    )
    place, trajectory, _ = _run_place(transform=transform)
    assert place.valid and trajectory.valid and place.object_goal_pose is not None
    expected = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    assert place.object_goal_pose.orientation_xyzw == pytest.approx(expected)


@pytest.mark.parametrize(
    "orientation",
    [
        (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        (0.0, math.sin(1e-4), 0.0, math.cos(1e-4)),
    ],
)
def test_place_rejects_tilted_confirmed_context_before_ik(
    orientation: tuple[float, float, float, float],
) -> None:
    context = replace(
        _confirmed_context(), object_orientation_world_xyzw_at_grasp=orientation
    )
    place, trajectory, fake = _run_place(context=context)
    assert not place.valid and not trajectory.valid and trajectory.waypoints == ()
    assert "竖直向上" in place.failure_reason and fake.calls == []


@pytest.mark.parametrize("yaw", [-2.7, -0.3, 0.0, 1.2, math.pi])
def test_place_allows_arbitrary_pure_yaw_context(yaw: float) -> None:
    orientation = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    context = replace(
        _confirmed_context(), object_orientation_world_xyzw_at_grasp=orientation
    )
    place, trajectory, fake = _run_place(context=context)
    assert place.valid and trajectory.valid and len(fake.calls) == 3


def test_place_calls_three_ik_stages_and_builds_four_waypoints() -> None:
    context = _confirmed_context()
    joints = _planner_actual_joints(timestamp_ns=125)
    before = context
    place, trajectory, fake = _run_place(context=context, joints=joints)
    assert place.valid and trajectory.valid and context == before
    assert len(fake.calls) == 3
    assert all(call["actual_joints"] is joints for call in fake.calls)
    assert all(call["target_slide"] == joints.position[0] for call in fake.calls)
    assert [call["left_target"] for call in fake.calls] == [
        place.left_preplace, place.left_release, place.left_post_release_retreat,
    ]
    assert [waypoint.phase for waypoint in trajectory.waypoints] == [
        ArmMotionPhase.PREPLACE, ArmMotionPhase.LOWER, ArmMotionPhase.RELEASE,
        ArmMotionPhase.POST_RELEASE_RETREAT,
    ]
    assert [waypoint.time_from_start_s for waypoint in trajectory.waypoints] == [1.5, 4.0, 4.5, 8.0]
    assert [waypoint.joint_position[9] for waypoint in trajectory.waypoints] == [0.2, 0.2, 0.8, 0.8]
    assert [waypoint.joint_position[16] for waypoint in trajectory.waypoints] == [0.2, 0.2, 0.8, 0.8]
    assert trajectory.trajectory_id == "place-7-box_pink-track-3-130"


def test_place_searches_nearby_official_slide_after_current_height_fails() -> None:
    fake = _SlideSearchFakeKDL(success_slide=0.12)
    place, trajectory, _ = _run_place(fake)
    slides = [float(call["target_slide"]) for call in fake.calls]

    assert place.valid and trajectory.valid
    assert slides == pytest.approx([0.1, 0.12, 0.12, 0.12])
    assert all(fake.limits[0] <= slide <= fake.limits[1] for slide in slides)
    assert all(
        waypoint.joint_position[0] == pytest.approx(0.12)
        for waypoint in trajectory.waypoints
    )


@pytest.mark.parametrize(("mode", "index"), [("fail_at", 0), ("half_at", 1), ("raise_at", 2)])
def test_place_any_ik_failure_fails_atomically(mode: str, index: int) -> None:
    adapter = _FakePlannerKDL(**{mode: index})
    place, trajectory, _ = _run_place(adapter)
    assert not place.valid and not trajectory.valid and trajectory.waypoints == ()
    assert place.failure_reason == trajectory.failure_reason


def test_arm_planner_rejects_dependency_without_solve_ik() -> None:
    with pytest.raises(TypeError, match="solve_ik"):
        ArmPlanner(object(), ArmPlanningConfig())  # type: ignore[arg-type]
