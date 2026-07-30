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
    IKResult,
    NavGoal,
    ObjectEstimate3D,
    Pose3D,
    RobotJointState,
    SlotType,
    TaskSpec,
)
from team_sorting.navigation import (
    Bounds3D,
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
    valid: bool = True,
) -> CameraIntrinsics:
    return CameraIntrinsics(
        k=k,
        width=640,
        height=480,
        frame_id="camera_optical_frame",
        timestamp_ns=10,
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
# NavigationController临时未实现约束
# ---------------------------------------------------------------------------

# 当前作用：防止站位和导航算法尚未实现时返回空值、零值或伪成功。
# TODO(navigation-implementation)：三个生产方法真正实现时，必须在同一提交中把本组替换
# 为抓取停车目标、放置停车目标、距离与航向控制、到达判定、TTL、超时和无效Odom测试；
# 在生产方法仍未实现时不得删除本组。
def test_navigation_controller_methods_remain_explicitly_unimplemented() -> None:
    controller = NavigationController()
    task = TaskSpec(
        task_id=1,
        instruction="move the pink box",
        target_kind="box",
        target_body="box_body",
        target_color="pink",
        place_type="point",
        place_world_xyz=(1.0, 2.0, 0.5),
        place_radius=0.1,
    )
    target = ObjectEstimate3D("pink", (0.5, 0.0, 0.8), 0.9, "odom", 100)
    base = _base()
    goal = NavGoal("pick-1", "pick", (0.4, 0.0, 0.0), "odom", 0.05, 0.1, 1_000)

    with pytest.raises(NotImplementedError, match="底盘2负责人"):
        controller.build_pick_goal(task, target, base, 100)
    with pytest.raises(NotImplementedError, match="底盘2负责人"):
        controller.build_place_goal(task, base, 100)
    with pytest.raises(NotImplementedError, match="底盘2负责人"):
        controller.update(base, goal, 100)


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
    # ArmPlanner 构造验证（不调用 plan_grasp，避免依赖完整 fake IK 环境）
    planner_smoke = ArmPlanner(OfficialKDLAdapter())
    assert planner_smoke is not None
    # plan_grasp/plan_place 的完整行为由下方 ArmPlanner 分区的专项测试覆盖


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
    assert _strict_finite_number(perception["sync_slop_s"]) >= 0.0
    assert _strict_finite_number(perception["depth_unit_scale_m"]) > 0.0

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
    assert str(task_dir / "assets" / "robot.xml") in str(captured["xml"])
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
# Perception3DEstimator临时未实现约束
# ---------------------------------------------------------------------------

# 当前作用：防止物体中心补偿和滤波尚未实现时，把表面点、空结果或伪成功继续下传。
# TODO(perception-3d-implementation)：estimate真正实现时，必须在同一提交中替换为单帧
# 世界坐标、深度失败、表面到中心补偿、时间戳一致、多帧过滤和置信度测试；生产方法仍
# 未实现时不得删除本组。
def test_perception_3d_estimator_remains_unimplemented() -> None:
    estimator = Perception3DEstimator(CameraTransformProvider())
    detection = Detection2D("pink", (1.0, 1.0, 3.0, 3.0), 0.9, 100)
    with pytest.raises(NotImplementedError, match="视觉2负责人"):
        estimator.estimate(
            (detection,),
            _depth(np.ones((5, 5))),
            _intrinsics(),
            _base(),
            _actual_joints(),
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
    assert str(tmp_path / "examples" / "material_sorting") in str(exc_info.value)
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
# ArmPlanner 规划行为测试
# ---------------------------------------------------------------------------

# 使用 fake IK 适配器验证 plan_grasp/plan_place 的多阶段轨迹生成、
# 输入校验、IK 失败处理、关节跳变检测及坐标系转换。


class _FakeIKAdapter:
    """可配置返回值的 fake IK 适配器，供 ArmPlanner 单元测试使用。"""

    def __init__(self, *, solutions: list | None = None) -> None:
        """solutions: 按顺序返回的 IKResult 列表；耗尽后返回失败。"""
        self._solutions = solutions or []
        self._call_count = 0
        self.calls: list[dict] = []

    def solve_ik(self, actual_joints, *, left_target=None, right_target=None,
                 target_slide=None):
        self.calls.append({
            "left_target": left_target,
            "right_target": right_target,
            "target_slide": target_slide,
        })
        if self._call_count < len(self._solutions):
            result = self._solutions[self._call_count]
            self._call_count += 1
            return result
        # 默认返回失败
        from team_sorting.interfaces import IKResult
        return IKResult(0.1, None, None, False, "fake IK 耗尽")


def _make_fake_ik_success(slide: float = 0.1,
                          left: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                          right: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)) -> IKResult:
    """构造成功 IKResult。"""
    return IKResult(slide, left, right, True)


def _make_fake_ik_failure(reason: str = "fake failure") -> IKResult:
    """构造失败 IKResult。"""
    return IKResult(0.1, None, None, False, reason)


def test_plan_grasp_success() -> None:
    """验证有效输入时 plan_grasp 返回正确的 5 路点轨迹。"""
    # 所有阶段 IK 均成功
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1),  # pre-grasp
        _make_fake_ik_success(0.1),  # grasp
        _make_fake_ik_success(0.1),  # lift
        _make_fake_ik_success(0.1),  # retreat
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D("pink", (0.0, 0.8, 0.3), 0.9, "footprint", 100)
    joints = _planning_joints()

    gt, jt = planner.plan_grasp(target, joints)

    assert gt.valid
    assert gt.confidence == 1.0
    assert jt.valid
    assert len(jt.waypoints) == 5  # pre, grasp, close_wait, lift, retreat

    # 路点时间递增
    times = [wp.time_from_start_s for wp in jt.waypoints]
    for i in range(1, len(times)):
        assert times[i] > times[i - 1], f"路点 {i} 时间未递增"

    # 夹爪状态：阶段0（预抓取）张开，阶段1-4闭合
    assert jt.waypoints[0].joint_position[9] == pytest.approx(0.5)   # left_gripper OPEN
    assert jt.waypoints[0].joint_position[16] == pytest.approx(0.5)  # right_gripper OPEN
    for i in range(1, 5):
        assert jt.waypoints[i].joint_position[9] == pytest.approx(0.0)
        assert jt.waypoints[i].joint_position[16] == pytest.approx(0.0)

    # 验证 grasp_context 已设置
    assert planner._grasp_context is not None
    assert "left_offset" in planner._grasp_context
    assert "right_offset" in planner._grasp_context


def test_plan_grasp_ik_failure() -> None:
    """验证 IK 失败时返回 invalid 结果，不抛异常。"""
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_failure("预抓取无解"),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D("pink", (0.0, 0.8, 0.3), 0.9, "footprint", 100)
    joints = _planning_joints()

    gt, jt = planner.plan_grasp(target, joints)
    assert not gt.valid
    assert "预抓取IK失败" in gt.failure_reason
    assert not jt.valid


def test_plan_grasp_joint_jump() -> None:
    """验证关节跳变超过阈值时返回 invalid。"""
    # 第一次返回小关节值，第二次返回大幅跳变
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1, left=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                              right=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),
        _make_fake_ik_success(0.1, left=(2.0, 0.2, 0.3, 0.4, 0.5, 0.6),  # 关节0跳变 >0.5
                              right=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D("pink", (0.0, 0.8, 0.3), 0.9, "footprint", 100)
    joints = _planning_joints()

    gt, jt = planner.plan_grasp(target, joints)
    assert not gt.valid
    assert "跳变" in gt.failure_reason


def test_plan_place_without_context() -> None:
    """验证未执行抓取直接调用 plan_place 返回 invalid。"""
    fake = _FakeIKAdapter()
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    task = TaskSpec(
        task_id=1, instruction="move pink box",
        target_kind="box", target_body="box_body", target_color="pink",
        place_type="point", place_world_xyz=(2.0, 3.0, 0.5), place_radius=0.1,
    )
    joints = _planning_joints()

    pt, jt = planner.plan_place(task, joints)
    assert not pt.valid
    assert "未执行抓取" in pt.failure_reason


def test_plan_place_success() -> None:
    """验证先抓取后放置成功生成有效轨迹且上下文被清除。"""
    fake = _FakeIKAdapter(solutions=[
        # plan_grasp: 4 stages
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        # plan_place: 3 stages
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D("pink", (0.0, 0.8, 0.3), 0.9, "footprint", 100)
    joints = _planning_joints()

    # 先抓取
    gt, _ = planner.plan_grasp(target, joints)
    assert gt.valid
    assert planner._grasp_context is not None

    # 再放置
    task = TaskSpec(
        task_id=1, instruction="move pink box",
        target_kind="box", target_body="box_body", target_color="pink",
        place_type="point", place_world_xyz=(0.0, 1.8, 0.3), place_radius=0.1,
    )
    pt, jt_place = planner.plan_place(task, joints)
    assert pt.valid
    assert len(jt_place.waypoints) == 3
    # 上下文应被清除
    assert planner._grasp_context is None


def test_plan_grasp_frame_conversion() -> None:
    """验证 odom 坐标系的临时转换生效。"""
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    # odom 坐标 (0.0, 1.35, 0.3) -> footprint _FOOTPRINT_ROBOT_XY robot base
    # 转换后: (0.0 - (-0.70), 1.35 - 0.55, 0.3) ≈ (0.70, 0.80, 0.3)
    target = ObjectEstimate3D("pink", (0.0, 1.35, 0.3), 0.9, "odom", 100)
    joints = _planning_joints()

    gt, jt = planner.plan_grasp(target, joints)
    assert gt.valid
    # 验证 IK 调用使用了 footprint 坐标系的 Pose3D
    assert len(fake.calls) >= 1
    first_call = fake.calls[0]
    assert first_call["left_target"].frame_id == "footprint"
    # 验证转换后的位置近似在 robot 局部坐标下
    # odom(0.0, 1.35) - _FOOTPRINT_ROBOT_XY = footprint(0.70, 0.80)
    # 预抓取位置：target_xy - 0.15 * dir, Z + 0.20
    left_pos = first_call["left_target"].position_xyz
    # 预期：odom(0.0,1.35,0.3) - _FOOTPRINT_ROBOT_XY = footprint(0.70,0.80,0.3)
    # 预抓取在目标上方 0.20m，沿抓取方向后退 0.15m，再加垂直偏移
    assert 0.5 < left_pos[0] < 0.8
    assert 0.7 < left_pos[1] < 0.9
    assert 0.45 < left_pos[2] < 0.55

def test_box_size_compensation() -> None:
    """验证抓取位姿的左右夹爪按箱体半宽 + 安全间隙偏移。"""
    # robot_xy=_FOOTPRINT_ROBOT_XY, target在footprint系(0.0, 0.8, 0.3)
    # 抓取方向: dx=0.70, dy=0.25, dist≈0.7433
    # perp 方向: (-0.3363, 0.9417), grasp_offset=0.10
    # 左夹爪 X ≈ 0.0 + (-0.3363)*0.10 = -0.0336
    # 右夹爪 X ≈ 0.0 - (-0.3363)*0.10 = +0.0336
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1), _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1), _make_fake_ik_success(0.1),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D("pink", (0.0, 0.8, 0.3), 0.9, "footprint", 100)
    joints = _planning_joints()

    gt, jt = planner.plan_grasp(target, joints)
    assert gt.valid

    # 抓取阶段左右夹爪沿垂直方向各偏移 grasp_offset，方向相反
    # 验证左右偏移符号相反、绝对值相等
    left_x = gt.left_grasp.position_xyz[0]
    right_x = gt.right_grasp.position_xyz[0]
    # 左/右夹爪在 X 轴上符号相反（perp_x 为负时左负右正）
    assert abs(left_x) == pytest.approx(abs(right_x), abs=0.001)
    assert left_x <= 0 <= right_x  # 左负右正（perp_x<0 时）
    # 偏移量约为 0.10（half_width+0.02）乘以 perp 的 X 分量绝对值
    assert abs(left_x) == pytest.approx(0.0336, abs=0.005)

    # 抓取阶段 Z 应与目标 Z 一致
    assert gt.left_grasp.position_xyz[2] == pytest.approx(0.3)
    assert gt.right_grasp.position_xyz[2] == pytest.approx(0.3)

    # 验证 box 半宽 + 安全间隙 = 0.10 已应用：任意一对阶段位姿中，
    # 左右夹爪在垂直于抓取方向上的分离距离 ≈ 2 * 0.10 = 0.20
    import math
    sep = math.hypot(
        gt.left_grasp.position_xyz[0] - gt.right_grasp.position_xyz[0],
        gt.left_grasp.position_xyz[1] - gt.right_grasp.position_xyz[1],
    )
    assert sep == pytest.approx(0.20, abs=0.01)


def test_place_height_compensation() -> None:
    """验证放置时 release 高度使用箱体半高补偿，不会悬空。"""
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1), _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1), _make_fake_ik_success(0.1),
        # plan_place: 3 stages
        _make_fake_ik_success(0.1), _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D("pink", (0.0, 0.8, 0.3), 0.9, "footprint", 100)
    joints = _planning_joints()

    gt, _ = planner.plan_grasp(target, joints)
    assert gt.valid
    # 验证上下文 Z 偏移使用箱体半高补偿，非原始 lift delta
    ctx = planner._grasp_context
    from team_sorting.arm_planning import _BOX_HALF_HEIGHT, _PLACE_SURFACE_OFFSET, _LIFT_DELTA
    expected_z = _BOX_HALF_HEIGHT + _PLACE_SURFACE_OFFSET
    assert ctx["left_offset"][2] == pytest.approx(expected_z)
    assert ctx["right_offset"][2] == pytest.approx(expected_z)
    # 确认不再是原始的 _LIFT_DELTA
    assert ctx["left_offset"][2] != pytest.approx(_LIFT_DELTA)

    # 执行放置，验证 release 位姿的 Z 使用了箱体补偿
    task = TaskSpec(
        task_id=1, instruction="move pink box",
        target_kind="box", target_body="box_body", target_color="pink",
        place_type="point", place_world_xyz=(0.0, 1.8, 0.3), place_radius=0.1,
    )
    pt, jt_place = planner.plan_place(task, joints)
    assert pt.valid
    # release Z = place_center_z + box_half_height + surface_offset = 0.3 + 0.095 + 0.02 = 0.415
    assert pt.left_release.position_xyz[2] == pytest.approx(0.3 + expected_z)
    assert pt.right_release.position_xyz[2] == pytest.approx(0.3 + expected_z)


def test_grasp_target_too_far() -> None:
    """验证目标超出 _MAX_GRASP_DIST 时返回 invalid。"""
    fake = _FakeIKAdapter()
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    # 目标在 footprint 系中距离 robot 中心约 3.0m，远超 _MAX_GRASP_DIST=1.5m
    target = ObjectEstimate3D("pink", (3.0, 3.0, 0.3), 0.9, "footprint", 100)
    joints = _planning_joints()

    gt, jt = planner.plan_grasp(target, joints)
    assert not gt.valid
    assert "超出最大抓取距离" in gt.failure_reason


def test_reset_context() -> None:
    """验证 reset_context 能清空抓取上下文。"""
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1), _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1), _make_fake_ik_success(0.1),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D("pink", (0.0, 0.8, 0.3), 0.9, "footprint", 100)
    joints = _planning_joints()

    gt, _ = planner.plan_grasp(target, joints)
    assert gt.valid
    assert planner._grasp_context is not None

    planner.reset_context()
    assert planner._grasp_context is None




def test_grasp_stages_include_close_wait() -> None:
    """验证抓取轨迹包含5个阶段且 close_wait 夹爪保持闭合。"""
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1),  # pre-grasp
        _make_fake_ik_success(0.1),  # grasp
        _make_fake_ik_success(0.1),  # lift
        _make_fake_ik_success(0.1),  # retreat
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D('pink', (0.0, 0.8, 0.3), 0.9, 'footprint', 100)
    joints = _planning_joints()

    gt, jt = planner.plan_grasp(target, joints)

    assert gt.valid
    assert len(jt.waypoints) == 5

    # 阶段顺序: pre(张开) -> grasp(闭合) -> close_wait(闭合) -> lift(闭合) -> retreat(闭合)
    wp_pre = jt.waypoints[0]
    wp_grasp = jt.waypoints[1]
    wp_close = jt.waypoints[2]
    wp_lift = jt.waypoints[3]
    wp_retreat = jt.waypoints[4]

    # 夹爪状态
    assert wp_pre.joint_position[9] == pytest.approx(0.5)   # 预抓取：张开
    assert wp_pre.joint_position[16] == pytest.approx(0.5)
    assert wp_grasp.joint_position[9] == pytest.approx(0.0)  # 抓取：闭合
    assert wp_grasp.joint_position[16] == pytest.approx(0.0)
    assert wp_close.joint_position[9] == pytest.approx(0.0)  # close_wait：保持闭合
    assert wp_close.joint_position[16] == pytest.approx(0.0)
    assert wp_lift.joint_position[9] == pytest.approx(0.0)   # 抬升：保持闭合
    assert wp_lift.joint_position[16] == pytest.approx(0.0)
    assert wp_retreat.joint_position[9] == pytest.approx(0.0)  # 撤离：保持闭合
    assert wp_retreat.joint_position[16] == pytest.approx(0.0)

    # close_wait 关节位置与 grasp 相同（保持）
    assert wp_close.joint_position[0:9] == wp_grasp.joint_position[0:9]
    assert wp_close.joint_position[10:16] == wp_grasp.joint_position[10:16]

    # close_wait 时间比 grasp 多 _CLOSE_WAIT_TIME
    from team_sorting.arm_planning import _CLOSE_WAIT_TIME
    assert wp_close.time_from_start_s == pytest.approx(wp_grasp.time_from_start_s + _CLOSE_WAIT_TIME)

    # 时间严格递增
    times = [wp.time_from_start_s for wp in jt.waypoints]
    for i in range(1, len(times)):
        assert times[i] > times[i - 1], f'路点 {i} 时间未递增'


def test_grasp_close_wait_no_extra_ik_call() -> None:
    """验证 close_wait 阶段不额外调用 IK（复用 grasp IK 结果）。"""
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D('pink', (0.0, 0.8, 0.3), 0.9, 'footprint', 100)
    joints = _planning_joints()

    planner.plan_grasp(target, joints)

    # 5个阶段只有4次IK调用（close_wait复用grasp的IK）
    assert len(fake.calls) == 4
    assert fake._call_count == 4


def test_place_release_height_compensation() -> None:
    """验证放置释放高度使用箱体补偿（_BOX_HALF_HEIGHT + _PLACE_SURFACE_OFFSET）
    而非原始 _LIFT_DELTA。"""
    from team_sorting.arm_planning import _BOX_HALF_HEIGHT, _PLACE_SURFACE_OFFSET, _LIFT_DELTA

    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1),  # plan_grasp: pre
        _make_fake_ik_success(0.1),  # plan_grasp: grasp
        _make_fake_ik_success(0.1),  # plan_grasp: lift
        _make_fake_ik_success(0.1),  # plan_grasp: retreat
        _make_fake_ik_success(0.1),  # plan_place: preplace
        _make_fake_ik_success(0.1),  # plan_place: release
        _make_fake_ik_success(0.1),  # plan_place: retreat
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D('pink', (0.0, 0.8, 0.3), 0.9, 'footprint', 100)
    joints = _planning_joints()

    # 先抓取
    gt, _ = planner.plan_grasp(target, joints)
    assert gt.valid

    # 再放置
    place_z = 0.3
    task = TaskSpec(
        task_id=1, instruction='move pink box',
        target_kind='box', target_body='box_body', target_color='pink',
        place_type='point', place_world_xyz=(0.0, 1.8, place_z), place_radius=0.1,
    )
    pt, jt = planner.plan_place(task, joints)
    assert pt.valid

    # 验证释放位姿Z = place_world_xyz.z + _BOX_HALF_HEIGHT + _PLACE_SURFACE_OFFSET
    expected_release_z = place_z + _BOX_HALF_HEIGHT + _PLACE_SURFACE_OFFSET
    wrong_release_z = place_z + _LIFT_DELTA

    assert pt.left_release.position_xyz[2] == pytest.approx(expected_release_z)
    assert pt.left_release.position_xyz[2] != pytest.approx(wrong_release_z)


def test_box_size_compensation_in_grasp() -> None:
    """验证抓取位置使用箱体半宽进行侧向偏移补偿。"""
    from team_sorting.arm_planning import _BOX_HALF_WIDTH

    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]

    target = ObjectEstimate3D('pink', (0.0, 0.8, 0.3), 0.9, 'footprint', 100)
    joints = _planning_joints()

    gt, jt = planner.plan_grasp(target, joints)
    assert gt.valid

    # grasp_offset = _BOX_HALF_WIDTH + 0.02 = 0.10
    expected_offset = _BOX_HALF_WIDTH + 0.02
    # 验证左右夹爪相对目标中心的侧向偏移距离为 expected_offset
    # robot_xy = _FOOTPRINT_ROBOT_XY 是临时固定值，因此抓取方向非纯沿轴
    tx, ty, tz = target.position_xyz
    import math
    left_dx = gt.left_grasp.position_xyz[0] - tx
    left_dy = gt.left_grasp.position_xyz[1] - ty
    right_dx = gt.right_grasp.position_xyz[0] - tx
    right_dy = gt.right_grasp.position_xyz[1] - ty
    # 左右夹爪到目标中心的水平距离应等于 expected_offset
    assert math.hypot(left_dx, left_dy) == pytest.approx(expected_offset, rel=1e-6)
    assert math.hypot(right_dx, right_dy) == pytest.approx(expected_offset, rel=1e-6)
    # 左右夹爪对称分布在目标两侧
    assert left_dx == pytest.approx(-right_dx)
    assert left_dy == pytest.approx(-right_dy)

    # 夹爪Z坐标等于目标中心Z
    assert gt.left_grasp.position_xyz[2] == pytest.approx(tz)
    assert gt.right_grasp.position_xyz[2] == pytest.approx(tz)

    # 预抓取在抓取位置基础上后退0.15m并抬高0.20m
    assert gt.left_pregrasp.position_xyz[2] == pytest.approx(0.3 + 0.20)


def test_plan_place_ik_failure() -> None:
    """验证放置IK失败时返回 invalid 并清除上下文。"""
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1),  # plan_grasp: pre
        _make_fake_ik_success(0.1),  # plan_grasp: grasp
        _make_fake_ik_success(0.1),  # plan_grasp: lift
        _make_fake_ik_success(0.1),  # plan_grasp: retreat
        _make_fake_ik_failure('放置IK无解'),  # plan_place: preplace 失败
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D('pink', (0.0, 0.8, 0.3), 0.9, 'footprint', 100)
    joints = _planning_joints()

    gt, _ = planner.plan_grasp(target, joints)
    assert gt.valid
    assert planner._grasp_context is not None

    task = TaskSpec(
        task_id=1, instruction='move pink box',
        target_kind='box', target_body='box_body', target_color='pink',
        place_type='point', place_world_xyz=(0.0, 1.8, 0.3), place_radius=0.1,
    )
    pt, jt = planner.plan_place(task, joints)
    assert not pt.valid
    assert '放置IK无解' in pt.failure_reason
    # IK失败后上下文应清除
    assert planner._grasp_context is None


def test_plan_place_joint_jump() -> None:
    """验证放置时关节跳变超限返回 invalid 并清除上下文。"""
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1),  # plan_grasp: pre
        _make_fake_ik_success(0.1),  # plan_grasp: grasp
        _make_fake_ik_success(0.1),  # plan_grasp: lift
        _make_fake_ik_success(0.1),  # plan_grasp: retreat
        _make_fake_ik_success(0.1),  # plan_place: preplace
        _make_fake_ik_success(0.2, left=(2.0, 0.2, 0.3, 0.4, 0.5, 0.6),
                              right=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),  # release 关节跳变
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D('pink', (0.0, 0.8, 0.3), 0.9, 'footprint', 100)
    joints = _planning_joints()

    gt, _ = planner.plan_grasp(target, joints)
    assert gt.valid

    task = TaskSpec(
        task_id=1, instruction='move pink box',
        target_kind='box', target_body='box_body', target_color='pink',
        place_type='point', place_world_xyz=(0.0, 1.8, 0.3), place_radius=0.1,
    )
    pt, jt = planner.plan_place(task, joints)
    assert not pt.valid
    assert '跳变' in pt.failure_reason
    assert planner._grasp_context is None


def test_place_time_monotonic() -> None:
    """验证放置轨迹时间严格递增。"""
    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1), _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1), _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1), _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D('pink', (0.0, 0.8, 0.3), 0.9, 'footprint', 100)
    joints = _planning_joints()

    gt, _ = planner.plan_grasp(target, joints)
    assert gt.valid

    task = TaskSpec(
        task_id=1, instruction='move pink box',
        target_kind='box', target_body='box_body', target_color='pink',
        place_type='point', place_world_xyz=(0.0, 1.8, 0.3), place_radius=0.1,
    )
    pt, jt = planner.plan_place(task, joints)
    assert pt.valid
    assert len(jt.waypoints) >= 2

    times = [wp.time_from_start_s for wp in jt.waypoints]
    for i in range(1, len(times)):
        assert times[i] > times[i - 1], f'放置路点 {i} 时间未递增'


def test_grasp_lift_is_above_object_center() -> None:
    """验证抬升阶段高度正确：物体中心 + LIFT_DELTA。"""
    from team_sorting.arm_planning import _LIFT_DELTA

    fake = _FakeIKAdapter(solutions=[
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
        _make_fake_ik_success(0.1),
    ])
    planner = ArmPlanner(fake)  # type: ignore[arg-type]
    target = ObjectEstimate3D('pink', (0.0, 0.8, 0.3), 0.9, 'footprint', 100)
    joints = _planning_joints()

    gt, jt = planner.plan_grasp(target, joints)
    assert gt.valid
    assert gt.lift_delta_m == pytest.approx(_LIFT_DELTA)

    # 第4个IK调用是lift，验证其Z坐标
    lift_call = fake.calls[2]  # IK calls: pre(0), grasp(1), lift(2), retreat(3)
    assert lift_call['left_target'].position_xyz[2] == pytest.approx(0.3 + _LIFT_DELTA)


def test_arm_planner_rejects_dependency_without_solve_ik() -> None:
    with pytest.raises(TypeError, match="solve_ik"):
        ArmPlanner(object())  # type: ignore[arg-type]
