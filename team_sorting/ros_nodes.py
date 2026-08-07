"""三个 ROS2 节点入口和消息薄适配层。

本文件只负责 ROS2 订阅/发布、消息与 dataclass 转换、时间缓存、普通 Python 模块组装、
定时控制循环和官方控制话题发布；不实现 YOLO、三维几何、导航、IK、机械臂轨迹或 FSM
具体判断。三个 console script 分别调用 ``main_team_client``、``main_perception`` 和
``main_recorder``。输入输出为 ROS 消息和团队接口，时间统一为纳秒，坐标系沿用消息
``header.frame_id``。

``rclpy``、``vision_msgs``、``message_filters`` 和 ``cv_bridge`` 都在节点启动时延迟
导入，因此没有比赛环境时仍可导入本模块并运行纯 Python 单元测试。
TimestampedCache
JointStateMapper
OfficialCommandPublisher
_create_team_client_node()
    └── _TeamClientNode

_create_perception_node()
    └── _PerceptionNode

_create_recorder_node()
    └── _DatasetRecorderNode
"""
# 大量函数可以分为四组

# 不要逐个函数孤立地看，可以按功能分组。

# 第一组：三个启动入口
# main_team_client()
# main_perception()
# main_recorder()

# 作用是分别启动三个节点。

# 第二组：ROS运行环境
# _load_ros_dependencies()
# _validate_vision_schema()
# _load_config()
# _resolve_config_path()
# _spin()

# 负责：

# 加载ROS依赖；
# 检查消息版本；
# 找配置文件；
# 初始化ROS；
# spin节点；
# 退出时销毁节点。

# 这部分相当于“程序启动器”。

# 第三组：节点工厂
# _create_team_client_node()
# _create_perception_node()
# _create_recorder_node()

# 它们根据已经加载好的ROS依赖，创建真正的ROS节点类。

# 这部分相当于“组装三个ROS节点”。

# 第四组：消息翻译工具
# _stamp_to_ns()
# _set_stamp()
# _base_state_from_odom()
# _estimates_to_vision()
# _estimates_from_vision()
# _validated_confidence()
# _vision_result_pose()

# 它们负责ROS消息和团队dataclass之间的转换。

# 例如：

# ROS Odometry
# → BaseState
# ObjectEstimate3D
# → ROS Detection3DArray
# ROS Detection3DArray
# → ObjectEstimate3D

# 这部分相当于“翻译器”。
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, fields, replace
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Mapping, Optional

from .action_mux import ActionMux, ActionMuxConfig
from .controller_manifest import validate_controller_config
from .arm_planning import ArmPlanner, OfficialKDLAdapter
from .arm_execution import ArmExecutionController
from .competition_context import CompetitionContext, CompetitionRunCoordinator
from .external_candidate import (
    ExternalCandidateConfig,
    ExternalCandidateConsumer,
    ExternalCandidateDecision,
)
from .fsm import FSMEvent, GlobalFSM, InstructionParser
from .interfaces import (
    ActionDispatchRecord,
    ActionMuxDecision,
    ArmExecutionConfig,
    ArmPlanningConfig,
    BaseCommand,
    BaseState,
    CameraIntrinsics,
    DepthFrame,
    DispatchGroupRecord,
    DispatchMode,
    Float64MultiArrayExactPayload,
    FSMStatus,
    FinalAction,
    GlobalPhase,
    GraspContext,
    GraspVerification,
    JointTrajectory,
    ManipulationCommand,
    ManipulationStatus,
    NavGoal,
    NavigationStatus,
    ObjectEstimate3D,
    RGBFrame,
    RigidTransform3D,
    RobotJointState,
    SensorSnapshot,
    SlotType,
    TaskSpec,
    TwistExactPayload,
    action_dispatch_to_json,
    final_action_from_json,
    final_action_to_json,
    fsm_status_from_json,
    fsm_status_to_json,
)
from .navigation import (
    Bounds3D,
    NavigationConfig,
    NavigationController,
    classify_slot_type,
)
from .perception_2d import Detection2DStabilizer, OfficialYoloAdapter
from .perception_3d import (
    CameraTransformProvider,
    Perception3DEstimator,
    VisualObservationVerifier,
)
from .recording_contracts import ActionPairingConfig
from .recorder_runtime import (
    RecorderRuntimeConfig,
    RecorderRuntimeManager,
    resolve_rosbag_qos_overrides_path,
)


@dataclass
class _RuntimeWiringState:
    """TeamClient私有、可清理的接线状态；缓存存在不表示业务已经成功。"""

    run_id: str = ""
    task_id: Optional[int] = None
    attempt: Optional[int] = None
    context_finished: bool = False
    last_handled_phase: Optional[GlobalPhase] = None
    phase_generation: int = 0
    active_target: Optional[ObjectEstimate3D] = None
    preferred_object_id: str = ""
    active_nav_goal: Optional[NavGoal] = None
    planned_grasp_context: Optional[GraspContext] = None
    confirmed_grasp_context: Optional[GraspContext] = None
    active_pick_trajectory: Optional[JointTrajectory] = None
    active_place_trajectory: Optional[JointTrajectory] = None
    active_trajectory_id: str = ""
    pick_observation_before_lift: Optional[ObjectEstimate3D] = None
    latest_grasp_verification: Optional[GraspVerification] = None
    place_observation_before_release: Optional[ObjectEstimate3D] = None
    latest_place_verification_observation: Optional[ObjectEstimate3D] = None
    last_navigation_status: Optional[NavigationStatus] = None
    last_manipulation_status: Optional[ManipulationStatus] = None
    phase_event_keys: set[tuple[object, ...]] = field(default_factory=set)
    phase_entry_failure_reason: str = ""
    last_event_tick_ns: Optional[int] = None
    event_attempted_this_tick: bool = False


@dataclass(frozen=True)
class _RuntimeFSMFeedback:
    """业务模块真实回执的私有适配器，不扩展公共消息契约。"""

    event: FSMEvent
    source_timestamp_ns: int
    run_id: str
    task_id: Optional[int]
    attempt: Optional[int]
    reason: str = ""
    object_id: str = ""
    goal_id: str = ""
    trajectory_id: str = ""
    confirmed_by_real_feedback: bool = False


_RUNTIME_ID_UNSET = object()
_MAX_PHASE_EVENT_KEYS = 32
_INTERNAL_STOP_PHASES = {
    GlobalPhase.WAIT_READY,
    GlobalPhase.LOAD_TASK,
    GlobalPhase.SAFE_HOLD,
    GlobalPhase.DONE,
    GlobalPhase.FAILED,
}


def _select_target_estimate(
    estimates: tuple[ObjectEstimate3D, ...],
    task: TaskSpec,
    now_ns: int,
    max_age_ns: int,
    *,
    phase_entered_ns: Optional[int] = None,
    preferred_object_id: Optional[str] = None,
    require_geometry: bool = False,
) -> Optional[ObjectEstimate3D]:
    """确定性选择当前任务的odom目标，不产生FSM事件或补造感知事实。"""

    if not isinstance(task, TaskSpec) or not task.valid:
        return None
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("now_ns必须是真正非负整数")
    if type(max_age_ns) is not int or max_age_ns < 0:
        raise ValueError("max_age_ns必须是真正非负整数")
    if phase_entered_ns is not None and (
        type(phase_entered_ns) is not int or phase_entered_ns < 0
    ):
        raise ValueError("phase_entered_ns必须是真正非负整数或None")
    preferred = (
        preferred_object_id.strip()
        if isinstance(preferred_object_id, str) and preferred_object_id.strip()
        else ""
    )
    candidates: list[ObjectEstimate3D] = []
    for estimate in estimates:
        if not isinstance(estimate, ObjectEstimate3D) or not estimate.valid:
            continue
        if estimate.frame_id != "odom" or estimate.class_id != task.target_color:
            continue
        if not isinstance(estimate.object_id, str) or not estimate.object_id.strip():
            continue
        if type(estimate.timestamp_ns) is not int:
            continue
        age_ns = now_ns - estimate.timestamp_ns
        if age_ns < 0 or age_ns > max_age_ns:
            continue
        try:
            confidence = float(estimate.confidence)
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            isinstance(estimate.confidence, bool)
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            continue
        if phase_entered_ns is not None and estimate.timestamp_ns < phase_entered_ns:
            continue
        if require_geometry and (
            estimate.orientation_xyzw is None or estimate.size_xyz_m is None
        ):
            continue
        if require_geometry:
            try:
                orientation = tuple(
                    float(value) for value in estimate.orientation_xyzw
                )
                size = tuple(float(value) for value in estimate.size_xyz_m)
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                len(orientation) != 4
                or not all(math.isfinite(value) for value in orientation)
                or math.sqrt(sum(value * value for value in orientation)) <= 1e-12
                or len(size) != 3
                or not all(math.isfinite(value) and value > 0.0 for value in size)
            ):
                continue
        candidates.append(estimate)
    if preferred and any(item.object_id.strip() == preferred for item in candidates):
        candidates = [
            item for item in candidates if item.object_id.strip() == preferred
        ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (-float(item.confidence), item.object_id.strip()),
    )


def _base_planar_transform_snapshot(
    base: BaseState,
    *,
    source_frame: str,
    target_frame: str,
) -> RigidTransform3D:
    """由BaseState构造world/odom与footprint间的平面刚体变换快照。

    当前冻结约定下world与odom数值原点及轴方向对齐；本辅助不发布TF。footprint只
    使用BaseState的x、y、yaw，明确忽略base_link可能包含的z、roll和pitch。
    """

    if not isinstance(base, BaseState) or not base.valid:
        raise ValueError("有效BaseState是构造footprint变换的必要条件")
    if base.frame_id != "odom":
        raise ValueError("当前冻结变换辅助只接受frame_id='odom'的BaseState")
    if source_frame not in {"world", "odom", "footprint"} or target_frame not in {
        "world",
        "odom",
        "footprint",
    }:
        raise ValueError("变换frame只允许world、odom和footprint")
    if (source_frame == "footprint") == (target_frame == "footprint"):
        raise ValueError("变换方向必须且只能有一端是footprint")
    x, y = float(base.position_xyz[0]), float(base.position_xyz[1])
    yaw = float(base.yaw)
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise ValueError("BaseState平面位姿必须是有限数")
    half_yaw = yaw / 2.0
    if source_frame == "footprint":
        translation = (x, y, 0.0)
        rotation = (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))
    else:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        translation = (-cosine * x - sine * y, sine * x - cosine * y, 0.0)
        rotation = (0.0, 0.0, -math.sin(half_yaw), math.cos(half_yaw))
    return RigidTransform3D(
        source_frame=source_frame,
        target_frame=target_frame,
        translation_xyz=translation,
        rotation_xyzw=rotation,
        timestamp_ns=base.timestamp_ns,
        valid=True,
    )


def _internal_fsm_publish_authorization(
    *,
    observe_only: bool,
    enable_official_publish: bool,
    context: Optional[CompetitionContext],
    snapshot: SensorSnapshot,
    fsm_status: FSMStatus,
    base_command: Optional[BaseCommand],
    manipulation_command: Optional[ManipulationCommand],
    final_action: FinalAction,
    now_ns: int,
    manipulation_source: str = "manipulation_command",
) -> bool:
    """纯判断未来内部FSM FULL发布资格；本轮控制链不会使用它打开发布。"""

    if type(observe_only) is not bool or type(enable_official_publish) is not bool:
        return False
    if observe_only or not enable_official_publish:
        return False
    if not isinstance(context, CompetitionContext) or not context.valid or context.finished:
        return False
    if context.active_task is None or context.current_task_id != context.active_task.task_id:
        return False
    if not isinstance(snapshot, SensorSnapshot) or not snapshot.valid:
        return False
    if snapshot.timestamp_ns != now_ns:
        return False
    if snapshot.task is None or snapshot.task.task_id != context.current_task_id:
        return False
    if snapshot.base is None or not snapshot.base.valid:
        return False
    if snapshot.joints is None or not snapshot.joints.valid:
        return False
    if not isinstance(fsm_status, FSMStatus) or fsm_status.global_phase in _INTERNAL_STOP_PHASES:
        return False
    if fsm_status.timestamp_ns != now_ns:
        return False
    if fsm_status.task_id != context.current_task_id:
        return False
    if not isinstance(final_action, FinalAction) or not final_action.valid:
        return False
    if final_action.timestamp_ns != now_ns:
        return False
    if final_action.global_phase is not fsm_status.global_phase:
        return False
    if type(now_ns) is not int or now_ns < 0:
        return False
    if manipulation_source != "manipulation_command":
        # External Candidate只保留既有head-only授权，不能借内部判断取得FULL权限。
        return False
    base_active = (
        isinstance(base_command, BaseCommand)
        and base_command.valid
        and now_ns < base_command.valid_until_ns
        and (base_command.v != 0.0 or base_command.w != 0.0)
    )
    manipulation_active = (
        isinstance(manipulation_command, ManipulationCommand)
        and manipulation_command.valid
        and now_ns < manipulation_command.valid_until_ns
        and any(manipulation_command.controlled_mask)
    )
    return base_active or manipulation_active


class TimestampedCache:
    """按纳秒时间保存并查找最近状态的有界缓存。

    参数：``max_items`` 为最大对象数。``put`` 输入时间戳和任意状态；``nearest`` 返回
    与目标时间最接近且不超过容差的对象。时间单位纳秒，空间坐标系由被缓存对象自身
    声明。空缓存或时间差过大时返回 ``None``，不伪造同步状态。
    """

    def __init__(self, max_items: int = 100) -> None:
        if max_items <= 0:
            raise ValueError("时间缓存容量必须大于 0")
        self._items: deque[tuple[int, Any]] = deque(maxlen=int(max_items))

    def put(self, timestamp_ns: int, value: Any) -> None:
        """插入一个带时间戳状态。

        参数：纳秒时间和状态对象；返回：无。对象单位/坐标系保持不变。失败：时间戳
        无法转为整数时抛出 ``TypeError``/``ValueError``；超过容量会自动丢弃最旧项。
        """

        self._items.append((int(timestamp_ns), value))

    def nearest(self, timestamp_ns: int, max_delta_ns: int) -> Optional[Any]:
        """寻找目标时间附近的最近状态。

        参数：目标纳秒和最大允许时间差纳秒。返回：最近对象或 ``None``；不改变对象的
        坐标系/单位。失败：容差为负时抛出 ``ValueError``。
        """

        if max_delta_ns < 0:
            raise ValueError("最大时间差不能为负数")
        if not self._items:
            return None
        best_time, best_value = min(self._items, key=lambda item: abs(item[0] - timestamp_ns))
        if abs(best_time - timestamp_ns) > max_delta_ns:
            return None
        return best_value

    def clear(self) -> None:
        """清空缓存；新消息已确认无效时，旧反馈不能继续冒充当前状态。"""

        self._items.clear()


class JointStateMapper:
    """ROS JointState 到团队固定 17 维顺序的映射器。

    参数：可选名称别名表。输入 ROS JointState，输出 ``RobotJointState``；slide 单位米、
    旋转关节弧度、夹爪沿用官方 0～1，时间来自消息 header。缺少必需关节、位置非有限
    或名称重复时抛出清晰 ``ValueError``，不会用零值冒充实际反馈。
    """

    def __init__(self, aliases: Optional[dict[str, str]] = None) -> None:
        from .interfaces import JOINT_NAMES

        self.joint_names = JOINT_NAMES
        self.aliases = dict(aliases or {})

    def map_message(self, message: Any) -> RobotJointState:
        """按名称重排一个 ROS ``sensor_msgs/JointState``。

        参数：含 name/position/velocity/effort/header 的消息。返回固定 17 维实际反馈，
        单位沿用 ROS/官方关节定义，坐标系不适用。失败：时间戳、必需名称、数组长度或
        数值不合法时抛出 ``ValueError``。
        """

        names = [self.aliases.get(str(name), str(name)) for name in message.name]
        if len(names) != len(set(names)):
            raise ValueError("JointState 经别名映射后出现重复关节名")
        index_by_name = {name: index for index, name in enumerate(names)}
        missing = [name for name in self.joint_names if name not in index_by_name]
        if missing:
            raise ValueError(f"JointState 缺少团队必需关节：{missing}")
        if len(message.position) < len(message.name):
            raise ValueError("JointState.position 长度小于 name 长度")

        def _optional_array(field_name: str) -> tuple[float, ...]:
            values = getattr(message, field_name, ())
            if len(values) == 0:
                return (0.0,) * 17
            if len(values) < len(message.name):
                raise ValueError(f"JointState.{field_name} 长度小于 name 长度")
            return tuple(float(values[index_by_name[name]]) for name in self.joint_names)

        position = tuple(float(message.position[index_by_name[name]]) for name in self.joint_names)
        return RobotJointState(
            position=position,
            velocity=_optional_array("velocity"),
            effort=_optional_array("effort"),
            timestamp_ns=_stamp_to_ns(message.header.stamp),
        )


_DISPATCH_GROUP_LAYOUT = (
    ("base", "cmd_vel", "geometry_msgs/msg/Twist", 0, 2),
    ("spine", "slide", "std_msgs/msg/Float64MultiArray", 2, 3),
    ("head", "head", "std_msgs/msg/Float64MultiArray", 3, 5),
    ("left_arm", "left_arm", "std_msgs/msg/Float64MultiArray", 5, 12),
    ("right_arm", "right_arm", "std_msgs/msg/Float64MultiArray", 12, 19),
)


def _empty_dispatch_group_records(topics: Mapping[str, str]) -> list[DispatchGroupRecord]:
    return [
        DispatchGroupRecord(group, topics[topic_key], message_type, False, None, None)
        for group, topic_key, message_type, _start, _stop in _DISPATCH_GROUP_LAYOUT
    ]


class OfficialPublishError(RuntimeError):
    """保留原 RuntimeError 传播语义，同时携带异常发生前的逐组调用事实。"""

    def __init__(self, message: str, group_records: tuple[DispatchGroupRecord, ...]) -> None:
        super().__init__(message)
        self.group_records = group_records


class OfficialCommandPublisher:
    """唯一允许发布五组官方机器人控制话题的适配器。

    参数：ROS2 node、话题配置和已加载依赖。输入同一个 ``FinalAction`` 对象；输出依次
    为底盘 Twist、slide 1 项、head 2 项、左臂含夹爪 7 项、右臂含夹爪 7 项。底盘单位
    m/s、rad/s，slide 米、臂关节弧度、夹爪 0～1。消息创建/发布失败会抛出
    ``RuntimeError``，本类不重建第二份动作或执行控制算法。
    """

    def __init__(
        self,
        node: Any,
        topics: dict[str, str],
        ros: Optional[SimpleNamespace] = None,
        head_target_tracking: Optional[dict[str, Any]] = None,
    ) -> None:
        self._ros = ros or _load_ros_dependencies(require_vision=False, require_filters=False)
        required = ("cmd_vel", "slide", "head", "left_arm", "right_arm")
        missing = [name for name in required if not topics.get(name)]
        if missing:
            raise RuntimeError(f"官方控制话题配置缺失：{missing}")
        self._node = node
        self._topics = dict(topics)
        self._head_topic = topics["head"]
        tracking = head_target_tracking or {
            "enabled": False,
            "fresh_reset_confirmed": False,
            "initial_yaw_target": 0.0,
            "initial_pitch_target": 0.0,
            "require_exclusive_writer": True,
            "yaw_lower": -0.5,
            "yaw_upper": 0.5,
            "pitch_lower": -1.18,
            "pitch_upper": 0.16,
        }
        self._head_tracking_enabled = bool(tracking["enabled"])
        self._require_exclusive_head_writer = bool(
            tracking["require_exclusive_writer"]
        )
        self._head_yaw_bounds = (
            float(tracking["yaw_lower"]),
            float(tracking["yaw_upper"]),
        )
        self._head_pitch_bounds = (
            float(tracking["pitch_lower"]),
            float(tracking["pitch_upper"]),
        )
        self.last_head_controller_target: Optional[list[float]] = None
        if self._head_tracking_enabled and tracking["fresh_reset_confirmed"]:
            self.last_head_controller_target = [
                float(tracking["initial_yaw_target"]),
                float(tracking["initial_pitch_target"]),
            ]
        self.last_head_publish_failure_reason = ""
        self._base = node.create_publisher(self._ros.Twist, topics["cmd_vel"], 10)
        self._slide = node.create_publisher(self._ros.Float64MultiArray, topics["slide"], 10)
        self._head = node.create_publisher(self._ros.Float64MultiArray, topics["head"], 10)
        self._left = node.create_publisher(self._ros.Float64MultiArray, topics["left_arm"], 10)
        self._right = node.create_publisher(self._ros.Float64MultiArray, topics["right_arm"], 10)

    def publish(self, action: FinalAction) -> None:
        """把唯一 FinalAction 拆分并发布到五组官方话题。

        参数：严格 19 维且有限的动作对象；返回：无。第 0/1 项为 base_link 速度，随后
        是 slide、head、左臂+夹爪、右臂+夹爪。失败：动作无效或 ROS 发布异常时抛出
        ``RuntimeError``；调用方必须把同一对象用于 ``/team/final_action`` 遥测。
        """

        self.publish_with_trace(action)

    def _empty_group_records(self) -> list[DispatchGroupRecord]:
        return _empty_dispatch_group_records(self._topics)

    def publish_with_trace(self, action: FinalAction) -> tuple[DispatchGroupRecord, ...]:
        """按原顺序 full 发布，并返回或随异常携带精确逐组 payload。"""

        records = self._empty_group_records()
        if not action.valid:
            raise OfficialPublishError(
                f"拒绝发布无效 FinalAction：{action.failure_reason}", tuple(records)
            )
        values = action.values
        base = self._ros.Twist()
        base.linear.x, base.linear.y, base.linear.z = values[0], 0.0, 0.0
        base.angular.x, base.angular.y, base.angular.z = 0.0, 0.0, values[1]
        slide = self._ros.Float64MultiArray()
        slide.data = list(values[2:3])
        head = self._ros.Float64MultiArray()
        head.data = list(values[3:5])
        left = self._ros.Float64MultiArray()
        left.data = list(values[5:12])
        right = self._ros.Float64MultiArray()
        right.data = list(values[12:19])
        calls = (
            (
                0,
                self._base,
                base,
                TwistExactPayload(
                    (base.linear.x, base.linear.y, base.linear.z),
                    (base.angular.x, base.angular.y, base.angular.z),
                ),
            ),
            (1, self._slide, slide, Float64MultiArrayExactPayload(tuple(slide.data))),
            (2, self._head, head, Float64MultiArrayExactPayload(tuple(head.data))),
            (3, self._left, left, Float64MultiArrayExactPayload(tuple(left.data))),
            (4, self._right, right, Float64MultiArrayExactPayload(tuple(right.data))),
        )
        for index, publisher, message, payload in calls:
            record = records[index]
            try:
                publisher.publish(message)
            except Exception as exc:  # noqa: BLE001 - 保留部分成功并停止后续调用
                records[index] = replace(
                    record,
                    attempted=True,
                    succeeded=False,
                    exact_payload=payload,
                    failure_reason=str(exc),
                )
                raise OfficialPublishError(
                    f"发布官方控制话题失败：{exc}", tuple(records)
                ) from exc
            records[index] = replace(
                record, attempted=True, succeeded=True, exact_payload=payload
            )
        return tuple(records)

    def head_writer_is_exclusive(self) -> bool:
        """只在ROS graph确认本节点是head话题唯一publisher时返回真。"""

        try:
            count = self._node.count_publishers(self._head_topic)
        except (AttributeError, RuntimeError):
            return False
        return type(count) is int and count == 1

    def publish_head(self, action: FinalAction) -> None:
        """以绝对controller target只发布head分组。

        ``FinalAction[3]``仍是ActionMux验证后的绝对yaw关节目标；pitch不能取
        ``FinalAction[4]``中的物理反馈保持值，而必须复用fresh-reset作用域内维护的
        controller target shadow。发布成功后才更新yaw shadow，失败时保持原值。
        """

        self.publish_head_with_trace(action)

    def publish_head_with_trace(
        self, action: FinalAction
    ) -> tuple[DispatchGroupRecord, ...]:
        """只发布 head，并精确记录 yaw 与 controller-target shadow pitch。"""

        records = self._empty_group_records()
        if not action.valid:
            raise OfficialPublishError(
                f"拒绝发布无效 FinalAction：{action.failure_reason}", tuple(records)
            )
        if not self._head_tracking_enabled:
            self.last_head_publish_failure_reason = "head_target_tracking_disabled"
            raise OfficialPublishError(self.last_head_publish_failure_reason, tuple(records))
        if self.last_head_controller_target is None:
            self.last_head_publish_failure_reason = "fresh_reset_not_confirmed"
            raise OfficialPublishError(self.last_head_publish_failure_reason, tuple(records))
        if self._require_exclusive_head_writer and not self.head_writer_is_exclusive():
            self.last_head_publish_failure_reason = "head_writer_not_exclusive"
            raise OfficialPublishError(self.last_head_publish_failure_reason, tuple(records))
        yaw_target = action.values[3]
        if isinstance(yaw_target, bool) or not isinstance(yaw_target, (int, float)):
            self.last_head_publish_failure_reason = "head_yaw_target_invalid"
            raise OfficialPublishError(self.last_head_publish_failure_reason, tuple(records))
        yaw_target = float(yaw_target)
        if not math.isfinite(yaw_target) or not (
            self._head_yaw_bounds[0] <= yaw_target <= self._head_yaw_bounds[1]
        ):
            self.last_head_publish_failure_reason = "head_yaw_target_out_of_range"
            raise OfficialPublishError(self.last_head_publish_failure_reason, tuple(records))
        pitch_target = self.last_head_controller_target[1]
        if not math.isfinite(pitch_target) or not (
            self._head_pitch_bounds[0]
            <= pitch_target
            <= self._head_pitch_bounds[1]
        ):
            self.last_head_publish_failure_reason = "head_pitch_shadow_invalid"
            raise OfficialPublishError(self.last_head_publish_failure_reason, tuple(records))
        head = self._ros.Float64MultiArray()
        head.data = [yaw_target, pitch_target]
        payload = Float64MultiArrayExactPayload(tuple(head.data))
        try:
            self._head.publish(head)
        except Exception as exc:  # noqa: BLE001 - ROS 中间件异常统一说明
            self.last_head_publish_failure_reason = "head_publish_failed"
            records[2] = replace(
                records[2],
                attempted=True,
                succeeded=False,
                exact_payload=payload,
                failure_reason=str(exc),
            )
            raise OfficialPublishError(
                f"发布官方head控制话题失败：{exc}", tuple(records)
            ) from exc
        records[2] = replace(
            records[2], attempted=True, succeeded=True, exact_payload=payload
        )
        self.last_head_controller_target[0] = yaw_target
        self.last_head_publish_failure_reason = ""
        return tuple(records)

    def publish_emergency_base_stop(self) -> None:
        """仅在未来明确授权的底盘控制故障路径尽力发布零速度。

        正常控制仍必须经过 ActionMux 和完整 ``FinalAction``。当可靠 JointState 已经
        不可用时，不能伪造 17 维全零关节目标；这个窄接口只停止底盘，不发布机械臂
        目标，也不能证明 Server 已接收或机器人已经停止。
        """

        base = self._ros.Twist()
        base.linear.x = 0.0
        base.angular.z = 0.0
        try:
            self._base.publish(base)
        except Exception as exc:  # noqa: BLE001 - 转成边界错误并由生命周期调用方记录
            raise RuntimeError(f"紧急发布底盘零速度失败：{exc}") from exc


def _build_action_dispatch_record(
    action: FinalAction,
    decision: ActionMuxDecision,
    *,
    publish_enabled: bool,
    publisher_created: bool,
    dispatch_mode: DispatchMode,
    group_records: tuple[DispatchGroupRecord, ...],
    failure_reason: str = "",
) -> ActionDispatchRecord:
    """从 publisher 边界事实构造固定 19 维、未发送项为 null 的记录。"""

    dispatched: list[Optional[float]] = [None] * 19
    for record, (_group, _topic_key, _message_type, start, stop) in zip(
        group_records, _DISPATCH_GROUP_LAYOUT
    ):
        if not record.attempted:
            continue
        if isinstance(record.exact_payload, TwistExactPayload):
            dispatched[0] = record.exact_payload.linear_xyz[0]
            dispatched[1] = record.exact_payload.angular_xyz[2]
        elif isinstance(record.exact_payload, Float64MultiArrayExactPayload):
            if len(record.exact_payload.data) != stop - start:
                raise ValueError(f"{record.group} exact_payload 长度与官方分组不一致")
            dispatched[start:stop] = record.exact_payload.data
        else:
            raise ValueError(f"{record.group} attempted 但缺少受支持的 exact_payload")
    mask = tuple(value is not None for value in dispatched)
    attempted = tuple(record.group for record in group_records if record.attempted)
    successful = tuple(record.group for record in group_records if record.succeeded is True)
    failed = tuple(record.group for record in group_records if record.succeeded is False)
    return ActionDispatchRecord(
        schema_name="MMK2ActionDispatchRecord",
        schema_version=1,
        sequence=action.sequence,
        timestamp_ns=action.timestamp_ns,
        final_action_sequence=action.sequence,
        decision=decision,
        calculated=True,
        publish_enabled=publish_enabled,
        publisher_created=publisher_created,
        publish_attempted=bool(attempted),
        publisher_call_succeeded=None if not attempted else not failed,
        dispatch_mode=dispatch_mode,
        dispatched_action=tuple(dispatched),
        dispatched_mask=mask,
        attempted_groups=attempted,
        successful_groups=successful,
        failed_groups=failed,
        group_records=group_records,
        controller_accepted=None,
        execution_confirmed=None,
        failure_reason=failure_reason,
    )


def main_team_client(args: Optional[list[str]] = None) -> None:
    """启动任务决策与控制客户端节点。

    参数：可选 ROS2 命令行参数；返回：正常关闭时无返回。节点处理消息各自的米/弧度和
    frame_id。失败：缺少 rclpy/vision_msgs/配置时抛出清晰 ``RuntimeError``；
    完整业务算法未实现时默认仅计算和发布团队诊断遥测，不创建官方控制发布器，也不
    伪造任务完成。只有显式通过全局发布门时才允许进入官方控制发布链。
    """

    ros = _load_ros_dependencies(require_vision=True, require_filters=False)
    config = _load_config()
    node_class = _create_team_client_node(ros)
    _spin(ros, node_class, config, args)


def main_perception(args: Optional[list[str]] = None) -> None:
    """启动独立二维/三维感知节点。

    参数：可选 ROS2 命令行参数；返回：正常关闭时无返回。RGB/Depth 近似同步，状态按
    纳秒选最近值，三维输出位于配置 frame、单位米。失败：缺少 ROS、vision_msgs、
    cv_bridge、官方 YOLO/MMK2FK 或资源时启动即清晰报错，不启用伪检测。
    """

    ros = _load_ros_dependencies(require_vision=True, require_filters=True)
    config = _load_config()
    node_class = _create_perception_node(ros)
    _spin(ros, node_class, config, args)


def main_recorder(args: Optional[list[str]] = None) -> None:
    """启动独立数据记录节点。

    参数：可选 ROS2 命令行参数；返回：正常关闭时无返回。节点保存任务、裁判和团队
    遥测元数据，并用外部 ``ros2 bag record`` 原样保存高带宽消息及其单位/frame。
    失败：未显式启用、ROS 缺失、目录不可写或 bag 子进程启动失败时清晰报错，不实现
    伪 ``rosbag2_py`` 写入。
    """

    ros = _load_ros_dependencies(require_vision=False, require_filters=False)
    config = _load_config()
    node_class = _create_recorder_node(ros)
    _spin(ros, node_class, config, args)


def _load_ros_dependencies(require_vision: bool, require_filters: bool) -> SimpleNamespace:
    modules: dict[str, Any] = {}
    required = {
        "rclpy": "rclpy",
        "Node": "rclpy.node",
        "Image": "sensor_msgs.msg",
        "CameraInfo": "sensor_msgs.msg",
        "JointState": "sensor_msgs.msg",
        "Odometry": "nav_msgs.msg",
        "String": "std_msgs.msg",
        "Int32": "std_msgs.msg",
        "Float64MultiArray": "std_msgs.msg",
        "Twist": "geometry_msgs.msg",
    }
    errors: list[str] = []
    for key, module_name in required.items():
        try:
            module = importlib.import_module(module_name)
            if key == "rclpy":
                modules[key] = module
            else:
                modules[key] = getattr(module, key)
        except Exception as exc:  # noqa: BLE001 - 汇总未 source 和缺包问题
            errors.append(f"{module_name}.{key}: {exc}")
    if errors:
        raise RuntimeError(
            "ROS2 基础依赖不可用；搜索了当前 PYTHONPATH/AMENT_PREFIX_PATH；"
            f"错误={errors}。请 source 评测环境 ROS2 setup.bash，并检查 ROS_DISTRO。"
        )

    if require_vision:
        vision_names = ("Detection3DArray", "Detection3D", "ObjectHypothesisWithPose")
        try:
            vision_module = importlib.import_module("vision_msgs.msg")
            for name in vision_names:
                modules[name] = getattr(vision_module, name)
            _validate_vision_schema(SimpleNamespace(**modules))
        except Exception as exc:  # noqa: BLE001 - 明确报告实际 schema/包问题
            raise RuntimeError(
                "vision_msgs 不可用或消息字段与适配层不兼容；搜索了当前 AMENT_PREFIX_PATH "
                f"中的 vision_msgs/msg/Detection3DArray，错误={exc}。请安装与评测环境一致的 "
                "vision_msgs，并运行 `ros2 interface show vision_msgs/msg/Detection3DArray` "
                "核对字段；本项目不提供自定义消息替代。"
            ) from exc
    if require_filters:
        for key, module_name in (("message_filters", "message_filters"), ("CvBridge", "cv_bridge")):
            try:
                module = importlib.import_module(module_name)
                modules[key] = module if key == "message_filters" else getattr(module, key)
            except Exception as exc:  # noqa: BLE001 - 节点启动依赖错误
                raise RuntimeError(
                    f"感知节点缺少 {module_name}；搜索了当前 PYTHONPATH，错误={exc}。"
                    "请安装 ROS2 对应包并重新 source 环境。"
                ) from exc
    return SimpleNamespace(**modules)


def _validate_vision_schema(ros: SimpleNamespace) -> None:
    array = ros.Detection3DArray()
    detection = ros.Detection3D()
    result = ros.ObjectHypothesisWithPose()
    if not hasattr(array, "header") or not hasattr(array, "detections"):
        raise RuntimeError("Detection3DArray 缺少 header/detections")
    if not all(hasattr(detection, name) for name in ("results", "id", "bbox")):
        raise RuntimeError("Detection3D 缺少 results/id/bbox")
    if not hasattr(detection.bbox, "size") or not all(
        hasattr(detection.bbox.size, axis) for axis in ("x", "y", "z")
    ):
        raise RuntimeError("Detection3D.bbox 缺少三轴 size")
    if not hasattr(detection.bbox, "center") or not all(
        hasattr(detection.bbox.center, name) for name in ("position", "orientation")
    ):
        raise RuntimeError("Detection3D.bbox 缺少 center pose")
    if not all(
        hasattr(detection.bbox.center.position, axis) for axis in ("x", "y", "z")
    ) or not all(
        hasattr(detection.bbox.center.orientation, axis)
        for axis in ("x", "y", "z", "w")
    ):
        raise RuntimeError("Detection3D.bbox.center 缺少 position/orientation 分量")
    hypothesis = getattr(result, "hypothesis", result)
    if not (hasattr(hypothesis, "class_id") or hasattr(hypothesis, "id")):
        raise RuntimeError("ObjectHypothesisWithPose 缺少 class_id/id")
    if not hasattr(hypothesis, "score") or not hasattr(result, "pose"):
        raise RuntimeError("ObjectHypothesisWithPose 缺少 score/pose")


def _load_config() -> dict[str, Any]:
    path = _resolve_config_path()
    try:
        yaml = importlib.import_module("yaml")
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML，无法读取 config.yaml；请安装 python3-yaml") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 配置错误统一包含路径
        raise RuntimeError(f"读取配置失败 {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"配置顶层必须是映射：{path}")
    try:
        validate_controller_config(data)
    except ValueError as exc:
        raise RuntimeError(f"配置与MMK2 Controller Manifest V1不一致 {path}: {exc}") from exc
    arm_planning = data.get("arm_planning")
    if not isinstance(arm_planning, Mapping):
        raise RuntimeError(f"机械臂规划配置无效 {path}: arm_planning必须是Mapping")
    arm_planning_enabled = arm_planning.get("enabled")
    if type(arm_planning_enabled) is not bool:
        raise RuntimeError(f"机械臂规划配置无效 {path}: enabled必须是严格bool")
    if arm_planning_enabled:
        try:
            _arm_planning_config_from_config(data)
        except ValueError as exc:
            raise RuntimeError(f"机械臂规划配置无效 {path}: {exc}") from exc
    _validated_action_dispatch_topic(data.get("topics"))
    return data


def _arm_planning_config_from_config(
    config: Mapping[str, Any],
) -> tuple[bool, ArmPlanningConfig]:
    """严格读取默认关闭的机械臂规划配置，但不构造规划器或官方依赖。"""

    if not isinstance(config, Mapping):
        raise ValueError("config 必须是 Mapping")
    section = config.get("arm_planning")
    if not isinstance(section, Mapping):
        raise ValueError("config['arm_planning'] 必须是 Mapping")
    enabled = section.get("enabled")
    if type(enabled) is not bool:
        raise ValueError("arm_planning.enabled 必须是严格 bool")

    config_fields = tuple(field.name for field in fields(ArmPlanningConfig))
    allowed = {"enabled", *config_fields}
    unknown = tuple(sorted(str(key) for key in section.keys() if key not in allowed))
    if unknown:
        raise ValueError(f"arm_planning 包含未知字段：{unknown}")
    missing = tuple(name for name in config_fields if name not in section)
    if missing:
        raise ValueError(f"arm_planning 缺少显式字段：{missing}")
    values = {name: section[name] for name in config_fields}
    return enabled, ArmPlanningConfig(**values)


def _navigation_config_from_config(config: Mapping[str, Any]) -> NavigationConfig:
    """只接受完整显式navigation字段；仓库默认缺失时调用方保持Optional不可用。"""

    if not isinstance(config, Mapping):
        raise ValueError("config必须是Mapping")
    section = config.get("navigation")
    if not isinstance(section, Mapping):
        raise ValueError("config.navigation缺失，不能采用NavigationConfig隐藏默认值")
    config_fields = tuple(item.name for item in fields(NavigationConfig))
    expected = set(config_fields)
    actual = set(section)
    if actual != expected:
        raise ValueError(
            "navigation字段不完整："
            f"缺失={sorted(expected - actual)}，未知={sorted(actual - expected)}"
        )
    return NavigationConfig(**{name: section[name] for name in config_fields})


def _arm_execution_config_from_config(
    config: Mapping[str, Any],
) -> ArmExecutionConfig:
    """严格构造无隐藏默认值的ArmExecutionConfig；null或缺字段均拒绝。"""

    if not isinstance(config, Mapping):
        raise ValueError("config必须是Mapping")
    section = config.get("arm_execution")
    if not isinstance(section, Mapping):
        raise ValueError("config.arm_execution必须是Mapping")
    config_fields = tuple(item.name for item in fields(ArmExecutionConfig))
    expected = set(config_fields)
    actual = set(section)
    if actual != expected:
        raise ValueError(
            "arm_execution字段不完整："
            f"缺失={sorted(expected - actual)}，未知={sorted(actual - expected)}"
        )
    missing_values = tuple(name for name in config_fields if section[name] is None)
    if missing_values:
        raise ValueError(
            f"ArmExecutionConfig仍有未标定null字段：{missing_values}"
        )
    return ArmExecutionConfig(**{name: section[name] for name in config_fields})


def _resolve_config_path() -> Path:
    candidates: list[Path] = []
    env_path = os.getenv("TEAM_SORTING_CONFIG", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    try:
        packages = importlib.import_module("ament_index_python.packages")
        share = Path(packages.get_package_share_directory("team_sorting"))
        candidates.append(share / "config" / "config.yaml")
    except Exception:
        pass
    candidates.append(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"找不到 config.yaml，搜索={candidates}；请设置 TEAM_SORTING_CONFIG 或正确安装包"
    )


def _spin(
    ros: SimpleNamespace,
    node_class: type,
    config: dict[str, Any],
    args: Optional[list[str]],
) -> None:
    ros.rclpy.init(args=args)
    node = None
    try:
        node = node_class(config, ros)
        ros.rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            # SIGINT handler 可能已经关闭 context；重复 shutdown 会抛 RCLError。节点
            # 必须先完成自身清理，而 context 仍有效时才由这里执行最终 shutdown。
            if _rclpy_context_ok(ros, node):
                ros.rclpy.shutdown()
#_TeamClientNode：比赛控制主节点

# 这是三个节点里最核心的一个。整个规则系统每个控制周期的数据组装与动作出口。

# 它订阅：

# /material/instruction比赛任务指令
#解析当前要搬运什么、放到哪里
# Odom底盘里程计
#导航与安全判断
# JointState实际关节状态
#机械臂控制、安全保持和执行反馈
# /team/object_estimates
#物体三维估计结果
#提供目标物体的位置和类别
# 然后按照固定控制频率执行_control_tick()。

# 完整过程是：

# 读取最新Odom
# +
# 读取最新JointState
# +
# 读取新鲜三维目标
# +
# 读取当前任务
#         ↓
# 构造SensorSnapshot
#         ↓
# 交给_compute_candidate_commands()
#         ↓
# 得到BaseCommand和ManipulationCommand
#         ↓
# 交给ActionMux
#         ↓
# 生成FinalAction[19]
#         ↓
# OfficialCommandPublisher发布

def _create_team_client_node(ros: SimpleNamespace) -> type:
    class _TeamClientNode(ros.Node):
        def __init__(self, config: dict[str, Any], ros_deps: SimpleNamespace) -> None:
            super().__init__("team_client_node")
            self._ros = ros_deps
            self._config = config
            topics = config["topics"]
            action_dispatch_topic = _validated_action_dispatch_topic(topics)
            timing = config["timing"]
            self._state_max_delta_ns = int(float(timing["state_max_delta_s"]) * 1e9)
            self._command_ttl_ns = int(float(timing["command_ttl_s"]) * 1e9)
            self._base_cache = TimestampedCache()
            self._joint_cache = TimestampedCache()
            self._latest_estimates: tuple[ObjectEstimate3D, ...] = ()
            self._parsed_tasks: tuple[TaskSpec, ...] = ()
            self._instruction_raw = ""
            self._coordinator = CompetitionRunCoordinator()
            self._active_context: Optional[CompetitionContext] = None
            self._last_activated_context: Optional[CompetitionContext] = None
            self._local_io_ready = False
            self._system_ready_submitted = False
            self._last_control_warning = ""
            self._last_non_ros_warning = ""
            self._destroy_failure_reason = ""
            self._input_issues: dict[str, str] = {}
            self._destroying = False
            self._control_timer: Optional[Any] = None
            self._parser = InstructionParser()
            self._fsm = GlobalFSM(max_pick_retries=int(config["fsm"]["max_pick_retries"]))
            self._mux = ActionMux(_action_mux_config(config))
            self._runtime_wiring = _RuntimeWiringState()

            # config.yaml 当前没有视觉验证器所需的完整正式阈值。保持 Optional
            # 不可用，避免采用算法构造器参数名背后的任何隐式或猜测默认值。
            self._visual_observation_verifier: Optional[
                VisualObservationVerifier
            ] = None
            self._visual_observation_verifier_unavailable_reason = (
                "config缺少VisualObservationVerifier全部正式严格阈值"
            )

            # 业务对象只在节点初始化时构造一次。默认配置没有正式navigation字段，
            # 因而不能偷偷采用NavigationConfig的模块默认值并宣称导航可用。
            self._navigation: Optional[NavigationController] = None
            self._navigation_unavailable_reason = ""
            try:
                navigation_config = _navigation_config_from_config(config)
                self._navigation = NavigationController(navigation_config)
            except (TypeError, ValueError) as exc:
                self._navigation_unavailable_reason = str(exc)

            self._kdl_adapter: Optional[OfficialKDLAdapter] = None
            self._arm_planning_config: Optional[ArmPlanningConfig] = None
            self._arm_planner: Optional[ArmPlanner] = None
            self._arm_planning_unavailable_reason = ""
            try:
                arm_planning_enabled, arm_planning_config = (
                    _arm_planning_config_from_config(config)
                )
                self._arm_planning_config = arm_planning_config
            except (TypeError, ValueError) as exc:
                arm_planning_enabled = False
                if not self._arm_planning_unavailable_reason:
                    self._arm_planning_unavailable_reason = (
                        f"ArmPlanningConfig不可用：{exc}"
                    )
            if not arm_planning_enabled:
                if not self._arm_planning_unavailable_reason:
                    self._arm_planning_unavailable_reason = "arm_planning disabled"
            elif self._arm_planning_config is None:
                if not self._arm_planning_unavailable_reason:
                    self._arm_planning_unavailable_reason = "机械臂规划依赖不完整"
            else:
                try:
                    # 只有enabled=True且抓放两类正式配置都完整时才允许导入官方KDL。
                    self._arm_planning_config.validate_for_grasp()
                    self._arm_planning_config.validate_for_place()
                    official = config.get("official", {})
                    if not isinstance(official, Mapping):
                        raise ValueError("config.official必须是Mapping")
                    self._kdl_adapter = OfficialKDLAdapter(
                        official_root=str(official.get("root", "")),
                        module_name=str(official.get("kdl_module", "mmk2_kdl")),
                    )
                    self._kdl_adapter.self_check()
                    self._arm_planner = ArmPlanner(
                        self._kdl_adapter, self._arm_planning_config
                    )
                except Exception as exc:  # noqa: BLE001 - 可选模块失败必须收窄并关闭
                    self._arm_planner = None
                    self._arm_planning_unavailable_reason = (
                        f"机械臂规划失败关闭：{exc}"
                    )

            self._arm_execution: Optional[ArmExecutionController] = None
            self._arm_execution_unavailable_reason = ""
            try:
                arm_execution_config = _arm_execution_config_from_config(config)
                self._arm_execution = ArmExecutionController(arm_execution_config)
            except (TypeError, ValueError) as exc:
                self._arm_execution_unavailable_reason = (
                    f"ArmExecutionConfig不可用，执行失败关闭：{exc}"
                )
            self._mapper = JointStateMapper(config.get("joint_aliases", {}))
            try:
                self._external_candidate_config = ExternalCandidateConfig.from_mapping(
                    config.get("external_candidate", {})
                )
            except ValueError as exc:
                raise RuntimeError(f"external_candidate 配置无效：{exc}") from exc
            self._external_candidate = ExternalCandidateConsumer(
                self._external_candidate_config
            )
            self._external_candidate_subscription: Optional[Any] = None
            self._last_control_tick_ns: Optional[int] = None
            control = _validated_control_config(config)
            self._observe_only = control["observe_only"]
            self._enable_official_publish = control["enable_official_publish"]
            self._simulation_only = control["simulation_only"]
            self._official_topics = dict(topics["official_commands"])
            self._publish_enabled = _official_publish_enabled(control)
            self._official_publisher: Optional[OfficialCommandPublisher] = None
            if self._publish_enabled:
                head_tracking = {
                    **control["head_target_tracking"],
                    "yaw_lower": self._mux.config.joint_lower[1],
                    "yaw_upper": self._mux.config.joint_upper[1],
                    "pitch_lower": self._mux.config.joint_lower[2],
                    "pitch_upper": self._mux.config.joint_upper[2],
                }
                self._official_publisher = OfficialCommandPublisher(
                    self,
                    topics["official_commands"],
                    ros_deps,
                    head_tracking,
                )
            else:
                reason = (
                    "observe_only"
                    if self._observe_only
                    else "enable_official_publish=false"
                )
                self.get_logger().info(
                    f"official_publish_disabled:{reason};仅发布团队诊断遥测"
                )
            self._action_pub = self.create_publisher(ros.String, topics["final_action"], 10)
            self._dispatch_pub = self.create_publisher(
                ros.String, action_dispatch_topic, 10
            )
            self._fsm_pub = self.create_publisher(ros.String, topics["fsm_status"], 10)
            self._context_pub = self.create_publisher(
                ros.String, topics["competition_context"], 10
            )
            self.create_subscription(ros.String, topics["instruction"], self._on_instruction, 10)
            self.create_subscription(
                ros.String, topics["referee_taskinfo"], self._on_referee_taskinfo, 10
            )
            self.create_subscription(
                ros.String, topics["referee_gameinfo"], self._on_referee_gameinfo, 10
            )
            self.create_subscription(
                getattr(ros, "Int32", ros.String),
                topics["referee_score"], self._on_referee_score, 10
            )
            if _external_candidate_subscription_enabled(self._external_candidate_config):
                self._external_candidate_subscription = self.create_subscription(
                    ros.String,
                    self._external_candidate_config.topic,
                    self._on_external_candidate,
                    10,
                )
            self.create_subscription(ros.Odometry, topics["odom"], self._on_odom, 30)
            self.create_subscription(ros.JointState, topics["joint_states"], self._on_joints, 30)
            self.create_subscription(
                ros.Detection3DArray, topics["object_estimates"], self._on_estimates, 10
            )
            period = 1.0 / float(timing["control_rate_hz"])
            self._control_timer = self.create_timer(period, self._control_tick)
            self._local_io_ready = True

            # 机械臂规划接线后由 arm_planning 业务链持有并按需自检 KDL；ROS 组装层
            # 不再为当前尚未使用的规划能力设置启动硬依赖。

        def _on_instruction(self, message: Any) -> None:
            now_ns = self.get_clock().now().nanoseconds
            try:
                tasks = self._parser.parse(message.data, now_ns)
                self._parsed_tasks = tasks
                self._instruction_raw = message.data
                changed = self._coordinator.update_tasks(tasks, now_ns)
                if changed:
                    self.get_logger().info("收到新的完整三任务集合，已创建新的本地run身份")
                self._refresh_competition_context(
                    now_ns, refresh_instruction_liveness=True
                )
            except ValueError as exc:
                self.get_logger().error(f"任务解析失败：{exc}")

        def _on_referee_taskinfo(self, message: Any) -> None:
            now_ns = self.get_clock().now().nanoseconds
            try:
                self._coordinator.referee.update_taskinfo(message.data, now_ns)
            except (AttributeError, TypeError, ValueError) as exc:
                self.get_logger().error(f"裁判taskinfo解析失败：{exc}")
            self._refresh_competition_context(now_ns)

        def _on_referee_gameinfo(self, message: Any) -> None:
            now_ns = self.get_clock().now().nanoseconds
            try:
                self._coordinator.referee.update_gameinfo(message.data, now_ns)
            except (AttributeError, TypeError, ValueError) as exc:
                self.get_logger().error(f"裁判gameinfo解析失败：{exc}")
            self._refresh_competition_context(now_ns)

        def _on_referee_score(self, message: Any) -> None:
            now_ns = self.get_clock().now().nanoseconds
            try:
                self._coordinator.referee.update_score(message.data, now_ns)
            except (AttributeError, TypeError, ValueError) as exc:
                self.get_logger().error(f"裁判score解析失败：{exc}")
            self._refresh_competition_context(now_ns)

        def _refresh_competition_context(
            self, now_ns: int, *, refresh_instruction_liveness: bool = False
        ) -> None:
            context = self._coordinator.context()
            message = self._ros.String()
            message.data = context.to_json()
            self._context_pub.publish(message)
            self._active_context = context
            self._synchronize_runtime_context(context)
            if not context.valid or context.finished or context.active_task is None:
                return
            active_instruction = json.dumps(
                context.to_dict()["active_task"], ensure_ascii=False,
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            if refresh_instruction_liveness:
                self._external_candidate.update_instruction(
                    active_instruction, context.current_task_id, now_ns
                )
            if not self._system_ready_submitted:
                return
            if not self._coordinator.active_task_changed(context):
                return
            previous = self._last_activated_context
            same_run_and_task = (
                previous is not None
                and previous.run_id == context.run_id
                and previous.current_task_id == context.current_task_id
            )
            if same_run_and_task:
                self.get_logger().info(
                    "attempt_transition: 官方attempt已结算，当前进入同一任务的下一次"
                    f"局内尝试：{previous.current_attempt_count} -> "
                    f"{context.current_attempt_count}。此操作只重建本地单任务FSM，"
                    "不代表Server、机器人或物品复位。"
                )
            else:
                old_id = None if previous is None else previous.current_task_id
                self.get_logger().info(
                    f"task_transition: 官方当前任务切换：{old_id} -> "
                    f"{context.current_task_id}；仅重建本地单任务FSM，"
                    "不代表Server、机器人或物品复位。"
                )
            # GlobalFSM is deliberately a one-task machine. Rebuilding it on
            # an official task/attempt transition is a local software reset
            # only; it never requests or claims a server, robot, or object reset.
            self._fsm = GlobalFSM(max_pick_retries=int(self._config["fsm"]["max_pick_retries"]))
            self._fsm.handle_event(FSMEvent.SYSTEM_READY, now_ns)
            self._fsm.submit_task(context.active_task, now_ns)
            self._last_activated_context = context
            if not refresh_instruction_liveness:
                self._external_candidate.update_instruction(
                    active_instruction, context.current_task_id, now_ns
                )

        def _on_external_candidate(self, message: Any) -> None:
            """Validate and reserve one candidate; never creates or publishes an action."""

            now_ns = self.get_clock().now().nanoseconds
            try:
                decision = self._external_candidate.receive(message.data, now_ns)
            except Exception as exc:  # noqa: BLE001 - callback must fail closed
                self.get_logger().error(
                    f"External Candidate回调内部失败，已拒绝：{type(exc).__name__}: {exc}"
                )
                return
            self._log_external_candidate_decision(decision)

        @staticmethod
        def _task_semantic_fingerprint(task: TaskSpec) -> tuple[Any, ...]:
            """提取任务业务语义；接收时间不同不代表任务内容发生变化。"""

            return (
                task.task_id,
                task.instruction,
                task.target_kind,
                task.target_body,
                task.target_color,
                task.place_type,
                task.place_world_xyz,
                task.place_radius,
                task.ref_prop,
                task.ref_prop_body,
                task.direction,
                task.valid,
                task.failure_reason,
            )

        def _reset_runtime_wiring(
            self,
            *,
            run_id: object = _RUNTIME_ID_UNSET,
            task_id: object = _RUNTIME_ID_UNSET,
            attempt: object = _RUNTIME_ID_UNSET,
            context_finished: object = _RUNTIME_ID_UNSET,
            reason: str = "",
        ) -> None:
            """清除接线缓存；不会把软件reset描述成真实机器人已经停止。"""

            state = self._runtime_wiring
            reset_failure = ""
            if self._arm_execution is not None:
                try:
                    # ArmExecution.reset只清除软件执行状态，不等于真实机器人已经停止；
                    # 真实安全动作仍必须由ActionMux统一仲裁。
                    self._arm_execution.reset()
                except Exception as exc:  # noqa: BLE001 - 清理其余状态仍必须继续
                    reset_failure = f"ArmExecutionController.reset失败：{exc}"
            if run_id is not _RUNTIME_ID_UNSET:
                state.run_id = "" if run_id is None else str(run_id)
            if task_id is not _RUNTIME_ID_UNSET:
                state.task_id = task_id if type(task_id) is int else None
            if attempt is not _RUNTIME_ID_UNSET:
                state.attempt = attempt if type(attempt) is int else None
            if context_finished is not _RUNTIME_ID_UNSET:
                state.context_finished = bool(context_finished)
            state.last_handled_phase = None
            state.phase_generation += 1
            state.active_target = None
            state.preferred_object_id = ""
            state.active_nav_goal = None
            state.planned_grasp_context = None
            state.confirmed_grasp_context = None
            state.active_pick_trajectory = None
            state.active_place_trajectory = None
            state.active_trajectory_id = ""
            state.pick_observation_before_lift = None
            state.latest_grasp_verification = None
            state.place_observation_before_release = None
            state.latest_place_verification_observation = None
            state.last_navigation_status = None
            state.last_manipulation_status = None
            state.phase_event_keys.clear()
            state.phase_entry_failure_reason = reset_failure
            state.last_event_tick_ns = None
            state.event_attempted_this_tick = False
            if reset_failure and self._context_ok():
                self.get_logger().error(
                    f"runtime_wiring_reset_failed:{reason}:{reset_failure}"
                )

        def _prepare_phase_transition(
            self,
            previous_phase: Optional[GlobalPhase],
            current_phase: GlobalPhase,
        ) -> None:
            """只清理新阶段已明确失效的业务缓存。"""

            state = self._runtime_wiring
            state.phase_generation += 1
            state.phase_entry_failure_reason = ""
            # 去重键只在产生它的阶段内有效；新阶段必须重新武装，但SAFE_HOLD仍保留
            # 业务载荷以便恢复被暂停阶段。
            state.phase_event_keys.clear()
            if current_phase is GlobalPhase.SAFE_HOLD or (
                previous_phase is GlobalPhase.SAFE_HOLD
                and current_phase not in {GlobalPhase.DONE, GlobalPhase.FAILED}
            ):
                # 安全暂停及恢复不能破坏被暂停阶段的业务生命周期。
                return
            if current_phase is GlobalPhase.SEARCH_TARGET:
                state.active_target = None
                state.preferred_object_id = ""
                state.active_nav_goal = None
                state.planned_grasp_context = None
                state.confirmed_grasp_context = None
                state.active_pick_trajectory = None
                state.active_place_trajectory = None
                state.active_trajectory_id = ""
                state.pick_observation_before_lift = None
                state.latest_grasp_verification = None
                state.place_observation_before_release = None
                state.latest_place_verification_observation = None
                state.last_navigation_status = None
                state.last_manipulation_status = None
            elif current_phase is GlobalPhase.NAV_TO_PICK:
                state.active_nav_goal = None
                state.last_navigation_status = None
            elif current_phase is GlobalPhase.REFINE_TARGET:
                state.preferred_object_id = (
                    state.active_target.object_id.strip()
                    if isinstance(state.active_target, ObjectEstimate3D)
                    and isinstance(state.active_target.object_id, str)
                    else ""
                )
                # SEARCH_TARGET 的完整观测只能提供跨阶段身份，不能作为近距离精定位。
                state.active_target = None
                state.active_nav_goal = None
                state.active_pick_trajectory = None
                state.active_trajectory_id = ""
                state.planned_grasp_context = None
                state.pick_observation_before_lift = None
                state.latest_grasp_verification = None
                state.last_navigation_status = None
            elif current_phase is GlobalPhase.PLAN_PICK:
                state.active_pick_trajectory = None
                state.active_trajectory_id = ""
                state.planned_grasp_context = None
                state.pick_observation_before_lift = None
                state.latest_grasp_verification = None
            elif current_phase is GlobalPhase.EXECUTE_PICK:
                state.last_manipulation_status = None
            elif current_phase is GlobalPhase.VERIFY_PICK:
                state.latest_grasp_verification = None
            elif current_phase is GlobalPhase.NAV_TO_PLACE:
                state.active_target = None
                state.preferred_object_id = ""
                state.active_nav_goal = None
                state.planned_grasp_context = None
                state.active_pick_trajectory = None
                state.active_place_trajectory = None
                state.active_trajectory_id = ""
                state.pick_observation_before_lift = None
                state.latest_grasp_verification = None
                state.place_observation_before_release = None
                state.latest_place_verification_observation = None
                state.last_navigation_status = None
                state.last_manipulation_status = None
            elif current_phase is GlobalPhase.PLAN_PLACE:
                state.active_place_trajectory = None
                state.active_trajectory_id = ""
                state.place_observation_before_release = None
                state.latest_place_verification_observation = None
            elif current_phase is GlobalPhase.EXECUTE_PLACE:
                state.last_manipulation_status = None
            elif current_phase is GlobalPhase.VERIFY_PLACE:
                state.latest_place_verification_observation = None
            elif current_phase is GlobalPhase.RETURN_END:
                state.active_target = None
                state.preferred_object_id = ""
                state.active_nav_goal = None
                state.planned_grasp_context = None
                state.confirmed_grasp_context = None
                state.active_pick_trajectory = None
                state.active_place_trajectory = None
                state.active_trajectory_id = ""
                state.pick_observation_before_lift = None
                state.latest_grasp_verification = None
                state.place_observation_before_release = None
                state.latest_place_verification_observation = None
                state.last_navigation_status = None
                state.last_manipulation_status = None
            elif current_phase in {GlobalPhase.DONE, GlobalPhase.FAILED}:
                state.active_target = None
                state.preferred_object_id = ""
                state.active_nav_goal = None
                state.active_pick_trajectory = None
                state.active_place_trajectory = None
                state.active_trajectory_id = ""
                state.last_navigation_status = None
                state.last_manipulation_status = None

        def _synchronize_runtime_context(
            self, context: Optional[CompetitionContext]
        ) -> None:
            """在run/task/attempt/finished身份变化时一次性丢弃旧业务缓存。"""

            if context is None:
                return
            state = self._runtime_wiring
            desired_run = context.run_id
            if desired_run != state.run_id:
                self._reset_runtime_wiring(
                    run_id=desired_run,
                    task_id=(
                        context.current_task_id
                        if context.valid and not context.finished
                        else None
                    ),
                    attempt=(
                        context.current_attempt_count
                        if context.valid and not context.finished
                        else None
                    ),
                    context_finished=context.finished,
                    reason="new_run",
                )
                return
            if context.finished:
                if not state.context_finished:
                    self._reset_runtime_wiring(
                        task_id=None,
                        attempt=None,
                        context_finished=True,
                        reason="context_finished",
                    )
                return
            if not context.valid or context.active_task is None:
                return
            if (
                state.context_finished
                or state.task_id != context.current_task_id
                or state.attempt != context.current_attempt_count
            ):
                reason = (
                    "new_task"
                    if state.task_id != context.current_task_id
                    else "new_attempt"
                )
                self._reset_runtime_wiring(
                    task_id=context.current_task_id,
                    attempt=context.current_attempt_count,
                    context_finished=False,
                    reason=reason,
                )

        def _handle_phase_entry(
            self,
            snapshot: SensorSnapshot,
            fsm_status: FSMStatus,
            context: Optional[CompetitionContext],
            now_ns: int,
        ) -> bool:
            """只在GlobalPhase真实变化时清理并准备阶段入口。"""

            state = self._runtime_wiring
            phase = fsm_status.global_phase
            if state.last_handled_phase is phase:
                return False
            previous_phase = state.last_handled_phase
            self._prepare_phase_transition(previous_phase, phase)
            state.last_handled_phase = phase
            if type(now_ns) is not int or now_ns < 0:
                state.phase_entry_failure_reason = "阶段入口时间无效"
                return True
            if not isinstance(snapshot, SensorSnapshot):
                state.phase_entry_failure_reason = "阶段入口缺少SensorSnapshot"
                return True
            if context is not None and (
                not isinstance(context, CompetitionContext)
                or not context.valid
                or context.finished
            ):
                state.phase_entry_failure_reason = "CompetitionContext不允许普通业务入口"
                return True
            if phase in {GlobalPhase.NAV_TO_PICK, GlobalPhase.NAV_TO_PLACE}:
                if self._navigation is None:
                    state.phase_entry_failure_reason = (
                        self._navigation_unavailable_reason or "NavigationController不可用"
                    )
                else:
                    state.phase_entry_failure_reason = "导航真实结果接口尚未接线"
            elif phase is GlobalPhase.RETURN_END:
                if self._navigation is None:
                    state.phase_entry_failure_reason = (
                        self._navigation_unavailable_reason or "NavigationController不可用"
                    )
                elif not callable(getattr(self._navigation, "build_return_goal", None)):
                    state.phase_entry_failure_reason = (
                        "NavigationController.build_return_goal不存在，返区失败关闭"
                    )
                else:
                    state.phase_entry_failure_reason = "返区真实结果接口尚未接线"
            elif phase in {GlobalPhase.PLAN_PICK, GlobalPhase.PLAN_PLACE}:
                if self._arm_planner is None:
                    state.phase_entry_failure_reason = (
                        self._arm_planning_unavailable_reason or "ArmPlanner不可用"
                    )
                else:
                    state.phase_entry_failure_reason = "规划真实结果接口尚未接线"
            elif phase in {GlobalPhase.EXECUTE_PICK, GlobalPhase.EXECUTE_PLACE}:
                if self._arm_execution is None:
                    state.phase_entry_failure_reason = (
                        self._arm_execution_unavailable_reason
                        or "ArmExecutionController不可用"
                    )
                elif (
                    phase is GlobalPhase.EXECUTE_PICK
                    and state.active_pick_trajectory is None
                ) or (
                    phase is GlobalPhase.EXECUTE_PLACE
                    and state.active_place_trajectory is None
                ):
                    state.phase_entry_failure_reason = "缺少当前阶段的有效轨迹，执行失败关闭"
                else:
                    state.phase_entry_failure_reason = "执行真实结果接口尚未接线"
            elif phase is GlobalPhase.SEARCH_TARGET:
                state.phase_entry_failure_reason = "等待真实、新鲜且具有稳定身份的odom目标"
            elif phase is GlobalPhase.REFINE_TARGET:
                state.phase_entry_failure_reason = (
                    "等待阶段入口后的同任务odom观测及真实orientation_xyzw/size_xyz_m"
                )
            elif phase in {GlobalPhase.VERIFY_PICK, GlobalPhase.VERIFY_PLACE}:
                state.phase_entry_failure_reason = "抓放验证真实反馈接口尚未接线"
            return True

        def _run_current_phase(
            self,
            snapshot: SensorSnapshot,
            fsm_status: FSMStatus,
            context: Optional[CompetitionContext],
            now_ns: int,
        ) -> tuple[
            Optional[BaseCommand],
            Optional[ManipulationCommand],
            Optional[_RuntimeFSMFeedback],
        ]:
            """运行一个周期的当前阶段逻辑；尚未接线的业务始终失败关闭。"""

            if fsm_status.global_phase in {
                GlobalPhase.SAFE_HOLD,
                GlobalPhase.DONE,
                GlobalPhase.FAILED,
            }:
                return None, None, None
            if context is not None and (
                not context.valid or context.finished or context.active_task is None
            ):
                return None, None, None
            base_command, manipulation_command = self._compute_candidate_commands(
                snapshot, fsm_status, now_ns
            )
            state = self._runtime_wiring
            phase = fsm_status.global_phase
            if phase not in {GlobalPhase.SEARCH_TARGET, GlobalPhase.REFINE_TARGET}:
                return base_command, manipulation_command, None
            if snapshot.task is None or snapshot.task.task_id != fsm_status.task_id:
                state.phase_entry_failure_reason = "视觉目标选择缺少当前FSM任务身份"
                return base_command, manipulation_command, None
            if self._config.get("frames", {}).get("planning") != "odom":
                state.phase_entry_failure_reason = (
                    "frames.planning不是严格odom，视觉业务事件失败关闭"
                )
                return base_command, manipulation_command, None
            if phase is GlobalPhase.REFINE_TARGET and (
                type(self._fsm.phase_entered_ns) is not int
                or self._fsm.phase_entered_ns < 0
            ):
                state.phase_entry_failure_reason = (
                    "REFINE_TARGET缺少有效阶段入口时间，视觉事件失败关闭"
                )
                return base_command, manipulation_command, None

            selected = _select_target_estimate(
                snapshot.object_estimates,
                snapshot.task,
                now_ns,
                self._state_max_delta_ns,
                phase_entered_ns=(
                    self._fsm.phase_entered_ns
                    if phase is GlobalPhase.REFINE_TARGET
                    else None
                ),
                preferred_object_id=(
                    state.preferred_object_id
                    if phase is GlobalPhase.REFINE_TARGET
                    else None
                ),
                require_geometry=phase is GlobalPhase.REFINE_TARGET,
            )
            if selected is None:
                state.phase_entry_failure_reason = (
                    "未收到阶段入口后的完整目标几何，REFINE_TARGET保持等待"
                    if phase is GlobalPhase.REFINE_TARGET
                    else "未收到匹配任务且新鲜、稳定的odom目标"
                )
                return base_command, manipulation_command, None

            state.active_target = selected
            state.phase_entry_failure_reason = ""
            return base_command, manipulation_command, _RuntimeFSMFeedback(
                event=(
                    FSMEvent.TARGET_FOUND
                    if phase is GlobalPhase.SEARCH_TARGET
                    else FSMEvent.TARGET_REFINED
                ),
                source_timestamp_ns=selected.timestamp_ns,
                run_id=state.run_id,
                task_id=state.task_id,
                attempt=state.attempt,
                object_id=selected.object_id or "",
                reason="真实视觉目标观测通过严格任务、frame、时间和身份校验",
                confirmed_by_real_feedback=True,
            )

        def _submit_fsm_event_once(
            self,
            event: FSMEvent,
            fsm_status: FSMStatus,
            now_ns: int,
            *,
            source_timestamp_ns: int,
            run_id: str,
            task_id: Optional[int],
            attempt: Optional[int],
            reason: str = "",
            object_id: str = "",
            goal_id: str = "",
            trajectory_id: str = "",
            confirmed_by_real_feedback: bool = False,
        ) -> bool:
            """校验真实反馈身份，并保证每周期最多向FSM尝试一个业务事件。"""

            if not isinstance(event, FSMEvent):
                raise TypeError("event必须是FSMEvent")
            if not isinstance(fsm_status, FSMStatus):
                raise TypeError("fsm_status必须是FSMStatus")
            if not isinstance(reason, str):
                raise TypeError("reason必须是字符串")
            if type(now_ns) is not int or now_ns < 0:
                raise ValueError("now_ns必须是真正非负整数")
            if type(source_timestamp_ns) is not int or source_timestamp_ns < 0:
                raise ValueError("source_timestamp_ns必须是真正非负整数")
            state = self._runtime_wiring
            if state.last_event_tick_ns != now_ns:
                state.last_event_tick_ns = now_ns
                state.event_attempted_this_tick = False
            if state.event_attempted_this_tick:
                return False
            state.event_attempted_this_tick = True

            def reject(detail: str) -> bool:
                state.phase_entry_failure_reason = detail
                return False

            if source_timestamp_ns > now_ns:
                return reject("事件来源时间晚于当前控制周期")
            entered_ns = self._fsm.phase_entered_ns
            if event is not FSMEvent.RESET and (
                entered_ns is None or source_timestamp_ns < entered_ns
            ):
                return reject("事件来源时间早于当前阶段入口")
            current_status = self._fsm.status(now_ns)
            if (
                fsm_status.global_phase is not current_status.global_phase
                or fsm_status.task_id != current_status.task_id
            ):
                return reject("FSMStatus已过期，阶段或任务身份不一致")
            if (
                not isinstance(run_id, str)
                or run_id != state.run_id
                or (event is not FSMEvent.RESET and not run_id)
                or (
                    task_id is not None and type(task_id) is not int
                )
                or task_id != state.task_id
                or (
                    attempt is not None and type(attempt) is not int
                )
                or attempt != state.attempt
            ):
                return reject("事件run/task/attempt身份不一致")
            if confirmed_by_real_feedback is not True:
                return reject("事件未由真实反馈明确确认")

            supplied_object_id = object_id.strip() if isinstance(object_id, str) else ""
            supplied_goal_id = goal_id.strip() if isinstance(goal_id, str) else ""
            supplied_trajectory_id = (
                trajectory_id.strip() if isinstance(trajectory_id, str) else ""
            )
            expected_object_id = ""
            if isinstance(state.active_target, ObjectEstimate3D):
                expected_object_id = (state.active_target.object_id or "").strip()
            elif isinstance(state.confirmed_grasp_context, GraspContext):
                expected_object_id = state.confirmed_grasp_context.object_id.strip()
            expected_goal_id = (
                state.active_nav_goal.goal_id.strip()
                if isinstance(state.active_nav_goal, NavGoal)
                else ""
            )
            expected_trajectory_id = state.active_trajectory_id.strip()

            object_events = {
                FSMEvent.TARGET_FOUND,
                FSMEvent.TARGET_REFINED,
                FSMEvent.PICK_PLAN_READY,
                FSMEvent.PICK_EXECUTED,
                FSMEvent.PICK_VERIFIED,
                FSMEvent.PICK_FAILED,
                FSMEvent.PLACE_NAV_REACHED,
                FSMEvent.PLACE_PLAN_READY,
                FSMEvent.PLACE_EXECUTED,
                FSMEvent.PLACE_VERIFIED,
            }
            goal_events = {
                FSMEvent.PICK_NAV_REACHED,
                FSMEvent.PLACE_NAV_REACHED,
                FSMEvent.RETURN_REACHED,
            }
            trajectory_events = {
                FSMEvent.PICK_PLAN_READY,
                FSMEvent.PICK_EXECUTED,
                FSMEvent.PLACE_PLAN_READY,
                FSMEvent.PLACE_EXECUTED,
            }
            if event in object_events and (
                not expected_object_id or supplied_object_id != expected_object_id
            ):
                return reject("事件object_id缺失或与当前目标不一致")
            if event in goal_events and (
                not expected_goal_id or supplied_goal_id != expected_goal_id
            ):
                return reject("事件goal_id缺失或与当前NavGoal不一致")
            if event in trajectory_events and (
                not expected_trajectory_id
                or supplied_trajectory_id != expected_trajectory_id
            ):
                return reject("事件trajectory_id缺失或与当前活动轨迹不一致")

            key = (
                state.run_id,
                state.task_id,
                state.attempt,
                fsm_status.global_phase.value,
                state.phase_generation,
                source_timestamp_ns,
                supplied_object_id,
                supplied_goal_id,
                supplied_trajectory_id,
                event.value,
            )
            if key in state.phase_event_keys:
                return reject("同一真实反馈结果已提交")
            if len(state.phase_event_keys) >= _MAX_PHASE_EVENT_KEYS:
                return reject("阶段事件去重表达到有界容量，拒绝新事件")
            state.phase_event_keys.add(key)
            accepted = self._fsm.handle_event(event, source_timestamp_ns, reason)
            if not accepted:
                return reject(f"FSM拒绝事件{event.value}，未发生状态转换")
            if event is FSMEvent.RESET:
                self._reset_runtime_wiring(
                    run_id="",
                    task_id=None,
                    attempt=None,
                    context_finished=False,
                    reason="fsm_reset",
                )
            return True


        def _check_readiness(
            self,
            now_ns: int,
            base: Optional[BaseState],
            joints: Optional[RobotJointState],
        ) -> None:
            """反馈可靠后尝试SYSTEM_READY；成功后不再提交，意外拒绝允许重试。"""

            if self._system_ready_submitted or not self._local_io_ready:
                return
            if base is None or joints is None or not base.valid or not joints.valid:
                return
            # Odom 和 JointState 是当前已冻结的就绪判据；没有新鲜反馈时
            # “节点已构造”不能等同于“系统已准备”。
            if not self._fsm.handle_event(FSMEvent.SYSTEM_READY, now_ns):
                _log_input_issue_on_change(
                    self,
                    self._input_issues,
                    "readiness",
                    "FSM 拒绝 SYSTEM_READY，保持当前状态并在反馈仍可靠时重试",
                )
                return
            self._system_ready_submitted = True
            _log_input_issue_on_change(self, self._input_issues, "readiness", "")
            self._refresh_competition_context(now_ns)

        def _on_odom(self, message: Any) -> None:
            try:
                state = _base_state_from_odom(message)
                self._base_cache.put(state.timestamp_ns, state)
                _log_input_issue_on_change(self, self._input_issues, "odom", "")
            except (AttributeError, TypeError, ValueError) as exc:
                self._base_cache.clear()
                _log_input_issue_on_change(
                    self, self._input_issues, "odom", f"Odom 转换失败：{exc}"
                )

        def _on_joints(self, message: Any) -> None:
            try:
                state = self._mapper.map_message(message)
                self._joint_cache.put(state.timestamp_ns, state)
                _log_input_issue_on_change(self, self._input_issues, "joints", "")
            except (AttributeError, TypeError, ValueError) as exc:
                self._joint_cache.clear()
                _log_input_issue_on_change(
                    self,
                    self._input_issues,
                    "joints",
                    f"JointState 转换失败：{exc}",
                )

        def _on_estimates(self, message: Any) -> None:
            try:
                estimates = _estimates_from_vision(message)
                table = _bounds_from_config(self._config["slot_bounds"]["table"])
                shelf = _bounds_from_config(self._config["slot_bounds"]["shelf"])
                # slot_type 只在 team client 收到米制三维位置后计算，不进入 ROS 消息。
                self._latest_estimates = tuple(
                    ObjectEstimate3D(
                        class_id=item.class_id,
                        position_xyz=item.position_xyz,
                        confidence=item.confidence,
                        frame_id=item.frame_id,
                        timestamp_ns=item.timestamp_ns,
                        slot_type=classify_slot_type(item.position_xyz, table, shelf),
                        valid=item.valid,
                        failure_reason=item.failure_reason,
                        object_id=item.object_id,
                        orientation_xyzw=item.orientation_xyzw,
                        size_xyz_m=item.size_xyz_m,
                    )
                    for item in estimates
                )
                _log_input_issue_on_change(self, self._input_issues, "estimates", "")
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                # 新消息已经证明本轮感知不可用，不能继续把上一次结果当成当前目标。
                self._latest_estimates = ()
                _log_input_issue_on_change(
                    self,
                    self._input_issues,
                    "estimates",
                    f"三维目标消息转换失败：{exc}",
                )

        def _fresh_estimates(self, now_ns: int) -> tuple[ObjectEstimate3D, ...]:
            """只保留当前状态容差内、时间不超前且有效的三维估计。"""

            return tuple(
                estimate
                for estimate in self._latest_estimates
                if estimate.valid
                and 0 <= now_ns - estimate.timestamp_ns <= self._state_max_delta_ns
            )

        def _fresh_control_state(self, cache: TimestampedCache, now_ns: int) -> Optional[Any]:
            """读取控制周期可用的状态，并拒绝时间超前的反馈。"""

            state = cache.nearest(now_ns, self._state_max_delta_ns)
            if state is None or state.timestamp_ns > now_ns:
                return None
            return state

        def _safe_hold_candidates(
            self, joints: RobotJointState, now_ns: int
        ) -> tuple[BaseCommand, ManipulationCommand]:
            """为明确要求主动保持的受控场景生成短TTL候选。

            本函数会把17项 ``controlled_mask`` 全部设为真，属于主动位置控制，不能作为
            节点默认骨架行为；当前Stage 2A普通周期和节点退出都不调用本函数。
            """

            base_command = BaseCommand(
                0.0, 0.0, now_ns, now_ns + self._command_ttl_ns
            )
            if self._arm_execution is None:
                raise RuntimeError(
                    self._arm_execution_unavailable_reason
                    or "ArmExecutionController不可用，不能生成主动保持"
                )
            hold = self._arm_execution.create_hold_command(
                joints, now_ns, self._command_ttl_ns
            )
            return base_command, hold

        def _compute_candidate_commands(
            self,
            snapshot: SensorSnapshot,
            fsm_status: FSMStatus,
            now_ns: int,
        ) -> tuple[BaseCommand | None, ManipulationCommand | None]:
            """把同周期快照交给唯一业务组装入口，返回两类候选命令。

            当前完整业务尚未接线，只产生短TTL零底盘候选，不把“没有机械臂业务候选”
            错误解释成17维主动位置保持。``ActionMux`` 仍基于实际反馈生成诊断
            ``FinalAction``；是否允许发布由独立全局官方发布门决定。
            """

            # 接线前只保留可审计的零底盘候选；不实现业务算法或主动关节保持。
            if fsm_status.global_phase in {
                GlobalPhase.SAFE_HOLD,
                GlobalPhase.DONE,
                GlobalPhase.FAILED,
            }:
                return None, None
            if snapshot.joints is None or not snapshot.joints.valid:
                return None, None
            return BaseCommand(
                0.0, 0.0, now_ns, now_ns + self._command_ttl_ns
            ), None

        def _log_control_warning_on_change(self, reason: str) -> None:
            """只在控制降级原因变化时记录，避免定时器高频刷屏。"""

            if reason == self._last_control_warning:
                return
            self._last_control_warning = reason
            if reason:
                if self._context_ok():
                    self.get_logger().warning(reason)
                else:
                    # ROS context失效后rosout本身不可用；保留普通Python诊断字段，
                    # 不再调用logger或任何publisher。
                    self._last_non_ros_warning = reason

        def _publish_emergency_base_stop(self, reason: str) -> None:
            """状态不足以构造安全19维动作时，尽力只停止底盘。"""

            if self._official_publisher is None:
                self._log_control_warning_on_change(
                    f"{reason}；official_publish_disabled；未创建或调用官方发布器"
                )
                return
            if not self._context_ok():
                self._log_control_warning_on_change(
                    f"{reason}；ROS context已失效，禁止尝试官方紧急发布"
                )
                return
            try:
                self._official_publisher.publish_emergency_base_stop()
                message = reason
            except RuntimeError as exc:
                message = f"{reason}；{exc}"
            self._log_control_warning_on_change(message)

        def _publish_final_action(
            self,
            action: FinalAction,
            *,
            decision: ActionMuxDecision,
            head_publish_authorized: bool = False,
            safety_exit_authorized: bool = False,
            diagnostic_only: bool = False,
        ) -> bool:
            """按本次调用授权发布有效动作，并始终发送诊断 FinalAction。

            普通控制周期只有在同周期确实消费了head_yaw-only External Candidate时才传入
            head授权，且只发布head分组。无候选、候选拒绝、非法mask和已消费旧候选形成
            的FinalAction都只能作为团队遥测。emergency stop和明确安全退出使用各自
            独立的安全路径。
            """

            group_records = tuple(
                _empty_dispatch_group_records(self._official_topics)
            )
            dispatch_mode = DispatchMode.NONE
            dispatch_failure_reason = ""
            if diagnostic_only:
                published = False
                dispatch_failure_reason = "diagnostic_only"
            elif self._official_publisher is None:
                published = False
                dispatch_failure_reason = (
                    "observe_only"
                    if self._observe_only
                    else "enable_official_publish=false"
                )
            elif not self._context_ok():
                self._log_control_warning_on_change(
                    "ROS context已失效，禁止尝试官方控制发布"
                )
                published = False
                dispatch_failure_reason = "ros_context_unavailable"
            elif not action.valid:
                self._publish_emergency_base_stop(
                    f"ActionMux 输出无效，禁止发布19维动作：{action.failure_reason}"
                )
                published = False
                dispatch_failure_reason = "invalid_final_action"
            elif safety_exit_authorized:
                dispatch_mode = DispatchMode.FULL
                try:
                    group_records = self._official_publisher.publish_with_trace(action)
                    published = True
                except OfficialPublishError as exc:
                    group_records = exc.group_records
                    dispatch_failure_reason = str(exc)
                    self._publish_emergency_base_stop(f"官方控制发布失败：{exc}")
                    published = False
            elif not head_publish_authorized:
                published = False
                dispatch_failure_reason = "current_cycle_not_authorized"
            else:
                dispatch_mode = DispatchMode.HEAD_ONLY
                try:
                    group_records = self._official_publisher.publish_head_with_trace(action)
                    published = True
                except OfficialPublishError as exc:
                    group_records = exc.group_records
                    dispatch_failure_reason = str(exc)
                    # head Candidate失败不能顺带授权cmd_vel或其他四组官方话题。
                    self._log_control_warning_on_change(
                        f"官方head控制发布失败：{exc}"
                    )
                    published = False
            # 遥测记录的是 ActionMux 的同一输出，不代表发布成功、Server接收或实际执行。
            action_message = self._ros.String()
            action_message.data = final_action_to_json(action)
            self._action_pub.publish(action_message)
            dispatch = _build_action_dispatch_record(
                action,
                decision,
                publish_enabled=self._publish_enabled,
                publisher_created=self._official_publisher is not None,
                dispatch_mode=dispatch_mode,
                group_records=group_records,
                failure_reason=dispatch_failure_reason,
            )
            dispatch_message = self._ros.String()
            dispatch_message.data = action_dispatch_to_json(dispatch)
            self._dispatch_pub.publish(dispatch_message)
            return published

        def _context_ok(self) -> bool:
            """Fail closed when this node's ROS context is absent or already shut down."""

            context = getattr(self, "context", None)
            ok = getattr(context, "ok", None)
            if not callable(ok):
                return False
            try:
                return bool(ok())
            except Exception:  # noqa: BLE001 - context query failure disables publishing
                return False

        def _official_publish_available(self) -> bool:
            """Return whether an existing publisher can safely be called now."""

            return self._official_publisher is not None and self._context_ok()

        def _control_tick(self) -> None:
            now_ns = self.get_clock().now().nanoseconds
            actual_dt_s = (
                None
                if self._last_control_tick_ns is None
                else (now_ns - self._last_control_tick_ns) / 1e9
            )
            self._last_control_tick_ns = now_ns
            watchdog_decision = self._external_candidate.watchdog(now_ns)
            if watchdog_decision is not None:
                self._log_external_candidate_decision(watchdog_decision)
            base = self._fresh_control_state(self._base_cache, now_ns)
            joints = self._fresh_control_state(self._joint_cache, now_ns)
            self._check_readiness(now_ns, base, joints)
            context = self._active_context
            state_reasons: list[str] = []
            if base is None:
                state_reasons.append("缺少新鲜 Odom")
            elif not base.valid:
                state_reasons.append(base.failure_reason or "Odom 状态无效")
            if joints is None:
                state_reasons.append("缺少新鲜 JointState")
            elif not joints.valid:
                state_reasons.append(joints.failure_reason or "JointState 状态无效")
            snapshot = SensorSnapshot(
                task=self._fsm.task,
                base=base,
                joints=joints,
                object_estimates=self._fresh_estimates(now_ns),
                timestamp_ns=now_ns,
                valid=not state_reasons,
                failure_reason="；".join(state_reasons),
            )
            status_before = self._fsm.status(now_ns)
            self._synchronize_runtime_context(context)
            self._handle_phase_entry(
                snapshot, status_before, context, now_ns
            )
            base_command, manipulation_command, business_event = (
                self._run_current_phase(snapshot, status_before, context, now_ns)
            )
            event_transitioned = False
            if business_event is not None:
                event_transitioned = self._submit_fsm_event_once(
                    business_event.event,
                    status_before,
                    now_ns,
                    source_timestamp_ns=business_event.source_timestamp_ns,
                    run_id=business_event.run_id,
                    task_id=business_event.task_id,
                    attempt=business_event.attempt,
                    reason=business_event.reason,
                    object_id=business_event.object_id,
                    goal_id=business_event.goal_id,
                    trajectory_id=business_event.trajectory_id,
                    confirmed_by_real_feedback=(
                        business_event.confirmed_by_real_feedback
                    ),
                )
            timeout_transitioned = False
            if not event_transitioned:
                timeout_transitioned = self._fsm.check_timeout(now_ns)
            status = self._fsm.status(now_ns)
            if (
                event_transitioned
                or timeout_transitioned
                or status.global_phase is not status_before.global_phase
                or status.global_phase
                in {GlobalPhase.SAFE_HOLD, GlobalPhase.DONE, GlobalPhase.FAILED}
                or (context is not None and context.finished)
            ):
                # 旧阶段候选不能跨越本周期状态转换；新阶段入口留到下个周期执行一次。
                base_command = None
                manipulation_command = None

            fsm_message = self._ros.String()
            fsm_message.data = fsm_status_to_json(status)
            self._fsm_pub.publish(fsm_message)

            if joints is None or not joints.valid:
                # 没有可靠实际关节时绝不能伪造17维全零；此时只能尽力让底盘停车。
                self._publish_emergency_base_stop(
                    snapshot.failure_reason or "JointState 不可靠，无法构造安全保持动作"
                )
                return

            if context is not None and context.finished:
                external_decision = ExternalCandidateDecision(
                    "no_pending_candidate", now_ns, False, "no_pending_candidate"
                )
            else:
                external_decision = self._external_candidate.take(
                    now_ns=now_ns,
                    actual_joints=joints,
                    fsm_status=status,
                    actual_dt_s=actual_dt_s,
                    existing_command=manipulation_command,
                )
            if external_decision.accepted:
                manipulation_command = external_decision.command
            elif external_decision.failure_reason != "no_pending_candidate":
                self._log_external_candidate_decision(external_decision)

            if snapshot.failure_reason:
                # Odom陈旧时不继续普通策略；可靠JointState只用于生成诊断保持动作。
                self._log_control_warning_on_change(
                    f"机器人状态降级，生成零底盘和实际关节保持诊断：{snapshot.failure_reason}"
                )
            else:
                self._log_control_warning_on_change("")
            final_action, mux_decision = self._mux.compose_with_decision(
                base_command,
                manipulation_command,
                joints,
                status,
                now_ns,
                manipulation_source=(
                    "external_candidate"
                    if external_decision.accepted
                    else "manipulation_command"
                ),
            )
            candidate_publish_authorized = (
                external_decision.accepted
                and external_decision.command is not None
                and external_decision.command.controlled_mask
                == (False, True, *([False] * 15))
            )
            if external_decision.accepted and not candidate_publish_authorized:
                self._log_control_warning_on_change(
                    "External Candidate不是严格head_yaw-only mask，禁止官方发布"
                )
            published = self._publish_final_action(
                final_action,
                decision=mux_decision,
                head_publish_authorized=candidate_publish_authorized,
                diagnostic_only=(
                    external_decision.accepted and not candidate_publish_authorized
                ),
            )
            if external_decision.accepted:
                head_publish_failure_reason = ""
                if (
                    candidate_publish_authorized
                    and not published
                    and self._official_publisher is not None
                ):
                    head_publish_failure_reason = (
                        self._official_publisher.last_head_publish_failure_reason
                    )
                self._log_external_candidate_decision(
                    replace(
                        external_decision,
                        event="candidate_action_mux_result",
                        failure_reason=(
                            external_decision.failure_reason
                            or head_publish_failure_reason
                        ),
                        final_action_sequence=final_action.sequence,
                        official_publish_attempted=(
                            candidate_publish_authorized
                            and final_action.valid
                            and self._official_publish_available()
                        ),
                        official_publish_success=published,
                    )
                )

        def _log_external_candidate_decision(
            self, decision: ExternalCandidateDecision
        ) -> None:
            """Emit versioned strict JSON diagnostics without file I/O or control effects."""

            payload = json.dumps(
                decision.audit_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            if decision.accepted:
                self.get_logger().info(f"external_candidate_audit:{payload}")
            else:
                self.get_logger().warning(f"external_candidate_audit:{payload}")

        def destroy_node(self) -> Any:
            """取消控制周期、清空pending并销毁节点，不发送任何官方控制消息。

            Stage 2A从未授权退出时的底盘或关节控制，因此不能把JointState反馈重新当作
            controller target发布。head target shadow也必须原样保留。父级``_spin``在
            ROS实体销毁后负责shutdown；context失效时本路径不调用logger或publisher。
            """

            if self._destroying:
                return None
            self._destroying = True
            context_valid = self._context_ok()
            errors: list[str] = []
            try:
                if self._control_timer is not None:
                    self._control_timer.cancel()
            except Exception as exc:  # noqa: BLE001 - 仍必须继续清空pending并销毁
                errors.append(f"取消control timer失败：{exc}")
            try:
                self._reset_runtime_wiring(
                    run_id="",
                    task_id=None,
                    attempt=None,
                    context_finished=True,
                    reason="destroy_node",
                )
            except Exception as exc:  # noqa: BLE001 - 外部consumer和父节点仍必须清理
                errors.append(f"清理TeamClient运行时接线失败：{exc}")
            try:
                # context失效后不再调用ROS clock；consumer只需一个诊断时间戳。
                now_ns = self.get_clock().now().nanoseconds if context_valid else 0
                self._external_candidate.shutdown(now_ns)
            except Exception as exc:  # noqa: BLE001 - 记录后仍继续父节点销毁
                errors.append(f"清空External Candidate失败：{exc}")
            if errors:
                self._destroy_failure_reason = "；".join(errors)
                if self._context_ok():
                    self.get_logger().error(self._destroy_failure_reason)
            return super().destroy_node()

    return _TeamClientNode

#它负责接收：

# RGB图像
# Depth图像
# CameraInfo内参
# Odom机器人底盘里程计
#提供机器人位置、朝向和速度
# JointState机器人实际关节状态
#提供升降柱、头部、左右机械臂和夹爪状态

# 工作流程：

# RGB + Depth近似同步
#         ↓
# 找到拍摄时刻附近的Odom和JointState
#         ↓
# ROS Image转成RGBFrame和DepthFrame
#         ↓
# 调用perception_2d.py的YOLO适配器
#         ↓
# 得到二维检测
#         ↓
# 调用perception_3d.py计算三维位置
#         ↓
# 发布 /team/object_estimates

# 关键点是：

# 它负责调用视觉算法，但不在本文件里实现视觉算法。
def _create_perception_node(ros: SimpleNamespace) -> type:
    class _PerceptionNode(ros.Node):
        def __init__(self, config: dict[str, Any], ros_deps: SimpleNamespace) -> None:
            super().__init__("perception_node")
            self._ros = ros_deps
            topics = config["topics"]
            timing = config["timing"]
            self._state_max_delta_ns = int(float(timing["state_max_delta_s"]) * 1e9)
            self._depth_unit_scale_m = float(config["perception"]["depth_unit_scale_m"])
            if not math.isfinite(self._depth_unit_scale_m) or self._depth_unit_scale_m <= 0.0:
                raise RuntimeError(
                    "perception.depth_unit_scale_m 必须是正的有限数；"
                    f"实际值={self._depth_unit_scale_m!r}"
                )
            # CameraInfo is static calibration in the official server: its
            # header is empty/zero.  Cache only K and dimensions here; bind it
            # to each synchronized image frame below instead of pretending the
            # calibration message itself is a new sensor measurement.
            self._camera_calibration_cache = StaticCameraCalibrationCache()
            self._last_frame_issue = ""
            self._input_issues: dict[str, str] = {}
            self._base_cache = TimestampedCache()
            self._joint_cache = TimestampedCache()
            self._mapper = JointStateMapper(config.get("joint_aliases", {}))
            self._bridge = ros_deps.CvBridge()

            official = config["official"]
            self._yolo = OfficialYoloAdapter(
                official_root=str(official.get("root", "")),
                checkpoint_path=str(official.get("yolo_checkpoint", "")),
                module_name=str(official.get("yolo_backend_module", "")),
                confidence_threshold=float(config["perception"]["confidence_threshold"]),
            )
            self._transform = CameraTransformProvider(
                official_root=str(official.get("root", "")),
                mjcf_path=str(official.get("mjcf_path", "")),
                module_name=str(official.get("fk_module", "discoverse.robots.mmk2.mmk2_fk")),
                output_frame=str(config["frames"]["planning"]),
            )
            self._yolo.self_check()
            self._transform.self_check()
            self._stabilizer, self._estimator = _perception_pipeline_from_config(
                config, self._transform
            )
            self._publisher = self.create_publisher(
                ros.Detection3DArray, topics["object_estimates"], 10
            )

            self.create_subscription(
                ros.CameraInfo, topics["camera_info"], self._on_camera_info, 10
            )
            self.create_subscription(ros.Odometry, topics["odom"], self._on_odom, 30)
            self.create_subscription(ros.JointState, topics["joint_states"], self._on_joints, 30)
            self._rgb_sub = ros_deps.message_filters.Subscriber(self, ros.Image, topics["rgb"])
            self._depth_sub = ros_deps.message_filters.Subscriber(self, ros.Image, topics["depth"])
            self._sync = ros_deps.message_filters.ApproximateTimeSynchronizer(
                [self._rgb_sub, self._depth_sub],
                queue_size=int(config["perception"]["sync_queue_size"]),
                slop=float(config["perception"]["sync_slop_s"]),
            )
            self._sync.registerCallback(self._on_rgb_depth)

        def _on_camera_info(self, message: Any) -> None:
            try:
                self._camera_calibration_cache.update(message)
                _log_input_issue_on_change(self, self._input_issues, "camera_info", "")
            except (AttributeError, TypeError, ValueError) as exc:
                _log_input_issue_on_change(
                    self,
                    self._input_issues,
                    "camera_info",
                    f"CameraInfo 转换失败，拒绝缓存：{exc}",
                )

        def _on_odom(self, message: Any) -> None:
            try:
                state = _base_state_from_odom(message)
                self._base_cache.put(state.timestamp_ns, state)
                _log_input_issue_on_change(self, self._input_issues, "odom", "")
            except (AttributeError, TypeError, ValueError) as exc:
                self._base_cache.clear()
                _log_input_issue_on_change(
                    self, self._input_issues, "odom", f"Odom 转换失败：{exc}"
                )

        def _on_joints(self, message: Any) -> None:
            try:
                state = self._mapper.map_message(message)
                self._joint_cache.put(state.timestamp_ns, state)
                _log_input_issue_on_change(self, self._input_issues, "joints", "")
            except (AttributeError, TypeError, ValueError) as exc:
                self._joint_cache.clear()
                _log_input_issue_on_change(
                    self,
                    self._input_issues,
                    "joints",
                    f"JointState 转换失败：{exc}",
                )

        def _log_frame_issue_on_change(self, reason: str, *, error: bool = False) -> None:
            """图像流只在问题变化时记录，避免每帧重复同一条日志。"""

            if reason == self._last_frame_issue:
                return
            self._last_frame_issue = reason
            if not reason:
                return
            if error:
                self.get_logger().error(reason)
            else:
                self.get_logger().warning(reason)

        def _on_rgb_depth(self, rgb_message: Any, depth_message: Any) -> None:
            try:
                timestamp_ns = _stamp_to_ns(rgb_message.header.stamp)
                base = self._base_cache.nearest(timestamp_ns, self._state_max_delta_ns)
                joints = self._joint_cache.nearest(timestamp_ns, self._state_max_delta_ns)
                missing: list[str] = []
                if self._camera_calibration_cache.value is None:
                    missing.append("CameraInfo")
                if base is None or not base.valid:
                    missing.append("邻近有效Odom")
                if joints is None or not joints.valid:
                    missing.append("邻近有效JointState")
                if missing:
                    self._log_frame_issue_on_change(
                        f"同步图像缺少 {'/'.join(missing)}，跳过该帧"
                    )
                    return
                rgb = RGBFrame(
                    image=self._bridge.imgmsg_to_cv2(rgb_message, desired_encoding="bgr8"),
                    encoding="bgr8",
                    frame_id=normalize_ros_frame_id(rgb_message.header.frame_id),
                    timestamp_ns=timestamp_ns,
                )
                depth = DepthFrame(
                    image=self._bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough"),
                    unit_scale_m=self._depth_unit_scale_m,
                    frame_id=normalize_ros_frame_id(depth_message.header.frame_id),
                    timestamp_ns=_stamp_to_ns(depth_message.header.stamp),
                )
                assert self._camera_calibration_cache.value is not None
                intrinsics = bind_static_camera_calibration(
                    self._camera_calibration_cache.value, rgb_message, depth_message
                )
                detections = self._stabilizer.update(
                    self._yolo.detect(rgb, depth, intrinsics),
                    frame_timestamp_ns=rgb.timestamp_ns,
                    frame_id=rgb.frame_id,
                )
                estimates = self._estimator.estimate(
                    detections, depth, intrinsics, base, joints
                )
                self._publisher.publish(
                    _estimates_to_vision(estimates, self._ros, rgb_message.header.stamp)
                )
                self._log_frame_issue_on_change("")
            except NotImplementedError as exc:
                self._log_frame_issue_on_change(str(exc), error=True)
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                self._log_frame_issue_on_change(f"感知帧处理失败：{exc}", error=True)

    return _PerceptionNode
# 它订阅：

# FinalAction
# FSMStatus
# 任务指令
# 裁判taskinfo
# 裁判gameinfo
# 裁判score

# 同时启动：

# ros2 bag record

# 保存高带宽数据，例如：

# RGB；
# Depth；
# Odom；
# JointState；
# 三维目标；
# 最终动作。

# 它的流程是：

# 节点启动
# → 创建Episode
# → 启动rosbag
# → 记录任务、FSM、动作和裁判元数据
# → 监控rosbag进程
# → 节点退出时停止rosbag
# → 完成Episode元数据

# 它解决的是：

# 以后训练VLA需要完整、时间对齐、可复现的专家轨迹。

def _create_recorder_node(ros: SimpleNamespace) -> type:
    class _DatasetRecorderNode(ros.Node):
        def __init__(self, config: dict[str, Any], ros_deps: SimpleNamespace) -> None:
            super().__init__("dataset_recorder_node")
            self._ros = ros_deps
            recorder_config = config["recorder"]
            self.declare_parameter("enabled", bool(recorder_config.get("enabled", False)))
            if not bool(self.get_parameter("enabled").value):
                raise RuntimeError(
                    "数据记录未启用；请使用 `ros2 launch team_sorting team.launch.xml "
                    "record_data:=true`，或为 recorder 节点设置 enabled:=true"
                )
            topics = config["topics"]
            self._parser = InstructionParser()
            try:
                pairing_config = ActionPairingConfig.from_mapping(
                    recorder_config.get("action_pairing")
                )
            except ValueError as exc:
                raise RuntimeError(f"Recorder action pairing配置无效：{exc}") from exc
            bag_shutdown = recorder_config["bag_shutdown"]
            control = _validated_control_config(config)
            config_path = _resolve_config_path()
            record_rosbag = recorder_config.get("record_rosbag", True)
            qos_overrides_path: Optional[Path] = None
            if record_rosbag is True:
                rosbag_config = recorder_config.get("rosbag")
                if not isinstance(rosbag_config, Mapping):
                    raise RuntimeError("record_rosbag=true时recorder.rosbag配置必须是映射")
                qos_overrides_path = resolve_rosbag_qos_overrides_path(
                    config_path,
                    rosbag_config.get("qos_overrides_path"),
                )
            runtime_config = RecorderRuntimeConfig(
                root_dir=Path(recorder_config["root_dir"]),
                record_rosbag=record_rosbag,
                rosbag_topics=tuple(
                    str(topic) for topic in recorder_config["rosbag_topics"]
                ),
                recovery_scan_enabled=recorder_config["recovery_scan_enabled"],
                bag_sigint_timeout_sec=bag_shutdown["sigint_timeout_sec"],
                bag_terminate_timeout_sec=bag_shutdown["terminate_timeout_sec"],
                bag_kill_timeout_sec=bag_shutdown["kill_timeout_sec"],
                bag_startup_timeout_sec=bag_shutdown.get(
                    "startup_timeout_sec", 10.0
                ),
                bag_startup_poll_interval_sec=bag_shutdown.get(
                    "startup_poll_interval_sec", 0.02
                ),
                observe_only=control["observe_only"],
                official_publish_enabled=_official_publish_enabled(control),
                rosbag_qos_overrides_path=qos_overrides_path,
                config_path=config_path,
            )
            self._runtime = RecorderRuntimeManager(
                runtime_config,
                pairing_config,
                process_factory=subprocess.Popen,
            )
            self._pairing_timer: Optional[Any] = None
            self._parent_destroyed = False
            self._runtime_closed = False
            now_ns = self.get_clock().now().nanoseconds
            try:
                self._runtime.start(now_ns)
                if runtime_config.record_rosbag:
                    self.create_timer(1.0, self._monitor_rosbag)
                self.create_subscription(ros.String, topics["final_action"], self._on_action, 50)
                if pairing_config.enabled:
                    self.create_subscription(
                        ros.String,
                        topics["action_dispatch"],
                        self._on_action_dispatch,
                        50,
                    )
                    self._pairing_timer = self.create_timer(
                        pairing_config.prune_period_sec,
                        self._prune_action_pairs,
                    )
                self.create_subscription(ros.String, topics["fsm_status"], self._on_fsm, 50)
                self.create_subscription(
                    ros.String, topics["instruction"], self._on_instruction, 10
                )
                self.create_subscription(
                    ros.String,
                    topics["competition_context"],
                    self._on_competition_context,
                    10,
                )
                self.create_subscription(
                    ros.String,
                    topics["referee_taskinfo"],
                    lambda message: self._on_referee_text(topics["referee_taskinfo"], message),
                    10,
                )
                self.create_subscription(
                    ros.String,
                    topics["referee_gameinfo"],
                    lambda message: self._on_referee_text(topics["referee_gameinfo"], message),
                    10,
                )
                self.create_subscription(
                    ros.Int32, topics["referee_score"], self._on_referee_score, 10
                )
            except Exception as exc:  # noqa: BLE001 - 初始化失败必须回滚后原样上抛
                cleanup_errors: list[str] = []
                if self._pairing_timer is not None:
                    self._pairing_timer.cancel()
                    self._pairing_timer = None
                try:
                    self._runtime.close(
                        self.get_clock().now().nanoseconds,
                        reason="node_initialization_failed",
                    )
                except Exception as cleanup_exc:  # noqa: BLE001 - 汇总到初始化错误
                    cleanup_errors.append(f"关闭Recorder runtime失败：{cleanup_exc}")
                details = f"；回滚问题={'；'.join(cleanup_errors)}" if cleanup_errors else ""
                raise RuntimeError(f"Recorder 初始化失败：{exc}{details}") from exc

        def _on_action(self, message: Any) -> None:
            try:
                if self._runtime.action_pairing_enabled:
                    issue_types = self._runtime.record_final_action_payload(
                        message.data,
                        self.get_clock().now().nanoseconds,
                        time.monotonic_ns(),
                    )
                    if issue_types:
                        self.get_logger().warning(
                            f"FinalAction pairing诊断：{','.join(issue_types)}"
                        )
                else:
                    self._runtime.record_final_action(
                        final_action_from_json(message.data)
                    )
            except (ValueError, RuntimeError) as exc:
                self.get_logger().error(f"记录 FinalAction 失败：{exc}")

        def _on_action_dispatch(self, message: Any) -> None:
            try:
                issue_types = self._runtime.record_action_dispatch_payload(
                    message.data,
                    self.get_clock().now().nanoseconds,
                    time.monotonic_ns(),
                )
                if issue_types:
                    self.get_logger().warning(
                        f"ActionDispatch pairing诊断：{','.join(issue_types)}"
                    )
            except RuntimeError as exc:
                self.get_logger().error(f"记录 ActionDispatch 失败：{exc}")

        def _prune_action_pairs(self) -> None:
            try:
                issue_types = self._runtime.prune_action_pairs(
                    self.get_clock().now().nanoseconds,
                    time.monotonic_ns(),
                )
                if issue_types:
                    self.get_logger().warning(
                        f"Action pairing超时清理：{','.join(issue_types)}"
                    )
            except RuntimeError as exc:
                self.get_logger().error(f"Action pairing定时清理失败：{exc}")

        def _stop_pairing_timer(self) -> None:
            if self._pairing_timer is not None:
                self._pairing_timer.cancel()
                self._pairing_timer = None

        def _on_fsm(self, message: Any) -> None:
            try:
                self._runtime.record_fsm_status(fsm_status_from_json(message.data))
            except (ValueError, RuntimeError) as exc:
                self.get_logger().error(f"记录 FSM 失败：{exc}")

        def _on_instruction(self, message: Any) -> None:
            try:
                task = self._runtime.record_instruction(
                    message.data,
                    self.get_clock().now().nanoseconds,
                    self._parser,
                )
                if task is None:
                    reason = self._runtime.metadata.instruction_parse_failure
                    self.get_logger().warning(f"任务原文已保存，但解析失败：{reason}")
            except RuntimeError as exc:
                self.get_logger().error(f"记录任务指令失败：{exc}")

        def _on_referee_text(self, topic: str, message: Any) -> None:
            try:
                self._runtime.record_referee_message(
                    topic, message.data, self.get_clock().now().nanoseconds
                )
            except (ValueError, RuntimeError) as exc:
                self.get_logger().error(f"记录裁判元数据失败：{exc}")

        def _on_competition_context(self, message: Any) -> None:
            try:
                self._runtime.record_competition_context_payload(
                    message.data,
                    self.get_clock().now().nanoseconds,
                    time.monotonic_ns(),
                )
            except (ValueError, RuntimeError) as exc:
                self.get_logger().error(f"记录CompetitionContext失败：{exc}")

        def _on_referee_score(self, message: Any) -> None:
            try:
                self._runtime.record_referee_message(
                    "/referee/score",
                    int(message.data),
                    self.get_clock().now().nanoseconds,
                )
            except (ValueError, RuntimeError) as exc:
                self.get_logger().error(f"记录裁判分数失败：{exc}")

        def _monitor_rosbag(self) -> None:
            try:
                failure = self._runtime.monitor_bag(
                    self.get_clock().now().nanoseconds
                )
                if failure:
                    self.get_logger().error(failure)
            except (ValueError, RuntimeError) as exc:
                self.get_logger().error(f"监控rosbag失败：{exc}")

        def destroy_node(self) -> Any:
            """幂等关闭当前Recorder Segment与rosbag并销毁ROS2节点。

            参数：无。返回父类销毁结果；bag按SIGINT、terminate、kill三级有界停止。
            bag或落盘收尾失败仍调用父类销毁，随后抛出``RuntimeError``。Segment不是
            官方Attempt或Training Episode，关闭也不会改变控制状态或发布开关。
            """

            if self._runtime_closed and self._parent_destroyed:
                return None
            errors: list[str] = []
            if not self._runtime_closed:
                try:
                    self._stop_pairing_timer()
                    self._runtime.close(self.get_clock().now().nanoseconds)
                    self._runtime_closed = True
                except Exception as exc:  # noqa: BLE001 - 允许后续调用重试收尾
                    errors.append(f"关闭Recorder runtime失败：{exc}")
            result = None
            if not self._parent_destroyed:
                # 即使runtime收尾失败，父ROS节点也只销毁一次；标志先置位避免父类
                # 内部异常导致后续调用重复执行非幂等ROS销毁。
                self._parent_destroyed = True
                result = super().destroy_node()
            if errors:
                raise RuntimeError("；".join(errors))
            return result

    return _DatasetRecorderNode


def _action_mux_config(config: dict[str, Any]) -> ActionMuxConfig:
    action = config["action_mux"]
    return ActionMuxConfig(
        max_abs_base_v=float(action["max_abs_base_v"]),
        max_abs_base_w=float(action["max_abs_base_w"]),
        joint_lower=tuple(float(value) for value in action["joint_lower"]),
        joint_upper=tuple(float(value) for value in action["joint_upper"]),
    )


def _validated_action_dispatch_topic(value: object) -> str:
    """冻结团队内部dispatch遥测入口，且在任何官方publisher创建前失败。"""

    if not isinstance(value, Mapping):
        raise RuntimeError("topics配置必须是映射")
    topic = value.get("action_dispatch")
    if topic != "/team/action_dispatch":
        raise RuntimeError("topics.action_dispatch必须严格为/team/action_dispatch")
    return topic


def _validated_control_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the fail-closed global official publish gate."""

    control = config.get("control")
    if not isinstance(control, dict):
        raise RuntimeError("control 配置必须是映射且显式包含三道全局发布门")
    required = ("observe_only", "enable_official_publish", "simulation_only")
    missing = [name for name in required if name not in control]
    if missing:
        raise RuntimeError(f"control 配置缺少字段：{missing}")
    invalid = [name for name in required if type(control[name]) is not bool]
    if invalid:
        raise RuntimeError(f"control 配置字段必须为严格 bool：{invalid}")
    normalized: dict[str, Any] = {name: control[name] for name in required}
    if not normalized["simulation_only"]:
        raise RuntimeError("control.simulation_only=false 被安全策略拒绝")
    normalized["head_target_tracking"] = _validated_head_target_tracking_config(
        control.get("head_target_tracking")
    )
    return normalized


def _validated_head_target_tracking_config(value: object) -> dict[str, Any]:
    """严格验证fresh-reset专用的head controller-target shadow配置。"""

    if not isinstance(value, dict):
        raise RuntimeError("control.head_target_tracking必须是显式映射")
    bool_fields = (
        "enabled",
        "require_fresh_reset",
        "fresh_reset_confirmed",
        "require_exclusive_writer",
    )
    numeric_fields = ("initial_yaw_target", "initial_pitch_target")
    required = (*bool_fields, *numeric_fields)
    missing = [name for name in required if name not in value]
    if missing:
        raise RuntimeError(f"control.head_target_tracking缺少字段：{missing}")
    unknown = sorted(set(value) - set(required))
    if unknown:
        raise RuntimeError(f"control.head_target_tracking存在未知字段：{unknown}")
    invalid_bool = [name for name in bool_fields if type(value[name]) is not bool]
    if invalid_bool:
        raise RuntimeError(f"head target tracking字段必须为严格bool：{invalid_bool}")
    if not value["require_fresh_reset"]:
        raise RuntimeError("head target tracking必须require_fresh_reset=true")
    if not value["require_exclusive_writer"]:
        raise RuntimeError("head target tracking必须require_exclusive_writer=true")
    targets: dict[str, float] = {}
    for name in numeric_fields:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeError(f"{name}必须是有限实数且不能是bool")
        number = float(raw)
        if not math.isfinite(number):
            raise RuntimeError(f"{name}必须是有限实数")
        targets[name] = number
    if targets != {"initial_yaw_target": 0.0, "initial_pitch_target": 0.0}:
        raise RuntimeError("Stage 2A初始head controller target必须严格为[0.0,0.0]")
    return {
        **{name: value[name] for name in bool_fields},
        **targets,
    }


def _official_publish_enabled(control: dict[str, bool]) -> bool:
    """Only the explicit simulation-only actuation combination opens the gate."""

    return (
        not control["observe_only"]
        and control["enable_official_publish"]
        and control["simulation_only"]
    )


def _rclpy_context_ok(ros: SimpleNamespace, node: Optional[Any]) -> bool:
    """Query rclpy context without treating an exception as permission to publish/shutdown."""

    context = getattr(node, "context", None) if node is not None else None
    try:
        return bool(ros.rclpy.ok(context=context))
    except TypeError:
        try:
            return bool(ros.rclpy.ok())
        except Exception:  # noqa: BLE001 - failed query is fail-closed
            return False
    except Exception:  # noqa: BLE001 - failed query is fail-closed
        return False


def _external_candidate_subscription_enabled(config: ExternalCandidateConfig) -> bool:
    """Keep the ROS subscription absent unless the first gate is explicitly enabled."""

    return config.enabled


def _perception_stabilizer_from_config(
    config: dict[str, Any],
) -> Detection2DStabilizer:
    """从真实节点配置构造二维稳定器，并拒绝缺失或未消费字段。"""

    perception = _config_mapping(config.get("perception"), "perception")
    values = _config_mapping(
        perception.get("stabilizer_2d"), "perception.stabilizer_2d"
    )
    required = {
        "iou_match_threshold",
        "min_confirmed_hits",
        "max_missed_frames",
        "bbox_smoothing_alpha",
        "confidence_smoothing_alpha",
    }
    _require_exact_config_keys(values, required, "perception.stabilizer_2d")
    try:
        return Detection2DStabilizer(
            iou_match_threshold=values["iou_match_threshold"],
            min_confirmed_hits=values["min_confirmed_hits"],
            max_missed_frames=values["max_missed_frames"],
            bbox_smoothing_alpha=values["bbox_smoothing_alpha"],
            confidence_smoothing_alpha=values["confidence_smoothing_alpha"],
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"perception.stabilizer_2d 配置无效：{exc}") from exc


def _perception_pipeline_from_config(
    config: dict[str, Any],
    transform: CameraTransformProvider,
) -> tuple[Detection2DStabilizer, Perception3DEstimator]:
    """构造 PerceptionNode 唯一使用的二维稳定与三维估计流水线。"""

    return (
        _perception_stabilizer_from_config(config),
        _perception_3d_estimator_from_config(config, transform),
    )


def _perception_3d_estimator_from_config(
    config: dict[str, Any],
    transform: CameraTransformProvider,
) -> Perception3DEstimator:
    """从 PerceptionNode 的真实配置构造三维估计器。

    中心补偿尺寸按 ``[width_m, height_m, depth_extent_m]`` 解释；局部完整尺寸按
    ``[size_x_m, size_y_m, size_z_m]`` 解释，两者使用独立配置且键都必须严格覆盖
    官方 YOLO 的三类目标。Detection/Depth/CameraInfo 的最大时间差复用 RGB/Depth
    同步的非零 ``sync_slop_s``，避免节点接线中出现两个互相漂移的时间窗口。
    """

    perception = _config_mapping(config.get("perception"), "perception")
    values = _config_mapping(
        perception.get("estimator_3d"), "perception.estimator_3d"
    )
    required = {
        "depth_radius_px",
        "ema_alpha",
        "converge_frames",
        "max_track_age_s",
        "max_position_jump_m",
        "object_dimensions_m",
        "object_local_size_xyz_m",
    }
    _require_exact_config_keys(values, required, "perception.estimator_3d")
    dimensions = _config_mapping(
        values["object_dimensions_m"],
        "perception.estimator_3d.object_dimensions_m",
    )
    expected_classes = set(OfficialYoloAdapter.CLASS_NAMES)
    _require_exact_config_keys(
        dimensions,
        expected_classes,
        "perception.estimator_3d.object_dimensions_m",
    )
    normalized_dimensions: dict[str, tuple[float, float, float]] = {}
    for class_id in OfficialYoloAdapter.CLASS_NAMES:
        raw = dimensions[class_id]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
            raise RuntimeError(
                "perception.estimator_3d.object_dimensions_m"
                f"[{class_id!r}] 必须是三个正有限数"
            )
        if len(raw) != 3:
            raise RuntimeError(
                "perception.estimator_3d.object_dimensions_m"
                f"[{class_id!r}] 必须恰好包含 [width_m,height_m,depth_extent_m]"
            )
        converted = tuple(
            _positive_config_number(
                item,
                "perception.estimator_3d.object_dimensions_m"
                f"[{class_id!r}][{index}]",
            )
            for index, item in enumerate(raw)
        )
        normalized_dimensions[class_id] = converted

    local_sizes = _config_mapping(
        values["object_local_size_xyz_m"],
        "perception.estimator_3d.object_local_size_xyz_m",
    )
    _require_exact_config_keys(
        local_sizes,
        expected_classes,
        "perception.estimator_3d.object_local_size_xyz_m",
    )
    normalized_local_sizes: dict[str, tuple[float, float, float]] = {}
    for class_id in OfficialYoloAdapter.CLASS_NAMES:
        raw = local_sizes[class_id]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
            raise RuntimeError(
                "perception.estimator_3d.object_local_size_xyz_m"
                f"[{class_id!r}] 必须是三个正有限数"
            )
        if len(raw) != 3:
            raise RuntimeError(
                "perception.estimator_3d.object_local_size_xyz_m"
                f"[{class_id!r}] 必须恰好包含 [size_x_m,size_y_m,size_z_m]"
            )
        normalized_local_sizes[class_id] = tuple(
            _positive_config_number(
                item,
                "perception.estimator_3d.object_local_size_xyz_m"
                f"[{class_id!r}][{index}]",
            )
            for index, item in enumerate(raw)
        )

    max_input_skew_s = _positive_config_number(
        perception.get("sync_slop_s"), "perception.sync_slop_s"
    )
    try:
        return Perception3DEstimator(
            transform,
            depth_radius_px=values["depth_radius_px"],
            ema_alpha=values["ema_alpha"],
            converge_frames=values["converge_frames"],
            max_track_age_s=values["max_track_age_s"],
            max_input_skew_s=max_input_skew_s,
            max_position_jump_m=values["max_position_jump_m"],
            object_dimensions_m=normalized_dimensions,
            object_local_size_xyz_m=normalized_local_sizes,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"perception.estimator_3d 配置无效：{exc}") from exc


def _config_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} 必须是映射")
    return value


def _require_exact_config_keys(
    values: dict[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(values)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    raise RuntimeError(
        f"{name} 字段不完整：缺失={missing}，未知={unknown}"
    )


def _positive_config_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} 必须是正有限数，不能使用 bool")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{name} 必须是正有限数，实际={value!r}")
    return number


def _bounds_from_config(values: list[float]) -> Bounds3D:
    if len(values) != 6:
        raise ValueError("slot_bounds 每个区域必须是 [xmin,xmax,ymin,ymax,zmin,zmax]")
    return Bounds3D(*(float(value) for value in values))


def _log_input_issue_on_change(
    node: Any,
    issues: dict[str, str],
    source: str,
    reason: str,
) -> None:
    """同一输入错误只记录一次，恢复后允许下次新问题再次出现。"""

    if issues.get(source, "") == reason:
        return
    issues[source] = reason
    if reason:
        node.get_logger().error(reason)


def _stamp_to_ns(stamp: Any) -> int:
    try:
        if (
            isinstance(stamp.sec, bool)
            or isinstance(stamp.nanosec, bool)
            or not isinstance(stamp.sec, int)
            or not isinstance(stamp.nanosec, int)
        ):
            raise ValueError("sec 和 nanosec 必须是真正整数，不能使用 bool")
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"ROS 时间戳格式无效：{exc}") from exc
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError(f"ROS 时间戳范围无效：sec={sec}, nanosec={nanosec}")
    return sec * 1_000_000_000 + nanosec


def _set_stamp(stamp: Any, timestamp_ns: int) -> None:
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
        raise ValueError("纳秒时间必须是非负整数，不能使用 bool")
    stamp.sec = timestamp_ns // 1_000_000_000
    stamp.nanosec = timestamp_ns % 1_000_000_000


def normalize_ros_frame_id(value: object) -> str:
    """Remove only legacy leading slashes from a ROS frame identifier."""

    if not isinstance(value, str):
        raise ValueError("frame_id必须是字符串")
    normalized = value.lstrip("/")
    if not normalized:
        raise ValueError("frame_id规范化后不能为空")
    return normalized


def static_camera_calibration_from_message(
    message: Any,
) -> tuple[tuple[float, ...], int, int]:
    """Validate and extract only static K/width/height from CameraInfo."""

    k = tuple(float(value) for value in message.k)
    if len(k) != 9:
        raise ValueError(f"CameraInfo.K 必须有9项，实际为{len(k)}项")
    if not all(math.isfinite(value) for value in k):
        raise ValueError("CameraInfo.K 包含 NaN 或 Inf")
    if k[0] <= 0.0 or k[4] <= 0.0:
        raise ValueError("CameraInfo 的 fx 和 fy 必须大于0")
    width, height = int(message.width), int(message.height)
    if width <= 0 or height <= 0:
        raise ValueError("CameraInfo 图像宽高必须大于0")
    return k, width, height


def bind_static_camera_calibration(
    calibration: tuple[tuple[float, ...], int, int],
    rgb_message: Any,
    depth_message: Any,
) -> CameraIntrinsics:
    """Bind static calibration to the current synchronized image context."""

    k, width, height = calibration
    rgb_frame = normalize_ros_frame_id(rgb_message.header.frame_id)
    depth_frame = normalize_ros_frame_id(depth_message.header.frame_id)
    if rgb_frame != depth_frame:
        raise ValueError(f"RGB/Depth frame冲突：{rgb_frame!r} != {depth_frame!r}")
    for label, message in (("RGB", rgb_message), ("Depth", depth_message)):
        if int(message.width) != width or int(message.height) != height:
            raise ValueError(
                f"{label}尺寸与CameraInfo不一致：{message.width}x{message.height} != {width}x{height}"
            )
    return CameraIntrinsics(
        k=k, width=width, height=height, frame_id=rgb_frame,
        timestamp_ns=_stamp_to_ns(rgb_message.header.stamp),
    )


class StaticCameraCalibrationCache:
    """Latch runtime CameraInfo changes invalid until node restart."""

    def __init__(self) -> None:
        self._value: Optional[tuple[tuple[float, ...], int, int]] = None
        self._rejected = False

    @property
    def value(self) -> Optional[tuple[tuple[float, ...], int, int]]:
        return self._value

    def update(self, message: Any) -> tuple[tuple[float, ...], int, int]:
        try:
            calibration = static_camera_calibration_from_message(message)
        except (AttributeError, TypeError, ValueError):
            self._value = None
            self._rejected = True
            raise
        if self._value is not None and calibration != self._value:
            self._value = None
            self._rejected = True
            raise ValueError("CameraInfo K或图像尺寸运行中突变，已清空标定并fail-closed")
        if self._rejected:
            raise ValueError("CameraInfo标定已因运行中突变锁定为无效")
        self._value = calibration
        return calibration


def _base_state_from_odom(message: Any) -> BaseState:
    try:
        pose = message.pose.pose
        twist = message.twist.twist
        position = (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        )
        q = pose.orientation
        quaternion = (float(q.x), float(q.y), float(q.z), float(q.w))
        linear_velocity = (
            float(twist.linear.x),
            float(twist.linear.y),
            float(twist.linear.z),
        )
        angular_velocity = (
            float(twist.angular.x),
            float(twist.angular.y),
            float(twist.angular.z),
        )
        timestamp_ns = _stamp_to_ns(message.header.stamp)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Odom 字段格式无效：{exc}") from exc
    finite_groups = (
        ("位置", position),
        ("四元数", quaternion),
        ("线速度", linear_velocity),
        ("角速度", angular_velocity),
    )
    for label, values in finite_groups:
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Odom {label}包含 NaN/Inf")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError("Odom 四元数范数为零")
    qx, qy, qz, qw = (value / norm for value in quaternion)
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return BaseState(
        position_xyz=position,
        orientation_xyzw=(qx, qy, qz, qw),
        yaw=yaw,
        linear_velocity_xyz=linear_velocity,
        angular_velocity_xyz=angular_velocity,
        frame_id=normalize_ros_frame_id(message.header.frame_id),
        timestamp_ns=timestamp_ns,
    )


def _estimates_to_vision(
    estimates: tuple[ObjectEstimate3D, ...], ros: SimpleNamespace, fallback_stamp: Any
) -> Any:
    message = ros.Detection3DArray()
    if estimates:
        message.header.frame_id = estimates[0].frame_id
        _set_stamp(message.header.stamp, estimates[0].timestamp_ns)
    else:
        _set_stamp(message.header.stamp, _stamp_to_ns(fallback_stamp))
    for estimate in estimates:
        if not estimate.valid:
            continue
        xyz = tuple(float(value) for value in estimate.position_xyz)
        confidence = _validated_confidence(estimate.confidence, "ObjectEstimate3D")
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("ObjectEstimate3D 三维位置包含 NaN/Inf")
        detection = ros.Detection3D()
        if hasattr(detection, "header"):
            detection.header.frame_id = estimate.frame_id
            _set_stamp(detection.header.stamp, estimate.timestamp_ns)
        result = ros.ObjectHypothesisWithPose()
        hypothesis = getattr(result, "hypothesis", result)
        if hasattr(hypothesis, "class_id"):
            hypothesis.class_id = estimate.class_id
        elif hasattr(hypothesis, "id"):
            hypothesis.id = estimate.class_id
        else:
            raise RuntimeError("vision_msgs hypothesis 没有 class_id/id 字段")
        hypothesis.score = confidence
        pose = _vision_result_pose(result)
        pose.position.x, pose.position.y, pose.position.z = xyz
        orientation = estimate.orientation_xyzw
        if orientation is None:
            # 团队内部 /team/object_estimates 契约：零四元数表示姿态不可用。
            orientation = (0.0, 0.0, 0.0, 0.0)
        else:
            orientation = tuple(float(value) for value in orientation)
            if len(orientation) != 4 or not all(
                math.isfinite(value) for value in orientation
            ):
                raise ValueError("ObjectEstimate3D 姿态必须是四项有限数")
            orientation_norm = math.sqrt(
                sum(value * value for value in orientation)
            )
            if orientation_norm <= 1e-12:
                raise ValueError("ObjectEstimate3D 已提供姿态的四元数范数接近零")
            orientation = tuple(value / orientation_norm for value in orientation)
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = orientation
        if not hasattr(detection, "id") or not hasattr(detection, "bbox"):
            raise RuntimeError("vision_msgs Detection3D 缺少 id/bbox 字段")
        detection.id = estimate.object_id or ""
        bbox_center = detection.bbox.center
        (
            bbox_center.position.x,
            bbox_center.position.y,
            bbox_center.position.z,
        ) = xyz
        (
            bbox_center.orientation.x,
            bbox_center.orientation.y,
            bbox_center.orientation.z,
            bbox_center.orientation.w,
        ) = orientation
        if estimate.size_xyz_m is None:
            size = (0.0, 0.0, 0.0)
        else:
            size = tuple(float(value) for value in estimate.size_xyz_m)
            if len(size) != 3 or not all(
                math.isfinite(value) and value > 0.0 for value in size
            ):
                raise ValueError("ObjectEstimate3D 已提供尺寸必须三轴均为有限正数")
        detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z = size
        detection.results.append(result)
        message.detections.append(detection)
    return message


def _estimates_from_vision(message: Any) -> tuple[ObjectEstimate3D, ...]:
    array_timestamp_ns = _stamp_to_ns(message.header.stamp)
    array_frame_id = str(message.header.frame_id)
    converted: list[ObjectEstimate3D] = []
    for detection in message.detections:
        if not detection.results:
            continue
        result = detection.results[0]
        hypothesis = getattr(result, "hypothesis", result)
        class_id = getattr(hypothesis, "class_id", getattr(hypothesis, "id", ""))
        if not class_id:
            raise RuntimeError("vision_msgs 三维结果缺少类别")
        header = detection.header if hasattr(detection, "header") else message.header
        position = _vision_result_position(result)
        xyz = (float(position.x), float(position.y), float(position.z))
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("vision_msgs 三维位置包含 NaN/Inf")
        confidence = _validated_confidence(hypothesis.score, "vision_msgs 三维结果")
        pose = _vision_result_pose(result)
        orientation_values = (
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        if not all(math.isfinite(value) for value in orientation_values):
            raise ValueError("vision_msgs 三维姿态包含 NaN/Inf")
        orientation_norm = math.sqrt(
            sum(value * value for value in orientation_values)
        )
        orientation = None if orientation_norm <= 1e-12 else orientation_values
        if not hasattr(detection, "id") or not hasattr(detection, "bbox"):
            raise RuntimeError("vision_msgs Detection3D 缺少 id/bbox 字段")
        size_values = (
            float(detection.bbox.size.x),
            float(detection.bbox.size.y),
            float(detection.bbox.size.z),
        )
        if not all(math.isfinite(value) for value in size_values):
            raise ValueError("vision_msgs bbox.size 包含 NaN/Inf")
        if all(value == 0.0 for value in size_values):
            size = None
        elif not all(value > 0.0 for value in size_values):
            raise ValueError("vision_msgs bbox.size 必须全零或三轴均为正数")
        else:
            size = size_values
        if size is not None:
            try:
                bbox_center = detection.bbox.center
                bbox_position = (
                    float(bbox_center.position.x),
                    float(bbox_center.position.y),
                    float(bbox_center.position.z),
                )
                bbox_orientation = (
                    float(bbox_center.orientation.x),
                    float(bbox_center.orientation.y),
                    float(bbox_center.orientation.z),
                    float(bbox_center.orientation.w),
                )
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"vision_msgs 非零 bbox.size 缺少合法 center pose：{exc}"
                ) from exc
            if not all(
                math.isfinite(value)
                for value in bbox_position + bbox_orientation
            ):
                raise ValueError("vision_msgs bbox.center 包含 NaN/Inf")
            if not all(
                math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
                for left, right in zip(bbox_position, xyz)
            ):
                raise ValueError(
                    "vision_msgs 非零 bbox.size 的 center.position 与结果中心不一致"
                )
            bbox_orientation_norm = math.sqrt(
                sum(value * value for value in bbox_orientation)
            )
            if orientation is None:
                if bbox_orientation_norm > 1e-12:
                    raise ValueError(
                        "vision_msgs 未知物体姿态必须在结果与bbox.center中同时使用零四元数"
                    )
            else:
                if bbox_orientation_norm <= 1e-12:
                    raise ValueError(
                        "vision_msgs 非零 bbox.size 的bbox.center缺少有效姿态"
                    )
                result_unit = tuple(
                    value / orientation_norm for value in orientation_values
                )
                bbox_unit = tuple(
                    value / bbox_orientation_norm for value in bbox_orientation
                )
                quaternion_error = min(
                    math.dist(result_unit, bbox_unit),
                    math.dist(result_unit, tuple(-value for value in bbox_unit)),
                )
                if quaternion_error > 1e-9:
                    raise ValueError(
                        "vision_msgs 非零 bbox.size 的bbox.center姿态与结果姿态不一致"
                    )
        try:
            timestamp_ns = _stamp_to_ns(header.stamp)
        except (AttributeError, TypeError, ValueError):
            timestamp_ns = array_timestamp_ns
        if timestamp_ns == 0 and array_timestamp_ns != 0:
            timestamp_ns = array_timestamp_ns
        converted.append(
            ObjectEstimate3D(
                class_id=str(class_id),
                position_xyz=xyz,
                confidence=confidence,
                frame_id=str(getattr(header, "frame_id", "") or array_frame_id),
                timestamp_ns=timestamp_ns,
                slot_type=SlotType.UNKNOWN,
                object_id=str(detection.id).strip() or None,
                orientation_xyzw=orientation,
                size_xyz_m=size,
            )
        )
    return tuple(converted)


def _validated_confidence(value: Any, source: str) -> float:
    """把消息置信度收窄为0到1的有限浮点数。"""

    if isinstance(value, bool):
        raise ValueError(f"{source} 置信度不能使用 bool")
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{source} 置信度格式无效：{value!r}") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{source} 置信度必须是0到1的有限数：{confidence!r}")
    return confidence


def _vision_result_position(result: Any) -> Any:
    return _vision_result_pose(result).position


def _vision_result_pose(result: Any) -> Any:
    pose = result.pose
    if hasattr(pose, "pose"):
        pose = pose.pose
    if not hasattr(pose, "position") or not hasattr(pose, "orientation"):
        raise RuntimeError("vision_msgs ObjectHypothesisWithPose.pose 缺少 position/orientation")
    return pose
