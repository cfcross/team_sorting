"""团队唯一的公共数据契约。

本文件统一定义模块之间交换的枚举、不可变 dataclass、19 维最终动作顺序及 JSON
格式。感知、导航、机械臂、FSM、仲裁器、记录器和 ROS2 适配层都依赖这些定义；业务
文件不得自行复制、重排或另造同义接口。本文件不实现 ROS2 消息转换、识别与控制
算法、状态推进或文件记录。

接口中的 ``timestamp_ns`` 是数据采样或对象生成时刻，单位纳秒；``valid`` 表示该
对象是否通过客户端当前的完整性/安全检查，不等于任务成功或机器人已经执行；
``failure_reason`` 用于说明无效或失败原因，便于诊断。空间量以各对象的
``frame_id`` 为准，长度通常为米、角度通常为弧度，不能默认 ``world == odom``。

各 docstring 中的生产者和消费者均指“典型”角色，用于说明模块边界；类型已经存在
不代表对应算法或完整业务接线已经完成。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from numbers import Real
from typing import Any, Mapping, Optional, Sequence


# 固定动作和关节名称

# 全仓库唯一的 19 维最终动作顺序：前 2 项是底盘 v/w，后 17 项依次对应非底盘
# 关节语义。业务文件只能按此顺序读写，禁止自行重排或重新拼接另一套顺序。
#组长和vla
ACTION_NAMES: tuple[str, ...] = (
    "base_v",
    "base_w",
    "slide",
    "head_yaw",
    "head_pitch",
    "left_arm_joint_1",
    "left_arm_joint_2",
    "left_arm_joint_3",
    "left_arm_joint_4",
    "left_arm_joint_5",
    "left_arm_joint_6",
    "left_gripper",
    "right_arm_joint_1",
    "right_arm_joint_2",
    "right_arm_joint_3",
    "right_arm_joint_4",
    "right_arm_joint_5",
    "right_arm_joint_6",
    "right_gripper",
)
#视觉2读取状态计算相机，机械臂1读取状态生成新17维运动指令，机械臂2对新17维运动指令做插值处理，时刻读取状态计算是否到位，组长读取17维和2维合并
# 仅描述 /joint_states 映射后的 17 个非底盘关节，不含底盘 v/w。它与 ACTION_NAMES
# 后 17 项语义对应，但名称字符串并不保证相同，不能把两个元组当成同一个列表使用。
JOINT_NAMES: tuple[str, ...] = (
    "slide_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_eef_gripper_joint",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_eef_gripper_joint",
)


# 状态枚举

class SlotType(str, Enum):
    """目标所在的粗粒度槽位类型。

    典型生产者是 team client 中按配置区域工作的槽位分类逻辑，典型消费者是 FSM、
    导航或抓放规划。它是对三维位置的业务分类估计，不是裁判结果；``UNKNOWN`` 表示
    位置无效、区域配置缺失或无法可靠分类。
    视觉2产生，fsm,底盘2，机械臂1读取
    """

    UNKNOWN = "unknown"#未知
    TABLE = "table"#桌子
    SHELF = "shelf"#架子


class GlobalPhase(str, Enum):
    """全局任务状态。

    典型生产者是 FSM，典型消费者是 ActionMux、遥测和 Recorder。状态覆盖等待、
    搜索、导航、抓取、放置、返区和失败路径，只描述客户端业务进度，不代表裁判已
    确认得分。

    fsm产生，组长和vla读取
    """

    WAIT_READY = "WAIT_READY" #等待节点、传感器和依赖准备好
    LOAD_TASK = "LOAD_TASK"#等待解析并装载当前任务
    SEARCH_TARGET = "SEARCH_TARGET"#搜索目标物体
    NAV_TO_PICK = "NAV_TO_PICK"#底盘导航到适合抓取的位置
    REFINE_TARGET = "REFINE_TARGET"#靠近后重新识别，得到更准确的三维位置
    PLAN_PICK = "PLAN_PICK"#生成抓取位姿、IK和轨迹
    EXECUTE_PICK = "EXECUTE_PICK"#执行机械臂抓取动作
    VERIFY_PICK = "VERIFY_PICK"#根据真实反馈检查是否抓住
    NAV_TO_PLACE = "NAV_TO_PLACE"#底盘导航到放置区域
    PLAN_PLACE = "PLAN_PLACE"#规划放置位姿和轨迹
    EXECUTE_PLACE = "EXECUTE_PLACE"#执行放置动作
    VERIFY_PLACE = "VERIFY_PLACE"#检查物体是否已正确释放
    RETURN_END = "RETURN_END"#返回规定的结束区域
    DONE = "DONE"#客户端认为任务流程完成
    # ————————————————————————————————
    # 【Codex修改-14：澄清安全暂停与失败终态】
    # 1. 修改前两个阶段的注释含义写反，容易把可恢复暂停误读成最终失败。
    # 2. 当前明确SAFE_HOLD等待真实恢复条件，FAILED只能通过RESET离开。
    # 3. 这样文档与FSM及ActionMux的安全语义一致，避免调用方采用错误恢复策略。
    # 4. 仅修改注释，枚举名称和值完全不变。
    SAFE_HOLD = "SAFE_HOLD"#临时安全暂停，保持安全状态，等待真实恢复条件。
    FAILED = "FAILED"#客户端任务已经失败，只能通过RESET离开。
    # ————————————————————————————————


class LocalPhase(str, Enum):
    """机械臂局部动作状态。

    典型生产者是 FSM 或机械臂执行器，典型消费者是 ActionMux、遥测和 Recorder。
    它描述当前抓放子阶段，不在本文件实现轨迹或双臂抓放算法。

    机械臂2产生，组长，fsm,vla读取
    """

    IDLE = "IDLE"#当前没有执行机械臂抓放动作
    MOVE_PREGRASP = "MOVE_PREGRASP"#移动到预抓取位置
    HUG_OPEN = "HUG_OPEN"#张开双臂或夹爪，准备包夹
    APPROACH = "APPROACH"#从预抓取位置接近物体
    HUG_CLOSE = "HUG_CLOSE"#闭合双臂或夹爪抓住物体
    TEST_LIFT = "TEST_LIFT"#小幅抬起，测试是否抓稳
    VERIFY = "VERIFY"#根据视觉、关节误差或 effort 验证抓取
    RETREAT = "RETREAT"#抓取后撤离物体原位置
    TRANSPORT_HOLD = "TRANSPORT_HOLD"#搬运过程中保持物体
    MOVE_PREPLACE = "MOVE_PREPLACE"#移动到预放置位置
    LOWER_OBJECT = "LOWER_OBJECT"#将物体逐步下降到目标区域
    RELEASE = "RELEASE"#松开夹爪或双臂释放物体
    STOW = "STOW"#收回机械臂到安全姿态
    FAILED = "FAILED"#当前机械臂局部动作失败


class ArmMotionPhase(str, Enum):
    """规划路点的动作区段，不表示执行器已经进入或完成对应实时状态。"""

    PREGRASP = "PREGRASP"
    GRASP = "GRASP"
    LIFT = "LIFT"
    RETREAT = "RETREAT"
    PREPLACE = "PREPLACE"
    LOWER = "LOWER"
    RELEASE = "RELEASE"
    POST_RELEASE_RETREAT = "POST_RELEASE_RETREAT"


# 通用校验

def ensure_finite_vector(values: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    """校验并冻结一个定长有限浮点向量。

    典型调用者是带固定维度向量的公共接口。该函数只检查长度、可转为浮点且数值
    有限，不推断单位、frame 或业务是否成功。

    参数：``values`` 为待校验序列；``length`` 为期望项数；``name`` 用于错误提示。
    返回：只包含 Python ``float`` 的元组，单位和坐标系由调用接口定义。
    失败：长度不符、无法转为浮点或出现 NaN/Inf 时抛出 ``ValueError``。
    """

    if len(values) != length:
        raise ValueError(f"{name} 必须包含 {length} 项，实际为 {len(values)} 项")
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须全部为数值") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} 必须全部为有限数，不能包含 NaN 或 Inf")
    return result


def _strict_nonnegative_ns(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} 必须是非负整数且不能是 bool")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} 必须是严格 bool")
    return value


def _strict_finite_vector(
    values: object, length: int, name: str
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} 必须包含 {length} 项真实有限数")
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须包含 {length} 项真实有限数") from exc
    if len(raw) != length:
        raise ValueError(f"{name} 必须包含 {length} 项，实际为 {len(raw)} 项")
    if any(type(item) is bool or not isinstance(item, Real) for item in raw):
        raise ValueError(f"{name} 每项必须是 numbers.Real 且不能使用 bool")
    result = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} 必须全部为有限数")
    return result


def _strict_optional_finite(
    value: Optional[float], name: str, *, positive: bool = False, nonnegative: bool = False
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数或 None，不能是 bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数")
    if positive and result <= 0.0:
        raise ValueError(f"{name} 必须大于 0")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} 必须大于等于 0")
    return result


def _normalized_optional_quaternion(
    value: Optional[Sequence[float]], name: str
) -> Optional[tuple[float, float, float, float]]:
    if value is None:
        return None
    x, y, z, w = _strict_finite_vector(value, 4, name)
    norm = math.hypot(x, y, z, w)
    if norm == 0.0:
        raise ValueError(f"{name} 四元数范数必须大于 0")
    return (x / norm, y / norm, z / norm, w / norm)


# 任务与机器人实际状态

# ————————————————————————————————
# 【Codex修改-15：纠正TaskSpec解析失败文档】
# 1. 修改前文档称解析失败会返回valid=False，与唯一InstructionParser实际抛异常冲突。
# 2. 当前明确非法JSON、缺字段或类型错误会抛出ValueError，不构造无效TaskSpec。
# 3. 这样调用方不会错误地等待一个永远不会返回的valid=False解析结果。
# 4. 仅修改TaskSpec docstring，字段、顺序、JSON格式和运行逻辑均不变。
@dataclass(frozen=True)
class TaskSpec:
    """结构化比赛任务。。当前比赛要求机器人完成什么任务。
    task_id                 任务编号
    instruction             原始任务文字
    target_kind/body/color  要找的物体属性
    place_type              放置要求类型
    place_world_xyz         物体最终目标位置
    place_radius            允许误差半径
    direction               方向要求
    valid                   任务是否有效
    FSM：决定当前任务流程；
    视觉1：筛选目标类别、颜色；
    底盘：确定需要去哪个区域；
    机械臂1：生成抓取和放置目标。

    典型生产者是唯一任务解析入口 ``InstructionParser``，典型消费者是 FSM、导航和
    抓放规划。参数来自 ``/material/instruction``，包括任务编号、目标属性和放置约束。
    ``place_world_xyz`` 是任务要求的物体中心目标，单位米，不是夹爪末端位姿；它必须
    经过规划转换为 ``PlaceTarget``。``None`` 表示可选字段未给出。当前唯一
    ``InstructionParser`` 当前遇到非法 JSON、缺少字段或类型错误时抛出 ``ValueError``；
    其他组装边界也可构造 ``valid=False`` 且只携带失败诊断的对象，此时不要求完整放置
    字段，以便表达解析失败而不伪造任务事实。
    有效任务的 ``instruction``、``target_kind``、``target_body``、``target_color``
    必须全部来自官方任务且非空；构造器不按 task ID 或颜色补造身份。
    """
    # ————————————————————————————————

    task_id: int
    instruction: str
    target_kind: str
    target_body: str
    target_color: str
    place_type: str = ""
    place_world_xyz: Optional[tuple[float, float, float]] = None
    place_frame_id: str = ""
    place_radius: Optional[float] = None
    ref_prop: Optional[str] = None
    ref_prop_body: Optional[str] = None
    direction: Optional[str] = None
    timestamp_ns: int = 0
    valid: bool = True
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if type(self.task_id) is not int or self.task_id < 0:
            raise ValueError("TaskSpec.task_id 必须是非负整数且不能是 bool")
        _strict_nonnegative_ns(self.timestamp_ns, "TaskSpec.timestamp_ns")
        _strict_bool(self.valid, "TaskSpec.valid")
        if not isinstance(self.failure_reason, str):
            raise ValueError("TaskSpec.failure_reason 必须是字符串")
        if not self.valid:
            if not self.failure_reason.strip():
                raise ValueError("无效 TaskSpec 必须提供非空 failure_reason")
            return

        if self.failure_reason:
            raise ValueError("有效 TaskSpec 不得携带 failure_reason")
        for name in ("instruction", "target_kind", "target_body", "target_color"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"有效 TaskSpec.{name} 必须是非空字符串")
        official_place_types = {"shelf_point", "table_point", "shelf_prop_side"}
        if self.place_type not in official_place_types:
            raise ValueError(
                "TaskSpec.place_type 必须是 shelf_point、table_point 或 shelf_prop_side"
            )
        if self.place_frame_id != "world":
            raise ValueError('有效 TaskSpec.place_frame_id 必须严格为 "world"')
        if self.place_world_xyz is None:
            raise ValueError("有效 TaskSpec.place_world_xyz 不能为空")
        object.__setattr__(
            self,
            "place_world_xyz",
            _strict_finite_vector(self.place_world_xyz, 3, "TaskSpec.place_world_xyz"),
        )
        radius = _strict_optional_finite(
            self.place_radius, "TaskSpec.place_radius", positive=True
        )
        if radius is None:
            raise ValueError("有效 TaskSpec.place_radius 不能为空")
        object.__setattr__(self, "place_radius", radius)
        if self.place_type == "shelf_prop_side":
            if not isinstance(self.ref_prop, str) or not self.ref_prop.strip():
                raise ValueError("shelf_prop_side 必须提供非空 ref_prop")
            if not isinstance(self.ref_prop_body, str) or not self.ref_prop_body.strip():
                raise ValueError("shelf_prop_side 必须提供非空 ref_prop_body")
            if self.direction not in {"left", "right"}:
                raise ValueError("shelf_prop_side.direction 必须是 left 或 right")
        elif any(
            value is not None
            for value in (self.ref_prop, self.ref_prop_body, self.direction)
        ):
            raise ValueError(
                "shelf_point/table_point 不得携带 ref_prop、ref_prop_body 或 direction"
            )


@dataclass(frozen=True)
class BaseState:
    """底盘实际状态。
    position_xyz              当前三维位置
    orientation_xyzw          当前姿态四元数
    yaw                       当前朝向角
    linear_velocity_xyz       实际线速度
    angular_velocity_xyz      实际角速度
    frame_id                  所属坐标系
    timestamp_ns              数据时间
    valid                     是否有效
    底盘负责人：计算当前位置和导航误差；
    视觉2：将目标从相机/底盘坐标转换到世界坐标；
    team client：生成 SensorSnapshot；
    FSM：间接判断是否到达区域。

    典型生产者是 Odom 到公共接口的适配逻辑，典型消费者是三维估计、导航和控制周期
    快照。它回答“底盘现在在哪里、怎样运动”，不能与 ``BaseCommand`` 候选速度混用。

    ``position_xyz`` 单位米，``orientation_xyzw`` 为 ROS 顺序四元数，线速度单位 m/s，
    角速度单位 rad/s；所有量均位于 ``frame_id`` 指定的坐标系。消息缺失或过期时
    ``valid=False``。
    """

    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    yaw: float
    linear_velocity_xyz: tuple[float, float, float]
    angular_velocity_xyz: tuple[float, float, float]
    frame_id: str
    timestamp_ns: int
    valid: bool = True
    failure_reason: str = ""


@dataclass(frozen=True)
class RobotJointState:
    """17 个非底盘关节的实际反馈。
    position[17]    实际关节位置
    velocity[17]    实际关节速度
    effort[17]      实际关节受力/力矩反馈
    joint_names     固定关节顺序
    timestamp_ns
    valid
    视觉2：计算相机实际位置；
    机械臂1：作为IK和轨迹规划起点；
    机械臂2：判断关节是否到位；
    ActionMux：未控制关节保持实际位置；
    抓取验证：可使用 effort 判断是否夹住。

    典型生产者是把实际 ``/joint_states`` 映射到团队顺序的 ``JointStateMapper``，
    典型消费者是三维估计、IK/规划、执行器和 ActionMux。它回答“现在在哪里”；
    ``IKResult`` 回答“希望去哪里”，例如实际反馈为 0.2、IK 目标可以是 0.3。

    顺序固定为 ``JOINT_NAMES``。位置中 slide 单位米，其余旋转关节为弧度，夹爪按
    官方 0～1 控制量；速度和 effort 沿用 ROS 消息单位。名称缺失、长度错误或非
    有限数会抛出 ``ValueError``。
    """

    position: tuple[float, ...]
    velocity: tuple[float, ...]
    effort: tuple[float, ...]
    timestamp_ns: int
    valid: bool = True
    failure_reason: str = ""
    joint_names: tuple[str, ...] = JOINT_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", ensure_finite_vector(self.position, 17, "关节位置"))
        object.__setattr__(self, "velocity", ensure_finite_vector(self.velocity, 17, "关节速度"))
        object.__setattr__(self, "effort", ensure_finite_vector(self.effort, 17, "关节 effort"))
        if tuple(self.joint_names) != JOINT_NAMES:
            raise ValueError("RobotJointState.joint_names 必须使用团队统一顺序")


# 感知接口

@dataclass(frozen=True)
class RGBFrame:
    """一帧彩色图像。
    image          图像数组
    encoding       rgb8、bgr8等格式
    frame_id       相机坐标系
    timestamp_ns
    valid
    视觉1 / perception_2d.py；
    YOLO适配器；
    Recorder/VLA也可能保存原图。

    它是 ``perception_2d`` 二维检测的主要业务输入。典型生产者是 perception node
    的 ROS 图像适配逻辑，典型消费者是 ``OfficialYoloAdapter``；任务类别或颜色可
    用于筛选检测。当前 OfficialYoloAdapter 为兼容官方统一 backend 接口，调用时仍
    要求传入 ``DepthFrame`` 和 ``CameraIntrinsics``。这不表示 ``perception_2d``
    负责深度反投影或三维坐标估计；深度、内参、``BaseState`` 和
    ``RobotJointState`` 的三维几何处理仍属于 ``perception_3d``。

    ``image`` 通常是 NumPy/OpenCV 数组，颜色编码由 ``encoding`` 明确；坐标系由
    ``frame_id`` 指定，时间为纳秒。图像转换失败时 ``valid=False``。
    """

    image: Any
    encoding: str
    frame_id: str
    timestamp_ns: int
    valid: bool = True
    failure_reason: str = ""


@dataclass(frozen=True)
class DepthFrame:
    """一帧与彩色图对齐的深度图。
    image           原始深度数组
    unit_scale_m    原始值换算成米的比例
    frame_id
    timestamp_ns
    valid

    典型生产者是 perception node 的深度消息适配逻辑，三维几何上的典型消费者是
    ``perception_3d`` 三维估计器。当前 OfficialYoloAdapter 为兼容官方统一 backend
    接口，调用时也要求传入该对象，但这不表示 ``perception_2d`` 负责深度反投影或
    三维坐标估计。

    原始像素乘 ``unit_scale_m`` 得到米；``frame_id`` 是相机坐标系。无效深度、编码
    不支持或时间差过大时由调用方标为无效。
    """

    image: Any
    unit_scale_m: float
    frame_id: str
    timestamp_ns: int
    valid: bool = True
    failure_reason: str = ""


@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参。
    k[9]       3×3相机内参矩阵
    width      图像宽度
    height     图像高度
    frame_id
    timestamp_ns
    valid

    典型生产者是 perception node 的 CameraInfo 适配逻辑，三维几何上的典型消费者
    是 ``perception_3d`` 三维估计器。当前 OfficialYoloAdapter 为兼容官方统一
    backend 接口，调用时也要求传入该对象，但这不表示 ``perception_2d`` 负责使用
    内参进行深度反投影或三维坐标估计。

    ``k`` 为按行展开的 3×3 矩阵，像素单位；宽高单位像素。它只描述内参，不包含
    camera 到 world/odom 的外参。矩阵长度错误时抛出 ``ValueError``。
    """

    k: tuple[float, ...]
    width: int
    height: int
    frame_id: str
    timestamp_ns: int
    valid: bool = True
    failure_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "k", ensure_finite_vector(self.k, 9, "相机内参 K"))


@dataclass(frozen=True)
class Detection2D:
    """单个二维目标框。
    class_id          类别
    bbox_xyxy         检测框坐标
    confidence        置信度
    timestamp_ns
    frame_id          检测框所属图像坐标系
    track_id          二维稳定器分配的稳定轨迹编号
    valid

    典型生产者是二维检测适配器，典型消费者是检测稳定逻辑和三维估计器。它是图像
    平面中的检测结果，不携带深度或三维中心。原始单帧检测的 ``track_id`` 为
    ``None``；经过二维稳定器确认后，输出携带非负稳定编号，供三维滤波维持一对一
    轨迹身份。

    ``bbox_xyxy`` 为图像像素坐标 ``(x0,y0,x1,y1)``；``confidence`` 范围建议为
    0～1；``timestamp_ns`` 为产生该框的 RGB 帧采样时间，单位纳秒；``frame_id``
    必须与该 RGB 像素坐标系一致。该接口不携带三维位置。框越界可由检测适配层
    裁剪，格式错误时返回无效结果。
    """

    class_id: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    timestamp_ns: int
    valid: bool = True
    failure_reason: str = ""
    frame_id: str = ""
    track_id: Optional[int] = None


@dataclass(frozen=True)
class ObjectEstimate3D:
    """目标物体中心的三维估计。
    | 字段               | 含义                                              |
    | ---------------- | ----------------------------------------------- |
    | `class_id`       | 物体类别，例如箱子、瓶子等                                   |
    | `position_xyz`   | 物体中心三维位置 `(x,y,z)`，通常单位为米                       |
    | `confidence`     | 三维估计可信程度，通常建议在0～1                               |
    | `frame_id`       | 三维位置属于哪个坐标系，例如 `camera_link`、`base_link`、`odom` |
    | `timestamp_ns`   | 这次三维估计对应的时间，单位纳秒                                |
    | `slot_type`      | 目标属于桌面、货架还是未知区域                                 |
    | `valid`          | 当前三维估计是否可以使用                                    |
    | `failure_reason` | 无效时记录原因，如深度缺失、坐标转换失败                            |
    | `object_id`      | 可选稳定轨迹身份；不是当前任务body                             |
    | `orientation_xyzw` | 可选物体观测姿态；未知时为None                             |
    | `size_xyz_m`     | 可选三轴尺寸；未知时为None                                 |

    业务上的典型生产者是 ``perception_3d`` 三维估计器，典型消费者是 team client、
    导航和抓放规划；``ros_nodes`` 只负责它与 ROS 消息之间的转换，不是三维估计算法
    生产者。该对象是物体中心估计，不是带朝向的 ``Pose3D`` 夹爪末端目标：箱子中心
    可以在中间，而左右夹爪目标分别位于箱子两侧。

    ``position_xyz`` 单位米，位于 ``frame_id`` 指定的 world/odom 等坐标系；它应是
    经过表面点补偿后的物体中心估计。``slot_type`` 可在 team client 收到后计算；
    估计失败时 ``valid=False``。本类型是纯感知事实，不携带当前任务的
    ``target_body``；有效中心估计不要求可选身份、姿态或尺寸存在。
    """

    class_id: str
    position_xyz: tuple[float, float, float]
    confidence: float
    frame_id: str
    timestamp_ns: int
    slot_type: SlotType = SlotType.UNKNOWN
    valid: bool = True
    failure_reason: str = ""
    object_id: Optional[str] = None
    orientation_xyzw: Optional[tuple[float, float, float, float]] = None
    size_xyz_m: Optional[tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_xyz",
            _strict_finite_vector(self.position_xyz, 3, "ObjectEstimate3D.position_xyz"),
        )
        confidence = _strict_optional_finite(
            self.confidence, "ObjectEstimate3D.confidence", nonnegative=True
        )
        if confidence is None or confidence > 1.0:
            raise ValueError("ObjectEstimate3D.confidence 必须位于 0 到 1")
        object.__setattr__(self, "confidence", confidence)
        _strict_nonnegative_ns(self.timestamp_ns, "ObjectEstimate3D.timestamp_ns")
        _strict_bool(self.valid, "ObjectEstimate3D.valid")
        if not isinstance(self.class_id, str) or not isinstance(self.frame_id, str):
            raise ValueError("ObjectEstimate3D.class_id/frame_id 必须是字符串")
        if self.valid and (not self.class_id.strip() or not self.frame_id.strip()):
            raise ValueError("有效 ObjectEstimate3D.class_id/frame_id 必须非空")
        if not isinstance(self.failure_reason, str):
            raise ValueError("ObjectEstimate3D.failure_reason 必须是字符串")
        if not self.valid and not self.failure_reason.strip():
            raise ValueError("无效 ObjectEstimate3D 必须提供 failure_reason")
        if not isinstance(self.slot_type, SlotType):
            raise ValueError("ObjectEstimate3D.slot_type 必须是 SlotType")
        if self.object_id is not None and (
            not isinstance(self.object_id, str) or not self.object_id.strip()
        ):
            raise ValueError("ObjectEstimate3D.object_id 必须是非空字符串或 None")
        object.__setattr__(
            self,
            "orientation_xyzw",
            _normalized_optional_quaternion(
                self.orientation_xyzw, "ObjectEstimate3D.orientation_xyzw"
            ),
        )
        if self.size_xyz_m is not None:
            size = _strict_finite_vector(
                self.size_xyz_m, 3, "ObjectEstimate3D.size_xyz_m"
            )
            if any(value <= 0.0 for value in size):
                raise ValueError("ObjectEstimate3D.size_xyz_m 三轴必须均大于 0")
            object.__setattr__(self, "size_xyz_m", size)


@dataclass(frozen=True)
class RigidTransform3D:
    """普通Python刚体变换快照；数值严格把source坐标转换到target坐标。

    frame只去除首尾空白，不删除前导斜杠或推断别名；同frame单位检查使用规范化名称。
    """

    source_frame: str
    target_frame: str
    translation_xyz: Optional[tuple[float, float, float]]
    rotation_xyzw: Optional[tuple[float, float, float, float]]
    timestamp_ns: int
    valid: bool
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_frame, str) or not self.source_frame.strip():
            raise ValueError("RigidTransform3D.source_frame 必须是非空字符串")
        if not isinstance(self.target_frame, str) or not self.target_frame.strip():
            raise ValueError("RigidTransform3D.target_frame 必须是非空字符串")
        object.__setattr__(self, "source_frame", self.source_frame.strip())
        object.__setattr__(self, "target_frame", self.target_frame.strip())
        _strict_nonnegative_ns(self.timestamp_ns, "RigidTransform3D.timestamp_ns")
        _strict_bool(self.valid, "RigidTransform3D.valid")
        if not isinstance(self.failure_reason, str):
            raise ValueError("RigidTransform3D.failure_reason 必须是字符串")
        if not self.valid:
            if not self.failure_reason.strip():
                raise ValueError("无效 RigidTransform3D 必须提供 failure_reason")
            if self.translation_xyz is not None:
                object.__setattr__(
                    self,
                    "translation_xyz",
                    _strict_finite_vector(
                        self.translation_xyz, 3, "RigidTransform3D.translation_xyz"
                    ),
                )
            if self.rotation_xyzw is not None:
                object.__setattr__(
                    self,
                    "rotation_xyzw",
                    _normalized_optional_quaternion(
                        self.rotation_xyzw, "RigidTransform3D.rotation_xyzw"
                    ),
                )
            return
        if self.failure_reason:
            raise ValueError("有效 RigidTransform3D 不得携带 failure_reason")
        if self.translation_xyz is None or self.rotation_xyzw is None:
            raise ValueError("有效 RigidTransform3D 必须提供平移和旋转")
        translation = _strict_finite_vector(
            self.translation_xyz, 3, "RigidTransform3D.translation_xyz"
        )
        rotation = _normalized_optional_quaternion(
            self.rotation_xyzw, "RigidTransform3D.rotation_xyzw"
        )
        assert rotation is not None
        object.__setattr__(self, "translation_xyz", translation)
        object.__setattr__(self, "rotation_xyzw", rotation)
        if self.source_frame == self.target_frame:
            identity_tolerance = 1e-9
            if any(abs(value) > identity_tolerance for value in translation):
                raise ValueError("同frame刚体变换只能使用零平移")
            if (
                abs(rotation[0]) > identity_tolerance
                or abs(rotation[1]) > identity_tolerance
                or abs(rotation[2]) > identity_tolerance
                or abs(abs(rotation[3]) - 1.0) > identity_tolerance
            ):
                raise ValueError("同frame刚体变换只能使用单位旋转")


@dataclass(frozen=True)
class ArmPlanningConfig:
    """允许部分未标定的机械臂规划配置；按抓取/放置操作分别验证完整性。

    左右夹爪min/max若提供必须位于官方actuator ``ctrlrange=[0,1]``；这不代表
    open/closed或夹持效果已经标定。
    """

    min_object_confidence: Optional[float] = None
    transform_max_age_ns: Optional[int] = None
    object_estimate_max_age_ns: Optional[int] = None
    joint_state_max_age_ns: Optional[int] = None
    planned_context_max_age_ns: Optional[int] = None
    confirmed_context_max_age_ns: Optional[int] = None
    pregrasp_distance_m: Optional[float] = None
    grasp_contact_offset_m: Optional[float] = None
    lift_distance_m: Optional[float] = None
    retreat_distance_m: Optional[float] = None
    preplace_height_m: Optional[float] = None
    release_offset_m: Optional[float] = None
    post_release_retreat_distance_m: Optional[float] = None
    settle_time_s: Optional[float] = None
    max_slide_waypoint_delta_m: Optional[float] = None
    max_arm_waypoint_delta_rad: Optional[float] = None
    max_gripper_waypoint_delta: Optional[float] = None
    pregrasp_duration_s: Optional[float] = None
    grasp_duration_s: Optional[float] = None
    lift_duration_s: Optional[float] = None
    retreat_duration_s: Optional[float] = None
    preplace_duration_s: Optional[float] = None
    lower_duration_s: Optional[float] = None
    release_duration_s: Optional[float] = None
    post_release_retreat_duration_s: Optional[float] = None
    left_gripper_min: Optional[float] = None
    left_gripper_max: Optional[float] = None
    right_gripper_min: Optional[float] = None
    right_gripper_max: Optional[float] = None
    left_gripper_open: Optional[float] = None
    left_gripper_closed: Optional[float] = None
    right_gripper_open: Optional[float] = None
    right_gripper_closed: Optional[float] = None
    gripper_verified_in_official_environment: bool = False

    def __post_init__(self) -> None:
        optional_positive = (
            "transform_max_age_ns", "object_estimate_max_age_ns",
            "joint_state_max_age_ns", "planned_context_max_age_ns",
            "confirmed_context_max_age_ns", "pregrasp_distance_m",
            "lift_distance_m", "retreat_distance_m", "preplace_height_m",
            "post_release_retreat_distance_m", "max_slide_waypoint_delta_m",
            "max_arm_waypoint_delta_rad", "max_gripper_waypoint_delta",
            "pregrasp_duration_s", "grasp_duration_s", "lift_duration_s",
            "retreat_duration_s", "preplace_duration_s", "lower_duration_s",
            "release_duration_s", "post_release_retreat_duration_s",
        )
        integer_names = {
            "transform_max_age_ns", "object_estimate_max_age_ns",
            "joint_state_max_age_ns", "planned_context_max_age_ns",
            "confirmed_context_max_age_ns",
        }
        for name in optional_positive:
            value = getattr(self, name)
            if value is None:
                continue
            if name in integer_names:
                if type(value) is not int or value <= 0:
                    raise ValueError(f"ArmPlanningConfig.{name} 必须是正整数纳秒")
            else:
                object.__setattr__(
                    self,
                    name,
                    _strict_optional_finite(value, f"ArmPlanningConfig.{name}", positive=True),
                )
        for name in ("grasp_contact_offset_m", "release_offset_m", "settle_time_s"):
            object.__setattr__(
                self,
                name,
                _strict_optional_finite(
                    getattr(self, name), f"ArmPlanningConfig.{name}", nonnegative=True
                ),
            )
        confidence = _strict_optional_finite(
            self.min_object_confidence,
            "ArmPlanningConfig.min_object_confidence",
            nonnegative=True,
        )
        if confidence is not None and confidence > 1.0:
            raise ValueError("ArmPlanningConfig.min_object_confidence 必须位于 0 到 1")
        object.__setattr__(self, "min_object_confidence", confidence)
        _strict_bool(
            self.gripper_verified_in_official_environment,
            "ArmPlanningConfig.gripper_verified_in_official_environment",
        )
        for side in ("left", "right"):
            lower_name = f"{side}_gripper_min"
            upper_name = f"{side}_gripper_max"
            lower = _strict_optional_finite(getattr(self, lower_name), lower_name)
            upper = _strict_optional_finite(getattr(self, upper_name), upper_name)
            if lower is not None and not 0.0 <= lower <= 1.0:
                raise ValueError(f"{lower_name} 必须位于官方 ctrlrange [0, 1]")
            if upper is not None and not 0.0 <= upper <= 1.0:
                raise ValueError(f"{upper_name} 必须位于官方 ctrlrange [0, 1]")
            if lower is not None and upper is not None and lower >= upper:
                raise ValueError(f"{side} gripper min 必须小于 max")
            object.__setattr__(self, lower_name, lower)
            object.__setattr__(self, upper_name, upper)
            values: list[float] = []
            for state in ("open", "closed"):
                name = f"{side}_gripper_{state}"
                value = _strict_optional_finite(getattr(self, name), name)
                if value is not None and lower is not None and upper is not None and not lower <= value <= upper:
                    raise ValueError(f"{name} 必须位于对应 gripper 范围内")
                object.__setattr__(self, name, value)
                if value is not None:
                    values.append(value)
            if len(values) == 2 and values[0] == values[1]:
                raise ValueError(f"{side} gripper open 和 closed 不能相同")

    def _require(self, names: Sequence[str], operation: str) -> None:
        for name in names:
            if getattr(self, name) is None:
                raise ValueError(f"{operation}缺少 ArmPlanningConfig.{name}")
        if not self.gripper_verified_in_official_environment:
            raise ValueError(
                f"{operation}要求 ArmPlanningConfig.gripper_verified_in_official_environment=True"
            )

    def validate_for_grasp(self) -> None:
        self._require(
            (
                "min_object_confidence", "transform_max_age_ns",
                "object_estimate_max_age_ns", "joint_state_max_age_ns",
                "planned_context_max_age_ns", "pregrasp_distance_m",
                "grasp_contact_offset_m", "lift_distance_m", "retreat_distance_m",
                "max_slide_waypoint_delta_m", "max_arm_waypoint_delta_rad",
                "max_gripper_waypoint_delta", "pregrasp_duration_s",
                "grasp_duration_s", "lift_duration_s", "retreat_duration_s",
                "left_gripper_open", "left_gripper_closed",
                "right_gripper_open", "right_gripper_closed",
                "left_gripper_min", "left_gripper_max",
                "right_gripper_min", "right_gripper_max",
            ),
            "抓取规划",
        )

    def validate_for_place(self) -> None:
        self._require(
            (
                "transform_max_age_ns", "joint_state_max_age_ns",
                "confirmed_context_max_age_ns", "preplace_height_m",
                "release_offset_m", "post_release_retreat_distance_m",
                "settle_time_s", "max_slide_waypoint_delta_m",
                "max_arm_waypoint_delta_rad", "max_gripper_waypoint_delta",
                "preplace_duration_s", "lower_duration_s", "release_duration_s",
                "post_release_retreat_duration_s", "left_gripper_open",
                "left_gripper_closed", "right_gripper_open", "right_gripper_closed",
                "left_gripper_min", "left_gripper_max",
                "right_gripper_min", "right_gripper_max",
            ),
            "放置规划",
        )


# 控制周期快照

@dataclass(frozen=True)
class SensorSnapshot:
    """team client 一个控制周期使用的轻量状态快照。
    | 字段                 | 含义                                  |
    | ------------------ | ----------------------------------- |
    | `task`             | 当前任务；尚未收到任务时可以是 `None`              |
    | `base`             | 当前底盘实际状态；Odom缺失时可以是 `None`          |
    | `joints`           | 当前17维实际关节反馈；JointState缺失时可以是 `None` |
    | `object_estimates` | 当前周期识别到的一个或多个三维物体                   |
    | `timestamp_ns`     | 这份快照的汇总时间                           |
    | `valid`            | 当前关键数据是否足够支持控制逻辑                    |
    | `failure_reason`   | 快照无效的原因，例如关节状态过期                    |


    典型生产者是 team client 的输入汇总逻辑，典型消费者是协调 FSM、导航和机械臂
    的业务逻辑。它汇总同周期数据，不会把估计值变成实际反馈。

    输入只含任务、底盘、实际关节和三维目标，不含 RGB、Depth 或 CameraInfo。时间为
    纳秒，空间量保留各自 frame。关键输入缺失或过期时 ``valid=False``。

    当前骨架状态：该快照已构造，但完整协调消费者尚未接入。
    """

    task: Optional[TaskSpec]
    base: Optional[BaseState]
    joints: Optional[RobotJointState]
    object_estimates: tuple[ObjectEstimate3D, ...]
    timestamp_ns: int
    valid: bool
    failure_reason: str = ""


# 机械臂规划接口

@dataclass(frozen=True)
class Pose3D:
    """带位置和方向的三维刚体目标位姿。
    | 字段                 | 含义                          |
    | ------------------ | --------------------------- |
    | `position_xyz`     | 目标位置 `(x,y,z)`，通常单位米        |
    | `orientation_xyzw` | 目标朝向，用四元数表示，顺序为 `(x,y,z,w)` |
    | `frame_id`         | 目标位姿属于哪个坐标系                 |

    典型生产者是抓放目标规划逻辑，典型消费者是 IK/KDL 适配器。它描述夹爪等刚体
    末端“希望到哪里、朝向哪里”，不能把 ``ObjectEstimate3D`` 的物体中心直接代入。

    位置单位米，四元数为 ``xyzw``，坐标系由 ``frame_id`` 指定。IK 通常要求
    ``base_link``/footprint 坐标系；传入 world 位姿而未转换时适配器会返回失败。
    构造时严格拒绝 bool、NaN/Inf、零四元数和空 frame，并冻结归一化四元数；本类型
    总是表示真实有效 Pose，不能复用 ROS 未知姿态哨兵，也不自动改写 frame 名称。
    """

    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    frame_id: str

    def __post_init__(self) -> None:
        position = _strict_finite_vector(
            self.position_xyz, 3, "Pose3D.position_xyz"
        )
        orientation = _normalized_optional_quaternion(
            self.orientation_xyzw, "Pose3D.orientation_xyzw"
        )
        assert orientation is not None
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("Pose3D.frame_id 必须是非空字符串")
        object.__setattr__(self, "position_xyz", position)
        object.__setattr__(self, "orientation_xyzw", orientation)
        object.__setattr__(self, "frame_id", self.frame_id.strip())


def _require_frozen_pose(value: object, name: str) -> None:
    """防御跨进程损坏对象；正常Pose3D已在构造时满足这些条件。"""

    if not isinstance(value, Pose3D):
        raise ValueError(f"{name} 必须是 Pose3D")
    try:
        position = _strict_finite_vector(value.position_xyz, 3, f"{name}.position_xyz")
        orientation = _strict_finite_vector(
            value.orientation_xyzw, 4, f"{name}.orientation_xyzw"
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} 是损坏的 Pose3D：{exc}") from exc
    norm = math.hypot(*orientation)
    if norm == 0.0 or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name}.orientation_xyzw 必须是已归一化非零四元数")
    if not isinstance(value.frame_id, str) or not value.frame_id.strip():
        raise ValueError(f"{name}.frame_id 必须是非空字符串")
    if value.frame_id != value.frame_id.strip():
        raise ValueError(f"{name}.frame_id 必须保存去除首尾空白后的值")
    if (
        any(type(item) is not float for item in value.position_xyz)
        or any(type(item) is not float for item in value.orientation_xyzw)
        or tuple(value.position_xyz) != position
        or tuple(value.orientation_xyzw) != orientation
    ):
        raise ValueError(f"{name} 必须保存规范化的有限float元组")


@dataclass(frozen=True)
class GraspContext:
    """计划抓取关系及其确认状态；确认不等于重新测量真实夹爪相对物体变换。

    ``object_from_*_gripper`` 始终是 ArmPlanner 生成的规划关系。``confirmed=True``
    只表示执行器依据实际反馈和抓取验证确认本次计划抓取成立；执行器不得在没有传感
    证据的情况下重算或把该关系描述为真实测量标定结果。
    """

    task_id: int
    target_body: str
    target_class_id: str
    object_id: str
    object_frame: str
    object_size_xyz_m: Optional[tuple[float, float, float]]
    object_from_left_gripper: Optional[RigidTransform3D]
    object_from_right_gripper: Optional[RigidTransform3D]
    object_orientation_world_xyzw_at_grasp: Optional[
        tuple[float, float, float, float]
    ]
    orientation_observed_at_ns: Optional[int]
    planned_at_ns: int
    confirmed_at_ns: Optional[int]
    confirmed: bool
    valid: bool
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if type(self.task_id) is not int or self.task_id < 0:
            raise ValueError("GraspContext.task_id 必须是非负整数且不能是 bool")
        _strict_nonnegative_ns(self.planned_at_ns, "GraspContext.planned_at_ns")
        _strict_bool(self.confirmed, "GraspContext.confirmed")
        _strict_bool(self.valid, "GraspContext.valid")
        for name in (
            "target_body", "target_class_id", "object_id", "object_frame"
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"GraspContext.{name} 必须是非空字符串")
            object.__setattr__(self, name, value.strip())
        if self.confirmed_at_ns is not None:
            _strict_nonnegative_ns(
                self.confirmed_at_ns, "GraspContext.confirmed_at_ns"
            )
        if self.confirmed:
            if not self.valid:
                raise ValueError("confirmed GraspContext 必须同时 valid=True")
            if self.confirmed_at_ns is None:
                raise ValueError("confirmed GraspContext 必须提供 confirmed_at_ns")
            if self.confirmed_at_ns < self.planned_at_ns:
                raise ValueError("confirmed_at_ns 不能早于 planned_at_ns")
        elif self.confirmed_at_ns is not None:
            raise ValueError("未确认 GraspContext 的 confirmed_at_ns 必须为 None")
        if not isinstance(self.failure_reason, str):
            raise ValueError("GraspContext.failure_reason 必须是字符串")
        if not self.valid:
            if not self.failure_reason.strip():
                raise ValueError("无效 GraspContext 必须提供 failure_reason")
            return
        if self.failure_reason:
            raise ValueError("有效 GraspContext 不得携带 failure_reason")
        if self.object_size_xyz_m is None:
            raise ValueError("有效 GraspContext 必须提供 object_size_xyz_m")
        size = _strict_finite_vector(
            self.object_size_xyz_m, 3, "GraspContext.object_size_xyz_m"
        )
        if any(value <= 0.0 for value in size):
            raise ValueError("GraspContext.object_size_xyz_m 三轴必须均大于 0")
        object.__setattr__(self, "object_size_xyz_m", size)
        for name in ("object_from_left_gripper", "object_from_right_gripper"):
            transform = getattr(self, name)
            if not isinstance(transform, RigidTransform3D) or not transform.valid:
                raise ValueError(f"有效 GraspContext.{name} 必须是有效刚体变换")
            if transform.target_frame != self.object_frame:
                raise ValueError(f"GraspContext.{name}.target_frame 必须等于 object_frame")
            if transform.timestamp_ns > self.planned_at_ns:
                raise ValueError(f"GraspContext.{name} 时间不能晚于 planned_at_ns")
        orientation = _normalized_optional_quaternion(
            self.object_orientation_world_xyzw_at_grasp,
            "GraspContext.object_orientation_world_xyzw_at_grasp",
        )
        if orientation is None:
            raise ValueError("有效 GraspContext 必须提供抓取时物体 world 朝向")
        object.__setattr__(
            self, "object_orientation_world_xyzw_at_grasp", orientation
        )
        if self.orientation_observed_at_ns is None:
            raise ValueError("有效 GraspContext 必须提供 orientation_observed_at_ns")
        _strict_nonnegative_ns(
            self.orientation_observed_at_ns,
            "GraspContext.orientation_observed_at_ns",
        )
        if self.orientation_observed_at_ns > self.planned_at_ns:
            raise ValueError("orientation_observed_at_ns 不能晚于 planned_at_ns")


# 导航接口

@dataclass(frozen=True)
class NavGoal:
    """底盘导航目标。
    | 字段                   | 含义                            |
    | -------------------- | ----------------------------- |
    | `goal_id`            | 导航目标唯一编号                      |
    | `goal_type`          | 目标类型，例如抓取点、放置点、结束区            |
    | `pose_xyyaw`         | 目标 `(x,y,yaw)`，前两项单位米，yaw单位弧度 |
    | `frame_id`           | 目标属于哪个坐标系，例如 `odom`           |
    | `position_tolerance` | 允许的位置误差，单位米                   |
    | `yaw_tolerance`      | 允许的朝向误差，单位弧度                  |
    | `deadline_ns`        | 该导航任务的截止时间                    |
    | `valid`              | 目标是否有效                        |
    | `failure_reason`     | 目标无效的原因                       |

    典型生产者是 FSM/任务规划逻辑，典型消费者是导航控制器。它是目标，不是底盘
    当前位姿或已经到达的证明。

    ``pose_xyyaw`` 的 XY 单位米、yaw 单位弧度，均位于 ``frame_id``；容差分别为米和
    弧度。完整导航尚未实现时规划器应抛出 ``NotImplementedError``，不得伪造到达。
    """

    goal_id: str
    goal_type: str
    pose_xyyaw: tuple[float, float, float]
    frame_id: str
    position_tolerance: float
    yaw_tolerance: float
    deadline_ns: int
    valid: bool = True
    failure_reason: str = ""


@dataclass(frozen=True)
class BaseCommand:
    """底盘候选速度命令。
    | 字段               | 含义                    |
    | ---------------- | --------------------- |
    | `v`              | 底盘前进或后退线速度，单位 `m/s`   |
    | `w`              | 底盘绕Z轴旋转角速度，单位 `rad/s` |
    | `timestamp_ns`   | 命令生成时间                |
    | `valid_until_ns` | 命令失效时间；到达此时间后不能继续使用   |
    | `valid`          | 导航模块是否认为这条候选命令有效      |
    | `failure_reason` | 无效原因，如导航目标无效          |



    典型生产者是导航控制器，典型消费者是 ActionMux。它是当前控制周期的短时速度
    建议，不是 Odom 实际状态；生成命令也不表示机器人已经到达目标。

    ``v`` 单位 m/s，``w`` 单位 rad/s，均为机器人底盘前向/绕 Z 轴命令。超过
    ``valid_until_ns`` 后 ActionMux 必须输出零底盘速度。失败时设置 ``valid=False``。
    """

    v: float
    w: float
    timestamp_ns: int
    valid_until_ns: int
    valid: bool = True
    failure_reason: str = ""


@dataclass(frozen=True)
class NavigationStatus:
    """导航技能执行状态。
    | 字段               | 含义                |
    | ---------------- | ----------------- |
    | `goal_id`        | 当前正在执行哪个导航目标      |
    | `state`          | 导航状态描述，如运行中、到达、超时 |
    | `distance_error` | 当前距离目标还有多少米       |
    | `yaw_error`      | 当前朝向与目标相差多少弧度     |
    | `success`        | 是否根据实际Odom确认到达    |
    | `failure_reason` | 导航失败或尚未完成的原因      |
    | `timestamp_ns`   | 状态判断时间            |


    典型生产者是依据实际 Odom 反馈评估目标误差的导航控制器，典型消费者是 FSM。
    ``BaseCommand`` 只是候选建议，不能单凭“已生成命令”把本状态判为成功。

    距离误差单位米、yaw 误差单位弧度；``success`` 只在实际里程计满足容差时为真。
    超时、目标无效或算法未实现时必须填写 ``failure_reason``。

    当前骨架状态：该接口已预留，导航闭环尚未完成。
    """

    goal_id: str
    state: str
    distance_error: float
    yaw_error: float
    success: bool
    failure_reason: str
    timestamp_ns: int


# 机械臂规划接口（续）

@dataclass(frozen=True)
class GraspTarget:
    """双臂抓取位姿集合。
    | 字段               | 含义            |
    | ---------------- | ------------- |
    | `left_pregrasp`  | 左夹爪抓取前的预备位姿   |
    | `right_pregrasp` | 右夹爪抓取前的预备位姿   |
    | `left_grasp`     | 左夹爪真正抓取位姿     |
    | `right_grasp`    | 右夹爪真正抓取位姿     |
    | `left/right_lift` | 抓住后的左右试抬位姿    |
    | `left/right_retreat` | 带物撤离位姿          |
    | `grasp_context`  | 规划物体—夹爪关系与姿态上下文 |
    | `confidence`     | 抓取目标几何的可信度    |
    | `valid`          | 当前抓取目标是否可用    |
    | `failure_reason` | 生成失败原因，如目标不可达 |


    典型生产者是根据 ``ObjectEstimate3D`` 生成抓取几何的规划器，典型消费者是 IK 和
    轨迹规划。它是规划目标，不是机械臂实际状态或已抓住物体的证明。

    所有位姿均应位于机器人基座坐标系，长度单位米。物体中心不等于夹爪末端；规划
    必须结合抓取方向和双臂间距。规划未实现或目标不可达时 ``valid=False``。
    """

    left_pregrasp: Optional[Pose3D]
    right_pregrasp: Optional[Pose3D]
    left_grasp: Optional[Pose3D]
    right_grasp: Optional[Pose3D]
    left_lift: Optional[Pose3D]
    right_lift: Optional[Pose3D]
    left_retreat: Optional[Pose3D]
    right_retreat: Optional[Pose3D]
    grasp_context: Optional[GraspContext]
    confidence: float
    valid: bool = True
    failure_reason: str = ""

    def __post_init__(self) -> None:
        confidence = _strict_optional_finite(
            self.confidence, "GraspTarget.confidence", nonnegative=True
        )
        if confidence is None or confidence > 1.0:
            raise ValueError("GraspTarget.confidence 必须位于 0 到 1")
        object.__setattr__(self, "confidence", confidence)
        _strict_bool(self.valid, "GraspTarget.valid")
        pose_names = (
            "left_pregrasp", "right_pregrasp", "left_grasp", "right_grasp",
            "left_lift", "right_lift", "left_retreat", "right_retreat",
        )
        if self.valid:
            if self.failure_reason:
                raise ValueError("有效 GraspTarget 不得携带 failure_reason")
            for name in pose_names:
                _require_frozen_pose(getattr(self, name), f"GraspTarget.{name}")
            if not isinstance(self.grasp_context, GraspContext) or not self.grasp_context.valid:
                raise ValueError("有效 GraspTarget 必须提供有效 planned GraspContext")
            if self.grasp_context.confirmed:
                raise ValueError("ArmPlanner 产生的 GraspTarget context 不得标记 confirmed")
        else:
            if not isinstance(self.failure_reason, str) or not self.failure_reason.strip():
                raise ValueError("无效 GraspTarget 必须提供 failure_reason")
            if any(getattr(self, name) is not None for name in pose_names):
                raise ValueError("无效 GraspTarget 不得携带伪 Pose3D")
            if self.grasp_context is not None:
                raise ValueError("无效 GraspTarget 不得携带 GraspContext")


@dataclass(frozen=True)
class PlaceTarget:
    """双臂放置位姿集合。
    | 字段                | 含义            |
    | ----------------- | ------------- |
    | `object_goal_pose` | 任务要求的物体最终中心和方向 |
    | `left_preplace`   | 左夹爪预放置位姿      |
    | `right_preplace`  | 右夹爪预放置位姿      |
    | `left_release`    | 左夹爪释放物体时的位姿   |
    | `right_release`   | 右夹爪释放物体时的位姿   |
    | `left/right_post_release_retreat` | 释放后的撤离位姿 |
    | `settle_time_s`   | 放下后等待物体稳定的时间  |
    | `valid`           | 放置目标是否有效      |
    | `failure_reason`  | 放置规划失败原因      |


    典型生产者是把 ``TaskSpec.place_world_xyz`` 转成双臂几何目标的规划器，典型
    消费者是 IK 和轨迹规划。``object_goal_pose.position_xyz`` 是任务要求的物体中心；
    左右手 ``preplace``/``release`` 才是机械臂末端 ``Pose3D``。二者必须经过规划
    转换，不能直接混用；规划失败时 ``valid=False``。
    """

    object_goal_pose: Optional[Pose3D]
    left_preplace: Optional[Pose3D]
    right_preplace: Optional[Pose3D]
    left_release: Optional[Pose3D]
    right_release: Optional[Pose3D]
    left_post_release_retreat: Optional[Pose3D]
    right_post_release_retreat: Optional[Pose3D]
    settle_time_s: Optional[float]
    valid: bool = True
    failure_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "settle_time_s",
            _strict_optional_finite(
                self.settle_time_s, "PlaceTarget.settle_time_s", nonnegative=True
            ),
        )
        _strict_bool(self.valid, "PlaceTarget.valid")
        pose_names = (
            "object_goal_pose", "left_preplace", "right_preplace",
            "left_release", "right_release", "left_post_release_retreat",
            "right_post_release_retreat",
        )
        if self.valid:
            if self.failure_reason:
                raise ValueError("有效 PlaceTarget 不得携带 failure_reason")
            for name in pose_names:
                _require_frozen_pose(getattr(self, name), f"PlaceTarget.{name}")
            if self.settle_time_s is None:
                raise ValueError("有效 PlaceTarget 必须提供 settle_time_s")
        else:
            if not isinstance(self.failure_reason, str) or not self.failure_reason.strip():
                raise ValueError("无效 PlaceTarget 必须提供 failure_reason")
            if any(getattr(self, name) is not None for name in pose_names):
                raise ValueError("无效 PlaceTarget 不得携带伪 Pose3D")
            if self.settle_time_s is not None:
                raise ValueError("无效 PlaceTarget.settle_time_s 必须为 None")


@dataclass(frozen=True)
class IKResult:
    """官方 KDL 返回的目标关节解。
    | 字段                   | 含义                     |
    | -------------------- | ---------------------- |
    | `target_slide`       | 升降柱目标位置，通常单位米          |
    | `left_joint_target`  | 左臂6个关节目标；无解时可以是 `None` |
    | `right_joint_target` | 右臂6个关节目标；无解时可以是 `None` |
    | `success`            | IK是否找到可用解              |
    | `failure_reason`     | 无解原因，如目标超出可达范围         |


    典型生产者是 IK/KDL 适配器，典型消费者是轨迹规划器。它回答“希望关节去哪里”，
    不表示机器人现在的位置，也不表示该目标已经执行。

    ``target_slide`` 单位米，左右数组单位弧度。它们是目标解，不是
    ``/joint_states`` 的实际反馈。无解、frame 不符或依赖缺失时 ``success=False``。
    """

    target_slide: float
    left_joint_target: Optional[tuple[float, ...]]
    right_joint_target: Optional[tuple[float, ...]]
    success: bool
    failure_reason: str = ""


@dataclass(frozen=True)
class JointWaypoint:
    """一条 17 维非底盘关节路点。
    | 字段                  | 含义                  |
    | ------------------- | ------------------- |
    | `time_from_start_s` | 从轨迹开始到该路点的时间，单位秒    |
    | `joint_position`    | 该时刻计划到达的17维关节位置     |
    | `controlled_mask`   | 17个布尔值，表示哪些关节由该路点控制 |
    | `phase`             | 独立规划区段，不是执行器实时状态       |


    典型生产者是轨迹规划器，典型消费者是机械臂执行器。它是完整轨迹中的一个计划
    采样点，不是单独的机器人反馈。

    ``time_from_start_s`` 单位秒；位置顺序与 ``JOINT_NAMES`` 一致，mask 为真才由该
    路点控制且至少一项必须为真；全False不能表示等待、停止或阶段标签。长度、数值或
    mask错误时抛出 ``ValueError``。
    """

    phase: ArmMotionPhase
    time_from_start_s: float
    joint_position: tuple[float, ...]
    controlled_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.phase, ArmMotionPhase):
            raise ValueError("JointWaypoint.phase 必须是 ArmMotionPhase")
        time_value = _strict_optional_finite(
            self.time_from_start_s,
            "JointWaypoint.time_from_start_s",
            nonnegative=True,
        )
        assert time_value is not None
        object.__setattr__(self, "time_from_start_s", time_value)
        object.__setattr__(
            self,
            "joint_position",
            _strict_finite_vector(self.joint_position, 17, "路点关节位置"),
        )
        if len(self.controlled_mask) != 17:
            raise ValueError("controlled_mask 必须包含 17 项")
        if any(type(value) is not bool for value in self.controlled_mask):
            raise ValueError("controlled_mask 每项都必须是严格 bool")
        if not any(self.controlled_mask):
            raise ValueError("有效 JointWaypoint.controlled_mask 至少一项必须为 True")
        object.__setattr__(self, "controlled_mask", tuple(self.controlled_mask))


@dataclass(frozen=True)
class JointTrajectory:
    """按时间排列的完整关节轨迹计划。
    | 字段               | 含义                       |
    | ---------------- | ------------------------ |
    | `trajectory_id`  | 轨迹唯一编号                   |
    | `task_id`        | 轨迹绑定的任务编号                |
    | `target_body`    | 轨迹绑定的任务目标body             |
    | `execution_phase`| 仅允许抓取或放置执行全局阶段          |
    | `waypoints`      | 按时间排列的多个 `JointWaypoint` |
    | `timestamp_ns`   | 轨迹生成时间                   |
    | `valid`          | 整条轨迹是否有效                 |
    | `failure_reason` | 轨迹无效原因，如路点为空、规划失败        |


    典型生产者是机械臂规划器，典型消费者是机械臂执行器。轨迹像整条路线，
    ``ManipulationCommand`` 像本周期走到哪一步；轨迹存在不代表已经开始或完成执行。

    ``timestamp_ns`` 为规划生成时间；轨迹只描述目标，不表示机械臂已经执行成功。
    有效抓取/放置轨迹分别要求各自四个完整有序阶段，同阶段可含多个路点；无效轨迹
    必须为空且带失败原因，可以不伪造 ``trajectory_id`` 或 ``target_body``。
    """

    trajectory_id: str
    task_id: int
    target_body: str
    execution_phase: GlobalPhase
    waypoints: tuple[JointWaypoint, ...]
    timestamp_ns: int
    valid: bool = True
    failure_reason: str = ""

    def __post_init__(self) -> None:
        _strict_bool(self.valid, "JointTrajectory.valid")
        _strict_nonnegative_ns(self.timestamp_ns, "JointTrajectory.timestamp_ns")
        if type(self.task_id) is not int or self.task_id < 0:
            raise ValueError("JointTrajectory.task_id 必须是非负整数且不能是 bool")
        if not isinstance(self.execution_phase, GlobalPhase):
            raise ValueError("JointTrajectory.execution_phase 必须严格使用 GlobalPhase")
        if self.execution_phase not in {
            GlobalPhase.EXECUTE_PICK, GlobalPhase.EXECUTE_PLACE
        }:
            raise ValueError("JointTrajectory.execution_phase 只能是 EXECUTE_PICK 或 EXECUTE_PLACE")
        if not isinstance(self.trajectory_id, str):
            raise ValueError("JointTrajectory.trajectory_id 必须是字符串")
        if not isinstance(self.target_body, str):
            raise ValueError("JointTrajectory.target_body 必须是字符串")
        try:
            waypoints = tuple(self.waypoints)
        except (TypeError, ValueError) as exc:
            raise ValueError("JointTrajectory.waypoints 必须可迭代") from exc
        object.__setattr__(self, "waypoints", waypoints)
        if not self.valid:
            if not isinstance(self.failure_reason, str) or not self.failure_reason.strip():
                raise ValueError("无效 JointTrajectory 必须提供 failure_reason")
            if waypoints:
                raise ValueError("无效 JointTrajectory.waypoints 必须为空")
            return
        if not self.trajectory_id.strip():
            raise ValueError("有效 JointTrajectory.trajectory_id 必须是非空字符串")
        if not self.target_body.strip():
            raise ValueError("有效 JointTrajectory.target_body 必须是非空字符串")
        if self.failure_reason:
            raise ValueError("有效 JointTrajectory 不得携带 failure_reason")
        if not waypoints:
            raise ValueError("有效 JointTrajectory.waypoints 不能为空")

        required_phases = (
            (
                ArmMotionPhase.PREGRASP,
                ArmMotionPhase.GRASP,
                ArmMotionPhase.LIFT,
                ArmMotionPhase.RETREAT,
            )
            if self.execution_phase is GlobalPhase.EXECUTE_PICK
            else (
                ArmMotionPhase.PREPLACE,
                ArmMotionPhase.LOWER,
                ArmMotionPhase.RELEASE,
                ArmMotionPhase.POST_RELEASE_RETREAT,
            )
        )
        phase_indices = {phase: index for index, phase in enumerate(required_phases)}
        previous_time: Optional[float] = None
        previous_phase_index = 0
        observed_phases: set[ArmMotionPhase] = set()
        for waypoint in waypoints:
            if not isinstance(waypoint, JointWaypoint):
                raise ValueError("JointTrajectory.waypoints 每项必须是 JointWaypoint")
            if not isinstance(waypoint.phase, ArmMotionPhase):
                raise ValueError("JointTrajectory.waypoint.phase 必须严格使用 ArmMotionPhase")
            if previous_time is not None and waypoint.time_from_start_s <= previous_time:
                raise ValueError("JointTrajectory 路点时间必须严格递增")
            previous_time = waypoint.time_from_start_s
            if waypoint.phase not in phase_indices:
                raise ValueError(
                    f"{self.execution_phase.value} 轨迹不允许阶段 {waypoint.phase.value}"
                )
            phase_index = phase_indices[waypoint.phase]
            if phase_index < previous_phase_index:
                raise ValueError("JointTrajectory.phase 只能按规定顺序前进，不得倒退")
            previous_phase_index = phase_index
            observed_phases.add(waypoint.phase)
        missing = tuple(phase.value for phase in required_phases if phase not in observed_phases)
        if missing:
            raise ValueError(f"有效 JointTrajectory 缺少必要阶段：{missing}")


# 机械臂执行与验证接口

@dataclass(frozen=True)
class ManipulationCommand:
    """机械臂执行器提交给 ActionMux 的候选命令。
    | 字段                | 含义                |
    | ----------------- | ----------------- |
    | `joint_target`    | 当前周期建议的17维关节目标    |
    | `controlled_mask` | 哪些关节使用候选，哪些保持实际位置 |
    | `local_phase`     | 当前机械臂处于哪个局部动作阶段   |
    | `timestamp_ns`    | 候选命令生成时间          |
    | `valid_until_ns`  | 候选命令失效时间          |
    | `valid`           | 候选命令是否有效          |
    | `failure_reason`  | 无效原因，如轨迹无效或插值失败   |


    典型生产者是按 ``JointTrajectory`` 取样的机械臂执行器，典型消费者是 ActionMux。
    它只表示本控制周期建议的 17 维目标；轨迹已规划或命令已生成都不等于实际执行。

    ``joint_target`` 为 17 维位置目标，单位同 ``RobotJointState``；mask 为真才覆盖
    安全保持值。命令过期后必须保持实际/最近安全关节位置，而不是写成全零。
    """

    joint_target: tuple[float, ...]
    controlled_mask: tuple[bool, ...]
    local_phase: LocalPhase
    timestamp_ns: int
    valid_until_ns: int
    valid: bool = True
    failure_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_target", ensure_finite_vector(self.joint_target, 17, "机械臂命令"))
        if len(self.controlled_mask) != 17:
            raise ValueError("机械臂 controlled_mask 必须包含 17 项")


@dataclass(frozen=True)
class ManipulationStatus:
    """机械臂局部执行状态。
    | 字段                | 含义                |
    | ----------------- | ----------------- |
    | `local_phase`     | 当前机械臂执行到哪个子阶段     |
    | `state`           | 执行状态，如运行、等待、超时、失败 |
    | `progress`        | 执行进度，通常建议0～1      |
    | `max_joint_error` | 当前目标与实际关节之间的最大误差  |
    | `success`         | 是否根据实际关节反馈确认完成    |
    | `failure_reason`  | 执行失败或未完成原因        |
    | `timestamp_ns`    | 状态计算时间            |



    典型生产者是结合实际 ``RobotJointState`` 判断误差和进度的执行器，典型消费者是
    FSM。它是基于反馈的执行评估，不能由目标轨迹或候选命令直接充当。

    ``max_joint_error`` 单位为对应关节单位，``progress`` 建议范围 0～1。只有实际反馈
    达标并完成验证后才能 ``success=True``；未实现、超时和轨迹错误都需说明原因。

    当前骨架状态：该接口已预留，机械臂执行闭环尚未完成。
    """

    local_phase: LocalPhase
    state: str
    progress: float
    max_joint_error: float
    success: bool
    failure_reason: str
    timestamp_ns: int


@dataclass(frozen=True)
class GraspVerification:
    """试抬后的抓取验证结果。
    | 字段                | 含义              |
    | ----------------- | --------------- |
    | `is_grasped`      | 当前是否判断物体已被抓住    |
    | `confidence`      | 抓取判断可信度         |
    | `visual_evidence` | 视觉证据摘要，如物体随夹爪移动 |
    | `effort_evidence` | 关节受力或夹爪反馈证据摘要   |
    | `success`         | 本次抓取验证过程是否成功完成  |
    | `failure_reason`  | 验证失败原因，如图像不可用   |
    | `timestamp_ns`    | 验证时间            |


    典型生产者是融合视觉、关节 effort 等实际观测的验证逻辑，典型消费者是 FSM。
    它不是抓取命令，也不是裁判最终结果。

    置信度建议为 0～1，视觉和 effort 证据为可读摘要。该结果来自机器人观测，不得用
    裁判最终 JSON 伪装成逐帧抓取真值。证据不足时 ``success=False``。

    当前骨架状态：该接口已预留，抓取验证闭环尚未完成。
    """

    is_grasped: bool
    confidence: float
    visual_evidence: str
    effort_evidence: str
    success: bool
    failure_reason: str
    timestamp_ns: int


# FSM与最终动作

# ————————————————————————————————
# 【Codex修改-16：扩充FSMStatus原因字段文档】
# 1. 修改前文档只称failure_reason保存失败原因，遗漏SAFE_HOLD暂停诊断语义。
# 2. 当前明确该字段也可以保存SAFE_HOLD的安全暂停原因。
# 3. 这样遥测、ActionMux和Recorder能按同一语义解释非终态安全原因。
# 4. 仅修改FSMStatus docstring，不改变字段、顺序、JSON schema或运行逻辑。
@dataclass(frozen=True)
class FSMStatus:
    """全局和局部状态机遥测。
    | 字段               | 含义             |
    | ---------------- | -------------- |
    | `task_id`        | 当前对应的任务编号      |
    | `global_phase`   | 整个任务处于哪个全局阶段   |
    | `local_phase`    | 机械臂处于哪个局部阶段    |
    | `retry_count`    | 当前已经重试多少次      |
    | `success`        | 客户端状态机是否认为任务完成 |
    | `failure_reason` | 当前失败原因         |
    | `timestamp_ns`   | 状态快照时间         |


    典型生产者是 FSM，典型消费者是 ActionMux、遥测和 Recorder。它描述客户端的
    全局/局部阶段；``DONE`` 或 ``success=True`` 只表达客户端状态机的完成判断，不
    自动等于裁判结算或得分，裁判结果也不应写进逐帧状态语义。

    时间单位纳秒，状态本身不带空间坐标。``failure_reason`` 可以记录客户端失败原因，
    也可以记录 ``SAFE_HOLD`` 的安全暂停原因；非法状态或解析失败时 JSON 解码函数
    抛出 ``ValueError``。
    """
    # ————————————————————————————————

    task_id: int
    global_phase: GlobalPhase
    local_phase: LocalPhase
    retry_count: int
    success: bool
    failure_reason: str
    timestamp_ns: int


@dataclass(frozen=True)
class FinalAction:
    """ActionMux 每个控制周期唯一生成的 19 维最终动作。
    | 字段               | 含义                 |
    | ---------------- | ------------------ |
    | `values`         | 固定19维最终动作          |
    | `sequence`       | ActionMux生成动作的递增序号 |
    | `timestamp_ns`   | 最终动作生成时间           |
    | `global_phase`   | 生成动作时的全局任务阶段       |
    | `local_phase`    | 生成动作时的机械臂局部阶段      |
    | `valid`          | 是否满足进入发布链路的条件      |
    | `clipped`        | 是否有合法候选因超出边界而被限幅   |
    | `failure_reason` | 安全降级、过期、无效反馈等原因    |


    典型生产者只能是 ActionMux，典型消费者是 ``OfficialCommandPublisher``、遥测和
    Recorder。顺序严格采用 ``ACTION_NAMES``：前两项分别为底盘 m/s 和 rad/s，其余
    为 slide 米、关节弧度和夹爪 0～1；业务模块不得自行拼接另一套动作数组。

    ``valid=False`` 的对象可以记录用于诊断，但不能直接作为专家训练动作；
    ``valid=True`` 只表示通过客户端结构和安全检查。进入 OfficialCommandPublisher
    发布链路后才是候选实发动作；Server 是否接收、控制器是否接受以及机器人是否
    实际执行，仍需结合 JointState、Odom 等反馈确认，不能用 publish 调用成功代替。
    长度不为 19 或含 NaN/Inf 时构造立即失败，防止危险值进入 ROS2。
    """

    values: tuple[float, ...]
    sequence: int
    timestamp_ns: int
    global_phase: GlobalPhase
    local_phase: LocalPhase
    valid: bool
    clipped: bool
    failure_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", ensure_finite_vector(self.values, 19, "FinalAction"))


class CandidateDisposition(str, Enum):
    """ActionMux 对一类候选的受控处置结果，不依赖 failure_reason 文本推断。"""

    ABSENT = "absent"
    ACCEPTED = "accepted"
    REJECTED_INVALID = "rejected_invalid"
    REJECTED_STALE = "rejected_stale"
    SAFETY_OVERRIDDEN = "safety_overridden"
    PARTIALLY_ACCEPTED = "partially_accepted"


class DispatchMode(str, Enum):
    """本周期官方发布路径；不表达 Server 接收或机器人执行。"""

    NONE = "none"
    HEAD_ONLY = "head_only"
    FULL = "full"


def _strict_bool_tuple(values: Sequence[bool], length: int, label: str) -> tuple[bool, ...]:
    result = tuple(values)
    if len(result) != length or any(type(value) is not bool for value in result):
        raise ValueError(f"{label} 必须包含 {length} 项严格 bool")
    return result


def _strict_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负整数且不能是 bool")
    return value


def _strict_optional_float_tuple(
    values: Sequence[Optional[float]], length: int, label: str
) -> tuple[Optional[float], ...]:
    result = tuple(values)
    if len(result) != length:
        raise ValueError(f"{label} 必须包含 {length} 项")
    normalized: list[Optional[float]] = []
    for index, value in enumerate(result):
        if value is None:
            normalized.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}[{index}] 必须是有限数或 null，不能是 bool")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] 必须是有限数或 null")
        normalized.append(number)
    return tuple(normalized)


@dataclass(frozen=True)
class ActionMuxDecision:
    """与一个 FinalAction 稳定关联的逐维候选仲裁事实。"""

    schema_name: str
    schema_version: int
    sequence: int
    timestamp_ns: int
    final_action_sequence: int
    requested_mask: tuple[bool, ...]
    commanded_mask: tuple[bool, ...]
    clipped_mask: tuple[bool, ...]
    safety_override_mask: tuple[bool, ...]
    base_candidate_present: bool
    manipulation_candidate_present: bool
    base_disposition: CandidateDisposition
    manipulation_disposition: CandidateDisposition
    base_source: str
    manipulation_source: str
    global_phase: GlobalPhase
    local_phase: LocalPhase
    valid: bool
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if (
            self.schema_name != "MMK2ActionMuxDecision"
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("ActionMuxDecision schema 必须为 MMK2ActionMuxDecision V1")
        _strict_nonnegative_int(self.sequence, "ActionMuxDecision.sequence")
        _strict_nonnegative_int(self.timestamp_ns, "ActionMuxDecision.timestamp_ns")
        _strict_nonnegative_int(
            self.final_action_sequence, "ActionMuxDecision.final_action_sequence"
        )
        if self.sequence != self.final_action_sequence:
            raise ValueError("ActionMuxDecision 必须与 FinalAction sequence 稳定关联")
        requested = _strict_bool_tuple(self.requested_mask, 19, "requested_mask")
        commanded = _strict_bool_tuple(self.commanded_mask, 19, "commanded_mask")
        clipped = _strict_bool_tuple(self.clipped_mask, 19, "clipped_mask")
        overridden = _strict_bool_tuple(
            self.safety_override_mask, 19, "safety_override_mask"
        )
        object.__setattr__(self, "requested_mask", requested)
        object.__setattr__(self, "commanded_mask", commanded)
        object.__setattr__(self, "clipped_mask", clipped)
        object.__setattr__(self, "safety_override_mask", overridden)
        if any(commanded[index] and not requested[index] for index in range(19)):
            raise ValueError("commanded_mask 不能包含 requested_mask 之外的维度")
        if any(clipped[index] and not commanded[index] for index in range(19)):
            raise ValueError("clipped_mask 只能标记已接受的业务命令")
        if any(commanded[index] and overridden[index] for index in range(19)):
            raise ValueError("commanded_mask 与 safety_override_mask 不能重叠")
        for label, value in (
            ("base_candidate_present", self.base_candidate_present),
            ("manipulation_candidate_present", self.manipulation_candidate_present),
            ("valid", self.valid),
        ):
            if type(value) is not bool:
                raise ValueError(f"{label} 必须是严格 bool")
        if not isinstance(self.base_disposition, CandidateDisposition) or not isinstance(
            self.manipulation_disposition, CandidateDisposition
        ):
            raise ValueError("候选 disposition 必须使用 CandidateDisposition")
        if not isinstance(self.global_phase, GlobalPhase) or not isinstance(
            self.local_phase, LocalPhase
        ):
            raise ValueError("ActionMuxDecision phase 必须使用受控枚举")
        if not isinstance(self.failure_reason, str):
            raise ValueError("ActionMuxDecision.failure_reason 必须是字符串")
        if requested[:2] != (self.base_candidate_present,) * 2:
            raise ValueError("requested_mask 的 base 维度必须反映 BaseCommand 是否存在")
        if not self.manipulation_candidate_present and any(requested[2:]):
            raise ValueError("无 ManipulationCommand 时不得请求位置维度")
        allowed_sources = {
            "none",
            "base_command",
            "manipulation_command",
            "external_candidate",
        }
        if (
            not isinstance(self.base_source, str)
            or not isinstance(self.manipulation_source, str)
            or self.base_source not in allowed_sources
            or self.manipulation_source not in allowed_sources
        ):
            raise ValueError("ActionMuxDecision 候选来源不是受控标签")
        if not self.base_candidate_present and self.base_source != "none":
            raise ValueError("无 BaseCommand 时 base_source 必须为 none")
        if not self.manipulation_candidate_present and self.manipulation_source != "none":
            raise ValueError("无 ManipulationCommand 时 manipulation_source 必须为 none")

        rejected = {
            CandidateDisposition.REJECTED_INVALID,
            CandidateDisposition.REJECTED_STALE,
            CandidateDisposition.SAFETY_OVERRIDDEN,
        }
        if not self.base_candidate_present:
            if self.base_disposition is not CandidateDisposition.ABSENT:
                raise ValueError("无 BaseCommand 时 base_disposition 必须为 absent")
            if any((*commanded[:2], *clipped[:2])):
                raise ValueError("无 BaseCommand 时 base commanded/clipped 必须全 false")
        else:
            if self.base_source != "base_command":
                raise ValueError("存在 BaseCommand 时 base_source 必须为 base_command")
            if self.base_disposition is CandidateDisposition.ABSENT:
                raise ValueError("存在 BaseCommand 时 base_disposition 不能为 absent")
        if self.base_disposition is CandidateDisposition.ACCEPTED and commanded[:2] != (
            True,
            True,
        ):
            raise ValueError("accepted BaseCommand 必须 commanded 两个 base 维度")
        if self.base_disposition in rejected and any((*commanded[:2], *clipped[:2])):
            raise ValueError("被拒绝或安全覆盖的 BaseCommand 不得 commanded/clipped")
        if self.base_disposition is CandidateDisposition.PARTIALLY_ACCEPTED:
            accepted_count = sum(commanded[:2])
            if accepted_count <= 0 or accepted_count >= sum(requested[:2]):
                raise ValueError("partially_accepted BaseCommand 必须只接受部分 requested 维度")

        if not self.manipulation_candidate_present:
            if self.manipulation_disposition is not CandidateDisposition.ABSENT:
                raise ValueError(
                    "无 ManipulationCommand 时 manipulation_disposition 必须为 absent"
                )
            if any((*commanded[2:], *clipped[2:])):
                raise ValueError(
                    "无 ManipulationCommand 时 manipulation commanded/clipped 必须全 false"
                )
        else:
            if self.manipulation_source not in {
                "manipulation_command",
                "external_candidate",
            }:
                raise ValueError(
                    "存在 ManipulationCommand 时 manipulation_source 无效"
                )
            if self.manipulation_disposition is CandidateDisposition.ABSENT:
                raise ValueError(
                    "存在 ManipulationCommand 时 manipulation_disposition 不能为 absent"
                )
        if (
            self.manipulation_disposition is CandidateDisposition.ACCEPTED
            and commanded[2:] != requested[2:]
        ):
            raise ValueError(
                "accepted ManipulationCommand 的 commanded 必须等于 requested"
            )
        if self.manipulation_disposition in rejected and any(
            (*commanded[2:], *clipped[2:])
        ):
            raise ValueError(
                "被拒绝或安全覆盖的 ManipulationCommand 不得 commanded/clipped"
            )
        if self.manipulation_disposition is CandidateDisposition.PARTIALLY_ACCEPTED:
            requested_count = sum(requested[2:])
            accepted_count = sum(commanded[2:])
            if accepted_count <= 0 or accepted_count >= requested_count:
                raise ValueError(
                    "partially_accepted ManipulationCommand 必须只接受部分 requested 维度"
                )


@dataclass(frozen=True)
class TwistExactPayload:
    """真正交给 Twist publisher 的六个分量。"""

    linear_xyz: tuple[float, float, float]
    angular_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) for value in (*self.linear_xyz, *self.angular_xyz)):
            raise ValueError("Twist exact_payload 不能用 bool 冒充数值")
        object.__setattr__(self, "linear_xyz", ensure_finite_vector(self.linear_xyz, 3, "Twist.linear"))
        object.__setattr__(self, "angular_xyz", ensure_finite_vector(self.angular_xyz, 3, "Twist.angular"))


@dataclass(frozen=True)
class Float64MultiArrayExactPayload:
    """真正交给 Float64MultiArray publisher 的 data。"""

    data: tuple[float, ...]

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) for value in self.data):
            raise ValueError("Float64MultiArray exact_payload 不能用 bool 冒充数值")
        object.__setattr__(self, "data", ensure_finite_vector(self.data, len(self.data), "Float64MultiArray.data"))


@dataclass(frozen=True)
class DispatchGroupRecord:
    """一个官方话题在本周期的精确本地 publisher 调用事实。"""

    group: str
    official_topic: str
    message_type: str
    attempted: bool
    succeeded: Optional[bool]
    exact_payload: Optional[TwistExactPayload | Float64MultiArrayExactPayload]
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str)
            for value in (self.group, self.official_topic, self.message_type, self.failure_reason)
        ):
            raise ValueError("DispatchGroupRecord 文本字段必须是字符串")
        if type(self.attempted) is not bool:
            raise ValueError("DispatchGroupRecord.attempted 必须是严格 bool")
        if self.succeeded is not None and type(self.succeeded) is not bool:
            raise ValueError("DispatchGroupRecord.succeeded 必须是 bool 或 null")
        if not self.attempted and (self.succeeded is not None or self.exact_payload is not None):
            raise ValueError("未尝试分组的 succeeded 和 exact_payload 必须为 null")
        if self.attempted and self.succeeded is None:
            raise ValueError("已尝试分组必须明确 succeeded")
        if self.attempted and self.exact_payload is None:
            raise ValueError("已尝试分组必须记录 exact_payload")
        if self.succeeded is True and self.failure_reason:
            raise ValueError("成功分组不得携带 failure_reason")


@dataclass(frozen=True)
class ActionDispatchRecord:
    """一个已计算 FinalAction 的逐组本地 dispatch 遥测，不是执行确认。"""

    schema_name: str
    schema_version: int
    sequence: int
    timestamp_ns: int
    final_action_sequence: int
    decision: ActionMuxDecision
    calculated: bool
    publish_enabled: bool
    publisher_created: bool
    publish_attempted: bool
    publisher_call_succeeded: Optional[bool]
    dispatch_mode: DispatchMode
    dispatched_action: tuple[Optional[float], ...]
    dispatched_mask: tuple[bool, ...]
    attempted_groups: tuple[str, ...]
    successful_groups: tuple[str, ...]
    failed_groups: tuple[str, ...]
    group_records: tuple[DispatchGroupRecord, ...]
    controller_accepted: None
    execution_confirmed: None
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if (
            self.schema_name != "MMK2ActionDispatchRecord"
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("ActionDispatchRecord schema 必须为 MMK2ActionDispatchRecord V1")
        _strict_nonnegative_int(self.sequence, "ActionDispatchRecord.sequence")
        _strict_nonnegative_int(self.timestamp_ns, "ActionDispatchRecord.timestamp_ns")
        _strict_nonnegative_int(
            self.final_action_sequence, "ActionDispatchRecord.final_action_sequence"
        )
        if not isinstance(self.decision, ActionMuxDecision):
            raise ValueError("ActionDispatchRecord.decision 必须是 ActionMuxDecision")
        if self.sequence != self.final_action_sequence or self.decision.final_action_sequence != self.final_action_sequence:
            raise ValueError("ActionDispatchRecord 必须与 Decision/FinalAction sequence 稳定关联")
        if self.timestamp_ns != self.decision.timestamp_ns:
            raise ValueError("ActionDispatchRecord 与 ActionMuxDecision timestamp_ns 必须一致")
        if not isinstance(self.failure_reason, str):
            raise ValueError("ActionDispatchRecord.failure_reason 必须是字符串")
        for label, value in (
            ("calculated", self.calculated),
            ("publish_enabled", self.publish_enabled),
            ("publisher_created", self.publisher_created),
            ("publish_attempted", self.publish_attempted),
        ):
            if type(value) is not bool:
                raise ValueError(f"{label} 必须是严格 bool")
        if not self.calculated:
            raise ValueError("ActionDispatchRecord V1 只描述已计算 FinalAction")
        if not isinstance(self.dispatch_mode, DispatchMode):
            raise ValueError("dispatch_mode 必须使用 DispatchMode")
        if self.publisher_call_succeeded is not None and type(self.publisher_call_succeeded) is not bool:
            raise ValueError("publisher_call_succeeded 必须是 bool 或 null")
        group_records = tuple(self.group_records)
        if any(not isinstance(record, DispatchGroupRecord) for record in group_records):
            raise ValueError("group_records 必须只包含 DispatchGroupRecord")
        object.__setattr__(self, "group_records", group_records)
        for field_name in ("attempted_groups", "successful_groups", "failed_groups"):
            groups = tuple(getattr(self, field_name))
            if any(not isinstance(group, str) for group in groups):
                raise ValueError(f"{field_name} 必须只包含字符串")
            object.__setattr__(self, field_name, groups)
        action = _strict_optional_float_tuple(
            self.dispatched_action, 19, "dispatched_action"
        )
        mask = _strict_bool_tuple(self.dispatched_mask, 19, "dispatched_mask")
        object.__setattr__(self, "dispatched_action", action)
        object.__setattr__(self, "dispatched_mask", mask)
        if any((value is not None) != mask[index] for index, value in enumerate(action)):
            raise ValueError("dispatched_action 的 null 位置必须与 dispatched_mask 一致")
        expected_groups = (
            ("base", "/cmd_vel", "geometry_msgs/msg/Twist", 0, 2),
            (
                "spine",
                "/spine_forward_position_controller/commands",
                "std_msgs/msg/Float64MultiArray",
                2,
                3,
            ),
            (
                "head",
                "/head_forward_position_controller/commands",
                "std_msgs/msg/Float64MultiArray",
                3,
                5,
            ),
            (
                "left_arm",
                "/left_arm_forward_position_controller/commands",
                "std_msgs/msg/Float64MultiArray",
                5,
                12,
            ),
            (
                "right_arm",
                "/right_arm_forward_position_controller/commands",
                "std_msgs/msg/Float64MultiArray",
                12,
                19,
            ),
        )
        if len(self.group_records) != len(expected_groups):
            raise ValueError("group_records 必须严格包含五个官方分组")
        reconstructed: list[Optional[float]] = [None] * 19
        for record, (group, topic, message_type, start, stop) in zip(
            self.group_records, expected_groups
        ):
            if (record.group, record.official_topic, record.message_type) != (
                group,
                topic,
                message_type,
            ):
                raise ValueError("group_records 顺序、话题或消息类型不符合官方映射")
            if not record.attempted:
                continue
            if group == "base":
                if not isinstance(record.exact_payload, TwistExactPayload):
                    raise ValueError("base exact_payload 必须是完整 Twist")
                reconstructed[0] = record.exact_payload.linear_xyz[0]
                reconstructed[1] = record.exact_payload.angular_xyz[2]
            else:
                if not isinstance(record.exact_payload, Float64MultiArrayExactPayload):
                    raise ValueError(f"{group} exact_payload 必须是 Float64MultiArray data")
                if len(record.exact_payload.data) != stop - start:
                    raise ValueError(f"{group} exact_payload 长度错误")
                reconstructed[start:stop] = record.exact_payload.data
        if tuple(reconstructed) != action:
            raise ValueError("dispatched_action 必须精确派生自逐组 exact_payload")
        attempted = tuple(record.group for record in self.group_records if record.attempted)
        successful = tuple(record.group for record in self.group_records if record.succeeded is True)
        failed = tuple(record.group for record in self.group_records if record.succeeded is False)
        if self.attempted_groups != attempted or self.successful_groups != successful or self.failed_groups != failed:
            raise ValueError("分组摘要必须与 group_records 完全一致")
        if self.publish_attempted != bool(attempted):
            raise ValueError("publish_attempted 必须与实际 attempted_groups 一致")
        expected_call_result = None if not attempted else not failed
        if self.publisher_call_succeeded is not expected_call_result:
            raise ValueError("publisher_call_succeeded 必须反映本地 publisher 调用结果")
        if self.dispatch_mode is DispatchMode.NONE and attempted:
            raise ValueError("none 模式不得尝试官方分组")
        if self.dispatch_mode is DispatchMode.HEAD_ONLY and any(
            group != "head" for group in attempted
        ):
            raise ValueError("head_only 模式只能尝试 head 分组")
        if self.dispatch_mode is DispatchMode.FULL:
            official_order = tuple(group[0] for group in expected_groups)
            if attempted != official_order[: len(attempted)]:
                raise ValueError("full 模式 attempted groups 必须是官方顺序的连续前缀")
            if len(failed) > 1:
                raise ValueError("full 模式最多允许一个失败组")
            if failed and failed[0] != attempted[-1]:
                raise ValueError("full 模式失败组必须是最后一个 attempted group")
        if self.publisher_created and not self.publish_enabled:
            raise ValueError("publisher_created 不能绕过 publish_enabled")
        if attempted and (not self.publish_enabled or not self.publisher_created):
            raise ValueError("官方 publisher 调用不能绕过创建门或发布门")
        if self.controller_accepted is not None or self.execution_confirmed is not None:
            raise ValueError("V1 无法确认 controller accepted 或 execution confirmed")


def _decision_to_dict(decision: ActionMuxDecision) -> dict[str, Any]:
    return {
        "schema_name": decision.schema_name,
        "schema_version": decision.schema_version,
        "sequence": decision.sequence,
        "timestamp_ns": decision.timestamp_ns,
        "final_action_sequence": decision.final_action_sequence,
        "requested_mask": list(decision.requested_mask),
        "commanded_mask": list(decision.commanded_mask),
        "clipped_mask": list(decision.clipped_mask),
        "safety_override_mask": list(decision.safety_override_mask),
        "base_candidate_present": decision.base_candidate_present,
        "manipulation_candidate_present": decision.manipulation_candidate_present,
        "base_disposition": decision.base_disposition.value,
        "manipulation_disposition": decision.manipulation_disposition.value,
        "base_source": decision.base_source,
        "manipulation_source": decision.manipulation_source,
        "global_phase": decision.global_phase.value,
        "local_phase": decision.local_phase.value,
        "valid": decision.valid,
        "failure_reason": decision.failure_reason,
    }


def _exact_payload_to_dict(
    payload: Optional[TwistExactPayload | Float64MultiArrayExactPayload],
) -> Optional[dict[str, Any]]:
    if payload is None:
        return None
    if isinstance(payload, TwistExactPayload):
        return {
            "linear": dict(zip(("x", "y", "z"), payload.linear_xyz)),
            "angular": dict(zip(("x", "y", "z"), payload.angular_xyz)),
        }
    return {"data": list(payload.data)}


def action_dispatch_to_json(record: ActionDispatchRecord) -> str:
    """严格序列化 ActionDispatchRecord V1，禁止 NaN/Inf。"""

    if not isinstance(record, ActionDispatchRecord):
        raise TypeError("action_dispatch_to_json 只接受 ActionDispatchRecord")
    payload = {
        "schema_name": record.schema_name,
        "schema_version": record.schema_version,
        "sequence": record.sequence,
        "timestamp_ns": record.timestamp_ns,
        "final_action_sequence": record.final_action_sequence,
        "decision": _decision_to_dict(record.decision),
        "calculated": record.calculated,
        "publish_enabled": record.publish_enabled,
        "publisher_created": record.publisher_created,
        "publish_attempted": record.publish_attempted,
        "publisher_call_succeeded": record.publisher_call_succeeded,
        "dispatch_mode": record.dispatch_mode.value,
        "dispatched_action": list(record.dispatched_action),
        "dispatched_mask": list(record.dispatched_mask),
        "attempted_groups": list(record.attempted_groups),
        "successful_groups": list(record.successful_groups),
        "failed_groups": list(record.failed_groups),
        "group_records": [
            {
                "group": group.group,
                "official_topic": group.official_topic,
                "message_type": group.message_type,
                "attempted": group.attempted,
                "succeeded": group.succeeded,
                "exact_payload": _exact_payload_to_dict(group.exact_payload),
                "failure_reason": group.failure_reason,
            }
            for group in record.group_records
        ],
        "controller_accepted": None,
        "execution_confirmed": None,
        "failure_reason": record.failure_reason,
    }
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _strict_json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"禁止非有限 JSON 数值 {value}")
            ),
        )
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} JSON 无效：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON 顶层必须是对象")
    return payload


def _require_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} 字段不匹配：missing={sorted(expected - set(payload))}, unknown={sorted(set(payload) - expected)}")


def _decision_from_dict(payload: object) -> ActionMuxDecision:
    if not isinstance(payload, dict):
        raise ValueError("decision 必须是对象")
    expected = {
        "schema_name", "schema_version", "sequence", "timestamp_ns",
        "final_action_sequence", "requested_mask", "commanded_mask", "clipped_mask",
        "safety_override_mask", "base_candidate_present", "manipulation_candidate_present",
        "base_disposition", "manipulation_disposition", "base_source",
        "manipulation_source", "global_phase", "local_phase", "valid", "failure_reason",
    }
    _require_keys(payload, expected, "decision")
    return ActionMuxDecision(
        schema_name=payload["schema_name"],
        schema_version=payload["schema_version"],
        sequence=payload["sequence"],
        timestamp_ns=payload["timestamp_ns"],
        final_action_sequence=payload["final_action_sequence"],
        requested_mask=tuple(payload["requested_mask"]),
        commanded_mask=tuple(payload["commanded_mask"]),
        clipped_mask=tuple(payload["clipped_mask"]),
        safety_override_mask=tuple(payload["safety_override_mask"]),
        base_candidate_present=payload["base_candidate_present"],
        manipulation_candidate_present=payload["manipulation_candidate_present"],
        base_disposition=CandidateDisposition(payload["base_disposition"]),
        manipulation_disposition=CandidateDisposition(payload["manipulation_disposition"]),
        base_source=payload["base_source"],
        manipulation_source=payload["manipulation_source"],
        global_phase=GlobalPhase(payload["global_phase"]),
        local_phase=LocalPhase(payload["local_phase"]),
        valid=payload["valid"],
        failure_reason=payload["failure_reason"],
    )


def _group_from_dict(payload: object) -> DispatchGroupRecord:
    if not isinstance(payload, dict):
        raise ValueError("group_record 必须是对象")
    expected = {
        "group", "official_topic", "message_type", "attempted", "succeeded",
        "exact_payload", "failure_reason",
    }
    _require_keys(payload, expected, "group_record")
    exact = payload["exact_payload"]
    parsed_exact: Optional[TwistExactPayload | Float64MultiArrayExactPayload]
    if exact is None:
        parsed_exact = None
    elif payload["message_type"] == "geometry_msgs/msg/Twist":
        if not isinstance(exact, dict) or set(exact) != {"linear", "angular"}:
            raise ValueError("Twist exact_payload 必须包含 linear/angular")
        for vector_name in ("linear", "angular"):
            if not isinstance(exact[vector_name], dict) or set(exact[vector_name]) != {"x", "y", "z"}:
                raise ValueError(f"Twist.{vector_name} 必须包含 x/y/z")
        parsed_exact = TwistExactPayload(
            tuple(exact["linear"][axis] for axis in ("x", "y", "z")),
            tuple(exact["angular"][axis] for axis in ("x", "y", "z")),
        )
    elif payload["message_type"] == "std_msgs/msg/Float64MultiArray":
        if not isinstance(exact, dict) or set(exact) != {"data"} or not isinstance(exact["data"], list):
            raise ValueError("Float64MultiArray exact_payload 必须只包含 data 数组")
        parsed_exact = Float64MultiArrayExactPayload(tuple(exact["data"]))
    else:
        raise ValueError("不支持的 group_record message_type")
    return DispatchGroupRecord(
        group=payload["group"],
        official_topic=payload["official_topic"],
        message_type=payload["message_type"],
        attempted=payload["attempted"],
        succeeded=payload["succeeded"],
        exact_payload=parsed_exact,
        failure_reason=payload["failure_reason"],
    )


def action_dispatch_from_json(raw: str) -> ActionDispatchRecord:
    """严格解析 ActionDispatchRecord V1，拒绝未知版本、维度和非有限数。"""

    try:
        payload = _strict_json_object(raw, "ActionDispatchRecord")
        expected = {
            "schema_name", "schema_version", "sequence", "timestamp_ns",
            "final_action_sequence", "decision", "calculated", "publish_enabled",
            "publisher_created", "publish_attempted", "publisher_call_succeeded",
            "dispatch_mode", "dispatched_action", "dispatched_mask", "attempted_groups",
            "successful_groups", "failed_groups", "group_records", "controller_accepted",
            "execution_confirmed", "failure_reason",
        }
        _require_keys(payload, expected, "ActionDispatchRecord")
        return ActionDispatchRecord(
            schema_name=payload["schema_name"],
            schema_version=payload["schema_version"],
            sequence=payload["sequence"],
            timestamp_ns=payload["timestamp_ns"],
            final_action_sequence=payload["final_action_sequence"],
            decision=_decision_from_dict(payload["decision"]),
            calculated=payload["calculated"],
            publish_enabled=payload["publish_enabled"],
            publisher_created=payload["publisher_created"],
            publish_attempted=payload["publish_attempted"],
            publisher_call_succeeded=payload["publisher_call_succeeded"],
            dispatch_mode=DispatchMode(payload["dispatch_mode"]),
            dispatched_action=tuple(payload["dispatched_action"]),
            dispatched_mask=tuple(payload["dispatched_mask"]),
            attempted_groups=tuple(payload["attempted_groups"]),
            successful_groups=tuple(payload["successful_groups"]),
            failed_groups=tuple(payload["failed_groups"]),
            group_records=tuple(_group_from_dict(item) for item in payload["group_records"]),
            controller_accepted=payload["controller_accepted"],
            execution_confirmed=payload["execution_confirmed"],
            failure_reason=payload["failure_reason"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ActionDispatchRecord JSON 无效：{exc}") from exc


# JSON编解码

def final_action_to_json(action: FinalAction) -> str:
    """把 FinalAction 序列化为唯一的 ROS2 遥测 JSON 格式。

    典型生产者是 ROS2 遥测适配层，典型消费者是 Recorder 或离线诊断工具；序列化只
    保留公共语义，不把 ``valid`` 提升为“已发布”或“已执行”。

    参数：``action`` 是已通过 19 维和有限数校验的最终动作。
    返回：UTF-8 JSON 字符串，数值单位和坐标语义保持不变。
    失败：对象类型错误时抛出 ``TypeError``；不会修补或伪造无效动作。
    """

    if not isinstance(action, FinalAction):
        raise TypeError("final_action_to_json 只接受 FinalAction")
    payload = {
        "schema_version": 1,
        "sequence": action.sequence,
        "timestamp_ns": action.timestamp_ns,
        "action": list(action.values),
        "global_phase": action.global_phase.value,
        "local_phase": action.local_phase.value,
        "valid": action.valid,
        "clipped": action.clipped,
        "failure_reason": action.failure_reason,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def final_action_from_json(raw: str) -> FinalAction:
    """从 ROS2 遥测 JSON 恢复并重新校验 FinalAction。

    典型调用者是 Recorder/离线读取逻辑。解码成功只证明 JSON 能恢复成公共接口，
    不证明动作曾被 Server 接收或被机器人执行。

    参数：``raw`` 为 ``/team/final_action`` 的 UTF-8 JSON 文本。
    返回：不可变的 19 维 ``FinalAction``，单位保持原定义。
    失败：JSON、schema、枚举、长度或有限数不合法时抛出 ``ValueError``。
    """

    try:
        payload = json.loads(raw)
        if payload.get("schema_version") != 1:
            raise ValueError("不支持的 FinalAction schema_version")
        return FinalAction(
            values=tuple(payload["action"]),
            sequence=int(payload["sequence"]),
            timestamp_ns=int(payload["timestamp_ns"]),
            global_phase=GlobalPhase(payload["global_phase"]),
            local_phase=LocalPhase(payload["local_phase"]),
            valid=bool(payload["valid"]),
            clipped=bool(payload["clipped"]),
            failure_reason=str(payload.get("failure_reason", "")),
        )
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"FinalAction JSON 无效：{exc}") from exc


def fsm_status_to_json(status: FSMStatus) -> str:
    """把 FSMStatus 序列化为团队统一 JSON。

    典型生产者是 ROS2 遥测适配层，典型消费者是 Recorder 或离线诊断工具；它保存
    客户端 FSM 状态，不编码裁判结算语义。

    参数：``status`` 为当前全局/局部状态；不包含空间量。
    返回：供 ``/team/fsm_status`` 和记录器使用的 UTF-8 JSON。
    失败：传入非 FSMStatus 时抛出 ``TypeError``。
    """

    if not isinstance(status, FSMStatus):
        raise TypeError("fsm_status_to_json 只接受 FSMStatus")
    payload = asdict(status)
    payload["schema_version"] = 1
    payload["global_phase"] = status.global_phase.value
    payload["local_phase"] = status.local_phase.value
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def fsm_status_from_json(raw: str) -> FSMStatus:
    """从团队 FSM JSON 恢复状态对象。

    典型调用者是 Recorder/离线读取逻辑。恢复出的 ``success`` 仍是客户端 FSM
    语义，不能据此推断裁判得分。

    参数：``raw`` 为 ``/team/fsm_status`` 文本。
    返回：``FSMStatus``，时间单位纳秒。
    失败：JSON 字段、schema 或枚举非法时抛出 ``ValueError``。
    """

    try:
        payload = json.loads(raw)
        if payload.get("schema_version") != 1:
            raise ValueError("不支持的 FSMStatus schema_version")
        return FSMStatus(
            task_id=int(payload["task_id"]),
            global_phase=GlobalPhase(payload["global_phase"]),
            local_phase=LocalPhase(payload["local_phase"]),
            retry_count=int(payload["retry_count"]),
            success=bool(payload["success"]),
            failure_reason=str(payload.get("failure_reason", "")),
            timestamp_ns=int(payload["timestamp_ns"]),
        )
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"FSMStatus JSON 无效：{exc}") from exc
