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
from typing import Any, Optional, Sequence


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
    SAFE_HOLD = "SAFE_HOLD"#客户端认为任务无法继续
    FAILED = "FAILED"#暂停普通动作，保持安全状态


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


# 任务与机器人实际状态

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
    经过规划转换为 ``PlaceTarget``。``None`` 表示可选字段未给出，解析失败时
    ``valid=False`` 并填写原因。
    """

    task_id: int
    instruction: str
    target_kind: str
    target_body: str
    target_color: str
    place_type: str
    place_world_xyz: Optional[tuple[float, float, float]]
    place_radius: Optional[float]
    ref_prop: Optional[str] = None
    ref_prop_body: Optional[str] = None
    direction: Optional[str] = None
    timestamp_ns: int = 0
    valid: bool = True
    failure_reason: str = ""


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
    valid

    典型生产者是二维检测适配器，典型消费者是检测稳定逻辑和三维估计器。它是图像
    平面中的检测结果，不携带深度或三维中心。

    ``bbox_xyxy`` 为图像像素坐标 ``(x0,y0,x1,y1)``；``confidence`` 范围建议为
    0～1。该接口不携带三维位置。框越界可由检测适配层裁剪，格式错误时返回无效结果。
    """

    class_id: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    timestamp_ns: int
    valid: bool = True
    failure_reason: str = ""


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

    业务上的典型生产者是 ``perception_3d`` 三维估计器，典型消费者是 team client、
    导航和抓放规划；``ros_nodes`` 只负责它与 ROS 消息之间的转换，不是三维估计算法
    生产者。该对象是物体中心估计，不是带朝向的 ``Pose3D`` 夹爪末端目标：箱子中心
    可以在中间，而左右夹爪目标分别位于箱子两侧。

    ``position_xyz`` 单位米，位于 ``frame_id`` 指定的 world/odom 等坐标系；它应是
    经过表面点补偿后的物体中心估计。``slot_type`` 可在 team client 收到后计算；
    估计失败时 ``valid=False``。
    """

    class_id: str
    position_xyz: tuple[float, float, float]
    confidence: float
    frame_id: str
    timestamp_ns: int
    slot_type: SlotType = SlotType.UNKNOWN
    valid: bool = True
    failure_reason: str = ""


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
    """

    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    frame_id: str


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
    | `lift_delta_m`   | 抓住后试抬或抬升的高度   |
    | `confidence`     | 抓取目标几何的可信度    |
    | `valid`          | 当前抓取目标是否可用    |
    | `failure_reason` | 生成失败原因，如目标不可达 |


    典型生产者是根据 ``ObjectEstimate3D`` 生成抓取几何的规划器，典型消费者是 IK 和
    轨迹规划。它是规划目标，不是机械臂实际状态或已抓住物体的证明。

    四个位姿均应位于机器人基座坐标系，长度单位米。物体中心不等于夹爪末端；规划
    必须结合抓取方向和双臂间距。规划未实现或目标不可达时 ``valid=False``。
    """

    left_pregrasp: Pose3D
    right_pregrasp: Pose3D
    left_grasp: Pose3D
    right_grasp: Pose3D
    lift_delta_m: float
    confidence: float
    valid: bool = True
    failure_reason: str = ""


@dataclass(frozen=True)
class PlaceTarget:
    """双臂放置位姿集合。
    | 字段                | 含义            |
    | ----------------- | ------------- |
    | `object_goal_xyz` | 任务要求的物体中心最终位置 |
    | `left_preplace`   | 左夹爪预放置位姿      |
    | `right_preplace`  | 右夹爪预放置位姿      |
    | `left_release`    | 左夹爪释放物体时的位姿   |
    | `right_release`   | 右夹爪释放物体时的位姿   |
    | `settle_time_s`   | 放下后等待物体稳定的时间  |
    | `valid`           | 放置目标是否有效      |
    | `failure_reason`  | 放置规划失败原因      |


    典型生产者是把 ``TaskSpec.place_world_xyz`` 转成双臂几何目标的规划器，典型
    消费者是 IK 和轨迹规划。``object_goal_xyz`` 是任务要求的物体中心目标，单位米；
    左右手 ``preplace``/``release`` 才是机械臂末端 ``Pose3D``。二者必须经过规划
    转换，不能直接混用；规划失败时 ``valid=False``。
    """

    object_goal_xyz: tuple[float, float, float]
    left_preplace: Pose3D
    right_preplace: Pose3D
    left_release: Pose3D
    right_release: Pose3D
    settle_time_s: float
    valid: bool = True
    failure_reason: str = ""


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


    典型生产者是轨迹规划器，典型消费者是机械臂执行器。它是完整轨迹中的一个计划
    采样点，不是单独的机器人反馈。

    ``time_from_start_s`` 单位秒；位置顺序与 ``JOINT_NAMES`` 一致，mask 为真才由该
    路点控制。长度或数值错误时抛出 ``ValueError``。
    """

    time_from_start_s: float
    joint_position: tuple[float, ...]
    controlled_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_position", ensure_finite_vector(self.joint_position, 17, "路点关节位置"))
        if len(self.controlled_mask) != 17:
            raise ValueError("controlled_mask 必须包含 17 项")


@dataclass(frozen=True)
class JointTrajectory:
    """按时间排列的完整关节轨迹计划。
    | 字段               | 含义                       |
    | ---------------- | ------------------------ |
    | `trajectory_id`  | 轨迹唯一编号                   |
    | `waypoints`      | 按时间排列的多个 `JointWaypoint` |
    | `timestamp_ns`   | 轨迹生成时间                   |
    | `valid`          | 整条轨迹是否有效                 |
    | `failure_reason` | 轨迹无效原因，如路点为空、规划失败        |


    典型生产者是机械臂规划器，典型消费者是机械臂执行器。轨迹像整条路线，
    ``ManipulationCommand`` 像本周期走到哪一步；轨迹存在不代表已经开始或完成执行。

    ``timestamp_ns`` 为规划生成时间；轨迹只描述目标，不表示机械臂已经执行成功。
    空轨迹、时间倒序或规划失败时应设置 ``valid=False``。
    """

    trajectory_id: str
    waypoints: tuple[JointWaypoint, ...]
    timestamp_ns: int
    valid: bool = True
    failure_reason: str = ""


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

    时间单位纳秒，状态本身不带空间坐标。``failure_reason`` 记录客户端失败原因；
    非法状态或解析失败时 JSON 解码函数抛出 ``ValueError``。
    """

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
