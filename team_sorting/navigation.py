"""底盘几何工具和导航控制接口。

本文件是普通 Python 业务模块，不导入 ROS2，也不直接发布 ``/cmd_vel``。完整链路是：

``FSMStatus``（当前阶段） + ``TaskSpec/ObjectEstimate3D``（任务和目标坐标）
+ ``BaseState``（Odom 实际状态）
→ ``NavigationController`` 生成 ``NavGoal/BaseCommand``
→ ``ActionMux`` 检查短 TTL、限幅并完成安全仲裁
→ ``OfficialCommandPublisher`` 把 ``v/w`` 写入 ``Twist.linear.x/angular.z``
→ 官方 ``MMK2ROS2`` 再按轮距和轮半径换算为左右轮角速度。

因此 ``BaseCommand`` 只是候选速度建议，前两项是底盘前向线速度 ``v``（m/s）和绕
Z 轴角速度 ``w``（rad/s），不是左右轮速。本模块不得复制官方轮速换算，也不负责
YOLO、机械臂 IK、全局 FSM 判断或 ROS 发布。

    当前实现包含基础站位生成和单周期比例控制；它不进行全局路径搜索、障碍物规划或
    坐标变换。``NavigationStatus`` 必须根据实际 Odom 判断，不能用命令生成代替到达。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from .interfaces import (
    BaseCommand,
    BaseState,
    NavGoal,
    NavigationStatus,
    ObjectEstimate3D,
    SlotType,
    TaskSpec,
)

_ODOM_FRAME_ID = "odom"
_WORLD_FRAME_ID = "world"
_RETURN_END_POSE_XYYAW = (-0.70, 0.55, math.pi / 2.0)
_EARLY_SIMULATION_PLACE_STRATEGY = "early_simulation_standoff_strategy"


def _finite_real(value: object, field_name: str) -> float:
    """把真实数转换为有限浮点数，并统一基础几何函数的错误说明。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name}必须是真实有限数")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name}必须是真实有限数")
    return converted


def _nonnegative_int(value: object, field_name: str) -> int:
    """校验纳秒时长等非负整数配置，拒绝 bool 和隐式浮点截断。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name}必须是非负整数")
    return value


@dataclass(frozen=True)
class NavigationConfig:
    """导航局部策略的不可变配置。

    默认值是团队当前的保守初值，不是官方性能参数。配置保留在本模块内并通过
    ``NavigationController`` 构造函数注入，避免修改公共数据接口或 ROS 组装层。
    """

    standoff_m: float = 0.60
    position_tolerance_m: float = 0.05
    yaw_tolerance_rad: float = 0.10
    goal_timeout_ns: int = 30_000_000_000
    command_ttl_ns: int = 200_000_000
    odom_max_age_ns: int = 150_000_000
    target_max_age_ns: int = 150_000_000
    max_abs_v_mps: float = 0.25
    max_abs_w_radps: float = 0.50
    linear_kp: float = 0.8
    angular_kp: float = 1.5
    heading_stop_rad: float = math.pi / 4.0
    # 以下停稳值是模块内保守提案，不是官方仿真标定结果；共享配置获批前不得据此
    # 宣称导航已可正式接线。连续确认按不同 Odom 时间戳计数，不能重复消费同一帧。
    max_settled_linear_speed_mps: float = 0.01
    max_settled_angular_speed_radps: float = 0.02
    settled_required_cycles: int = 3
    settled_max_odom_gap_ns: int = 200_000_000

    def __post_init__(self) -> None:
        positive_fields = (
            ("standoff_m", self.standoff_m),
            ("position_tolerance_m", self.position_tolerance_m),
            ("yaw_tolerance_rad", self.yaw_tolerance_rad),
            ("max_abs_v_mps", self.max_abs_v_mps),
            ("max_abs_w_radps", self.max_abs_w_radps),
            ("linear_kp", self.linear_kp),
            ("angular_kp", self.angular_kp),
            ("heading_stop_rad", self.heading_stop_rad),
        )
        for name, value in positive_fields:
            if _finite_real(value, f"NavigationConfig.{name}") <= 0.0:
                raise ValueError(f"NavigationConfig.{name}必须大于零")
        for name, value in (
            ("goal_timeout_ns", self.goal_timeout_ns),
            ("command_ttl_ns", self.command_ttl_ns),
            ("odom_max_age_ns", self.odom_max_age_ns),
            ("target_max_age_ns", self.target_max_age_ns),
        ):
            if _nonnegative_int(value, f"NavigationConfig.{name}") == 0:
                raise ValueError(f"NavigationConfig.{name}必须大于零")
        for name, value in (
            ("max_settled_linear_speed_mps", self.max_settled_linear_speed_mps),
            ("max_settled_angular_speed_radps", self.max_settled_angular_speed_radps),
        ):
            if _finite_real(value, f"NavigationConfig.{name}") < 0.0:
                raise ValueError(f"NavigationConfig.{name}必须大于等于零")
        if _nonnegative_int(
            self.settled_required_cycles,
            "NavigationConfig.settled_required_cycles",
        ) == 0:
            raise ValueError("NavigationConfig.settled_required_cycles必须大于零")
        if _nonnegative_int(
            self.settled_max_odom_gap_ns,
            "NavigationConfig.settled_max_odom_gap_ns",
        ) == 0:
            raise ValueError("NavigationConfig.settled_max_odom_gap_ns必须大于零")


def _read_xy(point: object, field_name: str) -> tuple[float, float]:
    """读取坐标序列前两项；多出的 Z 等分量不参与底盘平面距离。"""

    if isinstance(point, (str, bytes)):
        raise ValueError(f"{field_name}必须是至少包含XY两项的坐标序列")
    try:
        if len(point) < 2:  # type: ignore[arg-type]
            raise ValueError(f"{field_name}必须是至少包含XY两项的坐标序列")
        x_value = point[0]  # type: ignore[index]
        y_value = point[1]  # type: ignore[index]
    except (TypeError, KeyError, IndexError, OverflowError) as exc:
        raise ValueError(f"{field_name}必须是至少包含XY两项的坐标序列") from exc
    return (
        _finite_real(x_value, f"{field_name}.x"),
        _finite_real(y_value, f"{field_name}.y"),
    )


def wrap_to_pi(angle_rad: float) -> float:
    """把角度归一化到 ``[-π, π)``。

    参数和返回值单位均为弧度，不依赖坐标系。归一化后，控制器可以比较“向左转一点”
    和“绕另一侧转一大圈”，从而选择较短的偏航误差。布尔值、非数值及 NaN/Inf 都会
    抛出清晰的 ``ValueError``。
    """

    angle = _finite_real(angle_rad, "角度")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def distance_xy(first_xyz: tuple[float, ...], second_xyz: tuple[float, ...]) -> float:
    """计算同一坐标系下两点的水平欧氏距离。

    底盘在地面上运动，所以只使用坐标的前两项 XY，多出的 Z 不参与计算。参数和返回值
    单位均为米；函数不执行 frame 转换，调用方必须保证两点属于同一坐标系。非法序列、
    布尔值、字符串及 NaN/Inf 统一抛出 ``ValueError``。
    """

    first_x, first_y = _read_xy(first_xyz, "第一个坐标")
    second_x, second_y = _read_xy(second_xyz, "第二个坐标")
    delta_x = _finite_real(first_x - second_x, "X坐标差")
    delta_y = _finite_real(first_y - second_y, "Y坐标差")
    return _finite_real(math.hypot(delta_x, delta_y), "XY距离")


def _stand_pose_from_direction(
    target_xy: object,
    direction_xy: object,
    standoff_m: object,
) -> tuple[float, float, float]:
    """由已确认的站位到目标方向计算平面站位，不解析任何业务方向语义。"""

    target_x, target_y = _read_xy(target_xy, "target_xy")
    direction_x, direction_y = _read_xy(direction_xy, "direction_xy")
    try:
        if len(target_xy) != 2 or len(direction_xy) != 2:  # type: ignore[arg-type]
            raise ValueError("target_xy和direction_xy必须严格包含XY两项")
    except TypeError as exc:
        raise ValueError("target_xy和direction_xy必须严格包含XY两项") from exc
    direction_norm = _finite_real(
        math.hypot(direction_x, direction_y), "direction_xy长度"
    )
    if direction_norm <= 1e-9:
        raise ValueError("direction_xy长度过小，无法确定站位方向")
    standoff = _finite_real(standoff_m, "standoff_m")
    if standoff <= 0.0:
        raise ValueError("standoff_m必须大于零")
    unit_x = _finite_real(direction_x / direction_norm, "站位方向X分量")
    unit_y = _finite_real(direction_y / direction_norm, "站位方向Y分量")
    goal_x = _finite_real(target_x - standoff * unit_x, "站位X坐标")
    goal_y = _finite_real(target_y - standoff * unit_y, "站位Y坐标")
    facing_x = _finite_real(target_x - goal_x, "目标朝向X分量")
    facing_y = _finite_real(target_y - goal_y, "目标朝向Y分量")
    actual_distance = _finite_real(
        math.hypot(facing_x, facing_y), "实际站位退让距离"
    )
    if actual_distance <= 1e-9 or not math.isclose(
        actual_distance,
        standoff,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("请求站位在当前浮点精度下无法表示")
    goal_yaw = wrap_to_pi(math.atan2(facing_y, facing_x))
    return goal_x, goal_y, goal_yaw


@dataclass(frozen=True)
class Bounds3D:
    """用于粗粒度槽位分类的轴对齐三维区域。

    它像一个各边平行于坐标轴的长方体，可按配置粗分桌面、货架等区域；它不是碰撞
    检测器，也不执行坐标系转换。六个边界单位米，必须和待分类点处于同一 frame。
    边界非法或最小值大于最大值时，``contains`` 返回 ``False``，不会猜测类别。
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, point_xyz: tuple[float, float, float]) -> bool:
        """判断米制三维点是否位于包含边界的区域内。

        参数点必须和区域处于同一 frame；返回布尔值。点含 NaN/Inf 或区域配置非法时
        返回 ``False``，不会猜测槽位。
        """

        try:
            x_min = _finite_real(self.x_min, "x_min")
            x_max = _finite_real(self.x_max, "x_max")
            y_min = _finite_real(self.y_min, "y_min")
            y_max = _finite_real(self.y_max, "y_max")
            z_min = _finite_real(self.z_min, "z_min")
            z_max = _finite_real(self.z_max, "z_max")
        except ValueError:
            return False
        if x_min > x_max or y_min > y_max or z_min > z_max:
            return False
        if isinstance(point_xyz, (str, bytes)):
            return False
        try:
            if len(point_xyz) != 3:
                return False
            x = _finite_real(point_xyz[0], "point.x")
            y = _finite_real(point_xyz[1], "point.y")
            z = _finite_real(point_xyz[2], "point.z")
        except (TypeError, ValueError, KeyError, IndexError, OverflowError):
            return False
        return (
            x_min <= x <= x_max
            and y_min <= y <= y_max
            and z_min <= z <= z_max
        )


def classify_slot_type(
    point_xyz: tuple[float, float, float],
    table_bounds: Bounds3D,
    shelf_bounds: Bounds3D,
) -> SlotType:
    """在 team client 内按配置区域计算目标粗粒度槽位。

    参数点和两个区域必须位于同一 planning frame，坐标单位米。返回 ``TABLE``、
    ``SHELF`` 或 ``UNKNOWN``。无效点不抛异常而返回 ``UNKNOWN``。
    """

    # 重叠时 TABLE 优先是当前团队的确定性分类策略，不是裁判事实；配置应尽量避免重叠。
    if table_bounds.contains(point_xyz):
        return SlotType.TABLE
    if shelf_bounds.contains(point_xyz):
        return SlotType.SHELF
    return SlotType.UNKNOWN


class NavigationController:
    """根据目标中心生成操作站位，并用 Odom 计算单周期底盘候选命令。

    ``NavGoal`` 是要去的底盘站位，``BaseState`` 是 Odom 反馈的实际位置，
    ``BaseCommand`` 是本周期短时有效的 v/w 候选建议，``NavigationStatus`` 则是用实际
    反馈算出的执行状态。站位是机械臂能安全操作时底盘应停的位置，不等于物体中心。

    控制器不发布 ``/cmd_vel``，候选命令必须交给 ``ActionMux``；短 TTL 可防止控制循环
    中断后继续沿用旧速度。本类只做局部几何和比例控制，不代替全局路径规划或 FSM。
    """

    def __init__(self, config: NavigationConfig | None = None) -> None:
        self._config = config if config is not None else NavigationConfig()
        self._active_goal_key: tuple[object, ...] | None = None
        self._settled_samples = 0
        self._last_settled_odom_timestamp_ns: int | None = None
        self._arrival_latched = False
        self._execution_elapsed_ns = 0
        self._clock_checkpoint_ns: int | None = None
        self._control_granted_since_checkpoint = False
        self._goal_terminal = False

    def reset(self) -> None:
        """取消当前导航生命周期；不生成命令，也不代表机器人已经停止。"""

        self._active_goal_key = None
        self._reset_settled_confirmation()
        self._arrival_latched = False
        self._execution_elapsed_ns = 0
        self._clock_checkpoint_ns = None
        self._control_granted_since_checkpoint = False
        self._goal_terminal = False

    def _activate_goal(
        self, goal_key: tuple[object, ...], timestamp_ns: int
    ) -> None:
        """原子建立新目标的停稳确认和control-aware执行预算。"""

        self._active_goal_key = goal_key
        self._reset_settled_confirmation()
        self._arrival_latched = False
        self._execution_elapsed_ns = 0
        self._clock_checkpoint_ns = timestamp_ns
        self._control_granted_since_checkpoint = False
        self._goal_terminal = False

    def _advance_execution_clock(self, timestamp_ns: int) -> None:
        """只结算上个检查点之后已明确获准控制的区间。"""

        checkpoint_ns = self._clock_checkpoint_ns
        if checkpoint_ns is None:
            self._clock_checkpoint_ns = timestamp_ns
            return
        if timestamp_ns < checkpoint_ns:
            self._control_granted_since_checkpoint = False
            raise ValueError("导航控制权回报时间不得倒退")
        if self._control_granted_since_checkpoint and not self._goal_terminal:
            self._execution_elapsed_ns += timestamp_ns - checkpoint_ns
        self._clock_checkpoint_ns = timestamp_ns

    def record_control_result(
        self, timestamp_ns: int, control_granted: bool
    ) -> None:
        """记录同周期内部导航候选的仲裁与正式发布结果。

        ``True`` 只表示内部 Navigation ``BaseCommand`` 在该周期通过 ActionMux、FULL
        授权和本地 ROS publish 调用；不表示 Server 接收、机器人运动或导航成功。
        非法输入会撤销授权并抛出 ``ValueError``，未知区间不会计入执行预算。
        """

        try:
            timestamp = self._timestamp(timestamp_ns, "timestamp_ns")
        except ValueError:
            self._control_granted_since_checkpoint = False
            raise
        if type(control_granted) is not bool:
            self._control_granted_since_checkpoint = False
            checkpoint_ns = self._clock_checkpoint_ns
            if checkpoint_ns is None or timestamp >= checkpoint_ns:
                self._clock_checkpoint_ns = timestamp
            raise ValueError("control_granted必须是严格bool")
        self._advance_execution_clock(timestamp)
        self._control_granted_since_checkpoint = bool(
            control_granted
            and self._active_goal_key is not None
            and not self._goal_terminal
        )

    def _revoke_control_without_timestamp(self) -> None:
        """在没有可信时间戳时保守撤权，不结算或移动执行时钟。"""

        self._control_granted_since_checkpoint = False

    def build_pick_goal(
        self, task: TaskSpec, target: ObjectEstimate3D, base: BaseState, timestamp_ns: int
    ) -> NavGoal:
        """根据物体中心和当前底盘位置生成抓取时的底盘 ``NavGoal``。

        ``ObjectEstimate3D.position_xyz`` 是物体中心，不是底盘停车点。当前最小确定性策略
        沿“底盘到物体”的方向退让固定操作距离，再让底盘朝向物体；它不推测障碍物或
        机械臂可达范围。``target`` 与 ``base`` 必须已处于同一 frame。
        """

        # 先验证时间、任务和 Odom；任何几何计算都不能建立在无效状态上。
        now = self._timestamp(timestamp_ns, "timestamp_ns")
        self._validate_task(task, now)
        self._validate_base(base, now, require_fresh=True)
        if not isinstance(target, ObjectEstimate3D) or not target.valid:
            raise ValueError("ObjectEstimate3D 无效")
        if target.class_id != task.target_color:
            raise ValueError(
                "ObjectEstimate3D.class_id 必须与 TaskSpec.target_color 一致"
            )
        target_stamp = self._timestamp(target.timestamp_ns, "target.timestamp_ns")
        if target_stamp > now:
            raise ValueError("ObjectEstimate3D 时间戳晚于当前周期")
        if now - target_stamp > self._config.target_max_age_ns:
            raise ValueError("ObjectEstimate3D 已过期")
        if len(target.position_xyz) != 3:
            raise ValueError("target.position_xyz 必须包含三项")
        target_x, target_y = _read_xy(target.position_xyz, "target.position_xyz")
        _finite_real(target.position_xyz[2], "target.position_xyz.z")
        if base.frame_id != _ODOM_FRAME_ID or target.frame_id != _ODOM_FRAME_ID:
            raise ValueError('抓取导航要求 target/base frame_id 严格为 "odom"')
        if target.frame_id != base.frame_id:
            raise ValueError("目标与 BaseState frame 不一致，且仓库没有坐标转换接口")
        # 物体中心只用于反算站位，不能直接复制为底盘停车点。
        goal_x, goal_y, goal_yaw = self._stand_off_pose(
            target_x, target_y, base.position_xyz
        )
        self._finite_vector((goal_x, goal_y, goal_yaw), "抓取 NavGoal.pose_xyyaw")
        return NavGoal(
            f"pick-{task.task_id}-{now}",
            "pick",
            (goal_x, goal_y, goal_yaw),
            target.frame_id,
            self._config.position_tolerance_m,
            self._config.yaw_tolerance_rad,
            now + self._config.goal_timeout_ns,
        )

    def build_place_goal(self, task: TaskSpec, base: BaseState, timestamp_ns: int) -> NavGoal:
        """根据 ``TaskSpec.place_world_xyz`` 反算放置时的底盘站位。

        ``place_world_xyz`` 是物体最终中心，单位米，不是底盘停车点。当前使用标识为
        ``early_simulation_standoff_strategy`` 的固定退让策略，仅获准用于早期官方仿真
        联调，不代表已处理 place_type/direction、禁入区或机械臂可达性。当前按冻结的 F1
        约定把 world 数值坐标显式复制为 odom 数值坐标；这是绑定当前官方镜像的
        world/odom 对齐策略，不是通用 frame 转换，也不会创建或查询 ROS TF。
        """

        now = self._timestamp(timestamp_ns, "timestamp_ns")
        self._validate_task(task, now)
        self._validate_base(base, now, require_fresh=True)
        if task.place_frame_id != _WORLD_FRAME_ID:
            raise ValueError('TaskSpec.place_frame_id 必须严格为 "world"')
        if base.frame_id != _ODOM_FRAME_ID:
            raise ValueError('BaseState.frame_id 必须严格为 "odom"')
        if task.place_world_xyz is None or len(task.place_world_xyz) != 3:
            raise ValueError("TaskSpec.place_world_xyz 必须包含 world 三维坐标")
        place_x, place_y = _read_xy(task.place_world_xyz, "task.place_world_xyz")
        _finite_real(task.place_world_xyz[2], "task.place_world_xyz.z")
        # F1：当前官方镜像中 world 与 odom 数值对齐。此处是镜像绑定的数值复制，
        # 不是一般 ROS 坐标变换；若官方镜像改变，调用方必须先更新冻结约定。
        place_odom_xyz = (place_x, place_y, float(task.place_world_xyz[2]))
        goal_x, goal_y, goal_yaw = self._stand_off_pose(
            place_odom_xyz[0], place_odom_xyz[1], base.position_xyz
        )
        self._finite_vector((goal_x, goal_y, goal_yaw), "放置 NavGoal.pose_xyyaw")
        return NavGoal(
            f"place-{_EARLY_SIMULATION_PLACE_STRATEGY}-{task.task_id}-{now}",
            "place",
            (goal_x, goal_y, goal_yaw),
            _ODOM_FRAME_ID,
            self._config.position_tolerance_m,
            self._config.yaw_tolerance_rad,
            now + self._config.goal_timeout_ns,
        )

    def build_return_goal(self, base: BaseState, timestamp_ns: int) -> NavGoal:
        """生成冻结 F5 结束区目标，不依赖任务、视觉或退让站位。

        目标位于 ``odom``，XY 单位米、yaw 单位弧度；固定位置是官方结束区中心，
        ``pi/2`` 朝向 world/odom 的 +Y。到达仍必须由后续 ``update`` 使用实际 Odom
        判断，生成本目标本身不表示已经返区。
        """

        now = self._timestamp(timestamp_ns, "timestamp_ns")
        self._validate_base(base, now, require_fresh=True)
        if base.frame_id != _ODOM_FRAME_ID:
            raise ValueError('BaseState.frame_id 必须严格为 "odom"')
        return NavGoal(
            f"return-{now}",
            "return",
            _RETURN_END_POSE_XYYAW,
            _ODOM_FRAME_ID,
            self._config.position_tolerance_m,
            self._config.yaw_tolerance_rad,
            now + self._config.goal_timeout_ns,
        )

    def update(
        self, base: BaseState, goal: NavGoal, timestamp_ns: int
    ) -> tuple[BaseCommand, NavigationStatus]:
        """使用实际 Odom 推进一次导航控制周期。

        控制分为“朝向停车点”“向停车点前进”“原地对准最终 yaw”和“到达”四种情况。
        输出是带短 TTL 的 ``BaseCommand(v,w)`` 和基于同一份 Odom 的
        ``NavigationStatus``；每次调用只生成一个周期的建议。

        只有实际 Odom 同时满足位置、角度和停稳速度阈值，并由不同 Odom 帧连续确认后，
        才能 ``success=True``；deadline 过期、frame 不一致或目标无效时必须失败。生成
        ``NavGoal`` 或速度命令都不代表已经到达。
        """

        try:
            # 第一阶段：验证本周期时间、Odom、目标、frame、deadline 和容差。
            now = self._timestamp(timestamp_ns, "timestamp_ns")
            self._validate_base(base, now, require_fresh=True)
            if not isinstance(goal, NavGoal) or not goal.valid:
                raise ValueError("NavGoal 无效")
            if not goal.goal_id or not goal.frame_id:
                raise ValueError("NavGoal 标识或 frame 为空")
            if goal.frame_id != base.frame_id:
                raise ValueError("NavGoal 与 BaseState frame 不一致")
            deadline = self._timestamp(goal.deadline_ns, "goal.deadline_ns")
            if len(goal.pose_xyyaw) != 3:
                raise ValueError("goal.pose_xyyaw 必须包含三项")
            goal_x, goal_y = _read_xy(goal.pose_xyyaw, "goal.pose_xyyaw")
            goal_yaw = _finite_real(goal.pose_xyyaw[2], "goal.pose_xyyaw.yaw")
            position_tolerance = _finite_real(
                goal.position_tolerance, "goal.position_tolerance"
            )
            yaw_tolerance = _finite_real(goal.yaw_tolerance, "goal.yaw_tolerance")
            if position_tolerance < 0.0 or yaw_tolerance < 0.0:
                raise ValueError("NavGoal 容差不能为负数")

            goal_key = (
                goal.goal_id,
                goal.goal_type,
                goal.pose_xyyaw,
                goal.frame_id,
                goal.position_tolerance,
                goal.yaw_tolerance,
                goal.deadline_ns,
            )
            if goal_key != self._active_goal_key:
                self._activate_goal(goal_key, now)
            else:
                self._advance_execution_clock(now)
            if self._execution_elapsed_ns >= self._config.goal_timeout_ns:
                self._reset_settled_confirmation()
                self._goal_terminal = True
                self._control_granted_since_checkpoint = False
                return self._stopped(
                    goal.goal_id,
                    now,
                    "timeout",
                    "导航有效控制预算已达到 goal_timeout_ns",
                    valid=False,
                )

            # 第二阶段：所有输入确认可靠后，再计算位置误差和最终姿态误差。
            base_x, base_y = _read_xy(base.position_xyz, "base.position_xyz")
            distance_error = distance_xy((base_x, base_y), (goal_x, goal_y))
            final_yaw_error = wrap_to_pi(goal_yaw - base.yaw)
            linear_speed, angular_speed = self._base_speed_norms(base)
            if self._arrival_latched:
                still_settled = (
                    distance_error <= position_tolerance
                    and abs(final_yaw_error) <= yaw_tolerance
                    and linear_speed <= self._config.max_settled_linear_speed_mps
                    and angular_speed <= self._config.max_settled_angular_speed_radps
                )
                return (
                    BaseCommand(0.0, 0.0, now, now + self._config.command_ttl_ns),
                    NavigationStatus(
                        goal.goal_id,
                        "arrived" if still_settled else "moving",
                        distance_error,
                        final_yaw_error,
                        still_settled,
                        "" if still_settled else "目标已确认到达，等待 FSM 阶段切换并保持零速",
                        now,
                    ),
                )
            if distance_error <= position_tolerance:
                # 已进入位置容差：停止平移，只调整目标要求的最终 yaw。
                if abs(final_yaw_error) <= yaw_tolerance:
                    if (
                        linear_speed > self._config.max_settled_linear_speed_mps
                        or angular_speed
                        > self._config.max_settled_angular_speed_radps
                    ):
                        self._reset_settled_confirmation()
                        return (
                            BaseCommand(
                                0.0, 0.0, now, now + self._config.command_ttl_ns
                            ),
                            NavigationStatus(
                                goal.goal_id,
                                "moving",
                                distance_error,
                                final_yaw_error,
                                False,
                                "位置与朝向已满足，等待实际 Odom 速度停稳",
                                now,
                            ),
                        )
                    self._record_settled_sample(base.timestamp_ns)
                    if self._settled_samples < self._config.settled_required_cycles:
                        return (
                            BaseCommand(
                                0.0, 0.0, now, now + self._config.command_ttl_ns
                            ),
                            NavigationStatus(
                                goal.goal_id,
                                "moving",
                                distance_error,
                                final_yaw_error,
                                False,
                                "位置、朝向和速度已满足，等待连续 Odom 帧确认",
                                now,
                            ),
                        )
                    self._arrival_latched = True
                    self._goal_terminal = True
                    self._control_granted_since_checkpoint = False
                    return (
                        BaseCommand(
                            0.0, 0.0, now, now + self._config.command_ttl_ns
                        ),
                        NavigationStatus(
                            goal.goal_id,
                            "arrived",
                            distance_error,
                            final_yaw_error,
                            True,
                            "",
                            now,
                        ),
                    )
                self._reset_settled_confirmation()
                v = 0.0
                raw_w = _finite_real(
                    self._config.angular_kp * final_yaw_error, "最终对准角速度"
                )
                w = self._clamp(raw_w, self._config.max_abs_w_radps)
                state = "aligning_final_yaw"
                control_yaw_error = final_yaw_error
            else:
                self._reset_settled_confirmation()
                # 尚未到达停车点：先朝向停车点，再按距离比例向前推进。
                bearing = math.atan2(goal_y - base_y, goal_x - base_x)
                heading_error = wrap_to_pi(bearing - base.yaw)
                control_yaw_error = heading_error
                raw_w = _finite_real(
                    self._config.angular_kp * heading_error, "航向角速度"
                )
                w = self._clamp(raw_w, self._config.max_abs_w_radps)
                if abs(heading_error) >= self._config.heading_stop_rad:
                    # 偏航过大时禁止“边大幅转向边高速前进”。
                    v = 0.0
                    state = "aligning_to_goal"
                else:
                    # 距离比例项负责近目标减速，余弦项进一步抑制带偏角前进。
                    raw_v = _finite_real(
                        self._config.linear_kp * distance_error, "距离比例线速度"
                    )
                    v = min(self._config.max_abs_v_mps, raw_v)
                    v = _finite_real(
                        v * max(0.0, math.cos(heading_error)), "航向抑制线速度"
                    )
                    state = "moving"
            if not math.isfinite(v) or not math.isfinite(w):
                raise ValueError("导航控制计算得到非有限速度")
            return (
                BaseCommand(v, w, now, now + self._config.command_ttl_ns),
                NavigationStatus(
                    goal.goal_id,
                    state,
                    distance_error,
                    control_yaw_error,
                    False,
                    "",
                    now,
                ),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            # 非法输入或非有限计算结果统一转成无效零速候选，交给 ActionMux 安全仲裁。
            self._reset_settled_confirmation()
            self._goal_terminal = True
            self._control_granted_since_checkpoint = False
            goal_id = goal.goal_id if isinstance(goal, NavGoal) else ""
            safe_now = (
                timestamp_ns
                if isinstance(timestamp_ns, int)
                and not isinstance(timestamp_ns, bool)
                and timestamp_ns >= 0
                else 0
            )
            return self._stopped(goal_id, safe_now, "failed", str(exc), valid=False)

    @staticmethod
    def _timestamp(value: object, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} 必须是非负整数纳秒时间戳")
        return value

    @staticmethod
    def _validate_task(task: TaskSpec, now: int) -> None:
        if not isinstance(task, TaskSpec) or not task.valid:
            raise ValueError("TaskSpec 无效")
        if isinstance(task.task_id, bool) or not isinstance(task.task_id, int):
            raise ValueError("TaskSpec.task_id 必须是整数")
        stamp = NavigationController._timestamp(task.timestamp_ns, "task.timestamp_ns")
        if stamp > now:
            raise ValueError("TaskSpec 时间戳晚于当前周期")

    def _validate_base(
        self, base: BaseState, now: int, *, require_fresh: bool
    ) -> None:
        if not isinstance(base, BaseState) or not base.valid:
            raise ValueError("BaseState 无效")
        if not base.frame_id:
            raise ValueError("BaseState.frame_id 不能为空")
        stamp = self._timestamp(base.timestamp_ns, "base.timestamp_ns")
        if stamp > now:
            raise ValueError("BaseState 时间戳晚于当前周期")
        if require_fresh and now - stamp > self._config.odom_max_age_ns:
            raise ValueError("BaseState/Odom 已过期")
        if len(base.position_xyz) != 3:
            raise ValueError("base.position_xyz 必须包含三项")
        for index, value in enumerate(base.position_xyz):
            _finite_real(value, f"base.position_xyz[{index}]")
        _finite_real(base.yaw, "base.yaw")
        self._base_speed_norms(base)

    @staticmethod
    def _base_speed_norms(base: BaseState) -> tuple[float, float]:
        """读取 Odom 的三轴实际速度范数，拒绝损坏或非有限反馈。"""

        linear = NavigationController._validated_velocity_vector(
            base.linear_velocity_xyz, "base.linear_velocity_xyz"
        )
        angular = NavigationController._validated_velocity_vector(
            base.angular_velocity_xyz, "base.angular_velocity_xyz"
        )
        return (
            _finite_real(math.hypot(*linear), "BaseState线速度范数"),
            _finite_real(math.hypot(*angular), "BaseState角速度范数"),
        )

    @staticmethod
    def _validated_velocity_vector(
        values: object, field_name: str
    ) -> tuple[float, float, float]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{field_name}必须包含三项")
        try:
            if len(values) != 3:  # type: ignore[arg-type]
                raise ValueError(f"{field_name}必须包含三项")
            return tuple(
                _finite_real(values[index], f"{field_name}[{index}]")  # type: ignore[index]
                for index in range(3)
            )  # type: ignore[return-value]
        except (TypeError, KeyError, IndexError, OverflowError) as exc:
            raise ValueError(f"{field_name}必须包含三项") from exc

    def _record_settled_sample(self, odom_timestamp_ns: int) -> None:
        """只用新的、间隔正常的 Odom 帧累计连续停稳确认。"""

        previous = self._last_settled_odom_timestamp_ns
        if previous is None:
            self._settled_samples = 1
            self._last_settled_odom_timestamp_ns = odom_timestamp_ns
            return
        if odom_timestamp_ns < previous:
            raise ValueError("BaseState/Odom 时间戳相对上一停稳样本倒退")
        if odom_timestamp_ns == previous:
            return
        if odom_timestamp_ns - previous > self._config.settled_max_odom_gap_ns:
            raise ValueError("连续停稳确认的 Odom 时间间隔异常")
        self._settled_samples += 1
        self._last_settled_odom_timestamp_ns = odom_timestamp_ns

    def _reset_settled_confirmation(self) -> None:
        self._settled_samples = 0
        self._last_settled_odom_timestamp_ns = None

    def _stand_off_pose(
        self,
        target_x: float,
        target_y: float,
        base_position: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        base_x, base_y = _read_xy(base_position, "base.position_xyz")
        # 单位向量从底盘指向目标；从目标沿反方向退让，得到底盘中心停车点。
        dx = _finite_real(target_x - base_x, "目标与底盘X坐标差")
        dy = _finite_real(target_y - base_y, "目标与底盘Y坐标差")
        distance = _finite_real(math.hypot(dx, dy), "目标与底盘距离")
        if distance <= 1e-9:
            raise ValueError("底盘与目标中心重合，无法确定安全退让方向")
        return _stand_pose_from_direction(
            (target_x, target_y), (dx, dy), self._config.standoff_m
        )

    @staticmethod
    def _finite_vector(values: tuple[float, ...], field_name: str) -> None:
        for index, value in enumerate(values):
            _finite_real(value, f"{field_name}[{index}]")

    @staticmethod
    def _clamp(value: float, absolute_limit: float) -> float:
        return max(-absolute_limit, min(absolute_limit, value))

    def _stopped(
        self,
        goal_id: str,
        timestamp_ns: int,
        state: str,
        reason: str,
        *,
        valid: bool = True,
    ) -> tuple[BaseCommand, NavigationStatus]:
        # 失败命令仍携带短 TTL 和诊断原因，但 v/w 必须严格为零。
        return (
            BaseCommand(
                0.0,
                0.0,
                timestamp_ns,
                timestamp_ns + self._config.command_ttl_ns,
                valid=valid,
                failure_reason=reason,
            ),
            NavigationStatus(goal_id, state, 0.0, 0.0, False, reason, timestamp_ns),
        )
