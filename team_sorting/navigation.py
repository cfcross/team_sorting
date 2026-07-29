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

# 本仓库尚无经评审的导航参数配置段；以下是受 ActionMux 现有限幅约束的保守团队初值，
# 不是官方性能参数。待公共配置获批后应由组装层注入。
#
# 参数按用途分为四组：站位几何、目标有效期、反馈新鲜度和比例控制。集中声明可以避免
# 抓取、放置和 update 各自复制数值，也便于后续迁移到 config.yaml。
_STANDOFF_M = 0.60
_POSITION_TOLERANCE_M = 0.05
_YAW_TOLERANCE_RAD = 0.10
_GOAL_TIMEOUT_NS = 30_000_000_000
_COMMAND_TTL_NS = 200_000_000
_ODOM_MAX_AGE_NS = 150_000_000
_MAX_ABS_V_MPS = 0.25
_MAX_ABS_W_RADPS = 0.50
_LINEAR_KP = 0.8
_ANGULAR_KP = 1.5
_HEADING_STOP_RAD = math.pi / 4.0


def _finite_real(value: object, field_name: str) -> float:
    """把真实数转换为有限浮点数，并统一基础几何函数的错误说明。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name}必须是真实有限数")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name}必须是真实有限数")
    return converted


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
    return math.hypot(first_x - second_x, first_y - second_y)


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
        self._validate_base(base, now, require_fresh=False)
        if not isinstance(target, ObjectEstimate3D) or not target.valid:
            raise ValueError("ObjectEstimate3D 无效")
        target_stamp = self._timestamp(target.timestamp_ns, "target.timestamp_ns")
        if target_stamp > now:
            raise ValueError("ObjectEstimate3D 时间戳晚于当前周期")
        if len(target.position_xyz) != 3:
            raise ValueError("target.position_xyz 必须包含三项")
        target_x, target_y = _read_xy(target.position_xyz, "target.position_xyz")
        _finite_real(target.position_xyz[2], "target.position_xyz.z")
        if not target.frame_id or target.frame_id != base.frame_id:
            raise ValueError("目标与 BaseState frame 不一致，且仓库没有坐标转换接口")
        # 物体中心只用于反算站位，不能直接复制为底盘停车点。
        goal_x, goal_y, goal_yaw = self._stand_off_pose(
            target_x, target_y, base.position_xyz
        )
        return NavGoal(
            f"pick-{task.task_id}-{now}",
            "pick",
            (goal_x, goal_y, goal_yaw),
            target.frame_id,
            _POSITION_TOLERANCE_M,
            _YAW_TOLERANCE_RAD,
            now + _GOAL_TIMEOUT_NS,
        )

    def build_place_goal(self, task: TaskSpec, base: BaseState, timestamp_ns: int) -> NavGoal:
        """根据 ``TaskSpec.place_world_xyz`` 反算放置时的底盘站位。

        ``place_world_xyz`` 是物体最终中心，单位米，不是底盘停车点。由于公共接口没有
        world 到 odom 的转换，本方法只接受已经位于 world frame 的 ``BaseState``，避免
        暗中假设 ``world == odom``。
        """

        now = self._timestamp(timestamp_ns, "timestamp_ns")
        self._validate_task(task, now)
        self._validate_base(base, now, require_fresh=False)
        if task.place_world_xyz is None or len(task.place_world_xyz) != 3:
            raise ValueError("TaskSpec.place_world_xyz 必须包含 world 三维坐标")
        place_x, place_y = _read_xy(task.place_world_xyz, "task.place_world_xyz")
        _finite_real(task.place_world_xyz[2], "task.place_world_xyz.z")
        # 字段名明确规定 place_world_xyz 属于 world；没有变换时只能严格匹配。
        if base.frame_id != "world":
            raise ValueError(
                "place_world_xyz 位于 world，但 BaseState 不在 world，且仓库没有坐标转换接口"
            )
        goal_x, goal_y, goal_yaw = self._stand_off_pose(
            place_x, place_y, base.position_xyz
        )
        return NavGoal(
            f"place-{task.task_id}-{now}",
            "place",
            (goal_x, goal_y, goal_yaw),
            "world",
            _POSITION_TOLERANCE_M,
            _YAW_TOLERANCE_RAD,
            now + _GOAL_TIMEOUT_NS,
        )

    def update(
        self, base: BaseState, goal: NavGoal, timestamp_ns: int
    ) -> tuple[BaseCommand, NavigationStatus]:
        """使用实际 Odom 推进一次导航控制周期。

        控制分为“朝向停车点”“向停车点前进”“原地对准最终 yaw”和“到达”四种情况。
        输出是带短 TTL 的 ``BaseCommand(v,w)`` 和基于同一份 Odom 的
        ``NavigationStatus``；每次调用只生成一个周期的建议。

        只有实际 Odom 同时满足位置与角度容差时才能 ``success=True``；deadline 过期、
        frame 不一致或目标无效时必须失败。生成 ``NavGoal`` 或速度命令都不代表已经到达。
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
            if now > deadline:
                return self._stopped(
                    goal.goal_id,
                    now,
                    "timeout",
                    "导航目标已超过 deadline",
                    valid=False,
                )
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

            # 第二阶段：所有输入确认可靠后，再计算位置误差和最终姿态误差。
            base_x, base_y = _read_xy(base.position_xyz, "base.position_xyz")
            distance_error = distance_xy((base_x, base_y), (goal_x, goal_y))
            final_yaw_error = wrap_to_pi(goal_yaw - base.yaw)
            if distance_error <= position_tolerance:
                # 已进入位置容差：停止平移，只调整目标要求的最终 yaw。
                if abs(final_yaw_error) <= yaw_tolerance:
                    return (
                        BaseCommand(0.0, 0.0, now, now + _COMMAND_TTL_NS),
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
                v = 0.0
                w = self._clamp(_ANGULAR_KP * final_yaw_error, _MAX_ABS_W_RADPS)
                state = "aligning_final_yaw"
                control_yaw_error = final_yaw_error
            else:
                # 尚未到达停车点：先朝向停车点，再按距离比例向前推进。
                bearing = math.atan2(goal_y - base_y, goal_x - base_x)
                heading_error = wrap_to_pi(bearing - base.yaw)
                control_yaw_error = heading_error
                w = self._clamp(_ANGULAR_KP * heading_error, _MAX_ABS_W_RADPS)
                if abs(heading_error) >= _HEADING_STOP_RAD:
                    # 偏航过大时禁止“边大幅转向边高速前进”。
                    v = 0.0
                    state = "aligning_to_goal"
                else:
                    # 距离比例项负责近目标减速，余弦项进一步抑制带偏角前进。
                    v = min(_MAX_ABS_V_MPS, _LINEAR_KP * distance_error)
                    v *= max(0.0, math.cos(heading_error))
                    state = "moving"
            if not math.isfinite(v) or not math.isfinite(w):
                raise ValueError("导航控制计算得到非有限速度")
            return (
                BaseCommand(v, w, now, now + _COMMAND_TTL_NS),
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

    @classmethod
    def _validate_task(cls, task: TaskSpec, now: int) -> None:
        if not isinstance(task, TaskSpec) or not task.valid:
            raise ValueError("TaskSpec 无效")
        if isinstance(task.task_id, bool) or not isinstance(task.task_id, int):
            raise ValueError("TaskSpec.task_id 必须是整数")
        stamp = cls._timestamp(task.timestamp_ns, "task.timestamp_ns")
        if stamp > now:
            raise ValueError("TaskSpec 时间戳晚于当前周期")

    @classmethod
    def _validate_base(
        cls, base: BaseState, now: int, *, require_fresh: bool
    ) -> None:
        if not isinstance(base, BaseState) or not base.valid:
            raise ValueError("BaseState 无效")
        if not base.frame_id:
            raise ValueError("BaseState.frame_id 不能为空")
        stamp = cls._timestamp(base.timestamp_ns, "base.timestamp_ns")
        if stamp > now:
            raise ValueError("BaseState 时间戳晚于当前周期")
        if require_fresh and now - stamp > _ODOM_MAX_AGE_NS:
            raise ValueError("BaseState/Odom 已过期")
        if len(base.position_xyz) != 3:
            raise ValueError("base.position_xyz 必须包含三项")
        for index, value in enumerate(base.position_xyz):
            _finite_real(value, f"base.position_xyz[{index}]")
        _finite_real(base.yaw, "base.yaw")

    @staticmethod
    def _stand_off_pose(
        target_x: float, target_y: float, base_position: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        base_x, base_y = _read_xy(base_position, "base.position_xyz")
        # 单位向量从底盘指向目标；从目标沿反方向退让，得到底盘中心停车点。
        dx = target_x - base_x
        dy = target_y - base_y
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            raise ValueError("底盘与目标中心重合，无法确定安全退让方向")
        goal_x = target_x - _STANDOFF_M * dx / distance
        goal_y = target_y - _STANDOFF_M * dy / distance
        goal_yaw = math.atan2(target_y - goal_y, target_x - goal_x)
        return goal_x, goal_y, goal_yaw

    @staticmethod
    def _clamp(value: float, absolute_limit: float) -> float:
        return max(-absolute_limit, min(absolute_limit, value))

    @staticmethod
    def _stopped(
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
                timestamp_ns + _COMMAND_TTL_NS,
                valid=valid,
                failure_reason=reason,
            ),
            NavigationStatus(goal_id, state, 0.0, 0.0, False, reason, timestamp_ns),
        )
