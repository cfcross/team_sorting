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

第一版只实现可独立验证的几何函数。完整站位生成、航点导航、精对准和 creep 保留给
底盘2负责人；``NavigationStatus`` 必须根据实际 Odom 判断，不能用命令生成代替到达。
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
    """规则底盘导航器的普通 Python 接口骨架。

    ``NavGoal`` 是要去的底盘站位，``BaseState`` 是 Odom 反馈的实际位置，
    ``BaseCommand`` 是本周期短时有效的 v/w 候选建议，``NavigationStatus`` 则是用实际
    反馈算出的执行状态。站位是机械臂能安全操作时底盘应停的位置，不等于物体中心。

    未来由配置提供速度、加速度和容差。控制器不发布 ``/cmd_vel``，候选命令必须交给
    ``ActionMux``；短 TTL 可防止控制循环中断后继续沿用旧速度。完整算法未实现时三个
    公开方法都明确抛出 ``NotImplementedError``。
    """

    def build_pick_goal(
        self, task: TaskSpec, target: ObjectEstimate3D, base: BaseState, timestamp_ns: int
    ) -> NavGoal:
        """待底盘2根据物体中心反算抓取时的底盘 ``NavGoal``。

        ``ObjectEstimate3D.position_xyz`` 是物体中心，不是底盘停车点。后续实现需结合
        桌面/货架类型、抓取距离、机械臂可达范围、底盘朝向和碰撞余量生成站位，并确认
        ``task/target/base`` 的 frame 关系。当前不使用固定坐标兜底。
        """

        raise NotImplementedError("抓取站位生成尚未实现，请由底盘2负责人完成")

    def build_place_goal(self, task: TaskSpec, base: BaseState, timestamp_ns: int) -> NavGoal:
        """待底盘2根据 ``TaskSpec.place_world_xyz`` 反算放置站位。

        ``place_world_xyz`` 是物体最终中心，单位米，不是底盘停车点。后续实现需结合
        放置区域、机械臂可达范围和安全距离生成 ``NavGoal``，并明确目标与 Odom 的 frame
        关系；不能把物体中心直接当作底盘目标。
        """

        raise NotImplementedError("放置站位生成尚未实现，请由底盘2负责人完成")

    def update(
        self, base: BaseState, goal: NavGoal, timestamp_ns: int
    ) -> tuple[BaseCommand, NavigationStatus]:
        """待底盘2用实际 Odom 推进一次导航控制周期。

        后续根据 ``BaseState`` 与 ``NavGoal`` 计算 XY 距离误差和 yaw 误差，逐步实现粗
        导航、转向、直行、靠近目标后的精对准，以及低速小步接近的 creep。输出应是带
        短 TTL 的 ``BaseCommand(v,w)`` 和 ``NavigationStatus``。

        只有实际 Odom 同时满足位置与角度容差时才能 ``success=True``；deadline 过期、
        frame 不一致或目标无效时必须失败。生成 ``NavGoal`` 或速度命令都不代表已经到达。
        """

        raise NotImplementedError("底盘导航控制尚未实现，请由底盘2负责人完成")
