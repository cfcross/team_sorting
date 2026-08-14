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

当前实现包含基础站位、静态AABB安全路线和单周期比例控制；它不进行动态重规划、
坐标变换或ROS发布。``NavigationStatus`` 必须根据实际 Odom 判断，不能用命令生成代替到达。
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


def _strict_finite_float(value: object, field_name: str) -> float:
    """校验静态场景标量，只接受内建 ``int/float`` 且拒绝 ``bool``。"""

    if type(value) not in (int, float):
        raise ValueError(f"{field_name}必须是严格int或float有限数")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name}必须是严格int或float有限数") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field_name}必须是严格int或float有限数")
    return converted


@dataclass(frozen=True)
class _StaticAABB:
    """调用方已确认同一平面内的私有闭合静态边界，四项单位均为米。"""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        for name in ("min_x", "min_y", "max_x", "max_y"):
            object.__setattr__(
                self,
                name,
                _strict_finite_float(getattr(self, name), f"_StaticAABB.{name}"),
            )
        if self.min_x >= self.max_x:
            raise ValueError("_StaticAABB.min_x必须小于max_x")
        if self.min_y >= self.max_y:
            raise ValueError("_StaticAABB.min_y必须小于max_y")


def _inflate_aabb(bounds: _StaticAABB, inflation_m: float) -> _StaticAABB:
    """将闭合静态AABB向四侧膨胀指定米数，不注入任何安全默认值。"""

    if not isinstance(bounds, _StaticAABB):
        raise ValueError("bounds必须是_StaticAABB")
    if type(inflation_m) not in (int, float):
        raise ValueError("inflation_m必须是非负有限数")
    try:
        inflation = float(inflation_m)
    except (OverflowError, ValueError) as exc:
        raise ValueError("inflation_m必须是非负有限数") from exc
    if not math.isfinite(inflation) or inflation < 0.0:
        raise ValueError("inflation_m必须是非负有限数")
    expanded = (
        bounds.min_x - inflation,
        bounds.min_y - inflation,
        bounds.max_x + inflation,
        bounds.max_y + inflation,
    )
    if not all(math.isfinite(value) for value in expanded):
        raise ValueError("膨胀AABB在当前浮点精度下无法表示")
    if inflation > 0.0 and not (
        expanded[0] < bounds.min_x
        and expanded[1] < bounds.min_y
        and expanded[2] > bounds.max_x
        and expanded[3] > bounds.max_y
    ):
        raise ValueError("膨胀AABB边界在当前浮点精度下无法表示")
    return _StaticAABB(expanded[0], expanded[1], expanded[2], expanded[3])


def _strict_xy(point_xy: object, field_name: str) -> tuple[float, float]:
    """读取恰好二维的内建数值坐标，不附加frame或修改输入。"""

    if type(point_xy) not in (tuple, list):
        raise ValueError(f"{field_name}必须恰好包含两个严格int或float有限数")
    try:
        if len(point_xy) != 2:  # type: ignore[arg-type]
            raise ValueError(f"{field_name}必须恰好包含两个严格int或float有限数")
        x_value = point_xy[0]  # type: ignore[index]
        y_value = point_xy[1]  # type: ignore[index]
    except (TypeError, KeyError, IndexError, OverflowError) as exc:
        raise ValueError(
            f"{field_name}必须恰好包含两个严格int或float有限数"
        ) from exc
    return (
        _strict_finite_float(x_value, f"{field_name}.x"),
        _strict_finite_float(y_value, f"{field_name}.y"),
    )


def _point_intersects_aabb(point_xy: object, bounds: _StaticAABB) -> bool:
    """判断同一平面内的点是否落在闭合AABB内部或边界上。"""

    if not isinstance(bounds, _StaticAABB):
        raise ValueError("bounds必须是_StaticAABB")
    point_x, point_y = _strict_xy(point_xy, "point_xy")
    return (
        bounds.min_x <= point_x <= bounds.max_x
        and bounds.min_y <= point_y <= bounds.max_y
    )


def _segment_intersects_aabb_values(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    bounds: _StaticAABB,
) -> bool:
    """对已验证坐标执行闭合线段与AABB的解析区间裁剪。"""

    interval_min = 0.0
    interval_max = 1.0
    for start, end, lower, upper in (
        (start_xy[0], end_xy[0], bounds.min_x, bounds.max_x),
        (start_xy[1], end_xy[1], bounds.min_y, bounds.max_y),
    ):
        delta = end - start
        if not math.isfinite(delta):
            raise ValueError("线段差值在当前浮点精度下无法表示")
        if delta == 0.0:
            if start < lower or start > upper:
                return False
            continue
        lower_delta = lower - start
        upper_delta = upper - start
        if not math.isfinite(lower_delta) or not math.isfinite(upper_delta):
            raise ValueError("线段裁剪在当前浮点精度下无法表示")
        first = lower_delta / delta
        second = upper_delta / delta
        if not math.isfinite(first) or not math.isfinite(second):
            raise ValueError("线段裁剪在当前浮点精度下无法表示")
        entry = min(first, second)
        exit_ = max(first, second)
        interval_min = max(interval_min, entry)
        interval_max = min(interval_max, exit_)
        if interval_min > interval_max:
            return False
    return True


def _segment_intersects_aabb(
    start_xy: object, end_xy: object, bounds: _StaticAABB
) -> bool:
    """判断闭合线段是否穿过、接触或位于同一平面的闭合AABB。"""

    if not isinstance(bounds, _StaticAABB):
        raise ValueError("bounds必须是_StaticAABB")
    start = _strict_xy(start_xy, "start_xy")
    end = _strict_xy(end_xy, "end_xy")
    return _segment_intersects_aabb_values(start, end, bounds)


def _segment_intersects_any_aabb(
    start_xy: object,
    end_xy: object,
    obstacles: object,
) -> bool:
    """按输入顺序判断闭合线段是否接触任一私有静态AABB。"""

    start = _strict_xy(start_xy, "start_xy")
    end = _strict_xy(end_xy, "end_xy")
    if not isinstance(obstacles, (tuple, list)) or any(
        not isinstance(bounds, _StaticAABB) for bounds in obstacles
    ):
        raise ValueError("obstacles必须是_StaticAABB序列")
    return any(
        _segment_intersects_aabb_values(start, end, bounds) for bounds in obstacles
    )


def _plan_static_aabb_route(
    start_xy: object,
    goal_xy: object,
    obstacles: object,
    waypoint_margin_m: object,
    max_intermediate_waypoints: object,
) -> tuple[tuple[float, float], ...] | None:
    """在已膨胀静态AABB间选择确定性的最短折线路线。"""

    start = _strict_xy(start_xy, "start_xy")
    goal = _strict_xy(goal_xy, "goal_xy")
    if type(obstacles) is not tuple:
        raise ValueError("obstacles必须是内建tuple")
    if any(not isinstance(bounds, _StaticAABB) for bounds in obstacles):
        raise ValueError("obstacles成员必须是_StaticAABB")
    obstacles = tuple(
        sorted(
            obstacles,
            key=lambda bounds: (
                bounds.min_x,
                bounds.min_y,
                bounds.max_x,
                bounds.max_y,
            ),
        )
    )
    margin = _strict_finite_float(waypoint_margin_m, "waypoint_margin_m")
    if margin <= 0.0:
        raise ValueError("waypoint_margin_m必须严格大于0")
    if type(max_intermediate_waypoints) is not int:
        raise ValueError("max_intermediate_waypoints必须是非负内建int")
    if max_intermediate_waypoints < 0:
        raise ValueError("max_intermediate_waypoints必须是非负内建int")
    waypoint_limit = max_intermediate_waypoints

    raw_candidates: list[tuple[float, float]] = []
    for bounds in obstacles:
        min_x = bounds.min_x - margin
        min_y = bounds.min_y - margin
        max_x = bounds.max_x + margin
        max_y = bounds.max_y + margin
        if not all(math.isfinite(value) for value in (min_x, min_y, max_x, max_y)):
            raise ValueError("waypoint候选在当前浮点精度下无法表示")
        if not (
            min_x < bounds.min_x
            and min_y < bounds.min_y
            and max_x > bounds.max_x
            and max_y > bounds.max_y
        ):
            raise ValueError("waypoint margin在当前浮点精度下无法真实外移")
        raw_candidates.extend(
            (
                (min_x, min_y),
                (min_x, max_y),
                (max_x, min_y),
                (max_x, max_y),
            )
        )

    ordered_candidates = sorted(raw_candidates)
    candidates: list[tuple[float, float]] = []
    for candidate in ordered_candidates:
        if candidates and candidate == candidates[-1]:
            continue
        if candidate == start or candidate == goal:
            continue
        if any(_point_intersects_aabb(candidate, bounds) for bounds in obstacles):
            continue
        candidates.append(candidate)

    if any(_point_intersects_aabb(start, bounds) for bounds in obstacles):
        return None
    if any(_point_intersects_aabb(goal, bounds) for bounds in obstacles):
        return None

    def _segment_length(
        first: tuple[float, float], second: tuple[float, float]
    ) -> float:
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        if not math.isfinite(dx) or not math.isfinite(dy):
            raise ValueError("路线坐标差在当前浮点精度下无法表示")
        length = math.hypot(dx, dy)
        if not math.isfinite(length):
            raise ValueError("路线段长度在当前浮点精度下无法表示")
        return length

    if start == goal:
        if _segment_intersects_any_aabb(start, goal, obstacles):
            raise ValueError("内部相同起终点与障碍物相交")
        _segment_length(start, goal)
        return (goal,)

    nodes = (start, goal, *candidates)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for first_index in range(len(nodes)):
        for second_index in range(first_index + 1, len(nodes)):
            if _segment_intersects_any_aabb(
                nodes[first_index], nodes[second_index], obstacles
            ):
                continue
            length = _segment_length(nodes[first_index], nodes[second_index])
            if length <= 0.0:
                raise ValueError("不同节点路线段长度在当前浮点精度下无法可靠表示")
            adjacency[first_index].append((second_index, length))
            adjacency[second_index].append((first_index, length))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: nodes[item[0]])

    def _validate_route(
        route: tuple[tuple[float, float], ...]
    ) -> tuple[tuple[float, float], ...]:
        if not route or route[-1] != goal:
            raise ValueError("内部路线未以goal结束")
        intermediates = route[:-1]
        if len(intermediates) > waypoint_limit:
            raise ValueError("内部路线超过waypoint上限")
        if any(point == start for point in intermediates):
            raise ValueError("内部路线重复start")
        if len(route) != len(tuple(dict.fromkeys(route))):
            raise ValueError("内部路线包含重复点或环")
        for point in route:
            _strict_xy(point, "route_point")
        for point in intermediates:
            if any(_point_intersects_aabb(point, bounds) for bounds in obstacles):
                raise ValueError("内部waypoint位于障碍物内")
        if any(_point_intersects_aabb(goal, bounds) for bounds in obstacles):
            raise ValueError("内部goal位于障碍物内")
        total = 0.0
        previous = start
        for point in route:
            if _segment_intersects_any_aabb(previous, point, obstacles):
                raise ValueError("内部路线段与障碍物相交")
            segment_length = _segment_length(previous, point)
            if segment_length <= 0.0:
                raise ValueError("返回路线段长度在当前浮点精度下无法可靠表示")
            updated_total = total + segment_length
            if not math.isfinite(updated_total) or updated_total <= total:
                raise ValueError("正路线段长度在当前浮点精度下无法严格累计")
            total = updated_total
            previous = point
        return route

    if any(neighbor == 1 for neighbor, _length in adjacency[0]):
        return _validate_route((goal,))
    if waypoint_limit == 0:
        return None

    effective_limit = min(waypoint_limit, len(candidates))
    best_key: tuple[float, tuple[tuple[float, float], ...]] | None = None
    best_route: tuple[tuple[float, float], ...] | None = None

    states: dict[int, tuple[float, tuple[tuple[float, float], ...]]] = {
        0: (0.0, ())
    }
    for used_intermediate_count in range(effective_limit + 1):
        next_states: dict[int, tuple[float, tuple[tuple[float, float], ...]]] = {}
        for node_index, (prefix_length, intermediate_points) in states.items():
            for neighbor_index, edge_length in adjacency[node_index]:
                if neighbor_index == 0:
                    continue
                if edge_length <= 0.0:
                    raise ValueError("路线图边长在当前浮点精度下无法可靠表示")
                total_length = prefix_length + edge_length
                if not math.isfinite(total_length) or total_length <= prefix_length:
                    raise ValueError("正路线段长度在当前浮点精度下无法严格累计")
                if neighbor_index == 1:
                    key = (total_length, intermediate_points)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_route = (*intermediate_points, goal)
                    continue
                if used_intermediate_count >= effective_limit:
                    continue
                candidate_points = (*intermediate_points, nodes[neighbor_index])
                candidate_key = (total_length, candidate_points)
                existing = next_states.get(neighbor_index)
                if existing is None or candidate_key < existing:
                    next_states[neighbor_index] = candidate_key
        states = next_states
        if not states:
            break
    if best_route is None:
        return None
    return _validate_route(best_route)


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
        offset_x = _finite_real(
            self._config.standoff_m * dx / distance, "站位X退让量"
        )
        offset_y = _finite_real(
            self._config.standoff_m * dy / distance, "站位Y退让量"
        )
        goal_x = _finite_real(target_x - offset_x, "站位X坐标")
        goal_y = _finite_real(target_y - offset_y, "站位Y坐标")
        facing_x = _finite_real(target_x - goal_x, "目标朝向X分量")
        facing_y = _finite_real(target_y - goal_y, "目标朝向Y分量")
        goal_yaw = _finite_real(math.atan2(facing_y, facing_x), "站位yaw")
        return goal_x, goal_y, goal_yaw

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


@dataclass(frozen=True)
class _StaticObstacle:
    """带稳定诊断名称的私有odom静态障碍物。"""

    name: str
    bounds: _StaticAABB

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("静态障碍物name必须是非空字符串")
        if not isinstance(self.bounds, _StaticAABB):
            raise ValueError("静态障碍物bounds必须是_StaticAABB")
        object.__setattr__(self, "name", self.name.strip())


@dataclass(frozen=True)
class _StaticRouteConfig:
    """3.5静态路线的显式私有配置；输入障碍物尚未膨胀。"""

    frame_id: str
    inflation_radius_m: float
    waypoint_margin_m: float
    max_intermediate_waypoints: int
    static_obstacles: tuple[_StaticObstacle, ...]

    def __post_init__(self) -> None:
        if self.frame_id != _ODOM_FRAME_ID or type(self.frame_id) is not str:
            raise ValueError("静态路线frame_id必须精确为odom")
        inflation = _strict_finite_float(
            self.inflation_radius_m, "inflation_radius_m"
        )
        margin = _strict_finite_float(self.waypoint_margin_m, "waypoint_margin_m")
        if inflation <= 0.0 or margin <= 0.0:
            raise ValueError("inflation_radius_m和waypoint_margin_m必须严格大于0")
        if (
            type(self.max_intermediate_waypoints) is not int
            or self.max_intermediate_waypoints < 0
        ):
            raise ValueError("max_intermediate_waypoints必须是非负内建int")
        if type(self.static_obstacles) is not tuple or not self.static_obstacles:
            raise ValueError("static_obstacles必须是非空内建tuple")
        if any(not isinstance(item, _StaticObstacle) for item in self.static_obstacles):
            raise ValueError("static_obstacles成员必须是_StaticObstacle")
        names = tuple(item.name for item in self.static_obstacles)
        if len(set(names)) != len(names):
            raise ValueError("static_obstacles名称不得重复")
        object.__setattr__(self, "inflation_radius_m", inflation)
        object.__setattr__(self, "waypoint_margin_m", margin)

    def inflated_obstacles(self) -> tuple[_StaticAABB, ...]:
        return tuple(
            _inflate_aabb(item.bounds, self.inflation_radius_m)
            for item in self.static_obstacles
        )


@dataclass(frozen=True)
class _StaticRoutePlan:
    """不可变的分段路线；最终元素必须复用原始3.4 NavGoal对象。"""

    final_goal_key: tuple[object, ...]
    goals: tuple[NavGoal, ...]
    route_points: tuple[tuple[float, float], ...]
    generation: int


class _StaticRouteNavigator:
    """在单个NavigationController上增加私有静态路线和整路径共享预算。"""

    def __init__(
        self,
        controller: NavigationController,
        config: _StaticRouteConfig,
    ) -> None:
        if not isinstance(controller, NavigationController):
            raise TypeError("controller必须是NavigationController")
        if not isinstance(config, _StaticRouteConfig):
            raise TypeError("config必须是_StaticRouteConfig")
        self._controller = controller
        self._config = config
        self._generation = 0
        self.reset()

    @property
    def plan(self) -> _StaticRoutePlan | None:
        return self._plan

    @property
    def current_goal(self) -> NavGoal | None:
        if self._plan is None:
            return None
        if not 0 <= self._current_index < len(self._plan.goals):
            raise ValueError("静态路线waypoint索引非法")
        return self._plan.goals[self._current_index]

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def final_goal(self) -> NavGoal | None:
        return None if self._plan is None else self._plan.goals[-1]

    @property
    def diagnostic(self) -> str:
        return self._diagnostic

    @property
    def waiting_for_new_odom(self) -> bool:
        return self._waiting_for_new_odom

    @property
    def progressed_this_update(self) -> bool:
        return self._progressed_this_update

    def build_pick_goal(
        self,
        task: TaskSpec,
        target: ObjectEstimate3D,
        base: BaseState,
        timestamp_ns: int,
    ) -> NavGoal:
        return self._controller.build_pick_goal(task, target, base, timestamp_ns)

    def build_place_goal(
        self, task: TaskSpec, base: BaseState, timestamp_ns: int
    ) -> NavGoal:
        return self._controller.build_place_goal(task, base, timestamp_ns)

    def build_return_goal(self, base: BaseState, timestamp_ns: int) -> NavGoal:
        return self._controller.build_return_goal(base, timestamp_ns)

    @staticmethod
    def _goal_key(goal: NavGoal) -> tuple[object, ...]:
        return (
            goal.goal_id,
            goal.goal_type,
            goal.pose_xyyaw,
            goal.frame_id,
            goal.position_tolerance,
            goal.yaw_tolerance,
            goal.deadline_ns,
            goal.valid,
            goal.failure_reason,
        )

    def reset(self) -> None:
        self._controller.reset()
        self._plan: _StaticRoutePlan | None = None
        self._current_index = 0
        self._last_progress_odom_timestamp_ns: int | None = None
        self._waiting_for_new_odom = False
        self._progressed_this_update = False
        self._path_elapsed_ns = 0
        self._path_checkpoint_ns: int | None = None
        self._control_granted_since_checkpoint = False
        self._path_terminal = False
        self._diagnostic = ""

    def _advance_path_clock(self, timestamp_ns: int) -> None:
        now = NavigationController._timestamp(timestamp_ns, "timestamp_ns")
        checkpoint = self._path_checkpoint_ns
        if checkpoint is None:
            self._path_checkpoint_ns = now
            return
        if now < checkpoint:
            self._control_granted_since_checkpoint = False
            raise ValueError("整路径控制权回报时间不得倒退")
        if self._control_granted_since_checkpoint and not self._path_terminal:
            self._path_elapsed_ns += now - checkpoint
        self._path_checkpoint_ns = now

    def record_control_result(self, timestamp_ns: int, control_granted: bool) -> None:
        if type(control_granted) is not bool:
            self._control_granted_since_checkpoint = False
            raise ValueError("control_granted必须是严格bool")
        self._advance_path_clock(timestamp_ns)
        self._control_granted_since_checkpoint = bool(
            control_granted and self._plan is not None and not self._path_terminal
        )
        self._controller.record_control_result(timestamp_ns, control_granted)

    def _revoke_control_without_timestamp(self) -> None:
        self._control_granted_since_checkpoint = False
        self._controller._revoke_control_without_timestamp()

    def _odom_can_advance_route(self, base: BaseState) -> bool:
        """一个Odom反馈身份最多允许一次路线进度变化。"""

        odom_timestamp_ns = NavigationController._timestamp(
            base.timestamp_ns, "base.timestamp_ns"
        )
        consumed = self._last_progress_odom_timestamp_ns
        if consumed is None:
            return True
        if odom_timestamp_ns < consumed:
            raise ValueError("路线进度Odom.timestamp_ns相对已消费反馈倒退")
        return odom_timestamp_ns > consumed

    def plan_route(
        self, base: BaseState, final_goal: NavGoal, timestamp_ns: int
    ) -> _StaticRoutePlan:
        now = NavigationController._timestamp(timestamp_ns, "timestamp_ns")
        self._controller._validate_base(base, now, require_fresh=True)
        if base.frame_id != self._config.frame_id:
            raise ValueError("静态路线起点必须是odom BaseState")
        if not isinstance(final_goal, NavGoal) or not final_goal.valid:
            raise ValueError("静态路线最终NavGoal无效")
        if final_goal.frame_id != self._config.frame_id:
            raise ValueError("静态路线最终NavGoal必须位于odom")
        if len(final_goal.pose_xyyaw) != 3:
            raise ValueError("静态路线最终NavGoal.pose_xyyaw必须包含三项")
        start = _strict_xy(base.position_xyz[:2], "静态路线起点")
        goal = _strict_xy(final_goal.pose_xyyaw[:2], "静态路线终点")
        _finite_real(final_goal.pose_xyyaw[2], "静态路线最终yaw")
        inflated = self._config.inflated_obstacles()
        route = _plan_static_aabb_route(
            start,
            goal,
            inflated,
            self._config.waypoint_margin_m,
            self._config.max_intermediate_waypoints,
        )
        if route is None:
            raise ValueError(
                "无确定性安全静态路线；起点或3.4最终目标可能位于膨胀障碍物内"
            )
        if not route or route[-1] != goal or len(set(route)) != len(route):
            raise ValueError("静态路线结果为空、未保留最终目标或含重复点")
        goals: list[NavGoal] = []
        self._generation += 1
        for index, point in enumerate(route):
            if index == len(route) - 1:
                goals.append(final_goal)
                continue
            next_point = route[index + 1]
            yaw = wrap_to_pi(math.atan2(next_point[1] - point[1], next_point[0] - point[0]))
            goals.append(
                NavGoal(
                    goal_id=(
                        f"route-{self._generation}-{final_goal.goal_id}-waypoint-{index}"
                    ),
                    goal_type="intermediate",
                    pose_xyyaw=(point[0], point[1], yaw),
                    frame_id=self._config.frame_id,
                    position_tolerance=self._controller._config.position_tolerance_m,
                    yaw_tolerance=self._controller._config.yaw_tolerance_rad,
                    deadline_ns=final_goal.deadline_ns,
                )
            )
        plan = _StaticRoutePlan(
            final_goal_key=self._goal_key(final_goal),
            goals=tuple(goals),
            route_points=route,
            generation=self._generation,
        )
        self.reset()
        self._plan = plan
        self._path_checkpoint_ns = now
        return plan

    def update(
        self, base: BaseState, expected_goal: NavGoal, timestamp_ns: int
    ) -> tuple[BaseCommand, NavigationStatus]:
        now = NavigationController._timestamp(timestamp_ns, "timestamp_ns")
        self._controller._validate_base(base, now, require_fresh=True)
        if base.frame_id != self._config.frame_id:
            raise ValueError("静态路线运行BaseState必须位于odom")
        if self._plan is None:
            raise ValueError("静态路线尚未规划")
        if self._goal_key(self._plan.goals[-1]) != self._plan.final_goal_key:
            raise ValueError("静态路线最终目标身份已损坏")
        current = self.current_goal
        if current is None or expected_goal is not current:
            raise ValueError("活动NavGoal与静态路线当前段不一致")
        self._waiting_for_new_odom = False
        self._progressed_this_update = False
        odom_can_advance = self._odom_can_advance_route(base)
        self._advance_path_clock(now)
        if self._path_elapsed_ns >= self._controller._config.goal_timeout_ns:
            self._path_terminal = True
            self._control_granted_since_checkpoint = False
            self._diagnostic = "整条静态路径有效控制预算达到goal_timeout_ns"
            return self._controller._stopped(
                current.goal_id, now, "timeout", self._diagnostic, valid=False
            )
        if not odom_can_advance:
            self._waiting_for_new_odom = True
            distance = distance_xy(base.position_xyz, current.pose_xyyaw)
            return (
                BaseCommand(
                    0.0,
                    0.0,
                    now,
                    now + self._controller._config.command_ttl_ns,
                ),
                NavigationStatus(
                    current.goal_id,
                    "moving",
                    distance,
                    0.0,
                    False,
                    "等待timestamp严格更新的Odom后再推进静态路线",
                    now,
                ),
            )
        if self._current_index < len(self._plan.goals) - 1:
            distance = distance_xy(base.position_xyz, current.pose_xyyaw)
            if distance <= current.position_tolerance:
                self._current_index += 1
                self._controller.reset()
                next_goal = self.current_goal
                assert next_goal is not None
                self._last_progress_odom_timestamp_ns = base.timestamp_ns
                self._progressed_this_update = True
                return (
                    BaseCommand(
                        0.0,
                        0.0,
                        now,
                        now + self._controller._config.command_ttl_ns,
                    ),
                    NavigationStatus(
                        next_goal.goal_id,
                        "moving",
                        distance,
                        0.0,
                        False,
                        "中间waypoint到达，仅推进静态路线索引",
                        now,
                    ),
                )
        command, status = self._controller.update(base, current, now)
        if status.state in {"failed", "timeout"}:
            self._waiting_for_new_odom = False
            self._path_terminal = True
            self._control_granted_since_checkpoint = False
            self._diagnostic = status.failure_reason
        elif (
            self._current_index == len(self._plan.goals) - 1
            and status.success
            and status.state == "arrived"
        ):
            self._waiting_for_new_odom = False
            self._last_progress_odom_timestamp_ns = base.timestamp_ns
            self._path_terminal = True
            self._control_granted_since_checkpoint = False
        return command, status
