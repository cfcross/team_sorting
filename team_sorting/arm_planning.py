"""双臂抓放规划和官方 KDL 薄适配层。

FK（正运动学）根据关节角计算末端在哪里；IK（逆运动学）根据末端希望到达的位置和
方向反求关节目标。本文件的数据流是：

``ObjectEstimate3D`` 当前物体中心 / ``TaskSpec.place_world_xyz`` 放置后的物体中心
→ ``ArmPlanner`` 生成左右末端 ``Pose3D``
→ ``OfficialKDLAdapter`` 调用官方 ``MMK2Kdl``
→ ``IKResult``（slide + 左臂6轴 + 右臂6轴）
→ ``ArmPlanner`` 嵌入17维 ``JointWaypoint``
→ ``JointTrajectory``
→ ``arm_execution`` 插值、执行并依据实际反馈判断结果。

物体中心只有位置，不是夹爪末端位姿；夹爪还需要方向、双臂间距和安全偏移。官方双臂
运动学向量是13维 ``[slide,left×6,right×6]``，单臂是7维 ``[slide,arm×6]``；团队
17维路点还包含head和左右gripper，未控制关节必须保持实际反馈或使用
``controlled_mask=False``，不能补全为零。

本文件不复制官方解析IK，不导入 ``rclpy``，不发布或执行机械臂命令，也不判断抓取
成功。官方模块和 NumPy 均延迟导入；完整抓放Pose与轨迹规划仍由机械臂1负责人实现。
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  新手阅读指南                                                   ║
# ║                                                                ║
# ║  本文件做三件事：                                               ║
# ║  1. OfficialKDLAdapter —— 调用比赛官方的机械臂运动学求解器       ║
# ║     （告诉机械臂"手能伸到哪里"）                                 ║
# ║  2. ArmPlanner —— 规划抓取和放置的完整动作流程                   ║
# ║     （决定"先伸到这里、再抓住、再抬起来、再撤离"）               ║
# ║  3. 辅助函数 —— 校验数据、转换坐标、检查安全                     ║
# ║                                                                ║
# ║  阅读顺序建议：                                                  ║
# ║  常量区 → _finite_vector → OfficialKDLAdapter →                 ║
# ║  ArmPlanner.plan_grasp → ArmPlanner.plan_place                  ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import importlib
import math
from numbers import Real
import os
from pathlib import Path
import sys
from typing import Any, Optional

from .interfaces import (
    GraspTarget,
    IKResult,
    JointTrajectory,
    JointWaypoint,
    ObjectEstimate3D,
    PlaceTarget,
    Pose3D,
    RobotJointState,
    TaskSpec,
)


_KDL_TARGET_FRAME = "footprint"
# ──────────────────────────────────────────────────────────────────
# 临时常量：机器人底盘在 footprint 系中的固定位置（单位米）
# TODO: 待系统提供正式 TF 后移除本常量，改用 TF 动态查询
# ──────────────────────────────────────────────────────────────────
_FOOTPRINT_ROBOT_XY: tuple[float, float] = (-0.70, 0.55)
# ──────────────────────────────────────────────────────────────────
# 包装盒尺寸（长24cm × 宽16cm × 高19cm，单位米）
# 机械臂1负责人可根据实际硬件调整
# ──────────────────────────────────────────────────────────────────
_BOX_LENGTH: float = 0.24
_BOX_WIDTH: float = 0.16
_BOX_HEIGHT: float = 0.19
_BOX_HALF_WIDTH: float = 0.08
_BOX_HALF_HEIGHT: float = 0.095

# ──────────────────────────────────────────────────────────────────
# 夹爪控制量（0~1 范围，具体开/闭语义以官方协议为准）
# ──────────────────────────────────────────────────────────────────
_GRIPPER_OPEN: float = 0.5
_GRIPPER_CLOSED: float = 0.0

# ──────────────────────────────────────────────────────────────────
# 默认抓放几何偏移（单位米）——基于包装盒尺寸计算
# ──────────────────────────────────────────────────────────────────
_PRE_OFFSET_Z: float = 0.20      # 预抓取/预放置阶段在物体上方的安全高度
_LIFT_DELTA: float = 0.20        # 抓取后试抬竖向增量
_RETREAT_DX: float = -0.15       # 撤离水平后退距离
_RETREAT_DZ: float = 0.10        # 撤离额外竖直增量
_SETTLE_TIME: float = 1.0        # 放置后稳定等待时间

# ──────────────────────────────────────────────────────────────────
# 几何安全检查限制（单位米）——防止生成不可达或危险动作
# ──────────────────────────────────────────────────────────────────
_MAX_GRASP_DIST: float = 1.5     # 最大抓取距离（物体不能太远）
_MAX_LIFT_HEIGHT: float = 1.5    # 最大抬升高度
_MAX_RETREAT_DIST: float = 0.5   # 最大撤退距离
_MAX_PLACE_DESCENT: float = 0.5  # 最大放置下降距离

# ──────────────────────────────────────────────────────────────────
# 轨迹优化参数
# ──────────────────────────────────────────────────────────────────
_JOINT_DELTA_THRESHOLD: float = 0.5  # 关节跳变检测阈值（弧度）
_BASE_TIME: float = 1.0              # 基础每阶段时间（秒）
_TIME_PER_RADIAN: float = 2.0        # 每弧度关节变化所需时间（秒）
_MIN_STAGE_TIME: float = 0.5         # 最小阶段时间
_MAX_STAGE_TIME: float = 5.0
# ──────────────────────────────────────────────────────────────────
# 夹爪闭合等待时间（秒）——保证箱体夹紧后再抬升
# ──────────────────────────────────────────────────────────────────
_CLOSE_WAIT_TIME: float = 0.5

# ──────────────────────────────────────────────────────────────────
# 滑轨线速度（m/s）——用于与关节角速度分离计算阶段时间
# ──────────────────────────────────────────────────────────────────
_SLIDE_SPEED: float = 0.1

# ──────────────────────────────────────────────────────────────────
# 放置表面默认偏移（米）——当无法获取目标表面高度时使用
# 正值为向上补偿（释放位略高于最终中心），保证不插入目标区域
# ──────────────────────────────────────────────────────────────────
_PLACE_SURFACE_OFFSET: float = 0.02
         # 最大阶段时间



def _finite_vector(values: Any, expected_length: int, name: str) -> tuple[float, ...]:
    """【数据清洗】检查输入是否是合法数值向量。

    通俗理解：像一个严格的安检员，检查传入的数据：
    - 长度对不对？（比如关节必须17个值）
    - 每一项是不是真正的数字？（拒绝布尔值True/False冒充0/1）
    - 有没有非法值？（拒绝NaN/Inf）

    通过检查才放行，否则直接报错，不让脏数据污染后续计算。
    """

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name}必须恰好包含{expected_length}项真实有限数")
    try:
        if len(values) != expected_length:
            raise ValueError(f"{name}必须恰好包含{expected_length}项")
        raw_values = tuple(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}必须恰好包含{expected_length}项") from exc
    converted: list[float] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name}[{index}]必须是真实数，不能使用bool或字符串")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}]不能包含NaN或Inf")
        converted.append(number)
    return tuple(converted)


def _solver_slide_limits(solver: Any) -> Optional[tuple[float, float]]:
    """读取官方求解器公开的slide限位；未暴露时不在团队代码中猜测。"""

    spine = getattr(solver, "spine", None)
    limits = getattr(spine, "joint_limits", None) if spine is not None else None
    if limits is None:
        return None
    lower, upper = _finite_vector(limits, 2, "官方spine.joint_limits")
    if lower > upper:
        raise ValueError("官方spine.joint_limits下界大于上界")
    return lower, upper


def _first_ik_solution(
    solutions: Any, expected_length: int, np: Any
) -> tuple[Optional[tuple[float, ...]], str]:
    """【IK解提取】从官方求解器返回的一堆解中取出第一个可用的。

    通俗理解：官方KDL可能返回0个、1个或多个解。这个函数：
    1. 检查有没有解（没解→失败）
    2. 取出第一个解
    3. 验证解的格式和数值都合法
    4. 合法→返回；不合法→说明原因

    返回两个值：(关节角度元组, 失败原因字符串)
    成功时关节角度不是None，失败原因=""
    失败时关节角度=None，失败原因说明为什么
    """

    if solutions is None:
        return None, "官方 KDL 未找到合法关节解"
    try:
        iterator = iter(solutions)
    except TypeError:
        return None, "官方 KDL 返回结果不可迭代"
    try:
        first = next(iterator)
    except StopIteration:
        return None, "官方 KDL 返回空关节解"
    try:
        array = np.asarray(first)
    except (TypeError, ValueError, OverflowError) as exc:
        return None, f"官方 KDL 首个关节解无法转换为数组：{exc}"
    if array.ndim != 1:
        return None, "官方 KDL 首个关节解必须是一维向量"
    if array.shape[0] != expected_length:
        label = "双臂" if expected_length == 13 else "单臂"
        return None, f"{label} IK 结果长度不是 {expected_length}"
    try:
        # 使用原始一维项校验，避免NumPy把bool与float混排时先静默转换成0.0/1.0。
        solution = _finite_vector(first, expected_length, "官方KDL关节解")
    except ValueError as exc:
        return None, str(exc)
    return solution, ""


# ═══════════════════════════════════════════════════════════════════
# OfficialKDLAdapter：薄适配器（Wrapper）
# 
# 通俗理解：这是一个"翻译官"。
# 我们把"希望机械臂末端到达的位置（xyz坐标+方向）"交给它，
# 它调比赛官方的运动学求解器（MMK2Kdl）算出"每个关节应该转多少度"，
# 然后把结果翻译成我们团队统一的 IKResult 格式返回。
#
# 为什么叫"薄适配器"？因为它只做翻译，不做任何运动学算法。
# IK/FK 算法全在官方代码里，我们不复制也不重新实现。
# ═══════════════════════════════════════════════════════════════════

class OfficialKDLAdapter:
    """官方 MMK2Kdl/ArmKdl 的薄适配器。

    参数是官方代码根目录和 ``MMK2Kdl`` 模块名。输入为已转换到 ``footprint`` 的
    ``Pose3D``、实际17维 ``RobotJointState`` 和固定slide目标；输出 ``IKResult``。
    官方代码先从footprint经过slide和左右肩部安装变换，再调用单臂解析IK，因此这里
    不能把world、odom、base_link或base_footprint仅凭名字当成等价frame。

    ``Pose3D`` 同时提供位置和方向，转换成4×4齐次矩阵后交给KDL；矩阵左上3×3表示
    旋转，最后一列表示平移。实际关节只用作参考解，让求解器倾向选择接近当前姿态的
    分支，并不等于IK目标。slide单位米，机械臂关节单位弧度。
    失败：依赖缺失在 ``self_check`` 抛出 ``RuntimeError``；frame 不符或 IK 无解返回
    ``success=False``，不会返回伪关节角。
    """

    def __init__(self, official_root: str = "", module_name: str = "mmk2_kdl") -> None:
        """保存官方依赖配置，暂不导入 KDL。

        路径可由 ``MATERIAL_SORTING_OFFICIAL_ROOT`` 覆盖。构造没有单位或坐标输出，
        没有比赛环境时也可安全导入和实例化。
        """

        if not isinstance(official_root, str):
            raise ValueError("official_root必须是字符串，可以为空")
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError("module_name必须是非空字符串")
        self.official_root = official_root
        self.module_name = module_name.strip()
        self._solver: Any = None
        self._searched: list[str] = []

    def self_check(self) -> None:
        """延迟导入官方 KDL 并执行一次有限矩阵 FK 自检。

        参数：无。返回：成功时无返回值。
        输入自检向量顺序为 ``[slide,left×6,right×6]``，slide 米、关节弧度。
        失败：找不到 MMK2Kdl/ArmKdl、NumPy 或 FK 结果异常时抛出 ``RuntimeError``，
        错误中列出搜索项和 ``MATERIAL_SORTING_OFFICIAL_ROOT``。
        """

        # 每次自检都从未加载状态开始；本轮失败后不能继续误用上一次成功的求解器。
        self._searched.clear()
        self._solver = None
        root_text = os.getenv("MATERIAL_SORTING_OFFICIAL_ROOT", self.official_root).strip()
        if root_text:
            root = Path(root_text).expanduser()
            # official.root可能是工作区根、material_sorting目录或正式examples目录。
            for path in (
                root,
                root / "material_sorting",
                root / "examples" / "material_sorting",
            ):
                self._searched.append(str(path))
                if path.is_dir() and str(path) not in sys.path:
                    sys.path.insert(0, str(path))
        try:
            module = importlib.import_module(self.module_name)
        except Exception as exc:  # noqa: BLE001 - 官方依赖可能抛出多种导入异常
            raise RuntimeError(
                f"无法导入官方 MMK2Kdl（模块={self.module_name}，搜索={self._searched}）：{exc}。"
                "请设置 MATERIAL_SORTING_OFFICIAL_ROOT 或 config.yaml 的 official.root。"
            ) from exc
        if not hasattr(module, "MMK2Kdl"):
            raise RuntimeError(f"官方模块 {self.module_name} 中不存在 MMK2Kdl")
        try:
            import numpy as np

            solver = module.MMK2Kdl()
            left, right = solver.forward_kinematics(np.zeros(13, dtype=float))
            left_matrix = np.asarray(left)
            right_matrix = np.asarray(right)
            if left_matrix.shape != (4, 4) or right_matrix.shape != (4, 4):
                raise ValueError("FK 结果不是两个 4×4 矩阵")
            if not np.isfinite(left_matrix).all() or not np.isfinite(right_matrix).all():
                raise ValueError("FK 结果包含 NaN/Inf")
        except Exception as exc:  # noqa: BLE001 - 统一为启动自检错误
            raise RuntimeError(f"官方 MMK2Kdl FK 自检失败：{exc}") from exc
        self._solver = solver

    def solve_ik(
        self,
        actual_joints: RobotJointState,
        left_target: Optional[Pose3D] = None,
        right_target: Optional[Pose3D] = None,
        target_slide: Optional[float] = None,
    ) -> IKResult:
        """【核心方法】调用官方KDL求解逆运动学（IK）。

        通俗理解：给一个目标位置，让官方求解器算出机械臂每个关节该转到什么角度。

        输入：实际关节角度（参考用）、左手目标位姿、右手目标位姿、滑轨目标高度
        输出：IKResult（包含13个关节角 + 滑轨位置，或失败原因）

        步骤：
        1. 检查求解器是否已初始化
        2. 验证输入数据合法
        3. 把 Pose3D 转成 4×4 数学矩阵
        4. 调用官方 inverse_kinematics()
        5. 从返回的解中提取第一个可用的
        """

        # ── 步骤1：检查求解器是否已初始化 ──
        if self._solver is None:
            raise RuntimeError("OfficialKDLAdapter 尚未通过 self_check")
        # 先验证完整17维反馈，再读取第0项slide；损坏对象不能让控制调用泄漏索引或类型异常。
        try:
            actual_position = _finite_vector(
                getattr(actual_joints, "position", None),
                17,
                "RobotJointState.position",
            )
        except ValueError as exc:
            # IKResult的target_slide不是Optional；此失败对象没有任何手臂目标且success=False，
            # 其中0.0只是不可执行的占位，不能作为安全保持或控制命令使用。
            return IKResult(0.0, None, None, False, f"实际关节位置无效：{exc}")
        actual_slide = actual_position[0]
        if not actual_joints.valid:
            return IKResult(
                actual_slide,
                None,
                None,
                False,
                actual_joints.failure_reason or "实际关节状态无效",
            )
        if left_target is None and right_target is None:
            return IKResult(actual_slide, None, None, False, "左右末端目标不能同时为空")
        for target in (left_target, right_target):
            if target is not None and target.frame_id != _KDL_TARGET_FRAME:
                return IKResult(
                    actual_slide,
                    None,
                    None,
                    False,
                    "IK 目标必须先转换到官方KDL使用的footprint坐标系，"
                    f"实际frame={target.frame_id!r}",
                )
        try:
            slide = _finite_vector(
                (actual_slide if target_slide is None else target_slide,),
                1,
                "target_slide",
            )[0]
            limits = _solver_slide_limits(self._solver)
        except ValueError as exc:
            return IKResult(actual_slide, None, None, False, str(exc))
        if limits is not None and not limits[0] <= slide <= limits[1]:
            return IKResult(
                slide,
                None,
                None,
                False,
                f"target_slide={slide}超出官方spine.joint_limits={limits}",
            )
        try:
            import numpy as np

            left_matrix = None if left_target is None else _pose_to_matrix(left_target, np)
            right_matrix = None if right_target is None else _pose_to_matrix(right_target, np)
            if left_target is not None and right_target is not None:
                reference = np.asarray(
                    [actual_position[0], *actual_position[3:9], *actual_position[10:16]],
                    dtype=float,
                )
                expected_length = 13
            elif left_target is not None:
                reference = np.asarray(
                    [actual_position[0], *actual_position[3:9]], dtype=float
                )
                expected_length = 7
            else:
                reference = np.asarray(
                    [actual_position[0], *actual_position[10:16]], dtype=float
                )
                expected_length = 7
            # head和gripper不属于官方KDL；固定传target_height避免进入未验证的随机slide搜索。
            solutions = self._solver.inverse_kinematics(
                T_left=left_matrix,
                T_right=right_matrix,
                ref_pos=reference,
                target_height=slide,
            )
        except Exception as exc:  # noqa: BLE001 - 官方求解错误转为失败结果
            return IKResult(slide, None, None, False, f"调用官方 KDL 失败：{exc}")
        solution, failure_reason = _first_ik_solution(solutions, expected_length, np)
        if solution is None:
            return IKResult(slide, None, None, False, failure_reason)
        if not math.isclose(solution[0], slide, rel_tol=0.0, abs_tol=1e-9):
            return IKResult(
                slide,
                None,
                None,
                False,
                f"官方KDL返回的slide={solution[0]}与target_slide={slide}不一致",
            )
        if left_target is not None and right_target is not None:
            return IKResult(solution[0], solution[1:7], solution[7:13], True)
        if left_target is not None:
            return IKResult(solution[0], solution[1:7], None, True)
        return IKResult(solution[0], None, solution[1:7], True)



# ═══════════════════════════════════════════════════════════════════
# ArmPlanner：机械臂动作规划器（本文件的核心）
#
# 通俗理解：这是机械臂的"战术指挥官"。
# 它不直接控制机械臂，而是生成一份"动作计划书"（JointTrajectory），
# 告诉后续的执行模块（arm_execution）：
#   - 第0.5秒：双臂到达物体上方
#   - 第1.0秒：双臂下降到夹取位置
#   - 第1.5秒：夹爪闭合等待
#   - 第2.0秒：抬升物体
#   - 第3.0秒：安全撤离
#
# 每个时间点都包含17个关节的目标角度。
# 注意：生成了计划 ≠ 已经执行成功。执行和验证是 arm_execution 的事。
# ═══════════════════════════════════════════════════════════════════

class ArmPlanner:
    """物体中心到抓放位姿和关节轨迹的规划器。

    输入为 ``ObjectEstimate3D/TaskSpec/RobotJointState``，输出抓取/放置末端目标及
    ``JointTrajectory``。视觉2只提供物体中心；本类结合物体尺寸、抓取方向、
    object-to-gripper关系和安全偏移生成左右 ``Pose3D``，再转换到 ``footprint`` 求IK。

    ``pre-grasp`` 是接触前的安全预备位，``grasp`` 是包夹位置，``lift`` 是抓住后试抬，
    ``retreat`` 是带物撤离；``preplace`` 是释放前的安全位，``release`` 是松开时位姿。
    本类只生成17维路点计划：head通常保持实际反馈或不受控，gripper目标由抓放阶段协议
    决定。轨迹规划完成不表示已经执行或抓取成功，后两者属于 ``arm_execution``。

    一个 ``ArmPlanner`` 实例持有一个由组装层完成自检的 ``OfficialKDLAdapter``，规划
    方法复用其 ``solve_ik``；本类不会在每次规划时重新创建或自检求解器。

    包装盒尺寸（长24cm×宽16cm×高19cm）用于计算抓取/放置的几何偏移；
    world/odom到footprint的转换依赖尚未冻结，规划在footprint系中完成。
    """

    def __init__(self, ik_adapter: OfficialKDLAdapter) -> None:
        """注入已由组装层管理的官方KDL薄适配器。

        构造只保存依赖，不调用 ``self_check``，因此单元测试可以传入具有 ``solve_ik``
        方法的fake adapter。坐标变换依赖尚未形成稳定接口，本次不加入构造参数。
        """

        if not callable(getattr(ik_adapter, "solve_ik", None)):
            raise TypeError("ik_adapter必须提供可调用的solve_ik方法")
        self._ik_adapter = ik_adapter

        # ---- 内部执行状态预留（供后续执行模块扩展使用） ----
        # 当前规划阶段描述（str），如 "grasp_approach"/"place_release"
        self._current_plan_phase: str = "idle"
        # 期望抓取状态：是否预期已抓住物体
        self._expected_grasp_state: bool = False
        # 期望放置状态：是否预期已释放物体
        self._expected_place_state: bool = False
        # 抓取上下文（供 plan_place 反推释放位姿使用）
        # 存储抓取完成时左右夹爪相对物体中心的偏移及朝向，放置成功后由 plan_place 或
        # 外部调用 reset_context 清空。
        self._grasp_context: Optional[dict[str, Any]] = None

    def reset_context(self) -> None:
        """清空抓取上下文，供外部在任务重置或放置成功后调用。"""
        self._grasp_context = None

    # ------------------------------------------------------------------
    # 内部辅助：无效结果构造
    # ------------------------------------------------------------------

    @staticmethod
    def _null_grasp_result(
        traj_id: str, timestamp_ns: int, reason: str
    ) -> "tuple[GraspTarget, JointTrajectory]":
        """构造 valid=False 的抓取规划结果。"""
        null_pose = Pose3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), "")
        gt = GraspTarget(null_pose, null_pose, null_pose, null_pose, 0.0, 0.0, False, reason)
        jt = JointTrajectory(traj_id, (), timestamp_ns, False, reason)
        return gt, jt

    @staticmethod
    def _null_place_result(
        traj_id: str, timestamp_ns: int, reason: str
    ) -> "tuple[PlaceTarget, JointTrajectory]":
        """构造 valid=False 的放置规划结果。"""
        null_pose = Pose3D((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), "")
        pt = PlaceTarget(
            (0.0, 0.0, 0.0), null_pose, null_pose, null_pose, null_pose,
            0.0, False, reason,
        )
        jt = JointTrajectory(traj_id, (), timestamp_ns, False, reason)
        return pt, jt

    # ------------------------------------------------------------------
    # 内部辅助：路点构建
    # ------------------------------------------------------------------

    @staticmethod
    def _build_waypoint(
        ik: IKResult,
        actual_joints: RobotJointState,
        gripper_open: bool,
        time_from_start_s: float,
    ) -> JointWaypoint:
        """【路点打包】把IK求解结果+夹爪状态打包成一个完整的时间点动作。

        通俗理解：把"手臂关节该转到哪"和"夹爪该开还是关"合并成一条指令。
        17个关节 = 滑轨(1) + 头部(2) + 左臂(6) + 左夹爪(1) + 右臂(6) + 右夹爪(1)

        注意：头部关节不控制（保持实际位置），用 controlled_mask=False 标记。
        """
        actual_pos = actual_joints.position
        pos = list(actual_pos)

        pos[0] = ik.target_slide

        if ik.left_joint_target is not None:
            for j in range(6):
                pos[3 + j] = ik.left_joint_target[j]

        if ik.right_joint_target is not None:
            for j in range(6):
                pos[10 + j] = ik.right_joint_target[j]

        gripper_val = _GRIPPER_OPEN if gripper_open else _GRIPPER_CLOSED
        pos[9] = gripper_val
        pos[16] = gripper_val

        mask = [False] * 17
        mask[0] = True
        if ik.left_joint_target is not None:
            for j in range(3, 9):
                mask[j] = True
        if ik.right_joint_target is not None:
            for j in range(10, 16):
                mask[j] = True
        mask[9] = True
        mask[16] = True

        return JointWaypoint(
            time_from_start_s=time_from_start_s,
            joint_position=tuple(pos),
            controlled_mask=tuple(mask),
        )

    # ------------------------------------------------------------------
    # 内部辅助：动态时间计算
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stage_time(
        prev_joints: "Optional[tuple[float, ...]]",
        ik: IKResult,
        actual_joints: RobotJointState,
    ) -> float:
        """【动态计时】根据关节需要转多少来智能计算每个阶段的时间。

        通俗理解：关节转得多→给更多时间，转得少→少花时间。
        不像简单方案那样全部用固定2秒。

        特殊之处：关节（弧度）和滑轨（米）单位不同，必须分开算！
        - 关节时间 = 基础时间 + 最大关节变化量 × 每弧度耗时
        - 滑轨时间 = 基础时间 + 滑轨移动距离 ÷ 滑轨线速度
        - 最终时间 = 取两者最大值（确保两者都能完成）

        结果限制在 0.5~5.0 秒之间，不会太快也不会太慢。
        """
        if prev_joints is None:
            reference = actual_joints.position
        else:
            reference = prev_joints

        # ---- 关节时间：取所有手臂关节的最大变化量（单位 rad） ----
        max_joint_delta = 0.0
        left_target = ik.left_joint_target
        right_target = ik.right_joint_target

        if left_target is not None:
            for j in range(6):
                delta = abs(left_target[j] - reference[3 + j])
                if delta > max_joint_delta:
                    max_joint_delta = delta
        if right_target is not None:
            for j in range(6):
                delta = abs(right_target[j] - reference[10 + j])
                if delta > max_joint_delta:
                    max_joint_delta = delta

        joint_time = _BASE_TIME + max_joint_delta * _TIME_PER_RADIAN

        # ---- 滑轨时间：slide 变化量 / 滑轨线速度（单位 m） ----
        slide_delta = abs(ik.target_slide - reference[0])
        slide_time = _BASE_TIME + slide_delta / _SLIDE_SPEED if _SLIDE_SPEED > 0 else joint_time

        # ---- 取最大值，保证关节和滑轨都能在阶段时间内完成 ----
        time_s = max(joint_time, slide_time)
        return max(_MIN_STAGE_TIME, min(_MAX_STAGE_TIME, time_s))

    # ------------------------------------------------------------------
    # 内部辅助：关节跳变检测
    # ------------------------------------------------------------------

    @staticmethod
    def _check_joint_jump(
        stage_name: str,
        wp: JointWaypoint,
        prev_wp: JointWaypoint,
    ) -> "Optional[str]":
        """【安全检查】检测相邻动作之间的关节角度是否发生了"跳变"。

        通俗理解：如果上一秒关节在0度，下一秒突然要转到90度，
        说明规划可能有问题（IK跳到了另一个解的分支），
        或者目标位置不可达。这种"跳变"动作非常危险，
        会损坏机械臂或撞到东西。

        逐个检查17个关节，任何关节变化超过阈值（0.5弧度≈28°）就报警。
        """
        for j in range(17):
            delta = abs(wp.joint_position[j] - prev_wp.joint_position[j])
            if delta > _JOINT_DELTA_THRESHOLD:
                return (
                    f"{stage_name} 关节[{j}]跳变 "
                    f"{prev_wp.joint_position[j]:.4f}→{wp.joint_position[j]:.4f} "
                    f"(Δ={delta:.4f} > 阈值{_JOINT_DELTA_THRESHOLD})"
                )
        return None

    # ------------------------------------------------------------------
    # 内部辅助：几何安全检查（粗略安全过滤，不实现复杂碰撞检测）
    #
    # 注意：此检查仅根据简单距离/高度限制进行粗略过滤，防止生成明显
    # 不可达或危险的动作。它不替代完整的碰撞检测、工作空间分析或
    # 自碰撞检查。比赛现场如果出现复杂障碍物或非标物体，需要由
    # 更上层的感知和导航模块保证安全。
    # ------------------------------------------------------------------

    @staticmethod
    def _check_geometry_safety(
        check_type: str, value: float, limit: float, context: str
    ) -> "Optional[str]":
        """【安全防线】简单几何范围检查，防止生成明显不可达的动作。

        通俗理解：像一个"护栏"，防止机械臂被要求去够太远的物体、
        升得太高、或下降太多。超过限制就拒绝规划，防止危险。

        支持四种检查：
        - 'distance' : 物体太远？（>极限值→拒绝）
        - 'height'   : 高度太高？（>极限值→拒绝）
        - 'descent'  : 下降太多？（>极限值→拒绝）
        - 'retreat'  : 撤退太远？（>极限值→拒绝）

        注意：这只是粗略过滤，不是完整碰撞检测！
        复杂场景的安全由上层感知和导航模块保证。
        """
        if not math.isfinite(value):
            return f"{context}: 值非有限 ({value})"
        if value > limit:
            return f"{context}: {value:.3f}m 超出限制 {limit:.3f}m"
        return None

    # ------------------------------------------------------------------
    # 公开规划方法
    # ------------------------------------------------------------------


    def plan_grasp(
        self, target: ObjectEstimate3D, actual_joints: RobotJointState
    ) -> tuple[GraspTarget, JointTrajectory]:
        """【🌟 核心方法】规划完整的抓取动作流程。

        通俗理解：看到物体在哪里 → 设计一套"怎么抓"的动作。

        输入：感知模块给的物体3D位置、机械臂当前17个关节角度
        输出：抓取目标描述(GraspTarget) + 动作时间表(JointTrajectory)

        五阶段动作流水线：
        ┌──────────┬─────────────────────────────────┐
        │ 阶段     │ 做什么                           │
        ├──────────┼─────────────────────────────────┤
        │ approach │ 双臂移到物体上方（安全高度）     │
        │ grasp    │ 双臂下降到夹取位置               │
        │ close_wait│ 保持不动0.5秒，等夹爪夹紧        │
        │ lift     │ 竖直抬升物体                     │
        │ retreat  │ 水平后退 + 竖直上升，安全撤离     │
        └──────────┴─────────────────────────────────┘

        每个阶段：
        1. 计算左右手的3D目标位置（考虑箱子尺寸的偏移）
        2. 调用官方KDL求解器 → 得到关节角度
        3. 检查关节是否跳变（安全）
        4. 动态计算该阶段需要多长时间
        任何一步失败 → 返回 valid=False，绝不生成危险动作。
        """

        traj_id = f"grasp_{target.timestamp_ns}"
        ts = target.timestamp_ns

        # ---- 输入校验 ----
        if not target.valid or not actual_joints.valid:
            reason = f"输入无效: target.valid={target.valid}, joints.valid={actual_joints.valid}"
            return self._null_grasp_result(traj_id, ts, reason)

        # ---- 坐标系处理（临时方案，待系统提供正式 TF 后移除） ----
        target_xyz = target.position_xyz
        frame_id = target.frame_id
        if frame_id == "footprint":
            tx, ty, tz = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
        elif frame_id in ("world", "odom"):
            # 临时方案：使用 _FOOTPRINT_ROBOT_XY 常量，
            # 将 world/odom 坐标粗略转换到 footprint 系。
            # TODO: 待系统提供正式 TF 后移除本段临时转换。
            robot_x, robot_y = _FOOTPRINT_ROBOT_XY
            tx = float(target_xyz[0]) - robot_x
            ty = float(target_xyz[1]) - robot_y
            tz = float(target_xyz[2])
        else:
            return self._null_grasp_result(
                traj_id, ts, f"不支持坐标系 frame_id={frame_id}，仅支持 footprint/world/odom"
            )

        if not all(math.isfinite(v) for v in (tx, ty, tz)):
            return self._null_grasp_result(traj_id, ts, f"目标位置含非有限值: ({tx},{ty},{tz})")

        # ---- 抓取方向计算 ----
        # 机器人底盘中心在 footprint 系中的固定位置（临时方案，待 TF 替换）
        robot_xy = _FOOTPRINT_ROBOT_XY
        dx = tx - robot_xy[0]
        dy = ty - robot_xy[1]
        dist = math.hypot(dx, dy)
        # 粗略距离安全过滤（非碰撞检测），防止目标明显不可达
        if dist > _MAX_GRASP_DIST:
            return self._null_grasp_result(
                traj_id, ts, f"目标距离 {dist:.3f}m 超出最大抓取距离 {_MAX_GRASP_DIST}m"
            )
        if dist < 1e-9:
            return self._null_grasp_result(traj_id, ts, "目标与机器人中心重合，无法确定抓取方向")
        yaw = math.atan2(dy, dx)
        # 抓取方向单位向量（从机器人指向目标）及其垂直方向
        grasp_dir_x = dx / dist
        grasp_dir_y = dy / dist
        perp_x = -grasp_dir_y  # 逆时针旋转 90°
        perp_y = grasp_dir_x

        # 绕 Z 轴旋转 yaw 的四元数（xyzw 顺序）
        half_yaw = yaw / 2.0
        orient = (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

        # 夹爪偏移量：左右各偏移半箱宽 + 安全间隙
        grasp_offset = _BOX_HALF_WIDTH + 0.02  # 0.10m

        # ---- 阶段 1：预抓取（pre-grasp） ----
        # 目标上方 0.20m，沿抓取方向后退 0.15m
        pre_x = tx - 0.15 * grasp_dir_x
        pre_y = ty - 0.15 * grasp_dir_y
        pre_z = tz + _PRE_OFFSET_Z
        left_pregrasp_pose = Pose3D(
            (pre_x + perp_x * grasp_offset, pre_y + perp_y * grasp_offset, pre_z),
            orient, _KDL_TARGET_FRAME,
        )
        right_pregrasp_pose = Pose3D(
            (pre_x - perp_x * grasp_offset, pre_y - perp_y * grasp_offset, pre_z),
            orient, _KDL_TARGET_FRAME,
        )

        ik_pre = self._ik_adapter.solve_ik(
            actual_joints, left_target=left_pregrasp_pose, right_target=right_pregrasp_pose,
        )
        if not ik_pre.success:
            return self._null_grasp_result(traj_id, ts, f"预抓取IK失败: {ik_pre.failure_reason}")

        time_pre = self._compute_stage_time(None, ik_pre, actual_joints)
        wp_pre = self._build_waypoint(ik_pre, actual_joints, gripper_open=True, time_from_start_s=time_pre)

        # ---- 阶段 2：抓取（grasp） ----
        # 目标水平面，左右夹爪沿垂直方向偏移
        left_grasp_pose = Pose3D(
            (tx + perp_x * grasp_offset, ty + perp_y * grasp_offset, tz),
            orient, _KDL_TARGET_FRAME,
        )
        right_grasp_pose = Pose3D(
            (tx - perp_x * grasp_offset, ty - perp_y * grasp_offset, tz),
            orient, _KDL_TARGET_FRAME,
        )

        ik_grasp = self._ik_adapter.solve_ik(
            actual_joints, left_target=left_grasp_pose, right_target=right_grasp_pose,
        )
        if not ik_grasp.success:
            return self._null_grasp_result(traj_id, ts, f"抓取IK失败: {ik_grasp.failure_reason}")

        time_grasp = time_pre + self._compute_stage_time(wp_pre.joint_position, ik_grasp, actual_joints)
        wp_grasp = self._build_waypoint(ik_grasp, actual_joints, gripper_open=False, time_from_start_s=time_grasp)
        jump_reason = self._check_joint_jump("grasp", wp_grasp, wp_pre)
        if jump_reason is not None:
            return self._null_grasp_result(traj_id, ts, jump_reason)

        # ---- 阶段 3：夹紧等待（close_wait） ----
        # 位置同抓取，不改变关节，仅增加等待时间 0.5s
        time_close = time_grasp + _CLOSE_WAIT_TIME
        wp_close = self._build_waypoint(ik_grasp, actual_joints, gripper_open=False, time_from_start_s=time_close)

        # ---- 阶段 4：抬升（lift） ----
        # 目标正上方 _LIFT_DELTA，夹爪闭合
        left_lift_pose = Pose3D(
            (tx + perp_x * grasp_offset, ty + perp_y * grasp_offset, tz + _LIFT_DELTA),
            orient, _KDL_TARGET_FRAME,
        )
        right_lift_pose = Pose3D(
            (tx - perp_x * grasp_offset, ty - perp_y * grasp_offset, tz + _LIFT_DELTA),
            orient, _KDL_TARGET_FRAME,
        )

        # 抬升高度安全检查（粗略过滤，非完整碰撞检测）
        lift_z = tz + _LIFT_DELTA
        if lift_z > _MAX_LIFT_HEIGHT:
            return self._null_grasp_result(
                traj_id, ts, f"抬升高度 {lift_z:.3f}m 超出限制 {_MAX_LIFT_HEIGHT}m"
            )
        ik_lift = self._ik_adapter.solve_ik(
            actual_joints, left_target=left_lift_pose, right_target=right_lift_pose,
        )
        if not ik_lift.success:
            return self._null_grasp_result(traj_id, ts, f"抬升IK失败: {ik_lift.failure_reason}")

        time_lift = time_close + self._compute_stage_time(wp_close.joint_position, ik_lift, actual_joints)
        wp_lift = self._build_waypoint(ik_lift, actual_joints, gripper_open=False, time_from_start_s=time_lift)
        jump_reason = self._check_joint_jump("lift", wp_lift, wp_close)
        if jump_reason is not None:
            return self._null_grasp_result(traj_id, ts, jump_reason)

        # ---- 阶段 5：撤离（retreat） ----
        # 抬升基础上再沿抓取方向后退 0.15m 并升高 0.10m
        retreat_x = (tx + perp_x * grasp_offset) - 0.15 * grasp_dir_x
        retreat_y = (ty + perp_y * grasp_offset) - 0.15 * grasp_dir_y
        retreat_z = tz + _LIFT_DELTA + _RETREAT_DZ
        left_retreat_pose = Pose3D(
            (retreat_x, retreat_y, retreat_z), orient, _KDL_TARGET_FRAME,
        )
        right_retreat_pose = Pose3D(
            (retreat_x - 2.0 * perp_x * grasp_offset, retreat_y - 2.0 * perp_y * grasp_offset, retreat_z),
            orient, _KDL_TARGET_FRAME,
        )

        ik_retreat = self._ik_adapter.solve_ik(
            actual_joints, left_target=left_retreat_pose, right_target=right_retreat_pose,
        )
        if not ik_retreat.success:
            return self._null_grasp_result(traj_id, ts, f"撤离IK失败: {ik_retreat.failure_reason}")

        time_retreat = time_lift + self._compute_stage_time(wp_lift.joint_position, ik_retreat, actual_joints)
        wp_retreat = self._build_waypoint(ik_retreat, actual_joints, gripper_open=False, time_from_start_s=time_retreat)
        jump_reason = self._check_joint_jump("retreat", wp_retreat, wp_lift)
        if jump_reason is not None:
            return self._null_grasp_result(traj_id, ts, jump_reason)

        # ---- 组装轨迹 ----
        waypoints = (wp_pre, wp_grasp, wp_close, wp_lift, wp_retreat)
        jt = JointTrajectory(traj_id, waypoints, ts, True)

        # 构建 GraspTarget（四个夹爪位姿分别对应预抓取和抓取阶段）
        gt = GraspTarget(
            left_pregrasp=left_pregrasp_pose,
            right_pregrasp=right_pregrasp_pose,
            left_grasp=left_grasp_pose,
            right_grasp=right_grasp_pose,
            lift_delta_m=_LIFT_DELTA,
            confidence=1.0,
            valid=True,
        )

        # ---- 保存抓取上下文（供 plan_place 使用） ----
        # 记录抬升结束时左右夹爪末端相对物体中心的偏移
        self._grasp_context = {
            "yaw": yaw,
            "grasp_dir": (grasp_dir_x, grasp_dir_y),
            "perp_dir": (perp_x, perp_y),
            "grasp_offset": grasp_offset,
            # Z 偏移使用箱体半高 + 表面余量，保证放置时箱底贴合目标面、不悬空
            "left_offset": (
                perp_x * grasp_offset,
                perp_y * grasp_offset,
                _BOX_HALF_HEIGHT + _PLACE_SURFACE_OFFSET,
            ),
            "right_offset": (
                -perp_x * grasp_offset,
                -perp_y * grasp_offset,
                _BOX_HALF_HEIGHT + _PLACE_SURFACE_OFFSET,
            ),
            "orient": orient,
        }

        return gt, jt




    def plan_place(
        self, task: TaskSpec, actual_joints: RobotJointState
    ) -> tuple[PlaceTarget, JointTrajectory]:
        """【🌟 核心方法】规划完整的放置动作流程。

        通俗理解：已经抓着箱子了 → 设计一套"怎么放下"的动作。

        输入：任务给的放置目标位置、机械臂当前17个关节角度
        输出：放置目标描述(PlaceTarget) + 动作时间表(JointTrajectory)

        三阶段动作流水线：
        ┌──────────┬─────────────────────────────────┐
        │ 阶段     │ 做什么                           │
        ├──────────┼─────────────────────────────────┤
        │ preplace │ 携带物体到达目标上方安全高度     │
        │ release  │ 下降到释放位置，张开夹爪         │
        │ retreat  │ 竖直上升 + 后退，安全撤离         │
        └──────────┴─────────────────────────────────┘

        关键细节：释放高度不是简单的目标Z坐标。
        需要考虑箱体高度（19cm），保证箱子底部接触目标平面，
        不悬空、不怼进货架。
        """

        traj_id = f"place_{task.task_id}"
        # TaskSpec 无 timestamp_ns 字段时，使用实际关节时间戳作为后备
        ts = getattr(task, "timestamp_ns", actual_joints.timestamp_ns)

        # ---- 输入校验 ----
        if not task.valid or not actual_joints.valid:
            reason = f"输入无效: task.valid={task.valid}, joints.valid={actual_joints.valid}"
            return self._null_place_result(traj_id, ts, reason)

        if self._grasp_context is None:
            return self._null_place_result(traj_id, ts, "未执行抓取，无法规划放置")

        ctx = self._grasp_context
        yaw: float = ctx["yaw"]
        grasp_dir_x: float = ctx["grasp_dir"][0]
        grasp_dir_y: float = ctx["grasp_dir"][1]
        perp_x: float = ctx["perp_dir"][0]
        perp_y: float = ctx["perp_dir"][1]
        grasp_offset: float = ctx["grasp_offset"]
        left_offset: tuple = ctx["left_offset"]
        right_offset: tuple = ctx["right_offset"]
        orient: tuple = ctx["orient"]

        # ---- 坐标系处理（临时方案，待系统提供正式 TF 后移除） ----
        place_xyz = task.place_world_xyz
        # TaskSpec 没有 place_frame_id 字段，默认假设为 world/odom
        # TODO: 待 TaskSpec 增加 frame_id 字段后改用正式字段
        # 临时方案：TaskSpec 尚无 frame_id 字段，默认 place_world_xyz 为 world 系。
        # 待 TaskSpec 增加 frame_id 后替换为正式字段。
        place_frame = "world"
        if place_frame == "footprint":
            px, py, pz = float(place_xyz[0]), float(place_xyz[1]), float(place_xyz[2])
        elif place_frame in ("world", "odom"):
            # 临时方案：使用 _FOOTPRINT_ROBOT_XY 常量，
            # 将 world/odom 坐标粗略转换到 footprint 系。
            # TODO: 待系统提供正式 TF 后移除本段临时转换。
            robot_x, robot_y = _FOOTPRINT_ROBOT_XY
            px = float(place_xyz[0]) - robot_x
            py = float(place_xyz[1]) - robot_y
            pz = float(place_xyz[2])
        else:
            return self._null_place_result(
                traj_id, ts, f"不支持坐标系 place_frame_id={place_frame}，仅支持 footprint/world/odom"
            )

        if not all(math.isfinite(v) for v in (px, py, pz)):
            return self._null_place_result(traj_id, ts, f"放置位置含非有限值: ({px},{py},{pz})")

        # ---- 反推释放位姿：物体中心 + 抓取时保存的夹爪偏移 ----
        left_release_x = px + left_offset[0]
        left_release_y = py + left_offset[1]
        right_release_x = px + right_offset[0]
        right_release_y = py + right_offset[1]
        left_release_z = pz + left_offset[2]
        right_release_z = pz + right_offset[2]

        left_release_pose = Pose3D(
            (left_release_x, left_release_y, left_release_z),
            orient, _KDL_TARGET_FRAME,
        )
        right_release_pose = Pose3D(
            (right_release_x, right_release_y, right_release_z),
            orient, _KDL_TARGET_FRAME,
        )

        # ---- 阶段 1：预放置（pre-place） ----
        # 释放位姿上方 _PRE_OFFSET_Z，沿抓取方向略微后退
        preplace_x = left_release_x - 0.10 * grasp_dir_x
        preplace_y = left_release_y - 0.10 * grasp_dir_y
        preplace_z = left_release_z + _PRE_OFFSET_Z
        left_preplace_pose = Pose3D(
            (preplace_x, preplace_y, preplace_z), orient, _KDL_TARGET_FRAME,
        )
        right_preplace_pose = Pose3D(
            (right_release_x - 0.10 * grasp_dir_x, right_release_y - 0.10 * grasp_dir_y, right_release_z + _PRE_OFFSET_Z),
            orient, _KDL_TARGET_FRAME,
        )

        ik_preplace = self._ik_adapter.solve_ik(
            actual_joints, left_target=left_preplace_pose, right_target=right_preplace_pose,
        )
        if not ik_preplace.success:
            self._grasp_context = None
            return self._null_place_result(traj_id, ts, f"预放置IK失败: {ik_preplace.failure_reason}")

        time_pre = self._compute_stage_time(None, ik_preplace, actual_joints)
        wp_preplace = self._build_waypoint(ik_preplace, actual_joints, gripper_open=False, time_from_start_s=time_pre)

        # ---- 阶段 2：释放（release） ----
        # 下降到释放位姿，夹爪张开
        # 放置下降高度安全检查（粗略过滤，非完整碰撞检测）
        place_descent = preplace_z - left_release_z
        if place_descent > _MAX_PLACE_DESCENT:
            self._grasp_context = None
            return self._null_place_result(
                traj_id, ts, f"放置下降 {place_descent:.3f}m 超出限制 {_MAX_PLACE_DESCENT}m"
            )
        ik_release = self._ik_adapter.solve_ik(
            actual_joints, left_target=left_release_pose, right_target=right_release_pose,
        )
        if not ik_release.success:
            self._grasp_context = None
            return self._null_place_result(traj_id, ts, f"释放IK失败: {ik_release.failure_reason}")

        time_release = time_pre + self._compute_stage_time(wp_preplace.joint_position, ik_release, actual_joints)
        wp_release = self._build_waypoint(ik_release, actual_joints, gripper_open=True, time_from_start_s=time_release)
        jump_reason = self._check_joint_jump("release", wp_release, wp_preplace)
        if jump_reason is not None:
            self._grasp_context = None
            return self._null_place_result(traj_id, ts, jump_reason)

        # ---- 阶段 3：撤离（retreat） ----
        # 竖直抬升 0.10m 并水平后退 0.10m
        retreat_x = left_release_x - 0.10 * grasp_dir_x
        retreat_y = left_release_y - 0.10 * grasp_dir_y
        retreat_z = left_release_z + 0.10
        left_retreat_place_pose = Pose3D(
            (retreat_x, retreat_y, retreat_z), orient, _KDL_TARGET_FRAME,
        )
        right_retreat_place_pose = Pose3D(
            (right_release_x - 0.10 * grasp_dir_x, right_release_y - 0.10 * grasp_dir_y, right_release_z + 0.10),
            orient, _KDL_TARGET_FRAME,
        )

        ik_retreat = self._ik_adapter.solve_ik(
            actual_joints, left_target=left_retreat_place_pose, right_target=right_retreat_place_pose,
        )
        if not ik_retreat.success:
            self._grasp_context = None
            return self._null_place_result(traj_id, ts, f"撤离IK失败: {ik_retreat.failure_reason}")

        time_ret = time_release + self._compute_stage_time(wp_release.joint_position, ik_retreat, actual_joints)
        wp_retreat_place = self._build_waypoint(ik_retreat, actual_joints, gripper_open=True, time_from_start_s=time_ret)
        jump_reason = self._check_joint_jump("retreat", wp_retreat_place, wp_release)
        if jump_reason is not None:
            self._grasp_context = None
            return self._null_place_result(traj_id, ts, jump_reason)

        # ---- 组装轨迹 ----
        waypoints = (wp_preplace, wp_release, wp_retreat_place)
        jt = JointTrajectory(traj_id, waypoints, ts, True)

        place_goal = (px, py, pz)
        pt = PlaceTarget(
            object_goal_xyz=place_goal,
            left_preplace=left_preplace_pose,
            right_preplace=right_preplace_pose,
            left_release=left_release_pose,
            right_release=right_release_pose,
            settle_time_s=_SETTLE_TIME,
            valid=True,
        )

        # ---- 清除上下文 ----
        self._grasp_context = None

        return pt, jt


def _pose_to_matrix(pose: Pose3D, np: Any) -> Any:
    """【坐标转换】把位置(xyz)+朝向(四元数xyzw)打包成KDL需要的4×4数学矩阵。

    通俗理解：官方KDL求解器只认"4×4矩阵"这种格式。
    这个函数把我们的 Pose3D（位置+朝向）翻译成矩阵格式。

    矩阵结构（4行4列）：
    ┌               ┐
    │ R  R  R  X    │  上面3×3 = 旋转矩阵（朝向）
    │ R  R  R  Y    │  最右一列 = 平移向量（位置）
    │ R  R  R  Z    │  最下一行 = [0,0,0,1]（齐次坐标固定格式）
    │ 0  0  0  1    │
    └               ┘
    """

    position = _finite_vector(pose.position_xyz, 3, "Pose3D.position_xyz")
    x, y, z, w = _finite_vector(pose.orientation_xyzw, 4, "Pose3D.orientation_xyzw")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("IK 目标四元数范数为零")
    # 四元数长度不为1会让旋转矩阵失真；归一化只修正尺度，不改变目标朝向。
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4, dtype=float)
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("IK 目标旋转矩阵必须是有限3×3矩阵")
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(position, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("IK 目标齐次矩阵必须是有限4×4矩阵")
    if not np.array_equal(matrix[3, :], np.asarray((0.0, 0.0, 0.0, 1.0))):
        raise ValueError("IK 目标齐次矩阵最后一行必须是[0,0,0,1]")
    return matrix
