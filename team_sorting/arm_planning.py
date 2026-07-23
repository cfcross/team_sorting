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
    ObjectEstimate3D,
    PlaceTarget,
    Pose3D,
    RobotJointState,
    TaskSpec,
)


_KDL_TARGET_FRAME = "footprint"


def _finite_vector(values: Any, expected_length: int, name: str) -> tuple[float, ...]:
    """读取定长真实有限数向量，拒绝会被Python当作数字的布尔值。"""

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
    """提取官方首个一维解，并把所有异常输出收窄为明确失败原因。"""

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
        """调用官方固定 slide IK，并转换目标关节解。

        参数：实际关节反馈用于选择最近解；左右末端目标必须已经转换到官方确认的
        ``footprint``，位置米、姿态xyzw；slide单位米。返回 ``IKResult``。
        失败：未自检时抛 ``RuntimeError``；目标为空、frame 错误、输入无效或官方无解
        时返回 ``success=False``，不混淆实际关节与目标解。
        """

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


class ArmPlanner:
    """物体中心到抓放位姿和关节轨迹的规划器骨架。

    输入为 ``ObjectEstimate3D/TaskSpec/RobotJointState``，输出抓取/放置末端目标及
    ``JointTrajectory``。视觉2只提供物体中心；本类未来结合物体尺寸、抓取方向、
    object-to-gripper关系和安全偏移生成左右 ``Pose3D``，再转换到 ``footprint`` 求IK。

    ``pre-grasp`` 是接触前的安全预备位，``grasp`` 是包夹位置，``lift`` 是抓住后试抬，
    ``retreat`` 是带物撤离；``preplace`` 是释放前的安全位，``release`` 是松开时位姿。
    本类只生成17维路点计划：head通常保持实际反馈或不受控，gripper目标由抓放阶段协议
    决定。轨迹规划完成不表示已经执行或抓取成功，后两者属于 ``arm_execution``。

    一个 ``ArmPlanner`` 实例持有一个由组装层完成自检的 ``OfficialKDLAdapter``，规划
    方法未来复用它调用 ``solve_ik``；本类不会在每次规划时重新创建或自检求解器。
    当前 ``plan_grasp`` 没有 ``BaseState``，仓库也没有稳定的普通Python坐标变换接口，
    所以world/odom到footprint的转换依赖尚未冻结，不能在这里假设frame等价或暗中查TF。

    当前 ``plan_place`` 参数中没有抓取时的object-to-gripper关系。第一版暂定采用边界B：
    未来由同一个planner实例在成功的 ``plan_grasp`` 后保存计划抓取关系，Episode结束、
    抓取失败或任务重置时必须清除；存储结构和重置入口仍待系统评审，本次不实现状态。
    """

    def __init__(self, ik_adapter: OfficialKDLAdapter) -> None:
        """注入已由组装层管理的官方KDL薄适配器。

        构造只保存依赖，不调用 ``self_check``，因此单元测试可以传入具有 ``solve_ik``
        方法的fake adapter。坐标变换依赖尚未形成稳定接口，本次不加入构造参数。
        """

        if not callable(getattr(ik_adapter, "solve_ik", None)):
            raise TypeError("ik_adapter必须提供可调用的solve_ik方法")
        self._ik_adapter = ik_adapter

    def plan_grasp(
        self, target: ObjectEstimate3D, actual_joints: RobotJointState
    ) -> tuple[GraspTarget, JointTrajectory]:
        """生成 pre-grasp、grasp、lift 和 retreat 轨迹。

        机械臂1后续应检查目标有效性和frame，把物体中心转换到KDL基座系；结合箱体尺寸、
        抓取方向和双臂间距生成左右pre-grasp/grasp Pose，再生成lift/retreat，逐阶段
        求IK并检查连续性、可达性和简单碰撞风险。最后把13维解嵌入17维路点，未控制的
        head/gripper保持实际值或mask为False，并设置合理的路点时间。当前明确抛出异常。
        """

        raise NotImplementedError("双臂抓取位姿和轨迹规划尚未实现，请由机械臂1负责人完成")

    def plan_place(
        self, task: TaskSpec, actual_joints: RobotJointState
    ) -> tuple[PlaceTarget, JointTrajectory]:
        """未来把 ``TaskSpec.place_world_xyz`` 转换为双臂放置轨迹。

        ``place_world_xyz`` 是放置后的物体中心，不是夹爪Pose。机械臂1后续应利用抓取后
        的object-to-gripper关系生成左右release和preplace Pose，再规划下降、释放和撤离
        目标、调用IK并生成17维轨迹；不能直接把物体中心交给KDL。当前明确抛出异常。
        """

        raise NotImplementedError("双臂放置位姿和轨迹规划尚未实现，请由机械臂1负责人完成")


def _pose_to_matrix(pose: Pose3D, np: Any) -> Any:
    """把footprint中的末端Pose转换为KDL需要的4×4齐次矩阵，不执行frame转换。"""

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
