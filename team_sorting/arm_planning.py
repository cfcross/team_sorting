"""双臂抓放规划和官方 KDL 薄适配层。

FK（正运动学）根据关节角计算末端在哪里；IK（逆运动学）根据末端希望到达的位置和
方向反求关节目标。本文件的数据流是：

``ObjectEstimate3D`` 当前物体事实 / ``TaskSpec.place_world_xyz`` 放置后的物体中心
→ ``ArmPlanner`` 生成左右末端 ``Pose3D``
→ ``OfficialKDLAdapter`` 调用官方 ``MMK2Kdl``
→ ``IKResult``（slide + 左臂6轴 + 右臂6轴）
→ ``ArmPlanner`` 嵌入17维 ``JointWaypoint``
→ ``JointTrajectory``
→ ``arm_execution`` 插值、执行并依据实际反馈判断结果。

物体中心不是夹爪末端位姿；抓取还需要观测姿态、尺寸、双臂间距和安全偏移。官方双臂
运动学向量是13维 ``[slide,left×6,right×6]``，单臂是7维 ``[slide,arm×6]``；团队
17维路点还包含head和左右gripper，未控制关节必须保持实际反馈或使用
``controlled_mask=False``，不能补全为零。

本文件不复制官方解析IK，不导入 ``rclpy``，不发布或执行机械臂命令，也不判断抓取
成功。官方模块和 NumPy 均延迟导入；抓放规划使用普通Python几何并通过注入适配器求IK。
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
    ArmPlanningConfig,
    ArmMotionPhase,
    GlobalPhase,
    GraspContext,
    GraspTarget,
    IKResult,
    JointWaypoint,
    JointTrajectory,
    ObjectEstimate3D,
    PlaceTarget,
    Pose3D,
    RobotJointState,
    RigidTransform3D,
    TaskSpec,
)


_KDL_TARGET_FRAME = "footprint"
_WORLD_FRAME = "world"
_GEOMETRY_TOLERANCE = 1e-9
_VERTICAL_TOLERANCE = 1e-6


def _vector_add(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(a + b for a, b in zip(left, right))


def _vector_subtract(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(a - b for a, b in zip(left, right))


def _vector_scale(values: tuple[float, ...], scale: float) -> tuple[float, ...]:
    return tuple(value * scale for value in values)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize_vector(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    finite = _finite_vector(values, len(values), name)
    norm = math.sqrt(_dot(finite, finite))
    if norm <= _GEOMETRY_TOLERANCE:
        raise ValueError(f"{name}几何退化，范数接近零")
    return _vector_scale(finite, 1.0 / norm)


def _normalize_quaternion(
    quaternion: tuple[float, float, float, float], name: str
) -> tuple[float, float, float, float]:
    normalized = _normalize_vector(_finite_vector(quaternion, 4, name), name)
    return normalized  # type: ignore[return-value]


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _normalize_quaternion(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        "四元数乘积",
    )


def _quaternion_conjugate(
    quaternion: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x, y, z, w = _normalize_quaternion(quaternion, "四元数")
    return (-x, -y, -z, w)


def _rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = _normalize_quaternion(quaternion, "旋转四元数")
    q_vector = (x, y, z)
    twice_cross = _vector_scale(_cross(q_vector, vector), 2.0)
    rotated = _vector_add(
        vector,
        _vector_add(_vector_scale(twice_cross, w), _cross(q_vector, twice_cross)),
    )
    return _finite_vector(rotated, 3, "旋转后向量")  # type: ignore[return-value]


def _rotation_columns_to_quaternion(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """把按列给出的正交右手旋转基转换为xyzw四元数。"""

    x_axis = _normalize_vector(x_axis, "旋转基X轴")  # type: ignore[assignment]
    y_axis = _normalize_vector(y_axis, "旋转基Y轴")  # type: ignore[assignment]
    z_axis = _normalize_vector(z_axis, "旋转基Z轴")  # type: ignore[assignment]
    if (
        abs(_dot(x_axis, y_axis)) > _VERTICAL_TOLERANCE
        or abs(_dot(x_axis, z_axis)) > _VERTICAL_TOLERANCE
        or abs(_dot(y_axis, z_axis)) > _VERTICAL_TOLERANCE
        or _dot(_cross(x_axis, y_axis), z_axis) < 1.0 - _VERTICAL_TOLERANCE
    ):
        raise ValueError("旋转基必须是正交右手系")
    m00, m10, m20 = x_axis
    m01, m11, m21 = y_axis
    m02, m12, m22 = z_axis
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = ((m21 - m12) / scale, (m02 - m20) / scale,
                      (m10 - m01) / scale, 0.25 * scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quaternion = (0.25 * scale, (m01 + m10) / scale,
                      (m02 + m20) / scale, (m21 - m12) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quaternion = ((m01 + m10) / scale, 0.25 * scale,
                      (m12 + m21) / scale, (m02 - m20) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quaternion = ((m02 + m20) / scale, (m12 + m21) / scale,
                      0.25 * scale, (m10 - m01) / scale)
    return _normalize_quaternion(quaternion, "旋转基四元数")


def _transform_pose(transform: RigidTransform3D, pose: Pose3D) -> Pose3D:
    if pose.frame_id != transform.source_frame:
        raise ValueError(
            f"Pose frame={pose.frame_id!r}与变换source_frame={transform.source_frame!r}不一致"
        )
    assert transform.translation_xyz is not None and transform.rotation_xyzw is not None
    return Pose3D(
        _vector_add(
            _rotate_vector(transform.rotation_xyzw, pose.position_xyz),
            transform.translation_xyz,
        ),
        _quaternion_multiply(transform.rotation_xyzw, pose.orientation_xyzw),
        transform.target_frame,
    )


def _compose_pose_with_transform(
    target_pose: Pose3D, source_to_target: RigidTransform3D
) -> Pose3D:
    """计算T_frame_source = T_frame_target @ T_target_source。"""

    assert source_to_target.translation_xyz is not None
    assert source_to_target.rotation_xyzw is not None
    return Pose3D(
        _vector_add(
            target_pose.position_xyz,
            _rotate_vector(target_pose.orientation_xyzw, source_to_target.translation_xyz),
        ),
        _quaternion_multiply(
            target_pose.orientation_xyzw, source_to_target.rotation_xyzw
        ),
        target_pose.frame_id,
    )


def _relative_transform(
    target_pose: Pose3D,
    source_pose: Pose3D,
    source_frame: str,
    target_frame: str,
    timestamp_ns: int,
) -> RigidTransform3D:
    """反算T_target_source = inverse(T_frame_target) @ T_frame_source。"""

    if target_pose.frame_id != source_pose.frame_id:
        raise ValueError("相对Pose必须位于同一外部frame")
    inverse_rotation = _quaternion_conjugate(target_pose.orientation_xyzw)
    return RigidTransform3D(
        source_frame=source_frame,
        target_frame=target_frame,
        translation_xyz=_rotate_vector(
            inverse_rotation,
            _vector_subtract(source_pose.position_xyz, target_pose.position_xyz),
        ),
        rotation_xyzw=_quaternion_multiply(
            inverse_rotation, source_pose.orientation_xyzw
        ),
        timestamp_ns=timestamp_ns,
        valid=True,
    )


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
    """物体中心到抓放位姿和关节轨迹的纯Python规划器。

    输入为 ``ObjectEstimate3D/TaskSpec/RobotJointState``，输出抓取/放置末端目标及
    ``JointTrajectory``。视觉2提供物体中心及可选感知事实；本类结合物体尺寸、姿态、
    object-to-gripper关系和安全偏移生成左右 ``Pose3D``，再转换到 ``footprint`` 求IK。

    ``pre-grasp`` 是接触前的安全预备位，``grasp`` 是包夹位置，``lift`` 是抓住后试抬，
    ``retreat`` 是带物撤离；``preplace`` 是释放前的安全位，``release`` 是松开时位姿。
    本类只生成17维路点计划：head通常保持实际反馈或不受控，gripper目标由抓放阶段协议
    决定。轨迹规划完成不表示已经执行或抓取成功，后两者属于 ``arm_execution``。

    一个 ``ArmPlanner`` 实例持有一个由组装层完成自检的 ``OfficialKDLAdapter``，规划
    方法复用它调用 ``solve_ik``；本类不会在每次规划时重新创建或自检求解器。
    坐标变换由组装层以纯Python ``RigidTransform3D`` 快照显式传入，本类不查询TF。
    抓取时还需target到world的快照，用于记录物体观测到的world朝向；放置使用经过执行
    反馈确认的 ``GraspContext``，且planner自身不保存任何跨调用抓取状态。
    """

    def __init__(
        self, ik_adapter: OfficialKDLAdapter, config: ArmPlanningConfig
    ) -> None:
        """注入已由组装层管理的官方KDL薄适配器。

        构造只保存依赖，不调用 ``self_check``，因此单元测试可以传入具有 ``solve_ik``
        方法的fake adapter。变换由调用方通过规划方法参数显式传入；构造器只保存KDL
        adapter和config，本类不查询TF。
        """

        if not callable(getattr(ik_adapter, "solve_ik", None)):
            raise TypeError("ik_adapter必须提供可调用的solve_ik方法")
        if not isinstance(config, ArmPlanningConfig):
            raise TypeError("config必须是ArmPlanningConfig")
        self._ik_adapter = ik_adapter
        self._config = config

    def plan_grasp(
        self,
        task: TaskSpec,
        target: ObjectEstimate3D,
        target_to_footprint: RigidTransform3D,
        target_to_world: RigidTransform3D,
        actual_joints: RobotJointState,
        now_ns: int,
    ) -> tuple[GraspTarget, JointTrajectory]:
        """生成纯Python双臂包夹几何、IK目标和五个17维路点。

        所有预期输入、几何、IK和连续性失败均原子地返回同原因的无效目标与空轨迹；
        本方法不查询TF、不执行轨迹，也不把planned context标记为confirmed。
        """

        try:
            return self._plan_grasp_checked(
                task, target, target_to_footprint, target_to_world,
                actual_joints, now_ns,
            )
        except Exception as exc:  # noqa: BLE001 - 规划边界必须把依赖异常收窄为失败对象
            reason = f"抓取规划失败：{exc}"
            return _invalid_grasp_result(reason, task, now_ns)

    def _plan_grasp_checked(
        self,
        task: TaskSpec,
        target: ObjectEstimate3D,
        target_to_footprint: RigidTransform3D,
        target_to_world: RigidTransform3D,
        actual_joints: RobotJointState,
        now_ns: int,
    ) -> tuple[GraspTarget, JointTrajectory]:
        self._config.validate_for_grasp()
        _strict_now(now_ns)
        _validate_task(task, now_ns)
        if not isinstance(target, ObjectEstimate3D):
            raise ValueError("target必须是ObjectEstimate3D")
        if type(target.valid) is not bool or not target.valid:
            raise ValueError(target.failure_reason or "ObjectEstimate3D无效")
        if target.class_id != task.target_color:
            raise ValueError("target.class_id与TaskSpec.target_color不匹配")
        if not isinstance(target.object_id, str) or not target.object_id.strip():
            raise ValueError("抓取规划要求非空target.object_id稳定身份")
        if target.orientation_xyzw is None:
            raise ValueError("抓取规划要求target.orientation_xyzw")
        if target.size_xyz_m is None:
            raise ValueError("抓取规划要求target.size_xyz_m局部XYZ完整尺寸")
        confidence = _finite_scalar(target.confidence, "target.confidence")
        minimum_confidence = _config_float(self._config, "min_object_confidence")
        if confidence < minimum_confidence:
            raise ValueError("target.confidence低于ArmPlanningConfig.min_object_confidence")
        target_position = _finite_vector(target.position_xyz, 3, "target.position_xyz")
        target_orientation = _normalize_quaternion(
            target.orientation_xyzw, "target.orientation_xyzw"
        )
        target_size = _finite_vector(target.size_xyz_m, 3, "target.size_xyz_m")
        if any(value <= 0.0 for value in target_size):
            raise ValueError("target.size_xyz_m三轴必须均为有限正数")
        if not isinstance(target.frame_id, str) or not target.frame_id.strip():
            raise ValueError("target.frame_id必须是非空字符串")
        _validate_fresh_stamp(
            target.timestamp_ns, now_ns,
            _config_int(self._config, "object_estimate_max_age_ns"), "target",
        )
        actual_position = _validate_actual_joints(
            actual_joints, now_ns,
            _config_int(self._config, "joint_state_max_age_ns"),
        )
        _validate_actual_gripper_ranges(actual_position, self._config)
        transform_max_age_ns = _config_int(self._config, "transform_max_age_ns")
        _validate_transform(
            target_to_footprint, target.frame_id, _KDL_TARGET_FRAME, now_ns,
            transform_max_age_ns,
            "target_to_footprint",
        )
        _validate_transform_observation_time(
            target_to_footprint, target.timestamp_ns, transform_max_age_ns,
            "target_to_footprint",
        )
        _validate_transform(
            target_to_world, target.frame_id, _WORLD_FRAME, now_ns,
            transform_max_age_ns, "target_to_world",
        )
        _validate_transform_observation_time(
            target_to_world, target.timestamp_ns, transform_max_age_ns,
            "target_to_world",
        )

        target_object_pose = Pose3D(
            target_position, target_orientation, target.frame_id
        )
        footprint_object = _transform_pose(target_to_footprint, target_object_pose)
        world_object = _transform_pose(target_to_world, target_object_pose)
        _require_upright_object_orientation(
            footprint_object.orientation_xyzw, "footprint物体姿态"
        )
        _require_upright_object_orientation(
            world_object.orientation_xyzw, "world物体姿态"
        )
        center = footprint_object.position_xyz
        forward = _normalize_vector((center[0], center[1], 0.0), "物体水平径向方向")
        robot_left = (-forward[1], forward[0], 0.0)
        side_axis, selected_size = _select_side_axis(
            footprint_object.orientation_xyzw, target_size, robot_left
        )
        tool_x_raw = _vector_subtract(forward, _vector_scale(side_axis, _dot(forward, side_axis)))
        tool_x = _normalize_vector(tool_x_raw, "夹爪工具X轴")
        left_tool_z = side_axis
        right_tool_z = _vector_scale(side_axis, -1.0)
        left_orientation = _rotation_columns_to_quaternion(
            tool_x, _cross(left_tool_z, tool_x), left_tool_z
        )
        right_orientation = _rotation_columns_to_quaternion(
            tool_x, _cross(right_tool_z, tool_x), right_tool_z
        )
        contact_distance = selected_size / 2.0 + _config_float(
            self._config, "grasp_contact_offset_m"
        )
        pregrasp_distance = _config_float(self._config, "pregrasp_distance_m")
        left_grasp_position = _vector_add(center, _vector_scale(side_axis, contact_distance))
        right_grasp_position = _vector_subtract(center, _vector_scale(side_axis, contact_distance))
        left_grasp = Pose3D(left_grasp_position, left_orientation, _KDL_TARGET_FRAME)
        right_grasp = Pose3D(right_grasp_position, right_orientation, _KDL_TARGET_FRAME)
        left_pregrasp = Pose3D(
            _vector_add(left_grasp_position, _vector_scale(side_axis, pregrasp_distance)),
            left_orientation, _KDL_TARGET_FRAME,
        )
        right_pregrasp = Pose3D(
            _vector_subtract(right_grasp_position, _vector_scale(side_axis, pregrasp_distance)),
            right_orientation, _KDL_TARGET_FRAME,
        )
        lift_delta = (0.0, 0.0, _config_float(self._config, "lift_distance_m"))
        left_lift = Pose3D(_vector_add(left_grasp_position, lift_delta), left_orientation, _KDL_TARGET_FRAME)
        right_lift = Pose3D(_vector_add(right_grasp_position, lift_delta), right_orientation, _KDL_TARGET_FRAME)
        retreat_delta = _vector_scale(
            forward, -_config_float(self._config, "retreat_distance_m")
        )
        left_retreat = Pose3D(_vector_add(left_lift.position_xyz, retreat_delta), left_orientation, _KDL_TARGET_FRAME)
        right_retreat = Pose3D(_vector_add(right_lift.position_xyz, retreat_delta), right_orientation, _KDL_TARGET_FRAME)

        pose_pairs = (
            ("PREGRASP", left_pregrasp, right_pregrasp),
            ("GRASP", left_grasp, right_grasp),
            ("LIFT", left_lift, right_lift),
            ("RETREAT", left_retreat, right_retreat),
        )
        ik_results = tuple(
            _solve_dual_ik(self._ik_adapter, actual_joints, left, right,
                           actual_position[0], label)
            for label, left, right in pose_pairs
        )
        pregrasp_time = _config_float(self._config, "pregrasp_duration_s")
        half_grasp_time = _config_float(self._config, "grasp_duration_s") / 2.0
        approach_time = pregrasp_time + half_grasp_time
        close_time = approach_time + half_grasp_time
        lift_time = close_time + _config_float(self._config, "lift_duration_s")
        retreat_time = lift_time + _config_float(self._config, "retreat_duration_s")
        waypoints = (
            _joint_waypoint(ArmMotionPhase.PREGRASP, pregrasp_time, ik_results[0], actual_position,
                            _config_float(self._config, "left_gripper_open"),
                            _config_float(self._config, "right_gripper_open")),
            _joint_waypoint(ArmMotionPhase.GRASP, approach_time, ik_results[1], actual_position,
                            _config_float(self._config, "left_gripper_open"),
                            _config_float(self._config, "right_gripper_open")),
            _joint_waypoint(ArmMotionPhase.GRASP, close_time, ik_results[1], actual_position,
                            _config_float(self._config, "left_gripper_closed"),
                            _config_float(self._config, "right_gripper_closed")),
            _joint_waypoint(ArmMotionPhase.LIFT, lift_time, ik_results[2], actual_position,
                            _config_float(self._config, "left_gripper_closed"),
                            _config_float(self._config, "right_gripper_closed")),
            _joint_waypoint(ArmMotionPhase.RETREAT, retreat_time, ik_results[3], actual_position,
                            _config_float(self._config, "left_gripper_closed"),
                            _config_float(self._config, "right_gripper_closed")),
        )
        _validate_waypoint_continuity(actual_position, waypoints, self._config)
        object_frame = f"object/{target.object_id.strip()}"
        context = GraspContext(
            task_id=task.task_id,
            target_body=task.target_body,
            target_class_id=target.class_id,
            object_id=target.object_id,
            object_frame=object_frame,
            object_size_xyz_m=target_size,
            object_from_left_gripper=_relative_transform(
                footprint_object, left_grasp, "left_gripper", object_frame, now_ns
            ),
            object_from_right_gripper=_relative_transform(
                footprint_object, right_grasp, "right_gripper", object_frame, now_ns
            ),
            object_orientation_world_xyzw_at_grasp=world_object.orientation_xyzw,
            orientation_observed_at_ns=target.timestamp_ns,
            planned_at_ns=now_ns,
            confirmed_at_ns=None,
            confirmed=False,
            valid=True,
        )
        grasp_target = GraspTarget(
            left_pregrasp, right_pregrasp, left_grasp, right_grasp,
            left_lift, right_lift, left_retreat, right_retreat,
            context, confidence, True,
        )
        trajectory = JointTrajectory(
            f"pick-{task.task_id}-{task.target_body}-{target.object_id.strip()}-{now_ns}",
            task.task_id, task.target_body, GlobalPhase.EXECUTE_PICK,
            waypoints, now_ns,
        )
        return grasp_target, trajectory

    def plan_place(
        self,
        task: TaskSpec,
        world_to_footprint: RigidTransform3D,
        grasp_context: GraspContext,
        actual_joints: RobotJointState,
        now_ns: int,
    ) -> tuple[PlaceTarget, JointTrajectory]:
        """以确认的计划抓取关系生成精确物体目标和四个放置路点。"""

        try:
            return self._plan_place_checked(
                task, world_to_footprint, grasp_context, actual_joints, now_ns
            )
        except Exception as exc:  # noqa: BLE001 - 规划边界必须把依赖异常收窄为失败对象
            reason = f"放置规划失败：{exc}"
            return _invalid_place_result(reason, task, now_ns)

    def _plan_place_checked(
        self,
        task: TaskSpec,
        world_to_footprint: RigidTransform3D,
        grasp_context: GraspContext,
        actual_joints: RobotJointState,
        now_ns: int,
    ) -> tuple[PlaceTarget, JointTrajectory]:
        self._config.validate_for_place()
        _strict_now(now_ns)
        _validate_task(task, now_ns)
        if task.place_type not in {"shelf_point", "table_point", "shelf_prop_side"}:
            raise ValueError("TaskSpec.place_type不是官方三类放置任务")
        if task.place_world_xyz is None:
            raise ValueError("TaskSpec.place_world_xyz不能为空")
        if task.place_frame_id != _WORLD_FRAME:
            raise ValueError('TaskSpec.place_frame_id必须严格为"world"')
        _validate_transform(
            world_to_footprint, _WORLD_FRAME, _KDL_TARGET_FRAME, now_ns,
            _config_int(self._config, "transform_max_age_ns"), "world_to_footprint",
        )
        actual_position = _validate_actual_joints(
            actual_joints, now_ns,
            _config_int(self._config, "joint_state_max_age_ns"),
        )
        _validate_actual_gripper_ranges(actual_position, self._config)
        _validate_grasp_context(
            grasp_context, task, now_ns,
            _config_int(self._config, "confirmed_context_max_age_ns"),
        )
        assert grasp_context.object_orientation_world_xyzw_at_grasp is not None
        assert grasp_context.object_from_left_gripper is not None
        assert grasp_context.object_from_right_gripper is not None
        object_goal_world = Pose3D(
            _finite_vector(task.place_world_xyz, 3, "TaskSpec.place_world_xyz"),
            _normalize_quaternion(
                grasp_context.object_orientation_world_xyzw_at_grasp,
                "GraspContext.object_orientation_world_xyzw_at_grasp",
            ),
            _WORLD_FRAME,
        )
        object_goal = _transform_pose(world_to_footprint, object_goal_world)
        _require_upright_object_orientation(
            object_goal.orientation_xyzw, "footprint放置目标物体姿态"
        )
        release_object = Pose3D(
            _vector_add(
                object_goal.position_xyz,
                (0.0, 0.0, _config_float(self._config, "release_offset_m")),
            ),
            object_goal.orientation_xyzw, _KDL_TARGET_FRAME,
        )
        preplace_object = Pose3D(
            _vector_add(
                release_object.position_xyz,
                (0.0, 0.0, _config_float(self._config, "preplace_height_m")),
            ),
            object_goal.orientation_xyzw, _KDL_TARGET_FRAME,
        )
        left_release = _compose_pose_with_transform(
            release_object, grasp_context.object_from_left_gripper
        )
        right_release = _compose_pose_with_transform(
            release_object, grasp_context.object_from_right_gripper
        )
        left_preplace = _compose_pose_with_transform(
            preplace_object, grasp_context.object_from_left_gripper
        )
        right_preplace = _compose_pose_with_transform(
            preplace_object, grasp_context.object_from_right_gripper
        )
        forward = _normalize_vector(
            (object_goal.position_xyz[0], object_goal.position_xyz[1], 0.0),
            "放置目标水平径向方向",
        )
        retreat_delta = _vector_scale(
            forward,
            -_config_float(self._config, "post_release_retreat_distance_m"),
        )
        left_retreat = Pose3D(
            _vector_add(left_release.position_xyz, retreat_delta),
            left_release.orientation_xyzw, _KDL_TARGET_FRAME,
        )
        right_retreat = Pose3D(
            _vector_add(right_release.position_xyz, retreat_delta),
            right_release.orientation_xyzw, _KDL_TARGET_FRAME,
        )
        pose_pairs = (
            ("PREPLACE", left_preplace, right_preplace),
            ("LOWER", left_release, right_release),
            ("POST_RELEASE_RETREAT", left_retreat, right_retreat),
        )
        ik_results = tuple(
            _solve_dual_ik(self._ik_adapter, actual_joints, left, right,
                           actual_position[0], label)
            for label, left, right in pose_pairs
        )
        preplace_time = _config_float(self._config, "preplace_duration_s")
        lower_time = preplace_time + _config_float(self._config, "lower_duration_s")
        release_time = lower_time + _config_float(self._config, "release_duration_s")
        retreat_time = release_time + _config_float(
            self._config, "post_release_retreat_duration_s"
        )
        waypoints = (
            _joint_waypoint(ArmMotionPhase.PREPLACE, preplace_time, ik_results[0], actual_position,
                            _config_float(self._config, "left_gripper_closed"),
                            _config_float(self._config, "right_gripper_closed")),
            _joint_waypoint(ArmMotionPhase.LOWER, lower_time, ik_results[1], actual_position,
                            _config_float(self._config, "left_gripper_closed"),
                            _config_float(self._config, "right_gripper_closed")),
            _joint_waypoint(ArmMotionPhase.RELEASE, release_time, ik_results[1], actual_position,
                            _config_float(self._config, "left_gripper_open"),
                            _config_float(self._config, "right_gripper_open")),
            _joint_waypoint(ArmMotionPhase.POST_RELEASE_RETREAT, retreat_time, ik_results[2], actual_position,
                            _config_float(self._config, "left_gripper_open"),
                            _config_float(self._config, "right_gripper_open")),
        )
        _validate_waypoint_continuity(actual_position, waypoints, self._config)
        target = PlaceTarget(
            object_goal, left_preplace, right_preplace, left_release, right_release,
            left_retreat, right_retreat, _config_float(self._config, "settle_time_s"), True,
        )
        trajectory = JointTrajectory(
            f"place-{task.task_id}-{task.target_body}-{grasp_context.object_id}-{now_ns}",
            task.task_id, task.target_body, GlobalPhase.EXECUTE_PLACE,
            waypoints, now_ns,
        )
        return target, trajectory


def _finite_scalar(value: object, name: str) -> float:
    return _finite_vector((value,), 1, name)[0]


def _strict_now(now_ns: object) -> int:
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("now_ns必须是真正非负int，不能使用bool")
    return now_ns


def _config_float(config: ArmPlanningConfig, name: str) -> float:
    value = getattr(config, name)
    if value is None:
        raise ValueError(f"ArmPlanningConfig.{name}未配置")
    return _finite_scalar(value, f"ArmPlanningConfig.{name}")


def _config_int(config: ArmPlanningConfig, name: str) -> int:
    value = getattr(config, name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"ArmPlanningConfig.{name}必须是正整数")
    return value


def _validate_fresh_stamp(
    timestamp_ns: object, now_ns: int, max_age_ns: int, name: str
) -> int:
    if type(timestamp_ns) is not int or timestamp_ns < 0:
        raise ValueError(f"{name}.timestamp_ns必须是真正非负int")
    if timestamp_ns > now_ns:
        raise ValueError(f"{name}.timestamp_ns来自未来")
    if now_ns - timestamp_ns > max_age_ns:
        raise ValueError(f"{name}已过期")
    return timestamp_ns


def _validate_transform_observation_time(
    transform: RigidTransform3D,
    observation_timestamp_ns: int,
    max_delta_ns: int,
    name: str,
) -> None:
    """要求抓取变换快照与它所转换的物体观测处于允许时间窗口。"""

    if abs(transform.timestamp_ns - observation_timestamp_ns) > max_delta_ns:
        raise ValueError(
            f"{name}.timestamp_ns与target.timestamp_ns时间不匹配，"
            f"允许差值不超过{max_delta_ns}ns"
        )


def _require_upright_object_orientation(
    orientation_xyzw: tuple[float, float, float, float], name: str
) -> None:
    """只允许物体局部Z轴在当前frame中保持竖直向上；任意yaw合法。"""

    object_up = _rotate_vector(orientation_xyzw, (0.0, 0.0, 1.0))
    if (
        abs(object_up[0]) > _VERTICAL_TOLERANCE
        or abs(object_up[1]) > _VERTICAL_TOLERANCE
        or object_up[2] < 1.0 - _VERTICAL_TOLERANCE
    ):
        raise ValueError(f"{name}必须保持物体局部Z轴竖直向上")


def _validate_task(task: object, now_ns: int) -> None:
    if not isinstance(task, TaskSpec):
        raise ValueError("task必须是TaskSpec")
    if type(task.valid) is not bool or not task.valid:
        raise ValueError(task.failure_reason or "TaskSpec无效")
    if type(task.task_id) is not int or task.task_id < 0:
        raise ValueError("TaskSpec.task_id必须是真正非负int")
    for name in ("instruction", "target_kind", "target_body", "target_color"):
        value = getattr(task, name, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"TaskSpec.{name}必须是非空字符串")
    if type(task.timestamp_ns) is not int or task.timestamp_ns < 0:
        raise ValueError("TaskSpec.timestamp_ns必须是真正非负int")
    if task.timestamp_ns > now_ns:
        raise ValueError("TaskSpec.timestamp_ns来自未来")
    if task.place_type not in {"shelf_point", "table_point", "shelf_prop_side"}:
        raise ValueError("TaskSpec.place_type不是官方三类放置任务")
    if task.place_frame_id != _WORLD_FRAME:
        raise ValueError('TaskSpec.place_frame_id必须严格为"world"')
    if task.place_world_xyz is None:
        raise ValueError("TaskSpec.place_world_xyz不能为空")
    _finite_vector(task.place_world_xyz, 3, "TaskSpec.place_world_xyz")
    radius = _finite_scalar(task.place_radius, "TaskSpec.place_radius")
    if radius <= 0.0:
        raise ValueError("TaskSpec.place_radius必须为有限正数")


def _validate_actual_joints(
    actual_joints: object, now_ns: int, max_age_ns: int
) -> tuple[float, ...]:
    if not isinstance(actual_joints, RobotJointState):
        raise ValueError("actual_joints必须是RobotJointState")
    if type(actual_joints.valid) is not bool or not actual_joints.valid:
        raise ValueError(actual_joints.failure_reason or "RobotJointState无效")
    position = _finite_vector(actual_joints.position, 17, "actual_joints.position")
    _finite_vector(actual_joints.velocity, 17, "actual_joints.velocity")
    _finite_vector(actual_joints.effort, 17, "actual_joints.effort")
    _validate_fresh_stamp(
        actual_joints.timestamp_ns, now_ns, max_age_ns, "actual_joints"
    )
    return position


def _validate_actual_gripper_ranges(
    actual_position: tuple[float, ...], config: ArmPlanningConfig
) -> None:
    """验证真实夹爪反馈位于官方配置硬范围；不裁剪或替换反馈。"""

    checks = (
        ("left", 9),
        ("right", 16),
    )
    for side, index in checks:
        lower = _config_float(config, f"{side}_gripper_min")
        upper = _config_float(config, f"{side}_gripper_max")
        value = actual_position[index]
        if not lower <= value <= upper:
            raise ValueError(
                f"actual_joints.position[{index}] {side} gripper={value}"
                f"超出配置范围[{lower}, {upper}]"
            )


def _validate_transform(
    transform: object,
    source_frame: str,
    target_frame: str,
    now_ns: int,
    max_age_ns: int,
    name: str,
) -> None:
    if not isinstance(transform, RigidTransform3D):
        raise ValueError(f"{name}必须是RigidTransform3D")
    if type(transform.valid) is not bool or not transform.valid:
        raise ValueError(transform.failure_reason or f"{name}无效")
    if transform.source_frame != source_frame or transform.target_frame != target_frame:
        raise ValueError(
            f"{name}方向必须严格为{source_frame}→{target_frame}，实际为"
            f"{transform.source_frame}→{transform.target_frame}"
        )
    if transform.translation_xyz is None or transform.rotation_xyzw is None:
        raise ValueError(f"{name}缺少平移或旋转")
    _finite_vector(transform.translation_xyz, 3, f"{name}.translation_xyz")
    _normalize_quaternion(transform.rotation_xyzw, f"{name}.rotation_xyzw")
    _validate_fresh_stamp(transform.timestamp_ns, now_ns, max_age_ns, name)


def _select_side_axis(
    object_orientation: tuple[float, float, float, float],
    object_size: tuple[float, ...],
    robot_left: tuple[float, float, float],
) -> tuple[tuple[float, float, float], float]:
    candidates: list[tuple[str, tuple[float, ...], float]] = []
    for name, local_axis, size in (
        ("X", (1.0, 0.0, 0.0), object_size[0]),
        ("Y", (0.0, 1.0, 0.0), object_size[1]),
    ):
        rotated = _rotate_vector(object_orientation, local_axis)
        horizontal = _normalize_vector((rotated[0], rotated[1], 0.0), f"物体局部{name}水平投影")
        candidates.append((name, horizontal, size))
    score_x = abs(_dot(candidates[0][1], robot_left))
    score_y = abs(_dot(candidates[1][1], robot_left))
    if score_x > score_y + _GEOMETRY_TOLERANCE:
        selected = candidates[0]
    elif score_y > score_x + _GEOMETRY_TOLERANCE:
        selected = candidates[1]
    elif candidates[0][2] < candidates[1][2] - _GEOMETRY_TOLERANCE:
        selected = candidates[0]
    elif candidates[1][2] < candidates[0][2] - _GEOMETRY_TOLERANCE:
        selected = candidates[1]
    else:
        selected = candidates[1]
    axis = selected[1]
    if _dot(axis, robot_left) < 0.0:
        axis = _vector_scale(axis, -1.0)
    return _finite_vector(axis, 3, "抓取包夹轴"), selected[2]  # type: ignore[return-value]


def _solve_dual_ik(
    adapter: OfficialKDLAdapter,
    actual_joints: RobotJointState,
    left_target: Pose3D,
    right_target: Pose3D,
    target_slide: float,
    phase_name: str,
) -> IKResult:
    try:
        result = adapter.solve_ik(
            actual_joints=actual_joints,
            left_target=left_target,
            right_target=right_target,
            target_slide=target_slide,
        )
    except Exception as exc:  # noqa: BLE001 - 官方或fake依赖失败必须原子失败
        raise ValueError(f"{phase_name}调用OfficialKDLAdapter失败：{exc}") from exc
    if not isinstance(result, IKResult):
        raise ValueError(f"{phase_name} IK返回值必须是IKResult")
    if type(result.success) is not bool or not result.success:
        raise ValueError(result.failure_reason or f"{phase_name} IK失败")
    slide = _finite_scalar(result.target_slide, f"{phase_name} IK target_slide")
    if not math.isclose(slide, target_slide, rel_tol=0.0, abs_tol=_GEOMETRY_TOLERANCE):
        raise ValueError(f"{phase_name} IK返回slide与固定实际slide不一致")
    if result.left_joint_target is None or result.right_joint_target is None:
        raise ValueError(f"{phase_name} IK必须返回完整左右双臂解")
    _finite_vector(result.left_joint_target, 6, f"{phase_name} IK左臂解")
    _finite_vector(result.right_joint_target, 6, f"{phase_name} IK右臂解")
    return result


def _joint_waypoint(
    phase: ArmMotionPhase,
    time_from_start_s: float,
    ik_result: IKResult,
    actual_position: tuple[float, ...],
    left_gripper: float,
    right_gripper: float,
) -> JointWaypoint:
    assert ik_result.left_joint_target is not None
    assert ik_result.right_joint_target is not None
    position = list(actual_position)
    position[0] = ik_result.target_slide
    position[3:9] = ik_result.left_joint_target
    position[9] = left_gripper
    position[10:16] = ik_result.right_joint_target
    position[16] = right_gripper
    mask = tuple(index not in (1, 2) for index in range(17))
    return JointWaypoint(phase, time_from_start_s, tuple(position), mask)


def _validate_waypoint_continuity(
    actual_position: tuple[float, ...],
    waypoints: tuple[JointWaypoint, ...],
    config: ArmPlanningConfig,
) -> None:
    slide_limit = _config_float(config, "max_slide_waypoint_delta_m")
    arm_limit = _config_float(config, "max_arm_waypoint_delta_rad")
    gripper_limit = _config_float(config, "max_gripper_waypoint_delta")
    previous = actual_position
    for waypoint_index, waypoint in enumerate(waypoints):
        checks = (
            ((0,), slide_limit, "slide"),
            ((3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15), arm_limit, "arm"),
            ((9, 16), gripper_limit, "gripper"),
        )
        for indices, limit, label in checks:
            for index in indices:
                if waypoint.controlled_mask[index] and abs(
                    waypoint.joint_position[index] - previous[index]
                ) > limit + _GEOMETRY_TOLERANCE:
                    raise ValueError(
                        f"路点{waypoint_index}的{label}[{index}]连续性增量超限"
                    )
        previous = waypoint.joint_position


def _validate_grasp_context(
    context: object, task: TaskSpec, now_ns: int, max_age_ns: int
) -> None:
    if not isinstance(context, GraspContext):
        raise ValueError("grasp_context必须是GraspContext")
    if type(context.valid) is not bool or not context.valid:
        raise ValueError(context.failure_reason or "GraspContext无效")
    if type(context.confirmed) is not bool or not context.confirmed:
        raise ValueError("放置规划要求confirmed GraspContext")
    if context.confirmed_at_ns is None:
        raise ValueError("confirmed GraspContext缺少confirmed_at_ns")
    _validate_fresh_stamp(context.confirmed_at_ns, now_ns, max_age_ns, "grasp_context")
    if context.task_id != task.task_id or context.target_body != task.target_body:
        raise ValueError("GraspContext与TaskSpec任务身份不匹配")
    if context.target_class_id != task.target_color:
        raise ValueError("GraspContext.target_class_id与TaskSpec.target_color不匹配")
    for name in ("target_body", "target_class_id", "object_id", "object_frame"):
        value = getattr(context, name, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"GraspContext.{name}必须是非空字符串")
    if context.object_size_xyz_m is None:
        raise ValueError("GraspContext缺少object_size_xyz_m")
    size = _finite_vector(context.object_size_xyz_m, 3, "GraspContext.object_size_xyz_m")
    if any(value <= 0.0 for value in size):
        raise ValueError("GraspContext.object_size_xyz_m三轴必须为正")
    if context.object_orientation_world_xyzw_at_grasp is None:
        raise ValueError("GraspContext缺少抓取时world朝向")
    _normalize_quaternion(
        context.object_orientation_world_xyzw_at_grasp,
        "GraspContext.object_orientation_world_xyzw_at_grasp",
    )
    if context.orientation_observed_at_ns is None:
        raise ValueError("GraspContext缺少orientation_observed_at_ns")
    if type(context.orientation_observed_at_ns) is not int or context.orientation_observed_at_ns < 0:
        raise ValueError("GraspContext.orientation_observed_at_ns必须是真正非负int")
    if type(context.planned_at_ns) is not int or context.planned_at_ns < 0:
        raise ValueError("GraspContext.planned_at_ns必须是真正非负int")
    if context.planned_at_ns > context.confirmed_at_ns:
        raise ValueError("GraspContext.planned_at_ns不能晚于confirmed_at_ns")
    if context.orientation_observed_at_ns > context.planned_at_ns:
        raise ValueError("GraspContext.orientation_observed_at_ns不能晚于planned_at_ns")
    expected = (
        ("object_from_left_gripper", "left_gripper"),
        ("object_from_right_gripper", "right_gripper"),
    )
    for name, source_frame in expected:
        transform = getattr(context, name, None)
        if not isinstance(transform, RigidTransform3D) or not transform.valid:
            raise ValueError(f"GraspContext.{name}必须是有效RigidTransform3D")
        if transform.source_frame != source_frame or transform.target_frame != context.object_frame:
            raise ValueError(
                f"GraspContext.{name}方向必须为{source_frame}→{context.object_frame}"
            )
        if transform.translation_xyz is None or transform.rotation_xyzw is None:
            raise ValueError(f"GraspContext.{name}缺少平移或旋转")
        _finite_vector(transform.translation_xyz, 3, f"GraspContext.{name}.translation_xyz")
        _normalize_quaternion(transform.rotation_xyzw, f"GraspContext.{name}.rotation_xyzw")
        if type(transform.timestamp_ns) is not int or transform.timestamp_ns < 0:
            raise ValueError(f"GraspContext.{name}.timestamp_ns必须是真正非负int")
        if transform.timestamp_ns > context.planned_at_ns:
            raise ValueError(f"GraspContext.{name}.timestamp_ns不能晚于planned_at_ns")


def _safe_failure_identity(task: object, now_ns: object) -> tuple[int, str, int]:
    task_id = getattr(task, "task_id", 0)
    if type(task_id) is not int or task_id < 0:
        task_id = 0
    target_body = getattr(task, "target_body", "")
    if not isinstance(target_body, str):
        target_body = ""
    timestamp = now_ns if type(now_ns) is int and now_ns >= 0 else 0
    return task_id, target_body, timestamp


def _invalid_grasp_result(
    reason: str, task: object, now_ns: object
) -> tuple[GraspTarget, JointTrajectory]:
    task_id, target_body, timestamp = _safe_failure_identity(task, now_ns)
    return (
        GraspTarget(None, None, None, None, None, None, None, None,
                    None, 0.0, False, reason),
        JointTrajectory("", task_id, target_body, GlobalPhase.EXECUTE_PICK,
                        (), timestamp, False, reason),
    )


def _invalid_place_result(
    reason: str, task: object, now_ns: object
) -> tuple[PlaceTarget, JointTrajectory]:
    task_id, target_body, timestamp = _safe_failure_identity(task, now_ns)
    return (
        PlaceTarget(None, None, None, None, None, None, None,
                    None, False, reason),
        JointTrajectory("", task_id, target_body, GlobalPhase.EXECUTE_PLACE,
                        (), timestamp, False, reason),
    )


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
