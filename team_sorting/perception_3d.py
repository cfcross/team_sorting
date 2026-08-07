"""深度反投影、相机外参和三维目标估计接口。

本文件负责复用官方 ``box_detect.py`` 的针孔反投影思路，并通过官方 ``MMK2FK``
提供 camera 到 world/odom 的变换接口；不负责 YOLO、ROS2 同步、目标选择或抓取规划。
``perception_node`` 在获得同步 RGB/Depth 及最近 Odom/JointState 后调用本模块。输入为
``Detection2D``、深度、内参和机器人实际状态，输出为 ``ObjectEstimate3D``。

``ObjectEstimate3D`` 表示物体中心的三维估计，不是左右夹爪末端位姿，也不是最终抓取
点或放置点。后续 ``arm_planning`` 必须结合任务、箱体尺寸、抓取方向和安全偏移，另行
计算左右夹爪目标。Odom 只给出底盘位姿；slide 和 head 关节会改变头部相机相对底盘的
位置与朝向，因此还必须使用实际 ``RobotJointState`` 和 ``MMK2FK`` 闭合坐标链。

MMK2FK、MuJoCo、SciPy 和 NumPy 均按需延迟导入。启发式中心补偿尺寸和物体局部
XYZ 完整尺寸由调用方通过两个语义独立的映射注入：前者只用于沿相机射线把可见
表面点补偿到近似几何中心，后者只在来源明确时写入 ``ObjectEstimate3D.size_xyz_m``。
未知中心补偿尺寸时明确返回无效估计；未知局部尺寸时继续输出 ``None``，禁止把两类
尺寸互相冒充。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from itertools import permutations, product
import math
from numbers import Integral, Real
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Optional

from .interfaces import (
    BaseState,
    CameraIntrinsics,
    DepthFrame,
    Detection2D,
    GraspVerification,
    ObjectEstimate3D,
    RobotJointState,
)


def _finite_number(value: Any, name: str) -> float:
    """把真实数值收窄为有限浮点数，同时拒绝容易混入数值运算的 bool。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是真实数值，不能使用 bool 或其他类型")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 无法安全转换为有限浮点数") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数，不能包含 NaN 或 Inf")
    return result


def _finite_vector(values: Any, length: int, name: str) -> tuple[float, ...]:
    """校验外部几何接口返回的定长向量，不替它猜测缺失分量。"""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} 必须包含 {length} 项数值")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} 必须包含 {length} 项数值") from exc
    if len(items) != length:
        raise ValueError(f"{name} 必须包含 {length} 项，实际为 {len(items)} 项")
    return tuple(
        _finite_number(value, f"{name}[{index}]")
        for index, value in enumerate(items)
    )


def _require_matching_base_frame(base: BaseState, output_frame: str) -> None:
    """没有显式TF时，底盘状态只能用于同名输出坐标系。"""

    base_frame = base.frame_id
    if not isinstance(base_frame, str) or not base_frame.strip():
        raise ValueError(
            "BaseState.frame_id 为空；缺少显式 TF 时不能静默重标坐标系"
        )
    if base_frame != output_frame:
        raise ValueError(
            f"BaseState.frame_id ({base_frame!r}) 与 "
            f"CameraTransformProvider.output_frame ({output_frame!r}) 不一致；"
            "缺少显式 TF 时不能静默重标坐标系"
        )


def project_pixel_to_camera(
    u: float, v: float, depth_m: float, intrinsics: CameraIntrinsics
) -> tuple[float, float, float]:
    """使用针孔模型把像素和深度反投影到相机光学坐标系。

    参数：``u/v`` 为像素坐标，``depth_m`` 为沿光轴深度（米），``intrinsics`` 的 K
    为像素单位。返回 ``(X,Y,Z)``，单位米，坐标系为相机光学系。
    相机光学坐标通常为 X 向右、Y 向下、Z 沿镜头向前；本函数只完成这一步，不做
    camera-to-world 变换。失败：输入含 bool/NaN/Inf、内参无效、深度非正或焦距
    非正时抛出 ``ValueError``。
    """

    if not intrinsics.valid:
        raise ValueError(f"相机内参无效：{intrinsics.failure_reason}")
    u_value = _finite_number(u, "像素 u")
    v_value = _finite_number(v, "像素 v")
    depth_value = _finite_number(depth_m, "深度 depth_m")
    if depth_value <= 0.0:
        raise ValueError("深度必须为正的有限米制数")
    try:
        fx = _finite_number(intrinsics.k[0], "相机焦距 fx")
        fy = _finite_number(intrinsics.k[4], "相机焦距 fy")
        cx = _finite_number(intrinsics.k[2], "相机主点 cx")
        cy = _finite_number(intrinsics.k[5], "相机主点 cy")
    except (IndexError, TypeError) as exc:
        raise ValueError("相机内参 K 缺少 fx/fy/cx/cy") from exc
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("相机焦距 fx 和 fy 必须是正的有限数")

    # 像素偏离主点多少，决定相机射线的 X/Y 方向；depth_value 给出沿 Z 轴的尺度。
    result = (
        (u_value - cx) * depth_value / fx,
        (v_value - cy) * depth_value / fy,
        depth_value,
    )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("反投影结果包含 NaN 或 Inf")
    return result


def _rotation_matrix_to_xyzw(matrix: Any) -> tuple[float, float, float, float]:
    """把右手正交旋转矩阵转换为归一化 ``xyzw`` 四元数。"""

    m00, m01, m02 = (float(value) for value in matrix[0])
    m10, m11, m12 = (float(value) for value in matrix[1])
    m20, m21, m22 = (float(value) for value in matrix[2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(max(0.0, 1.0 + m00 - m11 - m22)) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(max(0.0, 1.0 + m11 - m00 - m22)) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(max(0.0, 1.0 + m22 - m00 - m11)) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale
    norm = math.hypot(qx, qy, qz, qw)
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError("点云旋转矩阵无法转换为非零有限四元数")
    quaternion = (qx / norm, qy / norm, qz / norm, qw / norm)
    # q 与 -q 表示同一旋转；固定符号可避免跨帧无意义跳变。
    if quaternion[3] < 0.0:
        quaternion = tuple(-value for value in quaternion)
    return quaternion


def _fit_cuboid_orientation_xyzw(
    points_xyz: Any,
    size_xyz_m: tuple[float, float, float],
    max_extent_error_ratio: float,
    np: Any,
) -> Optional[tuple[float, float, float, float]]:
    """由可见点云拟合已知长方体的对称等价局部轴姿态。

    PCA 给出可见面的三条正交主轴，再枚举轴排列并以已知局部 XYZ 完整尺寸筛选。
    长方体点云无法区分绕任意局部轴 180° 翻转，因此返回离输出 frame 单位姿态最近的
    等价代表；它足以描述同一个有向包围盒，但不能声称恢复了 MJCF body 轴的符号。
    """

    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        return None
    if not bool(np.all(np.isfinite(points))):
        return None
    centered = points - np.median(points, axis=0)
    covariance = centered.T @ centered / float(points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = eigenvectors[:, order]
    if float(eigenvalues[0]) <= 1e-12 or float(eigenvalues[1]) <= 1e-12:
        return None
    projections = centered @ axes
    extents = np.percentile(projections, 97.5, axis=0) - np.percentile(
        projections, 2.5, axis=0
    )

    best: Optional[tuple[float, tuple[int, int, int]]] = None
    for axis_order in permutations((0, 1, 2)):
        observed = tuple(float(extents[index]) for index in axis_order)
        supported = [
            index
            for index, extent in enumerate(observed)
            if extent >= 0.2 * size_xyz_m[index]
        ]
        if len(supported) < 2:
            continue
        errors = [
            abs(observed[index] - size_xyz_m[index]) / size_xyz_m[index]
            for index in supported
        ]
        if max(errors) > max_extent_error_ratio:
            continue
        score = sum(errors) / len(errors)
        candidate = (score, axis_order)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None

    rotation = axes[:, best[1]]
    candidates: list[tuple[float, float, float, float]] = []
    for signs in product((-1.0, 1.0), repeat=3):
        signed = rotation * np.asarray(signs, dtype=float)
        if float(np.linalg.det(signed)) <= 0.0:
            continue
        candidates.append(_rotation_matrix_to_xyzw(signed))
    if not candidates:
        return None
    # 箱体中心对称使四个右手符号组合几何等价；选最接近单位姿态的确定性代表。
    return max(candidates, key=lambda item: (abs(item[3]), item))


@dataclass(frozen=True)
class _DepthWindowStatistics:
    """一次深度窗口访问得到的中位深度和原中心窗口有效比例。"""

    depth_m: float
    valid_fraction: float


@dataclass(frozen=True)
class _DepthWindowRequest:
    """NumPy 转换前已完成校验的深度窗口参数。"""

    radius: int
    bbox: tuple[float, float, float, float]
    unit_scale_m: float


def _validate_depth_window_request(
    depth: DepthFrame,
    bbox_xyxy: tuple[float, float, float, float],
    radius_px: int,
) -> _DepthWindowRequest:
    """保持原失败优先级，在接触图像数组前校验轻量输入。"""

    if not depth.valid:
        raise ValueError(f"深度帧无效：{depth.failure_reason}")
    if isinstance(radius_px, bool) or not isinstance(radius_px, Integral):
        raise ValueError("radius_px 必须是非负整数，不能使用 bool")
    radius = int(radius_px)
    if radius < 0:
        raise ValueError("radius_px 必须是非负整数")
    bbox = _finite_vector(bbox_xyxy, 4, "bbox_xyxy")
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox_xyxy 必须满足 x1>x0 且 y1>y0")
    unit_scale_m = _finite_number(depth.unit_scale_m, "depth.unit_scale_m")
    if unit_scale_m <= 0.0:
        raise ValueError("depth.unit_scale_m 必须是正的有限数")
    return _DepthWindowRequest(radius, bbox, unit_scale_m)


def _depth_window_statistics(
    depth: DepthFrame,
    request: _DepthWindowRequest,
    image: Any,
    np: Any,
) -> _DepthWindowStatistics:
    """只访问一次深度数组窗口，同时保持既有深度与置信度计算语义。"""

    del depth
    radius = request.radius
    bbox = request.bbox
    x0, y0, x1, y1 = bbox
    unit_scale_m = request.unit_scale_m
    if image.ndim != 2:
        raise ValueError(f"深度图必须是严格二维数组，实际维度={image.ndim}")
    height, width = image.shape
    if height <= 0 or width <= 0:
        raise ValueError("深度图宽高必须大于0")
    if x1 <= 0.0 or y1 <= 0.0 or x0 >= width or y0 >= height:
        raise ValueError("bbox 完全位于深度图范围之外")

    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        raise ValueError("bbox 在像素网格上的宽和高必须至少为1像素")
    center_x = int(round((x0 + x1) / 2.0))
    center_y = int(round((y0 + y1) / 2.0))
    window_xa = max(0, center_x - radius)
    window_xb = min(width, center_x + radius + 1)
    window_ya = max(0, center_y - radius)
    window_yb = min(height, center_y + radius + 1)
    if window_xa >= window_xb or window_ya >= window_yb:
        raise ValueError("bbox 中心无法在深度图内形成有效采样窗口")
    # 深度数组只切片一次。中位深度使用其中的 bbox 交集子区域；有效比例仍使用
    # 原中心窗口，保持优化前 confidence 的数值完全不变。
    try:
        window = image[
            window_ya:window_yb,
            window_xa:window_xb,
        ].astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bbox 中心窗口的深度值无法转换为数值") from exc
    full_valid = np.isfinite(window) & (window > 0.0)
    valid_fraction = float(np.count_nonzero(full_valid) / window.size)

    # bbox 使用半开区间语义；ceil 可避免把 bbox 外的左/上边缘像素纳入采样，
    # 同时让整数 xyxy 框精确映射为 NumPy 切片。
    bbox_xa, bbox_xb = math.ceil(x0), math.ceil(x1)
    bbox_ya, bbox_yb = math.ceil(y0), math.ceil(y1)
    xa = max(window_xa, bbox_xa)
    xb = min(window_xb, bbox_xb)
    ya = max(window_ya, bbox_ya)
    yb = min(window_yb, bbox_yb)
    if xa >= xb or ya >= yb:
        raise ValueError("bbox 与中心深度窗口没有至少1像素的有效交集")
    patch = window[
        ya - window_ya : yb - window_ya,
        xa - window_xa : xb - window_xa,
    ]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise ValueError("bbox 中心窗口没有有效深度")
    result_m = float(np.median(valid)) * unit_scale_m
    if not math.isfinite(result_m) or result_m <= 0.0:
        raise ValueError("中位深度换算结果必须是正的有限米制数")
    return _DepthWindowStatistics(result_m, valid_fraction)


def median_depth_m(
    depth: DepthFrame, bbox_xyxy: tuple[float, float, float, float], radius_px: int = 4
) -> float:
    """在二维框与中心窗口的交集内读取非零中位深度。

    参数：对齐深度图、像素 bbox 和非负窗口半径（像素）。采样绝不越过 bbox，
    返回米制中位深度。
    中位数比单像素更不易受深度孔洞和少量离群值影响，但它通常仍落在物体可见表面，
    不能直接声称是物体中心深度。失败：NumPy 缺失、二维图像/bbox/单位比例错误或
    窗口内无有效深度时抛出 ``ValueError``，不会返回虚构距离。
    """

    request = _validate_depth_window_request(depth, bbox_xyxy, radius_px)
    try:
        import numpy as np  # 延迟导入，避免纯接口模块强依赖视觉环境
    except ImportError as exc:
        raise ValueError("深度处理中缺少 NumPy") from exc
    try:
        image = np.asarray(depth.image)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth.image 无法转换为深度数组") from exc
    return _depth_window_statistics(
        depth,
        request,
        image,
        np,
    ).depth_m


@dataclass(frozen=True)
class _HeadCameraPose:
    """当前帧头部相机在输出坐标系中的位姿。"""

    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


class CameraTransformProvider:
    """官方 MMK2FK 相机外参的薄适配器。

    参数：官方根目录、MJCF 路径、模块名和 FK 输出 frame。输入为相机光学系米制点、
    Odom 底盘状态和 17 维实际关节状态；输出为 ``output_frame`` 中的米制三维点。
    失败：MMK2FK/MuJoCo/资源缺失、frame 配置不明确或实际状态无效时抛出
    ``RuntimeError``/``ValueError``。本适配器不重写官方 FK。
    """

    def __init__(
        self,
        official_root: str = "",
        mjcf_path: str = "",
        module_name: str = "discoverse.robots.mmk2.mmk2_fk",
        output_frame: str = "world",
    ) -> None:
        """保存外部依赖配置，暂不加载 MMK2FK。

        路径可由 ``MATERIAL_SORTING_OFFICIAL_ROOT`` 和 ``TEAM_SORTING_MJCF`` 覆盖。
        ``output_frame`` 必须与官方 FK 真实输出一致；world/odom 是否重合仍待确认。
        """

        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError("module_name 必须是非空字符串")
        if not isinstance(output_frame, str) or not output_frame.strip():
            raise ValueError("output_frame 必须是非空字符串")
        self.official_root = official_root
        self.mjcf_path = mjcf_path
        self.module_name = module_name.strip()
        self.output_frame = output_frame.strip()
        self._fk: Any = None
        self._searched: list[str] = []

    def self_check(self) -> None:
        """延迟导入 MMK2FK，并用配置 MJCF 创建官方 FK 对象。

        参数：无。返回：成功时无返回值。
        单位/坐标系：仅检查资源，不输出空间量。
        失败：缺少 DISCOVERSE、MMK2FK、MuJoCo 或 MJCF 时抛出 ``RuntimeError``，错误
        中列出搜索路径及 ``MATERIAL_SORTING_OFFICIAL_ROOT``/``TEAM_SORTING_MJCF``。
        """

        # 重复自检必须先撤销旧成功状态；本轮失败后绝不能继续使用上一次的 FK。
        self._searched.clear()
        self._fk = None

        root_text = os.getenv("MATERIAL_SORTING_OFFICIAL_ROOT", self.official_root).strip()
        mjcf_text = os.getenv("TEAM_SORTING_MJCF", self.mjcf_path).strip()
        roots: list[Path] = []
        if root_text:
            root = Path(root_text).expanduser()
            for candidate in (
                root,
                root / "material_sorting",
                root / "examples" / "material_sorting",
            ):
                if candidate not in roots:
                    roots.append(candidate)
            for candidate in (root, root.parent):
                self._searched.append(f"python_path:{candidate}")
                if candidate.is_dir() and str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
        module_names = list(dict.fromkeys((self.module_name, "mmk2_fk")))
        module = None
        errors: list[str] = []
        for name in module_names:
            self._searched.append(f"module:{name}")
            try:
                module = importlib.import_module(name)
                break
            except Exception as exc:  # noqa: BLE001 - 汇总第三方导入错误
                errors.append(f"{name}: {exc}")
        if module is None or not hasattr(module, "MMK2FK"):
            raise RuntimeError(
                "无法导入官方 MMK2FK；"
                f"搜索={module_names}，错误={errors}。请设置 MATERIAL_SORTING_OFFICIAL_ROOT。"
            )

        source = self._resolve_mjcf(mjcf_text, roots)
        if source is None:
            raise RuntimeError(
                "找不到 MMK2FK 所需 MJCF；"
                f"搜索={self._searched}。请设置 TEAM_SORTING_MJCF 或 official.mjcf_path。"
            )
        load_path, temporary_path = self._prepare_mjcf(source)
        fk: Any = None
        init_error: Optional[Exception] = None
        try:
            fk = module.MMK2FK(str(load_path))
        except Exception as exc:  # noqa: BLE001 - 提供资源上下文
            init_error = exc
        cleanup_error: Optional[OSError] = None
        if temporary_path is not None:
            try:
                # 官方 MMK2FK 在构造阶段已由 MuJoCo 完整读取 XML，之后可删除临时展开文件。
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_error = exc
        if init_error is not None:
            details = f"；临时文件清理失败：{cleanup_error}" if cleanup_error else ""
            raise RuntimeError(
                f"MMK2FK 初始化失败，MJCF={source}：{init_error}{details}"
            ) from init_error
        if cleanup_error is not None:
            raise RuntimeError(f"MMK2FK 已初始化，但临时 MJCF 清理失败：{cleanup_error}")
        self._fk = fk

    def camera_to_output(
        self,
        camera_point_xyz: tuple[float, float, float],
        base: BaseState,
        joints: RobotJointState,
    ) -> tuple[float, float, float]:
        """把相机光学系三维点转换到官方 FK 输出坐标系。

        参数：相机点单位米；底盘姿态来自 Odom；关节为 17 维实际反馈。返回值单位米，
        坐标系为 ``output_frame``。失败：未自检、状态无效、依赖接口变化或数值异常时
        抛出 ``RuntimeError``/``ValueError``。
        """

        # 保留旧方法的失败顺序：共享状态先校验，随后校验逐点输入，再执行 FK。
        if self._fk is None:
            raise RuntimeError("CameraTransformProvider 尚未通过 self_check")
        if not base.valid:
            raise ValueError(f"底盘状态无效，不能计算相机外参：{base.failure_reason}")
        if not joints.valid:
            raise ValueError(f"实际关节状态无效，不能计算相机外参：{joints.failure_reason}")
        _require_matching_base_frame(base, self.output_frame)
        camera_point = _finite_vector(camera_point_xyz, 3, "camera_point_xyz")
        pose = self.compute_head_camera_pose(base, joints)
        return self.transform_camera_point(camera_point, pose)

    def compute_head_camera_pose(
        self,
        base: BaseState,
        joints: RobotJointState,
    ) -> _HeadCameraPose:
        """用当前底盘与关节反馈执行一次官方 FK，返回本帧可复用的相机位姿。"""

        if self._fk is None:
            raise RuntimeError("CameraTransformProvider 尚未通过 self_check")
        if not base.valid:
            raise ValueError(f"底盘状态无效，不能计算相机外参：{base.failure_reason}")
        if not joints.valid:
            raise ValueError(f"实际关节状态无效，不能计算相机外参：{joints.failure_reason}")
        _require_matching_base_frame(base, self.output_frame)
        base_position = _finite_vector(base.position_xyz, 3, "BaseState.position_xyz")
        qx, qy, qz, qw = _finite_vector(
            base.orientation_xyzw, 4, "BaseState.orientation_xyzw"
        )
        base_quaternion_norm = math.hypot(qx, qy, qz, qw)
        if base_quaternion_norm < 1e-12:
            raise ValueError("BaseState.orientation_xyzw 四元数范数为零")
        qx, qy, qz, qw = (
            qx / base_quaternion_norm,
            qy / base_quaternion_norm,
            qz / base_quaternion_norm,
            qw / base_quaternion_norm,
        )
        joint_position = _finite_vector(joints.position, 17, "RobotJointState.position")
        try:
            # ROS/Odom 使用 xyzw，官方 MMK2FK 明确接收 wxyz，不能直接传同一顺序。
            self._fk.set_base_pose(base_position, [qw, qx, qy, qz])
            # slide 和 head 会改变头部相机相对底盘的位姿；左右夹爪索引 9/16 跳过。
            self._fk.set_slide_joint(joint_position[0])
            self._fk.set_head_joints(joint_position[1:3])
            self._fk.set_left_arm_joints(joint_position[3:9])
            self._fk.set_right_arm_joints(joint_position[10:16])
            camera_position, camera_quaternion_wxyz = self._fk.get_head_camera_pose()
        except Exception as exc:  # noqa: BLE001 - 官方接口错误转为清晰异常
            raise RuntimeError(f"调用官方 MMK2FK 计算头部相机位姿失败：{exc}") from exc
        try:
            position = _finite_vector(camera_position, 3, "MMK2FK camera_position")
            quaternion = _finite_vector(
                camera_quaternion_wxyz, 4, "MMK2FK camera_quaternion_wxyz"
            )
        except ValueError as exc:
            raise RuntimeError(f"MMK2FK 返回的头部相机位姿无效：{exc}") from exc
        return _HeadCameraPose(position, quaternion)

    def transform_camera_point(
        self,
        camera_point_xyz: tuple[float, float, float],
        pose: _HeadCameraPose,
    ) -> tuple[float, float, float]:
        """使用已计算的当前帧相机位姿变换单个点，不再次执行官方 FK。"""

        camera_point = _finite_vector(camera_point_xyz, 3, "camera_point_xyz")
        try:
            position = _finite_vector(
                pose.position, 3, "MMK2FK camera_position"
            )
            quaternion = _finite_vector(
                pose.quaternion_wxyz, 4, "MMK2FK camera_quaternion_wxyz"
            )
            # headeye site 已含相机光学轴翻转；这里只按官方姿态旋转，再加相机世界位置。
            rotated = _rotate_by_wxyz(camera_point, quaternion)
        except ValueError as exc:
            raise RuntimeError(f"MMK2FK 返回的头部相机位姿无效：{exc}") from exc
        result = tuple(position[index] + rotated[index] for index in range(3))
        if not all(math.isfinite(value) for value in result):
            raise RuntimeError("MMK2FK 返回了非有限相机变换结果")
        return result

    def _resolve_mjcf(self, mjcf_text: str, roots: list[Path]) -> Optional[Path]:
        candidates: list[Path] = []
        if mjcf_text:
            candidates.append(Path(mjcf_text).expanduser())
        for root in roots:
            candidates.extend(
                [
                    root / "mjcf" / "material_competition.xml",
                    root / "material_sorting" / "mjcf" / "material_competition.xml",
                    root
                    / "examples"
                    / "material_sorting"
                    / "mjcf"
                    / "material_competition.xml",
                ]
            )
        for candidate in dict.fromkeys(candidates):
            self._searched.append(str(candidate))
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    return candidate
            except OSError as exc:
                self._searched.append(f"检查路径失败:{candidate}:{exc}")
        return None

    @staticmethod
    def _prepare_mjcf(source: Path) -> tuple[Path, Optional[Path]]:
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"无法以 UTF-8 读取 MJCF {source}：{exc}") from exc
        if "__REPO_ROOT__" not in text:
            return source, None
        # 官方模板中的占位符指 material_sorting 任务目录，而不是团队仓库或临时目录。
        task_dir = source.parent.parent if source.parent.name == "mjcf" else source.parent
        rendered = text.replace("__REPO_ROOT__", str(task_dir))
        temporary_path: Optional[Path] = None
        try:
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".xml",
                prefix="team_sorting_fk_",
                delete=False,
                encoding="utf-8",
            )
            temporary_path = Path(handle.name)
            with handle:
                handle.write(rendered)
        except (OSError, UnicodeError) as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise RuntimeError(f"无法写入 MMK2FK 临时 MJCF：{exc}") from exc
        return temporary_path, temporary_path


@dataclass
class _Track:
    """单个稳定二维轨迹在输出 frame 中的米制三维 EMA 状态。"""

    class_id: str
    ema: list[float]
    count: int
    last_ts_ns: int
    last_detection_ts_ns: int
    bbox_center_xy: tuple[float, float]
    last_surface_xyz: tuple[float, float, float]
    pose_quaternion_xyzw: Optional[list[float]] = None
    pose_count: int = 0
    last_pose_position_xyz: Optional[tuple[float, float, float]] = None


@dataclass(frozen=True)
class _IdentityCandidate:
    """当前检测用于稳定ID一致性检查的未补偿表面点。"""

    bbox_center_xy: tuple[float, float]
    surface_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class _SurfaceProbe:
    """单个检测一次深度窗口统计与反投影的可复用结果。"""

    u: float
    v: float
    depth_m: float
    valid_fraction: float
    camera_surface_xyz: tuple[float, float, float]
    camera_points_xyz: Any = None


class _SurfaceProbeError(ValueError):
    """保留原逐检测阶段前缀的内部预计算失败。"""


class Perception3DEstimator:
    """把二维框转换为输出 frame 中的物体三维中心估计。

    每条检测独立执行中心窗口中位深度、针孔反投影、已知尺寸中心补偿和官方相机外参
    变换。只使用 ``Detection2D.track_id`` 维持稳定的一对一三维 EMA；没有稳定
    编号的兼容输入仍可生成当前帧估计，但不写入持久历史，绝不退回类别/固定像素
    网格等不可靠身份。尺寸未知、输入不同步、frame 不一致、帧陈旧或三维跳变时
    返回带明确原因的无效估计，绝不把可见表面点或旧 EMA 冒充当前物体中心。
    """

    def __init__(
        self,
        transform_provider: CameraTransformProvider,
        *,
        depth_radius_px: int = 4,
        ema_alpha: float = 0.5,
        converge_frames: int = 5,
        max_track_age_s: float = 1.0,
        max_input_skew_s: float = 0.05,
        max_position_jump_m: float = 1.0,
        object_dimensions_m: Optional[
            dict[str, tuple[float, float, float]]
        ] = None,
        object_local_size_xyz_m: Optional[
            dict[str, tuple[float, float, float]]
        ] = None,
        ambiguity_ratio: float = 2.0,
        center_compensation_mode: str = "degraded",
        heuristic_center_reliability: float = 0.5,
        pose_refinement_enabled: bool = False,
        pose_min_points: int = 64,
        pose_required_frames: int = 3,
        pose_depth_band_m: float = 0.08,
        pose_max_position_delta_m: float = 0.03,
        pose_max_angular_delta_rad: float = 0.20,
        pose_max_extent_error_ratio: float = 0.45,
    ) -> None:
        """保存三维估计参数，构造阶段不加载任何官方依赖。

        ``object_dimensions_m`` 的值依次为宽、高、沿相机视线近似深度，单位米；它只
        服务当前启发式中心补偿，不是经过frame语义确认的物体局部XYZ尺寸生产源。
        ``object_local_size_xyz_m`` 的值是物体局部坐标系下完整 XYZ 三轴尺寸，单位
        米；只有该独立来源明确提供对应类别时，结果才填写 ``size_xyz_m``。随机 yaw
        不会交换这三个局部轴，也不会从 ``object_dimensions_m`` 猜测缺失尺寸。
        ``max_track_age_s`` 为轨迹超时秒数，``max_input_skew_s`` 为
        Detection/Depth/CameraInfo 最大绝对时间差秒数，``max_position_jump_m``
        为相邻有效轨迹点允许的最大三维跳变。``ambiguity_ratio`` 控制稳定ID与其他
        同类历史轨迹的距离比判定。中心补偿默认是降级的启发式估计；``strict`` 模式
        会拒绝未经真值验证的中心补偿。``pose_*`` 参数控制框内点云深度带、最少点数、
        连续收敛帧数及位置/角度/尺寸门限；该能力默认关闭。失败：参数类型、范围或物体
        尺寸不合法时抛出 ``ValueError``。
        """

        if isinstance(depth_radius_px, bool) or not isinstance(
            depth_radius_px, Integral
        ):
            raise ValueError("depth_radius_px 必须是非负整数")
        if int(depth_radius_px) < 0:
            raise ValueError("depth_radius_px 必须是非负整数")
        if isinstance(converge_frames, bool) or not isinstance(
            converge_frames, Integral
        ):
            raise ValueError("converge_frames 必须是正整数")
        if int(converge_frames) <= 0:
            raise ValueError("converge_frames 必须是正整数")
        if type(pose_refinement_enabled) is not bool:
            raise ValueError("pose_refinement_enabled 必须是布尔值")
        for value, name in (
            (pose_min_points, "pose_min_points"),
            (pose_required_frames, "pose_required_frames"),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
                raise ValueError(f"{name} 必须是正整数")

        alpha = _finite_number(ema_alpha, "ema_alpha")
        max_age_s = _finite_number(max_track_age_s, "max_track_age_s")
        input_skew_s = _finite_number(max_input_skew_s, "max_input_skew_s")
        position_jump_m = _finite_number(
            max_position_jump_m, "max_position_jump_m"
        )
        identity_ambiguity_ratio = _finite_number(
            ambiguity_ratio, "ambiguity_ratio"
        )
        center_reliability = _finite_number(
            heuristic_center_reliability, "heuristic_center_reliability"
        )
        pose_depth_band = _finite_number(pose_depth_band_m, "pose_depth_band_m")
        pose_position_delta = _finite_number(
            pose_max_position_delta_m, "pose_max_position_delta_m"
        )
        pose_angular_delta = _finite_number(
            pose_max_angular_delta_rad, "pose_max_angular_delta_rad"
        )
        pose_extent_error = _finite_number(
            pose_max_extent_error_ratio, "pose_max_extent_error_ratio"
        )
        if not 0.0 < alpha <= 1.0:
            raise ValueError("ema_alpha 必须位于 (0, 1] 范围")
        if max_age_s <= 0.0:
            raise ValueError("max_track_age_s 必须是正的有限秒数")
        if input_skew_s <= 0.0:
            raise ValueError("max_input_skew_s 必须是非零正有限秒数")
        if position_jump_m <= 0.0:
            raise ValueError("max_position_jump_m 必须是正的有限米数")
        if identity_ambiguity_ratio <= 1.0:
            raise ValueError("ambiguity_ratio 必须是大于1的有限数")
        if (
            not isinstance(center_compensation_mode, str)
            or center_compensation_mode not in {"degraded", "strict"}
        ):
            raise ValueError(
                "center_compensation_mode 只允许 'degraded' 或 'strict'"
            )
        if not 0.0 < center_reliability <= 1.0:
            raise ValueError(
                "heuristic_center_reliability 必须位于 (0, 1] 范围"
            )
        if pose_depth_band <= 0.0 or pose_position_delta <= 0.0:
            raise ValueError("点云深度带和refine位置阈值必须是正的有限米数")
        if not 0.0 < pose_angular_delta <= math.pi:
            raise ValueError("pose_max_angular_delta_rad 必须位于 (0, pi] 范围")
        if not 0.0 < pose_extent_error < 1.0:
            raise ValueError("pose_max_extent_error_ratio 必须位于 (0, 1) 范围")

        dimensions = dict(object_dimensions_m) if object_dimensions_m else {}
        normalized_dimensions: dict[str, tuple[float, float, float]] = {}
        for class_id, values in dimensions.items():
            if not isinstance(class_id, str) or not class_id:
                raise ValueError("object_dimensions_m 的类别键必须是非空字符串")
            width, height, depth_extent = _finite_vector(
                values, 3, f"object_dimensions_m[{class_id!r}]"
            )
            if width <= 0.0 or height <= 0.0 or depth_extent <= 0.0:
                raise ValueError(
                    f"object_dimensions_m[{class_id!r}] 的宽、高、深必须均为正数"
                )
            normalized_dimensions[class_id] = (width, height, depth_extent)

        local_sizes = (
            dict(object_local_size_xyz_m) if object_local_size_xyz_m else {}
        )
        normalized_local_sizes: dict[str, tuple[float, float, float]] = {}
        for class_id, values in local_sizes.items():
            if (
                not isinstance(class_id, str)
                or not class_id.strip()
                or class_id != class_id.strip()
            ):
                raise ValueError(
                    "object_local_size_xyz_m 的类别键必须是无首尾空白的非空字符串"
                )
            size_x, size_y, size_z = _finite_vector(
                values,
                3,
                f"object_local_size_xyz_m[{class_id!r}]",
            )
            if size_x <= 0.0 or size_y <= 0.0 or size_z <= 0.0:
                raise ValueError(
                    f"object_local_size_xyz_m[{class_id!r}] 的局部XYZ完整尺寸"
                    "必须均为正数"
                )
            normalized_local_sizes[class_id] = (size_x, size_y, size_z)

        self.transform_provider = transform_provider
        self.depth_radius_px = int(depth_radius_px)
        self.ema_alpha = alpha
        self.converge_frames = int(converge_frames)
        self.max_track_age_s = max_age_s
        self.max_input_skew_s = input_skew_s
        self.max_position_jump_m = position_jump_m
        self.ambiguity_ratio = identity_ambiguity_ratio
        self.center_compensation_mode = center_compensation_mode
        self.heuristic_center_reliability = center_reliability
        self.pose_refinement_enabled = pose_refinement_enabled
        self.pose_min_points = int(pose_min_points)
        self.pose_required_frames = int(pose_required_frames)
        self.pose_depth_band_m = pose_depth_band
        self.pose_max_position_delta_m = pose_position_delta
        self.pose_max_angular_delta_rad = pose_angular_delta
        self.pose_max_extent_error_ratio = pose_extent_error
        self._dims = normalized_dimensions
        self._local_sizes = normalized_local_sizes
        self._tracks: dict[str, _Track] = {}
        self._last_frame_ts_ns: Optional[int] = None
        self._last_detection_frame_ts_ns: Optional[int] = None

    def _uses_legacy_point_transform_provider(self) -> bool:
        """兼容只覆写旧 ``camera_to_output`` 公共入口的 Provider 子类。"""

        provider_type = type(self.transform_provider)
        return (
            provider_type.camera_to_output
            is not CameraTransformProvider.camera_to_output
            and provider_type.compute_head_camera_pose
            is CameraTransformProvider.compute_head_camera_pose
            and provider_type.transform_camera_point
            is CameraTransformProvider.transform_camera_point
        )

    def estimate(
        self,
        detections: tuple[Detection2D, ...],
        depth: DepthFrame,
        intrinsics: CameraIntrinsics,
        base: BaseState,
        joints: RobotJointState,
    ) -> tuple[ObjectEstimate3D, ...]:
        """返回与输入检测顺序一致的三维估计元组。

        输入深度原始值通过 ``DepthFrame.unit_scale_m`` 换算为米；输出位置单位米，
        frame 由 ``transform_provider.output_frame`` 指定，时间戳使用当前有效深度帧
        时间。Depth/CameraInfo 的 frame 或时间不同步会使整批无效；单条 Detection
        时间、frame、深度或轨迹失败只使对应结果无效，不会中断同批其他检测。
        """

        output_frame = self.transform_provider.output_frame
        failure_timestamp_ns = self._safe_failure_timestamp(
            getattr(depth, "timestamp_ns", 0)
        )
        try:
            (
                timestamp_ns,
                intrinsics_timestamp_ns,
            ) = self._validate_batch_inputs(depth, intrinsics)
        except ValueError as exc:
            return tuple(
                self._failure(
                    detection.class_id,
                    output_frame,
                    failure_timestamp_ns,
                    f"感知输入不同步：{exc}",
                )
                for detection in detections
            )

        if (
            self._last_frame_ts_ns is not None
            and timestamp_ns <= self._last_frame_ts_ns
        ):
            return tuple(
                self._failure(
                    detection.class_id,
                    output_frame,
                    timestamp_ns,
                    (
                        "陈旧感知帧：DepthFrame.timestamp_ns="
                        f"{timestamp_ns} 未严格晚于最近有效帧 "
                        f"{self._last_frame_ts_ns}"
                    ),
                )
                for detection in detections
            )

        self._remove_expired_tracks(timestamp_ns)
        context_errors: list[str] = [
            self._detection_context_error(
                detection,
                depth.frame_id,
                timestamp_ns,
                intrinsics_timestamp_ns,
            )
            for detection in detections
        ]
        current_detection_timestamps = {
            int(detection.timestamp_ns)
            for index, detection in enumerate(detections)
            if not context_errors[index] and detection.valid
        }
        if len(current_detection_timestamps) > 1:
            reason = (
                "二维检测上下文无效：同一批Detection2D必须来自同一RGB时间戳，"
                f"实际为 {sorted(current_detection_timestamps)}"
            )
            for index, detection in enumerate(detections):
                if not context_errors[index] and detection.valid:
                    context_errors[index] = reason
            current_detection_timestamps.clear()
        if current_detection_timestamps:
            current_detection_timestamp_ns = next(iter(current_detection_timestamps))
            if (
                self._last_detection_frame_ts_ns is not None
                and current_detection_timestamp_ns
                <= self._last_detection_frame_ts_ns
            ):
                reason = (
                    "陈旧二维检测帧：Detection2D.timestamp_ns="
                    f"{current_detection_timestamp_ns} 未严格晚于最近有效检测帧 "
                    f"{self._last_detection_frame_ts_ns}"
                )
                for index, detection in enumerate(detections):
                    if not context_errors[index] and detection.valid:
                        context_errors[index] = reason
                current_detection_timestamps.clear()
        associations, association_errors = self._associate_tracks(
            detections, context_errors
        )
        probes: dict[int, _SurfaceProbe] = {}
        probe_errors: dict[int, str] = {}
        depth_requests: dict[int, _DepthWindowRequest] = {}
        eligible_indices = [
            index
            for index, detection in enumerate(detections)
            if (
                detection.valid
                and not context_errors[index]
                and index not in association_errors
            )
        ]
        for index in eligible_indices:
            try:
                depth_requests[index] = _validate_depth_window_request(
                    depth,
                    detections[index].bbox_xyxy,
                    self.depth_radius_px,
                )
            except ValueError as exc:
                probe_errors[index] = f"深度提取失败：{exc}"
        if depth_requests:
            try:
                import numpy as np
            except ImportError:
                for index in depth_requests:
                    probe_errors[index] = "深度提取失败：深度处理中缺少 NumPy"
            else:
                try:
                    depth_array = np.asarray(depth.image)
                except (TypeError, ValueError):
                    for index in depth_requests:
                        probe_errors[index] = (
                            "深度提取失败：depth.image 无法转换为深度数组"
                        )
                else:
                    for index, request in depth_requests.items():
                        try:
                            probes[index] = self._surface_probe(
                                depth,
                                detections[index],
                                intrinsics,
                                request,
                                depth_array,
                                np,
                            )
                        except _SurfaceProbeError as exc:
                            probe_errors[index] = str(exc)

        head_pose: Optional[_HeadCameraPose] = None
        head_pose_error = ""
        legacy_point_transform = self._uses_legacy_point_transform_provider()
        if probes and not legacy_point_transform:
            try:
                head_pose = self.transform_provider.compute_head_camera_pose(
                    base, joints
                )
            except (ValueError, RuntimeError) as exc:
                head_pose_error = f"坐标变换失败：{exc}"
        identity_candidates, identity_errors = self._identity_consistency(
            detections,
            associations,
            context_errors,
            association_errors,
            probes,
            head_pose,
            base,
            joints,
            legacy_point_transform,
        )
        self._last_frame_ts_ns = timestamp_ns
        if current_detection_timestamps:
            self._last_detection_frame_ts_ns = next(
                iter(current_detection_timestamps)
            )

        results: list[ObjectEstimate3D] = []
        for index, detection in enumerate(detections):
            context_error = (
                context_errors[index]
                or association_errors.get(index, "")
                or identity_errors.get(index, "")
            )
            if context_error:
                results.append(
                    self._failure(
                        detection.class_id,
                        output_frame,
                        timestamp_ns,
                        context_error,
                    )
                )
                continue
            results.append(
                self._estimate_one(
                    detection,
                    associations[index],
                    output_frame,
                    timestamp_ns,
                    identity_candidates.get(index),
                    probes.get(index),
                    probe_errors.get(index, ""),
                    head_pose,
                    head_pose_error,
                    base,
                    joints,
                    legacy_point_transform,
                )
            )
        return tuple(results)

    def reset_tracks(self) -> None:
        """清空全部三维 EMA 轨迹，用于切换场景或确认长时间失跟后重置。"""

        self._tracks.clear()
        self._last_frame_ts_ns = None
        self._last_detection_frame_ts_ns = None

    def _estimate_one(
        self,
        detection: Detection2D,
        track_key: Optional[str],
        output_frame: str,
        timestamp_ns: int,
        identity_candidate: Optional[_IdentityCandidate],
        probe: Optional[_SurfaceProbe],
        probe_error: str,
        head_pose: Optional[_HeadCameraPose],
        head_pose_error: str,
        base: BaseState,
        joints: RobotJointState,
        legacy_point_transform: bool,
    ) -> ObjectEstimate3D:
        """独立处理一条检测，并把预期的数据错误转换为无效估计。"""

        if not detection.valid:
            reason = detection.failure_reason or "上游未提供原因"
            return self._failure(
                detection.class_id,
                output_frame,
                timestamp_ns,
                f"二维检测无效：{reason}",
            )

        if probe is None:
            return self._failure(
                detection.class_id,
                output_frame,
                timestamp_ns,
                probe_error or "深度提取失败：未生成深度表面探针",
            )

        compensated_point, compensated = self._compensate_to_center(
            detection.class_id, probe.camera_surface_xyz
        )
        if not compensated:
            return self._failure(
                detection.class_id,
                output_frame,
                timestamp_ns,
                (
                    "物体中心补偿失败："
                    f"类别 {detection.class_id!r} 未配置可靠物体尺寸，"
                    "当前深度点仅代表可见表面"
                ),
            )
        if self.center_compensation_mode == "strict":
            return self._failure(
                detection.class_id,
                output_frame,
                timestamp_ns,
                (
                    "surface point only, center compensation not validated；"
                    "严格模式拒绝启发式表面到中心补偿"
                ),
            )
        if head_pose is None and not legacy_point_transform:
            return self._failure(
                detection.class_id,
                output_frame,
                timestamp_ns,
                head_pose_error or "坐标变换失败：未生成当前帧相机位姿",
            )
        try:
            if legacy_point_transform:
                transformed_point = self.transform_provider.camera_to_output(
                    compensated_point,
                    base,
                    joints,
                )
            else:
                assert head_pose is not None
                transformed_point = self.transform_provider.transform_camera_point(
                    compensated_point, head_pose
                )
            world_point = _finite_vector(
                transformed_point,
                3,
                "camera_to_output 返回值",
            )
        except (ValueError, RuntimeError) as exc:
            return self._failure(
                detection.class_id,
                output_frame,
                timestamp_ns,
                f"坐标变换失败：{exc}",
            )

        try:
            confidence = _finite_number(
                detection.confidence, "Detection2D.confidence"
            )
        except ValueError as exc:
            return self._failure(
                detection.class_id,
                output_frame,
                timestamp_ns,
                f"置信度计算失败：{exc}",
            )
        filtered_point, count, track_error = self._update_track(
            track_key,
            detection,
            world_point,
            timestamp_ns,
            identity_candidate,
        )
        if track_error:
            return self._failure(
                detection.class_id,
                output_frame,
                timestamp_ns,
                track_error,
            )
        pose_candidate = self._point_cloud_pose_candidate(
            detection.class_id,
            probe,
            head_pose,
            base,
            joints,
            legacy_point_transform,
        )
        orientation, pose_count = self._update_pose_refinement(
            track_key,
            pose_candidate,
            world_point,
        )
        converge = min(1.0, count / self.converge_frames)
        confidence *= (
            probe.valid_fraction
            * converge
            * self.heuristic_center_reliability
        )
        confidence = max(0.0, min(1.0, confidence))
        return ObjectEstimate3D(
            class_id=detection.class_id,
            position_xyz=filtered_point,
            confidence=confidence,
            frame_id=output_frame,
            timestamp_ns=timestamp_ns,
            valid=True,
            failure_reason=(
                "heuristic center approximation "
                "(surface-to-center not validated)"
                + (
                    "; point-cloud cuboid pose converged"
                    if orientation is not None
                    else (
                        f"; point-cloud pose pending ({pose_count}/"
                        f"{self.pose_required_frames})"
                        if self.pose_refinement_enabled
                        else ""
                    )
                )
            ),
            object_id=track_key,
            orientation_xyzw=orientation,
            size_xyz_m=self._local_sizes.get(detection.class_id),
        )

    def _point_cloud_pose_candidate(
        self,
        class_id: str,
        probe: _SurfaceProbe,
        head_pose: Optional[_HeadCameraPose],
        base: BaseState,
        joints: RobotJointState,
        legacy_point_transform: bool,
    ) -> Optional[tuple[float, float, float, float]]:
        """把当前框点云变到输出 frame，并生成单帧长方体姿态候选。"""

        size = self._local_sizes.get(class_id)
        if not self.pose_refinement_enabled or size is None or probe.camera_points_xyz is None:
            return None
        try:
            import numpy as np
        except ImportError:
            return None
        try:
            camera_points = np.asarray(probe.camera_points_xyz, dtype=float)
            if legacy_point_transform:
                output_points = np.asarray(
                    [
                        self.transform_provider.camera_to_output(
                            tuple(float(value) for value in point), base, joints
                        )
                        for point in camera_points
                    ],
                    dtype=float,
                )
            else:
                if head_pose is None:
                    return None
                qw, qx, qy, qz = _finite_vector(
                    head_pose.quaternion_wxyz,
                    4,
                    "MMK2FK camera_quaternion_wxyz",
                )
                norm = math.hypot(qw, qx, qy, qz)
                if norm < 1e-12:
                    return None
                qw, qx, qy, qz = (
                    qw / norm,
                    qx / norm,
                    qy / norm,
                    qz / norm,
                )
                rotation = np.asarray(
                    (
                        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
                        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
                        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
                    ),
                    dtype=float,
                )
                output_points = camera_points @ rotation.T + np.asarray(
                    head_pose.position, dtype=float
                )
            return _fit_cuboid_orientation_xyzw(
                output_points,
                size,
                self.pose_max_extent_error_ratio,
                np,
            )
        except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError):
            return None

    def _update_pose_refinement(
        self,
        track_key: Optional[str],
        candidate_xyzw: Optional[tuple[float, float, float, float]],
        position_xyz: tuple[float, float, float],
    ) -> tuple[Optional[tuple[float, float, float, float]], int]:
        """同一稳定ID连续收敛后才发布姿态；断证据或跳变立即重新累计。"""

        if not self.pose_refinement_enabled or track_key is None:
            return None, 0
        track = self._tracks.get(track_key)
        if track is None:
            return None, 0
        if candidate_xyzw is None:
            track.pose_quaternion_xyzw = None
            track.pose_count = 0
            track.last_pose_position_xyz = None
            return None, 0
        candidate = list(candidate_xyzw)
        previous = track.pose_quaternion_xyzw
        reset = previous is None or track.last_pose_position_xyz is None
        if not reset:
            dot = sum(previous[index] * candidate[index] for index in range(4))
            if dot < 0.0:
                candidate = [-value for value in candidate]
                dot = -dot
            angular_delta = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
            position_delta = math.dist(track.last_pose_position_xyz, position_xyz)
            reset = (
                angular_delta > self.pose_max_angular_delta_rad
                or position_delta > self.pose_max_position_delta_m
            )
        if reset:
            track.pose_quaternion_xyzw = candidate
            track.pose_count = 1
        else:
            assert previous is not None
            blended = [
                self.ema_alpha * candidate[index]
                + (1.0 - self.ema_alpha) * previous[index]
                for index in range(4)
            ]
            norm = math.sqrt(sum(value * value for value in blended))
            if norm < 1e-12:
                track.pose_quaternion_xyzw = candidate
                track.pose_count = 1
            else:
                track.pose_quaternion_xyzw = [value / norm for value in blended]
                track.pose_count += 1
        track.last_pose_position_xyz = position_xyz
        if track.pose_count < self.pose_required_frames:
            return None, track.pose_count
        assert track.pose_quaternion_xyzw is not None
        return tuple(track.pose_quaternion_xyzw), track.pose_count

    def _surface_probe(
        self,
        depth: DepthFrame,
        detection: Detection2D,
        intrinsics: CameraIntrinsics,
        request: _DepthWindowRequest,
        depth_array: Any,
        np: Any,
    ) -> _SurfaceProbe:
        """对一条检测只统计一次深度窗口，并复用其中位数与有效比例。"""

        try:
            statistics = _depth_window_statistics(
                depth,
                request,
                depth_array,
                np,
            )
            x0, y0, x1, y1 = _finite_vector(
                detection.bbox_xyxy, 4, "bbox_xyxy"
            )
            u = (x0 + x1) / 2.0
            v = (y0 + y1) / 2.0
        except ValueError as exc:
            raise _SurfaceProbeError(f"深度提取失败：{exc}") from exc
        try:
            camera_surface = project_pixel_to_camera(
                u,
                v,
                statistics.depth_m,
                intrinsics,
            )
        except ValueError as exc:
            raise _SurfaceProbeError(f"反投影失败：{exc}") from exc
        camera_points = None
        if (
            self.pose_refinement_enabled
            and detection.class_id in self._local_sizes
        ):
            try:
                camera_points = self._bbox_point_cloud_camera(
                    detection,
                    intrinsics,
                    request,
                    depth_array,
                    statistics.depth_m,
                    np,
                )
            except ValueError:
                # 点云姿态是可选增强；失败不得让仍有依据的中心估计一起失效。
                camera_points = None
        return _SurfaceProbe(
            u,
            v,
            statistics.depth_m,
            statistics.valid_fraction,
            camera_surface,
            camera_points,
        )

    def _bbox_point_cloud_camera(
        self,
        detection: Detection2D,
        intrinsics: CameraIntrinsics,
        request: _DepthWindowRequest,
        depth_array: Any,
        surface_depth_m: float,
        np: Any,
    ) -> Any:
        """提取框内靠近可见表面的有限深度点并批量反投影。"""

        if depth_array.ndim != 2:
            raise ValueError("深度图必须是严格二维数组")
        height, width = depth_array.shape
        x0, y0, x1, y1 = _finite_vector(
            detection.bbox_xyxy, 4, "Detection2D.bbox_xyxy"
        )
        xa, xb = max(0, math.ceil(x0)), min(width, math.ceil(x1))
        ya, yb = max(0, math.ceil(y0)), min(height, math.ceil(y1))
        if xa >= xb or ya >= yb:
            raise ValueError("bbox 与深度图没有有效交集")
        try:
            raw = depth_array[ya:yb, xa:xb].astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("bbox 深度无法转换为数值") from exc
        depth_m = raw * request.unit_scale_m
        valid = (
            np.isfinite(depth_m)
            & (depth_m > 0.0)
            & (np.abs(depth_m - surface_depth_m) <= self.pose_depth_band_m)
        )
        rows, columns = np.nonzero(valid)
        if rows.size < self.pose_min_points:
            raise ValueError(
                f"点云有效点不足：{rows.size} < {self.pose_min_points}"
            )
        # 对大框做确定性均匀抽样，限制每目标每帧的 PCA 成本。
        if rows.size > 2048:
            selected = np.linspace(0, rows.size - 1, 2048, dtype=int)
            rows, columns = rows[selected], columns[selected]
        z = depth_m[rows, columns]
        fx = _finite_number(intrinsics.k[0], "相机焦距 fx")
        fy = _finite_number(intrinsics.k[4], "相机焦距 fy")
        cx = _finite_number(intrinsics.k[2], "相机主点 cx")
        cy = _finite_number(intrinsics.k[5], "相机主点 cy")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("相机焦距必须为正")
        u = columns.astype(float) + xa
        v = rows.astype(float) + ya
        points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
        if not bool(np.all(np.isfinite(points))):
            raise ValueError("反投影点云包含NaN或Inf")
        return points

    def _compensate_to_center(
        self,
        class_id: str,
        camera_point: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], bool]:
        """沿相机光学射线把可见表面点近似补偿到物体几何中心。

        假设物体深度轴近似与相机 Z 轴对齐；已知物体沿视线深度时，把 Z 后移半深，
        同时等比例放大 X/Y 以保持点位于同一光学射线上。未知尺寸或非正 Z 不补偿。
        此处维度不随随机yaw交换，也不输出为 ``ObjectEstimate3D.size_xyz_m``。
        """

        x, y, z = camera_point
        dimensions = self._dims.get(class_id)
        if dimensions is None or z <= 0.0:
            return camera_point, False
        half_depth = dimensions[2] / 2.0
        scale = (z + half_depth) / z
        return (x * scale, y * scale, z + half_depth), True

    def _associate_tracks(
        self,
        detections: tuple[Detection2D, ...],
        context_errors: list[str],
    ) -> tuple[dict[int, Optional[str]], dict[int, str]]:
        """按稳定ID做同帧一对一关联；无ID输入明确不建立持久轨迹。"""

        associations: dict[int, Optional[str]] = {}
        errors: dict[int, str] = {}
        reserved_indices: dict[str, int] = {}

        for index, detection in enumerate(detections):
            if context_errors[index]:
                continue
            if not detection.valid:
                associations[index] = None
                continue
            if detection.track_id is None:
                associations[index] = None
                continue
            key = f"stable:{detection.track_id}"
            if key in reserved_indices:
                previous_index = reserved_indices[key]
                reason = (
                    "二维稳定轨迹ID重复：同一帧中 "
                    f"track_id={detection.track_id} "
                    "只能关联一个目标"
                )
                errors[previous_index] = reason
                errors[index] = reason
                associations.pop(previous_index, None)
                continue
            associations[index] = key
            reserved_indices[key] = index
        return associations, errors

    def _identity_consistency(
        self,
        detections: tuple[Detection2D, ...],
        associations: dict[int, Optional[str]],
        context_errors: list[str],
        association_errors: dict[int, str],
        probes: dict[int, _SurfaceProbe],
        head_pose: Optional[_HeadCameraPose],
        base: BaseState,
        joints: RobotJointState,
        legacy_point_transform: bool,
    ) -> tuple[dict[int, _IdentityCandidate], dict[int, str]]:
        """在更新EMA前用未补偿三维表面点核验稳定ID，疑似交换时关闭失败。"""

        candidates: dict[int, _IdentityCandidate] = {}
        errors: dict[int, str] = {}
        if head_pose is None and not legacy_point_transform:
            return candidates, errors
        for index, detection in enumerate(detections):
            key = associations.get(index)
            if (
                key is None
                or context_errors[index]
                or index in association_errors
                or not detection.valid
                or index not in probes
            ):
                continue
            probe = probes[index]
            try:
                if legacy_point_transform:
                    transformed_surface = (
                        self.transform_provider.camera_to_output(
                            probe.camera_surface_xyz,
                            base,
                            joints,
                        )
                    )
                else:
                    assert head_pose is not None
                    transformed_surface = (
                        self.transform_provider.transform_camera_point(
                            probe.camera_surface_xyz, head_pose
                        )
                    )
                surface_xyz = _finite_vector(
                    transformed_surface,
                    3,
                    "camera_to_output 返回值",
                )
            except (ValueError, RuntimeError):
                # 正式估计路径会保留完整的深度、反投影或坐标变换失败原因。
                continue
            candidates[index] = _IdentityCandidate(
                (probe.u, probe.v),
                surface_xyz,
            )

        # 同类、同帧且候选在1像素/1厘米内近乎重合时没有足够证据区分身份，
        # 整对关闭失败。该阈值只用于“不可分辨”保护，不用于普通目标关联。
        candidate_items = list(candidates.items())
        for item_index, (left_index, left) in enumerate(candidate_items):
            for right_index, right in candidate_items[item_index + 1 :]:
                if detections[left_index].class_id != detections[right_index].class_id:
                    continue
                if (
                    math.dist(left.surface_xyz, right.surface_xyz) <= 0.01
                    and math.dist(
                        left.bbox_center_xy, right.bbox_center_xy
                    )
                    <= 1.0
                ):
                    reason = (
                        "二维轨迹ID一致性校验失败：同类目标完全重叠，"
                        "无法可靠区分身份"
                    )
                    errors[left_index] = reason
                    errors[right_index] = reason

        # 当前候选若明显更接近另一条同类历史轨迹，则上游ID很可能已贴反。
        for index, candidate in candidates.items():
            if index in errors:
                continue
            key = associations[index]
            if key is None:
                continue
            assigned = self._tracks.get(key)
            if assigned is None or assigned.class_id != detections[index].class_id:
                continue
            d_assigned = math.dist(candidate.surface_xyz, assigned.last_surface_xyz)
            other_distances = [
                math.dist(candidate.surface_xyz, track.last_surface_xyz)
                for other_key, track in self._tracks.items()
                if other_key != key and track.class_id == assigned.class_id
            ]
            if not other_distances:
                continue
            d_other_min = min(other_distances)
            if d_assigned > d_other_min * self.ambiguity_ratio:
                errors[index] = (
                    "二维轨迹ID一致性校验失败，疑似ID交换："
                    f"分配轨迹距离={d_assigned:.6f}m，"
                    f"其他同类轨迹最近距离={d_other_min:.6f}m，"
                    f"比率阈值={self.ambiguity_ratio:.6f}"
                )
        return candidates, errors

    def _update_track(
        self,
        key: Optional[str],
        detection: Detection2D,
        world_point: tuple[float, float, float],
        timestamp_ns: int,
        identity_candidate: Optional[_IdentityCandidate],
    ) -> tuple[tuple[float, float, float], int, str]:
        """以严格递增时间戳更新三维 EMA，并拒绝离群大跳变。"""

        if key is None:
            return world_point, 1, ""
        if identity_candidate is None:
            return (
                (0.0, 0.0, 0.0),
                0,
                "二维轨迹ID一致性校验失败：缺少有效三维身份候选",
            )

        detection_timestamp_ns = int(detection.timestamp_ns)
        track = self._tracks.get(key)
        if track is None:
            track = _Track(
                class_id=detection.class_id,
                ema=list(world_point),
                count=1,
                last_ts_ns=timestamp_ns,
                last_detection_ts_ns=detection_timestamp_ns,
                bbox_center_xy=identity_candidate.bbox_center_xy,
                last_surface_xyz=identity_candidate.surface_xyz,
            )
            self._tracks[key] = track
            return tuple(track.ema), track.count, ""

        if track.class_id != detection.class_id:
            return (
                (0.0, 0.0, 0.0),
                track.count,
                (
                    "二维稳定轨迹ID类别冲突："
                    f"历史类别 {track.class_id!r}，当前类别 {detection.class_id!r}"
                ),
            )
        if timestamp_ns <= track.last_ts_ns:
            return (
                (0.0, 0.0, 0.0),
                track.count,
                (
                    "陈旧轨迹样本：当前时间戳 "
                    f"{timestamp_ns} 未严格晚于轨迹时间 {track.last_ts_ns}"
                ),
            )
        if detection_timestamp_ns <= track.last_detection_ts_ns:
            return (
                (0.0, 0.0, 0.0),
                track.count,
                (
                    "陈旧二维轨迹样本：当前Detection时间戳 "
                    f"{detection_timestamp_ns} 未严格晚于轨迹Detection时间 "
                    f"{track.last_detection_ts_ns}"
                ),
            )
        jump_m = math.dist(tuple(track.ema), world_point)
        if jump_m > self.max_position_jump_m:
            return (
                (0.0, 0.0, 0.0),
                track.count,
                (
                    "三维位置跳变超限："
                    f"{jump_m:.6f}m > {self.max_position_jump_m:.6f}m，"
                    "拒绝离群值且不更新EMA"
                ),
            )
        for index, value in enumerate(world_point):
            track.ema[index] = (
                self.ema_alpha * value
                + (1.0 - self.ema_alpha) * track.ema[index]
            )
        track.count += 1
        track.last_ts_ns = timestamp_ns
        track.last_detection_ts_ns = detection_timestamp_ns
        track.bbox_center_xy = identity_candidate.bbox_center_xy
        track.last_surface_xyz = identity_candidate.surface_xyz
        return tuple(track.ema), track.count, ""

    def _remove_expired_tracks(self, timestamp_ns: int) -> None:
        """删除相对当前正序帧已超时的轨迹。"""

        max_age_ns = self.max_track_age_s * 1_000_000_000.0
        expired = [
            key
            for key, track in self._tracks.items()
            if timestamp_ns > track.last_ts_ns
            and timestamp_ns - track.last_ts_ns > max_age_ns
        ]
        for key in expired:
            del self._tracks[key]

    def _validate_batch_inputs(
        self,
        depth: DepthFrame,
        intrinsics: CameraIntrinsics,
    ) -> tuple[int, int]:
        """校验Depth/CameraInfo有效性、非负时间和严格frame一致性。"""

        if not depth.valid:
            raise ValueError(
                f"DepthFrame无效：{depth.failure_reason or '上游未提供原因'}"
            )
        if not intrinsics.valid:
            raise ValueError(
                f"CameraInfo无效：{intrinsics.failure_reason or '上游未提供原因'}"
            )
        depth_timestamp_ns = self._timestamp_ns(
            depth.timestamp_ns, "DepthFrame.timestamp_ns"
        )
        intrinsics_timestamp_ns = self._timestamp_ns(
            intrinsics.timestamp_ns, "CameraIntrinsics.timestamp_ns"
        )
        depth_frame = self._frame_id(depth.frame_id, "DepthFrame.frame_id")
        intrinsics_frame = self._frame_id(
            intrinsics.frame_id, "CameraIntrinsics.frame_id"
        )
        if depth_frame != intrinsics_frame:
            raise ValueError(
                "DepthFrame/CameraInfo frame不一致："
                f"{depth_frame!r} != {intrinsics_frame!r}"
            )
        self._require_time_window(
            depth_timestamp_ns,
            intrinsics_timestamp_ns,
            "DepthFrame/CameraInfo",
        )
        return depth_timestamp_ns, intrinsics_timestamp_ns

    def _detection_context_error(
        self,
        detection: Detection2D,
        expected_frame_id: str,
        depth_timestamp_ns: int,
        intrinsics_timestamp_ns: int,
    ) -> str:
        """返回单条Detection与当前Depth上下文不一致的原因。"""

        if not detection.valid:
            return ""
        try:
            detection_timestamp_ns = self._timestamp_ns(
                detection.timestamp_ns, "Detection2D.timestamp_ns"
            )
            detection_frame = self._frame_id(
                detection.frame_id, "Detection2D.frame_id"
            )
            if detection_frame != expected_frame_id:
                raise ValueError(
                    "Detection2D/DepthFrame frame不一致："
                    f"{detection_frame!r} != {expected_frame_id!r}"
                )
            self._require_time_window(
                detection_timestamp_ns,
                depth_timestamp_ns,
                "Detection2D/DepthFrame",
            )
            self._require_time_window(
                detection_timestamp_ns,
                intrinsics_timestamp_ns,
                "Detection2D/CameraInfo",
            )
            if detection.track_id is not None and (
                isinstance(detection.track_id, bool)
                or not isinstance(detection.track_id, Integral)
                or int(detection.track_id) < 0
            ):
                raise ValueError("Detection2D.track_id 必须是非负整数或None")
            x0, y0, x1, y1 = _finite_vector(
                detection.bbox_xyxy, 4, "Detection2D.bbox_xyxy"
            )
            if x1 <= x0 or y1 <= y0:
                raise ValueError(
                    "Detection2D.bbox_xyxy 必须满足 x1>x0 且 y1>y0"
                )
        except ValueError as exc:
            return f"二维检测上下文无效：{exc}"
        return ""

    def _require_time_window(
        self,
        first_timestamp_ns: int,
        second_timestamp_ns: int,
        label: str,
    ) -> None:
        max_skew_ns = self.max_input_skew_s * 1_000_000_000.0
        skew_ns = abs(first_timestamp_ns - second_timestamp_ns)
        if skew_ns > max_skew_ns:
            raise ValueError(
                f"{label} 时间差 {skew_ns}ns 超过允许窗口 "
                f"{int(max_skew_ns)}ns"
            )

    @staticmethod
    def _frame_id(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 必须是非空字符串")
        return value

    @staticmethod
    def _timestamp_ns(value: Any, name: str) -> int:
        """校验非负整数纳秒时间戳，避免bool或负值进入时序。"""

        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
        ):
            raise ValueError(f"{name} 必须是非负整数纳秒")
        return int(value)

    @staticmethod
    def _safe_failure_timestamp(value: Any) -> int:
        if (
            isinstance(value, Integral)
            and not isinstance(value, bool)
            and int(value) >= 0
        ):
            return int(value)
        return 0

    @staticmethod
    def _failure(
        class_id: str,
        output_frame: str,
        timestamp_ns: int,
        reason: str,
    ) -> ObjectEstimate3D:
        """构造不可用的诊断结果；零坐标仅作为无效值占位，绝不标记成功。"""

        return ObjectEstimate3D(
            class_id=class_id,
            position_xyz=(0.0, 0.0, 0.0),
            confidence=0.0,
            frame_id=output_frame,
            timestamp_ns=timestamp_ns,
            valid=False,
            failure_reason=reason,
            object_id=None,
            orientation_xyzw=None,
            size_xyz_m=None,
        )


class VisualObservationVerifier:
    """用新鲜的稳定三维观测完成试抬和放置后的视觉复检。

    本类只判断物体观测，不读取夹爪 effort、不推进 FSM，也不把视觉判断提升为
    官方控制器执行证明。阈值均由组装层显式注入：距离单位为米，时间单位为秒。
    当前只允许 ``odom`` 观测，因为本算法把该 frame 的 Z 轴解释为竖直方向。
    """

    def __init__(
        self,
        *,
        minimum_lift_delta_m: float,
        max_horizontal_drift_m: float,
        max_observation_gap_s: float,
        minimum_observation_confidence: float,
        required_frame_id: str,
        min_stationary_observations: int,
        max_stationary_spread_m: float,
    ) -> None:
        self.minimum_lift_delta_m = _finite_number(
            minimum_lift_delta_m, "minimum_lift_delta_m"
        )
        self.max_horizontal_drift_m = _finite_number(
            max_horizontal_drift_m, "max_horizontal_drift_m"
        )
        self.max_observation_gap_s = _finite_number(
            max_observation_gap_s, "max_observation_gap_s"
        )
        self.minimum_observation_confidence = _finite_number(
            minimum_observation_confidence, "minimum_observation_confidence"
        )
        if not isinstance(required_frame_id, str) or not required_frame_id.strip():
            raise ValueError("required_frame_id 必须是非空字符串")
        self.required_frame_id = required_frame_id.strip()
        self.max_stationary_spread_m = _finite_number(
            max_stationary_spread_m, "max_stationary_spread_m"
        )
        if self.minimum_lift_delta_m <= 0.0:
            raise ValueError("minimum_lift_delta_m 必须大于0")
        if self.max_horizontal_drift_m < 0.0:
            raise ValueError("max_horizontal_drift_m 不得小于0")
        if self.max_observation_gap_s <= 0.0:
            raise ValueError("max_observation_gap_s 必须大于0")
        if not 0.0 < self.minimum_observation_confidence <= 1.0:
            raise ValueError("minimum_observation_confidence 必须位于 (0,1]")
        if self.required_frame_id != "odom":
            raise ValueError("当前阶段required_frame_id必须严格为odom")
        if self.max_stationary_spread_m < 0.0:
            raise ValueError("max_stationary_spread_m 不得小于0")
        if (
            isinstance(min_stationary_observations, bool)
            or not isinstance(min_stationary_observations, Integral)
            or int(min_stationary_observations) < 2
        ):
            raise ValueError("min_stationary_observations 必须是至少为2的整数")
        self.min_stationary_observations = int(min_stationary_observations)

    def verify_test_lift(
        self,
        before: ObjectEstimate3D,
        after: ObjectEstimate3D,
        *,
        timestamp_ns: int,
    ) -> GraspVerification:
        """比较试抬前后同一稳定目标；验证完成与抓取成立是两个独立结论。"""

        now_ns = Perception3DEstimator._timestamp_ns(timestamp_ns, "timestamp_ns")
        error = self._pair_error(before, after, now_ns)
        if error:
            return self._grasp_failure(error, now_ns)

        dx = after.position_xyz[0] - before.position_xyz[0]
        dy = after.position_xyz[1] - before.position_xyz[1]
        dz = after.position_xyz[2] - before.position_xyz[2]
        horizontal_drift = math.hypot(dx, dy)
        is_grasped = (
            dz >= self.minimum_lift_delta_m
            and horizontal_drift <= self.max_horizontal_drift_m
        )
        evidence = (
            f"同一目标 {after.object_id!r}：竖直位移={dz:.4f}m，"
            f"水平漂移={horizontal_drift:.4f}m"
        )
        return GraspVerification(
            is_grasped=is_grasped,
            confidence=min(float(before.confidence), float(after.confidence)),
            visual_evidence=evidence,
            effort_evidence="未提供；当前结果仅包含视觉证据",
            success=True,
            failure_reason="",
            timestamp_ns=now_ns,
        )

    def stable_post_motion_observation(
        self,
        observations: tuple[ObjectEstimate3D, ...],
        *,
        timestamp_ns: int,
    ) -> ObjectEstimate3D:
        """确认运动后同一物体连续多帧收敛，并返回最后一帧观测事实。

        该结果不证明位置正确或夹爪已释放，不得单独触发 ``PLACE_VERIFIED``。
        """

        now_ns = Perception3DEstimator._timestamp_ns(timestamp_ns, "timestamp_ns")
        try:
            items = tuple(observations)
        except TypeError:
            items = ()
        if len(items) < self.min_stationary_observations:
            return self._post_motion_failure(
                items,
                now_ns,
                f"稳定观测不足：需要至少{self.min_stationary_observations}帧",
            )
        first = items[0]
        for index, item in enumerate(items):
            error = self._observation_error(item, f"observations[{index}]")
            if error:
                return self._post_motion_failure(items, now_ns, error)
            if (
                item.object_id != first.object_id
                or item.class_id != first.class_id
                or item.frame_id != first.frame_id
            ):
                return self._post_motion_failure(
                    items, now_ns, "放置后观测的目标身份、类别或frame不一致"
                )
            if index and item.timestamp_ns <= items[index - 1].timestamp_ns:
                return self._post_motion_failure(
                    items, now_ns, "放置后观测时间戳必须严格递增"
                )
        if self._outside_time_window(first.timestamp_ns, items[-1].timestamp_ns):
            return self._post_motion_failure(items, now_ns, "放置后观测序列时间跨度过大")
        if items[-1].timestamp_ns > now_ns:
            return self._post_motion_failure(items, now_ns, "最新观测时间晚于验证时间")
        if self._outside_time_window(items[-1].timestamp_ns, now_ns):
            return self._post_motion_failure(items, now_ns, "最新放置后观测已经过期")

        spread = max(
            math.dist(items[left].position_xyz, items[right].position_xyz)
            for left in range(len(items))
            for right in range(left + 1, len(items))
        )
        if spread > self.max_stationary_spread_m:
            return self._post_motion_failure(
                items,
                now_ns,
                f"放置后目标尚未稳定：最大两两距离={spread:.4f}m",
            )
        return items[-1]

    def _pair_error(
        self,
        before: ObjectEstimate3D,
        after: ObjectEstimate3D,
        now_ns: int,
    ) -> str:
        for name, item in (("试抬前观测", before), ("试抬后观测", after)):
            error = self._observation_error(item, name)
            if error:
                return error
        if (
            before.object_id != after.object_id
            or before.class_id != after.class_id
            or before.frame_id != after.frame_id
        ):
            return "试抬前后目标身份、类别或frame不一致"
        if after.timestamp_ns <= before.timestamp_ns:
            return "试抬后观测时间戳必须晚于试抬前观测"
        if self._outside_time_window(before.timestamp_ns, after.timestamp_ns):
            return "试抬前后观测时间间隔过大"
        if after.timestamp_ns > now_ns:
            return "试抬后观测时间晚于验证时间"
        if self._outside_time_window(after.timestamp_ns, now_ns):
            return "试抬后观测已经过期"
        return ""

    def _observation_error(self, item: Any, name: str) -> str:
        if not isinstance(item, ObjectEstimate3D):
            return f"{name} 必须是ObjectEstimate3D"
        if not item.valid:
            return f"{name}无效：{item.failure_reason or '上游未提供原因'}"
        if item.confidence < self.minimum_observation_confidence:
            return (
                f"{name}置信度{item.confidence:.4f}低于最低要求"
                f"{self.minimum_observation_confidence:.4f}"
            )
        if item.frame_id != self.required_frame_id:
            return (
                f"{name}frame必须为{self.required_frame_id}，"
                f"实际为{item.frame_id}"
            )
        if not isinstance(item.object_id, str) or not item.object_id.strip():
            return f"{name}缺少稳定object_id"
        try:
            _finite_vector(item.position_xyz, 3, f"{name}.position_xyz")
            Perception3DEstimator._timestamp_ns(
                item.timestamp_ns, f"{name}.timestamp_ns"
            )
        except ValueError as exc:
            return str(exc)
        return ""

    def _outside_time_window(self, earlier_ns: int, later_ns: int) -> bool:
        return later_ns - earlier_ns > self.max_observation_gap_s * 1_000_000_000.0

    @staticmethod
    def _grasp_failure(reason: str, timestamp_ns: int) -> GraspVerification:
        return GraspVerification(
            is_grasped=False,
            confidence=0.0,
            visual_evidence="视觉证据不足",
            effort_evidence="未提供；当前结果仅包含视觉证据",
            success=False,
            failure_reason=reason,
            timestamp_ns=timestamp_ns,
        )

    @staticmethod
    def _post_motion_failure(
        observations: tuple[ObjectEstimate3D, ...],
        timestamp_ns: int,
        reason: str,
    ) -> ObjectEstimate3D:
        valid_items = tuple(
            item for item in observations if isinstance(item, ObjectEstimate3D)
        )
        latest = valid_items[-1] if valid_items else None
        return Perception3DEstimator._failure(
            latest.class_id if latest is not None else "",
            latest.frame_id if latest is not None else "unknown",
            timestamp_ns,
            reason,
        )


def _rotate_by_wxyz(
    vector_xyz: tuple[float, float, float], quaternion_wxyz: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    # 使用 q*v*q^-1 展开式完成光学系到输出 frame 的旋转，避免强制依赖 SciPy。
    vx, vy, vz = _finite_vector(vector_xyz, 3, "待旋转相机点")
    w, x, y, z = _finite_vector(quaternion_wxyz, 4, "相机四元数 wxyz")
    norm = math.hypot(w, x, y, z)
    if norm < 1e-12:
        raise ValueError("相机四元数范数为零")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
