"""深度反投影、相机外参和三维目标估计接口。

本文件负责复用官方 ``box_detect.py`` 的针孔反投影思路，并通过官方 ``MMK2FK``
提供 camera 到 world/odom 的变换接口；不负责 YOLO、ROS2 同步、目标选择或抓取规划。
``perception_node`` 在获得同步 RGB/Depth 及最近 Odom/JointState 后调用本模块。输入为
``Detection2D``、深度、内参和机器人实际状态，输出为 ``ObjectEstimate3D``。

``ObjectEstimate3D`` 表示物体中心的三维估计，不是左右夹爪末端位姿，也不是最终抓取
点或放置点。后续 ``arm_planning`` 必须结合任务、箱体尺寸、抓取方向和安全偏移，另行
计算左右夹爪目标。Odom 只给出底盘位姿；slide 和 head 关节会改变头部相机相对底盘的
位置与朝向，因此还必须使用实际 ``RobotJointState`` 和 ``MMK2FK`` 闭合坐标链。

MMK2FK、MuJoCo、SciPy 和 NumPy 均按需延迟导入。第一版没有实现可靠的表面点到物体
中心补偿，因此完整估计入口会明确报告未实现，不会把表面点伪装成物体中心。
"""

from __future__ import annotations

import importlib
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
    ObjectEstimate3D,
    RobotJointState,
)


def _finite_number(value: Any, name: str) -> float:
    """把真实数值收窄为有限浮点数，同时拒绝容易混入数值运算的 bool。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是真实数值，不能使用 bool 或其他类型")
    result = float(value)
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


def median_depth_m(
    depth: DepthFrame, bbox_xyxy: tuple[float, float, float, float], radius_px: int = 4
) -> float:
    """在二维框中心附近读取非零中位深度。

    参数：对齐深度图、像素 bbox 和非负窗口半径（像素）。返回米制中位深度。
    中位数比单像素更不易受深度孔洞和少量离群值影响，但它通常仍落在物体可见表面，
    不能直接声称是物体中心深度。失败：NumPy 缺失、二维图像/bbox/单位比例错误或
    窗口内无有效深度时抛出 ``ValueError``，不会返回虚构距离。
    """

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
    try:
        import numpy as np  # 延迟导入，避免纯接口模块强依赖视觉环境
    except ImportError as exc:
        raise ValueError("深度处理中缺少 NumPy") from exc
    try:
        image = np.asarray(depth.image)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth.image 无法转换为深度数组") from exc
    if image.ndim != 2:
        raise ValueError(f"深度图必须是严格二维数组，实际维度={image.ndim}")
    height, width = image.shape
    if height <= 0 or width <= 0:
        raise ValueError("深度图宽高必须大于0")
    if x1 <= 0.0 or y1 <= 0.0 or x0 >= width or y0 >= height:
        raise ValueError("bbox 完全位于深度图范围之外")

    center_x = int(round((x0 + x1) / 2.0))
    center_y = int(round((y0 + y1) / 2.0))
    xa, xb = max(0, center_x - radius), min(width, center_x + radius + 1)
    ya, yb = max(0, center_y - radius), min(height, center_y + radius + 1)
    if xa >= xb or ya >= yb:
        raise ValueError("bbox 中心无法在深度图内形成有效采样窗口")
    try:
        patch = image[ya:yb, xa:xb].astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bbox 中心窗口的深度值无法转换为数值") from exc
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise ValueError("bbox 中心窗口没有有效深度")
    result_m = float(np.median(valid)) * unit_scale_m
    if not math.isfinite(result_m) or result_m <= 0.0:
        raise ValueError("中位深度换算结果必须是正的有限米制数")
    return result_m


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

        if self._fk is None:
            raise RuntimeError("CameraTransformProvider 尚未通过 self_check")
        if not base.valid:
            raise ValueError(f"底盘状态无效，不能计算相机外参：{base.failure_reason}")
        if not joints.valid:
            raise ValueError(f"实际关节状态无效，不能计算相机外参：{joints.failure_reason}")
        camera_point = _finite_vector(camera_point_xyz, 3, "camera_point_xyz")
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


class Perception3DEstimator:
    """二维框到物体三维中心的估计器骨架。

    视觉2未来需要为每个 ``Detection2D`` 选择可靠代表像素，从对齐深度中提取有效
    距离，反投影到相机光学系，再经 ``CameraTransformProvider`` 转到输出 frame。
    深度测到的通常是可见表面，必须完成表面点到物体中心的补偿后，才能组合类别、
    三维中心、置信度、frame 和时间戳生成 ``ObjectEstimate3D``。

    失败检测需要明确跳过或输出 ``valid=False``，并设计三维多帧滤波。BBOX 中心小
    窗口中位数、缩小 ROI、深度离群过滤、点云/平面拟合、已知尺寸中心补偿及三维
    EMA/中值滤波都只是待视觉2评审的候选方案，当前没有实现。本类不生成左右夹爪
    末端位姿、最终抓取点或放置点。
    """

    def __init__(self, transform_provider: CameraTransformProvider) -> None:
        """保存已配置的相机外参提供器。

        参数：``transform_provider`` 负责 camera 到输出 frame 的官方 FK 转换。
        返回：无。失败：不在构造阶段加载外部依赖。
        """

        self.transform_provider = transform_provider

    def estimate(
        self,
        detections: tuple[Detection2D, ...],
        depth: DepthFrame,
        intrinsics: CameraIntrinsics,
        base: BaseState,
        joints: RobotJointState,
    ) -> tuple[ObjectEstimate3D, ...]:
        """把二维检测转换为物体中心三维估计。

        参数包含像素框、米制深度、内参、Odom 和实际关节状态；目标输出 frame 由外参
        提供器定义。返回三维目标元组。
        失败：可靠像素/深度选择、物体表面点到中心的补偿、失败策略和滤波尚未实现，
        当前明确抛出 ``NotImplementedError``。保留异常是为了阻止表面点被伪装成
        物体中心后继续流入导航或抓取规划。
        """

        raise NotImplementedError("三维物体中心补偿与多帧滤波尚未实现，请由视觉2负责人完成")


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
