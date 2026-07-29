"""深度反投影、相机外参和三维目标估计接口。

本文件负责复用官方 ``box_detect.py`` 的针孔反投影思路，并通过官方 ``MMK2FK``
提供 camera 到 world/odom 的变换接口；不负责 YOLO、ROS2 同步、目标选择或抓取规划。
``perception_node`` 在获得同步 RGB/Depth 及最近 Odom/JointState 后调用本模块。输入为
``Detection2D``、深度、内参和机器人实际状态，输出为 ``ObjectEstimate3D``。

``ObjectEstimate3D`` 表示物体中心的三维估计，不是左右夹爪末端位姿，也不是最终抓取
点或放置点。后续 ``arm_planning`` 必须结合任务、箱体尺寸、抓取方向和安全偏移，另行
计算左右夹爪目标。Odom 只给出底盘位姿；slide 和 head 关节会改变头部相机相对底盘的
位置与朝向，因此还必须使用实际 ``RobotJointState`` 和 ``MMK2FK`` 闭合坐标链。

MMK2FK、MuJoCo、SciPy 和 NumPy 均按需延迟导入。物体尺寸由调用方注入；已知尺寸时
沿相机射线把可见表面点补偿到近似几何中心，未知尺寸时明确返回无效估计，禁止把
表面点冒充物体中心继续下传。
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _Track:
    """单个物体在输出 frame 中的米制三维 EMA 状态。"""

    ema: list[float]
    count: int
    last_ts_ns: int


class Perception3DEstimator:
    """把二维框转换为输出 frame 中的物体三维中心估计。

    每条检测独立执行中心窗口中位深度、针孔反投影、已知尺寸中心补偿和官方相机外参
    变换。最终世界坐标按类别和二维网格分桶做 EMA，以抑制深度逐帧噪声。尺寸未知时
    无法声称表面点已经到达几何中心，因此返回带明确原因的无效估计。本类不负责二维
    框关联、目标选择、抓取点或夹爪末端位姿。
    """

    def __init__(
        self,
        transform_provider: CameraTransformProvider,
        *,
        depth_radius_px: int = 4,
        ema_alpha: float = 0.5,
        converge_frames: int = 5,
        track_grid_px: float = 40.0,
        max_track_age_s: float = 1.0,
        object_dimensions_m: Optional[
            dict[str, tuple[float, float, float]]
        ] = None,
    ) -> None:
        """保存三维估计参数，构造阶段不加载任何官方依赖。

        ``object_dimensions_m`` 的值依次为宽、高、沿相机视线近似深度，单位米。
        ``track_grid_px`` 为二维轨迹分桶尺寸（像素），``max_track_age_s`` 为轨迹超时
        秒数。失败：参数类型、范围或物体尺寸不合法时抛出 ``ValueError``。
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

        alpha = _finite_number(ema_alpha, "ema_alpha")
        grid_px = _finite_number(track_grid_px, "track_grid_px")
        max_age_s = _finite_number(max_track_age_s, "max_track_age_s")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("ema_alpha 必须位于 (0, 1] 范围")
        if grid_px <= 0.0:
            raise ValueError("track_grid_px 必须是正的有限像素数")
        if max_age_s <= 0.0:
            raise ValueError("max_track_age_s 必须是正的有限秒数")

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

        self.transform_provider = transform_provider
        self.depth_radius_px = int(depth_radius_px)
        self.ema_alpha = alpha
        self.converge_frames = int(converge_frames)
        self.track_grid_px = grid_px
        self.max_track_age_s = max_age_s
        self._dims = normalized_dimensions
        self._tracks: dict[str, _Track] = {}

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
        frame 由 ``transform_provider.output_frame`` 指定，时间戳使用深度帧时间。
        单条检测失败时只把对应结果标为 ``valid=False``，不会中断同批其他检测。
        """

        timestamp_ns = self._timestamp_ns(depth.timestamp_ns)
        self._remove_expired_tracks(timestamp_ns)
        output_frame = self.transform_provider.output_frame
        return tuple(
            self._estimate_one(
                detection,
                depth,
                intrinsics,
                base,
                joints,
                output_frame,
                timestamp_ns,
            )
            for detection in detections
        )

    def reset_tracks(self) -> None:
        """清空全部三维 EMA 轨迹，用于切换场景或确认长时间失跟后重置。"""

        self._tracks.clear()

    def _estimate_one(
        self,
        detection: Detection2D,
        depth: DepthFrame,
        intrinsics: CameraIntrinsics,
        base: BaseState,
        joints: RobotJointState,
        output_frame: str,
        timestamp_ns: int,
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

        try:
            depth_m = median_depth_m(
                depth,
                detection.bbox_xyxy,
                radius_px=self.depth_radius_px,
            )
            x0, y0, x1, y1 = _finite_vector(
                detection.bbox_xyxy, 4, "bbox_xyxy"
            )
            u = (x0 + x1) / 2.0
            v = (y0 + y1) / 2.0
        except ValueError as exc:
            return self._failure(
                detection.class_id,
                output_frame,
                timestamp_ns,
                f"深度提取失败：{exc}",
            )

        try:
            camera_point = project_pixel_to_camera(u, v, depth_m, intrinsics)
        except ValueError as exc:
            return self._failure(
                detection.class_id,
                output_frame,
                timestamp_ns,
                f"反投影失败：{exc}",
            )

        compensated_point, compensated = self._compensate_to_center(
            detection.class_id, camera_point
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
        try:
            world_point = _finite_vector(
                self.transform_provider.camera_to_output(
                    compensated_point, base, joints
                ),
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
        valid_fraction = self._depth_valid_fraction(depth, u, v)
        key = self._track_key(detection.class_id, u, v)
        filtered_point, count = self._update_track(
            key, world_point, timestamp_ns
        )
        converge = min(1.0, count / self.converge_frames)
        confidence *= valid_fraction * converge
        confidence = max(0.0, min(1.0, confidence))
        return ObjectEstimate3D(
            detection.class_id,
            filtered_point,
            confidence,
            output_frame,
            timestamp_ns,
            valid=True,
        )

    def _compensate_to_center(
        self,
        class_id: str,
        camera_point: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], bool]:
        """沿相机光学射线把可见表面点近似补偿到物体几何中心。

        假设物体深度轴近似与相机 Z 轴对齐；已知物体沿视线深度时，把 Z 后移半深，
        同时等比例放大 X/Y 以保持点位于同一光学射线上。未知尺寸或非正 Z 不补偿。
        """

        x, y, z = camera_point
        dimensions = self._dims.get(class_id)
        if dimensions is None or z <= 0.0:
            return camera_point, False
        half_depth = dimensions[2] / 2.0
        scale = (z + half_depth) / z
        return (x * scale, y * scale, z + half_depth), True

    def _track_key(self, class_id: str, u: float, v: float) -> str:
        """按类别和二维中心网格生成三维轨迹键，不执行二维框关联或平滑。"""

        bucket_x = round(u / self.track_grid_px)
        bucket_y = round(v / self.track_grid_px)
        return f"{class_id}:{bucket_x}:{bucket_y}"

    def _update_track(
        self,
        key: str,
        world_point: tuple[float, float, float],
        timestamp_ns: int,
    ) -> tuple[tuple[float, float, float], int]:
        """以严格递增时间戳更新输出 frame 中的三维 EMA。"""

        track = self._tracks.get(key)
        max_age_ns = self.max_track_age_s * 1_000_000_000.0
        if (
            track is None
            or timestamp_ns - track.last_ts_ns > max_age_ns
        ):
            track = _Track(list(world_point), 1, timestamp_ns)
            self._tracks[key] = track
        elif timestamp_ns > track.last_ts_ns:
            for index, value in enumerate(world_point):
                track.ema[index] = (
                    self.ema_alpha * value
                    + (1.0 - self.ema_alpha) * track.ema[index]
                )
            track.count += 1
            track.last_ts_ns = timestamp_ns
        # 重复或倒序帧不得写入 EMA，也不得删除已有轨迹。
        return tuple(track.ema), track.count

    def _remove_expired_tracks(self, timestamp_ns: int) -> None:
        """删除相对当前正序帧已超时的轨迹；倒序帧不会误删状态。"""

        max_age_ns = self.max_track_age_s * 1_000_000_000.0
        expired = [
            key
            for key, track in self._tracks.items()
            if timestamp_ns > track.last_ts_ns
            and timestamp_ns - track.last_ts_ns > max_age_ns
        ]
        for key in expired:
            del self._tracks[key]

    def _depth_valid_fraction(
        self, depth: DepthFrame, u: float, v: float
    ) -> float:
        """计算深度中心窗口内有限且为正的原始像素占比。"""

        try:
            import numpy as np
        except ImportError:
            return 1.0
        image = np.asarray(depth.image)
        height, width = image.shape
        center_x = int(round(u))
        center_y = int(round(v))
        xa = max(0, center_x - self.depth_radius_px)
        xb = min(width, center_x + self.depth_radius_px + 1)
        ya = max(0, center_y - self.depth_radius_px)
        yb = min(height, center_y + self.depth_radius_px + 1)
        patch = image[ya:yb, xa:xb].astype(float)
        valid = np.isfinite(patch) & (patch > 0.0)
        return float(np.count_nonzero(valid) / patch.size)

    @staticmethod
    def _timestamp_ns(value: Any) -> int:
        """校验深度帧纳秒时间戳，避免 bool 或非整数参与轨迹时序。"""

        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("DepthFrame.timestamp_ns 必须是整数纳秒")
        return int(value)

    @staticmethod
    def _failure(
        class_id: str,
        output_frame: str,
        timestamp_ns: int,
        reason: str,
    ) -> ObjectEstimate3D:
        """构造不可用的诊断结果；零坐标仅作为无效值占位，绝不标记成功。"""

        return ObjectEstimate3D(
            class_id,
            (0.0, 0.0, 0.0),
            0.0,
            output_frame,
            timestamp_ns,
            valid=False,
            failure_reason=reason,
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
