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


def _require_matching_base_frame(base: BaseState, output_frame: str) -> None:
    """没有显式 TF 时，底盘状态只能用于同名输出坐标系。"""

    if not isinstance(base.frame_id, str) or not base.frame_id.strip():
        raise ValueError("BaseState.frame_id 为空，不能确定相机外参输出坐标系")
    if base.frame_id != output_frame:
        raise ValueError(
            f"BaseState.frame_id ({base.frame_id!r}) 与输出 frame "
            f"({output_frame!r}) 不一致；缺少显式 TF 时不能静默重标坐标系"
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

    # 中心窗口必须再与检测框相交，避免小目标附近的大面积背景深度主导中位数。
    bbox_xa = max(0, int(math.floor(x0)))
    bbox_xb = min(width, int(math.ceil(x1)))
    bbox_ya = max(0, int(math.floor(y0)))
    bbox_yb = min(height, int(math.ceil(y1)))
    center_x = min(
        max(int(round((x0 + x1) / 2.0)), bbox_xa),
        bbox_xb - 1,
    )
    center_y = min(
        max(int(round((y0 + y1) / 2.0)), bbox_ya),
        bbox_yb - 1,
    )
    xa = max(bbox_xa, center_x - radius)
    xb = min(bbox_xb, center_x + radius + 1)
    ya = max(bbox_ya, center_y - radius)
    yb = min(bbox_yb, center_y + radius + 1)
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
        _require_matching_base_frame(base, self.output_frame)
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
    """单个物体的一对一二维身份和米制三维 EMA 状态。"""

    class_id: str
    ema: list[float]
    count: int
    last_ts_ns: int
    last_detection_ts_ns: int
    center_xy: tuple[float, float]
    velocity_xy_per_ns: tuple[float, float]
    last_world_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class _Candidate:
    """当前帧中已完成几何计算、尚未写入轨迹的目标候选。"""

    center_xy: tuple[float, float]
    world_xyz: tuple[float, float, float]
    valid_fraction: float
    confidence: float


class Perception3DEstimator:
    """把二维框转换为输出 frame 中的物体三维中心估计。

    每条检测独立执行中心窗口中位深度、针孔反投影、已知尺寸中心补偿和官方相机外参
    变换。没有上游稳定 ID 时，按类别对当前候选和历史轨迹执行全局一对一关联，结合
    运动预测、二维距离和三维距离，绝不再使用固定像素网格作为身份。关联存在歧义、
    输入不同步、frame 不一致、帧陈旧或三维跳变时返回明确的无效结果，不会把旧 EMA
    标记成当前估计。尺寸未知或调用方未明确允许启发式中心补偿时，也不会把表面点
    冒充物体中心。
    """

    def __init__(
        self,
        transform_provider: CameraTransformProvider,
        *,
        object_dimensions_m: Optional[
            dict[str, tuple[float, float, float]]
        ] = None,
        depth_radius_px: int = 4,
        ema_alpha: float = 0.5,
        converge_frames: int = 5,
        max_track_age_s: float = 1.0,
        max_input_skew_s: float = 0.05,
        max_association_distance_px: float = 80.0,
        max_position_jump_m: float = 1.0,
        association_ambiguity_margin: float = 0.05,
        center_compensation_mode: str = "heuristic",
        heuristic_center_reliability: float = 0.5,
    ) -> None:
        """保存三维估计参数，构造阶段不加载任何官方依赖。

        ``object_dimensions_m`` 的值依次为宽、高、沿相机视线近似深度，单位米。
        该参数没有默认值，ROS 组装层必须显式从已评审配置传入，避免节点启动后所有
        类别才逐帧失败。``max_input_skew_s`` 是非零同步窗口；Detection 和
        CameraInfo 不能晚于当前 Depth，且三者两两时间差必须在窗口内。

        默认 ``center_compensation_mode='heuristic'`` 会用
        ``heuristic_center_reliability`` 明确降低中心近似置信度；``'strict'`` 会完全
        拒绝尚未标定验证的表面到中心启发式。失败：参数类型、范围或物体尺寸不合法
        时抛出 ``ValueError``。
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
        max_age_s = _finite_number(max_track_age_s, "max_track_age_s")
        input_skew_s = _finite_number(max_input_skew_s, "max_input_skew_s")
        association_distance_px = _finite_number(
            max_association_distance_px, "max_association_distance_px"
        )
        position_jump_m = _finite_number(
            max_position_jump_m, "max_position_jump_m"
        )
        ambiguity_margin = _finite_number(
            association_ambiguity_margin, "association_ambiguity_margin"
        )
        center_reliability = _finite_number(
            heuristic_center_reliability, "heuristic_center_reliability"
        )
        if not 0.0 < alpha <= 1.0:
            raise ValueError("ema_alpha 必须位于 (0, 1] 范围")
        if max_age_s <= 0.0:
            raise ValueError("max_track_age_s 必须是正的有限秒数")
        if input_skew_s <= 0.0:
            raise ValueError("max_input_skew_s 必须是非零正有限秒数")
        if association_distance_px <= 0.0:
            raise ValueError("max_association_distance_px 必须是正的有限像素数")
        if position_jump_m <= 0.0:
            raise ValueError("max_position_jump_m 必须是正的有限米数")
        if ambiguity_margin < 0.0:
            raise ValueError("association_ambiguity_margin 必须是非负有限数")
        if center_compensation_mode not in {"strict", "heuristic"}:
            raise ValueError(
                "center_compensation_mode 只允许 'strict' 或 'heuristic'"
            )
        if not 0.0 < center_reliability <= 1.0:
            raise ValueError(
                "heuristic_center_reliability 必须位于 (0, 1] 范围"
            )
        if not isinstance(object_dimensions_m, dict) or not object_dimensions_m:
            raise ValueError(
                "object_dimensions_m 必须由组装层显式传入非空类别尺寸配置"
            )

        dimensions = dict(object_dimensions_m)
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
        self.max_track_age_s = max_age_s
        self.max_input_skew_s = input_skew_s
        self.max_association_distance_px = association_distance_px
        self.max_position_jump_m = position_jump_m
        self.association_ambiguity_margin = ambiguity_margin
        self.center_compensation_mode = center_compensation_mode
        self.heuristic_center_reliability = center_reliability
        self._dims = normalized_dimensions
        self._tracks: dict[str, _Track] = {}
        self._next_track_id = 1
        self._last_frame_ts_ns: Optional[int] = None
        self._last_detection_frame_ts_ns: Optional[int] = None

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
        frame 由 ``transform_provider.output_frame`` 指定，时间戳使用当前 Depth 帧
        时间。批次级时序/frame 错误会使整批无效；单条深度或关联错误只影响对应目标。
        """

        output_frame = self.transform_provider.output_frame
        failure_timestamp_ns = self._safe_failure_timestamp(
            getattr(depth, "timestamp_ns", 0)
        )
        try:
            timestamp_ns, intrinsics_timestamp_ns, depth_frame = (
                self._validate_batch_inputs(depth, intrinsics)
            )
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
        context_errors = [
            self._detection_context_error(
                detection,
                depth_frame,
                timestamp_ns,
                intrinsics_timestamp_ns,
            )
            for detection in detections
        ]
        detection_timestamps = {
            int(detection.timestamp_ns)
            for index, detection in enumerate(detections)
            if detection.valid and not context_errors[index]
        }
        if len(detection_timestamps) > 1:
            reason = (
                "二维检测上下文无效：同一批 Detection2D 必须来自同一帧，"
                f"实际时间戳为 {sorted(detection_timestamps)}"
            )
            for index, detection in enumerate(detections):
                if detection.valid and not context_errors[index]:
                    context_errors[index] = reason
            detection_timestamps.clear()
        if detection_timestamps:
            detection_timestamp_ns = next(iter(detection_timestamps))
            if (
                self._last_detection_frame_ts_ns is not None
                and detection_timestamp_ns <= self._last_detection_frame_ts_ns
            ):
                reason = (
                    "陈旧二维检测帧：Detection2D.timestamp_ns="
                    f"{detection_timestamp_ns} 未严格晚于最近有效检测帧 "
                    f"{self._last_detection_frame_ts_ns}"
                )
                for index, detection in enumerate(detections):
                    if detection.valid and not context_errors[index]:
                        context_errors[index] = reason
                detection_timestamps.clear()

        candidates: dict[int, _Candidate] = {}
        candidate_errors: dict[int, str] = {}
        for index, detection in enumerate(detections):
            if context_errors[index] or not detection.valid:
                continue
            try:
                candidates[index] = self._candidate(
                    detection, depth, intrinsics, base, joints
                )
            except (ValueError, RuntimeError) as exc:
                candidate_errors[index] = str(exc)

        associations, association_errors = self._associate_tracks(
            detections, candidates, timestamp_ns
        )
        self._last_frame_ts_ns = timestamp_ns
        if detection_timestamps:
            self._last_detection_frame_ts_ns = next(iter(detection_timestamps))

        results: list[ObjectEstimate3D] = []
        for index, detection in enumerate(detections):
            if not detection.valid:
                reason = detection.failure_reason or "上游未提供原因"
                results.append(
                    self._failure(
                        detection.class_id,
                        output_frame,
                        timestamp_ns,
                        f"二维检测无效：{reason}",
                    )
                )
                continue
            error = (
                context_errors[index]
                or candidate_errors.get(index, "")
                or association_errors.get(index, "")
            )
            if error:
                results.append(
                    self._failure(
                        detection.class_id,
                        output_frame,
                        timestamp_ns,
                        error,
                    )
                )
                continue
            candidate = candidates[index]
            point, count, track_error = self._update_track(
                associations[index], detection, candidate, timestamp_ns
            )
            if track_error:
                results.append(
                    self._failure(
                        detection.class_id,
                        output_frame,
                        timestamp_ns,
                        track_error,
                    )
                )
                continue
            converge = min(1.0, count / self.converge_frames)
            confidence = (
                candidate.confidence
                * candidate.valid_fraction
                * converge
                * self.heuristic_center_reliability
            )
            results.append(
                ObjectEstimate3D(
                    detection.class_id,
                    point,
                    max(0.0, min(1.0, confidence)),
                    output_frame,
                    timestamp_ns,
                    valid=True,
                )
            )
        return tuple(results)

    def reset_tracks(self) -> None:
        """清空全部三维 EMA 轨迹，用于切换场景或确认长时间失跟后重置。"""

        self._tracks.clear()
        self._next_track_id = 1
        self._last_frame_ts_ns = None
        self._last_detection_frame_ts_ns = None

    def _candidate(
        self,
        detection: Detection2D,
        depth: DepthFrame,
        intrinsics: CameraIntrinsics,
        base: BaseState,
        joints: RobotJointState,
    ) -> _Candidate:
        """完成单条检测的当前帧几何计算，但暂不更新任何历史。"""

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
            raise ValueError(f"深度提取失败：{exc}") from exc

        try:
            camera_point = project_pixel_to_camera(u, v, depth_m, intrinsics)
        except ValueError as exc:
            raise ValueError(f"反投影失败：{exc}") from exc

        compensated_point, compensated = self._compensate_to_center(
            detection.class_id, camera_point
        )
        if not compensated:
            raise ValueError(
                "物体中心补偿失败："
                f"类别 {detection.class_id!r} 未配置可靠物体尺寸，"
                "当前深度点仅代表可见表面"
            )
        if self.center_compensation_mode == "strict":
            raise ValueError(
                "物体中心补偿失败：表面到中心启发式尚未标定验证，"
                "strict 模式拒绝把表面点冒充物体中心"
            )
        try:
            _require_matching_base_frame(base, self.transform_provider.output_frame)
            world_point = _finite_vector(
                self.transform_provider.camera_to_output(
                    compensated_point, base, joints
                ),
                3,
                "camera_to_output 返回值",
            )
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"坐标变换失败：{exc}") from exc

        try:
            confidence = _finite_number(
                detection.confidence, "Detection2D.confidence"
            )
        except ValueError as exc:
            raise ValueError(f"置信度计算失败：{exc}") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("置信度计算失败：Detection2D.confidence 必须位于 [0, 1]")
        return _Candidate(
            (u, v),
            world_point,
            self._depth_valid_fraction(depth, detection.bbox_xyxy),
            confidence,
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

    def _associate_tracks(
        self,
        detections: tuple[Detection2D, ...],
        candidates: dict[int, _Candidate],
        timestamp_ns: int,
    ) -> tuple[dict[int, str], dict[int, str]]:
        """为当前候选执行全局一对一关联；歧义时关闭失败。"""

        associations: dict[int, str] = {}
        errors: dict[int, str] = {}
        stable_groups: dict[int, list[int]] = {}
        untagged_by_class: dict[str, list[int]] = {}
        candidate_indices = list(candidates)
        for offset, left_index in enumerate(candidate_indices):
            left = candidates[left_index]
            for right_index in candidate_indices[offset + 1 :]:
                right = candidates[right_index]
                if detections[left_index].class_id != detections[right_index].class_id:
                    continue
                if (
                    math.dist(left.center_xy, right.center_xy) <= 1.0
                    and math.dist(left.world_xyz, right.world_xyz) <= 0.01
                ):
                    reason = (
                        "目标关联歧义：同类目标观测近乎完全重叠，"
                        "无法可靠维持一对一身份"
                    )
                    errors[left_index] = reason
                    errors[right_index] = reason
        for index in candidates:
            if index in errors:
                continue
            stable_id = getattr(detections[index], "track_id", None)
            if stable_id is not None:
                if (
                    isinstance(stable_id, bool)
                    or not isinstance(stable_id, Integral)
                    or int(stable_id) < 0
                ):
                    errors[index] = "二维稳定轨迹 ID 必须是非负整数"
                    continue
                stable_groups.setdefault(int(stable_id), []).append(index)
            else:
                untagged_by_class.setdefault(
                    detections[index].class_id, []
                ).append(index)

        for stable_id, indices in stable_groups.items():
            if len(indices) != 1:
                reason = (
                    f"二维稳定轨迹 ID 重复：track_id={stable_id} "
                    "在同一帧只能对应一个目标"
                )
                for index in indices:
                    errors[index] = reason
                continue
            index = indices[0]
            key = f"stable:{stable_id}"
            track = self._tracks.get(key)
            if track is not None and track.class_id != detections[index].class_id:
                errors[index] = (
                    "二维稳定轨迹 ID 类别冲突："
                    f"历史类别 {track.class_id!r}，"
                    f"当前类别 {detections[index].class_id!r}"
                )
                continue
            if track is not None:
                assigned_cost = self._association_cost(
                    candidates[index], track, timestamp_ns
                )
                other_costs = [
                    self._association_cost(
                        candidates[index], other_track, timestamp_ns
                    )
                    for other_key, other_track in self._tracks.items()
                    if other_key != key
                    and other_track.class_id == track.class_id
                    and timestamp_ns > other_track.last_ts_ns
                ]
                finite_other_costs = [
                    cost for cost in other_costs if math.isfinite(cost)
                ]
                if finite_other_costs and (
                    not math.isfinite(assigned_cost)
                    or assigned_cost
                    > min(finite_other_costs)
                    + self.association_ambiguity_margin
                ):
                    errors[index] = (
                        "二维稳定轨迹 ID 一致性校验失败，疑似 ID 交换；"
                        "当前观测更接近另一条同类历史轨迹"
                    )
                    continue
            associations[index] = key

        for class_id, indices in untagged_by_class.items():
            eligible_indices = [index for index in indices if index not in errors]
            track_keys = [
                key
                for key, track in self._tracks.items()
                if key.startswith("auto:")
                and track.class_id == class_id
                and timestamp_ns > track.last_ts_ns
            ]
            if not eligible_indices:
                continue
            if not track_keys:
                for index in eligible_indices:
                    associations[index] = self._new_track_key()
                continue

            real_costs: list[list[float]] = []
            for index in eligible_indices:
                real_costs.append(
                    [
                        self._association_cost(
                            candidates[index], self._tracks[key], timestamp_ns
                        )
                        for key in track_keys
                    ]
                )
            new_track_cost = 2.1
            impossible_cost = 1_000_000.0
            matrix = [
                [
                    cost if math.isfinite(cost) else impossible_cost
                    for cost in row
                ]
                + [new_track_cost] * len(eligible_indices)
                for row in real_costs
            ]
            assignment = self._minimum_cost_assignment(matrix)
            for row_index, column_index in enumerate(assignment):
                detection_index = eligible_indices[row_index]
                finite_costs = sorted(
                    cost for cost in real_costs[row_index] if math.isfinite(cost)
                )
                if (
                    len(finite_costs) >= 2
                    and finite_costs[1] - finite_costs[0]
                    <= self.association_ambiguity_margin
                ):
                    errors[detection_index] = (
                        "目标关联歧义：两个历史轨迹代价过于接近，"
                        "拒绝复用任一 EMA"
                    )
                elif (
                    column_index < len(track_keys)
                    and math.isfinite(real_costs[row_index][column_index])
                ):
                    associations[detection_index] = track_keys[column_index]
                else:
                    candidate = candidates[detection_index]
                    jump_distance = self._nearby_jump_distance(
                        candidate,
                        (self._tracks[key] for key in track_keys),
                        timestamp_ns,
                    )
                    if jump_distance is not None:
                        errors[detection_index] = (
                            "三维位置跳变超限："
                            f"{jump_distance:.6f}m > "
                            f"{self.max_position_jump_m:.6f}m，"
                            "拒绝离群值且不建立新轨迹绕过检查"
                        )
                    else:
                        associations[detection_index] = self._new_track_key()
        return associations, errors

    def _association_cost(
        self,
        candidate: _Candidate,
        track: _Track,
        timestamp_ns: int,
    ) -> float:
        """返回归一化二维/三维关联代价；门限外返回无穷大。"""

        delta_ns = timestamp_ns - track.last_ts_ns
        predicted_center = (
            track.center_xy[0] + track.velocity_xy_per_ns[0] * delta_ns,
            track.center_xy[1] + track.velocity_xy_per_ns[1] * delta_ns,
        )
        pixel_distance = math.dist(candidate.center_xy, predicted_center)
        position_distance = math.dist(
            candidate.world_xyz, track.last_world_xyz
        )
        if (
            pixel_distance > self.max_association_distance_px
            or position_distance > self.max_position_jump_m
        ):
            return math.inf
        return (
            pixel_distance / self.max_association_distance_px
            + position_distance / self.max_position_jump_m
        )

    def _nearby_jump_distance(
        self,
        candidate: _Candidate,
        tracks: Any,
        timestamp_ns: int,
    ) -> Optional[float]:
        """若候选在二维上延续旧轨迹但三维跳变，返回最近的超限距离。"""

        jump_distances: list[float] = []
        for track in tracks:
            delta_ns = timestamp_ns - track.last_ts_ns
            predicted_center = (
                track.center_xy[0] + track.velocity_xy_per_ns[0] * delta_ns,
                track.center_xy[1] + track.velocity_xy_per_ns[1] * delta_ns,
            )
            if (
                math.dist(candidate.center_xy, predicted_center)
                <= self.max_association_distance_px
            ):
                distance = math.dist(
                    candidate.world_xyz, track.last_world_xyz
                )
                if distance > self.max_position_jump_m:
                    jump_distances.append(distance)
        return min(jump_distances) if jump_distances else None

    @staticmethod
    def _minimum_cost_assignment(costs: list[list[float]]) -> list[int]:
        """Hungarian 算法：每个当前检测只分配一个历史轨迹或虚拟新轨迹。"""

        if not costs:
            return []
        row_count = len(costs)
        column_count = len(costs[0])
        if column_count < row_count or any(
            len(row) != column_count for row in costs
        ):
            raise ValueError("关联代价矩阵必须为规则矩阵且列数不少于行数")
        u = [0.0] * (row_count + 1)
        v = [0.0] * (column_count + 1)
        matched_row = [0] * (column_count + 1)
        previous_column = [0] * (column_count + 1)
        for row in range(1, row_count + 1):
            matched_row[0] = row
            column0 = 0
            minimum = [math.inf] * (column_count + 1)
            used = [False] * (column_count + 1)
            while True:
                used[column0] = True
                row0 = matched_row[column0]
                delta = math.inf
                column1 = 0
                for column in range(1, column_count + 1):
                    if used[column]:
                        continue
                    current = (
                        costs[row0 - 1][column - 1] - u[row0] - v[column]
                    )
                    if current < minimum[column]:
                        minimum[column] = current
                        previous_column[column] = column0
                    if minimum[column] < delta:
                        delta = minimum[column]
                        column1 = column
                for column in range(column_count + 1):
                    if used[column]:
                        u[matched_row[column]] += delta
                        v[column] -= delta
                    else:
                        minimum[column] -= delta
                column0 = column1
                if matched_row[column0] == 0:
                    break
            while True:
                column1 = previous_column[column0]
                matched_row[column0] = matched_row[column1]
                column0 = column1
                if column0 == 0:
                    break
        assignment = [-1] * row_count
        for column in range(1, column_count + 1):
            if matched_row[column] != 0:
                assignment[matched_row[column] - 1] = column - 1
        return assignment

    def _new_track_key(self) -> str:
        key = f"auto:{self._next_track_id}"
        self._next_track_id += 1
        return key

    def _update_track(
        self,
        key: str,
        detection: Detection2D,
        candidate: _Candidate,
        timestamp_ns: int,
    ) -> tuple[tuple[float, float, float], int, str]:
        """以严格递增时间戳更新 EMA；陈旧或跳变样本不写历史。"""

        track = self._tracks.get(key)
        detection_timestamp_ns = int(detection.timestamp_ns)
        if track is None:
            track = _Track(
                detection.class_id,
                list(candidate.world_xyz),
                1,
                timestamp_ns,
                detection_timestamp_ns,
                candidate.center_xy,
                (0.0, 0.0),
                candidate.world_xyz,
            )
            self._tracks[key] = track
            return tuple(track.ema), track.count, ""
        if track.class_id != detection.class_id:
            return (
                (0.0, 0.0, 0.0),
                track.count,
                "轨迹类别冲突，拒绝复用历史 EMA",
            )
        if timestamp_ns <= track.last_ts_ns:
            return (
                (0.0, 0.0, 0.0),
                track.count,
                (
                    "陈旧轨迹样本：当前时间戳 "
                    f"{timestamp_ns} 未严格晚于 {track.last_ts_ns}"
                ),
            )
        if detection_timestamp_ns <= track.last_detection_ts_ns:
            return (
                (0.0, 0.0, 0.0),
                track.count,
                (
                    "陈旧二维轨迹样本：当前 Detection 时间戳 "
                    f"{detection_timestamp_ns} 未严格晚于 "
                    f"{track.last_detection_ts_ns}"
                ),
            )
        jump_m = math.dist(candidate.world_xyz, track.last_world_xyz)
        if jump_m > self.max_position_jump_m:
            return (
                (0.0, 0.0, 0.0),
                track.count,
                (
                    "三维位置跳变超限："
                    f"{jump_m:.6f}m > {self.max_position_jump_m:.6f}m，"
                    "拒绝离群值且不更新 EMA"
                ),
            )
        delta_ns = timestamp_ns - track.last_ts_ns
        measured_velocity = (
            (candidate.center_xy[0] - track.center_xy[0]) / delta_ns,
            (candidate.center_xy[1] - track.center_xy[1]) / delta_ns,
        )
        # 使用最近两次观测的速度进行下一帧预测。交叉期间宁可因歧义关闭失败，
        # 也不能因过度平滑速度而把两个同类目标的身份互换。
        track.velocity_xy_per_ns = measured_velocity
        for index, value in enumerate(candidate.world_xyz):
            track.ema[index] = (
                self.ema_alpha * value
                + (1.0 - self.ema_alpha) * track.ema[index]
            )
        track.count += 1
        track.last_ts_ns = timestamp_ns
        track.last_detection_ts_ns = detection_timestamp_ns
        track.center_xy = candidate.center_xy
        track.last_world_xyz = candidate.world_xyz
        return tuple(track.ema), track.count, ""

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

    def _validate_batch_inputs(
        self,
        depth: DepthFrame,
        intrinsics: CameraIntrinsics,
    ) -> tuple[int, int, str]:
        """校验 Depth/CameraInfo 有效性、方向性时间窗口和 frame。"""

        if not depth.valid:
            raise ValueError(
                f"DepthFrame 无效：{depth.failure_reason or '上游未提供原因'}"
            )
        if not intrinsics.valid:
            raise ValueError(
                f"CameraInfo 无效：{intrinsics.failure_reason or '上游未提供原因'}"
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
                "DepthFrame/CameraInfo frame 不一致："
                f"{depth_frame!r} != {intrinsics_frame!r}"
            )
        self._require_past_window(
            intrinsics_timestamp_ns,
            depth_timestamp_ns,
            "CameraInfo/DepthFrame",
        )
        return depth_timestamp_ns, intrinsics_timestamp_ns, depth_frame

    def _detection_context_error(
        self,
        detection: Detection2D,
        expected_frame_id: str,
        depth_timestamp_ns: int,
        intrinsics_timestamp_ns: int,
    ) -> str:
        """返回单条 Detection 与当前深度/内参上下文不一致的原因。"""

        if not detection.valid:
            return ""
        try:
            detection_timestamp_ns = self._timestamp_ns(
                detection.timestamp_ns, "Detection2D.timestamp_ns"
            )
            self._require_past_window(
                detection_timestamp_ns,
                depth_timestamp_ns,
                "Detection2D/DepthFrame",
            )
            self._require_absolute_window(
                detection_timestamp_ns,
                intrinsics_timestamp_ns,
                "Detection2D/CameraInfo",
            )
            # 当前冻结接口没有 Detection2D.frame_id；上游补齐该字段后，本模块会立即
            # 严格校验，而不会要求再次修改三维算法。
            if hasattr(detection, "frame_id"):
                detection_frame = self._frame_id(
                    getattr(detection, "frame_id"), "Detection2D.frame_id"
                )
                if detection_frame != expected_frame_id:
                    raise ValueError(
                        "Detection2D/DepthFrame frame 不一致："
                        f"{detection_frame!r} != {expected_frame_id!r}"
                    )
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

    def _require_past_window(
        self,
        source_timestamp_ns: int,
        current_timestamp_ns: int,
        label: str,
    ) -> None:
        delta_ns = current_timestamp_ns - source_timestamp_ns
        max_delta_ns = self.max_input_skew_s * 1_000_000_000.0
        if delta_ns < 0:
            raise ValueError(
                f"{label} 时间窗口为负：当前帧早于来源 {abs(delta_ns)}ns"
            )
        if delta_ns > max_delta_ns:
            raise ValueError(
                f"{label} 时间差 {delta_ns}ns 超过允许窗口 "
                f"{int(max_delta_ns)}ns"
            )

    def _require_absolute_window(
        self,
        first_timestamp_ns: int,
        second_timestamp_ns: int,
        label: str,
    ) -> None:
        delta_ns = abs(first_timestamp_ns - second_timestamp_ns)
        max_delta_ns = self.max_input_skew_s * 1_000_000_000.0
        if delta_ns > max_delta_ns:
            raise ValueError(
                f"{label} 时间差 {delta_ns}ns 超过允许窗口 "
                f"{int(max_delta_ns)}ns"
            )

    def _depth_valid_fraction(
        self,
        depth: DepthFrame,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> float:
        """计算 bbox 与中心窗口交集内有限且为正的深度占比。"""

        try:
            import numpy as np
        except ImportError:
            return 1.0
        image = np.asarray(depth.image)
        height, width = image.shape
        x0, y0, x1, y1 = _finite_vector(
            bbox_xyxy, 4, "Detection2D.bbox_xyxy"
        )
        bbox_xa = max(0, int(math.floor(x0)))
        bbox_xb = min(width, int(math.ceil(x1)))
        bbox_ya = max(0, int(math.floor(y0)))
        bbox_yb = min(height, int(math.ceil(y1)))
        center_x = min(
            max(int(round((x0 + x1) / 2.0)), bbox_xa),
            bbox_xb - 1,
        )
        center_y = min(
            max(int(round((y0 + y1) / 2.0)), bbox_ya),
            bbox_yb - 1,
        )
        xa = max(bbox_xa, center_x - self.depth_radius_px)
        xb = min(bbox_xb, center_x + self.depth_radius_px + 1)
        ya = max(bbox_ya, center_y - self.depth_radius_px)
        yb = min(bbox_yb, center_y + self.depth_radius_px + 1)
        patch = image[ya:yb, xa:xb].astype(float)
        valid = np.isfinite(patch) & (patch > 0.0)
        return float(np.count_nonzero(valid) / patch.size)

    @staticmethod
    def _timestamp_ns(value: Any, name: str) -> int:
        """校验非负整数纳秒时间戳。"""

        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
        ):
            raise ValueError(f"{name} 必须是非负整数纳秒")
        return int(value)

    @staticmethod
    def _frame_id(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 必须是非空字符串")
        return value

    @staticmethod
    def _safe_failure_timestamp(value: Any) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
        ):
            return 0
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
