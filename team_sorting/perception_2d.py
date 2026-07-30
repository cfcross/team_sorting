"""二维物料检测及官方 YOLO 薄适配层。

本文件负责检查并调用官方 ``backends.py``/YOLO 权重，把官方字典输出转换为团队
``Detection2D``；不负责三维深度、相机外参、ROS2 订阅发布或目标抓取决策。
二维检测的主要业务输入是 ``RGBFrame``；当前官方统一 backend 接口仍要求同时传入
``DepthFrame`` 和 ``CameraIntrinsics``，但本文件不做深度反投影或三维坐标估计。

Ultralytics、NumPy 和官方 backends 均在适配器自检或调用时延迟导入，缺少比赛环境
不会阻止本模块被普通 Python 测试导入。

---- 数据流（供新成员快速了解本文件在感知链中的位置） ----

RGBFrame（ROS 消息转换后的彩色帧）
  → 官方 YoloBackend（加载比赛提供的 YOLO 权重，执行单帧推理）
  → 原始字典检测（每个 dict 含 class/x/y/w/h/conf，中心宽高格式）
  → 合法性检查（类别过滤、置信度阈值、NaN/Inf 拒绝、宽高正数检查、越界裁剪）
  → Detection2D（团队统一二维框，bbox_xyxy 为 RGB 像素坐标）
  → ROS 薄接线调用的多帧稳定器（输出稳定 track_id）
  → perception_3d（结合深度和内参做三维反投影）
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import math
from numbers import Integral, Real
import os
from pathlib import Path
import sys
from typing import Any, Optional

from .interfaces import CameraIntrinsics, DepthFrame, Detection2D, RGBFrame


class OfficialYoloAdapter:
    """官方 YOLO 检测能力的薄适配器。

    参数：官方代码根目录、权重路径、可选模块名和置信度阈值。路径可由调用者配置，
    也可通过 ``MATERIAL_SORTING_OFFICIAL_ROOT`` 与
    ``MATERIAL_SORTING_YOLO_CHECKPOINT`` 覆盖。
    返回：``detect`` 输出像素坐标系中的 ``Detection2D`` 元组。
    失败：官方模块、Ultralytics、权重或模型缺失时抛出含搜索路径的 ``RuntimeError``；
    不会静默切换颜色分割或伪检测器。
    """

    # 官方 YOLO 权重训练的三类彩色盒子，顺序必须与权重的 class id 一致。
    CLASS_NAMES = ("pink", "yellow", "brown")

    def __init__(
        self,
        official_root: str = "",
        checkpoint_path: str = "",
        module_name: str = "",
        confidence_threshold: float = 0.65,
    ) -> None:
        """保存依赖配置，但不在构造阶段强制导入深度学习依赖。

        参数路径可以为空，届时 ``self_check`` 会读取环境变量。置信度无单位，范围应为
        0～1，构造阶段即校验以避免把非法值传递到推理链路。
        构造本身不加载模型，因此没有官方环境时仍可创建对象。
        """

        self.official_root = official_root
        self.checkpoint_path = checkpoint_path
        self.module_name = module_name
        # 提前校验置信度阈值，避免把 NaN/Inf/超界/bool 传递到 detect 中。
        # bool 在 Python 中是 int 子类，float(True)==1.0 会绕过数值检查，
        # 因此必须显式拒绝 bool。
        if isinstance(confidence_threshold, bool) or not isinstance(
            confidence_threshold, Real
        ):
            raise ValueError(
                "confidence_threshold 必须是 0 到 1 的有限数值，"
                f"不能使用 bool、字符串或其他类型：{confidence_threshold!r}"
            )
        conf = float(confidence_threshold)
        if not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
            raise ValueError(
                f"confidence_threshold 必须是 0 到 1 的有限浮点数，实际值={conf!r}"
            )
        self.confidence_threshold = conf
        self._backend: Any = None
        self._searched: list[str] = []

    def self_check(self) -> None:
        """定位官方 backends 和权重，并实例化官方 YoloBackend。

        参数：无；使用构造配置和环境变量。
        返回：成功时无返回值，此后可调用 ``detect``。
        单位/坐标系：不产生空间数据。
        失败：缺少模块、Ultralytics、权重、类别映射错误或官方后端加载失败时抛出
        ``RuntimeError``，错误中给出搜索项和应设置的环境变量。

        多次调用不会累积陈旧搜索记录；每次 self_check 重新搜索，方便调试路径变更。
        """

        # 每次自检都从干净状态开始；若这次加载失败，不能继续误用上次的模型。
        self._searched.clear()
        self._backend = None

        root_text = os.getenv("MATERIAL_SORTING_OFFICIAL_ROOT", self.official_root).strip()
        checkpoint_text = os.getenv(
            "MATERIAL_SORTING_YOLO_CHECKPOINT", self.checkpoint_path
        ).strip()

        # 候选根目录：覆盖直接指向 material_sorting 的路径、
        # 指向官方工作区根目录（内含 material_sorting/）、
        # 以及官方工作区根目录内含 examples/material_sorting/ 的布局。
        # 多种候选保证不同部署环境（本地开发、官方镜像、CI）都能定位到 backends.py。
        candidate_roots: list[Path] = []
        if root_text:
            root = Path(root_text).expanduser()
            candidate_roots.extend(
                [
                    root,
                    root / "material_sorting",
                    root / "material_sorting" / "perception",
                    root / "perception",
                    # 兼容官方仓库以 examples/ 为中间目录的布局
                    root / "examples" / "material_sorting",
                    root / "examples" / "material_sorting" / "perception",
                ]
            )

        # 把存在的候选目录加入 sys.path，使 importlib 能找到 backends 模块。
        # 放在 sys.path 最前面，确保优先于系统中可能存在的同名包。
        for path in candidate_roots:
            self._searched.append(str(path))
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

        # 按优先级尝试导入官方 backends 模块：显式配置名 > 标准路径 > 简写。
        module_names = [
            name
            for name in (
                self.module_name,
                "material_sorting.perception.backends",
                "perception.backends",
                "backends",
            )
            if name
        ]
        module = None
        errors: list[str] = []
        for name in module_names:
            try:
                module = importlib.import_module(name)
                self._searched.append(f"module:{name}")
                break
            except Exception as exc:  # noqa: BLE001 - 需要汇总第三方导入错误
                errors.append(f"{name}: {exc}")
                # 导入失败可能是因为该路径下没有 backends，继续尝试下一个。
                # 所有备选都失败后统一报告，方便排查 PYTHONPATH 和目录结构。

        if module is None or not hasattr(module, "YoloBackend"):
            raise RuntimeError(
                "无法导入官方 YoloBackend。缺少官方 backends.py 或其依赖；"
                f"搜索项={self._searched or module_names}；错误={errors}。"
                "请设置 MATERIAL_SORTING_OFFICIAL_ROOT 或配置 official.root。"
            )

        # 权重文件查找：先检查直接指定的路径，再在候选根目录下按标准布局搜索。
        checkpoint = self._resolve_checkpoint(checkpoint_text, candidate_roots)
        if checkpoint is None:
            raise RuntimeError(
                "找不到官方 YOLO 权重 material_box.pt；"
                f"搜索项={self._searched}。请设置 MATERIAL_SORTING_YOLO_CHECKPOINT "
                "或 config.yaml 中 official.yolo_checkpoint。"
            )

        try:
            backend = module.YoloBackend(str(checkpoint), conf_thresh=self.confidence_threshold)
        except Exception as exc:  # noqa: BLE001 - 转换为带上下文的统一错误
            raise RuntimeError(
                f"官方 YoloBackend 初始化失败：{exc}；权重={checkpoint}。"
                "请检查 Ultralytics、PyTorch/CUDA 与官方权重版本。"
            ) from exc

        # 官方 YoloBackend 可能因为权重缺失而返回 model=None 但不抛异常，
        # 此时 detect 会静默返回空列表；这里显式检查，确保自检结果可靠。
        if getattr(backend, "model", None) is None:
            raise RuntimeError(
                f"官方 YoloBackend 未成功加载模型：{checkpoint}。"
                "请检查 Ultralytics/PyTorch 依赖和权重兼容性。"
            )

        # 类别顺序校验：如果后端声明了 CLASS_NAMES，必须与团队预期一致，
        # 否则 cls_id→颜色名 的映射会错位，导致 pink 被标成 yellow 等严重错误。
        class_names = tuple(getattr(backend, "CLASS_NAMES", ()))
        if class_names and class_names != self.CLASS_NAMES:
            raise RuntimeError(
                f"YOLO 类别顺序不匹配：实际={class_names}，期望={self.CLASS_NAMES}"
            )
        self._backend = backend

    def detect(
        self,
        rgb: RGBFrame,
        depth: DepthFrame,
        intrinsics: CameraIntrinsics,
        camera_to_world_hint: Optional[Any] = None,
    ) -> tuple[Detection2D, ...]:
        """调用官方后端并转换为团队二维检测接口。

        参数：主要业务输入为 RGB 图；对齐深度、相机内参和可选变换提示用于兼容官方
        统一 backend 签名，不表示本模块负责深度反投影或使用真值完成正式检测。
        bbox 输出位于 RGB 像素坐标系，置信度无单位。返回 ``Detection2D`` 元组。
        失败：未自检、NumPy 缺失、官方输出格式错误或推理异常时抛出 ``RuntimeError``；
        单帧没有检测到目标不是异常，返回空元组即可。
        """

        if self._backend is None:
            raise RuntimeError("OfficialYoloAdapter 尚未通过 self_check，不能执行检测")

        # 输入帧无效时直接返回空检测，不浪费推理资源。
        # 这包括图像转换失败、深度缺失、内参未就绪等情况。
        if not rgb.valid or not depth.valid or not intrinsics.valid:
            return ()

        # NumPy 延迟导入：纯接口测试和自检不需要 NumPy，只在真正推理时才加载。
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("二维检测需要 NumPy；请安装比赛环境规定的 NumPy") from exc

        # BBOX 属于 RGB 像素坐标，边界必须取实际图像尺寸；CameraInfo 的宽高可能
        # 因接线或缩放配置错误而与当前帧不一致，不能拿它裁剪当前图像的检测框。
        shape = getattr(rgb.image, "shape", None)
        if shape is None or len(shape) < 2:
            raise RuntimeError("RGBFrame.image 缺少有效的图像 shape，无法校验 BBOX 边界")
        try:
            img_h = int(shape[0])
            img_w = int(shape[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"RGBFrame.image 的图像尺寸无效：shape={shape!r}") from exc
        if img_w <= 0 or img_h <= 0:
            raise RuntimeError(f"RGBFrame.image 的图像尺寸必须为正数：shape={shape!r}")

        # 调用官方后端执行单帧推理。k_matrix 是 3×3 内参矩阵；额外变换参数
        # 只为保持官方统一签名兼容，正式二维检测不能据此改用真值结果。
        try:
            k_matrix = np.asarray(intrinsics.k, dtype=float).reshape(3, 3)
            raw_detections = self._backend.detect(
                rgb.image, depth.image, k_matrix, camera_to_world_hint
            )
        except Exception as exc:  # noqa: BLE001 - 统一包装第三方推理错误
            raise RuntimeError(f"官方 YOLO 推理失败：{exc}") from exc

        # 空序列表示这一帧确实没有检测；None 或单个 dict 则违反官方输出契约，
        # 不能静默伪装成“场景里没有目标”。
        if raw_detections is None:
            raise RuntimeError("官方 YOLO 输出为 None，无法区分推理失败与正常空检测")
        if isinstance(raw_detections, (str, bytes, Mapping)):
            raise RuntimeError(
                "官方 YOLO 输出必须是可迭代的检测结果序列，不能是字符串或单个字典"
            )
        try:
            raw_items = tuple(raw_detections)
        except Exception as exc:  # noqa: BLE001 - 第三方迭代器异常需转换成契约错误
            raise RuntimeError("官方 YOLO 输出不是可迭代的检测结果序列") from exc

        converted: list[Detection2D] = []

        for item in raw_items:
            # --- 字段提取与类型校验 ---
            if not isinstance(item, Mapping):
                raise RuntimeError(f"官方检测输出项必须是字典，实际值={item!r}")
            required_fields = ("class", "x", "y", "w", "h")
            missing_fields = [name for name in required_fields if name not in item]
            if missing_fields:
                raise RuntimeError(
                    f"官方检测输出字段无效：缺少 {missing_fields}，实际值={item!r}"
                )
            if "conf" in item:
                raw_confidence = item["conf"]
            elif "confidence" in item:
                raw_confidence = item["confidence"]
            else:
                raise RuntimeError(
                    f"官方检测输出字段无效：缺少 conf/confidence，实际值={item!r}"
                )

            numeric_values = (item["x"], item["y"], item["w"], item["h"], raw_confidence)
            if any(isinstance(value, bool) for value in numeric_values):
                raise RuntimeError(f"官方检测数值字段不能使用 bool：{item!r}")
            if not isinstance(item["class"], str):
                raise RuntimeError(f"官方检测 class 必须是字符串：{item!r}")
            try:
                # 官方字典使用中心宽高格式：x/y 是 bbox 中心像素，w/h 是像素尺寸。
                cx = float(item["x"])
                cy = float(item["y"])
                width = float(item["w"])
                height = float(item["h"])
                class_id = item["class"]
                # 官方当前使用 conf；兼容 confidence，避免仅因键名变体丢掉合法检测。
                confidence = float(raw_confidence)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"官方检测输出字段无效：{item!r}，数值字段类型错误"
                ) from exc

            # --- 类别过滤：只保留官方权重训练的三种彩色盒 ---
            if class_id not in self.CLASS_NAMES:
                continue

            # --- 置信度校验：必须为 0～1 的有限数，且不低于团队阈值 ---
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                continue
            if confidence < self.confidence_threshold:
                continue

            # --- 坐标数值校验：拒绝 NaN/Inf 的中心或尺寸 ---
            if not all(math.isfinite(v) for v in (cx, cy, width, height)):
                continue

            # --- 尺寸校验：宽高必须为正，零或负数意味着无效检测 ---
            if width <= 0.0 or height <= 0.0:
                continue

            # --- BBOX 坐标转换：中心宽高 → 左上右下 (xyxy) ---
            x0 = cx - width / 2.0
            y0 = cy - height / 2.0
            x1 = cx + width / 2.0
            y1 = cy + height / 2.0

            # --- 越界裁剪：框超出图像边界时裁剪到图像范围内 ---
            # 这样做的原因：YOLO 在图像边缘的预测框可能部分越界，
            # 但越界部分仍对应真实物体的可见部分，直接丢弃会损失信息。
            # 裁剪后面积为 0（框完全在图像外）才拒绝。
            x0 = max(0.0, min(float(x0), img_w))
            y0 = max(0.0, min(float(y0), img_h))
            x1 = max(0.0, min(float(x1), img_w))
            y1 = max(0.0, min(float(y1), img_h))

            if x1 <= x0 or y1 <= y0:
                # 裁剪后无面积，说明框完全越界或尺寸无效，跳过该项。
                continue

            converted.append(
                Detection2D(
                    class_id=class_id,
                    bbox_xyxy=(x0, y0, x1, y1),
                    confidence=confidence,
                    timestamp_ns=rgb.timestamp_ns,
                    frame_id=rgb.frame_id,
                )
            )

        return tuple(converted)

    def _resolve_checkpoint(
        self, checkpoint_text: str, candidate_roots: list[Path]
    ) -> Optional[Path]:
        """在候选根目录下按标准布局查找 material_box.pt 权重文件。

        搜索顺序：先检查直接指定的路径，再在候选根目录下按多种目录布局查找。
        返回第一个存在且非空的 .pt 文件路径，或 None。
        所有搜索过的路径都会记录到 self._searched 用于错误诊断。
        """

        candidates: list[Path] = []
        # 优先使用显式指定的权重路径。
        if checkpoint_text:
            candidates.append(Path(checkpoint_text).expanduser())

        for root in candidate_roots:
            candidates.extend(
                [
                    root / "checkpoints" / "material_box.pt",
                    root / "perception" / "checkpoints" / "material_box.pt",
                    root / "material_sorting" / "perception" / "checkpoints" / "material_box.pt",
                    # 兼容官方仓库以 examples/material_sorting 为中间目录的布局
                    root / "examples" / "material_sorting" / "perception" / "checkpoints" / "material_box.pt",
                ]
            )

        for path in candidates:
            self._searched.append(str(path))
            if path.is_file() and path.stat().st_size > 0:
                return path

        return None


@dataclass
class _DetectionTrack:
    """稳定器实例内部的可变轨迹；该类型不会跨公共接口暴露。"""

    track_id: int
    class_id: str
    frame_id: str
    bbox_xyxy: tuple[float, float, float, float]
    observed_bbox_xyxy: tuple[float, float, float, float]
    velocity_xy_per_ns: tuple[float, float]
    confidence: float
    last_timestamp_ns: int
    hit_count: int
    missed_frames: int
    confirmed: bool


def _validated_unit_interval(value: Any, name: str) -> float:
    """校验 0～1 的有限实数，同时拒绝 Python 中属于整数子类的 bool。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是 0 到 1 的有限数值，不能使用 bool 或其他类型")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须能安全转换为有限浮点数") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} 必须位于 0 到 1，实际值={result!r}")
    return result


def _validated_frame_count(value: Any, name: str, *, allow_zero: bool) -> int:
    """校验帧数参数，不把 True/False 当作 1/0 接受。"""

    if isinstance(value, bool) or not isinstance(value, Integral):
        qualifier = "非负" if allow_zero else "正"
        raise ValueError(f"{name} 必须是{qualifier}整数，不能使用 bool 或其他类型")
    result = int(value)
    if result < 0 or (not allow_zero and result == 0):
        qualifier = "非负" if allow_zero else "正"
        raise ValueError(f"{name} 必须是{qualifier}整数，实际值={result}")
    return result


def _detection_timestamp(value: Any) -> Optional[int]:
    """返回合法的非负纳秒时间；非法时间不参与帧时序判断。"""

    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    result = int(value)
    return result if result >= 0 else None


def _normalized_detection(detection: Any) -> Optional[Detection2D]:
    """把一条合法检测复制为规范浮点值；单条坏数据不会破坏同帧其他检测。"""

    if not isinstance(detection, Detection2D) or detection.valid is not True:
        return None
    if (
        not isinstance(detection.class_id, str)
        or not detection.class_id
        or detection.class_id not in OfficialYoloAdapter.CLASS_NAMES
    ):
        return None
    timestamp_ns = _detection_timestamp(detection.timestamp_ns)
    if timestamp_ns is None:
        return None
    if not isinstance(detection.frame_id, str):
        return None

    bbox = detection.bbox_xyxy
    if isinstance(bbox, (str, bytes)):
        return None
    try:
        bbox_items = tuple(bbox)
    except TypeError:
        return None
    if len(bbox_items) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in bbox_items):
        return None
    try:
        bbox_xyxy = tuple(float(value) for value in bbox_items)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in bbox_xyxy):
        return None
    x0, y0, x1, y1 = bbox_xyxy
    if x1 <= x0 or y1 <= y0:
        return None

    confidence_value = detection.confidence
    if isinstance(confidence_value, bool) or not isinstance(confidence_value, Real):
        return None
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None

    # 输入 dataclass 是不可变对象；复制后只把规范值写入私有轨迹，绝不修改上游对象。
    return Detection2D(
        class_id=detection.class_id,
        bbox_xyxy=(x0, y0, x1, y1),
        confidence=confidence,
        timestamp_ns=timestamp_ns,
        frame_id=detection.frame_id,
    )


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """计算两个合法 xyxy 像素框的交并比。"""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    if (
        not all(
            math.isfinite(value)
            for value in (intersection, first_area, second_area, union)
        )
        or union <= 0.0
    ):
        return 0.0
    score = intersection / union
    return score if math.isfinite(score) else 0.0


def _bbox_center(
    bbox_xyxy: tuple[float, float, float, float],
) -> tuple[float, float]:
    return (
        (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0,
        (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0,
    )


def _predicted_bbox(
    track: _DetectionTrack,
    timestamp_ns: int,
) -> tuple[float, float, float, float]:
    """按最近两次原始观测的中心速度预测当前框，避免交叉时按位置换ID。"""

    delta_ns = timestamp_ns - track.last_timestamp_ns
    if delta_ns <= 0:
        return track.observed_bbox_xyxy
    offset_x = track.velocity_xy_per_ns[0] * delta_ns
    offset_y = track.velocity_xy_per_ns[1] * delta_ns
    predicted = (
        track.observed_bbox_xyxy[0] + offset_x,
        track.observed_bbox_xyxy[1] + offset_y,
        track.observed_bbox_xyxy[2] + offset_x,
        track.observed_bbox_xyxy[3] + offset_y,
    )
    if not all(math.isfinite(value) for value in predicted):
        return track.observed_bbox_xyxy
    return predicted


class Detection2DStabilizer:
    """轻量、确定的二维多目标多帧稳定器。

    每次 ``update`` 视为处理一帧检测：只允许同类别、同 frame 轨迹按最近观测速度
    预测框与当前框的 IoU 做确定的一对一关联。预测框避免两个目标交叉时让轨迹身份
    简单跟随当前位置交换；新轨迹连续命中 ``min_confirmed_hits`` 次后才输出，匹配框
    和置信度分别使用 EMA 平滑。当前帧未匹配的轨迹只保留在实例内部，超过
    ``max_missed_frames`` 后删除，丢失帧不会重复输出旧检测。

    非空输入中，合法检测应来自同一 RGB 帧并具有同一纳秒时间戳；防御性处理混合
    时间戳时只采用其中最新的合法检测组。该时间戳若不严格晚于最近已处理的非空帧，
    整次更新会被视为乱序/重复帧：不修改轨迹、不增加丢失计数并返回空元组。ROS
    接线必须通过 ``frame_timestamp_ns``/``frame_id`` 传入当前 RGB 上下文，使空检测
    帧同样能执行时序检查；兼容调用若不给空帧时间，则只增加一次丢失计数。

    输入和输出均为 ``tuple[Detection2D, ...]``，bbox 始终是 RGB 像素坐标
    ``(x0,y0,x1,y1)``。确认后的输出把实例内非负轨迹编号写入
    ``Detection2D.track_id``，并保留 RGB ``frame_id``，供三维估计维持同类多目标
    的稳定身份。调用 ``reset`` 可恢复到刚构造的状态。
    """

    def __init__(
        self,
        iou_match_threshold: float = 0.3,
        min_confirmed_hits: int = 2,
        max_missed_frames: int = 2,
        bbox_smoothing_alpha: float = 0.5,
        confidence_smoothing_alpha: float = 0.5,
    ) -> None:
        """保存稳定策略参数并创建实例私有轨迹状态。

        IoU 和两个 EMA alpha 均为 0～1 的有限数；确认帧数必须为正整数，最大丢失
        帧数必须为非负整数。所有参数都显式拒绝 bool、NaN、Inf 和越界值。
        """

        self.iou_match_threshold = _validated_unit_interval(
            iou_match_threshold, "iou_match_threshold"
        )
        self.min_confirmed_hits = _validated_frame_count(
            min_confirmed_hits, "min_confirmed_hits", allow_zero=False
        )
        self.max_missed_frames = _validated_frame_count(
            max_missed_frames, "max_missed_frames", allow_zero=True
        )
        self.bbox_smoothing_alpha = _validated_unit_interval(
            bbox_smoothing_alpha, "bbox_smoothing_alpha"
        )
        self.confidence_smoothing_alpha = _validated_unit_interval(
            confidence_smoothing_alpha, "confidence_smoothing_alpha"
        )
        self._tracks: dict[int, _DetectionTrack] = {}
        self._next_track_id = 0
        self._last_timestamp_ns: Optional[int] = None

    def reset(self) -> None:
        """清空轨迹、编号和最近时间戳，恢复到刚构造后的状态。"""

        self._tracks.clear()
        self._next_track_id = 0
        self._last_timestamp_ns = None

    def update(
        self,
        detections: tuple[Detection2D, ...],
        *,
        frame_timestamp_ns: Optional[int] = None,
        frame_id: Optional[str] = None,
    ) -> tuple[Detection2D, ...]:
        """关联并平滑当前帧检测，只返回本帧实际匹配且已确认的目标。

        无效检测会逐条忽略。空帧会让现有轨迹增加一次丢失，但返回空元组；乱序或
        重复帧则完全不改变状态。节点接线应始终传入 RGB 帧时间与 frame，使空帧也
        具有明确上下文。输出时间戳始终来自本帧实际检测，不做平均。
        """

        explicit_timestamp_ns: Optional[int] = None
        if frame_timestamp_ns is not None:
            explicit_timestamp_ns = _detection_timestamp(frame_timestamp_ns)
            if explicit_timestamp_ns is None:
                raise ValueError("frame_timestamp_ns 必须是非负整数纳秒")
        if frame_id is not None and (
            not isinstance(frame_id, str) or not frame_id.strip()
        ):
            raise ValueError("frame_id 必须是非空字符串或None")

        normalized = tuple(
            item
            for detection in detections
            if (item := _normalized_detection(detection)) is not None
        )

        if explicit_timestamp_ns is not None:
            mismatched = tuple(
                item
                for item in normalized
                if item.timestamp_ns != explicit_timestamp_ns
                or (frame_id is not None and item.frame_id != frame_id)
            )
            if mismatched:
                raise ValueError(
                    "Detection2D 与显式 RGB 帧的 timestamp/frame不一致"
                )
            current = normalized
            current_timestamp_ns = explicit_timestamp_ns
        elif normalized:
            # OfficialYoloAdapter 会给同帧所有框相同时间；若上游违反该约定，只处理
            # 最新一组，避免较旧检测让某条轨迹的时间倒退。
            current_timestamp_ns = max(item.timestamp_ns for item in normalized)
            current = tuple(
                item
                for item in normalized
                if item.timestamp_ns == current_timestamp_ns
            )
        else:
            # 即使检测的类别、框或置信度无效，只要它携带合法帧时间，就仍可判断这
            # 是否是一帧陈旧输入；完全空帧则没有时间依据。
            timestamps = tuple(
                timestamp
                for detection in detections
                if isinstance(detection, Detection2D)
                and (timestamp := _detection_timestamp(detection.timestamp_ns)) is not None
            )
            if not timestamps:
                self._mark_tracks_missed(set(self._tracks))
                return ()
            current_timestamp_ns = max(timestamps)
            current = ()

        if (
            self._last_timestamp_ns is not None
            and current_timestamp_ns <= self._last_timestamp_ns
        ):
            return ()
        self._last_timestamp_ns = current_timestamp_ns

        # 先按类别和几何排序，避免后端输出顺序影响相同输入集合的关联结果。
        class_order = {
            class_id: index
            for index, class_id in enumerate(OfficialYoloAdapter.CLASS_NAMES)
        }
        ordered = tuple(
            sorted(
                current,
                key=lambda item: (
                    class_order[item.class_id],
                    *item.bbox_xyxy,
                    item.confidence,
                ),
            )
        )

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(ordered):
                if track.class_id != detection.class_id:
                    continue
                if track.frame_id != detection.frame_id:
                    continue
                score = max(
                    _bbox_iou(
                        _predicted_bbox(track, current_timestamp_ns),
                        detection.bbox_xyxy,
                    ),
                    _bbox_iou(
                        track.observed_bbox_xyxy,
                        detection.bbox_xyxy,
                    ),
                )
                # 即使调用者把阈值设为 0，也不能把完全无重叠的远目标合并。
                if score > 0.0 and score >= self.iou_match_threshold:
                    candidates.append((score, track_id, detection_index))

        # 分数高者优先；相同分数按稳定的轨迹编号和规范检测顺序打破平局。
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()
        for _, track_id, detection_index in candidates:
            if (
                track_id in matched_track_ids
                or detection_index in matched_detection_indices
            ):
                continue
            if not self._update_track(
                self._tracks[track_id], ordered[detection_index]
            ):
                continue
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)

        existing_track_ids = set(self._tracks)
        self._mark_tracks_missed(existing_track_ids - matched_track_ids)

        for detection_index, detection in enumerate(ordered):
            if detection_index in matched_detection_indices:
                continue
            track = self._create_track(detection)
            if track.confirmed:
                matched_track_ids.add(track.track_id)

        outputs = [
            (
                track_id,
                Detection2D(
                    class_id=track.class_id,
                    bbox_xyxy=track.bbox_xyxy,
                    confidence=track.confidence,
                    timestamp_ns=track.last_timestamp_ns,
                    valid=True,
                    failure_reason="",
                    frame_id=track.frame_id,
                    track_id=track.track_id,
                ),
            )
            for track_id in matched_track_ids
            if (track := self._tracks.get(track_id)) is not None and track.confirmed
        ]
        outputs.sort(
            key=lambda pair: (
                class_order[pair[1].class_id],
                pair[1].bbox_xyxy[0],
                pair[1].bbox_xyxy[1],
                pair[1].bbox_xyxy[2],
                pair[1].bbox_xyxy[3],
                pair[0],
            )
        )
        return tuple(item for _, item in outputs)

    def _create_track(self, detection: Detection2D) -> _DetectionTrack:
        track = _DetectionTrack(
            track_id=self._next_track_id,
            class_id=detection.class_id,
            frame_id=detection.frame_id,
            bbox_xyxy=detection.bbox_xyxy,
            observed_bbox_xyxy=detection.bbox_xyxy,
            velocity_xy_per_ns=(0.0, 0.0),
            confidence=detection.confidence,
            last_timestamp_ns=detection.timestamp_ns,
            hit_count=1,
            missed_frames=0,
            confirmed=self.min_confirmed_hits == 1,
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _update_track(self, track: _DetectionTrack, detection: Detection2D) -> bool:
        bbox_alpha = self.bbox_smoothing_alpha
        smoothed_bbox = tuple(
            bbox_alpha * current + (1.0 - bbox_alpha) * previous
            for previous, current in zip(track.bbox_xyxy, detection.bbox_xyxy)
        )
        x0, y0, x1, y1 = smoothed_bbox
        if (
            not all(math.isfinite(value) for value in smoothed_bbox)
            or x1 <= x0
            or y1 <= y0
        ):
            # 合法输入框的凸组合理论上仍合法；若浮点异常破坏该不变量，就拒绝更新，
            # 不能把危险框继续交给三维估计。
            return False

        confidence_alpha = self.confidence_smoothing_alpha
        smoothed_confidence = (
            confidence_alpha * detection.confidence
            + (1.0 - confidence_alpha) * track.confidence
        )
        if not math.isfinite(smoothed_confidence):
            return False

        previous_center = _bbox_center(track.observed_bbox_xyxy)
        current_center = _bbox_center(detection.bbox_xyxy)
        delta_ns = detection.timestamp_ns - track.last_timestamp_ns
        if delta_ns <= 0:
            return False
        velocity_xy_per_ns = (
            (current_center[0] - previous_center[0]) / delta_ns,
            (current_center[1] - previous_center[1]) / delta_ns,
        )
        if not all(math.isfinite(value) for value in velocity_xy_per_ns):
            return False

        track.bbox_xyxy = (x0, y0, x1, y1)
        track.observed_bbox_xyxy = detection.bbox_xyxy
        track.velocity_xy_per_ns = velocity_xy_per_ns
        track.confidence = min(1.0, max(0.0, smoothed_confidence))
        track.last_timestamp_ns = detection.timestamp_ns
        track.hit_count += 1
        track.missed_frames = 0
        if track.hit_count >= self.min_confirmed_hits:
            track.confirmed = True
        return True

    def _mark_tracks_missed(self, track_ids: set[int]) -> None:
        for track_id in sorted(track_ids):
            track = self._tracks.get(track_id)
            if track is None:
                continue
            track.missed_frames += 1
            # 候选轨迹必须连续命中才能确认；已确认轨迹则保留确认状态以支持短时遮挡。
            if not track.confirmed:
                track.hit_count = 0
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[track_id]
