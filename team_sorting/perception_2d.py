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
  → 后续多帧稳定（Detection2DStabilizer，由视觉1负责人实现）
  → perception_3d（结合深度和内参做三维反投影）
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import math
from numbers import Real
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


class Detection2DStabilizer:
    """二维检测多帧稳定器的接口骨架。

    本类由 **视觉1负责人** 实现完整的多帧跟踪与滤波算法。
    当前仅提供接口定义和 ``NotImplementedError`` 占位，确保上下游可以提前接入。

    输入：
      - ``detections``：当前帧的 ``Detection2D`` 元组，bbox 为 RGB 像素坐标

    输出：
      - 仍使用同一公共契约的稳定 ``Detection2D`` 元组，不新增私有输出格式

    状态：
      - 实现时可维护有限时间窗口内的检测历史；具体关联和滤波策略由视觉1评审确定

    建议测试项（视觉1实现后请补充到 tests/test_perception_2d.py）：
      - 连续相似检测能够稳定输出，孤立误检能够被过滤
      - 空帧和短时丢失按确定策略处理，过期历史不会永久保留
      - 时间戳、类别和 bbox 坐标语义保持不变

    坐标系/单位：bbox 为像素，时间为纳秒。
    失败：第一版算法尚未实现，调用 ``update`` 会抛出中文 ``NotImplementedError``。
    """

    def update(self, detections: tuple[Detection2D, ...]) -> tuple[Detection2D, ...]:
        """对一帧二维检测执行多帧稳定。

        参数：当前帧像素坐标检测；返回稳定检测元组。
        失败：第一版尚未确定跟踪和滤波策略，当前明确抛出 ``NotImplementedError``。
        """

        raise NotImplementedError(
            "二维多帧稳定算法尚未实现，请由视觉1负责人完成。"
            "入参：当前帧 Detection2D 元组；"
            "出参：稳定后的 Detection2D 元组；"
            "具体跟踪和滤波策略需另行评审并补充测试。"
        )
