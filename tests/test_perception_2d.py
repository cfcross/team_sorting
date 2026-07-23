"""视觉1负责的二维感知单元测试与安全回归测试。

测试模块：``perception_2d.py`` 中的 ``OfficialYoloAdapter``、二维检测转换与
``Detection2DStabilizer`` 占位接口。测试使用轻量 ``_FakeImage``、fake backend、
``MagicMock``、临时目录和导入 patch；不加载真实 YOLO 权重，也不需要官方比赛环境。
运行 ``detect`` 的用例仍需要项目正常依赖中的 NumPy，缺少依赖时只报告环境问题，
测试不会自动安装软件。

测试通过能够证明：在 fake 依赖下，置信度与检测字段校验、官方候选路径搜索、
中心宽高到 ``bbox_xyxy`` 的转换、类别与边界过滤，以及尚未实现接口的失败方式符合
当前约定。测试通过不能证明：真实 YOLO 精度、真实权重和官方后端兼容性、相机时间
同步、三维定位、ROS2 接线、机器人实际抓取或比赛端到端成功。

pytest 入门：``test_`` 开头的函数是测试用例；``assert`` 表示必须满足的条件；
``pytest.raises`` 表示预期必须抛出异常；``parametrize`` 用多组输入重复验证同一规则；
``tmp_path`` 和 ``monkeypatch`` 是 pytest 自动提供的 fixture。fake 模拟外部依赖，
mock 还能记录函数是否按约定被调用。测试函数只用于验证代码，不参与机器人运行。

单文件运行：
``python3 -m pytest -q tests/test_perception_2d.py -p no:cacheprovider``

全套运行：
``PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider``

主要覆盖：
  1. 非法置信度阈值（NaN/Inf/bool/负数/大于1）
  2. 官方路径 examples/material_sorting/perception 解析
  3. 权重路径解析（含 checkpoint_text 和候选目录搜索）
  4. 正常检测字典转换（中心宽高 → bbox_xyxy）
  5. 未知类别、低置信度和非法数值过滤
  6. BBOX 越界裁剪与后端输出契约
  7. 模型未加载时 detect 清晰失败
  8. Detection2DStabilizer 仍明确抛出 NotImplementedError
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from team_sorting.interfaces import CameraIntrinsics, DepthFrame, Detection2D, RGBFrame
from team_sorting.perception_2d import Detection2DStabilizer, OfficialYoloAdapter


# ---------------------------------------------------------------------------
# 共享构造器、fake 依赖与测试隔离
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离每个用例的 sys.path，避免官方路径搜索影响后续测试。

    ``self_check`` 会把候选目录放到 ``sys.path`` 前部；这里让每个测试使用一份副本，
    pytest 会在用例结束时自动恢复原对象，因此临时 ``backends.py`` 不会污染执行顺序。
    """

    monkeypatch.setattr(sys, "path", list(sys.path))


class _FakeImage:
    """只提供图像尺寸；fake backend 不读取像素内容。"""

    def __init__(self, width: int, height: int) -> None:
        self.shape = (height, width, 3)


def _rgb_frame(valid: bool = True, width: int = 640, height: int = 480) -> RGBFrame:
    """构造带真实 shape 语义的轻量 RGBFrame，不依赖 NumPy 图像。"""
    return RGBFrame(
        image=_FakeImage(width, height),
        encoding="bgr8",
        frame_id="camera_optical_frame",
        timestamp_ns=100,
        valid=valid,
        failure_reason="" if valid else "图像转换失败",
    )


def _depth_frame(valid: bool = True) -> DepthFrame:
    return DepthFrame(
        image=None,
        unit_scale_m=0.001,
        frame_id="camera_optical_frame",
        timestamp_ns=100,
        valid=valid,
    )


def _intrinsics(width: int = 640, height: int = 480, valid: bool = True) -> CameraIntrinsics:
    return CameraIntrinsics(
        k=(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0),
        width=width,
        height=height,
        frame_id="camera_optical_frame",
        timestamp_ns=10,
        valid=valid,
    )


def _make_fake_backend(detections: Optional[list[dict[str, Any]]] = None) -> MagicMock:
    """创建一个带 YoloBackend 接口的 fake 后端 MagicMock。

    detect 默认返回空列表；传入 detections 可模拟特定检测结果。
    """
    fake = MagicMock()
    fake.YoloBackend.return_value.model = "fake_model"
    fake.YoloBackend.return_value.CLASS_NAMES = ("pink", "yellow", "brown")
    fake.YoloBackend.return_value.detect.return_value = (
        detections if detections is not None else []
    )
    return fake


def _make_detection_dict(
    class_id: str = "pink",
    x: float = 320.0,
    y: float = 240.0,
    w: float = 80.0,
    h: float = 100.0,
    conf: float = 0.85,
) -> dict[str, Any]:
    """构造一个符合官方字典格式的单条检测结果。"""
    return {"class": class_id, "x": x, "y": y, "w": w, "h": h, "conf": conf}


# 少量测试会访问 ``_searched``、``_backend`` 或 ``_resolve_checkpoint``。它们用于
# 固定路径搜索、自检清理和 fake backend 注入等内部可靠性语义，属于白盒回归测试，
# 可能随内部重构同步调整；新增测试应优先通过公开接口，避免继续扩大私有实现依赖。


# ---------------------------------------------------------------------------
# 适配器配置、官方路径与自检
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_threshold",
    [
        float("nan"),
        float("inf"),
        -0.1,
        1.5,
        True,
        False,
        "0.5",
        None,
    ],
)
def test_confidence_threshold_rejects_invalid_values(bad_threshold: object) -> None:
    """构造阶段必须拒绝 NaN/Inf/bool/超界值，不能把非法值传递到推理链路。"""
    with pytest.raises(ValueError):
        OfficialYoloAdapter(confidence_threshold=bad_threshold)  # type: ignore[arg-type]


def test_confidence_threshold_boundary_values() -> None:
    """0.0 和 1.0 是合法的边界值。"""
    OfficialYoloAdapter(confidence_threshold=0.0)
    OfficialYoloAdapter(confidence_threshold=1.0)


# ---------------------------------------------------------------------------
# 2. 官方路径 examples/material_sorting/perception 解析
# ---------------------------------------------------------------------------


def test_self_check_searches_examples_material_sorting_paths(tmp_path: Path) -> None:
    """工作区根目录应能定位 examples 下的后端目录和权重。"""
    examples_dir = tmp_path / "examples" / "material_sorting" / "perception"
    checkpoint = examples_dir / "checkpoints" / "material_box.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("fake weights")
    (examples_dir / "backends.py").write_text(
        "# 目录布局占位；测试通过 fake module 避免加载真实 YOLO。\n"
    )

    adapter = OfficialYoloAdapter(official_root=str(tmp_path))
    fake_module = _make_fake_backend([])
    with patch(
        "team_sorting.perception_2d.importlib.import_module",
        return_value=fake_module,
    ):
        adapter.self_check()

    searched_str = " ".join(adapter._searched)
    assert "examples/material_sorting/perception" in searched_str
    assert str(checkpoint) in searched_str
    fake_module.YoloBackend.assert_called_once_with(str(checkpoint), conf_thresh=0.65)
    assert adapter._backend is fake_module.YoloBackend.return_value


# ---------------------------------------------------------------------------
# 3. 权重路径解析
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_searches_standard_layouts(tmp_path: Path) -> None:
    """权重搜索应覆盖多种标准目录布局。"""
    # 在 checkpoints 子目录中创建假权重
    ckpt_dir = tmp_path / "material_sorting" / "perception" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "material_box.pt").write_text("fake weights")

    adapter = OfficialYoloAdapter(official_root=str(tmp_path))
    # _resolve_checkpoint 是内部方法，直接调用验证搜索逻辑
    result = adapter._resolve_checkpoint("", [tmp_path])
    assert result is not None
    assert result.name == "material_box.pt"
    assert "material_sorting/perception/checkpoints" in str(result).replace(
        str(tmp_path), ""
    )


def test_resolve_checkpoint_direct_path_takes_priority(tmp_path: Path) -> None:
    """显式指定的权重路径应优先于候选目录搜索。"""
    direct = tmp_path / "my_weights.pt"
    direct.write_text("direct weights")

    adapter = OfficialYoloAdapter(official_root=str(tmp_path))
    result = adapter._resolve_checkpoint(str(direct), [tmp_path])
    assert result == direct


def test_resolve_checkpoint_rejects_empty_file(tmp_path: Path) -> None:
    """空文件不应作为有效权重返回。"""
    empty = tmp_path / "material_box.pt"
    empty.write_text("")

    adapter = OfficialYoloAdapter(official_root=str(tmp_path))
    result = adapter._resolve_checkpoint(str(empty), [tmp_path])
    assert result is None


def test_self_check_searched_cleared_on_repeat_calls(tmp_path: Path) -> None:
    """重复调用 self_check 不应无限累积旧搜索记录。"""
    adapter = OfficialYoloAdapter(official_root=str(tmp_path))

    with patch(
        "team_sorting.perception_2d.importlib.import_module",
        side_effect=ImportError("fake missing module"),
    ), pytest.raises(RuntimeError):
        adapter.self_check()
    first_count = len(adapter._searched)

    adapter._backend = object()
    with patch(
        "team_sorting.perception_2d.importlib.import_module",
        side_effect=ImportError("fake missing module"),
    ), pytest.raises(RuntimeError):
        adapter.self_check()
    second_count = len(adapter._searched)

    assert second_count == first_count, (
        f"重复 self_check 不应累积搜索记录：第一次={first_count}，第二次={second_count}"
    )
    assert adapter._backend is None, "自检失败后不能残留上一次加载的 backend"


# ---------------------------------------------------------------------------
# 单帧检测转换与输出校验
# ---------------------------------------------------------------------------


def test_detect_converts_center_wh_to_xyxy() -> None:
    """验证中心宽高格式正确转换为 bbox_xyxy 格式。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.5)
    fake = _make_fake_backend(
        [_make_detection_dict(class_id="pink", x=320.0, y=240.0, w=80.0, h=100.0, conf=0.85)]
    )
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        result = adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())

    assert len(result) == 1
    det = result[0]
    assert det.class_id == "pink"
    assert det.bbox_xyxy == (280.0, 190.0, 360.0, 290.0)
    assert det.confidence == 0.85
    assert det.valid


def test_detect_empty_list_returns_empty_tuple() -> None:
    """官方后端返回空列表时，应返回空元组而非 None 或异常。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.5)
    fake = _make_fake_backend([])
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        result = adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())

    assert result == ()


def test_detect_none_is_reported_as_backend_contract_error() -> None:
    """None 无法表示一次正常的空检测，必须报告后端契约错误。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.5)
    fake = MagicMock()
    fake.detect.return_value = None
    with patch.object(adapter, "_backend", fake, create=True):
        with pytest.raises(RuntimeError, match="输出为 None"):
            adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())


@pytest.mark.parametrize("raw_output", [123, "not detections", {"class": "pink"}])
def test_detect_rejects_non_sequence_backend_output(raw_output: object) -> None:
    """后端必须返回检测项序列，不能返回不可迭代值或单个字典。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.5)
    fake = MagicMock()
    fake.detect.return_value = raw_output
    with patch.object(adapter, "_backend", fake, create=True):
        with pytest.raises(RuntimeError, match="检测结果序列"):
            adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())


def test_detect_invalid_inputs_return_empty() -> None:
    """RGB/Depth/Intrinsics 任一无效时，应跳过推理直接返回空元组。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.5)
    fake = _make_fake_backend([_make_detection_dict()])
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        assert adapter.detect(_rgb_frame(valid=False), _depth_frame(), _intrinsics()) == ()
        assert adapter.detect(_rgb_frame(), _depth_frame(valid=False), _intrinsics()) == ()
        assert (
            adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics(valid=False)) == ()
        )


# ---------------------------------------------------------------------------
# 5. 未知类别过滤
# ---------------------------------------------------------------------------


def test_detect_filters_unknown_classes() -> None:
    """不在 CLASS_NAMES 中的类别应被静默跳过。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.3)
    detections = [
        _make_detection_dict(class_id="pink", conf=0.9),
        _make_detection_dict(class_id="blue", conf=0.9),  # 未知
        _make_detection_dict(class_id="yellow", conf=0.9),
        _make_detection_dict(class_id="green", conf=0.9),  # 未知
    ]
    fake = _make_fake_backend(detections)
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        result = adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())

    assert len(result) == 2
    assert {d.class_id for d in result} == {"pink", "yellow"}


# ---------------------------------------------------------------------------
# 6. 低置信度过滤
# ---------------------------------------------------------------------------


def test_detect_filters_low_confidence() -> None:
    """低于 confidence_threshold 的检测应被过滤。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.65)
    detections = [
        _make_detection_dict(class_id="pink", conf=0.9),
        _make_detection_dict(class_id="yellow", conf=0.5),  # 低于阈值
        _make_detection_dict(class_id="brown", conf=0.64),  # 低于阈值
    ]
    fake = _make_fake_backend(detections)
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        result = adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())

    assert len(result) == 1
    assert result[0].class_id == "pink"


def test_detect_confidence_at_threshold_is_kept() -> None:
    """置信度恰好等于阈值时应保留。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.65)
    detections = [_make_detection_dict(class_id="yellow", conf=0.65)]
    fake = _make_fake_backend(detections)
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        result = adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())

    assert len(result) == 1


# ---------------------------------------------------------------------------
# 7. NaN/Inf 检测拒绝
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_field",
    [
        {"x": float("nan")},
        {"y": float("inf")},
        {"w": float("-inf")},
        {"h": float("nan")},
        {"conf": float("nan")},
        {"conf": float("inf")},
    ],
)
def test_detect_rejects_nan_inf_fields(bad_field: dict[str, float]) -> None:
    """坐标或置信度包含 NaN/Inf 的检测项应被静默跳过。"""
    det = _make_detection_dict()
    det.update(bad_field)
    adapter = OfficialYoloAdapter(confidence_threshold=0.3)
    fake = _make_fake_backend([det])
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        result = adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())

    assert len(result) == 0, f"应拒绝 {bad_field}"


# ---------------------------------------------------------------------------
# 8. 非正宽高拒绝
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("w", "h"),
    [
        (0.0, 100.0),
        (80.0, 0.0),
        (-10.0, 100.0),
        (80.0, -5.0),
        (0.0, 0.0),
    ],
)
def test_detect_rejects_non_positive_wh(w: float, h: float) -> None:
    """宽或高为零或负数的检测应被拒绝。"""
    det = _make_detection_dict(w=w, h=h)
    adapter = OfficialYoloAdapter(confidence_threshold=0.3)
    fake = _make_fake_backend([det])
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        result = adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())

    assert len(result) == 0, f"应拒绝 w={w}, h={h}"


# ---------------------------------------------------------------------------
# 9. 模型未加载时清晰失败
# ---------------------------------------------------------------------------


def test_detect_without_self_check_raises() -> None:
    """未通过 self_check 时 detect 必须抛出清晰的 RuntimeError。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.5)
    with pytest.raises(RuntimeError, match="尚未通过 self_check"):
        adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())


# ---------------------------------------------------------------------------
# Detection2DStabilizer 临时未实现约束
# ---------------------------------------------------------------------------

# 当前作用：防止尚未实现的多帧算法返回空值、零值或伪成功。
# 替换条件：``update`` 真正实现时，必须在同一提交中把本组占位测试替换为行为测试；
# 生产方法仍未实现时不得删除。未来至少覆盖相邻帧关联、遮挡恢复、丢失删除和置信度稳定。


def test_stabilizer_update_raises_not_implemented() -> None:
    """Detection2DStabilizer.update 应明确抛出 NotImplementedError。"""
    stabilizer = Detection2DStabilizer()
    with pytest.raises(NotImplementedError, match="视觉1"):
        stabilizer.update(())


def test_stabilizer_docstring_has_todo_hints() -> None:
    """docstring 中应包含对视觉1负责人的明确提示。"""
    doc = Detection2DStabilizer.__doc__ or ""
    assert "视觉1" in doc, "docstring 应注明由视觉1负责人实现"


# ---------------------------------------------------------------------------
# BBOX、混合结果与异常字段补充回归
# ---------------------------------------------------------------------------


def test_detect_clips_bbox_to_image_boundaries() -> None:
    """越界框应裁剪到图像范围内，完全越界（裁剪后无面积）的框应被拒绝。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.3)
    # 宽/高 200，中心在 (0, 0)，左上角 (-100, -100) 越界
    partial_out = _make_detection_dict(
        class_id="pink", x=50.0, y=50.0, w=200.0, h=200.0, conf=0.9
    )
    # 中心在图像外很远，宽高正常，裁剪后面积为 0
    fully_out = _make_detection_dict(
        class_id="yellow", x=-500.0, y=-500.0, w=50.0, h=50.0, conf=0.9
    )
    fake = _make_fake_backend([partial_out, fully_out])
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        result = adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics(640, 480))

    # 部分越界的框被裁剪后仍保留
    assert len(result) == 1
    det = result[0]
    assert det.class_id == "pink"
    # 裁剪后左上角应 >= 0
    assert det.bbox_xyxy[0] >= 0.0
    assert det.bbox_xyxy[1] >= 0.0
    # 右下角不应超过图像尺寸
    assert det.bbox_xyxy[2] <= 640.0
    assert det.bbox_xyxy[3] <= 480.0


def test_detect_uses_rgb_shape_instead_of_camera_info_size() -> None:
    """BBOX 必须按当前 RGB 帧裁剪，不能误用不一致的 CameraInfo 尺寸。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.3)
    detection = _make_detection_dict(x=310.0, y=190.0, w=100.0, h=80.0)
    fake = _make_fake_backend([detection])
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        result = adapter.detect(
            _rgb_frame(width=320, height=200),
            _depth_frame(),
            _intrinsics(width=640, height=480),
        )

    assert result[0].bbox_xyxy == (260.0, 150.0, 320.0, 200.0)


def test_detect_multiple_valid_detections_mixed_with_invalid() -> None:
    """混合有效和无效检测时，只保留通过所有校验的项。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.5)
    detections = [
        _make_detection_dict(class_id="pink", conf=0.9),           # ✓ 正常
        _make_detection_dict(class_id="yellow", conf=0.3),         # ✗ 低置信度
        _make_detection_dict(class_id="unknown", conf=0.9),        # ✗ 未知类别
        _make_detection_dict(class_id="brown", conf=0.8, w=-1.0),  # ✗ 负宽
        _make_detection_dict(class_id="brown", conf=float("nan")), # ✗ NaN 置信度
    ]
    fake = _make_fake_backend(detections)
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        result = adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())

    assert len(result) == 1
    assert result[0].class_id == "pink"


def test_detect_missing_required_field_raises() -> None:
    """缺少必需字段（class/x/y/w/h）时应抛出 RuntimeError 而非静默跳过。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.3)
    # 缺少 "w" 字段
    bad_item = {"class": "pink", "x": 320, "y": 240, "h": 100, "conf": 0.9}
    fake = _make_fake_backend([bad_item])
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        with pytest.raises(RuntimeError, match="字段无效"):
            adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())


def test_detect_missing_confidence_field_raises() -> None:
    """缺少置信度不能默认为零后静默过滤，否则会掩盖后端契约变化。"""
    adapter = OfficialYoloAdapter(confidence_threshold=0.3)
    bad_item = {"class": "pink", "x": 320, "y": 240, "w": 80, "h": 100}
    fake = _make_fake_backend([bad_item])
    with patch.object(adapter, "_backend", fake.YoloBackend.return_value, create=True):
        with pytest.raises(RuntimeError, match="conf/confidence"):
            adapter.detect(_rgb_frame(), _depth_frame(), _intrinsics())


# ---------------------------------------------------------------------------
# 构造与延迟加载边界
# ---------------------------------------------------------------------------


def test_constructor_does_not_import_deep_learning_dependencies() -> None:
    """构造 OfficialYoloAdapter 不应触发 ultralytics/torch 等深度学习导入。"""
    # 构造只在简单赋值，不调用 importlib / 不读磁盘 / 不加载模型。
    adapter = OfficialYoloAdapter(
        official_root="/nonexistent",
        checkpoint_path="/nonexistent.pt",
        module_name="nonexistent.module",
        confidence_threshold=0.7,
    )
    assert adapter._backend is None
    assert adapter.confidence_threshold == 0.7
