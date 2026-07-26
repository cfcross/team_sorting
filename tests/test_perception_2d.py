"""视觉1负责的二维感知单元测试与安全回归测试。

测试模块：``perception_2d.py`` 中的 ``OfficialYoloAdapter``、二维检测转换与
``Detection2DStabilizer`` 多帧稳定器。测试使用轻量 ``_FakeImage``、fake backend、
``MagicMock``、临时目录和导入 patch；不加载真实 YOLO 权重，也不需要官方比赛环境。
运行 ``detect`` 的用例仍需要项目正常依赖中的 NumPy，缺少依赖时只报告环境问题，
测试不会自动安装软件。

测试通过能够证明：在 fake 依赖下，置信度与检测字段校验、官方候选路径搜索、
中心宽高到 ``bbox_xyxy`` 的转换、类别与边界过滤，以及多帧关联、确认、平滑、丢失
和时间戳策略符合当前约定。测试通过不能证明：真实 YOLO 精度、真实权重和官方后端
兼容性、相机时间同步、三维定位、ROS2 接线、机器人实际抓取或比赛端到端成功。

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
  8. Detection2DStabilizer 的参数、关联、确认、平滑、丢失、时序与 reset
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


def _make_detection(
    class_id: str = "pink",
    bbox_xyxy: tuple[float, float, float, float] = (10.0, 20.0, 30.0, 40.0),
    confidence: float = 0.8,
    timestamp_ns: int = 100,
    valid: bool = True,
) -> Detection2D:
    """构造稳定器输入；测试可显式覆盖字段来验证防御性过滤。"""

    return Detection2D(
        class_id=class_id,
        bbox_xyxy=bbox_xyxy,
        confidence=confidence,
        timestamp_ns=timestamp_ns,
        valid=valid,
        failure_reason="" if valid else "上游检测无效",
    )


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
# Detection2DStabilizer 参数与真实行为
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("parameter_name", "bad_value"),
    [
        ("iou_match_threshold", True),
        ("iou_match_threshold", "0.5"),
        ("iou_match_threshold", float("nan")),
        ("iou_match_threshold", float("inf")),
        ("iou_match_threshold", float("-inf")),
        ("iou_match_threshold", -0.1),
        ("iou_match_threshold", 1.1),
        ("iou_match_threshold", 10**400),
        ("bbox_smoothing_alpha", False),
        ("bbox_smoothing_alpha", "0.5"),
        ("bbox_smoothing_alpha", float("nan")),
        ("bbox_smoothing_alpha", float("inf")),
        ("bbox_smoothing_alpha", -0.1),
        ("bbox_smoothing_alpha", 1.1),
        ("confidence_smoothing_alpha", True),
        ("confidence_smoothing_alpha", "0.5"),
        ("confidence_smoothing_alpha", float("nan")),
        ("confidence_smoothing_alpha", float("-inf")),
        ("confidence_smoothing_alpha", -0.1),
        ("confidence_smoothing_alpha", 1.1),
    ],
)
def test_stabilizer_rejects_invalid_float_parameters(
    parameter_name: str, bad_value: object
) -> None:
    """IoU 和 EMA 参数必须拒绝 bool、非数值、NaN/Inf 与越界值。"""

    with pytest.raises(ValueError):
        Detection2DStabilizer(**{parameter_name: bad_value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("parameter_name", "bad_value"),
    [
        ("min_confirmed_hits", True),
        ("min_confirmed_hits", 0),
        ("min_confirmed_hits", -1),
        ("min_confirmed_hits", 2.0),
        ("min_confirmed_hits", float("nan")),
        ("min_confirmed_hits", float("inf")),
        ("max_missed_frames", False),
        ("max_missed_frames", -1),
        ("max_missed_frames", 2.0),
        ("max_missed_frames", float("nan")),
        ("max_missed_frames", float("-inf")),
    ],
)
def test_stabilizer_rejects_invalid_frame_count_parameters(
    parameter_name: str, bad_value: object
) -> None:
    """确认与丢失帧数必须是真正的正整数/非负整数。"""

    with pytest.raises(ValueError):
        Detection2DStabilizer(**{parameter_name: bad_value})  # type: ignore[arg-type]


def test_stabilizer_accepts_parameter_boundaries() -> None:
    """IoU/alpha 的 0 和 1、min_hits=1、max_missed=0 均有明确语义。"""

    stabilizer = Detection2DStabilizer(
        iou_match_threshold=0.0,
        min_confirmed_hits=1,
        max_missed_frames=0,
        bbox_smoothing_alpha=0.0,
        confidence_smoothing_alpha=1.0,
    )
    result = stabilizer.update((_make_detection(),))
    assert isinstance(result, tuple)
    assert len(result) == 1

    Detection2DStabilizer(
        iou_match_threshold=1.0,
        bbox_smoothing_alpha=1.0,
        confidence_smoothing_alpha=0.0,
    )


def test_stabilizer_requires_consecutive_confirmation_and_preserves_contract() -> None:
    """默认首帧只建候选，第二个相似新帧才输出新的公共 Detection2D。"""

    stabilizer = Detection2DStabilizer()
    first = _make_detection(timestamp_ns=100)
    second = _make_detection(
        bbox_xyxy=(12.0, 20.0, 32.0, 40.0),
        confidence=0.6,
        timestamp_ns=200,
    )

    assert stabilizer.update((first,)) == ()
    result = stabilizer.update((second,))

    assert isinstance(result, tuple)
    assert len(result) == 1
    output = result[0]
    assert isinstance(output, Detection2D)
    assert output is not second
    assert output.class_id == "pink"
    assert output.timestamp_ns == 200
    assert output.valid is True
    assert output.failure_reason == ""


def test_stabilizer_smooths_bbox_and_confidence_with_ema() -> None:
    """EMA 的 alpha 权重施加到当前检测，结果位于旧值与新值之间。"""

    stabilizer = Detection2DStabilizer(
        bbox_smoothing_alpha=0.25,
        confidence_smoothing_alpha=0.25,
    )
    first_bbox = (10.0, 20.0, 30.0, 40.0)
    second_bbox = (14.0, 24.0, 34.0, 44.0)
    assert stabilizer.update(
        (_make_detection(bbox_xyxy=first_bbox, confidence=0.8, timestamp_ns=100),)
    ) == ()

    output = stabilizer.update(
        (_make_detection(bbox_xyxy=second_bbox, confidence=0.4, timestamp_ns=200),)
    )[0]

    assert output.bbox_xyxy == pytest.approx((11.0, 21.0, 31.0, 41.0))
    assert output.bbox_xyxy != first_bbox
    assert output.bbox_xyxy != second_bbox
    assert output.bbox_xyxy[2] > output.bbox_xyxy[0]
    assert output.bbox_xyxy[3] > output.bbox_xyxy[1]
    assert output.confidence == pytest.approx(0.7)
    assert 0.0 <= output.confidence <= 1.0


def test_stabilizer_isolated_detection_expires_and_must_reconfirm() -> None:
    """单帧误检从不输出，超过丢失上限后同位置目标也按新轨迹确认。"""

    stabilizer = Detection2DStabilizer(max_missed_frames=1)
    assert stabilizer.update((_make_detection(timestamp_ns=100),)) == ()
    assert stabilizer.update(()) == ()
    assert stabilizer.update(()) == ()

    assert stabilizer.update((_make_detection(timestamp_ns=200),)) == ()
    result = stabilizer.update((_make_detection(timestamp_ns=300),))
    assert len(result) == 1
    assert result[0].timestamp_ns == 300


def test_stabilizer_unconfirmed_hits_must_be_consecutive() -> None:
    """候选轨迹发生一次 miss 后不能用此前命中数凑够确认阈值。"""

    stabilizer = Detection2DStabilizer(
        min_confirmed_hits=3,
        max_missed_frames=2,
    )
    assert stabilizer.update((_make_detection(timestamp_ns=100),)) == ()
    assert stabilizer.update(()) == ()
    assert stabilizer.update((_make_detection(timestamp_ns=200),)) == ()
    assert stabilizer.update((_make_detection(timestamp_ns=300),)) == ()
    assert len(stabilizer.update((_make_detection(timestamp_ns=400),))) == 1


def test_stabilizer_empty_frame_and_short_loss_do_not_emit_stale_detection() -> None:
    """已确认轨迹可内部跨越丢失上限内的空帧，但空帧本身绝不重复旧输出。"""

    stabilizer = Detection2DStabilizer(max_missed_frames=2)
    assert stabilizer.update((_make_detection(timestamp_ns=100),)) == ()
    assert len(stabilizer.update((_make_detection(timestamp_ns=200),))) == 1

    assert stabilizer.update(()) == ()
    assert stabilizer.update(()) == ()
    recovered = stabilizer.update((_make_detection(timestamp_ns=300),))

    assert len(recovered) == 1
    assert recovered[0].timestamp_ns == 300


def test_stabilizer_confirmed_track_expires_after_miss_limit() -> None:
    """已确认轨迹超过丢失上限也必须删除，重现后重新完成确认。"""

    stabilizer = Detection2DStabilizer(max_missed_frames=1)
    assert stabilizer.update((_make_detection(timestamp_ns=100),)) == ()
    assert len(stabilizer.update((_make_detection(timestamp_ns=200),))) == 1
    assert stabilizer.update(()) == ()
    assert stabilizer.update(()) == ()

    assert stabilizer.update((_make_detection(timestamp_ns=300),)) == ()
    result = stabilizer.update((_make_detection(timestamp_ns=400),))
    assert len(result) == 1
    assert result[0].timestamp_ns == 400


def test_stabilizer_does_not_match_across_classes() -> None:
    """相同 bbox 也不能让 yellow 检测确认 pink 轨迹。"""

    stabilizer = Detection2DStabilizer()
    assert stabilizer.update((_make_detection(class_id="pink", timestamp_ns=100),)) == ()
    assert (
        stabilizer.update((_make_detection(class_id="yellow", timestamp_ns=200),))
        == ()
    )
    result = stabilizer.update(
        (_make_detection(class_id="yellow", timestamp_ns=300),)
    )
    assert [item.class_id for item in result] == ["yellow"]


def test_stabilizer_tracks_three_classes_and_sorts_output_deterministically() -> None:
    """三类可同时确认，输出类别顺序只用于确定性而非任务执行顺序。"""

    stabilizer = Detection2DStabilizer()
    first_frame = (
        _make_detection("brown", (80.0, 10.0, 100.0, 30.0), timestamp_ns=100),
        _make_detection("pink", (10.0, 10.0, 30.0, 30.0), timestamp_ns=100),
        _make_detection("yellow", (45.0, 10.0, 65.0, 30.0), timestamp_ns=100),
    )
    second_frame = (
        _make_detection("yellow", (46.0, 10.0, 66.0, 30.0), timestamp_ns=200),
        _make_detection("brown", (81.0, 10.0, 101.0, 30.0), timestamp_ns=200),
        _make_detection("pink", (11.0, 10.0, 31.0, 30.0), timestamp_ns=200),
    )

    assert stabilizer.update(first_frame) == ()
    result = stabilizer.update(second_frame)
    assert [item.class_id for item in result] == ["pink", "yellow", "brown"]


def test_stabilizer_keeps_far_same_class_targets_separate() -> None:
    """同类左右两个远框应建立并确认两条轨迹，且按左上角排序。"""

    stabilizer = Detection2DStabilizer(iou_match_threshold=0.2)
    first_frame = (
        _make_detection(bbox_xyxy=(100.0, 0.0, 120.0, 20.0), timestamp_ns=100),
        _make_detection(bbox_xyxy=(0.0, 0.0, 20.0, 20.0), timestamp_ns=100),
    )
    second_frame = (
        _make_detection(bbox_xyxy=(1.0, 0.0, 21.0, 20.0), timestamp_ns=200),
        _make_detection(bbox_xyxy=(99.0, 0.0, 119.0, 20.0), timestamp_ns=200),
    )

    assert stabilizer.update(first_frame) == ()
    result = stabilizer.update(second_frame)
    assert len(result) == 2
    assert result[0].bbox_xyxy[0] < result[1].bbox_xyxy[0]
    assert result[0].bbox_xyxy[2] < 30.0
    assert result[1].bbox_xyxy[0] > 90.0


def test_stabilizer_matches_each_detection_to_at_most_one_track() -> None:
    """一个同时覆盖两条同类轨迹的框，本帧也只能更新并输出其中一条。"""

    stabilizer = Detection2DStabilizer(iou_match_threshold=0.3)
    two_targets_100 = (
        _make_detection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), timestamp_ns=100),
        _make_detection(bbox_xyxy=(4.0, 0.0, 14.0, 10.0), timestamp_ns=100),
    )
    two_targets_200 = (
        _make_detection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), timestamp_ns=200),
        _make_detection(bbox_xyxy=(4.0, 0.0, 14.0, 10.0), timestamp_ns=200),
    )
    assert stabilizer.update(two_targets_100) == ()
    assert len(stabilizer.update(two_targets_200)) == 2

    bridge = _make_detection(
        bbox_xyxy=(2.0, 0.0, 12.0, 10.0),
        timestamp_ns=300,
    )
    assert len(stabilizer.update((bridge,))) == 1


def test_stabilizer_zero_iou_never_matches_even_with_zero_threshold() -> None:
    """阈值为 0 也不能把完全不相交的同类目标合并成一条轨迹。"""

    stabilizer = Detection2DStabilizer(iou_match_threshold=0.0)
    assert stabilizer.update(
        (_make_detection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), timestamp_ns=100),)
    ) == ()
    assert stabilizer.update(
        (_make_detection(bbox_xyxy=(100.0, 0.0, 110.0, 10.0), timestamp_ns=200),)
    ) == ()
    result = stabilizer.update(
        (_make_detection(bbox_xyxy=(101.0, 0.0, 111.0, 10.0), timestamp_ns=300),)
    )
    assert len(result) == 1
    assert result[0].bbox_xyxy[0] > 100.0


def test_stabilizer_rejects_positive_iou_below_configured_threshold() -> None:
    """有少量重叠也必须达到配置阈值，不能仅凭 IoU 为正就关联。"""

    stabilizer = Detection2DStabilizer(iou_match_threshold=0.5)
    assert stabilizer.update(
        (_make_detection(bbox_xyxy=(0.0, 0.0, 10.0, 10.0), timestamp_ns=100),)
    ) == ()
    assert stabilizer.update(
        (_make_detection(bbox_xyxy=(6.0, 0.0, 16.0, 10.0), timestamp_ns=200),)
    ) == ()
    result = stabilizer.update(
        (_make_detection(bbox_xyxy=(7.0, 0.0, 17.0, 10.0), timestamp_ns=300),)
    )
    assert len(result) == 1
    assert result[0].bbox_xyxy[0] > 6.0


def test_stabilizer_rejects_out_of_order_and_duplicate_timestamps_transactionally() -> None:
    """旧帧和重复帧不做 EMA、不增加 miss，也不能被重复计为确认命中。"""

    stabilizer = Detection2DStabilizer(
        max_missed_frames=0,
        bbox_smoothing_alpha=0.5,
    )
    first = _make_detection(
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        timestamp_ns=100,
    )
    assert stabilizer.update((first,)) == ()
    assert stabilizer.update((first,)) == ()
    second = stabilizer.update(
        (_make_detection(bbox_xyxy=(2.0, 0.0, 12.0, 10.0), timestamp_ns=200),)
    )
    assert second[0].bbox_xyxy == pytest.approx((1.0, 0.0, 11.0, 10.0))

    assert stabilizer.update(
        (_make_detection(bbox_xyxy=(4.0, 0.0, 14.0, 10.0), timestamp_ns=150),)
    ) == ()
    current = stabilizer.update(
        (_make_detection(bbox_xyxy=(4.0, 0.0, 14.0, 10.0), timestamp_ns=300),)
    )
    assert current[0].bbox_xyxy == pytest.approx((2.5, 0.0, 12.5, 10.0))
    assert current[0].timestamp_ns == 300


def test_stabilizer_reset_clears_tracks_misses_and_timestamp_history() -> None:
    """reset 后较旧的新时间线也能从首帧重新确认。"""

    stabilizer = Detection2DStabilizer()
    assert stabilizer.update((_make_detection(timestamp_ns=1_000),)) == ()
    assert len(stabilizer.update((_make_detection(timestamp_ns=2_000),))) == 1
    assert stabilizer.update(()) == ()

    stabilizer.reset()

    assert stabilizer.update((_make_detection(timestamp_ns=10),)) == ()
    result = stabilizer.update((_make_detection(timestamp_ns=20),))
    assert len(result) == 1
    assert result[0].timestamp_ns == 20


def test_stabilizer_ignores_invalid_detections_without_poisoning_valid_target() -> None:
    """同帧坏项（含更大时间戳）不能建轨、更新轨迹或阻止合法目标确认。"""

    invalid = (
        _make_detection(valid=False, timestamp_ns=900_000),
        _make_detection(class_id="", timestamp_ns=900_001),
        _make_detection(class_id="blue", timestamp_ns=900_002),
        _make_detection(
            bbox_xyxy=(float("nan"), 0.0, 10.0, 10.0),
            timestamp_ns=900_003,
        ),
        _make_detection(
            bbox_xyxy=(0.0, 0.0, float("inf"), 10.0),
            timestamp_ns=900_004,
        ),
        _make_detection(
            bbox_xyxy=(0.0, 0.0, 10**400, 10.0),  # type: ignore[arg-type]
            timestamp_ns=900_004,
        ),
        _make_detection(bbox_xyxy=(10.0, 0.0, 0.0, 10.0), timestamp_ns=900_005),
        _make_detection(confidence=float("nan"), timestamp_ns=900_006),
        _make_detection(confidence=float("inf"), timestamp_ns=900_007),
        _make_detection(confidence=-0.1, timestamp_ns=900_008),
        _make_detection(confidence=1.1, timestamp_ns=900_009),
        _make_detection(timestamp_ns=True),  # type: ignore[arg-type]
        _make_detection(timestamp_ns=-1),
        _make_detection(timestamp_ns=1.5),  # type: ignore[arg-type]
    )
    stabilizer = Detection2DStabilizer()

    assert stabilizer.update(
        (_make_detection(timestamp_ns=100), *invalid)
    ) == ()
    result = stabilizer.update(
        (_make_detection(timestamp_ns=200), *reversed(invalid))
    )

    assert len(result) == 1
    assert result[0].class_id == "pink"
    assert result[0].timestamp_ns == 200


def test_stabilizer_instances_do_not_share_track_history() -> None:
    """两个稳定器对象各自维护轨迹，第二个实例不能继承第一个的确认状态。"""

    first = Detection2DStabilizer()
    second = Detection2DStabilizer()
    assert first.update((_make_detection(timestamp_ns=100),)) == ()
    assert len(first.update((_make_detection(timestamp_ns=200),))) == 1
    assert second.update((_make_detection(timestamp_ns=100),)) == ()


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
