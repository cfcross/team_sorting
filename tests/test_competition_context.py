import json
from types import SimpleNamespace

import pytest

from team_sorting.competition_context import (
    CompetitionContext,
    CompetitionRunCoordinator,
    RefereeProgressParser,
    task_set_fingerprint,
)
from team_sorting.fsm import InstructionParser
from team_sorting.recorder import EpisodeRecorder
from team_sorting.ros_nodes import (
    bind_static_camera_calibration,
    normalize_ros_frame_id,
    StaticCameraCalibrationCache,
    static_camera_calibration_from_message,
)


OFFICIAL_TASKS = [
    {"task": 1, "instruction": "抓取桌面左侧的粉色方块，放到货架空层",
     "target_kind": "cuboid_box", "target_body": "box_pink", "target_color": "pink",
     "place_type": "shelf_point", "place_world": [-2.68, 0.778, 1.156], "place_radius": 0.24},
    {"task": 2, "instruction": "抓取货架中的褐色方块，放到第一个方块原来在桌子上的位置",
     "target_kind": "cuboid_box", "target_body": "box_brown", "target_color": "brown",
     "place_type": "table_point", "place_world": [-1.0, 2.2, 0.834], "place_radius": 0.28},
    {"task": 3, "instruction": "抓取白色正方体顶部的黄色方块，放到货架中白色长方体的左边",
     "target_kind": "cuboid_box", "target_body": "box_yellow", "target_color": "yellow",
     "ref_prop": "packaging_box", "ref_prop_body": "prop_packaging_box", "direction": "left",
     "place_type": "shelf_prop_side", "place_world": [-2.68, 0.54, 0.498], "place_radius": 0.24},
]
RAW_TASKS = json.dumps(OFFICIAL_TASKS, ensure_ascii=False)


def tasks(timestamp=10):
    return InstructionParser().parse(RAW_TASKS, timestamp)


def feed(parser, task=1, attempt=0, step="-", elapsed="0.0", score=0, ts=20):
    parser.update_taskinfo(f"任务{task}: {OFFICIAL_TASKS[task - 1]['instruction']}", ts)
    parser.update_gameinfo(
        f"t={elapsed}s score={score} task={task}/3 best=[0, 0, 0] attempt={attempt} step={step}", ts
    )
    parser.update_score(score, ts)


def test_complete_official_instruction_and_fingerprint_ignore_receive_time():
    first, second = tasks(10), tasks(99)
    assert len(first) == 3
    assert [task.place_type for task in first] == ["shelf_point", "table_point", "shelf_prop_side"]
    assert [task.place_radius for task in first] == [0.24, 0.28, 0.24]
    assert task_set_fingerprint(first) == task_set_fingerprint(second)


@pytest.mark.parametrize("task", [1, 2, 3])
@pytest.mark.parametrize("step", ["-", "nav", "touch", "lift", "place"])
def test_official_taskinfo_gameinfo_values(task, step):
    parser = RefereeProgressParser()
    feed(parser, task=task, step=step)
    progress = parser.progress()
    assert progress.valid and progress.current_task_id == task and progress.current_step == step


def test_finished_and_timeout_are_terminal():
    parser = RefereeProgressParser()
    feed(parser, task=3, elapsed="599.9")
    parser.update_taskinfo("全部任务结束", 30)
    assert parser.progress().finished
    timeout = RefereeProgressParser()
    feed(timeout, task=2, elapsed="600.0")
    assert timeout.progress().finished


def test_mismatch_malformed_score_and_range_fail_closed():
    parser = RefereeProgressParser()
    feed(parser, task=1)
    parser.update_taskinfo(f"任务2: {OFFICIAL_TASKS[1]['instruction']}", 30)
    assert parser.progress().failure_reason == "taskinfo_gameinfo_task_mismatch"
    parser.update_gameinfo("bad", 31)
    assert parser.progress().failure_reason == "malformed_gameinfo"

    score = RefereeProgressParser()
    feed(score, task=1)
    score.update_score(1, 30)
    assert score.progress().failure_reason == "referee_score_mismatch"

    ranged = RefereeProgressParser()
    ranged.update_gameinfo("t=0.0s score=0 task=1/3 best=[0, 0, 0] attempt=4 step=-", 1)
    assert ranged.progress().failure_reason == "gameinfo_out_of_range"


@pytest.mark.parametrize("raw", ["", "t=NaNs score=0 task=1/3 best=[0,0,0] attempt=0 step=-",
                                  "t=0.0s score=-1 task=1/3 best=[0,0,0] attempt=0 step=-",
                                  "t=0.0s score=0 task=0/3 best=[0,0,0] attempt=0 step=-"])
def test_malformed_gameinfo_never_raises(raw):
    parser = RefereeProgressParser()
    assert not parser.update_gameinfo(raw, 1)
    assert not parser.progress().valid


def test_stale_and_regressing_messages_do_not_roll_context_back():
    parser = RefereeProgressParser()
    feed(parser, task=2, attempt=2, ts=20)
    parser.update_gameinfo("t=1.0s score=0 task=1/3 best=[0, 0, 0] attempt=0 step=-", 21)
    parser.update_gameinfo("t=1.0s score=0 task=2/3 best=[0, 0, 0] attempt=1 step=-", 22)
    parser.update_taskinfo(f"任务1: {OFFICIAL_TASKS[0]['instruction']}", 10)
    progress = parser.progress()
    assert progress.current_task_id == 2 and progress.current_attempt_count == 2


def test_coordinator_repeated_broadcast_is_same_run_and_switches_only_by_referee():
    coordinator = CompetitionRunCoordinator()
    assert coordinator.update_tasks(tasks(10), 10)
    run_id = coordinator.context().run_id
    assert not coordinator.update_tasks(tasks(20), 20)
    assert coordinator.context().run_id == run_id
    feed(coordinator.referee, task=2)
    context = coordinator.context()
    assert context.valid and context.active_task.task_id == 2
    assert context.current_attempt_count == 0
    assert coordinator.active_task_changed(context)
    assert not coordinator.active_task_changed(context)


def test_coordinator_activation_identity_includes_settled_attempt_count():
    coordinator = CompetitionRunCoordinator()
    coordinator.update_tasks(tasks(), 10)
    run_id = coordinator.context().run_id

    for attempt in (0, 1, 2):
        feed(coordinator.referee, task=1, attempt=attempt, ts=20 + attempt)
        context = coordinator.context()
        assert context.valid and context.active_task.task_id == 1
        assert context.run_id == run_id
        assert context.current_attempt_count == attempt
        assert coordinator.active_task_changed(context)
        assert not coordinator.active_task_changed(context)


def test_new_task_set_creates_run_and_drops_old_referee_progress():
    coordinator = CompetitionRunCoordinator()
    coordinator.update_tasks(tasks(), 10)
    feed(coordinator.referee, task=1)
    old_run = coordinator.context().run_id
    changed = json.loads(RAW_TASKS)
    changed[0]["target_color"] = "magenta"
    coordinator.update_tasks(InstructionParser().parse(json.dumps(changed), 30), 30)
    context = coordinator.context()
    assert context.run_id != old_run
    assert not context.valid and "missing_referee_topics" in context.failure_reason


def test_context_stable_json_roundtrip_and_strict_numbers():
    coordinator = CompetitionRunCoordinator()
    coordinator.update_tasks(tasks(), 10)
    feed(coordinator.referee, task=3, attempt=1, step="nav")
    context = coordinator.context()
    rendered = context.to_json()
    assert CompetitionContext.from_json(rendered) == context
    assert rendered == CompetitionContext.from_json(rendered).to_json()
    payload = json.loads(rendered)
    payload["score"] = True
    with pytest.raises(ValueError):
        CompetitionContext.from_json(json.dumps(payload))


@pytest.mark.parametrize("raw,expected", [("/odom", "odom"), ("odom", "odom"), ("///odom", "odom"),
                                            ("world", "world"), ("map", "map")])
def test_frame_normalization(raw, expected):
    assert normalize_ros_frame_id(raw) == expected


@pytest.mark.parametrize("raw", ["/", "", 1, None])
def test_frame_normalization_rejects_empty_or_non_string(raw):
    with pytest.raises(ValueError):
        normalize_ros_frame_id(raw)


def _image(frame="head_camera", sec=12, width=640, height=480):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame, stamp=SimpleNamespace(sec=sec, nanosec=34)),
        width=width, height=height,
    )


def test_official_headerless_camera_info_binds_to_current_images():
    info = SimpleNamespace(
        k=[500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
        width=640, height=480,
        header=SimpleNamespace(frame_id="", stamp=SimpleNamespace(sec=0, nanosec=0)),
    )
    calibration = static_camera_calibration_from_message(info)
    intrinsics = bind_static_camera_calibration(calibration, _image(), _image())
    assert intrinsics.frame_id == "head_camera"
    assert intrinsics.timestamp_ns == 12_000_000_034


def test_camera_binding_rejects_frame_or_size_conflict():
    calibration = ((500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0), 640, 480)
    with pytest.raises(ValueError, match="frame冲突"):
        bind_static_camera_calibration(calibration, _image(), _image("other_camera"))
    with pytest.raises(ValueError, match="尺寸"):
        bind_static_camera_calibration(calibration, _image(), _image(width=320))


def test_camera_calibration_runtime_change_latches_fail_closed():
    first = SimpleNamespace(
        k=[500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
        width=640, height=480,
    )
    changed = SimpleNamespace(k=list(first.k), width=320, height=480)
    cache = StaticCameraCalibrationCache()
    cache.update(first)
    cache.update(first)
    with pytest.raises(ValueError, match="突变"):
        cache.update(changed)
    assert cache.value is None
    with pytest.raises(ValueError, match="锁定"):
        cache.update(changed)


def test_recorder_keeps_one_raw_run_and_indexes_context_transitions(tmp_path):
    coordinator = CompetitionRunCoordinator()
    coordinator.update_tasks(tasks(), 10)
    feed(coordinator.referee, task=1, attempt=0)
    recorder = EpisodeRecorder(tmp_path)
    run_dir = recorder.start("raw_run", 1, "raw Run, not training Episode")
    first = coordinator.context()
    recorder.record_competition_context(first)
    recorder.record_competition_context(first)
    feed(coordinator.referee, task=1, attempt=1, ts=30)
    recorder.record_competition_context(coordinator.context())
    assert recorder.episode_dir == run_dir
    assert recorder.metadata.competition_context_count == 3
    assert [item["settled_attempt_count"] for item in recorder.metadata.competition_context_index] == [0, 1]
    assert len((run_dir / "competition_contexts.jsonl").read_text().splitlines()) == 3
