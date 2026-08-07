"""Global official-publish gate and TeamClientNode lifecycle regressions."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import team_sorting.ros_nodes as ros_nodes_module
from team_sorting.action_mux import ActionMux
from team_sorting.competition_context import CompetitionContext
from team_sorting.external_candidate import ExternalCandidateDecision
from team_sorting.fsm import FSMEvent
from team_sorting.interfaces import (
    ArmExecutionConfig,
    BaseCommand,
    BaseState,
    FSMStatus,
    GlobalPhase,
    LocalPhase,
    ManipulationCommand,
    ObjectEstimate3D,
    RobotJointState,
    SensorSnapshot,
    TaskSpec,
)
from team_sorting.ros_nodes import (
    _base_planar_transform_snapshot,
    _create_team_client_node,
    _internal_fsm_publish_authorization,
    _select_target_estimate,
    _spin,
)


NOW = 10_000_000_000
OFFICIAL_TOPICS = {
    "/cmd_vel",
    "/spine_forward_position_controller/commands",
    "/head_forward_position_controller/commands",
    "/left_arm_forward_position_controller/commands",
    "/right_arm_forward_position_controller/commands",
}
TEAM_TELEMETRY_TOPICS = {
    "/team/action_dispatch",
    "/team/competition_context",
    "/team/final_action",
    "/team/fsm_status",
}
OFFICIAL_TASKS = [
    {"task": 1, "instruction": "task one", "target_kind": "cuboid_box",
     "target_body": "box_1", "target_color": "pink", "place_type": "shelf_point",
     "place_world": [1.0, 2.0, 3.0], "place_radius": 0.24},
    {"task": 2, "instruction": "task two", "target_kind": "cuboid_box",
     "target_body": "box_2", "target_color": "brown", "place_type": "table_point",
     "place_world": [2.0, 2.0, 3.0], "place_radius": 0.28},
    {"task": 3, "instruction": "task three", "target_kind": "cuboid_box",
     "target_body": "box_3", "target_color": "yellow", "place_type": "shelf_prop_side",
     "place_world": [3.0, 2.0, 3.0], "place_radius": 0.24,
     "ref_prop": "packaging_box", "ref_prop_body": "prop", "direction": "left"},
]


class _Context:
    def __init__(self, events: list[str]) -> None:
        self.active = True
        self.events = events

    def ok(self) -> bool:
        return self.active


class _Logger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))


class _Message:
    def __init__(self) -> None:
        self.data: Any = ""


class _Twist:
    def __init__(self) -> None:
        self.linear = SimpleNamespace(x=0.0)
        self.angular = SimpleNamespace(z=0.0)


class _Publisher:
    def __init__(self, topic: str, context: _Context, events: list[str]) -> None:
        self.topic = topic
        self.context = context
        self.events = events
        self.messages: list[Any] = []
        self.fail = False

    def publish(self, message: Any) -> None:
        self.events.append(f"publish:{self.topic}")
        if not self.context.ok():
            raise RuntimeError("publisher context is invalid")
        if self.fail:
            raise RuntimeError("injected publish failure")
        self.messages.append(message)


class _Timer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.cancel_count = 0

    def cancel(self) -> None:
        self.cancel_count += 1
        self.events.append("timer_cancel")


class _Node:
    def __init__(self, _name: str) -> None:
        self.events: list[str] = []
        self.context = _Context(self.events)
        self.logger = _Logger()
        self.publishers: dict[str, _Publisher] = {}
        self.subscriptions: list[str] = []
        self.timers: list[_Timer] = []
        self.destroy_count = 0
        self.additional_publishers: dict[str, int] = {}
        self.now_ns = NOW

    def create_publisher(self, _type: object, topic: str, _depth: int) -> _Publisher:
        publisher = _Publisher(topic, self.context, self.events)
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(
        self, _type: object, topic: str, _callback: object, _depth: int
    ) -> object:
        self.subscriptions.append(topic)
        return object()

    def create_timer(self, _period: float, _callback: object) -> _Timer:
        timer = _Timer(self.events)
        self.timers.append(timer)
        return timer

    def get_logger(self) -> _Logger:
        return self.logger

    def get_clock(self) -> SimpleNamespace:
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=self.now_ns))

    def count_publishers(self, topic: str) -> int:
        return int(topic in self.publishers) + self.additional_publishers.get(topic, 0)

    def destroy_node(self) -> str:
        self.destroy_count += 1
        self.events.append("super_destroy")
        return "destroyed"


def _ros() -> SimpleNamespace:
    return SimpleNamespace(
        Node=_Node,
        String=_Message,
        Twist=_Twist,
        Float64MultiArray=_Message,
        Odometry=object,
        JointState=object,
        Detection3DArray=object,
    )


def _config(
    head_tracking_overrides: dict[str, Any] | None = None,
    **control_overrides: bool,
) -> dict[str, Any]:
    config = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    config["control"].update(control_overrides)
    if head_tracking_overrides:
        config["control"]["head_target_tracking"].update(head_tracking_overrides)
    return config


def _node(
    head_tracking_overrides: dict[str, Any] | None = None,
    **control_overrides: bool,
) -> Any:
    ros = _ros()
    return _create_team_client_node(ros)(
        _config(head_tracking_overrides, **control_overrides), ros
    )


def _tracking_enabled() -> dict[str, bool]:
    return {"enabled": True, "fresh_reset_confirmed": True}


def _message(data: Any) -> _Message:
    message = _Message()
    message.data = data
    return message


def _feed_official_context(
    node: Any, *, task: int, attempt: int, score: int = 0
) -> None:
    node._on_referee_taskinfo(
        _message(f"任务{task}: {OFFICIAL_TASKS[task - 1]['instruction']}")
    )
    node._on_referee_gameinfo(
        _message(
            f"t=1.0s score={score} task={task}/3 best=[0, 0, 0] "
            f"attempt={attempt} step=-"
        )
    )
    node._on_referee_score(_message(score))


def _joints() -> RobotJointState:
    return RobotJointState(
        position=(0.0,) * 17,
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=NOW,
    )


def _base() -> BaseState:
    return BaseState(
        position_xyz=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        yaw=0.0,
        linear_velocity_xyz=(0.0, 0.0, 0.0),
        angular_velocity_xyz=(0.0, 0.0, 0.0),
        frame_id="odom",
        timestamp_ns=NOW,
    )


def _task() -> TaskSpec:
    return TaskSpec(
        task_id=1,
        instruction="task one",
        target_kind="cuboid_box",
        target_body="box_1",
        target_color="pink",
        place_type="shelf_point",
        place_world_xyz=(1.0, 2.0, 3.0),
        place_frame_id="world",
        place_radius=0.24,
        timestamp_ns=NOW - 100,
    )


def _estimate(
    object_id: str,
    confidence: float,
    *,
    timestamp_ns: int = NOW - 10,
    orientation: bool = True,
    size: bool = True,
) -> ObjectEstimate3D:
    return ObjectEstimate3D(
        class_id="pink",
        position_xyz=(1.0, 0.0, 0.5),
        confidence=confidence,
        frame_id="odom",
        timestamp_ns=timestamp_ns,
        object_id=object_id,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0) if orientation else None,
        size_xyz_m=(0.2, 0.1, 0.1) if size else None,
    )


def _execution_config_values() -> dict[str, object]:
    return asdict(
        ArmExecutionConfig(
            feedback_max_age_ns=100,
            trajectory_max_age_ns=100,
            command_ttl_ns=100,
            max_control_period_ns=100,
            waypoint_timeout_margin_ns=100,
            total_timeout_margin_ns=100,
            max_slide_velocity_m_s=0.1,
            max_arm_velocity_rad_s=0.1,
            max_gripper_velocity_per_s=0.1,
            slide_tolerance_m=0.01,
            arm_tolerance_rad=0.01,
            gripper_tolerance=0.01,
            settle_cycles=1,
            initial_slide_error_limit_m=0.1,
            initial_arm_error_limit_rad=0.1,
            initial_gripper_error_limit=0.1,
        )
    )


def _prime_control_state(
    node: Any, joints: RobotJointState | None = None
) -> None:
    node._base_cache.put(NOW, _base())
    node._joint_cache.put(NOW, joints or _joints())


def _set_active_fsm(node: Any) -> None:
    status = FSMStatus(
        1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW
    )
    node._fsm.status = lambda _now: status


def _external_command(
    controlled_indices: tuple[int, ...] = (1,),
) -> ManipulationCommand:
    target = list(_joints().position)
    for index in controlled_indices:
        target[index] = 0.001
    return ManipulationCommand(
        joint_target=tuple(target),
        controlled_mask=tuple(index in controlled_indices for index in range(17)),
        local_phase=LocalPhase.IDLE,
        timestamp_ns=NOW,
        valid_until_ns=NOW + 100_000_000,
    )


def _official_message_count(node: Any) -> int:
    return sum(len(node.publishers[topic].messages) for topic in OFFICIAL_TOPICS)


def test_default_config_is_observe_only_and_creates_no_official_publishers() -> None:
    config = _config()
    assert {
        key: config["control"][key]
        for key in ("observe_only", "enable_official_publish", "simulation_only")
    } == {
        "observe_only": True,
        "enable_official_publish": False,
        "simulation_only": True,
    }
    assert config["control"]["head_target_tracking"] == {
        "enabled": False,
        "require_fresh_reset": True,
        "fresh_reset_confirmed": False,
        "initial_yaw_target": 0.0,
        "initial_pitch_target": 0.0,
        "require_exclusive_writer": True,
    }

    node = _node()

    assert node._official_publisher is None
    assert set(node.publishers) == TEAM_TELEMETRY_TOPICS
    assert not OFFICIAL_TOPICS.intersection(node.publishers)
    assert any("official_publish_disabled" in message for _, message in node.logger.messages)


def test_settled_attempts_rearm_only_the_local_single_task_fsm() -> None:
    node = _node()
    node._system_ready_submitted = True
    node._on_instruction(_message(json.dumps(OFFICIAL_TASKS)))

    _feed_official_context(node, task=1, attempt=0)
    run_id = node._active_context.run_id
    first_fsm = node._fsm
    assert node._fsm.task.task_id == 1

    _feed_official_context(node, task=1, attempt=0)
    assert node._fsm is first_fsm

    _feed_official_context(node, task=1, attempt=1)
    second_fsm = node._fsm
    assert second_fsm is not first_fsm
    assert node._fsm.task.task_id == 1
    assert node._active_context.run_id == run_id

    _feed_official_context(node, task=1, attempt=2)
    assert node._fsm is not second_fsm
    assert node._fsm.task.task_id == 1
    assert node._active_context.run_id == run_id

    logs = [message for _level, message in node.logger.messages]
    assert sum("task_transition" in message for message in logs) == 1
    assert sum("attempt_transition" in message for message in logs) == 2
    assert any("不代表Server、机器人或物品复位" in message for message in logs)
    assert not any("Server reset" in message for message in logs)


def test_repeated_official_instruction_refreshes_liveness_without_rearming_fsm() -> None:
    ros = _ros()
    config = _config()
    config["external_candidate"]["enabled"] = True
    node = _create_team_client_node(ros)(config, ros)
    node._system_ready_submitted = True
    raw_instruction = json.dumps(OFFICIAL_TASKS)
    node._on_instruction(_message(raw_instruction))
    _feed_official_context(node, task=1, attempt=0)

    run_id = node._active_context.run_id
    task_set_fingerprint = node._active_context.task_set_fingerprint
    task_identity = node._external_candidate.current_task_identity
    generation = node._external_candidate.bound_generation
    first_fsm = node._fsm

    for elapsed_ms in (500, 1000, 1500, 2000):
        node.now_ns = NOW + elapsed_ms * 1_000_000
        node._on_instruction(_message(raw_instruction))
        assert node._external_candidate.instruction_received_ns == node.now_ns
        assert node._external_candidate.watchdog(node.now_ns) is None

    assert node._active_context.run_id == run_id
    assert node._active_context.task_set_fingerprint == task_set_fingerprint
    assert node._external_candidate.current_task_identity == task_identity
    assert node._external_candidate.bound_generation == generation
    assert node._fsm is first_fsm
    logs = [message for _level, message in node.logger.messages]
    assert sum("新的本地run身份" in message for message in logs) == 1
    assert sum("task_transition" in message for message in logs) == 1
    assert not any("attempt_transition" in message for message in logs)


def test_official_publish_order_recovers_once_without_task1_fallback() -> None:
    node = _node()
    node._system_ready_submitted = True
    node._on_instruction(_message(json.dumps(OFFICIAL_TASKS)))
    _feed_official_context(node, task=1, attempt=0)
    task1_fsm = node._fsm

    node._on_referee_taskinfo(_message("任务2: task two"))
    after_taskinfo = json.loads(node.publishers["/team/competition_context"].messages[-1].data)
    assert not after_taskinfo["valid"]
    assert node._fsm is task1_fsm

    node._on_referee_gameinfo(
        _message("t=2.0s score=10 task=2/3 best=[0, 0, 0] attempt=0 step=-")
    )
    after_gameinfo = json.loads(node.publishers["/team/competition_context"].messages[-1].data)
    assert not after_gameinfo["valid"]
    assert node._fsm is task1_fsm

    node._on_referee_score(_message(10))
    task2_fsm = node._fsm
    recovered = node._active_context
    assert recovered.valid and recovered.active_task.task_id == 2
    assert task2_fsm is not task1_fsm

    _feed_official_context(node, task=2, attempt=0, score=10)
    assert node._fsm is task2_fsm
    logs = [message for _level, message in node.logger.messages]
    assert sum("task_transition" in message for message in logs) == 2


def test_malformed_referee_message_then_complete_topics_recovers_once() -> None:
    node = _node()
    node._system_ready_submitted = True
    node._on_instruction(_message(json.dumps(OFFICIAL_TASKS)))
    node._on_referee_gameinfo(_message("malformed"))
    assert not node._active_context.valid

    _feed_official_context(node, task=1, attempt=0)
    recovered_fsm = node._fsm
    assert node._active_context.valid and node._fsm.task.task_id == 1
    _feed_official_context(node, task=1, attempt=0)
    assert node._fsm is recovered_fsm


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"observe_only": True, "enable_official_publish": True}, False),
        ({"observe_only": False, "enable_official_publish": False}, False),
        ({"observe_only": False, "enable_official_publish": True}, True),
    ],
)
def test_only_explicit_simulation_actuation_combination_creates_official_publisher(
    overrides: dict[str, bool], expected: bool
) -> None:
    node = _node(**overrides)

    assert (node._official_publisher is not None) is expected
    assert OFFICIAL_TOPICS.issubset(node.publishers) is expected


def test_simulation_only_false_is_rejected_before_publishers_are_created() -> None:
    ros = _ros()
    with pytest.raises(RuntimeError, match="simulation_only=false"):
        _create_team_client_node(ros)(
            _config(
                observe_only=False,
                enable_official_publish=True,
                simulation_only=False,
            ),
            ros,
        )


@pytest.mark.parametrize("value", [None, "/team/wrong_dispatch"])
def test_action_dispatch_topic_is_strictly_validated_before_node_publishers(
    value: str | None,
) -> None:
    ros = _ros()
    config = _config(observe_only=False, enable_official_publish=True)
    if value is None:
        config["topics"].pop("action_dispatch")
    else:
        config["topics"]["action_dispatch"] = value
    with pytest.raises(RuntimeError, match="action_dispatch"):
        _create_team_client_node(ros)(config, ros)


def test_default_candidate_computation_has_no_active_manipulation_hold() -> None:
    node = _node()
    assert node._arm_execution is None
    snapshot = SensorSnapshot(None, _base(), _joints(), (), NOW, True)

    base_command, manipulation_command = node._compute_candidate_commands(
        snapshot, node._fsm.status(NOW), NOW
    )

    assert base_command is not None
    assert (base_command.v, base_command.w) == (0.0, 0.0)
    assert manipulation_command is None


def test_default_control_tick_publishes_only_team_diagnostic_telemetry() -> None:
    node = _node()
    _prime_control_state(node)
    assert node._arm_execution is None

    node._control_tick()

    assert len(node.publishers["/team/fsm_status"].messages) == 1
    assert len(node.publishers["/team/final_action"].messages) == 1
    assert len(node.publishers["/team/action_dispatch"].messages) == 1
    assert not OFFICIAL_TOPICS.intersection(node.publishers)
    payload = json.loads(node.publishers["/team/final_action"].messages[0].data)
    assert payload["action"] == [0.0] * 19
    dispatch = json.loads(node.publishers["/team/action_dispatch"].messages[0].data)
    assert dispatch["calculated"] is True
    assert dispatch["publish_enabled"] is False
    assert dispatch["publisher_created"] is False
    assert dispatch["publish_attempted"] is False
    assert dispatch["dispatch_mode"] == "none"
    assert dispatch["dispatched_action"] == [None] * 19


def test_kdl_is_constructed_once_but_never_imported_when_planning_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"construct": 0, "self_check": 0}

    class LazyKDL:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            calls["construct"] += 1

        def self_check(self) -> None:
            calls["self_check"] += 1
            pytest.fail("disabled arm planning must not import/check official KDL")

    monkeypatch.setattr(ros_nodes_module, "OfficialKDLAdapter", LazyKDL)
    node = _node()
    _prime_control_state(node)

    node._control_tick()
    node._control_tick()

    assert calls == {"construct": 1, "self_check": 0}
    assert node._arm_planner is None
    assert "disabled" in node._arm_planning_unavailable_reason


def test_enabled_kdl_self_check_failure_keeps_planner_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"self_check": 0}

    class ReadyPlanningConfig:
        def validate_for_grasp(self) -> None:
            return None

        def validate_for_place(self) -> None:
            return None

    class FailingKDL:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def self_check(self) -> None:
            calls["self_check"] += 1
            raise RuntimeError("official KDL unavailable")

    monkeypatch.setattr(
        ros_nodes_module,
        "_arm_planning_config_from_config",
        lambda _config: (True, ReadyPlanningConfig()),
    )
    monkeypatch.setattr(ros_nodes_module, "OfficialKDLAdapter", FailingKDL)

    node = _node()

    assert calls["self_check"] == 1
    assert node._arm_planner is None
    assert "official KDL unavailable" in node._arm_planning_unavailable_reason


def test_enabled_but_unconfigured_planning_never_imports_kdl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class KDLWithoutImport:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def self_check(self) -> None:
            pytest.fail("incomplete ArmPlanningConfig must fail before KDL self_check")

    monkeypatch.setattr(ros_nodes_module, "OfficialKDLAdapter", KDLWithoutImport)
    config = _config()
    config["arm_planning"]["enabled"] = True
    ros = _ros()

    node = _create_team_client_node(ros)(config, ros)

    assert node._arm_planner is None
    assert "缺少 ArmPlanningConfig" in node._arm_planning_unavailable_reason


def test_missing_arm_execution_config_is_explicitly_fail_closed() -> None:
    node = _node()

    assert node._arm_execution is None
    assert "ArmExecutionConfig" in node._arm_execution_unavailable_reason


def test_complete_arm_execution_config_constructs_controller_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"construct": 0, "reset": 0}

    class ExecutionSpy:
        def __init__(self, config: ArmExecutionConfig) -> None:
            assert isinstance(config, ArmExecutionConfig)
            calls["construct"] += 1

        def reset(self) -> None:
            calls["reset"] += 1

    monkeypatch.setattr(ros_nodes_module, "ArmExecutionController", ExecutionSpy)
    config = _config()
    config["arm_execution"] = _execution_config_values()
    ros = _ros()
    node = _create_team_client_node(ros)(config, ros)
    _prime_control_state(node)

    node._control_tick()
    node._control_tick()

    assert calls["construct"] == 1
    assert node._arm_execution is not None


def _dirty_runtime_wiring(node: Any) -> None:
    state = node._runtime_wiring
    state.active_target = _estimate("old", 0.8)
    state.active_nav_goal = object()
    state.planned_grasp_context = object()
    state.confirmed_grasp_context = object()
    state.active_pick_trajectory = object()
    state.active_place_trajectory = object()
    state.active_trajectory_id = "trajectory-old"
    state.pick_observation_before_lift = _estimate("old", 0.8)
    state.latest_grasp_verification = object()
    state.latest_place_verification_observation = _estimate("old", 0.8)
    state.last_navigation_status = object()
    state.last_manipulation_status = object()
    state.phase_event_keys.add(("old",))
    state.phase_entry_failure_reason = "old failure"


def _assert_runtime_payload_cleared(node: Any) -> None:
    state = node._runtime_wiring
    assert state.active_target is None
    assert state.active_nav_goal is None
    assert state.planned_grasp_context is None
    assert state.confirmed_grasp_context is None
    assert state.active_pick_trajectory is None
    assert state.active_place_trajectory is None
    assert state.active_trajectory_id == ""
    assert state.pick_observation_before_lift is None
    assert state.latest_grasp_verification is None
    assert state.latest_place_verification_observation is None
    assert state.last_navigation_status is None
    assert state.last_manipulation_status is None
    assert state.phase_event_keys == set()
    assert state.phase_entry_failure_reason == ""


def test_task_and_attempt_transitions_clear_all_runtime_wiring() -> None:
    node = _node()
    node._system_ready_submitted = True
    node._on_instruction(_message(json.dumps(OFFICIAL_TASKS)))
    _feed_official_context(node, task=1, attempt=0)

    _dirty_runtime_wiring(node)
    _feed_official_context(node, task=1, attempt=1)
    _assert_runtime_payload_cleared(node)
    assert node._runtime_wiring.task_id == 1
    assert node._runtime_wiring.attempt == 1

    _dirty_runtime_wiring(node)
    _feed_official_context(node, task=2, attempt=0)
    _assert_runtime_payload_cleared(node)
    assert node._runtime_wiring.task_id == 2
    assert node._runtime_wiring.attempt == 0


def test_new_run_clears_all_runtime_wiring() -> None:
    node = _node()
    node._system_ready_submitted = True
    node._on_instruction(_message(json.dumps(OFFICIAL_TASKS)))
    _feed_official_context(node, task=1, attempt=0)
    previous_run_id = node._runtime_wiring.run_id
    _dirty_runtime_wiring(node)

    changed_tasks = [dict(task) for task in OFFICIAL_TASKS]
    changed_tasks[0]["instruction"] = "task one changed"
    node._on_instruction(_message(json.dumps(changed_tasks)))

    assert node._runtime_wiring.run_id
    assert node._runtime_wiring.run_id != previous_run_id
    _assert_runtime_payload_cleared(node)


def test_finished_context_clears_runtime_and_stops_normal_phase_logic() -> None:
    node = _node()
    node._system_ready_submitted = True
    node._on_instruction(_message(json.dumps(OFFICIAL_TASKS)))
    _feed_official_context(node, task=1, attempt=0)
    _dirty_runtime_wiring(node)

    node._on_referee_taskinfo(_message("全部任务结束"))

    assert node._active_context.finished
    assert node._runtime_wiring.context_finished
    _assert_runtime_payload_cleared(node)
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)
    status = FSMStatus(1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW)
    assert node._run_current_phase(
        snapshot, status, node._active_context, NOW
    ) == (None, None, None)


def test_phase_entry_runs_once_per_phase_and_rearms_on_change() -> None:
    node = _node()
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)
    search = FSMStatus(1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW)

    assert node._handle_phase_entry(snapshot, search, None, NOW)
    generation = node._runtime_wiring.phase_generation
    assert not node._handle_phase_entry(snapshot, search, None, NOW + 1)
    assert node._runtime_wiring.phase_generation == generation

    refine = FSMStatus(1, GlobalPhase.REFINE_TARGET, LocalPhase.IDLE, 0, False, "", NOW + 2)
    assert node._handle_phase_entry(snapshot, refine, None, NOW + 2)
    assert node._runtime_wiring.phase_generation == generation + 1


def test_event_submission_is_bounded_and_deduplicated() -> None:
    node = _node()
    calls: list[FSMEvent] = []
    node._fsm.handle_event = lambda event, _now, _reason="": calls.append(event) or True
    status = FSMStatus(1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW)

    assert node._submit_fsm_event_once(
        FSMEvent.TARGET_FOUND, status, NOW, target_identity="target-a"
    )
    assert not node._submit_fsm_event_once(
        FSMEvent.FAILURE, status, NOW, target_identity="target-a"
    )
    assert not node._submit_fsm_event_once(
        FSMEvent.TARGET_FOUND, status, NOW + 1, target_identity="target-a"
    )
    assert calls == [FSMEvent.TARGET_FOUND]


def test_rejected_fsm_event_is_not_reported_as_transition() -> None:
    node = _node()
    calls = {"count": 0}

    def reject(*_args: object, **_kwargs: object) -> bool:
        calls["count"] += 1
        return False

    node._fsm.handle_event = reject
    status = FSMStatus(1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW)

    assert not node._submit_fsm_event_once(
        FSMEvent.TARGET_FOUND, status, NOW, target_identity="target-a"
    )
    assert calls["count"] == 1
    assert "拒绝" in node._runtime_wiring.phase_entry_failure_reason


def test_reset_event_clears_runtime_identity_and_payload() -> None:
    node = _node()
    node._runtime_wiring.run_id = "run-old"
    node._runtime_wiring.task_id = 1
    node._runtime_wiring.attempt = 2
    _dirty_runtime_wiring(node)
    status = node._fsm.status(NOW)

    assert node._submit_fsm_event_once(FSMEvent.RESET, status, NOW)

    _assert_runtime_payload_cleared(node)
    assert node._runtime_wiring.run_id == ""
    assert node._runtime_wiring.task_id is None
    assert node._runtime_wiring.attempt is None


def test_target_selection_is_deterministic_and_refine_requires_new_geometry() -> None:
    task = _task()
    tied_b = _estimate("b", 0.9)
    tied_a = _estimate("a", 0.9)

    assert _select_target_estimate(
        (tied_b, tied_a), task, NOW, 1_000
    ).object_id == "a"

    old_preferred = _estimate("preferred", 1.0, timestamp_ns=NOW - 100)
    new_other = _estimate("other", 0.99, timestamp_ns=NOW - 10)
    new_preferred = _estimate("preferred", 0.7, timestamp_ns=NOW - 5)
    selected = _select_target_estimate(
        (old_preferred, new_other, new_preferred),
        task,
        NOW,
        1_000,
        phase_entered_ns=NOW - 50,
        preferred_object_id="preferred",
        require_geometry=True,
    )
    assert selected is new_preferred
    assert _select_target_estimate(
        (_estimate("missing-geometry", 1.0, orientation=False),),
        task,
        NOW,
        1_000,
        phase_entered_ns=NOW - 50,
        require_geometry=True,
    ) is None


def test_planar_transform_snapshot_has_explicit_inverse_direction_and_time() -> None:
    base = BaseState(
        position_xyz=(1.0, 2.0, 9.0),
        orientation_xyzw=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
        yaw=math.pi / 2.0,
        linear_velocity_xyz=(0.0, 0.0, 0.0),
        angular_velocity_xyz=(0.0, 0.0, 0.0),
        frame_id="odom",
        timestamp_ns=123,
    )

    footprint_to_odom = _base_planar_transform_snapshot(
        base, source_frame="footprint", target_frame="odom"
    )
    odom_to_footprint = _base_planar_transform_snapshot(
        base, source_frame="odom", target_frame="footprint"
    )
    world_to_footprint = _base_planar_transform_snapshot(
        base, source_frame="world", target_frame="footprint"
    )

    assert footprint_to_odom.translation_xyz == pytest.approx((1.0, 2.0, 0.0))
    assert footprint_to_odom.rotation_xyzw == pytest.approx(
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    )
    assert odom_to_footprint.translation_xyz == pytest.approx((-2.0, 1.0, 0.0))
    assert odom_to_footprint.rotation_xyzw == pytest.approx(
        (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))
    )
    assert world_to_footprint.translation_xyz == pytest.approx((-2.0, 1.0, 0.0))
    assert world_to_footprint.timestamp_ns == 123
    assert (world_to_footprint.source_frame, world_to_footprint.target_frame) == (
        "world",
        "footprint",
    )


@pytest.mark.parametrize(
    "phase", [GlobalPhase.SAFE_HOLD, GlobalPhase.DONE, GlobalPhase.FAILED]
)
def test_terminal_and_safe_hold_phases_do_not_run_normal_modules(
    phase: GlobalPhase,
) -> None:
    node = _node()
    node._compute_candidate_commands = lambda *_args: pytest.fail(
        "terminal/safe phase must not run ordinary candidate logic"
    )
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)
    status = FSMStatus(
        1, phase, LocalPhase.IDLE, 0, phase is GlobalPhase.DONE, "", NOW
    )

    assert node._run_current_phase(snapshot, status, None, NOW) == (None, None, None)


def test_internal_fsm_publish_authorization_is_pure_and_fail_closed() -> None:
    task = _task()
    context = CompetitionContext(
        schema_name="team_sorting.competition_context",
        schema_version=1,
        run_id="run-1",
        task_set_fingerprint="fingerprint",
        current_task_id=task.task_id,
        current_attempt_count=0,
        elapsed_sim_s=1.0,
        score=0,
        best_scores=(0, 0, 0),
        current_step="-",
        finished=False,
        active_task=task,
        instruction_timestamp_ns=NOW - 100,
        referee_timestamp_ns=NOW - 50,
        valid=True,
        failure_reason="",
    )
    snapshot = SensorSnapshot(task, _base(), _joints(), (), NOW, True)
    status = FSMStatus(1, GlobalPhase.NAV_TO_PICK, LocalPhase.IDLE, 0, False, "", NOW)
    base_command = BaseCommand(0.1, 0.0, NOW, NOW + 100)
    final_action, _ = ActionMux().compose_with_decision(
        base_command, None, _joints(), status, NOW
    )

    assert _internal_fsm_publish_authorization(
        observe_only=False,
        enable_official_publish=True,
        context=context,
        snapshot=snapshot,
        fsm_status=status,
        base_command=base_command,
        manipulation_command=None,
        final_action=final_action,
        now_ns=NOW,
    )
    assert not _internal_fsm_publish_authorization(
        observe_only=True,
        enable_official_publish=True,
        context=context,
        snapshot=snapshot,
        fsm_status=status,
        base_command=base_command,
        manipulation_command=None,
        final_action=final_action,
        now_ns=NOW,
    )
    zero_base = BaseCommand(0.0, 0.0, NOW, NOW + 100)
    assert not _internal_fsm_publish_authorization(
        observe_only=False,
        enable_official_publish=True,
        context=context,
        snapshot=snapshot,
        fsm_status=status,
        base_command=zero_base,
        manipulation_command=None,
        final_action=final_action,
        now_ns=NOW,
    )
    assert not _internal_fsm_publish_authorization(
        observe_only=False,
        enable_official_publish=True,
        context=context,
        snapshot=snapshot,
        fsm_status=status,
        base_command=None,
        manipulation_command=_external_command(),
        final_action=final_action,
        now_ns=NOW,
        manipulation_source="external_candidate",
    )


def test_unwired_business_tick_never_advances_fsm_without_real_feedback() -> None:
    node = _node()
    _prime_control_state(node)
    node._control_tick()
    node._on_instruction(_message(json.dumps(OFFICIAL_TASKS)))
    _feed_official_context(node, task=1, attempt=0)
    assert node._fsm.phase is GlobalPhase.SEARCH_TARGET

    node._control_tick()
    node._control_tick()

    assert node._fsm.phase is GlobalPhase.SEARCH_TARGET


def test_destroy_node_resets_runtime_wiring_and_execution_memory() -> None:
    node = _node()
    calls = {"reset": 0}
    node._arm_execution = SimpleNamespace(
        reset=lambda: calls.__setitem__("reset", calls["reset"] + 1)
    )
    _dirty_runtime_wiring(node)

    node.destroy_node()

    assert calls["reset"] == 1
    _assert_runtime_payload_cleared(node)


def test_enabled_official_gate_without_candidate_does_not_publish_control() -> None:
    node = _node(observe_only=False, enable_official_publish=True)
    _prime_control_state(node)

    node._control_tick()

    assert _official_message_count(node) == 0
    assert len(node.publishers["/team/fsm_status"].messages) == 1
    assert len(node.publishers["/team/final_action"].messages) == 1
    assert len(node.publishers["/team/action_dispatch"].messages) == 1


def test_one_accepted_candidate_authorizes_exactly_one_control_cycle() -> None:
    node = _node(
        _tracking_enabled(), observe_only=False, enable_official_publish=True
    )
    positions = list(_joints().position)
    positions[2] = 0.002624
    joints = RobotJointState(
        position=tuple(positions),
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=NOW,
    )
    _prime_control_state(node, joints)
    _set_active_fsm(node)
    decisions = iter(
        (
            ExternalCandidateDecision(
                "candidate_consumed", NOW, True, command=_external_command()
            ),
            ExternalCandidateDecision(
                "no_pending_candidate", NOW, False, "no_pending_candidate"
            ),
        )
    )
    node._external_candidate.take = lambda **_kwargs: next(decisions)

    node._control_tick()

    assert len(node.publishers["/head_forward_position_controller/commands"].messages) == 1
    assert (
        node.publishers["/head_forward_position_controller/commands"].messages[0].data
        == [0.001, 0.0]
    )
    assert node._official_publisher.last_head_controller_target == [0.001, 0.0]
    final_payload = json.loads(
        node.publishers["/team/final_action"].messages[0].data
    )
    assert final_payload["action"][3:5] == [0.001, 0.002624]
    dispatch_payload = json.loads(
        node.publishers["/team/action_dispatch"].messages[0].data
    )
    assert dispatch_payload["dispatch_mode"] == "head_only"
    assert dispatch_payload["dispatched_action"][3:5] == [0.001, 0.0]
    assert dispatch_payload["dispatched_mask"] == [
        False,
        False,
        False,
        True,
        True,
        *([False] * 14),
    ]
    assert dispatch_payload["attempted_groups"] == ["head"]
    assert dispatch_payload["controller_accepted"] is None
    assert dispatch_payload["execution_confirmed"] is None
    assert all(
        len(node.publishers[topic].messages) == 0
        for topic in OFFICIAL_TOPICS
        if topic != "/head_forward_position_controller/commands"
    )

    node._control_tick()

    assert len(node.publishers["/head_forward_position_controller/commands"].messages) == 1
    assert all(
        len(node.publishers[topic].messages) == 0
        for topic in OFFICIAL_TOPICS
        if topic != "/head_forward_position_controller/commands"
    )
    assert len(node.publishers["/team/fsm_status"].messages) == 2
    assert len(node.publishers["/team/final_action"].messages) == 2
    assert len(node.publishers["/team/action_dispatch"].messages) == 2


def test_head_shadow_starts_at_confirmed_reset_target() -> None:
    node = _node(
        _tracking_enabled(), observe_only=False, enable_official_publish=True
    )

    assert node._official_publisher.last_head_controller_target == [0.0, 0.0]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"enabled": False, "fresh_reset_confirmed": True}, "head_target_tracking_disabled"),
        ({"enabled": True, "fresh_reset_confirmed": False}, "fresh_reset_not_confirmed"),
    ],
)
def test_head_tracking_gate_failure_does_not_publish(
    overrides: dict[str, Any], reason: str
) -> None:
    node = _node(overrides, observe_only=False, enable_official_publish=True)
    _prime_control_state(node)
    node._external_candidate.take = lambda **_kwargs: ExternalCandidateDecision(
        "candidate_consumed", NOW, True, command=_external_command()
    )

    node._control_tick()

    assert _official_message_count(node) == 0
    assert node._official_publisher.last_head_controller_target is None
    assert node._official_publisher.last_head_publish_failure_reason == reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", 1),
        ("require_fresh_reset", False),
        ("fresh_reset_confirmed", "yes"),
        ("require_exclusive_writer", False),
        ("initial_yaw_target", True),
        ("initial_pitch_target", float("nan")),
        ("initial_yaw_target", float("inf")),
        ("initial_pitch_target", 0.01),
    ],
)
def test_invalid_head_tracking_config_is_rejected_at_startup(
    field: str, value: Any
) -> None:
    ros = _ros()
    with pytest.raises(RuntimeError, match="head|initial|Stage 2A"):
        _create_team_client_node(ros)(
            _config({field: value}, observe_only=False, enable_official_publish=True),
            ros,
        )


def test_other_head_writer_fails_closed_and_records_reason() -> None:
    node = _node(
        _tracking_enabled(), observe_only=False, enable_official_publish=True
    )
    _prime_control_state(node)
    node.additional_publishers["/head_forward_position_controller/commands"] = 1
    node._external_candidate.take = lambda **_kwargs: ExternalCandidateDecision(
        "candidate_consumed", NOW, True, command=_external_command()
    )

    node._control_tick()

    assert _official_message_count(node) == 0
    assert node._official_publisher.last_head_controller_target == [0.0, 0.0]
    assert (
        node._official_publisher.last_head_publish_failure_reason
        == "head_writer_not_exclusive"
    )
    audit = json.loads(
        next(
            message.split("external_candidate_audit:", 1)[1]
            for _, message in reversed(node.logger.messages)
            if "external_candidate_audit:" in message
        )
    )
    assert audit["failure_reason"] == "head_writer_not_exclusive"
    assert audit["official_publish_success"] is False


def test_head_publish_failure_does_not_update_shadow_or_other_topics() -> None:
    node = _node(
        _tracking_enabled(), observe_only=False, enable_official_publish=True
    )
    _prime_control_state(node)
    node.publishers["/head_forward_position_controller/commands"].fail = True
    node._external_candidate.take = lambda **_kwargs: ExternalCandidateDecision(
        "candidate_consumed", NOW, True, command=_external_command()
    )

    node._control_tick()

    assert _official_message_count(node) == 0
    assert node._official_publisher.last_head_controller_target == [0.0, 0.0]
    assert node._official_publisher.last_head_publish_failure_reason == "head_publish_failed"


def test_rejected_candidate_does_not_publish_control_but_keeps_telemetry() -> None:
    node = _node(observe_only=False, enable_official_publish=True)
    _prime_control_state(node)
    node._external_candidate.take = lambda **_kwargs: ExternalCandidateDecision(
        "candidate_rejected", NOW, False, "joint_delta_exceeds_limit"
    )

    node._control_tick()

    assert _official_message_count(node) == 0
    assert len(node.publishers["/team/fsm_status"].messages) == 1
    assert len(node.publishers["/team/final_action"].messages) == 1


def test_accepted_candidate_with_non_head_only_mask_fails_closed() -> None:
    node = _node(observe_only=False, enable_official_publish=True)
    _prime_control_state(node)
    node._external_candidate.take = lambda **_kwargs: ExternalCandidateDecision(
        "candidate_consumed",
        NOW,
        True,
        command=_external_command((1, 2)),
    )

    node._control_tick()

    assert _official_message_count(node) == 0
    assert len(node.publishers["/team/fsm_status"].messages) == 1
    assert len(node.publishers["/team/final_action"].messages) == 1
    assert any(
        "不是严格head_yaw-only mask" in message
        for _, message in node.logger.messages
    )


def test_publish_final_action_without_official_publisher_is_diagnostic_only() -> None:
    node = _node()
    action, decision = ActionMux().compose_with_decision(
        None, None, _joints(), node._fsm.status(NOW), NOW
    )

    assert node._publish_final_action(action, decision=decision) is False
    assert len(node.publishers["/team/final_action"].messages) == 1
    assert len(node.publishers["/team/action_dispatch"].messages) == 1
    assert not OFFICIAL_TOPICS.intersection(node.publishers)


def test_emergency_stop_without_official_publisher_only_records_diagnostic() -> None:
    node = _node()
    before = {topic: len(pub.messages) for topic, pub in node.publishers.items()}

    node._publish_emergency_base_stop("JointState missing")

    assert {topic: len(pub.messages) for topic, pub in node.publishers.items()} == before
    assert any(
        "official_publish_disabled" in message for _, message in node.logger.messages
    )


def test_accepted_external_candidate_cannot_cross_observe_only_gate() -> None:
    node = _node()
    _prime_control_state(node)
    node._external_candidate.take = lambda **_kwargs: ExternalCandidateDecision(
        "candidate_consumed", NOW, True, command=_external_command()
    )

    node._control_tick()

    assert not OFFICIAL_TOPICS.intersection(node.publishers)
    audit_lines = [
        message.split("external_candidate_audit:", 1)[1]
        for _, message in node.logger.messages
        if "external_candidate_audit:" in message
    ]
    audit = json.loads(audit_lines[-1])
    assert audit["official_publish_attempted"] is False
    assert audit["official_publish_success"] is False


class _OfficialSpy:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.publish_count = 0
        self.emergency_count = 0

    def publish(self, _action: object) -> None:
        self.publish_count += 1
        self.events.append("official_publish")

    def publish_emergency_base_stop(self) -> None:
        self.emergency_count += 1
        self.events.append("official_emergency")


def test_destroy_node_cleans_observe_only_without_official_publish_and_is_idempotent() -> None:
    node = _node()
    node._external_candidate.shutdown = lambda _now: node.events.append(
        "candidate_shutdown"
    )

    assert node.destroy_node() == "destroyed"
    assert node.destroy_node() is None

    assert node.events.index("timer_cancel") < node.events.index("candidate_shutdown")
    assert node.events.index("candidate_shutdown") < node.events.index("super_destroy")
    assert node.destroy_count == 1
    assert not any(event.startswith("official_") for event in node.events)


def test_destroy_node_with_fresh_joints_clears_pending_without_any_control_publish() -> None:
    node = _node(
        _tracking_enabled(), observe_only=False, enable_official_publish=True
    )
    _prime_control_state(node)

    class PendingConsumer:
        pending = True

        def shutdown(self, _now: int) -> None:
            self.pending = False
            node.events.append("candidate_shutdown")

    pending = PendingConsumer()
    node._external_candidate = pending
    node._safe_hold_candidates = lambda *_args: pytest.fail(
        "destroy_node must not build a full-joint hold"
    )
    node._official_publisher.publish = lambda *_args: pytest.fail(
        "destroy_node must not call OfficialCommandPublisher.publish"
    )
    node._official_publisher.publish_head = lambda *_args: pytest.fail(
        "destroy_node must not publish head"
    )
    node._official_publisher.publish_emergency_base_stop = lambda: pytest.fail(
        "destroy_node must not publish cmd_vel"
    )
    shadow_before = list(node._official_publisher.last_head_controller_target)

    assert node.destroy_node() == "destroyed"
    assert node.destroy_node() is None

    assert pending.pending is False
    assert node._official_publisher.last_head_controller_target == shadow_before
    assert _official_message_count(node) == 0
    assert node.events.index("timer_cancel") < node.events.index("candidate_shutdown")
    assert node.events.index("candidate_shutdown") < node.events.index("super_destroy")
    assert node.destroy_count == 1


def test_destroy_node_never_publishes_after_context_is_invalid() -> None:
    node = _node(observe_only=False, enable_official_publish=True)
    spy = _OfficialSpy(node.events)
    node._official_publisher = spy
    node.context.active = False
    node.get_logger = lambda: pytest.fail(
        "destroy_node must not call ROS logger after context invalidation"
    )

    node.destroy_node()

    assert spy.publish_count == 0
    assert spy.emergency_count == 0
    assert node.destroy_count == 1
    assert node._last_non_ros_warning == ""


def test_spin_destroys_node_before_rclpy_shutdown() -> None:
    events: list[str] = []
    context = _Context(events)

    class Node:
        def __init__(self, _config: dict[str, object], _ros: object) -> None:
            self.context = context

        def destroy_node(self) -> None:
            events.append("destroy_node")

    class Rclpy:
        def init(self, args: object = None) -> None:
            events.append("init")

        def spin(self, _node: object) -> None:
            events.append("spin")

        def ok(self, context: _Context | None = None) -> bool:
            return bool(context and context.ok())

        def shutdown(self) -> None:
            events.append("shutdown")
            context.active = False

    _spin(SimpleNamespace(rclpy=Rclpy()), Node, {}, None)

    assert events == ["init", "spin", "destroy_node", "shutdown"]


def test_spin_does_not_repeat_shutdown_after_signal_closed_context() -> None:
    events: list[str] = []
    context = _Context(events)

    class Node:
        def __init__(self, _config: dict[str, object], _ros: object) -> None:
            self.context = context

        def destroy_node(self) -> None:
            events.append("destroy_node")

    class Rclpy:
        def init(self, args: object = None) -> None:
            events.append("init")

        def spin(self, _node: object) -> None:
            events.append("spin")
            context.active = False
            raise KeyboardInterrupt

        def ok(self, context: _Context | None = None) -> bool:
            return bool(context and context.ok())

        def shutdown(self) -> None:
            pytest.fail("shutdown must not be repeated for an invalid context")

    _spin(SimpleNamespace(rclpy=Rclpy()), Node, {}, None)

    assert events == ["init", "spin", "destroy_node"]
