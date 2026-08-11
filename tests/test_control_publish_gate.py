"""Global official-publish gate and TeamClientNode lifecycle regressions."""

from __future__ import annotations

from dataclasses import asdict, replace
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
    ArmMotionPhase,
    BaseCommand,
    BaseState,
    FSMStatus,
    GlobalPhase,
    GraspContext,
    GraspTarget,
    GraspVerification,
    JointTrajectory,
    JointWaypoint,
    LocalPhase,
    ManipulationCommand,
    ManipulationStatus,
    NavGoal,
    NavigationStatus,
    ObjectEstimate3D,
    PlaceTarget,
    Pose3D,
    RigidTransform3D,
    RobotJointState,
    SensorSnapshot,
    TaskSpec,
)
from team_sorting.ros_nodes import (
    _arm_execution_control_granted,
    _base_planar_transform_snapshot,
    _create_perception_node,
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


def _pose() -> Pose3D:
    return Pose3D((1.0, 0.0, 0.5), (0.0, 0.0, 0.0, 1.0), "footprint")


def _grasp_context(*, confirmed: bool = False) -> GraspContext:
    object_frame = "object/target-a"
    transform_left = RigidTransform3D(
        "left_gripper", object_frame, (0.0, 0.05, 0.0),
        (0.0, 0.0, 0.0, 1.0), NOW - 20, True,
    )
    transform_right = RigidTransform3D(
        "right_gripper", object_frame, (0.0, -0.05, 0.0),
        (0.0, 0.0, 0.0, 1.0), NOW - 20, True,
    )
    return GraspContext(
        task_id=1,
        target_body="box_1",
        target_class_id="pink",
        object_id="target-a",
        object_frame=object_frame,
        object_size_xyz_m=(0.2, 0.1, 0.1),
        object_from_left_gripper=transform_left,
        object_from_right_gripper=transform_right,
        object_orientation_world_xyzw_at_grasp=(0.0, 0.0, 0.0, 1.0),
        orientation_observed_at_ns=NOW - 30,
        planned_at_ns=NOW - 20,
        confirmed_at_ns=NOW - 5 if confirmed else None,
        confirmed=confirmed,
        valid=True,
    )


def _trajectory(phase: GlobalPhase) -> JointTrajectory:
    phases = (
        (
            ArmMotionPhase.PREGRASP,
            ArmMotionPhase.GRASP,
            ArmMotionPhase.LIFT,
            ArmMotionPhase.RETREAT,
        )
        if phase is GlobalPhase.EXECUTE_PICK
        else (
            ArmMotionPhase.PREPLACE,
            ArmMotionPhase.LOWER,
            ArmMotionPhase.RELEASE,
            ArmMotionPhase.POST_RELEASE_RETREAT,
        )
    )
    return JointTrajectory(
        trajectory_id=("pick-test" if phase is GlobalPhase.EXECUTE_PICK else "place-test"),
        task_id=1,
        target_body="box_1",
        execution_phase=phase,
        waypoints=tuple(
            JointWaypoint(item, float(index), (0.0,) * 17, (True, *([False] * 16)))
            for index, item in enumerate(phases, start=1)
        ),
        timestamp_ns=NOW - 5,
    )


def _grasp_target(context: GraspContext) -> GraspTarget:
    pose = _pose()
    return GraspTarget(
        pose, pose, pose, pose, pose, pose, pose, pose, context, 0.9, True
    )


def _place_target() -> PlaceTarget:
    pose = _pose()
    return PlaceTarget(pose, pose, pose, pose, pose, pose, pose, 0.1, True)


def _execution_config_values() -> dict[str, object]:
    return asdict(
        ArmExecutionConfig(
            feedback_max_age_ns=100,
            trajectory_max_age_ns=100,
            command_ttl_ns=100,
            max_control_period_ns=100,
            verification_timeout_ns=100,
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


def _visual_verifier_values() -> dict[str, object]:
    return {
        "minimum_lift_delta_m": 0.03,
        "max_horizontal_drift_m": 0.02,
        "max_observation_gap_s": 0.5,
        "minimum_observation_confidence": 0.7,
        "required_frame_id": "odom",
        "min_stationary_observations": 3,
        "max_stationary_spread_m": 0.01,
    }


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


def _event_ready_node(
    phase: GlobalPhase = GlobalPhase.SEARCH_TARGET,
) -> tuple[Any, FSMStatus]:
    node = _node()
    state = node._runtime_wiring
    state.run_id = "run-1"
    state.task_id = 1
    state.attempt = 0
    state.active_target = _estimate("target-a", 0.9)
    state.active_nav_goal = SimpleNamespace(goal_id="goal-a")
    state.active_trajectory_id = "trajectory-a"
    node._fsm.phase_entered_ns = NOW - 10
    status = FSMStatus(1, phase, LocalPhase.IDLE, 0, False, "", NOW)
    node._fsm.status = lambda _now: status
    node._fsm.handle_event = lambda *_args, **_kwargs: True
    return node, status


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


def test_kdl_is_not_constructed_or_imported_when_planning_disabled(
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

    assert calls == {"construct": 0, "self_check": 0}
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


def test_disabled_incomplete_arm_planning_does_not_break_config_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    assert config["arm_planning"]["enabled"] is False
    config["arm_planning"].pop("pregrasp_distance_m")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(ros_nodes_module, "_resolve_config_path", lambda: path)

    loaded = ros_nodes_module._load_config()

    assert loaded["arm_planning"]["enabled"] is False


def test_complete_arm_execution_config_constructs_controller_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"construct": 0, "reset": 0}

    class ExecutionSpy:
        def __init__(
            self,
            config: ArmExecutionConfig,
            *,
            require_control_feedback: bool,
        ) -> None:
            assert isinstance(config, ArmExecutionConfig)
            assert require_control_feedback is True
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
    state.phase_entered_ns = 123
    state.search_target = _estimate("search-old", 0.8)
    state.refined_target = _estimate("refined-old", 0.8)
    state.active_target = _estimate("old", 0.8)
    state.preferred_object_id = "old"
    state.active_nav_goal = object()
    state.planned_grasp_context = object()
    state.confirmed_grasp_context = object()
    state.active_pick_trajectory = object()
    state.active_place_trajectory = object()
    state.active_trajectory_id = "trajectory-old"
    state.active_trajectory_phase_generation = 99
    state.pick_observation_before_lift = _estimate("old", 0.8)
    state.pick_lift_started_ns = 123
    state.grasp_verification_after_observation_timestamp = 124
    state.latest_grasp_verification = object()
    state.latest_grasp_verification_trajectory_id = "trajectory-old"
    state.place_observation_before_release = _estimate("old", 0.8)
    state.place_release_or_completion_ns = 125
    state.place_post_release_observations.append(_estimate("old", 0.8))
    state.latest_place_verification_observation = _estimate("old", 0.8)
    state.last_navigation_status = object()
    state.navigation_success_submitted = True
    state.navigation_failure_submitted = True
    state.navigation_diagnostic = "old navigation"
    state.last_manipulation_status = object()
    state.planning_attempted = True
    state.trajectory_started = True
    state.phase_success_feedback_emitted = True
    state.phase_failure_feedback_emitted = True
    state.manipulation_diagnostic = "old manipulation"
    state.phase_event_keys.add(("old",))
    state.phase_entry_failure_reason = "old failure"
    node._latest_estimates = (_estimate("cached-old", 0.9),)


def _assert_runtime_payload_cleared(node: Any) -> None:
    state = node._runtime_wiring
    assert state.phase_entered_ns is None
    assert state.search_target is None
    assert state.refined_target is None
    assert state.active_target is None
    assert state.preferred_object_id == ""
    assert state.active_nav_goal is None
    assert state.planned_grasp_context is None
    assert state.confirmed_grasp_context is None
    assert state.active_pick_trajectory is None
    assert state.active_place_trajectory is None
    assert state.active_trajectory_id == ""
    assert state.active_trajectory_phase_generation is None
    assert state.pick_observation_before_lift is None
    assert state.pick_lift_started_ns is None
    assert state.grasp_verification_after_observation_timestamp is None
    assert state.latest_grasp_verification is None
    assert state.latest_grasp_verification_trajectory_id == ""
    assert state.place_observation_before_release is None
    assert state.place_release_or_completion_ns is None
    assert not state.place_post_release_observations
    assert state.latest_place_verification_observation is None
    assert state.last_navigation_status is None
    assert not state.navigation_success_submitted
    assert not state.navigation_failure_submitted
    assert state.navigation_diagnostic == ""
    assert state.last_manipulation_status is None
    assert not state.planning_attempted
    assert not state.trajectory_started
    assert not state.phase_success_feedback_emitted
    assert not state.phase_failure_feedback_emitted
    assert state.manipulation_diagnostic == ""
    assert state.phase_event_keys == set()
    assert state.phase_entry_failure_reason == ""
    assert node._latest_estimates == ()


def test_perception_identity_resets_only_at_context_and_fsm_reset_boundaries() -> None:
    perception_class = _create_perception_node(_ros())
    node = perception_class.__new__(perception_class)
    _Node.__init__(node, "perception_node_test")
    calls = {"stabilizer": 0, "estimator": 0}
    node._stabilizer = SimpleNamespace(
        reset=lambda: calls.__setitem__("stabilizer", calls["stabilizer"] + 1)
    )
    node._estimator = SimpleNamespace(
        reset_tracks=lambda: calls.__setitem__("estimator", calls["estimator"] + 1)
    )
    node._last_frame_issue = ""
    node._identity_context_key = None
    node._last_identity_fsm_phase = None
    task = _task()

    def context(*, attempt: int = 0, finished: bool = False) -> CompetitionContext:
        return CompetitionContext(
            schema_name="team_sorting.competition_context",
            schema_version=1,
            run_id="run-1",
            task_set_fingerprint="fingerprint",
            current_task_id=None if finished else 1,
            current_attempt_count=attempt,
            elapsed_sim_s=1.0,
            score=0,
            best_scores=(0, 0, 0),
            current_step="-",
            finished=finished,
            active_task=None if finished else task,
            instruction_timestamp_ns=NOW - 100,
            referee_timestamp_ns=NOW - 50,
            valid=True,
            failure_reason="",
        )

    node._on_competition_context(_message(context().to_json()))
    assert calls == {"stabilizer": 0, "estimator": 0}

    # 同一任务的正常视觉/导航阶段变化不得破坏稳定身份。
    for phase in (
        GlobalPhase.SEARCH_TARGET,
        GlobalPhase.NAV_TO_PICK,
        GlobalPhase.REFINE_TARGET,
    ):
        status = FSMStatus(1, phase, LocalPhase.IDLE, 0, False, "", NOW)
        node._on_fsm_status(_message(ros_nodes_module.fsm_status_to_json(status)))
    assert calls == {"stabilizer": 0, "estimator": 0}

    node._on_competition_context(_message(context(attempt=1).to_json()))
    assert calls == {"stabilizer": 1, "estimator": 1}
    reset_status = FSMStatus(
        1, GlobalPhase.WAIT_READY, LocalPhase.IDLE, 0, False, "", NOW
    )
    node._on_fsm_status(_message(ros_nodes_module.fsm_status_to_json(reset_status)))
    assert calls == {"stabilizer": 2, "estimator": 2}
    node._on_competition_context(_message(context(attempt=1, finished=True).to_json()))
    assert calls == {"stabilizer": 3, "estimator": 3}


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
    selected_after_entry = _estimate("selected-after-entry", 0.9)
    node._runtime_wiring.active_target = selected_after_entry
    assert not node._handle_phase_entry(snapshot, search, None, NOW + 1)
    assert node._runtime_wiring.phase_generation == generation
    assert node._runtime_wiring.active_target is selected_after_entry

    refine = FSMStatus(1, GlobalPhase.REFINE_TARGET, LocalPhase.IDLE, 0, False, "", NOW + 2)
    assert node._handle_phase_entry(snapshot, refine, None, NOW + 2)
    assert node._runtime_wiring.phase_generation == generation + 1
    assert node._runtime_wiring.preferred_object_id == "selected-after-entry"
    assert node._runtime_wiring.active_target is None


def test_navigation_pick_goal_is_built_once_and_reused_for_every_update() -> None:
    node = _node()
    goal = NavGoal(
        "pick-goal", "pick", (0.4, 0.0, 0.0), "odom", 0.05, 0.1, NOW + 1_000
    )
    calls: dict[str, list[object]] = {"build": [], "update": []}

    class NavigationSpy:
        def build_pick_goal(
            self,
            task: TaskSpec,
            target: ObjectEstimate3D,
            base: BaseState,
            timestamp_ns: int,
        ) -> NavGoal:
            calls["build"].append((task, target, base, timestamp_ns))
            return goal

        def update(
            self, base: BaseState, supplied_goal: NavGoal, timestamp_ns: int
        ) -> tuple[BaseCommand, NavigationStatus]:
            calls["update"].append((base, supplied_goal, timestamp_ns))
            return (
                BaseCommand(0.1, 0.0, timestamp_ns, timestamp_ns + 100),
                NavigationStatus(
                    supplied_goal.goal_id,
                    "moving",
                    0.4,
                    0.0,
                    False,
                    "",
                    timestamp_ns,
                ),
            )

    node._navigation = NavigationSpy()
    node._runtime_wiring.search_target = _estimate("stable-target", 0.9)
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)
    status = FSMStatus(1, GlobalPhase.NAV_TO_PICK, LocalPhase.IDLE, 0, False, "", NOW)

    assert node._handle_phase_entry(snapshot, status, None, NOW)
    assert not node._handle_phase_entry(snapshot, status, None, NOW + 1)
    assert node._runtime_wiring.active_nav_goal is goal
    node._compute_candidate_commands(snapshot, status, NOW)
    node._compute_candidate_commands(snapshot, status, NOW + 1)

    assert len(calls["build"]) == 1
    assert [call[1] for call in calls["update"]] == [goal, goal]


@pytest.mark.parametrize(
    ("phase", "expected_event"),
    (
        (GlobalPhase.NAV_TO_PICK, FSMEvent.PICK_NAV_REACHED),
        (GlobalPhase.NAV_TO_PLACE, FSMEvent.PLACE_NAV_REACHED),
        (GlobalPhase.RETURN_END, FSMEvent.RETURN_REACHED),
    ),
)
def test_navigation_arrival_emits_matching_real_feedback_once(
    phase: GlobalPhase, expected_event: FSMEvent
) -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id = "run-1"
    state.task_id = 1
    state.attempt = 0
    state.active_nav_goal = NavGoal(
        "goal-a", "test", (0.0, 0.0, 0.0), "odom", 0.05, 0.1, NOW + 1_000
    )

    class ArrivedNavigation:
        def update(
            self, _base_state: BaseState, supplied_goal: NavGoal, timestamp_ns: int
        ) -> tuple[BaseCommand, NavigationStatus]:
            return (
                BaseCommand(0.0, 0.0, timestamp_ns, timestamp_ns + 100),
                NavigationStatus(
                    supplied_goal.goal_id,
                    "arrived",
                    0.0,
                    0.0,
                    True,
                    "",
                    timestamp_ns,
                ),
            )

    node._navigation = ArrivedNavigation()
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)
    status = FSMStatus(1, phase, LocalPhase.IDLE, 0, False, "", NOW)

    first = node._run_current_phase(snapshot, status, None, NOW)
    second = node._run_current_phase(snapshot, status, None, NOW)

    assert first[0] is not None and first[0].valid
    assert (first[0].v, first[0].w) == (0.0, 0.0)
    assert first[2] is not None
    assert first[2].event is expected_event
    assert first[2].goal_id == "goal-a"
    assert first[2].confirmed_by_real_feedback
    assert second[2] is None


def test_navigation_goal_identity_mismatch_fails_once_with_invalid_zero_base() -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id = "run-1"
    state.task_id = 1
    state.attempt = 0
    state.active_nav_goal = NavGoal(
        "goal-a", "pick", (0.4, 0.0, 0.0), "odom", 0.05, 0.1, NOW + 1_000
    )

    class WrongGoalNavigation:
        def update(
            self, _base_state: BaseState, _goal: NavGoal, timestamp_ns: int
        ) -> tuple[BaseCommand, NavigationStatus]:
            return (
                BaseCommand(0.2, 0.1, timestamp_ns, timestamp_ns + 100),
                NavigationStatus(
                    "other-goal", "moving", 1.0, 0.0, False, "", timestamp_ns
                ),
            )

    node._navigation = WrongGoalNavigation()
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)
    status = FSMStatus(1, GlobalPhase.NAV_TO_PICK, LocalPhase.IDLE, 0, False, "", NOW)

    first = node._run_current_phase(snapshot, status, None, NOW)
    second = node._run_current_phase(snapshot, status, None, NOW)

    assert first[0] is not None and not first[0].valid
    assert (first[0].v, first[0].w) == (0.0, 0.0)
    assert first[2] is not None and first[2].event is FSMEvent.FAILURE
    assert "goal_id" in first[2].reason
    assert second[2] is None


def test_navigation_candidate_enters_action_mux_and_safe_hold_preserves_no_event() -> None:
    node = _node()
    state = node._runtime_wiring
    goal = NavGoal(
        "goal-a", "pick", (0.4, 0.0, 0.0), "odom", 0.05, 0.1, NOW + 1_000
    )
    state.active_nav_goal = goal
    state.navigation_success_submitted = True
    state.navigation_failure_submitted = True
    state.navigation_diagnostic = "latched"

    class MovingNavigation:
        def update(
            self, _base_state: BaseState, supplied_goal: NavGoal, timestamp_ns: int
        ) -> tuple[BaseCommand, NavigationStatus]:
            return (
                BaseCommand(0.1, -0.2, timestamp_ns, timestamp_ns + 100),
                NavigationStatus(
                    supplied_goal.goal_id,
                    "moving",
                    0.4,
                    -0.2,
                    False,
                    "",
                    timestamp_ns,
                ),
            )

    node._navigation = MovingNavigation()
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)
    nav_status = FSMStatus(
        1, GlobalPhase.NAV_TO_PICK, LocalPhase.IDLE, 0, False, "", NOW
    )
    command, manipulation = node._compute_candidate_commands(snapshot, nav_status, NOW)
    action, _ = node._mux.compose_with_decision(
        command, manipulation, _joints(), nav_status, NOW
    )

    assert command is not None and command.valid
    assert action.values[:2] == pytest.approx((0.1, -0.2))

    state.navigation_diagnostic = "latched"
    node._prepare_phase_transition(GlobalPhase.NAV_TO_PICK, GlobalPhase.SAFE_HOLD)
    safe_status = replace(nav_status, global_phase=GlobalPhase.SAFE_HOLD)
    assert node._run_current_phase(snapshot, safe_status, None, NOW) == (
        None,
        None,
        None,
    )
    assert state.active_nav_goal is goal
    assert state.navigation_success_submitted
    assert state.navigation_failure_submitted
    assert state.navigation_diagnostic == "latched"


@pytest.mark.parametrize("bad_base", ("stale", "wrong_frame"))
def test_navigation_bad_odom_fails_once_with_invalid_zero_base(
    bad_base: str,
) -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id = "run-1"
    state.task_id = 1
    state.attempt = 0
    state.active_nav_goal = NavGoal(
        "goal-a", "pick", (0.4, 0.0, 0.0), "odom", 0.05, 0.1, NOW + 1_000
    )
    base = (
        replace(_base(), timestamp_ns=NOW - 150_000_001)
        if bad_base == "stale"
        else replace(_base(), frame_id="world")
    )
    snapshot = SensorSnapshot(_task(), base, _joints(), (), NOW, True)
    status = FSMStatus(1, GlobalPhase.NAV_TO_PICK, LocalPhase.IDLE, 0, False, "", NOW)

    first = node._run_current_phase(snapshot, status, None, NOW)
    second = node._run_current_phase(snapshot, status, None, NOW)

    assert first[0] is not None and not first[0].valid
    assert (first[0].v, first[0].w) == (0.0, 0.0)
    assert first[2] is not None and first[2].event is FSMEvent.FAILURE
    expected = "过期" if bad_base == "stale" else "frame"
    assert expected in first[2].reason
    assert second[2] is None


def test_navigation_phase_transition_publishes_zero_not_old_nonzero_command() -> None:
    node = _node(observe_only=False, enable_official_publish=True)
    _prime_control_state(node)
    task = _task()
    context = CompetitionContext(
        schema_name="team_sorting.competition_context",
        schema_version=1,
        run_id="run-1",
        task_set_fingerprint="fingerprint",
        current_task_id=1,
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
    before = FSMStatus(
        1, GlobalPhase.NAV_TO_PICK, LocalPhase.IDLE, 0, False, "", NOW
    )
    after = replace(before, global_phase=GlobalPhase.REFINE_TARGET)
    statuses = iter((before, after))
    node._active_context = context
    node._fsm = SimpleNamespace(
        task=task,
        phase_entered_ns=NOW - 10,
        status=lambda _now: next(statuses),
        check_timeout=lambda _now: False,
    )
    node._check_readiness = lambda *_args: None
    node._synchronize_runtime_context = lambda _context: None
    node._handle_phase_entry = lambda *_args: False
    node._submit_fsm_event_once = lambda *_args, **_kwargs: True
    node._run_current_phase = lambda *_args: (
        BaseCommand(0.2, 0.1, NOW, NOW + 100),
        None,
        SimpleNamespace(
            event=FSMEvent.PICK_NAV_REACHED,
            source_timestamp_ns=NOW,
            run_id="run-1",
            task_id=1,
            attempt=0,
            reason="arrived",
            object_id="",
            goal_id="goal-a",
            trajectory_id="",
            confirmed_by_real_feedback=True,
        ),
    )

    node._control_tick()

    cmd_vel = node.publishers["/cmd_vel"].messages[-1]
    assert cmd_vel.linear.x == 0.0
    assert cmd_vel.angular.z == 0.0
    action = json.loads(node.publishers["/team/final_action"].messages[-1].data)
    assert action["action"][:2] == [0.0, 0.0]


def test_phase_specific_cleanup_preserves_cross_phase_pick_context() -> None:
    node = _node()
    state = node._runtime_wiring
    target = _estimate("target-a", 0.9)
    pick_trajectory = object()
    planned_context = object()
    reset_calls = {"count": 0}
    node._arm_execution = SimpleNamespace(
        reset=lambda: reset_calls.__setitem__("count", reset_calls["count"] + 1)
    )
    state.active_target = target
    state.active_nav_goal = object()
    node._prepare_phase_transition(GlobalPhase.SEARCH_TARGET, GlobalPhase.NAV_TO_PICK)
    assert state.active_target is target
    assert state.active_nav_goal is None

    state.active_nav_goal = object()
    state.active_pick_trajectory = object()
    state.planned_grasp_context = object()
    node._prepare_phase_transition(GlobalPhase.NAV_TO_PICK, GlobalPhase.REFINE_TARGET)
    assert state.preferred_object_id == "target-a"
    assert state.active_target is None
    assert state.active_nav_goal is None
    assert state.active_pick_trajectory is None
    assert state.planned_grasp_context is None

    node._prepare_phase_transition(GlobalPhase.REFINE_TARGET, GlobalPhase.PLAN_PICK)
    assert state.active_target is None
    state.active_pick_trajectory = pick_trajectory
    state.planned_grasp_context = planned_context
    node._prepare_phase_transition(GlobalPhase.PLAN_PICK, GlobalPhase.EXECUTE_PICK)
    assert state.active_pick_trajectory is pick_trajectory
    assert state.planned_grasp_context is planned_context

    state.latest_grasp_verification = object()
    node._prepare_phase_transition(GlobalPhase.EXECUTE_PICK, GlobalPhase.VERIFY_PICK)
    assert state.active_pick_trajectory is pick_trajectory
    assert state.planned_grasp_context is planned_context
    assert state.latest_grasp_verification is not None
    assert reset_calls["count"] == 0


@pytest.mark.parametrize("phase", [GlobalPhase.DONE, GlobalPhase.FAILED])
def test_terminal_phase_clears_visual_verification_caches(
    phase: GlobalPhase,
) -> None:
    node = _node()
    state = node._runtime_wiring
    state.pick_observation_before_lift = _estimate("target-a", 0.9)
    state.latest_grasp_verification = object()
    state.place_observation_before_release = _estimate("target-a", 0.9)
    state.latest_place_verification_observation = _estimate("target-a", 0.9)

    node._prepare_phase_transition(GlobalPhase.VERIFY_PLACE, phase)

    assert state.pick_observation_before_lift is None
    assert state.latest_grasp_verification is None
    assert state.place_observation_before_release is None
    assert state.latest_place_verification_observation is None


def test_place_phase_cleanup_keeps_only_confirmed_grasp_and_place_trajectory() -> None:
    node = _node()
    state = node._runtime_wiring
    confirmed_context = object()
    state.active_target = _estimate("target-a", 0.9)
    state.active_nav_goal = object()
    state.planned_grasp_context = object()
    state.confirmed_grasp_context = confirmed_context
    state.active_pick_trajectory = object()
    state.active_place_trajectory = object()
    state.active_trajectory_id = "old-trajectory"
    state.latest_grasp_verification = object()

    node._prepare_phase_transition(GlobalPhase.VERIFY_PICK, GlobalPhase.NAV_TO_PLACE)
    assert state.confirmed_grasp_context is confirmed_context
    assert state.active_target is None
    assert state.active_nav_goal is None
    assert state.planned_grasp_context is None
    assert state.active_pick_trajectory is None
    assert state.active_place_trajectory is None
    assert state.latest_grasp_verification is None

    node._prepare_phase_transition(GlobalPhase.NAV_TO_PLACE, GlobalPhase.PLAN_PLACE)
    assert state.confirmed_grasp_context is confirmed_context
    place_trajectory = object()
    state.active_place_trajectory = place_trajectory
    state.active_trajectory_id = "place-trajectory"
    node._prepare_phase_transition(GlobalPhase.PLAN_PLACE, GlobalPhase.EXECUTE_PLACE)
    assert state.active_place_trajectory is place_trajectory
    assert state.active_trajectory_id == "place-trajectory"
    assert state.confirmed_grasp_context is confirmed_context
    release_baseline = _estimate("released-target", 0.8)
    state.place_observation_before_release = release_baseline
    state.latest_place_verification_observation = _estimate("stale", 0.7)
    node._prepare_phase_transition(GlobalPhase.EXECUTE_PLACE, GlobalPhase.VERIFY_PLACE)
    assert state.place_observation_before_release is release_baseline
    assert state.latest_place_verification_observation is None
    assert state.confirmed_grasp_context is confirmed_context


def test_safe_hold_and_recovery_preserve_runtime_context_without_execution_reset() -> None:
    node = _node()
    _dirty_runtime_wiring(node)
    state = node._runtime_wiring
    before = {
        name: getattr(state, name)
        for name in (
            "active_target",
            "active_nav_goal",
            "planned_grasp_context",
            "confirmed_grasp_context",
            "active_pick_trajectory",
            "active_place_trajectory",
            "active_trajectory_id",
        )
    }
    reset_calls = {"count": 0}
    node._arm_execution = SimpleNamespace(
        reset=lambda: reset_calls.__setitem__("count", reset_calls["count"] + 1)
    )

    node._prepare_phase_transition(GlobalPhase.EXECUTE_PICK, GlobalPhase.SAFE_HOLD)
    assert {name: getattr(state, name) for name in before} == before
    node._prepare_phase_transition(GlobalPhase.SAFE_HOLD, GlobalPhase.EXECUTE_PICK)
    assert {name: getattr(state, name) for name in before} == before
    assert reset_calls["count"] == 0

    node._prepare_phase_transition(GlobalPhase.SAFE_HOLD, GlobalPhase.FAILED)
    assert state.active_target is None
    assert state.active_nav_goal is None
    assert state.active_pick_trajectory is None
    assert state.active_place_trajectory is None
    assert reset_calls["count"] == 0


def test_full_lifecycle_resets_call_execution_reset() -> None:
    node = _node()
    reset_calls = {"count": 0}
    node._arm_execution = SimpleNamespace(
        reset=lambda: reset_calls.__setitem__("count", reset_calls["count"] + 1)
    )
    state = node._runtime_wiring

    node._reset_runtime_wiring(run_id="run-1", task_id=1, attempt=0)
    _dirty_runtime_wiring(node)
    node._reset_runtime_wiring(task_id=2, attempt=0, reason="new_task")
    _dirty_runtime_wiring(node)
    node._reset_runtime_wiring(attempt=1, reason="new_attempt")
    _dirty_runtime_wiring(node)
    node._reset_runtime_wiring(context_finished=True, reason="finished")

    assert reset_calls["count"] == 4
    _assert_runtime_payload_cleared(node)
    assert state.context_finished


def test_event_submission_is_bounded_and_deduplicated() -> None:
    node = _node()
    node._runtime_wiring.run_id = "run-1"
    node._runtime_wiring.task_id = 1
    node._runtime_wiring.attempt = 0
    node._runtime_wiring.active_target = _estimate("target-a", 0.9)
    node._fsm.phase_entered_ns = NOW - 1
    _set_active_fsm(node)
    calls: list[FSMEvent] = []
    node._fsm.handle_event = lambda event, _now, _reason="": calls.append(event) or True
    status = FSMStatus(1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW)

    assert node._submit_fsm_event_once(
        FSMEvent.TARGET_FOUND,
        status,
        NOW,
        source_timestamp_ns=NOW,
        run_id="run-1",
        task_id=1,
        attempt=0,
        object_id="target-a",
        confirmed_by_real_feedback=True,
    )
    assert not node._submit_fsm_event_once(
        FSMEvent.FAILURE,
        status,
        NOW,
        source_timestamp_ns=NOW,
        run_id="run-1",
        task_id=1,
        attempt=0,
        confirmed_by_real_feedback=True,
    )
    assert not node._submit_fsm_event_once(
        FSMEvent.TARGET_FOUND,
        status,
        NOW + 1,
        source_timestamp_ns=NOW,
        run_id="run-1",
        task_id=1,
        attempt=0,
        object_id="target-a",
        confirmed_by_real_feedback=True,
    )
    assert calls == [FSMEvent.TARGET_FOUND]


def test_rejected_fsm_event_is_not_reported_as_transition() -> None:
    node = _node()
    node._runtime_wiring.run_id = "run-1"
    node._runtime_wiring.task_id = 1
    node._runtime_wiring.attempt = 0
    node._runtime_wiring.active_target = _estimate("target-a", 0.9)
    node._fsm.phase_entered_ns = NOW - 1
    _set_active_fsm(node)
    calls = {"count": 0}

    def reject(*_args: object, **_kwargs: object) -> bool:
        calls["count"] += 1
        return False

    node._fsm.handle_event = reject
    status = FSMStatus(1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW)

    assert not node._submit_fsm_event_once(
        FSMEvent.TARGET_FOUND,
        status,
        NOW,
        source_timestamp_ns=NOW,
        run_id="run-1",
        task_id=1,
        attempt=0,
        object_id="target-a",
        confirmed_by_real_feedback=True,
    )
    assert calls["count"] == 1
    assert "拒绝" in node._runtime_wiring.phase_entry_failure_reason


@pytest.mark.parametrize(
    ("source_timestamp_ns", "reason_fragment"),
    [(NOW - 11, "早于当前阶段"), (NOW + 1, "晚于当前控制周期")],
)
def test_event_submission_rejects_stale_and_future_feedback(
    source_timestamp_ns: int, reason_fragment: str
) -> None:
    node, status = _event_ready_node()

    assert not node._submit_fsm_event_once(
        FSMEvent.TARGET_FOUND,
        status,
        NOW,
        source_timestamp_ns=source_timestamp_ns,
        run_id="run-1",
        task_id=1,
        attempt=0,
        object_id="target-a",
        confirmed_by_real_feedback=True,
    )
    assert reason_fragment in node._runtime_wiring.phase_entry_failure_reason


@pytest.mark.parametrize(
    ("run_id", "task_id", "attempt"),
    [("run-old", 1, 0), ("run-1", 2, 0), ("run-1", 1, 1)],
)
def test_event_submission_rejects_runtime_identity_mismatch(
    run_id: str, task_id: int, attempt: int
) -> None:
    node, status = _event_ready_node()

    assert not node._submit_fsm_event_once(
        FSMEvent.TARGET_FOUND,
        status,
        NOW,
        source_timestamp_ns=NOW,
        run_id=run_id,
        task_id=task_id,
        attempt=attempt,
        object_id="target-a",
        confirmed_by_real_feedback=True,
    )
    assert "run/task/attempt" in node._runtime_wiring.phase_entry_failure_reason


@pytest.mark.parametrize(
    ("event", "phase", "identity_field", "failure_fragment"),
    [
        (FSMEvent.TARGET_FOUND, GlobalPhase.SEARCH_TARGET, "object_id", "object_id"),
        (FSMEvent.PICK_NAV_REACHED, GlobalPhase.NAV_TO_PICK, "goal_id", "goal_id"),
        (FSMEvent.PICK_PLAN_READY, GlobalPhase.PLAN_PICK, "trajectory_id", "trajectory_id"),
    ],
)
def test_event_submission_rejects_result_identity_mismatch(
    event: FSMEvent,
    phase: GlobalPhase,
    identity_field: str,
    failure_fragment: str,
) -> None:
    node, status = _event_ready_node(phase)
    identities = {
        "object_id": "target-a",
        "goal_id": "goal-a",
        "trajectory_id": "trajectory-a",
    }
    identities[identity_field] = "wrong"

    assert not node._submit_fsm_event_once(
        event,
        status,
        NOW,
        source_timestamp_ns=NOW,
        run_id="run-1",
        task_id=1,
        attempt=0,
        confirmed_by_real_feedback=True,
        **identities,
    )
    assert failure_fragment in node._runtime_wiring.phase_entry_failure_reason


def test_event_submission_requires_real_feedback_and_string_reason() -> None:
    node, status = _event_ready_node()
    assert not node._submit_fsm_event_once(
        FSMEvent.TARGET_FOUND,
        status,
        NOW,
        source_timestamp_ns=NOW,
        run_id="run-1",
        task_id=1,
        attempt=0,
        object_id="target-a",
    )
    assert "真实反馈" in node._runtime_wiring.phase_entry_failure_reason

    node, status = _event_ready_node()
    with pytest.raises(TypeError, match="reason"):
        node._submit_fsm_event_once(
            FSMEvent.TARGET_FOUND,
            status,
            NOW,
            source_timestamp_ns=NOW,
            run_id="run-1",
            task_id=1,
            attempt=0,
            reason=object(),  # type: ignore[arg-type]
            object_id="target-a",
            confirmed_by_real_feedback=True,
        )


def test_event_deduplication_table_is_bounded() -> None:
    node, status = _event_ready_node()
    node._runtime_wiring.phase_event_keys.update(
        {("existing", index) for index in range(32)}
    )

    assert not node._submit_fsm_event_once(
        FSMEvent.TARGET_FOUND,
        status,
        NOW,
        source_timestamp_ns=NOW,
        run_id="run-1",
        task_id=1,
        attempt=0,
        object_id="target-a",
        confirmed_by_real_feedback=True,
    )
    assert len(node._runtime_wiring.phase_event_keys) == 32
    assert "有界容量" in node._runtime_wiring.phase_entry_failure_reason


def test_reset_event_clears_runtime_identity_and_payload() -> None:
    node = _node()
    node._runtime_wiring.run_id = "run-old"
    node._runtime_wiring.task_id = 1
    node._runtime_wiring.attempt = 2
    _dirty_runtime_wiring(node)
    status = node._fsm.status(NOW)

    assert node._submit_fsm_event_once(
        FSMEvent.RESET,
        status,
        NOW,
        source_timestamp_ns=NOW,
        run_id="run-old",
        task_id=1,
        attempt=2,
        confirmed_by_real_feedback=True,
    )

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
    rejected = (
        replace(tied_a, valid=False, failure_reason="invalid"),
        replace(tied_a, frame_id="camera_link"),
        replace(tied_a, class_id="brown"),
        replace(tied_a, timestamp_ns=NOW + 1),
        replace(tied_a, timestamp_ns=NOW - 1_001),
        replace(tied_a, object_id=None),
    )
    assert _select_target_estimate(rejected, task, NOW, 1_000) is None


def _visual_phase_result(
    phase: GlobalPhase,
    estimates: tuple[ObjectEstimate3D, ...],
    *,
    phase_entered_ns: int = NOW - 100,
) -> tuple[Any, Any]:
    node = _node()
    state = node._runtime_wiring
    state.run_id = "run-1"
    state.task_id = 1
    state.attempt = 0
    node._fsm.phase_entered_ns = phase_entered_ns
    status = FSMStatus(1, phase, LocalPhase.IDLE, 0, False, "", NOW)
    snapshot = SensorSnapshot(_task(), _base(), _joints(), estimates, NOW, True)
    return node, node._run_current_phase(snapshot, status, None, NOW)[2]


def test_search_target_builds_real_feedback_from_selected_observation() -> None:
    lower = _estimate("target-b", 0.8, timestamp_ns=NOW - 20)
    selected = _estimate("target-a", 0.9, timestamp_ns=NOW - 10)

    node, feedback = _visual_phase_result(
        GlobalPhase.SEARCH_TARGET, (lower, selected)
    )

    assert node._runtime_wiring.active_target is selected
    assert feedback.event is FSMEvent.TARGET_FOUND
    assert feedback.source_timestamp_ns == selected.timestamp_ns
    assert feedback.object_id == "target-a"
    assert feedback.confirmed_by_real_feedback is True
    assert (feedback.run_id, feedback.task_id, feedback.attempt) == ("run-1", 1, 0)


def test_search_target_rejects_observation_before_phase_entry() -> None:
    node, feedback = _visual_phase_result(
        GlobalPhase.SEARCH_TARGET,
        (_estimate("target-a", 0.9, timestamp_ns=NOW - 101),),
        phase_entered_ns=NOW - 100,
    )

    assert feedback is None
    assert node._runtime_wiring.active_target is None


def test_real_search_feedback_advances_only_through_existing_event_adapter() -> None:
    node = _node()
    _prime_control_state(node)
    node._control_tick()
    node._on_instruction(_message(json.dumps(OFFICIAL_TASKS)))
    _feed_official_context(node, task=1, attempt=0)
    assert node._fsm.phase is GlobalPhase.SEARCH_TARGET
    node._latest_estimates = (_estimate("stable:7", 0.9, timestamp_ns=NOW),)

    node._control_tick()

    assert node._fsm.phase is GlobalPhase.NAV_TO_PICK
    assert node._runtime_wiring.active_target.object_id == "stable:7"


@pytest.mark.parametrize(
    "estimate",
    [
        replace(_estimate("id", 0.9), object_id=None),
        replace(_estimate("id", 0.9), class_id="brown"),
        replace(_estimate("id", 0.9), frame_id="world"),
        replace(_estimate("id", 0.9), timestamp_ns=NOW + 1),
        replace(_estimate("id", 0.9), timestamp_ns=NOW - 150_000_001),
        replace(_estimate("id", 0.9), valid=False, failure_reason="invalid"),
    ],
    ids=("missing-id", "wrong-class", "wrong-frame", "future", "stale", "invalid"),
)
def test_search_target_rejects_unsafe_visual_candidates(
    estimate: ObjectEstimate3D,
) -> None:
    node, feedback = _visual_phase_result(GlobalPhase.SEARCH_TARGET, (estimate,))

    assert feedback is None
    assert node._runtime_wiring.active_target is None


def test_non_odom_planning_config_keeps_visual_event_closed() -> None:
    node, _ = _visual_phase_result(
        GlobalPhase.SEARCH_TARGET, (_estimate("target-a", 0.9),)
    )
    node._runtime_wiring.active_target = None
    node._config["frames"]["planning"] = "world"
    status = FSMStatus(
        1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW
    )
    snapshot = SensorSnapshot(
        _task(), _base(), _joints(), (_estimate("target-a", 0.9),), NOW, True
    )

    assert node._run_current_phase(snapshot, status, None, NOW)[2] is None
    assert node._runtime_wiring.active_target is None
    assert "不是严格odom" in node._runtime_wiring.phase_entry_failure_reason


def test_same_search_observation_is_submitted_only_once() -> None:
    estimate = _estimate("target-a", 0.9, timestamp_ns=NOW - 10)
    node, feedback = _visual_phase_result(GlobalPhase.SEARCH_TARGET, (estimate,))
    status = FSMStatus(1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW)
    node._fsm.status = lambda _now: status
    calls: list[FSMEvent] = []
    node._fsm.handle_event = lambda event, *_args: calls.append(event) or True

    assert node._submit_fsm_event_once(
        feedback.event,
        status,
        NOW,
        source_timestamp_ns=feedback.source_timestamp_ns,
        run_id=feedback.run_id,
        task_id=feedback.task_id,
        attempt=feedback.attempt,
        object_id=feedback.object_id,
        confirmed_by_real_feedback=feedback.confirmed_by_real_feedback,
    )
    assert not node._submit_fsm_event_once(
        feedback.event,
        status,
        NOW + 1,
        source_timestamp_ns=feedback.source_timestamp_ns,
        run_id=feedback.run_id,
        task_id=feedback.task_id,
        attempt=feedback.attempt,
        object_id=feedback.object_id,
        confirmed_by_real_feedback=feedback.confirmed_by_real_feedback,
    )
    assert calls == [FSMEvent.TARGET_FOUND]


def test_refine_entry_preserves_only_identity_and_rejects_old_or_missing_geometry() -> None:
    node = _node()
    state = node._runtime_wiring
    original = _estimate("target-a", 0.9, timestamp_ns=NOW - 200)
    state.active_target = original
    state.phase_event_keys.add(("search",))
    node._fsm.phase_entered_ns = NOW - 100
    status = FSMStatus(1, GlobalPhase.REFINE_TARGET, LocalPhase.IDLE, 0, False, "", NOW)
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)

    assert node._handle_phase_entry(snapshot, status, None, NOW)
    assert state.preferred_object_id == "target-a"
    assert state.active_target is None
    assert state.phase_event_keys == set()

    old_complete = _estimate("target-a", 1.0, timestamp_ns=NOW - 101)
    missing_orientation = _estimate(
        "target-a", 1.0, timestamp_ns=NOW - 10, orientation=False
    )
    missing_size = _estimate("target-a", 1.0, timestamp_ns=NOW - 10, size=False)
    for estimate in (old_complete, missing_orientation, missing_size):
        result = node._run_current_phase(
            replace(snapshot, object_estimates=(estimate,)), status, None, NOW
        )
        assert result[2] is None
        assert state.active_target is None


def test_refine_prefers_original_identity_and_uses_only_new_complete_geometry() -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id = "run-1"
    state.task_id = 1
    state.attempt = 0
    state.preferred_object_id = "target-a"
    node._fsm.phase_entered_ns = NOW - 100
    status = FSMStatus(1, GlobalPhase.REFINE_TARGET, LocalPhase.IDLE, 0, False, "", NOW)
    other = _estimate("target-b", 1.0, timestamp_ns=NOW - 5)
    preferred = _estimate("target-a", 0.7, timestamp_ns=NOW - 4)
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (other, preferred), NOW, True)

    _, _, feedback = node._run_current_phase(snapshot, status, None, NOW)

    assert state.active_target is preferred
    assert feedback.event is FSMEvent.TARGET_REFINED
    assert feedback.object_id == "target-a"
    assert feedback.source_timestamp_ns == preferred.timestamp_ns


def test_refine_never_rebinds_when_preferred_identity_is_missing() -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id, state.task_id, state.attempt = "run-1", 1, 0
    state.preferred_object_id = "target-a"
    node._fsm.phase_entered_ns = NOW - 100
    status = FSMStatus(1, GlobalPhase.REFINE_TARGET, LocalPhase.IDLE, 0, False, "", NOW)
    snapshot = SensorSnapshot(
        _task(), _base(), _joints(),
        (_estimate("target-b", 1.0, timestamp_ns=NOW - 5),), NOW, True,
    )

    assert node._run_current_phase(snapshot, status, None, NOW)[2] is None
    assert state.active_target is None
    assert "禁止静默重绑定" in state.phase_entry_failure_reason


def test_current_perception_geometry_gap_keeps_refine_closed_and_never_plans() -> None:
    node = _node()
    node._fsm.phase_entered_ns = NOW - 100
    node._arm_planner = SimpleNamespace(
        plan_grasp=lambda *_args: pytest.fail("REFINE不得调用ArmPlanner.plan_grasp")
    )
    status = FSMStatus(1, GlobalPhase.REFINE_TARGET, LocalPhase.IDLE, 0, False, "", NOW)
    current_output = _estimate(
        "stable:7", 0.9, timestamp_ns=NOW - 10, orientation=False, size=False
    )
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (current_output,), NOW, True)

    assert node._run_current_phase(snapshot, status, None, NOW)[2] is None
    assert node._runtime_wiring.active_target is None


def test_missing_visual_verifier_config_is_unavailable_without_hidden_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    real = ros_nodes_module.VisualObservationVerifier

    def construct(**kwargs: object) -> object:
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(ros_nodes_module, "VisualObservationVerifier", construct)
    node = _node()

    assert node._visual_observation_verifier is None
    assert "unavailable" in node._visual_observation_verifier_unavailable_reason
    assert calls == []


def test_valid_visual_verifier_config_constructs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    real = ros_nodes_module.VisualObservationVerifier

    def construct(**kwargs: object) -> object:
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(ros_nodes_module, "VisualObservationVerifier", construct)
    config = _config()
    config["perception"]["visual_observation_verifier"] = (
        _visual_verifier_values()
    )
    ros = _ros()
    node = _create_team_client_node(ros)(config, ros)

    assert node._visual_observation_verifier is not None
    assert len(calls) == 1


@pytest.mark.parametrize(
    "configured",
    [
        None,
        {},
        {**_visual_verifier_values(), "unknown": 1},
        {**_visual_verifier_values(), "max_observation_gap_s": None},
        {**_visual_verifier_values(), "max_observation_gap_s": float("nan")},
        {**_visual_verifier_values(), "max_observation_gap_s": 0.0},
        {**_visual_verifier_values(), "min_stationary_observations": True},
    ],
)
def test_incomplete_extra_null_or_invalid_visual_verifier_config_fails_closed(
    configured: object,
) -> None:
    config = _config()
    config["perception"]["visual_observation_verifier"] = configured
    ros = _ros()
    node = _create_team_client_node(ros)(config, ros)

    assert node._visual_observation_verifier is None
    assert "配置无效" in node._visual_observation_verifier_unavailable_reason


@pytest.mark.parametrize("phase", [GlobalPhase.VERIFY_PICK, GlobalPhase.VERIFY_PLACE])
def test_visual_verification_phases_never_emit_business_success_alone(
    phase: GlobalPhase,
) -> None:
    node = _node()
    state = node._runtime_wiring
    state.latest_grasp_verification = object()
    state.latest_place_verification_observation = _estimate("target-a", 0.9)
    snapshot = SensorSnapshot(
        _task(), _base(), _joints(), (_estimate("target-a", 0.9),), NOW, True
    )
    status = FSMStatus(1, phase, LocalPhase.IDLE, 0, False, "", NOW)

    feedback = node._run_current_phase(snapshot, status, None, NOW)[2]
    assert feedback is None or feedback.event is not (
        FSMEvent.PICK_VERIFIED
        if phase is GlobalPhase.VERIFY_PICK
        else FSMEvent.PLACE_VERIFIED
    )


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
    zero_final_action, _ = ActionMux().compose_with_decision(
        zero_base, None, _joints(), status, NOW
    )
    assert _internal_fsm_publish_authorization(
        observe_only=False,
        enable_official_publish=True,
        context=context,
        snapshot=snapshot,
        fsm_status=status,
        base_command=zero_base,
        manipulation_command=None,
        final_action=zero_final_action,
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

    common = {
        "observe_only": False,
        "enable_official_publish": True,
        "context": context,
        "snapshot": snapshot,
        "fsm_status": status,
        "base_command": base_command,
        "manipulation_command": None,
        "final_action": final_action,
        "now_ns": NOW,
    }
    assert not _internal_fsm_publish_authorization(
        **{**common, "context": replace(context, finished=True)}
    )
    assert not _internal_fsm_publish_authorization(
        **{
            **common,
            "snapshot": replace(
                snapshot, valid=False, failure_reason="snapshot invalid"
            ),
        }
    )
    assert not _internal_fsm_publish_authorization(
        **{**common, "snapshot": replace(snapshot, joints=None)}
    )
    assert not _internal_fsm_publish_authorization(
        **{
            **common,
            "fsm_status": replace(status, global_phase=GlobalPhase.SAFE_HOLD),
        }
    )
    assert not _internal_fsm_publish_authorization(
        **{
            **common,
            "final_action": replace(
                final_action,
                global_phase=GlobalPhase.PLAN_PICK,
            ),
        }
    )
    assert not _internal_fsm_publish_authorization(
        **{
            **common,
            "final_action": replace(
                final_action, valid=False, failure_reason="invalid action"
            ),
        }
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


def test_control_tick_processes_real_feedback_before_timeout() -> None:
    node = _node()
    node._system_ready_submitted = True
    _prime_control_state(node)
    state = node._runtime_wiring
    state.run_id = "run-1"
    state.task_id = 1
    state.attempt = 0
    state.last_handled_phase = GlobalPhase.SEARCH_TARGET
    state.active_target = _estimate("target-a", 0.9)
    node._fsm.phase_entered_ns = NOW - 10
    transitioned = {"value": False}
    status_before = FSMStatus(
        1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW
    )
    status_after = replace(status_before, global_phase=GlobalPhase.NAV_TO_PICK)
    node._fsm.status = lambda _now: status_after if transitioned["value"] else status_before

    def handle_event(*_args: object, **_kwargs: object) -> bool:
        transitioned["value"] = True
        return True

    node._fsm.handle_event = handle_event
    node._fsm.check_timeout = lambda _now: pytest.fail(
        "真实事件已转换时不得再检查旧阶段超时"
    )
    feedback = ros_nodes_module._RuntimeFSMFeedback(
        event=FSMEvent.TARGET_FOUND,
        source_timestamp_ns=NOW,
        run_id="run-1",
        task_id=1,
        attempt=0,
        object_id="target-a",
        confirmed_by_real_feedback=True,
    )
    node._run_current_phase = lambda *_args: (None, None, feedback)

    node._control_tick()

    assert transitioned["value"]


def test_control_tick_checks_timeout_when_feedback_is_rejected() -> None:
    node = _node()
    node._system_ready_submitted = True
    _prime_control_state(node)
    state = node._runtime_wiring
    state.run_id = "run-1"
    state.task_id = 1
    state.attempt = 0
    state.last_handled_phase = GlobalPhase.SEARCH_TARGET
    state.active_target = _estimate("target-a", 0.9)
    node._fsm.phase_entered_ns = NOW - 10
    status = FSMStatus(
        1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", NOW
    )
    node._fsm.status = lambda _now: status
    node._fsm.handle_event = lambda *_args, **_kwargs: False
    timeout_calls = {"count": 0}
    node._fsm.check_timeout = lambda _now: timeout_calls.__setitem__(
        "count", timeout_calls["count"] + 1
    ) or False
    feedback = ros_nodes_module._RuntimeFSMFeedback(
        event=FSMEvent.TARGET_FOUND,
        source_timestamp_ns=NOW,
        run_id="run-1",
        task_id=1,
        attempt=0,
        object_id="target-a",
        confirmed_by_real_feedback=True,
    )
    node._run_current_phase = lambda *_args: (None, None, feedback)

    node._control_tick()

    assert timeout_calls["count"] == 1
    assert "FSM拒绝" in state.phase_entry_failure_reason


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


def test_plan_pick_entry_plans_once_with_locked_target_and_explicit_transforms() -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id, state.task_id, state.attempt = "run-1", 1, 0
    target = _estimate("target-a", 0.9, timestamp_ns=NOW - 10)
    state.active_target = target
    state.refined_target = target
    state.preferred_object_id = "target-a"
    context = _grasp_context()
    trajectory = _trajectory(GlobalPhase.EXECUTE_PICK)
    calls: list[tuple[object, ...]] = []

    def plan_grasp(*args: object) -> tuple[GraspTarget, JointTrajectory]:
        calls.append(args)
        return _grasp_target(context), trajectory

    node._arm_planner = SimpleNamespace(plan_grasp=plan_grasp)
    node._fsm.phase_entered_ns = NOW
    status = FSMStatus(1, GlobalPhase.PLAN_PICK, LocalPhase.IDLE, 0, False, "", NOW)
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (target,), NOW, True)

    assert node._handle_phase_entry(snapshot, status, None, NOW)
    assert not node._handle_phase_entry(snapshot, status, None, NOW)
    assert len(calls) == 1
    assert calls[0][1] is target
    target_to_footprint = calls[0][2]
    target_to_world = calls[0][3]
    assert (target_to_footprint.source_frame, target_to_footprint.target_frame) == (
        "odom", "footprint"
    )
    assert target_to_world == RigidTransform3D(
        "odom", "world", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
        target.timestamp_ns, True,
    )
    assert state.planned_grasp_context is context
    assert state.active_pick_trajectory is trajectory
    _, _, feedback = node._run_current_phase(snapshot, status, None, NOW)
    assert feedback.event is FSMEvent.PICK_PLAN_READY
    assert (feedback.object_id, feedback.trajectory_id) == (
        "target-a", trajectory.trajectory_id
    )


def test_plan_pick_missing_geometry_fails_once_without_calling_planner() -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id, state.task_id, state.attempt = "run-1", 1, 0
    target = _estimate("target-a", 0.9, orientation=False)
    state.active_target = target
    state.refined_target = target
    state.preferred_object_id = "target-a"
    node._arm_planner = SimpleNamespace(
        plan_grasp=lambda *_args: pytest.fail("缺姿态时不得调用plan_grasp")
    )
    node._fsm.phase_entered_ns = NOW
    status = FSMStatus(1, GlobalPhase.PLAN_PICK, LocalPhase.IDLE, 0, False, "", NOW)
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (target,), NOW, True)

    node._handle_phase_entry(snapshot, status, None, NOW)
    first = node._run_current_phase(snapshot, status, None, NOW)[2]
    second = node._run_current_phase(snapshot, status, None, NOW + 1)[2]
    assert first.event is FSMEvent.FAILURE
    assert second is None
    assert state.active_pick_trajectory is None


def test_plan_place_requires_confirmed_context_and_plans_once() -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id, state.task_id, state.attempt = "run-1", 1, 0
    state.confirmed_grasp_context = _grasp_context(confirmed=True)
    trajectory = _trajectory(GlobalPhase.EXECUTE_PLACE)
    calls: list[tuple[object, ...]] = []

    def plan_place(*args: object) -> tuple[PlaceTarget, JointTrajectory]:
        calls.append(args)
        return _place_target(), trajectory

    node._arm_planner = SimpleNamespace(plan_place=plan_place)
    node._fsm.phase_entered_ns = NOW
    status = FSMStatus(1, GlobalPhase.PLAN_PLACE, LocalPhase.IDLE, 0, False, "", NOW)
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)

    assert node._handle_phase_entry(snapshot, status, None, NOW)
    assert not node._handle_phase_entry(snapshot, status, None, NOW)
    assert len(calls) == 1
    world_to_footprint = calls[0][1]
    assert (world_to_footprint.source_frame, world_to_footprint.target_frame) == (
        "world", "footprint"
    )
    assert calls[0][2] is state.confirmed_grasp_context
    assert state.active_place_trajectory is trajectory
    _, _, feedback = node._run_current_phase(snapshot, status, None, NOW)
    assert feedback.event is FSMEvent.PLACE_PLAN_READY
    assert feedback.trajectory_id == trajectory.trajectory_id


def test_execute_pick_starts_once_and_steps_latest_joint_feedback() -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id, state.task_id, state.attempt = "run-1", 1, 0
    trajectory = _trajectory(GlobalPhase.EXECUTE_PICK)
    state.active_pick_trajectory = trajectory
    state.active_trajectory_id = trajectory.trajectory_id
    state.planned_grasp_context = _grasp_context()
    calls: dict[str, list[object]] = {"start": [], "step": []}
    command = ManipulationCommand(
        (0.0,) * 17, (True, *([False] * 16)), LocalPhase.MOVE_PREGRASP,
        NOW, NOW + 100,
    )

    class Execution:
        latest_grasp_verification = None

        def start_trajectory(self, value: object) -> ManipulationStatus:
            calls["start"].append(value)
            return ManipulationStatus(
                LocalPhase.IDLE, "LOADED", 0.0, float("inf"), False,
                "loaded", NOW - 5,
            )

        def step(self, joints: object, timestamp_ns: int) -> tuple[ManipulationCommand, ManipulationStatus]:
            calls["step"].append(joints)
            return replace(command, timestamp_ns=timestamp_ns, valid_until_ns=timestamp_ns + 100), ManipulationStatus(
                LocalPhase.MOVE_PREGRASP, "RUNNING", 0.1, 0.0, False,
                "running", timestamp_ns,
            )

    node._arm_execution = Execution()
    node._fsm.phase_entered_ns = NOW
    status = FSMStatus(1, GlobalPhase.EXECUTE_PICK, LocalPhase.IDLE, 0, False, "", NOW)
    first_joints = _joints()
    second_joints = replace(_joints(), timestamp_ns=NOW + 1)
    first_snapshot = SensorSnapshot(_task(), _base(), first_joints, (), NOW, True)
    second_snapshot = SensorSnapshot(_task(), _base(), second_joints, (), NOW + 1, True)

    node._handle_phase_entry(first_snapshot, status, None, NOW)
    first_base, first_command = node._compute_candidate_commands(first_snapshot, status, NOW)
    second_base, second_command = node._compute_candidate_commands(second_snapshot, status, NOW + 1)
    assert calls["start"] == [trajectory]
    assert calls["step"] == [first_joints, second_joints]
    assert first_command is not None and second_command is not None
    assert (first_base.v, first_base.w) == (0.0, 0.0)
    assert (second_base.v, second_base.w) == (0.0, 0.0)


def test_execute_place_emits_motion_complete_once_without_place_verified() -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id, state.task_id, state.attempt = "run-1", 1, 0
    state.confirmed_grasp_context = _grasp_context(confirmed=True)
    trajectory = _trajectory(GlobalPhase.EXECUTE_PLACE)
    state.active_place_trajectory = trajectory
    state.active_trajectory_id = trajectory.trajectory_id
    calls: list[object] = []
    command = ManipulationCommand(
        (0.0,) * 17, (True, *([False] * 16)), LocalPhase.STOW,
        NOW, NOW + 100,
    )

    class Execution:
        def start_trajectory(self, value: object) -> ManipulationStatus:
            calls.append(value)
            return ManipulationStatus(
                LocalPhase.IDLE, "LOADED", 0.0, float("inf"), False,
                "loaded", NOW - 5,
            )

        def step(self, _joints: object, timestamp_ns: int) -> tuple[ManipulationCommand, ManipulationStatus]:
            return replace(
                command, timestamp_ns=timestamp_ns,
                valid_until_ns=timestamp_ns + 100,
            ), ManipulationStatus(
                LocalPhase.STOW,
                "MOTION_COMPLETED_PLACE_VERIFICATION_PENDING",
                1.0,
                0.0,
                True,
                "motion complete only",
                timestamp_ns,
            )

    node._arm_execution = Execution()
    node._fsm.phase_entered_ns = NOW
    status = FSMStatus(1, GlobalPhase.EXECUTE_PLACE, LocalPhase.IDLE, 0, False, "", NOW)
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)

    node._handle_phase_entry(snapshot, status, None, NOW)
    first = node._run_current_phase(snapshot, status, None, NOW)[2]
    second = node._run_current_phase(snapshot, status, None, NOW + 1)[2]
    assert calls == [trajectory]
    assert first.event is FSMEvent.PLACE_EXECUTED
    assert first.event is not FSMEvent.PLACE_VERIFIED
    assert second is None


@pytest.mark.parametrize(
    ("is_grasped", "expected_event"),
    [(True, FSMEvent.PICK_VERIFIED), (False, FSMEvent.PICK_FAILED)],
)
def test_verify_pick_maps_public_result_and_preserves_sensor_time(
    is_grasped: bool, expected_event: FSMEvent
) -> None:
    node = _node()
    state = node._runtime_wiring
    state.run_id, state.task_id, state.attempt = "run-1", 1, 0
    state.active_target = _estimate("target-a", 0.9)
    state.planned_grasp_context = _grasp_context()
    state.active_pick_trajectory = _trajectory(GlobalPhase.EXECUTE_PICK)
    state.active_trajectory_id = state.active_pick_trajectory.trajectory_id
    state.latest_grasp_verification_trajectory_id = state.active_trajectory_id
    state.pick_lift_started_ns = NOW - 10
    state.phase_entered_ns = NOW
    verification = GraspVerification(
        is_grasped, 0.9, "visual", "none", True, "", NOW - 5
    )
    node._arm_execution = SimpleNamespace(latest_grasp_verification=verification)
    status = FSMStatus(1, GlobalPhase.VERIFY_PICK, LocalPhase.IDLE, 0, False, "", NOW)
    snapshot = SensorSnapshot(_task(), _base(), _joints(), (), NOW, True)

    feedback = node._run_current_phase(snapshot, status, None, NOW)[2]
    assert feedback.event is expected_event
    assert feedback.source_timestamp_ns == NOW
    assert state.latest_grasp_verification.timestamp_ns == NOW - 5
    if is_grasped:
        assert state.confirmed_grasp_context.confirmed
        assert state.confirmed_grasp_context.confirmed_at_ns == NOW - 5
    else:
        assert state.confirmed_grasp_context is None


def test_verify_place_requires_bounded_stable_same_object_inside_radius() -> None:
    config = _config()
    config["perception"]["visual_observation_verifier"] = (
        _visual_verifier_values()
    )
    ros = _ros()
    node = _create_team_client_node(ros)(config, ros)
    state = node._runtime_wiring
    state.run_id, state.task_id, state.attempt = "run-1", 1, 0
    state.confirmed_grasp_context = _grasp_context(confirmed=True)
    state.active_trajectory_id = "place-test"
    state.place_release_or_completion_ns = NOW - 90
    state.phase_entered_ns = NOW - 85
    status = FSMStatus(1, GlobalPhase.VERIFY_PLACE, LocalPhase.IDLE, 0, False, "", NOW)
    timestamps = (NOW - 80, NOW - 60, NOW - 40)
    feedback = None
    for index, timestamp_ns in enumerate(timestamps):
        observation = replace(
            _estimate("target-a", 0.9, timestamp_ns=timestamp_ns),
            position_xyz=(1.0 + index * 0.001, 2.0, 3.0),
        )
        snapshot = SensorSnapshot(_task(), _base(), _joints(), (observation,), NOW, True)
        feedback = node._run_current_phase(snapshot, status, None, NOW)[2]
        if index < 2:
            assert feedback is None
    assert feedback.event is FSMEvent.PLACE_VERIFIED
    assert feedback.object_id == "target-a"
    assert len(state.place_post_release_observations) == 3
    assert state.place_post_release_observations.maxlen == 12


def test_internal_publish_authorization_accepts_only_bound_arm_execution() -> None:
    task = _task()
    context = CompetitionContext(
        "team_sorting.competition_context", 1, "run-1", "fingerprint", 1, 0,
        1.0, 0, (0, 0, 0), "-", False, task, NOW - 100, NOW - 50, True, "",
    )
    snapshot = SensorSnapshot(task, _base(), _joints(), (), NOW, True)
    status = FSMStatus(
        1, GlobalPhase.EXECUTE_PICK, LocalPhase.MOVE_PREGRASP, 0, False, "", NOW
    )
    base_command = BaseCommand(0.0, 0.0, NOW, NOW + 100)
    manipulation = ManipulationCommand(
        (0.0,) * 17, (True, *([False] * 16)), LocalPhase.MOVE_PREGRASP,
        NOW, NOW + 100,
    )
    final_action, _ = ActionMux().compose_with_decision(
        base_command, manipulation, _joints(), status, NOW
    )
    runtime = ros_nodes_module._RuntimeWiringState(
        run_id="run-1",
        task_id=1,
        attempt=0,
        last_handled_phase=GlobalPhase.EXECUTE_PICK,
        phase_generation=4,
        active_pick_trajectory=_trajectory(GlobalPhase.EXECUTE_PICK),
        active_trajectory_id="pick-test",
        active_trajectory_phase_generation=4,
        last_manipulation_status=ManipulationStatus(
            LocalPhase.MOVE_PREGRASP, "RUNNING", 0.1, 0.0, False, "running", NOW
        ),
        trajectory_started=True,
    )

    assert _internal_fsm_publish_authorization(
        observe_only=False,
        enable_official_publish=True,
        context=context,
        snapshot=snapshot,
        fsm_status=status,
        base_command=base_command,
        manipulation_command=manipulation,
        final_action=final_action,
        now_ns=NOW,
        runtime_wiring=runtime,
    )
    runtime.active_trajectory_id = "old"
    assert not _internal_fsm_publish_authorization(
        observe_only=False,
        enable_official_publish=True,
        context=context,
        snapshot=snapshot,
        fsm_status=status,
        base_command=base_command,
        manipulation_command=manipulation,
        final_action=final_action,
        now_ns=NOW,
        runtime_wiring=runtime,
    )


def test_arm_execution_control_result_requires_mux_acceptance_and_publish_success() -> None:
    joints = _joints()
    status = FSMStatus(
        1, GlobalPhase.EXECUTE_PICK, LocalPhase.MOVE_PREGRASP,
        0, False, "", NOW,
    )
    accepted = ManipulationCommand(
        joints.position, (True, *([False] * 16)), LocalPhase.MOVE_PREGRASP,
        NOW, NOW + 100,
    )
    _, accepted_decision = ActionMux().compose_with_decision(
        BaseCommand(0.0, 0.0, NOW, NOW + 100),
        accepted,
        joints,
        status,
        NOW,
    )
    rejected = replace(accepted, valid=False, failure_reason="injected reject")
    _, rejected_decision = ActionMux().compose_with_decision(
        None, rejected, joints, status, NOW
    )
    stop_status = replace(status, global_phase=GlobalPhase.SAFE_HOLD)
    _, stop_decision = ActionMux().compose_with_decision(
        None, accepted, joints, stop_status, NOW
    )

    common = {
        "manipulation_command": accepted,
        "manipulation_source": "manipulation_command",
        "internal_publish_authorized": True,
    }
    assert not _arm_execution_control_granted(
        **common, fsm_status=status, mux_decision=rejected_decision,
        published=True,
    )
    assert not _arm_execution_control_granted(
        **common, fsm_status=stop_status, mux_decision=stop_decision,
        published=True,
    )
    assert not _arm_execution_control_granted(
        **common, fsm_status=status, mux_decision=accepted_decision,
        published=False,
    )
    assert _arm_execution_control_granted(
        **common, fsm_status=status, mux_decision=accepted_decision,
        published=True,
    )


def test_external_candidate_never_grants_arm_execution_clock() -> None:
    joints = _joints()
    status = FSMStatus(
        1, GlobalPhase.EXECUTE_PICK, LocalPhase.MOVE_PREGRASP,
        0, False, "", NOW,
    )
    command = ManipulationCommand(
        joints.position, (True, *([False] * 16)), LocalPhase.MOVE_PREGRASP,
        NOW, NOW + 100,
    )
    _, decision = ActionMux().compose_with_decision(
        None, command, joints, status, NOW,
        manipulation_source="external_candidate",
    )

    assert not _arm_execution_control_granted(
        fsm_status=status,
        manipulation_command=command,
        mux_decision=decision,
        manipulation_source="external_candidate",
        internal_publish_authorized=True,
        published=True,
    )


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
