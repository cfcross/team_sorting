import json
import inspect
from pathlib import Path

import pytest
import yaml

from team_sorting.action_mux import ActionMux
from team_sorting.competition_context import CompetitionContext
from team_sorting.fsm import InstructionParser
from team_sorting.interfaces import (
    BaseCommand, FSMStatus, GlobalPhase, LocalPhase, RobotJointState,
)
from team_sorting.pi05_policy_control import (
    HANDOFF_POLICY_STEPS, PolicyControlConfig, PolicyControlConsumer,
    PolicyControlDecoder,
)
from team_sorting.ros_nodes import (
    _create_team_client_node,
    _official_publish_enabled,
    _policy_control_subscription_enabled,
)


RAW_TASKS = json.dumps([
    {"task": 1, "instruction": "task one", "target_kind": "cuboid_box", "target_body": "box_1", "target_color": "pink", "place_type": "shelf_point", "place_world": [1.0, 2.0, 3.0], "place_radius": 0.24},
    {"task": 2, "instruction": "task two", "target_kind": "cuboid_box", "target_body": "box_2", "target_color": "brown", "place_type": "table_point", "place_world": [2.0, 2.0, 3.0], "place_radius": 0.28},
    {"task": 3, "instruction": "task three", "target_kind": "cuboid_box", "target_body": "box_3", "target_color": "yellow", "place_type": "shelf_prop_side", "place_world": [3.0, 2.0, 3.0], "place_radius": 0.24, "ref_prop": "packaging_box", "ref_prop_body": "prop", "direction": "left"},
])


def _config(**overrides):
    values = dict(
        enabled=True, enable_actuation=True, simulation_only=True,
        simulation_publish_enabled=True, action_step_period_s=0.0416666667,
        max_policy_response_latency_ms=625.0, candidate_ttl_ms=700.0,
        watchdog_timeout_ms=625.0,
    )
    values.update(overrides)
    return PolicyControlConfig(**values)


def _context(task_id=1, attempt=0, run_id="run-a", valid=True, finished=False):
    task = InstructionParser().parse(RAW_TASKS, 10)[task_id - 1]
    return CompetitionContext(
        schema_name="team_sorting.competition_context", schema_version=1,
        run_id=run_id, task_set_fingerprint="set", current_task_id=task_id,
        current_attempt_count=attempt, elapsed_sim_s=1.0, score=0,
        best_scores=(0, 0, 0), current_step="-", finished=finished,
        active_task=task, instruction_timestamp_ns=10, referee_timestamp_ns=20,
        valid=valid, failure_reason="" if valid else "invalid_context",
    )


def _status(phase=GlobalPhase.SEARCH_TARGET):
    return FSMStatus(1, phase, LocalPhase.IDLE, 0, phase is GlobalPhase.DONE, "", 1)


def _raw(consumer, request_id=1, generation="process:1", actions=None, **overrides):
    identity = consumer._context_key
    assert identity is not None
    rows = actions or [[float(row), 0.2, *([0.1] * 17)] for row in range(15)]
    values = {
        "schema_name": "MMK2Pi05PolicyControlCandidate", "schema_version": 1,
        "request_id": request_id, "generation_id": generation,
        "run_id": identity[0], "episode_id": identity[0], "task_id": identity[1],
        "attempt_count": identity[2], "instruction_fingerprint": identity[3],
        "task_set_fingerprint": identity[4], "active_task_fingerprint": identity[5],
        "model_id": "pi05_mmk2_task1_lora", "action_horizon": 15,
        "action_dim": 19, "actions": rows, "response_latency_ms": 100.0,
        "context_valid": True, "valid": True, "failure_reason": "",
        "published_to_robot": False,
    }
    values.update(overrides)
    return json.dumps(values, allow_nan=True)


def _consumer(config=None):
    consumer = PolicyControlConsumer(config or _config())
    consumer.update_context(_context())
    return consumer


def _decision_action(decision):
    assert decision.accepted
    assert decision.base_command is not None
    assert decision.manipulation_command is not None
    return (
        decision.base_command.v,
        decision.base_command.w,
        *decision.manipulation_command.joint_target,
    )


def _constant_actions(value, left_grip=0.25, right_grip=0.75):
    rows = []
    for index in range(15):
        row = [float(value + index)] * 19
        row[11] = left_grip
        row[18] = right_grip
        rows.append(row)
    return rows


def _assert_action_state_cleared(consumer):
    assert not consumer.pending
    assert consumer._last_output is None
    assert consumer._last_output_identity is None
    assert consumer._handoff_source is None
    assert consumer._handoff_start_ros_ns is None
    assert consumer._handoff_identity is None


def test_default_has_no_subscription_and_enabled_requires_explicit_timing():
    assert not _policy_control_subscription_enabled(PolicyControlConfig())
    with pytest.raises(ValueError, match="explicit"):
        PolicyControlConfig(enabled=True)
    with pytest.raises(ValueError, match="simulation_only"):
        _config(simulation_only=False)
    loaded = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    timing = loaded["pi05_policy_control"]
    assert timing["action_step_period_s"] == 0.0416666667
    assert timing["max_policy_response_latency_ms"] == 625.0
    assert timing["candidate_ttl_ms"] == 700.0
    assert timing["watchdog_timeout_ms"] == 625.0


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"model_id": "wrong"}, "model_id_mismatch"),
        ({"task_id": 2}, "task_not_allowed"),
        ({"actions": [[0.0] * 19] * 14}, "15x19"),
        ({"actions": [[True] + [0.0] * 18] + [[0.0] * 19] * 14}, "bool"),
        ({"actions": [[float("nan")] + [0.0] * 18] + [[0.0] * 19] * 14}, "non_finite"),
        ({"unknown": 1}, "fields"),
    ],
)
def test_strict_schema_shape_finite_model_and_task(override, reason):
    consumer = _consumer()
    decision = consumer.receive(_raw(consumer, **override), 1_000_000_000)
    assert not decision.accepted and reason in decision.failure_reason


def test_24hz_chunk_schedule_mapping_and_receding_horizon_replacement():
    consumer = _consumer()
    start = 1_000_000_000
    assert consumer.receive(_raw(consumer), start).accepted
    period = consumer.config.ns("action_step_period_s")
    first = consumer.take(now_ns=start, fsm_status=_status())
    second = consumer.take(now_ns=start + period, fsm_status=_status())
    assert (first.action_index, second.action_index) == (0, 1)
    assert first.base_command.v == 0.0 and second.base_command.v == 1.0
    assert first.manipulation_command.joint_target == (0.1,) * 17
    assert first.manipulation_command.controlled_mask == (True,) * 17

    replacement = [[99.0, -1.0, *([0.2] * 17)] for _ in range(15)]
    assert consumer.receive(_raw(consumer, request_id=2, actions=replacement), start + period).accepted
    replaced = consumer.take(now_ns=start + period, fsm_status=_status())
    assert replaced.request_id == 2 and replaced.action_index == 0
    assert replaced.base_command.v == second.base_command.v


def test_40hz_consumer_zero_order_holds_24hz_indices_without_systematic_skip():
    consumer = _consumer()
    start = 1_000_000_000
    assert consumer.receive(_raw(consumer), start).accepted
    indices = [
        consumer.take(now_ns=start + offset_ms * 1_000_000, fsm_status=_status()).action_index
        for offset_ms in (0, 25, 50, 75, 100, 125, 150)
    ]
    assert indices == [0, 0, 1, 1, 2, 2, 3]
    assert all(right - left in (0, 1) for left, right in zip(indices, indices[1:]))


def test_two_step_handoff_uses_last_emitted_output_and_tracks_new_24hz_rows():
    consumer = _consumer()
    start = 1_000_000_000
    period = consumer.config.ns("action_step_period_s")
    old = _constant_actions(10.0, left_grip=0.1, right_grip=0.2)
    new = _constant_actions(100.0, left_grip=0.8, right_grip=0.9)
    assert HANDOFF_POLICY_STEPS == 2
    assert consumer.receive(_raw(consumer, actions=old), start).accepted
    last = consumer.take(now_ns=start, fsm_status=_status())
    last_action = _decision_action(last)

    handoff_start = start + period
    assert consumer.receive(
        _raw(consumer, request_id=2, actions=new), handoff_start
    ).accepted
    first = consumer.take(now_ns=handoff_start, fsm_status=_status())
    halfway = consumer.take(now_ns=handoff_start + period, fsm_status=_status())
    complete = consumer.take(now_ns=handoff_start + 2 * period, fsm_status=_status())

    first_action = _decision_action(first)
    halfway_action = _decision_action(halfway)
    complete_action = _decision_action(complete)
    assert (first.action_index, halfway.action_index, complete.action_index) == (0, 1, 2)
    assert first_action[:11] == last_action[:11]
    assert first_action[12:18] == last_action[12:18]
    assert halfway_action[0] == pytest.approx((last_action[0] + 101.0) / 2.0)
    assert complete_action[0] == 102.0
    assert first_action[11] == halfway_action[11] == complete_action[11] == 0.8
    assert first_action[18] == halfway_action[18] == complete_action[18] == 0.9
    assert halfway.base_command.valid_until_ns == handoff_start + 2 * period


def test_last_policy_step_valid_until_never_exceeds_formal_watchdog_deadline():
    consumer = _consumer()
    received = 1_000_000_000
    period = consumer.config.ns("action_step_period_s")
    watchdog_deadline = received + 625_000_000
    assert received + 15 * period == watchdog_deadline + 5
    assert consumer.receive(_raw(consumer), received).accepted

    decision = consumer.take(
        now_ns=received + 14 * period, fsm_status=_status()
    )

    assert decision.accepted and decision.action_index == 14
    assert decision.base_command.valid_until_ns == watchdog_deadline
    assert decision.manipulation_command.valid_until_ns == watchdog_deadline


def test_valid_until_is_limited_by_watchdog_shorter_than_one_policy_step():
    consumer = _consumer(
        _config(candidate_ttl_ms=100.0, watchdog_timeout_ms=10.0)
    )
    received = 1_000_000_000
    assert consumer.receive(_raw(consumer), received).accepted

    decision = consumer.take(now_ns=received, fsm_status=_status())

    assert decision.base_command.valid_until_ns == received + 10_000_000
    assert decision.manipulation_command.valid_until_ns == received + 10_000_000


def test_valid_until_is_limited_by_candidate_ttl_when_it_is_earliest():
    consumer = _consumer(
        _config(candidate_ttl_ms=10.0, watchdog_timeout_ms=30.0)
    )
    received = 1_000_000_000
    assert consumer.receive(_raw(consumer), received).accepted

    decision = consumer.take(now_ns=received, fsm_status=_status())

    assert decision.base_command.valid_until_ns == received + 10_000_000
    assert decision.manipulation_command.valid_until_ns == received + 10_000_000


def test_valid_until_is_limited_by_current_policy_step_when_it_is_earliest():
    consumer = _consumer(
        _config(candidate_ttl_ms=100.0, watchdog_timeout_ms=100.0)
    )
    received = 1_000_000_000
    period = consumer.config.ns("action_step_period_s")
    assert consumer.receive(_raw(consumer), received).accepted

    decision = consumer.take(now_ns=received, fsm_status=_status())

    assert decision.base_command.valid_until_ns == received + period
    assert decision.manipulation_command.valid_until_ns == received + period


def test_generation_or_context_change_never_crossfades():
    start = 1_000_000_000
    for change in ("generation", "context"):
        consumer = _consumer()
        assert consumer.receive(
            _raw(consumer, actions=_constant_actions(10.0)), start
        ).accepted
        consumer.take(now_ns=start, fsm_status=_status())
        if change == "context":
            consumer.update_context(_context(attempt=1))
            generation = "process:1"
        else:
            generation = "process:2"
        assert consumer.receive(
            _raw(
                consumer, request_id=2, generation=generation,
                actions=_constant_actions(100.0),
            ),
            start + 1,
        ).accepted
        decision = consumer.take(now_ns=start + 1, fsm_status=_status())
        assert _decision_action(decision)[0] == 100.0
        assert consumer._handoff_source is None


def test_invalid_or_replayed_candidate_clears_pending_handoff_and_last_output():
    for failure in ("invalid", "replay"):
        consumer = _consumer()
        start = 1_000_000_000
        assert consumer.receive(_raw(consumer), start).accepted
        consumer.take(now_ns=start, fsm_status=_status())
        assert consumer.receive(_raw(consumer, request_id=2), start + 1).accepted
        assert consumer._handoff_source is not None
        if failure == "invalid":
            decision = consumer.receive("{}", start + 2)
        else:
            decision = consumer.receive(_raw(consumer, request_id=2), start + 2)
        assert not decision.accepted
        _assert_action_state_cleared(consumer)


@pytest.mark.parametrize(
    ("config", "offset_ns", "reason"),
    [
        (_config(candidate_ttl_ms=500.0, watchdog_timeout_ms=600.0), 500_000_000, "candidate_expired"),
        (_config(candidate_ttl_ms=700.0, watchdog_timeout_ms=625.0), 625_000_000, "policy_watchdog_timeout"),
        (_config(candidate_ttl_ms=1000.0, watchdog_timeout_ms=1000.0), 15 * 41_666_667, "policy_chunk_exhausted"),
    ],
)
def test_ttl_watchdog_and_chunk_exhaustion_immediately_clear_blend(config, offset_ns, reason):
    consumer = _consumer(config)
    start = 1_000_000_000
    assert consumer.receive(_raw(consumer), start).accepted
    consumer.take(now_ns=start, fsm_status=_status())
    assert consumer.receive(_raw(consumer, request_id=2), start + 1).accepted
    decision = consumer.take(now_ns=start + 1 + offset_ns, fsm_status=_status())
    assert not decision.accepted and decision.failure_reason == reason
    _assert_action_state_cleared(consumer)


@pytest.mark.parametrize("failure", ["future", "invalid_time", "fsm", "conflict"])
def test_time_fsm_and_exclusive_failures_immediately_clear_blend(failure):
    consumer = _consumer(_config(candidate_ttl_ms=1000.0, watchdog_timeout_ms=1000.0))
    start = 1_000_000_000
    assert consumer.receive(_raw(consumer), start).accepted
    consumer.take(now_ns=start, fsm_status=_status())
    received = start + 100
    assert consumer.receive(_raw(consumer, request_id=2), received).accepted
    kwargs = {"now_ns": received, "fsm_status": _status()}
    if failure == "future":
        kwargs["now_ns"] = received - 1
    elif failure == "invalid_time":
        kwargs["now_ns"] = -1
    elif failure == "fsm":
        kwargs["fsm_status"] = _status(GlobalPhase.SAFE_HOLD)
    else:
        kwargs["existing_base"] = BaseCommand(0.0, 0.0, received, received + 1)
    decision = consumer.take(**kwargs)
    assert not decision.accepted
    _assert_action_state_cleared(consumer)


def test_invalidate_and_shutdown_immediately_clear_all_action_state():
    for operation in ("invalidate", "shutdown"):
        consumer = _consumer()
        start = 1_000_000_000
        assert consumer.receive(_raw(consumer), start).accepted
        consumer.take(now_ns=start, fsm_status=_status())
        assert consumer.receive(_raw(consumer, request_id=2), start + 1).accepted
        getattr(consumer, operation)()
        _assert_action_state_cleared(consumer)


def test_replay_context_transitions_stop_phase_and_shutdown_clear_chunk():
    consumer = _consumer()
    raw = _raw(consumer, request_id=5)
    assert consumer.receive(raw, 100).accepted
    assert not consumer.receive(raw, 101).accepted
    consumer.update_context(_context(attempt=1))
    assert not consumer.pending
    for request_id, context in enumerate(
        (_context(task_id=2), _context(run_id="run-b"), _context(valid=False)), 6
    ):
        consumer.update_context(_context())
        assert consumer.receive(_raw(consumer, request_id=request_id), 200).accepted
        consumer.update_context(context)
        assert not consumer.pending
    consumer.update_context(_context())
    assert consumer.receive(_raw(consumer, request_id=9), 300).accepted
    assert consumer.take(now_ns=300, fsm_status=_status(GlobalPhase.SAFE_HOLD)).failure_reason == "fsm_stop_phase"
    assert consumer.receive(_raw(consumer, request_id=10), 400).accepted
    consumer.shutdown()
    assert not consumer.pending
    assert consumer.take(now_ns=400, fsm_status=_status()).failure_reason == "consumer_shutdown"


def test_new_generation_replaces_old_and_episode_must_match_run():
    consumer = _consumer()
    assert consumer.receive(_raw(consumer, request_id=1), 100).accepted
    assert consumer.receive(
        _raw(consumer, request_id=2, generation="process:2"), 200
    ).accepted
    decision = consumer.take(now_ns=200, fsm_status=_status())
    assert decision.generation_id == "process:2" and decision.action_index == 0
    mismatch = consumer.receive(
        _raw(consumer, request_id=3, episode_id="other-run"), 300
    )
    assert not mismatch.accepted and "identity_mismatch" in mismatch.failure_reason


@pytest.mark.parametrize(
    "gate",
    ["enabled", "enable_actuation", "simulation_publish_enabled"],
)
def test_each_independent_policy_publish_gate_is_required(gate):
    values = dict(
        enabled=True, enable_actuation=True, simulation_only=True,
        simulation_publish_enabled=True,
    )
    values[gate] = False
    assert not _config(**values).publish_authorized


@pytest.mark.parametrize(
    "control",
    [
        {"observe_only": True, "enable_official_publish": True, "simulation_only": True},
        {"observe_only": False, "enable_official_publish": False, "simulation_only": True},
    ],
)
def test_global_publish_gate_remains_independently_required(control):
    assert not _official_publish_enabled(control)


def test_ros_path_uses_action_mux_before_distinct_policy_full_authorization():
    source = inspect.getsource(_create_team_client_node)
    mux = source.index("self._mux.compose_with_decision(")
    publish = source.index("policy_control_publish_authorized=(", mux)
    assert mux < publish
    assert "elif policy_control_publish_authorized:" in source
    assert "self._official_publisher.publish_with_trace(action)" in source


@pytest.mark.parametrize(
    ("field", "reason"),
    [("candidate_ttl_ms", "candidate_expired"), ("watchdog_timeout_ms", "policy_watchdog_timeout")],
)
def test_ttl_and_watchdog_equality_are_expired(field, reason):
    values = {"candidate_ttl_ms": 200.0, "watchdog_timeout_ms": 200.0}
    values[field] = 100.0
    consumer = _consumer(_config(**values))
    start = 1_000_000_000
    assert consumer.receive(_raw(consumer), start).accepted
    decision = consumer.take(now_ns=start + 100_000_000, fsm_status=_status())
    assert not decision.accepted and decision.failure_reason == reason


def test_action_mux_remains_final_assembler_and_clips_policy_values():
    consumer = _consumer()
    start = 1_000_000_000
    actions = [[99.0, 9.0, *([9.0] * 17)] for _ in range(15)]
    assert consumer.receive(_raw(consumer, actions=actions), start).accepted
    decision = consumer.take(now_ns=start, fsm_status=_status())
    joints = RobotJointState((0.0,) * 17, (0.0,) * 17, (0.0,) * 17, start)
    action, mux = ActionMux().compose_with_decision(
        decision.base_command, decision.manipulation_command, joints, _status(), start,
        base_source="pi05_policy_control", manipulation_source="pi05_policy_control",
    )
    assert action.values[0] == 0.25 and action.values[1] == 0.5
    assert action.clipped
    assert mux.base_source == mux.manipulation_source == "pi05_policy_control"
