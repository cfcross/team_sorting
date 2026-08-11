from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PATH = Path("scripts/analyze_m10_b0_scheduler.py").resolve()
SPEC = importlib.util.spec_from_file_location("m10_b05_analysis", PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def _candidate(receipt_ns=0, request_id=1, identity=None, value=0.0, valid=True):
    identity = identity or ("run", 1, 0, "instruction", "set", "task", "generation")
    actions = tuple(
        tuple([value + index] * analysis.ACTION_DIM)
        for index in range(analysis.HORIZON)
    )
    return analysis.Candidate(
        receipt_ns, request_id, str(identity[-1]), identity, actions, 100.0, valid
    )


def test_20hz_skips_policy_steps_while_24hz_tracks_each_step():
    candidate = _candidate()
    end = 14 * analysis.POLICY_PERIOD_NS
    slow = analysis.scheduler_metrics(
        analysis.simulate([candidate], analysis.make_ticks(0, end, 20)), 1
    )
    matched = analysis.scheduler_metrics(
        analysis.simulate([candidate], analysis.make_ticks(0, end, 24)), 1
    )
    assert slow["skipped_action_steps"] > 0
    assert matched["skipped_action_steps"] == 0


def test_50hz_ideal_has_no_systematic_skip_and_repeats_indices():
    candidate = _candidate()
    end = 14 * analysis.POLICY_PERIOD_NS
    metrics = analysis.scheduler_metrics(
        analysis.simulate([candidate], analysis.make_ticks(0, end, 50)), 1
    )
    assert metrics["skipped_action_steps"] == 0
    assert metrics["repeated_action_indices"] > 0


def test_jitter_schedule_is_reproducible_and_seeded():
    first = analysis.make_ticks(0, 1_000_000_000, 50, jitter_ms=5, seed=7)
    second = analysis.make_ticks(0, 1_000_000_000, 50, jitter_ms=5, seed=7)
    third = analysis.make_ticks(0, 1_000_000_000, 50, jitter_ms=5, seed=8)
    assert first == second
    assert first != third


def test_immediate_replacement_starts_new_chunk_without_future_index_search():
    period = analysis.POLICY_PERIOD_NS
    candidates = [_candidate(0, 1, value=0), _candidate(period + 1, 2, value=100)]
    outputs = analysis.simulate(candidates, [0, period, period + 1], handoff="immediate")
    assert outputs[-1].request_id == 2
    assert outputs[-1].action_index == 0
    assert outputs[-1].action[0] == 100


def test_next_step_boundary_delays_handoff_and_starts_at_index_zero():
    period = analysis.POLICY_PERIOD_NS
    candidates = [_candidate(0, 1), _candidate(period + 1, 2, value=100)]
    ticks = [0, period, period + 1, period * 2]
    outputs = analysis.simulate(candidates, ticks, handoff="next-step-boundary")
    assert outputs[2].request_id == 1
    assert outputs[3].request_id == 2 and outputs[3].action_index == 0


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_crossfade_one_two_three_steps(steps):
    period = analysis.POLICY_PERIOD_NS
    candidates = [_candidate(0, 1), _candidate(period, 2, value=100)]
    ticks = [0, period, period * 2, period * 3, period * 4]
    outputs = analysis.simulate(candidates, ticks, handoff=f"crossfade-{steps}")
    at_handoff = outputs[1].action
    assert at_handoff is not None and at_handoff[0] == outputs[0].action[0]
    completed = outputs[min(1 + steps, len(outputs) - 1)]
    assert completed.action is not None and completed.action[0] >= 100
    assert completed.action[11] >= 100  # grippers are never interpolated


@pytest.mark.parametrize(
    "old_identity,new_identity",
    [
        (("run", 1, 0, "i", "s", "t", "g"), ("other", 1, 0, "i", "s", "t", "g")),
        (("run", 1, 0, "i", "s", "t", "g1"), ("run", 1, 0, "i", "s", "t", "g2")),
    ],
)
def test_context_or_generation_change_forbids_blend(old_identity, new_identity):
    old = _candidate(identity=old_identity).actions[0]
    new = _candidate(identity=new_identity, value=100).actions[0]
    assert analysis.blend_action(old, new, 0.1, identity_same=False) == new


def test_invalid_candidate_and_stale_watchdog_fail_closed_immediately():
    old = _candidate().actions[0]
    new = _candidate(value=100).actions[0]
    with pytest.raises(ValueError, match="fail_closed"):
        analysis.blend_action(old, new, 0.5, identity_same=True, candidate_valid=False)
    assert analysis.blend_action(old, new, 0.1, identity_same=True, immediate_safe=True) == new
    invalid = _candidate(valid=False)
    assert analysis.simulate([invalid], [0])[0].action is None
    expired = analysis.simulate([_candidate()], [700_000_000], ttl_ms=700)
    assert expired[0].action is None


@pytest.mark.parametrize(
    "actions,reason",
    [
        ([[0.0] * 19] * 14, "15_rows"),
        ([[0.0] * 18] * 15, "19d"),
        ([[True] + [0.0] * 18] + [[0.0] * 19] * 14, "real_not_bool"),
        ([[float("nan")] + [0.0] * 18] + [[0.0] * 19] * 14, "finite"),
    ],
)
def test_chunk_shape_and_finite_validation(actions, reason):
    with pytest.raises(ValueError, match=reason):
        analysis.validate_action_chunk(actions)
