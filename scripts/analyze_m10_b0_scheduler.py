#!/usr/bin/env python3
"""Read-only M10-B0 rosbag scheduler analysis and offline strategy simulation.

This module intentionally uses only sqlite3 and the Python standard library.  It
does not initialize ROS, publish, subscribe, or modify the input bag.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import struct
from typing import Iterable, Optional, Sequence


POLICY_TOPIC = "/mmk2_pi05_adapter/policy_control_candidate"
DISPATCH_TOPIC = "/team/action_dispatch"
FINAL_TOPIC = "/team/final_action"
CONTEXT_TOPIC = "/team/competition_context"
TOPICS = (POLICY_TOPIC, DISPATCH_TOPIC, FINAL_TOPIC, CONTEXT_TOPIC)
ACTION_DIM = 19
HORIZON = 15
POLICY_PERIOD_S = 1.0 / 24.0
POLICY_PERIOD_NS = round(1_000_000_000 / 24)
GRIPPER_INDICES = frozenset({11, 18})
DIMENSIONS = (
    "base_v", "base_w", "slide", "head_yaw", "head_pitch",
    "left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6",
    "left_grip", "right_j1", "right_j2", "right_j3", "right_j4",
    "right_j5", "right_j6", "right_grip",
)


@dataclass(frozen=True)
class Candidate:
    receipt_ns: int
    request_id: int
    generation_id: str
    identity: tuple[object, ...]
    actions: tuple[tuple[float, ...], ...]
    response_latency_ms: float
    valid: bool = True


@dataclass(frozen=True)
class TickOutput:
    timestamp_ns: int
    request_id: Optional[int]
    action_index: Optional[int]
    action: Optional[tuple[float, ...]]
    handoff: bool = False
    hold_reason: str = ""


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Sequence[float]) -> dict[str, Optional[float]]:
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def validate_action(action: object) -> tuple[float, ...]:
    if not isinstance(action, (list, tuple)) or len(action) != ACTION_DIM:
        raise ValueError("action_must_be_19d")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in action):
        raise ValueError("action_values_must_be_real_not_bool")
    result = tuple(float(value) for value in action)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("action_values_must_be_finite")
    return result


def validate_action_chunk(actions: object) -> tuple[tuple[float, ...], ...]:
    if not isinstance(actions, (list, tuple)) or len(actions) != HORIZON:
        raise ValueError("action_chunk_must_have_15_rows")
    return tuple(validate_action(row) for row in actions)


def decode_cdr_string(blob: bytes) -> str:
    if len(blob) < 9 or blob[0:2] not in (b"\x00\x00", b"\x00\x01"):
        raise ValueError("unsupported_cdr_encapsulation")
    endian = "<" if blob[1] == 1 else ">"
    length = struct.unpack_from(f"{endian}I", blob, 4)[0]
    if length < 1 or 8 + length > len(blob) or blob[8 + length - 1] != 0:
        raise ValueError("invalid_cdr_string_length")
    return blob[8 : 8 + length - 1].decode("utf-8")


def _strict_json(raw: str) -> dict[str, object]:
    value = json.loads(
        raw,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non_finite_json:{token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("json_payload_must_be_object")
    return value


def read_bag(db_path: Path) -> dict[str, list[tuple[int, dict[str, object]]]]:
    result = {topic: [] for topic in TOPICS}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT t.name, m.timestamp, m.data FROM messages m "
            "JOIN topics t ON t.id=m.topic_id "
            "WHERE t.name IN (?,?,?,?) ORDER BY m.timestamp",
            TOPICS,
        )
        for topic, timestamp, blob in rows:
            result[topic].append((int(timestamp), _strict_json(decode_cdr_string(blob))))
    missing = [topic for topic, rows in result.items() if not rows]
    if missing:
        raise ValueError(f"bag_missing_required_topics:{missing}")
    return result


def parse_candidates(rows: Iterable[tuple[int, dict[str, object]]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for timestamp, payload in rows:
        if payload.get("action_horizon") != HORIZON or payload.get("action_dim") != ACTION_DIM:
            raise ValueError("candidate_shape_contract_not_15x19")
        actions = validate_action_chunk(payload.get("actions"))
        latency = payload.get("response_latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            raise ValueError("invalid_response_latency_ms")
        latency_float = float(latency)
        if not math.isfinite(latency_float) or latency_float < 0:
            raise ValueError("invalid_response_latency_ms")
        identity = (
            payload.get("run_id"), payload.get("task_id"), payload.get("attempt_count"),
            payload.get("instruction_fingerprint"), payload.get("task_set_fingerprint"),
            payload.get("active_task_fingerprint"), payload.get("generation_id"),
        )
        candidates.append(
            Candidate(
                receipt_ns=timestamp,
                request_id=int(payload["request_id"]),
                generation_id=str(payload["generation_id"]),
                identity=identity,
                actions=actions,
                response_latency_ms=latency_float,
                valid=(
                    payload.get("context_valid") is True
                    and payload.get("valid") is True
                    and payload.get("failure_reason") == ""
                    and payload.get("published_to_robot") is False
                ),
            )
        )
    return candidates


def make_ticks(
    start_ns: int,
    end_ns: int,
    rate_hz: float,
    *,
    phase_fraction: float = 0.0,
    jitter_ms: float = 0.0,
    seed: int = 0,
) -> list[int]:
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz_must_be_positive_finite")
    if not 0.0 <= phase_fraction < 1.0:
        raise ValueError("phase_fraction_must_be_in_0_1")
    interval = round(1_000_000_000.0 / rate_hz)
    rng = random.Random(seed)
    ticks: list[int] = []
    index = 0
    while True:
        nominal = start_ns + phase_fraction * interval + index * interval
        jitter = rng.uniform(-jitter_ms, jitter_ms) * 1_000_000 if jitter_ms else 0.0
        timestamp = int(round(nominal + jitter))
        if timestamp > end_ns:
            break
        if timestamp >= start_ns and (not ticks or timestamp > ticks[-1]):
            ticks.append(timestamp)
        index += 1
    return ticks


def blend_action(
    previous: Sequence[float],
    target: Sequence[float],
    alpha: float,
    *,
    identity_same: bool,
    candidate_valid: bool = True,
    immediate_safe: bool = False,
    smooth_grippers: bool = False,
) -> tuple[float, ...]:
    old = validate_action(previous)
    new = validate_action(target)
    if immediate_safe:
        return new
    if not candidate_valid:
        raise ValueError("invalid_candidate_must_fail_closed")
    if not identity_same:
        return new
    if not math.isfinite(alpha):
        raise ValueError("alpha_must_be_finite")
    weight = min(1.0, max(0.0, float(alpha)))
    return tuple(
        new[index]
        if index in GRIPPER_INDICES and not smooth_grippers
        else old[index] + weight * (new[index] - old[index])
        for index in range(ACTION_DIM)
    )


def simulate(
    candidates: Sequence[Candidate],
    ticks: Sequence[int],
    *,
    handoff: str = "immediate",
    ttl_ms: float = 700.0,
    latency_limit_ms: float = 1000.0,
) -> list[TickOutput]:
    if handoff not in {"immediate", "next-step-boundary", "crossfade-1", "crossfade-2", "crossfade-3"}:
        raise ValueError("unsupported_handoff")
    accepted = [item for item in candidates if item.valid and item.response_latency_ms <= latency_limit_ms]
    outputs: list[TickOutput] = []
    cursor = 0
    active: Optional[Candidate] = None
    active_start_ns = 0
    queued: Optional[Candidate] = None
    queued_start_ns = 0
    last_action: Optional[tuple[float, ...]] = None
    fade_from: Optional[tuple[float, ...]] = None
    fade_start_ns = 0
    fade_steps = int(handoff[-1]) if handoff.startswith("crossfade-") else 0
    anchor_ns = ticks[0] if ticks else 0
    for tick_ns in ticks:
        arriving: list[Candidate] = []
        while cursor < len(accepted) and accepted[cursor].receipt_ns <= tick_ns:
            arriving.append(accepted[cursor])
            cursor += 1
        for candidate in arriving:
            if handoff == "next-step-boundary":
                boundary_number = max(0, math.ceil((candidate.receipt_ns - anchor_ns) / POLICY_PERIOD_NS))
                queued = candidate
                queued_start_ns = anchor_ns + boundary_number * POLICY_PERIOD_NS
            else:
                same_identity = active is None or candidate.identity == active.identity
                fade_from = last_action if fade_steps and same_identity else None
                fade_start_ns = candidate.receipt_ns
                active = candidate
                active_start_ns = candidate.receipt_ns
        handoff_now = False
        if queued is not None and tick_ns >= queued_start_ns:
            active = queued
            active_start_ns = queued_start_ns
            queued = None
            handoff_now = True
        elif arriving and handoff != "next-step-boundary":
            handoff_now = True
        if active is None:
            outputs.append(TickOutput(tick_ns, None, None, None, hold_reason="no_candidate"))
            last_action = None
            continue
        age_ns = tick_ns - active_start_ns
        index = age_ns // POLICY_PERIOD_NS
        if age_ns < 0 or age_ns >= round(ttl_ms * 1_000_000) or index >= HORIZON:
            reason = "candidate_ttl" if age_ns >= round(ttl_ms * 1_000_000) else "chunk_exhausted"
            outputs.append(TickOutput(tick_ns, None, None, None, hold_reason=reason))
            last_action = None
            continue
        target = active.actions[int(index)]
        action = target
        if fade_steps and fade_from is not None:
            alpha = (tick_ns - fade_start_ns) / (fade_steps * POLICY_PERIOD_NS)
            action = blend_action(
                fade_from, target, alpha, identity_same=True, smooth_grippers=False
            )
            if alpha >= 1.0:
                fade_from = None
        outputs.append(TickOutput(tick_ns, active.request_id, int(index), action, handoff_now))
        last_action = action
    return outputs


def _max_delta(left: Sequence[float], right: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def scheduler_metrics(outputs: Sequence[TickOutput], total_chunks: int) -> dict[str, object]:
    policy = [item for item in outputs if item.action is not None]
    seen: dict[int, list[int]] = {}
    skipped = 0
    repeated = 0
    boundary_deltas: list[float] = []
    previous: Optional[TickOutput] = None
    for item in policy:
        assert item.request_id is not None and item.action_index is not None
        indices = seen.setdefault(item.request_id, [])
        if not indices:
            skipped += item.action_index
        else:
            difference = item.action_index - indices[-1]
            skipped += max(0, difference - 1)
            repeated += int(difference == 0)
        indices.append(item.action_index)
        if previous is not None and previous.request_id != item.request_id:
            assert previous.action is not None and item.action is not None
            boundary_deltas.append(_max_delta(previous.action, item.action))
        previous = item
    unique_pairs = sum(len(set(indices)) for indices in seen.values())
    index0 = sum(0 in indices for indices in seen.values())
    first_gt_zero = sum(bool(indices) and indices[0] > 0 for indices in seen.values())
    denominator = unique_pairs + skipped
    coverage = [len(set(indices)) / HORIZON for indices in seen.values()]
    holds = len(outputs) - len(policy)
    hold_reasons: dict[str, int] = {}
    for item in outputs:
        if item.action is None:
            hold_reasons[item.hold_reason] = hold_reasons.get(item.hold_reason, 0) + 1
    return {
        "total_control_ticks": len(outputs),
        "policy_ticks": len(policy),
        "unique_action_indices_executed": unique_pairs,
        "skipped_action_steps": skipped,
        "action_skip_pct": 100.0 * skipped / denominator if denominator else 0.0,
        "repeated_action_indices": repeated,
        "duplicate_index_pct": 100.0 * repeated / len(policy) if policy else 0.0,
        "chunks_observed": len(seen),
        "chunks_index0_executed": index0,
        "chunks_first_index_gt_zero": first_gt_zero,
        "index0_coverage_pct": 100.0 * index0 / total_chunks if total_chunks else 0.0,
        "mean_per_chunk_index_coverage_pct": 100.0 * statistics.mean(coverage) if coverage else 0.0,
        "policy_hold_ticks": holds,
        "policy_hold_pct": 100.0 * holds / len(outputs) if outputs else 0.0,
        "hold_reasons": hold_reasons,
        "boundary_delta": distribution(boundary_deltas),
    }


def within_chunk_stats(candidates: Sequence[Candidate]) -> dict[str, object]:
    maxima: list[float] = []
    per_dimension = {name: [] for name in DIMENSIONS}
    for candidate in candidates:
        for previous, current in zip(candidate.actions, candidate.actions[1:]):
            differences = [abs(a - b) for a, b in zip(previous, current)]
            maxima.append(max(differences))
            for name, value in zip(DIMENSIONS, differences):
                per_dimension[name].append(value)
    return {
        "max_across_19d": distribution(maxima),
        "per_dimension": {name: distribution(values) for name, values in per_dimension.items()},
    }


def _actual_policy_outputs(
    candidates: Sequence[Candidate],
    dispatches: Sequence[tuple[int, dict[str, object]]],
    finals: Sequence[tuple[int, dict[str, object]]],
) -> list[TickOutput]:
    receipts = [candidate.receipt_ns for candidate in candidates]
    final_by_sequence = {int(payload["sequence"]): payload for _, payload in finals}
    outputs: list[TickOutput] = []
    for _, payload in dispatches:
        decision = payload["decision"]
        if not isinstance(decision, dict):
            continue
        if decision.get("base_source") != "pi05_policy_control" or decision.get("manipulation_source") != "pi05_policy_control":
            continue
        timestamp = int(payload["timestamp_ns"])
        position = bisect_right(receipts, timestamp) - 1
        if position < 0:
            continue
        final = final_by_sequence[int(payload["final_action_sequence"])]
        actual = validate_action(final["action"])
        mask = decision.get("clipped_mask")
        if not isinstance(mask, list) or len(mask) != ACTION_DIM:
            raise ValueError("invalid_clipped_mask")
        usable = [index for index, clipped in enumerate(mask) if not clipped]
        matches: list[tuple[float, Candidate, int]] = []
        for candidate_position in range(max(0, position - 1), min(len(candidates), position + 2)):
            candidate = candidates[candidate_position]
            for action_index, row in enumerate(candidate.actions):
                score = max(abs(row[index] - actual[index]) for index in usable)
                matches.append((score, candidate, action_index))
        score, candidate, index = min(matches, key=lambda item: item[0])
        if score > 1e-8:
            raise ValueError(
                f"cannot_match_policy_action_near:{candidates[position].request_id}:{score}"
            )
        outputs.append(
            TickOutput(timestamp, candidate.request_id, index, candidate.actions[index])
        )
    return outputs


def handoff_delta_stats(outputs: Sequence[TickOutput]) -> dict[str, object]:
    maxima: list[float] = []
    per_dimension = {name: [] for name in DIMENSIONS}
    previous: Optional[TickOutput] = None
    for item in outputs:
        if (
            previous is not None
            and previous.action is not None
            and item.action is not None
            and previous.request_id != item.request_id
        ):
            differences = [abs(a - b) for a, b in zip(previous.action, item.action)]
            maxima.append(max(differences))
            for name, value in zip(DIMENSIONS, differences):
                per_dimension[name].append(value)
        if item.action is not None:
            previous = item
    return {
        "max_across_19d": distribution(maxima),
        "per_dimension": {name: distribution(values) for name, values in per_dimension.items()},
    }


def _longest_hold_s(outputs: Sequence[TickOutput]) -> float:
    longest = current_start = previous_ns = None
    best = 0
    for item in outputs:
        if item.action is None:
            if current_start is None:
                current_start = item.timestamp_ns
            previous_ns = item.timestamp_ns
        else:
            if current_start is not None and previous_ns is not None:
                best = max(best, previous_ns - current_start)
            current_start = previous_ns = None
    if current_start is not None and previous_ns is not None:
        best = max(best, previous_ns - current_start)
    return best / 1e9


def latency_analysis(
    candidates: Sequence[Candidate], actual_ticks: Sequence[int]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for threshold in (250, 300, 400, 500, 625, 700, 1000):
        accepted = [item for item in candidates if item.response_latency_ms <= threshold and item.valid]
        outputs = simulate(
            candidates, actual_ticks, handoff="immediate", latency_limit_ms=threshold
        )
        metrics = scheduler_metrics(outputs, len(accepted))
        gaps_ms = [
            (right.receipt_ns - left.receipt_ns) / 1e6
            for left, right in zip(accepted, accepted[1:])
        ]
        ages = [item.response_latency_ms for item in accepted]
        result[str(threshold)] = {
            "accepted_candidates": len(accepted),
            "accepted_pct": 100.0 * len(accepted) / len(candidates),
            "rejected_pct": 100.0 * (len(candidates) - len(accepted)) / len(candidates),
            "policy_coverage_pct": 100.0 * metrics["policy_ticks"] / len(actual_ticks),
            "safe_hold_tick_pct": metrics["policy_hold_pct"],
            "longest_hold_s": _longest_hold_s(outputs),
            "accepted_response_age_ms": distribution(ages),
            "accepted_chunk_gap_ms": distribution(gaps_ms),
        }
    return result


def clipping_analysis(
    candidates: Sequence[Candidate],
    dispatches: Sequence[tuple[int, dict[str, object]]],
    finals: Sequence[tuple[int, dict[str, object]]],
) -> dict[str, object]:
    final_by_sequence = {int(payload["sequence"]): payload for _, payload in finals}
    receipts = [candidate.receipt_ns for candidate in candidates]
    affected = {name: {"count": 0, "max_exceedance": 0.0, "float_tail": 0, "physical": 0} for name in DIMENSIONS}
    clipped_ticks = 0
    policy_ticks = 0
    examples: list[dict[str, object]] = []
    for _, payload in dispatches:
        decision = payload.get("decision")
        if not isinstance(decision, dict) or decision.get("base_source") != "pi05_policy_control":
            continue
        policy_ticks += 1
        mask = decision.get("clipped_mask")
        if not isinstance(mask, list) or len(mask) != ACTION_DIM:
            raise ValueError("invalid_clipped_mask")
        if not any(mask):
            continue
        clipped_ticks += 1
        timestamp = int(payload["timestamp_ns"])
        position = bisect_right(receipts, timestamp) - 1
        candidate = candidates[position]
        index = int((timestamp - candidate.receipt_ns) // POLICY_PERIOD_NS)
        before = candidate.actions[index]
        final = final_by_sequence[int(payload["final_action_sequence"])]
        after = validate_action(final["action"])
        for dimension, clipped in enumerate(mask):
            if not clipped:
                continue
            exceedance = abs(before[dimension] - after[dimension])
            item = affected[DIMENSIONS[dimension]]
            item["count"] += 1
            item["max_exceedance"] = max(item["max_exceedance"], exceedance)
            kind = "float_tail" if exceedance <= 1e-6 else "physical"
            item[kind] += 1
            if len(examples) < 30:
                examples.append({
                    "sequence": int(payload["sequence"]), "request_id": candidate.request_id,
                    "action_index": index, "dimension": DIMENSIONS[dimension],
                    "before": before[dimension], "after": after[dimension],
                    "exceedance": exceedance, "classification": kind,
                })
    return {
        "policy_ticks": policy_ticks,
        "clipped_ticks": clipped_ticks,
        "clipping_rate_pct": 100.0 * clipped_ticks / policy_ticks if policy_ticks else 0.0,
        "affected_dimensions": {name: values for name, values in affected.items() if values["count"]},
        "examples": examples,
    }


def strategy_matrix(candidates: Sequence[Candidate], start_ns: int, end_ns: int) -> list[dict[str, object]]:
    within_p95 = within_chunk_stats(candidates)["max_across_19d"]["p95"]
    rows: list[dict[str, object]] = []
    risk = {
        "immediate": "low timing distortion; high discontinuity",
        "next-step-boundary": "low; <=1 policy-step delay",
        "crossfade-1": "low-moderate continuous-value interpolation",
        "crossfade-2": "moderate continuous-value interpolation",
        "crossfade-3": "higher; 3-step trajectory distortion",
    }
    latency = {
        "immediate": 0.0,
        "next-step-boundary": POLICY_PERIOD_S * 500.0,
        "crossfade-1": POLICY_PERIOD_S * 1000.0,
        "crossfade-2": POLICY_PERIOD_S * 2000.0,
        "crossfade-3": POLICY_PERIOD_S * 3000.0,
    }
    for rate in (20, 24, 30, 40, 50, 60):
        ticks = make_ticks(start_ns, end_ns, rate)
        for handoff in ("immediate", "next-step-boundary", "crossfade-1", "crossfade-2", "crossfade-3"):
            metrics = scheduler_metrics(simulate(candidates, ticks, handoff=handoff), len(candidates))
            boundary = metrics["boundary_delta"]
            rows.append({
                "control_rate_hz": rate,
                "handoff": handoff,
                "action_skip_pct": metrics["action_skip_pct"],
                "index0_coverage_pct": metrics["index0_coverage_pct"],
                "duplicate_index_pct": metrics["duplicate_index_pct"],
                "policy_hold_pct": metrics["policy_hold_pct"],
                "boundary_delta_p50": boundary["p50"],
                "boundary_delta_p95": boundary["p95"],
                "boundary_delta_max": boundary["max"],
                "within_chunk_p95": within_p95,
                "estimated_added_handoff_latency_ms": latency[handoff],
                "semantic_distortion_risk": risk[handoff],
            })
    return rows


def control_rate_analysis(candidates: Sequence[Candidate], start_ns: int, end_ns: int) -> dict[str, object]:
    report: dict[str, object] = {}
    for rate in (20, 24, 30, 40, 50, 60):
        ideal = scheduler_metrics(
            simulate(candidates, make_ticks(start_ns, end_ns, rate)), len(candidates)
        )
        phases = []
        for phase in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
            metrics = scheduler_metrics(
                simulate(candidates, make_ticks(start_ns, end_ns, rate, phase_fraction=phase)),
                len(candidates),
            )
            phases.append(metrics["action_skip_pct"])
        jitters: dict[str, object] = {}
        for jitter in (1, 2, 5, 10):
            skip_transition_rates = []
            for seed in range(20):
                outputs = simulate(
                    candidates,
                    make_ticks(start_ns, end_ns, rate, jitter_ms=jitter, seed=seed),
                )
                policy = [item for item in outputs if item.action is not None]
                transitions = skips = 0
                previous: Optional[TickOutput] = None
                for item in policy:
                    if previous is not None and item.request_id == previous.request_id:
                        transitions += 1
                        skips += int(item.action_index is not None and previous.action_index is not None and item.action_index - previous.action_index > 1)
                    previous = item
                skip_transition_rates.append(100.0 * skips / transitions if transitions else 0.0)
            jitters[f"plus_minus_{jitter}ms"] = {
                "mean_skip_transition_probability_pct": statistics.mean(skip_transition_rates),
                "max_seed_skip_transition_probability_pct": max(skip_transition_rates),
            }
        report[str(rate)] = {
            "tick_interval_ms": 1000.0 / rate,
            "ideal": ideal,
            "phase_sensitivity_action_skip_pct": {
                "min": min(phases), "max": max(phases), "samples": phases,
            },
            "jitter": jitters,
        }
    return report


def analyze(db_path: Path, archive_sha256: str = "") -> dict[str, object]:
    topics = read_bag(db_path)
    candidates = parse_candidates(topics[POLICY_TOPIC])
    dispatches = topics[DISPATCH_TOPIC]
    finals = topics[FINAL_TOPIC]
    contexts = topics[CONTEXT_TOPIC]
    all_timestamps = [timestamp for rows in topics.values() for timestamp, _ in rows]
    actual_outputs = _actual_policy_outputs(candidates, dispatches, finals)
    actual_metrics = scheduler_metrics(actual_outputs, len(candidates))
    transitions: dict[str, int] = {}
    previous: Optional[TickOutput] = None
    for output in actual_outputs:
        if previous is not None and output.request_id == previous.request_id:
            delta = int(output.action_index) - int(previous.action_index)
            transitions[str(delta)] = transitions.get(str(delta), 0) + 1
        previous = output
    context_values = [payload for _, payload in contexts]
    run_ids = sorted({str(item.get("run_id")) for item in context_values})
    task_ids = sorted({item.get("current_task_id") for item in context_values})
    attempts = sorted({item.get("current_attempt_count") for item in context_values})
    request_ids = [item.request_id for item in candidates]
    latencies = [item.response_latency_ms for item in candidates]
    analysis_start = candidates[0].receipt_ns
    analysis_end = min(max(all_timestamps), candidates[-1].receipt_ns + HORIZON * POLICY_PERIOD_NS)
    return {
        "schema_name": "M10B05SchedulerAnalysis",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "db3": str(db_path.resolve()),
            "archive_sha256": archive_sha256,
            "read_only": True,
        },
        "constants": {
            "action_step_period_s": POLICY_PERIOD_S,
            "action_horizon": HORIZON,
            "action_dim": ACTION_DIM,
            "chunk_duration_ms": HORIZON * POLICY_PERIOD_S * 1000.0,
        },
        "baseline": {
            "bag_duration_s": (max(all_timestamps) - min(all_timestamps)) / 1e9,
            "topic_counts": {topic: len(rows) for topic, rows in topics.items()},
            "candidate_count": len(candidates),
            "request_id_first": request_ids[0],
            "request_id_last": request_ids[-1],
            "request_ids_contiguous": request_ids == list(range(request_ids[0], request_ids[-1] + 1)),
            "generation_ids": sorted({item.generation_id for item in candidates}),
            "candidate_identity_count": len({item.identity for item in candidates}),
            "all_candidates_valid": all(item.valid for item in candidates),
            "all_chunks_15x19_finite": True,
            "context_run_ids": run_ids,
            "context_task_ids": task_ids,
            "context_attempt_counts": attempts,
            "context_all_valid": all(item.get("valid") is True for item in context_values),
            "policy_action_ticks": len(actual_outputs),
            "actual_scheduler": actual_metrics,
            "same_chunk_index_transitions": transitions,
            "response_latency_ms": distribution(latencies),
            "latency_exceedance_pct": {
                str(limit): 100.0 * sum(value > limit for value in latencies) / len(latencies)
                for limit in (250, 500, 625)
            },
        },
        "within_chunk_delta": within_chunk_stats(candidates),
        "actual_handoff_delta": handoff_delta_stats(actual_outputs),
        "control_rates": control_rate_analysis(candidates, analysis_start, analysis_end),
        "strategy_matrix": strategy_matrix(candidates, analysis_start, analysis_end),
        "latency_gates": latency_analysis(
            candidates,
            [
                int(payload["timestamp_ns"])
                for _, payload in dispatches
                if analysis_start <= int(payload["timestamp_ns"]) <= analysis_end
            ],
        ),
        "action_mux_clipping": clipping_analysis(candidates, dispatches, finals),
        "analysis_window": {"start_ns": analysis_start, "end_ns": analysis_end},
    }


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(report: dict[str, object]) -> str:
    baseline = report["baseline"]
    lines = [
        "# M10-B0.5-A scheduler offline analysis", "",
        f"- Bag duration: {_fmt(baseline['bag_duration_s'])} s",
        f"- Candidates: {baseline['candidate_count']} (request {baseline['request_id_first']}..{baseline['request_id_last']})",
        f"- Policy FinalAction ticks: {baseline['policy_action_ticks']}",
        f"- Generation IDs: `{baseline['generation_ids']}`", "",
        "## Control-rate and jitter comparison", "",
        "|Hz|interval ms|skip % ideal|index0 %|duplicate %|hold %|phase skip min..max|±1ms skip prob %|±2ms|±5ms|±10ms|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rate, item in report["control_rates"].items():
        ideal = item["ideal"]
        phase = item["phase_sensitivity_action_skip_pct"]
        jitter = item["jitter"]
        lines.append(
            f"|{rate}|{_fmt(item['tick_interval_ms'])}|{_fmt(ideal['action_skip_pct'])}|"
            f"{_fmt(ideal['index0_coverage_pct'])}|{_fmt(ideal['duplicate_index_pct'])}|"
            f"{_fmt(ideal['policy_hold_pct'])}|{_fmt(phase['min'])}..{_fmt(phase['max'])}|"
            + "|".join(_fmt(jitter[f"plus_minus_{ms}ms"]["mean_skip_transition_probability_pct"]) for ms in (1, 2, 5, 10))
            + "|"
        )
    lines += ["", "## Required strategy matrix", "",
        "|Hz|handoff|skip %|index0 %|duplicate %|hold %|boundary P50|P95|MAX|within P95|added ms|semantic risk|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["strategy_matrix"]:
        lines.append(
            "|{control_rate_hz}|{handoff}|{action_skip_pct:.4f}|{index0_coverage_pct:.4f}|"
            "{duplicate_index_pct:.4f}|{policy_hold_pct:.4f}|{p50}|{p95}|{maximum}|"
            "{within}|{estimated_added_handoff_latency_ms:.4f}|{semantic_distortion_risk}|".format(
                **row,
                p50=_fmt(row["boundary_delta_p50"]), p95=_fmt(row["boundary_delta_p95"]),
                maximum=_fmt(row["boundary_delta_max"]), within=_fmt(row["within_chunk_p95"]),
            )
        )
    lines += ["", "## Latency gates", "",
        "|limit ms|accepted %|rejected %|policy coverage %|safe hold %|longest hold s|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for limit, item in report["latency_gates"].items():
        lines.append(
            f"|{limit}|{_fmt(item['accepted_pct'])}|{_fmt(item['rejected_pct'])}|"
            f"{_fmt(item['policy_coverage_pct'])}|{_fmt(item['safe_hold_tick_pct'])}|{_fmt(item['longest_hold_s'])}|"
        )
    clipping = report["action_mux_clipping"]
    lines += ["", "## ActionMux clipping", "",
        f"- Policy ticks: {clipping['policy_ticks']}",
        f"- Clipped ticks: {clipping['clipped_ticks']} ({_fmt(clipping['clipping_rate_pct'])}%)",
        f"- Affected dimensions: `{json.dumps(clipping['affected_dimensions'], ensure_ascii=False)}`", "",
        "This report is offline telemetry analysis only. It does not prove control publication or execution.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path, help="rosbag2 sqlite3 .db3 path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-sha256", default="")
    args = parser.parse_args()
    report = analyze(args.bag, args.archive_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.output_dir / f"m10_b05_scheduler_{stamp}.json"
    markdown_path = args.output_dir / f"m10_b05_scheduler_{stamp}.md"
    if json_path.exists() or markdown_path.exists():
        raise SystemExit("refusing_to_overwrite_existing_report")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
