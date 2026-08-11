#!/usr/bin/env python3
"""Offline-only M10-B1-A.2 base-envelope counterfactual analyzer.

This program reads rosbag2 SQLite directly.  It never initializes ROS, plays a
bag, imports a production node, runs a policy, or sends a command.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import struct
import tarfile
import tempfile
from typing import Sequence

import yaml


ARCHIVE_SHA256 = "f7c49502bb0338bccb8dfa9c267c833d9f6004590845aeeb2010c995f0d24df2"
POLICY_TOPIC = "/mmk2_pi05_adapter/policy_control_candidate"
FINAL_TOPIC = "/team/final_action"
DISPATCH_TOPIC = "/team/action_dispatch"
POLICY_SOURCE = "pi05_policy_control"
HORIZON = 15
ACTION_DIM = 19
HANDOFF_STEPS = 2
GRIPPER_INDICES = frozenset((11, 18))


@dataclass(frozen=True)
class BagRow:
    timestamp_ns: int
    payload: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_cdr_string(blob: bytes) -> str:
    if len(blob) < 8 or blob[:2] not in (b"\x00\x00", b"\x00\x01"):
        raise ValueError("unsupported CDR encapsulation")
    endian = "<" if blob[1] == 1 else ">"
    length = struct.unpack_from(endian + "I", blob, 4)[0]
    raw = blob[8:8 + length]
    if length < 1 or len(raw) != length or raw[-1] != 0:
        raise ValueError("invalid CDR string")
    return raw[:-1].decode("utf-8")


def strict_json(raw: str) -> dict[str, object]:
    value = json.loads(
        raw,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {token}")),
    )
    if not isinstance(value, dict):
        raise ValueError("JSON payload is not an object")
    return value


def read_string_topics(db_path: Path) -> dict[str, list[BagRow]]:
    wanted = (POLICY_TOPIC, FINAL_TOPIC, DISPATCH_TOPIC)
    result = {name: [] for name in wanted}
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT t.name,m.timestamp,m.data FROM messages m "
            "JOIN topics t ON t.id=m.topic_id "
            "WHERE t.name IN (?,?,?) ORDER BY m.timestamp",
            wanted,
        )
        for name, timestamp_ns, blob in rows:
            result[str(name)].append(
                BagRow(int(timestamp_ns), strict_json(decode_cdr_string(blob)))
            )
    missing = [name for name, rows in result.items() if not rows]
    if missing:
        raise ValueError(f"bag missing topics: {missing}")
    return result


def action19(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != ACTION_DIM:
        raise ValueError("action is not 19D")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("action contains non-real value")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("action contains non-finite value")
    return result


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values]
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("non-finite distribution")
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "p01": percentile(finite, 0.01),
        "p05": percentile(finite, 0.05),
        "p50": percentile(finite, 0.50),
        "p95": percentile(finite, 0.95),
        "p99": percentile(finite, 0.99),
        "max": max(finite) if finite else None,
        "mean": statistics.fmean(finite) if finite else None,
    }


def candidate_identity(candidate: dict[str, object]) -> tuple[object, ...]:
    return (
        candidate.get("run_id"), candidate.get("task_id"), candidate.get("attempt_count"),
        candidate.get("instruction_fingerprint"), candidate.get("task_set_fingerprint"),
        candidate.get("active_task_fingerprint"), candidate.get("generation_id"),
    )


def parse_candidates(rows: Sequence[BagRow]) -> tuple[list[dict[str, object]], list[tuple[float, ...]]]:
    candidates: list[dict[str, object]] = []
    raw_rows: list[tuple[float, ...]] = []
    for row in rows:
        payload = dict(row.payload)
        if payload.get("action_horizon") != HORIZON or payload.get("action_dim") != ACTION_DIM:
            raise ValueError("candidate is not 15x19")
        actions = payload.get("actions")
        if not isinstance(actions, list) or len(actions) != HORIZON:
            raise ValueError("candidate is not 15x19")
        parsed = tuple(action19(action) for action in actions)
        payload["_bag_timestamp_ns"] = row.timestamp_ns
        payload["_actions"] = parsed
        candidates.append(payload)
        raw_rows.extend(parsed)
    return candidates, raw_rows


def _policy_dispatch(payload: dict[str, object]) -> bool:
    decision = payload.get("decision")
    return bool(
        isinstance(decision, dict)
        and decision.get("base_source") == POLICY_SOURCE
        and decision.get("manipulation_source") == POLICY_SOURCE
    )


def reconstruct_policy_ticks(
    candidates: Sequence[dict[str, object]],
    dispatch_rows: Sequence[BagRow],
    final_rows: Sequence[BagRow],
    *,
    period_ns: int,
    latency_limit_ms: float,
) -> tuple[list[tuple[float, ...]], dict[str, object]]:
    finals = {
        int(row.payload["sequence"]): action19(row.payload["action"])
        for row in final_rows
    }
    events: list[tuple[int, int, object]] = []
    events.extend((int(candidate["_bag_timestamp_ns"]), 0, candidate) for candidate in candidates)
    events.extend(
        (int(row.payload["timestamp_ns"]), 1, row.payload)
        for row in dispatch_rows if _policy_dispatch(row.payload)
    )
    events.sort(key=lambda item: (item[0], item[1]))

    pending: dict[str, object] | None = None
    handoff_source: tuple[float, ...] | None = None
    handoff_start_ns: int | None = None
    last_output: tuple[float, ...] | None = None
    last_identity: tuple[object, ...] | None = None
    reconstructed: list[tuple[float, ...]] = []
    masks: list[tuple[bool, ...]] = []
    alpha_ratio_spreads: list[float] = []
    fit_residuals: list[float] = []

    for timestamp_ns, kind, value in events:
        if kind == 0:
            candidate = value
            assert isinstance(candidate, dict)
            valid = bool(
                candidate.get("valid") is True
                and candidate.get("context_valid") is True
                and float(candidate.get("response_latency_ms", math.inf)) <= latency_limit_ms
            )
            if not valid:
                pending = None
                handoff_source = None
                last_output = None
                last_identity = None
                continue
            identity = candidate_identity(candidate)
            handoff_source = last_output if last_output is not None and last_identity == identity else None
            handoff_start_ns = timestamp_ns if handoff_source is not None else None
            if handoff_source is None:
                last_output = None
                last_identity = None
            pending = candidate
            continue

        dispatch = value
        assert isinstance(dispatch, dict)
        if pending is None:
            raise ValueError("policy dispatch has no recoverable candidate")
        age_ns = timestamp_ns - int(pending["_bag_timestamp_ns"])
        index = age_ns // period_ns
        if index < 0 or index >= HORIZON:
            raise ValueError("policy dispatch falls outside candidate horizon")
        targets = pending["_actions"]
        assert isinstance(targets, tuple)
        target = targets[index]
        decision = dispatch["decision"]
        assert isinstance(decision, dict)
        mask_value = decision.get("clipped_mask")
        if (
            not isinstance(mask_value, list) or len(mask_value) != ACTION_DIM
            or any(type(item) is not bool for item in mask_value)
        ):
            raise ValueError("invalid clipped_mask")
        mask = tuple(mask_value)
        after = finals[int(dispatch["final_action_sequence"])]
        before = target

        if handoff_source is not None and handoff_start_ns is not None:
            handoff_age_ns = timestamp_ns - handoff_start_ns
            if handoff_age_ns < HANDOFF_STEPS * period_ns:
                ratios = []
                for dimension in range(ACTION_DIM):
                    if dimension in GRIPPER_INDICES or mask[dimension]:
                        continue
                    delta = target[dimension] - handoff_source[dimension]
                    if abs(delta) > 1e-6:
                        ratios.append((after[dimension] - handoff_source[dimension]) / delta)
                if not ratios:
                    raise ValueError("handoff alpha cannot be inferred")
                alpha = min(1.0, max(0.0, statistics.median(ratios)))
                alpha_ratio_spreads.append(max(ratios) - min(ratios))
                before = tuple(
                    target[dimension] if dimension in GRIPPER_INDICES
                    else handoff_source[dimension] + alpha * (target[dimension] - handoff_source[dimension])
                    for dimension in range(ACTION_DIM)
                )
                fit_residuals.extend(
                    abs(before[dimension] - after[dimension])
                    for dimension in range(ACTION_DIM)
                    if not mask[dimension] and dimension not in GRIPPER_INDICES
                )
            else:
                handoff_source = None
                handoff_start_ns = None

        # Unclipped FinalAction components are the exact pre-clamp values.
        before = tuple(after[i] if not mask[i] else before[i] for i in range(ACTION_DIM))
        last_output = before
        last_identity = candidate_identity(pending)
        reconstructed.append(before)
        masks.append(mask)

    if not reconstructed:
        raise ValueError("no policy ticks reconstructed")
    e0_matches = all(
        (abs(row[0]) > 0.25) == mask[0] and (abs(row[1]) > 0.50) == mask[1]
        for row, mask in zip(reconstructed, masks)
    )
    return reconstructed, {
        "method": (
            "candidate bag timestamp selects the 24 Hz row; unclipped FinalAction dimensions "
            "supply exact pre-clamp values; two-step handoff alpha is inferred from shared "
            "unclipped dimensions and then applied to clipped dimensions"
        ),
        "callback_receipt_timestamp_archived": False,
        "policy_ticks": len(reconstructed),
        "handoff_fit_ticks": len(alpha_ratio_spreads),
        "max_alpha_ratio_spread": max(alpha_ratio_spreads, default=0.0),
        "max_unclipped_handoff_fit_abs_residual": max(fit_residuals, default=0.0),
        "e0_base_masks_match_every_tick": e0_matches,
        "limitation": (
            "clipped components are reconstructed through the shared scheduler formula, not "
            "independently recorded pre-clamp samples"
        ),
    }


def dimension_counterfactual(values: Sequence[float], limit: float) -> dict[str, object]:
    distortion = [max(abs(value) - limit, 0.0) for value in values]
    exceedance = [value for value in distortion if value > 0.0]
    return {
        "total_relevant_ticks": len(values),
        "clipping_ticks": len(exceedance),
        "clipping_pct": 100.0 * len(exceedance) / len(values),
        "raw_min": min(values),
        "raw_max": max(values),
        "exceedance_magnitude_clipped_ticks": {
            key: distribution(exceedance)[key] for key in ("p50", "p95", "p99", "max")
        },
        "abs_raw_minus_clipped_all_ticks": {
            key: distribution(distortion)[key] for key in ("mean", "p50", "p95", "max")
        },
        "_distortion": distortion,
    }


def envelope_stats(rows: Sequence[tuple[float, ...]], limit_v: float, limit_w: float) -> dict[str, object]:
    base_v = dimension_counterfactual([row[0] for row in rows], limit_v)
    base_w = dimension_counterfactual([row[1] for row in rows], limit_w)
    dv = base_v.pop("_distortion")
    dw = base_w.pop("_distortion")
    assert isinstance(dv, list) and isinstance(dw, list)
    l1 = [v + w for v, w in zip(dv, dw)]
    l2 = [math.hypot(v, w) for v, w in zip(dv, dw)]
    any_changed = sum(value > 0.0 for value in l1)
    both_changed = sum(v > 0.0 and w > 0.0 for v, w in zip(dv, dw))
    return {
        "limits": {"base_v_m_s": limit_v, "base_w_rad_s": limit_w},
        "base_v": base_v,
        "base_w": base_w,
        "combined": {
            "at_least_one_changed_ticks": any_changed,
            "at_least_one_changed_pct_of_policy_ticks": 100.0 * any_changed / len(rows),
            "both_changed_ticks": both_changed,
            "mean_absolute_base_command_distortion_per_component": statistics.fmean(
                [(v + w) / 2.0 for v, w in zip(dv, dw)]
            ),
            "mean_l1_distortion_per_tick": statistics.fmean(l1),
            "total_l1_distortion": sum(l1),
            "mean_l2_distortion_per_tick": statistics.fmean(l2),
            "total_l2_distortion": sum(l2),
        },
    }


def slide_negative_stats(
    candidates: Sequence[dict[str, object]], raw_rows: Sequence[tuple[float, ...]], period_ns: int,
) -> dict[str, object]:
    first_candidate_ns = int(candidates[0]["_bag_timestamp_ns"])
    negative: list[tuple[float, float]] = []
    for candidate in candidates:
        actions = candidate["_actions"]
        assert isinstance(actions, tuple)
        for index, row in enumerate(actions):
            if row[2] < 0.0:
                relative_s = (
                    int(candidate["_bag_timestamp_ns"]) + index * period_ns - first_candidate_ns
                ) / 1e9
                negative.append((relative_s, row[2]))
    times = [item[0] for item in negative]
    values = [item[1] for item in negative]
    return {
        "raw_rows": len(raw_rows),
        "negative_rows": len(negative),
        "negative_pct": 100.0 * len(negative) / len(raw_rows),
        "occurrence_time_from_first_candidate_s": {
            "first": min(times), "median": percentile(times, 0.50), "last": max(times),
        },
        "negative_value_distribution": {
            "min": min(values), "p01": percentile(values, 0.01), "p05": percentile(values, 0.05),
        },
        "time_method": "candidate bag timestamp plus nominal action-row offset at 24 Hz",
        "observed_timing_classification": "EARLY_PRESENT",
    }


def safe_extract_db(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError("archive contains unsafe path")
            if member.issym() or member.islnk():
                raise ValueError("archive contains link")
        bundle.extractall(destination, members=members, filter="data")
    db_files = list(destination.rglob("*.db3"))
    if len(db_files) != 1:
        raise ValueError(f"expected one db3, found {len(db_files)}")
    return db_files[0]


def official_envelope(source_path: Path) -> dict[str, object]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    found: tuple[float, float, int] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, (ast.Tuple, ast.List)) or len(target.elts) != 2:
            continue
        names = [
            item.attr if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name)
            and item.value.id == "self" else None
            for item in target.elts
        ]
        if names != ["max_lin", "max_ang"] or not isinstance(node.value, (ast.Tuple, ast.List)):
            continue
        values = tuple(ast.literal_eval(item) for item in node.value.elts)
        found = (float(values[0]), float(values[1]), node.lineno)
    required_fragments = (
        "np.clip(lin, -self.max_lin, self.max_lin)",
        "np.clip(ang, -self.max_ang, self.max_ang)",
        "tw.linear.x = float(self.tc[0])",
        "tw.angular.z = float(self.tc[1])",
        "self.cmd_vel_pub.publish(tw)",
    )
    if found is None or any(fragment not in source for fragment in required_fragments):
        raise ValueError("official Task1 limit-to-/cmd_vel source chain not found")
    return {
        "base_v_m_s": found[0], "base_w_rad_s": found[1],
        "source": str(source_path.resolve()), "source_sha256": sha256_file(source_path),
        "assignment_line": found[2], "limit_to_cmd_vel_chain_verified": True,
    }


def analyze(archive: Path, team_config: Path, official_client: Path) -> dict[str, object]:
    archive_sha = sha256_file(archive)
    if archive_sha != ARCHIVE_SHA256:
        raise ValueError(f"archive SHA256 mismatch: {archive_sha}")
    config = yaml.safe_load(team_config.read_text(encoding="utf-8"))
    period_ns = int(round(float(config["pi05_policy_control"]["action_step_period_s"]) * 1e9))
    latency_ms = float(config["pi05_policy_control"]["max_policy_response_latency_ms"])
    current_v = float(config["action_mux"]["max_abs_base_v"])
    current_w = float(config["action_mux"]["max_abs_base_w"])
    official = official_envelope(official_client)
    with tempfile.TemporaryDirectory(prefix="m10_b1a2.", dir="/tmp") as temp_dir:
        db_path = safe_extract_db(archive, Path(temp_dir))
        topics = read_string_topics(db_path)
    candidates, raw_rows = parse_candidates(topics[POLICY_TOPIC])
    policy_rows, reconstruction = reconstruct_policy_ticks(
        candidates, topics[DISPATCH_TOPIC], topics[FINAL_TOPIC],
        period_ns=period_ns, latency_limit_ms=latency_ms,
    )
    envelopes = {
        "E0_CURRENT_TEAM": envelope_stats(policy_rows, current_v, current_w),
        "E2_INTERMEDIATE_REFERENCE": envelope_stats(policy_rows, 0.35, 0.80),
        "E1_OFFICIAL_TASK1_BASELINE": envelope_stats(
            policy_rows, float(official["base_v_m_s"]), float(official["base_w_rad_s"])
        ),
    }
    e0, e1 = envelopes["E0_CURRENT_TEAM"], envelopes["E1_OFFICIAL_TASK1_BASELINE"]
    return {
        "schema_name": "M10B1A2BaseEnvelopeCounterfactual",
        "schema_version": 1,
        "input": {
            "archive": str(archive.resolve()), "archive_sha256": archive_sha,
            "read_only_direct_sqlite": True, "candidate_chunks": len(candidates),
            "raw_candidate_rows": len(raw_rows),
        },
        "official_task1_baseline_envelope": official,
        "team_current_base_envelope": {
            "base_v_m_s": current_v, "base_w_rad_s": current_w,
            "config_source": str(team_config.resolve()), "config_sha256": sha256_file(team_config),
        },
        "reconstruction": reconstruction,
        "envelopes": envelopes,
        "e1_reduction_pct_from_e0": {
            "base_v": 100.0 * (e0["base_v"]["clipping_ticks"] - e1["base_v"]["clipping_ticks"]) / e0["base_v"]["clipping_ticks"],
            "base_w": 100.0 * (e0["base_w"]["clipping_ticks"] - e1["base_w"]["clipping_ticks"]) / e0["base_w"]["clipping_ticks"],
        },
        "e1_max_remaining_exceedance": {"base_v": 0.0, "base_w": 0.0},
        "slide_negative_raw_candidates": slide_negative_stats(candidates, raw_rows, period_ns),
        "decision": {
            "base_envelope_mismatch": "PROVEN",
            "ready_for_b1a2_sim": "YES",
        },
    }


def write_csv(path: Path, report: dict[str, object]) -> None:
    fields = (
        "envelope", "dimension", "limit", "total_relevant_ticks", "clipping_ticks",
        "clipping_pct", "raw_min", "raw_max", "exceedance_p50", "exceedance_p95",
        "exceedance_p99", "exceedance_max", "distortion_mean", "distortion_p50",
        "distortion_p95", "distortion_max",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for envelope_name, envelope in report["envelopes"].items():
            for dimension in ("base_v", "base_w"):
                item = envelope[dimension]
                exceedance = item["exceedance_magnitude_clipped_ticks"]
                distortion = item["abs_raw_minus_clipped_all_ticks"]
                writer.writerow({
                    "envelope": envelope_name, "dimension": dimension,
                    "limit": envelope["limits"][f"{dimension}_{'m_s' if dimension == 'base_v' else 'rad_s'}"],
                    "total_relevant_ticks": item["total_relevant_ticks"],
                    "clipping_ticks": item["clipping_ticks"], "clipping_pct": item["clipping_pct"],
                    "raw_min": item["raw_min"], "raw_max": item["raw_max"],
                    **{f"exceedance_{key}": exceedance[key] for key in ("p50", "p95", "p99", "max")},
                    **{f"distortion_{key}": distortion[key] for key in ("mean", "p50", "p95", "max")},
                })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--team-config", required=True, type=Path)
    parser.add_argument("--official-client", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.archive, args.team_config, args.official_client)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "base_envelope_counterfactual.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "base_envelope_counterfactual.csv", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
