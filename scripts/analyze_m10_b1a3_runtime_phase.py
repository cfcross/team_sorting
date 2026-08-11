#!/usr/bin/env python3
"""Offline M10-B1-A.3 early-phase trajectory audit from rosbag2 SQLite."""

from __future__ import annotations

import argparse
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


POLICY_TOPIC = "/mmk2_pi05_adapter/policy_control_candidate"
FINAL_TOPIC = "/team/final_action"
DISPATCH_TOPIC = "/team/action_dispatch"
ODOM_TOPIC = "/slamware_ros_sdk_server_node/odom"
JOINT_TOPIC = "/joint_states"
POLICY_SOURCE = "pi05_policy_control"
ACTION_DIM = 19
HORIZON = 15
HANDOFF_STEPS = 2
GRIPPERS = frozenset((11, 18))
PHASES = (("0-5s", 0.0, 5.0), ("5-10s", 5.0, 10.0),
          ("10-15s", 10.0, 15.0), ("15-20s", 15.0, 20.0))


@dataclass(frozen=True)
class Row:
    bag_timestamp_ns: int
    payload: object


class CdrReader:
    def __init__(self, blob: bytes) -> None:
        if len(blob) < 4 or blob[:2] not in (b"\x00\x00", b"\x00\x01"):
            raise ValueError("unsupported CDR encapsulation")
        self.blob = blob
        self.endian = "<" if blob[1] == 1 else ">"
        self.offset = 4

    def align(self, size: int) -> None:
        self.offset += (-(self.offset - 4)) % size

    def unpack(self, code: str, size: int) -> int | float:
        self.align(size)
        if self.offset + size > len(self.blob):
            raise ValueError("truncated CDR")
        value = struct.unpack_from(self.endian + code, self.blob, self.offset)[0]
        self.offset += size
        return value

    def u32(self) -> int:
        return int(self.unpack("I", 4))

    def i32(self) -> int:
        return int(self.unpack("i", 4))

    def f64(self) -> float:
        return float(self.unpack("d", 8))

    def string(self) -> str:
        length = self.u32()
        raw = self.blob[self.offset:self.offset + length]
        self.offset += length
        if length < 1 or len(raw) != length or raw[-1] != 0:
            raise ValueError("invalid CDR string")
        return raw[:-1].decode("utf-8")

    def strings(self) -> tuple[str, ...]:
        return tuple(self.string() for _ in range(self.u32()))

    def f64s(self, count: int | None = None) -> tuple[float, ...]:
        width = self.u32() if count is None else count
        return tuple(self.f64() for _ in range(width))


def decode_string(blob: bytes) -> str:
    return CdrReader(blob).string()


def decode_header(reader: CdrReader) -> int:
    timestamp_ns = reader.i32() * 1_000_000_000 + reader.u32()
    reader.string()
    return timestamp_ns


def decode_odom(blob: bytes) -> tuple[int, float, float]:
    reader = CdrReader(blob)
    timestamp_ns = decode_header(reader)
    reader.string()
    reader.f64s(3 + 4 + 36)
    twist = reader.f64s(6)
    reader.f64s(36)
    return timestamp_ns, twist[0], twist[5]


def decode_joint_state(blob: bytes) -> tuple[int, tuple[str, ...], tuple[float, ...]]:
    reader = CdrReader(blob)
    timestamp_ns = decode_header(reader)
    names = reader.strings()
    positions = reader.f64s()
    reader.f64s()
    reader.f64s()
    if len(names) != len(positions):
        raise ValueError("JointState name/position length mismatch")
    return timestamp_ns, names, positions


def strict_json(raw: str) -> dict[str, object]:
    value = json.loads(
        raw,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {token}")),
    )
    if not isinstance(value, dict):
        raise ValueError("JSON payload is not an object")
    return value


def action19(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != ACTION_DIM:
        raise ValueError("action must be 19D")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("action must contain real values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("action contains non-finite value")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stats(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values]
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("non-finite statistics input")
    return {
        "count": len(finite), "mean": statistics.fmean(finite) if finite else None,
        "p05": percentile(finite, 0.05), "p50": percentile(finite, 0.50),
        "p95": percentile(finite, 0.95), "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def safe_extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        root = destination.resolve()
        for member in members:
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError("unsafe archive path")
            if member.issym() or member.islnk():
                raise ValueError("archive link is forbidden")
        bundle.extractall(destination, members=members, filter="data")
    dbs = list(destination.rglob("*.db3"))
    if len(dbs) != 1:
        raise ValueError(f"expected one db3, found {len(dbs)}")
    return dbs[0]


def read_bag(db_path: Path) -> dict[str, list[Row]]:
    wanted = {POLICY_TOPIC, FINAL_TOPIC, DISPATCH_TOPIC, ODOM_TOPIC, JOINT_TOPIC}
    result = {topic: [] for topic in wanted}
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        query = (
            "SELECT t.name,m.timestamp,m.data FROM messages m "
            "JOIN topics t ON t.id=m.topic_id ORDER BY m.timestamp"
        )
        for name, timestamp_ns, blob in connection.execute(query):
            topic = str(name)
            if topic not in wanted:
                continue
            if topic in (POLICY_TOPIC, FINAL_TOPIC, DISPATCH_TOPIC):
                payload: object = strict_json(decode_string(blob))
            elif topic == ODOM_TOPIC:
                payload = decode_odom(blob)
            else:
                payload = decode_joint_state(blob)
            result[topic].append(Row(int(timestamp_ns), payload))
    missing = sorted(topic for topic, rows in result.items() if not rows)
    if missing:
        raise ValueError(f"bag missing topics: {missing}")
    return result


def parse_candidates(rows: Sequence[Row]) -> list[dict[str, object]]:
    candidates = []
    for row in rows:
        assert isinstance(row.payload, dict)
        item = dict(row.payload)
        actions = item.get("actions")
        if item.get("action_horizon") != HORIZON or item.get("action_dim") != ACTION_DIM:
            raise ValueError("candidate shape is not 15x19")
        if not isinstance(actions, list) or len(actions) != HORIZON:
            raise ValueError("candidate shape is not 15x19")
        item["_actions"] = tuple(action19(action) for action in actions)
        item["_bag_timestamp_ns"] = row.bag_timestamp_ns
        candidates.append(item)
    return candidates


def identity(candidate: dict[str, object]) -> tuple[object, ...]:
    return (
        candidate.get("run_id"), candidate.get("task_id"), candidate.get("attempt_count"),
        candidate.get("instruction_fingerprint"), candidate.get("task_set_fingerprint"),
        candidate.get("active_task_fingerprint"), candidate.get("generation_id"),
    )


def is_valid_candidate(candidate: dict[str, object], latency_limit_ms: float) -> bool:
    return bool(
        candidate.get("valid") is True and candidate.get("context_valid") is True
        and float(candidate.get("response_latency_ms", math.inf)) <= latency_limit_ms
    )


def is_policy_dispatch(payload: dict[str, object]) -> bool:
    decision = payload.get("decision")
    return bool(
        isinstance(decision, dict) and decision.get("base_source") == POLICY_SOURCE
        and decision.get("manipulation_source") == POLICY_SOURCE
    )


def reconstruct(
    candidates: Sequence[dict[str, object]], dispatch_rows: Sequence[Row],
    final_rows: Sequence[Row], period_ns: int, latency_limit_ms: float,
) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    valid_candidates = [item for item in candidates if is_valid_candidate(item, latency_limit_ms)]
    if not valid_candidates:
        raise ValueError("no valid candidate")
    t0_ns = int(valid_candidates[0]["_bag_timestamp_ns"])
    finals = {
        int(row.payload["sequence"]): action19(row.payload["action"])
        for row in final_rows if isinstance(row.payload, dict)
    }
    events: list[tuple[int, int, object]] = [
        (int(item["_bag_timestamp_ns"]), 0, item) for item in candidates
    ]
    events.extend(
        (int(row.payload["timestamp_ns"]), 1, row.payload)
        for row in dispatch_rows if isinstance(row.payload, dict) and is_policy_dispatch(row.payload)
    )
    events.sort(key=lambda item: (item[0], item[1]))
    pending: dict[str, object] | None = None
    handoff_source: tuple[float, ...] | None = None
    handoff_start_ns: int | None = None
    last_output: tuple[float, ...] | None = None
    last_identity: tuple[object, ...] | None = None
    output: list[dict[str, object]] = []
    alpha_spreads: list[float] = []
    fit_residuals: list[float] = []
    selected_index_offsets: list[int] = []

    for timestamp_ns, kind, value in events:
        if kind == 0:
            candidate = value
            assert isinstance(candidate, dict)
            if not is_valid_candidate(candidate, latency_limit_ms):
                pending = None
                handoff_source = None
                last_output = None
                last_identity = None
                continue
            candidate_identity = identity(candidate)
            handoff_source = (
                last_output if last_output is not None and last_identity == candidate_identity else None
            )
            handoff_start_ns = timestamp_ns if handoff_source is not None else None
            if handoff_source is None:
                last_output = None
                last_identity = None
            pending = candidate
            continue

        dispatch = value
        assert isinstance(dispatch, dict)
        if pending is None:
            raise ValueError("policy dispatch without recoverable candidate")
        nominal_index = (timestamp_ns - int(pending["_bag_timestamp_ns"])) // period_ns
        if nominal_index < 0 or nominal_index >= HORIZON:
            raise ValueError("policy dispatch outside candidate horizon")
        actions = pending["_actions"]
        assert isinstance(actions, tuple)
        decision = dispatch["decision"]
        assert isinstance(decision, dict)
        mask_value = decision.get("clipped_mask")
        if not isinstance(mask_value, list) or len(mask_value) != ACTION_DIM:
            raise ValueError("invalid clipped mask")
        mask = tuple(bool(value) for value in mask_value)
        final = finals[int(dispatch["final_action_sequence"])]
        eligible = [i for i in range(ACTION_DIM) if not mask[i] and i not in GRIPPERS]
        candidate_indices = range(max(0, nominal_index - 1), min(HORIZON, nominal_index + 2))
        target = actions[nominal_index]
        raw = target
        best_score = math.inf
        best_index = nominal_index
        best_alpha: float | None = None
        if handoff_source is not None and handoff_start_ns is not None:
            handoff_age_ns = timestamp_ns - handoff_start_ns
            # The callback receipt timestamp is absent. Search only the nominal
            # row and its immediate neighbours, then select the shared-alpha fit
            # that best explains independently recorded unclipped dimensions.
            if handoff_age_ns < (HANDOFF_STEPS + 1) * period_ns:
                for candidate_index in candidate_indices:
                    candidate_target = actions[candidate_index]
                    pairs = []
                    for dimension in eligible:
                        delta = candidate_target[dimension] - handoff_source[dimension]
                        if abs(delta) > 1e-8:
                            pairs.append((delta, final[dimension] - handoff_source[dimension]))
                    denominator = sum(delta * delta for delta, _observed in pairs)
                    if not pairs or denominator <= 0.0:
                        continue
                    alpha = min(
                        1.0, max(0.0, sum(delta * observed for delta, observed in pairs) / denominator)
                    )
                    predicted = tuple(
                        candidate_target[i] if i in GRIPPERS
                        else handoff_source[i] + alpha * (candidate_target[i] - handoff_source[i])
                        for i in range(ACTION_DIM)
                    )
                    score = sum((predicted[i] - final[i]) ** 2 for i in eligible)
                    if score < best_score:
                        best_score = score
                        best_index = candidate_index
                        best_alpha = alpha
                        target = candidate_target
                        raw = predicted
                if best_alpha is None:
                    raise ValueError("cannot infer handoff alpha")
                pairs = []
                for dimension in eligible:
                    delta = target[dimension] - handoff_source[dimension]
                    if abs(delta) > 1e-8:
                        pairs.append((delta, final[dimension] - handoff_source[dimension]))
                stable_ratios = [observed / delta for delta, observed in pairs if abs(delta) > 1e-3]
                alpha_spreads.append(
                    (percentile(stable_ratios, 0.95) or 0.0)
                    - (percentile(stable_ratios, 0.05) or 0.0)
                    if stable_ratios else 0.0
                )
                fit_residuals.extend(
                    abs(raw[i] - final[i]) for i in range(ACTION_DIM)
                    if not mask[i] and i not in GRIPPERS
                )
            else:
                handoff_source = None
                handoff_start_ns = None
        if handoff_source is None:
            for candidate_index in candidate_indices:
                candidate_target = actions[candidate_index]
                score = sum((candidate_target[i] - final[i]) ** 2 for i in eligible)
                if score < best_score:
                    best_score = score
                    best_index = candidate_index
                    target = candidate_target
                    raw = candidate_target
            fit_residuals.extend(abs(raw[i] - final[i]) for i in eligible)
        selected_index_offsets.append(best_index - nominal_index)
        raw = tuple(final[i] if not mask[i] else raw[i] for i in range(ACTION_DIM))
        last_output = raw
        last_identity = identity(pending)
        output.append({
            "timestamp_ns": timestamp_ns, "relative_s": (timestamp_ns - t0_ns) / 1e9,
            "raw": raw, "final": final, "clipped_mask": mask,
            "nominal_action_index": nominal_index, "selected_action_index": best_index,
        })

    first_dispatch_ns = int(output[0]["timestamp_ns"])
    base_mask_matches = all(
        (abs(row["raw"][0]) > 0.45) == row["clipped_mask"][0]
        and (abs(row["raw"][1]) > 1.20) == row["clipped_mask"][1]
        for row in output
    )
    return t0_ns, output, {
        "valid_candidate_request_id": valid_candidates[0].get("request_id"),
        "candidate_bag_timestamp_ns": t0_ns,
        "candidates_before_t0": [
            {"request_id": item.get("request_id"),
             "response_latency_ms": item.get("response_latency_ms"),
             "passed_runtime_gate": is_valid_candidate(item, latency_limit_ms)}
            for item in candidates if int(item["_bag_timestamp_ns"]) < t0_ns
        ],
        "first_policy_dispatch_timestamp_ns": first_dispatch_ns,
        "first_dispatch_after_t0_ms": (first_dispatch_ns - t0_ns) / 1e6,
        "candidate_callback_receipt_timestamp_archived": False,
        "time_origin_error": (
            "actual TeamClient callback receipt time is absent; the candidate SQLite bag timestamp "
            "is used as its observable proxy, so absolute receipt-time error is not bounded by this artifact"
        ),
        "scheduler_method": (
            "24 Hz row selection plus shared two-step handoff alpha inferred from unclipped dimensions; "
            "unclipped FinalAction components are exact pre-clamp values"
        ),
        "handoff_fit_ticks": len(alpha_spreads),
        "max_stable_alpha_ratio_p95_minus_p05": max(alpha_spreads, default=0.0),
        "max_unclipped_fit_abs_residual": max(fit_residuals, default=0.0),
        "selected_index_offset_counts": {
            str(offset): selected_index_offsets.count(offset) for offset in (-1, 0, 1)
        },
        "e1_base_clip_masks_match_every_tick": base_mask_matches,
        "clipped_raw_limitation": (
            "clipped raw components are scheduler reconstructions, not independently logged pre-clamp values"
        ),
    }


def timed_stats(rows: Sequence[dict[str, object]], start_s: float, end_s: float) -> dict[str, object]:
    selected = [row for row in rows if start_s <= float(row["relative_s"]) < end_s]
    return {
        "raw": {
            "base_v": stats([row["raw"][0] for row in selected]),
            "base_w": stats([row["raw"][1] for row in selected]),
            "slide": stats([row["raw"][2] for row in selected]),
        },
        "final_action": {
            "base_v": stats([row["final"][0] for row in selected]),
            "base_w": stats([row["final"][1] for row in selected]),
            "slide": stats([row["final"][2] for row in selected]),
        },
    }


def sensor_stats(
    rows: Sequence[tuple[float, tuple[float, ...]]], start_s: float, end_s: float,
    dimensions: Sequence[str],
) -> dict[str, object]:
    selected = [values for relative_s, values in rows if start_s <= relative_s < end_s]
    return {
        name: stats([values[index] for values in selected])
        for index, name in enumerate(dimensions)
    }


def continuous_runs(
    rows: Sequence[dict[str, object]], predicate, nominal_period_s: float,
) -> list[tuple[float, float]]:
    qualifying = [float(row["relative_s"]) for row in rows if predicate(row)]
    if not qualifying:
        return []
    runs = []
    start = previous = qualifying[0]
    for current in qualifying[1:]:
        if current - previous > 1.5 * nominal_period_s:
            runs.append((start, previous + nominal_period_s))
            start = current
        previous = current
    runs.append((start, previous + nominal_period_s))
    return runs


def threshold_audit(
    rows: Sequence[dict[str, object]], source: str, threshold: float,
    nominal_period_s: float,
) -> dict[str, object]:
    index = 1
    predicate = lambda row: row[source][index] >= threshold
    runs = continuous_runs(rows, predicate, nominal_period_s)
    in_10_20 = [row for row in rows if 10.0 <= float(row["relative_s"]) < 20.0]
    qualifying = [row for row in in_10_20 if predicate(row)]
    return {
        "first_time_s": runs[0][0] if runs else None,
        "longest_continuous_duration_s": max((end - start for start, end in runs), default=0.0),
        "share_10_20_pct": 100.0 * len(qualifying) / len(in_10_20) if in_10_20 else None,
        "continuity_definition": "adjacent 40 Hz policy ticks, gap <=37.5 ms; duration includes one 25 ms tick",
    }


def phase_rows(
    policy_rows: Sequence[dict[str, object]], odom: Sequence[tuple[float, tuple[float, ...]]],
    slide_state: Sequence[tuple[float, tuple[float, ...]]],
) -> dict[str, object]:
    result = {}
    for name, start, end in PHASES:
        item = timed_stats(policy_rows, start, end)
        item["odom"] = sensor_stats(odom, start, end, ("linear_x", "angular_z"))
        item["slide_state"] = sensor_stats(slide_state, start, end, ("slide",))
        result[name] = item
    return result


def find_spin_onset(rows: Sequence[dict[str, object]], nominal_period_s: float) -> float | None:
    runs = continuous_runs(
        rows,
        lambda row: row["final"][1] >= 1.0 and abs(row["final"][0]) <= 0.08,
        nominal_period_s,
    )
    for start, end in runs:
        if end - start >= 1.0:
            return start
    return None


def flatten_csv(path: Path, phases: dict[str, object], onset_windows: dict[str, object]) -> None:
    columns = ("window", "stream", "dimension", "count", "mean", "p05", "p50", "p95", "min", "max")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for window, summary in {**phases, **onset_windows}.items():
            for source, dimensions in summary.items():
                for dimension, values in dimensions.items():
                    writer.writerow({"window": window, "stream": source, "dimension": dimension, **values})


def fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def markdown(report: dict[str, object]) -> str:
    lines = [
        "# M10-B1-A.3-A Runtime Early-Phase Trajectory Audit", "",
        "## Scope and time origin", "",
        "This report contains runtime facts only. It does not diagnose checkpoint quality or assign causality.", "",
        f"`t=0` is candidate request `{report['time_origin']['valid_candidate_request_id']}` at SQLite bag timestamp "
        f"`{report['time_origin']['candidate_bag_timestamp_ns']}` ns. The first policy dispatch follows by "
        f"{fmt(report['time_origin']['first_dispatch_after_t0_ms'])} ms.", "",
        report["time_origin"]["time_origin_error"] + ". " + report["time_origin"]["scheduler_method"] + ".", "",
        f"Handoff fit ticks: {report['time_origin']['handoff_fit_ticks']}; maximum per-tick stable "
        f"alpha-ratio P95-P05 {report['time_origin']['max_stable_alpha_ratio_p95_minus_p05']:.6g}; max unclipped fit residual "
        f"{report['time_origin']['max_unclipped_fit_abs_residual']:.6g}.", "",
        f"Nominal action-row offsets selected by the fit: {report['time_origin']['selected_index_offset_counts']}; "
        f"E1 base clip masks match every reconstructed tick: "
        f"{report['time_origin']['e1_base_clip_masks_match_every_tick']}.", "",
        report["time_origin"]["clipped_raw_limitation"] + ".", "",
        f"First-20-second policy timing: {report['policy_tick_timing_first20s']['tick_count']} ticks; "
        f"{report['policy_tick_timing_first20s']['continuity_gap_count']} gap(s) over 37.5 ms; "
        f"maximum gap {report['policy_tick_timing_first20s']['max_gap_s']:.6f} s.", "",
        "## Four five-second phases", "",
        "Each cell is `mean / P05 / P50 / P95 / min / max`.", "",
        "| Phase | Stream | Dimension | Statistics |", "|---|---|---|---|",
    ]
    for phase, summary in report["phases"].items():
        for source, dimensions in summary.items():
            for dimension, values in dimensions.items():
                ordered = " / ".join(fmt(values[key]) for key in ("mean", "p05", "p50", "p95", "min", "max"))
                lines.append(f"| {phase} | {source} | {dimension} | {ordered} |")
    lines.extend(["", "## Sustained positive rotation", "",
                  "Durations use adjacent 40 Hz policy ticks with gaps no larger than 37.5 ms and include one 25 ms tick.", "",
                  "| Source | Threshold | First time (s) | Longest run (s) | Share in 10–20 s |",
                  "|---|---:|---:|---:|---:|"])
    for source, thresholds in report["sustained_rotation"].items():
        for threshold, values in thresholds.items():
            lines.append(
                f"| {source} | {threshold} | {fmt(values['first_time_s'])} | "
                f"{fmt(values['longest_continuous_duration_s'])} | {fmt(values['share_10_20_pct'])}% |"
            )
    onset = report["spin_onset"]
    lines.extend(["", "## Spin onset", "", f"`SPIN_ONSET_S = {fmt(onset['spin_onset_s'])}`", ""])
    if onset["spin_onset_s"] is None:
        lines.append("No interval met the stated one-second onset definition.")
    else:
        lines.extend([
            "The first qualifying run continuously has `FinalAction base_w >= 1.0` and "
            "`abs(FinalAction base_v) <= 0.08` for at least one second.", "",
            "Each cell below is `mean / P05 / P50 / P95 / min / max`.", "",
            "| Window | Stream | Dimension | Statistics |", "|---|---|---|---|",
        ])
        for window, summary in onset["windows"].items():
            for source, dimensions in summary.items():
                for dimension, values in dimensions.items():
                    ordered = " / ".join(
                        fmt(values[key]) for key in ("mean", "p05", "p50", "p95", "min", "max")
                    )
                    lines.append(f"| {window} | {source} | {dimension} | {ordered} |")
    slide = report["slide_negative_first20s"]
    lines.extend(["", "## Raw slide below zero in first 20 seconds", "",
                  f"Count: {slide['count']}; first: {fmt(slide['first_time_s'])} s; last: "
                  f"{fmt(slide['last_time_s'])} s; minimum: {fmt(slide['min'])} m.", "",
                  "This is an independent observation, not a claimed primary cause.", "",
                  "## Offline declaration", "",
                  "The analyzer read rosbag2 SQLite directly in read-only mode. No ROS graph, Server, "
                  "TeamClient, Adapter or MuJoCo process was started, and production runtime was not modified.", ""])
    return "\n".join(lines)


def analyze(archive: Path) -> dict[str, object]:
    archive_sha = sha256_file(archive)
    with tempfile.TemporaryDirectory(prefix="m10_b1a3.", dir="/tmp") as temp_dir:
        db = safe_extract(archive, Path(temp_dir))
        config_files = list(Path(temp_dir).rglob("team_sorting_m10_b1a2_e1.yaml"))
        if len(config_files) != 1:
            raise ValueError("archived experiment config missing or ambiguous")
        config = yaml.safe_load(config_files[0].read_text(encoding="utf-8"))
        if config.get("simulation_experiment", {}).get("profile") != "m10_b1a2_e1":
            raise ValueError("archive is not m10_b1a2_e1")
        rows = read_bag(db)
    period_ns = int(round(float(config["pi05_policy_control"]["action_step_period_s"]) * 1e9))
    latency_ms = float(config["pi05_policy_control"]["max_policy_response_latency_ms"])
    candidates = parse_candidates(rows[POLICY_TOPIC])
    t0_ns, policy_rows, reconstruction = reconstruct(
        candidates, rows[DISPATCH_TOPIC], rows[FINAL_TOPIC], period_ns, latency_ms
    )
    first20 = [row for row in policy_rows if 0.0 <= float(row["relative_s"]) < 20.0]
    first20_times = [float(row["relative_s"]) for row in first20]
    continuity_gaps = [
        {"start_s": before, "end_s": after, "duration_s": after - before}
        for before, after in zip(first20_times, first20_times[1:]) if after - before > 0.0375
    ]
    odom = []
    odom_delay_ms = []
    for row in rows[ODOM_TOPIC]:
        timestamp_ns, linear_x, angular_z = row.payload
        odom.append(((timestamp_ns - t0_ns) / 1e9, (linear_x, angular_z)))
        odom_delay_ms.append((row.bag_timestamp_ns - timestamp_ns) / 1e6)
    slide_state = []
    joint_delay_ms = []
    for row in rows[JOINT_TOPIC]:
        timestamp_ns, names, positions = row.payload
        if "slide_joint" not in names:
            raise ValueError("slide_joint absent from JointState")
        slide_state.append(((timestamp_ns - t0_ns) / 1e9, (positions[names.index("slide_joint")],)))
        joint_delay_ms.append((row.bag_timestamp_ns - timestamp_ns) / 1e6)
    phases = phase_rows(first20, odom, slide_state)
    nominal_control_period_s = 0.025
    sustained = {
        source: {
            str(threshold): threshold_audit(first20, source, threshold, nominal_control_period_s)
            for threshold in (0.8, 1.0, 1.19)
        }
        for source in ("raw", "final")
    }
    onset_s = find_spin_onset(first20, nominal_control_period_s)
    onset_windows: dict[str, object] = {}
    if onset_s is not None:
        for name, start, end in (
            ("spin_onset_minus2_0", onset_s - 2.0, onset_s),
            ("spin_onset_0_plus2", onset_s, onset_s + 2.0),
        ):
            item = timed_stats(first20, start, end)
            # Onset output is intentionally limited to requested base and Odom dimensions.
            item["raw"].pop("slide")
            item["final_action"].pop("slide")
            item["odom"] = sensor_stats(odom, start, end, ("linear_x", "angular_z"))
            onset_windows[name] = item
    negatives = [row for row in first20 if row["raw"][2] < 0.0]
    return {
        "schema_name": "M10B1A3RuntimeEarlyPhaseAudit", "schema_version": 1,
        "input": {"archive": str(archive.resolve()), "archive_sha256": archive_sha,
                  "read_only_direct_sqlite": True, "candidate_count": len(candidates),
                  "policy_ticks_first20s": len(first20)},
        "time_origin": {**reconstruction,
            "odom_bag_minus_header_ms": stats(odom_delay_ms),
            "joint_state_bag_minus_header_ms": stats(joint_delay_ms)},
        "policy_tick_timing_first20s": {
            "tick_count": len(first20), "continuity_gap_threshold_s": 0.0375,
            "continuity_gap_count": len(continuity_gaps),
            "max_gap_s": max((item["duration_s"] for item in continuity_gaps), default=0.0),
            "gaps": continuity_gaps,
        },
        "phases": phases, "sustained_rotation": sustained,
        "spin_onset": {"definition": "FinalAction base_w>=1.0 and abs(base_v)<=0.08 continuously >=1.0 s",
                       "spin_onset_s": onset_s, "windows": onset_windows},
        "slide_negative_first20s": {
            "count": len(negatives),
            "first_time_s": float(negatives[0]["relative_s"]) if negatives else None,
            "last_time_s": float(negatives[-1]["relative_s"]) if negatives else None,
            "min": min((row["raw"][2] for row in negatives), default=None),
            "interpretation": "independent runtime observation; no causal assignment",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runtime_phase_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    flatten_csv(args.output_dir / "runtime_first20s.csv", report["phases"], report["spin_onset"]["windows"])
    (args.output_dir / "M10_B1A3_RUNTIME_PHASE_AUDIT.md").write_text(
        markdown(report), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
