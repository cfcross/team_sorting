#!/usr/bin/env python3
"""Offline M10-B1-A.3-B runtime state19 recovery audit.

The archive does not contain the policy observation timestamp or serialized
state19.  This analyzer therefore emits the closest reconstruction supported
by the artifact and labels it incomplete instead of presenting it as exact.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
from typing import Sequence


ACTION_DIM = 19
PHASES = (("0-5s", 0.0, 5.0), ("5-10s", 5.0, 10.0),
          ("10-15s", 10.0, 15.0), ("15-20s", 15.0, 20.0))
STATE_PREFIX = ("base_v_feedback", "base_w_feedback")


def load_phase_module():
    path = Path(__file__).with_name("analyze_m10_b1a3_runtime_phase.py")
    spec = importlib.util.spec_from_file_location("m10_b1a3_runtime_phase", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_joint_names(contracts_path: Path) -> tuple[str, ...]:
    tree = ast.parse(contracts_path.read_text(encoding="utf-8"), filename=str(contracts_path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "JOINT_NAMES" for target in targets):
                value = ast.literal_eval(node.value)
                if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
                    raise ValueError("Adapter JOINT_NAMES is not a tuple of strings")
                if len(value) != 17 or len(set(value)) != 17:
                    raise ValueError("Adapter JOINT_NAMES is not 17 unique joints")
                return value
    raise ValueError("Adapter JOINT_NAMES not found")


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
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
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "std": statistics.pstdev(finite) if finite else None,
        "min": min(finite) if finite else None,
        "p05": percentile(finite, 0.05),
        "p50": percentile(finite, 0.50),
        "p95": percentile(finite, 0.95),
        "max": max(finite) if finite else None,
    }


def latest_before(rows, timestamp_ns: int):
    timestamps = [row.bag_timestamp_ns for row in rows]
    index = bisect.bisect_right(timestamps, timestamp_ns) - 1
    if index < 0:
        raise ValueError("no sensor sample precedes estimated predict start")
    return rows[index]


def canonical_positions(payload: object, joint_names: tuple[str, ...]) -> tuple[int, tuple[float, ...]]:
    if not isinstance(payload, tuple) or len(payload) != 3:
        raise ValueError("unexpected decoded JointState")
    header_ns, names, positions = payload
    if len(names) != len(set(names)):
        raise ValueError("duplicate JointState names")
    mapping = dict(zip(names, positions))
    missing = [name for name in joint_names if name not in mapping]
    if missing:
        raise ValueError(f"JointState missing canonical joints: {missing}")
    values = tuple(float(mapping[name]) for name in joint_names)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite JointState position")
    return int(header_ns), values


def build_observation_rows(bag, candidates, t0_ns: int, latency_limit_ms: float,
                           joint_names: tuple[str, ...]) -> list[dict[str, object]]:
    phase = load_phase_module()
    odom_rows = bag[phase.ODOM_TOPIC]
    joint_rows = bag[phase.JOINT_TOPIC]
    output = []
    for candidate in candidates:
        candidate_ns = int(candidate["_bag_timestamp_ns"])
        relative_s = (candidate_ns - t0_ns) / 1e9
        if relative_s < 0.0 or relative_s >= 20.0:
            continue
        latency_ms = float(candidate["response_latency_ms"])
        estimated_predict_start_ns = candidate_ns - round(latency_ms * 1e6)
        odom = latest_before(odom_rows, estimated_predict_start_ns)
        joint = latest_before(joint_rows, estimated_predict_start_ns)
        odom_header_ns, linear_x, angular_z = odom.payload
        joint_header_ns, positions = canonical_positions(joint.payload, joint_names)
        state = tuple(f32(value) for value in (linear_x, angular_z, *positions))
        actions = candidate["_actions"]
        raw_action = tuple(float(value) for value in actions[0])
        output.append({
            "time_s": relative_s,
            "candidate_bag_timestamp_ns": candidate_ns,
            "request_id": int(candidate["request_id"]),
            "response_latency_ms": latency_ms,
            "passed_runtime_gate": bool(phase.is_valid_candidate(candidate, latency_limit_ms)),
            "estimated_predict_start_ns": estimated_predict_start_ns,
            "estimated_predict_start_time_s": (estimated_predict_start_ns - t0_ns) / 1e9,
            "odom_bag_timestamp_ns": odom.bag_timestamp_ns,
            "odom_header_timestamp_ns": int(odom_header_ns),
            "odom_age_at_estimated_predict_start_ms":
                (estimated_predict_start_ns - odom.bag_timestamp_ns) / 1e6,
            "joint_bag_timestamp_ns": joint.bag_timestamp_ns,
            "joint_header_timestamp_ns": joint_header_ns,
            "joint_age_at_estimated_predict_start_ms":
                (estimated_predict_start_ns - joint.bag_timestamp_ns) / 1e6,
            "state": state,
            "raw_action": raw_action,
            "raw_action_definition": "candidate action chunk row 0",
        })
    return output


def phase_statistics(rows: Sequence[dict[str, object]], dimensions: Sequence[str]) -> dict[str, object]:
    result = {}
    for phase_name, start_s, end_s in PHASES:
        selected = [row for row in rows if start_s <= float(row["time_s"]) < end_s]
        result[phase_name] = {
            f"state[{index}] {name}": stats([row["state"][index] for row in selected])
            for index, name in enumerate(dimensions)
        }
    return result


def analyze(archive: Path, adapter_root: Path) -> dict[str, object]:
    phase = load_phase_module()
    contracts = adapter_root / "ros2_ws/src/mmk2_pi05_bridge/mmk2_pi05_bridge/contracts.py"
    mapper = adapter_root / "ros2_ws/src/mmk2_pi05_bridge/mmk2_pi05_bridge/joint_state_mapper.py"
    observation_adapter = adapter_root / "ros2_ws/src/mmk2_pi05_bridge/mmk2_pi05_bridge/policy_observation_adapter.py"
    shadow_node = adapter_root / "ros2_ws/src/mmk2_pi05_bridge/mmk2_pi05_bridge/policy_shadow_node.py"
    policy_client = adapter_root / "ros2_ws/src/mmk2_pi05_bridge/mmk2_pi05_bridge/policy_client.py"
    bridge = adapter_root / "scripts/pi05_http_bridge.py"
    sources = (contracts, mapper, observation_adapter, shadow_node, policy_client, bridge)
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise ValueError(f"Adapter source missing: {missing}")
    joint_names = adapter_joint_names(contracts)
    dimensions = (*STATE_PREFIX, *joint_names)

    with tempfile.TemporaryDirectory(prefix="m10_b1a3b.", dir="/tmp") as temp_dir:
        db = phase.safe_extract(archive, Path(temp_dir))
        configs = list(Path(temp_dir).rglob("team_sorting_m10_b1a2_e1.yaml"))
        if len(configs) != 1:
            raise ValueError("archived E1 config missing or ambiguous")
        config = phase.yaml.safe_load(configs[0].read_text(encoding="utf-8"))
        bag = phase.read_bag(db)
    candidates = phase.parse_candidates(bag[phase.POLICY_TOPIC])
    latency_limit_ms = float(config["pi05_policy_control"]["max_policy_response_latency_ms"])
    valid = [candidate for candidate in candidates if phase.is_valid_candidate(candidate, latency_limit_ms)]
    if not valid:
        raise ValueError("no candidate passed the runtime gate")
    t0_ns = int(valid[0]["_bag_timestamp_ns"])
    rows = build_observation_rows(bag, candidates, t0_ns, latency_limit_ms, joint_names)
    paired_headers = sum(
        1 for row in rows if row["odom_header_timestamp_ns"] == row["joint_header_timestamp_ns"]
    )
    first15 = [row for row in rows if float(row["time_s"]) < 15.0]
    return {
        "schema_name": "m10_b1a3b_runtime_state_audit",
        "schema_version": 1,
        "runtime_state_recovery": "INCOMPLETE",
        "archive": {"path": str(archive.resolve()), "sha256": sha256_file(archive)},
        "time_origin": {
            "definition": "first policy_control_candidate passing the runtime gate",
            "request_id": int(valid[0]["request_id"]),
            "candidate_sqlite_bag_timestamp_ns": t0_ns,
            "same_as_m10_b1a3a": True,
            "actual_candidate_callback_receipt_timestamp_archived": False,
            "proxy_error_bound_ms": None,
            "proxy_limitation": (
                "The exact TeamClient callback receipt/activation timestamp is absent; the candidate "
                "SQLite bag timestamp is the same observable proxy used by M10-B1-A.3-A"
            ),
        },
        "adapter_source": {
            "root": str(adapter_root.resolve()),
            "git_head": git_head(adapter_root),
            "files": {str(path.relative_to(adapter_root)): sha256_file(path) for path in sources},
            "canonical_joint_order": list(joint_names),
            "state19_semantics": ["odom.twist.twist.linear.x", "odom.twist.twist.angular.z", *joint_names],
            "model_input_dtype": "float32",
        },
        "recovery_limit": {
            "exact_observation_payload_archived": False,
            "observation_timestamp_ns_archived_in_candidate": False,
            "adapter_latency_trace_archived": False,
            "method": (
                "estimate PolicyClient.predict start as candidate SQLite bag timestamp minus "
                "response_latency_ms; independently select the latest recorded Odom and JointState "
                "by bag receipt timestamp, apply Adapter canonical ordering, then quantize state19 to float32"
            ),
            "why_incomplete": (
                "response_latency_ms begins inside PolicyClient.predict after snapshot/adaptation, while the "
                "candidate bag timestamp occurs after response completion; the missing pre-request and "
                "post-response intervals leave the exact cached sensor frames unidentifiable"
            ),
            "timing_error_bound_ms": None,
            "values_are_estimates_not_exact_observation_claims": True,
        },
        "coverage": {
            "candidate_observations_first20s": len(rows),
            "candidate_observations_first15s": len(first15),
            "first_time_s": rows[0]["time_s"] if rows else None,
            "last_time_s": rows[-1]["time_s"] if rows else None,
            "runtime_gate_pass_count_first20s": sum(bool(row["passed_runtime_gate"]) for row in rows),
            "odom_joint_equal_header_timestamp_count": paired_headers,
            "odom_joint_equal_header_timestamp_share": paired_headers / len(rows) if rows else None,
        },
        "phase_statistics": phase_statistics(rows, dimensions),
        "observations": rows,
    }


def fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def markdown(report: dict[str, object]) -> str:
    origin = report["time_origin"]
    coverage = report["coverage"]
    limitation = report["recovery_limit"]
    lines = [
        "# M10-B1-A.3-B Runtime State19 Early-Phase Audit", "",
        "## Recovery result", "",
        "`RUNTIME_STATE_RECOVERY = INCOMPLETE`", "",
        "This is a runtime-only statement. It makes no checkpoint or policy-quality judgment.", "",
        f"The common `t=0` is request `{origin['request_id']}` at candidate SQLite bag timestamp "
        f"`{origin['candidate_sqlite_bag_timestamp_ns']}` ns.", "",
        origin["proxy_limitation"] + "; its absolute proxy error is not bounded by the archive.", "",
        limitation["why_incomplete"] + ". The archive provides no finite bound for that timing error.", "",
        "The tables and row-level export are therefore the closest observable reconstruction, not a claim "
        "that the exact per-request cached sensor frame was recovered.", "",
        "## Actual Adapter state contract", "",
        "The inspected Adapter constructs float32 state19 as `Odom linear.x`, `Odom angular.z`, followed by "
        "17 JointState positions rearranged by `JOINT_NAMES`. Commands and FinalAction are not used as state.", "",
        "| State index | Runtime source |", "|---:|---|",
    ]
    for index, name in enumerate(report["adapter_source"]["state19_semantics"]):
        lines.append(f"| {index} | {name} |")
    lines.extend([
        "", "## Coverage and row semantics", "",
        f"First 20 s contains {coverage['candidate_observations_first20s']} candidate-associated valid "
        f"policy responses; {coverage['candidate_observations_first15s']} lie in 0–15 s. "
        f"{coverage['runtime_gate_pass_count_first20s']} pass the runtime latency gate.", "",
        f"Odom and JointState selected rows have equal message-header timestamps in "
        f"{coverage['odom_joint_equal_header_timestamp_count']}/{coverage['candidate_observations_first20s']} cases.", "",
        "`time_s` is candidate activation time relative to the common `t=0`. `raw_action[0..18]` is row 0 "
        "of that candidate's model action chunk. The estimated request time and both selected sensor "
        "timestamps/ages are retained in CSV and JSON.", "",
        "## Four five-second phases", "",
        "Each cell is `count / mean / std / min / P05 / P50 / P95 / max`.", "",
        "| Phase | Dimension | Statistics |", "|---|---|---|",
    ])
    for phase_name, dimensions in report["phase_statistics"].items():
        for dimension, values in dimensions.items():
            ordered = " / ".join(fmt(values[key]) for key in
                                 ("count", "mean", "std", "min", "p05", "p50", "p95", "max"))
            lines.append(f"| {phase_name} | {dimension} | {ordered} |")
    lines.extend([
        "", "## Offline declaration", "",
        "The analysis read the archive and rosbag2 SQLite directly. It did not start a ROS graph, Server, "
        "TeamClient, Adapter, or MuJoCo, and it did not modify production runtime.", "",
    ])
    return "\n".join(lines)


def write_csv(path: Path, report: dict[str, object]) -> None:
    fixed = [
        "time_s", "candidate_bag_timestamp_ns", "request_id", "response_latency_ms",
        "passed_runtime_gate", "estimated_predict_start_ns", "estimated_predict_start_time_s",
        "odom_bag_timestamp_ns", "odom_header_timestamp_ns", "odom_age_at_estimated_predict_start_ms",
        "joint_bag_timestamp_ns", "joint_header_timestamp_ns", "joint_age_at_estimated_predict_start_ms",
    ]
    columns = fixed + [f"state_{index}" for index in range(ACTION_DIM)] + [
        f"raw_action_{index}" for index in range(ACTION_DIM)
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for observation in report["observations"]:
            row = {key: observation[key] for key in fixed}
            row.update({f"state_{index}": value for index, value in enumerate(observation["state"])})
            row.update({f"raw_action_{index}": value for index, value in enumerate(observation["raw_action"])})
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.archive, args.adapter_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "runtime_state_first20s.json"
    csv_path = args.output_dir / "runtime_state_first20s.csv"
    md_path = args.output_dir / "M10_B1A3B_RUNTIME_STATE_AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    write_csv(csv_path, report)
    md_path.write_text(markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
