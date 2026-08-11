from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_m10_b1a3_runtime_phase.py"
SPEC = importlib.util.spec_from_file_location("m10_b1a3_runtime_phase", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _row(t: float, v: float = 0.0, w: float = 0.0) -> dict[str, object]:
    action = (v, w, *(0.0,) * 17)
    return {"relative_s": t, "raw": action, "final": action}


def test_percentile_and_stats_are_linear_and_ordered() -> None:
    result = MODULE.stats([0.0, 10.0])
    assert result["p05"] == 0.5
    assert result["p50"] == 5.0
    assert result["p95"] == 9.5
    assert result["min"] == 0.0 and result["max"] == 10.0


def test_phase_intervals_are_lower_inclusive_upper_exclusive() -> None:
    rows = [_row(0.0, 1.0), _row(4.999, 2.0), _row(5.0, 3.0)]
    first = MODULE.timed_stats(rows, 0.0, 5.0)
    second = MODULE.timed_stats(rows, 5.0, 10.0)
    assert first["raw"]["base_v"]["count"] == 2
    assert second["raw"]["base_v"]["count"] == 1


def test_continuous_runs_split_on_more_than_one_and_half_periods() -> None:
    rows = [_row(0.0, w=1.0), _row(0.025, w=1.0), _row(0.075, w=1.0)]
    runs = MODULE.continuous_runs(rows, lambda row: row["final"][1] >= 1.0, 0.025)
    assert len(runs) == 2
    assert math.isclose(runs[0][1] - runs[0][0], 0.05)


def test_spin_onset_requires_at_least_one_second_and_low_linear_speed() -> None:
    qualifying = [_row(index * 0.025, v=0.08, w=1.0) for index in range(40)]
    assert MODULE.find_spin_onset(qualifying, 0.025) == 0.0
    too_short = qualifying[:-1]
    assert MODULE.find_spin_onset(too_short, 0.025) is None
    fast = [_row(index * 0.025, v=0.081, w=1.0) for index in range(40)]
    assert MODULE.find_spin_onset(fast, 0.025) is None
