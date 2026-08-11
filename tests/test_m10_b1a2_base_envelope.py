from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_m10_b1a2_base_envelope.py"
SPEC = importlib.util.spec_from_file_location("m10_b1a2_base_envelope", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_percentile_uses_linear_interpolation() -> None:
    assert MODULE.percentile([0.0, 10.0], 0.95) == 9.5
    assert MODULE.percentile([], 0.50) is None


def test_counterfactual_counts_and_distortion_definitions() -> None:
    rows = ((-0.6, 0.2), (0.1, 1.0), (0.5, -1.5), (0.0, 0.0))
    result = MODULE.envelope_stats(rows, 0.5, 1.0)
    assert result["base_v"]["clipping_ticks"] == 1
    assert result["base_w"]["clipping_ticks"] == 1
    assert result["combined"]["at_least_one_changed_ticks"] == 2
    assert result["combined"]["both_changed_ticks"] == 0
    assert result["combined"]["total_l1_distortion"] == 0.6
    assert result["combined"]["mean_l1_distortion_per_tick"] == 0.15


def test_threshold_equality_is_not_clipping() -> None:
    result = MODULE.dimension_counterfactual([-0.25, 0.25, 0.250001], 0.25)
    assert result["clipping_ticks"] == 1
    assert math.isclose(
        result["exceedance_magnitude_clipped_ticks"]["max"], 0.000001,
        rel_tol=0.0, abs_tol=1e-15,
    )
