from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_m10_b1a3b_runtime_state.py"
SPEC = importlib.util.spec_from_file_location("m10_b1a3b_runtime_state", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@dataclass(frozen=True)
class _Row:
    bag_timestamp_ns: int
    payload: object = None


def test_adapter_joint_names_comes_from_source_assignment(tmp_path: Path) -> None:
    source = tmp_path / "contracts.py"
    source.write_text("JOINT_NAMES: tuple[str, ...] = ('slide_joint', " +
                      ", ".join(repr(f"joint_{index}") for index in range(16)) + ")\n")
    names = MODULE.adapter_joint_names(source)
    assert len(names) == 17
    assert names[0] == "slide_joint"
    assert names[-1] == "joint_15"


def test_canonical_positions_uses_names_not_wire_order() -> None:
    canonical = ("slide_joint", *(f"joint_{index}" for index in range(16)))
    wire_names = tuple(reversed(canonical))
    wire_positions = tuple(float(index) for index in range(17))
    timestamp_ns, positions = MODULE.canonical_positions(
        (123, wire_names, wire_positions), canonical
    )
    assert timestamp_ns == 123
    assert positions == tuple(reversed(wire_positions))


def test_latest_before_is_lower_inclusive() -> None:
    rows = [_Row(10), _Row(20), _Row(30)]
    assert MODULE.latest_before(rows, 20).bag_timestamp_ns == 20
    assert MODULE.latest_before(rows, 29).bag_timestamp_ns == 20


def test_stats_include_population_std_and_requested_percentiles() -> None:
    result = MODULE.stats([0.0, 10.0])
    assert result == {
        "count": 2,
        "mean": 5.0,
        "std": 5.0,
        "min": 0.0,
        "p05": 0.5,
        "p50": 5.0,
        "p95": 9.5,
        "max": 10.0,
    }
