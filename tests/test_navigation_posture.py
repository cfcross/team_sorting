"""3.5-D1 INITIAL_ZERO_POSE私有判定合同。"""

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import team_sorting.navigation_posture as posture_module
from team_sorting.interfaces import JOINT_NAMES, RobotJointState
from team_sorting.navigation_posture import (
    _NavigationPostureConfig,
    _NavigationPostureState,
    _NavigationPostureTracker,
)
from team_sorting.ros_nodes import _navigation_posture_config_from_config


NOW = 10_000_000_000


def _config(**overrides: object) -> _NavigationPostureConfig:
    values: dict[str, object] = {
        "mode": "initial_zero_pose",
        "target": (0.0,) * 17,
        "joint_state_max_age_ns": 150_000_000,
        "max_feedback_gap_ns": 200_000_000,
        "settled_required_cycles": 3,
        "slide_tolerance_m": 0.005,
        "head_tolerance_rad": 0.02,
        "arm_tolerance_rad": 0.02,
        "gripper_tolerance": 0.01,
        "slide_velocity_tolerance_mps": 0.01,
        "angular_velocity_tolerance_radps": 0.02,
        "gripper_velocity_tolerance_per_s": 0.01,
    }
    values.update(overrides)
    return _NavigationPostureConfig(**values)  # type: ignore[arg-type]


def _joints(
    timestamp_ns: int,
    *,
    position: tuple[float, ...] = (0.0,) * 17,
    velocity: tuple[float, ...] = (0.0,) * 17,
) -> RobotJointState:
    return RobotJointState(
        position=position,
        velocity=velocity,
        effort=(0.0,) * 17,
        timestamp_ns=timestamp_ns,
    )


def _yaml_config() -> dict[str, object]:
    path = Path(__file__).parents[1] / "config/config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_navigation_posture_yaml_maps_exact_approved_contract() -> None:
    parsed = _navigation_posture_config_from_config(_yaml_config())
    assert parsed == _config(target=[0.0] * 17)
    assert parsed.target == (0.0,) * 17
    assert all(type(value) is float for value in parsed.target)


@pytest.mark.parametrize("mode", ["zero", "INITIAL_ZERO_POSE", " initial_zero_pose"])
def test_navigation_posture_config_rejects_wrong_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        _config(mode=mode)


@pytest.mark.parametrize("target", [(0.0,) * 16, (0.0,) * 18, None, "0" * 17])
def test_navigation_posture_config_rejects_wrong_target_width(
    target: object,
) -> None:
    with pytest.raises(ValueError, match="17"):
        _config(target=target)


@pytest.mark.parametrize(
    "target",
    [
        (1.0,) + (0.0,) * 16,
        (True,) + (0.0,) * 16,
        (math.nan,) + (0.0,) * 16,
        (math.inf,) + (0.0,) * 16,
        (np.float64(0.0),) + (0.0,) * 16,
        ("0",) + (0.0,) * 16,
    ],
)
def test_navigation_posture_config_rejects_nonzero_or_invalid_target(
    target: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="target|0.0"):
        _config(target=target)


def test_navigation_posture_config_guards_joint_name_order(monkeypatch) -> None:
    monkeypatch.setattr(posture_module, "JOINT_NAMES", tuple(reversed(JOINT_NAMES)))
    with pytest.raises(ValueError, match="JOINT_NAMES"):
        _config()


def test_navigation_posture_mapping_rejects_missing_unknown_and_wrong_nesting() -> None:
    source = _yaml_config()
    missing = {**source, "navigation_posture": dict(source["navigation_posture"])}
    missing["navigation_posture"].pop("head_tolerance_rad")
    with pytest.raises(ValueError, match="head_tolerance_rad"):
        _navigation_posture_config_from_config(missing)
    unknown = {**source, "navigation_posture": dict(source["navigation_posture"])}
    unknown["navigation_posture"]["hidden_default"] = 1
    with pytest.raises(ValueError, match="hidden_default"):
        _navigation_posture_config_from_config(unknown)
    with pytest.raises(ValueError, match="Mapping"):
        _navigation_posture_config_from_config({"navigation_posture": []})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("joint_state_max_age_ns", True),
        ("joint_state_max_age_ns", 0),
        ("max_feedback_gap_ns", -1),
        ("settled_required_cycles", True),
        ("settled_required_cycles", 0),
        ("settled_required_cycles", -1),
    ],
)
def test_navigation_posture_config_rejects_invalid_integer_fields(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slide_tolerance_m", True),
        ("head_tolerance_rad", 0.0),
        ("arm_tolerance_rad", -0.1),
        ("gripper_tolerance", math.nan),
        ("slide_velocity_tolerance_mps", math.inf),
        ("angular_velocity_tolerance_radps", np.float64(0.02)),
        ("gripper_velocity_tolerance_per_s", "0.01"),
    ],
)
def test_navigation_posture_config_rejects_invalid_tolerances(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


def test_navigation_posture_requires_three_real_feedback_frames() -> None:
    tracker = _NavigationPostureTracker(_config())
    assert tracker.observe(_joints(NOW), NOW) is _NavigationPostureState.SETTLING
    assert tracker.settled_cycles == 1 and not tracker.active
    assert tracker.observe(_joints(NOW + 1), NOW + 1) is _NavigationPostureState.SETTLING
    assert tracker.settled_cycles == 2 and not tracker.active
    assert tracker.observe(_joints(NOW + 2), NOW + 2) is _NavigationPostureState.ACTIVE
    assert tracker.settled_cycles == 3 and tracker.active
    tracker.observe(_joints(NOW + 3), NOW + 3)
    assert tracker.settled_cycles == 3


def test_navigation_posture_duplicate_feedback_is_idempotent_but_mutation_invalid() -> None:
    tracker = _NavigationPostureTracker(_config())
    first = _joints(NOW)
    tracker.observe(first, NOW)
    assert tracker.observe(first, NOW + 1) is _NavigationPostureState.SETTLING
    assert tracker.settled_cycles == 1
    mutated = _joints(NOW, velocity=(0.001,) + (0.0,) * 16)
    assert tracker.observe(mutated, NOW + 1) is _NavigationPostureState.INVALID
    assert tracker.settled_cycles == 0
    assert "相同timestamp" in tracker.failure_reason


def test_navigation_posture_gap_boundary_and_reset_policy() -> None:
    tracker = _NavigationPostureTracker(_config())
    tracker.observe(_joints(NOW), NOW)
    tracker.observe(_joints(NOW + 200_000_000), NOW + 200_000_000)
    assert tracker.settled_cycles == 2
    tracker.observe(_joints(NOW + 400_000_001), NOW + 400_000_001)
    assert tracker.state is _NavigationPostureState.SETTLING
    assert tracker.settled_cycles == 1


@pytest.mark.parametrize(
    ("timestamp_ns", "now_ns", "reason"),
    [
        (NOW + 1, NOW, "未来"),
        (NOW - 150_000_001, NOW, "max_age"),
    ],
)
def test_navigation_posture_rejects_future_and_stale_feedback(
    timestamp_ns: int, now_ns: int, reason: str
) -> None:
    tracker = _NavigationPostureTracker(_config())
    assert tracker.observe(_joints(timestamp_ns), now_ns) is _NavigationPostureState.INVALID
    assert reason in tracker.failure_reason


def test_navigation_posture_accepts_exact_age_boundary() -> None:
    tracker = _NavigationPostureTracker(_config())
    assert tracker.observe(_joints(NOW - 150_000_000), NOW) is _NavigationPostureState.SETTLING


def test_navigation_posture_timestamp_rollback_invalidates_and_latches() -> None:
    tracker = _NavigationPostureTracker(_config())
    tracker.observe(_joints(NOW), NOW)
    assert tracker.observe(_joints(NOW - 1), NOW) is _NavigationPostureState.INVALID
    assert tracker.observe(_joints(NOW + 1), NOW + 1) is _NavigationPostureState.INVALID


@pytest.mark.parametrize(
    ("index", "position_limit", "velocity_limit"),
    [
        (0, 0.005, 0.01),
        (1, 0.02, 0.02),
        (3, 0.02, 0.02),
        (9, 0.01, 0.01),
        (10, 0.02, 0.02),
        (16, 0.01, 0.01),
    ],
)
def test_navigation_posture_group_boundaries_are_closed_and_one_ulp_strict(
    index: int, position_limit: float, velocity_limit: float
) -> None:
    position = [0.0] * 17
    velocity = [0.0] * 17
    position[index] = position_limit
    velocity[index] = velocity_limit
    tracker = _NavigationPostureTracker(_config())
    assert tracker.observe(_joints(NOW, position=tuple(position), velocity=tuple(velocity)), NOW) is _NavigationPostureState.SETTLING

    position[index] = math.nextafter(position_limit, math.inf)
    tracker = _NavigationPostureTracker(_config())
    assert tracker.observe(_joints(NOW, position=tuple(position)), NOW) is _NavigationPostureState.PREPARING
    assert JOINT_NAMES[index] in tracker.failure_reason

    velocity[index] = math.nextafter(velocity_limit, math.inf)
    position[index] = 0.0
    tracker = _NavigationPostureTracker(_config())
    assert tracker.observe(_joints(NOW, velocity=tuple(velocity)), NOW) is _NavigationPostureState.PREPARING


def _activate(tracker: _NavigationPostureTracker) -> None:
    for offset in range(3):
        tracker.observe(_joints(NOW + offset), NOW + offset)
    assert tracker.active


@pytest.mark.parametrize("kind", ["position", "velocity"])
def test_navigation_posture_active_deviation_is_latched(kind: str) -> None:
    tracker = _NavigationPostureTracker(_config())
    _activate(tracker)
    position = (0.006,) + (0.0,) * 16 if kind == "position" else (0.0,) * 17
    velocity = (0.011,) + (0.0,) * 16 if kind == "velocity" else (0.0,) * 17
    assert tracker.observe(_joints(NOW + 3, position=position, velocity=velocity), NOW + 3) is _NavigationPostureState.DEVIATED
    assert not tracker.active and tracker.settled_cycles == 0 and tracker.failure_reason
    assert tracker.observe(_joints(NOW + 4), NOW + 4) is _NavigationPostureState.DEVIATED


def test_navigation_posture_invalid_feedback_is_latched() -> None:
    tracker = _NavigationPostureTracker(_config())
    invalid = replace(_joints(NOW), valid=False, failure_reason="sensor invalid")
    assert tracker.observe(invalid, NOW) is _NavigationPostureState.INVALID
    assert tracker.observe(_joints(NOW + 1), NOW + 1) is _NavigationPostureState.INVALID


@pytest.mark.parametrize("field", ["position", "velocity"])
@pytest.mark.parametrize("damage", [(0.0,) * 16, (math.nan,) + (0.0,) * 16, (math.inf,) + (0.0,) * 16])
def test_navigation_posture_damaged_vectors_fail_closed(
    field: str, damage: tuple[float, ...]
) -> None:
    actual = _joints(NOW)
    object.__setattr__(actual, field, damage)
    tracker = _NavigationPostureTracker(_config())
    assert tracker.observe(actual, NOW) is _NavigationPostureState.INVALID


@pytest.mark.parametrize("initial_state", ["settling", "active", "deviated", "invalid"])
def test_navigation_posture_reset_clears_all_evidence(initial_state: str) -> None:
    tracker = _NavigationPostureTracker(_config())
    if initial_state in {"active", "deviated"}:
        _activate(tracker)
    else:
        tracker.observe(_joints(NOW), NOW)
    if initial_state == "deviated":
        tracker.observe(_joints(NOW + 3, position=(0.006,) + (0.0,) * 16), NOW + 3)
    elif initial_state == "invalid":
        tracker.observe(_joints(NOW - 1), NOW)
    tracker.reset()
    assert tracker.state is _NavigationPostureState.IDLE
    assert tracker.settled_cycles == 0
    assert tracker.last_feedback_timestamp_ns is None
    assert tracker.failure_reason == ""
    assert tracker.observe(_joints(NOW + 10), NOW + 10) is _NavigationPostureState.SETTLING
    assert tracker.settled_cycles == 1


@pytest.mark.parametrize("now_ns", [True, -1, 1.0, "1", None])
def test_navigation_posture_rejects_invalid_observation_time(now_ns: object) -> None:
    with pytest.raises(ValueError, match="now_ns"):
        _NavigationPostureTracker(_config()).observe(_joints(NOW), now_ns)  # type: ignore[arg-type]
