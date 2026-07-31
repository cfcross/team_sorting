"""MMK2 Controller Manifest V1 and ActionMux range regressions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from team_sorting.action_mux import ActionMux
from team_sorting.controller_manifest import (
    MMK2_CONTROLLER_MANIFEST_V1,
    validate_controller_config,
    validate_controller_manifest,
)
from team_sorting.interfaces import (
    ACTION_NAMES,
    BaseCommand,
    FSMStatus,
    GlobalPhase,
    LocalPhase,
    ManipulationCommand,
    RobotJointState,
)


def _config() -> dict[str, object]:
    return yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config" / "config.yaml").read_text(
            encoding="utf-8"
        )
    )


def _joints(position: tuple[float, ...] = (0.0,) * 17) -> RobotJointState:
    return RobotJointState(
        position=position,
        velocity=(0.0,) * 17,
        effort=(0.0,) * 17,
        timestamp_ns=1_000,
    )


def _status() -> FSMStatus:
    return FSMStatus(
        task_id=1,
        global_phase=GlobalPhase.SEARCH_TARGET,
        local_phase=LocalPhase.IDLE,
        retry_count=0,
        success=False,
        failure_reason="",
        timestamp_ns=1_000,
    )


def test_manifest_is_immutable_19d_and_uses_canonical_order() -> None:
    manifest = MMK2_CONTROLLER_MANIFEST_V1
    validate_controller_manifest(manifest)

    assert manifest.schema_name == "MMK2ControllerManifest"
    assert manifest.schema_version == 1
    assert manifest.action_dim == len(manifest.actions) == 19
    assert manifest.runtime_source == "/tmp/material_competition_ros2_runtime.xml"
    assert "server_command_watchdog" in manifest.unverified_facts
    assert tuple(item.index for item in manifest.actions) == tuple(range(19))
    assert tuple(item.name for item in manifest.actions) == ACTION_NAMES
    with pytest.raises(AttributeError):
        manifest.action_dim = 8  # type: ignore[misc]


def test_units_semantics_topics_lengths_types_and_runtime_qos() -> None:
    manifest = MMK2_CONTROLLER_MANIFEST_V1
    assert tuple(item.semantic for item in manifest.actions) == (
        "velocity",
        "velocity",
        *("absolute_position",) * 9,
        "normalized_position",
        *("absolute_position",) * 6,
        "normalized_position",
    )
    assert tuple(item.unit for item in manifest.actions) == (
        "m/s",
        "rad/s",
        "m",
        *("rad",) * 8,
        "dimensionless",
        *("rad",) * 6,
        "dimensionless",
    )
    topics = {topic.group: topic for topic in manifest.official_topics}
    assert {group: topic.element_count for group, topic in topics.items()} == {
        "base": 2,
        "spine": 1,
        "head": 2,
        "left_arm": 7,
        "right_arm": 7,
    }
    assert topics["base"].message_type == "geometry_msgs/msg/Twist"
    assert all(
        topic.message_type == "std_msgs/msg/Float64MultiArray"
        for group, topic in topics.items()
        if group != "base"
    )
    assert all(
        (
            topic.qos.reliability,
            topic.qos.history,
            topic.qos.depth,
            topic.qos.durability,
        )
        == ("RELIABLE", "KEEP_LAST", 5, "VOLATILE")
        for topic in topics.values()
    )


def test_arm_safe_ranges_are_exact_intersections_symmetric_and_within_runtime() -> None:
    actions = MMK2_CONTROLLER_MANIFEST_V1.actions
    expected_safe = (
        (-3.14, 2.089),
        (-2.50, 0.181),
        (-0.094, 3.14),
        (-2.60, 2.60),
        (-1.859, 1.859),
        (-2.60, 2.60),
    )
    expected_runtime = (
        (-3.151, 2.089),
        (-2.963, 0.181),
        (-0.094, 3.161),
        (-3.012, 3.012),
        (-1.859, 1.859),
        (-3.017, 3.017),
    )
    for arm in (actions[5:11], actions[12:18]):
        assert tuple((item.safe_min, item.safe_max) for item in arm) == expected_safe
        assert tuple((item.runtime_min, item.runtime_max) for item in arm) == expected_runtime
        assert all(
            item.runtime_min <= item.safe_min <= item.safe_max <= item.runtime_max
            for item in arm
        )


def test_base_uses_twist_limits_not_internal_wheel_motor_range() -> None:
    base_v, base_w = MMK2_CONTROLLER_MANIFEST_V1.actions[:2]
    assert (base_v.safe_min, base_v.safe_max) == (-0.25, 0.25)
    assert (base_w.safe_min, base_w.safe_max) == (-0.50, 0.50)
    assert base_v.runtime_min is base_v.runtime_max is None
    assert base_w.runtime_min is base_w.runtime_max is None
    assert base_v.runtime_actuator_name is base_w.runtime_actuator_name is None
    assert -35.0 not in (base_v.safe_min, base_w.safe_min)
    assert 35.0 not in (base_v.safe_max, base_w.safe_max)


def test_loaded_config_exactly_matches_manifest() -> None:
    config = _config()
    validate_controller_config(config)
    action_mux = config["action_mux"]
    actions = MMK2_CONTROLLER_MANIFEST_V1.actions
    assert tuple(action_mux["joint_lower"]) == tuple(item.safe_min for item in actions[2:])
    assert tuple(action_mux["joint_upper"]) == tuple(item.safe_max for item in actions[2:])


@pytest.mark.parametrize("bad_value", [True, float("nan"), float("inf")])
def test_config_validation_rejects_bool_and_non_finite_numbers(bad_value: object) -> None:
    config = _config()
    config["action_mux"]["joint_lower"][3] = bad_value
    with pytest.raises(ValueError, match="有限实数"):
        validate_controller_config(config)


def test_manifest_validation_rejects_non_19d_and_out_of_runtime_range() -> None:
    manifest = MMK2_CONTROLLER_MANIFEST_V1
    with pytest.raises(ValueError, match="action_dim必须为19"):
        validate_controller_manifest(replace(manifest, action_dim=8))
    unsafe = replace(manifest.actions[5], safe_max=2.10)
    unsafe_actions = list(manifest.actions)
    unsafe_actions[5] = unsafe
    with pytest.raises(ValueError, match="超过运行时范围"):
        validate_controller_manifest(
            replace(manifest, actions=tuple(unsafe_actions))
        )


def test_manifest_validation_rejects_bool_index_and_wrong_qos() -> None:
    manifest = MMK2_CONTROLLER_MANIFEST_V1
    bad_index_actions = list(manifest.actions)
    bad_index_actions[1] = replace(bad_index_actions[1], index=True)
    with pytest.raises(ValueError, match="index必须连续"):
        validate_controller_manifest(
            replace(manifest, actions=tuple(bad_index_actions))
        )

    bad_qos = replace(manifest.official_topics[0].qos, depth=10)
    bad_topics = list(manifest.official_topics)
    bad_topics[0] = replace(bad_topics[0], qos=bad_qos)
    with pytest.raises(ValueError, match="QoS"):
        validate_controller_manifest(
            replace(manifest, official_topics=tuple(bad_topics))
        )


@pytest.mark.parametrize("action_index", [5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17])
def test_action_mux_clips_each_active_arm_axis_to_new_safe_intersection(
    action_index: int,
) -> None:
    mux = ActionMux()
    joint_index = action_index - 2
    target = [0.0] * 17
    target[joint_index] = mux.config.joint_upper[joint_index] + 1.0
    mask = tuple(index == joint_index for index in range(17))
    command = ManipulationCommand(
        joint_target=tuple(target),
        controlled_mask=mask,
        local_phase=LocalPhase.IDLE,
        timestamp_ns=1_000,
        valid_until_ns=2_000,
    )

    action = mux.compose(
        BaseCommand(0.0, 0.0, 1_000, 2_000),
        command,
        _joints(),
        _status(),
        1_500,
    )

    assert action.values[action_index] == mux.config.joint_upper[joint_index]
    assert action.clipped


def test_uncontrolled_out_of_range_feedback_is_preserved_not_active_hold() -> None:
    mux = ActionMux()
    position = [0.0] * 17
    position[3] = 2.50  # left joint1 exceeds the new safe upper bound 2.089

    action = mux.compose(
        BaseCommand(0.0, 0.0, 1_000, 2_000),
        None,
        _joints(tuple(position)),
        _status(),
        1_500,
    )

    assert action.values[5] == 2.50
    assert not action.valid
    assert not action.clipped
    assert "无机械臂候选命令" in action.failure_reason
    assert "保持实际反馈值" in action.failure_reason


def test_manifest_module_has_no_native_eight_dimensional_mapping() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "team_sorting"
        / "controller_manifest.py"
    ).read_text(encoding="utf-8")
    assert "mmk2_pi05" not in source
    assert "pi05_droid" not in source
    assert "8→19" not in source
    assert "(0.0,) * 19" not in source
