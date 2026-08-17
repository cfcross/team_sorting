"""4.4--4.6只规划标定工具的失败关闭回归。"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from team_sorting.interfaces import IKResult, JOINT_NAMES


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "arm_pick_place_calibration", ROOT / "scripts/arm_pick_place_calibration.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class FakeIK:
    def __init__(self, *, success: bool = True, arm_value: float = 0.0,
                 slide_limits: tuple[float, float] | None = (-0.04, 0.87)) -> None:
        self.success = success
        self.arm_value = arm_value
        self._limits = slide_limits

    def slide_limits(self):
        return self._limits

    def solve_ik(self, *, target_slide, **_kwargs):
        if not self.success:
            return IKResult(target_slide, None, None, False, "fake IK无解")
        arm = (self.arm_value,) * 6
        return IKResult(target_slide, arm, arm, True)


@pytest.fixture()
def config():
    return yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))


def _trial(**updates):
    values = {
        "min_object_confidence": 0.5,
        "transform_max_age_ns": 1000,
        "object_estimate_max_age_ns": 1000,
        "joint_state_max_age_ns": 1000,
        "planned_context_max_age_ns": 1000,
        "confirmed_context_max_age_ns": 1000,
        "pregrasp_distance_m": 0.1,
        "grasp_contact_offset_m": 0.01,
        "lift_distance_m": 0.05,
        "retreat_distance_m": 0.1,
        "preplace_height_m": 0.1,
        "release_offset_m": 0.01,
        "post_release_retreat_distance_m": 0.1,
        "settle_time_s": 0.1,
        "max_slide_waypoint_delta_m": 0.2,
        "max_arm_waypoint_delta_rad": 1.0,
        "max_gripper_waypoint_delta": 1.0,
        "pregrasp_duration_s": 1.0,
        "grasp_duration_s": 1.0,
        "lift_duration_s": 1.0,
        "retreat_duration_s": 1.0,
        "preplace_duration_s": 1.0,
        "lower_duration_s": 1.0,
        "release_duration_s": 1.0,
        "post_release_retreat_duration_s": 1.0,
    }
    values.update(updates)
    return values


def _task():
    return {
        "task_id": 1, "instruction": "move pink", "target_kind": "box",
        "target_body": "pink_box", "target_color": "pink",
        "place_type": "table_point", "place_world_xyz": [1.0, 0.0, 0.8],
        "place_frame_id": "world", "place_radius": 0.1, "timestamp_ns": 900,
    }


def _joints(timestamp_ns=900):
    return {
        "position": [0.1, 0.0, 0.0, *([0.0] * 6), 1.0, *([0.0] * 6), 1.0],
        "velocity": [0.0] * 17, "effort": [0.0] * 17,
        "joint_names": list(JOINT_NAMES), "timestamp_ns": timestamp_ns,
    }


def _transform(source, target):
    return {
        "source_frame": source, "target_frame": target,
        "translation_xyz": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "timestamp_ns": 900, "valid": True,
    }


def _pick_payload():
    return {
        "source": "stage2_calibration_fixture", "seed": 7,
        "scene": "table_side_left", "source_slot": "table_side",
        "now_ns": 1000, "expected_object_id": "track-1", "task": _task(),
        "object_estimate": {
            "class_id": "pink", "position_xyz": [-1.0, 2.2, 0.834],
            "confidence": 0.9, "frame_id": "odom", "timestamp_ns": 900,
            "object_id": "track-1", "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "size_xyz_m": [0.24, 0.16, 0.19],
        },
        "joint_state": _joints(), "trial_parameters": _trial(),
        "transforms": {
            "target_to_footprint": _transform("odom", "footprint"),
            "target_to_world": _transform("odom", "world"),
        },
    }


def _right_pick_payload():
    payload = _pick_payload()
    payload["seed"] = 20260709
    payload["scene"] = "table_side_right"
    payload["object_estimate"]["position_xyz"] = [-0.18, 2.20, 0.834]
    return payload


def _base_state(x=-0.7, y=0.55005, yaw=math.pi / 2, timestamp_ns=1000,
                linear=0.0, angular=0.0):
    return tool.BaseState(
        (x, y, 0.00157), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)), yaw,
        (linear, 0.0, 0.0), (0.0, 0.0, angular), "odom", timestamp_ns,
    )


def test_plan_pick_is_plan_only_and_reports_required_fields(config):
    result = tool.plan_pick(_pick_payload(), config, FakeIK())
    assert result["valid"] is True
    assert result["published_control"] is False
    assert set(result["poses"]) == {"pregrasp", "grasp/contact", "lift", "retreat"}
    assert len(result["waypoints"]) == 5
    assert result["joint_limits_ok"] is True


def test_all_six_stage2_fixtures_are_derived_from_current_config(config):
    expected_positions = [
        center
        for slot in ("table_side", "table_top", "shelf")
        for center in config["source_slots"]["slots"][slot]["centers"]
    ]
    fixtures = [tool.stage2_fixture(config, scene, "pink") for scene in tool.SCENE_FIXTURE_KEYS]
    assert [item["position_xyz"] for item in fixtures] == expected_positions
    assert all(item["source"] == "stage2_calibration_fixture" for item in fixtures)
    assert all(item["size_xyz_m"] == [0.24, 0.16, 0.19] for item in fixtures)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["object_estimate"].update(orientation_xyzw=None), "orientation"),
        (lambda p: p.update(expected_object_id="other"), "object_id"),
        (lambda p: p["object_estimate"].update(size_xyz_m=[0.0, 0.1, 0.1]), "大于 0"),
        (lambda p: p["object_estimate"].update(position_xyz=[float("nan"), 0.0, 1.0]), "有限"),
    ],
)
def test_plan_pick_rejects_missing_identity_and_invalid_geometry(config, mutation, message):
    payload = _pick_payload()
    mutation(payload)
    result = tool.plan_pick(payload, config, FakeIK())
    assert result["valid"] is False and result["published_control"] is False
    assert message in result["failure_reason"]


def test_plan_pick_reports_ik_failure(config):
    result = tool.plan_pick(_pick_payload(), config, FakeIK(success=False))
    assert result["ik_success"] is False
    assert "IK" in result["failure_reason"]
    assert result["status"] == "IK_NO_COMPLETE_PATH"
    assert result["ik_diagnostics"]["available"] is True
    assert result["status"] != "BASE_STAND_POSITION_REQUIRED"


def test_plan_pick_rejects_slide_outside_official_limit(config):
    payload = _pick_payload()
    payload["joint_state"]["position"][0] = 0.9
    result = tool.plan_pick(payload, config, FakeIK(slide_limits=(-0.04, 0.87)))
    assert result["valid"] is False
    assert "slide" in result["failure_reason"]


def test_plan_pick_rejects_joint_limit_even_if_fake_ik_returns_it(config):
    payload = _pick_payload()
    payload["trial_parameters"]["max_arm_waypoint_delta_rad"] = 20.0
    result = tool.plan_pick(payload, config, FakeIK(arm_value=10.0))
    assert result["valid"] is False
    assert "安全关节边界" in result["failure_reason"]


def test_plan_pick_rejects_discontinuous_waypoint(config):
    payload = _pick_payload()
    payload["trial_parameters"]["max_arm_waypoint_delta_rad"] = 0.01
    result = tool.plan_pick(payload, config, FakeIK(arm_value=0.2))
    assert result["valid"] is False
    assert "连续性" in result["failure_reason"] or "变化" in result["failure_reason"]


def test_plan_pick_rejects_stale_and_wrong_joint_identity(config):
    stale = _pick_payload()
    stale["joint_state"]["timestamp_ns"] = 0
    stale["trial_parameters"]["joint_state_max_age_ns"] = 10
    assert tool.plan_pick(stale, config, FakeIK())["valid"] is False
    wrong = _pick_payload()
    wrong["joint_state"]["joint_names"][0] = "wrong"
    result = tool.plan_pick(wrong, config, FakeIK())
    assert result["valid"] is False and "joint_names" in result["failure_reason"]


def test_plan_place_is_plan_only_and_reports_three_pose_groups(config):
    pick = _pick_payload()
    planning = tool.planning_config(config, pick["trial_parameters"])
    planner = tool.ArmPlanner(FakeIK(), planning)
    grasp_target, _ = planner.plan_grasp(
        tool._task(pick["task"]), tool._estimate(pick["object_estimate"]),
        tool._transform(pick["transforms"]["target_to_footprint"]),
        tool._transform(pick["transforms"]["target_to_world"]),
        tool._joints(pick["joint_state"]), pick["now_ns"],
    )
    assert grasp_target.grasp_context is not None
    confirmed = replace(
        grasp_target.grasp_context, confirmed=True, confirmed_at_ns=1000
    )
    place_task = _task()
    place_task["timestamp_ns"] = 1000
    payload = {
        "source": "stage2_calibration_fixture", "seed": 7,
        "scene": "table_side_left", "now_ns": 1100,
        "load_state": "carrying_object",
        "task": place_task, "joint_state": _joints(1050),
        "trial_parameters": _trial(), "grasp_context": tool._jsonable(confirmed),
        "transforms": {"world_to_footprint": {
            **_transform("world", "footprint"), "timestamp_ns": 1050,
        }},
    }
    result = tool.plan_place(payload, config, FakeIK())
    assert result["valid"] is True and result["published_control"] is False
    assert {"preplace", "release", "post_release_retreat"} <= set(result["poses"])
    assert len(result["waypoints"]) == 4


def test_plan_place_supports_explicit_empty_load_without_claiming_grasp(config):
    pick = _pick_plan(config)
    context = pick["planned_grasp_context"]
    task = _task()
    task["timestamp_ns"] = 1000
    payload = {
        "source": "stage2_calibration_fixture", "seed": 7,
        "scene": "table_side_left", "load_state": "empty", "now_ns": 1100,
        "task": task, "joint_state": _joints(1050), "trial_parameters": _trial(),
        "grasp_context": context,
        "transforms": {"world_to_footprint": {
            **_transform("world", "footprint"), "timestamp_ns": 1050,
        }},
    }
    result = tool.plan_place(payload, config, FakeIK())
    assert result["valid"] is True
    assert result["load_state"] == "empty"
    assert result["calibration_context_override"] == "EMPTY_LOAD_KINEMATIC_ONLY_NOT_GRASP_CONFIRMATION"


def test_place_lower_keeps_closed_release_opens_and_unconfirmed_carry_is_blocked(config):
    pick = _pick_plan(config)
    context = pick["planned_grasp_context"]
    task = _task()
    task["timestamp_ns"] = 1000
    payload = {
        "source": "stage2_calibration_fixture", "seed": 7,
        "scene": "table_side_left", "load_state": "empty", "now_ns": 1100,
        "task": task, "joint_state": _joints(1050), "trial_parameters": _trial(),
        "grasp_context": context,
        "transforms": {"world_to_footprint": {
            **_transform("world", "footprint"), "timestamp_ns": 1050,
        }},
    }
    plan = tool.plan_place(payload, config, FakeIK())
    assert plan["valid"] is True
    lower = tool._stage_target(plan, "lower")
    release = tool._stage_target(plan, "release")
    assert lower[9] == lower[16] == 0.1
    assert release[9] == release[16] == 1.0
    assert lower[9] != 0.0 and lower[16] != 0.0
    payload["load_state"] = "carrying_object"
    blocked = tool.plan_place(payload, config, FakeIK())
    assert blocked["valid"] is False
    assert "confirmed GraspContext" in blocked["failure_reason"]


def test_pick_and_place_later_stages_require_saved_previous_endpoint(
    config, monkeypatch
):
    pick = _pick_plan(config)
    result = _execute(
        pick, FakeRuntime(config, pick["start_joint_state"]), monkeypatch,
        stage="short-lift",
    )
    assert result["failure_classification"] == "START_MISMATCH"
    assert "禁止跳段" in result["failure_reason"]

    context = pick["planned_grasp_context"]
    task = _task()
    task["timestamp_ns"] = 1000
    payload = {
        "source": "stage2_calibration_fixture", "seed": 7,
        "scene": "table_side_left", "load_state": "empty", "now_ns": 1100,
        "task": task, "joint_state": _joints(1050), "trial_parameters": _trial(),
        "grasp_context": context,
        "transforms": {"world_to_footprint": {
            **_transform("world", "footprint"), "timestamp_ns": 1050,
        }},
    }
    place = tool.plan_place(payload, config, FakeIK())
    result = _execute(
        place, FakeRuntime(config, place["start_joint_state"]), monkeypatch,
        stage="release",
    )
    assert result["failure_classification"] == "START_MISMATCH"
    assert "禁止跳段" in result["failure_reason"]


def _execution_parameters(**updates):
    values = {
        "max_slide_velocity_m_s": 0.15, "max_arm_velocity_rad_s": 0.6,
        "max_gripper_velocity_per_s": 0.6, "control_rate_hz": 10.0,
        "timeout_s": 4.0, "feedback_max_age_s": 0.5,
        "slide_tolerance_m": 0.01, "arm_tolerance_rad": 0.01,
        "gripper_tolerance": 0.02, "settle_cycles": 2,
    }
    values.update(updates)
    return values


class FakeRuntime:
    def __init__(self, config, start, *, conflict=False, fresh=True,
                 interrupt=False, frozen=False, probe_pass=True, fail_poll=False,
                 feedback_interval_s=0.0, no_feedback=False,
                 validation_error=None, dds_residual_cycles=0,
                 instruction_raw=None, instruction_after_spins=0):
        self.config = config
        self.state = tool._joints(start)
        self.conflict = conflict
        self.fresh = fresh
        self.interrupt = interrupt
        self.frozen = frozen
        self.probe_pass = probe_pass
        self.fail_poll = fail_poll
        self.feedback_interval_s = feedback_interval_s
        self.no_feedback = no_feedback
        self.latest_joint_validation_error = validation_error
        self.publishers = {}
        self.commands = []
        self.now = 0.0
        self.joint_age_ns = 0 if fresh else 1_000_000_000
        self.joint_received_count = 1
        self.joint_received_ns = self.state.timestamp_ns
        self.feedback_accumulator_s = 0.0
        self.dds_residual_cycles = dds_residual_cycles
        self.publisher_start_count = 0
        self.instruction_raw = (
            instruction_raw if instruction_raw is not None else json.dumps(_task())
        )
        self.instruction_after_spins = instruction_after_spins
        self.instruction_spin_count = 0
        self.instruction_receive_call_count = 0
        self.latest_instruction_raw = None
        self.latest_instruction_received_ns = None
        self.instruction_qos_evidence = {
            "history": "KEEP_LAST", "depth": 10,
            "reliability": "RELIABLE", "durability": "VOLATILE",
            "compatibility_basis": "fake compatible endpoint",
        }

    def subscriber_counts(self):
        return {group: 1 for group in tool.ARM_TOPIC_GROUPS}

    def probe(self, **_kwargs):
        return {
            "official_offline_conditions_met": self.probe_pass,
            "blockers": [] if self.probe_pass else ["TEST_PROBE_BLOCKER"],
        }

    def other_publishers(self):
        if self.conflict:
            return {"spine": 1, "head": 0, "left_arm": 0, "right_arm": 0}
        if self.publishers:
            return {"spine": 1, "head": 0, "left_arm": 1, "right_arm": 1}
        return {group: 0 for group in tool.ARM_TOPIC_GROUPS}

    def wait_for_inputs(self, _timeout):
        return self.state

    def wait_for_instruction(self, timeout_s):
        deadline = self.now + timeout_s
        while self.now < deadline and self.latest_instruction_raw is None:
            self.instruction_spin_count += 1
            self.now += min(0.05, max(0.0, deadline - self.now))
            self.joint_received_count += 1
            self.state = replace(
                self.state, timestamp_ns=self.state.timestamp_ns + 50_000_000
            )
            if self.instruction_spin_count > self.instruction_after_spins:
                self.latest_instruction_raw = self.instruction_raw
                self.latest_instruction_received_ns = max(
                    1, int(self.now * 1_000_000_000)
                )
        if self.latest_instruction_raw is None:
            raise RuntimeError("等待/material/instruction完整JSON超时")
        return self.latest_instruction_raw, self.latest_instruction_received_ns

    def receive_instruction_tasks(self, timeout_s):
        self.instruction_receive_call_count += 1
        raw, received_ns = self.wait_for_instruction(timeout_s)
        return raw, received_ns, tuple(tool.InstructionParser().parse(raw, received_ns))

    def start_publishers(self):
        self.publisher_start_count += 1
        self.publishers = {"spine": object(), "left_arm": object(), "right_arm": object()}

    def wait_for_publisher_exclusivity(self, phase, started, _timeout_s):
        residual_samples = self.dds_residual_cycles
        self.dds_residual_cycles = 0
        external = self.conflict
        return {
            "phase": phase, "valid": not external,
            "converged": not external, "sample_count": residual_samples + 1,
            "groups": {
                group: {
                    "self_count": 1 if started and group != "head" else 0,
                    "external_count": 1 if external else 0,
                    "endpoints": ([{
                        "node_name": "old_or_external", "gid": "fake-gid",
                        "classification": "EXTERNAL_OR_DDS_RESIDUAL",
                    }] if external else []),
                } for group in tool.ARM_TOPIC_GROUPS
            },
        }

    def monotonic(self):
        self.now += 0.05
        return self.now

    def spin_control_period(self, period_s):
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt
        if self.fail_poll:
            self.fail_poll = False
            raise RuntimeError("fake JointState exception")
        self.joint_age_ns += int(period_s * 1_000_000_000)
        self.feedback_accumulator_s += period_s
        emit = not self.no_feedback and (
            self.feedback_interval_s <= 0.0
            or self.feedback_accumulator_s + 1e-12 >= self.feedback_interval_s
        )
        if emit:
            if self.feedback_interval_s > 0.0:
                self.feedback_accumulator_s -= self.feedback_interval_s
            position = self.state.position if self.frozen or not self.commands else self.commands[-1]
            self.state = replace(
                self.state, position=tuple(position),
                timestamp_ns=self.state.timestamp_ns + 50_000_000,
            )
            self.joint_age_ns = 0
            self.joint_received_count += 1
            self.joint_received_ns += 50_000_000

    def latest_joint_state(self):
        return self.state

    def latest_joint_age_ns(self):
        return self.joint_age_ns

    def latest_joint_receipt_ns(self):
        return self.joint_received_ns

    def poll_joint_state(self, timeout):
        self.spin_control_period(timeout)
        return self.state

    def publish_joint_target(self, position):
        self.commands.append(tuple(position))


def _pick_plan(config):
    result = tool.plan_pick(_pick_payload(), config, FakeIK())
    assert result["valid"] is True
    return result


def _execute(plan, runtime, monkeypatch, *, stage="pregrasp", simulation=True,
             confirm=tool.OFFICIAL_SIM_CONFIRMATION, parameters=None,
             expected_seed=None, expected_scene=None):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    return tool.execute_one_stage(
        plan, stage, simulation, confirm, parameters or _execution_parameters(),
        "用户观察", runtime, expected_seed=expected_seed,
        expected_scene=expected_scene,
    )


def test_execute_requires_simulation_single_stage_and_confirmation(config, monkeypatch):
    plan = _pick_plan(config)
    runtime = FakeRuntime(config, _joints())
    assert "official-offline-simulation" in _execute(
        plan, runtime, monkeypatch, simulation=False
    )["failure_reason"]
    assert "一次" in _execute(
        plan, runtime, monkeypatch, stage=["pregrasp", "retreat"]
    )["failure_reason"]
    assert "confirm" in _execute(
        plan, runtime, monkeypatch, confirm="wrong"
    )["failure_reason"]
    assert runtime.commands == []


def test_execute_rejects_failed_plan_other_publisher_and_stale_feedback(config, monkeypatch):
    plan = _pick_plan(config)
    failed = dict(plan, valid=False, failure_reason="IK failed")
    runtime = FakeRuntime(config, _joints())
    assert _execute(failed, runtime, monkeypatch)["published_control"] is False
    conflict = FakeRuntime(config, _joints(), conflict=True)
    assert "外部Publisher" in _execute(plan, conflict, monkeypatch)["failure_reason"]
    stale = FakeRuntime(config, _joints(), fresh=False)
    assert "过期" in _execute(plan, stale, monkeypatch)["failure_reason"]


def test_execute_rejects_nonfinite_and_out_of_limit_target(config, monkeypatch):
    plan = _pick_plan(config)
    damaged = json.loads(json.dumps(plan))
    damaged["waypoints"][0]["joint_position"][3] = float("nan")
    assert "有限" in _execute(
        damaged, FakeRuntime(config, _joints()), monkeypatch
    )["failure_reason"]
    damaged = json.loads(json.dumps(plan))
    damaged["waypoints"][0]["joint_position"][3] = 3.0
    assert "限位" in _execute(
        damaged, FakeRuntime(config, _joints()), monkeypatch
    )["failure_reason"]


def test_interpolation_obeys_three_explicit_velocity_limits():
    start = (0.0,) * 17
    target = (1.0,) * 17
    result = tool._interpolated_target(start, target, 0.5, _execution_parameters())
    assert result[0] == pytest.approx(0.075)
    assert result[3] == pytest.approx(0.3)
    assert result[9] == pytest.approx(0.3)
    assert result[1:3] == (0.0, 0.0)


def test_execute_timeout_and_ctrl_c_publish_latest_feedback_hold(config, monkeypatch):
    plan = _pick_plan(config)
    timeout = FakeRuntime(config, _joints(), frozen=True)
    result = _execute(
        plan, timeout, monkeypatch, stage="grasp-close",
        parameters=_execution_parameters(timeout_s=4.0)
    )
    assert result["timed_out"] is True and result["execution_success"] is False
    assert timeout.commands[-1] == tuple(timeout.state.position)
    interrupted = FakeRuntime(config, _joints(), interrupt=True)
    result = _execute(plan, interrupted, monkeypatch)
    assert result["interrupted"] is True
    assert interrupted.commands[-1] == tuple(interrupted.state.position)


def test_execute_writes_complete_log_and_never_claims_closed_means_gripped(
    config, monkeypatch, tmp_path
):
    plan = _pick_plan(config)
    close_target = tool._stage_target(plan, "grasp-close")
    assert close_target[9] == close_target[16] == 0.1
    open_target = tool._stage_target(plan, "grasp-open")
    assert open_target[9] == open_target[16] == 1.0
    runtime = FakeRuntime(config, _joints())
    result = _execute(plan, runtime, monkeypatch, stage="grasp-close")
    assert result["execution_success"] is True, result
    assert result["stable_grip_verified"] is False
    assert "是否稳定夹住箱子" in result["gripper_calibration_note"]
    path = tool.write_execution_log(result, tmp_path)
    logged = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "seed", "scene", "stage", "trial_parameters", "start_joint_state",
        "target_joint_state", "actual_joint_state_series", "maximum_error",
        "stable_frames", "timed_out", "interrupted", "execution_success",
        "failure_reason", "user_note",
    }
    assert required <= set(logged)
    assert logged["log_path"] == str(path)


def test_probe_is_read_only(config):
    class ProbeRuntime:
        def probe(self, tf_timeout_s):
            assert tf_timeout_s == 5.0
            return {"command": "probe", "published_control": False,
                    "official_offline_conditions_met": True}

    result = tool.probe_environment(config, ProbeRuntime())
    assert result["published_control"] is False


def _fake_odom(frame_id="/odom", quaternion=(0.0, 0.0, 2 ** -0.5, 2 ** -0.5)):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=frame_id),
        pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=-0.7, y=0.55, z=0.002),
            orientation=SimpleNamespace(
                x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
            ),
        )),
    )


class FakeTFBuffer:
    def __init__(self, available_after=1, *, known_frames=("odom", "base_link")):
        self.available_after = available_after
        self.known_frames = known_frames
        self.checks = 0
        self.can_requests = []
        self.lookup_requests = []

    def can_transform(self, target, source, _time):
        self.checks += 1
        self.can_requests.append((target, source))
        return self.checks >= self.available_after

    def lookup_transform(self, target, source, _time):
        self.lookup_requests.append((target, source))
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=123_000_000)),
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=-0.7, y=0.55, z=0.002),
                rotation=SimpleNamespace(
                    x=0.0002456, y=-0.0002455, z=0.7071066, w=0.7071069,
                ),
            ),
        )

    def all_frames_as_yaml(self):
        return yaml.safe_dump({frame: {} for frame in self.known_frames})


class OfflineProbeRuntime(tool.RosCalibrationRuntime):
    def __init__(self, config, *, available_after=1, known_frames=("odom", "base_link")):
        self.config = config
        self.latest_odom = _fake_odom()
        self.latest_joint = tool._joints(_joints(123_000_000))
        self.latest_joint_raw = {
            "name": list(JOINT_NAMES), "position": list(self.latest_joint.position),
        }
        self.tf_buffer = FakeTFBuffer(available_after, known_frames=known_frames)
        self.Time = lambda: object()
        self.now = 0.0

    def wait_for_inputs(self, _timeout):
        return self.latest_joint

    def monotonic(self):
        return self.now

    def spin_once(self, timeout_s):
        self.now += timeout_s

    def _topic_metadata(self):
        result = {
            self.config["topics"]["joint_states"]: ("sensor_msgs/msg/JointState",),
            self.config["topics"]["odom"]: ("nav_msgs/msg/Odometry",),
        }
        for item in tool.MMK2_CONTROLLER_MANIFEST_V1.official_topics:
            if item.group in tool.ARM_TOPIC_GROUPS:
                result[item.topic] = (item.message_type,)
        return result

    def other_publishers(self):
        return {group: 0 for group in tool.ARM_TOPIC_GROUPS}

    def subscriber_counts(self):
        return {group: 1 for group in tool.ARM_TOPIC_GROUPS}


def test_probe_normalizes_odom_and_requires_actual_base_link_only(config, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    runtime = OfflineProbeRuntime(config)
    result = runtime.probe(timeout_s=0.0, tf_timeout_s=5.0)
    assert result["actual_odom_frame"] == "odom"
    assert result["actual_base_frame"] == "base_link"
    assert result["transform_source"] == "tf:odom->base_link"
    assert result["world_equals_odom"] is True
    assert result["official_offline_conditions_met"] is True
    assert result["blockers"] == []
    assert result["published_control"] is False
    assert result["odom_tf_comparison"]["matches"] is True
    assert result["raw_base_transform"] is not None
    assert result["planarized_virtual_footprint_transform"]["translation_xyz"][2] == 0.0
    assert result["virtual_footprint_roll_pitch_yaw"][0:2] == [0.0, 0.0]
    assert result["planarization_scope"] == "official_offline_calibration_only"
    assert runtime.tf_buffer.lookup_requests == [("odom", "base_link")]


@pytest.mark.parametrize("frame", ["/odom", "odom"])
def test_odom_frame_normalization_accepts_optional_leading_slash(frame):
    assert tool.normalize_ros_frame(frame) == "odom"


def test_tf_discovery_spins_then_uses_target_odom_source_base_link(config):
    runtime = OfflineProbeRuntime(config, available_after=2)
    result = runtime.wait_for_transform("/odom", "base_link", 5.0)
    assert runtime.now > 0.0
    assert runtime.tf_buffer.can_requests == [("odom", "base_link")] * 2
    assert runtime.tf_buffer.lookup_requests == [("odom", "base_link")]
    assert result.source_frame == "base_link" and result.target_frame == "odom"


def test_tf_wait_timeout_blocks_and_does_not_publish(config, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    runtime = OfflineProbeRuntime(
        config, available_after=999, known_frames=("odom",)
    )
    result = runtime.probe(timeout_s=0.0, tf_timeout_s=0.1)
    assert result["official_offline_conditions_met"] is False
    assert result["actual_base_frame"] is None
    assert result["transform_source"] is None
    assert result["world_equals_odom"] is False
    assert result["published_control"] is False
    assert any("TF_WAIT_TIMEOUT" in blocker for blocker in result["blockers"])
    assert any("FRAME_NOT_FOUND:base_link" in blocker for blocker in result["blockers"])


def test_tf_timeout_distinguishes_disconnected_frames(config):
    runtime = OfflineProbeRuntime(config, available_after=999)
    with pytest.raises(RuntimeError, match="TRANSFORM_NOT_CONNECTED"):
        runtime.wait_for_transform("odom", "base_link", 0.1)


def test_tf_lookup_exception_is_not_reported_as_wait_timeout(config):
    runtime = OfflineProbeRuntime(config)

    def fail_lookup(_target, _source, _time):
        raise RuntimeError("unexpected buffer failure")

    runtime.tf_buffer.lookup_transform = fail_lookup
    with pytest.raises(RuntimeError, match="LOOKUP_EXCEPTION") as captured:
        runtime.wait_for_transform("odom", "base_link", 5.0)
    assert "TF_WAIT_TIMEOUT" not in str(captured.value)


def test_odom_tf_quaternion_comparison_accepts_q_and_negative_q():
    quaternion = (0.0, 0.0, 2 ** -0.5, 2 ** -0.5)
    base = tool.RigidTransform3D(
        "base_link", "odom", (-0.7, 0.55, 0.002),
        tuple(-item for item in quaternion), 123, True,
    )
    comparison = tool.compare_odom_and_tf_pose(_fake_odom(quaternion=quaternion), base)
    assert comparison["matches"] is True
    assert comparison["quaternion_matches"] is True
    assert comparison["quaternion_sign_equivalent"] is True


def test_odom_tf_comparison_rejects_inverse_direction_values():
    wrong_direction = tool.RigidTransform3D(
        "base_link", "odom", (-0.55, -0.7, -0.002),
        (0.0, 0.0, -(2 ** -0.5), 2 ** -0.5), 123, True,
    )
    assert tool.compare_odom_and_tf_pose(_fake_odom(), wrong_direction)["matches"] is False


def test_planarization_removes_small_roll_pitch_and_keeps_live_xy_yaw():
    raw = tool.RigidTransform3D(
        "base_link", "odom", (-0.7, 0.55, 0.00157),
        (0.0002456, -0.0002455, 0.7071066, 0.7071069), 123, True,
    )
    planar, diagnostics = tool.planarize_base_transform(raw)
    assert planar.translation_xyz == pytest.approx((-0.7, 0.55, 0.0))
    assert diagnostics["raw_roll_pitch_yaw"][0] != 0.0
    assert diagnostics["raw_roll_pitch_yaw"][1] != 0.0
    assert diagnostics["virtual_footprint_roll_pitch_yaw"][0:2] == [0.0, 0.0]
    assert diagnostics["virtual_footprint_roll_pitch_yaw"][2] == pytest.approx(
        diagnostics["raw_roll_pitch_yaw"][2]
    )
    assert math.sqrt(sum(value * value for value in planar.rotation_xyzw)) == pytest.approx(1.0)


def test_virtual_footprint_transform_order_at_ninety_degree_yaw():
    root_half = 2 ** -0.5
    base_in_odom = tool.RigidTransform3D(
        "base_link", "odom", (-0.7, 0.55, 0.002),
        (0.0, 0.0, root_half, root_half), 123, True,
    )
    planar, _ = tool.planarize_base_transform(base_in_odom)
    odom_to_virtual, diagnostics = tool.transform_object_to_virtual_footprint(
        (-0.18, 2.20, 0.834), (0.0, 0.0, 0.0, 1.0), planar
    )
    # inverse(T_odom_virtual_footprint)先减平移再旋转-90度：
    # (-0.18,2.20)-(-0.70,0.55)=(0.52,1.65) -> (1.65,-0.52)
    assert diagnostics["object_pose_in_virtual_footprint"]["position_xyz_m"] == pytest.approx(
        (1.65, -0.52, 0.834), abs=1e-12
    )
    assert diagnostics["object_pose_in_virtual_footprint"]["orientation_xyzw"] == pytest.approx(
        (0.0, 0.0, -root_half, root_half), abs=1e-12
    )
    assert diagnostics["object_local_z_in_virtual_footprint"] == pytest.approx(
        (0.0, 0.0, 1.0), abs=1e-12
    )
    assert odom_to_virtual.source_frame == "odom"
    assert odom_to_virtual.target_frame == "footprint"


def test_wrong_transform_order_is_detected_by_expected_position():
    root_half = 2 ** -0.5
    planar = tool.RigidTransform3D(
        "virtual_footprint", "odom", (-0.7, 0.55, 0.0),
        (0.0, 0.0, root_half, root_half), 123, True,
    )
    object_position = (-0.18, 2.20, 0.834)
    wrong = tuple(
        rotated + translated
        for rotated, translated in zip(
            tool._rotate_vector_by_quaternion(planar.rotation_xyzw, object_position),
            planar.translation_xyz,
        )
    )
    assert wrong != pytest.approx((1.65, -0.52, 0.834), abs=1e-12)


def test_production_upright_guard_remains_active_for_raw_tilt(config):
    payload = _pick_payload()
    payload["transforms"]["target_to_footprint"]["rotation_xyzw"] = [
        0.0002456, -0.0002455, -0.7071066, 0.7071069,
    ]
    result = tool.plan_pick(payload, config, FakeIK())
    assert result["valid"] is False
    assert "局部Z轴竖直向上" in result["failure_reason"]


def test_planarization_adapter_rejects_non_official_environment(config, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "7")
    runtime = OfflineProbeRuntime(config)
    with pytest.raises(RuntimeError, match="官方离线坐标链BLOCKED"):
        runtime._offline_planner_transforms("odom", 123_000_000)


def test_plan_pick_reports_planarized_object_diagnostics_without_publishing(config):
    root_half = 2 ** -0.5
    raw = tool.RigidTransform3D(
        "base_link", "odom", (-0.7, 0.55, 0.002),
        (0.0002456, -0.0002455, 0.7071066, 0.7071069), 900, True,
    )
    planar, planar_diagnostics = tool.planarize_base_transform(raw)
    odom_to_virtual, object_diagnostics = tool.transform_object_to_virtual_footprint(
        (-1.0, 2.2, 0.834), (0.0, 0.0, 0.0, 1.0), planar
    )
    payload = _pick_payload()
    payload["transforms"]["target_to_footprint"] = tool._jsonable(odom_to_virtual)
    payload["transforms"]["coordinate_diagnostics"] = {
        **planar_diagnostics, **object_diagnostics,
        "published_control": False,
    }
    result = tool.plan_pick(payload, config, FakeIK())
    assert result["valid"] is True
    assert result["published_control"] is False
    assert result["planarization_scope"] == "official_offline_calibration_only"
    assert result["object_local_z_in_virtual_footprint"] == pytest.approx(
        (0.0, 0.0, 1.0), abs=1e-12
    )
    assert result["object_pose_in_virtual_footprint"] is not None
    assert result["planarized_virtual_footprint_transform"]["rotation_xyzw"] == pytest.approx(
        (0.0, 0.0, root_half, root_half), abs=1e-6
    )


def test_offline_planner_adapter_is_explicit_and_never_queries_footprint_or_world(
    config, monkeypatch
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    runtime = OfflineProbeRuntime(config)
    transforms = runtime._offline_planner_transforms("/odom", 123_000_000)
    adapter = transforms["calibration_coordinate_adapter"]
    assert adapter == {
        "actual_odom_frame": "odom", "actual_base_frame": "base_link",
        "transform_source": "tf:odom->base_link", "planner_target_label": "footprint",
        "world_equals_odom": True, "scope": "official_offline_calibration_tool_only",
        "planarization_scope": "official_offline_calibration_only",
    }
    assert transforms["target_to_world"]["translation_xyz"] == [0.0, 0.0, 0.0]
    assert all(
        request == ("odom", "base_link")
        for request in runtime.tf_buffer.lookup_requests
    )


class TimingRuntime(tool.RosCalibrationRuntime):
    def __init__(self):
        self.now = 0.0
        self.joint_timing_samples = []
        self.odom_timing_samples = []
        self.tf_timing_samples = []

    def monotonic(self):
        return self.now

    def spin_once(self, timeout_s):
        self.now += timeout_s
        arrival = int(self.now * 1_000_000_000)
        sample = (arrival, arrival - 20_000_000)
        self.joint_timing_samples.append(sample)
        self.odom_timing_samples.append(sample)
        self.tf_timing_samples.append(sample)


def test_timing_reports_observed_streams_and_trial_not_fixture_values():
    result = TimingRuntime().timing(duration_s=1.0, safety_factor=3.0)
    assert result["published_control"] is False
    assert result["streams"]["joint_state"]["frequency_hz"] == pytest.approx(20.0)
    joint_trial = result["trial_candidates"]["joint_state_max_age_ns"]
    assert joint_trial["status"] == "TRIAL_NOT_FROZEN"
    assert joint_trial["value_ns"] == pytest.approx(150_000_000, abs=10)
    assert joint_trial["value_ns"] != 1000
    assert result["trial_candidates"]["object_estimate_max_age_ns"]["value_ns"] is None
    assert result["trial_candidates"]["planned_context_max_age_ns"]["value_ns"] is None
    assert result["trial_candidates"]["confirmed_context_max_age_ns"]["value_ns"] is None


class InstructionRuntime(tool.RosCalibrationRuntime):
    def __init__(self, config, raw):
        self.config = config
        self.raw = raw
        self.wait_calls = 0

    def wait_for_instruction(self, _timeout_s):
        self.wait_calls += 1
        return self.raw, 987_654_321


def test_prepare_pick_input_uses_complete_live_task_and_stage2_fixture(config, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    raw = json.dumps({"tasks": [{
        "task": 1, "instruction": "move pink", "target_kind": "box",
        "target_body": "box_pink", "target_color": "pink",
        "place_type": "table_point", "place_world": [1.25, -0.5, 0.81],
        "place_radius": 0.07,
    }]})
    runtime = InstructionRuntime(config, raw)
    result = runtime.prepare_pick_input(
        1, "table_side_right", 20260709, 0.9, 5.0
    )
    assert runtime.wait_calls == 1
    assert runtime.receive_instruction_tasks.__func__ is (
        tool.RosCalibrationRuntime.receive_instruction_tasks
    )
    assert result["valid"] is True and result["published_control"] is False
    payload = result["payload"]
    fixture = tool.stage2_fixture(config, "table_side_right", "pink")
    assert payload["source"] == "stage2_calibration_fixture"
    assert payload["seed"] == 20260709
    assert payload["task"]["place_world_xyz"] == [1.25, -0.5, 0.81]
    assert payload["task"]["place_radius"] == 0.07
    assert payload["task"]["timestamp_ns"] == 987_654_321
    for field in ("position_xyz", "size_xyz_m", "orientation_xyzw", "frame_id"):
        assert payload["object_estimate"][field] == fixture[field]
    assert payload["field_sources"]["task.place_world_xyz"].startswith("live:")
    assert payload["field_sources"]["task.place_frame_id"].startswith("contract:")
    assert payload["field_sources"]["object_estimate.confidence"].startswith("trial:")
    assert payload["raw_instruction_json"] == raw


def test_prepare_pick_input_blocks_missing_official_fields_without_guessing(config, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    raw = json.dumps({
        "task": 1, "target_body": "box_pink", "target_color": "pink",
        "place_type": "table_point",
    })
    result = InstructionRuntime(config, raw).prepare_pick_input(
        1, "table_side_right", 20260709, 0.9, 5.0
    )
    assert result["valid"] is False
    assert result["status"] == "BLOCKED"
    assert result["payload"] is None
    assert result["published_control"] is False


def test_prepare_pick_input_rejects_non_official_ros_domain(config, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "7")
    result = InstructionRuntime(config, "{}").prepare_pick_input(
        1, "table_side_right", 20260709, 0.9, 5.0
    )
    assert result["valid"] is False
    assert "ROS_DOMAIN_ID=99" in result["blockers"][0]


def test_inspect_does_not_publish_and_reports_safe_defaults(config):
    result = tool.inspect_environment(
        config, topic_output="/joint_states\n/material/instruction\n", check_kdl=False
    )
    assert result["published_control"] is False
    assert result["safe_gates"] == {
        "arm_planning_enabled": False, "observe_only": True,
        "enable_official_publish": False, "simulation_only": True,
    }
    assert "arm_execution.verification_timeout_ns" in result["null_configuration_fields"]


def test_summarize_complete_fields_and_rejects_out_of_order_feedback():
    base = _joints(100)
    payload = {
        "seed": 7, "scenario": "table-side", "task": 1, "object": "track-1",
        "stage": "pregrasp", "planned_target": base["position"],
        "controlled_mask": [True] * 17, "feedback_samples": [base, _joints(200), _joints(300)],
        "tolerances": {"slide_m": 0.01, "arm_rad": 0.01, "gripper": 0.02},
        "settle_cycles": 3, "command_timestamp_ns": 50, "timeout": False,
        "ik": {"success": True}, "joint_limits": {"ok": True}, "user_notes": "visible",
    }
    result = tool.summarize(payload)
    expected = {
        "seed", "scenario", "task", "object", "stage", "planned_target",
        "actual_joint_state", "maximum_error", "settle_time_ns", "stable_frames",
        "timed_out", "execution_success", "ik", "joint_limits", "failure_reason", "user_notes",
    }
    assert expected <= set(result)
    assert result["execution_success"] is True
    payload["feedback_samples"] = [_joints(200), _joints(100)]
    bad = tool.summarize(payload)
    assert bad["execution_success"] is False
    assert "乱序" in bad["failure_reason"]


def test_summarize_skips_duplicate_timestamp_reused_frames():
    # Step A 修复对应：DDS 复用帧（重复时间戳）必须跳过（不计数也不清零），
    # 不能误判为乱序而 break；仅严格倒退才是真乱序。
    position = [0.1, 0.0, 0.0, *([0.0] * 6), 1.0, *([0.0] * 6), 1.0]

    def sample(ts):
        return {
            "position": list(position), "velocity": [0.0] * 17, "effort": [0.0] * 17,
            "joint_names": list(JOINT_NAMES), "timestamp_ns": ts,
        }

    payload = {
        "seed": 7, "scenario": "table-side", "task": 1, "object": "track-1",
        "stage": "pregrasp", "planned_target": position,
        "controlled_mask": [True] * 17,
        # 100 新, 100 复用, 200 新, 200 复用, 300 新 -> 仅 3 个新帧进容差
        "feedback_samples": [sample(100), sample(100), sample(200), sample(200), sample(300)],
        "tolerances": {"slide_m": 0.01, "arm_rad": 0.01, "gripper": 0.02},
        "settle_cycles": 3, "command_timestamp_ns": 50, "timeout": False,
    }
    result = tool.summarize(payload)
    assert result["execution_success"] is True
    assert result["failure_reason"] == ""
    assert result["stable_frames"] == 3  # 仅新帧计数，复用帧跳过（修复前此用例会误判乱序失败）


def test_summarize_resets_stable_when_a_tick_leaves_tolerance():
    position = [0.1, 0.0, 0.0, *([0.0] * 6), 1.0, *([0.0] * 6), 1.0]

    def sample(ts, *, slide=0.1):
        pos = list(position)
        pos[0] = slide
        return {
            "position": pos, "velocity": [0.0] * 17, "effort": [0.0] * 17,
            "joint_names": list(JOINT_NAMES), "timestamp_ns": ts,
        }

    payload = {
        "seed": 7, "scenario": "table-side", "task": 1, "object": "track-1",
        "stage": "pregrasp", "planned_target": position,
        "controlled_mask": [True] * 17,
        # 进容差, 进容差, 出容差(slide=0.5>0.01), 进容差 -> 出容差帧把 stable 清零
        "feedback_samples": [sample(100), sample(200), sample(300, slide=0.5), sample(400)],
        "tolerances": {"slide_m": 0.01, "arm_rad": 0.01, "gripper": 0.02},
        "settle_cycles": 3, "command_timestamp_ns": 50, "timeout": False,
    }
    result = tool.summarize(payload)
    # 末帧是新帧且进容差，但前一帧出容差已把 stable 清零，故仅末帧 stable=1
    assert result["execution_success"] is False  # 未累计到 3 个连续新帧
    assert result["stable_frames"] == 1
    assert result["maximum_error"] == pytest.approx(0.4)  # |0.5 - 0.1|



def test_cli_requires_plan_and_official_confirmation_flag(capsys):
    with pytest.raises(SystemExit):
        tool.build_parser().parse_args(["execute-one-stage", "--stage", "pregrasp"])
    error = capsys.readouterr().err
    assert "--plan" in error
    help_text = tool.build_parser().format_help() + tool.build_parser()._subparsers._group_actions[0].choices[
        "execute-one-stage"
    ].format_help()
    assert "--official-offline-simulation" in help_text


def test_read_only_cli_commands_require_explicit_evidence_parameters():
    parser = tool.build_parser()
    probe = parser.parse_args(["probe", "--tf-timeout-s", "5"])
    assert probe.tf_timeout_s == 5.0
    timing = parser.parse_args(["timing", "--safety-factor", "3"])
    assert timing.duration_s == 10.0
    prepared = parser.parse_args([
        "prepare-pick-input", "--task-id", "1", "--scene", "table_side_right",
        "--seed", "20260709", "--fixture-confidence", "0.9",
    ])
    assert prepared.output == "/data/pick-input.json"
    sweep = parser.parse_args([
        "sweep-pick-stand", "--input", "/data/pick-input.json",
        "--standoff-min-m", "0.40", "--standoff-max-m", "1.00",
        "--standoff-step-m", "0.05",
    ])
    assert sweep.output == "/data/sweep-pick-stand.json"
    assert sweep.standoff_step_m == pytest.approx(0.05)


def test_plan_base_stand_reuses_production_generator_and_matches_current_scene(
    config, monkeypatch
):
    calls = []
    original = tool.NavigationController.build_pick_goal

    def spy(self, task, target, base, timestamp_ns):
        calls.append((task, target, base, timestamp_ns))
        return original(self, task, target, base, timestamp_ns)

    monkeypatch.setattr(tool.NavigationController, "build_pick_goal", spy)
    result = tool.plan_base_stand(
        _right_pick_payload(), config, _base_state(), 0.60, 0.05, 0.10
    )
    assert result["valid"] is True and result["published_control"] is False
    assert result["status"] == "TRIAL_NOT_FROZEN"
    assert len(calls) == 1
    assert result["goal"]["pose_xyyaw"] == pytest.approx(
        (-0.3603, 1.6277, 1.2656), abs=5e-4
    )
    assert result["parameter_sources"]["goal_generator"].endswith(
        "NavigationController.build_pick_goal"
    )


class SingleArmOnlyIK:
    def __init__(self, reason="未找到合法关节解"):
        self.reason = reason
        self.calls = []

    def slide_limits(self):
        return (0.1, 0.1)

    def solve_ik(self, *, left_target, right_target, target_slide, **_kwargs):
        self.calls.append((left_target is not None, right_target is not None, target_slide))
        arm = (0.0,) * 6
        if left_target is not None and right_target is None:
            return IKResult(target_slide, arm, None, True)
        if right_target is not None and left_target is None:
            return IKResult(target_slide, None, arm, True)
        return IKResult(target_slide, None, None, False, self.reason)


def test_standoff_scan_range_is_inclusive_and_validated():
    assert tool._standoff_values(0.40, 1.00, 0.05) == pytest.approx(
        [0.40 + 0.05 * index for index in range(13)]
    )
    with pytest.raises(ValueError):
        tool._standoff_values(1.0, 0.4, 0.05)
    with pytest.raises(ValueError):
        tool._standoff_values(0.4, 1.0, 0.0)


def test_sweep_calls_real_plan_entry_for_every_candidate_and_never_publishes(
    config, monkeypatch
):
    calls = []
    original = tool.plan_pick

    def recording_plan(*args, **kwargs):
        calls.append(args[0]["transforms"]["coordinate_diagnostics"])
        assert kwargs["include_full_ik_diagnostics"] is True
        return original(*args, **kwargs)

    monkeypatch.setattr(tool, "plan_pick", recording_plan)
    result = tool.sweep_pick_stand(
        _right_pick_payload(), config, _base_state(), FakeIK(), 0.40, 0.50, 0.05
    )
    assert len(calls) == 3
    assert len(result["candidates"]) == 3
    assert result["published_control"] is False
    assert all(item["published_control"] is False for item in result["candidates"])
    assert all(item["status"] == "TRIAL_NOT_FROZEN" for item in result["candidates"])
    stages = result["candidates"][0]["ik_diagnostics"]["slide_candidates"][0]["stages"]
    assert [item["stage"] for item in stages] == [
        "pregrasp", "grasp-open", "grasp-close", "short-lift", "retreat"
    ]
    assert all(set(item) >= {"left", "right", "dual"} for item in stages)


def test_single_arm_success_never_counts_as_complete_dual_pick(config):
    result = tool.sweep_pick_stand(
        _right_pick_payload(), config, _base_state(), SingleArmOnlyIK(),
        0.60, 0.60, 0.05,
    )
    candidate = result["candidates"][0]
    first_stage = candidate["ik_diagnostics"]["slide_candidates"][0]["stages"][0]
    assert first_stage["left"]["success"] is True
    assert first_stage["right"]["success"] is True
    assert first_stage["dual"]["success"] is False
    assert first_stage["single_arm_solution_without_dual"] is True
    assert candidate["full_pick_ik_success"] is False
    assert candidate["failed_arm"] == "dual"
    assert result["status"] == "NO_FEASIBLE_STANDOFF"
    assert result["recommended_candidate"] is None


def test_empty_kdl_solution_is_classified(config):
    result = tool.sweep_pick_stand(
        _right_pick_payload(), config, _base_state(),
        SingleArmOnlyIK("官方KDL未返回候选解"), 0.60, 0.60, 0.05,
    )
    stage = result["candidates"][0]["ik_diagnostics"]["slide_candidates"][0]["stages"][0]
    assert stage["dual"]["classification"] == "EMPTY_SOLUTION"
    assert stage["dual"]["kdl_raw_return"] == {
        "type": "NOT_EXPOSED_BY_ADAPTER", "length": None
    }


def test_raw_kdl_return_type_and_length_are_observed_without_changing_result():
    class Solver:
        def inverse_kinematics(self, **_kwargs):
            return []

    class Adapter:
        def __init__(self):
            self._solver = Solver()

        def solve_ik(self, *, target_slide, **kwargs):
            raw = self._solver.inverse_kinematics(**kwargs)
            assert raw == []
            return IKResult(target_slide, None, None, False, "官方 KDL 返回空关节解")

    adapter = Adapter()
    original_solver = adapter._solver
    result, raw = tool._solve_ik_with_raw_metadata(
        adapter, actual_joints=tool._joints(_joints()), left_target=object(),
        right_target=object(), target_slide=0.1,
    )
    assert result.success is False
    assert raw == {"type": "list", "length": 0}
    assert adapter._solver is original_solver


def test_joint_out_of_bounds_is_classified_in_detailed_ik(config):
    payload = _right_pick_payload()
    payload["trial_parameters"]["max_arm_waypoint_delta_rad"] = 20.0
    result = tool.sweep_pick_stand(
        payload, config, _base_state(), FakeIK(arm_value=10.0), 0.60, 0.60, 0.05
    )
    stage = result["candidates"][0]["ik_diagnostics"]["slide_candidates"][0]["stages"][0]
    assert stage["left"]["classification"] == "JOINT_OUT_OF_LIMITS"
    assert stage["right"]["classification"] == "JOINT_OUT_OF_LIMITS"
    assert stage["dual"]["classification"] == "JOINT_OUT_OF_LIMITS"
    assert result["feasible_candidate_count"] == 0


def test_sweep_recommendation_sorts_margin_before_delta(config, monkeypatch):
    def planned(payload, _config, _adapter, **_kwargs):
        relative_x = payload["transforms"]["coordinate_diagnostics"][
            "object_pose_in_virtual_footprint"
        ]["position_xyz_m"][0]
        key = min((0.40, 0.45, 0.50), key=lambda value: abs(value - relative_x))
        margin = {0.40: 0.1, 0.45: 0.2, 0.50: 0.2}[key]
        delta = {0.40: 0.01, 0.45: 0.5, 0.50: 0.1}[key]
        return {
            "valid": True, "failure_reason": "",
            "ik_diagnostics": {
                "all_stages_dual_ik_success": True,
                "slide_candidates": [{
                    "slide_m": 0.1, "complete_dual_path_success": True, "stages": []
                }],
            },
            "waypoints": [{
                "limit_margin_by_joint": {JOINT_NAMES[3]: margin},
                "delta_from_previous_by_joint": {JOINT_NAMES[3]: delta},
            }],
        }

    monkeypatch.setattr(tool, "plan_pick", planned)
    result = tool.sweep_pick_stand(
        _right_pick_payload(), config, _base_state(), FakeIK(), 0.40, 0.50, 0.05
    )
    assert result["feasible_candidate_count"] == 3
    assert result["recommended_candidate"]["standoff_m"] == pytest.approx(0.50)
    assert result["status"] == "TRIAL_NOT_FROZEN"


class FakeBaseRuntime:
    def __init__(self, states, *, publishers=0, subscribers=1, fresh=True,
                 interrupt=False, fail_poll=False, clock_step=0.01):
        self.states = list(states)
        self.last = self.states.pop(0) if self.states else None
        self.pre_publishers = publishers
        self.subscribers = subscribers
        self.fresh = fresh
        self.interrupt = interrupt
        self.fail_poll = fail_poll
        self.clock_step = clock_step
        self.now = 0.0
        self.publisher_started = False
        self.commands = []
        self.odom_age_ns = 0
        self.odom_received_count = 0 if self.last is None else 1

    def monotonic(self):
        self.now += self.clock_step
        return self.now

    def base_graph_counts(self):
        publishers = 1 if self.publisher_started else self.pre_publishers
        return publishers, self.subscribers

    def wait_for_base_state(self, _timeout):
        if self.last is None:
            raise RuntimeError("等待Odom超时")
        return self.last

    def odom_fresh(self, max_age):
        return self.fresh and self.last is not None and self.odom_age_ns <= max_age * 1e9

    def start_base_publisher(self):
        self.publisher_started = True

    def publish_base_velocity(self, v, w):
        self.commands.append((v, w))

    def spin_control_period(self, period_s):
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt
        if self.fail_poll:
            self.fail_poll = False
            raise RuntimeError("fake odom exception")
        self.odom_age_ns += int(period_s * 1_000_000_000)
        if self.states:
            state = self.states.pop(0)
            if state is not None:
                self.last = state
                self.odom_age_ns = 0
                self.odom_received_count += 1

    def latest_base_state(self):
        return self.last

    def latest_odom_age_ns(self):
        return None if self.last is None else self.odom_age_ns

    def poll_base_state(self, _timeout):
        if self.states:
            state = self.states.pop(0)
            if state is not None:
                self.last = state
                self.odom_received_count += 1
        else:
            self.last = replace(self.last, timestamp_ns=self.last.timestamp_ns + 1)
            self.odom_received_count += 1
        self.odom_age_ns = 0
        return self.last


def _base_plan(config):
    result = tool.plan_base_stand(
        _right_pick_payload(), config, _base_state(), 0.60, 0.05, 0.10
    )
    assert result["valid"] is True
    return result


def _precise_base_plan(config):
    result = tool.plan_base_stand(
        _right_pick_payload(), config, _base_state(), 0.75, 0.01, 0.02
    )
    assert result["valid"] is True
    return result


def _goal_states(plan, count=8):
    x, y, yaw = plan["goal"]["pose_xyyaw"]
    return [_base_state(x, y, yaw, 1100 + index) for index in range(count)]


def _execute_base(plan, config, runtime, monkeypatch, *, simulation=True,
                  confirmation=tool.OFFICIAL_BASE_CONFIRMATION):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    return tool.execute_base_stand(
        plan, config, simulation, confirmation, "用户观察", runtime
    )


@pytest.mark.parametrize(
    ("runtime_kwargs", "message"),
    [
        ({"publishers": 1}, "其他publisher"),
        ({"subscribers": 0}, "subscriber必须恰好为1"),
        ({"subscribers": 2}, "subscriber必须恰好为1"),
        ({"fresh": False}, "Odom过期"),
    ],
)
def test_execute_base_preflight_blocks_without_publishing(
    config, monkeypatch, runtime_kwargs, message
):
    plan = _base_plan(config)
    runtime = FakeBaseRuntime(_goal_states(plan), **runtime_kwargs)
    result = _execute_base(plan, config, runtime, monkeypatch)
    assert result["execution_success"] is False
    assert message in result["failure_reason"]
    assert runtime.commands == []


def test_execute_base_rejects_domain_confirmation_and_simulation_gate(config, monkeypatch):
    plan = _base_plan(config)
    runtime = FakeBaseRuntime(_goal_states(plan))
    monkeypatch.setenv("ROS_DOMAIN_ID", "7")
    result = tool.execute_base_stand(
        plan, config, True, tool.OFFICIAL_BASE_CONFIRMATION, "", runtime
    )
    assert "ROS_DOMAIN_ID=99" in result["failure_reason"] and runtime.commands == []
    assert _execute_base(plan, config, runtime, monkeypatch, simulation=False)[
        "execution_success"
    ] is False
    assert _execute_base(plan, config, runtime, monkeypatch, confirmation="wrong")[
        "execution_success"
    ] is False
    assert runtime.commands == []


def test_execute_base_requires_three_arrival_frames_and_confirms_zero_stop(config, monkeypatch):
    plan = _base_plan(config)
    runtime = FakeBaseRuntime(_goal_states(plan))
    result = _execute_base(plan, config, runtime, monkeypatch)
    assert result["execution_success"] is True
    assert len(result["command_series"]) == 3
    assert result["stop_evidence"]["confirmed_stopped"] is True
    assert runtime.commands[-3:] == [(0.0, 0.0)] * 3
    assert all(abs(v) <= 0.25 and abs(w) <= 0.50 for v, w in runtime.commands)


def test_precise_base_plan_persists_trial_tolerances_without_mutating_config(config):
    production_position = config["navigation"]["position_tolerance_m"]
    production_yaw = config["navigation"]["yaw_tolerance_rad"]
    plan = _precise_base_plan(config)
    assert plan["status"] == "TRIAL_NOT_FROZEN"
    assert plan["navigation_parameters"]["standoff_m"] == pytest.approx(0.75)
    assert plan["navigation_parameters"]["position_tolerance_m"] == pytest.approx(0.01)
    assert plan["navigation_parameters"]["yaw_tolerance_rad"] == pytest.approx(0.02)
    assert plan["goal"]["position_tolerance"] == pytest.approx(0.01)
    assert plan["goal"]["yaw_tolerance"] == pytest.approx(0.02)
    assert config["navigation"]["position_tolerance_m"] == production_position == 0.05
    assert config["navigation"]["yaw_tolerance_rad"] == production_yaw == 0.10
    assert plan["published_control"] is False


def test_coarse_production_arrival_does_not_pass_precise_trial(config, monkeypatch):
    plan = _precise_base_plan(config)
    x, y, yaw = plan["goal"]["pose_xyyaw"]
    coarse = [
        _base_state(
            x + 0.0432177986, y, yaw + 0.0595067851, 1100 + index
        )
        for index in range(8)
    ]
    result = _execute_base(
        plan, config, FakeBaseRuntime(coarse, clock_step=0.25), monkeypatch
    )
    assert result["execution_success"] is False
    assert result["final_distance_error_m"] == pytest.approx(0.0432177986)
    check = result["arm_reachability_precheck"]
    assert check["yaw_alignment_error_rad"] == pytest.approx(0.0595067851)
    assert check["meets_calibration_precision"] is False
    assert result["stop_evidence"]["confirmed_stopped"] is True


def test_precise_arrival_uses_plan_tolerances_and_reports_arm_precheck(
    config, monkeypatch
):
    plan = _precise_base_plan(config)
    x, y, yaw = plan["goal"]["pose_xyyaw"]
    precise = [
        _base_state(x + 0.005, y, yaw + 0.01, 1100 + index)
        for index in range(8)
    ]
    runtime = FakeBaseRuntime(precise)
    result = _execute_base(plan, config, runtime, monkeypatch)
    check = result["arm_reachability_precheck"]
    assert result["execution_success"] is True
    assert result["navigation_parameters"]["position_tolerance_m"] == pytest.approx(0.01)
    assert result["navigation_parameters"]["yaw_tolerance_rad"] == pytest.approx(0.02)
    assert check["actual_object_pose_in_virtual_footprint"] is not None
    assert check["expected"] == pytest.approx([0.75, 0.0, 0.834])
    assert check["planar_alignment_error_m"] < 0.01
    assert check["yaw_alignment_error_rad"] < 0.02
    assert check["meets_calibration_precision"] is True
    assert check["object_coordinates_modified"] is False
    assert result["stop_evidence"]["confirmed_stopped"] is True
    assert runtime.commands[-3:] == [(0.0, 0.0)] * 3


def test_control_24hz_reuses_fresh_odom_from_approximately_50ms_stream(config, monkeypatch):
    plan = _base_plan(config)
    x, y, yaw = plan["goal"]["pose_xyyaw"]
    states = [
        _base_state(x, y, yaw, 1000),
        None,  # 41.67ms control tick；下一帧约50ms才到
        _base_state(x, y, yaw, 1050),
        _base_state(x, y, yaw, 1100),
        _base_state(x, y, yaw, 1150),
        _base_state(x, y, yaw, 1200),
        _base_state(x, y, yaw, 1250),
    ]
    runtime = FakeBaseRuntime(states)
    result = _execute_base(plan, config, runtime, monkeypatch)
    assert result["execution_success"] is True
    assert result["odom_reused_tick_count"] >= 1
    assert result["latest_odom_age_ns_by_tick"][0] == 41_666_666
    assert result["max_observed_odom_age_ns"] < 150_000_000
    assert len(result["command_series"]) >= 3


def test_one_tick_without_new_odom_continues_while_cache_is_fresh(config, monkeypatch):
    plan = _base_plan(config)
    x, y, yaw = plan["goal"]["pose_xyyaw"]
    runtime = FakeBaseRuntime([
        _base_state(x, y, yaw, 1000), None,
        _base_state(x, y, yaw, 1050), _base_state(x, y, yaw, 1100),
        _base_state(x, y, yaw, 1150), _base_state(x, y, yaw, 1200),
        _base_state(x, y, yaw, 1250),
    ])
    result = _execute_base(plan, config, runtime, monkeypatch)
    assert result["failure_classification"] == ""
    assert result["command_series"][0]["timestamp_ns"] == 1000
    assert result["odom_reused_tick_count"] == 1


def test_cached_odom_older_than_150ms_stops_with_stale_classification(config, monkeypatch):
    plan = _base_plan(config)
    x, y, yaw = plan["goal"]["pose_xyyaw"]
    runtime = FakeBaseRuntime([
        _base_state(-0.7, 0.55005, math.pi / 2, 1000),
        None, None, None, None,
        _base_state(x, y, yaw, 1200), _base_state(x, y, yaw, 1250),
        _base_state(x, y, yaw, 1300),
    ])
    result = _execute_base(plan, config, runtime, monkeypatch)
    assert result["execution_success"] is False
    assert result["failure_classification"] == "STALE"
    assert result["latest_odom_age_ns_by_tick"][-1] > 150_000_000
    assert result["max_observed_odom_age_ns"] > 150_000_000
    assert runtime.commands[-3:] == [(0.0, 0.0)] * 3


def test_execute_base_never_received_odom_blocks_before_publisher(config, monkeypatch):
    plan = _base_plan(config)
    runtime = FakeBaseRuntime([])
    result = _execute_base(plan, config, runtime, monkeypatch)
    assert result["failure_classification"] == "NEVER_RECEIVED"
    assert result["published_control"] is False
    assert runtime.commands == []


def test_jittered_odom_sequence_reaches_three_distinct_settled_frames(config, monkeypatch):
    plan = _base_plan(config)
    x, y, yaw = plan["goal"]["pose_xyyaw"]
    runtime = FakeBaseRuntime([
        _base_state(x, y, yaw, 1000), None,
        _base_state(x, y, yaw, 1048), None,
        _base_state(x, y, yaw, 1098),
        _base_state(x, y, yaw, 1147), _base_state(x, y, yaw, 1199),
        _base_state(x, y, yaw, 1248),
    ])
    result = _execute_base(plan, config, runtime, monkeypatch)
    assert result["execution_success"] is True
    assert result["control_tick_count"] >= 4
    assert result["odom_received_count"] >= 3
    assert result["odom_stale_threshold_ns"] == 150_000_000
    assert result["stop_evidence"]["confirmed_stopped"] is True


@pytest.mark.parametrize("mode", ["timeout", "exception", "interrupt"])
def test_execute_base_all_runtime_exit_paths_publish_zero(config, monkeypatch, mode):
    plan = _base_plan(config)
    kwargs = {}
    if mode == "timeout":
        kwargs["clock_step"] = 31.0
    elif mode == "exception":
        kwargs["fail_poll"] = True
    else:
        kwargs["interrupt"] = True
    runtime = FakeBaseRuntime(_goal_states(plan), **kwargs)
    result = _execute_base(plan, config, runtime, monkeypatch)
    assert result["execution_success"] is False
    assert runtime.commands[-3:] == [(0.0, 0.0)] * 3
    if mode == "timeout":
        assert result["timed_out"] is True
    if mode == "interrupt":
        assert result["interrupted"] is True


def test_base_cli_exposes_plan_only_default_and_explicit_execution_gate():
    parser = tool.build_parser()
    planned = parser.parse_args([
        "plan-base-stand", "--input", "/data/pick-input.json",
        "--standoff-m", "0.60", "--position-tolerance-m", "0.01",
        "--yaw-tolerance-rad", "0.02",
    ])
    assert planned.output == "/data/plan-base-stand.json"
    assert planned.position_tolerance_m == pytest.approx(0.01)
    assert planned.yaw_tolerance_rad == pytest.approx(0.02)
    with pytest.raises(SystemExit):
        parser.parse_args([
            "plan-base-stand", "--input", "/data/pick-input.json",
            "--standoff-m", "0.75",
        ])
    executed = parser.parse_args([
        "execute-base-stand", "--plan", "/data/plan-base-stand.json",
        "--official-offline-simulation", "--confirm",
        tool.OFFICIAL_BASE_CONFIRMATION,
    ])
    assert executed.official_offline_simulation is True


def test_worktree_changes_are_limited_to_calibration_tool_files():
    allowed = {
        "scripts/arm_pick_place_calibration.py",
        "docs/arm_pick_place_calibration.md",
        "tests/test_arm_pick_place_calibration.py",
    }
    output = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    changed = {line[3:] for line in output if line}
    assert changed <= allowed


class SequentialBranchIK:
    def __init__(self, stage_values, *, alternatives=None):
        self.stage_values = list(stage_values)
        self.alternatives = alternatives or {}
        self.calls = []

    def slide_limits(self):
        return (0.1, 0.1)

    def solve_ik(self, *, target_slide, **_kwargs):
        return IKResult(target_slide, None, None, False, "use calibration candidates")

    def solve_ik_candidates(
        self, *, actual_joints, left_target, right_target, target_slide
    ):
        index = len(self.calls)
        self.calls.append({
            "position": tuple(actual_joints.position),
            "left_target": left_target,
            "right_target": right_target,
            "target_slide": target_slide,
        })
        value = self.stage_values[index]
        candidates = [
            IKResult(target_slide, tuple(value), tuple(value), True)
        ]
        for alternative in self.alternatives.get(index, []):
            candidates.append(
                IKResult(target_slide, tuple(alternative), tuple(alternative), True)
            )
        return candidates

    def forward_kinematics(self, q):
        left = [[1.0, 0.0, 0.0, q[1]], [0.0, 1.0, 0.0, q[2]],
                [0.0, 0.0, 1.0, q[3]], [0.0, 0.0, 0.0, 1.0]]
        right = [[1.0, 0.0, 0.0, q[7]], [0.0, 1.0, 0.0, q[8]],
                 [0.0, 0.0, 1.0, q[9]], [0.0, 0.0, 0.0, 1.0]]
        return left, right


class TransitionReadyIK(SequentialBranchIK):
    def __init__(self, stage_values):
        super().__init__(stage_values)
        self.production_calls = 0

    def solve_ik(self, *, target_slide, **_kwargs):
        value = self.stage_values[self.production_calls % len(self.stage_values)]
        self.production_calls += 1
        return IKResult(target_slide, tuple(value), tuple(value), True)


def _transition_pick_plan(config):
    high = (0.0, 0.0, 2.5942780707, 0.0, 0.0, 0.0)
    payload = _pick_payload()
    payload["trial_parameters"].update({
        "max_arm_waypoint_delta_rad": 1.0,
        "pregrasp_duration_s": 2.0,
        "grasp_duration_s": 4.0,
        "lift_duration_s": 2.0,
        "retreat_duration_s": 2.0,
    })
    result = tool.plan_pick(payload, config, TransitionReadyIK([high] * 4))
    assert result["plan_artifact_valid"] is True
    return result


def _transition_execution_parameters(**updates):
    values = _execution_parameters(
        control_rate_hz=24.0, timeout_s=6.0, settle_cycles=3,
        feedback_max_age_s=0.131175150,
    )
    values.update(updates)
    return values


def _live_sequence_fixture(config, *, instruction_after_spins=2, instruction_raw=None):
    plan = _transition_pick_plan(config)
    plan["trial_parameters"]["planned_context_max_age_ns"] = 10_000_000_000
    plan["planarized_virtual_footprint_transform"] = {
        "source_frame": "virtual_footprint", "target_frame": "odom",
        "translation_xyz": [-1.75, 2.2, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "timestamp_ns": 1000, "valid": True, "failure_reason": "",
    }
    runtime = FakeRuntime(
        config, plan["start_joint_state"],
        instruction_after_spins=instruction_after_spins,
        instruction_raw=instruction_raw,
    )
    runtime.probe = lambda **_kwargs: {
        "official_offline_conditions_met": True,
        "blockers": [],
        "planarized_virtual_footprint_transform": dict(
            plan["planarized_virtual_footprint_transform"]
        ),
    }
    return plan, runtime


def _calibration_pose_pairs():
    return tuple((name, object(), object()) for name in (
        "PREGRASP", "GRASP", "LIFT", "RETREAT"
    ))


def _analysis(config, adapter):
    payload = _pick_payload()
    planning = tool.planning_config(config, payload["trial_parameters"])
    return tool._calibration_pick_sequence_analysis(
        adapter, tool._joints(payload["joint_state"]),
        _calibration_pose_pairs(), planning, config,
    )


def test_calibration_joint_name_index_mapping_matches_official_1_plus_6_plus_6(config):
    adapter = SequentialBranchIK([(0.0,) * 6] * 4)
    analysis = _analysis(config, adapter)
    order = analysis["joint_order_and_limits"]
    assert order["official_kinematic_order"] == [
        "slide_joint", *JOINT_NAMES[3:9], *JOINT_NAMES[10:16]
    ]
    assert order["left_team_indices"] == [3, 4, 5, 6, 7, 8]
    assert order["right_team_indices"] == [10, 11, 12, 13, 14, 15]


def test_first_ik_uses_current_joint_state_seed_and_later_stages_use_previous_solution(config):
    values = [
        (0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.3, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.4, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    adapter = SequentialBranchIK(values)
    analysis = _analysis(config, adapter)
    assert analysis["available"] is True
    assert adapter.calls[0]["position"] == tuple(_joints()["position"])
    assert adapter.calls[1]["position"][3:9] == values[0]
    assert adapter.calls[2]["position"][3:9] == values[1]
    assert adapter.calls[3]["position"][10:16] == values[2]
    reports = analysis["stage_branch_selection"]
    assert reports[0]["seed_kind"] == "current_joint_state"
    assert all(item["seed_kind"] == "previous_stage_ik" for item in reports[1:])


def test_multiple_ik_solutions_select_minimum_weighted_change_then_margin(config):
    close = (0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    farther = (1.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    adapter = SequentialBranchIK(
        [farther, close, close, close], alternatives={0: [close]}
    )
    analysis = _analysis(config, adapter)
    first = analysis["stage_branch_selection"][0]
    assert first["legal_candidate_count"] == 2
    assert first["selected_joint_vector"][1:7] == pytest.approx(close)
    assert first["selection_policy"].startswith("within_effective_limits")
    assert first["periodic_equivalent_adjustment_by_calibration_tool"] is False


def test_continuity_failure_diagnostic_contains_values_speed_duration_and_seed(config):
    high = (0.0, 0.0, 2.4, 0.0, 0.0, 0.0)
    adapter = SequentialBranchIK([high, high, high, high])
    analysis = _analysis(config, adapter)
    violation = analysis["continuity_diagnostics"]["violations"][0]
    required = {
        "waypoint_index", "stage", "arm", "joint_name",
        "current_joint_value_rad", "target_joint_value_rad",
        "signed_delta_rad", "absolute_delta_rad", "continuity_limit_rad",
        "minimum_required_time_s_at_0_6_rad_s", "stage_duration_s",
        "satisfies_speed_limit", "ik_branch_and_seed",
    }
    assert required <= set(violation)
    assert violation["waypoint_index"] == 0
    assert violation["stage"] == "pregrasp"
    assert violation["arm"] == "left"
    assert violation["joint_name"] == "left_arm_joint3"
    assert violation["current_joint_value_rad"] == pytest.approx(0.0)
    assert violation["target_joint_value_rad"] == pytest.approx(2.4)
    assert violation["signed_delta_rad"] == pytest.approx(2.4)
    assert violation["absolute_delta_rad"] == pytest.approx(2.4)
    assert violation["continuity_limit_rad"] == pytest.approx(1.0)
    assert violation["minimum_required_time_s_at_0_6_rad_s"] == pytest.approx(4.0)
    assert violation["stage_duration_s"] == pytest.approx(1.0)
    assert violation["satisfies_speed_limit"] is False


def test_transition_segments_obey_step_speed_limits_joint_limits_and_fk(config):
    values = [
        (0.0, 0.0, 2.4, 0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.8, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.5, 0.0, 0.0, 0.0),
    ]
    analysis = _analysis(config, SequentialBranchIK(values))
    transition = analysis["transition_plan"]
    assert analysis["continuous_ik_branch_exists"] is False
    assert analysis["transition_required"] is True
    assert transition["segment_count"] == math.ceil(2.4 / 1.0)
    assert transition["max_single_segment_arm_delta_rad"] <= 1.0
    assert transition["all_step_limits_ok"] is True
    assert transition["all_speed_limits_ok"] is True
    assert transition["all_joint_limits_ok"] is True
    assert transition["all_fk_checks_ok"] is True
    assert all(item["fk"]["success"] for item in transition["segments"])
    assert all(item["left_gripper"] == item["right_gripper"] == 1.0
               for item in transition["segments"])
    assert transition["collision_visual_verification_required"] is False
    assert transition["manual_simulation_observation_recommended"] is True
    assert transition["collision_check_available"] is False
    assert transition["status"] == "NOT_AUTOMATICALLY_CHECKED"
    assert transition["executable"] is False
    assert transition["published_control"] is False


def test_plan_pick_promotes_only_initial_continuity_failure_to_shared_transition_artifact(
    config, monkeypatch
):
    high = (0.0, 0.0, 2.5942780707, 0.0, 0.0, 0.0)
    adapter = TransitionReadyIK([high, high, high, high])
    payload = _pick_payload()
    payload["trial_parameters"].update({
        "max_arm_waypoint_delta_rad": 1.0,
        "pregrasp_duration_s": 2.0,
        "grasp_duration_s": 4.0,
        "lift_duration_s": 2.0,
        "retreat_duration_s": 2.0,
    })
    original = tool._calibration_transition_plan
    calls = []

    def shared_planner(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(tool, "_calibration_transition_plan", shared_planner)
    result = tool.plan_pick(payload, config, adapter)
    transition = result["transition_plan"]
    assert calls, "plan-pick必须复用compare使用的唯一过渡规划函数"
    assert result["status"] == "TRANSITION_PLAN_READY_FOR_MANUAL_SINGLE_STAGE_SIMULATION"
    assert result["valid"] is True
    assert result["plan_artifact_valid"] is True
    assert result["automatic_execution_ready"] is False
    assert result["single_stage_execution_ready"] is True
    assert result["visual_transition_review_ready"] is True
    assert result["endpoint_ik_success"] is True
    assert result["joint_limits_ok"] is True
    assert result["transition_required"] is True
    assert result["transition_segment_count"] == math.ceil(2.5942780707 / 1.0)
    assert result["transition_segment_count"] == 3
    assert result["max_single_segment_joint_delta_rad"] <= 1.0
    assert result["total_transition_duration_s"] + 1e-12 >= 2.5942780707 / 0.6
    assert result["minimum_joint_limit_margin"] >= 0.0
    assert result["all_fk_checks_ok"] is True
    assert result["collision_verification_status"] == "NOT_AUTOMATICALLY_CHECKED"
    assert result["published_control"] is False
    assert result["transition_3_reaches_pregrasp"] is True
    assert transition["all_step_limits_ok"] is True
    assert transition["all_speed_limits_ok"] is True
    assert transition["all_joint_limits_ok"] is True
    assert transition["all_fk_checks_ok"] is True
    assert all(item["max_arm_delta_rad"] <= 1.0 for item in transition["segments"])
    assert all(item["left_gripper"] == item["right_gripper"] == 1.0
               for item in transition["segments"])
    assert all(item["fk"]["success"] for item in transition["segments"])
    assert all("left_end_position_xyz_m" in item["fk"] for item in transition["segments"])
    assert all("right_end_position_xyz_m" in item["fk"] for item in transition["segments"])
    assert [item["stage"] for item in result["calibration_waypoints"]] == [
        "initial-joint-state", "transition-1", "transition-2", "transition-3",
        "pregrasp", "grasp-open", "grasp-close", "short-lift", "retreat",
    ]
    assert all(item["joint_limits_ok"] and item["speed_limits_ok"]
               for item in result["waypoints"])


def test_transition_single_stage_executes_only_selected_segment(config, monkeypatch):
    plan = _transition_pick_plan(config)
    runtime = FakeRuntime(config, plan["start_joint_state"])
    result = _execute(
        plan, runtime, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    first = tuple(plan["transition_plan"]["segments"][0]["joint_position"])
    second = tuple(plan["transition_plan"]["segments"][1]["joint_position"])
    assert result["execution_success"] is True
    assert result["stage"] == "transition-1"
    assert result["reached_target"] is True
    assert result["final_max_joint_error"] <= 0.01
    assert result["settled_cycles"] >= 3
    assert result["next_stage"] == "transition-2"
    assert runtime.commands[-1] == first
    assert runtime.commands[-1] != second


def test_transition_24hz_reuses_fresh_joint_state_from_50ms_stream(
    config, monkeypatch
):
    plan = _transition_pick_plan(config)
    runtime = FakeRuntime(
        config, plan["start_joint_state"], feedback_interval_s=0.050,
    )
    result = _execute(
        plan, runtime, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert result["execution_success"] is True
    assert result["failure_classification"] == ""
    assert result["feedback_reused_tick_count"] >= 1
    assert result["max_feedback_age_s"] < 0.131175150
    assert result["feedback_received_count"] >= 4
    assert len(runtime.commands) >= result["control_tick_count"] > 3
    assert any(sample["cache_reused"] for sample in result["actual_joint_state_series"])


def test_transition_one_tick_without_new_feedback_continues_and_publishes(
    config, monkeypatch
):
    plan = _transition_pick_plan(config)
    runtime = FakeRuntime(
        config, plan["start_joint_state"], feedback_interval_s=0.0498,
    )
    result = _execute(
        plan, runtime, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert result["actual_joint_state_series"][0]["cache_reused"] is True
    assert result["feedback_age_s_by_tick"][0] == pytest.approx(1.0 / 24.0)
    assert result["execution_success"] is True
    assert len(runtime.commands) > 1


def test_transition_truly_stale_feedback_holds_latest_and_logs_final_errors(
    config, monkeypatch
):
    plan = _transition_pick_plan(config)
    runtime = FakeRuntime(config, plan["start_joint_state"], no_feedback=True)
    result = _execute(
        plan, runtime, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert result["execution_success"] is False
    assert result["failure_classification"] == "STALE"
    assert result["max_feedback_age_s"] > 0.131175150
    assert result["safe_stop_mode"] == "confirmed_real_joint_state_hold_then_stop_publishing"
    assert result["hold_evidence"]["status"] == "STOP_UNCONFIRMED"
    assert result["stop_status"] == "STOP_UNCONFIRMED"
    assert runtime.commands[-1] == tuple(runtime.state.position)
    assert result["final_joint_state"] is not None
    assert set(result["final_joint_error_by_joint"]) == set(JOINT_NAMES)
    assert result["final_max_joint_error"] == max(
        result["final_joint_error_by_joint"].values()
    )
    assert result["control_tick_count"] >= 4
    assert result["feedback_received_count"] == 1
    required = {
        "execution_success", "valid", "stage", "failure_reason",
        "failure_classification", "published_control", "control_tick_count",
        "feedback_received_count", "feedback_reused_tick_count",
        "max_feedback_age_s", "start_joint_state", "expected_start_joint_state",
        "start_error_by_joint", "target_joint_state", "final_joint_state",
        "final_error_by_joint", "final_max_joint_error", "settled_cycles",
        "next_stage", "timed_out", "interrupted",
        "publisher_conflict_evidence", "hold_evidence",
    }
    assert required <= set(result)


def test_transition_never_received_feedback_blocks_without_publisher(
    config, monkeypatch
):
    class NeverReceivedRuntime(FakeRuntime):
        def wait_for_inputs(self, _timeout):
            self.state = None
            self.joint_received_count = 0
            raise RuntimeError("等待/joint_states超时")

    plan = _transition_pick_plan(config)
    runtime = NeverReceivedRuntime(config, plan["start_joint_state"])
    result = _execute(
        plan, runtime, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert result["failure_classification"] == "NEVER_RECEIVED"
    assert result["published_control"] is False
    assert result["final_joint_state"] == {
        "available": False, "reason": "NO_VALID_FEEDBACK"
    }
    assert result["final_joint_error_by_joint"] == {}
    assert runtime.commands == []


@pytest.mark.parametrize(
    ("age_ns", "previous", "validation_error", "classification"),
    [
        (131_175_151, 1000, None, "STALE"),
        (0, 1001, None, "INVALID"),
        (0, 1000, "JointState缺少团队必需关节", "INVALID"),
    ],
)
def test_cached_joint_feedback_rejects_only_true_stale_or_invalid_samples(
    age_ns, previous, validation_error, classification
):
    state = tool._joints(_joints(timestamp_ns=1000))
    result = tool._evaluate_cached_feedback(
        state, age_ns, previous, 131_175_150, "JointState", validation_error,
    )
    assert result["valid"] is False
    assert result["classification"] == classification


def test_start_match_uses_separate_slide_arm_and_gripper_tolerances():
    expected = list(_joints()["position"])
    actual = list(expected)
    actual[0] += 0.009
    actual[3] += 0.009
    actual[9] -= 0.019
    diagnostics = tool._start_match_diagnostics(
        actual, expected, _transition_execution_parameters()
    )
    assert diagnostics["matches"] is True
    assert diagnostics["max_slide_error_m"] == pytest.approx(0.009)
    assert diagnostics["max_arm_error_joint"] == "left_arm_joint1"
    assert diagnostics["max_gripper_error_side"] == "left"
    for index, expected_failure in ((0, "slide"), (3, "arm"), (9, "gripper")):
        damaged = list(expected)
        damaged[index] += 0.021
        check = tool._start_match_diagnostics(
            damaged, expected, _transition_execution_parameters()
        )
        assert check["matches"] is False, expected_failure


def test_old_plan_contract_and_insufficient_timeout_fail_before_publish(
    config, monkeypatch
):
    plan = _transition_pick_plan(config)
    old = dict(plan)
    old.pop("execution_contract_version")
    runtime = FakeRuntime(config, plan["start_joint_state"])
    result = _execute(
        old, runtime, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert result["failure_classification"] == "OLD_PLAN_EXECUTION_CONTRACT"
    assert "OLD_PLAN_EXECUTION_CONTRACT" in result["failure_reason"]
    assert runtime.commands == []
    runtime = FakeRuntime(config, plan["start_joint_state"])
    result = _execute(
        plan, runtime, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(timeout_s=0.5),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert result["failure_classification"] == "TIMEOUT_CONFIGURATION_INVALID"
    assert result["timing_validation"]["timeout_valid"] is False
    assert "建议至少" in result["failure_reason"]
    assert runtime.commands == []


def test_publisher_self_identity_and_bounded_dds_residual_are_not_conflicts(
    config, monkeypatch
):
    plan = _transition_pick_plan(config)
    runtime = FakeRuntime(
        config, plan["start_joint_state"], dds_residual_cycles=2
    )
    result = _execute(
        plan, runtime, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert result["execution_success"] is True
    assert runtime.publisher_start_count == 1
    assert result["publisher_conflict_evidence"][0]["sample_count"] == 3
    assert all(item["valid"] for item in result["publisher_conflict_evidence"])


def test_same_process_pick_sequence_keeps_publishers_and_never_skips(
    config, monkeypatch
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    monkeypatch.setattr(
        tool, "_live_execution_context_evidence",
        lambda *_args, **_kwargs: {"valid": True, "blockers": []},
    )
    plan = _transition_pick_plan(config)
    runtime = FakeRuntime(config, plan["start_joint_state"])
    requested = []

    def confirm(stage, *_args):
        requested.append(stage)
        return tool.SEQUENCE_STAGE_CONFIRMATION

    result = tool.execute_pick_calibration_sequence(
        plan, True, tool.OFFICIAL_SIM_CONFIRMATION,
        _transition_execution_parameters(timeout_s=8.0), "test", runtime,
        7, "table_side_left", 5.0, confirm,
    )
    assert result["execution_success"] is True
    assert requested == list(tool.PICK_CALIBRATION_SEQUENCE)
    assert result["completed_stages"] == list(tool.PICK_CALIBRATION_SEQUENCE)
    assert runtime.publisher_start_count == 1
    assert all(item["hold_evidence"]["confirmed"] for item in result["stage_results"])
    assert "pregrasp" not in requested


def test_same_process_sequence_user_exit_holds_and_cannot_enter_next_stage(
    config, monkeypatch
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    monkeypatch.setattr(
        tool, "_live_execution_context_evidence",
        lambda *_args, **_kwargs: {"valid": True, "blockers": []},
    )
    plan = _transition_pick_plan(config)
    runtime = FakeRuntime(config, plan["start_joint_state"])

    def confirm(stage, *_args):
        return tool.SEQUENCE_STAGE_CONFIRMATION if stage == "transition-1" else "STOP"

    result = tool.execute_pick_calibration_sequence(
        plan, True, tool.OFFICIAL_SIM_CONFIRMATION,
        _transition_execution_parameters(timeout_s=8.0), "test", runtime,
        7, "table_side_left", 5.0, confirm,
    )
    assert result["completed_stages"] == ["transition-1"]
    assert result["failure_classification"] == "USER_SAFE_EXIT"
    assert result["hold_evidence"]["confirmed"] is True
    assert runtime.publisher_start_count == 1


def test_sequence_waits_for_delayed_live_instruction_on_shared_runtime(
    config, monkeypatch
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    plan, runtime = _live_sequence_fixture(config, instruction_after_spins=3)
    command_counts_at_confirmation = []

    def stop_at_confirmation(_stage, _latest, active_runtime, _params):
        command_counts_at_confirmation.append(len(active_runtime.commands))
        return "STOP"

    result = tool.execute_pick_calibration_sequence(
        plan, True, tool.OFFICIAL_SIM_CONFIRMATION,
        _transition_execution_parameters(timeout_s=8.0), "test", runtime,
        7, "table_side_left", 5.0, stop_at_confirmation,
    )
    evidence = result["instruction_reception_evidence"]
    assert result["failure_classification"] == "USER_SAFE_EXIT"
    assert result["execution_context_evidence"]["valid"] is True
    assert runtime.instruction_receive_call_count == 1
    assert runtime.instruction_spin_count == 4
    assert evidence["received"] is True
    assert evidence["joint_feedback_count_after_wait"] > evidence[
        "joint_feedback_count_before_wait"
    ]
    assert command_counts_at_confirmation == [0]
    assert result["stage_results"] == []
    assert result["instruction_qos"] == {
        "history": "KEEP_LAST", "depth": 10,
        "reliability": "RELIABLE", "durability": "VOLATILE",
        "compatibility_basis": "fake compatible endpoint",
    }


def test_sequence_live_instruction_timeout_is_classified_before_publishers(
    config, monkeypatch
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    plan, runtime = _live_sequence_fixture(config, instruction_after_spins=999)
    result = tool.execute_pick_calibration_sequence(
        plan, True, tool.OFFICIAL_SIM_CONFIRMATION,
        _transition_execution_parameters(timeout_s=8.0), "test", runtime,
        7, "table_side_left", 0.1,
        lambda *_args: tool.SEQUENCE_STAGE_CONFIRMATION,
    )
    assert result["failure_classification"] == "LIVE_CONTEXT_UNAVAILABLE"
    assert "完整JSON超时" in result["failure_reason"]
    assert result["instruction_reception_evidence"]["received"] is False
    assert runtime.publishers == {} and runtime.commands == []
    assert result["published_control"] is False


def test_sequence_live_instruction_parse_failure_is_context_unavailable(
    config, monkeypatch
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    plan, runtime = _live_sequence_fixture(
        config, instruction_after_spins=1, instruction_raw="{not-json",
    )
    result = tool.execute_pick_calibration_sequence(
        plan, True, tool.OFFICIAL_SIM_CONFIRMATION,
        _transition_execution_parameters(timeout_s=8.0), "test", runtime,
        7, "table_side_left", 5.0,
        lambda *_args: tool.SEQUENCE_STAGE_CONFIRMATION,
    )
    assert result["failure_classification"] == "LIVE_CONTEXT_UNAVAILABLE"
    assert "合法 JSON" in result["failure_reason"]
    assert runtime.publishers == {} and runtime.commands == []
    assert result["published_control"] is False


def test_sequence_cli_exposes_timeout_verbose_and_short_terminal_summary():
    parser = tool.build_parser()
    args = parser.parse_args([
        "execute-pick-calibration-sequence", "--plan", "/data/plan.json",
        "--expected-seed", "20260709", "--expected-scene", "table_side_right",
        "--instruction-timeout-s", "5", "--max-slide-velocity-m-s", "0.15",
        "--max-arm-velocity-rad-s", "0.6",
        "--max-gripper-velocity-per-s", "0.6", "--control-rate-hz", "24",
        "--timeout-s", "8", "--feedback-max-age-s", "0.131175150",
        "--slide-tolerance-m", "0.01", "--arm-tolerance-rad", "0.01",
        "--gripper-tolerance", "0.02", "--settle-cycles", "3", "--verbose",
    ])
    assert args.instruction_timeout_s == 5.0 and args.verbose is True
    summary = tool._sequence_terminal_summary({
        "command": "execute-pick-calibration-sequence",
        "execution_success": False, "completed_stages": [],
        "next_stage": "transition-1", "failure_classification": "TEST",
        "failure_reason": "test", "published_control": False,
        "probe_evidence": {"hundreds": list(range(100))},
        "final_joint_state": {"position": [0.0] * 17},
        "instruction_reception_evidence": {"received": True},
        "log_path": "/data/log.json",
    })
    assert summary["instruction_received"] is True
    assert "probe_evidence" not in summary and "final_joint_state" not in summary


def test_live_sequence_context_rejects_seed_scene_task_and_lost_base_precision(
    config, monkeypatch
):
    plan = _transition_pick_plan(config)
    plan["planarized_virtual_footprint_transform"] = {
        "source_frame": "virtual_footprint", "target_frame": "odom",
        "translation_xyz": [-1.75, 2.2, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "timestamp_ns": 1000, "valid": True, "failure_reason": "",
    }
    runtime = SimpleNamespace(
        latest_joint_state=lambda: tool._joints(_joints(timestamp_ns=1000)),
    )
    live_tasks = [tool._task(plan["task"])]
    good_probe = {
        "official_offline_conditions_met": True,
        "planarized_virtual_footprint_transform": dict(
            plan["planarized_virtual_footprint_transform"]
        ),
    }
    good = tool._live_execution_context_evidence(
        plan, runtime, good_probe, 7, "table_side_left", live_tasks, 1000
    )
    assert good["valid"] is True
    assert good["base_precision"]["meets_calibration_precision"] is True
    wrong_identity = tool._live_execution_context_evidence(
        plan, runtime, good_probe, 8, "table_side_right", live_tasks, 1000
    )
    assert {"SEED_MISMATCH", "SCENE_MISMATCH"} <= set(wrong_identity["blockers"])
    shifted = json.loads(json.dumps(good_probe))
    shifted["planarized_virtual_footprint_transform"]["translation_xyz"][0] += 0.02
    lost = tool._live_execution_context_evidence(
        plan, runtime, shifted, 7, "table_side_left", live_tasks, 1000
    )
    assert lost["valid"] is False
    assert "BASE_CALIBRATION_PRECISION_LOST" in lost["blockers"]
    runtime.latest_joint_state = lambda: tool._joints(_joints(timestamp_ns=3000))
    stale = tool._live_execution_context_evidence(
        plan, runtime, good_probe, 7, "table_side_left", live_tasks, 1000
    )
    assert "PLAN_CONTEXT_STALE" in stale["blockers"]
    runtime.latest_joint_state = lambda: tool._joints(_joints(timestamp_ns=800))
    reset = tool._live_execution_context_evidence(
        plan, runtime, good_probe, 7, "table_side_left", live_tasks, 1000
    )
    assert "SERVER_CLOCK_RESET_OR_PLAN_FROM_FUTURE" in reset["blockers"]


def test_transition_rejects_skip_and_mismatched_start(config, monkeypatch):
    plan = _transition_pick_plan(config)
    skipped = FakeRuntime(config, plan["start_joint_state"])
    result = _execute(
        plan, skipped, monkeypatch, stage="transition-2",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert "禁止跳段" in result["failure_reason"]
    assert skipped.publishers == {} and skipped.commands == []
    wrong = json.loads(json.dumps(plan["start_joint_state"]))
    wrong["position"][3] += 0.02
    mismatched = FakeRuntime(config, wrong)
    result = _execute(
        plan, mismatched, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert "段起点" in result["failure_reason"]
    assert mismatched.publishers == {} and mismatched.commands == []


def test_transition_allows_only_three_named_stages_and_matching_identity(
    config, monkeypatch
):
    parser = tool.build_parser()
    parsed = parser.parse_args([
        "execute-one-stage", "--plan", "/data/plan.json",
        "--stage", "transition-3", "--expected-seed", "7",
        "--expected-scene", "table_side_left", "--official-offline-simulation",
        "--confirm", tool.OFFICIAL_SIM_CONFIRMATION,
        "--max-slide-velocity-m-s", "0.15", "--max-arm-velocity-rad-s", "0.6",
        "--max-gripper-velocity-per-s", "0.6", "--control-rate-hz", "24",
        "--timeout-s", "6", "--feedback-max-age-s", "0.5",
        "--slide-tolerance-m", "0.01", "--arm-tolerance-rad", "0.01",
        "--gripper-tolerance", "0.02", "--settle-cycles", "3",
    ])
    assert parsed.stage == "transition-3"
    with pytest.raises(SystemExit):
        parser.parse_args([
            "execute-one-stage", "--plan", "/data/plan.json",
            "--stage", "transition-4",
        ])
    plan = _transition_pick_plan(config)
    runtime = FakeRuntime(config, plan["start_joint_state"])
    result = _execute(
        plan, runtime, monkeypatch, stage="transition-4",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert "一次" in result["failure_reason"]
    for seed, scene in ((8, "table_side_left"), (7, "table_side_right")):
        runtime = FakeRuntime(config, plan["start_joint_state"])
        result = _execute(
            plan, runtime, monkeypatch, stage="transition-1",
            parameters=_transition_execution_parameters(),
            expected_seed=seed, expected_scene=scene,
        )
        assert "严格匹配" in result["failure_reason"]
        assert runtime.commands == []


def test_transition_probe_publisher_and_nonofficial_environment_block(
    config, monkeypatch
):
    plan = _transition_pick_plan(config)
    for runtime, message in (
        (FakeRuntime(config, plan["start_joint_state"], probe_pass=False), "probe"),
        (FakeRuntime(config, plan["start_joint_state"], conflict=True), "外部Publisher"),
    ):
        result = _execute(
            plan, runtime, monkeypatch, stage="transition-1",
            parameters=_transition_execution_parameters(),
            expected_seed=7, expected_scene="table_side_left",
        )
        assert message in result["failure_reason"]
        assert runtime.publishers == {} and runtime.commands == []
    monkeypatch.setenv("ROS_DOMAIN_ID", "7")
    runtime = FakeRuntime(config, plan["start_joint_state"])
    result = tool.execute_one_stage(
        plan, "transition-1", True, tool.OFFICIAL_SIM_CONFIRMATION,
        _transition_execution_parameters(), "test", runtime,
        expected_seed=7, expected_scene="table_side_left",
    )
    assert "ROS_DOMAIN_ID" in result["failure_reason"]
    assert runtime.commands == []


@pytest.mark.parametrize("damage", ["speed", "limit", "fk"])
def test_transition_rejects_unsafe_saved_segment(config, monkeypatch, damage):
    plan = json.loads(json.dumps(_transition_pick_plan(config)))
    segment = plan["transition_plan"]["segments"][0]
    if damage == "speed":
        segment["speed_limits_ok"] = False
    elif damage == "limit":
        segment["joint_position"][3] = 99.0
    else:
        segment["fk"]["success"] = False
    runtime = FakeRuntime(config, plan["start_joint_state"])
    result = _execute(
        plan, runtime, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert result["execution_success"] is False
    assert runtime.publishers == {} and runtime.commands == []


@pytest.mark.parametrize("mode", ["interrupt", "exception"])
def test_transition_interrupt_and_exception_hold_latest_then_stop(
    config, monkeypatch, mode
):
    plan = _transition_pick_plan(config)
    runtime = FakeRuntime(
        config, plan["start_joint_state"], interrupt=mode == "interrupt",
        fail_poll=mode == "exception",
    )
    result = _execute(
        plan, runtime, monkeypatch, stage="transition-1",
        parameters=_transition_execution_parameters(),
        expected_seed=7, expected_scene="table_side_left",
    )
    assert result["execution_success"] is False
    assert result["safe_stop_mode"] == "confirmed_real_joint_state_hold_then_stop_publishing"
    assert result["hold_evidence"]["confirmed"] is True
    assert runtime.commands[-1] == tuple(runtime.state.position)
    assert result["final_joint_state"] is not None
    assert set(result["final_joint_error_by_joint"]) == set(JOINT_NAMES)
    assert result["final_max_joint_error"] is not None
    assert result["control_tick_count"] >= 0


def test_normal_ik_failure_never_fabricates_transition_artifact(config):
    result = tool.plan_pick(_pick_payload(), config, FakeIK(success=False))
    assert result["valid"] is False
    assert result.get("plan_artifact_valid") is not True
    assert result.get("transition_plan") is None
    assert result["published_control"] is False


def test_compare_standoffs_with_fixture_stays_blocked_and_plan_only(config):
    values = [(0.0,) * 6] * 8
    result = tool.compare_pick_standoffs(
        _right_pick_payload(), config, _base_state(), tool._joints(_joints()),
        SequentialBranchIK(values), evidence_source="test_fixture",
    )
    assert result["status"] == "BLOCKED"
    assert result["recommended_candidate"] is None
    assert result["published_control"] is False
    assert len(result["candidates"]) == 2
    assert all(item["published_control"] is False for item in result["candidates"])


def test_execution_rejects_diagnostic_3_2_guard_and_transition_plan(config, monkeypatch):
    plan = _pick_plan(config)
    plan["trial_parameters"]["max_arm_waypoint_delta_rad"] = 3.2
    result = _execute(plan, FakeRuntime(config, _joints()), monkeypatch)
    assert result["published_control"] is False
    assert "不得超过1.0" in result["failure_reason"]
    plan = _pick_plan(config)
    plan["calibration_analysis"] = {"transition_required": True}
    result = _execute(plan, FakeRuntime(config, _joints()), monkeypatch)
    assert result["published_control"] is False
    assert "仅供规划" in result["failure_reason"]


class ComparisonCaptureRuntime:
    def __init__(self, raw, base=None):
        self.latest_joint_raw = raw
        self._base = base or _base_state(timestamp_ns=2_000)
        self.publishers = {}
        self.base_publisher = None
        self.waited = None

    def wait_for_inputs(self, timeout_s):
        self.waited = timeout_s
        return object()

    def latest_base_state(self):
        return self._base


def _raw_joint_message_fixture(names=None):
    raw_names = list(reversed(JOINT_NAMES)) if names is None else list(names)
    canonical = {name: index / 100.0 for index, name in enumerate(JOINT_NAMES)}
    return {
        "name": raw_names,
        "position": [canonical[name] for name in raw_names],
        "velocity": [canonical[name] + 1.0 for name in raw_names],
        "effort": [canonical[name] + 2.0 for name in raw_names],
        "header": {
            "frame_id": "",
            "stamp": {"sec": 0, "nanosec": 1_000},
            "timestamp_ns": 1_000,
        },
        "tool_received_at_ns": 1_500,
    }


def test_capture_comparison_state_preserves_raw_and_normalizes_exact_joint_order(
    config, monkeypatch
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    raw = _raw_joint_message_fixture()
    runtime = ComparisonCaptureRuntime(raw)
    result = tool.capture_pick_comparison_state(
        config, runtime, "table_side_right", 20260709, 5.0
    )
    assert result["valid"] is True
    assert result["blockers"] == []
    assert result["source"] == "/joint_states"
    assert result["raw_joint_state"] == raw
    assert result["tool_received_at_ns"] == 1_500
    assert result["joint_state_header_timestamp_ns"] == 1_000
    assert result["joint_name_validation"]["exact_joint_set"] is True
    assert result["normalized_joint_names"] == list(JOINT_NAMES)
    assert result["normalized_position"] == pytest.approx(
        [index / 100.0 for index in range(17)]
    )
    assert result["normalized_position_by_joint"] == pytest.approx({
        name: index / 100.0 for index, name in enumerate(JOINT_NAMES)
    })
    assert result["normalized_velocity"] == pytest.approx(
        [index / 100.0 + 1.0 for index in range(17)]
    )
    assert result["normalized_effort"] == pytest.approx(
        [index / 100.0 + 2.0 for index in range(17)]
    )
    assert result["joint_state"]["joint_names"] == list(JOINT_NAMES)
    assert result["joint_state"]["timestamp_ns"] == 1_000
    assert result["publisher_objects_created"] is False
    assert result["published_control"] is False
    assert runtime.waited == pytest.approx(5.0)
    state, base, evidence, metadata = tool._comparison_fixture(result)
    assert state.position == pytest.approx(result["normalized_position"])
    assert base.frame_id == "odom"
    assert evidence == "saved_official_joint_state"
    assert metadata["state_fixture_mode"] == "recorded_official_joint_state"
    result["normalized_position_by_joint"]["slide_joint"] = 999.0
    with pytest.raises(ValueError, match="normalized_position_by_joint"):
        tool._comparison_fixture(result)


def test_capture_json_loads_into_both_real_compare_candidates(
    config, monkeypatch, tmp_path
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    captured = tool.capture_pick_comparison_state(
        config, ComparisonCaptureRuntime(_raw_joint_message_fixture()),
        "table_side_right", 20260709, 5.0,
    )
    fixture_path = tmp_path / "pick-comparison-state.json"
    fixture_path.write_text(json.dumps(captured), encoding="utf-8")
    prepared = _right_pick_payload()
    for field in ("joint_state", "now_ns", "transforms"):
        prepared.pop(field)
    input_path = tmp_path / "pick-input.json"
    input_path.write_text(json.dumps(prepared), encoding="utf-8")
    output_path = tmp_path / "compare.json"
    adapter = SequentialBranchIK([(0.0,) * 6] * 100)
    adapter.slide_limits = lambda: (-0.04, 0.87)
    adapter.self_check = lambda: None
    monkeypatch.setattr(tool, "OfficialKDLAdapter", lambda *_args: adapter)
    monkeypatch.setattr(
        tool, "RosCalibrationRuntime",
        lambda *_args: pytest.fail("compare不得创建ROS runtime或Publisher"),
    )
    return_code = tool.main([
        "--config", str(ROOT / "config/config.yaml"),
        "compare-pick-standoffs", "--input", str(input_path),
        "--state-fixture", str(fixture_path), "--output", str(output_path),
    ])
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert return_code == 0
    assert len(result["candidates"]) == 2
    assert [item["standoff_m"] for item in result["candidates"]] == [0.55, 0.75]
    assert result["state_fixture_mode"] == "recorded_official_joint_state"
    assert result["state_fixture_capture"]["tool_received_at_ns"] == 1_500
    assert result["published_control"] is False
    expected = tuple(captured["normalized_position"])
    assert adapter.calls
    assert adapter.calls[0]["position"] == expected
    assert expected != (0.0,) * 17


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda fixture: fixture.pop("joint_state"),
         "STATE_FIXTURE_FIELD_MISSING: joint_state"),
        (lambda fixture: fixture.update(joint_state={"recorded": fixture["joint_state"]}),
         "STATE_FIXTURE_JOINT_ORDER_INVALID"),
        (lambda fixture: fixture["raw_joint_state"]["name"].pop(),
         "STATE_FIXTURE_JOINT_SET_INVALID"),
        (lambda fixture: fixture["normalized_joint_names"].reverse(),
         "normalized_joint_names"),
        (lambda fixture: fixture["normalized_position"].__setitem__(0, float("nan")),
         "STATE_FIXTURE_VECTOR_INVALID"),
        (lambda fixture: fixture.update(tool_received_at_ns=999),
         "STATE_FIXTURE_TIMESTAMP_INVALID"),
    ],
)
def test_comparison_fixture_rejects_missing_nested_and_invalid_state(
    config, monkeypatch, mutation, message
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    fixture = tool.capture_pick_comparison_state(
        config, ComparisonCaptureRuntime(_raw_joint_message_fixture()),
        "table_side_right", 20260709, 5.0,
    )
    mutation(fixture)
    with pytest.raises(ValueError, match=message):
        tool._comparison_fixture(fixture)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scene", "table_side_left", "STATE_FIXTURE_SCENE_MISMATCH"),
        ("seed", 7, "STATE_FIXTURE_SEED_MISMATCH"),
    ],
)
def test_comparison_fixture_rejects_pick_input_identity_mismatch(
    config, monkeypatch, tmp_path, field, value, message
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    fixture = tool.capture_pick_comparison_state(
        config, ComparisonCaptureRuntime(_raw_joint_message_fixture()),
        "table_side_right", 20260709, 5.0,
    )
    path = tmp_path / "state.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    payload = _right_pick_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        tool._load_comparison_fixture(path, payload)


def test_comparison_fixture_requires_existing_file():
    with pytest.raises(ValueError, match="STATE_FIXTURE_FILE_NOT_FOUND"):
        tool._load_comparison_fixture(
            "/definitely/missing/pick-comparison-state.json",
            {"scene": "table_side_right", "seed": 20260709},
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw["name"].__setitem__(0, "unexpected_joint"), "17关节集合"),
        (lambda raw: raw["velocity"].pop(), "velocity"),
        (lambda raw: raw.update(tool_received_at_ns=999), "时钟域"),
    ],
)
def test_capture_comparison_state_rejects_incomplete_or_invalid_raw_joint_message(
    config, monkeypatch, mutate, message
):
    monkeypatch.setenv("ROS_DOMAIN_ID", "99")
    raw = _raw_joint_message_fixture()
    mutate(raw)
    result = tool.capture_pick_comparison_state(
        config, ComparisonCaptureRuntime(raw), "table_side_right", 20260709, 5.0
    )
    assert result["valid"] is False
    assert result["published_control"] is False
    assert any(message in blocker for blocker in result["blockers"])


def test_capture_and_compare_cli_arguments_are_explicit():
    parser = tool.build_parser()
    captured = parser.parse_args([
        "capture-pick-comparison-state", "--scene", "table_side_right",
        "--seed", "20260709", "--joint-state-timeout-s", "5",
        "--output", "/data/pick-comparison-state.json",
    ])
    assert captured.joint_state_timeout_s == pytest.approx(5.0)
    compared = parser.parse_args([
        "compare-pick-standoffs", "--input", "/data/pick-input.json",
        "--state-fixture", "/data/pick-comparison-state.json",
    ])
    assert compared.output == "/data/compare-pick-standoffs.json"


def test_compare_outputs_fixed_candidate_parameters_fk_and_collision_status(config):
    result = tool.compare_pick_standoffs(
        _right_pick_payload(), config, _base_state(), tool._joints(_joints()),
        SequentialBranchIK([(0.0,) * 6] * 8), evidence_source="test_fixture",
    )
    assert [
        (item["standoff_m"], item["lift_distance_m"], item["retreat_distance_m"])
        for item in result["candidates"]
    ] == [(0.55, 0.02, 0.02), (0.75, 0.05, 0.10)]
    for candidate in result["candidates"]:
        assert isinstance(candidate["continuous_ik_branch_exists"], bool)
        assert "transition_segment_count" in candidate
        assert "max_single_segment_joint_delta_rad" in candidate
        assert "minimum_joint_limit_margin" in candidate
        assert candidate["all_fk_checks_ok"] is True
        assert len(candidate["fk_checks"]) == 5
        assert candidate["collision_check_available"] is False
        assert candidate["collision_visual_verification_required"] is False
        assert candidate["collision_verification_status"] == "NOT_AUTOMATICALLY_CHECKED"
        assert candidate["published_control"] is False
        assert candidate["parameter_sources"]["max_arm_waypoint_delta_rad"].endswith("1.0")
