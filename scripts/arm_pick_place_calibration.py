#!/usr/bin/env python3
"""4.4--4.6 官方离线仿真专用机械臂标定助手。

``probe``、``plan-pick``、``plan-place``、``sweep-pick-stand``和``summarize``不发布控制命令。只有
``execute-one-stage``或人工逐段确认的``execute-pick-calibration-sequence``在显式官方离线
确认、有效plan文件和独占publisher检查全部通过后，才由本脚本发布标定目标；它不得用于
真实硬件或无人确认的完整自动抓放。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields, replace
import io
import inspect
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import team_sorting.arm_planning as _arm_planning  # noqa: E402
from team_sorting.arm_planning import ArmPlanner, OfficialKDLAdapter  # noqa: E402
from team_sorting.fsm import InstructionParser  # noqa: E402
from team_sorting.navigation import (  # noqa: E402
    NavigationConfig,
    NavigationController,
    wrap_to_pi,
)
from team_sorting.interfaces import (  # noqa: E402
    ArmPlanningConfig,
    BaseState,
    GraspContext,
    IKResult,
    JOINT_NAMES,
    ObjectEstimate3D,
    NavGoal,
    RigidTransform3D,
    RobotJointState,
    SlotType,
    TaskSpec,
)
from team_sorting.controller_manifest import MMK2_CONTROLLER_MANIFEST_V1  # noqa: E402

COMMAND_SCHEMA = "team_sorting.arm_calibration.v1"
EXECUTION_CONTRACT_VERSION = 2
OFFICIAL_SIM_CONFIRMATION = "I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION"
SEQUENCE_STAGE_CONFIRMATION = "I_CONFIRM_EXECUTE_NEXT_STAGE"
OFFICIAL_BASE_CONFIRMATION = "I_CONFIRM_OFFICIAL_OFFLINE_BASE_MOTION"
FIXTURE_SOURCE = "stage2_calibration_fixture"
GRIPPER_NOTE = (
    "closed=0.1是已验证闭合工作点；是否稳定夹住箱子仍由本轮动态测试决定。"
)
SCENE_FIXTURE_KEYS = {
    "table_side_left": ("table_side", 0),
    "table_side_right": ("table_side", 1),
    "table_top": ("table_top", 0),
    "shelf_low": ("shelf", 0),
    "shelf_middle": ("shelf", 1),
    "shelf_high": ("shelf", 2),
}
STAGES = (
    "pregrasp",
    "grasp-open",
    "grasp-close",
    "short-lift",
    "retreat",
    "preplace",
    "lower",
    "release",
    "post-release-retreat",
    "return-start",
)
TRANSITION_STAGES = ("transition-1", "transition-2", "transition-3")
PICK_CALIBRATION_SEQUENCE = (
    "transition-1", "transition-2", "transition-3", "grasp-open",
    "grasp-close", "short-lift", "retreat",
)

ARM_TOPIC_GROUPS = ("spine", "head", "left_arm", "right_arm")
OFFICIAL_ODOM_FRAME = "odom"
OFFICIAL_BASE_FRAME = "base_link"
ODOM_TF_TRANSLATION_TOLERANCE_M = 0.02
ODOM_TF_QUATERNION_DISTANCE_TOLERANCE = 0.02
BASE_CONTROL_RATE_HZ = 24.0
BASE_STOP_CONFIRMATION_TIMEOUT_S = 2.0
CALIBRATION_ARM_STEP_LIMIT_RAD = 1.0
CALIBRATION_SLIDE_STEP_LIMIT_M = 0.20
CALIBRATION_ARM_SPEED_LIMIT_RAD_S = 0.6
CALIBRATION_SLIDE_SPEED_LIMIT_M_S = 0.15
CALIBRATION_JOINT_WEIGHTS = (1.0,) * 12


def normalize_ros_frame(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("ROS frame必须是字符串")
    normalized = value.lstrip("/")
    if not normalized:
        raise ValueError("ROS frame去除前导斜杠后不能为空")
    return normalized


def _normalized_quaternion(value: Sequence[float]) -> tuple[float, float, float, float]:
    quaternion = tuple(float(item) for item in value)
    if len(quaternion) != 4 or not all(math.isfinite(item) for item in quaternion):
        raise ValueError("四元数必须包含4个有限数")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm == 0.0:
        raise ValueError("四元数范数不能为0")
    return tuple(item / norm for item in quaternion)  # type: ignore[return-value]


def _rotate_vector_by_quaternion(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    x, y, z, w = _normalized_quaternion(quaternion)
    vx, vy, vz = (float(item) for item in vector)
    cross_x = y * vz - z * vy
    cross_y = z * vx - x * vz
    cross_z = x * vy - y * vx
    return (
        vx + 2.0 * (w * cross_x + y * cross_z - z * cross_y),
        vy + 2.0 * (w * cross_y + z * cross_x - x * cross_z),
        vz + 2.0 * (w * cross_z + x * cross_y - y * cross_x),
    )


def _quaternion_multiply(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = _normalized_quaternion(left)
    rx, ry, rz, rw = _normalized_quaternion(right)
    return _normalized_quaternion((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ))


def _quaternion_to_rpy(quaternion: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = _normalized_quaternion(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def planarize_base_transform(
    base_in_odom: RigidTransform3D,
) -> tuple[RigidTransform3D, dict[str, Any]]:
    """把官方base_link六自由度位姿投影成只保留XY+yaw的虚拟footprint。"""

    if (
        normalize_ros_frame(base_in_odom.source_frame) != OFFICIAL_BASE_FRAME
        or normalize_ros_frame(base_in_odom.target_frame) != OFFICIAL_ODOM_FRAME
    ):
        raise ValueError("平面化输入必须是target=odom、source=base_link的TF2查询结果")
    if base_in_odom.translation_xyz is None or base_in_odom.rotation_xyzw is None:
        raise ValueError("base_link在odom中的位姿缺少平移或旋转")
    raw_rpy = _quaternion_to_rpy(base_in_odom.rotation_xyzw)
    yaw = raw_rpy[2]
    planar_quaternion = _normalized_quaternion(
        (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    )
    planar = RigidTransform3D(
        source_frame="virtual_footprint",
        target_frame=OFFICIAL_ODOM_FRAME,
        translation_xyz=(
            float(base_in_odom.translation_xyz[0]),
            float(base_in_odom.translation_xyz[1]),
            0.0,
        ),
        rotation_xyzw=planar_quaternion,
        timestamp_ns=base_in_odom.timestamp_ns,
        valid=True,
    )
    return planar, {
        "raw_base_transform": _jsonable(base_in_odom),
        "planarized_virtual_footprint_transform": _jsonable(planar),
        "raw_roll_pitch_yaw": list(raw_rpy),
        "virtual_footprint_roll_pitch_yaw": [0.0, 0.0, yaw],
        "planarization_scope": "official_offline_calibration_only",
    }


def transform_object_to_virtual_footprint(
    position_in_odom: Sequence[float],
    orientation_in_odom: Sequence[float],
    planar_footprint_in_odom: RigidTransform3D,
) -> tuple[RigidTransform3D, dict[str, Any]]:
    """计算 inverse(T_odom_virtual_footprint) * T_odom_object。"""

    odom_to_virtual = _inverse_transform(planar_footprint_in_odom, "footprint")
    if odom_to_virtual.translation_xyz is None or odom_to_virtual.rotation_xyzw is None:
        raise ValueError("odom到virtual footprint变换不完整")
    object_position = tuple(float(item) for item in position_in_odom)
    if len(object_position) != 3 or not all(math.isfinite(item) for item in object_position):
        raise ValueError("odom物体位置必须包含3个有限数")
    object_orientation = _normalized_quaternion(orientation_in_odom)
    position_in_virtual = tuple(
        rotated + translated
        for rotated, translated in zip(
            _rotate_vector_by_quaternion(odom_to_virtual.rotation_xyzw, object_position),
            odom_to_virtual.translation_xyz,
        )
    )
    orientation_in_virtual = _quaternion_multiply(
        odom_to_virtual.rotation_xyzw, object_orientation
    )
    local_z = _rotate_vector_by_quaternion(orientation_in_virtual, (0.0, 0.0, 1.0))
    diagnostics = {
        "object_pose_in_odom": {
            "position_xyz_m": list(object_position),
            "orientation_xyzw": list(object_orientation),
            "frame_id": OFFICIAL_ODOM_FRAME,
        },
        "object_pose_in_virtual_footprint": {
            "position_xyz_m": list(position_in_virtual),
            "orientation_xyzw": list(orientation_in_virtual),
            "frame_id": "virtual_footprint",
        },
        "object_local_z_in_virtual_footprint": list(local_z),
        "transform_equation": (
            "T_virtual_footprint_object="
            "inverse(T_odom_virtual_footprint)*T_odom_object"
        ),
    }
    return odom_to_virtual, diagnostics


def _inverse_transform(transform: RigidTransform3D, target_frame: str) -> RigidTransform3D:
    if transform.translation_xyz is None or transform.rotation_xyzw is None:
        raise ValueError("无法反转缺少平移或旋转的TF")
    x, y, z, w = _normalized_quaternion(transform.rotation_xyzw)
    inverse_rotation = (-x, -y, -z, w)
    inverse_translation = _rotate_vector_by_quaternion(
        inverse_rotation, tuple(-item for item in transform.translation_xyz)
    )
    return RigidTransform3D(
        source_frame=transform.target_frame,
        target_frame=target_frame,
        translation_xyz=inverse_translation,
        rotation_xyzw=inverse_rotation,
        timestamp_ns=transform.timestamp_ns,
        valid=True,
    )


def compare_odom_and_tf_pose(odom: Any, base_in_odom: RigidTransform3D) -> dict[str, Any]:
    if base_in_odom.translation_xyz is None or base_in_odom.rotation_xyzw is None:
        raise ValueError("TF base pose缺少平移或旋转")
    position = odom.pose.pose.position
    orientation = odom.pose.pose.orientation
    odom_translation = (float(position.x), float(position.y), float(position.z))
    odom_quaternion = _normalized_quaternion(
        (orientation.x, orientation.y, orientation.z, orientation.w)
    )
    tf_quaternion = _normalized_quaternion(base_in_odom.rotation_xyzw)
    translation_error = math.sqrt(sum(
        (left - right) ** 2
        for left, right in zip(odom_translation, base_in_odom.translation_xyz)
    ))
    same_sign_distance = math.sqrt(sum(
        (left - right) ** 2 for left, right in zip(odom_quaternion, tf_quaternion)
    ))
    opposite_sign_distance = math.sqrt(sum(
        (left + right) ** 2 for left, right in zip(odom_quaternion, tf_quaternion)
    ))
    quaternion_distance = min(same_sign_distance, opposite_sign_distance)
    translation_matches = translation_error <= ODOM_TF_TRANSLATION_TOLERANCE_M
    quaternion_matches = quaternion_distance <= ODOM_TF_QUATERNION_DISTANCE_TOLERANCE
    return {
        "comparison_direction": "odom_message_base_pose_vs_tf_target_odom_source_base_link",
        "translation_error_m": translation_error,
        "translation_tolerance_m": ODOM_TF_TRANSLATION_TOLERANCE_M,
        "translation_matches": translation_matches,
        "quaternion_sign_invariant_distance": quaternion_distance,
        "quaternion_distance_tolerance": ODOM_TF_QUATERNION_DISTANCE_TOLERANCE,
        "quaternion_matches": quaternion_matches,
        "quaternion_sign_equivalent": opposite_sign_distance < same_sign_distance,
        "matches": translation_matches and quaternion_matches,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("输入JSON顶层必须是object")
    return value


def _load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("配置顶层必须是mapping")
    return value


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _task(raw: Mapping[str, Any]) -> TaskSpec:
    return TaskSpec(**dict(raw))


def _joints(raw: Mapping[str, Any]) -> RobotJointState:
    data = dict(raw)
    if "joint_names" in data:
        data["joint_names"] = tuple(data["joint_names"])
    for name in ("position", "velocity", "effort"):
        data[name] = tuple(data[name])
    return RobotJointState(**data)


def _estimate(raw: Mapping[str, Any]) -> ObjectEstimate3D:
    data = dict(raw)
    data["slot_type"] = SlotType(data.get("slot_type", "unknown"))
    for name in ("position_xyz", "orientation_xyzw", "size_xyz_m"):
        if data.get(name) is not None:
            data[name] = tuple(data[name])
    return ObjectEstimate3D(**data)


def _transform(raw: Mapping[str, Any]) -> RigidTransform3D:
    data = dict(raw)
    for name in ("translation_xyz", "rotation_xyzw"):
        if data.get(name) is not None:
            data[name] = tuple(data[name])
    return RigidTransform3D(**data)


def _grasp_context(raw: Mapping[str, Any]) -> GraspContext:
    data = dict(raw)
    for name in ("object_from_left_gripper", "object_from_right_gripper"):
        if data.get(name) is not None:
            data[name] = _transform(data[name])
    for name in ("object_size_xyz_m", "object_orientation_world_xyzw_at_grasp"):
        if data.get(name) is not None:
            data[name] = tuple(data[name])
    return GraspContext(**data)


def planning_config(config: Mapping[str, Any], trial: Mapping[str, Any]) -> ArmPlanningConfig:
    """只允许显式试验值填补内存中的null，不覆盖仓库已确认事实。"""

    section = config.get("arm_planning")
    if not isinstance(section, Mapping):
        raise ValueError("config.arm_planning必须是mapping")
    allowed = {field.name for field in fields(ArmPlanningConfig)}
    if set(section) != {"enabled", *allowed}:
        raise ValueError("config.arm_planning字段必须与ArmPlanningConfig严格一致")
    base = ArmPlanningConfig(**{name: section[name] for name in allowed})
    unknown = sorted(set(trial) - allowed)
    if unknown:
        raise ValueError(f"trial_parameters含未知字段：{unknown}")
    values = {field.name: getattr(base, field.name) for field in fields(base)}
    for name, value in trial.items():
        if values[name] is not None and values[name] != value:
            raise ValueError(f"trial_parameters不得覆盖已配置事实 ArmPlanningConfig.{name}")
        values[name] = value
    return ArmPlanningConfig(**values)


def stage2_fixture(config: Mapping[str, Any], scene: str, class_id: str) -> dict[str, Any]:
    """从仓库冻结配置读取场景几何，不在脚本复制坐标、尺寸或yaw。"""

    if scene not in SCENE_FIXTURE_KEYS:
        raise ValueError(f"scene必须是{tuple(SCENE_FIXTURE_KEYS)}之一")
    slot_name, index = SCENE_FIXTURE_KEYS[scene]
    source_slots = config["source_slots"]
    slot = source_slots["slots"][slot_name]
    centers = slot["centers"]
    if index >= len(centers):
        raise ValueError(f"仓库source_slots缺少{scene}冻结中心")
    sizes = config["perception"]["estimator_3d"]["object_local_size_xyz_m"]
    if class_id not in sizes:
        raise ValueError(f"仓库缺少class_id={class_id!r}的冻结局部XYZ尺寸")
    yaw = float(slot["yaw_rad"])
    if not math.isfinite(yaw):
        raise ValueError("仓库冻结source-slot yaw不是有限值")
    return {
        "source": FIXTURE_SOURCE,
        "scene": scene,
        "position_xyz": [float(value) for value in centers[index]],
        "size_xyz_m": [float(value) for value in sizes[class_id]],
        "orientation_xyzw": [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
        "frame_id": str(source_slots["frame_id"]),
        "source_slot": slot_name,
    }


def validate_stage2_fixture(payload: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if payload.get("source") != FIXTURE_SOURCE:
        raise ValueError(f"输入source必须严格为{FIXTURE_SOURCE}")
    scene = payload.get("scene")
    estimate = payload.get("object_estimate")
    if not isinstance(estimate, Mapping):
        raise ValueError("输入缺少object_estimate")
    expected = stage2_fixture(config, str(scene), str(estimate.get("class_id", "")))
    checks = {
        "position_xyz": estimate.get("position_xyz"),
        "size_xyz_m": estimate.get("size_xyz_m"),
        "orientation_xyzw": estimate.get("orientation_xyzw"),
        "frame_id": estimate.get("frame_id"),
        "source_slot": payload.get("source_slot"),
    }
    for name, actual in checks.items():
        expected_value = expected[name]
        if isinstance(expected_value, list):
            try:
                actual_values = tuple(float(value) for value in actual)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"阶段2 fixture {name}格式无效") from exc
            if not all(math.isfinite(value) for value in actual_values):
                raise ValueError(f"阶段2 fixture {name}必须全部为有限值")
            if name == "size_xyz_m" and any(value <= 0.0 for value in actual_values):
                raise ValueError("阶段2 fixture size_xyz_m三轴必须均大于 0")
            if len(actual_values) != len(expected_value) or any(
                not math.isclose(value, frozen, rel_tol=0.0, abs_tol=1e-12)
                for value, frozen in zip(actual_values, expected_value)
            ):
                raise ValueError(f"阶段2 fixture {name}与当前仓库冻结值不一致")
        elif actual != expected_value:
            raise ValueError(f"阶段2 fixture {name}与当前仓库冻结值不一致")


def _pose(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "position_xyz_m": list(value.position_xyz),
        "orientation_xyzw": list(value.orientation_xyzw),
        "frame_id": value.frame_id,
    }


def _trajectory_report(trajectory: Any, actual: RobotJointState, config: Mapping[str, Any]) -> dict[str, Any]:
    lower = tuple(config["action_mux"]["joint_lower"])
    upper = tuple(config["action_mux"]["joint_upper"])
    previous = tuple(actual.position)
    previous_time = 0.0
    waypoints: list[dict[str, Any]] = []
    bounds_ok = True
    for index, waypoint in enumerate(trajectory.waypoints):
        position = tuple(waypoint.joint_position)
        margins = [
            min(position[i] - lower[i], upper[i] - position[i])
            if waypoint.controlled_mask[i] else None
            for i in range(17)
        ]
        bounds_ok = bounds_ok and all(item is None or item >= 0.0 for item in margins)
        waypoints.append({
            "index": index,
            "stage": waypoint.phase.value,
            "time_from_start_s": waypoint.time_from_start_s,
            "stage_duration_s": waypoint.time_from_start_s - previous_time,
            "slide_m": position[0],
            "left_joint_angles_rad": list(position[3:9]),
            "right_joint_angles_rad": list(position[10:16]),
            "left_gripper": position[9],
            "right_gripper": position[16],
            "joint_position": list(position),
            "controlled_mask": list(waypoint.controlled_mask),
            "limit_margin_by_joint": dict(zip(JOINT_NAMES, margins)),
            "delta_from_previous_by_joint": dict(
                zip(JOINT_NAMES, (position[i] - previous[i] for i in range(17)))
            ),
        })
        previous, previous_time = position, waypoint.time_from_start_s
    return {"waypoints": waypoints, "joint_limits_ok": bounds_ok}


def _invalid(kind: str, reason: str) -> dict[str, Any]:
    return {
        "schema": COMMAND_SCHEMA,
        "command": kind,
        "mode": "plan-only",
        "valid": False,
        "published_control": False,
        "ik_success": False,
        "failure_reason": reason,
    }


COORDINATE_DIAGNOSTIC_FIELDS = (
    "raw_base_transform",
    "planarized_virtual_footprint_transform",
    "raw_roll_pitch_yaw",
    "virtual_footprint_roll_pitch_yaw",
    "object_pose_in_odom",
    "object_pose_in_virtual_footprint",
    "object_local_z_in_virtual_footprint",
    "planarization_scope",
)


def _coordinate_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    transforms = payload.get("transforms", {})
    raw = transforms.get("coordinate_diagnostics", {}) if isinstance(transforms, Mapping) else {}
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        **{name: _jsonable(raw.get(name)) for name in COORDINATE_DIAGNOSTIC_FIELDS},
        "published_control": False,
    }


class _RawKdlSolverProbe:
    """只在标定诊断调用期间代理官方solver并记录原始返回元数据。"""

    def __init__(self, solver: Any) -> None:
        self._solver = solver
        self.last: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._solver, name)

    def inverse_kinematics(self, **kwargs: Any) -> Any:
        raw = self._solver.inverse_kinematics(**kwargs)
        try:
            length: int | None = len(raw)
        except TypeError:
            length = None
        self.last = {
            "type": type(raw).__name__,
            "length": length,
        }
        return raw


def _solve_ik_with_raw_metadata(adapter: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """调用真实adapter；OfficialKDLAdapter可额外观察其solver原始返回。"""

    original_solver = getattr(adapter, "_solver", None)
    probe = None if original_solver is None else _RawKdlSolverProbe(original_solver)
    if probe is not None:
        adapter._solver = probe
    try:
        result = adapter.solve_ik(**kwargs)
    finally:
        if probe is not None:
            adapter._solver = original_solver
    return result, (
        probe.last if probe is not None and probe.last is not None
        else {"type": "NOT_EXPOSED_BY_ADAPTER", "length": None}
    )


def _calibration_joint_bounds(
    adapter: Any, config: Mapping[str, Any]
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, Any]]:
    """取团队安全边界与官方KDL公开边界的交集，不推测周期关节。"""

    lower = [float(value) for value in config["action_mux"]["joint_lower"]]
    upper = [float(value) for value in config["action_mux"]["joint_upper"]]
    evidence: dict[str, Any] = {
        "team_joint_order": list(JOINT_NAMES),
        "official_kinematic_order": [
            "slide_joint",
            *JOINT_NAMES[3:9],
            *JOINT_NAMES[10:16],
        ],
        "left_team_indices": list(range(3, 9)),
        "right_team_indices": list(range(10, 16)),
        "units": {"slide_joint": "m", "arm_joints": "rad"},
        "official_limits_exposed": False,
        "periodic_equivalent_adjustment_by_calibration_tool": False,
    }
    solver = getattr(adapter, "_solver", None)
    if solver is None:
        return tuple(lower), tuple(upper), evidence
    try:
        slide_limits = tuple(float(value) for value in solver.spine.joint_limits)
        left_limits = tuple(
            tuple(float(value) for value in pair)
            for pair in solver.left_arm.dh.joints_limit
        )
        right_limits = tuple(
            tuple(float(value) for value in pair)
            for pair in solver.right_arm.dh.joints_limit
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        evidence["official_limit_error"] = str(exc)
        return tuple(lower), tuple(upper), evidence
    if len(slide_limits) != 2 or len(left_limits) != 6 or len(right_limits) != 6:
        evidence["official_limit_error"] = "官方KDL公开限位维度不符合1+6+6"
        return tuple(lower), tuple(upper), evidence
    for index, pair in [(0, slide_limits), *zip(range(3, 9), left_limits),
                        *zip(range(10, 16), right_limits)]:
        lower[index] = max(lower[index], pair[0])
        upper[index] = min(upper[index], pair[1])
        if lower[index] > upper[index]:
            raise ValueError(f"官方KDL与团队安全边界无交集：{JOINT_NAMES[index]}")
    evidence.update({
        "official_limits_exposed": True,
        "official_slide_limits_m": list(slide_limits),
        "official_left_arm_limits_rad": [list(pair) for pair in left_limits],
        "official_right_arm_limits_rad": [list(pair) for pair in right_limits],
        "effective_lower": list(lower),
        "effective_upper": list(upper),
    })
    return tuple(lower), tuple(upper), evidence


def _seeded_joint_state(
    source: RobotJointState, slide: float,
    left: Sequence[float], right: Sequence[float],
) -> RobotJointState:
    position = list(source.position)
    position[0] = float(slide)
    position[3:9] = tuple(float(value) for value in left)
    position[10:16] = tuple(float(value) for value in right)
    return replace(source, position=tuple(position))


def _ik_result_vector(result: IKResult) -> tuple[float, ...]:
    if not result.success or result.left_joint_target is None or result.right_joint_target is None:
        raise ValueError(result.failure_reason or "双臂IK候选不完整")
    return (
        float(result.target_slide),
        *tuple(float(value) for value in result.left_joint_target),
        *tuple(float(value) for value in result.right_joint_target),
    )


def _official_raw_dual_candidates(
    adapter: Any, seed: RobotJointState, left_target: Any,
    right_target: Any, slide: float,
) -> tuple[list[IKResult], dict[str, Any]] | None:
    """按已核实的MMK2Kdl API同时取得seed解与全部解析解。"""

    solver = getattr(adapter, "_solver", None)
    inverse = getattr(solver, "inverse_kinematics", None)
    if not callable(inverse):
        return None
    parameters = tuple(inspect.signature(inverse).parameters)
    required = {"T_left", "T_right", "ref_pos", "target_height"}
    if not required <= set(parameters):
        raise ValueError(f"官方MMK2Kdl.inverse_kinematics API不匹配：{parameters}")
    import numpy as np

    left_matrix = _arm_planning._pose_to_matrix(left_target, np)
    right_matrix = _arm_planning._pose_to_matrix(right_target, np)
    reference = np.asarray(
        [slide, *seed.position[3:9], *seed.position[10:16]], dtype=float
    )
    seeded_raw = inverse(
        T_left=left_matrix, T_right=right_matrix,
        ref_pos=reference, target_height=slide,
    )
    analytic_enumeration_error = ""
    try:
        all_raw = inverse(
            T_left=left_matrix, T_right=right_matrix,
            ref_pos=None, target_height=slide,
        )
    except (AssertionError, TypeError, ValueError) as exc:
        # 当前官方双臂固定高度入口在ref_pos=None时会下标访问None；这表示公开入口
        # 没有暴露全部组合，不能在标定工具内复制官方肩部变换或猜测seed来绕过。
        all_raw = None
        analytic_enumeration_error = f"{type(exc).__name__}: {exc}"
    candidates: list[IKResult] = []
    sources: list[str] = []
    for source, raw in (("official_seeded_ref_pos", seeded_raw),
                        ("official_all_analytic_solutions", all_raw)):
        if raw is None:
            continue
        for item in raw:
            values = tuple(float(value) for value in item)
            if len(values) != 13 or not all(math.isfinite(value) for value in values):
                continue
            if not math.isclose(values[0], slide, rel_tol=0.0, abs_tol=1e-9):
                continue
            result = IKResult(values[0], values[1:7], values[7:13], True)
            vector = _ik_result_vector(result)
            if any(_ik_result_vector(existing) == vector for existing in candidates):
                continue
            candidates.append(result)
            sources.append(source)
    return candidates, {
        "api": "MMK2Kdl.inverse_kinematics(T_left,T_right,ref_pos,target_height)",
        "signature_parameters": list(parameters),
        "seed_supported": True,
        "seed_source": "RobotJointState/previous_stage_ik",
        "seed_ref_pos": list(reference),
        "seeded_return_count": 0 if seeded_raw is None else len(seeded_raw),
        "analytic_return_count": 0 if all_raw is None else len(all_raw),
        "analytic_enumeration_error": analytic_enumeration_error,
        "multiple_solution_selection_available": bool(
            all_raw is not None and len(all_raw) > 1
        ),
        "deduplicated_candidate_count": len(candidates),
        "candidate_sources": sources,
    }


def _calibration_ik_candidates(
    adapter: Any, seed: RobotJointState, left_target: Any,
    right_target: Any, slide: float,
) -> tuple[list[IKResult], dict[str, Any]]:
    raw = _official_raw_dual_candidates(
        adapter, seed, left_target, right_target, slide
    )
    if raw is not None:
        return raw
    provider = getattr(adapter, "solve_ik_candidates", None)
    if callable(provider):
        values = provider(
            actual_joints=seed, left_target=left_target,
            right_target=right_target, target_slide=slide,
        )
        candidates = [item for item in values if isinstance(item, IKResult) and item.success]
        return candidates, {
            "api": "test_or_calibration_adapter.solve_ik_candidates",
            "seed_supported": True,
            "seed_source": "RobotJointState/previous_stage_ik",
            "deduplicated_candidate_count": len(candidates),
        }
    result = adapter.solve_ik(
        actual_joints=seed, left_target=left_target,
        right_target=right_target, target_slide=slide,
    )
    candidates = [result] if isinstance(result, IKResult) and result.success else []
    return candidates, {
        "api": "adapter.solve_ik(actual_joints=seed,...)",
        "seed_supported": True,
        "seed_source": "RobotJointState/previous_stage_ik",
        "deduplicated_candidate_count": len(candidates),
        "failure_reason": (
            result.failure_reason if isinstance(result, IKResult) and not result.success
            else ""
        ),
    }


def _select_continuous_ik_candidate(
    candidates: Sequence[IKResult], seed: RobotJointState,
    lower: Sequence[float], upper: Sequence[float], metadata: Mapping[str, Any],
) -> tuple[IKResult | None, dict[str, Any]]:
    ranked: list[tuple[float, float, tuple[float, ...], int, IKResult]] = []
    rejected = []
    seed_arms = (*seed.position[3:9], *seed.position[10:16])
    for ordinal, result in enumerate(candidates):
        try:
            vector = _ik_result_vector(result)
        except ValueError as exc:
            rejected.append({"candidate_index": ordinal, "reason": str(exc)})
            continue
        joint_indices = (0, *range(3, 9), *range(10, 16))
        if any(not lower[index] <= value <= upper[index]
               for index, value in zip(joint_indices, vector)):
            rejected.append({
                "candidate_index": ordinal,
                "reason": "candidate_outside_effective_joint_limits",
                "joint_vector": list(vector),
            })
            continue
        arms = vector[1:]
        weighted_change = sum(
            weight * abs(value - previous)
            for weight, value, previous in zip(
                CALIBRATION_JOINT_WEIGHTS, arms, seed_arms
            )
        )
        margin = min(
            min(value - lower[index], upper[index] - value)
            for index, value in zip(joint_indices, vector)
        )
        ranked.append((weighted_change, -margin, vector, ordinal, result))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = None if not ranked else ranked[0][4]
    report = {
        **dict(metadata),
        "candidate_count": len(candidates),
        "legal_candidate_count": len(ranked),
        "rejected_candidates": rejected,
        "selection_policy": (
            "within_effective_limits_then_min_weighted_absolute_change_"
            "then_larger_minimum_limit_margin"
        ),
        "joint_weights": list(CALIBRATION_JOINT_WEIGHTS),
        "periodic_equivalent_adjustment_by_calibration_tool": False,
        "selected_weighted_change": None if not ranked else ranked[0][0],
        "selected_min_joint_limit_margin": None if not ranked else -ranked[0][1],
        "selected_joint_vector": None if not ranked else list(ranked[0][2]),
        "selected_candidate_index": None if not ranked else ranked[0][3],
        "selected_candidate_source": (
            None if not ranked else (
                metadata.get("candidate_sources", [])[ranked[0][3]]
                if ranked[0][3] < len(metadata.get("candidate_sources", []))
                else metadata.get("api")
            )
        ),
    }
    return selected, report


def _pick_waypoints_from_ik(
    results: Sequence[IKResult], actual: RobotJointState,
    planning: ArmPlanningConfig,
) -> tuple[Any, ...]:
    if len(results) != 4:
        raise ValueError("抓取IK序列必须包含PREGRASP/GRASP/LIFT/RETREAT四项")
    pregrasp = float(planning.pregrasp_duration_s)
    half_grasp = float(planning.grasp_duration_s) / 2.0
    approach = pregrasp + half_grasp
    close = approach + half_grasp
    lift = close + float(planning.lift_duration_s)
    retreat = lift + float(planning.retreat_duration_s)
    actual_position = tuple(actual.position)
    return (
        _arm_planning._joint_waypoint(
            _arm_planning.ArmMotionPhase.PREGRASP, pregrasp, results[0], actual_position,
            float(planning.left_gripper_open), float(planning.right_gripper_open),
        ),
        _arm_planning._joint_waypoint(
            _arm_planning.ArmMotionPhase.GRASP, approach, results[1], actual_position,
            float(planning.left_gripper_open), float(planning.right_gripper_open),
        ),
        _arm_planning._joint_waypoint(
            _arm_planning.ArmMotionPhase.GRASP, close, results[1], actual_position,
            float(planning.left_gripper_closed), float(planning.right_gripper_closed),
        ),
        _arm_planning._joint_waypoint(
            _arm_planning.ArmMotionPhase.LIFT, lift, results[2], actual_position,
            float(planning.left_gripper_closed), float(planning.right_gripper_closed),
        ),
        _arm_planning._joint_waypoint(
            _arm_planning.ArmMotionPhase.RETREAT, retreat, results[3], actual_position,
            float(planning.left_gripper_closed), float(planning.right_gripper_closed),
        ),
    )


def _continuity_diagnostics(
    actual: RobotJointState, waypoints: Sequence[Any],
    stage_seed_reports: Sequence[Mapping[str, Any]], arm_limit_rad: float,
) -> dict[str, Any]:
    previous = tuple(actual.position)
    previous_time = 0.0
    checks = []
    violations = []
    stage_seed_indices = (0, 1, 1, 2, 3)
    for waypoint_index, waypoint in enumerate(waypoints):
        duration = float(waypoint.time_from_start_s) - previous_time
        seed_report = dict(stage_seed_reports[stage_seed_indices[waypoint_index]])
        for arm, indices in (("left", range(3, 9)), ("right", range(10, 16))):
            for index in indices:
                signed = float(waypoint.joint_position[index]) - previous[index]
                absolute = abs(signed)
                item = {
                    "waypoint_index": waypoint_index,
                    "stage": (
                        "grasp-open" if waypoint_index == 1 else
                        "grasp-close" if waypoint_index == 2 else
                        "short-lift" if waypoint_index == 3 else
                        waypoint.phase.value.lower()
                    ),
                    "arm": arm,
                    "joint_index_in_arm": index - (3 if arm == "left" else 10),
                    "team_joint_index": index,
                    "joint_name": JOINT_NAMES[index],
                    "current_joint_value_rad": previous[index],
                    "target_joint_value_rad": float(waypoint.joint_position[index]),
                    "signed_delta_rad": signed,
                    "absolute_delta_rad": absolute,
                    "continuity_limit_rad": arm_limit_rad,
                    "minimum_required_time_s_at_0_6_rad_s": (
                        absolute / CALIBRATION_ARM_SPEED_LIMIT_RAD_S
                    ),
                    "stage_duration_s": duration,
                    "satisfies_speed_limit": (
                        absolute <= CALIBRATION_ARM_SPEED_LIMIT_RAD_S * duration + 1e-12
                    ),
                    "continuity_limit_satisfied": absolute <= arm_limit_rad + 1e-12,
                    "ik_branch_and_seed": seed_report,
                }
                checks.append(item)
                if not item["continuity_limit_satisfied"]:
                    violations.append(item)
        previous = tuple(waypoint.joint_position)
        previous_time = float(waypoint.time_from_start_s)
    return {
        "available": True,
        "arm_speed_limit_rad_s": CALIBRATION_ARM_SPEED_LIMIT_RAD_S,
        "arm_continuity_limit_rad": arm_limit_rad,
        "checks": checks,
        "violations": violations,
        "continuous_ik_branch_exists": not violations,
        "max_arm_delta_rad": max(
            (item["absolute_delta_rad"] for item in checks), default=None
        ),
        "all_stage_durations_satisfy_speed_limit": all(
            item["satisfies_speed_limit"] for item in checks
        ),
        "published_control": False,
    }


def _fk_transition_check(adapter: Any, position: Sequence[float]) -> dict[str, Any]:
    q = [float(position[0]), *map(float, position[3:9]), *map(float, position[10:16])]
    solver = getattr(adapter, "_solver", None)
    forward = getattr(solver, "forward_kinematics", None)
    source = "official_MMK2Kdl.forward_kinematics"
    if not callable(forward):
        forward = getattr(adapter, "forward_kinematics", None)
        source = "injected_adapter.forward_kinematics"
    if not callable(forward):
        return {"success": False, "source": None, "failure_reason": "FK API不可用"}
    try:
        left, right = forward(q)
        left_rows = tuple(tuple(float(value) for value in row) for row in left)
        right_rows = tuple(tuple(float(value) for value in row) for row in right)
        if len(left_rows) != 4 or len(right_rows) != 4 or any(
            len(row) != 4 for row in (*left_rows, *right_rows)
        ):
            raise ValueError("FK结果不是两个4x4矩阵")
        if not all(math.isfinite(value) for row in (*left_rows, *right_rows) for value in row):
            raise ValueError("FK结果包含非有限值")
        return {
            "success": True, "source": source,
            "left_end_position_xyz_m": [left_rows[i][3] for i in range(3)],
            "right_end_position_xyz_m": [right_rows[i][3] for i in range(3)],
        }
    except Exception as exc:  # noqa: BLE001 - 诊断边界完整保留FK错误
        return {"success": False, "source": source, "failure_reason": str(exc)}


def _calibration_transition_plan(
    adapter: Any, actual: RobotJointState, pregrasp: Any,
    lower: Sequence[float], upper: Sequence[float],
    continuity: Mapping[str, Any],
) -> dict[str, Any] | None:
    violations = list(continuity.get("violations", []))
    if not violations or any(item["waypoint_index"] != 0 for item in violations):
        return None
    start = tuple(actual.position)
    target = list(pregrasp.joint_position)
    target[9] = 1.0
    target[16] = 1.0
    arm_indices = (*range(3, 9), *range(10, 16))
    max_arm_delta = max(abs(target[index] - start[index]) for index in arm_indices)
    slide_delta = abs(target[0] - start[0])
    segments = max(
        1,
        math.ceil(max_arm_delta / CALIBRATION_ARM_STEP_LIMIT_RAD),
        math.ceil(slide_delta / CALIBRATION_SLIDE_STEP_LIMIT_M),
    )
    previous = start
    elapsed = 0.0
    reports = []
    all_limits = True
    all_steps = True
    all_speeds = True
    all_fk = True
    for ordinal in range(1, segments + 1):
        ratio = ordinal / segments
        position = list(start)
        position[0] = start[0] + ratio * (target[0] - start[0])
        for index in arm_indices:
            position[index] = start[index] + ratio * (target[index] - start[index])
        position[9] = position[16] = 1.0
        arm_deltas = {JOINT_NAMES[index]: position[index] - previous[index] for index in arm_indices}
        gripper_deltas = {
            JOINT_NAMES[index]: position[index] - previous[index] for index in (9, 16)
        }
        segment_arm_max = max(abs(value) for value in arm_deltas.values())
        segment_slide = position[0] - previous[0]
        segment_gripper_max = max(abs(value) for value in gripper_deltas.values())
        duration = max(
            segment_arm_max / CALIBRATION_ARM_SPEED_LIMIT_RAD_S,
            abs(segment_slide) / CALIBRATION_SLIDE_SPEED_LIMIT_M_S,
            segment_gripper_max / CALIBRATION_ARM_SPEED_LIMIT_RAD_S,
        )
        elapsed += duration
        margins = {
            JOINT_NAMES[index]: min(position[index] - lower[index], upper[index] - position[index])
            for index in (0, *arm_indices)
        }
        all_margins = {
            JOINT_NAMES[index]: min(
                position[index] - lower[index], upper[index] - position[index]
            )
            for index in range(17)
        }
        limits_ok = all(value >= -1e-12 for value in all_margins.values())
        step_ok = (
            segment_arm_max <= CALIBRATION_ARM_STEP_LIMIT_RAD + 1e-12
            and abs(segment_slide) <= CALIBRATION_SLIDE_STEP_LIMIT_M + 1e-12
            and segment_gripper_max <= 1.0 + 1e-12
        )
        speed_ok = (
            segment_arm_max <= CALIBRATION_ARM_SPEED_LIMIT_RAD_S * duration + 1e-12
            and abs(segment_slide) <= CALIBRATION_SLIDE_SPEED_LIMIT_M_S * duration + 1e-12
            and segment_gripper_max <= CALIBRATION_ARM_SPEED_LIMIT_RAD_S * duration + 1e-12
        )
        fk = _fk_transition_check(adapter, position)
        joint_deltas = {
            JOINT_NAMES[index]: position[index] - previous[index] for index in range(17)
        }
        reports.append({
            "segment_index": ordinal - 1,
            "stage": f"transition-{ordinal}",
            "time_from_start_s": elapsed,
            "duration_s": duration,
            "joint_position": list(position),
            "left_gripper": position[9], "right_gripper": position[16],
            "slide_delta_m": segment_slide,
            "arm_delta_by_joint_rad": arm_deltas,
            "gripper_delta_by_joint": gripper_deltas,
            "joint_delta_by_joint": joint_deltas,
            "velocity_by_joint_per_s": {
                name: (value / duration if duration > 0.0 else 0.0)
                for name, value in joint_deltas.items()
            },
            "max_arm_delta_rad": segment_arm_max,
            "limit_margin_by_joint": margins,
            "all_joint_limit_margin_by_joint": all_margins,
            "joint_limits_ok": limits_ok,
            "step_limits_ok": step_ok,
            "speed_limits_ok": speed_ok,
            "fk": fk,
        })
        all_limits = all_limits and limits_ok
        all_steps = all_steps and step_ok
        all_speeds = all_speeds and speed_ok
        all_fk = all_fk and fk["success"]
        previous = tuple(position)
    return {
        "mode": "calibration-only-plan",
        "published_control": False,
        "executable": False,
        "interpolation_is_collision_safe": False,
        "collision_check_available": False,
        "collision_visual_verification_required": False,
        "manual_simulation_observation_recommended": True,
        "status": "NOT_AUTOMATICALLY_CHECKED",
        "segment_count": segments,
        "required_segment_formula": "max(ceil(max_arm_delta/1.0),ceil(slide_delta/0.20))",
        "max_arm_step_limit_rad": CALIBRATION_ARM_STEP_LIMIT_RAD,
        "max_slide_step_limit_m": CALIBRATION_SLIDE_STEP_LIMIT_M,
        "arm_speed_limit_rad_s": CALIBRATION_ARM_SPEED_LIMIT_RAD_S,
        "gripper_speed_limit_per_s": CALIBRATION_ARM_SPEED_LIMIT_RAD_S,
        "slide_speed_limit_m_s": CALIBRATION_SLIDE_SPEED_LIMIT_M_S,
        "total_duration_s": elapsed,
        "max_single_segment_arm_delta_rad": max(
            item["max_arm_delta_rad"] for item in reports
        ),
        "minimum_joint_limit_margin": min(
            value for item in reports for value in item["limit_margin_by_joint"].values()
        ),
        "all_joint_limits_ok": all_limits,
        "all_step_limits_ok": all_steps,
        "all_speed_limits_ok": all_speeds,
        "all_fk_checks_ok": all_fk,
        "segments": reports,
    }


def _calibration_endpoint_waypoint_reports(
    adapter: Any, waypoints: Sequence[Any], transition: Mapping[str, Any],
    lower: Sequence[float], upper: Sequence[float],
) -> list[dict[str, Any]]:
    """Place endpoint waypoints after the one shared transition-plan timeline."""

    segments = transition.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("过渡计划缺少segments")
    previous = tuple(float(value) for value in segments[-1]["joint_position"])
    elapsed = float(transition["total_duration_s"])
    previous_original_time = float(waypoints[0].time_from_start_s)
    reports: list[dict[str, Any]] = []
    for index, waypoint in enumerate(waypoints):
        position = tuple(float(value) for value in waypoint.joint_position)
        duration = (
            0.0 if index == 0 else
            float(waypoint.time_from_start_s) - previous_original_time
        )
        if duration < 0.0:
            raise ValueError("端点路点时间倒退")
        elapsed += duration
        deltas = {
            JOINT_NAMES[joint_index]: position[joint_index] - previous[joint_index]
            for joint_index in range(17)
        }
        arm_max = max(
            abs(deltas[JOINT_NAMES[joint_index]])
            for joint_index in (*range(3, 9), *range(10, 16))
        )
        slide_delta = deltas[JOINT_NAMES[0]]
        gripper_max = max(abs(deltas[JOINT_NAMES[j]]) for j in (9, 16))
        margins = {
            JOINT_NAMES[joint_index]: min(
                position[joint_index] - lower[joint_index],
                upper[joint_index] - position[joint_index],
            )
            for joint_index in (0, *range(3, 9), *range(10, 16))
        }
        all_margins = {
            JOINT_NAMES[joint_index]: min(
                position[joint_index] - lower[joint_index],
                upper[joint_index] - position[joint_index],
            )
            for joint_index in range(17)
        }
        limits_ok = all(value >= -1e-12 for value in all_margins.values())
        step_ok = (
            arm_max <= CALIBRATION_ARM_STEP_LIMIT_RAD + 1e-12
            and abs(slide_delta) <= CALIBRATION_SLIDE_STEP_LIMIT_M + 1e-12
            and gripper_max <= 1.0 + 1e-12
        )
        speed_ok = (
            (duration == 0.0 and arm_max <= 1e-12 and abs(slide_delta) <= 1e-12
             and gripper_max <= 1e-12)
            or (
                duration > 0.0
                and arm_max <= CALIBRATION_ARM_SPEED_LIMIT_RAD_S * duration + 1e-12
                and abs(slide_delta) <= CALIBRATION_SLIDE_SPEED_LIMIT_M_S * duration + 1e-12
                and gripper_max <= CALIBRATION_ARM_SPEED_LIMIT_RAD_S * duration + 1e-12
            )
        )
        stage = (
            "pregrasp" if index == 0 else
            "grasp-open" if index == 1 else
            "grasp-close" if index == 2 else
            "short-lift" if index == 3 else "retreat"
        )
        reports.append({
            "waypoint_index": index,
            "stage": stage,
            "time_from_start_s": elapsed,
            "duration_s": duration,
            "joint_position": list(position),
            "joint_delta_by_joint": deltas,
            "velocity_by_joint_per_s": {
                name: (value / duration if duration > 0.0 else 0.0)
                for name, value in deltas.items()
            },
            "slide_delta_m": slide_delta,
            "max_arm_delta_rad": arm_max,
            "left_gripper": position[9],
            "right_gripper": position[16],
            "limit_margin_by_joint": margins,
            "all_joint_limit_margin_by_joint": all_margins,
            "joint_limits_ok": limits_ok,
            "step_limits_ok": step_ok,
            "speed_limits_ok": speed_ok,
            "fk": _fk_transition_check(adapter, position),
        })
        previous = position
        previous_original_time = float(waypoint.time_from_start_s)
    return reports


def _calibration_pick_sequence_analysis(
    adapter: Any, actual: RobotJointState,
    pose_pairs: tuple[tuple[str, Any, Any], ...] | None,
    planning: ArmPlanningConfig, config: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "mode": "calibration-only-plan", "published_control": False,
        "continuous_ik_branch_exists": False,
        "transition_required": False, "transition_plan": None,
        "slide_candidates": [], "selected_slide_m": None,
    }
    if pose_pairs is None:
        return {**base, "available": False, "failure_reason": "抓取几何未进入IK"}
    lower, upper, order = _calibration_joint_bounds(adapter, config)
    successful = []
    for slide in _arm_planning._slide_search_candidates(
        adapter, actual.position[0], float(planning.max_slide_waypoint_delta_m)
    ):
        seed = actual
        results = []
        seed_reports = []
        failure = ""
        for stage_index, (stage, left, right) in enumerate(pose_pairs):
            candidates, metadata = _calibration_ik_candidates(
                adapter, seed, left, right, slide
            )
            selected, selection = _select_continuous_ik_candidate(
                candidates, seed, lower, upper, metadata
            )
            selection.update({
                "stage": stage,
                "stage_index": stage_index,
                "seed_kind": "current_joint_state" if stage_index == 0 else "previous_stage_ik",
                "seed_left_joint_values_rad": list(seed.position[3:9]),
                "seed_right_joint_values_rad": list(seed.position[10:16]),
            })
            seed_reports.append(selection)
            if selected is None:
                failure = selection.get("failure_reason") or f"{stage}没有限位内双臂IK候选"
                break
            results.append(selected)
            assert selected.left_joint_target is not None and selected.right_joint_target is not None
            seed = _seeded_joint_state(
                seed, selected.target_slide,
                selected.left_joint_target, selected.right_joint_target,
            )
        item: dict[str, Any] = {
            "slide_m": slide, "all_stages_ik_success": len(results) == len(pose_pairs),
            "failure_reason": failure, "stage_branch_selection": seed_reports,
        }
        if len(results) == len(pose_pairs):
            waypoints = _pick_waypoints_from_ik(results, actual, planning)
            fk_checks = [
                {
                    "waypoint_index": index,
                    "stage": (
                        "grasp-open" if index == 1 else
                        "grasp-close" if index == 2 else
                        "short-lift" if index == 3 else waypoint.phase.value.lower()
                    ),
                    **_fk_transition_check(adapter, waypoint.joint_position),
                }
                for index, waypoint in enumerate(waypoints)
            ]
            continuity = _continuity_diagnostics(
                actual, waypoints, seed_reports, float(planning.max_arm_waypoint_delta_rad)
            )
            transition = _calibration_transition_plan(
                adapter, actual, waypoints[0], lower, upper, continuity
            )
            endpoint_waypoints = (
                _calibration_endpoint_waypoint_reports(
                    adapter, waypoints, transition, lower, upper
                )
                if transition is not None else []
            )
            margins = [
                min(result.target_slide - lower[0], upper[0] - result.target_slide)
                for result in results
            ]
            for result in results:
                vector = _ik_result_vector(result)
                for index, value in zip((*range(3, 9), *range(10, 16)), vector[1:]):
                    margins.append(min(value - lower[index], upper[index] - value))
            item.update({
                "continuity_diagnostics": continuity,
                "continuous_ik_branch_exists": continuity["continuous_ik_branch_exists"],
                "transition_required": bool(continuity["violations"]),
                "transition_plan": transition,
                "endpoint_waypoints": endpoint_waypoints,
                "fk_checks": fk_checks,
                "all_fk_checks_ok": all(check["success"] for check in fk_checks),
                "collision_check_available": False,
                "collision_visual_verification_required": False,
                "manual_simulation_observation_recommended": True,
                "collision_verification_status": "NOT_AUTOMATICALLY_CHECKED",
                "minimum_joint_limit_margin": min(margins),
                "selected_joint_targets": [
                    {
                        "stage": pose_pairs[index][0],
                        "left_joint_target_rad": list(result.left_joint_target or ()),
                        "right_joint_target_rad": list(result.right_joint_target or ()),
                    }
                    for index, result in enumerate(results)
                ],
            })
            successful.append(item)
        base["slide_candidates"].append(item)
    if not successful:
        return {
            **base, "available": True, "joint_order_and_limits": order,
            "failure_reason": "所有slide候选均未形成完整逐段seed双臂IK序列",
        }
    successful.sort(key=lambda item: (
        not item["continuous_ik_branch_exists"],
        item["transition_plan"] is None,
        0 if item["transition_plan"] is None else item["transition_plan"]["segment_count"],
        -item["minimum_joint_limit_margin"],
    ))
    selected = successful[0]
    return {
        **base, "available": True, "failure_reason": "",
        "joint_order_and_limits": order,
        "continuous_ik_branch_exists": selected["continuous_ik_branch_exists"],
        "transition_required": selected["transition_required"],
        "transition_plan": selected["transition_plan"],
        "endpoint_waypoints": selected.get("endpoint_waypoints", []),
        "selected_slide_m": selected["slide_m"],
        "minimum_joint_limit_margin": selected["minimum_joint_limit_margin"],
        "continuity_diagnostics": selected["continuity_diagnostics"],
        "stage_branch_selection": selected["stage_branch_selection"],
        "selected_joint_targets": selected["selected_joint_targets"],
        "fk_checks": selected["fk_checks"],
        "all_fk_checks_ok": selected["all_fk_checks_ok"],
        "collision_check_available": False,
        "collision_visual_verification_required": False,
        "manual_simulation_observation_recommended": True,
        "collision_verification_status": "NOT_AUTOMATICALLY_CHECKED",
        "slide_candidates": base["slide_candidates"],
    }


def _ik_failure_category(reason: str, raw: Mapping[str, Any]) -> str:
    normalized = reason.lower()
    if raw.get("length") == 0 or "空" in reason or "未返回候选解" in reason:
        return "EMPTY_SOLUTION"
    if "长度" in reason or "length" in normalized:
        return "LENGTH_ERROR"
    if "有限" in reason or "nan" in normalized or "inf" in normalized:
        return "NON_FINITE"
    if "超出" in reason or "限位" in reason or "joint_limits" in normalized:
        return "JOINT_OUT_OF_LIMITS"
    if "未找到合法关节解" in reason or "无解" in reason or "ik失败" in normalized:
        return "NO_SOLUTION"
    if "返回值必须" in reason or "ikresult" in normalized:
        return "RETURN_TYPE_ERROR"
    return "OTHER_ERROR"


def _ik_attempt_report(
    adapter: Any,
    actual: RobotJointState,
    left_target: Any | None,
    right_target: Any | None,
    slide: float,
    arm_label: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    pose = left_target if arm_label == "left" else right_target
    target_pose = (
        {"left": _pose(left_target), "right": _pose(right_target)}
        if arm_label == "dual" else _pose(pose)
    )
    raw = {"type": "NOT_CALLED", "length": None}
    try:
        result, raw = _solve_ik_with_raw_metadata(
            adapter,
            actual_joints=actual,
            left_target=left_target if arm_label in {"left", "dual"} else None,
            right_target=right_target if arm_label in {"right", "dual"} else None,
            target_slide=slide,
        )
        if not isinstance(result, IKResult):
            return {
                "target_pose": target_pose, "success": False,
                "classification": "RETURN_TYPE_ERROR",
                "failure_reason": f"IK返回类型必须为IKResult，实际={type(result).__name__}",
                "kdl_raw_return": raw, "joint_target": None,
            }
        if not result.success:
            reason = result.failure_reason or "IK返回success=false且没有原因"
            return {
                "target_pose": target_pose, "success": False,
                "classification": _ik_failure_category(reason, raw),
                "failure_reason": reason, "kdl_raw_return": raw,
                "joint_target": None,
            }
        targets: list[tuple[str, tuple[float, ...] | None, range]] = []
        if arm_label in {"left", "dual"}:
            targets.append(("left", result.left_joint_target, range(3, 9)))
        if arm_label in {"right", "dual"}:
            targets.append(("right", result.right_joint_target, range(10, 16)))
        lower = tuple(float(value) for value in config["action_mux"]["joint_lower"])
        upper = tuple(float(value) for value in config["action_mux"]["joint_upper"])
        serialized: dict[str, list[float]] = {}
        margins: list[float] = []
        for name, target, indices in targets:
            if target is None:
                raise ValueError(f"{name}空关节解")
            if len(target) != 6:
                raise ValueError(f"{name}关节解长度错误：{len(target)}")
            values = tuple(float(value) for value in target)
            if not all(math.isfinite(value) for value in values):
                raise ArithmeticError(f"{name}关节解包含非有限值")
            for value, index in zip(values, indices):
                margin = min(value - lower[index], upper[index] - value)
                margins.append(margin)
                if margin < 0.0:
                    raise OverflowError(
                        f"{name}关节越界：{JOINT_NAMES[index]}={value},"
                        f"范围=[{lower[index]},{upper[index]}]"
                    )
            serialized[name] = list(values)
        return {
            "target_pose": target_pose, "success": True, "classification": "SUCCESS",
            "failure_reason": "", "kdl_raw_return": raw,
            "joint_target": serialized,
            "min_joint_limit_margin": min(margins) if margins else None,
        }
    except OverflowError as exc:
        category = "JOINT_OUT_OF_LIMITS"
        reason = str(exc)
    except ArithmeticError as exc:
        category = "NON_FINITE"
        reason = str(exc)
    except ValueError as exc:
        reason = str(exc)
        category = _ik_failure_category(reason, raw)
    except Exception as exc:  # noqa: BLE001 - 官方诊断边界必须完整记录
        category = "OTHER_ERROR"
        reason = f"{type(exc).__name__}: {exc}"
    return {
        "target_pose": target_pose, "success": False,
        "classification": category, "failure_reason": reason,
        "kdl_raw_return": raw, "joint_target": None,
    }


def _plan_grasp_with_pose_capture(
    planner: ArmPlanner,
    task: TaskSpec,
    target: ObjectEstimate3D,
    target_to_footprint: RigidTransform3D,
    target_to_world: RigidTransform3D,
    actual: RobotJointState,
    now_ns: int,
) -> tuple[Any, Any, tuple[tuple[str, Any, Any], ...] | None]:
    """让生产ArmPlanner生成Pose；标定层只捕获其真实IK入口参数。"""

    captured: dict[str, Any] = {}
    original = _arm_planning._solve_ik_sequence_with_slide_search

    def capture(adapter: Any, joints: RobotJointState, pose_pairs: Any,
                actual_slide: float, max_delta: float) -> Any:
        captured["pose_pairs"] = tuple(pose_pairs)
        return original(adapter, joints, pose_pairs, actual_slide, max_delta)

    _arm_planning._solve_ik_sequence_with_slide_search = capture
    try:
        grasp, trajectory = planner.plan_grasp(
            task, target, target_to_footprint, target_to_world, actual, now_ns
        )
    finally:
        _arm_planning._solve_ik_sequence_with_slide_search = original
    return grasp, trajectory, captured.get("pose_pairs")


def _detailed_pick_ik_diagnostics(
    adapter: Any,
    actual: RobotJointState,
    pose_pairs: tuple[tuple[str, Any, Any], ...] | None,
    planning: ArmPlanningConfig,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if pose_pairs is None:
        return {
            "available": False, "failure_reason": "抓取几何在进入IK前失败",
            "slide_candidates": [], "all_stages_dual_ik_success": False,
        }
    candidates = _arm_planning._slide_search_candidates(
        adapter, actual.position[0], planning.max_slide_waypoint_delta_m
    )
    pair_map = {label: (left, right) for label, left, right in pose_pairs}
    stage_map = (
        ("pregrasp", "PREGRASP"),
        ("grasp-open", "GRASP"),
        ("grasp-close", "GRASP"),
        ("short-lift", "LIFT"),
        ("retreat", "RETREAT"),
    )
    slide_reports = []
    for slide in candidates:
        stages = []
        for stage_name, pair_name in stage_map:
            left, right = pair_map[pair_name]
            arms = {
                "left": _ik_attempt_report(
                    adapter, actual, left, None, slide, "left", config
                ),
                "right": _ik_attempt_report(
                    adapter, actual, None, right, slide, "right", config
                ),
                "dual": _ik_attempt_report(
                    adapter, actual, left, right, slide, "dual", config
                ),
            }
            stages.append({
                "stage": stage_name,
                "left": arms["left"], "right": arms["right"],
                "dual": arms["dual"],
                "single_arm_solution_without_dual": (
                    (arms["left"]["success"] or arms["right"]["success"])
                    and not arms["dual"]["success"]
                ),
            })
        slide_reports.append({
            "slide_m": slide,
            "complete_dual_path_success": all(stage["dual"]["success"] for stage in stages),
            "stages": stages,
        })
    return {
        "available": True,
        "slide_candidates": slide_reports,
        "all_stages_dual_ik_success": any(
            item["complete_dual_path_success"] for item in slide_reports
        ),
    }


def _transition_artifact_readiness(
    analysis: Mapping[str, Any], production_failure_reason: str,
) -> tuple[bool, str]:
    transition = analysis.get("transition_plan")
    continuity = analysis.get("continuity_diagnostics")
    endpoints = analysis.get("endpoint_waypoints")
    if "路点0" not in production_failure_reason or "连续性" not in production_failure_reason:
        return False, "生产规划失败不是首个pregrasp连续性超限"
    if analysis.get("available") is not True:
        return False, "标定逐阶段IK分析不可用"
    if analysis.get("continuous_ik_branch_exists") is not False:
        return False, "存在连续IK分支，不应生成过渡计划"
    selected_targets = analysis.get("selected_joint_targets")
    if not isinstance(selected_targets, list) or len(selected_targets) != 4:
        return False, "没有完整PREGRASP/GRASP/LIFT/RETREAT双臂端点IK"
    if not isinstance(continuity, Mapping):
        return False, "缺少连续性诊断"
    if continuity.get("arm_continuity_limit_rad") != CALIBRATION_ARM_STEP_LIMIT_RAD:
        return False, "过渡工件只允许严格1.0rad连续性守卫"
    violations = continuity.get("violations")
    if (
        not isinstance(violations, list) or not violations
        or any(item.get("waypoint_index") != 0 for item in violations)
    ):
        return False, "连续性超限不只发生在首个pregrasp"
    if not isinstance(transition, Mapping):
        return False, "共享过渡规划器没有生成计划"
    required_transition_flags = (
        "all_joint_limits_ok", "all_step_limits_ok", "all_speed_limits_ok",
        "all_fk_checks_ok",
    )
    if any(transition.get(name) is not True for name in required_transition_flags):
        return False, "过渡计划未通过限位、步长、速度或FK检查"
    if not isinstance(endpoints, list) or len(endpoints) != 5:
        return False, "缺少五个完整抓取端点路点"
    if any(
        item.get("joint_limits_ok") is not True
        or item.get("step_limits_ok") is not True
        or item.get("speed_limits_ok") is not True
        or not isinstance(item.get("fk"), Mapping)
        or item["fk"].get("success") is not True
        for item in endpoints
    ):
        return False, "过渡后的抓取端点未通过限位、步长、速度或FK检查"
    required_time = max(
        float(item["absolute_delta_rad"]) / CALIBRATION_ARM_SPEED_LIMIT_RAD_S
        for item in violations
    )
    if float(transition.get("total_duration_s", -1.0)) + 1e-12 < required_time:
        return False, "过渡总时长不足以满足0.6rad/s速度限制"
    return True, ""


def plan_pick(
    payload: Mapping[str, Any], config: Mapping[str, Any], adapter: Any | None = None,
    *, include_full_ik_diagnostics: bool = False,
) -> dict[str, Any]:
    diagnostics = _coordinate_diagnostics(payload)
    try:
        validate_stage2_fixture(payload, config)
        expected_id = payload.get("expected_object_id")
        target = _estimate(payload["object_estimate"])
        if target.orientation_xyzw is None:
            raise ValueError("orientation缺失；禁止默认yaw=0")
        if expected_id is not None and target.object_id != expected_id:
            raise ValueError("object_id与expected_object_id不匹配")
        planning = planning_config(config, payload.get("trial_parameters", {}))
        if adapter is None:
            official = config["official"]
            adapter = OfficialKDLAdapter(official.get("root", ""), official["kdl_module"])
            adapter.self_check()
        planner = ArmPlanner(adapter, planning)
        task = _task(payload["task"])
        actual = _joints(payload["joint_state"])
        transforms = payload["transforms"]
        target_result, trajectory, pose_pairs = _plan_grasp_with_pose_capture(
            planner,
            task, target, _transform(transforms["target_to_footprint"]),
            _transform(transforms["target_to_world"]), actual, payload["now_ns"],
        )
        try:
            calibration_analysis = _calibration_pick_sequence_analysis(
                adapter, actual, pose_pairs, planning, config
            )
        except Exception as exc:  # noqa: BLE001 - 失败规划也必须保存诊断错误
            calibration_analysis = {
                "available": False,
                "mode": "calibration-only-plan",
                "published_control": False,
                "continuous_ik_branch_exists": False,
                "transition_required": False,
                "transition_plan": None,
                "failure_reason": f"连续性诊断失败：{type(exc).__name__}: {exc}",
            }
        if not target_result.valid or not trajectory.valid:
            production_failure = target_result.failure_reason or trajectory.failure_reason
            transition_ready, transition_blocker = _transition_artifact_readiness(
                calibration_analysis, production_failure
            )
            if transition_ready:
                transition = calibration_analysis["transition_plan"]
                endpoint_waypoints = calibration_analysis["endpoint_waypoints"]
                initial_fk = _fk_transition_check(adapter, actual.position)
                initial_waypoint = {
                    "stage": "initial-joint-state",
                    "time_from_start_s": 0.0,
                    "duration_s": 0.0,
                    "joint_position": list(actual.position),
                    "joint_delta_by_joint": {name: 0.0 for name in JOINT_NAMES},
                    "velocity_by_joint_per_s": {name: 0.0 for name in JOINT_NAMES},
                    "fk": initial_fk,
                }
                pose_map = {
                    label: {"left": _pose(left), "right": _pose(right)}
                    for label, left, right in (pose_pairs or ())
                }
                result = {
                    "schema": COMMAND_SCHEMA,
                    "execution_contract_version": EXECUTION_CONTRACT_VERSION,
                    "command": "plan-pick",
                    "mode": "calibration-only-plan",
                    "valid": True,
                    "plan_artifact_valid": True,
                    "automatic_execution_ready": False,
                    "single_stage_execution_ready": transition["segment_count"] == 3,
                    "visual_transition_review_ready": True,
                    "status": "TRANSITION_PLAN_READY_FOR_MANUAL_SINGLE_STAGE_SIMULATION",
                    "published_control": False,
                    "ik_success": True,
                    "endpoint_ik_success": True,
                    "joint_limits_ok": True,
                    "transition_required": True,
                    "transition_segment_count": transition["segment_count"],
                    "max_single_segment_joint_delta_rad": transition[
                        "max_single_segment_arm_delta_rad"
                    ],
                    "total_transition_duration_s": transition["total_duration_s"],
                    "minimum_joint_limit_margin": calibration_analysis[
                        "minimum_joint_limit_margin"
                    ] if transition.get("minimum_joint_limit_margin") is None else min(
                        calibration_analysis["minimum_joint_limit_margin"],
                        transition["minimum_joint_limit_margin"],
                    ),
                    "all_fk_checks_ok": (
                        transition["all_fk_checks_ok"] is True
                        and all(item["fk"]["success"] for item in endpoint_waypoints)
                    ),
                    "collision_verification_status": "NOT_AUTOMATICALLY_CHECKED",
                    "collision_verification_note": (
                        "未做自动碰撞检测；该信息不阻塞官方离线仿真手动单段执行"
                    ),
                    "collision_check_available": False,
                    "collision_visual_verification_required": False,
                    "manual_simulation_observation_recommended": True,
                    "blocking_reason": (
                        "不存在自动碰撞检测，因此不得自动连续执行；允许用户在官方离线"
                        "仿真中观察窗口并逐段手动执行"
                    ),
                    "failure_reason": "",
                    "production_planner_failure_reason": production_failure,
                    "trajectory_id": None,
                    "source": FIXTURE_SOURCE,
                    "seed": payload.get("seed"),
                    "scene": payload["scene"],
                    "source_slot": payload["source_slot"],
                    "task": _jsonable(task),
                    "object_estimate": _jsonable(target),
                    "trial_parameters": _jsonable(payload.get("trial_parameters", {})),
                    "start_joint_state": _jsonable(actual),
                    "planned_grasp_context": None,
                    "gripper_calibration": {
                        "open": 1.0, "closed": 0.1, "note": GRIPPER_NOTE
                    },
                    "poses": {
                        "pregrasp": pose_map.get("PREGRASP"),
                        "grasp/contact": pose_map.get("GRASP"),
                        "lift": pose_map.get("LIFT"),
                        "retreat": pose_map.get("RETREAT"),
                    },
                    "calibration_analysis": calibration_analysis,
                    "continuity_diagnostics": calibration_analysis[
                        "continuity_diagnostics"
                    ],
                    "transition_plan": transition,
                    "transition_3_reaches_pregrasp": (
                        transition["segment_count"] == 3
                        and transition["segments"][-1]["joint_position"]
                        == endpoint_waypoints[0]["joint_position"]
                    ),
                    "waypoints": endpoint_waypoints,
                    "calibration_waypoints": [
                        initial_waypoint,
                        *transition["segments"],
                        *endpoint_waypoints,
                    ],
                    **diagnostics,
                }
                if include_full_ik_diagnostics:
                    result["ik_diagnostics"] = _detailed_pick_ik_diagnostics(
                        adapter, actual, pose_pairs, planning, config
                    )
                return result
            result = _invalid("plan-pick", target_result.failure_reason or trajectory.failure_reason)
            result.update(diagnostics)
            result["calibration_analysis"] = calibration_analysis
            result["transition_artifact_blocker"] = transition_blocker
            result["continuity_diagnostics"] = calibration_analysis.get(
                "continuity_diagnostics", {
                    "available": False, "violations": [],
                    "published_control": False,
                    "failure_reason": calibration_analysis.get("failure_reason", ""),
                }
            )
            if "IK" in result["failure_reason"] or "slide候选" in result["failure_reason"]:
                result["status"] = "IK_NO_COMPLETE_PATH"
                result["ik_diagnostics"] = _detailed_pick_ik_diagnostics(
                    adapter, actual, pose_pairs, planning, config
                )
            return result
        report = _trajectory_report(trajectory, actual, config)
        valid = report["joint_limits_ok"]
        reason = "" if valid else "规划路点超出action_mux安全关节边界"
        result = {
            "schema": COMMAND_SCHEMA, "command": "plan-pick", "mode": "plan-only",
            "execution_contract_version": EXECUTION_CONTRACT_VERSION,
            "valid": valid, "published_control": False, "ik_success": valid,
            "failure_reason": reason, "trajectory_id": trajectory.trajectory_id,
            "source": FIXTURE_SOURCE, "seed": payload.get("seed"),
            "scene": payload["scene"], "source_slot": payload["source_slot"],
            "task": _jsonable(task), "object_estimate": _jsonable(target),
            "trial_parameters": _jsonable(payload.get("trial_parameters", {})),
            "start_joint_state": _jsonable(actual),
            "gripper_calibration": {"open": 1.0, "closed": 0.1, "note": GRIPPER_NOTE},
            "planned_grasp_context": _jsonable(target_result.grasp_context),
            "calibration_analysis": calibration_analysis,
            "continuity_diagnostics": calibration_analysis.get(
                "continuity_diagnostics", {"available": False, "violations": []}
            ),
            **diagnostics,
            "poses": {
                name: {"left": _pose(getattr(target_result, f"left_{attr}")),
                       "right": _pose(getattr(target_result, f"right_{attr}"))}
                for name, attr in (("pregrasp", "pregrasp"), ("grasp/contact", "grasp"),
                                   ("lift", "lift"), ("retreat", "retreat"))
            },
            **report,
        }
        if include_full_ik_diagnostics:
            result["ik_diagnostics"] = _detailed_pick_ik_diagnostics(
                adapter, actual, pose_pairs, planning, config
            )
        return result
    except Exception as exc:  # planning CLI boundary
        result = _invalid("plan-pick", str(exc))
        result.update(diagnostics)
        return result


def plan_place(payload: Mapping[str, Any], config: Mapping[str, Any], adapter: Any | None = None) -> dict[str, Any]:
    diagnostics = _coordinate_diagnostics(payload)
    try:
        if payload.get("source") != FIXTURE_SOURCE:
            raise ValueError(f"输入source必须严格为{FIXTURE_SOURCE}")
        if payload.get("scene") not in SCENE_FIXTURE_KEYS:
            raise ValueError(f"scene必须是{tuple(SCENE_FIXTURE_KEYS)}之一")
        planning = planning_config(config, payload.get("trial_parameters", {}))
        if adapter is None:
            official = config["official"]
            adapter = OfficialKDLAdapter(official.get("root", ""), official["kdl_module"])
            adapter.self_check()
        planner = ArmPlanner(adapter, planning)
        task = _task(payload["task"])
        actual = _joints(payload["joint_state"])
        context = _grasp_context(payload["grasp_context"])
        load_state = payload.get("load_state")
        calibration_override = ""
        if load_state == "empty":
            if context.confirmed:
                raise ValueError("空载测试必须使用未确认的planned GraspContext")
            context = replace(context, confirmed=True, confirmed_at_ns=payload["now_ns"])
            calibration_override = "EMPTY_LOAD_KINEMATIC_ONLY_NOT_GRASP_CONFIRMATION"
        elif load_state == "carrying_object":
            if not context.confirmed:
                raise ValueError("带物体测试必须提供真实验证后confirmed GraspContext")
        else:
            raise ValueError("load_state必须显式为empty或carrying_object")
        target_result, trajectory = planner.plan_place(
            task, _transform(payload["transforms"]["world_to_footprint"]),
            context, actual, payload["now_ns"],
        )
        if not target_result.valid or not trajectory.valid:
            result = _invalid("plan-place", target_result.failure_reason or trajectory.failure_reason)
            result.update(diagnostics)
            if "IK" in result["failure_reason"] or "slide候选" in result["failure_reason"]:
                result["status"] = "BASE_STAND_POSITION_REQUIRED"
            return result
        report = _trajectory_report(trajectory, actual, config)
        valid = report["joint_limits_ok"]
        reason = "" if valid else "规划路点超出action_mux安全关节边界"
        return {
            "schema": COMMAND_SCHEMA, "command": "plan-place", "mode": "plan-only",
            "execution_contract_version": EXECUTION_CONTRACT_VERSION,
            "valid": valid, "published_control": False, "ik_success": valid,
            "failure_reason": reason, "trajectory_id": trajectory.trajectory_id,
            "source": FIXTURE_SOURCE, "seed": payload.get("seed"),
            "scene": payload["scene"],
            "task": _jsonable(task), "grasp_context": _jsonable(context),
            "trial_parameters": _jsonable(payload.get("trial_parameters", {})),
            "load_state": load_state,
            "calibration_context_override": calibration_override,
            "start_joint_state": _jsonable(actual),
            "gripper_calibration": {"open": 1.0, "closed": 0.1, "note": GRIPPER_NOTE},
            **diagnostics,
            "poses": {
                "object_goal": _pose(target_result.object_goal_pose),
                "preplace": {"left": _pose(target_result.left_preplace), "right": _pose(target_result.right_preplace)},
                "release": {"left": _pose(target_result.left_release), "right": _pose(target_result.right_release)},
                "post_release_retreat": {"left": _pose(target_result.left_post_release_retreat),
                                             "right": _pose(target_result.right_post_release_retreat)},
            },
            "settle_time_s": target_result.settle_time_s,
            **report,
        }
    except Exception as exc:
        result = _invalid("plan-place", str(exc))
        result.update(diagnostics)
        return result


def _navigation_config_for_trial(
    config: Mapping[str, Any], standoff_m: float,
    position_tolerance_m: float, yaw_tolerance_rad: float,
) -> NavigationConfig:
    section = config.get("navigation")
    if not isinstance(section, Mapping):
        raise ValueError("config.navigation必须是mapping")
    expected = {field.name for field in fields(NavigationConfig)}
    if set(section) != expected:
        raise ValueError("config.navigation字段必须与NavigationConfig严格一致")
    if not math.isfinite(standoff_m) or standoff_m <= 0.0:
        raise ValueError("--standoff-m必须是有限正数")
    production = NavigationConfig(**dict(section))
    trials = {
        "--position-tolerance-m": (position_tolerance_m, production.position_tolerance_m),
        "--yaw-tolerance-rad": (yaw_tolerance_rad, production.yaw_tolerance_rad),
    }
    for name, (value, production_limit) in trials.items():
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0.0
        ):
            raise ValueError(f"{name}必须是有限正数")
        if float(value) > float(production_limit):
            raise ValueError(
                f"{name}={value}不得宽于生产导航容差{production_limit}"
            )
    return replace(
        production, standoff_m=float(standoff_m),
        position_tolerance_m=float(position_tolerance_m),
        yaw_tolerance_rad=float(yaw_tolerance_rad),
    )


def _base_state_from_odom(message: Any) -> BaseState:
    frame = normalize_ros_frame(message.header.frame_id)
    pose = message.pose.pose
    twist = message.twist.twist
    quaternion = _normalized_quaternion(
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    )
    yaw = _quaternion_to_rpy(quaternion)[2]
    stamp_ns = RosCalibrationRuntime._stamp_ns(message.header.stamp)
    return BaseState(
        position_xyz=(pose.position.x, pose.position.y, pose.position.z),
        orientation_xyzw=quaternion,
        yaw=yaw,
        linear_velocity_xyz=(twist.linear.x, twist.linear.y, twist.linear.z),
        angular_velocity_xyz=(twist.angular.x, twist.angular.y, twist.angular.z),
        frame_id=frame,
        timestamp_ns=stamp_ns,
    )


def plan_base_stand(
    payload: Mapping[str, Any], config: Mapping[str, Any], base: BaseState,
    standoff_m: float, position_tolerance_m: float, yaw_tolerance_rad: float,
) -> dict[str, Any]:
    """只规划候选站位；唯一几何入口是生产NavigationController.build_pick_goal。"""

    try:
        validate_stage2_fixture(payload, config)
        if not isinstance(base, BaseState) or not base.valid or base.frame_id != "odom":
            raise ValueError("plan-base-stand要求有效odom BaseState")
        navigation_config = _navigation_config_for_trial(
            config, standoff_m, position_tolerance_m, yaw_tolerance_rad
        )
        task = _task(payload["task"])
        target = _estimate(payload["object_estimate"])
        # stage2 fixture在本次实时Odom采样时实例化；几何不变，时间事实明确更新。
        target = replace(target, timestamp_ns=base.timestamp_ns)
        controller = NavigationController(navigation_config)
        goal = controller.build_pick_goal(task, target, base, base.timestamp_ns)
        dx = goal.pose_xyyaw[0] - base.position_xyz[0]
        dy = goal.pose_xyyaw[1] - base.position_xyz[1]
        bearing = math.atan2(dy, dx)
        parameters = _jsonable(navigation_config)
        return {
            "schema": COMMAND_SCHEMA,
            "command": "plan-base-stand",
            "mode": "plan-only",
            "valid": True,
            "status": "TRIAL_NOT_FROZEN",
            "published_control": False,
            "seed": payload.get("seed"),
            "scene": payload.get("scene"),
            "task": _jsonable(task),
            "target_object": _jsonable(target),
            "current_base_state": _jsonable(base),
            "goal": _jsonable(goal),
            "travel_distance_m": math.hypot(dx, dy),
            "initial_heading_turn_rad": wrap_to_pi(bearing - base.yaw),
            "final_yaw_turn_rad": wrap_to_pi(goal.pose_xyyaw[2] - base.yaw),
            "navigation_parameters": parameters,
            "parameter_sources": {
                "goal_generator": "team_sorting.navigation.NavigationController.build_pick_goal",
                "standoff_m": "trial_cli:--standoff-m (TRIAL_NOT_FROZEN)",
                "position_tolerance_m": (
                    "trial_cli:--position-tolerance-m (TRIAL_NOT_FROZEN)"
                ),
                "yaw_tolerance_rad": (
                    "trial_cli:--yaw-tolerance-rad (TRIAL_NOT_FROZEN)"
                ),
                "target_position": "stage2_calibration_fixture:/data/pick-input.json",
                "current_base_state": "live:/slamware_ros_sdk_server_node/odom",
                "remaining_navigation_parameters": "config:config.yaml.navigation unchanged",
            },
            "failure_reason": "",
        }
    except Exception as exc:
        return {
            "schema": COMMAND_SCHEMA, "command": "plan-base-stand",
            "mode": "plan-only", "valid": False, "status": "BLOCKED",
            "published_control": False, "failure_reason": str(exc),
        }


def _standoff_values(minimum: float, maximum: float, step: float) -> tuple[float, ...]:
    values = (float(minimum), float(maximum), float(step))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("standoff扫描范围必须全部为有限值")
    if minimum <= 0.0 or maximum < minimum or step <= 0.0:
        raise ValueError("standoff扫描要求0<min<=max且step>0")
    count = int(math.floor((maximum - minimum) / step + 1e-12)) + 1
    if count > 1000:
        raise ValueError("standoff扫描候选超过1000个，拒绝执行")
    result = [minimum + index * step for index in range(count)]
    if result[-1] < maximum - 1e-12:
        result.append(maximum)
    return tuple(round(value, 12) for value in result)


def _candidate_pick_payload(
    payload: Mapping[str, Any], goal: NavGoal, actual: RobotJointState
) -> dict[str, Any]:
    """Build one offline candidate around the explicitly recorded joint snapshot."""

    result = json.loads(json.dumps(payload))
    timestamp_ns = actual.timestamp_ns
    result["joint_state"] = _jsonable(actual)
    x, y, yaw = goal.pose_xyyaw
    planar = RigidTransform3D(
        source_frame="virtual_footprint", target_frame=OFFICIAL_ODOM_FRAME,
        translation_xyz=(x, y, 0.0),
        rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
        timestamp_ns=timestamp_ns, valid=True,
    )
    odom_to_footprint, object_diagnostics = transform_object_to_virtual_footprint(
        result["object_estimate"]["position_xyz"],
        result["object_estimate"]["orientation_xyzw"], planar,
    )
    result["now_ns"] = timestamp_ns
    result["object_estimate"]["timestamp_ns"] = timestamp_ns
    result["transforms"] = {
        "target_to_footprint": _jsonable(odom_to_footprint),
        "target_to_world": _jsonable(RigidTransform3D(
            source_frame=OFFICIAL_ODOM_FRAME, target_frame="world",
            translation_xyz=(0.0, 0.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            timestamp_ns=timestamp_ns, valid=True,
        )),
        "coordinate_diagnostics": {
            "raw_base_transform": None,
            "planarized_virtual_footprint_transform": _jsonable(planar),
            "raw_roll_pitch_yaw": None,
            "virtual_footprint_roll_pitch_yaw": [0.0, 0.0, yaw],
            **object_diagnostics,
            "planarization_scope": "official_offline_calibration_only",
            "published_control": False,
        },
    }
    return result


def _first_diagnostic_failure(diagnostics: Mapping[str, Any]) -> tuple[Any, Any, str]:
    for slide in diagnostics.get("slide_candidates", []):
        for stage in slide.get("stages", []):
            dual = stage["dual"]
            if dual["success"]:
                continue
            left_ok = stage["left"]["success"]
            right_ok = stage["right"]["success"]
            failed_arm = (
                "dual" if left_ok and right_ok else
                "left" if not left_ok and right_ok else
                "right" if left_ok and not right_ok else "left+right"
            )
            return stage["stage"], failed_arm, dual["failure_reason"]
    return None, None, str(diagnostics.get("failure_reason", "规划失败但没有IK阶段诊断"))


def _diagnostic_cause(diagnostics: Mapping[str, Any]) -> str:
    stages = [
        stage
        for slide in diagnostics.get("slide_candidates", [])
        for stage in slide.get("stages", [])
        if not stage["dual"]["success"]
    ]
    if not stages:
        return "NON_IK_PLANNING_FAILURE"
    categories = {stage["dual"]["classification"] for stage in stages}
    if categories & {"OTHER_ERROR", "RETURN_TYPE_ERROR", "LENGTH_ERROR", "NON_FINITE"}:
        return "KDL_CALL_OR_RETURN_PROBLEM"
    if any(stage["single_arm_solution_without_dual"] for stage in stages):
        return "DUAL_ARM_COUPLING_OR_SPACING"
    if "JOINT_OUT_OF_LIMITS" in categories:
        return "JOINT_LIMIT_CONSTRAINT"
    failed_names = {stage["stage"] for stage in stages}
    if failed_names <= {"short-lift", "retreat"}:
        return "TARGET_HEIGHT_OR_POST_GRASP_POSE_UNREACHABLE"
    return "TARGET_GRASP_POSE_OR_END_EFFECTOR_ORIENTATION_UNREACHABLE"


def _trajectory_quality(result: Mapping[str, Any]) -> tuple[float | None, float | None]:
    margins: list[float] = []
    deltas: list[float] = []
    kinematic_names = {
        JOINT_NAMES[index] for index in (0, *range(3, 9), *range(10, 16))
    }
    for waypoint in result.get("waypoints", []):
        margins.extend(
            float(value) for name, value in waypoint["limit_margin_by_joint"].items()
            if name in kinematic_names and value is not None
        )
        deltas.extend(
            abs(float(value))
            for name, value in waypoint["delta_from_previous_by_joint"].items()
            if name in kinematic_names
        )
    return (min(margins) if margins else None, max(deltas) if deltas else None)


def sweep_pick_stand(
    payload: Mapping[str, Any], config: Mapping[str, Any], current_base: BaseState,
    adapter: Any, standoff_min_m: float, standoff_max_m: float,
    standoff_step_m: float,
) -> dict[str, Any]:
    """纯规划扫描；每个候选均复用生产站位生成和完整ArmPlanner入口。"""

    output: dict[str, Any] = {
        "schema": COMMAND_SCHEMA, "command": "sweep-pick-stand",
        "mode": "plan-only", "status": "BLOCKED", "valid": False,
        "published_control": False, "candidate_status": "TRIAL_NOT_FROZEN",
        "feasible_candidate_count": 0, "recommended_candidate": None,
        "candidates": [], "failure_reason": "",
    }
    try:
        validate_stage2_fixture(payload, config)
        if not isinstance(current_base, BaseState) or not current_base.valid:
            raise ValueError("sweep-pick-stand要求有效实时BaseState用于生成站位及移动量排序")
        actual = _joints(payload["joint_state"])
        standoffs = _standoff_values(standoff_min_m, standoff_max_m, standoff_step_m)
        production_nav = config["navigation"]
        candidates = []
        for standoff in standoffs:
            base_plan = plan_base_stand(
                payload, config, current_base, standoff,
                production_nav["position_tolerance_m"],
                production_nav["yaw_tolerance_rad"],
            )
            if not base_plan["valid"]:
                raise ValueError(base_plan["failure_reason"])
            goal_raw = base_plan["goal"]
            goal = NavGoal(
                goal_raw["goal_id"], goal_raw["goal_type"], tuple(goal_raw["pose_xyyaw"]),
                goal_raw["frame_id"], goal_raw["position_tolerance"],
                goal_raw["yaw_tolerance"], goal_raw["deadline_ns"],
                goal_raw["valid"], goal_raw["failure_reason"],
            )
            candidate_payload = _candidate_pick_payload(
                payload, goal, actual
            )
            planned = plan_pick(
                candidate_payload, config, adapter,
                include_full_ik_diagnostics=True,
            )
            diagnostics = planned.get("ik_diagnostics", {})
            calibration = planned.get("calibration_analysis", {})
            full_ik = bool(
                planned.get("valid")
                and diagnostics.get("all_stages_dual_ik_success")
            )
            successful_slide = next((
                slide["slide_m"] for slide in diagnostics.get("slide_candidates", [])
                if slide.get("complete_dual_path_success")
            ), None)
            margin, delta = _trajectory_quality(planned)
            failed_stage, failed_arm, diagnostic_reason = _first_diagnostic_failure(diagnostics)
            reason = "" if full_ik else (planned.get("failure_reason") or diagnostic_reason)
            candidates.append({
                "standoff_m": standoff, "status": "TRIAL_NOT_FROZEN",
                "candidate_base_pose": list(goal.pose_xyyaw),
                "travel_distance_from_current_base_m": base_plan["travel_distance_m"],
                "object_pose_in_virtual_footprint": candidate_payload["transforms"]
                    ["coordinate_diagnostics"]["object_pose_in_virtual_footprint"],
                "full_pick_ik_success": full_ik,
                "successful_slide": successful_slide if full_ik else None,
                "failed_stage": None if full_ik else failed_stage,
                "failed_arm": None if full_ik else failed_arm,
                "failure_reason": reason,
                "failure_cause_classification": (
                    None if full_ik else _diagnostic_cause(diagnostics)
                ),
                "continuous_ik_branch_exists": bool(
                    calibration.get("continuous_ik_branch_exists")
                ),
                "transition_required": bool(calibration.get("transition_required")),
                "transition_segment_count": (
                    calibration.get("transition_plan", {}).get("segment_count")
                    if isinstance(calibration.get("transition_plan"), Mapping) else None
                ),
                "calibration_analysis": calibration,
                "min_joint_limit_margin": margin if full_ik else None,
                "max_joint_delta": delta if full_ik else None,
                "ik_diagnostics": diagnostics,
                "published_control": False,
            })
        feasible = [item for item in candidates if item["full_pick_ik_success"]]
        feasible.sort(key=lambda item: (
            -float(item["min_joint_limit_margin"]),
            float(item["max_joint_delta"]),
            float(item["travel_distance_from_current_base_m"]),
        ))
        output.update({
            "valid": bool(feasible),
            "status": "TRIAL_NOT_FROZEN" if feasible else "NO_FEASIBLE_STANDOFF",
            "feasible_candidate_count": len(feasible),
            "recommended_candidate": feasible[0] if feasible else None,
            "candidates": candidates,
            "diagnostic_summary": {
                cause: sum(
                    1 for item in candidates
                    if item["failure_cause_classification"] == cause
                )
                for cause in sorted({
                    item["failure_cause_classification"] for item in candidates
                    if item["failure_cause_classification"] is not None
                })
            },
            "failure_reason": "" if feasible else (
                f"{standoff_min_m:.6g}～{standoff_max_m:.6g}m扫描内没有完整双臂、"
                "完整抓取阶段可行站位；"
                "请查看各候选ik_diagnostics中的KDL分类，不得继续盲目移动底盘"
            ),
        })
    except Exception as exc:
        output["failure_reason"] = str(exc)
    return output


def compare_pick_standoffs(
    payload: Mapping[str, Any], config: Mapping[str, Any], current_base: BaseState,
    actual: RobotJointState, adapter: Any, *, evidence_source: str,
    state_fixture_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """离线比较固定0.55/0.75候选；不构造ROS runtime或publisher。"""

    output: dict[str, Any] = {
        "schema": COMMAND_SCHEMA,
        "command": "compare-pick-standoffs",
        "mode": "plan-only",
        "published_control": False,
        "valid": False,
        "status": "BLOCKED",
        "evidence_source": evidence_source,
        "state_fixture_mode": (
            state_fixture_metadata.get("state_fixture_mode")
            if isinstance(state_fixture_metadata, Mapping)
            else ("test_fixture" if evidence_source == "test_fixture" else None)
        ),
        "state_fixture_capture": (
            dict(state_fixture_metadata)
            if isinstance(state_fixture_metadata, Mapping) else None
        ),
        "candidates": [],
        "recommended_candidate": None,
        "failure_reason": "",
    }
    try:
        if evidence_source not in {"saved_official_joint_state", "test_fixture"}:
            raise ValueError(
                "evidence_source必须是saved_official_joint_state或test_fixture"
            )
        validate_stage2_fixture(payload, config)
        specifications = (
            (0.55, 0.02, 0.02),
            (0.75, 0.05, 0.10),
        )
        production_nav = config["navigation"]
        candidates = []
        for standoff, lift, retreat in specifications:
            base_plan = plan_base_stand(
                payload, config, current_base, standoff,
                production_nav["position_tolerance_m"],
                production_nav["yaw_tolerance_rad"],
            )
            if not base_plan["valid"]:
                raise ValueError(base_plan["failure_reason"])
            goal_raw = base_plan["goal"]
            goal = NavGoal(
                goal_raw["goal_id"], goal_raw["goal_type"], tuple(goal_raw["pose_xyyaw"]),
                goal_raw["frame_id"], goal_raw["position_tolerance"],
                goal_raw["yaw_tolerance"], goal_raw["deadline_ns"],
                goal_raw["valid"], goal_raw["failure_reason"],
            )
            candidate_payload = _candidate_pick_payload(
                payload, goal, actual
            )
            trial = candidate_payload.setdefault("trial_parameters", {})
            trial.update({
                "lift_distance_m": lift,
                "retreat_distance_m": retreat,
                "max_arm_waypoint_delta_rad": CALIBRATION_ARM_STEP_LIMIT_RAD,
            })
            planned = plan_pick(
                candidate_payload, config, adapter,
                include_full_ik_diagnostics=False,
            )
            analysis = planned.get("calibration_analysis", {})
            transition = analysis.get("transition_plan")
            candidates.append({
                "standoff_m": standoff,
                "lift_distance_m": lift,
                "retreat_distance_m": retreat,
                "continuous_ik_branch_exists": bool(
                    analysis.get("continuous_ik_branch_exists")
                ),
                "transition_required": bool(analysis.get("transition_required")),
                "transition_segment_count": (
                    transition.get("segment_count") if isinstance(transition, Mapping)
                    else None
                ),
                "max_single_segment_joint_delta_rad": (
                    transition.get("max_single_segment_arm_delta_rad")
                    if isinstance(transition, Mapping)
                    else analysis.get("continuity_diagnostics", {}).get("max_arm_delta_rad")
                ),
                "minimum_joint_limit_margin": analysis.get(
                    "minimum_joint_limit_margin"
                ),
                "selected_slide_m": analysis.get("selected_slide_m"),
                "all_fk_checks_ok": (
                    transition.get("all_fk_checks_ok")
                    if isinstance(transition, Mapping)
                    else analysis.get("all_fk_checks_ok")
                ),
                "fk_checks": analysis.get("fk_checks", []),
                "collision_check_available": False,
                "collision_visual_verification_required": bool(
                    analysis.get("collision_visual_verification_required")
                ),
                "collision_verification_status": analysis.get(
                    "collision_verification_status", "NOT_AVAILABLE_NO_COMPLETE_IK"
                ),
                "parameter_sources": {
                    "standoff_m": "fixed_comparison_specification",
                    "lift_distance_m": "fixed_comparison_specification",
                    "retreat_distance_m": "fixed_comparison_specification",
                    "max_arm_waypoint_delta_rad": "fixed_safety_limit_1.0",
                    "remaining_grasp_parameters": "explicit_cli_trial_or_repository_config",
                },
                "calibration_analysis": analysis,
                "published_control": False,
            })
        output["candidates"] = candidates
        usable = [item for item in candidates if (
            (item["continuous_ik_branch_exists"] and item["all_fk_checks_ok"] is True)
            or (
                item["transition_required"]
                and item["transition_segment_count"] is not None
                and item["all_fk_checks_ok"] is True
            )
        )]
        if evidence_source != "saved_official_joint_state":
            output["failure_reason"] = (
                "比较只基于测试夹具，没有真实JointState支持正式推荐"
            )
            return output
        if not usable:
            output["failure_reason"] = "两个候选都没有连续IK或FK通过的过渡计划"
            return output
        usable.sort(key=lambda item: (
            not item["continuous_ik_branch_exists"],
            item["transition_segment_count"] or 0,
            item["max_single_segment_joint_delta_rad"] or math.inf,
            -(item["minimum_joint_limit_margin"] or -math.inf),
        ))
        recommended = usable[0]
        output.update({
            "valid": True,
            "status": "TRIAL_NOT_FROZEN_COLLISION_NOT_AUTOMATICALLY_CHECKED",
            "recommended_candidate": {
                "standoff_m": recommended["standoff_m"],
                "reason": (
                    "优先连续IK；否则依次选择过渡段更少、最大单段变化更小、"
                    "最小限位余量更大的候选"
                ),
            },
        })
    except Exception as exc:  # noqa: BLE001 - CLI诊断边界
        output["failure_reason"] = str(exc)
    return output


def capture_pick_comparison_state(
    config: Mapping[str, Any], runtime: Any, scene: str, seed: int,
    joint_state_timeout_s: float,
) -> dict[str, Any]:
    """只读捕获官方JointState及同期Odom，生成compare可直接消费的fixture。"""

    source = str(config.get("topics", {}).get("joint_states", ""))
    output: dict[str, Any] = {
        "schema": COMMAND_SCHEMA,
        "command": "capture-pick-comparison-state",
        "valid": False,
        "blockers": [],
        "source": source,
        "scene": scene,
        "seed": seed,
        "evidence_source": "saved_official_joint_state",
        "raw_joint_state": None,
        "tool_received_at_ns": None,
        "joint_state_header_timestamp_ns": None,
        "joint_name_validation": None,
        "normalized_joint_names": list(JOINT_NAMES),
        "normalized_position": None,
        "normalized_position_by_joint": None,
        "normalized_velocity": None,
        "normalized_effort": None,
        "joint_state": None,
        "base_state": None,
        "publisher_objects_created": False,
        "published_control": False,
    }
    blockers: list[str] = []
    try:
        if os.getenv("ROS_DOMAIN_ID") != "99":
            blockers.append("ROS_DOMAIN_ID必须严格为99")
        if source != "/joint_states":
            blockers.append(f"配置JointState source必须严格为/joint_states，实际={source!r}")
        if scene not in SCENE_FIXTURE_KEYS:
            blockers.append(f"scene必须是{tuple(SCENE_FIXTURE_KEYS)}之一")
        if type(seed) is not int or seed < 0:
            blockers.append("seed必须是真正非负整数")
        if (
            isinstance(joint_state_timeout_s, bool)
            or not isinstance(joint_state_timeout_s, (int, float))
            or not math.isfinite(float(joint_state_timeout_s))
            or float(joint_state_timeout_s) <= 0.0
        ):
            blockers.append("joint_state_timeout_s必须是有限正数")
        if blockers:
            output["blockers"] = blockers
            return output

        runtime.wait_for_inputs(float(joint_state_timeout_s))
        raw = getattr(runtime, "latest_joint_raw", None)
        if not isinstance(raw, Mapping):
            raise ValueError("没有保存到原始/joint_states消息")
        names = raw.get("name")
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError("原始JointState.name必须是字符串数组")
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        missing_names = sorted(set(JOINT_NAMES) - set(names))
        unexpected_names = sorted(set(names) - set(JOINT_NAMES))
        exact_set = (
            len(names) == len(JOINT_NAMES)
            and not duplicate_names and not missing_names and not unexpected_names
        )
        validation = {
            "expected_count": len(JOINT_NAMES),
            "actual_count": len(names),
            "exact_joint_set": exact_set,
            "duplicate_names": duplicate_names,
            "missing_names": missing_names,
            "unexpected_names": unexpected_names,
            "raw_name_order": list(names),
            "normalized_name_order": list(JOINT_NAMES),
            "required_explicit_names": [
                "slide_joint",
                *JOINT_NAMES[3:10],
                *JOINT_NAMES[10:17],
            ],
        }
        output["joint_name_validation"] = validation
        if not exact_set:
            raise ValueError(
                "JointState必须严格包含团队17关节集合："
                f"missing={missing_names},unexpected={unexpected_names},"
                f"duplicates={duplicate_names},count={len(names)}"
            )
        indices = {name: index for index, name in enumerate(names)}
        normalized: dict[str, tuple[float, ...]] = {}
        for field in ("position", "velocity", "effort"):
            values = raw.get(field)
            if not isinstance(values, list) or len(values) != len(names):
                raise ValueError(
                    f"原始JointState.{field}必须与name同为严格17项，实际="
                    f"{None if not isinstance(values, list) else len(values)}"
                )
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"原始JointState.{field}包含非有限值")
            normalized[field] = tuple(float(values[indices[name]]) for name in JOINT_NAMES)
        header = raw.get("header")
        if not isinstance(header, Mapping):
            raise ValueError("原始JointState缺少header")
        header_timestamp_ns = header.get("timestamp_ns")
        received_ns = raw.get("tool_received_at_ns")
        if type(header_timestamp_ns) is not int or header_timestamp_ns < 0:
            raise ValueError("JointState header时间戳必须是真正非负整数")
        if type(received_ns) is not int or received_ns < 0:
            raise ValueError("工具接收时间必须是真正非负整数")
        if received_ns < header_timestamp_ns:
            raise ValueError("工具接收时间早于JointState header时间，时钟域不一致")
        joint_state = RobotJointState(
            position=normalized["position"],
            velocity=normalized["velocity"],
            effort=normalized["effort"],
            joint_names=JOINT_NAMES,
            timestamp_ns=header_timestamp_ns,
        )
        base = runtime.latest_base_state()
        if not isinstance(base, BaseState) or not base.valid or base.frame_id != "odom":
            raise ValueError("compare fixture要求同期有效odom BaseState")
        publishers = getattr(runtime, "publishers", {})
        base_publisher = getattr(runtime, "base_publisher", None)
        publisher_objects_created = bool(publishers) or base_publisher is not None
        if publisher_objects_created:
            raise ValueError("只读捕获期间出现控制publisher对象")
        output.update({
            "valid": True,
            "blockers": [],
            "raw_joint_state": _jsonable(raw),
            "tool_received_at_ns": received_ns,
            "joint_state_header_timestamp_ns": header_timestamp_ns,
            "normalized_position": list(normalized["position"]),
            "normalized_position_by_joint": dict(
                zip(JOINT_NAMES, normalized["position"])
            ),
            "normalized_velocity": list(normalized["velocity"]),
            "normalized_effort": list(normalized["effort"]),
            "joint_state": _jsonable(joint_state),
            "base_state": _jsonable(base),
            "publisher_objects_created": False,
            "published_control": False,
        })
    except Exception as exc:  # noqa: BLE001 - 捕获边界必须写出明确blocker
        blockers.append(str(exc))
        output["blockers"] = blockers
    return output


def _state_fixture_field(raw: Mapping[str, Any], name: str) -> Any:
    if name not in raw:
        raise ValueError(f"STATE_FIXTURE_FIELD_MISSING: {name}")
    return raw[name]


def _comparison_fixture(
    raw: Mapping[str, Any], *, expected_scene: str | None = None,
    expected_seed: int | None = None,
) -> tuple[RobotJointState, BaseState, str, dict[str, Any]]:
    required_fields = (
        "schema", "valid", "blockers", "scene", "seed", "source",
        "raw_joint_state", "joint_name_validation", "normalized_joint_names",
        "normalized_position", "normalized_position_by_joint",
        "normalized_velocity", "normalized_effort", "joint_state",
        "joint_state_header_timestamp_ns", "tool_received_at_ns", "base_state",
        "evidence_source", "publisher_objects_created", "published_control",
    )
    for name in required_fields:
        _state_fixture_field(raw, name)
    if raw["schema"] != COMMAND_SCHEMA:
        raise ValueError(
            f"STATE_FIXTURE_SCHEMA_INVALID: expected={COMMAND_SCHEMA!r},"
            f"actual={raw['schema']!r}"
        )
    if raw["valid"] is not True:
        raise ValueError("state fixture.valid必须严格为true")
    if raw["published_control"] is not False:
        raise ValueError("state fixture必须严格包含published_control=false")
    if raw["publisher_objects_created"] is not False:
        raise ValueError("state fixture必须证明publisher_objects_created=false")
    if raw["source"] != "/joint_states":
        raise ValueError("state fixture.source必须严格为/joint_states")
    if raw["blockers"] != []:
        raise ValueError("state fixture.blockers必须为空数组")
    scene = raw["scene"]
    seed = raw["seed"]
    if not isinstance(scene, str) or not scene:
        raise ValueError("STATE_FIXTURE_FIELD_INVALID: scene")
    if type(seed) is not int or seed < 0:
        raise ValueError("STATE_FIXTURE_FIELD_INVALID: seed")
    if expected_scene is not None and scene != expected_scene:
        raise ValueError(
            f"STATE_FIXTURE_SCENE_MISMATCH: fixture={scene!r},pick_input={expected_scene!r}"
        )
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(
            f"STATE_FIXTURE_SEED_MISMATCH: fixture={seed!r},pick_input={expected_seed!r}"
        )
    validation = raw["joint_name_validation"]
    if not isinstance(validation, Mapping) or validation.get("exact_joint_set") is not True:
        raise ValueError("state fixture缺少严格17关节集合验证")
    if raw["normalized_joint_names"] != list(JOINT_NAMES):
        raise ValueError("state fixture.normalized_joint_names顺序无效")
    raw_joint_state = raw["raw_joint_state"]
    if not isinstance(raw_joint_state, Mapping):
        raise ValueError("STATE_FIXTURE_FIELD_INVALID: raw_joint_state必须是object")
    raw_names = raw_joint_state.get("name")
    if (
        not isinstance(raw_names, list)
        or any(not isinstance(name, str) for name in raw_names)
        or len(raw_names) != len(JOINT_NAMES)
        or len(set(raw_names)) != len(JOINT_NAMES)
        or set(raw_names) != set(JOINT_NAMES)
    ):
        raise ValueError("STATE_FIXTURE_JOINT_SET_INVALID: raw_joint_state.name")
    raw_index = {name: index for index, name in enumerate(raw_names)}
    for field in ("position", "velocity", "effort"):
        raw_values = raw_joint_state.get(field)
        normalized_values = raw[f"normalized_{field}"]
        if (
            not isinstance(raw_values, list)
            or len(raw_values) != len(JOINT_NAMES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in raw_values
            )
        ):
            raise ValueError(f"STATE_FIXTURE_VECTOR_INVALID: raw_joint_state.{field}")
        if (
            not isinstance(normalized_values, list)
            or len(normalized_values) != len(JOINT_NAMES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in normalized_values
            )
        ):
            raise ValueError(f"STATE_FIXTURE_VECTOR_INVALID: normalized_{field}")
        normalized_from_raw = [
            float(raw_values[raw_index[name]]) for name in JOINT_NAMES
        ]
        if normalized_values != normalized_from_raw:
            raise ValueError(
                f"STATE_FIXTURE_NORMALIZATION_INVALID: normalized_{field}"
            )
    joint_raw = raw["joint_state"]
    base_raw = raw["base_state"]
    if not isinstance(joint_raw, Mapping) or not isinstance(base_raw, Mapping):
        raise ValueError("STATE_FIXTURE_FIELD_INVALID: joint_state与base_state必须是object")
    if joint_raw.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("STATE_FIXTURE_JOINT_ORDER_INVALID: joint_state.joint_names")
    base_data = dict(base_raw)
    for name in (
        "position_xyz", "orientation_xyzw", "linear_velocity_xyz",
        "angular_velocity_xyz",
    ):
        base_data[name] = tuple(base_data[name])
    evidence_source = raw["evidence_source"]
    if not isinstance(evidence_source, str):
        raise ValueError("state fixture缺少evidence_source")
    if evidence_source != "saved_official_joint_state":
        raise ValueError("CLI比较只接受saved_official_joint_state证据")
    state = _joints(joint_raw)
    if not state.valid:
        raise ValueError("STATE_FIXTURE_FIELD_INVALID: joint_state.valid必须为true")
    for field in ("position", "velocity", "effort"):
        expected = raw[f"normalized_{field}"]
        if expected != list(getattr(state, field)):
            raise ValueError(f"state fixture.normalized_{field}与joint_state不一致")
    expected_position_by_joint = dict(zip(JOINT_NAMES, state.position))
    if raw["normalized_position_by_joint"] != expected_position_by_joint:
        raise ValueError(
            "state fixture.normalized_position_by_joint与团队顺序位置不一致"
        )
    header_timestamp_ns = raw["joint_state_header_timestamp_ns"]
    received_ns = raw["tool_received_at_ns"]
    if type(header_timestamp_ns) is not int or header_timestamp_ns < 0:
        raise ValueError("STATE_FIXTURE_TIMESTAMP_INVALID: joint_state_header_timestamp_ns")
    if type(received_ns) is not int or received_ns < header_timestamp_ns:
        raise ValueError("STATE_FIXTURE_TIMESTAMP_INVALID: tool_received_at_ns")
    header = raw_joint_state.get("header")
    if not isinstance(header, Mapping):
        raise ValueError("STATE_FIXTURE_FIELD_INVALID: raw_joint_state.header")
    stamp = header.get("stamp")
    if not isinstance(stamp, Mapping):
        raise ValueError("STATE_FIXTURE_FIELD_INVALID: raw_joint_state.header.stamp")
    sec, nanosec = stamp.get("sec"), stamp.get("nanosec")
    if (
        type(sec) is not int or sec < 0 or type(nanosec) is not int
        or not 0 <= nanosec < 1_000_000_000
        or header.get("timestamp_ns") != sec * 1_000_000_000 + nanosec
        or raw_joint_state.get("tool_received_at_ns") != received_ns
    ):
        raise ValueError("STATE_FIXTURE_TIMESTAMP_INVALID: raw JointState header/receipt")
    if header_timestamp_ns != state.timestamp_ns:
        raise ValueError("state fixture header时间戳与joint_state不一致")
    metadata = {
        "state_fixture_mode": "recorded_official_joint_state",
        "scene": scene,
        "seed": seed,
        "joint_state_header_timestamp_ns": header_timestamp_ns,
        "tool_received_at_ns": received_ns,
    }
    return state, BaseState(**base_data), evidence_source, metadata


def _load_comparison_fixture(
    path: str | Path, payload: Mapping[str, Any]
) -> tuple[RobotJointState, BaseState, str, dict[str, Any]]:
    fixture_path = Path(path)
    if not fixture_path.is_file():
        raise ValueError(f"STATE_FIXTURE_FILE_NOT_FOUND: {fixture_path}")
    if "scene" not in payload:
        raise ValueError("PICK_INPUT_FIELD_MISSING: scene")
    if "seed" not in payload:
        raise ValueError("PICK_INPUT_FIELD_MISSING: seed")
    return _comparison_fixture(
        _load_json(fixture_path), expected_scene=payload["scene"],
        expected_seed=payload["seed"],
    )


def _validate_base_plan(plan: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[NavGoal, NavigationConfig]:
    if plan.get("command") != "plan-base-stand" or plan.get("valid") is not True:
        raise ValueError("必须提供有效plan-base-stand JSON")
    if plan.get("status") != "TRIAL_NOT_FROZEN" or plan.get("published_control") is not False:
        raise ValueError("base stand plan状态或plan-only标记无效")
    parameters = plan.get("navigation_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("plan缺少navigation_parameters")
    nav = NavigationConfig(**dict(parameters))
    current = _navigation_config_for_trial(
        config, nav.standoff_m, nav.position_tolerance_m, nav.yaw_tolerance_rad
    )
    for name in fields(NavigationConfig):
        if getattr(nav, name.name) != getattr(current, name.name):
            raise ValueError(f"plan导航参数与当前配置不一致：{name.name}")
    limits = {
        "max_abs_v_mps": 0.25, "max_abs_w_radps": 0.50,
        "odom_max_age_ns": 150_000_000, "goal_timeout_ns": 30_000_000_000,
        "settled_required_cycles": 3,
        "max_settled_linear_speed_mps": 0.01,
        "max_settled_angular_speed_radps": 0.02,
    }
    for name, required in limits.items():
        if getattr(nav, name) != required:
            raise ValueError(f"本轮授权导航安全参数不匹配：{name}")
    goal_raw = plan.get("goal")
    if not isinstance(goal_raw, Mapping):
        raise ValueError("plan缺少goal")
    goal_data = dict(goal_raw)
    goal_data["pose_xyyaw"] = tuple(goal_data["pose_xyyaw"])
    goal = NavGoal(**goal_data)
    if goal.goal_type != "pick" or goal.frame_id != "odom":
        raise ValueError("plan目标不是odom抓取站位")
    if (
        goal.position_tolerance != nav.position_tolerance_m
        or goal.yaw_tolerance != nav.yaw_tolerance_rad
    ):
        raise ValueError("plan目标容差与navigation_parameters试验容差不一致")
    return goal, nav


class RosCalibrationRuntime:
    """独立标定脚本的ROS读写边界；不导入生产OfficialCommandPublisher。"""

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import rclpy
            from rclpy.time import Time
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import JointState
            from geometry_msgs.msg import Twist
            from std_msgs.msg import Float64MultiArray, String
            from tf2_msgs.msg import TFMessage
            import tf2_ros
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
        except Exception as exc:  # pragma: no cover - only official client image
            raise RuntimeError(f"ROS2标定依赖不可用：{exc}") from exc
        self.rclpy = rclpy
        self.Time = Time
        self.Float64MultiArray = Float64MultiArray
        self.Twist = Twist
        self.config = config
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        # 唯一节点名使当前进程endpoint可与上一短进程尚未收敛的DDS残留区分。
        self.node = rclpy.create_node(
            f"arm_pick_place_calibration_{os.getpid()}_{time.time_ns()}"
        )
        self.latest_joint: RobotJointState | None = None
        self.latest_joint_raw: dict[str, Any] | None = None
        self.latest_joint_received_ns: int | None = None
        self.latest_valid_joint_received_ns: int | None = None
        self.latest_joint_validation_error: str | None = None
        self.joint_received_count = 0
        self.latest_joint_arrival = 0.0
        self.latest_odom: Any | None = None
        self.latest_odom_arrival = 0.0
        self.odom_received_count = 0
        self.latest_instruction_raw: str | None = None
        self.latest_instruction_received_ns: int | None = None
        self.joint_timing_samples: list[tuple[int, int]] = []
        self.odom_timing_samples: list[tuple[int, int]] = []
        self.tf_timing_samples: list[tuple[int, int]] = []
        self.node.create_subscription(
            JointState, config["topics"]["joint_states"], self._on_joint, 10
        )
        self.node.create_subscription(Odometry, config["topics"]["odom"], self._on_odom, 10)
        self.instruction_qos_evidence = {
            "history": "KEEP_LAST", "depth": 10,
            "reliability": "RELIABLE", "durability": "VOLATILE",
            "compatibility_basis": (
                "same explicit profile used by prepare-pick-input and verified "
                "against the official offline Server"
            ),
        }
        instruction_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.node.create_subscription(
            String, config["topics"]["instruction"], self._on_instruction,
            instruction_qos,
        )
        self.node.create_subscription(TFMessage, "/tf", self._on_tf, 50)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.publishers: dict[str, Any] = {}
        self.base_publisher: Any | None = None

    def _on_joint(self, message: Any) -> None:
        self.joint_received_count += 1
        try:
            received_ns = int(self.node.get_clock().now().nanoseconds)
            self.latest_joint_raw = {
                "name": [str(value) for value in message.name],
                "position": [float(value) for value in message.position],
                "velocity": [float(value) for value in message.velocity],
                "effort": [float(value) for value in message.effort],
                "header": {
                    "frame_id": str(message.header.frame_id),
                    "stamp": {
                        "sec": int(message.header.stamp.sec),
                        "nanosec": int(message.header.stamp.nanosec),
                    },
                    "timestamp_ns": self._stamp_ns(message.header.stamp),
                },
                "tool_received_at_ns": received_ns,
            }
            self.latest_joint_received_ns = received_ns
            names = [str(value) for value in message.name]
            if len(names) != len(set(names)):
                raise ValueError("JointState包含重复关节名")
            indices = {name: index for index, name in enumerate(names)}
            missing = [name for name in JOINT_NAMES if name not in indices]
            if missing:
                raise ValueError(f"JointState缺少团队必需关节：{missing}")
            if len(message.position) < len(names):
                raise ValueError("JointState.position长度小于name")
            def optional(field: str) -> tuple[float, ...]:
                values = getattr(message, field, ())
                if not values:
                    return (0.0,) * 17
                if len(values) < len(names):
                    raise ValueError(f"JointState.{field}长度小于name")
                return tuple(float(values[indices[name]]) for name in JOINT_NAMES)
            self.latest_joint = RobotJointState(
                position=tuple(float(message.position[indices[name]]) for name in JOINT_NAMES),
                velocity=optional("velocity"), effort=optional("effort"),
                timestamp_ns=self._stamp_ns(message.header.stamp),
            )
            self.latest_valid_joint_received_ns = received_ns
            self.latest_joint_validation_error = None
            self.latest_joint_arrival = time.monotonic()
            self.joint_timing_samples.append(
                (received_ns, self.latest_joint.timestamp_ns)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.latest_joint_validation_error = str(exc)
            self.node.get_logger().error(str(exc))

    def _on_odom(self, message: Any) -> None:
        self.latest_odom = message
        self.latest_odom_arrival = time.monotonic()
        self.odom_received_count += 1
        self.odom_timing_samples.append(
            (int(self.node.get_clock().now().nanoseconds), self._stamp_ns(message.header.stamp))
        )

    def _on_instruction(self, message: Any) -> None:
        self.latest_instruction_raw = str(message.data)
        self.latest_instruction_received_ns = int(self.node.get_clock().now().nanoseconds)

    def _on_tf(self, message: Any) -> None:
        arrival_ns = int(self.node.get_clock().now().nanoseconds)
        for transform in message.transforms:
            try:
                parent = normalize_ros_frame(transform.header.frame_id)
                child = normalize_ros_frame(transform.child_frame_id)
            except ValueError:
                continue
            if parent == OFFICIAL_ODOM_FRAME and child == OFFICIAL_BASE_FRAME:
                self.tf_timing_samples.append((arrival_ns, self._stamp_ns(transform.header.stamp)))

    def monotonic(self) -> float:
        return time.monotonic()

    def spin_once(self, timeout_s: float) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=max(0.0, timeout_s))

    def wait_for_inputs(self, timeout_s: float) -> RobotJointState:
        deadline = self.monotonic() + timeout_s
        while self.monotonic() < deadline and (
            self.latest_joint is None or self.latest_odom is None
        ):
            self.spin_once(min(0.05, max(0.0, deadline - self.monotonic())))
        if self.latest_joint is None:
            raise RuntimeError("等待/joint_states超时")
        if self.latest_odom is None:
            raise RuntimeError("等待Odom超时")
        return self.latest_joint

    def poll_joint_state(self, timeout_s: float) -> RobotJointState | None:
        previous = None if self.latest_joint is None else self.latest_joint.timestamp_ns
        deadline = self.monotonic() + timeout_s
        while self.monotonic() < deadline:
            self.spin_once(min(0.02, max(0.0, deadline - self.monotonic())))
            if self.latest_joint is not None and self.latest_joint.timestamp_ns != previous:
                return self.latest_joint
        return None

    def wait_for_base_state(self, timeout_s: float) -> BaseState:
        deadline = self.monotonic() + timeout_s
        while self.monotonic() < deadline and self.latest_odom is None:
            self.spin_once(min(0.05, max(0.0, deadline - self.monotonic())))
        if self.latest_odom is None:
            raise RuntimeError("等待Odom超时")
        return _base_state_from_odom(self.latest_odom)

    def poll_base_state(self, timeout_s: float) -> BaseState | None:
        previous = None
        if self.latest_odom is not None:
            previous = self._stamp_ns(self.latest_odom.header.stamp)
        deadline = self.monotonic() + timeout_s
        while self.monotonic() < deadline:
            self.spin_once(min(0.02, max(0.0, deadline - self.monotonic())))
            if self.latest_odom is not None:
                current = self._stamp_ns(self.latest_odom.header.stamp)
                if current != previous:
                    return _base_state_from_odom(self.latest_odom)
        return None

    def odom_fresh(self, max_age_s: float) -> bool:
        if self.latest_odom is None:
            return False
        arrival_fresh = self.monotonic() - self.latest_odom_arrival <= max_age_s
        age_ns = (
            int(self.node.get_clock().now().nanoseconds)
            - self._stamp_ns(self.latest_odom.header.stamp)
        )
        return arrival_fresh and 0 <= age_ns <= int(max_age_s * 1_000_000_000)

    def spin_control_period(self, period_s: float) -> None:
        deadline = self.monotonic() + period_s
        while True:
            remaining = deadline - self.monotonic()
            if remaining <= 0.0:
                break
            # 在整个控制周期内处理ROS回调，避免其他订阅活跃时只处理一个
            # callback，也避免有立即可用的callback时控制循环快于约24 Hz。
            self.spin_once(min(0.005, remaining))

    def latest_base_state(self) -> BaseState | None:
        return None if self.latest_odom is None else _base_state_from_odom(self.latest_odom)

    def latest_odom_age_ns(self) -> int | None:
        if self.latest_odom is None:
            return None
        return (
            int(self.node.get_clock().now().nanoseconds)
            - self._stamp_ns(self.latest_odom.header.stamp)
        )

    def latest_joint_state(self) -> RobotJointState | None:
        return self.latest_joint

    def latest_joint_age_ns(self) -> int | None:
        if self.latest_joint is None:
            return None
        return int(self.node.get_clock().now().nanoseconds) - self.latest_joint.timestamp_ns

    def latest_joint_receipt_ns(self) -> int | None:
        return self.latest_valid_joint_received_ns

    def base_graph_counts(self) -> tuple[int, int]:
        topic = self.config["topics"]["official_commands"]["cmd_vel"]
        return int(self.node.count_publishers(topic)), int(self.node.count_subscribers(topic))

    def base_publisher_exclusivity_evidence(
        self, phase: str, publisher_started: bool
    ) -> dict[str, Any]:
        topic = self.config["topics"]["official_commands"]["cmd_vel"]
        endpoints = [{
            "node_name": str(info.node_name),
            "node_namespace": str(info.node_namespace),
            "gid": self._endpoint_gid(info),
            "classification": (
                "SELF" if info.node_name == self.node.get_name()
                and info.node_namespace == self.node.get_namespace()
                else "EXTERNAL_OR_DDS_RESIDUAL"
            ),
        } for info in self.node.get_publishers_info_by_topic(topic)]
        own = [item for item in endpoints if item["classification"] == "SELF"]
        external = [item for item in endpoints if item["classification"] != "SELF"]
        subscribers = int(self.node.count_subscribers(topic))
        expected_self = 1 if publisher_started else 0
        return {
            "phase": phase, "topic": topic, "endpoints": endpoints,
            "self_count": len(own), "external_count": len(external),
            "expected_self_count": expected_self, "subscriber_count": subscribers,
            "valid": len(own) == expected_self and not external and subscribers == 1,
        }

    def start_base_publisher(self) -> None:
        if self.base_publisher is not None:
            raise RuntimeError("/cmd_vel标定publisher已经创建")
        topic = self.config["topics"]["official_commands"]["cmd_vel"]
        self.base_publisher = self.node.create_publisher(self.Twist, topic, 5)

    def publish_base_velocity(self, linear_mps: float, angular_radps: float) -> None:
        if self.base_publisher is None:
            raise RuntimeError("/cmd_vel标定publisher尚未创建")
        if not math.isfinite(linear_mps) or not math.isfinite(angular_radps):
            raise ValueError("底盘速度必须有限")
        message = self.Twist()
        message.linear.x = float(linear_mps)
        message.angular.z = float(angular_radps)
        self.base_publisher.publish(message)

    def joint_fresh(self, max_age_s: float) -> bool:
        age_ns = self.latest_joint_age_ns()
        return (
            self.latest_joint is not None
            and self.latest_joint_validation_error is None
            and age_ns is not None
            and 0 <= age_ns <= int(max_age_s * 1_000_000_000)
            and self.monotonic() - self.latest_joint_arrival <= max_age_s
        )

    def _topic_metadata(self) -> dict[str, tuple[str, ...]]:
        return {name: tuple(types) for name, types in self.node.get_topic_names_and_types()}

    def other_publishers(self) -> dict[str, int]:
        topics = {item.group: item.topic for item in MMK2_CONTROLLER_MANIFEST_V1.official_topics}
        return {group: int(self.node.count_publishers(topics[group])) for group in ARM_TOPIC_GROUPS}

    @staticmethod
    def _endpoint_gid(info: Any) -> str:
        raw = getattr(info, "endpoint_gid", b"")
        try:
            return bytes(raw).hex()
        except (TypeError, ValueError):
            return str(raw)

    def publisher_exclusivity_evidence(
        self, phase: str, publishers_started: bool
    ) -> dict[str, Any]:
        topics = {item.group: item.topic for item in MMK2_CONTROLLER_MANIFEST_V1.official_topics}
        own_name = self.node.get_name()
        own_namespace = self.node.get_namespace()
        groups: dict[str, Any] = {}
        valid = True
        for group in ARM_TOPIC_GROUPS:
            infos = self.node.get_publishers_info_by_topic(topics[group])
            endpoints = [{
                "node_name": str(info.node_name),
                "node_namespace": str(info.node_namespace),
                "gid": self._endpoint_gid(info),
                "classification": (
                    "SELF" if info.node_name == own_name
                    and info.node_namespace == own_namespace else "EXTERNAL_OR_DDS_RESIDUAL"
                ),
            } for info in infos]
            own = [item for item in endpoints if item["classification"] == "SELF"]
            external = [item for item in endpoints if item["classification"] != "SELF"]
            expected_own = 1 if publishers_started and group in {
                "spine", "left_arm", "right_arm"
            } else 0
            group_valid = len(own) == expected_own and not external
            valid = valid and group_valid
            groups[group] = {
                "topic": topics[group], "expected_self_count": expected_own,
                "self_count": len(own), "external_count": len(external),
                "endpoints": endpoints, "valid": group_valid,
            }
        return {
            "phase": phase, "node_name": own_name,
            "node_namespace": own_namespace, "groups": groups, "valid": valid,
        }

    def wait_for_publisher_exclusivity(
        self, phase: str, publishers_started: bool, timeout_s: float = 0.75
    ) -> dict[str, Any]:
        deadline = self.monotonic() + timeout_s
        history: list[dict[str, Any]] = []
        while True:
            evidence = self.publisher_exclusivity_evidence(phase, publishers_started)
            history.append(evidence)
            if evidence["valid"]:
                return {**evidence, "converged": True, "sample_count": len(history)}
            remaining = deadline - self.monotonic()
            if remaining <= 0.0:
                return {
                    **evidence, "converged": False, "sample_count": len(history),
                    "history": history,
                }
            self.spin_once(min(0.05, remaining))

    def subscriber_counts(self) -> dict[str, int]:
        topics = {item.group: item.topic for item in MMK2_CONTROLLER_MANIFEST_V1.official_topics}
        return {group: int(self.node.count_subscribers(topics[group])) for group in ARM_TOPIC_GROUPS}

    @staticmethod
    def _stamp_ns(stamp: Any) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def lookup_transform(self, target_frame: str, source_frame: str) -> RigidTransform3D:
        """按tf2原生顺序查询：target_frame在前，source_frame在后。"""

        target_frame = normalize_ros_frame(target_frame)
        source_frame = normalize_ros_frame(source_frame)
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, self.Time()
            )
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if "does not exist" in lowered or "not exist" in lowered:
                category = "FRAME_NOT_FOUND"
            elif "connect" in lowered or "tree" in lowered:
                category = "TRANSFORM_NOT_CONNECTED"
            else:
                category = "LOOKUP_EXCEPTION"
            raise RuntimeError(
                f"{category}:target={target_frame}:source={source_frame}:{message}"
            ) from exc
        value = transform.transform
        return RigidTransform3D(
            source_frame=source_frame,
            target_frame=target_frame,
            translation_xyz=(value.translation.x, value.translation.y, value.translation.z),
            rotation_xyzw=(value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w),
            timestamp_ns=self._stamp_ns(transform.header.stamp),
            valid=True,
        )

    def wait_for_transform(
        self, target_frame: str, source_frame: str, timeout_s: float
    ) -> RigidTransform3D:
        """通过spin等待TF discovery，禁止用固定sleep掩盖Buffer尚未就绪。"""

        target_frame = normalize_ros_frame(target_frame)
        source_frame = normalize_ros_frame(source_frame)
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("tf_timeout_s必须是有限非负数")
        deadline = self.monotonic() + timeout_s
        last_error = "FRAME_NOT_FOUND:TF Buffer尚未发现目标frame"
        while True:
            try:
                available = bool(
                    self.tf_buffer.can_transform(target_frame, source_frame, self.Time())
                )
            except Exception as exc:
                available = False
                last_error = f"CAN_TRANSFORM_EXCEPTION:{exc}"
            else:
                last_error = (
                    f"FRAME_NOT_FOUND_OR_NOT_CONNECTED:target={target_frame}:"
                    f"source={source_frame}"
                )
            if available:
                # Buffer已经明确报告可用；此后的lookup错误必须作为lookup异常立即暴露，
                # 不能被循环吞掉后误报为等待超时。
                return self.lookup_transform(target_frame, source_frame)
            remaining = deadline - self.monotonic()
            if remaining <= 0.0:
                break
            self.spin_once(min(0.05, remaining))

        known_frames: set[str] = set()
        try:
            raw_frames = yaml.safe_load(self.tf_buffer.all_frames_as_yaml()) or {}
            if isinstance(raw_frames, Mapping):
                known_frames = {normalize_ros_frame(str(frame)) for frame in raw_frames}
        except Exception:
            known_frames = set()
        missing = [
            frame for frame in (target_frame, source_frame)
            if known_frames and frame not in known_frames
        ]
        if missing:
            cause = f"FRAME_NOT_FOUND:{','.join(missing)}"
        elif known_frames:
            cause = "TRANSFORM_NOT_CONNECTED"
        else:
            cause = f"FRAME_NOT_FOUND:{target_frame},{source_frame}:{last_error}"
        raise RuntimeError(
            f"TF_WAIT_TIMEOUT:target={target_frame}:source={source_frame}:"
            f"timeout_s={timeout_s}:{cause}"
        )

    @staticmethod
    def _identity_transform(source: str, target: str, timestamp_ns: int) -> RigidTransform3D:
        return RigidTransform3D(
            normalize_ros_frame(source), normalize_ros_frame(target),
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), timestamp_ns, True,
        )

    def offline_coordinate_evidence(self, tf_timeout_s: float = 5.0) -> dict[str, Any]:
        blockers: list[str] = []
        actual_odom = ""
        if self.latest_odom is None:
            blockers.append("ODOM_NOT_RECEIVED")
        else:
            try:
                actual_odom = normalize_ros_frame(self.latest_odom.header.frame_id)
            except ValueError:
                blockers.append("ODOM_FRAME_INVALID")
        if actual_odom and actual_odom != OFFICIAL_ODOM_FRAME:
            blockers.append(f"ODOM_FRAME_UNEXPECTED:{actual_odom}")
        try:
            base_transform = self.wait_for_transform(
                OFFICIAL_ODOM_FRAME, OFFICIAL_BASE_FRAME, tf_timeout_s
            )
        except RuntimeError as exc:
            base_transform = None
            blockers.append(f"ODOM_TO_BASE_LINK_MISSING:{exc}")
        comparison = None
        planarized = None
        planarization = {
            "raw_base_transform": None,
            "planarized_virtual_footprint_transform": None,
            "raw_roll_pitch_yaw": None,
            "virtual_footprint_roll_pitch_yaw": None,
            "planarization_scope": "official_offline_calibration_only",
        }
        if base_transform is not None and self.latest_odom is not None:
            try:
                comparison = compare_odom_and_tf_pose(self.latest_odom, base_transform)
                if not comparison["matches"]:
                    blockers.append("ODOM_TF_POSE_MISMATCH")
            except (AttributeError, TypeError, ValueError) as exc:
                blockers.append(f"ODOM_TF_COMPARISON_INVALID:{exc}")
        if base_transform is not None:
            try:
                planarized, planarization = planarize_base_transform(base_transform)
            except (TypeError, ValueError) as exc:
                blockers.append(f"VIRTUAL_FOOTPRINT_PLANARIZATION_INVALID:{exc}")
        if os.getenv("ROS_DOMAIN_ID") != "99":
            blockers.append("ROS_DOMAIN_ID_NOT_99")
        return {
            "actual_odom_frame": actual_odom or None,
            "actual_base_frame": OFFICIAL_BASE_FRAME if base_transform is not None else None,
            "transform_source": "tf:odom->base_link" if base_transform is not None else None,
            "base_transform": None if base_transform is None else _jsonable(base_transform),
            "odom_tf_comparison": comparison,
            **planarization,
            "world_equals_odom": not blockers,
            "blockers": blockers,
        }

    def _offline_planner_transforms(self, object_frame: str, timestamp_ns: int) -> dict[str, Any]:
        probe = self.probe(timeout_s=0.0, tf_timeout_s=5.0)
        if not probe["official_offline_conditions_met"]:
            raise RuntimeError(f"官方离线坐标链BLOCKED：{probe['blockers']}")
        object_frame = normalize_ros_frame(object_frame)
        if object_frame != OFFICIAL_ODOM_FRAME:
            raise RuntimeError(f"阶段2 fixture frame必须归一化为odom，实际={object_frame}")
        raw_base = _transform(probe["raw_base_transform"])
        planarized = _transform(probe["planarized_virtual_footprint_transform"])
        actual = _inverse_transform(planarized, "footprint")
        # ArmPlanner公共契约仍要求target_frame=footprint；本标定工具仅在已确认的官方
        # offline里把实际base_link数值适配到该KDL入口，不声称ROS中存在footprint。
        to_kdl = RigidTransform3D(
            source_frame=OFFICIAL_ODOM_FRAME, target_frame="footprint",
            translation_xyz=actual.translation_xyz, rotation_xyzw=actual.rotation_xyzw,
            timestamp_ns=actual.timestamp_ns, valid=True,
        )
        to_world = self._identity_transform(OFFICIAL_ODOM_FRAME, "world", timestamp_ns)
        return {
            "target_to_footprint": _jsonable(to_kdl),
            "target_to_world": _jsonable(to_world),
            "coordinate_diagnostics": {
                "raw_base_transform": _jsonable(raw_base),
                "planarized_virtual_footprint_transform": _jsonable(planarized),
                "raw_roll_pitch_yaw": probe["raw_roll_pitch_yaw"],
                "virtual_footprint_roll_pitch_yaw": probe[
                    "virtual_footprint_roll_pitch_yaw"
                ],
                "object_pose_in_odom": None,
                "object_pose_in_virtual_footprint": None,
                "object_local_z_in_virtual_footprint": None,
                "planarization_scope": "official_offline_calibration_only",
                "published_control": False,
            },
            "calibration_coordinate_adapter": {
                "actual_odom_frame": OFFICIAL_ODOM_FRAME,
                "actual_base_frame": OFFICIAL_BASE_FRAME,
                "transform_source": "tf:odom->base_link",
                "planner_target_label": "footprint",
                "world_equals_odom": True,
                "scope": "official_offline_calibration_tool_only",
                "planarization_scope": "official_offline_calibration_only",
            },
        }

    def capture_live_payload(self, payload: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
        result = json.loads(json.dumps(payload))
        joints = self.wait_for_inputs(timeout_s)
        fixture = stage2_fixture(self.config, str(result["scene"]), str(result["object_estimate"]["class_id"]))
        result["source"] = FIXTURE_SOURCE
        result["source_slot"] = fixture["source_slot"]
        for name in ("position_xyz", "size_xyz_m", "orientation_xyzw", "frame_id"):
            result["object_estimate"][name] = fixture[name]
        result["joint_state"] = _jsonable(joints)
        result["now_ns"] = joints.timestamp_ns
        result["object_estimate"]["timestamp_ns"] = joints.timestamp_ns
        object_frame = fixture["frame_id"]
        result["transforms"] = self._offline_planner_transforms(
            object_frame, joints.timestamp_ns
        )
        transforms = result["transforms"]
        planarized = _transform(
            transforms["coordinate_diagnostics"]["planarized_virtual_footprint_transform"]
        )
        _odom_to_virtual, object_diagnostics = transform_object_to_virtual_footprint(
            result["object_estimate"]["position_xyz"],
            result["object_estimate"]["orientation_xyzw"],
            planarized,
        )
        transforms["coordinate_diagnostics"].update(object_diagnostics)
        return result

    def capture_live_place_payload(self, payload: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
        result = json.loads(json.dumps(payload))
        joints = self.wait_for_inputs(timeout_s)
        result["joint_state"] = _jsonable(joints)
        result["now_ns"] = joints.timestamp_ns
        probe = self.probe(timeout_s=0.0, tf_timeout_s=5.0)
        if not probe["official_offline_conditions_met"]:
            raise RuntimeError(f"官方离线坐标链BLOCKED：{probe['blockers']}")
        raw_base = _transform(probe["raw_base_transform"])
        planarized = _transform(probe["planarized_virtual_footprint_transform"])
        actual = _inverse_transform(planarized, "footprint")
        result["transforms"] = {
            "world_to_footprint": _jsonable(RigidTransform3D(
                "world", "footprint", actual.translation_xyz, actual.rotation_xyzw,
                actual.timestamp_ns, True,
            )),
            "calibration_coordinate_adapter": {
                "actual_odom_frame": OFFICIAL_ODOM_FRAME,
                "actual_base_frame": OFFICIAL_BASE_FRAME,
                "transform_source": "tf:odom->base_link",
                "world_equals_odom": True,
                "scope": "official_offline_calibration_tool_only",
                "planarization_scope": "official_offline_calibration_only",
            },
            "coordinate_diagnostics": {
                "raw_base_transform": _jsonable(raw_base),
                "planarized_virtual_footprint_transform": _jsonable(planarized),
                "raw_roll_pitch_yaw": probe["raw_roll_pitch_yaw"],
                "virtual_footprint_roll_pitch_yaw": probe[
                    "virtual_footprint_roll_pitch_yaw"
                ],
                "object_pose_in_odom": None,
                "object_pose_in_virtual_footprint": None,
                "object_local_z_in_virtual_footprint": None,
                "planarization_scope": "official_offline_calibration_only",
                "published_control": False,
            },
        }
        return result

    def probe(self, timeout_s: float = 3.0, tf_timeout_s: float = 5.0) -> dict[str, Any]:
        input_error = ""
        try:
            self.wait_for_inputs(timeout_s)
        except RuntimeError as exc:
            input_error = str(exc)
        topic_types = self._topic_metadata()
        expected = {item.topic: item.message_type for item in MMK2_CONTROLLER_MANIFEST_V1.official_topics
                    if item.group in ARM_TOPIC_GROUPS}
        configured = self.config["topics"]
        required = {
            configured["joint_states"]: "sensor_msgs/msg/JointState",
            configured["odom"]: "nav_msgs/msg/Odometry",
            **expected,
        }
        type_checks = {
            topic: {"expected": expected_type, "actual": list(topic_types.get(topic, ())),
                    "matches": expected_type in topic_types.get(topic, ())}
            for topic, expected_type in required.items()
        }
        publishers = self.other_publishers()
        subscribers = self.subscriber_counts()
        joint = None if self.latest_joint is None else _jsonable(self.latest_joint)
        coordinate = self.offline_coordinate_evidence(tf_timeout_s)
        blockers = list(coordinate["blockers"])
        blockers.extend(
            f"TOPIC_TYPE_MISMATCH:{topic}" for topic, item in type_checks.items()
            if not item["matches"]
        )
        blockers.extend(
            f"OTHER_CONTROL_PUBLISHER:{group}:{count}"
            for group, count in publishers.items() if count != 0
        )
        blockers.extend(
            f"SERVER_CONTROLLER_SUBSCRIBER_MISSING:{group}"
            for group, count in subscribers.items() if count <= 0
        )
        if joint is None:
            blockers.append("JOINT_STATE_NOT_RECEIVED")
        if input_error:
            blockers.append(f"INPUT:{input_error}")
        blockers = list(dict.fromkeys(blockers))
        conditions = not blockers
        return {
            "schema": COMMAND_SCHEMA, "command": "probe", "published_control": False,
            "ros_domain_id": os.getenv("ROS_DOMAIN_ID", "UNSET"),
            "actual_odom_frame": coordinate["actual_odom_frame"],
            "actual_base_frame": coordinate["actual_base_frame"],
            "transform_source": coordinate["transform_source"],
            "world_equals_odom": conditions,
            "topic_types": type_checks, "joint_state": joint,
            "joint_state_raw_name_order_and_position": self.latest_joint_raw,
            "joint_names_expected": list(JOINT_NAMES),
            "base_transform": coordinate["base_transform"],
            "raw_base_transform": coordinate["raw_base_transform"],
            "planarized_virtual_footprint_transform": coordinate[
                "planarized_virtual_footprint_transform"
            ],
            "raw_roll_pitch_yaw": coordinate["raw_roll_pitch_yaw"],
            "virtual_footprint_roll_pitch_yaw": coordinate[
                "virtual_footprint_roll_pitch_yaw"
            ],
            "object_pose_in_odom": None,
            "object_pose_in_virtual_footprint": None,
            "object_local_z_in_virtual_footprint": None,
            "planarization_scope": coordinate["planarization_scope"],
            "odom_tf_comparison": coordinate["odom_tf_comparison"],
            "other_control_publishers": publishers, "server_control_subscribers": subscribers,
            "official_offline_conditions_met": conditions,
            "blockers": blockers,
        }

    @staticmethod
    def _timing_stats(samples: Sequence[tuple[int, int]]) -> dict[str, Any]:
        ordered = sorted(samples)
        arrivals = [item[0] for item in ordered]
        ages = [arrival - stamp for arrival, stamp in ordered]
        gaps = [right - left for left, right in zip(arrivals, arrivals[1:])]
        span = arrivals[-1] - arrivals[0] if len(arrivals) >= 2 else 0
        frequency = ((len(arrivals) - 1) * 1_000_000_000 / span) if span > 0 else None
        return {
            "sample_count": len(samples),
            "frequency_hz": frequency,
            "max_interval_ns": max(gaps) if gaps else None,
            "max_data_age_ns": max(ages) if ages else None,
            "min_data_age_ns": min(ages) if ages else None,
            "clock_order_valid": bool(ages) and min(ages) >= 0,
        }

    def timing(self, duration_s: float, safety_factor: float) -> dict[str, Any]:
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s必须是有限正数")
        if not math.isfinite(safety_factor) or safety_factor < 1.0:
            raise ValueError("safety_factor必须是大于等于1的有限数")
        self.joint_timing_samples.clear()
        self.odom_timing_samples.clear()
        self.tf_timing_samples.clear()
        deadline = self.monotonic() + duration_s
        while self.monotonic() < deadline:
            self.spin_once(min(0.05, max(0.0, deadline - self.monotonic())))
        streams = {
            "joint_state": self._timing_stats(self.joint_timing_samples),
            "odom": self._timing_stats(self.odom_timing_samples),
            "tf_odom_to_base_link": self._timing_stats(self.tf_timing_samples),
        }

        def candidate(stream: str) -> dict[str, Any]:
            stats = streams[stream]
            evidence = (stats["max_interval_ns"], stats["max_data_age_ns"])
            if stats["sample_count"] < 2:
                return {"value_ns": None, "status": "TRIAL_BLOCKED_INSUFFICIENT_SAMPLES"}
            if not stats["clock_order_valid"]:
                return {"value_ns": None, "status": "TRIAL_BLOCKED_CLOCK_DOMAIN_MISMATCH"}
            value = int(math.ceil(safety_factor * max(int(item) for item in evidence if item is not None)))
            return {
                "value_ns": value, "status": "TRIAL_NOT_FROZEN",
                "basis": f"ceil({safety_factor} * max(max_interval_ns,max_data_age_ns))",
                "stream": stream,
            }

        return {
            "schema": COMMAND_SCHEMA, "command": "timing", "published_control": False,
            "duration_s": duration_s, "safety_factor": safety_factor, "streams": streams,
            "trial_candidates": {
                "transform_max_age_ns": candidate("tf_odom_to_base_link"),
                "joint_state_max_age_ns": candidate("joint_state"),
                "object_estimate_max_age_ns": {
                    "value_ns": None, "status": "TRIAL_BLOCKED_NO_OBJECT_ESTIMATE_SAMPLES",
                },
                "planned_context_max_age_ns": {
                    "value_ns": None, "status": "TRIAL_BLOCKED_REQUIRES_PICK_WORKFLOW_DURATION",
                },
                "confirmed_context_max_age_ns": {
                    "value_ns": None, "status": "TRIAL_BLOCKED_REQUIRES_CARRY_PLACE_DURATION",
                },
            },
        }

    def wait_for_instruction(self, timeout_s: float) -> tuple[str, int]:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("instruction timeout必须是有限正数")
        deadline = self.monotonic() + timeout_s
        while self.monotonic() < deadline and self.latest_instruction_raw is None:
            self.spin_once(min(0.05, max(0.0, deadline - self.monotonic())))
        if self.latest_instruction_raw is None or self.latest_instruction_received_ns is None:
            raise RuntimeError("等待/material/instruction完整JSON超时")
        return self.latest_instruction_raw, self.latest_instruction_received_ns

    def receive_instruction_tasks(
        self, timeout_s: float,
    ) -> tuple[str, int, tuple[TaskSpec, ...]]:
        """Receive and parse one complete instruction on this persistent node."""

        raw, received_ns = self.wait_for_instruction(timeout_s)
        return raw, received_ns, tuple(InstructionParser().parse(raw, received_ns))

    def prepare_pick_input(
        self, task_id: int, scene: str, seed: int, fixture_confidence: float,
        timeout_s: float,
    ) -> dict[str, Any]:
        try:
            if os.getenv("ROS_DOMAIN_ID") != "99":
                raise ValueError("prepare-pick-input只允许ROS_DOMAIN_ID=99官方离线域")
            if type(task_id) is not int or task_id < 0:
                raise ValueError("task_id必须是非负整数")
            if type(seed) is not int or seed < 0:
                raise ValueError("seed必须是非负整数")
            if not math.isfinite(fixture_confidence) or not 0.0 <= fixture_confidence <= 1.0:
                raise ValueError("fixture_confidence必须是0到1的显式试验值")
            raw, received_ns, tasks = self.receive_instruction_tasks(timeout_s)
            matches = [task for task in tasks if task.task_id == task_id]
            if len(matches) != 1:
                raise ValueError(f"实时instruction中task-id={task_id}匹配数量={len(matches)}")
            task = matches[0]
            fixture = stage2_fixture(self.config, scene, task.target_color)
            expected_slot = self.config["source_slots"]["task_source_slots"].get(task_id)
            if expected_slot != fixture["source_slot"]:
                raise ValueError(
                    f"task-id={task_id}冻结source slot={expected_slot!r}与scene={scene}不匹配"
                )
            object_id = f"fixture:seed-{seed}:task-{task_id}:{scene}"
            slot_name, center_index = SCENE_FIXTURE_KEYS[scene]
            payload = {
                "source": FIXTURE_SOURCE, "seed": seed, "scene": scene,
                "source_slot": fixture["source_slot"], "expected_object_id": object_id,
                "task": _jsonable(task),
                "object_estimate": {
                    "class_id": task.target_color, "position_xyz": fixture["position_xyz"],
                    "confidence": fixture_confidence, "frame_id": fixture["frame_id"],
                    "timestamp_ns": received_ns, "object_id": object_id,
                    "orientation_xyzw": fixture["orientation_xyzw"],
                    "size_xyz_m": fixture["size_xyz_m"],
                },
                "field_sources": {
                    "source": "tool:stage2_calibration_fixture marker",
                    "seed": "cli:--seed (must match running official Server)",
                    "scene": "cli:--scene",
                    "source_slot": f"config:source_slots.task_source_slots.{task_id}",
                    "expected_object_id": "tool:deterministic_fixture_identity",
                    "task.task_id": "live:/material/instruction parsed by InstructionParser",
                    "task.instruction": "live:/material/instruction parsed by InstructionParser",
                    "task.target_kind": "live:/material/instruction parsed by InstructionParser",
                    "task.target_body": "live:/material/instruction parsed by InstructionParser",
                    "task.target_color": "live:/material/instruction parsed by InstructionParser",
                    "task.place_type": "live:/material/instruction parsed by InstructionParser",
                    "task.place_world_xyz": "live:/material/instruction parsed by InstructionParser",
                    "task.place_frame_id": "contract:InstructionParser official world field semantics",
                    "task.place_radius": "live:/material/instruction parsed by InstructionParser",
                    "task.ref_prop": "live:/material/instruction parsed by InstructionParser",
                    "task.ref_prop_body": "live:/material/instruction parsed by InstructionParser",
                    "task.direction": "live:/material/instruction parsed by InstructionParser",
                    "task.timestamp_ns": "live:instruction_receive_time",
                    "task.valid": "contract:TaskSpec validated true",
                    "task.failure_reason": "contract:TaskSpec valid-task invariant",
                    "object_estimate.class_id": "live:parsed task.target_color",
                    "object_estimate.position_xyz": (
                        f"config:source_slots.slots.{slot_name}.centers[{center_index}]"
                    ),
                    "object_estimate.size_xyz_m": (
                        f"config:perception.estimator_3d.object_local_size_xyz_m.{task.target_color}"
                    ),
                    "object_estimate.orientation_xyzw": (
                        f"derived_from_config:source_slots.slots.{fixture['source_slot']}.yaw_rad"
                    ),
                    "object_estimate.frame_id": "config:source_slots.frame_id",
                    "object_estimate.timestamp_ns": "live:instruction_receive_time_fixture_instantiation",
                    "object_estimate.confidence": "trial:--fixture-confidence",
                    "object_estimate.object_id": "tool:deterministic_fixture_identity",
                    "raw_instruction_json": "live:/material/instruction complete message data",
                },
                "raw_instruction_json": raw,
            }
            return {
                "valid": True, "status": "READY", "payload": payload,
                "blockers": [], "published_control": False,
            }
        except Exception as exc:
            return {
                "valid": False, "status": "BLOCKED", "payload": None,
                "blockers": [str(exc)], "published_control": False,
            }

    def start_publishers(self) -> None:
        if self.publishers:
            raise RuntimeError("标定publisher已经创建")
        topics = {item.group: item.topic for item in MMK2_CONTROLLER_MANIFEST_V1.official_topics}
        for group in ("spine", "left_arm", "right_arm"):
            self.publishers[group] = self.node.create_publisher(
                self.Float64MultiArray, topics[group], 5
            )

    def publish_joint_target(self, position: Sequence[float]) -> None:
        values = _finite_joint_target(position)
        payloads = {
            "spine": values[0:1], "left_arm": values[3:10], "right_arm": values[10:17]
        }
        for group, data in payloads.items():
            message = self.Float64MultiArray()
            message.data = list(data)
            self.publishers[group].publish(message)

    def close(self) -> None:
        for publisher in tuple(self.publishers.values()):
            self.node.destroy_publisher(publisher)
        self.publishers.clear()
        if self.base_publisher is not None:
            self.node.destroy_publisher(self.base_publisher)
            self.base_publisher = None
        self.node.destroy_node()
        if self._owns_context and self.rclpy.ok():
            self.rclpy.shutdown()


def _base_sample(state: BaseState) -> dict[str, Any]:
    return {
        "timestamp_ns": state.timestamp_ns,
        "position_xyz": list(state.position_xyz),
        "yaw": state.yaw,
        "linear_velocity_xyz": list(state.linear_velocity_xyz),
        "angular_velocity_xyz": list(state.angular_velocity_xyz),
    }


def _publish_zero_and_confirm_stop(
    runtime: Any, nav: NavigationConfig, rate_hz: float
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    stable = 0
    deadline = runtime.monotonic() + BASE_STOP_CONFIRMATION_TIMEOUT_S
    while runtime.monotonic() < deadline:
        runtime.publish_base_velocity(0.0, 0.0)
        state = runtime.poll_base_state(1.0 / rate_hz)
        if state is None:
            continue
        linear = math.hypot(*state.linear_velocity_xyz)
        angular = math.hypot(*state.angular_velocity_xyz)
        stopped = (
            linear <= nav.max_settled_linear_speed_mps
            and angular <= nav.max_settled_angular_speed_radps
        )
        stable = stable + 1 if stopped else 0
        evidence.append({**_base_sample(state), "zero_command": True, "stopped": stopped})
        if stable >= nav.settled_required_cycles:
            break
    # 即使已经取得三帧证据，再明确发送三次零速，避免最后一帧后立即退出。
    for _ in range(3):
        runtime.publish_base_velocity(0.0, 0.0)
    return {
        "confirmed_stopped": stable >= nav.settled_required_cycles,
        "stable_frames": stable,
        "required_stable_frames": nav.settled_required_cycles,
        "samples": evidence,
        "zero_commands_after_evidence": 3,
    }


def _arm_reachability_precheck(
    plan: Mapping[str, Any], state: BaseState, goal: NavGoal,
    nav: NavigationConfig,
) -> dict[str, Any]:
    """Report the unmodified planned object pose relative to the measured base."""

    target = plan.get("target_object")
    if not isinstance(target, Mapping):
        raise ValueError("plan缺少target_object，无法执行机械臂可达性前置检查")
    position = target.get("position_xyz")
    orientation = target.get("orientation_xyzw")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("plan.target_object.position_xyz格式无效")
    if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
        raise ValueError("plan.target_object.orientation_xyzw格式无效")
    planar = RigidTransform3D(
        source_frame="virtual_footprint", target_frame=OFFICIAL_ODOM_FRAME,
        translation_xyz=(state.position_xyz[0], state.position_xyz[1], 0.0),
        rotation_xyzw=(
            0.0, 0.0, math.sin(state.yaw / 2.0), math.cos(state.yaw / 2.0)
        ),
        timestamp_ns=state.timestamp_ns, valid=True,
    )
    _transform, diagnostics = transform_object_to_virtual_footprint(
        position, orientation, planar
    )
    actual_pose = diagnostics["object_pose_in_virtual_footprint"]
    actual_position = actual_pose["position_xyz_m"]
    expected = [nav.standoff_m, 0.0, float(position[2])]
    standoff_error = abs(float(actual_position[0]) - expected[0])
    lateral_error = abs(float(actual_position[1]) - expected[1])
    yaw_error = abs(wrap_to_pi(goal.pose_xyyaw[2] - state.yaw))
    planar_error = math.hypot(standoff_error, lateral_error)
    return {
        "actual_object_pose_in_virtual_footprint": actual_pose,
        "expected": expected,
        "standoff_error_m": standoff_error,
        "lateral_error_m": lateral_error,
        "yaw_alignment_error_rad": yaw_error,
        "planar_alignment_error_m": planar_error,
        "position_tolerance_m": nav.position_tolerance_m,
        "yaw_tolerance_rad": nav.yaw_tolerance_rad,
        "meets_calibration_precision": (
            planar_error <= nav.position_tolerance_m
            and yaw_error <= nav.yaw_tolerance_rad
        ),
        "object_coordinates_modified": False,
        "source": "plan.target_object transformed by measured odom base pose",
    }


def _evaluate_cached_feedback(
    state: Any | None, age_ns: int | None, previous_timestamp_ns: int | None,
    maximum_age_ns: int, label: str, validation_error: str | None = None,
) -> dict[str, Any]:
    """Classify one control tick using a persistent subscription's latest cache."""

    if validation_error:
        return {
            "valid": False, "classification": "INVALID", "reused": False,
            "failure_reason": f"{label}数据无效：{validation_error}",
        }
    if state is None or age_ns is None:
        return {
            "valid": False, "classification": "NEVER_RECEIVED", "reused": False,
            "failure_reason": f"执行中从未收到{label}",
        }
    timestamp_ns = getattr(state, "timestamp_ns", None)
    if type(timestamp_ns) is not int or timestamp_ns < 0:
        return {
            "valid": False, "classification": "INVALID", "reused": False,
            "failure_reason": f"{label}消息时间戳非法：{timestamp_ns!r}",
        }
    if previous_timestamp_ns is not None and timestamp_ns < previous_timestamp_ns:
        return {
            "valid": False, "classification": "INVALID", "reused": False,
            "failure_reason": (
                f"{label}消息时间戳倒退：previous={previous_timestamp_ns},"
                f"current={timestamp_ns}"
            ),
        }
    if age_ns < 0 or age_ns > maximum_age_ns:
        return {
            "valid": False, "classification": "STALE", "reused": False,
            "failure_reason": (
                f"执行中{label}真正过期：age_ns={age_ns},"
                f"threshold_ns={maximum_age_ns}"
            ),
        }
    return {
        "valid": True, "classification": "", "failure_reason": "",
        "reused": timestamp_ns == previous_timestamp_ns,
    }


def execute_base_stand(
    plan: Mapping[str, Any], config: Mapping[str, Any],
    official_offline_simulation: bool, confirmation: str, note: str,
    runtime: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": COMMAND_SCHEMA, "command": "execute-base-stand",
        "valid": False, "execution_success": False, "published_control": False,
        "interrupted": False, "timed_out": False, "failure_reason": "",
        "seed": plan.get("seed"), "scene": plan.get("scene"),
        "user_note": note, "odom_series": [], "command_series": [],
        "stop_evidence": None, "planned_goal": plan.get("goal"),
        "navigation_parameters": plan.get("navigation_parameters"),
        "control_rate_hz": BASE_CONTROL_RATE_HZ,
        "control_tick_count": 0,
        "odom_received_count": 0,
        "odom_reused_tick_count": 0,
        "latest_odom_age_ns_by_tick": [],
        "max_observed_odom_age_ns": None,
        "odom_stale_threshold_ns": 150_000_000,
        "failure_classification": "",
        "publisher_conflict_evidence": [],
        "arm_reachability_precheck": None,
    }
    publisher_started = False
    nav: NavigationConfig | None = None
    try:
        if not official_offline_simulation:
            raise ValueError("必须显式指定--official-offline-simulation")
        if confirmation != OFFICIAL_BASE_CONFIRMATION:
            raise ValueError(f"--confirm必须严格为{OFFICIAL_BASE_CONFIRMATION}")
        if os.getenv("ROS_DOMAIN_ID") != "99":
            raise ValueError("execute-base-stand只允许ROS_DOMAIN_ID=99")
        goal, nav = _validate_base_plan(plan, config)
        result["odom_stale_threshold_ns"] = nav.odom_max_age_ns
        base_graph = getattr(runtime, "base_publisher_exclusivity_evidence", None)
        if callable(base_graph):
            evidence = base_graph("before_create", False)
            result["publisher_conflict_evidence"].append(evidence)
            if not evidence["valid"]:
                raise ValueError("/cmd_vel外部Publisher、DDS残留或subscriber身份无效")
        else:
            publishers, subscribers = runtime.base_graph_counts()
            if publishers != 0:
                raise ValueError(f"/cmd_vel存在其他publisher：{publishers}")
            if subscribers != 1:
                raise ValueError(f"/cmd_vel subscriber必须恰好为1，实际={subscribers}")
        initial = runtime.wait_for_base_state(3.0)
        if initial.frame_id != "odom":
            raise ValueError("Odom frame必须归一化为odom")
        if not runtime.odom_fresh(nav.odom_max_age_ns / 1_000_000_000):
            result["failure_classification"] = "STALE"
            raise ValueError("Odom过期，禁止创建/cmd_vel publisher")
        result["start_base_state"] = _jsonable(initial)
        runtime.start_base_publisher()
        publisher_started = True
        result["published_control"] = True
        if callable(base_graph):
            evidence = base_graph("after_create", True)
            result["publisher_conflict_evidence"].append(evidence)
            if not evidence["valid"]:
                result["failure_classification"] = "PUBLISHER_CONFLICT"
                raise ValueError("/cmd_vel自身endpoint未收敛或出现外部Publisher")
        controller = NavigationController(nav)
        started = runtime.monotonic()
        previous_odom_timestamp_ns = initial.timestamp_ns
        while True:
            if runtime.monotonic() - started >= nav.goal_timeout_ns / 1_000_000_000:
                result["timed_out"] = True
                result["failure_classification"] = "TIMEOUT"
                result["failure_reason"] = "execute-base-stand总超时30s"
                break
            if callable(base_graph):
                graph = base_graph("control_tick", True)
                graph_valid = graph["valid"]
                publishers = graph["self_count"] + graph["external_count"]
                subscribers = graph["subscriber_count"]
                if not graph_valid:
                    result["publisher_conflict_evidence"].append(graph)
            else:
                publishers, subscribers = runtime.base_graph_counts()
                graph_valid = publishers == 1 and subscribers == 1
            if not graph_valid:
                result["failure_reason"] = (
                    f"执行中ROS图身份变化：publishers={publishers},subscribers={subscribers}"
                )
                result["failure_classification"] = "ROS_GRAPH_CHANGED"
                break
            runtime.spin_control_period(1.0 / BASE_CONTROL_RATE_HZ)
            result["control_tick_count"] += 1
            state = runtime.latest_base_state()
            age_ns = runtime.latest_odom_age_ns()
            result["latest_odom_age_ns_by_tick"].append(age_ns)
            cached = _evaluate_cached_feedback(
                state, age_ns, previous_odom_timestamp_ns,
                nav.odom_max_age_ns, "Odom",
            )
            if age_ns is not None:
                maximum_age = result["max_observed_odom_age_ns"]
                result["max_observed_odom_age_ns"] = (
                    age_ns if maximum_age is None else max(maximum_age, age_ns)
                )
            if not cached["valid"]:
                result["failure_classification"] = cached["classification"]
                result["failure_reason"] = cached["failure_reason"]
                break
            assert state is not None
            if cached["reused"]:
                result["odom_reused_tick_count"] += 1
            else:
                previous_odom_timestamp_ns = state.timestamp_ns
            result["arm_reachability_precheck"] = _arm_reachability_precheck(
                plan, state, goal, nav
            )
            command, status = controller.update(state, goal, state.timestamp_ns)
            if not command.valid:
                result["failure_reason"] = status.failure_reason or "导航控制器返回无效命令"
                result["failure_classification"] = "CONTROLLER_REJECTED"
                break
            if abs(command.v) > nav.max_abs_v_mps or abs(command.w) > nav.max_abs_w_radps:
                result["failure_reason"] = "NavigationController输出超过授权速度上限"
                result["failure_classification"] = "SPEED_LIMIT_VIOLATION"
                break
            runtime.publish_base_velocity(command.v, command.w)
            controller.record_control_result(state.timestamp_ns, True)
            result["odom_series"].append(_base_sample(state))
            result["command_series"].append({
                "timestamp_ns": state.timestamp_ns, "v_mps": command.v,
                "w_radps": command.w, "state": status.state,
                "distance_error_m": status.distance_error,
                "yaw_error_rad": status.yaw_error,
            })
            result["final_distance_error_m"] = status.distance_error
            result["final_yaw_error_rad"] = status.yaw_error
            if status.success:
                if not result["arm_reachability_precheck"][
                    "meets_calibration_precision"
                ]:
                    result["failure_reason"] = (
                        "导航控制器报告到位但机械臂可达性前置检查未满足标定精度"
                    )
                    result["failure_classification"] = "REACHABILITY_PRECHECK_FAILED"
                    break
                result["valid"] = True
                result["execution_success"] = True
                break
    except KeyboardInterrupt:
        result["interrupted"] = True
        result["failure_classification"] = "INTERRUPTED"
        result["failure_reason"] = "用户Ctrl-C中断"
    except Exception as exc:
        result["failure_reason"] = str(exc)
        if not result["failure_classification"]:
            result["failure_classification"] = (
                "NEVER_RECEIVED" if "等待Odom超时" in str(exc) else "OTHER_EXCEPTION"
            )
    finally:
        result["odom_received_count"] = int(getattr(runtime, "odom_received_count", 0))
        if publisher_started and nav is not None:
            try:
                stop = _publish_zero_and_confirm_stop(runtime, nav, BASE_CONTROL_RATE_HZ)
                result["stop_evidence"] = stop
                if not stop["confirmed_stopped"]:
                    result["valid"] = False
                    result["execution_success"] = False
                    suffix = "零速发布后未取得连续3帧停稳Odom证据"
                    result["failure_reason"] = (
                        f"{result['failure_reason']};{suffix}" if result["failure_reason"] else suffix
                    )
            except Exception as exc:
                result["valid"] = False
                result["execution_success"] = False
                result["failure_reason"] = (
                    f"{result['failure_reason']};安全零速失败:{exc}"
                    if result["failure_reason"] else f"安全零速失败:{exc}"
                )
    return result


def write_base_execution_log(result: Mapping[str, Any], root: str | Path) -> Path:
    seed = str(result.get("seed", "unknown"))
    scene = str(result.get("scene", "unknown"))
    timestamp = time.time_ns()
    path = Path(root) / seed / scene / f"base-stand-{timestamp}.json"
    _safe_output(str(path), _render(result, "json"))
    return path


def probe_environment(
    config: Mapping[str, Any], runtime: Any | None = None, tf_timeout_s: float = 5.0
) -> dict[str, Any]:
    owned = runtime is None
    runtime = runtime or RosCalibrationRuntime(config)
    try:
        return runtime.probe(tf_timeout_s=tf_timeout_s)
    finally:
        if owned:
            runtime.close()


def inspect_environment(config: Mapping[str, Any], capture: Mapping[str, Any] | None = None,
                        topic_output: str | None = None, now_ns: int | None = None,
                        check_kdl: bool = True) -> dict[str, Any]:
    topics: list[str] = []
    topic_error = ""
    if topic_output is None:
        try:
            completed = subprocess.run(
                ["ros2", "topic", "list"], check=False, capture_output=True, text=True, timeout=5
            )
            topics = completed.stdout.splitlines()
            topic_error = completed.stderr.strip() if completed.returncode else ""
        except (OSError, subprocess.TimeoutExpired) as exc:
            topic_error = str(exc)
    else:
        topics = topic_output.splitlines()
    topic_config = config["topics"]
    server_topics = [topic_config["instruction"], topic_config["joint_states"],
                     topic_config["odom"], topic_config["referee_taskinfo"]]
    missing_nulls = [name for name, value in config["arm_planning"].items() if value is None]
    missing_nulls += [f"arm_execution.{name}" for name, value in config["arm_execution"].items() if value is None]
    joint_fresh: bool | str = "UNKNOWN_NO_CAPTURE"
    object_present: bool | str = topic_config["object_estimates"] in topics
    if capture is not None:
        now = int(capture.get("now_ns", now_ns if now_ns is not None else time.time_ns()))
        try:
            state = _joints(capture["joint_state"])
            max_age = capture.get("trial_parameters", {}).get(
                "joint_state_max_age_ns", config["arm_planning"]["joint_state_max_age_ns"]
            )
            joint_fresh = bool(max_age is not None and 0 <= now - state.timestamp_ns <= max_age and state.valid)
        except Exception:
            joint_fresh = False
        object_present = "object_estimate" in capture
    execution_nulls = [name for name, value in config["arm_execution"].items() if value is None]
    execution_constructible = not execution_nulls
    execution_reason = "" if not execution_nulls else f"ArmExecutionConfig仍有null字段：{execution_nulls}"
    planning = planning_config(config, {})
    planner_constructible = True
    kdl_self_check: bool | str = "NOT_REQUESTED"
    kdl_reason = ""
    try:
        adapter = OfficialKDLAdapter(config["official"].get("root", ""),
                                     config["official"]["kdl_module"])
        ArmPlanner(adapter, planning)
    except Exception as exc:
        planner_constructible = False
        kdl_reason = str(exc)
    else:
        if check_kdl:
            try:
                adapter.self_check()
                kdl_self_check = True
            except Exception as exc:
                kdl_self_check = False
                kdl_reason = str(exc)
    server_evidence = all(item in topics for item in server_topics)
    return {
        "schema": COMMAND_SCHEMA, "command": "inspect", "mode": "inspect-only",
        "published_control": False, "ros_domain_id": os.getenv("ROS_DOMAIN_ID", "UNSET"),
        "official_offline_simulation": "UNCONFIRMED",
        "official_offline_evidence": "required server topics present" if server_evidence else "insufficient",
        "server_original_topics": {name: name in topics for name in server_topics},
        "object_estimate_3d_present": object_present, "joint_state_fresh": joint_fresh,
        "safe_gates": {
            "arm_planning_enabled": config["arm_planning"]["enabled"],
            "observe_only": config["control"]["observe_only"],
            "enable_official_publish": config["control"]["enable_official_publish"],
            "simulation_only": config["control"]["simulation_only"],
        },
        "planner_constructible_without_self_check": planner_constructible,
        "kdl_self_check": kdl_self_check,
        "kdl_reason": kdl_reason,
        "executor_constructible": execution_constructible,
        "executor_reason": execution_reason,
        "null_configuration_fields": missing_nulls,
        "available_test_level": "PLAN_ONLY_WITH_CAPTURE_AND_EXPLICIT_TRIAL_PARAMETERS"
        if capture else "INSPECT_ONLY",
        "topic_query_error": topic_error,
        "single_stage_execution": "BLOCKED",
        "pick_visual_verification": "BLOCKED", "place_visual_verification": "BLOCKED",
        "full_end_to_end": "BLOCKED",
    }


def _finite_joint_target(values: Sequence[Any]) -> tuple[float, ...]:
    if len(values) != 17:
        raise ValueError("关节目标必须严格包含17项")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"关节目标第{index}项必须是有限实数")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"关节目标第{index}项不是有限数")
        result.append(number)
    return tuple(result)


def _transition_stage_data(
    plan: Mapping[str, Any], stage: str
) -> tuple[tuple[float, ...], tuple[float, ...], Mapping[str, Any], str | None]:
    if stage not in TRANSITION_STAGES:
        raise ValueError(f"只允许手动过渡阶段{TRANSITION_STAGES}")
    if plan.get("plan_artifact_valid") is not True:
        raise ValueError("过渡plan_artifact_valid必须严格为true")
    if plan.get("automatic_execution_ready") is not False:
        raise ValueError("过渡计划必须保持automatic_execution_ready=false")
    if plan.get("single_stage_execution_ready") is not True:
        raise ValueError("过渡计划缺少single_stage_execution_ready=true")
    transition = plan.get("transition_plan")
    if not isinstance(transition, Mapping):
        raise ValueError("计划缺少transition_plan")
    segments = transition.get("segments")
    if not isinstance(segments, list):
        raise ValueError("transition_plan.segments必须是数组")
    index = TRANSITION_STAGES.index(stage)
    if index >= len(segments):
        raise ValueError(f"计划不存在{stage}，禁止跳段或猜测目标")
    segment = segments[index]
    if not isinstance(segment, Mapping) or segment.get("stage") != stage:
        raise ValueError(f"计划{stage}层级或stage标记无效")
    if index == 0:
        start_raw = plan.get("start_joint_state")
        if not isinstance(start_raw, Mapping) or "position" not in start_raw:
            raise ValueError("transition-1缺少真实start_joint_state.position")
        start = _finite_joint_target(start_raw["position"])
    else:
        previous = segments[index - 1]
        if not isinstance(previous, Mapping) or "joint_position" not in previous:
            raise ValueError(f"{stage}缺少前一段终点")
        start = _finite_joint_target(previous["joint_position"])
    target = _finite_joint_target(segment.get("joint_position", ()))
    next_stage: str | None
    if index + 1 < len(segments) and index + 1 < len(TRANSITION_STAGES):
        next_stage = TRANSITION_STAGES[index + 1]
    elif stage == "transition-3" and plan.get("transition_3_reaches_pregrasp") is True:
        next_stage = "grasp-open"
    else:
        next_stage = "pregrasp"
    return start, target, segment, next_stage


def _stage_target(
    plan: Mapping[str, Any], stage: str, *, calibration_sequence: bool = False
) -> tuple[float, ...]:
    if not plan.get("valid") or plan.get("published_control") is not False:
        raise ValueError("plan-only输出无效或缺少published_control=false")
    if stage in TRANSITION_STAGES:
        _start, target, _segment, _next = _transition_stage_data(plan, stage)
        return target
    if plan.get("automatic_execution_ready") is False and not calibration_sequence:
        raise ValueError(
            "过渡计划automatic_execution_ready=false；只允许显式transition单段执行"
        )
    trial = plan.get("trial_parameters", {})
    if isinstance(trial, Mapping):
        arm_delta = trial.get("max_arm_waypoint_delta_rad")
        if isinstance(arm_delta, (int, float)) and not isinstance(arm_delta, bool):
            if float(arm_delta) > CALIBRATION_ARM_STEP_LIMIT_RAD:
                raise ValueError("拒绝执行：max_arm_waypoint_delta_rad不得超过1.0")
    analysis = plan.get("calibration_analysis")
    if (
        isinstance(analysis, Mapping) and analysis.get("transition_required")
        and not calibration_sequence
    ):
        raise ValueError("拒绝执行：calibration-only过渡路点仅供规划和视觉核查")
    if plan.get("source") != FIXTURE_SOURCE:
        raise ValueError("plan文件不是stage2_calibration_fixture输入生成")
    waypoints = plan.get("waypoints")
    if not isinstance(waypoints, list):
        raise ValueError("plan文件缺少waypoints")
    pick_indices = {"pregrasp": 0, "grasp-open": 1, "grasp-close": 2,
                    "short-lift": 3, "retreat": 4}
    place_indices = {
        "preplace": 0, "lower": 1, "release": 2, "post-release-retreat": 3
    }
    if stage == "return-start":
        start = plan.get("start_joint_state")
        if not isinstance(start, Mapping):
            raise ValueError("plan文件缺少start_joint_state")
        return _finite_joint_target(start["position"])
    indices = pick_indices if plan.get("command") == "plan-pick" else place_indices
    if stage not in indices or indices[stage] >= len(waypoints):
        raise ValueError(f"阶段{stage}不属于该plan文件")
    target = _finite_joint_target(waypoints[indices[stage]]["joint_position"])
    if stage == "grasp-open" and (target[9] != 1.0 or target[16] != 1.0):
        raise ValueError("grasp-open目标必须严格使用左右夹爪open=1.0")
    if stage == "grasp-close" and (target[9] != 0.1 or target[16] != 0.1):
        raise ValueError("grasp-close目标必须严格使用左右夹爪closed=0.1")
    if stage == "release" and (target[9] != 1.0 or target[16] != 1.0):
        raise ValueError("release目标必须严格使用左右夹爪open=1.0")
    if stage == "lower" and (target[9] != 0.1 or target[16] != 0.1):
        raise ValueError("lower目标必须保持左右夹爪closed=0.1")
    return target


def _planned_stage_duration(plan: Mapping[str, Any], stage: str) -> float | None:
    if stage in TRANSITION_STAGES:
        _start, _target, segment, _next = _transition_stage_data(plan, stage)
        value = segment.get("duration_s")
    else:
        waypoints = plan.get("waypoints")
        indices = (
            {"pregrasp": 0, "grasp-open": 1, "grasp-close": 2,
             "short-lift": 3, "retreat": 4}
            if plan.get("command") == "plan-pick" else
            {"preplace": 0, "lower": 1, "release": 2,
             "post-release-retreat": 3}
        )
        if not isinstance(waypoints, list) or stage not in indices:
            return None
        value = waypoints[indices[stage]].get("duration_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _expected_stage_start(
    plan: Mapping[str, Any], stage: str
) -> tuple[float, ...] | None:
    if stage in TRANSITION_STAGES:
        start, _target, _segment, _next = _transition_stage_data(plan, stage)
        return start
    previous = {
        "pregrasp": None, "grasp-open": "pregrasp",
        "grasp-close": "grasp-open", "short-lift": "grasp-close",
        "retreat": "short-lift", "preplace": None, "lower": "preplace",
        "release": "lower", "post-release-retreat": "release",
    }
    if stage not in previous or stage == "return-start":
        return None
    prior_stage = previous[stage]
    if prior_stage is None:
        raw = plan.get("start_joint_state")
        if not isinstance(raw, Mapping):
            raise ValueError("计划缺少首阶段start_joint_state")
        return _finite_joint_target(raw.get("position", ()))
    return _stage_target(plan, prior_stage, calibration_sequence=True)


def _execution_parameters(raw: Mapping[str, Any]) -> dict[str, Any]:
    required_positive = (
        "max_slide_velocity_m_s", "max_arm_velocity_rad_s",
        "max_gripper_velocity_per_s", "control_rate_hz", "timeout_s",
        "feedback_max_age_s", "slide_tolerance_m", "arm_tolerance_rad",
        "gripper_tolerance",
    )
    result: dict[str, Any] = {}
    for name in required_positive:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}必须显式提供有限正数")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name}必须显式提供有限正数")
        result[name] = number
    settle = raw.get("settle_cycles")
    if type(settle) is not int or settle <= 0:
        raise ValueError("settle_cycles必须显式提供正整数")
    result["settle_cycles"] = settle
    return result


def _validate_transition_execution_parameters(
    parameters: Mapping[str, Any], segment: Mapping[str, Any]
) -> None:
    exact = {
        "control_rate_hz": 24.0,
        "feedback_max_age_s": 0.131175150,
        "arm_tolerance_rad": 0.01,
        "slide_tolerance_m": 0.01,
        "gripper_tolerance": 0.02,
        "settle_cycles": 3,
    }
    for name, required in exact.items():
        if parameters.get(name) != required:
            raise ValueError(f"过渡单段执行要求{name}={required}")
    maximums = {
        "max_arm_velocity_rad_s": CALIBRATION_ARM_SPEED_LIMIT_RAD_S,
        "max_slide_velocity_m_s": CALIBRATION_SLIDE_SPEED_LIMIT_M_S,
        "max_gripper_velocity_per_s": CALIBRATION_ARM_SPEED_LIMIT_RAD_S,
    }
    for name, maximum in maximums.items():
        if float(parameters[name]) > maximum + 1e-12:
            raise ValueError(f"过渡单段执行{name}不得超过{maximum}")
    duration = segment.get("duration_s")
    if (
        isinstance(duration, bool) or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration)) or float(duration) <= 0.0
    ):
        raise ValueError("过渡段duration_s必须是有限正数")


def _validate_transition_segment(
    segment: Mapping[str, Any], start: Sequence[float], target: Sequence[float],
    lower: Sequence[float], upper: Sequence[float],
) -> None:
    for name in ("joint_limits_ok", "step_limits_ok", "speed_limits_ok"):
        if segment.get(name) is not True:
            raise ValueError(f"过渡段{name}未通过，拒绝执行")
    fk = segment.get("fk")
    if (
        not isinstance(fk, Mapping) or fk.get("success") is not True
        or not isinstance(fk.get("left_end_position_xyz_m"), list)
        or not isinstance(fk.get("right_end_position_xyz_m"), list)
    ):
        raise ValueError("过渡段FK未通过或缺少左右末端位置")
    if target[9] != 1.0 or target[16] != 1.0:
        raise ValueError("过渡段左右夹爪必须严格保持open=1.0")
    controlled = (0, *range(3, 17))
    if any(not lower[index] <= target[index] <= upper[index] for index in controlled):
        raise ValueError("过渡段目标超出controller manifest/action_mux安全限位")
    duration = float(segment["duration_s"])
    arm_max = max(
        abs(target[index] - start[index])
        for index in (*range(3, 9), *range(10, 16))
    )
    slide_delta = abs(target[0] - start[0])
    gripper_delta = max(abs(target[index] - start[index]) for index in (9, 16))
    if arm_max > CALIBRATION_ARM_STEP_LIMIT_RAD + 1e-12:
        raise ValueError("过渡段机械臂变化超过1.0rad")
    if slide_delta > CALIBRATION_SLIDE_STEP_LIMIT_M + 1e-12:
        raise ValueError("过渡段slide变化超过0.20m")
    if arm_max > CALIBRATION_ARM_SPEED_LIMIT_RAD_S * duration + 1e-12:
        raise ValueError("过渡段机械臂计划速度超过0.6rad/s")
    if slide_delta > CALIBRATION_SLIDE_SPEED_LIMIT_M_S * duration + 1e-12:
        raise ValueError("过渡段slide计划速度超过0.15m/s")
    if gripper_delta > CALIBRATION_ARM_SPEED_LIMIT_RAD_S * duration + 1e-12:
        raise ValueError("过渡段夹爪计划速度超过0.6/s")


def _interpolated_target(
    start: tuple[float, ...], target: tuple[float, ...], elapsed_s: float,
    parameters: Mapping[str, Any],
) -> tuple[float, ...]:
    result = list(start)
    for index in (0, *range(3, 17)):
        velocity = (
            parameters["max_slide_velocity_m_s"] if index == 0 else
            parameters["max_gripper_velocity_per_s"] if index in (9, 16) else
            parameters["max_arm_velocity_rad_s"]
        )
        delta = target[index] - start[index]
        limit = velocity * max(0.0, elapsed_s)
        result[index] = (
            target[index] if abs(delta) <= limit
            else start[index] + math.copysign(limit, delta)
        )
    return tuple(result)


def _target_reached(actual: Sequence[float], target: Sequence[float], parameters: Mapping[str, Any]) -> bool:
    for index in (0, *range(3, 17)):
        tolerance = (
            parameters["slide_tolerance_m"] if index == 0 else
            parameters["gripper_tolerance"] if index in (9, 16) else
            parameters["arm_tolerance_rad"]
        )
        if abs(float(actual[index]) - float(target[index])) > tolerance:
            return False
    return True


def _joint_error(actual: Sequence[float], target: Sequence[float]) -> float:
    return max(abs(float(actual[index]) - float(target[index])) for index in (0, *range(3, 17)))


def _start_match_diagnostics(
    actual: Sequence[float], expected: Sequence[float], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    errors = {
        name: float(actual[index]) - float(expected[index])
        for index, name in enumerate(JOINT_NAMES)
    }
    arm_indices = (*range(3, 9), *range(10, 16))
    arm_index = max(arm_indices, key=lambda index: abs(errors[JOINT_NAMES[index]]))
    grip_index = max((9, 16), key=lambda index: abs(errors[JOINT_NAMES[index]]))
    result = {
        "signed_error_by_joint": errors,
        "absolute_error_by_joint": {name: abs(value) for name, value in errors.items()},
        "max_slide_error_m": abs(errors[JOINT_NAMES[0]]),
        "max_arm_error_rad": abs(errors[JOINT_NAMES[arm_index]]),
        "max_arm_error_joint": JOINT_NAMES[arm_index],
        "max_gripper_error": abs(errors[JOINT_NAMES[grip_index]]),
        "max_gripper_error_side": "left" if grip_index == 9 else "right",
        "tolerances": {
            "slide_tolerance_m": parameters["slide_tolerance_m"],
            "arm_tolerance_rad": parameters["arm_tolerance_rad"],
            "gripper_tolerance": parameters["gripper_tolerance"],
        },
        "tolerance_source": "explicit execute CLI parameters",
    }
    result["matches"] = (
        result["max_slide_error_m"] <= parameters["slide_tolerance_m"]
        and result["max_arm_error_rad"] <= parameters["arm_tolerance_rad"]
        and result["max_gripper_error"] <= parameters["gripper_tolerance"]
    )
    return result


def _stage_timing_diagnostics(
    start: Sequence[float], target: Sequence[float], parameters: Mapping[str, Any],
    planned_duration_s: float | None,
) -> dict[str, Any]:
    slide_time = abs(float(target[0]) - float(start[0])) / parameters[
        "max_slide_velocity_m_s"
    ]
    arm_time = max(
        abs(float(target[index]) - float(start[index]))
        for index in (*range(3, 9), *range(10, 16))
    ) / parameters["max_arm_velocity_rad_s"]
    gripper_time = max(
        abs(float(target[index]) - float(start[index])) for index in (9, 16)
    ) / parameters["max_gripper_velocity_per_s"]
    theoretical = max(slide_time, arm_time, gripper_time)
    motion_time = max(theoretical, float(planned_duration_s or 0.0))
    settle_time = parameters["settle_cycles"] / parameters["control_rate_hz"]
    safety_margin = parameters["feedback_max_age_s"] + 0.5
    minimum_timeout = motion_time + settle_time + safety_margin
    return {
        "planned_duration_s": planned_duration_s,
        "minimum_slide_motion_s": slide_time,
        "minimum_arm_motion_s": arm_time,
        "minimum_gripper_motion_s": gripper_time,
        "minimum_motion_s": theoretical,
        "control_period_s": 1.0 / parameters["control_rate_hz"],
        "minimum_interpolation_ticks": math.ceil(
            theoretical * parameters["control_rate_hz"]
        ),
        "settle_extra_s": settle_time,
        "safety_margin_s": safety_margin,
        "minimum_timeout_s": minimum_timeout,
        "configured_timeout_s": parameters["timeout_s"],
        "timeout_valid": parameters["timeout_s"] + 1e-12 >= minimum_timeout,
    }


def _publisher_evidence(runtime: Any, phase: str, started: bool) -> dict[str, Any]:
    waiter = getattr(runtime, "wait_for_publisher_exclusivity", None)
    if callable(waiter):
        return waiter(phase, started, 0.75)
    counts = runtime.other_publishers()
    expected = {
        "spine": 1, "head": 0, "left_arm": 1, "right_arm": 1
    } if started else {group: 0 for group in ARM_TOPIC_GROUPS}
    valid = all(type(counts.get(group)) is int and counts[group] == count
                for group, count in expected.items())
    return {
        "phase": phase, "valid": valid, "converged": valid,
        "legacy_count_fallback": True, "counts": dict(counts),
        "expected_counts": expected,
    }


def _hold_and_confirm(
    runtime: Any, hold: RobotJointState, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    stable = 0
    previous = hold.timestamp_ns
    latest_state = hold
    deadline = runtime.monotonic() + max(
        1.0, parameters["feedback_max_age_s"] * 4.0
    )
    while runtime.monotonic() < deadline and stable < parameters["settle_cycles"]:
        try:
            runtime.publish_joint_target(hold.position)
            runtime.spin_control_period(1.0 / parameters["control_rate_hz"])
        except (Exception, KeyboardInterrupt) as exc:  # safe-stop must still return a log
            evidence.append({
                "feedback": None, "valid": False,
                "failure_reason": f"保持发布或反馈异常:{type(exc).__name__}:{exc}",
            })
            break
        state = runtime.latest_joint_state()
        age_ns = runtime.latest_joint_age_ns()
        cached = _evaluate_cached_feedback(
            state, age_ns, previous,
            int(parameters["feedback_max_age_s"] * 1_000_000_000),
            "JointState", getattr(runtime, "latest_joint_validation_error", None),
        )
        sample = {
            "feedback": None if state is None else _jsonable(state),
            "feedback_age_s": None if age_ns is None else age_ns / 1_000_000_000,
            "cache_reused": cached.get("reused", False),
            "valid": cached["valid"], "failure_reason": cached["failure_reason"],
        }
        evidence.append(sample)
        if not cached["valid"]:
            break
        assert state is not None
        latest_state = state
        if state.timestamp_ns != previous:
            previous = state.timestamp_ns
            stable = stable + 1 if _target_reached(
                state.position, hold.position, parameters
            ) else 0
    confirmed = stable >= parameters["settle_cycles"]
    return {
        "status": "HOLD_CONFIRMED" if confirmed else "STOP_UNCONFIRMED",
        "confirmed": confirmed, "hold_joint_position": list(hold.position),
        "final_joint_state": _jsonable(latest_state),
        "required_new_feedback_frames": parameters["settle_cycles"],
        "confirmed_new_feedback_frames": stable, "samples": evidence,
    }


def execute_one_stage(
    plan: Mapping[str, Any], stage: str | Sequence[str], official_offline_simulation: bool,
    confirm: str, parameters: Mapping[str, Any], note: str, runtime: Any,
    expected_seed: int | None = None, expected_scene: str | None = None,
    *, publishers_already_started: bool = False, skip_probe: bool = False,
    expected_start_override: Sequence[float] | None = None,
    calibration_sequence: bool = False, hold_on_success: bool = True,
) -> dict[str, Any]:
    """执行一个标定阶段；所有发布均经注入的独立标定runtime完成。"""

    stages = [stage] if isinstance(stage, str) else list(stage)
    base = {
        "schema": COMMAND_SCHEMA, "command": "execute-one-stage",
        "official_offline_simulation_asserted": official_offline_simulation,
        "stage": stages[0] if len(stages) == 1 else stages,
        "seed": plan.get("seed"), "scene": plan.get("scene"),
        "source": plan.get("source"), "trial_parameters": _jsonable(parameters),
        "plan_command": plan.get("command"), "trajectory_id": plan.get("trajectory_id"),
        "task": plan.get("task"),
        "object": plan.get("object_estimate", plan.get("grasp_context")),
        "published_control": False, "interrupted": False, "timed_out": False,
        "execution_success": False, "failure_reason": "", "user_note": note,
        "start_joint_state": {"available": False, "reason": "PRECHECK_NOT_REACHED"},
        "target_joint_state": {"available": False, "reason": "PLAN_NOT_VALIDATED"},
        "expected_start_joint_state": {
            "available": False, "reason": "EXPECTED_START_NOT_RESOLVED"
        }, "start_error_by_joint": {},
        "start_match_diagnostics": None, "timing_validation": None,
        "actual_joint_state_series": [], "actual_joint_state": None,
        "final_joint_state": {"available": False, "reason": "NO_VALID_FEEDBACK"},
        "final_joint_state_received_at_ns": None,
        "final_joint_error_by_joint": {}, "final_error_by_joint": {},
        "maximum_error": None, "stable_frames": 0, "settle_time_s": None,
        "gripper_calibration_note": GRIPPER_NOTE,
        "stable_grip_verified": False,
        "status": "BLOCKED",
        "reached_target": False,
        "final_max_joint_error": None,
        "settled_cycles": 0,
        "next_stage": None,
        "safe_stop_mode": None,
        "control_tick_count": 0,
        "feedback_received_count": 0,
        "feedback_reused_tick_count": 0,
        "feedback_age_s_by_tick": [],
        "max_feedback_age_s": None,
        "failure_classification": "",
        "publisher_conflict_evidence": [],
        "hold_evidence": None,
        "stop_status": "NOT_REQUESTED",
        "valid": False,
    }
    try:
        if not official_offline_simulation:
            raise ValueError("拒绝：必须显式指定--official-offline-simulation")
        if os.getenv("ROS_DOMAIN_ID") != "99":
            raise ValueError("拒绝：ROS_DOMAIN_ID必须严格为99")
        if len(stages) != 1 or stages[0] not in (*STAGES, *TRANSITION_STAGES):
            raise ValueError("拒绝：一次必须且只能指定一个允许阶段")
        if confirm != OFFICIAL_SIM_CONFIRMATION:
            raise ValueError(f"拒绝：--confirm必须严格为{OFFICIAL_SIM_CONFIRMATION}")
        if plan.get("schema") != COMMAND_SCHEMA:
            base["failure_classification"] = "PLAN_SCHEMA_INVALID"
            raise ValueError("计划schema不匹配当前标定工具")
        if plan.get("execution_contract_version") != EXECUTION_CONTRACT_VERSION:
            base["failure_classification"] = "OLD_PLAN_EXECUTION_CONTRACT"
            raise ValueError("OLD_PLAN_EXECUTION_CONTRACT:必须重新生成计划")
        transition_stage = stages[0] in TRANSITION_STAGES
        transition_start = None
        transition_segment = None
        planned_next_stage = None
        if transition_stage:
            if type(expected_seed) is not int or expected_seed != plan.get("seed"):
                raise ValueError("过渡单段执行要求--expected-seed与plan.seed严格匹配")
            if not isinstance(expected_scene, str) or expected_scene != plan.get("scene"):
                raise ValueError("过渡单段执行要求--expected-scene与plan.scene严格匹配")
            transition_start, target, transition_segment, planned_next_stage = (
                _transition_stage_data(plan, stages[0])
            )
        else:
            target = _stage_target(
                plan, stages[0], calibration_sequence=calibration_sequence
            )
        params = _execution_parameters(parameters)
        if transition_stage:
            assert transition_segment is not None
            _validate_transition_execution_parameters(params, transition_segment)
            probe = None if skip_probe else runtime.probe(
                timeout_s=min(3.0, params["timeout_s"]), tf_timeout_s=5.0
            )
            if probe is not None:
                base["probe_evidence"] = _jsonable(probe)
            if not skip_probe and (not isinstance(probe, Mapping) or probe.get(
                "official_offline_conditions_met"
            ) is not True):
                raise ValueError("官方环境probe未通过，拒绝过渡单段执行")
        if any(count <= 0 for count in runtime.subscriber_counts().values()):
            raise ValueError("官方机械臂controller订阅不完整，拒绝执行")
        base["trial_parameters"] = {
            "planning": _jsonable(plan.get("trial_parameters", {})),
            "execution": _jsonable(params),
        }
        base["target_joint_state"] = list(target)
        lower = tuple(float(value) for value in runtime.config["action_mux"]["joint_lower"])
        upper = tuple(float(value) for value in runtime.config["action_mux"]["joint_upper"])
        if transition_stage:
            assert transition_start is not None and transition_segment is not None
            _validate_transition_segment(
                transition_segment, transition_start, target, lower, upper
            )
        if any(not lower[i] <= target[i] <= upper[i] for i in (0, *range(3, 17))):
            raise ValueError("目标关节超出controller manifest/action_mux安全限位")
        pre_evidence = _publisher_evidence(
            runtime, "before_create" if not publishers_already_started else "sequence_active",
            publishers_already_started,
        )
        base["publisher_conflict_evidence"].append(pre_evidence)
        if not pre_evidence["valid"]:
            base["failure_classification"] = "PUBLISHER_CONFLICT"
            raise ValueError("检测到真正外部Publisher或DDS图未在有界时间内收敛")
        start = runtime.wait_for_inputs(min(3.0, params["timeout_s"]))
        latest = start
        initial_cached = _evaluate_cached_feedback(
            start, runtime.latest_joint_age_ns(), None,
            int(params["feedback_max_age_s"] * 1_000_000_000), "JointState",
            getattr(runtime, "latest_joint_validation_error", None),
        )
        if not initial_cached["valid"]:
            base["failure_classification"] = initial_cached["classification"]
            raise ValueError(f"{initial_cached['failure_reason']}，拒绝执行")
        base["start_joint_state"] = _jsonable(start)
        expected_start = (
            _finite_joint_target(expected_start_override)
            if expected_start_override is not None
            else _expected_stage_start(plan, stages[0])
        )
        if expected_start is not None:
            start_diagnostics = _start_match_diagnostics(
                start.position, expected_start, params
            )
            base["expected_start_joint_state"] = list(expected_start)
            base["start_match_diagnostics"] = start_diagnostics
            base["start_error_by_joint"] = start_diagnostics[
                "signed_error_by_joint"
            ]
            if not start_diagnostics["matches"]:
                base["failure_classification"] = "START_MISMATCH"
                raise ValueError(
                    f"当前JointState不匹配{stages[0]}保存的段起点，禁止跳段"
                )
        planned_duration = _planned_stage_duration(plan, stages[0])
        timing = _stage_timing_diagnostics(
            start.position, target, params, planned_duration
        )
        base["timing_validation"] = timing
        if not timing["timeout_valid"]:
            base["failure_classification"] = "TIMEOUT_CONFIGURATION_INVALID"
            raise ValueError(
                f"--timeout-s不足；建议至少{timing['minimum_timeout_s']:.9f}s"
            )
        if not publishers_already_started:
            runtime.start_publishers()
        post_evidence = _publisher_evidence(runtime, "after_create", True)
        base["publisher_conflict_evidence"].append(post_evidence)
        if not post_evidence["valid"]:
            base["failure_classification"] = "PUBLISHER_CONFLICT"
            raise ValueError("创建标定Publisher后检测到真正外部Publisher或自身endpoint未收敛")
        begin = runtime.monotonic()
        stable = 0
        max_error = _joint_error(start.position, target)
        samples: list[dict[str, Any]] = []
        previous_feedback_timestamp_ns = start.timestamp_ns
        maximum_feedback_age_ns = int(params["feedback_max_age_s"] * 1_000_000_000)
        # 注：planned_duration 为 None 时无法判定插值结束，退化为"全程插值 +
        # 从首帧起即判定稳定"，行进途中机械臂不在容差内故 stable 自然保持 0。
        while True:
            elapsed = runtime.monotonic() - begin
            runtime.spin_control_period(1.0 / params["control_rate_hz"])
            base["control_tick_count"] += 1
            feedback = runtime.latest_joint_state()
            feedback_age_ns = runtime.latest_joint_age_ns()
            validation_error = getattr(runtime, "latest_joint_validation_error", None)
            cached = _evaluate_cached_feedback(
                feedback, feedback_age_ns, previous_feedback_timestamp_ns,
                maximum_feedback_age_ns, "JointState", validation_error,
            )
            base["feedback_age_s_by_tick"].append(
                None if feedback_age_ns is None else feedback_age_ns / 1_000_000_000
            )
            if feedback_age_ns is not None:
                observed = feedback_age_ns / 1_000_000_000
                prior = base["max_feedback_age_s"]
                base["max_feedback_age_s"] = observed if prior is None else max(prior, observed)
            if not cached["valid"]:
                base["failure_classification"] = cached["classification"]
                raise RuntimeError(cached["failure_reason"])
            assert feedback is not None
            reused_feedback = bool(cached["reused"])
            if reused_feedback:
                base["feedback_reused_tick_count"] += 1
            else:
                previous_feedback_timestamp_ns = feedback.timestamp_ns
            latest = feedback
            sample = _jsonable(feedback)
            sample["feedback_age_s"] = feedback_age_ns / 1_000_000_000
            sample["cache_reused"] = reused_feedback
            receipt = getattr(runtime, "latest_joint_receipt_ns", None)
            sample["tool_received_at_ns"] = receipt() if callable(receipt) else None
            samples.append(sample)
            max_error = max(max_error, _joint_error(feedback.position, target))
            elapsed = runtime.monotonic() - begin
            # 插值阶段结束后只发精确目标；插值中按最大速度线性插值（安全）。
            interpolation_done = (
                planned_duration is not None and elapsed >= planned_duration
            )
            command = (
                target if interpolation_done
                else _interpolated_target(start.position, target, elapsed, params)
            )
            runtime.publish_joint_target(command)
            base["published_control"] = True
            active_evidence = _publisher_evidence(runtime, "control_tick", True)
            if not active_evidence["valid"]:
                base["publisher_conflict_evidence"].append(active_evidence)
                base["failure_classification"] = "PUBLISHER_CONFLICT"
                raise RuntimeError("执行期间检测到真正外部机械臂控制Publisher")
            if any(count <= 0 for count in runtime.subscriber_counts().values()):
                raise RuntimeError("执行期间官方机械臂controller订阅消失")
            # Step A 修复（arm pick-place 标定 settle counter bug）：
            # 稳定帧仅按"新时间戳 JointState 进容差"累计；复用帧跳过（不计数也不清零）；
            # 去掉过严的 command==target 门限；插值未完成时不累计（避免行进途中
            # 瞬时进容差被误判为到位）。
            in_settle_phase = interpolation_done or planned_duration is None
            if in_settle_phase and not reused_feedback:
                if _target_reached(feedback.position, target, params):
                    stable += 1
                else:
                    stable = 0
            if stable >= params["settle_cycles"]:
                base["execution_success"] = True
                base["valid"] = True
                base["status"] = "SUCCESS"
                base["settle_time_s"] = runtime.monotonic() - begin
                break
            # 超时边界：先处理最新反馈（已计入 stable）再判超时，避免最后一帧
            # 到位却被超时检查吞掉。
            if elapsed >= params["timeout_s"]:
                base["timed_out"] = True
                base["failure_classification"] = "TIMEOUT"
                raise TimeoutError("单阶段执行超时")
    except KeyboardInterrupt:
        base["interrupted"] = True
        base["failure_classification"] = "INTERRUPTED"
        base["failure_reason"] = "用户Ctrl-C中断；已请求保持最新真实JointState"
        if (
            "latest" in locals() and getattr(runtime, "publishers", None)
            and base["failure_classification"] != "PUBLISHER_CONFLICT"
        ):
            base["hold_evidence"] = _hold_and_confirm(runtime, latest, params)
            base["published_control"] = True
            base["safe_stop_mode"] = "confirmed_real_joint_state_hold_then_stop_publishing"
    except Exception as exc:
        base["failure_reason"] = str(exc)
        if not base["failure_classification"]:
            base["failure_classification"] = (
                "NEVER_RECEIVED" if "等待/joint_states超时" in str(exc)
                else "OTHER_EXCEPTION"
            )
        if (
            "latest" in locals() and getattr(runtime, "publishers", None)
            and base["failure_classification"] != "PUBLISHER_CONFLICT"
        ):
            base["hold_evidence"] = _hold_and_confirm(runtime, latest, params)
            base["published_control"] = True
            base["safe_stop_mode"] = "confirmed_real_joint_state_hold_then_stop_publishing"
    base["actual_joint_state_series"] = samples if "samples" in locals() else []
    base["actual_joint_state"] = _jsonable(latest) if "latest" in locals() else None
    if "latest" in locals():
        base["final_joint_state"] = base["actual_joint_state"]
    if "runtime" in locals() and hasattr(runtime, "latest_joint_receipt_ns"):
        base["final_joint_state_received_at_ns"] = runtime.latest_joint_receipt_ns()
    base["feedback_received_count"] = int(
        getattr(runtime, "joint_received_count", 0)
    )
    base["maximum_error"] = max_error if "max_error" in locals() else None
    base["stable_frames"] = stable if "stable" in locals() else 0
    base["settled_cycles"] = base["stable_frames"]
    if "latest" in locals() and "target" in locals():
        base["final_joint_error_by_joint"] = {
            name: abs(float(latest.position[index]) - float(target[index]))
            for index, name in enumerate(JOINT_NAMES)
        }
        base["final_error_by_joint"] = dict(base["final_joint_error_by_joint"])
        base["final_max_joint_error"] = _joint_error(latest.position, target)
        base["reached_target"] = _target_reached(latest.position, target, params)
    if base["execution_success"] and transition_stage:
        base["next_stage"] = planned_next_stage
    if base["execution_success"] and hold_on_success and "latest" in locals():
        base["hold_evidence"] = _hold_and_confirm(runtime, latest, params)
        if not base["hold_evidence"]["confirmed"]:
            base["execution_success"] = False
            base["valid"] = False
            base["failure_classification"] = "STOP_UNCONFIRMED"
            base["failure_reason"] = "阶段到位但位置保持未取得3个新反馈帧确认"
    if base["hold_evidence"] is None:
        no_control = base["published_control"] is False
        base["hold_evidence"] = {
            "status": (
                "NOT_REQUIRED_NO_CONTROL_PUBLISHED" if no_control
                else "STOP_UNCONFIRMED"
            ),
            "confirmed": no_control,
            "reason": (
                "no publisher command was sent" if no_control
                else "hold suppressed because a true publisher conflict was detected"
            ),
        }
    elif not base["hold_evidence"].get("confirmed", False):
        base["execution_success"] = False
        base["valid"] = False
        base["stop_status"] = "STOP_UNCONFIRMED"
    else:
        base["stop_status"] = base["hold_evidence"].get("status", "HOLD_CONFIRMED")
    held_final = base["hold_evidence"].get("final_joint_state")
    if isinstance(held_final, Mapping) and "position" in held_final and "target" in locals():
        base["final_joint_state"] = held_final
        held_position = held_final["position"]
        base["final_joint_error_by_joint"] = {
            name: abs(float(held_position[index]) - float(target[index]))
            for index, name in enumerate(JOINT_NAMES)
        }
        base["final_error_by_joint"] = dict(base["final_joint_error_by_joint"])
        base["final_max_joint_error"] = _joint_error(held_position, target)
        base["reached_target"] = _target_reached(held_position, target, params)
    return base


def _live_execution_context_evidence(
    plan: Mapping[str, Any], runtime: Any, probe: Mapping[str, Any],
    expected_seed: int, expected_scene: str,
    live_tasks: Sequence[TaskSpec], instruction_received_ns: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    if plan.get("seed") != expected_seed:
        blockers.append("SEED_MISMATCH")
    if plan.get("scene") != expected_scene:
        blockers.append("SCENE_MISMATCH")
    planned_state = plan.get("start_joint_state")
    trial = plan.get("trial_parameters")
    current_reader = getattr(runtime, "latest_joint_state", None)
    current_state = current_reader() if callable(current_reader) else None
    plan_age_ns: int | None = None
    if (
        not isinstance(planned_state, Mapping)
        or type(planned_state.get("timestamp_ns")) is not int
        or not isinstance(trial, Mapping)
        or type(trial.get("planned_context_max_age_ns")) is not int
        or current_state is None
    ):
        blockers.append("PLAN_CONTEXT_TIME_EVIDENCE_MISSING")
    else:
        plan_age_ns = current_state.timestamp_ns - planned_state["timestamp_ns"]
        if plan_age_ns < 0:
            blockers.append("SERVER_CLOCK_RESET_OR_PLAN_FROM_FUTURE")
        elif plan_age_ns > trial["planned_context_max_age_ns"]:
            blockers.append("PLAN_CONTEXT_STALE")
    task = plan.get("task")
    if type(instruction_received_ns) is not int or instruction_received_ns < 0:
        blockers.append("LIVE_INSTRUCTION_TIMESTAMP_INVALID")
    elif not isinstance(task, Mapping):
        blockers.append("PLAN_TASK_CONTEXT_MISSING")
    else:
        match = [item for item in live_tasks if item.task_id == task.get("task_id")]
        if len(match) != 1 or any(
            _jsonable(getattr(match[0], name)) != task.get(name)
            for name in (
                "target_body", "target_color", "place_type",
                "place_world_xyz", "place_frame_id",
            )
        ):
            blockers.append("LIVE_TASK_CONTEXT_MISMATCH")
    precision: dict[str, Any] | None = None
    try:
        object_estimate = plan["object_estimate"]
        planar = _transform(probe["planarized_virtual_footprint_transform"])
        _converted, diagnostics = transform_object_to_virtual_footprint(
            object_estimate["position_xyz"], object_estimate["orientation_xyzw"], planar
        )
        position = diagnostics["object_pose_in_virtual_footprint"]["position_xyz_m"]
        planned_planar = _transform(plan["planarized_virtual_footprint_transform"])
        current_yaw = _quaternion_to_rpy(planar.rotation_xyzw)[2]
        planned_yaw = _quaternion_to_rpy(planned_planar.rotation_xyzw)[2]
        precision = {
            "actual_object_pose_in_virtual_footprint": diagnostics[
                "object_pose_in_virtual_footprint"
            ],
            "expected": [0.75, 0.0, float(object_estimate["position_xyz"][2])],
            "standoff_error_m": abs(float(position[0]) - 0.75),
            "lateral_error_m": abs(float(position[1])),
            "yaw_alignment_error_rad": abs(wrap_to_pi(current_yaw - planned_yaw)),
            "position_tolerance_m": 0.01, "yaw_tolerance_rad": 0.02,
        }
        precision["meets_calibration_precision"] = (
            math.hypot(precision["standoff_error_m"], precision["lateral_error_m"])
            <= 0.01 and precision["yaw_alignment_error_rad"] <= 0.02
        )
        if not precision["meets_calibration_precision"]:
            blockers.append("BASE_CALIBRATION_PRECISION_LOST")
    except Exception as exc:  # noqa: BLE001 - missing old context is a replan blocker
        blockers.append(f"BASE_CONTEXT_INVALID:{exc}")
    if probe.get("official_offline_conditions_met") is not True:
        blockers.append("OFFICIAL_PROBE_FAILED")
    return {
        "valid": not blockers, "blockers": blockers,
        "seed": expected_seed, "scene": expected_scene,
        "instruction_received_ns": instruction_received_ns,
        "live_task_count": len(live_tasks),
        "planned_context_age_ns": plan_age_ns,
        "base_precision": precision,
        "server_liveness_source": "fresh JointState/Odom/TF and live /material/instruction",
    }


def execute_pick_calibration_sequence(
    plan: Mapping[str, Any], official_offline_simulation: bool, confirm: str,
    parameters: Mapping[str, Any], note: str, runtime: Any,
    expected_seed: int, expected_scene: str,
    instruction_timeout_s: float,
    confirmation_provider: Any,
) -> dict[str, Any]:
    """Same-process, manually gated 4.4 sequence; never auto-advances a stage."""

    result: dict[str, Any] = {
        "schema": COMMAND_SCHEMA, "command": "execute-pick-calibration-sequence",
        "stage": "pick-calibration-sequence",
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "valid": False, "execution_success": False, "published_control": False,
        "automatic_execution_ready": False, "seed": plan.get("seed"),
        "scene": plan.get("scene"), "stage_order": list(PICK_CALIBRATION_SEQUENCE),
        "stage_results": [], "completed_stages": [], "failure_reason": "",
        "failure_classification": "", "interrupted": False,
        "publisher_conflict_evidence": [], "hold_evidence": None,
        "user_note": note, "dynamic_pick_verification": "NOT_YET_VERIFIED",
        "4_4_kinematic_sequence_complete": False,
        "control_tick_count": 0, "feedback_received_count": 0,
        "feedback_reused_tick_count": 0, "max_feedback_age_s": None,
        "instruction_qos": _jsonable(getattr(runtime, "instruction_qos_evidence", {
            "history": "KEEP_LAST", "depth": 10,
            "reliability": "RELIABLE", "durability": "VOLATILE",
            "compatibility_basis": "runtime did not expose QoS evidence",
        })),
        "instruction_reception_evidence": {
            "topic": runtime.config.get("topics", {}).get(
                "instruction", "/material/instruction"
            ),
            "timeout_s": instruction_timeout_s,
            "received": False,
            "received_ns": None,
            "joint_feedback_count_before_wait": None,
            "joint_feedback_count_after_wait": None,
        },
        "start_joint_state": {"available": False, "reason": "PRECHECK_NOT_REACHED"},
        "expected_start_joint_state": {
            "available": False, "reason": "PLAN_NOT_VALIDATED"
        },
        "start_error_by_joint": {},
        "target_joint_state": {"available": False, "reason": "NO_STAGE_CONFIRMED"},
        "final_joint_state": {"available": False, "reason": "NO_VALID_FEEDBACK"},
        "final_error_by_joint": {}, "final_max_joint_error": None,
        "settled_cycles": 0, "next_stage": "transition-1",
        "timed_out": False,
    }
    params = _execution_parameters(parameters)
    latest: RobotJointState | None = None
    try:
        if not official_offline_simulation or os.getenv("ROS_DOMAIN_ID") != "99":
            raise ValueError("同进程序列只允许ROS_DOMAIN_ID=99官方离线仿真")
        if confirm != OFFICIAL_SIM_CONFIRMATION:
            raise ValueError(f"--confirm必须严格为{OFFICIAL_SIM_CONFIRMATION}")
        if plan.get("schema") != COMMAND_SCHEMA or plan.get(
            "execution_contract_version"
        ) != EXECUTION_CONTRACT_VERSION:
            raise ValueError("OLD_PLAN_EXECUTION_CONTRACT:必须重新生成计划")
        if plan.get("single_stage_execution_ready") is not True:
            raise ValueError("计划未授权官方离线逐段执行")
        before = _publisher_evidence(runtime, "sequence_before_create", False)
        result["publisher_conflict_evidence"].append(before)
        if not before["valid"]:
            result["failure_classification"] = "PUBLISHER_CONFLICT"
            raise ValueError("序列开始前存在外部Publisher或DDS残留未收敛")
        probe = runtime.probe(timeout_s=3.0, tf_timeout_s=5.0)
        result["probe_evidence"] = _jsonable(probe)
        before_instruction_joint_count = int(
            getattr(runtime, "joint_received_count", 0)
        )
        result["instruction_reception_evidence"][
            "joint_feedback_count_before_wait"
        ] = before_instruction_joint_count
        try:
            raw_instruction, instruction_received_ns, live_tasks = (
                runtime.receive_instruction_tasks(instruction_timeout_s)
            )
        except Exception as exc:
            result["failure_classification"] = "LIVE_CONTEXT_UNAVAILABLE"
            result["instruction_reception_evidence"][
                "joint_feedback_count_after_wait"
            ] = int(getattr(runtime, "joint_received_count", 0))
            result["instruction_reception_evidence"]["failure_reason"] = str(exc)
            raise ValueError(f"LIVE_CONTEXT_UNAVAILABLE: {exc}") from exc
        result["instruction_reception_evidence"].update({
            "received": True,
            "received_ns": instruction_received_ns,
            "raw_message_byte_count": len(raw_instruction.encode("utf-8")),
            "parsed_task_ids": [item.task_id for item in live_tasks],
            "joint_feedback_count_after_wait": int(
                getattr(runtime, "joint_received_count", 0)
            ),
        })
        latest_reader = getattr(runtime, "latest_joint_state", None)
        if callable(latest_reader):
            latest = latest_reader()
        context = _live_execution_context_evidence(
            plan, runtime, probe, expected_seed, expected_scene,
            live_tasks, instruction_received_ns,
        )
        result["execution_context_evidence"] = context
        if not context["valid"]:
            result["failure_classification"] = "LIVE_CONTEXT_MISMATCH"
            raise ValueError(f"实时执行上下文不匹配：{context['blockers']}")
        latest = runtime.wait_for_inputs(3.0)
        expected, _target, _segment, _next = _transition_stage_data(
            plan, "transition-1"
        )
        runtime.start_publishers()
        after = _publisher_evidence(runtime, "sequence_after_create", True)
        result["publisher_conflict_evidence"].append(after)
        if not after["valid"]:
            result["failure_classification"] = "PUBLISHER_CONFLICT"
            raise ValueError("序列Publisher自身endpoint或外部冲突检查失败")
        start_match = _start_match_diagnostics(latest.position, expected, params)
        result["sequence_start_match"] = start_match
        result["start_joint_state"] = _jsonable(latest)
        result["expected_start_joint_state"] = list(expected)
        result["start_error_by_joint"] = start_match["signed_error_by_joint"]
        if not start_match["matches"]:
            result["failure_classification"] = "START_MISMATCH"
            raise ValueError("序列起点不匹配transition-1，禁止跳段")
        expected_start = tuple(latest.position)
        for stage_index, stage in enumerate(PICK_CALIBRATION_SEQUENCE):
            decision = confirmation_provider(stage, latest, runtime, params)
            if decision != SEQUENCE_STAGE_CONFIRMATION:
                result["failure_classification"] = "USER_SAFE_EXIT"
                result["failure_reason"] = f"用户未确认{stage}；序列安全停止"
                result["hold_evidence"] = _hold_and_confirm(runtime, latest, params)
                break
            stage_result = execute_one_stage(
                plan, stage, True, OFFICIAL_SIM_CONFIRMATION, params, note, runtime,
                expected_seed=expected_seed, expected_scene=expected_scene,
                publishers_already_started=True, skip_probe=True,
                expected_start_override=expected_start,
                calibration_sequence=True, hold_on_success=True,
            )
            if stage_result["execution_success"]:
                stage_result["next_stage"] = (
                    PICK_CALIBRATION_SEQUENCE[stage_index + 1]
                    if stage_index + 1 < len(PICK_CALIBRATION_SEQUENCE) else None
                )
            result["stage_results"].append(stage_result)
            result["control_tick_count"] += stage_result["control_tick_count"]
            result["feedback_reused_tick_count"] += stage_result[
                "feedback_reused_tick_count"
            ]
            result["timed_out"] = result["timed_out"] or stage_result["timed_out"]
            result["target_joint_state"] = stage_result["target_joint_state"]
            result["final_joint_state"] = stage_result["final_joint_state"]
            result["final_error_by_joint"] = stage_result["final_error_by_joint"]
            result["final_max_joint_error"] = stage_result["final_max_joint_error"]
            result["settled_cycles"] = stage_result["settled_cycles"]
            result["next_stage"] = stage_result["next_stage"]
            age = stage_result["max_feedback_age_s"]
            if age is not None:
                result["max_feedback_age_s"] = (
                    age if result["max_feedback_age_s"] is None
                    else max(result["max_feedback_age_s"], age)
                )
            result["published_control"] = (
                result["published_control"] or stage_result["published_control"]
            )
            if not stage_result["execution_success"]:
                result["failure_reason"] = stage_result["failure_reason"]
                result["failure_classification"] = stage_result[
                    "failure_classification"
                ]
                result["hold_evidence"] = stage_result["hold_evidence"]
                break
            result["completed_stages"].append(stage)
            latest = _joints(stage_result["final_joint_state"])
            expected_start = tuple(latest.position)
        else:
            result["valid"] = True
            result["execution_success"] = True
            result["4_4_kinematic_sequence_complete"] = True
            result["failure_reason"] = ""
            result["hold_evidence"] = result["stage_results"][-1]["hold_evidence"]
    except KeyboardInterrupt:
        result["interrupted"] = True
        result["failure_classification"] = "INTERRUPTED"
        result["failure_reason"] = "用户Ctrl-C中断"
        if latest is not None and getattr(runtime, "publishers", None):
            result["hold_evidence"] = _hold_and_confirm(runtime, latest, params)
            result["published_control"] = True
    except Exception as exc:
        result["failure_reason"] = str(exc)
        if not result["failure_classification"]:
            result["failure_classification"] = "OTHER_EXCEPTION"
        if (
            latest is not None and getattr(runtime, "publishers", None)
            and result["failure_classification"] != "PUBLISHER_CONFLICT"
        ):
            result["hold_evidence"] = _hold_and_confirm(runtime, latest, params)
            result["published_control"] = True
    if isinstance(result.get("hold_evidence"), Mapping) and not result[
        "hold_evidence"
    ].get("confirmed", False):
        result["valid"] = False
        result["execution_success"] = False
        result["failure_classification"] = "STOP_UNCONFIRMED"
    result["feedback_received_count"] = int(
        getattr(runtime, "joint_received_count", 0)
    )
    if latest is not None and result["final_joint_state"].get("available") is False:
        result["final_joint_state"] = _jsonable(latest)
    return result


def _interactive_sequence_confirmation(
    stage: str, latest: RobotJointState, runtime: Any, parameters: Mapping[str, Any]
) -> str:
    responses: queue.Queue[str] = queue.Queue(maxsize=1)

    def read_confirmation() -> None:
        responses.put(input(
            f"\n下一阶段={stage}。输入{SEQUENCE_STAGE_CONFIRMATION}继续；其他输入安全退出: "
        ).strip())

    threading.Thread(target=read_confirmation, daemon=True).start()
    previous_timestamp_ns = latest.timestamp_ns
    while True:
        try:
            return responses.get_nowait()
        except queue.Empty:
            runtime.publish_joint_target(latest.position)
            runtime.spin_control_period(1.0 / parameters["control_rate_hz"])
            state = runtime.latest_joint_state()
            cached = _evaluate_cached_feedback(
                state, runtime.latest_joint_age_ns(), previous_timestamp_ns,
                int(parameters["feedback_max_age_s"] * 1_000_000_000),
                "JointState", getattr(runtime, "latest_joint_validation_error", None),
            )
            if not cached["valid"]:
                raise RuntimeError(
                    f"人工确认等待期间反馈失败：{cached['failure_reason']}"
                )
            assert state is not None
            previous_timestamp_ns = state.timestamp_ns
            graph = _publisher_evidence(runtime, "manual_confirmation_wait", True)
            if not graph["valid"]:
                raise RuntimeError("人工确认等待期间出现外部Publisher冲突")


def write_execution_log(result: Mapping[str, Any], data_root: str | Path) -> Path:
    seed = str(result.get("seed", "unknown"))
    scene = str(result.get("scene", "unknown"))
    stage = str(result.get("stage", "unknown"))
    if any("/" in value or value in {"", ".", ".."} for value in (seed, scene, stage)):
        raise ValueError("seed/scene/stage不能形成不安全日志路径")
    directory = Path(data_root).expanduser().resolve() / seed / scene
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{stage}-{time.time_ns()}.json"
    payload = {**result, "log_path": str(destination)}
    destination.write_text(_render(payload, "json") + "\n", encoding="utf-8")
    return destination


def summarize(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("command") == "execute-one-stage":
        required = (
            "seed", "scene", "stage", "trial_parameters", "start_joint_state",
            "target_joint_state", "actual_joint_state_series", "maximum_error",
            "stable_frames", "timed_out", "interrupted", "execution_success",
            "failure_reason", "user_note",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"执行日志缺少字段：{missing}")
        return {
            "schema": COMMAND_SCHEMA, "seed": payload["seed"], "scenario": payload["scene"],
            "stage": payload["stage"], "task": payload.get("task"),
            "object": payload.get("object"), "trial_parameters": payload["trial_parameters"],
            "start_joint_state": payload["start_joint_state"],
            "planned_target": payload["target_joint_state"],
            "actual_joint_state": payload.get("actual_joint_state"),
            "feedback_sample_count": len(payload["actual_joint_state_series"]),
            "maximum_error": payload["maximum_error"],
            "settle_time_s": payload.get("settle_time_s"),
            "stable_frames": payload["stable_frames"], "timed_out": payload["timed_out"],
            "interrupted": payload["interrupted"],
            "execution_success": payload["execution_success"],
            "failure_reason": payload["failure_reason"], "user_notes": payload["user_note"],
            "stable_grip_verified": False, "gripper_calibration_note": GRIPPER_NOTE,
            "pick_visual_verification": "BLOCKED", "place_visual_verification": "BLOCKED",
            "full_end_to_end": "BLOCKED",
        }
    required = ("seed", "scenario", "task", "object", "stage", "planned_target",
                "controlled_mask", "feedback_samples", "tolerances", "settle_cycles")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"摘要输入缺少字段：{missing}")
    target = tuple(payload["planned_target"])
    mask = tuple(payload["controlled_mask"])
    if len(target) != 17 or len(mask) != 17 or any(type(item) is not bool for item in mask):
        raise ValueError("planned_target/controlled_mask必须是17项团队顺序")
    tolerances = payload["tolerances"]
    required_settle = payload["settle_cycles"]
    if type(required_settle) is not int or required_settle <= 0:
        raise ValueError("settle_cycles必须是正整数")
    max_error = 0.0
    stable = 0
    reached_at = None
    actual = None
    last_stamp = -1
    failure = ""
    for sample in payload["feedback_samples"]:
        state = _joints(sample)
        if state.timestamp_ns < last_stamp:
            # 严格倒退才是真乱序；等于 last_stamp 属 DDS 复用帧（重复时间戳）
            failure = "JointState时间戳乱序"
            break
        if state.timestamp_ns == last_stamp:
            # 复用帧：跳过（不计数也不清零），与实时执行循环 Step A 修复一致
            continue
        last_stamp = state.timestamp_ns
        errors = [abs(state.position[i] - target[i]) for i in range(17) if mask[i]]
        max_error = max(max_error, max(errors, default=0.0))
        settled = all(
            abs(state.position[i] - target[i]) <= (
                tolerances["slide_m"] if i == 0 else
                tolerances["gripper"] if i in (9, 16) else tolerances["arm_rad"]
            ) for i in range(17) if mask[i]
        )
        stable = stable + 1 if settled else 0
        actual = list(state.position)
        if stable >= required_settle and reached_at is None:
            reached_at = state.timestamp_ns
    command_at = payload.get("command_timestamp_ns")
    settle_ns = None if reached_at is None or command_at is None else reached_at - command_at
    timed_out = bool(payload.get("timeout", False))
    success = reached_at is not None and not timed_out and not failure
    return {
        "schema": COMMAND_SCHEMA, "seed": payload["seed"], "scenario": payload["scenario"],
        "task": payload["task"], "object": payload["object"], "stage": payload["stage"],
        "planned_target": list(target), "actual_joint_state": actual,
        "maximum_error": max_error, "settle_time_ns": settle_ns,
        "stable_frames": stable, "timed_out": timed_out, "execution_success": success,
        "ik": payload.get("ik", {}), "joint_limits": payload.get("joint_limits", {}),
        "failure_reason": failure or payload.get("failure_reason", ""),
        "user_notes": payload.get("user_notes", ""),
        "pick_visual_verification": "BLOCKED", "place_visual_verification": "BLOCKED",
        "full_end_to_end": "BLOCKED",
    }


def _safe_output(path: str | None, text: str) -> None:
    if path is None:
        print(text)
        return
    destination = Path(path).expanduser().resolve()
    try:
        relative = destination.relative_to(REPO_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        checked = subprocess.run(["git", "check-ignore", "-q", str(relative)], cwd=REPO_ROOT)
        if checked.returncode != 0:
            raise ValueError("仓库内输出路径必须由.gitignore覆盖；推荐仓库外目录或team_sorting_dataset/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")


def _render(result: Mapping[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(result))
    writer.writeheader()
    writer.writerow({key: json.dumps(_jsonable(value), ensure_ascii=False) for key, value in result.items()})
    return buffer.getvalue().rstrip("\n")


def _sequence_terminal_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded terminal view; the execution log retains the full evidence."""

    hold = result.get("hold_evidence")
    instruction = result.get("instruction_reception_evidence")
    return {
        "command": result.get("command"),
        "execution_success": result.get("execution_success"),
        "completed_stages": result.get("completed_stages", []),
        "next_stage": result.get("next_stage"),
        "failure_classification": result.get("failure_classification"),
        "failure_reason": result.get("failure_reason"),
        "instruction_received": (
            instruction.get("received") if isinstance(instruction, Mapping) else None
        ),
        "hold_status": hold.get("status") if isinstance(hold, Mapping) else None,
        "published_control": result.get("published_control"),
        "log_path": result.get("log_path"),
    }


def _trial_cli(values: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        name, separator, raw = item.partition("=")
        if not separator or not name:
            raise ValueError("--trial必须使用字段名=JSON值")
        if name in result:
            raise ValueError(f"--trial重复字段：{name}")
        try:
            result[name] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--trial {name}不是合法JSON值") from exc
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config/config.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="只读检查官方ROS图、TF和JointState；绝不创建publisher")
    probe.add_argument("--tf-timeout-s", type=float, default=5.0)
    timing = sub.add_parser("timing", help="只读采样JointState、Odom和odom->base_link TF时序")
    timing.add_argument("--duration-s", type=float, default=10.0)
    timing.add_argument("--safety-factor", type=float, required=True)
    timing.add_argument("--output")
    prepare = sub.add_parser(
        "prepare-pick-input",
        help="从完整实时instruction与阶段2冻结fixture生成只读pick输入",
    )
    prepare.add_argument("--task-id", type=int, required=True)
    prepare.add_argument("--scene", choices=tuple(SCENE_FIXTURE_KEYS), required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--fixture-confidence", type=float, required=True)
    prepare.add_argument("--instruction-timeout-s", type=float, default=5.0)
    prepare.add_argument("--output", default="/data/pick-input.json")
    base_plan = sub.add_parser(
        "plan-base-stand", help="只读复用NavigationController生成试验候选抓取站位"
    )
    base_plan.add_argument("--input", required=True)
    base_plan.add_argument("--standoff-m", type=float, required=True)
    base_plan.add_argument("--position-tolerance-m", type=float, required=True)
    base_plan.add_argument("--yaw-tolerance-rad", type=float, required=True)
    base_plan.add_argument("--input-timeout-s", type=float, default=3.0)
    base_plan.add_argument("--output", default="/data/plan-base-stand.json")
    sweep = sub.add_parser(
        "sweep-pick-stand",
        help="纯规划扫描候选standoff并检查完整双臂抓取IK；绝不创建publisher",
    )
    sweep.add_argument("--input", required=True)
    sweep.add_argument("--standoff-min-m", type=float, required=True)
    sweep.add_argument("--standoff-max-m", type=float, required=True)
    sweep.add_argument("--standoff-step-m", type=float, required=True)
    sweep.add_argument("--input-timeout-s", type=float, default=3.0)
    sweep.add_argument("--trial", action="append", default=[], metavar="NAME=JSON_VALUE")
    sweep.add_argument("--output", default="/data/sweep-pick-stand.json")
    capture_comparison = sub.add_parser(
        "capture-pick-comparison-state",
        help="只读捕获完整/joint_states与同期Odom，生成纯规划比较fixture",
    )
    capture_comparison.add_argument(
        "--scene", choices=tuple(SCENE_FIXTURE_KEYS), required=True
    )
    capture_comparison.add_argument("--seed", type=int, required=True)
    capture_comparison.add_argument(
        "--joint-state-timeout-s", type=float, default=5.0
    )
    capture_comparison.add_argument(
        "--output", default="/data/pick-comparison-state.json"
    )
    compare = sub.add_parser(
        "compare-pick-standoffs",
        help="纯规划比较固定0.55/0.75抓取候选；不创建ROS runtime或publisher",
    )
    compare.add_argument("--input", required=True)
    compare.add_argument("--state-fixture", required=True)
    compare.add_argument("--trial", action="append", default=[], metavar="NAME=JSON_VALUE")
    compare.add_argument("--output", default="/data/compare-pick-standoffs.json")
    base_execute = sub.add_parser(
        "execute-base-stand", help="仅限官方离线仿真的单目标底盘站位闭环"
    )
    base_execute.add_argument("--plan", required=True)
    base_execute.add_argument("--official-offline-simulation", action="store_true")
    base_execute.add_argument("--confirm", default="")
    base_execute.add_argument("--note", default="")
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--capture")
    for name in ("plan-pick", "plan-place"):
        item = sub.add_parser(name)
        item.add_argument("--input", required=True)
        item.add_argument("--output")
        item.add_argument("--live", action="store_true", help="从ROS读取实时JointState/Odom/TF")
        item.add_argument("--input-timeout-s", type=float, default=3.0)
        item.add_argument("--trial", action="append", default=[], metavar="NAME=JSON_VALUE")
    execute = sub.add_parser("execute-one-stage")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--stage", required=True, choices=(*STAGES, *TRANSITION_STAGES))
    execute.add_argument("--expected-seed", type=int)
    execute.add_argument("--expected-scene", choices=tuple(SCENE_FIXTURE_KEYS))
    execute.add_argument("--official-offline-simulation", action="store_true")
    execute.add_argument("--confirm", default="")
    execute.add_argument("--note", default="")
    execute.add_argument("--max-slide-velocity-m-s", type=float, required=True)
    execute.add_argument("--max-arm-velocity-rad-s", type=float, required=True)
    execute.add_argument("--max-gripper-velocity-per-s", type=float, required=True)
    execute.add_argument("--control-rate-hz", type=float, required=True)
    execute.add_argument("--timeout-s", type=float, required=True)
    execute.add_argument("--feedback-max-age-s", type=float, required=True)
    execute.add_argument("--slide-tolerance-m", type=float, required=True)
    execute.add_argument("--arm-tolerance-rad", type=float, required=True)
    execute.add_argument("--gripper-tolerance", type=float, required=True)
    execute.add_argument("--settle-cycles", type=int, required=True)
    sequence = sub.add_parser("execute-pick-calibration-sequence")
    sequence.add_argument("--plan", required=True)
    sequence.add_argument("--expected-seed", type=int, required=True)
    sequence.add_argument("--expected-scene", choices=tuple(SCENE_FIXTURE_KEYS), required=True)
    sequence.add_argument("--official-offline-simulation", action="store_true")
    sequence.add_argument("--confirm", default="")
    sequence.add_argument("--note", default="")
    sequence.add_argument("--max-slide-velocity-m-s", type=float, required=True)
    sequence.add_argument("--max-arm-velocity-rad-s", type=float, required=True)
    sequence.add_argument("--max-gripper-velocity-per-s", type=float, required=True)
    sequence.add_argument("--control-rate-hz", type=float, required=True)
    sequence.add_argument("--timeout-s", type=float, required=True)
    sequence.add_argument("--feedback-max-age-s", type=float, required=True)
    sequence.add_argument("--instruction-timeout-s", type=float, required=True)
    sequence.add_argument("--slide-tolerance-m", type=float, required=True)
    sequence.add_argument("--arm-tolerance-rad", type=float, required=True)
    sequence.add_argument("--gripper-tolerance", type=float, required=True)
    sequence.add_argument("--settle-cycles", type=int, required=True)
    sequence.add_argument(
        "--verbose", action="store_true",
        help="终端输出完整probe和JointState；默认只输出简短状态，完整内容始终写日志",
    )
    summary = sub.add_parser("summarize")
    summary.add_argument("--input", required=True)
    summary.add_argument("--format", choices=("json", "csv"), default="json")
    summary.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    if args.command == "probe":
        result = probe_environment(config, tf_timeout_s=args.tf_timeout_s)
        print(_render(result, "json"))
        return 0 if result["official_offline_conditions_met"] else 2
    if args.command == "timing":
        runtime = RosCalibrationRuntime(config)
        try:
            result = runtime.timing(args.duration_s, args.safety_factor)
        finally:
            runtime.close()
        _safe_output(args.output, _render(result, "json"))
        return 0
    if args.command == "prepare-pick-input":
        runtime = RosCalibrationRuntime(config)
        try:
            result = runtime.prepare_pick_input(
                args.task_id, args.scene, args.seed, args.fixture_confidence,
                args.instruction_timeout_s,
            )
        finally:
            runtime.close()
        if result["valid"]:
            _safe_output(args.output, _render(result["payload"], "json"))
            result = {**result, "output_path": str(Path(args.output).expanduser())}
        print(_render(result, "json"))
        return 0 if result["valid"] else 2
    if args.command == "plan-base-stand":
        payload = _load_json(args.input)
        runtime = None
        try:
            if os.getenv("ROS_DOMAIN_ID") != "99":
                raise ValueError("plan-base-stand只允许ROS_DOMAIN_ID=99官方离线域")
            runtime = RosCalibrationRuntime(config)
            base = runtime.wait_for_base_state(args.input_timeout_s)
            result = plan_base_stand(
                payload, config, base, args.standoff_m,
                args.position_tolerance_m, args.yaw_tolerance_rad,
            )
        except Exception as exc:
            result = {
                "schema": COMMAND_SCHEMA, "command": "plan-base-stand",
                "mode": "plan-only", "valid": False, "status": "BLOCKED",
                "published_control": False, "failure_reason": str(exc),
            }
        finally:
            if runtime is not None:
                runtime.close()
        _safe_output(args.output, _render(result, "json"))
        return 0 if result["valid"] else 2
    if args.command == "sweep-pick-stand":
        payload = _load_json(args.input)
        payload.setdefault("trial_parameters", {}).update(_trial_cli(args.trial))
        runtime = None
        try:
            if os.getenv("ROS_DOMAIN_ID") != "99":
                raise ValueError("sweep-pick-stand只允许ROS_DOMAIN_ID=99官方离线域")
            runtime = RosCalibrationRuntime(config)
            payload = runtime.capture_live_payload(payload, args.input_timeout_s)
            current_base = runtime.latest_base_state()
            if current_base is None:
                raise RuntimeError("实时Odom缓存为空")
            official = config["official"]
            adapter = OfficialKDLAdapter(official.get("root", ""), official["kdl_module"])
            adapter.self_check()
            result = sweep_pick_stand(
                payload, config, current_base, adapter,
                args.standoff_min_m, args.standoff_max_m, args.standoff_step_m,
            )
        except Exception as exc:
            result = {
                "schema": COMMAND_SCHEMA, "command": "sweep-pick-stand",
                "mode": "plan-only", "valid": False, "status": "BLOCKED",
                "published_control": False, "feasible_candidate_count": 0,
                "recommended_candidate": None, "candidates": [],
                "failure_reason": str(exc),
            }
        finally:
            if runtime is not None:
                runtime.close()
        _safe_output(args.output, _render(result, "json"))
        return 0 if result["valid"] else 2
    if args.command == "capture-pick-comparison-state":
        runtime = None
        try:
            if os.getenv("ROS_DOMAIN_ID") != "99":
                raise ValueError(
                    "capture-pick-comparison-state只允许ROS_DOMAIN_ID=99官方离线域"
                )
            runtime = RosCalibrationRuntime(config)
            result = capture_pick_comparison_state(
                config, runtime, args.scene, args.seed,
                args.joint_state_timeout_s,
            )
        except Exception as exc:
            result = {
                "schema": COMMAND_SCHEMA,
                "command": "capture-pick-comparison-state",
                "valid": False, "blockers": [str(exc)],
                "source": config.get("topics", {}).get("joint_states", ""),
                "scene": args.scene, "seed": args.seed,
                "evidence_source": "saved_official_joint_state",
                "publisher_objects_created": False,
                "published_control": False,
            }
        finally:
            if runtime is not None:
                runtime.close()
        _safe_output(args.output, _render(result, "json"))
        return 0 if result["valid"] else 2
    if args.command == "compare-pick-standoffs":
        payload = _load_json(args.input)
        payload.setdefault("trial_parameters", {}).update(_trial_cli(args.trial))
        try:
            actual, current_base, evidence_source, fixture_metadata = (
                _load_comparison_fixture(args.state_fixture, payload)
            )
            official = config["official"]
            adapter = OfficialKDLAdapter(
                official.get("root", ""), official["kdl_module"]
            )
            adapter.self_check()
            result = compare_pick_standoffs(
                payload, config, current_base, actual, adapter,
                evidence_source=evidence_source,
                state_fixture_metadata=fixture_metadata,
            )
        except Exception as exc:
            result = {
                "schema": COMMAND_SCHEMA,
                "command": "compare-pick-standoffs",
                "mode": "plan-only", "valid": False, "status": "BLOCKED",
                "state_fixture_mode": "recorded_official_joint_state",
                "published_control": False, "candidates": [],
                "recommended_candidate": None, "failure_reason": str(exc),
            }
        _safe_output(args.output, _render(result, "json"))
        return 0 if result["valid"] else 2
    if args.command == "execute-base-stand":
        plan = _load_json(args.plan)
        runtime = RosCalibrationRuntime(config)
        try:
            result = execute_base_stand(
                plan, config, args.official_offline_simulation,
                args.confirm, args.note, runtime,
            )
            log_path = write_base_execution_log(result, "/data/arm_calibration")
            result["log_path"] = str(log_path)
            print(_render(result, "json"))
            return 0 if result["execution_success"] else 2
        finally:
            runtime.close()
    if args.command == "inspect":
        result = inspect_environment(config, _load_json(args.capture) if args.capture else None)
        print(_render(result, "json"))
        return 0
    if args.command == "execute-one-stage":
        plan = _load_json(args.plan)
        parameters = {
            "max_slide_velocity_m_s": args.max_slide_velocity_m_s,
            "max_arm_velocity_rad_s": args.max_arm_velocity_rad_s,
            "max_gripper_velocity_per_s": args.max_gripper_velocity_per_s,
            "control_rate_hz": args.control_rate_hz, "timeout_s": args.timeout_s,
            "feedback_max_age_s": args.feedback_max_age_s,
            "slide_tolerance_m": args.slide_tolerance_m,
            "arm_tolerance_rad": args.arm_tolerance_rad,
            "gripper_tolerance": args.gripper_tolerance,
            "settle_cycles": args.settle_cycles,
        }
        runtime = RosCalibrationRuntime(config)
        runtime.config = config
        try:
            result = execute_one_stage(
                plan, args.stage, args.official_offline_simulation, args.confirm,
                parameters, args.note, runtime,
                expected_seed=args.expected_seed, expected_scene=args.expected_scene,
            )
            log_path = write_execution_log(result, "/data/arm_calibration")
            result["log_path"] = str(log_path)
            print(_render(result, "json"))
            return 0 if result["execution_success"] else 2
        finally:
            runtime.close()
    if args.command == "execute-pick-calibration-sequence":
        plan = _load_json(args.plan)
        parameters = {
            "max_slide_velocity_m_s": args.max_slide_velocity_m_s,
            "max_arm_velocity_rad_s": args.max_arm_velocity_rad_s,
            "max_gripper_velocity_per_s": args.max_gripper_velocity_per_s,
            "control_rate_hz": args.control_rate_hz, "timeout_s": args.timeout_s,
            "feedback_max_age_s": args.feedback_max_age_s,
            "slide_tolerance_m": args.slide_tolerance_m,
            "arm_tolerance_rad": args.arm_tolerance_rad,
            "gripper_tolerance": args.gripper_tolerance,
            "settle_cycles": args.settle_cycles,
        }
        runtime = RosCalibrationRuntime(config)
        try:
            result = execute_pick_calibration_sequence(
                plan, args.official_offline_simulation, args.confirm, parameters,
                args.note, runtime, args.expected_seed, args.expected_scene,
                args.instruction_timeout_s,
                _interactive_sequence_confirmation,
            )
            log_path = write_execution_log(result, "/data/arm_calibration")
            result["log_path"] = str(log_path)
            print(_render(
                result if args.verbose else _sequence_terminal_summary(result),
                "json",
            ))
            return 0 if result["execution_success"] else 2
        finally:
            runtime.close()
    payload = _load_json(args.input)
    if args.command in {"plan-pick", "plan-place"}:
        payload.setdefault("trial_parameters", {}).update(_trial_cli(args.trial))
    if args.command in {"plan-pick", "plan-place"} and args.live:
        runtime = RosCalibrationRuntime(config)
        try:
            payload = (
                runtime.capture_live_payload(payload, args.input_timeout_s)
                if args.command == "plan-pick"
                else runtime.capture_live_place_payload(payload, args.input_timeout_s)
            )
        finally:
            runtime.close()
    if args.command == "plan-pick":
        result = plan_pick(payload, config)
        output_format = "json"
    elif args.command == "plan-place":
        result = plan_place(payload, config)
        output_format = "json"
    else:
        result = summarize(payload)
        output_format = args.format
    _safe_output(args.output, _render(result, output_format))
    return 0 if result.get("valid", result.get("execution_success", True)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
