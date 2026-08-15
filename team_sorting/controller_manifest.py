"""Versioned, ROS-independent MMK2 controller metadata.

The manifest freezes the runtime-verified controller boundary without creating
another action order: every dimension name is derived from
``interfaces.ACTION_NAMES``.  It describes command metadata only.  It does not
prove that a command was published, accepted by the Server, or executed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Optional

from .interfaces import ACTION_NAMES


@dataclass(frozen=True)
class QoSMetadata:
    """Runtime-observed ROS 2 subscription QoS settings."""

    reliability: str
    history: str
    depth: int
    durability: str
    verification_status: str


@dataclass(frozen=True)
class OfficialTopicMetadata:
    """One official command topic and its fixed payload width."""

    group: str
    topic: str
    message_type: str
    element_count: int
    qos: QoSMetadata


@dataclass(frozen=True)
class ActionDimensionMetadata:
    """Metadata for one index in the canonical 19-dimensional action."""

    index: int
    name: str
    group: str
    unit: str
    semantic: str
    safe_min: float
    safe_max: float
    runtime_min: Optional[float]
    runtime_max: Optional[float]
    official_topic: str
    topic_element_index: int
    runtime_actuator_name: Optional[str]
    verification_status: str
    notes: str = ""


@dataclass(frozen=True)
class ControllerManifest:
    """Immutable controller contract metadata for one schema version."""

    schema_name: str
    schema_version: int
    action_dim: int
    nominal_control_frequency_hz: float
    actions: tuple[ActionDimensionMetadata, ...]
    official_topics: tuple[OfficialTopicMetadata, ...]
    runtime_source: str
    unverified_facts: tuple[str, ...]


_RUNTIME_QOS = QoSMetadata(
    reliability="RELIABLE",
    history="KEEP_LAST",
    depth=5,
    durability="VOLATILE",
    verification_status="runtime_verified",
)

_TOPICS: tuple[OfficialTopicMetadata, ...] = (
    OfficialTopicMetadata("base", "/cmd_vel", "geometry_msgs/msg/Twist", 2, _RUNTIME_QOS),
    OfficialTopicMetadata(
        "spine",
        "/spine_forward_position_controller/commands",
        "std_msgs/msg/Float64MultiArray",
        1,
        _RUNTIME_QOS,
    ),
    OfficialTopicMetadata(
        "head",
        "/head_forward_position_controller/commands",
        "std_msgs/msg/Float64MultiArray",
        2,
        _RUNTIME_QOS,
    ),
    OfficialTopicMetadata(
        "left_arm",
        "/left_arm_forward_position_controller/commands",
        "std_msgs/msg/Float64MultiArray",
        7,
        _RUNTIME_QOS,
    ),
    OfficialTopicMetadata(
        "right_arm",
        "/right_arm_forward_position_controller/commands",
        "std_msgs/msg/Float64MultiArray",
        7,
        _RUNTIME_QOS,
    ),
)

_TOPIC_BY_GROUP = {topic.group: topic.topic for topic in _TOPICS}

# Names deliberately do not appear in this table.  They are injected from the
# canonical ACTION_NAMES tuple below, so the two orders cannot silently drift.
_ACTION_SPECS: tuple[
    tuple[
        str,
        str,
        str,
        float,
        float,
        Optional[float],
        Optional[float],
        int,
        Optional[str],
        str,
        str,
    ],
    ...,
] = (
    (
        "base", "m/s", "velocity", -0.25, 0.25, None, None, 0, None,
        "partially_verified",
        "Twist.linear.x已实测；内部wheel motor范围不是base_v范围，速度上限仍为团队保守值。",
    ),
    (
        "base", "rad/s", "velocity", -0.50, 0.50, None, None, 1, None,
        "partially_verified",
        "Twist.angular.z已实测；内部wheel motor范围不是base_w范围，速度上限仍为团队保守值。",
    ),
    ("spine", "m", "absolute_position", -0.04, 0.87, -0.04, 0.87, 0, "lift", "runtime_verified", ""),
    ("head", "rad", "absolute_position", -0.50, 0.50, -0.50, 0.50, 0, "head_yaw", "runtime_verified", ""),
    ("head", "rad", "absolute_position", -1.18, 0.16, -1.18, 0.16, 1, "head_pitch", "runtime_verified", ""),
    ("left_arm", "rad", "absolute_position", -3.14, 2.089, -3.151, 2.089, 0, "lft_joint1", "runtime_verified", ""),
    ("left_arm", "rad", "absolute_position", -2.50, 0.181, -2.963, 0.181, 1, "lft_joint2", "runtime_verified", ""),
    ("left_arm", "rad", "absolute_position", -0.094, 3.14, -0.094, 3.161, 2, "lft_joint3", "runtime_verified", ""),
    ("left_arm", "rad", "absolute_position", -2.60, 2.60, -3.012, 3.012, 3, "lft_joint4", "runtime_verified", ""),
    ("left_arm", "rad", "absolute_position", -1.859, 1.859, -1.859, 1.859, 4, "lft_joint5", "runtime_verified", ""),
    ("left_arm", "rad", "absolute_position", -2.60, 2.60, -3.017, 3.017, 5, "lft_joint6", "runtime_verified", ""),
    (
        "left_arm", "dimensionless", "normalized_position", 0.0, 1.0, 0.0, 1.0,
        6, "lft_gripper", "runtime_verified",
        "官方离线仿真已验证：open=1.0，closed=0.1。",
    ),
    ("right_arm", "rad", "absolute_position", -3.14, 2.089, -3.151, 2.089, 0, "rgt_joint1", "runtime_verified", ""),
    ("right_arm", "rad", "absolute_position", -2.50, 0.181, -2.963, 0.181, 1, "rgt_joint2", "runtime_verified", ""),
    ("right_arm", "rad", "absolute_position", -0.094, 3.14, -0.094, 3.161, 2, "rgt_joint3", "runtime_verified", ""),
    ("right_arm", "rad", "absolute_position", -2.60, 2.60, -3.012, 3.012, 3, "rgt_joint4", "runtime_verified", ""),
    ("right_arm", "rad", "absolute_position", -1.859, 1.859, -1.859, 1.859, 4, "rgt_joint5", "runtime_verified", ""),
    ("right_arm", "rad", "absolute_position", -2.60, 2.60, -3.017, 3.017, 5, "rgt_joint6", "runtime_verified", ""),
    (
        "right_arm", "dimensionless", "normalized_position", 0.0, 1.0, 0.0, 1.0,
        6, "rgt_gripper", "runtime_verified",
        "官方离线仿真已验证：open=1.0，closed=0.1。",
    ),
)


def _build_actions() -> tuple[ActionDimensionMetadata, ...]:
    if len(_ACTION_SPECS) != len(ACTION_NAMES):
        raise ValueError("Controller Manifest规格数量必须与ACTION_NAMES一致")
    return tuple(
        ActionDimensionMetadata(
            index=index,
            name=ACTION_NAMES[index],
            group=spec[0],
            unit=spec[1],
            semantic=spec[2],
            safe_min=spec[3],
            safe_max=spec[4],
            runtime_min=spec[5],
            runtime_max=spec[6],
            official_topic=_TOPIC_BY_GROUP[spec[0]],
            topic_element_index=spec[7],
            runtime_actuator_name=spec[8],
            verification_status=spec[9],
            notes=spec[10],
        )
        for index, spec in enumerate(_ACTION_SPECS)
    )


MMK2_CONTROLLER_MANIFEST_V1 = ControllerManifest(
    schema_name="MMK2ControllerManifest",
    schema_version=1,
    action_dim=len(ACTION_NAMES),
    nominal_control_frequency_hz=40.0,
    actions=_build_actions(),
    official_topics=_TOPICS,
    runtime_source="/tmp/material_competition_ros2_runtime.xml",
    unverified_facts=(
        "server_command_watchdog",
        "controller_target_lock_and_restart_semantics",
        "publish_delivery_and_physical_execution",
    ),
)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}必须是有限实数，不能是bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label}必须是有限实数")
    return number


def validate_controller_manifest(
    manifest: ControllerManifest = MMK2_CONTROLLER_MANIFEST_V1,
) -> None:
    """Validate ordering, ranges, groups and runtime containment."""

    if manifest.schema_name != "MMK2ControllerManifest":
        raise ValueError("Controller Manifest schema_name不受支持")
    if type(manifest.schema_version) is not int or manifest.schema_version != 1:
        raise ValueError("Controller Manifest schema_version必须为1")
    if type(manifest.action_dim) is not int or manifest.action_dim != 19:
        raise ValueError("Controller Manifest action_dim必须为19")
    frequency = _finite_number(
        manifest.nominal_control_frequency_hz, "nominal_control_frequency_hz"
    )
    if frequency <= 0.0:
        raise ValueError("nominal_control_frequency_hz必须大于0")
    if len(manifest.actions) != manifest.action_dim:
        raise ValueError("Controller Manifest动作数量与action_dim不一致")
    if any(type(action.index) is not int for action in manifest.actions) or tuple(
        action.index for action in manifest.actions
    ) != tuple(range(19)):
        raise ValueError("Controller Manifest index必须连续且唯一")
    if tuple(action.name for action in manifest.actions) != ACTION_NAMES:
        raise ValueError("Controller Manifest动作顺序必须与ACTION_NAMES完全一致")

    expected_topics = {
        "base": ("/cmd_vel", "geometry_msgs/msg/Twist", 2),
        "spine": (
            "/spine_forward_position_controller/commands",
            "std_msgs/msg/Float64MultiArray",
            1,
        ),
        "head": (
            "/head_forward_position_controller/commands",
            "std_msgs/msg/Float64MultiArray",
            2,
        ),
        "left_arm": (
            "/left_arm_forward_position_controller/commands",
            "std_msgs/msg/Float64MultiArray",
            7,
        ),
        "right_arm": (
            "/right_arm_forward_position_controller/commands",
            "std_msgs/msg/Float64MultiArray",
            7,
        ),
    }
    expected_group_lengths = {
        group: metadata[2] for group, metadata in expected_topics.items()
    }
    topic_groups = tuple(topic.group for topic in manifest.official_topics)
    if len(topic_groups) != len(set(topic_groups)) or set(topic_groups) != set(
        expected_group_lengths
    ):
        raise ValueError("官方话题分组必须唯一且完整")
    topic_by_group = {topic.group: topic for topic in manifest.official_topics}
    for group, expected_length in expected_group_lengths.items():
        topic = topic_by_group[group]
        if (
            topic.topic,
            topic.message_type,
            topic.element_count,
        ) != expected_topics[group] or type(topic.element_count) is not int:
            raise ValueError(f"{group}官方话题、消息类型或分组长度不匹配")
        if (
            topic.qos.reliability,
            topic.qos.history,
            topic.qos.depth,
            topic.qos.durability,
            topic.qos.verification_status,
        ) != ("RELIABLE", "KEEP_LAST", 5, "VOLATILE", "runtime_verified") or type(
            topic.qos.depth
        ) is not int:
            raise ValueError(f"{group}话题QoS与运行时实测不一致")

    group_indices: dict[str, list[int]] = {group: [] for group in expected_group_lengths}
    for action in manifest.actions:
        safe_min = _finite_number(action.safe_min, f"{action.name}.safe_min")
        safe_max = _finite_number(action.safe_max, f"{action.name}.safe_max")
        if safe_min > safe_max:
            raise ValueError(f"{action.name}安全下界不能大于上界")
        if action.group not in topic_by_group:
            raise ValueError(f"{action.name}使用未知话题分组")
        topic = topic_by_group[action.group]
        if action.official_topic != topic.topic:
            raise ValueError(f"{action.name}官方话题与分组不一致")
        if type(action.topic_element_index) is not int:
            raise ValueError(f"{action.name}.topic_element_index必须是整数")
        group_indices[action.group].append(action.topic_element_index)
        if action.group == "base":
            if action.runtime_min is not None or action.runtime_max is not None:
                raise ValueError("base_v/base_w不能使用wheel motor ctrlrange")
            if action.runtime_actuator_name is not None:
                raise ValueError("base_v/base_w不是直接runtime actuator轴")
        else:
            runtime_min = _finite_number(action.runtime_min, f"{action.name}.runtime_min")
            runtime_max = _finite_number(action.runtime_max, f"{action.name}.runtime_max")
            if runtime_min > runtime_max:
                raise ValueError(f"{action.name}运行时下界不能大于上界")
            if safe_min < runtime_min or safe_max > runtime_max:
                raise ValueError(f"{action.name}安全范围超过运行时范围")

    for group, indices in group_indices.items():
        if tuple(indices) != tuple(range(expected_group_lengths[group])):
            raise ValueError(f"{group}话题元素索引必须连续且唯一")
    left = manifest.actions[5:11]
    right = manifest.actions[12:18]
    if tuple((item.safe_min, item.safe_max) for item in left) != tuple(
        (item.safe_min, item.safe_max) for item in right
    ):
        raise ValueError("左右臂六轴安全范围必须一致")


def validate_controller_config(
    config: Mapping[str, object],
    manifest: ControllerManifest = MMK2_CONTROLLER_MANIFEST_V1,
) -> None:
    """Require one loaded config mapping to match the frozen manifest."""

    validate_controller_manifest(manifest)
    if not isinstance(config, Mapping):
        raise ValueError("配置顶层必须是映射")
    try:
        timing = config["timing"]
        action_mux = config["action_mux"]
        topics = config["topics"]
    except KeyError as exc:
        raise ValueError(f"配置缺少Controller Manifest所需字段：{exc}") from exc
    if not all(isinstance(value, Mapping) for value in (timing, action_mux, topics)):
        raise ValueError("timing/action_mux/topics必须是映射")

    frequency = _finite_number(timing.get("control_rate_hz"), "timing.control_rate_hz")
    if frequency != manifest.nominal_control_frequency_hz:
        raise ValueError("control_rate_hz与Controller Manifest不一致")
    max_v = _finite_number(action_mux.get("max_abs_base_v"), "action_mux.max_abs_base_v")
    max_w = _finite_number(action_mux.get("max_abs_base_w"), "action_mux.max_abs_base_w")
    raw_lower = action_mux.get("joint_lower")
    raw_upper = action_mux.get("joint_upper")
    if not isinstance(raw_lower, (list, tuple)) or not isinstance(raw_upper, (list, tuple)):
        raise ValueError("action_mux关节上下界必须是序列")
    if len(raw_lower) != 17 or len(raw_upper) != 17:
        raise ValueError("action_mux关节上下界必须各有17项")
    lower = tuple(_finite_number(value, f"joint_lower[{index}]") for index, value in enumerate(raw_lower))
    upper = tuple(_finite_number(value, f"joint_upper[{index}]") for index, value in enumerate(raw_upper))
    expected_lower = tuple(action.safe_min for action in manifest.actions)
    expected_upper = tuple(action.safe_max for action in manifest.actions)
    actual_lower = (-max_v, -max_w, *lower)
    actual_upper = (max_v, max_w, *upper)
    if actual_lower != expected_lower or actual_upper != expected_upper:
        raise ValueError("config.yaml ActionMux范围与Controller Manifest安全范围不一致")

    official_commands = topics.get("official_commands")
    if not isinstance(official_commands, Mapping):
        raise ValueError("topics.official_commands必须是映射")
    expected_topics = {topic.group: topic.topic for topic in manifest.official_topics}
    config_topics = {
        "base": official_commands.get("cmd_vel"),
        "spine": official_commands.get("slide"),
        "head": official_commands.get("head"),
        "left_arm": official_commands.get("left_arm"),
        "right_arm": official_commands.get("right_arm"),
    }
    if config_topics != expected_topics:
        raise ValueError("config.yaml官方话题与Controller Manifest不一致")


validate_controller_manifest()
