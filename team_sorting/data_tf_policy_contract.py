"""Strict, ROS-independent loader for the B3 Data/TF Policy v1 contract."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
import re
import sys
from types import MappingProxyType
from typing import Any, Mapping

from .interface_contract import load_interface_contract
from .recorder_contract import load_recorder_contract


SCHEMA_NAME = "team_sorting.data_tf_policy"
SCHEMA_VERSION = 1
POLICY_ID = "data_tf_policy_v1"
INTERFACE_SHA256 = "a86548b2a43581af70b8d585d523a06bb97a8d96e1fe52950097b12d061fdaea"
RECORDER_SHA256 = "e7965c34a38c11d551d9943d8d614c05bc8e28e186432ad5ff4d0eed243225cf"
_RELATIVE_PATH = Path("config") / "contracts" / "data_tf_policy_v1.json"
_INTERFACE_RELATIVE_PATH = "config/contracts/interface_v1.json"
_RECORDER_RELATIVE_PATH = "config/contracts/recorder_schema_v1.json"
_TOPIC_PATTERN = re.compile(r"^/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+$")
_MESSAGE_TYPE_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*/msg/[A-Za-z][A-Za-z0-9_]*$"
)
_BASELINE_TOPICS = (
    "/material/instruction",
    "/head_camera/color/image_raw",
    "/head_camera/aligned_depth_to_color/image_raw",
    "/head_camera/color/camera_info",
    "/slamware_ros_sdk_server_node/odom",
    "/joint_states",
    "/team/object_estimates",
    "/team/fsm_status",
    "/team/competition_context",
    "/team/final_action",
    "/referee/taskinfo",
    "/referee/gameinfo",
    "/referee/score",
)
_B3_TARGET_TOPICS = ("/tf", "/tf_static")
_ACTION_LAYERS = (
    "proposed_action",
    "selected_action",
    "dispatched_action",
    "publisher_call_succeeded",
    "execution_feedback",
)
_PROFILE_IDS = (
    "debug_audit",
    "formal_collection_candidate",
    "fast_regression",
)
_SEVERITIES = ("fatal", "error", "warning", "info")
_ELIGIBILITY = ("eligible", "conditionally_eligible", "ineligible")
_FINDING_CODES = (
    "active_marker_remaining",
    "complete_marker_missing",
    "rosbag_metadata_missing",
    "rosbag_metadata_empty",
    "rosbag_exit_nonzero",
    "recovery_finding_present",
    "json_invalid",
    "jsonl_midstream_corruption",
    "sqlite_integrity_failed",
    "run_id_mixed",
    "competition_context_missing",
    "competition_context_invalid",
    "required_topic_missing",
    "timestamp_regression",
    "rgb_depth_alignment_failed",
    "camera_info_missing",
    "joint_state_dimension_invalid",
    "odom_frame_invalid",
    "tf_dynamic_missing",
    "tf_static_required_edge_missing",
    "tf_graph_disconnected",
    "selected_action_missing",
    "dispatched_action_missing",
    "selected_dispatched_mismatch",
    "observe_only_not_formal_bc",
    "execution_feedback_missing",
)


def _ament_contract_candidate() -> Path | None:
    """Return the ament-share candidate without requiring ROS at import time."""

    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("team_sorting")) / _RELATIVE_PATH
    except (ImportError, LookupError):
        return None


def _installed_share_candidates(module_path: Path) -> tuple[Path, ...]:
    """Derive finite source/prefix candidates without scanning the filesystem."""

    resolved = module_path.resolve()
    return tuple(
        ancestor / "share" / "team_sorting" / _RELATIVE_PATH
        for ancestor in resolved.parents
        if ancestor.parent != ancestor
    )


def _candidate_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    ament_candidate = _ament_contract_candidate()
    if ament_candidate is not None:
        candidates.append(ament_candidate)
    module_path = Path(__file__).resolve()
    candidates.append(module_path.parents[1] / _RELATIVE_PATH)
    candidates.extend(_installed_share_candidates(module_path))
    candidates.append(Path(sys.prefix) / "share" / "team_sorting" / _RELATIVE_PATH)
    return tuple(dict.fromkeys(candidates))


def data_tf_policy_contract_path() -> Path:
    """Return the first readable source-tree or installed policy path."""

    candidates = _candidate_paths()
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Data/TF Policy v1 not found; searched: {searched}")


def default_data_tf_policy_contract_path() -> Path:
    """Public explicit-name alias for the default resource resolver."""

    return data_tf_policy_contract_path()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Data/TF Policy contains invalid JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Data/TF Policy contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    return tuple(value)


def _exact_keys(
    value: object, expected: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
    item = _mapping(value, label)
    actual = set(item)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(f"{label} fields mismatch; missing={missing}, extra={extra}")
    return item


def _strict_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer and not bool")
    return value


def _strict_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be bool")
    return value


def _text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        requirement = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{label} must be {requirement}")
    return value


def _text_array(value: object, label: str, *, unique: bool = True) -> tuple[str, ...]:
    items = _array(value, label)
    result = tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(items))
    if unique and len(result) != len(set(result)):
        raise ValueError(f"{label} entries must be unique")
    return result


def _positive_rate(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite positive number and not bool")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return result


def _validate_topic(value: object, label: str) -> str:
    topic = _text(value, label)
    if _TOPIC_PATTERN.fullmatch(topic) is None:
        raise ValueError(f"{label} is not a canonical absolute ROS topic: {topic!r}")
    return topic


def _validate_message_type(value: object, label: str) -> str:
    message_type = _text(value, label)
    if _MESSAGE_TYPE_PATTERN.fullmatch(message_type) is None:
        raise ValueError(f"{label} is not a canonical ROS message type: {message_type!r}")
    return message_type


def _dependency_path(policy_path: Path, reference: str) -> Path:
    relative = PurePosixPath(reference)
    if relative.is_absolute() or ".." in relative.parts or len(policy_path.parents) < 3:
        raise ValueError(f"dependency path is unsafe: {reference!r}")
    return policy_path.parents[2].joinpath(*relative.parts)


def _validate_dependency(
    value: object,
    label: str,
    *,
    expected_path: str,
    expected_schema: str,
    expected_version: int,
    expected_sha256: str,
    policy_path: Path,
) -> Path:
    item = _exact_keys(
        value,
        {"path", "schema_name", "schema_version", "frozen_sha256"},
        label,
    )
    if item["path"] != expected_path:
        raise ValueError(f"{label}.path must be {expected_path}")
    if item["schema_name"] != expected_schema:
        raise ValueError(f"{label}.schema_name must be {expected_schema}")
    if _strict_int(item["schema_version"], f"{label}.schema_version") != expected_version:
        raise ValueError(f"{label}.schema_version must be {expected_version}")
    if item["frozen_sha256"] != expected_sha256:
        raise ValueError(f"{label}.frozen_sha256 does not match the frozen value")
    dependency = _dependency_path(policy_path, expected_path)
    if dependency.is_symlink() or not dependency.is_file():
        raise FileNotFoundError(f"{label} resource missing or unsafe: {dependency}")
    actual = hashlib.sha256(dependency.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"{label} actual SHA256 mismatch: expected={expected_sha256}, actual={actual}"
        )
    return dependency


def _validate_raw_derived(value: object) -> None:
    item = _exact_keys(
        value,
        {
            "raw_in_place_modification",
            "raw_artifacts_read_only",
            "derived_root_template",
            "derived_outputs",
            "required_provenance_fields",
            "host_absolute_path_is_identity",
            "semantic_reproducibility_required",
            "automatic_raw_deletion",
            "raw_delete_or_archive_policy",
        },
        "raw_derived_policy",
    )
    if item["raw_in_place_modification"] != "forbidden":
        raise ValueError("raw in-place modification must be forbidden")
    if _text_array(item["raw_artifacts_read_only"], "raw_artifacts_read_only") != (
        "segment", "manifest", "events", "jsonl", "rosbag"
    ):
        raise ValueError("raw read-only artifact roles are not canonical")
    if item["derived_root_template"] != "derived/indexer_v1/<deterministic_build_id>/":
        raise ValueError("derived root template is not canonical")
    if _text_array(item["derived_outputs"], "derived_outputs") != (
        "index_build.json",
        "dataset_index.jsonl",
        "segment_qc.json",
        "run_qc.json",
        "sample_index.jsonl",
        "training_manifest.jsonl",
    ):
        raise ValueError("derived output names are not canonical")
    if set(_text_array(item["required_provenance_fields"], "required_provenance_fields")) != {
        "source_relative_path", "source_sha256", "tool_name", "tool_version",
        "tool_commit", "config_sha256", "deterministic_build_id",
    }:
        raise ValueError("derived provenance fields are incomplete")
    if _strict_bool(item["host_absolute_path_is_identity"], "host_absolute_path_is_identity"):
        raise ValueError("host absolute paths must not be training identity")
    if not _strict_bool(item["semantic_reproducibility_required"], "semantic_reproducibility_required"):
        raise ValueError("semantic reproducibility must be required")
    if item["automatic_raw_deletion"] != "forbidden":
        raise ValueError("automatic raw deletion must be forbidden")
    if item["raw_delete_or_archive_policy"] != "separate_explicit_human_approved_operation":
        raise ValueError("raw deletion/archive must be a separate approved operation")


def _validate_topic_entries(value: object, label: str) -> tuple[str, ...]:
    names: list[str] = []
    for index, raw in enumerate(_array(value, label)):
        item = _exact_keys(
            raw,
            {"name", "message_type", "role", "baseline_status", "default_retention"},
            f"{label}[{index}]",
        )
        names.append(_validate_topic(item["name"], f"{label}[{index}].name"))
        _validate_message_type(item["message_type"], f"{label}[{index}].message_type")
        _text(item["role"], f"{label}[{index}].role")
        _text(item["baseline_status"], f"{label}[{index}].baseline_status")
        _text(item["default_retention"], f"{label}[{index}].default_retention")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} topic names must be unique")
    return tuple(names)


def _validate_topic_policy(value: object) -> set[str]:
    item = _exact_keys(
        value,
        {
            "current_raw_baseline",
            "b3_target_topics",
            "action_dispatch_authority",
            "action_layer_distinction",
        },
        "topic_policy",
    )
    baseline = _validate_topic_entries(item["current_raw_baseline"], "current_raw_baseline")
    targets = _validate_topic_entries(item["b3_target_topics"], "b3_target_topics")
    if baseline != _BASELINE_TOPICS or targets != _B3_TARGET_TOPICS:
        raise ValueError("raw baseline or B3 target topic ordering is not canonical")
    if set(baseline) & set(targets):
        raise ValueError("raw baseline and B3 target topics must not overlap")
    authority = _exact_keys(
        item["action_dispatch_authority"],
        {"topic", "artifact", "artifact_is_authoritative", "rosbag_duplication_required_in_b3a"},
        "action_dispatch_authority",
    )
    if authority["topic"] != "/team/action_dispatch" or authority["artifact"] != "action_dispatches.jsonl":
        raise ValueError("ActionDispatch authority is not canonical")
    if not _strict_bool(authority["artifact_is_authoritative"], "artifact_is_authoritative"):
        raise ValueError("action_dispatches.jsonl must remain authoritative")
    if _strict_bool(authority["rosbag_duplication_required_in_b3a"], "rosbag_duplication_required_in_b3a"):
        raise ValueError("B3A must not require duplicate ActionDispatch rosbag storage")
    if _text_array(item["action_layer_distinction"], "action_layer_distinction") != _ACTION_LAYERS:
        raise ValueError("action layer distinction is not canonical")
    return set(baseline + targets)


def _validate_qos(
    value: object,
    label: str,
    expected: tuple[str, str, str, int],
) -> None:
    item = _exact_keys(value, {"reliability", "durability", "history", "depth"}, label)
    reliability = item["reliability"]
    durability = item["durability"]
    history = item["history"]
    depth = _strict_int(item["depth"], f"{label}.depth")
    if reliability not in {"best_effort", "reliable"}:
        raise ValueError(f"{label}.reliability is invalid")
    if durability not in {"volatile", "transient_local"}:
        raise ValueError(f"{label}.durability is invalid")
    if history not in {"keep_last", "keep_all"}:
        raise ValueError(f"{label}.history is invalid")
    if depth <= 0:
        raise ValueError(f"{label}.depth must be positive")
    if (reliability, durability, history, depth) != expected:
        raise ValueError(f"{label} does not match the frozen profile")


def _validate_tf_policy(value: object) -> None:
    item = _exact_keys(
        value,
        {"dynamic", "static", "known_dynamic_relationship", "world_odom_relationship", "raw_frame_id_rewrite"},
        "tf_policy",
    )
    dynamic = _exact_keys(
        item["dynamic"],
        {"topic", "message_type", "role", "subscription_qos", "observed_publisher_reliability", "observed_relationship"},
        "tf_policy.dynamic",
    )
    if (dynamic["topic"], dynamic["message_type"], dynamic["role"]) != (
        "/tf", "tf2_msgs/msg/TFMessage", "dynamic_transform"
    ):
        raise ValueError("dynamic TF identity is invalid")
    _validate_qos(
        dynamic["subscription_qos"],
        "tf_policy.dynamic.subscription_qos",
        ("best_effort", "volatile", "keep_last", 100),
    )
    if dynamic["observed_publisher_reliability"] != "reliable" or dynamic["observed_relationship"] != "odom_to_base_link":
        raise ValueError("observed dynamic TF facts are invalid")
    static = _exact_keys(
        item["static"],
        {
            "topic", "message_type", "role", "late_join_required",
            "subscription_qos", "current_fixed_scene_publisher_present",
            "publisher_absence_is_global_fatal", "profile_required_edge_finding",
        },
        "tf_policy.static",
    )
    if (static["topic"], static["message_type"], static["role"]) != (
        "/tf_static", "tf2_msgs/msg/TFMessage", "static_transform"
    ):
        raise ValueError("static TF identity is invalid")
    if not _strict_bool(static["late_join_required"], "late_join_required"):
        raise ValueError("static TF late join must be required")
    _validate_qos(
        static["subscription_qos"],
        "tf_policy.static.subscription_qos",
        ("reliable", "transient_local", "keep_last", 1),
    )
    if _strict_bool(static["current_fixed_scene_publisher_present"], "current_fixed_scene_publisher_present"):
        raise ValueError("current fixed scene must not invent a /tf_static publisher")
    if _strict_bool(static["publisher_absence_is_global_fatal"], "publisher_absence_is_global_fatal"):
        raise ValueError("absence of /tf_static publisher must not be globally fatal")
    if static["profile_required_edge_finding"] != "tf_static_required_edge_missing":
        raise ValueError("static required-edge finding code is invalid")
    relation = _exact_keys(
        item["known_dynamic_relationship"], {"parent_frame", "child_frame", "status"},
        "known_dynamic_relationship",
    )
    if tuple(relation.values()) != ("odom", "base_link", "observed"):
        raise ValueError("known dynamic relationship must be odom to base_link")
    world_odom = _exact_keys(
        item["world_odom_relationship"], {"status", "equivalent", "assumption_forbidden"},
        "world_odom_relationship",
    )
    if world_odom["status"] != "unknown" or _strict_bool(world_odom["equivalent"], "world_odom.equivalent"):
        raise ValueError("world and odom must remain non-equivalent and unknown")
    if not _strict_bool(world_odom["assumption_forbidden"], "world_odom.assumption_forbidden"):
        raise ValueError("world equals odom assumption must be forbidden")
    if item["raw_frame_id_rewrite"] != "forbidden":
        raise ValueError("raw frame ID rewriting must be forbidden")


def _validate_frame_policy(value: object) -> None:
    item = _exact_keys(
        value,
        {"normalization", "camera_info_empty_frame", "odom_frame", "tf_odom_frame", "traceability_required"},
        "frame_policy",
    )
    normalization = _exact_keys(
        item["normalization"],
        {"raw_field", "normalized_field", "normalization_rule", "preserve_raw_and_normalized", "empty_frame_guess_without_evidence"},
        "frame_policy.normalization",
    )
    if (
        normalization["raw_field"], normalization["normalized_field"],
        normalization["normalization_rule"], normalization["empty_frame_guess_without_evidence"]
    ) != ("raw_frame_id", "normalized_frame_id", "strip_leading_slashes", "forbidden"):
        raise ValueError("frame normalization fields or rule are invalid")
    if not _strict_bool(normalization["preserve_raw_and_normalized"], "preserve_raw_and_normalized"):
        raise ValueError("raw and normalized frame IDs must both be preserved")
    camera = _exact_keys(
        item["camera_info_empty_frame"],
        {"raw_frame_id", "effective_frame_binding_allowed", "effective_frame_id", "binding_source", "raw_message_rewrite"},
        "camera_info_empty_frame",
    )
    if camera["raw_frame_id"] != "" or camera["effective_frame_id"] != "head_camera" or camera["binding_source"] != "synchronized_rgb_depth":
        raise ValueError("CameraInfo empty-frame binding is invalid")
    if not _strict_bool(camera["effective_frame_binding_allowed"], "effective_frame_binding_allowed"):
        raise ValueError("evidence-based CameraInfo binding must be allowed")
    if camera["raw_message_rewrite"] != "forbidden":
        raise ValueError("raw CameraInfo rewriting must be forbidden")
    odom = _exact_keys(item["odom_frame"], {"historical_raw_frame_id", "normalized_frame_id"}, "odom_frame")
    tf_odom = _exact_keys(item["tf_odom_frame"], {"raw_frame_id", "normalized_frame_id"}, "tf_odom_frame")
    if tuple(odom.values()) != ("/odom", "odom") or tuple(tf_odom.values()) != ("odom", "odom"):
        raise ValueError("Odom raw/normalized frame facts are invalid")
    if not _strict_bool(item["traceability_required"], "traceability_required"):
        raise ValueError("frame traceability must be required")


def _validate_profile(value: object, index: int, allowed_topics: set[str]) -> str:
    label = f"profiles[{index}]"
    item = _exact_keys(
        value,
        {
            "profile_id", "purpose", "validation_status", "benchmark_required",
            "required_topics", "optional_topics", "target_rates_hz", "image_storage",
            "depth_storage", "tf_policy", "compression_policy", "raw_retention_policy",
            "training_allowed", "expected_use", "prohibited_use",
        },
        label,
    )
    profile_id = _text(item["profile_id"], f"{label}.profile_id")
    _text(item["purpose"], f"{label}.purpose")
    if item["validation_status"] not in {"validated_raw_baseline", "provisional"}:
        raise ValueError(f"{label}.validation_status is invalid")
    _strict_bool(item["benchmark_required"], f"{label}.benchmark_required")
    required = tuple(_validate_topic(topic, f"{label}.required_topics") for topic in _text_array(item["required_topics"], f"{label}.required_topics"))
    optional = tuple(_validate_topic(topic, f"{label}.optional_topics") for topic in _text_array(item["optional_topics"], f"{label}.optional_topics"))
    if set(required) & set(optional) or not set(required + optional) <= allowed_topics:
        raise ValueError(f"{label} required/optional topics overlap or are unknown")
    if "/tf" not in required or optional != ("/tf_static",):
        raise ValueError(f"{label} must require /tf and treat current /tf_static as optional")
    rates = _mapping(item["target_rates_hz"], f"{label}.target_rates_hz")
    if not rates:
        raise ValueError(f"{label}.target_rates_hz must not be empty")
    for rate_name, rate in rates.items():
        _text(rate_name, f"{label}.rate_name")
        _positive_rate(rate, f"{label}.target_rates_hz.{rate_name}")
    for prefix in ("joint_state", "odom", "tf_dynamic"):
        minimum, maximum = rates.get(f"{prefix}_min"), rates.get(f"{prefix}_max")
        if minimum is not None and maximum is not None and float(minimum) > float(maximum):
            raise ValueError(f"{label}.{prefix} minimum exceeds maximum")
    image = _exact_keys(
        item["image_storage"],
        {"online_codec", "retain_raw", "derived_candidate_codec", "codec_parameters_must_be_versioned"},
        f"{label}.image_storage",
    )
    if image["online_codec"] != "none" or not _strict_bool(image["retain_raw"], f"{label}.image.retain_raw"):
        raise ValueError(f"{label} online RGB encoding must remain disabled and raw retained")
    if not _strict_bool(image["codec_parameters_must_be_versioned"], f"{label}.codec_parameters_must_be_versioned"):
        raise ValueError(f"{label} derived RGB codec parameters must be versioned")
    if image["derived_candidate_codec"] not in {"none", "jpeg_versioned"}:
        raise ValueError(f"{label} derived RGB codec is invalid")
    depth = _exact_keys(
        item["depth_storage"],
        {"online_codec", "retain_raw", "lossy_allowed", "derived_candidate_codec"},
        f"{label}.depth_storage",
    )
    if depth["online_codec"] != "none" or not _strict_bool(depth["retain_raw"], f"{label}.depth.retain_raw"):
        raise ValueError(f"{label} online Depth encoding must remain disabled and raw retained")
    if _strict_bool(depth["lossy_allowed"], f"{label}.depth.lossy_allowed"):
        raise ValueError("lossy Depth encoding is forbidden")
    if depth["derived_candidate_codec"] not in {"none", "png16_lossless"}:
        raise ValueError(f"{label} derived Depth codec is invalid")
    tf_profile = _exact_keys(
        item["tf_policy"], {"record_dynamic", "record_static_when_published", "required_static_edges"},
        f"{label}.tf_policy",
    )
    if not _strict_bool(tf_profile["record_dynamic"], f"{label}.record_dynamic") or not _strict_bool(tf_profile["record_static_when_published"], f"{label}.record_static_when_published"):
        raise ValueError(f"{label} must target dynamic and available static TF")
    _text_array(tf_profile["required_static_edges"], f"{label}.required_static_edges")
    compression = _exact_keys(
        item["compression_policy"], {"storage_id", "rosbag_compression", "zstd_default_enabled"},
        f"{label}.compression_policy",
    )
    if compression["storage_id"] != "sqlite3" or compression["rosbag_compression"] != "none" or _strict_bool(compression["zstd_default_enabled"], f"{label}.zstd_default_enabled"):
        raise ValueError(f"{label} online storage defaults must remain sqlite3/uncompressed")
    retention = _exact_keys(
        item["raw_retention_policy"], {"mode", "automatic_delete", "long_term_bulk_suitable"},
        f"{label}.raw_retention_policy",
    )
    _text(retention["mode"], f"{label}.raw_retention_policy.mode")
    if _strict_bool(retention["automatic_delete"], f"{label}.automatic_delete"):
        raise ValueError("profiles must never automatically delete raw data")
    if _strict_bool(retention["long_term_bulk_suitable"], f"{label}.long_term_bulk_suitable"):
        raise ValueError("none of the unbenchmarked raw profiles is long-term bulk suitable")
    _strict_bool(item["training_allowed"], f"{label}.training_allowed")
    _text_array(item["expected_use"], f"{label}.expected_use")
    _text_array(item["prohibited_use"], f"{label}.prohibited_use")
    return profile_id


def _validate_profiles(value: object, allowed_topics: set[str]) -> None:
    profiles = _array(value, "profiles")
    ids = tuple(_validate_profile(profile, index, allowed_topics) for index, profile in enumerate(profiles))
    if ids != _PROFILE_IDS or len(ids) != len(set(ids)):
        raise ValueError("profile IDs must be canonical, ordered, and unique")
    by_id = {profile["profile_id"]: profile for profile in profiles}
    debug = by_id["debug_audit"]
    formal = by_id["formal_collection_candidate"]
    fast = by_id["fast_regression"]
    if debug["validation_status"] != "validated_raw_baseline" or debug["benchmark_required"] is not False:
        raise ValueError("debug_audit validation status is invalid")
    if formal["validation_status"] != "provisional" or formal["benchmark_required"] is not True:
        raise ValueError("formal_collection_candidate must remain provisional and benchmarked")
    if formal["target_rates_hz"].get("rgb") != 12.0 or formal["target_rates_hz"].get("derived_training") != 10.0:
        raise ValueError("formal 12 Hz/10 Hz candidate targets are missing")
    if formal["training_allowed"] is not True:
        raise ValueError("formal profile must allow only downstream post-QC training")
    if fast["validation_status"] != "provisional" or fast["training_allowed"] is not False:
        raise ValueError("fast_regression must be provisional and training-forbidden")


def _validate_storage_policy(value: object) -> None:
    item = _exact_keys(value, {"current_online_defaults", "zstd", "derived_codecs", "downsampling"}, "storage_policy")
    defaults = _exact_keys(
        item["current_online_defaults"],
        {"storage_id", "rosbag_compression", "online_rgb_codec", "online_depth_codec"},
        "current_online_defaults",
    )
    if tuple(defaults.values()) != ("sqlite3", "none", "none", "none"):
        raise ValueError("current online storage defaults must remain uncompressed sqlite3")
    zstd = _exact_keys(
        item["zstd"],
        {"environment_plugin_present", "competition_client_benchmark_status", "default_enabled", "required_benchmarks"},
        "storage_policy.zstd",
    )
    if not _strict_bool(zstd["environment_plugin_present"], "zstd.environment_plugin_present"):
        raise ValueError("observed zstd plugin presence must be recorded")
    if zstd["competition_client_benchmark_status"] != "not_validated_under_load" or _strict_bool(zstd["default_enabled"], "zstd.default_enabled"):
        raise ValueError("zstd must remain disabled pending load validation")
    if set(_text_array(zstd["required_benchmarks"], "zstd.required_benchmarks")) != {
        "cpu_usage", "message_loss", "shutdown_latency", "temporary_disk_space", "fail_closed_behavior"
    }:
        raise ValueError("zstd benchmark dimensions are incomplete")
    codecs = _exact_keys(
        item["derived_codecs"],
        {"rgb_candidate", "rgb_parameters_must_be_versioned", "depth_candidate", "depth_lossy_forbidden"},
        "derived_codecs",
    )
    if codecs["rgb_candidate"] != "jpeg_versioned" or codecs["depth_candidate"] != "png16_lossless":
        raise ValueError("derived codec candidates are invalid")
    if not _strict_bool(codecs["rgb_parameters_must_be_versioned"], "rgb_parameters_must_be_versioned") or not _strict_bool(codecs["depth_lossy_forbidden"], "depth_lossy_forbidden"):
        raise ValueError("codec versioning and lossless Depth must be mandatory")
    downsampling = _exact_keys(
        item["downsampling"],
        {"implemented_in_b3a", "profile_rates_are_policy_targets_only", "preserve_original_ros_timestamp", "preserve_source_message_identity", "unrecorded_every_n_rule"},
        "downsampling",
    )
    for key in ("profile_rates_are_policy_targets_only", "preserve_original_ros_timestamp", "preserve_source_message_identity"):
        if not _strict_bool(downsampling[key], f"downsampling.{key}"):
            raise ValueError(f"downsampling.{key} must be true")
    if _strict_bool(downsampling["implemented_in_b3a"], "downsampling.implemented_in_b3a"):
        raise ValueError("B3A must not implement downsampling")
    if downsampling["unrecorded_every_n_rule"] != "forbidden":
        raise ValueError("unrecorded every-N downsampling is forbidden")


def _validate_action_label_policy(value: object) -> None:
    item = _exact_keys(
        value,
        {
            "layers", "selected_action_source", "selected_action_alone_is_formal_bc_label",
            "publisher_call_succeeded_semantics", "publisher_success_proves_dds_controller_or_execution",
            "observe_only", "formal_bc_requirements", "r0_smoke",
        },
        "action_label_policy",
    )
    if _text_array(item["layers"], "action_label_policy.layers") != _ACTION_LAYERS:
        raise ValueError("action label layers are not canonical")
    if item["selected_action_source"] != "FinalAction" or _strict_bool(item["selected_action_alone_is_formal_bc_label"], "selected_action_alone_is_formal_bc_label"):
        raise ValueError("FinalAction alone must not be a formal BC label")
    if item["publisher_call_succeeded_semantics"] != "local_publisher_call_return_only" or _strict_bool(item["publisher_success_proves_dds_controller_or_execution"], "publisher_success_proves_dds_controller_or_execution"):
        raise ValueError("publisher success semantics are overstated")
    observe = _exact_keys(item["observe_only"], {"allowed_uses", "formal_bc_allowed"}, "action_label_policy.observe_only")
    if set(_text_array(observe["allowed_uses"], "observe_only.allowed_uses")) != {"perception", "state", "lifecycle", "diagnostics"}:
        raise ValueError("observe-only allowed uses are invalid")
    if _strict_bool(observe["formal_bc_allowed"], "observe_only.formal_bc_allowed"):
        raise ValueError("observe-only data must not be formal BC eligible")
    requirements = _exact_keys(
        item["formal_bc_requirements"],
        {
            "observe_only_must_be_false", "exact_dispatched_action_required",
            "action_contract_match_required", "valid_context_required",
            "finished_must_be_false", "safety_gate_semantics_required",
            "subsequent_execution_feedback_required",
        },
        "formal_bc_requirements",
    )
    if not all(_strict_bool(value, f"formal_bc_requirements.{key}") for key, value in requirements.items()):
        raise ValueError("all formal BC requirements must be enabled")
    smoke = _exact_keys(
        item["r0_smoke"],
        {"controlled_single_trajectory_allowed", "strict_annotation_required", "implies_formal_bc_qualification"},
        "r0_smoke",
    )
    if not _strict_bool(smoke["controlled_single_trajectory_allowed"], "r0.controlled_single_trajectory_allowed") or not _strict_bool(smoke["strict_annotation_required"], "r0.strict_annotation_required"):
        raise ValueError("R0 smoke requires a controlled, strictly annotated trajectory")
    if _strict_bool(smoke["implies_formal_bc_qualification"], "r0.implies_formal_bc_qualification"):
        raise ValueError("R0 smoke must not imply formal BC qualification")


def _validate_qc_policy(value: object) -> None:
    item = _exact_keys(value, {"severity", "eligibility", "finding_codes", "global_rules"}, "qc_policy")
    if _text_array(item["severity"], "qc_policy.severity") != _SEVERITIES:
        raise ValueError("QC severity values are not canonical")
    if _text_array(item["eligibility"], "qc_policy.eligibility") != _ELIGIBILITY:
        raise ValueError("training eligibility values are not canonical")
    if _text_array(item["finding_codes"], "qc_policy.finding_codes") != _FINDING_CODES:
        raise ValueError("QC finding codes must be canonical, ordered, and unique")
    rules = _exact_keys(
        item["global_rules"],
        {"tf_static_publisher_absent_is_global_fatal", "missing_profile_required_static_edge_code", "observe_only_formal_bc_eligibility"},
        "qc_policy.global_rules",
    )
    if _strict_bool(rules["tf_static_publisher_absent_is_global_fatal"], "tf_static_publisher_absent_is_global_fatal"):
        raise ValueError("missing /tf_static publisher must not be globally fatal")
    if rules["missing_profile_required_static_edge_code"] != "tf_static_required_edge_missing" or rules["observe_only_formal_bc_eligibility"] != "ineligible":
        raise ValueError("QC global eligibility rules are invalid")


def _validate_implementation_boundaries(value: object) -> None:
    item = _exact_keys(
        value,
        {
            "b3a_changes_recorder_runtime", "b3a_adds_tf_to_runtime_topics",
            "b3a_implements_compression", "b3a_implements_downsampling",
            "b3a_implements_indexer_or_qc", "b3a_generates_training_samples", "next_phases",
        },
        "implementation_boundaries",
    )
    for key in set(item) - {"next_phases"}:
        if _strict_bool(item[key], f"implementation_boundaries.{key}"):
            raise ValueError(f"implementation_boundaries.{key} must remain false in B3A")
    phases = _exact_keys(item["next_phases"], {"b3b", "b3c", "b3d"}, "next_phases")
    for key, value in phases.items():
        _text(value, f"next_phases.{key}")


def validate_data_tf_policy_contract(
    payload: Mapping[str, Any],
    *,
    contract_path: str | Path | None = None,
) -> None:
    """Validate all B3A policy identities, boundaries, enums, rates and hashes."""

    root = _exact_keys(
        payload,
        {
            "schema_name", "schema_version", "policy_id", "implementation_status",
            "interface_contract", "recorder_contract", "raw_derived_policy",
            "topic_policy", "tf_policy", "frame_policy", "profiles", "storage_policy",
            "action_label_policy", "qc_policy", "implementation_boundaries",
        },
        "Data/TF Policy root",
    )
    if root["schema_name"] != SCHEMA_NAME:
        raise ValueError(f"schema_name must be {SCHEMA_NAME}")
    if _strict_int(root["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version must be exactly 1")
    if root["policy_id"] != POLICY_ID:
        raise ValueError(f"policy_id must be {POLICY_ID}")
    if root["implementation_status"] != "contract_only":
        raise ValueError("implementation_status must be contract_only")

    resolved_policy_path = (
        Path(contract_path) if contract_path is not None else data_tf_policy_contract_path()
    )
    interface_path = _validate_dependency(
        root["interface_contract"],
        "interface_contract",
        expected_path=_INTERFACE_RELATIVE_PATH,
        expected_schema="team_sorting.interface",
        expected_version=1,
        expected_sha256=INTERFACE_SHA256,
        policy_path=resolved_policy_path,
    )
    recorder_path = _validate_dependency(
        root["recorder_contract"],
        "recorder_contract",
        expected_path=_RECORDER_RELATIVE_PATH,
        expected_schema="team_sorting.recorder",
        expected_version=1,
        expected_sha256=RECORDER_SHA256,
        policy_path=resolved_policy_path,
    )
    if load_interface_contract(interface_path)["schema_name"] != "team_sorting.interface":
        raise ValueError("referenced Interface contract failed identity validation")
    if load_recorder_contract(recorder_path)["schema_name"] != "team_sorting.recorder":
        raise ValueError("referenced Recorder contract failed identity validation")

    _validate_raw_derived(root["raw_derived_policy"])
    allowed_topics = _validate_topic_policy(root["topic_policy"])
    _validate_tf_policy(root["tf_policy"])
    _validate_frame_policy(root["frame_policy"])
    _validate_profiles(root["profiles"], allowed_topics)
    _validate_storage_policy(root["storage_policy"])
    _validate_action_label_policy(root["action_label_policy"])
    _validate_qc_policy(root["qc_policy"])
    _validate_implementation_boundaries(root["implementation_boundaries"])


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def load_data_tf_policy_contract(
    path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Strictly load, validate and deeply freeze a fresh policy object graph."""

    contract_path = Path(path) if path is not None else data_tf_policy_contract_path()
    payload = json.loads(
        contract_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    validate_data_tf_policy_contract(payload, contract_path=contract_path)
    return _freeze(payload)


def data_tf_policy_contract_sha256(path: str | Path | None = None) -> str:
    """Return lowercase SHA-256 for the exact Data/TF Policy contract bytes."""

    contract_path = Path(path) if path is not None else data_tf_policy_contract_path()
    return hashlib.sha256(contract_path.read_bytes()).hexdigest()
