"""Locate, validate and freeze the Recorder schema v1 machine contract.

This module is ROS-independent, has no dependency on the Recorder runtime, and
does not cache mutable contract state.  It supports source trees, ament shares,
ordinary Python installations and ``pip --prefix`` layouts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping

from .interface_contract import load_interface_contract


SCHEMA_NAME = "team_sorting.recorder"
SCHEMA_VERSION = 1
_RELATIVE_PATH = Path("config") / "contracts" / "recorder_schema_v1.json"
_MUTABILITY_VALUES = frozenset(
    {"immutable", "append_only", "end_only", "mutable_with_audit"}
)
_REQUIRED_DESCRIPTOR_KEYS = frozenset(
    {"type", "required", "nullable", "mutability", "source", "semantics"}
)
_REQUIRED_EVENT_TYPES = (
    "recorder_started",
    "run_bound",
    "run_changed",
    "task_transition",
    "attempt_transition",
    "instruction_updated",
    "competition_context_updated",
    "action_selected",
    "dispatch_attempted",
    "dispatch_succeeded",
    "dispatch_failed",
    "pairing_issue",
    "sample_dropped",
    "bag_started",
    "bag_stopped",
    "shutdown_requested",
    "recorder_finished",
    "unclean_shutdown_detected",
)
_ACTION_TERMS = (
    "proposed_action",
    "selected_action",
    "dispatched_action",
    "publisher_call_attempted",
    "publisher_call_succeeded",
    "publisher_failure_reason",
    "execution_feedback",
)
_REQUIRED_TOP_LEVEL = frozenset(
    {
        "schema_name",
        "schema_version",
        "schema_status",
        "implementation_phase",
        "interface_schema",
        "identity_contract",
        "layout_contract",
        "run_manifest_schema",
        "recorder_segment_schema",
        "event_schema",
        "raw_artifact_schema",
        "action_recording_schema",
        "shutdown_schema",
        "provenance_schema",
        "rosbag_schema",
        "legacy_compatibility",
        "schema_evolution",
        "planned_runtime_capabilities",
        "unresolved",
    }
)


def _ament_contract_candidate() -> Path | None:
    """Return the ament share candidate without making ament a dependency."""

    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("team_sorting")) / _RELATIVE_PATH
    except (ImportError, LookupError):
        return None


def _installed_share_candidates(module_path: Path) -> tuple[Path, ...]:
    """Derive finite adjacent share paths without probing global ``/share``."""

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


def recorder_contract_path() -> Path:
    """Return the first readable source-tree or installed Recorder contract."""

    candidates = _candidate_paths()
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Recorder schema v1 contract not found; searched: {searched}"
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Recorder contract contains invalid JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Recorder contract contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer and not bool")
    return value


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    return tuple(value)


def _reject_prohibited_identity_name(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "evaluation_run_id":
                raise ValueError(
                    f"{path} contains prohibited canonical field evaluation_run_id"
                )
            _reject_prohibited_identity_name(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_prohibited_identity_name(item, f"{path}[{index}]")
    elif value == "evaluation_run_id":
        raise ValueError(f"{path} contains prohibited canonical field evaluation_run_id")


def _validate_field_descriptors(section: object, label: str) -> None:
    fields = _mapping(_mapping(section, label).get("fields"), f"{label}.fields")
    if not fields:
        raise ValueError(f"{label}.fields must not be empty")
    for name, raw_descriptor in fields.items():
        _nonempty_text(name, f"{label} field name")
        descriptor = _mapping(raw_descriptor, f"{label}.fields.{name}")
        missing = _REQUIRED_DESCRIPTOR_KEYS - set(descriptor)
        if missing:
            raise ValueError(
                f"{label}.fields.{name} missing descriptor keys: {sorted(missing)}"
            )
        _nonempty_text(descriptor["type"], f"{label}.fields.{name}.type")
        if type(descriptor["required"]) is not bool:
            raise ValueError(f"{label}.fields.{name}.required must be bool")
        if type(descriptor["nullable"]) is not bool:
            raise ValueError(f"{label}.fields.{name}.nullable must be bool")
        if descriptor["mutability"] not in _MUTABILITY_VALUES:
            raise ValueError(f"{label}.fields.{name}.mutability is not controlled")
        _nonempty_text(descriptor["source"], f"{label}.fields.{name}.source")
        _nonempty_text(descriptor["semantics"], f"{label}.fields.{name}.semantics")
        if descriptor["nullable"]:
            _nonempty_text(
                descriptor.get("null_semantics"),
                f"{label}.fields.{name}.null_semantics",
            )
            representation = _mapping(
                descriptor.get("unavailable_representation"),
                f"{label}.fields.{name}.unavailable_representation",
            )
            if "value" not in representation or representation["value"] is not None:
                raise ValueError(
                    f"{label}.fields.{name} nullable representation must use null value"
                )


def _require_field_names(
    section: Mapping[str, Any], required: set[str], label: str
) -> Mapping[str, Any]:
    fields = _mapping(section.get("fields"), f"{label}.fields")
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{label}.fields missing required fields: {sorted(missing)}")
    return fields


def validate_recorder_contract(payload: Mapping[str, Any]) -> None:
    """Validate Recorder v1 identity, fields, boundaries and unresolved facts."""

    root = _mapping(payload, "Recorder contract root")
    missing_top = _REQUIRED_TOP_LEVEL - set(root)
    if missing_top:
        raise ValueError(f"Recorder contract missing top-level keys: {sorted(missing_top)}")
    if root.get("schema_name") != SCHEMA_NAME:
        raise ValueError(f"schema_name must be {SCHEMA_NAME}")
    if _strict_int(root.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version must be exactly 1")
    if root.get("schema_status") != "frozen":
        raise ValueError("schema_status must be frozen")
    if root.get("implementation_phase") != "contract_only":
        raise ValueError("implementation_phase must be contract_only")
    _reject_prohibited_identity_name(root)

    interface = _mapping(root["interface_schema"], "interface_schema")
    if interface.get("schema_name") != "team_sorting.interface":
        raise ValueError("interface_schema.schema_name must be team_sorting.interface")
    if _strict_int(interface.get("schema_version"), "interface schema version") != 1:
        raise ValueError("interface_schema.schema_version must be 1")

    identity = _mapping(root["identity_contract"], "identity_contract")
    if identity.get("run_identity_scope") != "team_local":
        raise ValueError("run_identity_scope must be team_local")
    canonical_fields = _array(identity.get("canonical_fields"), "canonical_fields")
    required_identity = {
        "run_id",
        "task_set_fingerprint",
        "task_id",
        "settled_attempt_count",
        "local_attempt_key",
        "recorder_segment_id",
    }
    if set(canonical_fields) != required_identity or len(canonical_fields) != len(
        required_identity
    ):
        raise ValueError("identity canonical_fields must be exact and unique")
    local_key = _mapping(identity.get("local_attempt_key"), "local_attempt_key")
    if tuple(local_key.get("components", ())) != (
        "run_id",
        "task_id",
        "settled_attempt_count",
    ):
        raise ValueError("local_attempt_key components are not canonical")
    if local_key.get("scope") != "team_local" or local_key.get(
        "official_attempt_id"
    ) is not False:
        raise ValueError("local_attempt_key must be team-local, not official")
    for flag in (
        "recorder_segment_equals_official_attempt",
        "recorder_segment_equals_training_episode",
        "task_transition_implies_physical_reset",
        "settled_attempt_transition_implies_server_or_robot_reset",
        "online_recorder_may_create_official_attempt_id",
    ):
        if identity.get(flag) is not False:
            raise ValueError(f"identity_contract.{flag} must be false")

    layout = _mapping(root["layout_contract"], "layout_contract")
    if _strict_int(layout.get("layout_version"), "layout_version") != 1:
        raise ValueError("layout_version must be 1")
    if tuple(layout.get("segment_kinds", ())) != ("bootstrap", "run_bound"):
        raise ValueError("segment kinds must be bootstrap and run_bound")
    bootstrap = _mapping(layout.get("bootstrap_segment"), "bootstrap_segment")
    if bootstrap.get("parent_run_id", object()) is not None:
        raise ValueError("bootstrap parent_run_id must be null")
    if bootstrap.get("retroactive_binding_forbidden") is not True:
        raise ValueError("bootstrap retroactive binding must be forbidden")
    run_bound = _mapping(layout.get("run_bound_segment"), "run_bound_segment")
    if "non-empty" not in str(run_bound.get("parent_run_id_rule")):
        raise ValueError("run_bound must require non-empty parent_run_id")
    if layout.get("b1_runtime_layout_implemented") is not False:
        raise ValueError("B1 must not claim runtime layout implementation")

    for section_name in (
        "run_manifest_schema",
        "recorder_segment_schema",
        "event_schema",
    ):
        _validate_field_descriptors(root[section_name], section_name)

    manifest_section = _mapping(root["run_manifest_schema"], "run_manifest_schema")
    manifest_fields = _require_field_names(
        manifest_section,
        {
            "schema_name", "schema_version", "recorder_schema_sha256",
            "interface_schema_name", "interface_schema_version",
            "interface_schema_sha256", "run_id", "run_identity_scope",
            "task_set_fingerprint", "project_commit", "project_branch",
            "dirty_worktree", "config_sha256", "official_server_image_id",
            "official_client_image_id", "ros_domain_id", "rmw_implementation",
            "observe_only", "official_publish_enabled", "start_ros_ns",
            "end_ros_ns", "start_wall_utc", "end_wall_utc",
            "recorder_segment_ids", "clean_shutdown", "shutdown_reason",
            "recovery_required", "provenance_warnings",
        },
        "run_manifest_schema",
    )
    if manifest_fields["run_id"]["mutability"] != "immutable":
        raise ValueError("run_id must be immutable")
    if manifest_fields["task_set_fingerprint"]["mutability"] != "immutable":
        raise ValueError("task_set_fingerprint must be immutable")
    if manifest_fields["recorder_segment_ids"]["mutability"] != "append_only":
        raise ValueError("recorder_segment_ids must be append_only")
    for name in (
        "end_ros_ns",
        "end_wall_utc",
        "clean_shutdown",
        "shutdown_reason",
        "recovery_required",
    ):
        if manifest_fields[name]["mutability"] != "end_only":
            raise ValueError(f"manifest {name} must be end_only")
    if manifest_fields["run_identity_scope"].get("allowed_values") != ["team_local"]:
        raise ValueError("manifest run_identity_scope must be team_local")

    segment = _mapping(root["recorder_segment_schema"], "recorder_segment_schema")
    segment_fields = _require_field_names(
        segment,
        {
            "recorder_segment_id", "parent_run_id", "segment_kind",
            "segment_sequence", "process_start_wall_utc", "process_end_wall_utc",
            "node_start_ros_ns", "node_end_ros_ns", "first_ros_timestamp_ns",
            "last_ros_timestamp_ns", "pid", "container_identity",
            "clean_shutdown", "shutdown_reason", "bag_path",
            "bag_storage_identifier", "bag_exit_code", "jsonl_artifacts",
            "message_counters", "dropped_counters", "pairing_counters",
            "warning_counters", "observed_task_ids",
            "observed_settled_attempt_counts", "context_valid_count",
            "context_invalid_count", "marker_state",
        },
        "recorder_segment_schema",
    )
    if tuple(segment.get("segment_kinds", ())) != ("bootstrap", "run_bound"):
        raise ValueError("segment schema kinds are not canonical")
    if segment_fields["parent_run_id"]["mutability"] != "immutable":
        raise ValueError("segment parent_run_id must be immutable")

    event = _mapping(root["event_schema"], "event_schema")
    _require_field_names(
        event,
        {
            "schema_name", "schema_version", "event_id", "event_type",
            "event_timestamp_ns", "receive_timestamp_ns", "receive_monotonic_ns",
            "monotonic_scope", "run_id", "recorder_segment_id", "task_id",
            "settled_attempt_count", "local_attempt_key", "payload", "source",
            "validity", "invalid_reasons", "source_event_ids",
        },
        "event_schema",
    )
    event_types = _array(event.get("event_types"), "event_types")
    if event_types != _REQUIRED_EVENT_TYPES or len(set(event_types)) != len(event_types):
        raise ValueError("event types must be complete, ordered and unique")
    introduced = _mapping(
        event.get("event_type_introduced_in"), "event_type_introduced_in"
    )
    if set(introduced) != set(event_types) or any(
        _strict_int(version, f"event introduced version for {name}") != 1
        for name, version in introduced.items()
    ):
        raise ValueError("every v1 event type must declare introduced_in=1")
    if event.get("events_jsonl_currently_implemented") is not False:
        raise ValueError("B1 must not claim events.jsonl is implemented")
    monotonic = event["fields"]["monotonic_scope"]
    if monotonic.get("allowed_values") != ["process_local"]:
        raise ValueError("monotonic_scope must be process_local")
    if "forbidden for cross-process ordering" not in event["fields"][
        "receive_monotonic_ns"
    ]["semantics"]:
        raise ValueError("monotonic cross-process prohibition is missing")

    raw = _mapping(root["raw_artifact_schema"], "raw_artifact_schema")
    expected_artifacts = (
        "metadata.json",
        "final_actions.jsonl",
        "action_dispatches.jsonl",
        "action_frames.jsonl",
        "action_pairing_issues.jsonl",
        "fsm_status.jsonl",
        "competition_contexts.jsonl",
        "rosbag/",
    )
    if tuple(raw.get("current_legacy_artifacts", ())) != expected_artifacts:
        raise ValueError("current legacy raw artifacts do not match the runtime")
    if raw.get("online_training_episode_creation") is not False:
        raise ValueError("online Recorder must not create Training Episodes")
    if raw.get("next_observation_binding") != "offline_only_commit_c":
        raise ValueError("next_observation must remain offline-only")
    if raw.get("training_eligibility") != "offline_qc_only_commit_c":
        raise ValueError("training eligibility must remain offline QC-only")

    action = _mapping(root["action_recording_schema"], "action_recording_schema")
    if tuple(action.get("terms", ())) != _ACTION_TERMS:
        raise ValueError("action recording terms do not match Interface v1")
    interface_contract = load_interface_contract()
    interface_terms = tuple(
        layer["name"]
        for layer in interface_contract["action_recording_contract"]["layers"]
    )
    if interface_terms != _ACTION_TERMS:
        raise ValueError("loaded Interface v1 action terms are incompatible")
    if action.get("final_action_is_actual_published_action") is not False:
        raise ValueError("FinalAction must not be declared actual published action")
    for confirmation in ("controller_accepted", "execution_confirmed"):
        fact = _mapping(action.get(confirmation), confirmation)
        if fact.get("value", object()) is not None or fact.get("status") != "unresolved":
            raise ValueError(f"{confirmation} must remain null and unresolved")
    if action.get("observe_only_formal_bc_label_eligible_by_default") is not False:
        raise ValueError("observe-only data must not be BC-label eligible by default")

    rosbag = _mapping(root["rosbag_schema"], "rosbag_schema")
    missing_topics = set(rosbag.get("currently_not_recorded_topics", ()))
    if missing_topics != {"/tf", "/tf_static", "/team/action_dispatch"}:
        raise ValueError("current unrecorded rosbag topics are not accurate")
    if tuple(rosbag.get("b3_planned_topics", ())) != ("/tf", "/tf_static"):
        raise ValueError("B3 TF plan is not canonical")
    if rosbag.get("b1_changes_runtime_topic_list") is not False:
        raise ValueError("B1 must not claim a runtime topic change")

    legacy = _mapping(root["legacy_compatibility"], "legacy_compatibility")
    if legacy.get("legacy_schema_id") != "legacy_flat_episode_v0":
        raise ValueError("legacy schema identity is not canonical")
    if legacy.get("unknown_major_reader_behavior") != "fail_closed":
        raise ValueError("unknown Recorder major versions must fail closed")
    evolution = _mapping(root["schema_evolution"], "schema_evolution")
    if evolution.get("unknown_major_reader_behavior") != "fail_closed":
        raise ValueError("schema evolution must fail closed on unknown major")

    shutdown = _mapping(root["shutdown_schema"], "shutdown_schema")
    if shutdown.get("b1_creates_markers") is not False:
        raise ValueError("B1 must not claim shutdown markers are implemented")
    provenance = _mapping(root["provenance_schema"], "provenance_schema")
    if provenance.get("docker_socket_dependency") != "forbidden":
        raise ValueError("Recorder provenance must not depend on Docker socket")
    capabilities = _array(
        root["planned_runtime_capabilities"], "planned_runtime_capabilities"
    )
    if not capabilities or any(
        not isinstance(item, Mapping) or item.get("implemented_in_b1") is not False
        for item in capabilities
    ):
        raise ValueError("planned capabilities must not claim B1 runtime implementation")

    unresolved = _array(root["unresolved"], "unresolved")
    if not unresolved:
        raise ValueError("unresolved must not be empty")
    keys: list[str] = []
    for index, raw_item in enumerate(unresolved):
        item = _mapping(raw_item, f"unresolved[{index}]")
        required = {
            "key",
            "status",
            "why_unresolved",
            "affected_components",
            "required_test_or_decision",
            "must_not_assume",
        }
        if not required.issubset(item):
            raise ValueError(f"unresolved[{index}] is incomplete")
        key = _nonempty_text(item["key"], f"unresolved[{index}].key")
        keys.append(key)
        if item["status"] != "unresolved":
            raise ValueError(f"unresolved[{index}].status must be unresolved")
        _nonempty_text(
            item["required_test_or_decision"],
            f"unresolved[{index}].required_test_or_decision",
        )
        _nonempty_text(
            item["must_not_assume"], f"unresolved[{index}].must_not_assume"
        )
    if len(keys) != len(set(keys)):
        raise ValueError("unresolved keys must be unique")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def load_recorder_contract(path: str | Path | None = None) -> Mapping[str, Any]:
    """Load, strictly validate and deeply freeze a fresh Recorder v1 graph."""

    contract_path = Path(path) if path is not None else recorder_contract_path()
    payload = json.loads(
        contract_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    validate_recorder_contract(payload)
    return _freeze(payload)


def recorder_contract_sha256(path: str | Path | None = None) -> str:
    """Return the lowercase SHA-256 of the exact Recorder contract bytes."""

    contract_path = Path(path) if path is not None else recorder_contract_path()
    return hashlib.sha256(contract_path.read_bytes()).hexdigest()
