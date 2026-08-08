"""Strict ROS-independent loader for Dataset Index/QC contract v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_NAME = "team_sorting.dataset_index"
SCHEMA_VERSION = 1
CONTRACT_ID = "dataset_index_v1"
_RELATIVE_PATH = Path("config/contracts/dataset_index_v1.json")
_TOP_KEYS = {
    "schema_name", "schema_version", "contract_id", "implementation_status",
    "dependency_contracts", "layout", "raw_policy", "build_identity",
    "provenance_fields", "finding_contract", "qc", "action_contract", "eligibility", "safety",
    "publication", "recovery_integrity", "source_validation", "topic_requirements", "qc_config", "cli_errors", "deferred",
}
_DEPENDENCIES = {
    "interface": ("team_sorting.interface", 1, "a86548b2a43581af70b8d585d523a06bb97a8d96e1fe52950097b12d061fdaea"),
    "recorder": ("team_sorting.recorder", 1, "e7965c34a38c11d551d9943d8d614c05bc8e28e186432ad5ff4d0eed243225cf"),
    "data_tf_policy": ("team_sorting.data_tf_policy", 1, "982934579d816d67c63c1ff8938ea49c54982cf7d75472a58c00fe6a9cefae80"),
}
_EXACT_ARRAYS = {
    "build_inputs": ('schema_name','schema_version','indexer_name','indexer_version','implementation_identity','qc_config_sha256','source_artifact_relative_paths_and_sha256','dependency_contract_sha256'),
    "build_excluded": ('generated_at_utc','host_absolute_path','username','temporary_directory','derived_outputs'),
    "provenance": ('source_relative_path','source_sha256','indexer_name','indexer_version','implementation_identity','qc_config_sha256','deterministic_build_id'),
    "finding_fields": ('code','severity','evaluation_status','artifact','relative_path','message','evidence','blocking_use_cases'),
    "deferred_qc": ('tf_static_required_edge_missing','tf_graph_disconnected','rgb_depth_alignment_failed','joint_state_dimension_invalid','odom_frame_invalid','selected_dispatched_mismatch','execution_feedback_missing'),
    "summary": ('selected_action_record_count','selected_action_valid_record_count','selected_action_invalid_record_count','selected_action_record_present','selected_action_present','action_dispatch_record_count','action_dispatch_valid_record_count','action_dispatch_invalid_record_count','action_dispatch_record_present','publish_attempted_record_count','exact_payload_record_count','exact_dispatched_action_present','publisher_call_succeeded_record_count','dispatched_action_present','execution_feedback_evaluation'),
    "formal": ('observe_only_false','publish_attempted_true','attempted_groups_nonempty','dispatched_mask_any_true','dispatched_action_any_nonnull','publisher_call_succeeded_true'),
    "deferred": ('sample_index.jsonl','training_manifest.jsonl','image_decode_export','image_compression','topic_downsampling','sample_pairing','action_interpolation','execution_feedback_window_pairing','tf_frame_graph_reconstruction','train_split'),
}
_EXPECTED_FINDING_CODES = tuple("active_marker_remaining complete_marker_missing marker_invalid marker_identity_mismatch bag_path_invalid rosbag_metadata_missing rosbag_metadata_empty rosbag_metadata_invalid rosbag_exit_nonzero segment_manifest_invalid segment_identity_mismatch segment_sequence_invalid unexpected_symlink source_path_escape recovery_finding_present recovery_required_by_manifest json_invalid jsonl_midstream_corruption sqlite_integrity_failed required_topic_missing topic_type_mismatch metadata_sqlite_count_mismatch timestamp_regression message_timestamp_span_invalid empty_required_topic competition_context_missing competition_context_invalid run_id_mixed segment_context_identity_mismatch camera_info_missing tf_dynamic_missing tf_static_required_edge_missing observe_only_not_formal_bc selected_action_missing selected_action_invalid dispatched_action_missing action_dispatch_invalid selected_dispatched_mismatch execution_feedback_missing run_manifest_missing run_manifest_invalid run_identity_mismatch run_manifest_segment_missing run_manifest_unlisted_segment run_manifest_segment_order_mismatch segment_parent_run_id_mismatch task_set_fingerprint_mismatch run_end_incomplete duplicate_segment_id tf_graph_disconnected rgb_depth_alignment_failed joint_state_dimension_invalid odom_frame_invalid".split())


def _ament_candidate() -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("team_sorting")) / _RELATIVE_PATH
    except (ImportError, LookupError, OSError):
        return None


def _candidates() -> tuple[Path, ...]:
    module = Path(__file__).resolve()
    values = [_ament_candidate(), module.parents[1] / _RELATIVE_PATH]
    values.extend(
        parent / "share/team_sorting" / _RELATIVE_PATH
        for parent in module.parents if parent.parent != parent
    )
    values.append(Path(sys.prefix) / "share/team_sorting" / _RELATIVE_PATH)
    return tuple(dict.fromkeys(path for path in values if path is not None))


def dataset_index_contract_path() -> Path:
    for path in _candidates():
        if path.is_file():
            return path
    raise FileNotFoundError(f"Dataset Index contract v1 not found; searched={_candidates()}")


def default_dataset_index_contract_path() -> Path:
    return dataset_index_contract_path()


def _constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _exact(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    item = _mapping(value, label)
    if set(item) != keys:
        raise ValueError(f"{label} fields mismatch; missing={sorted(keys-set(item))}, extra={sorted(set(item)-keys)}")
    return item


def _unique_texts(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{label} must be a non-empty-text array")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must be unique")
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dependency_path(contract_path: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError("dependency path must be text")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("dependency path must be safe repository-relative")
    candidate = contract_path.parents[2] / Path(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"dependency resource is missing or unsafe: {candidate}")
    return candidate


def validate_dataset_index_contract(payload: object, *, contract_path: Path) -> Mapping[str, Any]:
    root = _exact(payload, _TOP_KEYS, "contract")
    if root["schema_name"] != SCHEMA_NAME or type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValueError("unknown Dataset Index schema")
    if root["contract_id"] != CONTRACT_ID or root["implementation_status"] != "implemented_b3c_core":
        raise ValueError("Dataset Index contract identity mismatch")
    dependencies = _exact(root["dependency_contracts"], set(_DEPENDENCIES), "dependency_contracts")
    for name, expected in _DEPENDENCIES.items():
        ref = _exact(dependencies[name], {"path", "schema_name", "schema_version", "frozen_sha256"}, name)
        schema, version, digest = expected
        if ref["schema_name"] != schema or type(ref["schema_version"]) is not int or ref["schema_version"] != version or ref["frozen_sha256"] != digest:
            raise ValueError(f"{name} frozen identity mismatch")
        if _sha(_dependency_path(contract_path, ref["path"])) != digest:
            raise ValueError(f"{name} actual SHA256 mismatch")
    layout = _exact(root["layout"], {"derived_root", "implemented_outputs", "prohibited_outputs"}, "layout")
    if layout["derived_root"] != "derived/indexer_v1/<deterministic_build_id>" or _unique_texts(layout["implemented_outputs"], "implemented_outputs") != ("index_build.json", "dataset_index.jsonl", "segment_qc.json", "run_qc.json"):
        raise ValueError("output layout mismatch")
    if _unique_texts(layout["prohibited_outputs"], "prohibited_outputs") != ("sample_index.jsonl", "training_manifest.jsonl"): raise ValueError("prohibited outputs mismatch")
    raw = _exact(root["raw_policy"], {"immutable", "exclude_derived_from_sources", "source_paths_repo_relative", "absolute_source_paths_forbidden", "source_symlinks_forbidden", "automatic_repair", "marker_mutation"}, "raw_policy")
    if raw != {"immutable": True, "exclude_derived_from_sources": True, "source_paths_repo_relative": True, "absolute_source_paths_forbidden": True, "source_symlinks_forbidden": True, "automatic_repair": False, "marker_mutation": False}:
        raise ValueError("raw immutability policy mismatch")
    identity = _exact(root["build_identity"], {"hash_algorithm", "canonical_encoding", "inputs", "excluded"}, "build_identity")
    if identity["hash_algorithm"] != "sha256":
        raise ValueError("build hash algorithm mismatch")
    if _unique_texts(identity["inputs"], "build inputs") != _EXACT_ARRAYS["build_inputs"] or _unique_texts(identity["excluded"], "build exclusions") != _EXACT_ARRAYS["build_excluded"]: raise ValueError("build identity arrays mismatch")
    finding = _exact(root["finding_contract"], {"fields", "severity", "evaluation_status", "not_evaluated_is_pass", "fail_blocking_use_cases_mandatory", "blocking_scopes", "not_evaluated_auto_blocks"}, "finding_contract")
    if _unique_texts(finding["fields"], "finding fields") != _EXACT_ARRAYS["finding_fields"]: raise ValueError("finding fields mismatch")
    if _unique_texts(finding["severity"], "severity") != ("fatal", "error", "warning", "info"):
        raise ValueError("severity enum mismatch")
    if (_unique_texts(finding["evaluation_status"], "evaluation_status") != ("pass", "fail", "not_applicable", "not_evaluated") or finding["not_evaluated_is_pass"] is not False or finding["fail_blocking_use_cases_mandatory"] is not True or _unique_texts(finding["blocking_scopes"], "blocking scopes") != ("segment", "run") or finding["not_evaluated_auto_blocks"] is not False):
        raise ValueError("evaluation status semantics mismatch")
    action = _exact(root["action_contract"], {"file_or_object_presence_is_valid_record", "final_action_parser", "action_dispatch_parser", "invalid_record_always_finding", "summary_fields", "formal_bc_candidate_requirements", "publisher_call_succeeded_is_execution_confirmation"}, "action_contract")
    if action["file_or_object_presence_is_valid_record"] is not False or action["final_action_parser"] != "strict_final_action_from_json" or action["action_dispatch_parser"] != "strict_action_dispatch_from_json" or action["invalid_record_always_finding"] is not True or action["publisher_call_succeeded_is_execution_confirmation"] is not False:
        raise ValueError("action contract semantics mismatch")
    if _unique_texts(action["summary_fields"], "action summary fields") != _EXACT_ARRAYS["summary"] or _unique_texts(action["formal_bc_candidate_requirements"], "formal BC candidate requirements") != _EXACT_ARRAYS["formal"]: raise ValueError("action arrays mismatch")
    eligibility = _exact(root["eligibility"], {"use_cases", "values", "formal_bc_eligible_in_v1", "observe_only_formal_bc", "missing_dispatch_formal_bc", "feedback_not_evaluated_ceiling", "tf_dynamic_missing_perception", "tf_dynamic_missing_blocks"}, "eligibility")
    if _unique_texts(eligibility.get("use_cases"), "use_cases") != ("diagnostic", "perception", "formal_bc"):
        raise ValueError("use case enum mismatch")
    if _unique_texts(eligibility.get("values"), "eligibility values") != ("eligible", "conditionally_eligible", "ineligible"):
        raise ValueError("eligibility enum mismatch")
    if eligibility["formal_bc_eligible_in_v1"] is not False or eligibility["observe_only_formal_bc"] != "ineligible" or eligibility["missing_dispatch_formal_bc"] != "ineligible" or eligibility["feedback_not_evaluated_ceiling"] != "conditionally_eligible" or eligibility["tf_dynamic_missing_perception"] != "conditionally_eligible" or _unique_texts(eligibility["tf_dynamic_missing_blocks"], "tf blockers") != ("formal_bc",):
        raise ValueError("eligibility semantics mismatch")
    qc = _exact(root["qc"], {"finding_codes", "always_not_evaluated"}, "qc")
    codes = _unique_texts(qc.get("finding_codes"), "finding_codes")
    deferred_checks = _unique_texts(qc.get("always_not_evaluated"), "always_not_evaluated")
    if codes != _EXPECTED_FINDING_CODES or deferred_checks != _EXACT_ARRAYS["deferred_qc"] or not set(deferred_checks) <= set(codes):
        raise ValueError("deferred QC checks must be finding codes")
    safety = _exact(root["safety"], {"dataset_root_symlink_allowed", "source_symlink_allowed", "path_traversal_allowed", "sqlite_uri", "sqlite_query_only", "sqlite_attach_allowed", "sqlite_extension_loading_allowed", "size_limits_bytes"}, "safety")
    expected_safety = {"dataset_root_symlink_allowed": False, "source_symlink_allowed": False, "path_traversal_allowed": False, "sqlite_uri": "mode=ro&immutable=1", "sqlite_query_only": True, "sqlite_attach_allowed": False, "sqlite_extension_loading_allowed": False}
    if any(safety[key] != value for key, value in expected_safety.items()): raise ValueError("safety policy mismatch")
    limits = _mapping(safety["size_limits_bytes"], "size_limits")
    if set(limits) != {"json", "jsonl", "yaml", "sqlite"} or any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError("size limits must be positive strict integers")
    if _unique_texts(root["provenance_fields"], "provenance_fields") != _EXACT_ARRAYS["provenance"]: raise ValueError("provenance fields mismatch")
    publication = _exact(root["publication"], {"temporary_directory_same_parent", "fsync_before_publish", "atomic_directory_publish", "identical_existing_build", "different_existing_build", "overwrite_existing", "post_publish_raw_change_new_build", "failure_may_remove_reused_build", "existing_build_exact_files", "existing_extra_entry_policy", "existing_manifest_revalidated_against_current_material", "pre_rename_failure_cleanup", "post_rename_failure_cleanup", "cleanup_failure_preserves_primary_error"}, "publication")
    if publication["temporary_directory_same_parent"] is not True or publication["fsync_before_publish"] is not True or publication["atomic_directory_publish"] is not True:
        raise ValueError("atomic publication flags mismatch")
    if publication["identical_existing_build"] != "reuse" or publication["different_existing_build"] != "fail_closed" or publication["overwrite_existing"] is not False or publication["post_publish_raw_change_new_build"] != "safe_remove_then_fail" or publication["failure_may_remove_reused_build"] is not False:
        raise ValueError("publication semantics mismatch")
    if (_unique_texts(publication["existing_build_exact_files"], "existing build files") != ("index_build.json", "dataset_index.jsonl", "segment_qc.json", "run_qc.json") or publication["existing_extra_entry_policy"] != "fail_closed_preserve_existing" or publication["existing_manifest_revalidated_against_current_material"] is not True or publication["pre_rename_failure_cleanup"] != "temporary_only" or publication["post_rename_failure_cleanup"] != "newly_published_final_only" or publication["cleanup_failure_preserves_primary_error"] is not True):
        raise ValueError("existing build validation semantics mismatch")
    recovery = _exact(root["recovery_integrity"], {"strict_json_required", "symlink_allowed", "invalid_artifact_policy", "silent_ignore_allowed", "association_fields", "unsafe_or_missing_segment_reference_policy"}, "recovery_integrity")
    if recovery["strict_json_required"] is not True or recovery["symlink_allowed"] is not False or recovery["invalid_artifact_policy"] != "fail_closed" or recovery["silent_ignore_allowed"] is not False or _unique_texts(recovery["association_fields"], "recovery association fields") != ("source_segment_path", "source_parent_run_id") or recovery["unsafe_or_missing_segment_reference_policy"] != "fail_closed":
        raise ValueError("recovery integrity semantics mismatch")
    source = _exact(root["source_validation"], {"segment_manifest_schema", "run_manifest_schema", "schema_descriptors_are_authoritative", "marker_identity_strict", "manifest_bag_path_authoritative", "metadata_semantic_validation_required", "run_segment_list_exact", "recovery_required_blocks"}, "source_validation")
    if source != {"segment_manifest_schema": "recorder_segment_schema", "run_manifest_schema": "run_manifest_schema", "schema_descriptors_are_authoritative": True, "marker_identity_strict": True, "manifest_bag_path_authoritative": True, "metadata_semantic_validation_required": True, "run_segment_list_exact": True, "recovery_required_blocks": ["perception", "formal_bc"]}: raise ValueError("source validation mismatch")
    topics = _exact(root["topic_requirements"], {"perception_required", "perception_conditional", "non_perception_topics", "action_dispatch_jsonl_authoritative"}, "topic_requirements")
    if _unique_texts(topics["perception_required"], "perception required") != tuple(sorted(("/head_camera/color/image_raw", "/head_camera/aligned_depth_to_color/image_raw", "/head_camera/color/camera_info", "/joint_states", "/slamware_ros_sdk_server_node/odom", "/team/competition_context"), key=("/head_camera/color/image_raw", "/head_camera/aligned_depth_to_color/image_raw", "/head_camera/color/camera_info", "/joint_states", "/slamware_ros_sdk_server_node/odom", "/team/competition_context").index)) or topics["action_dispatch_jsonl_authoritative"] is not True: raise ValueError("topic requirements mismatch")
    qc_config = _exact(root["qc_config"], {"fields", "required_topics_known_policy_topics_only", "required_topics_unique_absolute_names", "required_static_edges_v1", "unknown_fields"}, "qc_config")
    if _unique_texts(qc_config["fields"], "qc config fields") != ("required_topics", "required_static_edges") or qc_config["required_topics_known_policy_topics_only"] is not True or qc_config["required_topics_unique_absolute_names"] is not True or qc_config["required_static_edges_v1"] != "empty_only" or qc_config["unknown_fields"] != "reject": raise ValueError("QC config contract mismatch")
    cli = _exact(root["cli_errors"], {"normalized_types", "json_summary_failure_status", "failure_exit_code"}, "cli_errors")
    if _unique_texts(cli["normalized_types"], "CLI errors") != ("DatasetIndexError", "ValueError", "TypeError", "OSError") or cli["json_summary_failure_status"] != "indexer_failed" or cli["failure_exit_code"] != 2: raise ValueError("CLI error contract mismatch")
    if _unique_texts(root["deferred"], "deferred") != _EXACT_ARRAYS["deferred"]: raise ValueError("deferred mismatch")
    return root


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def load_dataset_index_contract(path: str | Path | None = None) -> Mapping[str, Any]:
    selected = dataset_index_contract_path() if path is None else Path(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Dataset Index contract: {selected}: {exc}") from exc
    return _freeze(validate_dataset_index_contract(payload, contract_path=selected))


def dataset_index_contract_sha256(path: str | Path | None = None) -> str:
    return _sha(dataset_index_contract_path() if path is None else Path(path))
