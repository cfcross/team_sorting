"""Recorder schema v1 machine-contract regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import team_sorting.recorder_contract as recorder_contract_module
from team_sorting.recorder_contract import (
    _candidate_paths,
    _installed_share_candidates,
    load_recorder_contract,
    recorder_contract_path,
    recorder_contract_sha256,
    validate_recorder_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "contracts" / "recorder_schema_v1.json"


def _raw() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _contract():
    return load_recorder_contract(CONTRACT_PATH)


def _install_contract(prefix: Path, module_relative: str) -> tuple[Path, Path]:
    module_path = prefix / module_relative / "team_sorting" / "recorder_contract.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    contract_path = (
        prefix
        / "share"
        / "team_sorting"
        / "config"
        / "contracts"
        / "recorder_schema_v1.json"
    )
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_PATH.read_bytes())
    return module_path, contract_path


def test_source_tree_strict_load_and_identity() -> None:
    assert recorder_contract_path() == CONTRACT_PATH
    contract = _contract()
    assert contract["schema_name"] == "team_sorting.recorder"
    assert contract["schema_version"] == 1
    assert type(contract["schema_version"]) is int
    assert contract["schema_status"] == "frozen"
    assert contract["implementation_phase"] == "contract_only"


@pytest.mark.parametrize("version", [True, 0, 2])
def test_schema_version_requires_exact_integer_one(version: object) -> None:
    payload = _raw()
    payload["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        validate_recorder_contract(payload)


def test_interface_reference_is_frozen_to_interface_v1() -> None:
    interface = _contract()["interface_schema"]
    assert interface["schema_name"] == "team_sorting.interface"
    assert interface["schema_version"] == 1


def test_team_local_run_identity_and_canonical_fields() -> None:
    identity = _contract()["identity_contract"]
    assert identity["run_identity_scope"] == "team_local"
    assert tuple(identity["canonical_fields"]) == (
        "run_id",
        "task_set_fingerprint",
        "task_id",
        "settled_attempt_count",
        "local_attempt_key",
        "recorder_segment_id",
    )
    assert "official Server" in identity["run_id_semantics"]
    assert identity["run_continuity_across_client_restart"] == "unresolved"


def test_prohibited_evaluation_run_id_is_rejected_as_key_or_field() -> None:
    for payload in (_raw(), _raw()):
        if "evaluation_run_id" not in payload:
            if payload is not None and "fields" in payload["run_manifest_schema"]:
                payload["run_manifest_schema"]["fields"]["evaluation_run_id"] = dict(
                    payload["run_manifest_schema"]["fields"]["run_id"]
                )
        with pytest.raises(ValueError, match="evaluation_run_id"):
            validate_recorder_contract(payload)


def test_contract_contains_no_prohibited_canonical_identity_name() -> None:
    assert "evaluation_run_id" not in CONTRACT_PATH.read_text(encoding="utf-8")


def test_local_attempt_key_is_team_local_settled_count_tuple() -> None:
    key = _contract()["identity_contract"]["local_attempt_key"]
    assert key["components"] == (
        "run_id",
        "task_id",
        "settled_attempt_count",
    )
    assert key["scope"] == "team_local"
    assert key["official_attempt_id"] is False
    assert "not the current 1-based" in key["settled_attempt_count_semantics"]


def test_segment_attempt_episode_and_reset_boundaries_are_false() -> None:
    identity = _contract()["identity_contract"]
    for field in (
        "recorder_segment_equals_official_attempt",
        "recorder_segment_equals_training_episode",
        "task_transition_implies_physical_reset",
        "settled_attempt_transition_implies_server_or_robot_reset",
        "online_recorder_may_create_official_attempt_id",
    ):
        assert identity[field] is False


def test_layout_is_contract_only_and_forbids_moves_or_overwrite() -> None:
    layout = _contract()["layout_contract"]
    assert layout["layout_version"] == 1
    assert layout["implementation_status"] == "planned_b2"
    assert layout["b1_runtime_layout_implemented"] is False
    assert layout["cross_filesystem_move_for_binding"] == "forbidden"
    assert layout["overwrite_existing_directory"] == "forbidden"


def test_bootstrap_parent_is_null_and_never_retroactively_bound() -> None:
    bootstrap = _contract()["layout_contract"]["bootstrap_segment"]
    assert bootstrap["parent_run_id"] is None
    assert bootstrap["retroactive_binding_forbidden"] is True


def test_run_bound_parent_rule_is_nonempty_and_immutable() -> None:
    run_bound = _contract()["layout_contract"]["run_bound_segment"]
    assert "non-empty" in run_bound["parent_run_id_rule"]
    assert run_bound["parent_run_id_mutability"] == "immutable"
    segment = _contract()["recorder_segment_schema"]
    assert "run_bound requires a non-empty parent_run_id" in segment["cross_field_rules"]
    assert segment["fields"]["parent_run_id"]["mutability"] == "immutable"


@pytest.mark.parametrize(
    "section_name",
    ["run_manifest_schema", "recorder_segment_schema", "event_schema"],
)
def test_every_field_descriptor_has_required_shape(section_name: str) -> None:
    contract = _contract()
    required = {"type", "required", "nullable", "mutability", "source", "semantics"}
    for descriptor in contract[section_name]["fields"].values():
        assert required <= set(descriptor)
        assert type(descriptor["required"]) is bool
        assert type(descriptor["nullable"]) is bool


def test_mutability_values_are_controlled_and_used() -> None:
    contract = _contract()
    controlled = set(contract["field_descriptor_contract"]["mutability_values"])
    assert controlled == {
        "immutable",
        "append_only",
        "end_only",
        "mutable_with_audit",
    }
    for section in ("run_manifest_schema", "recorder_segment_schema", "event_schema"):
        assert all(
            descriptor["mutability"] in controlled
            for descriptor in contract[section]["fields"].values()
        )


def test_nullable_fields_define_null_semantics_and_representation() -> None:
    contract = _contract()
    for section in ("run_manifest_schema", "recorder_segment_schema", "event_schema"):
        for descriptor in contract[section]["fields"].values():
            if descriptor["nullable"]:
                assert descriptor["null_semantics"]
                assert descriptor["unavailable_representation"]["value"] is None


def test_descriptor_validator_rejects_missing_key_and_bad_nullable_rule() -> None:
    payload = _raw()
    del payload["event_schema"]["fields"]["event_id"]["source"]
    with pytest.raises(ValueError, match="missing descriptor keys"):
        validate_recorder_contract(payload)

    payload = _raw()
    del payload["event_schema"]["fields"]["run_id"]["null_semantics"]
    with pytest.raises(ValueError, match="null_semantics"):
        validate_recorder_contract(payload)


def test_manifest_identity_append_and_end_mutability() -> None:
    fields = _contract()["run_manifest_schema"]["fields"]
    assert fields["run_id"]["mutability"] == "immutable"
    assert fields["task_set_fingerprint"]["mutability"] == "immutable"
    assert fields["recorder_segment_ids"]["mutability"] == "append_only"
    assert all(
        fields[name]["mutability"] == "end_only"
        for name in (
            "end_ros_ns",
            "end_wall_utc",
            "clean_shutdown",
            "shutdown_reason",
            "recovery_required",
        )
    )


def test_manifest_provenance_unknowns_are_structured_not_empty_strings() -> None:
    fields = _contract()["run_manifest_schema"]["fields"]
    for name in (
        "project_commit",
        "project_branch",
        "dirty_worktree",
        "official_server_image_id",
        "official_client_image_id",
    ):
        assert fields[name]["type"].startswith("unknown_value<")
    envelope = _contract()["field_descriptor_contract"]["unknown_value_envelope"]
    assert envelope["unknown_value"] is None
    assert envelope["empty_string_for_unknown_forbidden"] is True


def test_event_types_are_complete_ordered_and_unique() -> None:
    event = _contract()["event_schema"]
    event_types = event["event_types"]
    assert len(event_types) == len(set(event_types)) == 18
    assert event_types[0] == "recorder_started"
    assert event_types[-1] == "unclean_shutdown_detected"
    assert {"run_bound", "attempt_transition", "pairing_issue"} <= set(event_types)
    assert set(event["event_type_introduced_in"]) == set(event_types)
    assert set(event["event_type_introduced_in"].values()) == {1}


def test_duplicate_or_missing_event_type_is_rejected() -> None:
    payload = _raw()
    payload["event_schema"]["event_types"][-1] = "recorder_started"
    with pytest.raises(ValueError, match="event types"):
        validate_recorder_contract(payload)


def test_event_monotonic_scope_is_process_local_only() -> None:
    fields = _contract()["event_schema"]["fields"]
    assert fields["monotonic_scope"]["allowed_values"] == ("process_local",)
    assert "forbidden for cross-process ordering" in fields["receive_monotonic_ns"][
        "semantics"
    ]


def test_event_identity_and_derived_source_rules_are_explicit() -> None:
    event = _contract()["event_schema"]
    assert "unique within recorder_segment_id" in event["event_id_scope"]
    assert "source_event_ids" in event["derived_event_source_rule"]
    assert event["events_jsonl_currently_implemented"] is False


def test_current_raw_artifacts_exactly_match_recorder_runtime() -> None:
    assert _contract()["raw_artifact_schema"]["current_legacy_artifacts"] == (
        "metadata.json",
        "final_actions.jsonl",
        "action_dispatches.jsonl",
        "action_frames.jsonl",
        "action_pairing_issues.jsonl",
        "fsm_status.jsonl",
        "competition_contexts.jsonl",
        "rosbag/",
    )


def test_online_samples_next_observation_and_training_eligibility_not_claimed() -> None:
    raw = _contract()["raw_artifact_schema"]
    assert raw["samples_jsonl"]["implementation_status"] == "not_implemented_online"
    assert raw["raw_records_jsonl"]["implementation_status"] == (
        "planned_offline_or_b2_optional"
    )
    assert raw["next_observation_binding"] == "offline_only_commit_c"
    assert raw["training_eligibility"] == "offline_qc_only_commit_c"
    assert raw["online_training_episode_creation"] is False


def test_action_terms_exactly_match_interface_v1() -> None:
    assert _contract()["action_recording_schema"]["terms"] == (
        "proposed_action",
        "selected_action",
        "dispatched_action",
        "publisher_call_attempted",
        "publisher_call_succeeded",
        "publisher_failure_reason",
        "execution_feedback",
    )


def test_current_action_availability_and_exact_dispatch_are_honest() -> None:
    action = _contract()["action_recording_schema"]
    current = action["current_implementation"]
    assert current["proposed_action"] == "not_available_complete_numeric_values"
    assert current["selected_action"] == "FinalAction"
    assert "ActionDispatchRecord exact" in current["dispatched_action"]
    assert action["final_action_is_actual_published_action"] is False
    assert action["head_only_exact_payload_source"].startswith("ActionDispatchRecord")


def test_publisher_success_disclaimer_and_confirmations_remain_unresolved() -> None:
    action = _contract()["action_recording_schema"]
    disclaimer = action["publisher_success_disclaimer"]
    assert all(word in disclaimer for word in ("DDS", "Server", "controller", "robot"))
    for name in ("controller_accepted", "execution_confirmed"):
        assert action[name]["value"] is None
        assert action[name]["status"] == "unresolved"
    assert action["observe_only_formal_bc_label_eligible_by_default"] is False
    assert action["training_eligibility_owner"] == "Commit C offline QC"


def test_tf_is_currently_absent_and_planned_for_b3() -> None:
    bag = _contract()["rosbag_schema"]
    assert set(bag["currently_not_recorded_topics"]) == {
        "/tf",
        "/tf_static",
        "/team/action_dispatch",
    }
    assert bag["b3_planned_topics"] == ("/tf", "/tf_static")
    assert bag["tf_static_qos_and_late_join_status"] == (
        "requires_real_humble_validation"
    )
    assert bag["must_not_assume_world_equals_odom"] is True
    assert bag["b1_changes_runtime_topic_list"] is False


def test_shutdown_states_markers_and_recovery_are_only_planned() -> None:
    shutdown = _contract()["shutdown_schema"]
    assert shutdown["implementation_status"] == "planned_b2"
    assert set(shutdown["markers"]) == {"ACTIVE", "COMPLETE"}
    assert shutdown["b1_creates_markers"] is False
    assert shutdown["kill_9_or_power_loss_finally_guaranteed"] is False
    recovery = shutdown["jsonl_recovery"]
    assert recovery["raw_in_place_truncation_or_rewrite"] == "forbidden"
    assert "fail closed" in recovery["interior_corruption"]


def test_provenance_categories_privacy_and_no_docker_socket() -> None:
    provenance = _contract()["provenance_schema"]
    assert set(provenance["source_categories"]) == {
        "auto_detected",
        "launcher_injected",
        "unavailable",
        "prohibited_collection",
    }
    assert provenance["docker_socket_dependency"] == "forbidden"
    assert provenance["hostname_and_container_identity_may_be_disabled_or_redacted"] is True
    assert set(provenance["sensitive_collection"].values()) == {
        "prohibited_collection"
    }
    assert provenance["missing_image_identity"]["value"] is None


def test_legacy_identity_no_in_place_rewrite_and_unknown_major_failure() -> None:
    legacy = _contract()["legacy_compatibility"]
    assert legacy["legacy_schema_id"] == "legacy_flat_episode_v0"
    assert legacy["in_place_modification"] == "forbidden"
    assert legacy["unknown_major_reader_behavior"] == "fail_closed"
    assert "first-task compatibility" in legacy["legacy_metadata_task_semantics"]
    assert _contract()["schema_evolution"]["raw_in_place_migration"] == "forbidden"


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    raw = CONTRACT_PATH.read_text(encoding="utf-8")
    duplicate = raw.replace(
        '"schema_name": "team_sorting.recorder",',
        '"schema_name": "team_sorting.recorder",\n  "schema_name": "duplicate",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_recorder_contract(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constant_is_rejected(tmp_path: Path, constant: str) -> None:
    raw = CONTRACT_PATH.read_text(encoding="utf-8")
    damaged = raw.replace('"schema_version": 1', f'"schema_version": {constant}', 1)
    path = tmp_path / "nonfinite.json"
    path.write_text(damaged, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON constant"):
        load_recorder_contract(path)


def test_all_unresolved_records_are_complete_and_unique() -> None:
    unresolved = _contract()["unresolved"]
    assert len(unresolved) == 16
    assert len({item["key"] for item in unresolved}) == len(unresolved)
    required = {
        "key",
        "status",
        "why_unresolved",
        "affected_components",
        "required_test_or_decision",
        "must_not_assume",
    }
    assert all(required <= set(item) for item in unresolved)
    assert all(item["status"] == "unresolved" for item in unresolved)


def test_loaded_contract_is_deeply_read_only_and_fresh() -> None:
    first = _contract()
    second = _contract()
    assert first is not second
    with pytest.raises(TypeError):
        first["schema_version"] = 2
    with pytest.raises(TypeError):
        first["identity_contract"]["local_attempt_key"]["scope"] = "official"
    with pytest.raises(AttributeError):
        first["event_schema"]["event_types"].append("new_event")


def test_candidate_paths_are_stably_deduplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix"
    module_path, contract_path = _install_contract(
        prefix, "lib/python3.10/site-packages"
    )
    monkeypatch.setattr(recorder_contract_module, "__file__", str(module_path))
    monkeypatch.setattr(
        recorder_contract_module, "_ament_contract_candidate", lambda: contract_path
    )
    candidates = _candidate_paths()
    assert candidates[0] == contract_path
    assert candidates.count(contract_path) == 1
    assert candidates == tuple(dict.fromkeys(candidates))


@pytest.mark.parametrize(
    "module_relative",
    ["lib/python3.10/site-packages", "local/lib/python3.10/dist-packages"],
)
def test_installed_layout_loading_is_cwd_and_ament_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_relative: str
) -> None:
    prefix = tmp_path / "prefix"
    module_path, contract_path = _install_contract(prefix, module_relative)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    monkeypatch.setattr(recorder_contract_module, "__file__", str(module_path))
    monkeypatch.setattr(
        recorder_contract_module, "_ament_contract_candidate", lambda: None
    )
    assert recorder_contract_path() == contract_path
    assert load_recorder_contract()["schema_name"] == "team_sorting.recorder"


def test_installed_candidate_derivation_never_probes_root_share() -> None:
    candidates = _installed_share_candidates(
        Path(
            "/prefix/local/lib/python3.10/dist-packages/team_sorting/recorder_contract.py"
        )
    )
    assert Path(
        "/prefix/local/share/team_sorting/config/contracts/recorder_schema_v1.json"
    ) in candidates
    assert Path(
        "/share/team_sorting/config/contracts/recorder_schema_v1.json"
    ) not in candidates


def test_missing_contract_lists_every_attempted_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempted = (
        tmp_path / "first" / "recorder_schema_v1.json",
        tmp_path / "second" / "recorder_schema_v1.json",
    )
    monkeypatch.setattr(recorder_contract_module, "_candidate_paths", lambda: attempted)
    with pytest.raises(FileNotFoundError) as error:
        recorder_contract_path()
    assert all(str(path) in str(error.value) for path in attempted)


def test_setup_installs_both_contracts_to_same_share_directory() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"config/contracts/interface_v1.json"' in setup_text
    assert '"config/contracts/recorder_schema_v1.json"' in setup_text
    assert setup_text.count('"share/" + package_name + "/config/contracts"') == 1


def test_recorder_contract_sha256_matches_exact_bytes_and_is_stable() -> None:
    expected = hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    assert recorder_contract_sha256(CONTRACT_PATH) == expected
    assert recorder_contract_sha256(CONTRACT_PATH) == expected
    assert len(expected) == 64
    assert expected == expected.lower()
    assert set(expected) <= set("0123456789abcdef")


def test_planned_capabilities_never_claim_b1_runtime_implementation() -> None:
    capabilities = _contract()["planned_runtime_capabilities"]
    assert capabilities
    assert all(item["implemented_in_b1"] is False for item in capabilities)
    assert {item["implementation_owner"] for item in capabilities} == {
        "Commit B2",
        "Commit B3",
        "Commit C offline Indexer/QC",
    }
