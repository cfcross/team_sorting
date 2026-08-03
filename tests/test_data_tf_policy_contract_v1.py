"""B3 Data/TF Policy v1 strict machine-contract regressions."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import team_sorting.data_tf_policy_contract as policy_module
from team_sorting.data_tf_policy_contract import (
    _candidate_paths,
    _installed_share_candidates,
    data_tf_policy_contract_path,
    data_tf_policy_contract_sha256,
    default_data_tf_policy_contract_path,
    load_data_tf_policy_contract,
    validate_data_tf_policy_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "contracts" / "data_tf_policy_v1.json"
INTERFACE_PATH = ROOT / "config" / "contracts" / "interface_v1.json"
RECORDER_PATH = ROOT / "config" / "contracts" / "recorder_schema_v1.json"
EXPECTED_POLICY_SHA256 = "982934579d816d67c63c1ff8938ea49c54982cf7d75472a58c00fe6a9cefae80"
EXPECTED_INTERFACE_SHA256 = "a86548b2a43581af70b8d585d523a06bb97a8d96e1fe52950097b12d061fdaea"
EXPECTED_RECORDER_SHA256 = "e7965c34a38c11d551d9943d8d614c05bc8e28e186432ad5ff4d0eed243225cf"
EXPECTED_TOPICS = {
    "/material/instruction": "std_msgs/msg/String",
    "/head_camera/color/image_raw": "sensor_msgs/msg/Image",
    "/head_camera/aligned_depth_to_color/image_raw": "sensor_msgs/msg/Image",
    "/head_camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/slamware_ros_sdk_server_node/odom": "nav_msgs/msg/Odometry",
    "/joint_states": "sensor_msgs/msg/JointState",
    "/team/object_estimates": "vision_msgs/msg/Detection3DArray",
    "/team/fsm_status": "std_msgs/msg/String",
    "/team/competition_context": "std_msgs/msg/String",
    "/team/final_action": "std_msgs/msg/String",
    "/referee/taskinfo": "std_msgs/msg/String",
    "/referee/gameinfo": "std_msgs/msg/String",
    "/referee/score": "std_msgs/msg/Int32",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
}


def _payload() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _contract():
    return load_data_tf_policy_contract(CONTRACT_PATH)


def _profiles(contract=None):
    source = _contract() if contract is None else contract
    return {profile["profile_id"]: profile for profile in source["profiles"]}


def _install_contract(prefix: Path, module_relative: str) -> tuple[Path, Path]:
    module_path = prefix / module_relative / "team_sorting" / "data_tf_policy_contract.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    share_contracts = prefix / "share" / "team_sorting" / "config" / "contracts"
    share_contracts.mkdir(parents=True)
    policy_path = share_contracts / CONTRACT_PATH.name
    policy_path.write_bytes(CONTRACT_PATH.read_bytes())
    (share_contracts / INTERFACE_PATH.name).write_bytes(INTERFACE_PATH.read_bytes())
    (share_contracts / RECORDER_PATH.name).write_bytes(RECORDER_PATH.read_bytes())
    return module_path, policy_path


def test_default_contract_loads_and_has_frozen_identity() -> None:
    assert data_tf_policy_contract_path() == CONTRACT_PATH
    assert default_data_tf_policy_contract_path() == CONTRACT_PATH
    contract = _contract()
    assert contract["schema_name"] == "team_sorting.data_tf_policy"
    assert contract["schema_version"] == 1
    assert type(contract["schema_version"]) is int
    assert contract["policy_id"] == "data_tf_policy_v1"
    assert contract["implementation_status"] == "contract_only"


def test_frozen_dependencies_match_exact_resources_and_hashes() -> None:
    contract = _contract()
    interface = contract["interface_contract"]
    recorder = contract["recorder_contract"]
    assert interface["path"] == "config/contracts/interface_v1.json"
    assert interface["schema_name"] == "team_sorting.interface"
    assert interface["schema_version"] == 1
    assert interface["frozen_sha256"] == EXPECTED_INTERFACE_SHA256
    assert recorder["path"] == "config/contracts/recorder_schema_v1.json"
    assert recorder["schema_name"] == "team_sorting.recorder"
    assert recorder["schema_version"] == 1
    assert recorder["frozen_sha256"] == EXPECTED_RECORDER_SHA256
    assert hashlib.sha256(INTERFACE_PATH.read_bytes()).hexdigest() == EXPECTED_INTERFACE_SHA256
    assert hashlib.sha256(RECORDER_PATH.read_bytes()).hexdigest() == EXPECTED_RECORDER_SHA256


def test_policy_sha256_matches_exact_bytes_and_is_stable() -> None:
    assert data_tf_policy_contract_sha256(CONTRACT_PATH) == EXPECTED_POLICY_SHA256
    assert data_tf_policy_contract_sha256(CONTRACT_PATH) == EXPECTED_POLICY_SHA256
    assert len(EXPECTED_POLICY_SHA256) == 64
    assert set(EXPECTED_POLICY_SHA256) <= set("0123456789abcdef")


def test_all_15_raw_and_tf_topics_have_exact_types_and_unique_names() -> None:
    topic_policy = _contract()["topic_policy"]
    topics = tuple(topic_policy["current_raw_baseline"]) + tuple(
        topic_policy["b3_target_topics"]
    )
    actual = {item["name"]: item["message_type"] for item in topics}
    assert len(topics) == 15
    assert len(actual) == 15
    assert actual == EXPECTED_TOPICS


def test_action_dispatch_remains_authoritative_structured_artifact() -> None:
    authority = _contract()["topic_policy"]["action_dispatch_authority"]
    assert authority["topic"] == "/team/action_dispatch"
    assert authority["artifact"] == "action_dispatches.jsonl"
    assert authority["artifact_is_authoritative"] is True
    assert authority["rosbag_duplication_required_in_b3a"] is False


def test_three_profiles_are_canonical_ordered_and_unique() -> None:
    profiles = _contract()["profiles"]
    ids = tuple(profile["profile_id"] for profile in profiles)
    assert ids == (
        "debug_audit",
        "formal_collection_candidate",
        "fast_regression",
    )
    assert len(ids) == len(set(ids))


def test_debug_profile_is_validated_raw_baseline() -> None:
    debug = _profiles()["debug_audit"]
    assert debug["validation_status"] == "validated_raw_baseline"
    assert debug["benchmark_required"] is False
    assert debug["target_rates_hz"]["rgb"] == 24.0
    assert debug["target_rates_hz"]["depth"] == 24.0
    assert debug["training_allowed"] is False


def test_formal_profile_is_provisional_benchmark_candidate_not_optimal_claim() -> None:
    formal = _profiles()["formal_collection_candidate"]
    assert formal["validation_status"] == "provisional"
    assert formal["benchmark_required"] is True
    assert formal["target_rates_hz"]["rgb"] == 12.0
    assert formal["target_rates_hz"]["depth"] == 12.0
    assert formal["target_rates_hz"]["derived_training"] == 10.0
    assert "claiming_12hz_is_optimal" in formal["prohibited_use"]
    assert "claiming_10hz_is_optimal" in formal["prohibited_use"]


def test_fast_profile_is_provisional_and_training_forbidden() -> None:
    fast = _profiles()["fast_regression"]
    assert fast["validation_status"] == "provisional"
    assert fast["target_rates_hz"]["rgb"] == 2.0
    assert fast["target_rates_hz"]["joint_state"] == 10.0
    assert fast["training_allowed"] is False
    assert set(fast["expected_use"]) == {
        "lifecycle_smoke", "tf_smoke", "indexer_smoke", "qc_smoke"
    }


def test_dynamic_tf_qos_is_compatible_best_effort_profile() -> None:
    dynamic = _contract()["tf_policy"]["dynamic"]
    assert dynamic["topic"] == "/tf"
    assert dynamic["message_type"] == "tf2_msgs/msg/TFMessage"
    assert dynamic["role"] == "dynamic_transform"
    assert dict(dynamic["subscription_qos"]) == {
        "reliability": "best_effort",
        "durability": "volatile",
        "history": "keep_last",
        "depth": 100,
    }
    assert dynamic["observed_publisher_reliability"] == "reliable"


def test_static_tf_qos_supports_late_join() -> None:
    static = _contract()["tf_policy"]["static"]
    assert static["late_join_required"] is True
    assert dict(static["subscription_qos"]) == {
        "reliability": "reliable",
        "durability": "transient_local",
        "history": "keep_last",
        "depth": 1,
    }


def test_absent_tf_static_publisher_is_not_globally_fatal() -> None:
    contract = _contract()
    static = contract["tf_policy"]["static"]
    rules = contract["qc_policy"]["global_rules"]
    assert static["current_fixed_scene_publisher_present"] is False
    assert static["publisher_absence_is_global_fatal"] is False
    assert rules["tf_static_publisher_absent_is_global_fatal"] is False
    assert rules["missing_profile_required_static_edge_code"] == (
        "tf_static_required_edge_missing"
    )


def test_world_and_odom_are_not_declared_equivalent() -> None:
    relation = _contract()["tf_policy"]["world_odom_relationship"]
    assert relation["status"] == "unknown"
    assert relation["equivalent"] is False
    assert relation["assumption_forbidden"] is True


def test_known_dynamic_relationship_is_odom_to_base_link() -> None:
    relation = _contract()["tf_policy"]["known_dynamic_relationship"]
    assert dict(relation) == {
        "parent_frame": "odom",
        "child_frame": "base_link",
        "status": "observed",
    }


def test_frame_normalization_preserves_raw_and_normalized_values() -> None:
    frames = _contract()["frame_policy"]
    normalization = frames["normalization"]
    assert normalization["raw_field"] == "raw_frame_id"
    assert normalization["normalized_field"] == "normalized_frame_id"
    assert normalization["normalization_rule"] == "strip_leading_slashes"
    assert normalization["preserve_raw_and_normalized"] is True
    assert normalization["empty_frame_guess_without_evidence"] == "forbidden"
    assert dict(frames["odom_frame"]) == {
        "historical_raw_frame_id": "/odom",
        "normalized_frame_id": "odom",
    }


def test_camera_info_empty_frame_binding_is_explicit_and_non_mutating() -> None:
    camera = _contract()["frame_policy"]["camera_info_empty_frame"]
    assert camera["raw_frame_id"] == ""
    assert camera["effective_frame_binding_allowed"] is True
    assert camera["effective_frame_id"] == "head_camera"
    assert camera["binding_source"] == "synchronized_rgb_depth"
    assert camera["raw_message_rewrite"] == "forbidden"


def test_raw_is_immutable_and_derived_outputs_are_separate_and_provenanced() -> None:
    policy = _contract()["raw_derived_policy"]
    assert policy["raw_in_place_modification"] == "forbidden"
    assert policy["derived_root_template"] == (
        "derived/indexer_v1/<deterministic_build_id>/"
    )
    assert tuple(policy["derived_outputs"]) == (
        "index_build.json",
        "dataset_index.jsonl",
        "segment_qc.json",
        "run_qc.json",
        "sample_index.jsonl",
        "training_manifest.jsonl",
    )
    assert "source_relative_path" in policy["required_provenance_fields"]
    assert "source_sha256" in policy["required_provenance_fields"]
    assert policy["host_absolute_path_is_identity"] is False
    assert policy["automatic_raw_deletion"] == "forbidden"


def test_online_image_and_zstd_defaults_remain_disabled() -> None:
    storage = _contract()["storage_policy"]
    assert dict(storage["current_online_defaults"]) == {
        "storage_id": "sqlite3",
        "rosbag_compression": "none",
        "online_rgb_codec": "none",
        "online_depth_codec": "none",
    }
    assert storage["zstd"]["environment_plugin_present"] is True
    assert storage["zstd"]["competition_client_benchmark_status"] == (
        "not_validated_under_load"
    )
    assert storage["zstd"]["default_enabled"] is False


def test_depth_lossy_encoding_is_forbidden_everywhere() -> None:
    contract = _contract()
    assert contract["storage_policy"]["derived_codecs"]["depth_candidate"] == (
        "png16_lossless"
    )
    assert contract["storage_policy"]["derived_codecs"]["depth_lossy_forbidden"] is True
    assert all(profile["depth_storage"]["lossy_allowed"] is False for profile in contract["profiles"])


def test_downsampling_is_only_a_traceable_future_policy_target() -> None:
    downsampling = _contract()["storage_policy"]["downsampling"]
    assert downsampling["implemented_in_b3a"] is False
    assert downsampling["profile_rates_are_policy_targets_only"] is True
    assert downsampling["preserve_original_ros_timestamp"] is True
    assert downsampling["preserve_source_message_identity"] is True
    assert downsampling["unrecorded_every_n_rule"] == "forbidden"


def test_action_layers_remain_distinct_and_selected_is_not_dispatched() -> None:
    action = _contract()["action_label_policy"]
    assert tuple(action["layers"]) == (
        "proposed_action",
        "selected_action",
        "dispatched_action",
        "publisher_call_succeeded",
        "execution_feedback",
    )
    assert action["selected_action_source"] == "FinalAction"
    assert action["selected_action_alone_is_formal_bc_label"] is False
    assert action["publisher_success_proves_dds_controller_or_execution"] is False


def test_observe_only_is_not_formal_bc_eligible() -> None:
    action = _contract()["action_label_policy"]
    assert set(action["observe_only"]["allowed_uses"]) == {
        "perception", "state", "lifecycle", "diagnostics"
    }
    assert action["observe_only"]["formal_bc_allowed"] is False
    assert _contract()["qc_policy"]["global_rules"][
        "observe_only_formal_bc_eligibility"
    ] == "ineligible"


def test_formal_bc_requires_dispatch_context_safety_and_feedback() -> None:
    requirements = _contract()["action_label_policy"]["formal_bc_requirements"]
    assert all(requirements.values())
    assert set(requirements) == {
        "observe_only_must_be_false",
        "exact_dispatched_action_required",
        "action_contract_match_required",
        "valid_context_required",
        "finished_must_be_false",
        "safety_gate_semantics_required",
        "subsequent_execution_feedback_required",
    }


def test_r0_smoke_does_not_claim_formal_bc_qualification() -> None:
    smoke = _contract()["action_label_policy"]["r0_smoke"]
    assert smoke["controlled_single_trajectory_allowed"] is True
    assert smoke["strict_annotation_required"] is True
    assert smoke["implies_formal_bc_qualification"] is False


def test_qc_severity_eligibility_and_finding_codes_are_frozen_unique() -> None:
    qc = _contract()["qc_policy"]
    assert tuple(qc["severity"]) == ("fatal", "error", "warning", "info")
    assert tuple(qc["eligibility"]) == (
        "eligible", "conditionally_eligible", "ineligible"
    )
    codes = tuple(qc["finding_codes"])
    assert len(codes) == 26
    assert len(codes) == len(set(codes))
    assert {
        "active_marker_remaining",
        "rosbag_exit_nonzero",
        "tf_static_required_edge_missing",
        "observe_only_not_formal_bc",
        "execution_feedback_missing",
    } <= set(codes)


def test_b3a_implementation_boundaries_never_claim_runtime_work() -> None:
    boundaries = _contract()["implementation_boundaries"]
    runtime_flags = {key: value for key, value in boundaries.items() if key != "next_phases"}
    assert runtime_flags
    assert all(value is False for value in runtime_flags.values())
    assert set(boundaries["next_phases"]) == {"b3b", "b3c", "b3d"}


def test_loaded_contract_is_deeply_read_only_and_fresh() -> None:
    first = _contract()
    second = _contract()
    assert first is not second
    with pytest.raises(TypeError):
        first["schema_version"] = 2
    with pytest.raises(TypeError):
        first["tf_policy"]["dynamic"]["topic"] = "/changed"
    with pytest.raises(AttributeError):
        first["profiles"].append("changed")


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    raw = CONTRACT_PATH.read_text(encoding="utf-8")
    duplicate = raw.replace(
        '"schema_name": "team_sorting.data_tf_policy",',
        '"schema_name": "team_sorting.data_tf_policy",\n  "schema_name": "duplicate",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_data_tf_policy_contract(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_constant_is_rejected(tmp_path: Path, constant: str) -> None:
    raw = CONTRACT_PATH.read_text(encoding="utf-8")
    damaged = raw.replace('"schema_version": 1', f'"schema_version": {constant}', 1)
    path = tmp_path / "nonfinite.json"
    path.write_text(damaged, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON constant"):
        load_data_tf_policy_contract(path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.__setitem__("schema_version", True), "integer and not bool"),
        (lambda p: p["profiles"][0]["target_rates_hz"].__setitem__("rgb", True), "not bool"),
        (lambda p: p["tf_policy"]["dynamic"]["subscription_qos"].__setitem__("depth", True), "integer and not bool"),
    ],
)
def test_bool_cannot_impersonate_integer_or_number(mutate, match: str) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        validate_data_tf_policy_contract(payload, contract_path=CONTRACT_PATH)


@pytest.mark.parametrize("rate", [0, -1.0, float("nan"), float("inf"), "12"])
def test_invalid_profile_rate_is_rejected(rate: object) -> None:
    payload = _payload()
    payload["profiles"][1]["target_rates_hz"]["rgb"] = rate
    with pytest.raises(ValueError, match="finite positive number"):
        validate_data_tf_policy_contract(payload, contract_path=CONTRACT_PATH)


@pytest.mark.parametrize("topic", ["tf", "/bad topic", "/double//slash", "/trailing/"])
def test_invalid_topic_is_rejected(topic: str) -> None:
    payload = _payload()
    payload["topic_policy"]["b3_target_topics"][0]["name"] = topic
    with pytest.raises(ValueError, match="canonical absolute ROS topic"):
        validate_data_tf_policy_contract(payload, contract_path=CONTRACT_PATH)


def test_duplicate_topic_is_rejected() -> None:
    payload = _payload()
    payload["topic_policy"]["b3_target_topics"][1]["name"] = "/tf"
    with pytest.raises(ValueError, match="topic names must be unique"):
        validate_data_tf_policy_contract(payload, contract_path=CONTRACT_PATH)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reliability", "sometimes"),
        ("durability", "persistent"),
        ("history", "unknown"),
    ],
)
def test_invalid_qos_enum_is_rejected(field: str, value: str) -> None:
    payload = _payload()
    payload["tf_policy"]["dynamic"]["subscription_qos"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_data_tf_policy_contract(payload, contract_path=CONTRACT_PATH)


def test_wrong_referenced_sha256_is_rejected() -> None:
    payload = _payload()
    payload["interface_contract"]["frozen_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen_sha256"):
        validate_data_tf_policy_contract(payload, contract_path=CONTRACT_PATH)


def test_actual_dependency_bytes_are_rehashed_in_installed_layout(tmp_path: Path) -> None:
    _, policy_path = _install_contract(tmp_path / "prefix", "lib/python3.10/site-packages")
    dependency = policy_path.parent / "interface_v1.json"
    dependency.write_bytes(dependency.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="actual SHA256 mismatch"):
        load_data_tf_policy_contract(policy_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.pop("policy_id"), "missing=.*policy_id"),
        (lambda p: p.__setitem__("extra", True), "extra=.*extra"),
        (lambda p: p["profiles"][0].__setitem__("extra", True), "extra=.*extra"),
        (lambda p: p["tf_policy"]["dynamic"]["subscription_qos"].__setitem__("extra", True), "extra=.*extra"),
        (lambda p: p.__setitem__("schema_name", "unknown.schema"), "schema_name"),
        (lambda p: p.__setitem__("schema_version", 2), "schema_version"),
    ],
)
def test_missing_extra_or_unknown_schema_fields_fail_closed(mutate, match: str) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        validate_data_tf_policy_contract(payload, contract_path=CONTRACT_PATH)


@pytest.mark.parametrize(
    "module_relative",
    ["lib/python3.10/site-packages", "local/lib/python3.10/dist-packages"],
)
def test_installed_layout_loading_is_cwd_and_ament_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_relative: str,
) -> None:
    module_path, policy_path = _install_contract(tmp_path / "prefix", module_relative)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    monkeypatch.setattr(policy_module, "__file__", str(module_path))
    monkeypatch.setattr(policy_module, "_ament_contract_candidate", lambda: None)
    assert data_tf_policy_contract_path() == policy_path
    assert load_data_tf_policy_contract()["policy_id"] == "data_tf_policy_v1"


def test_candidate_paths_are_deduplicated_and_ament_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path, policy_path = _install_contract(
        tmp_path / "prefix", "lib/python3.10/site-packages"
    )
    monkeypatch.setattr(policy_module, "__file__", str(module_path))
    monkeypatch.setattr(policy_module, "_ament_contract_candidate", lambda: policy_path)
    candidates = _candidate_paths()
    assert candidates[0] == policy_path
    assert candidates.count(policy_path) == 1
    assert candidates == tuple(dict.fromkeys(candidates))


def test_installed_candidates_never_probe_root_share() -> None:
    candidates = _installed_share_candidates(
        Path("/prefix/local/lib/python3.10/dist-packages/team_sorting/data_tf_policy_contract.py")
    )
    assert Path(
        "/prefix/local/share/team_sorting/config/contracts/data_tf_policy_v1.json"
    ) in candidates
    assert Path("/share/team_sorting/config/contracts/data_tf_policy_v1.json") not in candidates


def test_missing_contract_lists_all_attempted_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = (tmp_path / "first.json", tmp_path / "second.json")
    monkeypatch.setattr(policy_module, "_candidate_paths", lambda: attempted)
    with pytest.raises(FileNotFoundError) as error:
        data_tf_policy_contract_path()
    assert all(str(path) in str(error.value) for path in attempted)


def test_setup_installs_new_contract_and_document() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"config/contracts/data_tf_policy_v1.json"' in setup_text
    assert '"docs/data_tf_policy_v1.md"' in setup_text
    assert '"share/" + package_name + "/docs"' in setup_text
    assert setup_text.count('"share/" + package_name + "/config/contracts"') == 1


def test_readme_links_policy_and_limits_b3b_b3c_runtime_claims() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "config/contracts/data_tf_policy_v1.json" in readme
    assert "docs/data_tf_policy_v1.md" in readme
    assert "B3B第一步已把`/tf`和`/tf_static`" in readme
    assert "压缩、降频和训练样本生成仍未实现" in readme
    assert "只读Indexer/QC见下一节" in readme


def test_existing_frozen_contract_files_remain_exact() -> None:
    assert hashlib.sha256(INTERFACE_PATH.read_bytes()).hexdigest() == EXPECTED_INTERFACE_SHA256
    assert hashlib.sha256(RECORDER_PATH.read_bytes()).hexdigest() == EXPECTED_RECORDER_SHA256


def test_b3b_adds_only_tf_without_compression_to_runtime_config() -> None:
    config = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    recorder = (ROOT / "team_sorting" / "recorder.py").read_text(encoding="utf-8")
    assert config.count('    - "/tf"') == 1
    assert config.count('    - "/tf_static"') == 1
    assert "--compression-mode" not in recorder
    assert "--compression-format" not in recorder
