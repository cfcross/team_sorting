"""Interface v1 machine-contract regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import team_sorting.interface_contract as interface_contract_module
from team_sorting.controller_manifest import MMK2_CONTROLLER_MANIFEST_V1
from team_sorting.interface_contract import (
    _candidate_paths,
    _installed_share_candidates,
    interface_contract_path,
    load_interface_contract,
    validate_interface_contract,
)
from team_sorting.interfaces import ACTION_NAMES, JOINT_NAMES


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "contracts" / "interface_v1.json"


def _raw() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _contract():
    return load_interface_contract(CONTRACT_PATH)


def _install_contract(prefix: Path, module_relative: str) -> tuple[Path, Path]:
    module_path = prefix / module_relative / "team_sorting" / "interface_contract.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    contract_path = prefix / "share" / "team_sorting" / "config" / "contracts" / "interface_v1.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(CONTRACT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return module_path, contract_path


def test_source_tree_default_loading_still_succeeds() -> None:
    assert interface_contract_path() == CONTRACT_PATH
    assert load_interface_contract()["schema_name"] == "team_sorting.interface"


def test_installed_share_candidates_cover_plain_site_packages_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "prefix"
    module_path, contract_path = _install_contract(prefix, "lib/python3.10/site-packages")
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    monkeypatch.setattr(interface_contract_module, "__file__", str(module_path))
    monkeypatch.setattr(interface_contract_module, "_ament_contract_candidate", lambda: None)

    assert interface_contract_path() == contract_path
    loaded = load_interface_contract()
    assert loaded["schema_name"] == "team_sorting.interface"
    assert loaded["schema_version"] == 1


def test_installed_share_candidates_cover_local_dist_packages_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_prefix = tmp_path / "prefix" / "local"
    module_path, contract_path = _install_contract(
        local_prefix, "lib/python3.10/dist-packages"
    )
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(tmp_path / "wrong-ament-prefix"))
    monkeypatch.setattr(interface_contract_module, "__file__", str(module_path))
    monkeypatch.setattr(interface_contract_module, "_ament_contract_candidate", lambda: None)

    assert interface_contract_path() == contract_path
    loaded = load_interface_contract()
    with pytest.raises(TypeError):
        loaded["schema_version"] = 2


def test_candidate_paths_are_stably_deduplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "prefix"
    module_path, contract_path = _install_contract(prefix, "lib/python3.10/site-packages")
    share_dir = contract_path.parents[2]
    monkeypatch.setattr(interface_contract_module, "__file__", str(module_path))
    monkeypatch.setattr(
        interface_contract_module, "_ament_contract_candidate", lambda: contract_path
    )

    candidates = _candidate_paths()
    assert candidates[0] == share_dir / "config" / "contracts" / "interface_v1.json"
    assert candidates.count(contract_path) == 1
    assert candidates == tuple(dict.fromkeys(candidates))


def test_missing_contract_error_lists_all_attempted_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = (
        tmp_path / "first" / "interface_v1.json",
        tmp_path / "second" / "interface_v1.json",
    )
    monkeypatch.setattr(interface_contract_module, "_candidate_paths", lambda: attempted)

    with pytest.raises(FileNotFoundError) as error:
        interface_contract_path()
    assert all(str(path) in str(error.value) for path in attempted)


def test_installed_candidate_derivation_is_finite_and_does_not_probe_root_share() -> None:
    candidates = _installed_share_candidates(
        Path("/prefix/local/lib/python3.10/dist-packages/team_sorting/interface_contract.py")
    )
    assert Path("/prefix/local/share/team_sorting/config/contracts/interface_v1.json") in candidates
    assert Path("/prefix/share/team_sorting/config/contracts/interface_v1.json") in candidates
    assert Path("/share/team_sorting/config/contracts/interface_v1.json") not in candidates


def test_json_is_loadable_and_has_stable_schema_identity() -> None:
    contract = _contract()
    assert contract["schema_name"] == "team_sorting.interface"
    assert contract["schema_version"] == 1
    assert type(contract["schema_version"]) is int
    assert contract["contract_status"] == "frozen"


def test_schema_version_rejects_bool_and_other_versions() -> None:
    for value in (True, 2):
        payload = _raw()
        payload["schema_version"] = value
        with pytest.raises(ValueError, match="schema_version"):
            validate_interface_contract(payload)


def test_action_is_exactly_19_contiguous_canonical_dimensions() -> None:
    actions = _contract()["action_contract"]["dimensions"]
    assert len(actions) == 19
    assert tuple(action["index"] for action in actions) == tuple(range(19))
    assert tuple(action["name"] for action in actions) == ACTION_NAMES


def test_joint_state_is_exactly_17_canonical_dimensions() -> None:
    joint_contract = _contract()["joint_state_contract"]
    assert joint_contract["dimension_count"] == 17
    assert joint_contract["names"] == JOINT_NAMES


def test_action_joint_semantic_mapping_is_explicit_not_name_equality() -> None:
    contract = _contract()
    actions = contract["action_contract"]["dimensions"]
    assert tuple(item["joint_state_name"] for item in actions[2:]) == JOINT_NAMES
    assert ACTION_NAMES[2:] != JOINT_NAMES
    assert "distinct" in contract["joint_state_contract"]["semantic_alignment"]


def test_base_modes_units_fields_and_non_wheel_semantics() -> None:
    action_contract = _contract()["action_contract"]
    base_v, base_w = action_contract["dimensions"][:2]
    assert (base_v["control_mode"], base_v["unit"], base_v["official_message_field"]) == (
        "velocity", "m/s", "Twist.linear.x"
    )
    assert (base_w["control_mode"], base_w["unit"], base_w["official_message_field"]) == (
        "velocity", "rad/s", "Twist.angular.z"
    )
    assert action_contract["base_is_not_direct_wheel_command"] is True
    assert action_contract["official_cmd_vel_accepted_limits_status"] == "unresolved"
    assert base_v["runtime_range_status"] == base_w["runtime_range_status"] == "unresolved"


def test_position_modes_units_and_zero_hold_warning() -> None:
    actions = _contract()["action_contract"]["dimensions"]
    assert actions[2]["control_mode"] == "absolute_position"
    assert actions[2]["unit"] == "m"
    assert all(item["control_mode"] == "absolute_position" for item in (*actions[3:11], *actions[12:18]))
    assert all(item["unit"] == "rad" for item in (*actions[3:11], *actions[12:18]))
    assert all("0不是停止" in item["hold_semantics"] for item in actions[2:])


def test_safe_and_runtime_ranges_match_controller_manifest() -> None:
    actions = _contract()["action_contract"]["dimensions"]
    manifest = MMK2_CONTROLLER_MANIFEST_V1.actions
    assert tuple((x["safe_min"], x["safe_max"]) for x in actions) == tuple(
        (x.safe_min, x.safe_max) for x in manifest
    )
    assert tuple((x["runtime_min"], x["runtime_max"]) for x in actions) == tuple(
        (x.runtime_min, x.runtime_max) for x in manifest
    )


def test_gripper_ranges_and_direction_remain_unresolved() -> None:
    actions = _contract()["action_contract"]["dimensions"]
    for index in (11, 18):
        assert (actions[index]["safe_min"], actions[index]["safe_max"]) == (0, 1)
        assert actions[index]["unit"] == "dimensionless"
        assert actions[index]["status"] == "unresolved"
    assert _contract()["action_contract"]["gripper_open_close_direction_status"] == "unresolved"


def test_joint_missing_velocity_and_effort_warning_is_frozen() -> None:
    joint_contract = _contract()["joint_state_contract"]
    assert "zero tuple" in joint_contract["empty_velocity_behavior"]
    assert "zero tuple" in joint_contract["empty_effort_behavior"]
    assert "不一定表示真实物理零值" in joint_contract["missing_field_warning"]
    assert all(item["effort_unit_status"] == "unresolved" for item in joint_contract["dimensions"])


def test_five_official_topic_widths_and_slices_are_exact() -> None:
    topics = _contract()["official_control_topics"]
    assert tuple(
        (group["group"], group["payload_width"], group["action_slice"])
        for group in topics["groups"]
    ) == (
        ("base", 2, (0, 2)),
        ("spine", 1, (2, 3)),
        ("head", 2, (3, 5)),
        ("left_arm", 7, (5, 12)),
        ("right_arm", 7, (12, 19)),
    )
    assert topics["unique_arbiter"].endswith("ActionMux")
    assert topics["unique_production_boundary"].endswith("OfficialCommandPublisher")
    assert topics["direct_publish_by_team_modules_forbidden"] is True


def test_observations_preserve_camera_info_odom_tf_and_rate_evidence() -> None:
    observations = {item["canonical_name"]: item for item in _contract()["observation_contract"]["observations"]}
    camera = observations["head_camera_info"]
    assert camera["raw_header_stamp"] == "zero"
    assert camera["raw_header_frame_id"] == ""
    assert "do not claim" in camera["adapter_behavior"]
    assert observations["odom"]["frame"] == {"raw": "/odom", "normalized_at_ros_adapter": "odom"}
    assert observations["tf_odom_base"]["recorder_v1_collection"] == "planned_required"
    assert all("not protocol guarantee" in item["expected_rate_source"] for item in observations.values() if item["expected_rate_hz"] is not None)


def test_world_and_odom_are_not_declared_equal() -> None:
    frames = _contract()["frame_contract"]
    assert frames["world_to_odom_relationship"] == "unresolved"
    assert frames["world_equals_odom"] is False
    assert frames["must_not_assume_world_equals_odom"] is True
    assert frames["task_place_point_frame"] == "world"
    assert frames["planning_frame"] == "odom"


def test_timestamp_types_and_invalid_value_rules_are_explicit() -> None:
    times = _contract()["timestamp_contract"]
    assert "non-bool" in times["nanosecond_validation"]
    by_name = {item["name"]: item for item in times["types"]}
    assert set(by_name) == {"sensor_timestamp_ns", "receive_timestamp_ns", "receive_monotonic_ns", "generated_timestamp_ns", "sim_elapsed_s"}
    assert by_name["receive_monotonic_ns"]["cross_process_comparable"] is False
    assert by_name["receive_monotonic_ns"]["observation_action_pairing"] is False
    assert by_name["sim_elapsed_s"]["unit"] == "s"


def test_settled_attempt_is_not_current_one_based_attempt() -> None:
    identity = _contract()["competition_identity_contract"]
    description = identity["fields"]["settled_attempt_count"]
    assert "already settled" in description
    assert "not the current 1-based" in description
    assert identity["local_attempt_key"]["official_attempt_id"] is False


def test_recorder_segment_is_not_training_episode() -> None:
    identity = _contract()["competition_identity_contract"]
    assert identity["recorder_segment_is_training_episode"] is False
    segment = next(item for item in identity["entities"] if item["name"] == "Recorder Segment")
    assert "not a Training Episode" in segment["meaning"]


def test_action_recording_uses_precise_layers_and_disclaimers() -> None:
    recording = _contract()["action_recording_contract"]
    assert tuple(item["name"] for item in recording["layers"]) == (
        "proposed_action", "selected_action", "dispatched_action",
        "publisher_call_attempted", "publisher_call_succeeded",
        "publisher_failure_reason", "execution_feedback",
    )
    assert recording["ambiguous_action_field_forbidden"] is True
    assert "does not prove DDS delivery" in recording["publisher_success_disclaimer"]
    assert "not eligible" in recording["observe_only_bc_eligibility"]
    assert recording["training_eligibility"] == "derived only by offline QC"


def test_every_frozen_public_interface_has_a_version() -> None:
    items = _contract()["public_interfaces"]["items"]
    assert len(items) == 30
    assert {item["status"] for item in items} == {"frozen", "provisional", "restricted"}
    assert all(type(item["version"]) is int and item["version"] >= 1 for item in items if item["status"] == "frozen")
    assert next(item for item in items if item["name"] == "ExternalCandidate")["status"] == "restricted"


def test_unresolved_items_have_evidence_and_fail_closed_fields() -> None:
    unresolved = _contract()["unresolved"]
    assert len(unresolved) >= 11
    assert all(item["status"] == "unresolved" for item in unresolved)
    assert all(item["required_test"] and item["must_not_assume"] and item["blocking_roles"] for item in unresolved)


def test_json_has_no_nonstandard_nan_or_infinity_tokens() -> None:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def test_loader_result_is_deeply_read_only_and_not_shared_mutable_state() -> None:
    first = _contract()
    second = _contract()
    assert first is not second
    with pytest.raises(TypeError):
        first["schema_version"] = 2
    with pytest.raises(TypeError):
        first["action_contract"]["dimensions"][0]["name"] = "changed"
    with pytest.raises(AttributeError):
        first["unresolved"].append("changed")


def test_loader_rejects_null_range_without_unresolved_status() -> None:
    payload = _raw()
    payload["action_contract"]["dimensions"][2]["runtime_min"] = None
    with pytest.raises(ValueError, match="null requires unresolved"):
        validate_interface_contract(payload)


def test_install_data_includes_contract_at_share_path() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"share/" + package_name + "/config/contracts"' in setup_text
    assert '"config/contracts/interface_v1.json"' in setup_text
    assert CONTRACT_PATH.is_file()


def test_document_freezes_major_upgrade_and_review_rules() -> None:
    text = (ROOT / "docs" / "interface_v1.md").read_text(encoding="utf-8")
    for term in ("字段重命名", "单位变化", "索引变化", "时间源变化", "null语义变化"):
        assert term in text
    assert "必须升级major" in text
    assert "新增可选字段必须定义清晰的缺失语义" in text
    assert "未经队长审查不得修改Interface v1" in text


def test_contract_does_not_invent_generic_action_or_navigation_types() -> None:
    names = {item["name"] for item in _contract()["public_interfaces"]["items"]}
    assert "ActionCandidate" not in names
    assert "NavigationCommand" not in names
