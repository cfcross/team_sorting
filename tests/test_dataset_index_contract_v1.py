from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

import team_sorting.dataset_index_contract as contract_module
from team_sorting.dataset_index_contract import (
    dataset_index_contract_path,
    dataset_index_contract_sha256,
    load_dataset_index_contract,
    validate_dataset_index_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/contracts/dataset_index_v1.json"
DEPENDENCIES = {
    "interface_v1.json": "a86548b2a43581af70b8d585d523a06bb97a8d96e1fe52950097b12d061fdaea",
    "recorder_schema_v1.json": "e7965c34a38c11d551d9943d8d614c05bc8e28e186432ad5ff4d0eed243225cf",
    "data_tf_policy_v1.json": "982934579d816d67c63c1ff8938ea49c54982cf7d75472a58c00fe6a9cefae80",
}
EXPECTED_CONTRACT_SHA256 = "64493d8979e3fb545aa8b31624897a97900967731d30a0e3bf43c286148e785e"


def _payload() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_default_contract_identity_and_stable_sha() -> None:
    contract = load_dataset_index_contract()
    assert dataset_index_contract_path() == CONTRACT
    assert contract["schema_name"] == "team_sorting.dataset_index"
    assert contract["schema_version"] == 1
    assert contract["contract_id"] == "dataset_index_v1"
    assert contract["implementation_status"] == "implemented_b3c_core"
    assert dataset_index_contract_sha256() == EXPECTED_CONTRACT_SHA256


def test_dependencies_are_rehashed_from_real_resources() -> None:
    load_dataset_index_contract()
    for name, digest in DEPENDENCIES.items():
        assert hashlib.sha256((ROOT / "config/contracts" / name).read_bytes()).hexdigest() == digest


def test_layout_raw_and_deferred_boundaries_are_frozen() -> None:
    contract = load_dataset_index_contract()
    assert contract["layout"]["implemented_outputs"] == (
        "index_build.json", "dataset_index.jsonl", "segment_qc.json", "run_qc.json"
    )
    assert contract["layout"]["prohibited_outputs"] == ("sample_index.jsonl", "training_manifest.jsonl")
    assert contract["raw_policy"]["immutable"] is True
    assert contract["raw_policy"]["exclude_derived_from_sources"] is True
    assert "execution_feedback_window_pairing" in contract["deferred"]


def test_finding_and_eligibility_enums_are_exact() -> None:
    contract = load_dataset_index_contract()
    assert contract["finding_contract"]["severity"] == ("fatal", "error", "warning", "info")
    assert contract["finding_contract"]["evaluation_status"] == ("pass", "fail", "not_applicable", "not_evaluated")
    assert contract["finding_contract"]["not_evaluated_is_pass"] is False
    assert contract["eligibility"]["use_cases"] == ("diagnostic", "perception", "formal_bc")
    assert contract["eligibility"]["formal_bc_eligible_in_v1"] is False
    assert contract["finding_contract"]["fail_blocking_use_cases_mandatory"] is True
    assert contract["finding_contract"]["blocking_scopes"] == ("segment", "run")
    assert contract["eligibility"]["tf_dynamic_missing_blocks"] == ("formal_bc",)
    assert contract["action_contract"]["invalid_record_always_finding"] is True
    assert contract["publication"]["failure_may_remove_reused_build"] is False
    assert contract["publication"]["existing_build_exact_files"] == (
        "index_build.json", "dataset_index.jsonl", "segment_qc.json", "run_qc.json"
    )
    assert contract["publication"]["existing_manifest_revalidated_against_current_material"] is True
    assert contract["publication"]["cleanup_failure_preserves_primary_error"] is True
    assert contract["recovery_integrity"]["silent_ignore_allowed"] is False
    assert contract["recovery_integrity"]["invalid_artifact_policy"] == "fail_closed"
    assert contract["source_validation"]["schema_descriptors_are_authoritative"] is True
    assert contract["topic_requirements"]["action_dispatch_jsonl_authoritative"] is True
    assert contract["qc_config"]["required_static_edges_v1"] == "empty_only"
    assert contract["cli_errors"]["failure_exit_code"] == 2


def test_loaded_graph_is_deeply_immutable_and_fresh() -> None:
    first = load_dataset_index_contract()
    second = load_dataset_index_contract()
    assert isinstance(first, MappingProxyType) and first is not second
    with pytest.raises(TypeError):
        first["contract_id"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.pop("layout"),
        lambda p: p.__setitem__("extra", True),
        lambda p: p.__setitem__("schema_version", 2),
        lambda p: p["finding_contract"].__setitem__("severity", ["fatal"]),
        lambda p: p["eligibility"].__setitem__("use_cases", ["diagnostic"]),
        lambda p: p["safety"]["size_limits_bytes"].__setitem__("json", True),
        lambda p: p["dependency_contracts"]["interface"].__setitem__("frozen_sha256", "0" * 64),
        lambda p: p["layout"].__setitem__("extra", True),
        lambda p: p["raw_policy"].__setitem__("source_symlinks_forbidden", False),
        lambda p: p["publication"].__setitem__("atomic_directory_publish", False),
        lambda p: p["safety"].__setitem__("sqlite_query_only", False),
        lambda p: p["finding_contract"].__setitem__("fields", p["finding_contract"]["fields"][:-1]),
        lambda p: p["action_contract"].__setitem__("summary_fields", p["action_contract"]["summary_fields"][:-1]),
        lambda p: p["build_identity"].__setitem__("inputs", p["build_identity"]["inputs"][:-1]),
        lambda p: p["action_contract"].__setitem__("formal_bc_candidate_requirements", p["action_contract"]["formal_bc_candidate_requirements"][:-1]),
    ],
)
def test_invalid_contract_variants_fail_closed(mutation) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)
    with pytest.raises(ValueError):
        validate_dataset_index_contract(payload, contract_path=CONTRACT)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_is_rejected(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "contract.json"
    path.write_text(CONTRACT.read_text(encoding="utf-8").replace('"schema_version": 1', f'"schema_version": {constant}', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="constant"):
        load_dataset_index_contract(path)


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(CONTRACT.read_text(encoding="utf-8").replace("{", '{"schema_name":"duplicate",', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset_index_contract(path)


def test_setup_installs_contract_document_and_cli() -> None:
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"config/contracts/dataset_index_v1.json"' in setup
    assert '"docs/dataset_index_v1.md"' in setup
    assert "team_sorting_dataset_index = team_sorting.dataset_indexer:main" in setup


def test_installed_prefix_contract_and_dependencies_load_without_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "prefix/local"
    share = prefix / "share/team_sorting/config/contracts"
    module = prefix / "lib/python3.10/site-packages/team_sorting/dataset_index_contract.py"
    share.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    module.write_text("# installed", encoding="utf-8")
    for name in (*DEPENDENCIES, "dataset_index_v1.json"):
        (share / name).write_bytes((ROOT / "config/contracts" / name).read_bytes())
    monkeypatch.setattr(contract_module, "__file__", str(module))
    monkeypatch.setattr(contract_module, "_ament_candidate", lambda: None)
    monkeypatch.chdir(tmp_path)
    assert contract_module.dataset_index_contract_path() == share / "dataset_index_v1.json"
    assert contract_module.load_dataset_index_contract()["contract_id"] == "dataset_index_v1"
