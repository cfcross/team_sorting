from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest
import yaml
import team_sorting.dataset_indexer as indexer_module

from team_sorting.data_tf_policy_contract import load_data_tf_policy_contract
from team_sorting.dataset_indexer import DatasetIndexError, DatasetIndexer, main
from team_sorting.dataset_qc import aggregate_eligibility, finding
from team_sorting.action_mux import ActionMux
from team_sorting.interfaces import (
    BaseCommand, DispatchGroupRecord, DispatchMode, FSMStatus, GlobalPhase, LocalPhase,
    RobotJointState, TwistExactPayload, action_dispatch_to_json, final_action_to_json,
)
from team_sorting.ros_nodes import _build_action_dispatch_record


POLICY = load_data_tf_policy_contract()
TOPIC_TYPES = {
    item["name"]: item["message_type"]
    for item in (
        *POLICY["topic_policy"]["current_raw_baseline"],
        *POLICY["topic_policy"]["b3_target_topics"],
    )
}
GROUPS = (
    ("base", "/cmd_vel", "geometry_msgs/msg/Twist"),
    ("spine", "/spine_forward_position_controller/commands", "std_msgs/msg/Float64MultiArray"),
    ("head", "/head_forward_position_controller/commands", "std_msgs/msg/Float64MultiArray"),
    ("left_arm", "/left_arm_forward_position_controller/commands", "std_msgs/msg/Float64MultiArray"),
    ("right_arm", "/right_arm_forward_position_controller/commands", "std_msgs/msg/Float64MultiArray"),
)


def _valid_action_records(*, publish_succeeded: bool = True) -> tuple[str, str]:
    timestamp = 10_000
    joints = RobotJointState((0.0,) * 17, (0.0,) * 17, (0.0,) * 17, timestamp)
    status = FSMStatus(1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", timestamp)
    action, decision = ActionMux().compose_with_decision(
        BaseCommand(0.1, 0.2, timestamp, timestamp + 100), None, joints, status, timestamp
    )
    action = replace(action, sequence=1, timestamp_ns=timestamp)
    decision = replace(decision, sequence=1, final_action_sequence=1, timestamp_ns=timestamp)
    base = DispatchGroupRecord(
        "base", "/cmd_vel", "geometry_msgs/msg/Twist", True, publish_succeeded,
        TwistExactPayload((action.values[0], 0.0, 0.0), (0.0, 0.0, action.values[1])),
        "" if publish_succeeded else "publisher_failed",
    )
    groups = (base, *(DispatchGroupRecord(group, topic, kind, False, None, None) for group, topic, kind in GROUPS[1:]))
    dispatch = _build_action_dispatch_record(
        action, decision, publish_enabled=True, publisher_created=True,
        dispatch_mode=DispatchMode.FULL, group_records=groups,
        failure_reason="" if publish_succeeded else "publisher_failed",
    )
    return final_action_to_json(action), action_dispatch_to_json(dispatch)


def _valid_no_attempt_records() -> tuple[str, str]:
    timestamp = 10_000
    joints = RobotJointState((0.0,) * 17, (0.0,) * 17, (0.0,) * 17, timestamp)
    status = FSMStatus(1, GlobalPhase.SEARCH_TARGET, LocalPhase.IDLE, 0, False, "", timestamp)
    action, decision = ActionMux().compose_with_decision(None, None, joints, status, timestamp)
    action = replace(action, sequence=1, timestamp_ns=timestamp)
    decision = replace(decision, sequence=1, final_action_sequence=1, timestamp_ns=timestamp)
    groups = tuple(DispatchGroupRecord(group, topic, kind, False, None, None) for group, topic, kind in GROUPS)
    dispatch = _build_action_dispatch_record(
        action, decision, publish_enabled=False, publisher_created=False,
        dispatch_mode=DispatchMode.NONE, group_records=groups, failure_reason="observe_only",
    )
    return final_action_to_json(action), action_dispatch_to_json(dispatch)


def _write_actions(segment: Path, *, publish_succeeded: bool = True, append_invalid: bool = False) -> None:
    final_raw, dispatch_raw = _valid_action_records(publish_succeeded=publish_succeeded)
    suffix = "{}\n" if append_invalid else ""
    (segment / "final_actions.jsonl").write_text(final_raw + "\n" + suffix, encoding="utf-8")
    (segment / "action_dispatches.jsonl").write_text(dispatch_raw + "\n" + suffix, encoding="utf-8")


def _context(run_id: str = "run-a", fingerprint: str = "tasks-a") -> dict:
    return {
        "schema_name": "team_sorting.competition_context", "schema_version": 1,
        "run_id": run_id, "task_set_fingerprint": fingerprint,
        "current_task_id": None, "current_attempt_count": 0,
        "elapsed_sim_s": 1.0, "score": 0, "best_scores": [0, 0, 0],
        "current_step": "-", "finished": True, "active_task": None,
        "instruction_timestamp_ns": 1, "referee_timestamp_ns": 2,
        "valid": True, "failure_reason": "",
    }


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_snapshot(root: Path) -> dict[str, tuple[str, int, int]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "derived" not in path.relative_to(root).parts:
            result[path.relative_to(root).as_posix()] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
    return result


def _make_bag(segment: Path, *, include_tf: bool = True, metadata: bool = True, mismatch: bool = False, regression: bool = False, corrupt: bool = False) -> None:
    bag = segment / "rosbag"
    bag.mkdir()
    db = bag / "rosbag_0.db3"
    if corrupt:
        db.write_bytes(b"not sqlite")
        if metadata:
            (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n", encoding="utf-8")
        return
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, offered_qos_profiles TEXT)")
    connection.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
    counts: dict[str, int] = {}
    message_id = 1
    for topic_id, (name, message_type) in enumerate(TOPIC_TYPES.items(), 1):
        if name == "/tf" and not include_tf:
            continue
        connection.execute("INSERT INTO topics VALUES(?,?,?,?,?)", (topic_id, name, message_type, "cdr", "profile"))
        count = 0 if name == "/tf_static" else 2
        counts[name] = count
        for offset in range(count):
            timestamp = 200 if regression and name == "/joint_states" and offset == 0 else 100 + offset
            if regression and name == "/joint_states" and offset == 1:
                timestamp = 100
            connection.execute("INSERT INTO messages VALUES(?,?,?,?)", (message_id, topic_id, timestamp, b"x"))
            message_id += 1
    connection.commit()
    connection.close()
    if metadata:
        listed = [
            {"topic_metadata": {"name": name, "type": TOPIC_TYPES[name], "serialization_format": "cdr", "offered_qos_profiles": "profile"}, "message_count": count + (1 if mismatch and name == "/joint_states" else 0)}
            for name, count in counts.items()
        ]
        payload = {"rosbag2_bagfile_information": {"storage_identifier": "sqlite3", "duration": {"nanoseconds": 2}, "topics_with_message_count": listed}}
        (bag / "metadata.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def _segment(root: Path, relative: str, segment_id: str, kind: str, parent: str | None, **bag_options) -> Path:
    path = root / relative
    path.mkdir(parents=True)
    manifest = {
        "schema_name": "team_sorting.recorder.segment", "schema_version": 1,
        "recorder_segment_id": segment_id, "parent_run_id": parent, "segment_kind": kind,
        "segment_sequence": 0, "process_start_wall_utc": "2026-08-03T00:00:00Z",
        "process_end_wall_utc": "2026-08-03T00:01:00Z", "node_start_ros_ns": 1,
        "node_end_ros_ns": 300, "first_ros_timestamp_ns": 100, "last_ros_timestamp_ns": 200,
        "pid": 1, "container_identity": {"value": None, "status": "unavailable", "reason": "test", "source": "test"},
        "clean_shutdown": True, "shutdown_reason": "node_shutdown", "bag_path": "rosbag",
        "bag_storage_identifier": {"value": "sqlite3", "status": "available", "reason": None, "source": "test"},
        "bag_exit_code": 0, "jsonl_artifacts": [], "message_counters": {}, "dropped_counters": {},
        "pairing_counters": {}, "warning_counters": {}, "observed_task_ids": [],
        "observed_settled_attempt_counts": [], "context_valid_count": 1, "context_invalid_count": 0,
        "marker_state": "complete",
    }
    (path / "segment.json").write_text(json.dumps(manifest), encoding="utf-8")
    marker = {"schema_name": "team_sorting.recorder.marker", "schema_version": 1, "marker": "COMPLETE", "recorder_segment_id": segment_id, "parent_run_id": parent, "completed_wall_utc": "2026-08-03T00:01:00Z"}
    (path / "COMPLETE").write_text(json.dumps(marker) + "\n", encoding="utf-8")
    context = _context(parent or "run-a")
    (path / "competition_contexts.jsonl").write_text(json.dumps(context) + "\n", encoding="utf-8")
    _make_bag(path, **bag_options)
    return path


def _dataset(tmp_path: Path, *, observe_only: bool = True, ended: bool = False, **bag_options) -> Path:
    root = tmp_path / "dataset"
    root.mkdir(parents=True)
    _segment(root, "bootstrap/bootstrap-1", "bootstrap-1", "bootstrap", None, **bag_options)
    _segment(root, "runs/run-a/segments/run-segment-1", "run-segment-1", "run_bound", "run-a", **bag_options)
    run_dir = root / "runs/run-a"
    manifest = {
        "schema_name": "team_sorting.recorder.run_manifest", "schema_version": 1,
        "recorder_schema_sha256": "e7965c34a38c11d551d9943d8d614c05bc8e28e186432ad5ff4d0eed243225cf",
        "interface_schema_name": "team_sorting.interface", "interface_schema_version": 1,
        "interface_schema_sha256": "a86548b2a43581af70b8d585d523a06bb97a8d96e1fe52950097b12d061fdaea",
        "run_id": "run-a", "task_set_fingerprint": "tasks-a", "observe_only": observe_only,
        "run_identity_scope": "team_local", "project_commit": {"value": None, "status": "unavailable", "reason": "test", "source": "test"},
        "project_branch": {"value": None, "status": "unavailable", "reason": "test", "source": "test"},
        "dirty_worktree": {"value": None, "status": "unavailable", "reason": "test", "source": "test"},
        "config_sha256": {"value": None, "status": "unavailable", "reason": "test", "source": "test"},
        "official_server_image_id": {"value": None, "status": "unavailable", "reason": "test", "source": "test"},
        "official_client_image_id": {"value": None, "status": "unavailable", "reason": "test", "source": "test"},
        "ros_domain_id": {"value": None, "status": "unavailable", "reason": "test", "source": "test"},
        "rmw_implementation": {"value": None, "status": "unavailable", "reason": "test", "source": "test"},
        "official_publish_enabled": not observe_only, "start_ros_ns": 1, "start_wall_utc": "2026-08-03T00:00:00Z",
        "recorder_segment_ids": ["run-segment-1"], "end_ros_ns": 300 if ended else None,
        "end_wall_utc": "2026-08-03T00:00:00Z" if ended else None,
        "clean_shutdown": True if ended else None, "shutdown_reason": "node_shutdown" if ended else None,
        "recovery_required": False if ended else None, "provenance_warnings": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "recovery").mkdir()
    return root


def _build_files(result) -> tuple[dict, list[dict], dict, dict]:
    assert result.output_directory is not None
    build = _json(result.output_directory / "index_build.json")
    lines = [json.loads(line) for line in (result.output_directory / "dataset_index.jsonl").read_text().splitlines()]
    return build, lines, _json(result.output_directory / "segment_qc.json"), _json(result.output_directory / "run_qc.json")


def _codes(qc: dict) -> set[str]:
    return {finding["code"] for segment in qc["segments"] for finding in segment["findings"]}


def test_build_creates_only_four_derived_outputs_and_preserves_raw(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    before = _raw_snapshot(root)
    result = DatasetIndexer(root).build()
    after = _raw_snapshot(root)
    assert before == after
    assert result.raw_immutability_verified and result.segment_count == 2 and result.run_count == 1
    assert {path.name for path in result.output_directory.iterdir()} == {"index_build.json", "dataset_index.jsonl", "segment_qc.json", "run_qc.json"}
    assert not list(root.glob("**/sample_index.jsonl"))
    assert not list(root.glob("**/training_manifest.jsonl"))


def test_index_uses_sqlite_counts_types_timestamps_and_zero_tf_static(tmp_path: Path) -> None:
    result = DatasetIndexer(_dataset(tmp_path, ended=True)).build()
    _, lines, qc, _ = _build_files(result)
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    topics = {item["name"]: item for item in run["topics"]}
    assert topics["/tf"]["type"] == "tf2_msgs/msg/TFMessage" and topics["/tf"]["count"] == 2
    assert topics["/tf_static"]["count"] == 0
    assert topics["/joint_states"]["first_timestamp_ns"] == 100
    assert "tf_static_required_edge_missing" in _codes(qc)
    deferred = [f for s in qc["segments"] for f in s["findings"] if f["code"] == "tf_static_required_edge_missing"]
    assert all(item["evaluation_status"] == "not_evaluated" for item in deferred)


def test_same_input_reuses_build_and_derived_is_excluded_from_identity(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    first = DatasetIndexer(root).build()
    second = DatasetIndexer(root).build()
    assert second.build_id == first.build_id and second.reused is True
    (root / "derived/unrelated.txt").write_text("ignored", encoding="utf-8")
    third = DatasetIndexer(root).build()
    assert third.build_id == first.build_id and third.reused is True


def test_tampered_existing_build_fails_closed_without_overwrite(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    first = DatasetIndexer(root).build()
    target = first.output_directory / "dataset_index.jsonl"
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(DatasetIndexError, match="拒绝覆盖"):
        DatasetIndexer(root).build()
    assert target.read_text(encoding="utf-8") == "tampered\n"


@pytest.mark.parametrize("extra", ["sample_index.jsonl", "training_manifest.jsonl", "extra.txt"])
def test_existing_build_with_extra_file_is_never_reused(tmp_path: Path, extra: str) -> None:
    root = _dataset(tmp_path, ended=True)
    first = DatasetIndexer(root).build()
    added = first.output_directory / extra
    added.write_text("extra\n", encoding="utf-8")
    with pytest.raises(DatasetIndexError, match="文件集合"):
        DatasetIndexer(root).build()
    assert added.read_text(encoding="utf-8") == "extra\n"


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_existing_build_with_non_regular_entry_is_never_reused(tmp_path: Path, kind: str) -> None:
    root = _dataset(tmp_path, ended=True)
    first = DatasetIndexer(root).build()
    added = first.output_directory / "unexpected"
    if kind == "directory":
        added.mkdir()
    else:
        added.symlink_to(first.output_directory / "dataset_index.jsonl")
    with pytest.raises(DatasetIndexError, match="非普通文件"):
        DatasetIndexer(root).build()
    assert os.path.lexists(added)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_artifact_count", 999), ("source_artifacts", []),
        ("source_root_fingerprint", "0" * 64), ("dependency_contracts", {}),
        ("qc_config_sha256", "0" * 64), ("indexer_version", "tampered"),
        ("implementation_identity", "tampered"),
    ],
)
def test_existing_build_provenance_tampering_is_rejected(tmp_path: Path, field: str, value) -> None:
    root = _dataset(tmp_path, ended=True)
    first = DatasetIndexer(root).build()
    manifest_path = first.output_directory / "index_build.json"
    manifest = _json(manifest_path); manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetIndexError, match="provenance"):
        DatasetIndexer(root).build()
    assert _json(manifest_path)[field] == value


@pytest.mark.parametrize("value", ["", 1])
def test_existing_build_invalid_generated_time_is_rejected(tmp_path: Path, value) -> None:
    root = _dataset(tmp_path, ended=True)
    first = DatasetIndexer(root).build()
    manifest_path = first.output_directory / "index_build.json"
    manifest = _json(manifest_path); manifest["generated_at_utc"] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetIndexError, match="generated_at_utc"):
        DatasetIndexer(root).build()


def test_existing_build_output_size_tampering_is_rejected(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    first = DatasetIndexer(root).build()
    manifest_path = first.output_directory / "index_build.json"
    manifest = _json(manifest_path); manifest["outputs"]["dataset_index.jsonl"]["size"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetIndexError, match="provenance"):
        DatasetIndexer(root).build()


def test_check_only_writes_nothing(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    result = DatasetIndexer(root).build(check_only=True)
    assert result.output_directory is None
    assert not (root / "derived").exists()


def test_qc_config_changes_build_id_and_enumeration_order_does_not(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _dataset(tmp_path)
    normal = DatasetIndexer(root).build(check_only=True)
    indexer = DatasetIndexer(root)
    original = indexer._segments
    monkeypatch.setattr(indexer, "_segments", lambda: list(reversed(original())))
    reversed_result = indexer.build(check_only=True)
    changed = DatasetIndexer(root, {"required_topics": ["/tf"], "required_static_edges": []}).build(check_only=True)
    assert reversed_result.build_id == normal.build_id
    assert changed.build_id != normal.build_id


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda root, seg: (seg / "ACTIVE").write_text("{}"), "marker_invalid"),
        (lambda root, seg: (seg / "COMPLETE").unlink(), "complete_marker_missing"),
        (lambda root, seg: _mutate_json(seg / "segment.json", "bag_exit_code", 9), "rosbag_exit_nonzero"),
        (lambda root, seg: (seg / "rosbag/metadata.yaml").unlink(), "rosbag_metadata_missing"),
        (lambda root, seg: (seg / "rosbag/metadata.yaml").write_bytes(b""), "rosbag_metadata_invalid"),
        (lambda root, seg: (seg / "competition_contexts.jsonl").unlink(), "competition_context_missing"),
        (lambda root, seg: (seg / "rosbag/rosbag_0.db3").write_bytes(b"broken"), "sqlite_integrity_failed"),
    ],
)
def test_segment_faults_are_reported(tmp_path: Path, mutation, expected: str) -> None:
    root = _dataset(tmp_path, ended=True)
    segment = root / "runs/run-a/segments/run-segment-1"
    mutation(root, segment)
    result = DatasetIndexer(root).build()
    assert expected in _codes(_build_files(result)[2])


def _mutate_json(path: Path, key: str, value) -> None:
    payload = _json(path); payload[key] = value; path.write_text(json.dumps(payload), encoding="utf-8")


def _remove_topic(segment: Path, topic: str) -> None:
    db = segment / "rosbag/rosbag_0.db3"
    connection = sqlite3.connect(db)
    row = connection.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()
    if row:
        connection.execute("DELETE FROM messages WHERE topic_id=?", (row[0],))
        connection.execute("DELETE FROM topics WHERE id=?", (row[0],))
    connection.commit(); connection.close()
    metadata = yaml.safe_load((segment / "rosbag/metadata.yaml").read_text())
    info = metadata["rosbag2_bagfile_information"]
    info["topics_with_message_count"] = [item for item in info["topics_with_message_count"] if item["topic_metadata"]["name"] != topic]
    (segment / "rosbag/metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")


@pytest.mark.parametrize("key", ["schema_name", "schema_version", "pid"])
def test_segment_manifest_missing_frozen_field_blocks_all_uses(tmp_path: Path, key: str) -> None:
    root = _dataset(tmp_path, ended=True)
    path = root / "runs/run-a/segments/run-segment-1/segment.json"
    payload = _json(path); payload.pop(key); path.write_text(json.dumps(payload), encoding="utf-8")
    _, lines, qc, _ = _build_files(DatasetIndexer(root).build())
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    assert "segment_manifest_invalid" in _codes(qc)
    assert set(run["eligibility"].values()) == {"ineligible"}


def test_segment_bag_exit_bool_is_schema_invalid(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    _mutate_json(root / "runs/run-a/segments/run-segment-1/segment.json", "bag_exit_code", False)
    assert "segment_manifest_invalid" in _codes(_build_files(DatasetIndexer(root).build())[2])


def test_run_observe_only_string_is_invalid_and_blocks_all_uses(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    _mutate_json(root / "runs/run-a/manifest.json", "observe_only", "false")
    run = _build_files(DatasetIndexer(root).build())[3]["runs"][0]
    assert "run_manifest_invalid" in {item["code"] for item in run["findings"]}
    assert set(run["eligibility"].values()) == {"ineligible"}


@pytest.mark.parametrize("ids,code", [([], "run_manifest_unlisted_segment"), (["ghost"], "run_manifest_segment_missing"), ("run-segment-1", "run_manifest_invalid")])
def test_run_segment_list_integrity(tmp_path: Path, ids, code: str) -> None:
    root = _dataset(tmp_path, ended=True)
    _mutate_json(root / "runs/run-a/manifest.json", "recorder_segment_ids", ids)
    run = _build_files(DatasetIndexer(root).build())[3]["runs"][0]
    assert code in {item["code"] for item in run["findings"]}


def test_invalid_complete_marker_is_not_complete(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    segment = root / "runs/run-a/segments/run-segment-1"
    (segment / "COMPLETE").write_text("{}\n", encoding="utf-8")
    _, lines, qc, _ = _build_files(DatasetIndexer(root).build())
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    assert run["marker_state"]["complete"] is False and "marker_invalid" in _codes(qc)


def test_complete_marker_identity_mismatch(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    path = root / "runs/run-a/segments/run-segment-1/COMPLETE"
    _mutate_json(path, "recorder_segment_id", "wrong")
    assert "marker_identity_mismatch" in _codes(_build_files(DatasetIndexer(root).build())[2])


def test_manifest_bag_path_is_authoritative(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    _mutate_json(root / "runs/run-a/segments/run-segment-1/segment.json", "bag_path", "other")
    _, lines, qc, _ = _build_files(DatasetIndexer(root).build())
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    assert run["topics"] == [] and "bag_path_invalid" in _codes(qc)


@pytest.mark.parametrize("raw", ["{}\n", "rosbag2_bagfile_information:\n  storage_identifier: sqlite3\n  storage_identifier: mcap\n", "rosbag2_bagfile_information: {}\n"])
def test_metadata_semantic_invalidity_blocks_perception(tmp_path: Path, raw: str) -> None:
    root = _dataset(tmp_path, ended=True)
    (root / "runs/run-a/segments/run-segment-1/rosbag/metadata.yaml").write_text(raw, encoding="utf-8")
    _, lines, qc, _ = _build_files(DatasetIndexer(root).build())
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    assert "rosbag_metadata_invalid" in _codes(qc)
    assert run["eligibility"]["perception"] == "ineligible"


@pytest.mark.parametrize("topic", ["/team/final_action", "/referee/score", "/material/instruction", "/team/object_estimates"])
def test_non_perception_topic_missing_does_not_block_perception(tmp_path: Path, topic: str) -> None:
    root = _dataset(tmp_path, ended=True)
    _remove_topic(root / "runs/run-a/segments/run-segment-1", topic)
    run = next(item for item in _build_files(DatasetIndexer(root).build())[1] if item["segment_kind"] == "run_bound")
    assert run["eligibility"]["perception"] == "eligible"


@pytest.mark.parametrize("config", [{"required_topics": "/tf", "required_static_edges": []}, {"required_topics": [1], "required_static_edges": []}, {"required_topics": ["/tf"], "required_static_edges": "x"}, {"required_topics": [["bad"]], "required_static_edges": []}])
def test_invalid_qc_config_is_explicitly_rejected(tmp_path: Path, config: dict) -> None:
    with pytest.raises(DatasetIndexError, match="QC|required"):
        DatasetIndexer(_dataset(tmp_path), config)


def test_metadata_count_mismatch_and_timestamp_regression(tmp_path: Path) -> None:
    root = _dataset(tmp_path, mismatch=True, regression=True)
    result = DatasetIndexer(root).build()
    codes = _codes(_build_files(result)[2])
    assert {"metadata_sqlite_count_mismatch", "timestamp_regression"} <= codes


def test_missing_tf_is_found_but_perception_is_conditional_when_other_inputs_exist(tmp_path: Path) -> None:
    root = _dataset(tmp_path, include_tf=False)
    _, lines, qc, _ = _build_files(DatasetIndexer(root).build())
    assert "tf_dynamic_missing" in _codes(qc)
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    assert run["eligibility"]["perception"] == "conditionally_eligible"


def test_missing_metadata_blocks_perception_and_formal_bc(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    (root / "runs/run-a/segments/run-segment-1/rosbag/metadata.yaml").unlink()
    _, lines, qc, _ = _build_files(DatasetIndexer(root).build())
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    assert "rosbag_metadata_missing" in _codes(qc)
    assert run["eligibility"]["perception"] == "ineligible"
    assert run["eligibility"]["formal_bc"] == "ineligible"


def test_missing_run_manifest_blocks_every_run_use_case(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    (root / "runs/run-a/manifest.json").unlink()
    _, _, _, run_qc = _build_files(DatasetIndexer(root).build())
    run = run_qc["runs"][0]
    assert "run_manifest_missing" in {item["code"] for item in run["findings"]}
    assert run["eligibility"] == {"diagnostic": "ineligible", "perception": "ineligible", "formal_bc": "ineligible"}


def test_empty_object_action_records_are_invalid(tmp_path: Path) -> None:
    root = _dataset(tmp_path, observe_only=False, ended=True)
    segment = root / "runs/run-a/segments/run-segment-1"
    (segment / "final_actions.jsonl").write_text("{}\n", encoding="utf-8")
    (segment / "action_dispatches.jsonl").write_text("{}\n", encoding="utf-8")
    _, lines, qc, _ = _build_files(DatasetIndexer(root).build())
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    summary = run["action_summary"]
    assert summary["selected_action_present"] is False
    assert summary["action_dispatch_valid_record_count"] == 0
    assert summary["action_dispatch_invalid_record_count"] == 1
    assert summary["exact_dispatched_action_present"] is False
    assert {"selected_action_invalid", "action_dispatch_invalid"} <= _codes(qc)
    assert run["eligibility"]["formal_bc"] == "ineligible"


def test_mixed_valid_and_invalid_actions_keep_invalid_findings(tmp_path: Path) -> None:
    root = _dataset(tmp_path, observe_only=False, ended=True)
    _write_actions(root / "runs/run-a/segments/run-segment-1", append_invalid=True)
    _, lines, qc, _ = _build_files(DatasetIndexer(root).build())
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    summary = run["action_summary"]
    assert summary["selected_action_valid_record_count"] == summary["selected_action_invalid_record_count"] == 1
    assert summary["action_dispatch_valid_record_count"] == summary["action_dispatch_invalid_record_count"] == 1
    assert {"selected_action_invalid", "action_dispatch_invalid"} <= _codes(qc)


def test_publisher_failure_is_not_exact_dispatch(tmp_path: Path) -> None:
    root = _dataset(tmp_path, observe_only=False, ended=True)
    _write_actions(root / "runs/run-a/segments/run-segment-1", publish_succeeded=False)
    _, lines, _, _ = _build_files(DatasetIndexer(root).build())
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    assert run["action_summary"]["exact_dispatched_action_present"] is False
    assert run["eligibility"]["formal_bc"] == "ineligible"


def test_strict_observe_only_no_attempt_record_is_not_dispatch_candidate(tmp_path: Path) -> None:
    root = _dataset(tmp_path, observe_only=True, ended=True)
    segment = root / "runs/run-a/segments/run-segment-1"
    final_raw, dispatch_raw = _valid_no_attempt_records()
    (segment / "final_actions.jsonl").write_text(final_raw + "\n", encoding="utf-8")
    (segment / "action_dispatches.jsonl").write_text(dispatch_raw + "\n", encoding="utf-8")
    _, lines, _, _ = _build_files(DatasetIndexer(root).build())
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    assert run["action_summary"]["action_dispatch_record_present"] is True
    assert run["action_summary"]["action_dispatch_valid_record_count"] == 1
    assert run["action_summary"]["exact_dispatched_action_present"] is False
    assert run["eligibility"]["formal_bc"] == "ineligible"


def test_post_publish_raw_change_removes_only_new_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _dataset(tmp_path, ended=True)
    original = indexer_module._snapshot_raw
    calls = 0
    def changed_last(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = original(*args, **kwargs)
        return value + [{"relative_path": "changed", "sha256": "0", "size": 0, "mtime_ns": 0}] if calls == 4 else value
    monkeypatch.setattr(indexer_module, "_snapshot_raw", changed_last)
    with pytest.raises(DatasetIndexError, match="derived发布期间"):
        DatasetIndexer(root).build()
    output = root / "derived/indexer_v1"
    assert output.is_dir() and not [path for path in output.iterdir() if not path.name.startswith(".index-build-")]


@pytest.mark.parametrize("failure", [DatasetIndexError("snapshot failed"), OSError("snapshot failed")])
def test_post_publish_snapshot_exception_removes_new_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    root = _dataset(tmp_path, ended=True)
    original = indexer_module._snapshot_raw
    calls = 0
    def fail_last(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise failure
        return original(*args, **kwargs)
    monkeypatch.setattr(indexer_module, "_snapshot_raw", fail_last)
    with pytest.raises(type(failure), match="snapshot failed"):
        DatasetIndexer(root).build()
    output = root / "derived/indexer_v1"
    assert output.is_dir() and not list(output.iterdir())


def test_parent_fsync_failure_preserves_primary_error_and_removes_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _dataset(tmp_path, ended=True)
    original = indexer_module.os.fsync
    calls = 0
    def fail_parent(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 6:
            raise OSError("parent fsync failed")
        original(fd)
    monkeypatch.setattr(indexer_module.os, "fsync", fail_parent)
    with pytest.raises(OSError, match="parent fsync failed"):
        DatasetIndexer(root).build()
    assert not list((root / "derived/indexer_v1").iterdir())


def test_pre_rename_write_failure_removes_temporary_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _dataset(tmp_path, ended=True)
    original = Path.open
    def fail_payload(path: Path, mode: str = "r", *args, **kwargs):
        if path.name == "dataset_index.jsonl" and mode == "xb":
            raise OSError("payload write failed")
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail_payload)
    with pytest.raises(OSError, match="payload write failed"):
        DatasetIndexer(root).build()
    output = root / "derived/indexer_v1"
    assert output.is_dir() and not list(output.iterdir())


def test_cleanup_failure_reports_primary_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _dataset(tmp_path, ended=True)
    original = indexer_module._snapshot_raw
    calls = 0
    def fail_last(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise DatasetIndexError("primary snapshot failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(indexer_module, "_snapshot_raw", fail_last)
    monkeypatch.setattr(DatasetIndexer, "_remove_published", lambda *args: (_ for _ in ()).throw(DatasetIndexError("cleanup failure")))
    with pytest.raises(DatasetIndexError, match="primary snapshot failure.*cleanup failure"):
        DatasetIndexer(root).build()


def test_reuse_failure_never_removes_existing_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _dataset(tmp_path, ended=True)
    first = DatasetIndexer(root).build()
    before = _raw_snapshot(first.output_directory)
    original = indexer_module._snapshot_raw
    calls = 0
    def changed_on_reuse(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = original(*args, **kwargs)
        return value + [{"relative_path": "changed", "sha256": "0", "size": 0, "mtime_ns": 0}] if calls == 3 else value
    monkeypatch.setattr(indexer_module, "_snapshot_raw", changed_on_reuse)
    with pytest.raises(DatasetIndexError, match="复用校验"):
        DatasetIndexer(root).build()
    assert first.output_directory.is_dir()
    assert _raw_snapshot(first.output_directory) == before


def test_context_run_id_mixing_is_fatal(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    path = root / "runs/run-a/segments/run-segment-1/competition_contexts.jsonl"
    other = _context("run-b")
    path.write_text(path.read_text() + json.dumps(other) + "\n", encoding="utf-8")
    result = DatasetIndexer(root).build()
    assert {"run_id_mixed", "segment_context_identity_mismatch"} <= _codes(_build_files(result)[2])


def test_json_duplicate_and_jsonl_midstream_corruption_are_not_ignored(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    segment = root / "runs/run-a/segments/run-segment-1"
    (segment / "segment.json").write_text('{"recorder_segment_id":"x","recorder_segment_id":"y"}', encoding="utf-8")
    context = segment / "competition_contexts.jsonl"
    context.write_text('{}\nnot-json\n{}\n', encoding="utf-8")
    codes = _codes(_build_files(DatasetIndexer(root).build())[2])
    assert {"segment_manifest_invalid", "jsonl_midstream_corruption"} <= codes


def test_action_jsonl_midstream_corruption_stays_fail_closed(tmp_path: Path) -> None:
    root = _dataset(tmp_path, observe_only=False, ended=True)
    segment = root / "runs/run-a/segments/run-segment-1"
    final_raw, dispatch_raw = _valid_action_records()
    (segment / "final_actions.jsonl").write_text(final_raw + "\n", encoding="utf-8")
    (segment / "action_dispatches.jsonl").write_text(dispatch_raw + "\nnot-json\n{}\n", encoding="utf-8")
    _, lines, qc, _ = _build_files(DatasetIndexer(root).build())
    run = next(item for item in lines if item["segment_kind"] == "run_bound")
    assert "jsonl_midstream_corruption" in _codes(qc)
    assert run["action_summary"]["action_dispatch_valid_record_count"] == 0
    assert run["eligibility"]["formal_bc"] == "ineligible"


def test_recovery_and_incomplete_run_limit_eligibility(tmp_path: Path) -> None:
    root = _dataset(tmp_path, observe_only=True, ended=False)
    report = {"schema_name": "team_sorting.recorder.recovery_report", "source_parent_run_id": "run-a", "issue_types": ["active_marker_remaining"]}
    (root / "recovery/recovery-a.json").write_text(json.dumps(report), encoding="utf-8")
    _, _, _, run_qc = _build_files(DatasetIndexer(root).build())
    run = run_qc["runs"][0]
    codes = {item["code"] for item in run["findings"]}
    assert {"recovery_finding_present", "run_end_incomplete"} <= codes
    assert run["eligibility"]["formal_bc"] == "ineligible"
    assert run["incomplete_run_reasons"]


def test_manifest_recovery_required_blocks_training_uses(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    _mutate_json(root / "runs/run-a/manifest.json", "recovery_required", True)
    run = _build_files(DatasetIndexer(root).build())[3]["runs"][0]
    assert "recovery_required_by_manifest" in {item["code"] for item in run["findings"]}
    assert run["eligibility"]["perception"] == run["eligibility"]["formal_bc"] == "ineligible"
    assert run["incomplete_run_reasons"]


@pytest.mark.parametrize("raw", ["{bad", '{"source_parent_run_id":"run-a","source_parent_run_id":"run-b"}'])
def test_broken_recovery_report_fails_build(tmp_path: Path, raw: str) -> None:
    root = _dataset(tmp_path, ended=True)
    (root / "recovery/bad.json").write_text(raw, encoding="utf-8")
    with pytest.raises(DatasetIndexError):
        DatasetIndexer(root).build()
    assert not (root / "derived").exists()


def test_recovery_report_symlink_fails_build(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    target = root / "report-target.json"
    target.write_text(json.dumps({"source_parent_run_id": "run-a"}), encoding="utf-8")
    (root / "recovery/link.json").symlink_to(target)
    with pytest.raises(DatasetIndexError, match="符号链接"):
        DatasetIndexer(root).build()


@pytest.mark.parametrize("source", ["../escape", "/absolute/path", "runs/run-a/segments/missing"])
def test_unrelated_or_unsafe_recovery_segment_fails_build(tmp_path: Path, source: str) -> None:
    root = _dataset(tmp_path, ended=True)
    report = {"source_segment_path": source, "source_parent_run_id": "run-a"}
    (root / "recovery/orphan.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(DatasetIndexError, match="不存在或不安全"):
        DatasetIndexer(root).build()


def test_valid_recovery_segment_reference_is_associated(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    report = {"source_segment_path": "runs/run-a/segments/run-segment-1", "source_parent_run_id": "run-a"}
    (root / "recovery/valid.json").write_text(json.dumps(report), encoding="utf-8")
    _, _, _, run_qc = _build_files(DatasetIndexer(root).build())
    assert "recovery_finding_present" in {item["code"] for item in run_qc["runs"][0]["findings"]}


def test_recovery_unknown_parent_run_is_not_silently_ignored(tmp_path: Path) -> None:
    root = _dataset(tmp_path, ended=True)
    (root / "recovery/orphan.json").write_text(json.dumps({"source_parent_run_id": "missing-run"}), encoding="utf-8")
    with pytest.raises(DatasetIndexError, match="不存在的Run"):
        DatasetIndexer(root).build()


@pytest.mark.parametrize(
    ("code", "blockers"),
    [
        ("run_manifest_invalid", ("diagnostic", "perception", "formal_bc")),
        ("run_identity_mismatch", ("diagnostic", "perception", "formal_bc")),
        ("task_set_fingerprint_mismatch", ("diagnostic", "perception", "formal_bc")),
        ("run_id_mixed", ("diagnostic", "perception", "formal_bc")),
        ("recovery_finding_present", ("perception", "formal_bc")),
    ],
)
def test_run_findings_enforce_declared_blockers(code: str, blockers: tuple[str, ...]) -> None:
    item = finding(code, "fatal", "run", "runs/run-a", code, blocking_use_cases=blockers)
    result = aggregate_eligibility(
        [{"diagnostic": "eligible", "perception": "eligible", "formal_bc": "eligible"}],
        run_complete=True, findings=[item],
    )
    for use_case in blockers:
        assert result[use_case] == "ineligible"
    assert result["formal_bc"] != "eligible"


def test_run_end_incomplete_applies_conditional_ceiling_without_hard_block() -> None:
    result = aggregate_eligibility(
        [{"diagnostic": "eligible", "perception": "eligible", "formal_bc": "eligible"}],
        run_complete=False,
        findings=[finding("run_end_incomplete", "warning", "run_manifest", "runs/run-a", "incomplete")],
    )
    assert result == {"diagnostic": "conditionally_eligible", "perception": "conditionally_eligible", "formal_bc": "conditionally_eligible"}


def test_non_observe_only_with_dispatch_never_becomes_formal_eligible(tmp_path: Path) -> None:
    root = _dataset(tmp_path, observe_only=False, ended=True)
    _write_actions(root / "runs/run-a/segments/run-segment-1")
    _, lines, _, run_qc = _build_files(DatasetIndexer(root).build())
    run_segment = next(item for item in lines if item["segment_kind"] == "run_bound")
    assert run_segment["action_summary"]["exact_dispatched_action_present"] is True
    assert run_segment["action_summary"]["exact_payload_record_count"] == 1
    assert run_segment["eligibility"]["formal_bc"] == "conditionally_eligible"
    assert run_qc["runs"][0]["eligibility"]["formal_bc"] != "eligible"


def test_source_symlink_dataset_symlink_and_output_escape_are_rejected(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    target = root / "target.json"; target.write_text("{}", encoding="utf-8")
    (root / "bootstrap/link.json").symlink_to(target)
    with pytest.raises(DatasetIndexError, match="symlink"):
        DatasetIndexer(root).build(check_only=True)
    linked_root = tmp_path / "linked"; linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(DatasetIndexError, match="symlink"):
        DatasetIndexer(linked_root)
    with pytest.raises(DatasetIndexError, match="output_root"):
        DatasetIndexer(root, output_root=tmp_path / "escape")


def test_cli_json_summary_build_reuse_check_only_and_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _dataset(tmp_path)
    assert main(["--dataset-root", str(root), "--json-summary"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "built"
    assert main(["--dataset-root", str(root), "--json-summary"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "reused"
    other = _dataset(tmp_path / "other")
    assert main(["--dataset-root", str(other), "--check-only", "--json-summary"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "checked"
    assert main(["--dataset-root", str(tmp_path / "missing"), "--json-summary"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "indexer_failed"


def test_cli_normalizes_expected_oserror_to_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = _dataset(tmp_path)
    monkeypatch.setattr(DatasetIndexer, "build", lambda self, **kwargs: (_ for _ in ()).throw(OSError("expected io failure")))
    assert main(["--dataset-root", str(root), "--json-summary"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "indexer_failed" and "expected io failure" in payload["error"]
