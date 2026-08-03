"""Pure functions for B3C core findings and use-case eligibility."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


USE_CASES = ("diagnostic", "perception", "formal_bc")
ELIGIBILITY = ("eligible", "conditionally_eligible", "ineligible")
SEVERITY = ("fatal", "error", "warning", "info")
EVALUATION_STATUS = ("pass", "fail", "not_applicable", "not_evaluated")
PERCEPTION_TOPICS = frozenset(
    {
        "/head_camera/color/image_raw",
        "/head_camera/aligned_depth_to_color/image_raw",
        "/head_camera/color/camera_info",
        "/joint_states",
        "/slamware_ros_sdk_server_node/odom",
        "/team/competition_context",
    }
)
STATE_TOPICS = frozenset(
    {"/joint_states", "/slamware_ros_sdk_server_node/odom", "/team/competition_context"}
)
_IDENTITY_FATAL = frozenset(
    {
        "unexpected_symlink", "source_path_escape", "sqlite_integrity_failed",
        "segment_identity_mismatch", "segment_manifest_invalid",
        "jsonl_midstream_corruption", "run_identity_mismatch", "run_id_mixed",
    }
)


def finding(
    code: str,
    severity: str,
    artifact: str,
    relative_path: str,
    message: str,
    *,
    evidence: Mapping[str, Any] | None = None,
    blocking_use_cases: Sequence[str] = (),
    evaluation_status: str = "fail",
) -> dict[str, Any]:
    if not isinstance(code, str) or not code:
        raise ValueError("finding code is invalid")
    if not isinstance(severity, str) or severity not in SEVERITY or not isinstance(evaluation_status, str) or evaluation_status not in EVALUATION_STATUS:
        raise ValueError("finding enum is invalid")
    if any(not isinstance(use_case, str) or use_case not in USE_CASES for use_case in blocking_use_cases):
        raise ValueError("finding blocking use case is invalid")
    return {
        "code": code,
        "severity": severity,
        "evaluation_status": evaluation_status,
        "artifact": artifact,
        "relative_path": relative_path,
        "message": message,
        "evidence": dict(evidence or {}),
        "blocking_use_cases": list(blocking_use_cases),
    }


def not_evaluated(code: str, artifact: str, relative_path: str, message: str) -> dict[str, Any]:
    return finding(
        code,
        "info",
        artifact,
        relative_path,
        message,
        evaluation_status="not_evaluated",
    )


def segment_eligibility(
    findings: Iterable[Mapping[str, Any]],
    topics: Iterable[str],
    *,
    context_valid: bool,
    observe_only: bool | None,
    dispatched_action_present: bool,
    complete: bool,
) -> dict[str, str]:
    items = tuple(findings)
    codes = {str(item["code"]) for item in items if item["evaluation_status"] == "fail"}
    fatal_identity = bool(codes & _IDENTITY_FATAL)
    topic_set = set(topics)
    diagnostic = "ineligible" if fatal_identity else "eligible" if complete else "conditionally_eligible"
    if fatal_identity or not context_valid or not PERCEPTION_TOPICS <= topic_set:
        perception = "ineligible"
    elif "/tf" not in topic_set or not complete:
        perception = "conditionally_eligible"
    else:
        perception = "eligible"
    if fatal_identity or observe_only is not False or not dispatched_action_present:
        formal = "ineligible"
    elif not context_valid or not STATE_TOPICS <= topic_set or not complete:
        formal = "ineligible"
    else:
        formal = "conditionally_eligible"
    result = {"diagnostic": diagnostic, "perception": perception, "formal_bc": formal}
    # blocking_use_cases is normative: every failing finding applies its ceiling.
    for item in items:
        if item.get("evaluation_status") != "fail":
            continue
        for use_case in item.get("blocking_use_cases", ()):
            result[use_case] = "ineligible"
    return result


def aggregate_eligibility(
    segment_values: Iterable[Mapping[str, str]], *, run_complete: bool,
    findings: Iterable[Mapping[str, object]] = (),
) -> dict[str, str]:
    values = tuple(segment_values)
    rank = {"eligible": 0, "conditionally_eligible": 1, "ineligible": 2}
    result: dict[str, str] = {}
    for use_case in USE_CASES:
        selected = max((item[use_case] for item in values), key=rank.get, default="ineligible")
        if not run_complete and selected == "eligible":
            selected = "conditionally_eligible"
        if use_case == "formal_bc" and selected == "eligible":
            selected = "conditionally_eligible"
        result[use_case] = selected
    for item in findings:
        if item.get("evaluation_status") != "fail":
            continue
        for use_case in item.get("blocking_use_cases", ()):
            result[use_case] = "ineligible"
    return result
