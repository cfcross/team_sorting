"""Recorder 专用的 Action/Dispatch 严格配对契约。

本模块只处理已经发布到团队内部遥测话题的事实，不导入 ROS、不调用 ActionMux，
也不把本地 publisher 调用成功提升为 controller 接受或机器人执行确认。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from threading import RLock
from typing import Any, Mapping, Optional

from .interfaces import (
    ActionDispatchRecord,
    FinalAction,
    action_dispatch_from_json,
    action_dispatch_to_json,
    final_action_from_json,
    final_action_to_json,
)


FRAME_SCHEMA_NAME = "MMK2RecordedActionFrame"
FRAME_SCHEMA_VERSION = 1
ISSUE_SCHEMA_NAME = "MMK2ActionPairingIssue"
ISSUE_SCHEMA_VERSION = 1
MAX_CONTRACT_PREVIEW_CHARS = 65_536
MAX_ISSUE_DETAIL_CHARS = 4_096
MAX_KEY_ERROR_PREVIEW_ITEMS = 8
MAX_KEY_ERROR_ITEM_CHARS = 80

ARRIVAL_ORDERS = frozenset({"final_action_first", "dispatch_first"})
ISSUE_TYPES = frozenset(
    {
        "invalid_final_action_json",
        "invalid_action_dispatch_json",
        "duplicate_identical_final_action",
        "duplicate_conflicting_final_action",
        "duplicate_identical_dispatch",
        "duplicate_conflicting_dispatch",
        "late_duplicate_final_action",
        "late_duplicate_dispatch",
        "sequence_mismatch",
        "timestamp_mismatch",
        "missing_final_action",
        "missing_action_dispatch",
        "pending_capacity_eviction",
        "pending_age_timeout",
        "shutdown_orphan",
    }
)
ISSUE_SIDES = frozenset({"final_action", "action_dispatch", "pair"})


def _strict_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label}必须是非负整数且不能是bool")
    return value


def _strict_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label}必须是正整数且不能是bool")
    return value


def _strict_json_object(raw: str, label: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError(f"{label}必须是JSON字符串")

    def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                key_preview = key[:MAX_KEY_ERROR_ITEM_CHARS]
                truncated = "true" if len(key) > len(key_preview) else "false"
                raise ValueError(
                    f"{label}包含重复字段：key_preview={key_preview!r}, "
                    f"key_truncated={truncated}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"禁止非有限JSON数值：{value}")
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} JSON无效：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON顶层必须是对象")
    return payload


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(str(value) for value in expected - actual)
        unknown = sorted(str(value) for value in actual - expected)

        def _summary(name: str, values: list[str]) -> str:
            preview = [
                value[:MAX_KEY_ERROR_ITEM_CHARS]
                + ("…" if len(value) > MAX_KEY_ERROR_ITEM_CHARS else "")
                for value in values[:MAX_KEY_ERROR_PREVIEW_ITEMS]
            ]
            return (
                f"{name}_count={len(values)}, {name}_preview={preview}, "
                f"{name}_truncated={'true' if len(values) > len(preview) else 'false'}"
            )

        raise ValueError(
            f"{label}字段不匹配：{_summary('missing', missing)}; "
            f"{_summary('unknown', unknown)}"
        )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _payload_from_serialized(raw: str, label: str) -> dict[str, Any]:
    return _strict_json_object(raw, label)


def strict_final_action_from_json(raw: str) -> FinalAction:
    """在不改变公共 FinalAction 解码器的前提下施加 Recorder 严格边界。"""

    payload = _strict_json_object(raw, "FinalAction")
    expected = {
        "schema_version",
        "sequence",
        "timestamp_ns",
        "action",
        "global_phase",
        "local_phase",
        "valid",
        "clipped",
        "failure_reason",
    }
    _require_exact_keys(payload, expected, "FinalAction")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("FinalAction schema_version必须为1")
    _strict_nonnegative_int(payload["sequence"], "FinalAction.sequence")
    _strict_nonnegative_int(payload["timestamp_ns"], "FinalAction.timestamp_ns")
    if type(payload["valid"]) is not bool or type(payload["clipped"]) is not bool:
        raise ValueError("FinalAction valid/clipped必须是严格bool")
    if not all(
        isinstance(payload[name], str)
        for name in ("global_phase", "local_phase", "failure_reason")
    ):
        raise ValueError("FinalAction枚举和原因字段必须是字符串")
    action = payload["action"]
    if not isinstance(action, list) or len(action) != 19:
        raise ValueError("FinalAction.action必须是19维数组")
    for index, value in enumerate(action):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"FinalAction.action[{index}]必须是数值且不能是bool")
        if not math.isfinite(float(value)):
            raise ValueError(f"FinalAction.action[{index}]必须是有限数")
    return final_action_from_json(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    )


def strict_action_dispatch_from_json(raw: str) -> ActionDispatchRecord:
    """拒绝重复 JSON 字段后复用公共 Dispatch V1 的全部严格不变量。"""

    payload = _strict_json_object(raw, "ActionDispatchRecord")
    _prevalidate_action_dispatch_keys(payload)
    return action_dispatch_from_json(_canonical_json(payload))


def _prevalidate_action_dispatch_keys(payload: Mapping[str, Any]) -> None:
    """先以有界摘要检查所有对象键，避免公共解析器展开超大unknown列表。"""

    _require_exact_keys(
        payload,
        {
            "schema_name", "schema_version", "sequence", "timestamp_ns",
            "final_action_sequence", "decision", "calculated", "publish_enabled",
            "publisher_created", "publish_attempted", "publisher_call_succeeded",
            "dispatch_mode", "dispatched_action", "dispatched_mask", "attempted_groups",
            "successful_groups", "failed_groups", "group_records", "controller_accepted",
            "execution_confirmed", "failure_reason",
        },
        "ActionDispatchRecord",
    )
    decision = payload.get("decision")
    if isinstance(decision, Mapping):
        _require_exact_keys(
            decision,
            {
                "schema_name", "schema_version", "sequence", "timestamp_ns",
                "final_action_sequence", "requested_mask", "commanded_mask",
                "clipped_mask", "safety_override_mask", "base_candidate_present",
                "manipulation_candidate_present", "base_disposition",
                "manipulation_disposition", "base_source", "manipulation_source",
                "global_phase", "local_phase", "valid", "failure_reason",
            },
            "ActionMuxDecision",
        )
    groups = payload.get("group_records")
    if not isinstance(groups, list):
        return
    group_keys = {
        "group", "official_topic", "message_type", "attempted", "succeeded",
        "exact_payload", "failure_reason",
    }
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            continue
        _require_exact_keys(group, group_keys, f"group_records[{index}]")
        exact = group.get("exact_payload")
        if not isinstance(exact, Mapping):
            continue
        if group.get("group") == "base":
            _require_exact_keys(exact, {"linear", "angular"}, "base.exact_payload")
            for name in ("linear", "angular"):
                vector = exact.get(name)
                if isinstance(vector, Mapping):
                    _require_exact_keys(vector, {"x", "y", "z"}, f"base.{name}")
        else:
            _require_exact_keys(exact, {"data"}, f"{group.get('group')}.exact_payload")


@dataclass(frozen=True)
class ActionPairingConfig:
    """有界异步配对参数；等待时间单位纳秒，定时器周期单位秒。"""

    enabled: bool
    max_pending_per_side: int
    max_completed_sequences: int
    max_wait_ns: int
    prune_period_sec: float
    raw_payload_preview_chars: int

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("recorder.action_pairing.enabled必须是严格bool")
        _strict_positive_int(self.max_pending_per_side, "max_pending_per_side")
        _strict_positive_int(self.max_completed_sequences, "max_completed_sequences")
        _strict_positive_int(self.max_wait_ns, "max_wait_ns")
        if (
            isinstance(self.prune_period_sec, bool)
            or not isinstance(self.prune_period_sec, (int, float))
            or not math.isfinite(float(self.prune_period_sec))
            or float(self.prune_period_sec) <= 0.0
        ):
            raise ValueError("prune_period_sec必须是正有限数且不能是bool")
        _strict_positive_int(self.raw_payload_preview_chars, "raw_payload_preview_chars")
        if self.raw_payload_preview_chars > MAX_CONTRACT_PREVIEW_CHARS:
            raise ValueError(
                f"raw_payload_preview_chars不得超过{MAX_CONTRACT_PREVIEW_CHARS}"
            )
        object.__setattr__(self, "prune_period_sec", float(self.prune_period_sec))

    @classmethod
    def from_mapping(cls, value: object) -> "ActionPairingConfig":
        if not isinstance(value, Mapping):
            raise ValueError("recorder.action_pairing必须是映射")
        expected = {
            "enabled",
            "max_pending_per_side",
            "max_completed_sequences",
            "max_wait_ns",
            "prune_period_sec",
            "raw_payload_preview_chars",
        }
        _require_exact_keys(value, expected, "recorder.action_pairing")
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True)
class RecordedActionFrame:
    """同 sequence、同生成时间的 FinalAction/Dispatch 接收配对事实。"""

    schema_name: str
    schema_version: int
    sequence: int
    timestamp_ns: int
    recorder_timestamp_ns: int
    final_action_received_monotonic_ns: int
    dispatch_received_monotonic_ns: int
    pairing_completed_monotonic_ns: int
    arrival_order: str
    final_action: FinalAction
    action_dispatch: ActionDispatchRecord
    pairing_status: str
    controller_accepted: None
    execution_confirmed: None
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_name != FRAME_SCHEMA_NAME
            or type(self.schema_version) is not int
            or self.schema_version != FRAME_SCHEMA_VERSION
        ):
            raise ValueError("RecordedActionFrame schema必须为MMK2RecordedActionFrame V1")
        for label in (
            "sequence",
            "timestamp_ns",
            "recorder_timestamp_ns",
            "final_action_received_monotonic_ns",
            "dispatch_received_monotonic_ns",
            "pairing_completed_monotonic_ns",
        ):
            _strict_nonnegative_int(getattr(self, label), f"RecordedActionFrame.{label}")
        if not isinstance(self.final_action, FinalAction):
            raise ValueError("final_action必须是FinalAction")
        if not isinstance(self.action_dispatch, ActionDispatchRecord):
            raise ValueError("action_dispatch必须是ActionDispatchRecord")
        # FinalAction 公共 V1 解码器为历史兼容会做部分类型转换；Recorder Frame 必须
        # 重新经过本模块严格边界，防止 bool 或宽松字段进入训练基础设施。
        strict_final_action_from_json(final_action_to_json(self.final_action))
        sequences = (
            self.sequence,
            self.final_action.sequence,
            self.action_dispatch.sequence,
            self.action_dispatch.final_action_sequence,
            self.action_dispatch.decision.final_action_sequence,
        )
        if len(set(sequences)) != 1:
            raise ValueError("RecordedActionFrame sequence四层关联不一致")
        timestamps = (
            self.timestamp_ns,
            self.final_action.timestamp_ns,
            self.action_dispatch.timestamp_ns,
            self.action_dispatch.decision.timestamp_ns,
        )
        if len(set(timestamps)) != 1:
            raise ValueError("RecordedActionFrame timestamp三层关联不一致")
        if self.arrival_order not in ARRIVAL_ORDERS:
            raise ValueError("arrival_order不是V1受控值")
        if self.pairing_status != "paired":
            raise ValueError("RecordedActionFrame V1 pairing_status必须为paired")
        if self.controller_accepted is not None or self.execution_confirmed is not None:
            raise ValueError("RecordedActionFrame V1不能声明接受或执行确认")
        limitations = tuple(self.limitations)
        if not limitations or any(not isinstance(item, str) or not item for item in limitations):
            raise ValueError("limitations必须是非空字符串元组")
        object.__setattr__(self, "limitations", limitations)
        if self.pairing_completed_monotonic_ns < max(
            self.final_action_received_monotonic_ns,
            self.dispatch_received_monotonic_ns,
        ):
            raise ValueError("pairing完成单调时间不得早于两侧接收时间")


@dataclass(frozen=True)
class ActionPairingIssue:
    """Recorder 配对异常的有界、机器可读事实。"""

    schema_name: str
    schema_version: int
    issue_type: str
    sequence: Optional[int]
    recorder_timestamp_ns: int
    received_monotonic_ns: int
    side: str
    detail: str
    raw_payload_preview: str
    existing_digest: Optional[str]
    incoming_digest: Optional[str]
    final_action_present: bool
    dispatch_present: bool

    def __post_init__(self) -> None:
        if (
            self.schema_name != ISSUE_SCHEMA_NAME
            or type(self.schema_version) is not int
            or self.schema_version != ISSUE_SCHEMA_VERSION
        ):
            raise ValueError("ActionPairingIssue schema必须为MMK2ActionPairingIssue V1")
        if self.issue_type not in ISSUE_TYPES:
            raise ValueError("issue_type不是V1受控值")
        if self.sequence is not None:
            _strict_nonnegative_int(self.sequence, "ActionPairingIssue.sequence")
        _strict_nonnegative_int(self.recorder_timestamp_ns, "recorder_timestamp_ns")
        _strict_nonnegative_int(self.received_monotonic_ns, "received_monotonic_ns")
        if self.side not in ISSUE_SIDES:
            raise ValueError("side不是V1受控值")
        if not isinstance(self.detail, str) or not isinstance(self.raw_payload_preview, str):
            raise ValueError("detail/raw_payload_preview必须是字符串")
        if len(self.detail) > MAX_ISSUE_DETAIL_CHARS:
            raise ValueError("detail超过契约硬上限")
        if len(self.raw_payload_preview) > MAX_CONTRACT_PREVIEW_CHARS:
            raise ValueError("raw_payload_preview超过契约硬上限")
        for label, digest in (
            ("existing_digest", self.existing_digest),
            ("incoming_digest", self.incoming_digest),
        ):
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label}必须是SHA-256十六进制串或null")
        if type(self.final_action_present) is not bool or type(self.dispatch_present) is not bool:
            raise ValueError("presence字段必须是严格bool")


def recorded_action_frame_to_json(frame: RecordedActionFrame) -> str:
    if not isinstance(frame, RecordedActionFrame):
        raise TypeError("recorded_action_frame_to_json只接受RecordedActionFrame")
    payload = {
        "schema_name": frame.schema_name,
        "schema_version": frame.schema_version,
        "sequence": frame.sequence,
        "timestamp_ns": frame.timestamp_ns,
        "recorder_timestamp_ns": frame.recorder_timestamp_ns,
        "final_action_received_monotonic_ns": frame.final_action_received_monotonic_ns,
        "dispatch_received_monotonic_ns": frame.dispatch_received_monotonic_ns,
        "pairing_completed_monotonic_ns": frame.pairing_completed_monotonic_ns,
        "arrival_order": frame.arrival_order,
        "final_action": _payload_from_serialized(
            final_action_to_json(frame.final_action), "FinalAction"
        ),
        "action_dispatch": _payload_from_serialized(
            action_dispatch_to_json(frame.action_dispatch), "ActionDispatchRecord"
        ),
        "pairing_status": frame.pairing_status,
        "controller_accepted": None,
        "execution_confirmed": None,
        "limitations": list(frame.limitations),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def recorded_action_frame_from_json(raw: str) -> RecordedActionFrame:
    payload = _strict_json_object(raw, "RecordedActionFrame")
    expected = {
        "schema_name",
        "schema_version",
        "sequence",
        "timestamp_ns",
        "recorder_timestamp_ns",
        "final_action_received_monotonic_ns",
        "dispatch_received_monotonic_ns",
        "pairing_completed_monotonic_ns",
        "arrival_order",
        "final_action",
        "action_dispatch",
        "pairing_status",
        "controller_accepted",
        "execution_confirmed",
        "limitations",
    }
    _require_exact_keys(payload, expected, "RecordedActionFrame")
    if not isinstance(payload["final_action"], dict) or not isinstance(
        payload["action_dispatch"], dict
    ):
        raise ValueError("RecordedActionFrame嵌套动作必须是对象")
    if not isinstance(payload["limitations"], list):
        raise ValueError("limitations必须是数组")
    try:
        return RecordedActionFrame(
            schema_name=payload["schema_name"],
            schema_version=payload["schema_version"],
            sequence=payload["sequence"],
            timestamp_ns=payload["timestamp_ns"],
            recorder_timestamp_ns=payload["recorder_timestamp_ns"],
            final_action_received_monotonic_ns=payload[
                "final_action_received_monotonic_ns"
            ],
            dispatch_received_monotonic_ns=payload["dispatch_received_monotonic_ns"],
            pairing_completed_monotonic_ns=payload["pairing_completed_monotonic_ns"],
            arrival_order=payload["arrival_order"],
            final_action=strict_final_action_from_json(_canonical_json(payload["final_action"])),
            action_dispatch=strict_action_dispatch_from_json(
                _canonical_json(payload["action_dispatch"])
            ),
            pairing_status=payload["pairing_status"],
            controller_accepted=payload["controller_accepted"],
            execution_confirmed=payload["execution_confirmed"],
            limitations=tuple(payload["limitations"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"RecordedActionFrame JSON无效：{exc}") from exc


def action_pairing_issue_to_json(issue: ActionPairingIssue) -> str:
    if not isinstance(issue, ActionPairingIssue):
        raise TypeError("action_pairing_issue_to_json只接受ActionPairingIssue")
    return json.dumps(
        {
            "schema_name": issue.schema_name,
            "schema_version": issue.schema_version,
            "issue_type": issue.issue_type,
            "sequence": issue.sequence,
            "recorder_timestamp_ns": issue.recorder_timestamp_ns,
            "received_monotonic_ns": issue.received_monotonic_ns,
            "side": issue.side,
            "detail": issue.detail,
            "raw_payload_preview": issue.raw_payload_preview,
            "existing_digest": issue.existing_digest,
            "incoming_digest": issue.incoming_digest,
            "final_action_present": issue.final_action_present,
            "dispatch_present": issue.dispatch_present,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def action_pairing_issue_from_json(raw: str) -> ActionPairingIssue:
    payload = _strict_json_object(raw, "ActionPairingIssue")
    expected = {
        "schema_name",
        "schema_version",
        "issue_type",
        "sequence",
        "recorder_timestamp_ns",
        "received_monotonic_ns",
        "side",
        "detail",
        "raw_payload_preview",
        "existing_digest",
        "incoming_digest",
        "final_action_present",
        "dispatch_present",
    }
    _require_exact_keys(payload, expected, "ActionPairingIssue")
    try:
        return ActionPairingIssue(**{name: payload[name] for name in expected})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ActionPairingIssue JSON无效：{exc}") from exc


@dataclass(frozen=True)
class _PendingItem:
    value: FinalAction | ActionDispatchRecord
    canonical_json: str
    raw_payload_preview: str
    digest: str
    recorder_timestamp_ns: int
    received_monotonic_ns: int
    insertion_order: int
    acknowledged: bool = False


@dataclass(frozen=True)
class PairingOutcome:
    persist_raw: bool = False
    frames: tuple[RecordedActionFrame, ...] = ()
    issues: tuple[ActionPairingIssue, ...] = ()


@dataclass(frozen=True)
class PairingPlan:
    """一次不可变的持久化计划；只有外部追加成功后才允许 ``commit``。"""

    revision: int
    operation: str
    sequence: Optional[int] = None
    side: str = "pair"
    item: Optional[_PendingItem] = None
    raw_json: Optional[str] = None
    frame: Optional[RecordedActionFrame] = None
    issue: Optional[ActionPairingIssue] = None


class ActionDispatchPairer:
    """按 sequence 配对遥测；状态只在对应 JSONL 已成功追加后提交。"""

    def __init__(self, config: ActionPairingConfig) -> None:
        if not isinstance(config, ActionPairingConfig) or not config.enabled:
            raise ValueError("ActionDispatchPairer要求enabled=true的严格配置")
        self.config = config
        self.pending_final_actions: dict[int, _PendingItem] = {}
        self.pending_dispatches: dict[int, _PendingItem] = {}
        self._completed: OrderedDict[
            int, tuple[Optional[str], Optional[str]]
        ] = OrderedDict()
        # 精确终态账本与近期digest LRU职责不同：区间永久阻止本Episode旧sequence重开，
        # LRU只为近期late duplicate提供双方digest。单调ActionMux序列通常压缩为单区间。
        self._terminal_ranges: list[tuple[int, int]] = []
        self._insertion_order = 0
        self._closed = False
        self._revision = 0
        self._lock = RLock()

    @property
    def completed_sequences(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._completed)

    @property
    def terminal_sequence_ranges(self) -> tuple[tuple[int, int], ...]:
        with self._lock:
            return tuple(self._terminal_ranges)

    @property
    def ready_sequences(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(
                sorted(
                    set(self.pending_final_actions).intersection(
                        self.pending_dispatches
                    )
                )
            )

    def invalid_payload(
        self,
        side: str,
        raw_payload: object,
        detail: str,
        recorder_timestamp_ns: int,
        received_monotonic_ns: int,
    ) -> PairingOutcome:
        with self._lock:
            self._ensure_open()
            issue_type = (
                "invalid_final_action_json"
                if side == "final_action"
                else "invalid_action_dispatch_json"
            )
            raw = raw_payload if isinstance(raw_payload, str) else repr(raw_payload)
            issue = self._issue(
                issue_type,
                None,
                recorder_timestamp_ns,
                received_monotonic_ns,
                side,
                detail,
                raw,
                None,
                self._digest(raw),
                False,
                False,
            )
            return PairingOutcome(issues=(issue,))

    def prepare_invalid_payload(
        self,
        side: str,
        raw_payload: object,
        detail: str,
        recorder_timestamp_ns: int,
        received_monotonic_ns: int,
    ) -> PairingPlan:
        """准备无效消息issue；不改变状态，失败时可原样重新准备。"""

        outcome = self.invalid_payload(
            side,
            raw_payload,
            detail,
            recorder_timestamp_ns,
            received_monotonic_ns,
        )
        return PairingPlan(
            revision=self._revision,
            operation="diagnostic",
            side=side,
            issue=outcome.issues[0],
        )

    def add_final_action(
        self,
        action: FinalAction,
        raw_payload: str,
        recorder_timestamp_ns: int,
        received_monotonic_ns: int,
    ) -> PairingOutcome:
        plan = self.prepare_final_action(
            action, raw_payload, recorder_timestamp_ns, received_monotonic_ns
        )
        return self._apply_without_storage(plan)

    def add_dispatch(
        self,
        dispatch: ActionDispatchRecord,
        raw_payload: str,
        recorder_timestamp_ns: int,
        received_monotonic_ns: int,
    ) -> PairingOutcome:
        plan = self.prepare_dispatch(
            dispatch, raw_payload, recorder_timestamp_ns, received_monotonic_ns
        )
        return self._apply_without_storage(plan)

    def prepare_final_action(
        self,
        action: FinalAction,
        raw_payload: str,
        recorder_timestamp_ns: int,
        received_monotonic_ns: int,
    ) -> PairingPlan:
        if not isinstance(action, FinalAction):
            raise ValueError("action必须是FinalAction")
        return self._prepare_add(
            "final_action",
            action,
            final_action_to_json(action),
            raw_payload,
            recorder_timestamp_ns,
            received_monotonic_ns,
        )

    def prepare_dispatch(
        self,
        dispatch: ActionDispatchRecord,
        raw_payload: str,
        recorder_timestamp_ns: int,
        received_monotonic_ns: int,
    ) -> PairingPlan:
        if not isinstance(dispatch, ActionDispatchRecord):
            raise ValueError("dispatch必须是ActionDispatchRecord")
        return self._prepare_add(
            "action_dispatch",
            dispatch,
            action_dispatch_to_json(dispatch),
            raw_payload,
            recorder_timestamp_ns,
            received_monotonic_ns,
        )

    def prune(
        self, recorder_timestamp_ns: int, current_monotonic_ns: int
    ) -> PairingOutcome:
        issues: list[ActionPairingIssue] = []
        while True:
            plan = self.prepare_prune(recorder_timestamp_ns, current_monotonic_ns)
            if plan is None:
                return PairingOutcome(issues=tuple(issues))
            if plan.issue is not None:
                issues.append(plan.issue)
            self.commit(plan)

    def close(
        self, recorder_timestamp_ns: int, current_monotonic_ns: int
    ) -> PairingOutcome:
        issues: list[ActionPairingIssue] = []
        while True:
            plan = self.prepare_shutdown(recorder_timestamp_ns, current_monotonic_ns)
            if plan.operation in {"close", "already_closed"}:
                self.commit(plan)
                return PairingOutcome(issues=tuple(issues))
            if plan.issue is not None:
                issues.append(plan.issue)
            self.commit(plan)

    def _prepare_add(
        self,
        side: str,
        value: FinalAction | ActionDispatchRecord,
        canonical_wire_json: str,
        raw_payload: str,
        recorder_timestamp_ns: int,
        received_monotonic_ns: int,
    ) -> PairingPlan:
        with self._lock:
            self._ensure_open()
            _strict_nonnegative_int(recorder_timestamp_ns, "recorder_timestamp_ns")
            _strict_nonnegative_int(received_monotonic_ns, "received_monotonic_ns")
            sequence = value.sequence
            _strict_nonnegative_int(sequence, "sequence")
            canonical = _canonical_json(
                _payload_from_serialized(canonical_wire_json, side)
            )
            digest = self._digest(canonical)
            current = (
                self.pending_final_actions
                if side == "final_action"
                else self.pending_dispatches
            )
            if self._is_terminal(sequence):
                recent = self._completed.get(sequence)
                existing = None if recent is None else recent[0 if side == "final_action" else 1]
                detail = (
                    "sequence已经完成或终止，拒绝重新打开"
                    if recent is not None
                    else "终态sequence账本命中；近期digest已淘汰，旧sequence fail closed"
                )
                return PairingPlan(
                    revision=self._revision,
                    operation="diagnostic",
                    sequence=sequence,
                    side=side,
                    issue=self._issue(
                        f"late_duplicate_{'final_action' if side == 'final_action' else 'dispatch'}",
                        sequence,
                        recorder_timestamp_ns,
                        received_monotonic_ns,
                        side,
                        detail,
                        raw_payload,
                        existing,
                        digest,
                        True,
                        True,
                    ),
                )
            existing_item = current.get(sequence)
            if existing_item is not None:
                identical = existing_item.canonical_json == canonical
                if identical and not existing_item.acknowledged:
                    return PairingPlan(
                        revision=self._revision,
                        operation="resume",
                        sequence=sequence,
                        side=side,
                        item=existing_item,
                    )
                kind = "identical" if identical else "conflicting"
                suffix = "final_action" if side == "final_action" else "dispatch"
                return PairingPlan(
                    revision=self._revision,
                    operation="diagnostic",
                    sequence=sequence,
                    side=side,
                    issue=self._issue(
                        f"duplicate_{kind}_{suffix}",
                        sequence,
                        recorder_timestamp_ns,
                        received_monotonic_ns,
                        side,
                        "重复内容与首条一致" if identical else "冲突重复未覆盖首条候选",
                        raw_payload,
                        existing_item.digest,
                        digest,
                        side == "final_action",
                        side == "action_dispatch",
                    ),
                )
            item = _PendingItem(
                value=value,
                canonical_json=canonical,
                raw_payload_preview=self._preview(raw_payload),
                digest=digest,
                recorder_timestamp_ns=recorder_timestamp_ns,
                received_monotonic_ns=received_monotonic_ns,
                insertion_order=self._insertion_order + 1,
            )
            return PairingPlan(
                revision=self._revision,
                operation="raw_insert",
                sequence=sequence,
                side=side,
                item=item,
                raw_json=canonical_wire_json,
            )

    def prepare_pair(
        self,
        sequence: int,
        recorder_timestamp_ns: int,
        completed_monotonic_ns: int,
    ) -> Optional[PairingPlan]:
        with self._lock:
            final_item = self.pending_final_actions.get(sequence)
            dispatch_item = self.pending_dispatches.get(sequence)
            if final_item is None or dispatch_item is None:
                return None
            arrival_order = (
                "final_action_first"
                if final_item.insertion_order < dispatch_item.insertion_order
                else "dispatch_first"
            )
            frame, issue = self._pair(
                sequence,
                final_item,
                dispatch_item,
                arrival_order,
                recorder_timestamp_ns,
                completed_monotonic_ns,
            )
            return PairingPlan(
                revision=self._revision,
                operation="terminal_pair",
                sequence=sequence,
                frame=frame,
                issue=issue,
            )

    def prepare_capacity(
        self, recorder_timestamp_ns: int, current_monotonic_ns: int
    ) -> Optional[PairingPlan]:
        with self._lock:
            candidates: list[tuple[str, int, _PendingItem]] = []
            for side, pending in (
                ("final_action", self.pending_final_actions),
                ("action_dispatch", self.pending_dispatches),
            ):
                if len(pending) > self.config.max_pending_per_side:
                    sequence, item = min(
                        pending.items(),
                        key=lambda pair: (
                            pair[1].received_monotonic_ns,
                            pair[1].insertion_order,
                            pair[0],
                        ),
                    )
                    candidates.append((side, sequence, item))
            if not candidates:
                return None
            side, sequence, item = min(
                candidates,
                key=lambda entry: (
                    entry[2].received_monotonic_ns,
                    entry[2].insertion_order,
                    entry[1],
                    entry[0],
                ),
            )
            return self._pending_terminal_plan(
                "pending_capacity_eviction",
                side,
                sequence,
                item,
                recorder_timestamp_ns,
                current_monotonic_ns,
            )

    def prepare_prune(
        self, recorder_timestamp_ns: int, current_monotonic_ns: int
    ) -> Optional[PairingPlan]:
        with self._lock:
            if self._closed:
                return None
            _strict_nonnegative_int(recorder_timestamp_ns, "recorder_timestamp_ns")
            _strict_nonnegative_int(current_monotonic_ns, "current_monotonic_ns")
            expired: list[tuple[str, int, _PendingItem]] = []
            for side, pending in (
                ("final_action", self.pending_final_actions),
                ("action_dispatch", self.pending_dispatches),
            ):
                expired.extend(
                    (side, sequence, item)
                    for sequence, item in pending.items()
                    if current_monotonic_ns - item.received_monotonic_ns
                    >= self.config.max_wait_ns
                )
            if not expired:
                return None
            side, sequence, item = min(
                expired,
                key=lambda entry: (
                    entry[2].received_monotonic_ns,
                    entry[2].insertion_order,
                    entry[1],
                    entry[0],
                ),
            )
            return self._pending_terminal_plan(
                "pending_age_timeout",
                side,
                sequence,
                item,
                recorder_timestamp_ns,
                current_monotonic_ns,
            )

    def prepare_shutdown(
        self, recorder_timestamp_ns: int, current_monotonic_ns: int
    ) -> PairingPlan:
        with self._lock:
            _strict_nonnegative_int(recorder_timestamp_ns, "recorder_timestamp_ns")
            _strict_nonnegative_int(current_monotonic_ns, "current_monotonic_ns")
            if self._closed:
                return PairingPlan(self._revision, "already_closed")
            items = [
                ("final_action", sequence, item)
                for sequence, item in self.pending_final_actions.items()
            ] + [
                ("action_dispatch", sequence, item)
                for sequence, item in self.pending_dispatches.items()
            ]
            if not items:
                return PairingPlan(self._revision, "close")
            side, sequence, item = min(
                items,
                key=lambda entry: (
                    entry[2].received_monotonic_ns,
                    entry[2].insertion_order,
                    entry[1],
                    entry[0],
                ),
            )
            return self._pending_terminal_plan(
                "shutdown_orphan",
                side,
                sequence,
                item,
                recorder_timestamp_ns,
                current_monotonic_ns,
            )

    def acknowledge(self, side: str, sequence: int) -> None:
        with self._lock:
            pending = self._pending_for_side(side)
            item = pending.get(sequence)
            if item is not None and not item.acknowledged:
                pending[sequence] = _PendingItem(
                    value=item.value,
                    canonical_json=item.canonical_json,
                    raw_payload_preview=item.raw_payload_preview,
                    digest=item.digest,
                    recorder_timestamp_ns=item.recorder_timestamp_ns,
                    received_monotonic_ns=item.received_monotonic_ns,
                    insertion_order=item.insertion_order,
                    acknowledged=True,
                )
                self._revision += 1

    def commit(self, plan: PairingPlan) -> None:
        """在对应JSONL追加成功后提交计划；revision不一致时拒绝陈旧计划。"""

        if not isinstance(plan, PairingPlan):
            raise ValueError("commit只接受PairingPlan")
        with self._lock:
            if plan.operation in {"diagnostic", "resume", "already_closed"}:
                return
            if plan.revision != self._revision:
                raise RuntimeError("PairingPlan已陈旧，拒绝重复或乱序commit")
            if plan.operation == "raw_insert":
                if plan.item is None or plan.sequence is None:
                    raise RuntimeError("raw_insert计划缺少item/sequence")
                self._pending_for_side(plan.side)[plan.sequence] = plan.item
                self._insertion_order = max(
                    self._insertion_order, plan.item.insertion_order
                )
            elif plan.operation == "terminal_pair":
                if plan.sequence is None:
                    raise RuntimeError("terminal_pair计划缺少sequence")
                final_item = self.pending_final_actions.pop(plan.sequence)
                dispatch_item = self.pending_dispatches.pop(plan.sequence)
                self._remember_terminal(
                    plan.sequence, final_item.digest, dispatch_item.digest
                )
            elif plan.operation == "terminal_pending":
                if plan.sequence is None or plan.item is None:
                    raise RuntimeError("terminal_pending计划缺少item/sequence")
                pending = self._pending_for_side(plan.side)
                current = pending.get(plan.sequence)
                if current != plan.item:
                    raise RuntimeError("pending终态计划目标已变化")
                del pending[plan.sequence]
                final_digest = plan.item.digest if plan.side == "final_action" else None
                dispatch_digest = plan.item.digest if plan.side == "action_dispatch" else None
                self._remember_terminal(plan.sequence, final_digest, dispatch_digest)
            elif plan.operation == "close":
                if self.pending_final_actions or self.pending_dispatches:
                    raise RuntimeError("仍有pending时拒绝关闭Pairer")
                self._closed = True
            else:
                raise RuntimeError(f"未知PairingPlan operation：{plan.operation}")
            self._revision += 1

    def _apply_without_storage(self, plan: PairingPlan) -> PairingOutcome:
        """兼容纯Pairer单测：把每个计划视为已成功持久化并立即提交。"""

        persist_raw = plan.raw_json is not None
        frames: list[RecordedActionFrame] = []
        issues: list[ActionPairingIssue] = []
        if plan.frame is not None:
            frames.append(plan.frame)
        if plan.issue is not None:
            issues.append(plan.issue)
        if plan.operation == "diagnostic":
            return PairingOutcome(False, tuple(frames), tuple(issues))
        if plan.operation == "resume":
            assert plan.sequence is not None
            sequence = plan.sequence
        else:
            self.commit(plan)
            sequence = plan.sequence
        if sequence is None:
            return PairingOutcome(persist_raw, tuple(frames), tuple(issues))
        pair = self.prepare_pair(
            sequence,
            plan.item.recorder_timestamp_ns if plan.item is not None else 0,
            plan.item.received_monotonic_ns if plan.item is not None else 0,
        )
        if pair is not None:
            if pair.frame is not None:
                frames.append(pair.frame)
            if pair.issue is not None:
                issues.append(pair.issue)
            self.commit(pair)
        while True:
            capacity = self.prepare_capacity(
                plan.item.recorder_timestamp_ns if plan.item is not None else 0,
                plan.item.received_monotonic_ns if plan.item is not None else 0,
            )
            if capacity is None:
                break
            assert capacity.issue is not None
            issues.append(capacity.issue)
            self.commit(capacity)
        self.acknowledge(plan.side, sequence)
        return PairingOutcome(persist_raw, tuple(frames), tuple(issues))

    def _pair(
        self,
        sequence: int,
        final_item: _PendingItem,
        dispatch_item: _PendingItem,
        arrival_order: str,
        recorder_timestamp_ns: int,
        completed_monotonic_ns: int,
    ) -> tuple[Optional[RecordedActionFrame], Optional[ActionPairingIssue]]:
        final_action = final_item.value
        dispatch = dispatch_item.value
        assert isinstance(final_action, FinalAction)
        assert isinstance(dispatch, ActionDispatchRecord)
        sequences = (
            sequence,
            final_action.sequence,
            dispatch.sequence,
            dispatch.final_action_sequence,
            dispatch.decision.final_action_sequence,
        )
        if len(set(sequences)) != 1:
            return None, self._issue(
                "sequence_mismatch", sequence, recorder_timestamp_ns,
                completed_monotonic_ns, "pair", "sequence四层关联不一致", "",
                final_item.digest, dispatch_item.digest, True, True
            )
        timestamps = (
            final_action.timestamp_ns,
            dispatch.timestamp_ns,
            dispatch.decision.timestamp_ns,
        )
        if len(set(timestamps)) != 1:
            return None, self._issue(
                "timestamp_mismatch", sequence, recorder_timestamp_ns,
                completed_monotonic_ns, "pair", "timestamp三层关联不一致", "",
                final_item.digest, dispatch_item.digest, True, True
            )
        return RecordedActionFrame(
            schema_name=FRAME_SCHEMA_NAME,
            schema_version=FRAME_SCHEMA_VERSION,
            sequence=sequence,
            timestamp_ns=final_action.timestamp_ns,
            recorder_timestamp_ns=recorder_timestamp_ns,
            final_action_received_monotonic_ns=final_item.received_monotonic_ns,
            dispatch_received_monotonic_ns=dispatch_item.received_monotonic_ns,
            pairing_completed_monotonic_ns=completed_monotonic_ns,
            arrival_order=arrival_order,
            final_action=final_action,
            action_dispatch=dispatch,
            pairing_status="paired",
            controller_accepted=None,
            execution_confirmed=None,
            limitations=(
                "仅确认Recorder收到同周期且一致的FinalAction与ActionDispatchRecord",
                "不确认DDS送达、Server接收、controller接受或机器人执行",
                "不声明训练可用性或Episode成功",
            ),
        ), None

    def _pending_terminal_plan(
        self,
        issue_type: str,
        side: str,
        sequence: int,
        item: _PendingItem,
        recorder_timestamp_ns: int,
        received_monotonic_ns: int,
    ) -> PairingPlan:
        return PairingPlan(
            revision=self._revision,
            operation="terminal_pending",
            sequence=sequence,
            side=side,
            item=item,
            issue=self._orphan_issue(
                issue_type,
                side,
                sequence,
                item,
                recorder_timestamp_ns,
                received_monotonic_ns,
            ),
        )

    def _pending_for_side(self, side: str) -> dict[int, _PendingItem]:
        if side == "final_action":
            return self.pending_final_actions
        if side == "action_dispatch":
            return self.pending_dispatches
        raise ValueError("pending side必须是final_action或action_dispatch")

    def _remember_terminal(
        self,
        sequence: int,
        final_digest: Optional[str],
        dispatch_digest: Optional[str],
    ) -> None:
        self._completed[sequence] = (final_digest, dispatch_digest)
        self._completed.move_to_end(sequence)
        while len(self._completed) > self.config.max_completed_sequences:
            self._completed.popitem(last=False)
        self._add_terminal_range(sequence)

    def _add_terminal_range(self, sequence: int) -> None:
        start = end = sequence
        merged: list[tuple[int, int]] = []
        inserted = False
        for current_start, current_end in self._terminal_ranges:
            if current_end + 1 < start:
                merged.append((current_start, current_end))
            elif end + 1 < current_start:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((current_start, current_end))
            else:
                start = min(start, current_start)
                end = max(end, current_end)
        if not inserted:
            merged.append((start, end))
        self._terminal_ranges = merged

    def _is_terminal(self, sequence: int) -> bool:
        for start, end in self._terminal_ranges:
            if sequence < start:
                return False
            if sequence <= end:
                return True
        return False

    def _orphan_issue(
        self,
        issue_type: str,
        side: str,
        sequence: int,
        item: _PendingItem,
        recorder_timestamp_ns: int,
        received_monotonic_ns: int,
    ) -> ActionPairingIssue:
        missing = "action_dispatch" if side == "final_action" else "final_action"
        return self._issue(
            issue_type,
            sequence,
            recorder_timestamp_ns,
            received_monotonic_ns,
            side,
            f"pending仅有{side}，缺少{missing}",
            item.raw_payload_preview,
            item.digest,
            None,
            side == "final_action",
            side == "action_dispatch",
        )

    def _issue(
        self,
        issue_type: str,
        sequence: Optional[int],
        recorder_timestamp_ns: int,
        received_monotonic_ns: int,
        side: str,
        detail: str,
        raw_payload: str,
        existing_digest: Optional[str],
        incoming_digest: Optional[str],
        final_action_present: bool,
        dispatch_present: bool,
    ) -> ActionPairingIssue:
        return ActionPairingIssue(
            schema_name=ISSUE_SCHEMA_NAME,
            schema_version=ISSUE_SCHEMA_VERSION,
            issue_type=issue_type,
            sequence=sequence,
            recorder_timestamp_ns=recorder_timestamp_ns,
            received_monotonic_ns=received_monotonic_ns,
            side=side,
            detail=self._bounded_detail(detail),
            raw_payload_preview=self._preview(raw_payload),
            existing_digest=existing_digest,
            incoming_digest=incoming_digest,
            final_action_present=final_action_present,
            dispatch_present=dispatch_present,
        )

    def _preview(self, raw: str) -> str:
        return raw[: self.config.raw_payload_preview_chars]

    @staticmethod
    def _bounded_detail(detail: object) -> str:
        text = detail if isinstance(detail, str) else repr(detail)
        if len(text) <= MAX_ISSUE_DETAIL_CHARS:
            return text
        suffix = f"…[truncated=true, original_chars={len(text)}]"
        return text[: MAX_ISSUE_DETAIL_CHARS - len(suffix)] + suffix

    @staticmethod
    def _digest(raw: str) -> str:
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ActionDispatchPairer已经关闭")
