"""Fail-closed consumer for the Stage-2 adapter safe-candidate JSON contract.

This pure-Python module owns decoding, cross-repository identity checks, bounded
one-shot state, and conversion to the existing ``ManipulationCommand``.  It
does not import ROS, publish commands, advance the FSM, or create FinalAction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Real
from threading import Lock
from typing import Any, Mapping, Optional

from .interfaces import FSMStatus, JOINT_NAMES, ManipulationCommand, RobotJointState


_SCHEMA_FIELDS = frozenset(
    {
        "schema_version",
        "timestamp_ns",
        "request_id",
        "generation",
        "task_identity",
        "values",
        "controlled_mask",
        "valid",
        "valid_until_ns",
        "source",
        "mode",
        "failure_reason",
        "published_to_robot",
    }
)
_EXPECTED_MASK = tuple(index == 3 for index in range(19))
_HEAD_YAW_JOINT_INDEX = 1


def instruction_fingerprint(raw_json: str) -> str:
    """Independently reproduce the adapter's canonical JSON fingerprint."""

    if not isinstance(raw_json, str):
        raise ValueError("instruction raw value must be a string")
    try:
        parsed = json.loads(raw_json)
        normalized = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        material = normalized
    except (json.JSONDecodeError, TypeError, ValueError):
        material = raw_json
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


@dataclass(frozen=True)
class ExternalCandidateConfig:
    """Independent, default-off safety gates and bounds for the consumer."""

    enabled: bool = False
    enable_actuation: bool = False
    simulation_only: bool = True
    topic: str = "/mmk2_pi05_adapter/safe_candidate"
    schema_version: int = 1
    expected_source: str = "fixed_safe_candidate"
    expected_mode: str = "fixed_head_yaw"
    allowed_controlled_indices: tuple[int, ...] = (3,)
    candidate_ttl_ms: float = 250.0
    instruction_ttl_ms: float = 1500.0
    joint_state_ttl_ms: float = 250.0
    watchdog_timeout_ms: float = 300.0
    max_joint_delta_rad: float = 0.01
    max_joint_velocity_rad_s: float = 0.20
    nominal_control_period_s: float = 0.05
    provisional_head_yaw_lower_rad: float = -0.25
    provisional_head_yaw_upper_rad: float = 0.25
    require_unique_request_id: bool = True
    require_task_identity_match: bool = True
    require_generation_binding: bool = True
    allow_generation_binding: bool = False
    simulation_publish_enabled: bool = False
    expected_episode_id: str = ""
    request_id_cache_size: int = 128

    def __post_init__(self) -> None:
        bool_fields = (
            "enabled",
            "enable_actuation",
            "simulation_only",
            "require_unique_request_id",
            "require_task_identity_match",
            "require_generation_binding",
            "allow_generation_binding",
            "simulation_publish_enabled",
        )
        if any(type(getattr(self, name)) is not bool for name in bool_fields):
            raise ValueError("external candidate gates must be strict bool values")
        if not self.simulation_only:
            raise ValueError("external candidate simulation_only must remain true")
        if not isinstance(self.topic, str) or not self.topic.startswith("/"):
            raise ValueError("external candidate topic must be an absolute ROS topic")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("external candidate schema_version must be exactly 1")
        if self.expected_source != "fixed_safe_candidate":
            raise ValueError("expected_source must be fixed_safe_candidate")
        if self.expected_mode != "fixed_head_yaw":
            raise ValueError("expected_mode must be fixed_head_yaw")
        if tuple(self.allowed_controlled_indices) != (3,):
            raise ValueError("allowed_controlled_indices must be exactly [3]")
        object.__setattr__(self, "allowed_controlled_indices", (3,))
        for name in (
            "candidate_ttl_ms",
            "instruction_ttl_ms",
            "joint_state_ttl_ms",
            "watchdog_timeout_ms",
            "max_joint_delta_rad",
            "max_joint_velocity_rad_s",
            "nominal_control_period_s",
        ):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name))
        lower = self.provisional_head_yaw_lower_rad
        upper = self.provisional_head_yaw_upper_rad
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in (lower, upper)):
            raise ValueError("provisional head-yaw limits must be finite numbers")
        lower, upper = float(lower), float(upper)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("provisional head-yaw limits are invalid")
        object.__setattr__(self, "provisional_head_yaw_lower_rad", lower)
        object.__setattr__(self, "provisional_head_yaw_upper_rad", upper)
        if (
            isinstance(self.request_id_cache_size, bool)
            or not isinstance(self.request_id_cache_size, int)
            or self.request_id_cache_size <= 0
        ):
            raise ValueError("request_id_cache_size must be a positive integer")
        if not isinstance(self.expected_episode_id, str):
            raise ValueError("expected_episode_id must be a string")
        object.__setattr__(self, "expected_episode_id", self.expected_episode_id.strip())

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ExternalCandidateConfig":
        if not isinstance(values, Mapping):
            raise ValueError("external_candidate config must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown external_candidate config fields: {unknown}")
        normalized = dict(values)
        if "allowed_controlled_indices" in normalized:
            raw = normalized["allowed_controlled_indices"]
            if not isinstance(raw, (list, tuple)):
                raise ValueError("allowed_controlled_indices must be a list")
            normalized["allowed_controlled_indices"] = tuple(raw)
        return cls(**normalized)

    def ns(self, field: str) -> int:
        return int(float(getattr(self, field)) * 1_000_000)


@dataclass(frozen=True)
class ExternalCandidate:
    schema_version: int
    timestamp_ns: int
    request_id: str
    generation: str
    task_identity: str
    values: tuple[float, ...]
    controlled_mask: tuple[bool, ...]
    valid: bool
    valid_until_ns: int
    source: str
    mode: str
    failure_reason: str
    published_to_robot: bool


class ExternalCandidateDecoder:
    """Decode exactly the frozen 13-field adapter schema without repair."""

    def __init__(self, config: ExternalCandidateConfig) -> None:
        self.config = config

    def decode(self, raw: str) -> ExternalCandidate:
        try:
            payload = json.loads(
                raw,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non_finite_number:{value}")
                ),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"candidate_json_invalid:{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("candidate_json_not_object")
        actual = set(payload)
        if actual != _SCHEMA_FIELDS:
            raise ValueError(
                f"candidate_schema_fields_mismatch:missing={sorted(_SCHEMA_FIELDS-actual)},"
                f"unknown={sorted(actual-_SCHEMA_FIELDS)}"
            )
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported_schema_version")
        for name in ("timestamp_ns", "valid_until_ns"):
            if type(payload[name]) is not int or payload[name] < 0:
                raise ValueError(f"invalid_{name}")
        for name in ("request_id", "generation", "task_identity"):
            if not isinstance(payload[name], str) or not payload[name].strip():
                raise ValueError(f"invalid_{name}")
        if not isinstance(payload["values"], list) or len(payload["values"]) != 19:
            raise ValueError("candidate_values_must_have_19_items")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in payload["values"]):
            raise ValueError("candidate_values_must_be_real_not_bool")
        values = tuple(float(value) for value in payload["values"])
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate_values_must_be_finite")
        mask = payload["controlled_mask"]
        if not isinstance(mask, list) or len(mask) != 19:
            raise ValueError("candidate_mask_must_have_19_items")
        if any(type(value) is not bool for value in mask):
            raise ValueError("candidate_mask_must_contain_strict_bool")
        mask_tuple = tuple(mask)
        if type(payload["valid"]) is not bool or not payload["valid"]:
            raise ValueError("candidate_must_be_valid")
        if type(payload["published_to_robot"]) is not bool or payload["published_to_robot"]:
            raise ValueError("candidate_published_to_robot_must_be_false")
        if payload["source"] != self.config.expected_source:
            raise ValueError("candidate_source_mismatch")
        if payload["mode"] != self.config.expected_mode:
            raise ValueError("candidate_mode_mismatch")
        if not isinstance(payload["failure_reason"], str) or payload["failure_reason"]:
            raise ValueError("candidate_failure_reason_must_be_empty")
        if payload["valid_until_ns"] <= payload["timestamp_ns"]:
            raise ValueError("candidate_expiry_must_follow_timestamp")
        if values[0] != 0.0 or values[1] != 0.0:
            raise ValueError("candidate_base_values_must_be_zero")
        if mask_tuple != _EXPECTED_MASK:
            raise ValueError("candidate_mask_must_control_only_final_action_index_3")
        return ExternalCandidate(
            schema_version=payload["schema_version"],
            timestamp_ns=payload["timestamp_ns"],
            request_id=payload["request_id"],
            generation=payload["generation"],
            task_identity=payload["task_identity"],
            values=values,
            controlled_mask=mask_tuple,
            valid=True,
            valid_until_ns=payload["valid_until_ns"],
            source=payload["source"],
            mode=payload["mode"],
            failure_reason="",
            published_to_robot=False,
        )


@dataclass(frozen=True)
class ExternalCandidateDecision:
    event: str
    timestamp_ns: int
    accepted: bool
    failure_reason: str = ""
    candidate: Optional[ExternalCandidate] = None
    command: Optional[ManipulationCommand] = None
    instruction_received_ns: int = 0
    joint_state_timestamp_ns: int = 0
    consumer_valid_until_ns: int = 0
    q_actual: Optional[float] = None
    q_target: Optional[float] = None
    delta: Optional[float] = None
    allowed_delta: Optional[float] = None
    final_action_sequence: Optional[int] = None
    official_publish_attempted: bool = False
    official_publish_success: bool = False

    def audit_dict(self) -> dict[str, object]:
        candidate = self.candidate
        return {
            "schema_version": 1,
            "event": self.event,
            "timestamp_ns": self.timestamp_ns,
            "request_id": candidate.request_id if candidate else "",
            "generation": candidate.generation if candidate else "",
            "task_identity": candidate.task_identity if candidate else "",
            "candidate_timestamp_ns": candidate.timestamp_ns if candidate else 0,
            "candidate_valid_until_ns": candidate.valid_until_ns if candidate else 0,
            "instruction_received_ns": self.instruction_received_ns,
            "joint_state_timestamp_ns": self.joint_state_timestamp_ns,
            "accepted": self.accepted,
            "failure_reason": self.failure_reason,
            "consumer_valid_until_ns": self.consumer_valid_until_ns,
            "q_actual": self.q_actual,
            "q_target": self.q_target,
            "delta": self.delta,
            "allowed_delta": self.allowed_delta,
            "final_action_sequence": self.final_action_sequence,
            "official_publish_attempted": self.official_publish_attempted,
            "official_publish_success": self.official_publish_success,
        }


@dataclass(frozen=True)
class _PendingCandidate:
    candidate: ExternalCandidate
    received_ros_ns: int
    consumer_valid_until_ns: int


class ExternalCandidateConsumer:
    """Thread-safe bounded one-shot consumer; all mutable identity state is local."""

    def __init__(self, config: ExternalCandidateConfig) -> None:
        self.config = config
        self.decoder = ExternalCandidateDecoder(config)
        self._lock = Lock()
        self._raw_instruction = ""
        self._instruction_fingerprint = ""
        self._instruction_received_ns = 0
        self._selected_task_id: Optional[int] = None
        self._task_key: Optional[tuple[int, str]] = None
        self._current_task_identity = ""
        self._bound_generation = ""
        self._request_ids: deque[str] = deque()
        self._request_id_set: set[str] = set()
        self._pending: Optional[_PendingCandidate] = None
        self._last_candidate_received_ns = 0
        self._shutdown = False

    @property
    def current_task_identity(self) -> str:
        with self._lock:
            return self._current_task_identity

    @property
    def bound_generation(self) -> str:
        with self._lock:
            return self._bound_generation

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    @property
    def request_id_count(self) -> int:
        with self._lock:
            return len(self._request_ids)

    @property
    def instruction_received_ns(self) -> int:
        with self._lock:
            return self._instruction_received_ns

    def update_instruction(self, raw: str, selected_task_id: int, received_ros_ns: int) -> bool:
        """Refresh liveness every valid broadcast while preserving FSM dedup semantics."""

        if not isinstance(raw, str) or not raw:
            raise ValueError("instruction_raw_missing")
        if isinstance(selected_task_id, bool) or not isinstance(selected_task_id, int):
            raise ValueError("instruction_task_id_invalid")
        if isinstance(received_ros_ns, bool) or not isinstance(received_ros_ns, int) or received_ros_ns < 0:
            raise ValueError("instruction_timestamp_invalid")
        fingerprint = instruction_fingerprint(raw)
        episode = self.config.expected_episode_id
        identity = f"{episode}:{selected_task_id}:{fingerprint}" if episode else ""
        task_key = (selected_task_id, fingerprint)
        with self._lock:
            changed = task_key != self._task_key
            if changed:
                self._clear_task_state_locked()
            self._raw_instruction = raw
            self._instruction_fingerprint = fingerprint
            self._instruction_received_ns = received_ros_ns
            self._selected_task_id = selected_task_id
            self._task_key = task_key
            self._current_task_identity = identity
            return changed

    def receive(self, raw: str, received_ros_ns: int) -> ExternalCandidateDecision:
        """Decode outside the lock, then atomically authorize and reserve one pending slot."""

        if (
            isinstance(received_ros_ns, bool)
            or not isinstance(received_ros_ns, int)
            or received_ros_ns < 0
        ):
            return ExternalCandidateDecision(
                "candidate_rejected", 0, False, "candidate_receive_timestamp_invalid"
            )
        try:
            candidate = self.decoder.decode(raw)
        except ValueError as exc:
            return ExternalCandidateDecision("candidate_rejected", received_ros_ns, False, str(exc))
        with self._lock:
            failure = self._receive_failure_locked(candidate, received_ros_ns)
            if failure:
                return self._decision_locked("candidate_rejected", received_ros_ns, False, failure, candidate)
            consumer_valid_until = min(
                candidate.valid_until_ns,
                received_ros_ns + self.config.ns("candidate_ttl_ms"),
            )
            if self.config.require_generation_binding and not self._bound_generation:
                self._bound_generation = candidate.generation
            self._remember_request_id_locked(candidate.request_id)
            self._pending = _PendingCandidate(candidate, received_ros_ns, consumer_valid_until)
            self._last_candidate_received_ns = received_ros_ns
            return self._decision_locked(
                "candidate_pending", received_ros_ns, True, "", candidate, consumer_valid_until
            )

    def _receive_failure_locked(self, candidate: ExternalCandidate, now_ns: int) -> str:
        if self._shutdown:
            return "consumer_shutdown"
        if not self.config.enabled:
            return "external_candidate_disabled"
        if not self.config.enable_actuation:
            return "external_candidate_actuation_disabled"
        if not self.config.simulation_publish_enabled:
            return "simulation_publish_disabled"
        if self.config.require_task_identity_match:
            if not self.config.expected_episode_id:
                return "expected_episode_id_missing"
            if not self._current_task_identity or candidate.task_identity != self._current_task_identity:
                return "task_identity_mismatch"
        if self._instruction_received_ns <= 0:
            return "instruction_missing"
        instruction_age = now_ns - self._instruction_received_ns
        if instruction_age < 0:
            return "instruction_future_timestamp"
        if instruction_age > self.config.ns("instruction_ttl_ms"):
            return "instruction_stale"
        if candidate.timestamp_ns > now_ns:
            return "candidate_clock_mismatch"
        if now_ns >= candidate.valid_until_ns:
            return "candidate_expired"
        if self.config.require_unique_request_id and candidate.request_id in self._request_id_set:
            return "duplicate_request_id"
        if self._pending is not None:
            return "pending_candidate_exists"
        if self.config.require_generation_binding:
            if not self._bound_generation:
                if not self.config.allow_generation_binding:
                    return "generation_not_authorized"
            elif candidate.generation != self._bound_generation:
                return "generation_changed"
        return ""

    def take(
        self,
        *,
        now_ns: int,
        actual_joints: RobotJointState,
        fsm_status: FSMStatus,
        actual_dt_s: object,
        existing_command: Optional[ManipulationCommand],
    ) -> ExternalCandidateDecision:
        """Atomically remove pending, then validate live feedback and convert once."""

        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            return ExternalCandidateDecision(
                "candidate_rejected", 0, False, "control_timestamp_invalid"
            )
        with self._lock:
            pending = self._pending
            self._pending = None
            instruction_received_ns = self._instruction_received_ns
        if pending is None:
            return ExternalCandidateDecision("no_pending_candidate", now_ns, False, "no_pending_candidate")
        candidate = pending.candidate
        base = ExternalCandidateDecision(
            "candidate_consumed",
            now_ns,
            False,
            candidate=candidate,
            instruction_received_ns=instruction_received_ns,
            consumer_valid_until_ns=pending.consumer_valid_until_ns,
        )
        failure = self._take_failure(pending, now_ns, actual_joints, existing_command)
        if failure:
            return replace(base, event="candidate_rejected", failure_reason=failure)
        remaining_ttl_s = (pending.consumer_valid_until_ns - now_ns) / 1e9
        dt = self._effective_dt(actual_dt_s, remaining_ttl_s)
        allowed_delta = min(
            self.config.max_joint_delta_rad,
            self.config.max_joint_velocity_rad_s * dt,
        )
        q_actual = float(actual_joints.position[_HEAD_YAW_JOINT_INDEX])
        q_target = candidate.values[3]
        delta = q_target - q_actual
        checked = replace(
            base,
            joint_state_timestamp_ns=actual_joints.timestamp_ns,
            q_actual=q_actual,
            q_target=q_target,
            delta=delta,
            allowed_delta=allowed_delta,
        )
        if not (
            self.config.provisional_head_yaw_lower_rad
            <= q_target
            <= self.config.provisional_head_yaw_upper_rad
        ):
            return replace(checked, event="candidate_rejected", failure_reason="provisional_head_yaw_limit")
        if abs(delta) > allowed_delta:
            return replace(checked, event="candidate_rejected", failure_reason="joint_delta_exceeds_limit")
        command = ManipulationCommand(
            joint_target=tuple(candidate.values[2:]),
            controlled_mask=tuple(candidate.controlled_mask[2:]),
            local_phase=fsm_status.local_phase,
            timestamp_ns=pending.received_ros_ns,
            valid_until_ns=pending.consumer_valid_until_ns,
            valid=True,
            failure_reason="",
        )
        return replace(checked, accepted=True, command=command)

    def _take_failure(
        self,
        pending: _PendingCandidate,
        now_ns: int,
        joints: RobotJointState,
        existing_command: Optional[ManipulationCommand],
    ) -> str:
        if now_ns >= pending.consumer_valid_until_ns:
            return "candidate_expired"
        if existing_command is not None and any(existing_command.controlled_mask):
            return "manipulation_command_conflict"
        if not isinstance(joints, RobotJointState) or not joints.valid:
            return "joint_state_invalid"
        if tuple(joints.joint_names) != JOINT_NAMES or len(joints.position) != 17:
            return "joint_state_contract_invalid"
        if (
            isinstance(joints.timestamp_ns, bool)
            or not isinstance(joints.timestamp_ns, int)
            or joints.timestamp_ns <= 0
        ):
            return "joint_state_timestamp_missing"
        age = now_ns - joints.timestamp_ns
        if age < 0:
            return "joint_state_future_timestamp"
        if age > self.config.ns("joint_state_ttl_ms"):
            return "joint_state_stale"
        if not all(math.isfinite(value) for value in joints.position):
            return "joint_state_non_finite"
        return ""

    def _effective_dt(self, actual_dt_s: object, remaining_ttl_s: float) -> float:
        fallback = min(self.config.nominal_control_period_s, remaining_ttl_s)
        if (
            isinstance(actual_dt_s, bool)
            or not isinstance(actual_dt_s, Real)
            or not math.isfinite(float(actual_dt_s))
            or float(actual_dt_s) <= 0.0
            or float(actual_dt_s) > self.config.nominal_control_period_s * 2.0
        ):
            return fallback
        return min(float(actual_dt_s), self.config.nominal_control_period_s, remaining_ttl_s)

    def watchdog(self, now_ns: int) -> Optional[ExternalCandidateDecision]:
        """Clear stale state only; never creates or replays a command."""

        with self._lock:
            if self._shutdown:
                return None
            reason = ""
            if self._instruction_received_ns > 0:
                age = now_ns - self._instruction_received_ns
                if age < 0 or age > self.config.ns("instruction_ttl_ms"):
                    reason = "instruction_future_timestamp" if age < 0 else "instruction_stale"
                    self._pending = None
                    self._bound_generation = ""
            if not reason and self._pending is not None:
                if now_ns >= self._pending.consumer_valid_until_ns:
                    reason = "candidate_expired"
                    self._pending = None
                elif (
                    self._last_candidate_received_ns > 0
                    and now_ns - self._last_candidate_received_ns
                    > self.config.ns("watchdog_timeout_ms")
                ):
                    reason = "candidate_publisher_timeout"
                    self._pending = None
            if not reason:
                return None
            return self._decision_locked("watchdog_cleared", now_ns, False, reason)

    def shutdown(self, now_ns: int) -> ExternalCandidateDecision:
        with self._lock:
            self._shutdown = True
            self._pending = None
            return self._decision_locked("consumer_shutdown", now_ns, False, "consumer_shutdown")

    def _clear_task_state_locked(self) -> None:
        self._bound_generation = ""
        self._request_ids.clear()
        self._request_id_set.clear()
        self._pending = None
        self._last_candidate_received_ns = 0

    def _remember_request_id_locked(self, request_id: str) -> None:
        if len(self._request_ids) >= self.config.request_id_cache_size:
            evicted = self._request_ids.popleft()
            self._request_id_set.remove(evicted)
        self._request_ids.append(request_id)
        self._request_id_set.add(request_id)

    def _decision_locked(
        self,
        event: str,
        timestamp_ns: int,
        accepted: bool,
        failure_reason: str,
        candidate: Optional[ExternalCandidate] = None,
        consumer_valid_until_ns: int = 0,
    ) -> ExternalCandidateDecision:
        return ExternalCandidateDecision(
            event=event,
            timestamp_ns=timestamp_ns,
            accepted=accepted,
            failure_reason=failure_reason,
            candidate=candidate,
            instruction_received_ns=self._instruction_received_ns,
            consumer_valid_until_ns=consumer_valid_until_ns,
        )
