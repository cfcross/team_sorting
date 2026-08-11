"""Fail-closed pi0.5 15x19 policy chunk consumer for official simulation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from threading import Lock
from typing import Mapping, Optional

from .competition_context import CompetitionContext, task_semantic_fingerprint
from .interfaces import BaseCommand, FSMStatus, GlobalPhase, ManipulationCommand

SCHEMA_NAME = "MMK2Pi05PolicyControlCandidate"
SCHEMA_VERSION = 1
HORIZON = 15
ACTION_DIM = 19
HANDOFF_POLICY_STEPS = 2
_GRIPPER_INDICES = frozenset({11, 18})
_FIELDS = frozenset({"schema_name", "schema_version", "request_id", "generation_id", "run_id", "episode_id", "task_id", "attempt_count", "instruction_fingerprint", "task_set_fingerprint", "active_task_fingerprint", "model_id", "action_horizon", "action_dim", "actions", "response_latency_ms", "context_valid", "valid", "failure_reason", "published_to_robot"})
_STOP_PHASES = {GlobalPhase.WAIT_READY, GlobalPhase.LOAD_TASK, GlobalPhase.DONE, GlobalPhase.SAFE_HOLD, GlobalPhase.FAILED}


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an explicit positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be an explicit positive finite number")
    return result


def _instruction_fingerprint(raw: str) -> str:
    parsed = json.loads(raw)
    normalized = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PolicyControlConfig:
    enabled: bool = False
    enable_actuation: bool = False
    simulation_only: bool = True
    simulation_publish_enabled: bool = False
    topic: str = "/mmk2_pi05_adapter/policy_control_candidate"
    expected_model_id: str = "pi05_mmk2_task1_lora"
    allowed_task_ids: tuple[int, ...] = (1,)
    action_step_period_s: Optional[float] = None
    max_policy_response_latency_ms: Optional[float] = None
    candidate_ttl_ms: Optional[float] = None
    watchdog_timeout_ms: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("enabled", "enable_actuation", "simulation_only", "simulation_publish_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        if not self.simulation_only:
            raise ValueError("pi05 policy control simulation_only=false is forbidden")
        if not isinstance(self.topic, str) or not self.topic.startswith("/"):
            raise ValueError("topic must be an absolute ROS topic")
        if not isinstance(self.expected_model_id, str) or not self.expected_model_id:
            raise ValueError("expected_model_id must be non-empty")
        tasks = tuple(self.allowed_task_ids)
        if tasks != (1,) or any(type(task_id) is not int for task_id in tasks):
            raise ValueError("M10-A allowed_task_ids must be exactly [1]")
        object.__setattr__(self, "allowed_task_ids", tasks)
        timing = ("action_step_period_s", "max_policy_response_latency_ms", "candidate_ttl_ms", "watchdog_timeout_ms")
        if self.enabled:
            for name in timing:
                if getattr(self, name) is None:
                    raise ValueError(f"enabled policy control requires explicit {name}")
        for name in timing:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(value, name))

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PolicyControlConfig":
        if not isinstance(values, Mapping):
            raise ValueError("pi05_policy_control config must be a mapping")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown pi05_policy_control fields: {sorted(unknown)}")
        normalized = dict(values)
        if "allowed_task_ids" in normalized:
            if not isinstance(normalized["allowed_task_ids"], (list, tuple)):
                raise ValueError("allowed_task_ids must be a list")
            normalized["allowed_task_ids"] = tuple(normalized["allowed_task_ids"])
        return cls(**normalized)

    def ns(self, name: str) -> int:
        value = getattr(self, name)
        if value is None:
            raise ValueError(f"{name} is not configured")
        return int(round(value * (1_000_000_000 if name.endswith("_s") else 1_000_000)))

    @property
    def publish_authorized(self) -> bool:
        return self.enabled and self.enable_actuation and self.simulation_only and self.simulation_publish_enabled


@dataclass(frozen=True)
class PolicyControlCandidate:
    request_id: int
    generation_id: str
    run_id: str
    episode_id: str
    task_id: int
    attempt_count: int
    instruction_fingerprint: str
    task_set_fingerprint: str
    active_task_fingerprint: str
    model_id: str
    actions: tuple[tuple[float, ...], ...]
    response_latency_ms: float


class PolicyControlDecoder:
    def __init__(self, config: PolicyControlConfig) -> None:
        self.config = config

    def decode(self, raw: str) -> PolicyControlCandidate:
        try:
            data = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non_finite:{token}")))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"candidate_json_invalid:{exc}") from exc
        if not isinstance(data, dict) or set(data) != _FIELDS:
            raise ValueError("candidate_schema_fields_mismatch")
        if data["schema_name"] != SCHEMA_NAME or type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported_candidate_schema")
        for name in ("request_id", "task_id", "attempt_count", "action_horizon", "action_dim"):
            if type(data[name]) is not int or data[name] < 0:
                raise ValueError(f"invalid_{name}")
        if data["action_horizon"] != HORIZON or data["action_dim"] != ACTION_DIM:
            raise ValueError("action_shape_must_be_15x19")
        for name in ("generation_id", "run_id", "episode_id", "instruction_fingerprint", "task_set_fingerprint", "active_task_fingerprint", "model_id"):
            if not isinstance(data[name], str) or not data[name]:
                raise ValueError(f"invalid_{name}")
        if data["model_id"] != self.config.expected_model_id:
            raise ValueError("model_id_mismatch")
        if data["task_id"] not in self.config.allowed_task_ids:
            raise ValueError("task_not_allowed")
        actions = data["actions"]
        if not isinstance(actions, list) or len(actions) != HORIZON or any(not isinstance(row, list) or len(row) != ACTION_DIM for row in actions):
            raise ValueError("action_shape_must_be_15x19")
        if any(isinstance(value, bool) or not isinstance(value, Real) for row in actions for value in row):
            raise ValueError("actions_must_be_real_not_bool")
        matrix = tuple(tuple(float(value) for value in row) for row in actions)
        if not all(math.isfinite(value) for row in matrix for value in row):
            raise ValueError("actions_must_be_finite")
        latency = data["response_latency_ms"]
        if isinstance(latency, bool) or not isinstance(latency, Real) or not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid_response_latency_ms")
        if data["context_valid"] is not True or data["valid"] is not True:
            raise ValueError("candidate_must_be_valid")
        if data["failure_reason"] != "" or data["published_to_robot"] is not False:
            raise ValueError("candidate_validity_fields_invalid")
        return PolicyControlCandidate(data["request_id"], data["generation_id"], data["run_id"], data["episode_id"], data["task_id"], data["attempt_count"], data["instruction_fingerprint"], data["task_set_fingerprint"], data["active_task_fingerprint"], data["model_id"], matrix, float(latency))


@dataclass(frozen=True)
class PolicyControlDecision:
    accepted: bool
    failure_reason: str = ""
    base_command: Optional[BaseCommand] = None
    manipulation_command: Optional[ManipulationCommand] = None
    request_id: Optional[int] = None
    action_index: Optional[int] = None
    generation_id: str = ""


@dataclass(frozen=True)
class _Pending:
    candidate: PolicyControlCandidate
    received_ros_ns: int


class PolicyControlConsumer:
    """Thread-safe receding-horizon consumer using only subscriber ROS time."""

    def __init__(self, config: PolicyControlConfig) -> None:
        self.config = config
        self.decoder = PolicyControlDecoder(config)
        self._lock = Lock()
        self._context_key: Optional[tuple[object, ...]] = None
        self._pending: Optional[_Pending] = None
        self._last_request_id = -1
        self._generation_id = ""
        self._last_output: Optional[tuple[float, ...]] = None
        self._last_output_identity: Optional[tuple[object, ...]] = None
        self._handoff_source: Optional[tuple[float, ...]] = None
        self._handoff_start_ros_ns: Optional[int] = None
        self._handoff_identity: Optional[tuple[object, ...]] = None
        self._shutdown = False

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    def update_context(self, context: CompetitionContext) -> bool:
        key = self._context_identity(context)
        with self._lock:
            changed = key != self._context_key
            if changed or key is None:
                self._clear_active_actions()
                self._generation_id = ""
            self._context_key = key
            return changed

    @staticmethod
    def _context_identity(context: CompetitionContext) -> Optional[tuple[object, ...]]:
        if not context.valid or context.finished or context.active_task is None:
            return None
        active = context.to_dict()["active_task"]
        raw = json.dumps(active, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (context.run_id, context.current_task_id, context.current_attempt_count, _instruction_fingerprint(raw), context.task_set_fingerprint, task_semantic_fingerprint(context.active_task))

    def receive(self, raw: str, received_ros_ns: int) -> PolicyControlDecision:
        if type(received_ros_ns) is not int or received_ros_ns < 0:
            with self._lock:
                self._clear_active_actions()
            return PolicyControlDecision(False, "invalid_received_ros_ns")
        try:
            candidate = self.decoder.decode(raw)
        except ValueError as exc:
            with self._lock:
                self._clear_active_actions()
            return PolicyControlDecision(False, str(exc))
        with self._lock:
            if self._shutdown:
                self._clear_active_actions()
                return PolicyControlDecision(False, "consumer_shutdown")
            if not self.config.enabled or not self.config.enable_actuation:
                self._clear_active_actions()
                return PolicyControlDecision(False, "policy_control_disabled")
            identity = (candidate.run_id, candidate.task_id, candidate.attempt_count, candidate.instruction_fingerprint, candidate.task_set_fingerprint, candidate.active_task_fingerprint)
            output_identity = (*identity, candidate.generation_id)
            if (
                self._context_key is None
                or identity != self._context_key
                or candidate.episode_id != self._context_key[0]
            ):
                self._clear_active_actions()
                return PolicyControlDecision(False, "competition_context_identity_mismatch")
            if candidate.request_id <= self._last_request_id:
                self._clear_active_actions()
                return PolicyControlDecision(False, "stale_or_replayed_request")
            assert self.config.max_policy_response_latency_ms is not None
            if candidate.response_latency_ms > self.config.max_policy_response_latency_ms:
                self._clear_active_actions()
                return PolicyControlDecision(False, "policy_response_latency_exceeded")
            can_handoff = (
                self._generation_id == candidate.generation_id
                and self._last_output is not None
                and self._last_output_identity == output_identity
            )
            if can_handoff:
                self._handoff_source = self._last_output
                self._handoff_start_ros_ns = received_ros_ns
                self._handoff_identity = output_identity
            else:
                self._clear_active_actions()
            self._last_request_id = candidate.request_id
            self._generation_id = candidate.generation_id
            self._pending = _Pending(candidate, received_ros_ns)
            return PolicyControlDecision(True, request_id=candidate.request_id, generation_id=candidate.generation_id)

    def take(self, *, now_ns: int, fsm_status: FSMStatus, existing_base: Optional[BaseCommand] = None, existing_manipulation: Optional[ManipulationCommand] = None) -> PolicyControlDecision:
        with self._lock:
            if type(now_ns) is not int or now_ns < 0:
                self._clear_active_actions()
                return PolicyControlDecision(False, "invalid_now_ros_ns")
            if self._shutdown:
                self._clear_active_actions()
                return PolicyControlDecision(False, "consumer_shutdown")
            pending = self._pending
            if pending is None:
                return PolicyControlDecision(False, "no_policy_chunk")
            age = now_ns - pending.received_ros_ns
            if age < 0:
                self._clear_active_actions()
                return PolicyControlDecision(False, "candidate_future_receive_time")
            if age >= self.config.ns("candidate_ttl_ms"):
                self._clear_active_actions()
                return PolicyControlDecision(False, "candidate_expired")
            if age >= self.config.ns("watchdog_timeout_ms"):
                self._clear_active_actions()
                return PolicyControlDecision(False, "policy_watchdog_timeout")
            if fsm_status.global_phase in _STOP_PHASES:
                self._clear_active_actions()
                return PolicyControlDecision(False, "fsm_stop_phase")
            if existing_base is not None or existing_manipulation is not None:
                self._clear_active_actions()
                return PolicyControlDecision(False, "exclusive_control_conflict")
            index = age // self.config.ns("action_step_period_s")
            if index >= HORIZON:
                self._clear_active_actions()
                return PolicyControlDecision(False, "policy_chunk_exhausted")
            target = pending.candidate.actions[index]
            output_identity = (*self._context_key, pending.candidate.generation_id)
            action = target
            if (
                self._handoff_source is not None
                and self._handoff_start_ros_ns is not None
                and self._handoff_identity == output_identity
            ):
                duration_ns = HANDOFF_POLICY_STEPS * self.config.ns("action_step_period_s")
                handoff_age_ns = now_ns - self._handoff_start_ros_ns
                if handoff_age_ns < 0:
                    self._clear_active_actions()
                    return PolicyControlDecision(False, "candidate_future_receive_time")
                if handoff_age_ns < duration_ns:
                    alpha = handoff_age_ns / duration_ns
                    action = tuple(
                        target[dimension]
                        if dimension in _GRIPPER_INDICES
                        else self._handoff_source[dimension]
                        + alpha * (target[dimension] - self._handoff_source[dimension])
                        for dimension in range(ACTION_DIM)
                    )
                else:
                    self._clear_handoff()
            valid_until = min(
                pending.received_ros_ns + self.config.ns("candidate_ttl_ms"),
                pending.received_ros_ns + self.config.ns("watchdog_timeout_ms"),
                pending.received_ros_ns
                + (index + 1) * self.config.ns("action_step_period_s"),
            )
            base = BaseCommand(action[0], action[1], now_ns, valid_until)
            manipulation = ManipulationCommand(action[2:19], (True,) * 17, fsm_status.local_phase, now_ns, valid_until)
            self._last_output = action
            self._last_output_identity = output_identity
            return PolicyControlDecision(True, base_command=base, manipulation_command=manipulation, request_id=pending.candidate.request_id, action_index=int(index), generation_id=pending.candidate.generation_id)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            self._clear_active_actions()

    def invalidate(self) -> None:
        """Immediately discard a chunk when an external safety prerequisite fails."""
        with self._lock:
            self._clear_active_actions()

    def _clear_handoff(self) -> None:
        self._handoff_source = None
        self._handoff_start_ros_ns = None
        self._handoff_identity = None

    def _clear_active_actions(self) -> None:
        self._pending = None
        self._last_output = None
        self._last_output_identity = None
        self._clear_handoff()
