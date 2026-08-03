"""Official-offline competition lifecycle contracts.

This module is intentionally ROS-free.  The official server continues to own
the physical scene and referee state; these types only combine public topics
into a versioned team-internal snapshot.  Resetting a local task FSM never
means that the official server or its MuJoCo scene was reset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

from .interfaces import TaskSpec


SCHEMA_NAME = "team_sorting.competition_context"
SCHEMA_VERSION = 1
OFFICIAL_TASK_COUNT = 3
OFFICIAL_TIME_LIMIT_S = 600.0
OFFICIAL_MAX_SETTLED_ATTEMPTS = 3
OFFICIAL_STEPS = frozenset({"-", "nav", "touch", "lift", "place"})

_TASKINFO_RE = re.compile(r"\A任务([1-9][0-9]*): (.+)\Z")
_GAMEINFO_RE = re.compile(
    r"\At=(\d+(?:\.\d+)?)s score=(0|[1-9]\d*) "
    r"task=([1-9]\d*)/([1-9]\d*) best=(\[[^\r\n]*\]) "
    r"attempt=(\d+) step=(-|nav|touch|lift|place)\Z"
)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}必须是大于等于{minimum}的整数且不能是bool")
    return value


def _timestamp(value: Any, name: str) -> int:
    return _integer(value, name)


def _finite_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}必须是有限数且不能是bool")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name}必须是大于等于{minimum}的有限数")
    return result


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name}必须是非空字符串")
    return value


def _task_dict(task: TaskSpec) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "target_kind": task.target_kind,
        "target_body": task.target_body,
        "target_color": task.target_color,
        "place_type": task.place_type,
        "place_world_xyz": list(task.place_world_xyz or ()),
        "place_radius": task.place_radius,
        "ref_prop": task.ref_prop,
        "ref_prop_body": task.ref_prop_body,
        "direction": task.direction,
    }


def task_semantic_fingerprint(task: TaskSpec) -> str:
    """Return a stable fingerprint that excludes the receive timestamp."""

    if not isinstance(task, TaskSpec) or not task.valid:
        raise ValueError("fingerprint只接受有效TaskSpec")
    raw = json.dumps(
        _task_dict(task), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def task_set_fingerprint(tasks: Sequence[TaskSpec]) -> str:
    """Fingerprint the complete ordered task-set business semantics."""

    normalized = validate_official_task_set(tasks)
    raw = json.dumps(
        [_task_dict(task) for task in normalized], ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_official_task_set(tasks: Sequence[TaskSpec]) -> tuple[TaskSpec, ...]:
    if isinstance(tasks, (str, bytes)):
        raise ValueError("任务集合必须是TaskSpec序列")
    values = tuple(tasks)
    if len(values) != OFFICIAL_TASK_COUNT:
        raise ValueError("官方offline任务集合必须完整包含三条任务")
    if not all(isinstance(task, TaskSpec) and task.valid for task in values):
        raise ValueError("官方offline任务集合只能包含有效TaskSpec")
    if tuple(task.task_id for task in values) != (1, 2, 3):
        raise ValueError("官方offline任务编号和顺序必须严格为1、2、3")
    return values


@dataclass(frozen=True)
class RefereeProgress:
    current_task_id: Optional[int]
    current_attempt_count: int
    elapsed_sim_s: float
    score: int
    best_scores: tuple[int, ...]
    current_step: str
    finished: bool
    task_instruction: str
    taskinfo_timestamp_ns: int
    gameinfo_timestamp_ns: int
    score_timestamp_ns: int
    valid: bool
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if self.current_task_id is not None:
            _integer(self.current_task_id, "current_task_id", minimum=1)
        _integer(self.current_attempt_count, "current_attempt_count")
        _finite_number(self.elapsed_sim_s, "elapsed_sim_s")
        _integer(self.score, "score")
        if (
            not isinstance(self.best_scores, tuple)
            or len(self.best_scores) != OFFICIAL_TASK_COUNT
            or any(type(value) is not int or value < 0 for value in self.best_scores)
        ):
            raise ValueError("best_scores必须是三个非负整数")
        if self.current_step not in OFFICIAL_STEPS:
            raise ValueError("current_step不是官方值")
        if type(self.finished) is not bool or type(self.valid) is not bool:
            raise ValueError("finished/valid必须是bool")
        for name in ("taskinfo_timestamp_ns", "gameinfo_timestamp_ns", "score_timestamp_ns"):
            _timestamp(getattr(self, name), name)
        if self.valid == bool(self.failure_reason):
            raise ValueError("valid与failure_reason不一致")


@dataclass(frozen=True)
class CompetitionContext:
    schema_name: str
    schema_version: int
    run_id: str
    task_set_fingerprint: str
    current_task_id: Optional[int]
    current_attempt_count: int
    elapsed_sim_s: float
    score: int
    best_scores: tuple[int, ...]
    current_step: str
    finished: bool
    active_task: Optional[TaskSpec]
    instruction_timestamp_ns: int
    referee_timestamp_ns: int
    valid: bool
    failure_reason: str

    def __post_init__(self) -> None:
        if (
            self.schema_name != SCHEMA_NAME
            or type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError("CompetitionContext schema不受支持")
        _text(self.run_id, "run_id")
        _text(self.task_set_fingerprint, "task_set_fingerprint", allow_empty=not self.valid)
        if self.current_task_id is not None:
            _integer(self.current_task_id, "current_task_id", minimum=1)
        _integer(self.current_attempt_count, "current_attempt_count")
        _finite_number(self.elapsed_sim_s, "elapsed_sim_s")
        _integer(self.score, "score")
        if (
            not isinstance(self.best_scores, tuple)
            or len(self.best_scores) != OFFICIAL_TASK_COUNT
            or any(type(value) is not int or value < 0 for value in self.best_scores)
        ):
            raise ValueError("best_scores必须是三个非负整数")
        if self.current_step not in OFFICIAL_STEPS:
            raise ValueError("current_step不是官方值")
        if type(self.finished) is not bool or type(self.valid) is not bool:
            raise ValueError("finished/valid必须是bool")
        _timestamp(self.instruction_timestamp_ns, "instruction_timestamp_ns")
        _timestamp(self.referee_timestamp_ns, "referee_timestamp_ns")
        if self.active_task is not None and not isinstance(self.active_task, TaskSpec):
            raise ValueError("active_task必须是TaskSpec或None")
        if self.valid:
            if self.failure_reason:
                raise ValueError("有效context不得携带failure_reason")
            if not self.finished and (
                self.active_task is None
                or self.current_task_id != self.active_task.task_id
                or not self.active_task.valid
            ):
                raise ValueError("未结束的有效context必须绑定匹配的active_task")
        elif not self.failure_reason.strip():
            raise ValueError("无效context必须提供failure_reason")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["best_scores"] = list(self.best_scores)
        data["active_task"] = None if self.active_task is None else _task_dict(self.active_task)
        return data

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_json(cls, raw: str) -> "CompetitionContext":
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("CompetitionContext不是合法JSON") from exc
        if not isinstance(data, Mapping):
            raise ValueError("CompetitionContext顶层必须是对象")
        expected = tuple(cls.__dataclass_fields__)
        if set(data) != set(expected):
            raise ValueError("CompetitionContext字段集合不匹配")
        active = data["active_task"]
        task = None
        if active is not None:
            if not isinstance(active, Mapping) or set(active) != set(_task_dict(_dummy_task())):
                raise ValueError("active_task字段集合不匹配")
            task = TaskSpec(
                task_id=active["task_id"], instruction=active["instruction"],
                target_kind=active["target_kind"], target_body=active["target_body"],
                target_color=active["target_color"], place_type=active["place_type"],
                place_world_xyz=tuple(active["place_world_xyz"]), place_frame_id="world",
                place_radius=active["place_radius"], ref_prop=active["ref_prop"],
                ref_prop_body=active["ref_prop_body"], direction=active["direction"],
                timestamp_ns=data["instruction_timestamp_ns"],
            )
        return cls(
            schema_name=data["schema_name"], schema_version=data["schema_version"],
            run_id=data["run_id"], task_set_fingerprint=data["task_set_fingerprint"],
            current_task_id=data["current_task_id"],
            current_attempt_count=data["current_attempt_count"],
            elapsed_sim_s=data["elapsed_sim_s"], score=data["score"],
            best_scores=tuple(data["best_scores"]), current_step=data["current_step"],
            finished=data["finished"], active_task=task,
            instruction_timestamp_ns=data["instruction_timestamp_ns"],
            referee_timestamp_ns=data["referee_timestamp_ns"], valid=data["valid"],
            failure_reason=data["failure_reason"],
        )


def _dummy_task() -> TaskSpec:
    return TaskSpec(1, "x", "x", "x", "x", "shelf_point", (0.0, 0.0, 0.0), "world", 1.0)


class RefereeProgressParser:
    """Strictly combine the three public referee topics without guessing."""

    def __init__(self) -> None:
        self._taskinfo: Optional[tuple[Optional[int], str, bool, int]] = None
        self._gameinfo: Optional[tuple[float, int, int, int, tuple[int, ...], int, str, int]] = None
        self._score: Optional[tuple[int, int]] = None
        self._failures: dict[str, str] = {}

    def update_taskinfo(self, raw: str, timestamp_ns: int) -> bool:
        timestamp_ns = _timestamp(timestamp_ns, "taskinfo timestamp")
        if self._taskinfo is not None and timestamp_ns < self._taskinfo[3]:
            return False
        if raw == "全部任务结束":
            parsed = (None, "", True, timestamp_ns)
        else:
            match = _TASKINFO_RE.fullmatch(raw) if isinstance(raw, str) else None
            if match is None:
                self._failures["taskinfo"] = "malformed_taskinfo"
                return False
            parsed = (int(match.group(1)), match.group(2), False, timestamp_ns)
            if self._taskinfo is not None:
                previous_id = self._taskinfo[0]
                if previous_id is None or (previous_id is not None and parsed[0] < previous_id):
                    return False
        self._taskinfo = parsed
        self._failures.pop("taskinfo", None)
        return True

    def update_gameinfo(self, raw: str, timestamp_ns: int) -> bool:
        timestamp_ns = _timestamp(timestamp_ns, "gameinfo timestamp")
        if self._gameinfo is not None and timestamp_ns < self._gameinfo[7]:
            return False
        match = _GAMEINFO_RE.fullmatch(raw) if isinstance(raw, str) else None
        if match is None:
            self._failures["gameinfo"] = "malformed_gameinfo"
            return False
        elapsed = float(match.group(1))
        if not math.isfinite(elapsed):
            self._failures["gameinfo"] = "invalid_elapsed_sim_s"
            return False
        score, task_id, total = map(int, match.group(2, 3, 4))
        try:
            best_raw = json.loads(match.group(5))
        except json.JSONDecodeError:
            self._failures["gameinfo"] = "malformed_best_scores"
            return False
        if (
            not isinstance(best_raw, list) or len(best_raw) != OFFICIAL_TASK_COUNT
            or any(type(value) is not int or value < 0 for value in best_raw)
        ):
            self._failures["gameinfo"] = "invalid_best_scores"
            return False
        attempt = int(match.group(6))
        step = match.group(7)
        if total != OFFICIAL_TASK_COUNT or task_id > total or attempt > OFFICIAL_MAX_SETTLED_ATTEMPTS:
            self._failures["gameinfo"] = "gameinfo_out_of_range"
            return False
        parsed = (elapsed, score, task_id, total, tuple(best_raw), attempt, step, timestamp_ns)
        if self._gameinfo is not None:
            old_elapsed, old_score = self._gameinfo[0], self._gameinfo[1]
            old_task, old_best, old_attempt = (
                self._gameinfo[2], self._gameinfo[4], self._gameinfo[5]
            )
            if (
                elapsed < old_elapsed
                or score < old_score
                or any(new < old for new, old in zip(best_raw, old_best))
                or task_id < old_task
                or (task_id == old_task and attempt < old_attempt)
            ):
                return False
        self._gameinfo = parsed
        self._failures.pop("gameinfo", None)
        return True

    def update_score(self, score: int, timestamp_ns: int) -> bool:
        timestamp_ns = _timestamp(timestamp_ns, "score timestamp")
        try:
            score = _integer(score, "referee score")
        except ValueError:
            self._failures["score"] = "malformed_referee_score"
            return False
        if self._score is not None and timestamp_ns < self._score[1]:
            return False
        if self._score is not None and score < self._score[0]:
            return False
        self._score = (score, timestamp_ns)
        self._failures.pop("score", None)
        return True

    def progress(self) -> RefereeProgress:
        if self._failures:
            return self._invalid(";".join(self._failures[key] for key in sorted(self._failures)))
        missing = [
            name for name, value in (
                ("taskinfo", self._taskinfo), ("gameinfo", self._gameinfo), ("score", self._score)
            ) if value is None
        ]
        if missing:
            return self._invalid("missing_referee_topics:" + ",".join(missing))
        assert self._taskinfo is not None and self._gameinfo is not None and self._score is not None
        task_id, instruction, task_finished, task_ts = self._taskinfo
        elapsed, game_score, game_task, _total, best, attempt, step, game_ts = self._gameinfo
        public_score, score_ts = self._score
        if public_score != game_score:
            return self._invalid("referee_score_mismatch")
        timed_out = elapsed >= OFFICIAL_TIME_LIMIT_S
        if not task_finished and task_id != game_task:
            return self._invalid("taskinfo_gameinfo_task_mismatch")
        return RefereeProgress(
            current_task_id=None if task_finished else task_id,
            current_attempt_count=attempt, elapsed_sim_s=elapsed, score=game_score,
            best_scores=best, current_step=step,
            finished=task_finished or timed_out, task_instruction=instruction,
            taskinfo_timestamp_ns=task_ts, gameinfo_timestamp_ns=game_ts,
            score_timestamp_ns=score_ts, valid=True,
        )

    def _invalid(self, reason: str) -> RefereeProgress:
        task_ts = self._taskinfo[3] if self._taskinfo else 0
        game_ts = self._gameinfo[7] if self._gameinfo else 0
        score_ts = self._score[1] if self._score else 0
        return RefereeProgress(
            current_task_id=None, current_attempt_count=0, elapsed_sim_s=0.0,
            score=0, best_scores=(0, 0, 0), current_step="-", finished=False,
            task_instruction="", taskinfo_timestamp_ns=task_ts,
            gameinfo_timestamp_ns=game_ts, score_timestamp_ns=score_ts,
            valid=False, failure_reason=reason,
        )


class CompetitionRunCoordinator:
    """Select only the task identified by the official public referee."""

    def __init__(self) -> None:
        self._tasks: tuple[TaskSpec, ...] = ()
        self._task_set_fingerprint = ""
        self._run_id = uuid4().hex
        self._instruction_timestamp_ns = 0
        self.referee = RefereeProgressParser()
        self._last_active_key: Optional[tuple[str, int, int]] = None

    @property
    def tasks(self) -> tuple[TaskSpec, ...]:
        return self._tasks

    def update_tasks(self, tasks: Sequence[TaskSpec], timestamp_ns: int) -> bool:
        timestamp_ns = _timestamp(timestamp_ns, "instruction timestamp")
        normalized = validate_official_task_set(tasks)
        fingerprint = task_set_fingerprint(normalized)
        changed = fingerprint != self._task_set_fingerprint
        self._tasks = normalized
        self._instruction_timestamp_ns = max(self._instruction_timestamp_ns, timestamp_ns)
        if changed:
            self._task_set_fingerprint = fingerprint
            self._run_id = uuid4().hex
            self._last_active_key = None
            # A new official task set denotes a new local run identity.  Old
            # public referee samples cannot be carried across that boundary.
            self.referee = RefereeProgressParser()
        return changed

    def context(self) -> CompetitionContext:
        progress = self.referee.progress()
        referee_ts = max(
            progress.taskinfo_timestamp_ns,
            progress.gameinfo_timestamp_ns,
            progress.score_timestamp_ns,
        )
        failure = ""
        active: Optional[TaskSpec] = None
        if not self._tasks:
            failure = "missing_task_set"
        elif not progress.valid:
            failure = progress.failure_reason
        elif not progress.finished:
            active = next(
                (task for task in self._tasks if task.task_id == progress.current_task_id), None
            )
            if active is None:
                failure = "referee_task_not_in_task_set"
            elif active.instruction != progress.task_instruction:
                failure = "taskinfo_instruction_mismatch"
        valid = not failure
        return CompetitionContext(
            schema_name=SCHEMA_NAME, schema_version=SCHEMA_VERSION,
            run_id=self._run_id, task_set_fingerprint=self._task_set_fingerprint,
            current_task_id=progress.current_task_id,
            current_attempt_count=progress.current_attempt_count,
            elapsed_sim_s=progress.elapsed_sim_s, score=progress.score,
            best_scores=progress.best_scores, current_step=progress.current_step,
            finished=progress.finished, active_task=active,
            instruction_timestamp_ns=self._instruction_timestamp_ns,
            referee_timestamp_ns=referee_ts, valid=valid,
            failure_reason=failure,
        )

    def active_task_changed(self, context: CompetitionContext) -> bool:
        """Track local activation only; this is never a server/scene reset."""

        key = (
            None
            if context.active_task is None
            else (
                context.run_id,
                context.active_task.task_id,
                context.current_attempt_count,
            )
        )
        changed = key != self._last_active_key
        self._last_active_key = key
        return changed
