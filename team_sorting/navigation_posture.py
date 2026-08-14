"""导航前 INITIAL_ZERO_POSE 的纯反馈判定。

本模块只验证已经取得的 ``RobotJointState``，不生成关节命令、底盘命令、FSM事件或
发布授权。所有符号保持私有；ROS组装层只消费判定事实来门控导航，不在此模块回零。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real

from .interfaces import JOINT_NAMES, RobotJointState


_EXPECTED_JOINT_NAMES = (
    "slide_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_eef_gripper_joint",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_eef_gripper_joint",
)
_HEAD_INDICES = (1, 2)
_ARM_INDICES = tuple(range(3, 9)) + tuple(range(10, 16))
_GRIPPER_INDICES = (9, 16)


def _strict_positive_float(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name}必须是内建int或float，不能使用bool或其他数值类型")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}必须能表示为有限binary64数值") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name}必须是严格正有限数")
    return result


def _strict_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}必须是严格正内建int，不能使用bool")
    return value


def _strict_feedback_vector(values: object, name: str) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != 17:
        raise ValueError(f"{name}必须是严格17项tuple")
    result: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name}[{index}]必须是真实数且不能是bool")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}[{index}]无法表示为有限binary64数值") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}]不能包含NaN或Inf")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True)
class _NavigationPostureConfig:
    """3.5-D1私有显式配置；不提供隐藏默认值。"""

    mode: str
    target: tuple[float, ...]
    joint_state_max_age_ns: int
    max_feedback_gap_ns: int
    settled_required_cycles: int
    slide_tolerance_m: float
    head_tolerance_rad: float
    arm_tolerance_rad: float
    gripper_tolerance: float
    slide_velocity_tolerance_mps: float
    angular_velocity_tolerance_radps: float
    gripper_velocity_tolerance_per_s: float

    def __post_init__(self) -> None:
        if self.mode != "initial_zero_pose" or type(self.mode) is not str:
            raise ValueError("navigation_posture.mode必须精确为initial_zero_pose")
        if JOINT_NAMES != _EXPECTED_JOINT_NAMES:
            raise ValueError("navigation_posture.target顺序守卫与JOINT_NAMES不一致")
        if type(self.target) not in (tuple, list) or len(self.target) != 17:
            raise ValueError("navigation_posture.target必须是严格17项内建tuple或list")
        frozen_target: list[float] = []
        for index, value in enumerate(self.target):
            if type(value) not in (int, float):
                raise ValueError(
                    f"navigation_posture.target[{index}]必须是内建int或float，不能使用bool"
                )
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"navigation_posture.target[{index}]必须能表示为有限binary64数值"
                ) from exc
            if not math.isfinite(number):
                raise ValueError(f"navigation_posture.target[{index}]不能包含NaN或Inf")
            if number != 0.0:
                raise ValueError("initial_zero_pose的17维target必须全部为0.0")
            frozen_target.append(number)
        object.__setattr__(self, "target", tuple(frozen_target))
        for name in (
            "joint_state_max_age_ns",
            "max_feedback_gap_ns",
            "settled_required_cycles",
        ):
            object.__setattr__(self, name, _strict_positive_int(getattr(self, name), name))
        for name in (
            "slide_tolerance_m",
            "head_tolerance_rad",
            "arm_tolerance_rad",
            "gripper_tolerance",
            "slide_velocity_tolerance_mps",
            "angular_velocity_tolerance_radps",
            "gripper_velocity_tolerance_per_s",
        ):
            object.__setattr__(
                self, name, _strict_positive_float(getattr(self, name), name)
            )


class _NavigationPostureState(Enum):
    IDLE = "IDLE"
    PREPARING = "PREPARING"
    SETTLING = "SETTLING"
    ACTIVE = "ACTIVE"
    DEVIATED = "DEVIATED"
    INVALID = "INVALID"


class _NavigationPostureTracker:
    """只根据不同timestamp的真实反馈累计INITIAL_ZERO_POSE证据。"""

    def __init__(self, config: _NavigationPostureConfig) -> None:
        if not isinstance(config, _NavigationPostureConfig):
            raise TypeError("config必须是_NavigationPostureConfig")
        self._config = config
        self.reset()

    @property
    def state(self) -> _NavigationPostureState:
        return self._state

    @property
    def settled_cycles(self) -> int:
        return self._settled_cycles

    @property
    def active(self) -> bool:
        return self._state is _NavigationPostureState.ACTIVE

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    @property
    def last_feedback_timestamp_ns(self) -> int | None:
        return self._last_feedback_timestamp_ns

    def reset(self) -> None:
        """清除全部反馈证据；不生成目标、反馈或动作。"""

        self._state = _NavigationPostureState.IDLE
        self._settled_cycles = 0
        self._last_feedback_timestamp_ns: int | None = None
        self._last_feedback_signature: tuple[object, ...] | None = None
        self._failure_reason = ""

    def _latch_invalid(self, reason: str) -> _NavigationPostureState:
        self._state = _NavigationPostureState.INVALID
        self._settled_cycles = 0
        self._failure_reason = reason
        return self._state

    @staticmethod
    def _feedback_signature(actual: RobotJointState) -> tuple[object, ...]:
        return (
            actual.position,
            actual.velocity,
            actual.effort,
            actual.timestamp_ns,
            actual.valid,
            actual.failure_reason,
            actual.joint_names,
        )

    def _position_tolerance(self, index: int) -> float:
        if index == 0:
            return self._config.slide_tolerance_m
        if index in _HEAD_INDICES:
            return self._config.head_tolerance_rad
        if index in _GRIPPER_INDICES:
            return self._config.gripper_tolerance
        if index in _ARM_INDICES:
            return self._config.arm_tolerance_rad
        raise AssertionError("未知JOINT_NAMES索引")

    def _velocity_tolerance(self, index: int) -> float:
        if index == 0:
            return self._config.slide_velocity_tolerance_mps
        if index in _GRIPPER_INDICES:
            return self._config.gripper_velocity_tolerance_per_s
        if index in _HEAD_INDICES or index in _ARM_INDICES:
            return self._config.angular_velocity_tolerance_radps
        raise AssertionError("未知JOINT_NAMES索引")

    def observe(
        self, actual: RobotJointState, now_ns: int
    ) -> _NavigationPostureState:
        """验证一个反馈身份；异常事实锁存，只有显式reset才能重新累计。"""

        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns必须是非负内建int且不能使用bool")
        if self._state in {
            _NavigationPostureState.DEVIATED,
            _NavigationPostureState.INVALID,
        }:
            return self._state
        if not isinstance(actual, RobotJointState):
            return self._latch_invalid("反馈必须是RobotJointState实例")
        if type(getattr(actual, "valid", None)) is not bool or not actual.valid:
            reason = getattr(actual, "failure_reason", "")
            return self._latch_invalid(
                "RobotJointState无效" + (f"：{reason}" if isinstance(reason, str) and reason else "")
            )
        try:
            position = _strict_feedback_vector(actual.position, "RobotJointState.position")
            velocity = _strict_feedback_vector(actual.velocity, "RobotJointState.velocity")
            _strict_feedback_vector(actual.effort, "RobotJointState.effort")
        except ValueError as exc:
            return self._latch_invalid(str(exc))
        if tuple(getattr(actual, "joint_names", ())) != JOINT_NAMES:
            return self._latch_invalid("RobotJointState.joint_names必须严格等于JOINT_NAMES")
        timestamp_ns = getattr(actual, "timestamp_ns", None)
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            return self._latch_invalid("RobotJointState.timestamp_ns必须是非负内建int")
        if timestamp_ns > now_ns:
            return self._latch_invalid("RobotJointState.timestamp_ns不得来自未来")
        if now_ns - timestamp_ns > self._config.joint_state_max_age_ns:
            return self._latch_invalid("RobotJointState超过joint_state_max_age_ns")

        signature = self._feedback_signature(actual)
        previous_timestamp = self._last_feedback_timestamp_ns
        if previous_timestamp is not None:
            if timestamp_ns < previous_timestamp:
                return self._latch_invalid("RobotJointState.timestamp_ns倒退")
            if timestamp_ns == previous_timestamp:
                if signature == self._last_feedback_signature:
                    return self._state
                return self._latch_invalid("相同timestamp_ns的RobotJointState内容发生变化")

        gap_reset = (
            previous_timestamp is not None
            and timestamp_ns - previous_timestamp > self._config.max_feedback_gap_ns
        )
        if gap_reset:
            self._settled_cycles = 0
        self._last_feedback_timestamp_ns = timestamp_ns
        self._last_feedback_signature = signature

        violation = ""
        for index, name in enumerate(JOINT_NAMES):
            position_error = abs(position[index] - self._config.target[index])
            if not math.isfinite(position_error):
                return self._latch_invalid(f"{name}位置误差无法表示为有限binary64数值")
            position_tolerance = self._position_tolerance(index)
            if position_error > position_tolerance:
                violation = (
                    f"{name}[{index}]位置误差{position_error}超过容差{position_tolerance}"
                )
                break
            speed = abs(velocity[index])
            velocity_tolerance = self._velocity_tolerance(index)
            if speed > velocity_tolerance:
                violation = f"{name}[{index}]速度{speed}超过容差{velocity_tolerance}"
                break

        if violation:
            was_active = self._state is _NavigationPostureState.ACTIVE
            self._settled_cycles = 0
            self._failure_reason = violation
            self._state = (
                _NavigationPostureState.DEVIATED
                if was_active
                else _NavigationPostureState.PREPARING
            )
            return self._state

        self._failure_reason = ""
        self._settled_cycles = min(
            self._config.settled_required_cycles,
            self._settled_cycles + 1,
        )
        self._state = (
            _NavigationPostureState.ACTIVE
            if self._settled_cycles >= self._config.settled_required_cycles
            else _NavigationPostureState.SETTLING
        )
        return self._state
