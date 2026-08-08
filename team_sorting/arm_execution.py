"""机械臂纯关节轨迹执行与局部状态映射。

本模块只消费 ``JointTrajectory`` 和实际 ``RobotJointState``，生成短 TTL 的
``ManipulationCommand`` 及反馈驱动的 ``ManipulationStatus``。它不做 IK、不查询 TF、
不推进全局 FSM、不调用 ``ActionMux``、不发布 ROS，也不生成或确认 ``GraspContext``。
"""

from __future__ import annotations

import math
from numbers import Real

from .interfaces import (
    ArmExecutionConfig,
    ArmMotionPhase,
    GlobalPhase,
    GraspVerification,
    JOINT_NAMES,
    JointTrajectory,
    JointWaypoint,
    LocalPhase,
    ManipulationCommand,
    ManipulationStatus,
    RobotJointState,
)


_HEAD_INDICES = (1, 2)
_ARM_INDICES = tuple(range(3, 9)) + tuple(range(10, 16))
_GRIPPER_INDICES = (9, 16)


def _require_integer_ns(value: object, name: str, *, positive: bool) -> int:
    if type(value) is not int:
        raise ValueError(f"{name}必须是真正整数，不能使用bool、浮点数或字符串")
    if positive and value <= 0:
        raise ValueError(f"{name}必须大于0")
    if not positive and value < 0:
        raise ValueError(f"{name}必须大于等于0")
    return value


def _finite_vector_error(values: object, expected_length: int, name: str) -> str:
    if isinstance(values, (str, bytes)):
        return f"{name}必须恰好包含{expected_length}项真实有限数"
    try:
        if len(values) != expected_length:  # type: ignore[arg-type]
            return f"{name}必须恰好包含{expected_length}项"
        items = tuple(values)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return f"{name}必须恰好包含{expected_length}项"
    for index, value in enumerate(items):
        if isinstance(value, bool) or not isinstance(value, Real):
            return f"{name}[{index}]必须是真实数，不能使用bool或字符串"
        if not math.isfinite(float(value)):
            return f"{name}[{index}]不能包含NaN或Inf"
    return ""


def _trajectory_error(trajectory: object) -> str:
    """防御性检查公共轨迹，阻止损坏的反序列化对象进入控制循环。"""

    if not isinstance(trajectory, JointTrajectory):
        return "trajectory必须是JointTrajectory实例"
    valid = getattr(trajectory, "valid", None)
    if type(valid) is not bool:
        return "valid必须是严格bool"
    timestamp_ns = getattr(trajectory, "timestamp_ns", None)
    if type(timestamp_ns) is not int or timestamp_ns < 0:
        return "timestamp_ns必须是非负整数且不能是bool"
    task_id = getattr(trajectory, "task_id", None)
    if type(task_id) is not int or task_id < 0:
        return "task_id必须是非负整数且不能是bool"
    execution_phase = getattr(trajectory, "execution_phase", None)
    if not isinstance(execution_phase, GlobalPhase):
        return "execution_phase必须严格使用GlobalPhase"
    if execution_phase not in {GlobalPhase.EXECUTE_PICK, GlobalPhase.EXECUTE_PLACE}:
        return "execution_phase只能是EXECUTE_PICK或EXECUTE_PLACE"
    failure_reason = getattr(trajectory, "failure_reason", None)
    if not valid:
        if not isinstance(failure_reason, str) or not failure_reason.strip():
            return "无效轨迹必须提供非空failure_reason"
        return failure_reason

    trajectory_id = getattr(trajectory, "trajectory_id", None)
    if not isinstance(trajectory_id, str) or not trajectory_id.strip():
        return "trajectory_id必须是非空字符串"
    target_body = getattr(trajectory, "target_body", None)
    if not isinstance(target_body, str) or not target_body.strip():
        return "target_body必须是非空字符串"
    if not isinstance(failure_reason, str) or failure_reason:
        return "有效轨迹的failure_reason必须是空字符串"
    try:
        waypoints = tuple(getattr(trajectory, "waypoints", None))
    except (TypeError, ValueError, OverflowError):
        return "waypoints必须是非空可迭代路点序列"
    if not waypoints:
        return "轨迹不能为空"

    required_phases = (
        (ArmMotionPhase.PREGRASP, ArmMotionPhase.GRASP,
         ArmMotionPhase.LIFT, ArmMotionPhase.RETREAT)
        if execution_phase is GlobalPhase.EXECUTE_PICK
        else (ArmMotionPhase.PREPLACE, ArmMotionPhase.LOWER,
              ArmMotionPhase.RELEASE, ArmMotionPhase.POST_RELEASE_RETREAT)
    )
    phase_indices = {phase: index for index, phase in enumerate(required_phases)}
    previous_time: float | None = None
    previous_phase_index = 0
    observed_phases: set[ArmMotionPhase] = set()
    trajectory_mask: tuple[bool, ...] | None = None
    for waypoint_index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, JointWaypoint):
            return f"waypoints[{waypoint_index}]必须是JointWaypoint实例"
        phase = getattr(waypoint, "phase", None)
        if not isinstance(phase, ArmMotionPhase):
            return f"waypoints[{waypoint_index}].phase必须是ArmMotionPhase"
        if phase not in phase_indices:
            return f"{execution_phase.value}轨迹不允许阶段{phase.value}"
        phase_index = phase_indices[phase]
        if phase_index < previous_phase_index:
            return "waypoints.phase只能按规定顺序前进，不得倒退"
        previous_phase_index = phase_index
        observed_phases.add(phase)
        time_value = getattr(waypoint, "time_from_start_s", None)
        if isinstance(time_value, bool) or not isinstance(time_value, Real):
            return f"waypoints[{waypoint_index}].time_from_start_s必须是真实有限数"
        time_s = float(time_value)
        if not math.isfinite(time_s):
            return f"waypoints[{waypoint_index}].time_from_start_s不能包含NaN或Inf"
        if time_s < 0.0:
            return f"waypoints[{waypoint_index}].time_from_start_s不能小于0"
        if previous_time is not None and time_s <= previous_time:
            return "waypoints时间必须严格递增，不能重复或倒退"
        previous_time = time_s
        position_error = _finite_vector_error(
            getattr(waypoint, "joint_position", None), 17,
            f"waypoints[{waypoint_index}].joint_position",
        )
        if position_error:
            return position_error
        mask = getattr(waypoint, "controlled_mask", None)
        try:
            if len(mask) != 17:  # type: ignore[arg-type]
                return f"waypoints[{waypoint_index}].controlled_mask必须恰好包含17项"
            mask_items = tuple(mask)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return f"waypoints[{waypoint_index}].controlled_mask必须恰好包含17项"
        if any(type(item) is not bool for item in mask_items):
            return f"waypoints[{waypoint_index}].controlled_mask每项都必须是真正bool"
        if not any(mask_items):
            return f"waypoints[{waypoint_index}].controlled_mask至少一项必须为True"
        if mask_items[1] or mask_items[2]:
            return f"waypoints[{waypoint_index}]不得控制head关节"
        if mask_items[9] and not 0.0 <= float(waypoint.joint_position[9]) <= 1.0:
            return f"waypoints[{waypoint_index}]受控左夹爪目标必须位于[0,1]"
        if mask_items[16] and not 0.0 <= float(waypoint.joint_position[16]) <= 1.0:
            return f"waypoints[{waypoint_index}]受控右夹爪目标必须位于[0,1]"
        if trajectory_mask is None:
            trajectory_mask = mask_items
        elif mask_items != trajectory_mask:
            return "有效轨迹所有waypoint的controlled_mask必须完全一致"
    missing = tuple(phase.value for phase in required_phases if phase not in observed_phases)
    if missing:
        return f"轨迹缺少必要阶段：{missing}"
    return ""


def _safe_trajectory_timestamp(trajectory: object) -> int:
    value = getattr(trajectory, "timestamp_ns", None)
    return value if type(value) is int and value >= 0 else 0


class ArmExecutionController:
    """反馈驱动的双臂纯轨迹执行器。

    正式执行必须显式注入 ``ArmExecutionConfig``。暂时允许无参构造，仅用于尚未获准修改
    的 ROS 骨架兼容；该实例不能装载或执行轨迹，并始终失败关闭，不含任何默认参数。
    当前只提供 ``accept_grasp_verification`` 接收并缓存验证结果；是否据此解除
    ``VERIFY`` 锁定，仍由后续 ROS/FSM 接线评审决定，本执行器默认保持关闭。
    """

    def __init__(self, config: ArmExecutionConfig | None = None) -> None:
        if config is not None and not isinstance(config, ArmExecutionConfig):
            raise TypeError("config必须是ArmExecutionConfig")
        self._config = config
        self.local_phase = LocalPhase.IDLE
        self._trajectory: JointTrajectory | None = None
        self._waypoint_index = 0
        self._trajectory_started_ns: int | None = None
        self._stable_cycle_count = 0
        self._last_step_ns: int | None = None
        self._last_feedback_timestamp_ns: int | None = None
        self._last_command: tuple[float, ...] | None = None
        self._initial_position: tuple[float, ...] | None = None
        self._terminal_status: ManipulationStatus | None = None
        self._cached_verification: GraspVerification | None = None
        self._verification_received_ns: int | None = None

    def accept_grasp_verification(
        self, verification: "GraspVerification"
    ) -> None:
        """校验并缓存外部抓取验证，留待后续获批的 ROS/FSM 组装层消费。

        本入口只负责接收，不确认 ``GraspContext``，也不会改变局部阶段或让
        ``step`` 自动越过 ``VERIFY``。所有检查在写入前完成，避免非法输入
        覆盖此前已经接收的有效验证。

        校验规则：
        - ``is_grasped`` / ``success``：严格 ``bool``；
        - ``confidence``：有限实数，范围 ``[0.0, 1.0]``，禁止 ``bool``；
        - ``visual_evidence`` / ``effort_evidence`` / ``failure_reason``：``str``；
        - ``timestamp_ns``：非负 ``int``，禁止 ``bool``；
        - 时间单调性：旧 timestamp 拒绝；相同 timestamp 且内容相同可幂等；
          相同 timestamp 但内容不同则 fail closed。
        """

        if not isinstance(verification, GraspVerification):
            raise ValueError("verification必须是GraspVerification实例")

        required_fields = (
            "is_grasped",
            "confidence",
            "visual_evidence",
            "effort_evidence",
            "success",
            "failure_reason",
            "timestamp_ns",
        )
        missing_fields = tuple(
            name for name in required_fields if not hasattr(verification, name)
        )
        if missing_fields:
            raise ValueError(
                f"verification字段不完整，缺少：{', '.join(missing_fields)}"
            )

        # 1. 严格字段类型/范围校验（任何失败都不应覆盖缓存）
        if type(verification.is_grasped) is not bool:
            raise ValueError("verification.is_grasped必须是严格bool")

        confidence = verification.confidence
        if type(confidence) is bool or not isinstance(confidence, Real):
            raise ValueError("verification.confidence必须是数值且不能是bool")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("verification.confidence必须能转换为有限浮点数") from exc
        if not math.isfinite(confidence_value):
            raise ValueError("verification.confidence必须是有限数值")
        if not (0.0 <= confidence_value <= 1.0):
            raise ValueError("verification.confidence必须位于[0.0, 1.0]")

        if not isinstance(verification.visual_evidence, str):
            raise ValueError("verification.visual_evidence必须是字符串")
        if not isinstance(verification.effort_evidence, str):
            raise ValueError("verification.effort_evidence必须是字符串")

        if type(verification.success) is not bool:
            raise ValueError("verification.success必须是严格bool")
        if not isinstance(verification.failure_reason, str):
            raise ValueError("verification.failure_reason必须是字符串")

        timestamp_ns = verification.timestamp_ns
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            raise ValueError("verification.timestamp_ns必须是非负整数且不能是bool")

        # 2. 时间单调性校验（事务式：失败不覆盖缓存）
        cached = self._cached_verification
        if cached is not None:
            if timestamp_ns < cached.timestamp_ns:
                raise ValueError(
                    "verification.timestamp_ns不能早于已缓存的timestamp_ns"
                )
            if timestamp_ns == cached.timestamp_ns:
                if verification == cached:
                    return
                raise ValueError(
                    "同一timestamp_ns下收到内容不同的verification，拒绝冲突重复"
                )

        # 3. 全部校验通过，才覆盖缓存
        self._cached_verification = verification
        self._verification_received_ns = timestamp_ns

    def create_hold_command(
        self, actual_joints: RobotJointState, timestamp_ns: int, valid_for_ns: int
    ) -> ManipulationCommand:
        """显式创建主动保持候选；此辅助方法不由轨迹失败路径自动调用。"""

        timestamp_ns = _require_integer_ns(timestamp_ns, "timestamp_ns", positive=False)
        valid_for_ns = _require_integer_ns(valid_for_ns, "valid_for_ns", positive=True)
        structure_error = self._joint_state_structure_error(actual_joints)
        if structure_error:
            return ManipulationCommand(
                self._fallback_position(actual_joints), (False,) * 17, self.local_phase,
                timestamp_ns, timestamp_ns, False,
                f"实际关节状态不能生成安全保持：{structure_error}",
            )
        return ManipulationCommand(
            actual_joints.position, (True,) * 17, self.local_phase,
            timestamp_ns, timestamp_ns + valid_for_ns,
        )

    def start_trajectory(self, trajectory: JointTrajectory) -> ManipulationStatus:
        """原子装载轨迹；轨迹时钟只在第一次合法 ``step`` 启动。"""

        failure_reason = _trajectory_error(trajectory)
        if not failure_reason and self._config is None:
            failure_reason = "ArmExecutionConfig未注入，执行失败关闭"
        self.reset()
        if failure_reason:
            self.local_phase = LocalPhase.FAILED
            status = ManipulationStatus(
                self.local_phase, "REJECTED", 0.0, float("inf"), False,
                failure_reason, _safe_trajectory_timestamp(trajectory),
            )
            self._terminal_status = status
            return status
        self._trajectory = trajectory
        return ManipulationStatus(
            self.local_phase, "LOADED", 0.0, float("inf"), False,
            "轨迹已装载，等待首次合法反馈启动", _safe_trajectory_timestamp(trajectory),
        )

    def reset(self) -> None:
        """只清除全部内存执行状态；不生成控制命令，也不代表机器人已经停止。"""

        self._trajectory = None
        self._waypoint_index = 0
        self._trajectory_started_ns = None
        self._stable_cycle_count = 0
        self._last_step_ns = None
        self._last_feedback_timestamp_ns = None
        self._last_command = None
        self._initial_position = None
        self._terminal_status = None
        self._cached_verification = None
        self._verification_received_ns = None
        self.local_phase = LocalPhase.IDLE

    @staticmethod
    def _joint_state_structure_error(actual: object) -> str:
        if not isinstance(actual, RobotJointState):
            return "actual_joints必须是RobotJointState实例"
        if type(getattr(actual, "valid", None)) is not bool:
            return "actual_joints.valid必须是严格bool"
        if not actual.valid:
            reason = getattr(actual, "failure_reason", "")
            return "actual_joints无效" + (f"：{reason}" if isinstance(reason, str) and reason else "")
        error = _finite_vector_error(getattr(actual, "position", None), 17, "actual_joints.position")
        if error:
            return error
        try:
            if "joint_names" not in vars(actual):
                return "actual_joints.joint_names缺失，必须严格使用JOINT_NAMES顺序"
        except TypeError:
            return "actual_joints.joint_names缺失，必须严格使用JOINT_NAMES顺序"
        try:
            joint_names = tuple(getattr(actual, "joint_names"))
        except (AttributeError, TypeError, ValueError):
            return "actual_joints.joint_names必须严格使用JOINT_NAMES顺序"
        if joint_names != JOINT_NAMES:
            return "actual_joints.joint_names必须严格使用JOINT_NAMES顺序"
        if not 0.0 <= float(actual.position[9]) <= 1.0:
            return "actual_joints左夹爪反馈必须位于[0,1]"
        if not 0.0 <= float(actual.position[16]) <= 1.0:
            return "actual_joints右夹爪反馈必须位于[0,1]"
        return ""

    @classmethod
    def _actual_error(cls, actual: object, timestamp_ns: int, max_age_ns: int) -> str:
        structure_error = cls._joint_state_structure_error(actual)
        if structure_error:
            return structure_error
        assert isinstance(actual, RobotJointState)
        feedback_ns = getattr(actual, "timestamp_ns", None)
        if type(feedback_ns) is not int or feedback_ns < 0:
            return "actual_joints.timestamp_ns必须是非负整数且不能是bool"
        if feedback_ns > timestamp_ns:
            return "actual_joints.timestamp_ns不得来自未来"
        if timestamp_ns - feedback_ns > max_age_ns:
            return "actual_joints反馈已过期"
        return ""

    def _fallback_position(self, actual: object) -> tuple[float, ...]:
        if isinstance(actual, RobotJointState):
            error = _finite_vector_error(getattr(actual, "position", None), 17, "actual")
            if not error:
                return tuple(float(value) for value in actual.position)
        if self._last_command is not None:
            return self._last_command
        if self._initial_position is not None:
            return self._initial_position
        return (0.0,) * 17

    def _inactive_command(
        self, actual: object, timestamp_ns: int, reason: str
    ) -> ManipulationCommand:
        return ManipulationCommand(
            self._fallback_position(actual), (False,) * 17, self.local_phase,
            timestamp_ns, timestamp_ns, False, reason,
        )

    def _fail(
        self, actual: object, timestamp_ns: int, reason: str,
        max_error: float = float("inf"),
    ) -> tuple[ManipulationCommand, ManipulationStatus]:
        self.local_phase = LocalPhase.FAILED
        status = ManipulationStatus(
            self.local_phase, "FAILED", self._progress(), max_error,
            False, reason, timestamp_ns,
        )
        self._terminal_status = status
        return self._inactive_command(actual, timestamp_ns, reason), status

    def _progress(self) -> float:
        if self._trajectory is None:
            return 0.0
        try:
            waypoint_count = len(self._trajectory.waypoints)
        except (AttributeError, TypeError, ValueError):
            return 0.0
        if waypoint_count <= 0:
            return 0.0
        return min(1.0, self._waypoint_index / waypoint_count)

    def _phase_for_current(self, reached: bool) -> LocalPhase:
        assert self._trajectory is not None
        waypoint = self._trajectory.waypoints[self._waypoint_index]
        phase = waypoint.phase
        if self._trajectory.execution_phase is GlobalPhase.EXECUTE_PLACE:
            return {
                ArmMotionPhase.PREPLACE: LocalPhase.MOVE_PREPLACE,
                ArmMotionPhase.LOWER: LocalPhase.LOWER_OBJECT,
                ArmMotionPhase.RELEASE: LocalPhase.RELEASE,
                ArmMotionPhase.POST_RELEASE_RETREAT: LocalPhase.STOW,
            }[phase]
        if phase is ArmMotionPhase.PREGRASP:
            return (
                LocalPhase.HUG_OPEN
                if reached and self._is_last_waypoint_of_current_phase()
                else LocalPhase.MOVE_PREGRASP
            )
        if phase is ArmMotionPhase.GRASP:
            return (
                LocalPhase.HUG_CLOSE
                if self._is_last_waypoint_of_current_phase()
                else LocalPhase.APPROACH
            )
        if phase is ArmMotionPhase.LIFT:
            return (
                LocalPhase.VERIFY
                if reached and self._is_last_waypoint_of_current_phase()
                else LocalPhase.TEST_LIFT
            )
        return (
            LocalPhase.TRANSPORT_HOLD
            if reached and self._is_last_waypoint_of_current_phase()
            else LocalPhase.RETREAT
        )

    def _is_last_waypoint_of_current_phase(self) -> bool:
        assert self._trajectory is not None
        next_index = self._waypoint_index + 1
        return (
            next_index == len(self._trajectory.waypoints)
            or self._trajectory.waypoints[next_index].phase
            is not self._trajectory.waypoints[self._waypoint_index].phase
        )

    @staticmethod
    def _limit_for_index(index: int, config: ArmExecutionConfig) -> float:
        if index == 0:
            return config.max_slide_velocity_m_s
        if index in _GRIPPER_INDICES:
            return config.max_gripper_velocity_per_s
        return config.max_arm_velocity_rad_s

    @staticmethod
    def _tolerance_for_index(index: int, config: ArmExecutionConfig) -> float:
        if index == 0:
            return config.slide_tolerance_m
        if index in _GRIPPER_INDICES:
            return config.gripper_tolerance
        return config.arm_tolerance_rad

    def _initial_error(self, actual: tuple[float, ...], waypoint: object) -> str:
        assert self._config is not None
        limits = (
            self._config.initial_slide_error_limit_m,
            self._config.initial_arm_error_limit_rad,
            self._config.initial_gripper_error_limit,
        )
        errors = [abs(actual[i] - waypoint.joint_position[i]) for i in range(17)
                  if waypoint.controlled_mask[i]]
        if waypoint.controlled_mask[0] and abs(actual[0] - waypoint.joint_position[0]) > limits[0]:
            return "首路点slide误差超过initial_slide_error_limit_m"
        if any(waypoint.controlled_mask[i] and
               abs(actual[i] - waypoint.joint_position[i]) > limits[1] for i in _ARM_INDICES):
            return "首路点arm误差超过initial_arm_error_limit_rad"
        if any(waypoint.controlled_mask[i] and
               abs(actual[i] - waypoint.joint_position[i]) > limits[2] for i in _GRIPPER_INDICES):
            return "首路点gripper误差超过initial_gripper_error_limit"
        assert errors
        return ""

    def step(
        self, actual_joints: RobotJointState, timestamp_ns: int
    ) -> tuple[ManipulationCommand, ManipulationStatus]:
        """按实际反馈推进最多一个路点，并输出本周期经分组限速的候选。"""

        try:
            timestamp_ns = _require_integer_ns(timestamp_ns, "timestamp_ns", positive=False)
        except ValueError as exc:
            safe_ns = self._last_step_ns if self._last_step_ns is not None else 0
            if self._terminal_status is not None:
                return (
                    self._inactive_command(
                        actual_joints, safe_ns,
                        self._terminal_status.failure_reason or "执行器已进入终态",
                    ),
                    self._terminal_status,
                )
            return self._fail(actual_joints, safe_ns, str(exc))
        if self._last_step_ns is not None and timestamp_ns < self._last_step_ns:
            if self._terminal_status is not None:
                return (
                    self._inactive_command(
                        actual_joints, timestamp_ns,
                        self._terminal_status.failure_reason or "执行器已进入终态",
                    ),
                    self._terminal_status,
                )
            return self._fail(actual_joints, timestamp_ns, "step时间不得倒退")
        if self._terminal_status is not None:
            self._last_step_ns = timestamp_ns
            return (
                self._inactive_command(
                    actual_joints, timestamp_ns,
                    self._terminal_status.failure_reason or "执行器已进入终态",
                ),
                self._terminal_status,
            )
        if self._config is None:
            return self._fail(actual_joints, timestamp_ns, "ArmExecutionConfig未注入，执行失败关闭")
        actual_error = self._actual_error(actual_joints, timestamp_ns, self._config.feedback_max_age_ns)
        if actual_error:
            return self._fail(actual_joints, timestamp_ns, actual_error)
        actual = tuple(float(value) for value in actual_joints.position)
        feedback_timestamp_ns = actual_joints.timestamp_ns
        if (
            self._last_feedback_timestamp_ns is not None
            and feedback_timestamp_ns < self._last_feedback_timestamp_ns
        ):
            return self._fail(actual_joints, timestamp_ns, "actual_joints反馈时间倒退")
        is_new_feedback = (
            self._last_feedback_timestamp_ns is None
            or feedback_timestamp_ns > self._last_feedback_timestamp_ns
        )
        if is_new_feedback:
            self._last_feedback_timestamp_ns = feedback_timestamp_ns
        if self._trajectory is None:
            self._last_step_ns = timestamp_ns
            reason = "未装载JointTrajectory"
            return self._inactive_command(actual_joints, timestamp_ns, reason), ManipulationStatus(
                LocalPhase.IDLE, "NO_TRAJECTORY", 0.0, float("inf"), False, reason, timestamp_ns,
            )
        if self._last_step_ns is not None and timestamp_ns - self._last_step_ns > self._config.max_control_period_ns:
            return self._fail(actual_joints, timestamp_ns, "控制周期间隔超过max_control_period_ns")
        trajectory_error = _trajectory_error(self._trajectory)
        if trajectory_error:
            return self._fail(
                actual_joints, timestamp_ns,
                f"已装载轨迹运行期校验失败：{trajectory_error}",
            )

        if self._trajectory_started_ns is None:
            if self._trajectory.timestamp_ns > timestamp_ns:
                return self._fail(actual_joints, timestamp_ns, "轨迹生成时间不得来自未来")
            if timestamp_ns - self._trajectory.timestamp_ns > self._config.trajectory_max_age_ns:
                return self._fail(actual_joints, timestamp_ns, "轨迹在首次执行前已过期")
            self._trajectory_started_ns = timestamp_ns
            self._initial_position = actual
            self._last_command = actual
            first = self._trajectory.waypoints[0]
            if first.time_from_start_s == 0.0:
                initial_error = self._initial_error(actual, first)
                if initial_error:
                    return self._fail(actual_joints, timestamp_ns, initial_error)

        assert self._trajectory_started_ns is not None
        assert self._initial_position is not None
        assert self._last_command is not None
        elapsed_ns = timestamp_ns - self._trajectory_started_ns
        waypoint = self._trajectory.waypoints[self._waypoint_index]
        waypoint_time_ns = int(round(waypoint.time_from_start_s * 1_000_000_000))
        final_time_ns = int(round(self._trajectory.waypoints[-1].time_from_start_s * 1_000_000_000))
        if elapsed_ns > final_time_ns + self._config.total_timeout_margin_ns:
            return self._fail(actual_joints, timestamp_ns, "整条轨迹超过total_timeout_margin_ns")
        waiting_for_verification = (
            self._trajectory.execution_phase is GlobalPhase.EXECUTE_PICK
            and waypoint.phase is ArmMotionPhase.LIFT
            and self._is_last_waypoint_of_current_phase()
            and self.local_phase is LocalPhase.VERIFY
        )
        if (
            not waiting_for_verification
            and elapsed_ns > waypoint_time_ns + self._config.waypoint_timeout_margin_ns
        ):
            return self._fail(actual_joints, timestamp_ns, "当前路点超过waypoint_timeout_margin_ns")

        if self._waypoint_index == 0:
            start_position = self._initial_position
            start_time_ns = 0
        else:
            previous = self._trajectory.waypoints[self._waypoint_index - 1]
            start_position = previous.joint_position
            start_time_ns = int(round(previous.time_from_start_s * 1_000_000_000))
        duration_ns = waypoint_time_ns - start_time_ns
        alpha = 1.0 if duration_ns <= 0 else min(1.0, max(0.0, (elapsed_ns - start_time_ns) / duration_ns))
        desired = list(actual)
        for index in range(17):
            if waypoint.controlled_mask[index]:
                desired[index] = start_position[index] + alpha * (
                    waypoint.joint_position[index] - start_position[index]
                )
        dt_s = 0.0 if self._last_step_ns is None else (timestamp_ns - self._last_step_ns) / 1_000_000_000
        limited = list(actual)
        for index in range(17):
            if not waypoint.controlled_mask[index]:
                continue
            max_delta = self._limit_for_index(index, self._config) * dt_s
            delta = desired[index] - self._last_command[index]
            limited[index] = self._last_command[index] + max(-max_delta, min(max_delta, delta))

        errors = [abs(actual[i] - waypoint.joint_position[i]) for i in range(17)
                  if waypoint.controlled_mask[i]]
        reached = all(
            abs(actual[i] - waypoint.joint_position[i]) <= self._tolerance_for_index(i, self._config)
            for i in range(17) if waypoint.controlled_mask[i]
        )
        if not reached:
            self._stable_cycle_count = 0
        elif is_new_feedback:
            self._stable_cycle_count = min(
                self._config.settle_cycles,
                self._stable_cycle_count + 1,
            )
        settled = reached and self._stable_cycle_count >= self._config.settle_cycles
        self.local_phase = self._phase_for_current(settled)
        max_error = max(errors)
        command = ManipulationCommand(
            tuple(limited), tuple(waypoint.controlled_mask), self.local_phase,
            timestamp_ns, timestamp_ns + self._config.command_ttl_ns, True, "",
        )
        self._last_command = tuple(limited)
        self._last_step_ns = timestamp_ns

        if settled and elapsed_ns >= waypoint_time_ns:
            if (
                self._trajectory.execution_phase is GlobalPhase.EXECUTE_PICK
                and waypoint.phase is ArmMotionPhase.LIFT
                and self._is_last_waypoint_of_current_phase()
            ):
                self.local_phase = LocalPhase.VERIFY
                command = ManipulationCommand(
                    tuple(limited), tuple(waypoint.controlled_mask), self.local_phase,
                    timestamp_ns, timestamp_ns + self._config.command_ttl_ns, True, "",
                )
                return command, ManipulationStatus(
                    self.local_phase, "VERIFICATION_PENDING", self._progress(),
                    max_error, False,
                    "LIFT已稳定到位，等待真实GraspVerification；不得确认GraspContext",
                    timestamp_ns,
                )
        if settled and is_new_feedback and elapsed_ns >= waypoint_time_ns:
            self._stable_cycle_count = 0
            self._waypoint_index += 1
            if self._waypoint_index == len(self._trajectory.waypoints):
                self.local_phase = LocalPhase.IDLE
                reason = "放置轨迹运动学执行完成；物体位置与裁判语义仍待外部验证"
                status = ManipulationStatus(
                    self.local_phase,
                    "MOTION_COMPLETED_PLACE_VERIFICATION_PENDING",
                    1.0, max_error, False, reason, timestamp_ns,
                )
                self._terminal_status = status
                return self._inactive_command(actual_joints, timestamp_ns, status.failure_reason or "轨迹运动学执行已完成"), status

        return command, ManipulationStatus(
            self.local_phase, "RUNNING", self._progress(), max_error,
            False, "尚未完成当前轨迹", timestamp_ns,
        )
