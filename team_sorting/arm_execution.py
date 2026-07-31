"""机械臂轨迹执行和局部抓放状态机骨架。

本文件位于"轨迹计划"和"最终控制仲裁"之间，完整数据流是：

``JointTrajectory``
→ ``ArmExecutionController.start_trajectory`` 装载并校验
→ ``step`` 按实际 ``RobotJointState`` 插值和判断进度
→ ``ManipulationCommand`` 候选关节目标 + ``ManipulationStatus`` 执行状态
→ ``ActionMux`` 结合实际反馈、TTL和FSM安全阶段生成 ``FinalAction[19]``
→ ``OfficialCommandPublisher`` 拆分并发布官方关节话题。

``ManipulationCommand`` 只是本周期候选建议，不是已经发布或已经执行的动作；到位、试抬
和抓放验证必须依据实际反馈，不能只看目标命令或等待时间。本文件不负责IK、抓取位姿
规划、ROS2发布、全局FSM推进或视觉算法，也不能绕过 ``ActionMux``。

当前定位为：通用轨迹采样器——负责时间插值、controlled_mask、关节限速、到位判断和
稳定确认。不负责抓放局部阶段状态机（MOVE_PREGRASP 等），local_phase 不驱动 FSM 阶段
推进。FSM 应依据 ManipulationStatus.success 和 failure_reason 判断执行结果。
"""

from __future__ import annotations

import math
from numbers import Real

from .interfaces import (
    JointTrajectory,
    LocalPhase,
    ManipulationCommand,
    ManipulationStatus,
    RobotJointState,
)


def _require_integer_ns(value: object, name: str, *, positive: bool) -> int:
    """严格读取纳秒整数，避免 ``int()`` 把浮点数、字符串或bool悄悄改成时间。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}必须是真正整数，不能使用bool、浮点数或字符串")
    if positive and value <= 0:
        raise ValueError(f"{name}必须大于0")
    if not positive and value < 0:
        raise ValueError(f"{name}必须大于等于0")
    return value


def _finite_vector_error(values: object, expected_length: int, name: str) -> str:
    """返回定长真实有限数向量的错误原因；空字符串表示校验通过。"""

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


def _trajectory_error(trajectory: JointTrajectory) -> str:
    """防御性检查公共轨迹，防止损坏对象进入未来的控制循环。"""

    if not trajectory.valid:
        return trajectory.failure_reason or "轨迹标记为无效"
    if not isinstance(trajectory.trajectory_id, str) or not trajectory.trajectory_id.strip():
        return "trajectory_id必须是非空字符串"
    try:
        waypoints = tuple(trajectory.waypoints)
    except (TypeError, ValueError, OverflowError):
        return "waypoints必须是非空可迭代路点序列"
    if not waypoints:
        return "轨迹不能为空"

    previous_time: float | None = None
    for waypoint_index, waypoint in enumerate(waypoints):
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
            getattr(waypoint, "joint_position", None),
            17,
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
    return ""


from dataclasses import dataclass


@dataclass(frozen=True)
class ArmExecutionConfig:
    """机械臂执行器不可变安全参数。

    所有字段均可为 None 表示未配置。安全关键参数缺失时 step() 进入 fail closed 模式——
    拒绝返回 COMPLETED, 永不伪造成功。
    """
    joint_tolerance_17: tuple[float, ...] | None = None
    max_joint_velocity_17: tuple[float, ...] | None = None
    settle_cycles: int | None = None
    total_timeout_ns: int | None = None
    command_ttl_ns: int | None = None
    feedback_max_age_ns: int | None = None
    trajectory_max_age_ns: int | None = None

    def __post_init__(self) -> None:
        if self.joint_tolerance_17 is not None:
            err = _finite_vector_error(self.joint_tolerance_17, 17, "joint_tolerance_17")
            if err:
                raise ValueError(err)
            if not all(v > 0.0 for v in self.joint_tolerance_17):
                raise ValueError("joint_tolerance_17 各项必须为正")
        if self.max_joint_velocity_17 is not None:
            err = _finite_vector_error(self.max_joint_velocity_17, 17, "max_joint_velocity_17")
            if err:
                raise ValueError(err)
            if not all(v > 0.0 for v in self.max_joint_velocity_17):
                raise ValueError("max_joint_velocity_17 各项必须为正")
        for name, value in (
            ("settle_cycles", self.settle_cycles),
            ("total_timeout_ns", self.total_timeout_ns),
            ("command_ttl_ns", self.command_ttl_ns),
            ("feedback_max_age_ns", self.feedback_max_age_ns),
            ("trajectory_max_age_ns", self.trajectory_max_age_ns),
        ):
            if value is not None:
                _require_integer_ns(value, name, positive=True)


class ArmExecutionController:
    """双臂轨迹执行器和局部状态机骨架。

    输入为17维实际关节、目标轨迹和纳秒时间，输出17维候选命令及执行状态。slide单位
    米、旋转关节弧度、夹爪为官方0～1控制量。

    本 PR 范围为"通用轨迹采样器"：负责轨迹时间插值、controlled_mask 处理、关节限速、
    实际到位判断和稳定确认。不负责抓放局部阶段状态机（MOVE_PREGRASP/HUG_OPEN 等），
    local_phase 来源待上游协议冻结后接入。
    """

    def __init__(self, config: ArmExecutionConfig | None = None) -> None:
        """创建执行器，可选注入安全参数配置。

        config 为 None 时使用空配置，所有安全参数缺失 → 进入 fail closed 模式：
        轨迹可以插值计算目标，但决不返回 COMPLETED/success=True，
        直到队长确认参数后通过 ArmExecutionConfig 注入。
        """

        self._config = config or ArmExecutionConfig()
        self.local_phase = LocalPhase.IDLE
        self._trajectory: JointTrajectory | None = None
        # 这些字段为step实现预留统一清理点，拒绝新轨迹时不能沿用上一条的进度。
        self._waypoint_index = 0
        self._trajectory_started_ns: int | None = None
        self._stable_cycle_count = 0
        self._cached_verification = None
        self._last_step_ns: int | None = None
        self._completed = False
        self._implicit_start_position: tuple[float, ...] | None = None

    def create_hold_command(
        self, actual_joints: RobotJointState, timestamp_ns: int, valid_for_ns: int
    ) -> ManipulationCommand:
        """用实际反馈创建全关节安全保持命令。

        参数：17维实际关节、非负整数 ``timestamp_ns`` 和正整数 ``valid_for_ns``；二者
        单位都是纳秒。TTL像候选命令的保质期，过期后由 ``ActionMux`` 不再采用。

        实际反馈有效时，返回"目标严格等于当前实际位置、17项均受控"的保持候选；这表示
        主动维持当前姿态，不是回到全零。实际反馈无效时返回 ``valid=False`` 且不伪造
        关节目标。时间参数不合法时抛出 ``ValueError``。
        """

        timestamp_ns = _require_integer_ns(timestamp_ns, "timestamp_ns", positive=False)
        valid_for_ns = _require_integer_ns(valid_for_ns, "valid_for_ns", positive=True)
        if not actual_joints.valid:
            return ManipulationCommand(
                joint_target=actual_joints.position,
                controlled_mask=(False,) * 17,
                local_phase=self.local_phase,
                timestamp_ns=timestamp_ns,
                valid_until_ns=timestamp_ns,
                valid=False,
                failure_reason=(
                    "实际关节状态无效，不能生成安全保持"
                    + (f"：{actual_joints.failure_reason}" if actual_joints.failure_reason else "")
                ),
            )
        return ManipulationCommand(
            joint_target=actual_joints.position,
            controlled_mask=(True,) * 17,
            local_phase=self.local_phase,
            timestamp_ns=timestamp_ns,
            valid_until_ns=timestamp_ns + valid_for_ns,
            valid=True,
        )

    def start_trajectory(self, trajectory: JointTrajectory) -> ManipulationStatus:
        """装载一条待执行关节轨迹。

        参数是17维关节路点轨迹，路点时间单位秒；返回 ``ManipulationStatus``。该入口
        会再次检查ID、时间严格递增、17维有限目标和17项布尔mask，因为公共dataclass
        可能来自反序列化、旧代码或损坏测试对象，不能假定 ``valid=True`` 就一定安全。

        装载采用原子语义：先清除旧轨迹的索引、开始时间、稳定计数和验证缓存，再保存
        通过检查的新轨迹。拒绝新轨迹时同样清除旧执行上下文，避免下一周期误执行旧动作。
        ``LOADED`` 只表示轨迹已经保存，尚未开始插值，更不表示机器人到位。
        """

        failure_reason = _trajectory_error(trajectory)
        if failure_reason:
            self.reset()
            self.local_phase = LocalPhase.FAILED
            return ManipulationStatus(
                local_phase=self.local_phase,
                state="REJECTED",
                progress=0.0,
                max_joint_error=float("inf"),
                success=False,
                failure_reason=failure_reason,
                timestamp_ns=trajectory.timestamp_ns,
            )
        self.reset()
        self._trajectory = trajectory
        return ManipulationStatus(
            local_phase=self.local_phase,
            state="LOADED",
            progress=0.0,
            max_joint_error=float("inf"),
            success=False,
            failure_reason="轨迹已装载，但插值执行尚未开始",
            timestamp_ns=trajectory.timestamp_ns,
        )

    def reset(self) -> None:
        """清除当前轨迹和全部运行期状态，并回到 ``IDLE``。

        Episode结束、任务切换或失败恢复准备重新装载时可调用。该方法只清内存状态，不会
        生成关节命令；调用方仍需根据最新 ``RobotJointState`` 通过正常控制链创建保持
        候选，不能把reset理解成机器人已经停止。
        """

        self._trajectory = None
        self._waypoint_index = 0
        self._trajectory_started_ns = None
        self._stable_cycle_count = 0
        self._cached_verification = None
        self._last_step_ns = None
        self._completed = False
        self._implicit_start_position: tuple[float, ...] | None = None
        self.local_phase = LocalPhase.IDLE

    def step(
        self, actual_joints: RobotJointState, timestamp_ns: int
    ) -> tuple[ManipulationCommand, ManipulationStatus]:
        """按实际关节反馈推进一次局部轨迹与抓放状态机。

        参数实际关节单位同 ``RobotJointState``，时间为纳秒；未来每个控制周期返回一个
        短TTL ``ManipulationCommand`` 和对应 ``ManipulationStatus``。机械臂2实现时需要：

        1. 校验实际反馈、时间和已装载轨迹；按相对时间找到相邻路点并插值；
        2. 严格遵守 ``controlled_mask``，未受控关节保持实际位置而不是填零；
        3. 对slide、手臂、夹爪分别使用经评审的限速、容差和稳定周期；
        4. 依实际误差推进预抓、张开、靠近、合拢、试抬、验证、撤离、运输、预放、
           下放、释放和收回阶段；
        5. 轨迹超时、反馈无效、误差不收敛或验证失败时返回明确失败并保持安全；
        6. 把实际反馈证据形成结构化验证结果，再由ROS组装层/全局FSM转成业务事件。

        抓取验证至少要用实际关节反馈；视觉证据应由感知模块通过后续评审的结构化接口
        提供，本文件不能读取图像。当前仓库尚未把 ``GraspVerification`` 接入生产控制链，
        也没有稳定的提交方法，因此本次不猜测验证来源或改变 ``step`` 签名。

        上述算法仍未实现，当前始终抛出 ``NotImplementedError``。不能只因命令已发布、
        轨迹时间走完或目标误差偶然变小就报告成功。
        """

        # ---- 0. 已完成终态检查 ----
        if self._completed:
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason="轨迹已完成，不产生新命令",
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="COMPLETED",
                    progress=1.0,
                    max_joint_error=0.0,
                    success=True,
                    failure_reason="",
                    timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 1. 输入校验 ----
        timestamp_ns = _require_integer_ns(timestamp_ns, "timestamp_ns", positive=False)
        if not actual_joints.valid:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            reason = "实际关节状态无效，不能推进轨迹执行"
            if actual_joints.failure_reason:
                reason += f"：{actual_joints.failure_reason}"
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=reason,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=reason, timestamp_ns=timestamp_ns,
                ),
            )
        position_error = _finite_vector_error(
            actual_joints.position, 17, "actual_joints.position"
        )
        if position_error:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=position_error,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=position_error, timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 2. 先检查 FAILED ----
        if self.local_phase is LocalPhase.FAILED:
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason="局部执行已处于FAILED状态",
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason="局部执行已处于FAILED状态", timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 3. 无轨迹 → IDLE ----
        if self._trajectory is None:
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason="尚未装载有效轨迹",
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="IDLE", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason="尚未装载有效轨迹", timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 4. 配置完整性检查 ----
        cfg = self._config
        config_errors: list[str] = []
        if cfg.max_joint_velocity_17 is None:
            config_errors.append("max_joint_velocity_17")
        if cfg.total_timeout_ns is None:
            config_errors.append("total_timeout_ns")
        if cfg.feedback_max_age_ns is None:
            config_errors.append("feedback_max_age_ns")
        if cfg.trajectory_max_age_ns is None:
            config_errors.append("trajectory_max_age_ns")
        if config_errors:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            reason = f"安全关键配置缺失，拒绝执行：{', '.join(config_errors)}"
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=reason,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=reason, timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 5. 反馈和轨迹新鲜度检查 ----
        if actual_joints.timestamp_ns > timestamp_ns:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            reason = "实际关节反馈时间来自未来，不可信"
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=reason,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=reason, timestamp_ns=timestamp_ns,
                ),
            )
        age_ns = timestamp_ns - actual_joints.timestamp_ns
        if age_ns > cfg.feedback_max_age_ns:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            reason = f"实际关节反馈过期（{age_ns}ns > {cfg.feedback_max_age_ns}ns）"
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=reason,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=reason, timestamp_ns=timestamp_ns,
                ),
            )
        if self._trajectory.timestamp_ns > timestamp_ns:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            reason = "轨迹规划时间来自未来，拒绝执行"
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=reason,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=reason, timestamp_ns=timestamp_ns,
                ),
            )
        traj_age_ns = timestamp_ns - self._trajectory.timestamp_ns
        if traj_age_ns > cfg.trajectory_max_age_ns:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            reason = f"轨迹规划时间过期（{traj_age_ns}ns > {cfg.trajectory_max_age_ns}ns）"
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=reason,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=reason, timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 6. 时间单调性检查 ----
        if self._last_step_ns is not None and timestamp_ns <= self._last_step_ns:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            reason = "时间戳非严格递增，拒绝本周期"
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=reason,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=reason, timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 7. 首次计时 + 隐式起点 ----
        if self._trajectory_started_ns is None:
            self._trajectory_started_ns = timestamp_ns
            self._implicit_start_position = actual_joints.position
        elapsed_ns = timestamp_ns - self._trajectory_started_ns
        elapsed_s = elapsed_ns / 1e9

        # ---- 8. 总超时检查 ----
        if elapsed_ns > cfg.total_timeout_ns:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            reason = f"轨迹执行总超时（{elapsed_ns}ns > {cfg.total_timeout_ns}ns）"
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=reason,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=reason, timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 9. 路点查找（策略1） + mask 切换检查（问题8） ----
        waypoints = self._trajectory.waypoints
        first_wp_time = waypoints[0].time_from_start_s
        prev_wp = waypoints[-1]
        next_wp = waypoints[-1]
        alpha = 0.0
        found = False
        prev_mask = waypoints[0].controlled_mask
        for i in range(len(waypoints)):
            if i > 0 and waypoints[i].controlled_mask != prev_mask:
                self.local_phase = LocalPhase.FAILED
                self._trajectory = None
                reason = f"路点{i - 1}和{i}的controlled_mask不一致，规则未确认，拒绝执行"
                return (
                    ManipulationCommand(
                        joint_target=actual_joints.position,
                        controlled_mask=(True,) * 17,
                        local_phase=self.local_phase,
                        timestamp_ns=timestamp_ns,
                        valid_until_ns=timestamp_ns,
                        valid=False,
                        failure_reason=reason,
                    ),
                    ManipulationStatus(
                        local_phase=self.local_phase,
                        state="FAILED", progress=0.0,
                        max_joint_error=float("inf"), success=False,
                        failure_reason=reason, timestamp_ns=timestamp_ns,
                    ),
                )
            prev_mask = waypoints[i].controlled_mask
            if elapsed_s < waypoints[i].time_from_start_s:
                if i == 0:
                    prev_wp = waypoints[0]
                    next_wp = waypoints[0]
                else:
                    self._waypoint_index = i - 1
                    prev_wp = waypoints[i - 1]
                    next_wp = waypoints[i]
                    dt = next_wp.time_from_start_s - prev_wp.time_from_start_s
                    if dt <= 0.0:
                        self.local_phase = LocalPhase.FAILED
                        self._trajectory = None
                        reason = f"路点{i - 1}和{i}时间差非正"
                        return (
                            ManipulationCommand(
                                joint_target=actual_joints.position,
                                controlled_mask=(True,) * 17,
                                local_phase=self.local_phase,
                                timestamp_ns=timestamp_ns,
                                valid_until_ns=timestamp_ns,
                                valid=False,
                                failure_reason=reason,
                            ),
                            ManipulationStatus(
                                local_phase=self.local_phase,
                                state="FAILED", progress=0.0,
                                max_joint_error=float("inf"), success=False,
                                failure_reason=reason, timestamp_ns=timestamp_ns,
                            ),
                        )
                    alpha = (elapsed_s - prev_wp.time_from_start_s) / dt
                found = True
                break
        if not found:
            prev_wp = waypoints[-1]
            next_wp = waypoints[-1]

        # ---- 10. 隐式起点插值（首路点保护：问题2 + gap A） ----
        if self._implicit_start_position is not None and elapsed_s < first_wp_time and first_wp_time > 0.0:
            _IMPLICIT_RAMP_S = 0.05  # 待队长确认
            implicit_alpha = min(elapsed_s / _IMPLICIT_RAMP_S, 1.0)
            interpolated = tuple(
                self._implicit_start_position[i]
                + implicit_alpha * (waypoints[0].joint_position[i] - self._implicit_start_position[i])
                for i in range(17)
            )
        elif found or elapsed_s >= first_wp_time:
            if alpha < 0.0:
                alpha = 0.0
            elif alpha > 1.0:
                alpha = 1.0
            interpolated = tuple(
                prev_wp.joint_position[i] + alpha * (next_wp.joint_position[i] - prev_wp.joint_position[i])
                for i in range(17)
            )
        else:
            interpolated = self._implicit_start_position if self._implicit_start_position is not None else waypoints[0].joint_position

        # ---- 11. controlled_mask（策略2） + 全 False 拒绝（问题6） ----
        mask = prev_wp.controlled_mask
        if not any(mask):
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            reason = "轨迹所有关节均不受控（controlled_mask 全 False），拒绝执行"
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=reason,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=reason, timestamp_ns=timestamp_ns,
                ),
            )
        raw_target = tuple(
            interpolated[i] if mask[i] else actual_joints.position[i]
            for i in range(17)
        )

        # ---- 12. 关节限速（策略3）—— 命令用限速后 target，到位/误差用 raw_target ----
        target = raw_target
        if self._last_step_ns is not None:
            dt_s = (timestamp_ns - self._last_step_ns) / 1e9
            if dt_s > 0.0:
                target_list = list(target)
                for i in range(17):
                    max_delta = cfg.max_joint_velocity_17[i] * dt_s
                    ai = actual_joints.position[i]
                    ti = target_list[i]
                    if ti < ai - max_delta:
                        target_list[i] = ai - max_delta
                    elif ti > ai + max_delta:
                        target_list[i] = ai + max_delta
                target = tuple(target_list)

        # ---- 13. 构造候选命令 ----
        target_error = _finite_vector_error(target, 17, "插值/限速后的目标")
        if target_error:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            return (
                ManipulationCommand(
                    joint_target=actual_joints.position,
                    controlled_mask=(True,) * 17,
                    local_phase=self.local_phase,
                    timestamp_ns=timestamp_ns,
                    valid_until_ns=timestamp_ns,
                    valid=False,
                    failure_reason=target_error,
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED", progress=0.0,
                    max_joint_error=float("inf"), success=False,
                    failure_reason=target_error, timestamp_ns=timestamp_ns,
                ),
            )

        ttl = cfg.command_ttl_ns if cfg.command_ttl_ns is not None else 200_000_000
        command = ManipulationCommand(
            joint_target=target,
            controlled_mask=mask,
            local_phase=self.local_phase,
            timestamp_ns=timestamp_ns,
            valid_until_ns=timestamp_ns + ttl,
            valid=True,
        )

        # ---- 14. 实际误差计算（基于 raw_target） ----
        controlled_errors = [
            abs(raw_target[i] - actual_joints.position[i])
            for i in range(17) if mask[i]
        ]
        max_err = max(controlled_errors) if controlled_errors else 0.0

        # ---- 15. 到位判断（基于最终路点位置，不是本周期 raw_target） ----
        trajectory_ended = elapsed_s >= waypoints[-1].time_from_start_s
        final_wp = waypoints[-1]
        arrived = False
        if cfg.joint_tolerance_17 is not None:
            arrived = all(
                abs(final_wp.joint_position[i] - actual_joints.position[i]) <= cfg.joint_tolerance_17[i]
                for i in range(17) if mask[i]
            )

        # ---- 16. 稳定性确认（fail closed） ----
        if cfg.settle_cycles is not None and cfg.joint_tolerance_17 is not None:
            if arrived:
                self._stable_cycle_count += 1
            else:
                self._stable_cycle_count = 0
            settled = self._stable_cycle_count >= cfg.settle_cycles
        else:
            settled = False

        # ---- 17. 完成判断 + 终态锁定 ----
        if trajectory_ended and arrived and settled:
            success = True
            state = "COMPLETED"
            progress = 1.0
            failure_reason = ""
            self._completed = True
        else:
            success = False
            state = "RUNNING"
            final_time = waypoints[-1].time_from_start_s
            progress = min(elapsed_s / final_time, 1.0) if final_time > 0.0 else 1.0
            if trajectory_ended and not arrived:
                failure_reason = "轨迹时间已结束但实际关节未收敛或到位容差未配置"
            else:
                failure_reason = ""

        status = ManipulationStatus(
            local_phase=self.local_phase,
            state=state,
            progress=progress,
            max_joint_error=max_err,
            success=success,
            failure_reason=failure_reason,
            timestamp_ns=timestamp_ns,
        )

        self._last_step_ns = timestamp_ns
        return command, status