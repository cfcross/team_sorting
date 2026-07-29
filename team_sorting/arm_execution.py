"""机械臂轨迹执行和局部抓放状态机骨架。

本文件位于“轨迹计划”和“最终控制仲裁”之间，完整数据流是：

``JointTrajectory``
→ ``ArmExecutionController.start_trajectory`` 装载并校验
→ ``step``（待机械臂2实现）按实际 ``RobotJointState`` 插值和判断进度
→ ``ManipulationCommand`` 候选关节目标 + ``ManipulationStatus`` 执行状态
→ ``ActionMux`` 结合实际反馈、TTL和FSM安全阶段生成 ``FinalAction[19]``
→ ``OfficialCommandPublisher`` 拆分并发布官方关节话题。

``ManipulationCommand`` 只是本周期候选建议，不是已经发布或已经执行的动作；到位、试抬
和抓放验证必须依据实际反馈，不能只看目标命令或等待时间。本文件不负责IK、抓取位姿
规划、ROS2发布、全局FSM推进或视觉算法，也不能绕过 ``ActionMux``。

当前骨架只实现轨迹入口校验和基于实际反馈的安全保持。完整插值、限速、局部抓放阶段、
试抬验证与恢复仍由机械臂2负责人实现；未实现部分继续明确抛出
``NotImplementedError``，不会返回伪成功。
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


class ArmExecutionController:
    """双臂轨迹执行器和局部状态机骨架。

    输入为17维实际关节、目标轨迹和纳秒时间，输出17维候选命令及执行状态。slide单位
    米、旋转关节弧度、夹爪为官方0～1控制量。``local_phase`` 是执行器内部当前阶段：
    ``MOVE_PREGRASP`` 到预抓姿态，``HUG_OPEN`` 张开双夹爪，``APPROACH`` 靠近目标，
    ``HUG_CLOSE`` 合拢夹持，``TEST_LIFT`` 小幅试抬，``VERIFY`` 等待验证，
    ``RETREAT`` 撤离，``TRANSPORT_HOLD`` 运输保持，``MOVE_PREPLACE`` 到预放姿态，
    ``LOWER_OBJECT`` 下放，``RELEASE`` 释放，``STOW`` 收回，``FAILED`` 表示局部执行失败。

    这些枚举值存在不等于流程已经实现。当前 ``JointTrajectory`` 没有标明“抓取/放置”
    或起始局部阶段，所以仅装载轨迹时保持中性的 ``IDLE``，不能猜成
    ``MOVE_PREGRASP``。完整阶段推进仍由 ``step`` 的后续实现负责。

    """

    def __init__(self) -> None:
        """创建处于 IDLE 的局部执行器。

        参数和返回值均无；构造不依赖 ROS2 或官方 KDL。内部尚未装载任何轨迹。
        """

        self.local_phase = LocalPhase.IDLE
        self._trajectory: JointTrajectory | None = None
        # 这些字段为未来step实现预留统一清理点，拒绝新轨迹时不能沿用上一条的进度。
        self._waypoint_index = 0
        self._trajectory_started_ns: int | None = None
        self._stable_cycle_count = 0
        self._cached_verification = None

    def create_hold_command(
        self, actual_joints: RobotJointState, timestamp_ns: int, valid_for_ns: int
    ) -> ManipulationCommand:
        """用实际反馈创建全关节安全保持命令。

        参数：17维实际关节、非负整数 ``timestamp_ns`` 和正整数 ``valid_for_ns``；二者
        单位都是纳秒。TTL像候选命令的保质期，过期后由 ``ActionMux`` 不再采用。

        实际反馈有效时，返回“目标严格等于当前实际位置、17项均受控”的保持候选；这表示
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

        # === 参数开关：队长确认后改为 True 并填入对应数值 ===
        _VELOCITY_LIMIT_ENABLED = False     # 策略3：关节限速，依赖 max_joint_velocity_17
        _ARRIVAL_CHECK_ENABLED = False      # 策略4：基于实际反馈的到位判断，依赖 joint_tolerance_17
        _STABILITY_CHECK_ENABLED = False    # 策略5：稳定性确认，依赖 settle_cycles
        # === 未确认参数：队长确认后填入数值并删除 None 占位 ===
        _MAX_JOINT_VELOCITY_17 = None       # 【未确认】17维最大速度
        _JOINT_TOLERANCE_17 = None          # 【未确认】17维到位容差
        _SETTLE_CYCLES = None               # 【未确认】连续稳定周期阈值
        _TOTAL_TIMEOUT_NS = None            # 【未确认】轨迹总超时纳秒
        _COMMAND_TTL_NS = None               # 【未确认】候选命令有效期纳秒
        # === 以上开关为 False 时，对应策略跳过，使用简化行为 ===

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
                    state="FAILED",
                    progress=0.0,
                    max_joint_error=float("inf"),
                    success=False,
                    failure_reason=reason,
                    timestamp_ns=timestamp_ns,
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
                    state="FAILED",
                    progress=0.0,
                    max_joint_error=float("inf"),
                    success=False,
                    failure_reason=position_error,
                    timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 2. 先检查 FAILED（必须在检查 _trajectory 之前） ----
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
                    state="FAILED",
                    progress=0.0,
                    max_joint_error=float("inf"),
                    success=False,
                    failure_reason="局部执行已处于FAILED状态",
                    timestamp_ns=timestamp_ns,
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
                    state="IDLE",
                    progress=0.0,
                    max_joint_error=float("inf"),
                    success=False,
                    failure_reason="尚未装载有效轨迹",
                    timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 4. 时间基准 ----
        if self._trajectory_started_ns is not None and timestamp_ns < self._trajectory_started_ns:
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
                    failure_reason="时间戳倒退，拒绝本周期",
                ),
                ManipulationStatus(
                    local_phase=self.local_phase,
                    state="FAILED",
                    progress=0.0,
                    max_joint_error=float("inf"),
                    success=False,
                    failure_reason="时间戳倒退，拒绝本周期",
                    timestamp_ns=timestamp_ns,
                ),
            )
        if self._trajectory_started_ns is None:
            self._trajectory_started_ns = timestamp_ns
            self._last_step_ns = timestamp_ns  # type: ignore[attr-defined]

        elapsed_ns = timestamp_ns - self._trajectory_started_ns
        elapsed_s = elapsed_ns / 1e9

        # ---- 5. 总超时检查（参数未确认时跳过） ----
        if _TOTAL_TIMEOUT_NS is not None and elapsed_ns > _TOTAL_TIMEOUT_NS:
            self.local_phase = LocalPhase.FAILED
            self._trajectory = None
            reason = f"轨迹执行总超时（{elapsed_ns}ns > {_TOTAL_TIMEOUT_NS}ns）"
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
                    state="FAILED",
                    progress=0.0,
                    max_joint_error=float("inf"),
                    success=False,
                    failure_reason=reason,
                    timestamp_ns=timestamp_ns,
                ),
            )

        # ---- 6. 路点查找（策略1：轨迹时间插值） ----
        waypoints = self._trajectory.waypoints
        prev_wp = waypoints[-1]
        next_wp = waypoints[-1]
        alpha = 0.0
        found = False
        for i in range(len(waypoints)):
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
                                state="FAILED",
                                progress=0.0,
                                max_joint_error=float("inf"),
                                success=False,
                                failure_reason=reason,
                                timestamp_ns=timestamp_ns,
                            ),
                        )
                    alpha = (elapsed_s - prev_wp.time_from_start_s) / dt
                found = True
                break
        if not found:
            prev_wp = waypoints[-1]
            next_wp = waypoints[-1]

        # ---- 7. 线性插值（策略1续） ----
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        interpolated = tuple(
            prev_wp.joint_position[i] + alpha * (next_wp.joint_position[i] - prev_wp.joint_position[i])
            for i in range(17)
        )

        # ---- 8. controlled_mask（策略2） ----
        # TODO: 相邻路点mask不同时的规则待确认（mask_transition_rule），当前保守使用前一个路点的mask
        mask = prev_wp.controlled_mask
        target = tuple(
            interpolated[i] if mask[i] else actual_joints.position[i]
            for i in range(17)
        )

        # ---- 9. 关节限速（策略3） ----
        if _VELOCITY_LIMIT_ENABLED:
            last_ns = getattr(self, "_last_step_ns", None)  # type: ignore[attr-defined]
            if last_ns is not None:
                dt_s = (timestamp_ns - last_ns) / 1e9
            else:
                dt_s = 0.0
            if dt_s > 0.0 and _MAX_JOINT_VELOCITY_17 is not None:
                target_list = list(target)
                for i in range(17):
                    max_delta = _MAX_JOINT_VELOCITY_17[i] * dt_s
                    ai = actual_joints.position[i]
                    ti = target_list[i]
                    if ti < ai - max_delta:
                        target_list[i] = ai - max_delta
                    elif ti > ai + max_delta:
                        target_list[i] = ai + max_delta
                target = tuple(target_list)
        # _VELOCITY_LIMIT_ENABLED 为 False：策略2结果直接作为最终目标，不做截断
        # TODO: 等 _VELOCITY_LIMIT_ENABLED 和 _MAX_JOINT_VELOCITY_17 确认后启用

        # ---- 10. 构造候选命令 ----
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
                    state="FAILED",
                    progress=0.0,
                    max_joint_error=float("inf"),
                    success=False,
                    failure_reason=target_error,
                    timestamp_ns=timestamp_ns,
                ),
            )

        command = ManipulationCommand(
            joint_target=target,
            controlled_mask=mask,
            local_phase=self.local_phase,
            timestamp_ns=timestamp_ns,
            valid_until_ns=timestamp_ns + (_COMMAND_TTL_NS if _COMMAND_TTL_NS is not None else 200_000_000),
            valid=True,
        )

        # ---- 11. 实际误差计算 ----
        controlled_errors = [
            abs(target[i] - actual_joints.position[i])
            for i in range(17)
            if mask[i]
        ]
        max_err = max(controlled_errors) if controlled_errors else 0.0

        # ---- 12. 到位判断（策略4） ----
        trajectory_ended = elapsed_s >= waypoints[-1].time_from_start_s
        arrived = False
        if _ARRIVAL_CHECK_ENABLED:
            if _JOINT_TOLERANCE_17 is not None:
                arrived = all(
                    abs(target[i] - actual_joints.position[i]) <= _JOINT_TOLERANCE_17[i]
                    for i in range(17)
                    if mask[i]
                )
        else:
            # TODO：【临时行为，等 _ARRIVAL_CHECK_ENABLED 和 joint_tolerance_17 确认后替换】
            # 时间到了就认为到位，不检查实际反馈
            arrived = trajectory_ended

        # ---- 13. 稳定性确认（策略5） ----
        if _STABILITY_CHECK_ENABLED:
            if arrived:
                self._stable_cycle_count += 1
            else:
                self._stable_cycle_count = 0
            settled = (
                _SETTLE_CYCLES is not None
                and self._stable_cycle_count >= _SETTLE_CYCLES
            )
        else:
            settled = True  # 跳过稳定确认，到位即成功

        # ---- 14. 完成判断（策略6） ----
        if trajectory_ended and arrived and settled:
            success = True
            state = "COMPLETED"
            progress = 1.0
            failure_reason = ""
        else:
            success = False
            state = "RUNNING"
            final_time = waypoints[-1].time_from_start_s
            progress = min(elapsed_s / final_time, 1.0) if final_time > 0.0 else 1.0
            if trajectory_ended and not arrived:
                failure_reason = "轨迹时间已结束但实际关节未收敛"
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

        self._last_step_ns = timestamp_ns  # type: ignore[attr-defined]
        return command, status