"""19 维最终动作仲裁与安全总闸门。

``ActionMux`` 可以理解为机器人动作的“最终总闸门”：导航和机械臂执行模块都只能提交
候选动作，不能各自决定最终输出。它在每个控制周期接收：

- ``navigation.py`` 产生的 ``BaseCommand``；
- ``arm_execution.py`` 产生的 ``ManipulationCommand``；
- 由实际 ``/joint_states`` 转换得到的 ``RobotJointState``；
- ``GlobalFSM`` 产生的 ``FSMStatus``；
- 当前时间 ``now_ns``。

核心关系如下，其中箭头表示数据流，不表示对应动作已经执行：

::

    Navigation
    → BaseCommand ─────────────┐
                                │
    ArmExecution                ↓
    → ManipulationCommand → ActionMux
                                ↑
    RobotJointState + FSMStatus + now_ns
                                ↓
                         FinalAction[19]
                                ↓
                  OfficialCommandPublisher
                                ↓
                           官方 Server

输出是本周期唯一的 ``FinalAction``。其 ``values`` 固定 19 维：索引 0、1 分别是底盘
线速度 v（m/s）和角速度 w（rad/s），索引 2～18 是严格按照
``interfaces.JOINT_NAMES`` 对应顺序排列的 17 维非底盘关节目标。该顺序还必须同时匹配
``RobotJointState.position``、``ManipulationCommand.joint_target/controlled_mask`` 以及
``ActionMuxConfig.joint_lower/joint_upper``；业务文件不能重新定义第二套顺序。

本模块负责候选命令仲裁、TTL（命令有效期）检查、有限数检查、底盘速度限幅、受控关节
目标限幅、未受控关节保持、FSM 停止阶段覆盖、失败原因汇总和唯一 ``FinalAction``
生成。它不负责导航路径或底盘控制算法、IK、轨迹生成/插值、执行状态与抓取验证、FSM
状态推进、裁判结果判断、ROS2 发布、将 19 维拆成官方五组话题或 Server 通信。

``OfficialCommandPublisher`` 接收同一个 ``FinalAction``，检查 ``valid`` 后按固定顺序
拆分并发布。后续仍须结合 ``RobotJointState`` 和 Odom 反馈确认实际运动，因此：

``FinalAction`` 已生成 ≠ 已经发布 ≠ Server 已经接收 ≠ 机器人已经执行。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional

from .interfaces import (
    BaseCommand,
    FinalAction,
    FSMStatus,
    GlobalPhase,
    ManipulationCommand,
    RobotJointState,
)


# 动作安全配置
@dataclass(frozen=True)
class ActionMuxConfig:
    """最终动作出口使用的速度和关节安全边界。

    ``max_abs_base_v`` 是底盘线速度绝对值上限，单位 m/s；``max_abs_base_w`` 是底盘
    角速度绝对值上限，单位 rad/s。``joint_lower`` 和 ``joint_upper`` 各有 17 项，并且
    必须严格对应 ``interfaces.JOINT_NAMES``：slide 使用米，旋转关节使用弧度，夹爪
    使用官方控制范围。

    这些配置只是最终输出前的最后一道安全约束，不能代替机械臂规划、轨迹生成或执行
    反馈判断。长度、类型、有限性或上下界关系不合法时，构造配置会抛出 ``ValueError``，
    防止错误边界进入高频控制循环。
    """

    max_abs_base_v: float
    max_abs_base_w: float
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]

    # 配置合法性检查
    def __post_init__(self) -> None:
        """拒绝不能可靠参与安全比较的配置值。

        17 维长度用于保证索引与统一关节顺序一致；``bool`` 在 Python 中虽然可当作
        0 或 1，却不能混入安全边界，字符串也不能作为边界。所有数值还必须有限，速度
        绝对值上限不能为负数，每个关节的下界不能大于上界。
        """

        if len(self.joint_lower) != 17 or len(self.joint_upper) != 17:
            raise ValueError("ActionMux 关节上下界必须各有 17 项")
        values = (self.max_abs_base_v, self.max_abs_base_w, *self.joint_lower, *self.joint_upper)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError("ActionMux 安全边界必须全部为真实数值，不能使用 bool 或字符串")
        try:
            all_finite = all(math.isfinite(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ActionMux 安全边界必须全部为有限数") from exc
        if not all_finite:
            raise ValueError("ActionMux 安全边界必须全部为有限数")
        if self.max_abs_base_v < 0.0 or self.max_abs_base_w < 0.0:
            raise ValueError("底盘速度上限不能为负数")
        if any(lower > upper for lower, upper in zip(self.joint_lower, self.joint_upper)):
            raise ValueError("关节下界不能大于上界")

    # 默认保守安全边界
    @classmethod
    def conservative_defaults(cls) -> "ActionMuxConfig":
        """构造当前第一版保守默认边界，避免缺少配置时完全失去边界保护。

        这些数值不是官方 MMK2 最终限位，也不能因为函数名包含 ``defaults`` 就被视为
        比赛规则。正式仿真前必须与模型、运行配置和官方控制接口逐项核对：17 维关节
        顺序、slide 上下限、头部关节限位、左右臂各关节限位、夹爪控制范围，以及底盘
        最大安全线速度和角速度。
        """

        arm_lower = (-3.14, -2.50, -3.14, -2.60, -3.14, -2.60)
        arm_upper = (3.14, 2.50, 3.14, 2.60, 3.14, 2.60)
        return cls(
            max_abs_base_v=0.25,
            max_abs_base_w=0.50,
            joint_lower=(-0.04, -0.50, -1.18, *arm_lower, 0.0, *arm_lower, 0.0),
            joint_upper=(0.87, 0.50, 0.16, *arm_upper, 1.0, *arm_upper, 1.0),
        )


# 最终动作仲裁器
class ActionMux:
    """把两类动作申请审核为本控制周期唯一的 ``FinalAction``。

    ``BaseCommand`` 和 ``ManipulationCommand`` 像两个模块提交的动作申请；本类结合
    实际关节反馈、FSM 阶段、TTL 和配置边界决定最终输出。实际反馈无效时，仍可生成
    诊断用 ``FinalAction``，但其 ``valid=False``，不能进入正式发布链路。

    当前五个强制停止阶段的含义是：

    - ``WAIT_READY``：系统尚未准备好，不允许执行普通动作；
    - ``LOAD_TASK``：任务尚未完成装载，不允许提前运动；
    - ``DONE``：客户端流程已经结束，迟到命令不能让机器人重新运动；
    - ``SAFE_HOLD``：临时安全保持，等待安全条件恢复；
    - ``FAILED``：客户端流程已经失败，只允许安全停止与保持。

    这些阶段都把底盘 v/w 强制为零、忽略普通候选并保持实际关节位置。停止阶段覆盖
    普通命令不等于当前任务获得裁判成功。

    当前待确认：完整 FSM 阶段权限矩阵；底盘运动时是否允许机械臂同时运动；正式底盘
    速度上限和 17 维关节限位；Server watchdog 与正式发布频率；官方模型中的关节名称
    和顺序。本类不对这些尚未确认的协议作额外推断。
    """

    # 强制停止阶段
    _STOP_PHASES = {
        GlobalPhase.WAIT_READY,
        GlobalPhase.LOAD_TASK,
        GlobalPhase.DONE,
        GlobalPhase.SAFE_HOLD,
        GlobalPhase.FAILED,
    }

    def __init__(self, config: Optional[ActionMuxConfig] = None) -> None:
        """创建仲裁器并从零开始记录本实例生成的动作序号。

        ``config`` 缺省时使用第一版保守边界；内部 ``sequence`` 只是区分本实例各控制
        周期的输出，不能证明 Server 已接收，也不能代替 ROS 时间戳。
        """

        self.config = config or ActionMuxConfig.conservative_defaults()
        self._sequence = 0

    def compose(
        self,
        base_command: Optional[BaseCommand],
        manipulation_command: Optional[ManipulationCommand],
        actual_joints: RobotJointState,
        fsm_status: FSMStatus,
        now_ns: int,
    ) -> FinalAction:
        """合成本控制周期唯一的固定 19 维最终动作。

        五个输入分别承担不同职责：

        - ``base_command``：来自 ``navigation.py`` 的短时底盘速度候选，不是实际 Odom。
          无效、过期、格式错误或遇到停止阶段时可能被拒绝；合法但越界时会被限幅。
        - ``manipulation_command``：来自 ``arm_execution.py`` 的本周期 17 维关节候选，
          不是完整 ``JointTrajectory``。``controlled_mask`` 决定哪些索引允许采用候选值。
        - ``actual_joints``：来自实际 ``/joint_states``，是未受控关节保持的依据，不能用
          17 个全零代替。反馈无效时忽略普通候选，输出只能作为 ``valid=False`` 的诊断
          快照。
        - ``fsm_status``：客户端流程阶段快照，用于停止阶段覆盖；它不是裁判得分。
        - ``now_ns``：当前纳秒时间，用于 TTL 判断。TTL 像命令的保质期，当
          ``now_ns >= valid_until_ns`` 时候选已经过期。

        底盘候选过期后 v/w 归零，不能复用上次速度；机械臂候选过期后保持实际关节，
        不能把 17 维目标清零。TTL 只防止旧命令持续控制，不表示动作已经完成。

        ``controlled_mask`` 固定 17 项且严格对应 ``JOINT_NAMES``，像 17 个独立开关：
        ``True`` 采用本周期候选目标，``False`` 保持实际位置，但不表示该关节已经到达。
        例如实际位置 ``[0.1, 0.2, 0.3, ...]``、候选 ``[0.5, 0.6, 0.7, ...]``、mask
        为 ``[True, False, True, ...]`` 时，关节输出为 ``[0.5, 0.2, 0.7, ...]``。

        返回值 ``FinalAction.values`` 的索引 0、1 为 base_v/base_w，索引 2～18 为统一
        顺序的 17 维关节目标。``valid`` 表示是否满足进入发布链路的条件：实际反馈无效
        或未受控实际值越过配置边界时为 ``False``。``clipped`` 表示至少一个合法底盘或
        受控关节候选因越界被调整；它不必然使动作无效，也不表示机器人已经到达目标。
        ``failure_reason`` 可同时保存实际反馈问题和候选缺失、无效、过期等安全降级原因，
        所以 ``valid=True`` 的零底盘/关节保持动作也可能带有原因说明。

        每次调用都会递增 ``sequence``，包括正常动作、安全保持和无效诊断动作；该序号
        只区分本实例产生的不同 ``FinalAction``，不能证明发布、接收或实际执行。
        """

        reasons: list[str] = []
        clipped = False
        output_valid = actual_joints.valid
        stop_phase = fsm_status.global_phase in self._STOP_PHASES

        # 实际反馈问题优先保留，后续候选或 FSM 原因只能追加，不能覆盖它。
        if actual_joints.failure_reason:
            reasons.append(actual_joints.failure_reason)
        elif not actual_joints.valid:
            reasons.append("实际关节反馈无效")
        if stop_phase:
            # 停止阶段是总闸门覆盖：不论候选多新鲜，都只能零底盘并保持实际姿态。
            reasons.append(
                f"FSM 停止阶段 {fsm_status.global_phase.value} 覆盖普通底盘和机械臂命令；"
                "底盘归零并保持实际关节位置"
            )
            if fsm_status.failure_reason:
                reasons.append(fsm_status.failure_reason)
        elif not actual_joints.valid:
            reasons.append("实际关节反馈无效，忽略普通底盘和机械臂候选命令")

        # 底盘候选命令处理
        base_v = 0.0
        base_w = 0.0
        # TTL 防止旧速度持续生效；到达失效时刻就必须归零，不能复用上周期速度。
        if not stop_phase and actual_joints.valid:
            if base_command is None:
                reasons.append("无底盘候选命令，输出零速度")
            elif not base_command.valid:
                reasons.append(base_command.failure_reason or "底盘候选命令无效")
            elif now_ns >= base_command.valid_until_ns:
                reasons.append("底盘候选命令已过期，输出零速度")
            else:
                try:
                    if isinstance(base_command.v, bool) or isinstance(base_command.w, bool):
                        raise ValueError("bool 不能作为速度")
                    candidate_v = float(base_command.v)
                    candidate_w = float(base_command.w)
                except (TypeError, ValueError, OverflowError) as exc:
                    reasons.append(f"底盘候选速度格式无效，输出零速度：{exc}")
                else:
                    # 非有限速度属于非法输入并归零；有限但超界的速度才进入限幅流程。
                    if not math.isfinite(candidate_v) or not math.isfinite(candidate_w):
                        reasons.append("底盘候选速度包含 NaN 或 Inf，输出零速度")
                    else:
                        base_v, was_clipped = self._clip(
                            candidate_v, -self.config.max_abs_base_v, self.config.max_abs_base_v
                        )
                        clipped |= was_clipped
                        base_w, was_clipped = self._clip(
                            candidate_w, -self.config.max_abs_base_w, self.config.max_abs_base_w
                        )
                        clipped |= was_clipped

        # 机械臂候选命令处理
        # 后 17 项先复制实际反馈，避免缺失、无效或过期命令把“保持”错误写成全零姿态。
        joint_targets = list(actual_joints.position)
        controlled_mask = [False] * 17
        if not stop_phase and actual_joints.valid:
            if manipulation_command is None:
                reasons.append("无机械臂候选命令，保持实际关节位置")
            elif not manipulation_command.valid:
                reasons.append(manipulation_command.failure_reason or "机械臂候选命令无效")
            elif now_ns >= manipulation_command.valid_until_ns:
                reasons.append("机械臂候选命令已过期，保持实际关节位置")
            else:
                for index, controlled in enumerate(manipulation_command.controlled_mask):
                    if controlled:
                        controlled_mask[index] = True
                        joint_targets[index] = manipulation_command.joint_target[index]

        # 实际关节保持与越界检查
        # mask=True 才限幅候选；mask=False 必须原样保持实际值，不能借“限幅”主动移动。
        for index, target in enumerate(joint_targets):
            if controlled_mask[index]:
                joint_targets[index], was_clipped = self._clip(
                    float(target), self.config.joint_lower[index], self.config.joint_upper[index]
                )
                clipped |= was_clipped
            elif target < self.config.joint_lower[index] or target > self.config.joint_upper[index]:
                # 实际保持值超界时不能静默移动到边界；保留快照并阻止官方发布。
                output_valid = False
                reasons.append(
                    f"未受控的实际关节第 {index} 项超出配置边界 "
                    f"[{self.config.joint_lower[index]}, {self.config.joint_upper[index]}]；"
                    "保持实际反馈值"
                )

        # 19维最终动作组装
        # 固定顺序是 base_v、base_w，再接严格复用 JOINT_NAMES 语义的 17 维关节目标。
        values = (base_v, base_w, *joint_targets)
        # 正常、保持和无效诊断输出都占一个控制周期；序号不代表已发布或已执行。
        self._sequence += 1
        return FinalAction(
            values=values,
            sequence=self._sequence,
            timestamp_ns=int(now_ns),
            global_phase=fsm_status.global_phase,
            local_phase=fsm_status.local_phase,
            valid=output_valid,
            clipped=clipped,
            failure_reason="；".join(reasons),
        )

    # 通用限幅工具
    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> tuple[float, bool]:
        """把一个已确认有限的合法候选限制到闭区间，并报告数值是否被调整。

        本工具只处理候选目标；实际关节保持值不会为了落入配置区间而调用它。
        """

        clipped_value = min(max(value, lower), upper)
        return clipped_value, clipped_value != value
