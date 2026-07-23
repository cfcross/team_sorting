"""比赛任务解析和全局有限状态机（FSM）骨架。

FSM 可以理解为任务流程的指挥员：``phase`` 像流程表中当前高亮的一行，记录客户端
走到了哪一步；它不计算导航路线、物体三维位置、IK（逆运动学）或关节动作。
``InstructionParser`` 把原始任务 JSON 统一转换成 ``TaskSpec``；``FSMEvent`` 是感知、
导航、规划、执行或验证模块确认完成后提交的业务回执；``GlobalFSM`` 检查当前阶段是否
允许接收该回执；``FSMStatus`` 则是供控制与遥测读取的不可变状态快照。

核心关系如下：

``/material/instruction``
    → ``InstructionParser``
    → ``TaskSpec``
    → ``GlobalFSM``
    → ``FSMStatus``

真实业务反馈：
感知 / 导航 / 规划 / 执行 / 验证
                    ↓
                ``FSMEvent``

事件必须由真实反馈确认后提交，不能由 FSM 自行猜测成功。客户端进入 ``DONE`` 或
``FSMStatus.success=True`` 只表示客户端状态机认为流程完成，不自动等于裁判确认得分。
本文件不负责 ROS2 订阅发布，状态不带空间坐标，时间戳单位为纳秒。
"""

from __future__ import annotations

from enum import Enum
import json
import math
from typing import Any, Mapping, Optional

from .interfaces import FSMStatus, GlobalPhase, LocalPhase, TaskSpec


# 状态机事件

class FSMEvent(str, Enum):
    """其他业务模块提交给全局状态机的离散“完成回执”。

    事件不是 ROS 话题，也不是控制命令；它没有单位和坐标系。``GlobalFSM`` 只在当前
    阶段允许时接收它，顺序错误会返回 ``False``，不会越级切换状态。

    - 感知类：``TARGET_FOUND`` 应来自稳定且有效的 ``ObjectEstimate3D``；
      ``TARGET_REFINED`` 应来自靠近目标后重新定位得到的有效感知结果。
    - 导航类：``PICK_NAV_REACHED``、``PLACE_NAV_REACHED``、``RETURN_REACHED`` 必须
      来自基于实际 Odom 的 ``NavigationStatus``，不能因生成 ``NavGoal`` 或
      ``BaseCommand`` 就触发。
    - 规划类：``PICK_PLAN_READY``、``PLACE_PLAN_READY`` 表示 IK、轨迹等规划结果
      有效，而不只是“已经开始规划”。
    - 执行类：``PICK_EXECUTED``、``PLACE_EXECUTED`` 必须由 ``RobotJointState`` 等
      真实反馈确认，不能以命令已发布或轨迹时间结束代替。
    - 验证类：``PICK_VERIFIED``、``PICK_FAILED``、``PLACE_VERIFIED`` 必须来自明确
      的抓放验证结果。
    - 控制类：``SYSTEM_READY`` 表示客户端依赖已满足；``FAILURE`` 表示业务模块明确
      判断流程无法继续；``RESET`` 清空内存任务并重新等待。

    当前骨架状态：``ros_nodes`` 目前只接入 ``SYSTEM_READY``；感知、导航、规划、
    抓取、放置、返区、``FAILURE`` 和 ``RESET`` 事件尚未完成真实接线。事件名称存在
    不代表对应算法闭环已经完成。

    """
  # 系统依赖已经准备完成。
    # 例如：ROS2节点启动、必要话题收到数据、ActionMux可用。
    # 推动：WAIT_READY → LOAD_TASK
    SYSTEM_READY = "SYSTEM_READY"

    # 感知模块确认已经找到任务要求的目标物体。
    # 应基于稳定、有效、与任务匹配的 ObjectEstimate3D，
    # 不能只因为YOLO出现一个检测框就触发。
    # 推动：SEARCH_TARGET → NAV_TO_PICK
    TARGET_FOUND = "TARGET_FOUND"

    # 底盘导航模块根据实际Odom确认到达抓取区域。
    # 不能因为NavGoal已经生成或BaseCommand已经发布就触发。
    # 推动：NAV_TO_PICK → REFINE_TARGET
    PICK_NAV_REACHED = "PICK_NAV_REACHED"

    # 机器人靠近目标后，感知模块重新获得了更准确的三维目标位置。
    # 用于避免继续使用导航前的旧物体坐标。
    # 推动：REFINE_TARGET → PLAN_PICK
    TARGET_REFINED = "TARGET_REFINED"

    # 机械臂规划模块确认抓取规划已经准备好。
    # 一般表示：抓取位姿有效、IK成功、JointTrajectory有效。
    # 推动：PLAN_PICK → EXECUTE_PICK
    PICK_PLAN_READY = "PICK_PLAN_READY"

    # 机械臂执行模块确认抓取动作轨迹已经执行完成。
    # 应根据真实RobotJointState判断到位，
    # 不能只根据“轨迹已经发送”或“预计时间到了”判断。
    #
    # 注意：它只说明抓取动作执行完，不代表已经抓住物体。
    # 推动：EXECUTE_PICK → VERIFY_PICK
    PICK_EXECUTED = "PICK_EXECUTED"

    # 抓取验证模块确认物体已经被成功抓住。
    # 可以综合视觉、夹爪状态、关节effort和试抬结果。
    # 推动：VERIFY_PICK → NAV_TO_PLACE
    PICK_VERIFIED = "PICK_VERIFIED"

    # 抓取验证模块确认本次没有抓住物体。
    # FSM会根据剩余重试次数，决定重新精定位还是失败返区。
    #
    # 有重试次数：
    # VERIFY_PICK → REFINE_TARGET
    #
    # 重试耗尽：
    # VERIFY_PICK → RETURN_END → FAILED
    PICK_FAILED = "PICK_FAILED"

    # 底盘导航模块根据实际Odom确认到达放置区域。
    # 推动：NAV_TO_PLACE → PLAN_PLACE
    PLACE_NAV_REACHED = "PLACE_NAV_REACHED"

    # 机械臂规划模块确认放置规划已经准备好。
    # 一般表示：放置目标有效、IK成功、放置轨迹有效。
    # 推动：PLAN_PLACE → EXECUTE_PLACE
    PLACE_PLAN_READY = "PLACE_PLAN_READY"

    # 机械臂执行模块确认放置轨迹已经执行完成。
    # 例如机械臂到达释放位置并执行了夹爪释放动作。
    #
    # 注意：它只表示动作执行完，
    # 还不能证明物体最终放置正确。
    # 推动：EXECUTE_PLACE → VERIFY_PLACE
    PLACE_EXECUTED = "PLACE_EXECUTED"

    # 放置验证模块确认物体已经释放并正确放到目标区域。
    # 推动：VERIFY_PLACE → RETURN_END
    PLACE_VERIFIED = "PLACE_VERIFIED"

    # 底盘导航模块根据实际Odom确认机器人已返回结束区域。
    #
    # 正常任务：
    # RETURN_END → DONE
    #
    # 之前发生不可恢复失败：
    # RETURN_END → FAILED
    RETURN_REACHED = "RETURN_REACHED"

    # 某业务模块报告无法继续执行的严重失败。
    # 例如：传感器持续失效、导航无法恢复、IK持续无解、
    # 轨迹执行异常或关键控制模块故障。
    #
    # 已经装载任务时：
    # 当前阶段 → RETURN_END → FAILED
    #
    # 尚未装载任务时：
    # 当前阶段 → FAILED
    FAILURE = "FAILURE"

    # 清空当前任务、重试次数和失败原因，
    # 重新回到等待系统准备的状态。
    #
    # 推动：任意阶段 → WAIT_READY
    # DONE和FAILED也只能通过RESET重新开始。
    RESET = "RESET"


# 官方任务JSON解析

class InstructionParser:
    """官方任务 JSON 的唯一解析入口。

    原始 JSON 可以是一条任务、任务数组，或包含 ``tasks``、``instructions``、
    ``materials``、``data`` 外层键的对象；``parse`` 始终返回按输入顺序排列的
    ``TaskSpec`` 元组。字段别名用于兼容当前可能出现的格式，正式 schema 公布后以
    官方格式为准。

    全仓库只保留一个解析入口，是为了让字段别名、必填项和失败语义保持一致。必填字段
    缺失、结构或数值错误时直接抛出 ``ValueError``；成功构造 ``TaskSpec`` 只表示格式
    解析成功，不代表任务一定能够完成，也不能写成“所有失败都返回 ``valid=False``”。

    ``place_world_xyz`` 位于指令约定的 world 坐标系，单位米，表示物体中心目标，既
    不是底盘停车点，也不是夹爪末端位姿。``task_id`` 只接受整数或合法整数字符串；
    ``bool`` 不能冒充整数、浮点数或坐标，NaN/Inf 也不能进入任务接口。
    """

    def parse(self, raw_json: str, timestamp_ns: int) -> tuple[TaskSpec, ...]:
        """解析 ``/material/instruction`` 原始 JSON。

        ``raw_json`` 是消息原文，``timestamp_ns`` 是接收时间（纳秒）。顶层对象会按
        支持的外层键展开，单条任务会统一包装为序列，最终返回 ``TaskSpec`` 元组。

        原始 JSON 结构或字段错误 → 抛出 ``ValueError``；成功构造 ``TaskSpec`` →
        表示格式解析成功。放置坐标单位米，坐标系沿用官方指令约定。
        """

        try:
            payload = json.loads(raw_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("任务指令不是合法 JSON") from exc

        if isinstance(payload, Mapping):
            for key in ("tasks", "instructions", "materials", "data"):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("任务 JSON 顶层必须是对象或任务数组")

        tasks = tuple(self._parse_one(item, index, timestamp_ns) for index, item in enumerate(payload))
        if not tasks:
            raise ValueError("任务 JSON 中没有任务")
        return tasks

    # JSON字段读取与校验

    def _parse_one(self, item: Any, index: int, timestamp_ns: int) -> TaskSpec:
        if not isinstance(item, Mapping):
            raise ValueError(f"第 {index} 个任务不是 JSON 对象")

        task_id = self._integer(item, ("task", "task_id", "id", "index"), default=index)
        instruction = self._text(item, ("instruction", "text", "raw_instruction"), default="")
        target_kind = self._text(item, ("target_kind", "kind"), default="")
        target_body = self._text(item, ("target_body", "body", "object", "material"))
        target_color = self._text(item, ("target_color", "color"), default="")
        place_type = self._text(item, ("place_type", "target_place", "destination"))
        place_world = self._optional_xyz(item, ("place_world", "place_world_xyz", "position"))
        place_radius = self._optional_float(item, ("place_radius", "radius"))
        ref_prop = self._optional_text(item, ("ref_prop", "reference", "relative_to"))
        ref_prop_body = self._optional_text(item, ("ref_prop_body", "reference_body"))
        direction = self._optional_text(item, ("direction", "relative_direction"))
        return TaskSpec(
            task_id=task_id,
            instruction=instruction,
            target_kind=target_kind,
            target_body=target_body,
            target_color=target_color,
            place_type=place_type,
            place_world_xyz=place_world,
            place_radius=place_radius,
            ref_prop=ref_prop,
            ref_prop_body=ref_prop_body,
            direction=direction,
            timestamp_ns=int(timestamp_ns),
        )

    @staticmethod
    def _find(item: Mapping[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
        for key in keys:
            if key in item:
                return item[key]
        return default

    def _text(
        self,
        item: Mapping[str, Any],
        keys: tuple[str, ...],
        default: Optional[str] = None,
    ) -> str:
        value = self._find(item, keys, default)
        if not isinstance(value, str) or (default is None and not value.strip()):
            raise ValueError(f"任务缺少必需文本字段：{'/'.join(keys)}")
        return value.strip()

    def _optional_text(self, item: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[str]:
        value = self._find(item, keys)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"字段 {'/'.join(keys)} 必须是字符串")
        return value.strip() or None

    def _integer(
        self,
        item: Mapping[str, Any],
        keys: tuple[str, ...],
        default: int,
    ) -> int:
        value = self._find(item, keys, default)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                return int(value.strip(), 10)
            except ValueError as exc:
                raise ValueError(f"字段 {'/'.join(keys)} 必须是整数") from exc
        raise ValueError(f"字段 {'/'.join(keys)} 必须是整数")

    def _optional_float(self, item: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[float]:
        value = self._find(item, keys)
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"字段 {'/'.join(keys)} 必须是数值")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"字段 {'/'.join(keys)} 必须是数值") from exc
        if not math.isfinite(number):
            raise ValueError(f"字段 {'/'.join(keys)} 必须是有限数")
        return number

    def _optional_xyz(
        self,
        item: Mapping[str, Any],
        keys: tuple[str, ...],
    ) -> Optional[tuple[float, float, float]]:
        value = self._find(item, keys)
        if value is None:
            return None
        if isinstance(value, Mapping):
            value = (value.get("x"), value.get("y"), value.get("z"))
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"字段 {'/'.join(keys)} 必须是三项坐标")
        if any(isinstance(number, bool) for number in value):
            raise ValueError(f"字段 {'/'.join(keys)} 必须是数值坐标")
        try:
            xyz = tuple(float(number) for number in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"字段 {'/'.join(keys)} 必须是数值坐标") from exc
        if not all(math.isfinite(number) for number in xyz):
            raise ValueError(f"字段 {'/'.join(keys)} 不能包含 NaN 或 Inf")
        return xyz  # type: ignore[return-value]


# 全局状态机

class GlobalFSM:
    """按业务回执推进的最小全局任务状态机。

    内部字段含义如下：``phase`` 是全局流程当前阶段，像流程表中高亮的一行；
    ``local_phase`` 是机械臂当前局部阶段的遥测；``task`` 是当前结构化任务；
    ``retry_count`` 像内部重新尝试计数器，记录客户端已进行的抓取重试次数；
    ``failure_reason`` 保存当前尚未恢复或最终失败的原因；``_fail_after_return`` 决定
    到达结束区域后进入 ``FAILED`` 还是 ``DONE``。``max_pick_retries`` 是客户端配置
    参数，不是赛事正式允许重试次数，正式限制仍以比赛规则为准。

    正常流程可以读成：

    - ``WAIT_READY``：等待系统准备；``LOAD_TASK``：等待装载 ``TaskSpec``；
    - ``SEARCH_TARGET``：寻找目标；``NAV_TO_PICK``：导航到抓取位置；
    - ``REFINE_TARGET``：靠近后重新估计物体位置；
    - ``PLAN_PICK``：计算抓取目标、IK 和轨迹；``EXECUTE_PICK``：执行抓取轨迹；
    - ``VERIFY_PICK``：依据真实观测确认抓取结果；
    - ``NAV_TO_PLACE``：导航到放置区域；``PLAN_PLACE``：计算放置目标和轨迹；
    - ``EXECUTE_PLACE``：执行放置动作；``VERIFY_PLACE``：确认物体正确释放；
    - ``RETURN_END``：任务结束前先回到规定区域；
    - ``DONE`` / ``FAILED``：客户端终止状态，只能通过 ``RESET`` 离开。

    ``status`` 只生成 ``FSMStatus`` 快照；非法或错误顺序事件返回 ``False``。显式
    失败会进入返区或 ``FAILED``，绝不把阶段推进本身当作抓放成功。

    当前骨架状态：``max_pick_retries`` 已从配置传入，但 ``timestamp_ns`` 尚未用于超时
    判断，``phase_entered_ns`` 尚未实现；``SAFE_HOLD`` 像踩住刹车等待安全条件恢复，
    但其进入、恢复和退出逻辑尚未实现。状态和事件存在不代表完整业务闭环已经完成。
    """

    # 正常主流程转换

    _FORWARD_TRANSITIONS = {
        (GlobalPhase.SEARCH_TARGET, FSMEvent.TARGET_FOUND): GlobalPhase.NAV_TO_PICK,
        (GlobalPhase.NAV_TO_PICK, FSMEvent.PICK_NAV_REACHED): GlobalPhase.REFINE_TARGET,
        (GlobalPhase.REFINE_TARGET, FSMEvent.TARGET_REFINED): GlobalPhase.PLAN_PICK,
        (GlobalPhase.PLAN_PICK, FSMEvent.PICK_PLAN_READY): GlobalPhase.EXECUTE_PICK,
        (GlobalPhase.EXECUTE_PICK, FSMEvent.PICK_EXECUTED): GlobalPhase.VERIFY_PICK,
        (GlobalPhase.VERIFY_PICK, FSMEvent.PICK_VERIFIED): GlobalPhase.NAV_TO_PLACE,
        (GlobalPhase.NAV_TO_PLACE, FSMEvent.PLACE_NAV_REACHED): GlobalPhase.PLAN_PLACE,
        (GlobalPhase.PLAN_PLACE, FSMEvent.PLACE_PLAN_READY): GlobalPhase.EXECUTE_PLACE,
        (GlobalPhase.EXECUTE_PLACE, FSMEvent.PLACE_EXECUTED): GlobalPhase.VERIFY_PLACE,
        (GlobalPhase.VERIFY_PLACE, FSMEvent.PLACE_VERIFIED): GlobalPhase.RETURN_END,
        (GlobalPhase.RETURN_END, FSMEvent.RETURN_REACHED): GlobalPhase.DONE,
    }

    def __init__(self, max_pick_retries: int = 1) -> None:
        """初始化尚未装载任务的客户端状态。

        ``max_pick_retries`` 是允许回到 ``REFINE_TARGET`` 的客户端重试上限，不能据此
        推断赛事正式重试规则。负数会抛出 ``ValueError``。
        """

        if max_pick_retries < 0:
            raise ValueError("max_pick_retries 不能为负数")
        self.max_pick_retries = int(max_pick_retries)
        self.phase = GlobalPhase.WAIT_READY
        self.local_phase = LocalPhase.IDLE
        self.task: Optional[TaskSpec] = None
        self.retry_count = 0
        self.failure_reason = ""
        self._fail_after_return = False

    def handle_event(self, event: FSMEvent, timestamp_ns: int, reason: str = "") -> bool:
        """提交一个已由真实观测确认的状态机事件。

        ``event`` 是上游业务回执，``timestamp_ns`` 单位纳秒，``reason`` 是失败说明。
        返回值表示是否发生合法状态切换；顺序不合法时返回 ``False``。全局失败在已有
        任务时先进入 ``RETURN_END``，未装载任务时直接进入 ``FAILED``。

        当前骨架状态：本方法保留 ``timestamp_ns`` 接口，但尚未据此进行阶段超时判断，
        也没有 ``phase_entered_ns``；除 ``SYSTEM_READY`` 外，业务事件尚未在
        ``ros_nodes`` 中接入真实反馈。``SAFE_HOLD`` 转换与恢复也尚未实现。
        """

        del timestamp_ns
        if event is FSMEvent.RESET:
            self.reset()
            return True
        # 终止态必须稳定：迟到的业务回执不能让已结束任务“复活”，只有 RESET 能离开。
        if self.phase in {GlobalPhase.DONE, GlobalPhase.FAILED}:
            return False
        if self.phase is GlobalPhase.WAIT_READY and event is FSMEvent.SYSTEM_READY:
            self.phase = GlobalPhase.LOAD_TASK
            return True

        # 特殊失败与抓取重试

        if event is FSMEvent.FAILURE:
            self.failure_reason = reason or "业务模块报告失败"
            if self.task is None:
                self.phase = GlobalPhase.FAILED
            else:
                # 已有任务的失败仍先返区；RETURN_END 像结束前回到规定区域，而非成功。
                self.phase = GlobalPhase.RETURN_END
                self._fail_after_return = True
            return True
        if self.phase is GlobalPhase.VERIFY_PICK and event is FSMEvent.PICK_FAILED:
            self.failure_reason = reason or "抓取验证失败"
            if self.retry_count < self.max_pick_retries:
                self.retry_count += 1
                # 抓取接触可能已移动物体，重试必须重新精定位，不能沿用旧三维位置。
                self.phase = GlobalPhase.REFINE_TARGET
            else:
                # 重试耗尽意味着失败尚未恢复，原因必须保留到返区后的 FAILED。
                self.phase = GlobalPhase.RETURN_END
                self._fail_after_return = True
            return True
        if self.phase is GlobalPhase.RETURN_END and event is FSMEvent.RETURN_REACHED:
            self.phase = GlobalPhase.FAILED if self._fail_after_return else GlobalPhase.DONE
            return True
        next_phase = self._FORWARD_TRANSITIONS.get((self.phase, event))
        # 错误顺序的回执不能推动流程，否则“发出命令”可能被误当成“已经完成”。
        if next_phase is None:
            return False
        if self.phase is GlobalPhase.VERIFY_PICK and event is FSMEvent.PICK_VERIFIED:
            # 新验证已确认重试成功，旧的可恢复失败不应继续污染 DONE 或遥测。
            self.failure_reason = ""
        self.phase = next_phase
        return True

    # 任务装载

    def submit_task(self, task: TaskSpec, timestamp_ns: int) -> bool:
        """在 ``LOAD_TASK`` 阶段装载一个结构化任务。

        ``task`` 必须来自 ``InstructionParser``，``timestamp_ns`` 单位纳秒。只有
        ``LOAD_TASK`` 接受有效任务；阶段不对或任务无效时返回 ``False`` 且保持原状态。
        任务内坐标单位米，坐标系沿用指令约定。

        当前骨架状态：该时间戳尚未用于装载超时或阶段进入时间判断。
        """

        del timestamp_ns
        if self.phase is not GlobalPhase.LOAD_TASK or not task.valid:
            return False
        self.task = task
        self.phase = GlobalPhase.SEARCH_TARGET
        self.retry_count = 0
        self.failure_reason = ""
        self._fail_after_return = False
        return True

    # 局部阶段遥测

    def set_local_phase(self, phase: LocalPhase) -> None:
        """更新机械臂局部阶段遥测。

        ``phase`` 是机械臂局部状态，没有单位和坐标系。该方法只更新遥测，不推进全局
        FSM，也不证明动作成功；传入非 ``LocalPhase`` 时抛出 ``TypeError``。
        """

        if not isinstance(phase, LocalPhase):
            raise TypeError("phase 必须是 LocalPhase")
        self.local_phase = phase

    # 状态快照与重置

    def status(self, timestamp_ns: int) -> FSMStatus:
        """生成当前状态的不可变遥测快照。

        ``timestamp_ns`` 是快照生成时间（纳秒）。该方法只能读取当前内存状态并构造
        无坐标的 ``FSMStatus``，不能推进流程。``success`` 只在客户端 ``DONE`` 时为
        ``True``，不自动表示裁判确认得分。
        """

        # 快照必须无副作用；控制和Recorder读取状态不应反过来改变任务进度。
        return FSMStatus(
            task_id=self.task.task_id if self.task is not None else -1,
            global_phase=self.phase,
            local_phase=self.local_phase,
            retry_count=self.retry_count,
            success=self.phase is GlobalPhase.DONE,
            failure_reason=self.failure_reason,
            timestamp_ns=int(timestamp_ns),
        )

    def reset(self) -> None:
        """清空任务并回到等待状态。

        RESET 是 ``DONE`` 和 ``FAILED`` 离开终止态的唯一事件路径。该操作会丢弃当前
        内存任务、重试和失败原因，调用方应先完成 Episode 记录；它不删除磁盘数据。
        """

        self.phase = GlobalPhase.WAIT_READY
        self.local_phase = LocalPhase.IDLE
        self.task = None
        self.retry_count = 0
        self.failure_reason = ""
        self._fail_after_return = False
