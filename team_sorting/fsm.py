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

# ————————————————————————————————
# 【Codex修改-40：使用运行时可校验的Mapping类型】
# 1. 修改前从typing导入Mapping，不适合作为所有运行时输入校验的唯一来源。
# 2. 当前从collections.abc导入Mapping，构造器和解析器据此严格识别映射对象。
# 3. 这样超时策略和任务JSON不会把任意可迭代对象误当成键值映射。
# 4. 只调整内部依赖来源，不改变任何公共接口。
from collections.abc import Mapping
# ————————————————————————————————
from enum import Enum
import json
import math
# ————————————————————————————————
# 【Codex修改-41：引入只读超时策略包装】
# 1. 修改前没有冻结阶段超时配置所需的标准库工具。
# 2. 当前引入MappingProxyType，用于保存构造时复制并校验后的只读映射。
# 3. 这样调用方不能在构造后绕过校验修改正在运行的超时策略。
# 4. 只新增内部实现依赖，不改变公共数据结构。
from types import MappingProxyType
from typing import Any, Optional
# ————————————————————————————————

from .interfaces import FSMStatus, GlobalPhase, LocalPhase, TaskSpec


# ————————————————————————————————
# 【Codex修改-42：统一严格校验纳秒时间戳】
# 1. 修改前各入口未统一拒绝bool、负数、浮点数和字符串，时间语义可能含糊。
# 2. 当前所有FSM时间入口复用本函数，只接受非负的Python整数。
# 3. 这样状态转换、读取和任务解析使用同一规则，不会发生隐式数值转换。
# 4. 仅新增私有模块辅助函数，不增加公共方法或字段。
def _require_timestamp_ns(timestamp_ns: int) -> int:
    """严格读取调用方提供的非负纳秒时间，不使用墙钟或隐式数值转换。"""

    if (
        isinstance(timestamp_ns, bool)
        or not isinstance(timestamp_ns, int)
        or timestamp_ns < 0
    ):
        raise ValueError("timestamp_ns必须是非负整数，不能使用bool、浮点数或字符串")
    return timestamp_ns
# ————————————————————————————————


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
    # ————————————————————————————————
    # 【Codex修改-43：区分普通失败与安全阶段失败】
    # 1. 修改前FAILURE说明只覆盖“有任务先返区”，没有说明SAFE_HOLD和RETURN_END例外。
    # 2. 当前明确普通阶段失败先返区，而安全暂停或返区自身失败直接进入FAILED。
    # 3. 这样事件生产者不会在安全条件未恢复或返区已失败时继续等待导航成功。
    # 4. FAILURE名称和值均未改变，只纠正既有事件的失败边界说明。
    # 已经装载任务且处于普通阶段时：
    # 当前阶段 → RETURN_END → FAILED
    #
    # SAFE_HOLD或RETURN_END中：
    # 当前阶段 → FAILED
    # ————————————————————————————————
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

    # ————————————————————————————————
    # 【Codex修改-44：增加外部安全暂停与恢复事件】
    # 1. 修改前FSM没有事件表达外部安全监控要求暂停及确认恢复。
    # 2. 当前增加SAFETY_HOLD和SAFETY_RECOVERED，并限定恢复只能回到被中断阶段。
    # 3. 这样安全暂停不会被伪装成业务成功，也不会由普通业务回执擅自解除。
    # 4. 扩展了FSMEvent枚举，但没有改变已有事件名称、值或方法签名。
    # 安全监控明确要求暂停普通流程。FSM保存被中断阶段；ActionMux依据SAFE_HOLD
    # 覆盖普通候选命令。当前ros_nodes尚未接入该事件的真实生产者。
    SAFETY_HOLD = "SAFETY_HOLD"

    # 安全监控明确确认恢复条件已经满足。只能从SAFE_HOLD恢复到原阶段，不会越级
    # 宣称任何感知、导航、规划或执行步骤成功。当前ros_nodes尚未接入真实生产者。
    SAFETY_RECOVERED = "SAFETY_RECOVERED"
    # ————————————————————————————————


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

        # ————————————————————————————————
        # 【Codex修改-45：任务解析入口校验接收时间】
        # 1. 修改前解析器会把未经严格检查的timestamp_ns写入TaskSpec。
        # 2. 当前在解析JSON前调用统一时间校验，只接受非负整数且拒绝bool等类型。
        # 3. 这样非法时间不会进入公共任务对象，也不会污染后续FSM时间基准。
        # 4. InstructionParser.parse签名和成功返回类型均未改变。
        # 接收时间必须在唯一解析入口校验，避免非法时间写入公共 TaskSpec。
        timestamp_ns = _require_timestamp_ns(timestamp_ns)
        # ————————————————————————————————
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

        # ————————————————————————————————
        # 【Codex修改-46：拒绝负数放置半径】
        # 1. 修改前place_radius只校验为有限数，负数距离约束也可能进入TaskSpec。
        # 2. 当前在解析可选数值后显式拒绝小于零的半径。
        # 3. 这样规划和验证不会接收到物理意义不成立且容易导致错误比较的容差。
        # 4. 不改变JSON字段、别名或TaskSpec结构，只收紧非法输入。
        place_radius = self._optional_float(item, ("place_radius", "radius"))
        # 半径是距离约束，负值必须拒绝，不能静默修正。
        if place_radius is not None and place_radius < 0.0:
            raise ValueError("字段 place_radius/radius 不能为负数")
        # ————————————————————————————————

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
            # 官方字段名已经明确其坐标语义；TaskSpec构造器不隐藏补默认frame。
            place_frame_id="world",
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

    ``phase_entered_ns`` 记录最近一次真实进入或恢复当前阶段的时间；暂停前已累计的
    业务活动时间单独保存在私有偏移量中，安全暂停时长不计入原阶段耗时。可选的
    ``phase_timeouts_ns`` 由调用方显式注入；默认不启用任何阶段超时。配置阶段达到
    超时边界后直接进入 ``FAILED``，不会自动成功、重试、返区或进入可恢复安全暂停。
    外部 ``SAFETY_HOLD`` 仍只能由 ``SAFETY_RECOVERED`` 恢复到被中断阶段。状态和
    事件存在不代表完整业务闭环已经完成。
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
        # ————————————————————————————————
        # 【Codex修改-77：删除不可达的返区到达转换表项】
        # 1. 修改前表中还声明RETURN_END加RETURN_REACHED直接进入DONE，但该事件已被前置专用分支处理。
        # 2. 当前从表中删除该重复项，正常返区由专用分支进入DONE，失败返区则进入FAILED。
        # 3. 这样读表时不会误以为失败返区也会成功，同时不改变任何实际执行路径。
        # 4. 只清理私有转换表，不改变FSMEvent、GlobalPhase或公共方法。
        # ————————————————————————————————
    }

    # ————————————————————————————————
    # 【Codex修改-47：集中定义终止态和禁止超时阶段】
    # 1. 修改前终止态判断散落在事件处理中，也没有统一限制哪些阶段可配置普通超时。
    # 2. 当前集中定义DONE、FAILED终止态，并禁止WAIT_READY、SAFE_HOLD等阶段超时。
    # 3. 这样终止保护和超时配置使用同一集合，WAIT_READY不会因无可靠基准被误超时。
    # 4. 只新增类内实现常量，不改变GlobalPhase或公共方法。
    _TERMINAL_PHASES = {GlobalPhase.DONE, GlobalPhase.FAILED}
    # WAIT_READY 没有可靠启动时间基准，保守禁止配置普通阶段超时。
    _NON_TIMEOUT_PHASES = {
        *_TERMINAL_PHASES,
        GlobalPhase.WAIT_READY,
        GlobalPhase.SAFE_HOLD,
    }
    # ————————————————————————————————

    def __init__(
        self,
        max_pick_retries: int = 1,
        # ————————————————————————————————
        # 【Codex修改-48：允许显式注入阶段超时策略】
        # 1. 修改前构造器只能配置抓取重试，无法由调用方选择性提供阶段超时预算。
        # 2. 当前增加可选的GlobalPhase到纳秒整数映射，None表示保持默认无超时。
        # 3. 这样正式策略未确认时不会硬编码预算，确认后又能通过显式配置启用。
        # 4. GlobalFSM构造器增加了可选参数，已有只传重试次数的调用保持兼容。
        phase_timeouts_ns: Optional[Mapping[GlobalPhase, int]] = None,
        # ————————————————————————————————
    ) -> None:
        """初始化尚未装载任务的客户端状态。

        ``max_pick_retries`` 是允许回到 ``REFINE_TARGET`` 的客户端重试上限，不能据此
        推断赛事正式重试规则。``phase_timeouts_ns`` 是调用方提供的阶段超时纳秒映射；
        缺省时保持原有不自动超时行为。负数重试、非正超时、终止态或SAFE_HOLD超时
        配置会抛出 ``ValueError``；WAIT_READY 同样不能配置普通阶段超时。
        """

        # ————————————————————————————————
        # 【Codex修改-49：严格校验重试上限和超时策略】
        # 1. 修改前只拒绝负数重试，bool和错误类型可能被接受，也没有阶段超时校验。
        # 2. 当前严格检查重试类型、映射类型、阶段键、禁止阶段及正整数纳秒限制。
        # 3. 这样无效配置在启动时清晰失败，不会运行中产生隐式转换或危险超时。
        # 4. 只收紧构造参数校验；默认值和合法配置的公共行为保持兼容。
        if (
            isinstance(max_pick_retries, bool)
            or not isinstance(max_pick_retries, int)
            or max_pick_retries < 0
        ):
            raise ValueError("max_pick_retries 必须是非负整数")
        if phase_timeouts_ns is None:
            timeout_policy: dict[GlobalPhase, int] = {}
        elif not isinstance(phase_timeouts_ns, Mapping):
            raise ValueError("phase_timeouts_ns 必须是 Mapping 或 None")
        else:
            timeout_policy = dict(phase_timeouts_ns)
        for phase, timeout_ns in timeout_policy.items():
            if not isinstance(phase, GlobalPhase):
                raise ValueError("阶段超时策略的键必须是 GlobalPhase")
            if phase in self._NON_TIMEOUT_PHASES:
                raise ValueError(f"{phase.value} 不能配置阶段超时")
            if (
                isinstance(timeout_ns, bool)
                or not isinstance(timeout_ns, int)
                or timeout_ns <= 0
            ):
                raise ValueError(f"{phase.value} 的阶段超时必须是正整数纳秒")
        self.max_pick_retries = int(max_pick_retries)
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-50：复制并冻结阶段超时配置】
        # 1. 修改前即使新增映射参数，直接保存调用方字典也会允许构造后绕过校验修改。
        # 2. 当前先复制策略，再用MappingProxyType保存为只读映射。
        # 3. 这样运行中的超时条件稳定可审计，不受外部可变对象的后续写入影响。
        # 4. 不新增公共字段；只读属性仍返回Mapping接口。
        # 复制并冻结调用方策略：外部原字典后续变化不能绕过构造期校验影响运行状态。
        self._phase_timeouts_ns = MappingProxyType(timeout_policy)
        # ————————————————————————————————
        self.phase = GlobalPhase.WAIT_READY
        # ————————————————————————————————
        # 【Codex修改-51：记录真实阶段进入时间】
        # 1. 修改前FSM没有phase_entered_ns，无法计算阶段活动时间或审计真实转换时刻。
        # 2. 当前初始化为空，并由合法转换、恢复和RESET建立调用方提供的时间基准。
        # 3. 这样计时不依赖墙钟，也不会在构造时伪造尚未发生的阶段进入时间。
        # 4. 增加实例可读状态，但未改变已有公共方法签名或公共ROS协议。
        self.phase_entered_ns: Optional[int] = None
        # ————————————————————————————————
        self.local_phase = LocalPhase.IDLE
        self.task: Optional[TaskSpec] = None
        self.retry_count = 0
        self.failure_reason = ""
        self._fail_after_return = False
        # ————————————————————————————————
        # 【Codex修改-52：保存转换顺序与安全暂停上下文】
        # 1. 修改前FSM不记最近转换时间、被中断阶段或暂停前已用预算。
        # 2. 当前分别保存时间基准、原阶段和暂停前累计活动时间，供迟到检查与恢复使用。
        # 3. 这样旧回执不能倒序推进，多次安全暂停也不会丢失或重复计算业务耗时。
        # 4. 只增加私有运行状态，不改变FSMStatus字段或事件方法签名。
        self._last_transition_ns: Optional[int] = None
        self._interrupted_phase: Optional[GlobalPhase] = None
        # 保存暂停前的业务活动时间，恢复后继续使用原预算而不是重新计时。
        self._interrupted_elapsed_ns: Optional[int] = None
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-1：分离真实阶段时间与累计活动时间】
        # 1. 修改前通过伪造 phase_entered_ns 保存暂停前耗时，使该字段不再是真实时间。
        # 2. 当前用独立偏移量保存本次进入前已累计的业务活动时间。
        # 3. 这样恢复时间可用于审计，暂停时长又不会消耗业务预算，更不易混淆时间语义。
        # 4. 只新增私有状态，不改变任何公共接口或方法签名。
        self._phase_elapsed_offset_ns: int = 0
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-53：独立保存安全暂停原因】
        # 1. 修改前没有专用上下文区分业务失败原因与临时安全暂停原因。
        # 2. 当前以私有字段保存暂停原因，状态快照按需与原失败原因组合显示。
        # 3. 这样恢复后可清除临时原因，同时保留真正的业务根因用于诊断。
        # 4. 不扩展FSMStatus或其他公共接口。
        self._safe_hold_reason = ""
        # ————————————————————————————————

    def handle_event(self, event: FSMEvent, timestamp_ns: int, reason: str = "") -> bool:
        """提交一个已由真实观测确认的状态机事件。

        ``event`` 是上游业务回执，``timestamp_ns`` 单位纳秒，``reason`` 是字符串
        失败说明。返回值表示是否发生合法状态切换；顺序不合法时返回 ``False``。
        普通阶段的全局失败在已有任务时先进入 ``RETURN_END``，未装载任务时直接进入
        ``FAILED``；``SAFE_HOLD`` 或 ``RETURN_END`` 中的失败直接进入 ``FAILED``。

        时间戳必须是非负整数；负数或错误类型抛出 ``ValueError``；``reason`` 非字符串
        抛出 ``TypeError``；早于最近合法转换的迟到事件返回 ``False``。除
        ``SYSTEM_READY`` 外，业务事件尚未在 ``ros_nodes`` 中接入真实反馈，安全保持
        与恢复事件也只有定义和纯Python语义。
        """

        # ————————————————————————————————
        # 【Codex修改-54：严格校验事件、原因和事件时间】
        # 1. 修改前handle_event忽略timestamp_ns，也未拒绝错误事件类型或非字符串原因。
        # 2. 当前要求FSMEvent、str和严格非负整数时间，空字符串仍由各分支使用默认原因。
        # 3. 这样错误类型不会进入状态机，失败文本也不会在格式化时产生意外对象语义。
        # 4. handle_event签名和bool返回值不变，仅收紧非法输入。
        if not isinstance(event, FSMEvent):
            raise TypeError("event 必须是 FSMEvent")
        if not isinstance(reason, str):
            raise TypeError("reason 必须是 str")
        timestamp_ns = _require_timestamp_ns(timestamp_ns)
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-55：RESET优先并保护事件时间顺序】
        # 1. 修改前没有迟到检查，RESET也无法在仿真时钟回退后重建新的时间基准。
        # 2. 当前先处理RESET，再拒绝旧事件、保护终止态，并把外部安全请求送入暂停入口。
        # 3. 这样新生命周期可合法回退时钟，旧业务回执又不能复活终态或倒序推进。
        # 4. 不新增公共方法；RESET及安全事件仍通过原handle_event入口处理。
        # 显式 RESET 代表新生命周期，可在仿真时钟回退后重建时间基准。
        # RESET 的可信生产者仍必须由 ROS 生命周期/任务管理层限制，FSM 不自行触发。
        if event is FSMEvent.RESET:
            self._reset_at(timestamp_ns)
            return True
        if self._is_stale(timestamp_ns):
            return False
        # 终止态必须稳定：迟到的业务回执不能让已结束任务“复活”，只有 RESET 能离开。
        if self.phase in self._TERMINAL_PHASES:
            return False
        if event is FSMEvent.SAFETY_HOLD:
            return self._enter_safe_hold(
                timestamp_ns, reason or "安全监控请求进入SAFE_HOLD"
            )
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-2：安全暂停恢复与不可恢复失败分流】
        # 1. 修改前恢复时伪造阶段进入时间，SAFE_HOLD中的FAILURE还会落入普通返区路径。
        # 2. 当前恢复使用私有辅助方法；FAILURE则直接通过统一失败入口进入FAILED。
        # 3. 这样安全条件未恢复时不会尝试返区，真实恢复时刻和活动预算也能同时保留。
        # 4. 事件名称、handle_event签名和返回类型均未改变。
        if self.phase is GlobalPhase.SAFE_HOLD:
            if event is FSMEvent.SAFETY_RECOVERED:
                return self._resume_interrupted_phase(timestamp_ns)
            if event is FSMEvent.FAILURE:
                safe_failure_reason = (
                    reason
                    or self._safe_hold_reason
                    or "安全暂停期间发生不可恢复失败"
                )
                if self.failure_reason:
                    safe_failure_reason = (
                        f"原失败：{self.failure_reason}；"
                        f"安全暂停失败：{safe_failure_reason}"
                    )
                self._enter_failed(timestamp_ns, safe_failure_reason)
                return True
            return False
        if event is FSMEvent.SAFETY_RECOVERED:
            return False
        # ————————————————————————————————
        if self.phase is GlobalPhase.WAIT_READY and event is FSMEvent.SYSTEM_READY:
            # ————————————————————————————————
            # 【Codex修改-56：系统就绪转换建立真实时间基准】
            # 1. 修改前SYSTEM_READY只直接改phase，没有同步阶段进入和最近转换时间。
            # 2. 当前通过_transition进入LOAD_TASK并记录事件时间。
            # 3. 这样后续任务装载和状态读取能按统一时间顺序校验。
            # 4. SYSTEM_READY事件及handle_event公共契约不变。
            self._transition(GlobalPhase.LOAD_TASK, timestamp_ns)
            # ————————————————————————————————
            return True

        # 特殊失败与抓取重试

        if event is FSMEvent.FAILURE:
            # ————————————————————————————————
            # 【Codex修改-3：返区失败立即终止】
            # 1. 修改前RETURN_END收到FAILURE仍停留返区，缺少RETURN_REACHED时会永久卡住。
            # 2. 当前把原根因与本次返区失败最多组合一次，并直接切换到FAILED。
            # 3. 这样导航已经不可恢复时不会继续等待不可能到来的成功回执，原因也不会增长。
            # 4. 没有新增事件、状态、字段或公共方法。
            if self.phase is GlobalPhase.RETURN_END and self.task is not None:
                return_failure_reason = reason or "返区期间发生不可恢复失败"
                if self.failure_reason:
                    return_failure_reason = (
                        f"{self.failure_reason}；返区失败：{return_failure_reason}"
                    )
                self._enter_failed(timestamp_ns, return_failure_reason)
                return True
            # ————————————————————————————————
            self.failure_reason = reason or "业务模块报告失败"
            # ————————————————————————————————
            # 【Codex修改-57：普通失败清理残留暂停上下文】
            # 1. 修改前普通FAILURE没有安全暂停上下文可清理，新增上下文后可能泄漏到失败路径。
            # 2. 当前在进入返区或最终失败前清空被中断阶段、已用时间和暂停原因。
            # 3. 这样后续状态不会把已经失效的恢复目标或暂停诊断带入返区流程。
            # 4. 普通FAILURE有任务先返区、无任务直接失败的公共语义不变。
            self._interrupted_phase = None
            self._interrupted_elapsed_ns = None
            self._safe_hold_reason = ""
            # ————————————————————————————————
            if self.task is None:
                # ————————————————————————————————
                # 【Codex修改-4：无任务失败也使用统一失败入口】
                # 1. 修改前无任务FAILURE直接_transition，失败上下文清理由调用分支各自维护。
                # 2. 当前把最终失败集中交给_enter_failed，统一清理安全暂停和计时上下文。
                # 3. 这样所有FAILED入口具有同一安全后置条件，同时仍保留任务和重试诊断。
                # 4. 只调用新增私有方法，普通FAILURE公共语义和签名不变。
                self._enter_failed(timestamp_ns, self.failure_reason)
                # ————————————————————————————————
            else:
                # 已有任务的失败仍先返区；RETURN_END 像结束前回到规定区域，而非成功。
                self._fail_after_return = True
                # ————————————————————————————————
                # 【Codex修改-58：普通失败返区同步计时状态】
                # 1. 修改前有任务失败时直接改为RETURN_END，没有建立返区阶段时间基准。
                # 2. 当前通过_transition进入RETURN_END并记录真实事件时间。
                # 3. 这样返区的状态读取、迟到保护和可选超时都基于同一转换时刻。
                # 4. 失败仍先返区，没有改变既有业务策略或公共接口。
                self._transition(GlobalPhase.RETURN_END, timestamp_ns)
                # ————————————————————————————————
            return True
        if self.phase is GlobalPhase.VERIFY_PICK and event is FSMEvent.PICK_FAILED:
            self.failure_reason = reason or "抓取验证失败"
            if self.retry_count < self.max_pick_retries:
                self.retry_count += 1
                # 抓取接触可能已移动物体，重试必须重新精定位，不能沿用旧三维位置。
                # ————————————————————————————————
                # 【Codex修改-59：抓取重试转换同步计时状态】
                # 1. 修改前PICK_FAILED重试只直接修改phase，REFINE_TARGET没有新时间基准。
                # 2. 当前通过_transition回到REFINE_TARGET并记录真实失败回执时间。
                # 3. 这样重试阶段不会继承VERIFY_PICK的已用时间，迟到事件也能被拒绝。
                # 4. 重试次数和必须重新精定位的既有语义不变。
                self._transition(GlobalPhase.REFINE_TARGET, timestamp_ns)
                # ————————————————————————————————
            else:
                # 重试耗尽意味着失败尚未恢复，原因必须保留到返区后的 FAILED。
                self._fail_after_return = True
                # ————————————————————————————————
                # 【Codex修改-60：重试耗尽返区同步计时状态】
                # 1. 修改前重试耗尽直接改为RETURN_END，未记录进入返区的真实时间。
                # 2. 当前统一通过_transition进入RETURN_END。
                # 3. 这样失败返区拥有独立预算和时间顺序，不会沿用抓取验证阶段计时。
                # 4. 仍保留原失败原因并等待真实RETURN_REACHED，公共策略不变。
                self._transition(GlobalPhase.RETURN_END, timestamp_ns)
                # ————————————————————————————————
            return True
        if self.phase is GlobalPhase.RETURN_END and event is FSMEvent.RETURN_REACHED:
            # ————————————————————————————————
            # 【Codex修改-5：失败返区完成后统一进入FAILED】
            # 1. 修改前失败返区完成仅做普通转换，可能遗留失败路径的临时上下文。
            # 2. 当前失败路径调用_enter_failed，正常路径仍按原行为进入DONE。
            # 3. 这样FAILED状态入口一致，且不会把返区完成误判为任务成功。
            # 4. RETURN_REACHED事件和handle_event公共契约保持不变。
            if self._fail_after_return:
                self._enter_failed(
                    timestamp_ns,
                    self.failure_reason or "任务失败后已返回结束区域",
                )
            else:
                self._transition(GlobalPhase.DONE, timestamp_ns)
            # ————————————————————————————————
            return True
        next_phase = self._FORWARD_TRANSITIONS.get((self.phase, event))
        # 错误顺序的回执不能推动流程，否则“发出命令”可能被误当成“已经完成”。
        if next_phase is None:
            return False
        if self.phase is GlobalPhase.VERIFY_PICK and event is FSMEvent.PICK_VERIFIED:
            # 新验证已确认重试成功，旧的可恢复失败不应继续污染 DONE 或遥测。
            self.failure_reason = ""
        # ————————————————————————————————
        # 【Codex修改-61：正常主流程转换统一记录时间】
        # 1. 修改前合法前向事件只直接赋值phase，没有更新阶段进入或最近转换时间。
        # 2. 当前所有表驱动的正常转换统一调用_transition。
        # 3. 这样每个业务阶段都从真实完成回执时刻开始计时，并共享迟到保护。
        # 4. 合法阶段顺序、FSMEvent名称和handle_event返回语义均未改变。
        self._transition(next_phase, timestamp_ns)
        # ————————————————————————————————
        return True

    # ————————————————————————————————
    # 【Codex修改-62：增加显式阶段超时检查入口】
    # 1. 修改前FSM保留timestamp_ns却没有阶段计时或显式超时检查能力。
    # 2. 当前新增check_timeout，先校验时间、拒绝旧检查并跳过禁止超时的阶段。
    # 3. 这样status保持无副作用，集成层可在真实回执未转换后再主动检查预算。
    # 4. 新增公共读取/动作方法，但默认空策略保持原流程不自动超时。
    # 阶段计时与超时接口的正式使用方式仍需组长确认；默认策略保持为空。
    def check_timeout(self, timestamp_ns: int) -> bool:
        """显式检查当前阶段是否到达调用方配置的超时边界。

        默认策略为空，因此保持原有不自动超时行为。检查只使用调用方提供的纳秒时间；
        ``status`` 不调用本方法。配置阶段在 ``elapsed >= timeout`` 时直接进入
        ``FAILED``，不会进入 ``SAFE_HOLD``、自动重试、返区或判定成功。迟到检查返回
        ``False``，非法时间戳抛出 ``ValueError``。ROS 集成层应先处理本周期真实业务
        回执，再检查当前阶段是否超时，使截止时刻到达的真实成功反馈优先。
        """

        # 拒绝早于最近合法状态转换时间的旧检查。
        timestamp_ns = _require_timestamp_ns(timestamp_ns)
        if self._is_stale(timestamp_ns):
            return False
        if self.phase in self._NON_TIMEOUT_PHASES or self.phase_entered_ns is None:
            return False
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-6：阶段超时直接失败】
        # 1. 修改前超时进入SAFE_HOLD，恢复后因预算耗尽会立刻再次超时并形成循环。
        # 2. 当前在活动时间达到边界时直接进入FAILED，并记录阶段、实际时间和配置限制。
        # 3. 这样超时不会伪装成可恢复暂停，也不会自动成功、重试或驱动机器人返区。
        # 4. check_timeout签名和默认空策略兼容性保持不变。
        timeout_ns = self._phase_timeouts_ns.get(self.phase)
        if timeout_ns is None:
            return False
        elapsed_ns = self._current_phase_elapsed_ns(timestamp_ns)
        if elapsed_ns is None or elapsed_ns < timeout_ns:
            return False
        self._enter_failed(
            timestamp_ns,
            f"{self.phase.value}阶段超时：实际活动时间{elapsed_ns}ns，"
            f"配置限制{timeout_ns}ns",
        )
        return True
        # ————————————————————————————————

    # ————————————————————————————————
    # 【Codex修改-63：增加无副作用的阶段活动时间读取】
    # 1. 修改前外部无法读取阶段耗时，也无法区分业务活动时间和安全暂停时间。
    # 2. 当前新增phase_elapsed_ns，严格校验时间且只读取，不触发超时或状态推进。
    # 3. 这样诊断和集成层可观察统一活动时间，同时不会因读取改变比赛流程。
    # 4. 新增公共只读方法，不改变已有方法签名或FSMStatus结构。
    # ————————————————————————————————
    # 【Codex修改-78：澄清不同阶段的活动时间含义】
    # 1. 修改前docstring笼统称结果不含安全暂停时长，容易误解SAFE_HOLD中的返回值。
    # 2. 当前分别说明普通业务阶段、SAFE_HOLD阶段及恢复后的累计规则。
    # 3. 这样调用方能正确解释读数，不会把SAFE_HOLD自身持续时间当成原业务耗时。
    # 4. 只修改文档，不改变phase_elapsed_ns签名、计算逻辑或返回类型。
    def phase_elapsed_ns(self, timestamp_ns: int) -> Optional[int]:
        """返回当前阶段已持续的纳秒数，不推进状态或触发超时。

        构造后尚无调用方时间基准时返回 ``None``。时间戳必须是非负整数，且不能早于
        最近合法状态转换时间；否则抛出 ``ValueError``。普通业务阶段返回该阶段累计
        活动时间，并排除中间的安全暂停时长；处于 ``SAFE_HOLD`` 时返回本次安全暂停
        已持续的时间；恢复原业务阶段后继续累计暂停前的业务活动时间。本方法不读取
        墙钟。
        """
        # ————————————————————————————————

        timestamp_ns = _require_timestamp_ns(timestamp_ns)
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-7：活动时间读取使用真实基准和累计偏移】
        # 1. 修改前只做timestamp-phase_entered，且无法同时表达真实恢复时刻和历史活动时间。
        # 2. 当前先拒绝早于最近转换的读取，再通过私有辅助方法累加偏移和本段时间。
        # 3. 这样读取不会产生负耗时、不会计入暂停时间，也不会推进状态或触发超时。
        # 4. phase_elapsed_ns签名和Optional[int]返回语义没有改变。
        if self._is_stale(timestamp_ns):
            raise ValueError("timestamp_ns不能早于最近合法状态转换时间")
        if self.phase_entered_ns is None:
            return None
        return self._current_phase_elapsed_ns(timestamp_ns)
        # ————————————————————————————————

    # ————————————————————————————————
    # 【Codex修改-64：公开只读阶段超时策略】
    # 1. 修改前没有标准入口供测试和集成层确认FSM实际采用的超时配置。
    # 2. 当前通过属性返回构造时已校验并冻结的Mapping。
    # 3. 这样配置可观察但不可被调用方原地篡改，避免运行策略漂移。
    # 4. 新增只读公共属性，不开放写入接口。
    @property
    def phase_timeouts_ns(self) -> Mapping[GlobalPhase, int]:
        """返回构造时已校验并冻结的阶段超时策略。"""

        return self._phase_timeouts_ns
    # ————————————————————————————————

    # 任务装载

    def submit_task(self, task: TaskSpec, timestamp_ns: int) -> bool:
        """在 ``LOAD_TASK`` 阶段装载一个结构化任务。

        ``task`` 必须来自 ``InstructionParser``，``timestamp_ns`` 单位纳秒。只有
        ``LOAD_TASK`` 接受有效任务；阶段不对或任务无效时返回 ``False`` 且保持原状态。
        任务内坐标单位米，坐标系沿用指令约定。

        时间戳必须是非负整数且不能早于最近合法转换；不合法或迟到时不装载任务。
        """

        # ————————————————————————————————
        # 【Codex修改-65：任务装载校验时间顺序和公共类型】
        # 1. 修改前submit_task忽略时间戳，也可能访问非TaskSpec对象的valid属性。
        # 2. 当前严格校验纳秒时间、拒绝迟到装载，并对task执行明确类型检查。
        # 3. 这样旧任务或结构相似的临时对象不能污染当前FSM生命周期。
        # 4. submit_task签名与合法任务的bool返回语义不变，仅收紧非法输入。
        timestamp_ns = _require_timestamp_ns(timestamp_ns)
        if self._is_stale(timestamp_ns):
            return False
        if not isinstance(task, TaskSpec):
            raise TypeError("task 必须是 TaskSpec")
        # ————————————————————————————————
        if self.phase is not GlobalPhase.LOAD_TASK or not task.valid:
            return False
        self.task = task
        self.retry_count = 0
        self.failure_reason = ""
        self._fail_after_return = False
        # ————————————————————————————————
        # 【Codex修改-66：装载任务后统一建立搜索阶段时间】
        # 1. 修改前合法任务装载只直接把phase改为SEARCH_TARGET。
        # 2. 当前通过_transition进入搜索阶段并记录装载事件时间。
        # 3. 这样首个业务阶段也具备真实进入时刻、迟到保护和可选超时基准。
        # 4. 任务选择、字段和submit_task公共返回语义均未改变。
        self._transition(GlobalPhase.SEARCH_TARGET, timestamp_ns)
        # ————————————————————————————————
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

        # ————————————————————————————————
        # 【Codex修改-67：状态快照严格校验时间类型】
        # 1. 修改前status仅用int转换写入时间，bool、浮点数或字符串可能被静默接受。
        # 2. 当前在构造快照前调用统一时间校验。
        # 3. 这样遥测与Recorder不会收到经隐式转换产生的含糊时间戳。
        # 4. status签名和FSMStatus字段不变，仅收紧非法输入。
        timestamp_ns = _require_timestamp_ns(timestamp_ns)
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-8：状态快照拒绝跨越最近转换时间】
        # 1. 修改前合法类型但早于最近转换的时间仍可生成时间顺序错误的快照。
        # 2. 当前对照_last_transition_ns拒绝旧时间，相同时间戳继续允许。
        # 3. 这样Recorder和控制侧不会收到倒序状态；异常路径只读且不修改FSM。
        # 4. status签名、FSMStatus结构和success语义均未改变。
        if self._is_stale(timestamp_ns):
            raise ValueError("timestamp_ns不能早于最近合法状态转换时间")
        # ————————————————————————————————
        # 快照必须无副作用；控制和Recorder读取状态不应反过来改变任务进度。
        # ————————————————————————————————
        # 【Codex修改-68：安全暂停快照同时显示两类原因】
        # 1. 修改前快照只返回failure_reason，SAFE_HOLD原因或原业务根因会有一方不可见。
        # 2. 当前仅在SAFE_HOLD快照中一次性组合原失败原因和安全暂停原因。
        # 3. 这样诊断信息完整且不会反写内部状态，也不会让failure_reason字符串持续增长。
        # 4. 不增加FSMStatus字段或JSON字段，只扩充既有字段在SAFE_HOLD中的内容。
        displayed_reason = self.failure_reason
        if self.phase is GlobalPhase.SAFE_HOLD:
            # 不扩展 FSMStatus 字段，同时保留原失败与安全暂停原因。
            if self.failure_reason and self._safe_hold_reason:
                displayed_reason = (
                    f"原失败：{self.failure_reason}；安全暂停：{self._safe_hold_reason}"
                )
            else:
                displayed_reason = self._safe_hold_reason or self.failure_reason
        # ————————————————————————————————
        return FSMStatus(
            task_id=self.task.task_id if self.task is not None else -1,
            global_phase=self.phase,
            local_phase=self.local_phase,
            retry_count=self.retry_count,
            success=self.phase is GlobalPhase.DONE,
            # ————————————————————————————————
            # 【Codex修改-69：快照使用组合原因和已校验时间】
            # 1. 修改前FSMStatus直接使用内部failure_reason并再次用int隐式转换时间。
            # 2. 当前写入只读计算得到的displayed_reason和已经严格校验的timestamp_ns。
            # 3. 这样SAFE_HOLD诊断完整，快照时间也不会绕过入口校验。
            # 4. FSMStatus字段、顺序和JSON协议均未改变。
            failure_reason=displayed_reason,
            timestamp_ns=timestamp_ns,
            # ————————————————————————————————
        )

    def reset(self) -> None:
        """清空任务并回到等待状态。

        RESET 是 ``DONE`` 和 ``FAILED`` 离开终止态的唯一事件路径。该操作会丢弃当前
        内存任务、重试和失败原因，调用方应先完成 Episode 记录；它不删除磁盘数据。
        """

        self.phase = GlobalPhase.WAIT_READY
        # ————————————————————————————————
        # 【Codex修改-70：重置清除阶段进入时间】
        # 1. 修改前reset没有阶段时间状态可清理，新增计时后旧基准可能污染下一生命周期。
        # 2. 当前普通reset把phase_entered_ns恢复为None，由RESET事件再建立指定新基准。
        # 3. 这样直接reset不会伪造时间，事件RESET又能支持仿真时钟回退。
        # 4. reset签名及回到WAIT_READY的公共行为不变。
        # 重置必须清理计时与安全暂停上下文，避免跨生命周期继承旧预算。
        self.phase_entered_ns = None
        # ————————————————————————————————
        self.local_phase = LocalPhase.IDLE
        self.task = None
        self.retry_count = 0
        self.failure_reason = ""
        self._fail_after_return = False
        # ————————————————————————————————
        # 【Codex修改-71：重置清除顺序与暂停计时上下文】
        # 1. 修改前没有这些私有状态，新增后若残留会拒绝新事件或恢复旧阶段。
        # 2. 当前清除最近转换时间、被中断阶段和暂停前累计时间。
        # 3. 这样新任务不会继承旧任务的迟到门槛、恢复目标或活动预算。
        # 4. 只清理私有状态，不删除Recorder数据或改变公共协议。
        self._last_transition_ns = None
        self._interrupted_phase = None
        self._interrupted_elapsed_ns = None
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-9：重置清理活动时间偏移】
        # 1. 修改前没有独立活动时间偏移，新增字段若不清理会污染下一任务生命周期。
        # 2. 当前reset显式归零偏移；_reset_at调用reset后同样获得干净状态。
        # 3. 这样普通重置和仿真时钟回退重置都不会继承旧阶段预算。
        # 4. reset与RESET事件的公共接口和允许时钟回退行为保持不变。
        self._phase_elapsed_offset_ns = 0
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-72：重置清除安全暂停原因】
        # 1. 修改前新增的_safe_hold_reason若不清理，会污染下一任务的状态快照。
        # 2. 当前在reset末尾显式清空临时安全诊断。
        # 3. 这样RESET后的WAIT_READY不会显示上一生命周期的安全原因。
        # 4. 只清理私有字段，不改变FSMStatus结构。
        self._safe_hold_reason = ""
        # ————————————————————————————————

    # ————————————————————————————————
    # 【Codex修改-73：集中判断迟到时间】
    # 1. 修改前各入口没有统一的最近合法转换时间比较。
    # 2. 当前私有辅助方法只在timestamp_ns早于_last_transition_ns时判定为旧数据。
    # 3. 这样相同时间戳仍被允许，事件、任务和读取入口可共享一致顺序语义。
    # 4. 仅新增私有方法，不改变公共接口。
    def _is_stale(self, timestamp_ns: int) -> bool:
        return (
            self._last_transition_ns is not None
            and timestamp_ns < self._last_transition_ns
        )
    # ————————————————————————————————

    def _transition(self, phase: GlobalPhase, timestamp_ns: int) -> None:
        # ————————————————————————————————
        # 【Codex修改-10：新阶段重置活动时间偏移】
        # 1. 修改前没有独立偏移，普通阶段转换无法清晰开始一份全新的活动预算。
        # 2. 当前每次真实进入新阶段都把累计偏移归零，再记录真实进入时间。
        # 3. 这样前一阶段或SAFE_HOLD的活动时间不会泄漏到新阶段。
        # 4. _transition仍是私有方法，不改变公共状态机接口。
        self._phase_elapsed_offset_ns = 0
        self.phase = phase
        self.phase_entered_ns = timestamp_ns
        self._last_transition_ns = timestamp_ns
        # ————————————————————————————————

    # ————————————————————————————————
    # 【Codex修改-74：建立安全暂停入口并保存原阶段】
    # 1. 修改前没有实现SAFETY_HOLD的状态转换，也没有记录应恢复到哪个阶段。
    # 2. 当前拒绝终止态和重复暂停，并在进入SAFE_HOLD前保存当前业务阶段。
    # 3. 这样终止任务不会被安全事件复活，恢复也只能回到确实被中断的阶段。
    # 4. 仅新增私有辅助方法，外部仍通过handle_event提交既有事件。
    def _enter_safe_hold(self, timestamp_ns: int, reason: str) -> bool:
        if self.phase in self._TERMINAL_PHASES or self.phase is GlobalPhase.SAFE_HOLD:
            return False
        self._interrupted_phase = self.phase
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-11：安全暂停前累计完整活动时间】
        # 1. 修改前只保存本段时间，多次暂停恢复后会丢失更早已消耗的活动预算。
        # 2. 当前保存offset+timestamp-phase_entered，并让SAFE_HOLD通过_transition归零偏移。
        # 3. 这样任意次数暂停都只累计真实业务活动时间，暂停时长不会进入原阶段预算。
        # 4. SAFETY_HOLD事件和_enter_safe_hold私有签名均未改变。
        active_elapsed_ns = self._current_phase_elapsed_ns(timestamp_ns)
        self._interrupted_elapsed_ns = (
            self._phase_elapsed_offset_ns
            if active_elapsed_ns is None
            else active_elapsed_ns
        )
        # ————————————————————————————————
        # ————————————————————————————————
        # 【Codex修改-75：进入SAFE_HOLD保存原因并建立暂停时间】
        # 1. 修改前没有安全暂停状态实现，无法记录暂停原因或真实进入时刻。
        # 2. 当前保存安全原因，并通过_transition进入SAFE_HOLD、重置该阶段活动偏移。
        # 3. 这样暂停本身有可审计时间基准，原业务阶段预算则单独保存在中断上下文。
        # 4. 不扩展FSMStatus字段或公共ROS协议。
        self._safe_hold_reason = reason
        self._transition(GlobalPhase.SAFE_HOLD, timestamp_ns)
        return True
        # ————————————————————————————————

    # ————————————————————————————————
    # 【Codex修改-12：统一活动时间计算与安全恢复】
    # 1. 修改前时间计算散落在读取、超时和恢复分支，恢复还会伪造phase_entered_ns。
    # 2. 当前两个私有辅助方法统一计算累计活动时间，并用真实恢复时刻恢复原阶段。
    # 3. 这样所有消费者使用同一时间语义，多次暂停不会重复或遗漏预算。
    # 4. 仅新增私有方法，不增加公共接口。
    def _current_phase_elapsed_ns(self, timestamp_ns: int) -> Optional[int]:
        """返回当前阶段累计活动时间；调用方必须先完成时间合法性检查。"""

        if self.phase_entered_ns is None:
            return None
        return (
            self._phase_elapsed_offset_ns
            + timestamp_ns
            - self.phase_entered_ns
        )

    def _resume_interrupted_phase(self, timestamp_ns: int) -> bool:
        """以真实恢复时间恢复被中断阶段，并继承暂停前累计的活动时间。"""

        if self._interrupted_phase is None:
            return False
        resumed_phase = self._interrupted_phase
        active_elapsed_ns = self._interrupted_elapsed_ns or 0
        self.phase = resumed_phase
        self.phase_entered_ns = timestamp_ns
        self._phase_elapsed_offset_ns = active_elapsed_ns
        self._last_transition_ns = timestamp_ns
        self._interrupted_phase = None
        self._interrupted_elapsed_ns = None
        self._safe_hold_reason = ""
        return True
    # ————————————————————————————————

    # ————————————————————————————————
    # 【Codex修改-13：统一最终失败入口】
    # 1. 修改前各失败路径分别修改状态，容易遗漏暂停、返区或时间上下文清理。
    # 2. 当前统一保存非空原因、保留task和retry_count、清理临时上下文后进入FAILED。
    # 3. 这样所有不可恢复失败都有一致且可诊断的安全终态，不会误设success或删除记录。
    # 4. 仅新增私有方法，不修改FSMStatus、事件或公共方法签名。
    def _enter_failed(self, timestamp_ns: int, reason: str) -> None:
        """以非空原因进入FAILED，同时保留任务和重试诊断信息。"""

        self.failure_reason = (
            reason
            or self.failure_reason
            or self._safe_hold_reason
            or "业务模块报告失败"
        )
        self._fail_after_return = False
        self._interrupted_phase = None
        self._interrupted_elapsed_ns = None
        self._safe_hold_reason = ""
        # ————————————————————————————————
        # 【Codex修改-79：统一由阶段转换重置活动时间偏移】
        # 1. 修改前_enter_failed先把偏移归零，随后_transition又执行一次相同清零。
        # 2. 当前删除前一次重复赋值，只由_transition建立FAILED阶段的全新计时状态。
        # 3. 这样失败入口只有一个偏移重置来源，减少以后两处实现发生偏差的风险。
        # 4. 只清理私有重复操作，不改变FAILED结果、时间戳、原因或公共接口。
        self._transition(GlobalPhase.FAILED, timestamp_ns)
        # ————————————————————————————————
    # ————————————————————————————————

    # ————————————————————————————————
    # 【Codex修改-76：RESET按指定时间重建生命周期基准】
    # 1. 修改前RESET只调用reset，无法在时钟回退后建立新的合法转换时间。
    # 2. 当前先清空全部运行状态，再把WAIT_READY进入时间和最近转换时间设为事件时间。
    # 3. 这样仿真新生命周期可从较小时间重新开始，旧暂停和任务上下文仍被完全清除。
    # 4. 仅新增私有辅助方法，RESET事件名称和handle_event签名不变。
    def _reset_at(self, timestamp_ns: int) -> None:
        self.reset()
        self.phase_entered_ns = timestamp_ns
        self._last_transition_ns = timestamp_ns
    # ————————————————————————————————
