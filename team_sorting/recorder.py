"""Episode 元数据与轻量记录接口。

数据流如下：

机器人运行
→ ``dataset_recorder_node``
→ ``EpisodeRecorder``
→ ``metadata.json`` + FSM/FinalAction JSONL + rosbag
→ 未来离线转换脚本进行时间对齐、清洗、重采样和筛选
→ ACT/VLA 训练格式。

Session 可以理解为一次 Recorder 节点运行，Episode 是训练或评测使用的一段有边界数据；
当前临时实现把节点启动到停止作为一个 Episode，因此其中可能包含多任务、重试和恢复。
本文件只保存原始采集材料，不自行决定正式 Episode 边界，也不直接产出可训练数据集。

rosbag 保存 RGB、Depth、CameraInfo、Odom 和 JointState 等原始高带宽 ROS 消息；Python
不逐帧复制或重新编码图像。JSONL 保存逐周期 FSM 与唯一 FinalAction 遥测，metadata
保存任务、裁判、bag 状态和摘要。FSMStatus 只是阶段/诊断辅助标签；FinalAction 是
ActionMux 输出，不能单独证明官方话题发布成功或机器人实际执行。训练前仍需把语言、
观察、实际状态、动作和 Episode 结果离线对齐，并结合 JointState/Odom 筛选动作。
正常关闭时 JSONL 会完成当前行的刷新；为避免高频写放大，本文件不对每行执行
``fsync``。突然断电可能损坏最后一行，后续离线转换器应容忍末尾不完整的 JSONL；
``topic_counts`` 也只是辅助统计，不是数据完整性证明。

本模块不导入 ``rclpy``，不管理子进程，不实现 ``rosbag2_py``、数据清洗、ACT/VLA
训练或在线推理，也不会把裁判结果复制成逐帧标签。上传到AutoDL等训练环境之前仍需
运行独立的离线转换流程，不能把本目录直接称为可训练数据集。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import count
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Optional, Sequence

from .interfaces import (
    FSMStatus,
    FinalAction,
    TaskSpec,
    final_action_to_json,
    fsm_status_to_json,
)
from .fsm import InstructionParser


_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")
_EPISODE_ID_SEQUENCE = count()


def _require_integer(value: object, name: str, *, nonnegative: bool = False) -> int:
    """严格读取整数，防止 ``int()`` 把bool、浮点数或字符串伪装成合法时间/状态。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}必须是真正整数，不能使用bool、浮点数或字符串")
    if nonnegative and value < 0:
        raise ValueError(f"{name}必须大于等于0")
    return value


def _safe_path_component(value: object, name: str) -> str:
    """校验单级路径名；显式拒绝可指向当前目录或父目录的 ``.`` 与 ``..``。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}必须是非空字符串")
    if value in {".", ".."}:
        raise ValueError(f"{name}不能是'.'或'..'")
    if _SAFE_PATH_COMPONENT.fullmatch(value) is None:
        raise ValueError(
            f"{name}只能包含字母、数字、点、下划线和连字符，不能包含路径分隔符"
        )
    return value


def _resolved_child(parent: Path, component: str, name: str) -> Path:
    """解析子路径并确认它仍位于预期目录内，形成第二道路径逃逸保护。"""

    resolved_parent = parent.resolve()
    resolved_child = (resolved_parent / component).resolve()
    try:
        relative = resolved_child.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValueError(f"{name}解析后逃逸出预期目录：{resolved_child}") from exc
    if not relative.parts:
        raise ValueError(f"{name}不能指向预期目录本身")
    return resolved_child


def _validated_topics(topics: object) -> tuple[str, ...]:
    """校验rosbag参数中的话题并保持调用方顺序，拒绝选项注入和重复配置。"""

    if isinstance(topics, (str, bytes)):
        raise ValueError("rosbag topics必须是话题序列，不能是单个字符串或bytes")
    try:
        values = tuple(topics)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("rosbag topics必须是话题序列") from exc
    if not values:
        raise ValueError("rosbag 至少需要一个话题")
    seen: set[str] = set()
    for index, topic in enumerate(values):
        if not isinstance(topic, str) or not topic:
            raise ValueError(f"rosbag topics[{index}]必须是非空字符串")
        if topic.startswith("-"):
            raise ValueError(f"rosbag topics[{index}]不能以'-'开头：{topic!r}")
        if not topic.startswith("/"):
            raise ValueError(f"rosbag topics[{index}]必须以'/'开头：{topic!r}")
        if "\x00" in topic or any(character.isspace() for character in topic):
            raise ValueError(f"rosbag topics[{index}]不能包含空白字符或NUL：{topic!r}")
        if topic in seen:
            raise ValueError(f"rosbag topics不能重复：{topic}")
        seen.add(topic)
    return values  # type: ignore[return-value]


@dataclass
class EpisodeMetadata:
    """单个比赛 Episode 的元数据摘要。

    参数：Episode 标识、纳秒起止时间、原始/结构化任务、rosbag 状态、裁判原始结果和
    话题计数。``parsed_tasks`` 保存原始指令解析出的全部任务；``task`` 只保留首项作为
    旧消费者兼容字段，不表示 Recorder 选择了执行顺序。任务位置单位米、坐标系沿用
    官方指令；裁判结果只能作为 Episode 摘要，不能复制成逐帧训练标签。
    """

    episode_id: str
    started_at_ns: int
    boundary_policy: str
    instruction_raw: Optional[str] = None
    task: Optional[TaskSpec] = None
    parsed_tasks: tuple[TaskSpec, ...] = ()
    instruction_parse_failure: str = ""
    rosbag_started: bool = False
    rosbag_output: Optional[str] = None
    rosbag_exit_code: Optional[int] = None
    ended_at_ns: Optional[int] = None
    referee_messages: list[dict[str, Any]] = field(default_factory=list)
    final_result: Optional[dict[str, Any]] = None
    topic_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """生成可写入 JSON 的元数据字典。

        参数：无。返回：只含 JSON 兼容值的字典，时间单位纳秒；任务空间量保留原始
        world 语义。失败：字段中出现不可序列化的外部对象时，后续JSON序列化会失败，
        由原子写入边界转换为清晰的 ``RuntimeError``。
        """

        data = asdict(self)
        if self.task is not None:
            data["task"] = asdict(self.task)
        data["parsed_tasks"] = [asdict(task) for task in self.parsed_tasks]
        return data


class EpisodeRecorder:
    """轻量 Episode 记录器。

    参数：``root_dir`` 为可写数据根目录。调用 ``start`` 后可记录 FinalAction/FSM 和
    Episode 级裁判元数据；时间单位纳秒，动作/坐标单位沿用接口定义。目录不可写、重复
    Episode 或未启动就记录时抛出清晰异常，不会覆盖已有 Episode。

    ``topic_counts`` 只是本进程看到的回调次数，用于辅助诊断；高频消息不会每条重写
    metadata，最终值在 ``finish`` 时持久化。异常退出时计数可能不完整，且它不能证明
    rosbag 没有丢消息，更不能作为bag完整性的唯一证据。
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.metadata: Optional[EpisodeMetadata] = None
        self.episode_dir: Optional[Path] = None

    def check_root(self) -> Path:
        """检查并创建数据根目录。

        参数：无。返回：可写的绝对目录路径；无坐标和物理单位。失败：路径是普通文件、
        创建失败或当前进程无写权限时抛出 ``RuntimeError``。
        """

        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"无法创建数据目录 {self.root_dir}: {exc}") from exc
        if not self.root_dir.is_dir():
            raise RuntimeError(f"数据路径不是目录：{self.root_dir}")
        root = self.root_dir.resolve()

        # 目录“存在”不等于当前进程真的能落盘；同目录探针验证创建、写入、关闭和删除。
        probe_fd: Optional[int] = None
        probe_path: Optional[Path] = None
        try:
            probe_fd, raw_path = tempfile.mkstemp(
                prefix=".team_sorting_write_probe_", dir=str(root)
            )
            probe_path = Path(raw_path)
            stream = os.fdopen(probe_fd, "wb")
            probe_fd = None
            with stream:
                stream.write(b"team_sorting recorder write probe\n")
                stream.flush()
            probe_path.unlink()
            probe_path = None
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"数据目录不可可靠写入 {root}: {exc}") from exc
        finally:
            if probe_fd is not None:
                try:
                    os.close(probe_fd)
                except OSError:
                    pass
            if probe_path is not None:
                try:
                    probe_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return root

    def start(
        self,
        episode_id: str,
        started_at_ns: int,
        boundary_policy: str,
        task: Optional[TaskSpec] = None,
    ) -> Path:
        """创建一个新的 Episode 目录并写入初始元数据。

        参数：安全标识、开始纳秒、边界策略说明和可选任务。返回：Episode 目录；任务
        位置单位米且沿用任务坐标系。失败：标识含危险字符、目录已存在、当前记录未结束
        或目录不可写时抛出异常，绝不覆盖旧数据。初始metadata成功落盘后才算启动；
        写失败时仅尝试删除本次创建的空目录，含任何内容的目录都会保留。
        """

        if self.metadata is not None:
            raise RuntimeError("已有 Episode 正在记录，请先 finish")
        episode_id = _safe_path_component(episode_id, "episode_id")
        started_at_ns = _require_integer(
            started_at_ns, "started_at_ns", nonnegative=True
        )
        if not isinstance(boundary_policy, str) or not boundary_policy.strip():
            raise ValueError("boundary_policy必须是非空说明文本")
        root = self.check_root()
        episode_dir = _resolved_child(root, episode_id, "episode_id")
        if episode_dir.exists():
            raise FileExistsError(f"Episode 目录已存在，拒绝覆盖：{episode_dir}")
        try:
            episode_dir.mkdir()
        except OSError as exc:
            raise RuntimeError(f"无法创建 Episode 目录 {episode_dir}: {exc}") from exc
        candidate = EpisodeMetadata(
            episode_id=episode_id,
            started_at_ns=started_at_ns,
            boundary_policy=boundary_policy.strip(),
            task=task,
            parsed_tasks=() if task is None else (task,),
        )
        try:
            # 初始metadata落盘前不提交活动状态，失败后同一对象可安全重试。
            self._write_metadata(candidate, episode_dir)
        except Exception:
            # 只尝试删除本次刚创建且仍为空的目录；并发写入的任何内容都会使
            # rmdir失败并被保留，因此不会递归删除调用前已存在的用户数据。
            try:
                episode_dir.rmdir()
            except OSError:
                pass
            raise
        self.episode_dir = episode_dir
        self.metadata = candidate
        return episode_dir

    def record_final_action(self, action: FinalAction) -> None:
        """追加一条最终动作遥测。

        参数：严格19维 ``FinalAction``；返回：无。这里只记录ActionMux输出的同一个遥测
        对象，不重拼第二份动作。``valid=False`` 可保留用于诊断但不能直接作为专家动作；
        SAFE_HOLD、FAILED和恢复片段也应保留阶段信息，交给离线流程单独标记。

        当前遥测不能证明 ``OfficialCommandPublisher`` 发布成功，也不能证明Server接收
        或机器人执行；训练动作筛选还需发布状态以及JointState/Odom反馈。本记录器不会
        猜测并删除数据。Episode未开始或文件写入失败时抛出异常。
        """

        path = self._active_path("final_actions.jsonl")
        self._append_line(path, final_action_to_json(action))
        self._count("/team/final_action")

    def record_fsm_status(self, status: FSMStatus) -> None:
        """追加一条 FSM 状态遥测。

        参数：``FSMStatus``，时间单位纳秒；返回：无；无坐标系。FSM只提供阶段、重试和
        诊断辅助标签，不是VLA的主要观察或裁判真值。失败：Episode未开始或文件写入失败
        时抛出异常；序列化只调用 ``interfaces.fsm_status_to_json``。
        """

        path = self._active_path("fsm_status.jsonl")
        self._append_line(path, fsm_status_to_json(status))
        self._count("/team/fsm_status")

    def record_instruction(
        self,
        raw_text: str,
        timestamp_ns: int,
        parser: InstructionParser,
    ) -> Optional[TaskSpec]:
        """保存原始任务文本并尝试生成当前 ``TaskSpec``。

        参数：``raw_text`` 必须是真正字符串，``timestamp_ns`` 为非负整数，``parser``
        必须是唯一 ``InstructionParser``。全部解析结果写入 ``parsed_tasks``；返回首项
        只为兼容现有调用签名，不表示Recorder选择了任务顺序。空结果或解析失败会保留
        原文并写入 ``instruction_parse_failure``；文件写入失败仍抛出 ``RuntimeError``。
        """

        if not isinstance(raw_text, str):
            raise ValueError("instruction_raw必须是真正字符串")
        timestamp_ns = _require_integer(
            timestamp_ns, "instruction timestamp_ns", nonnegative=True
        )
        metadata = deepcopy(self._active_metadata())
        # 原文是最不可替代的证据；即使schema变化或解析失败，也必须先保留下来。
        metadata.instruction_raw = raw_text
        metadata.task = None
        metadata.parsed_tasks = ()
        metadata.instruction_parse_failure = ""
        task: Optional[TaskSpec] = None
        try:
            parsed = tuple(parser.parse(raw_text, timestamp_ns))
            if not parsed:
                metadata.instruction_parse_failure = "任务解析结果为空，未生成TaskSpec"
            elif not all(isinstance(item, TaskSpec) for item in parsed):
                metadata.instruction_parse_failure = "任务解析结果包含非TaskSpec对象"
            else:
                # 多条任务全部保存；task仅作旧metadata消费者的首项兼容字段。
                metadata.parsed_tasks = parsed
                task = parsed[0]
                metadata.task = task
        except ValueError as exc:
            metadata.instruction_parse_failure = str(exc)
        self._count("/material/instruction", metadata)
        self._write_metadata(metadata)
        self.metadata = metadata
        return task

    def record_referee_message(self, topic: str, raw_value: str | int, timestamp_ns: int) -> None:
        """记录一条裁判话题作为 Episode 级元数据。

        参数：话题名、String 原文或 Int32 分数、接收纳秒。返回：无；裁判数据没有
        团队坐标转换。永远保存 ``raw``，文本能解析为任意 JSON 时额外保存 ``parsed``；
        普通文本不会报错丢弃。``/referee/score`` 非整数时抛出 ``ValueError``。这些结果
        只用于 Episode 摘要，绝不自动扩展成逐帧图像标签。
        """

        timestamp_ns = _require_integer(
            timestamp_ns, "referee timestamp_ns", nonnegative=True
        )
        metadata = deepcopy(self._active_metadata())
        if topic == "/referee/score":
            raw_value = _require_integer(raw_value, "/referee/score")
        entry: dict[str, Any] = {
            "topic": topic,
            "timestamp_ns": timestamp_ns,
            "raw": raw_value,
        }
        if isinstance(raw_value, str):
            try:
                entry["parsed"] = json.loads(raw_value)
            except json.JSONDecodeError:
                pass
        metadata.referee_messages.append(entry)
        self._count(topic, metadata)
        self._write_metadata(metadata)
        self.metadata = metadata

    def mark_rosbag_started(self, output_path: str | Path) -> None:
        """在外部 rosbag 进程成功创建后更新 Episode 元数据。

        参数：``output_path`` 为 bag 输出目录；返回：无；消息单位和坐标系由 bag 内原始
        ROS 消息保存。Episode 未开始、路径为空或元数据写入失败时抛出异常，不能在进程
        未启动时调用本方法伪造成功。该标记只表示外部进程已成功创建，不证明bag已经
        写入任何消息，也不检查或伪造输出目录存在。单个Episode只允许成功标记一次启动。
        """

        current = self._active_metadata()
        if current.rosbag_started:
            raise RuntimeError("rosbag 已经启动过，当前Episode不支持重复启动")
        if not isinstance(output_path, (str, Path)):
            raise ValueError("rosbag 输出路径必须是字符串或Path")
        output = str(output_path)
        if not output.strip() or "\x00" in output:
            raise ValueError("rosbag 输出路径不能为空")
        metadata = deepcopy(current)
        metadata.rosbag_started = True
        metadata.rosbag_output = output
        metadata.rosbag_exit_code = None
        self._write_metadata(metadata)
        self.metadata = metadata

    def mark_rosbag_finished(self, exit_code: int) -> None:
        """记录外部 rosbag 进程的最终退出码。

        参数：``exit_code`` 为子进程整数退出码；返回：无；无空间单位/坐标系。Episode
        未开始、bag 从未启动或退出码非法时抛出异常。非零退出码会被如实保存，由节点
        继续报告为记录失败；退出码一旦记录便拒绝再次覆盖。metadata写入失败时不提交
        退出码，调用方可用同一退出码重试。
        """

        current = self._active_metadata()
        if not current.rosbag_started:
            raise RuntimeError("rosbag 尚未成功启动，不能记录退出码")
        if current.rosbag_exit_code is not None:
            raise RuntimeError("rosbag 退出码已经记录，拒绝重复覆盖")
        metadata = deepcopy(current)
        metadata.rosbag_exit_code = _require_integer(exit_code, "rosbag_exit_code")
        self._write_metadata(metadata)
        self.metadata = metadata

    def set_final_result(self, raw_json: str, timestamp_ns: int) -> None:
        """把官方最终结果保存为 Episode 级元数据。

        参数：最终结果 JSON 和接收纳秒。返回：无；无逐帧坐标或单位。失败：JSON 非
        对象、Episode 未开始或写入失败时抛出异常。该结果不能替代感知训练标签。
        """

        timestamp_ns = _require_integer(
            timestamp_ns, "final_result timestamp_ns", nonnegative=True
        )
        if not isinstance(raw_json, str):
            raise ValueError("最终结果必须是JSON字符串")
        current = self._active_metadata()
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError("最终结果不是合法 JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("最终结果 JSON 顶层必须是对象")
        metadata = deepcopy(current)
        metadata.final_result = {"timestamp_ns": timestamp_ns, "payload": dict(payload)}
        self._write_metadata(metadata)
        self.metadata = metadata

    def note_topic(self, topic: str) -> None:
        """只增加一个原始话题的 Episode 计数。

        参数：``topic`` 为ROS2话题名；返回：无；不读取数据内容、单位或坐标系。它只
        更新内存中的辅助计数，不逐条写metadata，避免高频写放大；``finish`` 才保证最终
        持久化。异常退出时计数可能不完整，不能据此证明rosbag完整。本方法不在Python
        中复制RGB或Depth。
        """

        self._active_metadata()
        self._count(topic)

    def finish(self, ended_at_ns: int) -> Path:
        """结束当前 Episode 并写回元数据。

        参数：结束时间（纳秒）。返回：元数据 JSON 路径；无空间坐标。失败：未开始、
        结束早于开始或文件写入失败时抛出异常。写入失败时Episode仍保持活动且
        ``ended_at_ns`` 不变，可在环境恢复后重试。正式 Episode 边界仍需与赛事方确认。
        """

        metadata = self._active_metadata()
        ended_at_ns = _require_integer(ended_at_ns, "ended_at_ns", nonnegative=True)
        if ended_at_ns < metadata.started_at_ns:
            raise ValueError("Episode 结束时间不能早于开始时间")
        candidate = deepcopy(metadata)
        candidate.ended_at_ns = ended_at_ns
        # 只有原子metadata更新成功后才释放活动状态；失败时调用方仍可诊断或重试。
        path = self._write_metadata(candidate)
        self.metadata = None
        self.episode_dir = None
        return path

    def build_rosbag_command(
        self,
        topics: Sequence[str],
        output_name: str = "rosbag",
    ) -> tuple[str, ...]:
        """生成但不执行 ``ros2 bag record`` 命令。

        参数：需要原样记录的话题列表和输出子目录名。返回：可交给 ``subprocess`` 的
        参数元组；消息保持自身单位/坐标系。失败：Episode 未开始、ros2 不在 PATH、话题
        为空、重复、像命令选项或输出名不安全时抛出异常；话题顺序保持不变。本方法不
        执行命令，也不实现 ``rosbag2_py``。
        """

        episode_dir = self._active_directory().resolve()
        output_name = _safe_path_component(output_name, "rosbag output_name")
        validated_topics = _validated_topics(topics)
        output = _resolved_child(episode_dir, output_name, "rosbag output_name")
        if output.exists():
            # 主动拒绝比依赖ros2的覆盖策略更可预测，也避免混入上一次bag残留。
            raise FileExistsError(f"rosbag 输出路径已存在，拒绝覆盖：{output}")
        if shutil.which("ros2") is None:
            raise RuntimeError("未找到 ros2 命令，无法启动 ros2 bag；请先 source ROS2 环境")
        return ("ros2", "bag", "record", "-o", str(output), *validated_topics)

    @staticmethod
    def make_episode_id(prefix: str = "episode") -> str:
        """按 UTC 时间生成可读 Episode 标识。

        参数：安全前缀；返回：UTC微秒时间加进程内递增后缀的安全字符串；无坐标系。
        后缀用于降低同一进程快速连续创建时的碰撞，不声称跨机器全局唯一。失败：前缀含
        路径或特殊字符时抛出 ``ValueError``。
        """

        prefix = _safe_path_component(prefix, "Episode prefix")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        episode_id = f"{prefix}_{stamp}_{next(_EPISODE_ID_SEQUENCE):06d}"
        return _safe_path_component(episode_id, "Episode ID")

    def _active_metadata(self) -> EpisodeMetadata:
        if self.metadata is None:
            raise RuntimeError("Episode 尚未 start")
        return self.metadata

    def _active_directory(self) -> Path:
        if self.episode_dir is None:
            raise RuntimeError("Episode 尚未 start")
        return self.episode_dir

    def _active_path(self, name: str) -> Path:
        return self._active_directory() / name

    def _count(
        self,
        topic: str,
        metadata: Optional[EpisodeMetadata] = None,
    ) -> None:
        metadata = self._active_metadata() if metadata is None else metadata
        metadata.topic_counts[topic] = metadata.topic_counts.get(topic, 0) + 1

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        """追加单行遥测，正常关闭会刷新，但不为每行 ``fsync``。

        这避免控制周期内的高频写放大；突然断电可能留下不完整的最后一行，离线
        转换器必须容忍该情况。
        """

        try:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.write("\n")
        except OSError as exc:
            raise RuntimeError(f"无法写入记录文件 {path}: {exc}") from exc

    def _write_metadata(
        self,
        metadata: Optional[EpisodeMetadata] = None,
        episode_dir: Optional[Path] = None,
    ) -> Path:
        """原子写入指定metadata，但不替调用方提交内存状态。

        先内存序列化，再将同目录临时文件 ``flush``/``fsync`` 后用 ``os.replace``
        切换可见版本。当前未对父目录执行 ``fsync``，因此主要防范进程异常和半文件，
        不声称在所有文件系统上完全抗突然断电。
        """

        selected_metadata = self._active_metadata() if metadata is None else metadata
        selected_directory = (
            self._active_directory() if episode_dir is None else episode_dir
        )
        path = selected_directory / "metadata.json"
        temporary_fd: Optional[int] = None
        temporary_path: Optional[Path] = None
        try:
            # 先在内存中完成序列化；对象不兼容时不会提前截断上一份有效metadata。
            payload = json.dumps(
                selected_metadata.to_dict(), ensure_ascii=False, indent=2
            ) + "\n"
            temporary_fd, raw_path = tempfile.mkstemp(
                prefix=".metadata_", suffix=".tmp", dir=str(path.parent)
            )
            temporary_path = Path(raw_path)
            stream = os.fdopen(temporary_fd, "w", encoding="utf-8")
            temporary_fd = None
            with stream:
                stream.write(payload)
                stream.flush()
                # fsync让已关闭前的内容尽量落盘，再由同目录原子替换切换可见版本。
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"无法原子写入元数据 {path}: {exc}") from exc
        finally:
            if temporary_fd is not None:
                try:
                    os.close(temporary_fd)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return path
