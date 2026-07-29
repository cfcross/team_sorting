# team_sorting 协作规则

## 1. 文件目的与适用范围

- 本文件适用于整个 `team_sorting` 仓库，约束团队成员和自动化代码工具的修改行为。
- `README.md` 负责解释架构和当前状态；本文件只规定开发边界、修改流程和验收标准。修改前必须完整阅读目标文件、`interfaces.py`、直接调用方/被调用方及相关测试。
- 无法从仓库或官方环境确认的事实必须报告为“待确认”，禁止猜测。
- 下文“必须”“禁止”是合并前硬性条件；“建议”可在说明理由后调整。

## 2. 事实类型与来源

按事实类型选择对应来源，不对不同类型的文档作统一排名：

- 仓库现有行为：以当前代码和测试为准。
- 开发约束：以 `AGENTS.md` 为准。
- 架构说明和完成状态：以 `README.md` 为准。
- 运行参数：以 `config/config.yaml` 为准。
- 比赛任务、裁判、消息、话题和坐标协议：正式规则、官方 Server 源码和实际 ROS 环境是最终依据。

- 文档、代码、测试或配置互相冲突时，必须停止相关修改并报告冲突，禁止自行选一个解释。
- 官方规则未确定时，禁止把临时结论写死到接口、配置或业务算法中。
- 禁止仅凭个人推测修改话题、坐标系、关节名称、裁判逻辑或任务边界。

## 3. 冻结的架构内容
### 冻结范围

核心生产模块结构、公共接口、节点职责和控制链已经冻结。

负责人可以在所属业务文件内实现预留算法；新增或修改公共业务模块、
公共接口、ROS节点入口和控制链必须先经队长确认。

允许经评审新增不改变业务架构的协作基础设施，包括：

- `.gitignore`
- `LICENSE`
- `CONTRIBUTING.md`
- `.github/`
- `CODEOWNERS`
- Issue和Pull Request模板
- CI配置
- `docs/`文档

未经批准，禁止新增第二套公共接口、manager、factory、自定义ROS消息包
或绕过现有控制链的新业务目录。
未经全队评审，禁止修改：

- 23 文件目录结构（含经队长批准的 `team_sorting/external_candidate.py` 和
  `team_sorting/controller_manifest.py`）；
- `perception_node`、`team_client_node`、`dataset_recorder_node` 三个 ROS2 节点；
- 十个业务文件的职责边界；
- 公共接口集中在 `interfaces.py`；
- `InstructionParser` 是任务 JSON 唯一解析入口，`ActionMux` 是动作唯一仲裁入口，`OfficialCommandPublisher` 是官方控制话题唯一出口；
- `FinalAction` 固定 19 维及其索引顺序；
- Recorder 作为旁路观察者，不参与 FSM 或控制；
- 除 `ros_nodes.py` 外，业务算法文件不直接依赖 `rclpy`；
- 官方能力只通过薄适配器调用，不复制官方源码、权重或资源。

架构冻结不等于算法冻结。负责人可以在所属文件内实现预留算法，但不得改变公共边界。

## 4. 文件负责人和修改范围

同一文件可由多人评审，但只有第一负责人或队长可以批准合并。

| 文件或目录 | 第一负责人 | 允许修改内容 | 禁止越界内容 |
|---|---|---|---|
| `team_sorting/interfaces.py` | 架构/系统 | dataclass、枚举、固定顺序、唯一 JSON 格式 | ROS 转换、业务算法、记录逻辑 |
| `team_sorting/perception_2d.py` | 视觉1 | 以 `RGBFrame` 为主要输入的官方 YOLO 适配、二维稳定和可选任务筛选 | Depth、CameraInfo、Odom、实际关节、三维外参、ROS 发布 |
| `team_sorting/perception_3d.py` | 视觉2 | `Detection2D`、深度、内参、底盘状态、实际关节参与的三维估计 | YOLO、目标抓取决策、ROS 同步 |
| `team_sorting/navigation.py` | 底盘 | 站位、航点、精对准、速度控制、区域分类 | 发布 `/cmd_vel`、推进 FSM、机械臂 IK |
| `team_sorting/arm_planning.py` | 机械臂1 | 官方 KDL 适配、抓放位姿、IK、轨迹规划 | 轨迹执行、局部状态机、官方话题发布 |
| `team_sorting/arm_execution.py` | 机械臂2 | 轨迹插值、局部状态、试抬、验证与恢复 | 重新实现 IK、全局任务选择、官方话题发布 |
| `team_sorting/fsm.py` | 系统/FSM | 唯一任务解析、状态转换、重试与失败路径 | ROS 发布、导航或机械臂算法 |
| `team_sorting/action_mux.py` | 控制安全 | 19 维仲裁、TTL、限幅、安全保持 | 轨迹规划、FSM 决策、ROS 发布 |
| `team_sorting/controller_manifest.py` | 架构/控制安全 | 从`ACTION_NAMES`派生的版本化控制元数据、运行时范围与配置一致性校验 | ROS发布、动作仲裁、执行确认、第二套动作顺序 |
| `team_sorting/external_candidate.py` | 控制安全/ROS 集成 | 默认关闭的外部Candidate严格解码、身份/新鲜度/TTL/one-shot安全消费及现有`ManipulationCommand`转换 | 导入rclpy、发布ROS、生成FinalAction、修改FSM、绕过ActionMux、接收pi05原生8维动作 |
| `team_sorting/recorder.py`、`team_sorting/recording_contracts.py` | 数据 | Episode 元数据、团队遥测、动作/dispatch严格配对、裁判原文、rosbag 命令辅助 | 参与控制、训练模型、逐帧复制图像 |
| `team_sorting/ros_nodes.py` | ROS 集成 | 三节点 I/O、缓存、转换、组装、唯一官方发布 | YOLO、导航、IK、轨迹算法 |
| `config/config.yaml` | ROS 集成 | 可部署参数、话题、限幅和记录配置 | 写入未确认规则或开发者绝对路径 |
| `launch/team.launch.xml` | ROS 集成 | 三节点启动及现有启动参数 | 新增架构节点或业务算法 |
| `README.md` | 架构/系统 | 架构、接口、运行状态和待确认事实 | 宣称未实现能力已完成 |
| `AGENTS.md` | 架构/系统 | 开发约束、流程和验收标准 | 改写代码事实或外部比赛规则 |
| `package.xml`、`setup.py`、`setup.cfg`、`resource/` | 架构/系统 | 包元数据、依赖、安装和入口 | 业务逻辑；删除 ament 资源标记 |
| `tests/test_geometry_and_planning.py` | 视觉2/底盘/机械臂1 | 各自负责的接口、几何和规划回归 | 用假成功掩盖未实现算法 |
| `tests/test_fsm_mux_recording.py` | 系统/控制安全/数据 | FSM、动作、JSON、TTL、记录回归 | 放宽安全断言或伪造官方环境结果 |

## 5. 强制接口规则

- 公共数据必须使用 `interfaces.py` 中的类型。
- 禁止在业务文件中重新定义 Task、Pose、Command、Status 或 Action 结构。
- 新增公共字段必须注明单位、`frame_id`、时间戳和失败语义；不适用时也要明确说明。
- `RobotJointState` 是实际反馈；`IKResult` 是目标关节解。禁止混用或使用含义模糊的变量名。
- `ObjectEstimate3D` 是物体中心估计；`place_world_xyz` 是目标物体中心。两者都不是夹爪末端位姿。
- `BaseCommand` 和 `ManipulationCommand` 只是候选建议；`FinalAction` 是 `ActionMux` 的输出。
- `FinalAction.values` 必须严格包含 19 项，禁止复制、重排或重新定义索引。
- `ActionMuxDecision`只能由同一次ActionMux仲裁产生；禁止从`FinalAction`或failure_reason反推mask。
- `ActionDispatchRecord`只能在官方publisher边界记录精确payload；本地调用成功不得提升为
  controller accepted或execution confirmed，未发送维度必须为null而不是补零。

19 维顺序只作边界提示，完整定义以 `interfaces.ACTION_NAMES` 为准：

```text
0–1 base | 2 slide | 3–4 head | 5–11 left arm + gripper | 12–18 right arm + gripper
```

完整字段、单位和坐标说明查阅 `interfaces.py` 和 `README.md`，不要在本文件另建第二份说明。

## 6. 模块依赖边界

```text
interfaces
  ↑
  ├─ perception_2d / perception_3d / navigation / arm_planning
  └─ arm_execution / fsm / controller_manifest / action_mux / recording_contracts / recorder
ros_nodes 组装上述模块并连接 ROS2
```

- `perception_3d` 禁止依赖 `perception_2d` 的内部实现，只通过 `Detection2D` 连接。
- `arm_execution` 禁止依赖 `arm_planning` 的内部实现，只通过 `JointTrajectory` 连接。
- `fsm` 禁止发布 ROS2 消息；`navigation` 禁止直接发布 `/cmd_vel`；机械臂模块禁止直接发布官方关节话题。
- `recorder` 禁止影响 FSM、候选命令或 `FinalAction`。
- `ros_nodes.py` 禁止实现 YOLO、导航、IK 或轨迹算法。
- 除 `ros_nodes.py` 外，业务模块禁止导入 `rclpy`。

## 7. 安全与失败规则

- `team_client_node` 默认必须处于全局 observe-only 模式：可以计算并发布团队诊断遥测，
  但不得创建或调用 `OfficialCommandPublisher`。External Candidate 的门不能替代此门。
- 只有 `observe_only=false`、`enable_official_publish=true`、`simulation_only=true` 同时
  成立才允许创建唯一官方发布器；observe-only优先关闭，非仿真配置必须拒绝。
- Stage 2A head controller-target shadow默认关闭，只限本节点与官方仿真Server共同启动并
  明确确认fresh reset、初值严格为`[0.0, 0.0]`且head话题唯一写入者的受控场景。
  `/joint_states`是物理反馈，禁止据此恢复未知controller target；节点单独重启、Server
  未reset或出现其他head publisher时必须重新授权并fail closed。
- 当前Stage 2A退出路径只能取消timer、清空External Candidate pending并销毁ROS实体；
  禁止用JointState拼接全量hold，禁止自动发布cmd_vel或任何关节controller target。
  emergency base stop接口仅供未来明确授权的底盘故障路径，不能由普通destroy调用。
- “没有机械臂业务候选”必须表示为 `ManipulationCommand=None`，不能默认转换为17维
  `controlled_mask=True`的主动位置保持。显式主动保持能力可以保留，但必须由授权场景调用。
- 官方依赖、资源或消息字段缺失时必须清晰失败；未实现算法必须抛出明确的 `NotImplementedError`，或返回 `valid/success=False` 及原因。禁止伪造结果或静默降级。
- 禁止用全零关节目标表示机械臂安全停止。
- 底盘命令过期后必须归零；机械臂命令过期后必须用实际反馈安全保持。
- `SAFE_HOLD` 和 `FAILED` 必须覆盖普通控制建议。
- 实际关节反馈无效时，禁止向官方控制话题发布有效动作。
- 超时不能自动跳入下一成功阶段；生成目标或命令也不能作为成功依据。
- FSM 状态推进必须依据真实业务反馈。
- 新增空间数据必须注明 frame 和单位，禁止默认 `world == odom`。
- 深度换算比例必须来自配置，禁止根据编码字符串随意猜测。

## 8. FinalAction 和数据记录规则

- `ActionMux` 每个控制周期只能生成一次 `FinalAction`。
- 官方控制拆分和 `/team/final_action` 遥测必须使用同一对象。
- 每个已生成`FinalAction`的控制周期还必须发布一条稳定关联的`/team/action_dispatch`；
  observe-only也必须记录none模式，但该团队话题不得触发任何官方控制。
- 禁止为 Recorder 重新拼接第二份 19 维动作。
- `valid=False` 的 `FinalAction` 可以记录用于诊断，但禁止直接作为专家训练动作。
- 只有 `valid=True` 且进入 `OfficialCommandPublisher` 发布链路的动作，才可视为候选专家动作。
- 是否实际执行必须结合 JointState、Odom 等反馈判断。
- `SAFE_HOLD`、`FAILED` 和恢复片段必须保留独立 phase 或标签。
- 裁判结果只作为 Episode 级元数据，禁止扩展成逐帧真值。
- 原始 RGB、Depth 必须由 rosbag 持久化。
- Recorder 禁止为了落盘逐帧复制或重新编码完整图像。
- `perception_node` 为推理和几何计算进行必要的图像转换不受上述限制。
- Recorder 只负责采集，禁止在其中实现 ACT/VLA 训练或控制决策。
- Recorder 配对只能按严格 sequence 和 timestamp 关联 FinalAction/ActionDispatch；不得
  邻近猜测、补造缺失侧或把配对成功提升为 controller 接受/机器人执行确认。
- Recorder 配对状态只能在对应 raw/Frame/Issue JSONL 追加成功后提交；写入失败必须保留
  可重试上下文，终态 sequence 即使近期 digest 淘汰也不得在同一 Episode 重开。

## 9. 禁止硬编码

- 禁止写死随机任务目标、位置和完整动作序列，包括固定 pink/yellow/brown 的执行顺序；禁止假设一次运行必然连续完成多个任务。
- 禁止仅凭 `task_id` 绕过 `TaskSpec`、感知结果和实际反馈来选择目标或动作。
- 允许固定安全姿态、home 姿态、标定偏移和机械限位，但必须放入配置或集中常量，并注明来源、单位和适用条件。
- 允许根据任务类型选择策略，但具体目标必须来自 `TaskSpec`、感知结果和正式规则。
- 禁止把临时裁判规则写入业务算法。
- 禁止在 Python 中重复写入 `config/config.yaml` 已有的话题和参数。
- 禁止复制官方 YOLO、KDL、MJCF、权重、场景或裁判源码。
- 禁止用颜色分割、全零数据或 GT 结果伪装正式算法成功。
- 调试后端必须通过显式配置启用，禁止静默降级。

## 10. 公共接口修改流程

修改以下公共文件前，必须先获得队长或第一负责人确认：

`interfaces.py`、`action_mux.py`、`ros_nodes.py`、`config/config.yaml`、
`launch/team.launch.xml`、`package.xml`、`setup.py`、`setup.cfg`。

修改前必须说明：

1. 当前问题；
2. 为什么不能只在负责人业务文件内解决；
3. 需要修改的公共接口；
4. 受影响的生产者和消费者；
5. 是否向后兼容；
6. 需要同步修改的测试；
7. `README.md` 或 `AGENTS.md` 是否需要更新。

公共接口改变后，必须同步所有调用方、相关测试、README 接口说明、AGENTS 约束及相关配置/launch。

## 11. 单文件开发规则

- 只修改任务明确指定的文件，并阅读其直接调用方、被调用方和测试。
- 禁止顺手修改无关文件、执行未要求的重构或扩大任务范围。
- 禁止修改公共接口来迁就单文件局部实现。
- 禁止新增 manager、factory、工具目录、自定义消息或新业务目录。
- 禁止删除 `NotImplementedError` 后返回伪成功值。
- 发现范围外问题时，只在最终报告中列出。
- 确需跨文件修改时，必须先报告影响范围并获得确认。

## 12. 测试与验收

所有 Python 修改至少运行：

```bash
python3 -m compileall -q team_sorting tests
PYTHONDONTWRITEBYTECODE=1 \
python3 -m pytest -q -p no:cacheprovider
```

涉及 ROS 包、入口、配置或 launch 时，还必须在 ROS2 环境运行：

```bash
colcon list
```

条件允许时运行：

```bash
colcon build --packages-select team_sorting
```

涉及官方 YOLO、MMK2FK、KDL、话题或关节名称时：

- 必须在官方镜像中执行自检；本地单元测试不能代替官方环境验证。
- 无法运行官方环境时，最终报告必须明确写“未验证”及原因。

禁止为通过测试而删除断言、放宽关键安全条件、修改测试数据掩盖错误或把失败改成伪成功。

## 13. 提交与合并检查

提交前必须确认：

- `git diff` 只包含计划内文件，且没有新增缓存或临时文件；
- 未提交 `build/`、`install/`、`log/`、数据集、权重、rosbag 或官方源码；
- 公共接口改动已同步调用方、测试和文档；
- `README.md` 与实际代码一致，且未绕过本文件约束；
- 测试结果已记录，未验证的官方环境内容已明确标注。

最终报告必须包含：

1. 修改文件；
2. 修改原因；
3. 是否改变公共接口；
4. 是否改变运行逻辑；
5. 测试命令和结果；
6. 未执行的测试及原因；
7. 仍需负责人确认的事项。

## 14. 外部规则与待确认事实

- 当前任务变体规则以赛事方正式发布为准。
- 正式任务开始/结束、裁判结算和允许重试规则未确认时，禁止写死到比赛业务逻辑。
- 团队可以定义版本化的内部 Episode 边界用于 VLA 数据采集，但必须明确标注为团队规范，并保持可配置、可调整。
- 话题、坐标系、关节名称、资源路径和 watchdog 必须优先在官方镜像中验证。
- 正式规则或官方接口变化后，由队长统一更新 README、AGENTS、配置和测试。
- 禁止个人依据群聊片段单独修改公共协议。
