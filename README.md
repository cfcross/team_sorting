# team_sorting

## 1. 项目一句话介绍

这是团队自己的 ROS 2（机器人软件通信框架）竞赛客户端仓库，不是官方比赛 Server
仓库：它读取官方任务和传感器数据，组织感知、导航与 MMK2 机械臂模块，把各模块建议
统一仲裁为 `FinalAction[19]`，再通过官方 ROS 2 控制话题发给机器人；同时可以在控制链
旁路记录后续 ACT（动作分块 Transformer）/VLA（视觉—语言—动作模型）需要的数据。

> 当前定位是“可导入、可测试、边界清晰的客户端骨架”，不是已经能完成比赛的成品。
> 真实相机/YOLO/MMK2FK 联调、导航参数实测标定与全局路径规划、抓放规划、机械臂执行和比赛闭环仍未完成。

### Stage 2.2 Interface v1

公共接口的机器可读冻结入口是 [`config/contracts/interface_v1.json`](config/contracts/interface_v1.json)，中文说明见 [`docs/interface_v1.md`](docs/interface_v1.md)。当前冻结的是名称、顺序、单位、身份及证据边界；`provisional`/`restricted` 项和 UNRESOLVED 在线事实仍保持显式，Interface对齐不代表算法或比赛闭环完成。

### Stage 2.3 Data/TF Policy v1

B3数据、TF/QoS、raw/derived边界、采集profile和训练资格的独立机器契约见
[`config/contracts/data_tf_policy_v1.json`](config/contracts/data_tf_policy_v1.json)，中文投影见
[`docs/data_tf_policy_v1.md`](docs/data_tf_policy_v1.md)。B3B第一步已把`/tf`和`/tf_static`
加入raw rosbag，并使用安装资源中的显式QoS override；压缩、降频和训练样本生成仍未实现，
只读Indexer/QC见下一节。

### Stage 2.3 Dataset Index/QC v1

B3C core的机器契约、只读Indexer和基础QC入口分别见
[`config/contracts/dataset_index_v1.json`](config/contracts/dataset_index_v1.json)、
[`docs/dataset_index_v1.md`](docs/dataset_index_v1.md)和`team_sorting_dataset_index`。
当前只生成四类derived索引/QC输出；不修改raw，不生成sample或训练manifest，formal BC资格
在feedback时间窗实现前不会被判为`eligible`。

### Stage 2A External Candidate Consumer

当前代码包含对 adapter 提交
`5fc0f2699645bc3735422e0841383b38a95d4b38` 的版本化 ROS String Candidate 的
默认关闭 consumer。它只接受显式 Trigger 产生的 `fixed_head_yaw` 安全 Candidate，
严格复核 JSON、任务身份、指令/JointState 新鲜度、ROS 时间、TTL、generation、request
去重、单周期 delta、速度和临时仿真限位，再一次性转换为现有
`ManipulationCommand`。pi05 原生 8D 动作没有被使用或映射。

`enabled`、`enable_actuation` 和临时第三道门 `simulation_publish_enabled` 默认均为
`false`；默认配置不创建外部 Candidate 订阅，也不会让它进入 ActionMux。即使显式启用，
Candidate 也不能绕过现有 `ActionMux -> FinalAction[19] -> OfficialCommandPublisher`
链路。当前阶段只完成 consumer 代码与测试，不代表仿真微动已经完成；head-yaw 临时边界
也不是官方物理限位，正式运动前仍须在官方镜像验证。

External Candidate三道门之外还有独立的全局官方发布门。默认
`control.observe_only=true`、`control.enable_official_publish=false`、
`control.simulation_only=true`：节点可以订阅状态、运行FSM和ActionMux，并发布
`/team/fsm_status`、`/team/final_action`、`/team/action_dispatch`诊断遥测，但不会创建
`OfficialCommandPublisher`或五组官方控制publisher，退出时也不会发布紧急底盘命令。
只有显式同时设置`observe_only=false`、`enable_official_publish=true`且
`simulation_only=true`才会创建官方发布器；observe-only优先关闭发布，
`simulation_only=false`在当前阶段直接拒绝启动。运行时覆盖应通过
`TEAM_SORTING_CONFIG`指向一份非持久配置，不能改写仓库默认值。

Stage 2A的head发布还受`control.head_target_tracking`约束。该shadow默认关闭，只能在
官方仿真Server与本节点共同全新启动、已确认刚完成reset、初始绝对controller target
严格为`[0.0, 0.0]`且head话题没有其他publisher时临时启用。`/joint_states`是物理反馈，
不能用于恢复或覆盖未知controller target；yaw-only发布复用shadow中的pitch target，
而不是回写当前pitch反馈。节点重启、Server未重启、reset身份不确定或出现其他head
writer时必须重新授权并fail closed。通用安全恢复仍需官方Server暴露controller target。
无论全局发布门是否开启，当前Stage 2A节点退出都只取消control timer、清空pending并
销毁ROS实体，不发布cmd_vel或任何关节controller target；JointState反馈不能作为退出
安全保持目标。`publish_emergency_base_stop()`仅保留给未来明确授权的底盘故障路径。

防重放状态在 Candidate 成功进入单元素 pending 槽时即生效：generation 在此时绑定、
request ID 在此时记为已使用。即使 Candidate 随后在控制周期因 JointState、delta 或临时
限位检查失败，这两项状态也不会回退或允许重放；这只是 fail-closed 防重放语义，不表示
Candidate 已执行、已发布或已被机器人采用。

## 2. 官方仓库与团队仓库的边界

官方工程提供场景、裁判、传感器、机器人模型、YOLO、运动学和控制入口；团队仓库负责把
这些能力组织成可协作、可测试的客户端。这里采用“薄适配器”：只检查依赖、转换输入、
调用官方能力、转换输出，不在本仓库重写官方算法。

| 团队文件 | 官方来源 | 复用能力 | 团队适配接口 | 当前状态 |
|---|---|---|---|---|
| `perception_2d.py` | 官方 `backends.py`、YOLO 权重 | 加载官方检测器并执行二维检测 | `OfficialYoloAdapter.self_check()` / `detect()` | 检测转换、稳定轨迹 ID 与回归测试完成，正式推理待联调 |
| `perception_3d.py` | DISCOVERSE 的 `MMK2FK`、官方 `box_detect.py` 的针孔反投影思路 | 根据实际底盘和关节状态获得相机外参 | `CameraTransformProvider.self_check()` / `camera_to_output()` | 深度中位数、反投影、尺寸中心补偿、独立局部尺寸输出、稳定 ID EMA 与时帧校验已实现；实时物体姿态和真实 ROS/FK 待联调 |
| `arm_planning.py` | 官方 `MMK2Kdl` / `ArmKdl` | FK 自检和固定 slide 的 IK（逆运动学：由末端目标求关节目标） | `OfficialKDLAdapter.self_check()` / `solve_ik()` | 导入和自检骨架完成，抓放规划待实现 |
| `fsm.py` | 官方 `/material/instruction` JSON | 接收官方任务字段，但解析规则由团队集中维护 | `InstructionParser.parse()` | 已实现并有测试 |
| `ros_nodes.py` | 官方 `/joint_states` | 按名称把反馈映射成团队固定的 17 维实际关节顺序 | `JointStateMapper.map_message()` | 已实现，待官方环境核对关节名称 |
| `ros_nodes.py` | 官方五组控制话题 | 把一个 `FinalAction[19]` 拆成底盘、slide、头部和左右臂命令 | `OfficialCommandPublisher.publish()` | 拆分接口完成，待官方环境端到端验证 |

边界规则：

- 不复制官方 YOLO、KDL、场景或裁判实现，也不复制权重和数据集。
- 官方场景和裁判仍由官方 Server 运行；团队代码只订阅它们公开的话题和消息。
- `InstructionParser`、`JointStateMapper` 等是团队侧协议转换，不代表复制了 Server 逻辑。
- 官方依赖、模型、权重、MJCF 或消息字段缺失时，适配器必须清楚报出模块、路径或字段
  问题；不能用颜色分割、全零值或假结果伪造成功。

## 3. 整体架构图

```text
官方 Server / DISCOVERSE / MMK2
  ├─ RGB + 对齐 Depth + CameraInfo + Odom + JointState
  │                         │
  │                         v
  │                  perception_node
  │                         │
  │             /team/object_estimates
  │                         │
  └─ /material/instruction ─┴───────────────┐
                                             v
                                     team_client_node
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    v                        v                        v
               GlobalFSM              Navigation              Arm Planning
                                                                + Arm Execution
                    └────────────────────────┬────────────────────────┘
                                             v
                                         ActionMux
                                             │
                                      FinalAction[19]
                                             │
                                OfficialCommandPublisher
                                             │
                       /cmd_vel + slide + head + 左臂 + 右臂
                                             │
                                             v
                                      官方机器人控制话题

旁路记录（不参与上面的控制决策）：

官方原始话题 + /team/object_estimates + FSMStatus + FinalAction + 裁判话题
                              │
                              v
                   dataset_recorder_node
                              │
                 rosbag + JSONL + metadata
```

图中 Arm Planning 和 Arm Execution 仍主要表示目标架构。`team_client_node` 已在
`NAV_TO_PICK`、`NAV_TO_PLACE`、`RETURN_END` 阶段把阶段入口生成的同一 `NavGoal`、
新鲜 Odom、`NavigationController.update()`、`BaseCommand`、`ActionMux` 和真实
`NavigationStatus` 串联；非导航阶段仍生成底盘零速候选，且不生成主动17维关节保持候选。
默认 observe-only 全局门仍禁止把 `FinalAction` 发布到官方控制话题。

## 4. 三个 ROS2 节点

ROS 2 节点可以理解为一个独立运行、通过话题交换消息的程序。本仓库只有三个节点入口，
都集中在 `ros_nodes.py`。

### `perception_node`

- **订阅**：RGB `/head_camera/color/image_raw`、对齐深度
  `/head_camera/aligned_depth_to_color/image_raw`、CameraInfo
  `/head_camera/color/camera_info`、Odom `/slamware_ros_sdk_server_node/odom`、
  `/joint_states`。RGB 与 Depth 近似同步；Odom 和 JointState 取时间上最近且未超出容差的值。
- **发布**：`/team/object_estimates`，类型为
  `vision_msgs/msg/Detection3DArray`，载荷包含类别、稳定身份（可用时）、三维中心、
  可选姿态/尺寸、置信度、时间和坐标系；`slot_type` 不通过该消息传输。团队内部适配
  约定以零四元数表示“姿态不可用”、全零 `bbox.size` 表示“尺寸不可用”，接收时恢复
  为 `None`；零四元数绝不归一化为单位姿态。非零 `bbox.size` 必须同时把
  `bbox.center` 写成与首个结果 pose 相同的中心和姿态（四元数正负号可等价），不一致
  的内部消息失败关闭。以上是团队私有载荷契约，不是官方 `vision_msgs` 协议；未知姿态
  加已知尺寸也不被表述为标准有效的 oriented bbox。
- **调用**：`perception_2d.py` 的 `OfficialYoloAdapter` 与
  `Detection2DStabilizer`，`perception_3d.py` 的 `CameraTransformProvider` 与
  `Perception3DEstimator`，以及 `interfaces.py` 中的感知接口。
- **不负责**：解析任务、目标选择、导航、机械臂规划、动作仲裁或官方控制发布。
- **是否必需**：默认 launch 会启动；完整比赛链路需要它提供三维目标。纯 Python
  三维估计与失败语义已有回归测试，但尚未在正式 ROS、相机、YOLO 与 MMK2FK
  组合环境完成端到端验证，所以仍不能声称感知链已经闭环。

### `team_client_node`

- **订阅**：`/material/instruction`、`/slamware_ros_sdk_server_node/odom`、
  `/joint_states`、`/team/object_estimates`。
- **发布**：团队遥测 `/team/fsm_status`、`/team/final_action`、`/team/action_dispatch`；只有全局发布门开启时，
  才通过唯一的 `OfficialCommandPublisher` 发布五组官方控制话题。默认不创建该实例。
- **调用**：当前实际调用 `fsm.py`、`action_mux.py`，并保留 `arm_execution.py` 的显式
  主动保持能力，以及 `navigation.py` 的区域分类；完整架构还应
  在这里按 FSM 阶段组织导航、抓放规划和执行。
- **不负责**：YOLO、深度图处理、相机外参算法、KDL 实现或数据落盘。
- **是否必需**：默认 launch 会启动；默认只观察和发团队遥测。只有全局发布门明确开启
  时，它才可以经`OfficialCommandPublisher`发官方控制话题。

`team_client_node` 每个控制周期构造的 `SensorSnapshot` 只含任务、底盘状态、实际关节
状态和三维目标，**不含 RGB、Depth 或 CameraInfo**。图像留在感知节点和 rosbag，避免
把高带宽数据复制进控制循环。

### `dataset_recorder_node`

- **订阅**：`/team/final_action`、`/team/action_dispatch`、`/team/fsm_status`、
  `/material/instruction`，以及
  `/referee/taskinfo`、`/referee/gameinfo`、`/referee/score`。配置中的原始传感器和
  `/team/object_estimates` 由它管理的外部 `ros2 bag record` 进程订阅。
- **发布**：无控制话题；输出是 dataset root 下的 bootstrap/run-bound Recorder
  Segment、run manifest、事件流、原有 JSONL/`metadata.json` 和每段独立 `rosbag/`。
- **调用**：`recorder_runtime.py` 的 `RecorderRuntimeManager` 管理身份、落盘和 bag
  生命周期；段内原始兼容文件仍由 `recorder.py` 的 `EpisodeRecorder` 写入。
- **不负责**：感知、控制、状态推进、训练 ACT/VLA，或把裁判结果扩展成逐帧标签。
- **是否必需**：可选，`recorder.enabled` 默认是 `false`，普通启动不创建该节点。

Recorder 是旁路观察者。即使不启动它，控制链也应独立工作；它的输出或故障不能作为
机器人动作决策的输入。

## 5. 十个业务文件

| 文件 | 主要职责 | 输入 | 输出 | 主负责人 |
|---|---|---|---|---|
| `interfaces.py` | 公共 dataclass、枚举、固定动作顺序和唯一 JSON 格式 | 各模块的结构化数据 | `TaskSpec`、状态、命令、`FinalAction` 等统一对象 | 架构/系统 |
| `perception_2d.py` | 官方 YOLO 薄适配、二维框稳定接口 | 主要输入 `RGBFrame`；可选使用当前任务的目标类别或颜色进行筛选 | `Detection2D` | 视觉1 |
| `perception_3d.py` | 深度反投影、官方相机外参适配、三维中心估计接口 | `Detection2D`、`DepthFrame`、`CameraIntrinsics`、Odom（`BaseState`）、`RobotJointState` | `ObjectEstimate3D` | 视觉2 |
| `navigation.py` | 区域分类、站位生成、航点/精对准和速度控制接口 | `TaskSpec`、`ObjectEstimate3D`、`BaseState`、`NavGoal` | `NavGoal`、`BaseCommand`、导航状态 | 底盘 |
| `arm_planning.py` | 官方 KDL 适配、抓放末端目标和关节轨迹规划接口 | 物体中心、任务、实际关节、末端目标 | `IKResult`、`JointTrajectory` 等 | 机械臂1 |
| `arm_execution.py` | 纯轨迹插值、分组限速、反馈到位和局部阶段映射 | `JointTrajectory`、`RobotJointState`、时间 | `ManipulationCommand`、执行状态 | 机械臂2 |
| `fsm.py` | 唯一任务解析、全局阶段转换和重试策略 | 原始任务 JSON、业务事件 | `TaskSpec`、`FSMStatus` | 系统/FSM |
| `action_mux.py` | 候选动作仲裁、TTL、限幅和安全保持 | 底盘/机械臂建议、实际关节、FSM 状态 | 唯一 `FinalAction[19]` | 控制安全 |
| `controller_manifest.py` | 版本化控制接口事实与配置一致性校验 | `ACTION_NAMES`、运行时实测范围、配置 | 不可变 `MMK2_CONTROLLER_MANIFEST_V1` | 架构/控制安全 |
| `recorder.py` | Episode 元数据、团队遥测和 rosbag 命令辅助 | 任务、FSM、最终动作、裁判消息 | metadata、JSONL、rosbag 路径 | 数据 |
| `ros_nodes.py` | 三节点 I/O、缓存、ROS 消息转换、模块组装和官方发布 | ROS 2 消息、`config.yaml` | ROS 2 消息和节点生命周期 | ROS 集成 |

这些拆分是在减少联调风险：

- **二维与三维感知分开**：二维模块只回答“图中哪里是什么”，三维模块才结合深度、
  内参和机器人姿态回答“物体中心在什么坐标系的哪里”。这样能分别定位检测问题和几何问题。
- **机械臂规划与执行分开**：规划回答“目标关节与轨迹是什么”，执行回答“此刻应该发
  哪个插值后的安全目标，以及真实反馈是否到位”。规划结果不等于执行成功。
- **ROS 2 适配集中在 `ros_nodes.py`**：业务模块保持为普通 Python 类，不导入
  `rclpy`，没有 ROS 环境也能做接口和几何测试。
- **公共接口集中在 `interfaces.py`**：所有人共享同一字段、单位、坐标系和失败表示，
  防止每个模块各自解释“目标”“关节”和“动作”。

## 6. 三条核心数据流

### 任务流

```text
/material/instruction
  -> InstructionParser
  -> TaskSpec
  -> GlobalFSM
  -> 搜索感知 -> 导航抓取位 -> 抓取 -> 导航放置位 -> 放置 -> 返区
```

只有 `InstructionParser` 可以解释原始任务 JSON。FSM（有限状态机：用明确阶段和事件管理
流程）只在业务模块给出真实事件后推进，阶段变化本身不代表抓取或得分成功。

### 控制流

```text
BaseCommand（底盘建议）
             +
ManipulationCommand（机械臂建议）
             +
FSMStatus（阶段和安全状态）
             +
RobotJointState（实际反馈，用于安全保持）
             |
             v
         ActionMux
             |
             v
      FinalAction[19]
             |
             v
OfficialCommandPublisher -> 五组官方控制话题
```

`FinalAction`诊断生成与官方发布是两个不同边界。默认observe-only仍执行到ActionMux并
发布团队遥测，但流程在OfficialCommandPublisher之前终止；External Candidate的开关
不能替代该全局门。

### 数据记录流

```text
RGB / Depth / CameraInfo / Odom / JointState / Instruction
ObjectEstimate3D / FSMStatus / FinalAction
Referee taskinfo / gameinfo / score
                         |
                         v
         rosbag / JSONL / metadata
                         |
                         v
             后续 ACT / VLA 数据处理
```

控制链不读取 Recorder 的输出；这条流只复制“发生过什么”，不改变“接下来做什么”。

## 7. 核心数据接口

下表只解释协作时最容易混淆的语义；完整字段和单位以 `interfaces.py` 的 docstring 为准。

| 接口 | 最关键的含义 |
|---|---|
| `TaskSpec` | `InstructionParser` 输出的结构化任务；有效任务只接受官方 `shelf_point`、`table_point`、`shelf_prop_side`，放置 frame 严格为 `world`。`place_world_xyz` 是目标物体中心，不是夹爪末端位姿。 |
| `BaseState` | 来自 Odom 的底盘实际位置、姿态和速度；无效或过期时不能作为到达依据。 |
| `RobotJointState` | `/joint_states` 映射出的 17 维**实际反馈**；它不是规划目标，也不是 IK 解。 |
| `SensorSnapshot` | `team_client_node` 一个周期的轻量快照：任务、底盘、实际关节、三维目标；不含图像。 |
| `Detection2D` | 图像像素坐标中的类别框、置信度、RGB frame 与时间；稳定器输出还携带非负 `track_id`，不带三维位置。 |
| `ObjectEstimate3D` | 纯感知事实：目标**物体中心估计**及可选 `object_id`、观测姿态、明确提供的物体局部 XYZ 尺寸；有效中心估计不要求三个可选事实存在，也不携带 `target_body`。当前三维估计器从独立官方模型尺寸配置生产尺寸，并仅在稳定ID的点云中心和姿态多帧收敛后成对生产 refined geometry。 |
| `Pose3D` | 严格有效 Pose：向量项必须是非bool `numbers.Real`，三项位置有限、四元数归一化非零且 frame 非空；不接受数字字符串，不跨 frame 重命名。 |
| `NavGoal` | 底盘在指定 frame 中的 XY、yaw 目标和容差；物体放置点不能直接当作停车点。 |
| `BaseCommand` | 导航模块提交给 `ActionMux` 的短时有效速度**建议**，不是已经发送的动作。 |
| `IKResult` | 官方 KDL 求得的 slide/左右臂**目标关节解**；不是 `RobotJointState` 的实际反馈。 |
| `RigidTransform3D` | 带 source/target frame、时间和失败语义的刚体变换快照；有效四元数会冻结为单位四元数，不自动交换 frame 求逆。 |
| `GraspContext` | 计划的 object-to-gripper 关系、抓取时物体 world 姿态及执行确认。`confirmed` 只确认计划抓取成立，不把规划关系提升为真实测量标定。 |
| `ArmPlanningConfig` | 允许未标定字段为 `None`，通过 `validate_for_grasp()` / `validate_for_place()` 分操作失败关闭；无笼统 `valid` 字段。 |
| `JointTrajectory` | 绑定任务与严格 `GlobalPhase` 的17维计划；抓取必须完整按 PREGRASP→GRASP→LIFT→RETREAT，放置必须完整按 PREPLACE→LOWER→RELEASE→POST_RELEASE_RETREAT。无效轨迹为空且不伪造 `target_body`。 |
| `ManipulationCommand` | 执行器在本周期提交的关节目标**建议**，`controlled_mask` 指明哪些关节可覆盖保持值。 |
| `FSMStatus` | 当前全局/局部阶段、重试、成功和失败原因的遥测；只有 `DONE` 才表示 FSM 成功。 |
| `FinalAction` | `ActionMux` 每周期生成的唯一 19 维最终动作对象；只有 `valid=True` 且经过 `OfficialCommandPublisher` 成功发布后，才能视为实际控制动作。 |

请始终记住三组区别：

1. `RobotJointState` 是机器人已经处于哪里的反馈；`IKResult` 是希望关节去哪里的求解结果。
2. `ObjectEstimate3D` 与 `TaskSpec.place_world_xyz` 描述物体中心；夹爪末端位姿还需要结合
   抓取方向、双臂间距和物体—夹爪关系另行规划。
3. `BaseCommand` 和 `ManipulationCommand` 都只是模块建议；`ActionMux` 先生成
   `FinalAction`；`ActionMuxDecision`记录请求与接受关系，`ActionDispatchRecord`记录
   本地publisher调用事实。即使publisher正常返回，也不能称为Server已接受或机器人已执行。

### 机械臂公共契约冻结（提交1）

放置姿态采用“保持抓取时观测到的 world 姿态”这一唯一来源：感知提供合法物体姿态后，
规划抓取生成 `GraspContext.object_orientation_world_xyzw_at_grasp`；执行器只能依据实际反馈
确认或拒绝同一规划上下文。未来 `plan_place` 以 `TaskSpec.place_world_xyz` 为物体目标中心、
以上下文姿态为物体目标方向，再由规划的 object-to-gripper 关系计算左右 release Pose。
目标姿态缺失、上下文未确认/过期或 frame 不成立时必须失败关闭；不得补单位四元数、忽略
yaw、根据颜色产生姿态，或把确认后的规划关系描述为真实测量。

`class_id` 是感知类别，`object_id` 是可选稳定轨迹身份，`TaskSpec.target_body` 是官方任务
中的物体 body 身份，三者不互相改名。未来 ROS 组装层必须依据当前任务与显式支持的类别
关联选择唯一稳定目标，把完整 `TaskSpec` 与该目标一起传给 `plan_grasp`；零个候选、多个
同等候选、身份不稳定或任务/类别不匹配均失败关闭。`perception_3d` 不读取当前任务语义。

`ArmMotionPhase` 仅用于 `JointWaypoint` 的规划区段；`LocalPhase` 仍是执行器/FSM 的实时
状态。ArmExecution在提交3A中按实际反馈把规划区段映射为局部执行状态，但不推进 FSM，
也不生成 `FinalAction` 或发布控制命令。
有效路点至少控制一个关节；全False mask不能表示等待、停止或阶段标签。等待由带实际
目标的路点时间表达，停止仍由正常安全控制链处理。

ArmExecution已把真实抓取验证接入纯Python执行闭环：抓取执行到LIFT稳定到位后进入
`VERIFY`，只有属于当前 `EXECUTE_PICK`、最后一个LIFT路点且位于显式等待窗口内的
`GraspVerification` 才能被接收。`success=False` 只表示证据不足并继续等待更新证据；
`success=True` 时无论明确抓住还是明确未抓住，都保存原始结论并继续安全RETREAT，
不得伪造业务成功。RETREAT仍需后续不同时间戳的真实JointState满足连续停稳要求，才返回
`MOTION_COMPLETED_PICK`；同一验证可通过只读 `latest_grasp_verification` 供
`VERIFY_PICK` 使用。放置撤离到位返回
`MOTION_COMPLETED_PLACE_VERIFICATION_PENDING`。两个运动完成态的 `success=True` 只表示
JointTrajectory已由真实关节反馈完整确认，不表示物体抓住、脱夹、位于目标范围或获得
`PICK_VERIFIED` / `PLACE_VERIFIED`。ROS组装层现已把PLAN_PICK/PLAN_PLACE的规划结果、
EXECUTE_PICK/EXECUTE_PLACE的短TTL候选、试抬视觉验证和放置后稳定多帧验证接入现有FSM与
ActionMux；阶段转换当周期会丢弃旧候选，机械臂候选只有在run/task/attempt、轨迹、阶段、
JointState和TTL全部匹配时才可能取得官方发布授权。执行配置无隐藏默认值；当前部署配置
仍保留未标定的 `null`，因此默认运行继续失败关闭，不能据此宣称官方机械臂已可执行。

`VisualObservationVerifier` 当前是纯算法组件：它仅接受达到最低置信度且 frame 严格为
`odom` 的新鲜同目标观测，以 `odom` Z 轴判断试抬高度，并以最大两两距离判断运动后观测
是否稳定。稳定的运动后观测本身不能证明放置位置正确或物体已释放，因而不得直接触发
`PLACE_VERIFIED`。TeamClient现为试抬冻结LIFT前同一object_id观测，并只消费更晚的新鲜
观测；放置验证只收集释放/后撤完成且VERIFY_PLACE入口之后、时间严格递增的同一物体观测，
使用有界缓存确认稳定后，再按当前F1 world/odom数值对齐约定检查三维中心距离与
`TaskSpec.place_radius`。该约定不是通用TF，也不会创建假TF。

夹爪绝对位置范围是官方 `[0,1]` 控制量；`max_gripper_velocity_per_s` 的单位是控制量/秒，
位置范围不能推出速度必须小于等于1，最终速度仍需官方仿真标定。ArmExecution生成的
`ManipulationCommand`只是候选，不证明 `ActionMux` 已接受或官方控制器已经执行。提交4
接线前必须保证：候选失去ActionMux控制权或被STOP覆盖时，组装层暂停或reset执行器，
不得让未实际发布的内部候选历史继续推进。

同一 `ArmMotionPhase` 允许多个连续路点；`HUG_OPEN`、`VERIFY`、
`TRANSPORT_HOLD` 等phase终点状态只在该phase最后一个路点到位时产生。提交3A要求一条
有效轨迹的所有路点使用完全相同的 `controlled_mask`，不允许用mask变化表示等待、阶段
标记或临时释放控制。执行反馈必须携带严格等于团队 `JOINT_NAMES` 的17维名称顺序；左右
夹爪实际反馈和受控路点目标都必须位于官方 `[0,1]`。没有轨迹时执行器保持
`NO_TRAJECTORY`，不应用活动轨迹控制周期超时。这些检查只约束团队候选生成，仍不证明
ActionMux接受候选或官方控制器执行动作。

有效 `TaskSpec` 必须保留官方提供的 `instruction`、`target_kind`、`target_body` 和
`target_color`，不得从 task ID 或颜色规则反向补造。`ros_nodes` 对 `arm_planning`
配置执行严格字段读取：总门必须为 bool、未知/缺失键拒绝、显式 null 保持 `None`；读取
不会构造 `ArmPlanner`，未来调用方仍须按操作分别验证配置。

左右夹爪 `min/max=[0,1]` 来自新版官方离线 `mmk2_control.xml` actuator
`ctrlrange="0. 1."`，属于已冻结的控制硬范围。`open/closed` 仍为未标定 `null`，
`gripper_verified_in_official_environment=false`；确认硬范围不代表开闭值或夹持效果已验证，
所以默认抓取和放置操作验证继续失败关闭；正式启用前仍须在官方仿真标定并填写完整
`arm_planning`、`arm_execution`和视觉验证时间窗口。

## 8. 19 维动作

固定顺序由 `interfaces.ACTION_NAMES` 统一定义：

对应的单位、语义、官方话题、运行时 `ctrlrange`、Server订阅QoS和团队安全范围冻结在
`controller_manifest.MMK2_CONTROLLER_MANIFEST_V1`；Manifest中的名称仍从
`ACTION_NAMES`派生，不是第二套动作顺序。详细实测边界见
[`docs/mmk2_controller_manifest_v1.md`](docs/mmk2_controller_manifest_v1.md)。

| 索引 | 含义 |
|---|---|
| 0 | `base_v`：底盘前向线速度，m/s |
| 1 | `base_w`：底盘绕 Z 轴角速度，rad/s |
| 2 | `slide`：升降/滑轨目标，m |
| 3–4 | `head_yaw`、`head_pitch`，rad |
| 5–10 | 左臂 6 个关节目标，rad |
| 11 | 左夹爪目标，官方 0～1 控制量 |
| 12–17 | 右臂 6 个关节目标，rad |
| 18 | 右夹爪目标，官方 0～1 控制量 |

全队只允许定义一次顺序，因为感知日志、控制发布、数据训练和回放只要有一处错位，就可能
把“夹爪闭合”解释成“手臂转动”。`ActionMux` 按该顺序创建对象，
`OfficialCommandPublisher` 只负责切片成 1 项 slide、2 项 head、左右各 7 项。

前两项是机器人底盘的 `Twist` 语义，即前向速度 `v` 和角速度 `w`，**不能直接当作
Server 内部的左右轮速数组**。轮子数量、方向和底盘解算属于官方控制器内部实现。

同一个不可变 `FinalAction` 对象用于官方话题发布和 `/team/final_action` 遥测，避免因
再次拼接、重新限幅或索引错位产生“记录值与实发值不同”。但对象由 `ActionMux` 生成
不等于已经控制机器人：只有 `valid=True` 且经过 `OfficialCommandPublisher` 成功发布的
动作，仍只能视为本地候选dispatch；`valid=False` 的对象只能用于诊断记录。逐维
requested/commanded/clipped/safety override以及真正publisher payload由
`/team/action_dispatch`的V1严格JSON表达，见
[`docs/mmk2_action_dispatch_telemetry_v1.md`](docs/mmk2_action_dispatch_telemetry_v1.md)。

## 9. ActionMux 与安全设计

`ActionMux` 是动作多路仲裁器：多个模块可以提出建议，但不能各自直接发布命令。否则
导航可能要求底盘前进，机械臂又假设底盘静止，后到的话题会覆盖先到的话题，系统无法知道
本周期究竟执行了什么。集中出口让限幅、过期处理和安全状态只做一次。

TTL（time to live，有效期）规定一条候选命令最多能存活多久。当前配置是 `0.20 s`：

- 底盘命令过期后归零，因为继续复用旧速度会让机器人在控制模块卡住后仍然移动。
- 机械臂命令过期后回到 17 维实际关节位置保持，因为“停止机械臂”通常是维持当前姿态，
  不是回零。
- `SAFE_HOLD` 和 `FAILED` 的优先级高于普通命令：无论候选命令是否仍在 TTL 内，都会
  输出底盘零速并保持实际关节。
- 机械臂不能直接填全零；零值是一组真实关节目标，可能让双臂突然向零位运动。若不知道
  安全目标，应使用实际反馈保持，并把命令标为无效或不控制相应关节。
- 实际关节反馈无效时，`FinalAction.valid=False`，节点会拒绝向官方控制话题发布。

一个简单例子：导航建议 `base_v > 0`，但机械臂正在抓取。最终动作仍由 `ActionMux`
唯一生成，不由导航或机械臂直接决定。**当前骨架只对 `SAFE_HOLD`/`FAILED` 做强制覆盖，
尚未实现“抓取阶段自动压制底盘”的普通阶段策略**；因此完整联调时应由 FSM/控制协调逻辑
在抓取阶段不给出前进命令（或给出零速），再由 `ActionMux` 完成最终限幅和安全检查。

## 10. VLA 数据记录

Recorder的数据身份、逻辑布局、manifest/segment/event字段、关闭恢复、provenance、legacy
兼容和演进规则由
[`config/contracts/recorder_schema_v1.json`](config/contracts/recorder_schema_v1.json)
冻结，中文说明见
[`docs/recorder_schema_v1.md`](docs/recorder_schema_v1.md)。B2 已实现该冻结契约的运行时
生命周期；契约文档中的 `planned_b2` 是冻结时的版本事实，不通过改写契约追记实现状态。
B3 的 TF 录制以及 Commit C 的离线索引、QC、Replay、sample 和训练 Episode 能力仍未实现。

Recorder 启动时先创建 `bootstrap/<segment_id>/`。第一条合法且未结束的
`CompetitionContext` 到达后，bootstrap 原地完整关闭，不被移动或追溯绑定；随后创建
`runs/<run_id>/manifest.json` 和新的 run-bound Segment。Task 或已结算 attempt 变化只写
transition 事件，不切 Segment、不重启 bag；run_id 变化或首次 `finished=true` 才关闭当前
run-bound Segment。Segment 是原始记录生命周期边界，不是官方 Attempt，也不是正式训练
Episode。当前能够覆盖的数据如下：

| 数据 | 保存位置 | 当前说明 |
|---|---|---|
| `instruction_raw` | `metadata.json`，同时在 rosbag 保留原消息 | 原文即使解析失败也保留 |
| 解析后的 `TaskSpec` | `metadata.json` | 由唯一 `InstructionParser` 生成；完整三任务保存在 `parsed_tasks`，旧 `task` 字段仅作 metadata 消费者兼容，不参与当前任务选择 |
| `CompetitionContext` | `competition_contexts.jsonl`、`metadata.json` | 连续记录公开裁判上下文，并按 `run_id`、`task_id`、已结算 attempt 建立索引；不因 Task/attempt 拆分 Segment |
| RGB、Depth、CameraInfo | rosbag | 原始高带宽 ROS 消息，不在 Python 回调中逐帧复制 |
| Odom、JointState | rosbag | 保留原消息、时间和 frame |
| `ObjectEstimate3D` | rosbag 中的 `/team/object_estimates` | 保存团队感知输出 |
| global/local phase | `fsm_status.jsonl`，同时话题进入 rosbag | 来自 `FSMStatus` |
| `FinalAction[19]` | `final_actions.jsonl`，同时话题进入 rosbag | 保存 `ActionMux` 的唯一最终输出；格式保持兼容，训练前还要核对 `valid` 和发布状态 |
| `ActionDispatchRecord` | `action_dispatches.jsonl` | 首条合法 dispatch 立即保存；本轮不把该话题加入 rosbag 列表 |
| 严格动作配对 | `action_frames.jsonl` | 同 sequence、同 timestamp 的两条遥测结构化配对，不是执行确认 |
| 配对异常 | `action_pairing_issues.jsonl` | 无效、重复、冲突、超时、容量淘汰与关闭孤儿 |
| referee 信息 | `metadata.json`，同时进入 rosbag | 原样记录 taskinfo、gameinfo、score，能解析 JSON 时附解析值 |
| success / score 等结果 | FSM JSONL、referee metadata | 官方累计分数来自公开 `/referee/score`，任务进度来自 taskinfo/gameinfo；`FSMStatus.success` 只是本地 FSM 诊断，不能冒充官方比赛结果 |

这些格式各司其职：

- **rosbag** 是 ROS 2 原始消息包，保存 RGB、Depth 等高带宽话题，也保留原消息类型、
  时间戳和坐标系。
- **JSONL** 是“一行一个 JSON 对象”，便于逐周期读取；保存 FSM、两条原始合法动作
  遥测、严格配对 Frame 和配对 Issue。
- **metadata** 是每个 Recorder Segment 内的兼容摘要，保存任务原文、完整 `TaskSpec`
  集合、CompetitionContext 索引、裁判消息、bag 状态、话题计数和可选最终结果。
- **manifest/segment/event** 分别保存 team-local Run 身份、单段生命周期和轻量事实；
  `ACTIVE`/`COMPLETE` 用于识别正常或异常关闭。启动恢复扫描只生成新的只读报告，不修改旧段。

同一Run的`events.jsonl`是跨Segment追加的共享流，因此已关闭Segment只记录其关闭时的
`byte_end_offset`与`sha256_prefix`，不把仍会增长的文件误报成最终全文件hash。恢复扫描会
对每个Run的共享事件流检查一次并区分尾部与中间损坏。`bag_storage_identifier`当前不会
猜测为`sqlite3`；运行时尚未解析真实`metadata.yaml`时保持结构化unavailable。

规范布局为：

```text
team_sorting_dataset/
├── bootstrap/<segment_id>/{segment.json,events.jsonl,ACTIVE|COMPLETE,...}
├── runs/<run_id>/{manifest.json,events.jsonl,segments/<segment_id>/...}
└── recovery/<recovery_report_id>.json
```

manifest 通过同目录临时文件、严格 JSON、文件 `fsync`、`os.replace` 和父目录 `fsync`
更新；marker 排他创建，正常结束仅在 `COMPLETE` 持久化后移除 `ACTIVE`。运行时只扫描上述
规范位置，不迁移、不修复、不覆盖旧扁平目录。路径不安全或身份冲突的 run_id 会 fail
closed，invalid Context 只留下事件诊断且不会替换最近合法身份。

provenance 自动记录 Python/package 版本、契约与配置 hash、ROS domain/RMW 和最终安全开关。
启动器可显式注入 `TEAM_SORTING_PROJECT_COMMIT`、`TEAM_SORTING_PROJECT_BRANCH`、
`TEAM_SORTING_DIRTY_WORKTREE`、`TEAM_SORTING_OFFICIAL_SERVER_IMAGE_ID`、
`TEAM_SORTING_OFFICIAL_CLIENT_IMAGE_ID`、`TEAM_SORTING_DOCKER_IMAGE_DIGEST` 和
`TEAM_SORTING_CONTAINER_IDENTITY`；未注入时使用结构化 unavailable，不访问 Docker socket，
也不采集 hostname、完整环境、token 或 SSH 信息。

`ActionMux` 生成、`valid=True` 且存在本地 publisher 成功事实的 `FinalAction` 也只满足
候选动作的最低条件；DDS交付、Server/controller接受和机器人执行仍未知，必须在Commit C
离线QC中结合实际反馈判断，B2不产生training eligibility。`valid=False` 的 FinalAction、
`SAFE_HOLD`、`FAILED` 和恢复片段可保留用于诊断。`IKResult` 只是目标解，
`JointTrajectory` 只是计划，未插值路点也不是实发动作，三者都不能替代训练标签。裁判
success/score 是比赛/Run级结果，不能复制成每帧真值。当前仓库不负责 ACT/VLA 数据清洗、
训练或在线推理。

Recorder 的任意到达顺序、严格关联、重复/冲突、monotonic 超时、容量淘汰和 shutdown
orphan 规则见
[`docs/mmk2_recorder_action_pairing_v1.md`](docs/mmk2_recorder_action_pairing_v1.md)。
`action_frames.jsonl` 仅证明 Recorder 收到一致的两条内部遥测，不证明 controller 接受
或机器人执行，也不自动成为训练样本。

## 11. 配置与正式话题

运行参数以 `config/config.yaml` 为准，可通过 `TEAM_SORTING_CONFIG` 指向另一份完整配置。
`perception.estimator_3d` 使用精确字段校验：旧的外部完整配置必须以当前默认配置为基线，
补齐稳定身份重关联字段、`object_local_size_xyz_m` 下严格覆盖 `pink`、`yellow`、`brown`
的三项映射及完整 `pose_refinement` 参数；`navigation` 同样必须精确覆盖当前
`NavigationConfig` 的全部字段。缺失、
多余或非法值都会使 PerceptionNode 拒绝启动，且不会退回复用中心补偿专用的
`object_dimensions_m`。这是有意的失败关闭配置迁移，不是局部覆盖或向后兼容默认值。
当前传感器、团队和官方控制话题如下：

| 用途 | 当前配置值 |
|---|---|
| 官方任务 | `/material/instruction` |
| RGB | `/head_camera/color/image_raw` |
| 对齐深度 | `/head_camera/aligned_depth_to_color/image_raw` |
| CameraInfo | `/head_camera/color/camera_info` |
| Odom（里程计） | `/slamware_ros_sdk_server_node/odom` |
| JointState | `/joint_states` |
| 三维目标 | `/team/object_estimates` |
| 最终动作遥测 | `/team/final_action` |
| 动作决策与dispatch遥测 | `/team/action_dispatch` |
| FSM 遥测 | `/team/fsm_status` |
| 官方底盘控制 | `/cmd_vel` |
| 官方 slide 控制 | `/spine_forward_position_controller/commands` |
| 官方头部控制 | `/head_forward_position_controller/commands` |
| 官方左臂控制 | `/left_arm_forward_position_controller/commands` |
| 官方右臂控制 | `/right_arm_forward_position_controller/commands` |

当前关键参数：

| 配置项 | 值 | 含义 |
|---|---:|---|
| `timing.control_rate_hz` | `20.0` | 客户端控制周期频率 |
| `timing.command_ttl_s` | `0.20` | 底盘/机械臂候选命令有效期 |
| `timing.state_max_delta_s` | `0.15` | 图像或控制时刻可接受的邻近状态最大时间差 |
| `control.observe_only` | `true` | 最高优先级观察模式；不创建或调用官方发布器 |
| `control.enable_official_publish` | `false` | 全局官方发布授权，默认关闭 |
| `control.simulation_only` | `true` | 当前只允许仿真；false时拒绝启动 |
| `control.head_target_tracking.enabled` | `false` | fresh-reset专用head target shadow总门，默认关闭 |
| `control.head_target_tracking.fresh_reset_confirmed` | `false` | 运行方对本节点与官方Server共同fresh reset的显式确认 |
| `control.head_target_tracking.initial_{yaw,pitch}_target` | `0.0` | Stage 2A固定reset controller target，单位rad，不能用JointState代替 |
| `control.head_target_tracking.require_exclusive_writer` | `true` | 每次head发布前要求ROS graph中只有本节点一个publisher |
| `perception.sync_slop_s` | `0.05` | RGB/Depth 近似同步及 Detection/Depth/CameraInfo 三方最大绝对时间差 |
| `perception.depth_unit_scale_m` | `0.001` | 深度原始值换算为米的乘数 |
| `perception.stabilizer_2d.min_confirmed_hits` | `2` | 轨迹连续命中两帧后才输出稳定 `track_id` |
| `perception.estimator_3d.ema_alpha` | `0.5` | 同一稳定 `track_id` 的三维中心 EMA 当前样本权重 |
| `perception.estimator_3d.max_position_jump_m` | `1.0` | 单轨迹相邻三维中心最大允许跳变 |
| `perception.estimator_3d.reassociation_*` | 见默认配置 | 二维可见片段中断后按 odom 三维事实恢复稳定 `object_id` 的年龄、距离、尺寸和歧义门限；均为早期仿真初值 |
| `perception.estimator_3d.max_identity_tracks` | `128` | 活跃片段与休眠稳定身份缓存的硬上限 |
| `perception.estimator_3d.object_dimensions_m.*` | `[0.24, 0.16, 0.19]` | 启发式中心补偿使用的宽、高、沿相机视线近似深度；不是物体局部XYZ尺寸 |
| `perception.estimator_3d.object_local_size_xyz_m.*` | `[0.24, 0.16, 0.19]` | 官方模型确认的物体局部坐标系完整XYZ尺寸；只用于`size_xyz_m`，不推断姿态 |
| `perception.estimator_3d.pose_refinement.*` | `enabled: false`，其余见默认配置 | 框内点云深度带、最少点数、连续帧数及位置/角度/尺寸误差门限；均为待真实相机标定的团队初值，正式联调前默认关闭 |
| `navigation.*` | 见默认配置 | `NavigationConfig` 的完整显式映射；米、弧度、m/s、rad/s 与 ns 参数均为待官方仿真实测标定的保守初值 |
| `recorder.enabled` | `false` | 默认不启动记录 |
| `recorder.record_rosbag` | `true` | 启动 Recorder 时同时管理 rosbag |
| `recorder.root_dir` | `./team_sorting_dataset` | Recorder schema v1 dataset root；旧扁平目录不自动迁移 |
| `recorder.recovery_scan_enabled` | `true` | 启动时只读扫描规范 Segment 并另写 recovery 报告 |
| `recorder.bag_shutdown.sigint_timeout_sec` | `30.0` | bag 收到 SIGINT 后的有界等待 |
| `recorder.bag_shutdown.terminate_timeout_sec` | `5.0` | SIGINT 超时后 terminate 的有界等待 |
| `recorder.bag_shutdown.kill_timeout_sec` | `2.0` | terminate 超时后 kill 的最终有界等待 |
| `recorder.rosbag.qos_overrides_path` | `rosbag_qos_overrides.yaml` | 相对实际config资源目录解析的显式TF订阅QoS；无效时fail closed |
| `recorder.action_pairing.enabled` | `true` | 订阅并严格配对 FinalAction/Dispatch 内部遥测 |
| `recorder.action_pairing.max_pending_per_side` | `256` | 每侧等待表容量 |
| `recorder.action_pairing.max_completed_sequences` | `1024` | 近期终态 digest LRU 容量；精确终态区间账本另行阻止旧 sequence 重开 |
| `recorder.action_pairing.max_wait_ns` | `2000000000` | 本地 monotonic 最大等待时间 |
| `recorder.action_pairing.prune_period_sec` | `0.5` | 独立清理定时器周期 |

官方 `mono16` 深度图的原始数值单位按当前代码和测试约定为毫米，因此乘
`depth_unit_scale_m=0.001` 转为米。例如原始值 `1200` 表示 `1.2 m`。
三项初值来自正式 `material_sorting/mjcf/material_competition.xml` 中
`movable_box size="0.12 0.08 0.095"` 的 MuJoCo 半尺寸，配置保存其两倍值；当前仅把
`object_dimensions_m` 第三项用于沿相机视线的启发式中心补偿。正式
`material_competition_layout.json` 还用 `half_size` 明确了相同三轴，并把三类颜色稳定
映射到 `box_pink`、`box_yellow`、`box_brown`；因此独立配置
`object_local_size_xyz_m` 将逐轴两倍值作为物体局部 XYZ 完整尺寸，三维估计器按类别原样
输出到 `size_xyz_m`。Server reset 随机的是颜色到槽位的分配，槽位 yaw 离散为 0 或
π/2；这不会交换物体局部轴。箱体使用自由关节，运行中不能据此假设 roll/pitch 永远为
零；该静态模型事实本身不能产生实时姿态。启用 `pose_refinement` 后，估计器只取二维框
内靠近可见表面深度的点，反投影到相机点云并变换到输出 frame；随后以 PCA 主轴和已知
局部尺寸筛选长方体 OBB，得到点云拟合中心和局部轴到输出 frame 的 `xyzw` 四元数候选；
只观测到单个表面时，结合相机位置与已知半尺寸从可见面恢复中心。同一稳定 `object_id`
对应的当前可见片段必须连续满足中心位置、角度和尺寸误差门限达到 `required_frames`，
才原子输出 refined
中心与姿态；否则继续输出原快速启发式中心且姿态为 `None`。中心对称长方体无法仅凭点云
区分局部轴的 180° 符号翻转，代码固定选择
同一有向包围盒中最接近单位姿态的等价代表，不能把它描述为恢复了 MJCF body 轴符号。
不得用单位四元数或 MJCF 初始姿态补造观测。该算法已有合成点云回归，但正式相机噪声、
遮挡和动态翻滚仍待在线验证。

`recorder.rosbag_topics` 当前逐项记录：

```text
/material/instruction
/head_camera/color/image_raw
/head_camera/aligned_depth_to_color/image_raw
/head_camera/color/camera_info
/slamware_ros_sdk_server_node/odom
/joint_states
/team/object_estimates
/team/fsm_status
/team/competition_context
/team/final_action
/referee/taskinfo
/referee/gameinfo
/referee/score
/tf
/tf_static
```

`/tf`使用best-effort/volatile/keep-last/depth 100；`/tf_static`使用
reliable/transient-local/keep-last/depth 1，以支持Recorder晚启动后的静态TF历史接收。
当前固定场景没有`/tf_static` Publisher时不阻止Recorder启动或Segment正常完成，消息数可为0；
本步骤只记录原始TF，不做frame改写、`world == odom`假设或TF graph QC。

`slot_bounds.table` 和 `slot_bounds.shelf` 当前故意使用反向边界，分类会安全返回
`UNKNOWN`；联调前需要在确认 planning frame 后填写真实区域。不要默认 `world == odom`。

## 12. 启动方式

先进入 ROS 2 工作空间，构建并 source 安装结果：

```bash
cd /path/to/ros2_ws
colcon build --packages-select team_sorting
source install/setup.bash
```

普通联调：

```bash
ros2 launch team_sorting team.launch.xml
```

它启动 `perception_node` 和 `team_client_node`，不启动 Recorder。

数据采集：

```bash
ros2 launch team_sorting team.launch.xml record_data:=true
```

它在上述两个节点之外启动 `dataset_recorder_node`，并显式把 `enabled` 设为 `true`；
若 `record_rosbag` 保持为 `true`，还会启动外部 `ros2 bag record` 子进程。

官方代码、YOLO 权重、MJCF、KDL、ROS 视觉依赖或消息字段未配置好时，相应节点会在
启动自检或处理时清楚报错。可用的路径覆盖变量为：

- `MATERIAL_SORTING_OFFICIAL_ROOT`：官方代码根目录；
- `MATERIAL_SORTING_YOLO_CHECKPOINT`：官方 YOLO 权重；
- `TEAM_SORTING_MJCF`：`MMK2FK` 所需 MJCF；
- `TEAM_SORTING_CONFIG`：运行时配置文件。

当前未实现算法会抛出中文 `NotImplementedError` 或返回 `valid/success=False`，不会
伪造成功。即使节点能启动，也不能据此推断当前代码已能完成比赛。

### 官方 offline Client 容器开发入口

`scripts/run_official_offline_client.sh` 将当前项目挂载到
`/workspace/baseline:ro`。`material_sorting:offline-client` 是 ROS 2 Humble 运行时镜像，
不包含 colcon；正式入口不会安装 colcon，也不会联网下载构建依赖，而是把必要源码复制到
持久 runtime 命名卷的可写 `src/`，再以 `--no-index --no-deps --no-build-isolation`
执行完全离线 pip 安装。默认卷 `team_sorting_offline_client_runtime_v1` 挂载到
`/opt/team_sorting_runtime`，实际安装前缀是
`/opt/team_sorting_runtime/prefix/local`；这是该镜像中 pip `--prefix` 自动增加的
`local` 层，不能把 `/opt/team_sorting_runtime/prefix` 直接当作 ROS 前缀。脚本固定使用 host network
和 host IPC（`--network host --ipc host`）、`ROS_DOMAIN_ID=99`、`rmw_cyclonedds_cpp`，
默认配置仍为 observe-only，不启动或修改官方 Server，也不联网下载模型。宿主机 `MATERIAL_SORTING_OFFICIAL_ROOT` 是必填目录，
脚本会在 Docker 启动前验证官方 Server 关键文件。`TEAM_SORTING_CLIENT_GPUS` 默认
为 `all`；无 NVIDIA GPU 的环境必须显式设为 `none`（空字符串同样禁用 GPU 参数）。

官方 Client 镜像的 Entrypoint 只负责 source ROS 环境并 `exec "$@"`，默认 Cmd 是
`sleep infinity`，不会自动执行团队程序；因此本脚本必须显式执行 `ros2 launch
team_sorting team.launch.xml`。与官方 README 一致，Client 容器使用 host network 和
host IPC，以保证 ROS_DOMAIN_ID=99 的 CycloneDDS 通信和显示/共享内存行为。
`/material/detections` 是官方 Baseline Client 侧 `box_detect.py` 感知节点发布的检测
结果，不是 Server 原生真值；Server 原生发布的是 `/material/instruction` 与裁判话题。
官方头部对齐深度话题 `/head_camera/aligned_depth_to_color/image_raw` 为 `mono16`，
原始数值单位是毫米，而团队三维感知按 `depth_unit_scale_m=0.001` 转换为米后再反投影。

```bash
MATERIAL_SORTING_OFFICIAL_ROOT=/absolute/path/to/official \
MATERIAL_SORTING_YOLO_CHECKPOINT=/absolute/path/to/material_box.pt \
TEAM_SORTING_MJCF=/absolute/path/to/material_competition.xml \
TEAM_SORTING_CLIENT_GPUS=all \
DRY_RUN=1 ./scripts/run_official_offline_client.sh
```

可用 `TEAM_SORTING_RUNTIME_VOLUME` 覆盖 runtime 卷名。脚本以 setup/package 元数据、
`resource/`、Python 包、launch 和 config 的路径与内容计算稳定 SHA256：源码未变化且
安装副本完整时直接 cache hit，不重复构建 wheel 或执行 pip install；源码变化时更新
卷内可写副本并重新离线安装。`TEAM_SORTING_CLEAN_BUILD=1` 会清理 runtime 卷内的
`src/`、`prefix/`、`pip-cache/` 和 `source.sha256` 后强制重建，不删除只读的
`/workspace/baseline`。旧 `TEAM_SORTING_COLCON_CACHE_VOLUME` 仅作弃用迁移兼容。

启动前脚本把 `prefix/local` 放在 `AMENT_PREFIX_PATH`、`PATH` 和 `PYTHONPATH` 最前，
过滤继承环境中直接指向宿主源码的 PYTHONPATH 项，并在 `/tmp` 验证 Python 导入路径、
`ros2 pkg prefix`、三个节点入口、launch 和 config 均来自持久安装副本。该镜像中的
pip wheel 会把 console scripts 安装到 `prefix/local/bin`；脚本会验证这些源入口并在
`prefix/local/lib/team_sorting` 创建符号链接，以提供 ROS 2 要求的标准包 libexec 布局。
旧 runtime 卷即使源码指纹未变，也会在 cache hit 前自动补建并验证这些链接。比赛现场不应
联网安装依赖；离线 pip 安装及静态真实性检查成功，也不等于 ROS 话题通信、QoS、传感器
消息或 observe-only 发布边界已经通过在线验证。容器内部 source ROS setup 时保留
`errexit`/`pipefail`，但不启用会与 ROS setup 冲突的 Bash `nounset`。

TeamClient 将完整三任务集合与 `/referee/taskinfo`、`/referee/gameinfo`、
`/referee/score` 严格组合，并在配置化的 `/team/competition_context` 发布 schema v1
JSON。`attempt` 是当前任务已结算次数；任务切换只重建本地单任务 `GlobalFSM`，不表示
官方物理场景复位。同一任务的 `attempt` 增加也只重新武装本地单任务 FSM，不改变
`run_id`、不生成新任务集合、不清空 Recorder 历史，也不代表 Server、机器人或物品
复位。Recorder 在同一 run-bound Segment 中连续保存这些变化，并用原有
`competition_contexts.jsonl`、metadata 索引和新增 transition 事件恢复
Run/Task/Attempt 层级；Segment 边界不被描述为正式训练 Episode 边界。

## 13. 当前已完成和未完成

### 架构已经完成

- 固定的文件和模块边界，以及业务模块不依赖 `rclpy` 的约束；
- 公共 dataclass、枚举、单位、坐标系和失败语义；
- 固定 `FinalAction[19]` 顺序与 JSON 往返校验；
- `ActionMux` 的限幅、TTL、安全状态覆盖和实际关节保持骨架；
- 三个 ROS 2 节点入口、消息转换、时间缓存和官方发布出口骨架；
- YOLO、MMK2FK、KDL 薄适配器及缺失依赖的清晰报错；
- YOLO 检测的 RGB frame 传递、二维稳定轨迹 ID 与 PerceptionNode 接线；
- 三维深度中位数、反投影、配置尺寸中心补偿、独立局部尺寸输出、点云长方体中心/姿态拟合、稳定 ID 多帧 refine、跳变拒绝与三方时帧校验；
- 机械臂规划、轨迹执行、试抬抓取验证、confirmed GraspContext、放置后稳定观测与
  `place_radius` 判断的ROS/FSM组装接线，以及执行阶段候选的ActionMux/发布授权门；
- Recorder schema v1 bootstrap/run-bound 生命周期、manifest/segment/event、只读恢复报告，
  以及兼容 metadata、FSM/动作 JSONL 和分段外部 rosbag 管理链；
- 几何、任务解析、FSM、19 维动作、安全覆盖与 Recorder 的测试骨架。

### 尚未完成

- 正式 YOLO/相机环境中的检测与二维稳定器参数联调；
- 点云中心/姿态 refine 在正式相机噪声与遮挡下的参数标定，以及正式 ROS/MMK2FK 环境中的三维坐标、时间同步和 planning frame 端到端验证（通过前默认关闭）；
- 导航参数的官方仿真实测标定、障碍物检测与全局路径规划；当前只有站位生成、局部精对准和比例控制闭环；
- 机械臂KDL、夹爪开闭值、速度、容差、轨迹时长和视觉验证窗口的官方仿真标定；默认
  配置在这些事实确认前保持关闭/null，因此尚未完成正式环境端到端动作验证；
- 正式规则下抓放失败恢复、返区和多任务结算的端到端比赛验证；
- ACT/VLA 数据处理、训练和推理。

当前 `team_client_node` 在三个导航阶段使用导航控制器返回的短 TTL `BaseCommand`；机械臂
执行阶段在完整显式配置与KDL自检成功后使用ArmExecution返回的短TTL
`ManipulationCommand`，其他普通阶段为零速且不主动生成机械臂保持。所有候选均经ActionMux。
默认observe-only不创建官方发布器，
因此“生成诊断FinalAction”不等于“发送位置保持命令”。`create_hold_command()`仍保留给
未来明确授权的主动保持场景。“仓库骨架完成”绝不等于“比赛代码完成”。

### 已由正式官方源码确认

- Server 每 0.5 秒在 `/material/instruction` 发布完整且有序的三任务列表；一次比赛运行按
  Task 1→2→3 连续推进，而不是从列表中只选择一种任务变体。
- Client 一次启动持续处理三项任务的生命周期。TeamClient 保留完整任务集合，并严格组合
  公开 `/referee/taskinfo`、`/referee/gameinfo`、`/referee/score` 选择当前 active task，
  不再固定采用 `tasks[0]`。
- 每项任务最多有三次已结算 attempt；放置成功或机会用尽后进入下一任务。同一 Task 的
  attempt 变化和 Task 切换都不会复位 Server、机器人或物品，只会重新武装本地单任务 FSM。
- Recorder 的 run-bound Segment 不因 Task/attempt 变化切分或清空历史；
  `competition_contexts.jsonl` 和 metadata 按 Run/Task/已结算 attempt 建立索引。
- 正式随机 Server 开发入口 `start_official_random_server.sh` 显式设置
  `MATERIAL_RANDOMIZE=1`。这描述的是正式启动入口，不把 Python 环境变量的缺省解析
  与脚本运行模式混为一谈。

以上 CompetitionContext 和记录边界对齐只说明比赛生命周期契约已经接通，不表示导航、
抓取、放置、机械臂执行或完整比赛闭环已经完成。当前默认仍是 observe-only。

### 仍需在线环境验证

- ROS 话题的实际 QoS、三条裁判消息的调度时序，以及瞬态不一致后的在线恢复表现；
- `world`、`odom` 与官方 `MMK2FK` 输出 frame 的在线一致性；
- Server 命令 watchdog、正式控制所需频率，以及五组控制话题在最终环境中的连接；
- 官方 Client 镜像内 YOLO、MJCF、KDL 的实际路径及 `vision_msgs` 版本；
- CameraInfo、Odom 和 TF 的在线消息 frame、时间戳与更新行为；
- 默认 observe-only 运行时是否确实没有任何官方控制话题发布。

## 14. 团队分工

| 成员角色 | 主责文件 | 输入 | 输出 | 禁止越界内容 |
|---|---|---|---|---|
| 架构/系统 | `interfaces.py`、`README.md`、`AGENTS.md`、包元数据 | 全队接口需求、官方协议事实 | 公共接口、文档、模块边界 | 在各业务文件中私自复制接口或另建动作顺序 |
| 视觉1 | `perception_2d.py` | 主要输入 `RGBFrame`；可选使用当前任务的目标类别或颜色进行筛选 | `Detection2D` | Depth、CameraInfo、Odom、`RobotJointState`、三维外参、抓取决策、ROS 发布 |
| 视觉2 | `perception_3d.py`、`tests/test_geometry_and_planning.py` 中相关测试 | `Detection2D`、`DepthFrame`、`CameraIntrinsics`、Odom（`BaseState`）、`RobotJointState` | `ObjectEstimate3D`、几何回归 | YOLO、导航、夹爪位姿规划 |
| 底盘 | `navigation.py` | 任务、三维目标、`BaseState`、`NavGoal` | `NavGoal`、`BaseCommand`、导航状态 | 直接发 `/cmd_vel`、改 FSM、做 IK |
| 机械臂1 | `arm_planning.py`、相关几何测试 | 物体中心、任务、实际关节、末端目标 | `IKResult`、`JointTrajectory` | 把 IK 当反馈、直接发布关节命令、执行状态机 |
| 机械臂2 | `arm_execution.py` | 轨迹、实际关节、时间 | `ManipulationCommand`、执行/验证状态 | 重新做 IK、绕过 `ActionMux`、用全零冒充保持 |
| 系统/FSM | `fsm.py`、`tests/test_fsm_mux_recording.py` 中 FSM 测试 | 原始任务、真实业务事件 | `TaskSpec`、`FSMStatus`、重试/失败路径 | 直接控制硬件、绕过唯一任务解析入口 |
| 控制安全 | `action_mux.py`、`controller_manifest.py`、动作/TTL/Manifest测试 | 两类候选命令、实际关节、FSM 状态、运行时控制事实 | 唯一 `FinalAction[19]`、版本化安全元数据 | 规划轨迹、发布 ROS 话题、定义第二套动作顺序 |
| 数据 | `recorder.py`、记录测试 | 任务、FSM、最终动作、裁判消息 | metadata、JSONL、rosbag 命令 | 参与控制、逐帧复制图像、把裁判结果扩成帧标签 |
| ROS 集成 | `ros_nodes.py`、`config/config.yaml`、`launch/team.launch.xml` | ROS 消息、配置、业务对象 | 三节点 I/O、缓存、转换、唯一官方发布 | 在适配层实现 YOLO/导航/IK/轨迹算法 |

跨模块需求先改 `interfaces.py` 并评审，再由上下游各自适配。任何成员都不能为了“先跑起来”
绕过 `InstructionParser`、`ActionMux` 或 `OfficialCommandPublisher`。

## 15. 队长推荐阅读顺序

```text
README
  -> AGENTS
  -> interfaces
  -> fsm
  -> action_mux
  -> ros_nodes
  -> perception_2d
  -> perception_3d
  -> navigation
  -> arm_planning
  -> arm_execution
  -> recorder
  -> tests
  -> config / launch / package / setup
```

前六项先建立“数据从哪里来、谁能做决定、谁能发动作”的整体观念；之后再按感知、底盘、
机械臂、记录的顺序深入。每读一个文件，都尝试回答：

1. 这个文件负责什么？
2. 输入是什么，单位和坐标系是什么？
3. 输出是什么，谁会消费它？
4. 谁调用它？
5. 失败如何表示：异常、`valid=False`、`success=False`，还是 FSM 事件？
6. 它是否越界做了其他模块的事情？

如果其中一题答不上来，先不要急着改算法；接口语义没统一时，联调问题通常会在下游放大。

## 16. 项目基础设施文件

| 文件 | 作用 |
|---|---|
| `team_sorting/__init__.py` | 标记 Python 包，保存简短包说明和 `__version__`；不会在导入包时加载 ROS 2 或官方依赖。 |
| `resource/team_sorting` | ament 资源索引标记。它通常是空文件（本仓库也只有空白内容），但 ROS 2 安装后靠它发现包，**不能删除**。 |
| `package.xml` | ROS 2 包清单：包名、版本、许可证以及运行/测试依赖。 |
| `setup.py` | Python/ament 安装规则，安装 config、launch、资源标记，并注册三个 console script。 |
| `setup.cfg` | 指定 ROS 2 可执行脚本安装到 `lib/team_sorting`，供 `ros2 run/launch` 查找。 |
| `launch/team.launch.xml` | 声明三个节点的启动关系；默认启动感知和客户端，`record_data:=true` 时再启动 Recorder。 |

仓库结构按 `AGENTS.md` 固定；不要随意增加 manager、factory、自定义 ROS 消息包或新的
业务目录。若确有结构性需求，应先由全队评审接口和迁移范围。
