# team_sorting

## 1. 项目一句话介绍

这是团队自己的 ROS 2（机器人软件通信框架）竞赛客户端仓库，不是官方比赛 Server
仓库：它读取官方任务和传感器数据，组织感知、导航与 MMK2 机械臂模块，把各模块建议
统一仲裁为 `FinalAction[19]`，再通过官方 ROS 2 控制话题发给机器人；同时可以在控制链
旁路记录后续 ACT（动作分块 Transformer）/VLA（视觉—语言—动作模型）需要的数据。

> 当前定位是“可导入、可测试、边界清晰的客户端骨架”，不是已经能完成比赛的成品。
> 真实相机/YOLO/MMK2FK 联调、导航、抓放规划、机械臂执行和比赛闭环仍未完成。

## 2. 官方仓库与团队仓库的边界

官方工程提供场景、裁判、传感器、机器人模型、YOLO、运动学和控制入口；团队仓库负责把
这些能力组织成可协作、可测试的客户端。这里采用“薄适配器”：只检查依赖、转换输入、
调用官方能力、转换输出，不在本仓库重写官方算法。

| 团队文件 | 官方来源 | 复用能力 | 团队适配接口 | 当前状态 |
|---|---|---|---|---|
| `perception_2d.py` | 官方 `backends.py`、YOLO 权重 | 加载官方检测器并执行二维检测 | `OfficialYoloAdapter.self_check()` / `detect()` | 检测转换、稳定轨迹 ID 与回归测试完成，正式推理待联调 |
| `perception_3d.py` | DISCOVERSE 的 `MMK2FK`、官方 `box_detect.py` 的针孔反投影思路 | 根据实际底盘和关节状态获得相机外参 | `CameraTransformProvider.self_check()` / `camera_to_output()` | 深度中位数、反投影、尺寸中心补偿、稳定 ID EMA 与时帧校验已实现；真实 ROS/FK 待联调 |
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

图中 Navigation、Arm Planning 和 Arm Execution 表示目标架构。当前 `team_client_node`
尚未把完整导航与抓放算法接入控制循环；它只生成底盘零速和基于实际关节反馈的保持命令。

## 4. 三个 ROS2 节点

ROS 2 节点可以理解为一个独立运行、通过话题交换消息的程序。本仓库只有三个节点入口，
都集中在 `ros_nodes.py`。

### `perception_node`

- **订阅**：RGB `/head_camera/color/image_raw`、对齐深度
  `/head_camera/aligned_depth_to_color/image_raw`、CameraInfo
  `/head_camera/color/camera_info`、Odom `/slamware_ros_sdk_server_node/odom`、
  `/joint_states`。RGB 与 Depth 近似同步；Odom 和 JointState 取时间上最近且未超出容差的值。
- **发布**：`/team/object_estimates`，类型为
  `vision_msgs/msg/Detection3DArray`，载荷包含类别、三维位置、置信度、时间和坐标系；
  `slot_type` 不通过该消息传输。
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
- **发布**：团队遥测 `/team/fsm_status`、`/team/final_action`，并通过唯一的
  `OfficialCommandPublisher` 发布五组官方控制话题。
- **调用**：当前实际调用 `fsm.py`、`action_mux.py`、`arm_execution.py` 的安全保持、
  `arm_planning.py` 的官方 KDL 启动自检，以及 `navigation.py` 的区域分类；完整架构还应
  在这里按 FSM 阶段组织导航、抓放规划和执行。
- **不负责**：YOLO、深度图处理、相机外参算法、KDL 实现或数据落盘。
- **是否必需**：默认 launch 会启动，也是控制链必需节点；只有它可以经
  `OfficialCommandPublisher` 发官方控制话题。

`team_client_node` 每个控制周期构造的 `SensorSnapshot` 只含任务、底盘状态、实际关节
状态和三维目标，**不含 RGB、Depth 或 CameraInfo**。图像留在感知节点和 rosbag，避免
把高带宽数据复制进控制循环。

### `dataset_recorder_node`

- **订阅**：`/team/final_action`、`/team/fsm_status`、`/material/instruction`，以及
  `/referee/taskinfo`、`/referee/gameinfo`、`/referee/score`。配置中的原始传感器和
  `/team/object_estimates` 由它管理的外部 `ros2 bag record` 进程订阅。
- **发布**：无控制话题；输出是 Episode 目录中的 `rosbag/`、JSONL 和 `metadata.json`。
- **调用**：`recorder.py` 的 `EpisodeRecorder`、`fsm.py` 的 `InstructionParser`，以及
  `interfaces.py` 中唯一的 FSM/动作 JSON 编解码函数。
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
| `arm_execution.py` | 轨迹插值、局部抓放状态、试抬和恢复接口 | `JointTrajectory`、`RobotJointState`、时间 | `ManipulationCommand`、执行状态 | 机械臂2 |
| `fsm.py` | 唯一任务解析、全局阶段转换和重试策略 | 原始任务 JSON、业务事件 | `TaskSpec`、`FSMStatus` | 系统/FSM |
| `action_mux.py` | 候选动作仲裁、TTL、限幅和安全保持 | 底盘/机械臂建议、实际关节、FSM 状态 | 唯一 `FinalAction[19]` | 控制安全 |
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
| `TaskSpec` | `InstructionParser` 输出的结构化任务；包含目标属性与放置约束。`place_world_xyz` 是目标物体中心，不是夹爪末端位姿。 |
| `BaseState` | 来自 Odom 的底盘实际位置、姿态和速度；无效或过期时不能作为到达依据。 |
| `RobotJointState` | `/joint_states` 映射出的 17 维**实际反馈**；它不是规划目标，也不是 IK 解。 |
| `SensorSnapshot` | `team_client_node` 一个周期的轻量快照：任务、底盘、实际关节、三维目标；不含图像。 |
| `Detection2D` | 图像像素坐标中的类别框、置信度、RGB frame 与时间；稳定器输出还携带非负 `track_id`，不带三维位置。 |
| `ObjectEstimate3D` | 目标**物体中心的估计位置**、frame、时间、置信度和 `valid/failure_reason`；已按配置尺寸沿光学射线补偿可见表面点，但不是夹爪末端位姿。 |
| `NavGoal` | 底盘在指定 frame 中的 XY、yaw 目标和容差；物体放置点不能直接当作停车点。 |
| `BaseCommand` | 导航模块提交给 `ActionMux` 的短时有效速度**建议**，不是已经发送的动作。 |
| `IKResult` | 官方 KDL 求得的 slide/左右臂**目标关节解**；不是 `RobotJointState` 的实际反馈。 |
| `JointTrajectory` | 带相对时间的 17 维关节路点计划；它不证明轨迹已执行。 |
| `ManipulationCommand` | 执行器在本周期提交的关节目标**建议**，`controlled_mask` 指明哪些关节可覆盖保持值。 |
| `FSMStatus` | 当前全局/局部阶段、重试、成功和失败原因的遥测；只有 `DONE` 才表示 FSM 成功。 |
| `FinalAction` | `ActionMux` 每周期生成的唯一 19 维最终动作对象；只有 `valid=True` 且经过 `OfficialCommandPublisher` 成功发布后，才能视为实际控制动作。 |

请始终记住三组区别：

1. `RobotJointState` 是机器人已经处于哪里的反馈；`IKResult` 是希望关节去哪里的求解结果。
2. `ObjectEstimate3D` 与 `TaskSpec.place_world_xyz` 描述物体中心；夹爪末端位姿还需要结合
   抓取方向、双臂间距和物体—夹爪关系另行规划。
3. `BaseCommand` 和 `ManipulationCommand` 都只是模块建议；`ActionMux` 先生成
   `FinalAction`，只有其中 `valid=True` 且经过 `OfficialCommandPublisher` 成功发布的
   动作，才是实际控制动作。

## 8. 19 维动作

固定顺序由 `interfaces.ACTION_NAMES` 统一定义：

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
动作，才能视为实际控制动作；`valid=False` 的对象只能用于诊断记录。

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

Episode 是一次连续采集片段。当前临时边界是 recorder 节点启动到停止；正式比赛的开始、
结束和任务切分规则仍待确认。一个 Episode 计划/当前能够覆盖的数据如下：

| 数据 | 保存位置 | 当前说明 |
|---|---|---|
| `instruction_raw` | `metadata.json`，同时在 rosbag 保留原消息 | 原文即使解析失败也保留 |
| 解析后的 `TaskSpec` | `metadata.json` | 由唯一 `InstructionParser` 生成；当前记录第一条任务 |
| RGB、Depth、CameraInfo | rosbag | 原始高带宽 ROS 消息，不在 Python 回调中逐帧复制 |
| Odom、JointState | rosbag | 保留原消息、时间和 frame |
| `ObjectEstimate3D` | rosbag 中的 `/team/object_estimates` | 保存团队感知输出 |
| global/local phase | `fsm_status.jsonl`，同时话题进入 rosbag | 来自 `FSMStatus` |
| `FinalAction[19]` | `final_actions.jsonl`，同时话题进入 rosbag | 保存 `ActionMux` 的唯一最终输出；训练前还要核对 `valid` 和发布状态 |
| referee 信息 | `metadata.json`，同时进入 rosbag | 原样记录 taskinfo、gameinfo、score，能解析 JSON 时附解析值 |
| success / score 等结果 | FSM JSONL、referee metadata | `FSMStatus.success` 和 score 已有记录链；官方 Episode 级 success/final result 的来源与接线待确认 |

三种格式各司其职：

- **rosbag** 是 ROS 2 原始消息包，保存 RGB、Depth 等高带宽话题，也保留原消息类型、
  时间戳和坐标系。
- **JSONL** 是“一行一个 JSON 对象”，便于逐周期读取；这里只保存最终动作和 FSM 状态。
- **metadata** 是一个 Episode 的摘要，保存任务原文、`TaskSpec`、裁判消息、bag 状态、
  话题计数和可选最终结果。

专家训练动作只能使用 `ActionMux` 生成、`valid=True` 且已经过
`OfficialCommandPublisher` 成功发布的 `FinalAction`。`valid=False` 的 `FinalAction`
可以保留用于诊断，但不能直接作为专家训练动作；`SAFE_HOLD`、`FAILED` 和恢复片段也可
保留，并在数据处理时单独标记。`IKResult` 只是目标解，`JointTrajectory` 只是计划，
未插值的路点也不是当时的实发动作，三者都不能替代训练标签。裁判 success/score 是
Episode 级结果，不能复制成每一帧的真值标签。当前仓库只负责可靠记录，不负责 ACT/VLA
的数据清洗、训练或在线推理。

## 11. 配置与正式话题

运行参数以 `config/config.yaml` 为准，可通过 `TEAM_SORTING_CONFIG` 指向另一份完整配置。
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
| `perception.sync_slop_s` | `0.05` | RGB/Depth 近似同步容差 |
| `perception.depth_unit_scale_m` | `0.001` | 深度原始值换算为米的乘数 |
| `perception.stabilizer_2d.min_confirmed_hits` | `2` | 轨迹连续命中两帧后才输出稳定 `track_id` |
| `perception.estimator_3d.ema_alpha` | `0.5` | 同一稳定 `track_id` 的三维中心 EMA 当前样本权重 |
| `perception.estimator_3d.max_position_jump_m` | `1.0` | 单轨迹相邻三维中心最大允许跳变 |
| `perception.estimator_3d.object_dimensions_m.*` | `[0.24, 0.16, 0.19]` | 三类包装盒完整宽/高/深，单位米 |
| `recorder.enabled` | `false` | 默认不启动记录 |
| `recorder.record_rosbag` | `true` | 启动 Recorder 时同时管理 rosbag |
| `recorder.root_dir` | `./team_sorting_dataset` | Episode 根目录 |

三维估计器还会把非零 `perception.sync_slop_s` 作为 Detection、Depth 与
CameraInfo 三者任意一对的最大绝对时间差，超窗输入一律输出无效结果。
官方 `mono16` 深度图的原始数值单位按当前代码和测试约定为毫米，因此乘
`depth_unit_scale_m=0.001` 转为米。例如原始值 `1200` 表示 `1.2 m`。
三类尺寸来自正式 `material_sorting/mjcf/material_competition.xml` 中
`movable_box size="0.12 0.08 0.095"` 的 MuJoCo 半尺寸，配置保存完整尺寸；
当前中心补偿使用第三项作为沿相机视线的近似物体深度。

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
/team/final_action
/referee/taskinfo
/referee/gameinfo
/referee/score
```

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

## 13. 当前已完成和未完成

### 架构已经完成

- 固定的文件和模块边界，以及业务模块不依赖 `rclpy` 的约束；
- 公共 dataclass、枚举、单位、坐标系和失败语义；
- 固定 `FinalAction[19]` 顺序与 JSON 往返校验；
- `ActionMux` 的限幅、TTL、安全状态覆盖和实际关节保持骨架；
- 三个 ROS 2 节点入口、消息转换、时间缓存和官方发布出口骨架；
- YOLO、MMK2FK、KDL 薄适配器及缺失依赖的清晰报错；
- YOLO 检测的 RGB frame 传递、二维稳定轨迹 ID 与 PerceptionNode 接线；
- 三维深度中位数、反投影、配置尺寸中心补偿、稳定 ID EMA、跳变拒绝与三方时帧校验；
- Episode metadata、FSM/动作 JSONL 和外部 rosbag 管理链；
- 几何、任务解析、FSM、19 维动作、安全覆盖与 Recorder 的测试骨架。

### 尚未完成

- 正式 YOLO/相机环境中的检测与二维稳定器参数联调；
- 正式 ROS/MMK2FK 环境中的三维坐标、时间同步和 planning frame 端到端验证；
- 抓取/放置站位生成、航点导航、精对准和底盘控制；
- 由物体中心生成抓取/放置末端位姿，以及完整 IK/轨迹规划；
- 机械臂轨迹插值、限速和局部抓放状态机；
- 试抬抓取验证、放置验证和失败恢复；
- 业务结果驱动 FSM、底盘与机械臂协同的完整比赛闭环；
- ACT/VLA 数据处理、训练和推理。

当前 `team_client_node` 在控制周期中只创建零速 `BaseCommand` 和实际关节保持命令；
“仓库骨架完成”绝不等于“比赛代码完成”。

### 待确认

以下事实无法从当前仓库代码确认，联调前应向赛事方或官方工程核对：

- 当前比赛是否为“一次运行随机选择一种任务变体”，以及任务数组与 Episode 的正式语义；
  当前 client 和 Recorder 都只使用解析结果的第一条任务，不能宣称一次运行会连续完成多项任务。
- `MATERIAL_RANDOMIZE` 的实际默认值；
- `world` 与 `odom` 是否永久重合，以及官方 `MMK2FK` 的真实输出 frame；
- 正式 Episode 的开始、结束、任务切分规则，以及官方最终 success/result 的来源和话题；
- Server 是否有命令 watchdog、官方控制所需频率，以及当前五组控制话题是否与最终比赛环境一致；
- 官方 Python 模块最终导入路径、YOLO 权重/MJCF 位置和 `vision_msgs` 具体版本。

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
| 控制安全 | `action_mux.py`、动作/TTL 测试 | 两类候选命令、实际关节、FSM 状态 | 唯一 `FinalAction[19]` | 规划轨迹、发布 ROS 话题、定义第二套动作顺序 |
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
