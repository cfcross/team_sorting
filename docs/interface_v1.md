# Team Sorting Interface v1

## 1. 适用范围与权威来源

本文件是 [`config/contracts/interface_v1.json`](../config/contracts/interface_v1.json) 的中文投影；机器契约是字段、顺序、单位、空值和状态的规范来源。它冻结协作边界，不改变 ActionMux、官方 publisher、Recorder、生命周期或算法，也不表示比赛闭环已经完成。schema 为 `team_sorting.interface`，major 版本为 `1`，状态为 `frozen`。

- `frozen`：v1 内名称和语义不可静默改变。
- `provisional`：类型真实存在，但算法、接线或在线事实尚未完成。
- `restricted`：只允许机器契约列出的受限用途；当前仅包括 head-yaw-only `ExternalCandidate`。

## 2. 19维动作契约

`ACTION_NAMES` 是唯一动作顺序。索引2–18与 `JOINT_NAMES` 按位置和语义对应，但两个名称数组并不相等。

| idx | name | group[index] | mode | unit | safe | runtime | status |
|---:|---|---|---|---|---|---|---|
| 0 | base_v | base[0] | velocity | m/s | [-0.25,0.25] | null | partially_verified |
| 1 | base_w | base[1] | velocity | rad/s | [-0.50,0.50] | null | partially_verified |
| 2 | slide | spine[0] | absolute_position | m | [-0.04,0.87] | [-0.04,0.87] | runtime_verified |
| 3 | head_yaw | head[0] | absolute_position | rad | [-0.50,0.50] | [-0.50,0.50] | runtime_verified |
| 4 | head_pitch | head[1] | absolute_position | rad | [-1.18,0.16] | [-1.18,0.16] | runtime_verified |
| 5 | left_arm_joint_1 | left_arm[0] | absolute_position | rad | [-3.14,2.089] | [-3.151,2.089] | runtime_verified |
| 6 | left_arm_joint_2 | left_arm[1] | absolute_position | rad | [-2.50,0.181] | [-2.963,0.181] | runtime_verified |
| 7 | left_arm_joint_3 | left_arm[2] | absolute_position | rad | [-0.094,3.14] | [-0.094,3.161] | runtime_verified |
| 8 | left_arm_joint_4 | left_arm[3] | absolute_position | rad | [-2.60,2.60] | [-3.012,3.012] | runtime_verified |
| 9 | left_arm_joint_5 | left_arm[4] | absolute_position | rad | [-1.859,1.859] | [-1.859,1.859] | runtime_verified |
| 10 | left_arm_joint_6 | left_arm[5] | absolute_position | rad | [-2.60,2.60] | [-3.017,3.017] | runtime_verified |
| 11 | left_gripper | left_arm[6] | normalized_position | dimensionless | [0,1] | [0,1] | unresolved |
| 12 | right_arm_joint_1 | right_arm[0] | absolute_position | rad | [-3.14,2.089] | [-3.151,2.089] | runtime_verified |
| 13 | right_arm_joint_2 | right_arm[1] | absolute_position | rad | [-2.50,0.181] | [-2.963,0.181] | runtime_verified |
| 14 | right_arm_joint_3 | right_arm[2] | absolute_position | rad | [-0.094,3.14] | [-0.094,3.161] | runtime_verified |
| 15 | right_arm_joint_4 | right_arm[3] | absolute_position | rad | [-2.60,2.60] | [-3.012,3.012] | runtime_verified |
| 16 | right_arm_joint_5 | right_arm[4] | absolute_position | rad | [-1.859,1.859] | [-1.859,1.859] | runtime_verified |
| 17 | right_arm_joint_6 | right_arm[5] | absolute_position | rad | [-2.60,2.60] | [-3.017,3.017] | runtime_verified |
| 18 | right_gripper | right_arm[6] | normalized_position | dimensionless | [0,1] | [0,1] | unresolved |

`base_v/base_w` 分别进入 `Twist.linear.x/angular.z`，不是左右轮直接命令；官方接受的最大 `cmd_vel` 范围未知。位置轴的0不是停止，未控制的位置轴不得补零。夹爪只确认 `[0,1]` 范围，0/1开闭方向和有效夹持值未知。head-only 的精确训练命令只能取 dispatch payload。

## 3. 17维JointState契约

| idx | JointState name | 对应Action | position单位 |
|---:|---|---:|---|
| 0 | slide_joint | 2 | m |
| 1 | head_yaw_joint | 3 | rad |
| 2 | head_pitch_joint | 4 | rad |
| 3–8 | left_arm_joint1..6 | 5–10 | rad |
| 9 | left_arm_eef_gripper_joint | 11 | dimensionless |
| 10–15 | right_arm_joint1..6 | 12–17 | rad |
| 16 | right_arm_eef_gripper_joint | 18 | dimensionless |

当前 ROS 适配器在源 `velocity` 或 `effort` 为空时补17个零。补零可能表示源字段缺失，不一定表示真实物理零值。官方源码没有给出 effort 的单位契约，故保持 UNRESOLVED。

## 4. 五组官方控制话题

| group | topic / type | width | Action slice | 语义 |
|---|---|---:|---|---|
| base | `/cmd_vel` / `geometry_msgs/msg/Twist` | 2 | `[0,2)` | velocity |
| spine | `/spine_forward_position_controller/commands` / `std_msgs/msg/Float64MultiArray` | 1 | `[2,3)` | absolute position |
| head | `/head_forward_position_controller/commands` / `std_msgs/msg/Float64MultiArray` | 2 | `[3,5)` | absolute position |
| left_arm | `/left_arm_forward_position_controller/commands` / `std_msgs/msg/Float64MultiArray` | 7 | `[5,12)` | 6 joint + gripper position |
| right_arm | `/right_arm_forward_position_controller/commands` / `std_msgs/msg/Float64MultiArray` | 7 | `[12,19)` | 6 joint + gripper position |

ActionMux 是唯一仲裁入口，只有 `OfficialCommandPublisher` 可以创建这些 publisher。组员模块不得直发。observe-only 时 publisher 不创建、调用不发生。

## 5. Observation契约

| Observation | topic | 时间/frame要点 | rate来源 | 缺失行为/状态 |
|---|---|---|---|---|
| head RGB | `/head_camera/color/image_raw` | ROS stamp, `head_camera` | 源码调用24Hz，非协议保证 | 拒绝样本/source_verified |
| aligned depth | `/head_camera/aligned_depth_to_color/image_raw` | ROS stamp, `head_camera` | 同上 | 拒绝同步估计/source_verified |
| head CameraInfo | `/head_camera/color/camera_info` | 原始zero stamp、空frame；适配器绑定RGB上下文不改写原始事实 | 源码timer 1Hz，非协议保证 | 无内参不产出3D/source_verified |
| Odom | `/slamware_ros_sdk_server_node/odom` | 原始`/odom`，ROS边界规范化为`odom` | 源码调用24Hz，非协议保证 | fail closed/source_verified |
| TF | `/tf` | `odom`→`base_link` | 源码调用24Hz，非协议保证 | Recorder v1计划必采/planned |
| JointState | `/joint_states` | ROS stamp，固定17维 | 源码调用24Hz，非协议保证 | 位置控制fail closed/source_verified |
| ObjectEstimate3D | `/team/object_estimates` | 当前规划frame `odom` | 同步感知驱动 | 无目标则保持不可用/provisional |
| instruction | `/material/instruction` | receive time；放置点`world` | 源码timer 2Hz，非协议保证 | 无有效任务/source_verified |
| CompetitionContext | `/team/competition_context` | 严格JSON生成时间 | event-driven | 身份依赖记录fail closed/frozen |
| referee三项 | `/referee/taskinfo`,`gameinfo`,`score` | receive time | 源码timer 2Hz，非协议保证 | 不推断转换/source_verified |
| FSMStatus | `/team/fsm_status` | generated timestamp | 配置20Hz，非协议保证 | 不推断成功/provisional |
| ActionDispatch | `/team/action_dispatch` | 与FinalAction同一生成时间 | 配置20Hz，非协议保证 | 不推断dispatch/frozen |
| execution feedback | JointState + Odom | 两个异步sensor stamp及receive time | 无统一保证 | 执行保持未确认/provisional |

## 6. 时间戳与frame规则

`sensor_timestamp_ns`、`receive_timestamp_ns`、`receive_monotonic_ns`、`generated_timestamp_ns` 必须是非bool、非负整数。monotonic只可用于同一进程内等待/淘汰，持久化后不得跨进程比较，也不用于Observation/Action语义配对。`sim_elapsed_s` 是有限非负秒数或null，绝不能与ROS纳秒互换。

ROS边界可以去掉frame的一个前导`/`，因此Odom原始`/odom`与TF parent `odom`规范化为`odom`。`RigidTransform3D` 本身不偷偷转换别名。Task放置点当前是`world`，三维规划当前是`odom`；两者关系UNRESOLVED，不得声明相等。frame未知或转换不可用时fail closed。

## 7. 生命周期身份

- Evaluation Run：一整局连续官方生命周期。
- Official Task：完整任务列表中的一项。
- Official Attempt：官方尝试；目前没有公开确认的开始身份。
- Recorder Segment：本地记录分段，不是 Training Episode。
- Training Episode：离线QC派生的学习单元。

`local_attempt_key = run_id + task_id + settled_attempt_count` 只是团队本地身份键，不是官方 `attempt_id`。`settled_attempt_count` 是已经结算的次数，不是当前1-based尝试号。

## 8. 动作记录层次与训练标签边界

固定术语依次为 `proposed_action`、`selected_action`、`dispatched_action`、`publisher_call_attempted`、`publisher_call_succeeded`、`publisher_failure_reason`、`execution_feedback`。v1禁止使用含义模糊的单独 `action` 字段。

`selected_action` 不能单独充当实际控制标签；`dispatched_action` 的非null精确payload只是命令标签候选。`publisher_call_succeeded` 只证明本地调用正常返回，不证明DDS交付、Server接收、controller接受或机器人执行。observe-only数据默认不具备正式BC控制标签资格，最终training eligibility只由离线QC派生。

## 9. 公共类型分类

机器契约逐项列出当前真实存在的30个公共类型、模块、生产者/消费者、ROS边界、不变量和版本。核心稳定身份/几何/动作审计类型为 `frozen`；尚待算法或在线接线的状态/传感/规划类型为 `provisional`；`ExternalCandidate` 为 `restricted`。分类不要求为了“冻结”而改写现有dataclass。

## 10. 组员禁止事项

- 不得绕过ActionMux或直接发布五组官方控制话题。
- 不得把未控制position维补零、把反馈当命令，或把publisher返回提升为执行确认。
- 不得假定world等于odom、夹爪端点方向、官方cmd_vel上限或watchdog行为。
- 不得把Recorder Segment称为Training Episode，或把settled count称为当前attempt号。
- 不得从observe-only记录直接制造BC标签。
- 不得创建仓库不存在的通用ActionCandidate或NavigationCommand来“补齐”契约。

## 11. UNRESOLVED事实

机器契约保留11项：夹爪开闭方向、有效夹持值、官方cmd_vel接受范围、Server watchdog、Client重启后的controller target保持、Server接收确认、执行确认容差、world到odom关系、MMK2 FK输出frame、JointState effort单位、官方attempt开始身份。每项都含阻塞角色、所需测试和禁止假设；验证前不得填写经验值。

## 12. schema升级规则

字段重命名、单位变化、索引变化、时间源变化、null语义变化必须升级major。新增可选字段必须定义清晰的缺失语义，并保持旧读取方安全失败或兼容。未经队长审查不得修改Interface v1；不允许用文档措辞静默覆盖机器契约。
