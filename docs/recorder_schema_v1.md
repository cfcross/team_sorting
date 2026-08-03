# Recorder schema v1

## 1. 适用范围

本文件是 [`config/contracts/recorder_schema_v1.json`](../config/contracts/recorder_schema_v1.json)
的中文投影。机器契约是字段、空值、可变性、身份和演进规则的规范来源。本契约只冻结
数据协作边界，不修改 Recorder、ROS、控制链或比赛生命周期。

## 2. schema名称和版本

- `schema_name`：`team_sorting.recorder`
- `schema_version`：`1`
- `schema_status`：`frozen`
- Interface依赖：`team_sorting.interface / 1`

未知major版本必须fail closed；原始数据不得原地迁移。

## 3. 当前implementation phase

`implementation_phase=contract_only`。B1只提供机器契约、加载器、文档、安装声明和测试。
manifest、segment切换、events、marker及TF话题均尚未在运行时实现。

## 4. Online与Offline边界

Online Raw Recorder只旁路保存原始消息、接收事实、动作审计、上下文和rosbag状态，不
决定Training Episode、`next_observation`或训练资格。Commit C的Offline Indexer负责构建
消息索引和时间轴；Offline QC负责完整性、TF、动作资格和样本筛选；Replay Tool负责回放；
Converter在QC结果上生成LeRobot/OpenPI等训练格式。

Recorder不得推进FSM、影响ActionMux或OfficialCommandPublisher，也不得读取Server私有
真值。Recorder Segment不能称为Training Episode。

## 5. team-local run身份

规范字段是`run_id`，其scope固定为`team_local`。它是
`CompetitionRunCoordinator`生成的本地UUID，不是官方Server公开的稳定评测局ID。Client
重启时，即使同一物理比赛仍继续，新的`run_id`也可能变化。禁止仅凭相同
`task_set_fingerprint`自动合并两个`run_id`；重启连续性保持UNRESOLVED。

本地身份字段为`run_id`、`task_set_fingerprint`、`task_id`、
`settled_attempt_count`、`local_attempt_key`和`recorder_segment_id`。
`local_attempt_key=(run_id, task_id, settled_attempt_count)`只是团队本地键，不是官方
`attempt_id`；settled count表示已经结算的次数，不是当前1-based尝试号。

## 6. Segment、Attempt与Episode

- Team-local Run：由一个本地`run_id`标识的记录片段。
- Official Task：公开CompetitionContext选择的任务。
- Official Attempt：官方尝试，但其公开开始身份尚未确认。
- Recorder Segment：本地原始记录进程分段。
- Training Episode：由Commit C离线Indexer/QC派生的学习单元。

因此Recorder Segment既不等于Official Attempt，也不等于Training Episode。task切换或
settled count增加均不表示Server、机器人、物品或物理场景复位。Recorder不得创建官方
attempt ID。

## 7. canonical逻辑目录

v1冻结以下逻辑角色：

```text
dataset_root/
├── bootstrap/<recorder_segment_id>/
├── runs/<run_id>/
│   ├── manifest.json
│   ├── events.jsonl
│   └── segments/<recorder_segment_id>/
├── recovery/
└── legacy/
```

bootstrap segment创建时没有合法run，`parent_run_id=null`，以后也不得改成正式run-bound
segment。run-bound segment创建时必须已经知道非空run ID，parent ID此后不可变。
`recovery`只保存恢复报告或派生视图，不覆盖原始文件；`legacy`是逻辑兼容角色，不要求移动
旧目录。禁止用跨文件系统移动完成绑定，也禁止覆盖已有目录。B2才实现运行时切换。

## 8. manifest字段

`run_manifest_schema.fields`逐字段定义type、required、nullable、mutability、source和
semantics。重要规则：

- `run_id`和`task_set_fingerprint`不可变。
- `recorder_segment_ids`只能追加。
- end ROS/wall时间、`clean_shutdown`、`shutdown_reason`和`recovery_required`只能在结束时写入。
- commit、branch、dirty状态和镜像身份使用结构化unknown envelope，不用空字符串冒充未知。
- manifest记录Recorder及Interface契约SHA-256。
- Recorder不连接Docker socket；镜像身份由启动器注入，否则明确为unavailable。

## 9. segment字段

segment记录ID、parent run、kind、顺序、process/node起止时间、首末ROS时间、诊断PID、
可选容器身份、shutdown状态、bag路径/storage/exit code、JSONL清单、消息/drop/pairing/warning
计数、观察到的task与settled count、context有效/无效数及marker状态。PID只用于诊断，
不是身份或安全锁。

`segment_kind`仅允许`bootstrap`和`run_bound`。bootstrap必须使用null parent；run-bound必须
使用创建时已知的非空parent。任何segment都不得命名为episode。

## 10. event字段

v1冻结18种事件：Recorder开始、run绑定/变化、task/attempt变化、instruction/context更新、
动作选择、dispatch尝试/成功/失败、pairing问题、drop、bag开始/停止、shutdown请求、
Recorder结束及unclean检测。

每条事件包含schema、segment内稳定唯一event ID、类型、语义时间、接收ROS时间、接收
monotonic时间、run/task/settled身份、payload、source、validity、invalid reasons和源事件ID。
`receive_monotonic_ns`可保存，但scope固定为`process_local`，禁止跨进程排序。派生事件必须
引用源事件或源消息身份。B1存在event schema不表示当前已写`events.jsonl`。

## 11. raw artifacts

当前legacy产物严格为：

- `metadata.json`
- `final_actions.jsonl`
- `action_dispatches.jsonl`
- `action_frames.jsonl`
- `action_pairing_issues.jsonl`
- `fsm_status.jsonl`
- `competition_contexts.jsonl`
- `rosbag/`

当前没有最终`raw_records.jsonl`或`samples.jsonl`。RGB、Depth、CameraInfo、Odom和
JointState继续由rosbag保存。B1不要求在线生成sample，不绑定`next_observation`，也不决定
training eligibility；raw index与samples由Commit C离线生成。

## 12. 动作记录层次

沿用Interface v1术语：`proposed_action`、`selected_action`、`dispatched_action`、
`publisher_call_attempted`、`publisher_call_succeeded`、`publisher_failure_reason`、
`execution_feedback`。

- 完整proposed数值当前不可用。
- selected对应FinalAction，但FinalAction不等于实际发布动作。
- dispatched必须来自ActionDispatchRecord精确payload；head-only pitch不得补造。
- publisher调用返回只证明本地调用，不证明DDS、Server、controller或机器人执行。
- `controller_accepted`与`execution_confirmed`保持null/unresolved。
- execution feedback是bag中的后续JointState/Odom，当前未与动作配对。
- observe-only数据默认没有正式BC控制标签资格；资格由Commit C离线QC派生。

## 13. TF计划

当前bag不包含`/tf`、`/tf_static`或`/team/action_dispatch`。Dispatch已由结构化JSONL保存，
v1不要求重复进入bag。B3计划加入`/tf`和`/tf_static`；`/tf_static`的transient-local QoS及
late-join必须在真实Humble环境验证。禁止以`world==odom`代替TF。

## 14. shutdown与恢复

B2计划使用`ACTIVE`和`COMPLETE` marker，状态为`active`、`shutdown_requested`、
`complete`、`unclean_detected`和`recovery_required`。B1不创建marker。

`kill -9`和断电不能保证执行Python finally；unclean状态由后续启动或恢复扫描发现。
JSONL最后一个不完整尾部只可在派生恢复视图中忽略；中间损坏必须fail closed。原始JSONL
不得原地截断或改写。恢复输出必须记录源SHA-256、有效字节截止位置、原因和工具版本。

## 15. provenance

来源类别为`auto_detected`、`launcher_injected`、`unavailable`和
`prohibited_collection`。配置/契约hash、ROS domain、RMW、Python/package版本可自动获取；
Git与镜像身份优先由启动器注入。未注入镜像身份时使用null、unavailable状态和原因。

Recorder不得依赖Docker socket，不采集token、SSH信息、完整环境变量或无关路径。
hostname和container identity必须允许禁用或脱敏。

## 16. legacy兼容

没有Recorder schema身份的旧扁平目录按`legacy_flat_episode_v0`读取。不得原地修改旧数据，
也不得补造run边界、clean shutdown或TF完整性。旧`metadata.task`只是首任务兼容字段，不能
冒充完整任务集。迁移或统一视图必须写到新目录并记录源路径、源hash和迁移工具版本。

## 17. schema演进

字段删除/重命名、单位或时间源变化、ID或null语义变化、目录角色变化、事件语义变化及动作
标签资格变化必须升级major。同major只能新增具有明确缺失语义的字段；新增事件必须声明
`introduced_in`。Reader可保留未知字段，但不得改变已知字段解释；未知major必须fail closed。

## 18. UNRESOLVED

机器契约保留16项：官方attempt开始身份、Client重启run连续性、world→odom、MMK2 FK
frame、tf_static QoS、ROS时间回退、bag storage/compression、JSONL fsync周期、磁盘满策略、
多Recorder writer、完整proposed telemetry、Server接收确认、执行确认容差、离线观测动作
配对窗口、镜像身份注入责任和bag失败rollover。每项均定义影响组件、所需测试/决策和禁止假设。

## 19. B1尚未实现的能力

B1没有实现canonical目录、manifest、segment rollover、events、marker、unclean检测、TF录制、
raw index、samples、next observation或training eligibility。前两组运行时能力属于B2，TF属于
B3，离线派生属于Commit C。
