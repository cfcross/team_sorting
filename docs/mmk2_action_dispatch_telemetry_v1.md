# MMK2 Action Decision 与 Dispatch Telemetry V1

本契约补充而不替换既有 `FinalAction` 和 `/team/final_action`。公共不可变类型及严格 JSON
编解码定义在 `team_sorting.interfaces`；`ActionMux` 在一次仲裁中同时生成 `FinalAction`
和 `ActionMuxDecision`，ROS 边界再生成一条稳定关联的 `ActionDispatchRecord`。

## ActionMuxDecision V1

- `requested_mask[19]`：候选明确请求的维度。显式零速 `BaseCommand` 仍请求 base 两维；
  `ManipulationCommand=None` 不请求任何位置轴。
- `commanded_mask[19]`：通过有效性、TTL、反馈和 FSM 检查后接受的普通业务命令。
- `clipped_mask[19]`：已接受候选中被 ActionMux 逐维限幅的维度；裁剪不取消 commanded。
- `safety_override_mask[19]`：FSM stop、无效反馈、无效或过期候选导致的安全覆盖。
- 两类候选分别记录 presence、受控 disposition 和可证明的 source；不解析
  `failure_reason`猜测结果。
- `sequence/timestamp_ns/final_action_sequence` 与同周期 `FinalAction` 稳定关联。

presence、source、disposition 与 mask 是一个不可拆分的不变量：候选不存在时 source 必须
为`none`、disposition必须为`absent`且对应 requested/commanded/clipped 全false；候选
存在时source必须匹配实际类型且disposition不能为`absent`。`accepted`必须接受对应全部
requested维度；拒绝、过期或安全覆盖不得保留commanded/clipped。保留的
`partially_accepted`仅允许接受requested的严格非空子集。

## ActionDispatchRecord V1

团队内部话题为 `/team/action_dispatch`，类型 `std_msgs/msg/String`。每当控制周期已经生成
`FinalAction`就发布一条严格 JSON；使用 `allow_nan=false`，解码拒绝 NaN、Inf、bool
冒充数值、错误维度、未知字段及未知 schema 版本。

记录包含 calculated、发布门、publisher是否创建、实际调用结果、模式、逐组记录，以及
固定19项的 `dispatched_action`/`dispatched_mask`。未交给 ROS publisher 的维度为 JSON
`null`，禁止补零。每个分组保存官方话题、消息类型、attempted/succeeded、失败原因和
真正构造的 payload：Twist 完整保存 linear/ angular 各 x/y/z 六项，数组话题保存实际
`data`及长度。

顶层`timestamp_ns`必须与内嵌`ActionMuxDecision.timestamp_ns`完全一致，sequence与
`final_action_sequence`也必须稳定关联；严格JSON解析会重新执行这些构造不变量。

## 三种模式

- `none`：observe-only、发布器未创建、发布门关闭或本周期未授权。无官方调用，19项均null。
- `head_only`：只允许head分组。yaw来自`FinalAction[3]`，pitch来自fresh-reset作用域内
  的controller-target shadow；因此它可以且确实可能不同于JointState来源的
  `FinalAction[4]`。
- `full`：保持既有 base→spine→head→left_arm→right_arm 顺序，但当前控制周期没有启用
  该路径。attempted groups只能是该顺序的连续前缀（允许空前缀）；最多一个失败组且只能
  位于前缀末尾。若中间调用失败，前序成功、当前失败及后序未尝试分别保留，异常仍按原
  语义停止。

`publisher_call_succeeded=true`只表示本地调用正常返回。V1无法观测 DDS delivery、Server
接收、controller target锁存或机器人执行，因此 `controller_accepted` 和
`execution_confirmed`始终为null。

Recorder现已通过`topics.action_dispatch`订阅该团队遥测并进行旁路落盘/严格配对，但
仍不把`/team/action_dispatch`加入rosbag列表。记录侧规则见
[`mmk2_recorder_action_pairing_v1.md`](mmk2_recorder_action_pairing_v1.md)；任何落盘和
配对都不得参与控制，也不得把本地调用事实提升为执行确认。
