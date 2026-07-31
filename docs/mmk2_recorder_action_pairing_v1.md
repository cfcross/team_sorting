# MMK2 Recorder Action Dispatch Pairing V1

本契约是 Recorder 的旁路审计层。它同时保留原始合法 `FinalAction`、原始合法
`ActionDispatchRecord` 和严格配对结果，不参与 ActionMux、FSM 或任何官方控制发布。
实现位于 `team_sorting/recording_contracts.py`，ROS 接线位于
`team_sorting/ros_nodes.py::_create_recorder_node`。

## 四个 JSONL 文件

| 文件 | 内容 | 写入条件 |
|---|---|---|
| `final_actions.jsonl` | 原有 FinalAction V1 JSON | 每个 sequence 的首条合法 FinalAction，格式与既有文件完全一致 |
| `action_dispatches.jsonl` | 原始合法 ActionDispatchRecord V1 | 首条合法 dispatch 到达后立即写，不等待 FinalAction |
| `action_frames.jsonl` | `MMK2RecordedActionFrame` V1 | 两侧同 sequence、同生成时间且全部契约校验通过 |
| `action_pairing_issues.jsonl` | `MMK2ActionPairingIssue` V1 | 无效、重复、冲突、容量淘汰、超时和关闭孤儿 |

JSONL 沿用 `EpisodeRecorder._append_line`：每条以追加模式打开，写完整行后由 `with`
关闭完成 flush/close，不长期持有文件句柄，也不为每行 `fsync`。动作相关写入由
`RLock` 串行保护。突然断电仍可能损坏最后一行，离线工具必须容忍该边界。

所有状态变化采用 prepare/persist/commit：Pairer 先产生不可变计划；只有目标 JSONL
追加正常返回，Recorder 才提交 raw-persisted、pending 删除、近期 digest、终态账本或
closed 状态。Frame/Issue 写失败时两侧上下文仍保留，重试会续写缺失产物；已经成功写入
的 raw 不会重复追加。该保证是进程内重试语义，不是跨文件系统事务或进程重启恢复。

## 严格关联

Frame 的 `sequence` 必须同时等于 `final_action.sequence`、`action_dispatch.sequence`、
`action_dispatch.final_action_sequence` 和
`action_dispatch.decision.final_action_sequence`。Frame 的 `timestamp_ns` 必须同时等于
`final_action.timestamp_ns`、`action_dispatch.timestamp_ns` 和
`action_dispatch.decision.timestamp_ns`。

只按精确 sequence 配对，不用邻近 timestamp 猜测，不修正字段，也不补造缺失侧。
Frame 内嵌两份结构化对象；未知字段、未知版本、bool 冒充整数、NaN/Inf 均被拒绝。

## 异步、重复和有界等待

两个 pending map 分别保存 FinalAction 与 Dispatch。任一侧先到都只入表并立即返回；
另一侧到达后同步校验并生成至多一个 Frame，回调不阻塞等待。

- pending 中同侧 canonical JSON 相同：保留首条，写 `duplicate_identical_*`；
- 同 sequence 但内容不同：不覆盖首条，写 `duplicate_conflicting_*` 和双方 SHA-256；
- sequence 已完成后再到：写 `late_duplicate_*`，不重新打开、不生成第二帧；
- completed LRU 只保存近期双方 digest，超出配置后遗忘最老摘要；独立的精确终态
  sequence 区间账本继续阻止旧 sequence 重开。摘要已淘汰的重放会 fail closed 并写
  `late_duplicate_*`，而不是生成第二个 Frame；
- pending 超容量时按 `(received_monotonic_ns, insertion_order, sequence)` 淘汰最老项；
- 等待年龄只用 `time.monotonic_ns()`，达到 `max_wait_ns` 边界即由独立定时器清理；
- 节点关闭先取消 prune timer，再把所有 pending 写为 `shutdown_orphan`；重复关闭安全。

配置入口为 `recorder.action_pairing`：`enabled` 是严格 bool，容量、等待纳秒和预览长度是
非 bool 正整数，定时器周期是正有限秒数。raw preview 同时受配置值和 65536 字符契约
硬上限约束，摘要固定使用 SHA-256。`topics.action_dispatch` 是订阅话题的唯一来源；
本轮故意不把该话题加入 `recorder.rosbag_topics`。

`raw_payload_preview` 的契约硬上限是 65536 字符，并进一步受配置值限制；`detail` 的
硬上限是 4096 字符。未知/缺失字段错误只报告总数、最多八项有界预览以及明确的
`*_truncated` 标记，不展开任意长字段列表。

## 语义限制

`action_frames.jsonl` 只表示：Recorder 同时收到了同一控制周期、相互一致的
FinalAction 与本地 ActionDispatchRecord。它不表示 DDS delivery 已确认、Server 已收到、
controller 已接受、MuJoCo target 已采用、机器人已执行、当前帧适合训练或 Episode 成功。
因此 `controller_accepted` 和 `execution_confirmed` 在 V1 中必须为 `null`，且没有
`training_eligible` 字段。

后续 QC 至少需要分类 `dispatch_mode=none`、`publisher_call_succeeded=false`、
`failed_groups`、`safety_override_mask`、`FinalAction.valid=false` 以及两个未知确认字段。
本轮不实现 Episode/attempt 边界、训练筛选、LeRobot/OpenPI 转换，也不覆盖 emergency
base stop 等紧急安全控制事件遥测。Visual2 候选尚未合入，且不属于本契约任务。
