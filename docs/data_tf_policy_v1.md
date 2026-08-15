# Data/TF Policy v1

## 1. 目的与权威来源

本文件是
[`config/contracts/data_tf_policy_v1.json`](../config/contracts/data_tf_policy_v1.json)
的中文投影。机器契约是策略身份、话题、QoS、profile、raw/derived边界、动作标签资格及
QC枚举的规范来源。本契约身份为`team_sorting.data_tf_policy / 1`，policy ID为
`data_tf_policy_v1`，实现状态为`contract_only`。

B3需要在不改变既有数据身份和Recorder字段语义的前提下冻结新的采集策略，因此使用独立
契约，而不改写已冻结的Interface v1或Recorder schema v1。加载器会重新读取两份实际资源、
计算SHA-256并与契约引用匹配；只修改引用字符串不能通过校验。

B3A不修改Recorder运行逻辑，不增加TF录制，不实现压缩、降频、Indexer、QC或训练导出。

## 2. 冻结依赖

| 契约 | schema | version | SHA-256 |
|---|---|---:|---|
| `config/contracts/interface_v1.json` | `team_sorting.interface` | 1 | `ff95b964723e0b681427245f503d0e18322f133e1254cbbc1c1681a443bc1922` |
| `config/contracts/recorder_schema_v1.json` | `team_sorting.recorder` | 1 | `e7965c34a38c11d551d9943d8d614c05bc8e28e186432ad5ff4d0eed243225cf` |

## 3. raw与derived双轨

原始Segment、manifest、events、JSONL和rosbag只读，禁止原地修改。Indexer、QC、训练导出
和迁移只能写到：

```text
derived/indexer_v1/<deterministic_build_id>/
```

冻结输出名为：

```text
index_build.json
dataset_index.jsonl
segment_qc.json
run_qc.json
sample_index.jsonl
training_manifest.jsonl
```

派生输出必须记录source相对路径和SHA-256、工具名/版本/commit、配置SHA-256及确定性build
ID。宿主绝对路径不是训练数据身份。相同输入、工具版本和配置必须产生相同语义输出。

任何profile都不得自动删除raw。删除或归档只能是独立、显式、经人工批准的运营动作。

## 4. 当前raw基线与B3目标话题

当前13项raw基线保持不变：instruction、raw RGB、aligned raw Depth、CameraInfo、Odom、
JointState、ObjectEstimates、FSMStatus、CompetitionContext、FinalAction及三条referee话题。
它们的精确topic和ROS消息类型逐项冻结在机器契约中。

B3目标增加：

| Topic | Type | Role | B3A状态 |
|---|---|---|---|
| `/tf` | `tf2_msgs/msg/TFMessage` | `dynamic_transform` | 仅冻结策略，尚未录制 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | `static_transform` | 仅冻结策略，尚未录制 |

`/team/action_dispatch`的权威raw记录仍是`action_dispatches.jsonl`；B3A不要求重复写入bag。

## 5. TF与QoS

`/tf`订阅策略为best-effort、volatile、keep-last、depth 100。当前官方Publisher实测为
RELIABLE；best-effort订阅兼容该可靠Publisher，也保留对未来best-effort Publisher的兼容性。
当前已知动态关系只有`odom → base_link`。

`/tf_static`订阅策略为reliable、transient-local、keep-last、depth 1，且要求late join。
Recorder晚于静态Publisher启动时也必须能够取得其历史静态变换。

当前固定场景没有`/tf_static` Publisher，这是已知环境事实，不是所有profile的全局fatal。
只有某个profile明确要求的静态frame边缺失时，QC才使用
`tf_static_required_edge_missing`并按该profile判定error或fatal。

禁止假设`world == odom`。原始frame ID必须保留，raw消息不得改写。

## 6. frame规范化与CameraInfo

派生层保存：

```text
raw_frame_id
normalized_frame_id
normalization_rule=strip_leading_slashes
```

历史Odom raw frame可能为`/odom`，规范化后为`odom`；TF中的对应raw frame为`odom`。
两者都必须可追溯。空frame不得在没有依据时猜测。

当前CameraInfo的raw frame为空。派生层可在证据完整时绑定：

```text
raw_frame_id=""
effective_frame_id="head_camera"
binding_source="synchronized_rgb_depth"
```

该绑定来自同步RGB/Depth上下文，不得写回或替换原始CameraInfo消息。

## 7. 三档profile

### debug_audit

- `validation_status=validated_raw_baseline`
- RGB/Depth目标24 Hz；JointState/Odom/TF目标20–24 Hz。
- 保留raw RGB和raw Depth。
- 在线图像编码关闭，rosbag压缩关闭。
- 用于故障审计、容量基线和frame验证。
- 不适合长期批量保存，不允许自动形成formal BC标签。

### formal_collection_candidate

- `validation_status=provisional`
- `benchmark_required=true`
- RGB/Depth候选目标12 Hz；JointState/Odom/TF目标20–24 Hz。
- 派生训练候选目标10 Hz。
- 在线JPEG/PNG和在线zstd默认关闭。
- RGB派生候选为参数版本化JPEG；Depth只允许16位PNG或经验证的等价无损格式。
- 12 Hz和10 Hz只是候选策略值，尚未证明最优。
- 只有通过离线QC及动作标签资格检查后才允许训练派生。

### fast_regression

- `validation_status=provisional`
- RGB/Depth候选2 Hz；JointState/Odom/TF候选10 Hz。
- 只用于生命周期、TF、Indexer和QC冒烟测试。
- `training_allowed=false`。

三个profile当前都把`/tf_static`列为可用时记录的optional topic；profile后续声明具体必需静态
边时，缺边才触发对应QC finding。profile中的rate是策略目标，不表示当前Recorder具备降频。

## 8. 压缩和降频边界

当前在线默认固定为SQLite3、无rosbag压缩、无在线RGB/Depth编码。环境中虽存在zstd插件，
但比赛Client负载下的CPU、消息丢失、关闭延迟、临时空间和fail-closed行为尚未完成基准，
因此不得默认开启。

JPEG只适合参数版本化的RGB派生训练数据。Depth禁止有损压缩，候选格式为16位无损PNG或
经过验证的等价无损格式。

按话题降频尚未实现。未来实现必须保留原ROS时间戳和源消息身份；禁止使用没有记录规则的
“每N条取一条”。

## 9. 动作标签资格

固定层次为：

```text
proposed_action
selected_action / FinalAction
dispatched_action
publisher_call_succeeded
execution_feedback
```

这些层次不得混同。FinalAction单独不是正式BC标签；publisher调用成功只证明本地调用返回，
不证明DDS送达、Controller接受或机器人执行。

`observe_only=true`的数据可用于感知、状态、生命周期和诊断，但不得用于formal BC动作监督。
formal BC至少要求：observe-only关闭、精确dispatch存在、Action契约匹配、Context有效且未
finished、安全门语义通过，并存在后续execution feedback。

R0流水线冒烟可使用严格标注的受控单轨迹样例，但这不表示该数据已经取得正式BC资格。

## 10. QC与eligibility

severity固定为`fatal/error/warning/info`；eligibility固定为
`eligible/conditionally_eligible/ineligible`。机器契约冻结26个finding code，覆盖marker、
rosbag、Recovery、JSON/JSONL/SQLite、run/context、话题/时间、RGB/Depth、CameraInfo、
JointState、Odom、TF图、动作层和execution feedback。

当前没有`/tf_static` Publisher本身不构成全局fatal；observe-only数据申请formal BC时必须
是`ineligible`。

## 11. 后续实施边界

- B3B：修改Recorder话题、实现TF QoS与经验证的存储能力。
- B3C：实现只读Indexer与QC执行器。
- B3D：实现训练样本索引与eligibility派生。

B3A只冻结上述边界和机器规则，不能被描述为这些能力已经实现。
