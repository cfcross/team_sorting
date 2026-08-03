# Dataset Index/QC v1

## 范围

B3C core是ROS无关的只读索引器。它读取Recorder schema v1的bootstrap、run、Segment、
JSONL和rosbag2 SQLite，结合Data/TF Policy v1生成四种derived输出；不生成sample、训练
manifest，不解码图像，不实现逐帧动作配对、反馈时间窗或TF frame graph。

机器契约是`config/contracts/dataset_index_v1.json`。独立契约避免修改已经冻结且描述raw
生产者的Recorder schema v1；Data/TF Policy仍是话题、TF和训练资格的上游策略来源。

## Raw不可变与安全边界

Indexer不修复JSON、不清除ACTIVE、不创建COMPLETE、不写SQLite，也不在bootstrap、runs、
recovery或Segment目录创建文件。`derived/`不进入source fingerprint。运行前后比较raw文件
集合、SHA256、大小和`mtime_ns`。

dataset root和source artifact不得是symlink；来源路径必须位于root内且使用规范相对路径。
读取前组合`lstat`、`resolve`和`relative_to`校验。Python标准库无法为整个多级路径提供单个
原子`O_NOFOLLOW`事务，因此不声称能抵御拥有同一目录写权限的并发恶意换链者；采集卷应以
只读方式挂载并避免并发写入。JSON/JSONL/YAML/SQLite均有大小上限。严格JSON拒绝重复key、
NaN、Infinity和非法UTF-8。JSONL中间损坏失败关闭；只有无换行的最后一条不完整记录会形成
finding。SQLite使用`mode=ro&immutable=1`、`query_only=ON`、禁用extension、拒绝ATTACH和
写操作，并执行`integrity_check`。不根据数据库payload打开外部文件。

## Derived布局与确定性

```text
derived/indexer_v1/<deterministic_build_id>/
  index_build.json
  dataset_index.jsonl
  segment_qc.json
  run_qc.json
```

build ID是规范JSON的SHA256，输入包括契约/工具身份、QC配置hash、排序后的source相对路径和
SHA256以及依赖契约hash；排除时间、用户名、绝对路径、临时目录和derived。相同raw、工具和
配置产生同一语义build ID。`generated_at_utc`只存在于首次发布的`index_build.json`，因此
新建文件字节不承诺跨时间相同；重复运行校验已发布输出hash并复用原目录，不覆盖。

所有输出先写入`derived/indexer_v1`同文件系统的隐藏临时目录，逐文件及目录fsync后原子
rename。相同ID且hash一致返回`reused`；不一致时失败关闭且保留原输出。`index_build.json`
不自引用自身hash，`outputs`记录另外三项的SHA256和大小。

## 四类输出

- `index_build.json`：build/provenance、source fingerprint、依赖、输出hash和确定性声明。
- `dataset_index.jsonl`：每Segment一行，含marker、manifest、SQLite实际topic统计、时间范围、
  payload字节、Context/动作摘要、QC引用和eligibility；不反序列化消息payload。
- `segment_qc.json`：Segment findings及三用途资格。
- `run_qc.json`：Run manifest、Segment、bootstrap、结束状态、Recovery、聚合finding和资格。

Finding固定包含code、severity、evaluation_status、artifact、relative_path、message、evidence、
blocking_use_cases。severity为fatal/error/warning/info；evaluation_status为pass/fail/
not_applicable/not_evaluated，`not_evaluated`绝不等于pass，也不会自动阻塞。所有`fail`
finding的`blocking_use_cases`是强制资格规则，Segment和Run两级都必须在各自ceiling之后执行。

动作JSONL文件或任意JSON对象存在不代表有效动作存在。FinalAction必须通过
`strict_final_action_from_json`，ActionDispatchRecord必须通过
`strict_action_dispatch_from_json`；只要存在任意非法记录，即使同文件也有有效记录，也产生
`selected_action_invalid`或`action_dispatch_invalid` finding。动作摘要冻结记录总数、有效数、
无效数、文件记录存在、publish attempt、精确payload、本地publisher成功数，以及最终有效动作
布尔值。错误证据只保留有界摘要。

## Eligibility

- `diagnostic`：身份可信且SQLite可读时通常可用；未完成为conditionally_eligible。
- `perception`：硬要求RGB、Depth、CameraInfo、JointState、Odom、Context和合法时间跨度；缺TF
  时最多conditionally_eligible，不能声称frame闭环。
- `formal_bc`：observe-only或缺dispatch必为ineligible。即使非observe-only且有dispatch，
  因execution feedback窗口未实现，v1最多conditionally_eligible，绝不输出eligible。

formal BC候选要求同一条严格有效Dispatch同时满足：非observe-only、publish attempted、
attempted groups非空、mask至少一位true、action至少一位非null、且
`publisher_call_succeeded=true`。本地Publisher调用返回不代表DDS送达、Controller接受、机器人
执行或execution feedback确认。`tf_dynamic_missing`只硬阻塞formal BC；其他感知输入完整时，
perception按专门例外降级为conditionally eligible。

新build完成rename后还会再次校验raw快照。若raw变化，只安全删除本轮刚发布、发布前不存在且
身份严格匹配的build并抛错，不留下虚假complete目录。复用或预先存在的build在本轮后续失败时
绝不删除。正式并发安全边界仍是source volume只读挂载。

Recovery目录中的每个JSON报告均严格读取，拒绝非法UTF-8、重复key、非法JSON、错误顶层、
symlink和不安全路径。报告必须通过`source_segment_path`和/或`source_parent_run_id`关联现有
Segment或Run；路径逃逸、绝对路径、不存在的Segment及完全无法关联的报告均失败关闭，不能被
当作“没有Recovery”。

复用已有build前，目录项必须恰好是四个冻结文件，且全部是非symlink普通文件。Indexer重新以
本轮raw快照、QC配置、依赖契约和工具身份逐字段复核`index_build.json` provenance，并重新计算
三个payload的SHA256和size；额外文件、子目录、symlink或任意manifest篡改都会拒绝复用且保留
原目录。

发布清理使用显式状态：rename前失败仅清理temporary；rename成功后任何snapshot、open、fsync
或校验异常只清理本轮新发布且身份匹配的final。reused build永不进入清理路径。若清理自身失败，
错误同时包含主失败和cleanup失败，并通过异常链保留主根因。

Segment和Run manifest不是“能解析JSON即可”。Indexer从冻结Recorder schema v1的
`recorder_segment_schema.fields`与`run_manifest_schema.fields`读取描述符，数据驱动检查required、
nullable、严格类型、allowed values、minimum和SHA256，再检查物理目录身份、Segment sequence、
Run Segment列表、终态一致性及`recovery_required`。不维护第二份字段Schema。

ACTIVE/COMPLETE必须是严格marker对象，schema、marker名称、Segment/Run身份和对应wall time字段
均须匹配；无效COMPLETE不算完成。manifest的`bag_path`是唯一bag来源，必须是安全Segment相对
目录，Indexer不会回退读取硬编码`rosbag/`。

rosbag2 metadata使用拒绝重复key的YAML加载器，并验证wrapper、storage、duration、topic数组、
topic name/type/serialization、非负message count、topic唯一性和可选总计。语义损坏产生
`rosbag_metadata_invalid`，SQLite仍可用于诊断，但perception和formal BC不可用。

Perception硬要求RGB、Depth、CameraInfo、JointState、Odom和CompetitionContext；仅缺`/tf`降级
为conditional。Instruction、Referee、Object estimates、FSM status和bag内FinalAction不单独阻塞
perception。严格ActionDispatch JSONL仍是formal BC动作标签权威。

QC配置只允许`required_topics`和`required_static_edges`。topics必须是Data/TF Policy已知、唯一、
非空绝对ROS名称；B3C尚未解析静态边，因此edges只允许空数组。CLI将DatasetIndexError、
ValueError、TypeError和预期OSError统一为退出码2；`--json-summary`始终输出合法失败JSON。

`tf_static`当前无Publisher本身不是fatal。只有profile声明静态必需边且未来实现TF解码后才
评估缺边。本阶段TF graph、RGB/Depth精确对齐、关节维度、Odom frame、selected/dispatch
逐帧匹配和execution feedback均明确为`not_evaluated`。

## CLI

```bash
team_sorting_dataset_index --dataset-root /data/run --json-summary
team_sorting_dataset_index --dataset-root /data/run --check-only --json-summary
team_sorting_dataset_index --dataset-root /data/run --qc-config qc.json
```

`--check-only`不写文件。Finding不自动导致非0；源身份不可信、无法索引或发布失败返回2。
CLI没有修复、删除、清理或raw重写选项。B3D继续负责sample index、training manifest、图像
导出、精确观测—动作—反馈配对和训练集划分。
