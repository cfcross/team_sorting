# 4.4～4.6 官方离线机械臂标定工具

`scripts/arm_pick_place_calibration.py` 只允许在官方离线仿真中使用。它不是生产节点，
不启动TeamClient，不导入生产 `OfficialCommandPublisher`，也不能用于真实硬件。
`probe`、`plan-pick`、`plan-place`、`summarize`绝不创建控制publisher；只有用户显式运行
`execute-one-stage`或同进程人工门控的`execute-pick-calibration-sequence`时，脚本才直接发布
slide、左臂+夹爪和右臂+夹爪位置目标。

脚本从当前仓库读取下列事实，不复制数值：六个source-slot中心和yaw来自
`config.source_slots`，局部尺寸来自
`perception.estimator_3d.object_local_size_xyz_m`，限位、话题和消息类型来自Controller
Manifest及config。fixture必须标记 `source=stage2_calibration_fixture`，只表示阶段2冻结的
标定输入，不冒充生产 `ObjectEstimate3D`。

## 两个终端

Terminal 1启动Server。以下是本机已恢复的ROS_DOMAIN_ID=99官方离线命令：

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --name material_sorting_server \
  -e DISPLAY="${DISPLAY}" \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e MUJOCO_GL=glfw \
  -e MATERIAL_ENABLE_RENDER=1 \
  -e MATERIAL_USE_GS=1 \
  -e MATERIAL_RANDOMIZE=1 \
  -e MATERIAL_SEED=20260709 \
  -e MATERIAL_ENABLE_SCORE=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v material_sorting_cache_cu128:/opt/torch_ext \
  material_sorting:offline-server \
  python3 examples/material_sorting/material_sorting_server.py
```

Terminal 2只启动offline-client的普通Bash，不启动TeamClient：

```bash
mkdir -p "$HOME/arm_calibration_data"
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v "$HOME/project/team_sorting:/workspace/baseline:ro" \
  -v "$HOME/arm_calibration_data:/data:rw" \
  material_sorting:offline-client \
  bash
```

进入Terminal 2后：

```bash
cd /workspace/baseline
python3 scripts/arm_pick_place_calibration.py probe --tf-timeout-s 5
```

probe检查 `/joint_states`、Odom、TF、四个机械臂控制话题的真实类型、Server订阅、其他
publisher及当前17维JointState。官方Server的Odom header可能写成 `/odom`；本工具只在
官方离线标定边界去除frame前导斜杠，并要求实际TF链严格为 `odom -> base_link`。
官方环境不发布`footprint`，也不单独发布`world`，因此probe不再要求这两个TF。只有
ROS_DOMAIN_ID=99、话题类型、控制publisher独占、Server订阅、JointState、Odom frame和
`odom -> base_link`全部通过时，才在本标定工具内应用已冻结的`world==odom`事实。

TF2 Python API参数顺序是`lookup_transform(target_frame, source_frame, ...)`。工具严格调用
`target_frame="odom", source_frame="base_link"`，取得base_link在odom中的位姿；新进程
创建Buffer和TransformListener后最多通过ROS spin等待`--tf-timeout-s`秒，不使用固定sleep。
超时错误会区分frame缺失、两frame不可连接和Buffer检查异常；Buffer报告可用后的lookup
异常则单独标记。工具还会把TF位姿与Odom pose比较，平移允许0.02 m实时采样差异，四元数
使用`q`与`-q`等价的符号无关距离（容差0.02）。比较不通过时保持失败关闭。

probe输出必须核对：

```text
actual_odom_frame=odom
actual_base_frame=base_link
transform_source=tf:odom->base_link
world_equals_odom=true
official_offline_conditions_met=true
blockers=[]
published_control=false
```

同时必须确认`odom_tf_comparison.matches=true`；comparison中明确记录比较方向、平移误差和
符号无关四元数距离。

生产`ArmPlanner`的现有KDL输入契约仍使用`footprint`标签。工具在上述严格检查通过后，
不会把base_link实测到的微小roll/pitch冒充平面footprint。它从实时TF保留base_link的
`x/y/yaw`，将`z/roll/pitch`设为0并生成归一化纯yaw四元数，得到
`T_odom_virtual_footprint`；随后严格计算：

```text
T_virtual_footprint_object =
  inverse(T_odom_virtual_footprint) * T_odom_object
```

这个顺序先将odom物体位置减去虚拟底盘平移，再以底盘yaw的逆旋转进入局部平面坐标。
物体真实orientation不会被修改；转换后的物体局部Z轴仍必须通过生产ArmPlanner原有的
竖直守卫。最后工具才把虚拟平面坐标适配给现有规划入口，并在plan输入中记录
`scope=official_offline_calibration_tool_only`；这不表示ROS图中存在`footprint`，也不改变
生产坐标规则。

probe输出包含`raw_base_transform`、`planarized_virtual_footprint_transform`、
`raw_roll_pitch_yaw`、`virtual_footprint_roll_pitch_yaw`和
`planarization_scope=official_offline_calibration_only`。实时plan-pick另包含
`object_pose_in_odom`、`object_pose_in_virtual_footprint`和
`object_local_z_in_virtual_footprint`。这些命令始终输出`published_control=false`。

## 10秒只读时序采样

测试中的1000 ns只是单元测试夹具，禁止作为仿真参数。先显式选择本轮试验裕量系数，再对
JointState、Odom和`odom -> base_link` TF采样10秒：

```bash
export TIMING_SAFETY_FACTOR='<本轮人工审查的试验裕量系数，必须>=1>'
python3 scripts/arm_pick_place_calibration.py timing \
  --duration-s 10 \
  --safety-factor "$TIMING_SAFETY_FACTOR" \
  --output /data/timing-20260709.json
```

输出包含每路样本数、频率、最大间隔和最大/最小数据年龄。`transform_max_age_ns`和
`joint_state_max_age_ns`按输出中的basis生成并标记`TRIAL_NOT_FROZEN`，不会写回config。
本命令没有`ObjectEstimate3D`、完整pick流程或carry/place流程样本，所以
`object_estimate_max_age_ns`、`planned_context_max_age_ns`和
`confirmed_context_max_age_ns`保持null并明确标记BLOCKED，禁止由脚本猜值。

## 4.4候选底盘站位（官方离线仿真专用）

`plan-pick`的IK失败现在返回`IK_NO_COMPLETE_PATH`，不会仅凭“无解”自动归因为底盘站位。
输出中的`ik_diagnostics`按每个stage、左右单臂、双臂和每个slide候选记录末端Pose、官方
KDL原始返回类型/长度、失败分类和单臂可行性。只有在用户明确选择一个standoff试验值时，
才使用下列只规划命令。它读取实时Odom，并且唯一调用生产
`NavigationController.build_pick_goal()`生成候选站位；不复制站位公式、不创建
`/cmd_vel` publisher。standoff、位置容差和yaw容差都必须由CLI显式传入；没有隐藏的宽松
默认值。本轮`0.75 m / 0.01 m / 0.02 rad`只标记`TRIAL_NOT_FROZEN`，不写回config：

```bash
python3 scripts/arm_pick_place_calibration.py plan-base-stand \
  --input /data/pick-input.json \
  --standoff-m 0.75 \
  --position-tolerance-m 0.01 \
  --yaw-tolerance-rad 0.02 \
  --input-timeout-s 3 \
  --output /data/plan-base-stand.json
```

```bash
python3 -m json.tool /data/plan-base-stand.json
```

只有`valid=true`、`status=TRIAL_NOT_FROZEN`、`published_control=false`，且用户确认目标、
距离、初始转角和最终yaw后才能考虑执行。计划记录目标生成函数、实时Odom、阶段2fixture、
CLI standoff及两项试验容差和其余config字段各自来源，不能标记为verified/frozen。计划中的
`goal.position_tolerance/yaw_tolerance`必须与`navigation_parameters`完全一致；试验容差只能
比生产默认0.05 m/0.10 rad更严，不能借标定工具放宽生产容差。

执行命令只允许官方离线ROS_DOMAIN_ID=99，并要求严格确认字符串：

```bash
python3 scripts/arm_pick_place_calibration.py execute-base-stand \
  --plan /data/plan-base-stand.json \
  --official-offline-simulation \
  --confirm I_CONFIRM_OFFICIAL_OFFLINE_BASE_MOTION \
  --note "table_side_right 4.4候选站位"
```

脚本在整个命令期间复用同一个ROS节点、Odom订阅和`latest_odom`缓存，不会在每个控制周期
重建订阅。创建publisher前要求`/cmd_vel`其他publisher为0、subscriber恰好为1、Odom新鲜且
frame为odom。执行循环在每个约41.67 ms控制周期内持续spin接收ROS回调，并以约24 Hz调用
生产`NavigationController.update()`。官方实测Odom约23.989 Hz且最大间隔约49.77 ms，因此
单个控制周期没有新Odom属于正常时序：只要ROS当前时间减缓存消息时间戳不超过
150000000 ns，就允许复用该帧继续闭环；不会因此放宽0.15 s阈值。重复时间戳可以用于当期
控制计算，但不会被生产控制器重复计为到位稳定帧。

本轮授权边界为0.25 m/s、0.50 rad/s、Odom 0.15 s、位置0.01 m、yaw 0.02 rad、总超时
30 s，并要求连续3个不同Odom帧同时满足位置、yaw、线速度不超过0.01 m/s和角速度不超过
0.02 rad/s。从未收到Odom时分类为`NEVER_RECEIVED`；缓存年龄真正超过150 ms时分类为
`STALE`，两者都会进入零速停止路径。

到位、超时、异常、Ctrl-C和finally都会进入同一零速路径：持续发布`Twist(0,0)`、读取新
Odom取得连续3帧停稳证据，随后再明确发布3次零速。若停稳证据不足，执行结果强制失败。
如需人工停止，按一次Ctrl-C并等待JSON结果和`stop_evidence.confirmed_stopped=true`；不要
立即关闭容器。完整日志写入：

```text
/data/arm_calibration/<seed>/<scene>/base-stand-<timestamp>.json
```

日志还记录`control_tick_count`、`odom_received_count`、`odom_reused_tick_count`、每周期
Odom年龄、观测到的最大年龄、固定过期阈值和失败分类，用于区分正常复用与真实过期。

每个有效Odom状态还会用计划中未经修改的`target_object`和实测base pose计算
`arm_reachability_precheck`。它输出`actual_object_pose_in_virtual_footprint`、本轮
`expected=[0.75,0,0.834]`、standoff/lateral误差、相对计划终点yaw误差，以及是否同时满足
0.01 m/0.02 rad标定精度。该检查只变换坐标，不修改物体坐标、不调用或放宽IK；若导航报告
成功但该前置检查不满足，执行结果失败关闭。

查看最新一次到位与停稳摘要：

```bash
python3 -c 'import json; from pathlib import Path; f=max(Path("/data/arm_calibration/20260709/table_side_right").glob("base-stand-*.json"),key=lambda p:p.stat().st_mtime_ns); p=json.load(open(f)); print(json.dumps({"file":str(f),"execution_success":p.get("execution_success"),"final_distance_error_m":p.get("final_distance_error_m"),"final_yaw_error_rad":p.get("final_yaw_error_rad"),"arm_reachability_precheck":p.get("arm_reachability_precheck"),"stop_evidence":p.get("stop_evidence"),"published_control":p.get("published_control")},ensure_ascii=False,indent=2))'
```

## plan输入

不要手抄可能被终端截断的instruction。下列只读命令订阅完整`/material/instruction`，使用
仓库唯一`InstructionParser`选取task-id=1；TaskSpec的放置坐标、半径和接收时间只来自实时
消息及接收时钟。位置、局部尺寸、离散方向、frame和source slot只来自当前阶段2冻结配置。
fixture confidence不是官方消息事实，必须由用户显式标记为试验输入：

```bash
export FIXTURE_CONFIDENCE='<本轮阶段2fixture试验置信度，0到1>'
python3 scripts/arm_pick_place_calibration.py prepare-pick-input \
  --task-id 1 \
  --scene table_side_right \
  --seed 20260709 \
  --fixture-confidence "$FIXTURE_CONFIDENCE" \
  --instruction-timeout-s 5 \
  --output /data/pick-input.json
```

输出文件逐字段记录`live`、`config/derived_from_config`、`cli`、`trial`或`tool`来源，并保留
完整原始JSON。若实时消息或仓库缺少必需字段，命令返回`BLOCKED`且不生成文件；绝不猜测
`place_world_xyz`、`place_radius`、timestamp或其他字段。`object_id`只是本轮fixture身份，
不是官方body。后续`plan-pick --live`读取实时JointState、Odom和TF。支持scene：
`table_side_left`、`table_side_right`、`table_top`、`shelf_low`、`shelf_middle`、
`shelf_high`。

所有未冻结参数由用户显式设置为环境变量，不写回config：

```bash
export MIN_CONFIDENCE='<实测候选值>'
export TRANSFORM_MAX_AGE_NS='<实测候选值>'
export OBJECT_MAX_AGE_NS='<实测候选值>'
export JOINT_MAX_AGE_NS='<实测候选值>'
export CONTEXT_MAX_AGE_NS='<实测候选值>'
export PREGRASP_M='<本轮候选值>'
export CONTACT_OFFSET_M='<本轮候选值>'
export LIFT_M='<本轮候选值>'
export RETREAT_M='<本轮候选值>'
export MAX_SLIDE_DELTA_M='<本轮候选值>'
export MAX_ARM_DELTA_RAD='<本轮候选值>'
export MAX_GRIPPER_DELTA='<本轮候选值>'
export PREGRASP_DURATION_S='<本轮候选值>'
export GRASP_DURATION_S='<本轮候选值>'
export LIFT_DURATION_S='<本轮候选值>'
export RETREAT_DURATION_S='<本轮候选值>'
```

第一次table-side plan命令：

```bash
python3 scripts/arm_pick_place_calibration.py plan-pick \
  --live \
  --input /data/pick-input.json \
  --output /data/plan-pick.json \
  --trial min_object_confidence="$MIN_CONFIDENCE" \
  --trial transform_max_age_ns="$TRANSFORM_MAX_AGE_NS" \
  --trial object_estimate_max_age_ns="$OBJECT_MAX_AGE_NS" \
  --trial joint_state_max_age_ns="$JOINT_MAX_AGE_NS" \
  --trial planned_context_max_age_ns="$CONTEXT_MAX_AGE_NS" \
  --trial pregrasp_distance_m="$PREGRASP_M" \
  --trial grasp_contact_offset_m="$CONTACT_OFFSET_M" \
  --trial lift_distance_m="$LIFT_M" \
  --trial retreat_distance_m="$RETREAT_M" \
  --trial max_slide_waypoint_delta_m="$MAX_SLIDE_DELTA_M" \
  --trial max_arm_waypoint_delta_rad="$MAX_ARM_DELTA_RAD" \
  --trial max_gripper_waypoint_delta="$MAX_GRIPPER_DELTA" \
  --trial pregrasp_duration_s="$PREGRASP_DURATION_S" \
  --trial grasp_duration_s="$GRASP_DURATION_S" \
  --trial lift_duration_s="$LIFT_DURATION_S" \
  --trial retreat_duration_s="$RETREAT_DURATION_S"
```

任何`valid=false`、`joint_limits_ok=false`或`IK_NO_COMPLETE_PATH`都禁止执行。即使输出是
有效的calibration-only过渡工件，只要`automatic_execution_ready=false`也禁止执行。不能
仅凭IK失败再次移动底盘。

### 纯规划standoff扫描

下列命令在0.40～1.00 m范围扫描。每个候选都复用生产
`NavigationController.build_pick_goal()`生成底盘Pose，以该Pose构造官方离线标定专用虚拟
footprint，再调用真实`ArmPlanner`与官方MMK2Kdl检查全部抓取阶段。命令只创建订阅和KDL
对象，不创建`/cmd_vel`或机械臂publisher：

```bash
python3 scripts/arm_pick_place_calibration.py sweep-pick-stand \
  --input /data/pick-input.json \
  --standoff-min-m 0.40 \
  --standoff-max-m 1.00 \
  --standoff-step-m 0.05 \
  --input-timeout-s 3 \
  --output /data/sweep-pick-stand.json \
  --trial min_object_confidence="$MIN_CONFIDENCE" \
  --trial transform_max_age_ns="$TRANSFORM_MAX_AGE_NS" \
  --trial object_estimate_max_age_ns="$OBJECT_MAX_AGE_NS" \
  --trial joint_state_max_age_ns="$JOINT_MAX_AGE_NS" \
  --trial planned_context_max_age_ns="$CONTEXT_MAX_AGE_NS" \
  --trial pregrasp_distance_m="$PREGRASP_M" \
  --trial grasp_contact_offset_m="$CONTACT_OFFSET_M" \
  --trial lift_distance_m="$LIFT_M" \
  --trial retreat_distance_m="$RETREAT_M" \
  --trial max_slide_waypoint_delta_m="$MAX_SLIDE_DELTA_M" \
  --trial max_arm_waypoint_delta_rad="$MAX_ARM_DELTA_RAD" \
  --trial max_gripper_waypoint_delta="$MAX_GRIPPER_DELTA" \
  --trial pregrasp_duration_s="$PREGRASP_DURATION_S" \
  --trial grasp_duration_s="$GRASP_DURATION_S" \
  --trial lift_duration_s="$LIFT_DURATION_S" \
  --trial retreat_duration_s="$RETREAT_DURATION_S"
```

简短查看总体结论和推荐项：

```bash
python3 -c 'import json; p=json.load(open("/data/sweep-pick-stand.json")); print(json.dumps({k:p.get(k) for k in ("status","feasible_candidate_count","recommended_candidate","diagnostic_summary","failure_reason")}, ensure_ascii=False, indent=2))'
```

只有完整生产抓取轨迹有效、同一slide覆盖全部双臂阶段且关节限位与连续性均通过的候选才
计入`feasible_candidate_count`。推荐顺序依次为最小关节限位余量更大、最大路点变化更小、
从当前实时底盘移动更短。全部无解时返回`NO_FEASIBLE_STANDOFF`且不推荐移动；诊断原因分类
只描述KDL观测支持的范围，不把“无解”擅自提升为已证明的姿态、高度或底盘原因。所有候选
均为`TRIAL_NOT_FROZEN`且`published_control=false`。

### 4.4连续IK与calibration-only过渡诊断

官方offline-server镜像中的真实API已只读核对：`MMK2Kdl.inverse_kinematics()`参数为
`T_left`、`T_right`、`ref_pos`、`target_height`，双臂`ref_pos`严格按
`[slide,left×6,right×6]`排列；`ArmKdl.inverse_kinematics(pose, ref_pos)`支持6维seed。
官方解析器在关节限位内枚举分支，seed存在时按与seed的绝对关节变化选择最近分支；官方
`limit_joints()`才负责在官方限位允许时处理`2π`等价角。标定工具不会自行加减`2π`。

`plan-pick`和`sweep-pick-stand`现在额外输出`calibration_analysis`。第一个IK阶段使用输入
`RobotJointState`作为seed，后续阶段使用上一阶段选中的双臂解作为seed。官方solver可用时，
工具会保留官方入口实际返回的所有候选，先拒绝团队安全边界与官方公开边界交集之外的解，
再按加权绝对变化最小、最小限位余量更大的顺序选择。当前官方双臂固定高度入口在
`ref_pos=None`时内部下标访问`None`，因此不能通过该公开入口额外枚举无seed组合；工具明确
记录`analytic_enumeration_error`，不会复制官方肩部变换或猜测seed绕过。该逻辑只属于标定
工具，不修改生产`ArmPlanner`。

每个连续性检查都记录waypoint/stage、left/right、团队关节名称与索引、当前值、目标值、
signed/absolute delta、1.0 rad限制、按0.6 rad/s计算的最短时间、stage duration、速度限制
结论和该阶段的seed/分支信息。即使生产规划因连续性守卫失败，这些字段仍写入JSON。

如果完整四阶段双臂端点IK与限位均有效、没有更连续合法分支，并且唯一连续性失败是实际
JointState到首个pregrasp超限，`plan-pick`与compare复用同一个
`_calibration_transition_plan()`生成`calibration-only-plan`过渡候选，不复制第二套算法。
段数为：

```text
max(ceil(max_arm_delta/1.0), ceil(slide_delta/0.20))
```

每段机械臂变化不超过1.0 rad、slide不超过0.20 m；每段持续时间由机械臂变化/0.6、slide
变化/0.15和夹爪变化/0.6中的最大值决定，不能沿用原先整个pregrasp的2秒。左右夹爪在全部
过渡路点保持open=1.0，并用官方`MMK2Kdl.forward_kinematics()`记录左右末端位置。线性
关节分段只证明数值连续、限位、速度和FK可计算，不证明无碰撞。本标定任务不实现碰撞
检测器，也不增加视觉审核记录命令；用户只需在官方离线仿真窗口观察是否有明显异常。
碰撞字段仅说明没有执行自动检查，不阻塞手动单段执行：

```text
collision_check_available=false
collision_visual_verification_required=false
collision_verification_status=NOT_AUTOMATICALLY_CHECKED
status=TRANSITION_PLAN_READY_FOR_MANUAL_SINGLE_STAGE_SIMULATION
executable=false
published_control=false
```

该工件还明确设置`plan_artifact_valid=true`、`endpoint_ik_success=true`、
`automatic_execution_ready=false`和`single_stage_execution_ready=true`。前者继续禁止自动
连续执行全部阶段；后者只允许用户一次显式选择一个`transition-1/2/3`。顶层
`calibration_waypoints`按顺序保存初始真实JointState、全部`transition-N`、pregrasp、
grasp-open、grasp-close、short-lift和retreat；每项保存增量、速度、持续时间、限位余量和
FK。`transition_3_reaches_pregrasp=true`时，第三段成功后的`next_stage`直接是`grasp-open`，
不得重复执行完全相同的pregrasp目标。
`max_arm_waypoint_delta_rad>1.0`同样继续拒绝；3.2 rad只能用于历史plan-only诊断，不能执行。

精确站位后的本轮真实plan命令：

```bash
ROS_DOMAIN_ID=99 python3 scripts/arm_pick_place_calibration.py plan-pick \
  --live \
  --input /home/lifan/arm_calibration_data/pick-input.json \
  --output /home/lifan/arm_calibration_data/plan-pick-075-transition.json \
  --trial min_object_confidence=1.0 \
  --trial transform_max_age_ns=149730654 \
  --trial object_estimate_max_age_ns=149425773 \
  --trial joint_state_max_age_ns=149425773 \
  --trial planned_context_max_age_ns=149425773 \
  --trial pregrasp_distance_m=0.10 \
  --trial grasp_contact_offset_m=0.01 \
  --trial lift_distance_m=0.05 \
  --trial retreat_distance_m=0.10 \
  --trial max_slide_waypoint_delta_m=0.20 \
  --trial max_arm_waypoint_delta_rad=1.0 \
  --trial max_gripper_waypoint_delta=1.0 \
  --trial pregrasp_duration_s=2.0 \
  --trial grasp_duration_s=4.0 \
  --trial lift_duration_s=2.0 \
  --trial retreat_duration_s=2.0
```

只读查看过渡摘要：

```bash
python3 -c 'import json; p=json.load(open("/home/lifan/arm_calibration_data/plan-pick-075-transition.json")); t=p.get("transition_plan") or {}; print(json.dumps({"status":p.get("status"),"endpoint_ik_success":p.get("endpoint_ik_success"),"joint_limits_ok":p.get("joint_limits_ok"),"transition_required":p.get("transition_required"),"transition_segment_count":p.get("transition_segment_count"),"max_single_segment_joint_delta_rad":p.get("max_single_segment_joint_delta_rad"),"total_transition_duration_s":p.get("total_transition_duration_s"),"minimum_joint_limit_margin":p.get("minimum_joint_limit_margin"),"all_fk_checks_ok":p.get("all_fk_checks_ok"),"collision_verification_status":p.get("collision_verification_status"),"automatic_execution_ready":p.get("automatic_execution_ready"),"single_stage_execution_ready":p.get("single_stage_execution_ready"),"transition_3_reaches_pregrasp":p.get("transition_3_reaches_pregrasp"),"waypoint_stages":[w.get("stage") for w in p.get("calibration_waypoints",[])],"segments":t.get("segments"),"published_control":p.get("published_control")},ensure_ascii=False,indent=2))'
```

### 4.4手动单段过渡执行

旧版plan文件没有`single_stage_execution_ready`、分段stage标记和完整执行校验证据，会失败
关闭；必须先用上一节更新后的`plan-pick --live`命令重新生成计划，禁止手工补字段。

每次命令内部重新执行官方环境probe，并要求plan seed/scene显式匹配、其他控制Publisher为0、
JointState新鲜且与本段保存起点在0.01 m/0.01 rad/0.02夹爪容差内一致。段内计划证据必须保持
限位、步长、速度与FK全部通过，左右夹爪目标严格为1.0。执行期间复用与
`execute-base-stand`相同的持久订阅/latest-feedback缓存判定：24 Hz每个控制周期持续spin并
发布一次限速目标，不要求41.67 ms内必有新帧；约50 ms才到达的JointState可在缓存消息年龄
不超过0.131175150 s时复用。只有从未收到、真实过期、关节集合/有限性无效、消息时间戳倒退
或订阅异常才停止。连续3帧到位只计算不同消息时间戳的新反馈，不用缓存复用tick凑数。
transition-1准确命令：

```bash
ROS_DOMAIN_ID=99 python3 scripts/arm_pick_place_calibration.py execute-one-stage \
  --plan /home/lifan/arm_calibration_data/plan-pick-075-transition.json \
  --stage transition-1 \
  --expected-seed 20260709 \
  --expected-scene table_side_right \
  --official-offline-simulation \
  --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION \
  --max-slide-velocity-m-s 0.15 \
  --max-arm-velocity-rad-s 0.6 \
  --max-gripper-velocity-per-s 0.6 \
  --control-rate-hz 24 \
  --timeout-s 5 \
  --feedback-max-age-s 0.131175150 \
  --slide-tolerance-m 0.01 \
  --arm-tolerance-rad 0.01 \
  --gripper-tolerance 0.02 \
  --settle-cycles 3 \
  --note "观察transition-1；明显异常立即Ctrl-C"
```

transition-1成功且`next_stage=transition-2`后，下一条命令只把`--stage`和note分别改成
`transition-2`；第三段同理改成`transition-3`。不得跳段。每段结束后读取命令输出的
`log_path`并检查简短结果。日志在成功、超时、过期、异常和Ctrl-C路径均保留最终真实
JointState、17项绝对误差、控制tick数、实际接收帧数、缓存复用tick数、最大反馈年龄与失败分类：

```bash
python3 -c 'import json; from pathlib import Path; f=max(Path("/home/lifan/arm_calibration_data/arm_calibration/20260709/table_side_right").glob("transition-*-*.json"),key=lambda p:p.stat().st_mtime_ns); p=json.load(open(f)); print(json.dumps({k:p.get(k) for k in ("execution_success","stage","reached_target","final_max_joint_error","settled_cycles","control_tick_count","feedback_received_count","feedback_reused_tick_count","max_feedback_age_s","failure_classification","next_stage","published_control","log_path")},ensure_ascii=False,indent=2))'
```

仿真窗口出现明显异常时按一次Ctrl-C；工具用最新真实JointState发布一次位置保持，随后退出并
停止发布。不要继续后续transition，也不需要提交正式视觉PASS记录。

上述跨进程单段命令只保留用于诊断。真实日志已经证明：每次销毁Publisher后，机械臂状态可能
在下一进程启动前漂移，而且DDS endpoint discovery可能暂时保留上一进程。正式下一轮4.4必须
优先使用同一进程入口。新计划包含`execution_contract_version=2`；缺少该字段的旧计划严格
返回`OLD_PLAN_EXECUTION_CONTRACT`，必须重生成，禁止手工补JSON。

### 4.4同进程、逐段人工确认执行（推荐）

该入口只创建一个唯一命名ROS节点、一组持久订阅和一组Publisher。它依次支持：

```text
transition-1 -> transition-2 -> transition-3 -> grasp-open -> grasp-close
-> short-lift -> retreat
```

`transition-3`已经到达pregrasp，所以序列不会重复发送pregrasp。程序在第一段前和每段之间都
等待用户输入`I_CONFIRM_EXECUTE_NEXT_STAGE`；等待期间持续spin、检查反馈和Publisher身份并
发布最新真实位置保持。任何其他输入安全退出，任一阶段失败都禁止进入下一段。它仍设置
`automatic_execution_ready=false`，不是自动完整抓取命令。

Publisher检查不再用“总数必须立刻等于1”猜测身份。每个ROS节点名含进程和时间唯一后缀，
endpoint按node name/namespace分类为`SELF`或`EXTERNAL_OR_DDS_RESIDUAL`，日志保存topic、
node、GID、阶段和有界收敛样本。创建前只允许外部为0；创建后要求spine/左右臂各恰好一个
SELF、head为0。旧进程唯一名称不同，因此DDS残留不会被误认为当前SELF；残留只能在0.75秒
有界spin内消失，否则失败关闭，不能用无界sleep掩盖。

每段起点分别记录slide 0.01 m、arm 0.01 rad、gripper 0.02的最大误差、关节/侧别和逐关节
signed error。timeout必须不小于`max(计划时长, slide/0.15, arm/0.6, gripper/0.6)`，再加
3/24秒稳定帧时间、0.131175150秒反馈年龄预算和0.5秒明确安全裕量；不足会在创建Publisher
前返回建议最小值。插值到达最后一步时直接使用保存的精确目标值，避免浮点加法留下近似末帧。

实时任务检查与`prepare-pick-input`共用同一个持久ROS节点上的接收与
`InstructionParser`解析函数。订阅显式使用已经由`prepare-pick-input`在官方离线Server验证
可工作的`KEEP_LAST(depth=10) / RELIABLE / VOLATILE` QoS。sequence在上下文校验前按
`--instruction-timeout-s`持续spin该节点，因此延迟到达的instruction与同期JointState回调都会
被处理；超时或解析失败统一分类为`LIVE_CONTEXT_UNAVAILABLE`，且发生在创建控制Publisher前。
默认终端仅显示简短状态，完整probe、JointState、QoS和接收证据写入JSON日志；诊断时显式增加
`--verbose`才在终端打印完整内容。

准确命令：

```bash
ROS_DOMAIN_ID=99 python3 scripts/arm_pick_place_calibration.py execute-pick-calibration-sequence \
  --plan /data/plan-pick-075-transition-v2.json \
  --expected-seed 20260709 \
  --expected-scene table_side_right \
  --official-offline-simulation \
  --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION \
  --max-slide-velocity-m-s 0.15 \
  --max-arm-velocity-rad-s 0.6 \
  --max-gripper-velocity-per-s 0.6 \
  --control-rate-hz 24 \
  --timeout-s 8 \
  --feedback-max-age-s 0.131175150 \
  --instruction-timeout-s 5 \
  --slide-tolerance-m 0.01 \
  --arm-tolerance-rad 0.01 \
  --gripper-tolerance 0.02 \
  --settle-cycles 3 \
  --note "4.4同进程逐段观察；明显异常立即Ctrl-C"
```

正常到位、Ctrl-C、超时、反馈异常和用户安全退出都会持续发布最后真实JointState，并等待至少
3个不同时间戳的新反馈帧确认仍在容差内，再停止发布。日志字段`hold_evidence.status`必须为
`HOLD_CONFIRMED`；否则明确为`STOP_UNCONFIRMED`，不得继续。仅销毁Publisher不是停稳证据。

短摘要：

```bash
python3 -c 'import json; from pathlib import Path; f=max(Path("/data/arm_calibration/20260709/table_side_right").glob("pick-calibration-sequence-*.json"),key=lambda p:p.stat().st_mtime_ns); p=json.load(open(f)); print(json.dumps({"file":str(f),"execution_success":p.get("execution_success"),"completed_stages":p.get("completed_stages"),"failure_classification":p.get("failure_classification"),"failure_reason":p.get("failure_reason"),"hold_status":(p.get("hold_evidence") or {}).get("status"),"published_control":p.get("published_control"),"stages":[{k:s.get(k) for k in ("stage","execution_success","final_max_joint_error","settled_cycles","next_stage","stop_status")} for s in p.get("stage_results",[])]},ensure_ascii=False,indent=2))'
```

### 不连接Server的0.55/0.75纯规划比较

仓库此前没有生成compare state fixture的CLI；`probe`虽然订阅JointState，但其JSON不能直接
传给比较命令。现在必须在官方离线Server已由用户单独启动、ROS_DOMAIN_ID=99时，用下列
只读命令捕获一次；本命令只创建JointState/Odom/TF/instruction订阅，不创建任何publisher：

```bash
ROS_DOMAIN_ID=99 python3 scripts/arm_pick_place_calibration.py capture-pick-comparison-state \
  --scene table_side_right \
  --seed 20260709 \
  --joint-state-timeout-s 5 \
  --output /data/pick-comparison-state.json
```

捕获文件保留原始`name/position/velocity/effort/header`、header纳秒时间和工具接收时间，并
额外保存按`interfaces.JOINT_NAMES`排列的normalized三向量及可直接反序列化的
`joint_state`。文件结构如下；数组内容全部来自同一条真实消息，不允许手填或补零：

```json
{
  "schema": "team_sorting.arm_calibration.v1",
  "command": "capture-pick-comparison-state",
  "valid": true,
  "blockers": [],
  "source": "/joint_states",
  "scene": "table_side_right",
  "seed": 20260709,
  "evidence_source": "saved_official_joint_state",
  "raw_joint_state": {
    "name": ["原始17项消息顺序"],
    "position": ["原始17项"],
    "velocity": ["原始17项"],
    "effort": ["原始17项"],
    "header": {
      "frame_id": "原始值",
      "stamp": {"sec": 0, "nanosec": 0},
      "timestamp_ns": 0
    },
    "tool_received_at_ns": 0
  },
  "tool_received_at_ns": 0,
  "joint_state_header_timestamp_ns": 0,
  "joint_name_validation": {
    "expected_count": 17,
    "actual_count": 17,
    "exact_joint_set": true,
    "duplicate_names": [],
    "missing_names": [],
    "unexpected_names": [],
    "raw_name_order": ["原始17项消息顺序"],
    "normalized_name_order": ["团队标准17项顺序"],
    "required_explicit_names": ["slide、左右臂各6轴及左右夹爪名称"]
  },
  "normalized_joint_names": ["团队标准17项顺序"],
  "normalized_position": ["真实消息按团队顺序重排的17项"],
  "normalized_position_by_joint": {
    "slide_joint": "真实值",
    "left_arm_joint1…6": "各关节真实值",
    "left_arm_eef_gripper_joint": "真实值",
    "right_arm_joint1…6": "各关节真实值",
    "right_arm_eef_gripper_joint": "真实值"
  },
  "normalized_velocity": ["真实消息按团队顺序重排的17项"],
  "normalized_effort": ["真实消息按团队顺序重排的17项"],
  "joint_state": {
    "position": ["normalized_position"],
    "velocity": ["normalized_velocity"],
    "effort": ["normalized_effort"],
    "joint_names": ["团队标准17项顺序"],
    "timestamp_ns": 0
  },
  "base_state": {
    "position_xyz": [0.0, 0.0, 0.0],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "yaw": 0.0,
    "linear_velocity_xyz": [0.0, 0.0, 0.0],
    "angular_velocity_xyz": [0.0, 0.0, 0.0],
    "frame_id": "odom",
    "timestamp_ns": 0,
    "valid": true,
    "failure_reason": ""
  },
  "publisher_objects_created": false,
  "published_control": false
}
```

捕获时严格要求原始name恰好是团队17关节集合、无重复/缺失/额外名称，position、velocity、
effort均为17个有限数，接收时间不早于header时间。显式关节包括`slide_joint`、
`left_arm_joint1`～`left_arm_joint6`、`left_arm_eef_gripper_joint`、
`right_arm_joint1`～`right_arm_joint6`和`right_arm_eef_gripper_joint`；head两轴也必须存在。
任一条件不满足即`valid=false`并写入`blockers`，比较命令拒绝消费。

`pick-input.json`和state fixture是两个独立对象：前者保存任务与目标几何，后者的
`joint_state`保存真实比较快照。`compare-pick-standoffs`在生成任一候选前先将
fixture中的`joint_state`注入候选payload，不要求`pick-input.json`重复该字段。比较入口
严格校验fixture文件、schema、scene/seed、17关节集合与顺序、位置有限性及时间顺序；
缺字段使用`STATE_FIXTURE_FIELD_MISSING: <field>`失败，不返回裸`KeyError`。

fixture的header时间和工具接收时间均保留在比较输出的`state_fixture_capture`中，
并标记`state_fixture_mode=recorded_official_joint_state`。这是标定工具的离线快照语义：
`now_ns`使用该消息时间，不使用运行compare时的墙钟时间计算过期。此规则不适用于
生产实时执行。

保存后可只读验证关键证据：

```bash
python3 -c 'import json,math; from team_sorting.interfaces import JOINT_NAMES; p=json.load(open("/data/pick-comparison-state.json")); r=p["raw_joint_state"]; n=list(JOINT_NAMES); assert p["valid"] and p["blockers"]==[] and p["source"]=="/joint_states" and p["published_control"] is False; assert len(r["name"])==17 and len(set(r["name"]))==17 and set(r["name"])==set(n); assert all(len(r[k])==17 and all(math.isfinite(float(x)) for x in r[k]) for k in ("position","velocity","effort")); q=[r["position"][r["name"].index(x)] for x in n]; assert p["normalized_joint_names"]==n and p["normalized_position"]==q and p["normalized_position_by_joint"]==dict(zip(n,q)); t=r["header"]["stamp"]["sec"]*1000000000+r["header"]["stamp"]["nanosec"]; assert t==r["header"]["timestamp_ns"]==p["joint_state_header_timestamp_ns"]==p["joint_state"]["timestamp_ns"]<=p["tool_received_at_ns"]; print("VALID 17-joint fixture; published_control=false")'
```

随后关闭Server；在含官方KDL但不启动Server、TeamClient或任何ROS节点的环境运行纯规划比较：

```bash
python3 scripts/arm_pick_place_calibration.py compare-pick-standoffs \
  --input /data/pick-input.json \
  --state-fixture /data/pick-comparison-state.json \
  --output /data/compare-pick-standoffs.json \
  --trial min_object_confidence=1.0 \
  --trial transform_max_age_ns=149730654 \
  --trial object_estimate_max_age_ns=149425773 \
  --trial joint_state_max_age_ns=149425773 \
  --trial planned_context_max_age_ns=149425773 \
  --trial pregrasp_distance_m=0.10 \
  --trial grasp_contact_offset_m=0.01 \
  --trial max_slide_waypoint_delta_m=0.20 \
  --trial max_gripper_waypoint_delta=1.0 \
  --trial pregrasp_duration_s=2.0 \
  --trial grasp_duration_s=4.0 \
  --trial lift_duration_s=2.0 \
  --trial retreat_duration_s=2.0
```

命令内部固定比较本轮获准的两组试验：0.55 m使用lift=0.02 m、retreat=0.02 m；0.75 m
使用lift=0.05 m、retreat=0.10 m；两者都严格使用1.0 rad连续性守卫。它不导入`rclpy`、
不创建ROS runtime或publisher。每个候选明确输出连续IK、过渡段数、最大单段变化、最小
限位余量、逐路点FK、`collision_check_available=false`、
`collision_verification_status=BLOCKED_VISUAL_VERIFICATION`和`published_control=false`。

只读查看两个候选的比较摘要：

```bash
python3 -c 'import json; p=json.load(open("/data/compare-pick-standoffs.json")); keys=("standoff_m","lift_distance_m","retreat_distance_m","continuous_ik_branch_exists","transition_required","transition_segment_count","max_single_segment_joint_delta_rad","minimum_joint_limit_margin","all_fk_checks_ok","collision_verification_status","published_control"); print(json.dumps({"status":p.get("status"),"recommended_candidate":p.get("recommended_candidate"),"failure_reason":p.get("failure_reason"),"candidates":[{k:c.get(k) for k in keys} for c in p.get("candidates",[])]},ensure_ascii=False,indent=2))'
```

## 单阶段执行公共参数

下列速度、容差、24Hz和稳定周期来自当前仓库已完成的4.1～4.3标定；阶段超时仍由用户为
每次试验显式设置：

```bash
export STAGE_TIMEOUT_S='<本阶段人工确认的超时上限>'
export EXEC_COMMON='--official-offline-simulation'
```

每条执行命令均使用：

```text
--confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION
--max-slide-velocity-m-s 0.15
--max-arm-velocity-rad-s 0.6
--max-gripper-velocity-per-s 0.6
--control-rate-hz 24
--feedback-max-age-s 0.131175150
--slide-tolerance-m 0.01
--arm-tolerance-rad 0.01
--gripper-tolerance 0.02
--settle-cycles 3
```

### 4.4逐阶段命令

每条命令结束后先观察日志和仿真，再由用户决定是否运行下一条：

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-pick.json --stage pregrasp --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "观察pregrasp"
```

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-pick.json --stage grasp-open --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "观察双臂开放接近"
```

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-pick.json --stage grasp-close --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "记录是否稳定接触；closed=0.1不等于已抓稳"
```

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-pick.json --stage short-lift --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "记录箱子是否离开支撑面"
```

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-pick.json --stage retreat --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "观察撤退净空"
```

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-pick.json --stage return-start --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "返回本次plan起点"
```

## 4.5 plan与逐阶段命令

把 `/data/plan-pick.json` 中的 `planned_grasp_context`复制到place输入的
`grasp_context`。空载先设置 `load_state=empty`；它只复用规划几何，并在输出标记
`EMPTY_LOAD_KINEMATIC_ONLY_NOT_GRASP_CONFIRMATION`。带物体测试必须设置
`load_state=carrying_object`并提供阶段5/人工动态测试真正确认后的confirmed context。

```bash
export PREPLACE_HEIGHT_M='<本轮候选值>'
export RELEASE_OFFSET_M='<本轮候选值>'
export POST_RELEASE_RETREAT_M='<本轮候选值>'
export SETTLE_TIME_S='<本轮候选值>'
export PREPLACE_DURATION_S='<本轮候选值>'
export LOWER_DURATION_S='<本轮候选值>'
export RELEASE_DURATION_S='<本轮候选值>'
export POST_RELEASE_DURATION_S='<本轮候选值>'
```

显式设置放置候选参数后运行：

```bash
python3 scripts/arm_pick_place_calibration.py plan-place --live --input /data/place-input.json --output /data/plan-place.json --trial transform_max_age_ns="$TRANSFORM_MAX_AGE_NS" --trial joint_state_max_age_ns="$JOINT_MAX_AGE_NS" --trial confirmed_context_max_age_ns="$CONTEXT_MAX_AGE_NS" --trial preplace_height_m="$PREPLACE_HEIGHT_M" --trial release_offset_m="$RELEASE_OFFSET_M" --trial post_release_retreat_distance_m="$POST_RELEASE_RETREAT_M" --trial settle_time_s="$SETTLE_TIME_S" --trial max_slide_waypoint_delta_m="$MAX_SLIDE_DELTA_M" --trial max_arm_waypoint_delta_rad="$MAX_ARM_DELTA_RAD" --trial max_gripper_waypoint_delta="$MAX_GRIPPER_DELTA" --trial preplace_duration_s="$PREPLACE_DURATION_S" --trial lower_duration_s="$LOWER_DURATION_S" --trial release_duration_s="$RELEASE_DURATION_S" --trial post_release_retreat_duration_s="$POST_RELEASE_DURATION_S"
```

随后逐条运行；每条结束后停止并观察：

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-place.json --stage preplace --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "空载观察preplace"
```

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-place.json --stage lower --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "保持closed=0.1观察下降"
```

确认lower到位且没有提前张开后，才允许release：

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-place.json --stage release --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "到达release位姿后open=1.0"
```

此处不运行命令，等待用户观察；确认后才运行：

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-place.json --stage post-release-retreat --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "观察释放后撤退净空"
```

```bash
python3 scripts/arm_pick_place_calibration.py execute-one-stage --plan /data/plan-place.json --stage return-start --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_SIMULATION --max-slide-velocity-m-s 0.15 --max-arm-velocity-rad-s 0.6 --max-gripper-velocity-per-s 0.6 --control-rate-hz 24 --timeout-s "$STAGE_TIMEOUT_S" --feedback-max-age-s 0.131175150 --slide-tolerance-m 0.01 --arm-tolerance-rad 0.01 --gripper-tolerance 0.02 --settle-cycles 3 --note "返回放置plan起点"
```

每次执行的完整JSON自动写到：

```text
/data/arm_calibration/<seed>/<scene>/<stage>-<timestamp>.json
```

使用命令输出中的 `log_path` 进行摘要：

```bash
python3 scripts/arm_pick_place_calibration.py summarize --input "$LOG_PATH"
```

Ctrl-C、超时、JointState过期/乱序、越限、其他publisher或Server订阅消失均失败关闭；
脚本会用最新真实JointState请求位置保持并写日志。`closed=0.1`只是已验证闭合工作点，
绝不自动提升为稳定夹持、抓取视觉成功、放置视觉成功或完整端到端成功。

## 重启Server后的完整重规划顺序

旧`plan-pick-075-transition.json`缺少execution contract v2，下一轮必须严格按以下顺序重建：

```bash
cd /workspace/baseline
python3 scripts/arm_pick_place_calibration.py probe --tf-timeout-s 5
python3 scripts/arm_pick_place_calibration.py prepare-pick-input --task-id 1 --scene table_side_right --seed 20260709 --fixture-confidence 1.0 --instruction-timeout-s 5 --output /data/pick-input-v2.json
python3 scripts/arm_pick_place_calibration.py plan-base-stand --input /data/pick-input-v2.json --standoff-m 0.75 --position-tolerance-m 0.01 --yaw-tolerance-rad 0.02 --input-timeout-s 3 --output /data/plan-base-stand-v2.json
python3 scripts/arm_pick_place_calibration.py execute-base-stand --plan /data/plan-base-stand-v2.json --official-offline-simulation --confirm I_CONFIRM_OFFICIAL_OFFLINE_BASE_MOTION --note "table_side_right 0.75m精确站位"
python3 scripts/arm_pick_place_calibration.py prepare-pick-input --task-id 1 --scene table_side_right --seed 20260709 --fixture-confidence 1.0 --instruction-timeout-s 5 --output /data/pick-input-v2.json
python3 scripts/arm_pick_place_calibration.py plan-pick --live --input /data/pick-input-v2.json --output /data/plan-pick-075-transition-v2.json --trial min_object_confidence=1.0 --trial transform_max_age_ns=149730654 --trial object_estimate_max_age_ns=10000000000 --trial joint_state_max_age_ns=149425773 --trial planned_context_max_age_ns=10000000000 --trial pregrasp_distance_m=0.10 --trial grasp_contact_offset_m=0.01 --trial lift_distance_m=0.05 --trial retreat_distance_m=0.10 --trial max_slide_waypoint_delta_m=0.20 --trial max_arm_waypoint_delta_rad=1.0 --trial max_gripper_waypoint_delta=1.0 --trial pregrasp_duration_s=2.0 --trial grasp_duration_s=4.0 --trial lift_duration_s=2.0 --trial retreat_duration_s=2.0
```

执行序列会再次读取实时instruction、JointState、Odom和TF；seed/scene、任务字段、0.75m底盘
精度、计划schema/version及Publisher独占任一不匹配都失败关闭。Server重启或600秒时限结束后
不能继续消费旧实时计划。当前命令显式使用`planned_context_max_age_ns=10000000000`，因此
plan生成后必须在10秒内启动序列；超时就重新执行第二次prepare和plan，不能放宽或复用旧计划。

## 4.4～4.6完成语义

- 4.4：transition数值计划、FK、限位和单段执行基础设施已具备；`grasp-close=0.1`后的稳定
  接触、short-lift离台、是否滑落及retreat保持仍必须由下一轮动态观察，当前不得标为完成。
- 4.5：纯规划已有`preplace -> lower(closed=0.1) -> release(open=1.0) ->
  post-release-retreat`四阶段。带物执行仍要求4.4产生真实confirmed GraspContext；否则
  `plan-place(load_state=carrying_object)`失败关闭。放置站位导航属于阶段5联调。
- 4.6：只有4.4稳定夹持/试抬/撤退、放置站位、4.5四阶段及最终物体位置或裁判证据全部存在
  才能完成。脚本结束或关节到位不构成4.6成功；当前状态为阶段5联调阻塞。
