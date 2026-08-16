# MMK2 Controller Manifest V1

`team_sorting.controller_manifest.MMK2_CONTROLLER_MANIFEST_V1` 是纯 Python、不可变、无 ROS
运行时依赖的控制接口事实表。名称严格从 `interfaces.ACTION_NAMES` 派生；Manifest 不建立
第二套动作顺序，也不接收或解释任何原生八维策略动作。

## 运行时实测边界

固定场景 Server 的运行时 XML 为 `/tmp/material_competition_ros2_runtime.xml`，共 19 个
actuator。五个 Server 订阅端均实测为 `RELIABLE / KEEP_LAST(5) / VOLATILE`。

| 分组 | 官方话题 | 消息类型 | 元素数 |
|---|---|---|---:|
| base | `/cmd_vel` | `geometry_msgs/msg/Twist` | 2 |
| spine | `/spine_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 1 |
| head | `/head_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 2 |
| left_arm | `/left_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 7 |
| right_arm | `/right_arm_forward_position_controller/commands` | `std_msgs/msg/Float64MultiArray` | 7 |

## 19 维顺序与范围

`safe` 是团队允许 ActionMux 输出的范围；双臂六轴取原团队范围与 Server 运行时
`ctrlrange` 的交集。`runtime` 是本次固定场景实测的 actuator 范围。

| idx | 名称 | 分组 | 单位 | 语义 | safe | runtime |
|---:|---|---|---|---|---|---|
| 0 | base_v | base | m/s | velocity | [-0.25, 0.25] | 不适用 |
| 1 | base_w | base | rad/s | velocity | [-0.50, 0.50] | 不适用 |
| 2 | slide | spine | m | absolute_position | [-0.04, 0.87] | [-0.04, 0.87] |
| 3 | head_yaw | head | rad | absolute_position | [-0.50, 0.50] | [-0.50, 0.50] |
| 4 | head_pitch | head | rad | absolute_position | [-1.18, 0.16] | [-1.18, 0.16] |
| 5 | left_arm_joint_1 | left_arm | rad | absolute_position | [-3.14, 2.089] | [-3.151, 2.089] |
| 6 | left_arm_joint_2 | left_arm | rad | absolute_position | [-2.50, 0.181] | [-2.963, 0.181] |
| 7 | left_arm_joint_3 | left_arm | rad | absolute_position | [-0.094, 3.14] | [-0.094, 3.161] |
| 8 | left_arm_joint_4 | left_arm | rad | absolute_position | [-2.60, 2.60] | [-3.012, 3.012] |
| 9 | left_arm_joint_5 | left_arm | rad | absolute_position | [-1.859, 1.859] | [-1.859, 1.859] |
| 10 | left_arm_joint_6 | left_arm | rad | absolute_position | [-2.60, 2.60] | [-3.017, 3.017] |
| 11 | left_gripper | left_arm | dimensionless | normalized_position | [0, 1] | [0, 1] |
| 12 | right_arm_joint_1 | right_arm | rad | absolute_position | [-3.14, 2.089] | [-3.151, 2.089] |
| 13 | right_arm_joint_2 | right_arm | rad | absolute_position | [-2.50, 0.181] | [-2.963, 0.181] |
| 14 | right_arm_joint_3 | right_arm | rad | absolute_position | [-0.094, 3.14] | [-0.094, 3.161] |
| 15 | right_arm_joint_4 | right_arm | rad | absolute_position | [-2.60, 2.60] | [-3.012, 3.012] |
| 16 | right_arm_joint_5 | right_arm | rad | absolute_position | [-1.859, 1.859] | [-1.859, 1.859] |
| 17 | right_arm_joint_6 | right_arm | rad | absolute_position | [-2.60, 2.60] | [-3.017, 3.017] |
| 18 | right_gripper | right_arm | dimensionless | normalized_position | [0, 1] | [0, 1] |

运行时的 `lft_wheel_motor/rgt_wheel_motor [-35,35]` 是 MuJoCo 内部轮执行器范围；
`base_v/base_w` 是 `/cmd_vel` 的 `linear.x`（m/s）和 `angular.z`（rad/s），二者之间还有
底盘运动学与控制器转换。因此绝不能把 wheel motor 范围写成 base_v/base_w 范围；V1
继续使用团队保守速度范围。

## 已实测与仍未知

已实测：运行时 actuator 数量、顺序、`ctrlrange`、五个话题的类型和 Server 订阅 QoS；官方离线仿真已验证
左右夹爪均为 `open=1.0`、`closed=0.1`。该端点标定不证明对任意物体的稳定夹持值。
仍未知或未实测：Server watchdog、position target 的锁存与
节点重启语义、ROS publish 后的 Server 接收以及物理执行确认。

## 四层动作事实不能混用

1. `FinalAction` 是 ActionMux 计算并通过客户端检查的结果，不等于已发布。
2. dispatched payload 是实际交给某个 ROS publisher 的分组消息；当前尚无统一记录遥测。
3. Server controller target 是 Server 接收并可能再次处理后的目标，不是 JointState。
4. JointState 是物理/仿真反馈，不是 controller target，也不能证明命令来源。

本 Manifest 只冻结接口和范围，不能证明机器人实际执行。本阶段没有修改 Recorder，也没有
增加 dispatch telemetry。训练动作契约仍需后续增加 `commanded_mask`，区分主动命令、
ActionMux 接受、发布使能、实际 publisher payload 与反馈保持快照；没有
`ManipulationCommand` 时的 JointState 快照不能称为主动 17 维 hold。
