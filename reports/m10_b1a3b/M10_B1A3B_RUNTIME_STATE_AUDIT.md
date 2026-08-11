# M10-B1-A.3-B Runtime State19 Early-Phase Audit

## Recovery result

`RUNTIME_STATE_RECOVERY = INCOMPLETE`

This is a runtime-only statement. It makes no checkpoint or policy-quality judgment.

The common `t=0` is request `3` at candidate SQLite bag timestamp `1786256185709649153` ns.

The exact TeamClient callback receipt/activation timestamp is absent; the candidate SQLite bag timestamp is the same observable proxy used by M10-B1-A.3-A; its absolute proxy error is not bounded by the archive.

response_latency_ms begins inside PolicyClient.predict after snapshot/adaptation, while the candidate bag timestamp occurs after response completion; the missing pre-request and post-response intervals leave the exact cached sensor frames unidentifiable. The archive provides no finite bound for that timing error.

The tables and row-level export are therefore the closest observable reconstruction, not a claim that the exact per-request cached sensor frame was recovered.

## Actual Adapter state contract

The inspected Adapter constructs float32 state19 as `Odom linear.x`, `Odom angular.z`, followed by 17 JointState positions rearranged by `JOINT_NAMES`. Commands and FinalAction are not used as state.

| State index | Runtime source |
|---:|---|
| 0 | odom.twist.twist.linear.x |
| 1 | odom.twist.twist.angular.z |
| 2 | slide_joint |
| 3 | head_yaw_joint |
| 4 | head_pitch_joint |
| 5 | left_arm_joint1 |
| 6 | left_arm_joint2 |
| 7 | left_arm_joint3 |
| 8 | left_arm_joint4 |
| 9 | left_arm_joint5 |
| 10 | left_arm_joint6 |
| 11 | left_arm_eef_gripper_joint |
| 12 | right_arm_joint1 |
| 13 | right_arm_joint2 |
| 14 | right_arm_joint3 |
| 15 | right_arm_joint4 |
| 16 | right_arm_joint5 |
| 17 | right_arm_joint6 |
| 18 | right_arm_eef_gripper_joint |

## Coverage and row semantics

First 20 s contains 66 candidate-associated valid policy responses; 52 lie in 0–15 s. 65 pass the runtime latency gate.

Odom and JointState selected rows have equal message-header timestamps in 66/66 cases.

`time_s` is candidate activation time relative to the common `t=0`. `raw_action[0..18]` is row 0 of that candidate's model action chunk. The estimated request time and both selected sensor timestamps/ages are retained in CSV and JSON.

## Four five-second phases

Each cell is `count / mean / std / min / P05 / P50 / P95 / max`.

| Phase | Dimension | Statistics |
|---|---|---|
| 0-5s | state[0] base_v_feedback | 18.000000 / -0.025808 / 0.037719 / -0.125144 / -0.095139 / -0.002471 / 0.001080 / 0.002149 |
| 0-5s | state[1] base_w_feedback | 18.000000 / 0.135986 / 0.121289 / -0.082899 / -0.044241 / 0.135386 / 0.286391 / 0.440020 |
| 0-5s | state[2] slide_joint | 18.000000 / 0.024735 / 0.022521 / -0.031198 / -0.000636 / 0.024354 / 0.065910 / 0.078386 |
| 0-5s | state[3] head_yaw_joint | 18.000000 / -0.000259 / 0.003622 / -0.009508 / -0.005261 / -0.000038 / 0.004224 / 0.005357 |
| 0-5s | state[4] head_pitch_joint | 18.000000 / 0.002753 / 0.000492 / 0.001975 / 0.002093 / 0.002657 / 0.003571 / 0.003923 |
| 0-5s | state[5] left_arm_joint1 | 18.000000 / -0.010495 / 0.011330 / -0.031644 / -0.029208 / -0.008556 / 0.002262 / 0.012053 |
| 0-5s | state[6] left_arm_joint2 | 18.000000 / -0.018865 / 0.026494 / -0.076466 / -0.063947 / -0.009965 / 0.012078 / 0.021074 |
| 0-5s | state[7] left_arm_joint3 | 18.000000 / -0.017942 / 0.010629 / -0.039990 / -0.028976 / -0.020749 / 0.000002 / 0.000030 |
| 0-5s | state[8] left_arm_joint4 | 18.000000 / -0.181888 / 0.180687 / -0.558692 / -0.495429 / -0.132257 / 0.017867 / 0.033463 |
| 0-5s | state[9] left_arm_joint5 | 18.000000 / -0.243809 / 0.203302 / -0.678533 / -0.527819 / -0.240190 / 0.006177 / 0.025018 |
| 0-5s | state[10] left_arm_joint6 | 18.000000 / 0.089710 / 0.136707 / -0.040843 / -0.013472 / 0.048175 / 0.392311 / 0.495673 |
| 0-5s | state[11] left_arm_eef_gripper_joint | 18.000000 / 0.883800 / 0.311546 / 0.002791 / 0.002791 / 0.994723 / 0.997674 / 0.997861 |
| 0-5s | state[12] right_arm_joint1 | 18.000000 / -0.002131 / 0.006261 / -0.015711 / -0.010478 / -0.001982 / 0.009465 / 0.010676 |
| 0-5s | state[13] right_arm_joint2 | 18.000000 / 0.002660 / 0.022036 / -0.047126 / -0.031653 / 0.000001 / 0.036503 / 0.040363 |
| 0-5s | state[14] right_arm_joint3 | 18.000000 / -0.002039 / 0.016319 / -0.036598 / -0.033531 / -0.002379 / 0.021199 / 0.022774 |
| 0-5s | state[15] right_arm_joint4 | 18.000000 / 0.031801 / 0.042571 / -0.036799 / -0.032313 / 0.026637 / 0.097195 / 0.125222 |
| 0-5s | state[16] right_arm_joint5 | 18.000000 / -0.038418 / 0.036912 / -0.111926 / -0.095890 / -0.028979 / 0.014477 / 0.015322 |
| 0-5s | state[17] right_arm_joint6 | 18.000000 / -0.070677 / 0.075869 / -0.195787 / -0.183605 / -0.080050 / 0.021695 / 0.101668 |
| 0-5s | state[18] right_arm_eef_gripper_joint | 18.000000 / 0.884726 / 0.311933 / 0.002664 / 0.002664 / 0.996641 / 1.000935 / 1.002773 |
| 5-10s | state[0] base_v_feedback | 16.000000 / -0.086350 / 0.067145 / -0.191640 / -0.168848 / -0.096330 / 0.013035 / 0.103775 |
| 5-10s | state[1] base_w_feedback | 16.000000 / 0.408307 / 0.132476 / 0.250065 / 0.282276 / 0.341877 / 0.639936 / 0.752462 |
| 5-10s | state[2] slide_joint | 16.000000 / 0.051055 / 0.024570 / 0.009664 / 0.021456 / 0.045562 / 0.089425 / 0.097084 |
| 5-10s | state[3] head_yaw_joint | 16.000000 / 0.001787 / 0.009763 / -0.014569 / -0.010229 / 0.000702 / 0.017807 / 0.019129 |
| 5-10s | state[4] head_pitch_joint | 16.000000 / 0.002688 / 0.000612 / 0.001706 / 0.001854 / 0.002586 / 0.003604 / 0.003952 |
| 5-10s | state[5] left_arm_joint1 | 16.000000 / -0.013109 / 0.018858 / -0.039026 / -0.037876 / -0.013282 / 0.014654 / 0.038976 |
| 5-10s | state[6] left_arm_joint2 | 16.000000 / -0.014470 / 0.017462 / -0.042860 / -0.042479 / -0.012254 / 0.006173 / 0.011747 |
| 5-10s | state[7] left_arm_joint3 | 16.000000 / -0.010950 / 0.012231 / -0.034670 / -0.031899 / -0.009476 / 0.008275 / 0.008756 |
| 5-10s | state[8] left_arm_joint4 | 16.000000 / -0.228624 / 0.058169 / -0.361009 / -0.340229 / -0.216904 / -0.148797 / -0.140880 |
| 5-10s | state[9] left_arm_joint5 | 16.000000 / -0.462736 / 0.095415 / -0.700002 / -0.652985 / -0.450579 / -0.351482 / -0.326935 |
| 5-10s | state[10] left_arm_joint6 | 16.000000 / 0.920077 / 0.291347 / 0.223401 / 0.270466 / 0.974313 / 1.297107 / 1.387874 |
| 5-10s | state[11] left_arm_eef_gripper_joint | 16.000000 / 0.998547 / 0.005264 / 0.987009 / 0.989713 / 0.998087 / 1.007044 / 1.008638 |
| 5-10s | state[12] right_arm_joint1 | 16.000000 / 0.000657 / 0.016890 / -0.027135 / -0.026771 / 0.001421 / 0.021377 / 0.028074 |
| 5-10s | state[13] right_arm_joint2 | 16.000000 / -0.015895 / 0.013878 / -0.038510 / -0.032839 / -0.019434 / 0.004587 / 0.019004 |
| 5-10s | state[14] right_arm_joint3 | 16.000000 / 0.010824 / 0.022474 / -0.042252 / -0.025567 / 0.012246 / 0.041011 / 0.050991 |
| 5-10s | state[15] right_arm_joint4 | 16.000000 / 0.157288 / 0.349658 / -0.086276 / -0.066594 / 0.019132 / 0.936992 / 1.233856 |
| 5-10s | state[16] right_arm_joint5 | 16.000000 / 0.037414 / 0.086249 / -0.089382 / -0.039479 / 0.024239 / 0.202646 / 0.303897 |
| 5-10s | state[17] right_arm_joint6 | 16.000000 / -0.099299 / 0.109868 / -0.306810 / -0.276520 / -0.110622 / 0.082162 / 0.162118 |
| 5-10s | state[18] right_arm_eef_gripper_joint | 16.000000 / 0.997365 / 0.003394 / 0.989935 / 0.992419 / 0.997541 / 1.001234 / 1.007260 |
| 10-15s | state[0] base_v_feedback | 18.000000 / -0.012318 / 0.016546 / -0.053168 / -0.037816 / -0.007623 / 0.005676 / 0.017409 |
| 10-15s | state[1] base_w_feedback | 18.000000 / 1.003668 / 0.203263 / 0.584262 / 0.592323 / 1.116130 / 1.176920 / 1.219614 |
| 10-15s | state[2] slide_joint | 18.000000 / 0.127313 / 0.063271 / 0.064243 / 0.064295 / 0.112363 / 0.227942 / 0.277503 |
| 10-15s | state[3] head_yaw_joint | 18.000000 / 0.004272 / 0.007602 / -0.008397 / -0.007678 / 0.003062 / 0.017249 / 0.019199 |
| 10-15s | state[4] head_pitch_joint | 18.000000 / 0.002864 / 0.000471 / 0.002026 / 0.002199 / 0.002862 / 0.003518 / 0.003634 |
| 10-15s | state[5] left_arm_joint1 | 18.000000 / 0.009270 / 0.011936 / -0.016349 / -0.010566 / 0.009221 / 0.027052 / 0.037759 |
| 10-15s | state[6] left_arm_joint2 | 18.000000 / -0.045021 / 0.018157 / -0.087164 / -0.080084 / -0.040446 / -0.023987 / -0.014360 |
| 10-15s | state[7] left_arm_joint3 | 18.000000 / 0.029154 / 0.014730 / 0.003692 / 0.008963 / 0.028082 / 0.055689 / 0.059341 |
| 10-15s | state[8] left_arm_joint4 | 18.000000 / -0.156319 / 0.067035 / -0.302145 / -0.267915 / -0.140478 / -0.062474 / -0.061279 |
| 10-15s | state[9] left_arm_joint5 | 18.000000 / -0.651443 / 0.064161 / -0.784193 / -0.765734 / -0.652922 / -0.565899 / -0.546543 |
| 10-15s | state[10] left_arm_joint6 | 18.000000 / 1.428369 / 0.038729 / 1.337132 / 1.350424 / 1.438708 / 1.468677 / 1.482662 |
| 10-15s | state[11] left_arm_eef_gripper_joint | 18.000000 / 0.999158 / 0.002982 / 0.991886 / 0.993738 / 0.998815 / 1.002988 / 1.004838 |
| 10-15s | state[12] right_arm_joint1 | 18.000000 / 0.020591 / 0.006178 / 0.009856 / 0.011882 / 0.020136 / 0.028926 / 0.030474 |
| 10-15s | state[13] right_arm_joint2 | 18.000000 / -0.117556 / 0.100344 / -0.378181 / -0.329713 / -0.079919 / -0.029753 / -0.011099 |
| 10-15s | state[14] right_arm_joint3 | 18.000000 / 0.021068 / 0.010745 / -0.004515 / 0.008853 / 0.022788 / 0.034868 / 0.049736 |
| 10-15s | state[15] right_arm_joint4 | 18.000000 / 0.042148 / 0.059921 / -0.015830 / -0.012581 / 0.036171 / 0.099080 / 0.258337 |
| 10-15s | state[16] right_arm_joint5 | 18.000000 / -0.003305 / 0.053332 / -0.124574 / -0.100029 / 0.002897 / 0.061822 / 0.099412 |
| 10-15s | state[17] right_arm_joint6 | 18.000000 / -0.444489 / 0.356481 / -1.317378 / -1.253656 / -0.310142 / -0.128559 / -0.079847 |
| 10-15s | state[18] right_arm_eef_gripper_joint | 18.000000 / 0.998493 / 0.002764 / 0.995410 / 0.995928 / 0.997556 / 1.003505 / 1.007511 |
| 15-20s | state[0] base_v_feedback | 14.000000 / 0.010978 / 0.047932 / -0.098628 / -0.070354 / 0.014611 / 0.071681 / 0.101023 |
| 15-20s | state[1] base_w_feedback | 14.000000 / 1.174530 / 0.034716 / 1.104702 / 1.111808 / 1.181138 / 1.213902 / 1.218007 |
| 15-20s | state[2] slide_joint | 14.000000 / 0.455307 / 0.062562 / 0.276920 / 0.366818 / 0.465004 / 0.533520 / 0.540215 |
| 15-20s | state[3] head_yaw_joint | 14.000000 / 0.004920 / 0.002819 / 0.000958 / 0.001378 / 0.004406 / 0.009806 / 0.010741 |
| 15-20s | state[4] head_pitch_joint | 14.000000 / 0.002847 / 0.000458 / 0.001665 / 0.002199 / 0.002863 / 0.003466 / 0.003716 |
| 15-20s | state[5] left_arm_joint1 | 14.000000 / -0.010140 / 0.011319 / -0.030712 / -0.029470 / -0.008107 / 0.006617 / 0.007442 |
| 15-20s | state[6] left_arm_joint2 | 14.000000 / -0.037642 / 0.020076 / -0.077824 / -0.063985 / -0.034132 / -0.006750 / 0.002640 |
| 15-20s | state[7] left_arm_joint3 | 14.000000 / 0.043282 / 0.021847 / 0.011901 / 0.012840 / 0.043896 / 0.076061 / 0.078004 |
| 15-20s | state[8] left_arm_joint4 | 14.000000 / -0.131427 / 0.081605 / -0.290997 / -0.259801 / -0.135473 / -0.004637 / 0.002743 |
| 15-20s | state[9] left_arm_joint5 | 14.000000 / -0.548109 / 0.074673 / -0.685983 / -0.672342 / -0.539231 / -0.453761 / -0.422253 |
| 15-20s | state[10] left_arm_joint6 | 14.000000 / 1.398090 / 0.089417 / 1.245944 / 1.254220 / 1.385696 / 1.515444 / 1.532921 |
| 15-20s | state[11] left_arm_eef_gripper_joint | 14.000000 / 0.999567 / 0.002452 / 0.994752 / 0.996583 / 0.999025 / 1.003485 / 1.003763 |
| 15-20s | state[12] right_arm_joint1 | 14.000000 / 0.020853 / 0.003872 / 0.016033 / 0.016766 / 0.019053 / 0.027873 / 0.031042 |
| 15-20s | state[13] right_arm_joint2 | 14.000000 / -0.278832 / 0.084505 / -0.422116 / -0.391054 / -0.274358 / -0.145404 / -0.102531 |
| 15-20s | state[14] right_arm_joint3 | 14.000000 / 0.027629 / 0.020518 / 0.000371 / 0.001649 / 0.022792 / 0.069796 / 0.071615 |
| 15-20s | state[15] right_arm_joint4 | 14.000000 / 0.024693 / 0.063652 / -0.121694 / -0.100876 / 0.035807 / 0.103087 / 0.124360 |
| 15-20s | state[16] right_arm_joint5 | 14.000000 / -0.008851 / 0.065027 / -0.143976 / -0.118510 / -0.005853 / 0.086019 / 0.088641 |
| 15-20s | state[17] right_arm_joint6 | 14.000000 / -0.805540 / 0.466801 / -1.668740 / -1.437876 / -0.644189 / -0.272435 / -0.258885 |
| 15-20s | state[18] right_arm_eef_gripper_joint | 14.000000 / 0.999115 / 0.003976 / 0.991311 / 0.994354 / 0.998123 / 1.006811 / 1.008614 |

## Offline declaration

The analysis read the archive and rosbag2 SQLite directly. It did not start a ROS graph, Server, TeamClient, Adapter, or MuJoCo, and it did not modify production runtime.
