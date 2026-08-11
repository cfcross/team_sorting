# M10-B1-A.3-A Runtime Early-Phase Trajectory Audit

## Scope and time origin

This report contains runtime facts only. It does not diagnose checkpoint quality or assign causality.

`t=0` is candidate request `3` at SQLite bag timestamp `1786256185709649153` ns. The first policy dispatch follows by 13.883917 ms.

actual TeamClient callback receipt time is absent; the candidate SQLite bag timestamp is used as its observable proxy, so absolute receipt-time error is not bounded by this artifact. 24 Hz row selection plus shared two-step handoff alpha inferred from unclipped dimensions; unclipped FinalAction components are exact pre-clamp values.

Handoff fit ticks: 1001; maximum per-tick stable alpha-ratio P95-P05 2.53131e-14; max unclipped fit residual 3.33067e-16.

Nominal action-row offsets selected by the fit: {'-1': 9, '0': 2486, '1': 1}; E1 base clip masks match every reconstructed tick: True.

clipped raw components are scheduler reconstructions, not independently logged pre-clamp values.

First-20-second policy timing: 774 ticks; 1 gap(s) over 37.5 ms; maximum gap 0.674862 s.

## Four five-second phases

Each cell is `mean / P05 / P50 / P95 / min / max`.

| Phase | Stream | Dimension | Statistics |
|---|---|---|---|
| 0-5s | raw | base_v | 0.089888 / -0.035057 / 0.063655 / 0.303379 / -0.042617 / 0.371207 |
| 0-5s | raw | base_w | 0.174253 / -0.046923 / 0.145425 / 0.552839 / -0.104022 / 0.594479 |
| 0-5s | raw | slide | 0.020949 / -0.026432 / 0.022949 / 0.072987 / -0.045715 / 0.089340 |
| 0-5s | final_action | base_v | 0.089888 / -0.035057 / 0.063655 / 0.303379 / -0.042617 / 0.371207 |
| 0-5s | final_action | base_w | 0.174253 / -0.046923 / 0.145425 / 0.552839 / -0.104022 / 0.594479 |
| 0-5s | final_action | slide | 0.021060 / -0.026432 / 0.022949 / 0.072987 / -0.040000 / 0.089340 |
| 0-5s | odom | linear_x | -0.027583 / -0.105437 / -0.003583 / 0.000869 / -0.128527 / 0.002882 |
| 0-5s | odom | angular_z | 0.157395 / -0.046834 / 0.143063 / 0.394671 / -0.082899 / 0.493188 |
| 0-5s | slide_state | slide | 0.026521 / -0.023278 / 0.025922 / 0.078386 / -0.032033 / 0.087434 |
| 5-10s | raw | base_v | 0.105087 / -0.113092 / 0.110146 / 0.327769 / -0.196854 / 0.409904 |
| 5-10s | raw | base_w | 0.436755 / 0.245987 / 0.346686 / 0.872816 / 0.224331 / 1.038242 |
| 5-10s | raw | slide | 0.045973 / 0.006683 / 0.048208 / 0.091565 / -0.006137 / 0.098362 |
| 5-10s | final_action | base_v | 0.105087 / -0.113092 / 0.110146 / 0.327769 / -0.196854 / 0.409904 |
| 5-10s | final_action | base_w | 0.436755 / 0.245987 / 0.346686 / 0.872816 / 0.224331 / 1.038242 |
| 5-10s | final_action | slide | 0.045973 / 0.006683 / 0.048208 / 0.091565 / -0.006137 / 0.098362 |
| 5-10s | odom | linear_x | -0.081073 / -0.161251 / -0.087099 / 0.008077 / -0.191640 / 0.103775 |
| 5-10s | odom | angular_z | 0.437327 / 0.286265 / 0.366540 / 0.753874 / 0.250065 / 0.943262 |
| 5-10s | slide_state | slide | 0.050355 / 0.014883 / 0.045562 / 0.085724 / 0.009330 / 0.097650 |
| 10-15s | raw | base_v | 0.027971 / -0.066764 / 0.000724 / 0.216780 / -0.086139 / 0.287020 |
| 10-15s | raw | base_w | 1.176514 / 0.560653 / 1.249872 / 1.537750 / 0.474749 / 1.573606 |
| 10-15s | raw | slide | 0.148686 / 0.047426 / 0.125965 / 0.276298 / 0.042900 / 0.402111 |
| 10-15s | final_action | base_v | 0.027971 / -0.066764 / 0.000724 / 0.216780 / -0.086139 / 0.287020 |
| 10-15s | final_action | base_w | 1.067338 / 0.560653 / 1.200000 / 1.200000 / 0.474749 / 1.200000 |
| 10-15s | final_action | slide | 0.148686 / 0.047426 / 0.125965 / 0.276298 / 0.042900 / 0.402111 |
| 10-15s | odom | linear_x | -0.017262 / -0.073463 / -0.013652 / 0.015735 / -0.104717 / 0.025568 |
| 10-15s | odom | angular_z | 1.047873 / 0.667495 / 1.128900 / 1.224047 / 0.518844 / 1.242301 |
| 10-15s | slide_state | slide | 0.149014 / 0.064214 / 0.129157 / 0.277613 / 0.062430 / 0.405063 |
| 15-20s | raw | base_v | 0.058634 / -0.039408 / 0.055726 / 0.192278 / -0.053475 / 0.216699 |
| 15-20s | raw | base_w | 1.337831 / 1.174824 / 1.324225 / 1.464042 / 1.086416 / 1.471796 |
| 15-20s | raw | slide | 0.468727 / 0.408502 / 0.470213 / 0.527452 / 0.405263 / 0.547784 |
| 15-20s | final_action | base_v | 0.058634 / -0.039408 / 0.055726 / 0.192278 / -0.053475 / 0.216699 |
| 15-20s | final_action | base_w | 1.195861 / 1.174824 / 1.200000 / 1.200000 / 1.086416 / 1.200000 |
| 15-20s | final_action | slide | 0.468727 / 0.408502 / 0.470213 / 0.527452 / 0.405263 / 0.547784 |
| 15-20s | odom | linear_x | 0.016189 / -0.037235 / 0.015557 / 0.067851 / -0.060566 / 0.101023 |
| 15-20s | odom | angular_z | 1.179637 / 1.116766 / 1.182569 / 1.216404 / 1.019193 / 1.240083 |
| 15-20s | slide_state | slide | 0.462174 / 0.414859 / 0.466151 / 0.530175 / 0.414181 / 0.545436 |

## Sustained positive rotation

Durations use adjacent 40 Hz policy ticks with gaps no larger than 37.5 ms and include one 25 ms tick.

| Source | Threshold | First time (s) | Longest run (s) | Share in 10–20 s |
|---|---:|---:|---:|---:|
| raw | 0.8 | 8.963976 | 3.975494 | 92.245989% |
| raw | 1.0 | 9.563960 | 3.975494 | 86.631016% |
| raw | 1.19 | 10.613807 | 3.300354 | 72.727273% |
| final | 0.8 | 8.963976 | 3.975494 | 92.245989% |
| final | 1.0 | 9.563960 | 3.975494 | 86.631016% |
| final | 1.19 | 10.613807 | 3.300354 | 72.727273% |

## Spin onset

`SPIN_ONSET_S = 12.713842`

The first qualifying run continuously has `FinalAction base_w >= 1.0` and `abs(FinalAction base_v) <= 0.08` for at least one second.

Each cell below is `mean / P05 / P50 / P95 / min / max`.

| Window | Stream | Dimension | Statistics |
|---|---|---|---|
| spin_onset_minus2_0 | raw | base_v | -0.017308 / -0.073228 / -0.007205 / 0.043070 / -0.086139 / 0.049991 |
| spin_onset_minus2_0 | raw | base_w | 1.051674 / 0.481800 / 1.100981 / 1.458662 / 0.474749 / 1.474229 |
| spin_onset_minus2_0 | final_action | base_v | -0.017308 / -0.073228 / -0.007205 / 0.043070 / -0.086139 / 0.049991 |
| spin_onset_minus2_0 | final_action | base_w | 0.993784 / 0.481800 / 1.100981 / 1.200000 / 0.474749 / 1.200000 |
| spin_onset_minus2_0 | odom | linear_x | -0.005165 / -0.028830 / -0.002777 / 0.021136 / -0.033021 / 0.025568 |
| spin_onset_minus2_0 | odom | angular_z | 0.978629 / 0.642610 / 1.045580 / 1.153027 / 0.584262 / 1.169385 |
| spin_onset_0_plus2 | raw | base_v | 0.054330 / -0.038853 / 0.007536 / 0.216439 / -0.058946 / 0.222555 |
| spin_onset_0_plus2 | raw | base_w | 1.360607 / 1.156928 / 1.400333 / 1.551142 / 1.061703 / 1.573606 |
| spin_onset_0_plus2 | final_action | base_v | 0.054330 / -0.038853 / 0.007536 / 0.216439 / -0.058946 / 0.222555 |
| spin_onset_0_plus2 | final_action | base_w | 1.191385 / 1.156928 / 1.200000 / 1.200000 / 1.061703 / 1.200000 |
| spin_onset_0_plus2 | odom | linear_x | -0.018233 / -0.057172 / -0.016527 / 0.012638 / -0.084448 / 0.016992 |
| spin_onset_0_plus2 | odom | angular_z | 1.171447 / 1.101978 / 1.170589 / 1.227236 / 0.997329 / 1.242301 |

## Raw slide below zero in first 20 seconds

Count: 38; first: 0.838779 s; last: 6.089001 s; minimum: -0.045715 m.

This is an independent observation, not a claimed primary cause.

## Offline declaration

The analyzer read rosbag2 SQLite directly in read-only mode. No ROS graph, Server, TeamClient, Adapter or MuJoCo process was started, and production runtime was not modified.
