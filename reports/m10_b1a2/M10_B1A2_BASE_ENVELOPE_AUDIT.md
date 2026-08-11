# M10-B1-A.2 Base Action Envelope Offline Verification

## Decision

`OFFICIAL_TASK1_BASELINE_ENVELOPE = {base_v: 0.45 m/s, base_w: 1.20 rad/s}`

`TEAM_CURRENT_BASE_ENVELOPE = {base_v: 0.25 m/s, base_w: 0.50 rad/s}`

`E0_RESULT = base_v 78/2147 clipped; base_w 720/2147 clipped; 798/2147 ticks changed`

`E2_RESULT = base_v 9/2147 clipped; base_w 220/2147 clipped; 229/2147 ticks changed`

`E1_RESULT = base_v 0/2147 clipped; base_w 0/2147 clipped; 0/2147 ticks changed`

`BASE_ENVELOPE_MISMATCH = PROVEN`

`READY_FOR_B1A2_SIM = YES`

The direct answer to the primary question is: replacing only the current Team base envelope with the actual official Task1 baseline envelope reduces the observed 720 `base_w` clipping ticks to **0**. Both `base_v` and `base_w` clipping fall by 100%. The largest remaining E1 exceedance is 0 m/s and 0 rad/s.

`READY_FOR_B1A2_SIM = YES` means only that the offline evidence justifies one temporary simulation-only A/B. It does not prove that E1 is a physical safety limit, that E1 is safer, that Task1 performance will improve, or that production/real-robot limits may be changed.

## Evidence and source trace

The archive was independently hashed before extraction:

- archive: `/home/ljc/Projects/m10_b1a_runs/m10_b1a_20260808_225300.tar.gz`
- SHA256: `f7c49502bb0338bccb8dfa9c267c833d9f6004590845aeeb2010c995f0d24df2` (matches the expected value)
- processing: read-only; extraction was into a new `/tmp/m10_b1a2.*` directory; SQLite was opened with `mode=ro`
- recovered evidence: 190 candidate chunks, 2,850 raw candidate rows, and 2,147 policy FinalAction/control ticks

The current official source is `material_sorting_official_offline/examples/material_sorting/client_task_1.py`, SHA256 `b93e9d730fbea443a061a883062b8dfaf12e40e83f8313174ed89f6562a3d2cd`. At line 134 it assigns `self.max_lin, self.max_ang = 0.45, 1.2`. `set_twist()` applies those values through `np.clip`; `ramp()` places the result in `tc[0:2]`; `smooth_and_publish()` assigns `tc[0]` to `Twist.linear.x`, `tc[1]` to `Twist.angular.z`, and publishes that message on `/cmd_vel`. Therefore 0.45/1.20 is proven as `OFFICIAL_TASK1_BASELINE_ENVELOPE`. No independent official physical safety-limit contract was found or claimed.

The Team source chain is:

1. `config/config.yaml` supplies `action_mux.max_abs_base_v=0.25` and `max_abs_base_w=0.50`, explicitly described there as Team initial safety limits rather than official performance values.
2. `_load_config()` selects the first existing complete config in this order: `TEAM_SORTING_CONFIG`, installed package share, repository fallback. Thus a runtime override exists only by selecting another complete config with `TEAM_SORTING_CONFIG`; these two values are not independently declared ROS parameters.
3. Every selected config passes `validate_controller_config()`, which requires its ActionMux bounds to exactly match Controller Manifest V1. The manifest labels both base ranges `partially_verified` and says they remain Team conservative values.
4. `TeamClientNode` calls `_action_mux_config(config)` and explicitly constructs `ActionMux` from the selected values. Direct `ActionMux()` construction instead uses the same manifest values as conservative defaults.
5. `ActionMux.compose_with_decision()` clamps finite candidate `v/w` to the configured symmetric ranges and records the per-dimension `clipped_mask`.

The current Team values are consequently an ActionMux safety envelope, not an official MMK2 physical safety limit.

## Candidate-to-control-tick reconstruction

The counterfactual uses the pre-clamp policy trajectory, never post-clamp FinalAction as a substitute for clipped raw candidates. The archived candidate does not contain TeamClient's callback receipt ROS timestamp, so the mapping is not an independently recorded 100% timestamp join.

Reconstruction used the production 24 Hz row schedule and two-policy-step handoff contract:

- candidate bag timestamp selects the nominal chunk row for each policy dispatch;
- for an unclipped component, FinalAction is exactly its pre-clamp value and replaces the nominal reconstruction;
- during a handoff, the single shared interpolation `alpha` is inferred from all eligible unclipped, non-gripper components, then applied to clipped components;
- clipped components therefore remain scheduler reconstructions rather than independently recorded pre-clamp samples.

There were 622 handoff-fit ticks. The maximum spread among independently inferred alpha ratios was `1.7351e-12`, and the maximum absolute fit residual on eligible unclipped components was `2.22045e-16`. Most importantly, reconstructed E0 base clipping agrees with the bag-native `clipped_mask` on every one of the 2,147 policy ticks: exactly 78 `base_v` and 720 `base_w` clips. This bounds observed numerical fit error, but it does not turn clipped raw values into a separately logged ground truth.

Percentiles use linear interpolation. “Exceedance” is `max(abs(raw)-limit, 0)` evaluated only on clipped ticks. “Absolute distortion” is `abs(raw-clipped)` evaluated across all 2,147 policy ticks, including zeros.

## Per-dimension counterfactual

### base_v

| Envelope | Relevant | Clip ticks | Clip % | Raw min / max (m/s) | Exceedance P50 / P95 / P99 / MAX (m/s) | Distortion mean / P50 / P95 / MAX (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| E0 0.25 | 2,147 | 78 | 3.6330% | -0.205743 / 0.392033 | 0.018782 / 0.139883 / 0.140947 / 0.142033 | 0.001572 / 0 / 0 / 0.142033 |
| E2 0.35 | 2,147 | 9 | 0.4192% | -0.205743 / 0.392033 | 0.039883 / 0.041469 / 0.041920 / 0.042033 | 0.000135 / 0 / 0 / 0.042033 |
| E1 0.45 | 2,147 | 0 | 0% | -0.205743 / 0.392033 | n/a / n/a / n/a / n/a | 0 / 0 / 0 / 0 |

### base_w

| Envelope | Relevant | Clip ticks | Clip % | Raw min / max (rad/s) | Exceedance P50 / P95 / P99 / MAX (rad/s) | Distortion mean / P50 / P95 / MAX (rad/s) |
|---|---:|---:|---:|---:|---:|---:|
| E0 0.50 | 2,147 | 720 | 33.5352% | -0.326496 / 1.037673 | 0.233242 / 0.460112 / 0.502825 / 0.537673 | 0.081133 / 0 / 0.407245 / 0.537673 |
| E2 0.80 | 2,147 | 220 | 10.2469% | -0.326496 / 1.037673 | 0.106456 / 0.196766 / 0.212056 / 0.237673 | 0.010898 / 0 / 0.107245 / 0.237673 |
| E1 1.20 | 2,147 | 0 | 0% | -0.326496 / 1.037673 | n/a / n/a / n/a / n/a | 0 / 0 / 0 / 0 |

## Combined distortion

“Mean absolute base command distortion” is the mean across both base components and all ticks. Per-tick L1 is `|dv|+|dw|`; per-tick L2 is `sqrt(dv²+dw²)`. Totals sum those per-tick values. No observed tick clipped both base components, so L1 and L2 are numerically equal in this run.

| Envelope | Any changed ticks / % | Both changed | Mean abs/component | Mean L1 / total L1 | Mean L2 / total L2 |
|---|---:|---:|---:|---:|---:|
| E0 | 798 / 37.1681% | 0 | 0.041352 | 0.082705 / 177.567430 | 0.082705 / 177.567430 |
| E2 | 229 / 10.6660% | 0 | 0.005517 | 0.011034 / 23.688973 | 0.011034 / 23.688973 |
| E1 | 0 / 0% | 0 | 0 | 0 / 0 | 0 / 0 |

Relative to E0, E1 reduces `base_v` clipping from 78 to 0 (100%) and `base_w` clipping from 720 to 0 (100%). All 798 E0 base-affected ticks lie inside the observed official Task1 baseline envelope.

## Training-demonstration comparison

`TRAINING_SIDE_REPORTED`: Task1 demonstration `base_v` is frequently around -0.35/+0.45 m/s, while `base_w` P95/P99/MAX approach +1.2 rad/s. No training-side complete distribution artifact is present on OMEN, so this audit does not invent exact training quantiles.

The Team E0 values 0.25/0.50 are substantially below both that reported demonstration range and the verified official Task1 baseline envelope. The runtime result independently shows the practical effect: E0 changes at least one base component on 37.17% of policy ticks, dominated by 720 `base_w` clips, whereas E1 changes none. Therefore E0 significantly truncates actions that are both common in the reported training demonstration distribution and within the official baseline's actual operating envelope. “Within the baseline envelope” is not a claim of an independent physical safety certification.

## Independent slide-negative observation

The slide limit was not changed or included in the base-envelope decision. Across all 2,850 raw candidate rows, 487 have `slide < 0` (17.0877%). Using each candidate bag timestamp plus its nominal 24 Hz row offset:

- first occurrence relative to the first candidate: 0.285453 s
- median occurrence: 5.593246 s
- last occurrence: 12.949679 s
- minimum: -0.0780658 m
- negative-row P01: -0.0682627 m
- negative-row P05: -0.0601332 m

The candidate-publication span is 54.225 s (54.809 s including the final nominal row projection), so the classification is `OBSERVED_TIMING = EARLY_PRESENT`. This is timing correlation only and makes no causal claim.

## PROVEN versus HYPOTHESIS

PROVEN from current source and archived evidence:

- official Task1 baseline envelope is 0.45 m/s and 1.20 rad/s and reaches `/cmd_vel`;
- Team current ActionMux envelope is 0.25 m/s and 0.50 rad/s and is a Team conservative safety envelope;
- B1-A has 190 candidate chunks, 2,850 raw rows, and 2,147 policy ticks;
- bag-native E0 clipping is 78 `base_v` and 720 `base_w` ticks;
- the E0/E2/E1 counterfactual counts and distortion above;
- under E1, the 720 `base_w` clips become 0;
- slide-negative raw-row distribution and `EARLY_PRESENT` observed timing.

HYPOTHESIS requiring simulation:

- reducing base clipping may reduce closed-loop distribution shift and may improve behavior.

It is not proven that a wider envelope makes the robot perform better or is safe outside the same official simulation setting.

## Next-stage boundary

The only recommended next action is one 5–10 second MuJoCo simulation-only A/B using a temporary config, the same checkpoint, the same scheduler, and all other ActionMux limits unchanged. Change only `base_v/base_w` from E0 to E1, retain the rosbag, and restore the default safety config afterward. Do not apply E1 to production or a real robot based on this audit.

## Offline execution declaration

This audit did not start the official Server, OpenPI, HTTP bridge, Adapter, TeamClient, ROS control, rosbag playback into a ROS graph, or MuJoCo robot execution. It did not modify production runtime, production config, Adapter runtime, OpenPI config, dataset, checkpoint, the archive, or any previously existing uncommitted work.
