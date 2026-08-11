# M10-B0.5-A offline recommendation

Authoritative machine-readable evidence:
`m10_b05_scheduler_20260808T134514Z.json`.

This is an offline recommendation only. No production scheduler, safety gate,
publisher, controller manifest, or runtime configuration was changed.

## A. B0 reproduction

The independent SQLite/CDR/JSON reconstruction matches the supplied baseline:

- duration 239.8568 s;
- 647 candidates, contiguous request IDs 4 through 650;
- one stable generation and one stable run/task-1/attempt-0 identity;
- every candidate valid, finite, and exactly 15x19, with
  `published_to_robot=false`;
- 3912 policy-sourced FinalAction ticks;
- same-chunk transitions: +1 = 2638, +2 = 627;
- chunk index 0 executed = 540, first index greater than 0 = 107;
- within-chunk 19D max-delta P50/P95/MAX =
  0.01014/0.01891/0.16517;
- immediate handoff max-delta P50/P95/MAX =
  0.12509/0.23707/0.54450.

## B/C. Control-rate recommendation

| Hz | interval ms | ideal skip % | index0 % | duplicate % | hold % | skip-transition probability ±1/±2/±5/±10 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 50.000 | 15.6930 | 83.3076 | 0.0000 | 0.5846 | 19.0581 / 18.9821 / 19.0017 / 20.0843 |
| 24 | 41.667 | 0.0213 | 100.0000 | 0.0213 | 0.5719 | 0.6004 / 1.3232 / 3.5448 / 6.7784 |
| 30 | 33.333 | 0.0000 | 100.0000 | 19.6453 | 0.6270 | 0 / 0 / 0.0192 / 1.3596 |
| 40 | 25.000 | 0.0000 | 100.0000 | 38.2176 | 0.5974 | 0 / 0 / 0 / 0.0223 |
| 50 | 20.000 | 0.0000 | 100.0000 | 50.3119 | 0.5795 | 0 / 0 / 0 / 0 |
| 60 | 16.667 | 0.0000 | 100.0000 | 58.3447 | 0.6015 | 0 / 0 / 0 / 0 |

The lowest reasonable B1 control rate is **40 Hz**. 30 Hz is adequate only if
timer jitter is reliably below about 5 ms. 40 Hz eliminates simulated skips
through ±5 ms and leaves only 0.0223% skip transitions at ±10 ms. 50 Hz is not
materially better for the measured/required jitter range: it removes that last
0.0223%, but raises duplicate-index ticks from 38.2% to 50.3%. 60 Hz adds still
more duplicates without a demonstrated benefit.

The policy timeline remains exactly 24 Hz; the recommendation changes only the
consumer sampling/control tick.

## D/E. Handoff recommendation

The complete 30-row rate-by-handoff matrix is in
`m10_b05_scheduler_20260808T134514Z.md`. At the recommended 40 Hz:

| handoff | skip % | index0 % | duplicate % | hold % | boundary P50 | P95 | MAX | within P95 | added latency ms | risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| immediate | 0 | 100 | 38.2176 | 0.5974 | 0.12646 | 0.23428 | 0.54450 | 0.01891 | 0 | high discontinuity |
| next-step-boundary | 0 | 100 | 39.9949 | 0.5974 | 0.12575 | 0.23416 | 0.54450 | 0.01891 | 20.83 mean | delay without meaningful delta reduction |
| crossfade-1 | 0 | 100 | 38.2176 | 0.5974 | 0.03778 | 0.10312 | 0.25408 | 0.01891 | 41.67 | low/moderate interpolation |
| crossfade-2 | 0 | 100 | 38.2176 | 0.5974 | 0.01889 | 0.05558 | 0.20775 | 0.01891 | 83.33 | moderate interpolation |
| crossfade-3 | 0 | 100 | 38.2176 | 0.5974 | 0.01259 | 0.03706 | 0.20775 | 0.01891 | 125.00 | higher semantic distortion |

Immediate replacement should be changed because its P95 boundary discontinuity
is about 12.4 times the within-chunk P95. Next-boundary handoff alone does not
solve it. The smallest well-separated choice is a **2-policy-step crossfade for
continuous dimensions**, starting from the last real output and retaining the
new chunk's time index. It cuts 40 Hz P50/P95 to 0.01889/0.05558 without selecting
a future action. Three steps gives diminishing benefit for another 41.67 ms of
trajectory distortion.

No blend is allowed across run/task/attempt/instruction/generation changes, or
after invalid/stale/watchdog/shutdown safety transitions. Those transitions
must remain immediate fail-closed.

## F. Grippers

Do **not** smooth grippers. Actual handoff deltas are numerical noise only:
left-grip MAX 8.54e-8 and right-grip MAX 1.15e-7. Their ActionMux clips are also
only floating tails (left 11 ticks, max 9.36e-9; right 19 ticks, max 3.40e-8).
Interpolation adds discrete-command ambiguity with no measured continuity gain.

## G/H. Latency, candidate TTL, and watchdog

| latency gate ms | accepted % | policy coverage % | safe-hold % | longest hold s |
|---:|---:|---:|---:|---:|
| 250 | 69.3972 | 80.8846 | 19.1154 | 3.6494 |
| 300 | 86.3988 | 89.7560 | 10.2440 | 1.8498 |
| 400 | 93.9722 | 93.6960 | 6.3040 | 1.5501 |
| 500 | 97.3725 | 96.3650 | 3.6350 | 1.2499 |
| 625 | 99.0726 | 98.2461 | 1.7539 | 0.6500 |
| 700 | 99.3818 | 98.6782 | 1.3218 | 0.6001 |
| 1000 | 100.0000 | 99.4408 | 0.5592 | 0.1498 |

Recommend **max_policy_response_latency_ms = 625** for B1. This is not chosen
only because the chunk is 625 ms: measured P99 is 607.5 ms, 625 ms accepts
99.07% and gives 98.25% policy coverage, while 700 ms gains only 0.31 percentage
points of candidate acceptance and admits responses older than the entire
policy horizon. 500 ms creates a materially larger 3.64% hold fraction and
1.25 s longest hold.

Recommend **candidate_ttl_ms = 625** and **watchdog_timeout_ms = 625** for the
first B1 patch. They cannot extend the 15-step horizon. The current consumer
checks candidate TTL before watchdog at equality, so a later patch should not
claim distinct watchdog telemetry without explicitly testing that ordering.

## I/J. Manifest and minimum production patch

Changing the Team control rate from 20 to 40 Hz requires a coordinated update
to `controller_manifest.py` and `config/config.yaml`, because the manifest
freezes and validates nominal control frequency. Handoff and latency/TTL policy
do not themselves require a Controller Manifest schema change.

Minimum expected B0.5-B production files:

- `team_sorting/pi05_policy_control.py` (continuous-only two-step handoff and
  immediate safety invalidation rules);
- `config/config.yaml` (40 Hz and explicit B1 timing values);
- `team_sorting/controller_manifest.py` (40 Hz frozen nominal rate);
- `tests/test_pi05_policy_control.py` and relevant control-gate tests;
- `team_sorting/ros_nodes.py` only if timer/config wiring cannot remain unchanged.

No `interfaces.py`, ActionMux limit, policy schema, action dimension, horizon,
or official publisher change is indicated by this analysis.

## ActionMux clipping

52/3912 policy ticks were clipped (1.3292%): slide 22 physical-limit clips with
maximum exceedance 0.008315 m; left grip 11 and right grip 19 were pure floating
tails. ActionMux limits should remain unchanged.

## K. Readiness

The alternatives are distinguishable and the safety invariants have synthetic
coverage. This is sufficient to enter a separate, reviewable offline production
hardening patch, but is not authorization to enable publishing or motion.

`READY_FOR_B05B = YES`
