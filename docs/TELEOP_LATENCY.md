# Teleop latency map

Where the delay between moving a Quest controller and the arm moving actually
goes, what is measured vs. estimated, and which knobs buy the most back.

Measured on pi-v3 (Raspberry Pi 5) 2026-08-21, against the `dls_projector`
whole-body configuration.

## The chain

Quest tracker → Quest app → network → `OculusSource` receive thread (1€ filter)
→ teleop loop (`wholebody_teleop.py`, `LOOP_RATE`) → RPC → `WholeBodyController`
solve loop (`control_hz`) → IK solve → arm dispatch loop (`arm_dispatch_hz`,
interpolated) → nerolib (`controller_freq_hz`) → CAN → motor.

## Budget

Typical case is at 0.05–0.15 m/s controller speed, which is where the measured
median teleop motion sits (0.136 m/s in the 2026-08-21 session).

**These sessions run with `--no-pose-filter`**, so stage 2 contributes nothing.
The filter row is kept because it is the default path and the numbers matter if
it is ever turned back on.

| # | Stage | Typical | Worst | Basis |
|---|---|---|---|---|
| 1 | Quest tracker → teleop process | **unknown** | — | needs instrumentation, see below |
| 2 | 1€ `PoseFilter` | **0 ms (disabled)** | — | `--no-pose-filter`; 25–36 ms if enabled at defaults |
| 3 | Teleop loop sampling (`LOOP_RATE` 30 Hz) | 16.7 ms | 33.3 ms | period/2, period |
| 4 | RPC teleop → robot | **unknown** | — | needs instrumentation |
| 5 | Solve-loop wait (`control_hz` 30 Hz) | 16.7 ms | 33.3 ms | period/2, period |
| 6 | IK solve compute | 7.8 ms | 13.0 ms | benchmarked, `dls_projector` + collisions |
| 7 | Arm dispatch interpolation (3 steps @ 90 Hz) | 11.1 ms | 22.2 ms | 1/3 and 2/3 of a 33.3 ms segment |
| 8 | Ruckig preview (`arm_preview_time`) | 10.8 ms | 10.8 ms | config |
| 9 | nerolib control loop (250 Hz) | 4.0 ms | 4.0 ms | `controller_freq_hz` |
| 10 | CAN + motor mechanical response | **unknown** | — | needs instrumentation |

**Known software total: ≈ 67 ms typical, ≈ 117 ms worst**, before the three
unknown stages.

The two 30 Hz sampling waits (stages 3 and 5) are **half the known budget**
between them. They are the thing to attack.

### Notes on individual stages

**(2) The filter, if it is ever re-enabled.** It is speed-dependent by design,
so it hurts most exactly when moving slowly and precisely. Measured lag of the
real `PoseFilter` against a constant-velocity ramp at the Quest's 72 Hz:

| controller speed | `min_cutoff=3.0` (default) | `6.0` | `10.0` |
|---|---|---|---|
| 0.05 m/s | 35.9 ms | 22.6 ms | 14.7 ms |
| 0.15 m/s | 25.0 ms | 18.2 ms | 12.9 ms |
| 0.30 m/s | 18.5 ms | 14.6 ms | 11.1 ms |
| 1.00 m/s | 9.6 ms | 8.4 ms | 7.2 ms |

Running with no filter at all is not obviously the right end of that trade.
Raw tracker noise now reaches the solver directly, and this whole workstream
has been about solver-side jitter and branch-switching — some of which may be
input noise being faithfully tracked. A *high* `min_cutoff` (10–15 Hz) removes
tracker noise for ~5–7 ms, which is a far better trade than either extreme.
Worth one A/B once the loop-rate work below is done, not before — changing two
things at once has already cost this project several uninterpretable runs.

**(3) Sampling a 72 Hz stream at 30 Hz aliases.** With no filter in front of
it, tracker noise above 15 Hz folds down into the passband rather than being
rejected. Raising `LOOP_RATE` helps this as well as the latency.

**(7) Interpolation lag is structural, not tuning slop.** `_arm_dispatch_tick`
walks `alpha` through 1/3, 2/3, 1 over three 11.1 ms ticks, so in steady state
the commanded joint position trails the solved goal by an average of one third
of a 33.3 ms segment.

**(5)/(6) The solve loop has far more headroom than it used to**, because
`dls_projector` is about 3x cheaper than `soft`. Benchmarked full `solve()`,
collision avoidance on, both arms, `max_iters=10`:

| mode | median | p95 | fits 30 Hz? | 60 Hz? |
|---|---|---|---|---|
| `soft` | 25.3 ms | 29.0 ms | barely (87% of budget at p95) | no |
| `dls_projector`, no swivel | 7.8 ms | 10.8 ms | yes (32%) | **yes (54%)** |
| `dls_projector`, swivel 1.0 (current default) | 20.6 ms | 25.1 ms | yes (75%) | **no (150%)** |
| `dls_projector` + manipulability | 39.5 ms | 45.0 ms | **no** | no |

**Updated 2026-08-21, and this changes the 60 Hz recommendation below.** The
elbow-swivel null-space objective added afterwards costs ~12 ms per solve, so
`dls_projector` at its current default no longer fits a 60 Hz budget. Either
run 30 Hz with swivel on, or turn swivel off
(`--nullspace-swivel-weight 0`) to buy the headroom back -- but swivel is what
suppresses the elbow branch-flipping, so that is a real trade, not a free one.
Manipulability does not fit any production rate; treat it as a diagnostic.

## What is not measured, and how to measure it

Three stages have no number. Note that `ControllerState.created_timestamp` is
**not** a headset timestamp — `parse_controller_state` sets it with
`time.time()` on the receiving side, so it says nothing about transport.

- **(1) Quest → teleop.** Needs the Quest app to stamp send-time, plus clock
  sync (or a round-trip echo). Without an app change, the only honest proxy is
  a round-trip measurement.
- **(4) RPC hop.** Teleop and robot are separate machines, so a one-way delta
  needs synced clocks. Measure RTT from the teleop side and halve it as an
  estimate; that is cheap and needs no protocol change.
- **(10) Command → actual motion.** `use_measured_arm_state` is `False`, so
  encoders are not even read per tick — the IK model runs open-loop and the
  `*_actual_ee_*` columns in the trajectory CSVs are **forward kinematics of
  the solver's own model, not the real arm.** Anything derived from those
  columns measures the solver, not the hardware. To get real response: enable
  measured arm state temporarily and log commanded vs. measured joint angles
  with timestamps, then cross-correlate.

The cheapest useful end-to-end number, if a single figure is wanted: sharp
controller flick, record the Quest pose stream and the arm encoder stream, and
cross-correlate the two. That folds all ten stages into one honest number
without needing per-stage clock sync.

## Levers, ranked by ms recovered per unit of risk

With the filter already off, the remaining budget is dominated by the two
30 Hz sampling waits, then by the dispatch chain.

1. **`control_hz` 30 → 60** — saves ~8.3 ms. **Conditional**: affordable only
   with the elbow-swivel objective off (54% of a 60 Hz budget at p95); with
   swivel at its default weight the solve needs ~25 ms p95 and does not fit.
   Decide which matters more before spending this one.
   Also collapses the target-hold aliasing measured on 2026-08-21 (27.8% of
   moving ticks preceded by a stale-target hold), since the solver would then
   sample faster than teleop produces. Watch: `dt` in `ik_config` is
   `1.0 / control_hz`, so this changes the solver's integration step and its
   velocity-limit scaling — re-check tracking error and branch-switch rate
   after, do not assume it is free.
2. **Teleop `LOOP_RATE` 30 → 60** — saves ~8.3 ms, and halves the aliasing of
   the ~72 Hz Quest stream. Pairs with (1): done alone, the robot-side 30 Hz
   wait absorbs most of the gain.
3. **`arm_dispatch_hz` 90 → 120 (keeping 3 steps)** — saves ~2.8 ms of ramp
   and ~2.7 ms of preview, since `arm_preview_time` must shrink with the
   dispatch period (see its docstring: a preview longer than one tick turns
   every update into stop-and-go and caps speed). Alternatively
   `arm_interpolation_steps` 3 → 2 for ~5.5 ms, at the cost of coarser
   smoothing — that smoothing is doing real work, so prefer the rate change.
4. **Re-enable the filter at a high `min_cutoff`** — *adds* ~5–7 ms but may
   buy back jitter that is currently reaching the solver raw. Only worth
   evaluating after 1–3, and only as a deliberate A/B.

Items 1–3 take the known software total from ≈67 ms to ≈45 ms typical.
Change them **one at a time** and re-measure — several interact (loop rates
with `dt`, dispatch rate with preview time), and
`artifacts/wholebody_logs/posture_fix_commands.md` has repeated examples of
two simultaneous changes making a result uninterpretable.
