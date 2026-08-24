# Base Swerve PID Gains

Two sets, both on PID slot 0 on all four swerve modules. Copied from
`YOR-v3-Problems-DON'T-USE/config/base_pid_manifest.json` on 2026-08-18
(commissioned) and 2026-08-22 (stock).

## Stock — the default

`config/base_pid_stock.json`. What the controllers hold in flash and revert to
on a power cycle, and what `robot/yor.py` applies unless told otherwise.

| Motor role | Kp | Ki | Kd | Velocity FF | Output range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drive | 0.2 | 0.0 | 0.1 | 0.0 | -1.0 to 1.0 |
| Steering | 2.0 | 0.0 | 0.01 | 0.0 | -1.0 to 1.0 |

These are **not** the REV factory zeros. Writing zeros would not restore
anything; it would leave the base limp until the next power cycle.

## Commissioned — opt-in

`config/base_pid_commissioned.json`, measured on the floor 2026-08-17. Apply
with `--base-pid-manifest config/base_pid_commissioned.json`.

| Motor role | Kp | Ki | Kd | Velocity FF | Output range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drive | 0.35 | 0.0 | 0.0 | 0.23 | -1.0 to 1.0 |
| Steering | 20.0 | 0.0 | 6.0 | 0.0 | -0.25 to 0.25 |

## Why stock is the default

The stock drive loop is P-only with no feed-forward, so it reaches only about
49% of its setpoint — which is exactly what `DRIVE_VEL_SCALE = 2.0` in
`robot/base_motor.py` was introduced to compensate for. Running stock keeps the
speed axis self-consistent: commanded m/s is roughly true m/s and
`BaseOdometry` is honest.

The commissioned drive loop tracks instead (steady error 0.000 native median),
which turns that same 2.0 into a stale doubling: the base runs at about twice
every commanded speed and the odometry records half of reality. Until
`tools/measure_drive_scale.py` settles that, the tuned set stays opt-in. See
`docs/BASE_COMMAND_LOOP_REVIEW.md` finding 6.

Second reason, independent of the scale: every floor run behind the
commissioned numbers clamped controller output to ±0.25, while the manifest
ships the drive range at ±1.0 — so P is effectively 7× and D 5× what they were
measured under.

**Trade-off to expect when running stock.** Steering Kp is 2.0 against the
commissioned 20.0, so module angle tracking is far softer and steady-state
angle error will be visibly larger. The `steer_cmd_*` / `steer_meas_*` columns
in the trajectory log measure exactly that, which makes a stock run a useful
commissioning baseline rather than just a compromise.

## CAN IDs

| Module | Drive | Steering |
| --- | ---: | ---: |
| Front left (FL) | 1 | 5 |
| Front right (FR) | 4 | 8 |
| Rear right (RR) | 3 | 7 |
| Rear left (RL) | 2 | 6 |

## Operational notes

The first three concern the **commissioned** set specifically.

- Drive `Kd` is deliberately zero. A tested value of 10 caused an audibly
  harsh 100 Hz torque ripple and rough deceleration during floor testing.
- The drive loop uses `velocity_ff = 0.23` and should be treated as
  feed-forward-dominated rather than as a conventional PD loop.
- The steering output is intentionally limited to +/-0.25.
- These settings live in controller RAM. A SPARK controller power cycle
  restores its stock settings, so the gains are reapplied at startup:
  `robot/yor.py` `init()` syncs every controller against
  the selected manifest before the base control loop starts, and leaves alone
  any controller that already holds it. `robot/yor.py` also writes the stock
  set back on shutdown, so a tuned set never outlives the process that asked
  for it.
- This file is the commissioning record. The values the robot actually applies
  live in `config/base_pid_stock.json` and
  `config/base_pid_commissioned.json`, and the code that applies them is
  `tools/base_pid_preflight.py` — which is also the standalone command for
  checking the controllers while the robot node is not running.
