# Base Swerve PID Gains

These gains were copied from the commissioned PID manifest in
`YOR-v3-Problems-DON'T-USE/config/base_pid_manifest.json` on 2026-08-18.
They apply to PID slot 0 on all four swerve modules.

| Motor role | Kp | Ki | Kd | Velocity FF | Output range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drive | 0.35 | 0.0 | 0.0 | 0.23 | -1.0 to 1.0 |
| Steering | 20.0 | 0.0 | 6.0 | 0.0 | -0.25 to 0.25 |

## CAN IDs

| Module | Drive | Steering |
| --- | ---: | ---: |
| Front left (FL) | 1 | 5 |
| Front right (FR) | 4 | 8 |
| Rear right (RR) | 3 | 7 |
| Rear left (RL) | 2 | 6 |

## Operational notes

- Drive `Kd` is deliberately zero. A tested value of 10 caused an audibly
  harsh 100 Hz torque ripple and rough deceleration during floor testing.
- The drive loop uses `velocity_ff = 0.23` and should be treated as
  feed-forward-dominated rather than as a conventional PD loop.
- The steering output is intentionally limited to +/-0.25.
- These settings live in controller RAM. A SPARK controller power cycle
  restores its stock settings, so the gains are reapplied at startup:
  `robot/yor.py` `init()` syncs every controller against
  `config/base_pid_manifest.json` before the base control loop starts, and
  leaves alone any controller that already holds them.
- This file is the commissioning record. The values the robot actually applies
  live in `config/base_pid_manifest.json`, and the code that applies them is
  `tools/base_pid_preflight.py` — which is also the standalone command for
  checking the controllers while the robot node is not running.
