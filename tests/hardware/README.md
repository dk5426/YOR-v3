# Hardware test suite

These tests drive the **real robot**. Arms swing, the lift carries them up and
down, and in the last test the chassis rolls with nobody commanding it.

Read this whole page before running anything. It takes five minutes and the
robot weighs rather more than you do.

> The headless tests in `tests/` (`test_wholebody_control.py`,
> `test_api_parity.py`, `test_sim_node.py`) need no hardware and should pass
> before you come here. If the solver is broken in simulation, it is broken on
> the robot too — find that out at your desk.

---

## Safety rules

**1. Somebody holds the e-stop.** Not "the e-stop is on the bench". A person
whose only job is the e-stop, watching the robot, within reach of it. This is
non-negotiable for anything past stage 0.

**2. Ctrl-C is the software stop.** Every test installs a handler that halts the
base and lift — and e-stops the arms where relevant — before the process dies.
It is reliable, but it goes over the network. **The physical e-stop is faster
and does not depend on the Pi being alive.**

**3. The prompts are the interlock.** Motion tests will not move until you type
`GO`, `READY` or `ROLL` exactly. Enter, `y`, or a stray keypress all abort. If a
prompt asks you to confirm the workspace is clear, go and look — do not answer
from memory. `--yes` skips *informational* pauses only; it can never skip a
safety confirmation.

**4. Run them in order.** Each stage assumes the one before it passed. The base
axis signs from `test_03` are what make `test_07` drive toward its target rather
than away from it. Do not skip ahead to the interesting one.

**5. Blocks before floor.** `test_03` runs on blocks by default. Only pass
`--floor` once the wheels have turned the way you expect with the robot up in
the air.

**6. One owner of the hardware.** These tests go through `robot/yor.py` over
RPC. Do not run `joystick.py`, a teleop client, or a second test at the same
time — the RPC server is a single REP socket and they will serialise behind each
other, which at best makes timing tests lie and at worst delays a stop command.

**7. If something looks wrong, stop.** A test that fails is information. A test
you push through after ignoring a failure is how equipment gets broken.

---

## Before you start

On the robot:

```bash
python robot/yor.py
```

Leave that console visible — solver errors and lift firmware lines print there,
and they are usually the real explanation when a test fails.

For the navigation tests (`test_06`), also start the sensor publisher on the
SLAM box:

```bash
python -m robot.odin_pub_node        # or: bash nav.sh
```

From your laptop (or the robot itself):

```bash
cd YOR-v3
python tests/hardware/test_00_connectivity.py --host 192.168.1.10
```

Common flags, accepted by every test:

| Flag | Meaning |
|---|---|
| `--host` | where `robot/yor.py` is (default `192.168.1.10`, the Pi) |
| `--port` | its RPC port (default `5557`) |
| `--timeout` | per-call RPC timeout in seconds (default `2.0`) |
| `--yes` | skip informational pauses. **Never** skips a safety confirmation. |
| `--slam-host` | `test_00`, `test_06` only — where `odin_pub_node` runs |
| `--floor` | `test_03` only — robot is on the floor, not on blocks |

Exit codes: `0` all checks passed, `1` something failed, `2` the run was
aborted before the end (treat as incomplete, not as a pass).

---

## The tests

### Stage 0 — nothing moves

Safe with the robot on blocks or on the floor, arms anywhere. No motion commands
are sent at all.

#### `test_00_connectivity.py`
Everything is talking. RPC answers, both arms report 7 finite joints, all four
swerve modules report encoders, the lift controller is on its serial port, and
(optionally) the Odin is publishing.

Run it first on a cold robot. When a later test fails, this is what tells you it
is not a cable.

#### `test_01_telemetry.py`
The numbers are trustworthy. Encoder timestamps advance, drive counts hold still
while the robot does, steer angles are not jittering, the lift height is stable,
the solver converges, and the SLAM pose is not drifting at rest.

Ends by asking you to spin a wheel by hand — which distinguishes "the encoder is
dead" from "the encoder is fine, nothing was moving".

Almost every confusing behaviour later (*the arm jumped*, *it drove the wrong
way*, *the map is smeared*) traces to one of these signals. Catching it here,
standing still, is far cheaper than catching it at speed.

### Stage 1 — one subsystem at a time

#### `test_02_lift.py` — ⚠️ the lift moves
**Preconditions:** nothing on, above or under the platform; arms clear of the
column; no cables that snag over the full travel.

Homing, the position-known contract, absolute moves through the firmware's
motion profile, stop responsiveness, the streamed-velocity mode, and the travel
constant against a tape measure.

Run this before any whole-body test: the solver uses the lift as a DOF and
trusts `get_lift_height()` completely.

The velocity stages are the ones to watch, because that is the path the
whole-body loop now uses. They run in increasing order of consequence — ±5 mm/s,
±10 mm/s, zero hold, reversal, command timeout, limit switches — and the first
of them checks that the controller advertises `lift_velocity_v1` at all. If it
does not, the board still has the older sketch: flash
`firmware/lift_controller/` before reading anything into the results, because
the whole-body loop falls back to bang-bang up/down/stop without it.

The command-timeout stage is the most important single check in this file. It
is the only thing that stops the column if the host process dies mid-move: the
firmware ramps to zero 300 ms after the last `vel` command and opens the driver
relay.

The travel check matters more than it looks. That number is duplicated in five
places (firmware, `base_motor.py`, `yor.py`, `wholebody_teleop.py`, the MJCF)
and they have drifted apart before — the model said 0.9176 m against a 0.900 m
lift, so the solver kept commanding a height the firmware's software limit would
never deliver. If the measurement disagrees, fix **all five**, listed in
`docs/RUNNING.md`.

#### `test_03_manual_base.py` — ⚠️ the wheels turn
**Preconditions:** on blocks with all four wheels off the ground (default), or
`--floor` with 2 m clear all round.

The axis-convention test, and the one to get right before anything else. Forward,
lateral and yaw each move the robot the way the name says; rotation is about the
centre; all four encoders register a straight run by a similar amount.

Also checks the command watchdog: `base_motor` disables the drive motors if no
command arrives for ~250 ms, which is what stops the robot if a client dies
mid-motion.

Every sign in the stack depends on this. A wrong one makes whole-body control
drive the chassis *away* from the target it is reaching for. Fix signs in
`BaseAxisMap` (`robot/wholebody_control.py`) and nowhere else.

#### `test_04_arms.py` — ⚠️ the arms move
**Preconditions:** a 1 m clear sphere around both arms, empty grippers, and you
standing outside their reach — not between them.

The base is held fixed throughout. Joint readback matches the solver's model
(this one is a safety check, not cosmetics — collision avoidance is computed on
the model), end effectors track Cartesian targets without wandering off-axis,
arms home cleanly, grippers work, and the lift can move while the hands hold
station.

### Stage 2 — subsystems together

#### `test_05_odometry.py` — ⚠️ the robot drives
**Preconditions:** floor, 3 m clear ahead, tape measure, `test_03` passed.

Drives a measured distance and rotation, integrates the encoders through the same
`SwerveOdom` the EKF uses, and compares against the tape. Calibrates
`METERS_PER_ROTATION`, `LENGTH` and `WIDTH` in
`robot/nav/odometry/swerve_odom.py` — and prints the corrected value when a
check fails.

Those constants are deliberately *not* the CAD values in `base_motor.py`: they
are fitted so the forward model matches measured motion. They decide how much the
EKF trusts the wheels between VIO frames, which is what carries you through a
tracking dropout.

#### `test_06_slam_pose.py` — ⚠️ the robot drives
**Preconditions:** `odin_pub_node` running, Odin bolted in its final position,
`T_cam_to_base` set in `config/odin.yaml`, floor with 3 m clear, a lit room with
visual texture.

Proves the SLAM pose is usable, then **calibrates `slam_yaw_sign`** — the one
value you must set before enabling `enable_slam_base_pose` in
`WholeBodyHardwareConfig`. The test rotates counter-clockwise, watches which way
the SLAM yaw goes, and prints the sign to use.

Getting that sign wrong is worse than leaving the feature off: the base pose
error then grows as you drive instead of staying bounded.

### Stage 3 — everything at once

#### `test_07_wholebody.py` — ⚠️ arms, lift **and wheels**
**Preconditions:** `test_02`, `test_03` and `test_04` all passed; 2 m clear in
every direction; a hand on the e-stop.

Run last. Graduates in three steps with a stop between each:

1. **base fixed** — only the arms move, and the test verifies the wheels were
   never commanded
2. **lift joins in** — the torso moves while the hands hold station
3. **base released** — a target beyond arm reach, and the chassis rolls to
   extend it

Step 3 needs the extra confirmation token `ROLL`, because a robot that drives
itself with nobody touching a stick is the most surprising thing this machine
does.

---

## When something fails

Each check prints a `[detail]` with the measured value, and failures usually
print a follow-up line naming the constant or file to look at. Beyond that:

| Symptom | Look at |
|---|---|
| `get_state()` times out | is `robot/yor.py` running? right `--host`? another client hogging the single REP socket? |
| Empty state returned | the node is up but not initialised — check its console for an init exception |
| Lift reports no height | normal before homing. `test_02` homes it. If homing fails, the upper limit switch never closed |
| `Home failed` | switch wiring, or `HOMING_TRAVEL_MM` in the `.ino` is not greater than `MAX_HEIGHT_MM` |
| Lift stalls short of target | the five-place travel constant disagrees with the metal — see `test_02` |
| Robot drives the wrong way | a sign in `BaseAxisMap`. Fix it there, not downstream |
| Rotation translates | `LENGTH`/`WIDTH` do not match the real module positions |
| One wheel does not move | that module's CAN link or encoder — `test_00` names which |
| Solver not converging | the node console prints the solver error; often an unreachable EE target |
| Arms drift from the model | WBC is configured open-loop (`use_measured_arm_state=False`); stop immediately because model-based collision avoidance no longer represents the physical arm pose |
| SLAM pose jitters at rest | lighting and texture, or the Odin is not rigidly mounted |
| SLAM translates during a pure spin | `T_cam_to_base` in `config/odin.yaml` is wrong |

Two behaviours that look like faults but are not:

* **The lift reports no height until it is homed.** The firmware has no zero
  until it has seen a limit switch, and the driver refuses to invent one. That
  is deliberate — a stale height is far more dangerous than a missing one.
* **`get_pose()` returns `None` during whole-body control.** The node only reads
  `slam/pose` while its base controller is in a nav mode, and whole-body dispatch
  pins it to `BASE_VEL`. `test_06` reads the pose from the publisher directly for
  exactly this reason.

## Adding a test

Use `_hw.py`: `check()` for assertions, `confirm()` before motion,
`precondition()` for physical setup, `guard()` around anything that moves, and
`run()` to tally. Keep each test small enough that a failure names one thing.

State what must be physically true in `precondition()` rather than in a comment —
comments do not stop a robot.
