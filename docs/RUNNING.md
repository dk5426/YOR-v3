# Running the Whole-Body IK Demo & Teleop

> **Two ways to drive the robot:**
> 1. **Standalone demo** (`tools/wholebody_ik_demo.py`) — single process, drag targets in the viewer. Sections 1–6 below.
> 2. **Server + teleop client** (`robot/yor_mujoco.py` + `robot/teleop/wholebody_teleop.py`) — the real streaming architecture over RPC. Section 7.

Interactive, kinematic whole-body IK for YORv3 (18 DOF: 3 planar base + 1 lift +
7 left arm + 7 right arm). Drag two target spheres and watch the solver
coordinate the base, lift, and both arms — with self-collision avoidance.

---

## 1. Launch

From the repo root (`YOR-v3/`):

```bash
conda run -n dev mjpython tools/wholebody_ik_demo.py
```

**Why `mjpython` and not `python`?** On macOS the MuJoCo passive viewer must run
on the main thread under `mjpython` (a thin wrapper shipped with the `mujoco`
package). Plain `python` will raise a "launch_passive requires mjpython" error.

On launch you should see a banner printing the solver, DOF layout, rate, and:

```
  Collision avoidance : ON (167 arm↔lift / arm↔base / arm↔arm / ↔floor pairs)
```

### Prerequisites (already set up in the `dev` env)
- `mujoco` (3.10.0), `mink`, `loop_rate_limiters`, and a QP backend (`pyqpmad`).
- Verify quickly:
  ```bash
  conda run -n dev python -c "import mujoco, mink, loop_rate_limiters, qpsolvers; print('ok')"
  ```

---

## 2. Keyboard Controls

| Key | Action |
|-----|--------|
| `W` / `S` | Move **left** target ±X (forward / back), 5 cm |
| `A` / `D` | Move **left** target ±Y (left / right), 5 cm |
| `Q` / `E` | Move **left** target ±Z (up / down), 5 cm |
| `M` | Toggle **auto-animation** (Figure-8 on both targets) |
| `C` | Toggle **self-collision avoidance** (arm↔lift / arm↔base / arm↔arm) |
| `ENTER` | Toggle **fix-base** (lock base x/y/θ — only arms + lift move) |
| `R` | Reset to the `home` keyframe |
| `SPACE` | Pause / resume the solver |

> The right target has **no keyboard binding** — move it by dragging (below) or
> via the numeric panel. Keyboard nudges only affect the left (green) target.

Every 2 s the terminal prints telemetry: per-arm position error, solve status,
iteration count, base pose `(x, y, θ)`, lift height, and `fix_base`.

---

## 3. Moving the Targets with the Mouse / Trackpad

Two target spheres are mocap bodies:
- **Green sphere** = `left_ik_target` → left end-effector
- **Magenta sphere** = `right_ik_target` → right end-effector

To drag one:
1. **Double-click** the sphere to select it (a selection marker appears).
2. Hold **Ctrl + right-mouse-button** and drag to **translate** it.
3. Hold **Ctrl + left-mouse-button** and drag to **rotate** it (orientation target).

### Trackpad tips (macOS)
"Ctrl + right-drag" is painful on a trackpad. Easiest options:
- **Plug in a real mouse** — by far the least frustrating.
- Or set **System Settings → Trackpad → bottom-right corner = Secondary click**,
  so a corner tap acts as right-click.
- Or skip dragging: select the sphere, then nudge its `pos` numerically from the
  **right-hand panel** fields (precise, trackpad-proof).
- Or just use **W/A/S/D/Q/E** for the left arm and **M** to auto-exercise both.

---

## 4. Exploring the Viewer

### Camera navigation
| Input | Action |
|-------|--------|
| Left-drag | Orbit the camera |
| Right-drag | Pan |
| Scroll / two-finger swipe | Zoom |
| Double-click a body | Select it (then the panel shows its data) |
| `Esc` | Deselect |

### The right-hand panel
Launched with `show_right_ui=True`. Useful tabs:
- **Rendering** — toggle contacts, transparency, convex hulls, etc.
- **Mocap** — when a target sphere is selected, edit its position/orientation
  numerically (the trackpad-free way to move targets).

### Visualization toggles (keys handled by the viewer itself)
- The demo starts with **frame axes off** (`mjFRAME_NONE`) to hide the ~20 finger
  site axes. Cycle frame display with the viewer's own frame toggle in the
  Rendering panel if you want them back.
- To *see* the collision proxies: in **Rendering → Geom group**, enable **group 3**
  (the collision spheres) and/or **group 2** (visual meshes). The IK collision
  avoidance is computed against the group-3 spheres + the chassis mesh.

### What to look for
- **Reachability coordination:** drag a target far out. With `fix_base` OFF the
  chassis rolls and the lift extends to help reach it; the `base(...)` telemetry
  changes. Press `ENTER` to lock the base and watch reach shrink to arms+lift.
- **Collision avoidance:** with `C` ON, push the left target across the body
  toward the right arm or down onto the chassis — the arm stops short instead of
  passing through. Toggle `C` OFF and repeat to see it interpenetrate.
- **Ground avoidance:** drive a target down to the floor (`E` repeatedly, or
  drag the sphere to z≈0) — the hand stops ~2 cm above the plane, fingers
  included, instead of clipping through the ground.

---

## 5. Tuning (edit `tools/wholebody_ik_demo.py` → `WholeBodyIKConfig`)

| Param | Effect |
|-------|--------|
| `base_posture_cost` | Higher = base more reluctant to move (demo uses `5e-2`). Lower to `1e-3` to make the chassis participate more. |
| `lift_posture_cost` | Lower = lift stretches more eagerly (demo `1e-4`). |
| `collision_gain` | Approach speed toward obstacles, (0, 1]. Raise toward `1.0` to hold the full 2 cm buffer; lower for smoother, more conservative motion. |
| `collision_min_distance` | Buffer (m) kept between avoided geoms (default `0.02`). |
| `collision_detect_distance` | Range (m) at which avoidance switches on (default `0.06`). |
| `enable_collision_avoidance` | Initial state of the `C` toggle. |
| `enable_ground_avoidance` | Keep arms + hands (finger meshes) clear of the floor plane (default on; shares the `C` toggle and buffer). |
| `SOLVER` (top of file) | QP backend. `pyqpmad` is fastest on this problem (~0.12 ms). |

---

## 6. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `launch_passive requires mjpython` | Run with `mjpython`, not `python`. |
| Viewer freezes/hangs on macOS | Don't add `print()` inside `key_callback` — terminal I/O on the GLFW callback thread deadlocks the viewer (already avoided in the script). |
| Targets won't drag | Use **double-click to select**, then **Ctrl + right-drag**; on a trackpad prefer a mouse or the numeric panel. |
| Animation works (`M`) but arms won't move otherwise | That's the right (magenta) target — it has no keyboard binding. Use WASD for the *left* arm, or drag. |
| `data.time` never advances | Expected: kinematic mode uses `mj_forward`, not `mj_step`. The Figure-8 uses wall-clock `time.time()` for this reason. |
| Base never moves | Check telemetry shows `fix_base=False` (press `ENTER` to unlock), then drag a target out of arm+lift reach. |

---

## 7. Whole-Body Teleop (server + client over RPC)

The production-shaped path: the sim runs as an RPC server, a separate teleop
client streams EE / lift targets to it at 30 Hz (the client's default rate;
the sim server below still solves whole-body IK at its own 108 Hz, unrelated
to the client rate — see the pipeline note below for how this differs on
hardware, where the two are matched 1:1).

### Start the server (terminal 1)
```bash
conda run -n dev mjpython robot/yor_mujoco.py
```
Opens the MuJoCo viewer, runs whole-body IK at 108 Hz, and serves commlink RPC
on **port 8081**.

### Start the client (terminal 2 — plain `python`, no viewer)
```bash
conda run -n dev python robot/teleop/wholebody_teleop.py --input keyboard
```

`--input` selects the backend:

| Backend | Needs | Notes |
|---------|-------|-------|
| `keyboard` (default) | nothing | nudge-based, works everywhere |
| `gamepad` | `pip install pygame` + controller | hold **L1**/**R1** to steer left/right arm with the sticks, D-pad = lift |
| `oculus` | Quest streaming to `--oculus-host` | clutch teleop: X/A engage, full 6-DoF pose following |

### Quest pose filtering (`--input oculus`)

Controller poses are smoothed on arrival with a 1€ filter, so tracker jitter
does not reach the IK and a tracking dropout holds the target instead of
throwing the arm at it. Dropped samples are reported on the console.

| Flag | Default | Effect |
|------|---------|--------|
| `--filter-min-cutoff` | `3.0` Hz | smoothing while the hand is still — lower is calmer but laggier |
| `--filter-beta` | `8.0` | how fast the filter opens up with hand speed — raise it if fast reaches feel sluggish |
| `--no-pose-filter` | off | stream raw poses (for comparing / debugging the headset) |

The 72 Hz input gives the 3 Hz resting cutoff 24 samples per cycle. The adaptive
cutoff rises with hand speed so a moving controller remains responsive.

On hardware, the command path is:

```text
Quest/filter ~72 Hz -> teleop RPC 30 Hz -> whole-body IK 30 Hz -+-> arm dispatch 90 Hz (3 sub-steps/solve)
                                                                 |      -> nerolib onboard tracking 250 Hz
                                                                 +-> lift / base dispatch 30 Hz
                                                                        -> base relay 108 Hz -> swerve profiling 324 Hz
```

Teleop RPC and whole-body IK are matched 1:1 (`WholeBodyHardwareConfig.control_hz`
and the teleop client's `LOOP_RATE` both 30 Hz). Arms are the one leg that does
*not* follow the solve rate straight to nerolib: each solved joint target is
interpolated against the previous one over `arm_interpolation_steps` (3)
sub-steps, dispatched by a separate thread at `arm_dispatch_hz` (90 Hz) with a
short `arm_preview_time` — this reproduces YOR_D's own arm-commanding chain
(a Cartesian target no faster than 30 Hz, smoothed by frequent short-duration
joint commands) rather than sending nerolib one coarse ~33 ms segment per
solve tick. Nerolib's 250 Hz onboard loop then tracks whichever of those
90 Hz segments it was last given. Lift and base dispatch, by contrast, go
straight from the 30 Hz solve loop with no such intermediate stage — a
velocity target held slightly longer just keeps driving at that velocity
rather than decelerating to a stop, so they don't need one. The base relay is
deliberately *not* matched to the WBC solve rate either: it stays at 108 Hz
so it never falls behind a (now slower) producer, and so the swerve loop's
own 3x oversampling (108 Hz relay -> 324 Hz profiling) stays exactly as
tuned. Quest tracking arrives at roughly 72 Hz independent of all of the
above and is 1€-filtered on its own receive thread; the 30 Hz teleop loop
just samples whatever pose that filter last produced. The lift's Arduino
height telemetry (~36 Hz) is likewise a separate, fixed hardware rate — it
now arrives at roughly the same cadence as the 30 Hz WBC loop consumes it,
rather than once every three ticks as when the WBC loop ran at 108 Hz.
Safety keepalives,
timeouts, variable step pulses and the separate 20 Hz navigation/SLAM loop are
deliberately not control samples in this hierarchy.

### Keyboard map (client terminal, not the viewer!)
```
Left arm   w/s ±X   a/d ±Y   q/e ±Z      (2 cm nudges; [/] resize)
Right arm  i/k ±X   j/l ±Y   u/o ±Z
Lift       r / f
Home       h (left)  n (right)  g (lift)
Toggles    t fix-base   c collision avoidance
Quit       x or ESC
```

A 1 Hz status line shows both target positions, lift, and toggle states.

### RPC API exposed by the server
`set_left_ee_target` / `set_right_ee_target` / `set_bimanual_ee_target`
(mink.SE3), `set_lift_target(m)`, `home_left_arm` / `home_right_arm` /
`lift_home`, `toggle_fix_base`, `toggle_collision_avoidance`,
`get_state()` (poses + lift + base + flags as plain types),
`get_lift_position`, `get_base_velocity` ([vx, vy, ω] from the last solve —
this is what the base consumes on hardware).

The hardware node (`robot/yor.py`, port 5557) exposes the same names, so the
same client drives it with `--target hw`. See section 8.

---

## 8. Running on the Robot

The hardware node runs the *same* solver over the *same* description; it reads
an initial arm-encoder seed followed by the previous commanded arm state,
plus measured lift height, instead of `data.qpos`, and dispatches to
nerolib, the PicoLift and the swerve base instead of writing `qpos`.

### Start the node (on the robot)
```bash
python robot/yor.py
```
`init()` starts by bringing the swerve controllers to the commissioned PID
gains in [config/base_pid_manifest.json](../config/base_pid_manifest.json)
itself — no separate step. It writes through the SparkFlex objects
`robot/base_motor.py` has already opened, before the base control loop starts,
so only one set of device handles ever touches the bus. Each controller is read
first and written only if it differs, so an ordinary restart writes nothing and
logs `already-set` eight times. Gains live in controller RAM, which a power
cycle clears, which is why this is checked on every start: a module that
reverted steers and drives differently from its three neighbours, and that is
far harder to diagnose from the robot's behaviour than a startup failure.

If a controller cannot be brought to the commissioned values, `init()` raises
and the base control loop is never started.

```bash
python robot/yor.py                       # sync the gains at startup (default)
python robot/yor.py --no-flash-base-pid   # start on whatever the controllers hold
python robot/yor.py --base-pid-manifest config/experimental_pid.json
```

`--no-flash-base-pid` is for starting the node on a bench with no CAN bus, or
for deliberately leaving a controller on the gains it is holding while
investigating it — remember that after a SPARK power cycle those are the stock
gains, not the commissioned ones. In code the same switch is
`YOR(flash_base_pid=False)`.

`python tools/base_pid_preflight.py` remains available for checking or applying
the gains while the robot process is *not* running — see
[Swerve PID preflight](#swerve-pid-preflight) below.

`robot/yor.py` then initialises both arms (each homes through nerolib), starts
the base control loop, then starts whole-body control at 30 Hz and serves
commlink RPC on **port 5557**.

### Drive it (from the operator machine)
```bash
python robot/teleop/wholebody_teleop.py --target hw --host <robot-ip> --input oculus
```
Everything else — keys, clutch behaviour, toggles — is identical to the sim.

### Bring-up checklist

Do these in order the first time. Steps 1–3 need no whole-body control at all.

1. **Base axes.** Run `robot/teleop/joystick.py` and confirm the robot drives
   the way the sticks point, and that the D-pad raises/lowers the lift.
2. **Lift travel.** Run `lift_home()` first — the firmware has no zero until
   it has seen a limit switch, and `get_lift_height()` returns `None` until it
   does. Then check the height spans 0 → 0.900 m bottom to top.

   That number is duplicated in **five** places and they must all agree. They
   drifted apart once (the model said 0.9176 m against a 0.900 m lift), which
   made the solver command heights the firmware's software limit would never
   reach:

   | Where | What |
   |---|---|
   | `firmware/lift_controller/lift_controller.ino` | `MAX_HEIGHT_MM` |
   | `robot/base_motor.py` | `LIFT_MAX_HEIGHT_M` |
   | `robot/yor.py` | `lift_*` `max_height_m` defaults |
   | `robot/teleop/wholebody_teleop.py` | `LIFT_RANGE` |
   | `description/robot_wholebody.xml` | `Slider 7` range **and** `lift_joint_pos` ctrlrange |
3. **Arm homing.** `python robot/arm/arm.py` moves one arm to its home pose.
4. **Whole-body, base disabled.** Start `robot/yor.py`, then immediately press
   `t` in the teleop client (fix-base) or call `toggle_base_motion(False)`.
   Nudge each hand a few centimetres and confirm the arms track.
5. **Lift under the solver.** Use `r` / `f`. The torso should move while the
   hands hold station — the arms compensate.
6. **Base last.** Clear floor space, then release fix-base and push a target
   beyond arm reach. The chassis should roll *slowly* (clamped to 0.25 m/s and
   0.6 rad/s) in the direction of the target. If it drives the wrong way, fix
   the signs in `BaseAxisMap` (`robot/wholebody_control.py`) rather than
   compensating elsewhere.

### Navigation (Odin 1 SLAM)

Localization and mapping run on a Manifold Odin 1 over its native C SDK — no
ROS, no ZED. Two processes, launched together by `nav.sh`:

```bash
bash setup_odin.sh --full     # once per machine: env, deps, build pyodin
bash nav.sh                   # tmux: odin_pub_node + slam_node_
bash nav.sh --kill
```

| Process | Publishes / serves | Purpose |
|---|---|---|
| `robot/odin_pub_node.py` | `slam/*` on **:6000** | device → Y-up poses, RGB, depth, cloud |
| `robot/slam_node_.py` | Viser on **:8099** | voxel map, 2D grid, A*, `follow_path` RPC |

Topic names and message layouts live in `robot/topics.py`. `robot/base.py`
subscribes to `slam/pose` on its own for closed-loop path following, so the
publisher must be up before `follow_path`/`move_to` will do anything.

Click a point in the Viser UI and A* plans a route, which is streamed to
`yor.py` as `follow_path` waypoints — **the robot drives on click**, so treat
:8099 as a live control surface.

Bring-up order that matters:

1. **Mount extrinsic first.** `T_cam_to_base` in `config/odin.yaml` is identity,
   which is only correct with the Odin off the robot. Measure it once bolted on
   — every pose the nav stack and base controller act on comes through it.
2. **USB and firmware.** USB **3.0** port and cable, Odin firmware ≥ 0.12.0,
   and the udev rule from `setup_odin.sh`'s header. Check with `lsusb -d 2207:`.
3. **One consumer at a time.** The Odin is a single USB device — never run two
   publishers or a stray driver alongside `nav.sh`.
4. **Stop with Ctrl-C, not `kill -9`.** A hard kill can wedge the device's
   onboard daemon, which then needs a power cycle. (The SDK segfaults harmlessly
   in its exit destructors; `odin_pub_node` hard-exits past it with `os._exit`.)

The pose feeding navigation is EKF-fused by default: swerve wheel odometry
predicts at 20 Hz from `get_base_encoders()` over the YOR RPC, and each VIO
frame corrects it with confidence-adaptive measurement noise. That means
`robot/yor.py` should be running — if it is not, the predict step logs RPC
timeouts and the filter coasts on VIO alone. Pass `--no-ekf` to
`robot.slam_node_` to use the raw VIO pose instead.

The odometry geometry in `robot/nav/odometry/swerve_odom.py` (`LENGTH`,
`WIDTH`, `METERS_PER_ROTATION`) is *calibrated*, deliberately different from the
nominal CAD values in `robot/base_motor.py`. Do not reconcile them by hand.

### The lift

Firmware lives in [firmware/lift_controller/](../firmware/lift_controller/) and
is flashed to the Arduino driving the stepper. `robot/base_motor.py`'s
`PicoLift` talks to it over serial at 115200.

The lift Arduino defaults to its stable udev path,
`/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG02XPI5-if00-port0`. For
replacement hardware or bench testing, override it with
`YOR_LIFT_SERIAL_PORT=/dev/...` when starting `robot/yor.py`.

```
up | down          continuous move until stop/limit
up <mm> | down <mm>  finite move, firmware runs a jerk-limited S-curve
vel <signed mm/s>  streamed velocity, + up / - down (whole-body path)
stop | home        home drives UP to the upper switch, which defines 900 mm
status             limit switches + motion state + height + capabilities
power on | off     driver relay
```

Four behaviours worth knowing before you debug it:

- **Height only streams while moving.** When idle the last value stands, so
  `get_lift_height()` returns `None` from boot until the first move or home
  (unless the lift happens to boot sitting on a limit switch).
- **Position can be lost.** The firmware reports `Height: unknown (run home)`
  and the host clears its cached height on that, on a failed/aborted home, and
  on the reset banner. `lift_position_known()` tells you which state you are
  in; `lift_to_height()` refuses to move when the position is not established,
  rather than acting on a stale reading.
- **A stop cuts driver power**, so the next move re-powers with a ~500 ms
  startup delay. That is why `lift_to_height()` now hands the whole distance to
  the firmware as one `up <mm>` / `down <mm>` command instead of bang-banging
  `up` then `stop`: the move gets a real acceleration profile and stops on an
  exact pulse count. Pass `profiled=False` for the old behaviour.

- **The whole-body path streams velocity, not up/down.** `set_lift_target()` is
  still a height in metres — nothing in the teleop client changes — but the
  controller now converts it to a velocity and streams that (see
  [The lift under whole-body control](#the-lift-under-whole-body-control)).

`get_lift_status()` returns height, position-known, homed, both limit switches,
motion state, the last notable firmware line, how old the height reading is,
and whether the firmware advertised streamed velocity.

### The lift under whole-body control

`WholeBodyController` runs a position PD against the measured height and
streams the result as a velocity:

```
velocity = Kp * (desired - measured) - Kd * filtered d(measured)/dt
```

| Setting | Value | |
|---|---:|---|
| `lift_kp` | 2.0 | 1/s |
| `lift_kd` | 0.05 | s |
| `lift_derivative_tau` | 0.1 | s — height arrives at ~36 Hz, the loop now runs at 30 Hz |
| `lift_velocity_deadband_m` | 0.005 | inside this the command is exactly zero |
| `lift_max_velocity_m_s` | 0.05 | the host's clamp; the firmware clamps at 50 mm/s too |
| `lift_feedback_max_age_s` | 0.5 | older than this, while driving, stops the lift |
| `lift_feedback_grace_s` | 1.0 | covers the firmware's 500 ms driver-relay delay |

Four things about this are worth knowing:

- **The derivative is of the measurement, not of the error.** An operator lift
  command is a step, and differentiating the error would ask for metres per
  second on the first cycle. The D term here is pure damping.
- **It only runs against a firmware that says it can.** The controller reads
  the `Capabilities: lift_velocity_v1` line; without it the loop falls back to
  the original `up`/`down`/`stop` deadband servo, and says which path it chose
  at startup. An older sketch answers `vel` with its usage banner and does not
  move, so the check has to be the controller's word, not the host's.
- **Stale or unknown height stops it, and keeps it stopped.** The refusal
  latches until a fresh reading arrives; a held lift is asked for a `status`
  every two seconds so it can recover on its own. `get_state()` reports
  `lift_velocity_mode`, `lift_command_velocity` and `lift_feedback_age_s`.
- **The firmware shapes what it is sent.** It plans a quintic minimum-jerk
  transition to each new velocity (200 mm/s², 2000 mm/s³), ramps through zero
  before changing the direction pin, and stops if no command arrives for
  300 ms. It is *not* a closed-loop velocity controller — it has no measured
  column velocity to close against.

### Swerve PID preflight

The commissioned gains live in
[config/base_pid_manifest.json](../config/base_pid_manifest.json):

| Motor role | Kp | Ki | Kd | Velocity FF | Output range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drive | 0.35 | 0.0 | 0.0 | 0.23 | -1.0 to 1.0 |
| Steering | 20.0 | 0.0 | 6.0 | 0.0 | -0.25 to 0.25 |

`robot/yor.py` applies these on every start, so the command below is for
working on the base without the robot node running — confirming what a
controller currently holds, or applying the gains after editing the manifest.

```bash
python tools/base_pid_preflight.py                # validate, apply, verify
python tools/base_pid_preflight.py --dry-run      # checks only, opens no device
python tools/base_pid_preflight.py --verify-only  # read back, write nothing
```

It validates the manifest, cross-checks the module CAN ids against
`robot/base_motor.py`, checks the CAN interface is up and that no other process
owns the controllers — and only then opens a device. Every field it writes is
read back, and any difference fails the run. Because it opens its own devices,
it refuses to run while `robot/yor.py` is up: two sets of SparkFlex objects on
one bus is not safe. The in-process sync at startup runs the same manifest
checks and the same readback, but reuses the handles the robot already holds.

The steering output limit of ±0.25 is part of the commissioned set. The
proposed full-range combination (`Kp=10`, `Kd=0`, output ±1.0) is **not**
validated and must stay a separate commissioning experiment; drive `Kd` is
deliberately zero, because a tested value of 10 produced an audibly harsh
100 Hz torque ripple.

### Feeding the SLAM pose into whole-body IK (optional)

By default the solver's base pose is dead-reckoned from the velocity it
commanded, so it drifts. `WholeBodyHardwareConfig.enable_slam_base_pose` wires
the Odin pose in as a **drift correction** — dead-reckoning stays primary and
the absolute pose is bled in under a rate limit, so a loop-closure jump never
reaches the IK as a step.

It ships **off**, because one value has to be calibrated first: `slam_yaw_sign`
(+1 or −1), the handedness of the SLAM planar frame relative to the IK one.
Everything else — rotation and translation between the frames — is solved
automatically from the first pose pair, so the correction always starts at zero.

Calibration, about 30 seconds:

```python
cfg = WholeBodyHardwareConfig(enable_slam_base_pose=True, slam_yaw_sign=+1.0)
```

1. Start `odin_pub_node` and `yor.py`. Watch for `[wholebody] SLAM base
   correction aligned` — no message means no pose is arriving.
2. Drive the base forward ~1 m with the joystick.
3. Poll `get_state()["slam_base_correction_m"]`.
   - **Stays small** (a few cm, not trending): sign is right.
   - **Grows steadily** with distance: flip `slam_yaw_sign` to −1.0 and repeat.

A wrong sign is worse than leaving the feature off — the error then grows as you
drive instead of staying bounded, which is why the default is disabled.

`get_state()` also reports `slam_base_pose_age` (seconds since the last pose,
`None` when the feature is off). If it exceeds `slam_pose_max_age_s` the
correction pauses and the loop coasts on dead-reckoning.

Two things that make this less obvious than it looks:

- The listener runs its **own** subscriber thread rather than reusing
  `BaseController`'s pose. Whole-body dispatch pins that controller to
  `"BASE_VEL"` mode, and in that mode its loop `continue`s before it ever calls
  `get_pose` — so `yor.pose` is stale exactly while whole-body control runs.
- The control loop never touches the network. commlink's subscriber is
  pull-mode (one round trip per read), which a 30 Hz loop cannot afford, so the
  loop only ever reads a cached value.

### Tuning individual arm joints in the solver

`WholeBodyIKConfig.arm_posture_cost` applies one weight to all 14 arm joints.
`arm_posture_cost_overrides` sets them per joint — higher cost means the solver
moves that joint less and reaches with the others instead:

```python
WholeBodyIKConfig(
    arm_posture_cost=1e-3,
    arm_posture_cost_overrides={
        "left_arm_joint1": 1e-2,   # 10x stiffer — spare a strained shoulder
        "left_arm_joint7": 1e-4,   # 10x softer  — let the wrist absorb it
    },
)
```

Or retune live, without restarting: `controller.ik.set_arm_posture_costs({...})`.
Merges by default; pass `replace=True` to drop existing overrides.

Only the ratio to `arm_posture_cost` matters. Measured on one 0.10 m EE target,
varying `left_arm_joint1` against `arm_posture_cost=1e-3`: 10× stiffer cuts that
joint's travel from 0.171 to 0.111 rad, 100× effectively pins it, and beyond
that it saturates — so ~0.1× to ~100× is the useful band. The arm's *total*
motion rises as one joint stiffens, because the others take up the slack.

An unknown joint name raises rather than being silently ignored, and a cost of 0
is rejected (mink requires a positive cost — use `1e-8` for "effectively free").
To stop a joint moving outright, limit its velocity in `_build_velocity_limits`
instead: that's a hard QP constraint, so unlike a cost it cannot be traded away
against the end-effector task.

### Arbitration between whole-body and direct control

Direct commands win, briefly. Any call to `set_base_velocity` / `move_to` /
`follow_path` suspends the solver's authority over the base for
`manual_override_timeout_s` (0.5 s by default); `lift_up`/`lift_down`/
`lift_stop` do the same for the lift, and `set_*_joint_target` for the arms.
Because joystick.py streams at 60 Hz, holding the stick keeps the base under
manual control continuously, and letting go hands it back.

To stop the solver for longer, use `park()` or `tuck_arms()` (both stop the
loop), then `resume_wholebody()`.

### Known limitations on hardware

| Limitation | Consequence |
|---|---|
| The *solver's* base odometry is dead-reckoned from the commanded velocity | The solver's idea of where the chassis is drifts. Fine for clutch-based teleop (targets are relative to the current EE pose); absolute world-frame targets degrade the longer the base drives. Fixable — see "Feeding the SLAM pose into whole-body IK" above — but off until `slam_yaw_sign` is calibrated, so it drifts by default. |
| The Odin's `slam/pcd` cloud is unordered | `pcd_source: slam` reshapes it to `(1,N,4)`, so mapping's 3×3 flying-pixel rejection spans unrelated points and is effectively inert. Acceptable because the Odin filters on-device, but do not read that filter as active. |
| `BaseAxisMap` signs are unverified | Checked against the conventions in `base.py`, not against the physical robot. Do step 1 of the checklist. |
| The lift PD is not a velocity loop | The host commands a velocity and the firmware shapes it, but nothing measures column velocity, so a stalled or slipping column is only visible as height that stops changing. The 5 mm deadband is the practical holding accuracy. |
| Against an older lift firmware the lift is bang-bang | Without `Capabilities: lift_velocity_v1` the loop falls back to the 1 cm deadband up/down servo, which cannot hold an arbitrary height as precisely. Check which path is live in `get_state()["lift_velocity_mode"]`. |
| Self-collision and ground avoidance share one toggle | `c` turns both on or off; separating them needs two `CollisionAvoidanceLimit` instances. |
| EE frames moved | The whole-body description puts `left_arm_ee`/`right_arm_ee` at the wrist flange with the Wuji hand attached, and the model frame is rotated 90° from the old per-arm description. Any pose recorded against the old `nero-welded-base-and-lift.mjcf` needs re-recording. |
