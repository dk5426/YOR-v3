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
client streams EE / lift targets to it at 30 Hz.

### Start the server (terminal 1)
```bash
conda run -n dev mjpython robot/yor_mujoco.py
```
Opens the MuJoCo viewer, runs whole-body IK at 100 Hz, and serves commlink RPC
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
the arm encoders and lift height instead of `data.qpos`, and dispatches to
nerolib, the PicoLift and the swerve base instead of writing `qpos`.

### Start the node (on the robot)
```bash
python robot/yor.py
```
This initialises both arms (each homes through nerolib), starts the base
control loop, then starts whole-body control at 100 Hz and serves commlink RPC
on **port 5557**.

### Drive it (from the operator machine)
```bash
python robot/teleop/wholebody_teleop.py --target hw --host <robot-ip> --input oculus
```
Everything else — keys, clutch behaviour, toggles — is identical to the sim.

### Bring-up checklist

Do these in order the first time. Steps 1–3 need no whole-body control at all.

1. **Base axes.** Run `robot/teleop/joystick.py` and confirm the robot drives
   the way the sticks point, and that the D-pad raises/lowers the lift.
2. **Lift travel.** Check `get_lift_height()` at the bottom and top of travel
   actually spans 0 → 0.9176 m. That number lives in two places that must
   agree: `LIFT_MAX_HEIGHT_M` in `robot/base_motor.py` and the `Slider 7`
   range in `description/robot_wholebody.xml`.
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
| Base odometry is dead-reckoned from the commanded velocity | The solver's idea of where the chassis is drifts. Fine for clutch-based teleop (targets are relative to the current EE pose); absolute world-frame targets degrade the longer the base drives. |
| `BaseAxisMap` signs are unverified | Checked against the conventions in `base.py`, not against the physical robot. Do step 1 of the checklist. |
| The lift is bang-bang | It servos to a deadband (1 cm by default), so it cannot hold an arbitrary height as precisely as the sim. |
| Self-collision and ground avoidance share one toggle | `c` turns both on or off; separating them needs two `CollisionAvoidanceLimit` instances. |
| EE frames moved | The whole-body description puts `left_arm_ee`/`right_arm_ee` at the wrist flange with the Wuji hand attached, and the model frame is rotated 90° from the old per-arm description. Any pose recorded against the old `nero-welded-base-and-lift.mjcf` needs re-recording. |
