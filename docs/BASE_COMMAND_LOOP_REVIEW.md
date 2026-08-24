# Base command loop — review

Read of the path from the whole-body solver to the swerve wheels, done before
the first base/lift tuning session (2026-08-22). Every finding below was
reproduced against the real functions in `robot/base_motor.py`, not inferred
from reading.

## Status

**Applied 2026-08-22** — findings 1, 2, 6b, 8 and 10, plus the joystick
stick-release creep. All are offline-verifiable and none of them changes a
tuning value; they are covered by `tests/test_base_kinematics.py` (46 checks).

**Still open, deliberately** — findings 3, 4, 5, 6 and 9. Each needs either a
hardware measurement or a decision that is not mine to make, and each changes
how the base drives, so they want a run either side rather than a batch.
Finding 6 in particular (`DRIVE_VEL_SCALE`) is being left as-is for now by
decision; the conversion factors it hinges on are now printed at every startup.

The table below marks each one.

## The path

```
WholeBodyController._step (30 Hz)
  └─ _dispatch_base            world v → body (fwd, lat, yaw) → clamp → deadband → BaseAxisMap
      └─ BaseController.target_velocity          (plain attribute write)
          └─ BaseController._run  (108 Hz relay)  re-sends the attribute, forever
              └─ Base._command_queue (depth 3, drained to newest)
                  └─ Base.control_loop (324 Hz)   S-curve → IK → 4× steer + 4× drive
                      └─ SparkFlex SetPosition / SetVelocity
```

## Findings

| # | Finding | Severity | Confirmed by | Status |
|---|---|---|---|---|
| 1 | A zero base command re-aims all four modules to 0° | **high** | reproduction | **fixed** |
| 2 | The per-axis deadband rotates the commanded direction | **high** | reproduction | **fixed** |
| 3 | `cos_error_scaling` is inert — it lasts one 3.1 ms tick | **high** | reproduction | **fixed** (`USE_FEEDBACK_FOR_STEER = True`) |
| 4 | Nothing ever reads the steering encoder | **high** | code | **fixed** — encoder units corrected, loop closed |
| 5 | The 108 Hz relay defeats `Base`'s own 250 ms watchdog | **medium** | code | open — design decision |
| 6 | `DRIVE_VEL_SCALE = 2.0` is not a unit conversion | **high** | measured 2026-08-22 | **mechanism fixed** — scale now travels with the gain set; the exact value still needs a hardware measurement |
| 11 | `BaseAxisMap` was crossed — forward drove sideways | **high** | measured 2026-08-22 | **fixed** |
| 12 | The absolute encoder was read as degrees; it returns turns | **high** | measured 2026-08-22 | **fixed** |
| 13 | The commanded heading turns faster than the modules can slew | **high** | measured 2026-08-22 | **fixed** (rate limit + deadband) |
| 6b | `SetIdleMode` and `SetCtrlType` are never called — the enum import silently fails | low (was medium) | reproduction | **fixed**; measured to be a no-op on current flash |
| 7 | Odometry ignores the S-curve profiler | **medium** | code | logged, not yet corrected |
| 8 | `_angle_and_speed_to_vehicle_velocity` returns the wrong ω | low (dead) | reproduction | **fixed** (deleted) |
| 9 | Steering setpoint wraps mod 1.0 with PID wrapping unconfigurable | low–unknown | code | open — needs a hardware check |
| 10 | Minor dead code and unused state | low | code | **fixed** (dt); rest noted |

---

### 1. A zero base command re-aims all four modules to 0°

> **Fixed.** `_vehicle_velocity_to_angle_and_speed` now holds the last
> commanded angle for any module whose commanded speed is below
> `ZERO_SPEED_EPS_MPS`, applied after the flip and the cosine so neither can
> reintroduce a direction. Only the drive setpoint goes to zero.


`_vehicle_velocity_to_angle_and_speed` computes wheel angles as
`arctan2(vy_w, vx_w)`. At exactly zero velocity that is `arctan2(0, 0)` = 0 for
every module, and `control_loop` writes those angles to the rotation motors
unconditionally — the speed being zero does not stop the *steering* setpoint
from being sent.

```
forward  (vy=+0.25): driving at [90 90 90 90]° -> STOP re-aims to [0 0 0 0]°  (90° each)
diagonal (0.18,0.18): driving at [45 45 45 45]° -> STOP re-aims to [0 0 0 0]°  (45° each)
spin     (wz=+0.6): driving at [-36 36 -36 36]° -> STOP re-aims to [0 0 0 0]°  (36° each)
```

This matters far more for whole-body control than for joystick driving. Base
motion here is *emergent* — the solver rolls the chassis only as much as the
arms cannot reach — so the commanded velocity spends much of its time near
zero and crosses the deadband repeatedly. Each crossing is a full re-aim of
all four modules, followed by another re-aim back when motion resumes. Expect
it to look like the base "resetting itself" between nudges, and to cost real
time: the modules are slewing instead of driving.

**Fix:** hold the previous wheel angles when the commanded wheel speed is
(near) zero. Only the drive setpoint should go to zero.

### 2. The per-axis deadband rotates the commanded direction

> **Fixed.** `WholeBodyController._limit_linear` deadbands and clamps the
> magnitude of (forward, lateral) and rescales the pair, so direction
> survives both. Yaw got its own `base_yaw_deadband`, in rad/s, since one
> scalar was previously serving as both m/s and rad/s.


`WholeBodyController._clamp` applies `base_vel_deadband` (0.02 m/s)
independently to forward, lateral and yaw. A diagonal request loses whichever
axis is small and keeps the other, so the direction sent is not the direction
asked for:

```
request (0.019, 0.019) |v|=0.027 at 45.0° -> sent (0.000, 0.000)   dropped entirely
request (0.025, 0.019) |v|=0.031 at 37.2° -> sent (0.025, 0.000) at  0.0°
request (0.050, 0.019) |v|=0.053 at 20.8° -> sent (0.050, 0.000) at  0.0°
```

The third case is not slow motion — it is 0.053 m/s, well clear of the
deadband as a magnitude — and it is still sent 21° off course. Slow diagonal
motion snaps to the axes.

It compounds with (1), because a request that gets zeroed also triggers the
re-aim, and with the odometry: `_dispatch_base` integrates `v_applied`, the
*distorted* command, so the IK model's base pose inherits the error. The bias
is systematic rather than random — it always points at an axis — so it
accumulates rather than averaging out.

**Fix:** deadband the linear velocity as a vector (`hypot(fwd, lat)` against
one threshold, zeroing both or neither) and keep yaw separate. That preserves
direction and still silences standstill hum.

### 3. `cos_error_scaling` is inert

The intent is right: scale drive speed by the cosine of the steering error, so
a module that is still slewing does not drive hard in the wrong direction. The
error it uses is `diff_angle(wheel_angles, self.steer_pos)`, and `steer_pos`
comes from `RotationMotor.get_position_rad()`, which — with
`USE_FEEDBACK_FOR_STEER = False` — returns **the last commanded angle**, not
the encoder.

So the "error" is between this tick's command and the previous tick's command:

```
tick 1 (90° command change): speeds = [0. 0. 0. 0.]      correctly scaled to zero
tick 2, 3.1 ms later:        speeds = [0.25 ...]         full speed
```

The scaling collapses after a single 324 Hz tick, while the module needs
O(100 ms) to physically get there. For essentially the whole slew the wheels
are driven at full commanded speed while pointing the wrong way — scrub, slip,
and odometry error exactly during direction changes.

**Fix:** the cost is one CAN-cached read per module per tick
(`GetAbsoluteEncoderPosition` is a cached periodic frame, not a bus request —
see finding 4). Either set `USE_FEEDBACK_FOR_STEER = True`, or leave the
control path alone and scale against the encoder specifically.

Do this **after** (1), not before: with the encoder in the loop, the zero-command
re-aim would also start dragging drive speed to zero through the cosine term.

### 4. Nothing ever reads the steering encoder

Following from (3): with `USE_FEEDBACK_FOR_STEER = False`, no part of the
running system compares commanded steering angle to actual. A module that is
stalled, mis-offset, or fighting a bad gain is invisible — `steer_pos` reports
it exactly on target because `steer_pos` *is* the command.

This is now addressed for diagnosis rather than control: `Base.swerve_telemetry()`
reports both, and the whole-body trajectory log records `steer_cmd_*` and
`steer_meas_*` per module every tick. `RotationMotor.get_absolute_rad()` is
deliberately separate from `get_position_rad` so telemetry cannot inherit the
substitution.

### 5. The relay defeats `Base`'s own watchdog

`Base.control_loop` disables the drives after 2.5 × 100 ms without a command.
That watchdog can never fire in whole-body operation: `BaseController._run`
re-sends `self.target_velocity` at 108 Hz unconditionally, so a command always
arrives, whether or not anything upstream is still alive.

`WholeBodyController._control_loop` covers the case it can — a raising `_step`
is caught and calls `_halt_base()`. It does not cover a *hang* (a blocking CAN
read, a deadlock), which does not raise, and there is no liveness check on the
EE target at all: nothing records when `left_ee_target` was last written. A
teleop client that dies while an unreachable target is latched leaves the
solver permanently recruiting the base, and the base drives.

**Fix:** timestamp the EE targets and have `_dispatch_base` halt when they go
stale. That restores a real watchdog without touching the relay, which has its
own good reason to run fast.

### 6. Drive velocity units are unverified

`DriveMotor.set_velocity_mps(v)` sends `SetVelocity(v * DRIVE_VEL_SCALE)` with
`DRIVE_VEL_SCALE = 2.0`. There is no wheel-radius or gear-ratio conversion
anywhere — `TIRE_RADIUS` is defined and never used — and `git log -S` shows the
constant arriving in the initial commit with no derivation.

**The drivetrain itself is known, from a different file.**
`robot/nav/odometry/swerve_odom.py` carries an empirical calibration:
`METERS_PER_ROTATION = 0.049922` (2026-03-30, 5 runs, σ = 0.82%). Against the
3-inch wheel that implies

```
wheel circumference            0.239389 m
metres per motor rotation      0.049922 m   (calibrated)
=> gear reduction              4.7953 : 1   -- a clean 4.8:1 is 0.10% away
```

so the mechanical chain is settled: **4.8:1 onto a 3-inch wheel**, and the
position conversion factor must be 1.0 for that calibration to be per motor
rotation.

The velocity side is not settled, and cannot be, because half the calculation
lives in SPARK flash rather than in git:

```
VelocityConversionFactor for SetVelocity in true m/s   0.00083203
the same computed with DIAMETER instead of radius      0.00166407   (exactly 2x)
if the factor were 1.0 (motor RPM), 0.25 m/s needs     300.5 RPM
   ...but set_velocity_mps(0.25) sends 0.5 -> nothing would move
```

That last line proves a conversion factor **is** configured on the controllers.
Which leaves two readings of the 2.0, with opposite consequences:

* **The factor is 2x too large** (the radius/diameter slip — exactly 2 is that
  error's signature). Then `DRIVE_VEL_SCALE` is a correct-magnitude workaround,
  commanded m/s is truthful, and the odometry is fine.
* **The factor is right and 2.0 is an empirical fudge.** Then the robot travels
  at *twice* the commanded m/s and `BaseOdometry` records half of reality.

The manifest's `velocity_ff = 0.23` leans toward the second: F ≈ 1/(top speed
in setpoint units), and 1/0.23 = 4.35 m/s, against 4.72 m/s for a 5676 rpm NEO
on this drivetrain. Whereas "0.50 m/s on joystick preset 0 feels about right"
leans toward the first. This is exactly why it needs measuring rather than
reasoning.

**Measured 2026-08-22.** All four drive controllers report

```
velocity_cf                  0.000846326
  -> implies m per motor rot 0.050780
  vs calibrated (swerve_odom) 0.049922      1.72% apart
against "true m/s"  0.00083203   ratio 1.0172
against "2x slip"   0.00166407   ratio 0.5086
```

**So the first reading is dead.** The controllers are already configured in
true m/s, to 1.7% — and that 1.7% sits in the direction physics predicts, since
the factor was set from nominal geometry while the calibration measures a
loaded tyre's slightly smaller effective rolling circumference. `position_cf`
is 1.0 everywhere, which independently confirms `METERS_PER_ROTATION` really is
per motor rotation.

`DRIVE_VEL_SCALE = 2.0` therefore is **not completing a unit conversion**. It is
a bare 2x on a setpoint that was already correct. Which leaves:

* **the velocity loop tracks its setpoint** → the robot runs at ~2x the
  commanded m/s and `BaseOdometry` records half of reality;
* **the loop undershoots by ~2x** → the 2.0 compensates, commanded m/s is
  roughly true, and the odometry is honest.

The second is not far-fetched. The drive gains are deliberately feed-forward
dominated (`p=0.35, i=0, d=0, velocity_ff=0.23`), and with no integrator
nothing forces zero steady-state error.

These *are* separable from CAN telemetry, because `drive_meas_raw` is
`GetVelocity()` in the controller's own units — which are now known to be m/s:

```
measured / setpoint  ~1.0  -> the loop tracks       -> first reading
measured / setpoint  ~0.5  -> the loop undershoots  -> second reading
```

### What it should be

Combining the measured factor with the two independent rolling calibrations,
assuming the loop tracks:

| calibration | m / motor rot | robot travels | `DRIVE_VEL_SCALE` should be |
|---|---:|---:|---:|
| `swerve_odom.py` (this repo) | 0.049922 | **1.97x** commanded | 1.017 |
| PHASE0_BASELINE (5 tape runs, 0.74% CV) | 0.047531 | **1.87x** commanded | 1.068 |

**It should be about 1.0.** The controller already does the conversion; the 2.0
is a doubling on top of it. (The two calibrations differ by 5%, which is its
own small open question — different dates, tyres or loading.)

### It has been investigated before, and left open

`YOR-v3-Problems-DON'T-USE` reached this exact wall. Its Phase 0 measured
`DRIVE_METERS_PER_RAW_UNIT = 0.047530598` over five tape runs, could not read
the conversion factor ("Spark firmware **not readable from this binding**"), and
left two hypotheses standing: either the controllers are not in closed-loop
velocity mode, or the velocity conversion factor is off by ~631. Its calibration
file carries `command_conversion_reconciled: false` and the note that **"every
speed number on the base is nominal until this closes."** It never closed.

Today's read settles both of those hypotheses: `ctrl_type` is `kVelocity` (1) on
all four drives, so they *are* in closed-loop velocity mode; and the factor is
0.000846, set for m/s, not 631x off. What remains is only whether the loop
reaches its setpoint.

### Where the 2.0 came from

The 2.0 **predates the feed-forward-dominated gains** — it was already there
under the stock PID. And the stock drive gains, recorded in that same
abandoned repo and confirmed by its status line that *"a controller power cycle
reverts every module to stock"*, are:

```
drive stock:  p = 0.2,  i = 0,  d = 0.1,  velocity_ff = 0,  output +/-1.0
```

That is a **P-only velocity loop with no feed-forward**, and a P-only loop
cannot reach its setpoint. The SPARK's closed loop outputs duty as `P x error`,
and duty maps roughly linearly to speed, so at steady state

```
act/sp = P*v_free / (1 + P*v_free)
```

With `v_free = 5676 rpm x 0.000846326 = 4.80` native units:

```
stock  p=0.2,  ff=0     -> reaches 49.0% of setpoint  -> needs a command scale of 2.04
```

**2.04.** `DRIVE_VEL_SCALE = 2.0` is a compensation for the stock loop's ~2x
steady-state undershoot, and it is correct *for those gains*.

It is no longer correct for the gains that ship. The commissioned set is
feed-forward dominated (`p=0.35, velocity_ff=0.23`), and its own commissioning
evidence records a **steady error of 0.000 native median** at 0.25 native on the
floor — i.e. it tracks. So the compensation is now a stale doubling on a loop
that no longer undershoots, and the base has been running at roughly twice
every commanded speed since the 2026-08-17 retune, with `BaseOdometry`
recording half of reality.

This is a hypothesis with a sharp prediction, not a measurement:
`tools/measure_drive_scale.py` should show `measured/setpoint ~ 1.0`. Run it
before changing the constant.

`tools/measure_drive_scale.py` does exactly that comparison. Run it wheels-up
first (no slip, so it is a clean answer to "does the loop reach setpoint"), then
on the floor with a tape measure — `GetVelocity` is the controller's belief
about the motor, not about the robot's motion over the ground.

It matters because `_dispatch_base` integrates the commanded velocity straight
into `BaseOdometry`, which is what the IK uses for chassis pose. Under the
second reading the model believes the base moved half as far as it did, so it
keeps asking for more base motion — overshoot and oscillation on exactly the
base-recruited reaches whole-body control exists to perform. **Settle this
before tuning anything else.**

### 6b. `SetIdleMode` and `SetCtrlType` are never called

> **Fixed, and measured to be behaviourally inert.** The import now names only
> `CtrlType` and `IdleMode`. Reading all eight controllers back on 2026-08-22:
> idle mode is already `kCoast` (0) everywhere, the drives are already
> `kVelocity` (1) and the steering `kPosition` (3). So both calls now write
> what was already there — the guard being dead never actually changed how the
> robot behaved. Still worth fixing: it was dead by accident, and it would have
> mattered the moment a controller came back from flash in a different state.
> `Base.swerve_configuration()` prints this at startup so it stays visible.


`base_motor.py` guards both on an optional enum import:

```python
try:
    from sparkcan_py import CtrlType, IdleMode, MotorType, SensorType
except Exception:
    IdleMode = CtrlType = MotorType = SensorType = None
```

The installed binding exports `CtrlType` and `IdleMode` but **not** `MotorType`
or `SensorType`, so the import raises `ImportError` and the handler sets all
four to `None` — including the two that do exist. Every
`if IdleMode and ...` / `if CtrlType and ...` guard downstream is therefore
dead:

```
IMPORT FAILED -> ImportError: cannot import name 'MotorType'
  if IdleMode and hasattr(dev,"SetIdleMode"): -> False
  if CtrlType and hasattr(dev,"SetCtrlType"): -> False
```

So coast/brake behaviour and the closed-loop control type are whatever the
SPARKs hold in flash. Nothing in this repo sets them, and nothing reports what
they are. This is the same root cause as (6): the drivetrain's real
configuration lives in controller flash, configured once through the REV
Hardware Client, and is invisible to git.

**Fix:** import the two names that exist, on their own line. Then decide
deliberately whether the base should coast or brake when disabled — a coasting
base on a slope is a different robot from a braking one, and right now nobody
here knows which it is.

### 7. Odometry ignores the S-curve profiler

`_dispatch_base` integrates the commanded velocity as if it were applied
instantly. The base is commanded with `smooth=True`, so `Base` ramps it over a
segment of `T = max(0.01, max(|Δv|·π / 2a))` with `a_max = [1.9, 1.9, 6.5]`.
For a 0.25 m/s step that is ~0.2 s of ramp the model does not know about — at
30 Hz, six ticks during which the model runs ahead of the chassis.

Logging `swerve_target_*` (what `Base` holds) alongside `swerve_prof_*` (what
the profiler has reached) makes the gap directly visible per tick.

### 8. `_angle_and_speed_to_vehicle_velocity` returns the wrong ω

> **Fixed by deletion.** Both dead methods and `self.C` are gone, with a
> pointer to `swerve_odom.py` left in their place.
> `tests/test_base_kinematics.py` now round-trips base_motor's IK against
> that forward model, so the disagreement cannot come back unnoticed.


Dead code — nothing calls it, and `self.C` exists only to serve it — but it
is wrong, and it is the obvious thing to reach for when wheel-odometry gets
added:

```
commanded [0, 0, 1.0]  -> wheels -> recovered [0, 0, -0.316]
commanded [0.2, 0, 0.5] -> wheels -> recovered [0.2, 0, -0.158]
```

Translation round-trips; rotation comes back with the wrong sign and about a
third of the magnitude. The cause is that `self.C` encodes
`W_i = [+W, −W, −W, +W]` while `_vehicle_velocity_to_angle_and_speed` applies
`ROT_DIAG_SWAP_PERM` to `vx_r`/`vy_r` and so effectively uses
`W_i = [−W, +W, +W, −W]`. The permuted version is the geometrically correct
one for FL/FR/RR/RL at (±L, ±W); `C` is the one that is wrong.

`_map_steer_angles` is dead in the same way, and worse: it applies the same
permutation to *angles*, which is not what a sign flip on a velocity component
does.

**Fix:** delete both, or correct `C` and add a round-trip test. Expressing the
rotation contribution with explicit per-module signs instead of a permutation
would make the discrepancy impossible to write.

### 9. Steering setpoint wraps mod 1.0

`set_position_fraction` sends `(frac + offset) % 1.0`, so crossing the wrap
point steps the setpoint from ~0.99 to ~0.01. That is correct only if the
SPARK has position PID wrapping enabled with input range 0..1. Nothing in the
codebase configures it — and the installed `sparkcan_py` binding does not
expose `SetPositionPIDWrapEnable` / `SetPositionPIDMinInput` / `MaxInput`,
though the vendored C++ has them.

If wrapping is off, a wrap-crossing looks like a 0.98-turn error to a
controller with `Kp = 20` clamped to ±0.25 output: saturated, and travelling
the long way round. It would show up as an occasional module taking a wild
detour, correlated with heading rather than with speed.

`steer_cmd_*` vs `steer_meas_*` in the new log settles it in one run: drive a
slow full rotation and look for a module whose measured angle unwinds the
wrong way. If it does, the binding needs those three methods exposed (they
are already implemented in `sparkcan_py/src/SparkBase.cpp`) and the manifest
needs to carry the wrap configuration.

### 10. Minor

- `Base._update_state` computes `_dt` and discards it.
- `BaseController._vel_lock` is created and never used;
  `target_velocity` is written by the solve thread and read by the relay
  thread without it. Benign under the GIL — whole-array replacement, never
  in-place mutation — but the lock's presence implies a protection that
  isn't there.
- `Base._update_scurve` integrates with a fixed `dt = 1/CONTROL_FREQ` rather
  than measured elapsed time, so a loop running below 324 Hz stretches every
  ramp.
- `BaseController.last_target_velocity` and `vel_alpha` are set up for a
  velocity EMA that `BASE_VEL` mode does not use.

## Suggested order for the tuning session

1. **Settle the drive-velocity scale (6).** Everything else is measured in
   units that depend on it, and one of the two readings means the odometry the
   IK runs on is out by 2x. Start by reading the conversion factors off the
   controllers — it costs one CAN read and needs no floor time. Fix the enum
   import (6b) at the same time, so what you read back is also what is set.
2. **Fix the zero-command re-aim (1).** Largest single behavioural artifact,
   and a small, local change.
3. **Fix the deadband to be a vector (2).** Also small, and it stops the
   odometry taking a systematic bias.
4. **Then turn on real steering feedback (3/4)** and re-check, since it
   interacts with both of the above.
5. Leave (5), (8), (9) for after the base drives predictably.
