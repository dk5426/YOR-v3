# Staged Hardware Validation — Whole-Body Teleoperation

Phase 8 of [WHOLEBODY_TELEOP_IMPROVEMENT_PLAN.md](WHOLEBODY_TELEOP_IMPROVEMENT_PLAN.md).

**Status: not executed.** Everything in phases 1–7 is implemented and tested in
software; nothing below has been run, because it moves the robot. This document
is the procedure, written against the commands that exist in the tree, for
whoever runs it.

Two rules apply throughout:

- Do not start with every subsystem enabled. Each stage exists so that a fault
  found there cannot be caused by the stages after it.
- Any unexpected motion ends the session. Fix the cause before continuing —
  a stage that "mostly worked" has not passed.

Have the emergency stop within reach for every stage from 4 onward.

## What software validation already covers

These are green, so a failure below is a hardware, wiring or configuration
fault rather than a logic error:

```bash
python tests/test_interface_contract.py     # the RPC and serial interfaces are unchanged
python tests/test_api_parity.py             # sim and hardware still serve the same calls
python tests/test_arm_config.py             # arm limits and gripper state
python tests/test_base_pid_preflight.py     # the manifest and the apply/verify logic
python tests/test_wholebody_control.py      # the whole-body loop against fake hardware
python tests/test_lift_velocity.py          # the lift PD, transport and refusals
python tests/test_lift_firmware.py          # the sketch compiles, and behaves
python tests/test_sim_node.py               # the simulation node
```

## 1. Emergency stop and direct stop

Before anything moves under its own control, prove it can be stopped.

- Physically verify the emergency stop cuts power where you expect.
- With `robot/yor.py` running and nothing else commanded, call
  `emergency_stop()` over RPC and confirm it returns.
- Confirm `lift_stop()` reaches the controller: the console shows
  `[PicoLift] sent 'stop'` and the firmware answers `Motion stopped by user.`

**Pass:** both stop paths are confirmed reachable and take effect.

## 2. Each arm alone

Base and lift disabled — run `robot/yor.py` and immediately
`toggle_base_motion(False)`, or start with `no_arms=False` and fix the base.

- Home each arm (`python robot/arm/arm.py` drives one arm at a time).
- Confirm the console prints `no gripper fitted — gripper control disabled`
  for both arms, and that no gripper actuator is commanded.
- Nudge one hand a few centimetres with the teleop client; the other arm and
  the lift should stay still.
- Confirm joint motion is bounded: the whole-body clamp and the native limit
  are both 3.0 rad/s, and the acceleration limit is 15.0 rad/s².

**Pass:** each arm tracks its own target, neither arm moves when the other is
commanded, and nothing tries to drive a gripper.

## 3. Home the lift

```python
yor.lift_home()
yor.lift_position_known()   # must become True
yor.get_lift_height()       # must become a number, not None
```

The firmware has no zero until it has seen the upper switch. Until then every
height it reports is meaningless, and both the PD and `lift_to_height()`
refuse to move.

**Pass:** `lift_position_known()` is True and the height spans 0 → 0.900 m.

## 4. Lift velocity, small then larger

Confirm the firmware is the new one first:

```python
yor.get_lift_status()["velocity_capable"]   # must be True
yor.get_lift_status()["capabilities"]       # ['lift_velocity_v1']
```

If that is False, the controller has the old sketch: flash
`firmware/lift_controller/` before continuing, or the whole-body loop will
silently use the bang-bang fallback.

Stages 4, 5 and 6 are implemented as interlocked hardware tests. Run them
rather than improvising:

```bash
python tests/hardware/test_02_lift.py --host <robot-ip>
```

It covers homing, the discrete moves, then the velocity stages in increasing
order of consequence: capability, ±5 mm/s, ±10 mm/s, zero hold, reversal,
command timeout, limit switches, and finally the travel constant. Every motion
stage waits for a typed confirmation and halts the lift on Ctrl-C.

To drive it by hand instead, the RPC surface is:

```python
yor.lift_supports_velocity()         # must be True before any of this means anything
yor.lift_set_velocity(0.005)         # +5 mm/s — re-send at least every 100 ms
yor.lift_set_velocity(0.0)
yor.lift_set_velocity(-0.005)        # -5 mm/s
```

The firmware stops by itself 300 ms after the last command, so a single call
produces a short move and then a stop. That is the contract, not a bug.

**Pass:** the column moves at the commanded speed in the commanded direction,
smoothly, at both magnitudes and both signs.

## 5. Velocity-mode safety behaviour

Covered by the same test file; verified separately here for reference:

| Check | How | Expected |
|---|---|---|
| Zero hold | Stream `0.0` for a few seconds | Column still, height telemetry keeps arriving, driver stays powered |
| Idle exit | Keep streaming `0.0` past 5 s | `Velocity idle; exiting velocity mode.`, relay opens |
| Reversal | Go from +10 mm/s to −10 mm/s in one command | Ramps through zero, *then* reverses; no jolt, no missed steps |
| Command timeout | Stop sending while moving | `Velocity command timeout; ramping to zero.` within 300 ms, then a smooth stop and `Velocity stopped: no command for 300 ms.` |
| Upper limit | Drive up onto the switch | `LIMIT HIT: upper limit.`, relay opens, height pinned to 900 mm |
| Lower limit | Drive down onto the switch | `LIMIT HIT: lower limit.`, relay opens, height pinned to 0 |
| Software travel | Drive to a travel end with the switch bypassed *only if you can do so safely* | `LIMIT HIT: software travel limit.` |
| Blocked direction | Ask to drive further into a closed switch | `UP blocked` / `DOWN blocked` once, no motion, and driving away still works |

**Pass:** every row behaves as described. The timeout row is the important
one — it is the only thing that stops the column if the host dies mid-move.

## 6. The discrete lift moves still work

The velocity mode is an addition; the old paths must be unchanged.

```python
yor.lift_up(); yor.lift_stop()
yor.lift_down(); yor.lift_stop()
yor.lift_to_height(0.400)        # profiled finite move
yor.lift_delta_height(-0.050)
```

**Pass:** the direction-specific smooth profiles are intact — an upward move
still starts around 35 mm/s and reaches 60 mm/s, a downward one 30 → 55 mm/s —
and `lift_to_height()` arrives within its tolerance.

## 7. Swerve gains, wheels unloaded

Wheels off the ground, or the robot on blocks.

```bash
python tools/base_pid_preflight.py --dry-run    # confirm the plan first
python tools/base_pid_preflight.py              # apply and read back
```

Every controller must print `OK`. A `MISMATCH` means that controller did not
take the value; do not proceed with a mixed set of gains.

This stage uses the standalone command because the robot node is not running.
In normal operation `robot/yor.py` does the same thing at startup, so a power
cycle between stages does not leave a module on stock gains.

**Pass:** all eight controllers report the commissioned values.

## 8. Steering at the commissioned limit

Still unloaded. Command small steering-only motions and confirm the modules
track without hunting or buzzing.

The output limit is ±0.25 and stays there for this run. The proposed
full-range combination (`Kp=10`, `Kd=0`, output ±1.0) is **not** validated and
is a separate commissioning experiment with its own approval record.

**Pass:** all four modules reach their commanded angle without oscillation.

## 9. Low drive commands and feed-forward

The drive loop is feed-forward dominated (`velocity_ff = 0.23`, `Kp = 0.35`,
`Kd = 0`). Command small wheel velocities and confirm the response is smooth
and roughly proportional, with no torque ripple on deceleration.

**Pass:** low-speed drive is smooth in both directions.

## 10. Whole-body with the base fixed

Wheels back on the ground, base locked:

```bash
python robot/yor.py    # syncs the swerve gains itself before the control loop
python robot/teleop/wholebody_teleop.py --target hw --host <robot-ip> --input keyboard
```

Press `t` (fix-base) immediately, or call `toggle_base_motion(False)`.

- Move each hand; the arms should track and the chassis must not move.
- Use `r` / `f` to raise and lower the lift. The torso moves while the hands
  hold station — the arms compensate.
- Check `get_state()`: `lift_velocity_mode` must be `True`, and
  `lift_feedback_age_s` must stay small while the lift is moving.

**Pass:** arms and lift work together under the solver with the base locked.

## 11. Base participation

Clear floor space, then release fix-base and push a target beyond arm reach.

The chassis should roll *slowly* — clamped to 0.25 m/s and 0.6 rad/s — in the
direction of the target. If it drives the wrong way, fix the signs in
`BaseAxisMap` in `robot/wholebody_control.py` rather than compensating
elsewhere.

**Pass:** the base moves only when the arms and lift cannot reach, and it moves
the right way.

## 12. The unchanged client

Finally, confirm the thing this whole exercise was meant to protect: the
existing server and the existing client, with no interface changes.

```bash
python robot/yor.py
python robot/teleop/wholebody_teleop.py --target hw --host <robot-ip> --input oculus
```

**Pass:** the client connects and drives the robot using exactly the calls it
used before — `set_lift_target()` still takes a height in metres, and nothing
in the client knows how the lift, the arms or the swerve gains are implemented.

## Completion

The integration is complete only when every stage above has passed, in order,
and:

- arm commands are bounded by the agreed limits;
- no absent gripper is enabled;
- every swerve controller passed gain readback before any motion;
- lift position targets produce bounded velocity commands;
- the firmware profiles those commands and stops safely on command loss;
- the old lift commands still work;
- stale or unknown feedback prevents lift motion;
- all non-hardware tests still pass.
