# Whole-Body Teleoperation Improvement Plan

## Purpose

Prepare the existing `YOR-v3` whole-body teleoperation system to run with the
agreed arm limits, commissioned swerve gains, and position-to-velocity lift
control.

This is a planning document only. It does not authorize or implement any code,
configuration, firmware, controller-RAM, or hardware changes.

## Architectural constraint

The current `YOR-v3` repository is the only architectural baseline.

- Keep the existing `YOR`, `BaseController`, `Base`, `PicoLift`,
  `WholeBodyController`, and `wholebody_teleop.py` structure.
- Keep all existing public teleoperation and RPC method names.
- Do not import, copy, or adapt runtime, lifecycle, coordinator, navigation,
  safety, or control architecture from `YOR-v3-Problems-DON'T-USE`.
- Do not replace complete `YOR-v3` files with files from another repository.
- Implement each approved change directly and minimally inside the component
  that already owns that responsibility in `YOR-v3`.
- Keep unrelated behavior unchanged.

## Target behavior

The completed system should retain the existing teleoperation flow:

```text
wholebody_teleop.py
        |
        | existing YOR commands and RPC interface
        v
YOR
        v
WholeBodyController
        |
        +---- arm joint targets ----> ArmNode
        |
        +---- lift position target -> host PD -> Arduino velocity profile
        |
        +---- base velocity --------> BaseController / Base
```

The teleoperation client should not need to know how arm limits, lift velocity
profiling, or swerve PID gains are implemented.

## Agreed settings

### Arms

| Setting | Target |
| --- | ---: |
| Native joint velocity limit | 3.0 rad/s |
| Native joint acceleration limit | 15.0 rad/s^2 |
| Whole-body joint-command limit | 3.0 rad/s |
| Native gripper | Disabled when no gripper is fitted |

The native arm limits are already present in the current `YOR-v3` arm
configuration. The implementation phase should verify them and change only the
whole-body limit or explicit gripper setting if they are still missing. Arm PD
gains, home positions, gravity compensation, and lifecycle behavior are outside
this plan.

### Swerve drive

Use the commissioned settings for the initial integrated run:

| Motor role | Kp | Ki | Kd | Velocity FF | Output range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Drive | 0.35 | 0.0 | 0.0 | 0.23 | -1.0 to 1.0 |
| Steering | 20.0 | 0.0 | 6.0 | 0.0 | -0.25 to 0.25 |

The proposed full-range steering combination (`Kp=10`, `Kd=0`, output
`-1.0 to 1.0`) is not validated. It must remain a separate commissioning
experiment and must not be introduced during the initial whole-body integration.

### Lift

| Setting | Target |
| --- | ---: |
| Host Kp | 2.0 1/s |
| Host Kd | 0.05 s |
| Derivative filter time constant | 0.1 s |
| Position deadband | 0.005 m |
| Maximum requested velocity | 0.05 m/s |
| Arduino maximum velocity | 50 mm/s |
| Arduino minimum active velocity | 0.5 mm/s |
| Arduino acceleration limit | 200 mm/s^2 |
| Arduino jerk limit | 2000 mm/s^3 |
| Arduino velocity timeout | 300 ms |

## Phase 1: protect the existing interfaces

Before implementation, record the current callable interface between:

- `wholebody_teleop.py` and `YOR`;
- `YOR` and `WholeBodyController`;
- `WholeBodyController` and the arms, lift, and base;
- `PicoLift` and the Arduino serial protocol.

Add regression tests around those existing interfaces before altering internal
behavior. The acceptance condition is that existing teleoperation commands and
RPC calls retain their current names, arguments, return values, and target
semantics.

## Phase 2: arm settings

1. Verify that both arms receive the 3.0 rad/s velocity and 15.0 rad/s^2
   acceleration limits.
2. Raise the existing whole-body per-joint command clamp to 3.0 rad/s so it does
   not silently reduce the configured native limit.
3. Explicitly disable native gripper control when no physical native gripper is
   installed.
4. Do not modify arm gains, home positions, gravity compensation, startup,
   homing, or command APIs.
5. Add a configuration-level test proving the final limits and gripper state.

## Phase 3: swerve PID preflight

Keep gain application outside the base-control architecture.

1. Store the commissioned values in a small `YOR-v3` configuration manifest.
2. Add a standalone preflight command that runs before the YOR hardware process
   opens the motor controllers.
3. Have the preflight verify the CAN interface, module CAN IDs, PID slot, and
   requested values before writing anything.
4. Apply gains to controller RAM only.
5. Read all fields back and fail the preflight if any controller differs.
6. Never run a second set of SparkFlex device objects while the existing base
   process owns the CAN devices.
7. Require this preflight after every controller power cycle because RAM values
   are not persistent.

The initial whole-body run must use the steering `-0.25 to 0.25` output range.
Full steering output requires a later, independent tuning and approval record.

## Phase 4: Arduino lift velocity mode

Extend the current `YOR-v3` Arduino sketch without replacing its structure or
removing its existing motion modes.

Preserve:

- `up`, `down`, `stop`, and `home`;
- finite-distance moves;
- the existing direction-specific smooth profiles;
- pulse-counted height;
- upper and lower hardware limit checks;
- software travel limits;
- status, telemetry, relay, and power behavior.

Add one serial command:

```text
vel <signed mm/s>
```

For this mode, the Arduino should:

1. Clamp the request to +/-50 mm/s.
2. Treat magnitudes below 0.5 mm/s as a zero-velocity hold.
3. Plan a quintic minimum-jerk transition from its current commanded velocity
   to the requested velocity.
4. Respect the configured acceleration and jerk limits.
5. Convert ramped velocity magnitude into Timer1 step frequency.
6. Ramp through zero before changing the direction pin.
7. Continue height telemetry while velocity mode is holding at zero.
8. Ramp to zero if no valid velocity command arrives for 300 ms.
9. Retain independent hardware and software limit enforcement.
10. Advertise an explicit protocol capability such as
    `Capabilities: lift_velocity_v1` at startup or in a status response.

The Arduino performs command profiling and step generation. It is not a
closed-loop velocity controller because it has no measured column-velocity
feedback.

## Phase 5: existing PicoLift transport

Extend the current `PicoLift` class in place:

1. Parse and retain the Arduino velocity capability.
2. Record the receive time of height telemetry so stale measurements can be
   rejected.
3. Add a signed velocity send method using millimetres per second on the wire.
4. Send meaningful command changes immediately.
5. Send unchanged commands at least every 100 ms as a keepalive.
6. Send zero immediately when transitioning from motion to hold.
7. Reject non-finite values before serial transmission.
8. Preserve all existing discrete lift methods and serial parsing.

Expose the velocity method through the existing `Base` object without adding a
new lift-control subsystem or coordinator.

## Phase 6: lift PD inside WholeBodyController

Keep `set_lift_target()` as a position command. The teleoperation client should
continue sending lift positions exactly as it does today.

Inside the existing `WholeBodyController`, calculate:

```text
error = desired_height - measured_height
velocity = Kp * error - Kd * filtered_measured_velocity
```

Implementation requirements:

1. Use the solver's lift position unless an explicit operator lift target is
   active.
2. Clamp the desired height to the existing model range.
3. Differentiate measured height rather than position error to avoid derivative
   kick when the target changes.
4. Low-pass filter the measured derivative.
5. Command exactly zero inside the 5 mm position deadband.
6. Clamp output to +/-0.05 m/s.
7. Reset derivative state after stale feedback, manual override, disarm, or a
   control-loop gap.
8. Refuse motion when height is unknown or stale.
9. Use streamed velocity only when the Arduino explicitly reports support.
10. Retain the current up/down/stop lift behavior as the old-firmware fallback.

Do not add or import a different runtime, ownership model, or lift coordinator.

## Phase 7: software validation

Run validation without allowing hardware motion:

1. Test arm limits and the whole-body command clamp.
2. Test lift proportional response, measurement derivative, filtering,
   deadband, velocity clamp, and reset behavior.
3. Test serial formatting, unit conversion, keepalive, zero command, invalid
   input, and capability detection.
4. Test fallback behavior when velocity capability is absent.
5. Test that stale or unknown lift height prevents motion.
6. Compile the Arduino sketch.
7. Run all existing headless and whole-body regression tests.
8. Confirm that simulation and hardware clients still use the same teleoperation
   methods.
9. Review the final diff and reject changes unrelated to arms, lift velocity,
   swerve PID preflight, tests, or documentation.

## Phase 8: staged hardware validation

Do not begin with every subsystem enabled.

1. Verify emergency-stop and direct stop behavior.
2. Test each arm independently with the base and lift disabled.
3. Home the lift and verify that position becomes known.
4. Test lift velocity at +/-5 mm/s, then +/-10 mm/s.
5. Verify zero hold, reversal through zero, 300 ms command timeout, both limit
   switches, and software travel limits.
6. Verify that existing discrete lift moves still have their original smooth
   behavior.
7. Apply and read back the commissioned swerve gains with the wheels unloaded.
8. Test steering at the commissioned +/-0.25 output limit.
9. Test low drive commands and confirm drive feed-forward behavior.
10. Run whole-body control with the base fixed, validating arms and lift first.
11. Enable base participation with small targets and a clear workspace.
12. Start the existing hardware server and existing `wholebody_teleop.py`
    client without changing their public interface.

## Completion criteria

The integration is complete only when:

- the current `wholebody_teleop.py` works without interface changes;
- arm commands are bounded by the agreed limits;
- no absent native gripper is enabled;
- every swerve controller passes gain readback before motion;
- lift position targets produce bounded host velocity commands;
- the Arduino profiles those commands and stops safely on command loss;
- old lift commands still work;
- stale or unknown feedback prevents lift motion;
- all non-hardware tests pass;
- staged hardware tests pass before unrestricted whole-body operation.
