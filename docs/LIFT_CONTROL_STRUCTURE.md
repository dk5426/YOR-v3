# Lift Control Structure and Arduino Firmware Differences

## Verified control structure

The lift velocity-control path described below was verified against the code in
`YOR-v3-Problems-DON'T-USE`. The overall description is correct:

```text
Whole-body IK
    |
    | desired lift position (m)
    v
WholeBodyController lift PD
    |
    | desired lift velocity (m/s)
    v
LiftCoordinator
    |
    | admitted, bounded, expiring velocity request
    v
PicoLift serial driver
    |
    | "vel <signed mm/s>"
    v
Arduino lift firmware
    |
    | minimum-jerk velocity ramp -> direction and step frequency
    v
Lift stepper driver and column
    |
    | pulse-counted height telemetry (36 Hz)
    +-------------------------------------------------> feedback to host
```

The whole-body solver produces a desired lift position. Normally this is
`result.lift_q`; an explicit operator lift target takes precedence when one is
present. The target is clamped to the lift's valid height range, and the
controller refuses to move from an unknown or stale measured height.

The host then converts position into velocity with `LiftVelocityPD`:

```text
position_error = desired_height - measured_height
velocity_command = Kp * position_error - Kd * filtered_measured_velocity
```

The derivative is taken from measured height rather than position error. This
avoids a derivative kick when the desired position changes suddenly. It is
low-pass filtered because Arduino height telemetry (a fixed ~36 Hz hardware
rate) and the whole-body loop (now 30 Hz, previously 108 Hz) don't arrive in
lockstep.

The verified default host settings are:

| Setting | Value |
| --- | ---: |
| Lift Kp | 2.0 1/s |
| Lift Kd | 0.05 s |
| Derivative filter time constant | 0.1 s |
| Position deadband | 0.005 m |
| Maximum commanded velocity | 0.05 m/s |
| Whole-body request TTL | 0.5 s |

Inside the 5 mm position deadband, the host commands zero velocity. Outside the
deadband, the result is clamped to +/-0.05 m/s. The `LiftCoordinator` owns lift
authority, rejects conflicting or unsafe commands, clamps the velocity again,
and expires a whole-body request if it is not refreshed.

The serial driver converts metres per second to millimetres per second and
sends:

```text
vel <signed mm/s>
```

Positive velocity moves up and negative velocity moves down. Commands that
change by less than 0.5 mm/s are rate-limited, but the driver sends a keepalive
at least every 100 ms. This is comfortably inside the Arduino's 300 ms command
timeout.

## What the Arduino controls

The Arduino receives a requested velocity; it does not apply that value as an
instantaneous step. It plans a quintic minimum-jerk transition from its current
commanded velocity to the new requested velocity. The ramp is recalculated when
the target changes and is applied at 324 Hz.

The velocity-mode settings are:

| Setting | Value |
| --- | ---: |
| Maximum velocity | 50 mm/s |
| Minimum non-zero velocity | 0.5 mm/s |
| Maximum acceleration | 200 mm/s^2 |
| Maximum jerk | 2000 mm/s^3 |
| Ramp duration range | 40-2000 ms |
| Velocity command timeout | 300 ms |
| Zero-velocity idle exit | 5000 ms |
| Step conversion | 320 pulses/mm |

The Arduino converts the magnitude of the ramped velocity into Timer1 step
frequency and uses the direction pin for its sign. A direction reversal first
ramps to zero, stops the pulse train, changes direction, and then ramps up in the
opposite direction. Timer frequency changes are committed at a timer overflow
to avoid dropped steps or audible hitches from changing Timer1 TOP mid-cycle.

This is trajectory shaping, not closed-loop velocity feedback: the Arduino does
not measure motor or column velocity and correct velocity error. It assumes the
stepper follows the generated pulses. Its position estimate is obtained by
counting pulses after homing establishes a known endpoint.

The firmware independently enforces the upper and lower limit switches, the
0-900 mm software travel range after homing, the 50 mm/s velocity clamp, and the
300 ms serial-command timeout. If velocity commands stop arriving, it ramps to
zero and disables the driver. A commanded zero is a temporary hold; after five
seconds at zero, velocity mode exits and the relay opens.

## Difference between the two Arduino sketches

The active sketches compared are:

- `YOR-v3/firmware/lift_controller/lift_controller.ino`
- `YOR-v3-Problems-DON'T-USE/firmware/lift_controller/lift_controller.ino`

The Problems sketch is not merely a parameter change. It adds the complete
streamed-velocity mode required by the whole-body PD path.

| Area | `YOR-v3` sketch | Problems sketch |
| --- | --- | --- |
| Whole-body velocity command | No `vel` command | Adds `vel <signed mm/s>` and `MOTION_VELOCITY` |
| Ordinary upward move | Quintic profile from 35 to 60 mm/s; 80 mm/s^2 acceleration and 200 mm/s^3 jerk | Fixed 50 mm/s |
| Ordinary downward move | Quintic profile from 30 to 55 mm/s; 60 mm/s^2 acceleration and 200 mm/s^3 jerk | Fixed 50 mm/s |
| Streamed-velocity shaping | Not available | Quintic ramp with 200 mm/s^2 acceleration and 2000 mm/s^3 jerk limits |
| Lost host command | No velocity stream to supervise | Ramps to zero after 300 ms without `vel` |
| Zero-velocity behavior | Not applicable | Holds with the driver active, then exits after 5 s |
| Direction reversal | A new discrete move stops the old move | Velocity ramp must pass through zero before DIR changes |
| Low-speed timer support | Timer prescaler fixed at 1 | Selects prescaler 1 or 8 |
| Timer frequency updates | Writes TOP immediately | Defers live updates to the overflow ISR |
| Height telemetry while holding | Only while a discrete move is active | Continues while velocity mode is active, including zero hold |

Both sketches retain the same physical scaling and core protection: 320
pulses/mm, 900 mm maximum travel, 1000 mm homing request, 35 mm/s homing speed,
hardware limit-switch checks, software travel limits after homing, pulse-counted
height, and the existing `up`, `down`, `stop`, `home`, distance, status, and
power commands.

One tradeoff is worth noting: discrete `up`, `down`, and distance moves are
direction-specific jerk-limited moves in the current `YOR-v3` sketch, but they
run at a fixed 50 mm/s in the Problems sketch. The Problems sketch adds smooth
velocity streaming while removing the newer smooth profile from the discrete
move path. A final merged firmware should retain both features.

## Compatibility caveat found during verification

The Problems host code asks `LiftCoordinator.supports_velocity()` before using
the velocity path. That check currently verifies that the Python port exposes a
callable `lift_set_velocity()` method; it does not perform a serial capability
handshake with the Arduino itself. Therefore, a current Problems Python driver
connected to the older `YOR-v3` sketch can be classified as velocity-capable
even though the sketch does not recognize `vel`.

The host and firmware velocity changes should consequently be deployed
together, or the protocol should gain an explicit firmware capability/version
response. Until then, the presence of the Python method alone does not prove
that the connected Arduino supports streamed velocity.
