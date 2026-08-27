"""
wholebody_control.py — whole-body control on the *real* YORv3 hardware.

This is the hardware counterpart of robot/yor_mujoco.py. Both run the same
solver (robot/arm/wholebody_ik.py) over the same description
(description/scene_wholebody.xml); they differ only in where the measured
configuration comes from and where the solution is dispatched:

                        simulation                  hardware
    measure     data.qpos (previous solve)   arm startup seed / previous command,
                                             lift height, base odometry
    dispatch    data.qpos + mj_forward       ArmNode.set_joint_target()   ×2
                                             PicoLift up/down/stop
                                             Base.set_target_base_velocity()

Control flow, once per cycle (default 30 Hz):

  1. read measured state and push it into the IK configuration
  2. solve whole-body IK for the current EE / lift targets
  3. dispatch arms, lift and base, each with its own clamp and its own
     enable flag
  4. integrate the base velocity that was actually commanded into the odometry
     estimate

Lift and base are dispatched directly from that 30 Hz loop. Arms are not: each
solved joint target is handed to a second, faster loop (default 90 Hz, see
`_arm_dispatch_tick`/`_arm_dispatch_loop`) that interpolates it against the
previous target over a few fixed sub-steps before sending anything to
nerolib. This reproduces YOR_D's own arm-commanding chain -- a Cartesian
target arriving no faster than 30 Hz, smoothed by frequent, short-duration
joint commands rather than one coarse command per solve tick.

Three things are worth knowing before running this on the robot:

* **Base odometry is dead-reckoned from the commanded velocity, and
  corrected by SLAM.** Integrating the commanded velocity is exact with
  respect to what the solver asked for and says nothing whatever about the
  floor: slip, a push, or a module that never reached its angle all move the
  robot without moving odometry. Used alone as the base PD's measurement --
  which is what `_dispatch_base` does with it -- that makes the loop an echo
  chamber, because the error decays when odometry moves whether or not the
  robot did.

  `enable_slam_base_pose` (**on** by default) closes that loop on the Odin
  VIO+lidar fix (`slam/pose`, from robot/odin_pub_node.py). Dead-reckoning
  still carries the estimate between fixes -- it is smooth and available at
  loop rate, which a 20 Hz absolute pose is not -- but every tick it is pulled
  toward the fix under a rate limit, so what the PD measures is where the
  robot actually is and a loop-closure jump is absorbed over ~1 s rather than
  arriving as a step. A dropout costs only the correction: the pose ages out
  and the base coasts on dead-reckoning.

  One value is not solved for and can be wrong: `slam_yaw_sign`, the
  handedness of the SLAM planar frame against the IK one. If
  `slam_base_correction_m` in `get_state()` grows steadily as you drive, flip
  it — see docs/RUNNING.md and tests/hardware/test_06_slam_pose.py.

* **The base is driven by a pose, not by a velocity.** `_dispatch_base` hands
  the solver's `base_position` -- where the IK believes the chassis should
  be -- and the dead-reckoned pose to `BasePoseController` (robot/base.py),
  which closes a PD on the difference. The solver's `base_velocity` is no
  longer dispatched: it describes a single tick, so anything the wheels did
  not deliver on that tick (clamp, deadband, filter, module slew) was lost
  when the next solve overwrote it, and the lag accumulated where nothing
  read it back. Everything downstream of the request is unchanged -- the
  heading-rate, acceleration and yaw-filter chain, the axis map, and the
  odometry integration all still act on the velocity that leaves this file.

* **Base axis mapping is a convention, not a measurement.** `BaseAxisMap`
  below encodes how the solver's body-frame velocity maps onto
  `Base.set_target_base_velocity`. Verify it with robot/teleop/joystick.py
  before enabling base motion — a wrong sign means the robot drives the wrong
  way while trying to help the arms reach.

* **The lift has two dispatch paths.** Against a controller that advertises
  `lift_velocity_v1`, the loop runs a position PD against measured height and
  streams the resulting velocity, which the firmware then shapes. Against an
  older controller — which only knows up / down / stop — it falls back to the
  original deadband bang-bang servo. Either way `set_lift_target()` takes a
  height in metres: the teleop client cannot tell the two apart.
"""

from __future__ import annotations

import csv
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import mink
from loop_rate_limiters import RateLimiter

from robot.arm.wholebody_ik import WholeBodyIK, WholeBodyIKConfig

# Column labels for the four swerve modules in the trajectory log. Taken from
# base_motor so the log can never disagree with the order Base.swerve_telemetry
# packs its arrays in. The fallback keeps this file importable without
# sparkcan_py (headless test runs, sim-only checkouts) -- it is the same
# literal base_motor defines, and the import is what keeps the two honest.
try:
    from robot.base_motor import MODULE_ORDER as _MODULE_LABELS, NUM_SWERVES as _NUM_SWERVES
except Exception:  # pragma: no cover - only when the CAN stack is absent
    _MODULE_LABELS, _NUM_SWERVES = ("FL", "FR", "RR", "RL"), 4

# The module geometry, for SwerveTwistShaper. Same provenance and the same
# fallback as the labels above: the shaper has to reproduce base_motor's wheel
# vectors *exactly* -- it exists to predict the angle base_motor will command
# -- so the constants are imported rather than restated, and the fallback is
# only so this file still imports without the CAN stack.
try:
    from robot.base_motor import (
        LENGTH as _MODULE_HALF_LENGTH,
        WIDTH as _MODULE_HALF_WIDTH,
        ROT_DIAG_SWAP_PERM as _MODULE_ROT_PERM,
        TRANS_OPPOSITE_MASK as _MODULE_TRANS_MASK,
        ZERO_SPEED_EPS_MPS as _MODULE_ZERO_SPEED_EPS,
    )
except Exception:  # pragma: no cover - only when the CAN stack is absent
    _MODULE_HALF_LENGTH, _MODULE_HALF_WIDTH = 0.1225, 0.170
    _MODULE_ROT_PERM = np.array([1, 0, 3, 2], dtype=int)
    _MODULE_TRANS_MASK = np.array([False, False, False, False], dtype=bool)
    _MODULE_ZERO_SPEED_EPS = 1e-3

# The base is driven by pose, and that controller lives with the rest of the
# base hardware in robot/base.py (base_motor.py stays swerve kinematics and
# motor control, and knows nothing about where the chassis is). Guarded for the
# same reason as the import above: a checkout without the CAN stack can still
# import this module for offline analysis, and only *constructing* the
# whole-body controller needs the real class.
try:
    from robot.base import BasePoseController
except Exception:  # pragma: no cover - only when the CAN stack is absent
    BasePoseController = None


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseAxisMap:
    """Maps a body-frame velocity onto ``Base.set_target_base_velocity``.

    In the description, the robot faces **−Y**: at the home keyframe both end
    effectors sit at y ≈ −0.25 m, and the left arm is at +X. So, in the base's
    own frame, forward = −y_model and left = +x_model.

    ``Base`` takes a 3-vector that is ``[forward, left, yaw]``. **This was
    wrong here until 2026-08-22**: the indices were crossed, so the solver's
    forward command went into the element the wheels treat as sideways, and a
    whole-body reach that needed the base to drive forward strafed instead.

    Three independent confirmations of the order:

    * ``robot/base_motor.py`` builds each wheel vector as
      ``atan2(target[1], target[0])``, so element 0 points the modules at 0
      degrees and element 1 at +90. Verified against the 2026-08-22 logs:
      commanded module angle matched ``atan2(target_1, target_0)`` to 0.00
      degrees on all four modules across 158 low-yaw ticks.
    * The previous codebase measured which of those is physically which, with
      two on-blocks probes: modules at 0 degrees is forward, +90 is left.
    * ``robot/teleop/joystick.py`` sends ``[vx, vy, w]`` with ``vx`` from the
      stick's *vertical* axis — i.e. it has always used element 0 as forward,
      and the joystick has always driven correctly. Only this map disagreed.

    The signs follow from the same evidence: +element 0 is forward, +element 1
    is left, and ``_world_to_body`` already returns ``lateral`` as +left. The
    yaw sign is the one value still unverified against the physical robot.
    """

    forward_index: int = 0
    lateral_index: int = 1
    yaw_index: int = 2
    forward_sign: float = +1.0
    lateral_sign: float = +1.0
    yaw_sign: float = +1.0

    def to_command(self, forward: float, lateral: float, yaw_rate: float) -> np.ndarray:
        cmd = np.zeros(3, dtype=float)
        cmd[self.forward_index] = self.forward_sign * forward
        cmd[self.lateral_index] = self.lateral_sign * lateral
        cmd[self.yaw_index] = self.yaw_sign * yaw_rate
        return cmd


class SwerveTwistShaper:
    """Bound how fast the *module angles* a twist implies may change.

    Every other shaper in `_dispatch_base` limits a component of the chassis
    twist — the linear magnitude, the linear direction, the linear
    acceleration, the yaw rate — and none of them limits the quantity the
    hardware actually has to serve, which is the four steering angles. Those
    are a function of all three components together::

        w_i = R_i(v_fwd, v_lat) + omega * a_i        theta_i = atan2(w_i)

    so a change confined to one component still swings the modules, and the
    swing is unbounded near the origin: `atan2` of a short vector is
    ill-conditioned, and `w_i` is short exactly when the chassis is creeping.

    That is the drive/spin transition. A twist of ``(0, 0, omega)`` points the
    modules tangent to the chassis circle; the instant a linear component
    appears they are asked to point somewhere else entirely, in one tick.
    Measured on the 2026-08-25/26 runs, folding out the 180 deg reversal a
    module serves for free with the drive sign: single-tick steering travel
    reached 88-90 deg against a 20 ms sample, i.e. ~4400 deg/s demanded of
    modules that slew at 265-353. 0.6-2.4% of moving ticks were over that
    ceiling, and 10-59% of those sat in the mixed regime where ``r*omega`` and
    ``|v|`` are within a factor of five of each other -- the transition
    itself. A module that cannot get there is a wheel dragging sideways, and
    four of them dragging in different directions is the twitch.

    The fix is to shape the twist in module space. Given the twist last issued
    and the one now wanted, walk as far along the segment between them as the
    modules can follow::

        q = q_prev + s * (q_des - q_prev),   s in [0, 1] as large as
        travel_i(q) <= budget_i  for every module

    A segment in twist space maps each ``w_i`` onto a straight line in the
    plane, and a point moving along a line sweeps its polar angle
    monotonically, so ``travel_i(s)`` is monotone and a short bisection finds
    the largest feasible ``s``. Whatever is left over is not lost: the caller
    integrates what was issued into `BaseOdometry`, so the residual reappears
    as pose error next tick and the PD asks for it again. The transition takes
    the several ticks it physically takes, instead of being demanded in one
    and delivered in none.

    ``travel`` is measured modulo pi. A 180 deg direction change costs a
    module nothing -- it flips the drive sign and does not turn -- and
    base_motor does exactly that, so charging for it would rate-limit a move
    the hardware makes instantly.

    ``budget`` is a scrub budget, not a flat angle rate. What hurts is a
    *rolling* wheel being dragged across its own direction; a wheel that is
    barely turning can be re-aimed for free, and has to be, or the base could
    never set off in a new direction from rest. So the per-module allowance is
    ``slew_limit`` at or above ``free_speed`` of wheel speed and opens up
    inversely below it, to at most ``free_ratio`` times the limit. Equivalent
    statement: ``|v_i| * dtheta_i/dt <= free_speed * slew_limit``, floored at
    the plain slew limit.
    """

    def __init__(self, slew_limit: float, free_speed: float,
                 free_ratio: float, iterations: int = 12) -> None:
        self.slew_limit = float(slew_limit)
        self.free_speed = float(free_speed)
        self.free_ratio = float(free_ratio)
        self.iterations = int(iterations)

        # Module arm vectors, written with base_motor's own expression so the
        # two cannot drift: w_r = omega * arm_i.
        w, l = float(_MODULE_HALF_WIDTH), float(_MODULE_HALF_LENGTH)
        arm_x = np.array([+w, -w, -w, +w], dtype=float)[_MODULE_ROT_PERM]
        arm_y = np.array([+l, +l, -l, -l], dtype=float)[_MODULE_ROT_PERM]
        self._arm = np.stack([arm_x, arm_y], axis=1)                  # (4, 2)
        self._trans_sign = np.where(
            _MODULE_TRANS_MASK, -1.0, 1.0).astype(float)[:, None]     # (4, 1)

        self._prev = np.zeros(3, dtype=float)
        self._dir = np.tile(np.array([1.0, 0.0]), (_NUM_SWERVES, 1))
        self.last_scale = 1.0

    # ── geometry ────────────────────────────────────────────────────────────

    def wheel_velocities(self, twist) -> np.ndarray:
        """Body twist (forward, lateral, yaw) → per-module velocity vectors."""
        q = np.asarray(twist, dtype=float).reshape(3)
        return self._trans_sign * q[:2] + self._arm * q[2]

    @staticmethod
    def _travel(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Steering travel from directions ``ref`` to ``target``, in [0, pi/2].

        Modulo pi, because a reversal is served by the drive sign. A module
        whose target vector is degenerate reads as zero travel, which is right:
        it has no direction to be asked for.
        """
        cross = ref[:, 0] * target[:, 1] - ref[:, 1] * target[:, 0]
        dot = ref[:, 0] * target[:, 0] + ref[:, 1] * target[:, 1]
        ang = np.abs(np.arctan2(cross, dot))
        return np.minimum(ang, math.pi - ang)

    def _budget(self, speeds: np.ndarray, dt: float) -> np.ndarray:
        limit = self.slew_limit * dt
        if self.free_speed <= 0.0:
            return np.full(_NUM_SWERVES, limit)
        relax = np.clip(self.free_speed / np.maximum(speeds, 1e-9),
                        1.0, max(self.free_ratio, 1.0))
        return limit * relax

    def _feasible(self, twist, budget: np.ndarray) -> bool:
        w = self.wheel_velocities(twist)
        return bool(np.all(self._travel(self._dir, w) <= budget))

    # ── state ───────────────────────────────────────────────────────────────

    def reset(self, module_angles=None) -> None:
        """Forget the issued twist; optionally re-seed from measured angles.

        Called whenever the base loses authority or halts. The *directions* are
        deliberately kept unless measurements are supplied: the modules do not
        move when the base is disarmed (base_motor leaves the steering
        setpoints standing), so where they are pointing is still the right
        reference for the tick that gets authority back.
        """
        self._prev = np.zeros(3, dtype=float)
        self.last_scale = 1.0
        if module_angles is None:
            return
        ang = np.asarray(module_angles, dtype=float).reshape(-1)
        if ang.size != _NUM_SWERVES or not np.all(np.isfinite(ang)):
            return
        self._dir = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    def _commit(self, twist: np.ndarray) -> None:
        self._prev = np.array(twist, dtype=float)
        w = self.wheel_velocities(self._prev)
        speed = np.linalg.norm(w, axis=1)
        moving = speed > _MODULE_ZERO_SPEED_EPS
        if np.any(moving):
            # Mirrors base_motor: a module asked for no speed holds its angle,
            # so the reference for the next tick is where it is still pointing.
            self._dir = np.where(moving[:, None],
                                 w / np.maximum(speed[:, None], 1e-12),
                                 self._dir)

    # ── the shaper ──────────────────────────────────────────────────────────

    def shape(self, twist, dt: float) -> np.ndarray:
        """Largest step from the last issued twist toward ``twist``."""
        want = np.asarray(twist, dtype=float).reshape(3)

        # Disabled, no time step, or a full stop. Stopping is exempt for the
        # same reason it is exempt from the acceleration limit: nothing may
        # delay the base coming to rest, and a twist of exactly zero leaves
        # base_motor holding the module angles rather than re-aiming them.
        if self.slew_limit <= 0.0 or dt <= 0.0 or not np.any(want):
            self.last_scale = 1.0
            self._commit(want)
            return want

        # Budgeted on the speed the wheels are *already* carrying, not on the
        # speed being asked for. Scrub is a rolling wheel dragged sideways, and
        # a module that is stopped is not rolling: base_motor scales each drive
        # by the cosine of its measured steering error, so a module commanded
        # to a new angle and a new speed at once gets no drive until it has
        # turned. Budgeting on the request instead would throttle every
        # departure from rest -- it measured 0.0022 m/s issued against a
        # 0.2 m/s strafe -- for a scrub that never happens.
        budget = self._budget(
            np.linalg.norm(self.wheel_velocities(self._prev), axis=1), dt)

        if self._feasible(want, budget):
            self.last_scale = 1.0
            self._commit(want)
            return want

        lo, hi = 0.0, 1.0
        delta = want - self._prev
        for _ in range(self.iterations):
            mid = 0.5 * (lo + hi)
            if self._feasible(self._prev + mid * delta, budget):
                lo = mid
            else:
                hi = mid

        out = self._prev + lo * delta
        self.last_scale = float(lo)
        self._commit(out)
        return out


@dataclass
class WholeBodyHardwareConfig:
    """Rates, clamps and enables for the hardware whole-body loop."""

    # One solve tick per 30 Hz teleop target -- matches the teleop client's
    # LOOP_RATE (robot/teleop/wholebody_teleop.py). The base relay deliberately
    # runs faster than this (108 Hz, see robot/yor.py) rather than matching it,
    # so it never falls behind; the swerve loop's own 3x-oversampled S-curve
    # profiling (base_motor.py CONTROL_FREQ) is unaffected by this rate.
    control_hz: float = 30.0

    # ── Arms ────────────────────────────────────────────────────────────────
    enable_arm_motion: bool = True
    # Arm joint targets are NOT sent to nerolib straight off the 30 Hz solve --
    # a fresh 7-DOF target only every 33.3 ms is a coarser stream than YOR_D's
    # proven arm-commanding chain (per-arm IK at 90 Hz, 3-step interpolation
    # per 30 Hz Cartesian target), which is what this pattern reproduces:
    # each newly-solved joint target is interpolated against the previous one
    # over `arm_interpolation_steps` sub-steps, dispatched at `arm_dispatch_hz`
    # by a dedicated thread decoupled from the 30 Hz solve loop. See
    # _dispatch_arms / _arm_dispatch_tick / _arm_dispatch_loop.
    arm_dispatch_hz: float = 90.0
    arm_interpolation_steps: int = 3
    # Minimum duration passed to nerolib/Ruckig for each interpolated
    # sub-target (i.e. one arm_dispatch_hz tick, not one solve tick). Keep
    # this close to one dispatch-tick period: a value much longer than the
    # actual dispatch period turns every update into a tiny stop-and-go
    # trajectory and caps observed speed (previously diagnosed at 55.6 ms
    # against a much faster loop). 0.0108 is the same ~0.97-of-one-tick
    # margin used at the former 108 Hz single-loop rate, now applied to the
    # 90 Hz arm dispatch tick.
    arm_preview_time: float = 0.0108
    # A joint target must differ from the WBC arm reference by more than this
    # before it is dispatched. The band is applied independently to all seven
    # joints. 0.05 was nerolib's resetToHome acceptance band, borrowed here
    # and far coarser than a streamed target needs; 0.005 keeps the chatter
    # guard an order of magnitude below it. Default since 2026-08-26.
    arm_joint_deadband_rad: float = 0.005
    # Hard cap on how far ahead of measured state a streamed joint target may
    # sit. This is a bounded look-ahead, not a per-cycle step: if the WBC stream
    # stops, Ruckig safely comes to rest at a target no farther than
    # arm_max_vel_rad_s * arm_command_lookahead_s away.
    arm_command_lookahead_s: float = 0.10
    # Hard cap used to derive that maximum look-ahead distance, independent of
    # the solver's own velocity limit.
    #
    # `joint_vel_max` in robot/arm/arm.py is per-joint now (2.98 rad/s on
    # joints 1-4, 3.72 on the wrist joints 5-7 -- 95% of each joint's live
    # firmware-reported max), not a single number. This clamp stays a single
    # scalar (it's a looser outer bound, not a per-joint match -- the
    # per-joint ceiling is arm.py's job), but it has to be at least the
    # highest of those, or it would silently override the wrist joints'
    # native limit -- they'd never reach the speed the controller is
    # configured for, and the discrepancy would only show up as sluggish
    # whole-body tracking on those joints specifically.
    arm_max_vel_rad_s: float = 3.72
    # Open-loop arm planning: seed from the encoders once at startup, then
    # advance IK from its previous commanded solution without reading arm
    # feedback each cycle. Nerolib's low-level motor PD remains closed-loop,
    # but the WBC model can drift from the physical arms if tracking is poor.
    use_measured_arm_state: bool = False
    # Record raw per-solve-tick joint/EE trajectories to
    # artifacts/wholebody_logs/trajectories/ for offline analysis -- the data
    # a real null-space projection (rather than the current soft posture
    # cost) will need to be designed against later. On by default; see
    # _TrajectoryRecorder.
    record_trajectories: bool = True

    # ── Lift ────────────────────────────────────────────────────────────────
    enable_lift_motion: bool = True
    # Stop band for the bang-bang fallback only. Below this the lift is
    # commanded to stop, which also keeps serial traffic down.
    lift_deadband_m: float = 0.01
    use_measured_lift: bool = True

    # Position PD, used when the controller advertises streamed velocity.
    #
    #     velocity = Kp * (desired - measured) - Kd * d(measured)/dt
    #
    # The derivative is taken from the *measurement*, never from the error, so
    # that moving the target does not produce a derivative kick — with Kd at
    # 0.05 s a 10 cm target step differentiated as error would ask for metres
    # per second on the first cycle.
    lift_kp: float = 2.0                  # 1/s
    lift_kd: float = 0.05                 # s
    # Height (Arduino telemetry, ~36 Hz -- unrelated to this loop's own rate)
    # now arrives about as often as this loop runs (30 Hz), rather than once
    # every three cycles as when this loop ran at 108 Hz. This is the time
    # constant that turns the raw sample-to-sample difference into a usable
    # velocity regardless of exactly how the two rates line up.
    lift_derivative_tau: float = 0.1      # s
    # Inside this band the command is exactly zero, not merely small.
    lift_velocity_deadband_m: float = 0.005
    lift_max_velocity_m_s: float = 0.05
    # Height older than this, while the lift is being driven, stops it.
    lift_feedback_max_age_s: float = 0.5
    # ...but not for this long after motion is first requested: the firmware
    # closes the driver relay and waits DRIVER_STARTUP_MS (500 ms) before it
    # generates a pulse or a height line, so a fresh command has to be allowed
    # to outlive one staleness window before it can be judged.
    lift_feedback_grace_s: float = 1.0
    # A gap this large between control cycles invalidates the derivative: the
    # loop stalled, and the height difference across the gap is not a velocity
    # anything should act on.
    lift_control_gap_s: float = 0.25

    # ── Base ────────────────────────────────────────────────────────────────
    # Whole-body base motion is *emergent*: the solver rolls the chassis only
    # when the arms and lift together cannot reach the target. Nothing sends
    # the base an explicit drive command on this path.
    enable_base_motion: bool = True
    # Applied to the linear velocity as a *vector*, not per axis. Per-axis
    # limits skew the commanded direction toward whichever axis saturates
    # first; the same reasoning is already written down for the SLAM
    # correction in BaseOdometry.apply_correction.
    base_max_lin_vel: float = 0.35   # m/s, magnitude of (forward, lateral)
    base_max_ang_vel: float = 1.60   # rad/s
    # Velocities below this are sent as zero, so solver noise doesn't leave the
    # swerve modules humming at a standstill. Also applied to the linear
    # velocity as a vector: deadbanding forward and lateral independently
    # rotated the command, so a 0.053 m/s request 21 degrees off the forward
    # axis went out as pure forward -- direction wrong, not just magnitude, and
    # the odometry then integrated the distorted version.
    # Raised from 0.02 on 2026-08-22. The commanded heading whirls when the
    # velocity vector is short -- atan2 of a small vector is ill-conditioned --
    # and the measured module-travel demand ran to a median 552 deg/s in the
    # 0.02-0.04 band against hardware that peaks near 300. The base spent
    # 25-41% of its moving ticks in that regime. This keeps it out.
    #
    # Turned into a hysteresis pair on 2026-08-25, mirroring the yaw axis
    # below: motion starts when |v| exceeds the entry threshold and stops
    # when it falls below the exit threshold. 0 disables the deadband
    # entirely (every command, however small, reaches the wheels).
    #
    # Default 0 as of the 2026-08-25 hardware sessions: with the low-pass
    # filter below in place the run was smoother, but the deadband made the
    # base move in start-stop bursts -- the filtered command hovered around
    # the thresholds, so the chassis kept halting mid-motion. The filter now
    # carries the noise-rejection duty the deadband was added for. The old
    # pair (0.05/0.025) is preserved behind --base-vel-deadband in yor.py;
    # if the modules hum or the heading whirls at standstill again (the
    # original 2026-08-22 reason for the deadband), restore it there.
    base_vel_deadband: float = 0.0         # m/s — entry: below this, stay stopped
    base_vel_deadband_exit: float = 0.0    # m/s — exit: once moving, stop below this
    # One-pole low-pass on the linear velocity request, applied per body-frame
    # component (so direction is filtered consistently, not just magnitude)
    # before the hysteresis deadband. The yaw axis has had this since
    # 2026-08-24; the linear axes originally did not because that corpus
    # showed no linear sign-flips. The 2026-08-25 run with the per-axis yaw
    # weight changed the regime: the solver's linear request flickered
    # between zero and above-entry (334 movement bouts, median 100 ms, 4.5
    # deadband crossings/s), and hysteresis alone cannot help when the
    # request genuinely returns to zero. The filter merges those bursts into
    # either a sustained command or nothing, at the cost of ~tau of onset
    # lag on translation. 0 disables.
    # Raised 0.08 -> 0.15 on 2026-08-25 alongside the null-space base
    # recentering objective (base_recenter_gain in WholeBodyIKConfig): the
    # 14:30 run still surged (peak speed 2x the mean within each moving
    # stretch); 0.15 matches the yaw tau the operator settled on.
    base_vel_filter_tau: float = 0.15      # s
    # Yaw gets its own deadband, in its own units, and it is a hysteresis
    # pair rather than a single threshold. The 2026-08-24 runs showed why: the
    # solver's yaw request is noise centred near zero (nonzero median
    # 0.019 rad/s) that crossed the old single 0.02 threshold 7.4 times per
    # second, so the base received rotation in single-tick pulses — and every
    # pulse re-aims all four swerve modules, which is the "constant
    # correction" felt on the floor. Real rotation in those runs sat at
    # 0.2-0.45 rad/s, far above either threshold: a 0.05 entry kept 82% of the
    # total commanded rotation while cutting the passing ticks by half.
    #
    # Rotation starts when the (filtered) request exceeds the entry threshold
    # and stops when it falls below the exit threshold. The gap is what kills
    # the chatter — a request hovering at either single threshold would still
    # toggle every tick.
    # Default 0 (disabled) as of 2026-08-25, the same decision as the linear
    # pair above and for the same reason: with the low-pass filter carrying
    # the noise-rejection duty, a deadband turns marginal requests into
    # start-stop bursts. The 3x-raised experiment (0.15/0.075) confirmed the
    # chassis had been micro-correcting yaw, and the raised threshold traded
    # that for chatter around the boundary instead. Both pairs are preserved
    # behind --base-yaw-deadband in yor.py; the pre-experiment value was
    # 0.05/0.025.
    base_yaw_deadband: float = 0.0         # rad/s — entry: below this, stay stopped
    base_yaw_deadband_exit: float = 0.0    # rad/s — exit: once rotating, stop below this
    # How strongly odometry corrects the solver's belief of the base pose,
    # per tick: belief += alpha * (odometry - belief). 1.0 is a hard reset to
    # odometry every tick (the closed-loop behaviour this replaced); 0.0 is
    # fully open loop -- the solver integrates its own base commands and
    # never reads odometry back, matching how the arms are run
    # (use_measured_arm_state=False).
    #
    # Measured on the 2026-08-24 replay: alpha 0.3 cuts command churn 39%
    # for +4% EE error; at 0.0 churn drops the same but the belief drifts
    # freely from the physical base (44-68 mm over a session in replay), and
    # the true-world EE error rises accordingly while the solver's own frame
    # still reads ~0.8 mm -- open loop always *looks* perfect from inside.
    # Odometry itself is still integrated and SLAM-corrected regardless, so
    # telemetry and any consumer of odometry.pose are unaffected.
    #
    # Since the base is driven by pose, this knob is no longer free: it feeds
    # odometry back into the very belief the pose target is read from, which
    # is the same signal the PD is trying to close on. Under a sustained
    # solver request `v`, the speed the base settles at is
    #
    #     u ≈ v * kp*dt / (alpha + kp*dt)
    #
    # so alpha 0 tracks the solver exactly (u = v), while alpha 0.3 with the
    # default gains (kp*dt = 0.05) leaves about a seventh of it -- the base
    # would crawl. (Simulated over the discrete loop: 15% at alpha 0.3, 5% at
    # a hard reset; the formula ignores where in the tick each step lands.)
    # Anything above 0 is therefore a deliberate trade, and init() says so out
    # loud. This is why the belief must free-run for pose control to work.
    base_feedback_alpha: float = 0.0
    # ── Base leash ──────────────────────────────────────────────────────────
    # Cap on how far the solver's belief of the chassis pose may sit from the
    # dead-reckoned pose. The solver advances that belief every tick, but the
    # chassis only ever delivers part of it -- the clamp, the deadband, the
    # acceleration ramp -- and nothing bounded the difference. Under a
    # sustained request (an operator holding a direction, which is most of a
    # teleop session) the gap grows at the whole velocity deficit: 0.25 m/s
    # asked for and 60% delivered is 0.1 m/s of divergence, ~200 mm after two
    # seconds. Two things follow, and only the second is obvious:
    #
    #   * the arms are solved against a chassis pose that has not happened, so
    #     the solver retracts them on the assumption the base closed the
    #     distance, and the EE falls short by exactly the gap;
    #   * the base keeps crawling after the operator stops, chasing a target
    #     still out in front of it.
    #
    # The clamped pose is written back into the IK configuration, so the
    # excess is *forgotten* rather than banked -- the same construct
    # target_leash_m applies to the EE targets one level up, and the
    # write-back is what makes it a leash rather than a rate limiter. Held
    # next to odometry, the belief lets the solver see that the target is
    # still far away, so it reaches with the arms instead: the base's
    # shortfall gets compensated without measuring anything new.
    #
    # A limit on the per-tick step would not do this. The divergence is not
    # spiky, it is a slow one-directional bleed in which every individual tick
    # is entirely reasonable.
    #
    # 0 disables. Sizing: the leash is the standing EE shortfall accepted in
    # exchange for the solver noticing, so it wants to be short enough to feel
    # and long enough not to bind while the base is keeping up. `base_leash`
    # in the trajectory log is the metres clamped off each tick -- non-zero on
    # most moving ticks means this is too short. 0.2 since 2026-08-26: at
    # 0.1 the leash clamped on most moving ticks; at 0.2 it engages on
    # 0.4-1.3% of them, which is the intended "only when the base is truly
    # falling behind".
    base_leash_m: float = 0.2
    # The same leash on yaw. Off by default: yaw diverges proportionally
    # faster than translation (base_max_ang_vel / base_max_lin_vel = 2.4), so
    # the equivalent of 0.1 m is about 0.24 rad -- but that ratio is arithmetic
    # rather than measurement until `base_leash` in the log says how far yaw
    # actually runs. Try 0.24.
    base_leash_rad: float = 0.0
    # One-pole low-pass on the yaw request, applied before the deadband.
    # It merges one-tick pulses into either a sustained rotation or nothing,
    # and flattens the solver's occasional absurd spikes (12 rad/s for one
    # tick was observed). The cost is lag on rotation onset only. 0 disables.
    #
    # Raised from 0.08 on 2026-08-25. With the yaw deadband removed the
    # filter is the only thing standing between solver yaw noise and the
    # modules, and the per-axis yaw weight makes the solver use yaw freely
    # (weight 1.0 -- a trial at 5.0 stopped chassis yaw entirely, so the
    # smoothing duty lands here, not on the cost). 0.25 measured well in
    # replay (halved churn, 80% kept, ~67 ms onset lag) but was clearly
    # worse on the floor -- the lag on rotation is felt harder than the
    # replay metrics suggest. 0.15 is the compromise the operator settled
    # on. Tunable via --base-yaw-filter-tau in yor.py.
    base_yaw_filter_tau: float = 0.15      # s
    # Ceiling on how fast the *direction* of the base command may turn, which
    # is a hardware limit rather than a preference: the swerve modules peak at
    # 265-353 deg/s of slew, and in the 2026-08-22 runs 27-44% of moving ticks
    # asked for more than that -- p99 was about ten times the limit. A module
    # that cannot reach its commanded angle drives the chassis somewhere other
    # than where the solver asked, so this bounds the ask to something the
    # hardware can actually serve. 200 deg/s leaves margin under the slowest
    # measured module. Set to 0 to disable.
    #
    # Only the translation direction is limited. Yaw is a scalar and does not
    # have the ill-conditioning problem, though it does move the module angles
    # through the rotation term -- that part is not bounded here.
    base_heading_rate_limit: float = 3.49   # rad/s (200 deg/s)
    # Ceiling on how fast the base *velocity vector* may change, in m/s^2.
    #
    # base_heading_rate_limit deliberately does not bound a reversal: a swerve
    # module answers a 180 deg direction change by flipping the drive rather
    # than turning, so for the module it costs nothing. For the chassis it
    # costs everything -- it has to decelerate, stop and accelerate the other
    # way -- and the 2026-08-24 runs did exactly that, reversing every 0.08 s
    # against a measured 167 ms response. 42.6% of heading changes were
    # 170-180 deg and every one passed the heading limiter untouched.
    #
    # 1.5 m/s^2 turns a full 0.25 -> -0.25 reversal into 0.33 s, about twice
    # the chassis response time, so a command the base cannot follow is never
    # issued. Kept below base_motor's own S-curve limit (1.9) so the profiler
    # is not the binding constraint -- and unlike the profiler, which
    # re-targets every tick and so never finishes a segment when the command
    # flips, this shapes the command itself.
    #
    # An exact zero is exempt: stopping is never delayed, so a deadbanded
    # command or a halt still takes effect immediately. Set to 0 to disable.
    base_max_accel: float = 1.5             # m/s^2
    # ── Swerve module slew (SwerveTwistShaper) ──────────────────────────────
    # The three limits above each bound one component of the twist. None of
    # them bounds the four steering angles, which depend on all three at once
    # -- so a twist that passes every one of them can still demand a 90 deg
    # module swing in a single tick, and the drive/spin transition demands
    # exactly that. See SwerveTwistShaper for the measurement.
    #
    # base_module_slew_limit is the ceiling on commanded module angular rate,
    # in rad/s. Same number and same justification as
    # base_heading_rate_limit -- modules measured at 265-353 deg/s, so 200
    # deg/s leaves 1.3-1.8x margin -- but applied to the angle the module is
    # actually given rather than to the translation heading alone. Set to 0 to
    # disable the shaper entirely.
    base_module_slew_limit: float = 3.49    # rad/s (200 deg/s)
    # Wheel speed at or above which the full limit applies. Below it the
    # allowance opens up inversely: a wheel that is barely rolling scrubs
    # nothing when it is re-aimed, and has to be free to re-aim or the base
    # could never set off in a new direction from a standstill. 0.03 m/s is
    # about half base_vel_deadband's old value and well above the 1 mm/s
    # base_motor treats as no direction at all.
    base_module_free_speed: float = 0.03    # m/s
    # How far that relaxation may go. 20x turns the 200 deg/s ceiling into
    # 4000 deg/s at a standstill, i.e. effectively unlimited -- the module is
    # then commanded straight to its new angle and base_motor's
    # cos_error_scaling holds the drive back until it arrives, which is the
    # correct "align, then go".
    base_module_free_ratio: float = 20.0
    # ── Base pose PD ────────────────────────────────────────────────────────
    # Gains for BasePoseController (robot/base.py), which turns the solver's
    # base *pose* target into the velocity request the chain above shapes.
    # The linear pair mirrors the navigation PID in the same file (k_pos 1.5,
    # kd_pos 0.15) and the yaw pair its k_theta/kd_theta, because both close on
    # the same chassis with the same mass: the difference here is only which
    # pose is being tracked, not what is being pushed.
    #
    # With kp 1.5, base_max_lin_vel is reached at 0.167 m of error, so any
    # lag larger than that is served at full speed and the gain only shapes
    # the arrival.
    base_pose_kp_xy: float = 1.5           # (m/s)/m
    base_pose_kd_xy: float = 0.15          # (m/s)/(m/s), damping on measurement
    base_pose_kp_yaw: float = 2.0          # (rad/s)/rad
    base_pose_kd_yaw: float = 0.2          # (rad/s)/(rad/s)
    # Pose errors this small are not worth moving for: inside them the
    # controller outputs exactly zero rather than leaving the modules humming.
    # The linear band is on the error *vector*, for the same reason
    # base_vel_deadband is -- a per-axis band turns the command as well as
    # shrinking it. Unlike the velocity deadbands above these can stay on:
    # the pose error is an accumulated quantity, not a per-tick noise
    # reading, so it does not flicker across the threshold.
    base_pose_deadband_m: float = 0.01
    base_pose_yaw_deadband_rad: float = 0.02
    # Time constant of the low-pass on the measured base velocity that feeds
    # the D term. Matches lift_derivative_tau, and for the same reason: the
    # raw sample-to-sample difference of a 30 Hz pose is more artefact than
    # velocity.
    base_pose_derivative_tau: float = 0.10  # s
    # Reference feedforward for BasePoseController -- see its `ff_gain`.
    # 1.0 (the whole target rate fed forward) by default since 2026-08-26;
    # 0 disables. It pairs with base_leash_m: the two only stop fighting
    # once speed comes from the reference rate rather than from accumulated
    # pose error.
    base_pose_ff_gain: float = 1.0
    base_pose_ff_max_frac: float = 0.8
    # Low-pass on the feedforward -- see BasePoseController.ff_tau. The
    # feedforward differentiates the reference, so it amplifies solver jitter;
    # this is the knob that buys the twitch back down. Try 0.04-0.06.
    base_pose_ff_tau: float = 0.0
    # Where the chassis' nose points at yaw 0, in the IK world frame. The
    # description has the robot facing -Y with +X to its left, so the forward
    # axis sits a quarter turn behind the yaw axis. Same convention as
    # BaseAxisMap and _world_to_body; changing one without the others turns
    # every base command.
    base_pose_heading_offset: float = -math.pi / 2.0   # rad
    base_axis_map: BaseAxisMap = field(default_factory=BaseAxisMap)
    # Which swerve PID gains were actually on the controllers for this run.
    # Set by robot/yor.py after the startup sync, and stamped into the
    # trajectory log. Nothing reads it at runtime: it exists because the two
    # 2026-08-22 runs had to have their gain set *inferred* from the drive
    # tracking ratio, which is a bad way to find out what an experiment was.
    base_pid_provenance: str = "unknown"

    # ── Target leash (gated, [S2]) ──────────────────────────────────────────
    # Cap how far an EE target may sit from the solver's current EE pose:
    # a target beyond the leash is pulled back onto the leash sphere (and
    # cone, for orientation) each tick, and the *stored* target is replaced
    # by the leashed one so the excess is forgotten rather than banked.
    # This kills clutch wind-up at the server: streaming past a constraint
    # no longer accumulates metres of pending error that the arm then
    # replays at full speed when the constraint releases -- the target is
    # never allowed further ahead than the leash. Also bounds the
    # per-iteration errors the solver races on, which is what dragged arms
    # through singular regions on a re-engage jump. 0 disables (current
    # behaviour). Sizing: at 30 Hz a leash of 0.15 m still permits
    # 0.15 m / 33 ms of *new* lead per tick, far above any real hand
    # speed, so tracking latency is unaffected -- only wind-up is. On by
    # default since 2026-08-26.
    target_leash_m: float = 0.15
    target_leash_rad: float = 0.8

    # ── Manual override ─────────────────────────────────────────────────────
    # Direct base / lift commands (joystick, nav, RPC) suspend the whole-body
    # loop's authority over that subsystem for this long after the last one,
    # so the two controllers never fight over the same actuator.
    manual_override_timeout_s: float = 0.5

    # ── SLAM base pose (PD feedback) ─────────────────────────────────────────
    # What closes the base pose loop on the floor instead of on the command.
    #
    # `_dispatch_base` measures with `self.odometry.pose`, and that pose is
    # dead-reckoned by integrating the velocity the base was *commanded*
    # (BaseOdometry.update). Left alone that makes the PD an echo chamber: the
    # error decays because odometry moved, whether or not the robot did. Wheel
    # slip, a push, a module that never reached its angle and a stalled drive
    # are all invisible to it, by construction.
    #
    # With this on, `_correct_base_from_slam` pulls that same pose toward the
    # Odin VIO+lidar fix (`slam/pose`, robot/odin_pub_node.py) every tick, so
    # what the PD measures is where the robot actually is. Dead-reckoning still
    # carries the estimate between fixes -- it is smooth, available at loop
    # rate, and exact with respect to the solver's intent -- but it no longer
    # gets to be wrong for free.
    #
    # This is the same signal robot/base.py's navigation modes have always
    # closed on; only the whole-body loop was running open.
    #
    # Frame handling is in SlamBaseFrame: `align()` pins the SLAM frame onto
    # the odometry pose at the first usable fix, so the correction starts at
    # exactly zero and only ever removes error accumulated after that. The one
    # value it cannot solve for is `slam_yaw_sign`, below.
    enable_slam_base_pose: bool = True
    # Where odin_pub_node publishes. Whole-body runs on the Pi, SLAM on Thor;
    # this mirrors THOR_IP in robot/base.py.
    slam_pose_host: str = "192.168.1.11"
    slam_pose_port: int = 6000
    # Poll rate of the background listener. The control loop never blocks on
    # the network — it reads whatever this thread last cached.
    slam_pose_hz: float = 20.0
    # Ignore a cached pose older than this and coast on dead-reckoning.
    slam_pose_max_age_s: float = 0.5
    # Handedness of the SLAM planar frame relative to the IK one. This is the
    # ONLY value that needs calibrating: +1 or -1. Drive the base forward a
    # metre with correction enabled and watch `slam_base_correction_m` in
    # get_state() — it should stay small. If it grows steadily, flip the sign.
    slam_yaw_sign: float = +1.0
    # How fast the correction may pull odometry toward the SLAM fix.
    #
    # These are the knob that decides whether this is *feedback* or merely
    # drift removal, so they are sized against the error rather than against
    # taste. The worst case is a command the floor refuses entirely -- full
    # speed ordered, robot stationary -- and then the gap between odometry and
    # truth opens at base_max_lin_vel (0.25 m/s) and base_max_ang_vel
    # (0.60 rad/s). A correction slower than that loses the race: odometry
    # stays wrong for as long as the slip lasts, which is exactly the interval
    # the PD needed to see it. These are ~4x and ~3.3x those ceilings, so the
    # correction always wins and the measured pose tracks the fix to within
    # about a tick's worth of motion.
    #
    # They are still limits, not a hard assignment, and that is what keeps a
    # loop closure survivable: the SLAM pose steps discontinuously when the
    # map snaps, and a step handed straight to the PD is a step in its error.
    # At these rates a 1 m jump is taken up over ~1 s instead of in one tick,
    # which the base can actually drive; sub-centimetre corrections, the
    # common case, are absorbed in a single tick and never rate-limited at all.
    #
    # Raised from 0.10 / 0.20 when this became the PD's measurement. Those
    # values were chosen when the correction only had to bleed off slow drift
    # *and* had to keep the IK configuration continuous, because odometry fed
    # it. It no longer does at base_feedback_alpha = 0 (the default), which is
    # what makes the faster rate safe -- see the warning in init().
    slam_correction_max_lin_rate: float = 1.00   # m/s
    slam_correction_max_yaw_rate: float = 2.00   # rad/s
    # Warn once if the standing offset exceeds this — it means dead-reckoning
    # and SLAM have diverged far more than drift explains (wrong yaw_sign,
    # wheel slip, or the robot was picked up).
    slam_offset_warn_m: float = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Lift position -> velocity
# ─────────────────────────────────────────────────────────────────────────────

class LiftVelocityPD:
    """Turns a lift *position* target into a bounded velocity request.

        velocity = Kp * (desired - measured) - Kd * filtered d(measured)/dt

    Two details matter more than the gains:

    * **The derivative is of the measurement, not of the error.** A target step
      — which is exactly what an operator lift command is — would otherwise
      differentiate to a huge transient. Differentiating the measurement makes
      the D term pure damping, which is what it is for here.

    * **The derivative is low-pass filtered.** Height (Arduino telemetry, ~36 Hz)
      and this loop (30 Hz) now run at close to the same rate, rather than the
      loop running 3x faster as before, so the unfiltered difference is less
      sample-and-hold-y than it used to be -- but `tau` still turns it into a
      usable velocity rather than an artefact of however the two rates land.

    The object is deliberately state-holding and resettable: every event that
    breaks the continuity of the measurement — a stale reading, a manual
    override, a disarm, a stalled control loop — must clear the filter, or the
    first cycle afterwards damps against a velocity that was never real.
    """

    def __init__(self, kp: float, kd: float, tau: float, deadband: float,
                 max_velocity: float, max_gap_s: float = 0.25) -> None:
        self.kp = float(kp)
        self.kd = float(kd)
        self.tau = float(tau)
        self.deadband = float(deadband)
        self.max_velocity = float(max_velocity)
        self.max_gap_s = float(max_gap_s)
        self.reset()

    def reset(self) -> None:
        self._last_height: Optional[float] = None
        self._last_time: Optional[float] = None
        self._filtered_velocity = 0.0

    @property
    def filtered_velocity(self) -> float:
        """The damping term's view of how fast the column is moving (m/s)."""
        return self._filtered_velocity

    def update(self, desired_m: float, measured_m: float, now: float) -> float:
        """One cycle. Returns the velocity to command, in m/s."""
        self._update_derivative(measured_m, now)

        error = float(desired_m) - float(measured_m)
        if abs(error) <= self.deadband:
            # Exactly zero, not a small residual: a lift that is close enough
            # must be still, so the column is not left humming against the
            # last few tenths of a millimetre.
            return 0.0

        velocity = self.kp * error - self.kd * self._filtered_velocity
        return float(np.clip(velocity, -self.max_velocity, self.max_velocity))

    def _update_derivative(self, measured_m: float, now: float) -> None:
        measured_m = float(measured_m)
        last_height, last_time = self._last_height, self._last_time
        self._last_height, self._last_time = measured_m, float(now)

        if last_height is None or last_time is None:
            return

        dt = float(now) - last_time
        if dt <= 0.0 or dt > self.max_gap_s:
            # Either time did not advance or the loop stalled. Neither gives a
            # velocity worth damping against.
            self._filtered_velocity = 0.0
            return

        raw = (measured_m - last_height) / dt
        alpha = dt / (self.tau + dt) if self.tau > 0.0 else 1.0
        self._filtered_velocity += alpha * (raw - self._filtered_velocity)


# ─────────────────────────────────────────────────────────────────────────────
# Base odometry
# ─────────────────────────────────────────────────────────────────────────────

class BaseOdometry:
    """The chassis pose (x, y, theta) the base PD measures, IK world frame.

    `update` integrates the velocity that was actually *commanded* to the
    base: exact with respect to the solver's intent, free of unit guesswork,
    and completely open-loop with respect to the floor. `apply_correction`
    is the other half -- it pulls that estimate toward an absolute fix, and
    `_correct_base_from_slam` drives it from the Odin pose once per tick.

    The two halves are not interchangeable. Dead-reckoning supplies rate:
    smooth, at loop rate, never missing. The fix supplies truth, at 20 Hz and
    with steps in it. Running on either alone gives up something the base PD
    needs, which is why this holds one pose that both write to.
    """

    def __init__(self) -> None:
        self._pose = np.zeros(3, dtype=float)

    @property
    def pose(self) -> np.ndarray:
        return self._pose.copy()

    def reset(self, pose: Optional[np.ndarray] = None) -> None:
        self._pose = np.zeros(3) if pose is None else np.asarray(pose, dtype=float).copy()

    def update(self, world_velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate a world-frame [vx, vy, omega] over dt."""
        self._pose += np.asarray(world_velocity, dtype=float) * dt
        self._pose[2] = math.atan2(math.sin(self._pose[2]), math.cos(self._pose[2]))
        return self.pose

    def offset_to(self, target: np.ndarray) -> tuple[float, float]:
        """(linear, angular) distance from the dead-reckoned pose to `target`."""
        target = np.asarray(target, dtype=float)
        lin = float(math.hypot(target[0] - self._pose[0], target[1] - self._pose[1]))
        ang = abs(_wrap_pi(float(target[2]) - float(self._pose[2])))
        return lin, ang

    def apply_correction(
        self,
        target: np.ndarray,
        dt: float,
        max_lin_rate: float,
        max_yaw_rate: float,
    ) -> tuple[float, float]:
        """Nudge the dead-reckoned pose toward an absolute `target`.

        The step is rate-limited, so an absolute source that jumps (SLAM loop
        closure) never produces a discontinuity in the IK configuration — the
        offset is bled off over many cycles instead. Returns the (linear,
        angular) offset that remained *before* this step, for diagnostics.

        The linear error is clamped as a vector rather than per-axis so the
        correction always points at the target instead of skewing toward
        whichever axis saturated first.
        """
        target = np.asarray(target, dtype=float)
        err_x = float(target[0] - self._pose[0])
        err_y = float(target[1] - self._pose[1])
        err_t = _wrap_pi(float(target[2]) - float(self._pose[2]))
        lin_before = float(math.hypot(err_x, err_y))
        ang_before = abs(err_t)

        max_lin_step = max(0.0, max_lin_rate) * dt
        if lin_before > max_lin_step and lin_before > 1e-12:
            scale = max_lin_step / lin_before
            err_x *= scale
            err_y *= scale

        max_yaw_step = max(0.0, max_yaw_rate) * dt
        err_t = float(np.clip(err_t, -max_yaw_step, max_yaw_step))

        self._pose[0] += err_x
        self._pose[1] += err_y
        self._pose[2] = _wrap_pi(self._pose[2] + err_t)
        return lin_before, ang_before


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# ─────────────────────────────────────────────────────────────────────────────
# SLAM base pose
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SlamBaseFrame:
    """Planar transform from the SLAM world frame into the IK world frame.

    The SLAM pose is Y-up with the robot moving in the X–Z plane (planar
    coordinates ``(t_x, t_z)``, yaw about Y). The IK base is planar
    ``base_x``/``base_y``/``base_yaw`` in a model frame where the robot faces
    −Y. The two therefore differ by a rotation, a translation, and — depending
    on how the induced planar frames come out — a reflection.

    Rather than ask anyone to derive that, ``align()`` solves the rotation and
    translation from one matched pair of poses, so the only thing left to
    choose is ``yaw_sign``: +1 or −1. Because alignment is exact at the moment
    it is taken, the correction starts at zero and only ever has to remove
    *relative* drift accumulated afterwards, which is precisely the error that
    dead-reckoning suffers from.
    """

    yaw_sign: float = +1.0
    # Where the IK frame's nose points at yaw 0, relative to its yaw axis. The
    # SLAM frame has no such offset -- its yaw *is* the nose direction -- so
    # this is the whole of the difference between the two yaw conventions, and
    # `align` needs it to turn a yaw difference into an axis rotation. Same
    # number and same meaning as BasePoseController.heading_offset and
    # base_pose_heading_offset; the controller and this must agree or the
    # correction and the PD would be working in frames a quarter turn apart.
    heading_offset: float = -math.pi / 2.0
    _rot: float = 0.0        # rotation from SLAM planar axes to IK planar axes
    _tx: float = 0.0
    _ty: float = 0.0
    _aligned: bool = False

    @property
    def aligned(self) -> bool:
        return self._aligned

    def reset(self) -> None:
        self._rot = self._tx = self._ty = 0.0
        self._aligned = False

    def _reflect(self, sx: float, sy: float) -> tuple[float, float]:
        """Flip the planar position so the frame handedness matches the IK one.

        The SLAM planar frame is left-handed. Its position axes are (t_x, t_z)
        of a Y-up pose and its yaw is about +Y, so the nose sits at
        (cos yaw, -sin yaw): increasing yaw rotates the heading *clockwise* in
        those coordinates, and the robot's physical left is at -90 degrees, not
        +90. Measured on the floor 2026-08-26 with
        tests/hardware/test_09_axis_match.py: commanding left moved the base
        -83.6 degrees off the nose, against the model's exact -90. The IK frame
        is the ordinary right-handed one (left at +90), so mapping one to the
        other needs a reflection, and `align()` cannot supply it -- it solves a
        rotation and a translation, and neither changes handedness.

        Exactly one of the two halves must flip. Until 2026-08-26 this returned
        the reflection only when `yaw_sign < 0`, which is the same flag `to_ik`
        uses to negate the yaw -- so the two always fired together and
        cancelled, and *both* settings produced a mirrored frame in which the
        path turned one way while the reported yaw turned the other. Neither
        value worked, so no amount of calibrating `slam_yaw_sign` could fix it:
        with the correction feeding the base PD, a 5 cm move diverged to 1.9 m.

        Reflecting whenever `yaw_sign >= 0` decouples them. `+1` now means
        "reflect the position, keep the yaw" and `-1` means "keep the position,
        negate the yaw" -- two proper rigid motions that are mirror images of
        each other, which is what a handedness knob should select between. +1
        is the value the floor run calls for.
        """
        return (sx, -sy) if self.yaw_sign >= 0 else (sx, sy)

    def align(self, slam_pose: np.ndarray, ik_pose: np.ndarray) -> None:
        """Solve the transform so that `slam_pose` maps exactly onto `ik_pose`.

        `_rot` has to rotate the *position* axes, so it must be the angle
        between the two frames' axes -- and that is not the difference of the
        two yaw numbers, because the two conventions measure yaw against
        different body axes. In the SLAM frame the nose sits at the yaw angle
        itself (body +X is forward). In the IK frame the robot faces -Y, so
        its nose sits at `yaw + heading_offset`, a quarter turn behind.

        Equating the yaw numbers, which is what this did until 2026-08-26,
        therefore leaves `_rot` short by exactly `heading_offset` and rotates
        every mapped position 90 degrees away from the heading it is paired
        with. Alignment still looked perfect -- both pose and yaw map exactly
        at the instant it is taken, because `_tx`/`_ty` absorb the position
        error -- and the frame only revealed itself once the robot *moved*:
        driving forward moved the corrected pose sideways, so the correction
        opposed the motion on 100% of moving ticks at 1.5x the commanded
        speed, and the base drove at its ceiling without ever arriving.

        Matching the nose *directions* instead is what makes it a real rigid
        motion:  ik_nose = slam_nose + _rot, i.e.
        `pyaw + heading_offset = yaw_sign * syaw + _rot`.
        """
        sx, sy, syaw = (float(v) for v in slam_pose)
        px, py, pyaw = (float(v) for v in ik_pose)
        self._rot = _wrap_pi(pyaw + self.heading_offset - self.yaw_sign * syaw)
        rx, ry = self._reflect(sx, sy)
        c, s = math.cos(self._rot), math.sin(self._rot)
        self._tx = px - (c * rx - s * ry)
        self._ty = py - (s * rx + c * ry)
        self._aligned = True

    def to_ik(self, slam_pose: np.ndarray) -> np.ndarray:
        """Map a SLAM planar pose into IK planar coordinates."""
        sx, sy, syaw = (float(v) for v in slam_pose)
        rx, ry = self._reflect(sx, sy)
        c, s = math.cos(self._rot), math.sin(self._rot)
        return np.array([
            c * rx - s * ry + self._tx,
            s * rx + c * ry + self._ty,
            # Inverse of the datum in `align`: the position rotation carries
            # the heading-convention offset, so the yaw has to take it back
            # out or the two halves would disagree by that same quarter turn.
            _wrap_pi(self.yaw_sign * syaw + self._rot - self.heading_offset),
        ], dtype=float)


class SlamPoseListener:
    """Background poller for the SLAM base pose, cached for a fast control loop.

    Two reasons this exists rather than reading an existing pose:

    * **The control loop must not touch the network.** commlink's subscriber is
      pull-mode — every read is a round trip to Thor. A 30 Hz loop cannot
      afford one, so this thread polls at its own rate and the loop reads
      whatever was cached.
    * **BaseController's pose is not usable here.** `_send_base_command` pins
      that controller to ``"BASE_VEL"`` mode on every dispatch, and its `_run`
      loop hits `continue` in that mode *before* reaching its `get_pose` call.
      So `yor.pose` goes stale exactly while whole-body control is running —
      the one time we would want it.

    Reads never raise: a missing publisher just means `latest()` returns None
    and the caller coasts on dead-reckoning.
    """

    def __init__(self, host: str, port: int, hz: float = 20.0):
        self._host = host
        self._port = port
        self._period = 1.0 / max(1.0, float(hz))
        self._lock = threading.Lock()
        self._pose: Optional[np.ndarray] = None
        self._stamp: float = 0.0
        self._stop_evt = threading.Event()
        self._warned = False
        self._thread = threading.Thread(
            target=self._run, name="wb-slam-pose", daemon=True
        )
        self._thread.start()

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        """(planar pose or None, age in seconds)."""
        with self._lock:
            if self._pose is None:
                return None, float("inf")
            return self._pose.copy(), time.monotonic() - self._stamp

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=1.0)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _planar_from_msg(msg) -> Optional[np.ndarray]:
        """slam/pose 20-float message → planar (x, y, yaw) in the SLAM frame.

        Uses the base pose (indices 0:7), matching `get_pose` in robot/base.py:
        planar position is (t_x, t_z) of the Y-up translation and yaw is taken
        about the Y axis as atan2(-R[2,0], R[0,0]).
        """
        if msg is None or len(msg) < 7:
            return None
        # Confidence, when the publisher provides it, gates unusable tracking.
        if len(msg) > 19 and float(msg[19]) < 10.0:
            return None
        qx, qy, qz, qw = (float(msg[i]) for i in range(4))
        tx, tz = float(msg[4]), float(msg[6])
        # Only the two entries of R that the yaw needs, expanded by hand so this
        # module does not pull in scipy.
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r20 = 2.0 * (qx * qz - qw * qy)
        return np.array([tx, tz, math.atan2(-r20, r00)], dtype=float)

    def _run(self) -> None:
        sub = None
        try:
            # Imported lazily so simulation and the headless tests never need
            # commlink just to construct a WholeBodyController.
            from commlink import Subscriber
            from robot.topics import POSE_TOPIC

            sub = Subscriber(host=self._host, port=self._port, topics=[POSE_TOPIC])
            while not self._stop_evt.is_set():
                try:
                    pose = self._planar_from_msg(sub[POSE_TOPIC])
                    if pose is not None:
                        with self._lock:
                            self._pose = pose
                            self._stamp = time.monotonic()
                except Exception:
                    pass    # publisher down; latest() ages out on its own
                self._stop_evt.wait(self._period)
        except Exception as exc:
            if not self._warned:
                print(f"[wholebody] SLAM pose listener unavailable: {exc}")
                self._warned = True
        finally:
            try:
                if sub is not None:
                    sub.stop()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Loop timing diagnostics
# ─────────────────────────────────────────────────────────────────────────────

class _LoopTimingMonitor:
    """Tracks a loop's actual wall-clock cadence and reports it periodically.

    Unit tests can only verify that the interpolation *math* is right --
    nothing about GIL contention, other threads on the box, or how long a
    CAN write actually takes shows up in a headless test. This answers the
    question a passing test can't: is this loop actually landing anywhere
    close to its target rate in real time, or is it running behind and
    bursty. Call `tick()` once per loop iteration; it prints a summary at
    most every `report_every_s` seconds once it has samples.
    """

    def __init__(self, name: str, target_hz: float, report_every_s: float = 5.0,
                 window: int = 500):
        self.name = name
        self.target_dt = 1.0 / float(target_hz)
        self.report_every_s = float(report_every_s)
        self._intervals: list[float] = []
        self._window = int(window)
        self._last_t: Optional[float] = None
        self._last_report = 0.0

    def tick(self) -> None:
        now = time.monotonic()
        if self._last_t is not None:
            self._intervals.append(now - self._last_t)
            if len(self._intervals) > self._window:
                self._intervals.pop(0)
        self._last_t = now

        if now - self._last_report < self.report_every_s or len(self._intervals) < 10:
            return
        self._last_report = now
        arr = np.asarray(self._intervals)
        mean_dt, max_dt = float(arr.mean()), float(arr.max())
        jitter_ms = float(arr.std()) * 1000.0
        over_2x = int(np.sum(arr > 2.0 * self.target_dt))
        print(
            f"[wholebody] {self.name} loop: {1.0/mean_dt:5.1f} Hz actual "
            f"(target {1.0/self.target_dt:.0f} Hz), jitter {jitter_ms:5.2f} ms std, "
            f"worst gap {max_dt*1000:6.2f} ms, {over_2x}/{len(arr)} ticks "
            f">2x target period"
        )


class _CommandJitterMonitor:
    """Tracks how much a stream of *commanded* joint vectors moves tick to
    tick, to answer the question _LoopTimingMonitor can't: are the numbers
    we're sending to nerolib themselves smooth, or is there noise in the
    software before it ever reaches the motors?

    This is the split that tells "our commands are noisy" apart from "the
    commands are clean and the arm just isn't damped/tuned for this":
    smooth, monotonic deltas with a near-zero reversal rate reaching the
    motor but a physically jittery arm points at gains/mechanical tuning,
    not software; a high reversal rate or erratic delta magnitude here means
    the software is the one injecting the noise, upstream of any tuning.

    Call `sample(q)` with each new commanded joint vector (7,); it prints a
    per-joint summary at most every `report_every_s` seconds.
    """

    # Deltas smaller than this are treated as "holding still" and excluded
    # from the reversal count, so quantization/float noise while parked
    # doesn't get counted as the arm reversing direction.
    _STILL_EPS_RAD = 1e-4

    def __init__(self, name: str, report_every_s: float = 5.0, window: int = 500):
        self.name = name
        self.report_every_s = float(report_every_s)
        self._window = int(window)
        self._deltas: list[np.ndarray] = []
        self._prev: Optional[np.ndarray] = None
        self._last_report = 0.0

    def sample(self, q) -> None:
        q = np.asarray(q, dtype=float)
        if self._prev is not None:
            self._deltas.append(q - self._prev)
            if len(self._deltas) > self._window:
                self._deltas.pop(0)
        self._prev = q.copy()

        now = time.monotonic()
        if now - self._last_report < self.report_every_s or len(self._deltas) < 10:
            return
        self._last_report = now

        d = np.asarray(self._deltas)  # (N, 7)
        moving = np.abs(d) > self._STILL_EPS_RAD
        # Sign-reversal rate per joint: of the ticks where *both* this step
        # and the previous one were real motion (not noise-floor holding),
        # what fraction flipped direction.
        sign = np.sign(d)
        both_moving = moving[1:] & moving[:-1]
        flipped = (sign[1:] != sign[:-1]) & both_moving
        denom = both_moving.sum(axis=0)
        reversal_rate = np.divide(
            flipped.sum(axis=0), denom, out=np.zeros(7), where=denom > 0
        )
        worst = int(np.argmax(reversal_rate))
        abs_d = np.abs(d)
        print(
            f"[wholebody] {self.name} command deltas: "
            f"mean|Δq| {abs_d.mean()*1000:.2f} mrad, max {abs_d.max()*1000:.1f} mrad, "
            f"worst-reversal joint {worst} (flips {reversal_rate[worst]*100:4.1f}% "
            f"of {int(denom[worst])} moving ticks, "
            f"|Δq| std {abs_d[:, worst].std()*1000:.2f} mrad)"
        )


class _NullSpaceMonitor:
    """Correlates one joint's tick-to-tick motion with how much the
    end-effector actually moved on the same ticks, to tell null-space wobble
    (the joint moves, the EE doesn't) apart from genuine required motion
    (both move together). `_CommandJitterMonitor` repeatedly flags joint
    index 6 (`*_arm_joint7`, the wrist joint furthest from the base and
    least load-bearing for EE tracking -- the "cheapest" DOF for a redundant
    solve to let wander) as the worst-reversal joint on both arms; this
    checks that directly against the solve's own forward kinematics rather
    than inferring it.

    Call `sample(q, T_ee)` once per solve with the *raw* solved arm_q (7,)
    and the resulting mink.SE3 EE pose -- both from the same `solve()` call,
    before any deadband/lookahead clamping, since the question is whether
    the solve itself is noisy, not whether dispatch is. Prints a summary at
    most every `report_every_s` seconds.
    """

    JOINT_INDEX = 6  # arm_joint7

    def __init__(self, name: str, report_every_s: float = 5.0, window: int = 500):
        self.name = name
        self.report_every_s = float(report_every_s)
        self._window = int(window)
        self._dq: list[float] = []
        self._ee_pos_mm: list[float] = []
        self._ee_ori_deg: list[float] = []
        self._prev_q: Optional[np.ndarray] = None
        self._prev_T = None
        self._last_report = 0.0

    def sample(self, q, T_ee) -> None:
        q = np.asarray(q, dtype=float)
        if self._prev_q is not None:
            dq = abs(float(q[self.JOINT_INDEX] - self._prev_q[self.JOINT_INDEX]))
            dp = float(np.linalg.norm(
                T_ee.translation() - self._prev_T.translation()
            )) * 1000.0
            d_rot = self._prev_T.rotation().inverse() @ T_ee.rotation()
            dori = float(np.linalg.norm(d_rot.log())) * (180.0 / np.pi)
            self._dq.append(dq)
            self._ee_pos_mm.append(dp)
            self._ee_ori_deg.append(dori)
            if len(self._dq) > self._window:
                self._dq.pop(0)
                self._ee_pos_mm.pop(0)
                self._ee_ori_deg.pop(0)
        self._prev_q, self._prev_T = q.copy(), T_ee

        now = time.monotonic()
        if now - self._last_report < self.report_every_s or len(self._dq) < 10:
            return
        self._last_report = now

        dq = np.asarray(self._dq)
        dp = np.asarray(self._ee_pos_mm)
        dori = np.asarray(self._ee_ori_deg)
        moving = dq > 1e-4
        n_moving = int(moving.sum())
        if n_moving == 0:
            print(f"[wholebody] {self.name}: joint7 held still over "
                  f"{len(dq)} ticks, nothing to correlate")
            return
        ee_pos_on_move = float(dp[moving].mean())
        ee_ori_on_move = float(dori[moving].mean())
        verdict = (
            "looks like null-space wobble (EE barely moves)"
            if ee_pos_on_move < 0.5 and ee_ori_on_move < 0.3
            else "EE is moving too -- not pure null-space"
        )
        print(
            f"[wholebody] {self.name}: joint7 moved on {n_moving}/{len(dq)} ticks "
            f"(mean {float(dq[moving].mean())*1000:.2f} mrad); EE moved "
            f"{ee_pos_on_move:.3f} mm / {ee_ori_on_move:.3f} deg on those same "
            f"ticks (overall mean {float(dp.mean()):.3f} mm) -- {verdict}"
        )


class _TrajectoryRecorder:
    """Appends raw per-solve-tick data for every subsystem to a CSV.

    The summary monitors above (_LoopTimingMonitor, _CommandJitterMonitor,
    _NullSpaceMonitor) print aggregate stats every few seconds -- useful for
    watching a live session, but not enough to tune anything against
    afterwards. This records the raw per-tick numbers instead. One row per
    solve tick (~30 Hz), written line-buffered.

    Three groups of columns, in the order the data flows:

    * **Arms and solver** -- both arms' solved joint vectors, target and
      achieved EE pose, solver iteration count and convergence flag. This is
      what the null-space work was designed against.
    * **Base** -- the solver's raw world-frame velocity request, the body-frame
      triple left after clamping and the deadband, the vector actually handed
      to the relay, and then the four modules' commanded *and measured* steer
      angles and drive velocities. Commanded-vs-measured is the whole point:
      `robot/base_motor.py` runs the steering open-loop
      (USE_FEEDBACK_FOR_STEER is False), so nothing in the control path ever
      compares the two, and a module that is not reaching its commanded angle
      is currently invisible.
    * **Lift** -- the clamped goal actually servoed to, the measured height,
      the velocity the PD asked for, the PD's own filtered velocity estimate,
      and the feedback age it acted on. Enough to fit lift_kp / lift_kd
      offline rather than by feel.

    Sampling caveat: this runs at the solve rate, so it sees the base at 30 Hz
    while the swerve loop runs at 324 Hz and the SPARKs stream status at 50 Hz.
    That resolves module slew (hundreds of ms) and PID settling fine; it does
    not resolve anything at the swerve loop's own rate.
    """

    def __init__(self, path: Path, ik_config, hw_config=None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered: a crash or Ctrl-C loses at most the in-flight row,
        # not the whole session.
        self._file = path.open("w", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            f"# refresh_posture_target={ik_config.refresh_posture_target}",
            f"arm_posture_cost={ik_config.arm_posture_cost}",
            f"arm_posture_cost_overrides={dict(ik_config.arm_posture_cost_overrides)}",
            f"ee_position_cost={ik_config.ee_position_cost}",
            f"ee_orientation_cost={ik_config.ee_orientation_cost}",
            f"redundancy_resolution={ik_config.redundancy_resolution}",
            f"dls_damping={ik_config.dls_damping}",
            f"nullspace_swivel_weight={ik_config.nullspace_swivel_weight}",
            f"elbow_swivel_gain={ik_config.elbow_swivel_gain}",
            f"elbow_swivel_targets={dict(ik_config.elbow_swivel_targets)}",
            f"nullspace_continuity_weight={ik_config.nullspace_continuity_weight}",
            f"nullspace_posture_weight={ik_config.nullspace_posture_weight}",
            f"enable_manipulability={ik_config.enable_manipulability}",
            f"manipulability_weight={ik_config.manipulability_weight}",
            f"base_motion_weight={ik_config.base_motion_weight}",
            f"base_motion_weight_min={ik_config.base_motion_weight_min}",
            f"base_motion_weight_yaw={ik_config.base_motion_weight_yaw}",
            f"base_weight_gate_on={ik_config.base_weight_gate_on}",
            f"base_weight_gate_full={ik_config.base_weight_gate_full}",
            f"base_recenter_gain={ik_config.base_recenter_gain}",
            f"base_recenter_max_vel={ik_config.base_recenter_max_vel}",
            # Gated experiments -- a run is uninterpretable without these.
            f"nullspace_home_gain={getattr(ik_config, 'nullspace_home_gain', 0.0)}",
            f"nullspace_home_weight={getattr(ik_config, 'nullspace_home_weight', 1.0)}",
            f"constrained_primary={getattr(ik_config, 'constrained_primary', False)}",
            f"dls_task_weighting={getattr(ik_config, 'dls_task_weighting', False)}",
            f"dls_adaptive_damping_sigma={getattr(ik_config, 'dls_adaptive_damping_sigma', 0.0)}",
            f"dls_damping_max={getattr(ik_config, 'dls_damping_max', 0.0)}",
            f"swivel_parallel_ref={getattr(ik_config, 'swivel_parallel_ref', False)}",
            f"swivel_relatch_err_rad={getattr(ik_config, 'swivel_relatch_err_rad', 0.0)}",
        ])
        # Second config line: the base and lift knobs. Separate row because
        # these are what a base/lift tuning session varies, and a run is
        # uninterpretable without knowing which values produced it.
        if hw_config is not None:
            self._writer.writerow([
                f"# control_hz={hw_config.control_hz}",
                f"enable_base_motion={hw_config.enable_base_motion}",
                f"base_max_lin_vel={hw_config.base_max_lin_vel}",
                f"base_max_ang_vel={hw_config.base_max_ang_vel}",
                f"base_vel_deadband={hw_config.base_vel_deadband}",
                f"base_vel_deadband_exit={hw_config.base_vel_deadband_exit}",
                f"base_vel_filter_tau={hw_config.base_vel_filter_tau}",
                f"base_feedback_alpha={hw_config.base_feedback_alpha}",
                f"base_leash_m={hw_config.base_leash_m}",
                f"base_leash_rad={hw_config.base_leash_rad}",
                f"enable_lift_motion={hw_config.enable_lift_motion}",
                f"lift_kp={hw_config.lift_kp}",
                f"lift_kd={hw_config.lift_kd}",
                f"lift_derivative_tau={hw_config.lift_derivative_tau}",
                f"lift_velocity_deadband_m={hw_config.lift_velocity_deadband_m}",
                f"lift_max_velocity_m_s={hw_config.lift_max_velocity_m_s}",
                f"lift_feedback_max_age_s={hw_config.lift_feedback_max_age_s}",
                f"use_measured_lift={hw_config.use_measured_lift}",
                f"use_measured_arm_state={hw_config.use_measured_arm_state}",
                f"base_pid={getattr(hw_config, 'base_pid_provenance', 'unknown')}",
                f"base_heading_rate_limit={hw_config.base_heading_rate_limit}",
                f"base_module_slew_limit={hw_config.base_module_slew_limit}",
                f"base_module_free_speed={hw_config.base_module_free_speed}",
                f"base_max_accel={hw_config.base_max_accel}",
                f"base_yaw_deadband={hw_config.base_yaw_deadband}",
                f"base_yaw_deadband_exit={hw_config.base_yaw_deadband_exit}",
                f"base_yaw_filter_tau={hw_config.base_yaw_filter_tau}",
                f"base_pose_kp_xy={hw_config.base_pose_kp_xy}",
                f"base_pose_kd_xy={hw_config.base_pose_kd_xy}",
                f"base_pose_kp_yaw={hw_config.base_pose_kp_yaw}",
                f"base_pose_kd_yaw={hw_config.base_pose_kd_yaw}",
                f"base_pose_deadband_m={hw_config.base_pose_deadband_m}",
                f"base_pose_yaw_deadband_rad={hw_config.base_pose_yaw_deadband_rad}",
                f"base_pose_derivative_tau={hw_config.base_pose_derivative_tau}",
                f"base_pose_ff_gain={hw_config.base_pose_ff_gain}",
                f"base_pose_ff_max_frac={hw_config.base_pose_ff_max_frac}",
                f"base_pose_ff_tau={hw_config.base_pose_ff_tau}",
                f"arm_joint_deadband_rad={hw_config.arm_joint_deadband_rad}",
                f"target_leash_m={getattr(hw_config, 'target_leash_m', 0.0)}",
                f"target_leash_rad={getattr(hw_config, 'target_leash_rad', 0.0)}",
                # Whether the base PD closed on SLAM or on dead-reckoning
                # alone, and under what frame convention. Without these a
                # finished run does not say which it was: the only trace the
                # correction leaves in the rows is that odometry stops being
                # the integral of the commanded velocity, which has to be
                # reconstructed from base_x/base_y minus the pose error before
                # you can even tell the feature was on. Two runs that differ
                # only in `slam_yaw_sign` are otherwise indistinguishable.
                f"enable_slam_base_pose={hw_config.enable_slam_base_pose}",
                f"slam_yaw_sign={hw_config.slam_yaw_sign}",
                f"slam_correction_max_lin_rate={hw_config.slam_correction_max_lin_rate}",
                f"slam_correction_max_yaw_rate={hw_config.slam_correction_max_yaw_rate}",
                f"slam_pose_max_age_s={hw_config.slam_pose_max_age_s}",
            ])

        header = ["t"]
        header += [f"left_q{i}" for i in range(7)]
        header += [f"right_q{i}" for i in range(7)]
        # wxyz_xyz: quaternion (w,x,y,z) then translation (x,y,z), matching
        # mink.SE3.wxyz_xyz and the convention already used by get_state().
        header += [f"left_target_ee_{i}" for i in range(7)]
        header += [f"right_target_ee_{i}" for i in range(7)]
        header += [f"left_actual_ee_{i}" for i in range(7)]
        header += [f"right_actual_ee_{i}" for i in range(7)]
        header += ["lift_q", "base_x", "base_y", "base_yaw", "iters", "solved"]

        # ── Base ────────────────────────────────────────────────────────────
        # The pose the base was asked for is base_x/base_y/base_yaw above --
        # that column trio is result.base_position, which is now the base
        # command rather than only a diagnostic. From there:
        # err_*:  the pose error the PD acted on, in the chassis frame
        #         (forward, left, yaw). Subtract it from base_x/y/yaw to
        #         recover the dead-reckoned pose.
        # req_*:  the PD's own velocity request, chassis frame, unshaped.
        # body_*: after the clamp, filter and deadband chain, still chassis
        #         frame.
        # sent_*: the 3-vector handed to Base.set_target_base_velocity, in the
        #         axis order BaseAxisMap produces (not forward/lateral/yaw).
        # ff_*:   the reference feedforward the PD added, chassis frame. req_*
        #         already includes it; subtract to see the feedback alone.
        # leash:  metres the base leash clamped off the solver's belief on
        #         this tick -- 0 when it did not bind, nan when the base was
        #         not being driven. Non-zero on most moving ticks means
        #         base_leash_m is too short.
        header += ["base_active",
                   "base_err_fwd", "base_err_lat", "base_err_yaw",
                   "base_req_fwd", "base_req_lat", "base_req_yaw",
                   "base_body_fwd", "base_body_lat", "base_body_yaw",
                   "base_sent_0", "base_sent_1", "base_sent_2",
                   "base_leash",
                   "base_ff_fwd", "base_ff_lat", "base_ff_yaw",
                   # Fraction of the shaped step SwerveTwistShaper allowed
                   # this tick: 1.0 when the modules could follow, less when
                   # the twist was walked toward the request instead of
                   # jumping to it. Anything persistently below 1 is the
                   # drive/spin transition being paid for over several ticks.
                   "base_slew_scale"]
        # What Base itself was holding, so a mismatch against base_sent_*
        # localises to the relay or to a manual override stealing the base.
        header += ["swerve_enabled",
                   "swerve_target_0", "swerve_target_1", "swerve_target_2",
                   "swerve_prof_0", "swerve_prof_1", "swerve_prof_2"]
        # Per module, in MODULE_ORDER (FL, FR, RR, RL).
        header += [f"steer_cmd_{m}" for m in _MODULE_LABELS]
        header += [f"steer_meas_{m}" for m in _MODULE_LABELS]
        header += [f"drive_cmd_{m}" for m in _MODULE_LABELS]
        header += [f"drive_meas_{m}" for m in _MODULE_LABELS]
        # Cumulative motor rotations -- differentiate offline for per-module
        # distance, or feed swerve_odom.py's forward model directly.
        header += [f"drive_pos_{m}" for m in _MODULE_LABELS]

        # ── Lift ────────────────────────────────────────────────────────────
        header += ["lift_active", "lift_mode", "lift_goal", "lift_meas",
                   "lift_cmd_vel", "lift_vel_est", "lift_age", "lift_blocked"]

        # ── Solver diagnostics ([S7]) ────────────────────────────────────────
        # Per-arm conditioning and swivel state plus the number of active
        # collision-constraint rows this tick. NaN when the solver did not
        # produce a value (diagnostics disabled, swivel faded out, or soft
        # mode, which never counts constraint rows). These are the columns
        # that turn "the arm got stuck" into a mechanism: sigma_min ->
        # singularity, manip -> reach, swivel_err -> fought branch,
        # collision_rows -> constraint contact.
        header += ["l_sigma_min", "r_sigma_min", "l_manip", "r_manip",
                   "l_swivel", "l_swivel_tgt", "l_swivel_err",
                   "r_swivel", "r_swivel_tgt", "r_swivel_err",
                   "collision_rows"]

        self._writer.writerow(header)
        self._t0 = time.monotonic()
        self.path = path

    @staticmethod
    def _f(value) -> str:
        """Format one float, keeping a missing reading missing.

        None and NaN both become "nan" rather than 0.0. A dropped CAN frame
        must not read back later as a module sitting at zero.
        """
        if value is None:
            return "nan"
        value = float(value)
        return "nan" if not math.isfinite(value) else f"{value:.6f}"

    def _vec(self, values, n: int) -> list:
        if values is None:
            return ["nan"] * n
        arr = np.asarray(values, dtype=float).ravel()
        return [self._f(arr[i]) if i < arr.size else "nan" for i in range(n)]

    def record(
        self, left_q, right_q, T_l_target, T_r_target, T_l_actual, T_r_actual,
        lift_q: float, base_xytheta, iters: int, solved: bool,
        base=None, swerve=None, lift=None, solver_diag=None,
    ) -> None:
        row = [time.monotonic() - self._t0]
        row += [f"{v:.6f}" for v in np.asarray(left_q, dtype=float)]
        row += [f"{v:.6f}" for v in np.asarray(right_q, dtype=float)]
        row += [f"{v:.6f}" for v in T_l_target.wxyz_xyz]
        row += [f"{v:.6f}" for v in T_r_target.wxyz_xyz]
        row += [f"{v:.6f}" for v in T_l_actual.wxyz_xyz]
        row += [f"{v:.6f}" for v in T_r_actual.wxyz_xyz]
        base_xytheta = np.asarray(base_xytheta, dtype=float)
        row += [
            f"{float(lift_q):.6f}",
            f"{base_xytheta[0]:.6f}", f"{base_xytheta[1]:.6f}", f"{base_xytheta[2]:.6f}",
            str(int(iters)), str(bool(solved)),
        ]

        base = base or {}
        row += [str(bool(base.get("active", False)))]
        row += self._vec(base.get("err"), 3)
        row += self._vec(base.get("req"), 3)
        row += self._vec(base.get("body"), 3)
        row += self._vec(base.get("sent"), 3)
        row += [self._f(base.get("leash"))]
        row += self._vec(base.get("ff"), 3)
        row += self._vec(base.get("slew_scale"), 1)

        swerve = swerve or {}
        row += [str(bool(swerve.get("motors_enabled", False)))]
        row += self._vec(swerve.get("v_target"), 3)
        row += self._vec(swerve.get("v_profiled"), 3)
        for key in ("steer_cmd_rad", "steer_meas_rad", "drive_cmd_mps",
                    "drive_meas_raw", "drive_pos_rot"):
            row += self._vec(swerve.get(key), _NUM_SWERVES)

        lift = lift or {}
        row += [
            str(bool(lift.get("active", False))),
            str(lift.get("mode", "off")),
            self._f(lift.get("goal")),
            self._f(lift.get("meas")),
            self._f(lift.get("cmd_vel")),
            self._f(lift.get("vel_est")),
            self._f(lift.get("age")),
            str(bool(lift.get("blocked", False))),
        ]

        diag = solver_diag or {}
        row += [self._f(diag.get(key)) for key in (
            "left_sigma_min", "right_sigma_min", "left_manip", "right_manip",
            "left_swivel", "left_swivel_target", "left_swivel_err",
            "right_swivel", "right_swivel_target", "right_swivel_err",
            "collision_rows")]

        self._writer.writerow(row)

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class WholeBodyController:
    """Runs whole-body IK against the real arms, lift and swerve base.

    Parameters
    ----------
    left_arm, right_arm : ArmNode
        Initialised arm nodes (``ArmNode.init()`` already called).
    base : robot.base_motor.Base
        Owns the swerve modules and the PicoLift.
    base_controller : robot.base.BaseController, optional
        When given, base velocities are routed through it (so its mode
        bookkeeping stays correct) instead of straight to ``base``.
    """

    def __init__(
        self,
        left_arm,
        right_arm,
        base,
        base_controller=None,
        scene_xml: Optional[str] = None,
        config: Optional[WholeBodyHardwareConfig] = None,
        ik_config: Optional[WholeBodyIKConfig] = None,
    ) -> None:
        if left_arm is None or right_arm is None:
            raise ValueError("whole-body control needs both arms; run YOR with no_arms=False")

        self.config = config or WholeBodyHardwareConfig()
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.base = base
        self.base_controller = base_controller

        self.dt = 1.0 / self.config.control_hz
        if ik_config is None:
            # The tuned weights from the interactive demo: base reluctant, lift
            # eager. Without these the chassis rolls in preference to
            # stretching the lift, which is the wrong behaviour on hardware.
            ik_config = WholeBodyIKConfig(
                dt=self.dt,
                solver="pyqpmad",
                max_iters=10,
                base_posture_cost=1e-1,
                lift_posture_cost=1e-4,
                arm_posture_cost=1e-3,
            )
        self.ik = WholeBodyIK(scene_xml=scene_xml, config=ik_config)

        # ── Targets / shared state ──────────────────────────────────────────
        self._lock = threading.Lock()
        self.left_ee_target: Optional[mink.SE3] = None
        self.right_ee_target: Optional[mink.SE3] = None
        self.lift_target: Optional[float] = None
        # Last gripper value each arm was commanded. The grippers themselves
        # report through the arm, but nothing else remembers what was *asked*
        # for, and a recorded episode needs the command as its action label.
        self.left_gripper_target: Optional[float] = None
        self.right_gripper_target: Optional[float] = None
        self._home_left: Optional[mink.SE3] = None
        self._home_right: Optional[mink.SE3] = None
        self._home_lift: float = 0.0

        self.odometry = BaseOdometry()
        # SLAM drift correction — inert unless enable_slam_base_pose is set.
        self.slam_frame = SlamBaseFrame(
            yaw_sign=self.config.slam_yaw_sign,
            heading_offset=self.config.base_pose_heading_offset,
        )
        self.slam_pose: Optional[SlamPoseListener] = None
        self._slam_offset_m = 0.0
        self._slam_pose_age = float("inf")
        self._slam_offset_warned = False
        if self.config.enable_slam_base_pose:
            self.slam_pose = SlamPoseListener(
                self.config.slam_pose_host,
                self.config.slam_pose_port,
                self.config.slam_pose_hz,
            )
        # The chassis is driven by pose: the solver says where the base
        # should be and this closes on it. It is handed `self.base` so it can
        # stand alone, but this file never lets it send -- `_dispatch_base`
        # takes its request through the shaping chain and out via
        # `_send_base_command`, so the BASE_VEL relay stays the only writer.
        if BasePoseController is None:  # pragma: no cover - CAN stack absent
            raise ImportError(
                "robot.base could not be imported; whole-body base control "
                "needs BasePoseController"
            )
        self.base_pose = BasePoseController(
            base=self.base,
            kp_xy=self.config.base_pose_kp_xy,
            kd_xy=self.config.base_pose_kd_xy,
            kp_yaw=self.config.base_pose_kp_yaw,
            kd_yaw=self.config.base_pose_kd_yaw,
            xy_deadband=self.config.base_pose_deadband_m,
            yaw_deadband=self.config.base_pose_yaw_deadband_rad,
            max_lin_vel=self.config.base_max_lin_vel,
            max_ang_vel=self.config.base_max_ang_vel,
            derivative_tau=self.config.base_pose_derivative_tau,
            heading_offset=self.config.base_pose_heading_offset,
            ff_gain=self.config.base_pose_ff_gain,
            ff_max_frac=self.config.base_pose_ff_max_frac,
            ff_tau=self.config.base_pose_ff_tau,
        )
        self.lift_pd = LiftVelocityPD(
            kp=self.config.lift_kp,
            kd=self.config.lift_kd,
            tau=self.config.lift_derivative_tau,
            deadband=self.config.lift_velocity_deadband_m,
            max_velocity=self.config.lift_max_velocity_m_s,
            max_gap_s=self.config.lift_control_gap_s,
        )

        self._last_base_velocity = np.zeros(3)   # world frame, as commanded
        self._last_base_command = np.zeros(3)    # what Base was actually sent
        self._yaw_filt = 0.0          # low-passed yaw request (rad/s)
        self._yaw_active = False      # hysteresis state: currently rotating?
        self._lin_filt = (0.0, 0.0)   # low-passed linear request (fwd, lat, m/s)
        self._lin_active = False      # hysteresis state: currently translating?
        # Per-tick dispatch telemetry for _TrajectoryRecorder. Written by the
        # dispatch methods, read once at the end of the same _step, so a logged
        # row always describes the tick that produced it rather than the
        # previous one.
        self._base_dispatch: dict = {}
        # Reference for the heading rate limiter; None until the base first
        # moves, then frozen across stops. See _limit_heading_rate.
        self._base_heading: Optional[float] = None
        # Last dispatched linear velocity, for the acceleration limiter.
        self._base_vel_prev: tuple[float, float] = (0.0, 0.0)
        # Metres the base leash clamped off the solver's belief this tick, for
        # the trajectory log. Rewritten every tick by _leash_base.
        self._base_leash: float = 0.0
        # Final stage of the base shaping chain: bounds the rate of change of
        # the four *module angles*, which is the thing the wheels have to
        # serve and the thing none of the per-component limiters above can
        # see. See SwerveTwistShaper.
        self._twist_shaper = SwerveTwistShaper(
            slew_limit=self.config.base_module_slew_limit,
            free_speed=self.config.base_module_free_speed,
            free_ratio=self.config.base_module_free_ratio,
        )
        self._lift_dispatch: dict = {}
        # The hardware sync already reads both arms once per control tick. Keep
        # those samples for dispatch instead of making two more blocking CAN
        # reads after solving.
        self._measured_left_q: Optional[np.ndarray] = None
        self._measured_right_q: Optional[np.ndarray] = None
        self._commanded_left_q: Optional[np.ndarray] = None
        self._commanded_right_q: Optional[np.ndarray] = None
        # Arm dispatch runs on its own thread/rate, decoupled from the 30 Hz
        # solve loop -- see _dispatch_arms / _arm_dispatch_tick. `_arm_seg_*`
        # is the interpolation state bridging the two: `_dispatch_arms` (30 Hz)
        # only ever writes `_arm_seg_goal`/`_arm_seg_revision` under the lock;
        # `_arm_dispatch_tick` (90 Hz) owns everything else and needs no lock
        # for it, since only that one thread ever touches it.
        self._arm_dispatch_lock = threading.Lock()
        self._arm_seg_goal: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._arm_seg_revision = {"left": 0, "right": 0}
        self._arm_seg_active_revision = {"left": -1, "right": -1}
        self._arm_seg_start: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._arm_seg_current: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._arm_seg_step = {"left": 0, "right": 0}
        self._last_lift_command: Optional[str] = None
        # Streamed-velocity bookkeeping. `_lift_driving_since` is what makes the
        # staleness check fair across the firmware's driver-startup delay.
        self._lift_velocity_mode: Optional[bool] = None
        self._lift_cmd_velocity = 0.0
        self._lift_driving_since: Optional[float] = None
        # Latched by a refusal, cleared only by feedback that is actually
        # fresh. Without the latch a refusal would clear its own precondition
        # and the lift would run in bursts against a dead link.
        self._lift_feedback_blocked = False
        self._lift_refresh_requested = 0.0
        self._last_solve_ok = False
        self._solve_error: Optional[str] = None

        self._manual_base_until = 0.0
        self._manual_lift_until = 0.0
        self._manual_arm_until = 0.0
        self._lift_unavailable_warned = False

        self._thread: Optional[threading.Thread] = None
        self._arm_thread: Optional[threading.Thread] = None
        self._running = False
        self.initialized = False
        self._solve_timing = _LoopTimingMonitor("solve (30 Hz)", self.config.control_hz)
        self._arm_dispatch_timing = _LoopTimingMonitor(
            "arm dispatch (90 Hz)", self.config.arm_dispatch_hz
        )
        # Goal-level jitter is the whole-body solve's own output (per 30 Hz
        # tick, before interpolation); dispatch-level is what actually reaches
        # nerolib (per 90 Hz sub-step). Comparing the two localizes noise: if
        # goal-level is already jittery, look at the solve / EE target; if
        # goal-level is clean but dispatch-level isn't, look at the
        # interpolation; if both are clean and the arm still shakes, that's
        # gains/mechanical tuning, not software.
        self._goal_jitter = {
            "left": _CommandJitterMonitor("left goal (30 Hz)"),
            "right": _CommandJitterMonitor("right goal (30 Hz)"),
        }
        self._dispatch_jitter = {
            "left": _CommandJitterMonitor("left dispatch (90 Hz)"),
            "right": _CommandJitterMonitor("right dispatch (90 Hz)"),
        }
        self._nullspace_monitor = {
            "left": _NullSpaceMonitor("left arm"),
            "right": _NullSpaceMonitor("right arm"),
        }
        self._trajectory_recorder: Optional[_TrajectoryRecorder] = None
        if self.config.record_trajectories:
            traj_dir = (
                Path(__file__).resolve().parent.parent
                / "artifacts" / "wholebody_logs" / "trajectories"
            )
            traj_path = traj_dir / f"traj_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            self._trajectory_recorder = _TrajectoryRecorder(
                traj_path, self.ik.config, self.config)
            print(f"[wholebody] recording trajectories to {traj_path}")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def init(self) -> None:
        """Seed the model from the robot's actual state and latch home poses."""
        self.ik.init_from_keyframe("home")

        T_l, T_r = self.ik.forward_kinematics()
        self._home_left, self._home_right = T_l.copy(), T_r.copy()
        self._home_lift = float(self.ik.configuration.q[self.ik._lift_qpos_adr])

        self.odometry.reset()
        # The odometry origin just moved, so any earlier SLAM alignment is void.
        self.slam_frame.reset()
        # Open-loop mode still needs one encoder snapshot so its initial model
        # and first command start at the robot's actual pose.
        self._sync_from_hardware(force_arm_read=True)
        self._commanded_left_q = self._measured_left_q.copy()
        self._commanded_right_q = self._measured_right_q.copy()
        # Arm dispatch interpolates from wherever it last actually sent a
        # command -- seed that with the same measured pose so the very first
        # segment is smoothed too, instead of jumping straight to the goal.
        self._arm_seg_current["left"] = self._commanded_left_q.copy()
        self._arm_seg_current["right"] = self._commanded_right_q.copy()

        T_l, T_r = self.ik.forward_kinematics()
        with self._lock:
            self.left_ee_target = T_l.copy()
            self.right_ee_target = T_r.copy()
            self.lift_target = None
        self.initialized = True
        print(
            f"[wholebody] initialised — lift travel {self.ik.lift_range[0]:.3f}"
            f"…{self.ik.lift_range[1]:.3f} m, "
            f"{self.ik.n_collision_pairs} collision pairs, "
            f"base motion {'ON' if self.config.enable_base_motion else 'OFF'}"
        )
        alpha = float(self.config.base_feedback_alpha)
        if alpha > 0.0 and self.config.enable_base_motion:
            kp_dt = self.config.base_pose_kp_xy * self.dt
            print(
                f"[wholebody] base: base_feedback_alpha={alpha} pulls the pose "
                f"target back toward odometry, so the base will settle at "
                f"{100.0 * kp_dt / (alpha + kp_dt):.0f}% of the speed the "
                f"solver asks for — see base_feedback_alpha"
            )
        if self.config.enable_slam_base_pose:
            print(
                f"[wholebody] base: pose loop closes on the Odin SLAM fix "
                f"({self.config.slam_pose_host}:{self.config.slam_pose_port}, "
                f"yaw_sign={self.config.slam_yaw_sign:+.0f}) — watch "
                f"slam_base_correction_m in get_state()"
            )
            if alpha > 0.0:
                # The correction rates are sized for a PD measurement, which
                # is several times faster than the IK configuration wants to
                # be moved. At alpha 0 that is free, because odometry does not
                # reach the configuration at all; above 0 it does, and a loop
                # closure then arrives at the solver as an end-effector jump.
                print(
                    f"[wholebody] base: base_feedback_alpha={alpha} feeds the "
                    f"SLAM-corrected odometry into the IK configuration at "
                    f"{self.config.slam_correction_max_lin_rate:.2f} m/s — a "
                    f"loop closure will move the EE targets. Use alpha 0, or "
                    f"lower slam_correction_max_lin_rate"
                )

    def start(self) -> None:
        if self._thread is not None:
            # emergency_stop() can race with a restart request: the worker may
            # still be unwinding even though _running is already false. Do not
            # mistake that stale thread object for an active control loop.
            if self._running and self._thread.is_alive():
                return
            if self._thread is not threading.current_thread():
                self._thread.join(timeout=2.0)
            self._thread = None
        if self._arm_thread is not None:
            if self._arm_thread is not threading.current_thread():
                self._arm_thread.join(timeout=2.0)
            self._arm_thread = None
        if not self.initialized:
            self.init()
        self._running = True
        self._thread = threading.Thread(
            target=self._control_loop, name="wholebody-control", daemon=True
        )
        self._arm_thread = threading.Thread(
            target=self._arm_dispatch_loop, name="wholebody-arm-dispatch", daemon=True
        )
        self._thread.start()
        self._arm_thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._running = False
        self._thread.join(timeout=2.0)
        self._thread = None
        if self._arm_thread is not None:
            self._arm_thread.join(timeout=2.0)
            self._arm_thread = None
        self._halt_base()
        self._halt_lift()
        if self.slam_pose is not None:
            self.slam_pose.stop()
            self.slam_pose = None
        if self._trajectory_recorder is not None:
            self._trajectory_recorder.close()
            self._trajectory_recorder = None

    def emergency_stop(self) -> None:
        """Stop the loop and freeze every actuator where it stands."""
        self._running = False
        # A stopped worker must not remain attached: start() uses _thread to
        # distinguish an already-running loop from one that needs launching.
        # Leaving it here made resume_wholebody() report success while no
        # control loop existed, so later EE targets were silently ignored.
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        # Join the arm-dispatch thread too, and before the hold-in-place
        # commands below, so it cannot race those with one more stale
        # interpolated step.
        arm_thread = self._arm_thread
        if arm_thread is not None and arm_thread is not threading.current_thread():
            arm_thread.join(timeout=2.0)
        self._arm_thread = None
        self._halt_base()
        self._halt_lift()
        for arm in (self.left_arm, self.right_arm):
            try:
                arm.set_joint_target(arm.get_joint_positions(), preview_time=0.0)
            except Exception as exc:  # a stop path must never raise
                print(f"[wholebody] e-stop: arm hold failed: {exc}")
        if self._trajectory_recorder is not None:
            self._trajectory_recorder.close()
            self._trajectory_recorder = None

    # ── Target API (mirrors robot/yor_mujoco.py so one client drives both) ───

    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: Optional[float] = None,
                           preview_time: float = 0.0) -> None:
        with self._lock:
            self.left_ee_target = ee_target
            if gripper_target is not None:
                self.left_gripper_target = float(gripper_target)
        if gripper_target is not None:
            self._set_gripper(self.left_arm, gripper_target)

    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: Optional[float] = None,
                            preview_time: float = 0.0) -> None:
        with self._lock:
            self.right_ee_target = ee_target
            if gripper_target is not None:
                self.right_gripper_target = float(gripper_target)
        if gripper_target is not None:
            self._set_gripper(self.right_arm, gripper_target)

    def set_bimanual_ee_target(
        self,
        L_ee_target: mink.SE3, R_ee_target: mink.SE3,
        L_gripper_target: Optional[float] = None, R_gripper_target: Optional[float] = None,
        L_preview_time: float = 0.0, R_preview_time: float = 0.0,
    ) -> None:
        with self._lock:
            self.left_ee_target = L_ee_target
            self.right_ee_target = R_ee_target
            if L_gripper_target is not None:
                self.left_gripper_target = float(L_gripper_target)
            if R_gripper_target is not None:
                self.right_gripper_target = float(R_gripper_target)
        if L_gripper_target is not None:
            self._set_gripper(self.left_arm, L_gripper_target)
        if R_gripper_target is not None:
            self._set_gripper(self.right_arm, R_gripper_target)

    def set_lift_target(self, lift_target: float) -> None:
        with self._lock:
            self.lift_target = self.ik.clamp_lift(float(lift_target))

    def home_left_arm(self) -> None:
        with self._lock:
            self.left_ee_target = self._home_left.copy()

    def home_right_arm(self) -> None:
        with self._lock:
            self.right_ee_target = self._home_right.copy()

    def lift_home(self) -> None:
        with self._lock:
            self.lift_target = self._home_lift

    def toggle_fix_base(self, fixed: Optional[bool] = None) -> bool:
        return self.ik.toggle_fix_base(fixed)

    def toggle_collision_avoidance(self, enable: Optional[bool] = None) -> bool:
        return self.ik.toggle_collision_avoidance(enable)

    def toggle_base_motion(self, enable: Optional[bool] = None) -> bool:
        """Enable/disable dispatch of the solver's base pose to the wheels."""
        self.config.enable_base_motion = (
            (not self.config.enable_base_motion) if enable is None else bool(enable)
        )
        if not self.config.enable_base_motion:
            self._halt_base()
        return self.config.enable_base_motion

    @property
    def _base_authority(self) -> bool:
        """Is the whole-body loop the thing driving the chassis right now?

        `_sync_from_hardware` and `_dispatch_base` must agree on this to the
        tick. If dispatch stands down while sync still lets the solver's base
        belief free-run, the belief keeps travelling while the robot is being
        driven by someone else, and the first tick that takes authority back
        measures the whole divergence as pose error and drives at it.
        """
        return bool(
            self.config.enable_base_motion
            and not self.ik.fix_base
            and not self.base_manually_overridden
        )

    def relatch_elbow_swivel(self, side: Optional[str] = None) -> bool:
        """[S5c] Accept the elbow branch each arm is currently in.

        Clears the latched swivel target(s); the next solve re-latches from
        the live pose. The cheap "fix my elbow" recovery -- until now the
        only way out of a fought branch was a full homing cycle.
        """
        sides = ("left", "right") if side is None else (str(side),)
        for s in sides:
            self.ik.set_elbow_swivel_target(s, None)
        print(f"[wholebody] elbow swivel re-latch requested: {', '.join(sides)}")
        return True

    # ── Manual-override hooks ────────────────────────────────────────────────

    def notify_manual_base_command(self) -> None:
        """Called by the direct base API; suspends whole-body base authority."""
        self._manual_base_until = time.monotonic() + self.config.manual_override_timeout_s

    def notify_manual_lift_command(self) -> None:
        """Called by the direct lift API; suspends whole-body lift authority."""
        self._manual_lift_until = time.monotonic() + self.config.manual_override_timeout_s

    def notify_manual_arm_command(self) -> None:
        """Called by the direct joint-space API; suspends whole-body arm authority.

        The suspension is a short window, not a mode: once it lapses the loop
        resumes driving the arms back toward their EE targets. Stop the loop
        instead if you want a joint-space pose to persist (e.g. tucking).
        """
        self._manual_arm_until = time.monotonic() + self.config.manual_override_timeout_s

    @property
    def base_manually_overridden(self) -> bool:
        return time.monotonic() < self._manual_base_until

    @property
    def lift_manually_overridden(self) -> bool:
        return time.monotonic() < self._manual_lift_until

    @property
    def arms_manually_overridden(self) -> bool:
        return time.monotonic() < self._manual_arm_until

    # ── Queries ──────────────────────────────────────────────────────────────

    def forward_kinematics(self) -> tuple[mink.SE3, mink.SE3]:
        return self.ik.forward_kinematics()

    def get_left_ee_pose(self) -> mink.SE3:
        return self.ik.forward_kinematics()[0]

    def get_right_ee_pose(self) -> mink.SE3:
        return self.ik.forward_kinematics()[1]

    def get_base_velocity(self) -> np.ndarray:
        with self._lock:
            return self._last_base_velocity.copy()

    def get_lift_position(self) -> float:
        height = self._measured_lift()
        if height is None:
            return float(self.ik.configuration.q[self.ik._lift_qpos_adr])
        return float(height)

    def get_state(self) -> dict:
        """Plain-type snapshot, matching YORMujoco.get_state()."""
        T_l, T_r = self.ik.forward_kinematics()
        q = self.ik.configuration.q
        # Written whole each tick by the control thread, so one read is one
        # tick's worth -- no lock needed, and no half-updated dict.
        dispatch = self._base_dispatch or {}
        pose_target = dispatch.get("target")
        pose_error = dispatch.get("err")
        with self._lock:
            base_vel = self._last_base_velocity.copy()
            base_cmd = self._last_base_command.copy()
            left_ee_target = self.left_ee_target
            right_ee_target = self.right_ee_target
            lift_target = self.lift_target
            left_gripper_target = self.left_gripper_target
            right_gripper_target = self.right_gripper_target
        return {
            "left_ee_wxyz_xyz": T_l.wxyz_xyz.tolist(),
            "right_ee_wxyz_xyz": T_r.wxyz_xyz.tolist(),
            "lift": self.get_lift_position(),
            "base_xytheta": self.odometry.pose.tolist(),
            "base_velocity": base_vel.tolist(),
            "base_command": base_cmd.tolist(),
            # What the base is being driven *to* and how far off it is
            # (chassis frame: forward, left, yaw). Without these the pose loop
            # is invisible from outside -- a base that is not moving looks the
            # same whether it has arrived or has lost authority.
            "base_pose_target": (
                None if pose_target is None
                else np.asarray(pose_target, dtype=float).tolist()
            ),
            "base_pose_error": (
                None if pose_error is None
                else np.asarray(pose_error, dtype=float).tolist()
            ),
            "fix_base": self.ik.fix_base,
            "fix_lift": self.ik.fix_lift,
            "collision_avoidance": self.ik.avoid_collisions,
            "base_motion_enabled": self.config.enable_base_motion,
            "lift_motion_enabled": self.config.enable_lift_motion,
            "solved": self._last_solve_ok,
            "solve_error": self._solve_error,
            "left_joint_positions": q[self.ik._left_arm_qpos_adrs].tolist(),
            "right_joint_positions": q[self.ik._right_arm_qpos_adrs].tolist(),
            # SLAM drift correction. `slam_base_pose_age` is None when the
            # feature is off; a steadily growing `slam_base_correction_m` while
            # driving is the signature of a wrong slam_yaw_sign.
            "slam_base_pose_age": (
                None if self.slam_pose is None or not np.isfinite(self._slam_pose_age)
                else float(self._slam_pose_age)
            ),
            "slam_base_correction_m": float(self._slam_offset_m),
            # Lift dispatch, for diagnosis: which path is live, what it last
            # asked for, and how old the height it acted on was.
            "lift_velocity_mode": bool(self._lift_velocity_mode),
            "lift_command_velocity": float(self._lift_cmd_velocity),
            "lift_feedback_age_s": self._lift_feedback_age(),
            # What was last *asked* for, alongside what the robot is actually
            # doing. Recorded episodes need the command as the action label;
            # the measured pose above is the observation it produced. None
            # means no command of that kind has arrived since initialise.
            "left_ee_target_wxyz_xyz": (
                None if left_ee_target is None else left_ee_target.wxyz_xyz.tolist()
            ),
            "right_ee_target_wxyz_xyz": (
                None if right_ee_target is None else right_ee_target.wxyz_xyz.tolist()
            ),
            "lift_target": None if lift_target is None else float(lift_target),
            "left_gripper_target": left_gripper_target,
            "right_gripper_target": right_gripper_target,
        }

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        rate = RateLimiter(self.config.control_hz, warn=False)
        while self._running:
            self._solve_timing.tick()
            try:
                self._step()
            except Exception as exc:
                # One bad cycle must not take the robot down: stop the wheels,
                # keep the arms where they are, and try again next tick.
                self._solve_error = repr(exc)
                self._last_solve_ok = False
                self._halt_base()
                print(f"[wholebody] control step failed: {exc}")
            rate.sleep()
        self._running = False

    def _step(self) -> None:
        with self._lock:
            T_l = self.left_ee_target
            T_r = self.right_ee_target
            lift_tgt = self.lift_target
        if T_l is None or T_r is None:
            return

        self._correct_base_from_slam()
        self._sync_from_hardware()
        T_l, T_r = self._leash_targets(T_l, T_r)
        result = self.ik.solve(T_l, T_r, lift_target=lift_tgt)
        self._last_solve_ok = bool(result.solved)
        self._solve_error = None

        # Raw solve output, before any deadband/lookahead clamping -- see
        # _NullSpaceMonitor for why this has to be the solve's own numbers,
        # not what dispatch ends up sending.
        T_l_actual, T_r_actual = self.ik.forward_kinematics()
        self._nullspace_monitor["left"].sample(result.left_arm_q, T_l_actual)
        self._nullspace_monitor["right"].sample(result.right_arm_q, T_r_actual)

        # After the monitor, which wants the solve's own numbers, and before
        # the base target is read out of `result` by dispatch and the log.
        self._leash_base(result)

        # A subsystem that is switched off still gets a row, marked inactive,
        # rather than carrying the last active tick's numbers forward.
        self._base_dispatch = {"active": False,
                               "target": np.asarray(result.base_position, dtype=float)}
        self._lift_dispatch = {"active": False, "mode": "off"}

        if self.config.enable_arm_motion:
            self._dispatch_arms(result)
        if self.config.enable_lift_motion:
            self._dispatch_lift(result)
        self._dispatch_base(result)

        # Recorded last, so base and lift dispatch have run and the row holds
        # the solve *and* what it turned into on this same tick.
        if self._trajectory_recorder is not None:
            self._trajectory_recorder.record(
                result.left_arm_q, result.right_arm_q,
                T_l, T_r, T_l_actual, T_r_actual,
                result.lift_q, result.base_position,
                result.iters, result.solved,
                base=self._base_dispatch,
                swerve=self._swerve_telemetry(),
                lift=self._lift_dispatch,
                solver_diag=getattr(self.ik, "diagnostics", None),
            )

    def _leash_base(self, result) -> None:
        """Pull the solver's belief of the base pose back onto the leash.

        The reference is the dead-reckoned pose, so what this bounds is how
        far the solver's chassis can run ahead of the chassis the wheels were
        actually asked for. See `base_leash_m` for why that gap grows and what
        it costs; the short version is that an unbounded belief makes the
        solver retract the arms for base motion that never happened.

        The clamped pose replaces both the target dispatch reads and the pose
        held in the IK configuration, so the next solve starts from the
        leashed value and the excess is *forgotten* rather than banked. That
        write-back is the whole mechanism: without it this would rate-limit
        the symptom and leave the belief drifting.

        Translation is clamped on the magnitude of the xy error rather than
        per axis, so the direction the solver asked for survives it -- the
        same reasoning as `_limit_linear` and `BaseOdometry.apply_correction`.

        No-op when both gates are 0, and while the base is not being driven:
        `fix_base` already stops the belief moving, and a disabled or
        manually overridden base freezes odometry, which is then not a
        reference worth leashing against.
        """
        self._base_leash = 0.0

        if (not self.config.enable_base_motion
                or self.ik.fix_base
                or self.base_manually_overridden):
            return

        leash_m = float(self.config.base_leash_m)
        leash_rad = float(self.config.base_leash_rad)
        if leash_m <= 0.0 and leash_rad <= 0.0:
            return

        belief = np.asarray(result.base_position, dtype=float).reshape(-1)[:3]
        ref = self.odometry.pose
        leashed = belief.copy()
        changed = False

        if leash_m > 0.0:
            dx = float(belief[0] - ref[0])
            dy = float(belief[1] - ref[1])
            dist = math.hypot(dx, dy)
            if dist > leash_m:
                scale = leash_m / dist
                leashed[0] = ref[0] + dx * scale
                leashed[1] = ref[1] + dy * scale
                self._base_leash = dist - leash_m
                changed = True

        if leash_rad > 0.0:
            dth = _wrap_pi(float(belief[2]) - float(ref[2]))
            if abs(dth) > leash_rad:
                leashed[2] = _wrap_pi(float(ref[2])
                                      + math.copysign(leash_rad, dth))
                changed = True

        if changed:
            result.base_position = leashed
            self.ik.set_measured_state(base=leashed)

    def _leash_targets(self, T_l: mink.SE3, T_r: mink.SE3) -> tuple[mink.SE3, mink.SE3]:
        """[S2] Pull each EE target back onto the leash around the current pose.

        Translation is clamped to `target_leash_m` of the solver's own FK
        (the pose the solve actually starts from), orientation to
        `target_leash_rad` along the geodesic. The leashed pose replaces the
        stored target under the lock, so wind-up is *forgotten*, not merely
        rate-limited -- when the operator stops pushing into a constraint,
        the pending error is the leash length, not the accumulated stream.
        A concurrent RPC write can lose at most one 30 Hz tick to the
        write-back (its own next tick overwrites again). No-op when both
        gates are 0.
        """
        leash_m = float(self.config.target_leash_m)
        leash_rad = float(self.config.target_leash_rad)
        if leash_m <= 0.0 and leash_rad <= 0.0:
            return T_l, T_r

        T_l_fk, T_r_fk = self.ik.forward_kinematics()
        out: list[mink.SE3] = []
        changed = False
        for T_tgt, T_fk in ((T_l, T_l_fk), (T_r, T_r_fk)):
            p_new = T_tgt.translation()
            if leash_m > 0.0:
                d = p_new - T_fk.translation()
                dist = float(np.linalg.norm(d))
                if dist > leash_m:
                    p_new = T_fk.translation() + d * (leash_m / dist)
                    changed = True
            R_new = T_tgt.rotation()
            if leash_rad > 0.0:
                aa = (T_fk.rotation().inverse() @ R_new).log()
                ang = float(np.linalg.norm(aa))
                if ang > leash_rad:
                    R_new = T_fk.rotation() @ mink.SO3.exp(aa * (leash_rad / ang))
                    changed = True
            out.append(mink.SE3.from_rotation_and_translation(R_new, p_new))

        if changed:
            with self._lock:
                self.left_ee_target = out[0]
                self.right_ee_target = out[1]
        return out[0], out[1]

    def _measured_module_angles(self) -> Optional[np.ndarray]:
        """Absolute steering angles, or None when the base cannot report them.

        Only used to re-seed `SwerveTwistShaper` after a halt or an authority
        change. Optional for the same reason `_swerve_telemetry` is: `Base` is
        stubbed in tests, and a missing reading must cost the shaper its seed,
        not the tick.
        """
        snapshot = self._swerve_telemetry()
        if not snapshot:
            return None
        angles = snapshot.get("steer_meas_rad")
        if angles is None:
            return None
        angles = np.asarray(angles, dtype=float).reshape(-1)
        if angles.size != _NUM_SWERVES or not np.all(np.isfinite(angles)):
            return None
        return angles

    def _swerve_telemetry(self) -> Optional[dict]:
        """Module-level commanded/measured state, or None if unavailable.

        Optional on purpose: `Base` is stubbed in tests and older checkouts
        predate `swerve_telemetry`, and neither should stop a run from being
        logged. A missing snapshot records as NaN, which is honest; a raised
        exception here would kill the control tick.
        """
        getter = getattr(self.base, "swerve_telemetry", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    # ── Measurement ──────────────────────────────────────────────────────────

    def _sync_from_hardware(self, force_arm_read: bool = False) -> None:
        """Push measured arm / lift / base state into the IK configuration."""
        left_q = right_q = None
        if force_arm_read or self.config.use_measured_arm_state:
            left_q = self.left_arm.get_joint_positions()
            right_q = self.right_arm.get_joint_positions()
            self._measured_left_q = np.asarray(left_q, dtype=float).copy()
            self._measured_right_q = np.asarray(right_q, dtype=float).copy()

        lift = self._measured_lift() if self.config.use_measured_lift else None

        # Base feedback is a first-order blend rather than a hard reset --
        # see base_feedback_alpha. alpha 1 reproduces the old behaviour
        # exactly; alpha 0 runs the base open loop, the same way the arms are
        # run. The forced startup sync always hard-seeds from odometry so an
        # open-loop run still starts at the robot's actual pose, mirroring
        # the one-time encoder snapshot the arms get.
        alpha = float(self.config.base_feedback_alpha)
        if force_arm_read or alpha >= 1.0 or not self._base_authority:
            # Hard-seed while someone else has the chassis (joystick, nav,
            # fix_base, base motion off). The belief must not free-run through
            # a stretch it is not commanding: with the pose loop closed on
            # SLAM, odometry follows the robot wherever it is driven, so a
            # belief that kept integrating would hand the first tick after the
            # override a pose error the size of however far the operator went
            # -- and the base would drive back at full speed to answer it.
            # Seeding every inactive tick keeps that error at zero instead.
            base = np.asarray(self.odometry.pose, dtype=float)
        else:
            belief = self.ik.configuration.q[self.ik.base_qpos_adrs]
            err = np.asarray(self.odometry.pose, dtype=float) - belief
            err[2] = _wrap_pi(err[2])
            base = belief + alpha * err
        self.ik.set_measured_state(
            left_q=left_q, right_q=right_q, lift=lift, base=base
        )

    def _correct_base_from_slam(self) -> None:
        """Pull the pose the base PD measures toward the Odin fix.

        This runs first in `_step`, before the solve and before
        `_dispatch_base`, so the pose the PD closes on this tick already
        carries the correction rather than lagging it by one.

        Deliberately a *correction* rather than a replacement, even though it
        is now fast enough to be feedback. Dead-reckoning is what carries the
        estimate between fixes: it is smooth, available at loop rate, and
        exact with respect to what the solver commanded, none of which a 20 Hz
        absolute fix is. SLAM supplies the one thing it cannot -- a statement
        about the floor -- and the rate limit is what stops a loop-closure
        step from reaching the PD as a step. A dropout costs nothing but the
        correction: `latest()` ages out, this returns early, and the base
        coasts on dead-reckoning exactly as it did before.

        Nothing here is conditional on the base being enabled or overridden.
        The estimate has to stay true while a human drives the chassis by
        joystick, or the first whole-body tick after the override lapses would
        measure from a pose that stopped tracking when authority changed hands.
        """
        if self.slam_pose is None:
            return

        pose, age = self.slam_pose.latest()
        self._slam_pose_age = age
        if pose is None or age > self.config.slam_pose_max_age_s:
            return

        if not self.slam_frame.aligned:
            # First usable fix: pin the SLAM frame onto wherever the odometry
            # currently thinks it is, so the correction starts at exactly zero
            # and only removes drift accumulated from here on.
            self.slam_frame.align(pose, self.odometry.pose)
            print("[wholebody] SLAM base correction aligned "
                  f"(yaw_sign={self.config.slam_yaw_sign:+.0f})")
            return

        target = self.slam_frame.to_ik(pose)
        lin, _ang = self.odometry.apply_correction(
            target,
            self.dt,
            self.config.slam_correction_max_lin_rate,
            self.config.slam_correction_max_yaw_rate,
        )
        self._slam_offset_m = lin

        if lin > self.config.slam_offset_warn_m and not self._slam_offset_warned:
            print(f"[wholebody] SLAM/odometry offset {lin:.2f} m — check "
                  f"slam_yaw_sign, or the base is slipping badly")
            self._slam_offset_warned = True
        elif lin < 0.5 * self.config.slam_offset_warn_m:
            self._slam_offset_warned = False

    def _measured_lift(self) -> Optional[float]:
        try:
            height = self.base.get_lift_height()
        except Exception:
            height = None
        if height is None:
            if not self._lift_unavailable_warned:
                print("[wholebody] lift height unavailable — running the lift open-loop")
                self._lift_unavailable_warned = True
            return None
        self._lift_unavailable_warned = False
        return float(height)

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def _dispatch_arms(self, result) -> None:
        if self.arms_manually_overridden:
            return
        max_lead = (
            self.config.arm_max_vel_rad_s
            * self.config.arm_command_lookahead_s
        )
        for side, arm, q_cmd, measured_q, commanded_q in (
            ("left", self.left_arm, result.left_arm_q,
             self._measured_left_q, self._commanded_left_q),
            ("right", self.right_arm, result.right_arm_q,
             self._measured_right_q, self._commanded_right_q),
        ):
            reference_q = measured_q if self.config.use_measured_arm_state else commanded_q
            if reference_q is None:
                # Initialization normally guarantees a reference. Keep this
                # fallback for isolated callers that invoke dispatch directly.
                reference_q = arm.get_joint_positions()
            q_now = np.asarray(reference_q, dtype=float)
            delta = np.asarray(q_cmd, dtype=float) - q_now
            deadband = max(0.0, float(self.config.arm_joint_deadband_rad))
            # A tiny numerical margin makes an intended 0.050000 rad change
            # behave like nerolib's inclusive homing tolerance despite binary
            # floating-point representation.
            outside_deadband = np.abs(delta) > deadband + 1e-12
            if not np.any(outside_deadband):
                continue
            # Hold each joint independently until its accumulated target error
            # exceeds the same 0.05 rad tolerance used by nerolib homing.
            delta = np.where(outside_deadband, delta, 0.0)
            q_safe = q_now + np.clip(
                delta,
                -max_lead,
                max_lead,
            )
            self._goal_jitter[side].sample(q_safe)
            # Stage the goal for the arm-dispatch thread rather than sending it
            # to nerolib directly -- see _arm_dispatch_tick for why (YOR_D-style
            # sub-step interpolation at a higher, decoupled rate).
            with self._arm_dispatch_lock:
                self._arm_seg_goal[side] = q_safe.copy()
                self._arm_seg_revision[side] += 1
            if side == "left":
                self._commanded_left_q = q_safe.copy()
            else:
                self._commanded_right_q = q_safe.copy()

    def _arm_dispatch_tick(self) -> None:
        """One interpolation step toward the latest staged arm target.

        Bridges the 30 Hz solve loop to nerolib at a steadier, higher rate:
        each newly-staged joint target (from `_dispatch_arms`) is interpolated
        against wherever dispatch last actually left off, over
        `arm_interpolation_steps` sub-steps, each sent with a short
        `arm_preview_time`. This is YOR_D's own proven arm-commanding pattern
        (fixed-step interpolation feeding nerolib every ~11 ms), reproduced
        here instead of one coarse ~33 ms command per solve tick.

        Called both by `_arm_dispatch_loop` (the real background thread) and
        directly by tests, so it takes no arguments and is safe to call any
        number of times -- it dispatches nothing once the current segment is
        exhausted and no new one has been staged.
        """
        if self.arms_manually_overridden or not self.config.enable_arm_motion:
            return
        steps = max(1, int(self.config.arm_interpolation_steps))
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            with self._arm_dispatch_lock:
                revision = self._arm_seg_revision[side]
                goal = self._arm_seg_goal[side]
            if goal is None:
                continue
            if revision != self._arm_seg_active_revision[side]:
                # A fresh target arrived. Interpolate from wherever dispatch
                # actually last left off (which may be mid-segment, not the
                # previous goal, if this revision changed before the last one
                # finished) so there is never a position discontinuity.
                self._arm_seg_active_revision[side] = revision
                current = self._arm_seg_current[side]
                self._arm_seg_start[side] = current if current is not None else goal
                self._arm_seg_step[side] = 0
            start = self._arm_seg_start[side]
            step = self._arm_seg_step[side]
            if start is None or step >= steps:
                continue
            alpha = float(step + 1) / float(steps)
            q_cmd = start + alpha * (goal - start)
            self._arm_seg_step[side] = step + 1
            self._arm_seg_current[side] = q_cmd
            self._dispatch_jitter[side].sample(q_cmd)
            try:
                arm.set_joint_target(q_cmd, preview_time=self.config.arm_preview_time)
            except Exception as exc:
                print(f"[wholebody] {side} arm dispatch failed: {exc}")

    def _arm_dispatch_loop(self) -> None:
        rate = RateLimiter(self.config.arm_dispatch_hz, warn=False)
        while self._running:
            self._arm_dispatch_timing.tick()
            try:
                self._arm_dispatch_tick()
            except Exception as exc:
                # One bad tick must not take the thread down -- the next
                # solve-tick revision will recover it.
                print(f"[wholebody] arm dispatch tick failed: {exc}")
            rate.sleep()

    def _dispatch_lift(self, result) -> None:
        if self.lift_manually_overridden:
            self._last_lift_command = None  # re-issue after the override lapses
            # The operator moved the column out from under the PD; its stored
            # measurement history describes a motion this loop did not command.
            self.lift_pd.reset()
            self._lift_cmd_velocity = 0.0
            self._lift_driving_since = None
            self._lift_feedback_blocked = False
            self._lift_dispatch = {"active": False, "mode": "override",
                                   "meas": self._measured_lift()}
            return

        # An explicit lift_target is an operator instruction, so it is what the
        # lift servos to. Otherwise follow the solver, which raises or lowers
        # the column on its own when that is what an EE target needs.
        #
        # These differ because of how the QP is posed: the lift sits in the
        # posture task (cost 1e-4) while the EE frames sit in tasks costing
        # 1.0, so with both hands pinned the solver will hold the lift still —
        # raising it would drag the hands off target. Servoing to the raw
        # solver output would therefore make an operator's lift request look
        # like a no-op. Driving the real lift instead feeds the new height
        # back in through set_measured_state(), and the arms adjust to keep
        # the hands where they were asked to be.
        goal = self.ik.clamp_lift(
            self.lift_target if self.lift_target is not None else result.lift_q
        )

        velocity_mode = self._lift_velocity_available()
        self._lift_dispatch = {
            "active": True,
            "mode": "velocity" if velocity_mode else "bang_bang",
            "goal": goal,
            "blocked": self._lift_feedback_blocked,
        }
        if velocity_mode:
            self._dispatch_lift_velocity(goal)
        else:
            self._dispatch_lift_bang_bang(goal)

    # -- streamed velocity (firmware advertises lift_velocity_v1) ------------

    def _lift_velocity_available(self) -> bool:
        """Whether to stream velocity rather than bang-bang up/down/stop.

        The answer comes from the *controller's* capability line, not from the
        presence of a Python method: an older sketch has no "vel" command and
        would sit still while the host thought it was driving. The age query
        has to exist too, since the PD is not allowed to run without being able
        to tell fresh feedback from stale.
        """
        supports = getattr(self.base, "lift_supports_velocity", None)
        has_age = getattr(self.base, "get_lift_height_age", None)
        try:
            available = bool(supports and has_age and supports())
        except Exception:
            available = False

        if available != self._lift_velocity_mode:
            if self._lift_velocity_mode is not None:
                # Switching paths mid-run: leave the one being abandoned in a
                # stopped state rather than with a command still standing.
                self._halt_lift()
            print("[wholebody] lift control: "
                  + ("streamed velocity (lift_velocity_v1)" if available
                     else "up/down/stop (no velocity capability reported)"))
            self._lift_velocity_mode = available
        return available

    def _lift_feedback_age(self) -> Optional[float]:
        try:
            age = self.base.get_lift_height_age()
        except Exception:
            return None
        return None if age is None else float(age)

    def _lift_position_known(self) -> Optional[bool]:
        known = getattr(self.base, "lift_position_known", None)
        if known is None:
            return None
        try:
            return known()
        except Exception:
            return None

    def _dispatch_lift_velocity(self, goal: float) -> None:
        height = self._measured_lift()
        self._lift_dispatch["meas"] = height
        if height is None or self._lift_position_known() is False:
            self._refuse_lift("height is unknown")
            return

        now = time.monotonic()
        age = self._lift_feedback_age()
        self._lift_dispatch["age"] = age
        fresh = age is not None and age <= self.config.lift_feedback_max_age_s

        if fresh:
            self._lift_feedback_blocked = False
        else:
            # A parked column reports nothing — the firmware streams height
            # while it moves, not at rest — so an old reading on a lift that is
            # standing still is simply the last true one, and starting from it
            # is allowed. What is not allowed is *continuing* on it: once
            # motion has been asked for, telemetry has to appear and keep
            # appearing. The grace window covers the firmware closing its
            # driver relay before the first height line can exist.
            started = self._lift_driving_since
            within_grace = (
                started is not None
                and (now - started) <= self.config.lift_feedback_grace_s
            )
            if self._lift_feedback_blocked or (started is not None and not within_grace):
                self._refuse_lift(f"height telemetry stale ({age})")
                return

        velocity = self.lift_pd.update(goal, height, now)
        # The PD's own filtered measurement derivative, alongside the velocity
        # it asked for. Fitting lift_kd offline needs the same estimate the
        # controller used, not one recomputed from logged heights.
        self._lift_dispatch["vel_est"] = self.lift_pd.filtered_velocity
        self._lift_dispatch["cmd_vel"] = velocity
        self._send_lift_velocity(velocity)

    def _send_lift_velocity(self, velocity: float) -> None:
        velocity = float(velocity)
        if not math.isfinite(velocity):
            self._refuse_lift("PD produced a non-finite velocity")
            return

        try:
            self.base.lift_set_velocity(velocity)
        except Exception as exc:
            print(f"[wholebody] lift velocity command failed: {exc}")
            return

        if velocity == 0.0:
            self._lift_driving_since = None
        elif self._lift_cmd_velocity == 0.0:
            self._lift_driving_since = time.monotonic()
        self._lift_cmd_velocity = velocity

    def _refuse_lift(self, reason: str) -> None:
        """Command zero, forget the derivative, and stay refused.

        The latch matters: a refusal sets the commanded velocity to zero, and
        without it the very next cycle would see a lift that is not being
        driven, allow it to start again, and the lift would inch along against
        feedback nobody can trust.
        """
        if not self._lift_feedback_blocked:
            print(f"[wholebody] lift held: {reason}")
        self._lift_feedback_blocked = True
        self._lift_dispatch["blocked"] = True
        self._lift_dispatch["cmd_vel"] = 0.0

        self.lift_pd.reset()
        self._lift_driving_since = None
        if self._lift_cmd_velocity != 0.0:
            try:
                self.base.lift_set_velocity(0.0)
            except Exception as exc:
                print(f"[wholebody] lift stop failed: {exc}")
        self._lift_cmd_velocity = 0.0
        self._request_lift_refresh()

    def _request_lift_refresh(self) -> None:
        """Ask the controller for a status line while the lift is held.

        The latch can only be cleared by a fresh height, and a held lift is not
        moving, so nothing would produce one. A `status` request does, without
        commanding any motion. Rate-limited because it is a serial round trip
        in a 30 Hz loop, not because it is expensive to be right.
        """
        now = time.monotonic()
        if now - self._lift_refresh_requested < 2.0:
            return
        self._lift_refresh_requested = now
        status = getattr(self.base, "get_lift_status", None)
        if status is None:
            return
        try:
            status()
        except Exception as exc:
            print(f"[wholebody] lift status request failed: {exc}")

    # -- bang-bang (older firmware: up / down / stop only) -------------------

    def _dispatch_lift_bang_bang(self, goal: float) -> None:
        height = self._measured_lift()
        self._lift_dispatch["meas"] = height
        if height is None:
            return  # no feedback → refuse to drive a bang-bang actuator

        error = goal - height
        if abs(error) <= self.config.lift_deadband_m:
            command = "stop"
        else:
            command = "up" if error > 0 else "down"

        # Only send on change: PicoLift talks over serial and repeats are waste.
        self._lift_dispatch["mode"] = f"bang_{command}"
        if command == self._last_lift_command:
            return
        self._last_lift_command = command
        try:
            {"up": self.base.lift_up, "down": self.base.lift_down,
             "stop": self.base.lift_stop}[command]()
        except Exception as exc:
            print(f"[wholebody] lift {command} failed: {exc}")
            self._last_lift_command = None

    def _dispatch_base(self, result) -> None:
        """Drive the chassis toward the base *pose* the solver asked for.

        `result.base_position` is the IK's own belief about where the base
        should be; `self.odometry.pose` is where it is -- dead-reckoned at
        loop rate and, with `enable_slam_base_pose`, corrected toward the Odin
        fix every tick by `_correct_base_from_slam`, so the difference the PD
        closes is a real one and not just the part of the command the wheels
        admitted to. `BasePoseController` (robot/base.py) closes a PD on that
        difference and returns a body-frame velocity request, which then goes
        through exactly the chain the solver's velocity used to: heading-rate
        limit, acceleration limit, yaw filter, axis map, and the same relay.

        Nothing about authority changes. A disabled base, `fix_base`, or a
        live manual override still stops the base and now also resets the PD,
        so the cycle that gets authority back starts from the pose measured
        then rather than damping against motion it never commanded.
        """
        target = np.asarray(result.base_position, dtype=float).reshape(-1)[:3]

        if not self._base_authority:
            if np.any(self._last_base_command):
                self._halt_base()
            # Forget the filters along with the motion: resuming later must
            # not inherit a filtered value or a moving hysteresis state from
            # before the override/disable.
            self._yaw_filt = 0.0
            self._yaw_active = False
            self._lin_filt = (0.0, 0.0)
            self._lin_active = False
            self._twist_shaper.reset(self._measured_module_angles())
            self.base_pose.reset()
            with self._lock:
                self._last_base_velocity = np.zeros(3)
            self._base_dispatch = {
                "active": False,
                "target": target.copy(),
                "err": np.zeros(3),
                "req": np.zeros(3),
                "body": np.zeros(3),
                "sent": np.zeros(3),
            }
            return

        forward, lateral, yaw_rate = self.base_pose.compute(
            target, self.odometry.pose, dt=self.dt)

        request = np.array([forward, lateral, yaw_rate], dtype=float)

        forward, lateral = self._limit_linear(forward, lateral)
        forward, lateral = self._limit_heading_rate(forward, lateral)
        forward, lateral = self._limit_accel(forward, lateral)
        yaw_rate = self._filter_yaw(yaw_rate)

        # Last, and on all three components at once. Everything above shapes
        # one component in isolation and so cannot see the module angles the
        # combination implies -- including the deadbands, whose drop to
        # exactly zero is a *step* in the geometry whenever the other channel
        # is still live: (f, l, w) -> (0, 0, w) re-aims all four modules by up
        # to 90 degrees in one tick, and passes the acceleration limiter
        # untouched because that one exempts zero. Putting the shaper after
        # them is what closes that hole.
        shaped = self._twist_shaper.shape((forward, lateral, yaw_rate), self.dt)
        forward, lateral, yaw_rate = (float(shaped[0]), float(shaped[1]),
                                      float(shaped[2]))
        # The acceleration limiter measures the next tick's change against
        # what was actually issued, not against the value the shaper declined
        # to send -- otherwise its budget is spent on motion that never
        # happened.
        self._base_vel_prev = (forward, lateral)

        command = self.config.base_axis_map.to_command(forward, lateral, yaw_rate)
        self._send_base_command(command)

        # `err` is the pose error the PD acted on, `req` its unshaped velocity
        # request, `body` what survived the clamps and the deadband, `sent`
        # what left this file. Logging all four separates "the base is already
        # where it was asked to be" from "the deadband ate it" from "the clamp
        # capped it" -- different base-tuning problems that look identical at
        # the wheels.
        self._base_dispatch = {
            "active": True,
            "target": target.copy(),
            "err": self.base_pose.last_error.copy(),
            "req": request,
            "body": np.array([forward, lateral, yaw_rate], dtype=float),
            "sent": command.copy(),
            "leash": self._base_leash,
            "ff": self.base_pose.last_feedforward.copy(),
            "slew_scale": float(self._twist_shaper.last_scale),
        }

        # Odometry integrates what was *commanded* after clamping, so the
        # model's base pose cannot run ahead of what the wheels were asked for.
        v_applied = self._body_to_world(forward, lateral, yaw_rate)
        self.odometry.update(v_applied, self.dt)
        with self._lock:
            self._last_base_velocity = v_applied
            self._last_base_command = command.copy()

    def _send_base_command(self, command: np.ndarray) -> None:
        try:
            if self.base_controller is not None:
                self.base_controller.mode = "BASE_VEL"
                self.base_controller.target_velocity = command
            else:
                self.base.set_target_base_velocity(command, smooth=True)
        except Exception as exc:
            print(f"[wholebody] base command failed: {exc}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _world_to_body(self, v_world: np.ndarray) -> tuple[float, float, float]:
        """World-frame [vx, vy, omega] → (forward, lateral, yaw_rate).

        The robot faces −Y in the description and its left side is +X, so in
        the chassis frame forward = −v_y and left = +v_x (see BaseAxisMap).

        This is the written-down form of that convention: `_body_to_world`
        inverts it for odometry, and `BasePoseController`'s `heading_offset`
        (base_pose_heading_offset, −π/2) reproduces it for the pose error.
        Held to it by tests/test_wholebody_control.py::test_axis_map_and_odometry.
        """
        theta = float(self.odometry.pose[2])
        c, s = math.cos(theta), math.sin(theta)
        vx_body = c * v_world[0] + s * v_world[1]
        vy_body = -s * v_world[0] + c * v_world[1]
        return -vy_body, vx_body, float(v_world[2])

    def _body_to_world(self, forward: float, lateral: float, yaw_rate: float) -> np.ndarray:
        """Inverse of `_world_to_body`, for odometry integration."""
        vx_body, vy_body = lateral, -forward
        theta = float(self.odometry.pose[2])
        c, s = math.cos(theta), math.sin(theta)
        return np.array([c * vx_body - s * vy_body, s * vx_body + c * vy_body, yaw_rate])

    def _limit_linear(self, forward: float, lateral: float) -> tuple[float, float]:
        """Clamp, low-pass and hysteresis-deadband the linear velocity.

        Same three stages as `_filter_yaw`, in the same order and for the
        same reasons:

        1. Clamp to base_max_lin_vel *before* filtering, so a one-tick
           solver spike charges the filter with at most one tick of full
           speed. The clamp acts on the magnitude of (forward, lateral) and
           rescales the pair, so the direction the solver asked for is the
           direction the wheels get -- clamping per axis would not merely
           scale the command but turn it, and `_dispatch_base` integrates
           the result into `BaseOdometry`, so a distortion that always
           points at an axis accumulates as pose bias rather than averaging
           out.
        2. One-pole low-pass, per component so the direction is filtered
           consistently with the magnitude. A single-tick burst decays
           without reaching the entry threshold; a sustained request passes
           with ~base_vel_filter_tau of onset lag. Filtering a convex
           combination of clamped inputs cannot exceed the clamp, so no
           re-clamp is needed.
        3. Hysteresis deadband: motion starts above base_vel_deadband and
           stops below base_vel_deadband_exit, so a request hovering near
           the boundary does not re-aim all four swerve modules every other
           tick.
        """
        limit = self.config.base_max_lin_vel
        speed = math.hypot(forward, lateral)
        if speed > limit:
            scale = limit / speed
            forward, lateral = forward * scale, lateral * scale

        tau = self.config.base_vel_filter_tau
        if tau > 0.0:
            alpha = self.dt / (tau + self.dt)
            ff, fl = self._lin_filt
            ff += alpha * (float(forward) - ff)
            fl += alpha * (float(lateral) - fl)
            self._lin_filt = (ff, fl)
            forward, lateral = ff, fl
            speed = math.hypot(forward, lateral)
        else:
            self._lin_filt = (float(forward), float(lateral))

        if self._lin_active:
            self._lin_active = speed >= self.config.base_vel_deadband_exit
        else:
            self._lin_active = speed >= self.config.base_vel_deadband
        if not self._lin_active:
            return 0.0, 0.0
        return float(forward), float(lateral)

    def _limit_heading_rate(self, forward: float, lateral: float) -> tuple[float, float]:
        """Bound how fast the commanded direction may turn, keeping its speed.

        The swerve modules slew at 265-353 deg/s. The solver has no idea that
        exists, and because `atan2` of a short vector is ill-conditioned it
        asked for a median 552 deg/s whenever the base was creeping. A module
        that never reaches its commanded angle sends the chassis somewhere the
        solver did not ask for, so the ask has to be bounded to something the
        hardware can serve.

        Magnitude is preserved, only the direction is rate-limited: the point
        is to stop the base whirling, not to slow it down. Throttling speed
        while a module is still turning is a separate mechanism and belongs at
        the wheel, where the measured angle is (`cos_error_scaling` in
        base_motor.py, which needs USE_FEEDBACK_FOR_STEER to do anything).

        Reversal is accounted for. A swerve module serves a 180 deg direction
        change by flipping the drive and not turning at all, so a heading
        change beyond 90 deg is measured against the *reversed* previous
        heading -- otherwise this would rate-limit a move the hardware makes
        instantly.

        While the base is stopped the reference heading is frozen rather than
        reset, so the next motion is limited from where the modules actually
        are, not from zero.
        """
        limit = self.config.base_heading_rate_limit
        speed = math.hypot(forward, lateral)
        if limit <= 0.0 or speed <= 0.0:
            return forward, lateral            # disabled, or nothing to aim

        heading = math.atan2(lateral, forward)
        if self._base_heading is None:
            self._base_heading = heading
            return forward, lateral

        reference = self._base_heading
        delta = _wrap_pi(heading - reference)
        if abs(delta) > math.pi / 2:
            # The modules would flip rather than turn the long way; measure the
            # real travel against the reversed reference.
            reference = _wrap_pi(reference + math.pi)
            delta = _wrap_pi(heading - reference)

        max_step = limit * self.dt
        if abs(delta) > max_step:
            heading = _wrap_pi(reference + math.copysign(max_step, delta))
            forward, lateral = speed * math.cos(heading), speed * math.sin(heading)

        self._base_heading = heading
        return float(forward), float(lateral)

    def _filter_yaw(self, yaw_rate: float) -> float:
        """Low-pass, hysteresis-deadband and clamp the yaw request.

        Three stages, in an order that matters:

        1. Clamp to the ang-vel limit *before* filtering, so a one-tick
           solver spike (12 rad/s was observed) charges the filter with at
           most one tick of full-rate rotation, not two hundred.
        2. One-pole low-pass. A single-tick pulse decays without ever
           reaching the entry threshold; a sustained request passes with
           ~base_yaw_filter_tau of onset lag.
        3. Hysteresis deadband: start rotating above base_yaw_deadband, stop
           below base_yaw_deadband_exit. The gap keeps a request hovering at
           the boundary from toggling the module geometry every tick — the
           chatter measured at 7.4 toggles/s on 2026-08-24.
        """
        limit = self.config.base_max_ang_vel
        yaw_rate = float(np.clip(yaw_rate, -limit, limit))

        tau = self.config.base_yaw_filter_tau
        if tau > 0.0:
            alpha = self.dt / (tau + self.dt)
            self._yaw_filt += alpha * (yaw_rate - self._yaw_filt)
        else:
            self._yaw_filt = yaw_rate

        mag = abs(self._yaw_filt)
        if self._yaw_active:
            self._yaw_active = mag >= self.config.base_yaw_deadband_exit
        else:
            self._yaw_active = mag >= self.config.base_yaw_deadband
        return self._yaw_filt if self._yaw_active else 0.0

    def _limit_accel(self, forward: float, lateral: float) -> tuple[float, float]:
        """Bound how fast the commanded velocity vector may change.

        This is the reversal guard that `_limit_heading_rate` cannot be: that
        one measures a >90 deg change against the reversed previous heading,
        because a module serves a reversal by flipping the drive. True for the
        module, false for the chassis, and the chassis is what has momentum.

        Stopping is exempt. An exact zero -- a deadbanded command, a halt --
        goes straight through, so nothing here can delay the base coming to
        rest.
        """
        limit = self.config.base_max_accel
        if limit <= 0.0:
            return forward, lateral
        if forward == 0.0 and lateral == 0.0:
            self._base_vel_prev = (0.0, 0.0)
            return forward, lateral

        prev_f, prev_l = self._base_vel_prev
        df, dl = forward - prev_f, lateral - prev_l
        change = math.hypot(df, dl)
        max_step = limit * self.dt
        if change > max_step and change > 1e-12:
            scale = max_step / change
            forward, lateral = prev_f + df * scale, prev_l + dl * scale
        self._base_vel_prev = (float(forward), float(lateral))
        return float(forward), float(lateral)

    def _clamp(self, value: float, limit: float,
               deadband: Optional[float] = None) -> float:
        if deadband is None:
            deadband = self.config.base_vel_deadband
        if abs(value) < deadband:
            return 0.0
        return float(np.clip(value, -limit, limit))

    def _halt_base(self) -> None:
        try:
            if self.base_controller is not None:
                self.base_controller.mode = "BASE_VEL"
                self.base_controller.target_velocity = np.zeros(3)
            else:
                self.base.set_target_base_velocity(np.zeros(3), smooth=True)
        except Exception as exc:
            print(f"[wholebody] base halt failed: {exc}")
        self._last_base_command = np.zeros(3)
        self._base_vel_prev = (0.0, 0.0)
        self._twist_shaper.reset(self._measured_module_angles())
        # A halt is a discontinuity in the measured pose as far as the PD is
        # concerned -- the base may be pushed, or simply stand still while the
        # solver's belief moves on. Damping the first cycle after a resume
        # against that would be damping against motion that never happened.
        self.base_pose.reset()

    def _halt_lift(self) -> None:
        """Stop the lift and forget everything the PD knew about its motion.

        `lift_stop` is the hard stop in both firmwares: it ends velocity mode
        outright rather than ramping, which is what a disarm should do.
        """
        try:
            self.base.lift_stop()
        except Exception as exc:
            print(f"[wholebody] lift stop failed: {exc}")
        self._last_lift_command = None
        self._lift_cmd_velocity = 0.0
        self._lift_driving_since = None
        self._lift_feedback_blocked = False
        self.lift_pd.reset()

    @staticmethod
    def _set_gripper(arm, value: float) -> None:
        try:
            arm.set_joint_target(arm.get_joint_positions(), gripper_target=float(value))
        except Exception as exc:
            print(f"[wholebody] gripper command failed: {exc}")
