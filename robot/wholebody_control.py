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
  4. integrate the commanded base velocity into the odometry estimate

Lift and base are dispatched directly from that 30 Hz loop. Arms are not: each
solved joint target is handed to a second, faster loop (default 90 Hz, see
`_arm_dispatch_tick`/`_arm_dispatch_loop`) that interpolates it against the
previous target over a few fixed sub-steps before sending anything to
nerolib. This reproduces YOR_D's own arm-commanding chain -- a Cartesian
target arriving no faster than 30 Hz, smoothed by frequent, short-duration
joint commands rather than one coarse command per solve tick.

Three things are worth knowing before running this on the robot:

* **Base odometry is dead-reckoned from the commanded velocity**, optionally
  corrected by SLAM. By default nothing absolute feeds this loop, so the IK's
  notion of where the chassis is drifts from reality over time. That is
  tolerable because teleop targets are generated relative to the *current* EE
  pose, but absolute world-frame targets get less accurate the longer the base
  drives.

  Setting `enable_slam_base_pose` attaches the Odin pose (`slam/pose`, from
  robot/odin_pub_node.py) as a **drift correction**: dead-reckoning remains the
  primary, always-available signal and the SLAM pose is bled in under a rate
  limit, so loop-closure jumps never reach the IK as a step. It ships off
  because `slam_yaw_sign` has to be calibrated first — see docs/RUNNING.md.

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

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import mink
from loop_rate_limiters import RateLimiter

from robot.arm.wholebody_ik import WholeBodyIK, WholeBodyIKConfig


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseAxisMap:
    """Maps a body-frame velocity onto ``Base.set_target_base_velocity``.

    In the description, the robot faces **−Y**: at the home keyframe both end
    effectors sit at y ≈ −0.25 m, and the left arm is at +X. So, in the base's
    own frame, forward = −y_model and left = +x_model.

    ``Base`` takes a 3-vector whose first element is the lateral component and
    whose second is the forward one — that is the ordering the working path
    follower in robot/base.py uses (``[pid(d_left), pid(d_forward), omega]``).
    The signs have *not* been verified against the physical robot; drive it
    with robot/teleop/joystick.py first and flip whichever sign is wrong.
    """

    lateral_index: int = 0
    forward_index: int = 1
    yaw_index: int = 2
    lateral_sign: float = +1.0
    forward_sign: float = +1.0
    yaw_sign: float = +1.0

    def to_command(self, forward: float, lateral: float, yaw_rate: float) -> np.ndarray:
        cmd = np.zeros(3, dtype=float)
        cmd[self.forward_index] = self.forward_sign * forward
        cmd[self.lateral_index] = self.lateral_sign * lateral
        cmd[self.yaw_index] = self.yaw_sign * yaw_rate
        return cmd


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
    # Match nerolib's resetToHome acceptance band: a joint target must differ
    # from the WBC arm reference by more than 0.05 rad before it is dispatched.
    # The band is applied independently to all seven joints.
    arm_joint_deadband_rad: float = 0.05
    # Hard cap on how far ahead of measured state a streamed joint target may
    # sit. This is a bounded look-ahead, not a per-cycle step: if the WBC stream
    # stops, Ruckig safely comes to rest at a target no farther than
    # arm_max_vel_rad_s * arm_command_lookahead_s away.
    arm_command_lookahead_s: float = 0.10
    # Hard cap used to derive that maximum look-ahead distance, independent of
    # the solver's own velocity limit.
    #
    # This matches `joint_vel_max` in robot/arm/arm.py (3.0 rad/s). A smaller
    # number here would silently override the commissioned native limit — the
    # arms would never reach the speed the controller is configured for, and
    # the discrepancy would only show up as sluggish whole-body tracking.
    arm_max_vel_rad_s: float = 3.0
    # Open-loop arm planning: seed from the encoders once at startup, then
    # advance IK from its previous commanded solution without reading arm
    # feedback each cycle. Nerolib's low-level motor PD remains closed-loop,
    # but the WBC model can drift from the physical arms if tracking is poor.
    use_measured_arm_state: bool = False

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
    base_max_lin_vel: float = 0.25   # m/s, per axis after mapping
    base_max_ang_vel: float = 0.60   # rad/s
    # Velocities below this are sent as zero, so solver noise doesn't leave the
    # swerve modules humming at a standstill.
    base_vel_deadband: float = 0.02
    base_axis_map: BaseAxisMap = field(default_factory=BaseAxisMap)

    # ── Manual override ─────────────────────────────────────────────────────
    # Direct base / lift commands (joystick, nav, RPC) suspend the whole-body
    # loop's authority over that subsystem for this long after the last one,
    # so the two controllers never fight over the same actuator.
    manual_override_timeout_s: float = 0.5

    # ── SLAM base pose (drift correction) ───────────────────────────────────
    # OFF by default, and it must stay off until `slam_yaw_sign` is calibrated
    # — feeding a mis-signed absolute pose into the IK is worse than the
    # dead-reckoning it replaces, because the error then grows as you drive
    # instead of staying bounded. See docs/RUNNING.md for the 30-second check.
    enable_slam_base_pose: bool = False
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
    # Rate limits on the correction. The SLAM pose steps discontinuously on
    # loop closure; applied directly at the WBC rate the solver would see the
    # end effectors teleport and command a large arm velocity on the next tick.
    # Bleeding the offset in at these rates keeps the IK configuration
    # continuous — 10 cm of accumulated drift is removed in ~1 s, while a 1 m
    # loop-closure jump is absorbed over ~10 s.
    slam_correction_max_lin_rate: float = 0.10   # m/s
    slam_correction_max_yaw_rate: float = 0.20   # rad/s
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
    """Dead-reckoned (x, y, theta) of the chassis in the IK world frame.

    Integrates the velocity that was actually *commanded* to the base, which
    is exact with respect to the solver's intent and free of unit guesswork,
    but open-loop with respect to the floor. Replace `update` with the
    SLAM/EKF-derived estimate when one is trusted; nothing else in this file
    needs to change.
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
        # yaw_sign < 0 means the planar frames have opposite handedness, which
        # is a reflection about the first axis — not just a rotation.
        return (sx, sy) if self.yaw_sign >= 0 else (sx, -sy)

    def align(self, slam_pose: np.ndarray, ik_pose: np.ndarray) -> None:
        """Solve the transform so that `slam_pose` maps exactly onto `ik_pose`."""
        sx, sy, syaw = (float(v) for v in slam_pose)
        px, py, pyaw = (float(v) for v in ik_pose)
        self._rot = _wrap_pi(pyaw - self.yaw_sign * syaw)
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
            _wrap_pi(self.yaw_sign * syaw + self._rot),
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
        self._home_left: Optional[mink.SE3] = None
        self._home_right: Optional[mink.SE3] = None
        self._home_lift: float = 0.0

        self.odometry = BaseOdometry()
        # SLAM drift correction — inert unless enable_slam_base_pose is set.
        self.slam_frame = SlamBaseFrame(yaw_sign=self.config.slam_yaw_sign)
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

    # ── Target API (mirrors robot/yor_mujoco.py so one client drives both) ───

    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: Optional[float] = None,
                           preview_time: float = 0.0) -> None:
        with self._lock:
            self.left_ee_target = ee_target
        if gripper_target is not None:
            self._set_gripper(self.left_arm, gripper_target)

    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: Optional[float] = None,
                            preview_time: float = 0.0) -> None:
        with self._lock:
            self.right_ee_target = ee_target
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
        """Enable/disable dispatch of the solver's base velocity to the wheels."""
        self.config.enable_base_motion = (
            (not self.config.enable_base_motion) if enable is None else bool(enable)
        )
        if not self.config.enable_base_motion:
            self._halt_base()
        return self.config.enable_base_motion

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
        with self._lock:
            base_vel = self._last_base_velocity.copy()
            base_cmd = self._last_base_command.copy()
        return {
            "left_ee_wxyz_xyz": T_l.wxyz_xyz.tolist(),
            "right_ee_wxyz_xyz": T_r.wxyz_xyz.tolist(),
            "lift": self.get_lift_position(),
            "base_xytheta": self.odometry.pose.tolist(),
            "base_velocity": base_vel.tolist(),
            "base_command": base_cmd.tolist(),
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
        }

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        rate = RateLimiter(self.config.control_hz, warn=False)
        while self._running:
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
        result = self.ik.solve(T_l, T_r, lift_target=lift_tgt)
        self._last_solve_ok = bool(result.solved)
        self._solve_error = None

        if self.config.enable_arm_motion:
            self._dispatch_arms(result)
        if self.config.enable_lift_motion:
            self._dispatch_lift(result)
        self._dispatch_base(result)

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
        self.ik.set_measured_state(
            left_q=left_q, right_q=right_q, lift=lift, base=self.odometry.pose
        )

    def _correct_base_from_slam(self) -> None:
        """Bleed dead-reckoning drift out of the base pose using the SLAM pose.

        Deliberately a *correction* rather than a replacement. Dead-reckoning
        stays the primary signal because it is smooth, always available and
        exact with respect to what the solver commanded; SLAM only supplies the
        absolute reference that stops it drifting. A dropout therefore costs
        nothing but the correction itself.
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
            try:
                arm.set_joint_target(q_cmd, preview_time=self.config.arm_preview_time)
            except Exception as exc:
                print(f"[wholebody] {side} arm dispatch failed: {exc}")

    def _arm_dispatch_loop(self) -> None:
        rate = RateLimiter(self.config.arm_dispatch_hz, warn=False)
        while self._running:
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

        if self._lift_velocity_available():
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
        if height is None or self._lift_position_known() is False:
            self._refuse_lift("height is unknown")
            return

        now = time.monotonic()
        age = self._lift_feedback_age()
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
        if height is None:
            return  # no feedback → refuse to drive a bang-bang actuator

        error = goal - height
        if abs(error) <= self.config.lift_deadband_m:
            command = "stop"
        else:
            command = "up" if error > 0 else "down"

        # Only send on change: PicoLift talks over serial and repeats are waste.
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
        if (not self.config.enable_base_motion
                or self.ik.fix_base
                or self.base_manually_overridden):
            if np.any(self._last_base_command):
                self._halt_base()
            with self._lock:
                self._last_base_velocity = np.zeros(3)
            return

        v_world = np.asarray(result.base_velocity, dtype=float)
        forward, lateral, yaw_rate = self._world_to_body(v_world)

        forward = self._clamp(forward, self.config.base_max_lin_vel)
        lateral = self._clamp(lateral, self.config.base_max_lin_vel)
        yaw_rate = self._clamp(yaw_rate, self.config.base_max_ang_vel)

        command = self.config.base_axis_map.to_command(forward, lateral, yaw_rate)
        self._send_base_command(command)

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

    def _clamp(self, value: float, limit: float) -> float:
        if abs(value) < self.config.base_vel_deadband:
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
