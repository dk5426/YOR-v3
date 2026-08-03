"""
wholebody_control.py — whole-body control on the *real* YORv3 hardware.

This is the hardware counterpart of robot/yor_mujoco.py. Both run the same
solver (robot/arm/wholebody_ik.py) over the same description
(description/scene_wholebody.xml); they differ only in where the measured
configuration comes from and where the solution is dispatched:

                        simulation                  hardware
    measure     data.qpos (previous solve)   arm encoders, lift height,
                                             base odometry
    dispatch    data.qpos + mj_forward       ArmNode.set_joint_target()   ×2
                                             PicoLift up/down/stop
                                             Base.set_target_base_velocity()

Control flow, once per cycle (default 100 Hz):

  1. read measured state and push it into the IK configuration
  2. solve whole-body IK for the current EE / lift targets
  3. dispatch arms, lift and base, each with its own clamp and its own
     enable flag
  4. integrate the commanded base velocity into the odometry estimate

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

* **The lift is bang-bang.** PicoLift only knows up / down / stop, so the loop
  runs a deadband servo against the solver's lift command rather than sending
  a position.
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

    control_hz: float = 100.0

    # ── Arms ────────────────────────────────────────────────────────────────
    enable_arm_motion: bool = True
    # Passed to nerolib as the minimum duration of the commanded move; it also
    # smooths the 500 Hz interpolation inside the controller.
    arm_preview_time: float = 0.05
    # Hard cap on how fast a joint may be asked to move, independent of the
    # solver's own velocity limit. Converted to a per-cycle step internally.
    arm_max_vel_rad_s: float = 2.0
    # Feed the arm encoders back into the IK each cycle. Closing the loop on
    # the real configuration keeps collision avoidance honest; turning it off
    # runs the solver open-loop from its own previous solution (smoother, but
    # the model can drift away from the robot).
    use_measured_arm_state: bool = True

    # ── Lift (bang-bang PicoLift) ───────────────────────────────────────────
    enable_lift_motion: bool = True
    # Stop band around the solver's lift command. Below this the lift is
    # commanded to stop, which also keeps serial traffic down.
    lift_deadband_m: float = 0.01
    use_measured_lift: bool = True

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
    # loop closure; applied directly at 100 Hz the solver would see the end
    # effectors teleport and command a large arm velocity on the next tick.
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
      pull-mode — every read is a round trip to Thor. A 100 Hz loop cannot
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
                base_posture_cost=5e-2,
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
        self._last_base_velocity = np.zeros(3)   # world frame, as commanded
        self._last_base_command = np.zeros(3)    # what Base was actually sent
        self._last_lift_command: Optional[str] = None
        self._last_solve_ok = False
        self._solve_error: Optional[str] = None

        self._manual_base_until = 0.0
        self._manual_lift_until = 0.0
        self._manual_arm_until = 0.0
        self._lift_unavailable_warned = False

        self._thread: Optional[threading.Thread] = None
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
        self._sync_from_hardware()

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
            return
        if not self.initialized:
            self.init()
        self._running = True
        self._thread = threading.Thread(
            target=self._control_loop, name="wholebody-control", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._running = False
        self._thread.join(timeout=2.0)
        self._thread = None
        self._halt_base()
        self._halt_lift()
        if self.slam_pose is not None:
            self.slam_pose.stop()
            self.slam_pose = None

    def emergency_stop(self) -> None:
        """Stop the loop and freeze every actuator where it stands."""
        self._running = False
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
            "collision_avoidance": self.ik.avoid_collisions,
            "base_motion_enabled": self.config.enable_base_motion,
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

    def _sync_from_hardware(self) -> None:
        """Push measured arm / lift / base state into the IK configuration."""
        left_q = right_q = None
        if self.config.use_measured_arm_state:
            left_q = self.left_arm.get_joint_positions()
            right_q = self.right_arm.get_joint_positions()

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
        max_step = self.config.arm_max_vel_rad_s * self.dt
        for arm, q_cmd in (
            (self.left_arm, result.left_arm_q),
            (self.right_arm, result.right_arm_q),
        ):
            q_now = arm.get_joint_positions()
            q_safe = q_now + np.clip(np.asarray(q_cmd) - q_now, -max_step, max_step)
            arm.set_joint_target(q_safe, preview_time=self.config.arm_preview_time)

    def _dispatch_lift(self, result) -> None:
        if self.lift_manually_overridden:
            self._last_lift_command = None  # re-issue after the override lapses
            return

        height = self._measured_lift()
        if height is None:
            return  # no feedback → refuse to drive a bang-bang actuator

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
        try:
            self.base.lift_stop()
        except Exception as exc:
            print(f"[wholebody] lift stop failed: {exc}")
        self._last_lift_command = None

    @staticmethod
    def _set_gripper(arm, value: float) -> None:
        try:
            arm.set_joint_target(arm.get_joint_positions(), gripper_target=float(value))
        except Exception as exc:
            print(f"[wholebody] gripper command failed: {exc}")
