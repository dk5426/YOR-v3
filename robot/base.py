import math
import threading
import time

import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R

from commlink import Subscriber
from loop_rate_limiters import RateLimiter

from .base_motor import Base
from .topics import SLAM_PUB_PORT, POSE_TOPIC

THOR_IP = "192.168.1.11"


# -----------------------------
# Pose / frame helpers
# -----------------------------
def xyzw_xyz_to_matrix(qt7):
    qt7 = np.asarray(qt7, dtype=np.float32).reshape(-1)
    if qt7.shape[0] < 7:
        raise ValueError(
            f"Expected 7 values [qx,qy,qz,qw,tx,ty,tz], got {qt7.shape}"
        )
    q = qt7[:4]
    t = qt7[4:7]
    R_mat = R.from_quat(q).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    return T


_ZUP_TO_YUP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _zup_to_yup_transform(T: np.ndarray) -> np.ndarray:
    return _ZUP_TO_YUP @ T @ _ZUP_TO_YUP.T


def get_cam_pose(_sub):
    pose_msg = _sub[POSE_TOPIC]
    return pose_msg[7:14]


def get_pose(sub):
    pose_msg = sub[POSE_TOPIC]
    base_qt7 = pose_msg[0:7]
    base_transform = xyzw_xyz_to_matrix(base_qt7)
    translation = base_transform[:3, 3].astype(np.float32)
    theta = float(np.arctan2(-base_transform[2, 0], base_transform[0, 0])) % (2 * np.pi)
    return translation, theta, base_transform


def _wrap_pi(a: float) -> float:
    return ((a + math.pi) % (2 * math.pi)) - math.pi


# -----------------------------
# Polyline helpers
# -----------------------------
def _cumlen(pts: list[np.ndarray]) -> tuple[np.ndarray, float]:
    if len(pts) < 2:
        s = np.array([0.0], dtype=float)
        return s, 0.0
    segs = np.linalg.norm(np.diff(np.vstack(pts), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segs)])
    return s, float(s[-1])


def _closest_point_on_polyline(
    x: float,
    y: float,
    pts: list[np.ndarray],
    s_cum: np.ndarray,
) -> tuple[float, np.ndarray, int, float]:
    p = np.array([x, y], dtype=float)
    best_d2, best = float("inf"), (0.0, pts[0], 0, 0.0)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        L2 = float(ab @ ab) if np.any(ab) else 1e-12
        t = float(np.clip(((p - a) @ ab) / L2, 0.0, 1.0))
        q = a + t * ab
        d2 = float(np.sum((p - q) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            s_here = float(s_cum[i] + t * np.linalg.norm(ab))
            best = (s_here, q, i, t)
    return best


def _point_at_s(
    pts: list[np.ndarray],
    s_cum: np.ndarray,
    s: float,
) -> tuple[np.ndarray, np.ndarray]:
    s = float(np.clip(s, 0.0, s_cum[-1]))
    i = int(np.searchsorted(s_cum, s, side="right") - 1)
    i = max(0, min(i, len(pts) - 2))
    a, b = pts[i], pts[i + 1]
    seg_L = float(s_cum[i + 1] - s_cum[i]) or 1e-12
    t = float((s - s_cum[i]) / seg_L)
    p = a + t * (b - a)
    tan = (
        (b - a) / np.linalg.norm(b - a)
        if np.linalg.norm(b - a) > 1e-9
        else np.array([1.0, 0.0])
    )
    return p, tan


# -----------------------------
# Heading helpers
# -----------------------------
def signed_angle_2d(v1, v2):
    return np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], np.dot(v1, v2))


def _norm2(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v), 0.0
    return v / n, n


def fwd_xz_from_T(T_base: np.ndarray, forward_col: int = 0) -> np.ndarray:
    R_bw = T_base[:3, :3]
    fwd_xz = np.array([R_bw[0, forward_col], R_bw[2, forward_col]], dtype=float)
    fwd_xz, _ = _norm2(fwd_xz)
    return fwd_xz


def heading_error_from_dir(
    T_base: np.ndarray,
    desired_dir_xz: np.ndarray,
    forward_col: int = 0,
    flip_sign: bool = True,
) -> float:
    fwd_xz = fwd_xz_from_T(T_base, forward_col=forward_col)
    ddir, n = _norm2(desired_dir_xz)
    if n < 1e-9:
        return 0.0
    e = float(signed_angle_2d(fwd_xz, ddir))
    return -e if flip_sign else e


def dir_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.sin(yaw), math.cos(yaw)], dtype=float)


def save_home_pose(translation, T_base):
    home = {
        "x": float(translation[0]),
        "y": float(translation[2]),
        "fwd_xz": fwd_xz_from_T(T_base, forward_col=0),
    }
    return home


# -----------------------------
# Control primitives
# -----------------------------
class PID:
    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        i_limit: float = 1.0,
        out_limit: float | None = None,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.out_limit = out_limit
        self.i = 0.0
        self.prev_e = None

    def reset(self):
        self.i = 0.0
        self.prev_e = None

    def step(self, e: float, dt: float) -> float:
        if dt <= 0.0:
            dt = 1e-3

        self.i += e * dt
        if self.i_limit is not None:
            self.i = max(-self.i_limit, min(self.i, self.i_limit))

        d = 0.0 if self.prev_e is None else (e - self.prev_e) / dt
        self.prev_e = e

        u = self.kp * e + self.ki * self.i + self.kd * d
        if self.out_limit is not None:
            u = max(-self.out_limit, min(self.out_limit, u))
        return float(u)


class BasePoseController:
    """Continuous PD from a base *pose* target to a base *velocity* command.

        forward / left = Kp_xy  * (target_xy - measured_xy), in the body frame
                         - Kd_xy  * filtered d(measured_xy)/dt
        yaw_rate       = Kp_yaw * wrap(target_yaw - measured_yaw)
                         - Kd_yaw * filtered d(measured_yaw)/dt

    This is what whole-body control drives the chassis with: it hands over
    `WholeBodyIKResult.base_position` -- where the solver believes the base
    should be -- together with the pose it measures, and this closes the gap.

    **Why a pose and not a velocity.** The solver's `base_velocity` is a
    difference of two consecutive base beliefs divided by dt, so it is only
    ever a statement about one tick. Everything the wheels fail to deliver on
    that tick -- the vector clamp, the deadband, the low-pass, the swerve
    slew, a module still turning -- is lost the moment the next solve
    overwrites it, because the next velocity does not know the previous one
    was not served. The shortfall accumulates silently as a position lag that
    nothing ever reads back. A pose target cannot lose it: whatever the base
    failed to travel is still sitting in the error on the next cycle, so the
    chassis converges to where the solver asked instead of to wherever the
    clamps happened to leave it.

    Three details matter more than the gains:

    * **The derivative is of the measurement, not of the error.** The target
      moves every tick and jumps outright whenever authority changes hands
      (an override lapsing, a re-enable, a SLAM correction); differentiating
      the error would answer each of those with a velocity spike. Measurement
      damping is pure damping, the same choice, for the same reason, as
      `LiftVelocityPD` in robot/wholebody_control.py.

    * **Both deadbands act on the error and produce an exact zero.** Linear
      motion is deadbanded on the *vector* magnitude, never per axis: a
      per-axis band turns the command as well as shrinking it, so a request
      21 degrees off the forward axis would go out as pure forward -- wrong
      direction, not merely wrong length. Yaw gets its own band in its own
      units. Inside either band the corresponding output is 0.0 exactly, so
      the modules are not left humming against the last few millimetres.

    * **It measures nothing itself.** Target and measured pose must arrive in
      the same frame, from the caller that owns both (whole-body control
      passes the solver's target and its own odometry). This class holds no
      odometry, no SLAM subscription and no notion of a map.

    Frame: `heading_offset` says where the chassis' nose points at yaw 0. The
    default -pi/2 matches the whole-body description, in which the robot faces
    -Y and its left side is +X -- the same convention `BaseAxisMap` and
    `WholeBodyController._world_to_body` are written against. The output is
    `[forward, left, yaw_rate]`, which is the order `Base` itself takes.

    Two entry points, because the sink differs by caller:

    * `compute()` returns the command and sends nothing. Use it when the
      caller has its own dispatch path -- whole-body control still runs its
      heading-rate, acceleration and yaw-filter chain over the result and
      writes it through the BASE_VEL relay, so the wheels never receive two
      writers.
    * `step()` computes and sends straight to `Base.set_target_base_velocity`,
      for a caller that owns the base outright.
    """

    def __init__(
        self,
        base=None,
        *,
        kp_xy: float = 1.5,
        kd_xy: float = 0.15,
        kp_yaw: float = 2.0,
        kd_yaw: float = 0.2,
        xy_deadband: float = 0.01,
        yaw_deadband: float = 0.02,
        max_lin_vel: float = 0.25,
        max_ang_vel: float = 0.60,
        derivative_tau: float = 0.10,
        max_gap_s: float = 0.25,
        heading_offset: float = -math.pi / 2.0,
    ) -> None:
        self.base = base
        self.kp_xy = float(kp_xy)
        self.kd_xy = float(kd_xy)
        self.kp_yaw = float(kp_yaw)
        self.kd_yaw = float(kd_yaw)
        self.xy_deadband = float(xy_deadband)
        self.yaw_deadband = float(yaw_deadband)
        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)
        self.derivative_tau = float(derivative_tau)
        self.max_gap_s = float(max_gap_s)
        self.heading_offset = float(heading_offset)

        # Never ask for more than the drive is configured to allow. Base takes
        # its own per-axis ceiling ([vx, vy, omega]) at construction and does
        # not enforce it in the control loop, so a controller that ignores it
        # would simply command past it.
        limits = getattr(base, "max_vel", None)
        if limits is not None:
            limits = np.asarray(limits, dtype=float).reshape(-1)
            if limits.size >= 3:
                self.max_lin_vel = min(
                    self.max_lin_vel, float(min(abs(limits[0]), abs(limits[1]))))
                self.max_ang_vel = min(self.max_ang_vel, float(abs(limits[2])))

        self.reset()

    # -- state ---------------------------------------------------------------

    def reset(self) -> None:
        """Forget the measurement history and the last command.

        Every event that breaks the continuity of the measured pose -- the
        base being disabled, an override taking it away, fix_base, an
        e-stop -- must call this, or the first cycle afterwards damps against
        a velocity the base never had.
        """
        self._last_pose: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None
        # Body-frame [forward, left, yaw] of the *measurement*, low-passed.
        self._vel_filt = np.zeros(3, dtype=float)
        self.last_error = np.zeros(3, dtype=float)
        self.last_command = np.zeros(3, dtype=float)

    @property
    def measured_velocity(self) -> np.ndarray:
        """The damping term's view of how the base is moving, body frame."""
        return self._vel_filt.copy()

    # -- control -------------------------------------------------------------

    def compute(
        self,
        target_pose,
        current_pose,
        dt: Optional[float] = None,
        now: Optional[float] = None,
    ) -> np.ndarray:
        """One cycle. Returns `[forward, left, yaw_rate]` (m/s, m/s, rad/s).

        `target_pose` and `current_pose` are both `[x, y, yaw]` in the same
        frame. `dt` is the control period; omit it to measure one off the
        monotonic clock.
        """
        target = self._as_pose(target_pose, "target_pose")
        current = self._as_pose(current_pose, "current_pose")
        now = time.monotonic() if now is None else float(now)

        self._update_derivative(current, now, dt)

        err_fwd, err_left = self._to_body(
            target[0] - current[0], target[1] - current[1], current[2])
        err_yaw = _wrap_pi(float(target[2]) - float(current[2]))
        self.last_error = np.array([err_fwd, err_left, err_yaw], dtype=float)

        # Deadband on the magnitude of the XY error vector, so the direction
        # the caller asked for survives it or nothing does.
        if math.hypot(err_fwd, err_left) <= self.xy_deadband:
            forward = lateral = 0.0
        else:
            forward = self.kp_xy * err_fwd - self.kd_xy * float(self._vel_filt[0])
            lateral = self.kp_xy * err_left - self.kd_xy * float(self._vel_filt[1])
            forward, lateral = self._clamp_linear(forward, lateral)

        if abs(err_yaw) <= self.yaw_deadband:
            yaw_rate = 0.0
        else:
            yaw_rate = self.kp_yaw * err_yaw - self.kd_yaw * float(self._vel_filt[2])
            yaw_rate = float(np.clip(yaw_rate, -self.max_ang_vel, self.max_ang_vel))

        self.last_command = np.array([forward, lateral, yaw_rate], dtype=float)
        return self.last_command.copy()

    def step(
        self,
        target_pose,
        current_pose,
        dt: Optional[float] = None,
        now: Optional[float] = None,
    ) -> np.ndarray:
        """`compute()`, then send the result to the base."""
        command = self.compute(target_pose, current_pose, dt=dt, now=now)
        self.send(command)
        return command

    def send(self, command) -> np.ndarray:
        """Hand a `[forward, left, yaw_rate]` command to the swerve drive."""
        if self.base is None:
            raise RuntimeError("BasePoseController has no base to command")
        command = np.asarray(command, dtype=float).reshape(-1)[:3]
        self.base.set_target_base_velocity(command, smooth=True)
        return command

    def halt(self) -> None:
        """Stop the base and forget the PD's history. Never raises."""
        self.reset()
        if self.base is None:
            return
        try:
            self.base.set_target_base_velocity(np.zeros(3, dtype=float), smooth=True)
        except Exception as exc:  # a stop path must never raise
            print(f"[base] pose controller halt failed: {exc}")

    # -- helpers -------------------------------------------------------------

    def _to_body(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        """Rotate a planar world vector into (forward, left) at `yaw`."""
        phi = float(yaw) + self.heading_offset
        c, s = math.cos(phi), math.sin(phi)
        return (c * float(x) + s * float(y), -s * float(x) + c * float(y))

    def _clamp_linear(self, forward: float, lateral: float) -> tuple[float, float]:
        """Clamp the speed, keeping the direction: as a vector, not per axis."""
        speed = math.hypot(forward, lateral)
        if speed > self.max_lin_vel and speed > 1e-12:
            scale = self.max_lin_vel / speed
            forward, lateral = forward * scale, lateral * scale
        return float(forward), float(lateral)

    def _update_derivative(
        self, current: np.ndarray, now: float, dt: Optional[float]
    ) -> None:
        last_pose, last_time = self._last_pose, self._last_time
        self._last_pose, self._last_time = current.copy(), float(now)

        if last_pose is None or last_time is None:
            return

        step = (float(now) - last_time) if dt is None else float(dt)
        if step <= 0.0 or step > self.max_gap_s:
            # Either time did not advance or the loop stalled. Neither gives a
            # velocity worth damping against.
            self._vel_filt[:] = 0.0
            return

        fwd, left = self._to_body(
            (current[0] - last_pose[0]) / step,
            (current[1] - last_pose[1]) / step,
            current[2],
        )
        raw = np.array(
            [fwd, left, _wrap_pi(float(current[2]) - float(last_pose[2])) / step],
            dtype=float,
        )
        alpha = step / (self.derivative_tau + step) if self.derivative_tau > 0.0 else 1.0
        self._vel_filt += alpha * (raw - self._vel_filt)

    @staticmethod
    def _as_pose(value, name: str) -> np.ndarray:
        pose = np.asarray(value, dtype=float).reshape(-1)
        if pose.size < 3:
            raise ValueError(f"{name} must be [x, y, yaw], got {pose.size} values")
        return pose[:3].astype(float)


class BaseController:
    def __init__(
        self,
        yor,
        base_max_vel,
        base_max_accel,
        origin: tuple[float, float],
        grid_res: float,
        control_hz: int = 30,
        relay_hz: int = 108,
        k_pos: float = 1.5,
        ki_pos: float = 0.01,
        kd_pos: float = 0.15,
        k_theta: float = 2.1,
        ki_theta: float = 0.01,
        kd_theta: float = 0.2,
        drive_vel_scale: Optional[float] = None,
        pos_tol: float = 0.05,   # comfortably above SLAM pose noise
        theta_tol: float = 0.03,
    ):
        self.origin = origin
        self.grid_res = grid_res
        # Two rates, because this loop does two unrelated jobs.
        #
        # `rate` paces the navigation modes (MOVE_TO, PATH_FOLLOWING). Their
        # PIDs close on the Odin SLAM pose, which arrives at 20 Hz, and the
        # gains below were tuned against that. Running them faster does not
        # produce new information: the pose is unchanged on the extra cycles,
        # so `(e - prev_e)/dt` reads zero four times and then spikes on the
        # fifth, and `vel_alpha` — an EMA applied once per tick — smooths over
        # a five-times-shorter window than it was tuned for.
        #
        # `relay_rate` paces BASE_VEL, which is not a controller at all: it
        # forwards whatever velocity was last written to `target_velocity`
        # straight through to the wheels. The whole-body loop writes that
        # attribute at 108 Hz (robot/wholebody_control.py), so anything slower
        # here silently discards solver output — the former 20 Hz relay let most
        # base velocities be overwritten before the wheels ever saw them. This
        # is the one path that has to keep up with the producer.
        self.rate = RateLimiter(control_hz, name="BaseController")
        self.relay_rate = RateLimiter(relay_hz, name="BaseController-relay",
                                      warn=False)

        self.yor = yor
        # drive_vel_scale travels with the PID manifest (robot/yor.py reads it
        # from the manifest); None keeps base_motor's built-in default.
        self.base = Base(max_vel=base_max_vel, max_accel=base_max_accel,
                         **({} if drive_vel_scale is None
                            else {"drive_vel_scale": float(drive_vel_scale)}))
        self.slam_sub = None

        self.pos_tol = pos_tol
        self.theta_tol = theta_tol

        # Weight on the *new* PID output; the remainder carries the previous
        # command forward. This exists to take the edge off the derivative
        # spike the 20 Hz SLAM pose puts on a 30 Hz loop, nothing more — the
        # jerk-limited ramp in Base.control_loop is what actually shapes the
        # motion now, and unlike this filter it stops lagging once the command
        # settles. Kept deliberately light (was 0.5, and 0.2 before that): a
        # heavy EMA here delays *every* sample, which reads as the base
        # responding late to the operator rather than as smoothness.
        self.vel_alpha = 0.8

        self._vel_lock = threading.Lock()
        self.last_target_velocity = np.zeros(3, dtype=float)
        self.last_t = time.monotonic()
        self.heading_gate = math.radians(25.0)

        self.vmin = 0.05
        self.vmax = 0.35
        self.omegamin = 0.05
        self.omegamax = 1.0

        i_limit_lin = 0.08
        i_limit_yaw = 0.15
        self.pid_x = PID(k_pos, ki_pos, kd_pos, i_limit=i_limit_lin)
        self.pid_y = PID(k_pos, ki_pos, kd_pos, i_limit=i_limit_lin)
        self.pid_th = PID(k_theta, ki_theta, kd_theta, i_limit=i_limit_yaw)

        self.mode = "BASE_VEL"
        self.target_velocity = np.zeros(3, dtype=float)

        self._path_world = None
        self._goal = None

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self.heading_gate_on = math.radians(35.0)
        self.heading_gate_off = math.radians(5.0)
        self._rot_only = False
        self._worker = threading.Thread(target=self._run, name="PathUpdater", daemon=True)
        self._worker.start()

        self._nav_lock = threading.Lock()
        self._nav = None

    def slam_sub_init(self, timeout_s: float = 1.0):
        if self.slam_sub is not None:
            return

        done = threading.Event()
        out = {"sub": None, "err": None}

        def _worker():
            try:
                out["sub"] = Subscriber(
                    host=THOR_IP,
                    port=SLAM_PUB_PORT,
                    topics=[POSE_TOPIC],
                    buffer=False,
                )
            except Exception as e:
                out["err"] = e
            finally:
                done.set()

        threading.Thread(target=_worker, daemon=True).start()

        if not done.wait(timeout_s):
            return
        if out["sub"] is None:
            return

        self.slam_sub = out["sub"]
        print("SLAM pose subscriber initialized")
        return

    def reset_pids(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_th.reset()
        self.last_target_velocity = np.zeros(3, dtype=float)
        self.last_t = time.monotonic()

    def get_nav_debug(self):
        with self._nav_lock:
            return None if self._nav is None else dict(self._nav)

    def stop(self):
        self._stop_evt.set()
        if self._worker.is_alive():
            self._worker.join(timeout=1.5)

    def _run(self):
        Ld_base: float = 0.32
        Ld_gain: float = 0.4
        Ld_min: float = 0.30   # was 0.20 — higher min prevents chasing a point under the nose
        Ld_max: float = 0.40
        end_dist_tol: float = 0.25  # was 0.08 — switch to MOVE_TO further out for smooth braking
        last_path_uid = None

        v_meas_filt = 0.0
        last_xy = None
        last_pose_t = None
        tau_v = 0.30

        last_path_sig = None
        waypoints = None
        s_cum = None
        total_len = 0.0
        prev_mode = self.mode

        last_sub_try = 0.0

        while not self._stop_evt.is_set():
            if self.mode == "BASE_VEL":
                self.base.set_target_base_velocity(
                    np.asarray(self.target_velocity, dtype=float), smooth=True
                )
                self.relay_rate.sleep()
                continue

            if self.slam_sub is None:
                now = time.monotonic()
                if (now - last_sub_try) > 1.0:
                    last_sub_try = now
                    self.slam_sub_init()

                self.yor.pose = None
                self.base.set_target_base_velocity(np.zeros(3, dtype=float), smooth=True)
                self.rate.sleep()
                continue

            try:
                pose = get_pose(self.slam_sub)
                self.yor.pose = pose
                translation, theta, T_base = pose
                theta = _wrap_pi(theta + math.pi)
                x = float(translation[0])
                y = float(translation[2])
            except Exception:
                self.yor.pose = None
                self.base.set_target_base_velocity(np.zeros(3, dtype=float), smooth=True)
                self.rate.sleep()
                continue

            if self.mode != prev_mode:
                self.reset_pids()
                prev_mode = self.mode

            tx, ty, tth = x, y, None
            dist_goal = None
            stop = False
            to_tgt = np.array([0.0, 0.0], dtype=float)

            match self.mode:
                case "MOVE_TO":
                    goal = self._goal
                    if not goal:
                        stop = True
                    else:
                        tx, ty, tth = goal
                        dist_goal = None
                        tx, ty = float(tx), float(ty)

                        heading_freeze_r = 0.10
                        dist = math.hypot(tx - x, ty - y)
                        dist_goal = dist

                        if tth is None:
                            to_tgt = np.array([tx - x, ty - y], dtype=float)
                            d_theta = heading_error_from_dir(
                                T_base, to_tgt, forward_col=0, flip_sign=True
                            )

                            if dist < self.pos_tol:
                                stop = True
                        else:
                            to_tgt = dir_from_yaw(tth + math.pi / 2.0) #to_tgt = dir_from_yaw(tth)
                            d_theta = heading_error_from_dir(
                                T_base, to_tgt, forward_col=0, flip_sign=True
                            )

                            if dist < self.pos_tol and abs(d_theta) < self.theta_tol:
                                stop = True

                case "PATH_FOLLOWING":
                    path_world = self._path_world
                    if not path_world:
                        self.base.set_target_base_velocity(np.zeros(3, dtype=float), smooth=True)
                        self.rate.sleep()
                        continue

                    path_uid = (len(path_world), path_world[0], path_world[-1])
                    if path_uid != last_path_uid:
                        d_start = math.hypot(
                            float(path_world[0][0]) - x, float(path_world[0][1]) - y
                        )
                        d_end = math.hypot(
                            float(path_world[-1][0]) - x, float(path_world[-1][1]) - y
                        )
                        last_path_uid = (len(path_world), path_world[0], path_world[-1])

                    path_sig = (len(path_world), path_world[0], path_world[-1])
                    if path_sig != last_path_sig or waypoints is None:
                        waypoints = [
                            np.array([float(px), float(pz)], dtype=float)
                            for (px, pz) in path_world
                        ]
                        s_cum, total_len = _cumlen(waypoints)
                        last_path_sig = path_sig

                    s_closest, _, _, _ = _closest_point_on_polyline(x, y, waypoints, s_cum)

                    t_now = time.monotonic()
                    if last_xy is None:
                        last_xy, last_pose_t = (x, y), t_now

                    dt_pose = max(1e-3, t_now - last_pose_t)
                    dist_xy = math.hypot(x - last_xy[0], y - last_xy[1])
                    v_inst = dist_xy / dt_pose
                    alpha_v = dt_pose / (tau_v + dt_pose)
                    v_meas_filt = (1 - alpha_v) * v_meas_filt + alpha_v * v_inst
                    last_xy, last_pose_t = (x, y), t_now

                    Ld = float(np.clip(Ld_base + Ld_gain * v_meas_filt, Ld_min, Ld_max))
                    s_tgt = s_closest + Ld

                    if (total_len - s_closest) < end_dist_tol:
                        last_pt = path_world[-1]
                        self._goal = (float(last_pt[0]), float(last_pt[1]), None)
                        self.mode = "MOVE_TO"
                        self.rate.sleep()
                        continue

                    p_tgt, tan = _point_at_s(waypoints, s_cum, s_tgt)
                    tx, ty = float(p_tgt[0]), float(p_tgt[1])

                    tan = np.asarray(tan, dtype=float)
                    to_tgt = np.array([tx - x, ty - y], dtype=float)
                    if float(tan @ to_tgt) < 0.0:
                        tan = -tan

                    tth = math.atan2((tx - x), (ty - y))

                case _:
                    print("Nav mode set is not in [BASE_VEL, PATH_FOLLOWING, MOVE_TO]")
                    self.rate.sleep()
                    continue

            if stop:
                self.mode = "BASE_VEL"
                self.target_velocity = np.zeros(3, dtype=float)
                self.base.set_target_base_velocity(self.target_velocity, smooth=True)
                self.rate.sleep()
                continue

            path_copy = None
            if self._path_world:
                path_copy = [(float(px), float(py)) for (px, py) in self._path_world]

            debug_info = {
                "mode": str(self.mode),
                "path_world": path_copy,
                "lookahead_xz": (float(tx), float(ty)),
                "pose_xz": (float(x), float(y)),
                "yaw": float(theta),
                "yaw_des": (None if tth is None else float(tth)),
                "rot_only": bool(self._rot_only),
            }

            d_theta = heading_error_from_dir(
                T_base,
                to_tgt,
                forward_col=0,
                flip_sign=True,
            )

            if self._rot_only:
                if abs(d_theta) < self.heading_gate_off:
                    self._rot_only = False
            else:
                if abs(d_theta) > self.heading_gate_on:
                    self._rot_only = True

            rotation_only = self._rot_only

            dx = float(tx - x)
            dz = float(ty - y)

            R_bw = T_base[:3, :3]
            d_world = np.array([dx, 0.0, dz], dtype=float)
            d_body = R_bw.T @ d_world

            d_fwd = -float(d_body[2])
            d_left = float(d_body[0])

            now = time.monotonic()
            dt = max(1e-3, min(0.25, now - self.last_t))
            self.last_t = now

            if rotation_only:
                self.pid_x.reset()
                self.pid_y.reset()
                vx, vy = 0.0, 0.0
            else:
                vx = self.pid_x.step(d_fwd, dt)
                vy = self.pid_y.step(d_left, dt)

            omega = self.pid_th.step(d_theta, dt)
            def _soft_clip(v, v_min, v_max):
                a_v = abs(v)
                if a_v < 1e-4:
                    return 0.0
                if a_v < v_min:
                    return v # Don't boost to v_min if it's already small
                return np.sign(v) * float(np.clip(a_v, v_min, v_max))

            vx = _soft_clip(vx, self.vmin, self.vmax)
            vy = _soft_clip(vy, self.vmin, self.vmax)
            omega = _soft_clip(omega, self.omegamin, self.omegamax)

            debug_info["d_fwd"] = float(d_fwd)
            debug_info["d_left"] = float(d_left)
            debug_info["d_theta_deg"] = float(math.degrees(d_theta))
            debug_info["cmd_vx"] = float(vx)
            debug_info["cmd_vy"] = float(vy)
            debug_info["cmd_omega"] = float(omega)

            with self._nav_lock:
                self._nav = debug_info

            if rotation_only:
                self.target_velocity = np.array([0.0, 0.0, omega], dtype=float)
                self.last_target_velocity = self.target_velocity
            else:
                new_cmd = np.array([vy, vx, omega], dtype=float)
                self.target_velocity = (
                    self.vel_alpha * new_cmd
                    + (1.0 - self.vel_alpha) * self.last_target_velocity
                )
                self.last_target_velocity = self.target_velocity

            self.base.set_target_base_velocity(self.target_velocity, smooth=True)
            self.rate.sleep()


if __name__ == "__main__":
    # Intentionally minimal: the controller depends on a yor object from your runtime.
    pass
