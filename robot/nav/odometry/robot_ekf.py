"""
robot_ekf.py — 3-state Extended Kalman Filter for swerve drive.

State:       x = [px, pz, theta]     (world frame, Y-up: robot moves in XZ plane)
Control:     u = [vx, vy, omega]     (robot frame, from swerve FK)
Measurement: z = [px, pz, theta]     (from VIO visual odometry)

Coordinate convention
---------------------
Theta (yaw) follows the VIO measurement convention:
  theta = 0   →  robot faces world +X
  theta = pi/2 →  robot faces world -Z
  Extracted as arctan2(-T[2,0], T[0,0]) from the Y-up world→base transform.

The predict kinematics are NOT the textbook (cos, −sin) pair — the device is
mounted facing backwards relative to the robot, so forward motion decreases
world Z:
  dx = (−vx·sin(θ) − vy·cos(θ)) · dt
  dz = (−vx·cos(θ) + vy·sin(θ)) · dt
where vx = robot forward speed, vy = robot left-lateral speed (from SwerveOdom).
This must stay identical to the world integration in swerve_odom.py:202-203 —
if you re-derive one, re-derive both.

Features
--------
- Motion-proportional process noise Q (scales with |u|⋅dt)
- Adaptive measurement noise via R_override parameter (caller inflates R on bad VIO frames)
- Mahalanobis chi-squared gate: rejects statistical outliers (loop-closure jumps)
- Joseph-form covariance update for numerical stability
- ZUPT: zero-velocity update that tightly anchors state when robot is stationary

Carpet tuning
-------------
Default Q values are increased ~3–4× vs. the original calibration to reflect two
carpet-specific sources of encoder uncertainty:
  1. Wheel compression under load changes effective rolling radius by 2–5 %.
  2. Lateral slip on carpet is significantly higher than on hard floors.
These larger Q values make the filter appropriately skeptical of encoder predictions
on carpet, letting the VIO (when confident) carry more weight.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


def wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


class RobotEKF:
    """3-state EKF: predict with swerve odometry, update with VIO pose.

    Coordinate convention: Y-up world frame.  Robot moves in the XZ plane.
    State = [px, pz, yaw_around_y].
    """

    def __init__(
        self,
        # Process noise coefficients (per unit travel) — carpet-tuned
        # Increased ~3-4× vs. hard-floor values to reflect:
        #   - wheel radius uncertainty from carpet compression (~2-5%)
        #   - lateral slip higher on carpet than hardwood
        q_x:  float = 0.030,    # was 0.008697 — carpet compression ~3.5×
        q_y:  float = 0.080,    # was 0.027119 — lateral slip on carpet ~3×
        q_th: float = 0.020,    # was 0.004949 — turning resistance varies ~4×
        # VIO measurement noise std (these are the MINIMUM / best-case values;
        # the update() caller can pass R_override to inflate them on bad frames)
        r_xy: float = 0.0024,   # m  — nominal VIO position noise at full confidence
        r_th: float = 0.0006,   # rad — nominal VIO heading noise
    ):
        # State [px, pz, theta]
        self.x = np.zeros(3)
        # Covariance — start uncertain so first VIO fix dominates
        self.P = np.eye(3) * 0.1

        # Process noise coefficients (per unit travel)
        self.q_x  = q_x
        self.q_y  = q_y
        self.q_th = q_th

        # Nominal measurement noise matrix (R at 100 % confidence)
        self.R = np.diag([r_xy ** 2, r_xy ** 2, r_th ** 2])

        # Diagnostic counters
        self._predict_calls = 0
        self._update_calls  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset filter state and covariance."""
        self.x = np.array([x, y, theta], dtype=float)
        self.P = np.eye(3) * 0.1

    def predict(self, u: np.ndarray, dt: float) -> None:
        """EKF predict step using swerve odometry control input.

        Args:
            u:  [vx, vy, omega] in robot frame (from SwerveOdom.forward_kinematics)
                vx = forward speed, vy = left-lateral speed, omega = CCW yaw rate
            dt: time step [s]

        Kinematics (F6 — empirically verified against VIO ground truth):
            dx = −vx·sin(θ) − vy·cos(θ)
            dz = −vx·cos(θ) + vy·sin(θ)
        """
        if dt < 1e-6 or not np.isfinite(dt):
            return

        if not np.all(np.isfinite(u)):
            return

        self._predict_calls += 1
        if self._predict_calls == 1:
            print(f"[EKF.predict] *** FIRST PREDICT CALL — encoder data is reaching EKF ***")
            print(f"[EKF.predict]   u=[{u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}]  dt={dt:.4f}s")
        if self._predict_calls % 200 == 0:
            print(f"[EKF.predict] call={self._predict_calls}  "
                  f"state=[{self.x[0]:.3f}, {self.x[1]:.3f}, {float(np.degrees(self.x[2])):.1f}°]  "
                  f"u=[{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}]")

        vx, vy, omega = float(u[0]), float(u[1]), float(u[2])
        theta = self.x[2]
        c = np.cos(theta)
        s = np.sin(theta)

        # Process model — F6 (empirically verified against VIO ground truth):
        #   world_X = −vx·sin(θ) − vy·cos(θ)
        #   world_Z = −vx·cos(θ) + vy·sin(θ)
        # Must match swerve_odom.py integration exactly.
        dx  = (-vx * s - vy * c) * dt   # world X
        dz  = (-vx * c + vy * s) * dt   # world Z
        dth =  omega * dt

        self.x[0] += dx
        self.x[1] += dz
        self.x[2] += dth
        self.x[2]  = wrap_pi(self.x[2])

        if not np.all(np.isfinite(self.x)):
            # Fallback for numerical instability
            self.x = np.nan_to_num(self.x)

        # Jacobian F = df/dx  (∂[dx,dz]/∂θ)
        # d(−vx·s − vy·c)/dθ = −vx·cos(θ) + vy·sin(θ)
        # d(−vx·c + vy·s)/dθ =  vx·sin(θ) + vy·cos(θ)
        F = np.array([
            [1.0, 0.0, (-vx * c + vy * s) * dt],
            [0.0, 1.0, ( vx * s + vy * c) * dt],
            [0.0, 0.0,  1.0],
        ])

        # Process noise Q — scales with motion magnitude
        # The constant floor terms (not 1e-6!) are critical: they represent
        # unconditional uncertainty (motor backlash, carpet deformation, IMU
        # drift) that prevent the covariance from collapsing after a VIO update
        # and locking out the Mahalanobis gate.
        Q_FLOOR_POS = 1e-4    # ~1 cm/s position wander
        Q_FLOOR_YAW = 5e-4   # ~1.3°/s heading wander
        Q = np.diag([
            self.q_x  * abs(vx)    * dt + Q_FLOOR_POS * dt,
            self.q_y  * abs(vy)    * dt + Q_FLOOR_POS * dt,
            self.q_th * abs(omega) * dt + Q_FLOOR_YAW * dt,
        ])

        new_P = F @ self.P @ F.T + Q
        if np.all(np.isfinite(new_P)):
            self.P = new_P
        else:
            # If covariance exploded, reset it to something sane
            self.P = np.eye(3) * 0.1

    def update(
        self,
        z: np.ndarray,
        R_override: Optional[np.ndarray] = None,
        gate_chi2: Optional[float] = 12.0,
    ) -> bool:
        """EKF update step using VIO pose measurement.

        Args:
            z:           [px, pz, theta] in world frame
            R_override:  if provided, use this R instead of self.R (adaptive noise)
            gate_chi2:   Mahalanobis chi-squared threshold (3 DOF).
                         Default 12.0 ≈ 99.3 % — rejects statistical outliers
                         (e.g. loop-closure teleport jumps).
                         Pass None to disable gating.

        Returns:
            True if measurement was accepted, False if gated out.
        """
        z    = np.asarray(z, dtype=float)
        if not np.all(np.isfinite(z)):
            return False

        H    = np.eye(3)
        R_use = R_override if R_override is not None else self.R

        # Innovation
        y    = z - H @ self.x
        y[2] = wrap_pi(y[2])      # angle wrapping

        # Innovation covariance
        S = H @ self.P @ H.T + R_use

        # ---- Mahalanobis gating ----
        if gate_chi2 is not None and gate_chi2 > 0.0:
            try:
                d2 = float(y @ np.linalg.solve(S, y))
            except np.linalg.LinAlgError:
                return False
            if d2 > gate_chi2:
                return False   # outlier — reject this measurement

        # ---- Kalman gain ----
        K = np.linalg.solve(S.T, H @ self.P.T).T

        # ---- State update ----
        self.x    = self.x + K @ y
        self.x[2] = wrap_pi(self.x[2])

        # ---- Covariance update (Joseph form for numerical stability) ----
        I_KH   = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_use @ K.T

        self._update_calls += 1
        if self._update_calls == 1:
            print(f"[EKF.update] *** FIRST VIO UPDATE ACCEPTED ***")
            print(f"[EKF.update]   z=[{z[0]:.3f}, {z[1]:.3f}, {float(np.degrees(z[2])):.1f}°]  "
                  f"innov_pos={float(np.hypot(y[0], y[1])):.4f}m")
        if self._update_calls % 50 == 0:
            print(f"[EKF.update] call={self._update_calls}  "
                  f"innov_pos={float(np.hypot(y[0], y[1])):.4f}m  "
                  f"innov_yaw={float(np.degrees(y[2])):.2f}°  "
                  f"state=[{self.x[0]:.3f}, {self.x[1]:.3f}, {float(np.degrees(self.x[2])):.1f}°]")

        return True

    def update_orb(
        self,
        z: np.ndarray,
        R_orb: Optional[np.ndarray] = None,
        gate_chi2: Optional[float] = 50.0,
        is_loop_closure: bool = False,
    ) -> bool:
        """EKF update step using an ORB-SLAM3 pose measurement.

        Identical math to update() but with a separate, tighter noise model
        reflecting that ORB-SLAM3 keyframe poses are globally consistent —
        especially after loop closures.

        Args:
            z:               [px, pz, theta] in the same Y-up world frame as the EKF.
            R_orb:           Measurement noise covariance (3×3). If None, uses a
                             default tuned for ORB-SLAM3 RGB-D accuracy (~5 mm, ~0.1°).
            gate_chi2:       Mahalanobis chi-squared threshold (3 DOF).
                             Wider than VIO gate (50 vs 12) to accept large loop-closure
                             corrections that would be rejected as outliers otherwise.
                             Pass None to disable gating entirely for loop closures.
            is_loop_closure: When True, the gate is disabled and the noise is further
                             tightened so the filter snaps to the globally-consistent
                             ORB-SLAM3 estimate without resistance.

        Returns:
            True if the measurement was accepted and applied; False if gated out.
        """
        z = np.asarray(z, dtype=float)
        if not np.all(np.isfinite(z)):
            return False

        if R_orb is None:
            # ORB-SLAM3 RGB-D typical accuracy:
            #   position: ~5 mm (0.005 m) at short baselines
            #   heading:  ~0.1° (0.00175 rad)
            R_orb = np.diag([0.005 ** 2, 0.005 ** 2, 0.00175 ** 2])

        if is_loop_closure:
            # Loop closures are globally consistent — trust them unconditionally.
            # Tighten R further and skip gating.
            R_orb = R_orb * 0.1
            gate_chi2 = None

        H = np.eye(3)
        y = z - H @ self.x
        y[2] = wrap_pi(y[2])  # angle wrapping

        S = H @ self.P @ H.T + R_orb

        # Mahalanobis gating
        if gate_chi2 is not None and gate_chi2 > 0.0:
            try:
                d2 = float(y @ np.linalg.solve(S, y))
            except np.linalg.LinAlgError:
                return False
            if d2 > gate_chi2:
                innov_pos = float(np.hypot(y[0], y[1]))
                innov_yaw = float(np.degrees(abs(y[2])))
                print(
                    f"[EKF.update_orb] GATED (chi2={d2:.1f} > {gate_chi2}): "
                    f"innov_pos={innov_pos:.3f}m  innov_yaw={innov_yaw:.1f}°"
                )
                return False

        K = np.linalg.solve(S.T, H @ self.P.T).T
        self.x = self.x + K @ y
        self.x[2] = wrap_pi(self.x[2])

        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_orb @ K.T

        innov_pos = float(np.hypot(y[0], y[1]))
        innov_yaw = float(np.degrees(abs(y[2])))
        lc_tag = " [LOOP-CLOSURE]" if is_loop_closure else ""
        if innov_pos > 0.02 or innov_yaw > 1.0:
            print(
                f"[EKF.update_orb]{lc_tag} innov_pos={innov_pos:.3f}m  "
                f"innov_yaw={innov_yaw:.2f}°  "
                f"new_state=[{self.x[0]:.3f}, {self.x[1]:.3f}, "
                f"{float(np.degrees(self.x[2])):.1f}°]"
            )
        return True

    def zupt(self, R_zupt: Optional[np.ndarray] = None) -> None:
        """Zero-Velocity Update: assert the robot hasn't moved.

        Injects a pseudo-measurement z = current_state with very tight R,
        resetting accumulated drift while the robot is stationary.

        Args:
            R_zupt: custom noise for ZUPT. Default: ~1 mm position, ~0.03 deg heading.
        """
        if R_zupt is None:
            R_zupt = np.diag([0.001 ** 2, 0.001 ** 2, 0.0005 ** 2])
        # Assert "I'm here" — no gating (we trust the stopped state)
        self.update(self.x.copy(), R_override=R_zupt, gate_chi2=None)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return current state estimate [px, pz, theta]."""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Return current 3×3 covariance matrix."""
        return self.P.copy()

    def get_uncertainty(self) -> np.ndarray:
        """Return 1-sigma uncertainties [sigma_px, sigma_pz, sigma_theta]."""
        return np.sqrt(np.diag(self.P))