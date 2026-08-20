"""
filters.py — Conditioning for teleop input streams.

Quest controller poses come straight off the headset's tracker at ~72 Hz,
carrying a couple of millimetres of jitter while the hand is held still plus
the occasional tracking glitch. Pushed through the clutch as an EE target, the
jitter becomes a buzz in the arms (the IK faithfully chases noise) and a glitch
becomes a lurch.

The 1€ filter (Casiez et al., CHI 2012) is the right tool here: a low-pass
whose cutoff rises with the measured speed, so it smooths hard while the
controller is nearly still and gets out of the way the moment it moves. A
fixed low-pass has to trade those two against each other.

  OneEuroFilter  the scalar / vector filter.
  PoseFilter     the same idea on an SE3 — translation as a vector, rotation
                 slerped by an adaptive weight — plus a gate that drops
                 samples no hand could have produced (tracking dropouts).

Both are driven by message *arrival* timestamps rather than a fixed dt, so a
dropped packet or a jittery publisher does not silently change how much
smoothing is applied.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from mink.lie import SE3, SO3


def _alpha(cutoff: float, dt: float) -> float:
    """Weight of the newest sample in a first-order low-pass at `cutoff` Hz."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Speed-adaptive first-order low-pass for scalars or fixed-length vectors.

    For vectors the cutoff is driven by the *norm* of the derivative, so every
    component is weighted identically — per-axis cutoffs would lag each axis
    differently and bend the direction of a diagonal motion.

    Args:
        min_cutoff: cutoff (Hz) at zero speed. Lower = smoother but laggier.
        beta: how fast the cutoff opens with speed (Hz per unit/s). Higher =
            more responsive while moving, at the cost of passing more noise.
        d_cutoff: cutoff (Hz) of the low-pass on the derivative estimate.
    """

    def __init__(self, min_cutoff: float = 3.0, beta: float = 8.0,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        self._x_hat: Optional[np.ndarray] = None
        self._dx_hat: Optional[np.ndarray] = None

    def __call__(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x_hat is None or dt <= 0.0:
            self._x_hat = x.copy()
            self._dx_hat = np.zeros_like(x)
            return self._x_hat.copy()

        a_d = _alpha(self.d_cutoff, dt)
        dx = (x - self._x_hat) / dt
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx_hat))
        a = _alpha(cutoff, dt)
        self._x_hat = a * x + (1.0 - a) * self._x_hat
        return self._x_hat.copy()


class PoseFilter:
    """1€ filter over a stream of SE3 poses, with a tracking-glitch gate.

    Feed it each pose as it arrives, tagged with the time it arrived; it hands
    back the pose to actually use. A sample that would need an impossible hand
    speed is rejected and the previous output repeated — that is what a
    tracking dropout looks like on the wire, and repeating beats lurching. If
    the rejections keep coming (`max_rejects` in a row) or the stream goes
    quiet for `max_gap`, the filter re-locks onto whatever is arriving now
    instead of fighting it forever.

    Args:
        min_cutoff, beta, d_cutoff: 1€ parameters for the translation, in
            metres (see OneEuroFilter).
        rot_min_cutoff, rot_beta: the same for the rotation, where speed is the
            angular rate in rad/s.
        max_speed: hand speed (m/s) above which a sample is treated as a glitch.
        jump_floor: absolute slack (m) added to that gate, so timestamp noise
            on a fast sample does not trip it.
        max_gap: silence (s) after which the stream counts as new, not delayed.
        max_rejects: consecutive rejections tolerated before re-locking.
    """

    def __init__(self, min_cutoff: float = 3.0, beta: float = 8.0,
                 rot_min_cutoff: float = 3.0, rot_beta: float = 1.5,
                 d_cutoff: float = 1.0, max_speed: float = 6.0,
                 jump_floor: float = 0.02, max_gap: float = 0.25,
                 max_rejects: int = 5):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.rot_min_cutoff = float(rot_min_cutoff)
        self.rot_beta = float(rot_beta)
        self.d_cutoff = float(d_cutoff)
        self.max_speed = float(max_speed)
        self.jump_floor = float(jump_floor)
        self.max_gap = float(max_gap)
        self.max_rejects = int(max_rejects)
        self.rejected = 0   # samples dropped as glitches, cumulative
        self.relocks = 0    # times the filter gave up and re-seeded
        self.reset()

    def reset(self) -> None:
        self._T_hat: Optional[SE3] = None
        self._t_prev = 0.0
        self._v_hat = np.zeros(3)
        self._w_hat = 0.0
        self._reject_streak = 0

    def _lock(self, T: SE3, t: float) -> SE3:
        """(Re)seed the filter state on `T`, passing it straight through."""
        if self._T_hat is not None:
            self.relocks += 1
        self._T_hat = T
        self._t_prev = t
        self._v_hat = np.zeros(3)
        self._w_hat = 0.0
        self._reject_streak = 0
        return T

    def __call__(self, T: SE3, t: float) -> SE3:
        if self._T_hat is None:
            return self._lock(T, t)

        dt = t - self._t_prev
        if dt <= 0.0:
            return self._T_hat  # duplicate or out-of-order sample
        if dt > self.max_gap:
            return self._lock(T, t)  # the stream stalled — start clean

        p_prev = self._T_hat.translation()
        R_prev = self._T_hat.rotation()
        p = T.translation()
        R = T.rotation()

        if np.linalg.norm(p - p_prev) > self.jump_floor + self.max_speed * dt:
            self.rejected += 1
            self._reject_streak += 1
            self._t_prev = t
            if self._reject_streak >= self.max_rejects:
                return self._lock(T, t)  # not a glitch: the hand really is there
            return self._T_hat
        self._reject_streak = 0

        a_d = _alpha(self.d_cutoff, dt)

        # Translation: 1€ with a cutoff shared across the three axes.
        self._v_hat = a_d * (p - p_prev) / dt + (1.0 - a_d) * self._v_hat
        a_p = _alpha(
            self.min_cutoff + self.beta * float(np.linalg.norm(self._v_hat)), dt)
        p_hat = a_p * p + (1.0 - a_p) * p_prev

        # Rotation: same adaptive weight, applied as a geodesic step (slerp).
        omega = float(np.linalg.norm((R_prev.inverse() @ R).log())) / dt
        self._w_hat = a_d * omega + (1.0 - a_d) * self._w_hat
        a_r = _alpha(self.rot_min_cutoff + self.rot_beta * self._w_hat, dt)
        R_hat: SO3 = R_prev.interpolate(R, min(max(a_r, 0.0), 1.0))

        self._t_prev = t
        self._T_hat = SE3.from_rotation_and_translation(R_hat, p_hat)
        return self._T_hat
