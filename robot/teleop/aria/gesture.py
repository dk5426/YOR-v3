"""gesture.py — hand poses read off the published landmarks, client-side.

The publisher owns the *shaka* toggle and sends the result as `paused`; it does
not send the landmarks' other meanings. This module reads the ones the robot
side needs out of `kp_mp`, which arrives on every fresh payload whether or not
the side is paused (pausing freezes `qpos`, not the landmarks).

Detecting here rather than in the publisher keeps the home gesture in the repo
that owns what home *means*, and adds no field to the wire. The bend measure is
the same one aria2robot's `utils/gesture.py` uses for the shaka, deliberately:
two detectors disagreeing about what "curled" means is a bug that only shows up
on somebody's hand.

Bend cosines are dot products of landmark differences, so they are invariant
under rotation and translation -- odom-frame landmarks give the same answer as
device-frame ones, and neither needs a calibration.
"""

from __future__ import annotations

import numpy as np

# MediaPipe joint chains for the bend cosine: (mcp, hinge, tip). The thumb
# hinges at IP (slot 3), the fingers at PIP
_CHAIN = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

# straight finger -> vectors oppose -> cos ~ -1;  curled -> cos ~ +1
_EXT_COS = -0.3    # extended if cos <= this (angle >= ~105 deg)
_CURL_COS = 0.05   # curled if cos >= this (angle <= ~87 deg)
_EPS = 1e-6


def _bend_cos(mcp: np.ndarray, hinge: np.ndarray, tip: np.ndarray) -> float:
    v1, v2 = mcp - hinge, tip - hinge
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < _EPS or n2 < _EPS:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def bend_ratios(kp: np.ndarray) -> dict[str, float]:
    """Per-finger bend cosine: ~-1 extended, ~+1 curled. Debug aid."""
    if kp is None or np.shape(kp) != (21, 3):
        return {}
    return {name: _bend_cos(kp[a], kp[b], kp[c])
            for name, (a, b, c) in _CHAIN.items()}


def is_thumbs_up(kp: np.ndarray, ext_cos: float = _EXT_COS,
                 curl_cos: float = _CURL_COS) -> bool:
    """Thumb extended, all four fingers curled.

    Mutually exclusive with the shaka by construction -- that one wants the
    pinky extended, this one wants it curled -- so the two can never both read
    true on one hand and the publisher's toggle is never ambiguous.
    """
    if kp is None or np.shape(kp) != (21, 3):
        return False
    cos = bend_ratios(np.asarray(kp))
    if cos["thumb"] > ext_cos:
        return False
    return all(cos[f] >= curl_cos for f in ("index", "middle", "ring", "pinky"))


class HoldTrigger:
    """Fires once when a gesture has been held `dwell_s`; re-arms on release.

    Args:
        release_s: how long the gesture must read absent before an in-progress
            hold is abandoned. Hand tracking drops frames often enough that a
            strict reset made a genuine hold restart partway through and never
            fire, so anything shorter is treated as tracking noise, not intent.
    """

    def __init__(self, dwell_s: float = 1.0, release_s: float = 0.2):
        self.dwell_s = float(dwell_s)
        self.release_s = float(release_s)
        self._hold_start: float | None = None
        self._off_start: float | None = None
        self._fired = False

    def update(self, gesture_on: bool, now: float) -> bool:
        """True on the single tick the hold completes."""
        if not gesture_on:
            if self._off_start is None:
                self._off_start = now
            if (now - self._off_start) >= self.release_s:
                self._hold_start = None
                self._fired = False
            return False
        # Same release test on the way back on, so a gap in calls counts as
        # absence rather than silently preserving a hold across it
        if self._off_start is not None and (now - self._off_start) >= self.release_s:
            self._hold_start = None
            self._fired = False
        self._off_start = None
        if self._hold_start is None:
            self._hold_start = now
            return False
        if not self._fired and (now - self._hold_start) >= self.dwell_s:
            self._fired = True
            return True
        return False

    def reset(self) -> None:
        """Abandon any hold in progress, without firing."""
        self._hold_start = None
        self._off_start = None
        self._fired = False

    def latch(self) -> None:
        """Mark as already fired: no further fire until the gesture is released.

        Not the same as reset(), which re-arms. Resetting a trigger whose
        gesture is still being held starts a fresh dwell under the standing
        thumb and fires again a second later -- on hardware that is a home
        sequence repeating for as long as the hand is up.
        """
        self._fired = True
        self._off_start = None

    def held_for(self, now: float) -> float:
        """Seconds the current hold has run, 0 if none."""
        return 0.0 if self._hold_start is None else max(0.0, now - self._hold_start)


class HomeGesture:
    """Both thumbs up, both hands disengaged -> home both arms.

    Homing is all-or-nothing on purpose. `home_arms` is not "the two single
    homes together": every variant on the node runs the same preamble -- lock
    the base, tear down the whole-body controller, drive the lift to 450 mm --
    so homing one arm costs exactly what homing both costs, and moves the lift
    either way. Making both hands ask for it means that never happens as a
    side effect of one thumb.

    Gated on both clutches being *released*, which is what makes it safe
    without a second confirmation: nothing is following either hand at that
    moment, and clutch reseed means re-engaging afterwards is a zero-delta
    anchor on wherever home left the arms. It also keeps the shaka symmetric
    -- disengage stays the stop gesture on both hands, which is what a
    startled operator reaches for.

    A single-hand session (`mapping.hand: left`/`right`) cannot make the
    gesture at all, so `available` comes up False and update() never fires.
    """

    def __init__(self, sides: tuple[str, ...], dwell_s: float = 1.0):
        self.sides = tuple(sides)
        # Two hands or nothing -- a one-handed session has no way to ask.
        self.available = len(self.sides) == 2
        self._both = HoldTrigger(dwell_s)

    def update(self, kp: dict, released: dict, now: float) -> bool:
        """True on the single tick the two-hand hold completes.

        Args:
            kp: side -> (21, 3) landmarks, or None where the hand is unseen.
            released: side -> True when that side's clutch is disengaged.
        """
        if not self.available:
            return False
        # The hold starts when the *second* thumb comes up, so a staggered
        # request still waits out a full dwell with both hands committed.
        # HoldTrigger's release_s absorbs the dropped frames in between.
        up = all(bool(released.get(s)) and is_thumbs_up(kp.get(s))
                 for s in self.sides)
        # HoldTrigger latches itself on fire, so a standing pair of thumbs
        # does not start a fresh dwell and home again a second later
        return self._both.update(up, now)

    def latch(self) -> None:
        """Require a release before the gesture can fire again."""
        self._both.latch()

    def reset(self) -> None:
        self._both.reset()
