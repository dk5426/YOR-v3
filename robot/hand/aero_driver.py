"""aero_driver.py — the Aero Hand Open hardware layer, both hands, one object.

Mirrors `wuji_driver.py`'s shape (same method names: `start`, `send`,
`commanded`, `actual`, `home`, `close`) but shares no base class with it --
aria2robot's own driver split (`WujiDriver | AeroDriver`, a bare union) never
shared one either, so this follows the same convention: duck-typed by method
name, not by inheritance.

`import aero_open_sdk` lives inside `HardwareAeroDriver.start()`, same rule as
`wuji_driver.py` and the teleop `InputSource` backends: importing this module
must not drag in an SDK the sim path has no use for.

Joint vector: **(16,) radians**, `{side}_{joint_name}` in the order
`canonical_joint_names()` names -- aero_open_sdk's `AeroHandConstants.joint_names`
prefixed per side, the order aria2robot publishes in. The device wants
degrees, so the conversion happens once, at the hardware boundary
(`np.rad2deg`), same place aria2robot's own `AeroDriver.step()` does it.

Startup-only homing
--------------------
`AeroHand.send_homing()` runs the manufacturer's firmware calibration -- up to
175 s, blocking, and the hand "will not respond to any other commands" while
it runs. `HardwareAeroDriver.start()` is the only place that may call it, and
it does so exactly once per opened side. `home()` -- the method
`Hands.open_hands()` reaches through the ordinary `send()` path on a thumbs-up
re-home -- is a **software** zero-ramp only, the same shape as
`wuji_driver.HardwareWujiDriver.home()`, and never touches `send_homing()`.
That split is the whole safety property: nothing on the re-home path can ever
trigger a 175 s firmware homing cycle mid-session.
"""

from __future__ import annotations

import threading
import time

import numpy as np

N_JOINTS = 16
SIDES = ("left", "right")

# Firmware powers on at speed=32766 (max) / torque=1000 (max) and resets to
# these on every power cycle. TetherIA's own ROS2 driver caps torque to
# 700/1000 at startup rather than trusting the firmware default -- full torque
# on every grasp has no headroom before an overcurrent/thermal fault.
DEFAULT_SPEED = 32766
DEFAULT_TORQUE = 700

# aero_open_sdk.aero_hand_constants.AeroHandConstants.joint_names, hardcoded
# here (no aero_open_sdk import at module level) the same way wuji_driver.py
# hardcodes its own names rather than importing wujihandpy for a name list.
_JOINT_SUFFIXES = (
    "thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp", "thumb_ip",
    "index_mcp_flex", "index_pip", "index_dip",
    "middle_mcp_flex", "middle_pip", "middle_dip",
    "ring_mcp_flex", "ring_pip", "ring_dip",
    "pinky_mcp_flex", "pinky_pip", "pinky_dip",
)


def canonical_joint_names(side: str) -> tuple[str, ...]:
    """Actuated-joint order for one side's Aero hand.

    aria2robot side-prefixes `AeroHandConstants.joint_names` the same way to
    match its vendored URDF, so the (16,) vector maps straight across with no
    reordering anywhere.
    """
    return tuple(f"{side}_{name}" for name in _JOINT_SUFFIXES)


class AeroDriver:
    """What the server may call. Every backend answers all of it."""

    name = "base"

    def __init__(self, sides=SIDES):
        self.sides = tuple(sides)
        self._last: dict[str, np.ndarray | None] = {s: None for s in self.sides}

    def start(self) -> None:
        pass

    def send(self, side: str, qpos: np.ndarray) -> None:
        self._last[side] = np.asarray(qpos, dtype=np.float64).reshape(-1).copy()

    def commanded(self, side: str) -> np.ndarray | None:
        """Last vector this driver was handed for `side`."""
        return self._last.get(side)

    def actual(self, side: str) -> np.ndarray | None:
        """Measured joint angles, or None when the backend cannot read back."""
        return None

    def home(self) -> None:
        pass

    def close(self) -> None:
        pass


class NullAeroDriver(AeroDriver):
    """Drives nothing. Sim and dry runs -- the targets still reach the model."""

    name = "none"

    def __init__(self, sides=SIDES):
        super().__init__(sides)
        self.sent = {s: 0 for s in self.sides}

    def send(self, side: str, qpos: np.ndarray) -> None:
        super().send(side, qpos)
        self.sent[side] += 1

    def home(self) -> None:
        for s in self.sides:
            self._last[s] = np.zeros(N_JOINTS)


class HardwareAeroDriver(AeroDriver):
    """aero_open_sdk, one `AeroHand` per side.

    Args:
        sides: which hands to open.
        ports: `{side: serial_port}`. Required when two hands are asked for --
            `AeroHand(port=None)` auto-detects only when exactly one candidate
            is on the bus, which is ambiguous with two plugged in. A
            single-hand session may leave it blank.
        speed, torque: per-actuator caps set on every opened hand
            (0..32766, 0..1000). Firmware resets to max on every power
            cycle, so these are re-applied here rather than trusted.
        ramp_s, ramp_steps: how long the *first* command takes to reach the
            hand, same rationale as WUJI's.
    """

    name = "hardware"

    def __init__(self, sides=SIDES, ports: dict[str, str] | None = None,
                 speed: int = DEFAULT_SPEED, torque: int = DEFAULT_TORQUE,
                 ramp_s: float = 1.5, ramp_steps: int = 30):
        super().__init__(sides)
        self.ports = dict(ports or {})
        self.speed = int(speed)
        self.torque = int(torque)
        self.ramp_s = float(ramp_s)
        self.ramp_steps = max(1, int(ramp_steps))
        self._lock = threading.Lock()
        self._hands: dict[str, object] = {}
        self._ramped: dict[str, bool] = {s: False for s in self.sides}

    def start(self) -> None:
        from aero_open_sdk.aero_hand import AeroHand  # deferred: sim must not need the SDK

        missing = [s for s in self.sides if not self.ports.get(s)]
        if len(self.sides) > 1 and missing:
            raise RuntimeError(
                "two hands need a port each so the sides cannot swap; set "
                f"hand.port.{{{','.join(missing)}}} in config/aria_teleop.yaml"
            )
        opened = []
        for side in self.sides:
            port = self.ports.get(side) or None
            try:
                hand = AeroHand(port=port)
                for i in range(7):
                    hand.set_speed(i, self.speed)
                    hand.set_torque(i, self.torque)
            except Exception as exc:
                print(f"[aero] {side} hand did not open ({exc}); "
                      "continuing without it")
                continue
            self._hands[side] = hand
            opened.append(side)
            print(f"[aero] {side} hand open"
                  + (f" (port {port})" if port else " (auto-detected)"))
        if not opened:
            raise RuntimeError(
                "no Aero hand opened; check the serial connection and "
                f"hand.port for {'+'.join(self.sides)}")

        # Startup-only firmware homing. Never called again -- Hands.open_hands()
        # (the thumbs-up re-home) only ever reaches home() below, which is a
        # software ramp. Two independent serial devices, so home in parallel
        # rather than paying ~175 s twice.
        homed: dict[str, bool] = {}
        errors: dict[str, Exception] = {}

        def _home_one(side: str) -> None:
            print(f"[aero] {side} homing (up to ~175s, do not touch the hand)...")
            try:
                self._hands[side].send_homing()
                homed[side] = True
                print(f"[aero] {side} homing complete")
            except Exception as exc:
                errors[side] = exc

        threads = [threading.Thread(target=_home_one, args=(side,), daemon=True)
                   for side in opened]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for side, exc in errors.items():
            print(f"[aero] {side} homing failed ({exc}); continuing without it")
            del self._hands[side]

        self.sides = tuple(s for s in opened if homed.get(s))
        if not self.sides:
            raise RuntimeError(
                "no Aero hand finished homing; check the serial connection "
                f"for {'+'.join(opened)}")
        self._ramped = {s: False for s in self.sides}

        self.home()

    def send(self, side: str, qpos: np.ndarray) -> None:
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        with self._lock:
            if not self._ramped.get(side, False):
                self._ramp_locked(side, q)
                self._ramped[side] = True
            else:
                self._write_locked(side, q)
            self._last[side] = q.astype(np.float64).copy()

    def home(self) -> None:
        """Ramp every side to the rest pose. Software only -- never re-triggers
        `send_homing()`, which `start()` already called once."""
        with self._lock:
            for side in self.sides:
                self._ramp_locked(side, np.zeros(N_JOINTS, dtype=np.float32),
                                  start=self._last.get(side))
                self._last[side] = np.zeros(N_JOINTS)

    def close(self) -> None:
        with self._lock:
            try:
                for side in self.sides:
                    self._ramp_locked(side, np.zeros(N_JOINTS, dtype=np.float32),
                                      start=self._last.get(side))
            finally:
                for side, hand in self._hands.items():
                    # No torque-disable equivalent to WUJI's write_joint_enabled
                    # (False) -- the hand physically holds its last pose.
                    try:
                        hand.close()
                    except Exception as exc:
                        print(f"[aero] {side} close failed: {exc}")

    # ── internals; all called with _lock held ───────────────────────────────

    def _write_locked(self, side: str, q: np.ndarray) -> None:
        hand = self._hands.get(side)
        if hand is None:
            return
        hand.set_joint_positions(np.rad2deg(q).tolist())

    def _ramp_locked(self, side: str, target: np.ndarray,
                     start: np.ndarray | None = None) -> None:
        """Interpolate from `start` (rest if None) to `target` over ramp_s."""
        q0 = (np.zeros(N_JOINTS, dtype=np.float32) if start is None
              else np.asarray(start, dtype=np.float32).reshape(-1))
        dt = self.ramp_s / self.ramp_steps
        for alpha in np.linspace(0.0, 1.0, self.ramp_steps, endpoint=True):
            self._write_locked(side, (1.0 - alpha) * q0 + alpha * target)
            time.sleep(dt)


def make_driver(backend: str, sides=SIDES, **kwargs) -> AeroDriver:
    """`backend` -> driver. Unknown names fail loudly rather than silently idle."""
    backend = str(backend).lower()
    if backend in ("none", "null", "sim"):
        return NullAeroDriver(sides)
    if backend == "hardware":
        return HardwareAeroDriver(sides, **kwargs)
    raise ValueError(f"unknown hand backend {backend!r}; want none|hardware")
