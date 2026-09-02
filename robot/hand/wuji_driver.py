"""wuji_driver.py — the WUJI hand hardware layer, both hands, one object.

Ported from aria2robot's `src/utils/wuji_driver.py`, which is the proven
single-hand path, with two changes that matter:

  * **Both hands.** aria2robot's `DualWujiDriver` is a `NotImplementedError`
    stub. `wujihandpy.Hand(serial_number=...)` is the addressing mechanism
    (wuji-retargeting's `example/teleop_real.py::WujiHandBackend` uses it), so
    two hands is two `Hand` instances, each with its own realtime controller.
  * **Backend swap.** `NullWujiDriver` drives nothing, so the server runs
    against the simulator, or dry, on a machine with no `wujihandpy` and no
    hand plugged in.

`import wujihandpy` lives inside `HardwareWujiDriver.start()`, the same rule
the teleop `InputSource` backends follow: importing this module must not drag
in an SDK the sim path has no use for.

Joint vector: **(20,) radians**, `{side}_finger{f}_joint{j}` for f in 1..5
(thumb..pinky) and j in 1..4 -- the order `canonical_joint_names()` names and
the order aria2robot publishes. The device wants `(5, 4)`, so the reshape is a
plain `qpos.reshape(5, 4)` and row f is finger f+1. Nothing here reorders.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import numpy as np

N_JOINTS = 20
SIDES = ("left", "right")


def canonical_joint_names(side: str) -> tuple[str, ...]:
    """Actuated-joint order for one side's WUJI hand.

    The MJCF names its hand joints exactly as wuji-description does, and
    aria2robot asserts the same order against the URDF before publishing, so
    the (20,) vector maps straight across with no reordering anywhere.
    """
    return tuple(f"{side}_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5))


class WujiDriver:
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


class NullWujiDriver(WujiDriver):
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


class HardwareWujiDriver(WujiDriver):
    """wujihandpy, one `Hand` + realtime controller per side.

    Args:
        sides: which hands to open.
        serials: `{side: serial_number}`. Required when two hands are asked
            for -- with both plugged in a bare `Hand()` picks whichever the
            USB bus enumerated first, which is a coin flip that ends with the
            left hand making the right hand's grasp. A single-hand session may
            leave it blank, which is what aria2robot does.
        ramp_s, ramp_steps: how long the *first* command takes to reach the
            hand. The first qpos after an operator engages is a full grasp, and
            stepping to it from rest is a real hazard, so each side ramps once
            and then streams.
        lowpass_hz: cutoff of the controller-side filter.
        tracking_csv: log per-step commanded vs measured angles here.
    """

    name = "hardware"

    def __init__(self, sides=SIDES, serials: dict[str, str] | None = None,
                 ramp_s: float = 1.5, ramp_steps: int = 30,
                 lowpass_hz: float = 5.0, tracking_csv: Path | None = None):
        super().__init__(sides)
        self.serials = dict(serials or {})
        self.ramp_s = float(ramp_s)
        self.ramp_steps = max(1, int(ramp_steps))
        self.lowpass_hz = float(lowpass_hz)
        self.tracking_csv = None if tracking_csv is None else Path(tracking_csv)
        self._lock = threading.Lock()
        self._hands: dict[str, object] = {}
        self._controllers: dict[str, object] = {}
        self._ramped: dict[str, bool] = {s: False for s in self.sides}
        self._wuji = None
        self._csv_fh = None
        self._csv_writer = None

    def start(self) -> None:
        import wujihandpy  # deferred: the sim path must not need the SDK

        self._wuji = wujihandpy
        missing = [s for s in self.sides if not self.serials.get(s)]
        if len(self.sides) > 1 and missing:
            raise RuntimeError(
                "two hands need a serial each so the sides cannot swap; set "
                f"hand.serial.{{{','.join(missing)}}} in config/aria_teleop.yaml"
            )
        opened = []
        for side in self.sides:
            serial = self.serials.get(side) or ""
            try:
                hand = (wujihandpy.Hand(serial_number=serial) if serial
                        else wujihandpy.Hand())
                # The server sends from its own loop thread and closes from the
                # signal handler; the SDK's check would reject the second thread
                disable_check = getattr(hand, "disable_thread_safe_check", None)
                if callable(disable_check):
                    disable_check()
                hand.write_joint_enabled(True)
                # enable_upstream costs bandwidth streaming state back; only pay
                # it when something is going to read it
                controller = hand.realtime_controller(
                    enable_upstream=self.tracking_csv is not None,
                    filter=wujihandpy.filter.LowPass(cutoff_freq=self.lowpass_hz),
                )
            except Exception as exc:
                # One hand unplugged must not cost the other one. Opening by
                # serial is unambiguous, so a side that does not answer is
                # absent, not mistaken for its twin -- and the blank-serial
                # refusal above has already run, so this cannot mask a swap.
                print(f"[wuji] {side} hand did not open ({exc}); "
                      "continuing without it")
                continue
            self._hands[side] = hand
            self._controllers[side] = controller
            opened.append(side)
            print(f"[wuji] {side} hand open"
                  + (f" (serial {serial})" if serial else " (first on the bus)"))
        if not opened:
            raise RuntimeError(
                "no WUJI hand opened; check the USB connection, ~/.wuji "
                f"provisioning and hand.serial for {'+'.join(self.sides)}")
        # Everything downstream keys off this, so narrow it to what is really
        # there rather than sending at a device that is not.
        self.sides = tuple(opened)
        time.sleep(0.5)

        if self.tracking_csv is not None:
            self.tracking_csv.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = open(self.tracking_csv, "w", newline="")
            self._csv_writer = csv.writer(self._csv_fh)
            self._csv_writer.writerow(
                ("t_wall", "side", "finger_idx", "joint_idx", "q_cmd", "q_actual"))
            self._csv_fh.flush()

        # Command the rest pose before anything else can. `write_joint_enabled`
        # only energises the joints -- the hand then holds whatever it was
        # physically left in, which after a killed process is whatever grasp it
        # died in. Without this, `send()`'s first-command ramp starts from a
        # `q0 = zeros` that is a hope rather than a fact, and the operator's
        # first engage steps a closed fist open before it ramps. aria2robot's
        # proven path did exactly this, as `WujiDriver.initialize_hand()`.
        # After the CSV, so the startup move is in the tracking log too.
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

    def actual(self, side: str) -> np.ndarray | None:
        ctrl = self._controllers.get(side)
        if ctrl is None or self.tracking_csv is None:
            return None
        try:
            return np.asarray(ctrl.get_joint_actual_position()).reshape(-1)
        except Exception:
            return None

    def home(self) -> None:
        """Ramp every side to the rest pose, from wherever it was commanded to.

        Two callers, both wanting the slow version: `start()`, to make the
        rest pose a fact before an operator can engage, and any caller that
        wants the hands open without the hazard of a step. `Hands.open_hands()`
        deliberately does *not* come here -- it runs mid-session, where a
        blocking ramp would stall the finger loop, so it steps to zero and
        lets the controller-side low-pass do the smoothing.
        """
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
                    try:
                        hand.write_joint_enabled(False)
                    except Exception as exc:
                        print(f"[wuji] {side} disable failed: {exc}")
                if self._csv_fh is not None and not self._csv_fh.closed:
                    self._csv_fh.close()

    # ── internals; all called with _lock held ───────────────────────────────

    def _write_locked(self, side: str, q: np.ndarray) -> None:
        ctrl = self._controllers.get(side)
        if ctrl is None:
            return
        q_2d = q.reshape(5, 4)
        ctrl.set_joint_target_position(q_2d)
        if self._csv_writer is None:
            return
        try:
            q_actual = np.asarray(ctrl.get_joint_actual_position()).reshape(5, 4)
        except Exception:
            return
        t_wall = time.time()
        self._csv_writer.writerows(
            (f"{t_wall:.6f}", side, f, j,
             f"{float(q_2d[f, j]):.6f}", f"{float(q_actual[f, j]):.6f}")
            for f in range(5) for j in range(4)
        )
        self._csv_fh.flush()

    def _ramp_locked(self, side: str, target: np.ndarray,
                     start: np.ndarray | None = None) -> None:
        """Interpolate from `start` (rest if None) to `target` over ramp_s."""
        q0 = (np.zeros(N_JOINTS, dtype=np.float32) if start is None
              else np.asarray(start, dtype=np.float32).reshape(-1))
        dt = self.ramp_s / self.ramp_steps
        for alpha in np.linspace(0.0, 1.0, self.ramp_steps, endpoint=True):
            self._write_locked(side, (1.0 - alpha) * q0 + alpha * target)
            time.sleep(dt)


def make_driver(backend: str, sides=SIDES, **kwargs) -> WujiDriver:
    """`backend` -> driver. Unknown names fail loudly rather than silently idle."""
    backend = str(backend).lower()
    if backend in ("none", "null", "sim"):
        return NullWujiDriver(sides)
    if backend == "hardware":
        return HardwareWujiDriver(sides, **kwargs)
    raise ValueError(f"unknown hand backend {backend!r}; want none|hardware")
