"""hands.py — both WUJI hands, as a component of a robot node.

`Hands` lives *inside* `robot/yor.py` and `robot/yor_mujoco.py`, the way
`ArmNode` and `Base` do, so one process owns the whole robot: one shutdown, one
`get_state()`, one thing to start.

Why the fingers never touch the node's RPC socket
-------------------------------------------------
commlink's `RPCServer` is a single ZMQ `REP` socket, and a REP socket is
strictly one request in flight at a time -- `threaded=True` only moves that
loop onto a thread, it does not serve two callers at once. So a finger target
sent through the node's own port really would queue behind the 30 Hz arm
targets, on hardware as well as in sim.

It doesn't have to. The fingers already arrive on the *same publisher* the arm
client reads, so this subscribes to it directly, on its own thread:

    aria2robot stream_pub --PUB "wuji"--+--> teleop client --RPC :5557--> arms
                                        +--> Hands (in the node) ------> fingers

Nothing shared but the publisher, and no RPC hop at all on the finger path.
For anything that is not a pair of glasses there is still an RPC surface, but
it gets a socket of its own (`hand.rpc_port`, 5558) rather than the node's, so
that path does not queue behind arm targets either.

Hold-last, everywhere
---------------------
A side's target changes only when a *usable* command arrives. Shaka-paused,
publisher silent, tracking lost, nothing sent yet -- all of them hold the last
pose rather than release it. aria2robot freezes `qpos` while paused and sends
`None` before the first engage, so the hands are never touched pre-engage, and
a link that goes quiet mid-grasp leaves the grasp alone instead of springing
the hand open on its own. There is deliberately no staleness gate.

Settings live in `config/aria_teleop.yaml` under `hand:`.

Joint vector: (20,) radians, `{side}_finger{f}_joint{j}` for f in 1..5
(thumb..pinky), j in 1..4 -- see wuji_driver.py.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from robot.hand.wuji_driver import N_JOINTS, make_driver
from robot.teleop.aria.config import AriaConfig

# A resend of the identical vector is wasted USB traffic, but the threshold has
# to stay well under the smallest motion an operator can see -- 1e-4 rad is
# ~0.006 degrees, about a thousandth of a finger's travel.
_CHANGE_EPS = 1e-4


class Hands:
    """Finger targets for both hands, an input thread, and a driver.

    The node holds one of these and calls `start()` / `stop()` with its own
    control loop. `targets()` is how the simulator reads what to write into
    `MjData`; on hardware the internal loop hands the same vectors to
    `wujihandpy`.
    """

    def __init__(self, cfg: AriaConfig, aria: bool = True, rpc: bool = True,
                 tracking_csv: Path | None = None):
        self.cfg = cfg
        hand_cfg = cfg.hand
        mapped = cfg.mapping["hand"]
        self.sides = ("left", "right") if mapped == "both" else (mapped,)
        self.backend = str(hand_cfg["backend"])
        self.rate_hz = int(hand_cfg["rate_hz"])
        self.rpc_port = int(hand_cfg["rpc_port"]) if rpc else 0
        self._want_aria = bool(aria)

        self._lock = threading.Lock()
        self._target: dict[str, np.ndarray | None] = {s: None for s in self.sides}
        self._engaged: dict[str, bool] = {s: False for s in self.sides}
        self._origin: dict[str, str] = {s: "-" for s in self.sides}
        self._sent: dict[str, np.ndarray | None] = {s: None for s in self.sides}
        self._sends: dict[str, int] = {s: 0 for s in self.sides}

        self.driver = make_driver(
            self.backend, self.sides,
            **({} if self.backend != "hardware" else dict(
                serials=hand_cfg["serial"],
                ramp_s=float(hand_cfg["ramp_s"]),
                lowpass_hz=float(hand_cfg["lowpass_hz"]),
                tracking_csv=tracking_csv,
            )),
        )
        self._stream = None
        self._rpc = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = False

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the driver and the inputs, then run the loop on a daemon thread."""
        if self._started:
            return
        self._started = True
        self.driver.start()
        if self._want_aria:
            from robot.teleop.aria.stream import AriaHandStream

            self._stream = AriaHandStream(
                self.cfg.publisher["host"], self.cfg.publisher["port"],
                sides=self.sides, stale_s=self.cfg.publisher["stale_s"] or None)
            self._stream.start()
        if self.rpc_port:
            from commlink import RPCServer

            # Its own socket, and its own thread: the point of the whole
            # arrangement is that nothing on the finger path waits on an arm
            # call. `_HandRPC` is what gets exposed, not `self` -- commlink
            # publishes every public method it is handed, and `stop()` is not
            # a surface a remote client should have.
            self._rpc = RPCServer(_HandRPC(self), self.rpc_port, threaded=True)
            self._rpc.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="wuji-hands",
                                        daemon=True)
        self._thread.start()
        print(f"[wuji] hands={'+'.join(self.sides)} backend={self.backend} "
              f"aria={'on' if self._want_aria else 'off'} "
              f"rpc={self.rpc_port or 'off'} rate={self.rate_hz} Hz")

    def stop(self) -> None:
        """Release the hands, then the sockets. Safe to call twice, or early."""
        if not self._started:
            # Never started: closing the driver would ramp a hand it never opened.
            return
        self._started = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self.driver.close()
        except Exception as exc:
            print(f"[wuji] driver close: {exc}")
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
        if self._rpc is not None:
            try:
                self._rpc.stop()
            except Exception:
                pass
            self._rpc = None

    # ── what the node reads ─────────────────────────────────────────────────

    def targets(self) -> dict[str, np.ndarray | None]:
        """Current commanded qpos per side; None where nothing has been sent."""
        with self._lock:
            return {s: (None if q is None else q.copy())
                    for s, q in self._target.items()}

    # ── command surface (also the RPC surface, via _HandRPC) ────────────────

    def set_hand_target(self, side: str, qpos) -> bool:
        """Command one hand: (20,) radians, finger1..5 x joint1..4."""
        return self._store(str(side), qpos, origin="rpc")

    def set_bimanual_hand_target(self, L_hand_target=None,
                                 R_hand_target=None) -> bool:
        """Command both hands in one call. A None side is left unchanged."""
        ok = True
        if L_hand_target is not None:
            ok &= self._store("left", L_hand_target, origin="rpc")
        if R_hand_target is not None:
            ok &= self._store("right", R_hand_target, origin="rpc")
        return ok

    def get_hand_state(self) -> dict:
        """Snapshot for clients (plain types only)."""
        with self._lock:
            return {
                "sides": list(self.sides),
                "backend": self.backend,
                "qpos": {s: (None if self._target[s] is None
                             else self._target[s].tolist())
                         for s in self.sides},
                "engaged": dict(self._engaged),
                "origin": dict(self._origin),
                "sends": dict(self._sends),
            }

    def home_hands(self, sides=None) -> bool:
        """Send `sides` (default: all) back to zero -- the model's home pose."""
        return self.open_hands(sides)

    def open_hands(self, sides=None) -> bool:
        """Zero every joint on `sides`, default all of them.

        A step to zero, smoothed by the same controller-side low-pass every
        other finger command goes through -- deliberately not `close()`'s slow
        ramp, which exists for the one command that arrives from rest.

        Hold-last then keeps the hands open: a paused operator sends nothing
        usable, so nothing overwrites this until they engage again.
        """
        want = (self.sides if sides is None
                else tuple(s for s in sides if s in self.sides))
        with self._lock:
            for s in want:
                self._target[s] = np.zeros(N_JOINTS)
                self._origin[s] = "home"
        return bool(want)

    # ── internals ───────────────────────────────────────────────────────────

    def _store(self, side: str, qpos, origin: str) -> bool:
        if side not in self.sides:
            print(f"[wuji] ignoring target for {side!r}; serving {self.sides}")
            return False
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.size != N_JOINTS:
            print(f"[wuji] {side}: want {N_JOINTS} joints, got {q.size}")
            return False
        with self._lock:
            self._target[side] = q
            self._origin[side] = origin
        return True

    def _pull_aria(self) -> None:
        """Adopt each side's published qpos, when it is one we may use."""
        if self._stream is None:
            return
        snap = self._stream.snapshot()
        for side in self.sides:
            s = snap[side]
            engaged = not s.paused
            with self._lock:
                self._engaged[side] = engaged
            # Paused freezes the fingers; None is the pre-engage state, where
            # nothing has been retargeted yet. Both hold the last pose.
            if engaged and s.qpos is not None:
                self._store(side, s.qpos[:N_JOINTS], origin="aria")

    def _push(self) -> None:
        """Hand changed targets to the driver. A no-op on the null backend."""
        for side, q in self.targets().items():
            if q is None:
                continue
            prev = self._sent[side]
            if prev is not None and np.max(np.abs(q - prev)) < _CHANGE_EPS:
                continue
            try:
                self.driver.send(side, q)
            except Exception as exc:
                print(f"[wuji] {side} send failed: {exc}")
                continue
            self._sent[side] = q
            self._sends[side] += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._pull_aria()
                self._push()
            except Exception as exc:  # a finger fault must not kill the node
                print(f"[wuji] loop: {exc}")
            self._stop.wait(1.0 / max(self.rate_hz, 1))


class _HandRPC:
    """The five methods a remote client may call, and nothing else.

    commlink exposes every public attribute of whatever object it is given, so
    handing it the `Hands` instance would also hand out `stop()` and `driver`.
    """

    def __init__(self, hands: Hands):
        self._hands = hands

    def set_hand_target(self, side: str, qpos) -> bool:
        return self._hands.set_hand_target(side, qpos)

    def set_bimanual_hand_target(self, L_hand_target=None,
                                 R_hand_target=None) -> bool:
        return self._hands.set_bimanual_hand_target(L_hand_target, R_hand_target)

    def get_hand_state(self) -> dict:
        return self._hands.get_hand_state()

    def home_hands(self, sides=None) -> bool:
        return self._hands.home_hands(sides)

    def open_hands(self, sides=None) -> bool:
        return self._hands.open_hands(sides)


def hands_from_args(args, force_backend: str | None = None) -> Hands | None:
    """Build a node's `Hands` from the shared hand flags. None when switched off.

    `force_backend` is how the simulator pins itself to "none": it renders
    fingers, it never drives them, so `hand.backend: hardware` in the YAML must
    not reach out to a USB device from a sim run.
    """
    if getattr(args, "no_hands", False):
        return None
    cfg = AriaConfig.load(getattr(args, "aria_config", None))
    if getattr(args, "pub_host", None):
        cfg.publisher["host"] = args.pub_host
    backend = force_backend or getattr(args, "hand_backend", None)
    if backend:
        cfg.hand["backend"] = str(backend)
    tracking = getattr(args, "tracking_csv", None)
    if tracking and cfg.hand["backend"] != "hardware":
        raise SystemExit("--tracking-csv needs --hand-backend hardware")
    return Hands(cfg, tracking_csv=Path(tracking) if tracking else None)


def add_hand_args(parser, backend_flag: bool = True) -> None:
    """The hand flags, identical on both nodes."""
    parser.add_argument("--no-hands", action="store_true",
                        help="do not drive the WUJI fingers at all")
    parser.add_argument("--aria-config", default=None,
                        help="settings file (default: config/aria_teleop.yaml)")
    parser.add_argument("--pub-host", default=None,
                        help="override hand publisher host -- where stream_pub runs")
    if backend_flag:
        parser.add_argument("--hand-backend", choices=["none", "hardware"],
                            default=None, help="override hand.backend for one run")
        parser.add_argument("--tracking-csv", default=None,
                            help="log commanded vs measured finger angles")
