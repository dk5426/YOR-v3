"""hands.py — both hands of the configured type, as a component of a robot node.

`Hands` lives *inside* `robot/yor.py` and `robot/yor_mujoco.py`, the way
`ArmNode` and `Base` do, so one process owns the whole robot: one shutdown, one
`get_state()`, one thing to start.

Which hand: `hand.type` (`none` | `wuji` | `aero`) in config/aria_teleop.yaml,
or `--hand` for one run (default: `none` -- arms only, nothing constructed
here at all). Picked explicitly, never auto-detected from the publisher --
but checked against the publisher's own `meta["hand"]` once it arrives
(`_check_hand_type()`), and a mismatch is fatal (`os._exit(1)`): it means
every joint vector is misread, which on real hardware is the wrong actuators
moving to the wrong angles. Silent -- not fatal -- only when the key is
absent entirely, for an older publisher that never sent it.
`robot/hand/wuji_driver.py` and
`robot/hand/aero_driver.py` are duck-typed the same way aria2robot's own
`WujiDriver`/`AeroDriver` are -- same method names, no shared base class --
so this class never needs to know which one it is holding beyond picking the
module up front.

Why the fingers never touch the node's RPC socket
-------------------------------------------------
commlink's `RPCServer` is a single ZMQ `REP` socket, and a REP socket is
strictly one request in flight at a time -- `threaded=True` only moves that
loop onto a thread, it does not serve two callers at once. So a finger target
sent through the node's own port really would queue behind the 30 Hz arm
targets, on hardware as well as in sim.

It doesn't have to. The fingers already arrive on the *same publisher* the arm
client reads, so this subscribes to it directly, on its own thread:

    aria2robot stream_pub --PUB "qpos"--+--> teleop client --RPC :5557--> arms
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

Joint vector: radians, sized and ordered per `hand.type` -- see
`wuji_driver.canonical_joint_names` (20, `{side}_finger{f}_joint{j}`) and
`aero_driver.canonical_joint_names` (16, `{side}_{joint_name}`).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from robot.hand import aero_driver, wuji_driver
from robot.teleop.aria.config import AriaConfig

_HAND_MODULES = {"wuji": wuji_driver, "aero": aero_driver}

_REPO = Path(__file__).resolve().parents[2]

# Which MJCF scene mounts which hand -- see description/robot_wholebody*.xml
# and CLAUDE.md's teleop-input-architecture section. "none" is the bare,
# hand-agnostic scene: arms/base/lift only, the default.
_SCENE_BY_TYPE = {
    "none": "description/scene_wholebody.xml",
    "wuji": "description/scene_wholebody_wuji.xml",
    "aero": "description/scene_wholebody_aero.xml",
}

# The body a hand's mount frame lives on -- see robot/arm/wholebody_ik.py's
# own copy of this table (used for collision/ground-avoidance geometry) and
# robot/teleop/aria/source.py / sim_viz.py (used for the flange->wrist
# lever-arm offset). One table, two independent readers -- both do their own
# try/except KeyError on a scene/hand.type mismatch rather than share a
# runtime dependency neither wants (wholebody_ik.py's mink/qpsolvers stack,
# or this module's driver machinery).
_MOUNT_ORIENT_BODY = {
    "wuji": "{side}_wuji_hand_orient",
    "aero": "{side}_aero_hand_orient",
}


def scene_for_hand_type(hand_type: str | None) -> Path:
    """MJCF scene path for a hand.type ("none"/absent -> the bare scene)."""
    key = str(hand_type or "none").lower()
    if key not in _SCENE_BY_TYPE:
        raise ValueError(f"unknown hand.type {key!r}; want none|wuji|aero")
    return _REPO / _SCENE_BY_TYPE[key]


def hand_mount_body(hand_type: str | None, side: str) -> str | None:
    """MJCF body name of the flange->hand mount frame, or None (no hand)."""
    tmpl = _MOUNT_ORIENT_BODY.get(str(hand_type or "none").lower())
    return tmpl.format(side=side) if tmpl else None

# A resend of the identical vector is wasted USB traffic, but the threshold has
# to stay well under the smallest motion an operator can see -- 1e-4 rad is
# ~0.006 degrees, about a thousandth of a finger's travel.
_CHANGE_EPS = 1e-4


class Hands:
    """Finger targets for both hands, an input thread, and a driver.

    The node holds one of these and calls `start()` / `stop()` with its own
    control loop. `targets()` is how the simulator reads what to write into
    `MjData`; on hardware the internal loop hands the same vectors to the
    configured driver (`wujihandpy` or `aero_open_sdk`).
    """

    def __init__(self, cfg: AriaConfig, aria: bool = True, rpc: bool = True,
                 tracking_csv: Path | None = None):
        self.cfg = cfg
        hand_cfg = cfg.hand
        self.sides = cfg.hand_sides()
        self.hand_type = str(hand_cfg["type"]).lower()
        if self.hand_type not in _HAND_MODULES:
            raise ValueError(
                f"unknown hand.type {self.hand_type!r}; want "
                f"{'|'.join(_HAND_MODULES)}")
        self._driver_mod = _HAND_MODULES[self.hand_type]
        self.n_joints = self._driver_mod.N_JOINTS
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
        self._hand_type_warned = False

        if self.backend != "hardware":
            driver_kwargs = {}
        elif self.hand_type == "wuji":
            driver_kwargs = dict(
                serials=hand_cfg["serial"],
                ramp_s=float(hand_cfg["ramp_s"]),
                lowpass_hz=float(hand_cfg["lowpass_hz"]),
                tracking_csv=tracking_csv,
            )
        else:  # aero
            driver_kwargs = dict(
                ports=hand_cfg["port"],
                ramp_s=float(hand_cfg["ramp_s"]),
                speed=int(hand_cfg["aero_speed"]),
                torque=int(hand_cfg["aero_torque"]),
            )
        self.driver = self._driver_mod.make_driver(
            self.backend, self.sides, **driver_kwargs)
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
        # The hardware driver drops a side whose hand did not answer -- one
        # unplugged hand must not cost the other. Follow it, so nothing
        # subscribes to, reports or sends at a device that is not there.
        self._narrow(tuple(self.driver.sides))
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
        self._thread = threading.Thread(target=self._loop, name="hand-loop",
                                        daemon=True)
        self._thread.start()
        print(f"[{self.hand_type}] hands={'+'.join(self.sides)} backend={self.backend} "
              f"aria={'on' if self._want_aria else 'off'} "
              f"rpc={self.rpc_port or 'off'} rate={self.rate_hz} Hz")

    def _narrow(self, sides: tuple[str, ...]) -> None:
        """Serve only `sides` from here on. Called once, before the loop runs."""
        if tuple(sides) == self.sides:
            return
        print(f"[{self.hand_type}] serving {'+'.join(sides) or 'no hands'}, "
              f"not {'+'.join(self.sides)}")
        self.sides = tuple(sides)
        with self._lock:
            for d in (self._target, self._engaged, self._origin, self._sent,
                      self._sends):
                for side in [s for s in d if s not in self.sides]:
                    del d[side]

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
            print(f"[{self.hand_type}] driver close: {exc}")
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

    def joint_names(self, side: str) -> tuple[str, ...]:
        """Actuated-joint order for one side, per the configured hand.type."""
        return self._driver_mod.canonical_joint_names(side)

    def set_hand_target(self, side: str, qpos) -> bool:
        """Command one hand: `n_joints` radians, in `joint_names(side)` order."""
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
                self._target[s] = np.zeros(self.n_joints)
                self._origin[s] = "home"
        return bool(want)

    # ── internals ───────────────────────────────────────────────────────────

    def _store(self, side: str, qpos, origin: str) -> bool:
        if side not in self.sides:
            print(f"[{self.hand_type}] ignoring target for {side!r}; serving {self.sides}")
            return False
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.size != self.n_joints:
            print(f"[{self.hand_type}] {side}: want {self.n_joints} joints, got {q.size}")
            return False
        with self._lock:
            self._target[side] = q
            self._origin[side] = origin
        return True

    def _check_hand_type(self) -> None:
        """Refuse to run if the publisher declares a different hand than configured.

        Explicit config picks the hand type, never auto-detection -- but a
        mismatch here means every joint vector is misread (an Aero-shaped
        vector driven as WUJI's, or the reverse), so this is fatal, not a
        warning: on real hardware that is the wrong actuators moving to the
        wrong angles. Silent -- not fatal -- when the key is absent entirely,
        for an older publisher that never sent it.

        `os._exit()`, not `raise`: this runs on `_loop()`'s daemon thread,
        which wraps its body in `except Exception` so a normal exception
        would only be printed and swallowed, leaving the node running with
        the wrong hand.
        """
        if self._hand_type_warned or self._stream is None:
            return
        meta_fn = getattr(self._stream, "meta", None)
        if meta_fn is None:
            return
        meta = meta_fn()
        if meta is None:
            return
        published = meta.get("hand")
        if published is not None and published != self.hand_type:
            import os

            msg = (f"[{self.hand_type}] FATAL: publisher declares "
                   f"hand={published!r} but hand.type={self.hand_type!r} is "
                   "configured here -- joint vectors would not line up. "
                   "Fix --hand / hand.type or restart the publisher with "
                   "the matching --hand.")
            print(f"\033[1;31m{msg}\033[0m", flush=True)
            os._exit(1)
        self._hand_type_warned = True

    def _pull_aria(self) -> None:
        """Adopt each side's published qpos, when it is one we may use."""
        if self._stream is None:
            return
        self._check_hand_type()
        snap = self._stream.snapshot()
        for side in self.sides:
            s = snap[side]
            engaged = not s.paused
            with self._lock:
                self._engaged[side] = engaged
            # Paused freezes the fingers; None is the pre-engage state, where
            # nothing has been retargeted yet. Both hold the last pose.
            if engaged and s.qpos is not None:
                self._store(side, s.qpos[:self.n_joints], origin="aria")

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
                print(f"[{self.hand_type}] {side} send failed: {exc}")
                continue
            self._sent[side] = q
            self._sends[side] += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._pull_aria()
                self._push()
            except Exception as exc:  # a finger fault must not kill the node
                print(f"[{self.hand_type}] loop: {exc}")
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


def resolved_aria_config(args) -> AriaConfig:
    """AriaConfig with the shared --hand/--hands/--pub-host CLI overrides applied.

    Shared by `hands_from_args()` and by yor.py/yor_mujoco.py's own scene
    selection (`resolved_aria_config(args).scene_path()`), so `--hand` picks
    both the driver and the MJCF from one place -- and picks it correctly
    even under `--no-hands`, which only short-circuits `hands_from_args`,
    not scene selection (a bare-armed run still needs the bare scene).
    """
    cfg = AriaConfig.load(getattr(args, "aria_config", None))
    if getattr(args, "pub_host", None):
        cfg.publisher["host"] = args.pub_host
    if getattr(args, "hands", None):
        cfg.hand["sides"] = args.hands
    if getattr(args, "hand", None):
        cfg.hand["type"] = args.hand
    return cfg


def hands_from_args(args, force_backend: str | None = None) -> Hands | None:
    """Build a node's `Hands` from the shared hand flags. None when switched off.

    `force_backend` is how the simulator pins itself to "none": it renders
    fingers, it never drives them, so `hand.backend: hardware` in the YAML must
    not reach out to a USB device from a sim run.
    """
    if getattr(args, "no_hands", False):
        return None
    cfg = resolved_aria_config(args)
    hand_type = str(cfg.hand["type"] or "none").lower()
    if hand_type == "none":
        print("[hand] no hand configured; arms only")
        return None
    if not cfg.hand_sides():
        print(f"[{hand_type}] no hands this session; arms only")
        return None
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
                        help="do not drive the fingers at all")
    parser.add_argument("--aria-config", default=None,
                        help="settings file (default: config/aria_teleop.yaml)")
    parser.add_argument("--pub-host", default=None,
                        help="override hand publisher host -- where stream_pub runs")
    parser.add_argument("--hands", choices=["both", "left", "right", "none"],
                        default=None,
                        help="which hands to drive (default: hand.sides); "
                             "the arms are teleoped either way")
    parser.add_argument("--hand", choices=["wuji", "aero"], default=None,
                        help="which hand to drive/render this run (default: "
                             "hand.type in config, itself 'none' -- arms "
                             "only)")
    if backend_flag:
        # This node (robot/yor.py) is hardware-only, so a --hand actually
        # given implies driving it for real -- defaulting this to "hardware"
        # is what makes "--hand aero" alone do the obvious thing, instead of
        # silently no-op'ing on the null driver until "--hand-backend
        # hardware" is also spelled out. Never reached at all when hand.type
        # is "none" (hands_from_args returns before backend is consulted).
        # "--hand-backend none" stays available as an explicit escape hatch
        # (e.g. exercising the software path with no hand plugged in).
        parser.add_argument("--hand-backend", choices=["none", "hardware"],
                            default="hardware",
                            help="override hand.backend for one run "
                                 "(default: hardware, whenever a hand is "
                                 "actually configured)")
        parser.add_argument("--tracking-csv", default=None,
                            help="log commanded vs measured finger angles")
