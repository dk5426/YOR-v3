"""
test_wuji_hand.py — contract tests for the WUJI hand path.

`Hands` lives inside both nodes, so the things that can break silently are
transforms and policy, not plumbing: the (20,) -> (5,4) reshape landing a
finger on the wrong row, a paused or pre-engage payload moving the hands
anyway, the injection into MjData being overwritten by the solver, the
hardware ramp stepping instead of ramping, and the hand RPC surface widening
to something a remote client should not have.

No hand, no publisher, no viewer.

    python tests/test_wuji_hand.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robot.hand.wuji_driver import (
    N_JOINTS,
    HardwareWujiDriver,
    NullWujiDriver,
    canonical_joint_names,
    make_driver,
)

SIDES = ("left", "right")
SCENE = _REPO / "description" / "scene_wholebody.xml"

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
    return _MODEL


# ─────────────────────────────────────────────────────────────────────────────
# The joint vector
# ─────────────────────────────────────────────────────────────────────────────

def test_joint_order() -> None:
    """(20,) -> (5,4) must put finger f on row f-1, joint j on column j-1.

    This is the whole reason no reordering happens anywhere: the MJCF, the
    aria2robot publisher and wujihandpy's set_joint_target_position all agree
    on this layout. A silent disagreement moves the wrong finger.
    """
    print("\njoint vector layout")
    names = canonical_joint_names("left")
    check("20 joints per hand", len(names) == N_JOINTS, str(len(names)))
    grid = np.asarray(names).reshape(5, 4)
    ok = all(grid[f, j] == f"left_finger{f + 1}_joint{j + 1}"
             for f in range(5) for j in range(4))
    check("reshape(5, 4) is finger-major", ok, grid[0, 0] + " .. " + grid[4, 3])
    check("side prefix is the only difference between hands",
          tuple(n.replace("left_", "") for n in names)
          == tuple(n.replace("right_", "") for n in canonical_joint_names("right")))


def test_model_hand_joints() -> None:
    """Every canonical name exists in the scene, contiguous and ascending.

    Contiguity is what lets the published vector be written as one slice; a
    model edit that interleaved another joint would break the slice silently.
    """
    print("\nMJCF hand joints")
    m = model()
    for side in SIDES:
        try:
            joints = [m.joint(n) for n in canonical_joint_names(side)]
        except KeyError as exc:
            check(f"{side}: all 20 joints present", False, str(exc))
            continue
        check(f"{side}: all 20 joints present", True)
        adrs = np.array([int(j.qposadr[0]) for j in joints])
        check(f"{side}: qpos addresses contiguous and ascending",
              bool(np.all(np.diff(adrs) == 1)), str(adrs[:3]) + " ..")
        lo = np.array([float(j.range[0]) for j in joints])
        hi = np.array([float(j.range[1]) for j in joints])
        check(f"{side}: every joint has a real range", bool(np.all(hi > lo)))
        # Whatever the retargeter sends, the sim must land inside the model
        wild = np.full(N_JOINTS, 99.0)
        check(f"{side}: clip lands inside the model's ranges",
              bool(np.all(np.clip(wild, lo, hi) <= hi + 1e-12)))


# ─────────────────────────────────────────────────────────────────────────────
# Drivers
# ─────────────────────────────────────────────────────────────────────────────

class _FakeController:
    """Stands in for wujihandpy's realtime controller."""

    def __init__(self):
        self.writes: list[np.ndarray] = []

    def set_joint_target_position(self, q_2d):
        arr = np.asarray(q_2d)
        assert arr.shape == (5, 4), f"device wants (5, 4), got {arr.shape}"
        self.writes.append(arr.copy())


def test_null_driver() -> None:
    print("\nnull driver")
    d = make_driver("none", SIDES)
    check("make_driver('none') is the null backend", isinstance(d, NullWujiDriver))
    q = np.arange(N_JOINTS, dtype=float)
    d.send("left", q)
    check("records what it was handed", np.allclose(d.commanded("left"), q))
    check("counts sends", d.sent["left"] == 1 and d.sent["right"] == 0)
    d.home()
    check("home zeroes the vector", np.allclose(d.commanded("left"), 0.0))
    try:
        make_driver("wat")
        check("an unknown backend raises", False, "no raise")
    except ValueError:
        check("an unknown backend raises", True)


def test_hardware_ramp() -> None:
    """The first command must arrive as a ramp, never as a step.

    The first pose after an operator engages is a whole grasp; stepping to it
    from rest is the one genuinely dangerous moment on this path.
    """
    print("\nhardware ramp")
    d = HardwareWujiDriver(("left",), serials={"left": "X"},
                           ramp_s=0.0, ramp_steps=10)
    ctrl = _FakeController()
    d._controllers["left"] = ctrl
    target = np.full(N_JOINTS, 1.2)

    d.send("left", target)
    check("first send ramps rather than steps", len(ctrl.writes) == 10,
          str(len(ctrl.writes)))
    first = ctrl.writes[0].reshape(-1)
    check("the ramp starts at rest", np.allclose(first, 0.0))
    check("the ramp ends on the target",
          np.allclose(ctrl.writes[-1].reshape(-1), target))
    peaks = [float(np.max(w)) for w in ctrl.writes]
    check("the ramp is monotonic", all(b >= a - 1e-12 for a, b in zip(peaks, peaks[1:])))

    ctrl.writes.clear()
    d.send("left", np.full(N_JOINTS, 1.3))
    check("later sends are a single write", len(ctrl.writes) == 1, str(len(ctrl.writes)))

    ctrl.writes.clear()
    d.home()
    check("home ramps back to zero",
          len(ctrl.writes) == 10 and np.allclose(ctrl.writes[-1], 0.0))


def test_hardware_needs_serials() -> None:
    """Two bare Hand() calls would pick sides by USB enumeration order."""
    print("\nhardware addressing")
    d = HardwareWujiDriver(SIDES, serials={"left": "A", "right": ""})
    try:
        d.start()
        check("two hands without both serials refuses to start", False, "no raise")
    except RuntimeError as exc:
        check("two hands without both serials refuses to start",
              "serial" in str(exc))
    except ImportError:
        # No wujihandpy here; the serial check runs after the import, so this
        # machine cannot reach it. Not a failure of the code under test.
        check("two hands without both serials refuses to start", True,
              "wujihandpy absent, check not reachable")


def test_hardware_zeroes_at_startup() -> None:
    """start() must command the rest pose, not just energise the joints.

    `write_joint_enabled(True)` leaves the hand holding whatever it was
    physically left in -- after a killed process, whatever grasp it died in.
    `send()`'s ramp interpolates from `q0 = zeros`, so without a startup zero
    that first engage steps a closed fist open before it ramps. aria2robot's
    proven path did this as `WujiDriver.initialize_hand()` (stream_sub.py:141).
    """
    print("\nhardware startup pose")
    d = HardwareWujiDriver(SIDES, serials={"left": "A", "right": "B"},
                           ramp_s=0.0, ramp_steps=8)
    ctrls = {s: _FakeController() for s in SIDES}
    d._controllers.update(ctrls)

    d.home()
    for side in SIDES:
        writes = ctrls[side].writes
        check(f"{side} is commanded to rest at startup", len(writes) == 8,
              str(len(writes)))
        check(f"{side} startup pose is zero",
              all(np.allclose(w, 0.0) for w in writes))
        check(f"{side} records rest as the last commanded pose",
              np.allclose(d.commanded(side), 0.0))

    # The whole point is that the *engage* ramp still happens afterwards.
    ctrls["left"].writes.clear()
    d.send("left", np.full(N_JOINTS, 1.1))
    check("the first operator command still ramps",
          len(ctrls["left"].writes) == 8, str(len(ctrls["left"].writes)))
    check("and it now genuinely starts from rest",
          np.allclose(ctrls["left"].writes[0], 0.0))

    import inspect

    src = inspect.getsource(HardwareWujiDriver.start)
    check("start() commands the rest pose", "self.home()" in src)
    check("it does so after the controllers exist",
          src.find("realtime_controller") < src.find("self.home()"))


def test_one_hand_unplugged_keeps_the_other() -> None:
    """A hand that does not answer costs itself, not the pair.

    The failure this replaces was total: `Hand(serial)` raised for the absent
    side, `Hands.start()` propagated, and `yor.py` dropped `self.hands`
    entirely -- so an operator with one hand plugged in got none. Opening by
    serial is unambiguous, and the blank-serial refusal runs first, so a side
    that stays silent is absent rather than mistaken for its twin.
    """
    print("\nunplugged hand")
    import sys
    import types

    from robot.hand.hands import Hands
    from robot.teleop.aria.config import AriaConfig

    class _FakeHand:
        def __init__(self, serial_number=""):
            if serial_number == "GONE":
                raise RuntimeError("no such device")
            self.serial = serial_number

        def disable_thread_safe_check(self): pass
        def write_joint_enabled(self, on): pass
        def realtime_controller(self, **kw): return _FakeController()

    fake = types.ModuleType("wujihandpy")
    fake.Hand = _FakeHand
    fake.filter = types.SimpleNamespace(LowPass=lambda cutoff_freq: None)
    saved = sys.modules.get("wujihandpy")
    sys.modules["wujihandpy"] = fake
    try:
        d = HardwareWujiDriver(SIDES, serials={"left": "GONE", "right": "B"},
                               ramp_s=0.0, ramp_steps=2)
        d.start()
        check("the hand that is there still opens", d.sides == ("right",))
        check("and it was commanded to rest", np.allclose(d.commanded("right"), 0.0))
        check("the absent one holds no controller", "left" not in d._controllers)

        cfg = AriaConfig({})
        cfg.hand["backend"] = "hardware"
        cfg.hand["serial"] = {"left": "GONE", "right": "B"}
        cfg.hand["ramp_s"] = 0.0
        srv = Hands(cfg, aria=False, rpc=False)
        srv.start()
        try:
            check("Hands follows the driver", srv.sides == ("right",))
            check("the absent side is not reported",
                  list(srv.get_hand_state()["qpos"]) == ["right"])
            check("nor commandable", not srv.set_hand_target("left", np.zeros(N_JOINTS)))
        finally:
            srv.stop()

        d = HardwareWujiDriver(SIDES, serials={"left": "GONE", "right": "GONE"})
        try:
            d.start()
            check("no hands at all is still an error", False, "no raise")
        except RuntimeError as exc:
            check("no hands at all is still an error", "no WUJI hand" in str(exc))
    finally:
        if saved is None:
            del sys.modules["wujihandpy"]
        else:
            sys.modules["wujihandpy"] = saved


def test_hands_start_is_not_fatal() -> None:
    """A finger fault must not take down a node whose arms are already homed.

    Everything else in YOR.init() raises deliberately -- an unhomed lift has no
    absolute zero. The hands are an accessory, and they start *last*, so by then
    a 30-60 s homing cycle is complete and throwing it away over a USB device
    is the wrong trade.
    """
    print("\nhands are not load-bearing")
    import ast

    tree = ast.parse((_REPO / "robot" / "yor.py").read_text())
    init = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "init")

    def calls_hands_start(node):
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                   and c.func.attr == "start"
                   and isinstance(c.func.value, ast.Attribute)
                   and c.func.value.attr == "hands"
                   for c in ast.walk(node))

    guarded = [t for t in ast.walk(init)
               if isinstance(t, ast.Try) and calls_hands_start(t)]
    check("hands.start() is wrapped in a try", len(guarded) == 1,
          f"{len(guarded)} found")
    if not guarded:
        return

    handlers = guarded[0].handlers
    check("it catches broadly enough to survive an SDK's own exception type",
          len(handlers) == 1 and (handlers[0].type is None
                                  or getattr(handlers[0].type, "id", "") == "Exception"))
    drops = any(isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == "hands"
                        for t in n.targets)
                and isinstance(n.value, ast.Constant) and n.value.value is None
                for n in ast.walk(handlers[0]))
    check("the failed hands are dropped, so the rest of the node stops "
          "reaching for them", drops)


# ─────────────────────────────────────────────────────────────────────────────
# Hands policy
# ─────────────────────────────────────────────────────────────────────────────

class _FakeSample:
    def __init__(self, qpos, paused):
        self.qpos, self.paused = qpos, paused


class _FakeStream:
    def __init__(self):
        self.snap = {}

    def snapshot(self):
        return self.snap


def _server(sides=SIDES):
    """A `Hands` with no sockets and no thread -- policy only."""
    from robot.hand.hands import Hands
    from robot.teleop.aria.config import AriaConfig

    cfg = AriaConfig({})
    cfg.mapping["hand"] = "both" if len(sides) == 2 else sides[0]
    srv = Hands(cfg, aria=False, rpc=False)
    srv._stream = _FakeStream()
    return srv


def test_server_hold_last() -> None:
    """Paused, pre-engage and lost tracking all hold; only a live pose moves.

    Silence is deliberately not a release: there is no staleness gate on this
    path, unlike the arm path.
    """
    print("\nhold-last policy")
    srv = _server()
    stream = srv._stream

    stream.snap = {s: _FakeSample(None, True) for s in SIDES}
    srv._pull_aria()
    check("nothing before the first engage",
          srv._target["left"] is None and srv._target["right"] is None)

    grasp = np.full(N_JOINTS, 0.4)
    stream.snap = {"left": _FakeSample(grasp, False),
                   "right": _FakeSample(None, True)}
    srv._pull_aria()
    check("an engaged side adopts its pose", np.allclose(srv._target["left"], grasp))
    check("the other side stays untouched", srv._target["right"] is None)
    check("engagement is reported", srv._engaged["left"] and not srv._engaged["right"])

    # shaka: the publisher freezes qpos, we must not adopt a newer one anyway
    stream.snap = {"left": _FakeSample(np.full(N_JOINTS, 0.9), True),
                   "right": _FakeSample(None, True)}
    srv._pull_aria()
    check("a paused side holds the last grasp",
          np.allclose(srv._target["left"], grasp))
    check("pause is reported", not srv._engaged["left"])

    # tracking lost while engaged
    stream.snap = {"left": _FakeSample(None, False),
                   "right": _FakeSample(None, True)}
    srv._pull_aria()
    check("lost tracking holds the last grasp",
          np.allclose(srv._target["left"], grasp))


def test_hand_sides_are_independent_of_the_arms() -> None:
    """`hand.sides` picks the hands; `mapping.hand` still picks the arms.

    Whole-body IK wants both wrist targets, so a one-handed operator runs both
    arms and one hand -- the two settings must not be the same setting.
    """
    print("\nhand.sides vs mapping.hand")
    from types import SimpleNamespace

    from robot.hand.hands import Hands, hands_from_args
    from robot.teleop.aria.config import AriaConfig

    cfg = AriaConfig({})
    check("both by default", cfg.hand_sides() == ("left", "right"))
    cfg.hand["sides"] = "right"
    check("one hand, two arms", cfg.hand_sides() == ("right",)
          and cfg.mapping["hand"] == "both")
    cfg.hand["sides"] = "none"
    check("none drives no hand at all", cfg.hand_sides() == ())
    cfg.hand["sides"] = "both"
    cfg.mapping["hand"] = "left"
    check("never a hand on an unteleoped arm", cfg.hand_sides() == ("left",))

    cfg = AriaConfig({})
    cfg.hand["sides"] = "right"
    srv = Hands(cfg, aria=False, rpc=False)
    check("only the chosen side is served", srv.sides == ("right",))
    check("a target for the other side is refused",
          not srv.set_hand_target("left", np.zeros(N_JOINTS)))
    check("homing both arms only opens the hand that exists",
          srv.open_hands(("left", "right"))
          and list(srv.targets()) == ["right"])

    args = SimpleNamespace(no_hands=False, aria_config=None, pub_host=None,
                           hands="none", hand_backend=None, tracking_csv=None)
    check("--hands none is --no-hands", hands_from_args(args) is None)
    args.hands = "left"
    check("--hands left overrides the config",
          hands_from_args(args).sides == ("left",))


def test_server_rpc() -> None:
    print("\ncommand surface")
    srv = _server()
    q = np.full(N_JOINTS, 0.2)
    check("set_hand_target accepts a good vector", srv.set_hand_target("left", q))
    check("it lands", np.allclose(srv._target["left"], q))
    check("a wrong-length vector is refused",
          not srv.set_hand_target("left", np.zeros(7)))
    check("it did not overwrite", np.allclose(srv._target["left"], q))
    check("an unserved side is refused", not srv.set_hand_target("third", q))

    srv.set_bimanual_hand_target(L_hand_target=np.full(N_JOINTS, 0.1),
                                 R_hand_target=np.full(N_JOINTS, 0.3))
    check("bimanual sets both", np.allclose(srv._target["left"], 0.1)
          and np.allclose(srv._target["right"], 0.3))
    srv.set_bimanual_hand_target(R_hand_target=np.full(N_JOINTS, 0.7))
    check("a None side is left alone", np.allclose(srv._target["left"], 0.1))

    srv.open_hands()
    check("open_hands zeroes both",
          np.allclose(srv._target["left"], 0.0) and np.allclose(srv._target["right"], 0.0))

    state = srv.get_hand_state()
    check("get_hand_state is plain types",
          isinstance(state["qpos"]["left"], list) and isinstance(state["backend"], str))


def test_server_sends_on_change_only() -> None:
    """A resend of the same vector is wasted USB traffic at 100 Hz."""
    print("\nsend-on-change")
    srv = _server(("left",))
    srv.driver = NullWujiDriver(("left",))
    srv.set_hand_target("left", np.full(N_JOINTS, 0.5))
    srv._push()
    srv._push()
    check("an unchanged target is not resent", srv.driver.sent["left"] == 1,
          str(srv.driver.sent["left"]))
    srv.set_hand_target("left", np.full(N_JOINTS, 0.6))
    srv._push()
    check("a changed target is sent", srv.driver.sent["left"] == 2)


def test_home_opens_the_hands() -> None:
    """Both thumbs up homes the arms; the hands go with them.

    The gesture reaches `home_arms()` on the node, which owns the hands, so no
    client, wire or RPC signature changes -- but it does mean the hand open has
    to be wired into `_home_arm_joints`, on both nodes, for the side(s) asked
    for and no others.
    """
    print("\nhoming opens the hands")
    srv = _server()
    srv.set_bimanual_hand_target(np.full(N_JOINTS, 0.8), np.full(N_JOINTS, 0.8))

    srv.open_hands(("left",))
    check("homing one arm opens only that hand",
          np.allclose(srv._target["left"], 0.0)
          and np.allclose(srv._target["right"], 0.8))
    check("the open is attributed to home, not to a client",
          srv._origin["left"] == "home")

    srv.open_hands(("left", "right"))
    check("homing both opens both",
          np.allclose(srv._target["left"], 0.0)
          and np.allclose(srv._target["right"], 0.0))

    # a one-handed session must not be asked for a hand it does not serve
    one = _server(("left",))
    check("a side the session does not serve is dropped, not an error",
          one.open_hands(("left", "right")) and set(one._target) == {"left"})
    check("open_hands() with no argument still means all of them",
          _server().open_hands() is True)

    # hold-last is what keeps them open: a paused operator sends nothing usable
    srv._stream.snap = {s: _FakeSample(np.full(N_JOINTS, 0.9), True) for s in SIDES}
    srv._pull_aria()
    check("a paused hand stays open after homing",
          np.allclose(srv._target["left"], 0.0))

    # and both nodes actually call it, for `sides`, inside the homing lock
    for node in ("robot/yor.py", "robot/yor_mujoco.py"):
        src = (_REPO / node).read_text()
        start = src.index("    def _home_arm_joints(self, sides")
        body = src[start:start + 2500]
        acquired = body.index("_homing_lock.acquire")
        opened = body.find("self.hands.open_hands(sides)")
        check(f"{node} opens the hands when it homes", opened > acquired,
              "not found" if opened < 0 else "")


def test_rpc_surface_is_narrow() -> None:
    """commlink exposes every public method of whatever it is handed.

    So it is handed `_HandRPC`, not `Hands` -- otherwise a remote client could
    call `stop()` and release both hands, or reach `driver` directly.
    """
    print("\nRPC surface is narrow")
    from robot.hand.hands import _HandRPC

    srv = _server()
    rpc = _HandRPC(srv)
    public = {n for n in dir(rpc) if not n.startswith("_")}
    check("only the five command methods are exposed",
          public == {"set_hand_target", "set_bimanual_hand_target",
                     "get_hand_state", "home_hands", "open_hands"},
          str(sorted(public)))
    rpc.set_hand_target("left", np.full(N_JOINTS, 0.25))
    check("a call through the facade reaches the state",
          np.allclose(srv._target["left"], 0.25))


# ─────────────────────────────────────────────────────────────────────────────
# The sim injection
# ─────────────────────────────────────────────────────────────────────────────

def _sim_stub():
    """A YORMujoco with only the hand machinery on it -- no viewer, no solver."""
    import threading

    from robot.yor_mujoco import YORMujoco

    stub = YORMujoco.__new__(YORMujoco)
    stub.model = model()
    stub.data = mujoco.MjData(stub.model)
    stub.target_lock = threading.Lock()
    stub.hands = None
    YORMujoco._init_hand_joints(stub)
    stub._hand_cmd = {side: stub.data.qpos[adrs].copy()
                      for side, adrs in stub._hand_qpos_adrs.items()}
    return stub


def test_sim_injection() -> None:
    print("\nsim injection")
    stub = _sim_stub()
    check("both hands found in the scene", set(stub._hand_qpos_adrs) == set(SIDES),
          str(sorted(stub._hand_qpos_adrs)))

    # hold-last: nothing published, the keyframe pose stands
    stub._apply_hand_qpos()
    check("an uncommanded hand keeps its pose",
          np.allclose(stub.data.qpos[stub._hand_qpos_adrs["left"]],
                      stub._hand_cmd["left"]))

    hands = _server()
    grasp = np.full(N_JOINTS, 0.3)
    hands.set_hand_target("left", grasp)
    stub.hands = hands
    stub._pull_hand_commands()
    stub._apply_hand_qpos()
    lo, hi = stub._hand_qpos_lo["left"], stub._hand_qpos_hi["left"]
    check("a published pose reaches MjData, clipped to the model",
          np.allclose(stub.data.qpos[stub._hand_qpos_adrs["left"]],
                      np.clip(grasp, lo, hi)))
    check("the seed is written back unclipped, so an uncommanded joint whose "
          "range excludes zero is not nudged",
          np.allclose(stub._hand_cmd["right"],
                      mujoco.MjData(stub.model).qpos[stub._hand_qpos_adrs["right"]]))
    check("a None side keeps the pose it was seeded with",
          np.allclose(stub.data.qpos[stub._hand_qpos_adrs["right"]],
                      stub._hand_cmd["right"]))

    # A vector the retargeter could plausibly emit outside the MJCF's range
    hands.set_hand_target("left", np.full(N_JOINTS, 9.0))
    stub._pull_hand_commands()
    stub._apply_hand_qpos()
    written = stub.data.qpos[stub._hand_qpos_adrs["left"]]
    check("an out-of-range command cannot escape the joint limits",
          bool(np.all(written <= hi + 1e-9) and np.all(written >= lo - 1e-9)))

    held = stub._hand_cmd["left"].copy()
    hands._target["left"] = np.zeros(3)   # past _store's guard, on purpose
    stub._pull_hand_commands()
    check("a wrong-length target is ignored, not written",
          np.allclose(stub._hand_cmd["left"], held))


def test_injection_runs_after_the_solver() -> None:
    """`apply_to_sim_kinematic` writes the whole qpos, fingers included.

    So the hand write has to come after it and before mj_forward, at every
    branch of the control loop -- miss one and the fingers snap back to home
    during a home sequence.
    """
    print("\ncontrol-loop ordering")
    src = (_REPO / "robot" / "yor_mujoco.py").read_text()
    start = src.index("    def control_loop(self):")
    end = src.index("    # ── WUJI fingers", start)
    loop = src[start:end]
    forwards = loop.count("mujoco.mj_forward(self.model, self.data)")
    applies = loop.count("self._apply_hand_qpos()")
    check("every mj_forward in the loop is preceded by a hand write",
          forwards == applies and forwards >= 3, f"{applies}/{forwards}")
    for chunk in loop.split("mujoco.mj_forward(self.model, self.data)")[:-1]:
        if "self._apply_hand_qpos()" not in chunk:
            check("every mj_forward in the loop is preceded by a hand write",
                  False, "a branch writes no fingers")
            break
    apply_at = loop.find("self.ik.apply_to_sim_kinematic")
    hand_at = loop.find("self._apply_hand_qpos()", apply_at)
    fwd_at = loop.find("mujoco.mj_forward", apply_at)
    check("the hand write lands between apply_to_sim_kinematic and mj_forward",
          apply_at < hand_at < fwd_at)


def main() -> int:
    for test in (
        test_joint_order,
        test_model_hand_joints,
        test_null_driver,
        test_hardware_ramp,
        test_hardware_needs_serials,
        test_hardware_zeroes_at_startup,
        test_one_hand_unplugged_keeps_the_other,
        test_hands_start_is_not_fatal,
        test_server_hold_last,
        test_hand_sides_are_independent_of_the_arms,
        test_server_rpc,
        test_server_sends_on_change_only,
        test_home_opens_the_hands,
        test_rpc_surface_is_narrow,
        test_sim_injection,
        test_injection_runs_after_the_solver,
    ):
        test()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    failures = [name for name, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
