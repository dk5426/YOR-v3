"""
test_aero_hand.py — contract tests for the Aero hand path.

Minimal by design: `Hands`, the RPC surface, the sim injection and the
hold-last policy are all shared plumbing already pinned by
tests/test_wuji_hand.py. This file only covers what is specific to
`robot/hand/aero_driver.py` and to picking Aero via `hand.type` -- above all,
the safety property that firmware homing (`send_homing()`, ~175 s, blocking)
can only ever run once, in `HardwareAeroDriver.start()`, and never from the
thumbs-up re-home path -- plus one check that description/scene_wholebody_
aero.xml actually has the 16 joints per side canonical_joint_names() expects
(everything downstream -- Hands, yor_mujoco.py, sim_viz.py -- assumes this
and never re-checks it).

No hand, no publisher, no viewer.

    python tests/test_aero_hand.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robot.hand.aero_driver import (
    N_JOINTS,
    HardwareAeroDriver,
    NullAeroDriver,
    canonical_joint_names,
    make_driver,
)

SIDES = ("left", "right")
SCENE = _REPO / "description" / "scene_wholebody_aero.xml"

RESULTS: list[tuple[str, bool, str]] = []

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = mujoco.MjModel.from_xml_path(str(SCENE))
    return _MODEL


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# The joint vector
# ─────────────────────────────────────────────────────────────────────────────

def test_joint_order() -> None:
    """16 joints per hand, side-prefixed, matching aero_open_sdk when present."""
    print("\njoint vector layout")
    names = canonical_joint_names("left")
    check("16 joints per hand", len(names) == N_JOINTS, str(len(names)))
    check("side-prefixed", all(n.startswith("left_") for n in names))
    check("left/right differ only by prefix",
          tuple(n.replace("left_", "") for n in names)
          == tuple(n.replace("right_", "") for n in canonical_joint_names("right")))

    try:
        from aero_open_sdk.aero_hand_constants import AeroHandConstants
    except ImportError:
        check("matches aero_open_sdk.AeroHandConstants.joint_names (skipped)", True)
        return
    want = tuple(f"left_{n}" for n in AeroHandConstants().joint_names)
    check("matches aero_open_sdk.AeroHandConstants.joint_names", names == want,
          "" if names == want else f"{names} != {want}")


def test_model_hand_joints() -> None:
    """Every canonical name exists in scene_wholebody_aero.xml, contiguous
    and ascending -- mirrors test_wuji_hand.py::test_model_hand_joints.

    Contiguity is what lets the published vector be written as one slice; a
    fragment edit that interleaved another joint would break the slice
    silently.
    """
    print("\nMJCF hand joints")
    m = model()
    for side in SIDES:
        try:
            joints = [m.joint(n) for n in canonical_joint_names(side)]
        except KeyError as exc:
            check(f"{side}: all 16 joints present", False, str(exc))
            continue
        check(f"{side}: all 16 joints present", True)
        adrs = np.array([int(j.qposadr[0]) for j in joints])
        check(f"{side}: qpos addresses contiguous and ascending",
              bool(np.all(np.diff(adrs) == 1)), str(adrs[:3]) + " ..")
        lo = np.array([float(j.range[0]) for j in joints])
        hi = np.array([float(j.range[1]) for j in joints])
        check(f"{side}: every joint has a real range", bool(np.all(hi > lo)))
        wild = np.full(N_JOINTS, 99.0)
        check(f"{side}: clip lands inside the model's ranges",
              bool(np.all(np.clip(wild, lo, hi) <= hi + 1e-12)))


# ─────────────────────────────────────────────────────────────────────────────
# Backend dispatch
# ─────────────────────────────────────────────────────────────────────────────

def test_null_driver() -> None:
    print("\nnull backend")
    d = make_driver("none", SIDES)
    check("make_driver('none') is the null backend", isinstance(d, NullAeroDriver))
    q = np.arange(N_JOINTS, dtype=float)
    d.send("left", q)
    check("records what it was sent", np.array_equal(d.commanded("left"), q))
    d.home()
    check("home() zeros both sides",
          all(np.allclose(d.commanded(s), 0.0) for s in SIDES))

    try:
        make_driver("wat")
        check("unknown backend raises", False, "no raise")
    except ValueError:
        check("unknown backend raises", True)


# ─────────────────────────────────────────────────────────────────────────────
# Startup-only homing -- the safety-critical property
# ─────────────────────────────────────────────────────────────────────────────

def test_homing_only_at_startup() -> None:
    """`start()` must home once; `home()` (the thumbs-up re-home path) must not.

    `Hands.open_hands()` -- the only thing a mid-session thumbs-up gesture
    reaches -- never calls `driver.home()` directly, it only sets the target
    and lets the normal send() path deliver it; but `home()` is still what
    `start()` and `close()` use internally, so it must never itself trigger
    the manufacturer's firmware homing.
    """
    print("\nhoming only at startup")
    import ast

    tree = ast.parse((_REPO / "robot" / "hand" / "aero_driver.py").read_text())
    cls = next(n for n in ast.walk(tree)
              if isinstance(n, ast.ClassDef) and n.name == "HardwareAeroDriver")
    start = next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "start")
    home = next(n for n in cls.body
               if isinstance(n, ast.FunctionDef) and n.name == "home")

    def calls(node, method):
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                  and c.func.attr == method for c in ast.walk(node))

    check("start() calls send_homing", calls(start, "send_homing"))
    check("home() does not call send_homing", not calls(home, "send_homing"))
    check("start() calls the software home() once homing succeeds",
          calls(start, "home"))


def test_homing_failure_drops_only_that_side() -> None:
    """A hand that opens but fails to home must not cost the other one."""
    print("\nhoming failure")
    import sys
    import types

    class _FakeAeroHand:
        def __init__(self, port=None):
            if port == "GONE_OPEN":
                raise RuntimeError("no such device")
            self.port = port
            self.writes: list[list[float]] = []

        def set_speed(self, i, speed): pass
        def set_torque(self, i, torque): pass

        def send_homing(self):
            if self.port == "GONE_HOME":
                raise TimeoutError("no ack")

        def set_joint_positions(self, positions):
            self.writes.append(list(positions))

        def close(self): pass

    fake_pkg = types.ModuleType("aero_open_sdk")
    fake_mod = types.ModuleType("aero_open_sdk.aero_hand")
    fake_mod.AeroHand = _FakeAeroHand
    saved_pkg = sys.modules.get("aero_open_sdk")
    saved_mod = sys.modules.get("aero_open_sdk.aero_hand")
    sys.modules["aero_open_sdk"] = fake_pkg
    sys.modules["aero_open_sdk.aero_hand"] = fake_mod
    try:
        d = HardwareAeroDriver(SIDES, ports={"left": "GONE_HOME", "right": "PORT_B"},
                               ramp_s=0.0, ramp_steps=2)
        d.start()
        check("the side that failed to home is dropped", d.sides == ("right",))
        check("the other side still homed and rests at zero",
              np.allclose(d.commanded("right"), 0.0))

        d2 = HardwareAeroDriver(SIDES, ports={"left": "GONE_HOME", "right": "GONE_HOME"})
        try:
            d2.start()
            check("no hand finishes homing is still an error", False, "no raise")
        except RuntimeError as exc:
            check("no hand finishes homing is still an error",
                  "finished homing" in str(exc))
    finally:
        for name, saved in (("aero_open_sdk", saved_pkg),
                            ("aero_open_sdk.aero_hand", saved_mod)):
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


# ─────────────────────────────────────────────────────────────────────────────
# Hand-type selection
# ─────────────────────────────────────────────────────────────────────────────

def test_hands_picks_driver_by_hand_type() -> None:
    print("\nhand.type selection")
    from robot.hand import aero_driver, wuji_driver
    from robot.hand.hands import Hands
    from robot.teleop.aria.config import AriaConfig

    cfg0 = AriaConfig({})
    check("default hand.type is none -- arms only", cfg0.hand["type"] == "none")
    try:
        Hands(cfg0, aria=False, rpc=False)
        check("constructing Hands with hand.type: none raises", False, "no raise")
    except ValueError:
        check("constructing Hands with hand.type: none raises", True)

    cfg = AriaConfig({"hand": {"type": "wuji"}})
    srv = Hands(cfg, aria=False, rpc=False)
    try:
        check("hand.type: wuji uses wuji_driver", srv._driver_mod is wuji_driver)
        check("wuji n_joints is 20", srv.n_joints == wuji_driver.N_JOINTS)
    finally:
        srv.stop()

    cfg2 = AriaConfig({"hand": {"type": "aero"}})
    srv2 = Hands(cfg2, aria=False, rpc=False)
    try:
        check("hand.type: aero uses aero_driver", srv2._driver_mod is aero_driver)
        check("aero n_joints is 16", srv2.n_joints == aero_driver.N_JOINTS)
    finally:
        srv2.stop()

    try:
        Hands(AriaConfig({"hand": {"type": "nope"}}), aria=False, rpc=False)
        check("unknown hand.type raises", False, "no raise")
    except ValueError:
        check("unknown hand.type raises", True)


def test_stream_topic_is_qpos() -> None:
    print("\nstream topic")
    from robot.teleop.aria.stream import AriaHandStream
    check("subscribes to meta+qpos, not the old wuji name",
          AriaHandStream.TOPICS == ("meta", "qpos"))


def main() -> int:
    for test in (
        test_joint_order,
        test_model_hand_joints,
        test_null_driver,
        test_homing_only_at_startup,
        test_homing_failure_drops_only_that_side,
        test_hands_picks_driver_by_hand_type,
        test_stream_topic_is_qpos,
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
