#!/usr/bin/env python3
"""test_04_arms.py — arms track targets, respect limits, and stop on demand.

⚠️  THE ARMS MOVE. Stand clear. The base is held still throughout.

Everything here runs with the base fixed (fix_base ON), so the only things that
can move are the arms and — in the last test — the lift. Whole-body base motion
is exercised separately in test_07.

    python tests/hardware/test_04_arms.py --host <robot-ip>
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hw import (  # noqa: E402
    ask_yes_no, banner, check, confirm, connect, countdown, guard, info,
    parse_args, precondition, run,
)

CLIENT = None
ARGS = None

STEP_M = 0.05      # how far to nudge an end effector
SETTLE_S = 2.5


def _state() -> dict:
    return CLIENT.call("get_state") or {}


def _ee(side: str):
    s = _state()
    v = s.get(f"{side}_ee_wxyz_xyz")
    return None if v is None else list(v)


def _wait_settle(seconds: float = SETTLE_S) -> None:
    time.sleep(seconds)


def test_preconditions():
    print("\npreconditions")
    precondition(
        "A 1 m clear sphere around BOTH arms — no people, no equipment, no walls.",
        "The arms are not holding anything, and the grippers are empty.",
        "Nothing is on the lift platform that the arms could strike.",
        "You can reach the physical e-stop / power cut.",
        "You are standing outside the arms' reach, not between them.",
    )
    check("operator confirmed the workspace is clear", True)


def test_fix_base():
    print("\nlock the base")
    fixed = CLIENT.call("toggle_fix_base", True)
    check("base is fixed for the duration of this test", fixed is True, f"fix_base={fixed}")
    if fixed is not True:
        info("Without fix_base the solver may roll the chassis to help the arms "
             "reach. Do not continue on the floor until this returns True.")


def test_collision_avoidance_on():
    print("\ncollision avoidance")
    enabled = CLIENT.call("toggle_collision_avoidance", True)
    check("self-collision and ground avoidance are enabled", enabled is True,
          f"collision_avoidance={enabled}")
    info("These are hard QP constraints, so the solver cannot trade them away "
         "against an end-effector target. Leave them on.")


def test_joint_readback():
    print("\njoint readback")
    left = CLIENT.call("get_left_joint_positions")
    right = CLIENT.call("get_right_joint_positions")
    for side, q in (("left", left), ("right", right)):
        check(f"{side} arm reports 7 finite joints",
              q is not None and len(q) == 7 and all(v == v for v in q),
              ", ".join(f"{v:+.3f}" for v in (q or [])))
    st = _state()
    for side in ("left", "right"):
        model_q = st.get(f"{side}_joint_positions")
        meas_q = left if side == "left" else right
        if model_q and meas_q:
            err = max(abs(a - b) for a, b in zip(model_q, meas_q))
            check(f"solver's {side} model matches the encoders", err < 0.05,
                  f"max |dq| = {err:.4f} rad")
            if err >= 0.05:
                info("The IK model has drifted from the real arm. Collision "
                     "avoidance is computed on the model, so this is a safety "
                     "issue, not a cosmetic one.")


def _nudge(side: str, axis: int, sign: float, axis_name: str):
    import mink

    before = _ee(side)
    if before is None:
        check(f"{side} EE pose available", False)
        return
    pos = list(before[4:7])
    quat = list(before[0:4])
    pos[axis] += sign * STEP_M

    confirm(f"Move the {side.upper()} hand {STEP_M * 100:.0f} cm along {axis_name}.")
    with guard(CLIENT, estop=True):
        countdown(3, f"{side} hand {axis_name}")
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(quat, dtype=float)),
            np.asarray(pos, dtype=float),
        )
        CLIENT.call(f"set_{side}_ee_target", target)
        _wait_settle()

    after = _ee(side)
    if after is None:
        check(f"{side} EE pose after move", False)
        return
    moved = np.asarray(after[4:7]) - np.asarray(before[4:7])
    achieved = float(moved[axis])
    check(f"{side} hand moved along {axis_name}",
          abs(achieved - sign * STEP_M) < 0.02,
          f"asked {sign * STEP_M:+.3f} m, got {achieved:+.3f} m")
    off_axis = math.sqrt(sum(moved[i] ** 2 for i in range(3) if i != axis))
    check(f"{side} hand did not wander off-axis", off_axis < 0.02,
          f"{off_axis * 1000:.0f} mm off-axis")


def test_left_hand_tracks():
    print("\nleft hand follows a target")
    _nudge("left", 2, +1.0, "+Z (up)")


def test_right_hand_tracks():
    print("\nright hand follows a target")
    _nudge("right", 2, +1.0, "+Z (up)")


def test_return_home():
    print("\nreturn to home pose")
    confirm("Send BOTH arms to their latched home poses.")
    with guard(CLIENT, estop=True):
        countdown(3, "homing both arms")
        CLIENT.call("home_left_arm")
        CLIENT.call("home_right_arm")
        _wait_settle(4.0)
    got = ask_yes_no("Did both arms return smoothly to their home pose?")
    if got is not None:
        check("both arms home cleanly", got)


def test_grippers():
    print("\ngrippers")
    confirm("Open then close BOTH grippers.")
    with guard(CLIENT, estop=True):
        for action in ("open", "close"):
            countdown(2, f"{action} grippers")
            CLIENT.call(f"{action}_left_gripper")
            CLIENT.call(f"{action}_right_gripper")
            time.sleep(2.0)
    got = ask_yes_no("Did both grippers open and close?")
    if got is not None:
        check("grippers respond", got)


def test_lift_under_solver():
    print("\nlift as a solver DOF")
    status = CLIENT.call("get_lift_status") or {}
    if status.get("position_known") is not True:
        check("lift homed before using it as a DOF", False,
              "run test_02_lift.py first — the solver trusts get_lift_height()")
        return

    start = CLIENT.call("get_lift_height")
    ee_before = _ee("left")
    target = max(0.10, min(0.80, start + 0.10))
    confirm(f"Ask the solver for lift = {target:.3f} m while the hands hold station.")
    with guard(CLIENT, estop=True):
        countdown(3, "lift under the solver")
        CLIENT.call("set_lift_target", target)
        _wait_settle(6.0)

    end = CLIENT.call("get_lift_height")
    ee_after = _ee("left")
    if end is not None and start is not None:
        check("lift moved toward the requested height",
              abs(end - target) < 0.03, f"{start:.3f} -> {end:.3f} m (asked {target:.3f})")
    if ee_before and ee_after:
        drift = math.dist(ee_before[4:7], ee_after[4:7])
        check("the hand held station while the torso moved", drift < 0.05,
              f"{drift * 1000:.0f} mm of hand drift")
        if drift >= 0.05:
            info("The arms are supposed to compensate for lift motion. Large "
                 "drift means the lift height feeding the IK is wrong or lagging.")


def main() -> int:
    global CLIENT, ARGS, np
    import numpy as _np
    np = _np
    ARGS = parse_args(__doc__)
    banner("STAGE 1 — ARMS",
           "*** THE ARMS MOVE. Base is held fixed. Ctrl-C e-stops. ***")
    CLIENT = connect(ARGS)
    try:
        return run(
            test_preconditions,
            test_fix_base,
            test_collision_avoidance_on,
            test_joint_readback,
            test_left_hand_tracks,
            test_right_hand_tracks,
            test_return_home,
            test_grippers,
            test_lift_under_solver,
        )
    finally:
        CLIENT.estop()


if __name__ == "__main__":
    raise SystemExit(main())
