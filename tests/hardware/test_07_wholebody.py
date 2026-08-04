#!/usr/bin/env python3
"""test_07_wholebody.py — the solver coordinates arms, lift and base.

⚠️  ARMS, LIFT **AND WHEELS** MOVE — the base moves without anyone commanding
    it. That is the whole point of whole-body control, and it is the single
    most surprising behaviour on this robot.

Run this LAST, and only after test_02 (lift), test_03 (base axes) and test_04
(arms) have all passed. It graduates in three steps and stops between each:

    1. base fixed          — only the arms may move
    2. base fixed, lift on — the torso moves, the hands hold station
    3. base released       — the chassis rolls to extend the arms' reach

    python tests/hardware/test_07_wholebody.py --host <robot-ip>
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from _hw import (  # noqa: E402
    ask_yes_no, banner, check, confirm, connect, countdown, guard, info,
    parse_args, precondition, run,
)

CLIENT = None
ARGS = None
SETTLE_S = 3.0


def _state() -> dict:
    return CLIENT.call("get_state") or {}


def _ee(side: str):
    v = _state().get(f"{side}_ee_wxyz_xyz")
    return None if v is None else list(v)


def _set_ee(side: str, pos, quat):
    import mink

    target = mink.SE3.from_rotation_and_translation(
        mink.SO3(np.asarray(quat, dtype=float)),
        np.asarray(pos, dtype=float),
    )
    CLIENT.call(f"set_{side}_ee_target", target)


def test_preconditions():
    print("\npreconditions")
    precondition(
        "test_02 (lift), test_03 (base axes) and test_04 (arms) have all PASSED.",
        "The robot is on the FLOOR with 2 m clear in EVERY direction.",
        "A 1 m clear sphere around both arms — nobody within reach.",
        "You are holding the e-stop, or standing next to the power cut.",
        "You accept that the chassis will roll on its own in the last step.",
    )
    check("operator confirmed the setup", True)


def test_solver_healthy():
    print("\nsolver health")
    states = [_state() for _ in range(10)]
    solved = sum(1 for s in states if s.get("solved"))
    check("solver converges", solved >= 9, f"{solved}/10")
    check("collision avoidance is ON", states[-1].get("collision_avoidance") is True,
          str(states[-1].get("collision_avoidance")))
    if states[-1].get("collision_avoidance") is not True:
        info("Turn it on before proceeding — it is a hard constraint that stops "
             "the arms hitting the lift column, each other and the floor.")


def test_stage1_base_fixed():
    print("\nstage 1 — base fixed, arms only")
    fixed = CLIENT.call("toggle_fix_base", True)
    check("base is fixed", fixed is True, f"fix_base={fixed}")

    before = _ee("left")
    if before is None:
        check("left EE readable", False)
        return
    pos = list(before[4:7])
    pos[2] += 0.08
    confirm("Raise the LEFT hand 8 cm with the base LOCKED.")
    with guard(CLIENT, estop=True):
        countdown(3, "left hand up, base locked")
        _set_ee("left", pos, before[0:4])
        time.sleep(SETTLE_S)

    st = _state()
    base_cmd = st.get("base_command") or [0, 0, 0]
    check("the base was not commanded while fixed",
          max(abs(v) for v in base_cmd) < 1e-6,
          ", ".join(f"{v:+.4f}" for v in base_cmd))
    after = _ee("left")
    if after:
        rose = after[6] - before[6]
        check("the hand rose about 8 cm", abs(rose - 0.08) < 0.03, f"{rose:+.3f} m")
    got = ask_yes_no("Did the wheels stay completely still?")
    if got is not None:
        check("wheels still while fix_base is on", got)


def test_stage2_lift_participates():
    print("\nstage 2 — lift joins in, hands hold station")
    status = CLIENT.call("get_lift_status") or {}
    if status.get("position_known") is not True:
        check("lift homed", False, "run test_02_lift.py first")
        return

    start_h = CLIENT.call("get_lift_height")
    ee_before = _ee("left")
    target_h = max(0.10, min(0.80, start_h + 0.12))

    confirm(f"Ask for lift = {target_h:.3f} m. The arms should compensate so the "
            f"hands stay put.")
    with guard(CLIENT, estop=True):
        countdown(3, "lift under the solver")
        CLIENT.call("set_lift_target", target_h)
        time.sleep(6.0)

    end_h = CLIENT.call("get_lift_height")
    ee_after = _ee("left")
    if end_h is not None:
        check("the lift reached the requested height", abs(end_h - target_h) < 0.03,
              f"{start_h:.3f} -> {end_h:.3f} m")
    if ee_before and ee_after:
        drift = math.dist(ee_before[4:7], ee_after[4:7])
        check("the hand held station while the torso moved", drift < 0.05,
              f"{drift * 1000:.0f} mm")
        if drift >= 0.05:
            info("The arms are not compensating. Either the lift height feeding "
                 "the IK is lagging, or lift_posture_cost is so low the solver "
                 "is happy to move the lift instead of holding the hand still.")


def test_stage3_base_released():
    print("\nstage 3 — base released  *** THE CHASSIS WILL ROLL ***")
    info("The solver rolls the base only when the arms and lift together cannot "
         "reach. Nothing is sending an explicit drive command.")
    precondition(
        "2 m clear in EVERY direction, right now.",
        "Everyone is clear of the robot's path, not just its arms.",
        "Your hand is on the e-stop.",
    )
    confirm("RELEASE the base and push a target beyond arm reach.", token="ROLL")

    released = CLIENT.call("toggle_fix_base", False)
    check("base released", released is False, f"fix_base={released}")

    before = _ee("left")
    base_before = _state().get("base_xytheta") or [0, 0, 0]
    if before is None:
        check("left EE readable", False)
        return

    pos = list(before[4:7])
    pos[1] -= 0.45          # the robot faces -Y: push the target out in front
    with guard(CLIENT, estop=True):
        countdown(5, "target beyond arm reach — CHASSIS WILL MOVE")
        _set_ee("left", pos, before[0:4])
        time.sleep(6.0)
        CLIENT.call("set_base_velocity", [0.0, 0.0, 0.0])

    st = _state()
    base_after = st.get("base_xytheta") or [0, 0, 0]
    rolled = math.hypot(base_after[0] - base_before[0], base_after[1] - base_before[1])
    check("the chassis rolled toward the target", rolled > 0.02, f"{rolled:.3f} m")
    if rolled <= 0.02:
        info("It did not roll. Check base_motion_enabled, and that "
             "base_posture_cost is not so high the solver refuses to move the base.")

    vel = st.get("base_velocity") or [0, 0, 0]
    check("base velocity stayed within the configured clamp",
          max(abs(v) for v in vel[:2]) <= 0.26 and abs(vel[2]) <= 0.61,
          ", ".join(f"{v:+.3f}" for v in vel))

    got = ask_yes_no("Did the chassis roll SLOWLY and in the direction of the target?")
    if got is not None:
        check("base rolled slowly, toward the target", got)
        if not got:
            info("Wrong direction means a BaseAxisMap sign is inverted — go back "
                 "to test_03. Do not 'fix' it anywhere but BaseAxisMap.")


def test_recover():
    print("\nrecovery")
    confirm("Re-fix the base and send both arms home.")
    with guard(CLIENT, estop=True):
        CLIENT.call("toggle_fix_base", True)
        countdown(3, "homing arms")
        CLIENT.call("home_left_arm")
        CLIENT.call("home_right_arm")
        time.sleep(5.0)
    st = _state()
    check("base is fixed again", st.get("fix_base") is True, str(st.get("fix_base")))
    check("solver still converging after the run", st.get("solved") is True,
          str(st.get("solved")))
    got = ask_yes_no("Is the robot back in a safe, stable pose?")
    if got is not None:
        check("robot recovered to a safe pose", got)


def main() -> int:
    global CLIENT, ARGS
    ARGS = parse_args(__doc__)
    banner("STAGE 3 — WHOLE-BODY CONTROL",
           "*** ARMS, LIFT AND WHEELS ALL MOVE. ***",
           "*** The chassis rolls with nobody commanding it. ***",
           "Ctrl-C e-stops.")
    CLIENT = connect(ARGS)
    try:
        return run(
            test_preconditions,
            test_solver_healthy,
            test_stage1_base_fixed,
            test_stage2_lift_participates,
            test_stage3_base_released,
            test_recover,
        )
    finally:
        CLIENT.estop()


if __name__ == "__main__":
    raise SystemExit(main())
