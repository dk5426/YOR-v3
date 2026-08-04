#!/usr/bin/env python3
"""test_03_manual_base.py — direct base control: does it go where you ask?

⚠️  THE WHEELS TURN. Run the first half with the robot ON BLOCKS.

This is the axis-convention test. Every sign in the stack — BaseAxisMap in
wholebody_control.py, the module order in base_motor.py, the swerve forward
kinematics in nav/odometry — depends on the base moving the way this test says
it should. Getting a sign wrong here makes the whole-body solver drive the
chassis *away* from the target it is reaching for.

    python tests/hardware/test_03_manual_base.py --host <robot-ip>          # on blocks
    python tests/hardware/test_03_manual_base.py --host <robot-ip> --floor  # on the floor

set_base_velocity takes [lateral, forward, yaw] — the ordering base.py's path
follower uses. That is the convention under test.
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

SLOW = 0.08          # m/s   — deliberately crawling
SLOW_YAW = 0.30      # rad/s
PULSE_S = 1.5        # how long each demo motion runs


def _drive(vec, seconds: float) -> None:
    """Hold a base velocity for `seconds`, then stop.

    Re-sent every 100 ms on purpose: base_motor's control loop disables the
    drive motors if no command arrives for 2.5 x POLICY_CONTROL_PERIOD (250 ms).
    That watchdog is the reason a crashed client cannot run the robot away.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        CLIENT.call("set_base_velocity", list(vec))
        time.sleep(0.1)
    CLIENT.call("set_base_velocity", [0.0, 0.0, 0.0])


def test_preconditions():
    print("\npreconditions")
    if ARGS.floor:
        precondition(
            "The robot is on the FLOOR with at least 2 m clear in every direction.",
            "The path is free of cables, feet and cliff edges (no stairs, no dock).",
            "You can reach the physical e-stop / power cut.",
            "Arms are tucked or well clear of anything they could strike.",
        )
    else:
        precondition(
            "The robot is ON BLOCKS with all four wheels off the ground.",
            "The blocks are stable and the chassis cannot tip or walk off them.",
            "Nothing is touching the wheels, and no fingers are near them.",
            "You can reach the physical e-stop / power cut.",
        )
    check("operator confirmed the setup", True)


def test_watchdog():
    print("\ncommand watchdog")
    info("base_motor disables the drive motors if no command arrives for ~250 ms. "
         "That is what stops the robot if this test process dies mid-motion.")
    confirm("Command a slow forward velocity ONCE, then send nothing.")
    with guard(CLIENT):
        countdown(2, "single velocity command")
        CLIENT.call("set_base_velocity", [0.0, SLOW, 0.0])
        time.sleep(2.0)
        CLIENT.call("set_base_velocity", [0.0, 0.0, 0.0])
    stopped = ask_yes_no("Did the wheels stop on their own well before 2 s elapsed?")
    if stopped is not None:
        check("watchdog halts the base without a stop command", stopped)
        if not stopped:
            info("The wheels kept turning on a single command. Check "
                 "POLICY_CONTROL_FREQ and the watchdog in base_motor.control_loop — "
                 "without it, a crashed client leaves the robot driving.")


def test_forward_axis():
    print("\nforward axis")
    confirm(f"Drive FORWARD at {SLOW:.2f} m/s for {PULSE_S:.1f} s.")
    with guard(CLIENT):
        countdown(3, "forward")
        _drive([0.0, SLOW, 0.0], PULSE_S)
    got = ask_yes_no("Did the robot move FORWARD (the direction the arms face)?")
    if got is not None:
        check("positive index 1 drives forward", got)
        if not got:
            info("Flip forward_sign in BaseAxisMap (robot/wholebody_control.py). "
                 "Do NOT compensate elsewhere — the solver and the path follower "
                 "both assume this convention.")


def test_lateral_axis():
    print("\nlateral axis")
    confirm(f"Strafe LEFT at {SLOW:.2f} m/s for {PULSE_S:.1f} s.")
    with guard(CLIENT):
        countdown(3, "strafe left")
        _drive([SLOW, 0.0, 0.0], PULSE_S)
    got = ask_yes_no("Did the robot strafe to its LEFT (no rotation)?")
    if got is not None:
        check("positive index 0 strafes left", got)
        if not got:
            info("Flip lateral_sign in BaseAxisMap. If it rotated instead of "
                 "strafing, a steer module is mis-ordered — check CAN_IDS_ROT "
                 "and ROTATION_OFFSETS in base_motor.py.")


def test_yaw_axis():
    print("\nyaw axis")
    confirm(f"Rotate COUNTER-CLOCKWISE at {SLOW_YAW:.2f} rad/s for {PULSE_S:.1f} s.")
    with guard(CLIENT):
        countdown(3, "rotate CCW")
        _drive([0.0, 0.0, SLOW_YAW], PULSE_S)
    got = ask_yes_no("Did the robot rotate COUNTER-CLOCKWISE (seen from above)?")
    if got is not None:
        check("positive index 2 rotates counter-clockwise", got)
        if not got:
            info("Flip yaw_sign in BaseAxisMap. A wrong yaw sign makes the EKF "
                 "diverge as soon as the robot turns.")


def test_pure_rotation_is_pure():
    print("\nrotation is about the robot centre")
    confirm(f"Rotate in place at {SLOW_YAW:.2f} rad/s for {PULSE_S * 2:.1f} s.")
    with guard(CLIENT):
        countdown(3, "rotate in place")
        _drive([0.0, 0.0, SLOW_YAW], PULSE_S * 2)
    got = ask_yes_no("Did it spin about its own centre, without drifting sideways?")
    if got is not None:
        check("rotation does not translate", got)
        if not got:
            info("Drift during pure rotation means LENGTH/WIDTH in base_motor.py "
                 "do not match the real module positions. That same geometry is "
                 "duplicated (calibrated) in nav/odometry/swerve_odom.py.")


def test_encoders_track_commanded_motion():
    print("\nencoders follow commanded motion")
    before = CLIENT.call("get_base_encoders")["drive_counts"]
    confirm(f"Drive forward at {SLOW:.2f} m/s for {PULSE_S:.1f} s while watching encoders.")
    with guard(CLIENT):
        countdown(2, "forward")
        _drive([0.0, SLOW, 0.0], PULSE_S)
    time.sleep(0.3)
    after = CLIENT.call("get_base_encoders")["drive_counts"]
    deltas = [a - b for a, b in zip(after, before)]
    names = ["FL", "FR", "RR", "RL"]
    info("delta counts: " + ", ".join(f"{n}={d:+.1f}" for n, d in zip(names, deltas)))

    moved = [abs(d) > 0.5 for d in deltas]
    check("all four drive encoders registered motion", all(moved),
          ", ".join(n for n, m in zip(names, moved) if not m) + " did not move"
          if not all(moved) else "")

    if all(moved):
        mags = [abs(d) for d in deltas]
        spread = (max(mags) - min(mags)) / max(mags)
        check("all four wheels turned by a similar amount", spread < 0.25,
              f"{spread * 100:.0f}% spread")
        if spread >= 0.25:
            info("One wheel is slipping, mis-steered, or has a bad encoder. "
                 "Swerve odometry least-squares will absorb this as fake rotation.")


def test_stop_command():
    print("\nexplicit stop")
    confirm("Drive forward, then send an explicit zero velocity.")
    with guard(CLIENT):
        countdown(2, "forward then stop")
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            CLIENT.call("set_base_velocity", [0.0, SLOW, 0.0])
            time.sleep(0.1)
        CLIENT.call("set_base_velocity", [0.0, 0.0, 0.0])
        time.sleep(0.5)
    vel = CLIENT.call("get_cmd_vel")
    cmd = vel[0] if isinstance(vel, (list, tuple)) else None
    if cmd is not None:
        check("commanded velocity reads back as zero",
              max(abs(v) for v in cmd) < 1e-9,
              ", ".join(f"{v:+.3f}" for v in cmd))
    got = ask_yes_no("Did the robot stop promptly and stay stopped?")
    if got is not None:
        check("explicit stop works", got)


def test_manual_overrides_wholebody():
    print("\nmanual command suspends whole-body base authority")
    state = CLIENT.call("get_state")
    if not state.get("base_motion_enabled"):
        info("base motion is disabled on the node, so there is nothing to override")
        check("whole-body base motion enabled for this check", False,
              "skipped — enable it with toggle_base_motion(True)")
        return
    info("Any direct base command suspends the solver's authority for "
         "manual_override_timeout_s (0.5 s by default), so the two controllers "
         "never fight over the wheels.")
    CLIENT.call("set_base_velocity", [0.0, 0.0, 0.0])
    time.sleep(0.1)
    during = CLIENT.call("get_state").get("base_command") or [0, 0, 0]
    check("solver is not driving the wheels right after a manual command",
          max(abs(v) for v in during) < 1e-6,
          ", ".join(f"{v:+.4f}" for v in during))


def main() -> int:
    global CLIENT, ARGS
    ARGS = parse_args(
        __doc__,
        extra=lambda p: p.add_argument(
            "--floor", action="store_true",
            help="robot is on the floor and free to move (default assumes blocks)"),
    )
    banner("STAGE 1 — MANUAL BASE CONTROL",
           "*** THE WHEELS TURN. " +
           ("ROBOT ON THE FLOOR." if "--floor" in sys.argv else "ROBOT ON BLOCKS.") +
           " Ctrl-C stops it. ***")
    CLIENT = connect(ARGS)
    try:
        return run(
            test_preconditions,
            test_watchdog,
            test_forward_axis,
            test_lateral_axis,
            test_yaw_axis,
            test_pure_rotation_is_pure,
            test_encoders_track_commanded_motion,
            test_stop_command,
            test_manual_overrides_wholebody,
        )
    finally:
        CLIENT.halt()


if __name__ == "__main__":
    raise SystemExit(main())
