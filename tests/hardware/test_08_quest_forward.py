#!/usr/bin/env python3
"""test_08_quest_forward.py — replay one synthetic Quest forward motion.

This does not connect to a headset. It feeds a pure forward controller
translation through the same OculusSource transform used by
robot/teleop/wholebody_teleop.py, then sends the resulting end-effector target
to a running hardware YOR node.

Only the selected arm(s) may move. The test:

* fixes the base in IK and disables base command dispatch;
* holds the lift at its measured starting height;
* preserves end-effector orientation;
* ramps 10 cm forward, holds briefly, then ramps back to the starting pose.

Quest/Unity tracking reports physical forward as raw +Z. oculus_msgs converts
that left-handed pose to right-handed -Z before OculusSource sees it. With the
normal 270-degree room correction, that must become robot/model -Y (forward).

Run robot/yor.py with arms enabled first:

    python robot/yor.py --no-flash-base-pid

Then, on the robot:

    python tests/hardware/test_08_quest_forward.py --host localhost
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mink  # noqa: E402
import numpy as np  # noqa: E402

from robot.teleop.wholebody_teleop import OculusSource  # noqa: E402
from _hw import (  # noqa: E402
    Abort,
    ask_yes_no,
    banner,
    check,
    confirm,
    connect,
    countdown,
    guard,
    parse_args,
    precondition,
    run,
)


CLIENT = None
ARGS = None
RATE_HZ = 36.0  # same target rate as wholebody_teleop.py


def _extra_args(parser) -> None:
    parser.add_argument(
        "--arm",
        choices=("left", "right", "both"),
        default="both",
        help="arm target(s) to move (default: both)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=0.10,
        help="forward travel in metres; limited to 0.005-0.10 (default: 0.10)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="seconds for each outbound/return ramp (default: 1.0)",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=1.0,
        help="seconds to hold the forward target (default: 1.0)",
    )
    parser.add_argument(
        "--oculus-yaw-correction",
        type=float,
        default=270.0,
        help="same translation yaw correction as wholebody_teleop (default: 270)",
    )


def _pose(values) -> mink.SE3:
    return mink.SE3(np.asarray(values, dtype=float))


def _state() -> dict:
    return CLIENT.call("get_state") or {}


def _selected_sides() -> tuple[str, ...]:
    return ("left", "right") if ARGS.arm == "both" else (ARGS.arm,)


def _send_targets(targets: dict[str, mink.SE3]) -> None:
    if ARGS.arm == "both":
        CLIENT.call(
            "set_bimanual_ee_target",
            L_ee_target=targets["left"],
            R_ee_target=targets["right"],
        )
    else:
        CLIENT.call(f"set_{ARGS.arm}_ee_target", ee_target=targets[ARGS.arm])


def _quest_target(
    mapper: OculusSource,
    start_ee: mink.SE3,
    distance_m: float,
) -> mink.SE3:
    # Identity is the controller pose when its clutch is engaged. A physical
    # raw Quest/Unity +Z displacement has become -Z after oculus_msgs performs
    # its left-handed -> right-handed conversion.
    clutch_pose = mink.SE3.from_translation(np.zeros(3))
    moved_pose = mink.SE3.from_translation(np.array([0.0, 0.0, -distance_m]))
    return mapper._ee_target(clutch_pose, start_ee, moved_pose)


def test_preconditions() -> None:
    if not 0.005 <= ARGS.distance <= 0.10:
        raise Abort("--distance must be between 0.005 and 0.10 metres")
    if not 0.25 <= ARGS.duration <= 5.0:
        raise Abort("--duration must be between 0.25 and 5.0 seconds")
    if not 0.0 <= ARGS.hold <= 5.0:
        raise Abort("--hold must be between 0 and 5.0 seconds")

    precondition(
        "robot/yor.py is running WITH arms enabled.",
        "No joystick or other teleop client is connected.",
        "There is a clear 1 m sphere around both arms.",
        "A person is holding the physical e-stop and watching the arms.",
        "The base must remain still; only the selected arm(s) may move.",
    )
    check("operator confirmed the setup", True)


def test_synthetic_forward() -> None:
    # This is harmless when the loop is already live and recovers a node that
    # was left e-stopped by a previous hardware test.
    if CLIENT.call("resume_wholebody") is not True:
        raise Abort("the whole-body control loop could not be started")
    initial_state = _state()
    if not initial_state:
        raise Abort("YOR returned no whole-body state; do not start it with --no-arms")

    lift_start = float(initial_state["lift"])
    start_targets = {
        side: _pose(initial_state[f"{side}_ee_wxyz_xyz"])
        for side in ("left", "right")
    }
    mapper = OculusSource(
        host="synthetic-no-headset",
        pose_filter=False,
        yaw_correction_deg=ARGS.oculus_yaw_correction,
    )

    # Prove the configured Quest mapping is a pure robot-forward translation
    # before sending any motion command.
    probe = _quest_target(mapper, start_targets["left"], ARGS.distance)
    mapped_delta = probe.translation() - start_targets["left"].translation()
    print(
        "  ....  synthetic Quest raw +Z maps to robot delta "
        f"[{mapped_delta[0]:+.4f}, {mapped_delta[1]:+.4f}, {mapped_delta[2]:+.4f}] m"
    )
    pure_forward = (
        abs(mapped_delta[0]) < 1e-6
        and mapped_delta[1] < 0.0
        and abs(mapped_delta[2]) < 1e-6
    )
    check("Quest forward maps only to robot/model -Y", pure_forward)
    if not pure_forward:
        raise Abort(
            "the configured yaw correction does not map Quest forward to robot "
            "forward; no actuator command was sent"
        )

    fixed = CLIENT.call("toggle_fix_base", True)
    base_motion = CLIENT.call("toggle_base_motion", False)
    CLIENT.call("toggle_collision_avoidance", True)
    CLIENT.call("set_lift_target", lift_start)
    check("base fixed in IK", fixed is True, f"fix_base={fixed}")
    check("base dispatch disabled", base_motion is False, f"enabled={base_motion}")

    sides = _selected_sides()
    distance_cm = ARGS.distance * 100.0
    confirm(
        f"Move {ARGS.arm.upper()} arm target(s) {distance_cm:.1f} cm straight "
        "forward and back. Base and lift must stay fixed."
    )

    steps = max(2, int(round(ARGS.duration * RATE_HZ)))

    def stream(fractions) -> None:
        for fraction in fractions:
            distance = ARGS.distance * float(fraction)
            targets = {
                side: _quest_target(mapper, start_targets[side], distance)
                for side in sides
            }
            _send_targets(targets)
            # Reassert the measured starting height, exactly as a fixed lift
            # target rather than allowing controller input to change it.
            CLIENT.call("set_lift_target", lift_start)
            time.sleep(1.0 / RATE_HZ)

    completed = False
    with guard(CLIENT, estop=True):
        countdown(3, "synthetic Quest forward motion")
        stream(np.linspace(0.0, 1.0, steps + 1))
        time.sleep(ARGS.hold)

        forward_state = _state()
        base_cmd = np.asarray(forward_state.get("base_command", [np.inf] * 3))
        lift_now = float(forward_state.get("lift", np.nan))
        check("base command stayed zero", np.max(np.abs(base_cmd)) < 1e-6,
              np.array2string(base_cmd, precision=5))
        check("lift stayed within 5 mm", abs(lift_now - lift_start) <= 0.005,
              f"start={lift_start:.4f}, now={lift_now:.4f}")
        for side in sides:
            actual = np.asarray(forward_state[f"{side}_ee_wxyz_xyz"], dtype=float)[4:7]
            delta = actual - start_targets[side].translation()
            check(
                f"{side} arm moved primarily forward",
                delta[1] < -0.5 * ARGS.distance
                and abs(delta[0]) < 0.02
                and abs(delta[2]) < 0.02,
                np.array2string(delta, precision=4),
            )

        stream(np.linspace(1.0, 0.0, steps + 1))
        time.sleep(0.5)
        completed = True

    # guard() freezes everything on exit. Resume only after a normal outbound
    # and return ramp, and leave both software base locks engaged.
    if completed:
        CLIENT.call("resume_wholebody")
        CLIENT.call("toggle_fix_base", True)
        CLIENT.call("toggle_base_motion", False)
        CLIENT.call("set_lift_target", lift_start)

    observed = ask_yes_no("Did the selected arm(s) move straight forward and back?")
    if observed is not None:
        check("operator observed forward-only motion", observed)


def main() -> int:
    global CLIENT, ARGS
    ARGS = parse_args(__doc__, extra=_extra_args)
    banner(
        "QUEST FORWARD-AXIS HARDWARE TEST",
        "*** SELECTED ARM(S) MOVE 10 CM BY DEFAULT. ***",
        "Base is fixed and disabled; lift target is held at its current height.",
        "No headset connection is used. Ctrl-C e-stops.",
    )
    CLIENT = connect(ARGS)
    return run(test_preconditions, test_synthetic_forward)


if __name__ == "__main__":
    raise SystemExit(main())
