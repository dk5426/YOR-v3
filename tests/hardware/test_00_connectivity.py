#!/usr/bin/env python3
"""test_00_connectivity.py — is everything talking? NOTHING MOVES.

The first thing to run on a cold robot. Every check here is read-only: it
proves the links exist and the subsystems answer, so that when a later test
fails you already know it is not a cabling problem.

    python tests/hardware/test_00_connectivity.py --host <robot-ip>

Safe to run at any time, with the robot on blocks or on the floor, arms in any
pose. It sends no motion commands of any kind.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hw import (  # noqa: E402
    Abort, TopicListener, banner, check, connect, info, parse_args, port_open, run,
)

CLIENT = None
ARGS = None


def test_rpc():
    print("\nRPC link to robot/yor.py")
    state = CLIENT.call("get_state")
    check("get_state returns a dict", isinstance(state, dict), type(state).__name__)
    expected = {
        "left_ee_wxyz_xyz", "right_ee_wxyz_xyz", "lift", "base_xytheta",
        "base_velocity", "fix_base", "collision_avoidance",
    }
    missing = expected - set(state or {})
    check("state carries every field the teleop client reads", not missing, str(missing))
    check("solver reports a result", "solved" in (state or {}),
          f"solved={state.get('solved')}")
    if not state.get("solved"):
        info("solver is not converging — check the robot/yor.py console")


def test_arms():
    print("\narms")
    left = CLIENT.call("get_left_joint_positions")
    right = CLIENT.call("get_right_joint_positions")
    check("left arm reports 7 joints", left is not None and len(left) == 7,
          f"{None if left is None else len(left)}")
    check("right arm reports 7 joints", right is not None and len(right) == 7,
          f"{None if right is None else len(right)}")
    for side, q in (("left", left), ("right", right)):
        if q is None:
            continue
        finite = all(v == v and abs(v) < 100 for v in q)   # v == v rejects NaN
        check(f"{side} joint values are finite and sane", finite,
              ", ".join(f"{v:+.3f}" for v in q))


def test_base_encoders():
    print("\nswerve base encoders")
    enc = CLIENT.call("get_base_encoders")
    check("get_base_encoders returns a dict", isinstance(enc, dict), type(enc).__name__)
    if not isinstance(enc, dict):
        return
    for field, n in (("steer_rad", 4), ("drive_counts", 4), ("steer_deg", 4),
                     ("drive_vel", 4)):
        val = enc.get(field)
        check(f"{field} has {n} entries", val is not None and len(val) == n,
              str(val if val is None else len(val)))
    steer = enc.get("steer_rad") or []
    check("steer angles are finite", all(v == v for v in steer),
          ", ".join(f"{v:+.3f}" for v in steer))
    counts = enc.get("drive_counts") or []
    check("drive counts are finite", all(v == v for v in counts),
          ", ".join(f"{v:.1f}" for v in counts))
    info("module order is FL, FR, RR, RL — matches CAN_IDS_DRIVE in base_motor.py")


def test_lift():
    print("\nlift controller")
    status = CLIENT.call("get_lift_status")
    if not isinstance(status, dict) or not status.get("available"):
        check("lift controller reachable", False,
              "get_lift_status says unavailable — check the USB serial cable "
              "and that pyserial is installed on the robot")
        return
    check("lift controller reachable", True)

    known = status.get("position_known")
    check("lift reports whether its position is known", known is not None,
          f"position_known={known}")
    if known is False:
        info("position NOT established — this is normal before homing. "
             "test_02_lift.py homes it.")
    height = status.get("height_m")
    if known:
        check("lift height is a sane number",
              height is not None and 0.0 - 0.01 <= height <= 0.91,
              f"{height} m")
    else:
        check("lift reports no height while its position is unknown",
              height is None, str(height))
    info(f"limits: upper={status.get('upper_limit')} lower={status.get('lower_limit')}, "
         f"motion={status.get('motion')}")


def test_slam_publisher():
    print("\nSLAM sensor publisher (optional)")
    from robot.topics import POSE_TOPIC, SLAM_PUB_PORT

    host = ARGS.slam_host
    info(f"looking for {POSE_TOPIC} on {host}:{SLAM_PUB_PORT}")

    if not port_open(host, SLAM_PUB_PORT, timeout_s=2.0):
        check("odin_pub_node is publishing", False,
              f"nothing listening on {host}:{SLAM_PUB_PORT}. Fine if you are not "
              "testing navigation yet — start it with "
              "`python -m robot.odin_pub_node` on the SLAM box.")
        return

    listener = None
    try:
        listener = TopicListener(host, SLAM_PUB_PORT, POSE_TOPIC)
        msg = listener.wait_for(timeout_s=5.0)
    except Abort as exc:
        check("odin_pub_node is publishing", False, str(exc))
        return
    finally:
        if listener is not None:
            listener.stop()

    if msg is None:
        check("odin_pub_node is publishing", False,
              "the port is open but no pose arrived in 5 s — the node is up but "
              "the Odin is probably not tracking (needs light and texture)")
        return
    check("odin_pub_node is publishing", True)
    check("pose message has the documented 20-float layout", len(msg) >= 20, str(len(msg)))
    if len(msg) >= 20:
        info(f"confidence={msg[19]:.0f}, base xyz=({msg[4]:+.2f}, {msg[5]:+.2f}, {msg[6]:+.2f})")


def main() -> int:
    global CLIENT, ARGS
    ARGS = parse_args(
        __doc__,
        extra=lambda p: p.add_argument(
            "--slam-host", default="192.168.1.11",
            help="host running odin_pub_node (default 192.168.1.11)"),
    )
    banner("STAGE 0 — CONNECTIVITY", "Read-only. Nothing moves.")
    CLIENT = connect(ARGS)
    return run(test_rpc, test_arms, test_base_encoders, test_lift, test_slam_publisher)


if __name__ == "__main__":
    raise SystemExit(main())
