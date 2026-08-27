#!/usr/bin/env python3
"""test_09_axis_match.py — which command axis moves the robot which way, per SLAM.

⚠️  THE ROBOT DRIVES. Floor, 2 m clear all round. Needs odin_pub_node running.

Answers one question: **when I command an axis, what does the SLAM pose do?**

It exists because the codebase disagrees with itself about the order of the
3-vector `set_base_velocity` takes:

  * `BaseAxisMap` (robot/wholebody_control.py) and robot/teleop/joystick.py
    treat element 0 as forward, element 1 as left.
  * tests/hardware/test_03_manual_base.py and test_06_slam_pose.py drive
    forward with `[0.0, SLOW, 0.0]` — element 1.

Both cannot be right, and nothing measured it against SLAM. So this probes each
element separately rather than assuming either:

    probe A   +element 0     nominally 10 cm
    probe B   +element 1     nominally 10 cm
    probe C   +element 2     nominally 45 deg

For each it records the SLAM displacement, expresses it in the robot's *own*
heading frame at the moment the probe started, and asks you what you actually
saw. Measurement alone cannot fix a frame's handedness — the (t_x, t_z) plane
is left-handed and the IK plane is not — so the eyeball check is not optional.

    python tests/hardware/test_09_axis_match.py --host <robot-ip> --slam-host <slam-ip>

Nothing here is written back to any config. It prints what to set.
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
    Abort, HwClient, TopicListener, ask_yes_no, banner, check, confirm,
    countdown, guard, info, parse_args, precondition, run,
)
from robot.topics import POSE_TOPIC, SLAM_PUB_PORT  # noqa: E402

CLIENT = None
ARGS = None
SUB = None

RESULTS: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# SLAM pose
# ─────────────────────────────────────────────────────────────────────────────

def _pose():
    """Latest SLAM sample as a dict, or None when unusable.

    `xz`      planar position (t_x, t_z) — the same two axes the whole-body
              listener uses, skipping t_y because the Odin frame is Y-up.
    `yaw`     about Y, atan2(-R[2,0], R[0,0]).
    `heading` the body +X axis projected into that plane, (R[0,0], R[2,0]),
              taken straight from the quaternion. This is the honest one: it
              is a direction in the plane, not an angle in a convention, so a
              handedness error cannot hide inside it.
    """
    msg = SUB.latest(max_age_s=1.0) if SUB is not None else None
    if msg is None or len(msg) < 7:
        return None
    if len(msg) > 19 and float(msg[19]) < 10.0:
        return None
    qx, qy, qz, qw = (float(msg[i]) for i in range(4))
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r20 = 2.0 * (qx * qz - qw * qy)
    return {
        "xz": np.array([float(msg[4]), float(msg[6])], dtype=float),
        "yaw": math.atan2(-r20, r00),
        "heading": np.array([r00, r20], dtype=float),
        "conf": float(msg[19]) if len(msg) > 19 else 100.0,
    }


def _pose_or_abort(what: str):
    p = _pose()
    if p is None:
        raise Abort(f"no usable SLAM pose while {what} — is odin_pub_node up "
                    f"and tracking? (confidence gates below 10)")
    return p


def _connect(args) -> HwClient:
    """Open the RPC link, tolerating a node that has no whole-body controller.

    `_hw.connect` insists `get_state()` come back non-empty, but that call is
    `@require_wholebody` in effect: it returns `{}` whenever `yor.wholebody`
    is None, which is a perfectly normal way to run the node and is all this
    test needs -- it only ever calls `set_base_velocity`.

    `get_cmd_vel` is the honest liveness probe instead. It is gated on
    initialisation but not on whole-body, so `None` means exactly one thing:
    `init()` has not run, and every drive command would be silently dropped
    by the same gate.
    """
    print(f"connecting to robot/yor.py at {args.host}:{args.port} ...")
    client = HwClient(args.host, args.port, args.timeout)
    try:
        alive = client.call("get_cmd_vel")
    except Exception as exc:
        raise Abort(str(exc))
    if alive is None:
        raise Abort(
            "the node is answering but has not been initialised -- get_cmd_vel() "
            "returned None, which is what @require_initialization does before "
            "init() runs. set_base_velocity() would be dropped the same way, so "
            "the robot would never move. Start the node so it initialises (check "
            "its console for the init errors), then re-run."
        )
    state = client.call("get_state") or {}
    if state:
        print(f"connected. solver={'ok' if state.get('solved') else 'not solving'}, "
              f"base_motion={'ON' if state.get('base_motion_enabled') else 'off'}")
    else:
        print("connected. whole-body controller not running -- fine for this "
              "test, which drives the base directly.")
    return client


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _signed_angle(a, b) -> float:
    """Signed angle a→b in the plane, CCW positive in raw (x, z) coordinates."""
    return math.atan2(a[0] * b[1] - a[1] * b[0], float(np.dot(a, b)))


# ─────────────────────────────────────────────────────────────────────────────
# Driving
# ─────────────────────────────────────────────────────────────────────────────

def _drive(vec, seconds: float) -> None:
    """Hold a velocity for `seconds`, then stop.

    Re-sent at 10 Hz because the base watchdog stops the wheels if commands
    stop arriving — the same reason test_03 does it this way.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        CLIENT.call("set_base_velocity", list(vec))
        time.sleep(0.1)
    CLIENT.call("set_base_velocity", [0.0, 0.0, 0.0])
    time.sleep(1.0)      # let the smoothing ramp down and the pose settle


def _probe(key: str, label: str, index: int, magnitude: float, seconds: float):
    """One axis probe: measure the SLAM pose either side of a timed pulse."""
    vec = [0.0, 0.0, 0.0]
    vec[index] = magnitude

    before = _pose_or_abort(f"starting {label}")
    countdown(3, f"{label}  (command element {index} = {magnitude:+.2f})")
    _drive(vec, seconds)
    after = _pose_or_abort(f"finishing {label}")

    d_xz = after["xz"] - before["xz"]
    dist = float(np.linalg.norm(d_xz))
    d_yaw = _wrap(after["yaw"] - before["yaw"])
    # Angle of travel measured against the robot's own heading at the start,
    # so the answer does not depend on where the SLAM origin happens to be.
    bearing = _signed_angle(before["heading"], d_xz) if dist > 1e-4 else float("nan")

    RESULTS[key] = {
        "label": label, "index": index, "vec": vec,
        "d_xz": d_xz, "dist": dist, "d_yaw": d_yaw, "bearing": bearing,
        "conf": after["conf"],
    }

    print(f"     SLAM Δ(t_x, t_z) = ({d_xz[0]:+.4f}, {d_xz[1]:+.4f}) m   "
          f"|Δ| = {dist:.4f} m")
    print(f"     SLAM Δyaw        = {math.degrees(d_yaw):+7.2f} deg")
    if dist > 1e-4:
        print(f"     travel vs heading = {math.degrees(bearing):+7.2f} deg "
              f"(0 = along its own nose, in raw (x,z) coordinates)")
    return RESULTS[key]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_preconditions():
    print("\npreconditions")
    precondition(
        "odin_pub_node is running on the SLAM box and is TRACKING (confidence > 10).",
        "The robot is on the floor with 2 m clear in every direction.",
        "Whole-body base motion is OFF, or the solver will fight these commands "
        "once each manual override lapses (0.5 s after the last one).",
        "You can see the robot and reach the e-stop.",
    )
    state = CLIENT.call("get_state") or {}
    if state.get("base_motion_enabled"):
        info("base_motion is ENABLED — each probe suspends it while commands "
             "flow, but it resumes between probes. Prefer --no-base-motion.")
    check("SLAM pose is arriving", _pose() is not None)


def test_stationary_noise():
    """A pose that wanders while the robot is still would fake every result."""
    print("\nstationary noise floor")
    p0 = _pose_or_abort("measuring the noise floor")
    time.sleep(2.0)
    p1 = _pose_or_abort("measuring the noise floor")
    drift = float(np.linalg.norm(p1["xz"] - p0["xz"]))
    dyaw = abs(math.degrees(_wrap(p1["yaw"] - p0["yaw"])))
    check("position is steady while stopped", drift < 0.02, f"{drift*100:.1f} cm / 2 s")
    check("yaw is steady while stopped", dyaw < 2.0, f"{dyaw:.2f} deg / 2 s")
    info(f"anything below ~{max(drift, 0.005)*100:.1f} cm in the probes is noise, not motion")


def test_probe_element_0():
    print("\nprobe A — command element 0")
    confirm("Drive element 0 positive (nominally 10 cm).")
    with guard(CLIENT):
        r = _probe("A", "element 0 positive", 0, ARGS.speed, ARGS.distance / ARGS.speed)
    check("the robot actually moved", r["dist"] > 0.02, f"{r['dist']*100:.1f} cm")
    r["saw_forward"] = ask_yes_no("Did the robot move FORWARD (nose first)?")
    if r["saw_forward"] is False:
        r["saw_left"] = ask_yes_no("Did it move to its own LEFT?")


def test_probe_element_1():
    print("\nprobe B — command element 1")
    confirm("Drive element 1 positive (nominally 10 cm).")
    with guard(CLIENT):
        r = _probe("B", "element 1 positive", 1, ARGS.speed, ARGS.distance / ARGS.speed)
    check("the robot actually moved", r["dist"] > 0.02, f"{r['dist']*100:.1f} cm")
    r["saw_forward"] = ask_yes_no("Did the robot move FORWARD (nose first)?")
    if r["saw_forward"] is False:
        r["saw_left"] = ask_yes_no("Did it move to its own LEFT?")


def test_probe_yaw():
    print("\nprobe C — command element 2 (yaw)")
    confirm("Rotate element 2 positive (nominally 45 deg).")
    with guard(CLIENT):
        r = _probe("C", "element 2 positive", 2, ARGS.yaw_speed,
                   math.radians(ARGS.yaw) / ARGS.yaw_speed)
    check("the robot actually rotated", abs(math.degrees(r["d_yaw"])) > 10.0,
          f"{math.degrees(r['d_yaw']):+.1f} deg")
    check("it barely translated while spinning", r["dist"] < 0.15,
          f"{r['dist']*100:.1f} cm of drift")
    r["saw_ccw"] = ask_yes_no("Viewed FROM ABOVE, did it rotate COUNTER-CLOCKWISE?")


def test_verdict():
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    a, b, c = RESULTS.get("A"), RESULTS.get("B"), RESULTS.get("C")
    if not (a and b and c):
        info("not all probes ran — no verdict")
        return

    # ── which element is forward ────────────────────────────────────────────
    print("\n  command axis order")
    for r in (a, b):
        bear = abs(math.degrees(r["bearing"]))
        along = "along the nose" if bear < 45 or bear > 135 else "across the nose"
        print(f"    element {r['index']}: moved {r['dist']*100:5.1f} cm, "
              f"{math.degrees(r['bearing']):+7.1f} deg off the nose  ({along})")

    fwd_idx = None
    if a.get("saw_forward") and not b.get("saw_forward"):
        fwd_idx = 0
    elif b.get("saw_forward") and not a.get("saw_forward"):
        fwd_idx = 1
    if fwd_idx is not None:
        lat_idx = 1 - fwd_idx
        lat = RESULTS["A" if lat_idx == 0 else "B"]
        side = ("LEFT" if lat.get("saw_left") else
                "RIGHT" if lat.get("saw_left") is False else "?")
        print(f"\n    ==>  element {fwd_idx} is FORWARD, element {lat_idx} is lateral "
              f"(positive = {side})")
        if fwd_idx == 0:
            print("         matches BaseAxisMap / joystick.py.")
            print("         tests/hardware/test_03 and test_06 are WRONG "
                  "(they drive forward with element 1).")
        else:
            print("         matches test_03 / test_06.")
            print("         BaseAxisMap.forward_index=0 is WRONG — the whole-body "
                  "solver's forward is going out as sideways.")
    else:
        print("\n    ==>  inconclusive — answer the forward/left questions to resolve it")

    # ── handedness of the SLAM planar frame ─────────────────────────────────
    print("\n  SLAM frame handedness")
    print(f"    reported Δyaw    {math.degrees(c['d_yaw']):+7.2f} deg")
    if c.get("saw_ccw") is None:
        print("    ==>  inconclusive — the rotation direction was not confirmed")
    else:
        ccw = bool(c["saw_ccw"])
        yaw_pos = c["d_yaw"] > 0
        print(f"    you saw           {'counter-clockwise' if ccw else 'clockwise'} "
              f"(from above)")
        if ccw == yaw_pos:
            print("    ==>  reported yaw increases CCW: the (t_x,t_z) plane reads "
                  "RIGHT-handed here")
            print("         set slam_yaw_sign = -1   (--slam-yaw-sign=-1)")
        else:
            print("    ==>  reported yaw increases CW: the (t_x,t_z) plane is "
                  "LEFT-handed, as expected")
            print("         set slam_yaw_sign = +1   (--slam-yaw-sign=+1)")

    # ── the coupling bug ────────────────────────────────────────────────────
    print("\n  ⚠  SlamBaseFrame._reflect ties the position reflection to the same")
    print("     flag that negates yaw, so BOTH settings produce a mirrored frame")
    print("     (path turn and reported yaw come out with opposite signs). The")
    print("     sign above only takes effect once _reflect is decoupled:")
    print("         return (sx, -sy) if self.yaw_sign >= 0 else (sx, sy)")
    print()


def main() -> int:
    global CLIENT, ARGS, SUB

    def extra(p):
        p.add_argument("--slam-host", default="192.168.1.11",
                       help="host running odin_pub_node (default 192.168.1.11)")
        p.add_argument("--distance", type=float, default=0.10,
                       help="translation probe distance in m (default 0.10)")
        p.add_argument("--speed", type=float, default=0.08,
                       help="translation probe speed in m/s (default 0.08)")
        p.add_argument("--yaw", type=float, default=45.0,
                       help="yaw probe angle in degrees (default 45)")
        p.add_argument("--yaw-speed", type=float, default=0.35,
                       help="yaw probe rate in rad/s (default 0.35)")

    ARGS = parse_args(__doc__, extra=extra)
    banner("STAGE 9 — COMMAND AXIS vs SLAM",
           "*** THE ROBOT DRIVES ON THE FLOOR. Ctrl-C stops it. ***",
           f"probes: {ARGS.distance*100:.0f} cm on element 0, "
           f"{ARGS.distance*100:.0f} cm on element 1, {ARGS.yaw:.0f} deg yaw")
    CLIENT = _connect(ARGS)
    SUB = TopicListener(ARGS.slam_host, SLAM_PUB_PORT, POSE_TOPIC)
    try:
        return run(
            test_preconditions,
            test_stationary_noise,
            test_probe_element_0,
            test_probe_element_1,
            test_probe_yaw,
            test_verdict,
        )
    finally:
        CLIENT.halt()
        try:
            SUB.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
