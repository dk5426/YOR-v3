#!/usr/bin/env python3
"""test_06_slam_pose.py — the Odin pose is sane, and slam_yaw_sign is calibrated.

⚠️  THE ROBOT DRIVES. Floor, 3 m clear. Needs odin_pub_node running.

Two jobs:

1. Prove the SLAM pose is usable: it arrives, it is confident, it holds still
   when the robot does, and it moves the right amount when the robot drives.
2. **Determine `slam_yaw_sign`** — the one value you must calibrate before
   enabling `enable_slam_base_pose` in WholeBodyHardwareConfig. Getting it wrong
   is worse than leaving the feature off, because the base pose error then grows
   as you drive instead of staying bounded.

    python tests/hardware/test_06_slam_pose.py --host <robot-ip> --slam-host <slam-ip>
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
    Abort, TopicListener, ask_float, banner, check, confirm, connect, countdown,
    guard, info, parse_args, precondition, run,
)
from robot.topics import POSE_TOPIC, SLAM_PUB_PORT  # noqa: E402

CLIENT = None
ARGS = None
SUB = None      # a TopicListener: background thread, non-blocking reads

SLOW = 0.10
SLOW_YAW = 0.35


def _planar():
    """Latest SLAM pose as (x, z, yaw) in the SLAM world frame, or None.

    Same extraction the whole-body listener uses: planar position is (t_x, t_z)
    of the Y-up translation, yaw is about Y as atan2(-R[2,0], R[0,0]).
    """
    msg = SUB.latest(max_age_s=1.0) if SUB is not None else None
    if msg is None or len(msg) < 7:
        return None
    if len(msg) > 19 and float(msg[19]) < 10.0:
        return None
    qx, qy, qz, qw = (float(msg[i]) for i in range(4))
    tx, tz = float(msg[4]), float(msg[6])
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r20 = 2.0 * (qx * qz - qw * qy)
    return np.array([tx, tz, math.atan2(-r20, r00)], dtype=float)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _drive(vec, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        CLIENT.call("set_base_velocity", list(vec))
        time.sleep(0.1)
    CLIENT.call("set_base_velocity", [0.0, 0.0, 0.0])
    time.sleep(0.7)


def test_preconditions():
    print("\npreconditions")
    precondition(
        "odin_pub_node is running on the SLAM box (bash nav.sh, or the node alone).",
        "The Odin is BOLTED to the robot in its final position.",
        "T_cam_to_base in config/odin.yaml has been set for that mounting.",
        "The robot is on the FLOOR with 3 m clear ahead and 1.5 m each side.",
        "The room is lit and has visual texture — bare white walls will not track.",
        "You can reach the physical e-stop / power cut.",
    )
    check("operator confirmed the setup", True)


def test_pose_arrives():
    print("\npose stream")
    global SUB
    # Raises Abort with a clear message if nothing is listening on the port,
    # rather than parking forever inside a blocking pull-mode read.
    SUB = TopicListener(ARGS.slam_host, SLAM_PUB_PORT, POSE_TOPIC)
    if SUB.wait_for(timeout_s=10.0) is None:
        raise Abort(
            f"the publisher port is open but no message arrived on {POSE_TOPIC} "
            f"at {ARGS.slam_host}:{SLAM_PUB_PORT} in 10 s."
        )
    pose = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and pose is None:
        pose = _planar()
        time.sleep(0.1)
    if pose is None:
        raise Abort(
            f"messages are arriving on {POSE_TOPIC} but none are confident "
            "(element 19 < 10). The Odin is not tracking — it needs light and "
            "visual texture."
        )
    check("a confident SLAM pose is arriving", True,
          f"x={pose[0]:+.3f} z={pose[1]:+.3f} yaw={math.degrees(pose[2]):+.1f} deg")


def test_pose_rate_and_noise():
    print("\npose rate and noise while stationary")
    samples, stamps = [], []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        p = _planar()
        if p is not None:
            samples.append(p)
            stamps.append(time.monotonic())
        time.sleep(0.02)

    check("pose samples collected", len(samples) > 20, f"{len(samples)} in 3 s")
    if len(samples) < 5:
        return

    arr = np.asarray(samples)
    spread = float(math.hypot(arr[:, 0].ptp(), arr[:, 1].ptp()))
    yaw_spread = math.degrees(float(np.ptp(np.unwrap(arr[:, 2]))))
    check("stationary position noise under 2 cm", spread < 0.02, f"{spread * 1000:.1f} mm")
    check("stationary yaw noise under 1 degree", yaw_spread < 1.0, f"{yaw_spread:.2f} deg")
    if spread >= 0.02 or yaw_spread >= 1.0:
        info("Noisy VIO. The EKF inflates its measurement noise with confidence, "
             "but this much jitter at rest will still show up in the map.")


def test_translation_scale():
    print("\ntranslation scale")
    before = _planar()
    confirm(f"Drive FORWARD at {SLOW:.2f} m/s for 6 s (~{SLOW * 6:.1f} m).")
    with guard(CLIENT):
        countdown(3, "forward run")
        _drive([0.0, SLOW, 0.0], 6.0)
    after = _planar()
    if before is None or after is None:
        check("pose available across the run", False, "tracking was lost")
        return

    moved = float(math.hypot(after[0] - before[0], after[1] - before[1]))
    info(f"SLAM says {moved:.3f} m")
    check("SLAM registered the motion", moved > 0.10, f"{moved:.3f} m")

    measured = ask_float("Measure the actual distance travelled, in metres")
    if measured is not None and measured > 0:
        err_pct = (moved / measured - 1.0) * 100.0
        check("SLAM distance within 5% of measured", abs(err_pct) < 5.0,
              f"SLAM {moved:.3f} m vs measured {measured:.3f} m ({err_pct:+.1f}%)")
        if abs(err_pct) >= 5.0:
            info("Scale error in VIO usually means the Odin's own calibration, "
                 "not anything in this repo. Re-run its calibration.")


def test_yaw_sign_calibration():
    print("\nslam_yaw_sign calibration  <-- the value you came here for")
    info("The base convention (test_03) is: positive yaw command = COUNTER-CLOCKWISE.")
    info("We rotate CCW and watch which way the SLAM yaw goes.")

    before = _planar()
    confirm(f"Rotate COUNTER-CLOCKWISE at {SLOW_YAW:.2f} rad/s for 4 s.")
    with guard(CLIENT):
        countdown(3, "CCW rotation")
        _drive([0.0, 0.0, SLOW_YAW], 4.0)
    after = _planar()

    if before is None or after is None:
        check("pose available across the rotation", False, "tracking was lost")
        return

    d_yaw = _wrap(after[2] - before[2])
    d_deg = math.degrees(d_yaw)
    check("SLAM registered the rotation", abs(d_deg) > 10.0, f"{d_deg:+.1f} deg")
    if abs(d_deg) <= 10.0:
        return

    sign = +1.0 if d_yaw > 0 else -1.0
    print()
    print("  " + "=" * 62)
    print(f"    A counter-clockwise rotation moved the SLAM yaw by {d_deg:+.1f} deg")
    print(f"    ==>  set  slam_yaw_sign = {sign:+.1f}")
    print("         in WholeBodyHardwareConfig (robot/wholebody_control.py),")
    print("         alongside enable_slam_base_pose=True")
    print("  " + "=" * 62)
    check("slam_yaw_sign determined", True, f"{sign:+.1f}")

    drift = float(math.hypot(after[0] - before[0], after[1] - before[1]))
    check("pure rotation did not translate much in SLAM", drift < 0.15,
          f"{drift * 100:.1f} cm")
    if drift >= 0.15:
        info("Large translation during a pure spin means T_cam_to_base is wrong: "
             "the Odin is offset from the rotation centre and the extrinsic is "
             "not accounting for it. Fix config/odin.yaml before trusting the pose.")


def test_pose_survives_a_loop():
    print("\nrepeatability (drive out and back)")
    start = _planar()
    confirm("Drive forward 4 s then backward 4 s; the pose should return near start.")
    with guard(CLIENT):
        countdown(3, "out and back")
        _drive([0.0, SLOW, 0.0], 4.0)
        _drive([0.0, -SLOW, 0.0], 4.0)
    end = _planar()
    if start is None or end is None:
        check("pose available across the loop", False, "tracking was lost")
        return
    residual = float(math.hypot(end[0] - start[0], end[1] - start[1]))
    check("SLAM pose returns within 15 cm of its start", residual < 0.15,
          f"{residual * 100:.1f} cm")
    info("Some residual is real robot error, not just VIO. Compare with the "
         "closed-loop result from test_05 to tell them apart.")


def main() -> int:
    global CLIENT, ARGS
    ARGS = parse_args(
        __doc__,
        extra=lambda p: p.add_argument(
            "--slam-host", default="192.168.1.11",
            help="host running odin_pub_node (default 192.168.1.11)"),
    )
    banner("STAGE 2 — SLAM POSE",
           "*** THE ROBOT DRIVES ON THE FLOOR. Ctrl-C stops it. ***")
    CLIENT = connect(ARGS)
    try:
        return run(
            test_preconditions,
            test_pose_arrives,
            test_pose_rate_and_noise,
            test_translation_scale,
            test_yaw_sign_calibration,
            test_pose_survives_a_loop,
        )
    finally:
        CLIENT.halt()
        if SUB is not None:
            try:
                SUB.stop()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
