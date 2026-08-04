#!/usr/bin/env python3
"""test_05_odometry.py — is swerve dead-reckoning telling the truth?

⚠️  THE ROBOT DRIVES. Floor, 3 m clear ahead. Run test_03 first.

Drives a measured distance and a measured rotation, integrates the wheel
encoders through the same SwerveOdom the EKF uses, and compares against a tape
measure. This calibrates the three constants in
robot/nav/odometry/swerve_odom.py — METERS_PER_ROTATION, LENGTH and WIDTH —
which are deliberately different from the CAD values in base_motor.py.

Those constants set how much the EKF trusts the wheels between VIO frames. If
they are wrong the fused pose drifts under exactly the conditions where you
need it most: fast motion, and VIO dropouts.

    python tests/hardware/test_05_odometry.py --host <robot-ip>
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
    ask_float, banner, check, confirm, connect, countdown, guard, info,
    parse_args, precondition, run,
)
from robot.nav.odometry.swerve_odom import (  # noqa: E402
    METERS_PER_ROTATION, LENGTH, WIDTH, SwerveOdom,
)

CLIENT = None
ARGS = None

SLOW = 0.10       # m/s
SLOW_YAW = 0.35   # rad/s


def _integrate_while_driving(vec, seconds: float) -> np.ndarray:
    """Drive, sampling encoders into a SwerveOdom. Returns [dx, dz, dtheta]."""
    odom = SwerveOdom()
    odom.reset(0.0, 0.0, 0.0)

    enc = CLIENT.call("get_base_encoders")
    last_t = float(enc.get("timestamp", time.time()))
    odom.update(np.asarray(enc["steer_rad"]), np.asarray(enc["drive_counts"]), 0.05)

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        CLIENT.call("set_base_velocity", list(vec))
        enc = CLIENT.call("get_base_encoders")
        now = float(enc.get("timestamp", time.time()))
        dt = float(np.clip(now - last_t, 1e-4, 0.5))
        last_t = now
        odom.update(np.asarray(enc["steer_rad"]), np.asarray(enc["drive_counts"]), dt)
        time.sleep(0.05)
    CLIENT.call("set_base_velocity", [0.0, 0.0, 0.0])

    # Drain the last motion after the stop command.
    for _ in range(10):
        enc = CLIENT.call("get_base_encoders")
        now = float(enc.get("timestamp", time.time()))
        dt = float(np.clip(now - last_t, 1e-4, 0.5))
        last_t = now
        odom.update(np.asarray(enc["steer_rad"]), np.asarray(enc["drive_counts"]), dt)
        time.sleep(0.05)

    return odom.get_pose()


def test_preconditions():
    print("\npreconditions")
    precondition(
        "The robot is on the FLOOR, on the surface it normally drives on.",
        "At least 3 m clear straight ahead and 1.5 m on every side.",
        "You have a tape measure and can mark the floor at the start position.",
        "test_03_manual_base.py has already passed — the axis signs are correct.",
        "You can reach the physical e-stop / power cut.",
    )
    check("operator confirmed the setup", True)


def test_constants_are_the_calibrated_ones():
    print("\nconstants under test")
    info(f"METERS_PER_ROTATION = {METERS_PER_ROTATION:.6f} m/rev")
    info(f"LENGTH = {LENGTH:.5f} m (half-wheelbase), WIDTH = {WIDTH:.5f} m (half-track)")
    info("These are odometry-calibrated, deliberately larger than the CAD values "
         "in base_motor.py. Do not reconcile them by hand — re-run this test.")
    check("constants loaded from swerve_odom.py", METERS_PER_ROTATION > 0 and LENGTH > 0)


def test_straight_line():
    print("\nstraight-line scale (calibrates METERS_PER_ROTATION)")
    confirm(f"Drive FORWARD at {SLOW:.2f} m/s for 8 s (~{SLOW * 8:.1f} m).")
    info("Mark the floor at the robot's start position now.")

    with guard(CLIENT):
        countdown(3, "forward run")
        pose = _integrate_while_driving([0.0, SLOW, 0.0], 8.0)

    odom_dist = float(math.hypot(pose[0], pose[1]))
    info(f"odometry says {odom_dist:.4f} m "
         f"(dx={pose[0]:+.4f}, dz={pose[1]:+.4f}, dtheta={math.degrees(pose[2]):+.2f} deg)")

    check("odometry integrated a plausible distance", 0.1 < odom_dist < 3.0,
          f"{odom_dist:.3f} m")
    check("a straight run produced little rotation", abs(pose[2]) < math.radians(10),
          f"{math.degrees(pose[2]):+.2f} deg")

    measured = ask_float("Measure the actual distance travelled, in metres")
    if measured is None or measured <= 0:
        return
    ratio = measured / odom_dist if odom_dist > 0 else float("nan")
    err_pct = (odom_dist / measured - 1.0) * 100.0
    check("odometry distance within 3% of measured", abs(err_pct) < 3.0,
          f"odom {odom_dist:.3f} m vs measured {measured:.3f} m ({err_pct:+.1f}%)")
    if abs(err_pct) >= 3.0:
        info(f"Scale correction: set METERS_PER_ROTATION = "
             f"{METERS_PER_ROTATION * ratio:.6f} in robot/nav/odometry/swerve_odom.py "
             f"(currently {METERS_PER_ROTATION:.6f}).")


def test_pure_rotation():
    print("\nrotation scale (calibrates LENGTH / WIDTH)")
    confirm(f"Rotate in place at {SLOW_YAW:.2f} rad/s for 5 s (~{math.degrees(SLOW_YAW * 5):.0f} deg).")
    info("Note the robot's starting heading — a floor mark or a tape line helps.")

    with guard(CLIENT):
        countdown(3, "rotation run")
        pose = _integrate_while_driving([0.0, 0.0, SLOW_YAW], 5.0)

    odom_deg = math.degrees(pose[2])
    drift = float(math.hypot(pose[0], pose[1]))
    info(f"odometry says {odom_deg:+.2f} deg, with {drift * 1000:.0f} mm of translation")

    check("odometry integrated a plausible rotation", 10.0 < abs(odom_deg) < 360.0,
          f"{odom_deg:+.1f} deg")
    check("pure rotation produced little translation", drift < 0.10,
          f"{drift * 1000:.0f} mm")
    if drift >= 0.10:
        info("Translation during pure rotation means the module positions "
             "(LENGTH/WIDTH) are wrong, or a wheel is slipping.")

    measured = ask_float("Measure the actual rotation, in degrees")
    if measured is None or abs(measured) < 1e-6:
        return
    err_pct = (abs(odom_deg) / abs(measured) - 1.0) * 100.0
    check("odometry rotation within 5% of measured", abs(err_pct) < 5.0,
          f"odom {odom_deg:+.1f} deg vs measured {measured:+.1f} deg ({err_pct:+.1f}%)")
    if abs(err_pct) >= 5.0:
        scale = abs(measured) / abs(odom_deg)
        info(f"Rotation is off by {err_pct:+.1f}%. The module radius scales it, so "
             f"try LENGTH={LENGTH * scale:.5f}, WIDTH={WIDTH * scale:.5f} "
             f"and re-run BOTH parts of this test — they interact.")


def test_return_to_start():
    print("\nclosed loop (out and back)")
    confirm("Drive forward 3 s then backward 3 s; odometry should return near zero.")
    with guard(CLIENT):
        countdown(3, "out and back")
        odom = SwerveOdom()
        odom.reset(0.0, 0.0, 0.0)
        enc = CLIENT.call("get_base_encoders")
        last_t = float(enc.get("timestamp", time.time()))
        odom.update(np.asarray(enc["steer_rad"]), np.asarray(enc["drive_counts"]), 0.05)

        for vec, secs in (([0.0, SLOW, 0.0], 3.0), ([0.0, -SLOW, 0.0], 3.0)):
            deadline = time.monotonic() + secs
            while time.monotonic() < deadline:
                CLIENT.call("set_base_velocity", list(vec))
                enc = CLIENT.call("get_base_encoders")
                now = float(enc.get("timestamp", time.time()))
                dt = float(np.clip(now - last_t, 1e-4, 0.5))
                last_t = now
                odom.update(np.asarray(enc["steer_rad"]),
                            np.asarray(enc["drive_counts"]), dt)
                time.sleep(0.05)
            CLIENT.call("set_base_velocity", [0.0, 0.0, 0.0])
            time.sleep(0.5)

    pose = odom.get_pose()
    residual = float(math.hypot(pose[0], pose[1]))
    check("odometry closes the loop within 10 cm", residual < 0.10,
          f"{residual * 100:.1f} cm residual")
    if residual >= 0.10:
        info("Asymmetric error between forward and reverse usually means steer "
             "offset error: the modules are not pointing where the FK thinks.")
    got = ask_float("How far is the robot from its starting mark now, in cm")
    if got is not None:
        check("the robot physically returned near its start", abs(got) < 15.0,
              f"{got:.1f} cm")


def main() -> int:
    global CLIENT, ARGS
    ARGS = parse_args(__doc__)
    banner("STAGE 2 — WHEEL ODOMETRY",
           "*** THE ROBOT DRIVES ON THE FLOOR. Ctrl-C stops it. ***")
    CLIENT = connect(ARGS)
    try:
        return run(
            test_preconditions,
            test_constants_are_the_calibrated_ones,
            test_straight_line,
            test_pure_rotation,
            test_return_to_start,
        )
    finally:
        CLIENT.halt()


if __name__ == "__main__":
    raise SystemExit(main())
