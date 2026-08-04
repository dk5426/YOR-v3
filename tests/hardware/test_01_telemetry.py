#!/usr/bin/env python3
"""test_01_telemetry.py — is the data trustworthy? NOTHING MOVES.

Stage 0 proved the links exist. This proves the numbers coming back are
meaningful: they update, they hold still when the robot is still, and they are
in the units the rest of the stack assumes.

Debugging value: almost every confusing behaviour later on ("the arm jumped",
"it drove the wrong way", "the map is smeared") traces back to one of these
signals being stale, noisy or in the wrong frame. Catch it here, standing still.

    python tests/hardware/test_01_telemetry.py --host <robot-ip>

Read-only. Sends no motion commands. Push the robot by hand when it asks.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hw import (  # noqa: E402
    banner, check, connect, info, pause, parse_args, run,
)

CLIENT = None
ARGS = None


def _sample(method: str, n: int, dt: float = 0.05):
    out = []
    for _ in range(n):
        out.append(CLIENT.call(method))
        time.sleep(dt)
    return out


def test_rpc_latency():
    print("\nRPC latency")
    t0 = time.monotonic()
    n = 20
    for _ in range(n):
        CLIENT.call("get_state")
    per = (time.monotonic() - t0) / n
    check("get_state round trip under 100 ms", per < 0.100, f"{per * 1000:.1f} ms")
    if per > 0.030:
        info("slower than expected. The RPC server is a single REP socket — "
             "another client (joystick, slam_node_) may be competing for it.")


def test_encoders_are_live():
    print("\nencoders update")
    samples = _sample("get_base_encoders", 10)
    ok = all(isinstance(s, dict) for s in samples)
    check("every encoder read returned data", ok)
    if not ok:
        return

    stamps = [s.get("timestamp") for s in samples]
    advancing = all(b > a for a, b in zip(stamps, stamps[1:]))
    check("encoder timestamps advance", advancing,
          f"span {stamps[-1] - stamps[0]:.3f} s")

    # Standing still, drive counts should be essentially constant.
    counts = [s["drive_counts"] for s in samples]
    drift = [max(c[i] for c in counts) - min(c[i] for c in counts) for i in range(4)]
    check("drive counts are steady while stationary", max(drift) < 1.0,
          "max drift " + ", ".join(f"{d:.2f}" for d in drift))
    if max(drift) >= 1.0:
        info("a wheel is turning, or an encoder is noisy — this feeds the EKF "
             "predict step and will corrupt the fused pose")

    steer = [s["steer_rad"] for s in samples]
    jitter = [statistics.pstdev([v[i] for v in steer]) for i in range(4)]
    check("steer angles are steady while stationary", max(jitter) < 0.02,
          "max sigma " + ", ".join(f"{j:.4f}" for j in jitter))


def test_lift_telemetry():
    print("\nlift telemetry")
    status = CLIENT.call("get_lift_status")
    if not isinstance(status, dict) or not status.get("available"):
        check("lift available", False, "skipping; see test_00")
        return

    known = status.get("position_known")
    if known is not True:
        check("lift position established", False,
              "run test_02_lift.py to home it — heights are meaningless until then")
        info("This is expected on a cold boot. Not a fault.")
        return

    heights = [CLIENT.call("get_lift_height") for _ in range(8)]
    heights = [h for h in heights if h is not None]
    check("lift height reads consistently", len(heights) == 8, f"{len(heights)}/8")
    if heights:
        spread = max(heights) - min(heights)
        check("lift height is stable while idle", spread < 0.001, f"{spread * 1000:.2f} mm")
        info(f"height = {statistics.mean(heights):.4f} m")
        check("lift height is inside the declared travel",
              -0.005 <= min(heights) and max(heights) <= 0.905,
              f"{min(heights):.4f} … {max(heights):.4f} m")


def test_pose_source():
    print("\nbase pose (from SLAM, via the node)")
    pose = CLIENT.call("get_pose")
    if not isinstance(pose, dict) or pose.get("x") is None:
        check("node has a SLAM pose", False,
              "get_pose returned None. The node only reads slam/pose while its "
              "base controller is in a nav mode; this is expected during "
              "whole-body control. Not a fault by itself.")
        return
    check("node has a SLAM pose", True,
          f"x={pose['x']:+.3f} y={pose['y']:+.3f} theta={math.degrees(pose['theta']):+.1f} deg")

    poses = _sample("get_pose", 10)
    xs = [p["x"] for p in poses if p and p.get("x") is not None]
    ys = [p["y"] for p in poses if p and p.get("y") is not None]
    if len(xs) >= 5:
        drift = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        check("pose is stable while stationary", drift < 0.02, f"{drift * 1000:.1f} mm spread")
        if drift >= 0.02:
            info("VIO is jittering. Check lighting and that the Odin is rigidly "
                 "mounted — the EKF will inherit this noise.")


def test_solver_state():
    print("\nwhole-body solver")
    states = _sample("get_state", 10)
    solved = sum(1 for s in states if s and s.get("solved"))
    check("solver converges consistently", solved >= 9, f"{solved}/10")
    if solved < 10:
        errs = {s.get("solve_error") for s in states if s and s.get("solve_error")}
        if errs:
            info("solver errors: " + "; ".join(str(e) for e in errs))

    s = states[-1]
    vel = s.get("base_velocity") or [0, 0, 0]
    check("solver is not commanding base motion while idle",
          max(abs(v) for v in vel) < 1e-6,
          ", ".join(f"{v:+.4f}" for v in vel))
    if max(abs(v) for v in vel) >= 1e-6:
        info("the solver wants to drive with no target change — the EE target "
             "may be unreachable, so it is trying to walk the base toward it")

    ee_l = s.get("left_ee_wxyz_xyz")
    ee_r = s.get("right_ee_wxyz_xyz")
    check("both EE poses are reported", ee_l is not None and ee_r is not None)
    if ee_l and ee_r:
        sep = math.dist(ee_l[4:7], ee_r[4:7])
        check("hands are a plausible distance apart", 0.05 < sep < 2.0, f"{sep:.3f} m")


def test_manual_push_moves_encoders():
    print("\nencoders respond to hand motion")
    info("This distinguishes 'encoder is dead' from 'encoder is fine, nothing moved'.")
    pause("Grab a wheel and be ready to spin it by hand")

    before = CLIENT.call("get_base_encoders")["drive_counts"]
    print("  ..   spin ANY drive wheel by hand now")
    moved = False
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        after = CLIENT.call("get_base_encoders")["drive_counts"]
        deltas = [abs(a - b) for a, b in zip(after, before)]
        if max(deltas) > 0.5:
            moved = True
            idx = deltas.index(max(deltas))
            info(f"module {['FL', 'FR', 'RR', 'RL'][idx]} moved {deltas[idx]:.2f} counts")
            break
        time.sleep(0.1)
    check("a hand-spun wheel shows up in drive_counts", moved,
          "no change in 10 s — the encoder or the CAN link for that module is dead"
          if not moved else "")


def main() -> int:
    global CLIENT, ARGS
    ARGS = parse_args(__doc__)
    banner("STAGE 0 — TELEMETRY", "Read-only. Nothing moves under power.")
    CLIENT = connect(ARGS)
    return run(
        test_rpc_latency,
        test_encoders_are_live,
        test_lift_telemetry,
        test_pose_source,
        test_solver_state,
        test_manual_push_moves_encoders,
    )


if __name__ == "__main__":
    raise SystemExit(main())
