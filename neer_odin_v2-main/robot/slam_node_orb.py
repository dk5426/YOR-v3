#!/usr/bin/env python3
"""
slam_node_orb.py — SLAM node with ORB-SLAM3 pose feedback.

This is a drop-in replacement for ``slam_node_.py`` that adds ORB-SLAM3
as a second, globally-consistent pose sensor fused into the EKF.

Architecture
------------

    zed_pub_node  ──ZMQ──►  ZedSub  ──►  EKFSlamSource  ──►  OrbEKFSlamSource
                                          (ZED VIO + wheel enc)        │
                                                                        │  ◄── OrbSlamSub
    orbslam_bridge ──ZMQ──► orb/pose ─────────────────────────────────►       (ORB-SLAM3)
                                                                        │
                                                              MapManager / A* / Viser

The ``--orb-host`` and ``--orb-port`` arguments point to the running
``orbslam_bridge.py`` process.  If the bridge is offline, the node falls back
silently to the existing ZED+EKF behaviour.

Usage
-----
    # 1. Start ZED publisher
    python robot/zed_pub_node.py

    # 2. Generate ORB-SLAM3 config (once)
    python -m robot.nav.orbslam3.gen_config --out /tmp/orb_zed.yaml

    # 3. Start ORB-SLAM3 bridge
    python -m robot.nav.odometry.orbslam_bridge \\
        --orb-bin ~/ORB_SLAM3/Examples/RGB-D/rgbd_tum \\
        --vocab robot/nav/orbslam3/ORBvoc.txt \\
        --config /tmp/orb_zed.yaml

    # 4. Start this node
    python robot/slam_node_orb.py --ekf

Arguments are a strict superset of slam_node_.py — all existing flags work
unchanged.  Additional flags:
    --orb-host  HOST    Host running orbslam_bridge.py (default: 127.0.0.1)
    --orb-port  PORT    ZMQ port for orb/pose           (default: 6001)
    --orb-hz    HZ      Max EKF update rate from ORB    (default: 5.0)
    --no-orb            Disable ORB-SLAM3 feedback entirely
"""

import argparse
import sys
import time
from typing import Optional

# ── Re-use the entire existing stack ─────────────────────────────────────────
from robot.slam_node_ import (
    Slam,
    ZedSub,
    EKFSlamSource,
    YOR_RPC_HOST,
    YOR_RPC_PORT,
    ZED_PUB_PORT,
)
from robot.nav.odometry.orb_slam_source import OrbSlamSub, OrbEKFSlamSource


def build_slam_with_orb(
    *,
    # ── Original Slam() args ──────────────────────────────────────────────────
    target_hz:    float,
    duration_s:   float,
    load_map:     bool,
    save_map:     bool,
    map_path:     Optional[str],
    yor_host:     str   = YOR_RPC_HOST,
    yor_port:     int   = YOR_RPC_PORT,
    zed_host:     str   = "127.0.0.1",
    zed_port:     int   = ZED_PUB_PORT,
    zed_up_axis:  str   = "y",
    path_step_m:  Optional[float] = None,
    use_ekf:      bool  = True,
    predict_hz:   float = 5.0,
    # ── ORB-SLAM3 args ────────────────────────────────────────────────────────
    use_orb:      bool  = True,
    orb_host:     str   = "127.0.0.1",
    orb_port:     int   = 6001,
    orb_hz:       float = 5.0,
    orb_lc_thr_m: float = 2.0,
) -> Slam:
    """Construct a ``Slam`` instance with optional ORB-SLAM3 feedback.

    When ``use_orb=True`` and ``use_ekf=True``, the datastream is:
        ZedSub → EKFSlamSource → OrbEKFSlamSource

    Otherwise falls back to the standard Slam() construction.
    """

    # ── Build the base Slam object (creates ZedSub + EKFSlamSource internally) ─
    slam = Slam(
        target_hz   = target_hz,
        duration_s  = duration_s,
        load_map    = load_map,
        save_map    = save_map,
        map_path    = map_path,
        yor_host    = yor_host,
        yor_port    = yor_port,
        zed_host    = zed_host,
        zed_port    = zed_port,
        zed_up_axis = zed_up_axis,
        path_step_m = path_step_m,
        use_ekf     = use_ekf,
        predict_hz  = predict_hz,
    )

    # ── Optionally wrap with ORB feedback ─────────────────────────────────────
    if use_orb and use_ekf:
        print(
            f"[slam_node_orb] ORB-SLAM3 feedback enabled — "
            f"subscribing to orb/pose on {orb_host}:{orb_port}"
        )
        orb_sub = OrbSlamSub(
            host            = orb_host,
            port            = orb_port,
            stale_timeout_s = 2.0,
            lc_threshold_m  = orb_lc_thr_m,
        )

        # slam.datastream is already an EKFSlamSource at this point
        orb_fused = OrbEKFSlamSource(
            ekf_source    = slam.datastream,
            orb_sub       = orb_sub,
            orb_update_hz = orb_hz,
        )

        # Swap the datastream in-place before Slam.run() is called —
        # the mapping datastream (PCD+POSE only) stays as-is since it
        # does not need the ORB correction (ORB corrects the navigation pose,
        # not the raw camera frame used for map integration).
        slam.datastream = orb_fused

        print(
            f"[slam_node_orb] OrbEKFSlamSource installed — "
            f"ORB updates at ≤{orb_hz:.0f} Hz, "
            f"loop-closure threshold = {orb_lc_thr_m:.2f} m"
        )
    elif use_orb and not use_ekf:
        print(
            "[slam_node_orb] WARN: --orb requires --ekf (EKF disabled). "
            "Running without ORB-SLAM3 feedback.",
            file=sys.stderr,
        )

    return slam


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SLAM node with ORB-SLAM3 pose feedback "
            "(superset of slam_node_.py arguments)"
        )
    )

    # ── Original slam_node_.py flags (kept identical) ─────────────────────────
    parser.add_argument(
        "--hz", type=float, default=10.0,
        help="Target mapping rate (Hz); 0 = as fast as possible",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Stop after N seconds (0 = run until Ctrl+C)",
    )
    parser.add_argument(
        "--load", action="store_true",
        help="Load previous map instead of starting new mapping",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the map on exit",
    )
    parser.add_argument(
        "--map-path", type=str, default=None,
        help="Optional .npz path to save/load map",
    )
    parser.add_argument(
        "--yor-host", type=str, default=YOR_RPC_HOST,
        help="Yor RPC host (follow_path via RPC)",
    )
    parser.add_argument(
        "--yor-port", type=int, default=YOR_RPC_PORT,
        help="Yor RPC port",
    )
    parser.add_argument(
        "--path-step-m", type=float, default=None,
        help="Dense waypoint spacing for follow_path (metres; 0 disables)",
    )
    parser.add_argument(
        "--zed-up-axis", type=str, default="y", choices=["y", "z"],
        help="Up axis for incoming ZED frames",
    )
    parser.add_argument(
        "--no-ekf", dest="ekf", action="store_false",
        help="Disable EKF fusion (enabled by default)",
    )
    parser.add_argument(
        "--predict-hz", type=float, default=5.0,
        help=(
            "EKF predict rate in Hz (default: 5.0). Each tick fires a "
            "get_base_encoders RPC to yor over the network. With yor on a "
            "separate Pi sharing a single ZMQ REP socket, lower values leave "
            "RPC bandwidth for joystick set_base_velocity and follow_path. "
            "Raise to 20 only on a low-latency LAN with no joystick contention."
        ),
    )
    parser.add_argument(
        "--zed-host", type=str, default="127.0.0.1",
        help="Host running zed_pub_node",
    )
    parser.add_argument(
        "--zed-port", type=int, default=ZED_PUB_PORT,
        help="ZMQ port for zed_pub_node",
    )

    # ── New ORB-SLAM3 flags ───────────────────────────────────────────────────
    parser.add_argument(
        "--no-orb", dest="orb", action="store_false",
        help=(
            "Disable ORB-SLAM3 feedback (enabled by default when EKF is on). "
            "The system still runs ZED+encoder EKF without ORB."
        ),
    )
    parser.add_argument(
        "--orb-host", type=str, default="127.0.0.1",
        help="Host running orbslam_bridge.py (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--orb-port", type=int, default=6001,
        help="ZMQ port for the orb/pose topic (default: 6001)",
    )
    parser.add_argument(
        "--orb-hz", type=float, default=5.0,
        help="Maximum EKF update rate from ORB-SLAM3 (default: 5.0 Hz)",
    )
    parser.add_argument(
        "--orb-lc-thr", type=float, default=2.0,
        help=(
            "Loop-closure position-jump threshold in metres (default: 2.0). "
            "Jumps larger than this trigger an unconditional EKF correction. "
            "Set high enough that spurious ORB jumps don't force-snap the EKF; "
            "lower (e.g. 0.3) only after the bridge is validated stable."
        ),
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  slam_node_orb — ORB-SLAM3 feedback SLAM node")
    print(f"  EKF:    {'ON' if args.ekf else 'OFF'}")
    print(f"  ORB:    {'ON' if args.orb else 'OFF'}")
    if args.orb and args.ekf:
        print(f"  orb host/port: {args.orb_host}:{args.orb_port}")
        print(f"  orb update hz: {args.orb_hz}")
        print(f"  LC threshold:  {args.orb_lc_thr} m")
    print("=" * 60 + "\n")

    slam = build_slam_with_orb(
        target_hz    = args.hz,
        duration_s   = args.duration,
        load_map     = args.load,
        save_map     = args.save,
        map_path     = args.map_path,
        yor_host     = args.yor_host,
        yor_port     = args.yor_port,
        zed_host     = args.zed_host,
        zed_port     = args.zed_port,
        zed_up_axis  = args.zed_up_axis,
        path_step_m  = args.path_step_m,
        use_ekf      = args.ekf,
        predict_hz   = args.predict_hz,
        use_orb      = args.orb,
        orb_host     = args.orb_host,
        orb_port     = args.orb_port,
        orb_hz       = args.orb_hz,
        orb_lc_thr_m = args.orb_lc_thr,
    )

    slam.run()


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
