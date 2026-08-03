#!/usr/bin/env python3
"""
orb_slam_node.py — SLAM node with ORB-SLAM3 loop-closure feedback.

This is the main entry point for orbv2.  It constructs the standard ``Slam``
object from the existing pipeline and wraps its datastream with
``OrbFusedSource`` to add ORB-SLAM3 pose corrections.

Everything you get from the original slam_node_.py — voxel map, Viser stream,
A* planning, path sending — works identically.  The only addition is that the
EKF now receives a second measurement source (ORB-SLAM3 keyframe poses) that
provides globally-consistent corrections after loop closures.

Architecture
------------
    zed_pub_node  ──ZMQ:6000──►  ZedSub  ──►  EKFSlamSource  ──►  OrbFusedSource
                                               (ZED VIO + encoders)       │
                                                                          │ ◄── orb/pose
    orbv2.orb_bridge ──ZMQ:6001──► orb/pose ──────────────────────►      (ORB-SLAM3)
                                                                          │
                                                               MapManager / A* / Viser

Usage
-----
    # 1. Start ZED publisher (unchanged)
    python -m robot.zed_pub_node --fresh

    # 2. Start ORB-SLAM3 bridge (orbv2)
    python -m orbv2.orb_bridge

    # 3. Start this node
    python -m orbv2.orb_slam_node

    # Or use the tmux launcher:
    bash orbv2/run.sh

All slam_node_.py flags are supported.  Additional flags:
    --orb-host  HOST    Host running orbv2.orb_bridge (default: 127.0.0.1)
    --orb-port  PORT    ZMQ port for orb/pose         (default: 6001)
    --orb-hz    HZ      Max EKF update rate from ORB  (default: 5.0)
    --orb-lc-thr M      Loop-closure position-jump threshold (default: 0.30 m)
    --no-orb            Disable ORB-SLAM3 feedback entirely
"""

import argparse
import sys
from typing import Optional

from robot.slam_node_ import (
    Slam,
    YOR_RPC_HOST,
    YOR_RPC_PORT,
    ZED_PUB_PORT,
)
from orbv2.orb_fused_source import OrbFusedSource, ORB_PUB_PORT
from orbv2.diagnostics import OrbHealthMonitor


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
    orb_port:     int   = ORB_PUB_PORT,
    orb_hz:       float = 5.0,
    orb_lc_thr_m: float = 0.30,
) -> Slam:
    """Construct a ``Slam`` instance with optional ORB-SLAM3 feedback.

    When ``use_orb=True`` and ``use_ekf=True``, the datastream is:
        ZedSub → EKFSlamSource → OrbFusedSource → Slam
    """

    # ── Build the base Slam object ────────────────────────────────────────────
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

    # ── Wrap with ORB feedback ────────────────────────────────────────────────
    if use_orb and use_ekf:
        print(
            f"[orbv2] ORB-SLAM3 feedback enabled — "
            f"subscribing to orb/pose on {orb_host}:{orb_port}"
        )

        orb_fused = OrbFusedSource(
            ekf_source     = slam.datastream,
            orb_host       = orb_host,
            orb_port       = orb_port,
            orb_update_hz  = orb_hz,
            lc_threshold_m = orb_lc_thr_m,
        )

        # Swap the datastream — mapping stream stays as-is (doesn't need ORB)
        slam.datastream = orb_fused

        # Start health monitor
        health = OrbHealthMonitor(orb_fused, log_interval_s=10.0)
        health.start()
        # Attach to slam so it stays alive with the object
        slam._orbv2_health = health

        print(
            f"[orbv2] OrbFusedSource installed — "
            f"ORB updates at ≤{orb_hz:.0f} Hz, "
            f"loop-closure threshold = {orb_lc_thr_m:.2f} m"
        )
    elif use_orb and not use_ekf:
        print(
            "[orbv2] WARN: --orb requires --ekf (EKF disabled). "
            "Running without ORB-SLAM3 feedback.",
            file=sys.stderr,
        )

    return slam


def main():
    import signal
    def handle_sigterm(*args):
        print("\n[orbv2] Termination signal received! Initiating graceful shutdown to save map...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGHUP, handle_sigterm)

    parser = argparse.ArgumentParser(
        description=(
            "orbv2 — SLAM node with ORB-SLAM3 loop-closure feedback "
            "(superset of slam_node_.py arguments)"
        )
    )

    # ── Original slam_node_.py flags ──────────────────────────────────────────
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
            "get_base_encoders RPC to yor over the network. Keep low "
            "(5 Hz) to avoid starving the joystick on the NUC's single "
            "REP socket."
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

    # ── ORB-SLAM3 flags ──────────────────────────────────────────────────────
    parser.add_argument(
        "--no-orb", dest="orb", action="store_false",
        help=(
            "Disable ORB-SLAM3 feedback (enabled by default when EKF is on). "
            "The system still runs ZED+encoder EKF without ORB."
        ),
    )
    parser.add_argument(
        "--orb-host", type=str, default="127.0.0.1",
        help="Host running orbv2.orb_bridge (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--orb-port", type=int, default=ORB_PUB_PORT,
        help="ZMQ port for the orb/pose topic (default: 6001)",
    )
    parser.add_argument(
        "--orb-hz", type=float, default=5.0,
        help="Maximum EKF update rate from ORB-SLAM3 (default: 5.0 Hz)",
    )
    parser.add_argument(
        "--orb-lc-thr", type=float, default=0.30,
        help=(
            "Loop-closure position-jump threshold in metres (default: 0.30). "
            "Jumps larger than this trigger an unconditional EKF correction."
        ),
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  orbv2 — ORB-SLAM3 Loop-Closure SLAM Node")
    print(f"  EKF:    {'ON' if args.ekf else 'OFF'}")
    print(f"  ORB:    {'ON' if args.orb else 'OFF'}")
    if args.orb and args.ekf:
        print(f"  orb host/port: {args.orb_host}:{args.orb_port}")
        print(f"  orb update hz: {args.orb_hz}")
        print(f"  LC threshold:  {args.orb_lc_thr} m")
    print("=" * 60 + "\n")

    map_path = args.map_path
    if args.save and not map_path:
        import datetime
        import os
        import glob
        import re
        
        this_dir = os.path.dirname(os.path.abspath(__file__))
        maps_dir = os.path.join(this_dir, "maps")
        os.makedirs(maps_dir, exist_ok=True)
        
        existing_maps = glob.glob(os.path.join(maps_dir, "lab_m*_*.npz"))
        max_idx = 0
        for m in existing_maps:
            basename = os.path.basename(m)
            match = re.match(r"lab_m(\d+)_", basename)
            if match:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx
        next_idx = max_idx + 1
        
        timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
        map_path = os.path.join(maps_dir, f"lab_m{next_idx:02d}_{timestamp}.npz")
        print(f"[orbv2] Auto-configured map save path: {map_path}")

    slam = build_slam_with_orb(
        target_hz    = args.hz,
        duration_s   = args.duration,
        load_map     = args.load,
        save_map     = args.save,
        map_path     = map_path,
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
