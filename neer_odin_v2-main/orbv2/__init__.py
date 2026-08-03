"""
orbv2 — Clean ORB-SLAM3 loop-closure integration for the ZED SLAM pipeline.

This package adds ORB-SLAM3 as a globally-consistent pose sensor fused into
the existing EKF, providing loop-closure corrections that reduce cumulative
drift.  The existing ZED + wheel-encoder pipeline is left entirely intact;
orbv2 wraps it with an additional measurement source.

Usage:
    # 1. Start ZED publisher (unchanged)
    python -m robot.zed_pub_node --fresh

    # 2. Start ORB-SLAM3 bridge (new)
    python -m orbv2.orb_bridge

    # 3. Start the ORB-enhanced SLAM node (new)
    python -m orbv2.orb_slam_node

    # Or use the tmux launcher:
    bash orbv2/run.sh
"""

__version__ = "0.1.0"
