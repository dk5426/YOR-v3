"""Offline-dataset recording for the YOR robot.

Captures raw ZED publisher output (image/depth/pcd/pose) and wheel-encoder
RPC samples into a rosbag-style directory so SLAM / odometry / mapping
pipelines can be re-run many times without re-driving the robot.
"""
