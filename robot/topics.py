"""Commlink topic names and port for the SLAM sensor stream.

`robot/odin_pub_node.py` is the only publisher; `robot/slam_node_.py` and
`robot/base.py` are the consumers. They are defined here so the wire contract
lives in exactly one place — renaming a topic must not mean grepping four files.

Message shapes (see robot/odin_pub_node.py for the producing code):

    slam/pose         20-float list
                        [0:7]   base-in-world  [qx,qy,qz,qw,tx,ty,tz]
                        [7:14]  cam-in-world   [qx,qy,qz,qw,tx,ty,tz]
                        [14:18] base pose      [x, y, z, yaw]
                        [18]    host timestamp (ns)
                        [19]    tracking confidence (0-100)
    slam/image        {"timestamp", "image": (H,W,3) uint8 RGB}
    slam/depth        {"timestamp", "depth": (H,W) float32 metres}
    slam/pcd          {"timestamp", "points": (H,W,4) float32, camera frame,
                       channel 3 = packed RGBA float}
    slam/camera_info  {"fx","fy","cx","cy","width","height"}

Every frame on the wire is Y-up: the Odin's native Z-up output is converted in
the publisher so consumers need no per-sensor knowledge.
"""

SLAM_PUB_PORT = 6000

POSE_TOPIC = "slam/pose"
IMAGE_TOPIC = "slam/image"
DEPTH_TOPIC = "slam/depth"
PCD_TOPIC = "slam/pcd"
CAMERA_INFO_TOPIC = "slam/camera_info"

__all__ = [
    "SLAM_PUB_PORT",
    "POSE_TOPIC",
    "IMAGE_TOPIC",
    "DEPTH_TOPIC",
    "PCD_TOPIC",
    "CAMERA_INFO_TOPIC",
]
