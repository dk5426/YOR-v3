#!/usr/bin/env python3
"""Odin 1 publisher node — drop-in replacement for robot/zed_pub_node.py.

Publishes the SAME commlink topics, port and message shapes as zed_pub_node.py,
so the downstream stack (slam_node_, mapping_torch, viserBridge, pathPlanning)
runs UNCHANGED. The source is a Manifold Odin 1 SLAM box reached through the
`pyodin` pybind bridge over the native C SDK — no ROS anywhere.

Topic contract (identical to zed_pub_node.py; names kept as "zed/..." on purpose
so slam_node_.py needs zero edits):
  zed/pose        -> 20-float list  base(0:7) cam(7:14) [x,y,z,yaw](14:18) ts(18) conf(19)
  zed/image       -> {"timestamp", "image": HxWx3 uint8 RGB}
  zed/depth       -> {"timestamp", "depth": HxW float32 m}
  zed/pcd         -> {"timestamp", "points": (H,W,4) float32 camera-frame, packed-RGBA}
  zed/camera_info -> {"fx","fy","cx","cy","width","height", ...}

Coordinate frames: Odin is right-handed Z-up / X-forward (REP-103). neer_slam's
world is Y-up. We convert here (reusing the codebase's own (x,y,z)->(x,z,-y) point
rule and M·T·Mᵀ pose rule) so consumers keep their default up_axis="y".
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from commlink import Publisher
from loop_rate_limiters import RateLimiter

from robot.utils.utils import theta_y_from_R

import pyodin

# ---------------------------------------------------------------------------
# Contract constants — DO NOT rename (kept identical to zed_pub_node.py).
# ---------------------------------------------------------------------------
ZED_PUB_PORT = 6000
POSE_TOPIC = "zed/pose"
IMAGE_TOPIC = "zed/image"
DEPTH_TOPIC = "zed/depth"
PCD_TOPIC = "zed/pcd"
CAMERA_INFO_TOPIC = "zed/camera_info"

# Z-up (Odin) -> Y-up (neer_slam) basis change. Same matrix as slam_node_.py:110-118.
M_ZUP_TO_YUP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

CONF_THRESHOLD = 30  # dToF confidence gate (README §4.4 recommends 30–35)


def _load_config(config_path: str):
    """Returns (T_cam_to_base 4x4, pcd_source). Defaults: identity, 'slam'.

    T_cam_to_base is the mount extrinsic (identity until the box is on the robot —
    the ZED used a hard-coded 23.8° tilt; Odin needs its own value).
    pcd_source selects the zed/pcd cloud: 'slam' (Odin on-device colored cloud,
    the only one available in SLAM mode) or 'dtof' (organized raw cloud — not
    streamed alongside SLAM on current firmware)."""
    T = np.eye(4, dtype=np.float64)
    pcd_source = "slam"
    cam_info = None
    if config_path and os.path.exists(config_path):
        try:
            import yaml

            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            m = cfg.get("T_cam_to_base")
            if m is not None:
                T = np.asarray(m, dtype=np.float64).reshape(4, 4)
            pcd_source = str(cfg.get("pcd_source", pcd_source)).lower()
            if isinstance(cfg.get("camera_info"), dict):
                cam_info = {k: cfg["camera_info"][k]
                            for k in ("fx", "fy", "cx", "cy", "width", "height")
                            if k in cfg["camera_info"]}
        except Exception as e:  # pragma: no cover - config is optional
            print(f"[odin_pub] could not load config {config_path}: {e}")
    return T, pcd_source, cam_info


def _odom_to_Twc_zup(odom: dict) -> np.ndarray:
    """Build the 4x4 WORLD_T_CAM (Z-up, Odin native) from an odom dict."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(np.asarray(odom["quat_xyzw"], dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(odom["pos"], dtype=np.float64)
    return T


def _zup_to_yup_pose(T_zup: np.ndarray) -> np.ndarray:
    return M_ZUP_TO_YUP @ T_zup @ M_ZUP_TO_YUP.T


def _decode_rgb(rgb_msg: dict):
    """Odin RGB -> HxWx3 uint8 RGB. Handles both JPEG and NV12 (host_sdk_sample.h:806)."""
    import cv2

    buf = rgb_msg["data"]
    w, h = int(rgb_msg["width"]), int(rgb_msg["height"])
    if rgb_msg["is_jpeg"]:
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # NV12: Y plane (h x w) followed by interleaved UV (h/2 x w)
    yuv = np.frombuffer(buf, dtype=np.uint8).reshape((h * 3 // 2, w))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)


def _dtof_to_zed_pcd(dc: dict) -> np.ndarray:
    """Organized dToF cloud (camera frame, Z-up) -> ZED-compatible (H,W,4) float32.

    Channels 0:3 = XYZ in the Y-up camera frame; channel 3 = packed-RGBA float
    (white) so mapping_torch's `_rgba_float_to_rgb_u8` decodes it unchanged.
    Invalid / low-confidence pixels are set to NaN so the GPU filter drops them.
    """
    xyz = np.asarray(dc["xyz"], dtype=np.float32)  # (H, W, 3) camera frame, Z-up
    h, w = xyz.shape[:2]
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    xyz_y = np.stack([x, z, -y], axis=-1)  # camera-frame axes Z-up -> Y-up

    # White, opaque packed-RGBA float (byte order irrelevant when all 255).
    rgba = np.full((h, w, 4), 255, dtype=np.uint8)
    packed = rgba.view(np.float32).reshape(h, w)

    out = np.empty((h, w, 4), dtype=np.float32)
    out[..., :3] = xyz_y
    out[..., 3] = packed

    invalid = ~np.isfinite(z) | (z <= 0.0)  # dToF: 0 range == no return
    conf = dc.get("confidence")
    if conf is not None:
        invalid |= np.asarray(conf) < CONF_THRESHOLD
    out[invalid] = np.nan
    return out


def _slam_to_zed_pcd(cloud: dict, Twc: np.ndarray):
    """Odin SLAM cloud (world frame, Z-up, colored) -> ZED-compatible (1,N,4) float32
    in the Y-up CAMERA frame, so mapping_torch transforms it back to world via the cam
    pose (pose_msg[7:14]). Channel 3 is packed-RGBA with the cloud's real colors.

    Note: the SLAM cloud is unordered, so mapping_torch's organized 3x3 flying-pixel
    rejection is only approximate here — acceptable because Odin already filters the
    cloud on-device. A cleaner Odin-native consumption path is a Phase-2 refinement.
    """
    arr = np.asarray(cloud["xyzrgba"])  # (n,7) int32
    if arr.shape[0] == 0:
        return None
    pts_w_z = arr[:, :3].astype(np.float64) * 1.0e-4          # 0.1 mm -> m, Z-up world
    rgb = (arr[:, 3:6] & 0xFF).astype(np.uint8)               # r,g,b (low byte)
    x, y, z = pts_w_z[:, 0], pts_w_z[:, 1], pts_w_z[:, 2]
    pts_w_y = np.stack([x, z, -y], axis=1)                    # Z-up -> Y-up world
    Twc_inv = np.linalg.inv(Twc)
    pts_cam = (pts_w_y @ Twc_inv[:3, :3].T) + Twc_inv[:3, 3]  # world -> camera (Y-up)
    n = pts_cam.shape[0]
    rgba = np.empty((n, 4), np.uint8)
    rgba[:, :3] = rgb
    rgba[:, 3] = 255
    packed = rgba.view(np.float32).reshape(n)
    out = np.empty((1, n, 4), dtype=np.float32)
    out[0, :, :3] = pts_cam.astype(np.float32)
    out[0, :, 3] = packed
    return out


def main():
    parser = argparse.ArgumentParser("Odin 1 publisher (drop-in for zed_pub_node)")
    parser.add_argument("--fps", type=int, default=30, help="publish rate cap")
    parser.add_argument("--wait", type=float, default=15.0, help="seconds to wait for the box")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "config", "odin.yaml"))
    parser.add_argument("--map", type=str, default=None,
                        help="relocalization map file to load (Mode 2)")
    parser.add_argument("--fresh", action="store_true",
                        help="start a fresh map (ignore any saved relocalization map)")
    args = parser.parse_args()

    T_cam_to_base, pcd_source, cam_info_cfg = _load_config(args.config)
    # slam_node_ BLOCKS at startup until zed/camera_info arrives, and the device
    # returns zero intrinsics on current firmware — so always have a fallback.
    # It only feeds the mapping carver's (self-consistent) projection model.
    cam_info_fallback = cam_info_cfg or {"fx": 700.0, "fy": 700.0, "cx": 800.0,
                                         "cy": 648.0, "width": 1600, "height": 1296}

    print("[odin_pub] starting Odin SDK (SLAM mode)…")
    sensor = pyodin.OdinSensor()
    if not sensor.start("slam", args.wait):
        raise RuntimeError("Odin device did not connect — check USB3 cable / udev rule / `lsusb`.")
    print(f"[odin_pub] connected. counts={sensor.counts()}")

    if args.map and not args.fresh:
        rc = sensor.set_relocalization_map(os.path.abspath(args.map))
        print(f"[odin_pub] set_relocalization_map({args.map}) -> {rc}")

    calib = sensor.get_calibration()
    if calib:
        print(f"[odin_pub] intrinsics fx={calib['fx']:.2f} fy={calib['fy']:.2f} "
              f"cx={calib['cx']:.2f} cy={calib['cy']:.2f} "
              f"size={calib['width']}x{calib['height']}")

    pub = Publisher("*", port=ZED_PUB_PORT)
    rate = RateLimiter(args.fps, name="odin_pub")
    print(f"[odin_pub] publishing {POSE_TOPIC}|{IMAGE_TOPIC}|{DEPTH_TOPIC}|{PCD_TOPIC} on :{ZED_PUB_PORT}")

    last_cam_info = 0.0
    last_odom_ts = -1
    last_odom_change = time.time()

    try:
        while True:
            odom = sensor.get_odom()
            if odom is None:
                rate.sleep()
                continue

            ts = time.time_ns()  # host stamp, matches zed_pub_node.py
            Twc = _zup_to_yup_pose(_odom_to_Twc_zup(odom))   # cam-in-world, Y-up
            Twb = Twc @ T_cam_to_base                         # base-in-world, Y-up

            # Confidence heuristic: high while odometry keeps advancing.
            now = time.time()
            if odom["timestamp_ns"] != last_odom_ts:
                last_odom_ts = odom["timestamp_ns"]
                last_odom_change = now
            confidence = 100 if (now - last_odom_change) < 0.5 and sensor.connected() else 0

            base_q = R.from_matrix(Twb[:3, :3]).as_quat()  # [qx,qy,qz,qw]
            cam_q = R.from_matrix(Twc[:3, :3]).as_quat()
            base = base_q.tolist() + Twb[:3, 3].tolist()
            cam = cam_q.tolist() + Twc[:3, 3].tolist()
            yaw = theta_y_from_R(Twb)  # rotation about the Y (up) axis
            base_pose = [float(Twb[0, 3]), float(Twb[1, 3]), float(Twb[2, 3]), float(yaw)]

            # Layout: base(0:7) cam(7:14) base_pose(14:18) ts(18) confidence(19)
            pose_msg = base + cam + base_pose + [ts, confidence]
            pub.publish(POSE_TOPIC, pose_msg)

            if confidence == 0:
                rate.sleep()  # tracking stale — skip sensor payloads (like zed_pub_node)
                continue

            rgb = sensor.get_rgb()
            if rgb is not None:
                img = _decode_rgb(rgb)
                if img is not None:
                    pub.publish(IMAGE_TOPIC, {"timestamp": ts, "image": img})

            dep = sensor.get_depth()
            if dep is not None:
                pub.publish(DEPTH_TOPIC, {"timestamp": ts, "depth": dep["depth"]})

            if pcd_source == "dtof":
                dc = sensor.get_dtof_cloud()
                if dc is not None:
                    pub.publish(PCD_TOPIC, {"timestamp": ts, "points": _dtof_to_zed_pcd(dc)})
            else:  # 'slam' — the only cloud available alongside SLAM streams
                sc = sensor.get_cloud()
                if sc is not None and sc["n"] > 0:
                    pts = _slam_to_zed_pcd(sc, Twc)
                    if pts is not None:
                        pub.publish(PCD_TOPIC, {"timestamp": ts, "points": pts})

            if now - last_cam_info > 1.0:
                ci = sensor.get_calibration()
                if ci and float(ci["fx"]) != 0.0:
                    msg = {"fx": float(ci["fx"]), "fy": float(ci["fy"]),
                           "cx": float(ci["cx"]), "cy": float(ci["cy"]),
                           "width": int(ci["width"]), "height": int(ci["height"])}
                else:
                    msg = dict(cam_info_fallback)
                pub.publish(CAMERA_INFO_TOPIC, msg)
                last_cam_info = now

            rate.sleep()
    except KeyboardInterrupt:
        print("[odin_pub] shutting down…")
    finally:
        sensor.stop()
        # The Odin SDK double-frees in its static-destructor teardown at process
        # exit (after deinit completes cleanly). Hard-exit past it.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
