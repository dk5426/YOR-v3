#!/usr/bin/env python3
"""Standalone Odin 1 + Viser test — no robot, no slam_node, no commlink.

Connects straight to the Odin 1 through the `pyodin` bridge and visualises, in
real time:
  * the box's on-device SLAM point cloud (colored) — i.e. the map it is building,
  * the live RGB camera feed,
  * the live camera pose / trajectory from odometry.

This lets you validate the sensor and watch the mapping output independently
before attaching anything to the robot.

Native Odin frame is right-handed Z-up / X-forward (REP-103), so the viewer is
configured Z-up to show the map exactly as the device reports it.

Run:
    conda activate slam-odin
    python tools/odin_viser_test.py
    # open http://localhost:8080
"""

import argparse
import os
import sys
import time

import numpy as np
import viser

import pyodin


def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel: float = 0.03):
    """Cheap voxel-grid dedup so the browser stays responsive (from viser_subscriber.py)."""
    if len(pts) == 0:
        return pts, cols
    keys = np.round(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx], cols[idx]


def decode_rgb(rgb: dict, max_w: int = 800):
    """Odin RGB dict -> HxWx3 uint8 RGB (handles JPEG and NV12), downscaled to
    max_w wide so the browser feed stays snappy (1600px frames are heavy)."""
    import cv2

    buf = rgb["data"]
    w, h = int(rgb["width"]), int(rgb["height"])
    if rgb["is_jpeg"]:
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        yuv = np.frombuffer(buf, dtype=np.uint8).reshape((h * 3 // 2, w))
        out = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
    if out.shape[1] > max_w:
        nh = int(round(max_w * out.shape[0] / out.shape[1]))
        out = cv2.resize(out, (max_w, nh), interpolation=cv2.INTER_AREA)
    return out


def parse_slam_cloud(cloud: dict):
    """{n, xyzrgba:(n,7) int32} -> (points (n,3) float32 m, colors (n,3) uint8)."""
    arr = np.asarray(cloud["xyzrgba"], dtype=np.int32)
    if arr.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    pts = arr[:, :3].astype(np.float32) * 1.0e-4           # 0.1 mm -> m
    cols = arr[:, 3:6].astype(np.uint8)                    # r, g, b (low byte)
    return pts, cols


def main():
    ap = argparse.ArgumentParser("Odin 1 standalone Viser test")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--voxel", type=float, default=0.04, help="map voxel size (m)")
    ap.add_argument("--max-points", type=int, default=1_000_000)
    ap.add_argument("--cloud-period", type=float, default=0.4,
                    help="seconds between point-cloud rebuilds (lower = snappier map, more CPU)")
    ap.add_argument("--max-range", type=float, default=6.0,
                    help="drop cloud points farther than this (m) from the camera (cuts far noise)")
    ap.add_argument("--wait", type=float, default=15.0)
    args = ap.parse_args()

    print("[viser-test] starting Odin SDK (SLAM mode)…")
    sensor = pyodin.OdinSensor()
    if not sensor.start("slam", args.wait):
        raise SystemExit("Odin device did not connect — check USB3 cable / udev rule / `lsusb`.")
    print(f"[viser-test] connected. counts={sensor.counts()}")

    calib = sensor.get_calibration()
    if calib:
        print(f"[viser-test] intrinsics: fx={calib['fx']:.2f} fy={calib['fy']:.2f} "
              f"cx={calib['cx']:.2f} cy={calib['cy']:.2f} size={calib['width']}x{calib['height']}")
        sensor.get_calib_file("odin_calib.yaml")  # pull fisheye polynomial for the record
        print("[viser-test] saved device calib.yaml -> odin_calib.yaml")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")            # Odin native Z-up
    server.initial_camera.up = (0, 0, 1)
    server.initial_camera.position = (-3, 0, 1.5)
    server.initial_camera.look_at = (1, 0, 0)      # X forward
    server.scene.add_frame("/world", show_axes=True, axes_length=0.5, axes_radius=0.01)

    gui_image = server.gui.add_image(np.zeros((480, 640, 3), np.uint8), label="Odin RGB")
    gui_mode = server.gui.add_dropdown(
        "Cloud display", ("Accumulate", "Latest frame"), initial_value="Accumulate"
    )
    gui_range = server.gui.add_slider(
        "Max range (m)", min=1.0, max=12.0, step=0.5, initial_value=float(args.max_range)
    )
    gui_status = server.gui.add_markdown("waiting for data…")
    gui_reset = server.gui.add_button("Reset map")

    map_pts = np.empty((0, 3), np.float32)
    map_cols = np.empty((0, 3), np.uint8)
    reset_flag = {"v": False}

    @gui_reset.on_click
    def _(_=None):
        reset_flag["v"] = True

    last_rgb_ts = -1
    last_cloud_push = 0.0
    last_status = 0.0
    cam_pos = None  # latest camera world position (for range filtering)

    try:
        while True:
            now = time.time()
            if reset_flag["v"]:
                map_pts = np.empty((0, 3), np.float32)
                map_cols = np.empty((0, 3), np.uint8)
                server.scene.add_point_cloud("/odin_map", points=map_pts, colors=map_cols, point_size=args.voxel)
                reset_flag["v"] = False

            # --- RGB every loop, but only re-decode genuinely new frames (keeps it smooth) ---
            rgb = sensor.get_rgb()
            if rgb is not None and rgb["timestamp"] != last_rgb_ts:
                last_rgb_ts = rgb["timestamp"]
                img = decode_rgb(rgb)
                if img is not None:
                    gui_image.image = img

            # --- Camera pose frame every loop (cheap geometry; no per-frame texture upload) ---
            odom = sensor.get_odom()
            if odom is not None:
                q = np.asarray(odom["quat_xyzw"], float)        # x, y, z, w
                wxyz = (q[3], q[0], q[1], q[2])                  # viser wants w, x, y, z
                cam_pos = np.asarray(odom["pos"], np.float32)
                server.scene.add_frame("/odin_cam", wxyz=wxyz, position=tuple(float(v) for v in cam_pos),
                                       axes_length=0.25, axes_radius=0.012)

            # --- Point cloud: TIME-throttled so it can never starve the RGB feed ---
            if now - last_cloud_push >= args.cloud_period:
                last_cloud_push = now
                cloud = sensor.get_cloud()
                if cloud is not None and cloud["n"] > 0:
                    pts, cols = parse_slam_cloud(cloud)
                    # Drop far/noisy outliers (each scan can span >13 m) — keep points
                    # within the GUI range of the current camera position.
                    if cam_pos is not None:
                        d = np.linalg.norm(pts - cam_pos, axis=1)
                        keep = d <= float(gui_range.value)
                        pts, cols = pts[keep], cols[keep]
                    if gui_mode.value == "Latest frame":
                        map_pts, map_cols = voxel_downsample(pts, cols, args.voxel)
                    else:
                        map_pts = np.vstack((map_pts, pts))
                        map_cols = np.vstack((map_cols, cols))
                        map_pts, map_cols = voxel_downsample(map_pts, map_cols, args.voxel)
                        if len(map_pts) > args.max_points:
                            keep = np.random.choice(len(map_pts), args.max_points, replace=False)
                            map_pts, map_cols = map_pts[keep], map_cols[keep]
                    server.scene.add_point_cloud(
                        "/odin_map", points=map_pts, colors=map_cols, point_size=args.voxel
                    )

            if now - last_status >= 0.5:
                last_status = now
                c = sensor.counts()
                gui_status.content = (
                    f"**Odin live** (mapping + loop closure)\n\n"
                    f"- map points: {len(map_pts):,}  ({gui_mode.value})\n"
                    f"- rgb={c['rgb']} cloud={c['cloud']} odom={c['odom']} imu={c['imu']}\n"
                    f"- connected: {sensor.connected()}"
                )
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("[viser-test] shutting down…")
    finally:
        sensor.stop()
        # The Odin SDK double-frees in its static-destructor teardown at exit; skip it.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
