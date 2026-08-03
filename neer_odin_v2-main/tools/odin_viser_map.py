#!/usr/bin/env python3
"""Odin 1 live 3D mapper — NO ROS. Same engine/accuracy as the ROS bridge.

Data flows straight from the Odin native SDK (via the pyodin pybind bridge)
into the same incremental voxel-map engine used by the Odin-Nav-Stack Viser
bridge (robot/nav/mapping/voxel_map_np.py): log-odds occupancy, ZED-style
dynamic-object carving, walking-trail decay, chunked real-time streaming.

    conda activate slam-odin
    cd ~/neer_odin
    python tools/odin_viser_map.py          # open http://localhost:8080

NOTE: one consumer at a time — stop any ROS driver before running this.
"""

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import viser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot.nav.mapping.voxel_map_np import VoxelMap, quat_to_rot  # noqa: E402

import pyodin  # noqa: E402


def device_ts_to_sec(ts):
    """Device timestamps may be ns/us/ms depending on stream — normalize to s."""
    ts = float(ts)
    if ts > 1e14:
        return ts / 1e9
    if ts > 1e11:
        return ts / 1e6
    if ts > 1e8:
        return ts / 1e3
    return ts


def decode_rgb(rgb, max_w=800):
    """pyodin RGB dict -> HxWx3 uint8 RGB (JPEG or NV12)."""
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


def save_ply(path, pts, cols):
    """Write a colored binary PLY (opens in MeshLab/CloudCompare/Open3D)."""
    n = len(pts)
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        rec.tofile(f)
    return n


def parse_cloud(cloud):
    """pyodin cloud dict -> (pts (n,3) f32 world Z-up meters, cols (n,3) u8)."""
    arr = np.asarray(cloud["xyzrgba"], dtype=np.int32)
    if arr.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    pts = arr[:, :3].astype(np.float32) * 1.0e-4      # 0.1 mm -> m
    cols = (arr[:, 3:6] & 0xFF).astype(np.uint8)
    m = np.isfinite(pts).all(axis=1)
    return pts[m], cols[m]


def main():
    ap = argparse.ArgumentParser("Odin live mapper (non-ROS)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--voxel", type=float, default=0.04)
    ap.add_argument("--max-range", type=float, default=5.0)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--carve-every", type=int, default=1)
    ap.add_argument("--carve-refresh", type=float, default=1.0)
    ap.add_argument("--carve-range", type=float, default=4.5)
    ap.add_argument("--promote-age", type=int, default=15)
    ap.add_argument("--trail-ttl", type=float, default=2.0)
    ap.add_argument("--max-points", type=int, default=1_500_000)
    ap.add_argument("--flush-period", type=float, default=0.5)
    ap.add_argument("--compact-at", type=int, default=100)
    ap.add_argument("--point-shape", choices=("square", "circle", "rounded", "diamond", "sparkle"),
                    default="square")
    ap.add_argument("--point-scale", type=float, default=1.2)
    ap.add_argument("--shading", choices=("gradient", "flat"), default="gradient")
    ap.add_argument("--map-mode", type=int, default=1,
                    help="device SLAM mode: 0 odometry, 1 mapping+loop-closure, 2 reloc")
    ap.add_argument("--save-dir", type=str, default="maps",
                    help="where 'Save map' writes PLY / device .bin files")
    ap.add_argument("--reloc-map", type=str, default=None,
                    help="device .bin map to relocalize against (use with --map-mode 2)")
    ap.add_argument("--wait", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- sensor (native SDK, no ROS) ----
    print("[mapper] starting Odin SDK (SLAM mode)…")
    sensor = pyodin.OdinSensor()
    if not sensor.start("slam", args.wait, args.map_mode):
        print("Odin did not connect — power-cycle the box / check USB3 / stop other consumers.")
        os._exit(1)
    print(f"[mapper] connected. counts={sensor.counts()}")
    if args.reloc_map:
        rc = sensor.set_relocalization_map(os.path.abspath(args.reloc_map))
        print(f"[mapper] set_relocalization_map({args.reloc_map}) -> rc={rc}"
              + ("" if rc == 0 else "  (nonzero = upload failed)"))

    # ---- viser ----
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/world", show_axes=True, axes_length=0.5, axes_radius=0.01)
    server.initial_camera.up = (0, 0, 1)
    server.initial_camera.position = (-3, 0, 1.5)
    server.initial_camera.look_at = (1, 0, 0)
    gui_image = server.gui.add_image(np.zeros((480, 640, 3), np.uint8), label="Odin RGB")
    gui_mode = server.gui.add_dropdown("Cloud display", ("Accumulate", "Latest frame"),
                                       initial_value="Accumulate")
    gui_range = server.gui.add_slider("Max range (m)", min=1.0, max=15.0, step=0.5,
                                      initial_value=float(args.max_range))
    gui_hits = server.gui.add_slider("Min hits (denoise)", min=1, max=8, step=1,
                                     initial_value=int(args.min_hits))
    gui_dyn = server.gui.add_checkbox("Dynamic removal (ZED carving)", initial_value=bool(args.dynamic))
    gui_ttl = server.gui.add_slider("Trail decay TTL (s)", min=0.5, max=8.0, step=0.5,
                                    initial_value=float(args.trail_ttl))
    with server.gui.add_folder("Appearance"):
        gui_shape = server.gui.add_dropdown(
            "Point shape", ("square", "circle", "rounded", "diamond", "sparkle"),
            initial_value=args.point_shape)
        gui_psize = server.gui.add_slider("Point size (×voxel)", min=0.4, max=2.5, step=0.1,
                                          initial_value=float(args.point_scale))
        gui_flat = server.gui.add_checkbox("Flat shading", initial_value=args.shading == "flat")
    gui_status = server.gui.add_markdown("waiting for sensor data…")
    reset_flag = {"v": False}
    style_dirty = {"v": False}
    save_req = {"ply": False, "bin": False}
    btn = server.gui.add_button("Reset map")

    @btn.on_click
    def _(_=None):
        reset_flag["v"] = True

    with server.gui.add_folder("Save"):
        btn_ply = server.gui.add_button("Save map (PLY)")
        btn_bin = server.gui.add_button("Save device map (.bin)")
        gui_save = server.gui.add_markdown("_no map saved yet_")

    @btn_ply.on_click
    def _(_=None):
        save_req["ply"] = True

    @btn_bin.on_click
    def _(_=None):
        save_req["bin"] = True

    @gui_shape.on_update
    def _(_=None):
        style_dirty["v"] = True

    @gui_psize.on_update
    def _(_=None):
        style_dirty["v"] = True

    @gui_flat.on_update
    def _(_=None):
        style_dirty["v"] = True

    def style():
        return dict(point_size=args.voxel * float(gui_psize.value),
                    point_shape=gui_shape.value,
                    point_shading="flat" if gui_flat.value else "gradient")

    chunk_handles = []
    latest_handle = [None]

    def add_chunk(pts, cols):
        h = server.scene.add_point_cloud(
            f"/odin_map/c{len(chunk_handles)}_{int(time.time()*1000)%100000}",
            points=pts, colors=cols, **style())
        chunk_handles.append(h)

    def clear_chunks():
        for h in chunk_handles:
            try:
                h.remove()
            except Exception:
                pass
        chunk_handles.clear()

    def rebuild_view(vmap):
        pts, cols = vmap.alive_points()
        clear_chunks()
        if len(pts):
            add_chunk(pts, cols)

    # ---- main loop ----
    vmap = VoxelMap(args.voxel, args.carve_range)
    odom_hist = deque(maxlen=200)          # (stamp_s, wxyz, pos)
    pending_pts, pending_cols = [], []
    last_cloud_ts = -1
    last_rgb_ts = -1
    last_flush = 0.0
    last_view_rebuild = 0.0
    last_decay = 0.0
    last_status = 0.0
    carved_since_view = 0
    frame_i = 0

    try:
        while True:
            loop_t0 = time.time()

            if reset_flag["v"]:
                vmap = VoxelMap(args.voxel, args.carve_range)
                pending_pts, pending_cols = [], []
                carved_since_view = 0
                clear_chunks()
                if latest_handle[0] is not None:
                    try:
                        latest_handle[0].remove()
                    except Exception:
                        pass
                    latest_handle[0] = None
                reset_flag["v"] = False

            # ---- odometry: poll fast, keep a stamped history ----
            od = sensor.get_odom()
            pose = None
            if od is not None:
                q = np.asarray(od["quat_xyzw"], float)
                wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
                pos = tuple(float(v) for v in od["pos"])
                stamp = device_ts_to_sec(od["timestamp_ns"])
                if not odom_hist or odom_hist[-1][0] != stamp:
                    odom_hist.append((stamp, wxyz, pos))
                pose = (wxyz, pos)
                server.scene.add_frame("/odin_cam", wxyz=wxyz, position=pos,
                                       axes_length=0.25, axes_radius=0.012)

            # ---- RGB (new frames only) ----
            rgb = sensor.get_rgb()
            if rgb is not None and rgb["timestamp"] != last_rgb_ts:
                last_rgb_ts = rgb["timestamp"]
                img = decode_rgb(rgb)
                if img is not None:
                    gui_image.image = img

            # ---- cloud: ingest new frames ----
            cloud = sensor.get_cloud()
            if cloud is not None and cloud["timestamp"] != last_cloud_ts and cloud["n"] > 0:
                last_cloud_ts = cloud["timestamp"]
                frame_i += 1
                pts_full, cols_full = parse_cloud(cloud)   # world frame Z-up

                # pose paired to the cloud's stamp (for the carver + range gate)
                m = pose
                if odom_hist:
                    cs = device_ts_to_sec(cloud["timestamp"])
                    best = min(odom_hist, key=lambda h: abs(h[0] - cs))
                    if abs(best[0] - cs) < 0.5:            # same clock -> use pairing
                        m = (best[1], best[2])

                pts_ins, cols_ins = pts_full, cols_full
                if m is not None:
                    d = np.linalg.norm(pts_full - np.asarray(m[1], np.float32), axis=1)
                    keep = d <= float(gui_range.value)
                    pts_ins, cols_ins = pts_full[keep], cols_full[keep]

                if gui_mode.value == "Latest frame":
                    if latest_handle[0] is not None:
                        try:
                            latest_handle[0].remove()
                        except Exception:
                            pass
                    latest_handle[0] = server.scene.add_point_cloud(
                        "/odin_latest", points=pts_ins, colors=cols_ins, **style())
                else:
                    if vmap.n_alive < args.max_points:
                        new = vmap.update(pts_ins, cols_ins, int(gui_hits.value), now=time.time())
                        if new is not None:
                            pending_pts.append(new[0])
                            pending_cols.append(new[1])
                    if gui_dyn.value and m is not None and frame_i % args.carve_every == 0:
                        carved_since_view += vmap.carve(pts_full, m[0], m[1])

            # ---- map saving ----
            if save_req["ply"]:
                save_req["ply"] = False
                pts_s, cols_s = vmap.alive_points()
                if len(pts_s):
                    name = time.strftime("odin_map_%Y%m%d_%H%M%S.ply")
                    path = os.path.join(args.save_dir, name)
                    n = save_ply(path, pts_s, cols_s)
                    gui_save.content = f"saved **{name}** ({n:,} pts)"
                    print(f"[mapper] saved {path} ({n:,} pts)")
                else:
                    gui_save.content = "_map empty — nothing to save_"
            if save_req["bin"]:
                save_req["bin"] = False
                name = time.strftime("odin_reloc_%Y%m%d_%H%M%S.bin")
                gui_save.content = f"pulling device map… (takes up to ~2 min, streaming may pause)"
                rc = sensor.save_map(os.path.abspath(args.save_dir), name, 120000)
                if rc == 0:
                    gui_save.content = f"saved device map **{name}** — reuse with `--map-mode 2 --reloc-map {args.save_dir}/{name}`"
                    print(f"[mapper] saved device map {args.save_dir}/{name}")
                else:
                    gui_save.content = f"_device map save failed (rc={rc})_"
                    print(f"[mapper] device map save failed rc={rc}")

            now = time.time()
            if style_dirty["v"]:
                style_dirty["v"] = False
                pending_pts, pending_cols = [], []
                rebuild_view(vmap)
                last_view_rebuild = now
                last_flush = now

            if gui_dyn.value and now - last_decay >= 0.5:
                last_decay = now
                carved_since_view += vmap.decay(now, args.promote_age, float(gui_ttl.value))

            stale_ok = carved_since_view > 0 and now - last_view_rebuild >= 3 * args.carve_refresh
            if (carved_since_view >= 50 or stale_ok) and \
                    now - last_view_rebuild >= args.carve_refresh:
                last_view_rebuild = now
                pending_pts, pending_cols = [], []
                rebuild_view(vmap)
                carved_since_view = 0
                last_flush = now
            elif pending_pts and now - last_flush >= args.flush_period:
                last_flush = now
                add_chunk(np.vstack(pending_pts), np.vstack(pending_cols))
                pending_pts, pending_cols = [], []
                if len(chunk_handles) > args.compact_at:
                    rebuild_view(vmap)

            if now - last_status >= 1.0:
                last_status = now
                c = sensor.counts()
                full = " — MAP FULL (Reset to continue)" if vmap.n_alive >= args.max_points else ""
                gui_status.content = (
                    f"**Odin live map (no ROS)** ({gui_mode.value})\n\n"
                    f"- map voxels: {vmap.n_alive:,} in {len(chunk_handles)} chunks{full}\n"
                    f"- dynamics carved: {vmap.carved_total:,} · trails decayed: {vmap.decayed_total:,}\n"
                    f"- sensor: rgb={c['rgb']} cloud={c['cloud']} odom={c['odom']} imu={c['imu']}\n"
                    f"- connected: {sensor.connected()}"
                )

            time.sleep(max(0.0, 0.03 - (time.time() - loop_t0)))
    except KeyboardInterrupt:
        print("[mapper] shutting down…")
    finally:
        sensor.stop()
        # The Odin SDK double-frees in static destructors at exit; hard-exit past it.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
