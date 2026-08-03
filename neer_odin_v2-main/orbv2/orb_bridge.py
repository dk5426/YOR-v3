#!/usr/bin/env python3
"""
orb_bridge.py — ZMQ bridge between the ZED publisher and ORB-SLAM3.

Subscribes to ``zed/image`` and ``zed/depth`` from zed_pub_node over ZMQ,
feeds synchronized RGB-D pairs to the ORB-SLAM3 ``orb_pipe`` binary, and
publishes the resulting camera pose as ``orb/pose`` on a separate ZMQ port.

This is a cleaned-up, simplified version of the legacy orbslam_bridge.py:
  - Only supports the ``orb_pipe`` binary (the fast, persistent path)
  - Uses identity frame alignment (ORB and EKF both start at origin)
  - Better health monitoring and diagnostics
  - Auto-generates ORB-SLAM3 config on startup if missing

Usage:
    # Start zed_pub_node first, then:
    python -m orbv2.orb_bridge

    # With explicit paths:
    python -m orbv2.orb_bridge \\
        --orb-bin ~/ORB_SLAM3/Examples/RGB-D/orb_pipe \\
        --vocab   robot/nav/orbslam3/ORBvoc.txt \\
        --config  /tmp/orb_zed.yaml
"""

from __future__ import annotations

import argparse
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── Topic names (must match zed_pub_node.py and slam_node_.py) ────────────────
IMAGE_TOPIC       = "zed/image"
DEPTH_TOPIC       = "zed/depth"
POSE_TOPIC        = "zed/pose"
CAMERA_INFO_TOPIC = "zed/camera_info"
ORB_POSE_TOPIC    = "orb/pose"

ORB_PUB_PORT      = 6001   # separate port so zed_pub_node is untouched
ZED_PUB_PORT      = 6000


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers: quaternion / matrix conversions (no scipy dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _invert_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4×4 rigid transform without linalg.inv."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [qx, qy, qz, qw] (Shepperd's method)."""
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w, x = 0.25 / s, (m[2, 1] - m[1, 2]) * s
        y, z = (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] → 3×3 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    return np.array([
        [1.0 - (yy + zz), xy - wz,          xz + wy         ],
        [xy + wz,          1.0 - (xx + zz), yz - wx          ],
        [xz - wy,          yz + wx,          1.0 - (xx + yy) ],
    ], dtype=np.float64)


def _orb_to_yup(T_orb: np.ndarray) -> np.ndarray:
    """Convert an ORB-SLAM3 camera pose (Z-forward, Y-down) to Y-up.

    ORB-SLAM3 camera convention: X=right, Y=down, Z=forward
    Target (ZED Y-up):           X=right, Y=up,   Z=backward

    The conversion rotates 180° around the X axis.
    """
    R_conv = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ], dtype=np.float64)
    T_out = T_orb.copy()
    T_out[:3, :3] = R_conv @ T_orb[:3, :3] @ R_conv.T
    T_out[:3,  3] = R_conv @ T_orb[:3, 3]
    return T_out


# ══════════════════════════════════════════════════════════════════════════════
#  OrbPipeBackend — persistent orb_pipe binary interface
# ══════════════════════════════════════════════════════════════════════════════

class OrbPipeBackend:
    """Drives the ORB-SLAM3 ``orb_pipe`` binary via binary stdin/stdout.

    Protocol (little-endian binary):
      STDIN per frame:
        4B  magic  = b"ORBS"
        4B  width  (uint32)
        4B  height (uint32)
        8B  timestamp_s (float64)
        W*H*3 B   RGB uint8
        W*H*4 B   depth float32 (metres, 0 = invalid)

      STDOUT per frame:
        1B  tracking_ok  (uint8: 1=OK, 0=LOST/INIT)
        128B  4×4 pose  (float64, row-major, identity if tracking_ok=0)
    """

    _MAGIC = b"ORBS"

    def __init__(self, orb_pipe_path: str, vocab: str, config: str):
        self._proc: Optional[subprocess.Popen] = None
        self._out_q: queue.Queue = queue.Queue(maxsize=4)

        orb_pipe = Path(orb_pipe_path)
        if not orb_pipe.exists():
            raise FileNotFoundError(
                f"orb_pipe binary not found: {orb_pipe}\n"
                f"Build ORB-SLAM3 first, or use --orb-bin /path/to/orb_pipe"
            )

        cmd = [str(orb_pipe), vocab, config, "1"]  # "1" = no viewer
        print(f"[OrbPipeBackend] Launching: {' '.join(cmd)}")

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            bufsize=0,
        )

        self._reader_thr = threading.Thread(
            target=self._reader_loop, name="orb-reader", daemon=True
        )
        self._reader_thr.start()

        # Vocabulary loading takes ~15-20s on first run
        print("[OrbPipeBackend] Waiting for ORB-SLAM3 to initialize (~20s) …")
        time.sleep(20.0)
        print("[OrbPipeBackend] Initialization wait complete.")

    def _reader_loop(self):
        """Continuously read (tracking_ok, 4×4 pose) from child stdout."""
        POSE_BYTES = 1 + 16 * 8   # 1 byte flag + 16 doubles
        while self._proc is not None and self._proc.poll() is None:
            try:
                raw = self._proc.stdout.read(POSE_BYTES)
                if len(raw) < POSE_BYTES:
                    break
                ok_flag = raw[0]
                T = np.frombuffer(raw[1:], dtype=np.float64).reshape(4, 4).copy()
                try:
                    self._out_q.put_nowait((bool(ok_flag), T))
                except queue.Full:
                    # Drop oldest, enqueue newest
                    try:
                        self._out_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._out_q.put_nowait((bool(ok_flag), T))
                    except queue.Full:
                        pass
            except Exception as exc:
                print(f"[OrbPipeBackend] Reader error: {exc}", file=sys.stderr)
                break

    def track(self, rgb: np.ndarray, depth: np.ndarray, ts: float) -> Optional[np.ndarray]:
        """Send a frame to ORB-SLAM3; return 4×4 Tcw pose or None if lost."""
        if self._proc is None or self._proc.poll() is not None:
            return None

        H, W = rgb.shape[:2]
        depth_f32 = np.ascontiguousarray(depth, dtype=np.float32)
        rgb_u8    = np.ascontiguousarray(rgb,   dtype=np.uint8)

        header = self._MAGIC + struct.pack("<IId", W, H, float(ts))
        try:
            stdin = self._proc.stdin
            stdin.write(header)
            stdin.write(rgb_u8.tobytes())
            stdin.write(depth_f32.tobytes())
            stdin.flush()
        except BrokenPipeError:
            print("[OrbPipeBackend] Child process pipe broken.", file=sys.stderr)
            return None

        try:
            ok, T = self._out_q.get(timeout=0.5)
            return T if ok else None
        except queue.Empty:
            return None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3.0)
            except Exception:
                self._proc.kill()
            self._proc = None


# ══════════════════════════════════════════════════════════════════════════════
#  OrbBridge — main bridge class
# ══════════════════════════════════════════════════════════════════════════════

class OrbBridge:
    """Subscribes to ZED RGB-D, feeds ORB-SLAM3, publishes orb/pose.

    ``orb/pose`` message layout (list of Python floats, length 9):
        [qx, qy, qz, qw, tx, ty, tz, ts_ns, tracking_ok]
         0   1   2   3   4   5   6   7       8
    """

    def __init__(
        self,
        zed_host:    str   = "127.0.0.1",
        zed_port:    int   = ZED_PUB_PORT,
        pub_port:    int   = ORB_PUB_PORT,
        orb_pipe:    str   = "",
        vocab:       str   = "",
        config:      str   = "/tmp/orb_zed.yaml",
        skip_frames: int   = 2,
        max_depth_m: float = 5.0,
    ):
        self._skip      = max(1, int(skip_frames))
        self._max_depth = float(max_depth_m)
        self._stop_evt  = threading.Event()
        self._frame_count = 0
        self._pub_port  = pub_port

        # ── ZMQ subscriber ────────────────────────────────────────────────────
        from commlink import Subscriber, Publisher

        self._sub = Subscriber(
            host=zed_host, port=zed_port,
            topics=[IMAGE_TOPIC, DEPTH_TOPIC, POSE_TOPIC],
            buffer=True,
            queue_size=60,
        )

        self._pub = Publisher("*", port=pub_port)
        print(f"[OrbBridge] Publishing orb/pose on port {pub_port}")

        # ── Backend ───────────────────────────────────────────────────────────
        self._backend = OrbPipeBackend(orb_pipe, vocab, config)

        # ── Diagnostics ───────────────────────────────────────────────────────
        self._frames_ok   = 0
        self._frames_lost = 0
        self._last_log_t  = 0.0
        self._gf_last_log = 0.0
        self._gf_calls    = 0
        self._gf_ok       = 0

    def _get_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Fetch the latest RGB, depth from the subscriber.

        Returns (rgb, depth, ts_ns) or None on timeout/error.
        """
        now = time.time()
        self._gf_calls += 1

        try:
            img_msg   = self._sub[IMAGE_TOPIC]
            depth_msg = self._sub[DEPTH_TOPIC]
        except Exception as exc:
            if (now - self._gf_last_log) > 3.0:
                self._gf_last_log = now
                print(f"[OrbBridge] subscriber raised: {exc!r}")
            return None

        if img_msg is None or depth_msg is None:
            if (now - self._gf_last_log) > 5.0:
                self._gf_last_log = now
                print(
                    f"[OrbBridge] Waiting for ZED frames — "
                    f"img={'OK' if img_msg else 'None'} "
                    f"depth={'OK' if depth_msg else 'None'}"
                )
            return None

        rgb   = img_msg.get("image")
        depth = depth_msg.get("depth")
        if rgb is None or depth is None:
            return None

        self._gf_ok += 1
        ts_ns = float(img_msg.get("timestamp", time.time_ns()))
        return rgb, depth, ts_ns

    def _publish_pose(
        self, T_world: np.ndarray, ts_ns: float, tracking_ok: bool
    ) -> None:
        """Publish an ORB-SLAM3 pose on the orb/pose ZMQ topic."""
        if not np.all(np.isfinite(T_world)):
            print("[OrbBridge] dropping pose: T_world has NaN/Inf")
            return
        quat = _mat_to_quat(T_world[:3, :3])
        t    = T_world[:3, 3]
        msg  = [
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
            float(t[0]),    float(t[1]),    float(t[2]),
            float(ts_ns),
            1.0 if tracking_ok else 0.0,
        ]
        self._pub.publish(ORB_POSE_TOPIC, msg)

    def run(self) -> None:
        """Block and process frames until stop() or Ctrl+C."""
        print("[OrbBridge] Starting main loop. Press Ctrl+C to stop.")

        try:
            while not self._stop_evt.is_set():
                result = self._get_frame()
                if result is None:
                    time.sleep(0.01)
                    continue

                rgb, depth, ts_ns = result
                self._frame_count += 1

                # Frame skip for CPU budget
                if self._frame_count % self._skip != 0:
                    continue

                # Clip depth: ORB-SLAM3 RGB-D uses depth==0 for invalid
                depth = np.where(
                    np.isfinite(depth) & (depth > 0.05) & (depth < self._max_depth),
                    depth,
                    0.0,
                ).astype(np.float32)

                ts_s = ts_ns / 1e9
                T_cw = self._backend.track(rgb, depth, ts_s)

                if T_cw is None:
                    self._frames_lost += 1
                    self._publish_pose(np.eye(4), ts_ns, tracking_ok=False)
                    continue

                # Sanity check before SE(3) inverse
                T_cw_t_norm = float(np.linalg.norm(T_cw[:3, 3])) if np.all(np.isfinite(T_cw)) else float("inf")
                T_cw_rot_max = float(np.max(np.abs(T_cw[:3, :3]))) if np.all(np.isfinite(T_cw[:3, :3])) else float("inf")

                if (
                    not np.all(np.isfinite(T_cw))
                    or T_cw_t_norm > 100.0
                    or T_cw_rot_max > 1.5
                ):
                    if self._frame_count % 30 == 1:
                        print(
                            f"[OrbBridge] rejecting bad T_cw "
                            f"|t|={T_cw_t_norm:.2e} m  max|R|={T_cw_rot_max:.2e}"
                        )
                    self._frames_lost += 1
                    self._publish_pose(np.eye(4), ts_ns, tracking_ok=False)
                    continue

                # orb_pipe returns T_cw (world→camera). Invert to get T_wc.
                T_cam = _invert_se3(T_cw)

                if not np.all(np.isfinite(T_cam)) or float(np.linalg.norm(T_cam[:3, 3])) > 50.0:
                    self._frames_lost += 1
                    self._publish_pose(np.eye(4), ts_ns, tracking_ok=False)
                    continue

                # Convert ORB pose to Y-up world frame
                T_yup = _orb_to_yup(T_cam)

                # Identity alignment: ORB and EKF both start at origin
                self._publish_pose(T_yup, ts_ns, tracking_ok=True)
                self._frames_ok += 1

                # Periodic status log
                now = time.time()
                if now - self._last_log_t > 5.0:
                    total = self._frames_ok + self._frames_lost
                    ratio = (self._frames_ok / total * 100) if total > 0 else 0
                    print(
                        f"[OrbBridge] frames={total}  "
                        f"ok={self._frames_ok}  lost={self._frames_lost}  "
                        f"tracking_ratio={ratio:.0f}%  "
                        f"backend_alive={self._backend.alive}"
                    )
                    self._last_log_t = now

        except KeyboardInterrupt:
            print("\n[OrbBridge] Ctrl+C — shutting down.")
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self._backend.stop()
        except Exception:
            pass
        try:
            self._sub.stop()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  Default paths
# ══════════════════════════════════════════════════════════════════════════════

def _default_orb_pipe() -> str:
    return os.path.expanduser("~/ORB_SLAM3/Examples/RGB-D/orb_pipe")


def _default_vocab() -> str:
    """Look for ORBvoc.txt in the project tree."""
    candidates = [
        Path(__file__).parent.parent / "robot" / "nav" / "orbslam3" / "ORBvoc.txt",
        Path(__file__).parent / "ORBvoc.txt",
        Path.home() / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])  # let the error be caught later


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="orbv2 bridge: ZED frames → ORB-SLAM3 → orb/pose topic"
    )
    parser.add_argument(
        "--orb-bin", default=_default_orb_pipe(),
        help="Path to the compiled orb_pipe binary",
    )
    parser.add_argument(
        "--vocab", default=_default_vocab(),
        help="Path to ORBvoc.txt vocabulary file",
    )
    parser.add_argument(
        "--config", default="/tmp/orb_zed.yaml",
        help="Path to the ORB-SLAM3 camera config YAML",
    )
    parser.add_argument(
        "--zed-host", default="127.0.0.1",
        help="Host running zed_pub_node",
    )
    parser.add_argument(
        "--zed-port", type=int, default=ZED_PUB_PORT,
        help="ZMQ port for zed_pub_node",
    )
    parser.add_argument(
        "--pub-port", type=int, default=ORB_PUB_PORT,
        help=f"ZMQ port to publish orb/pose on (default: {ORB_PUB_PORT})",
    )
    parser.add_argument(
        "--skip", type=int, default=2,
        help="Process every Nth frame (default: 2 = 15 fps at 30fps input)",
    )
    parser.add_argument(
        "--max-depth", type=float, default=5.0,
        help="Clip depth values beyond this (metres)",
    )
    parser.add_argument(
        "--gen-config", action="store_true",
        help="Auto-generate the ORB-SLAM3 config from live ZED camera info",
    )
    args = parser.parse_args()

    # Auto-generate config if requested or if config file doesn't exist
    if args.gen_config or not os.path.isfile(args.config):
        if not os.path.isfile(args.config):
            print(f"[OrbBridge] Config not found at {args.config} — auto-generating …")
        from orbv2.config import generate_config
        generate_config(
            output_path=args.config,
            zed_host=args.zed_host,
            zed_port=args.zed_port,
        )

    # Validate paths
    if not os.path.isfile(args.orb_bin):
        print(
            f"[OrbBridge] ERROR: orb_pipe binary not found: {args.orb_bin}\n"
            f"  Build ORB-SLAM3 first, or use --orb-bin /path/to/orb_pipe",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(args.vocab):
        print(
            f"[OrbBridge] ERROR: Vocabulary not found: {args.vocab}\n"
            f"  Run: bash robot/nav/orbslam3/download_vocab.sh",
            file=sys.stderr,
        )
        sys.exit(1)

    bridge = OrbBridge(
        zed_host    = args.zed_host,
        zed_port    = args.zed_port,
        pub_port    = args.pub_port,
        orb_pipe    = args.orb_bin,
        vocab       = args.vocab,
        config      = args.config,
        skip_frames = args.skip,
        max_depth_m = args.max_depth,
    )
    bridge.run()


if __name__ == "__main__":
    main()
