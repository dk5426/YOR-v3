#!/usr/bin/env python3
"""
orbslam_bridge.py — ZMQ bridge between the ZED pub node and ORB-SLAM3.

Architecture
------------
This script runs as a **standalone subprocess** (not imported as a library):

  1. Subscribe to ``zed/image`` and ``zed/depth`` from zed_pub_node over ZMQ.
  2. For each synchronized RGB-D pair, feed them to the ORB-SLAM3 system via
     the ``orbslam3`` Python bindings  *or*  the ``mpipe`` subprocess interface
     (selectable via ``--backend``).
  3. Parse the resulting 4×4 camera-to-world matrix returned by ORB-SLAM3.
  4. Publish the robot-base pose as a new ZMQ topic ``orb/pose``
     (layout: ``[qx, qy, qz, qw, tx, ty, tz, ts_ns, tracking_ok]``).

The SLAM node consumes ``orb/pose`` as an optional second EKF measurement.
If this process is not running, the SLAM node falls back to ZED-only mode.

Backend options
---------------
``--backend zmq`` (default)
    Spawns ORB-SLAM3 as a child process using a custom stdin/stdout pipe
    protocol (see _OrbProcessBackend).  No Python bindings required — only
    the ORB-SLAM3 RGB-D binary must be compiled.

``--backend bindings``
    Uses the ``orbslam3`` Python package (pip install orbslam3 or build from
    https://github.com/fabiovalse/orbslam3-python).  Faster (in-process) but
    requires native compilation.

Usage
-----
    # Start zed_pub_node first, then in another terminal:
    python -m robot.nav.odometry.orbslam_bridge \\
        --orb-bin ~/ORB_SLAM3/Examples/RGB-D/rgbd_tum \\
        --vocab   ~/ORB_SLAM3/Vocabulary/ORBvoc.txt \\
        --config  /tmp/orb_zed.yaml

    # Generate config first if needed:
    python -m robot.nav.orbslam3.gen_config --out /tmp/orb_zed.yaml
"""

from __future__ import annotations

import argparse
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── Topic names (must match zed_pub_node.py and slam_node_.py) ───────────────
IMAGE_TOPIC      = "zed/image"
DEPTH_TOPIC      = "zed/depth"
POSE_TOPIC       = "zed/pose"
CAMERA_INFO_TOPIC = "zed/camera_info"
ORB_POSE_TOPIC   = "orb/pose"

ORB_PUB_PORT     = 6001   # separate port so zed_pub_node is untouched

# ── Coordinate conventions ────────────────────────────────────────────────────
# ZED publishes in RIGHT_HANDED_Y_UP.
# ORB-SLAM3 RGB-D returns poses in its own world frame (right-handed, Z forward
# in camera frame at t=0).  We track the first-frame alignment transform and
# express all subsequent ORB poses in the ZED world frame.



# ══════════════════════════════════════════════════════════════════════════════
#  Helper: quaternion / matrix conversions (no scipy dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _invert_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4×4 rigid transform without going through linalg.inv."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [qx, qy, qz, qw]  (Shepperd's method)."""
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


def _orb_to_yup(T_orb: np.ndarray) -> np.ndarray:
    """Convert an ORB-SLAM3 camera pose (Z-forward, Y-down) to Y-up world frame.

    ORB-SLAM3 camera convention:
      X = right, Y = down, Z = forward

    Target (ZED Y-up, RIGHT_HANDED_Y_UP):
      X = right, Y = up, Z = backward (camera looks down −Z)

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
#  Frame alignment: register ORB world frame → ZED world frame
# ══════════════════════════════════════════════════════════════════════════════

class FrameAligner:
    """Alignment of the ORB-SLAM3 frame to the EKF world frame.

    Two modes, selectable via ``mode=``:

    - ``"identity"`` (default, recommended): both ORB-SLAM3 and the EKF start
      at the origin by construction (ORB's first keyframe is identity; the EKF
      is initialised at [0, 0, 0]). So no alignment transform is needed — we
      publish ORB poses directly in the Y-up world frame. This is robust to
      stale ZED area-memory offsets that would otherwise bake a large
      translation into the alignment.

    - ``"zed"``: aligns ORB to the first ZED pose. Use this only when you
      explicitly want ORB-SLAM3 to track the *ZED* world frame (e.g. when
      both share a saved area memory). Sensitive to ZED area-memory offsets:
      if the ZED reports e.g. (-268, -265) m at startup, that offset is
      transferred to every subsequent ORB pose.
    """

    def __init__(self, mode: str = "identity"):
        if mode not in ("identity", "zed"):
            raise ValueError(f"FrameAligner mode must be 'identity' or 'zed', got {mode!r}")
        self._mode = mode
        self._T_align: Optional[np.ndarray] = None  # 4×4
        self._lock = threading.Lock()
        if mode == "identity":
            # No alignment needed — publish ORB poses as-is in Y-up.
            self._T_align = np.eye(4, dtype=np.float64)

    @property
    def aligned(self) -> bool:
        with self._lock:
            return self._T_align is not None

    def register(self, T_zed: np.ndarray, T_orb_yup: np.ndarray) -> None:
        """Compute alignment from the first simultaneous pose pair (zed mode only).

        In identity mode this is a no-op — alignment is already set to I.
        """
        if self._mode == "identity":
            return
        with self._lock:
            if self._T_align is not None:
                return
            # T_align = T_zed @ inv(T_orb_yup)
            try:
                T_align = T_zed @ np.linalg.inv(T_orb_yup)
                self._T_align = T_align.astype(np.float64)
                t = T_align[:3, 3]
                print(
                    f"[FrameAligner] Alignment computed — translation offset: "
                    f"[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m"
                )
            except np.linalg.LinAlgError as exc:
                print(f"[FrameAligner] WARN: alignment failed ({exc}); will retry.", file=sys.stderr)

    def apply(self, T_orb_yup: np.ndarray) -> np.ndarray:
        """Express an ORB pose in the EKF world frame."""
        with self._lock:
            if self._T_align is None:
                return T_orb_yup
            return self._T_align @ T_orb_yup


# ══════════════════════════════════════════════════════════════════════════════
#  Backend A: ORB-SLAM3 Python bindings (in-process)
# ══════════════════════════════════════════════════════════════════════════════

class _BindingsBackend:
    """Wraps the ``orbslam3`` Python package for in-process pose estimation."""

    def __init__(self, vocab: str, config: str):
        import orbslam3  # type: ignore
        self._slam = orbslam3.System(
            vocab, config, orbslam3.Sensor.RGBD
        )
        self._slam.set_use_viewer(False)
        self._slam.initialize()
        print("[OrbBackend/bindings] ORB-SLAM3 initialized.")

    def track(self, rgb: np.ndarray, depth: np.ndarray, ts: float) -> Optional[np.ndarray]:
        """Feed an RGB-D frame; return 4×4 camera pose or None if lost."""
        import orbslam3  # type: ignore
        Tcw = self._slam.process_image_rgbd(rgb, depth, ts)
        if Tcw is None:
            return None
        return np.asarray(Tcw, dtype=np.float64)

    def stop(self):
        try:
            self._slam.shutdown()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  Backend B: ORB-SLAM3 as a subprocess (pipe protocol)
# ══════════════════════════════════════════════════════════════════════════════

class _ProcessBackend:
    """Runs ORB-SLAM3 as a child process, communicating via binary stdin/stdout.

    Protocol (little-endian binary):
      STDIN  per frame:
        4B  magic  = 0xORB5_LAM3 (uint32)
        4B  width  (uint32)
        4B  height (uint32)
        8B  timestamp_s (float64)
        W*H*3 B   RGB uint8
        W*H*4 B   depth float32 (metres, NaN = invalid)

      STDOUT per frame:
        1B  tracking_ok  (uint8: 1=OK, 0=LOST/INIT)
        16*8 B  4×4 pose (float64, row-major, identity if tracking_ok=0)
    """

    # Must match the C++ side: orb_pipe.cc declares
    #     static const uint8_t MAGIC[4] = {'O','R','B','S'};
    # i.e. raw bytes on the wire are O,R,B,S in order. Don't pack as
    # little-endian uint32 — that would reverse to S,B,R,O.
    _MAGIC = b"ORBS"

    def __init__(self, orb_bin: str, vocab: str, config: str, work_dir: str):
        self._proc: Optional[subprocess.Popen] = None
        self._out_q: queue.Queue = queue.Queue(maxsize=2)
        self._W = self._H = 0

        # Prefer the persistent pipe-native binary (orb_pipe) built alongside
        # the ORB-SLAM3 examples.  It stays alive for the entire session and
        # processes frames on-demand via stdin/stdout — no per-frame process
        # spawn overhead and no "pipe broken" race conditions.
        #
        # Fall back to the Python shim (which wraps rgbd_tum in a rolling-batch
        # mode) only if orb_pipe is not found.
        orb_pipe = Path(orb_bin).parent / "orb_pipe"
        if orb_pipe.exists():
            # orb_pipe.cc requires argc >= 4 (orb_pipe vocab config no_viewer).
            # Pass "1" for no_viewer to disable any GL/X11 viewer code paths.
            cmd = [
                str(orb_pipe),
                vocab,
                config,
                "1",
            ]
            print(f"[OrbBackend/process] Using persistent orb_pipe binary.")
            init_wait = 20.0   # vocabulary loading takes ~15-20 s on first run
        else:
            # Fall back: Python shim wrapping rgbd_tum (batch mode, ~1 Hz)
            shim = Path(__file__).parent / "_orbslam_shim.py"
            if not shim.exists():
                shim = self._write_shim(shim)
            cmd = [
                sys.executable, str(shim),
                "--orb-bin", orb_bin,
                "--vocab",   vocab,
                "--config",  config,
                "--work-dir", work_dir,
            ]
            print(f"[OrbBackend/process] orb_pipe not found — using Python shim.")
            init_wait = 5.0

        print(f"[OrbBackend/process] Launching: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,   # forward ORB-SLAM3 log to our stderr
            bufsize=0,
        )
        # Reader thread: parse stdout responses
        self._reader_thr = threading.Thread(
            target=self._reader_loop, name="orb-reader", daemon=True
        )
        self._reader_thr.start()
        print(f"[OrbBackend/process] Waiting for ORB-SLAM3 to initialize (~{init_wait:.0f}s) …")
        time.sleep(init_wait)

    # ── Shim writer ──────────────────────────────────────────────────────────

    @staticmethod
    def _write_shim(path: Path) -> Path:
        """Write the _orbslam_shim.py helper next to this file (first run only)."""
        shim_code = r'''#!/usr/bin/env python3
"""
_orbslam_shim.py — Thin wrapper that bridges ORB-SLAM3 binary ↔ binary pipe protocol.

This script is auto-generated by orbslam_bridge.py on first use.
It converts binary frames received on stdin into image files on disk,
calls the ORB-SLAM3 binary, and returns binary pose data on stdout.

For production use the ORB-SLAM3 binary must support:
  stdin frame feed + stdout pose output
OR this shim uses a shared temporary directory approach.
"""
import argparse, os, struct, sys, subprocess, tempfile, time
import numpy as np

MAGIC_EXPECTED = b"ORBS"

def read_frame(stdin):
    magic = stdin.read(4)
    if len(magic) < 4: return None
    if magic != MAGIC_EXPECTED: return None
    W, H = struct.unpack("<II", stdin.read(8))
    ts,  = struct.unpack("<d",  stdin.read(8))
    rgb   = np.frombuffer(stdin.read(W * H * 3), dtype=np.uint8).reshape(H, W, 3)
    depth = np.frombuffer(stdin.read(W * H * 4), dtype=np.float32).reshape(H, W)
    return W, H, ts, rgb, depth

def write_response(stdout, ok: bool, T: np.ndarray):
    stdout.write(struct.pack("<B", 1 if ok else 0))
    stdout.write(T.astype(np.float64).tobytes())
    stdout.flush()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orb-bin"); p.add_argument("--vocab")
    p.add_argument("--config");  p.add_argument("--work-dir", default="/tmp/orb_work")
    args = p.parse_args()

    import cv2
    work = args.work_dir
    os.makedirs(work, exist_ok=True)
    rgb_dir  = os.path.join(work, "rgb")
    dep_dir  = os.path.join(work, "depth")
    os.makedirs(rgb_dir,  exist_ok=True)
    os.makedirs(dep_dir,  exist_ok=True)

    # Write association file stub (updated per-frame)
    assoc = os.path.join(work, "associations.txt")
    pose_file = os.path.join(work, "orb_poses.txt")
    identity = np.eye(4, dtype=np.float64)
    stdin_b = sys.stdin.buffer
    stdout_b = sys.stdout.buffer

    frame_idx = 0
    while True:
        result = read_frame(stdin_b)
        if result is None: break
        W, H, ts, rgb, depth = result

        rgb_name  = f"rgb_{frame_idx:06d}.png"
        dep_name  = f"dep_{frame_idx:06d}.png"
        rgb_path  = os.path.join(rgb_dir,  rgb_name)
        dep_path  = os.path.join(dep_dir,  dep_name)

        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # ORB-SLAM3 TUM depth: uint16, scale = 1000 (1 unit = 1 mm)
        depth_mm = np.nan_to_num(depth, nan=0.0) * 1000.0
        depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        cv2.imwrite(dep_path, depth_mm)

        with open(assoc, "w") as f:
            f.write(f"{ts:.6f} rgb/{rgb_name} {ts:.6f} depth/{dep_name}\n")

        ret = subprocess.run(
            [args.orb_bin, args.vocab, args.config, assoc, pose_file],
            capture_output=True, cwd=work, timeout=2.0
        )
        T = identity.copy()
        ok = False
        if ret.returncode == 0 and os.path.exists(pose_file):
            try:
                lines = open(pose_file).readlines()
                if lines:
                    vals = list(map(float, lines[-1].split()))
                    if len(vals) >= 8:
                        # TUM format: ts qx qy qz qw tx ty tz
                        from scipy.spatial.transform import Rotation
                        q = vals[1:5]; t = vals[5:8]
                        T[:3,:3] = Rotation.from_quat(q).as_matrix()
                        T[:3,3]  = t
                        ok = True
            except Exception: pass
        write_response(stdout_b, ok, T)
        frame_idx += 1

if __name__ == "__main__": main()
'''
        path.write_text(shim_code)
        return path

    # ── Stdout reader ─────────────────────────────────────────────────────────

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
                    pass  # drop stale frame
            except Exception as exc:
                print(f"[OrbBackend] Reader error: {exc}", file=sys.stderr)
                break

    # ── Public interface ──────────────────────────────────────────────────────

    def track(self, rgb: np.ndarray, depth: np.ndarray, ts: float) -> Optional[np.ndarray]:
        """Send a frame to ORB-SLAM3; return 4×4 pose or None if lost."""
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
            print("[OrbBackend] Child process pipe broken.", file=sys.stderr)
            return None

        # Wait up to 500ms for the response
        try:
            ok, T = self._out_q.get(timeout=0.5)
            return T if ok else None
        except queue.Empty:
            return None

    def stop(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3.0)
            except Exception:
                self._proc.kill()
            self._proc = None


# ══════════════════════════════════════════════════════════════════════════════
#  OrbSlamBridge — main class
# ══════════════════════════════════════════════════════════════════════════════

class OrbSlamBridge:
    """
    Subscribes to ``zed/image`` and ``zed/depth`` over ZMQ,
    feeds frames to ORB-SLAM3, and publishes ``orb/pose`` over a new ZMQ port.

    ``orb/pose`` message layout (list of Python floats):
        [qx, qy, qz, qw, tx, ty, tz, ts_ns, tracking_ok]
         0   1   2   3   4   5   6   7        8

    ``tracking_ok`` is 1.0 when ORB-SLAM3 has a valid track, 0.0 otherwise.
    """

    def __init__(
        self,
        zed_host:    str   = "127.0.0.1",
        zed_port:    int   = 6000,
        pub_port:    int   = ORB_PUB_PORT,
        backend:     str   = "process",
        orb_bin:     str   = "",
        vocab:       str   = "",
        config:      str   = "/tmp/orb_zed.yaml",
        work_dir:    str   = "/tmp/orb_work",
        skip_frames: int   = 1,    # process every Nth frame (1 = every frame)
        max_depth_m: float = 5.0,  # clip depth beyond this value
        align:       str   = "identity",   # "identity" (recommended) or "zed"
    ):
        self._skip = max(1, int(skip_frames))
        self._max_depth = float(max_depth_m)
        self._aligner = FrameAligner(mode=align)
        print(f"[OrbBridge] Frame alignment mode: {align}")
        self._stop_evt = threading.Event()
        self._frame_count = 0
        self._pub_port = pub_port

        # ── ZMQ subscriber ────────────────────────────────────────────────────
        try:
            import zmq
            from commlink import Subscriber, Publisher
        except ImportError as exc:
            raise RuntimeError(
                "commlink / zmq not installed. Run: pip install commlink pyzmq"
            ) from exc

        self._sub = Subscriber(
            host=zed_host, port=zed_port,
            topics=[IMAGE_TOPIC, DEPTH_TOPIC, POSE_TOPIC],
            # buffer=True: we want every published RGB-D frame, not just the
            # latest. This ensures ORB-SLAM3 processes a continuous stream
            # rather than skipping frames between processing cycles.
            # (commlink 0.4.0 push mode delivers all frames into per-topic queues.)
            buffer=True,
            queue_size=60,   # ~2 s buffer at 30fps
        )
        # No socket timeout configuration needed — commlink 0.4.0 manages
        # its own I/O thread and queue; reads never block indefinitely.

        self._pub = Publisher("*", port=pub_port)
        print(f"[OrbBridge] Publishing orb/pose on port {pub_port}")

        # ── Backend ───────────────────────────────────────────────────────────
        if backend == "bindings":
            self._backend = _BindingsBackend(vocab, config)
        else:
            os.makedirs(work_dir, exist_ok=True)
            self._backend = _ProcessBackend(orb_bin, vocab, config, work_dir)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """Fetch the latest RGB, depth, and ZED base pose from the subscriber.

        Returns (rgb, depth, T_zed_4x4, ts_ns) or None on timeout/error.
        """
        # Diagnostic — log per-topic status every ~3 s so we can see why
        # _get_frame is returning None (subscriber not connected, one topic
        # missing, etc.).
        now = time.time()
        if not hasattr(self, "_gf_last_log"):
            self._gf_last_log = 0.0
            self._gf_calls = 0
            self._gf_ok = 0
        self._gf_calls += 1

        try:
            img_msg   = self._sub[IMAGE_TOPIC]
            depth_msg = self._sub[DEPTH_TOPIC]
            pose_msg  = self._sub[POSE_TOPIC]
        except Exception as exc:
            if (now - self._gf_last_log) > 3.0:
                self._gf_last_log = now
                print(f"[OrbBridge/get_frame] subscriber raised: {exc!r}")
            return None

        if (now - self._gf_last_log) > 3.0:
            self._gf_last_log = now
            def _shape(m, key=None):
                if m is None: return "None"
                if key:
                    v = m.get(key)
                    return f"{type(v).__name__}{getattr(v,'shape','')}"
                return type(m).__name__
            print(
                f"[OrbBridge/get_frame] calls={self._gf_calls}  ok={self._gf_ok}  "
                f"img={_shape(img_msg, 'image')}  depth={_shape(depth_msg, 'depth')}  "
                f"pose={_shape(pose_msg)}"
            )

        if img_msg is None or depth_msg is None or pose_msg is None:
            return None

        rgb   = img_msg.get("image")
        depth = depth_msg.get("depth")
        if rgb is None or depth is None:
            return None
        self._gf_ok += 1

        ts_ns = float(img_msg.get("timestamp", time.time_ns()))

        # Extract ZED camera pose (indices 7:14 = [qx,qy,qz,qw,tx,ty,tz])
        pose_arr = np.asarray(pose_msg, dtype=np.float64)
        if pose_arr.size < 14:
            return None
        cam_qt = pose_arr[7:14]
        from robot.slam_node_ import _quat_to_matrix
        T_zed = np.eye(4, dtype=np.float64)
        T_zed[:3, :3] = _quat_to_matrix(cam_qt[:4])
        T_zed[:3,  3] = cam_qt[4:7]

        return rgb, depth, T_zed, ts_ns

    def _publish_pose(
        self, T_world: np.ndarray, ts_ns: float, tracking_ok: bool
    ) -> None:
        """Publish an ORB-SLAM3 pose on the orb/pose ZMQ topic."""
        if not np.all(np.isfinite(T_world)):
            # Drop frame entirely — propagating NaN/Inf into the EKF corrupts
            # everything downstream (viser camera, A* start state, etc.).
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

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block and process frames until stop() is called or Ctrl+C."""
        print("[OrbBridge] Starting main loop. Press Ctrl+C to stop.")
        _last_log_t = 0.0
        _frames_ok  = 0
        _frames_lost = 0

        try:
            while not self._stop_evt.is_set():
                result = self._get_frame()
                if result is None:
                    time.sleep(0.01)
                    continue

                rgb, depth, T_zed, ts_ns = result
                self._frame_count += 1

                # Frame skip for CPU budget
                if self._frame_count % self._skip != 0:
                    continue

                # Clip depth. ORB-SLAM3 RGB-D follows TUM convention: depth==0
                # means invalid (NaN/Inf would propagate through ORB's depth
                # math and produce degenerate poses). Out-of-range values are
                # zeroed, not NaN'd.
                depth = np.where(
                    np.isfinite(depth) & (depth > 0.05) & (depth < self._max_depth),
                    depth,
                    0.0,
                ).astype(np.float32)

                ts_s = ts_ns / 1e9
                T_cw = self._backend.track(rgb, depth, ts_s)

                if T_cw is None:
                    # ORB-SLAM3 tracking lost — publish a "not ok" message at
                    # the ZED pose so the subscriber doesn't stall.
                    self._publish_pose(T_zed, ts_ns, tracking_ok=False)
                    _frames_lost += 1
                    continue

                # Sanity-reject ORB poses BEFORE running the SE(3) inverse —
                # an unstable initialisation can emit Tcw with translation
                # near ±1e30, which would overflow `-R.T @ t` inside the
                # inverter even though the inputs are technically finite.
                T_cw_t_norm = float(np.linalg.norm(T_cw[:3, 3])) if np.all(np.isfinite(T_cw)) else float("inf")
                T_cw_rot_max = float(np.max(np.abs(T_cw[:3, :3]))) if np.all(np.isfinite(T_cw[:3, :3])) else float("inf")
                if (
                    not np.all(np.isfinite(T_cw))
                    or T_cw_t_norm > 100.0     # any rigid pose should sit well inside this
                    or T_cw_rot_max > 1.5      # rotation entries are bounded by ±1 for SO(3)
                ):
                    if self._frame_count % 30 == 1:
                        print(
                            f"[OrbBridge] rejecting bad T_cw "
                            f"|t|={T_cw_t_norm:.2e} m  max|R|={T_cw_rot_max:.2e}"
                        )
                    self._publish_pose(T_zed, ts_ns, tracking_ok=False)
                    _frames_lost += 1
                    continue

                # orb_pipe writes T_cw (world→camera). The bridge's downstream
                # consumers (FrameAligner, OrbSlamSub) want T_wc (camera-in-
                # world) so the translation is the camera position in world.
                T_cam = _invert_se3(T_cw)

                if not np.all(np.isfinite(T_cam)) or float(np.linalg.norm(T_cam[:3, 3])) > 50.0:
                    if self._frame_count % 30 == 1:
                        print(
                            f"[OrbBridge] rejecting inverted pose "
                            f"|t|={float(np.linalg.norm(T_cam[:3, 3])):.2e} m"
                        )
                    self._publish_pose(T_zed, ts_ns, tracking_ok=False)
                    _frames_lost += 1
                    continue

                # ── Convert ORB pose to Y-up ──────────────────────────────────
                T_orb_yup = _orb_to_yup(T_cam)

                # ── Register frame alignment on first good pair ───────────────
                if not self._aligner.aligned:
                    self._aligner.register(T_zed, T_orb_yup)

                T_world = self._aligner.apply(T_orb_yup)

                self._publish_pose(T_world, ts_ns, tracking_ok=True)
                _frames_ok += 1

                # Periodic status log
                now = time.time()
                if now - _last_log_t > 5.0:
                    total = _frames_ok + _frames_lost
                    print(
                        f"[OrbBridge] frames={total}  "
                        f"ok={_frames_ok}  lost={_frames_lost}  "
                        f"aligned={self._aligner.aligned}"
                    )
                    _last_log_t = now

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
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ZMQ bridge: ZED frames → ORB-SLAM3 → orb/pose topic"
    )
    parser.add_argument(
        "--orb-bin",
        default=os.path.expanduser("~/ORB_SLAM3/Examples/RGB-D/rgbd_tum"),
        help="Path to the compiled ORB-SLAM3 RGB-D binary (default: ~/ORB_SLAM3/...)",
    )
    parser.add_argument(
        "--vocab",
        default=str(Path(__file__).parent.parent / "orbslam3" / "ORBvoc.txt"),
        help="Path to ORBvoc.txt vocabulary file",
    )
    parser.add_argument(
        "--config",
        default="/tmp/orb_zed.yaml",
        help="Path to the ORB-SLAM3 camera config YAML (generated by gen_config.py)",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/orb_work",
        help="Temporary directory for frame images (process backend)",
    )
    parser.add_argument(
        "--backend",
        choices=["process", "bindings"],
        default="process",
        help="'process' = ORB-SLAM3 binary subprocess; 'bindings' = Python bindings",
    )
    parser.add_argument(
        "--zed-host", default="127.0.0.1",
        help="Host running zed_pub_node",
    )
    parser.add_argument(
        "--zed-port", type=int, default=6000,
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
        "--align", choices=["identity", "zed"], default="identity",
        help=(
            "Frame alignment mode (default: identity). "
            "'identity' assumes ORB and EKF both start at origin — robust to "
            "stale ZED area-memory offsets. 'zed' aligns ORB to the first ZED "
            "pose (only useful if ORB and ZED share a saved area memory)."
        ),
    )
    parser.add_argument(
        "--gen-config", action="store_true",
        help="Auto-generate the ORB-SLAM3 config from live ZED camera info before starting",
    )

    args = parser.parse_args()

    # Auto-generate config if requested
    if args.gen_config:
        print("[OrbBridge] Auto-generating ORB-SLAM3 config …")
        import importlib
        gen = importlib.import_module("robot.nav.orbslam3.gen_config")
        live = gen.fetch_intrinsics_from_zed(args.zed_host, args.zed_port)
        intrinsics = gen._DEFAULT_INTRINSICS.copy()
        if live:
            intrinsics.update(live)
        yaml_str = gen.build_yaml(intrinsics)
        Path(args.config).parent.mkdir(parents=True, exist_ok=True)
        Path(args.config).write_text(yaml_str)
        print(f"[OrbBridge] Config written to {args.config}")

    # Validate paths
    if args.backend == "process":
        if not os.path.isfile(args.orb_bin):
            print(
                f"[OrbBridge] ERROR: ORB-SLAM3 binary not found: {args.orb_bin}\n"
                f"  Build ORB-SLAM3 first, or use --orb-bin /path/to/rgbd_tum",
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

    bridge = OrbSlamBridge(
        zed_host    = args.zed_host,
        zed_port    = args.zed_port,
        pub_port    = args.pub_port,
        backend     = args.backend,
        orb_bin     = args.orb_bin,
        vocab       = args.vocab,
        config      = args.config,
        work_dir    = args.work_dir,
        skip_frames = args.skip,
        max_depth_m = args.max_depth,
        align       = args.align,
    )
    bridge.run()


if __name__ == "__main__":
    main()
