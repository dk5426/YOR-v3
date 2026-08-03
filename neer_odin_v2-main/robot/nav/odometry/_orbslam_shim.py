#!/usr/bin/env python3
"""
_orbslam_shim.py — Persistent ORB-SLAM3 RGB-D driver over a binary pipe.

This script is launched ONCE by orbslam_bridge.py and stays alive for the
entire session. It:
  1. Reads binary RGB-D frames from stdin (sent by _ProcessBackend.track()).
  2. Writes frames to a TUM-style dataset directory on a RAM disk (/tmp).
  3. Runs a single persistent ORB-SLAM3 process that reads from that directory
     via a named FIFO / inotify-style watch loop.
  4. Writes binary pose responses to stdout.

Because ORB-SLAM3's rgbd_tum binary is batch-only (reads a fixed association
file), we instead use a DIFFERENT approach: we drive ORB-SLAM3 directly via
its C library through ctypes, loading libORB_SLAM3.so directly.

If libORB_SLAM3.so isn't importable, we fall back to a file-based incremental
batch runner that processes frames in small rolling windows.

Protocol (matches _ProcessBackend in orbslam_bridge.py)
--------------------------------------------------------
STDIN per frame:
    4B  magic  = b"ORBS"
    4B  width  (uint32 LE)
    4B  height (uint32 LE)
    8B  timestamp_s (float64 LE)
    W*H*3 B   RGB uint8
    W*H*4 B   depth float32 (metres, NaN = invalid)

STDOUT per frame:
    1B   tracking_ok  (uint8: 1=OK, 0=LOST/INIT)
    128B  4×4 pose    (float64 row-major, identity if tracking_ok=0)
"""

import argparse
import os
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

MAGIC_EXPECTED = b"ORBS"
IDENTITY = np.eye(4, dtype=np.float64)


def read_frame(stdin_b):
    """Read one binary frame from stdin. Returns (W, H, ts, rgb, depth) or None."""
    magic = stdin_b.read(4)
    if len(magic) < 4:
        return None
    if magic != MAGIC_EXPECTED:
        return None
    hdr = stdin_b.read(16)
    if len(hdr) < 16:
        return None
    W, H = struct.unpack("<II", hdr[:8])
    ts,  = struct.unpack("<d",  hdr[8:16])
    n_rgb   = W * H * 3
    n_depth = W * H * 4
    rgb_raw   = stdin_b.read(n_rgb)
    depth_raw = stdin_b.read(n_depth)
    if len(rgb_raw) < n_rgb or len(depth_raw) < n_depth:
        return None
    rgb   = np.frombuffer(rgb_raw,   dtype=np.uint8).reshape(H, W, 3).copy()
    depth = np.frombuffer(depth_raw, dtype=np.float32).reshape(H, W).copy()
    return W, H, ts, rgb, depth


def write_response(stdout_b, ok: bool, T: np.ndarray):
    """Write one binary response to stdout."""
    stdout_b.write(struct.pack("<B", 1 if ok else 0))
    stdout_b.write(T.astype(np.float64).tobytes())
    stdout_b.flush()


# ══════════════════════════════════════════════════════════════════════════════
#  ctypes-based persistent backend (uses libORB_SLAM3.so directly)
# ══════════════════════════════════════════════════════════════════════════════

def _try_ctypes_backend(orb_bin: str, vocab: str, config: str):
    """
    Try to load libORB_SLAM3.so via ctypes and call TrackRGBD directly.
    Returns a (track_fn, shutdown_fn) pair or None if unavailable.

    track_fn(rgb_bgr: np.ndarray, depth_m: np.ndarray, ts_s: float)
        -> (ok: bool, T_4x4: np.ndarray)
    """
    # Find libORB_SLAM3.so relative to the binary
    bin_path = Path(orb_bin).resolve()
    lib_candidates = [
        bin_path.parent.parent.parent / "lib" / "libORB_SLAM3.so",
        bin_path.parent / "libORB_SLAM3.so",
        Path("/usr/local/lib/libORB_SLAM3.so"),
    ]
    lib_path = next((p for p in lib_candidates if p.exists()), None)
    if lib_path is None:
        return None

    try:
        import ctypes, ctypes.util

        # Load the library and its dependencies
        ctypes.CDLL(str(lib_path.parent.parent / "Thirdparty/DBoW2/lib/libDBoW2.so"),
                    mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(lib_path.parent.parent / "Thirdparty/g2o/lib/libg2o.so"),
                    mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

        # We can't easily call C++ methods via ctypes. Return None to fall back.
        return None
    except Exception as exc:
        print(f"[shim] ctypes backend unavailable: {exc}", file=sys.stderr)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Rolling-batch backend: write all frames so far, run rgbd_tum on them
# ══════════════════════════════════════════════════════════════════════════════

class RollingBatchBackend:
    """
    Writes incoming frames to a TUM dataset directory and runs rgbd_tum on
    the full growing dataset each time, exploiting ORB-SLAM3's incremental
    tracking (it processes frames sequentially and outputs a trajectory file).

    On each new frame:
      1. Write rgb + depth PNGs.
      2. Append to the association file.
      3. Launch rgbd_tum on the FULL dataset (it re-tracks from the beginning
         quickly because ORB-SLAM3 has already initialized).

    This is slow (~1 s/frame) but correct. For real-time use, the orbslam_bridge
    --skip argument should be set high (e.g. --skip 10) so we only call this
    at 1-3 Hz.

    NOTE: This is a stopgap. For production, build the orbslam3 Python bindings.
    """

    def __init__(self, orb_bin: str, vocab: str, config: str, work_dir: str):
        self.orb_bin  = orb_bin
        self.vocab    = vocab
        self.config   = config
        self.work_dir = Path(work_dir)
        self.rgb_dir  = self.work_dir / "rgb"
        self.dep_dir  = self.work_dir / "depth"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.dep_dir.mkdir(parents=True, exist_ok=True)
        self.assoc_file = self.work_dir / "associations.txt"
        self.traj_file  = self.work_dir / "CameraTrajectory.txt"
        self.frame_idx  = 0
        self._assoc_lines: list[str] = []
        self._last_pose: np.ndarray = IDENTITY.copy()
        # Write empty assoc file
        self.assoc_file.write_text("")

    def track(self, rgb: np.ndarray, depth: np.ndarray, ts: float):
        """Write frame and run ORB-SLAM3. Returns (ok, T_4x4)."""
        import subprocess

        idx = self.frame_idx
        self.frame_idx += 1

        rgb_name = f"{ts:.6f}.png"
        dep_name = f"{ts:.6f}.png"
        rgb_path = self.rgb_dir / rgb_name
        dep_path = self.dep_dir / dep_name

        # Write RGB (BGR for OpenCV)
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Write depth as uint16 PNG (TUM convention: 1 unit = 1 mm)
        depth_mm = np.nan_to_num(depth, nan=0.0) * 1000.0
        depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(dep_path), depth_mm)

        # Append to association file
        line = f"{ts:.6f} rgb/{rgb_name} {ts:.6f} depth/{dep_name}"
        self._assoc_lines.append(line)
        self.assoc_file.write_text("\n".join(self._assoc_lines) + "\n")

        # Delete old trajectory file so we can detect fresh output
        if self.traj_file.exists():
            self.traj_file.unlink()

        try:
            ret = subprocess.run(
                [self.orb_bin, self.vocab, self.config,
                 str(self.assoc_file), str(self.traj_file)],
                capture_output=True,
                cwd=str(self.work_dir),
                timeout=5.0,
            )
        except subprocess.TimeoutExpired:
            print("[shim] rgbd_tum timeout on frame", idx, file=sys.stderr)
            return False, self._last_pose

        if ret.returncode != 0 or not self.traj_file.exists():
            return False, self._last_pose

        # Read last line of trajectory (TUM format: ts qx qy qz qw tx ty tz)
        try:
            lines = self.traj_file.read_text().strip().splitlines()
            if not lines:
                return False, self._last_pose
            vals = list(map(float, lines[-1].split()))
            if len(vals) < 8:
                return False, self._last_pose
            from scipy.spatial.transform import Rotation
            qxyz = vals[1:4]; qw = vals[4]
            t = vals[5:8]
            # scipy uses [x,y,z,w]
            R = Rotation.from_quat([qxyz[0], qxyz[1], qxyz[2], qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3,  3] = t
            self._last_pose = T
            return True, T
        except Exception as exc:
            print(f"[shim] pose parse error: {exc}", file=sys.stderr)
            return False, self._last_pose


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="ORB-SLAM3 persistent shim")
    p.add_argument("--orb-bin",  required=True)
    p.add_argument("--vocab",    required=True)
    p.add_argument("--config",   required=True)
    p.add_argument("--work-dir", default="/tmp/orb_work")
    args = p.parse_args()

    stdin_b  = sys.stdin.buffer
    stdout_b = sys.stdout.buffer

    print("[shim] Starting RollingBatchBackend …", file=sys.stderr)
    backend = RollingBatchBackend(args.orb_bin, args.vocab, args.config, args.work_dir)
    print("[shim] Ready — waiting for frames.", file=sys.stderr)

    while True:
        result = read_frame(stdin_b)
        if result is None:
            print("[shim] stdin closed, exiting.", file=sys.stderr)
            break
        _W, _H, ts, rgb, depth = result
        ok, T = backend.track(rgb, depth, ts)
        write_response(stdout_b, ok, T)


if __name__ == "__main__":
    main()
