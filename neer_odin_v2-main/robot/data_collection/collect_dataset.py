#!/usr/bin/env python3
"""
collect_dataset.py — Record raw ZED + wheel-encoder data to a reusable offline dataset.

Purpose
-------
Capture everything a SLAM / mapping / odometry pipeline might need, once, from
a single robot drive — so the same drive can be replayed many times offline
against different algorithms without touching the robot again.

What it records
---------------
Subscribes to the commlink topics already published by `robot/zed_pub_node.py`:

    zed/image  – (H, W, 3)        uint8   left RGB
    zed/depth  – (H, W)           float32 metric depth [m]
    zed/pcd    – (H, W, 4)        float32 XYZRGBA (packed colour in channel 3)
    zed/pose   – 20-float list    [base_qt(7), cam_qt(7), base_xyth(4), ts_ns, confidence]

…and polls `RPCClient.get_base_encoders()` on the YOR server for wheel data:

    {"timestamp", "steer_rad"[4], "steer_deg"[4], "steer_counts"[4],
     "drive_vel"[4], "drive_counts"[4], "lift_height_m"}

No direct dependency on pyzed.sl or base motor drivers — the recorder runs on
any machine that can reach the ZED publisher port + YOR RPC endpoint.

On-disk layout
--------------
    <out>/
      manifest.json              # config + topic list + start/end clock
      stats.json                 # counts, drops, bytes-written per topic
      zed_pose.csv               # header: recv_ns,pub_ns,<20 pose fields>
      encoders.csv               # header: recv_ns,pub_s,steer_rad_0..3,drive_counts_0..3,...
      zed_image/
        index.csv                # frame_idx,pub_ns,recv_ns,path
        000000.png, 000001.png, ...
      zed_depth/
        index.csv
        000000.npy, 000001.npy, ...       # uncompressed float32 (H, W)
      zed_pcd/
        index.csv
        000000.npz, 000001.npz, ...       # compressed float32 (H, W, 4)

Synchronization strategy
------------------------
Every topic records both:
  * `pub_ts` — the timestamp embedded in the payload by the publisher
  * `recv_ts` — the local monotonic clock at the moment the sample was polled
Offline code can use either clock; pub_ns is the canonical shared axis
because ZED's image/depth/pcd/pose all carry the same `time.time_ns()` stamp
for a single grab.

Streams are NOT aligned at record time; de-skew/interpolation is left to the
offline loader (standard rosbag pattern) so we never drop a perfectly good
sample just because a sibling topic was late.

Flow control
------------
Each topic has:
    poller thread  →  Queue(maxsize=N, drop-on-full)  →  writer thread  →  disk
If a writer can't keep up (e.g. PCD on a slow disk) the oldest queued frame
is dropped and the drop counter ticks up — nothing blocks upstream.
Printed stats tell you if that is happening.

Typical invocation
------------------
    # Record everything until Ctrl-C, dropping every other PCD frame
    python -m robot.data_collection.collect_dataset \
        --out /data/runs/2026-04-16_kitchen \
        --zed-host 192.168.1.11 --zed-port 6000 \
        --yor-host 192.168.1.10 --yor-port 5557 \
        --stride-pcd 2

    # 60-second fixed-duration capture without point cloud (smaller dataset)
    python -m robot.data_collection.collect_dataset \
        --out /data/runs/short_test \
        --duration 60 \
        --skip pcd
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from commlink import Subscriber, RPCClient  # type: ignore
except Exception as e:
    print(f"[collect_dataset] ERROR: commlink import failed: {e}", file=sys.stderr)
    print("This script depends on the robot's commlink pub/sub layer.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants (match robot/zed_pub_node.py and robot/slam_node_.py)
# ---------------------------------------------------------------------------
ZED_PUB_PORT = 6000
POSE_TOPIC   = "zed/pose"
IMAGE_TOPIC  = "zed/image"
DEPTH_TOPIC  = "zed/depth"
PCD_TOPIC    = "zed/pcd"

YOR_RPC_HOST = "194.168.1.10"
YOR_RPC_PORT = 5557

# Known topic names for CLI --skip
ALL_STREAMS = ("image", "depth", "pcd", "pose", "encoders")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _now_ns() -> int:
    """Local wall-clock nanoseconds — same clock base the ZED publisher uses."""
    return time.time_ns()


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
@dataclass
class StreamStats:
    name: str
    received: int = 0        # unique samples successfully read from the bus
    written:  int = 0        # samples actually serialised to disk
    dropped_queue: int = 0   # dropped because writer queue was full
    dropped_stride: int = 0  # skipped by --stride-<topic> N
    bytes_written: int = 0
    last_pub_ns: Optional[int] = None
    first_pub_ns: Optional[int] = None

    def as_dict(self) -> dict:
        dur_s = 0.0
        if self.first_pub_ns and self.last_pub_ns:
            dur_s = (self.last_pub_ns - self.first_pub_ns) / 1e9
        hz = (self.written / dur_s) if dur_s > 0 else 0.0
        return {
            "received": self.received,
            "written":  self.written,
            "dropped_queue":  self.dropped_queue,
            "dropped_stride": self.dropped_stride,
            "bytes_written":  self.bytes_written,
            "first_pub_ns":   self.first_pub_ns,
            "last_pub_ns":    self.last_pub_ns,
            "duration_s":     round(dur_s, 3),
            "effective_hz":   round(hz, 2),
        }


class Stats:
    """Thread-safe registry of per-stream stats."""
    def __init__(self, names: List[str]):
        self._lock = threading.Lock()
        self._streams: Dict[str, StreamStats] = {n: StreamStats(n) for n in names}

    def get(self, name: str) -> StreamStats:
        return self._streams[name]

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return {n: s.as_dict() for n, s in self._streams.items()}

    def inc(self, name: str, field: str, delta: int = 1) -> None:
        with self._lock:
            s = self._streams[name]
            setattr(s, field, getattr(s, field) + delta)

    def update_pub_ts(self, name: str, pub_ns: int) -> None:
        with self._lock:
            s = self._streams[name]
            if s.first_pub_ns is None:
                s.first_pub_ns = pub_ns
            s.last_pub_ns = pub_ns


# ---------------------------------------------------------------------------
# Topic polling + writer
# ---------------------------------------------------------------------------
class TopicPoller(threading.Thread):
    """Polls a single commlink topic, dedupes by pub timestamp, pushes to writer queue.

    `ts_getter(msg)` must return an int64 nanosecond timestamp embedded in the
    payload (or `None` to fall back to recv_ts).  Frames where `ts_getter` is
    unchanged since last poll are ignored — they are stale repeats of the
    latest-value subscriber.
    """
    def __init__(
        self,
        *,
        name: str,
        sub: Any,
        topic: str,
        ts_getter: Callable[[Any], Optional[int]],
        writer_queue: queue.Queue,
        stop_evt: threading.Event,
        stats: Stats,
        poll_hz: float,
        stride: int = 1,
    ):
        super().__init__(name=f"poll-{name}", daemon=True)
        self._short_name = name
        self._sub = sub
        self._topic = topic
        self._ts_getter = ts_getter
        self._q = writer_queue
        self._stop_evt = stop_evt
        self._stats = stats
        self._period = 1.0 / max(float(poll_hz), 1.0)
        self._stride = max(1, int(stride))
        self._last_pub_ns: Optional[int] = None
        self._stride_counter = 0

    def run(self) -> None:
        while not self._stop_evt.is_set():
            t0 = time.time()
            try:
                msg = self._sub[self._topic]
            except Exception:
                msg = None
            if msg is not None:
                pub_ns = self._ts_getter(msg)
                recv_ns = _now_ns()
                if pub_ns is None:
                    pub_ns = recv_ns
                # Dedupe: same-timestamped message is a repeat
                if pub_ns != self._last_pub_ns:
                    self._last_pub_ns = pub_ns
                    self._stats.inc(self._short_name, "received")
                    self._stats.update_pub_ts(self._short_name, pub_ns)
                    # Stride: drop (self._stride - 1) out of every self._stride frames
                    self._stride_counter += 1
                    if self._stride_counter % self._stride != 0:
                        self._stats.inc(self._short_name, "dropped_stride")
                    else:
                        try:
                            self._q.put_nowait((pub_ns, recv_ns, msg))
                        except queue.Full:
                            # Drop oldest, enqueue newest — keeps freshest frames flowing
                            try:
                                self._q.get_nowait()
                                self._stats.inc(self._short_name, "dropped_queue")
                            except queue.Empty:
                                pass
                            try:
                                self._q.put_nowait((pub_ns, recv_ns, msg))
                            except queue.Full:
                                self._stats.inc(self._short_name, "dropped_queue")
            # Sleep to next tick
            elapsed = time.time() - t0
            remaining = self._period - elapsed
            if remaining > 0:
                self._stop_evt.wait(timeout=remaining)


class TopicWriter(threading.Thread):
    """Drains a queue and serialises each frame with `serialise_fn`.

    `serialise_fn(frame_idx, pub_ns, recv_ns, msg, out_dir)` must return
    `(filename: str, bytes_written: int)`.
    """
    def __init__(
        self,
        *,
        name: str,
        out_dir: Path,
        in_queue: queue.Queue,
        index_file: Path,
        stop_evt: threading.Event,
        stats: Stats,
        serialise_fn: Callable[[int, int, int, Any, Path], Tuple[str, int]],
    ):
        super().__init__(name=f"write-{name}", daemon=True)
        self._short_name = name
        self._out_dir = out_dir
        self._q = in_queue
        self._stop_evt = stop_evt
        self._stats = stats
        self._serialise = serialise_fn
        self._index_path = index_file
        self._frame_idx = 0
        self._index_file = None  # type: Optional[Any]

    def run(self) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        # Open index CSV in line-buffered mode so tail -f works
        self._index_file = open(self._index_path, "w", buffering=1)
        self._index_file.write("frame_idx,pub_ns,recv_ns,path\n")
        try:
            while True:
                try:
                    item = self._q.get(timeout=0.1)
                except queue.Empty:
                    if self._stop_evt.is_set() and self._q.empty():
                        return
                    continue
                pub_ns, recv_ns, msg = item
                try:
                    fname, nbytes = self._serialise(
                        self._frame_idx, pub_ns, recv_ns, msg, self._out_dir
                    )
                    self._index_file.write(
                        f"{self._frame_idx},{pub_ns},{recv_ns},{fname}\n"
                    )
                    self._stats.inc(self._short_name, "written")
                    self._stats.inc(self._short_name, "bytes_written", nbytes)
                    self._frame_idx += 1
                except Exception as e:
                    # Don't crash — log and drop
                    print(f"[writer:{self._short_name}] write failed: {e}",
                          file=sys.stderr)
        finally:
            if self._index_file is not None:
                self._index_file.flush()
                self._index_file.close()


# ---------------------------------------------------------------------------
# Per-topic serialisers
# ---------------------------------------------------------------------------
def serialise_image(idx: int, pub_ns: int, recv_ns: int, msg: Any, out_dir: Path,
                    image_format: str = "png", jpeg_quality: int = 92) -> Tuple[str, int]:
    img = msg["image"]  # HxWx3 uint8 (BGR in practice — see zed_pub_node.py note)
    if not HAS_CV2:
        # Fallback: raw .npy; slower to view but preserves exact bytes
        fname = f"{idx:06d}.npy"
        fpath = out_dir / fname
        np.save(str(fpath), img)
        return fname, fpath.stat().st_size
    ext = image_format.lower()
    if ext == "jpg" or ext == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        fname = f"{idx:06d}.jpg"
    else:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        fname = f"{idx:06d}.png"
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {fname}")
    fpath = out_dir / fname
    fpath.write_bytes(buf.tobytes())
    return fname, len(buf)


def serialise_depth(idx: int, pub_ns: int, recv_ns: int, msg: Any,
                    out_dir: Path) -> Tuple[str, int]:
    depth = np.asarray(msg["depth"], dtype=np.float32)
    fname = f"{idx:06d}.npy"
    fpath = out_dir / fname
    np.save(str(fpath), depth)
    return fname, fpath.stat().st_size


def serialise_pcd_raw(idx: int, pub_ns: int, recv_ns: int, msg: Any,
                      out_dir: Path) -> Tuple[str, int]:
    pts = np.asarray(msg["points"])  # (H, W, 4) float32
    fname = f"{idx:06d}.npy"
    fpath = out_dir / fname
    np.save(str(fpath), pts)
    return fname, fpath.stat().st_size


def serialise_pcd_compressed(idx: int, pub_ns: int, recv_ns: int, msg: Any,
                             out_dir: Path) -> Tuple[str, int]:
    pts = np.asarray(msg["points"])
    fname = f"{idx:06d}.npz"
    fpath = out_dir / fname
    np.savez_compressed(str(fpath), points=pts)
    return fname, fpath.stat().st_size


# ---------------------------------------------------------------------------
# Pose — single CSV stream (small payload, no per-frame files)
# ---------------------------------------------------------------------------
class PoseCSVWriter(threading.Thread):
    """Polls the pose topic and appends a single CSV row per new pub_ns.

    Pose payload is a 20-float list:
        base_qt(7), cam_qt(7), base_xyth(4), pub_ts_ns, confidence
    """
    HEADER = (
        "recv_ns,pub_ns,"
        "base_qx,base_qy,base_qz,base_qw,base_tx,base_ty,base_tz,"
        "cam_qx,cam_qy,cam_qz,cam_qw,cam_tx,cam_ty,cam_tz,"
        "base_x,base_y,base_z,base_yaw,"
        "confidence"
    )

    def __init__(
        self,
        *,
        sub: Any,
        csv_path: Path,
        stop_evt: threading.Event,
        stats: Stats,
        poll_hz: float = 200.0,
    ):
        super().__init__(name="write-pose", daemon=True)
        self._sub = sub
        self._csv_path = csv_path
        self._stop_evt = stop_evt
        self._stats = stats
        self._period = 1.0 / max(float(poll_hz), 1.0)
        self._last_pub_ns: Optional[int] = None

    def run(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w", buffering=1) as f:
            f.write(self.HEADER + "\n")
            while not self._stop_evt.is_set():
                t0 = time.time()
                try:
                    msg = self._sub[POSE_TOPIC]
                except Exception:
                    msg = None
                if msg is not None and len(msg) >= 20:
                    # pub_ns is element 18 per zed_pub_node.py layout
                    try:
                        pub_ns = int(msg[18])
                    except Exception:
                        pub_ns = _now_ns()
                    if pub_ns != self._last_pub_ns:
                        self._last_pub_ns = pub_ns
                        recv_ns = _now_ns()
                        # Flatten first 18 fields + confidence (element 19)
                        vals = list(msg[:18]) + [msg[19]]
                        row = f"{recv_ns},{pub_ns}," + ",".join(f"{float(v):.9g}" for v in vals)
                        f.write(row + "\n")
                        self._stats.inc("pose", "received")
                        self._stats.inc("pose", "written")
                        self._stats.update_pub_ts("pose", pub_ns)
                # Rate-limit
                elapsed = time.time() - t0
                remaining = self._period - elapsed
                if remaining > 0:
                    self._stop_evt.wait(timeout=remaining)


# ---------------------------------------------------------------------------
# Encoder — single CSV stream via RPC
# ---------------------------------------------------------------------------
class EncoderCSVWriter(threading.Thread):
    HEADER = (
        "recv_ns,pub_s,"
        "steer_rad_0,steer_rad_1,steer_rad_2,steer_rad_3,"
        "steer_deg_0,steer_deg_1,steer_deg_2,steer_deg_3,"
        "steer_counts_0,steer_counts_1,steer_counts_2,steer_counts_3,"
        "drive_vel_0,drive_vel_1,drive_vel_2,drive_vel_3,"
        "drive_counts_0,drive_counts_1,drive_counts_2,drive_counts_3,"
        "lift_height_m"
    )

    def __init__(
        self,
        *,
        rpc: Any,
        csv_path: Path,
        stop_evt: threading.Event,
        stats: Stats,
        poll_hz: float = 100.0,
    ):
        super().__init__(name="write-enc", daemon=True)
        self._rpc = rpc
        self._csv_path = csv_path
        self._stop_evt = stop_evt
        self._stats = stats
        self._period = 1.0 / max(float(poll_hz), 1.0)
        self._last_pub_s: Optional[float] = None

    def run(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        consecutive_errors = 0
        with open(self._csv_path, "w", buffering=1) as f:
            f.write(self.HEADER + "\n")
            while not self._stop_evt.is_set():
                t0 = time.time()
                try:
                    enc = self._rpc.get_base_encoders()
                    consecutive_errors = 0
                except Exception as e:
                    enc = None
                    consecutive_errors += 1
                    if consecutive_errors in (1, 10, 100):
                        print(f"[encoders] RPC error ({consecutive_errors} in a row): {e}",
                              file=sys.stderr)
                if enc is not None:
                    pub_s = float(enc.get("timestamp", time.time()))
                    if pub_s != self._last_pub_s:
                        self._last_pub_s = pub_s
                        recv_ns = _now_ns()
                        steer_rad    = list(enc.get("steer_rad",    [float('nan')] * 4))
                        steer_deg    = list(enc.get("steer_deg",    [float('nan')] * 4))
                        steer_counts = list(enc.get("steer_counts", [float('nan')] * 4))
                        drive_vel    = list(enc.get("drive_vel",    [float('nan')] * 4))
                        drive_counts = list(enc.get("drive_counts", [float('nan')] * 4))
                        _lift_raw    = enc.get("lift_height_m", None)
                        lift         = float(_lift_raw) if _lift_raw is not None else float('nan')
                        cells: List[str] = [str(recv_ns), f"{pub_s:.9f}"]
                        for arr in (steer_rad, steer_deg, steer_counts, drive_vel, drive_counts):
                            arr = (arr + [float('nan')] * 4)[:4]
                            cells.extend(f"{float(v):.9g}" for v in arr)
                        cells.append(f"{lift:.6g}")
                        f.write(",".join(cells) + "\n")
                        self._stats.inc("encoders", "received")
                        self._stats.inc("encoders", "written")
                        # Store pub timestamp in ns for consistency in stats
                        self._stats.update_pub_ts("encoders", int(pub_s * 1e9))
                elapsed = time.time() - t0
                remaining = self._period - elapsed
                if remaining > 0:
                    self._stop_evt.wait(timeout=remaining)


# ---------------------------------------------------------------------------
# Stats printer
# ---------------------------------------------------------------------------
class StatsPrinter(threading.Thread):
    def __init__(self, stats: Stats, stop_evt: threading.Event, every_s: float = 2.0):
        super().__init__(name="stats", daemon=True)
        self._stats = stats
        self._stop_evt = stop_evt
        self._every = float(every_s)

    def run(self) -> None:
        t_start = time.time()
        while not self._stop_evt.is_set():
            self._stop_evt.wait(self._every)
            if self._stop_evt.is_set():
                break
            elapsed = time.time() - t_start
            snap = self._stats.snapshot()
            parts = [f"t={elapsed:6.1f}s"]
            for n in ("image", "depth", "pcd", "pose", "encoders"):
                if n not in snap:
                    continue
                s = snap[n]
                parts.append(
                    f"{n}: {s['written']}w "
                    f"({s['effective_hz']:.1f}Hz) "
                    f"dq={s['dropped_queue']} ds={s['dropped_stride']}"
                )
            print(" | ".join(parts))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    out_dir:       Path
    zed_host:      str
    zed_port:      int
    yor_host:      str
    yor_port:      int
    duration_s:    float
    encoder_hz:    float
    pose_hz:       float
    poll_hz:       float
    queue_size:    int
    image_format:  str
    jpeg_quality:  int
    pcd_compress:  bool
    stride_image:  int
    stride_depth:  int
    stride_pcd:    int
    skip:          List[str]
    notes:         str = ""

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["out_dir"] = str(self.out_dir)
        return d


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Record raw ZED + wheel-encoder data to a reusable offline dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", required=True, type=Path,
                   help="output dataset directory (will be created)")
    p.add_argument("--zed-host", default="127.0.0.1")
    p.add_argument("--zed-port", type=int, default=ZED_PUB_PORT)
    p.add_argument("--yor-host", default=YOR_RPC_HOST)
    p.add_argument("--yor-port", type=int, default=YOR_RPC_PORT)
    p.add_argument("--duration", type=float, default=0.0,
                   help="seconds to record (0 = until Ctrl-C)")
    p.add_argument("--encoder-hz", type=float, default=100.0)
    p.add_argument("--pose-hz",    type=float, default=200.0,
                   help="poll rate for zed/pose (dedupes by pub ts so extra is harmless)")
    p.add_argument("--poll-hz",    type=float, default=120.0,
                   help="poll rate for zed/image,depth,pcd topics")
    p.add_argument("--queue-size", type=int, default=16,
                   help="per-writer queue depth (drop-on-full)")
    p.add_argument("--image-format", choices=("png", "jpg"), default="png")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--pcd-compress", dest="pcd_compress", action="store_true", default=True,
                   help="save PCD as compressed .npz (default). Use --no-pcd-compress to save raw .npy (~4 MB/frame).")
    p.add_argument("--no-pcd-compress", dest="pcd_compress", action="store_false")
    p.add_argument("--stride-image", type=int, default=1,
                   help="keep 1 in N image frames (reduces disk usage)")
    p.add_argument("--stride-depth", type=int, default=1)
    p.add_argument("--stride-pcd",   type=int, default=1)
    p.add_argument("--skip", default="",
                   help=f"comma-separated streams to skip; any of: {','.join(ALL_STREAMS)}")
    p.add_argument("--notes", default="",
                   help="free-form text saved into manifest.json (e.g. lighting/location)")
    ns = p.parse_args()

    skip_list = [s.strip().lower() for s in ns.skip.split(",") if s.strip()]
    for s in skip_list:
        if s not in ALL_STREAMS:
            p.error(f"--skip: unknown stream '{s}'. Choose from {ALL_STREAMS}")

    return Config(
        out_dir      = ns.out.expanduser().resolve(),
        zed_host     = ns.zed_host,
        zed_port     = ns.zed_port,
        yor_host     = ns.yor_host,
        yor_port     = ns.yor_port,
        duration_s   = float(ns.duration),
        encoder_hz   = float(ns.encoder_hz),
        pose_hz      = float(ns.pose_hz),
        poll_hz      = float(ns.poll_hz),
        queue_size   = int(ns.queue_size),
        image_format = ns.image_format,
        jpeg_quality = int(ns.jpeg_quality),
        pcd_compress = bool(ns.pcd_compress),
        stride_image = int(ns.stride_image),
        stride_depth = int(ns.stride_depth),
        stride_pcd   = int(ns.stride_pcd),
        skip         = skip_list,
        notes        = ns.notes,
    )


def main() -> int:
    cfg = parse_args()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[collect_dataset] writing to: {cfg.out_dir}")
    print(f"[collect_dataset] zed pub-sub: tcp://{cfg.zed_host}:{cfg.zed_port}")
    print(f"[collect_dataset] yor RPC:     tcp://{cfg.yor_host}:{cfg.yor_port}")
    if not HAS_CV2:
        print("[collect_dataset] WARNING: OpenCV not available — images will be saved as .npy")

    # ZED subscriber for the four camera topics we care about
    topic_list: List[str] = []
    for s, t in (("image", IMAGE_TOPIC), ("depth", DEPTH_TOPIC),
                 ("pcd",   PCD_TOPIC),   ("pose",  POSE_TOPIC)):
        if s not in cfg.skip:
            topic_list.append(t)
    sub: Optional[Any] = None
    if topic_list:
        sub = Subscriber(host=cfg.zed_host, port=cfg.zed_port, topics=topic_list)

    # YOR RPC for wheel encoders
    rpc: Optional[Any] = None
    if "encoders" not in cfg.skip:
        rpc = RPCClient(cfg.yor_host, cfg.yor_port)

    # Stats registry
    active_names = [s for s in ALL_STREAMS if s not in cfg.skip]
    stats = Stats(active_names)

    # Coordinated stop event
    stop_evt = threading.Event()

    # Queues & workers
    pollers: List[threading.Thread] = []
    writers: List[threading.Thread] = []

    def _ts_of_zed_payload(msg: Any) -> Optional[int]:
        # image/depth/pcd all use {"timestamp": ns, ...}
        try:
            return int(msg["timestamp"])
        except Exception:
            return None

    if "image" not in cfg.skip:
        q = queue.Queue(maxsize=cfg.queue_size)
        pollers.append(TopicPoller(
            name="image", sub=sub, topic=IMAGE_TOPIC,
            ts_getter=_ts_of_zed_payload, writer_queue=q,
            stop_evt=stop_evt, stats=stats,
            poll_hz=cfg.poll_hz, stride=cfg.stride_image,
        ))
        def _img_fn(idx, pub_ns, recv_ns, msg, out_dir):
            return serialise_image(idx, pub_ns, recv_ns, msg, out_dir,
                                   image_format=cfg.image_format,
                                   jpeg_quality=cfg.jpeg_quality)
        writers.append(TopicWriter(
            name="image", out_dir=cfg.out_dir / "zed_image",
            in_queue=q, index_file=cfg.out_dir / "zed_image" / "index.csv",
            stop_evt=stop_evt, stats=stats, serialise_fn=_img_fn,
        ))

    if "depth" not in cfg.skip:
        q = queue.Queue(maxsize=cfg.queue_size)
        pollers.append(TopicPoller(
            name="depth", sub=sub, topic=DEPTH_TOPIC,
            ts_getter=_ts_of_zed_payload, writer_queue=q,
            stop_evt=stop_evt, stats=stats,
            poll_hz=cfg.poll_hz, stride=cfg.stride_depth,
        ))
        writers.append(TopicWriter(
            name="depth", out_dir=cfg.out_dir / "zed_depth",
            in_queue=q, index_file=cfg.out_dir / "zed_depth" / "index.csv",
            stop_evt=stop_evt, stats=stats, serialise_fn=serialise_depth,
        ))

    if "pcd" not in cfg.skip:
        q = queue.Queue(maxsize=cfg.queue_size)
        pollers.append(TopicPoller(
            name="pcd", sub=sub, topic=PCD_TOPIC,
            ts_getter=_ts_of_zed_payload, writer_queue=q,
            stop_evt=stop_evt, stats=stats,
            poll_hz=cfg.poll_hz, stride=cfg.stride_pcd,
        ))
        pcd_fn = serialise_pcd_compressed if cfg.pcd_compress else serialise_pcd_raw
        writers.append(TopicWriter(
            name="pcd", out_dir=cfg.out_dir / "zed_pcd",
            in_queue=q, index_file=cfg.out_dir / "zed_pcd" / "index.csv",
            stop_evt=stop_evt, stats=stats, serialise_fn=pcd_fn,
        ))

    if "pose" not in cfg.skip:
        pose_writer = PoseCSVWriter(
            sub=sub, csv_path=cfg.out_dir / "zed_pose.csv",
            stop_evt=stop_evt, stats=stats, poll_hz=cfg.pose_hz,
        )
        # Pose polls + writes in one thread (payload is trivially small)
        writers.append(pose_writer)

    if "encoders" not in cfg.skip:
        enc_writer = EncoderCSVWriter(
            rpc=rpc, csv_path=cfg.out_dir / "encoders.csv",
            stop_evt=stop_evt, stats=stats, poll_hz=cfg.encoder_hz,
        )
        writers.append(enc_writer)

    # Stats printer
    printer = StatsPrinter(stats, stop_evt, every_s=2.0)

    # Write initial manifest before we start (fills in end clock on shutdown)
    start_wall_ns = _now_ns()
    manifest = {
        "version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_wall_ns / 1e9)),
        "config": cfg.as_dict(),
        "streams": active_names,
        "start_wall_ns": start_wall_ns,
        "end_wall_ns":   None,
        "pose_topic_layout": [
            "base_qx", "base_qy", "base_qz", "base_qw",
            "base_tx", "base_ty", "base_tz",
            "cam_qx",  "cam_qy",  "cam_qz",  "cam_qw",
            "cam_tx",  "cam_ty",  "cam_tz",
            "base_x", "base_y", "base_z", "base_yaw",
            "pub_ts_ns", "confidence",
        ],
        "notes": cfg.notes,
    }
    _atomic_write_json(cfg.out_dir / "manifest.json", manifest)

    # Signal handling
    def _sigint(signum, frame):
        if not stop_evt.is_set():
            print("\n[collect_dataset] Ctrl-C received — finalising dataset…")
            stop_evt.set()
    signal.signal(signal.SIGINT, _sigint)
    try:
        signal.signal(signal.SIGTERM, _sigint)
    except Exception:
        pass

    # Launch everything — writers FIRST so no early-dropped frames
    for w in writers:
        w.start()
    for p in pollers:
        p.start()
    printer.start()

    # Wait for duration or signal
    try:
        if cfg.duration_s > 0:
            stop_evt.wait(cfg.duration_s)
            stop_evt.set()
        else:
            # Block until stop_evt fires (from SIGINT)
            while not stop_evt.is_set():
                stop_evt.wait(1.0)
    finally:
        stop_evt.set()

    # Drain + join
    print("[collect_dataset] draining queues…")
    # Give pollers a moment to finish current poll
    for p in pollers:
        p.join(timeout=2.0)
    # Writers may still have items queued; wait generously
    for w in writers:
        w.join(timeout=15.0)
    printer.join(timeout=2.0)

    end_wall_ns = _now_ns()
    manifest["end_wall_ns"] = end_wall_ns
    manifest["duration_s"]  = round((end_wall_ns - start_wall_ns) / 1e9, 3)
    _atomic_write_json(cfg.out_dir / "manifest.json", manifest)
    _atomic_write_json(cfg.out_dir / "stats.json", stats.snapshot())

    # Final summary
    snap = stats.snapshot()
    print("\n[collect_dataset] Done. Final stats:")
    for name, s in snap.items():
        print(f"  {name:9s} written={s['written']:6d}  "
              f"dropped_q={s['dropped_queue']:4d}  "
              f"dropped_stride={s['dropped_stride']:4d}  "
              f"hz≈{s['effective_hz']:.1f}  "
              f"bytes={s['bytes_written'] / 1e6:8.1f} MB")
    print(f"  total duration: {manifest['duration_s']:.2f}s")
    print(f"  dataset:        {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
