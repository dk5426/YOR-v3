#!/usr/bin/env python3
"""
dataset_loader.py — Read datasets written by collect_dataset.py.

Exposes two layers:

  1. `DatasetReader` — low-level random-access + time-ordered iteration over
     any subset of streams. Each sample is a `Sample(stream, pub_ns, recv_ns, data)`
     where `data` is decoded on demand (image → np.ndarray, etc.).

  2. `iter_synced_frames(reader, primary='pcd', tol_ns=...)` — yields bundles
     aligned to the `primary` stream's timestamps, with the nearest sample of
     every other stream attached (or None if outside tolerance). This is what
     most SLAM / odometry pipelines want.

Example
-------
    from robot.data_collection.dataset_loader import DatasetReader, iter_synced_frames

    reader = DatasetReader("/data/runs/2026-04-16_kitchen")
    print(reader.manifest["duration_s"], "s,", reader.manifest["streams"])

    for bundle in iter_synced_frames(reader, primary="pcd"):
        pts  = bundle["pcd"]       # (H, W, 4) float32
        pose = bundle["pose"]      # 20-float numpy array
        img  = bundle["image"]     # (H, W, 3) uint8 BGR/RGB (same bytes as publisher)
        enc  = bundle["encoders"]  # dict of wheel state
        # … run your SLAM / odometry here …
"""

from __future__ import annotations

import bisect
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Sample + stream abstractions
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    stream:  str
    pub_ns:  int
    recv_ns: int
    _payload: Any    # file path (str) for file-backed streams, ready-to-use value otherwise
    _decoder: Any    # callable or None

    @property
    def data(self) -> Any:
        if self._decoder is None:
            return self._payload
        return self._decoder(self._payload)


class _FileBackedStream:
    """image / depth / pcd — one file per sample, indexed by a CSV."""
    def __init__(self, root: Path, name: str, index_path: Path, decoder):
        self._root = root
        self.name = name
        self._decoder = decoder
        self._rows: List[Tuple[int, int, int, str]] = []   # (frame_idx, pub_ns, recv_ns, path)
        self._pub_ns: List[int] = []
        self._load_index(index_path)

    def _load_index(self, path: Path) -> None:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self._rows.append((
                    int(r["frame_idx"]),
                    int(r["pub_ns"]),
                    int(r["recv_ns"]),
                    r["path"],
                ))
        self._rows.sort(key=lambda x: x[1])  # by pub_ns
        self._pub_ns = [r[1] for r in self._rows]

    def __len__(self) -> int:
        return len(self._rows)

    def pub_ns_list(self) -> List[int]:
        return self._pub_ns

    def sample_at(self, i: int) -> Sample:
        frame_idx, pub_ns, recv_ns, rel = self._rows[i]
        return Sample(
            stream=self.name,
            pub_ns=pub_ns,
            recv_ns=recv_ns,
            _payload=str(self._root / rel),
            _decoder=self._decoder,
        )

    def nearest(self, target_ns: int, tol_ns: Optional[int]) -> Optional[Sample]:
        if not self._pub_ns:
            return None
        i = bisect.bisect_left(self._pub_ns, target_ns)
        best_i, best_dt = -1, None
        for j in (i - 1, i):
            if 0 <= j < len(self._pub_ns):
                dt = abs(self._pub_ns[j] - target_ns)
                if best_dt is None or dt < best_dt:
                    best_i, best_dt = j, dt
        if best_i < 0:
            return None
        if tol_ns is not None and best_dt is not None and best_dt > tol_ns:
            return None
        return self.sample_at(best_i)


class _CSVStream:
    """pose / encoders — a single CSV file, one row per sample."""
    def __init__(self, name: str, csv_path: Path):
        self.name = name
        self._rows: List[Dict[str, Any]] = []
        self._pub_ns: List[int] = []
        self._load(csv_path)

    def _load(self, path: Path) -> None:
        if self.name == "pose":
            self._load_pose(path)
        elif self.name == "encoders":
            self._load_encoders(path)
        else:
            raise ValueError(f"Unknown CSV stream: {self.name}")
        # Sort by pub_ns for bisect
        order = sorted(range(len(self._rows)), key=lambda i: self._rows[i]["_pub_ns"])
        self._rows = [self._rows[i] for i in order]
        self._pub_ns = [r["_pub_ns"] for r in self._rows]

    def _load_pose(self, path: Path) -> None:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                pub_ns = int(r["pub_ns"])
                row = {
                    "_pub_ns": pub_ns,
                    "_recv_ns": int(r["recv_ns"]),
                    "base_qt": np.array([
                        float(r["base_qx"]), float(r["base_qy"]),
                        float(r["base_qz"]), float(r["base_qw"]),
                        float(r["base_tx"]), float(r["base_ty"]), float(r["base_tz"]),
                    ], dtype=np.float32),
                    "cam_qt": np.array([
                        float(r["cam_qx"]), float(r["cam_qy"]),
                        float(r["cam_qz"]), float(r["cam_qw"]),
                        float(r["cam_tx"]), float(r["cam_ty"]), float(r["cam_tz"]),
                    ], dtype=np.float32),
                    "base_xyth": np.array([
                        float(r["base_x"]), float(r["base_y"]),
                        float(r["base_z"]), float(r["base_yaw"]),
                    ], dtype=np.float32),
                    "confidence": float(r["confidence"]),
                }
                self._rows.append(row)

    def _load_encoders(self, path: Path) -> None:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                pub_s = float(r["pub_s"])
                pub_ns = int(pub_s * 1e9)
                def _quad(prefix):
                    return np.array(
                        [float(r[f"{prefix}_0"]), float(r[f"{prefix}_1"]),
                         float(r[f"{prefix}_2"]), float(r[f"{prefix}_3"])],
                        dtype=np.float64,
                    )
                row = {
                    "_pub_ns": pub_ns,
                    "_recv_ns": int(r["recv_ns"]),
                    "pub_s":        pub_s,
                    "steer_rad":    _quad("steer_rad"),
                    "steer_deg":    _quad("steer_deg"),
                    "steer_counts": _quad("steer_counts"),
                    "drive_vel":    _quad("drive_vel"),
                    "drive_counts": _quad("drive_counts"),
                    "lift_height_m": float(r["lift_height_m"]),
                }
                self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def pub_ns_list(self) -> List[int]:
        return self._pub_ns

    def sample_at(self, i: int) -> Sample:
        row = self._rows[i]
        return Sample(
            stream=self.name,
            pub_ns=row["_pub_ns"],
            recv_ns=row["_recv_ns"],
            _payload=row,       # already decoded
            _decoder=None,
        )

    def nearest(self, target_ns: int, tol_ns: Optional[int]) -> Optional[Sample]:
        if not self._pub_ns:
            return None
        i = bisect.bisect_left(self._pub_ns, target_ns)
        best_i, best_dt = -1, None
        for j in (i - 1, i):
            if 0 <= j < len(self._pub_ns):
                dt = abs(self._pub_ns[j] - target_ns)
                if best_dt is None or dt < best_dt:
                    best_i, best_dt = j, dt
        if best_i < 0:
            return None
        if tol_ns is not None and best_dt is not None and best_dt > tol_ns:
            return None
        return self.sample_at(best_i)


# ---------------------------------------------------------------------------
# Decoders for file-backed streams
# ---------------------------------------------------------------------------
def _decode_image(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if not HAS_CV2:
        raise RuntimeError(f"cv2 required to decode {path}. `pip install opencv-python`.")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cv2.imread failed for {path}")
    return img


def _decode_depth(path: str) -> np.ndarray:
    return np.load(path)


def _decode_pcd(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        with np.load(path) as npz:
            return npz["points"]
    return np.load(path)


# ---------------------------------------------------------------------------
# DatasetReader
# ---------------------------------------------------------------------------
class DatasetReader:
    """Random-access reader for a collect_dataset.py output directory.

    Attributes
    ----------
    root      — pathlib.Path
    manifest  — dict from manifest.json
    streams   — dict[str, stream] for whatever streams are present
    """
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in {self.root}")
        with open(manifest_path, "r") as f:
            self.manifest: dict = json.load(f)

        self.streams: Dict[str, Any] = {}

        # File-backed streams
        for name, dir_name, decoder in (
            ("image", "zed_image", _decode_image),
            ("depth", "zed_depth", _decode_depth),
            ("pcd",   "zed_pcd",   _decode_pcd),
        ):
            idx = self.root / dir_name / "index.csv"
            if idx.exists():
                self.streams[name] = _FileBackedStream(self.root, name, idx, decoder)

        # CSV streams
        pose_csv = self.root / "zed_pose.csv"
        if pose_csv.exists():
            self.streams["pose"] = _CSVStream("pose", pose_csv)
        enc_csv = self.root / "encoders.csv"
        if enc_csv.exists():
            self.streams["encoders"] = _CSVStream("encoders", enc_csv)

    def __repr__(self) -> str:
        counts = {n: len(s) for n, s in self.streams.items()}
        return f"DatasetReader({self.root}, streams={counts})"

    def stream(self, name: str):
        return self.streams[name]

    def iter_stream(self, name: str) -> Iterator[Sample]:
        """Yield every sample of `name` in pub_ns order."""
        s = self.streams[name]
        for i in range(len(s)):
            yield s.sample_at(i)


# ---------------------------------------------------------------------------
# High-level aligned iteration
# ---------------------------------------------------------------------------
def iter_synced_frames(
    reader: DatasetReader,
    primary: str = "pcd",
    others: Optional[List[str]] = None,
    tol_ns: int = 50_000_000,   # ±50 ms default
    require: Optional[List[str]] = None,
    decode: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield dicts aligned to the `primary` stream's timestamps.

    For each sample of `primary`, the nearest sample of every other stream
    within `tol_ns` is attached. Streams farther than `tol_ns` come back as
    `None`. If any stream in `require` is None for a given frame, the frame
    is skipped.

    Args
    ----
    primary: anchor stream; typically "pcd" or "image".
    others:  which streams to attach. Defaults to all present except `primary`.
    tol_ns:  max allowed |primary.pub_ns - other.pub_ns| in nanoseconds.
    require: streams that MUST have a match within tol_ns; else the frame is dropped.
    decode:  if True, decode file-backed payloads immediately; else leave as Sample.

    Yields
    ------
    dict with keys:
        'pub_ns', 'recv_ns'            — timestamps of the primary sample
        '<primary>'                    — decoded payload of primary
        '<other>'                      — decoded payload of other (or None)
        '<other>_pub_ns'               — pub_ns of the matched other sample
                                         (None if not matched)
    """
    if primary not in reader.streams:
        raise KeyError(f"primary stream '{primary}' not in dataset")
    primary_stream = reader.streams[primary]
    other_names = list(others) if others is not None else [
        n for n in reader.streams.keys() if n != primary
    ]
    require = require or []

    for i in range(len(primary_stream)):
        p = primary_stream.sample_at(i)
        out: Dict[str, Any] = {
            "pub_ns":  p.pub_ns,
            "recv_ns": p.recv_ns,
            primary:   (p.data if decode else p),
        }
        missing_required = False
        for n in other_names:
            s = reader.streams[n].nearest(p.pub_ns, tol_ns=tol_ns)
            if s is None:
                out[n] = None
                out[f"{n}_pub_ns"] = None
                if n in require:
                    missing_required = True
                    break
            else:
                out[n] = (s.data if decode else s)
                out[f"{n}_pub_ns"] = s.pub_ns
        if missing_required:
            continue
        yield out


__all__ = ["DatasetReader", "Sample", "iter_synced_frames"]


if __name__ == "__main__":
    # Tiny CLI: point it at a dataset, print a summary.
    import argparse
    ap = argparse.ArgumentParser(description="Inspect a collect_dataset.py output directory")
    ap.add_argument("root", type=str)
    args = ap.parse_args()
    r = DatasetReader(args.root)
    print(r)
    print("manifest:", json.dumps(r.manifest, indent=2, default=str))
    for name, s in r.streams.items():
        n = len(s)
        if n == 0:
            print(f"  {name}: empty")
            continue
        pub = s.pub_ns_list()
        dur = (pub[-1] - pub[0]) / 1e9 if n > 1 else 0.0
        hz = (n - 1) / dur if dur > 0 else 0.0
        print(f"  {name}: {n} samples, ~{hz:.1f} Hz, "
              f"first={pub[0]}  last={pub[-1]}  duration={dur:.2f}s")
