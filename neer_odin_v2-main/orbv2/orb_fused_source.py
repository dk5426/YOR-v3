#!/usr/bin/env python3
"""
orb_fused_source.py — ORB-SLAM3 pose subscriber and EKF fusion layer.

Provides ``OrbFusedSource``, a drop-in replacement for ``EKFSlamSource`` that
adds a second EKF update step using ORB-SLAM3 poses for loop-closure
corrections.  Falls back gracefully to ZED+encoder-only behaviour if the
ORB-SLAM3 bridge is not running.

This is a consolidated, simplified version of the legacy
``orb_slam_source.py`` (OrbSlamSub + OrbEKFSlamSource) merged into a single
module with better diagnostics.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np

# ── Topic / port constants ────────────────────────────────────────────────────
ORB_POSE_TOPIC = "orb/pose"
ORB_PUB_PORT   = 6001


def _quat_to_matrix_3x3(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] → 3×3 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    return np.array([
        [1.0 - (yy + zz), xy - wz,          xz + wy         ],
        [xy + wz,          1.0 - (xx + zz), yz - wx          ],
        [xz - wy,          yz + wx,          1.0 - (xx + yy) ],
    ], dtype=np.float32)


class _OrbSubscriber:
    """Lightweight subscriber for the ``orb/pose`` ZMQ topic.

    Runs a background polling thread that caches the latest ORB-SLAM3 pose.
    All public methods are thread-safe.

    Parameters
    ----------
    host : str
        Host running orbv2.orb_bridge (default: localhost).
    port : int
        ZMQ port for the orb/pose topic (default: 6001).
    stale_timeout_s : float
        Poses older than this are considered stale and ignored.
    lc_threshold_m : float
        Minimum position jump between consecutive ORB-SLAM3 poses to classify
        the update as a loop closure, triggering an unconditional EKF correction.
    """

    def __init__(
        self,
        host:            str   = "127.0.0.1",
        port:            int   = ORB_PUB_PORT,
        stale_timeout_s: float = 2.0,
        lc_threshold_m:  float = 0.30,
    ):
        self._host           = host
        self._port           = port
        self._stale_timeout  = float(stale_timeout_s)
        self._lc_threshold_m = float(lc_threshold_m)

        self._lock        = threading.Lock()
        self._latest_msg  = None
        self._recv_time   = -1.0
        self._last_px_pz  = None

        self._sub         = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_evt    = threading.Event()
        self._connected   = False

        self._connect()

    def _connect(self) -> None:
        """Attempt to set up the commlink Subscriber for orb/pose."""
        try:
            from commlink import Subscriber
            self._sub = Subscriber(
                host=self._host, port=self._port,
                topics=[ORB_POSE_TOPIC],
            )
            self._poll_thread = threading.Thread(
                target=self._poll_loop, name="orbv2-sub-poll", daemon=True
            )
            self._poll_thread.start()
            self._connected = True
            print(
                f"[orbv2] Subscribed to orb/pose on "
                f"{self._host}:{self._port}"
            )
        except Exception as exc:
            print(
                f"[orbv2] WARN: Could not connect to orb/pose — "
                f"ORB-SLAM3 feedback disabled. ({exc})"
            )

    def _poll_loop(self) -> None:
        """Background thread: pull the latest orb/pose and cache it."""
        while not self._stop_evt.is_set():
            try:
                msg = self._sub[ORB_POSE_TOPIC]
                if msg is not None:
                    with self._lock:
                        self._latest_msg = msg
                        self._recv_time  = time.time()
                # Small sleep to prevent ZMQ socket contention
                # (avoids !_more assertion in fq.cpp under heavy load)
                time.sleep(0.01)
            except Exception:
                time.sleep(0.1)

    def is_fresh(self) -> bool:
        """Return True if a pose was received within stale_timeout_s."""
        with self._lock:
            return (
                self._latest_msg is not None
                and (time.time() - self._recv_time) < self._stale_timeout
            )

    def get_pose(self) -> Optional[Tuple[np.ndarray, float, bool, bool]]:
        """Return the latest ORB-SLAM3 pose if fresh, else None.

        Returns
        -------
        (z, ts_s, tracking_ok, is_loop_closure)  or  None

        z               : np.ndarray [px, pz, yaw]  in Y-up EKF world frame
        ts_s            : float  —  message timestamp in seconds
        tracking_ok     : bool
        is_loop_closure : bool  —  True if position jumped > lc_threshold_m
        """
        with self._lock:
            if self._latest_msg is None:
                return None
            age = time.time() - self._recv_time
            if age > self._stale_timeout:
                return None
            msg = list(self._latest_msg)

        if len(msg) < 9:
            return None

        quat = np.array(msg[0:4], dtype=np.float32)
        trans = np.array(msg[4:7], dtype=np.float32)
        ts_ns = float(msg[7])
        tracking_ok = float(msg[8]) > 0.5

        if not tracking_ok:
            return None

        if not (np.all(np.isfinite(quat)) and np.all(np.isfinite(trans))):
            return None

        # Extract yaw around Y axis (same convention as EKFSlamSource)
        R = _quat_to_matrix_3x3(quat)
        yaw = float(np.arctan2(-R[2, 0], R[0, 0]))
        px  = float(trans[0])
        pz  = float(trans[2])

        # Loop-closure detection: large jump between consecutive ORB poses
        is_loop_closure = False
        with self._lock:
            if self._last_px_pz is not None:
                jump = float(np.hypot(
                    px - self._last_px_pz[0],
                    pz - self._last_px_pz[1],
                ))
                if jump > self._lc_threshold_m:
                    is_loop_closure = True
                    print(
                        f"[orbv2] Loop-closure detected — "
                        f"position jump = {jump:.3f} m"
                    )
            self._last_px_pz = (px, pz)

        z = np.array([px, pz, yaw], dtype=float)
        return z, ts_ns / 1e9, tracking_ok, is_loop_closure

    def stop(self) -> None:
        self._stop_evt.set()
        if self._sub is not None:
            try:
                self._sub.stop()
            except Exception:
                pass


class OrbFusedSource:
    """Drop-in replacement for ``EKFSlamSource`` that adds ORB-SLAM3 corrections.

    Wraps an existing ``EKFSlamSource`` and runs a background thread that
    polls the ORB-SLAM3 pose subscriber at up to ``orb_update_hz`` and feeds
    loop-closure-aware updates into the EKF.

    If the ORB bridge goes offline, the system silently falls back to
    ZED + encoder-only fusion with zero overhead.

    Parameters
    ----------
    ekf_source : EKFSlamSource
        The existing EKF source (ZED VIO + wheel encoders).
    orb_host : str
        Host running orbv2.orb_bridge.
    orb_port : int
        ZMQ port for the orb/pose topic.
    orb_update_hz : float
        Maximum rate at which ORB updates are applied to the EKF.
    lc_threshold_m : float
        Position jump (m) threshold for loop-closure detection.
    """

    def __init__(
        self,
        ekf_source,
        orb_host:       str   = "127.0.0.1",
        orb_port:       int   = ORB_PUB_PORT,
        orb_update_hz:  float = 5.0,
        lc_threshold_m: float = 0.30,
    ):
        self._ekf = ekf_source
        self._orb = _OrbSubscriber(
            host=orb_host,
            port=orb_port,
            stale_timeout_s=2.0,
            lc_threshold_m=lc_threshold_m,
        )
        self._dt = 1.0 / max(1.0, float(orb_update_hz))

        self._stop_evt   = threading.Event()
        self._orb_thread = threading.Thread(
            target=self._orb_update_loop, name="orbv2-ekf-update", daemon=True
        )

        # Diagnostics
        self._last_orb_ts:      float = -1.0
        self._orb_update_count: int   = 0
        self._orb_lc_count:     int   = 0

        self._orb_thread.start()
        print(
            f"[OrbFusedSource] Started ORB-SLAM3 update thread at "
            f"≤{orb_update_hz:.0f} Hz"
        )

    def _orb_update_loop(self) -> None:
        """Poll ORB subscriber and apply updates to the EKF when fresh."""
        while not self._stop_evt.is_set():
            t0 = time.time()
            try:
                result = self._orb.get_pose()
                if result is not None:
                    z, ts_s, _ok, is_lc = result

                    # Debounce: skip if the same keyframe was already applied
                    if ts_s != self._last_orb_ts:
                        self._last_orb_ts = ts_s

                        # Access the inner EKF via the EKFSlamSource wrapper
                        accepted = self._ekf._ekf.update_orb(
                            z,
                            R_orb           = None,  # use default ORB noise
                            is_loop_closure = is_lc,
                        )

                        if accepted:
                            self._orb_update_count += 1
                            if is_lc:
                                self._orb_lc_count += 1
                            if self._orb_update_count % 20 == 1:
                                print(
                                    f"[OrbFusedSource] orb_updates="
                                    f"{self._orb_update_count}  "
                                    f"loop_closures={self._orb_lc_count}  "
                                    f"z=[{z[0]:.3f}, {z[1]:.3f}, "
                                    f"{float(np.degrees(z[2])):.1f}°]"
                                )
            except Exception as exc:
                print(f"[OrbFusedSource] WARN in orb update loop: {exc}")

            elapsed = time.time() - t0
            sleep_s = self._dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    # ── EKFSlamSource / ZedSub interface (fully delegated) ────────────────────

    def ready(self) -> bool:
        return self._ekf.ready()

    def stop(self) -> None:
        self._stop_evt.set()
        self._orb.stop()
        self._ekf.stop()

    def get_pose(self):
        return self._ekf.get_pose()

    def get_pcd_pose(self):
        return self._ekf.get_pcd_pose()

    def get_rgb_depth_pose(self):
        return self._ekf.get_rgb_depth_pose()

    def get_camera_info(self):
        return self._ekf.get_camera_info()

    def get_ekf_uncertainty(self) -> np.ndarray:
        return self._ekf.get_ekf_uncertainty()

    def get_tracking_confidence(self) -> float:
        return self._ekf.get_tracking_confidence()

    # ── Diagnostics accessors ─────────────────────────────────────────────────

    @property
    def orb_update_count(self) -> int:
        return self._orb_update_count

    @property
    def orb_loop_closure_count(self) -> int:
        return self._orb_lc_count

    @property
    def orb_is_fresh(self) -> bool:
        """True if ORB-SLAM3 bridge is alive and publishing recent poses."""
        return self._orb.is_fresh()
