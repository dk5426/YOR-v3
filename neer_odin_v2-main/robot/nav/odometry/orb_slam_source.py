"""
orb_slam_source.py — ORB-SLAM3 pose subscriber and EKF fusion layer.

Classes
-------
OrbSlamSub
    Subscribes to the ``orb/pose`` ZMQ topic published by orbslam_bridge.py
    and delivers the latest pose in the EKF's native [px, pz, yaw] format.

OrbEKFSlamSource
    Drop-in replacement for EKFSlamSource that adds a second EKF update
    step using ORB-SLAM3 poses whenever they are fresh.  Falls back to the
    existing ZED+encoder behaviour if ORB-SLAM3 is not publishing.

Coordinate conventions
----------------------
- EKF state:  [px, pz, yaw]  in Y-up world frame (robot moves in XZ plane).
- OrbSlamSub: converts the published [qx,qy,qz,qw,tx,ty,tz] into [px,pz,yaw]
  using the same arctan2 extraction already in ZedSub.get_pose().
- Loop-closure detection: a pose jump > `lc_threshold_m` between consecutive
  ORB-SLAM3 poses triggers an unconditional EKF update (gate disabled, tighter R).
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np


# ── Topic / port constants ────────────────────────────────────────────────────
ORB_POSE_TOPIC = "orb/pose"
ORB_PUB_PORT   = 6001   # matches orbslam_bridge.py default

# orb/pose message layout  (list of floats, length 9):
#   [0:4]  qx qy qz qw   (camera quaternion in ZED world frame, Y-up)
#   [4:7]  tx ty tz       (camera translation in ZED world frame, m)
#   [7]    ts_ns          (timestamp, nanoseconds as float)
#   [8]    tracking_ok    (1.0 = good, 0.0 = SLAM lost / initializing)


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


# ══════════════════════════════════════════════════════════════════════════════
#  OrbSlamSub — lightweight subscriber for orb/pose
# ══════════════════════════════════════════════════════════════════════════════

class OrbSlamSub:
    """Subscribes to ``orb/pose`` and exposes the latest pose for EKF fusion.

    Parameters
    ----------
    host : str
        Host running orbslam_bridge.py (default: localhost).
    port : int
        ZMQ port for the orb/pose topic (default: 6001).
    stale_timeout_s : float
        Poses older than this are considered stale and ignored (default: 2.0 s).
    lc_threshold_m : float
        Minimum position jump (metres) to classify a pose update as a
        loop-closure event, triggering an unconditional EKF update (default: 0.3 m).
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
        self._latest_msg  = None   # raw list[float] from orb/pose
        self._recv_time   = -1.0   # wall-clock time of last receipt
        self._last_px_pz  = None   # (px, pz) from previous accepted pose

        self._sub         = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_evt    = threading.Event()

        self._connected   = False
        self._connect()

    # ── Internal connection ───────────────────────────────────────────────────

    def _connect(self) -> None:
        """Attempt to set up the commlink Subscriber for orb/pose.

        Silently degrades if commlink is unavailable or the bridge is not
        running — the rest of the SLAM node works without ORB-SLAM3.

        commlink 0.4.0 pull mode: sub[topic] issues a DEALER round-trip to the
        publisher on each call and returns the latest cached value. No socket
        timeout configuration is needed.
        """
        try:
            from commlink import Subscriber
            # buffer=False (default) = pull mode: lowest-latency, returns latest.
            self._sub = Subscriber(
                host=self._host, port=self._port,
                topics=[ORB_POSE_TOPIC],
            )
            self._poll_thread = threading.Thread(
                target=self._poll_loop, name="orb-sub-poll", daemon=True
            )
            self._poll_thread.start()
            self._connected = True
            print(
                f"[OrbSlamSub] Subscribed to orb/pose on "
                f"{self._host}:{self._port}"
            )
        except Exception as exc:
            print(
                f"[OrbSlamSub] WARN: Could not connect to orb/pose — "
                f"ORB-SLAM3 feedback disabled. ({exc})"
            )

    def _poll_loop(self) -> None:
        """Background thread: pull the latest orb/pose and cache it.

        commlink 0.4.0 pull mode: sub[topic] blocks on the very first call until
        the publisher sends a value, then returns the cached value on subsequent
        calls without blocking. No zmq.Again is raised in this mode.
        """
        while not self._stop_evt.is_set():
            try:
                msg = self._sub[ORB_POSE_TOPIC]
                if msg is not None:
                    with self._lock:
                        self._latest_msg = msg
                        self._recv_time  = time.time()
            except Exception:
                # Bridge not up yet, or connection reset — back off briefly.
                time.sleep(0.1)

    # ── Public API ────────────────────────────────────────────────────────────

    def is_fresh(self) -> bool:
        """Return True if a pose was received within ``stale_timeout_s``."""
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

        z             : np.ndarray [px, pz, yaw]  in Y-up EKF world frame
        ts_s          : float  —  message timestamp in seconds
        tracking_ok   : bool   —  False when ORB-SLAM3 is lost / initializing
        is_loop_closure : bool —  True when position jumped > lc_threshold_m
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

        quat = np.array(msg[0:4], dtype=np.float32)   # [qx, qy, qz, qw]
        trans = np.array(msg[4:7], dtype=np.float32)   # [tx, ty, tz]
        ts_ns = float(msg[7])
        tracking_ok = float(msg[8]) > 0.5

        if not tracking_ok:
            return None

        # Validity check
        if not (np.all(np.isfinite(quat)) and np.all(np.isfinite(trans))):
            return None

        # Extract yaw around Y axis — same convention as ZedSub.get_pose()
        # and EKFSlamSource._apply_zed_update():
        #   yaw = arctan2(−T[2,0], T[0,0])
        R = _quat_to_matrix_3x3(quat)
        yaw = float(np.arctan2(-R[2, 0], R[0, 0]))
        px  = float(trans[0])
        pz  = float(trans[2])

        # Loop-closure detection: large jump between consecutive poses
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
                        f"[OrbSlamSub] Loop-closure detected — "
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


# ══════════════════════════════════════════════════════════════════════════════
#  OrbEKFSlamSource — wraps EKFSlamSource with ORB-SLAM3 correction
# ══════════════════════════════════════════════════════════════════════════════

class OrbEKFSlamSource:
    """Wraps ``EKFSlamSource`` and adds a secondary EKF update from ORB-SLAM3.

    Drop-in replacement for ``EKFSlamSource`` — exposes the same interface
    (``ready``, ``stop``, ``get_pose``, ``get_pcd_pose``, ``get_rgb_depth_pose``,
    ``get_camera_info``, ``get_ekf_uncertainty``).

    The ORB-SLAM3 update runs in a background thread at the keyframe rate
    (~1-5 Hz).  If ORB-SLAM3 stops publishing (bridge offline, tracking lost),
    the system silently falls back to ZED+encoder fusion.

    Parameters
    ----------
    ekf_source : EKFSlamSource
        The existing EKF source (ZED VIO + wheel encoders).
    orb_sub : OrbSlamSub
        The ORB-SLAM3 pose subscriber.
    orb_update_hz : float
        Maximum rate at which ORB updates are applied (default: 5 Hz).
        Set lower if you want to budget CPU during planning.
    orb_R : np.ndarray or None
        3×3 measurement noise for ORB-SLAM3 updates.  None = use EKF default.
    """

    def __init__(
        self,
        ekf_source,           # EKFSlamSource instance
        orb_sub: OrbSlamSub,
        orb_update_hz: float  = 5.0,
        orb_R: Optional[np.ndarray] = None,
    ):
        self._ekf   = ekf_source
        self._orb   = orb_sub
        self._R_orb = orb_R
        self._dt    = 1.0 / max(1.0, float(orb_update_hz))

        self._stop_evt   = threading.Event()
        self._orb_thread = threading.Thread(
            target=self._orb_update_loop, name="orb-ekf-update", daemon=True
        )
        self._last_orb_ts: float = -1.0   # prevent double-update on same keyframe
        self._orb_update_count: int = 0
        self._orb_lc_count:     int = 0

        self._orb_thread.start()
        print(
            f"[OrbEKFSlamSource] Started ORB-SLAM3 update thread at "
            f"≤{orb_update_hz:.0f} Hz"
        )

    # ── Background ORB update loop ────────────────────────────────────────────

    def _orb_update_loop(self) -> None:
        """Poll OrbSlamSub and apply updates to the EKF whenever fresh."""
        while not self._stop_evt.is_set():
            t0 = time.time()
            try:
                result = self._orb.get_pose()
                if result is not None:
                    z, ts_s, _ok, is_lc = result
                    # Debounce: skip if the same keyframe was already applied
                    if ts_s != self._last_orb_ts:
                        self._last_orb_ts = ts_s
                        accepted = self._ekf._ekf.update_orb(
                            z,
                            R_orb           = self._R_orb,
                            is_loop_closure = is_lc,
                        )
                        if accepted:
                            self._orb_update_count += 1
                            if is_lc:
                                self._orb_lc_count += 1
                            if self._orb_update_count % 20 == 1:
                                print(
                                    f"[OrbEKFSlamSource] orb_updates={self._orb_update_count}  "
                                    f"loop_closures={self._orb_lc_count}  "
                                    f"z=[{z[0]:.3f}, {z[1]:.3f}, "
                                    f"{float(np.degrees(z[2])):.1f}°]"
                                )
            except Exception as exc:
                print(f"[OrbEKFSlamSource] WARN in orb update loop: {exc}")

            elapsed = time.time() - t0
            sleep_s = self._dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    # ── ZedSub / EKFSlamSource interface (fully delegated) ───────────────────

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

    @property
    def orb_update_count(self) -> int:
        return self._orb_update_count

    @property
    def orb_loop_closure_count(self) -> int:
        return self._orb_lc_count
