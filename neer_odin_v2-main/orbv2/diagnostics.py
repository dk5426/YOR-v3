#!/usr/bin/env python3
"""
diagnostics.py — Health monitoring for the orbv2 ORB-SLAM3 pipeline.

Provides ``OrbHealthMonitor``, a lightweight diagnostic tracker that
periodically logs the status of the ORB-SLAM3 integration: tracking ratio,
loop-closure count, pose staleness, and EKF innovation magnitudes.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class OrbHealthMonitor:
    """Periodic health-status logger for the orbv2 pipeline.

    Attaches to an ``OrbFusedSource`` and prints a summary every
    ``log_interval_s`` seconds.

    Parameters
    ----------
    fused_source : OrbFusedSource
        The ORB+EKF fused pose source.
    log_interval_s : float
        Seconds between status log lines (default: 10.0).
    """

    def __init__(self, fused_source, log_interval_s: float = 10.0):
        self._source    = fused_source
        self._interval  = max(1.0, float(log_interval_s))
        self._stop_evt  = threading.Event()
        self._thread    = threading.Thread(
            target=self._log_loop, name="orbv2-health", daemon=True
        )
        self._started   = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()
        print(
            f"[OrbHealthMonitor] Logging every {self._interval:.0f}s"
        )

    def stop(self) -> None:
        self._stop_evt.set()

    def _log_loop(self) -> None:
        while not self._stop_evt.is_set():
            time.sleep(self._interval)
            try:
                self._print_status()
            except Exception as exc:
                print(f"[OrbHealthMonitor] Error: {exc}")

    def _print_status(self) -> None:
        src = self._source

        orb_updates = src.orb_update_count
        orb_lc      = src.orb_loop_closure_count
        orb_fresh   = src.orb_is_fresh

        # EKF uncertainty
        try:
            sigma = src.get_ekf_uncertainty()
            sigma_pos = float(np.hypot(sigma[0], sigma[1]))
            sigma_yaw_deg = float(np.degrees(sigma[2]))
        except Exception:
            sigma_pos = -1.0
            sigma_yaw_deg = -1.0

        status = "LIVE" if orb_fresh else "STALE/OFFLINE"

        print(
            f"[orbv2 health] ORB={status}  "
            f"updates={orb_updates}  loop_closures={orb_lc}  "
            f"σ_pos={sigma_pos:.4f}m  σ_yaw={sigma_yaw_deg:.2f}°"
        )

    def get_status_dict(self) -> dict:
        """Return a snapshot of health metrics as a dictionary."""
        src = self._source
        try:
            sigma = src.get_ekf_uncertainty()
            sigma_pos = float(np.hypot(sigma[0], sigma[1]))
            sigma_yaw = float(sigma[2])
        except Exception:
            sigma_pos = -1.0
            sigma_yaw = -1.0

        return {
            "orb_updates": src.orb_update_count,
            "orb_loop_closures": src.orb_loop_closure_count,
            "orb_fresh": src.orb_is_fresh,
            "ekf_sigma_pos_m": sigma_pos,
            "ekf_sigma_yaw_rad": sigma_yaw,
        }
