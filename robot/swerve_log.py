"""
swerve_log.py — per-module swerve telemetry to CSV, independent of whole-body.

`WholeBodyController` already logs swerve telemetry into its trajectory CSV,
but only when whole-body control is running, and only at the 30 Hz solve rate.
Neither suits base work:

* driving the base from robot/teleop/joystick.py with `yor.py --no-arms` builds
  no WholeBodyController at all, so nothing is recorded;
* 30 Hz samples a 50 Hz status stream, which is enough for module slew
  (hundreds of ms) but marginal for anything faster.

This runs off the base control loop instead, so a joystick run and a teleop run
produce comparable data. It duplicates some columns of the trajectory log on
purpose: that one exists to correlate the wheels with *solver* output on the
same tick, this one exists to look at the wheels on their own.

Reads are cheap -- `swerve_telemetry` returns arrays already in memory plus
cached periodic-status frames -- so the sample rate is bounded by what the
SPARKs publish (50 Hz for frame 2), not by bus traffic.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_HZ = 50.0          # matches the SPARK periodic status 2 period (20 ms)


class SwerveRecorder:
    """Samples `Base.swerve_telemetry()` on its own thread and writes a CSV."""

    def __init__(self, path: Path, base, module_labels: Sequence[str],
                 sample_hz: float = DEFAULT_HZ,
                 config_notes: Optional[Sequence[str]] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._base = base
        self._labels = list(module_labels)
        self._period = 1.0 / max(sample_hz, 1e-3)
        # Line-buffered, so a Ctrl-C loses at most the row in flight.
        self._file = self.path.open("w", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.writer(self._file)
        if config_notes:
            self._writer.writerow(["# " + str(config_notes[0])] + [str(n) for n in config_notes[1:]])
        header = ["t", "motors_enabled"]
        header += [f"v_target_{i}" for i in range(3)]
        header += [f"v_prof_{i}" for i in range(3)]
        for group in ("steer_cmd", "steer_meas", "drive_cmd", "drive_meas", "drive_pos"):
            header += [f"{group}_{m}" for m in self._labels]
        self._writer.writerow(header)
        self._n = len(self._labels)
        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="SwerveRecorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self._file.close()
        except Exception:
            pass

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _vec(values, n: int) -> list:
        """`n` formatted floats, with anything missing recorded as nan.

        A dropped frame must read back as missing, never as a module sitting
        at zero.
        """
        arr = (np.full(n, np.nan) if values is None
               else np.asarray(values, dtype=float).ravel())
        return [("nan" if i >= arr.size or not np.isfinite(arr[i])
                 else f"{float(arr[i]):.6f}") for i in range(n)]

    def sample(self) -> None:
        """Write one row. Never raises -- a logger must not stop the robot."""
        try:
            tel = self._base.swerve_telemetry()
        except Exception:
            return
        row = [f"{time.monotonic() - self._t0:.4f}",
               str(bool(tel.get("motors_enabled", False)))]
        row += self._vec(tel.get("v_target"), 3)
        row += self._vec(tel.get("v_profiled"), 3)
        for key in ("steer_cmd_rad", "steer_meas_rad", "drive_cmd_mps",
                    "drive_meas_raw", "drive_pos_rot"):
            row += self._vec(tel.get(key), self._n)
        try:
            self._writer.writerow(row)
        except Exception:
            pass

    def _run(self) -> None:
        next_at = time.monotonic()
        while not self._stop.is_set():
            self.sample()
            next_at += self._period
            delay = next_at - time.monotonic()
            if delay < 0:
                next_at = time.monotonic()      # fell behind; do not spiral
            else:
                self._stop.wait(delay)
