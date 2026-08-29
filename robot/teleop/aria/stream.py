"""stream.py — subscriber side of the Aria hand-tracking wire.

The publisher stays in the aria2robot repo (`python -m aria2robot.stream_pub
--wifi`): it owns the Project Aria SDK, the hand tracking and the finger
retargeting. Nothing here imports any of that. This module duck-types the
payload, so the whole YOR side runs on a Jetson with no Aria dependency at all.

Wire format, commlink topic `wuji`
----------------------------------
  envelope   {"T_odom_device": (4,4), "t_pub": float, "t_wall": float,
              "left": {...}, "right": {...}}
  per side   {"qpos":            (20,)  finger angles, wuji-description order
              "kp_mp":           (21,3) MediaPipe landmarks, DEVICE frame
              "kp_mp_scaled":    (21,3) what the retargeter actually chased
              "T_device_wrist":  (4,4)  Aria's own wrist frame — VIEWER ONLY
              "T_device_hand":   (4,4)  the WUJI hand root — drives control
              "paused":          bool   the shaka toggle}

`T_device_hand` is the frame to follow, never `T_device_wrist`: its origin is
the wrist landmark `kp_mp[0]` and its axes are the hand model's, so it is the
operator-side twin of the robot's `{side}_wuji_hand_orient` body. Aria's own
wrist frame has its origin at a joint centre a couple of cm away and a
different axis convention entirely, so following it puts the arm somewhere the
operator is not looking. A publisher that sends only the old field is reported
loudly and its side stays disengaged.

`t_pub` is the *publisher's* `time.monotonic()`. It is a change detector and
nothing else — it is not wall time and it does not compare across machines.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np


def landmarks_in_world(kp_device: np.ndarray, T_world_device: np.ndarray) -> np.ndarray:
    """Lift (N, 3) device-frame landmarks into the world frame."""
    h = np.concatenate([kp_device, np.ones((kp_device.shape[0], 1))], axis=1)
    return (T_world_device.astype(np.float64) @ h.T).T[:, :3].astype(np.float32)


def canonical_joint_names(side: str) -> tuple[str, ...]:
    """Expected actuated-joint order for a side's WUJI hand.

    The MJCF names its hand joints exactly as wuji-description does, so the
    published (20,) vector maps straight across with no reordering.
    """
    return tuple(f"{side}_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5))


class SideSample(NamedTuple):
    """One hand's worth of the latest payload, already lifted into odom."""
    T_odom_wrist: np.ndarray | None  # T_odom_device @ T_device_hand
    kp_odom: np.ndarray | None       # (21, 3) landmarks in odom
    qpos: np.ndarray | None          # (20,) finger angles, unclipped
    paused: bool


class AriaHandStream:
    """Background subscriber to the publisher's `meta` and `wuji` topics.

    One daemon thread per topic pulls the newest payload and drops it into a
    lock-protected slot; `snapshot()` reads that slot. Consumers therefore run
    at whatever rate suits them (30 Hz for the teleop client, 100 Hz for the
    sim viewer) without either rate being tied to the publisher's.

    Args:
        stale_s: Report every side as paused once nothing has arrived for this
            long. commlink's `buffer=False` subscriber hands back the last
            payload forever, so a publisher that dies mid-motion would
            otherwise leave `paused` False and the clutch engaged — holding a
            target the operator can no longer release by gesture. `None`
            disables the gate, which is right for the sim viewer and wrong for
            anything driving hardware.
    """

    TOPICS = ("meta", "wuji")
    STREAM_POLL_S = 0.005
    STATE_POLL_S = 0.2

    def __init__(self, host: str, port: int = 5555,
                 sides: Iterable[str] = ("left", "right"),
                 stale_s: float | None = None):
        self.host, self.port = host, int(port)
        self.sides = tuple(sides)
        self.stale_s = None if stale_s is None else float(stale_s)
        self._lock = threading.Lock()
        self._latest: dict[str, SideSample] = {
            s: SideSample(None, None, None, True) for s in self.sides
        }
        self._meta: dict | None = None
        self._t_recv: float | None = None
        self._warned_old = False
        self._stop = threading.Event()
        self._sub = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        import commlink  # deferred so the other teleop backends never load it

        self._sub = commlink.Subscriber(self.host, self.port,
                                        topics=list(self.TOPICS), buffer=False)
        for topic in self.TOPICS:
            t = threading.Thread(target=self._topic_loop, args=(topic,),
                                 name=f"aria-{topic}", daemon=True)
            t.start()
            self._threads.append(t)
        print(f"[aria] subscribing to tcp://{self.host}:{self.port} "
              f"topics={'+'.join(self.TOPICS)}")

    def stop(self) -> None:
        self._stop.set()
        if self._sub is not None:
            self._sub.stop()

    def meta(self) -> dict | None:
        with self._lock:
            return self._meta

    def wait_for_meta(self, timeout: float = 5.0) -> dict | None:
        """Block until the publisher's capability message lands, or give up."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stop.is_set():
            m = self.meta()
            if m is not None:
                return m
            time.sleep(0.05)
        return None

    def snapshot(self) -> dict[str, SideSample]:
        """The latest sample per side, with the staleness gate applied."""
        with self._lock:
            latest = dict(self._latest)
            t_recv = self._t_recv
        if self.stale_s is None or t_recv is None:
            return latest
        if time.monotonic() - t_recv <= self.stale_s:
            return latest
        return {s: v._replace(paused=True) for s, v in latest.items()}

    def _topic_loop(self, topic: str) -> None:
        """Pull one topic, forwarding only payloads newer than the last."""
        idle_wait = self.STATE_POLL_S if topic == "meta" else self.STREAM_POLL_S
        last_ts = None
        while not self._stop.is_set():
            msg = self._sub.get(topic)
            if not isinstance(msg, dict) or msg.get("t_pub") == last_ts:
                self._stop.wait(idle_wait)
                continue
            last_ts = msg.get("t_pub")
            if topic == "meta":
                with self._lock:
                    self._meta = msg
            else:
                self._ingest(msg)

    def _ingest(self, msg: dict) -> None:
        """Decode one `wuji` payload into per-side samples. Pure: dict in, slot out."""
        T_od = msg.get("T_odom_device")
        T_od = None if T_od is None else np.asarray(T_od, dtype=np.float64)
        samples: dict[str, SideSample] = {}
        for side in self.sides:
            side_msg = msg.get(side)
            if not isinstance(side_msg, dict):
                samples[side] = SideSample(None, None, None, True)
                continue
            qpos = side_msg.get("qpos")
            T_wrist = kp_odom = None
            # The WUJI hand root, not Aria's own wrist frame: its origin is
            # kp_mp[0] and its axes are the hand model's, so the arm follows
            # exactly the hand the operator sees
            T_dh = side_msg.get("T_device_hand")
            if T_od is not None and T_dh is not None:
                T_wrist = T_od @ np.asarray(T_dh, dtype=np.float64)
            elif side_msg.get("T_device_wrist") is not None and not self._warned_old:
                self._warned_old = True
                print("[aria] publisher sends T_device_wrist but no "
                      "T_device_hand -- update stream_pub; arms stay "
                      "disengaged rather than follow the wrong frame")
            kp_mp = side_msg.get("kp_mp")
            if T_od is not None and kp_mp is not None:
                kp_odom = landmarks_in_world(np.asarray(kp_mp), T_od)
            samples[side] = SideSample(
                T_odom_wrist=T_wrist,
                kp_odom=kp_odom,
                qpos=None if qpos is None else np.asarray(qpos, dtype=np.float64),
                paused=bool(side_msg.get("paused", True)),
            )
        with self._lock:
            self._latest = samples
            self._t_recv = time.monotonic()
