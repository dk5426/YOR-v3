"""stream.py — subscriber side of the Aria hand-tracking wire.

The publisher stays in the aria2robot repo (`python -m aria2robot.stream_pub
--wifi`): it owns the Project Aria SDK, the hand tracking and the finger
retargeting. Nothing here imports any of that. This module duck-types the
payload, so the whole YOR side runs on a Jetson with no Aria dependency at all.

Wire format, commlink topic `qpos`, version 2
---------------------------------------------
  envelope   {"wire": 2, "seq": int, "home_seq": int,
              "t_pub": float, "t_wall": float,
              "left": {...}, "right": {...}}
  per side   {"qpos":        (N,)   finger angles, in the active hand's own
                                    joint order (WUJI: 20, Aero: 16)
              "T_odom_hand": (4,4)  the hand root in odom — drives control
              "paused":      bool   the shaka toggle}

Which hand (WUJI or Aero) is picked by config (`hand.type`), not by this
payload -- the publisher also names it once, at low rate, in `meta["hand"]`,
which `robot/hand/hands.py` checks against the configured type and warns
loudly (never fatally) on a mismatch.

`T_odom_hand` arrives pre-composed. It used to be `T_odom_device` and
`T_device_hand` shipped separately for this module to multiply together, along
with the raw landmarks both the finger retargeting and the home gesture were
read from. All of that now happens in the publisher, which owns the Aria SDK
and is the only process with a reason to hold a landmark. What is left on the
wire is what the robot acts on, and it is a quarter of the size.

Its origin is the wrist landmark and its axes are the hand model's, so it is
the operator-side twin of the robot's `{side}_wuji_hand_orient` body — not
Aria's own `transform_device_wrist`, whose origin is a joint centre a couple of
cm away with a different axis convention. A pre-wire-2 publisher is composed
for, loudly, so an un-upgraded glasses host degrades to a warning rather than
to arms that never move.

`home_seq` counts completed two-hand thumbs-up gestures. A counter, not a flag:
PUB/SUB drops packets and this subscriber conflates them, so an edge can be
missed but a total cannot. See `HomeSeqWatcher`.

`t_pub` is the *publisher's* `time.monotonic()`. It is a change detector and
nothing else — it is not wall time and it does not compare across machines.
`seq` is the publisher's own publish count, and `received / seq` over an
interval is the drop ratio.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

# Defined in the hand package, which the sim node also reads it from -- one
# table, so the wire order and the model order cannot drift apart. Re-exported
# here because this is where the subscriber side has always found it.
from robot.hand.wuji_driver import canonical_joint_names

__all__ = ["AriaHandStream", "ClutchCmdWatcher", "HomeSeqWatcher", "SideSample",
           "canonical_joint_names"]


class SideSample(NamedTuple):
    """One hand's worth of the latest payload."""
    T_odom_wrist: np.ndarray | None  # the hand root in odom
    qpos: np.ndarray | None          # (N,) finger angles, unclipped
    paused: bool


class HomeSeqWatcher:
    """Fires on an increase in the publisher's home counter, and nothing else.

    Four transitions, three of which must stay silent:

      None -> n   a client joining a publisher that has already homed. Adopt
                  the value; homing on connect would be a surprise.
      n -> n      the common case.
      n -> m<n    the publisher restarted and its counter began again. Resync
                  without firing -- the same guard `status.py` puts on the
                  cumulative `sends` total for the same reason.
      n -> m>n    fire, *once*, whatever the size of the jump. commlink's
                  `buffer=False` subscriber conflates, so a jump of three is a
                  gesture whose intermediate packets we simply did not see, not
                  three requests. Homing twice in a row is a hardware hazard.
    """

    def __init__(self) -> None:
        self._last: int | None = None

    def update(self, seq: int | None) -> bool:
        if seq is None:
            return False
        prev, self._last = self._last, int(seq)
        if prev is None or seq < prev:
            return False
        return seq > prev


class AriaHandStream:
    """Background subscriber to the publisher's `meta` and `qpos` topics.

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

    TOPICS = ("meta", "qpos")
    STREAM_POLL_S = 0.005
    STATE_POLL_S = 0.2

    def __init__(self, host: str, port: int = 5555,
                 sides: Iterable[str] = ("left", "right"),
                 stale_s: float | None = None, stats=None):
        self.host, self.port = host, int(port)
        # Optional StreamStats (robot/teleop/aria/stats.py). Off unless a
        # caller wants the numbers: sizing every payload costs a pickle per
        # message, which the node's own finger path has no use for.
        self.stats = stats
        self.sides = tuple(sides)
        self.stale_s = None if stale_s is None else float(stale_s)
        self._lock = threading.Lock()
        self._latest: dict[str, SideSample] = {
            s: SideSample(None, None, True) for s in self.sides
        }
        self._meta: dict | None = None
        self._t_recv: float | None = None
        self._home_seq: int | None = None
        self._warned: set[str] = set()
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

    def home_seq(self) -> int | None:
        """The publisher's completed-home count, None until one arrives.

        None is also what a pre-wire-2 publisher gives, so "old publisher" and
        "not connected yet" collapse into "never fires" -- the safe default.
        """
        with self._lock:
            return self._home_seq

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[aria] {msg}")

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
            if self.stats is not None:
                # t_wall, not t_pub: the publisher's wall clock, which
                # ClockSync can put on the same footing as ours. Sizing the
                # payload here rather than on the wire is an approximation of
                # bandwidth, and the same one aria2robot's subscriber makes.
                import pickle
                self.stats.hit(topic, msg.get("t_wall"), t_recv=time.time(),
                               bytes_n=len(pickle.dumps(msg)))
            if topic == "meta":
                with self._lock:
                    self._meta = msg
            else:
                self._ingest(msg)

    def _ingest(self, msg: dict) -> None:
        """Decode one `qpos` payload into per-side samples. Pure: dict in, slot out."""
        # Pre-wire-2 publishers shipped the two halves separately and left the
        # multiply to us. Kept for one release: the two repos deploy to
        # different machines, and "the arms do not move" is a worse thing to
        # hand an operator than a yellow line.
        T_od = msg.get("T_odom_device")
        T_od = None if T_od is None else np.asarray(T_od, dtype=np.float64)
        samples: dict[str, SideSample] = {}
        for side in self.sides:
            side_msg = msg.get(side)
            if not isinstance(side_msg, dict):
                samples[side] = SideSample(None, None, True)
                continue
            samples[side] = SideSample(
                T_odom_wrist=self._hand_pose(side_msg, T_od),
                qpos=self._qpos(side_msg),
                paused=bool(side_msg.get("paused", True)),
            )
        home_seq = msg.get("home_seq")
        with self._lock:
            self._latest = samples
            self._t_recv = time.monotonic()
            if isinstance(home_seq, int):
                self._home_seq = home_seq

    def _hand_pose(self, side_msg: dict, T_od) -> np.ndarray | None:
        """The WUJI hand root in odom, upcast to float64 whatever arrives.

        The wire is float32 because `T_odom_device` was float32 upstream; every
        consumer downstream of here does SE(3) algebra and gets float64.
        """
        T_oh = side_msg.get("T_odom_hand")
        if T_oh is not None:
            return np.asarray(T_oh, dtype=np.float64)
        T_dh = side_msg.get("T_device_hand")
        if T_dh is not None and T_od is not None:
            self._warn_once("wire1", "publisher is pre-wire-2 (no T_odom_hand); "
                            "composing locally. The home gesture is unavailable "
                            "until stream_pub is updated")
            return T_od @ np.asarray(T_dh, dtype=np.float64)
        if side_msg.get("T_device_wrist") is not None:
            self._warn_once("wrist_only", "publisher sends T_device_wrist but no "
                            "hand frame -- update stream_pub; arms stay "
                            "disengaged rather than follow the wrong frame")
        return None

    @staticmethod
    def _qpos(side_msg: dict) -> np.ndarray | None:
        qpos = side_msg.get("qpos")
        return None if qpos is None else np.asarray(qpos, dtype=np.float64)


class ClutchCmdWatcher:
    """The recording station's (Thor) engage/disengage, off the control path.

    Two things need the station's latch and neither may block on a socket to
    read it: the teleop client, which gates the arms, and `Hands` inside the
    node, which gates the fingers. They subscribe separately, for the same
    reason the `qpos` path does -- the finger loop must never wait on anything
    the arm path is doing.

    Survives the station restarting, which is the whole reason this is not
    three lines. commlink's pull socket is a DEALER with no receive timeout:
    if the publisher dies between the request and its reply, `get()` blocks
    forever, and because DEALER does not resend on reconnect the reply never
    comes even once the station is back. The transport reconnects, `ss` shows
    ESTABLISHED, and the value is frozen at whatever it was -- silently, for
    the life of the process. So the fetch runs on a thread this class can
    abandon: if nothing has arrived for `stale_s`, the subscriber is rebuilt
    and the stuck thread is left to its fate (one leaked thread per station
    restart, which is a price worth paying).

    `engaged` is False until the station has actually said engage, and False
    again whenever the feed goes stale. A station that cannot be heard from
    holds the robot -- it never hands it over on the strength of a default or
    a value that stopped being refreshed.
    """

    POLL_S = 0.02
    STALE_S = 3.0
    DEFAULT_PORT = 5559
    TOPIC = "clutch_cmd"

    def __init__(self, host: str, port: int = DEFAULT_PORT,
                 stale_s: float = STALE_S):
        self.host = str(host)
        self.port = int(port)
        self.stale_s = float(stale_s)
        self._engaged = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._gen = 0
        self._last_ok = 0.0
        self._thread: threading.Thread | None = None

    @property
    def engaged(self) -> bool:
        with self._lock:
            return self._engaged

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._supervise,
                                        name="clutch-cmd-watch", daemon=True)
        self._thread.start()
        print(f"[aria] clutch_cmd <- tcp://{self.host}:{self.port} "
              f"(stale after {self.stale_s:.0f}s -> disengaged)")

    def stop(self) -> None:
        self._stop.set()
        self._gen += 1
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _disengage(self, why: str) -> None:
        with self._lock:
            was, self._engaged = self._engaged, False
        if was:
            print(f"[aria] clutch_cmd {why} -- holding disengaged")

    def _supervise(self) -> None:
        """Own the connection; rebuild it whenever the feed stops arriving."""
        import commlink

        while not self._stop.is_set():
            try:
                sub = commlink.Subscriber(self.host, self.port,
                                          topics=[self.TOPIC], buffer=False)
            except Exception as exc:
                self._disengage(f"connect failed ({exc})")
                self._stop.wait(1.0)
                continue
            self._gen += 1
            gen = self._gen
            self._last_ok = time.monotonic()
            threading.Thread(target=self._fetch, args=(sub, gen), daemon=True,
                             name=f"clutch-cmd-fetch-{gen}").start()
            while not self._stop.is_set() and gen == self._gen:
                if time.monotonic() - self._last_ok > self.stale_s:
                    # Abandon this generation: the fetch thread may be wedged
                    # in a recv that will never return, and nothing short of a
                    # new socket gets us out of it.
                    self._disengage("stale")
                    self._gen += 1
                    break
                self._stop.wait(0.2)

    def _fetch(self, sub, gen: int) -> None:
        while not self._stop.is_set() and gen == self._gen:
            try:
                m = sub.get(self.TOPIC)
            except Exception:
                m = None
            if gen != self._gen:      # abandoned while we were blocked
                return
            if m is not None:
                with self._lock:
                    self._engaged = bool(m.get("engage"))
                self._last_ok = time.monotonic()
            self._stop.wait(self.POLL_S)
