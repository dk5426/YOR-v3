"""stats.py — per-topic receive stats for the Aria subscription.

The counterpart of aria2robot's `stream_sub.py` table, on the YOR side and
without its dependencies. Same columns and the same meaning, so a number read
here is comparable with one read there.

Latency is the only part that is not self-contained. Publisher and subscriber
each stamp with their own `time.time()`, so `t_recv - t_wall` is the transport
delay *plus* whatever the two wall clocks disagree by -- on separate machines
that offset is routinely larger than the thing being measured, and it can be
negative. `ClockSync` removes it with the classic four-timestamp handshake
against the publisher's REP port, keeping the offset from the lowest-RTT
sample in a short history because the least-delayed round trip is the least
asymmetric one. Without the handshake the p50/p95 columns read `--` rather
than a confidently wrong number.

This is deliberately *not* `t_pub`, which the wire also carries: `t_pub` is
the publisher's `time.monotonic()`, a change detector whose origin is that
process's start. It cannot be compared across machines at all.
"""

from __future__ import annotations

import threading
import time
from collections import deque

WINDOW_S = 5.0
REFRESH_S = 10.0
SAMPLE_HISTORY = 7


def fmt_bw(bps: float) -> str:
    """Bytes/sec as Mbps at or above 1.0, kbps below."""
    mbps = bps * 8 / 1e6
    return f"{mbps:.2f} Mbps" if mbps >= 1.0 else f"{mbps * 1000:.1f} kbps"


class StreamStats:
    """Windowed count, rate, latency and bandwidth for a fixed set of topics.

    `hit()` runs on the receive threads and does no work beyond an append;
    everything is derived in `snapshot()`, which the 1 Hz redraw calls.
    """

    def __init__(self, topics: tuple[str, ...], window_s: float = WINDOW_S):
        self.topics = tuple(topics)
        # Every derived number is a division by this, and a zero window is
        # not a shorter window, it is a crash on the first snapshot
        self.window_s = max(float(window_s), 0.1)
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {t: 0 for t in self.topics}
        # (t_monotonic, latency_ms | None, bytes | None)
        self._events: dict[str, deque] = {t: deque() for t in self.topics}
        self._start = time.monotonic()
        self._offset_s: float | None = None

    def set_wall_offset(self, offset_s: float) -> None:
        """Publisher-vs-subscriber wall offset: pub_wall = sub_wall + offset."""
        with self._lock:
            self._offset_s = float(offset_s)

    @property
    def synced(self) -> bool:
        with self._lock:
            return self._offset_s is not None

    def hit(self, topic: str, t_wall: float | None = None,
            t_recv: float | None = None, bytes_n: int | None = None) -> None:
        """Record one received payload."""
        now = time.monotonic()
        with self._lock:
            offset = self._offset_s
            latency_ms = None
            if t_wall is not None and offset is not None:
                recv = time.time() if t_recv is None else t_recv
                latency_ms = (recv - (t_wall - offset)) * 1000.0
            self._counts[topic] += 1
            self._events[topic].append((now, latency_ms, bytes_n))

    def snapshot(self) -> dict[str, tuple[int, float, float | None, float | None,
                                          float, bool]]:
        """(count, fps, p50_ms, p95_ms, bytes/s, warm) per topic.

        `warm` is False until a full window has elapsed, because a rate taken
        over a partial window reads low and would look like a struggling link.
        """
        now = time.monotonic()
        warm = (now - self._start) >= self.window_s
        cutoff = now - self.window_s
        out = {}
        with self._lock:
            for t in self.topics:
                dq = self._events[t]
                while dq and dq[0][0] < cutoff:
                    dq.popleft()
                fps = len(dq) / self.window_s
                lat = sorted(a for (_, a, _) in dq if a is not None)
                if len(lat) >= 2:
                    p50 = lat[len(lat) // 2]
                    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
                else:
                    p50 = p95 = None
                total = sum(b for (_, _, b) in dq if b is not None)
                out[t] = (self._counts[t], fps, p50, p95,
                          total / self.window_s, warm)
        return out


class ClockSync:
    """NTP-style offset estimator against the publisher's clock REP port.

    Best-effort by construction: every failure path returns None and leaves
    the stats unsynced, so a publisher built without the clock socket costs
    the latency columns and nothing else.
    """

    def __init__(self, host: str, port: int, stats: StreamStats):
        self.host, self.port = host, int(port)
        self.stats = stats
        self._samples: deque = deque(maxlen=SAMPLE_HISTORY)  # (offset_s, rtt_s)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def handshake(self, timeout_ms: int = 1000) -> tuple[float, float] | None:
        """One four-timestamp exchange; (offset_s, rtt_s) or None."""
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.connect(f"tcp://{self.host}:{self.port}")
        try:
            t1 = time.time()
            sock.send(b"")
            reply = sock.recv_pyobj()
            t4 = time.time()
            t2, t3 = reply["t_recv"], reply["t_send"]
            rtt = (t4 - t1) - (t3 - t2)
            # A round trip over a second is not a measurement, it is a stall;
            # applying its offset would poison every later latency reading.
            if rtt < 0 or rtt > 1.0:
                return None
            return ((t2 - t1) + (t3 - t4)) / 2.0, rtt
        except Exception:
            return None
        finally:
            sock.close(linger=0)

    def _apply_best(self, sample: tuple[float, float]) -> None:
        self._samples.append(sample)
        self.stats.set_wall_offset(min(self._samples, key=lambda s: s[1])[0])

    def initial_sync(self, attempts: int = 5) -> tuple[float, float] | None:
        """Retry until one handshake lands. Returns that sample, or None."""
        for _ in range(attempts):
            sample = self.handshake()
            if sample is not None:
                self._apply_best(sample)
                return sample
            time.sleep(0.2)
        return None

    def start(self, on_sync=None) -> None:
        """Sync, then refresh every 10 s, all on a daemon thread.

        The first handshake is in here rather than in the caller because it
        retries for several seconds against a publisher that has no clock
        socket, and the caller is usually something an operator is waiting on.
        `on_sync` is called once with the first sample, or None if none landed.
        """
        if self._thread is not None:
            return

        def worker() -> None:
            sample = self.initial_sync()
            if on_sync is not None:
                on_sync(sample)
            while not self._stop.wait(REFRESH_S):
                sample = self.handshake()
                if sample is not None:
                    self._apply_best(sample)

        self._thread = threading.Thread(target=worker, name="aria-clock",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
