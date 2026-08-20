#!/usr/bin/env python3
"""Measure the incoming Meta Quest controller packet rate."""

from __future__ import annotations

import argparse
import statistics
import time

import zmq


TOPIC = b"oculus_controller"


def measure(host: str, port: int, duration: float, startup_timeout: float) -> int:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
    socket.connect(f"tcp://{host}:{port}")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print(f"Waiting for Quest input on tcp://{host}:{port} ...")
    try:
        if not poller.poll(int(startup_timeout * 1000)):
            print(f"No Quest input received within {startup_timeout:g} seconds.")
            return 1

        # Start the measurement window at the first received packet so ZMQ's
        # subscription handshake does not lower the reported rate.
        socket.recv_multipart()
        timestamps = [time.monotonic()]
        deadline = timestamps[0] + duration

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if poller.poll(max(1, int(remaining * 1000))):
                socket.recv_multipart()
                timestamps.append(time.monotonic())

    except KeyboardInterrupt:
        print("\nMeasurement cancelled.")
        return 130
    finally:
        socket.close()
        context.term()

    count = len(timestamps)
    window_rate = count / duration
    print(f"\nReceived {count} packets in {duration:.3f} seconds")
    print(f"Window rate: {window_rate:.2f} Hz")

    if count >= 2:
        intervals = [b - a for a, b in zip(timestamps, timestamps[1:])]
        interval_rate = 1.0 / statistics.mean(intervals)
        jitter_ms = statistics.pstdev(intervals) * 1000.0
        print(f"Inter-packet rate: {interval_rate:.2f} Hz")
        print(f"Interval jitter: {jitter_ms:.2f} ms (standard deviation)")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="10.21.116.241",
        help="Quest headset IP address (default: %(default)s)",
    )
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="measurement duration in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for the first packet (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.duration <= 0 or args.startup_timeout <= 0:
        parser.error("--duration and --startup-timeout must be greater than zero")

    return measure(args.host, args.port, args.duration, args.startup_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
