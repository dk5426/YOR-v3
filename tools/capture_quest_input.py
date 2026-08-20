#!/usr/bin/env python3
"""Capture the unmodified Meta Quest ZMQ controller stream to JSONL.

Each output line contains receive timestamps and every original ZMQ frame in
base64, so the wire payload can be reconstructed byte-for-byte. UTF-8 copies of
the topic and payload are included only for convenient inspection; no Oculus
parsing, coordinate conversion, filtering, or WBC logic is applied.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from pathlib import Path

import zmq


TOPIC = b"oculus_controller"


def make_record(sequence: int, frames: list[bytes]) -> dict:
    """Build one lossless, JSON-serializable record from a ZMQ multipart message."""
    record = {
        "sequence": sequence,
        "received_unix_ns": time.time_ns(),
        "received_monotonic_ns": time.monotonic_ns(),
        "frames_base64": [base64.b64encode(frame).decode("ascii") for frame in frames],
    }
    if frames:
        record["topic_utf8"] = frames[0].decode("utf-8", errors="replace")
    if len(frames) >= 2:
        record["payload_utf8"] = frames[1].decode("utf-8", errors="replace")
    return record


def capture(
    host: str,
    port: int,
    output: Path,
    duration: float,
    max_packets: int,
    startup_timeout: float,
    show: bool,
) -> int:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
    socket.connect(f"tcp://{host}:{port}")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Waiting for raw Quest input on tcp://{host}:{port} ...")
    print(f"Writing lossless JSONL to {output}")

    receive_times: list[float] = []
    started_at: float | None = None
    interrupted = False

    try:
        if not poller.poll(max(1, int(startup_timeout * 1000))):
            print(f"No Quest input received within {startup_timeout:g} seconds.")
            return 1

        # Line-buffering preserves completed packets if the program is stopped.
        with output.open("w", encoding="utf-8", buffering=1) as stream:
            while True:
                if started_at is not None and duration > 0:
                    remaining = duration - (time.monotonic() - started_at)
                    if remaining <= 0:
                        break
                    timeout_ms = max(1, min(200, int(remaining * 1000)))
                else:
                    timeout_ms = 200

                if not poller.poll(timeout_ms):
                    continue

                frames = socket.recv_multipart()
                received = time.monotonic()
                if started_at is None:
                    started_at = received

                record = make_record(len(receive_times), frames)
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                receive_times.append(received)

                if show:
                    payload = record.get("payload_utf8", "")
                    print(f"[{record['sequence']:06d}] {payload}")

                if max_packets > 0 and len(receive_times) >= max_packets:
                    break

    except KeyboardInterrupt:
        interrupted = True
        print("\nCapture stopped by operator.")
    finally:
        socket.close()
        context.term()

    count = len(receive_times)
    print(f"Captured {count} packet{'s' if count != 1 else ''} to {output}")
    if count >= 2:
        intervals = [b - a for a, b in zip(receive_times, receive_times[1:])]
        print(f"Input rate: {1.0 / statistics.mean(intervals):.2f} Hz")
        print(f"Interval jitter: {statistics.pstdev(intervals) * 1000.0:.2f} ms")

    return 130 if interrupted else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Quest headset IP address")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSONL path (default: artifacts/quest_input/quest_raw_<time>.jsonl)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="capture duration in seconds; 0 runs until Ctrl-C (default: %(default)s)",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=0,
        help="stop after this many packets; 0 means unlimited (default: %(default)s)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for the first packet (default: %(default)s)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also print each decoded payload while capturing",
    )
    args = parser.parse_args()

    if args.duration < 0 or args.max_packets < 0 or args.startup_timeout <= 0:
        parser.error(
            "--duration and --max-packets must be non-negative; "
            "--startup-timeout must be positive"
        )
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = Path("artifacts/quest_input") / f"quest_raw_{stamp}.jsonl"

    return capture(
        host=args.host,
        port=args.port,
        output=output,
        duration=args.duration,
        max_packets=args.max_packets,
        startup_timeout=args.startup_timeout,
        show=args.show,
    )


if __name__ == "__main__":
    raise SystemExit(main())
