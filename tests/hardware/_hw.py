"""_hw.py — shared scaffolding for the ON-ROBOT test suite.

Everything in tests/hardware/ drives real actuators. This module exists to make
that survivable:

* **One owner of the hardware.** Tests talk to a running `robot/yor.py` over
  commlink RPC rather than importing it, so they never fight the node for the
  CAN bus or the lift serial port.
* **Nothing moves without consent.** `confirm()` requires the operator to type
  a specific word. Enter, "y", or a stray keypress all abort.
* **Ctrl-C stops the robot.** `guard()` installs a SIGINT handler and an exit
  path that halt the base and lift (and optionally e-stop the arms) before the
  process dies. A test that raises halts the robot on the way out.
* **RPC cannot hang the test.** The REQ socket carries a receive timeout, and a
  timed-out socket is rebuilt rather than reused — a timed-out REQ is stuck in
  "expecting reply" and every later call on it fails with EFSM.

See README.md in this directory before running anything.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import time
from pathlib import Path

import zmq
from commlink import RPCClient

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_HOST = "192.168.1.10"   # the Pi, where robot/yor.py runs
DEFAULT_PORT = 5557


class Abort(Exception):
    """Operator declined a prompt, or a precondition was not met."""


class HwError(Exception):
    """The robot did not answer."""


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition, detail: str = "") -> bool:
    ok = bool(condition)
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return ok


def info(msg: str) -> None:
    print(f"  ....  {msg}")


def run(*tests) -> int:
    """Run each test, tally, return a shell exit code."""
    aborted = False
    for test in tests:
        if aborted:
            break
        try:
            test()
        except Abort as exc:
            print(f"\n  ABORTED: {exc}")
            aborted = True
        except Exception as exc:
            import traceback

            check(f"{test.__name__} raised", False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    failures = [name for name, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    if aborted:
        print("run was aborted before the end — treat the result as incomplete")
        return 2
    return 0 if not failures else 1


# ─────────────────────────────────────────────────────────────────────────────
# RPC client
# ─────────────────────────────────────────────────────────────────────────────

class HwClient:
    """RPC client that times out instead of hanging, and heals a wedged socket."""

    def __init__(self, host: str, port: int, timeout_s: float = 2.0):
        self.host = host
        self.port = port
        self.timeout_s = float(timeout_s)
        self._c = self._make()

    def _make(self) -> RPCClient:
        c = RPCClient(self.host, self.port)
        ms = max(1, int(self.timeout_s * 1000))
        # `socket` lives in the client's __dict__, so this is a local read —
        # RPCClient.__getattr__ (which would issue a remote call) is not hit.
        c.socket.setsockopt(zmq.RCVTIMEO, ms)
        c.socket.setsockopt(zmq.SNDTIMEO, ms)
        c.socket.setsockopt(zmq.LINGER, 0)
        return c

    def _rebuild(self) -> None:
        with contextlib.suppress(Exception):
            self._c.socket.close(linger=0)
        with contextlib.suppress(Exception):
            self._c.context.term()
        self._c = self._make()

    def call(self, method: str, *args, **kwargs):
        """Call a method on the robot. One rebuild-and-retry on socket failure."""
        last: Exception | None = None
        for attempt in (0, 1):
            try:
                return getattr(self._c, method)(*args, **kwargs)
            except zmq.ZMQError as exc:      # covers zmq.Again
                last = exc
                self._rebuild()
            except Exception:
                raise                        # a real remote exception: surface it
        raise HwError(
            f"{method}() got no reply in {self.timeout_s:.1f}s (twice). "
            f"Is robot/yor.py running on {self.host}:{self.port}?"
        ) from last

    def try_call(self, method: str, *args, **kwargs):
        """Like call(), but returns None instead of raising. For halt paths."""
        try:
            return self.call(method, *args, **kwargs)
        except Exception:
            return None

    # -- the halt paths, deliberately as simple as possible ------------------
    def halt(self) -> None:
        """Stop base and lift. Safe to call at any time, never raises."""
        self.try_call("set_base_velocity", [0.0, 0.0, 0.0])
        self.try_call("lift_stop")

    def estop(self) -> None:
        """Halt, then stop whole-body control and freeze the arms where they are."""
        self.halt()
        self.try_call("emergency_stop")


# ─────────────────────────────────────────────────────────────────────────────
# Pub/sub probing
# ─────────────────────────────────────────────────────────────────────────────

def port_open(host: str, port: int, timeout_s: float = 2.0) -> bool:
    """Is anything listening? Cheap, and never blocks longer than the timeout."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


class TopicListener:
    """Background poller for one commlink topic, with non-blocking reads.

    commlink's subscriber is pull-mode and a read blocks *indefinitely* when
    nothing is publishing — which would hang a test with no way out. So the
    blocking read lives on a daemon thread and the test reads a cache.

    Construction fails fast (Abort) when nothing is listening on the port, so
    the common "you forgot to start odin_pub_node" case gives a clear message
    instead of a silent stall.
    """

    def __init__(self, host: str, port: int, topic: str, connect_timeout_s: float = 2.0):
        import threading

        self.host, self.port, self.topic = host, port, topic
        if not port_open(host, port, connect_timeout_s):
            raise Abort(
                f"nothing is listening on {host}:{port} — start the publisher "
                f"(`python -m robot.odin_pub_node`) or pass the right --slam-host"
            )

        from commlink import Subscriber

        self._sub = Subscriber(host=host, port=port, topics=[topic])
        self._lock = threading.Lock()
        self._msg = None
        self._stamp = 0.0
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                msg = self._sub[self.topic]
            except Exception:
                msg = None
                time.sleep(0.1)
            if msg is not None:
                with self._lock:
                    self._msg = msg
                    self._stamp = time.monotonic()

    def latest(self, max_age_s: float = 1.0):
        """Newest message, or None if there is none or it is stale."""
        with self._lock:
            if self._msg is None or (time.monotonic() - self._stamp) > max_age_s:
                return None
            return self._msg

    def wait_for(self, timeout_s: float = 10.0):
        """Block up to timeout_s for the first message. Returns it, or None."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            msg = self.latest(max_age_s=timeout_s)
            if msg is not None:
                return msg
            time.sleep(0.05)
        return None

    def stop(self) -> None:
        self._stop_evt.set()
        # The worker may be parked in a blocking read; it is a daemon thread, so
        # do not wait on it — just release the socket and move on.
        self._thread.join(timeout=0.2)
        with contextlib.suppress(Exception):
            self._sub.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Operator interaction
# ─────────────────────────────────────────────────────────────────────────────

_ASSUME_YES = False


def confirm(prompt: str, token: str = "GO") -> None:
    """Require the operator to type `token` exactly. Anything else aborts.

    Deliberately not a y/n prompt: a reflexive Enter must never start a motion.
    --yes does NOT bypass this; it only skips the informational pauses.
    """
    print()
    print(f"  !! {prompt}")
    try:
        answer = input(f"     type {token} to proceed (anything else aborts): ").strip()
    except (EOFError, KeyboardInterrupt):
        raise Abort("no operator at the keyboard")
    if answer != token:
        raise Abort(f"operator declined: {prompt}")


def precondition(*lines: str) -> None:
    """State what must be physically true, and make the operator confirm it."""
    print()
    print("  PRECONDITIONS:")
    for line in lines:
        print(f"    - {line}")
    confirm("Confirm ALL of the above are true right now.", token="READY")


def pause(msg: str) -> None:
    """Informational stop. Skipped by --yes."""
    if _ASSUME_YES:
        info(msg)
        return
    try:
        input(f"  ..   {msg} (Enter to continue) ")
    except (EOFError, KeyboardInterrupt):
        raise Abort("interrupted")


def countdown(seconds: int, what: str) -> None:
    print(f"  ..   {what} in ", end="", flush=True)
    for i in range(seconds, 0, -1):
        print(f"{i} ", end="", flush=True)
        time.sleep(1.0)
    print("- go")


def ask_float(prompt: str) -> float | None:
    """Ask the operator to measure something. Blank means 'skip this check'."""
    try:
        raw = input(f"  ??   {prompt} (blank to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        raise Abort("interrupted")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("     not a number — skipping")
        return None


def ask_yes_no(prompt: str) -> bool | None:
    """Ask the operator to observe something. Blank means 'skip this check'."""
    try:
        raw = input(f"  ??   {prompt} [y/n, blank to skip]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise Abort("interrupted")
    if raw.startswith("y"):
        return True
    if raw.startswith("n"):
        return False
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Safety guard
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def guard(client: HwClient, estop: bool = False):
    """Halt the robot on the way out — normal exit, exception or Ctrl-C.

    `estop=True` also stops whole-body control and freezes the arms; use it for
    any test where the arms can move.
    """
    stop = client.estop if estop else client.halt
    fired = {"done": False}

    def _fire(reason: str) -> None:
        if fired["done"]:
            return
        fired["done"] = True
        print(f"\n  >> HALTING ({reason})")
        stop()

    def _on_sigint(signum, frame):
        _fire("Ctrl-C")
        # Re-raise as KeyboardInterrupt so the stack still unwinds normally.
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGINT, _on_sigint)
    try:
        yield
        _fire("test finished")
    except BaseException:
        _fire("exception")
        raise
    finally:
        signal.signal(signal.SIGINT, previous)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point plumbing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(description: str, extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--host", default=DEFAULT_HOST,
                   help=f"robot/yor.py host (default {DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"robot/yor.py RPC port (default {DEFAULT_PORT})")
    p.add_argument("--timeout", type=float, default=2.0,
                   help="per-RPC timeout in seconds (default 2.0)")
    p.add_argument("--yes", action="store_true",
                   help="skip informational pauses. Does NOT skip safety confirmations.")
    if extra is not None:
        extra(p)
    args = p.parse_args()
    global _ASSUME_YES
    _ASSUME_YES = bool(args.yes)
    return args


def connect(args) -> HwClient:
    """Open the RPC link and prove the node is actually answering."""
    print(f"connecting to robot/yor.py at {args.host}:{args.port} ...")
    client = HwClient(args.host, args.port, args.timeout)
    try:
        state = client.call("get_state")
    except HwError as exc:
        raise Abort(str(exc))
    if not state:
        raise Abort(
            "the node answered but returned an empty state — it is probably "
            "not initialised. Check the robot/yor.py console for errors."
        )
    print(f"connected. solver={'ok' if state.get('solved') else 'not solving'}, "
          f"base_motion={'ON' if state.get('base_motion_enabled') else 'off'}")
    return client


def banner(title: str, *notes: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    for note in notes:
        print(f"  {note}")
    print("=" * 72)
