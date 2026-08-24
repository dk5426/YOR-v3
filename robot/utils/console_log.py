"""console_log.py — mirror all console output to a timestamped file.

Console output from a live run (arm init, wholebody loop-timing/command-jitter
diagnostics, errors) only ever exists in whatever terminal it was run in.
Calling `start_console_log` once at process startup makes it durable without
needing `| tee` on every invocation.
"""

import sys
import threading
import time
from pathlib import Path


class _Tee:
    """Duplicates writes to N underlying streams.

    yor.py runs several threads (wholebody solve, arm dispatch, base
    controller + relay, lift/SLAM listeners) that all print concurrently.
    Without a lock, two threads' `write()` calls can interleave mid-call --
    each individual `print()` is itself two writes (the text, then the
    newline), so an unlocked Tee can end up splicing one thread's text
    between another's, including mid-multi-byte-UTF-8-character (this repo's
    own log lines use "—"/"…"), which turned a real log binary-looking
    enough that `file`/plain `grep` refused to treat it as text. The lock
    doesn't make whole print() statements atomic against each other (that
    would need patching stdout itself), but it does make each individual
    write() atomic, which is what was actually corrupting bytes.
    """

    def __init__(self, *streams, lock: threading.Lock):
        self._streams = streams
        # Shared across the stdout and stderr Tees, not one each -- both
        # ultimately write the same underlying log file, so a stdout write
        # and a stderr write need to serialize against each other too.
        self._lock = lock

    def write(self, data) -> int:
        with self._lock:
            for s in self._streams:
                s.write(data)
        return len(data)

    def flush(self) -> None:
        with self._lock:
            for s in self._streams:
                s.flush()

    def isatty(self) -> bool:
        return False


def start_console_log(name: str, log_dir: Path) -> Path:
    """Mirror stdout/stderr to a new timestamped file under `log_dir`.

    Returns the log's path. Safe to call at most once per process -- calling
    it again would nest Tees and duplicate every line once per prior call.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{name}_{stamp}.log"
    log_file = path.open("w", buffering=1, encoding="utf-8")
    lock = threading.Lock()
    sys.stdout = _Tee(sys.__stdout__, log_file, lock=lock)
    sys.stderr = _Tee(sys.__stderr__, log_file, lock=lock)
    print(f"[console-log] mirroring output to {path}")
    return path
