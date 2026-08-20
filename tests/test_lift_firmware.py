"""
test_lift_firmware.py — compile the lift sketch, then run it on the host.

Two things happen here:

    arduino-cli compile   the sketch is still valid for the real ATmega328P,
                          and still fits in its flash and RAM
    tests/firmware/       the same source, compiled natively against the
                          stand-ins in tests/firmware/include and driven
                          through simulated time, a simulated Timer1 and
                          simulated limit switches

The second is the one that catches behaviour: the velocity ramp's
acceleration and jerk, the reversal that has to pass through zero before DIR
moves, the 300 ms command timeout, the limit switches, and the fact that the
discrete up / down / distance / home moves still work as they did.

Either tool may be missing on a development machine. That is reported as a
skip, not a failure — but the two are counted separately, so a run with
everything skipped cannot be mistaken for a run that passed.

    python tests/test_lift_firmware.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
SKETCH_DIR = _REPO / "firmware/lift_controller"
HARNESS = _REPO / "tests/firmware/lift_harness.cpp"
SHIM_INCLUDE = _REPO / "tests/firmware/include"

FQBN = "arduino:avr:uno"

RESULTS: list[tuple[str, bool, str]] = []
SKIPPED: list[str] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def skip(name, reason):
    SKIPPED.append(f"{name} ({reason})")
    print(f"  SKIP  {name}  [{reason}]")


def test_arduino_compile() -> None:
    print("\narduino-cli: the sketch still builds for the board")
    if shutil.which("arduino-cli") is None:
        skip("sketch compiles for " + FQBN, "arduino-cli not installed")
        return

    result = subprocess.run(
        ["arduino-cli", "compile", "--fqbn", FQBN, str(SKETCH_DIR)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0 and "platform not installed" in (
            result.stderr + result.stdout).lower():
        skip("sketch compiles for " + FQBN, "arduino:avr core not installed")
        return

    check(f"sketch compiles for {FQBN}", result.returncode == 0,
          (result.stderr or result.stdout).strip().splitlines()[-1] if result.returncode else "")
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return

    # The board has 32 KB of flash and 2 KB of RAM, and the sketch allocates
    # Strings at runtime. Headroom is part of it building, not a nicety.
    output = result.stdout + result.stderr
    flash = _percentage(output, "program storage space")
    ram = _percentage(output, "dynamic memory")
    check("flash use leaves room to grow", flash is not None and flash < 80,
          f"{flash}% of 32 KB" if flash is not None else "unparsed")
    check("RAM use leaves room for locals and Strings",
          ram is not None and ram < 75, f"{ram}% of 2 KB" if ram is not None else "unparsed")


def _percentage(output: str, marker: str):
    for line in output.splitlines():
        if marker in line and "%" in line:
            for token in line.split():
                if token.startswith("(") and token.endswith("%)"):
                    try:
                        return int(token[1:-2])
                    except ValueError:
                        return None
    return None


def test_firmware_behaviour() -> None:
    print("\nhost harness: what the firmware actually does")
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        skip("firmware behaviour harness", "no C++ compiler found")
        return

    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "lift_harness"
        build = subprocess.run(
            [compiler, "-std=c++17", "-O1", "-o", str(binary),
             "-I", str(SHIM_INCLUDE), "-I", str(SKETCH_DIR), str(HARNESS)],
            capture_output=True, text=True, timeout=600,
        )
        check("harness compiles", build.returncode == 0,
              build.stderr.strip().splitlines()[-1] if build.returncode else "")
        if build.returncode != 0:
            print(build.stderr)
            return

        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=600)

    for line in run.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(("PASS", "FAIL")):
            ok = stripped.startswith("PASS")
            RESULTS.append((stripped[4:].strip(), ok, ""))
        print(line if line.startswith((" ", "\t")) else "  " + line)

    if run.returncode != 0 and not any(not ok for _n, ok, _d in RESULTS):
        check("harness exited cleanly", False, f"exit {run.returncode}")


def main() -> int:
    test_arduino_compile()
    test_firmware_behaviour()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed"
          + (f", {len(SKIPPED)} skipped" if SKIPPED else ""))
    for name in SKIPPED:
        print(f"  skipped: {name}")
    failures = [name for name, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
