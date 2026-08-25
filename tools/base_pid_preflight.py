#!/usr/bin/env python3
"""
base_pid_preflight.py — apply and verify the commissioned swerve PID gains.

There are two ways in, and both end up writing the same manifest:

    robot/yor.py init()          calls sync_from_manifest() against the
                                 SparkFlex objects robot/base_motor.py has
                                 already opened, before the base control loop
                                 starts. Nothing extra is opened, so the rule
                                 below is not broken. Controllers that already
                                 hold the commissioned values are left alone.

    python tools/base_pid_preflight.py    the standalone command, for when the
                                          robot process is not running:
                                            --dry-run      validate only, no CAN
                                            --verify-only  read back, never write

The standalone command opens its own devices, so it must run BEFORE
robot/yor.py (or anything else that opens the swerve controllers), and it
refuses to start if one of them is already running: the SPARK controllers are
owned by exactly one process at a time, and a second set of SparkFlex objects
on the same CAN bus while the base control loop is running is a way to corrupt
both.

The values live in the config/base_pid_*.json manifests, not here. What this
file owns is the order of operations, which for the standalone command is:

    1. read the manifest and reject anything out of range
    2. cross-check the module CAN IDs against robot/base_motor.py
    3. check the CAN interface is up
    4. refuse to run if another process already owns the controllers
    5. only then open the devices and write
    6. read every written field back and fail on any difference

Gains are written to controller RAM, which a power cycle clears — which is why
robot/yor.py re-checks them on every start. A controller that quietly reverted
to its stock gains steers and drives differently from its three neighbours,
which is much harder to diagnose from the robot's behaviour than a failed
preflight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Three manifests, same schema, same validation, same readback; only the
# numbers and the intent differ.
#
# STOCK is what the controllers hold in flash and revert to on a power cycle,
# and what robot/yor.py writes back on shutdown -- see
# YOR._restore_base_pid_gains. Its drive loop is P-only and reaches about half
# its setpoint, which its drive_command_scale of 2.0 compensates for.
#
# COMMISSIONED is the tuned, feed-forward-dominated set measured on the floor
# on 2026-08-17: drive tracks its setpoint, steering runs Kp=20/Kd=6 behind a
# deliberate +/-0.25 output clamp. Opt-in.
#
# HYBRID is the commissioned drive loop with the stock full-range steering
# loop, floor-validated on 2026-08-24. It is the default: the drive tracking
# fix is the single biggest responsiveness win, and it takes none of the
# steering-clamp risk (see finding 9 in docs/BASE_COMMAND_LOOP_REVIEW.md).
# Each manifest carries its own drive_command_scale, so switching sets cannot
# reintroduce the finding-6 speed mismatch.
STOCK_MANIFEST = _REPO / "config/base_pid_stock.json"
COMMISSIONED_MANIFEST = _REPO / "config/base_pid_commissioned.json"
HYBRID_MANIFEST = _REPO / "config/base_pid_hybrid.json"

# What gets applied when nothing says otherwise. Shutdown still restores
# STOCK_MANIFEST -- the default only decides what startup syncs to.
DEFAULT_MANIFEST = HYBRID_MANIFEST

# Every field the preflight writes, as (manifest key, setter, getter). Readback
# covers exactly this list, so a field can never be written without being
# checked.
PID_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("p", "SetP", "GetP"),
    ("i", "SetI", "GetI"),
    ("d", "SetD", "GetD"),
    ("velocity_ff", "SetVelocityFF", "GetVelocityFF"),
    ("output_min", "SetOutputMin", "GetOutputMin"),
    ("output_max", "SetOutputMax", "GetOutputMax"),
)

# Sanity envelope for the manifest itself. These are not tuning opinions; they
# are the range outside which a value is certainly a typo or a unit error.
LIMITS = {
    "p": (0.0, 100.0),
    "i": (0.0, 10.0),
    "d": (0.0, 100.0),
    "velocity_ff": (-10.0, 10.0),
    "output_min": (-1.0, 0.0),
    "output_max": (0.0, 1.0),
}

# Processes that own the swerve controllers. Matched against /proc cmdlines.
CAN_OWNING_PROCESSES = (
    "robot/yor.py",
    "robot/base.py",
    "robot/base_motor.py",
    "robot/teleop/joystick.py",
    "robot/get_base_telemetry.py",
)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeviceSpec:
    """One controller and the values it should end up holding."""

    module: str
    role: str
    can_id: int
    values: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.module} {self.role} (CAN {self.can_id})"


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def validate_manifest(manifest: dict) -> list[str]:
    """Everything wrong with the manifest, as human-readable strings."""
    errors: list[str] = []

    if not isinstance(manifest.get("can_interface"), str):
        errors.append("can_interface must be a string")

    slot = manifest.get("pid_slot")
    if not isinstance(slot, int) or isinstance(slot, bool) or not (0 <= slot <= 3):
        errors.append(f"pid_slot must be an integer 0-3, got {slot!r}")

    # Optional, but if present it has to be sane: it multiplies every wheel
    # velocity command, so a typo here drives the robot at the wrong speed and
    # corrupts the odometry with it.
    scale = manifest.get("drive_command_scale")
    if scale is not None:
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            errors.append(f"drive_command_scale must be a number, got {scale!r}")
        elif not math.isfinite(scale) or not (0.1 <= scale <= 10.0):
            errors.append(f"drive_command_scale={scale} is outside [0.1, 10.0]")

    roles = manifest.get("roles")
    if not isinstance(roles, dict) or not roles:
        errors.append("roles must be a non-empty object")
        return errors

    for role_name, role in roles.items():
        if not isinstance(role, dict):
            errors.append(f"role {role_name} must be an object")
            continue
        for key, _setter, _getter in PID_FIELDS:
            value = role.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{role_name}.{key} must be a number, got {value!r}")
                continue
            if not math.isfinite(value):
                errors.append(f"{role_name}.{key} must be finite, got {value!r}")
                continue
            low, high = LIMITS[key]
            if not (low <= value <= high):
                errors.append(f"{role_name}.{key}={value} is outside [{low}, {high}]")
        lo, hi = role.get("output_min"), role.get("output_max")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo >= hi:
            errors.append(f"{role_name}: output_min {lo} must be below output_max {hi}")

    modules = manifest.get("modules")
    if not isinstance(modules, dict) or not modules:
        errors.append("modules must be a non-empty object")
        return errors

    seen: dict[int, str] = {}
    for module_name, module in modules.items():
        if not isinstance(module, dict):
            errors.append(f"module {module_name} must be an object")
            continue
        for role_name, can_id in module.items():
            if role_name not in roles:
                errors.append(f"module {module_name} names unknown role {role_name!r}")
            if not isinstance(can_id, int) or isinstance(can_id, bool) or not (1 <= can_id <= 62):
                errors.append(f"{module_name}.{role_name} CAN id must be 1-62, got {can_id!r}")
                continue
            if can_id in seen:
                errors.append(f"CAN id {can_id} used by both {seen[can_id]} and "
                              f"{module_name}.{role_name}")
            seen[can_id] = f"{module_name}.{role_name}"

    return errors


def drive_command_scale(manifest_path: Path, default: float) -> tuple[float, str]:
    """The command scale a manifest declares, with a line explaining the choice.

    Bound to the manifest rather than left a module constant because it is only
    correct for one set of drive gains: the stock P-only loop needs 2.0 to
    reach the commanded speed, the feed-forward-dominated commissioned loop
    needs about 1.0, and applying either to the other is a 2x speed error that
    the odometry silently inherits. Selecting a manifest now selects both.
    """
    try:
        manifest = load_manifest(Path(manifest_path))
    except Exception as exc:
        return default, f"{default} (manifest unreadable: {exc}; using the built-in default)"
    scale = manifest.get("drive_command_scale")
    if scale is None:
        return default, f"{default} (manifest declares none; using the built-in default)"
    return float(scale), f"{float(scale)} (from {Path(manifest_path).name})"


def plan_writes(manifest: dict) -> list[DeviceSpec]:
    """The manifest flattened into one entry per controller."""
    roles = manifest["roles"]
    specs: list[DeviceSpec] = []
    for module_name, module in manifest["modules"].items():
        for role_name, can_id in module.items():
            role = roles[role_name]
            specs.append(DeviceSpec(
                module=module_name,
                role=role_name,
                can_id=int(can_id),
                values={key: float(role[key]) for key, _s, _g in PID_FIELDS},
            ))
    return sorted(specs, key=lambda s: s.can_id)


def check_can_ids_against_base(manifest: dict) -> list[str]:
    """The manifest and robot/base_motor.py must name the same controllers.

    base_motor.py is the file the running robot actually uses; a manifest that
    disagrees with it would apply gains to the wrong module, or to nothing.
    """
    try:
        from robot.base_motor import CAN_IDS_DRIVE, CAN_IDS_ROT, MODULE_ORDER
    except Exception as exc:
        return [f"could not import robot/base_motor.py to cross-check CAN ids: {exc}"]

    expected = {
        module: {"drive": int(CAN_IDS_DRIVE[i]), "steering": int(CAN_IDS_ROT[i])}
        for i, module in enumerate(MODULE_ORDER)
    }
    problems: list[str] = []
    modules = manifest.get("modules", {})

    for module, roles in expected.items():
        if module not in modules:
            problems.append(f"manifest is missing module {module}")
            continue
        for role, can_id in roles.items():
            actual = modules[module].get(role)
            if actual != can_id:
                problems.append(
                    f"{module}.{role}: manifest says CAN {actual}, base_motor.py says {can_id}")

    for module in modules:
        if module not in expected:
            problems.append(f"manifest names module {module}, which base_motor.py does not")

    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

def check_can_interface(name: str) -> tuple[bool, str]:
    """(usable, detail) for a SocketCAN interface."""
    sysfs = Path("/sys/class/net") / name
    if not sysfs.exists():
        return False, f"{name} does not exist"
    try:
        state = (sysfs / "operstate").read_text().strip()
    except OSError as exc:
        return False, f"{name}: cannot read operstate ({exc})"
    if state in ("up", "unknown"):
        # SocketCAN reports "unknown" when the link is up but the driver does
        # not track carrier, so it is accepted; "down" never is.
        return True, f"{name} operstate={state}"
    return False, f"{name} operstate={state} — bring it up with `ip link set {name} up`"


def find_conflicting_processes() -> list[str]:
    """Running processes that already own the swerve controllers."""
    found: list[str] = []
    self_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", "replace").strip()
        except OSError:
            continue
        if not cmdline:
            continue
        for marker in CAN_OWNING_PROCESSES:
            if marker in cmdline:
                found.append(f"pid {pid}: {cmdline}")
                break
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Apply / verify
# ─────────────────────────────────────────────────────────────────────────────

def _close_enough(expected: float, actual: float, tolerance: float) -> bool:
    if actual is None or not math.isfinite(actual):
        return False
    return abs(expected - actual) <= max(tolerance, tolerance * abs(expected))


def apply_to_device(device, spec: DeviceSpec, slot: int) -> None:
    """Write every manifest field into the controller's RAM."""
    for key, setter, _getter in PID_FIELDS:
        getattr(device, setter)(slot, spec.values[key])


def read_back(device, spec: DeviceSpec, slot: int) -> dict[str, Optional[float]]:
    readings: dict[str, Optional[float]] = {}
    for key, _setter, getter in PID_FIELDS:
        try:
            readings[key] = float(getattr(device, getter)(slot))
        except Exception:
            readings[key] = None
    return readings


def compare(spec: DeviceSpec, readings: dict[str, Optional[float]],
            tolerance: float) -> list[str]:
    return [
        f"{spec.label}: {key} expected {spec.values[key]}, controller reports {readings.get(key)}"
        for key, _s, _g in PID_FIELDS
        if not _close_enough(spec.values[key], readings.get(key), tolerance)
    ]


def run_devices(specs: list[DeviceSpec], interface: str, slot: int, *,
                write: bool, tolerance: float,
                device_factory: Callable[[str, int], object]) -> list[str]:
    """Open each controller in turn, optionally write, always read back."""
    problems: list[str] = []
    for spec in specs:
        try:
            device = device_factory(interface, spec.can_id)
        except Exception as exc:
            problems.append(f"{spec.label}: could not open device ({exc})")
            continue

        try:
            if write:
                apply_to_device(device, spec, slot)
            readings = read_back(device, spec, slot)
        except Exception as exc:
            problems.append(f"{spec.label}: {'write' if write else 'read'} failed ({exc})")
            continue

        mismatches = compare(spec, readings, tolerance)
        problems.extend(mismatches)
        status = "OK" if not mismatches else "MISMATCH"
        print(f"  {status:9s} {spec.label}: " + ", ".join(
            f"{key}={readings.get(key)}" for key, _s, _g in PID_FIELDS))

    return problems


def _spark_factory(interface: str, can_id: int):
    from sparkcan_py import SparkFlex
    return SparkFlex(interface, can_id)


# ─────────────────────────────────────────────────────────────────────────────
# In-process sync (robot/yor.py)
# ─────────────────────────────────────────────────────────────────────────────
#
# robot/yor.py calls sync_from_manifest() during init(), against the SparkFlex
# objects robot/base_motor.py has already opened for the swerve modules. That
# is the only way to bring the gains up from inside the robot process without
# breaking the rule the standalone command exists to enforce: exactly one set
# of SparkFlex objects per bus. Nothing here opens a device — the handles are
# passed in, and the caller still owns them.
#
# Every controller is read first and left alone if it already holds the
# commissioned values, so a restart that did not power-cycle the SPARKs writes
# nothing at all. A controller that does not answer a parameter read reads back
# as 0.0 (that is what the binding returns on timeout), and neither role in the
# manifest is all zeros, so silence can never be mistaken for "already set" —
# it costs a redundant write, which is the safe direction to be wrong in.

# A parameter read is one request/response with a 20 ms timeout while the
# SPARKs are streaming periodic status frames, so an occasional dropped answer
# is normal. Reads are retried rather than believed first time.
DEFAULT_READ_ATTEMPTS = 3


@dataclass
class SyncResult:
    """What the sync did to one controller."""

    spec: DeviceSpec
    status: str                                        # already-set | written | failed
    readings: dict[str, Optional[float]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "failed"


def _read_until_match(device, spec: DeviceSpec, slot: int, tolerance: float,
                      attempts: int) -> tuple[dict[str, Optional[float]], list[str]]:
    """Read the controller back, retrying while it disagrees with the manifest.

    A dropped answer looks exactly like a wrong gain, so a single disagreeing
    read is not enough to conclude anything. Agreement, on the other hand, is
    conclusive on the first try.
    """
    readings: dict[str, Optional[float]] = {}
    mismatches: list[str] = []
    for _ in range(max(1, attempts)):
        readings = read_back(device, spec, slot)
        mismatches = compare(spec, readings, tolerance)
        if not mismatches:
            break
    return readings, mismatches


def sync_device(device, spec: DeviceSpec, slot: int, *, tolerance: float = 1e-3,
                attempts: int = DEFAULT_READ_ATTEMPTS) -> SyncResult:
    """Bring one already-open controller to the manifest values.

    Reads first and writes only on a difference, then reads every written field
    back — a write that the controller ignored is a failure, not a success.
    """
    readings, mismatches = _read_until_match(device, spec, slot, tolerance, attempts)
    if not mismatches:
        return SyncResult(spec, "already-set", readings, [])

    try:
        apply_to_device(device, spec, slot)
    except Exception as exc:
        return SyncResult(spec, "failed", readings, [f"{spec.label}: write failed ({exc})"])

    readings, mismatches = _read_until_match(device, spec, slot, tolerance, attempts)
    if mismatches:
        return SyncResult(spec, "failed", readings, mismatches)
    return SyncResult(spec, "written", readings, [])


def sync_open_devices(devices: dict[int, object], manifest: dict, *,
                      tolerance: float = 1e-3, attempts: int = DEFAULT_READ_ATTEMPTS,
                      log: Optional[Callable[[str], None]] = print) -> list[SyncResult]:
    """Sync every controller the manifest names, using open device handles.

    `devices` maps CAN id to an object carrying the SPARK Get*/Set* methods —
    robot/base_motor.py's Base.swerve_devices() returns exactly that shape.
    """
    slot = int(manifest["pid_slot"])
    results: list[SyncResult] = []
    for spec in plan_writes(manifest):
        device = devices.get(spec.can_id)
        if device is None:
            results.append(SyncResult(
                spec, "failed", {}, [f"{spec.label}: no open device for CAN id {spec.can_id}"]))
            if log:
                log(f"  {'failed':11s} {spec.label}: not among the open controllers")
            continue

        result = sync_device(device, spec, slot, tolerance=tolerance, attempts=attempts)
        results.append(result)
        if log:
            log(f"  {result.status:11s} {spec.label}: " + ", ".join(
                f"{key}={result.readings.get(key)}" for key, _s, _g in PID_FIELDS))
    return results


def sync_from_manifest(devices: dict[int, object], *,
                       manifest_path: Path = DEFAULT_MANIFEST, tolerance: float = 1e-3,
                       attempts: int = DEFAULT_READ_ATTEMPTS,
                       log: Optional[Callable[[str], None]] = print) -> tuple[bool, list[str]]:
    """Validate the manifest, then bring the open controllers up to it.

    Returns (ok, problems). The manifest checks run first and on their own: a
    manifest that disagrees with robot/base_motor.py names the wrong
    controllers, and writing it would tune the wrong module.
    """
    def say(message: str) -> None:
        if log:
            log(message)

    try:
        manifest = load_manifest(Path(manifest_path))
    except Exception as exc:
        return False, [f"cannot read manifest {manifest_path} ({exc})"]

    errors = validate_manifest(manifest)
    if errors:
        return False, errors

    id_problems = check_can_ids_against_base(manifest)
    if id_problems:
        return False, id_problems

    say(f"{manifest_path}, PID slot {manifest['pid_slot']}")
    results = sync_open_devices(devices, manifest, tolerance=tolerance,
                                attempts=attempts, log=log)

    problems = [problem for result in results for problem in result.problems]
    already = sum(1 for result in results if result.status == "already-set")
    written = sum(1 for result in results if result.status == "written")
    say(f"{already} controller(s) already commissioned, {written} written, "
        f"{len(problems)} problem(s)")
    return not problems, problems


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="PID manifest to apply (default: config/base_pid_stock.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the manifest and the environment; touch no CAN device")
    parser.add_argument("--verify-only", action="store_true",
                        help="read the controllers back and compare; write nothing")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="readback tolerance (default 1e-3)")
    args = parser.parse_args(argv)

    if args.dry_run and args.verify_only:
        print("[preflight] --dry-run and --verify-only are mutually exclusive")
        return 2

    print(f"[preflight] manifest {args.manifest}")
    try:
        manifest = load_manifest(args.manifest)
    except Exception as exc:
        print(f"[preflight] FAIL: cannot read manifest ({exc})")
        return 1

    # ── 1. the manifest itself ──────────────────────────────────────────────
    errors = validate_manifest(manifest)
    if errors:
        print("[preflight] FAIL: manifest is not valid")
        for error in errors:
            print(f"    {error}")
        return 1
    print("[preflight] manifest valid")

    # ── 2. the CAN ids the robot actually uses ──────────────────────────────
    id_problems = check_can_ids_against_base(manifest)
    if id_problems:
        print("[preflight] FAIL: manifest disagrees with robot/base_motor.py")
        for problem in id_problems:
            print(f"    {problem}")
        return 1
    print("[preflight] module CAN ids match robot/base_motor.py")

    interface = manifest["can_interface"]
    slot = int(manifest["pid_slot"])
    specs = plan_writes(manifest)
    print(f"[preflight] interface {interface}, PID slot {slot}, "
          f"{len(specs)} controllers")
    for spec in specs:
        print("    " + spec.label + ": " + ", ".join(
            f"{key}={spec.values[key]}" for key, _s, _g in PID_FIELDS))

    # ── 3. the CAN interface ────────────────────────────────────────────────
    can_ok, can_detail = check_can_interface(interface)
    print(f"[preflight] {'OK  ' if can_ok else 'FAIL'} CAN interface: {can_detail}")
    if not can_ok and not args.dry_run:
        return 1

    # ── 4. nobody else owns the controllers ─────────────────────────────────
    conflicts = find_conflicting_processes()
    if conflicts:
        print("[preflight] FAIL: another process already owns the swerve controllers.")
        print("            Stop it first — a second set of SparkFlex objects on the")
        print("            same bus is not safe.")
        for conflict in conflicts:
            print(f"    {conflict}")
        return 1
    print("[preflight] no conflicting process holds the CAN devices")

    if args.dry_run:
        print("[preflight] dry run: nothing was written, no device was opened")
        return 0

    # ── 5 & 6. write RAM, then read every field back ────────────────────────
    write = not args.verify_only
    print(f"[preflight] {'applying gains to controller RAM' if write else 'reading back only'}")
    problems = run_devices(specs, interface, slot, write=write,
                           tolerance=args.tolerance, device_factory=_spark_factory)

    if problems:
        print(f"[preflight] FAIL: {len(problems)} problem(s)")
        for problem in problems:
            print(f"    {problem}")
        return 1

    print("[preflight] PASS: every controller reports the commissioned values")
    print("[preflight] these live in RAM — re-run after any controller power cycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
