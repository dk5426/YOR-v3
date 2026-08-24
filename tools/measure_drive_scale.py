#!/usr/bin/env python3
"""
measure_drive_scale.py — settle what DRIVE_VEL_SCALE is actually doing.

The command path multiplies every wheel velocity by a scale that is only
correct for one set of drive gains (`drive_command_scale` in
config/base_pid_*.json). Reading the controllers back on 2026-08-22 showed
velocity_cf = 0.000846326, i.e. they are already configured in true m/s to
within 1.7% -- so the historical 2.0 is not a unit conversion. It compensates
for the stock P-only loop, which six floor runs measured at 37-46% of setpoint.

This runs a straight line on the floor and answers both halves in one go:

  1. **Does the loop reach its setpoint?**  `GetVelocity` against what the
     controller was told. No tape measure needed; it is the controller's own
     view of the motor.
  2. **Does a commanded m/s equal a real m/s?**  The distance the chassis
     actually covered against the distance the command implies. This is the
     one that matters, because `BaseOdometry` integrates the *commanded*
     velocity -- if the two differ, the IK plans against a chassis pose that
     is wrong by exactly that ratio.

Only (2) settles the scale, and only a tape measure gives it. A wheels-up run
cannot: it removes the load the gains were commissioned under, and it has no
ground truth at all.

Procedure, three steps with a prompt between each:

    aim   -- the modules turn to straight-ahead at a crawl
    mark  -- you mark where the robot is
    drive -- a straight line at the requested speed, then a stop
    measure -- you tape the distance and type it in

    python tools/measure_drive_scale.py                  # 0.15 m/s for 5 s
    python tools/measure_drive_scale.py --velocity 0.10 --seconds 4
    python tools/measure_drive_scale.py --spin           # in place, no tape

Nothing here writes a controller parameter, and every exit path stops the base.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from robot.base_motor import Base, MODULE_ORDER                    # noqa: E402
from tools.base_pid_preflight import (                             # noqa: E402
    DEFAULT_MANIFEST, check_can_interface, compare, drive_command_scale,
    find_conflicting_processes, load_manifest, plan_writes, read_back,
    sync_from_manifest,
)

AIM_SPEED = 0.03        # m/s -- enough to steer the modules, barely enough to roll
AIM_S = 2.5
STOP_S = 1.0            # let the profiler and the wheels come to rest


def _drive(base, target, seconds, sample_from=0.0):
    """Hold `target` for `seconds`, returning (samples, elapsed)."""
    samples, t0 = [], time.monotonic()
    last = t0
    while True:
        now = time.monotonic()
        if now - t0 >= seconds:
            return samples, now - t0
        base.set_target_base_velocity(np.asarray(target, dtype=float), smooth=True)
        if now - t0 >= sample_from:
            tel = base.swerve_telemetry()
            samples.append((now - last, tel["drive_cmd_mps"].copy(),
                            tel["drive_meas_raw"].copy(),
                            tel["steer_cmd_rad"].copy(), tel["steer_meas_rad"].copy()))
        last = now
        time.sleep(0.02)


def _gains_ready(base, manifest_path, apply_gains) -> bool:
    """Refuse to measure unless the controllers hold the manifest's gains.

    `--manifest` selects a *pair*: the PID gains and the `drive_command_scale`
    that is correct for them. Reading only the scale, as this tool did until
    2026-08-24, produces a measurement of neither configuration -- the
    2026-08-24 commissioned run applied scale 1.068 on top of stock gains and
    looked like the commissioned loop underperforming badly, when the
    commissioned loop had never been loaded at all. The mismatch is invisible
    in the output, which is what makes it worth a hard stop rather than a
    warning.
    """
    devices = base.swerve_devices()
    try:
        manifest = load_manifest(Path(manifest_path))
        slot = int(manifest["pid_slot"])
        specs = plan_writes(manifest)
    except Exception as exc:
        print(f"[drive-scale] cannot read {manifest_path}: {exc}")
        return False

    if apply_gains:
        print(f"[drive-scale] flashing gains from {Path(manifest_path).name}")
        ok, problems = sync_from_manifest(devices, manifest_path=Path(manifest_path),
                                          log=lambda line: print(f"[drive-scale]   {line}"))
        if not ok:
            for problem in problems:
                print(f"[drive-scale]   {problem}")
            return False
        print("[drive-scale] NOTE: these gains live in controller RAM and outlive this "
              "process. Restore with:")
        print("[drive-scale]   python tools/base_pid_preflight.py "
              "--manifest config/base_pid_stock.json")
        return True

    mismatches = []
    for spec in specs:
        device = devices.get(spec.can_id)
        if device is None:
            mismatches.append(f"{spec.label}: no open device")
            continue
        mismatches.extend(compare(spec, read_back(device, spec, slot), 1e-3))

    if not mismatches:
        print(f"[drive-scale] controllers hold {Path(manifest_path).name}'s gains")
        return True

    print(f"[drive-scale] the controllers do NOT hold {Path(manifest_path).name}'s gains:")
    for line in mismatches[:6]:
        print(f"  {line}")
    if len(mismatches) > 6:
        print(f"  ... and {len(mismatches) - 6} more")
    print("\n[drive-scale] measuring now would apply that manifest's command scale to "
          "whatever gains are actually loaded, which describes no real configuration.")
    print("[drive-scale] either flash them first:")
    print(f"  python tools/base_pid_preflight.py --manifest {manifest_path}")
    print("[drive-scale] or re-run this with --apply-gains.")
    return False


def _ask_float(prompt):
    try:
        raw = input(prompt).strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("[drive-scale] not a number; skipping the distance analysis")
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--velocity", type=float, default=0.15,
                        help="chassis speed to hold, m/s (default 0.15)")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="how long to hold it (default 5)")
    parser.add_argument("--spin", action="store_true",
                        help="rotate in place instead of driving; skips the distance half")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help=f"manifest whose drive_command_scale is in force "
                             f"(default: {DEFAULT_MANIFEST.name})")
    parser.add_argument("--apply-gains", action="store_true",
                        help="flash the manifest's PID gains before measuring, instead of "
                             "refusing when the controllers do not already hold them")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args(argv)

    ok, detail = check_can_interface("can0")
    if not ok:
        print(f"[drive-scale] can0 unusable: {detail}")
        return 1
    conflicts = find_conflicting_processes()
    if conflicts:
        print("[drive-scale] another process owns the swerve controllers; stop it first:")
        for line in conflicts:
            print(f"  {line}")
        return 1

    scale, scale_note = drive_command_scale(args.manifest, 2.0)
    print(f"[drive-scale] drive_command_scale in force: {scale_note}")

    # Worst case is that the command is truthful and nothing is lost to the
    # loop, so budget the full commanded distance plus the ramps.
    reach = args.velocity * (args.seconds + AIM_S + STOP_S)
    if args.spin:
        print(f"[drive-scale] the robot will SPIN IN PLACE at {args.velocity:.2f} rad/s "
              f"for {args.seconds:.0f}s. Clear a turning circle.")
    else:
        print(f"[drive-scale] the robot will DRIVE FORWARD at {args.velocity:.2f} m/s "
              f"for {args.seconds:.0f}s.")
        print(f"[drive-scale] clear at least {reach + 1.0:.1f} m ahead of it "
              f"({reach:.1f} m of travel plus a margin).")
    print("[drive-scale] keep the e-stop or the power switch within reach.")
    if not args.yes and input("[drive-scale] type 'go' to continue: ").strip() != "go":
        print("[drive-scale] aborted")
        return 1

    target = (np.array([0.0, 0.0, args.velocity]) if args.spin
              else np.array([args.velocity, 0.0, 0.0]))
    aim = target / max(abs(args.velocity), 1e-9) * AIM_SPEED

    base = Base(drive_vel_scale=scale)

    if not _gains_ready(base, args.manifest, args.apply_gains):
        return 1

    samples, elapsed = [], 0.0
    base.start_control()          # without this nothing is ever sent to a wheel
    try:
        # 1. Aim. The modules turn to the commanded heading at a crawl; with
        #    USE_FEEDBACK_FOR_STEER on, cos_error_scaling holds the drive back
        #    while they are still turning, so the chassis barely moves.
        print(f"[drive-scale] aiming the modules ({AIM_S:.0f}s)...")
        _drive(base, aim, AIM_S)
        base.set_target_base_velocity(np.zeros(3), smooth=False)
        time.sleep(STOP_S)

        if not args.spin and not args.yes:
            input("[drive-scale] mark the robot's position on the floor, then press ENTER: ")

        print(f"[drive-scale] driving for {args.seconds:.0f}s...")
        samples, elapsed = _drive(base, target, args.seconds)
    finally:
        base.set_target_base_velocity(np.zeros(3), smooth=False)
        time.sleep(STOP_S)
        if base.control_loop_running:
            base.stop_control()
        try:
            base._pico_lift._shutdown()
        except Exception:
            pass

    if not samples:
        print("[drive-scale] no samples collected")
        return 1

    dts = np.array([s[0] for s in samples])
    commanded = np.array([s[1] for s in samples])
    measured = np.array([s[2] for s in samples])
    steer_cmd = np.array([s[3] for s in samples])
    steer_meas = np.array([s[4] for s in samples])
    setpoint = commanded * scale

    print(f"\n[drive-scale] {len(samples)} samples over {elapsed:.1f}s\n")
    print(f"{'module':8s} {'commanded':>11s} {'setpoint':>11s} {'measured':>11s} {'meas/setp':>10s}")
    print("-" * 56)
    tracking = []
    for i, name in enumerate(MODULE_ORDER):
        c, s_, m = (np.median(np.abs(commanded[:, i])), np.median(np.abs(setpoint[:, i])),
                    np.median(np.abs(measured[:, i])))
        t = m / s_ if s_ > 1e-9 else float("nan")
        tracking.append(t)
        print(f"{name:8s} {c:11.4f} {s_:11.4f} {m:11.4f} {t:10.3f}")
    track = float(np.nanmedian(tracking)) if np.any(np.isfinite(tracking)) else float("nan")
    print("-" * 56)
    print(f"{'median':8s} {'':11s} {'':11s} {'':11s} {track:10.3f}")

    if not np.isfinite(track):
        print("\n[drive-scale] no wheel was ever commanded above the noise floor. Either the "
              "velocity was too low to leave the deadband, or the base never armed.")
        return 1

    print(f"\n[drive-scale] the velocity loop reaches {track * 100:.0f}% of its setpoint.")
    if track > 0.8:
        print("[drive-scale]   -> it tracks; the gains are not the limitation.")
    else:
        print("[drive-scale]   -> it undershoots. Expected for the stock P-only gains "
              "(measured 37-46% on 2026-08-22).")

    # ── Was it actually a straight line? ─────────────────────────────────────
    # All four modules are steered to the same angle, so a straight run needs
    # all four wheels turning at the same speed. Any spread means the chassis
    # is being scrubbed sideways, and a curved path measured start-to-end as a
    # chord under-reports the distance travelled -- which biases the scale.
    def _wrap(a):
        return np.arctan2(np.sin(a), np.cos(a))

    align = np.degrees(np.abs(_wrap(steer_meas - steer_cmd)))
    align = np.minimum(align, 180.0 - align)
    speeds = np.median(np.abs(measured), axis=0)
    spread = (speeds.max() / speeds.min() - 1.0) * 100.0 if speeds.min() > 1e-9 else float("inf")

    print(f"\n{'module':8s} {'steer err':>10s} {'wheel speed':>13s} {'vs fastest':>11s}")
    print("-" * 46)
    for i, name in enumerate(MODULE_ORDER):
        e = align[:, i]
        e = np.median(e[np.isfinite(e)]) if np.any(np.isfinite(e)) else float("nan")
        print(f"{name:8s} {e:9.2f}d {speeds[i]:13.4f} "
              f"{(speeds[i] / speeds.max() - 1) * 100:10.1f}%")
    print("-" * 46)

    straight = spread < 2.0
    if straight:
        print(f"[drive-scale] wheel speeds agree to {spread:.1f}% -- the run was straight.")
    else:
        print(f"[drive-scale] wheel speeds disagree by {spread:.1f}%. All four modules point "
              f"the same way, so\n[drive-scale] that spread scrubs the chassis sideways and "
              f"curves the path.")
        print("[drive-scale] A curved path measured start-to-end is a chord, so it "
              "UNDER-reports the")
        print("[drive-scale] distance travelled, which biases the recommended scale HIGH. "
              "Treat it as an")
        print("[drive-scale] upper bound until the run is straight.")
        worst = MODULE_ORDER[int(np.argmin(speeds))]
        print(f"[drive-scale] slowest module: {worst}. If it is the same one every run, that is "
              f"mechanical\n[drive-scale] or a per-module gain issue, not something the "
              f"command path can fix.")

    # The half that actually settles the scale.
    cmd_distance = float(np.sum(np.median(np.abs(commanded), axis=1) * dts))
    meas_distance = float(np.sum(np.median(np.abs(measured), axis=1) * dts))
    if args.spin:
        print("\n[drive-scale] --spin gives no straight-line ground truth, so the scale "
              "is not settled by this run. Re-run without it when you have the space.")
        return 0

    print(f"\n[drive-scale] over the driving phase the command implies "
          f"{cmd_distance:.3f} m of travel")
    print(f"[drive-scale]   (this is exactly what BaseOdometry integrates)")
    print(f"[drive-scale] the wheels report {meas_distance:.3f} m")
    actual = _ask_float("\n[drive-scale] tape-measure the distance travelled, in metres "
                        "(blank to skip): ")
    if actual is None or actual <= 0:
        print("[drive-scale] skipped. Re-run and supply the distance to settle the scale.")
        return 0

    truth = actual / cmd_distance if cmd_distance > 1e-9 else float("nan")
    print(f"\n[drive-scale] actual / commanded = {truth:.3f}"
          + ("" if straight else "   (from a curved run -- an upper bound on the scale)"))
    if abs(truth - 1.0) < 0.05:
        print("[drive-scale]   -> commanded m/s IS true m/s. The odometry is honest and "
              "the scale is right for these gains.")
    else:
        over = "over" if truth < 1.0 else "under"
        print(f"[drive-scale]   -> commanded m/s is NOT true m/s. BaseOdometry {over}-reports "
              f"distance by {abs(1.0 / truth - 1.0) * 100:.0f}%.")
        # actual = k * commanded * scale, so truth = k * scale and the scale
        # that makes commanded m/s true is 1/k = scale / truth. Dividing, not
        # multiplying: travelling short of the command needs *more* scale.
        print(f"[drive-scale]   -> to make them agree, set drive_command_scale in "
              f"{Path(args.manifest).name} to {scale / truth:.3f} (currently {scale:g}).")
        print("[drive-scale]   -> that value is only valid for the gains in that manifest. "
              "Re-measure after any gain change.")
        if track < 0.8:
            print("[drive-scale]   -> but note the loop is undershooting, and a P-only loop's "
                  "undershoot moves with load, battery voltage and speed. Raising the scale "
                  "papers over that at one operating point. The durable fix is gains that "
                  "track (config/base_pid_commissioned.json) plus a scale near 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
