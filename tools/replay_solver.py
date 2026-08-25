#!/usr/bin/env python3
"""
replay_solver.py — run recorded end-effector trajectories through the solver.

Every hardware run so far has been judged against a different operator input,
which makes "does this configuration balance base, lift and arms better?"
unanswerable: the input changed at the same time as the solver did. This plays
one recorded trajectory through several solver configurations and reports them
side by side.

**The recordings already exist.** `_TrajectoryRecorder` writes
`left_target_ee_*` / `right_target_ee_*` every tick, and those columns *are*
the solver's input -- the same SE3 pair `solve()` is called with. Any file in
artifacts/wholebody_logs/trajectories/ is a replayable trace.

One caveat worth stating plainly. Teleop generates each target relative to
where the hand currently is, so in a live run the input depends on where the
robot got to; replaying an absolute target sequence under a different
configuration is therefore not a perfect re-enactment. It is a fair A/B --
identical requested hand trajectory, different whole-body resolution -- which
is exactly what picking between configurations needs.

    # compare the shipped defaults against the alternatives
    python tools/replay_solver.py --log artifacts/wholebody_logs/trajectories/traj_...csv

    # watch one of them
    python tools/replay_solver.py --log ...csv --view eager

    # try a value that has no preset
    python tools/replay_solver.py --log ...csv --set base_motion_weight=50

Touches no hardware and opens no CAN bus.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mink                                                        # noqa: E402
from robot.arm.wholebody_ik import WholeBodyIK, WholeBodyIKConfig  # noqa: E402

# Named configurations worth comparing. Each is applied on top of the shipped
# defaults, so "shipped" is deliberately empty rather than a restatement -- a
# restatement would silently stop tracking the defaults it claims to be.
PRESETS: dict[str, dict] = {
    "shipped": {},
    # Item 32's flat cost: rejects noise, but only recruits the base once the
    # arms have already contorted.
    "flat": {"base_motion_weight_min": 100.0},
    # Pre-item-32: no base preference at all. The base answered 24% of pure
    # tracker noise here.
    "unweighted": {"base_motion_weight": 1.0, "base_motion_weight_min": 1.0},
    # Opens the manipulability gate earlier, so the chassis commits sooner.
    "eager": {"base_weight_gate_on": 0.050, "base_weight_gate_full": 0.035},
    # Later and harder: base stays expensive until the arms are truly stretched.
    "reluctant": {"base_weight_gate_on": 0.035, "base_weight_gate_full": 0.015},
    # ("nocarry" retired 2026-08-25 along with base_velocity_continuity
    # itself -- the carry measured no effect anywhere and was removed.)
    # The other redundancy resolution, as a baseline.
    "soft": {"redundancy_resolution": "soft"},
}


def load_targets(path: Path, stride: int = 1, max_ticks: int | None = None):
    """(left, right) SE3 sequences from a trajectory CSV, plus its config row."""
    rows = list(csv.reader(Path(path).open()))
    cfg = [r for r in rows if r and r[0].startswith("#")]
    header = rows[len(cfg)]
    data = rows[len(cfg) + 1:]
    index = {k: n for n, k in enumerate(header)}
    for side in ("left", "right"):
        if f"{side}_target_ee_0" not in index:
            raise SystemExit(f"{path} has no {side}_target_ee_* columns "
                             "(recorded before those existed)")

    def se3(row, side):
        v = [float(row[index[f"{side}_target_ee_{k}"]]) for k in range(7)]
        return mink.SE3(np.array(v, dtype=float))

    data = data[::stride]
    if max_ticks:
        data = data[:max_ticks]
    return ([se3(r, "left") for r in data],
            [se3(r, "right") for r in data],
            cfg[1] if len(cfg) > 1 else [])


def run(name: str, overrides: dict, left, right, view: bool = False) -> dict:
    cfg = WholeBodyIKConfig(dt=1.0 / 30.0, solver="pyqpmad", max_iters=10, **overrides)
    ik = WholeBodyIK(config=cfg)
    ik.init_from_keyframe("home")

    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(ik.model, ik.configuration.data)

    errs, mus, base_v, base_q, lifts, solve_ms = [], [], [], [], [], []
    try:
        for T_l, T_r in zip(left, right):
            t0 = time.perf_counter()
            res = ik.solve(T_l, T_r)
            solve_ms.append((time.perf_counter() - t0) * 1000.0)
            A_l, A_r = ik.forward_kinematics()
            errs.append(max(
                float(np.linalg.norm(A_l.translation() - T_l.translation())),
                float(np.linalg.norm(A_r.translation() - T_r.translation()))))
            d = ik.configuration.data
            mus.append(min(
                float(np.exp(ik._log_manipulability(ik._arm_jacobian(s, d))))
                for s in ("left", "right")))
            base_v.append(np.asarray(res.base_velocity, dtype=float).copy())
            base_q.append(np.asarray(res.base_position, dtype=float).copy())
            lifts.append(float(res.lift_q))
            if viewer is not None:
                viewer.sync()
                time.sleep(1.0 / 30.0)
                if not viewer.is_running():
                    break
    finally:
        if viewer is not None:
            viewer.close()

    errs = np.array(errs); mus = np.array(mus)
    bv = np.array(base_v); lifts = np.array(lifts); bq = np.array(base_q)
    dt = 1.0 / 30.0
    lin = bv[:, :2]
    speed = np.hypot(lin[:, 0], lin[:, 1])
    live = speed > 1e-9
    h = np.arctan2(lin[:, 1], lin[:, 0])
    dh = np.abs(np.arctan2(np.sin(np.diff(h)), np.cos(np.diff(h))))
    both = live[1:] & live[:-1]
    # Forward / backward split of the base command, in the chassis frame. The
    # robot faces -Y at yaw 0, so its forward axis in the world is
    # (sin yaw, -cos yaw) -- the same convention as BaseAxisMap and
    # BasePoseController.heading_offset. Worth separating because the two are
    # not equivalent to an operator: driving under the work is the point of
    # recruiting the base, driving back out from under it is the cost.
    yaw = bq[:, 2]
    fwd_v = bv[:, 0] * np.sin(yaw) - bv[:, 1] * np.cos(yaw)
    fwd_m = float(fwd_v[fwd_v > 0].sum() * dt)
    back_m = float(-fwd_v[fwd_v < 0].sum() * dt)

    return {
        "name": name,
        "fwd_m": fwd_m,
        "back_m": back_m,
        "back_pct": 100.0 * back_m / max(fwd_m + back_m, 1e-9),
        "ee_med_mm": float(np.median(errs)) * 1000,
        "ee_p95_mm": float(np.percentile(errs, 95)) * 1000,
        "mu_min": float(mus.min()),
        "mu_med": float(np.median(mus)),
        "base_path_m": float(np.sum(speed) * dt),
        "base_yaw_deg": float(np.degrees(np.sum(np.abs(bv[:, 2])) * dt)),
        "lift_travel_m": float(np.sum(np.abs(np.diff(lifts)))),
        "reversals_pct": float((dh[both] > np.pi / 2).mean() * 100) if both.any() else 0.0,
        "solve_ms": float(np.median(solve_ms)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=None,
                        help="trajectory CSV to replay (default: the most recent one "
                             "with usable target columns)")
    parser.add_argument("--configs", default="shipped,flat,unweighted,eager,reluctant",
                        help="comma-separated preset names, or 'all' "
                             f"({', '.join(PRESETS)})")
    parser.add_argument("--set", action="append", default=[], metavar="K=V",
                        help="extra config override, applied to every preset; repeatable")
    parser.add_argument("--view", metavar="NAME",
                        help="also play this one configuration in the MuJoCo viewer")
    parser.add_argument("--stride", type=int, default=1,
                        help="replay every Nth tick (default 1)")
    parser.add_argument("--max-ticks", type=int, default=None)
    args = parser.parse_args(argv)

    path = args.log
    if path is None:
        candidates = sorted(glob.glob(
            str(_REPO / "artifacts/wholebody_logs/trajectories/*.csv")),
            key=os.path.getmtime, reverse=True)
        for c in candidates:
            head = Path(c).open().readline()
            if os.path.getsize(c) > 200_000:
                path = Path(c)
                break
        if path is None:
            raise SystemExit("no trajectory logs found; pass --log")

    extra: dict = {}
    for item in args.set:
        k, _, v = item.partition("=")
        try:
            extra[k.strip()] = float(v)
        except ValueError:
            extra[k.strip()] = v.strip()

    names = list(PRESETS) if args.configs == "all" else [
        n.strip() for n in args.configs.split(",") if n.strip()]
    for n in names + ([args.view] if args.view else []):
        if n not in PRESETS:
            raise SystemExit(f"unknown preset {n!r}; have {', '.join(PRESETS)}")

    left, right, cfg_row = load_targets(path, args.stride, args.max_ticks)
    print(f"replaying {Path(path).name}: {len(left)} ticks")
    pid = next((c for c in cfg_row if c.strip().startswith("base_pid=")), None)
    if pid:
        print(f"  recorded under {pid.strip()[9:70]}")
    if extra:
        print(f"  extra overrides applied to every preset: {extra}")
    print()

    rows = []
    for name in names:
        overrides = dict(PRESETS[name]); overrides.update(extra)
        rows.append(run(name, overrides, left, right))
        r = rows[-1]
        print(f"  {name:11s} done  (EE {r['ee_med_mm']:.2f} mm, mu_min {r['mu_min']:.4f})")

    print(f"\n{'config':11s} {'EE med':>8s} {'EE p95':>8s} | {'mu min':>8s} {'mu med':>8s}"
          f" | {'base path':>10s} {'fwd':>7s} {'back':>7s} {'back%':>6s} {'base yaw':>9s}"
          f" {'lift':>7s} | {'revers':>7s} {'solve':>7s}")
    print("-" * 126)
    for r in rows:
        print(f"{r['name']:11s} {r['ee_med_mm']:7.2f}m {r['ee_p95_mm']:7.2f}m |"
              f" {r['mu_min']:8.4f} {r['mu_med']:8.4f} |"
              f" {r['base_path_m']:9.2f}m {r['fwd_m']:6.2f}m {r['back_m']:6.2f}m"
              f" {r['back_pct']:5.0f}% {r['base_yaw_deg']:8.0f}d {r['lift_travel_m']:6.2f}m |"
              f" {r['reversals_pct']:6.1f}% {r['solve_ms']:6.1f}ms")
    print("\n  EE  = tracking error, lower is better (this is the actual task)")
    print("  mu  = worst-arm manipulability; HIGHER is better, 0.0506 is the home posture")
    print("  base path / yaw / lift = how much of the motion each subsystem absorbed")
    print("  fwd / back = chassis-frame split of that path; back is driving out from")
    print("               under the work, which is what recentering spends the base on")
    print("  revers = base direction reversals >90 deg, lower is steadier")

    if args.view:
        print(f"\nplaying '{args.view}' in the viewer -- close the window to finish")
        overrides = dict(PRESETS[args.view]); overrides.update(extra)
        run(args.view, overrides, left, right, view=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
