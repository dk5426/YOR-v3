#!/usr/bin/env python3
"""Odin cloud diagnostic — answers: is cloud_slam the device's growing map or a
per-frame scan, did mapping mode take, and does odometry track motion.

Run it and SLOWLY MOVE THE BOX ~1 m during the 12-second capture:
    conda activate slam-odin
    python tools/odin_diag.py
Then paste the SUMMARY block.
"""
import os, sys, time
import numpy as np
import pyodin


def out(*a):
    print("DIAG:", *a); sys.stdout.flush()


def main():
    s = pyodin.OdinSensor()
    out("connecting (mapping mode)…")
    if not s.start("slam", 15.0, 1):   # map_mode=1
        out("NOT CONNECTED — power-cycle the box and retry."); os._exit(1)

    rc = s.set_custom_parameter("custom_map_mode", 1)
    out(f"set custom_map_mode=1 -> rc={rc}  (0 == applied)")

    out(">>> MOVE THE BOX SLOWLY ~1 m over the next 12 seconds <<<")
    samples = []
    t0 = time.time()
    while time.time() - t0 < 12.0:
        time.sleep(2.0)
        od = s.get_odom()
        cl = s.get_cloud()
        if od is None or cl is None or cl["n"] == 0:
            out("  (waiting for data…)"); continue
        xyz = np.asarray(cl["xyzrgba"])[:, :3].astype(np.float64) * 1e-4
        cen = xyz.mean(axis=0)
        diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
        pos = np.asarray(od["pos"], float)
        samples.append((pos, int(cl["n"]), cen, diag))
        out(f"  odom={np.round(pos,3).tolist()}  n={cl['n']:6d}  "
            f"centroid={np.round(cen,3).tolist()}  bbox_diag={diag:.2f}m")

    if len(samples) >= 2:
        p0, n0, c0, _ = samples[0]
        p1, n1, c1, _ = samples[-1]
        odom_moved = float(np.linalg.norm(p1 - p0))
        cen_moved = float(np.linalg.norm(c1 - c0))
        growth = n1 / max(n0, 1)
        out("================ SUMMARY ================")
        out(f"odom moved over capture : {odom_moved:.3f} m")
        out(f"cloud centroid moved    : {cen_moved:.3f} m")
        out(f"cloud point-count n0->n1: {n0} -> {n1}  (x{growth:.2f})")
        if growth > 2.0:
            out("=> cloud_slam looks like the DEVICE'S GROWING MAP -> display LATEST (replace), don't accumulate.")
        else:
            out("=> cloud_slam looks like a PER-FRAME scan -> accumulate; map quality depends on odometry drift.")
        out("=========================================")
    s.stop(); sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
