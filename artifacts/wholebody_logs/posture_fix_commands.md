# joint7 posture-fix — command log

Tracking commands run against `robot/yor.py`'s `--posture-*` flags while
chasing the joint7 null-space wobble. Run from `/home/pi-v3/YOR-v3-try` with
`conda activate yor-nero`.

## Tried

1. **refresh-target** (2026-08-20, 15:38–16:52, logs `yor_20260820_{153834,163715,163743,164054,164706,165305}.log`)
   ```
   python robot/yor.py --no-base-motion --no-lift-motion --no-flash-base-pid --posture-refresh-target
   ```
   `refresh_posture_target=True`, no joint7 override. No null-space-wobble
   diagnostic hits. (EE tracking-error numbers originally logged here were
   computed against the wrong CSV columns — see correction under run 2 — and
   the source CSVs are gone, so they can't be recomputed. Not reliable, ignore.)

2. **stiffen-joint7** (2026-08-20, 16:52–16:58, logs `yor_20260820_{165034,165445}.log`;
   repeated 2026-08-20 17:49–17:53, log `yor_20260820_174938.log`)
   ```
   python robot/yor.py --no-base-motion --no-lift-motion --no-flash-base-pid --posture-stiffen-joint7
   ```
   `refresh_posture_target=False` (default/stale), joint7 cost 10x
   (`--posture-joint7-scale` default). Smoothest-feeling motion both times.
   Both null-space-wobble diagnostic hits of the whole session occurred in
   the first pass. **Correction**: the tracking-error numbers first reported
   for runs 1–2 (e.g. "16.1cm / 3.7cm") were wrong — CSV EE columns are
   `wxyz_xyz` (quaternion first, translation last), and that analysis read
   them as translation-first, so it was actually measuring quaternion-component
   distance, not position error in meters. The wobble-hit counts and
   held-still/moved breakdowns are unaffected (those come from the app's own
   `_NullSpaceMonitor`, not from my column indexing).

   In both stiffen-joint7-only passes, `_CommandJitterMonitor` keeps flagging
   joint **6** (`*_arm_joint7`) as worst-reversal — same joint the docstring
   says is normal/expected here — so the stiffening doesn't clearly suppress
   command-level jitter on that joint even though the null-space-wobble test
   (a different measure: EE motion correlation, not reversal rate) mostly
   passes.

3. **stiffen-joint7 + refresh-target** (2026-08-20, 17:45–17:48, log
   `yor_20260820_174514.log`, trajectories `traj_20260820_{174551,174623,
   174704,174817,174822}.csv`)
   ```
   python robot/yor.py --no-base-motion --no-lift-motion --no-flash-base-pid --posture-stiffen-joint7 --posture-refresh-target
   ```
   Confirmed smooth early on, but degrades over a sustained reach and then
   resets after re-homing — this is the "smooth but IK problems" run. Per
   trajectory-recording segment (each is one continuous reach before the
   next Quest re-home):

   | segment | dur | worst-reversal joint | flip rate | right-arm pos err (max) |
   |---|---|---|---|---|
   | 174551.csv | 29s | 0 / 6 | ≤2% | 0.10 mm |
   | 174623.csv | 37s | 0 (mostly) | 0% but std climbing (4→14 mrad) | 0.53 mm |
   | 174704.csv | 68s | **0, 3, 4** | up to **66.7%** | **1.00 mm** (at pos_threshold) |
   | 174817/174822.csv | <1s/24s | — (too short to report) | — | 0.00 mm (reset after re-home) |

   So `--posture-refresh-target` does stop joint7 (index 6) from being the
   worst-reversal joint — but the instability doesn't disappear, it moves to
   joints 1, 4, and 5 (indices 0, 3, 4: load-bearing, not the wrist) and
   *grows* the longer the reach continues, with tracking error climbing
   toward the 1mm convergence threshold in lockstep. Re-homing resets it.

   Likely mechanism: with `refresh_posture_target=True`, *every* joint's
   posture reference becomes "wherever we are right now," every tick — so
   the posture task stops being a pull toward a good nominal configuration
   and becomes pure rate-damping with no restoring force. A 7-DOF arm doing
   a 6-DOF EE task has exactly one redundant DOF; pinning joint7 harder just
   forces that DOF's slack onto whichever joint the local Jacobian null-space
   direction favors next — and with nothing pulling it back toward home over
   a long reach, it can drift/oscillate further before the QP's own damping
   catches it.

4. **lower joint7 stiffening, keep refresh-target on** (2026-08-20, 18:50–19:04,
   logs `yor_20260820_{185055,190033}.log`)
   ```
   python robot/yor.py --no-base-motion --no-lift-motion --no-flash-base-pid --posture-stiffen-joint7 --posture-refresh-target --posture-joint7-scale 3
   ```
   Run twice, but **not a clean repeat of the posture question** — `robot/arm/arm.py`'s
   `joint_acc_max` was edited between the two runs (95%-of-firmware →
   ~124%-of-firmware, see that file's inline comment for the full story),
   which is a separate, physical-layer confound: it changes how well the
   *hardware* tracks whatever the whole-body solver commands, independent of
   the posture-cost/null-space question these two runs were meant to isolate.
   First pass (95%-of-firmware accel) felt laggy, second (124%) felt less so.
   Tracking-error/reversal analysis of these two specifically for the
   posture question got interrupted before finishing — the accel-limit
   change needs to be held fixed if we want to re-read these for null-space
   behavior specifically.

5. **EE:posture weight-ratio diagnostic** (2026-08-20, 19:31–19:55, logs
   `yor_20260820_{193108,194410}.log` — `ee_cost_scale=1x` baseline vs `100x`)
   ```
   python robot/yor.py --no-base-motion --no-lift-motion --no-flash-base-pid --posture-stiffen-joint7 --posture-refresh-target --posture-joint7-scale 3 --ee-cost-scale 100
   ```
   **Result: structural, not a weighting problem.** Pushing the EE:posture
   ratio from ~1000:1 to ~100,000:1 did not shrink the instability —
   `_CommandJitterMonitor`'s aggregate stats were flat-to-worse at 100x
   (dispatch max|Δq| mean 86.8→110.5 mrad, worst-joint |Δq| std
   10.7→13.1 mrad, flip-rate max 6.4%→11.8%), and directly measuring
   joint-space discontinuities in the CSVs (Δq per tick vs. EE Δpos same
   tick) found a normalized rate of ~5.1 vs ~5.5 "branch-switch" events per
   1000 arm-ticks — indistinguishable given the sample size, if anything
   slightly higher at 100x. Soft weighting cannot fix this at any ratio;
   confirms a real null-space projector is needed (see the two remaining
   options below).

   **New characterization of "arm jumps between solutions":** the clearest
   branch-switch events (large joint reconfiguration, EE moves <1cm same
   tick) show a consistent signature — joints at array index **2 and 4**
   (mid-arm: elbow region) swinging by large, nearly equal-and-opposite
   amounts (up to ±0.79 rad, ~45°, in a single ~30-40ms tick) while the EE
   barely moves, e.g. `traj_20260820_195048.csv:4363`:
   `Δq = [-0.146, -0.030, +0.791, -0.089, -0.771, +0.089, 0.0]`. This is the
   classic "arm-angle self-motion" signature for a 7-DOF anthropomorphic
   arm (shoulder/elbow redundancy circle) — the QP jumping between two
   different points on the self-motion manifold that both satisfy the EE
   task, because nothing in the soft-cost objective enforces a single
   consistent choice along that manifold tick-to-tick. Different joints
   than the joint7/joint-0 story from runs 1-3, same underlying cause:
   wherever the redundant DOF's "cheapest" outlet currently is, it's
   contested by the same soft-QP structure.

6. **Implemented both real fixes, gated, default unchanged** (2026-08-20,
   code only — not yet run on hardware). Added `redundancy_resolution` to
   `WholeBodyIKConfig` (`robot/arm/wholebody_ik.py`): `"soft"` (default,
   bit-for-bit identical to before — verified via `tests/test_wholebody_control.py`,
   48/48 checks still pass), `"hard_constraint"` (EE tasks promoted to
   mink's `constraints=`), and `"dls_projector"` (hand-rolled damped-least-
   squares + explicit null-space projection, joint/collision limits enforced
   afterward by projecting onto them via a small auxiliary QP). Both new
   modes validated headless (small ramped targets converge cleanly in both;
   a deliberate large one-shot jump raises in `hard_constraint`, as expected
   for an exact-equality approach, while `dls_projector` still converges in
   14 iterations — matches the predicted tradeoff exactly). One real bug
   caught and fixed during validation: `_project_onto_limits` was projecting
   the *velocity* against inequality constraints that mink actually defines
   over *Δq* (configuration displacement) — off by a factor of `dt` (~90x),
   which made `dls_projector` fail to converge at all until fixed.

   New CLI flags in `robot/yor.py`:
   ```
   --redundancy-resolution {soft,hard_constraint,dls_projector}   (default: soft)
   --dls-damping FLOAT           # dls_projector only, default 0.05
   --hard-constraint-damping FLOAT   # hard_constraint only, default 1e-3
   ```
   Also stamped `redundancy_resolution`, `dls_damping`, and
   `hard_constraint_extra_damping` into each trajectory CSV's config-comment
   line, alongside the existing posture/EE-cost fields.

7. **A/B all three modes on hardware** (2026-08-20, 20:57–21:19, logs
   `yor_20260820_{205734,210126,210450,211117,211718}.log`) — 5 runs:
   1) baseline (`soft`), 2) `hard_constraint`, 3) `dls_projector` (kp=8,
   pre-accel-bump arm.py), 4) `dls_projector` + `default_kp` 8→10, 5) same
   + `arm.py`'s `joint_acc_max` raised further (see that file's inline
   comment — now the p95-of-real-demand estimate, pooled across 28 recorded
   sessions). Same posture flags across all 5
   (`--posture-stiffen-joint7 --posture-refresh-target --posture-joint7-scale 3`).

   **Branch-switching: confirmed fixed, in proportion to how "real" the
   projector is** — rate of joint jumps >100mrad with EE moving <1cm, per
   1000 arm-ticks:

   | run | mode | branch-switch rate | right pos err median |
   |---|---|---|---|
   | 1 | soft (baseline) | 4.06 | 0.04 mm |
   | 2 | hard_constraint | 3.60 | 0.02 mm |
   | 3 | dls_projector, kp=8 | **0.34** | 10.75 mm |
   | 4 | dls_projector, kp=10 | 0.47 | 17.94 mm |
   | 5 | dls_projector, kp=10, new accel | 1.95 | 22.28 mm |

   `hard_constraint` barely moved the needle here — surprising given the
   theory (exact null-space confinement should stop this outright); worth
   a closer look at *why* separately, but not blocking since `dls_projector`
   clearly is doing its job (8-12x fewer branch-switch events than baseline).

   **But real tracking error is 2-3 orders of magnitude worse under
   `dls_projector` than under `soft`/`hard_constraint`** (median 11-22mm vs.
   0.02-0.04mm) — this is almost certainly the "still needs work" feeling
   behind this round's request to optimize further. The headless synthetic
   test when this was built only showed ~2mm residual on a *static* target
   (`dls_damping`'s expected, bounded exactness tradeoff) — 11-22mm under
   real, continuously-moving teleop targets is much bigger than that
   predicted, and needs explaining before tuning blind.

   **Compute cost is not the bottleneck.** All 5 runs hold the same ~29.9Hz
   actual solve-loop rate and ~89Hz dispatch rate regardless of mode — the
   control loop isn't falling behind. Benchmarked `dls_projector` headless
   (home position, small offset target, this dev machine): DLS math alone
   (Jacobian + pseudoinverse + null-space projection) ~0.14ms/call, the
   limit-projection QP on top ~0.42ms/call — call it ≤10 × 0.56ms ≈ 5.6ms
   worst case per `solve()` (`max_iters=10`), well inside the 33ms budget.
   **Tried and reverted one speed idea**: special-casing
   `ConfigurationLimit`/`VelocityLimit` (both pure per-DOF box constraints)
   as a closed-form `np.clip` instead of routing them through the QP too —
   measured *slower* (~0.75ms/call vs ~0.56ms/call): daqp solves a box QP
   this small fast enough natively that the extra numpy bookkeeping to avoid
   it costs more than it saves. Reverted; documented in
   `_project_onto_limits`'s docstring so it isn't re-attempted blind.

   **Implemented**: `iters` and `solved` now recorded per-tick in the
   trajectory CSV (`robot/wholebody_control.py`'s `_TrajectoryRecorder`,
   verified 48/48 existing tests + a standalone header/row column-count
   check). This is the actual missing piece for the tracking-error
   question — right now there's no way to tell "solved=False, ran out of
   `max_iters` before converging" apart from "solved=True but the DLS
   pseudoinverse has an inherent small bias" from the CSV alone, and those
   two have completely different fixes (raise `max_iters` vs. lower
   `dls_damping`).

8. **Re-ran `dls_projector` (with `iters`/`solved` now logged) + a new
   `soft` + new-kp/accel comparison** (2026-08-20, 21:45–21:56, logs
   `yor_20260820_{214511,215019}.log`)
   ```
   python robot/yor.py --no-base-motion --no-lift-motion --no-flash-base-pid --posture-stiffen-joint7 --posture-refresh-target --posture-joint7-scale 3 --redundancy-resolution dls_projector
   ```
   ```
   python robot/yor.py --no-base-motion --no-lift-motion --no-flash-base-pid --posture-stiffen-joint7 --posture-refresh-target --posture-joint7-scale 3
   ```
   Both at run 5's kp=10 / current accel limits.

   **Definitive: iteration-starved, not a `dls_damping` bias.** Of 6815
   `dls_projector` ticks, only 18.4% converge (`solved=True`) — and those
   converge in exactly 1 iteration to 0.04-0.07mm, matching `soft`'s
   precision exactly. The other 81.6% hit `iters=10` (`max_iters`) without
   converging, and *those* are exactly the ticks carrying the 25-34mm mean
   error. Almost bimodal — barely any ticks land in between (2-9 iters is
   ~48 rows out of 6815). So the pseudoinverse isn't converging to a biased
   answer; for anything but a small per-tick target step it isn't finishing
   within budget at all. `dls_damping=0.05` throttles how much of the gap
   each iteration closes — that's the actual lever to pull, not a
   fundamental exactness ceiling.

   **`soft` + the new kp/accel tuning got *worse* at branch-switching, not
   better** — 7.20 events/1000 ticks vs. run 1's 4.06 (with the old, more
   conservative accel limits), while tracking stayed excellent (0.02mm
   median, matching run 1). This is an important, separate result: it rules
   out "the branch-switching was actually a hardware/acceleration artifact,
   not a software one" — giving the hardware more headroom to move fast
   made `soft`'s branch-switching *worse*, not better, because nothing in
   the soft-cost QP was ever what constrained it in the first place. Good
   confirmation that `dls_projector` (once its convergence issue is fixed)
   is solving a real, distinct problem, not a symptom of conservative accel
   limits that would have gone away on its own.

9. **Root-caused the `dls_projector` convergence failure — it was a bug,
   not a tuning problem** (2026-08-20, code + headless analysis only).

   `_solve_qp_dls_projector` takes only `ee_tasks`; it never received
   `base_fix_task` / `lift_fix_task`, which `solve()` puts in `other_tasks`
   for the other two modes. So the minimum-norm pseudoinverse was free to
   route EE motion through the base — by far the cheapest route in
   joint-norm terms (1 cm of base slide buys 1 cm of EE travel; the arms
   need several joints to cooperate for the same thing) — and then
   `solve()` zeroed all of it a few lines later, because
   `--no-base-motion` sets `fix_base=True`.

   Measured at a 10 cm dual-arm target under the hardware configuration
   (`fix_base`/`fix_lift` both set): **77.2% of each step's norm landed on
   the base, 4.7% on the lift, leaving ~15.6% to actually execute.**
   Per-iteration error curves, same target, same conditions:

   ```
   full-nv DLS (buggy) : 99.0 → 95.9 → 92.9 → 90.1 → … → 71.8   (~3%/iter, needs >100)
   restricted-DOF DLS  : 99.0 → 48.7 → 3.0  → 0.095 → 0.005     (converged, 4 iters)
   soft (reference)    : 99.0 → 45.1 → 4.3  → 0.124 → 0.007     (converged, 4 iters)
   ```

   This also explains run 8's bimodality exactly: a small per-tick target
   step needs only one of those crippled steps and converges in 1 iteration
   (the 18.4%); anything larger crawls at ~1/6 rate and hits `max_iters=10`
   with nothing in between. And it explains why `dls_damping` did nothing
   in the earlier sweep — damping was never the binding constraint, the
   discarded base motion was.

   **Two earlier hypotheses in this log are now superseded, keeping them
   noted so they aren't re-tried:**
   - "Lower `dls_damping`" (the old item 9): swept λ from 0.05 down to
     1e-6 headless, essentially no change to the convergence curve. Not
     the lever.
   - "The DLS step is conservative because minimum-norm doesn't use the
     full velocity budget the way `soft`'s QP does": `soft` *does* saturate
     `arm_vel_limit` on the first iteration of a large step where DLS peaks
     ~0.5 rad/s, but that turned out to be a symptom of `soft` solving the
     arm-only problem, not the cause of the speed gap. With the DOF
     restriction in place DLS converges in the same 3-4 iterations as
     `soft` with no velocity scaling added at all — so the "scale the DLS
     step toward the velocity limit" change was not needed and was not
     made.

   **Fix**: the DLS solve now runs over only the DOFs actually free to move
   — `fix_base` / `fix_lift` drop the base / lift columns from J before the
   pseudoinverse, the null-space projector is built in that reduced space,
   and the result is scattered back into a full-nv vector with the locked
   entries left at zero. Headless validation (`daqp`, `dt=1/30`,
   `max_iters=10`, hardware `fix_base`/`fix_lift` config):

   | case | soft | hard_constraint | dls_projector |
   |---|---|---|---|
   | one-shot 10 cm target | solved, 5 iters, 0.004 mm | *raises* (expected) | solved, 5 iters, 0.005 mm |
   | ramped teleop (20 steps) | 20/20, 1 iter, ≤0.04 mm | 20/20, 1 iter, ≤0.04 mm | 20/20, 1 iter, ≤0.22 mm |
   | base free (mobile case) | 20/20, ≤0.0001 mm | — | 20/20, ≤0.004 mm |

   Joint limits still respected (0 violations under an aggressive 25 cm
   30-step sweep) and the base stays at exactly 0.0 with `fix_base`. The
   residual ~0.2 mm gap vs `soft` on ramped teleop is `dls_damping`'s
   expected bounded looseness, well inside `pos_threshold` (1 mm).
   `tests/test_wholebody_control.py` still 48/48.

   (`tests/test_arm_config.py` shows 3 failures, unrelated to this and
   pre-existing: it asserts `default_kp=8.0` and
   `NATIVE_ACC_LIMIT=[4.60, 3.86, 5.05, 7.73, 6.33, 6.33, 8.41]`, while
   `arm.py` now runs kp=10.0 and the 3-file p95 line
   `[4.98, 3.42, 3.57, 9.57, 5.38, 5.76, 8.99]`. Note the values the test
   wants are the *commented-out* pooled-28-file line in `arm.py` — worth
   deciding which of the two is meant to be active, then syncing the test.)

10. **Hardware confirmation of the DOF fix, + an app-output-frequency A/B**
    (2026-08-20 22:32 and 2026-08-21 13:29, logs
    `yor_20260820_223234.log` / `yor_20260821_132905.log`). Identical IK
    config (`dls_projector`, λ=0.05, stiffen 3x, refresh-target); the only
    intended difference is the teleop app's output frequency, changed
    between them to match the controller rate.

    **The DOF fix is confirmed on hardware.** Against run 8 (same flags,
    pre-fix):

    | | run 8 (pre-fix) | A last night | B today |
    |---|---|---|---|
    | `solved` | 18.4% | 88.6% | 80.6% |
    | `iters=1` | 1211/6815 (18%) | 6970/9061 (77%) | 3365/5767 (58%) |
    | median right pos err | 20.07 mm | **0.041 mm** | **0.124 mm** |
    | median err on converged ticks | — | 0.00-0.33 mm | 0.03-0.26 mm |

    The `iters=10` pile-up that defined run 8 (81.6% of ticks) is gone;
    converged ticks now track to a few hundredths of a mm, matching `soft`.

    **A's alarming aggregates are one joint-limit event, not a solver
    problem.** A's p99 of 544 mm / max 587 mm comes almost entirely from a
    single 557-tick (18.6 s) stretch in `traj_20260820_223310.csv` where
    `right_arm_joint1` sat pinned at exactly +2.70 rad — its hard limit
    (`jnt_range` ±2.70) — while the operator kept driving the target
    further out. During it, EE x tracked perfectly (−0.353 vs −0.354) while
    y and z ran away (target y −0.330 vs actual −0.130). The solver did the
    right thing; the target was simply unreachable. `right_q0` is within
    0.01 of its limit on 6.1% of A's ticks and **0.0%** of B's.

    **With those ticks excluded, the two runs invert:**

    | (joint-limit ticks excluded) | A last night | B today |
    |---|---|---|
    | `solved` | 96.3% | 85.3% |
    | median err | 0.000 mm | 0.103 mm |
    | p95 err | 0.75 mm | 30.46 mm |
    | p99 err | 16.54 mm | 72.31 mm |
    | `iters=10` | 3.7% | 14.8% |

    **But the A/B is confounded and does not isolate the frequency change.**
    B's session involved substantially faster motion (target EE speed p90
    0.755 m/s vs A's 0.461; median target step 13.9 mm vs 9.1 mm) and never
    approached a joint limit. Binned by matched target speed, medians are
    near-identical across every bin (A 0.000-0.435 mm, B 0.001-0.435 mm);
    only the p90 differs, and only in the 0.05-0.30 m/s bins (A 0.62-3.88 mm
    vs B 33-36 mm). Two different human teleop sessions differing in speed,
    reach and workspace region cannot separate a frequency effect from a
    motion effect.

    **What the frequency change demonstrably did do** is alter the target
    stream's character: the fraction of moving ticks immediately preceded by
    a target *hold* (a controller tick where no new target arrived) went
    **0.8% → 27.8%**. So the app used to stream a fresh target essentially
    every controller tick, and now updates intermittently relative to it —
    the expected beat pattern between two free-running ~30 Hz loops.
    Measured effective update interval is ~30 Hz in both (median 33.4 ms vs
    33.7 ms). Tested whether that intermittency costs accuracy: it does not
    — in B, ticks immediately following a hold and ticks in a continuous
    stream have the same error distribution (p90 5.94 mm vs 5.89 mm). So
    the holds are visible but benign.

## Next

11. **Get a clean read on the app frequency** (not yet run). The two runs
    above cannot answer it. Options, cheapest first:
    - Repeat both settings back to back in one sitting, driving roughly the
      same motion envelope each time (similar speed, similar workspace
      region, no joint-limit excursions), then compare within matched
      target-speed bins as above. Two runs per setting would help, since
      session-to-session variance is clearly large.
    - Better: replay a scripted target trajectory instead of live teleop, so
      the input is bit-identical across settings and any difference is
      genuinely the frequency. Nothing in the repo does this today.
    Given the hold analysis showed intermittency is benign, the honest
    prior is that this change is close to neutral for tracking accuracy;
    it is worth settling only if it matters for some other reason (latency
    feel, CPU, app-side simplicity).

12. **Handle joint-limit saturation** — arguably now the most valuable
    remaining item, and independent of everything above. When
    `right_arm_joint1` pinned at its limit, the arm silently stopped
    tracking for 18.6 s while error grew past half a metre, with no
    operator-visible signal. `result.solved` was already False throughout,
    so the information exists; nothing surfaces it. Cheapest useful step is
    a rate-limited warning from `_step()` when a solve stays unconverged
    for more than ~0.5 s *and* some joint sits within a small epsilon of
    `jnt_range`, naming the joint — enough for the operator to know to back
    off rather than fight it. Distinguish from ordinary transient
    non-convergence during fast motion, which is normal and recovers.

11. Lower priority, from run 7's earlier A/B: why `hard_constraint` doesn't
    reduce branch-switching much despite being the theoretically-exact fix.
    Possible explanations: the exact-equality solve still converges to a
    *different* point on the self-motion manifold each time for essentially
    the same reason `soft` does (exact confinement stops posture from
    *fighting* the EE task, but adds no preference for continuity along the
    manifold), or `hard_constraint_extra_damping=1e-3` isn't doing enough
    regularizing work there. Not blocking since `dls_projector` is already
    far ahead on this metric.

10. Lower priority, from run 7's earlier A/B: why `hard_constraint` doesn't
    reduce branch-switching much despite being the theoretically-exact fix.
    Possible explanations: the exact-equality solve still converges to a
    *different* point on the self-motion manifold each time for essentially
    the same reason `soft` does (exact confinement stops posture from
    *fighting* the EE task, but adds no preference for continuity along the
    manifold), or `hard_constraint_extra_damping=1e-3` isn't doing enough
    regularizing work there. Not blocking since `dls_projector` is already
    far ahead on this metric once its convergence issue is fixed.

12. **Elbow-swivel hardware A/B** (2026-08-22, logs `yor_20260822_{173450,
    173907,174338,174835,175330}.log`) — 5 runs, identical apart from the
    null-space knobs, all `dls_projector` / stiffen 3x / refresh-target.

    Normalised by EE path length, because these are live teleop sessions of
    different lengths (32–66 m) and raw per-tick rates are not comparable:

    | run | config | EE path | flips/m | swivel °/m | >5° jumps | q̇ rev% |
    |---|---|---|---|---|---|---|
    | 1 | swivel 0 (control) | 32.4 m | 1.1 | **87°** | 11.8% | 16.6% |
    | 2 | swivel 1 | 65.7 m | **0.6** | **43°** | 5.0% | 16.4% |
    | 3 | swivel 5 | 59.2 m | 3.5 | **22°** | 3.4% | 15.4% |
    | 4 | swivel 1 + continuity 2 | 61.5 m | 2.1 | 39° | 5.2% | 15.1% |
    | 5 | swivel 1 + manipulability | 54.3 m | 0.9 | 9° | 0.0% | 14.5% |

    **Swivel confirmed on hardware, first time.** Its own direct measure --
    swivel-angle travel per metre of EE path -- falls monotonically with
    weight: 87 -> 43 -> 22 °/m, and the fraction of samples jumping >5°
    goes 11.8% -> 5.0% -> 3.4%. Tracking is unaffected throughout
    (median 0.05–0.14 mm) and command-level joint reversals are flat
    (14.5–16.6%), so none of this came out of EE accuracy or added command
    noise.

    **But weight 5 is worse on the discrete-flip metric** (3.5/m vs the
    control's 1.1), and those 205 events are spread across the run rather
    than being one incident. Reading: a stiff swivel holds the elbow until
    the task genuinely demands it move, then releases abruptly -- trading
    continuous drift for occasional snap. Weight 1 is the better balance and
    is what the operator independently rated best.

    **Run 5's jitter is a timing problem, not a solver-quality one.** Its
    solve loop missed rate constantly (only 9 of 37 reports at 29.9 Hz,
    versus 37 of 41 in run 2, with one report at 18.6 Hz), which is exactly
    the ~39 ms/solve manipulability cost against a 33.3 ms budget predicted
    on the bench. Its *output* was the cleanest of all five (lowest joint
    reversal rate, tightest swivel) -- it is simply not being delivered on
    time. Confirms manipulability as diagnostic-only.

    **Continuity 2 (run 4)** was roughly neutral on hardware rather than
    harmful as the bench sweep suggested -- similar swivel drift to run 2,
    somewhat more discrete flips. Not enough to justify changing the 0
    default; not enough to rule it out either.

    Caveat: run 1 (control) covered only 32 m against 54–66 m for the rest,
    so it is the weakest leg of the comparison. Worth one more control run
    at a comparable length before quoting the control numbers hard.

13. **`dls_projector` promoted to default; `hard_constraint` removed**
    (2026-08-22, code only).

    `WholeBodyIKConfig.redundancy_resolution` now defaults to
    `"dls_projector"`, so the sim node, the IK demo and the whole-body
    controller's internal fallback all match what runs on hardware.
    `robot/yor.py`'s flag defaults are the validated run-2 configuration --
    `--posture-stiffen-joint7` and `--posture-refresh-target` are on (both
    now `BooleanOptionalAction`, so `--no-...` is available for A/Bs),
    `--posture-joint7-scale 3`, swivel 1.0, continuity 0, manipulability
    off. The whole run is therefore just:

    ```
    python robot/yor.py --no-base-motion --no-lift-motion --no-flash-base-pid
    ```

    Removed as dead: the `hard_constraint` mode in full
    (`_solve_qp_hard_constraint`, `hard_constraint_extra_damping`,
    `--hard-constraint-damping`, its CSV stamp) and the `--ee-cost-scale`
    diagnostic, whose question is settled and recorded in item 5 above.
    `"soft"` is kept -- it is the baseline any future A/B needs.

    **Behavioural note worth knowing, found while doing this.** Against a
    target that is unreachable outright, `dls_projector` drives *further*
    into the joint limits than `soft` (7 pinned joints vs 4) and then
    settles, producing no motion above the 0.05 rad dispatch deadband --
    i.e. it goes quiet rather than twitching at the stop. That is the better
    behaviour, but it is also the same failure mode as the 18.6 s silent
    stall in item 12: when the operator drives past the workspace, nothing
    tells them, and now the arm is completely still rather than visibly
    straining. It strengthens the case for the joint-limit warning in item
    12, which is still not implemented.

    `tests/test_wholebody_control.py`'s manual-override test had been
    relying on the old twitch-at-the-limit behaviour: it used a 1.5 m
    (unreachable) target, so "authority returns after the window" passed
    only because `soft` kept jittering. Retargeted to 0.6 m -- out of
    arm-only reach so the base is still recruited, but reachable once it
    moves -- which tests the handover it is named for and passes in both
    modes. `tests/test_arm_config.py` was also synced to the live kp/accel
    values (was asserting the commented-out estimate).

14. **Base + lift brought into the loop: logging, shutdown restore, and a
    review of the base command path** (2026-08-22, code only).

    The arms are tuned; base and lift now run alongside them. Three changes.

    **(a) The trajectory log records every subsystem, not just the arms.**
    `_TrajectoryRecorder` grew from 49 to 90 columns and a second config
    header line carrying the base/lift knobs (`control_hz`,
    `base_vel_deadband`, `lift_kp`, `lift_kd`, ...), so a run is
    interpretable from the file alone.

    New columns, per solve tick:

    * base — `base_active`, `base_req_{vx,vy,wz}` (solver, unclamped),
      `base_body_{fwd,lat,yaw}` (after clamp + deadband),
      `base_sent_{0,1,2}` (what left `_dispatch_base`). Three separate
      stages on purpose: "the solver wanted little", "the deadband ate it"
      and "the clamp capped it" are different problems that look identical
      at the wheels.
    * swerve — `swerve_enabled`, `swerve_target_*` (what `Base` holds),
      `swerve_prof_*` (what the S-curve profiler has reached), and per
      module in FL/FR/RR/RL order `steer_cmd_*`, `steer_meas_*`,
      `drive_cmd_*`, `drive_meas_*`.
    * lift — `lift_active`, `lift_mode`, `lift_goal`, `lift_meas`,
      `lift_cmd_vel`, `lift_vel_est` (the PD's own filtered derivative, not
      one recomputed offline), `lift_age`, `lift_blocked`.

    Two things that were wrong before and are now right: the `record()` call
    moved to the *end* of `_step`, so a row's base and lift columns describe
    the tick that produced them rather than the previous one; and
    `RotationMotor.get_absolute_rad()` is a new accessor separate from
    `get_position_rad`, because the latter returns the last *commanded*
    angle while `USE_FEEDBACK_FOR_STEER` is False — logging it as measured
    would make every module look like it tracks perfectly.

    A missing reading records as `nan`, never 0.0. `Base.swerve_telemetry()`
    is read through `getattr`, so a stub base logs nan columns instead of
    killing the control tick. Costs no extra CAN traffic: `GetVelocity` and
    `GetAbsoluteEncoderPosition` read cached periodic-status frames.

    Sampling caveat: this is the 30 Hz solve rate against a 324 Hz swerve
    loop and 50 Hz SPARK status. It resolves module slew and PID settling;
    it does not resolve anything at the swerve loop's own rate.

    **(b) Shutdown restores the SPARK stock gains.** The commissioned gains
    live in controller RAM, which outlives the process — whatever opens the
    bus next inherits them silently, and the manifest itself records that
    some of them (the ±0.25 steering output clamp) are specific to this
    configuration. `config/base_pid_defaults.json` holds the stock values
    and goes through the same validated, read-back sync as startup.

    Guards: it only runs if startup actually flashed (so
    `--no-flash-base-pid` cannot clobber someone else's commissioning), it
    stops the control loop first (gains must not change under a live
    setpoint), and it cannot raise out of the shutdown path ahead of the arm
    drop. `--no-restore-base-pid` opts out.

    To confirm the stock numbers against real hardware: power-cycle the
    SPARKs, then `python tools/base_pid_preflight.py --manifest
    config/base_pid_defaults.json --verify-only`. A clean OK on all eight
    means the file matches what a SPARK actually comes up holding.

    **(c) Review of the base command path** — `docs/BASE_COMMAND_LOOP_REVIEW.md`.
    Ten findings, each reproduced against the real functions rather than
    inferred. **Nothing was changed**: base kinematics is what is about to
    be tuned, and editing it in the same session would make the results
    uninterpretable. The three that matter most:

    * a zero base command re-aims all four modules to 0° (90° of slew for a
      stop while driving forward), and whole-body base velocity crosses the
      deadband constantly because base motion is emergent;
    * the deadband is per-axis, so a 0.053 m/s request at 21° is sent at 0° —
      direction, not just magnitude, and the odometry integrates the
      distorted command;
    * `cos_error_scaling` is inert, because the "error" it uses is between
      consecutive *commands*; it collapses after one 3.1 ms tick while the
      module needs O(100 ms) to slew.

    Suggested order in the doc: settle the drive-velocity scale first
    (`DRIVE_VEL_SCALE = 2.0` with no wheel-radius conversion anywhere, and
    `TIRE_RADIUS` unused — everything else is measured in units that depend
    on it), then the re-aim, then the deadband, then real steering feedback.

    Run command, everything live:

    ```
    python robot/yor.py
    ```

    Base or lift only, to separate them:

    ```
    python robot/yor.py --no-lift-motion      # arms + base
    python robot/yor.py --no-base-motion      # arms + lift
    ```

    Tests: `test_base_pid_preflight` 98/98 (was 71; +27 for the stock
    manifest and the shutdown wiring), `test_wholebody_control` 57/57
    (was 48; +9 for the log). All other suites unchanged.
    `test_api_parity` 10/11 remains the pre-existing `home_arms` failure.

15. **Base command loop: the offline-verifiable fixes applied** (2026-08-22,
    code only). `DRIVE_VEL_SCALE` left at 2.0 by decision.

    Five of the ten review findings were fixable without a hardware
    measurement or a judgement call. Those are in; the other five are marked
    open in `docs/BASE_COMMAND_LOOP_REVIEW.md` with why.

    * **(1) A stop no longer re-aims the modules.**
      `_vehicle_velocity_to_angle_and_speed` holds the last commanded angle
      for any module below `ZERO_SPEED_EPS_MPS` (1e-3 m/s), applied *after*
      the flip and the cosine so neither can put a direction back. Only the
      drive setpoint goes to zero. Previously a stop from a forward drive
      re-aimed all four modules by 90 degrees, and whole-body base velocity
      crosses the deadband constantly.
    * **(2) The linear deadband and clamp act on the magnitude.**
      New `WholeBodyController._limit_linear` rescales the (forward, lateral)
      pair instead of treating the axes separately, so a 0.053 m/s request 21
      degrees off the forward axis goes out at 21 degrees rather than 0. Same
      reasoning already written down for `BaseOdometry.apply_correction`.
      Side effect worth knowing: a diagonal whose *axes* are each under the
      deadband but whose magnitude is not (0.019, 0.019 -> 27 mm/s) now
      survives, where before it was dropped entirely. Yaw got its own
      `base_yaw_deadband` in rad/s -- one scalar had been serving as both
      m/s and rad/s.
    * **(6b) The enum import is fixed, so two calls now actually run.**
      `from sparkcan_py import CtrlType, IdleMode, MotorType, SensorType`
      raised (the last two are not exported), and the bare `except` set all
      four to None -- disabling every `SetIdleMode` and `SetCtrlType` call.
      **Watch the first hardware run**: coast/brake behaviour on a disabled
      base may differ from every previous session, because until now it was
      whatever the SPARKs held in flash.
    * **(8) Dead and wrong forward kinematics deleted.** `self.C`,
      `_angle_and_speed_to_vehicle_velocity` (round-tripped +1.0 rad/s as
      -0.316) and `_map_steer_angles`. `swerve_odom.py` is the correct
      forward model and now guards the IK in the new test.
    * **(10) The S-curve integrates measured time.** `_update_state` was
      computing the real elapsed time and discarding it while the profiler
      used the nominal period; on a loaded Pi every ramp ran long.

    Also: **the joystick sends a zero on stick release.** It used to stop
    *sending* below an L1 threshold, leaving the last above-threshold value
    standing -- which the 108 Hz relay then re-sent forever, about 10 mm/s of
    creep nothing ever cleared.

    **New diagnostic, tied to the DRIVE_VEL_SCALE question.**
    `Base.swerve_configuration()` reads idle mode, control type and both
    conversion factors back off the eight controllers, and `yor.init()`
    prints them once. Those factors live in SPARK flash and are set through
    the REV Hardware Client, which is precisely why the 2.0 reads as a magic
    number -- it is one half of a unit conversion whose other half is not in
    git. `velocity_cf` decides which half:

    ```
    metres per motor rotation   0.049922    (calibrated, swerve_odom.py)
    => velocity_cf for true m/s 0.00083203
    => the same via DIAMETER    0.00166407  (exactly 2x)
    ```

    0.00083 means the robot travels at twice the commanded m/s and the IK's
    odometry records half of reality; 0.00166 means the commanded m/s is
    truthful. The startup line settles it on the next run at zero cost.

    Test script: `tests/test_base_kinematics.py`, 46 checks, no CAN bus.

    ```
    python tests/test_base_kinematics.py
    ```

    It stubs SparkFlex, runs the real `Base.control_loop` in its thread, and
    checks the fixes end to end -- including that a stop zeroes the wheel
    speed without moving the steering. It also round-trips base_motor's
    inverse kinematics against `swerve_odom.py`'s least-squares forward model
    (given base_motor's CAD geometry), which is what makes finding 8 a
    regression test rather than a deletion. And it checks base_motor's
    `sparkcan_py` import names against the *real* installed binding's
    exports, captured before the stub is installed, so 6b cannot come back.

    Suites: `test_base_kinematics` 46/46 (new), everything else unchanged and
    green. `test_api_parity` 10/11 remains the pre-existing `home_arms` gap.

16. **Swerve controller configuration read off the hardware** (2026-08-22).
    Read directly with `GetVelocityConversionFactor` etc. while nothing owned
    the bus — read-only, no heartbeats, no parameter writes.

    ```
    controller             vel_cf     pos_cf   idle   ctrl
    FL/FR/RR/RL drive  0.00084633    1.00000      0      1   (kCoast, kVelocity)
    FL/FR/RR/RL steer  1.00000000    1.00000      0      3   (kCoast, kPosition)
    ```

    **Two things settled.**

    *(a) `DRIVE_VEL_SCALE = 2.0` is not a unit conversion.* The drives are
    already configured in true m/s:

    ```
    velocity_cf                   0.000846326
      -> implies m per motor rot  0.050780
      vs calibrated (swerve_odom) 0.049922      1.72% apart
    against "true m/s" 0.00083203  ratio 1.0172
    against "2x slip"  0.00166407  ratio 0.5086
    ```

    The radius/diameter hypothesis is dead — that would have read 0.00166.
    The residual 1.7% is in the direction physics predicts: the factor was
    set from nominal geometry, the calibration measures a loaded tyre's
    slightly smaller effective rolling circumference. `position_cf = 1.0`
    everywhere independently confirms `METERS_PER_ROTATION` is per *motor*
    rotation, so the swerve_odom calibration and this factor are consistent.

    So the 2.0 is a bare doubling of an already-correct setpoint. Remaining
    question is whether the velocity loop tracks: if it does, the robot runs
    at ~2x commanded and `BaseOdometry` records half of reality; if it
    undershoots ~2x (plausible — `p=0.35, i=0, d=0, velocity_ff=0.23` has no
    integrator forcing zero steady-state error) the 2.0 compensates and the
    odometry is honest. New tool:

    ```
    python tools/measure_drive_scale.py            # wheels up, 0.10 m/s
    python tools/measure_drive_scale.py --spin     # in place, floor-safe
    ```

    It compares `drive_meas_raw` against the setpoint (`drive_cmd_mps` x
    DRIVE_VEL_SCALE). ~1.0 means the loop tracks, ~0.5 means it undershoots.
    Refuses to run if another process owns the bus; writes no parameters.

    *(b) The 6b fix is behaviourally inert on this robot.* Idle mode is
    already `kCoast` on all eight, drives already `kVelocity`, steering
    already `kPosition` — so the newly-enabled `SetIdleMode`/`SetCtrlType`
    calls write what was already there. The "watch the first hardware run for
    a coast/brake change" warning in item 15 does not apply. The fix still
    stands: the guard was dead by accident and would have mattered the moment
    a controller came back from flash differently.

17. **DRIVE_VEL_SCALE: it should be ~1.0** (2026-08-22).

    Two new facts closed most of this. First, the 2.0 **predates the
    feed-forward-dominated gains** — it was already there under the default
    PID — so it was not introduced to compensate for `velocity_ff = 0.23`.
    Second, `YOR-v3-Problems-DON'T-USE` had already investigated this and
    stalled at exactly the same point: its Phase 0 measured
    `DRIVE_METERS_PER_RAW_UNIT = 0.047530598` (five tape runs, 0.74% CV),
    could not read the conversion factor from the binding, and left two
    hypotheses — not in closed-loop velocity mode, or a conversion factor
    ~631x off. `config/calibration/yor-v3-base.yaml` carries
    `command_conversion_reconciled: false` and "every speed number on the
    base is nominal until this closes". It never closed.

    Today's hardware read closes both hypotheses: `ctrl_type = kVelocity` on
    all four drives (so they *are* closed-loop velocity), and
    `velocity_cf = 0.000846326` (set for m/s, not 631x off). Combining that
    with either rolling calibration, assuming the loop tracks:

    | calibration | m / motor rot | robot travels | scale should be |
    |---|---:|---:|---:|
    | swerve_odom.py | 0.049922 | 1.97x commanded | 1.017 |
    | PHASE0_BASELINE | 0.047531 | 1.87x commanded | 1.068 |

    So DRIVE_VEL_SCALE should be ~1.0, and the base has most likely been
    running at ~2x every commanded speed, with `BaseOdometry` recording half
    of reality. Unconfirmed only in that "the loop tracks its setpoint" is
    still an assumption -- `tools/measure_drive_scale.py` settles it.

    The two calibrations differ by 5% from each other, which is a separate
    small open question (different dates, tyres or loading).

    **Corroboration worth noting.** That repo's Phase 2 independently found
    five of the ten items in `docs/BASE_COMMAND_LOOP_REVIEW.md`: the profiler
    integrating a hard-coded 1/250 instead of measured time (10), per-axis
    limiting giving a diagonal sqrt(2) too much (2), `Base(max_vel=...)`
    stored and never read (noted under 6/10), and steering optimisation
    running against `get_position_rad()`'s commanded value (3/4). They are
    not speculative.

18. **Where DRIVE_VEL_SCALE = 2.0 came from, and a correction to the stock
    manifest** (2026-08-22).

    "Default"/"stock" does **not** mean the REV factory zeros. The values the
    SPARKs revert to on a power cycle are recorded in
    `YOR-v3-Problems-DON'T-USE/config/base_pid_manifest.json` (whose status
    field says "a controller power cycle reverts every module to stock"):

    ```
    drive stock:  p=0.2,  i=0,  d=0.1,   ff=0,  output +/-1.0
    steer stock:  p=2.0,  i=0,  d=0.01,  ff=0,  output +/-1.0
    ```

    `config/base_pid_defaults.json` had all zeros -- my assumption, and wrong.
    Corrected. The failure mode was specific: the shutdown restore would not
    have restored anything, it would have left the base **limp** until the
    next power cycle brought the real stock values back. A new check in
    `tests/test_base_pid_preflight.py` fails if that file ever goes all-zero
    again (101/101).

    **This also explains the 2.0.** The stock drive set is a P-only velocity
    loop with no feed-forward, and a P-only loop cannot reach setpoint. With
    duty = P x error and v_free = 5676 rpm x 0.000846326 = 4.80 native:

    ```
    act/sp = P*v_free / (1 + P*v_free)
    stock p=0.2, ff=0  ->  49.0% of setpoint  ->  command scale 2.04
    ```

    So DRIVE_VEL_SCALE = 2.0 was a correct compensation *for the stock gains*.
    It is not correct for the gains that ship: the commissioned set is
    feed-forward dominated (p=0.35, ff=0.23) and its own commissioning
    evidence records **steady error 0.000 native median** on the floor -- it
    tracks. So since the 2026-08-17 retune the base has most likely been
    running at ~2x every commanded speed, with BaseOdometry recording half of
    reality. Prediction for `tools/measure_drive_scale.py`:
    `measured/setpoint ~ 1.0`.

    **Prior tuning evidence found, worth reading before re-tuning anything.**
    That manifest carries full step-response records for both loops:

    * drive P=0.35/D=0/FF=0.23 -- rise 125 ms, settling 360 ms, overshoot
      6.92% median. D=10 rejected on the floor as audibly harsh (100 Hz
      torque ripple; the profiler dispatches at 100 Hz into a 1 kHz loop).
      **Caveat that matters: every floor run clamped output to +/-0.25 while
      the manifest ships +/-1.0, so P is effectively 7x and D 5x what they
      were measured under.** Re-measure before trusting them above 0.25.
    * steer P=20/D=6 -- t90 70 ms, stall 0.168 deg, peak slew 265-353 deg/s,
      dead time 20-25 ms irreducible by gain. Stall error is flat from P=12
      to P=20 at ~2 LSB of the absolute encoder, so it is the measurement
      floor, not stiction: more P will not improve accuracy. D=6 measured
      inert (Td = 0.3 ms) and is an operator preference. Output clamped to
      +/-0.25 saturates for any error above 4.5 deg, so most reorientations
      run bang-bang at the clamp -- which is also why no D value helps.
      `d_filter` has never been swept and is called the only untried lever.

    The steering slew figures corroborate review finding 1: a 90 degree
    re-aim is ~300 ms at 300 deg/s plus dead time, so the old zero-command
    re-aim was costing about 600 ms out and back on every pause.

19. **Stock gains are now the default** (2026-08-22, code only).

    The two manifests are renamed to say what they are, and the default
    flipped:

    | file | gains | when |
    |---|---|---|
    | `config/base_pid_stock.json` | drive p=0.2 d=0.1 ff=0; steer p=2.0 d=0.01, +/-1.0 | **default**, applied at startup |
    | `config/base_pid_commissioned.json` | drive p=0.35 ff=0.23; steer p=20 d=6, +/-0.25 | opt-in |

    ```
    python robot/yor.py                                              # stock
    python robot/yor.py --base-pid-manifest config/base_pid_commissioned.json
    ```

    **Why.** DRIVE_VEL_SCALE = 2.0 is correct for the stock P-only drive loop
    (~49% of setpoint) and wrong for the commissioned feed-forward one (which
    tracks). Running stock makes the speed axis self-consistent again:
    commanded m/s is roughly true m/s and BaseOdometry is honest -- which is
    the right baseline to tune the command path against. The commissioned set
    stays available but opt-in until `tools/measure_drive_scale.py` settles
    the scale.

    Second, independent reason to treat the commissioned set with care: every
    floor run behind those numbers clamped controller output to +/-0.25 while
    the manifest ships the drive range at +/-1.0, so P is effectively 7x and
    D 5x what they were measured under.

    **Expect softer steering.** Stock steer Kp is 2.0 against the
    commissioned 20.0, and stock never saturates (p=2.0 against a +/-1.0
    clamp saturates only past 180 deg of error) where the commissioned set
    runs bang-bang above 4.5 deg. Module angle tracking will be visibly
    looser and steady-state angle error larger. That is measurable now --
    `steer_cmd_*` vs `steer_meas_*` in the trajectory log -- which makes a
    stock run a genuine commissioning baseline rather than a compromise.

    The shutdown restore now **skips itself** when startup already applied
    the stock manifest, rather than printing "restoring stock" over eight
    no-op writes. It still runs, correctly, whenever a different manifest was
    flashed.

    `DEFAULTS_MANIFEST` is gone; `STOCK_MANIFEST` / `COMMISSIONED_MANIFEST`
    replace it and `DEFAULT_MANIFEST` aliases stock. CLI flag
    `--base-pid-defaults-manifest` renamed to `--base-pid-stock-manifest`.
    `docs/BASE_SWERVE_PID_GAINS.md` now documents both sets and why stock
    wins for now. Tests 102/102 (a new check asserts the default IS stock).

20. **Two runs analysed: the axis map was crossed, and the encoder was being
    read in the wrong units** (2026-08-22). Logs
    `traj_20260822_202723` (2542 rows, 2014 moving) and
    `traj_20260822_203406` (3624 rows, 1094 moving).

    **(a) "Forward did not go forward" — BaseAxisMap was crossed.** Three
    independent confirmations:

    * from the logs, `base_sent_0 == base_body_lat` and
      `base_sent_1 == base_body_fwd`, both to max|diff| = 0.00e+00;
    * `base_motor` builds each wheel vector as `atan2(target[1], target[0])`,
      so element 0 aims the modules at 0 deg and element 1 at +90 -- verified
      against the logs at **0.00 deg residual on all four modules** across
      158 low-yaw ticks;
    * the previous codebase measured on blocks that 0 deg is physically
      forward and +90 is left, and `joystick.py` has always put its forward
      stick in element 0, which is why the joystick drove correctly and only
      whole-body did not.

    Fixed: `forward_index=0, lateral_index=1`. The yaw sign is the one value
    still unverified. `tests/test_wholebody_control.py` had an assertion
    locking in the crossed order -- it asserted `[2.0, 1.0, 3.0]` -- corrected
    with a note.

    **(b) `GetAbsoluteEncoderPosition` returns TURNS, not degrees.** My
    `get_absolute_rad` divided by 360, which collapses a 0..1 reading to
    nearly zero and makes the recovered angle a constant fixed only by the
    module offset. The logs show it plainly: the four modules reported
    90.0 / 0.0 / -90.0 / -180.0 deg with a **total spread of 1 deg** while
    the commanded angles swept the full circle -- exactly the constants the
    offsets predict. The previous codebase's calibration states it directly:
    `steer_turns_per_raw_unit: 1.0`, "GetAbsoluteEncoderPosition already
    returns turns".

    This was worse than a logging bug: `get_position_rad` had the same
    division on its `USE_FEEDBACK_FOR_STEER` branch, so turning that flag on
    -- the recommended next experiment -- would have fed the steering
    optimizer a constant. Fixed via a new `get_absolute_turns()` primitive;
    `get_position_deg` now derives from it so its name stays true.

    **(c) Did the base reach its commanded velocity? No.** Steady-segment
    medians, `drive_meas / (drive_cmd x DRIVE_VEL_SCALE)`:

    | run | loop reaches | wheels turn at |
    |---|---:|---:|
    | 202723 | 40% of setpoint | 0.79x commanded m/s |
    | 203406 | 36% of setpoint | 0.73x commanded m/s |

    Per module 0.31-0.42, consistent across both runs. That is the stock
    P-only loop behaving as predicted -- the model in item 18 gave 49%
    unloaded, and these are loaded floor runs, so 36-40% is the right family.
    It also confirms the runs were on stock gains.

    So neither configuration is currently correct on the speed axis: **stock
    + 2.0 gives ~0.75x commanded, commissioned + 2.0 would give ~2x.** The
    right end state is the commissioned gains with DRIVE_VEL_SCALE ~= 1.0.

    **(d) Steering tracking under stock Kp=2.0**, after recovering the true
    angles from the mis-scaled logs: median error 12-15 deg (run 202723) and
    2-3 deg (run 203406), p90 27-65 deg. Against the commissioned Kp=20's
    measured 0.17 deg stall. So module angle error is a second, independent
    contributor to "the base does not go where it is told" -- on top of (a).

    **Gap found: the trajectory CSV does not record which PID manifest was
    flashed.** It had to be inferred from the tracking ratio. Worth stamping.

    Suites: `test_base_kinematics` 58/58 (+12), `test_wholebody_control`
    57/57, everything else unchanged.

21. **The trajectory log now stamps which gains it was driving on**
    (2026-08-22). Closing the gap found in item 20.

    `WholeBodyHardwareConfig.base_pid_provenance` is set by `yor.py` from what
    the startup sync actually did, and the recorder writes it into the second
    config row alongside `drive_vel_scale`:

    ```
    base_pid=base_pid_stock.json [drive p=0.2 ff=0, steer p=2 out=+/-1]
    base_pid=base_pid_commissioned.json [drive p=0.35 ff=0.23, steer p=20 out=+/-0.25]
    base_pid=not flashed (controllers hold whatever was there)
    ```

    The drive p/ff pair is stamped rather than just the file name, because the
    gain set changes what every speed number in the log means -- stock reaches
    ~40% of setpoint on the floor while the commissioned set tracks. A failed
    sync appends `SYNC FAILED`, so a log can never claim gains that were not
    applied. Tests 58/58.

22. **Six runs with the axis fix in: the steering problem is command churn,
    not gains** (2026-08-22). Runs 210848, 211006, 211308, 211453, 211710,
    211810 -- all stamped `base_pid_stock.json [drive p=0.2 ff=0, steer p=2]`,
    which is what the new provenance line is for.

    **Both fixes confirmed.** `steer_meas_*` now sweeps 355-360 deg per run
    (was a 1 deg spread). A forward-dominant request now lands in `sent_0`
    and steers the modules to 1-9 deg of straight ahead (was 90 deg).

    **Drive tracking unchanged, as expected.** Loop reaches 37-46% of
    setpoint, median 39.5%; wheels turn at **0.79x the commanded m/s**.
    Matches the 0.73-0.79x from the two earlier runs. Stock P-only loop
    behaving exactly as the model in item 18 predicts.

    **Steering error is large: median 11-19 deg, p90 57-69 deg, and 34-49% of
    moving ticks above 20 deg.** But almost none of that is the gains.
    Splitting by whether the commanded angle was holding still:

    | | median error | n |
    |---|---:|---:|
    | command parked (<20 deg/s) | **8-11 deg** | 2413 |
    | command slewing (>200 deg/s) | **23-28 deg** | 12590 |

    The parked figure is the soft Kp=2.0 and gains will fix it. The slewing
    figure is the module being physically outrun, and gains cannot.

    **How fast the commanded module angle is actually moving** (folded for the
    wheel reversal, so this is real required travel; hardware peak slew is
    265-353 deg/s):

    | requested \|v\| | median rate | p90 | above 300 deg/s |
    |---|---:|---:|---:|
    | 0.02-0.04 m/s | 552 deg/s | 2338 | **64.6%** |
    | 0.04-0.07 | 390 | 1807 | 58.7% |
    | 0.07-0.12 | 316 | 1002 | 52.6% |
    | 0.12-0.20 | 231 | 700 | 37.2% |
    | 0.20+ | 123 | 389 | 15.7% |

    **The mechanism is that the direction of a small vector is
    ill-conditioned.** `atan2(lat, fwd)` swings wildly when the vector is
    short, so the slower the base is asked to go, the faster its commanded
    heading whirls -- a clean 4.5x from the top band to the bottom. And the
    base spends 25-41% of its moving ticks within 2x the deadband, i.e. right
    in the worst band. Overall 27-44% of moving ticks demand a steering rate
    the modules physically cannot deliver, with p99 about 10x the limit.

    **Order of fixes matters here.** Going to the commissioned steering gains
    first would take the parked error from ~9 deg to ~0.2 deg and leave the
    23-28 deg slewing error untouched -- and the honest conclusion from that
    run would be "the gains barely helped". Slew-limit the commanded heading
    first.

    Suggested, in order:
    1. rate-limit the commanded base *heading* in `_dispatch_base` to ~200
       deg/s (under the 265 deg/s hardware floor, with margin), and integrate
       the limited command into odometry rather than the raw one;
    2. raise `base_vel_deadband` 0.02 -> 0.04-0.05 so the base is not
       commanded at all in the ill-conditioned regime -- costs fine
       positioning authority, so measure before keeping it;
    3. only then move to `base_pid_commissioned.json`.

23. **All offline base changes implemented and replay-validated** (2026-08-24).
    No hardware touched. Five changes, each testable without the robot.

    **(1) Heading rate limit.** New `WholeBodyController._limit_heading_rate`
    bounds how fast the *direction* of the base command may turn, at
    `base_heading_rate_limit = 3.49 rad/s` (200 deg/s, under the slowest
    measured module slew of 265). Magnitude is preserved -- the point is to
    stop the base whirling, not slow it down. Wheel reversal is accounted
    for: a heading change beyond 90 deg is measured against the *reversed*
    previous heading, because a module serves that by flipping the drive and
    not turning at all. The reference is frozen (not reset) across stops, so
    the next motion is limited from where the modules actually are. Odometry
    integrates the limited command, since the limiter runs before
    `_body_to_world`.

    **(2) `base_vel_deadband` 0.02 -> 0.04**, to keep the base out of the
    ill-conditioned band where the measured churn was worst.

    **(3) `USE_FEEDBACK_FOR_STEER = True`.** Safe now that the encoder units
    are right, and worth it now that the lag is measured. `cos_error_scaling`
    finally throttles drive speed while a module is still turning -- verified
    in the tests: full command, modules 90 deg off, drive speed goes to zero;
    modules arrived, full speed. Also fixed the encoder-loss fallback in
    `get_position_rad`, which returned the raw commanded fraction *without*
    removing the module offset -- an angle in a different frame from every
    other reading.

    **(4) The drive command scale travels with the gain set.** New
    `drive_command_scale` field in both manifests (stock 2.0, commissioned
    1.0), validated in `[0.1, 10.0]`, read by `yor.py` and threaded
    BaseController -> Base -> DriveMotor. `DRIVE_VEL_SCALE` is now only the
    fallback for direct `Base()` construction. Picking a manifest now picks
    both, so the documented footgun -- commissioned gains with the stock 2.0,
    a silent 2x overspeed -- is unreachable.

    **(5) Log stamps it**: `base_pid=base_pid_stock.json [drive p=0.2 ff=0,
    steer p=2 out=+/-1, scale=2]`, plus `base_heading_rate_limit`.

    **Replay validation over the six recorded runs.** The recorded solver
    output was pushed back through the new dispatch path offline. Module
    travel demanded, folded for reversal, against a 265-353 deg/s capability:

    | | median | p90 | above 300 deg/s |
    |---|---:|---:|---:|
    | before (0.02, no limit) | 167 deg/s | 854 | **33.2%** |
    | after (0.04, 200 deg/s) | 172 deg/s | **207** | **0.5%** |

    p90 falls 4x and the uncatchable fraction goes 33.2% -> 0.5%, while the
    median is unchanged -- the limiter is clipping the tail, not slowing
    ordinary motion. The raised deadband drops 20% of previously-commanded
    ticks, all from the worst band.

    Suites: `test_base_kinematics` 82/82 (+24), `test_wholebody_control`
    58/58, everything else green.

    **Still requires the robot, in this order:**
    1. `python tools/measure_drive_scale.py` (wheels up, ~6 s) -- confirm the
       stock loop still reads ~0.40 of setpoint, as a control.
    2. A run on stock to confirm the heading limit behaves as the replay
       predicts, and that steering error at parked command is still ~9 deg.
    3. `--base-pid-manifest config/base_pid_commissioned.json` -- this now
       brings scale 1.0 with it. Re-run `measure_drive_scale.py` first to
       confirm that loop tracks, then tape-measure a straight line and set
       the exact scale (arithmetic says 1.017-1.068).

24. **`measure_drive_scale.py` reworked for a floor run** (2026-08-24).
    Propping the robot up is not required, and the floor is the better test:
    it keeps the load the gains were commissioned under, and it is the only
    way to get ground truth. A wheels-up run can answer "does the loop reach
    setpoint" and nothing else.

    Three phases with a prompt between: **aim** (modules turn to
    straight-ahead at 0.03 m/s -- with USE_FEEDBACK_FOR_STEER on,
    cos_error_scaling holds the drive back while they turn, so the chassis
    barely moves), **mark**, **drive**, then **type in the tape measurement**.
    It prints the clear space needed before asking to proceed, and every exit
    path stops the base.

    It reports both halves:
    * `meas/setpoint` -- does the loop track? (the controller's own view)
    * `actual / commanded` -- does a commanded m/s equal a real m/s? This is
      the one that settles the scale, because `BaseOdometry` integrates the
      *commanded* velocity.

    ```
    python tools/measure_drive_scale.py                  # 0.15 m/s for 5 s
    python tools/measure_drive_scale.py --velocity 0.10 --seconds 4
    ```

    **Two bugs found by dry-running it against a stubbed CAN stack** (nothing
    on the bus, a fake controller modelling the stock loop's 40% gain):

    * the rewrite never called `base.start_control()`, so nothing would have
      been commanded and the run would have produced four columns of zeros;
    * the recommended scale was computed as `scale * truth` when it must be
      `scale / truth`. Travelling *short* of the command needs *more* scale,
      not less. The wrong formula turned a needed 2.49 into 1.61, and would
      have turned the commissioned case's correct 1.0 into 4.0 -- i.e. it
      pointed the wrong way in both directions.

    The dry run now reproduces the expected end-to-end result: 40% tracking,
    0.803 actual/commanded, "odometry over-reports by 25%", recommend 2.490.
    It also warns that raising the scale on an undershooting P-only loop only
    holds at one operating point, and the durable fix is the commissioned
    gains plus a scale near 1.

25. **Floor measurement with tape ground truth — the speed axis is now fully
    closed** (2026-08-24). Stock gains, scale 2.0, 0.15 m/s for 5 s.

    ```
    loop tracking      0.435 of setpoint  (per module 0.424-0.445)
    command implies    0.740 m
    wheels report      0.633 m
    tape               0.59  m
    actual/commanded   0.797
    ```

    **It agrees with the six logged runs**, which gave 0.73-0.79x commanded
    at 36-46% tracking -- but from ground truth rather than CAN telemetry, so
    it is an independent confirmation rather than the same number twice.

    **The whole chain is self-consistent.** Tracking x scale x the unit
    correction (PHASE0's metres-per-rotation over what the controllers were
    configured from) = 0.435 x 2.0 x 0.9360 = **0.814**, against a measured
    0.797. Within 2%. Every link -- velocity_cf, the gain set, the command
    scale, the rolling calibration -- now predicts the tape measure.

    **It settles the 5% calibration discrepancy from item 17.** The wheels
    turned 12.466 motor rotations for 0.59 m of ground:

    | calibration | predicted | vs tape |
    |---|---:|---:|
    | PHASE0 (5 tape runs) 0.047531 | 0.5925 m | **+0.4%** |
    | swerve_odom.py 0.049922 | 0.6223 m | +5.5% |

    PHASE0 is right for the robot as it stands. Likely explanation is not
    that anyone measured wrong: swerve_odom's calibration is dated
    2026-03-30, which predates the arms and lift, and a more heavily loaded
    tyre has a smaller effective rolling radius. **`swerve_odom.py` therefore
    over-reports distance by ~5.5%, and `robot/slam_node_.py` feeds it
    straight into the EKF.** Annotated in place with the evidence but
    deliberately not changed -- it feeds navigation, not base tuning, and the
    honest fix is to re-run `calibrate_drive.py` on the current robot.

    **`config/base_pid_commissioned.json` drive_command_scale 1.0 -> 1.068**,
    now measurement-backed: the controllers are configured from
    velocity_cf x 60 = 0.050780 m/rotation, so a loop that tracks perfectly
    needs 0.050780 / 0.047531 = 1.0684. Still provisional in one respect --
    it assumes the commissioned loop tracks *exactly* -- so re-run
    `measure_drive_scale.py --manifest config/base_pid_commissioned.json`
    and use the number it reports.

    **The tool's 2.510 recommendation for stock should NOT be taken.** It
    would make commanded m/s true at 0.15 m/s on this floor with this
    battery, and a P-only loop's undershoot moves with all three. The tool
    says so itself.

26. **The "commissioned" run never loaded the commissioned gains — tool bug**
    (2026-08-24).

    `measure_drive_scale.py --manifest config/base_pid_commissioned.json`
    read only `drive_command_scale` from the manifest. It never flashed the
    PID gains. So that run applied the commissioned **scale** (1.068) on top
    of the **stock gains**, which describes no real configuration -- and the
    output looked exactly like the commissioned loop underperforming badly.

    **Proof both runs used the same gains.** Same commanded 0.15 m/s,
    different scale so different setpoint:

    ```
    setpoint 0.2999 -> measured 0.1304
    setpoint 0.1601 -> measured 0.0562
    single fit:  measured = 0.531*setpoint - 0.0288   (both points exact)
    ```

    One P-only loop plus a stiction offset fits both points to four decimals.
    And the levels rule out the commissioned gains outright: feed-forward at
    0.23 would put the duty near the setpoint on its own, predicting ~0.300
    and ~0.160 against the 0.130 and 0.056 measured.

    **Bonus finding, and it matters.** The stiction offset means the tracking
    ratio is *not* constant -- it falls with speed (0.435 at setpoint 0.30,
    0.350 at 0.16). So the required command scale is speed-dependent: 2.51 at
    one speed, 2.96 at the other, an 18% spread across a 2x range. **A single
    `drive_command_scale` cannot correct a P-only loop at all.** That turns
    the tool's generic warning into a measured fact, and is the strongest
    argument yet for moving to the feed-forward gains, which cancel the
    offset rather than scaling around it.

    **Fixed:** the tool now reads the gains back and **refuses** unless the
    controllers hold the manifest's set, printing what is actually loaded and
    how to fix it. `--apply-gains` flashes them through the same
    readback-verified sync, and warns that they stay in RAM afterwards.
    Verified against fake controllers holding stock while asked for
    commissioned: exit code 1, mismatch listed, nothing driven.

    **To actually test the commissioned loop:**

    ```
    python tools/base_pid_preflight.py --manifest config/base_pid_commissioned.json
    python tools/measure_drive_scale.py --manifest config/base_pid_commissioned.json
    # afterwards, put stock back:
    python tools/base_pid_preflight.py --manifest config/base_pid_stock.json
    ```

    or in one step with `--apply-gains` on the measure command.

    `config/base_pid_commissioned.json`'s scale stays at the 1.068 derived in
    item 25; nothing in this run bears on it, because the commissioned loop
    was never running.

27. **Commissioned gains confirmed tracking; the scale is still not settled
    because the run curved** (2026-08-24).

    ```
    meas/setpoint   FL 1.000  FR 0.982  RR 1.002  RL 0.947   median 0.991
    command implies 0.740 m   wheels 0.761   tape 0.700
    actual/commanded 0.946 -> tool recommends 1.129
    ```

    **The commissioned loop tracks (99%)**, exactly as item 18's arithmetic
    predicted and as its own commissioning evidence claimed. The
    feed-forward gains do what the P-only ones cannot. That half is settled.

    **The distance half is not**, because the operator reported the motion
    was not straight, and the data says why: the four wheel speeds spread
    **5.8%**, with RL slowest. All four modules are steered to 0 deg, so
    unequal wheel speeds scrub the chassis sideways and curve the path -- and
    a curved path tape-measured start-to-end is a *chord*, so it under-reports
    distance travelled and biases the recommended scale **high**.

    Decomposition:

    | trusting | implies scale |
    |---|---:|
    | tape 0.70 m (chord of a curve) | 1.129 |
    | wheel travel 0.712 m (path length, no slip) | 1.110 |
    | PHASE0 calibration, arithmetic only | 1.068 |

    Implied metres-per-motor-rotation: 0.04733 from the earlier stock run,
    0.04671 from this one -- 1.3% apart, in the direction extra scrub pushes
    it. So `drive_command_scale` is somewhere near 1.07-1.13 and pinning it
    needs a straight run. **Left at 1.068**; the contamination is larger than
    the correction.

    **RL is the repeat offender.** Stock run: FL 0.436 FR 0.434 RR 0.445
    **RL 0.424** (5.0% spread). Commissioned run: FL 1.000 FR 0.982 RR 1.002
    **RL 0.947** (5.8% spread). Same module slowest, same module fastest (RR),
    across two completely different gain sets -- so it is not a gain fault.
    Candidates, in order: mechanical drag on RL (bearing, belt tension,
    gearbox), uneven load, or a worn tyre. Note also that both gain sets run
    `i = 0`, so nothing removes per-module steady-state droop; a small Ki, or
    a per-module `velocity_ff` trim, would equalise them if the cause turns
    out to be load rather than drag.

    **Tool now diagnoses this.** It reports per-module steering error and
    wheel speed, computes the spread, and when it exceeds 2% says the run was
    not straight, names the slowest module, and labels the scale
    recommendation an upper bound. Verified against a stub reproducing the
    measured 5.8% spread: output matches the real run to 0.002.

28. **`config/base_pid_hybrid.json` — commissioned drive, stock steering**
    (2026-08-24), requested to avoid the commissioned steering set's +/-0.25
    output clamp. Carries the same `drive_command_scale` as commissioned
    (1.068), since the scale depends only on the drive gains.

    ```
    python robot/yor.py --base-pid-manifest config/base_pid_hybrid.json
    python robot/teleop/wholebody_teleop.py --input oculus --target robot --no-pose-filter
    ```

    Shutdown restores stock automatically, since the flashed manifest is not
    the stock one.

    **What the clamp actually costs and buys.** SPARK position error is in
    *rotations*, so:

    | error | commissioned p=20, +/-0.25 | stock p=2.0, +/-1.0 |
    |---|---:|---:|
    | 0.5 deg | 0.028 | 0.003 |
    | 4.5 deg | 0.250 (clamps) | 0.025 |
    | 45 deg | 0.250 | 0.250 |
    | 90 deg | 0.250 | **0.500** |

    They cross at 45 deg. Stock is stronger only above that, and only by 2x;
    below it commissioned is stronger, by 10x at 4.5 deg. The measured
    consequence with stock loaded (2026-08-22 runs): median steering error
    **11-19 deg**, p90 57-69, 34-49% of moving ticks beyond 20 deg -- against
    the commissioned set's own measured 0.168 deg stall, which is the
    encoder's 2-LSB floor. A module 15 deg off its commanded angle drives the
    chassis 15 deg off course.

    **The clamp is also no longer the binding constraint it was.** Modules
    slew at 265-353 deg/s under +/-0.25, and `base_heading_rate_limit` now
    caps demand at 200 deg/s -- 1.3-1.8x headroom. If steering feels slow on
    this manifest, the ceiling is more likely that limiter than the clamp.
    Raising `base_heading_rate_limit` toward 260 is the cheaper experiment,
    and it is a config value rather than a flash.

    New guard: `test_every_shipped_manifest_is_valid` walks
    `config/base_pid_*.json` and checks each validates, names the right
    controllers, plans all eight, and declares its own
    `drive_command_scale` -- a manifest without one silently inherits the
    module default, which is right for exactly one gain set. 115/115.

29. **Drive position counts added to the trajectory log** (2026-08-24).
    `drive_pos_{FL,FR,RR,RL}` -- cumulative motor rotations from
    `GetPosition()`, 90 -> 94 columns.

    The log already carried both other encoder streams (`steer_meas_*` from
    the steering absolute encoder, `drive_meas_*` from `GetVelocity`), but not
    drive *position* -- which is the one that matters for distance. A counter
    does not accumulate the sampling error that integrating a 30 Hz-sampled
    velocity does, and it is exactly what
    `robot/nav/odometry/swerve_odom.py` integrates. Costs no extra bus
    traffic: `GetPosition` reads the same cached status frame 2 as
    `GetVelocity`.

    What it makes answerable offline, from an ordinary run, with no tape:

    * wheel odometry reconstructed per tick and compared against
      `BaseOdometry`'s commanded-velocity dead reckoning -- the ~25%
      over-report from item 25, measured continuously instead of once;
    * the `METERS_PER_ROTATION` discrepancy from item 25 (0.049922 in
      swerve_odom vs 0.047531 from PHASE0), by fitting rotations against SLAM
      or against a known course;
    * per-module *distance* spread, which is a cleaner read on RL's 5%
      shortfall than comparing velocity medians -- distance integrates the
      whole run rather than sampling it.

    Steering position counts were deliberately left out: the absolute encoder
    already gives the angle, so they would be redundant.

    Suites: `test_base_kinematics` 84/84, `test_wholebody_control` 59/59,
    `test_base_pid_preflight` 115/115, rest unchanged.

30. **Swerve telemetry now recorded independently of whole-body control**
    (2026-08-24). New `robot/swerve_log.py`, written by `yor.py` for the life
    of the base control loop.

    The trajectory CSV only exists when a `WholeBodyController` does, so
    `yor.py --no-arms` -- which is how the base gets driven from joystick.py --
    recorded nothing at all. And even with arms it sampled at the 30 Hz solve
    rate against a 50 Hz status stream. This runs off the base loop instead,
    so a joystick run and a teleop run are directly comparable.

    ```
    artifacts/wholebody_logs/swerve/swerve_<timestamp>.csv     28 columns @ 50 Hz
      t, motors_enabled, v_target_{0,1,2}, v_prof_{0,1,2},
      steer_cmd_*, steer_meas_*, drive_cmd_*, drive_meas_*, drive_pos_*
    ```

    Config row stamps the PID provenance and the sample rate. Default 50 Hz,
    matching the SPARK periodic status 2 period -- sampling faster than the
    controllers publish would only duplicate rows. `--no-swerve-log` disables,
    `--swerve-log-hz` overrides.

    Started right after `base.start_control()` in `init()` (so it covers
    startup homing too) and stopped in `graceful_shutdown` *before* the base
    loop goes down. Never raises: a telemetry failure writes no row rather
    than a wrong one, and the thread keeps going -- verified in the tests by
    failing the fake bus mid-run and checking sampling resumes.

    It duplicates some trajectory-log columns on purpose. That one exists to
    correlate the wheels with *solver* output on the same tick; this one
    exists to look at the wheels on their own, at the rate they actually
    report.

    Both logs are written when whole-body is running. Suites:
    `test_base_kinematics` 98/98 (+14), rest unchanged.

31. **Why the base wanders on hardware but not in sim** (2026-08-24). Run
    `traj_20260824_195659`, 199 s, hybrid gains, 33.8% of ticks commanding the
    base.

    **The command alternates direction roughly every other tick.** Tick-to-tick
    change in the commanded base heading is purely bimodal:

    ```
      0- 10 deg : 1027  (57.4%)
     10-170 deg :    0  ( 0.0%)
    170-180 deg :  763  (42.6%)   <- outright reversals
    ```

    764 forward sign reversals and 820 lateral over 1922 commanded ticks --
    one every **0.08 s**, with a median burst length of 0.03 s.

    **The hardware base responds with 167 ms of lag** (peak cross-correlation
    0.904 at 5 ticks between commanded speed and measured wheel speed).

    167 ms of lag against an 80 ms reversal period. The chassis is always
    accelerating toward a direction that has already flipped, so it never
    converges -- it shuttles. Measured: **6.807 m of odometry path for 0.300 m
    of net displacement, 22.7x.**

    **In sim there is no lag at all.** `yor_mujoco.py` calls
    `ik.apply_to_sim_kinematic(self.data, result)`, which writes the solve
    straight into `data.qpos`; `_animate_swerve` only spins the wheel meshes.
    The base pose is whatever the solver decided, instantly and exactly, so the
    same alternating command stream produces a base that sits still (the flips
    cancel) and the solver looks fine. Same solver, same output, opposite
    outcome -- the difference is entirely the plant.

    **The heading rate limiter from item 23 does not catch this, by design.**
    It measures a >90 deg heading change against the *reversed* previous
    heading, because a swerve module serves a reversal by flipping the drive
    rather than turning. That is right for the module and wrong for the
    chassis, which has to decelerate, stop and accelerate the other way. All
    42.6% of the reversals pass through unlimited.

    Worse, the replay validation in item 23 measured **module travel folded for
    reversal** -- precisely the metric that hides this. The replay was correct
    about what it measured (p90 854 -> 207 deg/s, uncatchable ticks 33.2% ->
    0.5%) and blind to what actually destabilises the base.

    **Also correcting item 31's first pass:** an initial table here reported
    "wheels travelled 3.7-12.6 m" from `sum|diff|` of `drive_pos`. That was
    jitter-inflated -- range/sum ran as low as 0.064, i.e. 94% of the sum was
    sampling noise. Position differencing needs the 50 Hz swerve log and a
    velocity cross-check, not the 30 Hz trajectory log.

    **Candidate fixes, in order of principle:**
    1. Penalise base *velocity change* in the solver. This is the same
       bimodality the arms had with elbow flips, and the cure there was the
       null-space continuity term -- `nullspace_continuity_weight` already
       exists and is 0.0.
    2. Slew-limit the base velocity **vector**, reversals included, sized to
       the measured 167 ms response: a full reversal should take at least
       ~2x that.
    3. Low-pass the base velocity vector before dispatch.

    (2) and (3) are dispatch-side patches on a solver that is producing a
    physically unrealisable command; (1) fixes the command.

32. **Both base-stability fixes implemented; the first one had to be
    rewritten after measurement** (2026-08-24).

    **(2) Chassis acceleration limit — worked as designed.**
    `WholeBodyHardwareConfig.base_max_accel = 1.5 m/s^2`, applied in
    `_dispatch_base` after the heading limiter. An exact zero is exempt, so a
    deadbanded command or a halt still stops the base immediately, and
    `_halt_base` resets the reference so resuming is not limited from a
    velocity the chassis no longer has.

    Replayed over the real runs:

    | run | reversals >90 deg before | after | commanded path before | after |
    |---|---:|---:|---:|---:|
    | 195538 | 61.7% | **4.4%** | 3.52 m | 2.57 m |
    | 195659 | 42.6% | **10.4%** | 10.99 m | 8.35 m |

    **(1) Solver-side continuity — the first two attempts did not work, and
    finding out why is the actual result.**

    *Attempt A, carry the previous base velocity into the primary solve.* The
    carry referenced `self._prev_vel`, which `solve()` rewrites **every
    iteration**, so after a converged solve it holds the last and smallest
    refinement step rather than anything the chassis was asked for. Fixed by
    adding `_prev_base_vel` (the tick-level value actually dispatched) and
    applying the carry on the first iteration only -- the reported base
    velocity is a sum over iterations, so applying it every iteration would
    multiply it by the iteration count. Effect after the fix: real but tiny.
    In a regime built to reproduce the failure (target the arms can reach,
    2 mm of EE noise), reversals went 58.8% -> 57.8% at carry 0.95.

    *Attempt B, weight base DOFs in the primary damped inverse.* Also nearly
    inert: base response to pure noise fell only from 0.0923 to 0.0848 at a
    **1000x** weight.

    *Why.* The base has two independent routes to the same motion. Blocking
    the primary path leaves the null-space objectives (posture, swivel)
    supplying it; switching those off leaves the primary supplying it. Only
    both at once collapse it -- 0.0923 -> 0.0002.

    *The fix that works:* weight base DOFs in **both** the primary solve and
    the null-space secondary. The second half is the principled one -- among
    all solutions serving the EE equally, prefer the one that moves the base
    least, which `N z` can express for free because it cannot change the EE
    task. `base_motion_weight = 100.0`.

    Measured, against pure 2 mm EE noise, fraction of ticks whose base command
    clears the 0.04 dispatch deadband:

    | weight | median \|v\| | past deadband | reversals |
    |---|---:|---:|---:|
    | 1 | 0.0554 | 71.3% | 77.9% |
    | 30 | 0.0287 | 26.7% | 82.4% |
    | **100** | **0.0138** | **0.7%** | **0.0%** |

    And it is a preference, not a prohibition: a target 0.60 m beyond arm
    reach still rolls the base 0.31 m and converges to 0.00 mm, where the
    arms alone leave 359 mm.

    Root cause worth recording: unweighted, the base answered **24%** of a
    pure 2 mm EE noise input, because the damped inverse treats every free
    DOF as equally cheap and the chassis has the most leverage per unit of
    joint motion. "Base motion is emergent" was aspirational, not true.

    New CLI: `--base-motion-weight`, `--base-velocity-continuity`,
    `--base-max-accel`. Suites all green.

33. **Base cost gated on arm manipulability** (2026-08-24). The operator's
    report: "it moved forward but only after awkwardly putting the arms
    forward, and when I moved back it just moved the lift up and the arms
    back."

    **Confirmed in the data before changing anything.** Reconstructing
    manipulability from three runs (1757 samples), base motion by mu band:

    | worst-arm mu | base motion |
    |---|---:|
    | > 0.050 (healthy) | 0.00001 m/tick |
    | 0.045-0.050 | 0.00154 |
    | 0.020-0.030 | 0.00192 |
    | < 0.020 | 0.00198 |

    mu is 0.0506 at the home keyframe, so the chassis was doing essentially
    nothing until the arms had already dropped below their home posture --
    and mu's 10th percentile across those runs was 0.0009, i.e. the arms were
    genuinely reaching singularity. The base was a last resort *after* the
    posture degraded, when it should move *so that* it does not.

    **Cause was item 32's own fix.** `base_motion_weight = 100` is the right
    magnitude for rejecting tracker noise and the wrong shape for recruiting
    the base: a flat cost cannot tell "the arms are fine, this is noise" from
    "the arms are running out".

    **Fix: gate the cost on manipulability.** `_gated_base_weight()` ramps the
    base cost from `base_motion_weight` (100) at mu >= `base_weight_gate_on`
    (0.045) down to `base_motion_weight_min` (1.0) at
    `base_weight_gate_full` (0.025), by smoothstep. Worst arm governs -- one
    arm at a singularity is reason enough to move the chassis, and averaging
    would let a comfortable arm hide it.

    Measured, pushing a target 0.60 m out:

    | | min mu | median mu | noise past deadband |
    |---|---:|---:|---:|
    | flat weight 100 | 0.000063 | 0.01604 | 0.7% |
    | gated (shipped) | **0.012558** | **0.02701** | **0.7%** |

    The arms no longer reach singularity -- min mu improves 200x -- and
    **noise rejection is completely unchanged**, because tracker noise arrives
    while the arms are still comfortable and the gate is shut. That is the
    property that makes this work rather than just trading one problem for
    the other.

    **Affordable because only the value is needed, not the gradient.** Two
    Jacobians and two 6x7 SVDs per solve, cached across the iterations of one
    solve since posture barely moves within a tick. Measured 2.03 ms per solve
    against a 33.3 ms budget -- against the 28 perturbed kinematics
    evaluations `_manipulability_gradient` costs, which is why
    `enable_manipulability` is still off.

    New CLI: `--base-motion-weight-min`, `--base-weight-gate ON FULL`. Setting
    the two weights equal disables the gate. A more aggressive band
    (`--base-weight-gate 0.050 0.035`) held min mu at 0.0182 in the sweep, at
    the cost of the gate being partly open almost always -- worth trying if
    the arms still look stretched on hardware.

34. **`tools/replay_solver.py` — A/B solver configurations on one recorded
    trajectory** (2026-08-24).

    Every hardware run so far was judged against a different operator input,
    which makes "does this balance base, lift and arms better?" unanswerable:
    the input changed at the same time as the solver did.

    **The recordings already existed.** `_TrajectoryRecorder` writes
    `left_target_ee_*` / `right_target_ee_*` every tick, and those columns
    *are* the solver's input -- the same SE3 pair `solve()` is called with.
    Verified on traj_20260824_213002: the target moves on 94% of ticks with a
    median 4.95 mm step, and `lift_goal == lift_q` on 100% of ticks, so the
    operator never set a lift target and `lift_target=None` is faithful.
    Every file in artifacts/wholebody_logs/trajectories/ is replayable.

    Caveat stated in the tool: teleop generates each target relative to where
    the hand currently is, so replaying an absolute sequence under a different
    configuration is not a perfect re-enactment. It is a fair A/B -- identical
    requested hand trajectory, different whole-body resolution.

    ```
    python tools/replay_solver.py --log <traj>.csv                 # compare presets
    python tools/replay_solver.py --log <traj>.csv --view eager    # watch one
    python tools/replay_solver.py --log <traj>.csv --set base_motion_weight=50
    ```

    **First result, on traj_20260824_213002 (897 ticks):**

    | config | EE med | EE p95 | mu min | mu med | base path | base yaw | lift | revers |
    |---|---:|---:|---:|---:|---:|---:|---:|---:|
    | shipped | 0.19 | 1.81 | 0.0241 | 0.0356 | 1.74 m | 83d | 1.36 m | 1.2% |
    | flat (item 32) | 0.37 | 5.70 | 0.0125 | 0.0300 | 1.65 m | 61d | **1.63 m** | 1.0% |
    | unweighted | 0.07 | 5.02 | 0.0207 | 0.0406 | 2.89 m | 252d | 1.00 m | 3.0% |
    | **eager (0.050->0.035)** | **0.12** | **1.39** | **0.0263** | **0.0419** | 1.92 m | 108d | **1.25 m** | 2.6% |
    | reluctant | 0.28 | 3.82 | 0.0203 | 0.0318 | 1.63 m | 66d | 1.47 m | 1.3% |

    **`eager` beats `shipped` on every column that matters** -- lower EE error
    at both median and p95, better worst-case and typical arm posture, and
    less lift travel -- by letting the chassis commit sooner. And the table
    shows the operator's complaint directly: **lift travel moves inversely
    with base willingness** (flat 1.63 m at 61 deg of yaw, eager 1.25 m at
    108 deg). The lift was substituting for a base that would not move.

    **Gate sweep is non-monotonic, so it has to be measured, not interpolated:**

    ```
    0.045->0.025   EE p95  1.81   mu_min 0.0241   base 1.74m
    0.050->0.035   EE p95  1.39   mu_min 0.0263   base 1.92m
    0.055->0.040   EE p95 31.45   mu_min 0.0000   base 5.00m   <- unstable
    0.060->0.045   EE p95  0.89   mu_min 0.0245   base 2.75m
    ```

    0.055->0.040 loses tracking outright (mu hits 0 -- a singularity) and
    should not be used. 0.060 sits *above* the home mu of 0.0506, so the gate
    is fully open always and it degenerates to `unweighted`; its good numbers
    on this trace do not carry, because a recorded trace cannot test noise
    rejection -- the operator's input is already in it. That property was
    measured separately (synthetic 2 mm jitter, item 33).

    Recommendation: `--base-weight-gate 0.050 0.035` on the next run. Margin
    to the home posture is thin (0.0506 vs 0.050), which is the aggressive
    part of the choice; the sweep says it pays.
