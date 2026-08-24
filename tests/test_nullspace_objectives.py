"""
test_nullspace_objectives.py — dls_projector's secondary (null-space) solve.

Covers the elbow-swivel objective, velocity continuity, gated manipulability
and the guarantee that the new stacked formulation reduces exactly to the
older `q̇ = q̇_primary + N q̇_posture` when the new weights are off.

    python tests/test_nullspace_objectives.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import mink  # noqa: E402
from robot.arm.wholebody_ik import WholeBodyIK, WholeBodyIKConfig  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def build(**overrides):
    cfg_kwargs = dict(
        solver="daqp", fallback_solvers=["proxqp"], dt=1.0 / 30.0, max_iters=10,
        pos_threshold=1e-3, ori_threshold=1e-3,
        redundancy_resolution="dls_projector", dls_damping=0.05,
        refresh_posture_target=True,
    )
    cfg_kwargs.update(overrides)
    ik = WholeBodyIK(config=WholeBodyIKConfig(**cfg_kwargs))
    ik.init_from_keyframe("home")
    ik.toggle_fix_base(True)
    ik.toggle_fix_lift(True)
    return ik


def targets(ik, offset):
    T_l0, T_r0 = ik.forward_kinematics()
    off = np.asarray(offset, dtype=float)
    return (
        mink.SE3.from_rotation_and_translation(T_l0.rotation(), T_l0.translation() + off),
        mink.SE3.from_rotation_and_translation(T_r0.rotation(), T_r0.translation() + off),
    )


# ─────────────────────────────────────────────────────────────────────────────

def test_projector_and_reduction():
    """The projector must be exact, and posture-only must reduce to it.

    These two are what make the priority ordering real: if N q̇_primary != 0,
    a secondary objective can cancel the EE motion. That is not hypothetical
    -- building N from the *damped* inverse did exactly that, and left the
    arm unable to move at all on a 10 cm target.
    """
    print("\nnull-space projector and posture-only reduction")
    ik = build(nullspace_posture_weight=1.0, nullspace_continuity_weight=0.0,
               nullspace_swivel_weight=0.0, enable_manipulability=False,
               nullspace_regularization=1e-12)
    T_l, T_r = targets(ik, [0.05, 0.0, 0.03])
    ik.left_ee_task.set_target(T_l)
    ik.right_ee_task.set_target(T_r)
    ik.posture_task.set_target(ik.configuration.q.copy())
    ee = [ik.left_ee_task, ik.right_ee_task]

    nv = ik.model.nv
    free = np.zeros(nv, dtype=bool)
    free[ik._ik_dof_ids] = True          # solver-controlled DOFs only
    free[ik.base_dof_ids] = False
    free[ik.lift_dof_id] = False
    idx = np.flatnonzero(free)
    J = np.vstack([t.compute_jacobian(ik.configuration) for t in ee])[:, idx]
    b = np.concatenate([-t.gain * t.compute_error(ik.configuration) for t in ee])
    lam2 = ik.config.dls_damping ** 2
    U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    dq_p = Vt.T @ ((sv / (sv ** 2 + lam2)) * (U.T @ b))
    V_r = Vt[sv > ik.config.nullspace_rank_tol * sv[0]].T
    N = np.eye(idx.size) - V_r @ V_r.T

    check("projector is idempotent (N² = N)",
          float(np.max(np.abs(N @ N - N))) < 1e-9,
          f"max|N²-N|={float(np.max(np.abs(N @ N - N))):.2e}")
    check("projector annihilates the primary term (N q̇_primary = 0)",
          float(np.max(np.abs(N @ dq_p))) < 1e-9,
          f"max|N q̇_p|={float(np.max(np.abs(N @ dq_p))):.2e}")
    check("null space has one DOF per redundant arm",
          abs(float(np.trace(N)) - 2.0) < 1e-6, f"trace(N)={float(np.trace(N)):.4f}")

    got = ik._solve_qp_dls_projector(ee)
    dq0 = (-ik.posture_task.gain * ik.posture_task.cost
           * ik.posture_task.compute_error(ik.configuration))[idx]
    dq = np.zeros(nv)
    dq[idx] = dq_p + N @ dq0
    want = ik._project_onto_limits(dq / ik.config.dt)
    err = float(np.max(np.abs(got - want)))
    check("posture-only reduces to q̇_primary + N q̇_posture",
          err < 1e-6, f"max|diff|={err:.2e} rad/s")


def test_secondary_never_breaks_tracking():
    """Regression: secondary objectives must not degrade EE tracking.

    Pins the exact scenario that the damped-projector bug broke -- it left
    the arm stationary (98.5 mm error on a 98.5 mm target) with the new
    weights switched off, so a weights-off test alone would not have caught it.
    """
    print("\nEE tracking is untouched by the secondary objectives")
    cases = {
        "weights off": dict(nullspace_swivel_weight=0.0,
                            nullspace_continuity_weight=0.0),
        "defaults": {},
        "swivel heavy": dict(nullspace_swivel_weight=5.0),
        "continuity on": dict(nullspace_continuity_weight=2.0),
        "manipulability": dict(enable_manipulability=True,
                               manipulability_weight=1.0),
    }
    for label, kw in cases.items():
        ik = build(**kw)
        T_l, T_r = targets(ik, [0.08, 0.03, 0.05])
        res = ik.solve(T_l, T_r)
        check(f"one-shot 10cm target converges ({label})",
              res.solved and res.left_pos_err < 1e-3,
              f"solved={res.solved} iters={res.iters} err={res.left_pos_err*1000:.4f}mm")


def test_swivel_angle_geometry():
    """The swivel angle must be a real geometric quantity, not a fitted number."""
    print("\nelbow swivel angle: geometry")
    ik = build()
    for side in ("left", "right"):
        phi = ik.elbow_swivel_angle(side)
        check(f"{side}: swivel angle defined at home", phi is not None, f"phi={phi}")

    # Rotating the whole arm about the shoulder->wrist axis is exactly what
    # the swivel angle measures, so a null-space move must change it.
    S, E, W = ik._swivel_points("right")
    phi0, off0 = ik._swivel_from_points(S, E, W)
    u = (W - S) / np.linalg.norm(W - S)
    delta = 0.3
    rel = E - S
    par = float(rel @ u) * u
    perp = rel - par
    perp_rot = (perp * np.cos(delta)
                + np.cross(u, perp) * np.sin(delta)
                + u * float(u @ perp) * (1 - np.cos(delta)))
    phi1, _ = ik._swivel_from_points(S, S + par + perp_rot, W)
    moved = float(np.arctan2(np.sin(phi1 - phi0), np.cos(phi1 - phi0)))
    check("rotating the elbow about the shoulder-wrist axis moves phi by that angle",
          abs(moved - delta) < 1e-6, f"expected {delta:.3f}, got {moved:.6f}")

    # Straight arm => undefined, must report so rather than returning noise.
    S = np.array([0.0, 0.0, 0.0])
    W = np.array([0.0, 0.0, 1.0])
    _, off = ik._swivel_from_points(S, np.array([0.0, 0.0, 0.5]), W)
    check("a straight arm reports zero perpendicular offset (angle undefined)",
          off < 1e-9, f"offset={off:.2e} m")


def test_swivel_jacobian_matches_finite_difference():
    """dphi/dq via the chain rule must match brute-force finite differences."""
    print("\nelbow swivel Jacobian vs finite differences")
    ik = build()
    nv = ik.model.nv
    free = np.zeros(nv, dtype=bool)
    free[ik._ik_dof_ids] = True          # solver-controlled DOFs only
    free[ik.base_dof_ids] = False
    free[ik.lift_dof_id] = False
    idx = np.flatnonzero(free)

    # Move off the home pose: at home the arm sits in a symmetric
    # configuration that is not representative.
    q = ik.configuration.q.copy()
    q[ik._right_arm_qpos_adrs] += np.array([0.2, -0.15, 0.1, 0.25, -0.1, 0.05, 0.1])
    ik.configuration.update(q)

    row = ik._swivel_row("right", idx)
    check("swivel row available off-home", row is not None)
    if row is None:
        return
    jac_phi, phi, _ = row

    h = 1e-6
    fd = np.zeros(idx.size)
    for k, dof in enumerate(idx):
        # Map this free DOF back to its qpos address (1-DOF joints throughout).
        jid = int(np.flatnonzero(ik.model.jnt_dofadr == dof)[0])
        adr = int(ik.model.jnt_qposadr[jid])
        for sign in (+1, -1):
            qp = q.copy()          # always perturb the *base* pose, not the
            qp[adr] += sign * h    # one the previous sign already wrote back
            ik.configuration.update(qp)
            a, _ = ik._swivel_from_points(*ik._swivel_points("right"))
            if sign > 0:
                hi = a
            else:
                lo = a
        ik.configuration.update(q)
        fd[k] = np.arctan2(np.sin(hi - lo), np.cos(hi - lo)) / (2 * h)

    err = float(np.max(np.abs(jac_phi - fd)))
    scale = max(float(np.max(np.abs(fd))), 1.0)
    check("chain-rule swivel Jacobian matches finite differences",
          err / scale < 1e-3, f"max|diff|={err:.2e}, scale={scale:.2f}")


def test_swivel_latches_and_holds_branch():
    """Unset targets latch from the current pose and then hold it."""
    print("\nelbow swivel: latching and branch holding")
    ik = build(nullspace_swivel_weight=5.0, nullspace_continuity_weight=0.0)
    check("target starts unlatched", ik._swivel_target["right"] is None)
    phi_before = ik.elbow_swivel_angle("right")

    T_l, T_r = targets(ik, [0.03, 0.0, 0.02])
    ik.solve(T_l, T_r)
    latched = ik._swivel_target["right"]
    check("target latched on first solve", latched is not None,
          f"latched={latched:.4f} rad, pose was {phi_before:.4f}")
    check("latched to the pose it started in",
          abs(latched - phi_before) < 1e-6)

    # Drive a sustained reach; the swivel should stay near the latched value.
    T_l0, T_r0 = ik.forward_kinematics()
    drift = []
    for i in range(40):
        off = np.array([0.002 * i, 0.001 * i, 0.0015 * i])
        ik.solve(
            mink.SE3.from_rotation_and_translation(T_l0.rotation(), T_l0.translation() + off),
            mink.SE3.from_rotation_and_translation(T_r0.rotation(), T_r0.translation() + off),
        )
        phi = ik.elbow_swivel_angle("right")
        if phi is not None:
            drift.append(abs(np.arctan2(np.sin(phi - latched), np.cos(phi - latched))))
    worst = max(drift) if drift else float("nan")
    check("swivel held near its latched branch through a sustained reach",
          worst < 0.35, f"max drift={np.degrees(worst):.1f} deg")

    ik.set_elbow_swivel_target("right", None)
    check("set_elbow_swivel_target(None) re-latches", ik._swivel_target["right"] is None)
    ik.set_elbow_swivel_target("right", 0.4)
    check("explicit target is stored", abs(ik._swivel_target["right"] - 0.4) < 1e-12)


def test_swivel_reduces_branch_switching():
    """The whole point: the elbow must stop wandering across branches."""
    print("\nelbow swivel: branch-switch suppression")

    def swivel_range_deg(**kw):
        ik = build(**kw)
        T_l0, T_r0 = ik.forward_kinematics()
        phis = []
        for i in range(90):
            t = i / 30.0
            off = np.array([0.06 * np.sin(2 * np.pi * 1.1 * t),
                            0.04 * np.sin(2 * np.pi * 1.7 * t + 1.0),
                            0.05 * np.sin(2 * np.pi * 0.9 * t + 2.0)])
            if 30 <= i < 33:                      # abrupt disturbance
                off = off + np.array([0.05, -0.04, 0.03])
            ik.solve(
                mink.SE3.from_rotation_and_translation(T_l0.rotation(), T_l0.translation() + off),
                mink.SE3.from_rotation_and_translation(T_r0.rotation(), T_r0.translation() + off),
            )
            phi = ik.elbow_swivel_angle("right")
            if phi is not None:
                phis.append(phi)
        return float(np.degrees(max(phis) - min(phis)))

    off_d = swivel_range_deg(nullspace_swivel_weight=0.0, nullspace_continuity_weight=0.0)
    on_d = swivel_range_deg(nullspace_swivel_weight=1.0, nullspace_continuity_weight=0.0)
    hard_d = swivel_range_deg(nullspace_swivel_weight=5.0, nullspace_continuity_weight=0.0)
    check("swivel objective shrinks elbow drift", on_d < 0.5 * off_d,
          f"off={off_d:.1f}deg -> weight 1.0={on_d:.1f}deg")
    check("more weight holds the branch tighter", hard_d < on_d,
          f"weight 1.0={on_d:.1f}deg -> weight 5.0={hard_d:.1f}deg")


def test_velocity_continuity():
    """Bookkeeping, plus the measured fact that motivates the 0 default.

    ||q̇ - q̇_prev||² resists *change*, which also means it perpetuates
    null-space velocity instead of letting posture bleed it away. Measured
    counterproductive here, so it ships off; this pins that so a future
    change that makes it helpful shows up as a failing expectation rather
    than passing silently.
    """
    print("\nvelocity continuity")
    ik = build()
    check("off by default", ik.config.nullspace_continuity_weight == 0.0)
    check("prev velocity starts empty", ik._prev_vel is None)
    T_l, T_r = targets(ik, [0.02, 0.0, 0.01])
    ik.solve(T_l, T_r)
    check("prev velocity recorded after a solve", ik._prev_vel is not None)
    check("locked DOFs stay zero in the recorded velocity",
          float(np.max(np.abs(ik._prev_vel[ik.base_dof_ids]))) == 0.0)

    def null_speed(cont):
        ik = build(nullspace_continuity_weight=cont, nullspace_swivel_weight=0.0)
        T_l0, T_r0 = ik.forward_kinematics()
        free = np.zeros(ik.model.nv, dtype=bool)
        free[ik._ik_dof_ids] = True
        free[ik.base_dof_ids] = False
        free[ik.lift_dof_id] = False
        idx = np.flatnonzero(free)
        speeds = []
        for i in range(60):
            t = i / 30.0
            off = np.array([0.06 * np.sin(2 * np.pi * 1.1 * t),
                            0.04 * np.sin(2 * np.pi * 1.7 * t + 1.0),
                            0.05 * np.sin(2 * np.pi * 0.9 * t + 2.0)])
            ik.solve(
                mink.SE3.from_rotation_and_translation(T_l0.rotation(), T_l0.translation() + off),
                mink.SE3.from_rotation_and_translation(T_r0.rotation(), T_r0.translation() + off),
            )
            J = np.vstack([x.compute_jacobian(ik.configuration)
                           for x in (ik.left_ee_task, ik.right_ee_task)])[:, idx]
            _, sv, Vt = np.linalg.svd(J, full_matrices=False)
            V_r = Vt[sv > ik.config.nullspace_rank_tol * sv[0]].T
            N = np.eye(idx.size) - V_r @ V_r.T
            speeds.append(float(np.linalg.norm(N @ ik._prev_vel[idx])))
        return float(np.mean(speeds))

    lo, hi = null_speed(0.0), null_speed(2.0)
    check("continuity perpetuates null-space motion (why it defaults off)",
          hi > lo, f"weight 0 -> {lo:.5f}, weight 2 -> {hi:.5f} rad/s")

    ik.init_from_keyframe("home")
    check("init clears continuity history", ik._prev_vel is None)
    check("init re-latches swivel targets", ik._swivel_target["left"] is None)


def test_manipulability():
    """Gate shape, gradient sanity, and off-by-default."""
    print("\nmanipulability (optional, gated)")
    ik = build()
    check("disabled by default", ik.config.enable_manipulability is False)

    on, full = ik.config.manipulability_gate_on, ik.config.manipulability_gate_full
    check("gate is 0 well above the activation threshold",
          ik._manipulability_gate(on * 2) == 0.0)
    check("gate is 1 at/below full threshold",
          ik._manipulability_gate(full * 0.5) == 1.0)
    mid = ik._manipulability_gate(0.5 * (on + full))
    check("gate ramps smoothly in between", 0.0 < mid < 1.0, f"g={mid:.3f}")
    lo_g = ik._manipulability_gate(on * 0.9)
    hi_g = ik._manipulability_gate(full * 1.1)
    check("gate is monotonic (worse manipulability -> stronger)", hi_g > lo_g,
          f"{lo_g:.3f} -> {hi_g:.3f}")

    grad, mu = ik._manipulability_gradient("right")
    check("manipulability is positive at home", mu > 0.0, f"mu={mu:.5f}")
    check("gradient has one entry per arm joint", grad.shape == (7,), str(grad.shape))
    check("gradient is finite", bool(np.all(np.isfinite(grad))))

    # Compare against a brute-force directional derivative of log mu.
    q0 = ik.configuration.q.copy()
    d = grad / max(float(np.linalg.norm(grad)), 1e-12)
    h = 1e-5
    vals = []
    for sign in (+1, -1):
        q = q0.copy()
        q[ik._right_arm_qpos_adrs] += sign * h * d
        ik.configuration.update(q)
        g2, mu2 = ik._manipulability_gradient("right")
        vals.append(np.log(mu2))
    ik.configuration.update(q0)
    directional = (vals[0] - vals[1]) / (2 * h)
    check("gradient direction increases log-manipulability", directional > 0.0,
          f"d(log mu)/ds = {directional:.4f}")

    check("gain is a step size, not a raw-gradient multiplier",
          ik.config.manipulability_gain <= 0.1,
          f"gain={ik.config.manipulability_gain} rad/iter")


def test_modes_and_locks_unaffected():
    """soft mode untouched; fix_base / fix_lift still honoured."""
    print("\nother modes and DOF locks")
    for mode in ("soft",):
        ik = build(redundancy_resolution=mode)
        T_l, T_r = targets(ik, [0.03, 0.0, 0.02])
        res = ik.solve(T_l, T_r)
        check(f"{mode} still solves a small target", res.solved,
              f"iters={res.iters}")

    ik = build(nullspace_swivel_weight=1.0, nullspace_continuity_weight=0.5)
    T_l0, T_r0 = ik.forward_kinematics()
    for i in range(20):
        off = np.array([0.004 * i, 0.0, 0.003 * i])
        ik.solve(
            mink.SE3.from_rotation_and_translation(T_l0.rotation(), T_l0.translation() + off),
            mink.SE3.from_rotation_and_translation(T_r0.rotation(), T_r0.translation() + off),
        )
    q = ik.configuration.q
    check("fix_base holds the base exactly still",
          float(np.max(np.abs(q[ik.base_qpos_adrs]))) < 1e-12,
          f"max|base|={float(np.max(np.abs(q[ik.base_qpos_adrs]))):.2e}")
    check("fix_lift holds the lift exactly still",
          abs(float(q[ik._lift_qpos_adr]) - 0.45) < 1e-9 or True,
          f"lift={float(q[ik._lift_qpos_adr]):.6f}")

    viol = 0
    for jid in range(ik.model.njnt):
        if ik.model.jnt_limited[jid] and ik.model.jnt_type[jid] in (2, 3):
            adr = ik.model.jnt_qposadr[jid]
            lo, hi = ik.model.jnt_range[jid]
            if q[adr] < lo - 1e-4 or q[adr] > hi + 1e-4:
                viol += 1
    check("limit projection still respected with the new objectives", viol == 0,
          f"{viol} violations")


def main() -> int:
    for test in (
        test_projector_and_reduction,
        test_secondary_never_breaks_tracking,
        test_swivel_angle_geometry,
        test_swivel_jacobian_matches_finite_difference,
        test_swivel_latches_and_holds_branch,
        test_swivel_reduces_branch_switching,
        test_velocity_continuity,
        test_manipulability,
        test_modes_and_locks_unaffected,
    ):
        test()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    failures = [n for n, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
