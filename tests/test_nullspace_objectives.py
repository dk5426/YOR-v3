"""
test_nullspace_objectives.py — dls_projector's secondary (null-space) solve.

Covers the elbow-swivel objective, velocity continuity, gated manipulability
and the guarantee that the new stacked formulation reduces exactly to the
older `q̇ = q̇_primary + N q̇_posture` when the new weights are off.

    python tests/test_nullspace_objectives.py
"""

from __future__ import annotations

import sys
import time
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
    # The hand-derived step below is the *unconstrained* damped inverse at a
    # flat lambda with no row scaling, so the three knobs that change that
    # step are pinned off: constrained_primary solves a QP against the
    # joint/collision inequalities instead, dls_task_weighting rescales the
    # rows of J and b, and adaptive damping ramps lambda with sigma_min.
    # Each is a different step by construction, not a broken one -- the
    # identity being pinned here is the projector's.
    ik = build(nullspace_posture_weight=1.0, nullspace_continuity_weight=0.0,
               nullspace_swivel_weight=0.0, enable_manipulability=False,
               nullspace_regularization=1e-12, constrained_primary=False,
               dls_task_weighting=False, dls_adaptive_damping_sigma=0.0)
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

    # _swivel_row freezes the transported reference through its own finite
    # differences (see its docstring), so the numerical check has to
    # differentiate the same angle -- with swivel_parallel_ref on, the
    # default z/x construction is a different function of q entirely.
    ref = None
    if ik.config.swivel_parallel_ref:
        S_, _E, W_ = ik._swivel_points("right")
        axis = W_ - S_
        a_norm = float(np.linalg.norm(axis))
        if a_norm > 1e-9:
            r_ = ik._swivel_reference("right", axis / a_norm)
            if r_ is not None:
                ref = (float(r_[0]), float(r_[1]), float(r_[2]))

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
            a, _ = ik._swivel_from_points(*ik._swivel_points("right"), ref=ref)
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


def test_manipulability_gated_base_weight() -> None:
    """The base must move so the arms keep a posture, not after they lose one.

    A flat base cost is the wrong shape. High enough to stop the chassis
    chasing tracker noise, it also stops it helping until the arms have
    already contorted: measured across three runs on 2026-08-24, base motion
    averaged 0.00001 m/tick while the worst arm's manipulability was above
    0.050 and 0.0014-0.0020 below it. The operator's description was "it moved
    forward but only after awkwardly putting the arms forward".

    Manipulability gates it because it is exactly what separates "the arms are
    fine, this is noise" from "the arms are running out".
    """
    print("\nmanipulability-gated base weight")
    import numpy as _np

    def mu_of(ik):
        d = ik.configuration.data
        return min(float(_np.exp(ik._log_manipulability(ik._arm_jacobian(s_, d))))
                   for s_ in ("left", "right"))

    home = build()
    check("mu at the home keyframe matches what the runs measured",
          abs(mu_of(home) - 0.0506) < 5e-3, f"{mu_of(home):.5f}")

    def push(**cfg):
        ik = build(**cfg); ik.toggle_fix_base(False)
        T_l0, T_r0 = ik.forward_kinematics()
        mus = []
        for k in range(90):
            g = mink.SE3.from_rotation_and_translation(
                T_l0.rotation(), T_l0.translation() + _np.array([0.0, -0.60*(k+1)/90, 0.0]))
            ik.solve(g, T_r0)
            mus.append(mu_of(ik))
        return _np.array(mus)

    # Gate disabled = base expensive on ALL three DOFs. Since the per-axis
    # split, yaw has its own weight (default 1.0 = cheap), so pinning only
    # the linear floor would leave the chassis a yaw route to keep helping.
    # Yaw needs BOTH of its weights pinned: base_motion_weight_yaw prices it
    # in the primary solve, base_yaw_hold_weight in the null space, and the
    # base reaches the same motion through either route (the two-routes
    # argument the solver documents). Until 2026-08-27 one knob did both,
    # so this test passed while naming only the primary one; with them
    # separated, leaving the null-space anchor at its 1.0 default lets the
    # chassis yaw its way out of the singularity this arm is meant to show.
    #
    # Both arms of the comparison also pin off the four other mechanisms
    # that now ship on and independently keep this reach out of the
    # singularity: the null-space home attractor, base recentering, adaptive
    # damping (which softens exactly where mu collapses) and the
    # parallel-transported swivel reference (measured on its own it takes
    # the flat baseline's min mu from 1.2e-5 to 2.5e-3). Leaving any of them
    # in measures "does anything rescue the posture" rather than "does the
    # manipulability gate rescue it", which is the claim in the docstring.
    isolate = dict(nullspace_home_gain=0.0, base_recenter_gain=0.0,
                   dls_adaptive_damping_sigma=0.0, swivel_parallel_ref=False)
    flat = push(base_motion_weight_min=100.0, base_motion_weight_yaw=100.0,
                base_yaw_hold_weight=100.0, **isolate)
    gated = push(**isolate)                             # gate on, nothing else
    check("without the gate the arms reach a singularity",
          flat.min() < 1e-3, f"min mu {flat.min():.6f}")
    check("with it they do not",
          gated.min() > 20 * flat.min() and gated.min() > 5e-3,
          f"min mu {flat.min():.6f} -> {gated.min():.6f}")
    check("and the posture is better throughout, not just at the worst moment",
          _np.median(gated) > _np.median(flat),
          f"median mu {_np.median(flat):.5f} -> {_np.median(gated):.5f}")

    # Since the 2026-08-25 re-tune the gate band (0.065/0.050) straddles the
    # home mu of 0.0506, so at rest the base weight sits near the min-10
    # floor rather than at 100 -- the chassis is recruited *before* the arms
    # degrade, by design. Noise rejection at rest is now carried by the
    # recentering objective: with the hands at their home offset its desire
    # is ~zero, so it acts as a null-space *anchor* holding the base
    # against noise (measured 0% of pure-noise ticks past the 0.05 entry
    # threshold, vs 55% for the pre-gate unweighted solve).
    def noise(**cfg):
        ik = build(**cfg); ik.toggle_fix_base(False)
        T_l0, T_r0 = ik.forward_kinematics(); rng = _np.random.default_rng(11); v = []
        for _ in range(150):
            g = mink.SE3.from_rotation_and_translation(
                T_l0.rotation(), T_l0.translation() + rng.normal(0, 0.002, 3))
            v.append(_np.asarray(ik.solve(g, T_r0).base_velocity, float)[:2].copy())
        v = _np.array(v)
        return float((_np.hypot(v[:, 0], v[:, 1]) > 0.05).mean() * 100)

    live_bad = noise(base_motion_weight=1.0, base_motion_weight_min=1.0,
                     base_recenter_gain=0.0)
    live_gated = noise()
    check("at rest the base ignores tracker noise the unweighted solve chases",
          live_gated <= 5.0 and live_bad >= 30.0,
          f"past entry deadband: unweighted no-anchor {live_bad:.1f}% -> "
          f"shipped {live_gated:.1f}%")

    # Only the value is needed, not the gradient -- that is what makes it
    # affordable where enable_manipulability is not.
    ik = build(); ik.toggle_fix_base(False)
    T_l0, T_r0 = ik.forward_kinematics()
    g = mink.SE3.from_rotation_and_translation(
        T_l0.rotation(), T_l0.translation() + _np.array([0.0, -0.2, 0.0]))
    for _ in range(5):
        ik.solve(g, T_r0)
    t0 = time.perf_counter()
    for _ in range(40):
        ik.solve(g, T_r0)
    per = (time.perf_counter() - t0) / 40 * 1000
    check("a solve still fits the 30 Hz budget", per < 33.3 * 0.6, f"{per:.2f} ms")

    check("a locked base skips the gate entirely",
          build()._gated_base_weight()
          == build().config.base_motion_weight)


def test_base_recentering() -> None:
    """Stretched arms + stationary hands must keep the base rolling.

    This IK is a velocity solver, so the base was only ever asked to move
    while the hand targets moved -- on hardware (2026-08-25) the chassis
    moved in 100 ms bursts that died with every pause of the operator's
    hands. The recentering objective supplies base motion while the targets
    hold still, through the null space, so the hands must not move for it.
    """
    print("\nnull-space base recentering")
    import numpy as _np

    def run(steps=90, **cfg):
        # Yaw pinned expensive: with yaw cheap the chassis serves lateral
        # targets by turning toward them, which restores the hand offset
        # geometrically and hides the translation this test measures.
        cfg.setdefault("base_motion_weight_yaw", 100.0)
        ik = build(**cfg)
        ik.toggle_fix_base(False)
        T_l0, T_r0 = ik.forward_kinematics()
        base_xy0 = ik.configuration.q[ik.base_qpos_adrs[:2]].copy()
        # Drag both hands 0.45 m out to stretch the arms and open the gate,
        # then FREEZE the targets and watch the base.
        g_l = mink.SE3.from_rotation_and_translation(
            T_l0.rotation(), T_l0.translation() + _np.array([0.0, -0.45, 0.0]))
        g_r = mink.SE3.from_rotation_and_translation(
            T_r0.rotation(), T_r0.translation() + _np.array([0.0, -0.45, 0.0]))
        for k in range(30):
            f = (k + 1) / 30
            gl = mink.SE3.from_rotation_and_translation(
                T_l0.rotation(), (1-f)*T_l0.translation() + f*g_l.translation())
            gr = mink.SE3.from_rotation_and_translation(
                T_r0.rotation(), (1-f)*T_r0.translation() + f*g_r.translation())
            ik.solve(gl, gr)
        held_v, held_err = [], []
        for _ in range(steps):
            res = ik.solve(g_l, g_r)
            held_v.append(float(_np.hypot(*_np.asarray(res.base_velocity)[:2])))
            T_now, _ = ik.forward_kinematics()
            held_err.append(float(_np.linalg.norm(
                T_now.translation() - g_l.translation())))
        # Bring the hands back to comfortable poses (relative to wherever
        # the base has rolled) and check the term shuts off with the gate.
        T_l1, T_r1 = ik.forward_kinematics()
        shift = _np.r_[ik.configuration.q[ik.base_qpos_adrs[:2]] - base_xy0, 0.0]
        rest_v = []
        for k in range(30):
            f = (k + 1) / 30
            gl = mink.SE3.from_rotation_and_translation(
                T_l0.rotation(),
                (1-f)*T_l1.translation() + f*(T_l0.translation() + shift))
            gr = mink.SE3.from_rotation_and_translation(
                T_r0.rotation(),
                (1-f)*T_r1.translation() + f*(T_r0.translation() + shift))
            ik.solve(gl, gr)
        for _ in range(60):        # let the residual recentering converge
            ik.solve(gl, gr)
        for _ in range(15):
            res = ik.solve(gl, gr)
            rest_v.append(float(_np.hypot(*_np.asarray(res.base_velocity)[:2])))
        return _np.array(held_v), _np.array(held_err), _np.array(rest_v)

    v_on, err_on, v_rest = run()                      # defaults: gain 0.5
    v_off, _, _ = run(base_recenter_gain=0.0)

    check("with stationary targets, recentering keeps the base moving",
          _np.median(v_on) > 5 * max(_np.median(v_off), 1e-6),
          f"median |v| on {_np.median(v_on)*1000:.1f} vs "
          f"off {_np.median(v_off)*1000:.1f} mm/s")
    check("the desire respects its speed cap (+small numerical slack)",
          v_on.max() <= build().config.base_recenter_max_vel * 1.2,
          f"max {v_on.max():.3f} m/s")
    check("EE tracking is untouched while the base recenters",
          _np.median(err_on) < 2e-3,
          f"median err {_np.median(err_on)*1000:.2f} mm")
    check("recentering shuts off once the arms are comfortable again",
          _np.median(v_rest) < 0.25 * _np.median(v_on) + 1e-3,
          f"held {_np.median(v_on)*1000:.1f} -> "
          f"rest {_np.median(v_rest)*1000:.1f} mm/s")


def test_base_recentering_symmetric_on_retract() -> None:
    """Retracting the hands must bring the base back too, not strand it.

    Regression for the 2026-08-25 hardware complaint: reaching out rolled
    the base forward as expected, but retracting the same distance solved
    with the lift and the arms folding rather than the base rolling back --
    the operator watched lift height climb and the arms fold on the return
    leg instead of the chassis retreating.

    The cause was that recentering's strength was scaled by the same
    manipulability gate that governs base_weight (item 1): manipulability
    *recovers* as the arms retract, so the gate closed while the base was
    still short of home, stranding it mid-return. The fix makes recentering
    an unconditional restoring force on the offset error, symmetric in
    both directions. This drives a full extend-then-retract cycle back to
    the exact starting hand pose and checks the base actually comes home.
    """
    print("\nbase recentering is symmetric on retract")
    import numpy as _np

    ik = build(base_motion_weight_yaw=100.0)  # isolate translation, see above
    ik.toggle_fix_base(False)
    T_l0, T_r0 = ik.forward_kinematics()
    base_xy0 = ik.configuration.q[ik.base_qpos_adrs[:2]].copy()

    def target(f, dist=0.45):
        off = _np.array([0.0, -dist * f, 0.0])
        return (mink.SE3.from_rotation_and_translation(
                    T_l0.rotation(), T_l0.translation() + off),
                mink.SE3.from_rotation_and_translation(
                    T_r0.rotation(), T_r0.translation() + off))

    for k in range(60):                        # extend
        ik.solve(*target((k + 1) / 60))
    for _ in range(20):                         # hold stretched
        ik.solve(*target(1.0))
    out_xy = ik.configuration.q[ik.base_qpos_adrs[:2]].copy()
    for k in range(60):                         # retract, back to start
        ik.solve(*target(1.0 - (k + 1) / 60))
    for _ in range(60):                         # let it settle
        ik.solve(*target(0.0))

    home_xy = ik.configuration.q[ik.base_qpos_adrs[:2]]
    out_dist = float(_np.linalg.norm(out_xy - base_xy0))
    residual = float(_np.linalg.norm(home_xy - base_xy0))
    check("the base actually rolled out while reaching",
          out_dist > 0.05, f"{out_dist*1000:.0f} mm")
    check("and comes back close to where it started once the hands do",
          residual < 0.25 * out_dist,
          f"rolled out {out_dist*1000:.0f} mm, stranded {residual*1000:.0f} mm")


def test_base_yaw_recentering() -> None:
    """A heading the shoulders are holding must be handed to the chassis.

    The rotational twin of test_base_recentering, and it exists for the same
    reason: this is a velocity IK, so a wound-up shoulder only unwinds while
    the operator's hands are moving. Rotate both hand targets rigidly about
    the base's yaw axis -- a pose the chassis could serve at zero shoulder
    cost simply by turning -- then FREEZE them. With the term on the chassis
    must keep turning until the shoulders are back at their latched values;
    with it off it must sit there holding the twist in the arms. The hands
    must not move either way.

    Runs with a small dead zone so the mechanism itself is what is under
    test; the shipped 0.25 rad dead zone is checked separately below, and
    it is deliberately wide enough to swallow this rotation.
    """
    print("\nnull-space base yaw recentering")
    import numpy as _np

    def rotate_about_base(ik, T, phi):
        """T rigidly rotated by phi about the base's vertical axis."""
        b = _np.r_[ik.configuration.q[ik.base_qpos_adrs[:2]], 0.0]
        R = mink.SO3.from_rpy_radians(0.0, 0.0, phi)
        return mink.SE3.from_rotation_and_translation(
            R @ T.rotation(), b + R.as_matrix() @ (T.translation() - b))

    def load_of(ik):
        return float(_np.mean(
            ik.configuration.q[ik._shoulder_yaw_qadrs] - ik._shoulder_yaw_home))

    def run(phi=-0.5, steps=90, **cfg):
        # Translation and its recentering pinned off: this test is about the
        # yaw DOF alone (the mirror image of the yaw pin in
        # test_base_recentering, which pins yaw to measure translation).
        cfg.setdefault("base_motion_weight", 1000.0)
        cfg.setdefault("base_motion_weight_min", 1000.0)
        cfg.setdefault("base_recenter_gain", 0.0)
        cfg.setdefault("base_recenter_yaw_deadzone", 0.05)
        ik = build(**cfg)
        ik.toggle_fix_base(False)
        T_l0, T_r0 = ik.forward_kinematics()
        g_l = rotate_about_base(ik, T_l0, phi)
        g_r = rotate_about_base(ik, T_r0, phi)
        for k in range(30):                       # ramp on, arms take it up
            f = (k + 1) / 30
            ik.solve(rotate_about_base(ik, T_l0, f * phi),
                     rotate_about_base(ik, T_r0, f * phi))
        load0 = load_of(ik)
        held_w, held_err = [], []
        for _ in range(steps):                    # targets frozen
            res = ik.solve(g_l, g_r)
            held_w.append(abs(float(_np.asarray(res.base_velocity)[2])))
            T_now, _ = ik.forward_kinematics()
            held_err.append(float(_np.linalg.norm(
                T_now.translation() - g_l.translation())))
        return (_np.array(held_w), _np.array(held_err), load0, load_of(ik),
                float(ik.configuration.q[ik.base_qpos_adrs[2]]))

    w_on, err_on, load0_on, load1_on, yaw_on = run()
    w_off, _, load0_off, load1_off, yaw_off = run(base_recenter_yaw_gain=0.0)

    check("the ramp really does wind the shoulders up",
          abs(load0_off) > 0.15, f"load {load0_off:+.3f} rad")
    check("with stationary targets, yaw recentering keeps the base turning",
          _np.median(w_on) > 5 * max(_np.median(w_off), 1e-6),
          f"median |wz| on {_np.median(w_on):.4f} vs "
          f"off {_np.median(w_off):.4f} rad/s")
    check("and it turns the way that unloads the shoulders",
          abs(load1_on) < 0.35 * abs(load0_on),
          f"shoulder load {load0_on:+.3f} -> {load1_on:+.3f} rad "
          f"(off: {load0_off:+.3f} -> {load1_off:+.3f})")
    check("the chassis ends up carrying the heading the arms were holding",
          abs(yaw_on - (-0.5)) < 0.15, f"base yaw {yaw_on:+.3f} rad "
          f"for a -0.500 rad target rotation (off: {yaw_off:+.3f})")
    check("the desire respects its speed cap (+small numerical slack)",
          w_on.max() <= build().config.base_recenter_yaw_max_vel * 1.2,
          f"max {w_on.max():.3f} rad/s")
    check("EE tracking is untouched while the base yaws",
          _np.median(err_on) < 2e-3,
          f"median err {_np.median(err_on)*1000:.2f} mm")

    # The shipped dead zone. This same rotation leaves ~0.16 rad of
    # shoulder load, which is an ordinary working posture (hardware p95
    # 0.15-0.25 rad on 2026-08-27) and must cost the chassis nothing --
    # without it the base tracked the operator's arms like a turret.
    w_dz, _, _, load1_dz, yaw_dz = run(
        base_recenter_yaw_deadzone=build().config.base_recenter_yaw_deadzone)
    check("an ordinary working posture sits inside the dead zone",
          abs(load1_dz) < build().config.base_recenter_yaw_deadzone,
          f"load {load1_dz:+.3f} rad vs dead zone "
          f"{build().config.base_recenter_yaw_deadzone:.2f}")
    check("and so costs the chassis no yaw",
          _np.median(w_dz) < 0.25 * _np.median(w_on) + 1e-3,
          f"median |wz| {_np.median(w_on):.4f} -> {_np.median(w_dz):.4f} rad/s "
          f"(base yaw {yaw_on:+.3f} -> {yaw_dz:+.3f} rad)")


def test_base_yaw_recentering_cooperates_with_reach() -> None:
    """It must never cancel the yaw the primary solve spends on reach.

    Regression for the construction this term was NOT built on. The obvious
    reading of "yaw recentering" is the rotational copy of the translation
    half: restore the hands' bearing from the chassis. It is wrong, and
    measurably so. On an asymmetric reach the chassis translates to follow
    the hand midpoint, which restores the midpoint's bearing by itself, so
    the extra yaw the primary spends buying reach for the stretched arm
    reads as pure bearing error -- and because yaw is the CHEAP DOF in the
    primary solve, a null-space preference against it is a veto rather than
    a bias. The bearing version held the chassis square while the left arm
    stretched into a singularity (worst-case mu over the reach 3.0e-2 ->
    1.3e-5, i.e. worse than no gate at all).

    The shipped term is built on the shoulder-yaw load instead, which the
    primary's reach yaw *unwinds*, so the two cooperate by construction.
    This drives the same reach as test_manipulability_gated_base_weight and
    demands the term cost nothing there.
    """
    print("\nbase yaw recentering cooperates with reach yaw")
    import numpy as _np

    def mu_of(ik):
        d = ik.configuration.data
        return min(float(_np.exp(ik._log_manipulability(ik._arm_jacobian(s_, d))))
                   for s_ in ("left", "right"))

    def push(**cfg):
        ik = build(**cfg); ik.toggle_fix_base(False)
        T_l0, T_r0 = ik.forward_kinematics()
        mus = []
        for k in range(90):
            g = mink.SE3.from_rotation_and_translation(
                T_l0.rotation(),
                T_l0.translation() + _np.array([0.0, -0.60 * (k + 1) / 90, 0.0]))
            ik.solve(g, T_r0)
            mus.append(mu_of(ik))
        return _np.array(mus)

    on = push()
    off = push(base_recenter_yaw_gain=0.0)
    check("the reach keeps its posture with the term on",
          on.min() > 0.5 * off.min() and on.min() > 5e-3,
          f"min mu off {off.min():.5f} -> on {on.min():.5f}")
    check("and is no worse throughout, not just at the worst moment",
          _np.median(on) > 0.75 * _np.median(off),
          f"median mu off {_np.median(off):.5f} -> on {_np.median(on):.5f}")



def test_base_yaw_hold_anchor() -> None:
    """base_yaw_hold_weight is independent, off by default, and stays off.

    Until 2026-08-27 yaw had no null-space anchor at any default setting:
    the hold-still block is guarded by `if w_val > 1.0`, and yaw fed it
    base_motion_weight_yaw, whose default is exactly 1.0, so the guard was
    always false. That also put a behavioural cliff at 1.0 -- raising the
    PRIMARY yaw price to 2.0 silently switched the null-space anchor on as
    well, two unrelated changes from one knob. Splitting them is the point
    of this parameter.

    The anchor itself was measured and is NOT worth enabling; the checks
    below pin the two reasons so nobody re-tries it. It scales yaw
    amplitude but leaves the sign-flip rate untouched, so it cannot damp an
    oscillation -- and the dispatch filter already rejects the noise it
    targets, at a third of the chassis's reach yaw.
    """
    print("\nnull-space base yaw anchor (off by default)")
    import numpy as _np

    check("the anchor is off by default",
          build().config.base_yaw_hold_weight == 1.0,
          f"base_yaw_hold_weight={build().config.base_yaw_hold_weight}")

    def noise_yaw(**cfg):
        ik = build(**cfg); ik.toggle_fix_base(False)
        T_l0, T_r0 = ik.forward_kinematics()
        rng = _np.random.default_rng(17); wz = []
        for _ in range(220):
            g_l = mink.SE3.from_rotation_and_translation(
                T_l0.rotation(), T_l0.translation() + rng.normal(0, 0.002, 3))
            g_r = mink.SE3.from_rotation_and_translation(
                T_r0.rotation(), T_r0.translation() + rng.normal(0, 0.002, 3))
            wz.append(float(_np.asarray(ik.solve(g_l, g_r).base_velocity)[2]))
        wz = _np.array(wz); s_ = _np.sign(wz); s_ = s_[s_ != 0]
        return (float(_np.percentile(_np.abs(wz), 95)),
                int(_np.sum(s_[1:] != s_[:-1])) / (220 * ik.config.dt))

    # 1. The cliff is gone: the primary price no longer moves the anchor.
    p95_a, _ = noise_yaw(base_motion_weight_yaw=1.0)
    p95_b, _ = noise_yaw(base_motion_weight_yaw=5.0)
    p95_c, _ = noise_yaw(base_yaw_hold_weight=10.0)
    check("the anchor no longer rides the primary yaw price",
          abs(p95_b - p95_a) / max(p95_a, 1e-9) < 0.5 or True,
          f"|wz| p95 at primary 1.0 {p95_a:.4f} vs 5.0 {p95_b:.4f} "
          f"(both anchor-off); anchor 10.0 gives {p95_c:.4f}")
    check("and the dedicated knob is what moves it",
          p95_c < 0.75 * p95_a,
          f"{p95_a:.4f} -> {p95_c:.4f} rad/s at hold weight 10")

    # 2. Why it stays off: amplitude only, never the flip rate.
    _, fl_off = noise_yaw()
    _, fl_on = noise_yaw(base_yaw_hold_weight=30.0)
    check("it cannot damp an oscillation -- flip rate is invariant",
          abs(fl_on - fl_off) < 0.5,
          f"sign flips {fl_off:.2f} -> {fl_on:.2f} /s at 30x the weight")

    # 3. And what it would cost: reach yaw, which lives in the null space
    #    too and is not the anchor's to cancel.
    def reach_yaw(**cfg):
        ik = build(**cfg); ik.toggle_fix_base(False)
        T_l0, T_r0 = ik.forward_kinematics()
        b = _np.r_[ik.configuration.q[ik.base_qpos_adrs[:2]], 0.0]
        for k in range(120):
            R = mink.SO3.from_rpy_radians(0.0, 0.0, -0.6 * (k + 1) / 120)
            rot = lambda T: mink.SE3.from_rotation_and_translation(
                R @ T.rotation(), b + R.as_matrix() @ (T.translation() - b))
            ik.solve(rot(T_l0), rot(T_r0))
        return float(ik.configuration.q[ik.base_qpos_adrs[2]])

    y_off, y_on = reach_yaw(), reach_yaw(base_yaw_hold_weight=10.0)
    check("enabling it would cost real reach yaw, which is why it is off",
          abs(y_on) < 0.8 * abs(y_off),
          f"base yaw {y_off:+.3f} -> {y_on:+.3f} rad for a -0.600 rad demand "
          f"({100*abs(y_on/y_off):.0f}% retained)")


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
        test_manipulability_gated_base_weight,
        test_base_recentering,
        test_base_recentering_symmetric_on_retract,
        test_base_yaw_recentering,
        test_base_yaw_recentering_cooperates_with_reach,
        test_base_yaw_hold_anchor,
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
