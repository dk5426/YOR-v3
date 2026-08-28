"""
wholebody_ik.py — Whole-body IK solver for YORv3.

Architecture
------------
The mobile base uses three explicit planar joints (base_x slide,
base_y slide, base_yaw hinge) rather than a freejoint. This removes
z / roll / pitch from the physics entirely, so the chassis cannot drift
off the ground plane (an earlier freejoint + DampingTask approach let it
slowly float upward — see context.md). mink optimises all three base DOFs
directly; no ground-lock DampingTask is needed.

  Base  (3 DOF):  base_x, base_y, base_yaw
  Lift  (1 DOF):  Slider 7
  Left  (7 DOF):  left_arm_joint1..7
  Right (7 DOF):  right_arm_joint1..7
  ─────────────────────────────────────────────────
  Total IK DOF: 3 base + 1 lift + 7L + 7R = 18   (model nq = 66)

Self-collision avoidance (optional, runtime-toggleable via
`avoid_collisions`) adds hard QP inequality constraints so the solver can
never drive an arm into the lift column, the chassis, the other arm, or
the floor.

The same solver drives both the simulation (robot/yor_mujoco.py) and the
real robot (robot/wholebody_control.py). The only differences are where
the measured configuration comes from and where the result is dispatched.

Usage (kinematic mode — simulation / IK demo):
----------------------------------------------
    from robot.arm.wholebody_ik import WholeBodyIK
    import mink, mujoco

    ik = WholeBodyIK()                # defaults to description/scene_wholebody.xml
    ik.init_from_keyframe("home")

    T_left  = mink.SE3.from_translation([0.5, 0.3, 1.2])
    T_right = mink.SE3.from_translation([0.5, -0.3, 1.2])

    result = ik.solve(T_left, T_right)

    # Kinematic mode: write full qpos directly
    data.qpos[:] = result.q
    mujoco.mj_forward(model, data)

Usage (real hardware):
----------------------
    ik.set_measured_state(left_q=..., right_q=..., lift=..., base=...)
    result = ik.solve(T_left, T_right, lift_target=h)
    #   result.left_arm_q / right_arm_q → ArmNode.set_joint_target()
    #   result.lift_q                   → lift servo (bang-bang on PicoLift)
    #   result.base_velocity            → Base.set_target_base_velocity()
    # robot/wholebody_control.py does exactly this.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import mujoco
import numpy as np
import qpsolvers

import mink

# Repo-relative description paths (repo_root/description/…), so the solver
# works the same from a checkout on the robot as it does on a dev machine.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTION_DIR = _REPO_ROOT / "description"
DEFAULT_SCENE = DESCRIPTION_DIR / "scene_wholebody.xml"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WholeBodyIKConfig:
    """Tunable parameters for the whole-body IK solver.

    Solver benchmark on YORv3 (18 IK DOF, M-series Mac, 50 solves):
      pyqpmad  0.12ms  ← fastest, recommended for real-time
      daqp     0.13ms
      quadprog 0.14ms
      proxqp   0.25ms
      qpalm    0.26ms
      osqp     0.45ms
      clarabel 0.42ms
      ecos/scs/piqp  0.5-0.6ms
      cvxopt   24ms avg / 1155ms worst  ← DO NOT USE
      jaxopt   272ms avg               ← DO NOT USE (JIT overhead)
      highs / qpax / sip               ← failed on this problem
    """

    # QP solver
    solver: str = "pyqpmad"
    fallback_solvers: list[str] = field(
        default_factory=lambda: ["daqp", "quadprog", "proxqp", "qpalm", "osqp"]
    )

    # Integration time-step
    dt: float = 0.01

    # IK iterations per solve call
    max_iters: int = 20

    # Convergence
    pos_threshold: float = 1e-3    # metres
    ori_threshold: float = 1e-3    # radians

    # ── Task costs ──────────────────────────────────────────────────────────
    ee_position_cost: float    = 1.0
    ee_orientation_cost: float = 0.5
    ee_lm_damping: float       = 1.0

    # Posture regularisation (how much to penalise moving away from home).
    # NOTE: these dataclass defaults are NOT what runs on the robot —
    # WholeBodyController, yor_mujoco.py and tools/wholebody_ik_demo.py each
    # construct their own config. Change the call site, not these.
    base_posture_cost: float   = 1e-4   # low → base moves freely for reachability
    lift_posture_cost: float   = 1e-2   # moderate → stay near current height
    arm_posture_cost: float    = 1e-3   # light → smooth arm configs

    # Per-joint posture cost, overriding `arm_posture_cost` for named joints.
    # Keys are model joint names ("left_arm_joint1" … "right_arm_joint7").
    # Higher cost → the solver penalises moving that joint more and reaches with
    # the others instead; lower → that joint takes up the slack. Only the ratio
    # to `arm_posture_cost` matters, so scale against it:
    #   10× stiffer  → {"left_arm_joint1": 1e-2}
    #   10× softer   → {"left_arm_joint7": 1e-4}
    #
    # Measured response, one EE target 0.10 m out from home, joint1 varied
    # against arm_posture_cost=1e-3 (|dq| for that joint / summed over the arm):
    #     1e-3  (1×)      0.171 rad / 0.947      the uniform baseline
    #     1e-2  (10×)     0.111 rad / 0.991
    #     1e-1  (100×)    0.000 rad / 1.076      effectively pinned
    #     1e0   (1000×)   0.000 rad / 1.103      no further effect
    # So ~0.1× to ~100× is the useful band; beyond that it saturates. Note the
    # arm total *rises* as one joint stiffens — the others absorb the motion.
    #
    # A cost of 0 is not "free": mink needs a positive cost, so use a small
    # value (1e-8, the vector's floor) to mean effectively free.
    # Unknown joint names raise at construction rather than being ignored.
    arm_posture_cost_overrides: dict[str, float] = field(default_factory=dict)

    # Whether solve() refreshes the posture task's *entire* reference to the
    # current configuration every call, or only touches it (lift DOF only)
    # when a lift_target is given -- the latter is legacy behavior: with no
    # lift_target (e.g. arms-only sessions where the lift is never touched),
    # the posture reference is set once at init_from_keyframe() and never
    # refreshed again, so it becomes a fixed, increasingly stale attractor
    # for the whole session. True turns the posture task into a proper
    # minimum-norm null-space regularizer (discourage unnecessary null-space
    # velocity each solve) instead of a pull toward a fixed, distant point.
    refresh_posture_target: bool = False

    # ── Redundancy resolution strategy ────────────────────────────────────────
    # How the null space (the freedom left over once both 6-DOF EE tasks are
    # satisfied) gets resolved. All three should reach the same EE targets;
    # they differ in HOW the leftover freedom is used -- which is exactly
    # what's been under test, see artifacts/wholebody_logs/
    # posture_fix_commands.md for the empirical story (soft weighting was
    # tried up to a ~100,000:1 EE:posture ratio and the instability neither
    # shrank nor moved -- confirming it's structural, not a tuning problem).
    #
    #   "soft"            - today's/default behavior: EE tasks and posture
    #                        both soft costs in one QP (mink.solve_ik's
    #                        tasks=). Simple and always finds a best-effort
    #                        answer, but nothing stops the posture term from
    #                        perturbing EE tracking a little, or the
    #                        null-space optimum from jumping discontinuously
    #                        between two equally-cheap points on the arm's
    #                        self-motion manifold (observed on joints 3/5 --
    #                        elbow region -- under refresh_posture_target).
    #                        (A third mode, "hard_constraint", promoted the
    #                        EE tasks to mink's constraints= for exact
    #                        null-space confinement. It was implemented, A/B'd
    #                        on hardware 2026-08-20, and removed: it barely
    #                        improved on "soft" (3.6 vs 5.3 elbow flips per
    #                        1000 ticks, against dls_projector's 2.2) and could
    #                        return no solution at all near a singularity.)
    #   "dls_projector"    - hand-rolled damped-least-squares pseudoinverse
    #                        plus explicit null-space projection
    #                        (q̇ = J⁺ẋ + (I - J⁺J)q̇₀), bypassing mink's task
    #                        stack for the EE+posture resolution entirely.
    #                        Never infeasible on its own -- degrades smoothly
    #                        through singularities/self-motion ambiguity
    #                        instead of jumping or failing -- at the cost of
    #                        exactness (bounded leakage, set by dls_damping).
    #                        Joint/collision limits aren't part of the same
    #                        solve; they're enforced afterward by projecting
    #                        the result onto them via a small auxiliary QP.
    redundancy_resolution: Literal["soft", "dls_projector"] = "dls_projector"

    # Tikhonov/DLS damping λ for "dls_projector"'s pseudoinverse
    # J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹. Larger = more robust near singularities, at the
    # cost of slower/less exact EE tracking there. 0.01-0.1 is the usual
    # range in the redundancy-resolution literature. Unused by "soft".
    dls_damping: float = 0.05

    # ── dls_projector null-space (secondary) objectives ──────────────────────
    # Everything in this block shapes ONLY redundancy_resolution=
    # "dls_projector"; "soft" ignores it entirely.
    #
    # The secondary solve picks z in  q̇ = q̇_primary + N z  by minimising a
    # stacked weighted least-squares of the objectives below, rather than
    # projecting a single hand-built gradient step. Weights are relative to
    # each other; only their ratios matter.
    #
    # Setting continuity/swivel/manipulability weights to 0 with
    # nullspace_posture_weight=1 reproduces the older behaviour
    # (q̇ = q̇_primary + N q̇_posture) exactly -- see
    # tests/test_nullspace_objectives.py.
    #
    # Note dim(null(J)) is only 2 with both arms tracking 6-DOF targets and
    # the base/lift fixed (14 free DOF - 12 task rows), i.e. one swivel DOF
    # per arm. A strict task-priority hierarchy would leave nothing below the
    # first level, so these are weights in one solve, not nested projections.
    nullspace_posture_weight: float = 1.0

    # Penalise ||q̇ - q̇_prev||, where q̇_prev is the velocity the previous
    # solve() call actually applied. Damps tick-to-tick changes in null-space
    # motion (the elbow "swimming" between equivalent solutions) without
    # touching EE tracking, which lives in the primary term.
    # Default 0: measured counterproductive on this robot, see the sweep
    # quoted in _solve_qp_dls_projector. Raise only with a metric in hand.
    nullspace_continuity_weight: float = 0.0

    # Elbow swivel: hold each arm's elbow at a chosen angle about the
    # shoulder->wrist axis, instead of letting the solver pick whichever
    # elbow branch is locally cheapest each tick.
    # Measured elbow drift over a reversing target with an abrupt
    # disturbance: weight 0 -> 26.9°, 1.0 -> 8.2°, 5.0 -> 1.9°, with EE
    # tracking unchanged (0.287 -> 0.235 -> 0.235 mm median). 1.0 is a
    # conservative starting point; 5.0 held noticeably tighter in the
    # kinematic sweep and is worth an A/B on hardware.
    nullspace_swivel_weight: float = 1.0
    elbow_swivel_gain: float = 1.0
    # Per-side target swivel angle in radians, keys "left"/"right". Any side
    # left out is latched from the pose at the first solve after
    # init_from_keyframe()/init_from_qpos(), so the arm keeps the elbow
    # branch it started in rather than choosing one.
    elbow_swivel_targets: dict[str, float] = field(default_factory=dict)
    # The swivel angle is undefined when the elbow lies on the shoulder-wrist
    # axis (arm straight). Its weight is faded out smoothly as the elbow's
    # perpendicular offset falls below this (metres), so a straightening arm
    # relaxes the objective instead of chasing a singular angle.
    elbow_swivel_min_offset: float = 0.03

    # Optional manipulability maximisation. OFF by default: the gradient is
    # finite-differenced (7 perturbed Jacobians per arm per iteration), which
    # is the most expensive thing in the solve when enabled.
    #
    # NOTE (2026-08-25): with the gate defaults below, enabling the flag is
    # pure cost -- manipulability_gate_on = 0.02 sits far below the operating
    # mu of ~0.042, so the gate never opens; measured on the 2026-08-24
    # replay it gives mu_p05 identical to shipped for 2.4x the solve time.
    # Re-tune the gates before expecting this flag to have any effect. The
    # base-weight gate (base_weight_gate_on/-_full) reaches +32% mu_p05 for
    # +0% solve time and is the mechanism actually in use.
    enable_manipulability: bool = False
    manipulability_weight: float = 0.5
    # Step size in radians along the (normalised) ascent direction, per
    # solver iteration -- not a multiplier on the raw gradient, whose scale
    # is arbitrary.
    manipulability_gain: float = 0.02
    # Smoothstep gate on mu = sqrt(det(J J^T)) of each arm's 6x7 Jacobian:
    # inactive at mu >= gate_on, full weight at mu <= gate_full. Keeps the
    # objective out of the way except when the arm is actually getting close
    # to a singular configuration. gate_on must exceed gate_full.
    manipulability_gate_on: float = 0.02
    manipulability_gate_full: float = 0.005
    manipulability_fd_step: float = 1e-4

    # base_velocity_continuity was removed on 2026-08-25. It carried a
    # fraction of the previous base velocity into the next solve as a warm
    # start (dq = dq_ref + J^+(b - J dq_ref)). The 0.80 default was derived
    # as 1 - dt/0.167 from a chassis lag figure that actually decomposes
    # into ~125 ms of *transport delay* plus only ~27 ms of first-order time
    # constant; a first-order carry is the right shape for a lag and does
    # nothing for a delay. Swept 0.0/0.5/0.9/0.97 on the 2026-08-24 replay
    # at both the shipped and an open base weight: every metric was
    # identical, EE p95 moved by <0.02 mm. If chassis lag needs solver-side
    # compensation, re-derive against a measured lag (feed-forward of the
    # transport delay is the mechanism-matched shape), don't resurrect the
    # carry.

    # Cost multiplier on base DOFs in the primary solve, relative to 1.0 for
    # every arm and lift DOF. 1.0 is the unweighted damped inverse and is
    # arithmetically identical to having no weighting at all.
    #
    # Raising it is what makes "base motion is emergent" true rather than
    # aspirational. Unweighted, the base answers 24% of a pure 2 mm EE noise
    # input, because it has the most leverage per unit of joint motion; the
    # base then chases tracker jitter and reverses direction every other tick.
    # See docs/BASE_COMMAND_LOOP_REVIEW.md and the 2026-08-24 analysis.
    # 100 is the knee measured on 2026-08-24: against pure 2 mm EE noise it
    # takes the fraction of ticks whose base command clears the dispatch
    # deadband from 71.3% to 0.7%, while a target 0.60 m beyond arm reach
    # still rolls the base 0.31 m and converges to 0.00 mm. Weight 30 only
    # reaches 26.7%. Preference, not prohibition: when the arms cannot reach,
    # the primary term supplies base motion no null-space weight can cancel.
    base_motion_weight: float = 100.0

    # What that cost falls to when the arms are running out of posture, and
    # the manipulability band it falls across. Noise rejection is not lost,
    # because tracker noise arrives while the arms are still comfortable and
    # the gate is shut.
    #
    # Band re-picked on the 2026-08-24 replay corpus. mu is 0.0506 at the
    # home keyframe; the earlier band (0.045/0.025) sat entirely *below*
    # that, so the chassis was only recruited after the posture had already
    # degraded -- the opposite of what this gate is documented to do. With
    # gate_on above home the base moves *so that* the posture does not
    # degrade: measured over 55 replay windows, arm reach while driving
    # 0.890 -> 0.807, manipulability p05 +32%, EE error p95 0.769 ->
    # 0.646 mm, at the cost of ~2.5x more base path.
    #
    # The floor is 10, not 1: it keeps the linear DOFs above
    # base_motion_weight_yaw when the gate opens, so yaw stays the cheap
    # route into the base. Measured with the floor at 1 the yaw split below
    # does nothing (yaw_share 0.065 vs 0.209 with the floor).
    base_motion_weight_min: float = 10.0
    base_weight_gate_on: float = 0.065
    base_weight_gate_full: float = 0.050

    # Cost multiplier on the base YAW dof, independent of base_motion_weight,
    # which applies to base_x/base_y only. 1.0 = as cheap as an arm joint.
    # What matters is the RATIO to the linear weight, not the absolute value:
    # with both cheap the minimum-norm solve still reaches with the arms.
    # Measured on the 2026-08-24 replay (with the gate band above and the
    # min-10 floor): chassis share of demanded yaw 0.038 -> 0.209.
    # (A hardware trial at 5.0 on 2026-08-25 killed chassis yaw outright --
    # the requested |wz| p95 fell to 0.000 rad/s. Smooth the yaw at the
    # dispatch filter instead of pricing it out here.)
    base_motion_weight_yaw: float = 1.0

    # Null-space weight of "prefer not to yaw the chassis", independent of
    # the primary price above. Both are needed and they are not the same
    # lever -- exactly the two-routes argument this file already makes for
    # the linear axes: the base can reach a given motion through the
    # primary solve or through the null space, and blocking one route
    # simply hands the job to the other.
    #
    # Until 2026-08-27 yaw had NO null-space anchor at any default setting.
    # The hold-still block below is guarded by `if w_val > 1.0`, and yaw
    # fed it base_motion_weight_yaw, which defaults to exactly 1.0 -- so
    # the guard was always false and the term never built. That also put a
    # behavioural cliff at 1.0 that the parameter name gives no hint of:
    # raising base_motion_weight_yaw to 2.0 does not merely make yaw 2x
    # pricier in the primary, it switches the null-space anchor on for the
    # first time.
    #
    # Why it matters: the manipulability gate is fully open (value p50
    # 0.95-1.00) through every hardware run measured on 2026-08-27, so
    # base_motion_weight sits at its min-10 floor rather than 100, and the
    # chassis is permanently in its cheap regime. Translation is still
    # anchored there -- by the recentering objective, whose desire goes to
    # ~zero at the latched offset and which therefore doubles as a
    # null-space anchor. Yaw recentering used to do the same job for yaw,
    # until its dead zone (added the same day) made it contribute exactly
    # nothing inside +/-0.25 rad. Yaw was then cheap in the primary AND
    # unanchored in the null space, free to chase tracker noise with the
    # arms compensating -- the suspected mechanism behind the ~4 Hz arm
    # limit cycle in traj_20260827_204403 (j5/j6 moving 3.8-4.0 rad with
    # ~0 net displacement and 7.9 sign reversals/s, hands stationary).
    #
    # DEFAULT 1.0 = anchor off, and it stays off, because the anchor was
    # measured and does not pay. Sweep under 2 mm EE noise, against the
    # reach yaw retained on a -0.6 rad rotation demand:
    #
    #   weight   |wz| p95   raw flips/s   post-filter >0.05   reach kept
    #      1.0     0.2918         20.32               1.4%         100%
    #      2.0     0.2695         20.32               0.5%          95%
    #      5.0     0.2192         20.32               0.5%          76%
    #     10.0     0.1672         20.32               0.0%          66%
    #     30.0     0.0864         20.32               0.0%          58%
    #
    # Two things kill it. The flip rate is INVARIANT -- identical to two
    # decimals at every weight, before and after the dispatch filter -- so
    # the anchor scales yaw amplitude and cannot change how often the
    # request reverses. It therefore cannot damp an oscillation, which was
    # the whole reason it was built (the ~4 Hz arm limit cycle above). And
    # base_vel_filter_tau/base_yaw_filter_tau at 0.15 s already reject the
    # noise on their own: unanchored, only 1.4% of filtered ticks clear the
    # old 0.05 entry threshold. Weight 10 buys that last 1.4 points for a
    # third of the chassis's reach yaw.
    #
    # Kept as a knob, and kept independent, for the cliff alone: before
    # this the anchor could only be switched on by raising the PRIMARY
    # price, so --base-motion-weight-yaw 2.0 silently did two unrelated
    # things at once. Runs from 2026-08-27 at that setting had an anchor of
    # 2.0; to reproduce them now, pass base_yaw_hold_weight=2.0 as well.
    # The block subtracts 1.0, matching the linear axes.
    base_yaw_hold_weight: float = 1.0

    # Null-space base recentering: continuously prefer rolling the chassis
    # toward the pose that restores the hands' latched home-pose offset from
    # the base, at gain * distance, capped at max_vel, instead of preferring
    # zero base motion. Added 2026-08-25.
    #
    # Why: this is a velocity IK -- the base is only ever asked to move
    # while the hand targets are moving, so on hardware the chassis moved
    # in 100 ms bursts that started and died with every pause of the
    # operator's hands, however displaced the base still was. This term
    # supplies the missing "keep gliding under the work while the hands
    # hold still": it acts purely in the null space (the arms counter-move,
    # EE tracking is untouched by construction), self-attenuates as the
    # offset shrinks (desire -> 0 at desire -> 0 error, no explicit gate
    # needed), and is applied on the first solver iteration only so
    # per-tick motion is exactly min(gain * dist, max_vel) * dt.
    #
    # It is NOT scaled by the manipulability gate that drives base_weight
    # (item 1/#3), even though it looks like it should be -- that was the
    # first version, and it was wrong. Manipulability recovers as the arms
    # retract, so on a reach-then-retract motion the gate closed while the
    # base was still short of home, stranding it there mid-return; the last
    # part of "bring the hand back" then fell to the arm folding and the
    # lift instead of the chassis. Manipulability answers "is the arm
    # stretched", not "is the base still displaced" -- the offset error
    # already answers that on its own, symmetrically in both directions.
    #
    # The *desire* is exactly min(gain * dist, max_vel); the achieved speed
    # is the least-squares balance of that desire against the hold-still
    # base term and the posture term's penalty on the arms' counter-motion,
    # so it lands at a fraction of the desire, which is why this has its
    # own weight rather than riding the hold-still one. gain = 0 disables.
    # x/y and yaw are separate objectives with separate gains: the
    # translation one below, and base_recenter_yaw_gain. Yaw was originally
    # left out on the reasoning that it is already cheap in the primary
    # solve -- but "cheap when asked" is not the same as "asked at all",
    # which is the whole point of a recentering term.
    #
    # Gain raised 0.5 -> 1.5 on 2026-08-26 from a three-way hardware
    # comparison of ~2.5 min teleop runs (traj_20260826_182603 /
    # _183018 / _183916). Against gain 0.5 it cuts peak arm extension
    # 0.68/0.74 m -> 0.60/0.67 m and the time spent with sigma_min < 0.05
    # from 8.9% to ~2%, which is where the EE error actually lives: error
    # above sigma_min 0.1 is ~1.5 mm in every run, below 0.05 it is 12-26 mm.
    base_recenter_gain: float = 1.5       # 1/s: m/s of desire per metre offset
    base_recenter_max_vel: float = 0.15   # m/s cap on the recentering desire
    base_recenter_weight: float = 100.0   # null-space weight of the desire

    # Null-space base YAW recentering: the rotational twin of the three
    # parameters above. Where those keep the chassis standing under the
    # hands, this keeps it FACING the way the arms are working, so the
    # shoulders do not sit holding a heading the chassis could hold for
    # free.
    #
    # The error is deliberately NOT the hands' bearing from the chassis,
    # which is the obvious construction and is wrong. Measured on the
    # asymmetric reach in test_manipulability_gated_base_weight (left hand
    # dragged 0.60 m sideways, right hand held), a bearing term *fights the
    # primary solve*: the chassis translates to follow the hand midpoint,
    # which restores the midpoint's bearing on its own, so the extra yaw the
    # primary spends buying reach for the stretched arm reads as bearing
    # error and the null-space term cancels it. The chassis then stayed
    # square while the left arm stretched into a singularity -- worst-case
    # manipulability over that reach 3.0e-2 -> 1.3e-5. That conflict is
    # exactly why yaw was left out of recentering originally: yaw is the
    # cheap DOF in the primary solve (base_motion_weight_yaw 1.0 vs 100 for
    # x/y), so a null-space preference against it is a veto, not a bias.
    #
    # The error used instead is the SHOULDER YAW LOAD: how far the two
    # shoulder-yaw joints have drifted from the values latched with the
    # offset above, averaged over the arms. Measured on this model, base
    # yaw and both joint1s trade one-for-one and in the same sense (base
    # yaw +0.15 rad moves left/right joint1 by -0.162 / -0.173 rad with the
    # hands held fixed), so "the chassis turns, the shoulders unwind" is a
    # single well-conditioned exchange, and the desire is simply
    # gain * mean(q1 - q1_latched).
    #
    # Being built on that exchange makes the term cooperative with the
    # primary rather than opposed to it: yaw the primary spends buying
    # reach unwinds the shoulders, which shrinks this desire instead of
    # provoking it. Everything the translation half documents still holds --
    # null-space only (the arms counter-rotate, EE tracking untouched by
    # construction), ungated (manipulability answers "is the arm
    # stretched", not "are the shoulders wound up"), symmetric in both
    # directions, self-attenuating in the error, and applied on the first
    # solver iteration only so the per-tick desire is
    # min(gain * |err|, max_vel) * dt and not that times max_iters.
    #
    # The cap is 0.30 rad/s, inside the 0.2-0.45 rad/s band the 2026-08-24
    # runs measured for real commanded rotation, so recentering can never be
    # the fastest yaw on the robot. gain = 0 disables.
    #
    # DEAD ZONE, added 2026-08-27 after the first hardware runs (traj
    # 20260827_1735..1750). The mechanism measured correct there -- the
    # sign is right (corr(shoulder load, commanded yaw) +0.32..+0.84 across
    # the runs, versus ~0 before) and it does its job (shoulders ended at
    # 0.02-0.08 rad, where the 16:19 run before the change left them stuck
    # at 0.659) -- but with no dead zone it acted on ANY drift, and working
    # naturally off to one side is a comfortable posture, not a fault. The
    # chassis ended up tracking the operator's arms like a turret: yaw rate
    # while the hands moved rose from 0.05-0.11 to 0.148 rad/s and one
    # 106 s run wandered 3.45 rad (~198 deg) net, against <= 0.86 rad for
    # every run before it. Since the chassis heading is the frame
    # everything else is referenced against, that reads as the robot
    # turning away underneath you.
    #
    # 0.25 rad splits the two populations those runs measured cleanly:
    # ordinary working load sits at p95 0.15-0.25 rad, genuine wind-up at
    # 0.5-0.66. Subtracted rather than compared, so the desire enters
    # continuously at the edge instead of stepping. 0 = no dead zone.
    #
    # Verified on traj_20260827_193853 (238 s of teleop, dead zone 0.25).
    # The defect the first runs showed was a PHASE one: commanded yaw
    # tracked the shoulder load tightly but lagged the operator's hand
    # motion by 1.0-2.1 s, which on a few-seconds gesture is past
    # anti-phase, so the chassis turned a beat late and therefore against
    # the hands. The dead zone removed the small-signal response that
    # carried that lag. Measured, against the same runs without it:
    #
    #   lag, hand motion -> commanded yaw   1.0-2.1 s  ->  0.07 s
    #   corr(shoulder load, commanded yaw)  0.32-0.84  ->  0.18
    #   net yaw wander                      3.45 rad   ->  0.44 rad
    #                                       (over 106 s)   (over 238 s)
    #   |wz| while the hands move           <= 0.148   ->  0.074 rad/s
    #   shoulder load at end of run         0.02-0.51  ->  0.081 rad
    #
    # and the term is not thereby dormant: it is past the dead zone 11.2%
    # of that run, commanding 0.069 rad/s when engaged against 0.010 when
    # not, with the load never exceeding 0.496 rad.
    #
    # A metric NOT to be misled by, having been: "fraction of active-yaw
    # ticks whose sign opposes the hand-bearing rate" reads 55% here and
    # looks damning until you take the baseline -- runs with no yaw
    # recentering at all sit at 48-70%, median ~60%. Chassis yaw has never
    # followed hand bearing, by design (see above: a bearing-follower is
    # the construction this term deliberately avoids), so that number does
    # not isolate this term at all.
    #
    # Note the term is NOT gated on the hands holding still, even though
    # the logs show it acts mostly while they move. That was the other
    # candidate fix considered for the phase problem, and the dead zone
    # made it unnecessary: shoulder wind-up is worst exactly while the
    # operator is working, and a pause-only term would arrive after the
    # posture had already degraded -- the same mistake as gating
    # recentering on manipulability, one axis over. If the phase problem
    # ever returns at loads past the dead zone, that gate is the next
    # thing to try, since a corrector that cannot keep up with what it is
    # correcting is better off trimming in the gaps.
    base_recenter_yaw_gain: float = 1.0      # 1/s: rad/s per rad of shoulder load
    base_recenter_yaw_max_vel: float = 0.30  # rad/s cap on the desire
    base_recenter_yaw_deadzone: float = 0.25  # rad of shoulder load ignored
    base_recenter_yaw_weight: float = 100.0  # null-space weight of the desire

    nullspace_regularization: float = 1e-8

    # Singular values below this fraction of the largest are treated as zero
    # when building the null-space projector, i.e. a direction the arm has
    # effectively lost is handed to the secondary objectives rather than
    # being fought over by the primary one.
    nullspace_rank_tol: float = 1e-6

    # DampingTask costs for ground constraint (lock z, roll, pitch)
    ground_lock_cost: float    = 200.0  # high → robot stays upright and on ground

    # DampingTask cost for fix_base mode (lock vx, vy, wz too)
    base_damping_cost: float   = 100.0
    # DampingTask cost for arm-only hardware mode (lock lift velocity).
    lift_damping_cost: float   = 100.0

    # ── Velocity limits ──────────────────────────────────────────────────────
    # Freejoint linear velocity (m/s) and angular velocity (rad/s)
    base_lin_vel_limit: float  = 0.5    # vx, vy
    base_ang_vel_limit: float  = 1.0    # wz
    lift_vel_limit: float      = 0.05   # m/s
    arm_vel_limit: float       = 8.0   # rad/s

    # ── Self-collision avoidance ───────────────────────────────────────────────
    # Hard QP inequality (mink.CollisionAvoidanceLimit): the solver can never
    # produce a velocity that drives an arm into the lift column or the other
    # arm.  Implemented as a constraint, not a soft cost, so it cannot be
    # overridden by EE/posture tasks.
    enable_collision_avoidance: bool = True
    # Also keep arms + hands (finger meshes) clear of the floor plane. Shares
    # the same buffer/gain and the same runtime toggle as self-collision.
    enable_ground_avoidance: bool    = True
    # Gain in (0, 1]: how fast geoms may approach each other. Lower = safer/slower.
    collision_gain: float            = 0.5
    # Buffer (m) the solver must always leave between any avoided geom pair.
    collision_min_distance: float    = 0.02
    # Range (m) at which the constraint switches on. Larger = earlier, costlier.
    collision_detect_distance: float = 0.06

    # ── Gated experiments (2026-08-25 wave) ──────────────────────────────────
    # Every field in this block defaults to "off" / current behaviour, so a
    # config built without touching them reproduces the pre-gate solver
    # exactly. Each is wired to a yor.py flag; see docs/GATED_CHANGES.md.

    # [S1] Null-space home attractor: a weak, always-on pull of the ARM
    # joints toward the home-keyframe posture, applied purely in the null
    # space (the same pattern as base_recenter_*, so it cannot perturb EE
    # tracking by construction). This is the restoring force the solver
    # otherwise lacks: with refresh_posture_target=True the posture task is
    # a velocity regulariser only -- its reference is wherever the arm
    # already is -- so a contorted configuration is as much an equilibrium
    # as a comfortable one. The desire is min(gain * error, max_vel) per
    # joint, self-attenuating near home, applied on the first solver
    # iteration only (like base recentering). gain = 0 disables.
    nullspace_home_gain: float = 0.3       # 1/s: rad/s of desire per rad of error
    nullspace_home_max_vel: float = 0.3    # rad/s per-joint cap on the desire
    nullspace_home_weight: float = 1.0     # weight in the secondary stack

    # [S3] Constraint-aware primary: solve the primary EE step as a QP
    # subject to the same joint/collision inequalities the post-projection
    # uses, instead of computing an unconstrained damped step and clipping
    # it afterwards. The difference is rerouting: when arm folding is
    # blocked by the collision buffer, the clip mostly zeroes the motion,
    # while the constrained QP trades it into base/lift motion that still
    # serves the EE task. Falls back to the unconstrained step + clip on
    # any QP failure, so it can never make the solve less robust than the
    # gate-off behaviour. Costs roughly one extra small QP per iteration.
    constrained_primary: bool = True

    # [S4a] Apply the FrameTask cost weighting (ee_position_cost /
    # ee_orientation_cost, as row scaling on J and b -- mink's own
    # _weighted_residual convention) inside the dls_projector stack. Off,
    # those two knobs are silently ignored in dls mode and position metres
    # weigh the same as orientation radians.
    dls_task_weighting: bool = True

    # [S4b] Adaptive damping: when the smallest singular value of the
    # (weighted) task Jacobian falls below this threshold, lambda ramps
    # smoothly from dls_damping up to dls_damping_max at sigma = 0.
    # Keeps tracking crisp in well-conditioned poses and only softens near
    # singularity, instead of paying the flat lambda everywhere. 0 = off.
    dls_adaptive_damping_sigma: float = 0.05
    dls_damping_max: float = 0.2

    # [S5a] Parallel-transported swivel reference: the elbow swivel angle
    # needs an in-plane reference vector, and the default construction
    # switches discontinuously between world +z and world +x when the
    # shoulder->wrist axis passes |u_z| = 0.9 (~64 deg from horizontal) --
    # the measured angle then jumps while the latched target does not, and
    # the objective yanks the elbow. This gate replaces it with a per-side
    # reference vector that is rotated along with the axis (parallel
    # transport), which is continuous everywhere. Slow reference drift over
    # closed axis loops (sphere holonomy) is possible; the re-latch policy
    # below is the backstop.
    swivel_parallel_ref: bool = True

    # [S5b] Swivel re-latch policy: if the swivel error stays above this
    # threshold for swivel_relatch_after_s continuously, accept the branch
    # the arm is actually in (re-latch the target to the measured angle)
    # instead of fighting it across a region it may not be able to cross.
    # 0 = off (fight forever, recover only by homing).
    swivel_relatch_err_rad: float = 1.57
    swivel_relatch_after_s: float = 1.0

    # [S7] Per-solve diagnostics (sigma_min / manipulability per arm, swivel
    # angle/target/error per side, active collision-constraint rows),
    # published as WholeBodyIK.diagnostics and recorded by the trajectory
    # CSV. Cheap (two 6x7 SVDs per solve). On by default; the gate exists
    # so a latency-critical run can shed even that.
    record_solver_diagnostics: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WholeBodyIKResult:
    """Output of a single solve() call."""
    # Full qpos (nq values) — write directly to data.qpos for kinematic mode
    q: np.ndarray

    # Decomposed outputs for hardware control
    base_position: np.ndarray    # [x, y, theta] in world frame
    base_velocity: np.ndarray    # [vx, vy, omega] for base.py / swerve drive
    lift_q: float                # lift height in metres
    left_arm_q: np.ndarray       # 7 joint positions (rad)
    right_arm_q: np.ndarray      # 7 joint positions (rad)

    # Convergence diagnostics
    left_pos_err: float
    left_ori_err: float
    right_pos_err: float
    right_ori_err: float
    solved: bool
    iters: int


# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

class WholeBodyIK:
    """
    Whole-body IK for YORv3 using mink (QP differential IK).

    The mobile base uses three planar joints (base_x, base_y, base_yaw), so
    mink optimises (x, y, theta) directly with no ground-lock task needed.
    Arms and lift are controlled as usual. Optional self-collision
    avoidance (a hard QP constraint) keeps the arms clear of the lift
    column, the chassis, and each other.

    Parameters
    ----------
    scene_xml : str, optional
        Path to the scene XML. Defaults to the in-repo
        ``description/scene_wholebody.xml``.
    config : WholeBodyIKConfig, optional
    """

    # Joint / actuator names
    _BASE_FREEJOINT  = "base_freejoint"
    _BASE_JOINTS     = ["base_x", "base_y", "base_yaw"]
    _LIFT_JOINT      = "Slider 7"
    # Shoulder / elbow / wrist bodies used by the elbow-swivel objective.
    # Verified against the model at the home keyframe: link1+link2 share an
    # origin at the shoulder, link3+link4 at the elbow, link5+link6 at the
    # wrist (upper arm 0.310 m, forearm 0.270 m on this robot).
    _SWIVEL_BODIES   = {"shoulder": "{side}_arm_link1",
                        "elbow":    "{side}_arm_link3",
                        "wrist":    "{side}_arm_link5"}
    _LEFT_JOINTS     = [f"left_arm_joint{i}"  for i in range(1, 8)]
    _RIGHT_JOINTS    = [f"right_arm_joint{i}" for i in range(1, 8)]
    _LEFT_EE_SITE    = "left_arm_ee"
    _RIGHT_EE_SITE   = "right_arm_ee"
    _LEFT_ACTUATORS  = [f"left_arm_joint{i}_pos"  for i in range(1, 8)]
    _RIGHT_ACTUATORS = [f"right_arm_joint{i}_pos" for i in range(1, 8)]
    _LIFT_ACTUATOR   = "lift_joint_pos"

    # Bodies whose group-3 collision spheres are used for self-collision
    # avoidance. The lift column carries a small sphere proxy and is paired
    # with every arm link. The chassis (base_link) has no usable sphere — its
    # only sphere is a 0.28 m bounding ball that engulfs the arm roots — so the
    # base is avoided via its convex mesh, paired with the *distal* arm links
    # only (proximal links sit 1-2 cm from the chassis at home and would lock).
    _BODY_COLLISION_LINKS = ["lift_slide_1"]
    _BASE_LINK            = "base_link"
    _LEFT_ARM_LINKS  = (
        [f"left_arm_link{i}"  for i in range(1, 8)]
        + ["left_arm_end_effector", "left_wuji_nero_mount"]
    )
    _RIGHT_ARM_LINKS = (
        [f"right_arm_link{i}" for i in range(1, 8)]
        + ["right_arm_end_effector", "right_wuji_nero_mount"]
    )
    # Distal links only — safe to avoid against the chassis (≥7.7 cm at home).
    _LEFT_DISTAL_LINKS  = (
        [f"left_arm_link{i}"  for i in (5, 6, 7)]
        + ["left_arm_end_effector", "left_wuji_nero_mount"]
    )
    _RIGHT_DISTAL_LINKS = (
        [f"right_arm_link{i}" for i in (5, 6, 7)]
        + ["right_arm_end_effector", "right_wuji_nero_mount"]
    )
    # Ground avoidance: floor plane vs arm spheres AND the hand's actual
    # collision meshes (palm + fingers). The finger meshes matter — fingertips
    # reach ~5 cm below the EE proxy sphere, so spheres alone would either
    # miss finger strikes or need a buffer so large the hand couldn't reach low.
    _FLOOR_GEOM = "floor"
    _HAND_BODIES = {
        side: [f"{side}_wuji_hand_orient"]
        + [f"{side}_finger{f}_link{l}" for f in range(1, 6) for l in range(1, 5)]
        for side in ("left", "right")
    }

    def __init__(
        self,
        scene_xml: Optional[str] = None,
        config: Optional[WholeBodyIKConfig] = None,
    ) -> None:
        self.config = config or WholeBodyIKConfig()
        self.scene_xml = str(Path(scene_xml or DEFAULT_SCENE).resolve())

        # ── Load model ───────────────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data  = mujoco.MjData(self.model)

        # ── Base / Arm / Lift DOF ids ────────────────────────────────────────
        self.base_dof_ids = [self.model.joint(j).dofadr[0] for j in self._BASE_JOINTS]
        self.lift_dof_id  = self.model.joint(self._LIFT_JOINT).dofadr[0]
        self.arm_dof_ids  = [self.model.joint(n).dofadr[0] for n in self._LEFT_JOINTS + self._RIGHT_JOINTS]
        # Per-arm DOF ids, for objectives that are defined one arm at a time
        # (elbow swivel, manipulability) rather than over the whole body.
        # Every DOF the whole-body solver is responsible for, in one array.
        self._ik_dof_ids = np.array(
            list(self.base_dof_ids) + [self.lift_dof_id] + list(self.arm_dof_ids))
        self._left_arm_dof_ids  = np.array(
            [self.model.joint(n).dofadr[0] for n in self._LEFT_JOINTS])
        self._right_arm_dof_ids = np.array(
            [self.model.joint(n).dofadr[0] for n in self._RIGHT_JOINTS])

        # qpos addresses
        self.base_qpos_adrs = np.array([int(self.model.joint(j).qposadr) for j in self._BASE_JOINTS])
        self._lift_qpos_adr = int(self.model.joint(self._LIFT_JOINT).qposadr)
        self._left_arm_qpos_adrs  = np.array([int(self.model.joint(n).qposadr) for n in self._LEFT_JOINTS])
        self._right_arm_qpos_adrs = np.array([int(self.model.joint(n).qposadr) for n in self._RIGHT_JOINTS])

        # Lift travel, taken from the model so the description stays the single
        # source of truth for it (currently 0 → 0.900 m).
        self.lift_range: tuple[float, float] = (
            float(self.model.joint(self._LIFT_JOINT).range[0]),
            float(self.model.joint(self._LIFT_JOINT).range[1]),
        )

        # ── Actuator IDs (for physics mode ctrl writing) ─────────────────────
        self.left_actuator_ids  = np.array([self.model.actuator(n).id for n in self._LEFT_ACTUATORS])
        self.right_actuator_ids = np.array([self.model.actuator(n).id for n in self._RIGHT_ACTUATORS])
        self.lift_actuator_id   = self.model.actuator(self._LIFT_ACTUATOR).id

        # ── mink configuration ───────────────────────────────────────────────
        self.configuration = mink.Configuration(self.model)

        # ── EE Tasks ─────────────────────────────────────────────────────────
        self.left_ee_task = mink.FrameTask(
            frame_name=self._LEFT_EE_SITE,
            frame_type="site",
            position_cost=self.config.ee_position_cost,
            orientation_cost=self.config.ee_orientation_cost,
            lm_damping=self.config.ee_lm_damping,
        )
        self.right_ee_task = mink.FrameTask(
            frame_name=self._RIGHT_EE_SITE,
            frame_type="site",
            position_cost=self.config.ee_position_cost,
            orientation_cost=self.config.ee_orientation_cost,
            lm_damping=self.config.ee_lm_damping,
        )

        # ── Posture task (regularises toward initial configuration) ──────────
        self.posture_task = mink.PostureTask(
            self.model, cost=self._build_posture_cost()
        )

        # ── Base fix task ────────────────────────────────────────────────────
        bf = np.zeros(self.model.nv)
        for i in self.base_dof_ids:
            bf[i] = self.config.base_damping_cost
        self.base_fix_task = mink.DampingTask(self.model, bf)
        lf = np.zeros(self.model.nv)
        lf[self.lift_dof_id] = self.config.lift_damping_cost
        self.lift_fix_task = mink.DampingTask(self.model, lf)

        # ── Limits ───────────────────────────────────────────────────────────
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model, self._build_velocity_limits()),
        ]

        # ── Self-collision avoidance (hard constraint, runtime-toggleable) ────
        # Built once and kept aside; _solve_qp adds it to the active limits only
        # while `avoid_collisions` is True, so it can be toggled live like fix_base.
        self.collision_limit = self._build_collision_limit()
        self.n_collision_pairs = (
            len(self.collision_limit.geom_id_pairs) if self.collision_limit else 0
        )
        self.avoid_collisions = bool(
            self.config.enable_collision_avoidance and self.collision_limit is not None
        )
        # Precomputed limit lists so _solve_qp doesn't allocate per call: the
        # hardware path may call it max_iters times on every 108 Hz tick.
        self._limits_with_collision = (
            self.limits + [self.collision_limit] if self.collision_limit else self.limits
        )

        # ── State ────────────────────────────────────────────────────────────
        # dls_projector secondary-objective state.
        # _prev_vel is the velocity the last solve() actually applied (after
        # fix_base/fix_lift zeroing), used by the continuity objective.
        # _swivel_target latches per side so the arm keeps the elbow branch
        # it started in; None means "latch on next solve".
        self._prev_vel: Optional[np.ndarray] = None
        self._swivel_target: dict[str, Optional[float]] = {
            side: self.config.elbow_swivel_targets.get(side)
            for side in ("left", "right")
        }
        # [S5a] Parallel-transported swivel reference, per side: the in-plane
        # reference vector and the shoulder->wrist axis it was last valid
        # for. None = seed from the discrete convention on next use.
        self._swivel_ref: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._swivel_u_prev: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        # [S5b] When each side's swivel error first exceeded the re-latch
        # threshold (monotonic time), or None while below it.
        self._swivel_err_since: dict[str, Optional[float]] = {"left": None, "right": None}
        # [S1] Home-keyframe arm posture (14 values, left 1-7 then right
        # 1-7), latched by init_from_keyframe/init_from_qpos.
        self._home_arm_q: Optional[np.ndarray] = None
        self._recenter_offset_body: Optional[np.ndarray] = None
        self._shoulder_yaw_home: Optional[np.ndarray] = None
        self._arm_qpos_adrs_all = np.concatenate(
            [self._left_arm_qpos_adrs, self._right_arm_qpos_adrs])
        # The two shoulder-yaw joints, whose drift from the latched values
        # is the error base_recenter_yaw_gain hands back to the chassis.
        self._shoulder_yaw_qadrs = np.array(
            [self._left_arm_qpos_adrs[0], self._right_arm_qpos_adrs[0]])
        self._arm_dof_ids_arr = np.asarray(self.arm_dof_ids)
        # [S7] Published per-solve diagnostics; {} when disabled or before
        # the first solve. `_last_swivel` is written by the swivel objective
        # (phi, target, err); `_last_ineq_collision_rows` by
        # _limit_inequalities.
        self.diagnostics: dict = {}
        self._last_swivel: dict[str, tuple] = {}
        self._last_ineq_collision_rows = 0
        # Scratch MjData for finite-difference gradients, so perturbing a
        # configuration never touches the live one.
        self._fd_data = mujoco.MjData(self.model)
        # Reused MuJoCo Jacobian buffers -- these are filled on every swivel
        # row, several times per solver iteration.
        self._jac_buf_p = np.zeros((3, self.model.nv))
        self._jac_buf_r = np.zeros((3, self.model.nv))

        self.initialized  = False
        self.fix_base     = False
        self.fix_lift     = False

    # ── Public API ───────────────────────────────────────────────────────────

    def init_from_keyframe(self, key_name: str = "home") -> None:
        """Reset to named keyframe and sync IK configuration."""
        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.model.key(key_name).id
        )
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        self._reset_nullspace_state()
        self._latch_recenter_offset()
        self._latch_home_arm_posture()
        self.initialized = True

    def init_from_qpos(self, qpos: np.ndarray) -> None:
        """Initialise from an arbitrary full qpos vector."""
        self.configuration.update(qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        self._reset_nullspace_state()
        self._latch_recenter_offset()
        self._latch_home_arm_posture()
        self.initialized = True

    def _latch_home_arm_posture(self) -> None:
        """Record the arm posture the [S1] home attractor pulls toward.

        Latched at init like the recenter offset: on hardware both init
        paths run right after the arms physically home, so this is the
        known-comfortable configuration -- exactly what "recover from a
        weird pose" should mean.
        """
        self._home_arm_q = self.configuration.q[self._arm_qpos_adrs_all].copy()

    def _latch_recenter_offset(self) -> None:
        """Record where the hands naturally sit relative to the chassis.

        The recentering objective must not drive the base *under* the hands
        -- at a comfortable posture the hands sit well in front of the
        chassis. It drives the base to wherever restores this offset,
        expressed in the base frame so it rotates with the chassis.

        The shoulder-yaw values the yaw half of recentering restores
        (base_recenter_yaw_gain) are latched here too, so both halves of
        "where do the hands belong relative to the chassis" come from the
        same measurement and can never disagree about what home means.
        """
        T_l, T_r = self.forward_kinematics()
        mid_xy = 0.5 * (T_l.translation()[:2] + T_r.translation()[:2])
        base_xy = self.configuration.q[self.base_qpos_adrs[:2]]
        yaw = float(self.configuration.q[self.base_qpos_adrs[2]])
        c, s = np.cos(yaw), np.sin(yaw)
        world = mid_xy - base_xy
        self._recenter_offset_body = np.array(
            [c * world[0] + s * world[1], -s * world[0] + c * world[1]])
        self._shoulder_yaw_home = self.configuration.q[
            self._shoulder_yaw_qadrs].copy()

    def update_configuration(self, qpos: np.ndarray) -> None:
        """Sync IK with current robot state (call each control cycle)."""
        self.configuration.update(qpos)

    def set_measured_state(
        self,
        left_q: Optional[np.ndarray] = None,
        right_q: Optional[np.ndarray] = None,
        lift: Optional[float] = None,
        base: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Overwrite the measured DOFs of the IK configuration, keep the rest.

        This is the hardware counterpart of ``update_configuration``: on the
        real robot only some DOFs are observable (arm encoders, lift height,
        base odometry), while the rest of ``qpos`` — hands, wheels, steering —
        has no bearing on the IK and is left at whatever the model holds.

        Any argument left as ``None`` keeps its current value, so a missing
        sensor (e.g. the lift returning ``None``) degrades to open-loop on
        that DOF instead of zeroing it.

        Returns the qpos vector that was applied.
        """
        q = self.configuration.q.copy()
        if left_q is not None:
            q[self._left_arm_qpos_adrs] = np.asarray(left_q, dtype=float)
        if right_q is not None:
            q[self._right_arm_qpos_adrs] = np.asarray(right_q, dtype=float)
        if lift is not None:
            q[self._lift_qpos_adr] = self.clamp_lift(lift)
        if base is not None:
            q[self.base_qpos_adrs] = np.asarray(base, dtype=float)
        self.configuration.update(q)
        return q

    def clamp_lift(self, height: float) -> float:
        """Clamp a lift height to the travel declared by the description."""
        return float(np.clip(height, self.lift_range[0], self.lift_range[1]))

    def toggle_fix_base(self, fix: Optional[bool] = None) -> bool:
        self.fix_base = (not self.fix_base) if fix is None else fix
        return self.fix_base

    def toggle_fix_lift(self, fix: Optional[bool] = None) -> bool:
        self.fix_lift = (not self.fix_lift) if fix is None else fix
        return self.fix_lift

    def toggle_collision_avoidance(self, enable: Optional[bool] = None) -> bool:
        """Enable/disable self-collision avoidance live. No-op if no limit built."""
        if self.collision_limit is None:
            return False
        self.avoid_collisions = (
            (not self.avoid_collisions) if enable is None else enable
        )
        return self.avoid_collisions

    def solve(
        self,
        T_left: mink.SE3,
        T_right: mink.SE3,
        lift_target: Optional[float] = None,
    ) -> WholeBodyIKResult:
        """
        Solve whole-body IK for given EE targets.
        """
        if not self.initialized:
            raise RuntimeError("Call init_from_keyframe() first.")

        self.left_ee_task.set_target(T_left)
        self.right_ee_task.set_target(T_right)

        if self.config.refresh_posture_target:
            # Minimum-norm null-space regularizer: reference wherever we
            # currently are, every solve, rather than a fixed point from
            # init -- see the field's docstring in WholeBodyIKConfig.
            q_ref = self.configuration.q.copy()
            if lift_target is not None:
                q_ref[self._lift_qpos_adr] = self.clamp_lift(lift_target)
            self.posture_task.set_target(q_ref)
        elif lift_target is not None:
            q_ref = self.configuration.q.copy()
            q_ref[self._lift_qpos_adr] = self.clamp_lift(lift_target)
            self.posture_task.set_target(q_ref)

        ee_tasks = [self.left_ee_task, self.right_ee_task]
        other_tasks = [self.posture_task]
        if self.fix_base:
            other_tasks.append(self.base_fix_task)
        if self.fix_lift:
            other_tasks.append(self.lift_fix_task)

        l_pos = l_ori = r_pos = r_ori = np.inf
        iters = 0
        prev_base_q = self.configuration.q[self.base_qpos_adrs]
        # [S7] Stale swivel numbers must not outlive the tick that made
        # them: a faded-out swivel objective leaves no sample, and that
        # absence is itself the datum (arm straight / objective off).
        self._last_swivel = {}

        for iters in range(1, self.config.max_iters + 1):
            self._first_iteration = iters == 1
            if self._first_iteration:
                self._base_weight_cached = None
            vel = self._solve_qp(ee_tasks, other_tasks)
            # Arm-only hardware mode needs exact base and lift locks. High-cost
            # damping tasks alone still permit small virtual motion when an EE
            # task competes with them, even though dispatch to the physical
            # actuators is disabled. Zeroing these solved velocities means only
            # arm joints are integrated, keeping the IK model consistent with
            # the stationary chassis and column.
            if self.fix_base:
                vel[self.base_dof_ids] = 0.0
            if self.fix_lift:
                vel[self.lift_dof_id] = 0.0
            self.configuration.integrate_inplace(vel, self.config.dt)
            # Velocity actually applied this iteration (post base/lift
            # zeroing). dls_projector's continuity objective compares the
            # next solve() against this, so it must be what was applied,
            # not the raw solver output.
            self._prev_vel = vel.copy()

            l_err = self.left_ee_task.compute_error(self.configuration)
            r_err = self.right_ee_task.compute_error(self.configuration)
            l_pos = float(np.linalg.norm(l_err[:3]))
            l_ori = float(np.linalg.norm(l_err[3:]))
            r_pos = float(np.linalg.norm(r_err[:3]))
            r_ori = float(np.linalg.norm(r_err[3:]))

            if (l_pos <= self.config.pos_threshold and
                    l_ori <= self.config.ori_threshold and
                    r_pos <= self.config.pos_threshold and
                    r_ori <= self.config.ori_threshold):
                break

        q = self.configuration.q

        # Extract components
        curr_base_q = q[self.base_qpos_adrs]
        base_vel = (curr_base_q - prev_base_q) / self.config.dt

        left_arm_q  = q[self._left_arm_qpos_adrs]
        right_arm_q = q[self._right_arm_qpos_adrs]
        lift_q      = float(q[self._lift_qpos_adr])

        solved = (l_pos <= self.config.pos_threshold and
                  l_ori <= self.config.ori_threshold and
                  r_pos <= self.config.pos_threshold and
                  r_ori <= self.config.ori_threshold)

        # [S7] Per-solve diagnostics: enough to attribute a "stuck" event to
        # its mechanism offline -- singularity (sigma_min), reach limit
        # (manipulability), a fought swivel branch (swivel_err), or an
        # active constraint (collision_rows). Costs two 6x7 SVDs.
        if self.config.record_solver_diagnostics:
            data = self.configuration.data
            diag: dict = {"collision_rows": int(self._last_ineq_collision_rows)}
            for side in ("left", "right"):
                J_arm = self._arm_jacobian(side, data)
                svals = np.linalg.svd(J_arm, compute_uv=False)
                diag[f"{side}_sigma_min"] = float(svals[-1])
                diag[f"{side}_manip"] = float(
                    np.exp(self._log_manipulability(J_arm)))
                swv = self._last_swivel.get(side)
                diag[f"{side}_swivel"] = None if swv is None else float(swv[0])
                diag[f"{side}_swivel_target"] = None if swv is None else float(swv[1])
                diag[f"{side}_swivel_err"] = None if swv is None else float(swv[2])
            self.diagnostics = diag
        else:
            self.diagnostics = {}

        return WholeBodyIKResult(
            q=q.copy(),
            base_position=curr_base_q,
            base_velocity=base_vel,
            lift_q=lift_q,
            left_arm_q=left_arm_q.copy(),
            right_arm_q=right_arm_q.copy(),
            left_pos_err=l_pos, left_ori_err=l_ori,
            right_pos_err=r_pos, right_ori_err=r_ori,
            solved=solved,
            iters=iters,
        )

    # ── Convenience ──────────────────────────────────────────────────────────

    def forward_kinematics(self) -> tuple[mink.SE3, mink.SE3]:
        return (
            self.configuration.get_transform_frame_to_world(self._LEFT_EE_SITE,  "site"),
            self.configuration.get_transform_frame_to_world(self._RIGHT_EE_SITE, "site"),
        )

    def apply_to_sim_kinematic(self, data: mujoco.MjData, result: WholeBodyIKResult) -> None:
        """Kinematic mode: write full qpos to data, call mj_forward yourself."""
        data.qpos[:] = result.q

    def apply_to_ctrl(self, data: mujoco.MjData, result: WholeBodyIKResult) -> None:
        """Physics mode: write position targets to ctrl (requires well-tuned gains)."""
        data.ctrl[self.left_actuator_ids]  = result.left_arm_q
        data.ctrl[self.right_actuator_ids] = result.right_arm_q
        data.ctrl[self.lift_actuator_id]   = result.lift_q

    # ── Private helpers ───────────────────────────────────────────────────────

    def _solve_qp(self, ee_tasks: list, other_tasks: list) -> np.ndarray:
        """Dispatch to the configured redundancy-resolution strategy. See
        WholeBodyIKConfig.redundancy_resolution for what each mode does and
        why; all three return an nv-length velocity in the same convention
        (v = Δq / dt), so solve()'s integration loop doesn't need to know
        which one ran."""
        mode = self.config.redundancy_resolution
        if mode == "soft":
            return self._solve_qp_soft(ee_tasks + other_tasks)
        elif mode == "dls_projector":
            return self._solve_qp_dls_projector(ee_tasks)
        else:
            raise ValueError(
                f"Unknown redundancy_resolution {mode!r}; expected "
                "'soft' or 'dls_projector'."
            )

    def _solve_qp_soft(self, tasks: list) -> np.ndarray:
        """Default. Every task (EE + posture + fix) competes as a soft cost
        in one QP -- unchanged from the solver's original behavior."""
        solver = self.config.solver
        limits = self._limits_with_collision if self.avoid_collisions else self.limits
        try:
            return mink.solve_ik(
                self.configuration, tasks,
                self.config.dt, solver=solver,
                damping=1e-5, limits=limits,
            )
        except Exception:
            for fb in self.config.fallback_solvers:
                try:
                    return mink.solve_ik(
                        self.configuration, tasks,
                        self.config.dt, solver=fb,
                        damping=1e-5, limits=limits,
                    )
                except Exception:
                    continue
            raise RuntimeError(
                f"All solvers failed: {[solver] + self.config.fallback_solvers}"
            )

    def _solve_qp_dls_projector(self, ee_tasks: list) -> np.ndarray:
        """Damped-least-squares pseudoinverse with an *optimised* null space.

            q̇ = q̇_primary + N z,   q̇_primary = J⁺b,   J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹,
            N = I - J⁺J

        The primary term tracks both EE targets. `z` is then chosen by a
        weighted least-squares over the secondary objectives below, rather
        than by projecting a single hand-built gradient step -- so they trade
        off against each other properly instead of one silently winning:

          * posture      -- pull toward the posture reference (as before)
          * continuity   -- ||q̇ - q̇_prev||, damping tick-to-tick changes in
                            null-space motion
          * elbow swivel -- one scalar task per arm holding the elbow at a
                            chosen angle about the shoulder->wrist axis, which
                            is what stops elbow-up/down branch flipping
          * manipulability (optional, gated) -- ascend log mu near singularities

        Each contributes a residual that is affine in q̇, stacked into
        `A z ≈ r` and solved as z = (AᵀA + λ_z I)⁻¹Aᵀr. AᵀA is only n×n
        (n = free DOFs, 14 with base and lift fixed) so this is cheap. λ_z
        (`nullspace_regularization`) is what makes it well posed: every block
        carries N on the right, so any component of z in null(N) affects
        nothing and is regularised to zero.

        With continuity/swivel/manipulability weights at 0 and posture weight
        1 this reduces to the previous q̇ = q̇_primary + N q̇_posture, because
        NᵀN = N and N q̇_primary = 0 *for an exact projector*. That identity
        needs N idempotent, which only holds as dls_damping → 0: with damping
        the eigenvalues of J⁺J are σ²/(σ²+λ²) < 1, so N is not idempotent and
        N q̇_primary ≠ 0. The two therefore differ slightly at the damping
        actually in use, and this least-squares form is the more correct of
        the two -- it minimises what it claims to minimise, where the
        classical formula assumes an exactness the damped inverse does not
        have. tests/test_nullspace_objectives.py pins the reduction in the
        λ → 0 limit.

        Note dim(null(J)) is 2 with both arms tracking and base/lift fixed --
        one swivel DOF per arm -- so these objectives share very little room.
        A strict priority hierarchy would starve everything below the first
        level; the ordering asked for (tracking ≫ continuity/swivel/posture ≫
        gated manipulability) is expressed through the relative weights and
        the manipulability gate, with EE tracking kept strictly first by the
        projector itself.

        As in every mode, the solve runs over only the DOFs that are free to
        move: fix_base / fix_lift drop the base / lift columns from J before
        the pseudoinverse, and the result is scattered back into a full-nv
        vector with those entries left at zero. See the git history and
        artifacts/wholebody_logs/posture_fix_commands.md for why -- solving
        over locked DOFs sent 77% of every step into a base that solve()
        then zeroed, crippling convergence.

        Joint/collision limits are not part of this solve; the result is
        projected onto them afterward by `_project_onto_limits`.
        """
        nv = self.model.nv

        # Only the DOFs this solver actually controls: base (3), lift (1),
        # arms (14). The model carries ~62 nv in total -- fingers, wheels,
        # steering -- and every one of those is a zero column in the EE
        # Jacobian, so they contribute nothing to the primary term but do
        # land in the null space, where the secondary objectives would push
        # them around, and they inflate every matrix here from 18x18 to
        # 62x62. Excluding them also makes dim(null(J)) actually equal the
        # 2 redundant arm DOFs it is reasoned about as being.
        free = np.zeros(nv, dtype=bool)
        free[self._ik_dof_ids] = True
        if self.fix_base:
            free[self.base_dof_ids] = False
        if self.fix_lift:
            free[self.lift_dof_id] = False
        free_ids = np.flatnonzero(free)
        n = free_ids.size

        J_rows, b_rows = [], []
        for task in ee_tasks:
            Jt = task.compute_jacobian(self.configuration)
            bt = -task.gain * task.compute_error(self.configuration)
            if self.config.dls_task_weighting:
                # [S4a] The same row scaling mink's _weighted_residual applies
                # in "soft" mode: task.cost is [position_cost x3,
                # orientation_cost x3], so this is what makes
                # ee_position_cost / ee_orientation_cost mean something in
                # dls mode. Note it rescales the whole stack relative to
                # lambda, so A/B it with the damping in mind.
                Jt = task.cost[:, None] * Jt
                bt = task.cost * bt
            J_rows.append(Jt)
            b_rows.append(bt)
        J = np.vstack(J_rows)[:, free_ids]
        b = np.concatenate(b_rows)

        # One SVD serves both terms, and they must NOT use the same inverse:
        #
        #   primary   -- damped (singularity-robust):
        #                q̇_p = V diag(σ/(σ²+λ²)) Uᵀ b  ==  Jᵀ(JJᵀ+λ²I)⁻¹b
        #   null space -- exact orthogonal projector onto null(J), from the
        #                right singular vectors of the numerically-nonzero
        #                singular values: N = I - V_r V_rᵀ
        #
        # Building N from the *damped* inverse instead (N = I - J⁺J with the
        # damped J⁺) is wrong and quietly destroys EE tracking: that N is not
        # idempotent, and in poorly-conditioned directions it approaches the
        # identity, so the secondary term's -N q̇_p component cancels the
        # primary motion it is supposed to leave alone. With the exact
        # projector, N q̇_p = 0 holds identically -- q̇_p lies in range(Jᵀ),
        # which N annihilates -- so no secondary objective can ever perturb
        # the EE task, which is the whole point of the priority ordering.
        U, sv, Vt = np.linalg.svd(J, full_matrices=False)

        # ── Base motion cost ─────────────────────────────────────────────────
        # The damped inverse above treats every free DOF as equally cheap, and
        # the base has by far the largest leverage on an end effector -- a
        # chassis translation moves both hands one-for-one. So the minimum-norm
        # solution reaches for the base first, for everything, including noise:
        # measured against 2 mm of pure EE jitter, 24% of the response went to
        # the base. That is what produced the reversals, and it contradicts
        # what this controller is documented to do -- roll the chassis only
        # when the arms and lift together cannot reach.
        #
        # Weighting fixes it at the source. Minimising ||W q̇||² instead of
        # ||q̇||² subject to the same task makes base motion expensive relative
        # to arm motion, so the arms absorb high-frequency error and the base
        # moves only once they genuinely run out. Implemented by scaling the
        # columns of J: with y = W q̇, solving in y and mapping back gives the
        # weighted damped inverse exactly.
        #
        # Weight 1.0 is the unweighted solve, so this is off by default in the
        # arithmetic sense -- Jw is J and the SVD below is the same one.
        # x/y carry the gated base_motion_weight; yaw has its own cost so
        # "rotate the chassis" can be cheaper than "translate the chassis"
        # without being cheaper than nothing -- see base_motion_weight_yaw.
        weight = np.ones(n)
        base_weight = self._gated_base_weight()
        yaw_weight = float(self.config.base_motion_weight_yaw)
        if not self.fix_base:
            slots_xy = np.flatnonzero(np.isin(free_ids, self.base_dof_ids[:2]))
            slot_yaw = np.flatnonzero(np.isin(free_ids, self.base_dof_ids[2:3]))
            weight[slots_xy] = max(base_weight, 1e-6)
            weight[slot_yaw] = max(yaw_weight, 1e-6)

        if np.all(weight == 1.0):
            Uw, svw, Vtw = U, sv, Vt        # reuse the SVD taken for N
            Jw = J
        else:
            Jw = J / weight                 # J @ diag(1/weight)
            Uw, svw, Vtw = np.linalg.svd(Jw, full_matrices=False)

        lam2 = self.config.dls_damping ** 2
        sig_thresh = float(self.config.dls_adaptive_damping_sigma)
        if sig_thresh > 0.0 and svw.size:
            # [S4b] Ramp lambda^2 from dls_damping^2 (sigma_min >= threshold)
            # up to dls_damping_max^2 (sigma_min -> 0), smoothstepped so the
            # damping itself does not step. svw is sorted descending.
            x = float(np.clip(1.0 - svw[-1] / sig_thresh, 0.0, 1.0))
            gate = x * x * (3.0 - 2.0 * x)
            lam2 = lam2 + gate * (self.config.dls_damping_max ** 2 - lam2)

        y = Vtw.T @ ((svw / (svw ** 2 + lam2)) * (Uw.T @ b))
        dq_primary = y / weight

        # [S3] Constraint-aware primary: replace the unconstrained damped
        # step with the minimiser of the *same* objective,
        #     min ||J dq - b||^2 + lam2 ||W dq||^2   s.t.  G dq <= h,
        # so that when a direction is blocked (collision buffer, joint
        # limit) the optimiser reroutes through whatever free DOFs still
        # serve the task -- base and lift included -- instead of the
        # post-hoc clip zeroing the motion. With no constraint rows active
        # the QP minimiser IS the damped step above, so the gate only
        # changes behaviour near limits. Any failure falls back to the
        # already-computed unconstrained step.
        ineq_cache = None
        if self.config.constrained_primary:
            ineq_cache = self._limit_inequalities()
            if ineq_cache is not None:
                G_full, h_ineq = ineq_cache
                P = J.T @ J + lam2 * np.diag(weight ** 2)
                q_lin = -(J.T @ b)
                problem = qpsolvers.Problem(P, q_lin, G_full[:, free_ids], h_ineq)
                for solver in [self.config.solver] + self.config.fallback_solvers:
                    try:
                        res = qpsolvers.solve_problem(problem, solver=solver)
                        if res.found:
                            dq_primary = res.x
                            break
                    except Exception:
                        continue

        rank_tol = self.config.nullspace_rank_tol * (sv[0] if sv.size else 0.0)
        V_r = Vt[sv > rank_tol].T
        N = np.eye(n) - V_r @ V_r.T

        blocks_A: list[np.ndarray] = []
        blocks_r: list[np.ndarray] = []

        def add_velocity_objective(weight: float, desired: np.ndarray) -> None:
            """Residual for 'q̇ should be close to `desired`', weight >= 0."""
            if weight <= 0.0:
                return
            w = float(np.sqrt(weight))
            blocks_A.append(w * N)
            blocks_r.append(w * (desired - dq_primary))

        # ── Base motion, in the null space too ───────────────────────────────
        # Weighting the primary solve alone does not work, and the reason is
        # instructive: the base has two independent routes to the same motion.
        # Measured against pure 2 mm EE noise, a 100x primary weight moved
        # |base| from 0.0923 to 0.0848, and switching off the posture and
        # swivel objectives moved it to 0.0903 -- block either path and the
        # other simply supplies it. Both at once gives 0.0002.
        #
        # So the preference has to be stated here as well: among all solutions
        # that serve the end effectors equally, prefer the one that moves the
        # base least. That is exactly what the null space is for -- N z cannot
        # change the EE task, so trading base motion for arm motion through it
        # is free. When the arms genuinely cannot reach, the primary term
        # supplies base motion that no amount of null-space preference can
        # cancel, which is the behaviour this controller documents.
        #
        # Split per axis for the same reason the primary weight is: if the
        # null-space term stayed uniform it would re-impose on yaw exactly
        # the cost the primary term just relaxed -- the two-routes problem
        # above, applied to the fix for it.
        #
        # Yaw takes base_yaw_hold_weight, NOT the primary's
        # base_motion_weight_yaw: the two roles are independent, and tying
        # them together left yaw unanchored at every default setting. See
        # base_yaw_hold_weight.
        #
        if not self.fix_base:
            for dof_ids, w_val in (
                    (self.base_dof_ids[:2], base_weight),
                    (self.base_dof_ids[2:3],
                     float(self.config.base_yaw_hold_weight))):
                if w_val > 1.0:
                    slots = np.flatnonzero(np.isin(free_ids, dof_ids))
                    if slots.size:
                        w = float(np.sqrt(w_val - 1.0))
                        blocks_A.append(w * N[slots, :])
                        blocks_r.append(w * (0.0 - dq_primary[slots]))

        # ── Base recentering, x/y, always on ─────────────────────────────────
        # Continuously prefer rolling the chassis toward under the hand
        # targets -- see base_recenter_gain. Proportional to the offset
        # error, so it is self-attenuating (desire -> 0 as error -> 0) and
        # needs no explicit gate.
        #
        # It USED to be scaled by the manipulability-gate "openness" that
        # also drives base_weight (item 1) -- reasoning that recentering
        # should only matter when the arms are genuinely reach-limited. That
        # was wrong: manipulability recovers as the arms retract, so on a
        # reach-then-retract motion the gate closed while the base was still
        # short of home, stranding it there -- the last part of "bring the
        # hand back" then fell to the arm folding and the lift, exactly the
        # 2026-08-25 hardware complaint. Manipulability answers "is the arm
        # stretched"; it does not answer "is the base still displaced",
        # which is what this term needs. The offset error already answers
        # that on its own, symmetrically in both directions.
        #
        # A separate objective from the hold-still term above (own weight,
        # not folded into base_weight's) because it needs to win not just
        # against hold-still but against the posture term's penalty on the
        # arms' counter-motion, which is the dominant attenuation.
        # First iteration only, like every velocity-shaped term, so the
        # per-tick desire is not multiplied by the iteration count.
        hand_xy = [
            task.transform_target_to_world.translation()[:2]
            for task in (self.left_ee_task, self.right_ee_task)
            if task.transform_target_to_world is not None
        ]
        gain = float(self.config.base_recenter_gain)
        if (gain > 0.0 and hand_xy and self._first_iteration
                and not self.fix_base
                and self._recenter_offset_body is not None):
            base_xy = self.configuration.q[self.base_qpos_adrs[:2]]
            # Where the base would sit if the hands kept their
            # home-pose offset from the chassis (offset latched at
            # init, rotated by the current yaw) -- NOT under the
            # hands, which would pull forward even at rest.
            yaw = float(self.configuration.q[self.base_qpos_adrs[2]])
            c, s = np.cos(yaw), np.sin(yaw)
            off = self._recenter_offset_body
            off_world = np.array([c * off[0] - s * off[1],
                                  s * off[0] + c * off[1]])
            goal_xy = np.mean(hand_xy, axis=0) - off_world
            v = gain * (goal_xy - base_xy)
            speed = float(np.linalg.norm(v))
            cap = float(self.config.base_recenter_max_vel)
            if 0.0 < cap < speed:
                v *= cap / speed
            dq_recenter = v * self.config.dt
            slots = np.flatnonzero(
                np.isin(free_ids, self.base_dof_ids[:2]))
            if slots.size and dq_recenter.any():
                w = float(np.sqrt(self.config.base_recenter_weight))
                order = np.searchsorted(
                    np.asarray(self.base_dof_ids[:2]), free_ids[slots])
                blocks_A.append(w * N[slots, :])
                blocks_r.append(
                    w * (dq_recenter[order] - dq_primary[slots]))

        # ── Base recentering, yaw ────────────────────────────────────────────
        # The rotational twin of the block above: among all solutions that
        # serve the end effectors equally, prefer the one that hands the
        # shoulders' accumulated yaw back to the chassis. See
        # base_recenter_yaw_gain for why the error is the shoulder-yaw load
        # and not the hands' bearing -- the bearing version cancels the yaw
        # the primary solve spends on reach, because yaw is the cheap DOF
        # there and a null-space preference against a cheap DOF is a veto.
        #
        # Sign: base yaw and both shoulder-yaw joints trade one-for-one in
        # the same sense with the hands held fixed, so turning the chassis
        # by +d unwinds a +d load. The gain absorbs the ~1.1 rad/rad of
        # that exchange; it is not a separate calibration.
        ygain = float(self.config.base_recenter_yaw_gain)
        if (ygain > 0.0 and self._first_iteration and not self.fix_base
                and self._shoulder_yaw_home is not None):
            slots = np.flatnonzero(np.isin(free_ids, self.base_dof_ids[2:3]))
            if slots.size:
                load = float(np.mean(
                    self.configuration.q[self._shoulder_yaw_qadrs]
                    - self._shoulder_yaw_home))
                # Dead zone: an ordinary working posture is not a fault, so
                # ignore drift below it. Subtracted, not compared, so the
                # desire enters continuously at the edge rather than
                # stepping to gain * deadzone. See the config field.
                dz = float(self.config.base_recenter_yaw_deadzone)
                if dz > 0.0:
                    load = float(np.sign(load) * max(abs(load) - dz, 0.0))
                wz = ygain * load
                ycap = float(self.config.base_recenter_yaw_max_vel)
                if ycap > 0.0:
                    wz = float(np.clip(wz, -ycap, ycap))
                dq_yaw = wz * self.config.dt
                if dq_yaw != 0.0:
                    w = float(np.sqrt(self.config.base_recenter_yaw_weight))
                    blocks_A.append(w * N[slots, :])
                    blocks_r.append(w * (dq_yaw - dq_primary[slots]))

        # ── [S1] Arm home attractor, always on when gated in ─────────────────
        # The recovery force refresh_posture_target=True removed: among all
        # solutions that serve the end effectors equally, prefer the one
        # that moves the ARM joints toward the home posture. Null-space
        # only, so it cannot perturb EE tracking; proportional to the
        # error and capped per joint, so it is self-attenuating near home
        # and bounded far from it. First iteration only, like every
        # velocity-shaped term. Interplay to know about: the achieved creep
        # is a weighted fraction of the desire (it trades against the
        # min-norm posture term), and it must exceed the arm dispatch
        # deadband (arm_joint_deadband_rad) to reach the hardware at all --
        # pair this gate with the 0.005 deadband when testing.
        hgain = float(self.config.nullspace_home_gain)
        if hgain > 0.0 and self._first_iteration and self._home_arm_q is not None:
            err_home = self._home_arm_q - self.configuration.q[self._arm_qpos_adrs_all]
            cap = float(self.config.nullspace_home_max_vel)
            v_home = np.clip(hgain * err_home, -cap, cap)
            dq_home_full = np.zeros(nv)
            dq_home_full[self._arm_dof_ids_arr] = v_home * self.config.dt
            slots = np.flatnonzero(np.isin(free_ids, self._arm_dof_ids_arr))
            if slots.size:
                w = float(np.sqrt(self.config.nullspace_home_weight))
                blocks_A.append(w * N[slots, :])
                blocks_r.append(
                    w * (dq_home_full[free_ids[slots]] - dq_primary[slots]))

        # ── Posture (unchanged objective, now weighted alongside the rest) ──
        posture_error = self.posture_task.compute_error(self.configuration)
        dq_posture = (
            -self.posture_task.gain * self.posture_task.cost * posture_error
        )[free_ids]
        add_velocity_objective(self.config.nullspace_posture_weight, dq_posture)

        # ── Velocity continuity ──────────────────────────────────────────────
        # Units: this solve works in per-iteration displacement Δq, not
        # velocity, so the stored velocity is scaled by dt before being
        # compared against one. (Mixing them makes the objective 1/dt times
        # too strong -- 30x at 30 Hz -- and drives the solve straight to the
        # velocity limit.)
        #
        # Targeting q̇_prev and targeting N q̇_prev are the same thing here:
        # the residual is N z - (q̇_prev - q̇_p), and N q̇_p = 0 identically for
        # the exact projector above, so the primary component drops out on
        # its own. No need to project the target first.
        #
        # OFF by default, on measurement. ||q̇ - q̇_prev||² is a *momentum*
        # term, not a damping one: it resists change, which also means it
        # perpetuates whatever null-space velocity already exists rather than
        # letting posture bleed it off. Swept against a reversing target with
        # an abrupt disturbance, raising it made every metric worse at every
        # swivel weight -- e.g. at swivel 1.0, weight 0 -> 0.5 -> 2.0 took
        # null-space jerk 0.264 -> 0.279 -> 0.292, null-space speed
        # 0.768 -> 0.838 -> 0.928 rad/s, and elbow drift 8.2° -> 8.8° -> 11.6°.
        # The swivel objective is the effective anchor; this is kept as a
        # tunable because the trade-off may differ on hardware, where command
        # smoothness matters in ways a kinematic replay does not show.
        if self._prev_vel is not None:
            add_velocity_objective(
                self.config.nullspace_continuity_weight,
                self._prev_vel[free_ids] * self.config.dt,
            )

        # ── Elbow swivel, one scalar row per arm ─────────────────────────────
        swivel_weight = self.config.nullspace_swivel_weight
        if swivel_weight > 0.0:
            for side in ("left", "right"):
                row = self._swivel_row(side, free_ids)
                if row is None:
                    continue
                jac_phi, phi, offset = row
                # Fade out as the arm straightens and phi loses meaning.
                min_off = max(self.config.elbow_swivel_min_offset, 1e-9)
                fade = float(np.clip(offset / min_off, 0.0, 1.0))
                if fade <= 0.0:
                    continue
                if self._swivel_target[side] is None:
                    # First solve since a reset: keep the elbow branch we are
                    # already in rather than choosing one.
                    self._swivel_target[side] = phi
                err = float(np.arctan2(
                    np.sin(phi - self._swivel_target[side]),
                    np.cos(phi - self._swivel_target[side]),
                ))
                # [S5b] Re-latch on a persistently lost branch: fighting a
                # near-pi error through a region the constraints may not
                # allow crossing is how the arm wedges. Accepting the
                # branch it actually reached is the recovery. Timed on the
                # first iteration only, so the dwell is wall-clock, not
                # iteration count.
                relatch = float(self.config.swivel_relatch_err_rad)
                if relatch > 0.0 and self._first_iteration:
                    if abs(err) > relatch:
                        now = time.monotonic()
                        since = self._swivel_err_since[side]
                        if since is None:
                            self._swivel_err_since[side] = now
                        elif now - since >= self.config.swivel_relatch_after_s:
                            print(f"[wholebody-ik] {side} elbow swivel "
                                  f"re-latched: err {err:+.2f} rad held "
                                  f"{now - since:.1f} s, accepting the "
                                  f"current branch ({phi:+.2f} rad)")
                            self._swivel_target[side] = phi
                            self._swivel_err_since[side] = None
                            err = 0.0
                    else:
                        self._swivel_err_since[side] = None
                self._last_swivel[side] = (phi, self._swivel_target[side], err)
                # Per-iteration Δφ, matching the Δq space of this solve.
                dphi_desired = -self.config.elbow_swivel_gain * err
                w = float(np.sqrt(swivel_weight * fade))
                blocks_A.append(w * (jac_phi @ N).reshape(1, n))
                blocks_r.append(
                    np.array([w * (dphi_desired - jac_phi @ dq_primary)])
                )

        # ── Manipulability, gated (optional) ─────────────────────────────────
        if (self.config.enable_manipulability
                and self.config.manipulability_weight > 0.0):
            dq_manip = np.zeros(nv)
            gate_total = 0.0
            for side in ("left", "right"):
                grad, mu = self._manipulability_gradient(side)
                gate = self._manipulability_gate(mu)
                if gate <= 0.0:
                    continue
                norm = float(np.linalg.norm(grad))
                if norm < 1e-12:
                    continue
                dof_ids = (self._left_arm_dof_ids if side == "left"
                           else self._right_arm_dof_ids)
                # Direction only: |d log mu / dq| is ~0.3 at home against a
                # ~1e-3 tracking step, so using it raw would swamp everything
                # else. manipulability_gain is then a real step size in rad.
                dq_manip[dof_ids] = self.config.manipulability_gain * grad / norm
                gate_total = max(gate_total, gate)
            # Gate enters once, through the weight -- not also through the
            # desired step, which would square it.
            if gate_total > 0.0:
                add_velocity_objective(
                    self.config.manipulability_weight * gate_total,
                    dq_manip[free_ids],
                )

        # ── Solve the stacked secondary least-squares for z ──────────────────
        if blocks_A:
            A = np.vstack(blocks_A)
            r = np.concatenate(blocks_r)
            lam_z = self.config.nullspace_regularization
            z = np.linalg.solve(A.T @ A + lam_z * np.eye(n), A.T @ r)
            dq_free = dq_primary + N @ z
        else:
            dq_free = dq_primary

        dq = np.zeros(nv)
        dq[free_ids] = dq_free

        vel = dq / self.config.dt
        # Reuse the constraint rows the [S3] primary already built this
        # iteration (the configuration has not changed since), instead of
        # paying compute_qp_inequalities twice.
        return self._project_onto_limits(vel, ineq=ineq_cache)

    # ── dls_projector: null-space objective helpers ──────────────────────────

    def _reset_nullspace_state(self) -> None:
        """Drop continuity history and re-latch swivel targets.

        Called from both init paths: after a reset the previous velocity is
        meaningless, and any latched swivel angle belongs to the old pose.
        Sides named explicitly in `config.elbow_swivel_targets` keep their
        configured angle; the rest re-latch on the next solve.
        """
        self._prev_vel = None
        self._first_iteration = True
        self._base_weight_cached = None
        self._swivel_target = {
            side: self.config.elbow_swivel_targets.get(side)
            for side in ("left", "right")
        }
        self._swivel_ref = {"left": None, "right": None}
        self._swivel_u_prev = {"left": None, "right": None}
        self._swivel_err_since = {"left": None, "right": None}
        self._last_swivel = {}

    def set_elbow_swivel_target(
        self, side: str, angle: Optional[float] = None
    ) -> Optional[float]:
        """Set (or re-latch, with angle=None) one arm's target swivel angle.

        Safe to call while the control loop runs -- the next solve picks it
        up. Returns the stored value, or None if it will re-latch.
        """
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self._swivel_target[side] = None if angle is None else float(angle)
        return self._swivel_target[side]

    def elbow_swivel_angle(self, side: str) -> Optional[float]:
        """Current swivel angle (rad) of one arm, or None where undefined.

        None means the elbow is too close to the shoulder-wrist axis for the
        angle to be meaningful -- see `elbow_swivel_min_offset`.
        """
        S, E, W = self._swivel_points(side)
        # Read-only: use the transported reference if one is live, without
        # advancing the transport (that is _swivel_row's job).
        ref = None
        if self.config.swivel_parallel_ref and self._swivel_ref[side] is not None:
            r = self._swivel_ref[side]
            ref = (float(r[0]), float(r[1]), float(r[2]))
        phi, offset = self._swivel_from_points(S, E, W, ref=ref)
        return None if offset < 1e-9 else phi

    def _swivel_points(self, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.configuration.data
        return tuple(
            d.xpos[self.model.body(self._SWIVEL_BODIES[k].format(side=side)).id].copy()
            for k in ("shoulder", "elbow", "wrist")
        )

    @staticmethod
    def _swivel_scalar(
        sx: float, sy: float, sz: float,
        ex: float, ey: float, ez: float,
        wx: float, wy: float, wz: float,
        ref: Optional[tuple[float, float, float]] = None,
    ) -> tuple[float, float]:
        """Swivel angle and perpendicular offset, in plain floats.

        Same maths as `_swivel_from_points`, written without numpy on
        purpose: the Jacobian finite-differences this 18 times per arm per
        solver iteration, where numpy's per-call dispatch overhead on
        3-vectors dominated everything else (measured 74 us/call, i.e.
        1.3 ms of a 1.6 ms swivel row -- more than the whole rest of the
        solve). Scalar arithmetic makes the same work ~20x cheaper.
        """
        ax, ay, az = wx - sx, wy - sy, wz - sz
        a_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if a_norm < 1e-9:
            return 0.0, 0.0
        ux, uy, uz = ax / a_norm, ay / a_norm, az / a_norm

        rx, ry, rz = ex - sx, ey - sy, ez - sz
        proj = rx * ux + ry * uy + rz * uz
        vx, vy, vz = rx - proj * ux, ry - proj * uy, rz - proj * uz
        offset = math.sqrt(vx * vx + vy * vy + vz * vz)
        if offset < 1e-9:
            return 0.0, 0.0

        # In-plane reference: world +z, or +x when the arm axis is nearly
        # vertical -- NOTE this switch is a step, not continuous: crossing
        # |u_z| = 0.9 jumps the measured angle (see swivel_parallel_ref for
        # the gated fix). A caller may pass `ref` to supply its own
        # (continuous) reference vector instead.
        if ref is not None:
            rfx, rfy, rfz = ref
        elif abs(uz) > 0.9:
            rfx, rfy, rfz = 1.0, 0.0, 0.0
        else:
            rfx, rfy, rfz = 0.0, 0.0, 1.0
        dot = rfx * ux + rfy * uy + rfz * uz
        n1x, n1y, n1z = rfx - dot * ux, rfy - dot * uy, rfz - dot * uz
        n1_norm = math.sqrt(n1x * n1x + n1y * n1y + n1z * n1z)
        if n1_norm < 1e-9:
            return 0.0, 0.0  # reference parallel to the axis: undefined
        n1x, n1y, n1z = n1x / n1_norm, n1y / n1_norm, n1z / n1_norm
        n2x = uy * n1z - uz * n1y
        n2y = uz * n1x - ux * n1z
        n2z = ux * n1y - uy * n1x
        return (
            math.atan2(vx * n2x + vy * n2y + vz * n2z,
                       vx * n1x + vy * n1y + vz * n1z),
            offset,
        )

    @classmethod
    def _swivel_from_points(
        cls, S: np.ndarray, E: np.ndarray, W: np.ndarray,
        ref: Optional[tuple[float, float, float]] = None,
    ) -> tuple[float, float]:
        """Swivel angle of the elbow about the shoulder->wrist axis.

        Returns (angle_rad, perpendicular_offset_m). The offset is how far
        the elbow sits off that axis: it goes to zero as the arm straightens,
        at which point the angle is undefined and the caller should fade the
        objective out rather than chase it.
        """
        return cls._swivel_scalar(S[0], S[1], S[2], E[0], E[1], E[2],
                                  W[0], W[1], W[2], ref=ref)

    def _swivel_reference(self, side: str, u: np.ndarray) -> Optional[np.ndarray]:
        """[S5a] Continuous in-plane reference vector for one arm's swivel.

        Parallel-transports a stored per-side reference along changes of the
        shoulder->wrist axis `u`: rotate it by the minimal rotation taking
        the previous axis onto the current one, re-orthonormalise, store.
        Continuous everywhere, unlike the fixed z/x convention it replaces
        (which steps at |u_z| = 0.9). Path-dependent by nature (transport
        around a closed axis loop picks up the enclosed solid angle), which
        is slow drift the [S5b] re-latch policy bounds. Returns None when no
        usable reference exists this call.
        """
        ref = self._swivel_ref[side]
        u_prev = self._swivel_u_prev[side]
        if ref is None or u_prev is None:
            # Seed with the discrete convention, so the angle (and any
            # target latched from it) agrees with gate-off at the moment of
            # enabling.
            ref = (np.array([1.0, 0.0, 0.0]) if abs(u[2]) > 0.9
                   else np.array([0.0, 0.0, 1.0]))
        else:
            c = float(np.dot(u_prev, u))
            k = np.cross(u_prev, u)
            s2 = float(np.dot(k, k))
            if c < -0.999999:
                # Axis flipped ~180 deg in one step: no continuous
                # transport exists. Re-seed rather than guess.
                ref = (np.array([1.0, 0.0, 0.0]) if abs(u[2]) > 0.9
                       else np.array([0.0, 0.0, 1.0]))
            elif s2 > 1e-16:
                # Rodrigues rotation taking u_prev onto u, with
                # k = u_prev x u (|k| = sin theta, c = cos theta).
                ref = (ref * c + np.cross(k, ref)
                       + k * (float(np.dot(k, ref)) * (1.0 - c) / s2))
            # else: axes identical, transport is the identity.
        ref = ref - float(np.dot(ref, u)) * u
        nrm = float(np.linalg.norm(ref))
        if nrm < 1e-9:
            return None
        ref = ref / nrm
        self._swivel_ref[side] = ref
        self._swivel_u_prev[side] = np.asarray(u, dtype=float).copy()
        return ref

    def _swivel_row(self, side: str, free_ids: np.ndarray):
        """One arm's swivel Jacobian row and current angle.

        Returns (J_phi over free DOFs, angle, offset), or None when the angle
        is undefined. dphi/dq is assembled by the chain rule through the three
        point positions:  dphi/dq = sum_P (dphi/dP)^T J_P, with J_P the 3xnv
        world-frame translational Jacobians MuJoCo gives directly, and the
        3-vector partials finite-differenced on the cheap pure-numpy scalar
        above (no forward kinematics in the inner loop).

        With swivel_parallel_ref the transported reference is fetched once
        and held FROZEN through the finite differences: the reference's own
        dependence on q (through the axis) is second-order in the
        perturbation and not worth the transport bookkeeping inside the FD
        loop.
        """
        S, E, W = self._swivel_points(side)
        ref = None
        if self.config.swivel_parallel_ref:
            axis = W - S
            a_norm = float(np.linalg.norm(axis))
            if a_norm > 1e-9:
                r = self._swivel_reference(side, axis / a_norm)
                if r is not None:
                    ref = (float(r[0]), float(r[1]), float(r[2]))
        phi, offset = self._swivel_from_points(S, E, W, ref=ref)
        if offset < 1e-9:
            return None

        coords = [S[0], S[1], S[2], E[0], E[1], E[2], W[0], W[1], W[2]]
        grads = [np.zeros(3), np.zeros(3), np.zeros(3)]
        h = 1e-6
        two_h = 2.0 * h
        for j in range(9):
            base = coords[j]
            coords[j] = base + h
            a_hi, off_hi = self._swivel_scalar(*coords, ref=ref)
            coords[j] = base - h
            a_lo, off_lo = self._swivel_scalar(*coords, ref=ref)
            coords[j] = base
            if off_lo < 1e-9 or off_hi < 1e-9:
                return None
            # Wrap the difference: phi is an angle, so a perturbation
            # straddling +-pi must not read as a ~2pi gradient.
            d = a_hi - a_lo
            grads[j // 3][j % 3] = math.atan2(math.sin(d), math.cos(d)) / two_h

        jac_phi = np.zeros(self.model.nv)
        jacp, jacr = self._jac_buf_p, self._jac_buf_r
        for key, g in zip(("shoulder", "elbow", "wrist"), grads):
            body_id = self.model.body(self._SWIVEL_BODIES[key].format(side=side)).id
            jacp[:] = 0.0
            mujoco.mj_jacBody(
                self.model, self.configuration.data, jacp, jacr, body_id
            )
            jac_phi += g @ jacp
        return jac_phi[free_ids], phi, offset

    def _arm_jacobian(self, side: str, data) -> np.ndarray:
        """6x7 world-frame EE Jacobian w.r.t. one arm's own joints."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        site = self._LEFT_EE_SITE if side == "left" else self._RIGHT_EE_SITE
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.model.site(site).id)
        cols = (self._left_arm_dof_ids if side == "left"
                else self._right_arm_dof_ids)
        return np.vstack([jacp, jacr])[:, cols]

    @staticmethod
    def _log_manipulability(J_arm: np.ndarray) -> float:
        """sum(log sigma_i) of a 6x7 Jacobian == log sqrt(det(J J^T)).

        Computed from singular values rather than a determinant: det(J J^T)
        underflows hard near a singularity, where this objective is precisely
        the one that has to stay meaningful.
        """
        sv = np.linalg.svd(J_arm, compute_uv=False)
        sv = np.maximum(sv, 1e-12)
        return float(np.sum(np.log(sv)))

    def _manipulability_gradient(self, side: str) -> tuple[np.ndarray, float]:
        """(d log mu / dq over that arm's 7 joints, mu) by central differences.

        Costs 14 perturbed forward-kinematics + Jacobian evaluations per arm,
        which is why enable_manipulability defaults to False.
        """
        qpos_adrs = (self._left_arm_qpos_adrs if side == "left"
                     else self._right_arm_qpos_adrs)
        h = self.config.manipulability_fd_step
        d = self._fd_data
        d.qpos[:] = self.configuration.q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        mu = float(np.exp(self._log_manipulability(self._arm_jacobian(side, d))))

        grad = np.zeros(len(qpos_adrs))
        for i, adr in enumerate(qpos_adrs):
            original = float(d.qpos[adr])
            d.qpos[adr] = original + h
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            hi = self._log_manipulability(self._arm_jacobian(side, d))
            d.qpos[adr] = original - h
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            lo = self._log_manipulability(self._arm_jacobian(side, d))
            d.qpos[adr] = original
            grad[i] = (hi - lo) / (2.0 * h)
        return grad, mu

    def _gated_base_weight(self) -> float:
        """Base motion cost, made cheap when the arms are running out of posture.

        A flat cost is the wrong shape. High enough to stop the base chasing
        tracker noise, it also stops the base helping until the arms have
        already contorted -- measured on 2026-08-24 across three runs, base
        motion averaged 0.00001 m/tick while the worst arm's manipulability was
        above 0.050 and 0.0014-0.0020 below it. The chassis was a last resort
        after the posture had degraded, which is the wrong way round: it should
        move *so that* the posture does not degrade.

        Manipulability is the right discriminator because it is exactly the
        quantity that distinguishes "the arms are fine, this is noise" from
        "the arms are running out". mu at the home keyframe is 0.0506, and the
        gate ramps from base_motion_weight at `gate_on` down to
        `base_motion_weight_min` at `gate_full`.

        Only the *value* is needed, not the gradient, which is what makes this
        affordable: two Jacobians and two 6x7 SVDs per solve, against the 28
        perturbed kinematics evaluations `_manipulability_gradient` costs and
        that keep enable_manipulability off.

        The worst arm governs. One arm at a singularity is enough of a reason
        to move the chassis, and averaging would let a comfortable arm hide it.
        """
        full = float(self.config.base_motion_weight)
        floor = float(self.config.base_motion_weight_min)
        if full == floor or self.fix_base:
            return full
        if not self._first_iteration and self._base_weight_cached is not None:
            # Posture barely moves within one solve; compute it once per tick.
            return self._base_weight_cached

        data = self.configuration.data
        mu = min(
            float(np.exp(self._log_manipulability(self._arm_jacobian(side, data))))
            for side in ("left", "right")
        )
        on = float(self.config.base_weight_gate_on)
        gate_full = float(self.config.base_weight_gate_full)
        if on <= gate_full:
            gate = 1.0 if mu <= gate_full else 0.0
        else:
            x = float(np.clip((on - mu) / (on - gate_full), 0.0, 1.0))
            gate = x * x * (3.0 - 2.0 * x)          # smoothstep
        self._base_weight_cached = full + (floor - full) * gate
        return self._base_weight_cached

    def _manipulability_gate(self, mu: float) -> float:
        """Smoothstep: 0 at mu >= gate_on, 1 at mu <= gate_full."""
        on = self.config.manipulability_gate_on
        full = self.config.manipulability_gate_full
        if on <= full:
            return 1.0 if mu <= full else 0.0
        x = float(np.clip((on - mu) / (on - full), 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _limit_inequalities(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Stacked (G, h) of every active limit's Δq inequalities, or None.

        Shared by [S3]'s constrained primary and `_project_onto_limits`, so
        the two act on exactly the same constraint set. Also counts the rows
        contributed by the collision limit into
        `_last_ineq_collision_rows` for the [S7] diagnostics -- an active
        collision constraint is the difference between "the solver is
        stuck" and "the solver is being stopped", and nothing logged it.
        """
        limits = self._limits_with_collision if self.avoid_collisions else self.limits
        G_list, h_list = [], []
        collision_rows = 0
        for limit in limits:
            ineq = limit.compute_qp_inequalities(self.configuration, self.config.dt)
            if not ineq.inactive:
                G_list.append(ineq.G)
                h_list.append(ineq.h)
                if limit is self.collision_limit:
                    collision_rows = ineq.G.shape[0]
        self._last_ineq_collision_rows = collision_rows
        if not G_list:
            return None
        return np.vstack(G_list), np.hstack(h_list)

    def _project_onto_limits(
        self, vel: np.ndarray,
        ineq: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Clip a velocity onto the same hard joint/collision limits the
        other two modes get for free from mink's QP, by solving the
        closest-feasible-point QP min_Δq' ||Δq' - Δq||^2 s.t. GΔq' <= h.
        Only dls_projector needs this -- it bypasses mink.solve_ik (and
        therefore its limits=) entirely for the EE+posture resolution.

        Note the QP is posed in Δq (configuration displacement), not
        velocity: every mink Limit's compute_qp_inequalities bounds Δq
        (mink's own solve_ik variable is Δq, only divided by dt at the very
        end to return v) -- projecting `vel` directly against those (G, h)
        would be off by a factor of dt and over-clip almost everything.

        Tried special-casing ConfigurationLimit/VelocityLimit (both pure
        per-DOF box constraints) as a closed-form np.clip to skip the QP for
        the common case -- measured slower, not faster: daqp solves this
        small a box QP fast enough natively that the extra numpy bookkeeping
        (scanning G for nonzero structure, min/max reduction per DOF) costs
        more than it saves (~0.75ms/call vs ~0.56ms/call, benchmarked at
        home + a small offset target). Left as one QP; the real per-solve
        cost floor is compute_qp_inequalities itself (~0.4ms, mostly
        mj_differentiatePos), not the solve on top of it.

        `ineq` lets a caller that already built this configuration's (G, h)
        via _limit_inequalities pass it in rather than rebuilding it."""
        if ineq is None:
            ineq = self._limit_inequalities()
        if ineq is None:
            return vel
        G, h = ineq
        dq = vel * self.config.dt
        nv = vel.shape[0]
        problem = qpsolvers.Problem(np.eye(nv), -dq, G, h)
        for solver in [self.config.solver] + self.config.fallback_solvers:
            try:
                result = qpsolvers.solve_problem(problem, solver=solver)
                if result.found:
                    return result.x / self.config.dt
            except Exception:
                continue
        raise RuntimeError(
            "All solvers failed projecting dls_projector's velocity onto "
            f"joint/collision limits: {[self.config.solver] + self.config.fallback_solvers}"
        )

    def _build_posture_cost(self) -> np.ndarray:
        """Per-DOF posture cost vector (nv). Small epsilon avoids NaN in mink."""
        cost = np.full(self.model.nv, 1e-8)
        # Base DOFs (x, y, yaw)
        for dof_id in self.base_dof_ids:
            cost[dof_id] = self.config.base_posture_cost
        # Lift
        cost[self.model.joint(self._LIFT_JOINT).dofadr] = self.config.lift_posture_cost
        # Arms — uniform, then per-joint overrides on top.
        for jn in self._LEFT_JOINTS + self._RIGHT_JOINTS:
            cost[self.model.joint(jn).dofadr] = self.config.arm_posture_cost
        for jn, c in self.config.arm_posture_cost_overrides.items():
            cost[self._arm_joint_dofadr(jn)] = float(c)
        return cost

    def _arm_joint_dofadr(self, joint_name: str) -> int:
        """DOF address of an arm joint, with a useful error for a bad name.

        model.joint() raises a bare KeyError on a typo, which during tuning
        reads as "the override did nothing" — so name the valid set here.
        """
        if joint_name not in self.arm_joint_names:
            raise ValueError(
                f"Unknown arm joint {joint_name!r}. Valid names: "
                f"{', '.join(self.arm_joint_names)}"
            )
        return int(self.model.joint(joint_name).dofadr)

    @property
    def arm_joint_names(self) -> list[str]:
        """The 14 arm joint names, in solver order (left 1-7 then right 1-7)."""
        return list(self._LEFT_JOINTS) + list(self._RIGHT_JOINTS)

    def set_arm_posture_costs(
        self,
        overrides: dict[str, float],
        *,
        replace: bool = False,
    ) -> np.ndarray:
        """Retune per-joint arm posture costs on a live solver.

        Higher cost on a joint → the solver moves it less and reaches with the
        other DOFs instead. Lower → that joint absorbs more of the motion.

        Parameters
        ----------
        overrides : {joint_name: cost}
            Merged into any existing overrides, or used as the whole set when
            ``replace=True``. Pass ``{}`` with ``replace=True`` to go back to a
            uniform ``arm_posture_cost``.
        replace : bool
            Discard the current overrides instead of merging.

        Returns the new nv-length cost vector.

        Safe to call while the control loop is running: mink reads the cost
        vector at solve time, so the change takes effect on the next solve.
        Costs are a *soft* preference — to stop a joint moving outright, limit
        its velocity instead (see ``_build_velocity_limits``).
        """
        merged = {} if replace else dict(self.config.arm_posture_cost_overrides)
        merged.update(overrides)
        for jn, c in merged.items():
            self._arm_joint_dofadr(jn)          # validate before mutating
            if float(c) <= 0.0:
                raise ValueError(
                    f"Posture cost for {jn!r} must be > 0 (mink divides by it); "
                    f"use a small value like 1e-8 to mean 'effectively free'."
                )
        self.config.arm_posture_cost_overrides = merged
        cost = self._build_posture_cost()
        self.posture_task.set_cost(cost)
        return cost

    def _collision_geoms(self, body_names: list[str]) -> list[int]:
        """Return the group-3 collision-sphere geom id for each named body.

        Each robot link carries one simplified sphere proxy (group 3) used for
        cheap, robust distance queries — far better than mesh-mesh for IK.
        Bodies without such a sphere are skipped.
        """
        geoms: list[int] = []
        for bn in body_names:
            try:
                bid = self.model.body(bn).id
            except KeyError:
                continue
            for g in range(self.model.ngeom):
                gm = self.model.geom(g)
                if (gm.bodyid[0] == bid
                        and gm.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE
                        and gm.group[0] == 3):
                    geoms.append(g)
                    break
        return geoms

    def _mesh_geom(self, body_name: str) -> list[int]:
        """Return the convex-mesh geom id for a body (used for the chassis,
        which has no usable sphere proxy). Empty list if none."""
        try:
            bid = self.model.body(body_name).id
        except KeyError:
            return []
        for g in range(self.model.ngeom):
            gm = self.model.geom(g)
            if gm.bodyid[0] == bid and gm.type[0] == mujoco.mjtGeom.mjGEOM_MESH:
                return [g]
        return []

    def _hand_collision_geoms(self, side: str) -> list[int]:
        """Collision-mesh geoms of the hand (palm + fingers) for one side.

        These are the geoms with contype > 0 (the physics collision meshes,
        not the group-1 visuals). Finger link1 bodies carry visuals only and
        are skipped automatically.
        """
        geoms: list[int] = []
        for bn in self._HAND_BODIES[side]:
            try:
                bid = self.model.body(bn).id
            except KeyError:
                continue
            for g in range(self.model.ngeom):
                gm = self.model.geom(g)
                if gm.bodyid[0] == bid and gm.contype[0] > 0:
                    geoms.append(g)
        return geoms

    def _build_collision_limit(self) -> Optional["mink.CollisionAvoidanceLimit"]:
        """Build a CollisionAvoidanceLimit for arm↔body and arm↔arm pairs.

        mink's pair filter (`_is_pass_contype_conaffinity_check`) only keeps a
        pair when ``contype[a] & conaffinity[b]`` (or the reverse) is non-zero.
        The proxy spheres ship with ``contype=0``, so every pair would be
        silently dropped.  contype/conaffinity is read *only* at construction
        time (the runtime query uses ``mj_geomDistance``, which ignores them),
        so we flip the flags on just the involved geoms, build the limit, then
        restore — leaving ``mj_forward`` physics untouched.
        """
        body_geoms   = self._collision_geoms(self._BODY_COLLISION_LINKS)
        base_geoms   = self._mesh_geom(self._BASE_LINK)
        left_geoms   = self._collision_geoms(self._LEFT_ARM_LINKS)
        right_geoms  = self._collision_geoms(self._RIGHT_ARM_LINKS)
        left_distal  = self._collision_geoms(self._LEFT_DISTAL_LINKS)
        right_distal = self._collision_geoms(self._RIGHT_DISTAL_LINKS)

        if not (left_geoms or right_geoms):
            return None

        collision_pairs = []
        if body_geoms:
            collision_pairs.append((left_geoms,  body_geoms))   # left arm  ↔ lift column
            collision_pairs.append((right_geoms, body_geoms))   # right arm ↔ lift column
        if base_geoms:
            collision_pairs.append((left_distal,  base_geoms))  # left distal  ↔ chassis
            collision_pairs.append((right_distal, base_geoms))  # right distal ↔ chassis
        collision_pairs.append((left_geoms, right_geoms))        # left arm ↔ right arm

        # Ground avoidance: floor plane ↔ arm spheres + hand collision meshes.
        # (Wheels/chassis are excluded — they rest on the floor by construction.)
        floor_geoms: list[int] = []
        left_hand = right_hand = []
        if self.config.enable_ground_avoidance:
            try:
                floor_geoms = [self.model.geom(self._FLOOR_GEOM).id]
            except KeyError:
                pass  # scene without a floor plane (e.g. arm-only test scenes)
            if floor_geoms:
                left_hand  = self._hand_collision_geoms("left")
                right_hand = self._hand_collision_geoms("right")
                collision_pairs.append((left_geoms  + left_hand,  floor_geoms))
                collision_pairs.append((right_geoms + right_hand, floor_geoms))

        involved = (set(body_geoms) | set(base_geoms)
                    | set(left_geoms) | set(right_geoms)
                    | set(left_hand) | set(right_hand) | set(floor_geoms))
        saved = {g: (int(self.model.geom_contype[g]),
                     int(self.model.geom_conaffinity[g])) for g in involved}
        try:
            for g in involved:
                self.model.geom_contype[g]     = 1
                self.model.geom_conaffinity[g] = 1
            limit = mink.CollisionAvoidanceLimit(
                model=self.model,
                geom_pairs=collision_pairs,
                gain=self.config.collision_gain,
                minimum_distance_from_collisions=self.config.collision_min_distance,
                collision_detection_distance=self.config.collision_detect_distance,
            )
        finally:
            for g, (ct, ca) in saved.items():
                self.model.geom_contype[g]     = ct
                self.model.geom_conaffinity[g] = ca
        return limit

    def _build_velocity_limits(self) -> dict:
        """Per-joint velocity limits for mink.VelocityLimit.

        Note: freejoint is intentionally excluded — mink.VelocityLimit does not
        support free joints. Base velocity is controlled via DampingTask instead.
        """
        lims: dict = {}
        lims[self._LIFT_JOINT] = self.config.lift_vel_limit
        for jn in self._LEFT_JOINTS + self._RIGHT_JOINTS:
            lims[jn] = self.config.arm_vel_limit
        return lims
