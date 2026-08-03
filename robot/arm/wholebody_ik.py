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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

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

    # DampingTask costs for ground constraint (lock z, roll, pitch)
    ground_lock_cost: float    = 200.0  # high → robot stays upright and on ground

    # DampingTask cost for fix_base mode (lock vx, vy, wz too)
    base_damping_cost: float   = 100.0

    # ── Velocity limits ──────────────────────────────────────────────────────
    # Freejoint linear velocity (m/s) and angular velocity (rad/s)
    base_lin_vel_limit: float  = 0.5    # vx, vy
    base_ang_vel_limit: float  = 1.0    # wz
    lift_vel_limit: float      = 0.15   # m/s
    arm_vel_limit: float       = 2.5    # rad/s

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
    Arms and lift are controlled as usual. Optional hard-constraint
    self-collision avoidance keeps the arms clear of the lift column, the
    chassis, and each other.

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

        # qpos addresses
        self.base_qpos_adrs = np.array([int(self.model.joint(j).qposadr) for j in self._BASE_JOINTS])
        self._lift_qpos_adr = int(self.model.joint(self._LIFT_JOINT).qposadr)
        self._left_arm_qpos_adrs  = np.array([int(self.model.joint(n).qposadr) for n in self._LEFT_JOINTS])
        self._right_arm_qpos_adrs = np.array([int(self.model.joint(n).qposadr) for n in self._RIGHT_JOINTS])

        # Lift travel, taken from the model so the description stays the single
        # source of truth for it (currently 0 → 0.9176 m).
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
        # Precomputed limit lists so _solve_qp doesn't allocate per call
        # (it runs max_iters × 200 Hz).
        self._limits_with_collision = (
            self.limits + [self.collision_limit] if self.collision_limit else self.limits
        )

        # ── State ────────────────────────────────────────────────────────────
        self.initialized  = False
        self.fix_base     = False

    # ── Public API ───────────────────────────────────────────────────────────

    def init_from_keyframe(self, key_name: str = "home") -> None:
        """Reset to named keyframe and sync IK configuration."""
        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.model.key(key_name).id
        )
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.initialized = True

    def init_from_qpos(self, qpos: np.ndarray) -> None:
        """Initialise from an arbitrary full qpos vector."""
        self.configuration.update(qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.initialized = True

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

        if lift_target is not None:
            q_ref = self.configuration.q.copy()
            q_ref[self._lift_qpos_adr] = self.clamp_lift(lift_target)
            self.posture_task.set_target(q_ref)

        tasks = [
            self.left_ee_task,
            self.right_ee_task,
            self.posture_task,
        ]
        if self.fix_base:
            tasks.append(self.base_fix_task)

        l_pos = l_ori = r_pos = r_ori = np.inf
        iters = 0
        prev_base_q = self.configuration.q[self.base_qpos_adrs]

        for iters in range(1, self.config.max_iters + 1):
            vel = self._solve_qp(tasks)
            self.configuration.integrate_inplace(vel, self.config.dt)

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

    def _solve_qp(self, tasks: list) -> np.ndarray:
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

