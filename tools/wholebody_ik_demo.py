"""
wholebody_ik_demo.py — Interactive whole-body IK demo for YORv3.

Mode: KINEMATIC (write qpos directly + mj_forward — no physics/oscillation).

Controls
--------
  SPACE       Pause / resume
  ENTER       Toggle fix-base  (lock base x/y/theta, arms + lift only)
  C           Toggle self-collision avoidance (arm↔lift / arm↔base / arm↔arm)
  M           Toggle auto-animation (Figure-8)
  W A S D Q E Nudge the LEFT target ±5 cm (X / Y / Z)
  R           Reset to home keyframe
  Drag the green/magenta target spheres to move the EE targets:
    left_ik_target   → left end-effector target  (green)
    right_ik_target  → right end-effector target (magenta)
  (Double-click a sphere to select, then Ctrl + right-drag to move it.)

Solver
------
  Change SOLVER below to benchmark different QP back-ends.
  Benchmark on YORv3 (18 DOF, M-series Mac):
    pyqpmad  0.12ms  ← fastest, used by default
    daqp     0.13ms
    quadprog 0.14ms
    proxqp   0.25ms
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
from robot.arm.wholebody_ik import WholeBodyIK, WholeBodyIKConfig, DEFAULT_SCENE

# ── Solver ────────────────────────────────────────────────────────────────────
SOLVER  = "pyqpmad"   # change to benchmark others
IK_FREQ = 200         # Hz

# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Build IK solver ───────────────────────────────────────────────────────
    cfg = WholeBodyIKConfig(
        solver=SOLVER,
        dt=1.0 / IK_FREQ,
        max_iters=10,
        pos_threshold=2e-3,
        ori_threshold=5e-3,
        ee_position_cost=1.0,
        ee_orientation_cost=0.5,
        arm_posture_cost=1e-3,
        lift_posture_cost=1e-4,   # <== Make lift eager to move
        base_posture_cost=5e-2,   # <== Make base reluctant to move
        ground_lock_cost=200.0,
        arm_vel_limit=2.5,
        base_lin_vel_limit=0.4,
    )
    ik = WholeBodyIK(scene_xml=str(DEFAULT_SCENE), config=cfg)
    ik.init_from_keyframe("home")

    model = ik.model
    data  = ik.data

    # ── Viewer state ──────────────────────────────────────────────────────────
    paused = False
    reset  = False
    auto_move = False  # Start OFF so WASD works

    # Hoisted mocap ids (avoid name lookups inside callbacks / the 200 Hz loop)
    l_mid = model.body("left_ik_target").mocapid
    r_mid = model.body("right_ik_target").mocapid

    def key_callback(key: int) -> None:
        nonlocal paused, reset, auto_move
        if key == ord(" "):
            paused = not paused
            print(f"[IK] {'Paused' if paused else 'Resumed'}")
        elif key == 13:          # ENTER
            fix = ik.toggle_fix_base()
            print(f"[IK] fix_base = {fix}")
        elif key in (ord("C"), ord("c")):
            avoid = ik.toggle_collision_avoidance()
            print(f"[IK] collision_avoidance = {avoid}")
        elif key in (ord("R"), ord("r")):
            reset = True
        elif key in (ord("M"), ord("m")):
            auto_move = not auto_move
            print(f"[IK] Auto-animation {'ON' if auto_move else 'OFF'}")
        
        # ── WASD Keyboard controls for Left Arm ──
        step = 0.05  # 5cm per key press
        mid = l_mid
        if key in (ord("W"), ord("w")): data.mocap_pos[mid][0] += step
        if key in (ord("S"), ord("s")): data.mocap_pos[mid][0] -= step
        if key in (ord("A"), ord("a")): data.mocap_pos[mid][1] += step
        if key in (ord("D"), ord("d")): data.mocap_pos[mid][1] -= step
        if key in (ord("Q"), ord("q")): data.mocap_pos[mid][2] += step
        if key in (ord("E"), ord("e")): data.mocap_pos[mid][2] -= step

    # ── Launch viewer ─────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=True,     # Enabled right UI for Mocap sliders
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        # Turn OFF site-frame axes (removes the 20+ finger axis clutter)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        # Sync simulation data to home keyframe and move mocap to current EE poses
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        # Snap the big target spheres to the actual EE positions
        mink.move_mocap_to_frame(model, data, "left_ik_target",  "left_arm_ee",  "site")
        mink.move_mocap_to_frame(model, data, "right_ik_target", "right_arm_ee", "site")

        rate  = RateLimiter(frequency=IK_FREQ, warn=False)
        step  = 0

        print("=" * 60)
        print(" YORv3 Whole-Body IK — Kinematic Demo")
        print("=" * 60)
        print(f"  Solver : {SOLVER}")
        print(f"  Mode   : kinematic (qpos write + mj_forward)")
        print(f"  DOFs   : 3 base (planar x/y/θ) + 1 lift + 7L + 7R")
        print(f"  Rate   : {IK_FREQ} Hz")
        print(f"  Collision avoidance : "
              f"{'ON' if ik.avoid_collisions else 'OFF'} "
              f"({ik.n_collision_pairs} pairs: arm↔lift/base/arm + hand↔floor)")
        print()
        print("  HOW TO MOVE THE ARMS:")
        print("  1. Keyboard (Left Arm):")
        print("     - W / S : Forward / Back")
        print("     - A / D : Left / Right")
        print("     - Q / E : Up / Down")
        print()
        print("  2. Double-Click (Trackpad):")
        print("     - DOUBLE-CLICK the green/magenta sphere to select it")
        print("     - Once selected, sliders will appear on the right panel!")
        print()
        print("  KEYBOARD CONTROLS:")
        print("  SPACE  — pause/resume")
        print("  M      — toggle AUTO-ANIMATION (Figure-8)")
        print("  ENTER  — toggle fix_base (lock base, only arms move)")
        print("  C      — toggle self-collision avoidance (arm↔lift/base/arm)")
        print("  R      — reset to home")
        print("=" * 60)

        t_offset = time.time()
        
        while viewer.is_running():

            # ── Reset ─────────────────────────────────────────────────────────
            if reset:
                ik.init_from_keyframe("home")
                mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
                mujoco.mj_forward(model, data)
                mink.move_mocap_to_frame(model, data, "left_ik_target",  "left_arm_ee",  "site")
                mink.move_mocap_to_frame(model, data, "right_ik_target", "right_arm_ee", "site")
                reset = False
                t_offset = time.time()
                print("[IK] Reset to home.")

            if paused:
                viewer.sync()
                rate.sleep()
                continue
                
            # ── Auto-Animation (Figure-8 pattern) ─────────────────────────────
            if auto_move:
                t = time.time() - t_offset

                # Base poses from home keyframe
                base_l = np.array([0.37, -0.25, 0.36])
                base_r = np.array([-0.37, -0.25, 0.36])
                
                # Figure-8
                data.mocap_pos[l_mid] = base_l + np.array([np.sin(t)*0.2, 0.0, np.sin(t*2)*0.1])
                data.mocap_pos[r_mid] = base_r + np.array([-np.sin(t)*0.2, 0.0, np.sin(t*2)*0.1])

            # ── Read mocap targets (dragged by user in viewer) ────────────────
            T_left  = mink.SE3.from_mocap_name(model, data, "left_ik_target")
            T_right = mink.SE3.from_mocap_name(model, data, "right_ik_target")

            # ── Sync IK with current data.qpos ────────────────────────────────
            ik.update_configuration(data.qpos)

            # ── Solve IK ──────────────────────────────────────────────────────
            result = ik.solve(T_left, T_right)

            # ── Kinematic mode: write qpos directly, call mj_forward ──────────
            ik.apply_to_sim_kinematic(data, result)
            mujoco.mj_forward(model, data)

            # ── Update FK visualisation (thin axes on top of actual EE pos) ──
            T_l_fk, T_r_fk = ik.forward_kinematics()
            _set_mocap(model, data, "left_arm_ee_fk_viz",  T_l_fk)
            _set_mocap(model, data, "right_arm_ee_fk_viz", T_r_fk)

            viewer.sync()

            # ── Telemetry every 2 s ───────────────────────────────────────────
            step += 1
            if step % (IK_FREQ * 2) == 0:
                bp = result.base_position
                print(
                    f"[IK] L={result.left_pos_err*100:.1f}cm  "
                    f"R={result.right_pos_err*100:.1f}cm  "
                    f"solved={result.solved}  iters={result.iters}  "
                    f"base(x={bp[0]:.2f} y={bp[1]:.2f} θ={np.degrees(bp[2]):.1f}°)  "
                    f"lift={result.lift_q:.3f}m  fix_base={ik.fix_base}"
                )

            rate.sleep()


# ─────────────────────────────────────────────────────────────────────────────

def _set_mocap(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    name:  str,
    T:     mink.SE3,
) -> None:
    """Write SE3 pose into a named mocap body (MuJoCo wxyz quat convention)."""
    mid = model.body(name).mocapid
    data.mocap_pos[mid]  = T.translation()
    data.mocap_quat[mid] = T.rotation().wxyz


if __name__ == "__main__":
    main()
