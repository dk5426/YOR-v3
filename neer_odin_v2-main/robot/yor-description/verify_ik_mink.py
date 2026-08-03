import sys
from pathlib import Path
import numpy as np

# Append repo root to path to import robot.arm.ik_solver
_HERE = Path(__file__).parent
sys.path.append(str(_HERE.parent.parent))

from robot.arm.ik_solver import SingleArmIK
import mujoco
import mink

def verify_ik_and_fk():
    old_mjcf = (_HERE / "nero-welded-base-and-lift.mjcf").as_posix()
    new_mjcf = (_HERE / "robot_mujoco.xml").as_posix()
    
    solver_old = SingleArmIK(old_mjcf)
    solver_new = SingleArmIK(new_mjcf)
    
    # --- 1. FK Check ---
    test_q = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.2])
    
    solver_old.init(test_q)
    solver_new.init(test_q)
    
    fk_old_ee = solver_old.forward_kinematics() # World to EE
    fk_new_ee = solver_new.forward_kinematics()
    
    def get_base_xform(solver):
        m = solver.model
        d = solver.configuration.data
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_arm_base_link")
        pos = d.xpos[body_id]
        mat = d.xmat[body_id].reshape(3, 3)
        return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(mat), pos)
        
    old_base_se3 = get_base_xform(solver_old)
    new_base_se3 = get_base_xform(solver_new)
    
    old_base_T_ee = old_base_se3.inverse() @ fk_old_ee
    new_base_T_ee = new_base_se3.inverse() @ fk_new_ee
    
    print("=== Forward Kinematics (Base to End-Effector) ===")
    print("Old Local Trans: ", np.round(old_base_T_ee.translation(), 6))
    print("New Local Trans: ", np.round(new_base_T_ee.translation(), 6))
    diff_trans = np.linalg.norm(old_base_T_ee.translation() - new_base_T_ee.translation())
    print(f"Trans Diff: {diff_trans:.8f}")
    
    print("\nOld Local Rot:   \n", np.round(old_base_T_ee.rotation().as_matrix(), 5))
    print("New Local Rot:   \n", np.round(new_base_T_ee.rotation().as_matrix(), 5))
    
    
    # --- 2. IK Check ---
    # Take the exact local target we computed from FK
    target_local = old_base_T_ee
    
    target_old_world = old_base_se3 @ target_local
    target_new_world = new_base_se3 @ target_local
    
    # Reset both configurations to slightly different from target to force IK to solve
    zero_q = np.zeros(7)
    solver_old.init(zero_q)
    solver_new.init(zero_q)
    
    q_old, success_old = solver_old.solve_ik(target_old_world, max_iter=500, pos_eps=1e-5, rot_eps=1e-4)    
    q_new, success_new = solver_new.solve_ik(target_new_world, max_iter=500, pos_eps=1e-5, rot_eps=1e-4)
    
    print("\n=== Inverse Kinematics ===")
    print(f"IK Old Success: {success_old}, q: {np.round(q_old, 5)}")
    print(f"IK New Success: {success_new}, q: {np.round(q_new, 5)}")
    
    delta_q = np.linalg.norm(q_old - q_new)
    print(f"IK Divergence (L2 Norm of difference in q): {delta_q:.8f}")

if __name__ == "__main__":
    verify_ik_and_fk()
