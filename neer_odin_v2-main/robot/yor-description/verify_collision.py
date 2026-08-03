import sys
from pathlib import Path
import numpy as np

# Append repo root
_HERE = Path(__file__).parent
sys.path.append(str(_HERE.parent.parent))

from robot.arm.ik_solver import SingleArmIK
import mink
import mujoco

def verify_collision():
    mjcf = (_HERE / "robot_mujoco.xml").as_posix()
    print("[TEST] Initializing solver...")
    solver = SingleArmIK(mjcf)
    
    # 1. Initialize to typical retracted position (h = 0)
    print("\n[TEST] Retracting Lift to 0.0m configuration natively...")
    solver.init(np.zeros(7))
    solver.update_lift_height(0.0)
    
    # 2. Let's aim the arm exactly down into the base
    # First get the absolute world pose of the arms mount so we can shoot below it
    arms_mount_id = mujoco.mj_name2id(solver.model, mujoco.mjtObj.mjOBJ_BODY, "arms_mount")
    curr_z_height = solver.configuration.data.xpos[arms_mount_id][2]
    print(f"[TEST] Physical Native Z-height of arms mount after lower: {curr_z_height:.4f}m")
    
    # Generate a target deep inside the physical boundary of the base chassis (Z < 0.22)
    danger_target_z = 0.15 # 15cm above world origin, physically inside the 22cm base collision box!
    danger_target = mink.SE3.from_rotation_and_translation(mink.SO3.identity(), np.array([0.5, -0.125, danger_target_z]))
    
    print("\n[TEST] Commanding target coordinate mathematically INSIDE the base limit (base = Z 0.22m)")
    q_danger, success = solver.solve_ik(danger_target, max_iter=200)
    
    print(f"\n[TEST] Solve Result:")
    print(f"Success (Fully reached mathematical origin inside rock): {success}")
    
    # Let's inspect where it actually Halted
    actual_ee = solver.forward_kinematics()
    
    # Calculate physical lowest point of the arm
    min_z = float('inf')
    arm_geoms = solver.limits[1].geom_pairs[0][1] # Get the arm geoms from collision limit
    for g in arm_geoms:
        min_z = min(min_z, solver.configuration.data.geom_xpos[g][2])
        
    print(f"Actual Halted EE Elevation: {actual_ee.translation()[2]:.4f}m")
    print(f"Physical Lowest Arm Geometry Z-Center: {min_z:.4f}m")
    
    if min_z >= 0.22:
        print("PASS: The collision repulsion boundary successfully locked the IK solver from breaching the chassis!")
    else:
        print("FAIL: The arm penetrated the collision mesh bounds.")

if __name__ == "__main__":
    verify_collision()
