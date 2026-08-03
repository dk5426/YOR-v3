import time
import sys
from pathlib import Path
import numpy as np

# Add the parent directory to sys.path if needed
sys.path.append(str(Path(__file__).parent.parent))

from robot.yor import YOR
import mink

def print_help():
    print("""
Available commands:
  j left <q1> <q2> <q3> <q4> <q5> <q6> <q7>  : Set left joint angles (radians)
  j right <q1> <q2> <q3> <q4> <q5> <q6> <q7> : Set right joint angles (radians)
  p left <qw> <qx> <qy> <qz> <x> <y> <z>     : Set left EE pose (qw qx qy qz x y z)
  p right <qw> <qx> <qy> <qz> <x> <y> <z>    : Set right EE pose (qw qx qy qz x y z)
  status                                     : Print current joint angles and EE poses
  help                                       : Show this message
  quit / exit                                : Stop the robot and exit
""")

def main():
    # Initialize YOR with both arms enabled
    yor = YOR(no_arms=False)
    
    print("Initializing YOR...")
    yor.init()
    
    print("\n--- YOR Initialized ---")
    
    print("Enabling Compliant Mode on BOTH arms...")
    # By default, set_compliant_mode uses Kp=0.0 meaning it won't track geometric targets.
    # We supply a low Kp so the arms feel soft but actually track our keyboard targets!
    soft_kp = [5.0] * 7
    soft_kd = [0.5] * 7
    yor.left_arm.set_compliant_mode(kp=soft_kp, kd=soft_kd) 
    yor.right_arm.set_compliant_mode(kp=soft_kp, kd=soft_kd)
    
    print("Enabling Gravity Compensation on BOTH arms...")
    yor.left_arm.set_gravity_comp(True)
    yor.right_arm.set_gravity_comp(True)
    
    # Scale factor: nerolib defaults to 0.0 internally, so this MUST be set.
    GC_SCALE = 1.0
    yor.left_arm.set_gravity_comp_scale(GC_SCALE)
    yor.right_arm.set_gravity_comp_scale(GC_SCALE)
    
    print(f"Gravity comp scale set to {GC_SCALE} for both arms.")
    print("\nArms are now in compliant mode WITH gravity compensation.")
    print("You can physically move the arms or use commands below.")
    
    print_help()

    try:
        while True:
            try:
                cmd_input = input("Command > ").strip()
            except EOFError:
                break
                
            cmd = cmd_input.split()
            if not cmd:
                continue
            
            action = cmd[0].lower()
            
            if action in ['quit', 'exit']:
                break
            elif action == 'help':
                print_help()
            elif action == 'status':
                lq = yor.get_left_joint_positions()
                rq = yor.get_right_joint_positions()
                lee = yor.get_left_ee_pose()
                ree = yor.get_right_ee_pose()
                
                print(f"Left joints:  {[round(x, 3) for x in lq]}")
                print(f"Right joints: {[round(x, 3) for x in rq]}")
                if lee: print(f"Left EE (wxyz_xyz):  {[round(x, 3) for x in lee.wxyz_xyz]}")
                if ree: print(f"Right EE (wxyz_xyz): {[round(x, 3) for x in ree.wxyz_xyz]}")
            
            elif action == 'j':
                if len(cmd) != 9:
                    print("Usage: j <left|right> <q1> <q2> <q3> <q4> <q5> <q6> <q7>")
                    continue
                side = cmd[1].lower()
                try:
                    q = np.array([float(x) for x in cmd[2:9]])
                    if side == 'left':
                        yor.set_left_joint_target(q)
                        print(f"Left joint target set to {q}")
                    elif side == 'right':
                        yor.set_right_joint_target(q)
                        print(f"Right joint target set to {q}")
                    else:
                        print("Side must be 'left' or 'right'")
                except ValueError:
                    print("Error parsing joint angles. Must be floats.")
                    
            elif action == 'p':
                if len(cmd) != 9:
                    print("Usage: p <left|right> <qw> <qx> <qy> <qz> <x> <y> <z>")
                    continue
                side = cmd[1].lower()
                try:
                    vals = np.array([float(x) for x in cmd[2:9]])
                    ee_target = mink.SE3(vals)
                    if side == 'left':
                        yor.set_left_ee_target(ee_target)
                        print(f"Left EE target set to {vals}")
                    elif side == 'right':
                        yor.set_right_ee_target(ee_target)
                        print(f"Right EE target set to {vals}")
                    else:
                        print("Side must be 'left' or 'right'")
                except Exception as e:
                    print(f"Error parsing pose: {e}")
            else:
                print(f"Unknown command: {action}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
        
    finally:
        print("Stopping arms...")
        if hasattr(yor, 'left_arm') and yor.left_arm is not None:
             yor.left_arm.stop()
        if hasattr(yor, 'right_arm') and yor.right_arm is not None:
             yor.right_arm.stop()

if __name__ == "__main__":
    main()
