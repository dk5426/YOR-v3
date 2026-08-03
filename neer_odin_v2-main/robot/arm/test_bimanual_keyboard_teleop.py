import time
import sys
import tty
import termios
from pathlib import Path
import numpy as np

# Add the parent directory to sys.path if needed
sys.path.append(str(Path(__file__).parent.parent))

from robot.yor import YOR
import mink

def get_key():
    """Reads a single keypress from stdin without echoing or requiring enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # handle special character sequences like arrows, which start with \x1b
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                ch = ch + ch2 + ch3
            else:
                ch = ch + ch2
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def print_help():
    print("""
\r========= Keyboard Teleop =========
\rActive Arm controls:
\r  W / S : Move Forward (+X) / Backward (-X)
\r  A / D : Move Left (+Y) / Right (-Y)
\r  R / F : Move Up (+Z) / Down (-Z)
\r  
\r  [ : Switch to LEFT arm
\r  ] : Switch to RIGHT arm
\r  
\r  C : Print current status
\r  H : Print this help menu
\r  Q / ESC : Quit
\r===================================
""")

def main():
    yor = YOR(no_arms=False)
    
    print("Initializing YOR...")
    yor.init()
    
    print("\n--- YOR Initialized ---")
    print("Enabling Compliant Mode on BOTH arms...")
    # By default, set_compliant_mode uses Kp=0.0 meaning it won't track geometric targets.
    # We supply a low Kp so the arms feel soft but actually track our keyboard targets!
    soft_kp = [5.0] * 7
    soft_kd = [0.8] * 7
    yor.left_arm.set_compliant_mode(kp=soft_kp, kd=soft_kd) 
    yor.right_arm.set_compliant_mode(kp=soft_kp, kd=soft_kd)
    
    yor.left_arm.set_gravity_comp(True)
    yor.right_arm.set_gravity_comp(True)
    
    GC_SCALE = 1.0
    yor.left_arm.set_gravity_comp_scale(GC_SCALE)
    yor.right_arm.set_gravity_comp_scale(GC_SCALE)
    
    # Optional: sync target so the loop doesn't jump initially
    yor.left_arm.sync_target()
    yor.right_arm.sync_target()
    time.sleep(0.1)
    
    print("\nArms are compliant WITH gravity compensation.")
    
    active_arm = "left"
    delta = 0.015 # Move by 1.5 cm per key press
    
    # Store persistent targets to prevent backwards-jitter on rapid keystrokes!
    lee_init = yor.get_left_ee_pose()
    ree_init = yor.get_right_ee_pose()
    target_pose = {
        "left": lee_init.wxyz_xyz.copy() if lee_init else None,
        "right": ree_init.wxyz_xyz.copy() if ree_init else None
    }
    
    print_help()
    print(f"\r--> Currently controlling: {active_arm.upper()} arm")
    
    try:
        while True:
            key = get_key()
            
            if key in ['q', 'Q', '\x1b']:
                print("\rQuit command received.")
                break
                
            if key in ['h', 'H']:
                print_help()
                continue
                
            if key in ['c', 'C']:
                lee = yor.get_left_ee_pose()
                ree = yor.get_right_ee_pose()
                if lee: print(f"\rLeft EE (wxyz_xyz):  {[round(x, 3) for x in lee.wxyz_xyz]}")
                if ree: print(f"\rRight EE (wxyz_xyz): {[round(x, 3) for x in ree.wxyz_xyz]}")
                continue
                
            if key == '[':
                active_arm = "left"
                print(f"\r--> Switched to {active_arm.upper()} arm")
                lee = yor.get_left_ee_pose()
                if lee: target_pose["left"] = lee.wxyz_xyz.copy()
                yor.left_arm.sync_target()
                continue
            elif key == ']':
                active_arm = "right"
                print(f"\r--> Switched to {active_arm.upper()} arm")
                ree = yor.get_right_ee_pose()
                if ree: target_pose["right"] = ree.wxyz_xyz.copy()
                yor.right_arm.sync_target()
                continue
                
            # XYZ Deltas
            dx, dy, dz = 0.0, 0.0, 0.0
            if key in ['w', 'W']: dx = delta
            elif key in ['s', 'S']: dx = -delta
            elif key in ['a', 'A']: dy = delta
            elif key in ['d', 'D']: dy = -delta
            elif key in ['r', 'R']: dz = delta
            elif key in ['f', 'F']: dz = -delta
            
            if dx != 0.0 or dy != 0.0 or dz != 0.0:
                if target_pose[active_arm] is not None:
                    pose_arr = target_pose[active_arm]
                    pose_arr[4] += dx
                    pose_arr[5] += dy
                    pose_arr[6] += dz
                    
                    new_target = mink.SE3(pose_arr)
                    
                    if active_arm == "left":
                        yor.set_left_ee_target(new_target)
                    else:
                        yor.set_right_ee_target(new_target)
                        
                    print(f"\rJogged {active_arm} -> dx={dx}, dy={dy}, dz={dz}")
                    
    except KeyboardInterrupt:
        pass
    finally:
        # Re-enable standard terminal echoing
        print("\n\rStopping arms...")
        if hasattr(yor, 'left_arm') and yor.left_arm is not None:
             yor.left_arm.stop()
        if hasattr(yor, 'right_arm') and yor.right_arm is not None:
             yor.right_arm.stop()

if __name__ == "__main__":
    main()
