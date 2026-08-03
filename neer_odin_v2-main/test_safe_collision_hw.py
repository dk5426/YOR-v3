import time
import numpy as np
import mink
from robot.yor import YOR

def main():
    print("=== DANGEROUS HARDWARE TEST: Collision Avoidance ===")
    print("WARNING: KEEP YOUR HAND ON THE E-STOP / KILL SWITCH!")
    
    # Initialize the robot
    # Note: no_arms=False so we actually connect to hardware
    yor = YOR(no_arms=False)
    
    print("\n[STEP 1] Initializing YOR. The lift will home to 0.0m.")
    yor.init()
    
    # It's currently at 0.0m natively if the init worked correctly.
    
    print("\n[STEP 2] Setting Arm into 'Spring Mode' (Ultra Low Stiffness).")
    print("This ensures that IF it touches the base, it won't have enough force to break anything.")
    # Set KP extremely low so it moves softly
    yor.set_left_gain(kp=3.0, kd=0.3)
    time.sleep(1.0)
    
    print("\n[STEP 3] Moving arm to a SAFE floating position in front of the robot...")
    # Safe floating target: x=0.4m forward, aligned with arm Y axis, z=0.6m in the air
    safe_target = mink.SE3.from_rotation_and_translation(
        mink.SO3.identity(),
        np.array([0.4, -0.125, 0.6])
    )
    # Slow preview time prevents jerky movements
    yor.set_left_ee_target(safe_target, preview_time=3.0)
    time.sleep(3.5)
    
    input("\n[STEP 4] Press ENTER to command the arm straight down INTO the base (Z = 0.15m)...")
    
    print("Commanding dive...")
    danger_target = mink.SE3.from_rotation_and_translation(
        mink.SO3.identity(),
        np.array([0.4, -0.125, 0.15])
    )
    
    # We command it to go into the base. If collision limits work, the internal IK solver will halt at ~Z=0.22.
    yor.set_left_ee_target(danger_target, preview_time=3.0)
    
    # Wait and track where it halts
    for i in range(40):
        time.sleep(0.1)
        actual_ee = yor.get_left_ee_pose()
        print(f"Tracking... Current Z: {actual_ee.translation()[2]:.3f}m")
        
    print("\n=== TEST CONCLUDED ===")
    print(f"Final Physical Target commanded: Z = 0.15m")
    print(f"Final IK Solver Halted at: Z = {yor.get_left_ee_pose().translation()[2]:.3f}m")
    if yor.get_left_ee_pose().translation()[2] > 0.22:
        print("SUCCESS! The arm safely hovered above the chassis limit.")
    else:
        print("WARNING: ARM PENETRATED BOUNDARIES.")

if __name__ == "__main__":
    main()
