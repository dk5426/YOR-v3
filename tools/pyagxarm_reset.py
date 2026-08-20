#!/usr/bin/env python3
"""
Reset and cycle enable/disable for joints 1-7 on can_left and can_right using pyAgxArm.
Also inspects motor error status and driver feedback.
"""
import time
from pyAgxArm import AgxArmFactory, create_agx_arm_config, NeroFW

channels = ["can_left", "can_right"]

print("================ pyAgxArm Arm Reset & Status Check (Joints 1-7) ================")

for ch in channels:
    print(f"\n[*] Connecting to arm on {ch}...")
    try:
        cfg = create_agx_arm_config(
            robot="nero",
            firmeware_version=NeroFW.V111,
            interface="socketcan",
            channel=ch,
        )
        r = AgxArmFactory.create_arm(cfg)
        r.connect()
        time.sleep(0.5)
        
        # Read firmware
        fw = r.get_firmware()
        print(f"  [{ch}] Firmware: {fw}")
        
        # Check current enable status of all joints 1 to 7
        st_before = [r.get_joint_enable_status(i) for i in range(1, 8)]
        print(f"  [{ch}] Enable status BEFORE reset (joints 1-7): {st_before}")
        
        # Disable all joints
        print(f"  [{ch}] Sending DISABLE all joints...")
        r.disable()
        time.sleep(0.5)
        st_dis = [r.get_joint_enable_status(i) for i in range(1, 8)]
        print(f"  [{ch}] Enable status AFTER disable (joints 1-7): {st_dis}")
        
        # Enable all joints
        print(f"  [{ch}] Sending ENABLE all joints...")
        r.enable()
        time.sleep(0.5)
        st_en = [r.get_joint_enable_status(i) for i in range(1, 8)]
        print(f"  [{ch}] Enable status AFTER enable (joints 1-7): {st_en}")

        # Detailed per-joint status
        print(f"  [{ch}] Per-joint status details:")
        for j in range(1, 8):
            ds = r.get_driver_states(j)
            if ds is not None:
                foc = ds.msg.foc_status
                print(f"    Joint {j}: enabled={foc.driver_enable_status}, error={foc.driver_error_status}, low_vol={foc.voltage_too_low}, overheat={foc.motor_overheating}")
            else:
                print(f"    Joint {j}: No low-speed driver status frame received.")
        
        r.disconnect()
        print(f"  [{ch}] Done.")
    except Exception as e:
        print(f"  [{ch}] Error during pyAgxArm reset on {ch}: {e}")

print("\n================ Reset Complete ================")
