"""
arm.py — joint-space driver for one AgileX Nero 7-DOF arm.

ArmNode is deliberately *kinematics-free*: it commands joint targets and
reports joint state over nerolib, and nothing more. Cartesian control for
YORv3 lives in robot/arm/wholebody_ik.py, which solves both arms, the lift
and the base together — a per-arm IK solver here would fight it, because
neither would know what the other DOFs were doing.

For end-effector targets and forward kinematics, use
robot/wholebody_control.py (hardware) or robot/yor_mujoco.py (simulation).
"""

import time
from pathlib import Path
from typing import Optional
import numpy as np

try:
    from nerolib import NeroController, ControllerConfig, FirmwareVersion, JointState, Gain, ControlMode, MoveMode
except ImportError:
    print("nerolib not found. Please install it or use the 'nerolib' conda environment.")
    raise

from robot.arm.gripper import Gripper

# Dynamixel gripper caliberated in dxl.py
# DXL_ID_RIGHT = 2, DXL_ID_LEFT = 3
BAUDRATE = 1000000
DXL_ID_RIGHT = 2
DXL_ID_LEFT = 3


class ArmNode:
    def __init__(
        self,
        can_port: str,
        is_left_arm: bool = True,
        dynamixel_gripper: bool = False,
        native_gripper: bool = False,
        default_kp: Optional[float | list[float]] = 10.0,
        default_kd: Optional[float | list[float]] = 1.0,
        gravity_comp_scale: float = 1.0,
        firmware_version=None,
    ):
        _ROOT = Path(__file__).parent.parent.parent
        self.can_port = can_port
        self.is_left_arm = is_left_arm
        # URDF is for nerolib's own dynamics (gravity compensation), not IK.
        if is_left_arm:
            self.urdf_path = (_ROOT / "nerolib/urdf/nero_cone-e_left_fixed.urdf").as_posix()
        else:
            self.urdf_path = (_ROOT / "nerolib/urdf/right_arm_final.urdf").as_posix()

        # Initialize nerolib NeroController
        self.control_mode_set = False
        try:
            print(f"[ArmNode] Initializing {can_port} with nerolib...")
            
            self.config = ControllerConfig()
            self.config.interface_name = can_port
            self.config.urdf_path = self.urdf_path
            # Run nerolib's native interpolation loop at its commissioned rate.
            self.config.controller_freq_hz = 250.0
            
            # Defines home position (originally in ControllerConfig)
            # UPDATED: Using a safe intermediate home based on current readouts to avoid J1 limits/issues
            self.home_position = (
                [0.0, 1.32, -1.71, 1.31, 0.0, 0.0, 0.0] 
                # [1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Modified J1 from 1.38 to -1.38 to match current state sign
                if is_left_arm
                else [0.0, 1.32, 1.71, 1.31, 0.0, 0.0, 0.0]
            )
            
            self.config.home_position = self.home_position
            
            # Live-queried from both arms' firmware over CAN
            # (pyAgxArm get_joint_angle_vel_limits / get_joint_acc_limits),
            # 2026-08-20: joints 1-4 report max_vel=3.14 rad/s, joints 5-7
            # (wrist) report 3.92 rad/s; joints 1-2 report max_acc=2.0
            # rad/s^2, joints 3-7 report 2.5. Set to 95% of each, per joint,
            # rather than the firmware number exactly, leaving headroom for
            # Ruckig/discretization to occasionally nudge over the nominal
            # target. Note MIT mode (all nerolib ever uses) does not actually
            # enforce these firmware values -- they belong to the onboard
            # JOINT/CPV trajectory mode nerolib never engages; MIT mode's own
            # bounds check is only the raw CAN wire-encoding range (position
            # +-12.5 rad, velocity +-45 rad/s, see nero_interface.h), and it
            # checks nothing for acceleration at all. So these are a
            # deliberate software ceiling, not something the firmware would
            # otherwise cap for us: the previous joint_acc_max=15.0 was
            # 6-7.5x the firmware's own reported figure, silently asking for
            # more torque than these joints can produce, without triggering
            # any error -- just tracking lag.
            #
            # 2026-08-20 follow-up: the 95%-of-firmware acceleration above
            # fixed jitter completely on hardware but felt laggy (slow to
            # ramp up / change direction). Raised 1.3x (below) fixed some of
            # that but hardware testing still showed lag -- ~10-32% of
            # samples in real teleop sessions demanded more acceleration
            # than that ceiling allowed (measured by differentiating
            # recorded WBC trajectories; see
            # artifacts/wholebody_logs/trajectories/). Progression since,
            # each superseding the last (all measured the same way -- the
            # per-joint Nth-percentile of real demand from recorded
            # sessions; see that directory for the raw data):
            #   1.3x accel                                    -> some lag
            #   p90 demand, 1 file/session x 3 sessions        -> less lag
            #   p95 demand, 1 file/session x 3 sessions        -> outlier-
            #     event filtering (high-accel + near-zero-EE-motion ticks,
            #     the "elbow up/down flip" check) barely changed this
            #     estimate (<5%), so those rare (~0.1-0.3% of samples)
            #     events aren't what's driving the high joint 3/6 numbers
            #   p95 demand, ALL 28 recorded files pooled (117k samples/
            #     joint), outlier-filtered -- below, active now. More
            #     robust than the 3-file estimates above (e.g. joint 3 came
            #     down from 10.03 to 7.73 once more sessions were included).
            # All of these remain well above the firmware's own reported
            # achievable accel (2.0-2.5 rad/s^2) on several joints, so any
            # of them may reintroduce the original jitter -- if so, revert
            # to whichever earlier line was last confirmed jitter-free.
            self.config.joint_vel_max = [2.98, 2.98, 2.98, 2.98, 3.72, 3.72, 3.72]
            # Previous (1.3x of 95%-of-firmware; smooth, still some lag):
            # self.config.joint_acc_max = [2.47, 2.47, 3.09, 3.09, 3.09, 3.09, 3.09]
            # Previous (p90 demand, 3-file estimate; better, still lag):
            # self.config.joint_acc_max = [4.2, 2.6, 2.6, 7.1, 4.1, 4.8, 7.8]
            # Previous (p95 demand, 3-file estimate, outliers removed):
            self.config.joint_acc_max = [4.98, 3.42, 3.57, 9.57, 5.38, 5.76, 8.99]
            # self.config.joint_acc_max = [4.60, 3.86, 5.05, 7.73, 6.33, 6.33, 8.41]
            
            if default_kp is not None:
                if isinstance(default_kp, (int, float)):
                    self.config.default_kp = [float(default_kp)] * 7
                else:
                    self.config.default_kp = [float(x) for x in default_kp]
            
            if default_kd is not None:
                if isinstance(default_kd, (int, float)):
                    self.config.default_kd = [float(default_kd)] * 7
                else:
                    self.config.default_kd = [float(x) for x in default_kd]

            self.config.gravity_compensation = True
            self.config.gravity_comp_scale = gravity_comp_scale

            if firmware_version is not None:
                self.config.firmware_version = firmware_version

            self.nero = NeroController(self.config)
            
            if not self.nero.start():
                 print(f"[ArmNode] Failed to start NeroController on {can_port}")
                 self.nero = None
            else:
                 print(f"[ArmNode] {can_port} initialized and started.")
                 self.control_mode_set = True
            
        except Exception as e:
            print(f"[ArmNode] Failed to initialize arm on {can_port}: {e}")
            self.nero = None

        self.gripper_target: Optional[float] = None
        self.q_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.dynamixel_gripper = dynamixel_gripper
        # The arm's *own* gripper, driven by nerolib over the arm's CAN bus.
        # YORv3 ships with no gripper on either arm, so this is off by default
        # and gripper values from the whole-body / teleop path are dropped
        # rather than sent to an actuator that is not there. Set it True only
        # when a native gripper is physically fitted.
        self.native_gripper = bool(native_gripper)
        self.gripper = None

        if not self.native_gripper and not self.dynamixel_gripper:
            print(f"[ArmNode] {can_port}: no gripper fitted — gripper control disabled")

        if self.dynamixel_gripper:
            # If your Gripper class uses /dev/ttyUSB0 internally, this will fail when unplugged.
            # Auto-disable instead of crashing.
            try:
                dxl_id = DXL_ID_LEFT if is_left_arm else DXL_ID_RIGHT
                self.gripper = Gripper(baudrate=BAUDRATE, dxl_id=dxl_id)

                open_gripper_value, close_gripper_value = self.gripper.dxl.calibrate_motor()
                self.open_gripper_value = open_gripper_value
                self.close_gripper_value = close_gripper_value
                self.gripper_range = open_gripper_value - close_gripper_value

                print("[ArmNode] Dynamixel gripper enabled")

            except (FileNotFoundError, OSError) as e:
                print(f"[ArmNode] Dynamixel gripper not found ({e}). Continuing without gripper.")
                self.dynamixel_gripper = False
                self.gripper = None







    def init(self) -> bool:
        """Move all seven joints through nerolib's coordinated home trajectory."""
        if self.nero is None:
            print("[ArmNode] Warning: Nero not initialized, init() skipping hardware calls.")
            return False

        # Move to home position
        print(f"[ArmNode] Moving to home position...")
        self.nero.reset_to_home()
        
        q = self.get_joint_positions()
        print(f"q_reached: {np.round(q, 4)}")
        return True


    def home(self, gripper_target: float = 1.0):
        if self.nero:
            try:
                self.nero.reset_to_home()
            except Exception as e:
                print(f"[ArmNode] Home failed: {e}")
        
        if self.dynamixel_gripper:
            self.gripper.move_to_pos(int(gripper_target * self.gripper_range + self.close_gripper_value))
        time.sleep(2.0)

    def tuck_arms(self):
        self.set_joint_target(np.zeros(7), gripper_target=1.00, preview_time=2.0)

    def set_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.01
    ):
        if self.nero:
            try:
                # nerolib expects lists or std::vectors, typically python lists/arrays work with pybind11
                # The API signature: set_target(new_target_pos, new_target_gripper_pos, minimum_duration, new_target_vel, new_target_acc)
                # We map joint_target to new_target_pos.
                
                target_pos = (joint_target + self.q_offset).tolist()
                
                # Nerolib expects normalized gripper position (0 for close, 1 for fully open).
                # With no native gripper fitted the field still has to be sent, but it is
                # pinned open so a teleop gripper value can never command a missing
                # actuator; the dynamixel gripper, when present, is driven separately below.
                target_gripper = 1.0
                if gripper_target is not None and self.native_gripper:
                    target_gripper = float(gripper_target)
                
                self.nero.set_target(
                    new_target_pos=target_pos,
                    new_target_gripper_pos=target_gripper if not self.dynamixel_gripper else 0.0, # Don't conflict if dyn gripper used
                    minimum_duration=preview_time
                )
                
            except Exception as e:
                 print(f"Set target failed: {e}")

        if gripper_target is not None and self.dynamixel_gripper:
            self.gripper.move_to_pos(int(gripper_target * self.gripper_range + self.close_gripper_value))

    def open_gripper(self):
        if self.dynamixel_gripper:
            self.gripper.move_to_pos(self.open_gripper_value)
        elif self.native_gripper:
            q = self.get_joint_positions()
            self.set_joint_target(q, gripper_target=1.0)  
            time.sleep(0.5)

    def close_gripper(self):
        if self.dynamixel_gripper:
            self.gripper.move_to_pos(self.close_gripper_value)
        elif self.native_gripper:
            q = self.get_joint_positions()
            self.set_joint_target(q, gripper_target=0.0)
            time.sleep(0.5)

    def set_gain(self, kp, kd):
        """
        Set Kp and Kd gains for the controller.
        kp, kd can be float (applied to all joints) or 7-element lists/arrays.
        """
        if self.nero:
            gain = Gain()
            if isinstance(kp, (int, float)):
                gain.kp = [float(kp)] * 7
            else:
                gain.kp = [float(x) for x in kp]
            
            if isinstance(kd, (int, float)):
                gain.kd = [float(kd)] * 7
            else:
                gain.kd = [float(x) for x in kd]
            
            self.nero.set_gain(gain)

    def set_gravity_comp(self, enable: bool):
        if self.nero:
            self.nero.enable_gravity_compensation(enable)

    def set_gravity_comp_scale(self, scale: float):
        if self.nero:
            self.nero.set_gravity_comp_scale(scale)

    def set_mode(self, control_mode: ControlMode, move_mode: MoveMode):
        if self.nero:
            self.nero.set_mode(control_mode, move_mode)

    def set_compliant_mode(self, kp=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], kd=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]):
        """preset for manual guidance. Default: Stiffness=0, Damping=0.5."""
        self.sync_target()
        self.set_gain(kp, kd)
        self.set_gravity_comp(True)
        self.set_gravity_comp_scale(1.0)

    def set_spring_mode(self, kp=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], kd=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]):
        """preset for springy behavior. Default: Stiffness=2, Damping=0.3."""
        self.sync_target()
        self.set_gain(kp, kd)
        self.set_gravity_comp(True)
        self.set_gravity_comp_scale(1.0)

    def set_firm_mode(self, kp=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0], kd=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]):
        """preset for accurate tracking. Default: Stiffness=15, Damping=0.8."""
        self.sync_target()
        self.set_gain(kp, kd)
        self.set_gravity_comp(True)
        self.set_gravity_comp_scale(1.0 )

    def set_admittance_mode(self, kp=[25.0, 25.0, 20.0, 20.0, 15.0, 15.0, 15.0], kd=[1.2, 1.2, 1.0, 1.0, 0.8, 0.8, 0.8]):
        """
        High stiffness and damping, optimized for a high-level admittance loop.
        """
        self.sync_target()
        self.set_gain(kp, kd)
        self.set_gravity_comp(True)
        self.set_gravity_comp_scale(1.0)

    def sync_target(self):
        """Update the controller's target to match the current arm position."""
        if self.nero:
            q = self.get_joint_positions()
            self.nero.set_target(
                new_target_pos=(q + self.q_offset).tolist(),
                new_target_gripper_pos=self.get_gripper_pose(),
                minimum_duration=0.0
            )
            # Give the 250 Hz C++ control loop a few cycles to ingest this
            # and 'snap' its internal trajectory state before we change gains.
            time.sleep(0.02)

    def get_joint_positions(self) -> np.ndarray:
        if self.nero is None:
             return np.zeros(7)
             
        try:
            state = self.nero.get_current_state()
            q = np.array(state.pos)
            return q - self.q_offset
        except Exception:
            return np.zeros(7)

    def get_joint_velocities(self) -> np.ndarray:
        if self.nero is None:
             return np.zeros(7)
             
        try:
            state = self.nero.get_current_state()
            return np.array(state.vel)
        except Exception:
            return np.zeros(7)

    def get_joint_torques(self) -> np.ndarray:
        if self.nero is None:
             return np.zeros(7)
             
        try:
            state = self.nero.get_current_state()
            return np.array(state.torque)
        except Exception:
            return np.zeros(7)

    def get_gripper_pose(self):
        if not self.dynamixel_gripper:
            if self.nero:
                state = self.nero.get_current_state()
                return state.gripper_pos
            return 0.0 
        else:
            unnormalized_pos = self.gripper.dxl.get_present_position()
            return float(unnormalized_pos - self.close_gripper_value) / float(
                self.gripper_range
            ) 

    def set_q_offset(self, q_offset: np.ndarray):
        self.q_offset = q_offset

    def stop(self):
        print("called stop")
        if self.nero:
             self.nero.stop()

if __name__ == "__main__":
    left_arm = ArmNode(can_port="can_left", is_left_arm=True)
    # right_arm = ArmNode(can_port="can_right", is_left_arm=False)
    input("Press enter to init left arm")
    left_arm.init()
    # input("Press enter to init right arm")
    # right_arm.init()
    input("Press enter to close gripper")
    left_arm.close_gripper()
    # right_arm.close_gripper()
    input("Press enter to open gripper")
    left_arm.open_gripper()
    # right_arm.open_gripper()
    input("Press enter to exit")
    left_arm.stop()
    # right_arm.stop()
    exit()
