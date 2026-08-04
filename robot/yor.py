# yor.py — YORv3 hardware node.
#
# Exposes the robot over commlink RPC (port 5557). End-effector control runs
# through whole-body IK (robot/wholebody_control.py), which coordinates both
# arms, the lift and the swerve base as one 18-DOF system.
#
# The RPC surface deliberately mirrors robot/yor_mujoco.py (the simulation
# node) so robot/teleop/wholebody_teleop.py drives either one unchanged — the
# only difference is the port.
#
# Direct, per-subsystem control is kept alongside it: set_base_velocity,
# follow_path / move_to and the lift up/down/stop calls all still work, and
# are what joystick.py uses. Any direct base or lift command suspends the
# whole-body loop's authority over that subsystem for a moment
# (manual_override_timeout_s), so the two controllers never fight over the
# same actuator.
import sys
import functools
import time
import numpy as np
import mink
import atexit
from pathlib import Path
from typing import Optional

# Add project root to sys.path
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot.arm.arm import ArmNode
from robot.base import BaseController
from robot.wholebody_control import WholeBodyController, WholeBodyHardwareConfig
from commlink import RPCServer
from nerolib import FirmwareVersion

THOR_IP = '192.168.1.11'

YOR_PORT = 5557


def require_initialization(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            print(f"Warning: {func.__name__} called before YOR was initialized")
            return None
        return func(self, *args, **kwargs)

    return wrapper


def require_wholebody(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.wholebody is None:
            print(f"Warning: {func.__name__} needs whole-body control (run with arms enabled)")
            return None
        return func(self, *args, **kwargs)

    return wrapper


class YOR():
    def __init__(
        self,
        base_max_vel=np.array((1.0, 1.0, 1.57)),
        base_max_accel=np.array((1.0, 1.0, 1.57)),
        no_arms: bool = False,
        wholebody: bool = True,
        wholebody_config: Optional[WholeBodyHardwareConfig] = None,
    ):
        self._initialized = False

        self.slam_sub = None
        self._reset_nav = False

        self.pose = None        # tuple of ((x,y,z), theta_z, 4x4_pose)

        self.base_controller = BaseController(
            yor=self,
            base_max_vel=base_max_vel,
            base_max_accel=base_max_accel,
            origin=(0.0, 0.0),
            grid_res=0.05,
            control_hz=20,
        )
        self.base = self.base_controller.base
        self.no_arms = no_arms
        self.left_arm = None
        self.right_arm = None
        self.wholebody: Optional[WholeBodyController] = None
        self._wholebody_requested = wholebody and not no_arms
        self._wholebody_config = wholebody_config

        if not self.no_arms:
            self.left_arm = ArmNode(
                can_port="can_left",
                dynamixel_gripper=False,
                firmware_version=FirmwareVersion.DEFAULT,
            )
            self.right_arm = ArmNode(
                can_port="can_right",
                is_left_arm=False,
                dynamixel_gripper=False,
                firmware_version=FirmwareVersion.DEFAULT,
            )

    def init(self):
        if self._initialized:
            print("Warning: YOR already initialized")
            return

        # Start the SparkFlex control loop
        self.base.start_control()
        time.sleep(0.5)

        # No homing needed for Pico lift; ignore if present
        time.sleep(0.5)

        # Arms remain optional
        if not self.no_arms:
            self.left_arm.init()
            self.right_arm.init()

        self._initialized = True

        if self._wholebody_requested:
            self.wholebody = WholeBodyController(
                left_arm=self.left_arm,
                right_arm=self.right_arm,
                base=self.base,
                base_controller=self.base_controller,
                config=self._wholebody_config,
            )
            self.wholebody.start()

    # ─────────────────────────────────────────────────────────────────────────
    # Base — direct control (joystick, nav). Suspends whole-body base authority.
    # ─────────────────────────────────────────────────────────────────────────

    @require_initialization
    def set_base_velocity(self, velocity: np.ndarray):
        if self.wholebody is not None:
            self.wholebody.notify_manual_base_command()
        self.base_controller.mode = "BASE_VEL"
        self.base_controller.target_velocity = velocity

    @require_initialization
    def follow_path(self, path=None):
        if self.wholebody is not None:
            self.wholebody.notify_manual_base_command()
        self.base_controller.slam_sub_init()

        if path is None:
            self.base_controller._path_world = None
            self.base_controller.mode = "BASE_VEL"
            self.base_controller.target_velocity = np.zeros(3, dtype=float)
            print("[YOR] follow_path: cleared")
            return True

        clean = [(float(p[0]), float(p[1])) for p in path]
        self.base_controller._path_world = clean
        self.base_controller.mode = "PATH_FOLLOWING"
        print(f"[YOR] follow_path: n={len(clean)} first={clean[0]} last={clean[-1]}")
        return True

    @require_initialization
    def get_nav_debug(self):
        if hasattr(self.base_controller, "get_nav_debug"):
            return self.base_controller.get_nav_debug()
        return None

    @require_initialization
    def move_to(self, goal = None):
        if self.wholebody is not None:
            self.wholebody.notify_manual_base_command()
        self.base_controller.slam_sub_init()
        self.base_controller._goal = goal
        self.base_controller.mode = "MOVE_TO"

    @require_initialization
    def move_by(self, deltas = None):
        if self.wholebody is not None:
            self.wholebody.notify_manual_base_command()
        self.base_controller.slam_sub_init()
        if self.pose is None:
            print("Warning: move_by called before pose is available")
            return
        if deltas is None:
            print("Warning: move_by called without deltas")
            return
        translation, theta, T_base = self.pose               # (x,y,z), theta_z, 4x4 transform
        x, y = float(translation[0]), float(translation[2])  # (x,z) plane

        self.base_controller._goal = (x+deltas[0], y+deltas[1], theta+deltas[2])
        self.base_controller.mode = "MOVE_TO"

    @require_initialization
    def get_cmd_vel(self):
        # returns ([vx, vy, omega], timestamp)
        v = np.asarray(self.base_controller.target_velocity, dtype=float)
        return v.tolist(), time.time()

    @require_initialization
    def get_base_velocity(self):
        """[vx, vy, omega] the whole-body solver last asked the base for."""
        if self.wholebody is None:
            return np.zeros(3).tolist()
        return self.wholebody.get_base_velocity().tolist()

    @require_initialization
    def get_base_encoders(self) -> dict:
        """Return steer positions (rad) and drive velocities (raw) for all 4 modules."""
        base = self.base
        return {
            "timestamp": time.time(),
            "steer_rad":    [m.get_position_rad()    for m in base.rotation_motors],
            "steer_deg":    [m.get_position_deg()    for m in base.rotation_motors],
            "steer_counts": [m.get_position_counts() for m in base.rotation_motors],
            "drive_vel":    [m.get_velocity_raw()    for m in base.drive_motors],
            "drive_counts": [m.get_position_counts() for m in base.drive_motors],
            "lift_height_m": base.get_lift_height(),
        }

    @require_initialization
    def get_pose(self) -> dict:
        """Return the latest SLAM pose: x, y, theta (yaw in radians).
        x = translation[0], y = translation[2] (robot moves in XZ plane).
        """
        if self.pose is None:
            return {"x": None, "y": None, "theta": None}
        translation, theta, _ = self.pose
        return {
            "x": float(translation[0]),
            "y": float(translation[2]),
            "theta": float(theta),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Lift — direct control (joystick D-pad, scripts)
    # ─────────────────────────────────────────────────────────────────────────

    def _manual_lift(self) -> None:
        if self.wholebody is not None:
            self.wholebody.notify_manual_lift_command()

    @require_initialization
    def lift_up(self) -> None:
        self._manual_lift()
        if hasattr(self.base, "lift_up"):
            self.base.lift_up()

    @require_initialization
    def lift_down(self) -> None:
        self._manual_lift()
        if hasattr(self.base, "lift_down"):
            self.base.lift_down()

    @require_initialization
    def lift_stop(self) -> None:
        self._manual_lift()
        if hasattr(self.base, "lift_stop"):
            self.base.lift_stop()

    @require_initialization
    def lift_home(self) -> None:
        """Send the lift home.

        With whole-body control running this resets the solver's lift target;
        otherwise it falls back to the hardware homing routine, which drives
        the lift straight to its limit switch.
        """
        if self.wholebody is not None:
            self.wholebody.lift_home()
            return
        self._manual_lift()
        self.base.lift_home()

    @require_initialization
    def get_lift_height(self) -> float:
        return self.base.get_lift_height()

    @require_initialization
    def get_lift_status(self) -> dict:
        """Full lift snapshot: height, position-known, homed, limits, motion.

        Requests a fresh `status` from the controller, so the limit-switch
        fields reflect the switches right now. Everything is a plain type, so
        it crosses the RPC boundary unchanged.
        """
        if hasattr(self.base, "get_lift_status"):
            return self.base.get_lift_status()
        return {"available": False}

    @require_initialization
    def lift_position_known(self) -> Optional[bool]:
        """Whether the lift controller has an established zero.

        False means every height it reports is meaningless — run lift_home().
        None means it has not said either way yet.
        """
        if hasattr(self.base, "lift_position_known"):
            return self.base.lift_position_known()
        return None

    @require_initialization
    def get_lift_position(self) -> float:
        """Alias of get_lift_height(), for parity with the simulation node."""
        if self.wholebody is not None:
            return self.wholebody.get_lift_position()
        height = self.base.get_lift_height()
        return float(height) if height is not None else 0.0

    @require_initialization
    def set_lift_target(self, lift_target: float):
        """Ask the whole-body solver for a lift height (metres)."""
        if self.wholebody is None:
            print("[YOR] set_lift_target needs whole-body control; use lift_to_height()")
            return
        self.wholebody.set_lift_target(lift_target)

    @require_initialization
    def lift_delta_height(
        self,
        delta_m: float,
        tolerance_m: float = 0.002,
        timeout_s: float = 30.0,
        min_height_m: float = 0.0,
        max_height_m: float = 0.900,
    ) -> bool:
        if not hasattr(self.base, "lift_delta_height"):
            print("[YOR] base has no lift_delta_height()")
            return False
        self._manual_lift()
        try:
            return bool(self.base.lift_delta_height(
                delta_m,
                tolerance_m=tolerance_m,
                timeout_s=timeout_s,
                min_height_m=min_height_m,
                max_height_m=max_height_m,
            ))
        except TypeError:
            return bool(self.base.lift_delta_height(delta_m))

    @require_initialization
    def lift_to_height(
        self,
        target_m: float,
        tolerance_m: float = 0.002,
        timeout_s: float = 30.0,
        min_height_m: float = 0.0,
        max_height_m: float = 0.900,
    ) -> bool:
        """Blocking absolute lift move, bypassing the whole-body solver."""
        if not hasattr(self.base, "lift_to_height"):
            print("[YOR] base has no lift_to_height()")
            return False
        self._manual_lift()
        return bool(self.base.lift_to_height(
            target_m,
            tolerance_m=tolerance_m,
            timeout_s=timeout_s,
            min_height_m=min_height_m,
            max_height_m=max_height_m,
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # Arms — end-effector control (whole-body)
    # ─────────────────────────────────────────────────────────────────────────

    @require_initialization
    @require_wholebody
    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None,
                           preview_time: float = 0.1):
        self.wholebody.set_left_ee_target(ee_target, gripper_target, preview_time)

    @require_initialization
    @require_wholebody
    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None,
                            preview_time: float = 0.1):
        self.wholebody.set_right_ee_target(ee_target, gripper_target, preview_time)

    @require_initialization
    @require_wholebody
    def set_bimanual_ee_target(self,
                               L_ee_target: mink.SE3, R_ee_target: mink.SE3,
                               L_gripper_target: float | None = None, L_preview_time: float = 0.1,
                               R_gripper_target: float | None = None, R_preview_time: float = 0.1):
        self.wholebody.set_bimanual_ee_target(
            L_ee_target, R_ee_target,
            L_gripper_target=L_gripper_target, R_gripper_target=R_gripper_target,
            L_preview_time=L_preview_time, R_preview_time=R_preview_time,
        )

    @require_initialization
    @require_wholebody
    def home_left_arm(self):
        """Reset the left EE target to its home pose (the solver drives there)."""
        self.wholebody.home_left_arm()

    @require_initialization
    @require_wholebody
    def home_right_arm(self):
        """Reset the right EE target to its home pose (the solver drives there)."""
        self.wholebody.home_right_arm()

    @require_initialization
    @require_wholebody
    def toggle_fix_base(self, fixed: bool | None = None) -> bool:
        """Lock the base in the solver: only the arms and lift move."""
        return self.wholebody.toggle_fix_base(fixed)

    @require_initialization
    @require_wholebody
    def toggle_collision_avoidance(self, enable: bool | None = None) -> bool:
        return self.wholebody.toggle_collision_avoidance(enable)

    @require_initialization
    @require_wholebody
    def toggle_base_motion(self, enable: bool | None = None) -> bool:
        """Allow / forbid the solver from driving the wheels at all."""
        return self.wholebody.toggle_base_motion(enable)

    @require_initialization
    def get_state(self) -> dict:
        """Snapshot for teleop clients (plain types), matching the sim node."""
        if self.wholebody is None:
            return {}
        state = self.wholebody.get_state()
        state["lift"] = self.get_lift_position()
        return state

    @require_initialization
    def get_left_ee_pose(self) -> mink.SE3:
        """Left end-effector pose in the world frame."""
        if self.wholebody is None:
            return None
        return self.wholebody.get_left_ee_pose()

    @require_initialization
    def get_right_ee_pose(self) -> mink.SE3:
        """Right end-effector pose in the world frame."""
        if self.wholebody is None:
            return None
        return self.wholebody.get_right_ee_pose()

    @require_initialization
    def get_arm_relative_pose(self) -> tuple[mink.SE3, mink.SE3]:
        left_ee_pose = self.get_left_ee_pose()
        right_ee_pose = self.get_right_ee_pose()
        l2r = right_ee_pose.inverse() @ left_ee_pose
        r2l = left_ee_pose.inverse() @ right_ee_pose

        return r2l, l2r

    # ─────────────────────────────────────────────────────────────────────────
    # Arms — joint space / grippers (direct)
    # ─────────────────────────────────────────────────────────────────────────

    def _manual_arms(self) -> None:
        if self.wholebody is not None:
            self.wholebody.notify_manual_arm_command()

    @require_initialization
    def set_left_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        if self.no_arms:
            print("left arm disabled")
            return
        self._manual_arms()
        self.left_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_right_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        if self.no_arms:
            print("right arm disabled")
            return
        self._manual_arms()
        self.right_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_left_gain(self, kp: np.ndarray, kd: np.ndarray):
        if self.no_arms:
            print("left arm disabled")
            return
        self.left_arm.set_gain(kp, kd)

    @require_initialization
    def set_right_gain(self, kp: np.ndarray, kd: np.ndarray):
        if self.no_arms:
            print("right arm disabled")
            return
        self.right_arm.set_gain(kp, kd)

    @require_initialization
    def park(self, gripper_target: float = 1.0):
        """Stop whole-body control and send both arms to the hardware home pose.

        Unlike home_left_arm() / home_right_arm(), which only move the solver's
        targets, this hands the arms back to nerolib's homing routine and
        leaves the whole-body loop stopped. Call resume_wholebody() to restart.
        """
        if self.wholebody is not None:
            self.wholebody.stop()
            self.wholebody = None
        if self.no_arms:
            return
        self.left_arm.home(gripper_target)
        self.right_arm.home(gripper_target)

    @require_initialization
    def tuck_arms(self):
        """Stop whole-body control and tuck both arms (zero joint pose)."""
        if self.wholebody is not None:
            self.wholebody.stop()
            self.wholebody = None
        if self.no_arms:
            return
        self.left_arm.tuck_arms()
        self.right_arm.tuck_arms()

    @require_initialization
    def resume_wholebody(self) -> bool:
        """Restart whole-body control after park() / tuck_arms() / emergency_stop()."""
        if self.no_arms:
            print("[YOR] cannot resume whole-body control without arms")
            return False
        if self.wholebody is None:
            self.wholebody = WholeBodyController(
                left_arm=self.left_arm,
                right_arm=self.right_arm,
                base=self.base,
                base_controller=self.base_controller,
                config=self._wholebody_config,
            )
        self.wholebody.start()
        return True

    @require_initialization
    def emergency_stop(self):
        """Freeze everything: wheels stopped, lift stopped, arms held in place."""
        if self.wholebody is not None:
            self.wholebody.emergency_stop()
        self.base_controller.mode = "BASE_VEL"
        self.base_controller.target_velocity = np.zeros(3, dtype=float)
        self.base.set_target_base_velocity(np.zeros(3), smooth=False)

    @require_initialization
    def open_left_gripper(self):
        if self.no_arms:
            print("left arm disabled")
            return
        self.left_arm.open_gripper()

    @require_initialization
    def close_left_gripper(self):
        if self.no_arms:
            print("left arm disabled")
            return
        self.left_arm.close_gripper()

    @require_initialization
    def open_right_gripper(self):
        if self.no_arms:
            print("right arm disabled")
            return
        self.right_arm.open_gripper()

    @require_initialization
    def close_right_gripper(self):
        if self.no_arms:
            print("right arm disabled")
            return
        self.right_arm.close_gripper()

    @require_initialization
    def get_left_joint_positions(self) -> np.ndarray:
        if self.no_arms:
            print("left arm disabled")
            return None
        return self.left_arm.get_joint_positions()

    @require_initialization
    def get_right_joint_positions(self) -> np.ndarray:
        if self.no_arms:
            print("right arm disabled")
            return None
        return self.right_arm.get_joint_positions()

    @require_initialization
    def get_left_gripper_pose(self):
        if self.no_arms:
            print("left arm disabled")
            return None
        return self.left_arm.get_gripper_pose()

    @require_initialization
    def get_right_gripper_pose(self):
        if self.no_arms:
            print("right arm disabled")
            return None
        return self.right_arm.get_gripper_pose()

    @require_initialization
    def get_bimanual_state(self) -> list:
        """
        All bimanual state in one call, for high-speed data logging.
        Flat row: [t, L_ee(7), L_q(7), L_grip, R_ee(7), R_q(7), R_grip, lift].
        """
        row = [0.0] * (1 + 7 + 7 + 1 + 7 + 7 + 1 + 1)
        row[0] = time.time()
        if self.no_arms:
            row[1:8] = [0.90724, -0.41142, 0.075, -0.04495, 0.10741, 0.11358, 0.89066] # roughly tucked
            row[8:15] = [0.0] * 7
            row[15] = 1.0 # fully open
            row[16:23] = [0.90029, 0.42914, 0.06059, 0.04051, 0.10338, -0.53731, 0.89969]
            row[23:30] = [0.0] * 7
            row[30] = 1.0
            row[31] = 0.0
            return row

        left_ee = self.get_left_ee_pose()
        right_ee = self.get_right_ee_pose()
        row[1:8] = left_ee.wxyz_xyz.tolist() if left_ee is not None else [0.0] * 7
        row[8:15] = self.left_arm.get_joint_positions().tolist()
        row[15] = self.left_arm.get_gripper_pose()
        row[16:23] = right_ee.wxyz_xyz.tolist() if right_ee is not None else [0.0] * 7
        row[23:30] = self.right_arm.get_joint_positions().tolist()
        row[30] = self.right_arm.get_gripper_pose()
        lift = self.base.get_lift_height()
        row[31] = float(lift) if lift is not None else 0.0
        return row


def main():
    yor = YOR(no_arms=False)
    yor.init()
    server = RPCServer(yor, port=YOR_PORT, threaded=True)

    def graceful_shutdown():
        print("\nRPC Server stopping...")
        server.stop()

        if yor.wholebody is not None:
            yor.wholebody.stop()

        if not yor.no_arms:
            if yor.left_arm is not None:
                input("\n[YOR] Press ENTER to drop LEFT arm...")
                yor.left_arm.stop()

            if yor.right_arm is not None:
                input("\n[YOR] Press ENTER to drop RIGHT arm...")
                yor.right_arm.stop()

    atexit.register(graceful_shutdown)
    server.start()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
