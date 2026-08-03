# yor.py
import functools
import time
import numpy as np
import mink
import atexit
from pathlib import Path

# Import Base from either package layout (robot/base.py) or flat (base.py)
from robot.base import Base

from robot.arm.arm import ArmNode
from robot.base import BaseController
from commlink import RPCServer, Subscriber
import threading

THOR_IP = '194.168.1.11'
ZED_PUB_PORT = 6000
POSE_TOPIC = "zed/pose"

YOR_PORT = 5557

def require_initialization(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            print(f"Warning: {func.__name__} called before YOR was initialized")
            return None
        return func(self, *args, **kwargs)

    return wrapper

def require_zed(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._zed_initialized:
            print(f"Warning: {func.__name__} called before Zed Subscriber was initialized")
            return None
        return func(self, *args, **kwargs)

    return wrapper

# class x(BaseController):
#     def __init__(self):
#         super(self, BaseController).__init__(
#             # contructor params
#         )
#         self.njhdj = dhajhd
        


class YOR():
    def __init__(
        self,
        base_max_vel=np.array((1.0, 1.0, 1.57)),
        base_max_accel=np.array((1.0, 1.0, 1.57)),
        no_arms: bool = False,  # default to True since Base is independent now
    ):
        self._initialized = False
        self._zed_initialized = False

        self.zed_sub = None
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
        # New Base uses SparkFlex drive (IDs 1–4) + rotation (IDs 5–8)
        # base -> BaseMotorController
        # Base(max_vel, max_accel, pose_host, pose_port)
        # base thread: control thread for move_to, track_path, set_velocity
            # these should interrupt each other
        self.base = self.base_controller.base
        self.no_arms = no_arms
        if not self.no_arms:
            _HERE = Path(__file__).parent
            self.left_arm = ArmNode(
                can_port="can_left",
                mjcf_path=(_HERE / "yor-description/robot_mujoco.xml").as_posix(),
                dynamixel_gripper=False,
            )
            self.right_arm = ArmNode(
                can_port="can_right",
                mjcf_path=(_HERE / "yor-description/robot_mujoco.xml").as_posix(),
                is_left_arm=False,
                dynamixel_gripper=False,
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

    # Base
    @require_initialization
    def set_base_velocity(self, velocity: np.ndarray):
        self.base_controller.mode = "BASE_VEL"
        self.base_controller.target_velocity = velocity

    # @require_initialization   
    # def follow_path(self, path = None):
    #     self.base_controller.zed_sub_init()
    #     self.base_controller._path_world = path
    #     self.base_controller.mode = "PATH_FOLLOWING"
    #     return {"ok": True, "n": 0 if path is None else len(path)}
    
    @require_initialization
    def follow_path(self, path=None):
        self.base_controller.zed_sub_init()

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
        self.base_controller.zed_sub_init()
        self.base_controller._goal = goal
        self.base_controller.mode = "MOVE_TO"
    
    @require_initialization   
    def move_by(self, deltas = None):
        self.base_controller.zed_sub_init()
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


    # Lift controls (lift uses Pico serial up/down/stop)
    @require_initialization
    def lift_up(self) -> None:
        if hasattr(self.base, "lift_up"):
            self.base.lift_up()

    @require_initialization
    def lift_down(self) -> None:
        if hasattr(self.base, "lift_down"):
            self.base.lift_down()

    @require_initialization
    def lift_stop(self) -> None:
        if hasattr(self.base, "lift_stop"):
            self.base.lift_stop()

    @require_initialization
    def lift_home(self) -> None:
        self.base.lift_home()

    @require_initialization
    def get_lift_height(self) -> float:
        return self.base.get_lift_height()

    @require_initialization
    def lift_delta_height(
        self,
        delta_m: float,
        tolerance_m: float = 0.002,
        timeout_s: float = 30.0,
        min_height_m: float = 0.0,
        max_height_m: float = 0.5,
    ) -> bool:
        if not hasattr(self.base, "lift_delta_height"):
            print("[YOR] base has no lift_delta_height()")
            return False
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
        max_height_m: float = 0.5,
    ) -> bool:
        if not hasattr(self.base, "lift_to_height"):
            print("[YOR] base has no lift_to_height()")
            return False
        return bool(self.base.lift_to_height(
            target_m,
            tolerance_m=tolerance_m,
            timeout_s=timeout_s,
            min_height_m=min_height_m,
            max_height_m=max_height_m,
        ))


    #-------------------ARMS---------------------
    # Left arm
    @require_initialization
    def set_left_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        if self.no_arms:
            print("left arm disabled")
            return
        self.left_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None, preview_time: float = 0.1):
        if self.no_arms:
            print("left arm disabled")
            return
        self.left_arm.set_ee_target(ee_target, gripper_target, preview_time)

    @require_initialization
    def set_left_gain(self, kp: np.ndarray, kd: np.ndarray):
        if self.no_arms:
            print("left arm disabled")
            return
        self.left_arm.set_gain(kp, kd)

    @require_initialization
    def home_left_arm(self, gripper_target: float = 0.0):
        if self.no_arms:
            print("left arm disabled")
            return
        # Delegate to ArmNode if it provides homing; ignore if not available
        if hasattr(self.left_arm, "home"):
            self.left_arm.home(gripper_target)
        else:
            print("left arm: home() not available")

    @require_initialization
    def tuck_left_arm(self):
        if self.no_arms:
            print("left arm disabled")
            return
        self.left_arm.tuck_arms()

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
    def get_left_ee_pose(self) -> mink.SE3:
        """
        Returns the pose of the left arm's end effector in the world frame (qw, qx, qy, qz, x, y, z).
        """
        if self.no_arms:
            print("left arm disabled")
            return None
        if hasattr(self.left_arm, "get_ee_pose"):
            return self.left_arm.get_ee_pose()
        print("left arm: get_ee_pose() not available")
        return None

    @require_initialization
    def get_left_joint_positions(self) -> np.ndarray:
        if self.no_arms:
            print("left arm disabled")
            return None
        return self.left_arm.get_joint_positions()

    @require_initialization
    def get_left_gripper_pose(self):
        if self.no_arms:
            print("left arm disabled")
            return None
        return self.left_arm.get_gripper_pose()

    # Right arm
    @require_initialization
    def set_right_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        if self.no_arms:
            print("right arm disabled")
            return
        self.right_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None, preview_time: float = 0.1):
        if self.no_arms:
            print("right arm disabled")
            return
        self.right_arm.set_ee_target(ee_target, gripper_target, preview_time)

    @require_initialization
    def set_right_gain(self, kp: np.ndarray, kd: np.ndarray):
        if self.no_arms:
            print("right arm disabled")
            return
        self.right_arm.set_gain(kp, kd)

    @require_initialization
    def home_right_arm(self, gripper_target: float = 1.0):
        if self.no_arms:
            print("right arm disabled")
            return
        if hasattr(self.right_arm, "home"):
            self.right_arm.home(gripper_target)
        else:
            print("right arm: home() not available")

    @require_initialization
    def tuck_right_arm(self):
        if self.no_arms:
            print("right arm disabled")
            return
        self.right_arm.tuck_arms()

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
    def get_right_ee_pose(self) -> mink.SE3:
        if self.no_arms:
            print("right arm disabled")
            return None
        if hasattr(self.right_arm, "get_ee_pose"):
            return self.right_arm.get_ee_pose()
        print("right arm: get_ee_pose() not available")
        return None

    @require_initialization
    def get_right_joint_positions(self) -> np.ndarray:
        if self.no_arms:
            print("right arm disabled")
            return None
        return self.right_arm.get_joint_positions()

    @require_initialization
    def get_right_gripper_pose(self):
        if self.no_arms:
            print("right arm disabled")
            return None
        return self.right_arm.get_gripper_pose()

    @require_initialization
    def get_arm_relative_pose(self) -> tuple[mink.SE3, mink.SE3]:
        left_ee_pose = self.get_left_ee_pose()
        right_ee_pose = self.get_right_ee_pose()
        l2r = right_ee_pose.inverse() @ left_ee_pose
        r2l = left_ee_pose.inverse() @ right_ee_pose

        return r2l, l2r

    @require_initialization
    def get_bimanual_state(self) -> dict:
        """
        Get all bimanual state data in a single call for high-speed data logging.
        Returns dict with left/right ee poses and joint positions and lift position.
        Note: Lift position is 0.0 as new hardware doesn't support position feedback.
        """
        row = [0.0] * (1 + 7 + 6 + 1 + 7 + 6 + 1 + 1)
        if not self.no_arms:
            row[0] = time.time()
            row[1:8] = self.left_arm.get_ee_pose().wxyz_xyz.tolist()
            row[8:14] = self.left_arm.get_joint_positions().tolist()
            row[14] = self.left_arm.get_gripper_pose()
            row[15:22] = self.right_arm.get_ee_pose().wxyz_xyz.tolist()
            row[22:28] = self.right_arm.get_joint_positions().tolist()
            row[28] = self.right_arm.get_gripper_pose()
            row[29] = 0.0  # lift position not available on new hardware
            return row
        else:
            row[0] = time.time()
            row[1:8] = [0.90724, -0.41142, 0.075, -0.04495, 0.10741, 0.11358, 0.89066] # roughly tucked position
            row[8:14] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # tucked position
            row[14] = 1.0 # fully open
            row[15:22] = [0.90029, 0.42914, 0.06059, 0.04051, 0.10338, -0.53731, 0.89969]
            row[22:28] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # tucked position
            row[28] = 1.0
            row[29] = 0.0  # lift position not available on new hardware
            return row

    @require_initialization
    def set_bimanual_ee_target(self,
                           L_ee_target: mink.SE3, R_ee_target: mink.SE3,
                           L_gripper_target: float | None = None, L_preview_time: float = 0.1,
                           R_gripper_target: float | None = None, R_preview_time: float = 0.1):
        self.left_arm.set_ee_target(L_ee_target, L_gripper_target, L_preview_time)
        self.right_arm.set_ee_target(R_ee_target, R_gripper_target, R_preview_time)

    @require_initialization
    def get_cmd_vel(self):
        # returns ([vx, vy, omega], timestamp)
        v = np.asarray(self.base_controller.target_velocity, dtype=float)
        return v.tolist(), time.time()

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
            "battery_voltage_v": base.get_battery_voltage(),
            "battery_percent": base.get_battery_percent(),
        }

    @require_initialization
    def get_battery_voltage(self) -> float | None:
        if hasattr(self.base, "get_battery_voltage"):
            return self.base.get_battery_voltage()
        return None

    @require_initialization
    def get_battery_percent(self) -> float | None:
        if hasattr(self.base, "get_battery_percent"):
            return self.base.get_battery_percent()
        return None

    @require_initialization
    def get_pose(self) -> dict:
        """Return ZED IMU-fused pose: x, y, theta (yaw in radians).
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

    @require_initialization
    def set_lift_velocity(self, velocity: float) -> None:
        """
        Command lift direction for the whole-body controller.
        velocity > 0  → lift_up(), velocity < 0 → lift_down(), 0 → lift_stop().
        """
        if velocity > 0:
            self.lift_up()
        elif velocity < 0:
            self.lift_down()
        else:
            self.lift_stop()

    # ──────────────────────────────────────────────────────────────────────────
    # Whole-body control
    # ──────────────────────────────────────────────────────────────────────────

    def init_whole_body(self, mjcf_path: str | None = None, control_hz: float = 20.0, **ik_kw) -> None:
        """
        Instantiate and start the WholeBodyController.

        mjcf_path defaults to yor-description/robot_mujoco.xml next to this file.
        Must be called after init().
        """
        from robot.whole_body_controller import WholeBodyController

        if mjcf_path is None:
            mjcf_path = (Path(__file__).parent / "yor-description" / "robot_mujoco.xml").as_posix()

        self._wb_controller = WholeBodyController(self, mjcf_path, control_hz=control_hz, **ik_kw)
        self._wb_controller.start()

    @require_initialization
    def set_whole_body_target(
        self,
        left_xyz:    "np.ndarray | None" = None,
        right_xyz:   "np.ndarray | None" = None,
        camera_xyz:  "np.ndarray | None" = None,
        orient_wxyz: "np.ndarray | None" = None,
    ) -> None:
        """
        Set EE position targets in MuJoCo Z-up world frame (= SLAM XZ-plane + height).

          left_xyz / right_xyz  : [x, y, z]  desired gripper position  [m]
          camera_xyz            : [x, y, z]  desired ZED camera position [m]
          orient_wxyz           : shared orientation quaternion [w, x, y, z]
                                  None = identity

        Requires init_whole_body() to have been called first.
        """
        if not hasattr(self, "_wb_controller"):
            print("[YOR] call init_whole_body() first")
            return
        self._wb_controller.set_target_xyz(left_xyz, right_xyz, camera_xyz, orient_wxyz)

    @require_initialization
    def stop_whole_body(self) -> None:
        """Stop the whole-body control loop."""
        if hasattr(self, "_wb_controller"):
            self._wb_controller.stop()


def main():    
    yor = YOR(no_arms=True)
    yor.init()
    server = RPCServer(yor, port=YOR_PORT, threaded = True)
    atexit.register(server.stop)
    server.start()
    while True:
        time.sleep(1)



if __name__ == "__main__":
    main()
