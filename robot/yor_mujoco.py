import threading
import atexit
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

# Inject the repo root into sys.path so 'robot' and 'commlink' import cleanly
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from robot.arm.wholebody_ik import WholeBodyIK, WholeBodyIKConfig, DEFAULT_SCENE
from commlink import RPCServer


class YORMujoco:
    """
    Simulation node for YORv3 — the sim twin of robot/yor.py.

    Runs WholeBodyIK (18 DOF: base + lift + 2×arms) in kinematic mode and
    serves the same RPC surface as the hardware node, so
    robot/teleop/wholebody_teleop.py drives either by changing only the port.
    """

    _SWERVE_MODULES = (
        "front_left",
        "front_right",
        "back_right",
        "back_left",
    )
    _TIRE_RADIUS_M = 0.0381
    _MAX_STEER_RATE_RAD_S = 8.0
    _SWERVE_SPEED_DEADBAND_M_S = 0.02
    _ARM_HOME_LIFT_M = 0.450
    _ARM_HOME_DURATION_S = 3.0
    _ARM_HOME_TIMEOUT_S = 20.0

    def __init__(self, mjcf_path: Optional[str] = None, solver_dt: float = 1.0 / 108.0):
        self.mjcf_path = str(mjcf_path or DEFAULT_SCENE)
        self.solver_dt = solver_dt

        # ── Initialize IK Solver ──────────────────────────────────────────────
        # Same tuned weights the hardware controller uses: base reluctant to
        # roll, lift eager to stretch.
        cfg = WholeBodyIKConfig(
            dt=self.solver_dt,
            solver="pyqpmad",
            max_iters=10,
            base_posture_cost=1e-1,
            lift_posture_cost=1e-4,
            arm_posture_cost=1e-3,
        )
        self.ik = WholeBodyIK(scene_xml=self.mjcf_path, config=cfg)
        self.ik.init_from_keyframe("home")

        self.model = self.ik.model
        self.data = self.ik.data
        self._init_swerve_animation()

        # ── Launch Viewer ─────────────────────────────────────────────────────
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        # ── State / Targets ───────────────────────────────────────────────────
        self.target_lock = threading.Lock()
        # Initialize targets to the current forward kinematics
        T_l, T_r = self.ik.forward_kinematics()
        self.left_ee_target: mink.SE3 = T_l.copy()
        self.right_ee_target: mink.SE3 = T_r.copy()
        # Home poses (for home_left_arm / home_right_arm RPC)
        self._home_left: mink.SE3 = T_l.copy()
        self._home_right: mink.SE3 = T_r.copy()
        self._home_lift: float = float(self.data.qpos[self.ik._lift_qpos_adr])
        self._home_left_q = self.data.qpos[self.ik._left_arm_qpos_adrs].copy()
        self._home_right_q = self.data.qpos[self.ik._right_arm_qpos_adrs].copy()
        self.lift_target: Optional[float] = None
        self.base_fixed: bool = False
        self._last_base_velocity = np.zeros(3)
        self._homing_lock = threading.Lock()
        self._homing_request: Optional[dict] = None

        # ── Control Loop ──────────────────────────────────────────────────────
        self.control_loop_thread: Optional[threading.Thread] = None
        self.control_loop_running = False

    # ── RPC API / External Commands ───────────────────────────────────────────

    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float = 0.0, preview_time: float = 0.0):
        with self.target_lock:
            self.left_ee_target = ee_target

    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: float = 0.0, preview_time: float = 0.0):
        with self.target_lock:
            self.right_ee_target = ee_target

    def set_lift_target(self, lift_target: float):
        """Set the desired lift height in metres (0.0 to 0.92)"""
        with self.target_lock:
            self.lift_target = lift_target

    def toggle_fix_base(self, fixed: Optional[bool] = None) -> bool:
        """Lock the mobile base in place, only arms and lift will move."""
        with self.target_lock:
            self.base_fixed = self.ik.toggle_fix_base(fixed)
            return self.base_fixed

    def get_left_ee_pose(self) -> mink.SE3:
        T_l, _ = self.ik.forward_kinematics()
        return T_l

    def get_right_ee_pose(self) -> mink.SE3:
        _, T_r = self.ik.forward_kinematics()
        return T_r

    def get_left_joint_positions(self) -> np.ndarray:
        # Return the 7-DOF arm positions
        return self.data.qpos[self.ik._left_arm_qpos_adrs].copy()

    def get_right_joint_positions(self) -> np.ndarray:
        # Return the 7-DOF arm positions
        return self.data.qpos[self.ik._right_arm_qpos_adrs].copy()

    def set_bimanual_ee_target(
        self,
        L_ee_target: mink.SE3, R_ee_target: mink.SE3,
        L_gripper_target: float = 0.0, R_gripper_target: float = 0.0,
        L_preview_time: float = 0.0, R_preview_time: float = 0.0,
    ):
        """Set both EE targets atomically (single lock acquisition)."""
        with self.target_lock:
            self.left_ee_target = L_ee_target
            self.right_ee_target = R_ee_target

    def toggle_collision_avoidance(self, enable: Optional[bool] = None) -> bool:
        """Enable/disable the solver's self-collision avoidance constraint."""
        with self.target_lock:
            return self.ik.toggle_collision_avoidance(enable)

    def get_base_velocity(self) -> np.ndarray:
        """Current [vx, vy, omega_z] of the mobile base from the last IK solve."""
        with self.target_lock:
            return self._last_base_velocity.copy()

    def get_lift_position(self) -> float:
        """Current lift height in metres."""
        return float(self.data.qpos[self.ik._lift_qpos_adr])

    def get_state(self) -> dict:
        """Snapshot of the sim state for teleop clients (plain types only)."""
        with self.target_lock:
            base_vel = self._last_base_velocity.copy()
        T_l, T_r = self.ik.forward_kinematics()
        q = self.data.qpos
        return {
            "left_ee_wxyz_xyz": T_l.wxyz_xyz.tolist(),
            "right_ee_wxyz_xyz": T_r.wxyz_xyz.tolist(),
            "lift": float(q[self.ik._lift_qpos_adr]),
            "base_xytheta": q[self.ik.base_qpos_adrs].tolist(),
            "base_velocity": base_vel.tolist(),
            "fix_base": self.ik.fix_base,
            "collision_avoidance": self.ik.avoid_collisions,
            # Present for parity with the hardware node, where base motion can
            # be disabled independently of fix_base.
            "base_motion_enabled": True,
            "swerve_steer_angles": self._swerve_steer_angles.tolist(),
            "swerve_wheel_angles": self._swerve_wheel_angles.tolist(),
            "left_joint_positions": q[self.ik._left_arm_qpos_adrs].tolist(),
            "right_joint_positions": q[self.ik._right_arm_qpos_adrs].tolist(),
        }

    def _home_arm_joints(self, sides: tuple[str, ...]) -> bool:
        """Animate the same ordered Quest homing sequence used on hardware."""
        if not self._homing_lock.acquire(blocking=False):
            print("[sim] Quest home already in progress")
            return False

        request = {
            "sides": sides,
            "stage": "lift",
            "previous_fix_base": False,
            "event": threading.Event(),
            "success": False,
        }
        try:
            with self.target_lock:
                request["previous_fix_base"] = bool(self.ik.fix_base)
                self.base_fixed = self.ik.toggle_fix_base(True)
                self.lift_target = self._ARM_HOME_LIFT_M
                request["lift_q_start"] = float(
                    self.data.qpos[self.ik._lift_qpos_adr]
                )
                request["lift_started_at"] = time.monotonic()
                request["lift_duration"] = max(
                    abs(request["lift_q_start"] - self._ARM_HOME_LIFT_M) / 0.15,
                    0.1,
                )
                self._homing_request = request
            print("[sim] Quest home: base locked; lift -> 450 mm")

            if not request["event"].wait(self._ARM_HOME_TIMEOUT_S):
                with self.target_lock:
                    if self._homing_request is request:
                        self.base_fixed = self.ik.toggle_fix_base(
                            request["previous_fix_base"]
                        )
                        self._homing_request = None
                print("[sim] Quest home timed out")
                return False
            return bool(request["success"])
        finally:
            self._homing_lock.release()

    def _finish_arm_home(self, request: dict, success: bool) -> None:
        """Finish a control-loop-owned homing request and wake its RPC call."""
        with self.target_lock:
            if self._homing_request is not request:
                return
            if success:
                # Latch both current poses so the normal IK loop resumes from
                # the completed joint-space pose without pulling either arm.
                T_l, T_r = self.ik.forward_kinematics()
                self.left_ee_target = T_l.copy()
                self.right_ee_target = T_r.copy()
                self.lift_target = self._ARM_HOME_LIFT_M
            self.base_fixed = self.ik.toggle_fix_base(
                request["previous_fix_base"]
            )
            self._homing_request = None
            request["success"] = bool(success)
            request["event"].set()

    def home_left_arm(self) -> bool:
        """Quest Y: lock base, lift to 450 mm, then home all left joints."""
        return self._home_arm_joints(("left",))

    def home_right_arm(self) -> bool:
        """Quest B: lock base, lift to 450 mm, then home all right joints."""
        return self._home_arm_joints(("right",))

    def home_arms(self) -> bool:
        """Quest Y+B: home both arms after one base/lift preamble."""
        return self._home_arm_joints(("left", "right"))

    def lift_home(self):
        """Reset the lift target to its home height."""
        with self.target_lock:
            self.lift_target = self._home_lift

    def init(self):
        self.start_control()

    # ── Thread Management ─────────────────────────────────────────────────────

    def start_control(self):
        if self.control_loop_thread is not None:
            return
        self.control_loop_running = True
        self.control_loop_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_loop_thread.start()

    def stop_control(self):
        if self.control_loop_thread is None:
            return
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None
        self.viewer.close()

    def control_loop(self):
        freq = int(1.0 / self.solver_dt)
        rate_limiter = RateLimiter(freq, warn=False)
        
        while self.control_loop_running and self.viewer.is_running():
            with self.target_lock:
                T_l = self.left_ee_target.copy()
                T_r = self.right_ee_target.copy()
                lift_tgt = self.lift_target
                homing = self._homing_request

            # The physical sequence moves the column while every arm joint
            # holds, so animate that DOF directly instead of asking whole-body
            # IK to keep the end effectors fixed while the lift rises.
            if homing is not None and homing["stage"] == "lift":
                elapsed = time.monotonic() - homing["lift_started_at"]
                u = float(np.clip(elapsed / homing["lift_duration"], 0.0, 1.0))
                blend = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
                self.data.qpos[self.ik._lift_qpos_adr] = (
                    (1.0 - blend) * homing["lift_q_start"]
                    + blend * self._ARM_HOME_LIFT_M
                )
                self.ik.update_configuration(self.data.qpos)
                self._animate_swerve(np.zeros(3))
                mujoco.mj_forward(self.model, self.data)
                with self.target_lock:
                    self._last_base_velocity = np.zeros(3)
                    if self._homing_request is homing and u >= 1.0:
                        homing["stage"] = "arms"
                        homing["arm_started_at"] = time.monotonic()
                        homing["left_q_start"] = self.data.qpos[
                            self.ik._left_arm_qpos_adrs
                        ].copy()
                        homing["right_q_start"] = self.data.qpos[
                            self.ik._right_arm_qpos_adrs
                        ].copy()
                        print("[sim] Quest home: lift at 450 mm; homing arm joints")
                self.viewer.sync()
                rate_limiter.sleep()
                continue

            # Once the lift has reached 450 mm, animate the requested arm
            # joints directly to their keyframe values.  This is deliberately
            # joint-space motion: an EE-only target is under-constrained and
            # cannot guarantee that all seven joints actually reach home.
            if homing is not None and homing["stage"] == "arms":
                elapsed = time.monotonic() - homing["arm_started_at"]
                u = float(np.clip(elapsed / self._ARM_HOME_DURATION_S, 0.0, 1.0))
                blend = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
                if "left" in homing["sides"]:
                    self.data.qpos[self.ik._left_arm_qpos_adrs] = (
                        (1.0 - blend) * homing["left_q_start"]
                        + blend * self._home_left_q
                    )
                if "right" in homing["sides"]:
                    self.data.qpos[self.ik._right_arm_qpos_adrs] = (
                        (1.0 - blend) * homing["right_q_start"]
                        + blend * self._home_right_q
                    )
                self.ik.update_configuration(self.data.qpos)
                self._animate_swerve(np.zeros(3))
                mujoco.mj_forward(self.model, self.data)
                with self.target_lock:
                    self._last_base_velocity = np.zeros(3)
                self.viewer.sync()
                if u >= 1.0:
                    print(
                        "[sim] Quest home: "
                        + " + ".join(homing["sides"])
                        + " arm joints home"
                    )
                    self._finish_arm_home(homing, True)
                rate_limiter.sleep()
                continue

            # Sync IK with current data
            self.ik.update_configuration(self.data.qpos)

            # Solve Whole-Body IK
            result = self.ik.solve(T_l, T_r, lift_target=lift_tgt)

            # Apply in Kinematic Mode
            self.ik.apply_to_sim_kinematic(self.data, result)
            self._animate_swerve(result.base_velocity)
            mujoco.mj_forward(self.model, self.data)

            with self.target_lock:
                self._last_base_velocity = result.base_velocity

            self.viewer.sync()
            rate_limiter.sleep()

        self.control_loop_running = False
        with self.target_lock:
            abandoned_home = self._homing_request
        if abandoned_home is not None:
            self._finish_arm_home(abandoned_home, False)

    # ── Swerve visualization ────────────────────────────────────────────────

    @staticmethod
    def _wrap_pi(angle):
        return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def _init_swerve_animation(self) -> None:
        """Cache model-derived module geometry and animation joint addresses.

        The IK moves the chassis through its three planar joints.  Steering and
        wheel joints are deliberately outside the 18 IK DOFs, so without this
        visualization layer the chassis slides while all four modules remain
        frozen.  Module anchors and each wheel's zero-angle rolling direction
        come from the MJCF rather than a second set of hand-written positions.
        """
        steer_joints = [
            self.model.joint(f"{name}_steer_joint")
            for name in self._SWERVE_MODULES
        ]
        wheel_joints = [
            self.model.joint(f"{name}_wheel_joint")
            for name in self._SWERVE_MODULES
        ]
        self._swerve_steer_joint_ids = np.array(
            [int(j.id) for j in steer_joints], dtype=int
        )
        self._swerve_wheel_joint_ids = np.array(
            [int(j.id) for j in wheel_joints], dtype=int
        )
        self._swerve_steer_qpos_adrs = np.array(
            [int(j.qposadr) for j in steer_joints], dtype=int
        )
        self._swerve_wheel_qpos_adrs = np.array(
            [int(j.qposadr) for j in wheel_joints], dtype=int
        )

        mujoco.mj_forward(self.model, self.data)
        base_xy = self.data.xpos[self.model.body("base_link").id, :2]
        self._swerve_module_xy = (
            self.data.xanchor[self._swerve_steer_joint_ids, :2] - base_xy
        ).copy()

        # Positive wheel angular velocity rolls the chassis along axle × up.
        wheel_axes = self.data.xaxis[self._swerve_wheel_joint_ids]
        rolling_dirs = np.cross(wheel_axes, np.array([0.0, 0.0, 1.0]))[:, :2]
        rolling_dirs /= np.linalg.norm(rolling_dirs, axis=1, keepdims=True)
        self._swerve_zero_headings = np.arctan2(
            rolling_dirs[:, 1], rolling_dirs[:, 0]
        )
        self._swerve_steer_angles = self.data.qpos[
            self._swerve_steer_qpos_adrs
        ].copy()
        self._swerve_wheel_angles = self.data.qpos[
            self._swerve_wheel_qpos_adrs
        ].copy()

    def _animate_swerve(self, world_velocity: np.ndarray) -> None:
        """Animate steer and wheel joints for a world-frame base velocity."""
        vx_world, vy_world, omega = np.asarray(world_velocity, dtype=float)
        theta = float(self.data.qpos[self.ik.base_qpos_adrs[2]])
        c, s = np.cos(theta), np.sin(theta)
        vx_body = c * vx_world + s * vy_world
        vy_body = -s * vx_world + c * vy_world

        x = self._swerve_module_xy[:, 0]
        y = self._swerve_module_xy[:, 1]
        module_vx = vx_body - omega * y
        module_vy = vy_body + omega * x
        wheel_speeds = np.hypot(module_vx, module_vy)

        moving = wheel_speeds >= self._SWERVE_SPEED_DEADBAND_M_S
        wheel_speeds[~moving] = 0.0
        desired = self._swerve_steer_angles.copy()
        desired[moving] = self._wrap_pi(
            np.arctan2(module_vy[moving], module_vx[moving])
            - self._swerve_zero_headings[moving]
        )

        # A swerve module can reach the same rolling direction by turning 180°
        # and reversing the wheel.  Choose the shorter steering motion, just as
        # the hardware driver does.
        steer_error = self._wrap_pi(desired - self._swerve_steer_angles)
        reverse = moving & (np.abs(steer_error) > np.pi / 2.0)
        desired[reverse] = self._wrap_pi(desired[reverse] + np.pi)
        wheel_speeds[reverse] *= -1.0

        steer_error = self._wrap_pi(desired - self._swerve_steer_angles)
        max_step = self._MAX_STEER_RATE_RAD_S * self.solver_dt
        self._swerve_steer_angles = self._wrap_pi(
            self._swerve_steer_angles
            + np.clip(steer_error, -max_step, max_step)
        )
        self._swerve_wheel_angles = self._wrap_pi(
            self._swerve_wheel_angles
            + wheel_speeds / self._TIRE_RADIUS_M * self.solver_dt
        )

        self.data.qpos[self._swerve_steer_qpos_adrs] = self._swerve_steer_angles
        self.data.qpos[self._swerve_wheel_qpos_adrs] = self._swerve_wheel_angles


if __name__ == "__main__":
    from robot.utils.console_log import start_console_log
    start_console_log("yor_mujoco", _REPO / "artifacts" / "wholebody_logs")

    yor_mujoco = YORMujoco()
    yor_mujoco.start_control()

    rpc_server = RPCServer(yor_mujoco, 8081, threaded=False)
    atexit.register(rpc_server.stop)

    print("YORMujoco RPC Server started on port 8081.")
    print("Running WholeBodyIK with scene:", yor_mujoco.mjcf_path)
    try:
        rpc_server.start()
    except KeyboardInterrupt:
        pass
    yor_mujoco.stop_control()
