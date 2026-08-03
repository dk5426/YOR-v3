import threading
import atexit
import sys
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

    def __init__(self, mjcf_path: Optional[str] = None, solver_dt: float = 0.01):
        self.mjcf_path = str(mjcf_path or DEFAULT_SCENE)
        self.solver_dt = solver_dt

        # ── Initialize IK Solver ──────────────────────────────────────────────
        # Same tuned weights the hardware controller uses: base reluctant to
        # roll, lift eager to stretch.
        cfg = WholeBodyIKConfig(
            dt=self.solver_dt,
            solver="pyqpmad",
            max_iters=10,
            base_posture_cost=5e-2,
            lift_posture_cost=1e-4,
            arm_posture_cost=1e-3,
        )
        self.ik = WholeBodyIK(scene_xml=self.mjcf_path, config=cfg)
        self.ik.init_from_keyframe("home")

        self.model = self.ik.model
        self.data = self.ik.data

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
        self.lift_target: Optional[float] = None
        self.base_fixed: bool = False
        self._last_base_velocity = np.zeros(3)

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
            "left_joint_positions": q[self.ik._left_arm_qpos_adrs].tolist(),
            "right_joint_positions": q[self.ik._right_arm_qpos_adrs].tolist(),
        }

    def home_left_arm(self):
        """Reset the left EE target to its home pose."""
        with self.target_lock:
            self.left_ee_target = self._home_left.copy()

    def home_right_arm(self):
        """Reset the right EE target to its home pose."""
        with self.target_lock:
            self.right_ee_target = self._home_right.copy()

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

            # Sync IK with current data
            self.ik.update_configuration(self.data.qpos)

            # Solve Whole-Body IK
            result = self.ik.solve(T_l, T_r, lift_target=lift_tgt)

            # Apply in Kinematic Mode
            self.ik.apply_to_sim_kinematic(self.data, result)
            mujoco.mj_forward(self.model, self.data)

            with self.target_lock:
                self._last_base_velocity = result.base_velocity

            self.viewer.sync()
            rate_limiter.sleep()

        self.control_loop_running = False


if __name__ == "__main__":
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
