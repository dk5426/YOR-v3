import mujoco
import mujoco.viewer
import mink
import time
import numpy as np
from typing import Any, Optional
from pathlib import Path
import argparse

from dora import Node

from robot.arm.ik_solver import BimanualArmIK
from robot.msgs.bimanual_pose import BimanualPose, BimanualArmCommand


class BimanualArmMujoco:
    def __init__(self, mjcf_path: str, solver_dt: float = 0.03):
        self.mjcf_path = mjcf_path
        self.solver_dt = solver_dt

        # launch mujoco
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=True,
            show_right_ui=True,
        )
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # initialize arm
        self.left_target: Optional[mink.SE3] = None
        self.right_target: Optional[mink.SE3] = None
        self.left_gripper: Optional[float] = None
        self.right_gripper: Optional[float] = None
        self.ik_solver = BimanualArmIK(mjcf_path, solver_dt=self.solver_dt)

        # communication
        self.node = Node()
        self.init()

    def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
        current_time = time.perf_counter_ns()
        delay = (current_time - timestamp) / 1e9
        if delay > max_delay or delay < 0:
            print(f"Skipping message because of delay: {delay}s")
            return False
        return True

    def bimanual_arm_command_handler(self, event: dict[str, Any]):
        bimanual_arm_command = BimanualArmCommand.decode(event["value"], event["metadata"])
        if not self.check_timestamp(bimanual_arm_command.timestamp, 0.1):
            return

        left_pose = mink.SE3(bimanual_arm_command.left_wxyz_xyz)
        right_pose = mink.SE3(bimanual_arm_command.right_wxyz_xyz)
        self.left_target = left_pose
        self.right_target = right_pose
        self.left_gripper = bimanual_arm_command.left_gripper
        self.right_gripper = bimanual_arm_command.right_gripper

    def init(self):
        # home
        home_q = self.ik_solver.get_home_q(home_key="home")
        print(f"home_q: {home_q}")
        self.data.qpos[self.ik_solver.dof_ids] = home_q
        mujoco.mj_forward(self.model, self.data)
        time.sleep(1.0)
        q = self.data.qpos.copy()
        mink.move_mocap_to_frame(self.model, self.data, "left/pinch_site_target", "left/ee", "site")
        mink.move_mocap_to_frame(self.model, self.data, "right/pinch_site_target", "right/ee", "site")
        self.viewer.sync()
        self.ik_solver.init(q)
        self.left_target, self.right_target = self.ik_solver.forward_kinematics(q)
        print(f"left_target: {self.left_target}")
        print(f"right_target: {self.right_target}")

    def step(self):
        q = self.data.qpos.copy()
        left_ee_pose, right_ee_pose = self.ik_solver.forward_kinematics(q)
        if self.left_target is not None and self.right_target is not None:  # TODO: check timestamp for the target
            # update mocap viz
            self.data.mocap_pos[self.model.body("left/pinch_site_target").mocapid[0]] = self.left_target.translation()
            self.data.mocap_quat[self.model.body("left/pinch_site_target").mocapid[0]] = (
                self.left_target.rotation().wxyz
            )
            self.data.mocap_pos[self.model.body("right/pinch_site_target").mocapid[0]] = self.right_target.translation()
            self.data.mocap_quat[self.model.body("right/pinch_site_target").mocapid[0]] = (
                self.right_target.rotation().wxyz
            )
            # solve ik
            qd_left, qd_right = self.ik_solver.solve_ik(self.left_target, self.right_target)
            self.data.qpos[self.ik_solver.dof_ids] = np.concatenate([qd_left, qd_right])
            if self.left_gripper is not None:
                print(f"left_gripper: {self.left_gripper}")
            if self.right_gripper is not None:
                print(f"right_gripper: {self.right_gripper}")
        # step mujoco
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        # send ee pose
        pose_msg = BimanualPose(time.perf_counter_ns(), left_ee_pose.wxyz_xyz, right_ee_pose.wxyz_xyz)
        self.node.send_output("bimanual_ee_pose", *pose_msg.encode())

    def stop(self):
        self.viewer.close()

    def spin(self):
        for event in self.node:
            event_type = event["type"]
            if event_type == "INPUT":
                event_id = event["id"]

                if event_id == "bimanual_arm_command":
                    self.bimanual_arm_command_handler(event)

                elif event_id == "tick":
                    self.step()

            elif event_type == "STOP":
                self.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf_path", type=str, default="mujoco/scene_bimanual.xml")
    args = parser.parse_args()

    _HERE = Path(__file__).parent
    bimanual_arm_mujoco = BimanualArmMujoco(mjcf_path=(_HERE / args.mjcf_path).as_posix())
    bimanual_arm_mujoco.spin()


if __name__ == "__main__":
    main()
