import zmq
import numpy as np
import time
import threading
from typing import Any

import mink

from dora import Node

from robot.teleop.oculus_msgs import parse_controller_state
# from robot.network import VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC
from robot.msgs.bimanual_pose import BimanualPose, BimanualArmCommand
from robot.msgs.base_command import BaseCommand, CommandType

# VR Constants
VR_TCP_HOST = "10.19.165.216"
# VR_TCP_HOST = "10.19.189.139"
VR_TCP_PORT = 5555
VR_CONTROLLER_TOPIC = b"oculus_controller"
GRIPPER_ANGLE_MAX = 0.7

def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))

class OculusBimanualNode:
    def __init__(self):
        # teleop state
        self.bimanual_pose = None
        self.left_arm_teleop_enabled = False
        self.right_arm_teleop_enabled = False
        self.H = mink.SE3.from_rotation(mink.SO3.from_matrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])))
        self.X_O_Cleft_init = None # Controller left in Oculus frame
        self.X_O_Cright_init = None # Controller right in Oculus frame
        self.X_R_EEleft_init = None # left EE in Robot frame
        self.X_R_EEright_init = None # right EE in Robot frame

        # communication
        self.node = Node()

        # Oculus thread
        self.controller_state_lock_ = threading.Lock()
        self.controller_state = None
        self.stop_event = threading.Event()

        self.oculus_thread = threading.Thread(target=self.oculus_handler)
        self.oculus_thread.start()

    def oculus_handler(self):
        context = zmq.Context()
        poller = zmq.Poller()

        # Connect to the VR controller
        stick_socket = context.socket(zmq.SUB)
        stick_socket.connect("tcp://{}:{}".format(VR_TCP_HOST, VR_TCP_PORT))
        stick_socket.subscribe(VR_CONTROLLER_TOPIC)

        # initialize the polling set
        poller.register(stick_socket, zmq.POLLIN)

        while not self.stop_event.is_set():
            events = dict(poller.poll(1000))
            if stick_socket in events:
                _, message = stick_socket.recv_multipart()
                with self.controller_state_lock_:
                    self.controller_state = parse_controller_state(message.decode())

        # close the context
        poller.unregister(stick_socket)
        stick_socket.close()
        context.destroy()

    def bimanual_ee_pose_handler(self, event: dict[str, Any]):
        bimanual_ee_pose = BimanualPose.decode(event["value"], event["metadata"])
        self.bimanual_pose = bimanual_ee_pose

    def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
        current_time = time.perf_counter_ns()
        delay = (current_time - timestamp) / 1e9
        if delay > max_delay or delay < 0:
            print(f"Skipping message because of delay: {delay}s")
            return False
        return True

    def step(self):
        with self.controller_state_lock_:
            controller_state = self.controller_state
        if controller_state is None:
            print("WARN: no controller state yet")
            return

        if self.bimanual_pose is None:
            print("WARN: no bimanual ee pose yet")
            return

        if not self.check_timestamp(self.bimanual_pose.timestamp, 0.05):
            print("WARN: bimanual pose timestamp is too old")
            return

        X_R_EEleft = mink.SE3(self.bimanual_pose.left_wxyz_xyz)
        X_R_EEright = mink.SE3(self.bimanual_pose.right_wxyz_xyz)

        # left controller
        if controller_state.left_x:
            print("start left teleop")
            self.X_O_Cleft_init = controller_state.left_SE3
            self.X_R_EEleft_init = X_R_EEleft
            self.left_arm_teleop_enabled = True
        if controller_state.right_a:
            print("start right teleop")
            self.X_O_Cright_init = controller_state.right_SE3
            self.X_R_EEright_init = X_R_EEright
            self.right_arm_teleop_enabled = True

        if controller_state.left_y:
            self.left_arm_teleop_enabled = False
        if controller_state.right_b:
            self.right_arm_teleop_enabled = False

        if self.left_arm_teleop_enabled:
            if self.X_O_Cleft_init is None or self.X_R_EEleft_init is None:
                print("WARN: no initial pose yet")
                return
            X_O_Ctarget = controller_state.left_SE3
            X_O_Cdelta = self.X_O_Cleft_init.inverse().multiply(X_O_Ctarget)
            X_R_EEdelta = self.H.inverse() @ X_O_Cdelta @ self.H
            # translation
            p_R_EEtarget = self.X_R_EEleft_init.translation() + X_R_EEdelta.translation()
            # rotation
            R_R_EEtarget = self.X_R_EEleft_init.rotation() @ X_R_EEdelta.rotation()

            # publish the target pose
            X_R_EEleft_desired = mink.SE3(np.concatenate([R_R_EEtarget.wxyz, p_R_EEtarget]))
        else:
            X_R_EEleft_desired = X_R_EEleft # keep the current pose

        if self.right_arm_teleop_enabled:
            if self.X_O_Cright_init is None or self.X_R_EEright_init is None:
                print("WARN: no initial pose yet")
                return
            X_O_Ctarget = controller_state.right_SE3
            X_O_Cdelta = self.X_O_Cright_init.inverse().multiply(X_O_Ctarget)
            X_R_EEdelta = self.H.inverse() @ X_O_Cdelta @ self.H
            # translation
            p_R_EEtarget = self.X_R_EEright_init.translation() + X_R_EEdelta.translation()
            # rotation
            R_R_EEtarget = self.X_R_EEright_init.rotation() @ X_R_EEdelta.rotation()

            # publish the target pose
            X_R_EEright_desired = mink.SE3(np.concatenate([R_R_EEtarget.wxyz, p_R_EEtarget]))
        else:
            X_R_EEright_desired = X_R_EEright # keep the current pose

        left_gripper = GRIPPER_ANGLE_MAX if controller_state.left_index_trigger < 0.5 else 0.0
        right_gripper = GRIPPER_ANGLE_MAX if controller_state.right_index_trigger < 0.5 else 0.0

        # BASE CONTROL
        vy = -controller_state.right_thumbstick_axes[0]
        vx = controller_state.right_thumbstick_axes[1]
        w = -controller_state.left_thumbstick_axes[0]
        max_vel = np.array([0.5, 0.5, 1.57])
        target_velocity = np.array([vx, vy, w])
        target_velocity = apply_deadzone(target_velocity)
        target_velocity = max_vel * target_velocity
        if sum(np.abs(target_velocity)) > 0.0:
            base_command = BaseCommand(
                timestamp=time.perf_counter_ns(),
                type=CommandType.BASE_VELOCITY,
                target=target_velocity.ravel(),
            )
            self.node.send_output("base_command", *base_command.encode())

        # LIFT CONTROL
        if controller_state.left_hand_trigger > 0.5:
            lift_target = 0.0
        elif controller_state.right_hand_trigger > 0.5:
            lift_target = 0.39
        else:
            lift_target = -1 # don't send command if no trigger is pressed
        if lift_target >= 0:
            lift_command = BaseCommand(
                timestamp=time.perf_counter_ns(),
                type=CommandType.LIFT_POSITION,
                target=np.array([lift_target]),
            )
            self.node.send_output("base_command", *lift_command.encode())

        # publish the target poses
        if self.left_arm_teleop_enabled or self.right_arm_teleop_enabled:
            bimanual_arm_command = BimanualArmCommand(
                timestamp=time.perf_counter_ns(),
                left_wxyz_xyz=X_R_EEleft_desired.wxyz_xyz,
                right_wxyz_xyz=X_R_EEright_desired.wxyz_xyz,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
            )
            self.node.send_output("bimanual_arm_command", *bimanual_arm_command.encode())

    def spin(self):
        for event in self.node:
            event_type = event["type"]
            if event_type == "INPUT":
                event_id = event["id"]

                if event_id == "bimanual_ee_pose":
                    self.bimanual_ee_pose_handler(event)

                elif event_id == "tick":
                    self.step()

            elif event_type == "STOP":
                self.stop_event.set()
                self.oculus_thread.join()

def main():
    oculus_bimanual_node = OculusBimanualNode()
    oculus_bimanual_node.spin()


if __name__ == "__main__":
    main()
