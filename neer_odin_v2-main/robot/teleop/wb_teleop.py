import csv
from datetime import datetime
import zmq
import numpy as np
import atexit
import time
import threading
import os

from loop_rate_limiters import RateLimiter
import mink

from robot.teleop.oculus_msgs import parse_controller_state

from commlink import RPCClient
from robot.yor import YOR
import argparse


def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


# VR Constants
VR_TCP_HOST = "10.0.0.111" # on netgear local router
# VR_TCP_HOST = "10.21.64.218"
# VR_TCP_HOST = "10.21.83.133"
# VR_TCP_HOST = "10.19.189.139"
VR_TCP_PORT = 5555
VR_CONTROLLER_TOPIC = b"oculus_controller"
GRIPPER_ANGLE_MAX = -22.0

BUTTON_DEBOUNCE_TIME = 0.2  # seconds
class OculusBimanualBaseReader:
    def __init__(self, reset_base_after_data_collection: bool = False):
        self.reset_base_after_data_collection = reset_base_after_data_collection
        # teleop state
        self.ee_pose = None
        self.start_teleop_left = False
        self.start_teleop_right = False
        self.start_base_lift_control = False
        self.data_list = []
        self.record_data = False
        self.H = mink.SE3.from_rotation(mink.SO3.from_matrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])))
        self.X_Cinit_left = None
        self.X_ee_init_left = None
        self.X_Cinit_right = None
        self.X_ee_init_right = None

        # base control state
        self.max_vels = [np.array([0.5, 0.5, 1.57]), np.array([0.25, 0.25, 1.57]), np.array([0.75, 0.75, 1.57]), np.array([0.15, 0.15, 1.57/2.5])] # the first one is teleop, the last one is for recording
        self.max_vel_setting = 0
        self.vel_alpha = 0.9
        self.last_target_velocity = np.array([0.0, 0.0, 0.0])

        self.yor: YOR = RPCClient(host="localhost", port=8081)
        self.yor.init()
        self.has_arms = True
        try:
            self.yor.home_left_arm()
            self.yor.home_right_arm()
        except (ConnectionError, AttributeError, RuntimeError, Exception) as e:
            print(f"Warning: Could not home arms: {e}, I have no arms :(")
            self.has_arms = False
        self.default_kp = np.array([2.5, 2.5, 2.5, 2.5, 3.0, 3.0])
        self.default_kd = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        self.stop_event = threading.Event()

        self.latest_controller_state = None
        self.controller_state_lock = threading.Lock()
        self.thread = threading.Thread(target=self.oculus_thread, daemon=True)
        self.thread.start()

    def oculus_thread(self):
        zmq_context = zmq.Context()
        stick_socket = zmq_context.socket(zmq.SUB)
        stick_socket.connect("tcp://{}:{}".format(VR_TCP_HOST, VR_TCP_PORT))
        stick_socket.subscribe(VR_CONTROLLER_TOPIC)

        while not self.stop_event.is_set():
            _, message = stick_socket.recv_multipart()
            controller_state = parse_controller_state(message.decode())
            # print("Received controller state", end='\r')
            with self.controller_state_lock:
                self.latest_controller_state = controller_state

        stick_socket.close()
        zmq_context.destroy()

    def control_loop(self):
        rate = RateLimiter(30, name="bimanual_wholebody")
        reset_rate = RateLimiter(30, name="bimanual_wholebody_reset")

        while not self.stop_event.is_set():
            with self.controller_state_lock:
                controller_state = self.latest_controller_state
            if controller_state is None:
                continue

            # Arm teleop mode control
            if controller_state.left_x and self.has_arms:
                if not self.start_teleop_left:
                    ee_pose_left = self.yor.get_left_ee_pose()
                    self.X_Cinit_left = controller_state.left_SE3
                    self.X_ee_init_left = ee_pose_left
                    x_Re_L = ee_pose_left
                    self.start_teleop_left = True
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    print("start teleop left", end='\n')
                else:
                    self.start_teleop_left = False
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    print("stop teleop left", end='\n')

            if controller_state.left_y and self.has_arms:
                print("Homing left arm", end='\n')
                self.yor.home_left_arm()
                self.start_teleop_left = False

            if controller_state.right_a and self.has_arms:
                if not self.start_teleop_right:
                    ee_pose_right = self.yor.get_right_ee_pose()
                    self.X_Cinit_right = controller_state.right_SE3
                    self.X_ee_init_right = ee_pose_right
                    x_Re_R = ee_pose_right
                    self.start_teleop_right = True
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    print("start teleop right", end='\n')
                else:
                    self.start_teleop_right = False
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    print("stop teleop right", end='\n')

            if controller_state.right_b and self.has_arms:
                print("Homing right arm", end='\n')
                self.yor.home_right_arm()
                self.start_teleop_right = False

            # Left arm teleop
            if self.start_teleop_left:
                if self.X_Cinit_left is None or self.X_ee_init_left is None:
                    print("WARN: no initial pose yet")
                    time.sleep(0.01)
                    continue
                X_Ctarget = controller_state.left_SE3
                X_Cdelta = self.X_Cinit_left.inverse().multiply(X_Ctarget)
                X_Rdelta = self.H.inverse() @ X_Cdelta @ self.H

                x_Re_L = self.X_ee_init_left @ X_Rdelta

                gripper_L = 1 if controller_state.left_index_trigger < 0.5 else 0
                preview_time_L = 1/15

            # Right arm teleop
            if self.start_teleop_right:
                if self.X_Cinit_right is None or self.X_ee_init_right is None:
                    print("WARN: no initial pose yet")
                    time.sleep(0.01)
                    continue
                X_Ctarget = controller_state.right_SE3
                X_Cdelta = self.X_Cinit_right.inverse().multiply(X_Ctarget)
                X_Rdelta = self.H.inverse() @ X_Cdelta @ self.H

                x_Re_R = self.X_ee_init_right @ X_Rdelta

                gripper_R = 1 if controller_state.right_index_trigger < 0.5 else 0
                preview_time_R = 1/15

            # Send arm commands
            if self.start_teleop_left and not self.start_teleop_right:
                self.yor.set_left_ee_target(
                    ee_target=x_Re_L,
                    gripper_target=gripper_L,
                    preview_time=preview_time_L,
                )
            if self.start_teleop_right and not self.start_teleop_left:
                self.yor.set_right_ee_target(
                    ee_target=x_Re_R,
                    gripper_target=gripper_R,
                    preview_time=preview_time_R,
                )
            if self.start_teleop_left and self.start_teleop_right:
                self.yor.set_bimanual_ee_target(
                    L_ee_target=x_Re_L,
                    L_gripper_target=gripper_L,
                    L_preview_time=preview_time_L,
                    R_ee_target=x_Re_R,
                    R_gripper_target=gripper_R,
                    R_preview_time=preview_time_R,
                )

            # Base control: Velocity from right thumbstick and angular from left thumbstick
            vy = -controller_state.right_thumbstick_axes[0]  # Right stick horizontal
            vx = controller_state.right_thumbstick_axes[1]  # Right stick vertical
            w = -controller_state.left_thumbstick_axes[0]    # Left stick horizontal

            # print((vx, vy, w))  # debug
            target_velocity = np.array([vx, vy, w])
            target_velocity = apply_deadzone(target_velocity)

            target_velocity = self.max_vels[self.max_vel_setting] * target_velocity
            target_velocity = (1 - self.vel_alpha) * target_velocity + self.vel_alpha * self.last_target_velocity
            self.last_target_velocity = target_velocity

            if controller_state.left_hand_trigger > 0.5 and controller_state.right_hand_trigger > 0.5:
                self.start_base_lift_control = not self.start_base_lift_control
                print(f"Base and lift control toggled to {self.start_base_lift_control}", end='\n')
                time.sleep(BUTTON_DEBOUNCE_TIME)

            # Send base velocity command
            lift_target: int = 0
            if self.start_base_lift_control:
                if sum(np.abs(target_velocity)) > 1e-2:
                    self.yor.set_base_velocity(target_velocity)
            # Lift control: left_hand_trigger for lift up, right_hand_trigger for lift down
                if controller_state.left_hand_trigger > 0.5 and self.start_base_lift_control:
                    self.yor.lift_up()
                    lift_target = 1
                elif controller_state.right_hand_trigger > 0.5 and self.start_base_lift_control:
                    self.yor.lift_down()
                    lift_target = -1
                else:
                    self.yor.lift_stop()
                    lift_target = 0
            else:
                target_velocity = np.array([0.0,0.0,0.0])

            if self.record_data:
                # start_time = time.time()
                curr_pose = self.yor.get_bimanual_state()
                # print(f"this cost freqeuency: {1.0/(time.time()-curr_pose[0]):.1f} Hz", end='\n')
                if self.start_teleop_left:
                    L_ee_target_list = x_Re_L.wxyz_xyz.tolist()
                else:
                    L_ee_target_list = curr_pose[1:8]
                if self.start_teleop_right:
                    R_ee_target_list = x_Re_R.wxyz_xyz.tolist()
                else:
                    R_ee_target_list = curr_pose[15:22]

                record_row = curr_pose[:-1] + L_ee_target_list + R_ee_target_list
                lift_position = curr_pose[29]/0.41
                record_row += target_velocity.tolist()
                record_row += [lift_position, lift_target]
                self.data_list.append(record_row)

            if controller_state.left_menu:
                # recording toggled
                if not self.record_data:
                    # initialize recording
                    self.record_data = True
                    if self.has_arms:
                        self.yor.set_left_gain(5*self.default_kp, 5*self.default_kd)
                        self.yor.set_right_gain(5*self.default_kp, 5*self.default_kd)
                    self.max_vel_setting = 3
                    timestamp_str = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                    filename = f'teleop_data/bimanual_wb_log_{timestamp_str}.csv'
                    header = ['timestamp']
                    header += [f'left_ee_{i}' for i in ['w','qx','qy','qz','x','y','z']]
                    header += [f'left_joint_{i}' for i in range(6)]  # Assuming 6 joints per arm
                    header += ['left_gripper_pose']
                    header += [f'right_ee_{i}' for i in ['w','qx','qy','qz','x','y','z']]
                    header += [f'right_joint_{i}' for i in range(6)]  # Assuming 6 joints per arm
                    header += ['right_gripper_pose']
                    header += [f'left_ee_{i}_target' for i in ['w','qx','qy','qz','x','y','z']]
                    header += [f'right_ee_{i}_target' for i in ['w','qx','qy','qz','x','y','z']]
                    header += ['base_vx_target', 'base_vy_target', 'base_w_target', 'lift_position', 'lift_target']
                    self.data_list = []
                    self.data_list.append(header)
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    print("Start recording data", end='\n')
                else:
                    # stop recording and save data
                    self.record_data = False
                    self.start_teleop_left = False
                    self.start_teleop_right = False
                    print("Stop recording data", end='\n')
                    if self.has_arms:
                        self.yor.set_left_gain(self.default_kp, self.default_kd)
                        self.yor.set_right_gain(self.default_kp, self.default_kd)
                    self.max_vel_setting = 0
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(self.data_list)
                        f.flush()
                    print(f"Saved recorded data to {filename}", end='\n')
                    time.sleep(BUTTON_DEBOUNCE_TIME)
                    # we are going to reset the base
                    if self.reset_base_after_data_collection:
                        reset_start_time = time.time()
                        for i in range(len(self.data_list)-1, 0, -1):
                            *_, vx, vy, w, _, lift_target = self.data_list[i]
                            if lift_target == 1: # the lift was raised so now we lower
                                self.yor.set_lift_position(np.array([0.0]))
                            elif lift_target == -1:  # the lift was lowered so now we raise
                                self.yor.set_lift_position(np.array([0.41]))
                            if sum(np.abs([vx, vy, w])) > 1e-2:
                                self.yor.set_base_velocity(np.array([-vx, -vy, -w]))
                            if lift_target != 0 or sum(np.abs([vx, vy, w])) > 1e-2:
                                reset_rate.sleep()
                            else:
                                reset_rate.sleep()
                        print(f"Reset base and lift in {time.time()-reset_start_time:.2f} seconds", end='\n')

                    if self.has_arms:
                        print("Homing both arms after recording", end='\n')
                        self.yor.home_left_arm()
                        self.yor.home_right_arm()

                    self.data_list = []

            rate.sleep()

    def stop(self):
        self.stop_event.set()
        self.thread.join()


def main(reset_base):
    oculus_reader = OculusBimanualBaseReader(reset_base_after_data_collection=reset_base)
    atexit.register(oculus_reader.stop)
    oculus_reader.control_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset-base', '-r', action='store_true', help='Reset base after data collection')
    args = parser.parse_args()
    main(args.reset_base)