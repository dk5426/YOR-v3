import zmq
import numpy as np
import atexit
import time
import threading

from loop_rate_limiters import RateLimiter
import mink

from robot.teleop.oculus_msgs import parse_controller_state

from commlink import RPCClient
from robot.yor import YOR


def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


# VR Constants
VR_TCP_HOST = "192.168.1.111" # on netgear local router
# VR_TCP_HOST = "10.19.165.216"
# VR_TCP_HOST = "10.19.189.139"
VR_TCP_PORT = 5555
VR_CONTROLLER_TOPIC = b"oculus_controller"
GRIPPER_ANGLE_MAX = -22.0


class OculusReader:
    def __init__(self):
        # teleop state
        self.ee_pose = None
        self.start_teleop = False
        self.H = mink.SE3.from_rotation(mink.SO3.from_matrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])))
        self.X_Cinit = None
        self.X_ee_init = None

        self.yor: YOR = RPCClient(host="localhost", port=8081)
        self.yor.init()
        self.yor.home_right_arm()

        self.interval_history = []
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
        last_command_timestamp = None

        while not self.stop_event.is_set():
            _, message = stick_socket.recv_multipart()
            controller_state = parse_controller_state(message.decode())
            print(f"Received controller state: {controller_state}")
            with self.controller_state_lock:
                self.latest_controller_state = controller_state
            if last_command_timestamp is not None:
                interval = time.time() - last_command_timestamp
                self.interval_history.append(interval)
            else:
                interval = 0.01
            last_command_timestamp = time.time()

        stick_socket.close()
        zmq_context.destroy()

    def control_loop(self):
        # context = zmq.Context()
        # stick_socket = context.socket(zmq.SUB)
        # stick_socket.connect("tcp://{}:{}".format(VR_TCP_HOST, VR_TCP_PORT))
        # stick_socket.subscribe(VR_CONTROLLER_TOPIC)

        # last_command_timestamp = None
        rate_limiter = RateLimiter(30)  # Limit to 100 Hz

        while not self.stop_event.is_set():
            # _, message = stick_socket.recv_multipart()
            # controller_state = parse_controller_state(message.decode())
            # if last_command_timestamp is not None:
            #     interval = time.time() - last_command_timestamp
            #     self.interval_history.append(interval)
            # else:
            #     interval = 0.01
            # last_command_timestamp = time.time()

            with self.controller_state_lock:
                controller_state = self.latest_controller_state

            if controller_state is None:
                rate_limiter.sleep()
                continue

            ee_pose = self.yor.get_right_ee_pose()

            if controller_state.right_a:
                print("start teleop")
                self.X_Cinit = controller_state.right_SE3
                self.X_ee_init = ee_pose
                self.start_teleop = True

            if controller_state.right_b:
                self.start_teleop = False

            if self.start_teleop:
                if self.X_Cinit is None or self.X_ee_init is None:
                    print("WARN: no initial pose yet")
                    time.sleep(0.01)
                    continue
                X_Ctarget = controller_state.right_SE3
                X_Cdelta = self.X_Cinit.inverse().multiply(X_Ctarget)
                X_Rdelta = self.H.inverse() @ X_Cdelta @ self.H
                # translation
                p_REt = self.X_ee_init.translation() + X_Rdelta.translation()
                # rotation
                R_REt = self.X_ee_init.rotation() @ X_Rdelta.rotation()

                # publish the target pose
                gripper = 0.0 if controller_state.right_index_trigger < 0.5 else 1.0
                ee_distance = np.linalg.norm(p_REt - ee_pose.translation())
                preview_time = ee_distance / 0.5  # 0.05 m/s speed
                print(f"Setting target pose with preview time: {preview_time:.4f}")
                self.yor.set_right_ee_target(
                    ee_target=mink.SE3(np.concatenate([R_REt.wxyz, p_REt])),
                    gripper_target=gripper,
                    preview_time=preview_time,
                )
            rate_limiter.sleep()

    def stop(self):
        np.array(self.interval_history).tofile("interval_history.bin")
        self.stop_event.set()
        self.thread.join()


def main():
    oculus_reader = OculusReader()
    atexit.register(oculus_reader.stop)
    oculus_reader.control_loop()


if __name__ == "__main__":
    main()
