# oculus_all_in_one.py
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


def apply_deadzone(arr, dz=0.05):
    arr = np.asarray(arr, dtype=float)
    return np.where(np.abs(arr) <= dz, 0.0,
                    np.sign(arr) * (np.abs(arr) - dz) / (1 - dz))


class OculusAllInOne:
    """
    One Oculus controller drives:
      - Base velocity (vx, vy, w) from thumbsticks
      - Arm EE pose via 6-DoF teleop (A=start/zero, B=stop)
      - Gripper via right index trigger
      - Lift via left_x/left_y buttons (optional)
    """

    # ---------------- CONFIG (edit if you want different bindings) ----------------
    # Buttons (booleans)
    BTN_START_TELEOP = "right_a"
    BTN_STOP_TELEOP  = "right_b"
    BTN_SPEED_CYCLE  = "left_thumbstick"   # click to cycle speed presets

    # Triggers
    TRIG_GRIPPER     = "right_index_trigger"  # float [0,1]; >0.5 closes

    # Sticks / axes arrays
    # right_thumbstick_axes -> [x, y]; we use y for forward/back (vx) and x for strafe (vy)
    # left_thumbstick_axes  -> [x, y]; we use x for yaw (w)
    AX_RIGHT_STICK   = "right_thumbstick_axes"
    AX_LEFT_STICK    = "left_thumbstick_axes"

    # Lift buttons (optional; mapped to left buttons since no D-pad field)
    BTN_LIFT_UP      = "left_y"
    BTN_LIFT_DOWN    = "left_x"

    # Pose field
    POSE_RIGHT       = "right_SE3"
    # -----------------------------------------------------------------------------

    VR_TCP_HOST = "192.168.1.111"
    VR_TCP_PORT = 5555
    VR_CONTROLLER_TOPIC = b"oculus_controller"

    def __init__(self):
        # Arm teleop state
        self.start_teleop = False
        self.H = mink.SE3.from_rotation(
            mink.SO3.from_matrix(np.array([[0, -1, 0],
                                           [0,  0, 1],
                                           [-1, 0, 0]]))
        )
        self.X_Cinit = None
        self.X_ee_init = None

        # RPC (same server for base/arm/lift)
        self.yor: YOR = RPCClient(host="localhost", port=8081)
        self.yor.init()
        self.yor.home_right_arm()

        # Base control params
        self.max_vels = [
            np.array([0.50, 0.50, 1.57]),  # medium
            np.array([0.25, 0.25, 1.57]),  # slow
            np.array([0.75, 0.75, 1.57]),  # fast
        ]
        self.max_vel_setting = 0
        self.vel_alpha = 0.9
        self.last_target_velocity = np.zeros(3, dtype=float)
        self._speed_cycle_debounce = 0.0

        # ZMQ receiver thread
        self.interval_history = []
        self.stop_event = threading.Event()
        self.latest_controller_state = None
        self.controller_state_lock = threading.Lock()
        self.thread = threading.Thread(target=self._oculus_thread, daemon=True)
        self.thread.start()

    # ---------- ZMQ receiver thread ----------
    def _oculus_thread(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://{self.VR_TCP_HOST}:{self.VR_TCP_PORT}")
        sock.subscribe(self.VR_CONTROLLER_TOPIC)
        last_ts = None

        while not self.stop_event.is_set():
            try:
                _, message = sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue

            st = parse_controller_state(message.decode())
            with self.controller_state_lock:
                self.latest_controller_state = st

            if last_ts is not None:
                self.interval_history.append(time.time() - last_ts)
            last_ts = time.time()

        sock.close()
        ctx.term()

    # ---------- Helpers ----------
    def _maybe_cycle_speed(self, st, now):
        pressed = bool(getattr(st, self.BTN_SPEED_CYCLE))
        if pressed and (now - self._speed_cycle_debounce > 0.25):
            self.max_vel_setting = (self.max_vel_setting + 1) % len(self.max_vels)
            print("Max velocity setting:", self.max_vel_setting)
            self._speed_cycle_debounce = now

    # ---------- Main control loop ----------
    def control_loop(self):
        rate = RateLimiter(60, name="oculus_all_in_one")  # 60 Hz loop
        arm_update_period = 1.0 / 30.0
        last_arm_update = 0.0

        while not self.stop_event.is_set():
            with self.controller_state_lock:
                st = self.latest_controller_state

            if st is None:
                rate.sleep()
                continue

            now = time.time()

            # ------ Base control (sticks -> vx, vy, w) ------
            # Right stick: [x, y] => vy, vx (note many controllers report up as negative y)
            rs = getattr(st, self.AX_RIGHT_STICK)  # np.array([x, y])
            ls = getattr(st, self.AX_LEFT_STICK)   # np.array([x, y])

            rsx = float(rs[0]) if rs is not None and len(rs) > 0 else 0.0
            rsy = float(rs[1]) if rs is not None and len(rs) > 1 else 0.0
            lsx = float(ls[0]) if ls is not None and len(ls) > 0 else 0.0

            # Conventions (tweak signs if motion feels inverted):
            vx = -rsy    # push stick up (y negative) -> positive forward
            vy =  rsx    # right is positive +y strafe
            w  =  lsx    # left stick X -> yaw CCW positive

            target_velocity = np.array([vx, vy, w], dtype=float)
            target_velocity = apply_deadzone(target_velocity)
            target_velocity = self.max_vels[self.max_vel_setting] * target_velocity
            target_velocity = (1 - self.vel_alpha) * target_velocity + self.vel_alpha * self.last_target_velocity
            self.last_target_velocity = target_velocity

            if np.linalg.norm(target_velocity, ord=1) > 1e-2:
                self.yor.set_base_velocity(target_velocity)

            # Speed preset cycling
            self._maybe_cycle_speed(st, now)

            # ------ Lift via buttons (optional) ------
            lift_up   = bool(getattr(st, self.BTN_LIFT_UP))
            lift_down = bool(getattr(st, self.BTN_LIFT_DOWN))
            if lift_up and not lift_down:
                self.yor.lift_up()
            elif lift_down and not lift_up:
                self.yor.lift_down()
            else:
                self.yor.lift_stop()

            # ------ Arm teleop (pose) ------
            start_btn = bool(getattr(st, self.BTN_START_TELEOP))
            stop_btn  = bool(getattr(st, self.BTN_STOP_TELEOP))
            ee_pose = self.yor.get_right_ee_pose()

            if start_btn:
                X_C = getattr(st, self.POSE_RIGHT)
                print("Arm teleop: START (zeroed)")
                self.X_Cinit = X_C
                self.X_ee_init = ee_pose
                self.start_teleop = True

            if stop_btn:
                if self.start_teleop:
                    print("Arm teleop: STOP")
                self.start_teleop = False

            if self.start_teleop and (now - last_arm_update) >= arm_update_period:
                last_arm_update = now
                if self.X_Cinit is not None and self.X_ee_init is not None:
                    X_Ctarget = getattr(st, self.POSE_RIGHT)
                    X_Cdelta = self.X_Cinit.inverse().multiply(X_Ctarget)
                    X_Rdelta = self.H.inverse() @ X_Cdelta @ self.H

                    p_REt = self.X_ee_init.translation() + X_Rdelta.translation()
                    R_REt = self.X_ee_init.rotation() @ X_Rdelta.rotation()

                    trig = float(getattr(st, self.TRIG_GRIPPER))
                    gripper = 0.0 if trig < 0.5 else 1.0

                    ee_distance = np.linalg.norm(p_REt - ee_pose.translation())
                    preview_time = ee_distance / 0.5  # change to 0.05 for slower

                    self.yor.set_right_ee_target(
                        ee_target=mink.SE3(np.concatenate([R_REt.wxyz, p_REt])),
                        gripper_target=gripper,
                        preview_time=preview_time,
                    )

            rate.sleep()

    def stop(self):
        np.array(self.interval_history, dtype=float).tofile("interval_history.bin")
        self.stop_event.set()
        self.thread.join()


def main():
    node = OculusAllInOne()
    atexit.register(node.stop)
    node.control_loop()


if __name__ == "__main__":
    main()
