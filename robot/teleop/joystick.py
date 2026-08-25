# joystick.py
"""
To run joystick on Jetson Nano, install xpad drivers, add user to the input group,
and set appropriate udev rules.
"""

import time
import numpy as np
import pygame
from pygame.joystick import Joystick
from loop_rate_limiters import RateLimiter

from commlink import RPCClient

XBOX_CONTROLLER_MAP = {
    "start": 7,
    "back": 6,
    "l1": 4,
    "r1": 5,
    "left_horizontal_axis": 0,
    "left_vertical_axis": 1,
    "right_horizontal_axis": 3,
    "right_vertical_axis": 4,
}

PS4_CONTROLLER_MAP = {
    "start": 3,
    "back": 0,  # square
    "l1": 4,
    "r1": 5,
    "left_horizontal_axis": 0,
    "right_horizontal_axis": 3,
    "right_vertical_axis": 4,
    "pad_y": 7,
}

controller_map = XBOX_CONTROLLER_MAP


# Ticks of explicit zero to send after the stick is released, at the 60 Hz
# joystick rate. One would do — the relay latches it — but a short burst
# survives a dropped RPC without leaving the base holding its last command.
STOP_TAIL_TICKS = 15

# Raised from 0.05. This is a stop threshold, not just a jitter filter. The
# existing creep guard below keys off the command falling under 1e-2, which a
# stick that returns cleanly to centre does and a stick whose rest position
# sits outside the deadzone never does — the latter streams ~10 mm/s forever
# and the guard never fires. 0.05 is inside the resting scatter of a worn
# stick; 0.12 is not.
STICK_DEADZONE = 0.12


def apply_deadzone(arr, dz=STICK_DEADZONE):
    return np.where(np.abs(arr) <= dz, 0.0, np.sign(arr) * (np.abs(arr) - dz) / (1 - dz))


class JoystickNode:
    def __init__(self):
        pygame.init()
        if pygame.joystick.get_count() < 1:
            raise RuntimeError("No joystick detected")
        self.joystick = Joystick(0)

        # Three speed presets (m/s, m/s, rad/s) for Base.set_target_base_velocity,
        # cycled with L1 in the order default -> precision -> fast.
        #
        # Lowered from 0.50/0.25/0.75. Ramp time scales with how far there is to
        # go, so top speed is itself a latency knob: the old 0.75 preset spent
        # ~280 ms reaching speed no matter how hard the ramp pushed. These also
        # bring teleop back under the rest of the stack, which was already more
        # conservative — navigation caps at 0.35 m/s (BaseController.vmax) and
        # the whole-body solver at 0.25 (base_max_lin_vel). The joystick was the
        # only thing on the robot asking for 0.75.
        self.max_vels = [
            np.array([0.35, 0.35, 1.20]),
            np.array([0.18, 0.18, 0.80]),
            np.array([0.50, 0.50, 1.57]),
        ]
        self.max_vel_setting = 0

        # Weight on the *previous* command. Only here to reject stick jitter
        # that survives the deadzone — 0.9 was heavy enough to be the de facto
        # motion profile, and it lagged the stick by about a third of a second
        # in both directions. Shaping the motion is the jerk-limited ramp's job
        # (BASE_MAX_* in robot/base_motor.py); this just cleans up the input.
        self.vel_alpha = 0.25

        self.control_loop_running = False

        # RPC to YOR (which wraps the new Base)
        self.yor = RPCClient(host="localhost", port=5557)
        self.yor.init()  # starts Base control loop on the server

        # D-pad debug state
        self.last_pad_y = 0
        self.warned_no_hat = False

    def display_joystick_inputs(self):
        """Display current joystick input values"""
        print("\n" + "="*50)
        print("JOYSTICK INPUTS:")
        print("="*50)
        
        # Display axis values
        print("AXES:")
        for i in range(self.joystick.get_numaxes()):
            axis_value = self.joystick.get_axis(i)
            print(f"  Axis {i}: {axis_value:6.3f}")
        
        # Display button states
        print("BUTTONS:")
        for i in range(self.joystick.get_numbuttons()):
            button_state = self.joystick.get_button(i)
            print(f"  Button {i}: {'PRESSED' if button_state else 'RELEASED'}")
        
        # Display hat values
        num_hats = self.joystick.get_numhats()
        if num_hats > 0:
            print("HATS:")
        for i in range(num_hats):
                hat_value = self.joystick.get_hat(i)
                print(f"  Hat {i}: {hat_value}")
        else:
            print("HATS: None available")
        
        # Display current control state
        print(f"CONTROL STATE: {'RUNNING' if self.control_loop_running else 'STOPPED'}")
        print(f"MAX VEL SETTING: {self.max_vel_setting}")
        print("="*50)

    def _halt_base(self) -> None:
        """Command zero velocity and make it the last thing the relay heard.

        The relay in robot/base.py latches its last command and re-sends it at
        108 Hz indefinitely, so a joystick that merely stops talking leaves the
        base driving. Every exit from the control block goes through here: the
        stop button, and the process terminating. Sent a few times because a
        single dropped RPC would otherwise leave the robot moving.
        """
        for _ in range(3):
            try:
                self.yor.set_base_velocity(np.zeros(3, dtype=float))
            except Exception as exc:
                print(f"WARN: base halt failed: {exc}")
                return

    def control_loop(self):
        rate = RateLimiter(60, name="joystick")
        last_target_velocity = np.zeros(3, dtype=float)
        zero_tail = 0
        display_counter = 0  # Counter to control display frequency

        while True:
            pygame.event.pump()
            
            if not self.control_loop_running and self.joystick.get_button(controller_map["start"]):
                self.control_loop_running = True
                print("Control started")
                self.display_joystick_inputs()

            if self.control_loop_running and self.joystick.get_button(controller_map["back"]):
                self.control_loop_running = False
                # Stopping control has to stop the *robot*. Everything below is
                # skipped once this flag is false, so without an explicit halt
                # the relay keeps forwarding the last velocity it was given and
                # the base drives away from an operator who just pressed stop.
                self._halt_base()
                last_target_velocity = np.zeros(3, dtype=float)
                zero_tail = 0
                print("Control stopped")
                self.display_joystick_inputs()

            if self.control_loop_running:
                left_bumper = self.joystick.get_button(controller_map["l1"])
                right_bumper = self.joystick.get_button(controller_map["r1"])
                
                if left_bumper:
                    self.max_vel_setting = (self.max_vel_setting + 1) % len(self.max_vels)
                    print("Max velocity setting:", self.max_vel_setting)
                    time.sleep(0.1)
                
                # Display joystick inputs when right bumper is pressed
                if right_bumper:
                    self.display_joystick_inputs()
                    time.sleep(0.1)
                
                # Display inputs every 5 seconds (300 frames at 60Hz)
                display_counter += 1
                if display_counter >= 300:
                    self.display_joystick_inputs()
                    display_counter = 0

                # Right stick → translation (vx, vy), Left stick X → yaw rate
                vy = -self.joystick.get_axis(controller_map["right_horizontal_axis"])
                vx = -self.joystick.get_axis(controller_map["right_vertical_axis"])
                w  = -self.joystick.get_axis(controller_map["left_horizontal_axis"])
                raw = apply_deadzone(np.array([vx, vy, w], dtype=float))

                # Centred sticks are a hard zero, not a small number. Deciding
                # this on the *raw* stick rather than on the filtered magnitude
                # is what makes released mean released: the EMA decays toward
                # zero but never arrives, so a magnitude test can only ever be
                # a threshold, and any stick resting above that threshold never
                # trips it.
                sticks_idle = not raw.any()

                # scale & smooth
                target_velocity = self.max_vels[self.max_vel_setting] * raw
                target_velocity = (1 - self.vel_alpha) * target_velocity + self.vel_alpha * last_target_velocity
                if sticks_idle:
                    target_velocity = np.zeros(3, dtype=float)
                last_target_velocity = target_velocity

                # Send to Base (SparkFlex swerve).
                #
                # The zero has to be *sent*, not merely not-sent. Releasing the
                # stick decays this EMA, and simply falling silent below the
                # threshold left the last above-threshold value standing --
                # which BaseController's 108 Hz relay then re-sent forever,
                # about 10 mm/s of creep that nothing ever cleared.
                if not sticks_idle:
                    self.yor.set_base_velocity(target_velocity)
                    zero_tail = STOP_TAIL_TICKS
                elif zero_tail > 0:
                    zero_tail -= 1
                    self.yor.set_base_velocity(target_velocity)

                # D-pad up/down controls lift via Pico serial
                num_hats = self.joystick.get_numhats()
                if num_hats < 1:
                    if not self.warned_no_hat:
                        print("WARN: Controller reports 0 hats; D-pad unsupported by this device/driver")
                        self.warned_no_hat = True
                    pad_y = 0
                else:
                    pad = self.joystick.get_hat(0)
                    pad_y = pad[1]

                if pad_y != self.last_pad_y:
                    print(f"D-pad Y changed: {self.last_pad_y} -> {pad_y}")
                    if pad_y > 0:
                        print("RPC: lift_up()")
                        self.yor.lift_up()
                    elif pad_y < 0:
                        print("RPC: lift_down()")
                        self.yor.lift_down()
                    else:
                        print("RPC: lift_stop()")
                        self.yor.lift_stop()
                    self.last_pad_y = pad_y

            rate.sleep()


def main():
    node = JoystickNode()
    try:
        node.control_loop()
    finally:
        # Ctrl-C included: the relay does not time out on its own.
        node._halt_base()


if __name__ == "__main__":
    main()
