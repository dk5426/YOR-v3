import os

os.environ["CTR_TARGET"] = "Hardware"  # pylint: disable=wrong-import-position
import time
import math
import numpy as np

import phoenix6
import phoenix6.unmanaged

from phoenix6 import controls

from robot.base import SteerMotor, DriveMotor


def test_steer_motor():
    motor = SteerMotor(1)
    last_time = time.time()
    CONTROL_PERIOD = 0.004  # 250Hz

    position_log = []
    # targets =  [0.0, math.pi/4, math.pi / 2, 3*math.pi/4, math.pi]
    # targets = [
    #     -0.8 * math.pi,
    #     0.2 * math.pi,
    #     -0.5 * math.pi,
    #     0.9 * math.pi,
    #     -0.3 * math.pi
    # ]

    targets = np.random.uniform(low=-np.pi, high=np.pi, size=5)
    print(targets)
    input("Press Enter to continue...")

    try:
        i = 0
        while i < 1250:
            phoenix6.BaseStatusSignal.refresh_all(motor.status_signals)
            pos = motor.get_position()
            position_log.append(pos)

            phoenix6.unmanaged.feed_enable(0.1)

            if i >= 1000:
                motor.set_position(targets[4])
            elif i >= 750:
                motor.set_position(targets[3])
            elif i >= 500:
                motor.set_position(targets[2])
            elif i >= 250:
                motor.set_position(targets[1])
            else:
                motor.set_position(targets[0])

            i += 1
            step_time = last_time - time.time()
            if step_time < CONTROL_PERIOD:
                time.sleep(CONTROL_PERIOD - step_time)
            last_time = time.time()
    finally:
        motor.set_neutral()
        import matplotlib.pyplot as plt

        plt.plot(position_log[::5])
        # draw horizontal line at target
        for t in targets:
            plt.axhline(y=t, color="r", linestyle="--")

        plt.legend(["TalonFX"])

        plt.savefig("position_plot.png")
        plt.close()


def test_drive_motor():
    motor = DriveMotor(8)
    last_time = time.time()
    CONTROL_PERIOD = 0.004  # 250Hz

    vel_log = []
    target = 0.5

    try:
        motor.set_velocity(target)
        while True:
            phoenix6.BaseStatusSignal.refresh_all(motor.status_signals)
            vel = motor.get_velocity()
            vel_log.append(vel)
            phoenix6.unmanaged.feed_enable(0.1)

            step_time = last_time - time.time()
            if step_time < CONTROL_PERIOD:
                time.sleep(CONTROL_PERIOD - step_time)
            last_time = time.time()
    finally:
        motor.set_neutral()

        import matplotlib.pyplot as plt

        plt.plot(vel_log[::10])
        # draw horizontal line at target
        plt.axhline(y=target, color="r", linestyle="--")
        plt.legend(["TalonFX"])

        plt.savefig("velocity_plot.png")
        plt.close()


if __name__ == "__main__":
    test_steer_motor()
    # test_drive_motor()
