import os
import time
import warnings
from typing import cast

os.environ["CTR_TARGET"] = "Hardware"
import phoenix6
from phoenix6 import configs, controls, hardware
from loop_rate_limiters import RateLimiter

CONTROL_FREQ = 250

class Lift:
    def __init__(self):
        self.lift_motor = hardware.TalonFX(9, canbus="Drivetrain")

        assert self.lift_motor.get_is_pro_licensed()

        self.position_signal = self.lift_motor.get_position()
        self.velocity_signal = self.lift_motor.get_velocity()
        self.status_signals = cast(list[phoenix6.BaseStatusSignal], [self.position_signal, self.velocity_signal])

        self.lift_motor_cfg = configs.TalonFXConfiguration()

        self.lift_ratio = 0.006 * (16 / 60)

        # velocity control gains
        self.lift_motor_cfg.slot1.k_p = 5.0
        self.lift_motor_cfg.slot1.k_i = 0.0
        self.lift_motor_cfg.slot1.k_d = 0.0

        self.lift_motor_cfg.torque_current.peak_forward_torque_current = 20
        self.lift_motor_cfg.torque_current.peak_reverse_torque_current = -20
        self.lift_motor_cfg.audio.beep_on_boot = False

        self.min_pos, self.max_pos = 0.0, 0.39 # [m]

        phoenix6.BaseStatusSignal.set_update_frequency_for_all(CONTROL_FREQ, self.status_signals)

        status = self.lift_motor.configurator.apply(self.lift_motor_cfg)
        if not status.is_ok():
            raise Exception(f"Failed to apply TalonFX configuration: {status}")

        self.velocity_request = controls.VelocityTorqueCurrentFOC(0, slot=1)
        self.neutral_request = controls.NeutralOut()
        self._homed = False
        self._homing = False

    def get_position(self) -> float:
        # zero position - fully retracted - 0
        # lead screw pitch = 6mm per rev
        # motor pulley teeth = 16
        # shaft pulley teeth = 60
        # 16/60 * 6 = 1.6mm per rev
        return -self.position_signal.value * self.lift_ratio # [m]

    def get_velocity(self) -> float:
        return -self.velocity_signal.value * self.lift_ratio # [m/s]

    def home(self, upper_limit: bool = True, call_enable: bool = True) -> None:
        vel = 0.05 if upper_limit else -0.05
        self.update_state()

        pos_update_counter = 0
        pos_tolerance = 0.0005
        last_pos = self.get_position()
        current_pos = last_pos

        rate_limiter = RateLimiter(100)
        self._homing = True

        while True:
            self.update_state()
            # if call_enable:
            phoenix6.unmanaged.feed_enable(0.02)
            self.set_velocity_control(vel)
            pos_update_counter += 1
            if pos_update_counter > 50 and pos_update_counter % 5 == 0:
                current_pos = self.get_position()
                print("current_pos: ", current_pos)
                if abs(current_pos - last_pos) < pos_tolerance:
                    print("Lift homing complete")
                    self.lift_motor.set_position(-self.max_pos/self.lift_ratio if upper_limit else 0.0)
                    break
                last_pos = current_pos
            rate_limiter.sleep()

        self._homing = False
        self._homed = True

    def set_neutral(self) -> None:
        if self._homing: return # don't set neutral when homing
        self.lift_motor.set_control(self.neutral_request)

    def set_velocity_control(self, velocity: float) -> None:
        if not self._homed and not self._homing:
            warnings.warn("Lift is not homed, setting velocity to 0")
            return
        self.lift_motor.set_control(self.velocity_request.with_velocity(-velocity / self.lift_ratio))

    def update_state(self):
        phoenix6.BaseStatusSignal.refresh_all(self.status_signals)


if __name__ == "__main__":
    lift = Lift()
    lift.set_neutral()
    time.sleep(2.0)
    input("Press Enter to start")
    HOME = True
    if HOME:
        lift.home()
    else:
        lift._homed = True
        rate = RateLimiter(100)
        positions = []
        try:
            while True:
                lift.update_state()
                phoenix6.unmanaged.feed_enable(0.1)
                lift.set_velocity_control(-0.05)
                print("position: ", lift.get_position())
                positions.append(lift.get_position())
                rate.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            lift.set_neutral()
