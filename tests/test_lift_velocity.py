"""
test_lift_velocity.py — the position-to-velocity lift path, end to end on the host.

Covers the three pieces that turn `set_lift_target(height)` into millimetres
per second on a serial port, without a lift, a serial port or a robot:

    LiftVelocityPD        proportional response, measurement derivative,
                          filtering, deadband, clamp, reset
    PicoLift              wire format, unit conversion, keepalive, immediate
                          stop, invalid input, capability detection
    WholeBodyController   which path it picks, and what stops it moving

    python tests/test_lift_velocity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robot.base_motor import (  # noqa: E402
    LIFT_MAX_VELOCITY_MM_S, LIFT_MIN_VELOCITY_MM_S, LIFT_VELOCITY_CAPABILITY,
    LIFT_VELOCITY_KEEPALIVE_S, PicoLift,
)
from robot.wholebody_control import (  # noqa: E402
    LiftVelocityPD, WholeBodyController, WholeBodyHardwareConfig,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# LiftVelocityPD
# ─────────────────────────────────────────────────────────────────────────────

def make_pd(**kwargs) -> LiftVelocityPD:
    defaults = dict(kp=2.0, kd=0.05, tau=0.1, deadband=0.005,
                    max_velocity=0.05, max_gap_s=0.25)
    defaults.update(kwargs)
    return LiftVelocityPD(**defaults)


def test_pd_proportional() -> None:
    print("\nPD: proportional response")
    pd = make_pd()
    # 2 cm below target, held still: Kp * 0.02 = 0.04 m/s, under the clamp.
    v = pd.update(desired_m=0.52, measured_m=0.50, now=0.0)
    check("commands upward for a target above", v > 0, f"{v:.4f} m/s")
    check("magnitude is Kp * error", abs(v - 0.04) < 1e-9, f"{v:.4f} m/s")

    pd.reset()
    v = pd.update(desired_m=0.48, measured_m=0.50, now=0.0)
    check("commands downward for a target below", abs(v + 0.04) < 1e-9, f"{v:.4f} m/s")

    pd.reset()
    v = pd.update(desired_m=0.50, measured_m=0.50, now=0.0)
    check("zero error is zero velocity", v == 0.0, f"{v:.4f} m/s")


def test_pd_deadband() -> None:
    print("\nPD: 5 mm deadband")
    pd = make_pd()
    inside = pd.update(0.5040, 0.5, 0.0)
    check("4 mm of error commands exactly zero", inside == 0.0, f"{inside!r}")

    pd.reset()
    edge = pd.update(0.005, 0.0, 0.0)      # exactly 5 mm, in exact arithmetic
    check("5 mm is still inside the band", edge == 0.0, f"{edge!r}")

    pd.reset()
    outside = pd.update(0.5060, 0.5, 0.0)
    check("6 mm is outside and moves", outside > 0.0, f"{outside:.5f} m/s")


def test_pd_clamp() -> None:
    print("\nPD: velocity clamp")
    pd = make_pd()
    v = pd.update(0.9, 0.0, 0.0)          # 0.9 m of error -> 1.8 m/s uncapped
    check("clamped to +0.05 m/s", abs(v - 0.05) < 1e-12, f"{v:.5f} m/s")
    pd.reset()
    v = pd.update(0.0, 0.9, 0.0)
    check("clamped to -0.05 m/s", abs(v + 0.05) < 1e-12, f"{v:.5f} m/s")


def test_pd_derivative_is_of_the_measurement() -> None:
    print("\nPD: the derivative damps, and never kicks")
    # A stationary lift and a target that jumps 10 cm. Differentiating the
    # error would produce a huge transient; differentiating the measurement
    # cannot, because the measurement did not move.
    pd = make_pd()
    for i in range(5):
        pd.update(0.50, 0.50, i * 0.01)
    kicked = pd.update(0.60, 0.50, 0.05)
    check("a target step produces no derivative kick",
          abs(kicked - 0.05) < 1e-12, f"{kicked:.5f} m/s (clamped Kp term)")
    check("the filtered measurement velocity stayed zero",
          abs(pd.filtered_velocity) < 1e-12, f"{pd.filtered_velocity:.6f}")

    # Now the lift itself moves upward at 0.05 m/s while the target sits above
    # it: the D term must subtract from the command.
    pd = make_pd()
    height, t = 0.50, 0.0
    for _ in range(200):                   # 2 s at 100 Hz
        pd.update(0.70, height, t)
        height += 0.05 * 0.01
        t += 0.01
    check("the derivative converges on the true speed",
          abs(pd.filtered_velocity - 0.05) < 1e-3, f"{pd.filtered_velocity:.4f} m/s")

    # Compared just below the clamp, where the D term can still be seen: 1 cm
    # of error is 0.02 m/s of P against 0.0025 m/s of D.
    near = height + 0.01
    undamped_v = make_pd(kd=0.0).update(near, height, t)
    damped_v = pd.update(near, height, t)
    check("damping reduces the command while the lift is already moving",
          damped_v < undamped_v, f"{damped_v:.5f} < {undamped_v:.5f}")
    check("by Kd times the measured speed",
          abs((undamped_v - damped_v) - 0.05 * pd.filtered_velocity) < 1e-6,
          f"{undamped_v - damped_v:.6f} m/s")


def test_pd_filtering() -> None:
    print("\nPD: derivative filtering across mismatched rates")
    # Height arrives at 36 Hz; the loop runs at 108 Hz. Two cycles in three see
    # no change, so the raw difference is a 0 / 0 / spike pattern. Filtered, it
    # has to settle near the true 0.05 m/s.
    pd = make_pd()
    height, t, samples = 0.50, 0.0, []
    dt = 1.0 / 108.0
    for cycle in range(540):
        if cycle % 3 == 0:
            height += 0.05 / 36.0          # one 36 Hz sample of travel
        pd.update(0.90, height, t)
        if cycle >= 432:                   # after the filter has settled
            samples.append(pd.filtered_velocity)
        t += dt
    mean = sum(samples) / len(samples)
    peak = max(samples)
    check("filtered derivative tracks the average speed",
          abs(mean - 0.05) < 5e-3, f"mean {mean:.4f} m/s")
    check("and its ripple stays small", peak < 0.08, f"peak {peak:.4f} m/s")

    unfiltered = make_pd(tau=0.0)
    height, t, worst = 0.50, 0.0, 0.0
    for cycle in range(108):
        if cycle % 3 == 0:
            height += 0.05 / 36.0
        unfiltered.update(0.90, height, t)
        worst = max(worst, abs(unfiltered.filtered_velocity))
        t += dt
    check("an unfiltered derivative would spike well past the truth",
          worst > 0.14, f"peak {worst:.3f} m/s vs true 0.05")


def test_pd_reset() -> None:
    print("\nPD: reset and control-loop gaps")
    pd = make_pd()
    height, t = 0.50, 0.0
    for _ in range(100):
        pd.update(0.90, height, t)
        height += 0.05 * 0.01
        t += 0.01
    check("derivative is populated before the reset", pd.filtered_velocity > 0.01,
          f"{pd.filtered_velocity:.4f}")

    pd.reset()
    check("reset clears the filter", pd.filtered_velocity == 0.0)
    v = pd.update(0.90, height, t + 0.01)
    check("the first cycle after a reset is pure proportional",
          abs(v - 0.05) < 1e-12, f"{v:.5f} m/s")

    # A stalled loop: the height difference across the gap is not a velocity.
    pd = make_pd()
    pd.update(0.90, 0.50, 0.0)
    pd.update(0.90, 0.55, 5.0)             # 5 s later, 5 cm on
    check("a control-loop gap invalidates the derivative",
          pd.filtered_velocity == 0.0, f"{pd.filtered_velocity:.4f}")

    pd = make_pd()
    pd.update(0.90, 0.50, 1.0)
    pd.update(0.90, 0.50, 1.0)             # time did not advance
    check("a repeated timestamp does not divide by zero",
          pd.filtered_velocity == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# PicoLift transport
# ─────────────────────────────────────────────────────────────────────────────

class FakeSerial:
    """Enough pyserial to run PicoLift's writer without a port."""

    is_open = True

    def __init__(self):
        self.writes: list[bytes] = []

    def write(self, payload):
        self.writes.append(bytes(payload))
        return len(payload)

    def flush(self):
        pass

    def close(self):
        self.is_open = False

    def readline(self):
        return b""

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass


def make_lift() -> tuple[PicoLift, FakeSerial]:
    """A PicoLift with no port opened and no drain thread running."""
    lift = PicoLift.__new__(PicoLift)
    lift.device_path, lift.baud, lift.timeout = "/dev/null", 115200, 0.2
    lift._lock = __import__("threading").Lock()
    lift._vel_lock = __import__("threading").Lock()
    lift._height_lock = __import__("threading").Lock()
    lift._last_cmd = None
    lift._last_send_ts = 0.0
    lift._min_repeat_interval = 0.05
    lift._drain_thread = None
    lift._drain_stop = __import__("threading").Event()
    lift._last_velocity_mm_s = None
    lift._last_velocity_send_ts = 0.0
    lift._height_m = None
    lift._height_ts = None
    lift._capabilities = set()
    lift._position_known = None
    lift._homed = None
    lift._upper_limit = None
    lift._lower_limit = None
    lift._motion = None
    lift._last_event = None

    serial_port = FakeSerial()
    lift._ser = serial_port
    lift._ensure_open = lambda: None
    return lift, serial_port


def sent(port: FakeSerial) -> list[str]:
    return [w.decode().strip() for w in port.writes if w.strip() not in (b"", b"\r\n")]


def test_wire_format() -> None:
    print("\nPicoLift: wire format and units")
    lift, port = make_lift()

    check("accepts a finite velocity", lift.set_velocity_mm_s(12.5) is True)
    check("sends signed mm/s", sent(port) == ["vel 12.50"], str(sent(port)))

    port.writes.clear()
    lift.set_velocity_mm_s(-7.25)
    check("negative is down", sent(port) == ["vel -7.25"], str(sent(port)))

    port.writes.clear()
    lift.set_velocity_mm_s(999.0)
    check("clamped to the firmware maximum",
          sent(port) == [f"vel {LIFT_MAX_VELOCITY_MM_S:.2f}"], str(sent(port)))

    port.writes.clear()
    lift.set_velocity_mm_s(-999.0)
    check("clamped symmetrically",
          sent(port) == [f"vel {-LIFT_MAX_VELOCITY_MM_S:.2f}"], str(sent(port)))

    port.writes.clear()
    lift.set_velocity_mm_s(0.2)            # below the minimum active velocity
    check("a sub-threshold creep becomes an explicit zero",
          sent(port) == ["vel 0.00"], str(sent(port)))


def test_invalid_input() -> None:
    print("\nPicoLift: invalid input never reaches the port")
    for bad in (float("nan"), float("inf"), float("-inf")):
        lift, port = make_lift()
        accepted = lift.set_velocity_mm_s(bad)
        check(f"rejects {bad}", accepted is False and not sent(port), str(sent(port)))

    lift, port = make_lift()
    check("rejects a non-number", lift.set_velocity_mm_s("fast") is False and not sent(port))
    lift, port = make_lift()
    check("rejects None", lift.set_velocity_mm_s(None) is False and not sent(port))


def test_keepalive_and_stop() -> None:
    print("\nPicoLift: keepalive, change and stop")
    lift, port = make_lift()

    lift.set_velocity_mm_s(10.0)
    port.writes.clear()
    for _ in range(50):                    # a fast loop repeating the same value
        lift.set_velocity_mm_s(10.0)
    check("an unchanged command is not re-sent every cycle", len(sent(port)) == 0,
          f"{len(sent(port))} frames")

    time.sleep(LIFT_VELOCITY_KEEPALIVE_S + 0.01)
    lift.set_velocity_mm_s(10.0)
    check("but it is refreshed as a keepalive", sent(port) == ["vel 10.00"], str(sent(port)))

    port.writes.clear()
    lift.set_velocity_mm_s(10.0 + LIFT_MIN_VELOCITY_MM_S)
    check("a meaningful change goes out at once",
          sent(port) == ["vel 10.50"], str(sent(port)))

    port.writes.clear()
    lift.set_velocity_mm_s(10.6)           # 0.1 mm/s later: not meaningful
    check("a change the lift cannot express waits for the keepalive",
          not sent(port), str(sent(port)))

    port.writes.clear()
    lift.set_velocity_mm_s(0.0)
    check("stopping is never deferred", sent(port) == ["vel 0.00"], str(sent(port)))

    port.writes.clear()
    lift.set_velocity_mm_s(0.0)
    check("a repeated zero is rate-limited like any other value", not sent(port))

    # The keepalive has to be comfortably inside the firmware's timeout.
    check("keepalive is inside the firmware's 300 ms command timeout",
          LIFT_VELOCITY_KEEPALIVE_S <= 0.15, f"{LIFT_VELOCITY_KEEPALIVE_S}s")


def test_stop_clears_the_gate() -> None:
    print("\nPicoLift: a discrete stop re-arms the stream")
    lift, port = make_lift()
    lift.set_velocity_mm_s(10.0)
    lift.stop()
    port.writes.clear()
    lift.set_velocity_mm_s(10.0)
    check("the same velocity is re-sent after a stop",
          sent(port) == ["vel 10.00"], str(sent(port)))


def test_capability_detection() -> None:
    print("\nPicoLift: capability detection")
    lift, _port = make_lift()
    check("no capability until the firmware says so", lift.supports_velocity() is False)

    lift._parse_line("Lift controller ready.")
    lift._parse_line("Capabilities: lift_velocity_v1")
    check("the capability line is parsed", lift.supports_velocity() is True)
    check("tokens are retained", LIFT_VELOCITY_CAPABILITY in lift.get_capabilities(),
          str(lift.get_capabilities()))

    lift._parse_line("Lift controller ready.")
    check("a controller reset invalidates the capability",
          lift.supports_velocity() is False)

    old = make_lift()[0]
    for line in ("Lift controller ready.",
                 "Python commands: up, down, stop, home",
                 "Optional commands: up 200, down 200, status, power on, power off"):
        old._parse_line(line)
    check("an older firmware never claims the capability",
          old.supports_velocity() is False)


def test_height_age() -> None:
    print("\nPicoLift: height telemetry age")
    lift, _port = make_lift()
    check("no age before the first reading", lift.get_height_age() is None)

    lift._parse_line("Height: 250.000 mm")
    age = lift.get_height_age()
    check("a reading has an age", age is not None and age < 0.1, str(age))
    check("and a height", abs(lift.get_height() - 0.250) < 1e-9, str(lift.get_height()))

    lift._parse_line("Height: unknown (run home)")
    check("an unknown height clears the age", lift.get_height_age() is None)
    check("and the height", lift.get_height() is None)

    lift._parse_line("Height: 300.000 mm")
    lift._parse_line("Home failed: upper limit was not reached.")
    check("a failed home clears both", lift.get_height() is None
          and lift.get_height_age() is None)


# ─────────────────────────────────────────────────────────────────────────────
# WholeBodyController dispatch
# ─────────────────────────────────────────────────────────────────────────────

class FakeArm:
    def __init__(self, q0):
        self.q = np.asarray(q0, dtype=float).copy()

    def get_joint_positions(self):
        return self.q.copy()

    def set_joint_target(self, joint_target, gripper_target=None, preview_time=0.1):
        self.q = np.asarray(joint_target, dtype=float).copy()


class VelocityBase:
    """A base whose lift takes streamed velocity, and integrates it."""

    def __init__(self, lift_height=0.20, capable=True, stale=False, known=True):
        self.lift_height = float(lift_height)
        self.capable = capable
        self.stale = stale
        self.known = known
        self.velocity_commands: list[float] = []
        self.discrete_commands: list[str] = []
        self.velocity = 0.0
        self._t = time.monotonic()

    # -- lift, streamed --
    def lift_supports_velocity(self):
        return self.capable

    def lift_set_velocity(self, velocity_m_s):
        self._integrate()
        self.velocity = float(velocity_m_s)
        self.velocity_commands.append(self.velocity)
        return True

    def get_lift_height_age(self):
        if self.stale:
            return 5.0
        return None if self.lift_height is None else 0.01

    def lift_position_known(self):
        return self.known

    # -- lift, discrete --
    def lift_up(self):
        self._integrate()
        self.discrete_commands.append("up")
        self.velocity = 0.05

    def lift_down(self):
        self._integrate()
        self.discrete_commands.append("down")
        self.velocity = -0.05

    def lift_stop(self):
        self._integrate()
        self.discrete_commands.append("stop")
        self.velocity = 0.0

    def get_lift_height(self):
        self._integrate()
        return self.lift_height

    def _integrate(self):
        now = time.monotonic()
        dt, self._t = now - self._t, now
        if self.lift_height is not None:
            self.lift_height = float(np.clip(
                self.lift_height + self.velocity * dt, 0.0, 0.900))

    # -- drive --
    def set_target_base_velocity(self, target, smooth=False):
        pass


def build(base: VelocityBase, **cfg) -> WholeBodyController:
    cfg.setdefault("enable_base_motion", False)
    wbc = WholeBodyController(
        FakeArm([0.0, 1.32, -1.71, 1.31, 0.0, 0.0, 0.0]),
        FakeArm([0.0, 1.32, 1.71, 1.31, 0.0, 0.0, 0.0]),
        base, config=WholeBodyHardwareConfig(**cfg))
    wbc.init()
    return wbc


def test_controller_picks_the_velocity_path() -> None:
    print("\nWholeBodyController: streamed velocity when the firmware supports it")
    base = VelocityBase(lift_height=0.20)
    wbc = build(base)
    wbc.set_lift_target(0.45)
    for _ in range(30):
        wbc._step()

    check("velocity was streamed", bool(base.velocity_commands),
          f"{len(base.velocity_commands)} commands")
    check("no bang-bang command was sent", not base.discrete_commands,
          str(base.discrete_commands))
    check("it drove upward toward the target", base.velocity_commands[-1] > 0,
          f"{base.velocity_commands[-1]:.4f} m/s")
    check("every command is inside the clamp",
          all(abs(v) <= wbc.config.lift_max_velocity_m_s + 1e-12
              for v in base.velocity_commands),
          f"peak {max(abs(v) for v in base.velocity_commands):.4f} m/s")
    check("the state snapshot says which path is live",
          wbc.get_state()["lift_velocity_mode"] is True)


def test_controller_reaches_and_holds() -> None:
    print("\nWholeBodyController: converges and then holds at zero")
    base = VelocityBase(lift_height=0.20)
    wbc = build(base)
    wbc.set_lift_target(0.24)
    for _ in range(400):
        wbc._step()
        time.sleep(0.002)

    check("the lift reached the target", abs(base.lift_height - 0.24) < 0.01,
          f"{base.lift_height:.4f} m")
    check("and was commanded to exactly zero", base.velocity_commands[-1] == 0.0,
          f"{base.velocity_commands[-1]}")


def test_fallback_without_capability() -> None:
    print("\nWholeBodyController: fallback when the firmware cannot stream")
    base = VelocityBase(lift_height=0.20, capable=False)
    wbc = build(base)
    wbc.set_lift_target(0.45)
    for _ in range(30):
        wbc._step()

    check("falls back to up/down/stop", bool(base.discrete_commands),
          str(base.discrete_commands))
    check("no velocity was streamed", not base.velocity_commands,
          str(base.velocity_commands))
    check("it still drove the right way", base.discrete_commands[0] == "up",
          str(base.discrete_commands))
    check("the state snapshot says so", wbc.get_state()["lift_velocity_mode"] is False)


def test_stale_feedback_stops_the_lift() -> None:
    print("\nWholeBodyController: stale height stops the lift")
    base = VelocityBase(lift_height=0.20)
    wbc = build(base, lift_feedback_grace_s=0.0)
    wbc.set_lift_target(0.45)
    for _ in range(5):
        wbc._step()
    check("it was moving first", base.velocity_commands[-1] > 0,
          f"{base.velocity_commands[-1]:.4f}")

    base.stale = True
    for _ in range(10):
        wbc._step()
    check("stale telemetry brings it to zero", base.velocity_commands[-1] == 0.0,
          f"{base.velocity_commands[-1]}")
    check("and the derivative state was dropped", wbc.lift_pd.filtered_velocity == 0.0)

    before = len(base.velocity_commands)
    for _ in range(20):
        wbc._step()
    check("it does not restart while the height stays stale",
          all(v == 0.0 for v in base.velocity_commands[before:]),
          str(set(base.velocity_commands[before:])))


def test_unknown_height_stops_the_lift() -> None:
    print("\nWholeBodyController: unknown height stops the lift")
    base = VelocityBase(lift_height=0.20)
    wbc = build(base)
    wbc.set_lift_target(0.45)
    for _ in range(5):
        wbc._step()
    check("it was moving first", base.velocity_commands[-1] > 0)

    base.known = False                      # the controller lost its zero
    for _ in range(10):
        wbc._step()
    check("an unestablished zero stops it", base.velocity_commands[-1] == 0.0,
          f"{base.velocity_commands[-1]}")

    base.known = True
    base.lift_height = None                 # telemetry gone entirely
    base.velocity_commands.clear()
    for _ in range(10):
        wbc._step()
    check("so does a missing height",
          all(v == 0.0 for v in base.velocity_commands) if base.velocity_commands else True,
          str(base.velocity_commands))


def test_manual_override_and_halt() -> None:
    print("\nWholeBodyController: override and halt reset the PD")
    base = VelocityBase(lift_height=0.20)
    wbc = build(base)
    wbc.set_lift_target(0.45)
    for _ in range(20):
        wbc._step()
        time.sleep(0.001)
    check("the derivative was running", wbc.lift_pd.filtered_velocity != 0.0,
          f"{wbc.lift_pd.filtered_velocity:.5f}")

    wbc.notify_manual_lift_command()
    commands_before = len(base.velocity_commands)
    for _ in range(10):
        wbc._step()
    check("a manual lift command suspends the loop's authority",
          len(base.velocity_commands) == commands_before)
    check("and clears the derivative", wbc.lift_pd.filtered_velocity == 0.0)

    wbc._manual_lift_until = 0.0
    for _ in range(10):
        wbc._step()
        time.sleep(0.001)
    check("authority returns afterwards", len(base.velocity_commands) > commands_before)

    wbc._halt_lift()
    check("halting stops the lift", base.discrete_commands[-1] == "stop",
          str(base.discrete_commands[-1]))
    check("and resets the PD", wbc.lift_pd.filtered_velocity == 0.0)


def test_target_stays_a_position() -> None:
    print("\nWholeBodyController: set_lift_target is still a position")
    base = VelocityBase(lift_height=0.20)
    wbc = build(base)
    wbc.set_lift_target(5.0)                # far above the model's travel
    check("an out-of-range target is clamped to the model",
          wbc.lift_target == wbc.ik.lift_range[1], f"{wbc.lift_target}")
    wbc.set_lift_target(-5.0)
    check("and at the bottom", wbc.lift_target == wbc.ik.lift_range[0],
          f"{wbc.lift_target}")
    wbc.set_lift_target(0.42)
    check("an in-range target is kept as metres", abs(wbc.lift_target - 0.42) < 1e-9)


def main() -> int:
    for test in (
        test_pd_proportional,
        test_pd_deadband,
        test_pd_clamp,
        test_pd_derivative_is_of_the_measurement,
        test_pd_filtering,
        test_pd_reset,
        test_wire_format,
        test_invalid_input,
        test_keepalive_and_stop,
        test_stop_clears_the_gate,
        test_capability_detection,
        test_height_age,
        test_controller_picks_the_velocity_path,
        test_controller_reaches_and_holds,
        test_fallback_without_capability,
        test_stale_feedback_stops_the_lift,
        test_unknown_height_stops_the_lift,
        test_manual_override_and_halt,
        test_target_stays_a_position,
    ):
        test()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    failures = [name for name, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
