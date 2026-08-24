"""
test_base_kinematics.py — the swerve command path, without a CAN bus.

Covers the fixes from docs/BASE_COMMAND_LOOP_REVIEW.md that were applied on
2026-08-22, plus the geometry they sit on. Everything here runs against the
real functions in robot/base_motor.py and robot/wholebody_control.py with a
stubbed SparkFlex, so it exercises the code the robot runs rather than a
restatement of it.

The two behavioural fixes worth naming:

  * a zero base command used to re-aim all four modules to 0 degrees, because
    arctan2(0, 0) is 0 and the steering setpoint was written unconditionally;
  * the whole-body dispatch deadbanded and clamped forward and lateral
    independently, which rotated the commanded direction rather than merely
    scaling it.

    python tests/test_base_kinematics.py
"""

from __future__ import annotations

import math
import sys
import time
import types
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# ─────────────────────────────────────────────────────────────────────────────
# Fake hardware
# ─────────────────────────────────────────────────────────────────────────────

class FakeSpark:
    """A SPARK that records setpoints and reports them back as feedback."""

    def __init__(self, interface, device_id):
        self.can_id = int(device_id)
        self.position_setpoints: list[float] = []
        self.velocity_setpoints: list[float] = []
        self.heartbeats = 0
        self.idle_mode = None
        self.ctrl_type = None
        self.status2_period = None

    def SetPosition(self, f): self.position_setpoints.append(float(f))
    def SetVelocity(self, v): self.velocity_setpoints.append(float(v))
    def Heartbeat(self): self.heartbeats += 1
    def SetIdleMode(self, mode): self.idle_mode = mode
    def SetCtrlType(self, t): self.ctrl_type = t
    def SetPeriodicStatus2Period(self, p): self.status2_period = p

    # Feedback: the absolute encoder mirrors the last position setpoint, so a
    # module is modelled as reaching whatever it was told. Reported in TURNS,
    # which is what the real GetAbsoluteEncoderPosition returns -- not degrees.
    def GetAbsoluteEncoderPosition(self):
        return self.position_setpoints[-1] if self.position_setpoints else 0.0

    def GetVelocity(self):
        return self.velocity_setpoints[-1] if self.velocity_setpoints else 0.0

    def GetPosition(self): return 0.0
    def GetIdleModeRaw(self): return 0
    def GetCtrlType(self): return 1
    def GetVelocityConversionFactor(self): return 0.00083203
    def GetPositionConversionFactor(self): return 1.0


class FakePicoLift:
    def __init__(self, *a, **k): pass
    def _shutdown(self): pass
    def get_capabilities(self): return set()


# What the installed binding really exports, captured before the stub goes in.
# base_motor.py imports enums from sparkcan_py behind a bare `except`, so a name
# that does not exist takes the whole import down and silently disables every
# guard that depends on *any* of them -- which is exactly the bug this file
# regression-tests. Checking the stub would prove nothing, so ask the real one.
try:
    import sparkcan_py as _real
    REAL_SPARKCAN_EXPORTS = {n for n in dir(_real) if not n.startswith("_")}
    del sys.modules["sparkcan_py"], _real
except Exception:                       # pragma: no cover - binding not built
    REAL_SPARKCAN_EXPORTS = None


class _FakeEnum(int):
    pass


_stub = types.ModuleType("sparkcan_py")
_stub.SparkFlex = FakeSpark
_stub.CtrlType = types.SimpleNamespace(kVelocity=_FakeEnum(1), kPosition=_FakeEnum(0))
_stub.IdleMode = types.SimpleNamespace(kCoast=_FakeEnum(0), kBrake=_FakeEnum(1))
sys.modules["sparkcan_py"] = _stub

import robot.base_motor as bm                                        # noqa: E402
from robot.nav.odometry.swerve_odom import SwerveOdom                # noqa: E402

bm.PicoLift = FakePicoLift


RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def _ang(f, l): return math.atan2(l, f)
def _wrap(a): return math.atan2(math.sin(a), math.cos(a))


def bare_base():
    """A Base with only what the kinematics touches — no motors, no threads."""
    base = bm.Base.__new__(bm.Base)
    base.steer_pos = np.zeros(bm.NUM_SWERVES)
    return base


def live_base():
    """A fully constructed Base against fake controllers."""
    return bm.Base()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

def test_inverse_kinematics_round_trips() -> None:
    """base_motor's IK must agree with the shipped forward model.

    Cross-checked against robot/nav/odometry/swerve_odom.py's least-squares
    solver, given base_motor's own CAD geometry rather than swerve_odom's
    calibrated one. Two independent implementations agreeing is what makes
    this a check and not a restatement.

    This is also the regression test for the forward-kinematics matrix that
    used to live in base_motor.py: `self.C` had the wrong sign on the
    omega->vx coupling and round-tripped +1.0 rad/s back as -0.316, which is
    why it was deleted rather than kept.
    """
    print("\ninverse kinematics")
    base = bare_base()

    # base_motor applies ROT_DIAG_SWAP_PERM to the rotation contribution, which
    # works out to modules at (+/-LENGTH, +/-WIDTH) in FL, FR, RR, RL order.
    positions = np.array([
        [+bm.LENGTH, +bm.WIDTH],
        [+bm.LENGTH, -bm.WIDTH],
        [-bm.LENGTH, -bm.WIDTH],
        [-bm.LENGTH, +bm.WIDTH],
    ], dtype=float)
    odom = SwerveOdom()

    cases = {
        "pure forward": [0.0, 0.25, 0.0],
        "pure lateral": [0.25, 0.0, 0.0],
        "pure spin": [0.0, 0.0, 1.0],
        "diagonal": [0.18, 0.18, 0.0],
        "arc": [0.20, 0.0, 0.5],
        "everything": [0.12, -0.09, -0.4],
    }
    for label, v in cases.items():
        base.steer_pos = np.zeros(4)
        speeds, angles = base._vehicle_velocity_to_angle_and_speed(
            np.array(v, dtype=float), cos_error_scaling=False)
        back = odom._solve_fk_displacement(angles, speeds, positions)
        # swerve_odom regularises with LAMBDA=1e-3, which biases omega by
        # ~0.5% because the yaw column of A'A is much smaller than the
        # translation ones. 1.5% covers that without hiding a real error.
        err = np.abs(back - np.array(v))
        tol = 0.015 * max(np.abs(v).max(), 1e-6) + 1e-9
        check(f"{label} round-trips through the forward model",
              bool(np.all(err <= max(tol, 0.005))),
              f"{np.round(v,3)} -> {np.round(back,4)}")

    check("the wrong forward-kinematics matrix is gone",
          not hasattr(bm.Base, "_angle_and_speed_to_vehicle_velocity")
          and not hasattr(bm.Base, "_map_steer_angles"))


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 — a stop must not re-aim the modules
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_command_holds_steering() -> None:
    print("\nzero command holds the steering angle")
    base = bare_base()

    for label, v in (("forward", [0.0, 0.25, 0.0]),
                     ("diagonal", [0.18, 0.18, 0.0]),
                     ("spin", [0.0, 0.0, 0.6])):
        base.steer_pos = np.zeros(4)
        _speeds, angles = base._vehicle_velocity_to_angle_and_speed(
            np.array(v, dtype=float), cos_error_scaling=False)
        base.steer_pos = angles.copy()            # module reached the command

        speeds0, angles0 = base._vehicle_velocity_to_angle_and_speed(
            np.zeros(3), cos_error_scaling=False)
        moved = np.degrees(np.abs(bm.diff_angle(angles0, angles)))
        check(f"{label}: a stop leaves the modules where they are",
              bool(np.all(moved < 1e-9)),
              f"was {np.round(np.degrees(angles),1)} deg, moved {np.round(moved,1)} deg")
        check(f"{label}: a stop still commands zero drive speed",
              bool(np.all(np.abs(speeds0) < 1e-12)))

    # The hold must not freeze a module that is genuinely being asked to move.
    base.steer_pos = np.zeros(4)
    _s, a1 = base._vehicle_velocity_to_angle_and_speed(
        np.array([0.0, 0.25, 0.0]), cos_error_scaling=False)
    check("a real command still re-aims", bool(np.all(np.abs(a1) > 1.0)),
          f"{np.round(np.degrees(a1),1)} deg")

    # ...and the threshold has to sit below anything ever commanded. The
    # whole-body dispatch deadband is 0.02 m/s at the chassis.
    base.steer_pos = np.zeros(4)
    s_small, a_small = base._vehicle_velocity_to_angle_and_speed(
        np.array([0.0, 0.02, 0.0]), cos_error_scaling=False)
    check("the smallest command the dispatch can send still steers",
          bool(np.all(np.abs(a_small) > 1.0)) and bool(np.all(s_small > 0)),
          f"{np.round(np.degrees(a_small),1)} deg at {np.round(s_small,4)} m/s")


def test_absolute_encoder_is_read_as_turns() -> None:
    """GetAbsoluteEncoderPosition returns turns, and nothing may divide by 360.

    The bug this pins: dividing a 0..1 turns reading by 360 collapses it to
    nearly zero, so the recovered angle becomes a constant fixed only by the
    module offset. It is silent -- every module reports a plausible, steady
    angle -- and it was only caught because the 2026-08-22 logs showed all four
    modules pinned at 90.0 / 0.0 / -90.0 / -180.0 degrees with 1 degree of
    total spread while the commanded angles swept the full circle.

    It also mattered more than a logging bug: `get_position_rad` had the same
    division on its USE_FEEDBACK_FOR_STEER branch, so turning that flag on --
    the recommended next experiment -- would have fed the steering optimizer a
    constant.
    """
    print("\nabsolute encoder units")
    for offset in (0.0, 0.25, 0.5, 0.75):
        motor = bm.RotationMotor("can0", 5, offset)
        worst = 0.0
        for cmd in (-3.0, -1.57, -0.4, 0.0, 0.5, 1.57, 3.0):
            motor.set_position_fraction(float(bm.rad_to_frac(cmd)))
            motor.dev.position_setpoints[-1] %= 1.0          # the encoder wraps
            got = motor.get_absolute_rad()
            want = math.atan2(math.sin(cmd), math.cos(cmd))
            worst = max(worst, abs(math.atan2(math.sin(got - want),
                                              math.cos(got - want))))
        check(f"offset {offset}: measured angle round-trips",
              worst < 1e-9, f"worst {math.degrees(worst):.4f} deg")

    motor = bm.RotationMotor("can0", 5, 0.25)
    motor.set_position_fraction(0.30)
    check("a full turn of travel is a full turn of reading",
          abs(motor.get_position_deg() - 360.0 * ((0.30 + 0.25) % 1.0)) < 1e-6,
          f"{motor.get_position_deg():.2f} deg")
    motor.dev.GetAbsoluteEncoderPosition = lambda: float("nan")
    check("a dead encoder reads nan, not a plausible angle",
          math.isnan(motor.get_absolute_rad()))


def test_axis_map_is_not_crossed() -> None:
    """Forward must reach the element the wheels steer to 0 degrees.

    base_motor builds each wheel vector as atan2(target[1], target[0]), so
    element 0 aims the modules at 0 degrees and element 1 at +90. The previous
    codebase measured on blocks that 0 degrees is physically forward. BaseAxisMap
    had those two crossed, so a whole-body reach that needed the base to drive
    forward strafed sideways instead.
    """
    print("\nbase axis map")
    from robot.wholebody_control import BaseAxisMap
    amap = BaseAxisMap()
    base = bare_base()

    for label, (fwd, lat, yaw), want_deg in (
        ("forward", (0.2, 0.0, 0.0), 0.0),
        ("left", (0.0, 0.2, 0.0), 90.0),
        ("back", (-0.2, 0.0, 0.0), 0.0),      # reversed wheel, same axis
    ):
        cmd = amap.to_command(fwd, lat, yaw)
        base.steer_pos = np.zeros(4)
        speeds, angles = base._vehicle_velocity_to_angle_and_speed(cmd, False)
        got = np.degrees(np.abs(angles))
        got = np.minimum(got, 180.0 - got)     # fold out the 180-degree reversal
        check(f"{label} steers the modules to {want_deg:.0f} deg",
              bool(np.all(np.abs(got - want_deg) < 1e-6)),
              f"{np.round(np.degrees(angles), 1)} deg")

    check("forward is element 0", amap.forward_index == 0)
    check("lateral is element 1", amap.lateral_index == 1)
    check("the joystick agrees: it has always put its forward stick in element 0",
          "target_velocity = np.array([vx, vy, w]"
          in (_REPO / "robot/teleop/joystick.py").read_text())


def test_shortest_path_flip_survives() -> None:
    """Reversing rather than slewing 180 degrees still has to work."""
    print("\nshortest-path wheel flip")
    base = bare_base()
    base.steer_pos = np.zeros(4)                          # pointing +x
    speeds, angles = base._vehicle_velocity_to_angle_and_speed(
        np.array([-0.25, 0.0, 0.0]), cos_error_scaling=False)
    check("a 180-degree reversal drives backwards instead of slewing",
          bool(np.all(np.abs(angles) < 1e-9)) and bool(np.all(speeds < 0)),
          f"{np.round(np.degrees(angles),1)} deg at {np.round(speeds,3)} m/s")


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — the deadband and clamp must not rotate the command
# ─────────────────────────────────────────────────────────────────────────────

def test_linear_limits_preserve_direction() -> None:
    print("\nlinear deadband and clamp")
    from robot.wholebody_control import WholeBodyController, WholeBodyHardwareConfig

    wbc = WholeBodyController.__new__(WholeBodyController)
    wbc.config = WholeBodyHardwareConfig()
    dead = wbc.config.base_vel_deadband
    limit = wbc.config.base_max_lin_vel

    # Direction is preserved wherever the command survives at all. Probes are
    # derived from the configured deadband, so raising it cannot quietly turn
    # these into checks on vectors that get dropped anyway.
    for fwd, lat in ((1.6 * dead, 1.2 * dead), (2.5 * dead, 0.95 * dead),
                     (0.40, 0.15), (-1.5 * dead, 1.0 * dead)):
        f2, l2 = wbc._limit_linear(fwd, lat)
        want = math.degrees(math.atan2(lat, fwd))
        got = math.degrees(math.atan2(l2, f2))
        check(f"({fwd:+.3f},{lat:+.3f}) keeps its direction",
              abs(want - got) < 1e-9, f"{want:.1f} deg -> {got:.1f} deg")

    # The deadband is on the magnitude, so it is symmetric in direction.
    small = (0.6 * dead / math.sqrt(2), 0.6 * dead / math.sqrt(2))
    check("a request below the deadband is dropped whole",
          wbc._limit_linear(*small) == (0.0, 0.0),
          f"|v|={math.hypot(*small):.4f} < {dead}")
    just_over = (1.02 * dead / math.sqrt(2), 1.02 * dead / math.sqrt(2))
    check("...and a diagonal just over it survives",
          math.hypot(*wbc._limit_linear(*just_over)) > 0.0,
          f"|v|={math.hypot(*just_over):.4f} vs deadband {dead}")

    # The behaviour change worth naming: per-axis deadbanding dropped this
    # entirely (both components under 0.02) even though the robot was being
    # asked for 27 mm/s. On the magnitude it survives, pointing where it was
    # asked to point.
    diag = (0.9 * dead, 0.9 * dead)          # each axis under, magnitude over
    f2, l2 = wbc._limit_linear(*diag)
    check("a diagonal whose axes are each under the deadband is no longer lost",
          math.hypot(f2, l2) > 0.0 and abs(f2 - l2) < 1e-12,
          f"|v|={math.hypot(*diag):.4f} > {dead}, each axis {diag[0]:.4f} < {dead}")

    # The clamp is on the magnitude too.
    f2, l2 = wbc._limit_linear(1.0, 1.0)
    check("the clamp limits the magnitude, not each axis",
          abs(math.hypot(f2, l2) - limit) < 1e-9, f"|v|={math.hypot(f2,l2):.4f}")
    check("a clamped diagonal is still diagonal",
          abs(f2 - l2) < 1e-12, f"({f2:.4f},{l2:.4f})")
    check("each axis is still within the limit (old contract kept)",
          abs(f2) <= limit + 1e-12 and abs(l2) <= limit + 1e-12)

    # Yaw keeps its own deadband, in its own units.
    check("yaw has a separate deadband",
          hasattr(wbc.config, "base_yaw_deadband"))
    check("yaw below its deadband is zeroed",
          wbc._clamp(0.01, 0.6, wbc.config.base_yaw_deadband) == 0.0)
    check("yaw above it is passed through",
          abs(wbc._clamp(0.3, 0.6, wbc.config.base_yaw_deadband) - 0.3) < 1e-12)
    check("yaw is clamped to its own limit",
          abs(wbc._clamp(5.0, 0.6, wbc.config.base_yaw_deadband) - 0.6) < 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Profiler, controller configuration, telemetry
# ─────────────────────────────────────────────────────────────────────────────

def test_heading_rate_limit() -> None:
    """The commanded direction may not turn faster than the modules can.

    Measured on 2026-08-22: the solver asked for a median 552 deg/s of module
    travel whenever the base was creeping, against hardware that peaks at
    265-353 deg/s, and 27-44% of moving ticks were beyond reach. The cause is
    that atan2 of a short vector is ill-conditioned, so the slower the base is
    asked to go the faster its heading whirls.
    """
    print("\nheading rate limit")
    from robot.wholebody_control import WholeBodyController, WholeBodyHardwareConfig

    wbc = WholeBodyController.__new__(WholeBodyController)
    wbc.config = WholeBodyHardwareConfig()
    wbc.dt = 1.0 / wbc.config.control_hz
    wbc._base_heading = None
    limit = wbc.config.base_heading_rate_limit
    step = limit * wbc.dt

    check("a limit is configured under the slowest measured module (265 deg/s)",
          0 < math.degrees(limit) < 265, f"{math.degrees(limit):.0f} deg/s")

    # First command seeds the reference and passes through untouched.
    f, l = wbc._limit_heading_rate(0.2, 0.0)
    check("the first command is not limited", abs(f - 0.2) < 1e-12 and abs(l) < 1e-12)

    # A 90-degree flick is held to one step, with the speed preserved.
    f, l = wbc._limit_heading_rate(0.0, 0.2)
    turned = abs(_ang(f, l) - 0.0)
    check("a 90 deg flick is limited to one step",
          abs(turned - step) < 1e-9, f"{math.degrees(turned):.2f} vs {math.degrees(step):.2f} deg")
    check("...and the speed is preserved, not shrunk",
          abs(math.hypot(f, l) - 0.2) < 1e-12, f"{math.hypot(f, l):.4f}")

    # It converges rather than sticking: repeated ticks walk it round.
    for _ in range(200):
        f, l = wbc._limit_heading_rate(0.0, 0.2)
    check("it converges on the requested heading",
          abs(_wrap(_ang(f, l) - math.pi / 2)) < 1e-9,
          f"{math.degrees(_ang(f, l)):.2f} deg")

    # A reversal is free: the module flips the drive instead of turning.
    wbc._base_heading = 0.0
    f, l = wbc._limit_heading_rate(-0.2, 0.0)
    check("a 180 deg reversal is not rate-limited (the wheel flips instead)",
          abs(_wrap(_ang(f, l) - math.pi)) < 1e-9, f"{math.degrees(_ang(f, l)):.1f} deg")

    # A change just short of 180 is a small move once reversed, so it passes
    # through untouched -- where measuring it the long way round would have
    # rate-limited a move the hardware makes in one tick.
    for deg in (180.0 - math.degrees(step) * 0.5, 175.0):
        wbc._base_heading = 0.0
        want = math.radians(deg)
        f, l = wbc._limit_heading_rate(0.2 * math.cos(want), 0.2 * math.sin(want))
        check(f"a {deg:.0f} deg change passes through as the small move it really is",
              abs(_wrap(_ang(f, l) - want)) < 1e-9,
              f"asked {deg:.1f}, got {math.degrees(_ang(f, l)):.1f} deg")

    # ...but a reversed move still larger than one step is limited.
    wbc._base_heading = 0.0
    want = math.radians(180.0 - math.degrees(step) * 3.0)
    f, l = wbc._limit_heading_rate(0.2 * math.cos(want), 0.2 * math.sin(want))
    moved = abs(_wrap(_ang(f, l) - math.pi))
    check("a larger reversed move is still rate-limited",
          abs(moved - step) < 1e-9,
          f"moved {math.degrees(moved):.2f} of {math.degrees(step):.2f} deg")

    # A stop must freeze the reference, not reset it.
    wbc._base_heading = 1.0
    check("a zero command leaves the reference alone",
          wbc._limit_heading_rate(0.0, 0.0) == (0.0, 0.0) and wbc._base_heading == 1.0)

    # Disabling restores the old behaviour exactly.
    wbc.config.base_heading_rate_limit = 0.0
    wbc._base_heading = 0.0
    check("limit 0 disables it", wbc._limit_heading_rate(0.0, 0.2) == (0.0, 0.2))


def test_deadband_keeps_the_base_out_of_the_bad_regime() -> None:
    print("\nlinear deadband value")
    from robot.wholebody_control import WholeBodyHardwareConfig
    cfg = WholeBodyHardwareConfig()
    # The measured churn bands were 0.02-0.04 (552 deg/s median) and 0.04-0.07
    # (390). The deadband has to exclude the first one outright.
    check("the deadband excludes the worst-measured churn band",
          cfg.base_vel_deadband >= 0.04, f"{cfg.base_vel_deadband}")
    check("...but still allows most of the working range",
          cfg.base_vel_deadband < 0.25 * cfg.base_max_lin_vel,
          f"{cfg.base_vel_deadband} vs {cfg.base_max_lin_vel}")


def test_drive_scale_travels_with_the_gain_set() -> None:
    """Changing gains without changing the scale is a 2x speed error."""
    print("\ndrive command scale")
    from tools.base_pid_preflight import (
        COMMISSIONED_MANIFEST, STOCK_MANIFEST, drive_command_scale, load_manifest,
        validate_manifest,
    )
    stock, stock_note = drive_command_scale(STOCK_MANIFEST, 99.0)
    comm, comm_note = drive_command_scale(COMMISSIONED_MANIFEST, 99.0)
    check("stock declares the scale its P-only loop needs", stock == 2.0, stock_note)
    check("commissioned declares its own, near 1", 0.9 <= comm <= 1.1, comm_note)
    check("the two differ, which is the whole point", stock != comm)

    missing, note = drive_command_scale(Path("/no/such/manifest.json"), 2.0)
    check("an unreadable manifest falls back to the built-in default",
          missing == 2.0 and "default" in note, note)

    bad = dict(load_manifest(STOCK_MANIFEST)); bad["drive_command_scale"] = 40.0
    check("an absurd scale is rejected by validation",
          any("drive_command_scale" in e for e in validate_manifest(bad)))

    # And it must actually reach the motor, not just be read.
    base = bm.Base(drive_vel_scale=comm)
    base.drive_motors[0].set_velocity_mps(0.25)
    check("the scale reaches SetVelocity",
          abs(base.drive_motors[0].dev.velocity_setpoints[-1] - 0.25 * comm) < 1e-9,
          f"{base.drive_motors[0].dev.velocity_setpoints[-1]:.4f}")


def test_steering_feedback_is_closed() -> None:
    print("\nsteering feedback")
    check("the steering loop reads the encoder", bm.USE_FEEDBACK_FOR_STEER is True)

    motor = bm.RotationMotor("can0", 5, 0.25)
    motor.set_position_fraction(float(bm.rad_to_frac(0.8)))
    motor.dev.position_setpoints[-1] %= 1.0
    check("get_position_rad returns the measurement",
          abs(motor.get_position_rad() - 0.8) < 1e-9, f"{motor.get_position_rad():.4f}")

    # On encoder loss it must fall back in the SAME frame, not the raw fraction.
    motor.dev.GetAbsoluteEncoderPosition = lambda: float("nan")
    check("a dead encoder falls back to the commanded angle, offset removed",
          abs(motor.get_position_rad() - 0.8) < 1e-9, f"{motor.get_position_rad():.4f}")

    # With feedback closed, cos_error_scaling finally throttles a slewing module.
    base = bare_base()
    base.steer_pos = np.zeros(4)                       # modules point at 0 deg
    speeds, _ = base._vehicle_velocity_to_angle_and_speed(
        np.array([0.0, 0.25, 0.0]), cos_error_scaling=True)   # asked to go 90 deg away
    check("drive speed is throttled while the module is 90 deg off",
          bool(np.all(np.abs(speeds) < 1e-9)), str(np.round(speeds, 4)))
    base.steer_pos = np.full(4, np.pi / 2)             # now they have arrived
    speeds, _ = base._vehicle_velocity_to_angle_and_speed(
        np.array([0.0, 0.25, 0.0]), cos_error_scaling=True)
    check("...and full once it has arrived",
          bool(np.all(np.abs(np.abs(speeds) - 0.25) < 1e-9)), str(np.round(speeds, 4)))


def test_scurve_uses_measured_time() -> None:
    print("\nS-curve profiler")
    base = bare_base()
    base._v_prof = np.zeros(3)
    base._seg_v0 = np.zeros(3)
    base._seg_v1 = np.array([0.25, 0.0, 0.0])
    base._seg_t, base._seg_T = 0.0, 0.10

    half = base._update_scurve(0.05).copy()
    check("half a segment is half-way through the raised cosine",
          abs(half[0] - 0.125) < 1e-9, f"{half[0]:.4f}")
    done = base._update_scurve(0.05).copy()
    check("the segment completes", abs(done[0] - 0.25) < 1e-9, f"{done[0]:.4f}")

    base._seg_t, base._seg_T, base._v_prof = 0.0, 0.10, np.zeros(3)
    slow = base._update_scurve(0.10).copy()
    check("a slow tick advances further than a fast one",
          slow[0] > half[0], f"{slow[0]:.4f} vs {half[0]:.4f}")

    check("the loop measures its own dt rather than assuming the nominal one",
          "self._loop_dt" in (_REPO / "robot/base_motor.py").read_text())


def test_controller_setup_is_applied_and_reported() -> None:
    print("\ncontroller configuration")
    check("the enum import resolves", bm.IdleMode is not None and bm.CtrlType is not None,
          f"IdleMode={bm.IdleMode} CtrlType={bm.CtrlType}")

    # The real regression: base_motor.py must not import a name the installed
    # binding lacks. It does so behind a bare `except`, so a missing name does
    # not fail loudly -- it sets every enum to None and quietly disables
    # SetIdleMode and SetCtrlType.
    if REAL_SPARKCAN_EXPORTS is None:
        check("installed binding exports every name base_motor imports",
              True, "skipped: sparkcan_py not importable")
    else:
        import ast
        tree = ast.parse((_REPO / "robot/base_motor.py").read_text())
        imported = {alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module == "sparkcan_py"
                    for alias in node.names}
        missing = sorted(imported - REAL_SPARKCAN_EXPORTS)
        check("installed binding exports every name base_motor imports",
              not missing, f"missing {missing}" if missing else str(sorted(imported)))

    base = live_base()
    check("drive controllers are put in velocity mode",
          all(m.dev.ctrl_type is not None for m in base.drive_motors))
    check("idle mode is actually set on every module",
          all(m.dev.idle_mode is not None
              for m in (*base.drive_motors, *base.rotation_motors)))

    config = base.swerve_configuration()
    check("configuration is reported for all eight controllers", len(config) == 8,
          str(len(config)))
    check("the velocity conversion factor is among it",
          all(c["velocity_cf"] is not None for c in config.values()))


def test_control_loop_end_to_end() -> None:
    """Run the real control loop against fake controllers."""
    print("\ncontrol loop")
    base = live_base()
    base.start_control()
    try:
        base.set_target_base_velocity(np.array([0.0, 0.25, 0.0]), smooth=False)
        time.sleep(0.25)
        driving = base.swerve_telemetry()

        base.set_target_base_velocity(np.zeros(3), smooth=False)
        time.sleep(0.25)
        stopped = base.swerve_telemetry()
    finally:
        base.stop_control()

    check("driving commands a non-zero wheel speed",
          bool(np.all(np.abs(driving["drive_cmd_mps"]) > 1e-6)),
          str(np.round(driving["drive_cmd_mps"], 3)))
    check("driving aims the modules off straight-ahead",
          bool(np.all(np.abs(driving["steer_cmd_rad"]) > 1.0)),
          str(np.round(np.degrees(driving["steer_cmd_rad"]), 1)))
    check("stopping zeroes the wheel speed",
          bool(np.all(np.abs(stopped["drive_cmd_mps"]) < 1e-9)),
          str(np.round(stopped["drive_cmd_mps"], 6)))
    check("stopping does NOT re-aim the modules",
          bool(np.all(np.abs(bm.diff_angle(stopped["steer_cmd_rad"],
                                           driving["steer_cmd_rad"])) < 1e-9)),
          f"{np.round(np.degrees(driving['steer_cmd_rad']),1)} -> "
          f"{np.round(np.degrees(stopped['steer_cmd_rad']),1)} deg")
    check("measured steer angle tracks the command through the telemetry path",
          bool(np.all(np.abs(bm.diff_angle(stopped["steer_meas_rad"],
                                           stopped["steer_cmd_rad"])) < 1e-6)),
          str(np.round(np.degrees(stopped["steer_meas_rad"]), 1)))
    check("telemetry reports the motors as enabled while commands arrive",
          driving["motors_enabled"] is True)


def main() -> int:
    for test in (
        test_inverse_kinematics_round_trips,
        test_zero_command_holds_steering,
        test_shortest_path_flip_survives,
        test_absolute_encoder_is_read_as_turns,
        test_axis_map_is_not_crossed,
        test_linear_limits_preserve_direction,
        test_heading_rate_limit,
        test_deadband_keeps_the_base_out_of_the_bad_regime,
        test_drive_scale_travels_with_the_gain_set,
        test_steering_feedback_is_closed,
        test_scurve_uses_measured_time,
        test_controller_setup_is_applied_and_reported,
        test_control_loop_end_to_end,
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
