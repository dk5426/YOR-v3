"""
test_base_pose_control.py — the continuous base-pose PD in robot/base.py.

`BasePoseController` is what turns the whole-body solver's base *pose* target
into a swerve velocity command. It is pure control: no odometry, no SLAM, no
hardware, so it can be exercised exactly as the robot runs it.

What is worth pinning here is the behaviour the pose loop exists for, and the
behaviour that would quietly break the robot if it drifted:

  * exact zeros inside both deadbands, the linear one measured on the error
    *vector* rather than per axis
  * a yaw error that takes the short way round
  * clamps that scale the command without turning it
  * damping taken from the measurement, and forgotten on every discontinuity
  * a standing pose error keeps commanding motion — the whole reason the base
    is driven by pose instead of by the solver's per-tick velocity

    python tests/test_base_pose_control.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robot.base import BasePoseController, _wrap_pi   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fakes and helpers
# ─────────────────────────────────────────────────────────────────────────────

class FakeBase:
    """Records what `Base` would have been told to drive."""

    def __init__(self, max_vel=(1.0, 1.0, 1.57)):
        self.max_vel = np.asarray(max_vel, dtype=float)
        self.commands: list[np.ndarray] = []
        self.smooth: list[bool] = []

    def set_target_base_velocity(self, target, smooth=False):
        self.commands.append(np.asarray(target, dtype=float).copy())
        self.smooth.append(bool(smooth))


def body_to_world(forward: float, lateral: float, yaw: float) -> tuple[float, float]:
    """(forward, left) at `yaw` → world (vx, vy), for the default frame.

    The description has the robot facing −Y with +X to its left, so the
    forward axis is (sin yaw, −cos yaw) and the left axis is (cos yaw, sin
    yaw). Written out longhand rather than reusing the controller's own
    rotation, so a sign flip in `_to_body` cannot pass by cancelling itself.
    """
    vx = forward * math.sin(yaw) + lateral * math.cos(yaw)
    vy = -forward * math.cos(yaw) + lateral * math.sin(yaw)
    return vx, vy


def drive(controller, target, pose, ticks: int, dt: float = 1.0 / 30.0):
    """Close the loop against a base that executes exactly what it is told."""
    pose = np.asarray(pose, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    last = np.zeros(3)
    for _ in range(ticks):
        last = controller.compute(target, pose, dt=dt)
        vx, vy = body_to_world(last[0], last[1], float(pose[2]))
        pose[0] += vx * dt
        pose[1] += vy * dt
        pose[2] = _wrap_pi(pose[2] + last[2] * dt)
    return pose, last


RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def test_frame_decomposition():
    print("\npose error resolves into the chassis frame")

    # Facing −Y at yaw 0: a target one metre along −Y is straight ahead.
    cmd = BasePoseController().compute([0.0, -1.0, 0.0], [0.0, 0.0, 0.0], dt=1 / 30)
    check("a target ahead is pure forward",
          cmd[0] > 0.0 and abs(cmd[1]) < 1e-9 and cmd[2] == 0.0, str(cmd.round(4)))

    # +X is the robot's left.
    cmd = BasePoseController().compute([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], dt=1 / 30)
    check("a target to +X is pure left",
          cmd[1] > 0.0 and abs(cmd[0]) < 1e-9, str(cmd.round(4)))

    # Turn the robot a quarter turn and the same world offset changes meaning.
    cmd = BasePoseController().compute(
        [1.0, 0.0, math.pi / 2], [0.0, 0.0, math.pi / 2], dt=1 / 30)
    check("the same world offset is forward once the robot has turned",
          cmd[0] > 0.0 and abs(cmd[1]) < 1e-9, str(cmd.round(4)))

    # An arbitrary yaw: the decomposition must invert exactly.
    yaw, err = 0.7, np.array([0.13, -0.09])
    cmd = BasePoseController(kp_xy=1.0, kd_xy=0.0, max_lin_vel=10.0).compute(
        [err[0], err[1], yaw], [0.0, 0.0, yaw], dt=1 / 30)
    vx, vy = body_to_world(cmd[0], cmd[1], yaw)
    check("forward/left rotate back onto the world error",
          abs(vx - err[0]) < 1e-9 and abs(vy - err[1]) < 1e-9,
          f"{(vx, vy)} vs {tuple(err)}")

    # Distance is a rotation invariant: the command magnitude may not depend
    # on which way the robot happens to be pointing.
    mags = []
    for yaw in np.linspace(-math.pi, math.pi, 9):
        c = BasePoseController(kp_xy=1.0, kd_xy=0.0, max_lin_vel=10.0).compute(
            [0.3, 0.4, yaw], [0.0, 0.0, yaw], dt=1 / 30)
        mags.append(math.hypot(c[0], c[1]))
    check("command magnitude is independent of heading",
          max(mags) - min(mags) < 1e-9 and abs(mags[0] - 0.5) < 1e-9,
          f"{min(mags):.6f}..{max(mags):.6f}")


def test_deadbands():
    print("\ndeadbands: exactly zero, and the linear one is a vector")

    ctrl = BasePoseController(xy_deadband=0.01, yaw_deadband=0.02)

    cmd = ctrl.compute([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], dt=1 / 30)
    check("no error commands exactly nothing",
          cmd[0] == 0.0 and cmd[1] == 0.0 and cmd[2] == 0.0, str(cmd))

    cmd = BasePoseController().compute([0.004, -0.003, 0.0], [0.0, 0.0, 0.0], dt=1 / 30)
    check("a sub-deadband offset is exactly zero, not merely small",
          cmd[0] == 0.0 and cmd[1] == 0.0, str(cmd))

    # 8 mm on each axis is inside a *per-axis* 10 mm band but outside the
    # vector one — and the direction that comes out must still be the 45° the
    # error asked for, which is what deadbanding per axis would have destroyed.
    cmd = BasePoseController(kd_xy=0.0).compute(
        [0.008, -0.008, 0.0], [0.0, 0.0, 0.0], dt=1 / 30)
    check("the linear deadband is measured on the error vector",
          math.hypot(cmd[0], cmd[1]) > 0.0, str(cmd.round(5)))
    check("and the direction it asked for survives it",
          abs(cmd[0] - cmd[1]) < 1e-9 and cmd[0] > 0.0,
          f"fwd={cmd[0]:.6f} left={cmd[1]:.6f}")

    # The two bands are independent: neither may silence the other axis.
    cmd = BasePoseController().compute([0.0, 0.0, 0.5], [0.0, 0.0, 0.0], dt=1 / 30)
    check("a yaw-only error rotates and does not translate",
          cmd[0] == 0.0 and cmd[1] == 0.0 and cmd[2] > 0.0, str(cmd.round(4)))

    cmd = BasePoseController().compute([0.0, -0.5, 0.0], [0.0, 0.0, 0.0], dt=1 / 30)
    check("a translation-only error drives and does not rotate",
          cmd[2] == 0.0 and cmd[0] > 0.0, str(cmd.round(4)))

    # Deadbands act on the error, so they hold however large the gains are.
    hot = BasePoseController(kp_xy=50.0, kp_yaw=50.0)
    cmd = hot.compute([0.005, 0.0, 0.01], [0.0, 0.0, 0.0], dt=1 / 30)
    check("a large gain does not push a sub-deadband error out of the band",
          cmd[0] == 0.0 and cmd[1] == 0.0 and cmd[2] == 0.0, str(cmd))


def test_yaw_wrapping():
    print("\nyaw error takes the short way round")

    # 3.0 → −3.0 rad is +0.283 rad the short way, not −6.0.
    cmd = BasePoseController().compute([0.0, 0.0, -3.0], [0.0, 0.0, 3.0], dt=1 / 30)
    expected = _wrap_pi(-3.0 - 3.0)
    check("a wrap across ±pi turns the short way",
          cmd[2] > 0.0 and abs(cmd[2] - 2.0 * expected) < 1e-9,
          f"{cmd[2]:.4f} for a {expected:.4f} rad error")

    cmd = BasePoseController().compute([0.0, 0.0, 3.0], [0.0, 0.0, -3.0], dt=1 / 30)
    check("and the other way round is the mirror image", cmd[2] < 0.0, f"{cmd[2]:.4f}")

    # Half a turn is the boundary; it must not come out as a full turn.
    cmd = BasePoseController(max_ang_vel=100.0).compute(
        [0.0, 0.0, 0.0], [0.0, 0.0, math.pi - 0.01], dt=1 / 30)
    check("an almost-half turn stays within pi of the target",
          abs(cmd[2]) <= 2.0 * math.pi + 1e-9 and abs(cmd[2] / 2.0) <= math.pi,
          f"{cmd[2]:.4f}")

    # The reported error is wrapped too, not just the command.
    ctrl = BasePoseController()
    ctrl.compute([0.0, 0.0, -3.0], [0.0, 0.0, 3.0], dt=1 / 30)
    check("the logged error is the wrapped one",
          abs(ctrl.last_error[2] - expected) < 1e-9, f"{ctrl.last_error[2]:.4f}")


def test_limits():
    print("\nvelocity limits")

    ctrl = BasePoseController(max_lin_vel=0.25, max_ang_vel=0.60, kd_xy=0.0, kd_yaw=0.0)
    cmd = ctrl.compute([3.0, -4.0, 0.0], [0.0, 0.0, 0.0], dt=1 / 30)
    speed = math.hypot(cmd[0], cmd[1])
    check("a large error saturates at the linear limit",
          abs(speed - 0.25) < 1e-12, f"{speed:.6f}")
    # 3:−4 in world is a fixed direction in the body frame too; clamping as a
    # vector keeps that ratio, clamping per axis would not.
    raw = ctrl._to_body(3.0, -4.0, 0.0)
    check("the clamp scales the command without turning it",
          abs(cmd[0] / speed - raw[0] / 5.0) < 1e-9
          and abs(cmd[1] / speed - raw[1] / 5.0) < 1e-9,
          str(cmd.round(4)))

    cmd = ctrl.compute([0.0, 0.0, 2.0], [0.0, 0.0, 0.0], dt=1 / 30)
    check("a large yaw error saturates at the angular limit",
          abs(cmd[2] - 0.60) < 1e-12, f"{cmd[2]:.6f}")

    # The drive's own ceiling wins when it is the lower of the two: Base takes
    # max_vel at construction and does not enforce it in its control loop, so
    # a controller that ignored it would simply command past it.
    slow = BasePoseController(FakeBase(max_vel=(0.1, 0.1, 0.2)),
                              max_lin_vel=0.25, max_ang_vel=0.60, kd_xy=0.0)
    cmd = slow.compute([3.0, -4.0, 2.0], [0.0, 0.0, 0.0], dt=1 / 30)
    check("the base's own max_vel lowers the limits",
          abs(math.hypot(cmd[0], cmd[1]) - 0.1) < 1e-12 and abs(cmd[2] - 0.2) < 1e-12,
          f"{math.hypot(cmd[0], cmd[1]):.4f}, {cmd[2]:.4f}")

    fast = BasePoseController(FakeBase(max_vel=(5.0, 5.0, 5.0)),
                              max_lin_vel=0.25, max_ang_vel=0.60)
    check("a permissive drive does not raise them",
          fast.max_lin_vel == 0.25 and fast.max_ang_vel == 0.60,
          f"{fast.max_lin_vel}, {fast.max_ang_vel}")


def test_damping():
    print("\ndamping comes from the measurement")

    dt = 1 / 30

    # Standing still at a 0.1 m error versus closing on it at 0.2 m/s: the
    # second must ask for less, because the D term sees the approach.
    still = BasePoseController(max_lin_vel=10.0)
    still.compute([0.0, -0.5, 0.0], [0.0, -0.4, 0.0], dt=dt)
    stationary = still.compute([0.0, -0.5, 0.0], [0.0, -0.4, 0.0], dt=dt)

    # The robot faces −Y, so approaching a target at −0.5 means y falling.
    moving = BasePoseController(max_lin_vel=10.0)
    moving.compute([0.0, -0.5, 0.0], [0.0, -0.4 + 0.2 * dt, 0.0], dt=dt)
    approaching = moving.compute([0.0, -0.5, 0.0], [0.0, -0.4, 0.0], dt=dt)

    check("closing on the target damps the command",
          approaching[0] < stationary[0], f"{approaching[0]:.4f} < {stationary[0]:.4f}")
    check("and a stationary base is not damped at all",
          abs(stationary[0] - 1.5 * 0.1) < 1e-9, f"{stationary[0]:.4f}")

    # A target that jumps must not: the derivative never looks at the target.
    jumped = BasePoseController(max_lin_vel=10.0)
    jumped.compute([0.0, -0.4, 0.0], [0.0, -0.4, 0.0], dt=dt)
    after = jumped.compute([0.0, -1.4, 0.0], [0.0, -0.4, 0.0], dt=dt)
    check("a one-metre target step produces no derivative kick",
          abs(after[0] - 1.5 * 1.0) < 1e-9, f"{after[0]:.4f}")

    # A stalled loop is not a velocity measurement.
    stalled = BasePoseController(max_lin_vel=10.0)
    stalled.compute([0.0, -0.5, 0.0], [0.0, 0.0, 0.0], dt=dt)
    stalled.compute([0.0, -0.5, 0.0], [0.0, -0.3, 0.0], dt=5.0)
    check("a gap longer than max_gap_s drops the derivative",
          np.allclose(stalled.measured_velocity, 0.0), str(stalled.measured_velocity))

    # …and neither is a pose that changed while we were not driving.
    live = BasePoseController(max_lin_vel=10.0)
    live.compute([0.0, -0.5, 0.0], [0.0, 0.0, 0.0], dt=dt)
    live.compute([0.0, -0.5, 0.0], [0.0, -0.2, 0.0], dt=dt)
    check("motion while we were driving is remembered",
          not np.allclose(live.measured_velocity, 0.0), str(live.measured_velocity.round(3)))
    live.reset()
    check("reset forgets it", np.allclose(live.measured_velocity, 0.0)
          and live._last_pose is None, str(live.measured_velocity))
    resumed = live.compute([0.0, -0.5, 0.0], [0.0, -0.2, 0.0], dt=dt)
    check("the cycle after a reset is undamped",
          abs(resumed[0] - 1.5 * 0.3) < 1e-9, f"{resumed[0]:.4f}")


def test_convergence():
    print("\nclosed loop against a base that does as it is told")

    ctrl = BasePoseController()
    target = np.array([0.4, -0.6, 0.5])
    pose, last = drive(ctrl, target, np.zeros(3), ticks=600)

    lin_err = math.hypot(target[0] - pose[0], target[1] - pose[1])
    yaw_err = abs(_wrap_pi(target[2] - pose[2]))
    check("the base arrives inside both deadbands",
          lin_err <= ctrl.xy_deadband and yaw_err <= ctrl.yaw_deadband,
          f"lin {lin_err*1e3:.1f} mm, yaw {math.degrees(yaw_err):.2f} deg")
    check("and then commands exactly zero", np.all(last == 0.0), str(last))

    # It converges from every direction, including one that needs the wrap.
    worst = 0.0
    for tgt in ([-0.3, 0.2, -2.9], [0.0, 0.5, 3.0], [0.25, 0.25, 0.0]):
        pose, _ = drive(BasePoseController(), tgt, [0.0, 0.0, 3.0], ticks=900)
        worst = max(worst, math.hypot(tgt[0] - pose[0], tgt[1] - pose[1]))
    check("from any start, including one that wraps", worst <= 0.01,
          f"worst {worst*1e3:.1f} mm")

    # The point of pose control: a target the base has not reached keeps
    # commanding motion for as long as the error stands, however long the
    # solver's own per-tick velocity has been zero.
    stalled = BasePoseController()
    cmds = [stalled.compute([0.0, -0.5, 0.0], [0.0, 0.0, 0.0], dt=1 / 30)[0]
            for _ in range(50)]
    check("a standing error keeps asking, tick after tick",
          all(c > 0.0 for c in cmds) and abs(cmds[-1] - cmds[0]) < 1e-9,
          f"{cmds[0]:.3f} … {cmds[-1]:.3f}")


def test_dispatch():
    print("\nwhat reaches the drive")

    base = FakeBase()
    ctrl = BasePoseController(base, kd_xy=0.0)
    cmd = ctrl.step([0.0, -1.0, 0.3], [0.0, 0.0, 0.0], dt=1 / 30)

    check("step sends exactly one command", len(base.commands) == 1)
    check("it sends [forward, left, yaw] as computed",
          np.allclose(base.commands[-1], cmd), f"{base.commands[-1]} vs {cmd}")
    check("smoothing is left to the drive's own ramp", base.smooth[-1] is True)
    check("the command is forward and turning",
          cmd[0] > 0.0 and cmd[2] > 0.0, str(cmd.round(3)))

    ctrl.halt()
    check("halt stops the base", np.allclose(base.commands[-1], 0.0),
          str(base.commands[-1]))
    check("halt forgets the PD history", ctrl._last_pose is None)

    # A controller with no drive is still a usable calculator, but must say so
    # rather than silently dropping the command.
    detached = BasePoseController()
    try:
        detached.send(np.zeros(3))
        raised = False
    except RuntimeError:
        raised = True
    check("a detached controller refuses to send", raised)
    detached.halt()   # must not raise
    check("but halting one is harmless", True)

    for bad in ([0.0, 1.0], 3.0):
        try:
            BasePoseController().compute(bad, [0.0, 0.0, 0.0], dt=1 / 30)
            ok = False
        except (ValueError, TypeError):
            ok = True
        check(f"a malformed pose is rejected ({bad})", ok)


def main() -> int:
    for test in (
        test_frame_decomposition,
        test_deadbands,
        test_yaw_wrapping,
        test_limits,
        test_damping,
        test_convergence,
        test_dispatch,
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
