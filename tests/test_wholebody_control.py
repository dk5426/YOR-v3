"""
test_wholebody_control.py — headless checks for the hardware whole-body loop.

Runs robot/wholebody_control.py against fake arms and a fake base, so the
control logic (state sync → solve → dispatch → odometry) can be exercised
without CAN, serial or a robot. Nothing here touches nerolib or sparkcan_py.

    python tests/test_wholebody_control.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robot.wholebody_control import (  # noqa: E402
    BaseAxisMap, BaseOdometry, WholeBodyController, WholeBodyHardwareConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fake hardware
# ─────────────────────────────────────────────────────────────────────────────

class FakeArm:
    """A perfectly obedient 7-DOF arm: reaches each commanded target at once."""

    def __init__(self, q0):
        self.q = np.asarray(q0, dtype=float).copy()
        self.commands = 0
        self.gripper = 0.0

    def get_joint_positions(self):
        return self.q.copy()

    def set_joint_target(self, joint_target, gripper_target=None, preview_time=0.1):
        self.q = np.asarray(joint_target, dtype=float).copy()
        self.commands += 1
        if gripper_target is not None:
            self.gripper = float(gripper_target)


class FakeBase:
    """Swerve base + PicoLift stand-in.

    The lift integrates at a fixed speed while it is driven, which is what the
    bang-bang servo in the controller has to cope with on the real robot.
    """

    LIFT_SPEED = 0.05  # m/s

    def __init__(self, lift_height=0.2):
        self.lift_height = float(lift_height)
        self.lift_state = "stop"
        self.velocity_commands = []
        self._t = time.monotonic()

    # -- lift --
    def lift_up(self):
        self._integrate()
        self.lift_state = "up"

    def lift_down(self):
        self._integrate()
        self.lift_state = "down"

    def lift_stop(self):
        self._integrate()
        self.lift_state = "stop"

    def get_lift_height(self):
        self._integrate()
        return self.lift_height

    def _integrate(self):
        now = time.monotonic()
        dt, self._t = now - self._t, now
        direction = {"up": 1.0, "down": -1.0, "stop": 0.0}[self.lift_state]
        self.lift_height = float(np.clip(
            self.lift_height + direction * self.LIFT_SPEED * dt, 0.0, 0.900))

    # -- drive --
    def set_target_base_velocity(self, target, smooth=False):
        self.velocity_commands.append(np.asarray(target, dtype=float).copy())


def build(**cfg_kwargs):
    """A controller wired to fake hardware, stepped manually (no thread)."""
    left = FakeArm([0.0, 1.32, -1.71, 1.31, 0.0, 0.0, 0.0])
    right = FakeArm([0.0, 1.32, 1.71, 1.31, 0.0, 0.0, 0.0])
    base = FakeBase()
    config = WholeBodyHardwareConfig(**cfg_kwargs)
    wbc = WholeBodyController(left, right, base, config=config)
    wbc.init()
    return wbc, left, right, base


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def test_init():
    print("\ninit")
    wbc, left, right, base = build()
    T_l, T_r = wbc.forward_kinematics()
    check("targets seeded from FK", wbc.left_ee_target is not None)
    check("home poses latched", wbc._home_left is not None and wbc._home_right is not None)
    check("odometry starts at origin", np.allclose(wbc.odometry.pose, 0))
    check("lift range from description", wbc.ik.lift_range == (0.0, 0.900),
          str(wbc.ik.lift_range))
    check("model sees measured lift", abs(
        wbc.ik.configuration.q[wbc.ik._lift_qpos_adr] - base.lift_height) < 1e-9)


def test_arm_tracking():
    print("\narm tracking (EE target 10 cm forward)")
    wbc, left, right, base = build(enable_base_motion=False)
    T_l, T_r = wbc.forward_kinematics()
    goal = T_l.translation() + np.array([0.0, -0.10, 0.0])   # robot faces -Y
    import mink
    wbc.set_left_ee_target(mink.SE3.from_rotation_and_translation(T_l.rotation(), goal))

    start_err = np.linalg.norm(wbc.forward_kinematics()[0].translation() - goal)
    for _ in range(200):
        wbc._step()
    end_err = np.linalg.norm(wbc.forward_kinematics()[0].translation() - goal)

    check("arms were commanded", left.commands > 100, f"{left.commands} commands")
    check("EE converged toward target", end_err < 0.01,
          f"{start_err*100:.1f} cm -> {end_err*100:.2f} cm")
    check("right arm held still", np.linalg.norm(
        wbc.forward_kinematics()[1].translation() - T_r.translation()) < 0.02)


def test_arm_step_clamp():
    print("\nper-cycle joint step clamp")
    wbc, left, right, base = build(enable_base_motion=False, arm_max_vel_rad_s=0.5)
    max_step = 0.5 * wbc.dt
    import mink
    T_l, _ = wbc.forward_kinematics()
    far = mink.SE3.from_rotation_and_translation(
        T_l.rotation(), T_l.translation() + np.array([0.3, -0.3, 0.2]))
    wbc.set_left_ee_target(far)

    worst = 0.0
    for _ in range(50):
        before = left.q.copy()
        wbc._step()
        worst = max(worst, float(np.max(np.abs(left.q - before))))
    check("no joint moved faster than the clamp", worst <= max_step + 1e-9,
          f"worst {worst:.5f} rad vs limit {max_step:.5f}")


def test_lift_servo():
    print("\nlift servo (bang-bang against a moving lift)")
    wbc, left, right, base = build(enable_base_motion=False)
    start = base.lift_height
    wbc.set_lift_target(0.45)
    for _ in range(400):
        wbc._step()
        time.sleep(0.002)  # let the fake lift actually travel
    check("lift drove upward", base.lift_height > start + 0.05,
          f"{start:.3f} m -> {base.lift_height:.3f} m")
    check("lift command is one of up/down/stop",
          base.lift_state in ("up", "down", "stop"), base.lift_state)

    # Deadband: once the solver is satisfied the lift must be told to stop.
    wbc.set_lift_target(base.lift_height)
    for _ in range(50):
        wbc._step()
    check("lift stops inside the deadband", base.lift_state == "stop", base.lift_state)


def test_lift_without_feedback():
    print("\nlift with no height feedback")
    wbc, left, right, base = build(enable_base_motion=False)
    base.get_lift_height = lambda: None      # sensor dropout
    wbc.set_lift_target(0.6)
    before = base.lift_state
    for _ in range(20):
        wbc._step()
    check("refuses to drive a bang-bang lift blind", base.lift_state == before,
          f"state stayed {base.lift_state}")


def test_base_motion():
    print("\nbase motion (target out of arm reach)")
    wbc, left, right, base = build()
    import mink
    T_l, _ = wbc.forward_kinematics()
    unreachable = mink.SE3.from_rotation_and_translation(
        T_l.rotation(), T_l.translation() + np.array([0.0, -1.5, 0.0]))
    wbc.set_left_ee_target(unreachable)
    for _ in range(100):
        wbc._step()

    commands = np.array(base.velocity_commands) if base.velocity_commands else np.zeros((1, 3))
    peak = np.abs(commands).max(axis=0)
    check("base was commanded", bool(np.any(np.abs(commands) > 1e-6)), f"peak {peak.round(3)}")
    check("linear command within clamp",
          peak[0] <= wbc.config.base_max_lin_vel + 1e-9 and peak[1] <= wbc.config.base_max_lin_vel + 1e-9,
          f"{peak[:2].round(3)} vs {wbc.config.base_max_lin_vel}")
    check("angular command within clamp", peak[2] <= wbc.config.base_max_ang_vel + 1e-9,
          f"{peak[2]:.3f} vs {wbc.config.base_max_ang_vel}")
    check("odometry advanced", np.linalg.norm(wbc.odometry.pose[:2]) > 1e-3,
          f"pose {wbc.odometry.pose.round(3)}")


def test_fix_base_and_toggles():
    print("\nfix_base / base-motion toggles")
    wbc, left, right, base = build()
    import mink
    T_l, _ = wbc.forward_kinematics()
    unreachable = mink.SE3.from_rotation_and_translation(
        T_l.rotation(), T_l.translation() + np.array([0.0, -1.5, 0.0]))

    wbc.toggle_fix_base(True)
    wbc.set_left_ee_target(unreachable)
    base.velocity_commands.clear()
    for _ in range(50):
        wbc._step()
    nonzero = [c for c in base.velocity_commands if np.any(np.abs(c) > 1e-9)]
    check("fix_base suppresses base commands", not nonzero, f"{len(nonzero)} non-zero")

    wbc.toggle_fix_base(False)
    check("toggle_collision_avoidance flips", wbc.toggle_collision_avoidance() is False)
    check("toggle_collision_avoidance restores", wbc.toggle_collision_avoidance() is True)
    check("toggle_base_motion off", wbc.toggle_base_motion(False) is False)
    check("toggle_base_motion on", wbc.toggle_base_motion(True) is True)


def test_manual_override():
    print("\nmanual override arbitration")
    wbc, left, right, base = build()
    import mink
    T_l, _ = wbc.forward_kinematics()
    wbc.set_left_ee_target(mink.SE3.from_rotation_and_translation(
        T_l.rotation(), T_l.translation() + np.array([0.0, -1.5, 0.0])))

    wbc.notify_manual_base_command()
    base.velocity_commands.clear()
    for _ in range(20):
        wbc._step()
    nonzero = [c for c in base.velocity_commands if np.any(np.abs(c) > 1e-9)]
    check("base yields to a manual drive command", not nonzero, f"{len(nonzero)} non-zero")

    wbc.notify_manual_arm_command()
    commands_before = left.commands
    for _ in range(20):
        wbc._step()
    check("arms yield to a manual joint command", left.commands == commands_before)

    wbc._manual_base_until = 0.0    # expire the window
    wbc._manual_arm_until = 0.0
    base.velocity_commands.clear()
    for _ in range(20):
        wbc._step()
    check("authority returns after the window", left.commands > commands_before)


def test_axis_map_and_odometry():
    print("\nbase frame conventions")
    m = BaseAxisMap()
    cmd = m.to_command(forward=1.0, lateral=2.0, yaw_rate=3.0)
    check("axis map places forward/lateral/yaw", np.allclose(cmd, [2.0, 1.0, 3.0]), str(cmd))

    wbc, *_ = build()
    # The robot faces -Y, so world -Y velocity must read as pure forward.
    fwd, lat, yaw = wbc._world_to_body(np.array([0.0, -0.5, 0.0]))
    check("world -Y is forward", abs(fwd - 0.5) < 1e-9 and abs(lat) < 1e-9,
          f"fwd={fwd:.3f} lat={lat:.3f}")
    # …and world +X is to the left.
    fwd, lat, yaw = wbc._world_to_body(np.array([0.4, 0.0, 0.0]))
    check("world +X is left", abs(lat - 0.4) < 1e-9 and abs(fwd) < 1e-9,
          f"fwd={fwd:.3f} lat={lat:.3f}")
    # Round-tripping must be lossless, or odometry would fight the command.
    rt = wbc._body_to_world(*wbc._world_to_body(np.array([0.3, -0.2, 0.1])))
    check("body/world round-trip is exact", np.allclose(rt, [0.3, -0.2, 0.1]), str(rt.round(4)))

    odo = BaseOdometry()
    odo.update(np.array([1.0, 0.0, 0.0]), 0.5)
    check("odometry integrates", np.allclose(odo.pose, [0.5, 0.0, 0.0]), str(odo.pose))


def test_state_snapshot():
    print("\nRPC state snapshot")
    wbc, *_ = build()
    state = wbc.get_state()
    required = {
        "left_ee_wxyz_xyz", "right_ee_wxyz_xyz", "lift", "base_xytheta",
        "base_velocity", "fix_base", "collision_avoidance",
    }
    missing = required - set(state)
    check("carries every key the teleop client reads", not missing, str(missing))
    check("values are plain types", all(
        isinstance(state[k], (list, float, int, bool, str, type(None)))
        for k in state), str({k: type(v).__name__ for k, v in state.items()}))


def main() -> int:
    for test in (
        test_init,
        test_arm_tracking,
        test_arm_step_clamp,
        test_lift_servo,
        test_lift_without_feedback,
        test_base_motion,
        test_fix_base_and_toggles,
        test_manual_override,
        test_axis_map_and_odometry,
        test_state_snapshot,
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
