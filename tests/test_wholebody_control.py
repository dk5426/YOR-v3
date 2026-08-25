"""
test_wholebody_control.py — headless checks for the hardware whole-body loop.

Runs robot/wholebody_control.py against fake arms and a fake base, so the
control logic (state sync → solve → dispatch → odometry) can be exercised
without CAN, serial or a robot. Nothing here touches nerolib or sparkcan_py.

    python tests/test_wholebody_control.py
"""

from __future__ import annotations

import math
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
        self.reads = 0
        self.gripper = 0.0

    def get_joint_positions(self):
        self.reads += 1
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
    # Headless tests have no business writing real files into
    # artifacts/wholebody_logs/trajectories/ -- that's for live robot runs.
    cfg_kwargs.setdefault("record_trajectories", False)
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
        # Arm dispatch runs on its own decoupled loop in the real system (see
        # _arm_dispatch_tick); flush one full interpolated segment per solve
        # tick here so the fake arm actually receives commands.
        for _ in range(wbc.config.arm_interpolation_steps):
            wbc._arm_dispatch_tick()
    end_err = np.linalg.norm(wbc.forward_kinematics()[0].translation() - goal)

    check("arm commands crossed the deadband", left.commands > 0,
          f"{left.commands} commands")
    check("EE converged toward target", end_err < 0.01,
          f"{start_err*100:.1f} cm -> {end_err*100:.2f} cm")
    check("right arm held still", np.linalg.norm(
        wbc.forward_kinematics()[1].translation() - T_r.translation()) < 0.02)


def test_arm_command_lookahead():
    print("\nbounded arm command look-ahead")
    wbc, left, right, base = build(
        enable_base_motion=False,
        arm_max_vel_rad_s=0.5,
        arm_command_lookahead_s=0.10,
    )
    max_lead = 0.5 * 0.10
    import mink
    T_l, _ = wbc.forward_kinematics()
    far = mink.SE3.from_rotation_and_translation(
        T_l.rotation(), T_l.translation() + np.array([0.3, -0.3, 0.2]))
    wbc.set_left_ee_target(far)

    worst = 0.0
    for _ in range(50):
        before = left.q.copy()
        wbc._step()
        for _ in range(wbc.config.arm_interpolation_steps):
            wbc._arm_dispatch_tick()
        worst = max(worst, float(np.max(np.abs(left.q - before))))
    check("command never exceeds the bounded look-ahead", worst <= max_lead + 1e-9,
          f"worst {worst:.5f} rad vs limit {max_lead:.5f}")

    # Open-loop WBC is seeded once during init and must not read encoders during
    # ordinary control ticks.
    left.reads = right.reads = 0
    wbc._step()
    check("open-loop WBC performs no arm reads during a control tick",
          left.reads == 0 and right.reads == 0,
          f"left={left.reads} right={right.reads}")

    # The low-level fake can report a different physical state, but open-loop
    # dispatch must continue from its last command rather than snapping its
    # reference to that feedback.
    commanded_before = wbc._commanded_left_q.copy()
    left.q += 0.25
    left.reads = 0
    wbc._step()
    check("open-loop command reference ignores changed encoder feedback",
          left.reads == 0 and np.max(np.abs(
              wbc._commanded_left_q - commanded_before)) <= max_lead + 1e-9,
          f"reads={left.reads}")


def test_arm_joint_deadband():
    print("\narm joint deadband")
    wbc, left, right, base = build(arm_joint_deadband_rad=0.05)
    from types import SimpleNamespace

    left_q = wbc._commanded_left_q.copy()
    right_q = wbc._commanded_right_q.copy()
    before_left = left.commands
    before_right = right.commands

    # Exactly 0.05 rad is inside the band, matching resetToHome's `> 0.05`
    # test. No low-level command should be emitted for either arm.
    wbc._dispatch_arms(SimpleNamespace(
        left_arm_q=left_q + 0.05,
        right_arm_q=right_q - 0.049,
    ))
    check("50 mrad or less holds the previous arm command",
          left.commands == before_left and right.commands == before_right)

    target = left_q.copy()
    target[0] += 0.051
    target[1] += 0.020
    wbc._dispatch_arms(SimpleNamespace(
        left_arm_q=target,
        right_arm_q=right_q,
    ))
    # _dispatch_arms only stages the goal now; one dispatch tick sends the
    # first (and here, only necessary) interpolated sub-step.
    wbc._arm_dispatch_tick()
    moved = wbc._commanded_left_q - left_q
    check("a joint beyond 50 mrad is dispatched", left.commands == before_left + 1,
          f"joint 1 delta={moved[0]:.3f} rad")
    check("deadband is applied independently per joint",
          abs(moved[0] - 0.051) < 1e-12 and abs(moved[1]) < 1e-12,
          str(moved[:2]))


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
        for _ in range(wbc.config.arm_interpolation_steps):
            wbc._arm_dispatch_tick()
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


def test_yaw_filter_and_hysteresis():
    """The yaw dispatch path: filter + hysteresis deadband, added 2026-08-24.

    The failure this guards against was measured, not hypothetical: solver yaw
    noise straddling a single 0.02 threshold sent one-tick rotation pulses at
    7.4/s, each of which re-aimed all four swerve modules. The contract is:
    single-tick pulses die in the filter, sustained rotation passes, a request
    hovering between the exit and entry thresholds does not toggle, and losing
    base authority resets the filter state.

    The deadband defaults to 0/0 (disabled) since 2026-08-25 -- with the
    filter carrying noise rejection, the deadband caused start-stop bursts
    on the floor. The mechanism stays (restorable via --base-yaw-deadband),
    so this test pins explicit thresholds rather than the defaults.
    """
    print("\nyaw filter and hysteresis deadband")
    wbc, left, right, base = build()
    cfg = wbc.config
    cfg.base_yaw_deadband, cfg.base_yaw_deadband_exit = 0.05, 0.025

    check("the deadband is disabled by default",
          type(cfg).base_yaw_deadband == 0.0
          and type(cfg).base_yaw_deadband_exit == 0.0)

    # A single-tick pulse at the old chatter magnitude produces no rotation.
    wbc._yaw_filt, wbc._yaw_active = 0.0, False
    out = [wbc._filter_yaw(0.03)] + [wbc._filter_yaw(0.0) for _ in range(30)]
    check("a single-tick pulse never leaves the deadband",
          all(v == 0.0 for v in out), f"max {max(np.abs(out)):.4f}")

    # A sustained real rotation passes, at close to its asked rate.
    wbc._yaw_filt, wbc._yaw_active = 0.0, False
    for _ in range(60):
        last = wbc._filter_yaw(0.3)
    check("a sustained request passes the deadband", last != 0.0, f"{last:.3f}")
    check("and converges to the asked rate", abs(last - 0.3) < 0.01, f"{last:.3f}")

    # Hovering inside the hysteresis gap must not toggle: once rotating at a
    # rate between exit and entry, rotation continues; once stopped, the same
    # rate never starts it.
    gap = 0.5 * (cfg.base_yaw_deadband + cfg.base_yaw_deadband_exit)
    still_on = [wbc._filter_yaw(gap) for _ in range(30)]
    check("a request inside the gap keeps an active rotation alive",
          all(v != 0.0 for v in still_on))
    wbc._yaw_filt, wbc._yaw_active = 0.0, False
    stay_off = [wbc._filter_yaw(gap * 0.999) for _ in range(5)]
    # the filter converges toward `gap`, which is below entry — never activates
    stay_off += [wbc._filter_yaw(gap * 0.999) for _ in range(60)]
    check("the same request from rest never starts one",
          all(v == 0.0 for v in stay_off))

    # A one-tick solver spike is clamped before it charges the filter.
    wbc._yaw_filt, wbc._yaw_active = 0.0, False
    first = wbc._filter_yaw(12.0)
    check("a 12 rad/s spike charges the filter with at most one clamped tick",
          abs(wbc._yaw_filt) <= cfg.base_max_ang_vel * (wbc.dt / (cfg.base_yaw_filter_tau + wbc.dt)) + 1e-9,
          f"filt {wbc._yaw_filt:.4f}")

    # Losing base authority forgets the filter state.
    wbc._yaw_filt, wbc._yaw_active = 0.4, True
    wbc.toggle_fix_base(True)
    class _R: base_position = np.zeros(3)
    wbc._dispatch_base(_R())
    check("losing base authority resets the yaw filter",
          wbc._yaw_filt == 0.0 and not wbc._yaw_active,
          f"filt {wbc._yaw_filt}, active {wbc._yaw_active}")
    wbc.toggle_fix_base(False)


def test_base_pose_dispatch():
    """The base is driven by the solver's *pose*, not its per-tick velocity.

    Everything here is about the property the velocity path could not have:
    an error that the wheels have not worked off yet is still an error on the
    next cycle. Under the old dispatch a tick whose velocity was clamped,
    deadbanded or filtered away simply lost that motion, because the next
    solve's velocity said nothing about the shortfall.
    """
    print("\nbase pose dispatch")
    wbc, left, right, base = build()

    class _R:
        # Where the solver wants the chassis: 0.5 m ahead (the robot faces -Y)
        # and a quarter turn round. Deliberately carries a zero base_velocity,
        # which under the old dispatch would have meant "do not move".
        base_position = np.array([0.0, -0.5, 0.4])
        base_velocity = np.zeros(3)

    wbc.odometry.reset(np.zeros(3))
    wbc.base_pose.reset()
    base.velocity_commands.clear()
    wbc._dispatch_base(_R())
    first = base.velocity_commands[-1]
    check("a pose error drives the base even at zero solver velocity",
          first[0] > 0.0 and first[2] > 0.0, str(first.round(4)))
    check("the dispatch telemetry carries the target and the error",
          np.allclose(wbc._base_dispatch["target"], _R.base_position)
          and wbc._base_dispatch["err"][0] > 0.0,
          str(wbc._base_dispatch["err"].round(4)))

    # The pose controller must resolve the error in the same frame the rest of
    # this file uses; a disagreement would drive the base sideways.
    for yaw in (0.0, 0.6, -2.4):
        wbc.odometry.reset(np.array([0.0, 0.0, yaw]))
        world = np.array([0.3, -0.2])
        fwd, lat, _ = wbc._world_to_body(np.array([world[0], world[1], 0.0]))
        err = wbc.base_pose._to_body(world[0], world[1], yaw)
        agree = abs(err[0] - fwd) < 1e-9 and abs(err[1] - lat) < 1e-9
        check(f"pose error and _world_to_body agree at yaw {yaw}", agree,
              f"{np.round(err, 4)} vs {(round(fwd, 4), round(lat, 4))}")

    # Closed loop: odometry integrates what was commanded, so repeated
    # dispatch against a standing target has to converge onto it.
    wbc.odometry.reset(np.zeros(3))
    wbc.base_pose.reset()
    base.velocity_commands.clear()
    for _ in range(400):
        wbc._dispatch_base(_R())
    pose = wbc.odometry.pose
    lin_err = float(np.hypot(*(_R.base_position[:2] - pose[:2])))
    yaw_err = abs(float(pose[2] - _R.base_position[2]))
    check("the base converges onto the pose the solver asked for",
          lin_err < 0.02 and yaw_err < 0.02,
          f"lin {lin_err*1e3:.1f} mm, yaw {math.degrees(yaw_err):.2f} deg")
    check("and settles instead of hunting",
          np.max(np.abs(base.velocity_commands[-1])) < 0.01,
          str(base.velocity_commands[-1].round(4)))

    # Losing authority resets the PD: while the base is not ours the measured
    # pose can move without us, and the first cycle back must not damp against
    # motion it never commanded.
    for name, take_authority, give_back in (
        ("fix_base", lambda: wbc.toggle_fix_base(True), lambda: wbc.toggle_fix_base(False)),
        ("a disabled base", lambda: wbc.toggle_base_motion(False),
         lambda: wbc.toggle_base_motion(True)),
        ("a manual override", wbc.notify_manual_base_command,
         lambda: setattr(wbc, "_manual_base_until", 0.0)),
    ):
        wbc.odometry.reset(np.zeros(3))
        wbc._dispatch_base(_R())          # give the PD some history
        take_authority()
        wbc._dispatch_base(_R())
        check(f"{name} resets the pose PD",
              wbc.base_pose._last_pose is None
              and np.allclose(wbc.base_pose.measured_velocity, 0.0)
              and not wbc._base_dispatch["active"])
        give_back()

    stopped, *_ = build(enable_base_motion=True)
    stopped._dispatch_base(_R())
    stopped.emergency_stop()
    check("an e-stop resets the pose PD too", stopped.base_pose._last_pose is None)


def test_fix_base_and_toggles():
    print("\nfix_base / base-motion toggles")
    wbc, left, right, base = build()
    import mink
    T_l, _ = wbc.forward_kinematics()
    unreachable = mink.SE3.from_rotation_and_translation(
        T_l.rotation(), T_l.translation() + np.array([0.0, -1.5, 0.0]))

    wbc.toggle_fix_base(True)
    base_model_before = wbc.ik.configuration.q[wbc.ik.base_qpos_adrs].copy()
    wbc.set_left_ee_target(unreachable)
    base.velocity_commands.clear()
    for _ in range(50):
        wbc._step()
    nonzero = [c for c in base.velocity_commands if np.any(np.abs(c) > 1e-9)]
    check("fix_base suppresses base commands", not nonzero, f"{len(nonzero)} non-zero")
    check("fix_base suppresses virtual base motion", np.allclose(
        wbc.ik.configuration.q[wbc.ik.base_qpos_adrs], base_model_before),
        str(wbc.ik.configuration.q[wbc.ik.base_qpos_adrs]))

    wbc.toggle_fix_base(False)
    check("toggle_collision_avoidance flips", wbc.toggle_collision_avoidance() is False)
    check("toggle_collision_avoidance restores", wbc.toggle_collision_avoidance() is True)
    check("toggle_base_motion off", wbc.toggle_base_motion(False) is False)
    check("toggle_base_motion on", wbc.toggle_base_motion(True) is True)

    lift_wbc, *_ = build(enable_base_motion=False, enable_lift_motion=False)
    lift_before = float(
        lift_wbc.ik.configuration.q[lift_wbc.ik._lift_qpos_adr]
    )
    check("toggle_fix_lift on", lift_wbc.ik.toggle_fix_lift(True) is True)
    lift_left, _ = lift_wbc.forward_kinematics()
    lift_goal = mink.SE3.from_rotation_and_translation(
        lift_left.rotation(), lift_left.translation() + np.array([0.0, -0.4, 0.2]))
    lift_wbc.set_left_ee_target(lift_goal)
    for _ in range(50):
        lift_wbc._step()
    lift_after = float(
        lift_wbc.ik.configuration.q[lift_wbc.ik._lift_qpos_adr]
    )
    check("fix_lift suppresses virtual lift motion",
          abs(lift_after - lift_before) < 1e-4,
          f"{lift_before:.6f} -> {lift_after:.6f}")


def test_manual_override():
    print("\nmanual override arbitration")
    # Keep the override live for the whole deterministic manual-stepping
    # section; collision-QP runtime varies enough across machines that the
    # production timeout can otherwise expire before 20 steps complete.
    wbc, left, right, base = build(manual_override_timeout_s=60.0)
    import mink
    T_l, _ = wbc.forward_kinematics()
    # 0.6 m: out of arm-only reach, so the base is genuinely recruited (which
    # is what the "base yields" check needs), but still reachable once it
    # moves. Deliberately NOT the 1.5 m used before: that is unreachable
    # outright, so the arm ends up pinned against its joint limits and this
    # test then measures saturation behaviour rather than authority handover.
    # dls_projector settles when saturated (correctly -- it drives 7 joints
    # to their limits and stops), where soft keeps twitching above the 0.05
    # rad dispatch deadband, so the old target passed only by accident.
    wbc.set_left_ee_target(mink.SE3.from_rotation_and_translation(
        T_l.rotation(), T_l.translation() + np.array([0.0, -0.6, 0.0])))

    wbc.notify_manual_base_command()
    base.velocity_commands.clear()
    for _ in range(20):
        wbc._step()
        for _ in range(wbc.config.arm_interpolation_steps):
            wbc._arm_dispatch_tick()
    nonzero = [c for c in base.velocity_commands if np.any(np.abs(c) > 1e-9)]
    check("base yields to a manual drive command", not nonzero, f"{len(nonzero)} non-zero")

    wbc.notify_manual_arm_command()
    commands_before = left.commands
    for _ in range(20):
        wbc._step()
        for _ in range(wbc.config.arm_interpolation_steps):
            wbc._arm_dispatch_tick()
    check("arms yield to a manual joint command", left.commands == commands_before)

    wbc._manual_base_until = 0.0    # expire the window
    wbc._manual_arm_until = 0.0
    base.velocity_commands.clear()
    for _ in range(20):
        wbc._step()
        for _ in range(wbc.config.arm_interpolation_steps):
            wbc._arm_dispatch_tick()
    check("authority returns after the window", left.commands > commands_before)


def test_emergency_stop_resume():
    print("\nemergency-stop / resume lifecycle")
    wbc, left, right, base = build(enable_base_motion=False)
    wbc.start()
    time.sleep(0.05)

    wbc.emergency_stop()
    commands_at_stop = left.commands
    check("e-stop removes the finished worker", wbc._thread is None)
    check("e-stop marks control stopped", wbc._running is False)

    import mink
    T_l, _ = wbc.forward_kinematics()
    wbc.set_left_ee_target(mink.SE3.from_rotation_and_translation(
        T_l.rotation(), T_l.translation() + np.array([0.0, -0.10, 0.0])))
    wbc.start()
    time.sleep(0.05)
    check("start launches a new worker after e-stop",
          wbc._thread is not None and wbc._thread.is_alive())
    check("resumed worker dispatches arm commands", left.commands > commands_at_stop,
          f"{commands_at_stop} -> {left.commands}")
    wbc.stop()


def test_axis_map_and_odometry():
    print("\nbase frame conventions")
    m = BaseAxisMap()
    cmd = m.to_command(forward=1.0, lateral=2.0, yaw_rate=3.0)
    # [forward, left, yaw]. This assertion used to read [2.0, 1.0, 3.0] and so
    # locked in the crossed mapping that made the base strafe when the solver
    # asked it to drive forward. See BaseAxisMap's docstring for the three
    # independent confirmations of this order, and
    # tests/test_base_kinematics.py::test_axis_map_is_not_crossed for the
    # end-to-end check through the wheel kinematics.
    check("axis map places forward/lateral/yaw", np.allclose(cmd, [1.0, 2.0, 3.0]), str(cmd))

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
        "base_motion_enabled", "lift_motion_enabled", "fix_lift",
    }
    missing = required - set(state)
    check("carries every key the teleop client reads", not missing, str(missing))
    check("values are plain types", all(
        isinstance(state[k], (list, float, int, bool, str, type(None)))
        for k in state), str({k: type(v).__name__ for k, v in state.items()}))


def test_posture_fix_gates():
    print("\nposture-fix gates (--posture-fix none/stiffen-joint7/refresh-target)")
    from robot.arm.wholebody_ik import WholeBodyIKConfig
    import mink

    def build_with(ik_config):
        left = FakeArm([0.0, 1.32, -1.71, 1.31, 0.0, 0.0, 0.0])
        right = FakeArm([0.0, 1.32, 1.71, 1.31, 0.0, 0.0, 0.0])
        base = FakeBase()
        wbc = WholeBodyController(
            left, right, base,
            config=WholeBodyHardwareConfig(record_trajectories=False),
            ik_config=ik_config,
        )
        wbc.init()
        return wbc

    def move_away_from_home(wbc):
        T_l, _ = wbc.forward_kinematics()
        target = mink.SE3.from_rotation_and_translation(
            T_l.rotation(), T_l.translation() + np.array([0.0, -0.15, 0.05]))
        wbc.set_left_ee_target(target)
        for _ in range(100):
            wbc._step()

    # "none": legacy behavior unchanged -- posture target frozen at init,
    # so once the arm has moved away (with lift_target never set, matching
    # an arms-only session), the posture task's error grows.
    base_ik_cfg = dict(
        dt=1.0 / 30.0, solver="pyqpmad", max_iters=10,
        base_posture_cost=1e-1, lift_posture_cost=1e-4, arm_posture_cost=1e-3,
    )
    wbc = build_with(WholeBodyIKConfig(**base_ik_cfg, refresh_posture_target=False))
    move_away_from_home(wbc)
    stale_err = wbc.ik.posture_task.compute_error(wbc.ik.configuration)
    check("posture-fix=none: frozen target leaves a real posture error "
          "after moving away with no lift_target ever set",
          float(np.linalg.norm(stale_err)) > 0.05,
          f"|err|={float(np.linalg.norm(stale_err)):.4f}")

    # "refresh-target": posture reference tracks current configuration at
    # the start of every solve (not a fixed point from startup), so it can
    # never accumulate more error than roughly one solve's own convergence
    # motion -- unlike "none", which keeps compounding against an
    # increasingly stale reference for the whole session. Not exactly zero:
    # the reference is captured before that solve's max_iters QP loop moves
    # the arm further, so a residual proportional to one tick's motion
    # remains -- the claim is "far smaller than frozen", not "zero".
    wbc = build_with(WholeBodyIKConfig(**base_ik_cfg, refresh_posture_target=True))
    move_away_from_home(wbc)
    fresh_err = wbc.ik.posture_task.compute_error(wbc.ik.configuration)
    fresh_norm, stale_norm = float(np.linalg.norm(fresh_err)), float(np.linalg.norm(stale_err))
    check("posture-fix=refresh-target: posture error stays far smaller "
          "than the frozen-target case after the same move",
          fresh_norm < 0.05 and fresh_norm < stale_norm / 10.0,
          f"refresh |err|={fresh_norm:.4f} vs none |err|={stale_norm:.4f}")

    # "stiffen-joint7": the override reaches the actual posture cost vector
    # mink solves against, without touching sibling joints.
    wbc = build_with(WholeBodyIKConfig(
        **base_ik_cfg,
        arm_posture_cost_overrides={"left_arm_joint7": 1e-2, "right_arm_joint7": 1e-2},
    ))
    cost = wbc.ik._build_posture_cost()
    j7_cost = cost[wbc.ik.model.joint("left_arm_joint7").dofadr][0]
    j6_cost = cost[wbc.ik.model.joint("left_arm_joint6").dofadr][0]
    check("posture-fix=stiffen-joint7: joint7 cost raised 10x",
          abs(j7_cost - 1e-2) < 1e-12, f"joint7 cost={j7_cost}")
    check("posture-fix=stiffen-joint7: sibling joint6 cost untouched",
          abs(j6_cost - 1e-3) < 1e-12, f"joint6 cost={j6_cost}")


def test_trajectory_log():
    """One solve tick must log the solve *and* what it dispatched.

    The ordering is the part worth pinning: the recorder used to run before
    dispatch, so a row's base and lift columns would describe the previous
    tick. Tuning a controller against data offset by one cycle from the
    command that produced it is worse than not logging it at all.
    """
    print("\ntrajectory log")
    import csv
    import tempfile
    from robot.wholebody_control import _TrajectoryRecorder

    wbc, left, right, base = build(enable_base_motion=True)
    T_l, T_r = wbc.forward_kinematics()
    goal = T_l.translation() + np.array([0.0, -1.0, 0.0])   # far: recruits the base
    import mink
    wbc.set_left_ee_target(mink.SE3.from_rotation_and_translation(T_l.rotation(), goal))
    wbc.set_lift_target(0.5)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traj.csv"
        wbc._trajectory_recorder = _TrajectoryRecorder(path, wbc.ik.config, wbc.config)
        for _ in range(5):
            wbc._step()
        wbc._trajectory_recorder.close()

        rows = list(csv.reader(path.read_text().splitlines()))

    config_rows = [r for r in rows if r and r[0].startswith("#")]
    header = rows[len(config_rows)]
    data = [dict(zip(header, r)) for r in rows[len(config_rows) + 1:]]

    check("both config lines are stamped", len(config_rows) == 2, str(len(config_rows)))
    check("the base/lift knobs are recorded",
          any("lift_kp" in cell for cell in config_rows[1])
          and any("base_vel_deadband" in cell for cell in config_rows[1]))
    # Which gains were on the controllers changes what every speed number in
    # the log means. The 2026-08-22 runs had to have theirs inferred from the
    # drive tracking ratio; this stops that recurring.
    # The gain set carries its own command scale (see base_pid_*.json
    # drive_command_scale), so the provenance string covers both.
    check("the swerve gain set and the base limits are recorded",
          any("base_pid=" in cell for cell in config_rows[1])
          and any("base_heading_rate_limit=" in cell for cell in config_rows[1]),
          str(config_rows[1][-3:]))
    check("one row per solve tick", len(data) == 5, str(len(data)))
    check("every row is the full width",
          all(len(r) == len(header) for r in rows[len(config_rows) + 1:]))

    last = data[-1]
    # atol, because the column holds six decimals: half an ulp of that is
    # larger than the default relative tolerance on a command of a few
    # centimetres per second, and what is under test is the plumbing.
    check("the base command it actually sent is logged",
          last["base_active"] == "True"
          and np.isclose(float(last["base_sent_0"]), base.velocity_commands[-1][0],
                         atol=1e-6)
          and np.isclose(float(last["base_sent_1"]), base.velocity_commands[-1][1],
                         atol=1e-6),
          f"{last['base_sent_0']},{last['base_sent_1']} vs {base.velocity_commands[-1]}")
    check("the pose error and the unshaped request are logged alongside it",
          all(math.isfinite(float(last[k]))
              for k in ("base_err_fwd", "base_err_lat", "base_err_yaw",
                        "base_req_fwd", "base_req_lat", "base_req_yaw")))
    check("the lift goal and measurement are logged",
          last["lift_active"] == "True"
          and abs(float(last["lift_goal"]) - 0.5) < 1e-9
          and abs(float(last["lift_meas"]) - base.lift_height) < 0.05,
          f"goal={last['lift_goal']} meas={last['lift_meas']}")

    # FakeBase has no swerve_telemetry, so the module columns must degrade to
    # nan rather than raising or recording a module that is really at zero.
    check("a base without module telemetry logs nan, not zero",
          last["steer_meas_FL"] == "nan" and last["swerve_enabled"] == "False")

    base.swerve_telemetry = lambda: {
        "motors_enabled": True,
        "v_target": np.zeros(3), "v_profiled": np.zeros(3),
        "steer_cmd_rad": np.array([0.1, 0.2, 0.3, 0.4]),
        "steer_meas_rad": np.array([0.11, 0.19, 0.31, 0.39]),
        "drive_cmd_mps": np.full(4, 0.25), "drive_meas_raw": np.full(4, 0.24),
        "drive_pos_rot": np.array([10.5, 11.5, 12.5, 13.5]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traj2.csv"
        wbc._trajectory_recorder = _TrajectoryRecorder(path, wbc.ik.config, wbc.config)
        wbc._step()
        wbc._trajectory_recorder.close()
        rows = list(csv.reader(path.read_text().splitlines()))
    row = dict(zip(rows[2], rows[3]))
    check("cumulative drive position is logged when available",
          abs(float(row["drive_pos_FL"]) - 10.5) < 1e-9
          and abs(float(row["drive_pos_RL"]) - 13.5) < 1e-9,
          str({k: row[k] for k in ("drive_pos_FL", "drive_pos_RL")}))
    check("module commanded/measured pairs are logged when available",
          abs(float(row["steer_cmd_FL"]) - 0.1) < 1e-9
          and abs(float(row["steer_meas_FL"]) - 0.11) < 1e-9
          and abs(float(row["drive_cmd_RL"]) - 0.25) < 1e-9
          and abs(float(row["drive_meas_RL"]) - 0.24) < 1e-9,
          str({k: row[k] for k in ("steer_cmd_FL", "steer_meas_FL")}))


def main() -> int:
    for test in (
        test_init,
        test_arm_tracking,
        test_arm_command_lookahead,
        test_arm_joint_deadband,
        test_lift_servo,
        test_lift_without_feedback,
        test_base_motion,
        test_yaw_filter_and_hysteresis,
        test_base_pose_dispatch,
        test_fix_base_and_toggles,
        test_manual_override,
        test_emergency_stop_resume,
        test_axis_map_and_odometry,
        test_state_snapshot,
        test_posture_fix_gates,
        test_trajectory_log,
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
