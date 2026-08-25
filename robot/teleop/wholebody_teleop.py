"""
wholebody_teleop.py — Whole-body teleoperation client for YORv3.

Streams end-effector / lift targets over commlink RPC to a whole-body IK
server, which coordinates base + lift + both arms automatically — this client
only moves the *targets*. The same client drives either server, because both
expose the same API:

  --target sim   robot/yor_mujoco.py   port 8081   (simulation)
  --target hw    robot/yor.py          port 5557   (the real robot)

Input backends (pick with --input):
  keyboard  Terminal keys, zero extra deps (default; runs anywhere).
  gamepad   Xbox / PS4 controller via pygame.
  oculus    Meta Quest controllers via ZMQ (reuses oculus_msgs protocol).
            Controller poses are 1€-filtered on arrival — tune with
            --filter-min-cutoff / --filter-beta, or disable with
            --no-pose-filter.

Run
---
  # Simulation — start the server (macOS needs mjpython for the viewer):
  conda run -n dev mjpython robot/yor_mujoco.py
  # then, in another terminal (plain python is fine):
  conda run -n dev python robot/teleop/wholebody_teleop.py --input keyboard

  # Hardware — on the robot:
  python robot/yor.py
  # then, from the operator machine:
  python robot/teleop/wholebody_teleop.py --target hw --host <robot-ip> --input oculus

Keyboard controls
-----------------
  Left arm   w/s ±X   a/d ±Y   q/e ±Z        (nudges, default 2 cm)
  Right arm  i/k ±X   j/l ±Y   u/o ±Z
  Lift       r / f    up / down (2 cm)
  h / n      home left / right arm           g  home lift
  t          toggle fix-base                 c  toggle collision avoidance
  [ / ]      halve / double nudge step
  x or ESC   quit
"""

from __future__ import annotations

import argparse
import select
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import mink

_REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO))

from commlink import RPCClient
from robot.teleop.filters import PoseFilter

# Quest tracking arrives at approximately 72 Hz (may change -- unconfirmed) and
# is 1-euro filtered on OculusSource's own receive thread at that native rate,
# independent of this loop's own rate. This client tick just samples whatever
# pose the filter last produced, so it does not need to divide evenly into the
# Quest rate. 30 Hz matches the whole-body controller's own dispatch rate on
# hardware -- see WholeBodyHardwareConfig.control_hz in wholebody_control.py.
LOOP_RATE = 30  # Hz
# Matches the "Slider 7" range in description/robot_wholebody.xml; the server
# clamps to the model regardless, this just keeps the client's own bookkeeping
# from drifting past the real travel.
LIFT_RANGE = (0.0, 0.900)  # metres
BUTTON_DEBOUNCE_TIME = 0.2  # seconds

# Default RPC ports of the two servers this client can drive.
TARGET_PORTS = {"sim": 8081, "hw": 5557}


def _translated(T: mink.SE3, delta: np.ndarray) -> mink.SE3:
    """Return T with its translation shifted by `delta` (rotation unchanged)."""
    return mink.SE3.from_rotation_and_translation(
        T.rotation(), T.translation() + delta
    )


def apply_deadzone(arr: np.ndarray, dz: float = 0.10) -> np.ndarray:
    return np.where(
        np.abs(arr) <= dz, 0.0, np.sign(arr) * (np.abs(arr) - dz) / (1 - dz)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Command / state shared between input sources and the client
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TeleopState:
    """Client-side mirror of the current targets (sources read + update this)."""
    left_target: mink.SE3
    right_target: mink.SE3
    lift_target: float
    fix_base: bool = False
    collision_avoidance: bool = True


@dataclass
class TeleopCommand:
    """What an input source wants this tick. None = leave unchanged."""
    left_target: Optional[mink.SE3] = None
    right_target: Optional[mink.SE3] = None
    lift_target: Optional[float] = None
    # Gripper: 1.0 open, 0.0 closed. Set only on a change, never every tick --
    # the server applies a gripper value by sending the arm a joint target of
    # its *measured* pose, which competes with the interpolated targets the
    # whole-body dispatch loop is streaming. One command per open/close is a
    # blip the next dispatch tick corrects; one per tick would fight it.
    left_gripper: Optional[float] = None
    right_gripper: Optional[float] = None
    home_left: bool = False
    home_right: bool = False
    home_arms: bool = False
    home_lift: bool = False
    toggle_fix_base: bool = False
    toggle_collisions: bool = False
    quit: bool = False


class InputSource:
    """Base class for teleop input backends."""

    # [T1] Set by WholeBodyTeleop before the loop starts: a zero-argument
    # callable returning the server's get_state() dict (or None on failure).
    # Lets a source re-anchor its bookkeeping to the robot's *actual* pose
    # at meaningful moments (OculusSource uses it on clutch engage).
    state_refresh = None

    def start(self) -> None:  # acquire device
        pass

    def stop(self) -> None:  # release device
        pass

    def update(self, state: TeleopState, dt: float) -> TeleopCommand:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# 1. Keyboard (raw terminal — stdlib only)
# ─────────────────────────────────────────────────────────────────────────────

class KeyboardSource(InputSource):
    """Discrete nudge teleop from a raw (cbreak) terminal.

    Terminals deliver key-down repeats but no key-up events, so keyboard
    control is nudge-based (fixed step per keypress) rather than velocity.
    """

    _HELP = (
        "Left arm w/s a/d q/e | Right arm i/k j/l u/o | lift r/f | "
        "home h/n/g | t fix-base | c collisions | [/] step | x quit"
    )

    def __init__(self, step: float = 0.02):
        self.step = step
        self._old_attrs = None

    def start(self) -> None:
        import termios
        import tty

        if not sys.stdin.isatty():
            raise RuntimeError("keyboard input requires an interactive terminal")
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        print(f"[keyboard] {self._HELP}")

    def stop(self) -> None:
        if self._old_attrs is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def _read_keys(self) -> str:
        keys = ""
        while select.select([sys.stdin], [], [], 0)[0]:
            keys += sys.stdin.read(1)
        return keys

    def update(self, state: TeleopState, dt: float) -> TeleopCommand:
        cmd = TeleopCommand()
        s = self.step
        dl = np.zeros(3)  # left-arm nudge
        dr = np.zeros(3)  # right-arm nudge

        for key in self._read_keys():
            k = key.lower()
            if k == "w":   dl[0] += s
            elif k == "s": dl[0] -= s
            elif k == "a": dl[1] += s
            elif k == "d": dl[1] -= s
            elif k == "q": dl[2] += s
            elif k == "e": dl[2] -= s
            elif k == "i": dr[0] += s
            elif k == "k": dr[0] -= s
            elif k == "j": dr[1] += s
            elif k == "l": dr[1] -= s
            elif k == "u": dr[2] += s
            elif k == "o": dr[2] -= s
            elif k == "r":
                cmd.lift_target = min(LIFT_RANGE[1], state.lift_target + s)
            elif k == "f":
                cmd.lift_target = max(LIFT_RANGE[0], state.lift_target - s)
            elif k == "h": cmd.home_left = True
            elif k == "n": cmd.home_right = True
            elif k == "g": cmd.home_lift = True
            elif k == "t": cmd.toggle_fix_base = True
            elif k == "c": cmd.toggle_collisions = True
            elif k == "[":
                self.step = max(0.005, self.step / 2)
                print(f"\n[keyboard] step = {self.step*100:.1f} cm")
            elif k == "]":
                self.step = min(0.16, self.step * 2)
                print(f"\n[keyboard] step = {self.step*100:.1f} cm")
            elif k in ("x", "\x1b"):  # x or ESC
                cmd.quit = True

        if np.any(dl):
            cmd.left_target = _translated(state.left_target, dl)
        if np.any(dr):
            cmd.right_target = _translated(state.right_target, dr)
        return cmd


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gamepad (pygame — Xbox / PS4)
# ─────────────────────────────────────────────────────────────────────────────

class GamepadSource(InputSource):
    """Velocity-style teleop from a gamepad.

    Hold L1 → sticks drive the LEFT arm; hold R1 → sticks drive the RIGHT arm.
      left stick   X/Y translation
      right stick  vertical = Z translation
    D-pad up/down = lift. START toggles fix-base, BACK toggles collisions,
    X/Y (or equivalents) home left/right.
    """

    ARM_SPEED = 0.25   # m/s at full stick deflection
    LIFT_SPEED = 0.10  # m/s

    def __init__(self):
        self._debounce: dict[int, float] = {}

    def start(self) -> None:
        import pygame  # deferred so keyboard mode needs no pygame

        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            raise RuntimeError("no gamepad detected")
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        print(f"[gamepad] using: {self.js.get_name()}")

    def stop(self) -> None:
        self._pygame.quit()

    def _pressed(self, button: int) -> bool:
        """Debounced button read."""
        if self.js.get_button(button):
            now = time.time()
            if now - self._debounce.get(button, 0.0) > BUTTON_DEBOUNCE_TIME:
                self._debounce[button] = now
                return True
        return False

    def update(self, state: TeleopState, dt: float) -> TeleopCommand:
        pg = self._pygame
        pg.event.pump()
        cmd = TeleopCommand()

        axes = apply_deadzone(np.array([
            -self.js.get_axis(1),  # left stick vertical   → +X (push up = fwd)
            -self.js.get_axis(0),  # left stick horizontal → +Y
            -self.js.get_axis(4),  # right stick vertical  → +Z
        ]))
        delta = axes * self.ARM_SPEED * dt

        if self.js.get_button(4) and np.any(delta):    # L1 held → left arm
            cmd.left_target = _translated(state.left_target, delta)
        if self.js.get_button(5) and np.any(delta):    # R1 held → right arm
            cmd.right_target = _translated(state.right_target, delta)

        hat_y = self.js.get_hat(0)[1] if self.js.get_numhats() > 0 else 0
        if hat_y != 0:
            lift = state.lift_target + hat_y * self.LIFT_SPEED * dt
            cmd.lift_target = float(np.clip(lift, *LIFT_RANGE))

        if self._pressed(7): cmd.toggle_fix_base = True     # START
        if self._pressed(6): cmd.toggle_collisions = True   # BACK
        if self._pressed(2): cmd.home_left = True           # X
        if self._pressed(3): cmd.home_right = True          # Y
        return cmd


# ─────────────────────────────────────────────────────────────────────────────
# 3. Oculus / Meta Quest (ZMQ, clutch-based 6-DoF)
# ─────────────────────────────────────────────────────────────────────────────

class OculusSource(InputSource):
    """Clutch-based 6-DoF teleop from Quest controllers.

    X (left) / A (right) toggle per-arm engagement. While engaged, the arm
    target follows the controller's pose delta since engagement (position and
    orientation), using the same frame decomposition as the hardware teleop.
    Y / B run the safe home sequence for the left / right arm; pressing Y+B
    together homes both after a single base-lock and lift-to-450-mm preamble.
    Right thumbstick Y drives the lift.

    Poses are 1€-filtered as they arrive (see robot/teleop/filters.py), at the
    headset's ~72 Hz rather than the 30 Hz teleop loop, so tracker jitter does
    not reach the IK and a tracking dropout holds the target instead of
    throwing it. Pass pose_filter=False to stream the raw poses.
    """

    VR_CONTROLLER_TOPIC = b"oculus_controller"
    LIFT_SPEED = 0.15  # m/s at full stick
    GRIPPER_TRIGGER_THRESHOLD = 0.5  # index trigger past this = closed

    def __init__(self, host: str, port: int = 5555, pose_filter: bool = True,
                 filter_min_cutoff: float = 3.0, filter_beta: float = 8.0,
                 yaw_correction_deg: float = 270.0, legacy_oculus_app: bool = False,
                 clutch_reseed: bool = False):
        self.host, self.port = host, port
        # [T1] On engage, re-anchor the clutch to the robot's *actual* EE
        # pose (one get_state() RPC) instead of the client's wound-up local
        # target. Without this, streaming into a constraint banks the
        # blocked distance in the client's bookkeeping, and the next engage
        # starts from a target the robot never reached -- controls feel
        # dead until the operator has unwound the phantom offset by hand.
        self._clutch_reseed = bool(clutch_reseed)
        # v0.1 (com.GRAIL.YORTeleop) sends Left|Right; v0.2
        # (com.GRAIL.Yor_Teleop) sends Head|Left|Right. Gated explicitly
        # rather than auto-detected -- see parse_controller_state.
        self._legacy_oculus_app = bool(legacy_oculus_app)
        self._filters = {
            side: PoseFilter(min_cutoff=filter_min_cutoff,
                             rot_min_cutoff=filter_min_cutoff,
                             beta=filter_beta)
            for side in ("left", "right")
        } if pose_filter else {}
        self._last_glitch_report = 0.0
        self._reported_drops = 0
        # controller frame → robot frame (translation only, see H_rot below)
        self.H = mink.SE3.from_rotation(mink.SO3.from_matrix(
            np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        ))
        # Rotation delta gets its own conjugation matrix -- it used to reuse
        # H above, which is tuned for translation axes and happens to be a
        # forward 3-cycle permutation (ctrl X->robot Y, Y->Z, Z->X). Reusing
        # it for rotation carried that same permutation into orientation,
        # so pitching the controller rolled the EE, yawing it pitched the
        # EE, and rolling it yawed the EE -- confirmed both on left and
        # right arms via extra/diagnose_*_teleop_axes.py calibration runs
        # (see extra/*_teleop_axes_2026*.json). H_rot is the inverse
        # 3-cycle (ctrl X->EE Z, Y->X, Z->Y), which cancels it out. Row
        # signs on Y and Z were then flipped after a live left-arm
        # recheck: pitch (X row) came back correct as first written, but
        # yaw (Y row) came out mirrored (commanded yaw_left read as yaw
        # right); flipping only that row would make the matrix a
        # reflection (det -1, rejected by mink.SO3), so the Z (roll) row
        # flips with it to keep det = +1. Roll's own sign wasn't
        # separately confirmed live -- if it now reads backward, that's
        # the row to revisit.
        self.H_rot = mink.SE3.from_rotation(mink.SO3.from_matrix(
            np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        ))
        # The Quest tracking space is fixed, but its planar axes are rotated
        # relative to the robot as it sits in the room.  Keep this calibration
        # separate from H: it rotates only translation about robot Z and must
        # not change the controller-local orientation mapping.
        self.translation_yaw_correction = mink.SO3.from_z_radians(
            np.deg2rad(float(yaw_correction_deg))
        )
        self.yaw_correction_deg = float(yaw_correction_deg)
        self._latest = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._engaged = {"left": False, "right": False}
        self._clutch: dict[str, tuple[mink.SE3, mink.SE3]] = {}
        self._debounce: dict[str, float] = {}
        self._button_down: dict[str, bool] = {}
        # Last gripper value actually sent per side, so update() can send on
        # change only. Cleared on disengage: whatever the operator holds when
        # they re-engage is then re-sent, rather than assumed still in effect.
        self._gripper_sent: dict[str, Optional[float]] = {"left": None, "right": None}

    def start(self) -> None:
        import zmq  # deferred so keyboard mode needs no zmq

        from robot.teleop.oculus_msgs import parse_controller_state
        self._parse = parse_controller_state
        self._zmq = zmq
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[oculus] subscribing to tcp://{self.host}:{self.port} ...")

    def stop(self) -> None:
        self._stop.set()

    # Bound on how many backlogged messages _worker will discard in one go.
    # Real backlogs are a handful of packets (a brief GIL stall); this is
    # just a safety cap against looping forever if a publisher ever floods.
    _MAX_DRAIN = 64

    def _worker(self) -> None:
        ctx = self._zmq.Context()
        sock = ctx.socket(self._zmq.SUB)
        sock.connect(f"tcp://{self.host}:{self.port}")
        sock.subscribe(self.VR_CONTROLLER_TOPIC)
        sock.setsockopt(self._zmq.RCVTIMEO, 200)
        while not self._stop.is_set():
            try:
                _, message = sock.recv_multipart()
            except self._zmq.ZMQError:
                continue  # timeout — loop and re-check stop flag

            # If the receive thread fell behind for a moment (GIL contention
            # with the main loop's RPC calls, OS scheduling, ...), ZMQ has
            # queued whatever the headset sent meanwhile. Draining to the
            # newest one here -- rather than filtering every queued message
            # -- is what actually fixes the burst: processing a backlog one
            # message at a time feeds the 1-euro filter a run of samples
            # whose *local* receive times are compressed into microseconds
            # while the real positions moved over the true, much longer
            # gap. That looks like an impossible hand speed and gets
            # rejected as a glitch -- which is what produced the "in
            # brakes" stutter, not a bug in the filter's own math (a replay
            # of a raw capture through PoseFilter with correct timestamps
            # shows zero false rejections). Only the freshest pose is ever
            # useful for teleop anyway; the stale intermediate ones are
            # simply discarded, not fed through the filter as fake motion.
            for _ in range(self._MAX_DRAIN):
                try:
                    _, message = sock.recv_multipart(flags=self._zmq.NOBLOCK)
                except self._zmq.ZMQError:
                    break

            # Timestamp the sample now, once we actually have the freshest
            # one in hand -- not cs.created_timestamp (time.time() inside
            # parse_controller_state), and monotonic so it can't jump
            # backwards on a system clock adjustment.
            recv_time = time.monotonic()
            cs = self._parse(message.decode(), legacy=self._legacy_oculus_app)
            poses = self._filtered(cs, recv_time)
            with self._latest_lock:
                self._latest = (cs, poses)
        sock.close()
        ctx.destroy()

    def _filtered(self, cs, recv_time: float) -> dict[str, mink.SE3]:
        """Smooth this sample's controller poses (identity if filtering is off).

        Runs on the receive thread, timestamped with `recv_time` (the actual
        local arrival time of the freshest queued message) rather than
        `cs.created_timestamp`, which the filter's glitch gate is sensitive to
        -- see the backlog-draining comment in _worker for why.
        """
        poses = {"left": cs.left_SE3, "right": cs.right_SE3}
        if not self._filters:
            return poses
        for side, filt in self._filters.items():
            poses[side] = filt(poses[side], recv_time)

        # Report dropouts, but at most every couple of seconds and only when
        # the count has actually moved — this runs at ~72 Hz.
        dropped = sum(f.rejected for f in self._filters.values())
        now = time.time()
        if dropped > self._reported_drops and now - self._last_glitch_report > 2.0:
            print(f"\n[oculus] dropped {dropped - self._reported_drops} pose "
                  f"samples as tracking glitches ({dropped} total)")
            self._reported_drops, self._last_glitch_report = dropped, now
        return poses

    def _debounced(self, name: str, value: bool) -> bool:
        # A long-running home RPC can outlast the old time-only debounce. Use
        # a rising edge as well, so holding Y/B never launches a second home
        # the instant the first one returns.
        was_down = self._button_down.get(name, False)
        self._button_down[name] = bool(value)
        if value and not was_down:
            now = time.time()
            if now - self._debounce.get(name, 0.0) > BUTTON_DEBOUNCE_TIME:
                self._debounce[name] = now
                return True
        return False

    def _ee_target(self, X_Cinit: mink.SE3, X_ee_init: mink.SE3,
                   X_Ctarget: mink.SE3) -> mink.SE3:
        """Controller delta → EE target (translate + rotate decomposed)."""
        X_Cdelta = X_Cinit.inverse().multiply(X_Ctarget)
        X_Rdelta = self.H_rot.inverse() @ X_Cdelta @ self.H_rot

        # Translation belongs to the fixed Quest tracking frame.  Taking it
        # from X_Cdelta would express it in the controller's local frame at
        # clutch time, making merely tilting the controller rotate the meaning
        # of "forward".  Map the tracking-frame displacement into robot axes,
        # then apply the room calibration about robot Z.
        controller_displacement = (
            X_Ctarget.translation() - X_Cinit.translation()
        )
        robot_displacement = self.H.rotation().inverse().apply(
            controller_displacement
        )
        target_pos = (
            X_ee_init.translation()
            + self.translation_yaw_correction.apply(robot_displacement)
        )
        target_rot = X_ee_init.rotation() @ X_Rdelta.rotation()
        return mink.SE3(np.concatenate([target_rot.wxyz, target_pos]))

    def update(self, state: TeleopState, dt: float) -> TeleopCommand:
        cmd = TeleopCommand()
        with self._latest_lock:
            latest = self._latest
        if latest is None:
            return cmd
        cs, poses = latest
        home_pressed: dict[str, bool] = {}

        for side, engage_btn, home_btn, ctrl_T, tgt_attr, trigger in (
            ("left",  cs.left_x,  cs.left_y,  poses["left"],  "left_target",
             cs.left_index_trigger),
            ("right", cs.right_a, cs.right_b, poses["right"], "right_target",
             cs.right_index_trigger),
        ):
            if self._debounced(f"{side}_engage", engage_btn):
                self._engaged[side] = not self._engaged[side]
                if self._engaged[side]:
                    # [T1] Anchor to the robot's actual EE pose, not the
                    # client's local target -- see __init__. Falls back to
                    # the local target if the RPC fails, which is exactly
                    # the pre-gate behaviour.
                    if self._clutch_reseed and self.state_refresh is not None:
                        srv = self.state_refresh()
                        key = f"{side}_ee_wxyz_xyz"
                        if srv and srv.get(key) is not None:
                            setattr(state, tgt_attr,
                                    mink.SE3(np.array(srv[key])))
                        else:
                            print(f"[oculus] {side} clutch reseed failed -- "
                                  "using local target")
                    self._clutch[side] = (ctrl_T, getattr(state, tgt_attr))
                else:
                    self._gripper_sent[side] = None
                print(f"[oculus] {side} {'engaged' if self._engaged[side] else 'disengaged'}")
            home_pressed[side] = self._debounced(f"{side}_home", home_btn)
            if home_pressed[side]:
                self._engaged[side] = False
                self._gripper_sent[side] = None
                setattr(cmd, f"home_{side}", True)
            if self._engaged[side]:
                X_Cinit, X_ee_init = self._clutch[side]
                setattr(cmd, tgt_attr, self._ee_target(X_Cinit, X_ee_init, ctrl_T))
                # Index trigger is the grip: squeezed closes, released opens.
                # Thresholded rather than passed through as an analogue value
                # because the servo is driven to a calibrated open/close pair,
                # and because a continuous stream would re-command the arm's
                # measured pose on every tick (see TeleopCommand.left_gripper).
                grip = 0.0 if trigger > self.GRIPPER_TRIGGER_THRESHOLD else 1.0
                if grip != self._gripper_sent[side]:
                    self._gripper_sent[side] = grip
                    setattr(cmd, f"{side}_gripper", grip)

        if home_pressed.get("left") and home_pressed.get("right"):
            cmd.home_left = False
            cmd.home_right = False
            cmd.home_arms = True

        stick_y = apply_deadzone(np.array([cs.right_thumbstick_axes[1]]))[0]
        if stick_y != 0.0:
            lift = state.lift_target + stick_y * self.LIFT_SPEED * dt
            cmd.lift_target = float(np.clip(lift, *LIFT_RANGE))
        return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

class WholeBodyTeleop:
    """Streams targets from an InputSource to the whole-body IK sim server."""

    def __init__(self, source: InputSource, host: str = "localhost",
                 port: int = 8081, rate_hz: int = LOOP_RATE):
        self.source = source
        self.rate_hz = rate_hz
        print(f"[teleop] connecting to {host}:{port} ...")
        self.yor = RPCClient(host, port)
        self.yor.init()

        # Seed local target state from the server's actual pose
        srv = self.yor.get_state()
        if not srv:
            raise RuntimeError(
                "server returned an empty state — the hardware node is running "
                "without whole-body control (started with no_arms, or the arms "
                "failed to initialise). Check the yor.py console output."
            )
        self.state = TeleopState(
            left_target=mink.SE3(np.array(srv["left_ee_wxyz_xyz"])),
            right_target=mink.SE3(np.array(srv["right_ee_wxyz_xyz"])),
            lift_target=float(srv["lift"]),
            fix_base=bool(srv["fix_base"]),
            collision_avoidance=bool(srv["collision_avoidance"]),
        )
        print(f"[teleop] synced: lift={self.state.lift_target:.3f} m, "
              f"fix_base={self.state.fix_base}, "
              f"collisions={self.state.collision_avoidance}")

    def _server_state(self) -> Optional[dict]:
        """One get_state() RPC, or None on any failure (never raises)."""
        try:
            srv = self.yor.get_state()
            return srv or None
        except Exception as exc:
            print(f"\n[teleop] get_state failed: {exc}")
            return None

    def _dispatch(self, cmd: TeleopCommand) -> None:
        st = self.state
        home_result = None
        home_label = None
        if cmd.home_arms:
            home_result = self.yor.home_arms()
            home_label = "both arms"
        elif cmd.home_left:
            home_result = self.yor.home_left_arm()
            home_label = "left arm"
        elif cmd.home_right:
            home_result = self.yor.home_right_arm()
            home_label = "right arm"

        if home_label is not None:
            if home_result:
                srv = self.yor.get_state()
                st.left_target = mink.SE3(np.array(srv["left_ee_wxyz_xyz"]))
                st.right_target = mink.SE3(np.array(srv["right_ee_wxyz_xyz"]))
                st.lift_target = float(srv["lift"])
                st.fix_base = bool(srv["fix_base"])
                st.collision_avoidance = bool(srv["collision_avoidance"])
                print(f"\n[teleop] {home_label} home sequence complete")
            else:
                print(f"\n[teleop] {home_label} home sequence FAILED")
        if cmd.home_lift:
            self.yor.lift_home()
            st.lift_target = float(self.yor.get_state()["lift"])
            print("\n[teleop] lift → home")
        if cmd.toggle_fix_base:
            st.fix_base = self.yor.toggle_fix_base()
            print(f"\n[teleop] fix_base = {st.fix_base}")
        if cmd.toggle_collisions:
            st.collision_avoidance = self.yor.toggle_collision_avoidance()
            print(f"\n[teleop] collision_avoidance = {st.collision_avoidance}")

        # Targets: send atomically when both move, individually otherwise.
        # A gripper change rides along with the pose it belongs to; it only
        # travels on its own when that arm sent no pose this tick.
        if cmd.left_target is not None and cmd.right_target is not None:
            st.left_target, st.right_target = cmd.left_target, cmd.right_target
            self.yor.set_bimanual_ee_target(
                L_ee_target=st.left_target, R_ee_target=st.right_target,
                L_gripper_target=cmd.left_gripper,
                R_gripper_target=cmd.right_gripper)
        elif cmd.left_target is not None:
            st.left_target = cmd.left_target
            self.yor.set_left_ee_target(ee_target=st.left_target,
                                        gripper_target=cmd.left_gripper)
            if cmd.right_gripper is not None:
                self.yor.set_right_ee_target(ee_target=st.right_target,
                                             gripper_target=cmd.right_gripper)
        elif cmd.right_target is not None:
            st.right_target = cmd.right_target
            self.yor.set_right_ee_target(ee_target=st.right_target,
                                         gripper_target=cmd.right_gripper)
            if cmd.left_gripper is not None:
                self.yor.set_left_ee_target(ee_target=st.left_target,
                                            gripper_target=cmd.left_gripper)
        else:
            if cmd.left_gripper is not None:
                self.yor.set_left_ee_target(ee_target=st.left_target,
                                            gripper_target=cmd.left_gripper)
            if cmd.right_gripper is not None:
                self.yor.set_right_ee_target(ee_target=st.right_target,
                                             gripper_target=cmd.right_gripper)

        if cmd.lift_target is not None:
            st.lift_target = cmd.lift_target
            self.yor.set_lift_target(st.lift_target)

    def run(self) -> None:
        from loop_rate_limiters import RateLimiter

        rate = RateLimiter(self.rate_hz, warn=False)
        dt = 1.0 / self.rate_hz
        last_hud = 0.0
        # [T1] Give the source a way to read the server's live state (used
        # by OculusSource's clutch reseed). Same thread as _dispatch, so the
        # RPC client is never shared across threads.
        self.source.state_refresh = self._server_state
        self.source.start()
        try:
            while True:
                cmd = self.source.update(self.state, dt)
                if cmd.quit:
                    break
                self._dispatch(cmd)

                now = time.time()
                if now - last_hud > 1.0:  # 1 Hz status line
                    lp = self.state.left_target.translation()
                    rp = self.state.right_target.translation()
                    print(
                        f"\r[teleop] L=({lp[0]:+.2f},{lp[1]:+.2f},{lp[2]:+.2f}) "
                        f"R=({rp[0]:+.2f},{rp[1]:+.2f},{rp[2]:+.2f}) "
                        f"lift={self.state.lift_target:.2f} "
                        f"fix_base={self.state.fix_base} "
                        f"col={self.state.collision_avoidance}   ",
                        end="", flush=True,
                    )
                    last_hud = now
                rate.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            self.source.stop()
            print("\n[teleop] stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", choices=["keyboard", "gamepad", "oculus"],
                        default="keyboard", help="input backend")
    parser.add_argument("--target", choices=["sim", "hw"], default="sim",
                        help="which whole-body server to drive (sets the default port)")
    parser.add_argument("--host", default="localhost", help="RPC server host")
    parser.add_argument("--port", type=int, default=None,
                        help="RPC server port (default: 8081 for sim, 5557 for hw)")
    parser.add_argument("--rate", type=int, default=LOOP_RATE, help="loop rate (Hz)")
    parser.add_argument("--oculus-host", default="10.21.116.241",
                        help="Quest headset IP (oculus input only)")
    parser.add_argument("--no-pose-filter", action="store_true",
                        help="stream raw Quest poses (skip 1€ filtering)")
    parser.add_argument("--clutch-reseed", action="store_true",
                        help="[T1] on engage, anchor the clutch to the robot's "
                             "actual EE pose (one get_state RPC) instead of the "
                             "client's local target -- clears any wind-up banked "
                             "while streaming into a constraint (oculus input "
                             "only; default: off = current behaviour)")
    parser.add_argument("--filter-min-cutoff", type=float, default=3.0,
                        help="1€ cutoff (Hz) at rest — lower = smoother, laggier")
    parser.add_argument("--filter-beta", type=float, default=8.0,
                        help="1€ speed coefficient — higher = more responsive "
                             "while moving, passes more jitter")
    parser.add_argument("--oculus-yaw-correction", type=float, default=270.0,
                        help="CCW Quest-to-robot XY correction in degrees")
    parser.add_argument("--legacy-oculus-app", action="store_true",
                        help="parse the v0.1 Quest app's packets "
                             "(com.GRAIL.YORTeleop, Left|Right) instead of "
                             "v0.2's (com.GRAIL.Yor_Teleop, Head|Left|Right) "
                             "-- default is v0.2")
    parser.add_argument("--step", type=float, default=0.02,
                        help="keyboard nudge step in metres")
    args = parser.parse_args()
    port = args.port if args.port is not None else TARGET_PORTS[args.target]

    source: InputSource
    if args.input == "keyboard":
        source = KeyboardSource(step=args.step)
    elif args.input == "gamepad":
        source = GamepadSource()
    else:
        source = OculusSource(host=args.oculus_host,
                              pose_filter=not args.no_pose_filter,
                              filter_min_cutoff=args.filter_min_cutoff,
                              filter_beta=args.filter_beta,
                              yaw_correction_deg=args.oculus_yaw_correction,
                              legacy_oculus_app=args.legacy_oculus_app,
                              clutch_reseed=args.clutch_reseed)

    print(f"[teleop] target = {args.target}")
    WholeBodyTeleop(source, host=args.host, port=port, rate_hz=args.rate).run()


if __name__ == "__main__":
    main()
