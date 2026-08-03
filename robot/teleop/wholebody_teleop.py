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

LOOP_RATE = 30  # Hz
# Matches the "Slider 7" range in description/robot_wholebody.xml; the server
# clamps to the model regardless, this just keeps the client's own bookkeeping
# from drifting past the real travel.
LIFT_RANGE = (0.0, 0.9176)  # metres
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
    home_left: bool = False
    home_right: bool = False
    home_lift: bool = False
    toggle_fix_base: bool = False
    toggle_collisions: bool = False
    quit: bool = False


class InputSource:
    """Base class for teleop input backends."""

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
    Y / B home the arms. Right thumbstick Y drives the lift.
    """

    VR_CONTROLLER_TOPIC = b"oculus_controller"
    LIFT_SPEED = 0.15  # m/s at full stick

    def __init__(self, host: str, port: int = 5555):
        self.host, self.port = host, port
        # controller frame → robot frame
        self.H = mink.SE3.from_rotation(mink.SO3.from_matrix(
            np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        ))
        self._latest = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._engaged = {"left": False, "right": False}
        self._clutch: dict[str, tuple[mink.SE3, mink.SE3]] = {}
        self._debounce: dict[str, float] = {}

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

    def _worker(self) -> None:
        ctx = self._zmq.Context()
        sock = ctx.socket(self._zmq.SUB)
        sock.connect(f"tcp://{self.host}:{self.port}")
        sock.subscribe(self.VR_CONTROLLER_TOPIC)
        sock.setsockopt(self._zmq.RCVTIMEO, 200)
        while not self._stop.is_set():
            try:
                _, message = sock.recv_multipart()
                cs = self._parse(message.decode())
                with self._latest_lock:
                    self._latest = cs
            except self._zmq.ZMQError:
                pass  # timeout — loop and re-check stop flag
        sock.close()
        ctx.destroy()

    def _debounced(self, name: str, value: bool) -> bool:
        if value:
            now = time.time()
            if now - self._debounce.get(name, 0.0) > BUTTON_DEBOUNCE_TIME:
                self._debounce[name] = now
                return True
        return False

    def _ee_target(self, X_Cinit: mink.SE3, X_ee_init: mink.SE3,
                   X_Ctarget: mink.SE3) -> mink.SE3:
        """Controller delta → EE target (translate + rotate decomposed)."""
        X_Cdelta = X_Cinit.inverse().multiply(X_Ctarget)
        X_Rdelta = self.H.inverse() @ X_Cdelta @ self.H
        target_pos = X_ee_init.translation() + X_Rdelta.translation()
        target_rot = X_ee_init.rotation() @ X_Rdelta.rotation()
        return mink.SE3(np.concatenate([target_rot.wxyz, target_pos]))

    def update(self, state: TeleopState, dt: float) -> TeleopCommand:
        cmd = TeleopCommand()
        with self._latest_lock:
            cs = self._latest
        if cs is None:
            return cmd

        for side, engage_btn, home_btn, ctrl_T, tgt_attr in (
            ("left",  cs.left_x,  cs.left_y,  cs.left_SE3,  "left_target"),
            ("right", cs.right_a, cs.right_b, cs.right_SE3, "right_target"),
        ):
            if self._debounced(f"{side}_engage", engage_btn):
                self._engaged[side] = not self._engaged[side]
                if self._engaged[side]:
                    self._clutch[side] = (ctrl_T, getattr(state, tgt_attr))
                print(f"[oculus] {side} {'engaged' if self._engaged[side] else 'disengaged'}")
            if self._debounced(f"{side}_home", home_btn):
                self._engaged[side] = False
                setattr(cmd, f"home_{side}", True)
            if self._engaged[side]:
                X_Cinit, X_ee_init = self._clutch[side]
                setattr(cmd, tgt_attr, self._ee_target(X_Cinit, X_ee_init, ctrl_T))

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

    def _dispatch(self, cmd: TeleopCommand) -> None:
        st = self.state
        if cmd.home_left:
            self.yor.home_left_arm()
            st.left_target = mink.SE3(np.array(self.yor.get_state()["left_ee_wxyz_xyz"]))
            print("\n[teleop] left arm → home")
        if cmd.home_right:
            self.yor.home_right_arm()
            st.right_target = mink.SE3(np.array(self.yor.get_state()["right_ee_wxyz_xyz"]))
            print("\n[teleop] right arm → home")
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

        # Targets: send atomically when both move, individually otherwise
        if cmd.left_target is not None and cmd.right_target is not None:
            st.left_target, st.right_target = cmd.left_target, cmd.right_target
            self.yor.set_bimanual_ee_target(
                L_ee_target=st.left_target, R_ee_target=st.right_target)
        elif cmd.left_target is not None:
            st.left_target = cmd.left_target
            self.yor.set_left_ee_target(ee_target=st.left_target)
        elif cmd.right_target is not None:
            st.right_target = cmd.right_target
            self.yor.set_right_ee_target(ee_target=st.right_target)

        if cmd.lift_target is not None:
            st.lift_target = cmd.lift_target
            self.yor.set_lift_target(st.lift_target)

    def run(self) -> None:
        from loop_rate_limiters import RateLimiter

        rate = RateLimiter(self.rate_hz, warn=False)
        dt = 1.0 / self.rate_hz
        last_hud = 0.0
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
        source = OculusSource(host=args.oculus_host)

    print(f"[teleop] target = {args.target}")
    WholeBodyTeleop(source, host=args.host, port=port, rate_hz=args.rate).run()


if __name__ == "__main__":
    main()
