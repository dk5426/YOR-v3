import math
import os
import re
import threading
import time
from enum import IntEnum
from queue import Queue
from typing import Any, Optional, Tuple, Union

import numpy as np
import queue  # for queue.Empty / queue.Full

from loop_rate_limiters import RateLimiter

# --- New motor API ---
from sparkcan_py import SparkFlex

# Optional enums if your binding exposes them (safe to ignore if not present)
# Only the two enums this file actually uses. They used to be imported
# alongside MotorType and SensorType, which the installed binding does not
# export -- so the import raised, the handler set *all four* to None, and every
# `if IdleMode ...` / `if CtrlType ...` guard below was silently dead. Idle mode
# and control type were therefore whatever the SPARKs held in flash, unset and
# unreported. Import only what exists, so a future missing name fails loudly
# instead of disabling unrelated calls.
try:
    from sparkcan_py import CtrlType, IdleMode
except Exception:  # pragma: no cover - binding without the enums
    IdleMode = CtrlType = None  # type: ignore

# Pico lift (serial)
try:
    import serial  # type: ignore
except Exception:
    serial = None  # type: ignore


# ----------------------------
# Constants / config
# ----------------------------
drivetrain_can = "can0"

POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD_NS = int(1e9 / POLICY_CONTROL_FREQ)

# Hz control loop. Three ticks per whole-body cycle: the BASE_VEL relay in
# robot/base.py forwards a new velocity at 108 Hz, and the S-curve profiler
# below integrates at this rate, so 3× gives it something to interpolate
# across instead of stepping once per arriving command.
#
# Each tick puts 16 frames on the 1 Mbit/s bus (8 heartbeats + 4 steer
# positions + 4 drive velocities), so this is also the knob that sets bus
# load: 5184 frames/s here, on top of the 50 Hz periodic status each of the
# 8 SPARKs streams back. Raising it further is a bus-utilisation decision,
# not a free one.
CONTROL_FREQ = 324
CONTROL_PERIOD = 1.0 / CONTROL_FREQ

NUM_SWERVES = 4
LENGTH = 0.1225  # m
WIDTH = 0.170  # m
TIRE_RADIUS = 0.0381  # m

# Usable lift travel. Must match the "Slider 7" joint range in
# description/robot_wholebody.xml — the whole-body solver clamps its lift
# commands to the model, so a smaller number here would silently truncate
# the top of the workspace.
LIFT_MAX_HEIGHT_M = 0.900  # m

# Stable udev path for this robot's FT232-connected lift Arduino. Unlike
# /dev/ttyUSB0, this does not change when USB devices are enumerated in a
# different order. YOR_LIFT_SERIAL_PORT can override it for replacement
# hardware or bench testing.
LIFT_SERIAL_PORT = (
    "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG02XPI5-if00-port0"
)

# Streamed-velocity limits. These mirror VEL_MAX_MM_S and VEL_MIN_ACTIVE_MM_S in
# firmware/lift_controller/lift_controller.ino. The firmware clamps as well —
# it has to, since it cannot trust the host — but clamping here too means the
# host always knows exactly what it asked for.
LIFT_MAX_VELOCITY_MM_S = 50.0
LIFT_MIN_VELOCITY_MM_S = 0.5
# Refresh interval for an unchanged velocity. Comfortably inside the firmware's
# 300 ms command timeout, so a steady command never looks like a dead link.
LIFT_VELOCITY_KEEPALIVE_S = 0.100
# Capability token the firmware advertises when it understands "vel".
LIFT_VELOCITY_CAPABILITY = "lift_velocity_v1"

MODULE_ORDER = ("FL", "FR", "RR", "RL")

CAN_IDS_DRIVE = (1, 4, 3, 2)  # [FL, FR, RR, RL]
CAN_IDS_ROT = (5, 8, 7, 6)  # [FL, FR, RR, RL]

ROTATION_OFFSETS = np.array([0.75, 0.00, 0.25, 0.50], dtype=float)

ROT_DIAG_SWAP_PERM = np.array([1, 0, 3, 2], dtype=int)
TRANS_OPPOSITE_MASK = np.array([False, False, False, False], dtype=bool)

TWO_PI = 2.0 * math.pi

# Close the steering loop on the absolute encoder rather than on the last
# command. This was False for two reasons, both now gone: the encoder was being
# read as degrees when it returns turns (so feedback would have been a
# constant -- see RotationMotor.get_absolute_turns), and nobody had measured
# whether the modules actually lag. The 2026-08-22 runs measured it: 8-11 deg
# of error with the command parked and 23-28 deg while it slews. With this
# False, `cos_error_scaling` compared each command against the *previous
# command* and so collapsed to 1.0 after a single 3.1 ms tick; with it True the
# scaling finally throttles drive speed while a module is still turning.
USE_FEEDBACK_FOR_STEER = True

# Native controller setpoint per m/s of wheel speed. Correct only for a
# particular set of drive gains, so it is declared per manifest
# (`drive_command_scale` in config/base_pid_*.json) and passed in; this is the
# fallback for callers that construct Base() directly. 2.0 is the stock value:
# the stock loop is P-only and reaches about 40% of setpoint on the floor, and
# 2.0 is what compensates for that. See docs/BASE_COMMAND_LOOP_REVIEW.md
# finding 6.
DRIVE_VEL_SCALE = 2.0

# Below this commanded wheel speed a module has no meaningful direction, and its
# steering setpoint is held rather than recomputed. See
# _vehicle_velocity_to_angle_and_speed. In m/s at the wheel: a pure spin at
# 0.005 rad/s puts about 1 mm/s on each wheel, so this is comfortably under any
# velocity the base is ever asked for and safely above float noise.
ZERO_SPEED_EPS_MPS = 1e-3

# Ceiling on how fast a *measured* steering angle may plausibly change, used to
# reject bad absolute-encoder samples in Base._update_state.
#
# The modules were measured at 265-353 deg/s (4.6-6.2 rad/s) turning flat out,
# so anything past 7 rad/s did not happen mechanically -- it is a Period5 CAN
# frame that arrived late, was dropped, or was decoded from a stale cache.
# Measured on the 2026-08-27 wholebody runs: while the base was actually
# driving, 11-12% of samples implied a rate above 12 rad/s, with jumps as large
# as pi. Stationary the same figure is 0.11%, which is the tell -- a stale frame
# is invisible when the true angle is not changing.
#
# That matters because steer_pos is not just telemetry. It decides the
# shortest-path flip in _vehicle_velocity_to_angle_and_speed, it scales the
# drive setpoint through cos_error_scaling, and it *becomes* the steering
# command for a module below ZERO_SPEED_EPS_MPS. One bad sample therefore
# jumped a module command by pi and reversed its drive velocity: the same runs
# logged 386 pi-jumps and 228 drive sign reversals, every reversal on the same
# tick as a jump, against a mean drive command of only 0.047 m/s.
#
# Deliberately a slew limit on the measurement, not a hysteresis band on the
# flip test. The bad samples do not dither around the pi/2 flip threshold, they
# teleport across it -- median error at the flip tick was 2.6 rad -- so the band
# needed to reject them is wider than pi, which is the same as never flipping at
# all and costs a 180 deg re-aim on every direction change. The defect is in the
# measurement, so it is fixed in the measurement, once, for all three consumers.
STEER_MAX_MEAS_RATE = 7.0  # rad/s

# Ceiling on the travel budget the gate will let accrue between moves of its
# estimate, as a time: 3 Period5 frame periods, the measured p90 gap between
# frames (median 20 ms, p90 60 ms -- the occasional dropped frame).
#
# Without it the budget grows without bound while a module sits still, because
# a reading identical to the estimate never moves it and so never restarts the
# clock. Measured on the 2026-08-27 logs, a parked module holds a bit-identical
# reading for up to 820 ms, which would have left 5.7 rad of budget waiting for
# whatever sample came next -- and parked-then-moving is exactly when the base
# sets off, which is exactly when the bad frames appear.
#
# Capping is sound because identical frames from a parked module corroborate
# its position rather than hiding motion behind it: no travel is accumulating
# unobserved, so no budget should accumulate either. Well clear of what real
# motion needs -- a module at the 6.2 rad/s it can actually reach covers
# 0.12 rad per frame against the 0.42 rad this allows.
STEER_MAX_BUDGET_S = 0.060

# ----------------------------
# Base velocity ramp
# ----------------------------
# The profiler in Base.control_loop is a jerk-limited velocity ramp: it
# integrates acceleration toward the commanded velocity, and the acceleration
# itself may only change by BASE_MAX_JERK per second. That is what rounds the
# edges off a step command without a low-pass filter -- a filter lags *every*
# sample by its time constant, including the steady-state ones, whereas a ramp
# is only behind while the command is actually changing and tracks exactly once
# it settles.
#
# Replaces the raised-cosine S-curve that planned a fixed-duration segment and
# re-planned it whenever the target moved. Reading the live target every tick
# is most of the responsiveness: a command that changes mid-move is picked up
# on the next tick instead of after the current segment finishes.
#
# Tuned for low perceived latency rather than maximum smoothness:
#
#   * Jerk is high. It only shapes the first ~a_max/j seconds of a move
#     (~38 ms here), enough to stop the wheels slipping and the mast rocking,
#     short enough that the operator reads it as immediate. It has to be raised
#     alongside acceleration -- the *ratio* sets how long the rounding lasts,
#     so bumping accel alone makes a move feel softer, not snappier.
#   * Acceleration is a traction and tip-over budget, not a comfort setting.
#   * Deceleration is deliberately stronger than acceleration (~1.6x). Starts
#     are where smoothness is noticed; stops are where latency is noticed, and
#     an operator releasing the stick wants the robot to be *done* moving.
#
# THESE ARE PAST THE POINT WHERE SOFTWARE IS THE BINDING CONSTRAINT. 7.0 m/s^2
# is ~0.71 g, and two physical ceilings sit right around there: wheel traction
# (roughly 0.7-0.9 g on a clean level floor, less on dust or a ramp), and
# tip-over about the short axis, roughly g * (half-track / CoG height) -- with
# a half-track of 0.17 m that shrinks as the lift goes up.
#
# The ramp only shapes the *command*; the SPARK velocity PID still has to
# follow it, and under config/base_pid_stock.json it does not -- that loop is
# P-only and reaches about 40% of setpoint. Tuning these numbers cannot fix
# tracking that happens downstream of them.
#
# Order is [vx, vy, omega]: m/s^2 and rad/s^2 here, m/s^3 and rad/s^3 for jerk.
BASE_MAX_ACCEL = np.array([4.5, 4.5, 14.0], dtype=float)
BASE_MAX_DECEL = np.array([7.0, 7.0, 22.0], dtype=float)
BASE_MAX_JERK = np.array([120.0, 120.0, 400.0], dtype=float)


# ----------------------------
# Math helpers
# ----------------------------
def wrap_pi(a: np.ndarray) -> np.ndarray:
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def diff_angle(a: np.ndarray, b: Union[np.ndarray, float]) -> np.ndarray:
    return ((a - b) + math.pi) % (2 * math.pi) - math.pi


def frac_to_rad(f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return ((np.array(f) + 0.5) % 1.0 - 0.5) * TWO_PI


def rad_to_frac(rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return (np.array(rad) / TWO_PI) % 1.0


# ----------------------------
# Pico lift (serial)
# ----------------------------
class PicoLift:
    """Host driver for firmware/lift_controller/lift_controller.ino.

    Wire protocol (see the sketch for the authoritative list):

        sent      up | down | stop | home | "up <mm>" | "down <mm>"
                  "vel <signed mm/s>" | status | power on | power off
        received  "Capabilities: lift_velocity_v1"  (banner and status)
                  "Height: <n> mm"            at 36 Hz *while moving*
                  "Height: unknown (run home)"  position not established
                  "Home complete." / "Home failed: ..." / "Home stopped."
                  "Move complete." / "LIMIT HIT: ..." / "Motion stopped..."
                  "Upper limit: ACTIVE|clear" / "Lower limit: ..."  (status)
                  "Motion: IDLE|UP|DOWN"                            (status)

    Two consequences of that protocol worth knowing:

    * **Height is only streamed while the lift is moving.** When idle the last
      value stands, which is correct but means `get_height()` returns None
      until the first move or home after a boot away from a limit switch.
    * **Position can become unknown.** The firmware says so explicitly, and we
      must clear the cached height when it does — otherwise a controller reset
      leaves the host confidently reporting a stale height that
      `lift_to_height` would then act on.

    Streamed velocity (`set_velocity_mm_s`) is only usable against a firmware
    that says so: `supports_velocity()` reflects the controller's own
    capability line, not the presence of this method. An older sketch answers
    "vel" with its usage banner and keeps sitting still, so a host that assumed
    support would silently command nothing.
    """

    _HEIGHT_PATTERN = re.compile(r"Height:\s*(-?[\d.]+)\s*mm")
    _HEIGHT_UNKNOWN_PATTERN = re.compile(r"Height:\s*unknown", re.IGNORECASE)
    _UPPER_LIMIT_PATTERN = re.compile(r"Upper limit:\s*(ACTIVE|clear)", re.IGNORECASE)
    _LOWER_LIMIT_PATTERN = re.compile(r"Lower limit:\s*(ACTIVE|clear)", re.IGNORECASE)
    _MOTION_PATTERN = re.compile(r"Motion:\s*(IDLE|UP|DOWN)", re.IGNORECASE)
    _CAPABILITIES_PATTERN = re.compile(r"Capabilities:\s*(.+)", re.IGNORECASE)
    # The firmware banner. Seeing it mid-run means the board reset (USB glitch,
    # watchdog, someone re-flashed) and every cached fact about it is void.
    _READY_PATTERN = re.compile(r"Lift controller ready", re.IGNORECASE)

    def __init__(
        self,
        device_path: str = LIFT_SERIAL_PORT,
        baud: int = 115200,
        timeout: float = 0.2,
    ):
        # Allow replacement hardware and bench setups to select another port
        # without editing robot code.
        self.device_path = os.environ.get("YOR_LIFT_SERIAL_PORT", device_path)
        self.baud = baud
        self.timeout = timeout

        self._ser = None
        self._lock = threading.Lock()
        self._last_cmd = None
        self._last_send_ts = 0.0
        self._min_repeat_interval = 0.05

        self._drain_thread = None
        self._drain_stop = threading.Event()

        # Streamed-velocity gating. Separate from _lock, which _send() holds.
        self._vel_lock = threading.Lock()
        self._last_velocity_mm_s: Optional[float] = None
        self._last_velocity_send_ts = 0.0

        self._height_lock = threading.Lock()
        self._height_m: Optional[float] = None
        # When that height arrived (time.monotonic). A height with no age is a
        # height nobody should servo against — see get_height_age().
        self._height_ts: Optional[float] = None
        # Capability tokens from the firmware's "Capabilities:" line. Empty
        # means it has not said, which is not the same as "no".
        self._capabilities: set[str] = set()
        # None until the firmware tells us either way. False means the lift has
        # no established zero and every height it reports is meaningless.
        self._position_known: Optional[bool] = None
        # Result of the last home: True complete, False failed/aborted.
        self._homed: Optional[bool] = None
        self._upper_limit: Optional[bool] = None
        self._lower_limit: Optional[bool] = None
        self._motion: Optional[str] = None      # "IDLE" | "UP" | "DOWN"
        self._last_event: Optional[str] = None  # last notable firmware line

        self._ensure_open()
        self._ensure_drain()

    def _ensure_open(self) -> None:
        if serial is None:
            print("[PicoLift] pyserial not installed; lift disabled")
            return
        if self._ser is not None and getattr(self._ser, "is_open", False):
            return
        try:
            print(f"[PicoLift] Opening serial {self.device_path} @ {self.baud}...")
            self._ser = serial.Serial(
                self.device_path,
                self.baud,
                timeout=self.timeout,
                write_timeout=0.02,
                rtscts=False,
                dsrdtr=False,
                xonxoff=False,
            )
            try:
                self._ser.reset_input_buffer()
                self._ser.reset_output_buffer()
            except Exception:
                pass
            print("[PicoLift] Serial open OK")
        except Exception as e:
            print(f"[PicoLift] Failed to open {self.device_path}: {e}")
            self._ser = None

    def _ensure_drain(self) -> None:
        if serial is None:
            return
        if self._drain_thread is not None and self._drain_thread.is_alive():
            return
        self._drain_stop.clear()
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def _shutdown(self) -> None:
        """Stop reconnecting and close the serial device during node exit."""
        self._drain_stop.set()

        # Stop the mechanism before closing an active link.  Write directly so
        # shutdown can never trigger _send()'s reconnect path.
        with self._lock:
            if self._ser is not None and getattr(self._ser, "is_open", False):
                try:
                    self._ser.write(b"\r\nstop\r\n")
                    self._ser.flush()
                except Exception:
                    pass

        if self._drain_thread is not None and self._drain_thread.is_alive():
            self._drain_thread.join(timeout=max(1.0, self.timeout + 0.5))

        with self._lock:
            try:
                if self._ser is not None:
                    self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _drain_loop(self) -> None:
        while not self._drain_stop.is_set():
            try:
                if self._ser is None or not getattr(self._ser, "is_open", False):
                    self._ensure_open()
                    time.sleep(0.1)
                    continue

                data = self._ser.readline()
                if data:
                    try:
                        line = data.decode("utf-8").strip()
                        # "Height:" streams at 36 Hz during a move; logging every
                        # one of them buries everything else in the node's output.
                        if not self._HEIGHT_PATTERN.search(line):
                            print(f"[LIFT] {line}")
                        self._parse_line(line)
                    except Exception:
                        print(f"[LIFT] Raw: {data}")

            except Exception:
                try:
                    if self._ser:
                        self._ser.close()
                except Exception:
                    pass
                self._ser = None
                time.sleep(0.1)

    def _parse_line(self, line: str) -> None:
        """Fold one firmware line into the cached lift state.

        Ordering matters: the "unknown" form is checked before the numeric one
        so a position loss can never be mistaken for a reading, and the reset
        banner is checked first of all because it invalidates everything.
        """
        if self._READY_PATTERN.search(line):
            with self._height_lock:
                self._height_m = None
                self._height_ts = None
                self._position_known = None
                self._homed = None
                self._motion = "IDLE"
                self._last_event = "controller reset"
                # A reset board may be a *different* build. It re-announces its
                # capabilities on the next line; until then we know nothing.
                self._capabilities = set()
            with self._vel_lock:
                self._last_velocity_mm_s = None
                self._last_velocity_send_ts = 0.0
            print("[PicoLift] controller reset — height invalidated, needs home")
            return

        match = self._CAPABILITIES_PATTERN.search(line)
        if match:
            tokens = {t.strip().lower() for t in match.group(1).split() if t.strip()}
            with self._height_lock:
                self._capabilities |= tokens
            print(f"[PicoLift] firmware capabilities: {', '.join(sorted(tokens))}")
            return

        if self._HEIGHT_UNKNOWN_PATTERN.search(line):
            with self._height_lock:
                self._height_m = None
                self._height_ts = None
                self._position_known = False
            return

        match = self._HEIGHT_PATTERN.search(line)
        if match:
            with self._height_lock:
                self._height_m = float(match.group(1)) / 1000.0
                self._height_ts = time.monotonic()
                self._position_known = True
            return

        match = self._UPPER_LIMIT_PATTERN.search(line)
        if match:
            with self._height_lock:
                self._upper_limit = match.group(1).lower() == "active"
            return

        match = self._LOWER_LIMIT_PATTERN.search(line)
        if match:
            with self._height_lock:
                self._lower_limit = match.group(1).lower() == "active"
            return

        match = self._MOTION_PATTERN.search(line)
        if match:
            with self._height_lock:
                self._motion = match.group(1).upper()
            return

        lowered = line.lower()
        if lowered.startswith("home complete"):
            with self._height_lock:
                self._homed = True
                self._position_known = True
                self._last_event = line
        elif lowered.startswith("home failed") or lowered.startswith("home stopped"):
            # The firmware clears positionKnown in both cases, so we must too —
            # otherwise a failed home leaves a plausible-looking stale height.
            with self._height_lock:
                self._homed = False
                self._position_known = False
                self._height_m = None
                self._height_ts = None
                self._last_event = line
        elif (lowered.startswith("move complete")
              or lowered.startswith("motion stopped")
              or lowered.startswith("limit hit")
              or "blocked" in lowered):
            with self._height_lock:
                self._motion = "IDLE"
                self._last_event = line

    def _send(self, cmd: str) -> None:
        if serial is None:
            return
        now = time.monotonic()
        with self._lock:
            self._ensure_open()
            if self._ser is None:
                print(f"[PicoLift] Not connected; drop cmd '{cmd}'")
                return

            if (
                cmd != "stop"
                and cmd == self._last_cmd
                and (now - self._last_send_ts) < self._min_repeat_interval
            ):
                return

            try:
                payload = (cmd + "\n").encode()
                if cmd == "stop":
                    try:
                        self._ser.write(b"\r\n")
                    except Exception:
                        pass
                    payload = (cmd + "\r\n").encode()

                self._ser.write(payload)

                if cmd == "stop":
                    try:
                        self._ser.flush()
                    except Exception:
                        pass

                self._last_cmd = cmd
                self._last_send_ts = now
                print(f"[PicoLift] sent '{cmd}'")

            except Exception as e:
                print(f"[PicoLift] write error: {e}")
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    def up(self) -> None:
        self._send("up")

    def down(self) -> None:
        self._send("down")

    def home(self) -> None:
        self._send("home")

    def stop(self) -> None:
        self._last_cmd = None
        # A discrete stop ends velocity mode in the firmware, so the next
        # streamed command must go out immediately rather than be deduped
        # against a value the controller has already forgotten.
        with self._vel_lock:
            self._last_velocity_mm_s = None
        self._send("stop")

    def move_mm(self, distance_mm: float, up: bool) -> bool:
        """Ask the firmware for a finite move of `distance_mm`.

        This is the "up 200" / "down 200" form. The firmware runs its own
        jerk-limited S-curve to an exact pulse count and stops itself, which is
        strictly better than the host bang-banging `up`/`stop` over serial: it
        gets a real acceleration profile and there is no round-trip latency in
        the stop. Returns False for a non-positive distance rather than sending
        a command the firmware would reject.
        """
        distance_mm = float(distance_mm)
        if not (distance_mm > 0.0) or not math.isfinite(distance_mm):
            return False
        self._last_cmd = None   # never dedupe a finite move against a previous one
        self._send(f"{'up' if up else 'down'} {distance_mm:.2f}")
        return True

    def set_velocity_mm_s(self, velocity_mm_s: float) -> bool:
        """Stream a signed velocity to the firmware: + is up, - is down.

        Millimetres per second is the wire unit; the rest of the robot works in
        metres, so `Base.lift_set_velocity()` is the converting entry point and
        this is the transport.

        Returns True when the value was accepted — which is not the same as
        "a frame was written". A command identical to the last one is only
        refreshed every LIFT_VELOCITY_KEEPALIVE_S, because the firmware treats
        a steady stream as proof the host is alive rather than as new
        information. Three things always go out immediately:

        * a meaningful change (>= the firmware's minimum active velocity),
        * any transition to or from zero, so a stop is never delayed,
        * the first command after a reset or a mode change.

        False means nothing was sent and nothing will be: the value was not a
        finite number.
        """
        try:
            velocity_mm_s = float(velocity_mm_s)
        except (TypeError, ValueError):
            print(f"[PicoLift] refusing non-numeric velocity {velocity_mm_s!r}")
            return False

        if not math.isfinite(velocity_mm_s):
            print(f"[PicoLift] refusing non-finite velocity {velocity_mm_s!r}")
            return False

        velocity_mm_s = float(np.clip(
            velocity_mm_s, -LIFT_MAX_VELOCITY_MM_S, LIFT_MAX_VELOCITY_MM_S))
        if abs(velocity_mm_s) < LIFT_MIN_VELOCITY_MM_S:
            velocity_mm_s = 0.0

        now = time.monotonic()
        with self._vel_lock:
            last = self._last_velocity_mm_s
            if last is None:
                send = True
            elif velocity_mm_s == 0.0 and last != 0.0:
                send = True                      # stopping is never deferred
            elif abs(velocity_mm_s - last) >= LIFT_MIN_VELOCITY_MM_S:
                send = True                      # a change the lift can express
            else:
                send = (now - self._last_velocity_send_ts) >= LIFT_VELOCITY_KEEPALIVE_S

            if send:
                self._last_velocity_mm_s = velocity_mm_s
                self._last_velocity_send_ts = now

        if send:
            # Bypass _send()'s generic repeat suppression: the gate above is
            # this command's rate limiter, and a suppressed keepalive would
            # look to the firmware like a lost host.
            self._last_cmd = None
            self._send(f"vel {velocity_mm_s:.2f}")
        return True

    def supports_velocity(self) -> bool:
        """Whether the *connected firmware* advertised streamed velocity.

        This reads the controller's capability line, so it stays False against
        an older sketch that has no "vel" command — which is the whole point:
        the presence of set_velocity_mm_s() proves nothing about the board.
        """
        with self._height_lock:
            return LIFT_VELOCITY_CAPABILITY in self._capabilities

    def get_capabilities(self) -> set:
        """Every capability token the firmware has announced."""
        with self._height_lock:
            return set(self._capabilities)

    def get_height_age(self) -> Optional[float]:
        """Seconds since the last height line, or None if there has been none.

        Height is only streamed while the lift is moving or holding in velocity
        mode, so an old age is normal for a parked lift. It matters to anything
        that servos against the measurement: a control loop must refuse to act
        on a height that stopped arriving.
        """
        with self._height_lock:
            if self._height_ts is None:
                return None
            return time.monotonic() - self._height_ts

    def request_status(self) -> None:
        """Ask for a status report; the reply lands in the cached state."""
        self._last_cmd = None
        self._send("status")

    def set_power(self, on: bool) -> None:
        """Explicit driver-relay control. The firmware also cuts power itself
        after a stop, a limit hit or a home, so this is only for parking."""
        self._last_cmd = None
        self._send("power on" if on else "power off")

    def get_height(self) -> Optional[float]:
        """Height in metres, or None when the firmware has no established zero."""
        with self._height_lock:
            return self._height_m

    def is_position_known(self) -> Optional[bool]:
        """True/False once the firmware has said either way, else None."""
        with self._height_lock:
            return self._position_known

    def is_homed(self) -> Optional[bool]:
        """True after "Home complete.", False after a failed or aborted home."""
        with self._height_lock:
            return self._homed

    def get_limits(self) -> Tuple[Optional[bool], Optional[bool]]:
        """(upper, lower) switch states as of the last `status` reply."""
        with self._height_lock:
            return self._upper_limit, self._lower_limit

    def get_motion(self) -> Optional[str]:
        """"IDLE" | "UP" | "DOWN" as last reported, or None."""
        with self._height_lock:
            return self._motion

    def get_last_event(self) -> Optional[str]:
        """The last notable firmware line (completion, limit, home result)."""
        with self._height_lock:
            return self._last_event


# ----------------------------
# Commands
# ----------------------------
class CommandType(IntEnum):
    BASE_VELOCITY = 1
    BASE_POSITION = 2
    LIFT_POSITION = 3


# ----------------------------
# SparkFlex wrappers
# ----------------------------
class RotationMotor:
    """Rotation motor driven by SparkFlex position setpoint (fraction 0..1)."""

    def __init__(self, can_if: str, can_id: int, offset_frac: float = 0.0):
        self.dev = SparkFlex(can_if, can_id)
        self.can_id = int(can_id)
        self.offset = float(offset_frac)
        self.last_cmd_frac = 0.0

        try:
            if IdleMode and hasattr(self.dev, "SetIdleMode"):
                self.dev.SetIdleMode(IdleMode.kCoast)
        except Exception:
            pass

    def heartbeat(self) -> None:
        self.dev.Heartbeat()

    def set_position_fraction(self, frac_0_1: float) -> None:
        f = (frac_0_1 + self.offset) % 1.0
        self.last_cmd_frac = f
        self.dev.SetPosition(float(f))

    def get_absolute_turns(self) -> float:
        """Raw absolute-encoder reading, in TURNS.

        `GetAbsoluteEncoderPosition` already returns turns (0..1), not degrees
        -- the same fact the commissioned calibration in the previous codebase
        records as `steer_turns_per_raw_unit: 1.0`. Everything here used to
        divide it by 360, which collapsed every reading to nearly zero and made
        the reported angle a constant fixed by the module offset. Confirmed
        against the 2026-08-22 logs: the four modules reported 90.0, 0.0, -90.0
        and -180.0 degrees with a total spread of 1 degree, while the commanded
        angles swept the full 360.
        """
        try:
            turns = float(self.dev.GetAbsoluteEncoderPosition())
        except Exception:
            return float("nan")
        return turns if math.isfinite(turns) else float("nan")

    def get_position_deg(self) -> float:
        turns = self.get_absolute_turns()
        return float("nan") if math.isnan(turns) else turns * 360.0

    def get_position_rad(self) -> float:
        if not USE_FEEDBACK_FOR_STEER:
            frac_no_off = (self.last_cmd_frac - self.offset) % 1.0
            return float(frac_to_rad(frac_no_off))

        measured = self.get_absolute_rad()
        if math.isnan(measured):
            # Fall back to the last *command*, in the same offset-removed frame
            # the measurement uses. Returning the raw fraction here (as this
            # branch used to) would hand the caller an angle in a different
            # frame than every other reading, off by the module offset.
            return float(frac_to_rad((self.last_cmd_frac - self.offset) % 1.0))
        return measured

    def get_absolute_rad(self) -> float:
        """Measured steering angle from the absolute encoder, offset removed.

        Deliberately separate from `get_position_rad`, which is the *control
        path* and returns the last commanded angle while USE_FEEDBACK_FOR_STEER
        is False. Telemetry must not inherit that substitution: logging the
        commanded angle as if it were measured would make every module look
        like it tracks perfectly. NaN if the encoder frame has not arrived.
        """
        turns = self.get_absolute_turns()
        if math.isnan(turns):
            return float("nan")
        frac_no_off = ((turns % 1.0) - self.offset) % 1.0
        return float(frac_to_rad(frac_no_off))

    def get_position_counts(self) -> float:
        try:
            return float(self.dev.GetPosition())
        except Exception:
            return float("nan")


class DriveMotor:
    """Drive motor driven by SparkFlex velocity setpoint (PID on controller)."""

    def __init__(self, can_if: str, can_id: int, vel_scale: float = DRIVE_VEL_SCALE):
        self.dev = SparkFlex(can_if, can_id)
        self.can_id = int(can_id)
        self.vel_scale = float(vel_scale)

        if IdleMode and hasattr(self.dev, "SetIdleMode"):
            try:
                self.dev.SetIdleMode(IdleMode.kCoast)
            except Exception:
                pass
                
        if CtrlType and hasattr(self.dev, "SetCtrlType"):
            try:
                self.dev.SetCtrlType(CtrlType.kVelocity)
            except Exception:
                pass
                
        # Request Status Frame 2 (velocity + position) at 20 ms
        if hasattr(self.dev, "SetPeriodicStatus2Period"):
            try:
                self.dev.SetPeriodicStatus2Period(20)
            except Exception:
                pass

    def heartbeat(self) -> None:
        self.dev.Heartbeat()

    def set_velocity_mps(self, v_mps: float) -> None:
        self.dev.SetVelocity(float(v_mps * self.vel_scale))

    def get_velocity_raw(self) -> float:
        try:
            return float(self.dev.GetVelocity())
        except Exception:
            return float("nan")

    def get_position_counts(self) -> float:
        try:
            return float(self.dev.GetPosition())
        except Exception:
            return float("nan")


# ----------------------------
# Base (swerve control)
# ----------------------------
class Base:
    def __init__(
        self,
        max_vel=np.array((1.0, 1.0, 1.57)),
        max_accel=np.array((1.0, 1.0, 1.57)),
        drive_vel_scale: float = DRIVE_VEL_SCALE,
    ):
        self.max_vel = max_vel
        self.max_accel = max_accel
        # Travels with the PID manifest, because it is only correct for a
        # particular set of drive gains -- see the DRIVE_VEL_SCALE comment.
        self.drive_vel_scale = float(drive_vel_scale)

        # NOTE: there used to be a forward-kinematics matrix `self.C` and an
        # `_angle_and_speed_to_vehicle_velocity` here. Both were dead, and the
        # matrix had the wrong sign on the omega->vx coupling: round-tripping a
        # pure spin of +1.0 rad/s came back as -0.316. The correct forward
        # model, with geometry calibrated against measured motion rather than
        # CAD, is robot/nav/odometry/swerve_odom.py -- use that.

        self.rotation_motors = [
            RotationMotor(drivetrain_can, CAN_IDS_ROT[i], ROTATION_OFFSETS[i])
            for i in range(NUM_SWERVES)
        ]
        self.drive_motors = [
            DriveMotor(drivetrain_can, CAN_IDS_DRIVE[i], self.drive_vel_scale)
            for i in range(NUM_SWERVES)
        ]

        self._pico_lift = PicoLift()

        self.steer_pos = np.zeros(NUM_SWERVES)
        self.drive_vel = np.zeros(NUM_SWERVES)

        # --- Steering measurement gate (see _gate_steer_measurement) ---
        # `_gate_prev` is the current estimate, `_move_t` when each module's
        # estimate last moved (the clock the rate budget accrues on), and
        # `_rejects` a per-module count of clamped samples for the periodic
        # health line. Seeded on the first read rather than here, because zero
        # is a real angle and filtering toward it would fight startup homing.
        self._steer_gate_seeded = False
        self._steer_gate_prev = np.zeros(NUM_SWERVES)
        self._steer_move_t = np.zeros(NUM_SWERVES)
        self._steer_rejects = np.zeros(NUM_SWERVES, dtype=np.int64)
        self._steer_gate_last_report = 0.0
        self.x = np.zeros(3)
        self.dx = np.zeros(3)

        # What the control loop last asked each module for, alongside the
        # measured state above. Written only by control_loop; read by
        # swerve_telemetry from other threads, which is safe because every
        # write replaces the whole array rather than mutating it in place.
        self.steer_cmd = np.zeros(NUM_SWERVES)
        self.drive_cmd = np.zeros(NUM_SWERVES)
        self._motors_enabled = False

        self._command_queue: Queue[dict[str, Any]] = Queue(3)
        self.base_target = np.zeros(3)

        self.control_loop_thread: Optional[threading.Thread] = threading.Thread(
            target=self.control_loop, daemon=True
        )
        self.control_loop_running = False

        self._last_loop_time = time.monotonic()
        self._loop_dt = CONTROL_PERIOD

        # --- Jerk-limited ramp state (optional per-command) ---
        self._smooth_active = False  # whether to apply the ramp to the *current* command
        self._v_prof = np.zeros(3, dtype=float)  # velocity the wheels are being given
        self._a_prof = np.zeros(3, dtype=float)  # acceleration currently being held

        self._a_max_accel = BASE_MAX_ACCEL.copy()
        self._a_max_decel = BASE_MAX_DECEL.copy()
        self._j_max = BASE_MAX_JERK.copy()

    def swerve_devices(self) -> dict[int, Any]:
        """{CAN id: SparkFlex} for all eight swerve controllers.

        The handles this process already owns. robot/yor.py writes the
        commissioned PID gains through these at startup (see
        tools/base_pid_preflight.sync_from_manifest) rather than opening a
        second set of SparkFlex objects on the same bus, which is not safe.
        """
        return {motor.can_id: motor.dev
                for motor in (*self.rotation_motors, *self.drive_motors)}

    def swerve_configuration(self) -> dict:
        """What the controllers report about how they are configured.

        None of this is set by this file except idle mode and control type, and
        those two only started being set once the enum import above was fixed.
        Conversion factors in particular are configured out-of-band through the
        REV Hardware Client and live in SPARK flash, which is why
        DRIVE_VEL_SCALE reads as a magic number: it is one half of a unit
        conversion whose other half is not in this repository.

        Printing it at startup is the cheapest way to stop that being invisible.
        `velocity_cf` is the number that decides whether `set_velocity_mps`
        speaks true m/s:

            metres per motor rotation   0.049922  (calibrated, swerve_odom.py)
            => velocity_cf for true m/s 0.00083203
            => the same via DIAMETER    0.00166407  (exactly 2x -- see
               docs/BASE_COMMAND_LOOP_REVIEW.md finding 6)

        Reads are parameter round-trips, not cached status frames, so this is
        a startup call -- not something to put in a control loop.
        """
        def read(device, name):
            getter = getattr(device, name, None)
            if getter is None:
                return None
            try:
                return getter()
            except Exception:
                return None

        report: dict[str, dict] = {}
        for role, motors, ids in (("drive", self.drive_motors, CAN_IDS_DRIVE),
                                  ("steering", self.rotation_motors, CAN_IDS_ROT)):
            for module, motor in zip(MODULE_ORDER, motors):
                report[f"{module} {role}"] = {
                    "can_id": motor.can_id,
                    "idle_mode": read(motor.dev, "GetIdleModeRaw"),
                    "ctrl_type": read(motor.dev, "GetCtrlType"),
                    "velocity_cf": read(motor.dev, "GetVelocityConversionFactor"),
                    "position_cf": read(motor.dev, "GetPositionConversionFactor"),
                }
        return report

    def swerve_telemetry(self) -> dict:
        """One snapshot of what the four modules were asked for and report back.

        Everything here is cheap: the commanded arrays are already in memory,
        and the two Get* calls read a cached periodic-status frame under a
        mutex rather than putting a request on the bus (SparkBase::GetVelocity
        and GetAbsoluteEncoderPosition read period2_/period5_). So this is safe
        to call from the 30 Hz whole-body loop without adding CAN traffic.

        Module order is MODULE_ORDER: FL, FR, RR, RL.

        `v_target` is the velocity the loop was last handed; `v_profiled` is
        what the jerk-limited ramp had actually reached when smoothing is on.
        The two diverge while the ramp is converging on a changed command, and
        that gap is the ramp's contribution -- worth logging separately from
        the command, because tuning the base against `v_target` alone
        attributes the ramp's lag to the wheels.
        """
        return {
            "steer_cmd_rad": np.asarray(self.steer_cmd, dtype=float).copy(),
            "steer_meas_rad": np.array(
                [m.get_absolute_rad() for m in self.rotation_motors], dtype=float),
            "drive_cmd_mps": np.asarray(self.drive_cmd, dtype=float).copy(),
            "drive_meas_raw": np.array(
                [m.get_velocity_raw() for m in self.drive_motors], dtype=float),
            # Cumulative motor rotations. Logged alongside velocity because it
            # is the better distance signal: a counter does not accumulate the
            # sampling error that integrating a 30 Hz-sampled velocity does,
            # and it is what robot/nav/odometry/swerve_odom.py integrates. With
            # it in the trajectory log, wheel odometry can be reconstructed
            # offline and compared against BaseOdometry's commanded-velocity
            # dead reckoning without another tape run. Same cached status frame
            # 2 as GetVelocity, so it costs no extra bus traffic.
            "drive_pos_rot": np.array(
                [m.get_position_counts() for m in self.drive_motors], dtype=float),
            "v_target": np.asarray(self.base_target, dtype=float).copy(),
            "v_profiled": np.asarray(self._v_prof, dtype=float).copy(),
            "motors_enabled": bool(self._motors_enabled),
        }

    # --- Lift controls ---
    def lift_up(self) -> None:
        if self._pico_lift:
            self._pico_lift.up()

    def lift_down(self) -> None:
        if self._pico_lift:
            self._pico_lift.down()

    def lift_home(self) -> None:
        if self._pico_lift:
            self._pico_lift.home()

    def lift_stop(self) -> None:
        if self._pico_lift:
            self._pico_lift.stop()

    def get_lift_height(self) -> Optional[float]:
        if self._pico_lift:
            return self._pico_lift.get_height()
        return None

    def lift_set_velocity(self, velocity_m_s: float) -> bool:
        """Stream a lift velocity in metres per second: + is up, - is down.

        The wire protocol is millimetres per second; this is where the robot's
        unit becomes the controller's. Returns False when the value was not a
        finite number, or when there is no lift attached — never because the
        command was merely rate-limited.

        Only call this against a firmware that reports the capability; check
        `lift_supports_velocity()` first. An older controller answers "vel"
        with its usage banner and does not move.
        """
        if self._pico_lift is None:
            return False
        try:
            metres_per_second = float(velocity_m_s)
        except (TypeError, ValueError):
            return False
        return bool(self._pico_lift.set_velocity_mm_s(metres_per_second * 1000.0))

    def lift_supports_velocity(self) -> bool:
        """Whether the attached lift firmware advertised streamed velocity."""
        if self._pico_lift is None:
            return False
        return bool(self._pico_lift.supports_velocity())

    def get_lift_height_age(self) -> Optional[float]:
        """Seconds since the last height telemetry line, or None if none yet."""
        if self._pico_lift is None:
            return None
        return self._pico_lift.get_height_age()

    def lift_position_known(self) -> Optional[bool]:
        """Whether the firmware has an established zero. None if not yet known.

        False means every height it reports is meaningless — run lift_home().
        """
        if self._pico_lift:
            return self._pico_lift.is_position_known()
        return None

    def get_lift_status(self) -> dict:
        """Snapshot of the lift for diagnostics / RPC.

        Requests a fresh `status` from the firmware, which is what refreshes the
        limit-switch fields. Those are only populated after a status reply, so
        the first call after boot may return None for them.
        """
        if self._pico_lift is None:
            return {"available": False}
        self._pico_lift.request_status()
        upper, lower = self._pico_lift.get_limits()
        return {
            "available": True,
            "height_m": self._pico_lift.get_height(),
            "height_age_s": self._pico_lift.get_height_age(),
            "position_known": self._pico_lift.is_position_known(),
            "homed": self._pico_lift.is_homed(),
            "upper_limit": upper,
            "lower_limit": lower,
            "motion": self._pico_lift.get_motion(),
            "last_event": self._pico_lift.get_last_event(),
            "velocity_capable": self._pico_lift.supports_velocity(),
            "capabilities": sorted(self._pico_lift.get_capabilities()),
        }

    def lift_delta_height(
        self,
        delta_m: float,
        tolerance_m: float = 0.002,
        timeout_s: float = 30.0,
        min_height_m: float = 0.0,
        max_height_m: float = LIFT_MAX_HEIGHT_M,
    ) -> bool:
        """
        Move lift up/down by delta in meters (positive=up, negative=down).
        Returns True if target reached within tolerance, False otherwise.
        """
        if self._pico_lift is None:
            return False

        current_height = self._pico_lift.get_height()
        if current_height is None:
            print("[lift_delta_height] Lift height unknown; cannot move by delta")
            return False

        return self.lift_to_height(
            target_m=current_height + float(delta_m),
            tolerance_m=tolerance_m,
            timeout_s=timeout_s,
            min_height_m=min_height_m,
            max_height_m=max_height_m,
        )


    def lift_to_height(
        self,
        target_m: float,
        tolerance_m: float = 0.002,
        timeout_s: float = 30.0,
        min_height_m: float = 0.0,
        max_height_m: float = LIFT_MAX_HEIGHT_M,
        profiled: bool = True,
    ) -> bool:
        """
        Move lift to an absolute height position (blocking).
        Returns True if target reached within tolerance, False on timeout/stall/unknown height.

        With `profiled` (the default) the distance is handed to the firmware as
        a single "up <mm>" / "down <mm>" command, so the move runs the firmware's
        jerk-limited S-curve and stops itself on an exact pulse count. The old
        behaviour — start a continuous move, watch the height, send `stop` — is
        still available with `profiled=False`; it is at the mercy of serial
        round-trip latency at the stop, and re-accelerates from zero on every
        correction because the firmware cuts driver power after a user stop.

        The supervision below is unchanged and applies either way: it is the
        backstop for a firmware that never reports arrival.

        Safety features:
        - clamps target within [min_height_m, max_height_m]
        - timeout
        - overshoot stop
        - stall detection
        """
        if self._pico_lift is None:
            return False

        # Clamp target
        target_m = float(max(min_height_m, min(max_height_m, float(target_m))))

        current_height = self._pico_lift.get_height()
        if current_height is None:
            print("[lift_to_height] Lift height unknown; cannot move to target")
            return False

        # A height only means something once the firmware has a zero. Without
        # this check a stale reading from before a controller reset would send
        # the lift off in a plausible-looking but wrong direction.
        if self._pico_lift.is_position_known() is False:
            print("[lift_to_height] Lift position not established; run lift_home() first")
            return False

        error = target_m - float(current_height)
        if abs(error) <= tolerance_m:
            return True

        moving_up = error > 0.0
        if profiled:
            # Aim at the target directly; the loop below still corrects for any
            # residual, which costs a second (much shorter) profiled move.
            if not self._pico_lift.move_mm(abs(error) * 1000.0, up=moving_up):
                return False
        else:
            (self.lift_up if moving_up else self.lift_down)()

        rate = RateLimiter(60)
        start_time = time.monotonic()

        last_height = float(current_height)
        stall_start: Optional[float] = None
        # A profiled move ends by itself, so "stopped short of target" is a
        # normal outcome (pulse quantisation, a little sag under load) rather
        # than a fault. Re-aim a bounded number of times before calling it a
        # stall; each retry is a much shorter move than the first.
        corrections = 0
        MAX_CORRECTIONS = 3

        try:
            while True:
                now = time.monotonic()
                if (now - start_time) > timeout_s:
                    print(f"[lift_to_height] Timeout after {timeout_s:.1f}s")
                    self.lift_stop()
                    return False

                height = self._pico_lift.get_height()
                if height is None:
                    # If we lost telemetry, safest is stop and fail
                    print("[lift_to_height] Lost height telemetry; stopping")
                    self.lift_stop()
                    return False

                height = float(height)
                error = target_m - height

                # Within tolerance
                if abs(error) <= tolerance_m:
                    self.lift_stop()
                    return True

                # Overshoot detection
                if (moving_up and height > target_m) or ((not moving_up) and height < target_m):
                    self.lift_stop()
                    return True

                # Stall detection (no meaningful movement for >1s)
                if abs(height - last_height) < 0.0005:
                    if stall_start is None:
                        stall_start = now
                    elif (now - stall_start) > 1.0:
                        # A profiled move that has simply finished short looks
                        # identical to a stall from here. Re-aim before failing.
                        if profiled and corrections < MAX_CORRECTIONS:
                            corrections += 1
                            moving_up = error > 0.0
                            print(f"[lift_to_height] {error * 1000.0:+.1f} mm short; "
                                  f"correction {corrections}/{MAX_CORRECTIONS}")
                            if not self._pico_lift.move_mm(abs(error) * 1000.0, up=moving_up):
                                self.lift_stop()
                                return False
                            stall_start = None
                            last_height = height
                            rate.sleep()
                            continue
                        print(f"[lift_to_height] Stall detected at {height:.4f} m")
                        self.lift_stop()
                        return False
                else:
                    stall_start = None
                    last_height = height

                rate.sleep()

        except KeyboardInterrupt:
            self.lift_stop()
            return False
        except Exception as e:
            print(f"[lift_to_height] Error: {e}")
            self.lift_stop()
            return False


    # def lift_delta_height(self, delta_m: float) -> None:
    #     if self._pico_lift is None:
    #         return
    #     current_height = self._pico_lift.get_height()
    #     if current_height is None:
    #         print("Lift height unknown; cannot move by delta")
    #         return

    #     target_height = current_height + delta_m
    #     if target_height <= 0.0:
    #         target_height = 0.0
    #         if delta_m < 0.0:
    #             return

    #     if target_height > current_height:
    #         self.lift_up()
    #     else:
    #         self.lift_down()

    #     rate = RateLimiter(60)
    #     while True:
    #         height = self._pico_lift.get_height()
    #         if height is None:
    #             break
    #         if (delta_m > 0 and height >= target_height) or (
    #             delta_m < 0 and height <= target_height
    #         ):
    #             break
    #         rate.sleep()

    #     self.lift_stop()

    # --- Public API ---
    def start_control(self):
        if self.control_loop_thread is None:
            print("To initiate a new control loop, create a new Base() instance first")
            return
        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self):
        if self.control_loop_thread is None:
            print("Control loop not running")
            return
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None

    def set_target_base_velocity(self, target: np.ndarray, smooth: bool = False):
        """target: np.array([vx, vy, omega]) in vehicle frame (m/s, m/s, rad/s)"""
        self._enqueue_command(
            {
                "type": CommandType.BASE_VELOCITY,
                "target": np.array(target, dtype=float),
                "smooth": bool(smooth),
            }
        )

    # ---------------- control loop ----------------
    def control_loop(self):
        rate_limiter = RateLimiter(CONTROL_FREQ, name="base-controller")
        disable_motors = True
        last_command_time_ns = time.perf_counter_ns()

        while self.control_loop_running:
            cmd = None
            try:
                while True:
                    cmd = self._command_queue.get_nowait()
            except queue.Empty:
                pass

            if cmd is not None:
                self.base_target = np.array(cmd["target"], dtype=float)
                self._smooth_active = bool(cmd.get("smooth", False))
                last_command_time_ns = time.perf_counter_ns()
                if cmd["type"] == CommandType.BASE_VELOCITY:
                    disable_motors = False

            if (time.perf_counter_ns() - last_command_time_ns) > 2.5 * POLICY_CONTROL_PERIOD_NS:
                disable_motors = True

            for m in self.drive_motors:
                m.heartbeat()
            for m in self.rotation_motors:
                m.heartbeat()

            self._update_state()
            self._report_steer_gate(time.monotonic())

            self._motors_enabled = not disable_motors

            if disable_motors:
                for d in self.drive_motors:
                    d.set_velocity_mps(0.0)
                # Steering setpoints are deliberately left standing: a disabled
                # base should stop driving, not re-aim its wheels.
                self.drive_cmd = np.zeros(NUM_SWERVES)

                # The drives have been held at zero for longer than the
                # watchdog, so the robot really is stopped. Clear the ramp too,
                # or the next command resumes from a profile that describes
                # motion that is no longer happening.
                self._v_prof[:] = 0.0
                self._a_prof[:] = 0.0

            else:
                dt = self._loop_dt
                v_cmd = self.base_target

                if self._smooth_active:
                    # No segment to re-plan: the ramp reads the live target
                    # every tick, so a command that changes mid-move is picked
                    # up on the next tick rather than after the current profile
                    # finishes. That is most of the responsiveness.
                    v_used = self._update_velocity_ramp(v_cmd, dt)
                else:
                    # Keep ramp state consistent so enabling smoothing later doesn't jump from stale state
                    self._v_prof = v_cmd.copy()
                    self._a_prof[:] = 0.0
                    v_used = v_cmd

                wheel_speeds, wheel_angles = self._vehicle_velocity_to_angle_and_speed(
                    v_used, cos_error_scaling=True
                )

                target_fracs = rad_to_frac(wheel_angles)
                for i, rm in enumerate(self.rotation_motors):
                    rm.set_position_fraction(float(target_fracs[i]))

                for i, dm in enumerate(self.drive_motors):
                    dm.set_velocity_mps(float(wheel_speeds[i]))

                self.steer_cmd = wheel_angles.copy()
                self.drive_cmd = wheel_speeds.copy()

            rate_limiter.sleep()

    # -------------- helpers --------------
    def _update_state(self) -> None:
        now = time.monotonic()
        # Real elapsed time, not the nominal period: the S-curve profiler
        # integrates with this, and on a loaded Pi the loop does not always
        # hit CONTROL_FREQ. Clamped because a pathological gap (a debugger
        # pause, a long GC) should finish the ramp, not overshoot past it.
        self._loop_dt = float(min(max(now - self._last_loop_time, 0.0),
                                  10.0 * CONTROL_PERIOD))
        self._last_loop_time = now

        raw = np.array([rm.get_position_rad() for rm in self.rotation_motors],
                       dtype=float)
        self.steer_pos = self._gate_steer_measurement(raw, now)

        for i, dm in enumerate(self.drive_motors):
            self.drive_vel[i] = dm.get_velocity_raw()

    def _gate_steer_measurement(self, raw: np.ndarray, now: float) -> np.ndarray:
        """Reject absolute-encoder readings that imply non-physical motion.

        Returns the angle the control path should believe. The estimate is
        allowed to chase the encoder, but never faster than a module can
        actually turn (see STEER_MAX_MEAS_RATE). One rule, applied per module::

            budget = STEER_MAX_MEAS_RATE * (now - when this estimate last moved)
            estimate += clip(diff_angle(reading, estimate), +/-budget)

        Measuring the budget from the last time the *estimate* moved, rather
        than per control tick, is what makes it fit the hardware. This loop runs
        at CONTROL_FREQ while Period5 frames arrive around 50 Hz, so steer_pos
        is a staircase: most ticks re-read an unchanged cache, then one tick
        carries a whole frame period of real motion at once. A per-tick budget
        would call that legitimate step implausible and clamp genuine slew.
        Accruing it while the estimate sits idle between frames means that by
        the time a frame lands there is exactly one frame period of travel
        available to spend on it, so a module slewing flat out passes through
        untouched and with no lag.

        Everything else falls out of the same rule. An isolated bad frame moves
        the estimate by at most one frame period of travel before the next good
        one pulls it back, so a jump of pi costs a few degrees of excursion for
        a few tens of milliseconds instead of reversing a wheel. And nothing
        latches: a reading that persistently disagrees is converged on at the
        physical rate, closing even a pi disagreement in under half a second, so
        an estimate that is genuinely wrong -- a module back-driven by hand
        while the base was disabled -- recovers on its own, with no timeout to
        tune and no stale value the gate can get stuck on.

        Angles are compared with diff_angle, so a module crossing +/-pi is a
        small step and not a 2pi jump.
        """
        raw = np.asarray(raw, dtype=float)
        raw = np.where(np.isfinite(raw), raw, self._steer_gate_prev)

        # First sample after construction or a reset: seed, do not filter.
        # Startup homing needs the true angle immediately, and there is no
        # previous estimate for a rate to be meaningful against.
        if not self._steer_gate_seeded:
            self._steer_gate_seeded = True
            self._steer_gate_prev = raw.copy()
            self._steer_move_t = np.full(NUM_SWERVES, float(now))
            return raw.copy()

        prev = self._steer_gate_prev
        step = diff_angle(raw, prev)

        # Budget accrues from when each module's estimate last moved, not from
        # the last tick -- see the docstring. An unchanged reading costs nothing
        # and leaves the clock running, which is what pays for the next frame;
        # STEER_MAX_BUDGET_S stops that running away while a module is parked.
        max_step = float(STEER_MAX_MEAS_RATE) * np.clip(
            now - self._steer_move_t, 0.0, STEER_MAX_BUDGET_S)

        clamped = np.abs(step) > max_step
        gated = prev + np.clip(step, -max_step, max_step)
        # Wrap once at the end: prev+delta can leave [-pi, pi) even when both
        # inputs were inside it.
        gated = diff_angle(gated, 0.0)

        self._steer_move_t = np.where(gated != prev, now, self._steer_move_t)
        self._steer_rejects += clamped
        self._steer_gate_prev = gated
        return gated.copy()

    def _report_steer_gate(self, now: float, report_every_s: float = 30.0) -> None:
        """Print how much the steering gate is vetoing, at most twice a minute.

        The gate is silent by construction -- it repairs the signal in place --
        so without this there is no way to tell a healthy CAN bus from one the
        gate is quietly papering over. A rising count on one module is the
        symptom to chase; steady zeros mean the encoders are behaving and the
        gate is costing nothing.
        """
        if now - self._steer_gate_last_report < report_every_s:
            return
        prev_t = self._steer_gate_last_report
        self._steer_gate_last_report = now
        if prev_t == 0.0:                    # first call: start the window
            self._steer_rejects[:] = 0
            return
        n = self._steer_rejects.copy()
        self._steer_rejects[:] = 0
        if not n.any():
            return
        window = now - prev_t
        per = ", ".join(f"{lbl} {int(c)}" for lbl, c in zip(MODULE_ORDER, n))
        print(f"[YOR] base: steering gate rejected {int(n.sum())} sample(s) in "
              f"{window:.0f}s ({per}) -- non-physical encoder steps above "
              f"{STEER_MAX_MEAS_RATE:.0f} rad/s")

    def _update_velocity_ramp(self, v_target: np.ndarray, dt: float) -> np.ndarray:
        """Advance the jerk-limited ramp one tick toward ``v_target``.

        Per axis this is a double integrator driven at its jerk limit::

            a += clip(a_desired - a, -j*dt, +j*dt)
            v += a * dt

        with ``a_desired`` chosen so the ramp lands *on* the target instead of
        sailing through it. Unlike a low-pass filter it converges in finite
        time and then tracks exactly, so a held command carries no lag at all.
        """
        v_target = np.asarray(v_target, dtype=float)
        v = self._v_prof
        a = self._a_prof

        dv = v_target - v

        # Is each axis being asked for more speed, or less? Anything that
        # reduces the magnitude counts as a stop -- including a reversal, which
        # has to pass through zero first -- and gets the larger deceleration
        # budget. Once a reversal is through zero the signs agree again and the
        # axis goes back onto the acceleration budget by itself.
        speeding_up = (np.abs(v_target) > np.abs(v)) & (v_target * v >= 0.0)
        a_lim = np.where(speeding_up, self._a_max_accel, self._a_max_decel)

        # Largest acceleration that can still be unwound to zero, at the jerk
        # limit, within the velocity that is left. Without this the ramp
        # overshoots every target, since `a` cannot be dropped to zero in a
        # single tick; near the target it also fades `a` out on its own, which
        # is the arrival half of the S.
        #
        # The continuous bound is sqrt(2*j*|dv|), but riding it exactly is only
        # marginally feasible: on a discrete tick the ramp ends up a fraction
        # above the curve, cannot shed `a` fast enough to stay on it, and
        # arrives with acceleration still on the clock. The -j*dt/2 term is the
        # discrete correction; it keeps the ramp just inside the feasible
        # region so it lands with `a` already at zero.
        j_dt = self._j_max * dt
        a_arrival = 0.5 * (np.sqrt(j_dt * j_dt + 8.0 * self._j_max * np.abs(dv)) - j_dt)
        a_desired = np.sign(dv) * np.minimum(a_lim, a_arrival)

        # The jerk limit itself. This is the only thing rounding the start of a
        # move, so a high _j_max keeps that rounding brief.
        a = a + np.clip(a_desired - a, -j_dt, j_dt)

        # Integrate, but never step past the target: on a tick that would cross
        # it, the increment is capped at the error that is actually left. That
        # is what lands the ramp exactly on the command instead of ringing
        # around it -- and capping the *step* rather than zeroing `a` is what
        # keeps the last tick of a move inside the jerk limit, since `a` then
        # bleeds off at j while v sits pinned on the target. A tick pushing
        # away from the target (the carry-over acceleration a reversal still
        # has to unwind) is left alone; that motion is real.
        step = a * dt
        step = np.where(
            step * dv >= 0.0,
            np.sign(dv) * np.minimum(np.abs(step), np.abs(dv)),
            step,
        )
        v = v + step

        self._v_prof = v
        self._a_prof = a
        return v

    def _vehicle_velocity_to_angle_and_speed(
        self, u_3dof: np.ndarray, cos_error_scaling: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        vx, vy, omega = float(u_3dof[0]), float(u_3dof[1]), float(u_3dof[2])

        vx_t = np.array([vx, vx, vx, vx], dtype=float)
        vy_t = np.array([vy, vy, vy, vy], dtype=float)
        sign = np.where(TRANS_OPPOSITE_MASK, -1.0, 1.0)
        vx_t *= sign
        vy_t *= sign

        vx_r = np.array(
            [+WIDTH * omega, -WIDTH * omega, -WIDTH * omega, +WIDTH * omega], dtype=float
        )
        vy_r = np.array(
            [+LENGTH * omega, +LENGTH * omega, -LENGTH * omega, -LENGTH * omega],
            dtype=float,
        )
        vx_r = vx_r[ROT_DIAG_SWAP_PERM]
        vy_r = vy_r[ROT_DIAG_SWAP_PERM]

        vx_w = vx_t + vx_r
        vy_w = vy_t + vy_r

        wheel_speeds = np.hypot(vx_w, vy_w)
        wheel_angles = np.arctan2(vy_w, vx_w)

        # A module being asked for no speed has no direction to point in:
        # arctan2(0, 0) is 0, so without this a stop command would re-aim all
        # four modules to straight-ahead. That is not a corner case here --
        # whole-body base velocity is emergent and crosses the dispatch
        # deadband constantly, so every pause would cost a full re-aim out and
        # back (90 degrees each, stopping from a forward drive). Hold the last
        # commanded angle instead and let only the drive setpoint go to zero.
        moving = wheel_speeds > ZERO_SPEED_EPS_MPS

        error = diff_angle(wheel_angles, self.steer_pos)
        wheel_angles = np.where(
            np.abs(error) > np.pi / 2, diff_angle(wheel_angles, np.pi), wheel_angles
        )
        wheel_speeds = np.where(np.abs(error) > np.pi / 2, -wheel_speeds, wheel_speeds)

        if cos_error_scaling:
            wheel_speeds *= np.cos(diff_angle(wheel_angles, self.steer_pos))

        # Applied last, so the flip and the cosine above cannot reintroduce a
        # direction for a module that was never asked to move.
        wheel_angles = np.where(moving, wheel_angles, self.steer_pos)
        wheel_speeds = np.where(moving, wheel_speeds, 0.0)

        return wheel_speeds, wheel_angles

    def _enqueue_command(self, cmd: dict) -> None:
        if self._command_queue is None:
            return

        try:
            self._command_queue.put_nowait(cmd)
            return
        except queue.Full:
            pass

        try:
            while True:
                _ = self._command_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self._command_queue.put_nowait(cmd)
        except queue.Full:
            pass


# ---------------- Example usage ----------------
if __name__ == "__main__":
    base = Base()
    base.start_control()
    rate = RateLimiter(50)
    t0 = time.time()
    try:
        while time.time() - t0 < 5.0:
            # Without smoothing (default)
            # base.set_target_base_velocity(np.array([0.0, 0.0, 0.5]))

            # With smoothing (per-call)
            base.set_target_base_velocity(np.array([0.0, 0.0, 0.5]), smooth=True)
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        base.stop_control()
