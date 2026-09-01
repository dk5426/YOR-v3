# yor.py — YORv3 hardware node.
#
# Exposes the robot over commlink RPC (port 5557). End-effector control runs
# through whole-body IK (robot/wholebody_control.py), which coordinates both
# arms, the lift and the swerve base as one 18-DOF system.
#
# The RPC surface deliberately mirrors robot/yor_mujoco.py (the simulation
# node) so robot/teleop/wholebody_teleop.py drives either one unchanged — the
# only difference is the port.
#
# Direct, per-subsystem control is kept alongside it: set_base_velocity,
# follow_path / move_to and the lift up/down/stop calls all still work, and
# are what joystick.py uses. Any direct base or lift command suspends the
# whole-body loop's authority over that subsystem for a moment
# (manual_override_timeout_s), so the two controllers never fight over the
# same actuator.

import argparse
import atexit
import functools
import json
import sys
import threading
import time
from pathlib import Path

import mink
import numpy as np

# Add project root to sys.path
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from commlink import RPCServer

from nerolib import FirmwareVersion
from robot.arm.arm import ArmNode
from robot.arm.wholebody_ik import WholeBodyIKConfig
from robot.base import BaseController
from robot.base_motor import DRIVE_VEL_SCALE as _DRIVE_VEL_SCALE
from robot.base_motor import MODULE_ORDER
from robot.hand.hands import Hands, add_hand_args, hands_from_args
from robot.swerve_log import DEFAULT_HZ as SWERVE_LOG_HZ
from robot.swerve_log import SwerveRecorder
from robot.wholebody_control import WholeBodyController, WholeBodyHardwareConfig
from tools.base_pid_preflight import (
    COMMISSIONED_MANIFEST,
    DEFAULT_MANIFEST,
    STOCK_MANIFEST,
    drive_command_scale,
    sync_from_manifest,
)

THOR_IP = '192.168.1.11'

YOR_PORT = 5557
LIFT_STARTUP_HOME_WAIT_S = 30.0
LIFT_STARTUP_HEIGHT_M = 0.625


def require_initialization(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            print(f"Warning: {func.__name__} called before YOR was initialized")
            return None
        return func(self, *args, **kwargs)

    return wrapper


def require_wholebody(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.wholebody is None:
            print(f"Warning: {func.__name__} needs whole-body control (run with arms enabled)")
            return None
        return func(self, *args, **kwargs)

    return wrapper


class YOR:
    def __init__(
        self,
        base_max_vel=np.array((1.0, 1.0, 1.57)),
        base_max_accel=np.array((1.0, 1.0, 1.57)),
        no_arms: bool = False,
        wholebody: bool = True,
        wholebody_config: WholeBodyHardwareConfig | None = None,
        ik_config: WholeBodyIKConfig | None = None,
        flash_base_pid: bool = True,
        base_pid_manifest: Path | None = None,
        restore_base_pid: bool = True,
        base_pid_stock_manifest: Path | None = None,
        swerve_log: bool = True,
        swerve_log_hz: float = SWERVE_LOG_HZ,
        gripper: str = "none",
        hands: Hands | None = None,
    ):
        self._initialized = False
        self._flash_base_pid = bool(flash_base_pid)
        self._base_pid_manifest = Path(base_pid_manifest) if base_pid_manifest else DEFAULT_MANIFEST
        self._restore_base_pid = bool(restore_base_pid)
        self._base_pid_provenance = "unknown"
        self._swerve_log = bool(swerve_log)
        self._swerve_log_hz = float(swerve_log_hz)
        self._swerve_recorder: SwerveRecorder | None = None
        self._base_pid_stock_manifest = (
            Path(base_pid_stock_manifest) if base_pid_stock_manifest
            else STOCK_MANIFEST
        )

        # The command scale is only correct for one set of drive gains, so it
        # comes from the same manifest the gains do -- picking a manifest picks
        # both, and they cannot drift apart. See docs/BASE_COMMAND_LOOP_REVIEW.md
        # finding 6 for why that matters.
        self._drive_vel_scale, scale_note = drive_command_scale(
            self._base_pid_manifest, _DRIVE_VEL_SCALE)
        print(f"[YOR] base: drive command scale {scale_note}")

        self.slam_sub = None
        self._reset_nav = False

        self.pose = None        # tuple of ((x,y,z), theta_z, 4x4_pose)

        self.base_controller = BaseController(
            yor=self,
            base_max_vel=base_max_vel,
            base_max_accel=base_max_accel,
            origin=(0.0, 0.0),
            grid_res=0.05,
            # Navigation closes on the SLAM pose, so it is paced by it. The
            # publisher measures 30 Hz (2026-08-27), so 20 here is slower
            # than it needs to be -- raising it is a nav-tuning change, not a
            # polling one, because the PIDs' dt and vel_alpha were tuned at
            # this rate, so it is left alone. The `pose_sig` gate in
            # robot/base.py is what keeps repeats harmless either way.
            control_hz=20,
            # Whole-body control now publishes base commands at 30 Hz
            # (WholeBodyHardwareConfig.control_hz). The relay is deliberately
            # kept faster than that, not matched to it: it only has to stay at
            # or above the producer's rate to never discard a solver update,
            # and 108 Hz also preserves the swerve loop's own 3x-oversampled
            # S-curve profiling (base_motor.py CONTROL_FREQ = 324 = 108 * 3),
            # which has no reason to be retuned here.
            relay_hz=108,
            drive_vel_scale=self._drive_vel_scale,
        )
        self.base = self.base_controller.base
        self.no_arms = no_arms
        self.left_arm = None
        self.right_arm = None
        # The WUJI fingers, in this process but off this RPC socket: they
        # arrive on the aria2robot publisher, on a thread of their own, so a
        # finger target never queues behind a 30 Hz arm target. See
        # robot/hand/hands.py.
        self.hands: Hands | None = hands
        self.wholebody: WholeBodyController | None = None
        self._wholebody_requested = wholebody and not no_arms
        self._wholebody_config = wholebody_config
        self._ik_config = ik_config
        self._homing_lock = threading.Lock()

        # Which gripper hardware is fitted, if any. Off by default: with no
        # gripper attached a value arriving from teleop is dropped rather than
        # sent to an actuator that is not there, and -- more to the point --
        # ArmNode calibrates a dynamixel gripper by physically driving it shut
        # and back open at startup, which must never happen by accident.
        self.gripper_kind = str(gripper)
        if self.gripper_kind not in ("none", "dynamixel", "native"):
            raise ValueError(
                f"gripper must be none|dynamixel|native, got {gripper!r}")
        dynamixel_gripper = self.gripper_kind == "dynamixel"
        native_gripper = self.gripper_kind == "native"

        if not self.no_arms:
            self.left_arm = ArmNode(
                can_port="can_left",
                dynamixel_gripper=dynamixel_gripper,
                native_gripper=native_gripper,
                firmware_version=FirmwareVersion.V111,
            )
            self.right_arm = ArmNode(
                can_port="can_right",
                is_left_arm=False,
                dynamixel_gripper=dynamixel_gripper,
                native_gripper=native_gripper,
                firmware_version=FirmwareVersion.V111,
            )

    def init(self):
        if self._initialized:
            print("Warning: YOR already initialized")
            return

        # The commissioned swerve gains, before anything drives a wheel. The
        # SPARKs hold them in RAM, so a controller power cycle silently
        # restores stock gains and that one module then steers and drives
        # unlike its three neighbours.
        self._sync_base_pid_gains()
        self._report_base_configuration()

        # Actively hold the chassis still throughout startup homing.
        print("[YOR] base: locked at zero velocity for startup homing")
        self.base_controller.mode = "BASE_VEL"
        self.base_controller.target_velocity = np.zeros(3, dtype=float)
        self.base.start_control()
        self._start_swerve_log()
        self.base.set_target_base_velocity(np.zeros(3), smooth=False)
        time.sleep(0.5)

        # Wait for the lift controller to finish booting before sending home.
        print("[YOR] lift: waiting for controller to be ready...")
        ready_start = time.time()
        while time.time() - ready_start < 3.0:
            if self.base._pico_lift.get_capabilities():
                break
            time.sleep(0.1)

        # Establish an absolute lift zero, then move to the arm-safe height.
        print(
            f"[YOR] lift: homing at startup; waiting up to "
            f"{LIFT_STARTUP_HOME_WAIT_S:.0f}s"
        )
        self.base.lift_home()
        home_start = time.time()
        while time.time() - home_start < LIFT_STARTUP_HOME_WAIT_S:
            lift_status = self.base.get_lift_status()
            if lift_status.get("homed") is True:
                break
            time.sleep(0.5)

        lift_status = self.base.get_lift_status()
        if (
            not lift_status.get("available", False)
            or lift_status.get("homed") is not True
            or lift_status.get("position_known") is not True
        ):
            self.base.lift_stop()
            raise RuntimeError(
                "lift startup home did not complete within 30s: "
                f"homed={lift_status.get('homed')}, "
                f"position_known={lift_status.get('position_known')}, "
                f"last_event={lift_status.get('last_event')!r}"
            )
        print(f"[YOR] lift: home complete at {lift_status.get('height_m')} m")

        print(f"[YOR] lift: moving to {LIFT_STARTUP_HEIGHT_M * 1000:.0f} mm from zero")
        if not self.base.lift_to_height(LIFT_STARTUP_HEIGHT_M):
            self.base.lift_stop()
            raise RuntimeError(
                f"lift startup move to {LIFT_STARTUP_HEIGHT_M * 1000:.0f} mm "
                "did not complete"
            )
        print(f"[YOR] lift: startup position {self.base.get_lift_height()} m")

        # Home the arms sequentially so only one arm moves at a time.
        if not self.no_arms:
            print("[YOR] arms: homing all 7 left-arm joints")
            if not self.left_arm.init():
                raise RuntimeError("left arm joint homing did not complete")
            print("[YOR] arms: left arm home; homing all 7 right-arm joints")
            if not self.right_arm.init():
                raise RuntimeError("right arm joint homing did not complete")
            print("[YOR] arms: all joints home")

        self._initialized = True

        if self._wholebody_requested:
            # Stamp the gain set into the config before the controller builds
            # its trajectory recorder, so every log says what it was driving on.
            if self._wholebody_config is None:
                self._wholebody_config = WholeBodyHardwareConfig()
            self._wholebody_config.base_pid_provenance = self._base_pid_provenance
            self.wholebody = WholeBodyController(
                left_arm=self.left_arm,
                right_arm=self.right_arm,
                base=self.base,
                base_controller=self.base_controller,
                config=self._wholebody_config,
                ik_config=self._ik_config,
            )
            if not self.wholebody.config.enable_base_motion:
                self.wholebody.toggle_fix_base(True)
            if not self.wholebody.config.enable_lift_motion:
                self.wholebody.ik.toggle_fix_lift(True)
            self.wholebody.start()

        # Last, after the arms have homed: the first pose an engaged operator
        # sends is a whole grasp, and nothing should be closing a hand while
        # an arm is still travelling to home.
        #
        # Never fatal. Everything above this raises for a living, and rightly
        # so -- a lift that did not home has no absolute zero and an arm that
        # did not home cannot be driven. The hands are an accessory, and by
        # this point the expensive part is done: a missing wujihandpy, an
        # unprovisioned ~/.wuji, a wrong serial or a busy USB device must not
        # throw away a completed 30-60 s homing cycle and leave the operator
        # with nothing. Drop them and run the arms.
        if self.hands is not None:
            try:
                self.hands.start()
            except Exception as exc:
                print(f"[YOR] hands failed to start ({exc!r})")
                print("[YOR] continuing WITHOUT fingers; the arms are unaffected")
                # A partial open leaves one hand energised and unowned.
                # Hands.stop() marks itself started before the driver opens, so
                # this reaches the driver's close() and disables whatever came up.
                try:
                    self.hands.stop()
                except Exception as stop_exc:
                    print(f"[YOR] hands cleanup also failed ({stop_exc!r})")
                self.hands = None

    def _sync_base_pid_gains(self) -> None:
        """Bring the swerve controllers to the selected PID manifest.

        Defaults to config/base_pid_stock.json -- the gains the controllers
        hold in flash, and the ones DRIVE_VEL_SCALE = 2.0 is correct for. The
        tuned set is config/base_pid_commissioned.json, opt-in via
        --base-pid-manifest; see that file's description for why.

        Runs through the SparkFlex objects base_motor.py already opened, and
        before base.start_control(): no second set of device handles touches
        the bus, and the control loop is not yet sending setpoints. Each
        controller is read first and written only if it differs, so a restart
        that did not power-cycle the SPARKs writes nothing.

        Raises if any controller cannot be brought to the commissioned values.
        Driving on gains nobody can name is the failure this exists to prevent,
        and it is far harder to diagnose from the robot's behaviour than a
        refusal to start. Pass flash_base_pid=False to start without it; the
        same check is still available as `python tools/base_pid_preflight.py`.
        """
        if not self._flash_base_pid:
            print("[YOR] base PID sync skipped (flash_base_pid=False)")
            self._base_pid_provenance = (
                f"not flashed (controllers hold whatever was there), "
                f"scale={self._drive_vel_scale:g}")
            return

        print("[YOR] base PID: syncing swerve gains")
        started = time.time()
        ok, problems = sync_from_manifest(
            self.base.swerve_devices(),
            manifest_path=self._base_pid_manifest,
            log=lambda line: print(f"[YOR] base PID: {line}"),
        )
        print(f"[YOR] base PID: {time.time() - started:.1f}s")
        self._base_pid_provenance = self._describe_pid_manifest(self._base_pid_manifest, ok)

        if not ok:
            for problem in problems:
                print(f"[YOR] base PID: {problem}")
            raise RuntimeError(
                f"swerve PID sync failed with {len(problems)} problem(s); the base "
                "control loop was not started. Fix the CAN bus or the manifest, or "
                "construct YOR with flash_base_pid=False to start without it."
            )

    def _describe_pid_manifest(self, path: Path, ok: bool) -> str:
        """One line naming the gains this run is actually driving on.

        The gain set changes what every speed number in the log means -- the
        stock drive loop reaches about 40% of its setpoint on the floor while
        the commissioned one tracks -- so the file name alone is not enough.
        The drive p/ff pair identifies the set unambiguously.
        """
        try:
            roles = json.loads(Path(path).read_text())["roles"]
            drive, steer = roles["drive"], roles["steering"]
            gains = (f"drive p={drive['p']:g} ff={drive['velocity_ff']:g}, "
                     f"steer p={steer['p']:g} out=+/-{steer['output_max']:g}")
        except Exception:
            gains = "gains unreadable"
        return (f"{Path(path).name} [{gains}, scale={self._drive_vel_scale:g}]"
                + ("" if ok else " SYNC FAILED"))

    def _start_swerve_log(self) -> None:
        """Record per-module swerve telemetry for the life of the base loop.

        Deliberately not tied to whole-body control. The trajectory log only
        exists when a WholeBodyController does, so `--no-arms` -- which is how
        the base gets driven from joystick.py -- recorded nothing at all, and
        even with arms it samples at the 30 Hz solve rate rather than the 50 Hz
        the SPARKs publish at. This runs off the base loop instead, so a
        joystick run and a teleop run are directly comparable.
        """
        if not self._swerve_log:
            return
        path = (_ROOT / "artifacts" / "wholebody_logs" / "swerve"
                / f"swerve_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        try:
            self._swerve_recorder = SwerveRecorder(
                path, self.base, MODULE_ORDER, sample_hz=self._swerve_log_hz,
                config_notes=[f"base_pid={self._base_pid_provenance}",
                              f"sample_hz={self._swerve_log_hz}"])
            self._swerve_recorder.start()
            print(f"[YOR] recording swerve telemetry to {path}")
        except Exception as exc:
            print(f"[YOR] swerve log unavailable ({exc!r}); continuing without it")
            self._swerve_recorder = None

    def stop_swerve_log(self) -> None:
        if self._swerve_recorder is not None:
            self._swerve_recorder.stop()
            self._swerve_recorder = None

    def _report_base_configuration(self) -> None:
        """Print what the swerve controllers say about their own setup.

        Idle mode, control type and the two conversion factors all live in
        SPARK flash, set out-of-band through the REV Hardware Client. Nothing
        in this repository chose them and nothing recorded them, which is
        exactly why `DRIVE_VEL_SCALE = 2.0` reads as a magic number -- it is
        one half of a unit conversion whose other half is not in git.

        `velocity_cf` is the one that decides whether `set_velocity_mps` is
        speaking true m/s: 0.00083 means yes, 0.00166 means the controllers
        believe twice the real speed. See docs/BASE_COMMAND_LOOP_REVIEW.md
        finding 6. Diagnostic only -- nothing here changes behaviour.
        """
        report = getattr(self.base, "swerve_configuration", None)
        if report is None:
            return
        try:
            config = report()
        except Exception as exc:
            print(f"[YOR] base config: unavailable ({exc!r})")
            return

        def summarise(key):
            values = {c[key] for c in config.values()}
            only = values.pop() if len(values) == 1 else None
            return f"{only}" if only is not None else "MIXED " + str(sorted(
                (name, c[key]) for name, c in config.items()))

        print(f"[YOR] base config: idle_mode={summarise('idle_mode')}"
              f" ctrl_type={summarise('ctrl_type')}")
        print(f"[YOR] base config: velocity_cf={summarise('velocity_cf')}"
              f" position_cf={summarise('position_cf')}"
              f" (drive_command_scale={self._drive_vel_scale})")

    def _restore_base_pid_gains(self) -> None:
        """Write the SPARK stock gains back on the way out.

        The commissioned gains live in controller RAM, which outlives this
        process: whatever opens the bus next — joystick.py, a nav run, a bare
        base_motor.py — inherits them silently, and the manifest's own notes
        record that some of them (the +/-0.25 steering output clamp in
        particular) are specific to this configuration. Restoring stock on the
        way out means gains are only ever in effect while the process that
        asked for them is running.

        Three things make this safe to do here and not elsewhere:

        * It only runs if `_sync_base_pid_gains` actually wrote. If the node
          started with `--no-flash-base-pid` it changed nothing, and undoing
          a change nobody made would clobber someone else's commissioning.
        * The control loop has to be stopped first. Changing gains underneath
          a live velocity setpoint is a step change in the plant, not a
          configuration edit; the guard below stops the loop rather than
          trusting the caller to have done it.
        * It goes through the same validated, read-back sync as startup, so a
          write the controller ignored is reported rather than assumed.

        Never raises. A failure here cannot be allowed to skip the arm drop
        that follows it in the shutdown sequence.
        """
        if not self._restore_base_pid:
            return
        if not self._flash_base_pid:
            print("[YOR] base PID: restore skipped (gains were never flashed)")
            return
        try:
            same = (self._base_pid_manifest.resolve()
                    == self._base_pid_stock_manifest.resolve())
        except Exception:
            same = False
        if same:
            # Startup applied the stock manifest, so there is nothing to undo.
            # Saying so beats printing "restoring stock" over eight no-op writes.
            print("[YOR] base PID: restore not needed (started on the stock manifest)")
            return

        try:
            if self.base.control_loop_running:
                # Gains must not change under a live setpoint.
                self.base.set_target_base_velocity(np.zeros(3), smooth=False)
                self.base.stop_control()

            print("[YOR] base PID: restoring stock swerve gains")
            started = time.time()
            ok, problems = sync_from_manifest(
                self.base.swerve_devices(),
                manifest_path=self._base_pid_stock_manifest,
                log=lambda line: print(f"[YOR] base PID: {line}"),
            )
            print(f"[YOR] base PID: {time.time() - started:.1f}s")
            if not ok:
                for problem in problems:
                    print(f"[YOR] base PID: {problem}")
                print(
                    f"[YOR] base PID: {len(problems)} controller(s) still hold the "
                    "commissioned gains. Power-cycle the SPARKs, or run "
                    "`python tools/base_pid_preflight.py "
                    "--manifest config/base_pid_stock.json`."
                )
        except Exception as exc:
            print(f"[YOR] base PID: restore failed ({exc!r}); shutdown continues")

    # ─────────────────────────────────────────────────────────────────────────
    # Base — direct control (joystick, nav). Suspends whole-body base authority.
    # ─────────────────────────────────────────────────────────────────────────

    @require_initialization
    def set_base_velocity(self, velocity: np.ndarray):
        if self.wholebody is not None:
            self.wholebody.notify_manual_base_command()
        self.base_controller.mode = "BASE_VEL"
        self.base_controller.target_velocity = velocity

    @require_initialization
    def follow_path(self, path=None):
        if self.wholebody is not None:
            self.wholebody.notify_manual_base_command()
        self.base_controller.slam_sub_init()

        if path is None:
            self.base_controller._path_world = None
            self.base_controller.mode = "BASE_VEL"
            self.base_controller.target_velocity = np.zeros(3, dtype=float)
            print("[YOR] follow_path: cleared")
            return True

        clean = [(float(p[0]), float(p[1])) for p in path]
        self.base_controller._path_world = clean
        self.base_controller.mode = "PATH_FOLLOWING"
        print(f"[YOR] follow_path: n={len(clean)} first={clean[0]} last={clean[-1]}")
        return True

    @require_initialization
    def get_nav_debug(self):
        if hasattr(self.base_controller, "get_nav_debug"):
            return self.base_controller.get_nav_debug()
        return None

    @require_initialization
    def set_base_pose_limits(self, max_lin_vel = None, max_ang_vel = None):
        """Runtime speed ceiling for the POSE_TARGET servo.

        BasePoseController defaults to 0.25 m/s because whole-body control
        drives the chassis under an arm; navigation wants the drive's full
        0.35. Set it per-session rather than editing the default, so the two
        callers do not have to share one number.

        Still clamped by whatever ceiling `Base` was constructed with -- that
        limit is not enforced in the drive's control loop, so a controller
        that ignored it would simply command past it. Returns what was
        actually applied.
        """
        ctl = self.base_controller.pose_ctl
        limits = getattr(ctl.base, "max_vel", None)
        lin_cap = ang_cap = None
        if limits is not None:
            lim = np.asarray(limits, dtype=float).reshape(-1)
            if lim.size >= 3:
                lin_cap = float(min(abs(lim[0]), abs(lim[1])))
                ang_cap = float(abs(lim[2]))
        if max_lin_vel is not None:
            v = float(max_lin_vel)
            ctl.max_lin_vel = v if lin_cap is None else min(v, lin_cap)
        if max_ang_vel is not None:
            w = float(max_ang_vel)
            ctl.max_ang_vel = w if ang_cap is None else min(w, ang_cap)
        return {"max_lin_vel": float(ctl.max_lin_vel),
                "max_ang_vel": float(ctl.max_ang_vel),
                "drive_cap_lin": lin_cap, "drive_cap_ang": ang_cap}

    @require_initialization
    def set_base_pose_target(self, target = None):
        """Stream a base pose setpoint, servo'd by BasePoseController.

        `target` is [u, v, psi] in the CONTROL FRAME:
            u   = world x
            v   = -world z
            psi = SLAM yaw, unflipped (NOT the +pi value get_nav_debug reports)

        The caller sends the target already in this frame, so nothing is
        transformed on the Pi. Call repeatedly to stream a moving setpoint;
        it is stateless, each call just replaces the target.

        Unlike follow_path(), heading comes from `psi` directly rather than
        from the bearing to a lookahead point, so heading and translation stop
        competing for the same job. Pass None to stop.
        """
        if self.wholebody is not None:
            self.wholebody.notify_manual_base_command()
        self.base_controller.slam_sub_init()
        if target is None:
            self.base_controller._pose_target = None
            self.base_controller.pose_ctl.reset()
            self.base_controller.mode = "BASE_VEL"
            self.base_controller.target_velocity = np.zeros(3, dtype=float)
            return
        if self.base_controller.mode != "POSE_TARGET":
            # New authority over the base: forget the damping history, or the
            # first cycle damps against a velocity the base never had.
            self.base_controller.pose_ctl.reset()
        self.base_controller._pose_target = [float(v) for v in target][:3]
        self.base_controller.mode = "POSE_TARGET"

    @require_initialization
    def move_to(self, goal = None):
        if self.wholebody is not None:
            self.wholebody.notify_manual_base_command()
        self.base_controller.slam_sub_init()
        self.base_controller._goal = goal
        self.base_controller.mode = "MOVE_TO"

    @require_initialization
    def move_by(self, deltas = None):
        if self.wholebody is not None:
            self.wholebody.notify_manual_base_command()
        self.base_controller.slam_sub_init()
        if self.pose is None:
            print("Warning: move_by called before pose is available")
            return
        if deltas is None:
            print("Warning: move_by called without deltas")
            return
        translation, theta, T_base = self.pose               # (x,y,z), theta_z, 4x4 transform
        x, y = float(translation[0]), float(translation[2])  # (x,z) plane

        self.base_controller._goal = (x+deltas[0], y+deltas[1], theta+deltas[2])
        self.base_controller.mode = "MOVE_TO"

    @require_initialization
    def get_cmd_vel(self):
        # returns ([vx, vy, omega], timestamp)
        v = np.asarray(self.base_controller.target_velocity, dtype=float)
        return v.tolist(), time.time()

    @require_initialization
    def get_base_velocity(self):
        """[vx, vy, omega] the whole-body solver last asked the base for."""
        if self.wholebody is None:
            return np.zeros(3).tolist()
        return self.wholebody.get_base_velocity().tolist()

    @require_initialization
    def get_base_encoders(self) -> dict:
        """Return steer positions (rad) and drive velocities (raw) for all 4 modules."""
        base = self.base
        return {
            "timestamp": time.time(),
            "steer_rad":    [m.get_position_rad()    for m in base.rotation_motors],
            "steer_deg":    [m.get_position_deg()    for m in base.rotation_motors],
            "steer_counts": [m.get_position_counts() for m in base.rotation_motors],
            "drive_vel":    [m.get_velocity_raw()    for m in base.drive_motors],
            "drive_counts": [m.get_position_counts() for m in base.drive_motors],
            "lift_height_m": base.get_lift_height(),
        }

    @require_initialization
    def get_pose(self) -> dict:
        """Return the latest SLAM pose: x, y, theta (yaw in radians).
        x = translation[0], y = translation[2] (robot moves in XZ plane).
        """
        if self.pose is None:
            return {"x": None, "y": None, "theta": None}
        translation, theta, _ = self.pose
        return {
            "x": float(translation[0]),
            "y": float(translation[2]),
            "theta": float(theta),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Lift — direct control (joystick D-pad, scripts)
    # ─────────────────────────────────────────────────────────────────────────

    def _manual_lift(self) -> None:
        if self.wholebody is not None:
            self.wholebody.notify_manual_lift_command()

    @require_initialization
    def lift_up(self) -> None:
        self._manual_lift()
        if hasattr(self.base, "lift_up"):
            self.base.lift_up()

    @require_initialization
    def lift_down(self) -> None:
        self._manual_lift()
        if hasattr(self.base, "lift_down"):
            self.base.lift_down()

    @require_initialization
    def lift_stop(self) -> None:
        self._manual_lift()
        if hasattr(self.base, "lift_stop"):
            self.base.lift_stop()

    @require_initialization
    def lift_home(self) -> None:
        """Send the lift home.

        With whole-body control running this resets the solver's lift target;
        otherwise it falls back to the hardware homing routine, which drives
        the lift straight to its limit switch.
        """
        if self.wholebody is not None:
            self.wholebody.lift_home()
            return
        self._manual_lift()
        self.base.lift_home()

    @require_initialization
    def lift_set_velocity(self, velocity_m_s: float) -> bool:
        """Stream a lift velocity in metres per second: + is up, - is down.

        Direct control, like lift_up/lift_down/lift_stop: it suspends the
        whole-body loop's authority over the lift for manual_override_timeout_s
        so the two never fight over the column. The firmware stops by itself if
        no command arrives for 300 ms, so a caller that wants continuous motion
        must keep calling — at least every 100 ms.

        Returns False when the value is not a finite number, or when the
        attached controller cannot stream velocity; check
        lift_supports_velocity() first.
        """
        if not hasattr(self.base, "lift_set_velocity"):
            print("[YOR] base has no lift_set_velocity()")
            return False
        self._manual_lift()
        return bool(self.base.lift_set_velocity(velocity_m_s))

    @require_initialization
    def lift_supports_velocity(self) -> bool:
        """Whether the attached lift firmware advertised streamed velocity."""
        if not hasattr(self.base, "lift_supports_velocity"):
            return False
        return bool(self.base.lift_supports_velocity())

    @require_initialization
    def get_lift_height(self) -> float:
        return self.base.get_lift_height()

    @require_initialization
    def get_lift_status(self) -> dict:
        """Full lift snapshot: height, position-known, homed, limits, motion.

        Requests a fresh `status` from the controller, so the limit-switch
        fields reflect the switches right now. Everything is a plain type, so
        it crosses the RPC boundary unchanged.
        """
        if hasattr(self.base, "get_lift_status"):
            return self.base.get_lift_status()
        return {"available": False}

    @require_initialization
    def lift_position_known(self) -> bool | None:
        """Whether the lift controller has an established zero.

        False means every height it reports is meaningless — run lift_home().
        None means it has not said either way yet.
        """
        if hasattr(self.base, "lift_position_known"):
            return self.base.lift_position_known()
        return None

    @require_initialization
    def get_lift_position(self) -> float:
        """Alias of get_lift_height(), for parity with the simulation node."""
        if self.wholebody is not None:
            return self.wholebody.get_lift_position()
        height = self.base.get_lift_height()
        return float(height) if height is not None else 0.0

    @require_initialization
    def set_lift_target(self, lift_target: float):
        """Ask the whole-body solver for a lift height (metres)."""
        if self.wholebody is None:
            print("[YOR] set_lift_target needs whole-body control; use lift_to_height()")
            return
        self.wholebody.set_lift_target(lift_target)

    @require_initialization
    def lift_delta_height(
        self,
        delta_m: float,
        tolerance_m: float = 0.002,
        timeout_s: float = 30.0,
        min_height_m: float = 0.0,
        max_height_m: float = 0.900,
    ) -> bool:
        if not hasattr(self.base, "lift_delta_height"):
            print("[YOR] base has no lift_delta_height()")
            return False
        self._manual_lift()
        try:
            return bool(self.base.lift_delta_height(
                delta_m,
                tolerance_m=tolerance_m,
                timeout_s=timeout_s,
                min_height_m=min_height_m,
                max_height_m=max_height_m,
            ))
        except TypeError:
            return bool(self.base.lift_delta_height(delta_m))

    @require_initialization
    def lift_to_height(
        self,
        target_m: float,
        tolerance_m: float = 0.002,
        timeout_s: float = 30.0,
        min_height_m: float = 0.0,
        max_height_m: float = 0.900,
    ) -> bool:
        """Blocking absolute lift move, bypassing the whole-body solver."""
        if not hasattr(self.base, "lift_to_height"):
            print("[YOR] base has no lift_to_height()")
            return False
        self._manual_lift()
        return bool(self.base.lift_to_height(
            target_m,
            tolerance_m=tolerance_m,
            timeout_s=timeout_s,
            min_height_m=min_height_m,
            max_height_m=max_height_m,
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # Arms — end-effector control (whole-body)
    # ─────────────────────────────────────────────────────────────────────────

    @require_initialization
    @require_wholebody
    def set_left_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None,
                           preview_time: float = 0.1):
        self.wholebody.set_left_ee_target(ee_target, gripper_target, preview_time)

    @require_initialization
    @require_wholebody
    def set_right_ee_target(self, ee_target: mink.SE3, gripper_target: float | None = None,
                            preview_time: float = 0.1):
        self.wholebody.set_right_ee_target(ee_target, gripper_target, preview_time)

    @require_initialization
    @require_wholebody
    def set_bimanual_ee_target(self,
                               L_ee_target: mink.SE3, R_ee_target: mink.SE3,
                               L_gripper_target: float | None = None, L_preview_time: float = 0.1,
                               R_gripper_target: float | None = None, R_preview_time: float = 0.1):
        self.wholebody.set_bimanual_ee_target(
            L_ee_target, R_ee_target,
            L_gripper_target=L_gripper_target, R_gripper_target=R_gripper_target,
            L_preview_time=L_preview_time, R_preview_time=R_preview_time,
        )

    def _home_arm_joints(self, sides: tuple[str, ...]) -> bool:
        """Run the Quest arm-home sequence with exclusive actuator ownership."""
        if self.no_arms:
            print("[YOR] arm homing refused: arms are disabled")
            return False
        if not self._homing_lock.acquire(blocking=False):
            print("[YOR] arm homing already in progress")
            return False

        active_wholebody = self.wholebody
        runtime_config = (
            active_wholebody.config
            if active_wholebody is not None
            else self._wholebody_config
        )
        previous_fix_base = False
        previous_fix_lift = False
        previous_collisions = True
        try:
            # Home means "return to a known pose", and the hand is part of
            # that. First, before the lift moves, so anything being held falls
            # from where it is rather than from 625 mm up. Safe as a gesture
            # because the home gesture needs both hands *released*.
            if self.hands is not None:
                self.hands.open_hands(sides)
            if active_wholebody is not None:
                previous_fix_base = bool(active_wholebody.ik.fix_base)
                previous_fix_lift = bool(active_wholebody.ik.fix_lift)
                previous_collisions = bool(active_wholebody.ik.avoid_collisions)
                active_wholebody.toggle_fix_base(True)

            print("[YOR] Quest home: locking base at zero velocity")
            self.base_controller.mode = "BASE_VEL"
            self.base_controller.target_velocity = np.zeros(3, dtype=float)
            self.base.set_target_base_velocity(np.zeros(3), smooth=False)
            time.sleep(0.1)

            if active_wholebody is not None:
                active_wholebody.stop()
                self.wholebody = None

            lift_status = self.base.get_lift_status()
            if lift_status.get("position_known") is not True:
                raise RuntimeError(
                    "lift position is unknown; restart the hardware node to home the lift"
                )

            print(
                f"[YOR] Quest home: moving lift to "
                f"{LIFT_STARTUP_HEIGHT_M * 1000:.0f} mm from zero"
            )
            if not self.base.lift_to_height(LIFT_STARTUP_HEIGHT_M):
                raise RuntimeError("lift did not reach the 625 mm arm-home height")

            for side in sides:
                arm = self.left_arm if side == "left" else self.right_arm
                print(f"[YOR] Quest home: homing all 7 {side}-arm joints")
                if not arm.init():
                    raise RuntimeError(f"{side} arm joint homing did not complete")

            # Seed a fresh controller from the new joint and lift state so a
            # stale pre-home target cannot pull the arm back after resuming.
            if self._wholebody_requested:
                self.wholebody = WholeBodyController(
                    left_arm=self.left_arm,
                    right_arm=self.right_arm,
                    base=self.base,
                    base_controller=self.base_controller,
                    config=runtime_config,
                    ik_config=self._ik_config,
                )
                self.wholebody.toggle_fix_base(previous_fix_base)
                self.wholebody.ik.toggle_fix_lift(previous_fix_lift)
                self.wholebody.toggle_collision_avoidance(previous_collisions)
                self.wholebody.start()

            print("[YOR] Quest home: complete")
            return True
        except Exception as exc:
            self.base_controller.mode = "BASE_VEL"
            self.base_controller.target_velocity = np.zeros(3, dtype=float)
            self.base.set_target_base_velocity(np.zeros(3), smooth=False)
            self.base.lift_stop()
            self.wholebody = None
            print(f"[YOR] Quest home failed: {exc}")
            return False
        finally:
            self._homing_lock.release()

    @require_initialization
    def home_left_arm(self) -> bool:
        """Quest Y: lock base, lift to 625 mm, then home all left joints."""
        return self._home_arm_joints(("left",))

    @require_initialization
    def home_right_arm(self) -> bool:
        """Quest B: lock base, lift to 625 mm, then home all right joints."""
        return self._home_arm_joints(("right",))

    @require_initialization
    def home_arms(self) -> bool:
        """Quest Y+B: run one lift preamble, then home left and right arms."""
        return self._home_arm_joints(("left", "right"))

    @require_initialization
    @require_wholebody
    def toggle_fix_base(self, fixed: bool | None = None) -> bool:
        """Lock the base in the solver: only the arms and lift move."""
        return self.wholebody.toggle_fix_base(fixed)

    @require_initialization
    @require_wholebody
    def toggle_collision_avoidance(self, enable: bool | None = None) -> bool:
        return self.wholebody.toggle_collision_avoidance(enable)

    @require_initialization
    @require_wholebody
    def toggle_base_motion(self, enable: bool | None = None) -> bool:
        """Allow / forbid the solver from driving the wheels at all."""
        return self.wholebody.toggle_base_motion(enable)

    @require_initialization
    @require_wholebody
    def relatch_elbow_swivel(self, side: str | None = None) -> bool:
        """Accept the elbow branch the arm(s) are currently in.

        Clears the latched swivel target for `side` ("left"/"right", or both
        when omitted); the next solve re-latches from the live pose. The
        cheap recovery from a fought elbow branch -- no homing cycle needed.
        """
        return self.wholebody.relatch_elbow_swivel(side)

    @require_initialization
    def get_state(self) -> dict:
        """Snapshot for teleop clients (plain types), matching the sim node."""
        if self.wholebody is None:
            return {}
        state = self.wholebody.get_state()
        state["lift"] = self.get_lift_position()
        # Reported here, though not commanded here, so one get_state() is a
        # snapshot of the whole robot -- arms and fingers -- at one instant.
        hand = self.hands.targets() if self.hands is not None else {}
        for side in ("left", "right"):
            q = hand.get(side)
            state[f"{side}_hand_qpos"] = None if q is None else q.tolist()
        # The rest of what `Hands` knows -- engagement, where the pose came
        # from, and the cumulative count of writes that reached the driver.
        # A total, not a rate: the client differentiates it across redraws.
        state["hands"] = (
            None if self.hands is None
            else {k: v for k, v in self.hands.get_hand_state().items()
                  if k != "qpos"})
        return state

    @require_initialization
    def get_left_ee_pose(self) -> mink.SE3:
        """Left end-effector pose in the world frame."""
        if self.wholebody is None:
            return None
        return self.wholebody.get_left_ee_pose()

    @require_initialization
    def get_right_ee_pose(self) -> mink.SE3:
        """Right end-effector pose in the world frame."""
        if self.wholebody is None:
            return None
        return self.wholebody.get_right_ee_pose()

    @require_initialization
    def get_arm_relative_pose(self) -> tuple[mink.SE3, mink.SE3]:
        left_ee_pose = self.get_left_ee_pose()
        right_ee_pose = self.get_right_ee_pose()
        l2r = right_ee_pose.inverse() @ left_ee_pose
        r2l = left_ee_pose.inverse() @ right_ee_pose

        return r2l, l2r

    # ─────────────────────────────────────────────────────────────────────────
    # Arms — joint space / grippers (direct)
    # ─────────────────────────────────────────────────────────────────────────

    def _manual_arms(self) -> None:
        if self.wholebody is not None:
            self.wholebody.notify_manual_arm_command()

    @require_initialization
    def set_left_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        if self.no_arms:
            print("left arm disabled")
            return
        self._manual_arms()
        self.left_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_right_joint_target(
        self, joint_target: np.ndarray, gripper_target: float | None = None, preview_time: float = 0.1
    ):
        if self.no_arms:
            print("right arm disabled")
            return
        self._manual_arms()
        self.right_arm.set_joint_target(joint_target, gripper_target, preview_time)

    @require_initialization
    def set_left_gain(self, kp: np.ndarray, kd: np.ndarray):
        if self.no_arms:
            print("left arm disabled")
            return
        self.left_arm.set_gain(kp, kd)

    @require_initialization
    def set_right_gain(self, kp: np.ndarray, kd: np.ndarray):
        if self.no_arms:
            print("right arm disabled")
            return
        self.right_arm.set_gain(kp, kd)

    @require_initialization
    def park(self, gripper_target: float = 1.0):
        """Stop whole-body control and send both arms to the hardware home pose.

        Unlike home_left_arm() / home_right_arm(), which only move the solver's
        targets, this hands the arms back to nerolib's homing routine and
        leaves the whole-body loop stopped. Call resume_wholebody() to restart.
        """
        if self.wholebody is not None:
            self.wholebody.stop()
            self.wholebody = None
        if self.no_arms:
            return
        self.left_arm.home(gripper_target)
        self.right_arm.home(gripper_target)

    @require_initialization
    def tuck_arms(self):
        """Stop whole-body control and tuck both arms (zero joint pose)."""
        if self.wholebody is not None:
            self.wholebody.stop()
            self.wholebody = None
        if self.no_arms:
            return
        self.left_arm.tuck_arms()
        self.right_arm.tuck_arms()

    @require_initialization
    def resume_wholebody(self) -> bool:
        """Restart whole-body control after park() / tuck_arms() / emergency_stop()."""
        if self.no_arms:
            print("[YOR] cannot resume whole-body control without arms")
            return False
        if self.wholebody is None:
            self.wholebody = WholeBodyController(
                left_arm=self.left_arm,
                right_arm=self.right_arm,
                base=self.base,
                base_controller=self.base_controller,
                config=self._wholebody_config,
                ik_config=self._ik_config,
            )
            if not self.wholebody.config.enable_base_motion:
                self.wholebody.toggle_fix_base(True)
            if not self.wholebody.config.enable_lift_motion:
                self.wholebody.ik.toggle_fix_lift(True)
        self.wholebody.start()
        return True

    @require_initialization
    def emergency_stop(self):
        """Freeze wheels, lift and arms in place. The hands hold their grasp.

        Deliberately: a stop that sprang an open hand would drop whatever is
        being carried. `graceful_shutdown` is what ramps them open, when the
        arms are already coming down.
        """
        if self.wholebody is not None:
            self.wholebody.emergency_stop()
        self.base_controller.mode = "BASE_VEL"
        self.base_controller.target_velocity = np.zeros(3, dtype=float)
        self.base.set_target_base_velocity(np.zeros(3), smooth=False)

    @require_initialization
    def open_left_gripper(self):
        if self.no_arms:
            print("left arm disabled")
            return
        self.left_arm.open_gripper()

    @require_initialization
    def close_left_gripper(self):
        if self.no_arms:
            print("left arm disabled")
            return
        self.left_arm.close_gripper()

    @require_initialization
    def open_right_gripper(self):
        if self.no_arms:
            print("right arm disabled")
            return
        self.right_arm.open_gripper()

    @require_initialization
    def close_right_gripper(self):
        if self.no_arms:
            print("right arm disabled")
            return
        self.right_arm.close_gripper()

    @require_initialization
    def get_left_joint_positions(self) -> np.ndarray:
        if self.no_arms:
            print("left arm disabled")
            return None
        return self.left_arm.get_joint_positions()

    @require_initialization
    def get_right_joint_positions(self) -> np.ndarray:
        if self.no_arms:
            print("right arm disabled")
            return None
        return self.right_arm.get_joint_positions()

    @require_initialization
    def get_left_gripper_pose(self):
        if self.no_arms:
            print("left arm disabled")
            return None
        return self.left_arm.get_gripper_pose()

    @require_initialization
    def get_right_gripper_pose(self):
        if self.no_arms:
            print("right arm disabled")
            return None
        return self.right_arm.get_gripper_pose()

    @require_initialization
    def get_bimanual_state(self) -> list:
        """
        All bimanual state in one call, for high-speed data logging.
        Flat row: [t, L_ee(7), L_q(7), L_grip, R_ee(7), R_q(7), R_grip, lift].
        """
        row = [0.0] * (1 + 7 + 7 + 1 + 7 + 7 + 1 + 1)
        row[0] = time.time()
        if self.no_arms:
            row[1:8] = [0.90724, -0.41142, 0.075, -0.04495, 0.10741, 0.11358, 0.89066] # roughly tucked
            row[8:15] = [0.0] * 7
            row[15] = 1.0 # fully open
            row[16:23] = [0.90029, 0.42914, 0.06059, 0.04051, 0.10338, -0.53731, 0.89969]
            row[23:30] = [0.0] * 7
            row[30] = 1.0
            row[31] = 0.0
            return row

        left_ee = self.get_left_ee_pose()
        right_ee = self.get_right_ee_pose()
        row[1:8] = left_ee.wxyz_xyz.tolist() if left_ee is not None else [0.0] * 7
        row[8:15] = self.left_arm.get_joint_positions().tolist()
        row[15] = self.left_arm.get_gripper_pose()
        row[16:23] = right_ee.wxyz_xyz.tolist() if right_ee is not None else [0.0] * 7
        row[23:30] = self.right_arm.get_joint_positions().tolist()
        row[30] = self.right_arm.get_gripper_pose()
        lift = self.base.get_lift_height()
        row[31] = float(lift) if lift is not None else 0.0
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flash-base-pid", action=argparse.BooleanOptionalAction, default=True,
        help="sync the swerve PID gains from the manifest before the base control "
             "loop starts, writing only the controllers that differ (default: on). "
             "--no-flash-base-pid starts on whatever gains the controllers hold, "
             "which after a SPARK power cycle are the stock ones")
    parser.add_argument(
        "--base-pid-manifest", type=Path, default=None, metavar="PATH",
        help=f"PID manifest to sync at startup (default: {DEFAULT_MANIFEST.name} — "
             f"commissioned drive gains with stock full-range steering, "
             f"floor-validated 2026-08-24; each manifest carries its own "
             f"drive_command_scale). {STOCK_MANIFEST.name} is the flash state "
             f"a power cycle reverts to and what shutdown restores; "
             f"{COMMISSIONED_MANIFEST.name} adds the clamped Kp=20 steering "
             f"loop — see its description before using it.")
    parser.add_argument(
        "--base-motion-weight", type=float, default=None, metavar="W",
        help="cost of base motion relative to arm motion in the solver (default 100). "
             "1.0 is the unweighted solve, which lets the base answer 24%% of pure EE "
             "tracker noise and reverse direction every other tick. Raising it makes "
             "'base motion is emergent' true: the arms absorb high-frequency error and "
             "the base moves only when they run out of reach.")
    parser.add_argument(
        "--base-motion-weight-yaw", type=float, default=None, metavar="W",
        help="cost of base YAW alone, independent of --base-motion-weight "
             "(which covers base_x/base_y). Default 1.0 makes yaw ~100x "
             "cheaper than driving, so the solver reaches for yaw first and "
             "the chassis alternates between rotating and driving. Raise it "
             "to make yaw less of a hair trigger; what matters is the ratio "
             "to the linear weight, not the absolute value.")
    parser.add_argument(
        "--base-motion-weight-min", type=float, default=5.0, metavar="W",
        help="what --base-motion-weight falls to when the arms run out of posture "
             "(default 5; the config value is 10). Either way it keeps the linear "
             "DOFs above --base-motion-weight-yaw so yaw stays the cheap route into "
             "the base. The cost is gated on arm "
             "manipulability, so the chassis moves to keep the arms working rather "
             "than only after they have contorted. Set equal to --base-motion-weight "
             "to disable the gate.")
    parser.add_argument(
        "--base-weight-gate", type=float, nargs=2, default=None, metavar=("ON", "FULL"),
        help="manipulability band the base cost ramps across (default 0.045 0.025). "
             "mu is 0.0506 at the home keyframe; ON sits just below it so ordinary "
             "motion does not open the gate, FULL at the quartile where the arm is "
             "visibly stretched.")
    parser.add_argument(
        "--base-max-accel", type=float, default=None, metavar="A",
        help="ceiling on how fast the commanded base velocity vector may change, m/s^2 "
             "(default 1.5). This is the reversal guard: the heading limiter treats a "
             "180-degree flip as free because a module answers it by reversing the drive, "
             "which is true for the module and false for the chassis. 0 disables.")
    parser.add_argument(
        "--base-vel-deadband", type=float, nargs=2, default=None,
        metavar=("ENTRY", "EXIT"),
        help="hysteresis deadband on the linear base command, m/s: motion "
             "starts above ENTRY and stops below EXIT. Default 0 0 "
             "(disabled) since 2026-08-25 -- with the low-pass filter on "
             "the linear request the deadband caused start-stop bursts. "
             "Pass 0.05 0.025 to restore the previous behaviour.")
    parser.add_argument(
        "--base-yaw-deadband", type=float, nargs=2, default=None,
        metavar=("ENTRY", "EXIT"),
        help="hysteresis deadband on the base yaw command, rad/s: rotation "
             "starts above ENTRY and stops below EXIT. Default 0 0 "
             "(disabled) since 2026-08-25, same reasoning as the linear "
             "pair. The pre-experiment value was 0.05 0.025; the raised "
             "experiment used 0.15 0.075.")
    parser.add_argument(
        "--base-vel-filter-tau", type=float, default=None, metavar="TAU",
        help="one-pole low-pass time constant on the linear base request, s "
             "(default 0.15). Merges the solver's burst-shaped translation "
             "requests into sustained commands; costs ~TAU of onset lag. "
             "0 disables.")
    parser.add_argument(
        "--base-recenter", type=float, nargs=2, default=None,
        metavar=("GAIN", "MAXVEL"),
        help="null-space base recentering: continuously roll the chassis "
             "toward the pose that restores the hands' home-pose offset "
             "from the base, at GAIN * distance m/s, capped at MAXVEL, with "
             "the arms counter-moving so the hands stay put. Symmetric --  "
             "not gated by arm reach, so it pulls the base home on retract "
             "as well as reach. Default 1.5 0.15 (raised from 0.5 on "
             "2026-08-26 -- see WholeBodyIKConfig.base_recenter_gain); "
             "0 0 disables.")
    parser.add_argument(
        "--base-yaw-hold-weight", type=float, default=None, metavar="W",
        help="null-space weight of 'prefer not to yaw the chassis', "
             "independent of --base-motion-weight-yaw (which prices yaw in "
             "the primary solve). Default 1 = off, and measured not worth "
             "raising: it scales yaw amplitude but leaves the sign-flip "
             "rate identical at every weight, so it cannot damp an "
             "oscillation, and the 0.15 s dispatch filter already cuts "
             "noise-driven yaw to 1.4%% of ticks. Weight 10 buys that last "
             "1.4 points for a third of the chassis's reach yaw. Exposed "
             "because before 2026-08-27 the anchor could only be switched "
             "on by raising the primary price, so --base-motion-weight-yaw "
             "2.0 silently did two unrelated things; runs at that setting "
             "had an anchor of 2.0 and need this passed to reproduce.")
    parser.add_argument(
        "--base-recenter-yaw", type=float, nargs=2, default=[2.0, 0.8],
        metavar=("GAIN", "MAXVEL"),
        help="null-space base YAW recentering: the rotational twin of "
             "--base-recenter. Continuously turns the chassis so it carries "
             "the heading the shoulders are holding, at GAIN * the mean "
             "shoulder-yaw drift from home rad/s, capped at MAXVEL rad/s, "
             "with the arms counter-rotating so the hands stay put. "
             "Symmetric and ungated, like the translation term, and driven "
             "by the shoulder load rather than the hands' bearing so it "
             "cannot cancel the yaw the primary solve spends on reach -- "
             "see WholeBodyIKConfig.base_recenter_yaw_gain. "
             "Default 2.0 0.8; the config value is 1.0 0.30. 0 0 disables.")
    parser.add_argument(
        "--base-recenter-yaw-deadzone", type=float, default=0.10, metavar="RAD",
        help="shoulder-yaw drift ignored by --base-recenter-yaw, radians "
             "(default 0.10; the config value is 0.25). Working off to one "
             "side is a comfortable "
             "posture, not a fault: without this the chassis tracked the "
             "operator's arms like a turret (2026-08-27, 3.45 rad of net "
             "wander in one 106 s run). Ordinary working load measured p95 "
             "0.15-0.25 rad, genuine wind-up 0.5-0.66. 0 disables the dead "
             "zone.")
    parser.add_argument(
        "--base-yaw-filter-tau", type=float, default=None, metavar="TAU",
        help="one-pole low-pass time constant on the base yaw request, s "
             "(default 0.15; the only yaw smoothing since the yaw deadband "
             "was removed). 0.08 is the old light setting; 0.25 measured "
             "well in replay but felt clearly worse on the floor; "
             "0 disables.")
    parser.add_argument(
        "--slam-base-pose", action=argparse.BooleanOptionalAction, default=None,
        help="close the whole-body base pose loop on the Odin VIO+lidar fix "
             "(slam/pose on :6000) instead of on dead-reckoning alone (default: "
             "on). --no-slam-base-pose runs the base open-loop with respect to "
             "the floor: slip and pushes stop being visible to the PD.")
    parser.add_argument(
        "--slam-yaw-sign", type=float, default=None, choices=(1.0, -1.0),
        metavar="SIGN",
        help="handedness of the SLAM planar frame against the IK one, +1 or -1 "
             "(default: +1). Wrong means the correction grows as you drive "
             "instead of staying small -- watch slam_base_correction_m in "
             "get_state(), or run tests/hardware/test_06_slam_pose.py.")
    parser.add_argument(
        "--slam-correction-rate", type=float, nargs=2, default=None,
        metavar=("LIN", "YAW"),
        help="ceiling on how fast the SLAM correction may move the measured "
             "base pose, m/s and rad/s (default: 1.0 2.0). Sized above the "
             "base's own speed ceilings so the correction always wins; lower "
             "it toward 0.1/0.2 to go back to bleeding off drift only.")
    parser.add_argument(
        "--slam-pose-host", type=str, default=None, metavar="HOST",
        help="where odin_pub_node publishes slam/pose (default: 192.168.1.11)")
    parser.add_argument(
        "--swerve-log", action=argparse.BooleanOptionalAction, default=True,
        help="record per-module swerve telemetry (commanded and measured steer angle, "
             "drive velocity and drive position) to artifacts/wholebody_logs/swerve/ for "
             "the life of the base control loop. Independent of whole-body control, so it "
             "covers joystick runs and --no-arms.")
    parser.add_argument(
        "--swerve-log-hz", type=float, default=SWERVE_LOG_HZ, metavar="HZ",
        help=f"swerve log sample rate (default {SWERVE_LOG_HZ:g}, matching the SPARK "
             f"periodic status 2 period)")
    parser.add_argument(
        "--restore-base-pid", action=argparse.BooleanOptionalAction, default=True,
        help="on shutdown, write the SPARK stock gains back over the commissioned "
             "ones, so nothing that opens the bus afterwards inherits them. Has no "
             "effect if the gains were never flashed (--no-flash-base-pid).")
    parser.add_argument(
        "--base-pid-stock-manifest", type=Path, default=None, metavar="PATH",
        help=f"stock-gain manifest restored on shutdown (default: {STOCK_MANIFEST.name}). "
             f"The restore is skipped when startup already applied it.")
    parser.add_argument(
        "--no-arms", action="store_true",
        help="skip both arm controllers and their startup joint homing")
    parser.add_argument(
        "--gripper", choices=("none", "dynamixel", "native"), default="none",
        help="which gripper hardware is fitted (default: none, gripper "
             "commands dropped). 'dynamixel' drives the U2D2 servos on "
             "/dev/ttyUSB0 and CALIBRATES THEM AT STARTUP by closing and "
             "reopening each gripper -- only pass it with the hands attached "
             "and clear. 'native' routes the gripper through the arm's own "
             "CAN gripper via nerolib.")
    parser.add_argument(
        "--no-base-motion", action="store_true",
        help="keep the whole-body IK base fixed and disable wheel dispatch")
    parser.add_argument(
        "--no-lift-motion", action="store_true",
        help="keep the whole-body IK lift fixed and disable lift dispatch")
    parser.add_argument(
        "--no-console-log", action="store_true",
        help="don't mirror console output to artifacts/wholebody_logs/ "
             "(mirroring is on by default)")
    parser.add_argument(
        "--no-trajectory-log", action="store_true",
        help="don't record per-tick joint/EE trajectories to "
             "artifacts/wholebody_logs/trajectories/ (recording is on by "
             "default -- this is the raw data null-space-projection work "
             "will need later, not just a debugging aid)")
    parser.add_argument(
        "--posture-stiffen-joint7", action=argparse.BooleanOptionalAction,
        default=True,
        help="raise the posture cost on *_arm_joint7 by "
             "--posture-joint7-scale (default: on). Carried over from before "
             "the null-space projector existed, when it was the only lever "
             "against joint7 wobble; it has never been A/B'd against "
             "--redundancy-resolution dls_projector, which addresses the same "
             "problem structurally. Try --no-posture-stiffen-joint7 if the "
             "elbow-swivel objective seems short of authority -- both compete "
             "for the same 2 null-space DOFs.")
    parser.add_argument(
        "--posture-refresh-target", action=argparse.BooleanOptionalAction,
        default=True,
        help="keep the posture task's reference tracking the current "
             "configuration every solve instead of freezing it at startup "
             "(default: on; see refresh_posture_target in "
             "robot/arm/wholebody_ik.py). Off leaves the reference to go "
             "stale over a session, which measurably degraded long reaches.")
    parser.add_argument(
        "--posture-joint7-scale", type=float, default=3.0,
        help="only with --posture-stiffen-joint7: multiplier on "
             "arm_posture_cost for *_arm_joint7 (default: %(default)s, i.e. "
             "3x stiffer -- see the measured-response table in "
             "WholeBodyIKConfig.arm_posture_cost_overrides's docstring for "
             "what other multipliers do)")
    parser.add_argument(
        "--redundancy-resolution", choices=("soft", "dls_projector"),
        default="dls_projector",
        help="how the null space is resolved (default: %(default)s, the "
             "configuration validated on hardware 2026-08-22). 'soft' is the "
             "original behaviour -- EE tasks and posture as competing soft "
             "costs in one QP -- kept as a baseline and still what the sim "
             "node and IK demo use. It cannot fix the elbow branch-flipping: "
             "we pushed the EE:posture weight ratio to ~100,000:1 and the "
             "problem neither shrank nor moved, which is what established "
             "that it is structural. 'dls_projector' resolves the redundancy "
             "properly -- damped-least-squares pseudoinverse for the EE task "
             "plus an exact null-space projector for posture, elbow swivel "
             "and the optional objectives -- and measured 2.2 vs 5.3 elbow "
             "flips per 1000 ticks against 'soft' on hardware. See "
             "artifacts/wholebody_logs/posture_fix_commands.md.")
    parser.add_argument(
        "--dls-damping", type=float, default=0.02,
        help="only with --redundancy-resolution dls_projector: Tikhonov/DLS "
             "damping λ for the pseudoinverse J⁺=Jᵀ(JJᵀ+λ²I)⁻¹ (default: "
             "%(default)s). Larger = more robust near singularities, at the "
             "cost of looser EE tracking there; 0.01-0.1 is the usual range.")
    parser.add_argument(
        "--nullspace-swivel-weight", type=float, default=1.0,
        help="only with --redundancy-resolution dls_projector: weight on the "
             "elbow-swivel objective, which holds each arm's elbow at a fixed "
             "angle about its shoulder->wrist axis instead of letting the "
             "solver pick whichever elbow branch is locally cheapest. This is "
             "the direct fix for the elbow branch-flipping documented in "
             "artifacts/wholebody_logs/posture_fix_commands.md. Measured elbow "
             "drift over a reversing target with an abrupt disturbance: 0 -> "
             "26.9deg, 1.0 -> 8.2deg, 5.0 -> 1.9deg, EE tracking unchanged. "
             "0 disables (default: %(default)s)")
    parser.add_argument(
        "--elbow-swivel-gain", type=float, default=0.05,
        help="only with --redundancy-resolution dls_projector: proportional "
             "gain pulling each elbow back to its target swivel angle "
             "(default: %(default)s)")
    parser.add_argument(
        "--elbow-swivel-target", type=float, default=None, metavar="RAD",
        help="only with --redundancy-resolution dls_projector: hold both "
             "elbows at this swivel angle in radians. Omit (the default) to "
             "latch each arm's angle from its pose at startup, which keeps "
             "the elbow branch the arm homes into rather than choosing one.")
    parser.add_argument(
        "--nullspace-continuity-weight", type=float, default=0.5,
        help="only with --redundancy-resolution dls_projector: weight on "
             "||qdot - qdot_prev||. The default is the value the hardware "
             "runs; kinematic replay measured raising it counterproductive -- "
             "it resists *change*, which also means it "
             "perpetuates existing null-space motion instead of letting "
             "posture bleed it off (raising it worsened null-space jerk, "
             "null-space speed and elbow drift at every swivel weight) -- so "
             "0 remains the replay-preferred setting (default: %(default)s)")
    parser.add_argument(
        "--enable-manipulability", action="store_true",
        help="only with --redundancy-resolution dls_projector: add a gated "
             "manipulability-maximisation objective, pushing each arm away "
             "from singular configurations. Costs roughly 19 ms extra per "
             "solve (finite-differenced gradient, 14 perturbed Jacobians per "
             "arm per iteration), which does NOT fit the 30 Hz budget -- "
             "treat as a diagnostic, not a production setting.")
    parser.add_argument(
        "--manipulability-weight", type=float, default=0.5,
        help="only with --enable-manipulability (default: %(default)s)")
    parser.add_argument(
        "--manipulability-gain", type=float, default=0.02,
        help="only with --enable-manipulability: step size in radians per "
             "iteration along the normalised ascent direction "
             "(default: %(default)s)")
    # ── Gated experiments (2026-08-25 wave). Every flag below defaults to
    #    off / current behaviour; enable individually to A/B. Restore point:
    #    git tag pre-gates. ────────────────────────────────────────────────
    parser.add_argument(
        "--arm-joint-deadband", type=float, default=None, metavar="RAD",
        help="[T2] per-joint deadband below which a solved arm target is "
             "not dispatched (default: keep config value, currently 0.05 "
             "rad ~ 2.9 deg -- the cause of the slow-motion staircase; try "
             "0.005)")
    parser.add_argument(
        "--target-leash-m", type=float, default=None, metavar="M",
        help="[S2] cap how far an EE target may sit from the current EE "
             "pose; excess is forgotten each tick, killing clutch wind-up "
             "at the server (default 0.15; 0 = off)")
    parser.add_argument(
        "--target-leash-rad", type=float, default=None, metavar="RAD",
        help="[S2] same leash for orientation, along the geodesic "
             "(default 0.8; 0 = off)")
    parser.add_argument(
        "--base-leash-m", type=float, default=None, metavar="M",
        help="cap how far the solver's belief of the chassis pose may sit "
             "from the dead-reckoned pose; the excess is forgotten each tick, "
             "so the solver sees the base falling behind and reaches with the "
             "arms instead (default: keep config value, currently 0.2; "
             "0 = off)")
    parser.add_argument(
        "--base-leash-rad", type=float, default=0.24, metavar="RAD",
        help="the same leash on yaw (default 0.24; the config value is "
             "0 = off)")
    parser.add_argument(
        "--base-pose-kp-xy", type=float, default=None, metavar="KP",
        help="proportional gain of the base pose PD, in (m/s)/m. At the "
             "default 1.5 the base does not reach base_max_lin_vel until it "
             "is 0.167 m behind, and that standing error is felt as lag; "
             "raising it shortens the error needed for a given speed "
             "(default: keep config value, currently 1.5; try 3.0)")
    parser.add_argument(
        "--base-pose-ff-tau", type=float, default=None, metavar="TAU",
        help="low-pass on the base pose feedforward, seconds. The "
             "feedforward differentiates the target, so it amplifies solver "
             "jitter -- on 2026-08-26 it tripled the base yaw sign-flip rate "
             "(0.4 -> 1.1 /s), felt as twitch. This buys that back for some "
             "onset lag (default: keep config value, currently 0 = off; "
             "try 0.05)")
    parser.add_argument(
        "--base-pose-ff-gain", type=float, default=None, metavar="GAIN",
        help="feed the base pose target's own rate forward into the PD, so "
             "speed stops depending on accumulated error -- this is what lets "
             "a short --base-leash-m coexist with a responsive base "
             "(default: keep config value, currently 1.0; 0 = off)")
    parser.add_argument(
        "--nullspace-home-gain", type=float, default=1.0, metavar="GAIN",
        help="[S1] null-space pull of the arm joints toward the home "
             "posture, in (rad/s per rad of error); the missing recovery "
             "force for contorted poses. Default 1.0; the config value is "
             "0.3. 0 = off")
    parser.add_argument(
        "--nullspace-home-weight", type=float, default=40.0, metavar="W",
        help="[S1] weight of the home attractor in the secondary stack "
             "(default 40; the config value is 1.0)")
    parser.add_argument(
        "--nullspace-home-max-vel", type=float, default=None, metavar="V",
        help="[S1] per-joint cap on the home-attractor desire in rad/s "
             "(default: config 0.3)")
    parser.add_argument(
        "--constrained-primary", action=argparse.BooleanOptionalAction,
        default=True,
        help="[S3] solve the primary EE step subject to the joint/collision "
             "inequalities instead of clipping afterwards, so blocked arm "
             "motion reroutes through base/lift (the backward-motion fix). "
             "Falls back to the unconstrained step on any QP failure. "
             "Costs ~1 extra small QP per iteration. On by default; "
             "--no-constrained-primary restores the clip.")
    parser.add_argument(
        "--dls-task-weighting", action=argparse.BooleanOptionalAction,
        default=True,
        help="[S4a] apply ee_position_cost/ee_orientation_cost row scaling "
             "inside dls_projector (off, those knobs are silently ignored "
             "and 1 m weighs the same as 1 rad). On by default; "
             "--no-dls-task-weighting restores the unweighted stack.")
    parser.add_argument(
        "--dls-adaptive-damping", type=float, default=None, metavar="SIGMA",
        help="[S4b] sigma_min threshold below which lambda ramps from "
             "--dls-damping up to --dls-damping-max; keeps rotation crisp "
             "away from singularity, softens only near it (default 0.05; "
             "0 = off)")
    parser.add_argument(
        "--dls-damping-max", type=float, default=None, metavar="LAM",
        help="[S4b] lambda at sigma_min = 0 with --dls-adaptive-damping "
             "(default: config 0.2)")
    parser.add_argument(
        "--swivel-parallel-ref", action=argparse.BooleanOptionalAction,
        default=True,
        help="[S5a] parallel-transported swivel reference: removes the "
             "reference-frame step at |u_z| = 0.9 that yanks the elbow "
             "during high/low reaches. On by default; "
             "--no-swivel-parallel-ref restores the z/x convention.")
    parser.add_argument(
        "--swivel-relatch-err", type=float, default=0.25, metavar="RAD",
        help="[S5b] if the swivel error stays above this for "
             "--swivel-relatch-time, accept the branch the arm is actually "
             "in instead of fighting it (default 0.25; the config value is "
             "1.57. 0 = off)")
    parser.add_argument(
        "--swivel-relatch-time", type=float, default=None, metavar="S",
        help="[S5b] dwell before a re-latch (default: config 1.0 s)")
    parser.add_argument(
        "--no-solver-diagnostics", action="store_true",
        help="[S7] skip the per-solve sigma_min/manipulability/swivel/"
             "collision-row diagnostics (on by default; two 6x7 SVDs per "
             "solve)")
    add_hand_args(parser)
    args = parser.parse_args()

    if not args.no_console_log:
        from robot.utils.console_log import start_console_log
        start_console_log("yor", _ROOT / "artifacts" / "wholebody_logs")

    wholebody_config = WholeBodyHardwareConfig(
        enable_base_motion=not args.no_base_motion,
        enable_lift_motion=not args.no_lift_motion,
        record_trajectories=not args.no_trajectory_log,
        **({} if args.base_max_accel is None
           else {"base_max_accel": args.base_max_accel}),
        **({} if args.base_vel_deadband is None
           else {"base_vel_deadband": args.base_vel_deadband[0],
                 "base_vel_deadband_exit": args.base_vel_deadband[1]}),
        **({} if args.base_yaw_deadband is None
           else {"base_yaw_deadband": args.base_yaw_deadband[0],
                 "base_yaw_deadband_exit": args.base_yaw_deadband[1]}),
        **({} if args.base_yaw_filter_tau is None
           else {"base_yaw_filter_tau": args.base_yaw_filter_tau}),
        **({} if args.base_vel_filter_tau is None
           else {"base_vel_filter_tau": args.base_vel_filter_tau}),
        **({} if args.arm_joint_deadband is None
           else {"arm_joint_deadband_rad": args.arm_joint_deadband}),
        **({} if args.target_leash_m is None
           else {"target_leash_m": args.target_leash_m}),
        **({} if args.target_leash_rad is None
           else {"target_leash_rad": args.target_leash_rad}),
        **({} if args.base_leash_m is None
           else {"base_leash_m": args.base_leash_m}),
        **({} if args.base_leash_rad is None
           else {"base_leash_rad": args.base_leash_rad}),
        **({} if args.base_pose_kp_xy is None
           else {"base_pose_kp_xy": args.base_pose_kp_xy}),
        **({} if args.base_pose_ff_gain is None
           else {"base_pose_ff_gain": args.base_pose_ff_gain}),
        **({} if args.base_pose_ff_tau is None
           else {"base_pose_ff_tau": args.base_pose_ff_tau}),
        **({} if args.slam_base_pose is None
           else {"enable_slam_base_pose": args.slam_base_pose}),
        **({} if args.slam_yaw_sign is None
           else {"slam_yaw_sign": args.slam_yaw_sign}),
        **({} if args.slam_correction_rate is None
           else {"slam_correction_max_lin_rate": args.slam_correction_rate[0],
                 "slam_correction_max_yaw_rate": args.slam_correction_rate[1]}),
        **({} if args.slam_pose_host is None
           else {"slam_pose_host": args.slam_pose_host}),
    )

    # Same base tuning WholeBodyController would build internally (see its
    # __init__ -- passing any ik_config here bypasses that block entirely,
    # so it has to be reproduced rather than dropped).
    ik_config = WholeBodyIKConfig(
        dt=1.0 / wholebody_config.control_hz,
        solver="pyqpmad",
        max_iters=10,
        base_posture_cost=1e-1,
        lift_posture_cost=1e-4,
        arm_posture_cost=1e-3,
        refresh_posture_target=args.posture_refresh_target,
        arm_posture_cost_overrides=(
            {
                "left_arm_joint7": 1e-3 * args.posture_joint7_scale,
                "right_arm_joint7": 1e-3 * args.posture_joint7_scale,
            }
            if args.posture_stiffen_joint7
            else {}
        ),
        redundancy_resolution=args.redundancy_resolution,
        dls_damping=args.dls_damping,
        **({} if args.base_motion_weight is None
           else {"base_motion_weight": args.base_motion_weight}),
        **({} if args.base_motion_weight_min is None
           else {"base_motion_weight_min": args.base_motion_weight_min}),
        **({} if args.base_motion_weight_yaw is None
           else {"base_motion_weight_yaw": args.base_motion_weight_yaw}),
        **({} if args.base_yaw_hold_weight is None
           else {"base_yaw_hold_weight": args.base_yaw_hold_weight}),
        **({} if args.base_weight_gate is None
           else {"base_weight_gate_on": args.base_weight_gate[0],
                 "base_weight_gate_full": args.base_weight_gate[1]}),
        **({} if args.base_recenter is None
           else {"base_recenter_gain": args.base_recenter[0],
                 "base_recenter_max_vel": args.base_recenter[1]}),
        **({} if args.base_recenter_yaw is None
           else {"base_recenter_yaw_gain": args.base_recenter_yaw[0],
                 "base_recenter_yaw_max_vel": args.base_recenter_yaw[1]}),
        **({} if args.base_recenter_yaw_deadzone is None
           else {"base_recenter_yaw_deadzone": args.base_recenter_yaw_deadzone}),
        nullspace_swivel_weight=args.nullspace_swivel_weight,
        elbow_swivel_gain=args.elbow_swivel_gain,
        elbow_swivel_targets=(
            {} if args.elbow_swivel_target is None
            else {"left": args.elbow_swivel_target,
                  "right": args.elbow_swivel_target}
        ),
        nullspace_continuity_weight=args.nullspace_continuity_weight,
        enable_manipulability=args.enable_manipulability,
        manipulability_weight=args.manipulability_weight,
        manipulability_gain=args.manipulability_gain,
        # ── Gated experiments ──
        **({} if args.nullspace_home_gain is None
           else {"nullspace_home_gain": args.nullspace_home_gain}),
        **({} if args.nullspace_home_weight is None
           else {"nullspace_home_weight": args.nullspace_home_weight}),
        **({} if args.nullspace_home_max_vel is None
           else {"nullspace_home_max_vel": args.nullspace_home_max_vel}),
        constrained_primary=args.constrained_primary,
        dls_task_weighting=args.dls_task_weighting,
        **({} if args.dls_adaptive_damping is None
           else {"dls_adaptive_damping_sigma": args.dls_adaptive_damping}),
        **({} if args.dls_damping_max is None
           else {"dls_damping_max": args.dls_damping_max}),
        swivel_parallel_ref=args.swivel_parallel_ref,
        **({} if args.swivel_relatch_err is None
           else {"swivel_relatch_err_rad": args.swivel_relatch_err}),
        **({} if args.swivel_relatch_time is None
           else {"swivel_relatch_after_s": args.swivel_relatch_time}),
        record_solver_diagnostics=not args.no_solver_diagnostics,
    )
    gates_on = [label for label, on in (
        (f"arm_deadband={args.arm_joint_deadband}", args.arm_joint_deadband is not None),
        (f"leash_m={args.target_leash_m}", args.target_leash_m is not None),
        (f"leash_rad={args.target_leash_rad}", args.target_leash_rad is not None),
        (f"base_leash_m={args.base_leash_m}", args.base_leash_m is not None),
        (f"base_leash_rad={args.base_leash_rad}", args.base_leash_rad is not None),
        (f"base_kp_xy={args.base_pose_kp_xy}", args.base_pose_kp_xy is not None),
        (f"base_ff_gain={args.base_pose_ff_gain}", args.base_pose_ff_gain is not None),
        (f"base_ff_tau={args.base_pose_ff_tau}", args.base_pose_ff_tau is not None),
        (f"base_yaw_weight={args.base_motion_weight_yaw}", args.base_motion_weight_yaw is not None),
        (f"home_gain={args.nullspace_home_gain}", args.nullspace_home_gain is not None),
        ("NO constrained_primary", not args.constrained_primary),
        ("NO dls_task_weighting", not args.dls_task_weighting),
        (f"adaptive_damping={args.dls_adaptive_damping}", args.dls_adaptive_damping is not None),
        ("NO swivel_parallel_ref", not args.swivel_parallel_ref),
        (f"swivel_relatch={args.swivel_relatch_err}", args.swivel_relatch_err is not None),
    ) if on]
    print("[yor] overrides: "
          + (", ".join(gates_on) if gates_on else "none (config defaults)"))
    print(
        f"[yor] posture-fix: stiffen_joint7={args.posture_stiffen_joint7}"
        + (f" (scale={args.posture_joint7_scale:g}x)" if args.posture_stiffen_joint7 else "")
        + f", refresh_target={args.posture_refresh_target}"
        + f", redundancy_resolution={args.redundancy_resolution}"
        + (f" (dls_damping={args.dls_damping:g})" if args.redundancy_resolution == "dls_projector" else "")
    )
    if args.redundancy_resolution == "dls_projector":
        print(
            f"[yor] null-space: swivel_weight={args.nullspace_swivel_weight:g}"
            + f" (gain={args.elbow_swivel_gain:g}, target="
            + ("latched-from-pose" if args.elbow_swivel_target is None
               else f"{args.elbow_swivel_target:g} rad") + ")"
            + f", continuity_weight={args.nullspace_continuity_weight:g}"
            + f", manipulability={args.enable_manipulability}"
            + (f" (weight={args.manipulability_weight:g},"
               f" gain={args.manipulability_gain:g})"
               if args.enable_manipulability else "")
        )

    yor = YOR(
        no_arms=args.no_arms,
        wholebody_config=wholebody_config,
        ik_config=ik_config,
        flash_base_pid=args.flash_base_pid,
        base_pid_manifest=args.base_pid_manifest,
        restore_base_pid=args.restore_base_pid,
        swerve_log=args.swerve_log,
        swerve_log_hz=args.swerve_log_hz,
        base_pid_stock_manifest=args.base_pid_stock_manifest,
        gripper=args.gripper,
        hands=hands_from_args(args),
    )
    server = None
    shutdown_started = False

    def graceful_shutdown():
        """Bring the robot down in order, and get all the way to the end.

        Registered *before* init(), because init() raises for a living — a
        failed lift home, a failed arm home, a failed PID sync — and the
        controllers are already open and possibly already written by then.
        Everything here therefore has to tolerate a robot that was only
        half-started: `server` is None until the RPC layer exists, and each
        hardware step runs independently, because a failure in one must not
        skip the ones after it. The base PID restore in particular sits
        behind two steps that both touch hardware.
        """
        nonlocal shutdown_started
        if shutdown_started:
            return
        shutdown_started = True

        print("\nRPC Server stopping...")

        def attempt(label, action):
            try:
                action()
            except Exception as exc:
                print(f"[YOR] shutdown: {label} failed ({exc!r}); continuing")

        if yor.wholebody is not None:
            attempt("whole-body stop", yor.wholebody.stop)

        # Before the arms drop: ramp the fingers open and disable them.
        if yor.hands is not None:
            attempt("hands stop", yor.hands.stop)

        # Stop hardware workers before RPC teardown/interpreter shutdown so
        # PicoLift cannot keep reconnecting after Ctrl-C.
        attempt("swerve log stop", yor.stop_swerve_log)
        attempt("base control loop stop", yor.base.stop_control)

        # Then hand the swerve controllers back in stock condition, while the
        # device handles are still open and nothing is commanding a wheel.
        # Never raises on its own; see YOR._restore_base_pid_gains.
        yor._restore_base_pid_gains()

        attempt("lift serial shutdown", yor.base._pico_lift._shutdown)

        if server is not None:
            attempt("RPC server stop", server.stop)

        if not yor.no_arms:
            if yor.left_arm is not None:
                input("\n[YOR] Press ENTER to drop LEFT arm...")
                attempt("left arm stop", yor.left_arm.stop)

            if yor.right_arm is not None:
                input("\n[YOR] Press ENTER to drop RIGHT arm...")
                attempt("right arm stop", yor.right_arm.stop)

    atexit.register(graceful_shutdown)

    yor.init()
    server = RPCServer(yor, port=YOR_PORT, threaded=True)
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        graceful_shutdown()


if __name__ == "__main__":
    main()
