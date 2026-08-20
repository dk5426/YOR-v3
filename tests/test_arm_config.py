"""
test_arm_config.py — the agreed arm limits and gripper state, proved.

Three separate numbers decide how fast a joint may actually move, and they
have to agree or the slowest one silently wins:

    robot/arm/arm.py              joint_vel_max / joint_acc_max  (nerolib)
    robot/wholebody_control.py    arm_max_vel_rad_s              (look-ahead clamp)
    the solver's own limits       robot/arm/wholebody_ik.py

and one boolean decides whether a gripper that is not fitted gets commanded.

nerolib and the Dynamixel SDK are not installed on a development machine, so
ArmNode is exercised against stand-ins registered in `sys.modules`. That is
enough to prove what this file is about: which values reach ControllerConfig,
and which gripper values reach `set_target`.

    python tests/test_arm_config.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# ─────────────────────────────────────────────────────────────────────────────
# Stand-ins for the compiled dependencies
# ─────────────────────────────────────────────────────────────────────────────

class FakeControllerConfig:
    """Accepts whatever ArmNode assigns, and remembers all of it."""


class FakeNeroController:
    instances: list["FakeNeroController"] = []

    def __init__(self, config):
        self.config = config
        self.targets: list[dict] = []
        self.gains: list = []
        self.homed = 0
        self.state = types.SimpleNamespace(
            pos=[0.0] * 7, vel=[0.0] * 7, torque=[0.0] * 7, gripper_pos=1.0)
        FakeNeroController.instances.append(self)

    def start(self):
        return True

    def set_target(self, new_target_pos, new_target_gripper_pos, minimum_duration):
        self.targets.append({
            "pos": list(new_target_pos),
            "gripper": float(new_target_gripper_pos),
            "duration": float(minimum_duration),
        })

    def get_current_state(self):
        return self.state

    def set_gain(self, gain):
        self.gains.append(gain)

    def reset_to_home(self):
        self.homed += 1

    def stop(self):
        pass


def install_stubs() -> None:
    nerolib = types.ModuleType("nerolib")
    nerolib.NeroController = FakeNeroController
    nerolib.ControllerConfig = FakeControllerConfig
    nerolib.FirmwareVersion = types.SimpleNamespace(DEFAULT="default")
    nerolib.JointState = object
    nerolib.Gain = type("Gain", (), {})
    nerolib.ControlMode = types.SimpleNamespace()
    nerolib.MoveMode = types.SimpleNamespace()
    sys.modules["nerolib"] = nerolib

    dxl_sdk = types.ModuleType("dynamixel_sdk")
    dxl_sdk.COMM_SUCCESS = 0
    dxl_sdk.PortHandler = object
    dxl_sdk.PacketHandler = object
    sys.modules["dynamixel_sdk"] = dxl_sdk


install_stubs()

from robot.arm.arm import ArmNode                            # noqa: E402
from robot.wholebody_control import WholeBodyHardwareConfig   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Agreed values
# ─────────────────────────────────────────────────────────────────────────────

NATIVE_VEL_LIMIT = 3.0     # rad/s
NATIVE_ACC_LIMIT = 15.0    # rad/s^2
WHOLEBODY_VEL_LIMIT = 3.0  # rad/s
ARM_CONTROLLER_HZ = 250.0

# Commissioned construction-time gains -- matches YOR_D's legacy (non-v2) arm
# mode, confirmed as the profile to keep for YOR-v3. The stiffer kp=15 remains
# available through `set_firm_mode` at runtime.
DEFAULT_KP = 8.0
DEFAULT_KD = 1.0

# Gravity compensation is hardcoded on in ArmNode.__init__, unconditionally,
# so it is already active from construction (not just after the first mode
# preset call).
DEFAULT_GRAVITY_COMP = True

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def build_arm(**kwargs) -> ArmNode:
    FakeNeroController.instances.clear()
    return ArmNode(can_port=kwargs.pop("can_port", "can_left"), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def test_native_limits() -> None:
    print("\nnative arm limits reach nerolib")
    for side, is_left in (("left", True), ("right", False)):
        arm = build_arm(can_port=f"can_{side}", is_left_arm=is_left)
        cfg = arm.config
        check(f"{side}: joint_vel_max is {NATIVE_VEL_LIMIT} rad/s on all 7 joints",
              list(cfg.joint_vel_max) == [NATIVE_VEL_LIMIT] * 7, str(cfg.joint_vel_max))
        check(f"{side}: joint_acc_max is {NATIVE_ACC_LIMIT} rad/s^2 on all 7 joints",
              list(cfg.joint_acc_max) == [NATIVE_ACC_LIMIT] * 7, str(cfg.joint_acc_max))
        check(f"{side}: native interpolation runs at {ARM_CONTROLLER_HZ:.0f} Hz",
              cfg.controller_freq_hz == ARM_CONTROLLER_HZ,
              str(cfg.controller_freq_hz))


def test_wholebody_clamp() -> None:
    print("\nwhole-body per-joint look-ahead clamp")
    cfg = WholeBodyHardwareConfig()
    check(f"arm_max_vel_rad_s is {WHOLEBODY_VEL_LIMIT} rad/s",
          cfg.arm_max_vel_rad_s == WHOLEBODY_VEL_LIMIT, str(cfg.arm_max_vel_rad_s))
    check("the clamp does not undercut the native limit",
          cfg.arm_max_vel_rad_s >= NATIVE_VEL_LIMIT,
          f"{cfg.arm_max_vel_rad_s} vs native {NATIVE_VEL_LIMIT}")

    lead = cfg.arm_max_vel_rad_s * cfg.arm_command_lookahead_s
    check("command look-ahead is bounded to 300 mrad",
          abs(lead - 0.30) < 1e-12, f"{lead*1000:.1f} mrad")
    check("whole-body solve rate matches the 30 Hz teleop rate 1:1",
          cfg.control_hz == 30.0, f"{cfg.control_hz:.0f} Hz")
    check("arm dispatch runs decoupled from the solve loop, at 90 Hz",
          cfg.arm_dispatch_hz == 90.0, str(cfg.arm_dispatch_hz))
    check("arm dispatch interpolates each solved target over 3 sub-steps",
          cfg.arm_interpolation_steps == 3, str(cfg.arm_interpolation_steps))
    check("arm sub-target minimum trajectory duration is 10.8 ms",
          cfg.arm_preview_time == 0.0108, str(cfg.arm_preview_time))
    check("WBC arm deadband matches the 50 mrad nerolib home tolerance",
          cfg.arm_joint_deadband_rad == 0.05,
          str(cfg.arm_joint_deadband_rad))
    check("WBC arm-state feedback is open-loop after startup",
          cfg.use_measured_arm_state is False,
          str(cfg.use_measured_arm_state))


def test_gripper_disabled_by_default() -> None:
    print("\nnative gripper is off when none is fitted")
    arm = build_arm()
    nero = FakeNeroController.instances[-1]

    check("native_gripper defaults to False", arm.native_gripper is False)
    check("dynamixel_gripper defaults to False", arm.dynamixel_gripper is False)

    # A gripper value from teleop must not become a gripper command.
    arm.set_joint_target(np.zeros(7), gripper_target=0.0)
    arm.set_joint_target(np.zeros(7), gripper_target=0.35)
    arm.set_joint_target(np.zeros(7))
    grippers = [t["gripper"] for t in nero.targets]
    check("teleop gripper values never vary the commanded gripper",
          len(set(grippers)) == 1, str(grippers))
    check("the commanded gripper stays open", grippers == [1.0] * 3, str(grippers))

    before = len(nero.targets)
    arm.open_gripper()
    arm.close_gripper()
    check("open_gripper / close_gripper send nothing",
          len(nero.targets) == before, f"{len(nero.targets) - before} extra commands")

    check("home() drives no gripper", arm.home(0.0) is None and nero.homed == 1)


def test_gripper_enabled_when_fitted() -> None:
    print("\nnative gripper still works when one is fitted")
    arm = build_arm(native_gripper=True)
    nero = FakeNeroController.instances[-1]

    arm.set_joint_target(np.zeros(7), gripper_target=0.25)
    check("an explicit gripper value is passed through",
          nero.targets[-1]["gripper"] == 0.25, str(nero.targets[-1]["gripper"]))

    arm.close_gripper()
    check("close_gripper closes", nero.targets[-1]["gripper"] == 0.0,
          str(nero.targets[-1]["gripper"]))
    arm.open_gripper()
    check("open_gripper opens", nero.targets[-1]["gripper"] == 1.0,
          str(nero.targets[-1]["gripper"]))


def test_construction_defaults() -> None:
    """What an ArmNode is before anybody calls a mode preset.

    Home position was never in question. The gains and the gravity-compensation
    flag were: both were re-recorded when the native-gripper work landed, so
    this states what they are now and why, rather than asserting they never
    moved. A failure here still means somebody changed how the arm comes up —
    which is worth knowing about deliberately.
    """
    print("\nconstruction-time arm settings")
    left = build_arm(can_port="can_left", is_left_arm=True)
    right = build_arm(can_port="can_right", is_left_arm=False)
    check("left home position unchanged",
          left.config.home_position == [0.0, 1.32, -1.71, 1.31, 0.0, 0.0, 0.0],
          str(left.config.home_position))
    check("right home position unchanged",
          right.config.home_position == [0.0, 1.32, 1.71, 1.31, 0.0, 0.0, 0.0],
          str(right.config.home_position))
    check(f"default gains are kp={DEFAULT_KP} kd={DEFAULT_KD} on all 7 joints",
          list(left.config.default_kp) == [DEFAULT_KP] * 7
          and list(left.config.default_kd) == [DEFAULT_KD] * 7,
          f"kp={left.config.default_kp[0]} kd={left.config.default_kd[0]}")
    check("both arms come up on the same gains",
          list(right.config.default_kp) == list(left.config.default_kp)
          and list(right.config.default_kd) == list(left.config.default_kd),
          f"kp={right.config.default_kp[0]} kd={right.config.default_kd[0]}")
    check("gravity compensation is on at construction",
          left.config.gravity_compensation is DEFAULT_GRAVITY_COMP,
          str(left.config.gravity_compensation))

    # Gravity comp being on from the start is what makes the scale load-bearing:
    # it now shapes the very first command, not just whatever happens after a
    # preset is applied. So prove the constructor argument actually arrives.
    check("gravity_comp_scale defaults to 1.0", left.config.gravity_comp_scale == 1.0,
          str(left.config.gravity_comp_scale))
    scaled = build_arm(can_port="can_left", gravity_comp_scale=0.5)
    check("an explicit gravity_comp_scale reaches the config",
          scaled.config.gravity_comp_scale == 0.5,
          str(scaled.config.gravity_comp_scale))


def test_yor_disables_gripper_explicitly() -> None:
    print("\nthe hardware node says so at the construction site")
    source = (_REPO / "robot/yor.py").read_text()
    check("yor.py passes native_gripper=False for both arms",
          source.count("native_gripper=False") == 2,
          f"{source.count('native_gripper=False')} occurrences")
    check("yor.py still passes dynamixel_gripper=False for both arms",
          source.count("dynamixel_gripper=False") == 2)


def main() -> int:
    for test in (
        test_native_limits,
        test_wholebody_clamp,
        test_gripper_disabled_by_default,
        test_gripper_enabled_when_fitted,
        test_construction_defaults,
        test_yor_disables_gripper_explicitly,
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
