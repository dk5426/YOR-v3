"""
test_interface_contract.py — freeze the interfaces the teleop stack depends on.

The whole-body improvement work changes *internals*: arm clamps, a streamed
lift velocity, swerve PID preflight. None of it may change what
robot/teleop/wholebody_teleop.py, robot/teleop/joystick.py or any RPC client
can call, or what those calls mean. This file is the regression net around
that, and it is deliberately the first thing to run.

Four boundaries are recorded here:

    wholebody_teleop.py  ->  YOR                 (the RPC surface)
    YOR                  ->  WholeBodyController (delegation)
    WholeBodyController  ->  ArmNode / Base      (dispatch)
    PicoLift             ->  Arduino sketch      (the serial protocol)

Everything is read with `ast` and plain text, so this runs on a laptop where
nerolib and sparkcan_py — and therefore robot/yor.py — cannot be imported.

    python tests/test_interface_contract.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Frozen signatures
#
# Recorded from the tree before the whole-body improvement work started. A
# method may be ADDED, but an existing one may not change name, arguments or
# defaults — that is exactly what would break a client that only knows the old
# call. Grow this map when a genuinely new call is added; never edit a line to
# make a failing check pass.
# ─────────────────────────────────────────────────────────────────────────────

FROZEN_YOR = {
    "close_left_gripper": "(self)",
    "close_right_gripper": "(self)",
    "emergency_stop": "(self)",
    "follow_path": "(self, path=None)",
    "get_arm_relative_pose": "(self)",
    "get_base_encoders": "(self)",
    "get_base_velocity": "(self)",
    "get_bimanual_state": "(self)",
    "get_cmd_vel": "(self)",
    "get_left_ee_pose": "(self)",
    "get_left_gripper_pose": "(self)",
    "get_left_joint_positions": "(self)",
    "get_lift_height": "(self)",
    "get_lift_position": "(self)",
    "get_lift_status": "(self)",
    "get_nav_debug": "(self)",
    "get_pose": "(self)",
    "get_right_ee_pose": "(self)",
    "get_right_gripper_pose": "(self)",
    "get_right_joint_positions": "(self)",
    "get_state": "(self)",
    "home_left_arm": "(self)",
    "home_right_arm": "(self)",
    "init": "(self)",
    "lift_delta_height": "(self, delta_m, tolerance_m=0.002, timeout_s=30.0, "
                         "min_height_m=0.0, max_height_m=0.9)",
    "lift_down": "(self)",
    "lift_home": "(self)",
    "lift_position_known": "(self)",
    "lift_stop": "(self)",
    "lift_to_height": "(self, target_m, tolerance_m=0.002, timeout_s=30.0, "
                      "min_height_m=0.0, max_height_m=0.9)",
    "lift_up": "(self)",
    "move_by": "(self, deltas=None)",
    "move_to": "(self, goal=None)",
    "open_left_gripper": "(self)",
    "open_right_gripper": "(self)",
    "park": "(self, gripper_target=1.0)",
    "resume_wholebody": "(self)",
    "set_base_velocity": "(self, velocity)",
    "set_bimanual_ee_target": "(self, L_ee_target, R_ee_target, L_gripper_target=None, "
                              "L_preview_time=0.1, R_gripper_target=None, R_preview_time=0.1)",
    "set_left_ee_target": "(self, ee_target, gripper_target=None, preview_time=0.1)",
    "set_left_gain": "(self, kp, kd)",
    "set_left_joint_target": "(self, joint_target, gripper_target=None, preview_time=0.1)",
    "set_lift_target": "(self, lift_target)",
    "set_right_ee_target": "(self, ee_target, gripper_target=None, preview_time=0.1)",
    "set_right_gain": "(self, kp, kd)",
    "set_right_joint_target": "(self, joint_target, gripper_target=None, preview_time=0.1)",
    "toggle_base_motion": "(self, enable=None)",
    "toggle_collision_avoidance": "(self, enable=None)",
    "toggle_fix_base": "(self, fixed=None)",
    "tuck_arms": "(self)",
}

FROZEN_WHOLEBODY = {
    "arms_manually_overridden": "(self)",
    "base_manually_overridden": "(self)",
    "emergency_stop": "(self)",
    "forward_kinematics": "(self)",
    "get_base_velocity": "(self)",
    "get_left_ee_pose": "(self)",
    "get_lift_position": "(self)",
    "get_right_ee_pose": "(self)",
    "get_state": "(self)",
    "home_left_arm": "(self)",
    "home_right_arm": "(self)",
    "init": "(self)",
    "lift_home": "(self)",
    "lift_manually_overridden": "(self)",
    "notify_manual_arm_command": "(self)",
    "notify_manual_base_command": "(self)",
    "notify_manual_lift_command": "(self)",
    "set_bimanual_ee_target": "(self, L_ee_target, R_ee_target, L_gripper_target=None, "
                              "R_gripper_target=None, L_preview_time=0.0, R_preview_time=0.0)",
    "set_left_ee_target": "(self, ee_target, gripper_target=None, preview_time=0.0)",
    "set_lift_target": "(self, lift_target)",
    "set_right_ee_target": "(self, ee_target, gripper_target=None, preview_time=0.0)",
    "start": "(self)",
    "stop": "(self)",
    "toggle_base_motion": "(self, enable=None)",
    "toggle_collision_avoidance": "(self, enable=None)",
    "toggle_fix_base": "(self, fixed=None)",
}

FROZEN_BASE = {
    "control_loop": "(self)",
    "get_lift_height": "(self)",
    "get_lift_status": "(self)",
    "lift_delta_height": "(self, delta_m, tolerance_m=0.002, timeout_s=30.0, "
                         "min_height_m=0.0, max_height_m=LIFT_MAX_HEIGHT_M)",
    "lift_down": "(self)",
    "lift_home": "(self)",
    "lift_position_known": "(self)",
    "lift_stop": "(self)",
    "lift_to_height": "(self, target_m, tolerance_m=0.002, timeout_s=30.0, "
                      "min_height_m=0.0, max_height_m=LIFT_MAX_HEIGHT_M, profiled=True)",
    "lift_up": "(self)",
    "set_target_base_velocity": "(self, target, smooth=False)",
    "start_control": "(self)",
    "stop_control": "(self)",
}

FROZEN_PICOLIFT = {
    "down": "(self)",
    "get_height": "(self)",
    "get_last_event": "(self)",
    "get_limits": "(self)",
    "get_motion": "(self)",
    "home": "(self)",
    "is_homed": "(self)",
    "is_position_known": "(self)",
    "move_mm": "(self, distance_mm, up)",
    "request_status": "(self)",
    "set_power": "(self, on)",
    "stop": "(self)",
    "up": "(self)",
}

FROZEN_ARMNODE = {
    "close_gripper": "(self)",
    "get_gripper_pose": "(self)",
    "get_joint_positions": "(self)",
    "get_joint_torques": "(self)",
    "get_joint_velocities": "(self)",
    "home": "(self, gripper_target=1.0)",
    "init": "(self)",
    "open_gripper": "(self)",
    "set_admittance_mode": "(self, kp=[25.0, 25.0, 20.0, 20.0, 15.0, 15.0, 15.0], "
                           "kd=[1.2, 1.2, 1.0, 1.0, 0.8, 0.8, 0.8])",
    "set_compliant_mode": "(self, kp=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "
                          "kd=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])",
    "set_firm_mode": "(self, kp=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0], "
                     "kd=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])",
    "set_gain": "(self, kp, kd)",
    "set_gravity_comp": "(self, enable)",
    "set_gravity_comp_scale": "(self, scale)",
    "set_joint_target": "(self, joint_target, gripper_target=None, preview_time=0.01)",
    "set_mode": "(self, control_mode, move_mode)",
    "set_q_offset": "(self, q_offset)",
    "set_spring_mode": "(self, kp=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], "
                       "kd=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])",
    "stop": "(self)",
    "sync_target": "(self)",
    "tuck_arms": "(self)",
}

# The lines the firmware emits that PicoLift's parser must keep understanding.
# Each one is checked in both directions: the sketch must still be able to
# produce it, and PicoLift must still recognise it.
FIRMWARE_LINES = {
    "Height: 123.456 mm": ("Height: ", " mm"),
    "Height: unknown (run home)": ("Height: unknown (run home)",),
    "Upper limit: ACTIVE": ("Upper limit: ", "ACTIVE"),
    "Lower limit: clear": ("Lower limit: ", "clear"),
    "Motion: IDLE": ("Motion: ", "IDLE"),
    "Home complete.": ("Home complete.",),
    "Home failed: upper limit was not reached.": ("Home failed: upper limit was not reached.",),
    "Home stopped.": ("Home stopped.",),
    "Move complete.": ("Move complete.",),
    "Motion stopped by user.": ("Motion stopped by user.",),
    "LIMIT HIT: upper limit.": ("LIMIT HIT: upper limit.",),
    "Lift controller ready.": ("Lift controller ready.",),
    # Added by the streamed-velocity work; listed here so it is held to the
    # same two-way contract as everything above it.
    "Capabilities: lift_velocity_v1": ("Capabilities: lift_velocity_v1",),
}

# Verbs PicoLift puts on the wire. The sketch must still parse every one.
WIRE_COMMANDS = ("up", "down", "stop", "home", "status", "power on", "power off", "vel")


RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Source inspection
# ─────────────────────────────────────────────────────────────────────────────

def signatures(path: Path, class_name: str) -> dict[str, str]:
    """`{method: "(self, a, b=1)"}` for every public method of a class."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out = {}
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if item.name.startswith("_"):
                    continue
                out[item.name] = _render_args(item.args)
            return out
    raise LookupError(f"class {class_name} not found in {path}")


def _render_args(a: ast.arguments) -> str:
    names = [x.arg for x in a.posonlyargs + a.args]
    defaults = [ast.unparse(d) for d in a.defaults]
    padded = [None] * (len(names) - len(defaults)) + defaults
    parts = [n if d is None else f"{n}={d}" for n, d in zip(names, padded)]
    if a.vararg:
        parts.append(f"*{a.vararg.arg}")
    for kw, d in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(kw.arg if d is None else f"{kw.arg}={ast.unparse(d)}")
    if a.kwarg:
        parts.append(f"**{a.kwarg.arg}")
    return "(" + ", ".join(parts) + ")"


def attribute_calls(path: Path, receiver: str) -> set[str]:
    """Every `<receiver>.<method>(` in a file, e.g. `self.wholebody.foo(`."""
    return set(re.findall(rf"{re.escape(receiver)}\.(\w+)\(", path.read_text()))


def frozen_check(label: str, path: Path, class_name: str, frozen: dict[str, str]) -> None:
    print(f"\n{label}")
    current = signatures(path, class_name)
    missing = sorted(set(frozen) - set(current))
    check(f"{class_name}: no frozen method removed", not missing, str(missing))

    changed = [
        f"{name}: {frozen[name]} -> {current[name]}"
        for name in sorted(frozen)
        if name in current and current[name] != frozen[name]
    ]
    check(f"{class_name}: no frozen signature changed", not changed, "; ".join(changed))

    added = sorted(set(current) - set(frozen))
    if added:
        print(f"    (new since the freeze, allowed: {', '.join(added)})")


# ─────────────────────────────────────────────────────────────────────────────
# Boundaries
# ─────────────────────────────────────────────────────────────────────────────

def test_frozen_signatures() -> None:
    frozen_check("wholebody_teleop.py -> YOR (RPC surface)",
                 _REPO / "robot/yor.py", "YOR", FROZEN_YOR)
    frozen_check("YOR -> WholeBodyController",
                 _REPO / "robot/wholebody_control.py", "WholeBodyController", FROZEN_WHOLEBODY)
    frozen_check("WholeBodyController -> Base",
                 _REPO / "robot/base_motor.py", "Base", FROZEN_BASE)
    frozen_check("Base -> PicoLift",
                 _REPO / "robot/base_motor.py", "PicoLift", FROZEN_PICOLIFT)
    frozen_check("WholeBodyController -> ArmNode",
                 _REPO / "robot/arm/arm.py", "ArmNode", FROZEN_ARMNODE)


def test_delegation() -> None:
    print("\ndelegation targets exist")
    yor = _REPO / "robot/yor.py"
    wbc = _REPO / "robot/wholebody_control.py"

    wholebody_api = set(signatures(wbc, "WholeBodyController"))
    called = attribute_calls(yor, "self.wholebody")
    missing = sorted(called - wholebody_api)
    check("every self.wholebody.* call in yor.py exists", not missing, str(missing))

    base_api = set(signatures(_REPO / "robot/base_motor.py", "Base"))
    base_called = attribute_calls(yor, "self.base") | attribute_calls(wbc, "self.base")
    missing = sorted(base_called - base_api)
    check("every self.base.* call in yor.py / wholebody_control.py exists",
          not missing, str(missing))

    arm_api = set(signatures(_REPO / "robot/arm/arm.py", "ArmNode"))
    arm_called = (attribute_calls(yor, "self.left_arm")
                  | attribute_calls(yor, "self.right_arm")
                  | attribute_calls(wbc, "self.left_arm")
                  | attribute_calls(wbc, "self.right_arm")
                  | attribute_calls(wbc, "arm"))
    missing = sorted(arm_called - arm_api)
    check("every arm call in yor.py / wholebody_control.py exists", not missing, str(missing))

    pico_api = set(signatures(_REPO / "robot/base_motor.py", "PicoLift"))
    pico_called = attribute_calls(_REPO / "robot/base_motor.py", "self._pico_lift")
    missing = sorted(pico_called - pico_api)
    check("every self._pico_lift.* call in Base exists", not missing, str(missing))


def test_lift_homes_before_startup_control() -> None:
    print("\nlift startup ordering")
    source = (_REPO / "robot/yor.py").read_text()
    init_start = source.index("    def init(self):")
    init_end = source.index("    def _sync_base_pid_gains", init_start)
    init_source = source[init_start:init_end]

    base_start_at = init_source.index("self.base.start_control()")
    home_at = init_source.index("self.base.lift_home()")
    wait_at = init_source.index(
        "while time.time() - home_start < LIFT_STARTUP_HOME_WAIT_S"
    )
    height_at = init_source.index("self.base.lift_to_height(LIFT_STARTUP_HEIGHT_M)")
    initialized_at = init_source.index("self._initialized = True")

    check("startup lift-home wait is exactly 30 seconds",
          "LIFT_STARTUP_HOME_WAIT_S = 30.0" in source)
    check("base is locked before lift homing and whole-body control",
          base_start_at < home_at < wait_at < height_at < initialized_at)
    check("startup moves the lift to the absolute 450 mm arm-safe height",
          "LIFT_STARTUP_HEIGHT_M = 0.450" in source)
    check("failed startup homing stops the lift and aborts initialization",
          "self.base.lift_stop()" in init_source
          and "raise RuntimeError(" in init_source)


def test_lift_serial_protocol() -> None:
    print("\nPicoLift <-> Arduino serial protocol")
    sketch = (_REPO / "firmware/lift_controller/lift_controller.ino").read_text()
    driver = (_REPO / "robot/base_motor.py").read_text()

    unparsed = [c for c in WIRE_COMMANDS if f'"{c}"' not in sketch]
    check("sketch still parses every command PicoLift sends", not unparsed, str(unparsed))

    # Distance moves ("up 200") are parsed by splitting on a space, so the
    # sketch must keep doing that rather than only comparing whole strings.
    check("sketch still parses '<verb> <number>' distance moves",
          'command.indexOf(\' \')' in sketch and "toFloat()" in sketch)

    unprintable = [
        line for line, fragments in FIRMWARE_LINES.items()
        if any(f'"{frag}"' not in sketch for frag in fragments)
    ]
    check("sketch can still emit every line PicoLift parses", not unprintable,
          str(unprintable))

    # And the driver's regexes must still match those lines. Compile them from
    # the driver's own class attributes so the check follows any rename.
    patterns = dict(re.findall(r"(_\w+_PATTERN) = re\.compile\((r?\"[^\"]+\")", driver))
    unmatched = []
    for line in FIRMWARE_LINES:
        matched = any(re.search(eval(p), line, re.IGNORECASE) for p in patterns.values())
        if not matched and not _matched_by_prefix(line, driver):
            unmatched.append(line)
    check("PicoLift still recognises every firmware line", not unmatched, str(unmatched))


def _matched_by_prefix(line: str, driver: str) -> bool:
    """Lines PicoLift folds in with `lowered.startswith(...)` rather than a regex."""
    prefixes = re.findall(r"startswith\(\"([^\"]+)\"\)", driver)
    return any(line.lower().startswith(p) for p in prefixes)


def test_teleop_client_contract() -> None:
    print("\nteleop client calls resolve on both servers")
    client = (_REPO / "robot/teleop/wholebody_teleop.py").read_text()
    calls = set(re.findall(r"self\.yor\.(\w+)\(", client))

    hardware = set(signatures(_REPO / "robot/yor.py", "YOR"))
    sim = set(signatures(_REPO / "robot/yor_mujoco.py", "YORMujoco"))
    check("hardware node serves every teleop call", not (calls - hardware),
          str(sorted(calls - hardware)))
    check("simulation node serves every teleop call", not (calls - sim),
          str(sorted(calls - sim)))

    joystick = (_REPO / "robot/teleop/joystick.py").read_text()
    joy_calls = set(re.findall(r"self\.yor\.(\w+)\(", joystick))
    check("joystick calls resolve on the hardware node", not (joy_calls - hardware),
          str(sorted(joy_calls - hardware)))

    # set_lift_target must stay a *position* command: the client sends metres
    # and nothing about that changes when the lift gains a velocity path.
    check("teleop still sends set_lift_target a height in metres",
          "LIFT_RANGE = (0.0, 0.900)" in client
          and "self.yor.set_lift_target(st.lift_target)" in client)


def main() -> int:
    test_frozen_signatures()
    test_delegation()
    test_lift_homes_before_startup_control()
    test_lift_serial_protocol()
    test_teleop_client_contract()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    failures = [name for name, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
