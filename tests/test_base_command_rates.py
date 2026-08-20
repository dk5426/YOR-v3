"""
test_base_command_rates.py — the base command chain must not decimate.

A base velocity crosses three loops on its way from the solver to the wheels,
each running at its own rate:

    WholeBodyController._control_loop   writes base_controller.target_velocity
    BaseController._run (BASE_VEL)      forwards it to Base
    Base.control_loop                   S-curve profiles it onto the swerves

Nothing in the middle queues: BASE_VEL reads whatever `target_velocity` holds
right now. So if the relay is slower than the solver, the extra solutions are
overwritten before the wheels ever see them, silently and with no error — the
robot simply tracks worse than the loop rate suggests. That is exactly what
happened when the relay sat at 20 Hz under a faster solver: most base velocities
were discarded.

These checks freeze the relationship rather than the numbers. Change all three
rates together and they still pass; change one and they fail.

Read with `ast` and plain text, so this runs where nerolib and sparkcan_py —
and therefore robot/yor.py — cannot be imported.

    python tests/test_base_command_rates.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Source inspection
# ─────────────────────────────────────────────────────────────────────────────

def module_constant(path: Path, name: str) -> float | None:
    """Value of a module-level `NAME = <number>` assignment."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return float(ast.literal_eval(node.value))
                    except (ValueError, TypeError):
                        return None
    return None


def dataclass_field(path: Path, class_name: str, field: str) -> float | None:
    """Default of an annotated dataclass field, e.g. `control_hz: float = 108.0`."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (isinstance(item, ast.AnnAssign)
                        and isinstance(item.target, ast.Name)
                        and item.target.id == field
                        and item.value is not None):
                    try:
                        return float(ast.literal_eval(item.value))
                    except (ValueError, TypeError):
                        return None
    return None


def call_kwarg(path: Path, func_name: str, kwarg: str) -> float | None:
    """Value of a keyword argument at a `func_name(...)` call site."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == func_name):
            for kw in node.keywords:
                if kw.arg == kwarg:
                    try:
                        return float(ast.literal_eval(kw.value))
                    except (ValueError, TypeError):
                        return None
    return None


def param_default(path: Path, class_name: str, method: str, param: str) -> float | None:
    """Default of a named parameter in a method signature."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    args = item.args
                    named = args.args[1:] + args.kwonlyargs
                    defaults = ([None] * (len(args.args) - 1 - len(args.defaults))
                                + list(args.defaults) + list(args.kw_defaults))
                    for a, d in zip(named, defaults):
                        if a.arg == param and d is not None:
                            try:
                                return float(ast.literal_eval(d))
                            except (ValueError, TypeError):
                                return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def test_relay_matches_solver() -> None:
    print("\nBASE_VEL relay keeps up with whole-body control")

    solver_hz = dataclass_field(
        _REPO / "robot/wholebody_control.py", "WholeBodyHardwareConfig", "control_hz")
    relay_hz = call_kwarg(_REPO / "robot/yor.py", "BaseController", "relay_hz")

    check("WholeBodyHardwareConfig.control_hz is readable", solver_hz is not None,
          str(solver_hz))
    check("yor.py passes relay_hz to BaseController", relay_hz is not None,
          str(relay_hz))

    if solver_hz is not None and relay_hz is not None:
        check("relay keeps up with the solver that feeds it",
              relay_hz >= solver_hz,
              f"relay={relay_hz:.0f} Hz, solver={solver_hz:.0f} Hz")


def test_relay_uses_its_own_limiter() -> None:
    print("\nBASE_VEL sleeps on the relay rate, not the navigation rate")

    base_src = (_REPO / "robot/base.py").read_text()

    # The BASE_VEL branch runs from the `if self.mode == "BASE_VEL":` guard to
    # its `continue`. Whatever it sleeps on is what paces the relay.
    branch = re.search(
        r'if self\.mode == "BASE_VEL":(.*?)continue', base_src, re.S)
    check("BASE_VEL branch found in BaseController._run", branch is not None)

    if branch:
        body = branch.group(1)
        check("BASE_VEL sleeps on relay_rate", "self.relay_rate.sleep()" in body)
        check("BASE_VEL does not sleep on the navigation rate",
              "self.rate.sleep()" not in body)

    check("BaseController builds a separate relay limiter",
          "self.relay_rate = RateLimiter(" in base_src)


def test_navigation_tracks_slam() -> None:
    print("\nnavigation stays paced by the SLAM pose it closes on")

    # The nav PIDs differentiate the Odin pose. Running them faster than it
    # arrives reads a zero derivative on the repeat cycles and a spike on the
    # update, and shrinks the window vel_alpha was tuned over.
    nav_hz = call_kwarg(_REPO / "robot/yor.py", "BaseController", "control_hz")
    slam_hz = dataclass_field(
        _REPO / "robot/wholebody_control.py", "WholeBodyHardwareConfig", "slam_pose_hz")

    check("yor.py sets control_hz for navigation", nav_hz is not None, str(nav_hz))
    check("slam_pose_hz is readable", slam_hz is not None, str(slam_hz))

    if nav_hz is not None and slam_hz is not None:
        check("navigation does not outrun the SLAM pose",
              nav_hz <= slam_hz,
              f"nav={nav_hz:.0f} Hz, slam={slam_hz:.0f} Hz")


def test_motor_loop_oversamples_relay() -> None:
    print("\nswerve loop oversamples the relay so the S-curve has room")

    control_freq = module_constant(_REPO / "robot/base_motor.py", "CONTROL_FREQ")
    relay_hz = call_kwarg(_REPO / "robot/yor.py", "BaseController", "relay_hz")

    check("CONTROL_FREQ is readable", control_freq is not None, str(control_freq))

    if control_freq is not None and relay_hz is not None:
        ratio = control_freq / relay_hz
        check("swerve loop gives each relay command three interpolation ticks",
              ratio == 3.0,
              f"{control_freq:.0f}/{relay_hz:.0f} = {ratio:.1f}x")

    # The watchdog is derived from POLICY_CONTROL_FREQ, not CONTROL_FREQ, so
    # retuning the control loop must not quietly shorten the stale-command
    # timeout out from under the relay.
    policy_freq = module_constant(_REPO / "robot/base_motor.py", "POLICY_CONTROL_FREQ")
    if policy_freq is not None and relay_hz is not None:
        timeout_s = 2.5 / policy_freq
        check("stale-command watchdog outlasts several relay periods",
              timeout_s > 5.0 / relay_hz,
              f"watchdog={timeout_s*1e3:.0f} ms, relay period={1e3/relay_hz:.0f} ms")


def test_relay_default_is_safe() -> None:
    print("\nBaseController's own default does not reintroduce decimation")

    relay_default = param_default(
        _REPO / "robot/base.py", "BaseController", "__init__", "relay_hz")
    solver_hz = dataclass_field(
        _REPO / "robot/wholebody_control.py", "WholeBodyHardwareConfig", "control_hz")

    check("relay_hz has a default", relay_default is not None, str(relay_default))
    if relay_default is not None and solver_hz is not None:
        check("default relay_hz keeps up with the solver",
              relay_default >= solver_hz,
              f"default={relay_default:.0f} Hz, solver={solver_hz:.0f} Hz")


def main() -> int:
    test_relay_matches_solver()
    test_relay_uses_its_own_limiter()
    test_navigation_tracks_slam()
    test_motor_loop_oversamples_relay()
    test_relay_default_is_safe()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    failures = [name for name, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
