"""
test_api_parity.py — the two servers must keep exposing the same API.

robot/teleop/wholebody_teleop.py drives either robot/yor.py (hardware) or
robot/yor_mujoco.py (simulation) by changing only a port, which only holds
while both classes expose every method the client calls. This check reads the
source with `ast`, so it works on a laptop where nerolib and sparkcan_py — and
therefore robot/yor.py — cannot be imported at all.

    python tests/test_api_parity.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# What the teleop client calls on whichever server it is pointed at.
CLIENT_CALLS = {
    "get_state",
    "set_left_ee_target",
    "set_right_ee_target",
    "set_bimanual_ee_target",
    "set_lift_target",
    "home_left_arm",
    "home_right_arm",
    "lift_home",
    "toggle_fix_base",
    "toggle_collision_avoidance",
}

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def public_methods(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not item.name.startswith("_")
            }
    raise LookupError(f"class {class_name} not found in {path}")


def client_calls_in_source() -> set[str]:
    """Every `self.yor.<method>(` the teleop client actually makes."""
    source = (_REPO / "robot/teleop/wholebody_teleop.py").read_text()
    return set(re.findall(r"self\.yor\.(\w+)\(", source))


def main() -> int:
    hardware = public_methods(_REPO / "robot/yor.py", "YOR")
    sim = public_methods(_REPO / "robot/yor_mujoco.py", "YORMujoco")

    print("teleop client contract")
    for name in sorted(CLIENT_CALLS):
        check(f"{name} on both servers",
              name in hardware and name in sim,
              ("missing on hardware" if name not in hardware else "") +
              ("missing on sim" if name not in sim else ""))

    print("\nthe declared contract matches what the client calls")
    actual = client_calls_in_source() - {"init"}
    check("no uncovered client calls", actual <= CLIENT_CALLS,
          str(sorted(actual - CLIENT_CALLS)))

    print("\nhardware-only extensions (informational)")
    print("  " + ", ".join(sorted(hardware - sim)) or "  none")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    failures = [name for name, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
