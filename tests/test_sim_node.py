"""
test_sim_node.py — headless end-to-end check of the simulation node.

Starts robot/yor_mujoco.py's control loop with the MuJoCo viewer stubbed out,
puts a real commlink RPC server in front of it, and drives it with an RPC
client — the same path robot/teleop/wholebody_teleop.py takes. This is what
catches SE3-over-RPC serialisation problems and API drift between the sim and
hardware nodes.

    python tests/test_sim_node.py
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import mujoco  # noqa: E402
import mink  # noqa: E402


# ── Stub the passive viewer so this runs without a window (or mjpython) ──────
class _StubViewer:
    def __init__(self, *args, **kwargs):
        self.cam = types.SimpleNamespace()
        self.opt = types.SimpleNamespace(frame=None)

    def is_running(self):
        return True

    def sync(self):
        pass

    def close(self):
        pass


# Registered in sys.modules, not just as an attribute: yor_mujoco does
# `import mujoco.viewer`, which would otherwise re-bind the real submodule
# (and refuse to run outside mjpython on macOS).
_viewer_stub = types.ModuleType("mujoco.viewer")
_viewer_stub.launch_passive = lambda **kwargs: _StubViewer()
sys.modules["mujoco.viewer"] = _viewer_stub
mujoco.viewer = _viewer_stub
mujoco.mjv_defaultFreeCamera = lambda model, cam: None

from commlink import RPCClient, RPCServer  # noqa: E402
from robot.yor_mujoco import YORMujoco  # noqa: E402

PORT = 8099  # not the production port, so this never collides with a live sim

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def main() -> int:
    print("starting sim node…")
    node = YORMujoco()
    node.start_control()
    server = RPCServer(node, PORT, threaded=True)
    server.start()
    time.sleep(0.5)

    client = RPCClient("localhost", PORT)
    client.init()

    try:
        print("\nRPC surface")
        state = client.get_state()
        required = {
            "left_ee_wxyz_xyz", "right_ee_wxyz_xyz", "lift", "base_xytheta",
            "base_velocity", "fix_base", "collision_avoidance",
        }
        check("get_state carries the teleop keys", not (required - set(state)),
              str(required - set(state)))

        print("\nSE3 over the wire")
        T_l = mink.SE3(np.array(state["left_ee_wxyz_xyz"]))
        goal = T_l.translation() + np.array([0.0, -0.08, 0.05])
        client.set_left_ee_target(
            ee_target=mink.SE3.from_rotation_and_translation(T_l.rotation(), goal))
        time.sleep(1.5)
        reached = mink.SE3(np.array(client.get_state()["left_ee_wxyz_xyz"])).translation()
        err = float(np.linalg.norm(reached - goal))
        check("left EE tracks an SE3 sent over RPC", err < 0.01, f"{err*100:.2f} cm")

        print("\nbimanual + lift")
        state = client.get_state()
        T_l = mink.SE3(np.array(state["left_ee_wxyz_xyz"]))
        T_r = mink.SE3(np.array(state["right_ee_wxyz_xyz"]))
        client.set_bimanual_ee_target(L_ee_target=T_l, R_ee_target=T_r)
        check("set_bimanual_ee_target accepted", True)
        client.set_lift_target(0.35)
        time.sleep(0.5)
        check("set_lift_target accepted", True)

        print("\ntoggles and homes")
        fix = client.toggle_fix_base()
        check("toggle_fix_base returns a bool", isinstance(fix, bool), str(fix))
        client.toggle_fix_base(fixed=False)
        col = client.toggle_collision_avoidance()
        check("toggle_collision_avoidance returns a bool", isinstance(col, bool), str(col))
        client.toggle_collision_avoidance(enable=True)
        client.home_left_arm()
        client.home_right_arm()
        client.lift_home()
        time.sleep(0.5)
        check("homes ran without error", True)

        print("\nqueries")
        check("get_lift_position is a float",
              isinstance(client.get_lift_position(), float))
        check("get_base_velocity has 3 elements",
              len(np.asarray(client.get_base_velocity()).reshape(-1)) == 3)
        check("joint positions are 7-DOF",
              len(client.get_left_joint_positions()) == 7
              and len(client.get_right_joint_positions()) == 7)
    finally:
        server.stop()
        node.stop_control()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    failures = [name for name, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
