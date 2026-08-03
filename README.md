# YOR-v3

Whole-body control for the YORv3 mobile manipulator: a swerve base, a Z-axis
lift, two 7-DOF AgileX Nero arms and two Wuji five-finger hands, solved as one
18-DOF system rather than as independent subsystems.

```
  Base  (3 DOF)   base_x, base_y, base_yaw     planar joints, not a freejoint
  Lift  (1 DOF)   Slider 7                     0 → 0.9176 m
  Left  (7 DOF)   left_arm_joint1..7
  Right (7 DOF)   right_arm_joint1..7
  ──────────────────────────────────────────────────────────────────────
  18 IK DOF                                    (full model nq = 66)
```

Ask for an end-effector pose and the solver decides how to get there: bend the
arm if it can, stretch the lift if it must, roll the chassis only as a last
resort. Self-collision and ground avoidance are hard QP constraints, so the
solver cannot produce a motion that drives an arm into the lift column, the
chassis, the other arm, or the floor.

## Layout

```
description/            MuJoCo model — the single source of truth for kinematics
  scene_wholebody.xml     scene: floor plane + draggable mocap IK targets
  robot_wholebody.xml     the robot itself (base + lift + arms + hands)
  meshes/ wuji_mjcf/      77 STLs, all referenced
robot/
  arm/wholebody_ik.py     the solver (mink QP differential IK, pyqpmad)
  wholebody_control.py    hardware loop: measure → solve → dispatch
  yor.py                  hardware node, commlink RPC on port 5557
  yor_mujoco.py           simulation node, commlink RPC on port 8081
  arm/arm.py              joint-space driver for one Nero arm (nerolib)
  base.py base_motor.py   swerve base, PicoLift, path following
  teleop/                 whole-body client (keyboard/gamepad/oculus) + joystick
  nav/ slam_node_.py      mapping, path planning, ZED pose publishing
tools/wholebody_ik_demo.py   interactive solver demo, drag targets in the viewer
tests/                       headless checks, no hardware required
docs/RUNNING.md              how to run everything, incl. the bring-up checklist
```

`robot/yor.py` and `robot/yor_mujoco.py` expose the same RPC surface, so
`robot/teleop/wholebody_teleop.py` drives either one — `--target sim` or
`--target hw`.

## Quickstart

```bash
# simulation (macOS needs mjpython for the viewer)
conda run -n dev mjpython robot/yor_mujoco.py
conda run -n dev python robot/teleop/wholebody_teleop.py --input keyboard

# interactive solver demo, no RPC
conda run -n dev mjpython tools/wholebody_ik_demo.py

# on the robot
python robot/yor.py
python robot/teleop/wholebody_teleop.py --target hw --host <robot-ip> --input oculus
```

Read [docs/RUNNING.md](docs/RUNNING.md) before the first hardware run — the
bring-up checklist there exists because whole-body control can drive the wheels
without anyone asking it to.

## Tests

No hardware, no viewer, no CAN bus:

```bash
python tests/test_wholebody_control.py   # hardware loop against fake arms/base
python tests/test_sim_node.py            # sim node end-to-end over real RPC
python tests/test_api_parity.py          # the two nodes expose the same API
```

## Dependencies

`pip install -r requirements.txt` covers the solver and teleop stack. Three
things are not on PyPI and must be present separately:

| Package | Where | Needed for |
|---|---|---|
| `commlink` | internal | all RPC (both nodes, every client) |
| `nerolib` | submodule, built C++/pybind11 | arm control over CAN |
| `sparkcan_py` | submodule | swerve motor control over CAN |

The last two are hardware-only: simulation, the demo and every test run without
them.
