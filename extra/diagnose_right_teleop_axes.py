#!/usr/bin/env python3
"""Interactively diagnose one arm's Quest-to-EE axis mapping in YOR-v3.

This script controls only the selected arm. Every test starts from the EE pose
captured after homing, uses the target-mapping function and wholebody/single-arm
IK solver used by YOR-v3, and records the operator's observation.

Keep an emergency stop within reach and support the arm before disabling it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import mink
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from robot.arm.arm import ArmNode
from robot.arm.wholebody_ik import DEFAULT_SCENE, WholeBodyIK, WholeBodyIKConfig

try:
    from nerolib import FirmwareVersion
except ImportError:
    FirmwareVersion = None

SETTLE_VELOCITY_TOLERANCE = 0.08  # rad/s
SETTLE_STABLE_TIME = 0.3  # seconds
TARGET_POSITION_TOLERANCE = 0.03  # metres
TARGET_ROTATION_TOLERANCE = math.radians(8.0)  # rad (~8.0 deg)

# Wholebody teleop coordinate transforms
QUEST_TO_ROBOT_TRANSLATION_FRAME = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
)
QUEST_TO_EE_ROTATION_AXES = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
)
# Dedicated rotation conjugation matrix for the default "wholebody_teleop"
# mapping mode. QUEST_TO_ROBOT_TRANSLATION_FRAME is a forward 3-cycle
# permutation (ctrl X->Y, Y->Z, Z->X); reusing it to conjugate rotation
# deltas carries that same cycle into orientation (confirmed by the
# *_teleop_axes_2026*.json calibration runs in this directory: pitch
# commands rolled the EE, yaw commands pitched it, roll commands yawed
# it, on both arms). This is the inverse 3-cycle (ctrl X->Z, Y->X, Z->Y),
# which cancels it out. Y and Z rows carry a sign flip on top of that
# permutation from a live left-arm recheck: pitch (X row) read correct,
# but yaw (Y row) came out mirrored; flipping only Y would make this a
# reflection (det -1, rejected by mink.SO3), so Z (roll) flips with it
# to keep det = +1 -- roll's own sign wasn't separately confirmed live.
# See robot/teleop/wholebody_teleop.py's OculusSource.H_rot for the
# matching production fix.
QUEST_TO_EE_ROTATION_FRAME = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
)


class DiagnosticArmController:
    """Controls a single arm on hardware while solving IK via WholeBodyIK."""

    def __init__(
        self,
        side: str,
        scene_xml: Optional[str] = None,
        can_port: Optional[str] = None,
        control_hz: float = 60.0,
    ):
        self.side = side.lower()
        self.is_left = self.side == "left"
        self.can_port = can_port or f"can_{self.side}"
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz

        # Initialize WholeBodyIK solver with base and lift fixed
        self.scene_xml = str(Path(scene_xml or DEFAULT_SCENE).resolve())
        cfg = WholeBodyIKConfig(
            dt=self.dt,
            solver="pyqpmad",
            max_iters=15,
            base_posture_cost=100.0,
            lift_posture_cost=100.0,
            arm_posture_cost=1e-3,
            enable_collision_avoidance=True,
        )
        self.ik = WholeBodyIK(scene_xml=self.scene_xml, config=cfg)
        self.ik.init_from_keyframe("home")
        self.ik.fix_base = True
        self.ik.fix_lift = True

        # Home FK poses in model world frame
        T_l, T_r = self.ik.forward_kinematics()
        self._home_left_ee = T_l.copy()
        self._home_right_ee = T_r.copy()

        # Hardware arm node
        fw = getattr(FirmwareVersion, "V111", None) if FirmwareVersion else None
        self.arm_node = ArmNode(
            can_port=self.can_port,
            is_left_arm=self.is_left,
            dynamixel_gripper=False,
            native_gripper=False,
            firmware_version=fw,
        )

        self._target_lock = threading.Lock()
        self._current_target: Optional[mink.SE3] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready = False

    def init(self) -> bool:
        """Home the arm on hardware, sync IK state, and start the solver loop."""
        print(f"[Diagnostic] Initializing hardware on {self.can_port}...")
        if not self.arm_node.init():
            print(f"[Diagnostic] ArmNode init failed on {self.can_port}")
            return False

        # Read actual initial joint positions and sync IK
        q_meas = self.arm_node.get_joint_positions()
        if self.is_left:
            self.ik.set_measured_state(left_q=q_meas)
        else:
            self.ik.set_measured_state(right_q=q_meas)

        T_l, T_r = self.ik.forward_kinematics()
        current_ee = T_l if self.is_left else T_r

        with self._target_lock:
            self._current_target = mink.SE3(current_ee.wxyz_xyz.copy())

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._solver_loop,
            name=f"diag_solver_{self.side}",
            daemon=True,
        )
        self._thread.start()
        self._ready = True
        return True

    def _solver_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            with self._target_lock:
                target = self._current_target

            if target is not None:
                try:
                    q_meas = self.arm_node.get_joint_positions()
                    if self.is_left:
                        self.ik.set_measured_state(left_q=q_meas)
                        res = self.ik.solve(
                            T_left=target, T_right=self._home_right_ee
                        )
                        q_cmd = res.left_arm_q
                    else:
                        self.ik.set_measured_state(right_q=q_meas)
                        res = self.ik.solve(
                            T_left=self._home_left_ee, T_right=target
                        )
                        q_cmd = res.right_arm_q

                    self.arm_node.set_joint_target(
                        q_cmd, preview_time=float(self.dt)
                    )
                except Exception as e:
                    print(f"[Diagnostic] Solve error: {e}")

            elapsed = time.monotonic() - loop_start
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def set_ee_target(self, target: mink.SE3, gripper_target: float | None = None) -> bool:
        if not self._ready:
            return False
        with self._target_lock:
            self._current_target = mink.SE3(target.wxyz_xyz.copy())
        return True

    def get_ee_pose(self) -> mink.SE3:
        """Compute the current EE pose from measured joint angles."""
        q_meas = self.arm_node.get_joint_positions()
        if self.is_left:
            self.ik.set_measured_state(left_q=q_meas)
            return self.ik.forward_kinematics()[0]
        else:
            self.ik.set_measured_state(right_q=q_meas)
            return self.ik.forward_kinematics()[1]

    def get_joint_positions(self) -> np.ndarray:
        return self.arm_node.get_joint_positions()

    def get_joint_velocities(self) -> np.ndarray:
        return self.arm_node.get_joint_velocities()

    def is_ready(self) -> bool:
        return self._ready

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.arm_node.stop()
        self._ready = False


def make_mapper(*, mapping_mode: str = "wholebody_teleop") -> SimpleNamespace:
    return SimpleNamespace(
        H=mink.SE3.from_rotation(
            mink.SO3.from_matrix(QUEST_TO_ROBOT_TRANSLATION_FRAME)
        ),
        H_rot=mink.SE3.from_rotation(
            mink.SO3.from_matrix(QUEST_TO_EE_ROTATION_FRAME)
        ),
        rotation_axis_map=mink.SO3.from_matrix(QUEST_TO_EE_ROTATION_AXES),
        mapping_mode=mapping_mode,
    )


def compute_target(
    mapper: SimpleNamespace,
    controller_initial: mink.SE3,
    ee_initial: mink.SE3,
    controller_target: mink.SE3,
) -> mink.SE3:
    """Compute the EE target pose using teleop frame decomposition."""
    if mapper.mapping_mode == "v3_local_frame":
        controller_translation = (
            controller_target.translation() - controller_initial.translation()
        )
        controller_translation_local = controller_initial.rotation().inverse().apply(
            controller_translation
        )
        ee_translation_local = mapper.rotation_axis_map.apply(
            controller_translation_local
        )
        target_pos = ee_initial.translation() + ee_initial.rotation().apply(
            ee_translation_local
        )

        controller_rotation = (
            controller_initial.rotation().inverse() @ controller_target.rotation()
        )
        robot_rotation_delta = (
            mapper.rotation_axis_map
            @ controller_rotation
            @ mapper.rotation_axis_map.inverse()
        )
        target_rot = ee_initial.rotation() @ robot_rotation_delta
        return mink.SE3(np.concatenate([target_rot.wxyz, target_pos]))

    # Default: wholebody_teleop mapping
    X_Cdelta = controller_initial.inverse().multiply(controller_target)
    X_Rdelta = mapper.H_rot.inverse() @ X_Cdelta @ mapper.H_rot

    controller_displacement = (
        controller_target.translation() - controller_initial.translation()
    )
    robot_displacement = mapper.H.rotation().inverse().apply(
        controller_displacement
    )
    target_pos = ee_initial.translation() + robot_displacement
    target_rot = ee_initial.rotation() @ X_Rdelta.rotation()
    return mink.SE3(np.concatenate([target_rot.wxyz, target_pos]))


def pose_list(pose: mink.SE3) -> list[float]:
    return [float(value) for value in pose.wxyz_xyz]


def rotation_vector(pose: mink.SE3) -> list[float]:
    return [float(value) for value in pose.rotation().log()]


def pose_from_rotation_translation(
    rotation: mink.SO3, translation: np.ndarray
) -> mink.SE3:
    return mink.SE3(
        np.concatenate([rotation.wxyz, np.asarray(translation, dtype=float)])
    )


def controller_target_from_local_delta(
    controller_initial: mink.SE3,
    translation_local: np.ndarray,
    rotation_vector_local: np.ndarray,
) -> mink.SE3:
    """Apply a synthetic gesture in the engagement controller's local frame."""
    target_rotation = controller_initial.rotation() @ mink.SO3.exp(
        np.asarray(rotation_vector_local, dtype=float)
    )
    target_translation = (
        controller_initial.translation()
        + controller_initial.rotation().apply(
            np.asarray(translation_local, dtype=float)
        )
    )
    return pose_from_rotation_translation(target_rotation, target_translation)


def wait_for_arm_settle(
    arm: DiagnosticArmController,
    target: mink.SE3,
    timeout: float,
) -> tuple[dict, mink.SE3]:
    """Wait for low measured joint velocity, then audit the final pose."""
    started = time.monotonic()
    stable_since: float | None = None
    max_joint_velocity = math.inf
    velocity_settled = False

    while time.monotonic() - started < timeout:
        velocities = arm.get_joint_velocities()
        if velocities.shape == (7,) and np.all(np.isfinite(velocities)):
            max_joint_velocity = float(np.max(np.abs(velocities)))
        else:
            max_joint_velocity = math.inf

        now = time.monotonic()
        if max_joint_velocity <= SETTLE_VELOCITY_TOLERANCE:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= SETTLE_STABLE_TIME:
                velocity_settled = True
                break
        else:
            stable_since = None
        time.sleep(0.05)

    actual = arm.get_ee_pose()
    position_error = float(
        np.linalg.norm(actual.translation() - target.translation())
    )
    rotation_error = float(
        np.linalg.norm(
            (target.rotation().inverse() @ actual.rotation()).log()
        )
    )
    pose_within_tolerance = (
        position_error <= TARGET_POSITION_TOLERANCE
        and rotation_error <= TARGET_ROTATION_TOLERANCE
    )
    report = {
        "success": bool(velocity_settled and pose_within_tolerance),
        "velocity_settled": velocity_settled,
        "pose_within_tolerance": pose_within_tolerance,
        "elapsed_s": float(time.monotonic() - started),
        "max_joint_velocity_rad_s": (
            max_joint_velocity if math.isfinite(max_joint_velocity) else None
        ),
        "position_error_m": position_error,
        "rotation_error_rad": rotation_error,
        "rotation_error_deg": math.degrees(rotation_error),
    }
    return report, actual


def save_results(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def prompt_action(message: str) -> str:
    response = input(f"\n{message}\nPress Enter to command it, 's' to skip, or 'q' to stop: ")
    return response.strip().lower()


def main(
    default_arm: str = "right",
    default_rotations_only: bool = False,
    default_v3_mapping: bool = False,
    default_translation_mm: float = 50.0,
    default_rotation_deg: float = 10.0,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=("left", "right"),
        default=default_arm,
        help=f"Arm to test (default: {default_arm})",
    )
    parser.add_argument(
        "--translation-mm",
        type=float,
        default=default_translation_mm,
        help=(
            "Controller translation used for each position test "
            f"(default: {default_translation_mm:g})"
        ),
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=default_rotation_deg,
        help=(
            "Controller rotation used for each orientation test "
            f"(default: {default_rotation_deg:g})"
        ),
    )
    parser.add_argument(
        "--rotations-only",
        action="store_true",
        default=default_rotations_only,
        help="Skip the three translation tests",
    )
    mapping_group = parser.add_mutually_exclusive_group()
    mapping_group.add_argument(
        "--v3-local-frame",
        dest="v3_local_frame",
        action="store_true",
        help="test v3 controller-local translation through the captured EE frame",
    )
    mapping_group.add_argument(
        "--teleop-mapping",
        dest="v3_local_frame",
        action="store_false",
        help="test standard YOR-v3 wholebody teleop mapping (default)",
    )
    parser.set_defaults(v3_local_frame=default_v3_mapping)
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=8.0,
        help="Maximum seconds to wait for each outbound/return motion (default: 8)",
    )
    args = parser.parse_args()

    side = args.arm
    can_port = f"can_{side}"
    side_upper = side.upper()

    if not math.isfinite(args.translation_mm) or not 0.0 < args.translation_mm <= 100.0:
        raise SystemExit("--translation-mm must be in (0, 100]")
    if not math.isfinite(args.rotation_deg) or not 0.0 < args.rotation_deg <= 20.0:
        raise SystemExit("--rotation-deg must be in (0, 20]")
    if not math.isfinite(args.settle_timeout) or not 1.0 <= args.settle_timeout <= 30.0:
        raise SystemExit("--settle-timeout must be in [1, 30]")

    translation = args.translation_mm / 1000.0
    rotation = math.radians(args.rotation_deg)
    mapping_mode = "v3_local_frame" if args.v3_local_frame else "wholebody_teleop"
    mapper = make_mapper(mapping_mode=mapping_mode)

    if args.v3_local_frame:
        controller_initial = pose_from_rotation_translation(
            mink.SO3.exp(np.array([0.0, math.radians(37.0), 0.0])),
            np.array([0.23, 1.12, -0.41]),
        )
    else:
        controller_initial = mink.SE3.identity()

    # Positions here are after oculus_msgs.py's left- to right-handed
    # conversion: physical forward is -Z, right is +X, and up is +Y.
    test_specs = [
        (
            "translation_forward",
            f"Simulate moving the Quest controller {args.translation_mm:g} mm FORWARD",
            np.array([0.0, 0.0, -translation]),
            np.zeros(3),
        ),
        (
            "translation_right",
            f"Simulate moving the Quest controller {args.translation_mm:g} mm RIGHT",
            np.array([translation, 0.0, 0.0]),
            np.zeros(3),
        ),
        (
            "translation_up",
            f"Simulate moving the Quest controller {args.translation_mm:g} mm UP",
            np.array([0.0, translation, 0.0]),
            np.zeros(3),
        ),
        (
            "rotation_pitch_up",
            f"Simulate pitching the controller nose UP by {args.rotation_deg:g} degrees",
            np.zeros(3),
            np.array([rotation, 0.0, 0.0]),
        ),
        (
            "rotation_yaw_left",
            f"Simulate yawing the controller nose LEFT by {args.rotation_deg:g} degrees",
            np.zeros(3),
            np.array([0.0, rotation, 0.0]),
        ),
        (
            "rotation_about_forward",
            f"Simulate rolling the controller about its FORWARD axis by {args.rotation_deg:g} degrees",
            np.zeros(3),
            np.array([0.0, 0.0, -rotation]),
        ),
    ]
    tests = [
        (
            name,
            description,
            controller_target_from_local_delta(
                controller_initial,
                translation_local,
                rotation_vector_local,
            ),
            translation_local,
            rotation_vector_local,
        )
        for name, description, translation_local, rotation_vector_local in test_specs
    ]
    if args.rotations_only:
        tests = [test for test in tests if test[0].startswith("rotation_")]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_stem = (
        f"{side}_teleop_v3_calibration"
        if args.v3_local_frame
        else f"{side}_teleop_axes"
    )
    output_path = Path(__file__).resolve().parent / f"{output_stem}_{timestamp}.json"
    results = {
        "created_at": timestamp,
        "arm": side,
        "can_port": can_port,
        "mapping_mode": mapping_mode,
        "scene_xml": str(DEFAULT_SCENE),
        "scene_sha256": hashlib.sha256(DEFAULT_SCENE.read_bytes()).hexdigest(),
        "controller_initial_wxyz_xyz": pose_list(controller_initial),
        "translation_mm": args.translation_mm,
        "rotation_deg": args.rotation_deg,
        "settle_timeout_s": args.settle_timeout,
        "settle_velocity_tolerance_rad_s": SETTLE_VELOCITY_TOLERANCE,
        "settle_stable_time_s": SETTLE_STABLE_TIME,
        "target_position_tolerance_m": TARGET_POSITION_TOLERANCE,
        "target_rotation_tolerance_rad": TARGET_ROTATION_TOLERANCE,
        "quest_to_robot_translation_frame": (
            QUEST_TO_ROBOT_TRANSLATION_FRAME.tolist()
        ),
        "quest_to_ee_rotation_axes": QUEST_TO_EE_ROTATION_AXES.tolist(),
        "quest_to_ee_rotation_frame": QUEST_TO_EE_ROTATION_FRAME.tolist(),
        "tests": [],
    }

    arm: DiagnosticArmController | None = None
    try:
        title = "teleop-axis diagnostic (YOR-v3)"
        print(f"\n{side_upper} ARM ONLY {title}")
        print("Clear the workspace, support the arm if needed, and keep E-stop ready.")
        input(
            f"Press Enter to enable and HOME the {side} arm, or Ctrl-C to abort: "
        )

        arm = DiagnosticArmController(
            side=side,
            can_port=can_port,
        )
        if not arm.init():
            raise RuntimeError(f"{side}-arm initialization failed on {can_port}")

        ee_initial = arm.get_ee_pose()
        results["home_ee_wxyz_xyz"] = pose_list(ee_initial)
        results["home_joint_angles"] = arm.get_joint_positions().tolist()
        save_results(output_path, results)

        print(
            f"\n{side_upper.title()} arm homed. "
            "The captured EE pose is the baseline for every test."
        )
        print(f"Results will be saved after every observation to:\n  {output_path}")

        for (
            name,
            description,
            controller_target,
            controller_translation_local,
            controller_rotation_vector_local,
        ) in tests:
            action = prompt_action(description)
            if action == "q":
                break
            if action == "s":
                results["tests"].append({"name": name, "skipped": True})
                save_results(output_path, results)
                continue

            target = compute_target(
                mapper, controller_initial, ee_initial, controller_target
            )
            target_translation_delta = (
                target.translation() - ee_initial.translation()
            )
            target_translation_delta_local = (
                ee_initial.rotation().inverse().apply(target_translation_delta)
            )
            target_rotation_delta = (
                ee_initial.rotation().inverse() @ target.rotation()
            )

            print(
                "Commanded robot-frame translation delta (m):",
                np.round(target_translation_delta, 5),
            )
            print(
                "Commanded EE-local translation delta (m):",
                np.round(target_translation_delta_local, 5),
            )
            print(
                "Commanded EE-local rotation vector (rad):",
                np.round(target_rotation_delta.log(), 5),
            )
            command_accepted = arm.set_ee_target(target)
            if not command_accepted:
                results["tests"].append(
                    {
                        "name": name,
                        "description": description,
                        "controller_target_wxyz_xyz": pose_list(controller_target),
                        "controller_local_translation_delta_m": (
                            controller_translation_local.tolist()
                        ),
                        "controller_local_rotation_vector_rad": (
                            controller_rotation_vector_local.tolist()
                        ),
                        "command_accepted": False,
                    }
                )
                save_results(output_path, results)
                print("Arm rejected the target; stopping this diagnostic run.")
                break

            settle_report, actual_pose = wait_for_arm_settle(
                arm, target, args.settle_timeout
            )
            print(
                "Measured target error: "
                f"{settle_report['position_error_m'] * 1000.0:.1f} mm, "
                f"{settle_report['rotation_error_deg']:.2f} deg"
            )
            if not settle_report["success"]:
                print(
                    "Note: motion did not fully settle within nominal tolerance, but continuing."
                )

            observation = input(
                "Describe the ACTUAL physical direction/axis you observed: "
            ).strip()
            observed_translation_delta = (
                actual_pose.translation() - ee_initial.translation()
            )
            observed_translation_delta_local = (
                ee_initial.rotation().inverse().apply(
                    observed_translation_delta
                )
            )
            observed_rotation_delta = (
                ee_initial.rotation().inverse() @ actual_pose.rotation()
            )
            record = {
                "name": name,
                "description": description,
                "controller_target_wxyz_xyz": pose_list(controller_target),
                "controller_local_translation_delta_m": (
                    controller_translation_local.tolist()
                ),
                "controller_local_rotation_vector_rad": (
                    controller_rotation_vector_local.tolist()
                ),
                "command_accepted": command_accepted,
                "commanded_ee_wxyz_xyz": pose_list(target),
                "commanded_translation_delta_m": target_translation_delta.tolist(),
                "commanded_ee_local_translation_delta_m": (
                    target_translation_delta_local.tolist()
                ),
                "commanded_local_rotation_vector_rad": rotation_vector(
                    mink.SE3.from_rotation(target_rotation_delta)
                ),
                "observed_ee_wxyz_xyz": pose_list(actual_pose),
                "observed_translation_delta_m": (
                    observed_translation_delta.tolist()
                ),
                "observed_ee_local_translation_delta_m": (
                    observed_translation_delta_local.tolist()
                ),
                "observed_ee_local_rotation_vector_rad": (
                    observed_rotation_delta.log().tolist()
                ),
                "observed_joint_angles": arm.get_joint_positions().tolist(),
                "outbound_settle": settle_report,
                "operator_observation": observation,
            }
            results["tests"].append(record)
            save_results(output_path, results)

            input("Press Enter to command the EE back to the captured home pose: ")
            baseline_accepted = arm.set_ee_target(ee_initial)
            record["baseline_command_accepted"] = baseline_accepted
            if baseline_accepted:
                baseline_report, baseline_pose = wait_for_arm_settle(
                    arm, ee_initial, args.settle_timeout
                )
                record["baseline_settle"] = baseline_report
                record["baseline_observed_ee_wxyz_xyz"] = pose_list(
                    baseline_pose
                )
            else:
                baseline_report = {"success": False}
            save_results(output_path, results)

            if not baseline_accepted or not baseline_report["success"]:
                cont = input(
                    "Baseline not fully reacquired. Press Enter to proceed to next test or 'q' to stop: "
                ).strip().lower()
                if cont == "q":
                    break
            else:
                input("Arm is back at baseline. Press Enter to continue: ")

        print(f"\nDiagnostic observations saved to:\n  {output_path}")
        return 0
    except KeyboardInterrupt:
        print("\nDiagnostic interrupted.")
        return 130
    finally:
        if arm is not None:
            if arm.is_ready():
                try:
                    input(
                        f"Support the {side_upper} arm, then press Enter to disable it: "
                    )
                except (KeyboardInterrupt, EOFError):
                    print(f"\nDisabling the {side} arm immediately.")
            arm.stop()


if __name__ == "__main__":
    raise SystemExit(main())
