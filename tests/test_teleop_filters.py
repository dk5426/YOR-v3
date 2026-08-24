"""
test_teleop_filters.py — headless checks for the Quest input conditioning.

Replays synthetic controller streams (jitter, ramps, tracking dropouts)
through robot/teleop/filters.py and through OculusSource's receive-side hook,
so the smoothing can be tuned without a headset in the room.

    python tests/test_teleop_filters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from mink.lie import SE3, SO3

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robot.teleop.filters import OneEuroFilter, PoseFilter  # noqa: E402

RATE = 72.0  # what the Quest publishes at
DT = 1.0 / RATE


def pose(xyz, rot: SO3 | None = None) -> SE3:
    return SE3.from_rotation_and_translation(
        rot if rot is not None else SO3.identity(), np.asarray(xyz, dtype=float))


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def test_one_euro_scalar():
    print("\n1€ filter")
    rng = np.random.default_rng(0)
    filt = OneEuroFilter(min_cutoff=3.0, beta=8.0)
    raw = 1.0 + rng.normal(0.0, 0.003, 400)
    out = np.array([filt(np.array([v]), DT)[0] for v in raw])
    settled = out[100:]
    check("jitter attenuated at rest",
          settled.std() < raw.std() / 3,
          f"in {raw.std()*1000:.2f} mm → out {settled.std()*1000:.2f} mm")
    check("no steady-state bias", abs(settled.mean() - 1.0) < 1e-3,
          f"{(settled.mean()-1.0)*1000:+.3f} mm")

    fast, slow = OneEuroFilter(beta=8.0), OneEuroFilter(beta=0.0)
    truth = 0.0
    for _ in range(200):  # 0.5 m/s ramp
        truth += 0.5 * DT
        f_out = fast(np.array([truth]), DT)[0]
        s_out = slow(np.array([truth]), DT)[0]
    check("beta buys responsiveness", (truth - f_out) < (truth - s_out) / 2,
          f"lag {(truth-f_out)*1000:.1f} mm vs {(truth-s_out)*1000:.1f} mm")


def test_pose_jitter():
    print("\nPoseFilter — jitter")
    rng = np.random.default_rng(1)
    filt = PoseFilter()
    truth = np.array([0.3, 0.1, 1.2])
    raw, out = [], []
    t = 0.0
    for _ in range(400):
        t += DT
        sample = truth + rng.normal(0.0, 0.003, 3)
        raw.append(sample)
        out.append(filt(pose(sample), t).translation())
    raw, out = np.array(raw), np.array(out[100:])
    check("held pose stops shaking",
          out.std(axis=0).max() < raw.std(axis=0).max() / 2,
          f"in {raw.std(axis=0).max()*1000:.2f} mm → "
          f"out {out.std(axis=0).max()*1000:.2f} mm")
    check("stays on the true pose",
          np.linalg.norm(out.mean(axis=0) - truth) < 2e-3,
          f"{np.linalg.norm(out.mean(axis=0)-truth)*1000:.2f} mm")
    check("nothing dropped in clean data", filt.rejected == 0, str(filt.rejected))


def test_pose_tracking_lag():
    print("\nPoseFilter — following a real motion")
    filt = PoseFilter()
    t, p = 0.0, np.zeros(3)
    for _ in range(int(2 * RATE)):  # 0.5 m/s reach, 2 s
        t += DT
        p = p + np.array([0.5, 0.0, 0.0]) * DT
        out = filt(pose(p), t).translation()
    lag = np.linalg.norm(p - out)
    check("keeps up with a 0.5 m/s reach", lag < 0.03, f"{lag*1000:.1f} mm behind")

    filt = PoseFilter()
    t = 0.0
    R = SO3.identity()
    for _ in range(int(2 * RATE)):  # 1 rad/s wrist rotation
        t += DT
        R = R @ SO3.from_z_radians(1.0 * DT)
        out_R = filt(pose([0, 0, 0], R), t).rotation()
    err = np.linalg.norm((out_R.inverse() @ R).log())
    check("keeps up with a 1 rad/s twist", err < 0.15, f"{np.degrees(err):.1f}° behind")
    check("output stays a unit quaternion",
          abs(np.linalg.norm(out_R.wxyz) - 1.0) < 1e-9)


def test_glitch_rejection():
    print("\nPoseFilter — tracking glitches")
    filt = PoseFilter()
    t = 0.0
    for _ in range(50):  # settle on a held pose
        t += DT
        out = filt(pose([0.3, 0.0, 1.0]), t)
    before = out.translation().copy()

    t += DT
    out = filt(pose([1.3, 0.5, 0.2]), t)  # one-sample teleport
    check("single teleport ignored",
          np.allclose(out.translation(), before) and filt.rejected == 1,
          f"moved {np.linalg.norm(out.translation()-before)*1000:.2f} mm")

    for _ in range(20):  # ... but a real move to that spot is honoured
        t += DT
        out = filt(pose([1.3, 0.5, 0.2]), t)
    check("sustained move re-locks", filt.relocks == 1 and
          np.linalg.norm(out.translation() - np.array([1.3, 0.5, 0.2])) < 1e-9,
          f"relocks={filt.relocks}")

    filt = PoseFilter()
    t = 0.0
    for _ in range(20):
        t += DT
        filt(pose([0.0, 0.0, 1.0]), t)
    out = filt(pose([0.9, 0.0, 1.0]), t + 3.0)  # stream stalled, then resumed
    check("stall re-locks instead of rejecting",
          np.allclose(out.translation(), [0.9, 0.0, 1.0]) and filt.rejected == 0)

    repeat = filt(pose([0.9, 0.0, 1.5]), t + 3.0)  # same timestamp
    check("duplicate timestamp holds output",
          np.allclose(repeat.translation(), [0.9, 0.0, 1.0]))


def test_oculus_source_hook():
    print("\nOculusSource wiring")
    from robot.teleop.oculus_msgs import ControllerState
    from robot.teleop.wholebody_teleop import LOOP_RATE, OculusSource

    def sample(t, left_xyz, right_xyz):
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # x,y,z,w
        zero = np.zeros(2)
        return ControllerState(
            t, False, False, False, False, 0.0, 0.0, zero,
            np.asarray(left_xyz, dtype=float), quat,
            False, False, False, False, 0.0, 0.0, zero,
            np.asarray(right_xyz, dtype=float), quat)

    rng = np.random.default_rng(2)
    src = OculusSource(host="127.0.0.1")
    check("teleop loop consumes Quest samples slower than they arrive "
          "(filtering on the receive thread matters)",
          LOOP_RATE < RATE, f"{RATE:.0f}/{LOOP_RATE:.0f}")
    check("pose-filter minimum cutoff defaults to 3 Hz",
          all(f.min_cutoff == 3.0 for f in src._filters.values()))
    raw, out = [], []
    t = 0.0
    for _ in range(300):
        t += DT
        noisy = np.array([0.2, 0.1, 1.0]) + rng.normal(0.0, 0.003, 3)
        raw.append(noisy)
        out.append(src._filtered(sample(t, noisy, [0.0, 0.0, 0.0]), t)["left"].translation())
    raw, out = np.array(raw), np.array(out[100:])
    check("receive hook smooths the left controller",
          out.std(axis=0).max() < raw.std(axis=0).max() / 2,
          f"out {out.std(axis=0).max()*1000:.2f} mm")
    check("both controllers filtered", set(src._filters) == {"left", "right"})

    off = OculusSource(host="127.0.0.1", pose_filter=False)
    cs = sample(0.1, [0.2, 0.1, 1.0], [0.0, 0.0, 0.0])
    check("--no-pose-filter passes poses through",
          np.allclose(off._filtered(cs, 0.1)["left"].translation(),
                      cs.left_SE3.translation()))


def test_quest_app_versions():
    print("\nQuest app v0.1 / v0.2 packet gating")
    from robot.teleop.oculus_msgs import parse_controller_state
    from robot.teleop.wholebody_teleop import OculusSource

    def section(label, extra):
        return f"{label}:;{extra};"

    left = section("Left Controller", "  Left X: False;  Left Y: False;  Left Menu: False;"
                    "  Left Thumbstick: False;  Left Index Trigger: 0;  Left Hand Trigger: 0;"
                    "  Left Thumbstick Axes: 0,0;  Left Local Position: 0.1,0.2,0.3;"
                    "  Left Local Rotation: 0,0,0,1")
    right = section("Right Controller", "  Right A: False;  Right B: False;  Right Menu: False;"
                     "  Right Thumbstick: False;  Right Index Trigger: 0;  Right Hand Trigger: 0;"
                     "  Right Thumbstick Axes: 0,0;  Right Local Position: 0.4,0.5,0.6;"
                     "  Right Local Rotation: 0,0,0,1")
    head = section("Head", "  Head Position: 0,1.6,0;  Head Rotation: 0,0,0,1")

    v01_payload = left + "|" + right
    v02_payload = head + "|" + left + "|" + right

    cs = parse_controller_state(v01_payload, legacy=True)
    check("legacy=True parses a v0.1 (Left|Right) payload",
          np.allclose(cs.left_local_position, [0.1, 0.2, 0.3])
          and np.allclose(cs.right_local_position, [0.4, 0.5, 0.6]))

    cs = parse_controller_state(v02_payload)
    check("legacy=False (default) parses a v0.2 (Head|Left|Right) payload",
          np.allclose(cs.left_local_position, [0.1, 0.2, 0.3])
          and np.allclose(cs.right_local_position, [0.4, 0.5, 0.6]))

    cs = parse_controller_state(v01_payload)
    check("legacy=False also parses a v0.1 payload (label match is order/count agnostic)",
          np.allclose(cs.left_local_position, [0.1, 0.2, 0.3]))

    raised = False
    try:
        parse_controller_state(v02_payload, legacy=True)
    except ValueError:
        raised = True
    check("legacy=True on a v0.2 payload fails loudly instead of misparsing", raised)

    src = OculusSource(host="127.0.0.1", legacy_oculus_app=True)
    check("OculusSource(legacy_oculus_app=True) stores the gate",
          src._legacy_oculus_app is True)
    src = OculusSource(host="127.0.0.1")
    check("OculusSource defaults to the v0.2 (non-legacy) parser",
          src._legacy_oculus_app is False)


def main() -> int:
    for test in (
        test_one_euro_scalar,
        test_pose_jitter,
        test_pose_tracking_lag,
        test_glitch_rejection,
        test_oculus_source_hook,
        test_quest_app_versions,
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
