"""
test_aria_mapping.py — contract tests for the Aria-hands → YORv3 mapping.

Ported from aria2robot's own suite. The things that can break silently: the
MJCF's hand joint names drifting away from wuji-description's ordering (fingers
would move, just the wrong ones), the clutch delta math picking up a sign or
frame flip, and either of the two wrist-rendering fixes being dropped.

No headset, no publisher, no browser.

    python tests/test_aria_mapping.py
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robot.teleop.aria.clutch import (
    MANO_WRIST_AXES,
    YOR_WRIST_AXES,
    Clutch,
    convention_matrix,
)
from robot.teleop.aria.stream import (
    AriaHandStream,
    HomeSeqWatcher,
    canonical_joint_names,
)

SIDES = ("left", "right")
AXES = ("middle", "palm", "thumb")
SCENE = _REPO / "description" / "scene_wholebody.xml"

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def close(a, b, atol=1e-9) -> bool:
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol))


def _pose(rotation: np.ndarray, position: Sequence[float]) -> np.ndarray:
    """4x4 wrist pose from a rotation matrix and a translation."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


_MODEL = None


def model():
    """The scene, loaded once and posed at the home keyframe."""
    global _MODEL
    if _MODEL is None:
        m = mujoco.MjModel.from_xml_path(str(SCENE))
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        mujoco.mj_forward(m, d)
        _MODEL = (m, d)
    return _MODEL


# ─────────────────────────────────────────────────────────────────────────────
# The two axis tables and the rotation they compose to
# ─────────────────────────────────────────────────────────────────────────────

def test_wrist_axis_tables():
    print("\nWrist axis tables")
    # Pin both tables to literals: every other test reads them, so none can
    # guard them. MANO's is the WUJI hand root — middle +Z, palm normal +X,
    # thumb -Y left / +Y right. YOR's is the `*_arm_ee` site frame, measured off
    # the MJCF at the home keyframe — middle +Y, palm +X, thumb +Z left / -Z right.
    expected = {
        "mano": {
            "left": {"middle": [0, 0, 1], "palm": [1, 0, 0], "thumb": [0, -1, 0]},
            "right": {"middle": [0, 0, 1], "palm": [1, 0, 0], "thumb": [0, 1, 0]},
        },
        "yor": {
            "left": {"middle": [0, 1, 0], "palm": [1, 0, 0], "thumb": [0, 0, 1]},
            "right": {"middle": [0, 1, 0], "palm": [1, 0, 0], "thumb": [0, 0, -1]},
        },
    }
    ok = True
    for table, key in ((MANO_WRIST_AXES, "mano"), (YOR_WRIST_AXES, "yor")):
        for side in SIDES:
            ok &= set(table[side]) == {"middle", "palm", "thumb"}
            for axis, want in expected[key][side].items():
                ok &= np.array_equal(table[side][axis], want)
    check("both axis tables match the measured literals", ok)

    for side in SIDES:
        C = convention_matrix(side)
        # A mirrored table would give det -1 and silently flip one axis of the
        # delta. Both conventions carry the L/R mirror on the *thumb* axis, so
        # the product must come out proper on both sides.
        check(f"convention_matrix({side}) is a proper rotation",
              close(C @ C.T, np.eye(3), 1e-12) and abs(np.linalg.det(C) - 1.0) < 1e-12,
              f"det {np.linalg.det(C):+.6f}")
        for axis in AXES:
            check(f"C maps {side} {axis} axis MANO→YOR",
                  close(C @ MANO_WRIST_AXES[side][axis],
                        YOR_WRIST_AXES[side][axis], 1e-12))


def test_convention_matrix_matches_the_mjcf_hand_mount():
    print("\nConvention vs the MJCF hand mount")
    # Both ends wear the same WUJI hand, so the rotation from the operator's
    # hand frame to the robot's must be exactly the model's own
    # `arm_ee -> hand` mount rotation. If it is not, one of the two axis tables
    # is describing a hand that does not exist. Tolerance is the MJCF's
    # `euler="0 0 1.57"` rounding.
    _, d = model()
    for side in SIDES:
        R_ee = d.site(f"{side}_arm_ee").xmat.reshape(3, 3)
        R_hand = d.body(f"{side}_wuji_hand_orient").xmat.reshape(3, 3)
        err = np.abs(convention_matrix(side) - R_ee.T @ R_hand).max()
        check(f"{side} convention == MJCF arm_ee→wuji_hand_orient", err < 1e-3,
              f"max |Δ| {err:.1e}")


# ─────────────────────────────────────────────────────────────────────────────
# Hand joints
# ─────────────────────────────────────────────────────────────────────────────

def test_hand_joints():
    print("\nHand joints")
    m, _ = model()
    for side in SIDES:
        names = canonical_joint_names(side)
        adrs = [int(m.joint(n).qposadr[0]) for n in names]
        # Contiguous and ascending: the (20,) vector writes as one slice, in order
        check(f"{side} hand joints match wuji-description order",
              len(names) == 20 and adrs == list(range(adrs[0], adrs[0] + 20)),
              f"qposadr {adrs[0]}..{adrs[-1]}")

    for side in SIDES:
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        mujoco.mj_forward(m, d)
        joints = [m.joint(n) for n in canonical_joint_names(side)]
        adrs = np.array([int(j.qposadr[0]) for j in joints])
        lo = np.array([float(j.range[0]) for j in joints])
        hi = np.array([float(j.range[1]) for j in joints])
        tip = f"{side}_finger2_link4"
        before = d.body(tip).xpos.copy()
        # A curl well outside the joint ranges, clipped exactly as sim_viz does
        d.qpos[adrs] = np.clip(np.full(20, 10.0), lo, hi)
        mujoco.mj_forward(m, d)
        moved = float(np.linalg.norm(d.body(tip).xpos - before))
        check(f"{side} qpos write reaches the fingertips",
              close(d.qpos[adrs], hi) and moved > 0.01, f"tip moved {moved*100:.1f} cm")


# ─────────────────────────────────────────────────────────────────────────────
# Clutch delta math
# ─────────────────────────────────────────────────────────────────────────────

def test_clutch_engage_release():
    print("\nClutch — engage / release")
    clutch = Clutch("left")
    check("inert until engaged",
          not clutch.engaged and clutch.target(np.eye(4)) is None)

    for side in SIDES:
        ee0 = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_x_radians(0.6), np.array([0.3, -0.25, 0.4]))
        engage = _pose(mink.SO3.from_z_radians(1.3).as_matrix(), [0.2, -0.3, 0.9])
        c = Clutch(side)
        c.engage(engage, ee0)
        t = c.target(engage)
        # Engaging pins the two frames together: zero delta must command zero motion
        check(f"{side} target at engage is the engage pose",
              close(t.translation(), ee0.translation())
              and close(t.rotation().as_matrix(), ee0.rotation().as_matrix())
              and close(c.travel(engage), np.zeros(3)))
        c.release()
        check(f"{side} release stops producing targets", c.target(engage) is None)


def test_clutch_translation_wrist_frame():
    print("\nClutch — translation, wrist frame")
    for side in SIDES:
        for axis in AXES:
            # Pushing along one of your own hand's axes moves the EE along the robot's
            ee0 = mink.SE3.from_rotation_and_translation(mink.SO3.identity(),
                                                         np.zeros(3))
            c = Clutch(side, translation_frame="wrist")
            c.engage(np.eye(4), ee0)
            wrist = _pose(np.eye(3), 0.10 * MANO_WRIST_AXES[side][axis])
            check(f"{side} push along {axis} lands on the robot's {axis}",
                  close(c.target(wrist).translation(),
                        0.10 * YOR_WRIST_AXES[side][axis]))

    # What the wrist frame costs, and the reason "world" is the default: the
    # same physical push lands somewhere else for every engage orientation.
    C = convention_matrix("left")
    for engage_yaw in (0.0, 0.9, -2.1):
        ee0 = mink.SE3.from_rotation_and_translation(mink.SO3.identity(), np.zeros(3))
        c = Clutch("left", translation_frame="wrist")
        R0 = mink.SO3.from_z_radians(engage_yaw).as_matrix()
        engage = _pose(R0, [0.2, -0.3, 0.9])
        c.engage(engage, ee0)
        push = np.array([0.10, 0.0, 0.0])
        wrist = _pose(R0, engage[:3, 3] + push)
        check(f"travel tracks engage orientation (yaw {engage_yaw:+.1f})",
              close(c.target(wrist).translation(), C @ R0.T @ push))

    ee0 = mink.SE3.from_rotation_and_translation(mink.SO3.identity(), np.zeros(3))
    c = Clutch("left", position_scale=0.5, translation_frame="wrist")
    c.engage(np.eye(4), ee0)
    push = np.array([0.10, 0.0, 0.0])
    turn = Rotation.from_rotvec(0.5 * MANO_WRIST_AXES["left"]["thumb"]).as_matrix()
    t = c.target(_pose(turn, push))
    check("position_scale scales translation only",
          close(t.translation(), 0.5 * C @ push)
          and abs(float(np.linalg.norm(t.rotation().log())) - 0.5) < 1e-9)


def test_clutch_translation_world_frame():
    print("\nClutch — translation, world frame")
    rng = np.random.default_rng(7)

    def rand_rot():
        return mink.SO3.exp(rng.normal(size=3)).as_matrix()

    # The headline, and the whole reason the two are decomposed: up is up at
    # every engage pose, which is exactly what the wrist frame gave up.
    up, flat = np.array([0.0, 0.0, 0.10]), np.array([0.10, -0.05, 0.0])
    vertical = horizontal = length = True
    for side in SIDES:
        for _ in range(24):
            R_e, t_e = rand_rot(), rng.normal(size=3)
            ee0 = mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rand_rot()), rng.normal(size=3))
            c = Clutch(side)
            c.engage(_pose(R_e, t_e), ee0)
            moved = c.target(_pose(R_e, t_e + up)).translation() - ee0.translation()
            vertical &= close(moved, up, 1e-12)
            flat_moved = (c.target(_pose(R_e, t_e + flat)).translation()
                          - ee0.translation())
            horizontal &= abs(float(flat_moved[2])) < 1e-12
            length &= abs(float(np.linalg.norm(flat_moved)
                                - np.linalg.norm(flat))) < 1e-12
    check("a metre up is a metre up, at every engage pose", vertical)
    check("horizontal motion stays horizontal", horizontal)
    check("the map is a rotation: no push is stretched", length)

    # Heading still comes from engage, which is what absorbs odom's arbitrary
    # yaw origin and is why there is no room calibration to set.
    ee0 = mink.SE3.from_rotation_and_translation(mink.SO3.identity(), np.zeros(3))
    push, seen = np.array([0.10, 0.0, 0.0]), []
    for engage_yaw in (0.0, 0.9, -2.1):
        c = Clutch("left")
        R0 = mink.SO3.from_z_radians(engage_yaw).as_matrix()
        c.engage(_pose(R0, np.zeros(3)), ee0)
        seen.append(c.target(_pose(R0, push)).translation())
    spun = all(not close(a, b, 1e-6) for a, b in zip(seen, seen[1:]))
    check("engage heading rotates the horizontal mapping (not odom-locked)",
          spun and all(abs(float(v[2])) < 1e-12 for v in seen))

    # Rotation is untouched by the split: both frames command the same one.
    same = True
    for side in SIDES:
        R_e, t_e = rand_rot(), rng.normal(size=3)
        ee0 = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rand_rot()), rng.normal(size=3))
        pair = [Clutch(side, translation_frame=f) for f in ("world", "wrist")]
        for c in pair:
            c.engage(_pose(R_e, t_e), ee0)
        turn = _pose(rand_rot(), t_e)
        same &= close(pair[0].target(turn).rotation().as_matrix(),
                      pair[1].target(turn).rotation().as_matrix(), 1e-12)
        # And engaging is still a zero-delta anchor in either frame
        same &= all(close(c.target(_pose(R_e, t_e)).translation(),
                          ee0.translation(), 1e-12) for c in pair)
    check("rotation and the engage anchor are identical in both frames", same)

    check("position_scale still scales translation only",
          _world_scale_only())

    # An engage pose whose horizontal map is a flip rather than a turn — no
    # yaw describes it, so the heading is genuinely undefined. Fall back to
    # odom's own axes and say so, rather than inventing one or dividing by ~0.
    flip = np.diag([1.0, -1.0, -1.0])
    R_e = flip @ convention_matrix("left")   # makes the full wrist map `flip`
    c = Clutch("left")
    c.engage(_pose(R_e, np.zeros(3)),
             mink.SE3.from_rotation_and_translation(mink.SO3.identity(), np.zeros(3)))
    t = c.target(_pose(R_e, np.array([0.10, 0.0, 0.0])))
    check("a headless engage pose falls back to odom axes instead of failing",
          t is not None and close(t.translation(), [0.10, 0.0, 0.0]))

    for bad in ("odom", "", "World"):
        try:
            Clutch("left", translation_frame=bad)
            rejected = False
        except ValueError:
            rejected = True
        check(f"translation_frame rejects {bad!r}", rejected)


def _world_scale_only() -> bool:
    """position_scale moves the wrist without touching the rotation delta."""
    ee0 = mink.SE3.from_rotation_and_translation(mink.SO3.identity(), np.zeros(3))
    full, half = Clutch("left"), Clutch("left", position_scale=0.5)
    turn = Rotation.from_rotvec(0.5 * MANO_WRIST_AXES["left"]["thumb"]).as_matrix()
    for c in (full, half):
        c.engage(np.eye(4), ee0)
    moved = _pose(turn, [0.10, 0.0, 0.0])
    return (close(half.target(moved).translation(),
                  0.5 * full.target(moved).translation())
            and abs(float(np.linalg.norm(half.target(moved).rotation().log()))
                    - 0.5) < 1e-9)


def test_clutch_rotation():
    print("\nClutch — rotation")
    for side in SIDES:
        for axis in AXES:
            # Turning your hand about one of its axes turns the EE about the same one
            theta = 0.4
            ee0 = mink.SE3.from_rotation_and_translation(
                mink.SO3.from_x_radians(1.1), np.zeros(3))
            c = Clutch(side)
            engage = _pose(mink.SO3.from_y_radians(0.7).as_matrix(), np.zeros(3))
            c.engage(engage, ee0)
            turn = Rotation.from_rotvec(theta * MANO_WRIST_AXES[side][axis]).as_matrix()
            wrist = _pose(engage[:3, :3] @ turn, np.zeros(3))
            expected = (ee0.rotation().as_matrix()
                        @ Rotation.from_rotvec(
                            theta * YOR_WRIST_AXES[side][axis]).as_matrix())
            check(f"{side} turn about {axis} turns the EE about {axis}",
                  close(c.target(wrist).rotation().as_matrix(), expected))


def test_clutch_pinned_orientation():
    print("\nClutch — pinned orientation")
    # Frozen rotation isolates translation, for checking the mapping in halves
    ee0 = mink.SE3.from_rotation_and_translation(mink.SO3.from_x_radians(1.1),
                                                 np.zeros(3))
    wrist = _pose(mink.SO3.from_z_radians(0.4).as_matrix(), [0.10, 0.0, 0.0])
    held = moves = True
    for frame in Clutch.TRANSLATION_FRAMES:
        c = Clutch("left", follow_orientation=False, translation_frame=frame)
        c.engage(np.eye(4), ee0)
        t = c.target(wrist)
        held &= close(t.rotation().as_matrix(), ee0.rotation().as_matrix())
        moves &= float(np.linalg.norm(t.translation() - ee0.translation())) > 0.09
    check("pinned rotation holds the engage orientation, either frame", held)
    check("pinned rotation still teleoperates translation, either frame", moves)
    # The frames disagree about *where* that push lands, which is the point
    c_w = Clutch("left", follow_orientation=False, translation_frame="wrist")
    c_w.engage(np.eye(4), ee0)
    check("wrist frame puts the push through the convention rotation",
          close(c_w.target(wrist).translation(),
                ee0.rotation().as_matrix() @ convention_matrix("left")
                @ [0.10, 0.0, 0.0]))

    # Without pin_rotation, "frozen" means whatever the arm was holding when the
    # clutch fired, so a re-engage after any drift silently commands a different
    # wrist angle.
    home = mink.SO3.from_x_radians(1.1)
    c = Clutch("left", follow_orientation=False, pin_rotation=home)
    seen = []
    for engage_rot, ee_rot in ((mink.SO3.identity(), home),
                               (mink.SO3.from_z_radians(2.3),
                                mink.SO3.from_y_radians(-0.8))):
        engage = _pose(engage_rot.as_matrix(), np.zeros(3))
        c.engage(engage, mink.SE3.from_rotation_and_translation(ee_rot, np.zeros(3)))
        seen.append(c.target(engage).rotation().as_matrix())
    check("pin_rotation survives re-engage",
          close(seen[0], home.as_matrix()) and close(seen[1], seen[0]))

    for side in SIDES:
        # The drawn operator triad sits on the target even when rotation is pinned
        ee0 = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_x_radians(0.7), np.array([0.3, -0.25, 0.4]))
        c = Clutch(side, follow_orientation=False)
        c.engage(_pose(mink.SO3.from_z_radians(0.8).as_matrix(), [0.2, -0.3, 0.9]), ee0)
        wrist = _pose(mink.SO3.from_y_radians(0.5).as_matrix(), [0.25, -0.2, 1.0])
        check(f"{side} operator_frame origin is the wrist target",
              close(c.operator_frame(wrist).translation(),
                    c.wrist_target(wrist).translation(), 1e-12)
              and not close(c.operator_frame(wrist).rotation().as_matrix(),
                            c.wrist_target(wrist).rotation().as_matrix()))


# ─────────────────────────────────────────────────────────────────────────────
# The flange→wrist offset
# ─────────────────────────────────────────────────────────────────────────────

def test_wrist_offset():
    print("\nFlange→wrist offset")
    m, _ = model()
    for side in SIDES:
        d = mujoco.MjData(m)

        def offset():
            site, body = d.site(f"{side}_arm_ee"), d.body(f"{side}_wuji_hand_orient")
            return site.xmat.reshape(3, 3).T @ (body.xpos - site.xpos)

        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        mujoco.mj_forward(m, d)
        at_home = offset()
        # It is only safe to snapshot because nothing between the site and the
        # hand articulates; a joint added there would make the mapping drift
        # with arm pose, silently and only at some configurations.
        rng = np.random.default_rng(0)
        for j in range(1, 8):
            d.qpos[m.joint(f"{side}_arm_joint{j}").qposadr[0]] += rng.uniform(-1, 1)
        mujoco.mj_forward(m, d)
        check(f"{side} offset is rigid across arm poses",
              np.linalg.norm(at_home) > 0.01 and close(offset(), at_home),
              f"|offset| {np.linalg.norm(at_home)*1e3:.1f} mm")

    offset = np.array([0.0, 0.0375, 0.0])
    ee0 = mink.SE3.from_rotation_and_translation(mink.SO3.from_x_radians(0.3),
                                                 np.array([0.3, -0.25, 0.4]))
    c = Clutch("left", wrist_offset=offset)
    c.engage(np.eye(4), ee0)
    wrist_home = ee0.translation() + ee0.rotation().as_matrix() @ offset
    turn = Rotation.from_rotvec(0.6 * MANO_WRIST_AXES["left"]["thumb"]).as_matrix()
    turned = _pose(turn, np.zeros(3))
    # Turning the wrist pivots the hand in place and swings the flange behind it.
    # Without the offset the flange is what holds still, which drags the hand
    # through an arc — visible as the target sitting above the hand, not on it.
    check("offset makes the wrist the pivot, not the flange",
          close(c.target(np.eye(4)).translation(), ee0.translation())
          and close(c.wrist_target(np.eye(4)).translation(), wrist_home)
          and close(c.wrist_target(turned).translation(), wrist_home)
          and np.linalg.norm(c.target(turned).translation()
                             - ee0.translation()) > 0.01)

    ee0 = mink.SE3.from_rotation_and_translation(mink.SO3.from_y_radians(0.9),
                                                 np.array([0.3, -0.25, 0.4]))
    wrist = _pose(np.eye(3), [0.1, -0.05, 0.02])
    moved = []
    for off in (np.zeros(3), np.array([0.0, 0.0375, 0.0])):
        c = Clutch("right", wrist_offset=off)
        c.engage(np.eye(4), ee0)
        moved.append(c.target(wrist).translation() - ee0.translation())
    check("offset leaves a pure push alone", close(moved[0], moved[1], 1e-12))


def test_operator_frame_lands_on_the_ik_target():
    print("\nOperator frame vs the commanded wrist")
    # The overlay triad is the axis-correctness diagnostic now that the hand
    # skeleton is gone, so it has to sit exactly on the commanded wrist — at
    # every pose, not just at engage. A wrong row in either axis table shows up
    # here as a rotation, and on screen as a mirrored triad.
    offset = np.array([0.0, 0.0375, 0.0])
    for side in SIDES:
        ee0 = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_x_radians(1.2), np.array([0.3, -0.25, 0.4]))
        R0, o0 = mink.SO3.from_z_radians(0.8).as_matrix(), [0.2, -0.3, 0.9]
        wrist_home = ee0.translation() + ee0.rotation().as_matrix() @ offset
        at_engage = tracks = True
        # Both frames: under "world" position and orientation come from
        # different maps, so this is exactly where they could diverge
        for frame in Clutch.TRANSLATION_FRAMES:
            c = Clutch(side, wrist_offset=offset, translation_frame=frame)
            c.engage(_pose(R0, o0), ee0)
            at_engage &= close(c.operator_frame(_pose(R0, o0)).translation(),
                               wrist_home)
            for rpy, origin in (((0.5, 0.0, 0.8), [0.2, -0.3, 0.9]),
                                ((-1.1, 0.7, 0.2), [0.3, 0.1, 1.0])):
                R = mink.SO3.from_rpy_radians(*rpy).as_matrix()
                got, want = c.operator_frame(_pose(R, origin)), c.wrist_target(
                    _pose(R, origin))
                tracks &= close(got.translation(), want.translation())
                tracks &= close(got.rotation().as_matrix(),
                                want.rotation().as_matrix())
        check(f"{side} operator frame is the robot's wrist at engage", at_engage)
        check(f"{side} operator frame tracks the commanded wrist everywhere",
              tracks)


# ─────────────────────────────────────────────────────────────────────────────
# The wire
# ─────────────────────────────────────────────────────────────────────────────

def test_wire_decode():
    print("\nWire decode")
    T_od = _pose(mink.SO3.from_rpy_radians(0.3, -0.2, 1.1).as_matrix(),
                 [1.0, 2.0, 0.5])
    T_oh = _pose(mink.SO3.from_z_radians(0.4).as_matrix(), [0.1, 0.2, 0.3])

    stream = AriaHandStream("localhost", sides=("left",))
    stream._ingest({
        "wire": 2, "seq": 3, "home_seq": 0,
        "left": {"T_odom_hand": T_oh.astype(np.float32),
                 "qpos": np.zeros(20, np.float32), "paused": False},
    })
    s = stream.snapshot()["left"]
    check("T_odom_hand passes through as the wrist pose",
          close(s.T_odom_wrist, T_oh, 1e-6))
    # The wire is float32 because T_odom_device was; everything downstream of
    # here does SE(3) algebra and wants float64.
    check("the pose is upcast to float64",
          s.T_odom_wrist.dtype == np.float64 and s.qpos.dtype == np.float64)
    check("paused passes through", s.paused is False)
    check("home_seq is read off the envelope", stream.home_seq() == 0)

    # A pre-wire-2 publisher still works, composed locally, for one release:
    # the two repos deploy to different machines and "the arms do not move" is
    # a worse thing to hand an operator than a warning.
    T_dh = _pose(mink.SO3.from_z_radians(0.4).as_matrix(), [0.1, 0.2, 0.3])
    stream = AriaHandStream("localhost", sides=("left",))
    stream._ingest({"T_odom_device": T_od,
                    "left": {"T_device_hand": T_dh, "paused": False}})
    check("a pre-wire-2 publisher is composed locally",
          close(stream.snapshot()["left"].T_odom_wrist, T_od @ T_dh))
    check("and says so exactly once", stream._warned == {"wire1"})
    check("with no home counter to act on", stream.home_seq() is None)

    # Never fall back to Aria's own wrist frame: its origin is a joint centre a
    # couple of cm off the landmark and its axes are a different convention.
    stream = AriaHandStream("localhost", sides=("left",))
    stream._ingest({"T_odom_device": T_od,
                    "left": {"T_device_wrist": T_dh, "paused": False}})
    check("a hand-frame-less publisher leaves the side unfollowable",
          stream.snapshot()["left"].T_odom_wrist is None)


def test_home_seq():
    print("\nHome counter")
    # A counter rather than an edge because PUB/SUB drops packets and this
    # subscriber conflates them: an edge can be missed, a total cannot.
    w = HomeSeqWatcher()
    check("no counter yet never fires", not w.update(None))
    check("joining a publisher mid-session adopts without firing",
          not w.update(5))
    check("an unchanged counter does not fire", not w.update(5))
    check("an increase fires", w.update(6))
    check("and only once", not w.update(6))
    # buffer=False means the client routinely skips values. A jump of three is
    # one gesture whose packets we did not see, not three requests -- homing
    # twice in a row is a hardware hazard.
    check("a jump of several fires exactly once", w.update(20))
    check("still only once", not w.update(20))
    # Same guard status.py puts on the cumulative `sends` total.
    check("a restarted publisher resyncs without firing", not w.update(0))
    check("and fires again from the new baseline", w.update(1))


def test_staleness_gate():
    print("\nStaleness gate")
    # commlink's buffer=False subscriber hands back the last payload forever, so
    # a publisher that dies mid-motion would otherwise leave the clutch engaged
    # on a target the operator can no longer release by gesture.
    T_oh = np.eye(4)
    stream = AriaHandStream("localhost", sides=("left",), stale_s=0.2)
    stream._ingest({"left": {"T_odom_hand": T_oh, "paused": False}})
    check("fresh sample is not forced paused",
          stream.snapshot()["left"].paused is False)
    stream._t_recv -= 0.5
    check("stale sample is forced paused",
          stream.snapshot()["left"].paused is True)

    stream = AriaHandStream("localhost", sides=("left",), stale_s=None)
    stream._ingest({"left": {"T_odom_hand": T_oh, "paused": False}})
    stream._t_recv -= 10.0
    check("stale_s=None disables the gate",
          stream.snapshot()["left"].paused is False)


# ─────────────────────────────────────────────────────────────────────────────
# AriaSource — the RPC path
# ─────────────────────────────────────────────────────────────────────────────

class _FakeStream:
    """Stands in for AriaHandStream so update() can be driven without commlink."""

    def __init__(self, sample, right=None, home_seq=None, meta=None):
        self.sample = sample
        self.right = right
        self._home_seq = home_seq
        self._meta = meta

    def start(self):
        pass

    def stop(self):
        pass

    def home_seq(self):
        return self._home_seq

    def meta(self):
        return self._meta

    def bump(self):
        """The publisher completed a home gesture."""
        self._home_seq = 1 if self._home_seq is None else self._home_seq + 1

    def snapshot(self):
        from robot.teleop.aria.stream import SideSample
        return {"left": self.sample,
                "right": self.right or SideSample(None, None, True)}


def test_aria_source():
    print("\nAriaSource")
    from robot.teleop.aria.source import AriaSource
    from robot.teleop.aria.stream import SideSample
    from robot.teleop.wholebody_teleop import InputSource, TeleopState

    src_text = (_REPO / "robot/teleop/aria/source.py").read_text()
    check("AriaSource is an InputSource with the full contract",
          issubclass(AriaSource, InputSource)
          and all(callable(getattr(AriaSource, m)) for m in
                  ("start", "stop", "update"))
          and AriaSource.state_refresh is None)
    # state_refresh() is the only way a source may talk to the server;
    # test_interface_contract.py regexes self.yor.<method>( out of the client.
    check("source.py never touches the RPC client directly",
          "self.yor." not in src_text)

    ee = mink.SE3.from_rotation_and_translation(mink.SO3.from_x_radians(0.5),
                                                np.array([0.3, -0.25, 0.4]))
    other = mink.SE3.from_rotation_and_translation(mink.SO3.identity(),
                                                   np.array([-0.3, -0.25, 0.4]))
    state = TeleopState(left_target=ee, right_target=other, lift_target=0.2)
    wrist = _pose(mink.SO3.from_z_radians(0.7).as_matrix(), [0.2, -0.3, 0.9])
    sample = SideSample(wrist, np.zeros(20), False)

    src = AriaSource("localhost", hand="left")
    src._stream = _FakeStream(sample)
    src.start()
    cmd = src.update(state, 1 / 30)
    check("an engaged hand produces an EE target", cmd.left_target is not None)
    # Arms only: no gripper, no homing, no toggles
    check("nothing but the arm target and the lift pin is commanded",
          cmd.left_gripper is None and cmd.right_gripper is None
          and not cmd.home_left and not cmd.home_right and not cmd.home_arms
          and not cmd.home_lift and not cmd.toggle_fix_base
          and not cmd.toggle_collisions and not cmd.quit)
    check("the idle hand is left alone", cmd.right_target is None)

    # Saying nothing about the lift does not hold it: both nodes start with
    # lift_target None, which the solver reads as "the lift is yours". Pin it
    # once, then never again.
    check("the lift is pinned on the first tick",
          cmd.lift_target is not None
          and abs(cmd.lift_target - state.lift_target) < 1e-12)
    check("the lift is not re-commanded after that",
          src.update(state, 1 / 30).lift_target is None)

    src = AriaSource("localhost", hand="left", hold_lift=False)
    src._stream = _FakeStream(sample)
    src.start()
    check("--no-aria-hold-lift leaves the lift to the solver",
          src.update(state, 1 / 30).lift_target is None)

    # Clutch reseed: engaging anchors on the robot's actual EE, so the first
    # target after engage is that pose exactly — zero delta, zero motion.
    server_ee = mink.SE3.from_rotation_and_translation(
        mink.SO3.from_y_radians(1.1), np.array([0.35, -0.30, 0.55]))
    src = AriaSource("localhost", hand="left", clutch_reseed=True)
    src._stream = _FakeStream(sample)
    src.start()
    src.state_refresh = lambda: {"left_ee_wxyz_xyz": server_ee.wxyz_xyz.tolist()}
    cmd = src.update(state, 1 / 30)
    check("clutch reseed anchors on the server's actual EE",
          close(cmd.left_target.translation(), server_ee.translation())
          and close(cmd.left_target.rotation().as_matrix(),
                    server_ee.rotation().as_matrix()))

    src = AriaSource("localhost", hand="left", clutch_reseed=False)
    src._stream = _FakeStream(sample)
    src.start()
    src.state_refresh = lambda: {"left_ee_wxyz_xyz": server_ee.wxyz_xyz.tolist()}
    cmd = src.update(state, 1 / 30)
    check("--no-clutch-reseed anchors on the local target",
          close(cmd.left_target.translation(), ee.translation()))

    # Home: the publisher detected both thumbs up on two released hands and
    # bumped its counter. The dwell and the released gate ran up there; what
    # is tested here is that the counter is acted on exactly once, and never
    # while this side is still following a hand.
    paused = SideSample(wrist, np.zeros(20), True)
    engaged = SideSample(wrist, np.zeros(20), False)

    def run(src, ticks=10):
        return [c for c in (src.update(state, 1 / 30) for _ in range(ticks))
                if c.home_arms or c.home_left or c.home_right]

    src = AriaSource("localhost", hand="both")
    stream = _FakeStream(paused, paused, home_seq=0)
    src._stream = stream
    src.start()
    check("an unchanged counter does not home", not run(src))
    stream.bump()
    fired = run(src)
    check("a counter bump runs the node's home_arms sequence",
          len(fired) == 1 and fired[0].home_arms
          and not fired[0].home_left and not fired[0].home_right,
          f"{len(fired)} home commands")
    check("and does not repeat while the counter holds", not run(src))

    # Joining a publisher that has already homed must not home on connect.
    src = AriaSource("localhost", hand="both")
    src._stream = _FakeStream(paused, paused, home_seq=7)
    src.start()
    check("a counter that is merely nonzero at connect does not home",
          not run(src))

    # Belt and braces: the publisher required both sides paused, but "nothing
    # is following either hand" is the whole safety argument and is worth
    # asserting locally rather than trusting a remote definition of paused.
    src = AriaSource("localhost", hand="both")
    stream = _FakeStream(engaged, paused, home_seq=0)
    src._stream = stream
    src.start()
    src.update(state, 1 / 30)          # engage the left clutch
    stream.bump()
    check("a bump while a hand is still engaged never homes", not run(src))

    # A one-handed session cannot make the gesture at all, whatever arrives
    src = AriaSource("localhost", hand="left")
    src._stream = _FakeStream(paused, home_seq=0)
    src.start()
    check("a single-hand session has no home gesture", src._home is None)
    src._stream._home_seq = 9
    check("and ignores a counter that moves", not run(src))

    src = AriaSource("localhost", hand="both", home_gesture=False)
    stream = _FakeStream(paused, paused, home_seq=0)
    src._stream = stream
    src.start()
    stream.bump()
    check("home.gesture false disables it entirely", not run(src))

    # Releasing when the publisher pauses, and no target while released
    src = AriaSource("localhost", hand="left")
    src._stream = _FakeStream(sample)
    src.start()
    src.update(state, 1 / 30)
    src._stream = _FakeStream(sample._replace(paused=True))
    cmd = src.update(state, 1 / 30)
    check("a paused publisher releases the clutch",
          not src._clutches["left"].engaged and cmd.left_target is None)


# ─────────────────────────────────────────────────────────────────────────────
# The two rendering fixes
# ─────────────────────────────────────────────────────────────────────────────

def test_marker_rides_the_wrist():
    print("\nRendering fix A — marker on the hand, not the flange")
    from robot.arm.wholebody_ik import WholeBodyIK, WholeBodyIKConfig

    ik = WholeBodyIK(str(SCENE), WholeBodyIKConfig(dt=0.01, max_iters=10))
    ik.init_from_keyframe("home")
    m, d = ik.model, ik.data
    offset = {side: d.site(f"{side}_arm_ee").xmat.reshape(3, 3).T
              @ (d.body(f"{side}_wuji_hand_orient").xpos
                 - d.site(f"{side}_arm_ee").xpos)
              for side in SIDES}

    T_l, T_r = ik.forward_kinematics()
    targets = {
        "left": mink.SE3.from_rotation_and_translation(
            T_l.rotation() @ mink.SO3.from_z_radians(0.35),
            T_l.translation() + np.array([0.02, -0.06, 0.04])),
        "right": T_r,
    }
    for _ in range(60):
        result = ik.solve(targets["left"], targets["right"], lift_target=None)
    ik.apply_to_sim_kinematic(d, result)
    mujoco.mj_forward(m, d)

    for side in SIDES:
        R = targets[side].rotation().as_matrix()
        hand = d.body(f"{side}_wuji_hand_orient").xpos
        fixed = np.linalg.norm(
            targets[side].translation() + R @ offset[side] - hand)
        flange = np.linalg.norm(targets[side].translation() - hand)
        # The fixed form must land on the hand, AND the unfixed one must not --
        # so this fails if the offset is dropped or silently zeroed.
        check(f"{side} marker sits on the WUJI hand", fixed < 1e-3,
              f"{fixed*1e3:.4f} mm")
        check(f"{side} flange is a real distance behind it", flange > 0.03,
              f"{flange*1e3:.1f} mm")


def test_overlay_offset_ordering():
    print("\nRendering fix B — overlays ride mjviser's scene offset")
    # mjviser assigns _scene_offset inside update_from_mjdata, so the overlays
    # drawn below have to ride the value it just used. Frozen as text because
    # the ordering is the regression and there is no browser here.
    text = (_REPO / "robot/teleop/aria/sim_viz.py").read_text()
    i_update = text.index("scene.update_from_mjdata(")
    i_sync = text.index("sync_overlay_offset()", i_update)
    i_draw = text.index("draw_operator_frame(side", i_update)
    check("update_from_mjdata → sync_overlay_offset → draw_*",
          i_update < i_sync < i_draw)

    import re
    names = re.findall(r'(?:add_frame|add_line_segments|add_point_cloud|'
                       r'remove_by_name)\(\s*(?:server,\s*)?f?"([^"]+)"', text)
    stray = [n for n in names if not n.startswith("/overlay")]
    check("every viser node hangs off /overlay", not stray, ", ".join(stray))


def test_config():
    print("\nConfig")
    from robot.teleop.aria.config import DEFAULT_CONFIG, AriaConfig

    check("config/aria_teleop.yaml ships with the repo", DEFAULT_CONFIG.exists())
    cfg = AriaConfig.load()
    check("every section parses",
          all(hasattr(cfg, s) for s in ("publisher", "mapping", "clutch", "sim")))
    # Relative paths resolve against the repo root, since either entry point may
    # be run from anywhere.
    check("scene resolves to an absolute path that exists",
          Path(cfg.mapping["scene"]).is_absolute()
          and Path(cfg.mapping["scene"]).exists(),
          str(cfg.mapping["scene"]))
    check("hand is one the sources accept",
          cfg.mapping["hand"] in ("left", "right", "both"))
    check("translation_frame is one the clutch accepts",
          cfg.mapping["translation_frame"] in Clutch.TRANSLATION_FRAMES,
          str(cfg.mapping["translation_frame"]))

    # A partial file must still run: missing keys fall back, they do not crash.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("publisher:\n  host: 10.0.0.9\n")
        partial = f.name
    cfg = AriaConfig.load(partial)
    check("a partial file keeps its override and defaults the rest",
          cfg.publisher["host"] == "10.0.0.9" and cfg.publisher["port"] == 5555
          and cfg.clutch["hold_lift"] is True)

    try:
        AriaConfig.load("/nonexistent/aria.yaml")
        missing_raises = False
    except FileNotFoundError:
        missing_raises = True
    check("an explicit path that is not there fails loudly", missing_raises)


def main() -> int:
    for test in (
        test_config,
        test_wrist_axis_tables,
        test_convention_matrix_matches_the_mjcf_hand_mount,
        test_hand_joints,
        test_clutch_engage_release,
        test_clutch_translation_wrist_frame,
        test_clutch_translation_world_frame,
        test_clutch_rotation,
        test_clutch_pinned_orientation,
        test_wrist_offset,
        test_operator_frame_lands_on_the_ik_target,
        test_wire_decode,
        test_home_seq,
        test_staleness_gate,
        test_aria_source,
        test_marker_rides_the_wrist,
        test_overlay_offset_ordering,
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
