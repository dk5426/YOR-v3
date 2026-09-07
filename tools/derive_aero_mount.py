"""Derive the Aero hand's flange-mount rotation + flush-mount translation.

Replaces two placeholder numbers in description/robot_wholebody_aero.xml's
{side}_aero_hand_orient body:

  rotation     was copied verbatim from aria2robot's own MuJoCo scene
               (aero-hand-open/sim_rl/simulation/mujoco/{side}_hand.xml),
               calibrated against that scene's own `tetheria_mount` body --
               a frame with no relationship to this robot's flange.
  translation  was that same scene's `palm` offset, plus a separate
               WUJI-bracket-shaped outer `{side}_aero_mount` body -- neither
               has anything to do with Aero, which has no measured mount.

This script instead derives both from first principles:

  rotation     aria2robot's aero_retargeting.py::calib_rotation(side, urdf)
               computes the rotation from the Aero URDF's `{side}_base_link`
               native mesh axes into the "wrist-frame convention" (MANO/WUJI
               axes) -- the same convention `{side}_wuji_hand_orient` already
               sits in on the WUJI side (see mano_wrist_frame's docstring in
               aria2robot). Composing that with WUJI's own already-measured
               flange -> wrist-frame-convention rotation (the two eulers on
               {side}_wuji_nero_mount / {side}_wuji_hand_orient) gives a
               flange -> Aero-base_link rotation with no free placeholder:
               "mount the hand facing the same way WUJI does" is the only
               assumption, and it's an explicit, named one.
  translation  each side's own {side}_base_link.STL bounding box, rotated by
               that same composite, tells us how far the mesh sits behind base_link's
               own origin along whichever flange axis WUJI's own mesh is
               flush against (found empirically from WUJI's own, working,
               mount) -- solved so the palm's back face lands at 0 on that
               axis instead of clipping through or floating off the flange.

Pure numpy + a hand-rolled binary-STL reader -- no yourdfpy/aria2robot
import, so this stays a one-off derivation, not a runtime dependency. Rerun
this if the Aero URDF or WUJI's own mount numbers ever change.
"""

import re
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parent.parent

# -- 1. calib_rotation, reimplemented -----------------------------------
# Ported verbatim (no aria2robot import) from:
#   aria2robot/src/utils/retargeting/aero_retargeting.py::calib_rotation
#   aria2robot/external/wuji-retargeting/wuji_retargeting/mediapipe.py
#     ::estimate_frame_from_hand_points, OPERATOR2MANO_{LEFT,RIGHT}

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float)
_OPERATOR2MANO = {"left": OPERATOR2MANO_LEFT, "right": OPERATOR2MANO_RIGHT}


def estimate_frame_from_hand_points(kp: np.ndarray) -> np.ndarray:
    points = kp[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    points = points - np.mean(points, axis=0, keepdims=True)
    _u, _s, v = np.linalg.svd(points)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    return np.stack([x, normal, z], axis=1)


def calib_rotation(side: str, index_mcp: np.ndarray, middle_mcp: np.ndarray) -> np.ndarray:
    """Rotation from {side}_base_link's native mesh axes into wrist-frame convention."""
    pts = np.zeros((21, 3))
    pts[5], pts[9] = index_mcp, middle_mcp
    r_anchor_in_wristframe = estimate_frame_from_hand_points(pts) @ _OPERATOR2MANO[side]
    return r_anchor_in_wristframe.T


# index_mcp_flex / middle_mcp_flex joint origins, relative to {side}_base_link,
# read straight from the ROS2 URDF (aero-hand-open/ros2/src/
# aero_hand_open_description/urdf/aero_hand_open_{side}.urdf). This USED to
# read the Playground MuJoCo scene's equivalent points instead (that scene's
# palm frame mirrors left/right by negating Y, vs. the URDF's negating X --
# two different conventions for "the same" base_link/palm frame), because at
# the time description/aero_mjcf/{side}_hand_attach.xml's finger skeleton was
# itself still sourced from that Playground scene, and calib_rotation's
# reference points have to share the skeleton's own native frame.
# tools/rebuild_aero_skeleton.py has since rebuilt that skeleton directly
# from this same URDF (the Playground scene and the URDF are two
# independently-parameterized sources, not just differently-mirrored -- e.g.
# the thumb's CMC-abduction axis sign is flipped between them), so mesh,
# skeleton and these reference points are now all one source -- back to the
# URDF, no reconciliation needed.
_INDEX_MCP = {
    "right": np.array([0.030626, -0.0011782, 0.10542]),
    "left": np.array([-0.030626, -0.0011782, 0.10542]),
}
_MIDDLE_MCP = {
    "right": np.array([0.0075, 0.0, 0.106]),
    "left": np.array([-0.0075, 0.0, 0.106]),
}

# -- 2. WUJI's own measured flange -> wrist-frame-convention rotation ----
# Read straight out of the live XML so this never drifts out of sync with
# the actual mount in use.

_WUJI_XML = REPO / "description" / "robot_wholebody_wuji.xml"


def _read_euler(xml_text: str, body_name: str) -> np.ndarray:
    m = re.search(rf'<body name="{body_name}"[^>]*\beuler="([^"]+)"', xml_text)
    if m is None:
        raise ValueError(f"no euler on body {body_name!r}")
    return np.array([float(x) for x in m.group(1).split()])


def _read_pos(xml_text: str, body_name: str) -> np.ndarray:
    m = re.search(rf'<body name="{body_name}"[^>]*\bpos="([^"]+)"', xml_text)
    if m is None:
        raise ValueError(f"no pos on body {body_name!r}")
    return np.array([float(x) for x in m.group(1).split()])


def wuji_composite(side: str) -> Rotation:
    """Flange -> {side}_wuji_hand_orient rotation only (== wrist-frame convention)."""
    xml_text = _WUJI_XML.read_text()
    mount_euler = _read_euler(xml_text, f"{side}_wuji_nero_mount")
    orient_euler = _read_euler(xml_text, f"{side}_wuji_hand_orient")
    # MuJoCo's default eulerseq "xyz" is extrinsic (fixed parent-frame) axes,
    # applied in x,y,z order -- scipy's uppercase 'XYZ' is the same convention.
    r_mount = Rotation.from_euler("XYZ", mount_euler)
    r_orient = Rotation.from_euler("XYZ", orient_euler)
    return r_mount * r_orient


def wuji_hand_orient_pose_in_flange(side: str) -> tuple[Rotation, np.ndarray]:
    """Full (rotation, translation) of {side}_wuji_hand_orient, in flange coords."""
    xml_text = _WUJI_XML.read_text()
    mount_euler = _read_euler(xml_text, f"{side}_wuji_nero_mount")
    orient_euler = _read_euler(xml_text, f"{side}_wuji_hand_orient")
    mount_pos = _read_pos(xml_text, f"{side}_wuji_nero_mount")
    orient_pos = _read_pos(xml_text, f"{side}_wuji_hand_orient")
    r_mount = Rotation.from_euler("XYZ", mount_euler)
    r_orient = Rotation.from_euler("XYZ", orient_euler)
    pos = mount_pos + r_mount.apply(orient_pos)
    return r_mount * r_orient, pos


# -- 3. Minimal binary-STL vertex reader (no extra deps) -----------------


def stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n_tri = struct.unpack_from("<I", data, 80)[0]
    verts = np.empty((n_tri * 3, 3), dtype=np.float64)
    offset = 84
    for i in range(n_tri):
        v = struct.unpack_from("<9f", data, offset + 12)
        verts[3 * i : 3 * i + 3] = np.array(v).reshape(3, 3)
        offset += 50
    return verts


def quat_wxyz(r: Rotation) -> np.ndarray:
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


# A dead end, kept as a comment rather than code so it isn't tried again:
# mirroring one side's confirmed-correct rotation through a flange-local-Y
# reflection got the OTHER side's palm-normal right but left its fingers
# pointing 180deg backward. That symptom -- a rotation can partially align a
# mirror-image mesh but never fully -- was the tell that the two sides
# weren't using genuinely mirror-image meshes at all: both left_frame_link
# and right_frame_link were pointing at the exact same shared, generic
# aero_palm_link.STL (the RL sim's approximation, fine for physics-only
# training, wrong for anything that has to look like a real chiral hand).
# Confirmed by md5: that shared file matched aero-hand-open/sim_rl/.../
# assets/base_link.STL, not either side's real
# aero_hand_open_description/meshes/{side}_base_link.STL. Once each side
# points at its own true mesh, each side's rotation is independently
# derivable straight from calib_rotation -- no mirroring needed.


# Manual correction on the flush axis, on top of the derived push -- see its
# use in main() below. Not derived from mesh/URDF data; there's no measured
# Aero mount to derive a clearance margin from, so this is eyeballed against
# the sim render.
_FLUSH_CLEARANCE_NUDGE = 0.01


def main() -> None:
    wuji_mesh = stl_vertices(REPO / "description" / "meshes" / "wuji_nero_mount_left.stl")
    aero_mesh_native = {
        side: stl_vertices(
            REPO / "description" / "meshes" / "aero_meshes" / side / f"{side}_base_link.STL"
        )
        for side in ("left", "right")
    }

    for side in ("left", "right"):
        r_calib = calib_rotation(side, _INDEX_MCP[side], _MIDDLE_MCP[side])
        q_flange_to_baselink = wuji_composite(side) * Rotation.from_matrix(r_calib)

        print(f"\n== {side} ==")
        print("flange -> aero_hand_orient quat (w x y z):", quat_wxyz(q_flange_to_baselink))

        # Where does WUJI's own mesh sit, in flange coordinates, along each
        # axis, given its already-working mount? Establishes which axis
        # (and which sign) is "into the flange" for THIS naming convention.
        r_wuji_full, t_wuji_full = wuji_hand_orient_pose_in_flange(side)
        wuji_pts_flange = r_wuji_full.apply(wuji_mesh) + t_wuji_full
        wuji_min = wuji_pts_flange.min(axis=0)
        wuji_max = wuji_pts_flange.max(axis=0)
        print("WUJI mesh bbox in flange frame: min", wuji_min, "max", wuji_max)

        # Aero mesh, rotated only (translation not yet applied) -- where
        # would it sit if placed with zero translation at the flange?
        aero_pts_rot = q_flange_to_baselink.apply(aero_mesh_native[side])
        aero_min = aero_pts_rot.min(axis=0)
        aero_max = aero_pts_rot.max(axis=0)
        print("Aero mesh bbox (rotated, untranslated): min", aero_min, "max", aero_max)

        # Identify the flush axis as the one where WUJI's OWN mesh bbox does
        # NOT straddle 0 -- i.e. the whole mesh sits on one side of the mount
        # origin along that axis (nothing pokes through behind the flange).
        # This used to be `argmin(abs(wuji_min))`, which is the wrong test:
        # it just finds whichever axis happens to have the smallest-magnitude
        # minimum, regardless of whether that axis straddles 0 at all. For
        # WUJI (left) that picked axis 2 (min -0.0209, max +0.0189 -- clearly
        # straddling 0, an incidental near-tie, not "flush"), when axis 1
        # (min +0.0375, max +0.0735 -- entirely positive, nothing behind the
        # mount) is the one that actually matches the flush semantics, and
        # matches this file's own original comment ("WUJI's mesh sits flush
        # on local Y") before the heuristic silently disagreed with it.
        # Result: axis 2 was wrongly given the flush (clearance) push, and
        # axis 1 was wrongly left at 0/centered instead of getting the
        # clearance push -- which is exactly backwards from what's needed,
        # confirmed by the user visually (a centering fix on axis 1 moved the
        # hand "too far back"; axis 2 needed a reduction instead).
        same_sign = [
            np.sign(wuji_min[a]) == np.sign(wuji_max[a]) and np.sign(wuji_min[a]) != 0
            for a in range(3)
        ]
        flush_candidates = [a for a in range(3) if same_sign[a]]
        assert len(flush_candidates) == 1, (
            f"ambiguous/ununique flush axis for {side}: min={wuji_min} max={wuji_max}"
        )
        flush_axis = flush_candidates[0]
        pos = np.zeros(3)
        for axis in range(3):
            if axis == flush_axis:
                # The derived push alone still left the palm's base slightly
                # intersecting the flange face -- confirmed visually by the
                # user. _FLUSH_CLEARANCE_NUDGE is a manual correction on top
                # of the derived value, not itself derived from any mesh/URDF
                # data (there's no measured Aero mount to derive it from).
                pos[axis] = -aero_min[axis] + _FLUSH_CLEARANCE_NUDGE
            else:
                pos[axis] = -(aero_min[axis] + aero_max[axis]) / 2.0
        print(f"flush axis (0=x,1=y,2=z): {flush_axis} (WUJI range there: {wuji_min[flush_axis]:.6f} to {wuji_max[flush_axis]:.6f})")
        print("flange -> aero_hand_orient pos:", pos)


if __name__ == "__main__":
    main()
