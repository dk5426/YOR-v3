#!/usr/bin/env python3
"""Rebuild the Aero finger/thumb skeleton in aero_mjcf/{side}_hand_attach.xml
directly from the ROS2 URDF's own joint origins/axes/limits, replacing the
values previously inherited from aero-hand-open's MuJoCo Playground RL scene.

Why: the URDF and the Playground scene are two independently-parameterized
sources, not just mirrored differently -- e.g. the thumb's CMC-abduction
joint axis sign is flipped between them, and position values differ by
4-20% per axis, well beyond a sign flip. aria2robot itself builds the whole
Aero hand -- mesh AND skeleton -- from this exact URDF alone (via
yourdfpy/ViserUrdf), never touching the Playground scene. Mixing sources in
this repo produced a real mismatch between where the palm mesh's knuckle
bumps are and where the finger-skeleton bodies actually sit. See the
"Fix Aero hand" plan for the full derivation.

This is a mechanical, deterministic translation (URDF's <origin>+<axis>+
<limit> *is* MuJoCo's body pos/quat + joint axis/range) -- not an estimated
constant -- so it edits the fragment files directly. Review with `git diff`.

Run: python3 tools/rebuild_aero_skeleton.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parent.parent
URDF_DIR = REPO / "aero-hand-open/ros2/src/aero_hand_open_description/urdf"
FRAGMENT_DIR = REPO / "description/aero_mjcf"

NONTHUMB_FINGERS = ["index", "middle", "ring", "pinky"]

_DAMPING_BY_ROLE = {
    "thumb_cmc_abd": (0.02, 0.001559127, 0.02),
    "thumb_cmc_flex": (0.02, 0.001559127, 0.02),
    "thumb_mcp": (0.02, 0.001559127, 0.02),
    "thumb_ip": (0.02, 0.001559127, 0.02),
}


def _damping_for(role):
    if role in _DAMPING_BY_ROLE:
        return _DAMPING_BY_ROLE[role]
    if role.endswith("mcp_flex"):
        return (0.1, 0.001559127, 0.02)
    if role.endswith("pip") or role.endswith("dip"):
        return (0.05, 0.001559127, 0.02)
    raise ValueError(f"no damping table entry for joint role {role!r}")


def _fmt(x):
    return f"{x:.10g}"


def _vec_str(xyz):
    return " ".join(_fmt(v) for v in xyz)


def _quat_wxyz(rot):
    x, y, z, w = rot.as_quat()
    return f"{_fmt(w)} {_fmt(x)} {_fmt(y)} {_fmt(z)}"


def _is_identity_rpy(rpy):
    return all(abs(v) < 1e-12 for v in rpy)


def parse_urdf_joints(side):
    root = ET.parse(URDF_DIR / f"aero_hand_open_{side}.urdf").getroot()
    joints = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
        rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
        axis_el = j.find("axis")
        axis = tuple(float(v) for v in axis_el.get("xyz").split()) if axis_el is not None else None
        limit_el = j.find("limit")
        limit = (
            (float(limit_el.get("lower")), float(limit_el.get("upper")))
            if limit_el is not None
            else None
        )
        joints[j.get("name")] = dict(xyz=xyz, rpy=rpy, axis=axis, limit=limit)
    return joints


def set_pos_quat(el, xyz, rpy):
    if any(abs(v) > 1e-12 for v in xyz):
        el.set("pos", _vec_str(xyz))
    elif "pos" in el.attrib:
        del el.attrib["pos"]
    if not _is_identity_rpy(rpy):
        el.set("quat", _quat_wxyz(Rotation.from_euler("xyz", rpy)))
    elif "quat" in el.attrib:
        del el.attrib["quat"]


def update_body_and_joint(root, body_name, joint_name, urdf_joints, side):
    body_el = root.find(f".//body[@name='{body_name}']")
    if body_el is None:
        raise ValueError(f"body {body_name!r} not found")
    j = urdf_joints[joint_name]
    set_pos_quat(body_el, j["xyz"], j["rpy"])
    joint_el = body_el.find(f"./joint[@name='{joint_name}']")
    if joint_el is None:
        raise ValueError(f"joint {joint_name!r} not found inside body {body_name!r}")
    joint_el.set("axis", _vec_str(j["axis"]))
    joint_el.set("range", f"{_fmt(j['limit'][0])} {_fmt(j['limit'][1])}")
    role = joint_name[len(side) + 1 :]
    damping, armature, frictionloss = _damping_for(role)
    joint_el.set("damping", _fmt(damping))
    joint_el.set("armature", _fmt(armature))
    joint_el.set("frictionloss", _fmt(frictionloss))


def update_tip_geoms(root, distal_body_name, tip_mesh_name, tip_joint):
    distal_el = root.find(f".//body[@name='{distal_body_name}']")
    if distal_el is None:
        raise ValueError(f"body {distal_body_name!r} not found")
    geoms = distal_el.findall(f"./geom[@mesh='{tip_mesh_name}']")
    if not geoms:
        raise ValueError(f"no tip geoms with mesh {tip_mesh_name!r} in {distal_body_name!r}")
    for g in geoms:
        set_pos_quat(g, tip_joint["xyz"], tip_joint["rpy"])


HEADER = """<!-- Aero {Side} hand; wire order matches robot/hand/aero_driver.py::
     canonical_joint_names('{side}') exactly (16 joints, no reordering).
     Body pos/quat and joint axis/range are derived directly from the ROS2
     URDF (aero-hand-open/ros2/src/aero_hand_open_description/urdf/
     aero_hand_open_{side}.urdf) via tools/rebuild_aero_skeleton.py -- NOT
     from aero-hand-open/sim_rl's MuJoCo Playground scene. That scene and the
     URDF are two independently-parameterized sources (not just mirrored
     differently -- e.g. the thumb's CMC-abduction axis sign is flipped
     between them), and mixing them with the URDF-sourced palm mesh produced
     a real mismatch between the mesh's knuckle bumps and the finger
     skeleton. The URDF's own `{{finger}}_f_link` spring-visualization stub
     bodies (Playground-RL-tendon-model-only, no URDF equivalent) are
     dropped; each finger's proximal link is a direct child of this body,
     matching the URDF's own base_link -> proximal_link joint. Per-joint
     damping/armature/frictionloss aren't in the URDF and are kept from the
     original Playground-derived values, keyed by joint role. Inertials and
     the primitive-box collision proxies are also kept as-is (Playground- and
     hand-tuned respectively; not implicated in the mesh/skeleton mismatch).
     MuJoCo ignores euler/quat on the root body of an included fragment --
     mount frame lives in the parent scene file ({side}_aero_hand_orient). -->
"""


def rebuild(side):
    urdf_joints = parse_urdf_joints(side)
    path = FRAGMENT_DIR / f"{side}_hand_attach.xml"
    # The existing header comment contains "--" sequences, invalid inside an
    # XML comment per spec (expat rejects it even though MuJoCo tolerates it)
    # -- strip it; we replace it with our own header below anyway.
    body_text = path.read_text().split("-->", 1)[1]
    root = ET.fromstring(f"<root>{body_text}</root>")
    hand_body = root.find("body")

    thumb_chain = [
        (f"{side}_t_link", f"{side}_thumb_cmc_abd"),
        (f"{side}_thumb_mcp_link", f"{side}_thumb_cmc_flex"),
        (f"{side}_thumb_proximal_link", f"{side}_thumb_mcp"),
        (f"{side}_thumb_distal_link", f"{side}_thumb_ip"),
    ]
    for body_name, joint_name in thumb_chain:
        update_body_and_joint(root, body_name, joint_name, urdf_joints, side)
    update_tip_geoms(
        root,
        f"{side}_thumb_distal_link",
        f"{side}_thumb_tip_link",
        urdf_joints[f"{side}_thumb_tip"],
    )

    for finger in NONTHUMB_FINGERS:
        f_link_name = f"{side}_{finger}_f_link"
        f_link_el = hand_body.find(f"./body[@name='{f_link_name}']")
        if f_link_el is None:
            raise ValueError(f"body {f_link_name!r} not found")
        proximal_el = f_link_el.find("body")
        if proximal_el is None:
            raise ValueError(f"{f_link_name!r} has no nested body to reparent")
        idx = list(hand_body).index(f_link_el)
        hand_body.remove(f_link_el)
        hand_body.insert(idx, proximal_el)

        chain = [
            (f"{side}_{finger}_proximal_link", f"{side}_{finger}_mcp_flex"),
            (f"{side}_{finger}_middle_link", f"{side}_{finger}_pip"),
            (f"{side}_{finger}_distal_link", f"{side}_{finger}_dip"),
        ]
        for body_name, joint_name in chain:
            update_body_and_joint(root, body_name, joint_name, urdf_joints, side)
        update_tip_geoms(
            root,
            f"{side}_{finger}_distal_link",
            f"{side}_{finger}_tip_link",
            urdf_joints[f"{side}_{finger}_tip"],
        )

    ET.indent(root, space="  ")
    body_str = ET.tostring(hand_body, encoding="unicode")
    header = HEADER.format(Side=side.capitalize(), side=side)
    path.write_text(header + body_str + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    for side in ("left", "right"):
        rebuild(side)
