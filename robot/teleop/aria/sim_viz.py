"""sim_viz.py — Aria hands drive the whole YORv3 in simulation, in the browser.

One process, plain `python`: subscribes to the aria2robot publisher, runs YOR's
own whole-body IK over description/scene_wholebody.xml, and renders the result
through mjviser. No RPC hop, no `mjpython`, no yor_mujoco.py.

Each wrist pose commands one arm's end-effector; base, lift and both 7-DOF arms
are solved together, so the chassis rolls and the lift extends on their own once
you reach past the arms. The retargeted finger angles go straight into the hand
joints of the same model — no wire, no `Hands`, so this stays the shortest way
to see all 20 of them. The node path reaches them too, through
robot/hand/hands.py, which is the one that also drives real hands.

    # publisher, from the aria2robot repo
    python -m aria2robot.stream_pub --wifi
    # here
    python robot/teleop/aria/sim_viz.py --pub-host <ip>
    # -> http://localhost:8080

Settings live in config/aria_teleop.yaml, shared with `--input aria`; only
--config, --pub-host and --hand are on the command line.

Hold a shaka for ~0.5 s to engage or disengage a side. Engaging pins your wrist
frame to the robot's; everything after is a delta in that frame. While
disengaged the arm target and the fingers both freeze.

Two things on screen are worth knowing how to read:

  the ik_target sphere sits INSIDE the WUJI palm, not on the thin flange triad
  37.5 mm behind it -- the marker rides the wrist, not the site the IK targets.

  the mapped operator triad (long thin needles) sits coincident and parallel
  with that sphere's own thick capsules. Mirrored or 90-degrees-off means an
  axis table in clutch.py is wrong; a gap with matching axes is IK tracking.

The operator's hand used to be drawn here as a 21-point skeleton as well. The
landmarks it needed no longer cross the wire -- the publisher retargets and
detects gestures itself now -- so the triad is the axis-correctness diagnostic,
and it answers the same question with two fewer scene nodes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from loop_rate_limiters import RateLimiter
from mjviser import ViserMujocoScene
from rich.console import Group
from rich.live import Live
from rich.table import Table

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from robot.arm.wholebody_ik import WholeBodyIK, WholeBodyIKConfig
from robot.teleop.aria.clutch import Clutch
from robot.teleop.aria.config import AriaConfig
from robot.teleop.aria.stream import AriaHandStream, HomeSeqWatcher, canonical_joint_names
from robot.teleop.status import console, fmt_xyz
from robot.teleop.status import log as _log

DEFAULT_SCENE = _REPO / "description" / "scene_wholebody.xml"

SIDE_INDEX = {"left": 0, "right": 1}
# Pushing every solve to the browser is wasted work; 1-in-3 of 100 Hz is plenty
RENDER_EVERY = 3


def log(msg: str, style: str = "cyan") -> None:
    """One `[aria]` line, through the client's shared console.

    Not `print`: once the status table is live, a bare write tears through it
    instead of scrolling above it.
    """
    _log(msg, style=style, prefix="aria")


def _tracking_table(rows: list[dict], banner: str) -> Table:
    """Where each hand is commanded, and whether the clutch is following it.

    The mapping error `d` used to sit here, as the distance between the
    commanded wrist and the operator's own wrist landmark. Measuring it needed
    the landmarks, which no longer arrive; the operator triad in the 3D view
    now carries that check, and `pos_err` in the next table still carries how
    far the arm lags the command.
    """
    table = Table(title=banner, expand=False, show_lines=False)
    table.add_column("Hand", style="cyan", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("ik_target", style="green", justify="right", no_wrap=True)
    for r in rows:
        table.add_row(
            r["side"],
            ("[green]ENGAGED[/green]" if r["engaged"] else "[yellow]paused[/yellow]"),
            r["ik_target"],
        )
    return table


def _mapping_table(rows: list[dict], result) -> Table:
    """Clutch travel and IK tracking error.

    The footer carries the DOFs the hands never command.
    """
    bx, by, bt = result.base_position
    table = Table(expand=False, show_lines=False,
                  caption=(f"lift {result.lift_q:.3f} m   "
                           f"base ({bx:+.2f}, {by:+.2f}, {bt:+.2f})   "
                           f"solved={result.solved} iters={result.iters}"))
    table.add_column("Hand", style="cyan", no_wrap=True)
    table.add_column("travel", justify="right", no_wrap=True)
    table.add_column("pos mm", style="magenta", justify="right", no_wrap=True)
    table.add_column("ori mrad", style="magenta", justify="right", no_wrap=True)
    for r in rows:
        table.add_row(r["side"], r["travel"],
                      f"{r['pos_err']:.1f}", f"{r['ori_err']:.1f}")
    return table


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--config", default=None,
                   help="settings file (default: config/aria_teleop.yaml)")
    p.add_argument("--pub-host", default=None,
                   help="override the config's publisher host")
    p.add_argument("--hand", choices=["left", "right", "both"], default=None,
                   help="override the config's hand")
    args = p.parse_args()

    cfg = AriaConfig.load(args.config)
    if args.pub_host:
        cfg.publisher["host"] = args.pub_host
    if args.hand:
        cfg.mapping["hand"] = args.hand
    console.print(cfg.describe(), markup=False, highlight=False)

    hand = cfg.mapping["hand"]
    sides = ("left", "right") if hand == "both" else (hand,)
    ik_rate = int(cfg.sim["ik_rate_hz"])

    log(f"scene: {cfg.mapping['scene']}")
    ik = WholeBodyIK(
        str(cfg.mapping["scene"]),
        WholeBodyIKConfig(dt=1.0 / ik_rate, solver=cfg.sim["solver"],
                          max_iters=10,
                          base_posture_cost=float(cfg.sim["base_posture_cost"])),
    )
    ik.init_from_keyframe("home")
    model, mj_data = ik.model, ik.data
    log(f"IK: {ik_rate} Hz, {ik.n_collision_pairs} collision pairs, "
        f"sides={'+'.join(sides)}")

    # The MJCF names hand joints exactly as wuji-description does, so the
    # published (20,) vector maps straight across with no reordering
    hand_adrs: dict[str, np.ndarray] = {}
    hand_lo: dict[str, np.ndarray] = {}
    hand_hi: dict[str, np.ndarray] = {}
    for side in sides:
        joints = [model.joint(n) for n in canonical_joint_names(side)]
        hand_adrs[side] = np.array([int(j.qposadr[0]) for j in joints])
        hand_lo[side] = np.array([float(j.range[0]) for j in joints])
        hand_hi[side] = np.array([float(j.range[1]) for j in joints])

    mocap_id = {side: int(model.body(f"{side}_ik_target").mocapid[0])
                for side in ("left", "right")}

    # The IK site is the arm's flange, ~3.7 cm behind the hand it carries. Rigid,
    # so one reading at home holds for every configuration.
    wrist_offset = {
        side: mj_data.site(f"{side}_arm_ee").xmat.reshape(3, 3).T
        @ (
            mj_data.body(f"{side}_wuji_hand_orient").xpos
            - mj_data.site(f"{side}_arm_ee").xpos
        )
        for side in ("left", "right")
    }

    # Frozen rotation is pinned to the home pose rather than to whatever the arm
    # was holding at engage, so re-engaging never quietly changes the wrist angle
    home_fk = ik.forward_kinematics()
    clutches = {
        side: Clutch(
            side,
            position_scale=cfg.mapping["position_scale"],
            follow_orientation=cfg.mapping["follow_orientation"],
            pin_rotation=home_fk[SIDE_INDEX[side]].rotation(),
            wrist_offset=wrist_offset[side],
            translation_frame=cfg.mapping["translation_frame"],
        )
        for side in sides
    }

    # No staleness gate here, unlike the RPC path: a stalled publisher in sim
    # just freezes the target, and nothing can be driven into anything.
    stream = AriaHandStream(cfg.publisher["host"], cfg.publisher["port"],
                            sides=sides, stale_s=None)
    stream.start()

    # Single-key boxes so viser callbacks hand work to the solve loop rather
    # than mutating MjData or the IK configuration off-thread
    gui_state = {"reset": False, "realign": False}

    # ── Viser + mjviser ─────────────────────────────────────────────────────
    server = viser.ViserServer(port=int(cfg.sim["viser_port"]), verbose=False)
    server.gui.configure_theme(dark_mode=False)
    # The frame every target is expressed in; worth seeing without hunting for it
    server.scene.world_axes.visible = True
    if cfg.sim["share"]:
        server.request_share_url()
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.create_visualization_gui()

    # mjviser keeps the tracked body centred by drawing the whole MuJoCo scene at
    # `world - base_link`, so anything added at raw world coordinates lands that
    # far off the robot — 9 cm in z at home, more once the base rolls. Overlays
    # hang off this frame, which carries the same offset, so what the debug
    # numbers say and what the picture shows agree.
    overlay = server.scene.add_frame("/overlay", show_axes=False)
    warned_no_offset = [False]

    def sync_overlay_offset() -> None:
        """Match the overlay root to mjviser's current scene offset."""
        off = getattr(scene, "_scene_offset", None)
        if off is None and not warned_no_offset[0]:
            # Private mjviser attribute. Degrading to zero silently would put
            # every overlay back where the bug had it, so say so once.
            warned_no_offset[0] = True
            log("mjviser has no _scene_offset -- overlays will be drawn at "
                "raw world coordinates and will not line up", style="yellow")
        pos = np.zeros(3) if off is None else np.asarray(off, dtype=np.float64)
        overlay.position = pos
        # Same reason: the world triad is a claim about where the origin is
        server.scene.world_axes.position = pos

    with server.gui.add_folder("Teleop"):
        gui_engaged = {side: server.gui.add_text(side, initial_value="paused",
                                                 disabled=True)
                       for side in sides}
        gui_show_operator_frame = server.gui.add_checkbox("Operator Hand Frame", True)
        gui_fix_base = server.gui.add_checkbox("Fix Base", ik.fix_base)
        gui_avoid_collisions = server.gui.add_checkbox("Collision Avoidance",
                                                       ik.avoid_collisions)
        btn_home = server.gui.add_button("Reset to Home")

        @gui_fix_base.on_update
        def _(_e) -> None:
            """Lock the mobile base so only arms and lift move."""
            ik.toggle_fix_base(gui_fix_base.value)

        @gui_avoid_collisions.on_update
        def _(_e) -> None:
            """Toggle the solver's hard self- and ground-collision constraints."""
            ik.toggle_collision_avoidance(gui_avoid_collisions.value)

        @btn_home.on_click
        def _(_e) -> None:
            """Ask the solve loop to snap back to the home keyframe."""
            gui_state["reset"] = True

    with server.gui.add_folder("Mapping"):
        gui_scale = server.gui.add_slider("Position scale", min=0.25, max=2.0,
                                          step=0.05,
                                          initial_value=cfg.mapping["position_scale"])
        gui_follow = server.gui.add_checkbox("Follow Orientation",
                                             cfg.mapping["follow_orientation"])
        gui_frame = server.gui.add_dropdown(
            "Translation Frame", Clutch.TRANSLATION_FRAMES,
            initial_value=cfg.mapping["translation_frame"])

        @gui_scale.on_update
        def _(_e) -> None:
            """Retune travel per metre; re-anchor so the arm does not jump."""
            for c in clutches.values():
                c.set_alignment(position_scale=gui_scale.value)
            gui_state["realign"] = True

        @gui_follow.on_update
        def _(_e) -> None:
            """Toggle wrist-orientation following; re-anchor for the same reason."""
            for c in clutches.values():
                c.set_alignment(follow_orientation=gui_follow.value)
            gui_state["realign"] = True

        @gui_frame.on_update
        def _(_e) -> None:
            """Swap which frame translation is read in; the map is fixed at engage."""
            for c in clutches.values():
                c.set_alignment(translation_frame=gui_frame.value)
            gui_state["realign"] = True

    # The operator's hand frame, mapped by the same clutch that drives the arm.
    # It should sit on the `{side}_ik_target` triad, which rides the robot's wrist:
    # same origin, same axes. Any standing gap is IK tracking, not the mapping.
    drawn_frames: set[str] = set()

    def draw_operator_frame(side: str, T_odom_wrist: np.ndarray | None) -> None:
        """Draw (or clear) the mapped operator hand triad for one side."""
        name = f"/overlay/operator_hand_{side}"
        frame = None
        if gui_show_operator_frame.value and T_odom_wrist is not None:
            frame = clutches[side].operator_frame(T_odom_wrist)
        if frame is None:
            if side in drawn_frames:
                server.scene.remove_by_name(name)
                drawn_frames.discard(side)
            return
        server.scene.add_frame(
            name,
            wxyz=frame.rotation().wxyz,
            position=frame.translation(),
            # Longer and thinner than the mocap body's own 0.12 m / 0.008 m
            # capsules, so when the two coincide it reads as a needle poking out
            # of each thick axis rather than as one triad hiding the other
            axes_length=0.20,
            axes_radius=0.004,
        )
        drawn_frames.add(side)

    # ── Solve loop ──────────────────────────────────────────────────────────
    rate = RateLimiter(ik_rate, warn=False)
    targets = dict(zip(("left", "right"), ik.forward_kinematics()))
    hand_cmd = {s: mj_data.qpos[hand_adrs[s]].copy() for s in sides}
    was_engaged = {s: False for s in sides}
    home_watch = (HomeSeqWatcher()
                  if cfg.home["gesture"] and len(sides) == 2 else None)
    if home_watch is None:
        log(f"home gesture off: it needs both hands (hand={'+'.join(sides)})",
            style="yellow")
    travel: dict[str, np.ndarray | None] = {s: None for s in sides}
    tick = 0
    last_dbg = 0.0

    log(f"viser on http://localhost:{cfg.sim['viser_port']}")
    banner = f"YORv3 sim - Aria hands ({'+'.join(sides)})"
    try:
        with Live(_tracking_table([], banner), console=console,
                  refresh_per_second=2) as live:
            while True:
                if gui_state["reset"]:
                    gui_state["reset"] = False
                    ik.init_from_keyframe("home")
                    targets = dict(zip(("left", "right"), ik.forward_kinematics()))
                    for side in sides:
                        clutches[side].release()
                        was_engaged[side] = False

                snap = stream.snapshot()
                realign = gui_state["realign"]
                gui_state["realign"] = False

                # The publisher detected both thumbs up on two released
                # hands. There is no RPC here, so it lands on the same keyframe
                # reset the GUI button drives -- this node owns the model
                if (home_watch is not None
                        and home_watch.update(stream.home_seq())
                        and not any(clutches[s].engaged for s in sides)):
                    log("both thumbs up -> home arms", style="yellow")
                    gui_state["reset"] = True

                for side in sides:
                    clutch, s = clutches[side], snap[side]
                    # Engaged means: the publisher isn't paused and we have a wrist
                    # to follow. Anchoring is deferred until both hold, so engaging
                    # with the hand out of view doesn't latch a stale pose.
                    want = not s.paused and s.T_odom_wrist is not None
                    if want and (not clutch.engaged or realign):
                        clutch.engage(s.T_odom_wrist,
                                      ik.forward_kinematics()[SIDE_INDEX[side]])
                    elif not want and clutch.engaged:
                        clutch.release()
                    if clutch.engaged != was_engaged[side]:
                        was_engaged[side] = clutch.engaged
                        log(f"{side} arm: "
                            f"{'ENGAGED' if clutch.engaged else 'released'}",
                            style="green" if clutch.engaged else "yellow")
                    travel[side] = None
                    if s.T_odom_wrist is not None:
                        target = clutch.target(s.T_odom_wrist)
                        if target is not None:
                            targets[side] = target
                            travel[side] = clutch.travel(s.T_odom_wrist)
                    # Shaka stops everything: the arm target freezes because the
                    # clutch is released, and the fingers freeze here rather than
                    # relying on the publisher to stop updating qpos while paused
                    if not s.paused and s.qpos is not None:
                        hand_cmd[side] = np.clip(s.qpos[:20], hand_lo[side],
                                                 hand_hi[side])

                # Feed the commanded finger angles back into the IK configuration: the
                # ground-avoidance limit measures distances off the finger meshes, so a
                # stale home-pose hand would let the fingertips clip through the floor
                q = ik.configuration.q.copy()
                for side in sides:
                    q[hand_adrs[side]] = hand_cmd[side]
                ik.update_configuration(q)

                result = ik.solve(targets["left"], targets["right"], lift_target=None)

                ik.apply_to_sim_kinematic(mj_data, result)
                for side in sides:
                    mj_data.qpos[hand_adrs[side]] = hand_cmd[side]
                    # Marker rides the wrist, not the flange, so it sits on the hand
                    R = targets[side].rotation()
                    mj_data.mocap_pos[mocap_id[side]] = (
                        targets[side].translation() + R.as_matrix() @ wrist_offset[side]
                    )
                    mj_data.mocap_quat[mocap_id[side]] = R.wxyz
                mujoco.mj_forward(model, mj_data)

                # 1 Hz readout of the two points that are supposed to be the same
                # place: the `{side}_ik_target` triad (the mocap marker, riding the
                # commanded wrist) and the mapped hand-tracking wrist landmark the
                # overlay is drawn from. Both in robot world, metres. `d` is the
                # mapping error only — how far the arm then lags its target is
                # `pos_err`. `--` while released: no mapping to evaluate.
                #
                # Still 1 Hz, not the render rate: these are numbers to read, and a
                # table that changes 30 times a second cannot be read.
                now = time.monotonic()
                if now - last_dbg >= 1.0:
                    last_dbg = now
                    rows = []
                    for side in sides:
                        ik_t = np.asarray(mj_data.mocap_pos[mocap_id[side]], float)
                        rows.append({
                            "side": side,
                            "engaged": clutches[side].engaged,
                            "ik_target": fmt_xyz(ik_t),
                            "travel": fmt_xyz(travel[side]),
                            "pos_err": (result.left_pos_err if side == "left"
                                        else result.right_pos_err) * 1e3,
                            "ori_err": (result.left_ori_err if side == "left"
                                        else result.right_ori_err) * 1e3,
                        })
                    live.update(Group(_tracking_table(rows, banner),
                                      _mapping_table(rows, result)))

                tick += 1
                if tick % RENDER_EVERY == 0:
                    # Offset first: mjviser recomputes it from this mj_data, and the
                    # overlays drawn below have to ride the value it just used
                    scene.update_from_mjdata(mj_data)
                    sync_overlay_offset()
                    for side in sides:
                        gui_engaged[side].value = (
                            "ENGAGED" if clutches[side].engaged else "paused")
                        draw_operator_frame(side, snap[side].T_odom_wrist)
                rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        server.stop()
        log("stopped.", style="yellow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
