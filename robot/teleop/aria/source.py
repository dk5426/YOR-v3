"""source.py — Project Aria hand tracking as a whole-body teleop backend.

`AriaSource` is the `--input aria` backend of robot/teleop/wholebody_teleop.py:
each wrist pose commands one arm's end-effector, and the whole-body solver on
the other end of the RPC decides how much of the reach the base, lift and arm
each take.

Arms only, on purpose. The publisher also sends 20 retargeted finger angles per
hand, and nothing is done with them here: they are read off the same `wuji`
topic by `Hands`, which lives inside yor.py / yor_mujoco.py and subscribes on a
thread of its own. Two independent paths off one publisher, so finger targets
never queue behind arm targets on either node's single RPC socket -- a ZMQ REP
socket serves one caller at a time. See robot/hand/hands.py.
robot/teleop/aria/sim_viz.py renders the fingers in-process, without either.

The lift is pinned once, on the first tick, to the height the server reports and
never touched again. That single command is deliberate: both nodes start with
`lift_target = None`, which the solver reads as "the lift is yours", so a client
that simply never mentions the lift does not leave it where it is -- it hands
the column to the solver, which then drives it to help the arms reach. Pinning
once costs one RPC and makes "Aria moves the arms" true of the hardware as
well. Pass `hold_lift=False` for the free-lift behaviour, which is what
sim_viz.py runs.

Engagement is the recording station's (Thor) `clutch_cmd`, latched: engage
takes both arms, disengage releases them, and nothing else engages anything.
The publisher's shaka is an emergency stop and only that -- it stops both arms
and latches, and only the station's next engage clears it. The shaka is read as
a *gesture*, not as the `paused` level it toggles: the publisher comes up
paused, so the level says nothing about intent, and it is the change that means
stop. A publisher that goes quiet trips the same stop through `stale_s`.

With no station in the session (`--cmd-host none`) the latch is on for good and
the shaka goes back to being the clutch, read as a level -- how this backend
behaved before Thor had a say. Engaging pins
the operator's wrist frame to the robot's; everything after that is a delta from
that anchor -- rotation read in the wrist frame, translation in the world with
only its heading taken from engage, so up stays up (see clutch.py, which
carries the reasoning, and `mapping.translation_frame` for the older
fully wrist-framed behaviour).

Homing is the one thing besides the arm targets this sends: *both* thumbs up
with *both* hands disengaged runs the node's `home_arms` sequence. There is no
single-arm variant -- that sequence locks the base and drives the lift to
450 mm whichever arms it was asked for, so one thumb would move the whole robot
to home one arm.

Detecting it is the publisher's job, because the landmarks it needs are the
publisher's and there is no reason to ship 21 points per hand across a wireless
link so the robot can measure a thumb. What arrives here is `home_seq`, a count
of completed gestures; this module watches for it to go up. Both the dwell and
the released-hands gate live upstream, so a client that never sees the counter
move can never home -- and because it is a total rather than an edge, a dropped
packet costs nothing.
"""

from __future__ import annotations

from pathlib import Path

import mink
import numpy as np

from robot.teleop.aria.clutch import Clutch
from robot.teleop.aria.config import AriaConfig
from robot.teleop.aria.stats import ClockSync, StreamStats
from robot.teleop.aria.stream import AriaHandStream, HomeSeqWatcher
from robot.teleop.status import SideStatus, SourceStatus, StreamRow, log
from robot.teleop.wholebody_teleop import InputSource, TeleopCommand, TeleopState

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_SCENE = _REPO / "description" / "scene_wholebody.xml"


class AriaSource(InputSource):
    """Clutch-based 6-DoF teleop from Aria hand tracking.

    Args:
        host, port: where the aria2robot publisher is.
        hand: "left", "right" or "both" — the idle arm is never commanded.
        position_scale: robot EE travel per metre of wrist travel.
        follow_orientation: rotate the EE with the wrist. Off pins it to the
            model's home orientation and teleoperates translation alone.
        translation_frame: "world" keeps up meaning up and takes only the
            heading from engage; "wrist" reads translation in the engage
            orientation too. Rotation is wrist-framed either way. See clutch.py.
        clutch_reseed: on engage, anchor to the robot's actual EE pose (one
            get_state RPC) rather than the client's local target, so wind-up
            banked while streaming into a constraint does not carry over.
        stale_s: release if the publisher goes quiet this long (0/None off).
        hold_lift: pin the lift to its current height on the first tick, so the
            solver cannot claim it. Off leaves the lift a free DOF.
        scene_xml: MJCF the flange->wrist offset and home orientation come from.
        hand_type: "wuji"/"aero"/None -- which hand (if any) that scene has
            mounted, so the flange->wrist offset can be read off the right
            body. None/"none" means no hand: the offset is zero, mapping the
            operator's hand straight onto the bare flange.
        home_gesture: act on the publisher's two-hand thumbs-up home. Needs
            hand="both"; a single-hand session has no way to make the gesture,
            and this is the local veto on one that can. The dwell itself is a
            publisher setting (`stream_pub --home-dwell-s`).
    """

    def __init__(self, host: str, port: int = 5555, hand: str = "both",
                 position_scale: float = 1.0, follow_orientation: bool = True,
                 clutch_reseed: bool = True, stale_s: float | None = 0.5,
                 hold_lift: bool = True, scene_xml: str | None = None,
                 hand_type: str | None = None,
                 home_gesture: bool = True,
                 translation_frame: str = "world", stats: bool = True,
                 clock_port: int = 5556,
                 cmd_sub: object | None = None):
        self._sides = ("left", "right") if hand == "both" else (hand,)
        self._hand_type = str(hand_type or "none").lower()
        self._position_scale = float(position_scale)
        self._follow_orientation = bool(follow_orientation)
        self._translation_frame = Clutch._checked_frame(str(translation_frame))
        self._clutch_reseed = bool(clutch_reseed)
        self._hold_lift = bool(hold_lift)
        self._lift_pinned = False
        self._scene_xml = Path(scene_xml) if scene_xml else DEFAULT_SCENE
        # Measured on the client's own subscription -- not the node's, which
        # reads the same publisher for the fingers on a link of its own.
        self._stats = (StreamStats(AriaHandStream.TOPICS) if stats else None)
        self._clock_sync = (ClockSync(host, int(clock_port), self._stats)
                            if self._stats is not None and clock_port else None)
        self._stream = AriaHandStream(host, port, sides=self._sides,
                                      stale_s=stale_s, stats=self._stats)
        self._clutches: dict[str, Clutch] = {}
        # Last tick's per-side row for the client's status table. Written by
        # update() rather than rebuilt on demand, so what the table shows is
        # the sample that was acted on and not a fresher one taken since.
        self._status: dict[str, SideStatus] = {}
        # Two hands or nothing, same rule the publisher applies: homing is one
        # indivisible sequence on the robot, so one thumb must not reach it.
        self._home_wanted = bool(home_gesture)
        self._home = (HomeSeqWatcher()
                      if home_gesture and len(self._sides) == 2 else None)
        self._warned_no_home = False
        # Engage/disengage from the recording station (Thor), latched: it is
        # the authority, and the shaka is an emergency stop ANDed against it
        # (see update()). A background thread owns the blocking commlink
        # get(); update() only reads the latest value -- a control loop must
        # never sit on a socket.
        #
        # No recording station this session means latched on for good and the
        # shaka read as a level -- the shaka-only clutch this backend had
        # before Thor had a say, which `_shaka_is_clutch` selects.
        self._cmd_sub = cmd_sub
        self._shaka_is_clutch = cmd_sub is None
        self._thor_engaged = cmd_sub is None
        self._last_ext: bool | None = None
        # `cmd_sub` is a ClutchCmdWatcher: it owns the socket, the thread and
        # the recovery, so nothing here ever waits on the station. Its
        # `engaged` goes False when the feed goes stale, which arrives below
        # as an ordinary disengage -- the station going quiet releases the
        # arms rather than leaving them latched to a value nobody is refreshing.
        # The emergency stop: latched per side, cleared only by a fresh engage
        # from the station. `_prev_paused` is what turns the publisher's shaka
        # *toggle* into a gesture edge -- see update().
        self._stopped: dict[str, bool] = {s: False for s in self._sides}
        self._prev_paused: dict[str, bool | None] = {
            s: None for s in self._sides}

    @classmethod
    def from_config(cls, cfg: AriaConfig, cmd_sub: object | None = None) -> AriaSource:
        """Build from config/aria_teleop.yaml — the way main() constructs one."""
        return cls(
            cmd_sub=cmd_sub,
            host=cfg.publisher["host"], port=cfg.publisher["port"],
            hand=cfg.mapping["hand"],
            position_scale=cfg.mapping["position_scale"],
            follow_orientation=cfg.mapping["follow_orientation"],
            translation_frame=cfg.mapping["translation_frame"],
            clutch_reseed=cfg.clutch["reseed"],
            stale_s=cfg.publisher["stale_s"] or None,
            hold_lift=cfg.clutch["hold_lift"],
            scene_xml=str(cfg.scene_path()),
            hand_type=cfg.hand["type"],
            home_gesture=cfg.home["gesture"],
            stats=cfg.publisher["stats"],
            clock_port=cfg.publisher["clock_port"],
        )

    def start(self) -> None:
        offset, home_rot = self._model_anchors()
        # Frozen rotation is pinned to the home pose rather than to whatever the
        # arm was holding at engage, so re-engaging never quietly changes the
        # wrist angle.
        self._clutches = {
            side: Clutch(
                side,
                position_scale=self._position_scale,
                follow_orientation=self._follow_orientation,
                pin_rotation=home_rot[side],
                wrist_offset=offset[side],
                translation_frame=self._translation_frame,
            )
            for side in self._sides
        }
        log(f"sides={'+'.join(self._sides)} "
            f"scale={self._position_scale:.2f} "
            f"follow_orientation={self._follow_orientation} "
            f"translation={self._translation_frame}", prefix="aria")
        if self._home_wanted and self._home is None:
            log("home gesture off: it needs both hands "
                f"(hand={'+'.join(self._sides)})", style="yellow", prefix="aria")
        self._stream.start()
        if self._cmd_sub is not None:
            self._cmd_sub.start()
        # Best-effort and off-thread: the first handshake retries for several
        # seconds against a publisher that has no clock socket, and the arms
        # are waiting on start(). Latency reads '--' until it lands.
        if self._clock_sync is not None:
            self._clock_sync.start(on_sync=self._log_clock)

    @staticmethod
    def _log_clock(sample: tuple[float, float] | None) -> None:
        """Report the first handshake, from the clock thread."""
        if sample is None:
            log("clock handshake failed -- stream latency will read '--'",
                style="yellow", prefix="aria")
        else:
            log(f"clock offset {sample[0] * 1e3:+.2f} ms "
                f"(rtt {sample[1] * 1e3:.2f} ms)", prefix="aria")

    def stop(self) -> None:
        if self._cmd_sub is not None:
            self._cmd_sub.stop()
        if self._clock_sync is not None:
            self._clock_sync.stop()
        self._stream.stop()

    def update(self, state: TeleopState, dt: float) -> TeleopCommand:
        cmd = TeleopCommand()
        # Once, on the first tick: claim the lift so the solver does not. See
        # the module docstring for why silence is not the same as holding.
        if self._hold_lift and not self._lift_pinned:
            self._lift_pinned = True
            cmd.lift_target = float(state.lift_target)
            log(f"lift pinned at {cmd.lift_target:.3f} m", prefix="aria")
        snap = self._stream.snapshot()
        ext = self._poll_ext_cmd()
        if ext is not None:
            self._thor_engaged = ext
            if ext:
                # Engage doubles as the stop reset: the station is the only
                # thing that clears a shaka, which is what makes it a stop
                # rather than a pause.
                for s_ in self._sides:
                    self._stopped[s_] = False
            log(f"thor: {'engage' if ext else 'disengage'}",
                style="green" if ext else "yellow", prefix="aria")
        # Any change to `paused` is a stop, whichever way it lands. It is a
        # toggle the operator flips by shaka, and nothing else here reads its
        # level -- the publisher keeps streaming poses while paused, so a
        # paused arm can be engaged and moving. Trip on the rising edge only
        # and stopping would take one shaka or two depending on a state the
        # operator cannot see; on the change, one shaka always stops. One hand
        # stops both: an emergency stop that left the other arm running would
        # not be one. A publisher that goes quiet trips it too, since
        # `stale_s` reports silence as paused.
        if not self._shaka_is_clutch:
            shaka = False
            for side in self._sides:
                prev, now = self._prev_paused[side], snap[side].paused
                self._prev_paused[side] = now
                shaka = shaka or (prev is not None and now != prev)
            if shaka and self._thor_engaged and not all(
                    self._stopped[s_] for s_ in self._sides):
                for s_ in self._sides:
                    self._stopped[s_] = True
                log("STOP (shaka) -- station must re-engage",
                    style="red", prefix="aria")

        for side in self._sides:
            clutch, s = self._clutches[side], snap[side]
            # The switch. The station engages and disengages; the shaka only
            # ever stops. Deferring the anchor until a wrist is actually
            # tracked is what stops a hand held out of view from latching a
            # stale pose -- an arm whose hand is not in view engages as soon
            # as it is, without the station saying anything again.
            want = (self._thor_engaged and not self._stopped[side]
                    and s.T_odom_wrist is not None)
            if self._shaka_is_clutch:
                # No station: the shaka is the clutch, read as a level, the
                # way this backend behaved before Thor had a say.
                want = not s.paused and s.T_odom_wrist is not None
            if want and not clutch.engaged:
                clutch.engage(s.T_odom_wrist, self._engage_pose(side, state))
                log(f"{side} arm: ENGAGED", style="green", prefix="aria")
            elif not want and clutch.engaged:
                clutch.release()
                log(f"{side} arm: released", style="yellow", prefix="aria")
            if s.T_odom_wrist is None:
                # Distinct from "paused": the publisher is talking, the hand
                # is just not in view, and no shaka will fix it.
                self._status[side] = SideStatus("no track")
                continue
            target = clutch.target(s.T_odom_wrist)
            if target is not None:
                setattr(cmd, f"{side}_target", target)
            if clutch.engaged:
                state_str = "ENGAGED"
            elif self._shaka_is_clutch:
                state_str = "paused"
            elif self._stopped[side]:
                state_str = "STOP"
            else:
                state_str = "standby"
            self._status[side] = SideStatus(state_str)
        self._maybe_home(cmd, snap)
        return cmd

    def status(self) -> SourceStatus:
        """Engagement off the tick `update()` just ran, plus the link's stats.

        Engagement is cached rather than re-snapshotted, so the table
        describes the sample that was actually acted on. A side the session
        does not run is absent, which the display renders as dashes.
        """
        streams: tuple[StreamRow, ...] = ()
        if self._stats is not None:
            snap = self._stats.snapshot()
            streams = tuple(
                StreamRow(t, *snap[t]) for t in self._stats.topics)
        return SourceStatus(sides=dict(self._status), streams=streams)

    def _poll_ext_cmd(self):
        """The station's engagement, on the ticks it changes -- else None.

        Edge-triggered so a republished value costs nothing; `update()` holds
        the level it reports."""
        if self._cmd_sub is None:
            return None
        want = bool(self._cmd_sub.engaged)
        if want == self._last_ext:
            return None
        self._last_ext = want
        return want

    def _maybe_home(self, cmd: TeleopCommand, snap: dict) -> None:
        """The publisher's home counter went up -> the node's home_arms."""
        if self._home is None:
            return
        self._check_publisher_can_home()
        if not self._home.update(self._stream.home_seq()):
            return
        # Belt and braces. The publisher already required both sides paused for
        # a full dwell, and the loop above has released their clutches by now,
        # so this is free -- but "nothing is following either hand" is the whole
        # safety argument for homing without a confirmation, and it is worth
        # asserting locally rather than trusting a remote definition of paused.
        if any(self._clutches[s].engaged for s in self._sides):
            log("ignoring home: a hand is still engaged", style="yellow",
                prefix="aria")
            return
        # home_arms is the node's own sequence -- base lock, lift to 450 mm,
        # then both arms. home_left_arm / home_right_arm run that same
        # preamble for one arm, which is why no gesture asks for them
        cmd.home_arms = True
        log("both thumbs up -> home arms", style="yellow", prefix="aria")

    def _check_publisher_can_home(self) -> None:
        """Say so once if the publisher physically cannot send the gesture.

        A `--hand left` publisher never increments the counter, so a two-handed
        client would otherwise wait for a home that can never arrive.
        """
        if self._warned_no_home:
            return
        meta = self._stream.meta()
        if meta is None:
            return
        self._warned_no_home = True
        if meta.get("home") is False:
            log(f"publisher runs hand={'+'.join(meta.get('sides') or ['?'])}; "
                "the home gesture needs both -- homing is off this session",
                style="yellow", prefix="aria")

    def _engage_pose(self, side: str, state: TeleopState) -> mink.SE3:
        """Where to anchor the clutch: the robot's actual EE, or the local target."""
        if self._clutch_reseed and self.state_refresh is not None:
            srv = self.state_refresh()
            key = f"{side}_ee_wxyz_xyz"
            if srv and srv.get(key) is not None:
                return mink.SE3(np.array(srv[key]))
            log(f"{side} engage reseed failed -- using local target",
                style="yellow", prefix="aria")
        return getattr(state, f"{side}_target")

    def _model_anchors(self) -> tuple[dict[str, np.ndarray], dict[str, mink.SO3]]:
        """Read the flange->wrist offset and the home EE orientation off the MJCF.

        The IK site is the arm's flange, ~3.7 cm behind the hand it carries.
        Rigid, so one reading at home holds for every configuration -- but it is
        applied through a rotation, so a value that disagrees with the server's
        model does not become a constant bias, it becomes an arc the hand swings
        through as the wrist turns. Reading it from the same description the
        server loads is the only way it cannot drift.

        With no hand mounted (or a scene/hand.type mismatch -- the mount body
        this hand.type names is simply absent), the offset is zero: the
        operator's hand maps straight onto the bare flange.

        Deliberately no WholeBodyIK here: after init_from_keyframe("home") its
        forward kinematics is this same qpos, and skipping the solver keeps the
        RPC client down to mujoco + mink + numpy + commlink.
        """
        import mujoco

        from robot.hand.hands import hand_mount_body

        model = mujoco.MjModel.from_xml_path(str(self._scene_xml))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        offset, home_rot = {}, {}
        for side in self._sides:
            R_ee = data.site(f"{side}_arm_ee").xmat.reshape(3, 3)
            ee_pos = data.site(f"{side}_arm_ee").xpos
            mount_name = hand_mount_body(self._hand_type, side)
            mount_pos = ee_pos
            if mount_name is not None:
                try:
                    mount_pos = data.body(mount_name).xpos
                except KeyError:
                    pass
            offset[side] = R_ee.T @ (mount_pos - ee_pos)
            home_rot[side] = mink.SO3.from_matrix(R_ee)
        return offset, home_rot
