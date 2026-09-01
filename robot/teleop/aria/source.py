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

Engagement is the publisher's shaka toggle, sent as `paused`. Engaging pins the
operator's wrist frame to the robot's; everything after that is a delta from
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
        home_gesture: act on the publisher's two-hand thumbs-up home. Needs
            hand="both"; a single-hand session has no way to make the gesture,
            and this is the local veto on one that can. The dwell itself is a
            publisher setting (`stream_pub --home-dwell-s`).
    """

    def __init__(self, host: str, port: int = 5555, hand: str = "both",
                 position_scale: float = 1.0, follow_orientation: bool = True,
                 clutch_reseed: bool = True, stale_s: float | None = 0.5,
                 hold_lift: bool = True, scene_xml: str | None = None,
                 home_gesture: bool = True,
                 translation_frame: str = "world", stats: bool = True,
                 clock_port: int = 5556):
        self._sides = ("left", "right") if hand == "both" else (hand,)
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

    @classmethod
    def from_config(cls, cfg: AriaConfig) -> AriaSource:
        """Build from config/aria_teleop.yaml — the way main() constructs one."""
        return cls(
            host=cfg.publisher["host"], port=cfg.publisher["port"],
            hand=cfg.mapping["hand"],
            position_scale=cfg.mapping["position_scale"],
            follow_orientation=cfg.mapping["follow_orientation"],
            translation_frame=cfg.mapping["translation_frame"],
            clutch_reseed=cfg.clutch["reseed"],
            stale_s=cfg.publisher["stale_s"] or None,
            hold_lift=cfg.clutch["hold_lift"],
            scene_xml=cfg.mapping["scene"],
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
        for side in self._sides:
            clutch, s = self._clutches[side], snap[side]
            # Engaged means: the publisher isn't paused and we have a wrist to
            # follow. Deferring the anchor until both hold is what stops a hand
            # held out of view from latching a stale pose.
            #
            # Unlike the Quest backend this is a level, not a button edge: the
            # publisher's own toggle debounces with a shaka dwell (0.5 s) and comes
            # up paused. A subscriber restarted while the publisher is running
            # therefore engages on the first packet -- harmless, because the
            # reseed below anchors on the robot's real pose, so zero delta is
            # zero motion.
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
            self._status[side] = SideStatus(
                "ENGAGED" if clutch.engaged else "paused")
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

        Deliberately no WholeBodyIK here: after init_from_keyframe("home") its
        forward kinematics is this same qpos, and skipping the solver keeps the
        RPC client down to mujoco + mink + numpy + commlink.
        """
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(self._scene_xml))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        offset, home_rot = {}, {}
        for side in self._sides:
            R_ee = data.site(f"{side}_arm_ee").xmat.reshape(3, 3)
            offset[side] = R_ee.T @ (
                data.body(f"{side}_wuji_hand_orient").xpos
                - data.site(f"{side}_arm_ee").xpos
            )
            home_rot[side] = mink.SO3.from_matrix(R_ee)
        return offset, home_rot
