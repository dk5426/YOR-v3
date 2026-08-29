"""source.py — Project Aria hand tracking as a whole-body teleop backend.

`AriaSource` is the `--input aria` backend of robot/teleop/wholebody_teleop.py:
each wrist pose commands one arm's end-effector, and the whole-body solver on
the other end of the RPC decides how much of the reach the base, lift and arm
each take.

Arms only. The publisher also sends 20 retargeted finger angles per hand, but
the RPC surface carries a single open/close gripper value, so nothing is done
with them here — robot/teleop/aria/sim_viz.py renders them in full against its
own model.

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
to home one arm. Read off the published landmarks here rather than in the
publisher, so the wire is unchanged and the gesture lives with the code that
knows what home means. See gesture.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import mink
import numpy as np

from robot.teleop.aria.clutch import Clutch
from robot.teleop.aria.config import AriaConfig
from robot.teleop.aria.gesture import HomeGesture
from robot.teleop.aria.stream import AriaHandStream
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
        home_gesture: enable the two-hand thumbs-up home. Needs hand="both";
            a single-hand session has no way to make the gesture.
        home_dwell_s: how long both thumbs must be held before it fires.
    """

    def __init__(self, host: str, port: int = 5555, hand: str = "both",
                 position_scale: float = 1.0, follow_orientation: bool = True,
                 clutch_reseed: bool = True, stale_s: float | None = 0.5,
                 hold_lift: bool = True, scene_xml: str | None = None,
                 home_gesture: bool = True, home_dwell_s: float = 1.0,
                 translation_frame: str = "world"):
        self._sides = ("left", "right") if hand == "both" else (hand,)
        self._position_scale = float(position_scale)
        self._follow_orientation = bool(follow_orientation)
        self._translation_frame = Clutch._checked_frame(str(translation_frame))
        self._clutch_reseed = bool(clutch_reseed)
        self._hold_lift = bool(hold_lift)
        self._lift_pinned = False
        self._scene_xml = Path(scene_xml) if scene_xml else DEFAULT_SCENE
        self._stream = AriaHandStream(host, port, sides=self._sides,
                                      stale_s=stale_s)
        self._clutches: dict[str, Clutch] = {}
        self._home = (HomeGesture(self._sides, home_dwell_s)
                      if home_gesture else None)
        # Wall clock, not accumulated dt: the loop is handed a nominal 1/rate,
        # so a slow tick would stretch the dwell past what the config says.
        # Swappable so the gesture tests can run a dwell without sleeping
        self._clock = time.monotonic

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
            home_dwell_s=cfg.home["dwell_s"],
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
        print(f"[aria] sides={'+'.join(self._sides)} "
              f"scale={self._position_scale:.2f} "
              f"follow_orientation={self._follow_orientation} "
              f"translation={self._translation_frame}")
        if self._home is not None and not self._home.available:
            print("[aria] home gesture off: it needs both hands "
                  f"(hand={'+'.join(self._sides)})")
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()

    def update(self, state: TeleopState, dt: float) -> TeleopCommand:
        cmd = TeleopCommand()
        # Once, on the first tick: claim the lift so the solver does not. See
        # the module docstring for why silence is not the same as holding.
        if self._hold_lift and not self._lift_pinned:
            self._lift_pinned = True
            cmd.lift_target = float(state.lift_target)
            print(f"\n[aria] lift pinned at {cmd.lift_target:.3f} m")
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
                print(f"\n[aria] {side} arm: ENGAGED")
            elif not want and clutch.engaged:
                clutch.release()
                print(f"\n[aria] {side} arm: released")
            if s.T_odom_wrist is None:
                continue
            target = clutch.target(s.T_odom_wrist)
            if target is not None:
                setattr(cmd, f"{side}_target", target)
        self._maybe_home(cmd, snap)
        return cmd

    def _maybe_home(self, cmd: TeleopCommand, snap: dict) -> None:
        """Both thumbs up, both hands disengaged -> the node's home_arms."""
        if self._home is None:
            return
        fired = self._home.update(
            kp={side: snap[side].kp_odom for side in self._sides},
            released={side: not self._clutches[side].engaged
                      for side in self._sides},
            now=self._clock(),
        )
        if not fired:
            return
        # home_arms is the node's own sequence -- base lock, lift to 450 mm,
        # then both arms. home_left_arm / home_right_arm run that same
        # preamble for one arm, which is why no gesture asks for them
        cmd.home_arms = True
        print("\n[aria] both thumbs up -> home arms")

    def _engage_pose(self, side: str, state: TeleopState) -> mink.SE3:
        """Where to anchor the clutch: the robot's actual EE, or the local target."""
        if self._clutch_reseed and self.state_refresh is not None:
            srv = self.state_refresh()
            key = f"{side}_ee_wxyz_xyz"
            if srv and srv.get(key) is not None:
                return mink.SE3(np.array(srv[key]))
            print(f"\n[aria] {side} engage reseed failed -- using local target")
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
