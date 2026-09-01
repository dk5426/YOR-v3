"""Wrist-pose clutch: freeze on release, follow the pose delta since engage.

The instant a side engages, the operator's wrist frame is identified with the
robot's wrist frame; from then on the robot is commanded that anchor plus the
operator's motion since engage, re-expressed through the fixed MANO->YOR
convention rotation. Nothing is calibrated at runtime — engaging *is* the
alignment.

Rotation and translation are mapped **separately**, the same split
`OculusSource._ee_target` makes:

- **Rotation** is a delta in the wrist frame, `R_d = C·(R_engage⁻¹·R_now)·Cᵀ`,
  applied in the end-effector's own frame. Turning your hand about one of its
  anatomical axes turns the robot's hand about the matching one, whatever
  either was doing at engage.
- **Translation** is a delta in the **world**, with only its heading taken from
  engage: the odom displacement is mapped by `Rz(ψ)`, ψ being the yaw of the
  full wrist map about the robot's vertical. Up stays up — raising your hand
  raises the end-effector however you were holding it when you engaged — while
  the arbitrary yaw of Aria's odometry frame is still resolved by the engage
  pose. So there is still no room calibration to get right; Quest needs one
  (`--oculus-yaw-correction`) only because its tracking frame outlives the
  clutch.

`translation_frame="wrist"` restores the older, fully wrist-framed action space,
where translation rides the engage orientation too. That one made "raise your
hand" mean "raise the end-effector" only if you engaged with your hand roughly
in the robot's hand pose — the price the decomposition above removes while
keeping the orientation mapping that made it worth paying.

The world mode rests on one assumption, and only one: that odom's `ODOM_UP` is
the operator's true vertical. Aria's VIO odometry frame is gravity-aligned, so
it is — but a publisher that ever changes that makes "up" mean something else,
which is why the axis is named here and overridable from the config.

The operator's frame is the WUJI hand root — the frame the operator's hand URDF
is drawn at, published as `T_odom_hand` — not Aria's own `transform_device_wrist`.
That choice is what makes the mapping exact rather than measured: the robot wears
the same WUJI hand, so both ends of the mapping are the same physical frame on the
same hand model, and the operator's wrist landmark lands on the robot's hand base
by construction. Both sides carry the left/right mirror on the *thumb* axis, which
is why the convention rotation comes out identical on left and right.

Engagement is the shaka toggle the publisher already sends as `paused`; the same
gesture that freezes finger retargeting freezes the arm.
"""


import math

import mink
import numpy as np

X, Y, Z = np.eye(3)

# Robot world up: the lift's axis, and the axis the base yaws about.
ROBOT_UP = Z
# Odom up. Aria's VIO odometry frame is gravity-aligned with +Z up, so the only
# unknown between it and the robot is a yaw -- which engaging resolves. Every
# gravity claim `translation_frame="world"` makes rests on this vector.
ODOM_UP = Z

# The WUJI hand root, i.e. MANO axes: what `mano_wrist_frame` builds and what
# the operator's hand URDF is anchored on. Derived by running the real
# `estimate_frame_from_hand_points` on hands of known anatomy, not by eye
MANO_WRIST_AXES = {
    "left": {"middle": Z, "palm": X, "thumb": -Y},
    "right": {"middle": Z, "palm": X, "thumb": Y},
}
# YOR's, i.e. the `*_arm_ee` site frame the IK already targets. Measured off the
# MJCF at the home keyframe, not taken on faith
YOR_WRIST_AXES = {
    "left": {"middle": Y, "palm": X, "thumb": Z},
    "right": {"middle": Y, "palm": X, "thumb": -Z},
}


def convention_matrix(side: str) -> np.ndarray:
    """R_yor_mano: one anatomical direction, re-expressed in YOR wrist axes.

    Comes out equal to the MJCF's own `arm_ee -> wuji_hand_orient` rotation on
    both sides, which is the check that the two tables describe the same robot;
    `test_convention_matrix_matches_the_mjcf_hand_mount` pins it there.
    """

    def basis(axes: dict) -> np.ndarray:
        return np.column_stack([axes["middle"], axes["palm"], axes["thumb"]])

    return basis(YOR_WRIST_AXES[side]) @ basis(MANO_WRIST_AXES[side]).T


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation of `angle` about a unit `axis`."""
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _rotation_onto(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Shortest rotation carrying unit `u` onto unit `v`."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    axis = np.cross(u, v)
    sin_a = float(np.linalg.norm(axis))
    if sin_a < 1e-12:
        if float(u @ v) > 0.0:
            return np.eye(3)
        # Antiparallel: the axis is any perpendicular, so pick the stablest one
        perp = np.eye(3)[int(np.argmin(np.abs(u)))]
        axis = np.cross(u, perp)
        return _rodrigues(axis / np.linalg.norm(axis), math.pi)
    return _rodrigues(axis / sin_a, math.atan2(sin_a, float(u @ v)))


def _yaw_about_up(A: np.ndarray) -> float | None:
    """Yaw of `A` about +Z: the rotation about +Z closest to it.

    A 2x2 Procrustes rather than an Euler extraction, so it degrades gracefully
    as the tilt grows instead of only being right for a near-upright `A`.
    `None` when `A` maps both horizontal axes onto the vertical, where the yaw
    it implies is genuinely undefined.
    """
    cos_p, sin_p = A[0, 0] + A[1, 1], A[1, 0] - A[0, 1]
    if math.hypot(cos_p, sin_p) < 1e-6:
        return None
    return math.atan2(sin_p, cos_p)


class Clutch:
    """Per-side wrist clutch producing end-effector targets from wrist motion.

    Args:
        side: Which hand — picks the MANO->YOR convention rotation.
        position_scale: Robot EE travel per metre of wrist travel.
        follow_orientation: Rotate the EE with the wrist. Off means frozen.
        translation_frame: "world" maps hand displacement in the world with
            only its heading taken from engage, so up stays up. "wrist" maps
            it in the operator's wrist axes at engage, the older behaviour.
            Rotation is wrist-framed either way.
        odom_up: The operator's vertical, in odom. Only `translation_frame=
            "world"` reads it, and everything it claims about gravity is this
            vector being right.
        pin_rotation: Orientation to hold while frozen. `None` falls back to
            whatever the EE held at engage, which drifts between engagements;
            passing a fixed pose makes "frozen" mean the same pose every time.
        wrist_offset: Where the robot's wrist sits in end-effector coordinates.
            The IK frame is the arm's flange, a few cm behind the hand it
            carries; without this the operator's hand maps onto the flange
            instead of the hand, and turning the wrist swings the hand through
            an arc rather than pivoting in place. Zero means the two coincide.

    `wrist_offset` is the only lever arm here, and it is rigid — read off the
    model. The operator side needs none: `T_odom_hand`'s origin already *is*
    the wrist landmark the hand is drawn from.
    """

    TRANSLATION_FRAMES = ("world", "wrist")

    def __init__(
        self,
        side: str,
        position_scale: float = 1.0,
        follow_orientation: bool = True,
        pin_rotation: mink.SO3 | None = None,
        wrist_offset: np.ndarray | None = None,
        translation_frame: str = "world",
        odom_up: np.ndarray | None = None,
    ) -> None:
        self.side = side
        self.position_scale = position_scale
        self.follow_orientation = follow_orientation
        self.pin_rotation = pin_rotation
        self.translation_frame = self._checked_frame(translation_frame)
        self._C = convention_matrix(side)
        self._offset = (
            np.zeros(3) if wrist_offset is None else np.asarray(wrist_offset, float)
        )
        # Odom axes -> robot axes, up onto up. Identity while both are +Z
        self._G = _rotation_onto(ODOM_UP if odom_up is None else odom_up, ROBOT_UP)
        self._R_engage: np.ndarray | None = None
        self._t_engage: np.ndarray | None = None
        self._ee_engage: mink.SE3 | None = None
        self._wrist_engage: np.ndarray | None = None
        self._A_t: np.ndarray | None = None

    @classmethod
    def _checked_frame(cls, frame: str) -> str:
        """Reject a typo in the config here, not by mapping motion sideways."""
        if frame not in cls.TRANSLATION_FRAMES:
            raise ValueError(
                f"translation_frame must be one of {cls.TRANSLATION_FRAMES}, "
                f"got {frame!r}"
            )
        return frame

    @property
    def engaged(self) -> bool:
        """True while the clutch is following wrist motion."""
        return self._ee_engage is not None

    def set_alignment(
        self,
        position_scale: float | None = None,
        follow_orientation: bool | None = None,
        translation_frame: str | None = None,
    ) -> None:
        """Retune the mapping live. Callers should re-engage so the arm doesn't jump.

        A new `translation_frame` takes effect at the next engage: the current
        one is baked into the map fixed at engage, and swapping it underneath a
        live delta is exactly the jump the re-engage is there to avoid.
        """
        if position_scale is not None:
            self.position_scale = position_scale
        if follow_orientation is not None:
            self.follow_orientation = follow_orientation
        if translation_frame is not None:
            self.translation_frame = self._checked_frame(translation_frame)

    def engage(self, T_odom_wrist: np.ndarray, T_world_ee: mink.SE3) -> None:
        """Anchor to the current hand pose and end-effector pose.

        `T_odom_wrist` is the operator's WUJI hand-root pose in odom — the
        published `T_odom_hand`, which the publisher composes, not Aria's own
        wrist frame.
        """
        T = np.asarray(T_odom_wrist, dtype=np.float64)
        self._R_engage = T[:3, :3].copy()
        self._t_engage = T[:3, 3].copy()
        self._ee_engage = T_world_ee.copy()
        self._wrist_engage = (
            T_world_ee.translation() + T_world_ee.rotation().as_matrix() @ self._offset
        )
        self._A_t = self._translation_map()

    def _translation_map(self) -> np.ndarray:
        """Odom displacement -> robot-world displacement, fixed at engage.

        `wrist` is the full wrist map, which carries the engage tilt with it.
        `world` keeps only its yaw about the robot's vertical, which is what
        leaves the vertical axis untouched: a metre up in odom is a metre up
        for the end-effector, at every engage pose.
        """
        A = self._ee_engage.rotation().as_matrix() @ self._C @ self._R_engage.T
        if self.translation_frame == "wrist":
            return A
        yaw = _yaw_about_up(A @ self._G.T)
        if yaw is None:
            # Engaged with the hand mapping both horizontal axes onto the
            # vertical -- pointing straight up, say. There is no heading to
            # read, so keep odom's own and say so rather than invent one
            print(f"[aria] {self.side}: engage pose has no heading; "
                  "translation keeps odom's axes")
            return self._G
        c, sn = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]) @ self._G

    def release(self) -> None:
        """Stop following; the caller keeps commanding the last target."""
        self._R_engage = None
        self._t_engage = None
        self._ee_engage = None
        self._wrist_engage = None
        self._A_t = None

    def travel(self, T_odom_wrist: np.ndarray) -> np.ndarray | None:
        """Where the hand went since engage, in robot world axes, scaled.

        World axes rather than wrist axes so it reads against everything else
        on the debug line, all of which is robot world.
        """
        delta = self._delta(T_odom_wrist)
        return None if delta is None else delta[1]

    def operator_frame(self, T_odom_wrist: np.ndarray) -> mink.SE3 | None:
        """The operator's hand frame itself, mapped into robot world.

        Its origin is the operator's wrist landmark by construction, so drawn
        against the robot's wrist triad it reads the mapping directly: the two
        are coincident, and any standing gap is IK tracking error rather than
        the mapping. Always carries the true mapped rotation, unlike
        `wrist_target`, which pins it when `follow_orientation` is off.
        """
        delta = self._delta(T_odom_wrist)
        if delta is None:
            return None
        R_d, t_d = delta
        R0 = self._ee_engage.rotation()
        return mink.SE3.from_rotation_and_translation(
            R0 @ mink.SO3.from_matrix(R_d), self._wrist_engage + t_d
        )

    def wrist_target(self, T_odom_wrist: np.ndarray) -> mink.SE3 | None:
        """Where the robot's wrist should go — the frame the operator's hand maps to."""
        frame = self.operator_frame(T_odom_wrist)
        if frame is None or self.follow_orientation:
            return frame
        R0 = self._ee_engage.rotation()
        return mink.SE3.from_rotation_and_translation(
            R0 if self.pin_rotation is None else self.pin_rotation, frame.translation()
        )

    def target(self, T_odom_wrist: np.ndarray) -> mink.SE3 | None:
        """End-effector target for the current wrist pose, or None if released.

        The wrist target backed off along the flange offset, so the rotation
        pivots about the wrist rather than dragging it around the flange.
        """
        wrist = self.wrist_target(T_odom_wrist)
        if wrist is None:
            return None
        return mink.SE3.from_rotation_and_translation(
            wrist.rotation(),
            wrist.translation() - wrist.rotation().as_matrix() @ self._offset,
        )

    def _delta(
        self, T_odom_wrist: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Wrist motion since engage: rotation in wrist axes, translation in world.

        The two halves are the decomposition the module docstring describes —
        `R_d` is applied in the end-effector's own frame by the caller, `t_d`
        is already a robot-world displacement.
        """
        if self._ee_engage is None:
            return None
        T = np.asarray(T_odom_wrist, dtype=np.float64)
        R_d = self._C @ (self._R_engage.T @ T[:3, :3]) @ self._C.T
        t_d = self._A_t @ (T[:3, 3] - self._t_engage)
        return R_d, t_d * self.position_scale
