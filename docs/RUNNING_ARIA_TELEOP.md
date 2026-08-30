# Running Aria Hand Teleop

Project Aria glasses drive both YORv3 arms by hand tracking: shaka to engage,
both thumbs up to home. This is the **subscriber** side only — the publisher
lives in the **aria2robot** repo, which owns the Aria SDK, the hand tracking and
the finger retargeting. Nothing here imports it.

The two gestures are owned by different repos, and "paused" means different
things at each hop — §4 is the one section to read before trusting the shaka as
a stop.

General whole-body teleop, the other input backends, and everything about the
robot itself are in [docs/RUNNING.md](../../../docs/RUNNING.md); the
architecture is in [CLAUDE.md](../../../CLAUDE.md).

> **Env:** run in **`aria2robot`**, not `aria2` — `aria2` is on numpy ≥ 2.3,
> where `WholeBodyIK.__init__` raises at `wholebody_ik.py:795` (`int()` on a
> size-1 `qposadr`). Only the publisher runs in `aria2`.

---

## 1. Two ways to run it

### A. `sim_viz.py` — one process, browser, fingers

```bash
# 1. publisher, from ~/nyu/aria2robot
python -m aria2robot.stream_pub --wifi

# 2. here
python robot/teleop/aria/sim_viz.py --pub-host <publisher-ip>
# -> http://localhost:8080
```

Subscribes, runs YOR's own whole-body IK over `description/scene_wholebody.xml`
and renders through mjviser. No RPC hop, no `mjpython`, no `yor_mujoco.py`. It
is the **only view that renders the 20 finger joints**, and the only one where
you can see the operator triad against the target — use it to validate a frame
change before touching hardware.

Three flags: `--config`, `--pub-host`, `--hand`. Everything else is YAML.

### B. `--input aria` — the RPC path (sim or robot)

Three processes, in this order:

```bash
# 1. publisher — on the machine the glasses stream to, from ~/nyu/aria2robot
python -m aria2robot.stream_pub --wifi

# 2. the whole-body server
mjpython robot/yor_mujoco.py      # sim,   :8081
python robot/yor.py               # robot, :5557 (on the robot)

# 3. the client — same machine as step 1, so publisher.host defaults to localhost
python robot/teleop/wholebody_teleop.py --input aria --target sim      # sim
python robot/teleop/wholebody_teleop.py --input aria --host <robot-ip> # robot

# 3'. the client somewhere else — on the robot, say, with the publisher remote
python robot/teleop/wholebody_teleop.py --input aria \
    --host localhost --pub-host <publisher-ip>
```

`--target hw` is the default, so the robot form is the shorter one. Nothing on
the robot changes between Aria and Quest — the backend is entirely client-side.

**`--pub-host` is not `--host`.** The first is where step 1 runs; the second is
the robot's RPC server. They are usually different machines. `--pub-host`
overrides `publisher.host` for one run and nothing else, so the config file
stays the description of a *session* rather than of a network.

Which machine should run the client? Prefer the **robot**. The RPC client is a
blocking `zmq.REQ` (`commlink/rpc_client.py:25`), so every tick is a round trip
and WiFi stalls the 30 Hz loop directly; the hand stream is one-way PUB/SUB that
already drains to the newest sample and has a staleness gate, so it is the link
that should cross the network. On a good LAN either placement works.

Arms only: this backend never commands the base, and touches the lift exactly
once (§5).

---

## 2. Configuration

Two flags: `--aria-config` (`--config` in `sim_viz.py`) picks the file, and
`--pub-host` overrides `publisher.host` for one run. Everything else is
[config/aria_teleop.yaml](../../../config/aria_teleop.yaml), which is commented
in full and shared by both entry points. **Edit the YAML, not the command
line** — `--pub-host` is the exception because it is the one setting that
follows the machine you happen to be on rather than the session you are
running. The startup line echoes what it resolved, so a typo is visible
immediately:

```
[aria] config aria_teleop.yaml: 10.21.63.99:5555 hand=both scale=1.0 …
```

| Key | Default | Effect |
| ----- | --------- | -------- |
| `publisher.host` / `.port` | `localhost` / `5555` | where the publisher is; `--pub-host` overrides the host per run |
| `publisher.stale_s` | `0.5` | release if the publisher goes quiet this long; `0` disables |
| `mapping.hand` | `both` | the idle arm is never commanded |
| `mapping.position_scale` | `1.0` | robot EE travel per metre of wrist travel |
| `mapping.follow_orientation` | `true` | `false` pins the EE to the model's home orientation — the way to check the position mapping alone |
| `mapping.translation_frame` | `world` | which frame hand *translation* is read in. `world` keeps up meaning up; `wrist` is the older behaviour where translation rides the engage orientation too. Rotation is wrist-framed either way — see §3 |
| `mapping.scene` | `description/scene_wholebody.xml` | where the flange→wrist offset and pinned home orientation are read from |
| `clutch.reseed` | `true` | engage anchors on the robot's actual EE, not the local target |
| `clutch.hold_lift` | `true` | claim the lift on the first tick — see §5 |
| `home.gesture` | `true` | both thumbs up, both hands *disengaged*, homes both arms |
| `home.dwell_s` | `1.0` | how long both thumbs must be held |
| `sim.*` | — | `sim_viz.py` only: solve rate, base posture cost, QP solver, viser port, share |

`mapping.scene` must be the same robot the server is running. That offset is
applied *through a rotation*, so a value that disagrees is not a constant bias —
it is an arc the hand swings through.

---

## 3. Gestures

### Engage — shaka

Thumb + pinky out, three middle fingers curled, held ~1 s. Both sides start
disengaged, so the robot never lurches on startup. The same gesture disengages,
and disengaging freezes the arm target **and** the fingers — the *target*, not
the arm; see §4.

Engaging pins your wrist frame to the robot's. Everything after is a delta from
that anchor, and **rotation and translation are read in different frames** — the
same split Quest makes:

| | Frame | What that buys |
| --- | --- | --- |
| **Rotation** | your wrist, at engage | turning your hand about one of its own axes turns the robot's hand about the matching one, whatever either was doing at engage |
| **Translation** | the world, heading from engage | **up is up** — raising your hand raises the EE at every engage pose |

The translation map is `Rz(ψ)`: a pure turn about the robot's vertical, so your
vertical passes straight through. Only ψ comes from engage, read off the same
wrist map rotation uses. That is what absorbs the arbitrary yaw origin of Aria's
odometry frame, and it is why there is still **no yaw or room calibration
anywhere in this backend** — engaging *is* the alignment. (Quest needs one,
`--oculus-yaw-correction`, because its tracking frame outlives the clutch.)

Engage pose still matters, but less, and differently: it sets your *heading*, so
engage roughly facing the way you want "forward" to mean. It no longer decides
whether up is up.

One assumption underneath all of that: odom's +Z is your true vertical. Aria's
VIO odometry frame is gravity-aligned, so it is. That axis is `ODOM_UP` in
[clutch.py](clutch.py), beside the wrist tables, and is deliberately not a
config key.

`mapping.translation_frame: wrist` restores the old behaviour — translation
rides the engage orientation too, and raising your hand raises the EE only if
you engaged with your hand roughly in the robot's hand pose. It is kept for
comparison; `sim_viz` has a **Translation Frame** dropdown under *Mapping* so
the two can be swapped live (it re-anchors, so the arm does not jump).

### Home — both thumbs up, on two released hands

Hold a thumbs up on **both** hands for `home.dwell_s`, with **both** hands
disengaged. That runs the node's `home_arms` sequence — base lock, lift to
450 mm, then both arms' joints. In `sim_viz` it lands on the same keyframe
reset the **Reset to Home** button drives, since that node owns the model and
there is no RPC.

**There is no single-arm gesture.** `home_left_arm` / `home_right_arm` are not
cheaper: every variant runs the same preamble, so homing one arm locks the
base, tears down the whole-body controller and drives the lift to 450 mm
exactly as homing both does. Two hands makes that a deliberate request rather
than something one thumb can trip. A one-handed session (`mapping.hand: left`
or `right`) therefore has no home gesture at all, and says so at startup.

One thumb does nothing, and dropping either thumb mid-hold cancels. The dwell
runs from the **second** thumb, so a staggered pair still commits both hands
for the full time.

The disengaged gate is the whole safety argument, and it is why the shaka stays
symmetric: **disengage is the stop gesture on both hands**, which is what a
startled operator reaches for. Shaka off on both hands, then thumbs up. A thumb
on a hand that is still following you is ignored. Re-engaging afterwards is
zero-delta, because clutch reseed anchors on wherever home left the arms.

It fires **once per hold** — keeping the thumbs up does not re-home. Set
`home.gesture: false` to turn it off.

Thumbs up cannot be confused with the shaka: that one wants the pinky extended,
this one curled. Detection is client-side, off landmarks the publisher keeps
sending while paused, so no publisher change is involved — §4 for why that
works and where the line between the two repos falls.

---

## 4. Pause: who detects it, what crosses the wire, what the robot knows

Three separate questions, and they have three different answers. Most surprises
in this backend come from collapsing them.

### Who detects which gesture

| Gesture | Detected in | On the wire | Effect |
| --- | --- | --- | --- |
| **shaka** | the **publisher** (aria2robot `utils/gesture.py`, latched by `PauseToggle`, `stream_pub --shaka-dwell-s`, default 0.5 s) | a per-side `paused` bool | publisher stops retargeting fingers; this client releases that arm's clutch |
| **thumbs up** | **here** (`gesture.py`), read off the published `kp_mp` landmarks | *nothing* — no field, no publisher change | `home_arms` |

The asymmetry is deliberate. The shaka gates the publisher's own work, so it
needs latched, debounced state with exactly one owner; this repo never looks for
a shaka, it reads `paused`. Home is defined by what the *robot* does — the
`home_arms` sequence and its lift preamble — so it lives with the code that
knows what that means, and detecting it client-side keeps the wire unchanged.

Both read the same landmarks, and `gesture.py`'s bend measure is deliberately
the same one aria2robot uses for the shaka: two detectors disagreeing about
"curled" is a bug that only shows up on somebody's hand. The two poses are
mutually exclusive by construction — the shaka wants the pinky extended, thumbs
up wants it curled.

### What the publisher still sends while paused

Pausing skips **one call**, `retarget()`. Everything else in the payload keeps
being computed and published:

| Field | While paused |
| --- | --- |
| `kp_mp`, `kp_mp_scaled` | **live** — recomputed every tick |
| `T_device_hand` | **live** — derived from the landmarks (`mano_wrist_frame`), not from the retargeted joints |
| `qpos` (20 finger angles) | **frozen, not dropped** — the last retargeted value, republished unchanged |
| `paused` | `true` |

So it is not "human hands but no wuji hands": the wuji pose is still on the
wire, just stale. Two consequences:

- **`qpos` is `None` until the first unpause.** The publisher comes up paused
  and has nothing to republish yet, so `sim_viz` renders fingers at the model
  default until you first engage.
- **`T_device_hand` staying live is what makes engaging zero-delta.** The clutch
  anchors on a frame that tracked your hand the entire time it was released. Had
  it been frozen alongside `qpos`, engaging would anchor on a stale pose and the
  arm would jump.

Landmarks arriving while paused are also what let the home gesture work at all —
that is the mechanism behind "thumbs up on a *disengaged* hand" in §3.

Freshness overrides both: if a hand has not been seen for the publisher's
`HAND_STALE_S`, every field goes `None` and only `paused` survives, which is why
`stream.py` treats a missing `T_device_hand` as "stay disengaged" rather than as
a pause.

### What "paused" means on the robot: nothing

There is **no pause RPC**, on either node. The whole surface is `get_state`, the
`set_*_target` calls, the homes and the two toggles — frozen by
`tests/test_interface_contract.py`. The chain stops entirely on the client:

```
publisher `paused` → SideSample.paused → want = False → clutch.release()
  → clutch.target() returns None → cmd.*_target stays None
  → _dispatch makes zero RPC calls
```

With both sides released the client simply goes quiet. From the robot's side
that is indistinguishable from an operator standing still, or from the network
dying.

> **The shaka freezes the target, not the robot.** `set_*_ee_target` sets a
> *standing* goal that the whole-body controller keeps solving toward at 30 Hz —
> it is not a stream that decays. Disengage mid-reach, or while streaming into a
> constraint, and the arm keeps closing on the last goal you sent. What you get
> is "no new commands", not a stop. `home_arms` and the **physical e-stop** are
> the things that actually halt motion — the same rule the hardware tests run
> under ([tests/hardware/README.md](../../../tests/hardware/README.md)).

`publisher.stale_s` (0.5 s) is the client's own watchdog: nothing received for
that long and `snapshot()` rewrites every side to `paused=True`, releasing the
clutch. commlink hands back the last payload forever, so without it a dead
publisher would leave the arm following a pose you can no longer release by
gesture. It, too, only stops new targets.

---

## 5. About the lift

Both the hardware and sim nodes start with *no* lift target, which the solver
reads as "the lift is yours". A client that simply never mentions the lift does
not leave the column where it is — it hands it to the solver.
`clutch.hold_lift` sends one `set_lift_target` on the first tick to claim it
back, and prints `[aria] lift pinned at …`.

It is a **preference, not a lock**: the lift target is a soft posture term
(`lift_posture_cost`, `1e-4` on the sim node), so the solver still trades the
column away to help the arms reach. Measured drift over a 12 s reachable-motion
run: 0.200 → 0.115 m with the pin, versus 0.200 → 0.000 without. Holding it hard
would need a `fix_lift` RPC, which neither node exposes today.

---

## 6. Reading the screen (`sim_viz`)

Two things are worth knowing how to read:

- **The `ik_target` sphere must sit inside the WUJI palm**, not on the thin
  red/green/blue `{side}_arm_ee_axis_*` capsules 37.5 mm behind it. The marker
  rides the wrist, not the site the IK targets. The `d` column in the 1 Hz
  console line is the numeric version and is **structurally zero** — anything
  above a few mm is a bug, not tracking error.
- **The mapped operator triad** (long thin needles, 0.20 m × 0.004 m) must sit
  coincident *and parallel* with that sphere's own thick 0.12 m × 0.008 m
  capsules. Mirrored or 90°-rotated means an axis table in
  [`clutch.py`](clutch.py) is wrong. A gap with matching axes is IK tracking
  error, which the `pos_err` column names separately (`--` while released:
  nothing is mapped).

The 1 Hz line's own frames: `ik_target`, `hand_wrist` and `travel` are all
robot **world** metres (`travel` is your hand's displacement since engage,
already scaled), while `gap` is in robot **wrist** axes so a nonzero value names
the axis to correct.

To prove the overlays are not silently offset — mjviser recentres its scene on
`base_link`, and a 9 cm shift reads exactly like tracking error — uncheck **Fix
Base** and reach far enough that the chassis rolls. The hand skeleton must stay
welded to the sphere as the base moves.

---

## 7. Checks that need no glasses

```bash
python tests/test_aria_mapping.py
```

115 checks: the config loader, both axis tables against literals, the
convention rotation against this repo's MJCF hand mount, the clutch delta math
in **both** translation frames (that a metre up is a metre up at every engage
pose, that horizontal stays horizontal, that the heading still comes from
engage, that rotation and the engage anchor are identical either way, and that
a heading-less engage pose falls back instead of failing), the rigid flange
offset, the wire decode, the staleness gate, the home gesture
(detection, the disengaged gate, that one thumb never homes, that a staggered
pair dwells from the second thumb, and that held thumbs home exactly once), and
both rendering fixes — the marker one measured as 0.0000 mm on the hand versus
37.5 mm at the flange.

---

## 8. Troubleshooting

| Symptom | Cause / Fix |
| --- | --- |
| `publisher sends T_device_wrist but no T_device_hand` | The publisher is old. `T_device_wrist` is Aria's own frame — different origin, different convention. That side stays disengaged until `stream_pub` is updated. |
| Nothing arrives; no `[aria] subscribing …` follow-through | `publisher.host` points at the wrong machine, or step 1 is not running. It is *not* `--host` — pass `--pub-host` and check the config line it echoes at startup. |
| Arm releases on its own mid-motion | `publisher.stale_s` fired — the publisher went quiet. commlink hands back the last payload forever, so without this gate a dead publisher leaves the arm holding a target you cannot release by gesture. |
| Arm keeps moving after you shaka off | Working as designed, and the thing to internalise: disengaging stops *new* targets, and the last one is a standing goal the controller keeps solving toward. See §4. |
| Fingers sit at the model default until you first engage | The publisher republishes the last `qpos` while paused and has none before the first unpause. §4. |
| Robot ignores the shaka entirely | It never sees it. There is no pause RPC; the client just stops sending. If the arm is still tracking you, the *client* is still engaged — check its console for `released`. |
| `WholeBodyIK.__init__` raises at `wholebody_ik.py:795` | Wrong conda env. Use `aria2robot`, not `aria2` (numpy ≥ 2.3). |
| Raising your hand does not raise the EE | Under the default `translation_frame: world` this should not happen — check the config is not set to `wrist`, in which case the delta rides your engage orientation and you must engage with your hand roughly in the robot's hand pose. |
| "Forward" points somewhere odd | Heading comes from the engage pose. Disengage, face the way you want forward to mean, re-engage. |
| `engage pose has no heading; translation keeps odom's axes` | You engaged in a pose whose horizontal mapping is a flip, not a turn — no yaw describes it. Harmless, but re-engage in a normal pose to get a heading. |
| Target sphere floats behind the hand | The 37.5 mm flange offset regressed. It is read off the model at startup and must never be hardcoded. `tests/test_aria_mapping.py` pins it. |
| Operator triad mirrored or 90° off | An axis table in `clutch.py` (`MANO_WRIST_AXES` / `YOR_WRIST_AXES`). Edit them there and nowhere else. |
| Homing repeats while the thumb is up | `HoldTrigger.latch()` regressed to `reset()`. Pinned by "does not repeat while held". |
| Lift drifts down under load | Expected — `hold_lift` is a soft preference, see §5. |
