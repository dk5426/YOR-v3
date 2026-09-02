# Running Aria Hand Teleop

Project Aria glasses drive both YORv3 arms and both WUJI hands by hand tracking:
shaka to engage, both thumbs up to home. This is the **subscriber** side only —
the publisher lives in the **aria2robot** repo, which owns the Aria SDK, the hand
tracking and the finger retargeting. Nothing here imports it.

Arms and fingers ride the same stream but are **two independent subscribers**,
and the fingers never touch the node's RPC socket — §5. Both gestures are now
detected in the publisher, but "paused" still means different things at each
hop, and something different again for the fingers than for the arms — §4 then
§5 are the two sections to read before trusting the shaka as a stop.

> **Upgrading:** this is wire 2. The payload no longer carries hand landmarks,
> and `T_odom_hand` arrives pre-composed. **Deploy this repo before the
> publisher.** A new client understands an old publisher and says so; an old
> client against a new publisher finds no field it recognises and simply stops
> moving the arms, with nothing on screen to explain it.

General whole-body teleop, the other input backends, and everything about the
robot itself are in [docs/RUNNING.md](RUNNING.md); the
architecture is in [CLAUDE.md](../CLAUDE.md).

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
and renders through mjviser. No RPC hop, no `mjpython`, no `yor_mujoco.py`, and
no `Hands` — it drives the 20 finger joints straight into the same model, so
it stays the **shortest** way to see them, and the only one where you can see the
operator triad against the target. Use it to validate a frame change before
touching hardware.

Three flags: `--config`, `--pub-host`, `--hand`. Everything else is YAML.

### B. `--input aria` — the RPC path (sim or robot)

Three processes. Order matters only for legibility — every link is latest-value,
so any of them can be restarted underneath the others:

```bash
# 1. publisher — on the machine the glasses stream to, from ~/nyu/aria2robot
python -m aria2robot.stream_pub --wifi

# 2. the whole-body server — the hands live in here too (§5)
mjpython robot/yor_mujoco.py --pub-host <publisher-ip>   # sim,   :8081
python robot/yor.py --pub-host <publisher-ip> \
    --hand-backend hardware                              # robot, :5557

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
[config/aria_teleop.yaml](../config/aria_teleop.yaml), which is commented
in full and shared by both entry points. **Edit the YAML, not the command
line** — `--pub-host` is the exception because it is the one setting that
follows the machine you happen to be on rather than the session you are
running. The startup line echoes what it resolved, so a typo is visible
immediately:

```
[aria] config aria_teleop.yaml: 10.21.63.99:5555 arms=both hands=left scale=1.0 …
```

| Key | Default | Effect |
| ----- | --------- | -------- |
| `publisher.host` / `.port` | `localhost` / `5555` | where the publisher is; `--pub-host` overrides the host per run |
| `publisher.stale_s` | `0.5` | release if the publisher goes quiet this long; `0` disables |
| `mapping.hand` | `both` | which **arms** are teleoped; the idle arm is never commanded. Not the hands — see `hand.sides` |
| `mapping.position_scale` | `1.0` | robot EE travel per metre of wrist travel |
| `mapping.follow_orientation` | `true` | `false` pins the EE to the model's home orientation — the way to check the position mapping alone |
| `mapping.translation_frame` | `world` | which frame hand *translation* is read in. `world` keeps up meaning up; `wrist` is the older behaviour where translation rides the engage orientation too. Rotation is wrist-framed either way — see §3 |
| `mapping.scene` | `description/scene_wholebody.xml` | where the flange→wrist offset and pinned home orientation are read from |
| `clutch.reseed` | `true` | engage anchors on the robot's actual EE, not the local target |
| `clutch.hold_lift` | `true` | claim the lift on the first tick — see §6 |
| `home.gesture` | `true` | act on the publisher's two-thumbs home, which homes both arms **and opens both hands**. The dwell is `stream_pub --home-dwell-s`; this is the local veto |
| `sim.*` | — | `sim_viz.py` only: solve rate, base posture cost, QP solver, viser port, share |
| `hand.backend` | `none` | `none` commands nothing; the simulator still moves its fingers, since it reads the targets in-process. `hardware` drives real hands through `wujihandpy` (`yor.py` only) |
| `hand.sides` | `both` | which **hands** are driven: `both`, `left`, `right`, `none`. Independent of `mapping.hand` — both arms stay teleoped either way. `--hands` overrides it per run. See §5 |
| `hand.serial.left` / `.right` | `""` | which physical hand is which — **required for two hands**, see §5 |
| `hand.rpc_port` | `5558` | a socket of the hands' own, separate from the node's — see §5. `0` disables it |
| `hand.rate_hz` | `100` | hand loop rate; it sends on change only |
| `hand.ramp_s` | `1.5` | how long the *first* command to each hand takes to arrive — see §5 |
| `hand.lowpass_hz` | `5.0` | cutoff of wujihandpy's own controller-side filter |

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
[clutch.py](../robot/teleop/aria/clutch.py), beside the wrist tables, and is deliberately not a
config key.

`mapping.translation_frame: wrist` restores the old behaviour — translation
rides the engage orientation too, and raising your hand raises the EE only if
you engaged with your hand roughly in the robot's hand pose. It is kept for
comparison; `sim_viz` has a **Translation Frame** dropdown under *Mapping* so
the two can be swapped live (it re-anchors, so the arm does not jump).

### Home — both thumbs up, on two released hands

Hold a thumbs up on **both** hands for the publisher's `--home-dwell-s`
(default 1.0 s), with **both** hands disengaged. That runs the node's `home_arms` sequence — **both hands to zero**,
base lock, lift to 450 mm, then both arms' joints. The hands go first, so
anything still held falls from where it is rather than from 450 mm up; see §5.
In `sim_viz` it lands on the same keyframe reset the **Reset to Home** button
drives, since that node owns the model and there is no RPC.

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
`home.gesture: false` here to ignore it, or run the publisher `--hand left`/
`right` so it is never detected in the first place.

Thumbs up cannot be confused with the shaka: that one wants the pinky extended,
this one curled. Detection is in the **publisher**, which already holds the
landmarks; what reaches the robot is `home_seq`, a running count of completed
gestures. A dropped packet therefore cannot lose a home, and a jump of several
still homes exactly once — §4 for where the line between the two repos falls.

---

## 4. Pause: who detects it, what crosses the wire, what the robot knows

Three separate questions, and they have three different answers. Most surprises
in this backend come from collapsing them.

### Who detects which gesture

| Gesture | Detected in | On the wire | Effect |
| --- | --- | --- | --- |
| **shaka** | the **publisher** (aria2robot `utils/gesture.py`, latched by `PauseToggle`, `stream_pub --shaka-dwell-s`, default 0.5 s) | a per-side `paused` bool | publisher stops retargeting fingers; this client releases that arm's clutch |
| **thumbs up** | the **publisher** (same module, latched by `HoldTrigger`, `stream_pub --home-dwell-s`, default 1.0 s) | `home_seq`, an envelope-level count of completed gestures | `home_arms` |

Both live upstream now, for the same reason: the landmarks are the publisher's,
and shipping 21 points per hand across a wireless link so the robot could
measure a thumb was most of the payload. They share `_CHAIN` and one bend
measure, deliberately — two detectors disagreeing about "curled" is a bug that
only shows up on somebody's hand. The two poses are mutually exclusive by
construction: the shaka wants the pinky extended, thumbs up wants it curled.

They use **different latches**, which is the one thing not to unify.
`PauseToggle` flips a state it owns; `HoldTrigger` reports an event the
subscriber acts on. Home also crosses as a **counter, not a flag**: PUB/SUB
drops packets and this subscriber conflates them, so an edge can be missed and
a total cannot. `HomeSeqWatcher` fires on an increase, adopts whatever value is
there when it connects (joining a publisher that has already homed must not
home), and treats a decrease as a publisher restart.

The **released gate** is enforced twice. The publisher requires both sides
`paused` for the full dwell, which is a strict subset of what this client calls
engaged; and `AriaSource` re-checks its own clutches before acting, because
"nothing is following either hand" is the entire safety argument for homing
without a confirmation and is worth asserting locally.

### What the publisher still sends while paused

Pausing skips **one call**, `retarget()`. Everything else in the payload keeps
being computed and published:

| Field | While paused |
| --- | --- |
| `T_odom_hand` | **live** — derived from the landmarks (`mano_wrist_frame`) composed with VIO, not from the retargeted joints |
| `qpos` (20 finger angles) | **frozen, not dropped** — the last retargeted value, republished unchanged |
| `paused` | `true` |
| `home_seq` | live — a paused hand is exactly the one allowed to ask for home |

So it is not "human hands but no wuji hands": the wuji pose is still on the
wire, just stale. Two consequences:

- **`qpos` is `None` until the first unpause.** The publisher comes up paused
  and has nothing to republish yet, so `sim_viz` renders fingers at the model
  default until you first engage.
- **`T_odom_hand` staying live is what makes engaging zero-delta.** The clutch
  anchors on a frame that tracked your hand the entire time it was released. Had
  it been frozen alongside `qpos`, engaging would anchor on a stale pose and the
  arm would jump.

The publisher still computing landmarks while paused is what lets the home
gesture work at all — that is the mechanism behind "thumbs up on a *disengaged*
hand" in §3. They just no longer cross the wire to do it.

Freshness overrides both: if a hand has not been seen for the publisher's
`HAND_STALE_S`, every field goes `None` and only `paused` survives, which is why
`stream.py` treats a missing `T_odom_hand` as "stay disengaged" rather than as
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
> under ([tests/hardware/README.md](../tests/hardware/README.md)).

`publisher.stale_s` (0.5 s) is the client's own watchdog: nothing received for
that long and `snapshot()` rewrites every side to `paused=True`, releasing the
clutch. commlink hands back the last payload forever, so without it a dead
publisher would leave the arm following a pose you can no longer release by
gesture. It, too, only stops new targets.

---

## 5. The hands: a second subscriber, inside the node

The arms and the fingers come off the **same** `wuji` payload, but they are read
by two different subscribers that share nothing else:

```
aria2robot stream_pub ──PUB "wuji"──┬──▶ wholebody_teleop.py --input aria
   (publisher host)                  │      ──RPC :5557/:8081──▶ arms, lift, base
                                     │
                                     └──▶ Hands, inside yor.py / yor_mujoco.py
                                            ├─ RPC :5558   other clients (own socket)
                                            ├─ wujihandpy ──▶ the real hands
                                            └─ targets() ──▶ the simulator's data.qpos
```

The hands are part of the robot node — started with it, stopped with it, and
reported in its `get_state()` as `left_hand_qpos` / `right_hand_qpos`, so one
call is a snapshot of the whole robot at one instant.

But they are deliberately **not on the node's RPC surface**. commlink's
`RPCServer` is a single ZMQ `REP` socket, and a REP socket serves one request at
a time — `threaded=True` only moves that loop onto a thread, it does not serve
two callers at once. A 20-float finger target at 100 Hz sent there would queue
against the 30 Hz arm targets, and the arms would pay for the hands. So `Hands`
subscribes to the publisher itself, on a thread of its own, and its RPC surface
for non-Aria clients gets a socket of its own on `hand.rpc_port` (5558).

`sim_viz.py` needs none of this: it renders the fingers in-process, off its own
subscription (§1A).

### Running it

```bash
mjpython robot/yor_mujoco.py --pub-host <ip>                 # sim; backend is
                                                             # always "none"
python robot/yor.py --pub-host <ip> --hand-backend hardware   # real hands
python robot/yor.py --pub-host <ip>                           # dry: nothing driven
python robot/yor.py --pub-host <ip> --hand-backend hardware \
    --hands right                                             # one hand, both arms
mjpython robot/yor_mujoco.py --no-hands                       # no subscription
python robot/yor.py --no-hands                                # arms only, as before
```

Four flags, identical on both nodes: `--no-hands`, `--hands`, `--aria-config`,
`--pub-host`. `yor.py` adds `--hand-backend {none,hardware}` and
`--tracking-csv`. The simulator is pinned to backend `none` — it renders
fingers, it never drives a USB device — so `hand.backend: hardware` in the YAML
cannot reach out of a sim run.

To exercise the whole path without putting the glasses on, call
`set_hand_target` / `set_bimanual_hand_target` / `open_hands` on port 5558.

### One hand, or none

`mapping.hand` is the **arms** and `hand.sides` is the **hands**, and they are
deliberately not the same setting. Whole-body IK is one QP over both arms and
wants both wrist targets, so the arms stay a pair; the fingers are the separate
path drawn above, and how many WUJI hands are plugged into a given robot is a
property of that robot on that day.

```yaml
hand:
  sides: right      # both | left | right | none (default: both)
```

`--hands both|left|right|none` overrides it for one run, on either node — the
same reason `--pub-host` is a flag: it tracks the machine, not the session.
`--hands none` is exactly `--no-hands`, and the node says so:

```
[wuji] no hands this session; arms only
[wuji] hands=right backend=hardware aria=on rpc=5558 rate=100 Hz
```

A side left out is never subscribed to, never driven, and never opened by a
home — `open_hands` filters to the sides being served, so the two-thumbs home
still homes both arms and simply has one fewer hand to open. Its arm is
teleoped exactly as before: the clutch, the targets and the status table are
untouched. Nothing else changes, on either node or on the wire; the publisher
can keep sending both hands' `qpos` and the unserved one is dropped on arrival.

**You do not have to set it just because a hand is unplugged.** With both
serials in the config, `HardwareWujiDriver.start()` opens each side
independently and a side that does not answer is dropped with a line, not
raised:

```
[wuji] left hand did not open (no such device); continuing without it
[wuji] right hand open (serial B)
[wuji] serving right, not left+right
```

`Hands` then narrows to what actually opened, so nothing subscribes to,
reports or sends at a device that is not there. Only *no* hand opening is an
error. Opening by serial is unambiguous and the blank-serial refusal runs
first, so a silent side is absent rather than mistaken for its twin — which is
why this can be tolerant without risking a swap. Set `hand.sides` when you
want a plugged-in hand *not* driven; leave it at `both` when you simply have
one plugged in.

Two consequences worth knowing:

- **One hand needs no serial.** The refusal in §5's checklist is about *two*
  hands sharing a USB bus. With `hand.sides: right` a blank `hand.serial.right`
  takes whichever hand enumerated first, which is the right one when it is the
  only one plugged in. Leaving both serials filled in is harmless — the unused
  one is ignored.
- **`sim_viz` reads the same key.** Its `--hand` flag is still the arms; the
  fingers of an unserved side hold the model's home pose while its arm follows
  you. There is no `--hands` flag there, to keep the two names apart.

Note `none` and the default `both` are the two ends of the key — there is no
"unset" state to reason about. A config file written before this key gets
`both`, and since the result is intersected with `mapping.hand`, a one-armed
session still gets exactly its own hand.

### The shaka means something different here

§4 is about the arms: a released clutch stops *new* targets and the last one
stands as a goal the controller keeps solving toward. For the fingers the same
shaka does something simpler and stronger:

| | Arms | Hands |
| --- | --- | --- |
| shaka off | clutch releases; no new targets; the arm keeps closing on the last goal | the last finger pose is **held**, exactly, and nothing moves |
| publisher goes quiet | `publisher.stale_s` releases the clutch | **nothing** — the grasp is held indefinitely |
| before the first engage | no targets sent | hands never touched; they sit at the model's home pose |

The absence of a staleness gate on the hand path is deliberate, not an
oversight. A closed hand is the last thing the operator asked for; opening it on
its own because a link went quiet would drop whatever it is holding. The arm
client's `publisher.stale_s` still releases the *arms*, so the robot stops
moving while the hand keeps its grip.

This falls out of the wire described in §4 with no extra machinery: the
publisher freezes `qpos` while paused and sends `None` before the first unpause,
so "hold the last usable command" already covers shaka, lost tracking, a dead
publisher and startup.

### Homing opens the hands

Both thumbs up on two released hands homes the arms (§3) — and now sends both
hands to zero as part of it. The open happens **first**, before the base locks
and before the lift travels to 450 mm, so anything still being held falls from
where it is rather than from up there.

It is not the slow ramp that shutdown uses; it is a step to zero, smoothed by
the same controller-side low-pass every other finger command goes through. The
hands then *stay* open, because a paused operator sends nothing usable and
hold-last has nothing to overwrite it with — until the next engage, which takes
over immediately.

Quest's Y+B homes the same way. Both gestures reach `home_arms()` on the node,
which is where the hands live, so nothing on the wire or in the client changed.
Homing a single arm (`home_left_arm`) opens only that side's hand.

One nit worth knowing: "zero" lands `finger1_joint1` — the thumb's first joint —
at its own lower limit (`+0.0583` rad left, `+0.0368` right) rather than at
0.000, because the simulator clips an incoming vector to the MJCF ranges and
that one joint's range excludes zero. A ~3° difference on one joint, invisible
in practice, but it means homing is not bit-identical to the `home` keyframe.

### Seeing what the hands are doing

`get_hand_state()` on port 5558 — or `get_state()` on the node — reports it:

| field | meaning |
| --- | --- |
| `qpos` | the (20,) vector each side is currently commanded to, or `None` if it has never had one |
| `engaged` | the publisher's `paused` bit, inverted. It flips the instant the operator shakas — but *not engaged* does **not** mean the hand opened. It means the pose stopped changing. |
| `origin` | where that pose came from: the glasses (`aria`), an RPC client (`rpc`), or a home (`home`) |
| `sends` | writes actually made *to the device*. Identical vectors are not resent, so a perfectly still engaged hand stops counting up. |

### The joint vector, and why nothing reorders it

`(20,)` radians, `{side}_finger{f}_joint{j}` for f in 1..5 (thumb..pinky) and
j in 1..4. The MJCF, the aria2robot publisher and `wujihandpy`'s `(5, 4)`
`set_joint_target_position` all agree on that order, so the device layout is a
plain `qpos.reshape(5, 4)` with row *f* = finger *f*+1 — no permutation
anywhere. The one table is `canonical_joint_names` in
[`robot/hand/wuji_driver.py`](../robot/hand/wuji_driver.py);
`robot/teleop/aria/stream.py` re-exports it. `tests/test_wuji_hand.py` pins it,
because a silent disagreement moves the wrong finger.

The simulator clips an arriving vector to the MJCF joint ranges as it enters,
and writes it verbatim after that. Clipping on the write instead would nudge an
*uncommanded* hand, because several joints have a range that excludes zero
(`left_finger1_joint1` starts at 0.0583 rad).

### Before the first hardware run

1. **Both serials must be set.** `hand.serial.left` / `hand.serial.right`.
   `wujihandpy.Hand()` with no serial takes whichever hand the USB bus
   enumerated first, so with two plugged in the sides are a coin flip — and a
   swap means the left hand making the right hand's grasp. The driver refuses to
   start rather than guess. A single-hand session (`hand.sides: left`, or
   `--hands left`) may leave it blank.
2. **The hands are commanded to rest at startup**, ramping over `hand.ramp_s`,
   as the last thing `HardwareWujiDriver.start()` does. Enabling the joints
   does not place them: until this runs the hand holds whatever pose it was
   physically left in, and after a crash that is whatever grasp it died in.
   Expect 1.5 s per hand at the end of node startup.
3. **The first operator command to each hand ramps too**, over the same
   `hand.ramp_s`, from rest to the grasp. That first pose is a whole grasp and
   it arrives the instant the operator engages; do not shorten the ramp to make
   engaging feel snappier. Later commands are single writes. Note the ramp
   blocks the finger loop, so two hands engaging take 1.5 s each in turn, and
   the pose the hand reaches is that old -- the next tick writes the operator's
   current pose in one go, smoothed only by `hand.lowpass_hz`. Engage with your
   hands still.
4. **`yor.py` starts the hands last**, after the arms have homed, so nothing
   closes a hand while an arm is still travelling -- and a hand that fails to
   open is **not fatal**, at either grain: one absent side leaves the other
   running, and only a total failure drops the fingers. The node logs `hands failed to start` and runs the
   arms without fingers, rather than discarding a completed homing cycle. If
   you expected fingers and have none, that line is the first thing to look
   for in the startup log.
5. **Watch `engaged`, and know what it does not mean.** See the table above.

> **E-stop deliberately does not open the hands.** `emergency_stop()` freezes
> wheels, lift and arms; the hands keep whatever pose they were last given,
> because a stop that sprang an open hand would drop whatever is being carried.
> What ramps them open and disables the joints is the node's shutdown — Ctrl-C
> on `yor.py` — and it runs *before* the arms are dropped. One process, one
> Ctrl-C: "stop the robot" is one action.

---

## 6. About the lift

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

## 7. Reading the screen (`sim_viz`)

Two things are worth knowing how to read:

- **The `ik_target` sphere must sit inside the WUJI palm**, not on the thin
  red/green/blue `{side}_arm_ee_axis_*` capsules 37.5 mm behind it. The marker
  rides the wrist, not the site the IK targets.
- **The mapped operator triad** (long thin needles, 0.20 m × 0.004 m) must sit
  coincident *and parallel* with that sphere's own thick 0.12 m × 0.008 m
  capsules. Mirrored or 90°-rotated means an axis table in
  [`clutch.py`](../robot/teleop/aria/clutch.py) is wrong. A gap with matching axes is IK tracking
  error, which the `pos_err` column names separately (`--` while released:
  nothing is mapped).

The 21-point operator hand skeleton used to be drawn here too, with `d` and
`gap` columns measuring its wrist against the commanded one. The landmarks it
needed no longer cross the wire, so the triad is the axis-correctness check now
— it answers the same question, on the same two maps, with two fewer scene
nodes.

The numbers arrive as two Rich tables refreshed at 1 Hz, not a scrolling line —
`[aria]` engage/release messages print above them. The first is where each hand
is commanded and whether the clutch is following it; the second is travel and
IK error and, in its caption, the DOFs the hands never command (lift, base,
solver status).

Their frames: `ik_target` and `travel` are robot **world** metres (`travel` is
your hand's displacement since engage, already scaled); `pos_err` / `ori_err`
are how far the arm lags the command it was given.

To prove the overlays are not silently offset — mjviser recentres its scene on
`base_link`, and a 9 cm shift reads exactly like tracking error — uncheck **Fix
Base** and reach far enough that the chassis rolls. The operator triad must stay
welded to the sphere as the base moves.

---

## 7b. Reading the screen (`--input aria`)

The RPC client has no browser and no overlays, so it prints three 1 Hz tables
instead — arms, stream, hands. Layout and columns:
[RUNNING.md](RUNNING.md), "The status table". Three things are worth knowing
here specifically.

**The `Stream` table is the client's own link, and it is where a bad session
shows up first.** `wuji` FPS well under the publisher's rate means packets are
not arriving, not that the mapping is wrong; a `p95` far above `p50` is a
link that stalls in bursts, which reaches the arms as stutter. Compare it
against the publisher's own table (`stream_sub.py`) — same columns, same
meaning — to place the loss on one side or the other.

**Latency needs the clock handshake**, against `publisher.clock_port` (5556).
It runs off-thread at startup and logs one line:

```
[aria] clock offset +3.63 ms (rtt 7.60 ms)
```

Until it lands, `p50`/`p95` read `--` rather than a confidently wrong number:
the two machines' wall clocks disagree by more than the delay being measured.
`publisher.stats: false` turns the whole table off.

**`State` means different things in the two tables, and §5 is why.** The arm
`State` is the clutch, so it says `paused` the instant you shaka. The hand
`State` is `Hands`' engagement and the fingers *hold* instead of releasing —
a `paused` arm beside a `held` hand whose angles stopped changing is exactly
right, and is not the hands ignoring the gesture. `Send Hz` 0.0 on an
`ENGAGED` hand is likewise correct: identical vectors are not resent.

What this client does **not** have is `sim_viz`'s 3D view. Validate a frame
change there (§1A, §7); these tables are for watching a session run.

---

## 8. Checks that need no glasses

```bash
python tests/test_aria_mapping.py     # 112 checks — the arm mapping
python tests/test_wuji_hand.py        #  60 checks — the finger path
python tests/test_teleop_status.py    #  44 checks — the client status table
cd ../aria2robot && pytest            #  27 tests — gestures, wire, frames
```

112 checks: the config loader, both axis tables against literals, the
convention rotation against this repo's MJCF hand mount, the clutch delta math
in **both** translation frames (that a metre up is a metre up at every engage
pose, that horizontal stays horizontal, that the heading still comes from
engage, that rotation and the engage anchor are identical either way, and that
a heading-less engage pose falls back instead of failing), the rigid flange
offset, the wire-2 decode *and* the pre-wire-2 compat path, the staleness gate,
the home counter (adopt-on-connect, fires once, a multi-step jump still fires
once, a restarted publisher resyncs silently, an engaged hand vetoes), and both
rendering fixes — the marker one measured as 0.0000 mm on the hand versus
37.5 mm at the flange.

The publisher's suite covers what moved up there: the thumbs-up detector, that
it and the shaka can never both read true, `HoldTrigger`'s dwell and latch
semantics, that a two-hand hold fires exactly once when driven at 200 Hz off a
30 Hz gesture update (the trap that a naive port falls into), the `wuji`
payload's key set and dtypes, and — load-bearing for the whole pre-composition
— that `mano_wrist_frame` commutes with a rigid transform.

`test_wuji_hand.py` covers the finger path with no hand and no publisher: the
`(20,)` → `(5, 4)` layout against the MJCF's own joint names, that every hand
joint address is contiguous so the published vector writes as one slice, the
hold-last policy on all four of its cases (paused, pre-engage, lost tracking,
nothing sent), the RPC surface's refusals and that it stays narrow,
send-on-change, the hardware ramp being a ramp and not a step, and the
simulator's injection — including that it lands *after* `apply_to_sim_kinematic` at every
branch of the control loop, which is the one ordering a refactor can quietly
break.

---

## 9. Troubleshooting

| Symptom | Cause / Fix |
| --- | --- |
| `publisher is pre-wire-2 (no T_odom_hand); composing locally` | The publisher has not been updated. Arms still work — the client does the multiply itself — but the home gesture is unavailable, since an old publisher sends no `home_seq`. Update `stream_pub`. |
| `publisher sends T_device_wrist but no hand frame` | Older still. `T_device_wrist` is Aria's own frame — different origin, different convention. That side stays disengaged rather than follow it. |
| Arms dead, no warning at all, publisher clearly running | An **old client against a new publisher**. Nothing in the wire-2 payload is recognisable to it and there is no field left to complain about. Update this repo; deploy it *before* the publisher next time. |
| `publisher runs hand=left; the home gesture needs both` | The publisher was started `--hand left`/`right`, so it never detects the two-thumb gesture and `home_seq` never moves. Restart it with `--hand both`. |
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
| Homing repeats while the thumb is up | `HoldTrigger.latch()` regressed to `reset()` — now in aria2robot's `utils/gesture.py`, pinned by "fires once and needs a release" there. |
| Thumbs up never homes, everything else works | If the publisher's console shows `home: both thumbs up -> home_seq=N`, the gesture was detected and this client ignored it: check `home.gesture` and that a clutch is not still engaged. If it shows nothing, the publisher never saw it — both hands must be shaka-off *and* in view. |
| Lift drifts down under load | Expected — `hold_lift` is a soft preference, see §6. |
| Arms follow but the sim's fingers never move | `yor_mujoco.py` was started with `--no-hands`, or without `--pub-host <publisher-ip>` so it is subscribing to `localhost`. The sim holds the home keyframe and says nothing — a silent publisher is a no-op by design. |
| Fingers move in `sim_viz` but not through `--input aria` | Different paths. `sim_viz` renders them off its own subscription; the node needs `--pub-host <publisher-ip>` (and not `--no-hands`) to reach the same stream. §5. |
| One hand never moves; the other is fine | `hand.sides` (or `--hands`) is naming one side. The startup line `[wuji] hands=…` says which. §5. |
| Fingers never move, `[wuji] no hands this session` in the log | `hand.sides: none` or `--hands none` — the arms-only setting. §5. |
| Real hands do not move, sim fingers do | `hand.backend` is still `none` (the default). `--backend hardware`, and check the startup line names the serials. |
| `two hands need a serial each so the sides cannot swap` | `hand.serial.left` / `.right` are unset. Deliberate refusal — a bare `Hand()` picks by USB enumeration order, so the sides would be a coin flip. §5. |
| The left hand makes the right hand's grasp | The two serials are swapped in the config. |
| One hand opens, the other logs `did not open` | Deliberate: that side is unplugged or unprovisioned, and the other keeps working. `[wuji] serving …` names what is left. §5. |
| Arms work, fingers never move, no error | The node caught a hand failure and carried on. Search the startup log for `hands failed to start` — the exception is printed verbatim. Usual causes: `wujihandpy` not installed in the node's env, `~/.wuji` not provisioned for that serial, USB permissions, or the device still held by a previous run. |
| The hand jumps when the operator first engages | Expected only if the startup rest ramp did not run — i.e. the driver failed to start, or a previous process left the hand enabled. §5.2. |
| Hands stay closed after the operator shakas off | Working as designed — `held` means the pose stopped changing, not that the hand opened. §5. |
| Hands stay closed after the publisher dies | Also by design: no staleness gate on the finger path, so a grasp is never dropped by a network fault. Ctrl-C the node, or call `open_hands()` on port 5558, to ramp them open. §5. |
| Hands keep their grip through an e-stop | Deliberate: a stop that sprang an open hand would drop the load. The node's shutdown is what ramps them open. §5. |
| `Send Hz` reads 0.0 while a hand is `ENGAGED` | Identical vectors are not resent. A still hand is 0.0; that is the send rate to the device, not the publish rate. |
| The wrong finger moves | A joint-order disagreement between the MJCF, the publisher and `wujihandpy`. Nothing in this repo permutes the vector — run `python tests/test_wuji_hand.py`, which pins the layout against the model's own joint names. |
