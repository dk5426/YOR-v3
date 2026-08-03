# YOR SLAM — ORB-SLAM3 Pipeline Live Startup

Running SLAM with **ORB-SLAM3 globally-consistent pose** fused into the EKF on top
of ZED VIO + swerve wheel encoders.

## Hardware topology

- **Jetson Thor** — `194.168.1.11`. Runs the SLAM stack: `zed_pub_node`,
  `orbslam_bridge`, `slam_node_orb`.
- **Raspberry Pi** — `194.168.1.10`. Runs the base + arms + joystick: `yor.py`,
  `robot/teleop/joystick.py`.
- The two communicate over the wired LAN (`enP2p1s0` on Jetson):
  - ZMQ pub/sub: ZED frames on port 6000, ORB pose on port 6001 (Jetson-local).
  - RPC: yor on port **5557** (Pi → Jetson over network) using a single ZMQ REP socket.

> **Important** — yor's RPC server uses a single REP socket. Multiple clients
> (slam_node + joystick) get serialized on the Pi. Keep slam_node's predict
> rate low (`--predict-hz 5`, default) so joystick `set_base_velocity` and
> click-to-goal `follow_path` aren't starved.

All commands use the `slam-zed` conda environment from the `neer_slam` directory.

---

## Architecture

```
ZED Camera
   │  pyzed
   ▼
zed_pub_node.py  ──ZMQ:6000──►  slam_node_orb.py
                                       │
                                       ├── EKFSlamSource  (ZED VIO + encoders)
                                       │
                                       └── OrbEKFSlamSource ◄─── orbslam_bridge.py
                                                                       ▲
                                                       ZMQ:6001 orb/pose

ZED RGB+depth  ──ZMQ:6000──►  orbslam_bridge.py  ──►  orb_pipe (ORB-SLAM3)
                                                              │
                                                              ▼
                                                       orb/pose @ 6001
```

ORB updates are fused into the same EKF that already runs ZED VIO + wheel
encoders. If the bridge stops publishing, the system silently falls back to the
existing ZED+encoder behaviour.

---

## Prerequisites (one-time setup)

```bash
# 1. ORB-SLAM3 built with the orb_pipe binary (already present on this Jetson Thor)
ls ~/ORB_SLAM3/Examples/RGB-D/orb_pipe       # must exist + be executable

# 2. Vocabulary
ls robot/nav/orbslam3/ORBvoc.txt             # 145 MB — must exist

# 3. Generate ORB-SLAM3 config from live ZED intrinsics (or use --gen-config below)
conda activate slam-zed
python -m robot.nav.orbslam3.gen_config --out /tmp/orb_zed.yaml
```

---

## Startup — 5 terminals across two machines

### On the Raspberry Pi — Terminal 1: yor.py (RPC server)

Base + lift + (optional) arms RPC server on port 5557. **Must be running first** —
joystick, click-to-goal, and slam_node all talk to this.

```bash
# SSH into the Pi (194.168.1.10)
cd ~/neer_slam        # or wherever the Pi's copy lives
conda activate slam-zed
python -m robot.yor
```

Wait until you see the RPC server log line before continuing.

### On the Raspberry Pi — Terminal 2: joystick.py

Reads pygame joystick events and forwards velocity commands to yor via RPC.
`yor.py` itself does **not** read the joystick — this separate process does.

```bash
# Same Pi, second SSH session
cd ~/neer_slam
conda activate slam-zed
python -m robot.teleop.joystick --start-enabled
```

`--start-enabled` makes the controller live immediately (no Start button
needed). You should see:
```
[joystick] pygame detected 1 joystick(s)
[joystick] using device: name='…'  axes=N  buttons=N  hats=N
[joystick] control_loop_running = True at launch. Sticks are LIVE.
[joystick] alive  running=True  buttons_held=[]
```

If the heartbeat (`alive` line) keeps appearing every ~2 s, the loop is fine.
If you see `set_base_velocity RPC failed: …`, the Pi's yor isn't reachable
from this terminal (it should be — both processes are on the same Pi).

> **Without `--start-enabled`** you must press the controller's Start button
> (Xbox button 7) once to enable the loop. Watch for `Control started`.

### On the Jetson Thor — Terminal 3: ZED Publisher

Streams ZED pose, point cloud, RGB, and depth over ZMQ (port 6000).

```bash
cd /home/hello-stretch/neer_slam
conda activate slam-zed
python -m robot.zed_pub_node --fresh
```

> **Important:** Use `--fresh` for the first ORB run. This deletes the saved
> `saved_map.area` so the ZED starts from the current position as the origin
> instead of restoring a stale offset (e.g. −268 m, −265 m). With the new
> `--align identity` default this is not strictly required for ORB itself, but
> it keeps the ZED odometry and the viser map aligned to the same origin.

### On the Jetson Thor — Terminal 4: ORB-SLAM3 Bridge

Subscribes to ZED RGB-D and publishes `orb/pose` on port 6001.

```bash
cd /home/hello-stretch/neer_slam
conda activate slam-zed
python -m robot.nav.odometry.orbslam_bridge \
    --orb-bin ~/ORB_SLAM3/Examples/RGB-D/orb_pipe \
    --vocab   robot/nav/orbslam3/ORBvoc.txt \
    --config  /tmp/orb_zed.yaml \
    --align   identity \
    --skip    2
```

Watch the terminal for:
- `[OrbBackend/process] Using persistent orb_pipe binary.` — good (real-time path)
- `[OrbBackend/process] orb_pipe not found — using Python shim.` — bad (slow batch)
- `[OrbBridge] frames=… ok=N lost=M aligned=True` every 5 s

If `ok` stays at 0 the camera is not seeing enough texture for ORB to initialise
— move the robot ~30 cm forward/back to seed the map.

### On the Jetson Thor — Terminal 5: SLAM Node (ORB feedback enabled)

Runs mapping, EKF fusion (ZED + encoders + ORB), A\* planner, and Viser on 8099.

```bash
cd /home/hello-stretch/neer_slam
conda activate slam-zed
python -m robot.slam_node_orb --predict-hz 5
```

`--predict-hz 5` (default) keeps EKF predict RPCs to the Pi at 5 Hz. Bump to
10 Hz once everything is stable; do NOT raise to 20 Hz while joystick is
active — the Pi's single REP socket can't serve 20 Hz predict + 60 Hz joystick
+ click-to-goal concurrently and joystick commands get starved.

> **Key flags (superset of `slam_node_.py`):**
> | Flag | Effect |
> |---|---|
> | `--hz 10` | Mapping rate (default 10 Hz) |
> | `--save` | Save map to `.npz` on exit |
> | `--map-path robot/mymap.npz` | Path for save/load |
> | `--load` | Load a previously saved map |
> | `--no-ekf` | Disable EKF fusion (raw ZED pose only) |
> | `--no-orb` | Disable ORB feedback (ZED + encoders only) |
> | `--orb-host` | Host running orbslam_bridge.py (default 127.0.0.1) |
> | `--orb-port` | ZMQ port for `orb/pose` (default 6001) |
> | `--orb-hz` | Max EKF update rate from ORB (default 5 Hz) |
> | `--orb-lc-thr` | Loop-closure jump threshold m (default 2.0) |

**Interactive controls** (focus Terminal 5):

| Key | Action |
|---|---|
| `q` | Freeze static map → switch to pure navigation mode |
| `w` | Graceful shutdown |
| `Ctrl+C` | Emergency stop |

---

## Host Computer

### SSH Tunnel

```bash
ssh -N -L 8008:127.0.0.1:8099 -L 8101:127.0.0.1:8101 hello-stretch@100.121.217.126
```

### Browser

```
http://localhost:8008/
```

The viewport will auto-center on the robot. With the new sanity clamp, if the
pose is ever > 100 m from origin (runaway ORB / stale area memory) the camera
falls back to the grid origin and you'll see a `[Viser] Pose out of range`
log in Terminal 4.

---

## Troubleshooting

### "Joystick does nothing"

Four independent things must all be true (joystick + yor live on the Pi; SLAM
lives on the Jetson — they talk over the LAN):

1. **`yor.py` (Pi Terminal 1) is running** and bound to all interfaces.
   ```bash
   # on the Pi:
   ps aux | grep -E "robot\.yor" | grep -v grep
   ss -tlnp | grep 5557      # should show *:5557 LISTEN
   ```
2. **`robot.teleop.joystick` (Pi Terminal 2) is running** and printing the
   `[joystick] alive  running=True` heartbeat every 2 s.
3. **Either** you launched it with `--start-enabled`, **or** you pressed the
   controller's Start button and saw `Control started`.
4. **Jetson's slam_node is using `--predict-hz 5`** (or lower). At 20 Hz the
   Pi's REP socket gets saturated and joystick `set_base_velocity` calls back
   up — robot looks dead even though everything is "running".

Quick bisect: stop slam_node on the Jetson. If the joystick suddenly works,
the Pi's RPC was being starved — lower `--predict-hz`.

Other gotchas:
- The deadzone in `joystick.py:154` is `|target| > 1e-2`. Below that nothing is
  sent. So if `slam_node_orb` put the base in `PATH_FOLLOWING`, a barely-moved
  stick won't reclaim it. Push the stick decisively.
- If `pygame.joystick.get_count() < 1` the process crashes at start — check
  Terminal 2 for `RuntimeError: No joystick detected`.

### "Click-to-goal does nothing"

Look in Terminal 5 for:
```
[slam_node_new] follow_path RPC failed: …
```

If you see this, `yor.py` is unreachable on port 5557 — restart Terminal 1.
If you don't see this but nothing happens, check Terminal 1 for the
`[YOR] follow_path: n=… first=… last=…` log — that confirms the RPC arrived.
If yor logs the path but the robot doesn't move, the base controller is in a
state that won't follow it (e.g. emergency stop, lift mid-motion).

### "Viser opens with the camera weirdly far away or stuck on top of the robot"

Pre-fix, this was caused by the `FrameAligner` baking a ZED area-memory offset
into every ORB pose. Two layers of defence are now in place:

1. `orbslam_bridge.py --align identity` (default) makes ORB and EKF share
   `[0, 0, 0]` as origin by construction — no transfer of ZED offsets.
2. The viser auto-center clamps to grid origin when the pose is > 100 m.

If you still see weirdness, run with `--no-orb` (Terminal 4) to confirm whether
the issue is in the ORB fusion or the underlying ZED VIO.

### "EKF state keeps jumping"

The loop-closure threshold (`--orb-lc-thr`) controls when an ORB jump is
treated as a globally-consistent loop closure and force-snaps the EKF. Default
is **2.0 m** (deliberately conservative). Lower it (e.g. `0.5`) only once you
have confirmed the bridge is producing stable poses.

### "ORB-SLAM3 binary not found"

```
[OrbBridge] ERROR: ORB-SLAM3 binary not found: ~/ORB_SLAM3/Examples/RGB-D/rgbd_tum
```

Pass `--orb-bin` explicitly:
```bash
--orb-bin ~/ORB_SLAM3/Examples/RGB-D/orb_pipe
```

The bridge auto-detects `orb_pipe` next to `--orb-bin` and uses it; the
fallback Python shim is **much slower** and only useful for offline debug.

---

## System Architecture (brief)

| Component | Port | Topic(s) |
|---|---|---|
| `zed_pub_node` | 6000 | `zed/image`, `zed/depth`, `zed/pose`, `zed/camera_info`, `zed/pcd` |
| `orbslam_bridge` | 6001 | `orb/pose` |
| `slam_node_orb` (viser) | 8099 | — |
| `yor.py` (RPC) | 5557 | — |

- **EKF predict** runs at 20 Hz from swerve wheel encoders via `yor.get_base_encoders()`.
- **EKF ZED update** every ZED frame, R scaled by tracking confidence (0–100).
- **EKF ORB update** ≤ 5 Hz, tight R (5 mm position, 0.1° heading), gated by Mahalanobis χ² = 50.
- **Planning auto-starts** once the voxel map reaches ≥ 500 voxels.
- **Paths** stream to the base via RPC (`follow_path`).
