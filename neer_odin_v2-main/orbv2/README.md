# orbv2 — ORB-SLAM3 Loop-Closure Integration

Clean ORB-SLAM3 integration for the ZED-based SLAM pipeline.  Adds
globally-consistent loop-closure corrections to the existing EKF, reducing
cumulative pose drift.

**The existing SLAM pipeline is completely untouched** — you can still run
`slam_node_.py` as before.  orbv2 wraps it with an additional ORB-SLAM3
measurement source.

---

## Quick Start

### Prerequisites

1. **ORB-SLAM3** compiled with the `orb_pipe` binary:
   ```bash
   ls ~/ORB_SLAM3/Examples/RGB-D/orb_pipe   # must exist + be executable
   ```

2. **ORB vocabulary** (145 MB):
   ```bash
   ls robot/nav/orbslam3/ORBvoc.txt
   ```

3. **`slam-zed` conda environment** with all dependencies:
   ```bash
   conda activate slam-zed
   python -c "import commlink, cv2, numpy, viser; print('All OK')"
   ```

### Option A: Tmux Launcher (recommended)

Starts all 3 components in a single tmux session:

```bash
cd /home/hello-stretch/neer_slam
bash orbv2/run.sh
```

This creates 3 tmux panes:
1. ZED publisher (`zed_pub_node --fresh`)
2. ORB-SLAM3 bridge (`orbv2.orb_bridge`)
3. SLAM node (`orbv2.orb_slam_node`)

Kill with: `bash orbv2/run.sh --kill`

### Option B: Manual Launch (3 terminals)

**Terminal 1 — ZED Publisher** (unchanged from your existing workflow):
```bash
cd /home/hello-stretch/neer_slam
conda activate slam-zed
python -m robot.zed_pub_node --fresh
```

**Terminal 2 — ORB-SLAM3 Bridge:**
```bash
cd /home/hello-stretch/neer_slam
conda activate slam-zed
python -m orbv2.orb_bridge --gen-config
```

Wait until you see:
```
[OrbPipeBackend] Initialization wait complete.
[OrbBridge] Starting main loop.
```

**Terminal 3 — SLAM Node (ORB-enhanced):**
```bash
cd /home/hello-stretch/neer_slam
conda activate slam-zed
python -m orbv2.orb_slam_node --predict-hz 5
```

### Viewing the Map

Open in browser (from host machine):
```bash
ssh -N -L 8008:127.0.0.1:8099 hello-stretch@<jetson-ip>
# Then open http://localhost:8008/
```

---

## How It Works

### Architecture

```
ZED 2i Camera
   │  pyzed.sl
   ▼
zed_pub_node.py ──ZMQ:6000──┬──► orbv2/orb_bridge.py ──► orb_pipe (ORB-SLAM3)
                             │                                │
                             │                          ZMQ:6001 orb/pose
                             │                                │
                             └──► orbv2/orb_slam_node.py ◄────┘
                                        │
                              ┌─────────┼──────────────┐
                              ▼         ▼              ▼
                         EKF+ORB    MapManager    A* Planner
                         (fused)    (voxel map)   (path→NUC)
                              │
                         Viser :8099
```

### What ORB-SLAM3 Adds

The existing EKF fuses:
- **Predict**: swerve wheel encoders at 5-20 Hz (from NUC via RPC)
- **Update**: ZED VIO pose each frame (adaptive noise based on tracking confidence)

orbv2 adds a **third measurement source**:
- **ORB update**: ORB-SLAM3 keyframe poses at ≤5 Hz, with special handling for
  loop closures (the EKF gate is disabled and noise is tightened so the filter
  snaps to the globally-consistent ORB estimate)

### Loop Closure

When the robot revisits a previously seen area, ORB-SLAM3 detects matching ORB
features and performs a global pose-graph optimization.  This causes a "jump" in
the ORB-SLAM3 pose output.  orbv2 detects this jump (position change > threshold)
and applies it as an unconditional EKF correction, pulling the entire pose
estimate to the globally-consistent position.

---

## Configuration

### Command-Line Flags

All `slam_node_.py` flags are supported, plus:

| Flag | Default | Description |
|---|---|---|
| `--no-orb` | — | Disable ORB feedback (baseline ZED+encoder mode) |
| `--orb-host` | `127.0.0.1` | Host running `orbv2.orb_bridge` |
| `--orb-port` | `6001` | ZMQ port for `orb/pose` topic |
| `--orb-hz` | `5.0` | Max EKF update rate from ORB |
| `--orb-lc-thr` | `0.30` | Loop-closure jump threshold (metres) |

### ORB-SLAM3 Config

The bridge auto-generates the config from live ZED intrinsics on first run.
To regenerate:

```bash
python -m orbv2.config --out /tmp/orb_zed.yaml
```

Or with hardcoded defaults (no ZED needed):
```bash
python -m orbv2.config --out /tmp/orb_zed.yaml --no-zed
```

---

## Verifying Loop Closure

### Signs That Loop Closure Is Working

1. **Console log**: Watch Terminal 2 (bridge) for steady tracking:
   ```
   [OrbBridge] frames=200  ok=180  lost=20  tracking_ratio=90%
   ```

2. **Loop-closure detection**: Watch Terminal 3 (SLAM node) for:
   ```
   [orbv2] Loop-closure detected — position jump = 0.452 m
   [OrbFusedSource] orb_updates=42  loop_closures=1
   ```

3. **Health monitor**: Every 10 seconds:
   ```
   [orbv2 health] ORB=LIVE  updates=42  loop_closures=1  σ_pos=0.0032m  σ_yaw=0.15°
   ```

4. **Viser**: After driving a loop, the robot's return position should be
   closer to its start position compared to running without ORB.

### Comparison Test

Run the same trajectory twice:

```bash
# Baseline (no ORB):
python -m robot.slam_node_ --no-ekf
# or:
python -m orbv2.orb_slam_node --no-orb

# With ORB:
python -m orbv2.orb_slam_node
```

Compare the end-to-end drift for a round trip.

---

## Troubleshooting

### "ORB-SLAM3 binary not found"
```
[OrbPipeBackend] orb_pipe binary not found: ...
```
Specify the correct path:
```bash
python -m orbv2.orb_bridge --orb-bin ~/ORB_SLAM3/Examples/RGB-D/orb_pipe
```

### "Vocabulary not found"
```
[OrbBridge] ERROR: Vocabulary not found: ...
```
Download it:
```bash
bash robot/nav/orbslam3/download_vocab.sh
```

### ORB tracking ratio is 0%

- **Dark/textureless environment**: ORB needs visual texture. Try better lighting.
- **Camera motion too fast**: ORB loses tracking on fast motions. Move slowly.
- **Config mismatch**: Regenerate config while ZED is running:
  ```bash
  python -m orbv2.config --out /tmp/orb_zed.yaml
  ```

### "orbv2 health: ORB=STALE/OFFLINE"

The bridge is not publishing.  Check:
1. Is Terminal 2 (bridge) still running?
2. Is the `orb_pipe` process alive? Check for errors in its stderr output.
3. Is ZED publishing? Check Terminal 1.

### EKF keeps jumping

The loop-closure threshold (`--orb-lc-thr`) controls when an ORB jump triggers
an unconditional EKF correction.  Default is 0.30 m.  Raise it (e.g. `1.0`)
if you see false positives.

### Graceful Fallback

If ORB-SLAM3 stops working at any point, the system automatically falls back
to ZED + encoder-only EKF fusion.  No crash, no error storm — just a log message:
```
[orbv2 health] ORB=STALE/OFFLINE  updates=42  loop_closures=1
```

---

## File Structure

```
orbv2/
├── __init__.py           # Package marker
├── config.py             # ORB-SLAM3 YAML config generator
├── orb_bridge.py         # ZED → ORB-SLAM3 → orb/pose bridge
├── orb_fused_source.py   # ORB subscriber + EKF fusion layer
├── orb_slam_node.py      # Main entry point (wraps Slam + OrbFusedSource)
├── diagnostics.py        # Health monitoring
├── run.sh                # Tmux launcher
└── README.md             # This file
```

All modules import from the existing `robot.*` package — no code duplication.
The existing pipeline (`slam_node_.py`) is completely untouched.
