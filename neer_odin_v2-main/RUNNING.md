# YOR SLAM — Live Startup Procedure

Running live SLAM with EKF fusion and ZED camera.  
All commands use the `slam-zed` conda environment from the `YOR_Slam` directory.

---

## Robot PC

### Terminal 1 — ZED Publisher

Streams ZED camera pose, point cloud, RGB, and depth over ZMQ (port 6000).

```bash
cd /home/hello-stretch/YOR_Slam
conda activate slam-zed
python -m robot.zed_pub_node
```

### Terminal 2 — SLAM Node (EKF enabled by default)

Runs the mapping, EKF fusion, A\* planner, and Viser visualisation server (port 8099).

```bash
cd /home/hello-stretch/YOR_Slam
conda activate slam-zed
python -m robot.slam_node_
```

> **Key flags:**
> | Flag | Effect |
> |---|---|
> | `--hz 10` | Mapping rate (default 10 Hz) |
> | `--save` | Save map to `.npz` on exit |
> | `--map-path robot/mymap.npz` | Path for save/load |
> | `--load` | Load a previously saved map |
> | `--no-ekf` | Disable EKF fusion (raw ZED pose only) |

**Interactive controls** (focus the Terminal 2 window):

| Key | Action |
|---|---|
| `q` | Freeze static map → switch to pure navigation mode |
| `w` | Graceful shutdown |
| `Ctrl+C` | Emergency stop |

---

## Host Computer

### SSH Tunnel

Forward the Viser UI and any auxiliary port from the robot to your local machine.

```bash
ssh -N -L 8008:127.0.0.1:8099 -L 8101:127.0.0.1:8101 hello-stretch@100.121.217.126
```

| Local port | Robot port | Service |
|---|---|---|
| `8008` | `8099` | Viser 3D visualisation |
| `8101` | `8101` | Auxiliary (reserved) |

### Browser

Open the Viser live map viewer:

```
http://localhost:8008/
```

---

## System Architecture (brief)

```
ZED Camera
   │  pyzed
   ▼
zed_pub_node.py  ──(ZMQ port 6000)──►  slam_node_.py
                                            │
                              ┌─────────────┼──────────────┐
                              ▼             ▼              ▼
                         EKFSlamSource  MapManager    AStarPlanner
                         (20 Hz predict) (voxel map)  (path → YOR RPC)
                              │                            │
                         Viser :8099              YOR base :5557
```

- **EKF predict** runs at 20 Hz using swerve wheel encoders via `yor.get_base_encoders()`.
- **EKF update** fires every ZED frame with adaptive measurement noise scaled by ZED tracking confidence (0–100).
- **Planning auto-starts** once the voxel map reaches ≥ 500 voxels.
- **Paths** are streamed to the robot base via RPC (`follow_path`).
