# neer_odin_v2 — YOR robot navigation on the Manifold Odin 1 (no ROS)

Full YOR robot stack (from `neer_slam`) driven by a **Manifold Odin 1** SLAM box
(fisheye RGB + dToF LiDAR + IMU + on-device VIO/SLAM with loop closure) instead
of the ZED 2i. **No ROS 1/ROS 2 anywhere** — the Odin is driven through its
native C SDK via a small pybind11 bridge (`pyodin`), and the robot stack keeps
its original `commlink` pub/sub architecture.

```
                    ┌── tools/odin_viser_map.py ──▶ Viser :8080   (standalone live map,
Odin 1 ──USB3──▶ pyodin                                            dynamic-object removal,
 (SDK, no ROS)      └── robot/odin_pub_node.py ──commlink:6000──▶ robot/slam_node_.py
                        (drop-in for zed_pub_node)                 ├─ mapping (torch)
                                                                   ├─ A* path planning
                                                                   ├─ Viser :8099 (click a
                                                                   │   point → robot drives)
                                                                   └─ follow_path RPC → base
```

## Requirements

- Jetson (tested: **Jetson Thor, JetPack 7 / L4T r38.x**) or other aarch64/x86 Linux
- **Odin 1 firmware ≥ 0.12.0** (older firmware mis-speaks the SDK protocol —
  symptoms: heartbeat length errors, garbled clouds. Update with Manifold's
  `odin1_firmware_update_tool`.)
- USB **3.0** port + cable
- miniconda

## Fresh machine setup

```bash
git clone <this-repo> neer_odin_v2 && cd neer_odin_v2

# 1. system deps
sudo apt update && sudo apt install -y build-essential cmake git tmux libusb-1.0-0-dev libeigen3-dev

# 2. USB permission for the Odin (Rockchip VID 2207)
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2207", MODE="0666"' | sudo tee /etc/udev/rules.d/99-odin.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
# replug the Odin, then check:  lsusb -d 2207:

# 3. conda env `slam-odin` + python deps + build pyodin
bash setup_odin.sh

# 4. (full nav only) PyTorch for mapping_torch — use NVIDIA's JetPack-matched
#    aarch64 CUDA wheel (NOT plain `pip install torch`), then:
bash setup_odin.sh --full
```

## 1) Standalone live mapping (no robot needed)

```bash
bash run_viser_map.sh                 # → http://localhost:8080
```
Real-time incremental voxel map with the ZED pipeline's dynamic-object logic
(log-odds occupancy, range-image carving, walking-trail decay) ported to
numpy/cv2. GUI: range/denoise sliders, dynamic-removal + trail TTL, point
shape/size/shading, **Save map (PLY)** and **Save device map (.bin)** buttons
(files land in `maps/`).

Useful flags: `--voxel 0.03`, `--min-hits 3`, `--trail-ttl 1.5`,
`--point-shape circle`, `--save-dir maps`.

### Map persistence / relocalization
- **PLY** = the cleaned host map (view in MeshLab/CloudCompare, use for planning).
- **.bin** = the Odin's internal loop-closed map. Reuse it so a later session
  localizes in the SAME world frame:
  ```bash
  bash run_viser_map.sh --map-mode 2 --reloc-map maps/odin_reloc_<ts>.bin
  ```

## 2) Full robot navigation (camera mounted on the robot)

Same workflow as the ZED-era `nav.sh` — the sensor publisher is swapped, the
rest of the stack is untouched:

```bash
bash nav.sh            # tmux: [1] odin_pub_node  [2] slam_node_
                       # slam UI → http://<robot-ip>:8099
bash nav.sh --kill     # stop
```

- `robot/odin_pub_node.py` publishes the ZED-compatible commlink topics
  (`zed/pose` 20-float, `zed/image`, `zed/depth`, `zed/pcd` world→camera-frame
  cloud, `zed/camera_info`) on port 6000, converting Odin Z-up → stack Y-up.
- **Click-to-navigate**: in the slam_node_ Viser UI (:8099), click a point on
  the map → A* plans a path → `follow_path` RPC drives the base (requires the
  robot base stack running; see `robot/` and `docs/APP_ARCHITECTURE.md`).
- **Mount extrinsic**: set `T_cam_to_base` in `config/odin.yaml` once the Odin
  is bolted to the robot (identity by default; the ZED used a 23.8° tilt —
  measure yours).

## Layout (Odin-specific parts)

| Path | What |
|---|---|
| `odin_sdk/` | Vendored Manifold SDK: headers + `liblydHostApi_{arm,amd}.a` |
| `pyodin/` | pybind11 bridge (`build.sh` compiles into the env) |
| `robot/odin_pub_node.py` | Drop-in sensor publisher (replaces `zed_pub_node.py`) |
| `robot/nav/mapping/voxel_map_np.py` | numpy/cv2 voxel map engine (log-odds + carving + decay) |
| `tools/odin_viser_map.py` | Standalone live mapper + map saving |
| `tools/odin_pipe_publisher.cpp`, `tools/viser_subscriber_pipe.py` | early C++ pipe prototype (reference) |
| `config/odin.yaml` | mount extrinsic, pcd source, ports |
| `setup_odin.sh` / `nav.sh` / `run_viser_map.sh` | setup + launchers |

Everything else is the original `neer_slam` robot stack (arm, base, teleop,
msgs, nav; iOS app removed).

## Notes & gotchas

- **One consumer at a time** — the Odin is a single USB device; don't run two
  drivers/mappers at once.
- Stop programs with **Ctrl-C**, not `kill -9` (hard kills can wedge the
  device's onboard daemon → power-cycle the box to recover).
- The SDK segfaults harmlessly in its exit destructors; launchers guard this
  with `os._exit`.
- `sparkcan_py/` and `nerolib/` (robot base hardware drivers) are empty dirs
  here; clone them when building the physical-robot side:
  `https://github.com/vedAnts256/sparkcan_pybindings.git` and
  `https://github.com/dk5426/nerolib.git`, then follow `setup_thor.sh`.
- ZED remnants (`setup_thor.sh`, `setup_nav.sh`, `RUNNING*.md`) are kept for
  reference; the Odin path is `setup_odin.sh`.
