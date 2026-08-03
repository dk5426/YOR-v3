# YOR iPad App — Architecture (v2)

The app no longer hardcodes a robot IP or boots straight into the control
screen. Flow: **discover a robot → pick a mode → robot lazily starts that
mode's scripts → control → leave mode → robot returns to idle**.

```
                         ┌──────────────────────── iPad (SwiftUI) ───────────────────────┐
                         │ RobotDiscoveryView → ModeSelectionView → Navigation/Mapping/  │
                         │        (NWBrowser)        (mode cards)        Teleop views    │
                         └──────┬──────────────────────┬──────────────────────┬──────────┘
                  mDNS browse + │            REST      │            HTTP/WS   │
                  GET /discovery│      /mode/start|stop│       8088 + 8099    │
                                ▼                      ▼                      ▼
   ┌──────────────────────────────────────── robot (Jetson Thor) ──────────────────────────┐
   │  app_listener.py  :8090   ← the ONLY process running at boot (_yor._tcp Bonjour)      │
   │        │ starts/stops tmux sessions per mode                                          │
   │        ├─ navigation: zed_pub_node :6000 → slam_node_ (Viser :8099) + yor_gateway     │
   │        │              :8088 ── ZMQ RPC ──▶ yor.py RPCServer 194.168.1.10:5557         │
   │        ├─ mapping:    serve_map_viser.py :8099 (latest .npz)                          │
   │        └─ teleop:     yor_teleop.py (Quest relay via iPad — future)                   │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

## Robot side

### `robot/app_listener.py` (port **8090**, runs at boot)

* Advertises `_yor._tcp.local.` via zeroconf. TXT records: `name`, `ip`,
  `modes`, `status`, `version`. The iPad reads these straight from the browse
  result — no resolution round-trip. The advertised/`/discovery` IP prefers
  the **Tailscale address** (override with `YOR_ADVERTISE_IP`), so a card
  found on WiFi keeps working when the iPad later roams off that network.
* REST API:
  * `GET /discovery` → `{"name": "YOR-thor", "ip": "...", "modes": [...], "status": "idle"}`
  * `POST /mode/start` `{"mode": "navigation", "fresh_map": true, "map_path": null}`
  * `POST /mode/stop` — zero velocity via gateway → `C-c` every pane (slam
    saves its map, ZED closes its area file) → wait ≤10 s → `tmux kill-session`
  * `GET /mode/status` → `{"mode", "status": idle|starting|running|error, "processes": [...], "detail"}`
* `POST /mode/start` is **asynchronous** (like stop): it validates, returns
  `"starting"` immediately, and builds the tmux session in a worker thread —
  pane-creation staggers take longer than the iPad's HTTP timeout.
* Modes:
  * **navigation** — `zed_pub_node` + `yor_gateway` only. **No SLAM, no
    point cloud, no mapping** — just the live RGB feed (with lane guides)
    and pose telemetry read straight off the ZED topics. Nothing is written
    to disk; ZED area memory is left untouched (no `--fresh`). Pure
    locomotion with minimal Jetson load.
  * **mapping** — full orbv2 ORB-SLAM3 pipeline (`zed_pub_node` →
    `orb_bridge` → `orb_slam_node` → `yor_gateway`, mirroring `orbv2/run.sh`
    with its 5 s / 30 s startup staggers). `/mode/start` takes either
    `map_name` (new map, saved to `orbv2/maps/<name>_<timestamp>.npz` on
    exit via `--save`) or `load_map` (an `.npz` filename from `GET /maps`,
    opened with `--load`, ZED keeps its area memory).
  * **teleop** — `yor_teleop.py`.
* `GET /maps` lists saved `.npz` maps (newest first) for the iPad's load
  picker; `DELETE /maps/{file}` permanently removes one (filenames are
  basename-sanitized so deletions can't escape the maps directory). The iPad
  exposes this as swipe-to-delete with a confirmation in the Load Map list.
* Pane liveness is tracked with exit-sentinel files in `/tmp/yor_listener/`
  (the `sleep N && …` startup staggers make tmux's current-command useless
  for crash detection).
* `POST /mode/stop` is **asynchronous**: it returns immediately with status
  `"stopping"` and the iPad polls `/mode/status` until `"idle"`. Mapping's
  graceful stop window is 40 s because `orb_slam_node` saves the map on
  SIGINT. The body accepts `{"save_map": false}` (mapping only) to discard
  instead: the SLAM node is SIGKILLed before shutdown so its save handler
  never runs. The iPad asks Save / Don't Save when leaving a new-map
  session; loaded-map sessions exit without the prompt.
* One mode at a time; starting a second returns **409**. Starting the active
  mode again is idempotent.
* A mode counts as **running** once every pane is alive and its required
  ports answer (navigation: 8088 + 8099, mapping: 8099). The iPad polls
  `/mode/status` once per second until then (75 s timeout — ZED init is slow).
* Boots via systemd: `robot/systemd/yor-app-listener.service` (copy to
  `/etc/systemd/system/`, `systemctl enable --now yor-app-listener`;
  restarts on failure after 3 s; logs in `journalctl -u yor-app-listener -f`).
* **Base control (NUC)**: `GET /base/status`, `POST /base/start`,
  `POST /base/stop` — Thor SSHes into the NUC (`194.168.1.10`, passwordless)
  and runs `cd ~/YOR && ./create_windows.sh` / kills its tmux session
  (`robot`). Start/stop are async; the iPad's mode-selection screen shows a
  Base Control bar (off → starting → running) polling every 3 s. Overrides:
  `YOR_NUC_HOST`, `YOR_NUC_USER`, `YOR_BASE_SESSION`, `YOR_BASE_START_CMD`.
  Driving modes (navigation, mapping) are gated behind it — the iPad alerts
  "Start Base Control first" if it isn't running.
* **Battery** comes from the NUC power sensor via `yor.py`
  (`get_battery_percent` / `get_battery_voltage`, only valid once Base
  Control is up). Two surfaces:
  * In-mode: the gateway adds `battery_percent` / `battery_voltage_v` to the
    telemetry frame (status-lane RPC, 5 s cache) and `GET /battery`; shown
    in the HUD `TelemetryBar`.
  * Idle home screen: `app_listener` includes `battery` in `GET /discovery`
    via `BatteryMonitor`, which polls the NUC RPC only while Base Control is
    running (timeout-guarded, never blocks discovery). Shown on the
    discovery card. `None` everywhere when the base is off.

### `robot/yor_gateway.py` (port **8088**, navigation mode only)

Same bridge as before (drive/lift/telemetry, velocity clamps 0.5 m/s ·
1.57 rad/s, 0.5 s drive watchdog), plus:

* Telemetry pose comes **directly from the `zed/pose` topic** (RPC
  `get_pose()` as fallback). The RPC path freezes during pure joystick
  driving because `BaseController` only refreshes `yor.pose` outside
  `BASE_VEL` mode — reading the publisher keeps X/Y/HDG live regardless of
  drive mode.
* **Stall/offline hardening:** two independent RPC lanes (drive vs status)
  so a stuck lift/pose call can't block joystick commands; every RPC call
  has a hard timeout and a timed-out lane is abandoned + rebuilt (commlink
  REQ sockets otherwise hang forever); telemetry echoes the last commanded
  velocity locally and caches lift height (≈1 RPC/s instead of 30); camera
  frames are grabbed + encoded by a **single producer** shared by all MJPEG
  clients, so iPad reconnect storms can't multiply the encode cost. iPad
  side: 1.5 s drive timeout, a 5 s stall watchdog on the camera stream, and
  a telemetry frame-clock keepalive that force-reconnects silent dead
  sockets.

* `GET /camera/stream` — MJPEG (`multipart/x-mixed-replace`, ~15 FPS) from the
  `zed/image` commlink topic. ZED SDK frames are BGRA, so we JPEG-encode the
  first three channels as BGR directly; set `YOR_CAMERA_BGR=0` if your
  publisher produces true RGB.
* `GET /camera/info` — ZED intrinsics (fx/fy/cx/cy + frame size). The iPad's
  **lane-guide overlay is drawn natively** (`LaneGuideOverlay.swift`,
  attached to the aspect-fitted video so it tracks the image on any
  device/orientation): crisper than the old baked-in version, zero Jetson
  cost, instant toggle, and ready for steering-curved guides later. The
  robot-side baked overlay still exists but defaults OFF
  (`YOR_LANE_GUIDE=1` / `POST /camera/lane-guide` for plain-browser
  viewers).
* `GET|POST /nav/waypoints` — the waypoint gate (below).
* Config via env: `YOR_RPC_HOST/PORT`, `YOR_ZED_HOST/PORT`, `YOR_CAMERA_FPS`.

### Waypoint gating

`slam_node_._path_sender_loop` used to stream `follow_path()` to the base
whenever the planner had a path. Now:

1. iPad toggles **Waypoints OFF** → `POST /nav/waypoints {"enabled": false}`.
2. Gateway immediately calls `follow_path(None)` (clears the active path,
   zeroes velocity) and remembers the flag.
3. `slam_node_` polls `GET /nav/waypoints` at 2 Hz and, while disabled, both
   suppresses its path sender **and drops click-to-goal events at the
   source** — so orbiting the map with fingers while waypoints are OFF can
   never queue a goal that fires later. **Fail-open**: if the gateway is
   unreachable (standalone `nav.sh` runs), the gate re-opens so old workflows
   still work.

## iPad side (`ios/YOR/`, SwiftUI, Apple frameworks only)

| File | Role |
|---|---|
| `YORApp.swift` | Entry; `RootView` switches on `RobotSession.stage` with spring transitions |
| `RobotSession.swift` | State machine (discovery → modeSelection → activeMode), owns clients, mode start/stop, E-stop |
| `RobotDiscoveryView.swift` | Two sections: **Saved Robots** (persisted IPs, probed every 3 s — the Tailscale path, since mDNS can't cross subnets) and **Nearby** (Bonjour). Manual connects are auto-saved |
| `SavedRobotsStore.swift` | UserDefaults-persisted robot IPs + concurrent `/discovery` status probing |
| `ModeSelectionView.swift` | Mode cards; shows start progress (`/mode/status` detail) and errors |
| `NavigationModeView.swift` | Fullscreen RGB camera + drive HUD (no map — disables waypoints on entry) |
| `MappingModeView.swift` | Drive-while-mapping: live Viser map + drive HUD + Waypoints toggle, map touch always on |
| `Components/DriveControls.swift` | Shared `DriveController` (20 Hz loop, speed ladders, release watchdog) + `DriveBottomBar` (joysticks, E-stop, lift) |
| `TeleopModeView.swift` | Placeholder ("Connect Meta Quest…") |
| `YORClient.swift` | `AppListenerClient` (8090) + `YORClient` gateway client (8088), host injected |
| `TelemetryClient.swift` | WS `/telemetry`, auto-reconnect, dynamic host |
| `VirtualJoystickView.swift` | Pure input widget — **no network calls** |
| `Components/` | Theme, ConnectionBadge/TelemetryBar/EStopButton, LiftControlPanel, MapWebView, VideoPIPView (MJPEG parser) |

`Info.plist` gained `NSLocalNetworkUsageDescription` + `NSBonjourServices`
(`_yor._tcp`) — required for local-network scanning since iOS 14.

### Joystick fixes

* **Rotation speed**: drive loop applies separate linear/angular ladders
  (linear 30/60/100 % of 0.5 m/s; angular 40/70/100 % of **1.57 rad/s**) with a
  signed-square curve per axis — fine control near centre, full hardware rate
  at full deflection. Rotation was previously capped at 0.6 rad/s.
* **Simultaneous sticks**: releasing a joystick no longer fires a
  `drive(0,0,0)` from the widget (that stomped the other stick). The widget
  only writes its bindings; one persistent 20 Hz drive loop owns all traffic.
* **Release watchdog**: when all axes read zero, the loop sends one explicit
  zero immediately and a confirmation at 200 ms, then idles. The gateway's
  0.5 s watchdog remains the network-loss backstop, and a telemetry WS drop
  zeroes the sticks client-side.

### Camera stream

`MJPEGStreamModel` parses the multipart stream by scanning for JPEG
SOI/EOI markers (`URLSessionDataDelegate`, no third-party deps).
`CameraStreamView` is the fullscreen stage of navigation mode (auto-reconnect,
"stream lost" banner); `VideoPIPView` (draggable/resizable floating window)
remains available for reuse.

### Viser touch (mapping)

**The big one:** in viser 0.2.x, while *any* scene-pointer callback is
registered (`on_scene_pointer`), the client disables `CameraControls` on
every pointerdown — mouse drag and touch do nothing (this is why camera
movement was keyboard-only on laptops too). The fix is in
`viserBridge.set_click_to_goal_enabled()`: the click-to-goal callback is now
**only registered while the iPad has Waypoints ON**, synced by
`slam_node_._apply_waypoint_gate`. Waypoints OFF (default) → full camera
control; ON → tap sets goals.

On top of that, `MapWebView` makes WKWebView deliver gestures to the page at
all: native scroll/pan/pinch recognizers disabled, injected viewport meta
with `user-scalable=no` (without it WebKit consumes two-finger gestures for
viewport zoom before JS sees them), `touch-action: none` CSS on document and
canvases, suppressed Safari `gesture*` events. Viser's `camera-controls`
library has native touch mappings: one-finger orbit, two-finger pinch zoom,
two-finger pan.

**Camera key pads (touch fallback):** mapping mode also shows on-screen
MOVE CAM (W/A/S/D + Q/E) and LOOK (arrow keys) hold-buttons — the same keys
viser binds in a desktop browser. They work by synthesizing
`keydown`/`keyup` `KeyboardEvent`s on `document` (viser's `hold-event`
library matches `event.code` there), via `window.__yorKey` injected by
`MapWebView` and driven natively through `ViserCameraController`. Toggle
with the keyboard button in the top bar.

## Teleop (Meta Quest via iPad relay)

```
Quest (PUB, binds :5555) ──► iPad relay (VRRelay) ──► NUC teleop SUB (YOR_D)
```

* **Transport:** the Quest's Unity app (`xarm-vr`) is a NetMQ `PublisherSocket`
  that `Bind`s `tcp://*:5555` and publishes `[oculus_controller, state]`. The
  robot's `oculus_bimanual_wholebody_teleop.py` is a ZMQ SUB. The iPad inserts
  itself as a **transparent TCP proxy** (`VRRelay.swift`, Network framework):
  it LISTENS on :5555 (the robot connects here) and, on connect, dials the
  Quest's :5555, forwarding raw bytes both ways. ZMTP rides straight through —
  no ZMQ library on iOS.
* **iPad UI** (`TeleopModeView.swift`): enter the Quest IP → Start Relay. Shows
  Quest/iPad/Robot link state and whether controller data is flowing. The
  E-stop cuts the relay (stops the command stream instantly); Exit leaves the
  mode (kills the NUC stack).
* **Robot launch:** teleop mode is a **NUC-SSH mode** (`NucLaunch` in
  `app_listener.py`), not a Thor tmux session. `/mode/start` writes the iPad's
  IP (the request's client host) to `~/.yor_vr_host` on the NUC, then runs
  `cd ~/YOR_D && ./create_windows.sh`. The teleop script's `_resolve_vr_host()`
  reads that file, so it subscribes to the iPad relay instead of the Quest
  directly. `/mode/stop` SSH-kills the NUC session.
* **Mutually exclusive with Base Control:** both use the NUC `robot` tmux
  session and `yor.py` on port 5557. Teleop start kills any existing `robot`
  session first. Stop Base Control before starting teleop.
* Env overrides: `YOR_TELEOP_DIR` (`~/YOR_D`), `YOR_TELEOP_SESSION`,
  `YOR_TELEOP_START_CMD`, `YOR_TELEOP_PANE_SEQUENCE`, `YOR_VR_HOST_FILE`.

## Safety invariants

* E-stop is rendered in **every** mode view and `RobotSession.emergencyStop()`
  is always wired to the gateway's `/stop`.
* Gateway zeroes velocity on: drive watchdog timeout, telemetry client
  disconnect, and process shutdown.
* `/mode/stop` zeroes velocity **before** tearing the stack down.
