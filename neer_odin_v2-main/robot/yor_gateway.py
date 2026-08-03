#!/usr/bin/env python3
"""
yor_gateway.py — HTTP/WebSocket bridge between the iPad app and the robot's
ZMQ RPC server (robot.yor.YOR via commlink).

Runs on port 8088 in navigation/mapping mode (started by robot/app_listener.py).

Endpoints
---------
  GET  /health                liveness + RPC connectivity
  GET  /status                robot_connected, pose, cmd_vel
  POST /drive                 {vx, vy, omega} m/s, m/s, rad/s — velocity-clamped
  POST /stop                  zero velocity immediately
  POST /lift/up|down|stop|home
  POST /lift/to-height        {target_m}
  GET  /pose                  {x, y, theta}
  GET  /encoders              raw swerve encoder dump
  GET  /cmd-vel               last commanded velocity
  WS   /telemetry             10 Hz push: {t, pose, cmd_vel, lift_height_m}
  GET  /camera/stream         MJPEG (multipart/x-mixed-replace) from zed/image
  GET  /nav/waypoints         {"enabled": bool} — polled by slam_node_
  POST /nav/waypoints         {"enabled": bool} — set by iPad; on disable the
                              gateway calls follow_path(None) to clear any
                              active path

Robustness design (field WiFi/Tailscale is flaky; RPC peers can wedge)
----------------------------------------------------------------------
  * Two independent RPC lanes: DRIVE (set_base_velocity/stop only) and
    STATUS (lift, encoders, pose fallback). A stuck status call can never
    stall joystick commands.
  * Every RPC call has a hard timeout. A timed-out lane is abandoned (the
    stuck worker thread keeps the dead socket) and rebuilt on the next call —
    no permanent hangs.
  * Telemetry echoes the last commanded velocity locally and caches lift
    height for 1 s, so the 10 Hz telemetry loop generates ~1 RPC/s instead
    of 30.
  * Camera frames are grabbed + JPEG-encoded ONCE by a single producer
    thread; every /camera/stream client just reads the latest shared JPEG.
    Reconnect storms can't multiply the encode cost, and the producer idles
    when nobody is watching.
  * Drive watchdog: no /drive for 0.5 s while moving → auto-zero velocity.

Env overrides
-------------
  YOR_RPC_HOST   (default 194.168.1.10)   robot.yor RPC server
  YOR_RPC_PORT   (default 5557)
  YOR_ZED_HOST   (default 127.0.0.1)      zed_pub_node publisher
  YOR_ZED_PORT   (default 6000)
  YOR_GATEWAY_PORT (default 8088)
  YOR_CAMERA_FPS (default 15)             MJPEG frame rate
  YOR_CAMERA_BGR (default 1)              ZED SDK frames are BGRA; set 0 if the
                                          published array is genuinely RGB
  YOR_LANE_GUIDE (default 1)              backup-camera lane overlay on/off
  YOR_CAM_HEIGHT_M / YOR_CAM_PITCH_DEG / YOR_ROBOT_WIDTH_M / YOR_CLEARANCE_M
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from commlink import RPCClient, Subscriber

# ── Config ────────────────────────────────────────────────────────────────
YOR_RPC_HOST = os.environ.get("YOR_RPC_HOST", "194.168.1.10")
YOR_RPC_PORT = int(os.environ.get("YOR_RPC_PORT", "5557"))
ZED_HOST = os.environ.get("YOR_ZED_HOST", "127.0.0.1")
ZED_PORT = int(os.environ.get("YOR_ZED_PORT", "6000"))
GATEWAY_PORT = int(os.environ.get("YOR_GATEWAY_PORT", "8088"))

MAX_VX_MPS = 0.5
MAX_VY_MPS = 0.5
MAX_OMEGA_RPS = 1.57

DRIVE_WATCHDOG_S = 0.5
TELEMETRY_HZ = 10.0
CAMERA_FPS = float(os.environ.get("YOR_CAMERA_FPS", "15"))
CAMERA_IS_BGR = os.environ.get("YOR_CAMERA_BGR", "1") != "0"

IMAGE_TOPIC = "zed/image"
POSE_TOPIC = "zed/pose"
CAMERA_INFO_TOPIC = "zed/camera_info"

# Lane-guide overlay baked into /camera/stream — DEFAULT OFF: the iPad now
# draws its own crisp native overlay (LaneGuideOverlay.swift) using the
# intrinsics from GET /camera/info. Set YOR_LANE_GUIDE=1 (or POST
# /camera/lane-guide) only if you want the guides visible to plain-browser
# viewers of the MJPEG stream.
LANE_GUIDE_DEFAULT = os.environ.get("YOR_LANE_GUIDE", "0") != "0"
lane_guide_enabled = LANE_GUIDE_DEFAULT
CAM_HEIGHT_M = float(os.environ.get("YOR_CAM_HEIGHT_M", "0.21"))
CAM_PITCH_DEG = float(os.environ.get("YOR_CAM_PITCH_DEG", "23.8"))
ROBOT_WIDTH_M = float(os.environ.get("YOR_ROBOT_WIDTH_M", "0.34"))
CLEARANCE_RADIUS_M = float(os.environ.get("YOR_CLEARANCE_M", "0.30"))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _pose_from_msg(msg) -> Optional[dict]:
    """{x, y, theta} from a zed/pose message.

    Layout (zed_pub_node): base quat+trans [0:7] as [qx,qy,qz,qw,tx,ty,tz].
    Mirrors robot/base.py get_pose(): x = tx, y = tz (XZ ground plane),
    theta = atan2(-R[2][0], R[0][0]) mod 2π — so HDG matches what RPC
    get_pose() used to report.
    """
    try:
        qx, qy, qz, qw, tx, _ty, tz = (float(v) for v in list(msg)[:7])
    except Exception:
        return None
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r20 = 2.0 * (qx * qz - qw * qy)
    theta = math.atan2(-r20, r00) % (2.0 * math.pi)
    return {"x": tx, "y": tz, "theta": theta}


# ── Lane guide overlay ────────────────────────────────────────────────────
def draw_lane_guide(
    frame,
    fx: float, fy: float, cx: float, cy: float,
    cam_height_m: float = CAM_HEIGHT_M,
    cam_pitch_deg: float = CAM_PITCH_DEG,
    robot_width_m: float = ROBOT_WIDTH_M,
    clearance_radius_m: float = CLEARANCE_RADIUS_M,
    d_near_m: float = 0.25,
    d_far_m: float = 3.0,
    is_bgr: bool = True,
):
    """Draw backup-camera style lane guides on a camera frame, in place.

    Pure static geometry — no depth, no detection. Ground points at forward
    distance d and lateral offset w (camera height h, pitched down by θ) map
    into the optical frame as:

        x_cam = w
        y_cam = h·cosθ − d·sinθ        (y is down; ground is below camera)
        z_cam = h·sinθ + d·cosθ

    then through the pinhole: u = cx + fx·x/z, v = cy + fy·y/z. Straight
    ground lines stay straight in the image, so each guide is just two
    projected endpoints. Sanity check: the optical axis meets the ground at
    d = h/tanθ ≈ 0.48 m, which lands exactly on the image centre.

    Drawn (assuming the camera sits on the robot's centreline):
      * solid green rails at ±robot_width/2  — where the robot's body goes
      * thin amber rails at ±clearance_radius — the keep-out envelope
      * green crossbars + labels at 0.5 / 1 / 2 m ground distance
    """
    import cv2  # noqa: PLC0415

    th = math.radians(cam_pitch_deg)
    sin_t, cos_t = math.sin(th), math.cos(th)
    h = cam_height_m

    def project(w: float, d: float):
        z = h * sin_t + d * cos_t
        if z <= 1e-6:
            return None
        y = h * cos_t - d * sin_t
        return (int(round(cx + fx * w / z)), int(round(cy + fy * y / z)))

    def line3d(w0, d0, w1, d1, color, thickness):
        p0, p1 = project(w0, d0), project(w1, d1)
        if p0 is None or p1 is None:
            return
        # dark underlay so the guide reads on bright floors
        cv2.line(frame, p0, p1, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)

    green = (80, 220, 80)
    amber = (0, 190, 255) if is_bgr else (255, 190, 0)
    half_w = robot_width_m / 2.0

    # Clearance envelope (thin amber)
    for s in (-1.0, 1.0):
        line3d(s * clearance_radius_m, d_near_m, s * clearance_radius_m, d_far_m, amber, 1)

    # Robot-width lane (solid green)
    for s in (-1.0, 1.0):
        line3d(s * half_w, d_near_m, s * half_w, d_far_m, green, 2)

    # Distance crossbars + labels
    for d in (0.5, 1.0, 2.0):
        line3d(-half_w, d, half_w, d, green, 1)
        p = project(half_w + 0.04, d)
        if p is not None:
            label = f"{d:g}m"
            cv2.putText(frame, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def _encode_jpeg(frame, quality: int, is_bgr: bool) -> Optional[bytes]:
    # Prefer OpenCV (fast, available in the slam-zed env); fall back to PIL.
    try:
        import cv2  # noqa: PLC0415

        bgr = frame if is_bgr else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None
    except ImportError:
        pass
    try:
        import io  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        rgb = frame[..., ::-1] if is_bgr else frame
        out = io.BytesIO()
        Image.fromarray(rgb).save(out, format="JPEG", quality=quality)
        return out.getvalue()
    except Exception:
        return None


# ── RPC lanes ─────────────────────────────────────────────────────────────
class RPCLane:
    """One commlink RPCClient behind a single-worker executor.

    The worker serializes calls (REQ sockets demand strict send/recv
    alternation) and gives every call a hard timeout. commlink has no socket
    timeout of its own, so a lost reply would otherwise block forever — the
    classic 'robot shows offline and never comes back'. On timeout the whole
    lane (executor + client) is abandoned: the stuck thread keeps the dead
    socket and the next call gets a fresh one.
    """

    def __init__(self, host: str, port: int, name: str, timeout_s: float):
        self.host = host
        self.port = port
        self.name = name
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._client: Optional[RPCClient] = None
        self.last_error: Optional[str] = None

    def _do_call(self, method: str, args: tuple):
        if self._client is None:
            self._client = RPCClient(self.host, self.port)
        try:
            return getattr(self._client, method)(*args)
        except Exception as e:
            # EFSM = REQ socket out of step; drop the client so the next
            # call on this worker rebuilds it.
            if "current state" in str(e):
                self._client = None
            raise

    def call(self, method: str, *args):
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"rpc-{self.name}")
            executor = self._executor
        future = executor.submit(self._do_call, method, args)
        try:
            return future.result(timeout=self.timeout_s)
        except FuturesTimeout:
            with self._lock:
                if self._executor is executor:
                    # Abandon: stuck worker keeps the dead socket; rebuild.
                    executor.shutdown(wait=False)
                    self._executor = None
                    self._client = None
            self.last_error = f"{method}: timeout after {self.timeout_s}s"
            print(f"[gateway] RPC lane {self.name!r} timed out on {method} — lane rebuilt")
            raise TimeoutError(self.last_error)
        except Exception as e:
            self.last_error = f"{method}: {e}"
            raise

    def try_call(self, method: str, *args):
        try:
            return self.call(method, *args)
        except Exception:
            return None


class RobotBridge:
    """Drive + status access to the YOR RPC server on independent lanes."""

    def __init__(self, host: str, port: int):
        # Drive gets a tight timeout: better to drop one 20 Hz command than
        # to stall the stream. Status calls can afford a little longer.
        self.drive_lane = RPCLane(host, port, "drive", timeout_s=0.8)
        self.status_lane = RPCLane(host, port, "status", timeout_s=1.5)

        # Local echo of the last commanded velocity (for telemetry — no RPC).
        self.last_cmd = (0.0, 0.0, 0.0)
        self.last_cmd_t = 0.0

        # Lift height cache (1 s TTL — it moves slowly).
        self._lift_h: Optional[float] = None
        self._lift_h_t = 0.0

        # Battery cache (5 s TTL — the pico reports ~1 Hz and it changes slowly).
        self._batt: dict = {"percent": None, "voltage_v": None}
        self._batt_t = 0.0

        # Drive watchdog state
        self._last_drive_t = 0.0
        self._last_cmd_nonzero = False
        self._watchdog_stop = threading.Event()
        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    @property
    def last_error(self) -> Optional[str]:
        return self.drive_lane.last_error or self.status_lane.last_error

    @property
    def connected(self) -> bool:
        return self.status_lane.try_call("get_pose") is not None

    # -- drive (drive lane only) --------------------------------------------
    def drive(self, vx: float, vy: float, omega: float):
        vx = _clamp(vx, -MAX_VX_MPS, MAX_VX_MPS)
        vy = _clamp(vy, -MAX_VY_MPS, MAX_VY_MPS)
        omega = _clamp(omega, -MAX_OMEGA_RPS, MAX_OMEGA_RPS)
        self._last_drive_t = time.time()
        self._last_cmd_nonzero = abs(vx) + abs(vy) + abs(omega) > 1e-4
        self.drive_lane.call("set_base_velocity", [vx, vy, omega])
        self.last_cmd = (vx, vy, omega)
        self.last_cmd_t = time.time()

    def stop_base(self):
        self._last_cmd_nonzero = False
        self.drive_lane.call("set_base_velocity", [0.0, 0.0, 0.0])
        self.last_cmd = (0.0, 0.0, 0.0)
        self.last_cmd_t = time.time()

    def _watchdog_loop(self):
        """Auto-zero the base if /drive goes quiet while moving (link loss)."""
        while not self._watchdog_stop.is_set():
            time.sleep(0.1)
            if not self._last_cmd_nonzero:
                continue
            if time.time() - self._last_drive_t > DRIVE_WATCHDOG_S:
                print("[gateway] drive watchdog fired — zeroing velocity")
                self._last_cmd_nonzero = False
                if self.drive_lane.try_call("set_base_velocity", [0.0, 0.0, 0.0]) is not None:
                    self.last_cmd = (0.0, 0.0, 0.0)
                    self.last_cmd_t = time.time()

    # -- status (status lane) -------------------------------------------------
    def lift_height(self) -> Optional[float]:
        now = time.time()
        if now - self._lift_h_t > 1.0:
            h = self.status_lane.try_call("get_lift_height")
            if h is not None:
                self._lift_h = float(h)
            self._lift_h_t = now
        return self._lift_h

    def battery(self) -> dict:
        """{'percent', 'voltage_v'} from the NUC's yor.py (None until Base
        Control is up and the pico has reported). Cached 5 s."""
        now = time.time()
        if now - self._batt_t > 5.0:
            pct = self.status_lane.try_call("get_battery_percent")
            volt = self.status_lane.try_call("get_battery_voltage")
            # Keep the last good reading if a single poll missed.
            if pct is not None:
                self._batt["percent"] = float(pct)
            if volt is not None:
                self._batt["voltage_v"] = float(volt)
            self._batt_t = now
        return self._batt

    def shutdown(self):
        self._watchdog_stop.set()
        self.drive_lane.try_call("set_base_velocity", [0.0, 0.0, 0.0])


# ── ZED stream (camera frames + pose) ─────────────────────────────────────
class ZedSource:
    """Lazy, non-blocking subscriber to the ZED publisher.

    Serves pose for /telemetry (RPC get_pose freezes during pure joystick
    driving — BaseController only refreshes yor.pose outside BASE_VEL mode)
    and raw frames for the camera broadcaster.

    Subscriber construction can block while the publisher is down, so it runs
    in a background thread (same pattern as base.py zed_sub_init) and is
    retried at most every 2 s. Callers just get None until it's ready.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sub: Optional[Subscriber] = None
        self._lock = threading.Lock()
        self._initializing = False
        self._last_try = 0.0

    def _kick_init(self):
        with self._lock:
            if self._sub is not None or self._initializing:
                return
            if time.time() - self._last_try < 2.0:
                return
            self._last_try = time.time()
            self._initializing = True

        def worker():
            sub = None
            try:
                sub = Subscriber(host=self.host, port=self.port,
                                 topics=[IMAGE_TOPIC, POSE_TOPIC, CAMERA_INFO_TOPIC])
            except Exception:
                pass
            with self._lock:
                if sub is not None:
                    self._sub = sub
                    print("[gateway] ZED subscriber ready")
                self._initializing = False

        threading.Thread(target=worker, daemon=True).start()

    def get(self, topic: str):
        with self._lock:
            sub = self._sub
        if sub is None:
            self._kick_init()
            return None
        try:
            return sub[topic]
        except Exception:
            return None

    def latest_pose(self) -> Optional[dict]:
        msg = self.get(POSE_TOPIC)
        return None if msg is None else _pose_from_msg(msg)

    def latest_jpeg(self, quality: int = 80) -> Optional[bytes]:
        msg = self.get(IMAGE_TOPIC)
        if msg is None:
            return None
        frame = msg.get("image") if isinstance(msg, dict) else None
        if frame is None:
            return None

        if lane_guide_enabled:
            info = self.get(CAMERA_INFO_TOPIC)  # published at 1 Hz, pull is cached
            if isinstance(info, dict) and "fx" in info:
                try:
                    import numpy as np  # noqa: PLC0415

                    frame = np.ascontiguousarray(frame)  # cv2 needs writable, contiguous
                    draw_lane_guide(
                        frame,
                        fx=float(info["fx"]), fy=float(info["fy"]),
                        cx=float(info["cx"]), cy=float(info["cy"]),
                        is_bgr=CAMERA_IS_BGR,
                    )
                except Exception:
                    pass  # overlay is cosmetic — never break the stream

        return _encode_jpeg(frame, quality=quality, is_bgr=CAMERA_IS_BGR)


# ── Camera broadcaster ────────────────────────────────────────────────────
class CameraBroadcaster:
    """Single producer, many consumers for the MJPEG stream.

    Exactly one thread grabs + encodes frames, no matter how many clients are
    connected (iPad reconnect storms used to spawn a fresh grab+encode loop
    per connection, which slowly ate the CPU and threadpool until the gateway
    went 'offline'). The producer idles when nobody is watching.
    """

    def __init__(self, zed: ZedSource):
        self.zed = zed
        self._cond = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._seq = 0
        self._clients = 0
        self._thread: Optional[threading.Thread] = None

    def _ensure_producer(self):
        with self._cond:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._producer, daemon=True)
                self._thread.start()

    def _producer(self):
        interval = 1.0 / max(1.0, CAMERA_FPS)
        while True:
            with self._cond:
                idle = self._clients == 0
            if idle:
                time.sleep(0.2)
                continue

            t0 = time.time()
            jpeg = self.zed.latest_jpeg()
            if jpeg is not None:
                with self._cond:
                    self._jpeg = jpeg
                    self._seq += 1
                    self._cond.notify_all()
            else:
                time.sleep(0.5)  # camera not publishing — don't busy-spin
                continue
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    def frames(self, boundary: str):
        """Per-client generator: serve the latest shared JPEG as it updates."""
        self._ensure_producer()
        with self._cond:
            self._clients += 1
        try:
            last_seq = 0
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._seq != last_seq, timeout=5.0)
                    if self._seq == last_seq:
                        continue  # producer quiet (camera down) — keep waiting
                    last_seq = self._seq
                    jpeg = self._jpeg
                yield (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode() + jpeg + b"\r\n"
        finally:
            with self._cond:
                self._clients -= 1


# ── Waypoint gate ─────────────────────────────────────────────────────────
class WaypointGate:
    """Whether slam_node_ may stream follow_path() targets to the base.

    Default disabled — matches the iPad's "Waypoints OFF" default. slam_node_
    polls GET /nav/waypoints (~2 Hz) and suppresses its path sender while
    disabled. Disabling also clears any active path immediately via RPC.
    """

    def __init__(self):
        self._enabled = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set(self, enabled: bool):
        with self._lock:
            self._enabled = bool(enabled)


# ── FastAPI app ───────────────────────────────────────────────────────────
bridge = RobotBridge(YOR_RPC_HOST, YOR_RPC_PORT)
zed = ZedSource(ZED_HOST, ZED_PORT)
broadcaster = CameraBroadcaster(zed)
waypoint_gate = WaypointGate()


def current_pose() -> Optional[dict]:
    """Live ZED pose, falling back to RPC get_pose (stale in BASE_VEL mode)."""
    p = zed.latest_pose()
    if p is not None:
        return p
    rp = bridge.status_lane.try_call("get_pose")
    if rp is not None and rp.get("x") is not None:
        return rp
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[gateway] up on :{GATEWAY_PORT} → RPC {YOR_RPC_HOST}:{YOR_RPC_PORT}, ZED {ZED_HOST}:{ZED_PORT}")
    yield
    bridge.shutdown()
    print("[gateway] shut down — velocity zeroed")


app = FastAPI(title="YOR Gateway", lifespan=lifespan)


class DriveRequest(BaseModel):
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


class LiftHeightRequest(BaseModel):
    target_m: float


class WaypointsRequest(BaseModel):
    enabled: bool


# -- health / status -------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "robot_connected": bridge.connected}


@app.get("/status")
def status():
    vx, vy, omega = bridge.last_cmd
    recently_driving = time.time() - bridge.last_cmd_t < 2.0
    return {
        "robot_connected": recently_driving or bridge.connected,
        "pose": current_pose(),
        "cmd_vel": {"vx": vx, "vy": vy, "omega": omega},
    }


# -- drive -------------------------------------------------------------------
@app.post("/drive")
def drive(req: DriveRequest):
    try:
        bridge.drive(req.vx, req.vy, req.omega)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.post("/stop")
def stop():
    try:
        bridge.stop_base()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


# -- lift ---------------------------------------------------------------------
def _lift(method: str, *args):
    try:
        bridge.status_lane.call(method, *args)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.post("/lift/up")
def lift_up():
    return _lift("lift_up")


@app.post("/lift/down")
def lift_down():
    return _lift("lift_down")


@app.post("/lift/stop")
def lift_stop():
    return _lift("lift_stop")


@app.post("/lift/home")
def lift_home():
    return _lift("lift_home")


@app.post("/lift/to-height")
def lift_to_height(req: LiftHeightRequest):
    return _lift("lift_to_height", req.target_m)


# -- introspection ------------------------------------------------------------
@app.get("/pose")
def pose():
    p = current_pose()
    if p is None:
        return JSONResponse({"error": "pose unavailable"}, status_code=502)
    return p


@app.get("/encoders")
def encoders():
    e = bridge.status_lane.try_call("get_base_encoders")
    if e is None:
        return JSONResponse({"error": "robot unreachable"}, status_code=502)
    return e


@app.get("/cmd-vel")
def cmd_vel():
    vx, vy, omega = bridge.last_cmd
    return {"vx": vx, "vy": vy, "omega": omega, "t": bridge.last_cmd_t}


@app.get("/battery")
def battery():
    return bridge.battery()


# -- waypoint gate --------------------------------------------------------------
@app.get("/nav/waypoints")
def get_waypoints():
    return {"enabled": waypoint_gate.enabled}


@app.post("/nav/waypoints")
def set_waypoints(req: WaypointsRequest):
    waypoint_gate.set(req.enabled)
    if not req.enabled:
        # Clear any active autonomous path right now; slam_node_ stops sending
        # new ones as soon as it sees the gate flip on its next poll.
        cleared = bridge.status_lane.try_call("follow_path", None) is not None
        print(f"[gateway] waypoints disabled — follow_path(None) {'ok' if cleared else 'FAILED'}")
    else:
        print("[gateway] waypoints enabled")
    return {"ok": True}


# -- camera ---------------------------------------------------------------------
class LaneGuideRequest(BaseModel):
    enabled: bool


@app.get("/camera/info")
def camera_info():
    """ZED intrinsics for the streamed frames — the iPad's lane-guide overlay
    projects ground geometry through these client-side."""
    info = zed.get(CAMERA_INFO_TOPIC)
    if not isinstance(info, dict) or "fx" not in info:
        return JSONResponse({"error": "camera info unavailable"}, status_code=503)
    return {
        "fx": float(info["fx"]), "fy": float(info["fy"]),
        "cx": float(info["cx"]), "cy": float(info["cy"]),
        "width": int(info["width"]), "height": int(info["height"]),
    }


@app.get("/camera/lane-guide")
def get_lane_guide():
    return {"enabled": lane_guide_enabled}


@app.post("/camera/lane-guide")
def set_lane_guide(req: LaneGuideRequest):
    global lane_guide_enabled
    lane_guide_enabled = bool(req.enabled)
    print(f"[gateway] lane guide {'ON' if lane_guide_enabled else 'OFF'}")
    return {"ok": True}


@app.get("/camera/stream")
def camera_stream():
    """MJPEG: multipart/x-mixed-replace at ~CAMERA_FPS (shared producer)."""
    boundary = "yorframe"
    return StreamingResponse(
        broadcaster.frames(boundary),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-cache, no-store"},
    )


# -- telemetry websocket ----------------------------------------------------------
@app.websocket("/telemetry")
async def telemetry(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    interval = 1.0 / TELEMETRY_HZ
    try:
        while True:
            t0 = time.time()

            def fetch():
                # pose: ZED topic (no RPC); lift + battery: cached → ~1 RPC/s.
                return current_pose(), bridge.lift_height(), bridge.battery()

            p, h, batt = await loop.run_in_executor(None, fetch)
            vx, vy, omega = bridge.last_cmd

            frame = {"t": time.time(),
                     "cmd_vel": {"vx": vx, "vy": vy, "omega": omega}}
            if p is not None:
                frame["pose"] = p
            if h is not None:
                frame["lift_height_m"] = h
            if batt.get("percent") is not None:
                frame["battery_percent"] = batt["percent"]
            if batt.get("voltage_v") is not None:
                frame["battery_voltage_v"] = batt["voltage_v"]
            if p is None:
                frame["error"] = bridge.last_error or "pose unavailable"

            await ws.send_text(json.dumps(frame))

            dt = time.time() - t0
            await asyncio.sleep(max(0.0, interval - dt))
    except (WebSocketDisconnect, RuntimeError):
        # Client gone: make sure nothing keeps moving on a dead link.
        bridge.drive_lane.try_call("set_base_velocity", [0.0, 0.0, 0.0])
        print("[gateway] telemetry client disconnected — velocity zeroed")


def main():
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT, log_level="warning")


if __name__ == "__main__":
    main()
