#!/usr/bin/env python3
"""
app_listener.py — the ONLY service running on the robot at boot.

Lightweight discovery + mode orchestrator for the YOR iPad app:

  * Advertises the robot over mDNS/Bonjour as `_yor._tcp.local.` with TXT
    records: name, ip, modes, status.
  * Serves a minimal REST API (port 8090):
        GET  /discovery     {"name", "ip", "modes", "status", "version"}
        GET  /maps          saved .npz maps (newest first)
        POST /mode/start    {"mode": "navigation"|"mapping"|"teleop",
                             "map_name": str?,   # mapping: new map name
                             "load_map": str?,   # mapping: existing .npz file
                             "fresh_map": bool?}
        POST /mode/stop     async graceful stop → poll /mode/status until idle
        GET  /mode/status   {"mode", "status": idle|starting|running|
                             stopping|error, "processes", "detail"}
  * Starts/stops the per-mode scripts lazily in tmux sessions. Idle robot =
    just this process.

Mode → tmux session:
  navigation  "yor-navigation": zed_pub_node + yor_gateway only — camera-only
              driving, NO SLAM / point cloud / mapping, nothing written to
              disk.
  mapping     "yor-mapping": full orbv2 ORB-SLAM3 pipeline (zed_pub_node →
              orb_bridge → orb_slam_node → yor_gateway). New maps save to
              orbv2/maps/<name>_<timestamp>.npz on exit (or are discarded
              via /mode/stop {"save_map": false}); load_map reopens a
              previous .npz.
  teleop      "yor-teleop": robot.teleop.yor_teleop

Run on boot (e.g. systemd or crontab @reboot):
    conda run -n slam-zed python -m robot.app_listener

Env overrides:
  YOR_ROBOT_NAME     advertised name (default: YOR-<hostname>)
  YOR_LISTENER_PORT  default 8090
  YOR_CONDA_ENV      default slam-zed
  YOR_MAPS_DIR       default <repo>/orbv2/maps
  YOR_ADVERTISE_IP   IP handed to the iPad (default: Tailscale IP if up,
                     else LAN IP)
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from commlink import RPCClient
    HAVE_COMMLINK = True
except ImportError:
    HAVE_COMMLINK = False
    print("[app_listener] WARNING: commlink unavailable — battery readout disabled.")

try:
    from zeroconf import ServiceInfo, Zeroconf
    HAVE_ZEROCONF = True
except ImportError:
    HAVE_ZEROCONF = False
    print("[app_listener] WARNING: `pip install zeroconf` for Bonjour discovery. "
          "Manual IP entry on the iPad still works.")

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent     # neer_slam/
LISTENER_PORT = int(os.environ.get("YOR_LISTENER_PORT", "8090"))
CONDA_ENV = os.environ.get("YOR_CONDA_ENV", "slam-zed")
ROBOT_NAME = os.environ.get("YOR_ROBOT_NAME", f"YOR-{socket.gethostname()}")
MAPS_DIR = Path(os.environ.get("YOR_MAPS_DIR", str(PROJECT_DIR / "orbv2" / "maps")))
SENTINEL_DIR = Path("/tmp/yor_listener")
GATEWAY_PORT = 8088
VISER_PORT = 8099
VERSION = "2.1"

MODES = ["navigation", "mapping", "teleop"]

# Shell prefix replicated from nav.sh so panes get the right conda env.
CONDA_PREFIX = (
    "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || "
    "source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true && "
    f"cd {PROJECT_DIR} && conda activate {CONDA_ENV}"
)


def _outbound_ip(probe_addr: str) -> Optional[str]:
    """IP of the interface that would route to probe_addr (no packets sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((probe_addr, 53))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


def get_advertise_ip() -> str:
    """IP to hand to the iPad, in TXT records and /discovery.

    Preference order:
      1. YOR_ADVERTISE_IP env override
      2. Tailscale IP (100.64.0.0/10) — the iPad reaches it from any network,
         including the robot's own WiFi, as long as both are on the tailnet
      3. LAN IP
    """
    override = os.environ.get("YOR_ADVERTISE_IP")
    if override:
        return override
    # 100.100.100.100 is Tailscale's MagicDNS resolver; routing to it selects
    # the tailscale interface if one is up.
    ts = _outbound_ip("100.100.100.100")
    if ts and ts.startswith("100."):
        return ts
    return _outbound_ip("8.8.8.8") or "127.0.0.1"


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ── tmux helpers ──────────────────────────────────────────────────────────
def tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


def session_exists(name: str) -> bool:
    return tmux("has-session", "-t", name).returncode == 0


def kill_session(name: str):
    tmux("kill-session", "-t", name)


def sentinel_path(session: str, pane_idx: int) -> Path:
    return SENTINEL_DIR / f"{session}.{pane_idx}.exited"


def clear_sentinels(session: str):
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    for p in SENTINEL_DIR.glob(f"{session}.*.exited"):
        p.unlink(missing_ok=True)


def pane_command(cmd: str, label: str, session: str, pane_idx: int) -> str:
    """Wrap a mode command: write an exit sentinel when it stops (this is how
    the listener detects crashed processes — `sleep N &&`-style startup
    staggers make pane_current_command useless for liveness), then idle so
    the pane stays readable."""
    sentinel = sentinel_path(session, pane_idx)
    return (
        f"echo '=== {label} ===' && {CONDA_PREFIX} && {cmd}; "
        f'echo "[{label}] exited (code $?)." | tee {sentinel}; sleep 86400'
    )


# ── Base control on the NUC (over SSH) ───────────────────────────────────
NUC_HOST = os.environ.get("YOR_NUC_HOST", "194.168.1.10")
NUC_USER = os.environ.get("YOR_NUC_USER", "robotlab")  # NUC login user (override via env)
YOR_RPC_HOST = os.environ.get("YOR_RPC_HOST", "194.168.1.10")
YOR_RPC_PORT = int(os.environ.get("YOR_RPC_PORT", "5557"))
BASE_SESSION = os.environ.get("YOR_BASE_SESSION", "robot")
BASE_START_CMD = os.environ.get("YOR_BASE_START_CMD",
                                "cd ~/YOR && ./create_windows.sh")
# create_windows.sh deliberately stages the pane commands WITHOUT pressing
# Enter (the stack has a dependency order: CAN setup → driver → joystick).
# After the session is up we press Enter in each pane like a human would,
# pausing between steps. Format: "pane:label:wait_after_s,..."
BASE_PANE_SEQUENCE = os.environ.get(
    "YOR_BASE_PANE_SEQUENCE",
    "0.2:CAN setup:6,0.1:yor.py driver:4,0.0:joystick:0",
)

# Teleop runs on the NUC too, from the YOR_D copy (different conda env). Same
# SSH/tmux pattern as base control, plus we write the iPad relay IP to a file
# the teleop script reads (oculus_bimanual_wholebody_teleop.py _resolve_vr_host).
TELEOP_DIR = os.environ.get("YOR_TELEOP_DIR", "~/YOR_D")
TELEOP_SESSION = os.environ.get("YOR_TELEOP_SESSION", "robot")
TELEOP_START_CMD = os.environ.get(
    "YOR_TELEOP_START_CMD", f"cd {TELEOP_DIR} && ./create_windows.sh")
# If YOR_D's create_windows.sh stages commands without pressing Enter (like the
# base one), list its panes here; the 0.0 pane runs the VR teleop instead of the
# joystick. Empty → assume the script runs everything itself.
TELEOP_PANE_SEQUENCE = os.environ.get(
    "YOR_TELEOP_PANE_SEQUENCE",
    "0.2:CAN setup:6,0.1:yor.py driver:4,0.0:VR teleop:0",
)
# File on the NUC that the teleop script reads to find the iPad relay.
VR_HOST_FILE = os.environ.get("YOR_VR_HOST_FILE", "~/.yor_vr_host")


def _ssh_target() -> str:
    return f"{NUC_USER}@{NUC_HOST}" if NUC_USER else NUC_HOST


def ssh_run(cmd: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
    """Run a command on the NUC (Thor has passwordless SSH to it)."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         _ssh_target(), cmd],
        capture_output=True, text=True, timeout=timeout,
    )


class BaseControl:
    """Start/stop the NUC's motor-control tmux stack over SSH.

    create_windows.sh spins up tmux session BASE_SESSION ("robot") with the
    CAN setup / yor.py driver / joystick panes. Status = does that session
    exist on the NUC (checked over SSH, cached briefly so iPad polling
    doesn't hammer sshd).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.phase = "unknown"     # off | starting | running | error | unknown
        self.detail = ""
        self._busy = False
        self._last_check = 0.0

    def status(self) -> dict:
        with self._lock:
            busy = self._busy
            stale = time.time() - self._last_check > 3.0
        if not busy and stale:
            self._refresh()
        with self._lock:
            return {"status": self.phase, "detail": self.detail}

    def _refresh(self):
        try:
            r = ssh_run(f"tmux has-session -t {BASE_SESSION} 2>/dev/null", timeout=8)
            phase = "running" if r.returncode == 0 else "off"
            detail = ""
        except Exception as e:
            phase, detail = "error", f"NUC unreachable: {e}"
        with self._lock:
            # Don't clobber an in-flight start/stop transition.
            if not self._busy:
                self.phase = phase
                self.detail = detail
            self._last_check = time.time()

    def start(self) -> dict:
        with self._lock:
            if self._busy:
                return {"status": self.phase, "detail": self.detail}
            self._busy = True
            self.phase = "starting"
            self.detail = "launching base stack on NUC…"
        threading.Thread(target=self._start_worker, daemon=True).start()
        return {"status": "starting", "detail": self.detail}

    def _set_detail(self, detail: str):
        with self._lock:
            self.detail = detail

    def _start_worker(self):
        phase, detail = "error", ""
        try:
            # Guard: create_windows.sh has no `set -e` — running it while the
            # session already exists would TYPE stray text into live driver
            # panes. If it's already up, do nothing.
            r = ssh_run(f"tmux has-session -t {BASE_SESSION} 2>/dev/null", timeout=8)
            if r.returncode == 0:
                phase = "running"
                detail = "base stack was already up"
            else:
                # The script ends with `tmux attach`, which fails without a
                # TTY — ignore the exit code and verify via has-session.
                run = ssh_run(f"bash -lc '{BASE_START_CMD}'", timeout=30)
                r = ssh_run(f"tmux has-session -t {BASE_SESSION} 2>/dev/null", timeout=8)
                if r.returncode != 0:
                    # Surface the script's own stderr/stdout so the failure is
                    # diagnosable from the iPad instead of opaque.
                    msg = (run.stderr or run.stdout or "").strip().replace("\n", " ")
                    if not msg:
                        msg = f"exit {run.returncode} (no output — is tmux on the NUC's non-interactive PATH? is the cd path right?)"
                    detail = f"session '{BASE_SESSION}' not created: {msg[:240]}"
                else:
                    # The script stages each pane's command without pressing
                    # Enter. Fire them in dependency order with pauses:
                    # CAN setup → driver → joystick.
                    for step in BASE_PANE_SEQUENCE.split(","):
                        pane, label, wait_s = step.strip().split(":")
                        self._set_detail(f"starting {label}…")
                        ssh_run(
                            f"tmux send-keys -t {BASE_SESSION}:{pane} C-m",
                            timeout=8,
                        )
                        time.sleep(float(wait_s))
                    phase = "running"
                    detail = ""
        except Exception as e:
            detail = f"SSH failed: {e}"
        with self._lock:
            self.phase = phase
            self.detail = detail
            self._busy = False
            self._last_check = time.time()
        print(f"[app_listener] base start → {phase} {detail}")

    def stop(self) -> dict:
        with self._lock:
            if self._busy:
                return {"status": self.phase, "detail": self.detail}
            self._busy = True
            self.phase = "stopping"
            self.detail = "stopping base stack on NUC…"
        threading.Thread(target=self._stop_worker, daemon=True).start()
        return {"status": "stopping", "detail": self.detail}

    def _stop_worker(self):
        phase, detail = "off", ""
        try:
            ssh_run(f"tmux kill-session -t {BASE_SESSION} 2>/dev/null", timeout=15)
        except Exception as e:
            phase, detail = "error", f"SSH failed: {e}"
        with self._lock:
            self.phase = phase
            self.detail = detail
            self._busy = False
            self._last_check = time.time()
        print(f"[app_listener] base stop → {phase} {detail}")


# ── Battery monitor (NUC yor.py RPC) ─────────────────────────────────────
class BatteryMonitor:
    """Background-polled battery state from the NUC's yor.py RPC, so the iPad
    discovery card can show it while the robot is idle.

    Battery only exists when Base Control is up (yor.py running on the NUC),
    so polling is gated on `should_poll()` — when the base is off we never
    touch the RPC (which would just hang). Queries run on a worker with a
    hard timeout; a stuck call abandons the client and rebuilds it, and the
    reading goes stale → None after a while.
    """

    POLL_S = 5.0
    TIMEOUT_S = 2.0
    STALE_S = 20.0

    def __init__(self, host: str, port: int, should_poll):
        self.host = host
        self.port = port
        self.should_poll = should_poll
        self._lock = threading.Lock()
        self._percent: Optional[float] = None
        self._voltage: Optional[float] = None
        self._ts = 0.0
        self._executor: Optional[ThreadPoolExecutor] = None
        self._client = None
        if HAVE_COMMLINK:
            threading.Thread(target=self._loop, daemon=True).start()

    def _query(self):
        if self._client is None:
            self._client = RPCClient(self.host, self.port)
        return (self._client.get_battery_percent(),
                self._client.get_battery_voltage())

    def _loop(self):
        while True:
            time.sleep(self.POLL_S)
            if not self.should_poll():
                # Base is off → no yor.py to ask. Forget the stale reading.
                with self._lock:
                    self._percent = self._voltage = None
                    self._ts = 0.0
                self._client = None
                continue
            self._refresh()

    def _refresh(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="battery")
        future = self._executor.submit(self._query)
        try:
            pct, volt = future.result(timeout=self.TIMEOUT_S)
            with self._lock:
                if pct is not None:
                    self._percent = float(pct)
                if volt is not None:
                    self._voltage = float(volt)
                if pct is not None or volt is not None:
                    self._ts = time.time()
        except FuturesTimeout:
            # Abandon the stuck worker + socket; rebuild next cycle.
            self._executor.shutdown(wait=False)
            self._executor = None
            self._client = None
        except Exception:
            self._client = None

    def snapshot(self) -> dict:
        with self._lock:
            if self._ts and time.time() - self._ts < self.STALE_S:
                return {"percent": self._percent, "voltage_v": self._voltage}
            return {"percent": None, "voltage_v": None}


# ── Maps ──────────────────────────────────────────────────────────────────
def list_maps() -> list[dict]:
    if not MAPS_DIR.is_dir():
        return []
    out = []
    for p in sorted(MAPS_DIR.glob("*.npz"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        st = p.stat()
        out.append({
            "file": p.name,
            "name": p.stem,
            "size_mb": round(st.st_size / 1e6, 1),
            "modified": st.st_mtime,
        })
    return out


def sanitize_map_name(name: Optional[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return cleaned or "map"


# ── Mode definitions ──────────────────────────────────────────────────────
class NucLaunch:
    """Describes a mode that runs on the NUC over SSH (not Thor tmux)."""
    def __init__(self, session: str, start_cmd: str, pane_sequence: str = "",
                 vr_host_file: Optional[str] = None, ipad_ip: Optional[str] = None):
        self.session = session
        self.start_cmd = start_cmd
        self.pane_sequence = pane_sequence
        self.vr_host_file = vr_host_file    # write ipad_ip here before launch
        self.ipad_ip = ipad_ip


class ModeSpec:
    def __init__(self, name: str, session: str, panes: list[tuple[str, str]],
                 required_ports: list[int], pane_delay_s: float = 0.0,
                 stop_grace_s: float = 10.0, detail: str = "",
                 nuc: Optional[NucLaunch] = None):
        self.name = name
        self.session = session
        self.panes = panes                      # [(label, command), ...]
        self.required_ports = required_ports    # all open ⇒ mode "running"
        self.pane_delay_s = pane_delay_s        # stagger pane startup
        self.stop_grace_s = stop_grace_s        # graceful-exit wait on stop
        self.detail = detail
        self.nuc = nuc                          # set → SSH-launched on the NUC


def build_mode_spec(mode: str, fresh_map: Optional[bool] = None,
                    map_name: Optional[str] = None,
                    load_map: Optional[str] = None,
                    ipad_ip: Optional[str] = None) -> ModeSpec:
    if mode == "navigation":
        # Camera-only driving: NO SLAM, no point cloud, no mapping. Just the
        # ZED publisher (RGB feed + pose for telemetry, read directly by the
        # gateway) and the gateway itself. Nothing is written to disk and the
        # ZED area memory is left untouched (no --fresh). This also keeps the
        # Jetson load minimal while joystick driving.
        return ModeSpec(
            name="navigation",
            session="yor-navigation",
            panes=[
                ("ZED Publisher", "python -m robot.zed_pub_node"),
                ("Gateway",       "python -m robot.yor_gateway"),
            ],
            required_ports=[GATEWAY_PORT],
            pane_delay_s=1.0,
        )

    if mode == "mapping":
        # Full orbv2 ORB-SLAM3 loop-closure pipeline (mirrors orbv2/run.sh).
        # Startup staggers: bridge waits 5 s for ZED, slam node waits 30 s for
        # ORB-SLAM3 vocabulary load.
        # --no-orb --no-ekf: waypoint following needs the planner and the
        # base follower to share a world frame. The follower on the NUC
        # localizes from the RAW zed/pose topic, so EKF/ORB corrections in
        # the slam node rotate the planner frame away from it — clicked
        # goals then steer consistently wrong (confirmed in the field:
        # robot always veered left until these flags were added).
        # TODO(future): re-anchor paths into the raw ZED frame at send time
        # in slam_node_._path_sender_loop, then re-enable ORB/EKF here for
        # loop-closure map quality.
        if load_map:
            target = MAPS_DIR / Path(load_map).name
            if not target.is_file():
                raise ValueError(f"map not found: {target.name}")
            slam_cmd = (f"sleep 30 && python -m orbv2.orb_slam_node "
                        f"--predict-hz 5 --no-orb --no-ekf --load --map-path {target}")
            fresh = False if fresh_map is None else fresh_map  # keep area memory
            detail = f"loading map {target.name}"
        else:
            MAPS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%m%d%H%M%S")
            target = MAPS_DIR / f"{sanitize_map_name(map_name)}_{stamp}.npz"
            slam_cmd = (f"sleep 30 && python -m orbv2.orb_slam_node "
                        f"--predict-hz 5 --no-orb --no-ekf --save --map-path {target}")
            fresh = True if fresh_map is None else fresh_map   # new map frame
            detail = f"new map → {target.name}"

        zed_cmd = "python -m robot.zed_pub_node" + (" --fresh" if fresh else "")
        return ModeSpec(
            name="mapping",
            session="yor-mapping",
            panes=[
                ("ZED Publisher", zed_cmd),
                ("ORB Bridge",    "sleep 5 && python -m orbv2.orb_bridge --gen-config"),
                ("SLAM Node",     slam_cmd),
                ("Gateway",       "python -m robot.yor_gateway"),
            ],
            required_ports=[GATEWAY_PORT, VISER_PORT],
            pane_delay_s=1.0,
            stop_grace_s=40.0,   # orb_slam_node saves the map on SIGINT
            detail=detail,
        )

    if mode == "teleop":
        # Teleop runs the YOR_D stack on the NUC over SSH (driver + bimanual
        # oculus teleop). The iPad runs a transparent relay; we write its IP to
        # the NUC so the teleop script subscribes to the iPad instead of the
        # Quest directly. The robot's VR stream therefore flows:
        #   Quest (PUB) → iPad relay → NUC teleop (SUB).
        detail = "launching teleop on NUC"
        if ipad_ip:
            detail += f" (relay {ipad_ip})"
        return ModeSpec(
            name="teleop",
            session=TELEOP_SESSION,   # NUC-side session name
            panes=[],
            required_ports=[],
            stop_grace_s=15.0,
            detail=detail,
            nuc=NucLaunch(
                session=TELEOP_SESSION,
                start_cmd=TELEOP_START_CMD,
                pane_sequence=TELEOP_PANE_SEQUENCE,
                vr_host_file=VR_HOST_FILE,
                ipad_ip=ipad_ip,
            ),
        )
    raise ValueError(f"unknown mode {mode!r}")


# ── Orchestrator ──────────────────────────────────────────────────────────
class Orchestrator:
    """Starts/stops mode tmux sessions and tracks status for /discovery + mDNS."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active: Optional[ModeSpec] = None
        self.phase = "idle"          # idle | starting | running | stopping | error
        self.detail = ""
        self._started_at = 0.0
        self._session_created = False   # start worker has built the tmux session
        self._nuc_alive = False         # cached NUC session liveness (teleop)
        self._nuc_alive_t = 0.0
        self.on_status_change = lambda: None   # hooked by mDNS advertiser

    # -- public status -----------------------------------------------------
    @property
    def status_string(self) -> str:
        """What we advertise: idle | starting | stopping | <mode name>."""
        with self._lock:
            if self.active is None:
                return "idle"
            if self.phase in ("starting", "stopping"):
                return self.phase
            return self.active.name

    def mode_status(self) -> dict:
        with self._lock:
            if self.active is None:
                return {"mode": None, "status": "idle", "processes": [], "detail": ""}
            spec = self.active

            if self.phase == "stopping":
                return {"mode": spec.name, "status": "stopping",
                        "processes": [], "detail": self.detail}

            nuc_spec = spec.nuc

        # NUC-launched modes (teleop): the SSH session check is slow, so do it
        # outside the lock, then reconcile.
        if nuc_spec is not None:
            return self._mode_status_nuc(spec)

        with self._lock:
            procs = self._process_health(spec)

            if not session_exists(spec.session):
                if self.phase == "starting" and not self._session_created:
                    # Start worker is still building the panes.
                    return {"mode": spec.name, "status": "starting",
                            "processes": [], "detail": self.detail}
                self.phase = "error"
                self.detail = "tmux session died"
            elif self.phase == "starting":
                if all(p["alive"] for p in procs) and all(
                    p["port_open"] for p in procs if p["port"] is not None
                ):
                    self.phase = "running"
                    self.detail = ""
                elif any(not p["alive"] for p in procs):
                    dead = [p["name"] for p in procs if not p["alive"]]
                    self.phase = "error"
                    self.detail = f"process died: {', '.join(dead)}"
                else:
                    waiting = [p["name"] for p in procs
                               if p["port"] is not None and not p["port_open"]]
                    self.detail = (f"waiting on: {', '.join(waiting)}"
                                   if waiting else "processes booting…")
                    if spec.name == "mapping" and time.time() - self._started_at < 45:
                        self.detail += " (ORB-SLAM3 init takes ~1 min)"
            elif self.phase == "running":
                if any(not p["alive"] for p in procs):
                    dead = [p["name"] for p in procs if not p["alive"]]
                    self.phase = "error"
                    self.detail = f"process crashed: {', '.join(dead)}"

            return {
                "mode": spec.name,
                "status": self.phase,
                "processes": procs,
                "detail": self.detail,
            }

    def _process_health(self, spec: ModeSpec) -> list[dict]:
        # A pane's process is alive iff its exit sentinel has not been written
        # (works through `sleep N && cmd` startup staggers).
        port_for = {}
        if spec.name == "navigation":
            port_for = {"Gateway": GATEWAY_PORT}
        elif spec.name == "mapping":
            port_for = {"SLAM Node": VISER_PORT, "Gateway": GATEWAY_PORT}

        procs = []
        for i, (label, _cmd) in enumerate(spec.panes):
            alive = not sentinel_path(spec.session, i).exists()
            port = port_for.get(label)
            procs.append({
                "name": label,
                "alive": alive,
                "port": port,
                "port_open": port_open(port) if port is not None else None,
            })
        return procs

    # -- start ---------------------------------------------------------------
    def start(self, mode: str, fresh_map: Optional[bool] = None,
              map_name: Optional[str] = None,
              load_map: Optional[str] = None,
              ipad_ip: Optional[str] = None) -> dict:
        """Validate, mark "starting", and return immediately; tmux session
        creation happens in a worker thread. Creating the panes takes several
        seconds (startup staggers), which is longer than the iPad's HTTP
        timeout — the iPad already polls /mode/status afterwards anyway."""
        with self._lock:
            if self.phase == "stopping":
                raise RuntimeError("previous mode is still stopping — try again shortly")
            if self.active is not None:
                if self.active.name == mode:
                    return {"mode": self.active.name, "status": self.phase,
                            "processes": [], "detail": self.detail}
                raise RuntimeError(
                    f"mode {self.active.name!r} is already active — stop it first")

            # May raise ValueError (unknown mode / missing map) — synchronous
            # so the iPad gets a proper 400 instead of a late error status.
            spec = build_mode_spec(mode, fresh_map=fresh_map,
                                   map_name=map_name, load_map=load_map,
                                   ipad_ip=ipad_ip)

            self.active = spec
            self.phase = "starting"
            self.detail = spec.detail or "launching processes…"
            self._started_at = time.time()
            self._session_created = False

        self.on_status_change()
        threading.Thread(target=self._start_worker, args=(spec,), daemon=True).start()
        print(f"[app_listener] starting mode {mode!r} ({spec.detail or spec.session})")
        return {"mode": spec.name, "status": "starting",
                "processes": [], "detail": self.detail}

    def _start_worker(self, spec: ModeSpec):
        if spec.nuc is not None:
            self._start_worker_nuc(spec)
            return
        try:
            kill_session(spec.session)  # stale leftovers
            clear_sentinels(spec.session)
            first_label, first_cmd = spec.panes[0]
            r = tmux("new-session", "-d", "-s", spec.session,
                     pane_command(first_cmd, first_label, spec.session, 0))
            if r.returncode != 0:
                raise RuntimeError(f"tmux failed: {r.stderr.strip()}")
            for i, (label, cmd) in enumerate(spec.panes[1:], start=1):
                if spec.pane_delay_s:
                    time.sleep(spec.pane_delay_s)
                tmux("split-window", "-t", spec.session,
                     pane_command(cmd, label, spec.session, i))
            tmux("select-layout", "-t", spec.session, "even-vertical")
            with self._lock:
                self._session_created = True
        except Exception as e:
            with self._lock:
                if self.active is spec:
                    self.phase = "error"
                    self.detail = str(e)
            self.on_status_change()

    def _nuc_graceful_kill(self, session: str, grace_s: float):
        """SIGINT every pane (so yor.py's shutdown tucks the arms), wait out the
        grace period, then hard-kill the session. No-op if it doesn't exist."""
        if ssh_run(f"tmux has-session -t {session} 2>/dev/null",
                   timeout=8).returncode != 0:
            return
        try:
            panes = ssh_run(f"tmux list-panes -t {session} -F '#{{pane_id}}'",
                            timeout=8).stdout.strip().splitlines()
            for pane in panes:
                ssh_run(f"tmux send-keys -t {pane} C-c", timeout=8)
            print(f"[app_listener] NUC '{session}': SIGINT sent, waiting {grace_s:.0f}s for arm tuck…")
            time.sleep(grace_s)
            ssh_run(f"tmux kill-session -t {session}", timeout=15)
        except Exception as e:
            print(f"[app_listener] NUC graceful kill failed: {e}")
            ssh_run(f"tmux kill-session -t {session}", timeout=15)

    def _start_worker_nuc(self, spec: ModeSpec):
        """Launch a NUC-side stack (teleop) over SSH: write the iPad relay IP,
        run create_windows.sh in YOR_D, then press the staged pane Enters."""
        nuc = spec.nuc
        try:
            # Gracefully clear a stale NUC session so its arms tuck before we
            # relaunch (e.g. re-entering teleop after a crash/force-quit). In
            # the normal Exit→re-enter flow the stop already tucked, so this
            # finds nothing and adds no delay. (teleop and base both use the
            # NUC 'robot' session + yor.py port 5557 — mutually exclusive.)
            self._nuc_graceful_kill(nuc.session, spec.stop_grace_s)
            time.sleep(0.5)

            # Tell the NUC teleop script where the iPad relay is.
            if nuc.vr_host_file and nuc.ipad_ip:
                ssh_run(f"printf '%s' '{nuc.ipad_ip}' > {nuc.vr_host_file}", timeout=8)
                print(f"[app_listener] teleop relay IP {nuc.ipad_ip} → NUC {nuc.vr_host_file}")

            # Launch (create_windows.sh ends in `tmux attach`, which fails with
            # no TTY — ignore exit code, verify via has-session).
            run = ssh_run(f"bash -lc '{nuc.start_cmd}'", timeout=40)
            if ssh_run(f"tmux has-session -t {nuc.session} 2>/dev/null",
                       timeout=8).returncode != 0:
                msg = (run.stderr or run.stdout or "").strip().replace("\n", " ")
                raise RuntimeError(
                    f"NUC session '{nuc.session}' not created: {msg[:200] or 'no output (check YOR_D path / tmux on PATH)'}")

            # Press Enter in staged panes (dependency order), like base control.
            for step in (s for s in nuc.pane_sequence.split(",") if s.strip()):
                pane, label, wait_s = step.strip().split(":")
                with self._lock:
                    if self.active is spec:
                        self.detail = f"starting {label}…"
                ssh_run(f"tmux send-keys -t {nuc.session}:{pane} C-m", timeout=8)
                time.sleep(float(wait_s))

            with self._lock:
                self._session_created = True
                self._nuc_alive = True
                self._nuc_alive_t = time.time()
        except Exception as e:
            with self._lock:
                if self.active is spec:
                    self.phase = "error"
                    self.detail = str(e)
            self.on_status_change()

    def _nuc_session_alive(self, session: str) -> bool:
        """SSH `has-session` on the NUC, cached 3 s (SSH is slow)."""
        with self._lock:
            if time.time() - self._nuc_alive_t < 3.0:
                return self._nuc_alive
        alive = ssh_run(f"tmux has-session -t {session} 2>/dev/null",
                        timeout=8).returncode == 0
        with self._lock:
            self._nuc_alive = alive
            self._nuc_alive_t = time.time()
        return alive

    def _mode_status_nuc(self, spec: ModeSpec) -> dict:
        alive = self._nuc_session_alive(spec.nuc.session)
        with self._lock:
            if self.active is not spec:
                # Mode changed under us.
                return self.mode_status()
            if self.phase == "starting":
                if not self._session_created:
                    pass  # worker still launching
                elif alive:
                    self.phase = "running"
                    self.detail = ""
                else:
                    self.phase = "error"
                    self.detail = "NUC teleop session exited during startup"
            elif self.phase == "running" and not alive:
                self.phase = "error"
                self.detail = "NUC teleop session exited"
            procs = [{
                "name": "Teleop stack (NUC)",
                "alive": alive,
                "port": None,
                "port_open": None,
            }]
            return {"mode": spec.name, "status": self.phase,
                    "processes": procs, "detail": self.detail}

    # -- stop ------------------------------------------------------------------
    def stop(self, save_map: bool = True) -> dict:
        """Kick an async graceful stop; returns immediately. Poll /mode/status
        until idle. (Map saves on mapping exit can take tens of seconds — far
        longer than the iPad's HTTP timeout.)

        save_map=False (mapping only): the SLAM node is SIGKILLed before the
        graceful shutdown, so its SIGINT save handler never runs and the map
        is discarded.
        """
        with self._lock:
            spec = self.active
            if spec is None:
                return {"mode": None, "status": "idle", "processes": [], "detail": ""}
            if self.phase == "stopping":
                return {"mode": spec.name, "status": "stopping",
                        "processes": [], "detail": self.detail}
            self.phase = "stopping"
            if spec.name == "mapping":
                self.detail = "saving map…" if save_map else "discarding map…"
            else:
                self.detail = "shutting down…"

        self.on_status_change()
        threading.Thread(target=self._stop_worker, args=(spec, save_map),
                         daemon=True).start()
        return {"mode": spec.name, "status": "stopping",
                "processes": [], "detail": self.detail}

    def _stop_worker(self, spec: ModeSpec, save_map: bool = True):
        print(f"[app_listener] stopping mode {spec.name!r} (save_map={save_map})…")

        # NUC-launched modes (teleop): kill the NUC session over SSH. No Thor
        # tmux, no local gateway to zero.
        if spec.nuc is not None:
            # SIGINT panes → yor.py tucks the arms → wait → kill-session.
            self._nuc_graceful_kill(spec.nuc.session, spec.stop_grace_s)
            with self._lock:
                self.active = None
                self.phase = "idle"
                self.detail = ""
                self._nuc_alive = False
                self._nuc_alive_t = 0.0
            self.on_status_change()
            print("[app_listener] robot is idle")
            return

        # Discard requested: kill the SLAM node outright (no SIGINT, no save)
        # before gracefully stopping everything else. Same match pattern as
        # orbv2/run.sh --kill.
        if spec.name == "mapping" and not save_map:
            subprocess.run(["pkill", "-KILL", "-f", "orbv2.orb_slam_node"],
                           capture_output=True)

        # 1) Zero velocity through the gateway before anything dies.
        if spec.name in ("navigation", "mapping", "teleop"):
            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"http://127.0.0.1:{GATEWAY_PORT}/stop", method="POST"),
                    timeout=2,
                )
            except Exception:
                pass

        # 2) SIGINT every pane so processes exit cleanly (orb_slam_node saves
        #    its map, ZED closes its area file).
        if session_exists(spec.session):
            out = tmux("list-panes", "-t", spec.session, "-F", "#{pane_id}")
            for pane in out.stdout.strip().splitlines():
                tmux("send-keys", "-t", pane, "C-c", "")

            deadline = time.time() + spec.stop_grace_s
            n_panes = len(spec.panes)
            while time.time() < deadline:
                exited = sum(sentinel_path(spec.session, i).exists()
                             for i in range(n_panes))
                if exited >= n_panes or not session_exists(spec.session):
                    break
                time.sleep(0.5)

            # 3) Hard kill whatever is left.
            kill_session(spec.session)

        clear_sentinels(spec.session)
        with self._lock:
            self.active = None
            self.phase = "idle"
            self.detail = ""
        self.on_status_change()
        print("[app_listener] robot is idle")


# ── mDNS advertiser ───────────────────────────────────────────────────────
class Advertiser:
    def __init__(self, orchestrator: Orchestrator):
        self.orch = orchestrator
        self.zc: Optional[Zeroconf] = None
        self.info: Optional[ServiceInfo] = None
        self.ip = get_advertise_ip()

    def _build_info(self) -> "ServiceInfo":
        return ServiceInfo(
            "_yor._tcp.local.",
            f"{ROBOT_NAME}._yor._tcp.local.",
            addresses=[socket.inet_aton(self.ip)],
            port=LISTENER_PORT,
            properties={
                "name": ROBOT_NAME,
                "ip": self.ip,
                "modes": ",".join(MODES),
                "status": self.orch.status_string,
                "version": VERSION,
            },
        )

    def start(self):
        if not HAVE_ZEROCONF:
            return
        self.zc = Zeroconf()
        self.info = self._build_info()
        self.zc.register_service(self.info)
        self.orch.on_status_change = self.update
        print(f"[app_listener] advertising {ROBOT_NAME} at {self.ip}:{LISTENER_PORT} (_yor._tcp)")

    def update(self):
        if self.zc is None:
            return
        try:
            new_info = self._build_info()
            self.zc.update_service(new_info)
            self.info = new_info
        except Exception as e:
            print(f"[app_listener] mDNS update failed: {e}")

    def stop(self):
        if self.zc is not None:
            try:
                self.zc.unregister_all_services()
                self.zc.close()
            except Exception:
                pass


# ── FastAPI app ───────────────────────────────────────────────────────────
orch = Orchestrator()
advertiser = Advertiser(orch)
base = BaseControl()
# Battery readout for the idle discovery card. Gate on whether yor.py's RPC
# port is actually open (a fast TCP probe to the NUC) rather than on SSH/tmux
# base detection — battery is readable whenever yor.py is up, regardless of
# how the base was launched, and the probe fails fast when it's down (so the
# RPC query never hangs).
battery = BatteryMonitor(
    YOR_RPC_HOST, YOR_RPC_PORT,
    should_poll=lambda: port_open(YOR_RPC_PORT, host=YOR_RPC_HOST),
)


class ModeStartRequest(BaseModel):
    mode: str
    fresh_map: Optional[bool] = None   # None → per-mode default
    map_name: Optional[str] = None     # mapping: name for a NEW map
    load_map: Optional[str] = None     # mapping: .npz filename to load
    ipad_ip: Optional[str] = None      # teleop: iPad relay IP (else request host)


class ModeStopRequest(BaseModel):
    save_map: Optional[bool] = True    # mapping: False discards the map


app = FastAPI(title="YOR App Listener")


@app.get("/discovery")
def discovery():
    return {
        "name": ROBOT_NAME,
        "ip": advertiser.ip,
        "modes": MODES,
        "status": orch.status_string,
        "version": VERSION,
        "battery": battery.snapshot(),
    }


@app.get("/maps")
def maps():
    return {"maps": list_maps()}


@app.delete("/maps/{file_name}")
def delete_map(file_name: str):
    # Path(...).name strips any directory components — deletions can only
    # ever touch files directly inside MAPS_DIR.
    target = MAPS_DIR / Path(file_name).name
    if target.suffix != ".npz" or not target.is_file():
        return JSONResponse({"ok": False, "error": "map not found"}, status_code=404)
    try:
        target.unlink()
    except OSError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    print(f"[app_listener] deleted map {target.name}")
    return {"ok": True}


@app.get("/mode/status")
def mode_status():
    return orch.mode_status()


@app.post("/mode/start")
def mode_start(req: ModeStartRequest, request: Request):
    if req.mode not in MODES:
        return JSONResponse(
            {"mode": None, "status": "error", "processes": [],
             "detail": f"unknown mode {req.mode!r}"},
            status_code=400)
    # Teleop relay IP: prefer an explicit value, else the IP the iPad connected
    # from (its Tailscale address, which the NUC can also reach).
    ipad_ip = req.ipad_ip or (request.client.host if request.client else None)
    try:
        return orch.start(req.mode, fresh_map=req.fresh_map,
                          map_name=req.map_name, load_map=req.load_map,
                          ipad_ip=ipad_ip)
    except RuntimeError as e:
        return JSONResponse(
            {"mode": orch.active.name if orch.active else None,
             "status": "error", "processes": [], "detail": str(e)},
            status_code=409)
    except ValueError as e:
        return JSONResponse(
            {"mode": None, "status": "error", "processes": [], "detail": str(e)},
            status_code=400)


@app.post("/mode/stop")
def mode_stop(req: Optional[ModeStopRequest] = None):
    save = True if req is None or req.save_map is None else bool(req.save_map)
    return orch.stop(save_map=save)


# ── Base control (NUC motor stack over SSH) ──────────────────────────────
@app.get("/base/status")
def base_status():
    return base.status()


@app.post("/base/start")
def base_start():
    return base.start()


@app.post("/base/stop")
def base_stop():
    return base.stop()


def main():
    if shutil.which("tmux") is None:
        raise SystemExit("[app_listener] tmux is required but not installed")
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    advertiser.start()
    try:
        uvicorn.run(app, host="0.0.0.0", port=LISTENER_PORT, log_level="warning")
    finally:
        # A listener exit means the iPad lost orchestration; return the robot
        # to idle rather than leaving headless processes running. stop() is
        # async — block here so the map save finishes before we exit.
        orch.stop()
        for _ in range(120):
            if orch.status_string == "idle":
                break
            time.sleep(0.5)
        advertiser.stop()


if __name__ == "__main__":
    main()
