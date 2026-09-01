"""status.py — the teleop client's 1 Hz Rich status display.

The client is the only process holding every half of the picture. Device state
(clutch engagement, and how the subscription is actually performing) lives in
an `InputSource` and is on nobody's RPC surface; robot state arrives as one
`get_state()`, which both nodes build as a single-instant snapshot of arms
*and* fingers. sim_viz.py renders its own version of this from one process.

Three tables, and they answer three different questions:

  Arms    is the robot reaching what it was told to reach
  Stream  is the operator's data arriving, and how late
  Hands   are the fingers being commanded, and how often

Everything redraws at 1 Hz, not the 30 Hz loop rate: these are numbers to
read, and a table that changes 30 times a second cannot be read. `get_state()`
is called only on a redraw, so the display costs one extra RPC per second
against a socket already carrying 30 target writes.

Nothing here shows a pose as xyz. The client already knows where it aimed and
the node reports where the arm got to; what is worth a column is the distance
between them, which is one number instead of six.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table

from robot.teleop.aria.stats import fmt_bw

# One console for the whole client: a bare print() tears a Live table apart,
# so every teleop line goes through `log` below and scrolls above it instead.
console = Console()

# joint1 of each finger in the canonical (20,) vector -- the MCP flexion, the
# one angle per finger that reads as "how curled is this" at a glance
_MCP_ADRS = [0, 4, 8, 12, 16]
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


class SideStatus(NamedTuple):
    """What an InputSource says about one arm."""
    state: str                        # "ENGAGED" / "paused" / "no track" / "--"


class StreamRow(NamedTuple):
    """One subscribed topic, as the client's own receive threads have seen it.

    `p50`/`p95` are None until the clock-sync handshake lands: publisher and
    subscriber stamp with their own wall clocks, and an uncorrected difference
    is dominated by the offset between them rather than by the link.
    """
    name: str
    recv: int
    fps: float
    p50_ms: float | None
    p95_ms: float | None
    bps: float
    warm: bool = True


class SourceStatus(NamedTuple):
    """Everything an InputSource contributes to the display."""
    sides: dict[str, SideStatus] | None = None
    streams: tuple[StreamRow, ...] = ()


def log(msg: str, style: str = "cyan", prefix: str = "teleop") -> None:
    """One prefixed line through the shared console.

    The prefix is escaped because Rich would otherwise read `[teleop]` as a
    style tag.
    """
    console.print(rf"[dim]\[{prefix}][/dim] [{style}]{msg}[/{style}]")


def fmt_xyz(v) -> str:
    """One xyz triple, or dashes when there is nothing to show.

    Unused by the tables here -- a pose is six digits saying what a single
    distance says better. It stays because `sim_viz.py`, which does show
    poses against an overlay you can see them next to, formats them this way.
    """
    return ("     --      --      --" if v is None
            else " ".join(f"{x:+.4f}" for x in np.asarray(v, dtype=float)))


def _fmt(x, spec: str = ".1f") -> str:
    return "--" if x is None else format(x, spec)


def _arm_table(rows: list[dict], banner: str, caption: str) -> Table:
    """How far each arm is from what it was told to do, two ways.

    `lag` and `pos` are different measurements and are both here on purpose.
    `lag` is this client's target against the EE the node reports, so it
    carries the dispatch hop and anything the hardware controller leashes or
    deadbands. `pos` is the solver's own residual against the target it was
    given. On a healthy sim run they agree; on hardware they come apart, and
    that gap is where to look.
    """
    table = Table(title=banner, caption=caption or None, expand=False)
    table.add_column("Arm", style="cyan", no_wrap=True, width=6)
    table.add_column("State", no_wrap=True, width=9)
    table.add_column("lag mm", style="yellow", justify="right", width=8)
    table.add_column("pos mm", style="magenta", justify="right", width=8)
    table.add_column("ori mrad", style="magenta", justify="right", width=9)
    for r in rows:
        table.add_row(r["side"], _state(r["state"]),
                      _fmt(r["lag"]), _fmt(r["pos_err"]), _fmt(r["ori_err"]))
    return table


def _stream_table(rows: tuple[StreamRow, ...]) -> Table:
    """The subscription itself: is the data arriving, at what rate, how late.

    Same columns as aria2robot's own subscriber table, so a number read here
    is comparable with one read there. `Recv` is cumulative; everything else
    is over a trailing 5 s window, and reads `--` until that window has filled
    rather than showing a rate averaged over a partial one.
    """
    table = Table(expand=False,
                  caption="5 s window; latency needs the clock handshake")
    table.add_column("Stream", style="cyan", no_wrap=True, width=8)
    table.add_column("Recv", style="magenta", justify="right", width=8)
    table.add_column("FPS", style="green", justify="right", width=7)
    table.add_column("p50 ms", style="yellow", justify="right", width=8)
    table.add_column("p95 ms", style="yellow", justify="right", width=8)
    table.add_column("Bandwidth", style="cyan", justify="right", width=11)
    for r in rows:
        table.add_row(
            r.name, f"{r.recv}",
            _fmt(r.fps) if r.warm else "--",
            _fmt(r.p50_ms) if r.warm else "--",
            _fmt(r.p95_ms) if r.warm else "--",
            fmt_bw(r.bps) if r.warm else "--",
        )
    return table


def _hand_table(rows: list[dict]) -> Table:
    """The fingers, as the node reports them.

    They never travel over this client's RPC -- `Hands` reads the same
    publisher on its own thread inside the node -- but `get_state()` reports
    them, so one call is arms and fingers at one instant.

    `Send Hz` is writes reaching the driver, not the rate the publisher sends
    at: identical vectors are not resent, so a still hand reads 0.0 while
    engaged, and `held` means the pose stopped changing, never that the hand
    opened.
    """
    table = Table(expand=False,
                  caption="Send Hz is writes to the driver; fingers are "
                          "joint1, MCP flexion, degrees")
    table.add_column("Hand", style="cyan", no_wrap=True, width=6)
    table.add_column("State", no_wrap=True, width=9)
    table.add_column("Send Hz", style="green", justify="right", width=8)
    for name in _FINGERS:
        table.add_column(name, style="blue", justify="right", width=6)
    for r in rows:
        table.add_row(r["side"], _state(r["state"]), r["send_hz"], *r["mcp"])
    return table


def _state(state: str) -> str:
    return (f"[green]{state}[/green]" if state == "ENGAGED"
            else f"[yellow]{state}[/yellow]")


class StatusDisplay:
    """The 1 Hz Live tables, or a no-op when switched off.

    Switched off it still costs nothing: `update()` returns before calling the
    `server` callable, so no extra `get_state()` goes out either.
    """

    def __init__(self, banner: str, enabled: bool = True,
                 period_s: float = 1.0):
        self.banner = banner
        self.enabled = bool(enabled)
        self.period_s = float(period_s)
        self._live: Live | None = None
        self._last: float | None = None
        self._ticks = 0
        # Cumulative driver-write counts at the previous redraw. The node
        # reports a total, not a rate, because a total cannot be wrong about
        # the interval it was measured over -- the rate is this display's job.
        self._sends: dict[str, int] = {}

    def __enter__(self) -> StatusDisplay:
        if self.enabled:
            self._live = Live(_arm_table([], self.banner, ""),
                              console=console, refresh_per_second=2)
            self._live.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None
        return False

    def update(self, now: float, state, device: SourceStatus | None,
               server: Callable[[], dict | None]) -> None:
        """One loop tick; redraws at most every `period_s`."""
        self._ticks += 1
        if not self.enabled or self._live is None:
            return
        if self._last is not None and now - self._last < self.period_s:
            return
        elapsed = None if self._last is None else max(now - self._last, 1e-9)
        hz = None if elapsed is None else self._ticks / elapsed
        self._last, self._ticks = now, 0

        srv = server() or {}
        device = device or SourceStatus()
        sides = device.sides or {}

        rows = []
        for side in ("left", "right"):
            dev = sides.get(side)
            target = getattr(state, f"{side}_target").translation()
            pose = srv.get(f"{side}_ee_wxyz_xyz")
            # wxyz_xyz: rotation first, so the translation is the last three
            ee = None if pose is None else np.asarray(pose, dtype=float)[4:]
            rows.append({
                "side": side,
                "state": "--" if dev is None else dev.state,
                "lag": (None if ee is None
                        else float(np.linalg.norm(target - ee)) * 1e3),
                "pos_err": _scaled(srv.get(f"{side}_pos_err"), 1e3),
                "ori_err": _scaled(srv.get(f"{side}_ori_err"), 1e3),
            })

        tables = [_arm_table(rows, self.banner, _caption(state, srv, hz))]
        if device.streams:
            tables.append(_stream_table(device.streams))
        hands = self._hand_rows(srv, elapsed)
        if hands:
            tables.append(_hand_table(hands))
        self._live.update(Group(*tables))

    def _hand_rows(self, srv: dict, elapsed: float | None) -> list[dict]:
        """Per-hand rows, differentiating the node's cumulative send counts."""
        hands = srv.get("hands") or {}
        engaged = hands.get("engaged") or {}
        sends = hands.get("sends") or {}
        rows = []
        for side in ("left", "right"):
            qpos = srv.get(f"{side}_hand_qpos")
            if qpos is None:
                continue
            mcp = np.rad2deg(np.asarray(qpos, dtype=float)[_MCP_ADRS])
            total = sends.get(side)
            prev = self._sends.get(side)
            # First redraw has no interval to divide by, and a restarted node
            # can hand back a smaller total than last time
            rate = (None if total is None or prev is None or elapsed is None
                    or total < prev else (total - prev) / elapsed)
            if total is not None:
                self._sends[side] = total
            rows.append({
                "side": side,
                "state": ("ENGAGED" if engaged.get(side)
                          else ("held" if side in engaged else "--")),
                "send_hz": _fmt(rate),
                "mcp": [f"{a:.1f}" for a in mcp],
            })
        return rows


def _scaled(x, factor: float) -> float | None:
    return None if x is None else float(x) * factor


def _caption(state, srv: dict, hz: float | None) -> str:
    """The scalars that need one line rather than a column each.

    An empty `srv` means the `get_state()` for this redraw failed, which is
    worth saying: every robot-side column above is then stale, not zero.
    """
    if not srv:
        return "[red]server state unavailable[/red]"
    base = srv.get("base_xytheta") or [float("nan")] * 3
    solved = srv.get("solved")
    return (
        f"lift {srv.get('lift', float('nan')):.3f} m "
        f"(target {state.lift_target:.3f})   "
        f"base ({base[0]:+.2f}, {base[1]:+.2f}, {base[2]:+.2f})   "
        f"fix_base={srv.get('fix_base')} col={srv.get('collision_avoidance')}   "
        f"solved={'--' if solved is None else solved} "
        f"iters={srv.get('solve_iters', '--')}   "
        f"{'--' if hz is None else f'{hz:.1f}'} Hz"
    )
