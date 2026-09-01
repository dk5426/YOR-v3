"""
test_teleop_status.py — contract tests for the client's status display.

The table straddles two processes: device state comes from an `InputSource`
and robot state from one `get_state()`. What can break silently is the seam
-- a source that stops reporting engagement, a node that stops reporting the
solver's residuals, a redraw that costs an RPC on every 30 Hz tick, or a
`print` that tears the Live apart. None of it needs a robot, a publisher or
a viewer.

    python tests/test_teleop_status.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import mink
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from robot.teleop.status import (SideStatus, SourceStatus, StatusDisplay,
                                 StreamRow)
from robot.teleop.wholebody_teleop import InputSource, OculusSource, TeleopState

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def state() -> TeleopState:
    return TeleopState(
        left_target=mink.SE3.identity(),
        right_target=mink.SE3.from_rotation_and_translation(
            mink.SO3.identity(), np.array([0.3, -0.2, 0.5])),
        lift_target=0.45,
    )


def server_state() -> dict:
    return {
        "left_ee_wxyz_xyz": [1, 0, 0, 0, 0.001, 0.002, -0.003],
        "right_ee_wxyz_xyz": [1, 0, 0, 0, 0.31, -0.19, 0.502],
        "lift": 0.448, "base_xytheta": [0.02, -0.01, 0.3],
        "fix_base": False, "collision_avoidance": True,
        "left_pos_err": 0.0032, "left_ori_err": 0.0041,
        "right_pos_err": 0.0011, "right_ori_err": 0.0002,
        "solved": True, "solve_iters": 4,
        "left_hand_qpos": (np.arange(20) * 0.05).tolist(),
        "right_hand_qpos": (np.arange(20) * 0.02).tolist(),
        "hands": {"sides": ["left", "right"], "backend": "none",
                  "engaged": {"left": True, "right": False},
                  "origin": {"left": "aria", "right": "-"},
                  "sends": {"left": 100, "right": 40}},
    }


def source_status(streams: tuple = ()) -> SourceStatus:
    return SourceStatus(
        sides={"left": SideStatus("ENGAGED"), "right": SideStatus("paused")},
        streams=streams)


STREAMS = (StreamRow("meta", 12, 1.0, 3.2, 8.1, 900.0),
           StreamRow("wuji", 903, 30.1, 11.4, 28.9, 1.4e6))


class _RecordingLive:
    """Stands in for rich.live.Live, keeping the renderable instead of drawing."""

    def __init__(self):
        self.frames = []

    def update(self, renderable):
        self.frames.append(renderable)


def _display(enabled: bool = True) -> tuple[StatusDisplay, _RecordingLive]:
    d = StatusDisplay("test", enabled=enabled)
    live = _RecordingLive()
    d._live = live
    return d, live


# ─────────────────────────────────────────────────────────────────────────────
# The redraw budget
# ─────────────────────────────────────────────────────────────────────────────

def test_one_rpc_per_period() -> None:
    """30 ticks a second must not become 30 get_state() calls a second."""
    d, live = _display()
    calls = []

    def server():
        calls.append(1)
        return server_state()

    st = state()
    for i in range(90):                       # 3 s at 30 Hz
        d.update(100.0 + i / 30.0, st, None, server)
    check("redraw is 1 Hz, not the loop rate", len(calls) == 3, f"{len(calls)} rpcs/3s")
    check("one frame per redraw", len(live.frames) == 3)


def test_disabled_costs_nothing() -> None:
    """Switched off, the display must not reach the server at all."""
    d, _ = _display(enabled=False)
    calls = []
    for i in range(90):
        d.update(100.0 + i / 30.0, state(), None,
                 lambda: (calls.append(1), server_state())[1])
    check("disabled display sends no RPC", not calls, f"{len(calls)} rpcs")


# ─────────────────────────────────────────────────────────────────────────────
# What the rows actually say
# ─────────────────────────────────────────────────────────────────────────────

def _cells(table) -> list[list[str]]:
    return [list(c) for c in zip(*[col._cells for col in table.columns])]


def test_rows_read_the_server() -> None:
    d, live = _display()
    d.update(100.0, state(), source_status(STREAMS), server_state)
    arms, streams, hands = live.frames[0].renderables
    rows = _cells(arms)
    check("a row per arm", len(rows) == 2, f"{len(rows)}")
    check("engagement comes from the source", "ENGAGED" in rows[0][1])
    check("a released side is not styled as engaged", "ENGAGED" not in rows[1][1])
    # target (0,0,0) vs ee (1,2,-3) mm -> |.| = sqrt(14) mm
    check("lag is target vs the EE the node reports",
          abs(float(rows[0][2]) - np.sqrt(14.0)) < 0.05, rows[0][2])
    check("pos/ori are the solver's own residuals, in mm/mrad",
          rows[0][3] == "3.2" and rows[0][4] == "4.1",
          f"{rows[0][3]} {rows[0][4]}")
    check("no pose is printed as xyz",
          [c.header for c in arms.columns] ==
          ["Arm", "State", "lag mm", "pos mm", "ori mrad"])
    srows = _cells(streams)
    check("a row per subscribed topic", len(srows) == 2, f"{len(srows)}")
    check("stream row carries recv/fps/latency",
          srows[1][:5] == ["wuji", "903", "30.1", "11.4", "28.9"], str(srows[1]))
    check("bandwidth is formatted", "Mbps" in srows[1][5], srows[1][5])
    check("fingers come off the same get_state", len(_cells(hands)) == 2)
    # 0.05 rad on joint1 of finger 2 -> index column of the left hand
    check("finger columns are joint1 of each finger, in degrees",
          _cells(hands)[0][4] == f"{np.rad2deg(0.05 * 4):.1f}",
          _cells(hands)[0][4])


def test_send_hz_is_differentiated() -> None:
    """The node reports a total; the rate is the display's job."""
    d, live = _display()
    srv = server_state()
    d.update(100.0, state(), source_status(), lambda: srv)
    first = _cells(live.frames[0].renderables[-1])
    check("no rate on the first redraw -- no interval yet",
          first[0][2] == "--", first[0][2])
    srv = server_state()
    srv["hands"]["sends"] = {"left": 130, "right": 40}
    d.update(102.0, state(), source_status(), lambda: srv)
    row = _cells(live.frames[1].renderables[-1])
    check("send Hz is the count delta over the redraw interval",
          row[0][2] == "15.0", row[0][2])
    check("an idle hand reads 0.0, not blank", row[1][2] == "0.0", row[1][2])
    check("hand state is the node's engagement, not the arm clutch",
          "ENGAGED" in row[0][1] and "held" in row[1][1])
    # A restarted node hands back a smaller total than last time
    srv = server_state()
    srv["hands"]["sends"] = {"left": 3, "right": 0}
    d.update(104.0, state(), source_status(), lambda: srv)
    row = _cells(live.frames[2].renderables[-1])
    check("a counter that went backwards is not a negative rate",
          row[0][2] == "--", row[0][2])


def test_stream_table_is_source_driven() -> None:
    """No stream rows, no table -- Quest has no publisher clock to measure."""
    d, live = _display()
    d.update(100.0, state(), source_status(), server_state)
    check("no stream table without stream rows",
          len(live.frames[0].renderables) == 2)
    d.update(102.0, state(), source_status(STREAMS), server_state)
    check("stream table appears when the source reports rows",
          len(live.frames[1].renderables) == 3)


def test_cold_and_unsynced_streams() -> None:
    """A partial window and a failed clock handshake must not invent numbers."""
    d, live = _display()
    cold = (StreamRow("wuji", 12, 4.0, None, None, 900.0, warm=False),)
    d.update(100.0, state(), source_status(cold), server_state)
    row = _cells(live.frames[0].renderables[1])[0]
    check("a cold window reads dashes, not a low rate",
          row[2:] == ["--", "--", "--", "--"], str(row))
    warm_unsynced = (StreamRow("wuji", 900, 30.0, None, None, 1.4e6),)
    d.update(102.0, state(), source_status(warm_unsynced), server_state)
    row = _cells(live.frames[1].renderables[1])[0]
    check("rate survives an unsynced clock; latency does not",
          row[2] == "30.0" and row[3] == "--" and row[4] == "--", str(row))


def test_survives_a_dead_server() -> None:
    """A failed get_state() must read as unavailable, not as zero error."""
    d, live = _display()
    d.update(100.0, state(), source_status(STREAMS), lambda: None)
    arms = live.frames[0].renderables[0]
    rows = _cells(arms)
    check("no hand table without server state",
          len(live.frames[0].renderables) == 2)
    check("robot-side columns go to dashes", rows[0][-1] == "--", rows[0][-1])
    check("the caption says so, rather than showing stale numbers",
          "unavailable" in str(arms.caption))
    check("device-side engagement still shown", "ENGAGED" in rows[0][1])
    check("the stream table is unaffected -- it is not the server's",
          len(_cells(live.frames[0].renderables[1])) == 2)


def test_missing_solver_keys() -> None:
    """An older node that reports no residuals still renders."""
    d, live = _display()
    srv = server_state()
    for k in ("left_pos_err", "left_ori_err", "solve_iters", "solved"):
        srv.pop(k)
    d.update(100.0, state(), None, lambda: srv)
    rows = _cells(live.frames[0].renderables[0])
    check("absent residuals are dashes, not a crash", rows[0][3] == "--")
    check("lag still renders -- the client measures that itself",
          rows[0][2] != "--", rows[0][2])


# ─────────────────────────────────────────────────────────────────────────────
# The seam with the sources and the nodes
# ─────────────────────────────────────────────────────────────────────────────

def test_source_contract() -> None:
    check("InputSource.status() defaults to nothing to say",
          InputSource().status() is None)
    src = OculusSource.__new__(OculusSource)
    src._engaged = {"left": True, "right": False}
    st = src.status()
    check("OculusSource reports engagement per side",
          st.sides["left"].state == "ENGAGED"
          and st.sides["right"].state != "ENGAGED")
    check("a transport with no publisher clock reports no stream rows",
          st.streams == ())


def test_aria_source_fills_status() -> None:
    """AriaSource must fill _status on every branch of its update loop."""
    src = Path(_REPO / "robot/teleop/aria/source.py").read_text()
    tree = ast.parse(src)
    update = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "update")
    writes = [n for n in ast.walk(update)
              if isinstance(n, ast.Subscript)
              and isinstance(n.value, ast.Attribute)
              and n.value.attr == "_status"]
    check("update() writes a status row on the tracked and untracked branch",
          len(writes) >= 2, f"{len(writes)} writes")


def test_node_reports_residuals() -> None:
    """The sim node carries the residual keys the table reads.

    The hardware controller deliberately does not -- those columns read `--`
    against `yor.py`, and the client measures `lag` for itself either way.
    """
    keys = ("left_pos_err", "left_ori_err", "right_pos_err", "right_ori_err",
            "solve_iters")
    src = Path(_REPO / "robot/yor_mujoco.py").read_text()
    missing = [k for k in keys if f'"{k}"' not in src]
    check("yor_mujoco.py reports the solver's residuals", not missing,
          ", ".join(missing))


def test_both_nodes_report_hand_sends() -> None:
    """Send Hz needs `hands` on both nodes -- it is the same `Hands` object."""
    for path in ("robot/yor_mujoco.py", "robot/yor.py"):
        src = Path(_REPO / path).read_text()
        check(f"{Path(path).name} reports the hand bookkeeping",
              '"hands"' in src and "get_hand_state" in src)


def test_no_bare_prints_on_the_live_path() -> None:
    """A bare print() tears the Live apart -- these files use log()."""
    for path in ("robot/teleop/aria/source.py", "robot/teleop/status.py"):
        tree = ast.parse(Path(_REPO / path).read_text())
        prints = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "print"]
        check(f"{Path(path).name} has no bare print()", not prints,
              f"{len(prints)} left")


def test_stats_tracker() -> None:
    """Latency stays absent until the clock handshake lands."""
    from robot.teleop.aria.stats import StreamStats, fmt_bw

    st = StreamStats(("wuji",))
    for _ in range(4):
        st.hit("wuji", t_wall=1000.0, t_recv=1000.05, bytes_n=100)
    cnt, _, p50, _, _, _ = st.snapshot()["wuji"]
    check("hits are counted before any sync", cnt == 4, str(cnt))
    check("no latency without a clock offset", p50 is None)

    st.set_wall_offset(0.0)
    for _ in range(4):
        st.hit("wuji", t_wall=1000.0, t_recv=1000.05, bytes_n=100)
    _, _, p50, _, _, _ = st.snapshot()["wuji"]
    check("latency is recv minus publish, in ms",
          p50 is not None and abs(p50 - 50.0) < 1e-6, str(p50))
    check("a publisher ahead of us does not read as negative latency",
          _offset_sign_is_subtracted())
    check("bandwidth switches units at 1 Mbps",
          fmt_bw(2e5).endswith("Mbps") and fmt_bw(2e4).endswith("kbps"),
          f"{fmt_bw(2e5)} / {fmt_bw(2e4)}")


def _offset_sign_is_subtracted() -> bool:
    """pub_wall = sub_wall + offset, so the offset comes off t_wall."""
    from robot.teleop.aria.stats import StreamStats

    st = StreamStats(("wuji",))
    st.set_wall_offset(0.100)          # publisher's clock runs 100 ms ahead
    for _ in range(4):
        st.hit("wuji", t_wall=1000.100, t_recv=1000.05)
    _, _, p50, _, _, _ = st.snapshot()["wuji"]
    return p50 is not None and abs(p50 - 50.0) < 1e-6


def main() -> int:
    for test in (
        test_one_rpc_per_period,
        test_disabled_costs_nothing,
        test_rows_read_the_server,
        test_send_hz_is_differentiated,
        test_stream_table_is_source_driven,
        test_cold_and_unsynced_streams,
        test_survives_a_dead_server,
        test_missing_solver_keys,
        test_source_contract,
        test_aria_source_fills_status,
        test_node_reports_residuals,
        test_both_nodes_report_hand_sends,
        test_no_bare_prints_on_the_live_path,
        test_stats_tracker,
    ):
        print(f"\n{test.__name__.removeprefix('test_').replace('_', ' ')}")
        test()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    failures = [name for name, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
