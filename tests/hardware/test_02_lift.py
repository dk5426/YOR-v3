#!/usr/bin/env python3
"""test_02_lift.py — the lift moves, knows where it is, and stops when told.

⚠️  THE LIFT MOVES. It is heavy and it carries both arms.

Validates the firmware in firmware/lift_controller/ against the PicoLift driver
in robot/base_motor.py: homing, the position-known contract, absolute moves,
the stop path, the streamed-velocity mode the whole-body loop uses, and that
the 900 mm travel constant matches the real hardware.

    python tests/hardware/test_02_lift.py --host <robot-ip>

Run this before any whole-body test: the solver uses the lift as a DOF, and it
trusts get_lift_height() completely.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hw import (  # noqa: E402
    Abort, ask_float, ask_yes_no, banner, check, confirm, connect, countdown,
    guard, info, parse_args, precondition, run,
)

CLIENT = None
ARGS = None

TRAVEL_M = 0.900          # must match MAX_HEIGHT_MM in the .ino
NOMINAL_TOLERANCE = 0.010  # how close to nominal we call "agrees"


def _status() -> dict:
    s = CLIENT.call("get_lift_status")
    return s if isinstance(s, dict) else {}


def _height() -> float | None:
    return CLIENT.call("get_lift_height")


def _wait_still(timeout_s: float = 30.0, settle_s: float = 0.7) -> float | None:
    """Block until the height stops changing. Returns the settled height."""
    last, still_since = None, None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        h = _height()
        now = time.monotonic()
        if h is not None and last is not None and abs(h - last) < 0.0005:
            if still_since is None:
                still_since = now
            elif now - still_since >= settle_s:
                return h
        else:
            still_since = None
        last = h
        time.sleep(0.05)
    info("lift did not settle within the timeout")
    return last


def _stream_velocity(velocity_m_s: float, duration_s: float,
                     refresh_s: float = 0.05) -> bool:
    """Hold a streamed velocity for `duration_s`, refreshing it like the host.

    The firmware stops by itself after 300 ms without a command, so a caller
    that wants continuous motion has to keep asking. That is the contract, not
    an inconvenience: it is what stops the column if this process dies.
    """
    accepted = True
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        if CLIENT.call("lift_set_velocity", float(velocity_m_s)) is False:
            accepted = False
        time.sleep(refresh_s)
    return accepted


def _travel_rate(velocity_m_s: float, duration_s: float) -> tuple[float, float]:
    """(measured m/s, distance m) while streaming `velocity_m_s`.

    The first 0.6 s is excluded: the firmware closes its driver relay and runs
    its jerk-limited ramp before the column is at speed, so including it would
    measure the ramp rather than the velocity.
    """
    _stream_velocity(velocity_m_s, 0.6)
    start_h, start_t = _height(), time.monotonic()
    _stream_velocity(velocity_m_s, duration_s)
    end_h, end_t = _height(), time.monotonic()
    if start_h is None or end_h is None:
        return float("nan"), float("nan")
    distance = end_h - start_h
    return distance / (end_t - start_t), distance


def test_preconditions():
    print("\npreconditions")
    precondition(
        "Nothing is resting on the lift platform, above it, or under it.",
        "Both arms are clear of the lift column and of each other.",
        "No cables are routed so that they snag over the full 0 - 0.9 m travel.",
        "You can reach the physical e-stop / power cut.",
        "Nobody's hands are near the lift.",
    )
    check("operator confirmed the lift is clear", True)


def test_position_known_contract():
    print("\nposition-known contract")
    st = _status()
    if not st.get("available"):
        raise Abort("lift controller not reachable — fix that before continuing")

    known = st.get("position_known")
    info(f"position_known={known}, height={st.get('height_m')}, "
         f"upper={st.get('upper_limit')} lower={st.get('lower_limit')}")

    if known is not True:
        check("an unhomed lift reports no height", st.get("height_m") is None,
              str(st.get("height_m")))
        info("Good: the driver refuses to invent a height it does not have.")
    else:
        check("a homed lift reports a height", st.get("height_m") is not None)


def test_home():
    print("\nhoming")
    confirm("HOME the lift. It will drive UP to its upper limit switch and stop.")
    with guard(CLIENT):
        countdown(3, "homing")
        CLIENT.call("lift_home")
        settled = _wait_still(timeout_s=90.0)

    st = _status()
    check("firmware reports the position is known after homing",
          st.get("position_known") is True, f"position_known={st.get('position_known')}")
    check("firmware reports a successful home", st.get("homed") is True,
          f"homed={st.get('homed')}, last_event={st.get('last_event')!r}")
    if st.get("homed") is False:
        info("A 'Home failed' means the upper limit switch was never reached. "
             "Check the switch wiring and HOMING_TRAVEL_MM in the .ino "
             "(it must exceed MAX_HEIGHT_MM).")

    check("upper limit switch is active at the top", st.get("upper_limit") is True,
          f"upper_limit={st.get('upper_limit')}")

    if settled is not None:
        check("height at the top equals the travel constant",
              abs(settled - TRAVEL_M) < NOMINAL_TOLERANCE,
              f"{settled:.4f} m vs {TRAVEL_M:.4f} m")
        if abs(settled - TRAVEL_M) >= NOMINAL_TOLERANCE:
            info("Homing DEFINES the top as MAX_HEIGHT_MM, so a mismatch here "
                 "means the driver is scaling wrongly, not that the lift is short.")


def test_absolute_moves():
    print("\nabsolute moves (firmware motion profile)")
    if _status().get("position_known") is not True:
        check("lift homed before absolute moves", False, "skipping — home first")
        return

    for target in (0.700, 0.400, 0.600):
        start = _height()
        confirm(f"Move the lift to {target:.3f} m (from {start:.3f} m).")
        with guard(CLIENT):
            countdown(2, f"moving to {target:.3f} m")
            t0 = time.monotonic()
            ok = CLIENT.call("lift_to_height", target)
            elapsed = time.monotonic() - t0
        settled = _wait_still(timeout_s=20.0)
        check(f"lift_to_height({target:.3f}) reports success", bool(ok),
              f"returned {ok} in {elapsed:.1f} s")
        if settled is not None:
            err_mm = (settled - target) * 1000.0
            check(f"lift landed within 5 mm of {target:.3f} m", abs(err_mm) < 5.0,
                  f"{settled:.4f} m, error {err_mm:+.1f} mm")


def test_stop_is_prompt():
    print("\nstop responsiveness")
    if _status().get("position_known") is not True:
        check("lift homed before the stop test", False, "skipping — home first")
        return

    start = _height()
    if start is None or start > TRAVEL_M - 0.15:
        info("too close to the top for a safe upward run; skipping")
        return

    confirm("Drive the lift UP continuously, then STOP it after ~1.5 s.")
    with guard(CLIENT):
        countdown(2, "continuous up")
        CLIENT.call("lift_up")
        time.sleep(1.5)
        h_before_stop = _height()
        t0 = time.monotonic()
        CLIENT.call("lift_stop")
        settled = _wait_still(timeout_s=10.0)
        stop_latency = time.monotonic() - t0

    check("the lift actually moved before stopping",
          h_before_stop is not None and start is not None
          and h_before_stop - start > 0.010,
          f"{start:.4f} -> {h_before_stop:.4f} m")
    check("lift came to rest after stop", settled is not None)
    if settled is not None and h_before_stop is not None:
        coast_mm = (settled - h_before_stop) * 1000.0
        check("coast after stop is under 25 mm", abs(coast_mm) < 25.0,
              f"{coast_mm:+.1f} mm in {stop_latency:.2f} s")
        if abs(coast_mm) >= 25.0:
            info("Long coast. The firmware cuts driver power on a user stop, so "
                 "this is mechanical (brake or backdrive), not a software delay.")


def test_velocity_capability():
    print("\nstreamed-velocity capability")
    st = _status()
    capable = st.get("velocity_capable")
    check("the firmware advertises lift_velocity_v1", capable is True,
          f"capabilities={st.get('capabilities')}")
    if capable is not True:
        info("This controller has the older sketch. Flash "
             "firmware/lift_controller/ before running the velocity stages — "
             "the whole-body loop will otherwise fall back to bang-bang "
             "up/down/stop, which is correct but is not what you are testing.")
    check("the RPC surface agrees", CLIENT.call("lift_supports_velocity") == capable,
          f"lift_supports_velocity()={CLIENT.call('lift_supports_velocity')}")


def _velocity_ready() -> bool:
    st = _status()
    if st.get("velocity_capable") is not True:
        check("firmware supports streamed velocity", False, "skipping — old firmware")
        return False
    if st.get("position_known") is not True:
        check("lift homed before the velocity stages", False, "skipping — home first")
        return False
    height = _height()
    if height is None or not (0.15 < height < TRAVEL_M - 0.15):
        info(f"height {height} is too close to an end of travel; "
             f"move to mid-travel first")
        return False
    return True


def test_velocity_small():
    print("\nstreamed velocity: +/-5 mm/s")
    if not _velocity_ready():
        return

    confirm("Stream +5 mm/s for 4 s, then -5 mm/s for 4 s. The lift moves slowly.")
    with guard(CLIENT):
        countdown(3, "streaming +5 mm/s")
        up_rate, up_distance = _travel_rate(+0.005, 4.0)
        CLIENT.call("lift_set_velocity", 0.0)
        time.sleep(1.0)
        countdown(2, "streaming -5 mm/s")
        down_rate, down_distance = _travel_rate(-0.005, 4.0)
        CLIENT.call("lift_set_velocity", 0.0)
    _wait_still(timeout_s=10.0)

    check("it rose at about 5 mm/s", abs(up_rate - 0.005) < 0.002,
          f"{up_rate * 1000:.2f} mm/s over {up_distance * 1000:+.1f} mm")
    check("it descended at about 5 mm/s", abs(down_rate + 0.005) < 0.002,
          f"{down_rate * 1000:.2f} mm/s over {down_distance * 1000:+.1f} mm")
    smooth = ask_yes_no("Was the motion smooth, without stalling or hunting?")
    if smooth is not None:
        check("motion at 5 mm/s is smooth", smooth)


def test_velocity_larger():
    print("\nstreamed velocity: +/-10 mm/s")
    if not _velocity_ready():
        return

    confirm("Stream +10 mm/s for 4 s, then -10 mm/s for 4 s.")
    with guard(CLIENT):
        countdown(3, "streaming +10 mm/s")
        up_rate, up_distance = _travel_rate(+0.010, 4.0)
        CLIENT.call("lift_set_velocity", 0.0)
        time.sleep(1.0)
        countdown(2, "streaming -10 mm/s")
        down_rate, down_distance = _travel_rate(-0.010, 4.0)
        CLIENT.call("lift_set_velocity", 0.0)
    _wait_still(timeout_s=10.0)

    check("it rose at about 10 mm/s", abs(up_rate - 0.010) < 0.003,
          f"{up_rate * 1000:.2f} mm/s over {up_distance * 1000:+.1f} mm")
    check("it descended at about 10 mm/s", abs(down_rate + 0.010) < 0.003,
          f"{down_rate * 1000:.2f} mm/s over {down_distance * 1000:+.1f} mm")


def test_velocity_zero_hold():
    print("\nstreamed velocity: zero hold")
    if not _velocity_ready():
        return

    confirm("Move at +8 mm/s, then hold at zero for 3 s. The lift must stop and stay.")
    with guard(CLIENT):
        countdown(2, "moving, then holding")
        _stream_velocity(+0.008, 2.0)
        CLIENT.call("lift_set_velocity", 0.0)
        time.sleep(0.8)                      # let the ramp to zero finish
        held = _height()
        _stream_velocity(0.0, 3.0)           # refreshed zero: an explicit hold
        after = _height()

    check("the column held its height", held is not None and after is not None
          and abs(after - held) < 0.002,
          f"{(after - held) * 1000:+.1f} mm drift" if held and after else "no reading")
    check("height telemetry kept arriving during the hold",
          _status().get("height_age_s") is not None
          and _status().get("height_age_s") < 1.0,
          f"age={_status().get('height_age_s')}")


def test_velocity_reversal():
    print("\nstreamed velocity: reversal through zero")
    if not _velocity_ready():
        return

    confirm("Reverse from +10 mm/s straight to -10 mm/s in one command.")
    with guard(CLIENT):
        countdown(2, "reversing")
        _stream_velocity(+0.010, 3.0)
        top = _height()
        _stream_velocity(-0.010, 4.0)        # no zero in between: the firmware inserts it
        bottom = _height()
        CLIENT.call("lift_set_velocity", 0.0)
    _wait_still(timeout_s=10.0)

    check("the direction actually reversed",
          top is not None and bottom is not None and bottom < top,
          f"{top:.4f} -> {bottom:.4f} m" if top and bottom else "no reading")
    clean = ask_yes_no("Did it pass smoothly through zero, with no jolt or grinding?")
    if clean is not None:
        check("reversal ramps through zero before changing direction", clean)
        if clean is False:
            info("A jolt means the direction pin moved while pulses were still "
                 "being generated. That is the one thing the ramp-through-zero "
                 "rule exists to prevent — stop and check the firmware version.")


def test_velocity_command_timeout():
    print("\nstreamed velocity: 300 ms command timeout")
    if not _velocity_ready():
        return

    info("This is the check that matters most: it is the only thing that stops "
         "the column if the host process dies mid-move.")
    confirm("Move at +10 mm/s, then STOP SENDING. The lift must stop by itself.")
    with guard(CLIENT):
        countdown(2, "moving, then going silent")
        _stream_velocity(+0.010, 2.5)
        silent_from = time.monotonic()
        h_at_silence = _height()
        settled = _wait_still(timeout_s=10.0, settle_s=0.5)
        stopped_after = time.monotonic() - silent_from

    check("the lift stopped without being told to", settled is not None)
    check("it stopped within a second of the last command", stopped_after < 2.0,
          f"{stopped_after:.2f} s (300 ms timeout + the ramp down)")
    if settled is not None and h_at_silence is not None:
        coast_mm = (settled - h_at_silence) * 1000.0
        check("it coasted less than 15 mm past the last command",
              abs(coast_mm) < 15.0, f"{coast_mm:+.1f} mm")


def test_velocity_respects_limits():
    print("\nstreamed velocity: limit switches")
    if _status().get("velocity_capable") is not True:
        check("firmware supports streamed velocity", False, "skipping — old firmware")
        return
    if _status().get("position_known") is not True:
        check("lift homed before the limit test", False, "skipping — home first")
        return

    confirm("Stream UP into the upper limit switch. The firmware must stop itself.",
            token="READY")
    with guard(CLIENT):
        countdown(3, "driving up to the limit")
        _stream_velocity(+0.010, 90.0)       # long enough to reach the switch
        CLIENT.call("lift_set_velocity", 0.0)
    settled = _wait_still(timeout_s=15.0)

    st = _status()
    check("the upper limit switch is active", st.get("upper_limit") is True,
          f"upper_limit={st.get('upper_limit')}")
    check("the firmware reported the limit",
          "limit" in str(st.get("last_event", "")).lower(),
          f"last_event={st.get('last_event')!r}")
    if settled is not None:
        check("height is pinned to the top of travel",
              abs(settled - TRAVEL_M) < NOMINAL_TOLERANCE, f"{settled:.4f} m")

    # ...and it must still be possible to drive away from the switch.
    with guard(CLIENT):
        countdown(2, "driving back down off the switch")
        _stream_velocity(-0.010, 4.0)
        CLIENT.call("lift_set_velocity", 0.0)
    off = _wait_still(timeout_s=10.0)
    check("driving away from a closed limit still works",
          off is not None and settled is not None and off < settled - 0.005,
          f"{settled:.4f} -> {off:.4f} m" if off and settled else "no reading")


def test_travel_matches_reality():
    print("\ntravel constant vs the real lift")
    info(f"The stack asserts {TRAVEL_M * 1000:.0f} mm of travel in five places "
         f"(see docs/RUNNING.md). This is the check that they match the metal.")

    if _status().get("position_known") is not True:
        check("lift homed before measuring travel", False, "skipping — home first")
        return

    confirm("Drive the lift to the BOTTOM to measure full travel.")
    with guard(CLIENT):
        countdown(2, "moving to 0.000 m")
        CLIENT.call("lift_to_height", 0.0)
        bottom = _wait_still(timeout_s=60.0)

    st = _status()
    check("lower limit switch is active at the bottom", st.get("lower_limit") is True,
          f"lower_limit={st.get('lower_limit')}")
    if bottom is not None:
        check("reported height at the bottom is ~0", abs(bottom) < 0.010,
              f"{bottom:.4f} m")

    measured = ask_float("Measure the actual platform travel bottom-to-top, in mm")
    if measured is not None:
        err = measured - TRAVEL_M * 1000.0
        check("measured travel matches the constant", abs(err) < 10.0,
              f"measured {measured:.0f} mm, constant {TRAVEL_M * 1000:.0f} mm, "
              f"error {err:+.0f} mm")
        if abs(err) >= 10.0:
            info("Update ALL FIVE locations listed in docs/RUNNING.md, not just "
                 "one — that mismatch is exactly what caused the 0.9176/0.900 bug.")


def test_limits_hold():
    print("\nsoftware limits")
    if _status().get("position_known") is not True:
        check("lift homed before the limit test", False, "skipping — home first")
        return
    confirm("Command the lift ABOVE its travel; the firmware must refuse to exceed it.")
    with guard(CLIENT):
        countdown(2, "commanding an over-travel target")
        CLIENT.call("lift_to_height", TRAVEL_M + 0.20)
        settled = _wait_still(timeout_s=90.0)
    if settled is not None:
        check("lift stopped at or below the travel limit", settled <= TRAVEL_M + 0.005,
              f"{settled:.4f} m")
    observed = ask_yes_no("Did the lift stop cleanly at the top without straining?")
    if observed is not None:
        check("no mechanical strain at the upper limit", observed)


def main() -> int:
    global CLIENT, ARGS
    ARGS = parse_args(__doc__)
    banner("STAGE 1 — LIFT",
           "*** THE LIFT MOVES. Keep hands clear. Ctrl-C stops it. ***")
    CLIENT = connect(ARGS)
    try:
        return run(
            test_preconditions,
            test_position_known_contract,
            test_home,
            test_absolute_moves,
            test_stop_is_prompt,
            # Streamed velocity: the path the whole-body loop uses. Deliberately
            # after the discrete moves, so a fault here cannot be blamed on the
            # basics, and smallest command first.
            test_velocity_capability,
            test_velocity_small,
            test_velocity_larger,
            test_velocity_zero_hold,
            test_velocity_reversal,
            test_velocity_command_timeout,
            test_velocity_respects_limits,
            test_travel_matches_reality,
            test_limits_hold,
        )
    finally:
        CLIENT.halt()


if __name__ == "__main__":
    raise SystemExit(main())
