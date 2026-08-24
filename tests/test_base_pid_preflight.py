"""
test_base_pid_preflight.py — the swerve PID preflight, without a CAN bus.

tools/base_pid_preflight.py is the only thing that writes gains to the SPARK
controllers, and it runs at a moment when nothing else is watching: as
robot/yor.py starts, or from the command line before it does. So the parts that
decide *whether to write at all* are checked here against fake controllers — a
bad manifest, a wrong CAN id, a controller that silently keeps its old value,
and a controller that is already commissioned and must be left alone.

    python tests/test_base_pid_preflight.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from tools.base_pid_preflight import (  # noqa: E402
    COMMISSIONED_MANIFEST, DEFAULT_MANIFEST, PID_FIELDS, STOCK_MANIFEST,
    check_can_ids_against_base,
    check_can_interface, compare, load_manifest, main, plan_writes, read_back,
    run_devices, sync_from_manifest, sync_open_devices, validate_manifest,
)

# The commissioned numbers, restated here so a silent edit to the manifest
# fails this file rather than reaching the wheels.
COMMISSIONED = {
    "drive": {"p": 0.35, "i": 0.0, "d": 0.0, "velocity_ff": 0.23,
              "output_min": -1.0, "output_max": 1.0},
    "steering": {"p": 20.0, "i": 0.0, "d": 6.0, "velocity_ff": 0.0,
                 "output_min": -0.25, "output_max": 0.25},
}

# What the controllers revert to on a power cycle -- the values persisted in
# SPARK flash, restated here for the same reason as COMMISSIONED. These are
# deliberately NOT the REV factory zeros: writing zeros on shutdown would not
# restore anything, it would leave the base limp until the next power cycle.
STOCK = {
    "drive": {"p": 0.2, "i": 0.0, "d": 0.1, "velocity_ff": 0.0,
              "output_min": -1.0, "output_max": 1.0},
    "steering": {"p": 2.0, "i": 0.0, "d": 0.01, "velocity_ff": 0.0,
                 "output_min": -1.0, "output_max": 1.0},
}

RESULTS: list[tuple[str, bool, str]] = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Fake controllers
# ─────────────────────────────────────────────────────────────────────────────

class FakeSpark:
    """A SPARK that remembers what it was told, per slot."""

    opened: list[tuple[str, int]] = []

    def __init__(self, interface: str, can_id: int, *, deaf_field: str | None = None,
                 drop_reads: int = 0):
        self.interface = interface
        self.can_id = can_id
        self.deaf_field = deaf_field       # a field that refuses to change
        self.drop_reads = drop_reads       # first N reads answer 0.0, as a timeout does
        self.writes = 0
        self.slots: dict[int, dict[str, float]] = {}
        FakeSpark.opened.append((interface, can_id))

    def _set(self, slot, key, value):
        self.writes += 1
        if key == self.deaf_field:
            return
        self.slots.setdefault(int(slot), {})[key] = float(value)

    def _get(self, slot, key):
        # The binding returns 0.0 when a controller does not answer in time,
        # rather than raising, so a dropped frame is a plausible-looking value.
        if self.drop_reads > 0:
            self.drop_reads -= 1
            return 0.0
        return self.slots.get(int(slot), {}).get(key, 0.0)

    def SetP(self, slot, v): self._set(slot, "p", v)
    def SetI(self, slot, v): self._set(slot, "i", v)
    def SetD(self, slot, v): self._set(slot, "d", v)
    def SetVelocityFF(self, slot, v): self._set(slot, "velocity_ff", v)
    def SetOutputMin(self, slot, v): self._set(slot, "output_min", v)
    def SetOutputMax(self, slot, v): self._set(slot, "output_max", v)

    def GetP(self, slot): return self._get(slot, "p")
    def GetI(self, slot): return self._get(slot, "i")
    def GetD(self, slot): return self._get(slot, "d")
    def GetVelocityFF(self, slot): return self._get(slot, "velocity_ff")
    def GetOutputMin(self, slot): return self._get(slot, "output_min")
    def GetOutputMax(self, slot): return self._get(slot, "output_max")


def factory(**kwargs):
    FakeSpark.opened = []
    devices: dict[int, FakeSpark] = {}

    def make(interface, can_id):
        devices[can_id] = FakeSpark(interface, can_id, **kwargs)
        return devices[can_id]

    make.devices = devices
    return make


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def test_manifest_values() -> None:
    print("\nthe shipped manifest holds the commissioned values")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    check("manifest validates", not validate_manifest(manifest),
          str(validate_manifest(manifest)))
    for role, expected in COMMISSIONED.items():
        actual = {k: manifest["roles"][role][k] for k in expected}
        check(f"{role} gains are the commissioned set", actual == expected, str(actual))

    check("steering output stays at the commissioned +/-0.25",
          manifest["roles"]["steering"]["output_min"] == -0.25
          and manifest["roles"]["steering"]["output_max"] == 0.25)
    check("the unvalidated full-range steering combination is not present",
          not (manifest["roles"]["steering"]["p"] == 10.0
               and manifest["roles"]["steering"]["output_max"] == 1.0))
    check("PID slot is an explicit integer", isinstance(manifest["pid_slot"], int),
          repr(manifest["pid_slot"]))


def test_can_ids() -> None:
    print("\nCAN ids agree with the running base code")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    check("manifest matches robot/base_motor.py", not check_can_ids_against_base(manifest),
          str(check_can_ids_against_base(manifest)))

    wrong = copy.deepcopy(manifest)
    wrong["modules"]["FL"]["drive"] = 9
    problems = check_can_ids_against_base(wrong)
    check("a wrong CAN id is caught", any("FL.drive" in p for p in problems), str(problems))

    missing = copy.deepcopy(manifest)
    del missing["modules"]["RR"]
    check("a missing module is caught",
          any("missing module RR" in p for p in check_can_ids_against_base(missing)))

    check("all eight controllers are planned", len(plan_writes(manifest)) == 8)


def test_manifest_validation() -> None:
    print("\nbad manifests are rejected before anything is written")
    manifest = load_manifest(COMMISSIONED_MANIFEST)

    cases = {
        "non-finite gain": ("roles", lambda m: m["roles"]["drive"].__setitem__("p", float("inf"))),
        "gain out of range": ("roles", lambda m: m["roles"]["steering"].__setitem__("p", 5000.0)),
        "output range inverted": ("roles", lambda m: m["roles"]["drive"].update(
            {"output_min": 1.0, "output_max": -1.0})),
        "output beyond duty cycle": ("roles", lambda m: m["roles"]["drive"].__setitem__(
            "output_max", 4.0)),
        "string gain": ("roles", lambda m: m["roles"]["drive"].__setitem__("d", "6.0")),
        "bad pid slot": ("slot", lambda m: m.__setitem__("pid_slot", 9)),
        "duplicate CAN id": ("modules", lambda m: m["modules"]["FR"].__setitem__("drive", 1)),
        "CAN id out of range": ("modules", lambda m: m["modules"]["FR"].__setitem__("drive", 999)),
        "unknown role": ("modules", lambda m: m["modules"]["FL"].__setitem__("lift", 12)),
    }
    for label, (_area, mutate) in cases.items():
        broken = copy.deepcopy(manifest)
        mutate(broken)
        check(f"rejects: {label}", bool(validate_manifest(broken)))


def test_apply_and_verify() -> None:
    print("\napply to RAM, then read every field back")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    specs = plan_writes(manifest)
    slot = manifest["pid_slot"]

    make = factory()
    problems = run_devices(specs, "can0", slot, write=True, tolerance=1e-3,
                           device_factory=make)
    check("a clean apply reports no problems", not problems, str(problems))
    check("every controller was opened exactly once",
          sorted(d.can_id for d in make.devices.values()) == [1, 2, 3, 4, 5, 6, 7, 8])

    drive = make.devices[1].slots[slot]
    steer = make.devices[5].slots[slot]
    check("drive controller holds the drive gains", drive == COMMISSIONED["drive"], str(drive))
    check("steering controller holds the steering gains",
          steer == COMMISSIONED["steering"], str(steer))
    check("nothing was written to another PID slot",
          all(set(d.slots) == {slot} for d in make.devices.values()))


def test_readback_catches_a_deaf_controller() -> None:
    print("\na controller that ignores a write fails the preflight")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    specs = plan_writes(manifest)
    slot = manifest["pid_slot"]

    problems = run_devices(specs, "can0", slot, write=True, tolerance=1e-3,
                           device_factory=factory(deaf_field="d"))
    check("the mismatch is reported", bool(problems), str(problems[:1]))
    check("it names the field that differs", all("d expected" in p for p in problems),
          str(problems[:1]))
    # Only the steering modules have a non-zero D, so only those can mismatch.
    check("only the controllers that actually differ are flagged", len(problems) == 4,
          f"{len(problems)} problems")


def test_verify_only_never_writes() -> None:
    print("\n--verify-only writes nothing")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    specs = plan_writes(manifest)
    slot = manifest["pid_slot"]

    make = factory()
    problems = run_devices(specs, "can0", slot, write=False, tolerance=1e-3,
                           device_factory=make)
    check("stock controllers fail a verify-only run", bool(problems), f"{len(problems)} problems")
    check("no slot was written", all(not d.slots for d in make.devices.values()))


def test_unreachable_device() -> None:
    print("\na controller that is not on the bus")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    specs = plan_writes(manifest)

    def make(interface, can_id):
        if can_id == 7:
            raise OSError("no response from device")
        return FakeSpark(interface, can_id)

    problems = run_devices(specs, "can0", manifest["pid_slot"], write=True,
                           tolerance=1e-3, device_factory=make)
    check("the unreachable controller is reported",
          any("CAN 7" in p and "could not open" in p for p in problems), str(problems))
    check("the run does not stop at the first failure", len(problems) == 1, str(problems))


def test_tolerance() -> None:
    print("\nreadback tolerance")
    spec = plan_writes(load_manifest(DEFAULT_MANIFEST))[0]
    exact = dict(spec.values)
    check("an exact readback compares equal", not compare(spec, exact, 1e-3))

    nudged = dict(spec.values, p=spec.values["p"] + 1e-5)
    check("a float round-trip is tolerated", not compare(spec, nudged, 1e-3))

    wrong = dict(spec.values, p=spec.values["p"] * 2)
    check("a real difference is not", bool(compare(spec, wrong, 1e-3)))

    missing = dict(spec.values)
    missing["p"] = None
    check("an unreadable field counts as a mismatch", bool(compare(spec, missing, 1e-3)))


# ─────────────────────────────────────────────────────────────────────────────
# In-process sync — the path robot/yor.py takes at startup
# ─────────────────────────────────────────────────────────────────────────────

def commissioned_devices(**kwargs) -> dict[int, FakeSpark]:
    """Eight fakes already holding the manifest values, as after a good sync."""
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    slot = manifest["pid_slot"]
    devices = {}
    for spec in plan_writes(manifest):
        device = FakeSpark("can0", spec.can_id, **kwargs)
        device.slots[slot] = dict(spec.values)
        device.writes = 0
        devices[spec.can_id] = device
    return devices


def test_sync_writes_stock_controllers() -> None:
    print("\nstartup sync: stock controllers are written")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    slot = manifest["pid_slot"]
    devices = {spec.can_id: FakeSpark("can0", spec.can_id) for spec in plan_writes(manifest)}

    results = sync_open_devices(devices, manifest, log=None)
    check("every controller was written", all(r.status == "written" for r in results),
          str(sorted({r.status for r in results})))
    check("no problems reported", not [p for r in results for p in r.problems])
    check("drive controller holds the drive gains",
          devices[1].slots[slot] == COMMISSIONED["drive"], str(devices[1].slots[slot]))
    check("steering controller holds the steering gains",
          devices[5].slots[slot] == COMMISSIONED["steering"], str(devices[5].slots[slot]))
    check("nothing was written to another PID slot",
          all(set(d.slots) == {slot} for d in devices.values()))


def test_sync_skips_commissioned_controllers() -> None:
    print("\nstartup sync: a controller that already has the gains is left alone")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    devices = commissioned_devices()

    results = sync_open_devices(devices, manifest, log=None)
    check("every controller is reported as already set",
          all(r.status == "already-set" for r in results),
          str(sorted({r.status for r in results})))
    check("not one field was written", all(d.writes == 0 for d in devices.values()),
          str({i: d.writes for i, d in devices.items() if d.writes}))

    # One module reverted — a single SPARK power-cycled on its own.
    devices[5].slots.clear()
    results = sync_open_devices(devices, manifest, log=None)
    written = [r.spec.can_id for r in results if r.status == "written"]
    check("only the reverted controller is rewritten", written == [5], str(written))
    check("its neighbours were still not touched",
          all(d.writes == 0 for can_id, d in devices.items() if can_id != 5))


def test_sync_retries_a_dropped_read() -> None:
    print("\nstartup sync: a dropped parameter read is retried, not believed")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    # Enough dropped reads to spoil the first readback pass of every field.
    devices = commissioned_devices(drop_reads=len(PID_FIELDS))

    results = sync_open_devices(devices, manifest, log=None)
    check("the retry finds the gains already in place",
          all(r.status == "already-set" for r in results),
          str(sorted({r.status for r in results})))
    check("so nothing was rewritten", all(d.writes == 0 for d in devices.values()))


def test_sync_fails_on_a_deaf_controller() -> None:
    print("\nstartup sync: a controller that ignores the write fails")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    devices = {spec.can_id: FakeSpark("can0", spec.can_id, deaf_field="p")
               for spec in plan_writes(manifest)}

    results = sync_open_devices(devices, manifest, log=None)
    check("it is reported as failed", all(r.status == "failed" for r in results),
          str(sorted({r.status for r in results})))
    check("the failing field is named",
          all(any("p expected" in p for p in r.problems) for r in results))
    check("failed results are not ok", not any(r.ok for r in results))


def test_sync_missing_device() -> None:
    print("\nstartup sync: a controller that is not among the open handles")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    devices = {spec.can_id: FakeSpark("can0", spec.can_id) for spec in plan_writes(manifest)}
    devices.pop(7)

    results = sync_open_devices(devices, manifest, log=None)
    failed = [r for r in results if not r.ok]
    check("exactly the missing controller fails", [r.spec.can_id for r in failed] == [7],
          str([r.spec.can_id for r in failed]))
    check("the message says no device was open",
          all("no open device" in p for r in failed for p in r.problems),
          str(failed[0].problems))


def test_sync_from_manifest_guards(tmp: Path) -> None:
    print("\nstartup sync: the manifest is checked before any device is touched")
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    devices = {spec.can_id: FakeSpark("can0", spec.can_id) for spec in plan_writes(manifest)}

    ok, problems = sync_from_manifest(devices, log=None)
    check("the shipped manifest syncs cleanly", ok and not problems, str(problems))

    broken = tmp / "broken_manifest.json"
    manifest["roles"]["steering"]["p"] = 1e6
    broken.write_text(json.dumps(manifest))
    fresh = {spec.can_id: FakeSpark("can0", spec.can_id) for spec in plan_writes(manifest)}
    ok, problems = sync_from_manifest(fresh, manifest_path=broken, log=None)
    check("an out-of-range gain is refused", not ok and bool(problems), str(problems))
    check("and nothing was written", all(d.writes == 0 for d in fresh.values()))

    ok, problems = sync_from_manifest(devices, manifest_path=tmp / "no_such.json", log=None)
    check("a missing manifest is refused", not ok and bool(problems), str(problems))


def test_every_shipped_manifest_is_valid() -> None:
    """Whatever is in config/ must be applyable, not just the two named ones.

    Manifests get added for experiments (base_pid_hybrid.json was added on
    2026-08-24 to pair the commissioned drive gains with stock steering), and a
    typo in one is only discovered at the moment the robot is about to drive.
    """
    print("\nevery shipped manifest")
    shipped = sorted((_REPO / "config").glob("base_pid_*.json"))
    check("there are manifests to check", len(shipped) >= 2, str(len(shipped)))
    for path in shipped:
        manifest = load_manifest(path)
        check(f"{path.name} validates", not validate_manifest(manifest),
              str(validate_manifest(manifest)))
        check(f"{path.name} names the controllers base_motor.py uses",
              not check_can_ids_against_base(manifest),
              str(check_can_ids_against_base(manifest)))
        check(f"{path.name} plans all eight controllers", len(plan_writes(manifest)) == 8)
        # A manifest without a scale silently inherits the module default, which
        # is only right for one gain set.
        check(f"{path.name} declares its own drive_command_scale",
              isinstance(manifest.get("drive_command_scale"), (int, float)),
              str(manifest.get("drive_command_scale")))


def test_defaults_manifest() -> None:
    """The stock manifest restored on shutdown."""
    print("\nstock manifest")
    manifest = load_manifest(STOCK_MANIFEST)

    check("stock manifest validates", not validate_manifest(manifest),
          str(validate_manifest(manifest)))
    check("stock manifest matches robot/base_motor.py",
          not check_can_ids_against_base(manifest), str(check_can_ids_against_base(manifest)))
    check("all eight controllers are planned", len(plan_writes(manifest)) == 8)

    for role, expected in STOCK.items():
        actual = {key: manifest["roles"][role][key] for key, _s, _g in PID_FIELDS}
        check(f"{role} holds the stock values", actual == expected, str(actual))

    # Both manifests write the same slot, or the restore would leave the
    # commissioned gains in place and merely blank a slot nobody uses.
    commissioned = load_manifest(COMMISSIONED_MANIFEST)
    check("stock and commissioned manifests target the same PID slot",
          manifest["pid_slot"] == commissioned["pid_slot"],
          f"{manifest['pid_slot']} vs {commissioned['pid_slot']}")
    check("the two manifests actually differ",
          manifest["roles"] != commissioned["roles"])
    # The failure this guards against is subtle: an all-zero "stock" file looks
    # plausible (they are the REV factory values) but would wipe the
    # controllers rather than restore them, since these SPARKs hold non-zero
    # gains in flash.
    check("the stock manifest is what the robot applies by default",
          DEFAULT_MANIFEST == STOCK_MANIFEST, str(DEFAULT_MANIFEST))
    check("stock is a real gain set, not all zeros",
          any(manifest["roles"][role]["p"] > 0.0 for role in manifest["roles"]),
          str({r: manifest["roles"][r]["p"] for r in manifest["roles"]}))


def test_restore_writes_stock_over_commissioned() -> None:
    """A restore has to reach controllers that already hold the tuned gains."""
    print("\nrestoring stock over commissioned gains")
    FakeSpark.opened = []
    commissioned = load_manifest(COMMISSIONED_MANIFEST)
    devices = {spec.can_id: FakeSpark("can0", spec.can_id)
               for spec in plan_writes(commissioned)}

    sync_open_devices(devices, commissioned, log=None)   # a commissioned robot
    ok, problems = sync_from_manifest(devices, manifest_path=STOCK_MANIFEST, log=None)
    check("the restore reports no problems", ok, str(problems))

    slot = commissioned["pid_slot"]
    for spec in plan_writes(load_manifest(STOCK_MANIFEST)):
        held = devices[spec.can_id].slots[slot]
        check(f"{spec.label} ends at stock",
              all(held[key] == spec.values[key] for key, _s, _g in PID_FIELDS), str(held))


def test_yor_syncs_before_the_control_loop() -> None:
    """robot/yor.py must sync the gains before base.start_control().

    Read from the source: importing robot/yor.py needs nerolib and a CAN bus,
    neither of which exists off the robot.
    """
    print("\nrobot/yor.py wiring")
    import ast

    source = (_REPO / "robot/yor.py").read_text()
    tree = ast.parse(source)
    init = next(
        item
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "YOR"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "init"
    )
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    check("init() syncs the base PID gains", "self._sync_base_pid_gains" in calls, str(calls))
    check("it does so before the control loop starts",
          "self.base.start_control" in calls
          and calls.index("self._sync_base_pid_gains") < calls.index("self.base.start_control"),
          str(calls))
    check("the sync goes through the already-open devices",
          "self.base.swerve_devices()" in source)
    check("base_motor.py hands those devices out",
          "def swerve_devices" in (_REPO / "robot/base_motor.py").read_text())
    check("a failed sync stops the start-up",
          "raise RuntimeError" in source.split("def _sync_base_pid_gains")[1].split("def ")[0])

    # The sync has to be switchable without editing the file: --no-flash-base-pid
    # is how you start the node on a bench with no CAN bus, or deliberately keep
    # whatever gains a controller is holding while investigating one.
    main_body = source.split("def main()")[1]
    check("the command line exposes the switch", "--flash-base-pid" in main_body)
    check("and it reaches the constructor",
          "flash_base_pid=args.flash_base_pid" in main_body)
    check("the manifest can be overridden too",
          "--base-pid-manifest" in main_body
          and "base_pid_manifest=args.base_pid_manifest" in main_body)
    check("the default keeps the sync on", "default=True" in main_body)


def test_yor_restores_defaults_on_shutdown() -> None:
    """robot/yor.py must hand the controllers back in stock condition.

    Commissioned gains live in controller RAM, which outlives this process.
    Whatever opens the bus next inherits them silently, so the shutdown path
    is the only place that can bound how long they are in effect.
    """
    print("\nrobot/yor.py shutdown wiring")
    source = (_REPO / "robot/yor.py").read_text()

    check("YOR knows how to restore the stock gains",
          "def _restore_base_pid_gains" in source)

    body = source.split("def _restore_base_pid_gains")[1].split("\n    def ")[0]
    check("the restore reads the stock manifest, not the commissioned one",
          "self._base_pid_stock_manifest" in body and "COMMISSIONED_MANIFEST" not in body)
    check("it only undoes a change it made",
          "if not self._flash_base_pid" in body)
    check("it stops the control loop before changing gains",
          "control_loop_running" in body and "stop_control" in body)
    check("it cannot raise out of the shutdown path", "except Exception" in body)

    # Ordering: the shutdown sequence has to stop the base *before* the
    # restore, and the restore has to happen while the device handles are
    # still open -- i.e. before the interpreter tears the process down.
    shutdown = source.split("def graceful_shutdown")[1].split("atexit.register")[0]
    check("shutdown calls the restore", "_restore_base_pid_gains" in shutdown, shutdown.strip()[:0])
    check("the base is stopped before the gains change",
          shutdown.index("yor.base.stop_control")
          < shutdown.index("yor._restore_base_pid_gains"))
    check("the whole-body loop is stopped before that",
          shutdown.index("yor.wholebody.stop") < shutdown.index("yor.base.stop_control"))

    # init() raises for a living -- a failed lift home, a failed arm home, a
    # failed PID sync -- and by then the gains may already be written. If the
    # handler were registered after init(), the one case where the robot is
    # left in a strange state is the one case the restore would not run.
    body = source.split("def main()")[1]
    check("the shutdown handler is registered before init()",
          body.index("atexit.register(graceful_shutdown)") < body.index("yor.init()"))
    check("a teardown failure does not skip the steps after it",
          "def attempt(" in body and "attempt(\"base control loop stop\"" in body)

    main_body = source.split("def main()")[1]
    check("the command line exposes the switch", "--restore-base-pid" in main_body)
    check("and it reaches the constructor",
          "restore_base_pid=args.restore_base_pid" in main_body)
    check("the stock manifest can be overridden too",
          "--base-pid-stock-manifest" in main_body
          and "base_pid_stock_manifest=args.base_pid_stock_manifest" in main_body)


def test_environment_guards() -> None:
    print("\nenvironment guards")
    ok, detail = check_can_interface("definitely-not-an-interface")
    check("a missing CAN interface is refused", not ok, detail)
    ok, detail = check_can_interface("lo")
    check("an up interface is accepted", ok, detail)

    # The guard against a second owner is a live process scan, so this run is
    # itself the fixture: this test process must not be mistaken for the robot.
    from tools.base_pid_preflight import find_conflicting_processes
    conflicts = find_conflicting_processes()
    check("the preflight's own test run is not mistaken for the robot",
          not any("test_base_pid_preflight" in c for c in conflicts), str(conflicts))


def test_cli_dry_run(tmp: Path) -> None:
    print("\ncommand-line behaviour")
    check("a dry run of the shipped manifest passes",
          main(["--dry-run", "--manifest", str(DEFAULT_MANIFEST)]) == 0)

    broken = tmp / "broken_manifest.json"
    manifest = load_manifest(COMMISSIONED_MANIFEST)
    manifest["roles"]["drive"]["p"] = -1.0
    broken.write_text(json.dumps(manifest))
    check("a dry run of a broken manifest fails",
          main(["--dry-run", "--manifest", str(broken)]) == 1)

    check("a missing manifest fails", main(["--dry-run", "--manifest", "/no/such/file"]) == 1)
    check("--dry-run and --verify-only are refused together",
          main(["--dry-run", "--verify-only"]) == 2)


def main_() -> int:
    import tempfile

    test_manifest_values()
    test_can_ids()
    test_manifest_validation()
    test_apply_and_verify()
    test_readback_catches_a_deaf_controller()
    test_verify_only_never_writes()
    test_unreachable_device()
    test_tolerance()
    test_sync_writes_stock_controllers()
    test_sync_skips_commissioned_controllers()
    test_sync_retries_a_dropped_read()
    test_sync_fails_on_a_deaf_controller()
    test_sync_missing_device()
    test_every_shipped_manifest_is_valid()
    test_defaults_manifest()
    test_restore_writes_stock_over_commissioned()
    test_yor_syncs_before_the_control_loop()
    test_yor_restores_defaults_on_shutdown()
    test_environment_guards()
    with tempfile.TemporaryDirectory() as tmp:
        test_sync_from_manifest_guards(Path(tmp))
        test_cli_dry_run(Path(tmp))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    failures = [name for name, ok, _ in RESULTS if not ok]
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main_())
