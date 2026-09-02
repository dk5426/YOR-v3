"""config.py — the one place Aria teleop settings come from.

`config/aria_teleop.yaml` holds them, commented; both entry points read it so
there is a single description of a session rather than a command line each has
to get right. Every key is optional and falls back to the default here, so a
missing or partial file still runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = _REPO / "config" / "aria_teleop.yaml"

DEFAULTS: dict[str, dict[str, Any]] = {
    "publisher": {"host": "localhost", "port": 5555, "stale_s": 0.5,
                  "clock_port": 5556, "stats": True},
    "mapping": {"hand": "both", "position_scale": 1.0,
                "follow_orientation": True, "translation_frame": "world",
                "scene": "description/scene_wholebody.xml"},
    "clutch": {"reseed": True, "hold_lift": True},
    "home": {"gesture": True},
    "sim": {"ik_rate_hz": 100, "base_posture_cost": 1e-4, "solver": "pyqpmad",
            "viser_port": 8080, "share": False},
    # robot/hand/hands.py -- the finger path, which both nodes own in-process
    # and which reads this same publisher on a thread of its own.
    "hand": {"backend": "none", "sides": "both",
             "serial": {"left": "", "right": ""},
             "rpc_port": 5558, "rate_hz": 100, "ramp_s": 1.5,
             "lowpass_hz": 5.0},
}


class AriaConfig:
    """Parsed `config/aria_teleop.yaml`, with defaults filled in.

    Sections are reachable as attributes (`cfg.publisher["host"]`); `scene` is
    resolved to an absolute path against the repo root, since both entry points
    are run from wherever the operator happens to be.
    """

    def __init__(self, data: dict, path: Path | None = None):
        self.path = path
        for section, defaults in DEFAULTS.items():
            merged = dict(defaults)
            given = data.get(section) or {}
            merged.update(given)
            # `hand.serial` is the one nested mapping; a file that names only
            # one side would otherwise drop the other key entirely
            for key, value in defaults.items():
                if isinstance(value, dict) and isinstance(given.get(key), dict):
                    merged[key] = {**value, **given[key]}
            setattr(self, section, merged)
        scene = Path(self.mapping["scene"])
        self.mapping["scene"] = scene if scene.is_absolute() else _REPO / scene

    def hand_sides(self) -> tuple[str, ...]:
        """Which WUJI hands to drive: a subset of the teleoped arms, possibly none.

        Whole-body IK needs both wrist targets, so `mapping.hand` stays the
        arms' setting; the fingers are a separate path and one hand -- or
        neither -- may be plugged in. The default `both` is what every session
        did before the key existed: the intersection below means a one-armed
        session still gets exactly its own hand.
        """
        want = str(self.hand["sides"] or "both").lower()
        if want == "none":
            return ()
        arms = (("left", "right") if self.mapping["hand"] == "both"
                else (self.mapping["hand"],))
        sides = ("left", "right") if want == "both" else (want,)
        return tuple(s for s in sides if s in arms)

    @classmethod
    def load(cls, path: str | Path | None = None) -> AriaConfig:
        """Read the YAML, or return pure defaults if the file is absent."""
        p = Path(path) if path else DEFAULT_CONFIG
        if not p.exists():
            if path is not None:  # an explicit path that isn't there is an error
                raise FileNotFoundError(f"aria config not found: {p}")
            print(f"[aria] no {p.name}, using built-in defaults")
            return cls({}, None)
        import yaml

        with open(p) as f:
            return cls(yaml.safe_load(f) or {}, p)

    def describe(self) -> str:
        src = self.path.name if self.path else "defaults"
        return (f"[aria] config {src}: "
                f"{self.publisher['host']}:{self.publisher['port']} "
                f"arms={self.mapping['hand']} "
                f"hands={'+'.join(self.hand_sides()) or 'none'} "
                f"scale={self.mapping['position_scale']} "
                f"follow_orientation={self.mapping['follow_orientation']} "
                f"translation={self.mapping['translation_frame']}")
