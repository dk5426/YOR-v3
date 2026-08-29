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
    "publisher": {"host": "localhost", "port": 5555, "stale_s": 0.5},
    "mapping": {"hand": "both", "position_scale": 1.0,
                "follow_orientation": True, "translation_frame": "world",
                "scene": "description/scene_wholebody.xml"},
    "clutch": {"reseed": True, "hold_lift": True},
    "home": {"gesture": True, "dwell_s": 1.0},
    "sim": {"ik_rate_hz": 100, "base_posture_cost": 1e-4, "solver": "pyqpmad",
            "viser_port": 8080, "share": False},
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
            merged.update(data.get(section) or {})
            setattr(self, section, merged)
        scene = Path(self.mapping["scene"])
        self.mapping["scene"] = scene if scene.is_absolute() else _REPO / scene

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
                f"hand={self.mapping['hand']} "
                f"scale={self.mapping['position_scale']} "
                f"follow_orientation={self.mapping['follow_orientation']} "
                f"translation={self.mapping['translation_frame']}")
