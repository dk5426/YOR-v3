#!/usr/bin/env python3
"""Run the teleop-axis diagnostic on only the left Nero arm in YOR-v3."""

from diagnose_right_teleop_axes import main


if __name__ == "__main__":
    raise SystemExit(main(default_arm="left", default_rotations_only=False))
