#!/usr/bin/env bash
# Odin 1 live 3D mapper — no ROS. SDK -> pyodin -> voxel map -> Viser.
#
#   bash ~/neer_odin/run_viser_map.sh                # http://localhost:8080
#   bash ~/neer_odin/run_viser_map.sh --voxel 0.03 --min-hits 3
#
# One consumer at a time: stop any ROS driver (ROS1 nav-stack or ROS2 odin_ws)
# before running this. Stop with Ctrl-C.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate slam-odin
cd "$HOME/neer_odin"
exec python tools/odin_viser_map.py "$@"
