#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_odin.sh — build the pyodin bridge and install the SLAM-side deps.
#
#   bash setup_odin.sh            # publisher only (no torch, no nav stack)
#   bash setup_odin.sh --full     # + verify torch for the mapping/nav stack
#
# This covers the SLAM box only. The arm/base side (nerolib, sparkcan_py,
# mujoco, mink) is separate — see README.md and docs/RUNNING.md.
#
# Prerequisites on a fresh machine:
#   sudo apt install -y build-essential cmake git tmux libusb-1.0-0-dev libeigen3-dev
#   echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2207", MODE="0666"' \
#       | sudo tee /etc/udev/rules.d/99-odin.rules
#   sudo udevadm control --reload-rules && sudo udevadm trigger   # then replug
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

ENV_NAME="${YOR_CONDA_ENV:-slam-odin}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda not found. Install miniconda first." && exit 1
fi

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "==> Creating conda env '$ENV_NAME' (python 3.10)…"
    conda create -n "$ENV_NAME" -c conda-forge --override-channels python=3.10 -y
fi
conda activate "$ENV_NAME"

echo "==> Python deps for the publisher…"
# pyodin is built with the SYSTEM g++ — we only need pybind11 headers from pip,
# not a conda compiler (which can mismatch the system libusb).
python -m pip install --upgrade pip
python -m pip install pybind11 numpy pyzmq loop-rate-limiters scipy \
                      opencv-python pyyaml msgpack msgpack-numpy

echo "==> Building the pyodin SDK bridge…"
bash "$HERE/pyodin/build.sh"

if [ "$FULL" = "1" ]; then
    echo "==> Nav stack deps…"
    python -m pip install viser
    echo "==> Verifying torch (mapping_torch + voxel_map need CUDA to be useful)."
    echo "    On Jetson install the JetPack-matched NVIDIA aarch64 wheel — NOT plain 'pip install torch'."
    python -c "import torch; print('    torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
        || echo "    !! torch not installed — install it before running robot.slam_node_"
fi

echo
echo "==> Verifying…"
python -c "import pyodin; print('pyodin OK')"
echo "NOTE: 'commlink' is an internal package and is not installed here — see README.md."
echo
echo "Done. Next:"
echo "  conda activate $ENV_NAME"
echo "  python -m robot.odin_pub_node      # publisher alone"
[ "$FULL" = "1" ] && echo "  bash nav.sh                        # publisher + slam_node_"
exit 0
