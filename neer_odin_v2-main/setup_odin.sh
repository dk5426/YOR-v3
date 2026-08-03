#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_odin.sh — environment bring-up for neer_odin on a fresh Jetson Thor.
# Replaces setup_thor.sh + setup_nav.sh (no ZED / no pyzed).
#
#   bash setup_odin.sh           # base: standalone Odin + Viser test + pyodin
#   bash setup_odin.sh --full    # also install torch + nav stack for slam_node_
# ──────────────────────────────────────────────────────────────────────────────
set -e

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

ENV_NAME="slam-odin"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# ── conda ─────────────────────────────────────────────────────────────────────
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda not found. Install miniconda first." && exit 1
fi

echo "==> Creating conda env '$ENV_NAME' (python 3.10)…"
# Use conda-forge with --override-channels so we don't trip the defaults-channel ToS gate.
conda create -n "$ENV_NAME" -c conda-forge --override-channels python=3.10 -y
conda activate "$ENV_NAME"

echo "==> Core Python deps (publisher + standalone Viser test)…"
# pyodin is built with the SYSTEM g++ (validated on Thor) — we only need pybind11
# headers from pip, NOT a conda compiler (which can mismatch the system libusb).
# Use `python -m pip` because a freshly created env may not have its pip on PATH yet.
python -m pip install --upgrade pip
python -m pip install pybind11 numpy pyzmq commlink loop-rate-limiters viser \
            opencv-python scipy msgpack msgpack-numpy pyyaml

echo "==> Building the pyodin SDK bridge…"
bash "$HERE/pyodin/build.sh"

# Register the `robot` package so `python -m robot.odin_pub_node` resolves, but
# WITHOUT dragging the heavy robot/sim deps (phoenix6, mujoco, dynamixel…).
echo "==> Installing the robot package (no deps)…"
python -m pip install -e . --no-deps

if [ "$FULL" = "1" ]; then
    echo "==> [full] PyTorch w/ CUDA for mapping_torch."
    echo "    Jetson Thor needs the JetPack-matched NVIDIA aarch64 wheel — NOT default pip torch."
    echo "    Install it from the NVIDIA Jetson/sbsa index for your JetPack, then re-run with --full,"
    echo "    or if torch is already present this step just verifies it."
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
        || echo "    !! torch not installed — install the Jetson CUDA wheel before running slam_node_."
    echo "==> [full] nav extras (viser/opencv/scipy already present)…"
    python -m pip install -e ".[nav]" --no-deps || true
fi

echo ""
echo "==> Verifying…"
python -c "import pyodin; print('pyodin OK')"
python -c "import robot; print('robot package OK')"
echo ""
echo "Done. Next:"
echo "  conda activate $ENV_NAME"
echo "  python tools/odin_viser_test.py     # standalone sensor + map test (http://localhost:8080)"
[ "$FULL" = "1" ] && echo "  bash nav.sh                         # full pipeline (odin_pub_node + slam_node_)"
