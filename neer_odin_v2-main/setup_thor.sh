#!/bin/bash
set -e

# Initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda initialization script not found."
    exit 1
fi

echo "Creating conda environment 'yor'..."
conda create -n yor python=3.10 -y

echo "Activating conda environment 'yor'..."
conda activate yor

echo "Installing system dependencies..."
conda install -y -c conda-forge pinocchio spdlog catch2 boost pybind11 gxx cxx-compiler || exit 1

echo "Installing build tools..."
pip install scikit-build-core cmake ruckig || exit 1

echo "Installing hardware drivers (sparkcan_py)..."
cd sparkcan_py
pip install . || exit 1
cd ..

echo "Installing nerolib..."
cd nerolib
bash install.sh || exit 1
cd ..

echo "Installing main YOR package..."
pip install -e . || exit 1

echo "Checking package installations..."
python -c "import robot; print('Robot package installed successfully')" || exit 1
python -c "import nerolib; import sparkcan_py; print('Hardware drivers installed successfully')" || exit 1

echo "Setup complete for Thor!"
