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

echo "Activating conda environment 'yor'..."
conda activate yor

echo "Installing PyTorch with CUDA..."
conda install -y "pytorch=*=*cuda*" torchvision torchaudio -c conda-forge -c pytorch -c nvidia || exit 1

echo "Installing Navigation dependencies..."
pip install -e ".[nav]" || exit 1

echo "Installing PyZED wrapper..."
if [ -f "/usr/local/zed/pyzed-5.2-cp310-cp310-linux_aarch64.whl" ]; then
    echo "Found local pyzed wheel for Jetson. Installing..."
    pip install /usr/local/zed/pyzed-5.2-cp310-cp310-linux_aarch64.whl || exit 1
else
    echo "Running ZED Python API script..."
    python -m pyzed.sl.get_python_api || echo "Could not install PyZED automatically"
fi

echo "Verifying PyTorch CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA built: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')" || exit 1

echo "Navigation setup complete!"
