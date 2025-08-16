# Install CUDA optionally if your Nvidia GPU isn't yet supported by the latest version of PyTorch
set -euo pipefail

# Install PyTorch + TorchVision built for CUDA 12.9
pip install --upgrade \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu129
