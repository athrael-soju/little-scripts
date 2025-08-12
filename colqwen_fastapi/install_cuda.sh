# Install CUDA optionally if your Nvidia GPU isn't yet supported by the latest version of PyTorch
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade
