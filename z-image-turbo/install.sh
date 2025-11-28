#!/bin/bash
# Flash Attention Installation Script for Z-Image-Turbo
# Installs PyTorch 2.7.0 with CUDA 12.8 and Flash Attention 2.7.4

set -e

echo "Installing PyTorch 2.7.0 with CUDA 12.8..."
uv pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "Installing Flash Attention 2.7.4..."
uv pip install --no-cache-dir \
    https://github.com/loscrossos/lib_flashattention/releases/download/v2.7.4.post1_crossos00/flash_attn-2.7.4.post1+cu129torch2.7.0-cp312-cp312-linux_x86_64.whl

echo "Installing project dependencies..."
uv pip install --no-cache-dir -r requirements.txt

echo ""
echo "Installation complete!"
echo ""
echo "Verify installation with:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}')\""
echo "  python -c \"import flash_attn; print('Flash Attention: OK')\""
