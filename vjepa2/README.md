# Video Feature Extraction with V-JEPA 2

This project demonstrates video feature extraction using V-JEPA 2 (Video Joint Embedding Predictive Architecture) for video understanding tasks.

## Overview

The script downloads a sample video from the Kinetics dataset and extracts visual features using a pre-trained V-JEPA 2 model from Hugging Face.

## Features

- Video frame extraction using Decord (compatible with nightly PyTorch builds)
- Feature extraction using pre-trained V-JEPA 2 model
- GPU acceleration support
- Handles online video URLs

## Setup

1. Create and activate a virtual environment (recommended):
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. To run V-JEPA 2 model, ensure you have installed the latest transformers:
```
pip install -U git+https://github.com/huggingface/transformers
```

3a. For up to RTX 4 series GPU, install:

```
pip install torch torchvision torchcodec
```

3b. For newer GPUs (RTX 5 series) install dependencies with CUDA 12.8 support:
```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Note for RTX 5 series users**: Due to compatibility issues between torchcodec and nightly PyTorch builds, this script uses `decord` for video loading instead of `torchcodec` when using nightly PyTorch versions.

4. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

5. Run the script:
```bash
python app.py
```

## How it works

1. **Video Loading**: Downloads and loads a sample video from Kinetics dataset using Decord VideoReader
2. **Frame Sampling**: Extracts 64 frames from the video with automatic length checking
3. **Preprocessing**: Converts frames to the format expected by the V-JEPA 2 model using AutoVideoProcessor
4. **Feature Extraction**: Uses V-JEPA 2 model to extract video embeddings with `get_vision_features()`
5. **Output**: Prints the shape of the extracted video features

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Decord for video decoding (more compatible with nightly PyTorch builds)
- Transformers library (latest version from GitHub)
- CUDA (optional, for GPU acceleration)
- Internet connection (for downloading the model and sample video)

## Model

The script uses `facebook/vjepa2-vith-fpc64-256`, a V-JEPA 2 (Video Joint Embedding Predictive Architecture) model with Vision Transformer backbone, designed for self-supervised video representation learning.

## Output

The script outputs the shape of the extracted video embeddings, which can be used for downstream tasks like video classification, retrieval, or analysis.

## Troubleshooting

- **Internet connection**: Make sure you have a stable internet connection for downloading the model and video
- **Memory issues**: If you encounter memory issues, try reducing the number of frames processed by modifying the `frame_idx` array
- **GPU issues**: The script will automatically use the model's default device; ensure CUDA is available for GPU acceleration 