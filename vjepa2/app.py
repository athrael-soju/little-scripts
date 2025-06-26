import torch
import numpy as np
from transformers import AutoVideoProcessor, AutoModel
import requests
import tempfile
import os
from decord import VideoReader, cpu
from huggingface_hub import snapshot_download
from transformers import utils as transformers_utils

# Device detection and setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Display cache directory
cache_dir = transformers_utils.default_cache_path
print(f"HuggingFace cache directory: {cache_dir}")

hf_repo = "facebook/vjepa2-vitg-fpc64-256"
# facebook/vjepa2-vitl-fpc64-256
# facebook/vjepa2-vith-fpc64-256
# facebook/vjepa2-vitg-fpc64-256
# facebook/vjepa2-vitg-fpc64-384

print(f"Loading model from {hf_repo}...")
print("Note: Model will be cached in HuggingFace default cache directory for future use")

# Explicitly use default cache directory and ensure model is downloaded
model = AutoModel.from_pretrained(
    hf_repo,
    cache_dir=None,  # Use default cache
    local_files_only=False,  # Allow downloading if not cached
    force_download=False  # Don't re-download if already cached
)
model = model.to(device)  # Move model to GPU if available

processor = AutoVideoProcessor.from_pretrained(
    hf_repo,
    cache_dir=None,  # Use default cache
    local_files_only=False,  # Allow downloading if not cached
    force_download=False  # Don't re-download if already cached
)

print("Model and processor loaded successfully from cache")

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

print("Downloading video...")
# Download video to temporary file
response = requests.get(video_url)
with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
    tmp_file.write(response.content)
    tmp_video_path = tmp_file.name

try:
    print("Processing video frames...")
    # Load video using decord
    vr = VideoReader(tmp_video_path, ctx=cpu(0))
    frame_idx = np.arange(0, min(64, len(vr)))  # ensure we don't exceed video length
    video_frames = vr.get_batch(frame_idx).asnumpy()  # T x H x W x C
    video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2)  # T x C x H x W
    
    print("Running inference...")
    video = processor(video_frames, return_tensors="pt")
    # Move video tensors to the same device as model
    video = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in video.items()}
    
    with torch.no_grad():
        video_embeddings = model.get_vision_features(**video)

    print(f"Video embeddings shape: {video_embeddings.shape}")
    print(f"Embeddings device: {video_embeddings.device}")
    
finally:
    # Clean up temporary file
    os.unlink(tmp_video_path)
    print("Cleanup completed.")