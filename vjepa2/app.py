import torch
import numpy as np
from transformers import AutoVideoProcessor, AutoModel
import requests
import tempfile
import os
from decord import VideoReader, cpu

hf_repo = "facebook/vjepa2-vith-fpc64-256"

model = AutoModel.from_pretrained(hf_repo)
processor = AutoVideoProcessor.from_pretrained(hf_repo)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

# Download video to temporary file
response = requests.get(video_url)
with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
    tmp_file.write(response.content)
    tmp_video_path = tmp_file.name

try:
    # Load video using decord
    vr = VideoReader(tmp_video_path, ctx=cpu(0))
    frame_idx = np.arange(0, min(64, len(vr)))  # ensure we don't exceed video length
    video_frames = vr.get_batch(frame_idx).asnumpy()  # T x H x W x C
    video_frames = torch.from_numpy(video_frames).permute(0, 3, 1, 2)  # T x C x H x W
    
    video = processor(video_frames, return_tensors="pt").to(model.device)
    with torch.no_grad():
        video_embeddings = model.get_vision_features(**video)

    print(video_embeddings.shape)
    
finally:
    # Clean up temporary file
    os.unlink(tmp_video_path)