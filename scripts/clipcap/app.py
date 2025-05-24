#!/usr/bin/env python3
"""
ClipCap Image Captioning Script

This script uses ClipCap to generate captions for images using a combination 
of CLIP and GPT-2 models.
"""

import os
import sys
from pathlib import Path
import torch
import clip
import requests
from PIL import Image
from transformers import GPT2Tokenizer

try:
    from clipcap import ClipCaptionModel
except ImportError:
    print("Error: clipcap module not found. Please install it with:")
    print("pip install clipcap")
    sys.exit(1)

def main():
    # Configuration
    model_path = "clipcap-base-captioning-ft-hl-narratives/pytorch_model.pt"
    img_url = 'images/atlas.jpg'
    prefix_length = 10
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please download the ClipCap model and update the model_path variable.")
        sys.exit(1)
    
    print("Loading models...")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load CLIP
        clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("✓ CLIP model loaded")
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print("✓ GPT-2 tokenizer loaded")
        
        # Load ClipCap
        model = ClipCaptionModel(prefix_length, tokenizer=tokenizer)
        model.from_pretrained(model_path)
        model = model.eval()
        model = model.to(device)
        print("✓ ClipCap model loaded")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)
    
    try:
        # Load and process image
        print(f"Downloading image from: {img_url}")
        response = requests.get(img_url, stream=True, timeout=30)
        response.raise_for_status()
        raw_image = Image.open(response.raw).convert('RGB')
        print("✓ Image loaded and converted")
        
        # Extract image features
        print("Extracting image features...")
        image = preprocess(raw_image).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        
        # Generate caption
        print("Generating caption...")
        caption = model.generate_beam(embed=prefix_embed)[0]
        
        print(f"\nGenerated Caption: {caption}")
        
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image or generating caption: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
