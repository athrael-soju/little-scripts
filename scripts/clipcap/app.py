#!/usr/bin/env python3
"""
ClipCap Image Captioning Script

This script uses ClipCap to generate captions for images using a combination 
of CLIP and GPT-2 models.
"""

import os
import sys
import argparse
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

def download_model(model_path, model_url):
    """Download the ClipCap model if it doesn't exist."""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Downloading model from {model_url}...")
    print("This may take a few minutes...")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✓ Model downloaded successfully to {model_path}")
        
    except requests.RequestException as e:
        print(f"\nError downloading model: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError saving model: {e}")
        sys.exit(1)

def load_image(image_input):
    """Load image from either a local file path or URL."""
    try:
        # Check if input is a URL
        if image_input.startswith(('http://', 'https://')):
            print(f"Downloading image from: {image_input}")
            response = requests.get(image_input, stream=True, timeout=30)
            response.raise_for_status()
            image = Image.open(response.raw).convert('RGB')
            print("✓ Image downloaded and loaded")
        else:
            # Local file path
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_input}")
            
            print(f"Loading local image: {image_input}")
            image = Image.open(image_path).convert('RGB')
            print("✓ Local image loaded")
        
        return image
        
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate captions for images using ClipCap')
    parser.add_argument('image', nargs='?', 
                       help='Path to local image file or URL (if not provided, uses default example)')
    parser.add_argument('--model-path', default='models/pytorch_model.pt',
                       help='Path to the ClipCap model file (default: models/pytorch_model.pt)')
    
    args = parser.parse_args()
    
    # Configuration
    model_path = args.model_path
    model_url = "https://huggingface.co/michelecafagna26/clipcap-base-captioning-ft-hl-actions/resolve/main/pytorch_model.pt?download=true"
    
    # Use provided image or default example
    if args.image:
        image_input = args.image
    else:
        # Default example image
        image_input = 'https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png'
    
    prefix_length = 10
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        download_model(model_path, model_url)
    else:
        print(f"✓ Model file found at {model_path}")
    
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
        raw_image = load_image(image_input)
        
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
        
    except Exception as e:
        print(f"Error processing image or generating caption: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
