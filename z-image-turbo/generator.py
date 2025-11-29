"""Image generation logic for Z-Image-Turbo application."""

import torch
import gradio as gr
from diffusers import FlowMatchEulerDiscreteScheduler
from utils import get_resolution


def generate_image(
    pipe,
    prompt,
    resolution="1024x1024",
    seed=42,
    guidance_scale=5.0,
    num_inference_steps=50,
    shift=3.0,
    max_sequence_length=1024,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate an image using the Z-Image pipeline.

    Args:
        pipe: ZImagePipeline instance
        prompt (str): Text prompt describing the desired image
        resolution (str): Resolution in format "WIDTHxHEIGHT"
        seed (int): Random seed for reproducibility
        guidance_scale (float): Classifier-free guidance scale
        num_inference_steps (int): Number of denoising steps
        shift (float): Time shift parameter for the flow matching scheduler
        max_sequence_length (int): Maximum sequence length for text encoding
        progress: Gradio progress tracker

    Returns:
        PIL.Image: Generated image
    """
    width, height = get_resolution(resolution)

    generator = torch.Generator("cuda").manual_seed(seed)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    pipe.scheduler = scheduler

    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        max_sequence_length=max_sequence_length,
    ).images[0]

    return image


def warmup_model(pipe, resolutions):
    """
    Warm up the model by generating dummy images at various resolutions.

    Args:
        pipe: ZImagePipeline instance
        resolutions (list): List of resolution strings to warm up
    """
    print("Starting warmup phase...")

    dummy_prompt = "warmup"

    for res_str in resolutions:
        print(f"Warming up for resolution: {res_str}")
        try:
            for i in range(3):
                generate_image(
                    pipe,
                    prompt=dummy_prompt,
                    resolution=res_str,
                    num_inference_steps=9,
                    guidance_scale=0.0,
                    seed=42 + i,
                )
        except Exception as e:
            print(f"Warmup failed for {res_str}: {e}")

    print("Warmup completed.")
