"""Model loading and management for Z-Image-Turbo application."""

import os
import torch
from diffusers import AutoencoderKL
from transformers import AutoModel, AutoTokenizer
from diffusers import ZImagePipeline
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel


def load_models(model_path, enable_compile=False, attention_backend="native"):
    """
    Load and initialize all required models for Z-Image generation.

    Args:
        model_path (str): Path to the model directory or HuggingFace model ID
        enable_compile (bool): Whether to enable torch.compile optimizations
        attention_backend (str): Attention backend to use ("flash", "_flash_3", or "native")

    Returns:
        ZImagePipeline: Initialized pipeline ready for image generation
    """
    print(f"Loading models from {model_path}...")

    if not os.path.exists(model_path):
        vae = AutoencoderKL.from_pretrained(
            f"{model_path}",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        text_encoder = AutoModel.from_pretrained(
            f"{model_path}",
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            f"{model_path}", subfolder="tokenizer"
        )
    else:
        vae = AutoencoderKL.from_pretrained(
            os.path.join(model_path, "vae"),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        text_encoder = AutoModel.from_pretrained(
            os.path.join(model_path, "text_encoder"),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))

    tokenizer.padding_side = "left"

    if enable_compile:
        print("Enabling torch.compile optimizations...")
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        torch._inductor.config.max_autotune_gemm = True
        torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"
        torch._inductor.config.triton.cudagraphs = False

    pipe = ZImagePipeline(
        scheduler=None,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
    )

    if enable_compile:
        pipe.vae.disable_tiling()

    if not os.path.exists(model_path):
        transformer = ZImageTransformer2DModel.from_pretrained(
            f"{model_path}", subfolder="transformer"
        ).to("cuda", torch.bfloat16)
    else:
        transformer = ZImageTransformer2DModel.from_pretrained(
            os.path.join(model_path, "transformer")
        ).to("cuda", torch.bfloat16)

    pipe.transformer = transformer
    pipe.transformer.set_attention_backend(attention_backend)

    if enable_compile:
        print("Compiling transformer...")
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune-no-cudagraphs", fullgraph=False
        )

    pipe.to("cuda", torch.bfloat16)

    return pipe
