"""Z-Image-Turbo: High-performance image generation application."""

import os
import warnings

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Filter specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from config import (
    MODEL_PATH,
    ENABLE_COMPILE,
    ENABLE_WARMUP,
    ATTENTION_BACKEND,
    RESOLUTION_SET,
)
from models import load_models
from generator import warmup_model
from prompt_expander import create_prompt_expander
from ui import create_ui


def main():
    """Main entry point for Z-Image-Turbo application."""
    print("=" * 60)
    print("Z-Image-Turbo - Image Generation Application")
    print("=" * 60)

    # Load models
    print(f"\nConfiguration:")
    print(f"  Model Path: {MODEL_PATH}")
    print(f"  Attention Backend: {ATTENTION_BACKEND}")
    print(f"  Compile Enabled: {ENABLE_COMPILE}")
    print(f"  Warmup Enabled: {ENABLE_WARMUP}")
    print()

    pipe = load_models(
        model_path=MODEL_PATH,
        enable_compile=ENABLE_COMPILE,
        attention_backend=ATTENTION_BACKEND,
    )

    # Warm up the model if enabled
    if ENABLE_WARMUP:
        warmup_model(pipe, RESOLUTION_SET)

    # Initialize prompt expander
    prompt_expander = create_prompt_expander(backend="api", model="qwen3-max-preview")

    # Create and launch the UI
    print("\nStarting Gradio interface...")
    demo = create_ui(pipe, prompt_expander)
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
