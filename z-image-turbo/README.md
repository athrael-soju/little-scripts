# Z-Image-Turbo

A high-performance image generation application powered by the Z-Image-Turbo model with a Gradio interface for text-to-image synthesis.

## Features

- **Text-to-Image Generation**: Generate high-quality images from text prompts using the Z-Image-Turbo diffusion transformer model
- **Multiple Resolutions**: Support for 20+ resolution options across 1024px and 1280px bases with various aspect ratios (1:1, 4:3, 3:2, 16:9, 21:9, and more)
- **Prompt Enhancement**: AI-powered prompt expansion via Qwen API for better image generation
- **Performance Optimization**: Automatic Flash Attention 2/3 detection and optional PyTorch compilation
- **Clean Web Interface**: Intuitive Gradio-based UI with preset example prompts

## Requirements

- Python 3.8+
- NVIDIA CUDA-compatible GPU
- Minimum 12GB VRAM recommended

## Installation

1. **Navigate to the project directory:**

   ```bash
   cd z-image-turbo
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

4. **Optional: Install Flash Attention for faster inference:**

   ```bash
   bash install_flash_attn.sh
   ```

5. **Configure environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Configuration

Edit the `.env` file to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `Tongyi-MAI/Z-Image-Turbo` | HuggingFace model ID or local path |
| `ATTENTION_BACKEND` | `auto` | Attention backend: `auto`, `flash`, `_flash_3`, `native`, `xformers`, `sage`, `flex` |
| `ENABLE_COMPILE` | `false` | Enable PyTorch compilation (faster after warmup) |
| `ENABLE_WARMUP` | `false` | Precompile all resolutions at startup |
| `DASHSCOPE_API_KEY` | - | API key for prompt enhancement (optional) |

## Usage

1. **Start the application:**

   ```bash
   python app.py
   ```

2. **Open your browser to:**

   ```
   http://localhost:7860
   ```

3. **Generate images:**
   - Enter a text prompt describing your desired image
   - Select resolution category (1024px or 1280px base)
   - Choose aspect ratio
   - Adjust inference steps and time shift if needed
   - Click "Generate"

## Supported Resolutions

**1024px Base:**
- 1024x1024 (1:1), 1152x896 (9:7), 896x1152 (7:9)
- 1152x864 (4:3), 864x1152 (3:4)
- 1248x832 (3:2), 832x1248 (2:3)
- 1280x720 (16:9), 720x1280 (9:16)
- 1344x576 (21:9), 576x1344 (9:21)

**1280px Base:**
- 1280x1280 (1:1), 1440x1120 (9:7), 1120x1440 (7:9)
- 1472x1104 (4:3), 1104x1472 (3:4)
- 1536x1024 (3:2), 1024x1536 (2:3)
- 1600x896 (16:9), 896x1600 (9:16)
- 1680x720 (21:9), 720x1680 (9:21)

## Performance Tips

- **Install Flash Attention**: Significantly accelerates inference when available
- **Enable Compilation**: Set `ENABLE_COMPILE=true` for faster generation after initial warmup
- **Enable Warmup**: Set `ENABLE_WARMUP=true` to eliminate first-generation delays (increases startup time)
- **Reduce Resolution**: If running low on VRAM, use smaller resolutions

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce resolution, close GPU-using applications, verify 12GB+ VRAM |
| Slow generation | Install Flash Attention, enable compilation, enable warmup |
| Model loading fails | Check disk space, network connectivity, CUDA installation |
| API enhancement fails | Verify `DASHSCOPE_API_KEY` is set correctly |

## Project Structure

```
z-image-turbo/
├── app.py              # Application entry point
├── config.py           # Configuration management
├── models.py           # Model initialization and loading
├── generator.py        # Image synthesis logic
├── prompt_expander.py  # AI prompt enhancement
├── ui.py               # Gradio interface components
├── utils.py            # Utility functions
├── pe.py               # Prompt engineering template
├── requirements.txt    # Python dependencies
├── .env.example        # Configuration template
├── install_flash_attn.sh # Flash Attention installer
└── .gitignore          # Git ignore rules
```

## Attribution

- **Model**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- **UI Framework**: [Gradio](https://gradio.app/)
- **Backend**: [Diffusers](https://github.com/huggingface/diffusers)
