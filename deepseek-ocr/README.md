# DeepSeek OCR Service

A FastAPI microservice that wraps `deepseek-ai/DeepSeek-OCR` with a GPU-friendly API for multi-format document analysis. Upload images or PDFs, choose a processing profile, and receive cleaned text, Markdown with inline figures, bounding boxes, and annotated previews in one response.

## Features

- Five processing modes (`Gundam`, `Tiny`, `Small`, `Base`, `Large`) to balance latency and fidelity.
- Task-aware prompts for markdown conversion, plain OCR, locate queries, descriptive captions, or fully custom instructions.
- PDF pipeline rasterizes each page at 300 DPI, runs OCR page-by-page, and aggregates the results.
- Optional grounding output that includes structured bounding boxes, cropped figure snippets, and an annotated overview image.
- REST endpoints for health, service metadata, and uploads, secured with configurable CORS rules.

## Requirements

- Python 3.10+
- CUDA-capable GPU (Flash Attention 2 is enabled automatically when CUDA is available)
- NVIDIA drivers + `nvidia-container-toolkit` if you plan to run the Docker deployment
- Hugging Face credentials if the configured model is private

## Quick Start

### Local (Python)

1. Install dependencies and set up the environment:
   ```bash
   cd deepseek-ocr
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   ```
2. (Optional) edit `.env` to point at a different model, port, or allowed origins.
3. Launch the API:
   ```bash
   python main.py
   ```
4. Browse to `http://localhost:8200/docs` for the interactive Swagger UI.

### Docker (GPU)

```bash
cd deepseek-ocr
docker compose up --build
```

The Compose file builds the service image, mounts a persistent Hugging Face cache (`hf-cache` volume), and reserves all visible NVIDIA GPUs.

## API Overview

| Method | Path      | Description                                   |
|--------|-----------|-----------------------------------------------|
| GET    | `/health` | Readiness probe with model, device, dtype     |
| GET    | `/info`   | Lists processing modes, tasks, and configs    |
| POST   | `/api/ocr`| Upload an image/PDF and receive OCR results   |

### Example Request

```bash
curl -X POST "http://localhost:8200/api/ocr" \
  -F "image=@sample.png" \
  -F "mode=Gundam" \
  -F "task=markdown" \
  -F "include_grounding=true" \
  -F "include_images=true"
```

Successful responses contain:

- `text`: cleaned plain text with optional grounding markers
- `markdown`: Markdown output with inline, base64-encoded figures (if requested)
- `raw`: untouched model output for debugging
- `bounding_boxes`: structured coordinates (`x1`, `y1`, `x2`, `y2`, `label`)
- `crops`: base64 figure snippets
- `annotated_image`: base64 image with bounding boxes overlaid

## Configuration

Settings can be supplied via `.env` (recommended) or environment variables. Defaults live in `app/core/config.py`.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `deepseek-ai/DeepSeek-OCR` | Hugging Face identifier for the checkpoint |
| `DEVICE` | Auto-detected (`cuda` if available) | Override to force CPU/GPU |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | `/models` (local) / `/data/hf-cache` (Docker) | Shared cache for model weights |
| `API_HOST` | `0.0.0.0` | Bind address for FastAPI |
| `API_PORT` | `8200` | Service port |
| `ALLOWED_ORIGINS` | `*` | Comma-separated list for CORS |
| `MAX_UPLOAD_SIZE_MB` | `100` | Hard limit enforced by the reverse proxy / server |

## Processing Modes

| Mode    | Base Size | Image Size | Crop Mode | When to use |
|---------|-----------|------------|-----------|-------------|
| Gundam  | 1024      | 640        | Yes       | Default profile with tight crops for dense pages |
| Tiny    | 512       | 512        | No        | Lowest VRAM footprint for previews |
| Small   | 640       | 640        | No        | Balanced throughput for receipts or slides |
| Base    | 1024      | 1024       | No        | High quality without aggressive cropping |
| Large   | 1280      | 1280       | No        | Maximal fidelity (requires more VRAM) |

### Task Presets

- `markdown`: `<|grounding|>` prompt that outputs Markdown + figure markers
- `plain_ocr`: plain text without grounding anchors
- `describe`: free-form caption describing the visual content
- `locate`: `<|ref|>` prompt that highlights user-provided text snippets
- `custom`: raw prompt provided by the request (`custom_prompt` form field)

Tasks that require a custom prompt (`locate`, `custom`) must include `custom_prompt` in the upload form. Grounding data is only included when the prompt contains `<|grounding|>` or uses a task preset that enables it.

## Project Structure

- `main.py` - Uvicorn entry point that bootstraps the FastAPI app.
- `app/main.py` - Application factory, CORS middleware, startup/shutdown hooks.
- `app/api/routes.py` - `/health`, `/info`, and `/api/ocr` endpoints plus file validation.
- `app/services/model_service.py` - Hugging Face model loader with Flash Attention 2 toggles.
- `app/services/ocr_processor.py` - Prompt selection, PDF rasterization, post-processing helpers.
- `app/utils/image_processing.py` - Grounding reference parsing, bounding boxes, base64 utilities.

## Troubleshooting

- **Model download stalls** - Ensure `huggingface-cli login` is configured if the repo is gated, and confirm the cache path is writable inside containers.
- **CUDA out-of-memory** - Switch to `Small` or `Tiny` mode, reduce the PDF DPI, or close other GPU workloads.
- **CPU fallback is slow** - The service runs on CPU, but latency increases significantly because Flash Attention 2 is disabled. Confirm `torch.cuda.is_available()` for best performance.
- **No bounding boxes in responses** - Only prompts containing `<|grounding|>` (e.g., `markdown`, `locate`) emit grounding metadata. Ensure `include_grounding=true` in the request.

Happy OCR'ing!
