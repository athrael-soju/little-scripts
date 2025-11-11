# DeepSeek OCR Service

A FastAPI microservice that wraps the `deepseek-ai/DeepSeek-OCR` model with a GPU-friendly API for multi-format document analysis. Upload single images or PDFs, choose between speed/quality profiles, and receive structured text, Markdown with inline figures, bounding boxes, and annotated previews in one response.

## Features

- Multi-mode processing (`Gundam`, `Tiny`, `Small`, `Base`, `Large`) to trade off speed vs. fidelity.
- Task-specific prompts for `markdown`, `plain_ocr`, `describe`, `locate`, and fully custom instructions.
- PDF ingestion pipeline that rasterizes every page at 300 DPI and aggregates results.
- Optional bounding boxes (`<|grounding|>`) with annotated preview images plus cropped regions as base64.
- Automatic figure extraction with inline Markdown embeds for downstream readers or notebooks.
- REST endpoints for health, model metadata, and file uploads, protected with configurable CORS settings.

## Requirements

- Python 3.10+
- CUDA-capable GPU (Flash Attention 2 is enabled automatically when `torch.cuda.is_available()`).
- NVIDIA drivers + `nvidia-container-toolkit` if you plan to use Docker with GPU passthrough.
- Hugging Face credentials configured globally if the model is private (defaults to public weights).

## Setup

```bash
# From repo root
cd deepseek-ocr

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and adjust as needed
cp .env.example .env
```

Key variables inside `.env`:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MODEL_NAME` | Hugging Face identifier for the OCR checkpoint | `deepseek-ai/DeepSeek-OCR` |
| `HF_HOME` | On-disk cache for model weights | `/models` |
| `API_HOST` / `API_PORT` | Where FastAPI listens | `0.0.0.0` / `8200` |
| `ALLOWED_ORIGINS` | Comma-separated list used by CORS middleware | `*` |
| `MAX_UPLOAD_SIZE_MB` | Upper bound enforced by the reverse proxy / server | `100` |

Start the API:

```bash
python main.py
```

The server logs when the model is ready and publishes links to `/docs` and `/health`.

## Docker (GPU ready)

The included `docker-compose.yml` builds the app image, mounts a persistent Hugging Face cache, and reserves every visible NVIDIA GPU:

```bash
cd deepseek-ocr
docker compose up --build
```

Environment variables are read from `.env`. The compose file also sets `HUGGINGFACE_HUB_CACHE`/`HF_HOME` to `/data/hf-cache` inside the container so repeated startups reuse downloads.

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness/readiness plus model/device metadata. |
| `GET` | `/info` | Enumerates processing modes, task types, and the active configuration. |
| `POST` | `/api/ocr` | Accepts a file upload (`image` form field) plus optional form fields described below. |

`POST /api/ocr` form fields:

- `mode`: One of `Gundam`, `Tiny`, `Small`, `Base`, `Large` (default: `Gundam`).
- `task`: `markdown`, `plain_ocr`, `locate`, `describe`, or `custom` (default: `markdown`).
- `custom_prompt`: Required when `task` is `custom` or `locate`; may include `<|grounding|>` tokens.
- `include_grounding`: `true`/`false`, determines whether bounding boxes are extracted.
- `include_images`: Controls figure cropping + Markdown embedding.

Response payload (`OCRResponse`) bundles cleaned text, Markdown, raw model output, bounding boxes, cropped figures (base64), and an optional annotated image preview.

### Example request

```bash
curl -X POST "http://localhost:8200/api/ocr" \
  -F "image=@sample-doc.png" \
  -F "mode=Gundam" \
  -F "task=markdown" \
  -F "include_grounding=true" \
  -F "include_images=true"
```

## Processing Modes & Tasks

| Mode | Base Size | Image Size | Crop Mode | When to use |
| --- | --- | --- | --- | --- |
| `Gundam` | 1024 | 640 | Yes | Best text fidelity with tight crops (default). |
| `Tiny` | 512 | 512 | No | Fastest inference, good for previews. |
| `Small` | 640 | 640 | No | Balanced for small receipts/slides. |
| `Base` | 1024 | 1024 | No | High quality for dense documents. |
| `Large` | 1280 | 1280 | No | Highest resolution (requires more VRAM). |

Task helpers live in `app/core/config.py` and encapsulate canonical prompts. `markdown` keeps figure placeholders, `plain_ocr` returns unformatted text, `locate` builds `<|ref|>` prompts automatically, `describe` is image-caption style, and `custom` hands full control to you.

## Development Notes

- `/app/services/model_service.py` handles lazy loading, toggles Flash Attention 2 when GPUs are available, and sanitizes stdout from the upstream `infer` helper.
- `/app/services/ocr_processor.py` orchestrates page rasterization, prompt selection, and post-processing (cleaning, bounding boxes, base64 helpers).
- Unit-style experimentation is easiest via the FastAPI docs UI at `http://localhost:8200/docs`.

## Troubleshooting

- **Model download stalls**: ensure you have `git lfs` + valid Hugging Face credentials if the repo is gated.
- **OOM on GPU**: switch to `Small`/`Tiny` mode or reduce PDF DPI; the default 300 DPI rendering is memory hungry.
- **No bounding boxes**: grounding only appears when the prompt contains `<|grounding|>` tokens (`markdown`, `locate`, or custom prompts that include them).
- **CPU fallback**: the service runs on CPU, but inference is slow and Flash Attention 2 is disabled; confirm `torch.cuda.is_available()` for best results.

Happy hacking!
