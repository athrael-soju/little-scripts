# PaddleOCR-VL FastAPI Service

AA GPU-accelerated document-understanding service, based on PaddleOCR-VL. It exposes a single `/api/v1/ocr/extract-document` endpoint that accepts images or PDFs and returns structured JSON and Markdown representations of the detected content. Inspired by: https://github.com/Mark-Engineering-Inc/paddleocr-vl-service and transformed to work on a single 50XX GPU, or less.

## Features

- GPU-aware, lazily initialised PaddleOCR-VL pipeline
- Handles images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`) and PDF uploads up to 50 MB by default
- Returns structured blocks with bounding boxes, aggregated plain text, and Markdown
- Provides health and status endpoints for operational monitoring
- Docker image and Compose file tuned for CUDA 13.0 runtimes

## Quick Start

### Local (Python)

> **Prerequisite:** Install the matching PaddlePaddle GPU wheel prior to installing the service dependencies. For CUDA 13.0, run:
>
> ```bash
> python -m pip install --upgrade pip
> python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
> ```

1. Install requirements:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Export optional overrides (see [Configuration](#configuration)):
   ```bash
   export APP_PORT=8100
   export LOG_LEVEL=INFO
   ```
3. Start the API:
   ```bash
   python main.py
   ```
4. Visit `http://localhost:8100/api/v1/docs` for interactive documentation.

### Docker

```bash
docker compose up --build
```

The Compose file already wires GPU resources, health checks, and a shared volume for model caching.

## API Overview

| Method | Path                         | Description                               |
|--------|------------------------------|-------------------------------------------|
| GET    | `/`                          | Basic service metadata                    |
| GET    | `/health`                    | Health probe including pipeline status    |
| POST   | `/api/v1/ocr/extract-document` | Upload an image/PDF and receive OCR output |

### Example Request

```bash
curl -X POST \"http://localhost:8100/api/v1/ocr/extract-document\" \
  -F \"file=@sample.pdf\"
```

Successful responses include:

- `elements`: array of structured document blocks (type, bbox, text)
- `markdown`: aggregated Markdown rendering
- `processing_time`: end-to-end latency (seconds)

Errors return an `ErrorResponse` payload with a descriptive message and type.

## Configuration

Settings are provided via environment variables or an `.env` file. Defaults are defined in `config/settings.py`.

| Variable              | Default                    | Description                                   |
|-----------------------|----------------------------|-----------------------------------------------|
| `APP_NAME`            | `PaddleOCR-VL Service`     | Display name                                   |
| `APP_VERSION`         | `1.0.0`                    | Semantic version                               |
| `APP_PORT`            | `8100`                     | Application port                               |
| `APP_HOST`            | `0.0.0.0`                  | Bind address                                   |
| `DEBUG`               | `false`                    | Enables Uvicorn reload mode                    |
| `API_V1_PREFIX`       | `/api/v1`                  | Base path for versioned routes                 |
| `MAX_UPLOAD_SIZE`     | `52428800`                 | Upload limit (bytes)                           |
| `ALLOWED_EXTENSIONS`  | `.jpg,.jpeg,.png,...`      | Allowed file suffixes                          |
| `USE_GPU`             | `true`                     | Toggle GPU usage in PaddleOCR-VL               |
| `DEVICE`              | `gpu`                      | Device hint (`gpu` or `cpu`)                   |
| `LOG_LEVEL`           | `INFO`                     | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_FORMAT`          | `json`                     | `json` or `text`                               |

## Project Structure

- `main.py` – FastAPI entrypoint wiring routers, logging, and lifespan hooks
- `routers/ocr_router.py` – Upload validation and API contract
- `services/paddleocr_vl_service.py` – Lazily instantiated PaddleOCR-VL service
- `config/` – Pydantic settings and structured logging helpers
- `Dockerfile`, `docker-compose.yml` – GPU-ready deployment manifests

## Refactored Service Highlights

- Injectable pipeline factory makes the service unit-test friendly
- Thread-safe lazy loading keeps startup lightweight while supporting concurrency
- `DocumentElement` dataclass centralises serialisation logic for API responses
- Markdown output builder now reuses structured results to preserve ordering

## Troubleshooting

- **`OSError: (No such file or directory)`** – Ensure the required PaddlePaddle GPU wheel matches the host CUDA version before installing `paddleocr` dependencies.
- **Slow first request** – The first request loads PaddleOCR-VL models. Subsequent calls reuse the initialised pipeline.
- **CPU-only systems** – Set `USE_GPU=false` and `DEVICE=cpu`. Expect higher latency without GPU acceleration.
