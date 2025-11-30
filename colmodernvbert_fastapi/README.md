# ColModernVBert FastAPI Service

A high-performance FastAPI service for generating embeddings from images and text queries using the ColModernVBert model from the ColPali engine.

## Overview

This service provides a RESTful API for generating multimodal embeddings optimized for document retrieval and visual understanding tasks. It leverages the ColModernVBert model to create rich vector representations of both images and text queries, enabling powerful similarity search and document understanding applications.

## Features

- **Image Embeddings**: Generate embeddings for images with automatic boundary detection
- **Query Embeddings**: Create embeddings for text queries optimized for retrieval
- **Patch Calculation**: Calculate patch numbers for given image dimensions
- **Interpretability Maps**: Generate visual explanations showing query-document token correspondence
- **GPU Acceleration**: Supports Flash Attention 2 for optimal performance
- **Service Management**: Built-in restart endpoint for container orchestration
- **Health Monitoring**: Comprehensive health checks and model information endpoints
- **Structured Logging**: Request IDs and timing middleware for observability

## API Endpoints

### Core Endpoints

- `GET /` - Service information and version
- `GET /health` - Health check with device information
- `GET /info` - Detailed model configuration and capabilities

### Embedding Generation

- `POST /embed/queries` - Generate embeddings for text queries
  - Input: Single query string or list of queries
  - Output: List of query embeddings

- `POST /embed/images` - Generate embeddings for uploaded images
  - Input: One or more image files (JPEG/PNG/WebP)
  - Output: Embeddings with image-token boundaries

### Advanced Features

- `POST /patches` - Calculate patch dimensions for image sizes
  - Input: List of image dimensions (width, height)
  - Output: Patch counts (n_patches_x, n_patches_y) for each dimension

- `POST /interpret` - Generate interpretability maps
  - Input: Query text + image file
  - Output: Per-token similarity maps showing document regions that contribute to each query token

### Management

- `POST /restart` - Gracefully restart the service (requires container restart policy)

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, 12GB+ VRAM)
- Docker (optional, for containerized deployment)

### Local Setup

1. **Clone the repository:**

   ```bash
   cd colmodernvbert_fastapi
   ```

2. **Install PyTorch with CUDA support:**

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install Flash Attention 2 (optional, for better performance):**

   ```bash
   pip install flash-attn --no-build-isolation
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the service:**

   ```bash
   python main.py
   ```

   The service will be available at `http://0.0.0.0:7000`

### Docker Deployment

1. **Build the Docker image:**

   ```bash
   docker compose build
   ```

2. **Run the service:**

   ```bash
   docker compose up -d
   ```

3. **Check logs:**

   ```bash
   docker compose logs -f
   ```

## Configuration

The service can be configured via environment variables or the `.env` file:

```bash
# Model Configuration
MODEL_ID=vidore/colmodernvbert  # HuggingFace model ID

# Device Configuration
DEVICE=cuda                      # cuda, cpu, or specific device like cuda:0
TORCH_DTYPE=bfloat16            # bfloat16, float16, or float32

# CPU Configuration (if DEVICE=cpu)
CPU_THREADS=4                    # Number of CPU threads for inference

# API Configuration
API_VERSION=1.0.0
API_HOST=0.0.0.0
API_PORT=7000
```

## Usage Examples

### Generate Query Embeddings

```bash
curl -X POST "http://localhost:7000/embed/queries" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is the main topic of this document?", "Find financial data"]
  }'
```

### Generate Image Embeddings

```bash
curl -X POST "http://localhost:7000/embed/images" \
  -F "files=@document1.jpg" \
  -F "files=@document2.png"
```

### Calculate Patches

```bash
curl -X POST "http://localhost:7000/patches" \
  -H "Content-Type: application/json" \
  -d '{
    "dimensions": [
      {"width": 1024, "height": 768},
      {"width": 1920, "height": 1080}
    ]
  }'
```

### Generate Interpretability Maps

```bash
curl -X POST "http://localhost:7000/interpret" \
  -F "query=financial summary" \
  -F "file=@document.jpg"
```

### Get Model Information

```bash
curl http://localhost:7000/info
```

## Architecture

### Project Structure

```
colmodernvbert_fastapi/
├── app/
│   ├── api/
│   │   └── routes.py              # API endpoint handlers
│   ├── core/
│   │   ├── config.py              # Configuration management
│   │   └── logging.py             # Logging setup
│   ├── models/
│   │   └── schemas.py             # Pydantic models
│   ├── services/
│   │   ├── model_service.py       # Model loading and management
│   │   └── embedding_processor.py # Embedding generation logic
│   ├── utils/
│   │   └── image_processing.py    # Image utilities
│   └── main.py                    # FastAPI app factory
├── middleware/
│   ├── request_id.py              # Request ID injection
│   └── timing.py                  # Response time logging
├── main.py                        # Service entry point
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### Key Components

- **ModelService**: Manages ColModernVBert model loading and configuration
- **EmbeddingProcessor**: Handles embedding generation and interpretability
- **ThreadPoolExecutors**: Separate pools for query and image processing to prevent blocking
- **Middleware**: Request tracking and performance monitoring

## Performance Optimization

### Flash Attention 2

The service automatically enables Flash Attention 2 if available, providing significant speed improvements:

```bash
pip install flash-attn --no-build-isolation
```

### GPU Memory Management

For optimal performance with limited GPU memory:

1. Use `bfloat16` or `float16` precision
2. Process images in smaller batches
3. Monitor memory usage with `nvidia-smi`

### CPU Mode

For CPU-only deployments, configure thread count:

```bash
DEVICE=cpu
CPU_THREADS=8  # Adjust based on available cores
```

## API Documentation

Interactive API documentation is available at:

- Swagger UI: `http://localhost:7000/docs`
- ReDoc: `http://localhost:7000/redoc`

## Middleware Features

### Request ID Tracking

Every request receives a unique `X-Request-ID` header for tracing and debugging:

```
X-Request-ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

### Response Timing

Response times are automatically logged for performance monitoring:

```
INFO: Request completed - method=POST path=/embed/queries duration=0.245s
```

## Health Checks

The `/health` endpoint provides service status:

```json
{
  "status": "healthy",
  "device": "cuda:0"
}
```

Use this for container orchestration health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:7000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

## Troubleshooting

### Model Loading Issues

If the model fails to load:

1. **Check CUDA availability:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Verify HuggingFace access:**
   - Some models require authentication
   - Set `HF_TOKEN` environment variable if needed

3. **Monitor memory:**
   - ColModernVBert requires ~12GB VRAM
   - Use smaller batch sizes or CPU mode if needed

### Flash Attention Installation

If Flash Attention 2 installation fails:

```bash
# Ensure CUDA toolkit is installed
nvcc --version

# Install with specific CUDA version
pip install flash-attn --no-build-isolation
```

### Permission Issues

For Docker deployments with GPU:

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Restart Docker daemon
sudo systemctl restart docker
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .
```

## License

Open source - feel free to use and modify as needed.

## Related Projects

- [ColPali](https://github.com/illuin-tech/colpali) - The underlying ColPali engine
- [ColQwen2.5 FastAPI](../colqwen_fastapi/) - Alternative embedding service with ColQwen2.5
- [DuckDB FastAPI](../duckdb_fastapi/) - Columnar analytics for OCR data

## Acknowledgments

Built on the excellent [ColPali engine](https://github.com/illuin-tech/colpali) by Illuin Technology.
