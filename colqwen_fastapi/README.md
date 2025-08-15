# ColQwen2.5 FastAPI Service

A FastAPI-based service for generating embeddings from images and text queries using the ColQwen2.5 model.

## Features

- 🖼️ **Image Embeddings**: Generate embeddings for images
- 🔤 **Text Embeddings**: Generate embeddings for text queries
- 🚀 **High Performance**: Utilizes Flash Attention 2 when available
- 📊 **RESTful API**: Easy integration with other services
- 🏥 **Health Monitoring**: Built-in health check and version endpoints
- 📏 **Patch Calculation**: Utility endpoint for patch calculations
- 🔄 **Batch Processing**: Support for processing multiple images/queries in a single request

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) (optional, for better performance)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/athrael.soju/little-scripts.git
   cd little-scripts/colqwen_fastapi
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install Flash Attention 2 (optional, for better performance):
   ```bash
   pip install flash-attn --no-build-isolation
   ```

## Usage

### Starting the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 7000 --reload
```

### API Endpoints

#### Root Endpoint
- `GET /`: Returns a welcome message
  ```json
  {
    "message": "Welcome to ColQwen2.5 Embedding Service"
  }
  ```

#### Health Check
- `GET /health`: Health check endpoint
  ```json
  {
    "status": "healthy",
    "timestamp": "2025-08-12T16:43:30.123456"
  }
  ```

#### Version
- `GET /version`: Get service version information
  ```json
  {
    "version": "0.2.0",
    "model": "vidore/colqwen2.5-v0.2",
    "device": "cuda:0"
  }
  ```

#### Patch Calculation
- `POST /patches`: Calculate number of patches for given image dimensions
  **Request Body**:
  ```json
  {
    "width": 1024,
    "height": 768
  }
  ```
  **Response**:
  ```json
  {
    "n_patches_x": 32,
    "n_patches_y": 24
  }
  ```

#### Text Embeddings
- `POST /embed/queries`: Generate embeddings for text queries
  **Request Body**:
  ```json
  {
    "queries": ["a photo of a cat", "a photo of a dog"]
  }
  ```
  **Response**:
  ```json
  {
    "embeddings": [
      [[0.1, 0.2, ...], ...],
      [[0.3, 0.4, ...], ...]
    ]
  }
  ```

#### Image Embeddings
- `POST /embed/images`: Generate embeddings for uploaded images
  **Request**: `multipart/form-data` with image files
  **Response**:
  ```json
  {
    "embeddings": [
      [[0.1, 0.2, ...], ...],
      [[0.3, 0.4, ...], ...]
    ]
  }
  ```

## Error Handling

The API returns appropriate HTTP status codes and error messages in JSON format:

- `400 Bad Request`: Invalid input parameters
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server-side error

## Development

To run the development server with auto-reload:

```bash
uvicorn app:app --reload
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Model Information

This service uses the `vidore/colqwen2.5-v0.2` model, which is based on ColQwen2.5 architecture.
- `GET /health`: Health check
- `GET /version`: Get API version
- `POST /patches`: Calculate number of patches for given image dimensions
- `POST /embed/queries`: Generate embeddings for text queries
- `POST /embed/images`: Generate embeddings for uploaded images

### Example Requests

#### Get Embeddings for Text Queries

```bash
curl -X POST "http://localhost:7000/embed/queries" \
     -H "Content-Type: application/json" \
     -d '{"queries": ["example query"]}'
```

#### Get Embeddings for Images

```bash
curl -X POST "http://localhost:7000/embed/images" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```

## API Documentation

### Interactive Documentation

Once the server is running, visit `http://localhost:7000/docs` for interactive API documentation powered by Swagger UI.

### Generating OpenAPI Specification

To generate the OpenAPI specification file (`openapi.json`), run:

```bash
python generate_openapi.py
```

This will create an `openapi.json` file in the project root containing the complete API specification. You can use this file for:

- API client generation
- Documentation generation
- Importing into API testing tools
- Version control of your API contract

The OpenAPI spec is automatically generated from your FastAPI application's route definitions and type hints, ensuring it stays in sync with your code.

## Configuration

The service automatically uses GPU if available. To force CPU usage or configure other settings, modify the model loading parameters in `app.py`.

## License

This project is part of the Little Scripts monorepo. See the main [LICENSE](../LICENSE) file for details.
