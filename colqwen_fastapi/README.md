# ColQwen2.5 FastAPI Service

A FastAPI-based service for generating embeddings from images and text queries using the ColQwen2.5 model.

## Features

- 🖼️ **Image Embeddings**: Generate embeddings for images
- 🔤 **Text Embeddings**: Generate embeddings for text queries
- 🚀 **High Performance**: Utilizes Flash Attention 2 when available
- 📊 **RESTful API**: Easy integration with other services
- 🏥 **Health Check**: Built-in health check endpoint
- 📏 **Patch Calculation**: Utility endpoint for patch calculations

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support

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

## Usage

### Starting the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check
- `GET /version`: Get API version
- `POST /patches`: Calculate number of patches for given image dimensions
- `POST /embed/queries`: Generate embeddings for text queries
- `POST /embed/images`: Generate embeddings for uploaded images

### Example Requests

#### Get Embeddings for Text Queries

```bash
curl -X POST "http://localhost:8000/embed/queries" \
     -H "Content-Type: application/json" \
     -d '{"queries": ["example query"]}'
```

#### Get Embeddings for Images

```bash
curl -X POST "http://localhost:8000/embed/images" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```

## API Documentation

### Interactive Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

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
