# Vespa Vision RAG

A refactored PDF retrieval system using ColQWen2 (ColPali) with Vespa for efficient document retrieval from visual features.

## Overview

This application demonstrates how to build a modern document retrieval system that:
- Processes PDF documents by converting them to images and extracting text
- Uses ColQWen2 (ColPali) model for generating visual embeddings
- Stores embeddings in Vespa with binary compression for efficient storage
- Provides both BM25 text-based and nearest neighbor visual similarity search
- Offers a clean, modular architecture with proper separation of concerns

## Features

- **Visual Document Understanding**: Uses ColQWen2 to understand document layout and content from images
- **Efficient Storage**: Binary quantization reduces embedding storage by 32x
- **Dual Retrieval**: Supports both text-based (BM25) and visual similarity search
- **Scalable Architecture**: Built on Vespa for handling large document collections
- **Modular Design**: Clean separation of concerns across different components
- **Dual Deployment**: Supports both Vespa Cloud and local Docker deployment

## Architecture

```
vespa_vision_rag/
├── config.py              # Configuration management
├── models/
│   └── colqwen_handler.py  # ColQwen2 model operations
├── pdf/
│   └── processor.py        # PDF processing and conversion
├── vespa/
│   └── client.py          # Vespa operations and schema
├── data/
│   └── processor.py       # Data processing and quantization
├── utils/
│   └── display.py         # Display utilities
├── main.py                # Main application orchestrator
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile             # Application container
└── requirements.txt       # Python dependencies
```

## Prerequisites

### System Requirements
- Python 3.8+ (for local development)
- Docker and Docker Compose (for containerized deployment)
- poppler-utils (for PDF processing)

### Install poppler-utils (Local Development)

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install poppler-utils -y
```

**macOS (with Homebrew):**
```bash
brew install poppler
```

**Windows:**
Download from [poppler for Windows](https://poppler.freedesktop.org/) and add to PATH.

## Deployment Options

### Option 1: Docker Deployment (Recommended)

This is the easiest way to get started. The Docker setup includes both Vespa and the application.

#### Quick Start with Docker

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd vespa_vision_rag
   ```

2. **Start Vespa container:**
   ```bash
   docker-compose up -d vespa
   ```

3. **Wait for Vespa to be ready (about 60 seconds):**
   ```bash
   docker-compose logs -f vespa
   ```

4. **Run the application:**
   ```bash
   # Set environment for local Vespa
   export USE_LOCAL_VESPA=true
   export VESPA_ENDPOINT=http://localhost:8080
   
   # Run with Docker
   docker-compose --profile app up app
   ```

#### Docker Environment Variables

Create a `.env` file in the project root:
```env
# For local Vespa deployment
USE_LOCAL_VESPA=true
VESPA_ENDPOINT=http://vespa:8080

# For Vespa Cloud deployment (alternative)
# VESPA_TEAM_API_KEY=your-api-key-here
# VESPA_TENANT_NAME=your-tenant-name
```

#### Docker Commands

```bash
# Start only Vespa
docker-compose up -d vespa

# Start Vespa and application
docker-compose --profile app up

# View logs
docker-compose logs -f vespa
docker-compose logs -f app

# Stop all services
docker-compose down

# Remove volumes (clean slate)
docker-compose down -v
```

### Option 2: Vespa Cloud Deployment

For production or when you want to use Vespa Cloud:

1. **Setup Vespa Cloud:**
   - Create a tenant at [console.vespa-cloud.com](https://console.vespa-cloud.com/)
   - Get your API key

2. **Set environment variables:**
   ```bash
   export VESPA_TEAM_API_KEY="your-api-key-here"
   export VESPA_TENANT_NAME="your-tenant-name"
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

### Option 3: Local Development

For development and testing:

1. **Install system dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install poppler-utils -y
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start local Vespa with Docker:**
   ```bash
   docker-compose up -d vespa
   ```

4. **Configure for local Vespa:**
   ```bash
   export USE_LOCAL_VESPA=true
   export VESPA_ENDPOINT=http://localhost:8080
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

### Quick Start

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Load the ColQwen2 model
2. Process sample PDF documents
3. Generate embeddings
4. Deploy to Vespa (Cloud or local)
5. Index documents
6. Run sample queries

### Programmatic Usage

```python
from vespa_vision_rag.main import VespaVisionRAG

# Initialize application
app = VespaVisionRAG()

# Load model
app.load_model()

# Process custom PDFs
pdf_configs = [
    {
        "title": "My Document",
        "url": "https://example.com/document.pdf"
    }
]
app.process_pdfs(pdf_configs)

# Generate embeddings
app.generate_embeddings()

# Deploy and index
app.deploy_vespa_application()
await app.index_documents()

# Query
app.prepare_query_embeddings(["What is the main topic?"])
await app.query_with_bm25("What is the main topic?")
```

### Individual Components

Each component can be used independently:

```python
# PDF Processing
from vespa_vision_rag.pdf.processor import PDFProcessor
processor = PDFProcessor()
images, texts = processor.extract_pdf_content("document.pdf")

# Model Operations
from vespa_vision_rag.models.colqwen_handler import ColQwen2_5Handler
model = ColQwen2_5Handler()
embeddings = model.generate_image_embeddings(images)

# Vespa Operations
from vespa_vision_rag.vespa.client import VespaClient
client = VespaClient()
client.deploy()
```

## Configuration

The `config.py` file contains all configuration settings:

```python
class Config:
    # Model configuration
    MODEL_NAME = "vidore/colqwen2-v0.1"
    
    # Vespa Cloud configuration
    VESPA_APP_NAME = "visionrag"
    VESPA_TENANT_NAME = "your-tenant"
    
    # Docker/Local Vespa configuration
    USE_LOCAL_VESPA = False  # Set to True for local deployment
    VESPA_ENDPOINT = "http://localhost:8080"
    
    # Processing parameters
    BATCH_SIZE = 2
    MAX_IMAGE_HEIGHT = 800
    
    # ... other settings
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_LOCAL_VESPA` | Use local Vespa instead of cloud | `false` |
| `VESPA_ENDPOINT` | Local Vespa endpoint | `http://localhost:8080` |
| `VESPA_TEAM_API_KEY` | Vespa Cloud API key | None |
| `VESPA_TENANT_NAME` | Vespa Cloud tenant name | `vespa-team` |

## Docker Configuration

### Services

The `docker-compose.yml` defines two services:

1. **vespa**: The Vespa search engine
   - Ports: 8080 (search/feed), 19071 (config)
   - Memory: 4GB limit, 2GB reservation
   - Health checks enabled

2. **app**: The Python application (optional)
   - Depends on Vespa being healthy
   - Mounts current directory as `/app`
   - Uses `--profile app` to start

### Volumes

- `vespa_data`: Persistent storage for Vespa
- `app_data`: Application data storage

## Key Components

### 1. ColQwen2_5Handler (`models/colqwen_handler.py`)
- Loads and manages the ColQWen2 model
- Generates embeddings for images and text queries
- Handles batch processing for efficiency

### 2. PDFProcessor (`pdf/processor.py`)
- Downloads PDFs from URLs
- Converts PDF pages to images
- Extracts text content
- Handles image resizing and base64 encoding

### 3. VespaClient (`vespa/client.py`)
- Defines Vespa schema and rank profiles
- Manages deployment to Vespa Cloud or local Docker
- Handles document indexing and querying
- Supports both BM25 and nearest neighbor search

### 4. DataProcessor (`data/processor.py`)
- Performs binary quantization of embeddings
- Prepares data for Vespa indexing
- Handles embedding format conversions

### 5. DisplayUtils (`utils/display.py`)
- Provides result visualization
- Supports both Jupyter notebook and console output
- Creates formatted summaries

## Binary Quantization

The application uses binary quantization to reduce embedding storage:
- Converts 128-dimensional float vectors to 128 bits (16 bytes)
- Reduces storage by 32x compared to float representations
- Uses Hamming distance for fast approximate similarity search
- Maintains good retrieval accuracy on benchmark datasets

## Ranking Strategies

### BM25 + MaxSim (default)
- First phase: BM25 ranking on extracted text
- Second phase: MaxSim with full float embeddings
- Good for text-heavy documents

### Binary Retrieval + MaxSim (retrieval-and-rerank)
- First phase: Binary representations with inverted Hamming distance
- Second phase: MaxSim with full float embeddings
- Better for visual similarity search

## Performance Tuning

Key parameters to adjust:
- `BATCH_SIZE`: Increase for better GPU utilization
- `TARGET_HITS_PER_QUERY_TENSOR`: Balance between speed and accuracy
- `SECOND_PHASE_RERANK_COUNT`: Number of documents to rerank
- `HNSW_MAX_LINKS`: HNSW index connectivity

## Troubleshooting

### Common Issues

1. **Model loading fails**: Ensure sufficient GPU memory or use CPU
2. **PDF processing errors**: Check poppler-utils installation
3. **Vespa deployment fails**: Verify API key and tenant configuration
4. **Docker issues**: Ensure Docker daemon is running and ports are available
5. **Out of memory**: Reduce batch size or image resolution

### Docker-specific Issues

1. **Port conflicts**: Change ports in docker-compose.yml if 8080 is taken
2. **Memory issues**: Increase Docker's memory limit
3. **Volume permissions**: Ensure Docker can write to mounted volumes

### Debug Mode
Set environment variable for detailed logging:
```bash
export VESPA_DEBUG=1
python main.py
```

## Health Checks

The application includes health checks:
- Vespa: `http://localhost:8080/ApplicationStatus`
- Application: Python import test

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both Docker and local deployment
5. Add tests if applicable
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [ColQwen2 Model](https://huggingface.co/vidore/colqwen2-v0.1)
- [Vespa Documentation](https://docs.vespa.ai/)
- [Vespa Docker Hub](https://hub.docker.com/r/vespaengine/vespa/)
- [Vespa Quick Start](https://docs.vespa.ai/en/vespa-quick-start.html)
- [ColPali Paper](https://arxiv.org/abs/2407.01449)
- [Scaling ColPali Blog Post](https://blog.vespa.ai/scaling-colpali-to-billions/)

## Acknowledgments

- Built on the original ColPali notebook example
- Uses the ColQwen2 model from the Vidore team
- Leverages Vespa's vector search capabilities
- Docker configuration based on official Vespa documentation 