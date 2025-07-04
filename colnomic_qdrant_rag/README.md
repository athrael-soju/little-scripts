# Beautiful RAG with ColNomic + Qdrant + MinIO

🚀 A powerful multimodal document retrieval system built with **ColPali** (Column-based Patch Interaction) and binary quantization for efficient document search and analysis.

For a more detailed read, have a look at the [Athrael.net Blog Post](https://athrael.net/blog/little-scripts/colnomic-qdrant-rag).

## 📖 Overview

This application provides an intelligent document retrieval system that can:
- **Index and search** PDF documents and images using natural language queries
- **Leverage binary quantization** for efficient vector storage and faster retrieval
- **Provide AI-powered conversational responses** using OpenAI integration
- **Handle multimodal content** with both text queries and image understanding
- **Scale efficiently** with optimized vector storage and retrieval

### Key Features

- 🔍 **Natural Language Search**: Query documents using plain English
- 🤖 **AI-Powered Analysis**: Get conversational responses about document content
- 📄 **PDF Support**: Automatically process and index PDF documents
- 🖼️ **Image Understanding**: Advanced visual document analysis using ColPali
- ⚡ **Binary Quantization**: Efficient storage with minimal quality loss
- 🎯 **Interactive CLI**: User-friendly command-line interface
- 🐳 **Docker Ready**: Easy deployment with Docker Compose

## 🏗️ Architecture

<div align="center">
  <img src="architecture.svg" alt="Architecture diagram" width="75%" />
</div>

### Components

- **ColPali Model**: Advanced multimodal embeddings using `nomic-ai/colnomic-embed-multimodal-3b`
- **Qdrant Vector Database**: High-performance vector search with binary quantization
- **MinIO Object Storage**: Scalable image and document storage
- **OpenAI Integration**: Enhanced conversational analysis capabilities

## 🎯 Why Colnomic?

### State-of-the-Art Performance

With an impressive **61.2 NDCG@5** on the Vidore-v2 benchmark, Colnomic Embed Multimodal 3B represents a significant leap forward in multimodal document retrieval. Here's why it's the perfect choice for this RAG implementation:

#### Performance Benchmarks

| Model                            | Avg.     | ESG Restaurant Human | Econ Macro Multi. | AXA Multi. | MIT Bio  | ESG Restaurant Synth. |
| -------------------------------- | -------- | -------------------- | ----------------- | ---------- | -------- | --------------------- |
| **ColNomic Embed Multimodal 7B** | **62.7** | **73.9**             | **54.7**          | **61.3**   | **66.1** | **57.3**              |
| **ColNomic Embed Multimodal 3B** | **61.2** | **65.8**             | **55.4**          | **61.0**   | **63.5** | **56.6**              |
| Nomic Embed Multimodal 7B        | 59.7     | 65.7                 | 57.7              | 59.3       | 64.0     | 49.2                  |
| Nomic Embed Multimodal 3B        | 58.8     | 59.8                 | 57.5              | 58.8       | 62.5     | 49.4                  |
| Voyage Multimodal 3              | 55.0     | 56.1                 | 55.0              | 59.5       | 56.4     | 47.2                  |

*ColNomic 3B achieves remarkable performance at 61.2 NDCG@5, positioning it as the second-best model overall while being significantly more efficient than its 7B counterpart.*

### Unified Text-Image Processing

Unlike traditional RAG systems that require complex OCR preprocessing and separate text extraction pipelines, Colnomic **directly encodes interleaved text and images** without any preprocessing. This means:

- **No more lossy OCR conversion steps** - preserves original document fidelity
- **Preserved visual context and layout information** - understands document structure
- **Faster processing** by eliminating preprocessing bottlenecks
- **More complete information capture** from documents

### Perfect for Visual Documents

Colnomic excels at handling the types of documents that challenge traditional text-only systems:

- **Research papers** with equations, diagrams, and complex tables
- **Technical documentation** with code blocks, flowcharts, and screenshots
- **Financial reports** with charts, graphs, and numerical data
- **Product catalogs** with images, specifications, and visual elements

### Open and Accessible

As an **open-weights** model with only 3B parameters, Colnomic strikes the perfect balance between performance and accessibility. It's powerful enough for production use while being lightweight enough to run on consumer hardware.

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (recommended for optimal performance)
- **Docker & Docker Compose** (for infrastructure services)
- **Poppler** (for PDF processing):
  - **Windows**: Download from [Poppler Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
  - **macOS**: `brew install poppler`
  - **Linux**: `sudo apt-get install poppler-utils`

### 1. Clone and Setup

```bash
git clone https://github.com/your-username/colpali-binary-quant.git
cd colpali-binary-quant

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Start Infrastructure Services

```bash
# Start Qdrant (CPU version) and MinIO services
docker-compose --profile cpu up -d

# OR: Start Qdrant (GPU version) and MinIO services (requires NVIDIA GPU)
docker-compose --profile gpu up -d

# Verify services are running
docker-compose ps
```

> **Note**: The GPU version requires:
> - NVIDIA GPU with CUDA support
> - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
> - Docker configured to use GPU resources

### 3. Configure Environment

Create a `.env` file in the project root:

```env
# Optional: OpenAI API key for conversational features
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini

# MinIO Configuration (defaults work with Docker Compose)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

### 4. Run the Application

```bash
# Interactive mode (recommended for beginners)
python main.py interactive

# Or use direct commands
python main.py upload  # Index default UFO dataset
python main.py ask "What are some interesting UFO sightings?"
```

## 💻 Usage

### Interactive Mode

The interactive mode provides the most user-friendly experience:

```bash
python main.py interactive
```

Once in interactive mode, you can:

```
🔍 colpali[Basic]> What are UFO sightings in California?
🤖 colpali[Conversational]> set-mode conversational
🤖 colpali[Conversational]> Analyze the visual patterns in these documents
🔍 colpali[Basic]> upload --file my_document.pdf
🔍 colpali[Basic]> show-status
```

### Command Line Interface

#### Search Commands

```bash
# Basic search (fast, document retrieval only)
python main.py ask "UFO sightings in Texas"

# AI-powered analysis (includes conversational response)
python main.py analyze "What do these UFO images reveal about encounter patterns?"
```

#### Document Management

```bash
# Upload and index documents
python main.py upload --file path/to/document.pdf
python main.py upload --file path/to/documents.txt
python main.py upload  # Use default UFO dataset

# Clear all documents
python main.py clear-collection

# Check system status
python main.py show-status
```

### Search Modes

1. **Basic Mode** 🔍
   - Fast document retrieval
   - Returns relevant documents with similarity scores
   - No AI analysis

2. **Conversational Mode** 🤖
   - AI-powered responses using OpenAI
   - Contextual analysis of retrieved documents
   - Streaming responses with citations

## ⚙️ Configuration

### Core Settings (`config.py`)

```python
# Model Configuration
MODEL_NAME = "nomic-ai/colnomic-embed-multimodal-3b"
VECTOR_SIZE = 128

# Search Configuration
SEARCH_LIMIT = 3          # Number of results to return
OVERSAMPLING = 2.0        # Improve recall with oversampling

# Processing Configuration
BATCH_SIZE = 4            # Batch size for indexing
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for conversational features | None |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4.1-mini` |
| `QDRANT_URL` | Qdrant database URL | `http://localhost:6333` |
| `MINIO_ENDPOINT` | MinIO server endpoint | `localhost:9000` |

### Binary Quantization

The system uses **binary quantization** in Qdrant for:
- **90%+ storage reduction** compared to full-precision vectors
- **Faster similarity search** operations
- **Minimal impact on search quality** due to ColPali's robust embeddings

## 📁 Project Structure

```
colpali-binary-quant/
├── main.py                 # Application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Infrastructure services
├── utils.py              # Utility functions
├── core/                 # Core application logic
│   ├── cli.py           # Command-line interface
│   ├── commands.py      # Command implementations
│   └── pipeline.py      # Document processing pipeline
└── handlers/            # External service handlers
    ├── model.py        # ColPali model operations
    ├── qdrant.py       # Vector database operations
    ├── minio.py        # Object storage operations
    └── openai.py       # OpenAI integration
```

## 🔧 Advanced Usage

### Custom Model Configuration

To use a different ColPali model:

```python
# In config.py
MODEL_NAME = "your-custom-colpali-model"
PROCESSOR_NAME = "your-custom-colpali-model"
```

### GPU Optimization

For newer NVIDIA GPUs (RTX 5090, etc.):

```bash
# Install PyTorch nightly with CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Batch Processing

For large document collections:

```python
# Increase batch size for better GPU utilization
BATCH_SIZE = 8  # Adjust based on GPU memory
```

## 🐳 Docker Deployment

### Development Setup

```bash
docker-compose up -d
```

This starts:
- **Qdrant** on port 6333
- **MinIO** on ports 9000 (API) and 9001 (Console)

### Production Considerations

For production deployment:

1. **Update MinIO credentials** in `docker-compose.yml`
2. **Configure persistent volumes** for data retention
3. **Set up network security** for service communication
4. **Configure resource limits** based on expected load

### Service Monitoring

Access service dashboards:
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   BATCH_SIZE = 2
   ```

2. **Poppler Not Found**
   ```bash
   # Install poppler for PDF processing
   # See prerequisites section for platform-specific instructions
   ```

3. **Qdrant Connection Error**
   ```bash
   # Ensure Docker services are running
   docker-compose ps
   docker-compose up -d
   ```

4. **Model Download Issues**
   ```bash
   # Set HuggingFace token if needed
   export HF_TOKEN=your_huggingface_token
   ```

### Performance Optimization

- **GPU Memory**: Adjust `BATCH_SIZE` based on available VRAM
- **Search Speed**: Tune `OVERSAMPLING` for quality vs speed trade-off
- **Storage**: Binary quantization reduces storage by ~90%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install ruff  # For code formatting

# Run code formatting
ruff check --fix .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ColPali Team** for the innovative multimodal retrieval approach
- **Qdrant** for high-performance vector search capabilities
- **Nomic AI** for the excellent embedding models
- **OpenAI** for conversational AI capabilities

## 📚 Resources

- [ColPali Paper](https://arxiv.org/abs/2407.01449)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [MinIO Documentation](https://min.io/docs/)
- [Binary Quantization Guide](https://qdrant.tech/documentation/guides/quantization/)

---

**Made with ❤️ for efficient document retrieval and analysis**
