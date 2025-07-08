# Little Scripts

<div align="center">
  <img src="little-scripts.svg" alt="Little Scripts Logo" width="600">
</div>

## About

A monorepo containing various utility scripts, tools, and applications for development, automation, and AI-powered tasks.

## 📁 Projects

### 🤖 [ColPali(ColNomic) + Qdrant + MinIO Retrieval System](./colnomic_qdrant_rag/)

A powerful multimodal document retrieval system built with **ColNomic** (Using Late Interaction) and binary quantization for efficient document search and analysis.

**Key Features:**
- 🔍 **Natural Language Search**: Query documents using plain English
- 🤖 **AI-Powered Analysis**: Get conversational responses about document content
- 📄 **PDF Support**: Automatically process and index PDF documents
- 🖼️ **Image Understanding**: Advanced visual document analysis using ColPali
- ⚡ **Binary Quantization**: Efficient storage with minimal quality loss
- 🎯 **Interactive CLI**: User-friendly command-line interface
- 🐳 **Docker Ready**: Easy deployment with Docker Compose

**Tech Stack:** Python, ColPali, Qdrant, MinIO, OpenAI, Docker

[📖 View Full Documentation](./colnomic_qdrant_rag/README.md)

---

### 🖼️ [EOMT Panoptic Segmentation App](./eomt_panoptic_seg/)

An interactive web application for panoptic segmentation using the **EOMT (Encoder-only Mask Transformer)** model - a minimalist approach that repurposes a plain Vision Transformer for image segmentation, as presented in the CVPR 2025 paper ["Your ViT is Secretly an Image Segmentation Model"](https://www.tue-mps.org/eomt/).

**Key Features:**
- 🎨 **Interactive Web Interface**: User-friendly Gradio interface for image upload and processing
- 🔍 **Multiple Visualization Types**: Original+Mask, Overlay, Contours, Segment Info, and All Views
- 🧠 **EOMT Model**: Plain ViT-based segmentation with the `tue-mps/coco_panoptic_eomt_large_640` model
- 📊 **Detailed Analytics**: Segment statistics and color-coded visualization
- 🖼️ **Sample Images**: Built-in test images for immediate experimentation
- ⚡ **Real-time Processing**: Up to 4× faster than complex methods while maintaining accuracy

**Tech Stack:** Python, PyTorch, Transformers, Gradio, OpenCV, Matplotlib

[📖 View Full Documentation](./eomt_panoptic_seg/README.md)

---

### 🔧 Future Projects

More utility scripts and tools will be added to this monorepo over time. Each project will have its own directory with dedicated documentation.

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Poetry** (for dependency management)
- **Docker & Docker Compose** (for projects requiring infrastructure)

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/athrael.soju/little-scripts.git
   cd little-scripts
   ```

2. **Setup using the automated script (recommended):**
   ```bash
   # Install everything
   python setup.py

   # Or install specific projects
   python setup.py --project colnomic    # ColPali Qdrant RAG only
   python setup.py --project panoptic    # EOMT Panoptic Segmentation only
   ```

   **Manual installation alternative:**
   ```bash
   # Install Poetry dependencies
   poetry install --with dev,colnomic,panoptic

   # Install PyTorch nightly
   poetry run pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

3. **Activate the Poetry environment:**
   ```bash
   poetry shell
   ```

4. **Run a specific project:**
   ```bash
   poetry run python colnomic_qdrant_rag/main.py
   poetry run python eomt_panoptic_seg/app.py
   ```



## 📖 Project Structure

```
little-scripts/
├── pyproject.toml                 # Poetry configuration with dependency groups
├── setup.py                      # Automated setup script for easy installation
├── colnomic_qdrant_rag/          # Col Based RAG System
├── eomt_panoptic_seg/            # EOMT Panoptic Segmentation App
└── [future-projects]/            # Additional projects will be added here
```

## 🔧 Dependency Management

This monorepo uses **Poetry** for unified dependency management with optional dependency groups for each project. All packages are configured to use the latest compatible versions (using "*" constraints) to maintain flexibility and match the original requirements.txt approach.

### Project Structure

- **Base dependencies**: Common packages used across multiple projects (requests, pillow, numpy, etc.)
- **dev group**: Development tools (pre-commit, ruff, testing tools)
- **colnomic group**: Dependencies specific to the ColPali Qdrant RAG project
- **panoptic group**: Dependencies specific to the EOMT Panoptic Segmentation project

### Manual Installation Alternative

If you prefer manual control over the installation process:

```bash
# 1. Install Poetry dependencies
poetry install --with dev,colnomic,panoptic  # Everything
poetry install --with dev,colnomic           # ColPali Qdrant RAG project
poetry install --with dev,panoptic           # EOMT Panoptic Segmentation project

# 2. Install PyTorch nightly (required for all projects)
poetry run pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Adding New Dependencies

#### Add to base dependencies (shared across projects)
```bash
poetry add requests
```

#### Add to development dependencies
```bash
poetry add --group dev pytest
```

#### Add to project-specific groups
```bash
poetry add --group colnomic new-package
poetry add --group panoptic another-package
```

### Managing Dependencies

```bash
# Update dependencies
poetry update

# Show installed packages
poetry show

# Show dependency tree
poetry show --tree

# Lock current versions
poetry lock

# Export requirements (if needed for Docker/CI)
poetry export -f requirements.txt --output requirements.txt
poetry export -f requirements.txt --with dev,colnomic --output requirements-colnomic.txt
poetry export -f requirements.txt --with dev,panoptic --output requirements-panoptic.txt
```

### Special Notes

#### PyTorch Nightly Installation
PyTorch nightly builds are required for all projects (commented out of Poetry dependencies to avoid conflicts):
```bash
poetry run pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

This ensures compatibility with newer NVIDIA GPUs and provides the latest PyTorch features across all projects.

#### System Dependencies
Some packages require system-level dependencies:

**PDF2Image (for ColPali project):**
- **Windows**: Download from https://github.com/oschwartz10612/poppler-windows/releases/
- **Mac**: `brew install poppler`
- **Linux**: `sudo apt-get install poppler-utils`

### Benefits Over requirements.txt

1. **Dependency Resolution**: Poetry automatically resolves compatible versions
2. **Lock Files**: `poetry.lock` ensures reproducible builds
3. **Virtual Environment Management**: Built-in venv handling
4. **Dependency Groups**: Optional groups for different project needs
5. **Unified Management**: Single configuration for the entire monorepo

This approach eliminates duplicate dependency management while allowing each project to have its specific requirements.

## 🤝 Contributing

We welcome contributions to any of the projects in this monorepo!

### Adding a New Project

1. Create a new directory for your project
2. Include a comprehensive README.md with:
   - Project description and features
   - Installation instructions
   - Usage examples
   - Configuration details
3. Add your project to the main README's project list
4. Follow the existing code style and documentation patterns

### Contributing to Existing Projects

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📝 License

Open source - feel free to use and modify as needed.

## 🏷️ Repository Topics

- **ai-tools**
- **automation**
- **computer-vision**
- **document-retrieval**
- **gradio**
- **machine-learning**
- **panoptic-segmentation**
- **python**
- **rag-system**
- **transformers**
- **utilities**
- **vector-database**

---

<div align="center">
  <p>⭐ If you find this repository useful, please consider giving it a star!</p>
</div>
