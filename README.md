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
- **Docker & Docker Compose** (for projects requiring infrastructure)

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/athrael.soju/little-scripts.git
   cd little-scripts
   ```

2. **Navigate to a specific project:**
   ```bash
   cd colnomic_qdrant_rag
   ```

3. **Follow the project-specific README** for setup instructions.

## 📖 Project Structure

```
little-scripts/
├── colnomic_qdrant_rag/           # Col Based RAG System
├── eomt_panoptic_seg/             # EOMT Panoptic Segmentation App
└── [future-projects]/             # Additional projects will be added here
```

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
