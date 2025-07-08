# Little Scripts

<div align="center">
  <img src="little-scripts.svg" alt="Little Scripts Logo" width="600">
</div>

## About

A monorepo containing various utility scripts, tools, and applications for development, automation, and AI-powered tasks.

## 📁 Projects

<details>
<summary><strong>🤖 ColPali(ColNomic) + Qdrant + MinIO Retrieval System</strong></summary>

A powerful multimodal document retrieval system that combines ColPali embeddings with vector search for intelligent document analysis.

**What it does:**
- 🔍 Search PDF documents and images using natural language queries
- 💬 Get AI-powered conversational responses about document content
- 📊 Handle complex visual documents with charts, diagrams, and mixed content
- ⚡ Scale efficiently with optimized vector storage and retrieval

**Key highlights:** 13x faster search performance, binary quantization, Docker deployment, interactive CLI

[📖 View Full Documentation](./colnomic_qdrant_rag/README.md)

</details>

<details>
<summary><strong>🖼️ EOMT Panoptic Segmentation App</strong></summary>

An interactive web application for panoptic segmentation using the EOMT (Encoder-only Mask Transformer) model - a minimalist Vision Transformer approach for image segmentation.

**What it does:**
- 🖥️ Interactive web interface for image segmentation
- 🎨 Multiple visualization types (masks, overlays, contours, analytics)
- ⚡ Real-time processing with detailed segment statistics
- 🧪 Built-in test images for experimentation

**Key highlights:** Up to 4× faster than complex methods, Gradio interface, comprehensive analytics

[📖 View Full Documentation](./eomt_panoptic_seg/README.md)

</details>

<details>
<summary><strong>🔧 Future Projects</strong></summary>

More utility scripts and tools will be added to this monorepo over time. Each project will have its own directory with dedicated documentation.

</details>

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

3. **Follow the project-specific README** for detailed setup instructions.

## 📖 Project Structure

```
little-scripts/
├── colnomic_qdrant_rag/           # Multimodal document retrieval system
├── eomt_panoptic_seg/             # Image segmentation web app
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
- **colpali**
- **computer-vision**
- **document-retrieval**
- **gradio**
- **machine-learning**
- **multimodal-search**
- **panoptic-segmentation**
- **python**
- **qdrant**
- **rag-system**
- **reranking**
- **transformers**
- **utilities**
- **vector-database**

---

<div align="center">
  <p>⭐ If you find this repository useful, please consider giving it a star!</p>
</div>
