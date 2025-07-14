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
- 🔍 **Conversational Search**: Just ask questions in natural language - no commands needed
- 💬 **AI-Powered Responses**: Get intelligent, contextual answers about your documents
- 📄 **PDF & Image Support**: Process complex visual documents with charts, diagrams, and mixed content
- ⚡ **Optimized Performance**: 13x faster search with binary quantization and reranking optimization
- 🤖 **Streamlined Interface**: Simple conversational CLI that starts ready to use

**Key technical features:**
- Binary quantization for 90%+ storage reduction
- Mean pooling reranking optimization (enabled by default)
- Background image processing pipeline
- Docker deployment with Qdrant + MinIO
- Graceful handling of optional services (OpenAI, MinIO)

**Usage:** Simply run `python main.py` and start asking questions about your documents!

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

### Development Setup

Before contributing, please set up pre-commit hooks to ensure code quality:

1. **Install pre-commit:**
   ```bash
   pip install pre-commit
   ```

2. **Install the hooks:**
   ```bash
   pre-commit install
   ```

3. **Run hooks on all files (optional):**
   ```bash
   pre-commit run --all-files
   ```

The pre-commit hooks will automatically run on each commit to check for:
- Code formatting and style
- Import sorting
- Trailing whitespace and other common issues
- Project-specific linting rules

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

## 🛠️ Troubleshooting

### Pre-commit Hook Issues

#### Problem: "pre-commit not found" when committing from Cursor/VS Code

**Symptoms:**
- Commits work fine from terminal but fail from IDE with error: `pre-commit not found. Did you forget to activate your virtualenv?`
- Pre-commit hooks are installed and work in terminal

**Cause:**
The pre-commit hook is trying to use a Python executable path that's not accessible from the IDE's environment (e.g., WSL paths when running on Windows).

**Solution:**
Modify the `.git/hooks/pre-commit` file to use the system Python instead of a specific virtualenv path:

1. **Open the pre-commit hook file:**
   ```bash
   # Edit .git/hooks/pre-commit
   ```

2. **Find the conditional block (around line 12-21) and replace:**
   ```bash
   # Change this:
   if [ -x "$INSTALL_PYTHON" ]; then
       exec "$INSTALL_PYTHON" -mpre_commit "${ARGS[@]}"
   
   # To this:
   if command -v python > /dev/null; then
       exec python -mpre_commit "${ARGS[@]}"
   ```

3. **Test the fix:**
   ```bash
   # This should now work from both terminal and IDE
   git commit -m "Test commit"
   ```

**Alternative Solutions:**
- **Option A:** Commit from terminal where virtualenv is activated
- **Option B:** Skip hooks temporarily: `git commit --no-verify -m "message"`
- **Option C:** Use IDE's integrated terminal with virtualenv activated

#### Problem: Ruff formatting conflicts

**Symptoms:**
- Pre-commit hooks fail with ruff formatting errors
- Code appears correctly formatted but hooks still fail

**Solution:**
1. **Run ruff manually to see specific issues:**
   ```bash
   ruff check .
   ruff format .
   ```

2. **Configure ruff settings in `pyproject.toml`** (if needed):
   ```toml
   [tool.ruff]
   line-length = 88
   target-version = "py310"
   ```

3. **Run pre-commit on all files to fix batch issues:**
   ```bash
   pre-commit run --all-files
   ```

### Docker and Infrastructure Issues

#### Problem: Docker services not starting

**Symptoms:**
- Services fail to start with port conflicts
- Docker containers exit immediately

**Common Solutions:**
1. **Check if ports are already in use:**
   ```bash
   # Windows
   netstat -ano | findstr :6333
   
   # Linux/macOS
   lsof -i :6333
   ```

2. **Stop conflicting services:**
   ```bash
   docker-compose down
   docker system prune -f
   ```

3. **Restart Docker daemon** and try again

#### Problem: Permission issues with Docker volumes

**Solution:**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./data
```

### Python Environment Issues

#### Problem: Module not found errors

**Symptoms:**
- ImportError or ModuleNotFoundError when running scripts
- Works in one environment but not another

**Solutions:**
1. **Verify you're in the correct environment:**
   ```bash
   which python
   pip list | grep [package-name]
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Check Python path conflicts:**
   ```bash
   python -c "import sys; print(sys.path)"
   ```

### Getting Help

If you encounter issues not covered here:

1. **Check project-specific READMEs** for additional troubleshooting
2. **Search existing issues** in the repository
3. **Create a new issue** with:
   - Clear problem description
   - Steps to reproduce
   - Environment details (OS, Python version, etc.)
   - Error messages and logs

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
