# ColQwen3.5-4.5B-v3 Visual Document Retrieval

A Gradio-based visual document retrieval app powered by [ColQwen3.5-4.5B](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3) and [colpali-engine](https://github.com/illuin-tech/colpali). Upload document pages (images or PDFs), index them once, then run multiple text queries to find the most relevant pages using ColBERT-style late interaction scoring.

- **Model**: [athrael-soju/colqwen3.5-4.5B](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3)
- **Space**: [athrael-soju/colqwen3.5-4.5B](https://huggingface.co/spaces/athrael-soju/colqwen3.5-4.5B)

## Features

- **PDF & Image Support**: Upload PDFs (rendered at 150 DPI per page) and images
- **Index-then-Search**: Encode documents once, query repeatedly without re-encoding
- **Batched Encoding**: Pages encoded in batches of 4 with streaming progress
- **Top-K Selection**: Choose 1-5 top results per query
- **MaxSim Scoring**: ColBERT-style multi-vector late interaction retrieval
- **100 Page Limit**: Supports up to 100 pages across all uploads
- **Modern UI**: Gradio 6 with Soft theme, sidebar controls, gallery results

## Architecture

ColQwen3.5 is built on Qwen3.5-4B, a natively multimodal model using early fusion with a hybrid architecture of Gated Delta Networks (linear attention) + Gated Attention + Sparse MoE. The model produces 128-dimensional multi-vector embeddings for both document images and text queries, scored via MaxSim.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (8.6GB+ VRAM for model, ~11.7GB total)
- `HF_TOKEN` environment variable (if model is gated)

## Quick Start

```bash
cd colqwen3.5-v3-space
uv venv .venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

For the fast path (Gated DeltaNet acceleration), also install:

```bash
pip install flash-linear-attention
# causal-conv1d requires a prebuilt wheel matching your CUDA/PyTorch versions:
pip install causal-conv1d
```

### Run

```bash
HF_TOKEN=your_token python app.py
```

The app launches at `http://localhost:7860`.

## Usage

1. **Upload**: Add document page images or PDFs in the sidebar
2. **Index**: Click "Index Documents" - progress streams in the status box
3. **Search**: Enter a text query, select Top K, click "Search"
4. **Browse**: Results appear ranked by relevance score in the gallery

## Notes

- `model.rope_deltas` is cleared between forward passes to prevent tensor shape mismatches (Qwen3.5 multimodal position encoding artifact)
- Embeddings are cached on CPU to support both local GPU and HF Spaces ZeroGPU environments
- Different image resolutions produce different token counts; embeddings are zero-padded before concatenation (zero-padded positions don't affect MaxSim scoring)
