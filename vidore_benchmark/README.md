# ViDoRe Benchmark Runner

A tiny script to run the ViDoRe (v1 and v2) benchmarks from MTEB using the pre-defined ViDoRe model wrapper.

## What it does

- Uses `mteb.get_model("vidore/colqwen2.5-v0.2")` to load the official ViDoRe model wrapper
- Runs the ViDoRe benchmark suites via `mteb.MTEB`
- Prints the aggregated results to stdout

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended
- PyTorch (nightly recommended for latest CUDA builds)

## Setup

```bash
# From repo root
cd vidore_benchmark

# Base deps
pip install -r requirements.txt

# Optional: latest PyTorch nightly (CUDA 12.9 index)
pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129
```

## Usage

```bash
python app.py
```

Notes:
- The script uses a conservative `batch_size=1` to minimize VRAM usage.
- Results are printed as a Python object/dict. Redirect to a file if desired.

## Customization

Edit `vidore_benchmark/app.py` to tweak:
- `model_name`: change the backbone, e.g. another ViDoRe-compatible model
- `benchmarks`: choose specific ViDoRe suites
- `encode_kwargs["batch_size"]`: raise for faster runs if you have more VRAM

## Troubleshooting

- Out-of-memory (OOM): lower `batch_size` further or close other GPU workloads
- Slow downloads: models and datasets are cached on first run; subsequent runs are faster
- CPU fallback: possible but much slower; ensure a proper CUDA PyTorch install for best performance
- Still facing issues? Check the [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark) and [mteb](https://github.com/embeddings-benchmark/mteb) repos for more details.

## Interested in helping?
- Quite curious to see how it performs on your CPU/Cuda GPU, or even especially MPS