# mteb_run_vidore.py
import os, torch, mteb

# --- (optional but helpful) allocator + precision tweaks ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]  # backward compat
torch.set_float32_matmul_precision("medium")  # slightly lighter kernels

# --- load the pre-defined model from MTEB ---
model_name = "vidore/colqwen2.5-v0.2"
model = mteb.get_model(model_name)  # uses the correct wrapper internally

# --- select the ViDoRe benchmarks ---
benchmarks = mteb.get_benchmarks(names=["ViDoRe(v1)", "ViDoRe(v2)"])
evaluator = mteb.MTEB(tasks=benchmarks)

# --- run with small batches to stay under ~32 GB VRAM ---
results = evaluator.run(
    model,
    encode_kwargs={"batch_size": 4},     # Lower value for less VRAM usage. 4 is the max I've managed to get to work with an RTX 5090.
    verbosity=2,
)
print(results)