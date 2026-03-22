import os

# Ensure CUDA 13 runtime is on LD_LIBRARY_PATH for causal-conv1d
try:
    import nvidia.cu13
    cu13_lib = os.path.join(nvidia.cu13.__path__[0], "lib")
    os.environ["LD_LIBRARY_PATH"] = cu13_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    import ctypes
    ctypes.CDLL(os.path.join(cu13_lib, "libcudart.so.13"))
except ImportError:
    pass

import fitz  # pymupdf
import gradio as gr
import torch
from PIL import Image

from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

MAX_PAGES = 100
BATCH_SIZE = 4
MODEL_ID = "athrael-soju/colqwen3.5-4.5B-v3"

print("Loading model...")
model = ColQwen3_5.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    token=os.environ.get("HF_TOKEN"),
).eval()

processor = ColQwen3_5Processor.from_pretrained(
    MODEL_ID,
    token=os.environ.get("HF_TOKEN"),
)
print("Model loaded.")

# In-memory index: cached image embeddings + PIL images
index = {"embeddings": None, "images": []}


def pdf_to_images(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


def load_documents(files):
    imgs = []
    for f in files:
        path = f if isinstance(f, str) else f.name
        if path.lower().endswith(".pdf"):
            imgs.extend(pdf_to_images(path))
        else:
            imgs.append(Image.open(path).convert("RGB"))
    return imgs


def index_documents(files):
    if not files:
        index["embeddings"] = None
        index["images"] = []
        yield "No documents uploaded. Index cleared."
        return

    yield "Loading documents..."
    imgs = load_documents(files)

    if len(imgs) > MAX_PAGES:
        imgs = imgs[:MAX_PAGES]

    all_embeddings = []
    total = len(imgs)

    for i in range(0, total, BATCH_SIZE):
        batch = imgs[i : i + BATCH_SIZE]
        yield f"Encoding pages {i + 1}-{min(i + BATCH_SIZE, total)} of {total}..."
        batch_images = processor.process_images(batch).to(model.device)
        with torch.no_grad():
            emb = model(**batch_images)
            model.rope_deltas = None
        all_embeddings.append(emb.cpu())

    yield "Finalizing index..."

    # Pad embeddings to same seq_len (different images produce different token counts)
    max_seq_len = max(e.shape[1] for e in all_embeddings)
    padded = []
    for e in all_embeddings:
        if e.shape[1] < max_seq_len:
            pad = torch.zeros(e.shape[0], max_seq_len - e.shape[1], e.shape[2], dtype=e.dtype)
            e = torch.cat([e, pad], dim=1)
        padded.append(e)

    index["embeddings"] = torch.cat(padded, dim=0)
    index["images"] = imgs

    yield f"Indexed {total} page(s). Ready to search."


def search(query, top_k):
    if index["embeddings"] is None:
        gr.Warning("Please upload and index documents first.")
        return []
    if not query or not query.strip():
        gr.Warning("Please enter a query.")
        return []

    batch_queries = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
        model.rope_deltas = None

    scores = processor.score(query_embeddings, index["embeddings"].to(model.device))[0]

    ranked = sorted(
        zip(index["images"], scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(img, f"Score: {score:.4f}") for img, score in ranked[:top_k]]


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="linear-gradient(90deg, *primary_400, *primary_600)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_300, *primary_500)",
    button_primary_text_color="white",
    button_primary_shadow="*shadow_drop",
    block_shadow="*shadow_drop",
    block_title_text_weight="600",
)

css = """
.main-header {
    text-align: center;
    margin-bottom: 0.5rem;
}
.main-header h1 {
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.05rem;
    margin-bottom: 1rem;
}
"""

with gr.Blocks(title="ColQwen3.5-4.5B-v3") as demo:
    gr.Markdown(
        "# ColQwen3.5-4.5B-v3 Visual Document Retrieval",
        elem_classes=["main-header"],
    )
    gr.Markdown(
        "Upload document pages (images or PDFs, max 100 pages), index them, then search with multiple queries. "
        "Powered by [ColQwen3.5-4.5B-v3](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3) "
        "and [colpali-engine](https://github.com/illuin-tech/colpali).",
        elem_classes=["subtitle"],
    )

    with gr.Sidebar(label="Controls", open=True, width=380):
        files_input = gr.File(
            label="Document Pages (images or PDFs)",
            file_types=["image", ".pdf"],
            file_count="multiple",
            height=250,
        )
        index_btn = gr.Button("Index Documents", variant="secondary", size="lg")
        index_status = gr.Textbox(label="Index Status", interactive=False)

        gr.Markdown("---")

        query_input = gr.Textbox(
            label="Query",
            placeholder="e.g., What is the total revenue for Q4 2024?",
            lines=2,
        )
        top_k_input = gr.Dropdown(
            label="Top K Results",
            choices=[1, 2, 3, 4, 5],
            value=3,
        )
        search_btn = gr.Button("Search", variant="primary", size="lg")

    results_gallery = gr.Gallery(
        label="Results (ranked by relevance)",
        columns=3,
        height="auto",
        object_fit="contain",
        allow_preview=True,
        buttons=["download", "fullscreen"],
    )

    index_btn.click(
        fn=index_documents,
        inputs=[files_input],
        outputs=index_status,
    )
    search_btn.click(
        fn=search,
        inputs=[query_input, top_k_input],
        outputs=results_gallery,
    )
    query_input.submit(
        fn=search,
        inputs=[query_input, top_k_input],
        outputs=results_gallery,
    )

demo.launch(theme=theme, css=css)
