from io import BytesIO
from typing import List, Union

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available

# Initialize FastAPI app
app = FastAPI(
    title="ColQwen2.5 Embedding API",
    description="API for generating embeddings from images and queries",
)

# Load model and processor
model = ColQwen2_5.from_pretrained(
    "vidore/colqwen2.5-v0.2",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")


class QueryRequest(BaseModel):
    queries: Union[str, List[str]]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[List[float]]]  # 3D: [batch, sequence_length, hidden_dim]


class PatchResponse(BaseModel):
    n_patches: int


class PatchRequest(BaseModel):
    width: int
    height: int


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load PIL Image from bytes"""
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def generate_query_embeddings(queries: List[str]) -> List[torch.Tensor]:
    """Generate embeddings for text queries"""
    device = model.device
    embeddings = []

    with torch.no_grad():
        batch_query = processor.process_queries(queries).to(device)
        query_embeddings = model(**batch_query)
        embeddings = list(torch.unbind(query_embeddings.to("cpu")))

    return embeddings


def generate_image_embeddings(images: List[Image.Image]) -> List[torch.Tensor]:
    """Generate embeddings for images"""
    device = model.device
    embeddings = []

    with torch.no_grad():
        batch_images = processor.process_images(images).to(device)
        image_embeddings = model(**batch_images)
        embeddings = list(torch.unbind(image_embeddings.to("cpu")))

    return embeddings


# API Endpoints


@app.get("/")
async def root():
    return {"message": "ColQwen2.5 Embedding API", "version": "0.0.2"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "device": str(model.device)}


@app.get("/info")
async def version():
    """Version endpoint"""
    return {
        "version": "0.0.2",
        "device": str(model.device),
        "dtype": str(model.dtype),
        "flash_attn": is_flash_attn_2_available(),
        "spatial_merge_size": model.spatial_merge_size,
        "parameters": model.parameters(),
        "dim": model.dim,
        "image_token_id": processor.image_token_id,
    }


@app.post("/patches", response_model=PatchResponse)
async def get_n_patches(request: PatchRequest):
    """Calculate number of patches for given image size and spatial merge size
    
    Args:
        request: PatchRequest containing:
            - width: int - width of the image in pixels
            - height: int - height of the image in pixels
    """
    try:
        # Pass image size as (width, height) tuple as expected by the processor
        image_size = (request.width, request.height)
        return processor.get_n_patches(
            image_size, spatial_merge_size=model.spatial_merge_size
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting number of patches: {str(e)}"
        )


@app.post("/embed/queries", response_model=EmbeddingResponse)
async def embed_queries(request: QueryRequest):
    """Generate embeddings for text queries"""
    try:
        # Handle single query or list of queries
        queries = (
            [request.queries] if isinstance(request.queries, str) else request.queries
        )

        if not queries:
            raise HTTPException(status_code=400, detail="No queries provided")

        # Generate embeddings
        embeddings_tensors = generate_query_embeddings(queries)

        # Convert tensors to lists
        embeddings_list = [embedding.tolist() for embedding in embeddings_tensors]

        return EmbeddingResponse(embeddings=embeddings_list)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating query embeddings: {str(e)}"
        )


@app.post("/embed/images", response_model=EmbeddingResponse)
async def embed_images(files: List[UploadFile] = File(...)):
    """Generate embeddings for uploaded images"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No images provided")

        images = []
        for file in files:
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, detail=f"File {file.filename} is not an image"
                )

            # Read and convert to PIL Image
            image_bytes = await file.read()
            image = load_image_from_bytes(image_bytes)
            images.append(image)

        # Generate embeddings
        embeddings_tensors = generate_image_embeddings(images)

        # Convert tensors to lists
        embeddings_list = [embedding.tolist() for embedding in embeddings_tensors]

        return EmbeddingResponse(embeddings=embeddings_list)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating image embeddings: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, workers=1)
