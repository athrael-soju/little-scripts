import os

from dotenv import load_dotenv

load_dotenv()

"""Configuration for the ColPali retrieval demo."""

# Qdrant Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "le-collection-3"

# Model Configuration
MODEL_NAME = "nomic-ai/colnomic-embed-multimodal-3b"
PROCESSOR_NAME = "nomic-ai/colnomic-embed-multimodal-3b"

# Vector Database Configuration
VECTOR_SIZE = 128
DISTANCE_METRIC = "Cosine"  # Using string representation for simplicity

# Indexing Configuration
BATCH_SIZE = 4
OPTIMIZE_COLLECTION = False

# Background processing settings
MINIO_UPLOAD_WORKERS = 4  # Number of concurrent MinIO upload workers
MINIO_UPLOAD_RETRY_ATTEMPTS = 3  # Number of retry attempts for failed uploads
MINIO_UPLOAD_TIMEOUT = 30  # Timeout for background worker shutdown in seconds

# Search Configuration
SEARCH_LIMIT = 3
OVERSAMPLING = 2.0

# Pooling and Reranking Optimization
ENABLE_RERANKING_OPTIMIZATION = True  # This creates multi-vector collections with mean-pooled embeddings for faster search
RERANKING_PREFETCH_LIMIT = 200  # Number of candidates to prefetch with pooled vectors
RERANKING_SEARCH_LIMIT = 20  # Final number of results after reranking

# Dataset Configuration
DATASET_NAME = "davanstrien/ufo-ColPali"
DATASET_SPLIT = "train"

# Image Save Configuration
MAX_SAVE_IMAGES = 3
IMAGE_FORMAT = (
    "JPEG"  # Options: "PNG", "JPEG" - JPEG is faster and smaller, PNG is lossless
)
IMAGE_QUALITY = 85  # JPEG quality (1-100), ignored for PNG

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MAX_TOKENS = 500
OPENAI_TEMPERATURE = 0.7

# MinIO Configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "le-images-3"
MINIO_USE_SSL = False
