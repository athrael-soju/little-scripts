import os

from dotenv import load_dotenv

load_dotenv()

"""Configuration for the ColPali retrieval demo."""

# Qdrant Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "le-collection"

# Model Configuration
MODEL_NAME = "nomic-ai/colnomic-embed-multimodal-3b"
PROCESSOR_NAME = "nomic-ai/colnomic-embed-multimodal-3b"

# Vector Database Configuration
VECTOR_SIZE = 128
DISTANCE_METRIC = "Cosine"  # Using string representation for simplicity

# Indexing Configuration
BATCH_SIZE = 4
OPTIMIZE_COLLECTION = False

# Search Configuration
SEARCH_LIMIT = 3
OVERSAMPLING = 2.0

# Dataset Configuration
DATASET_NAME = "davanstrien/ufo-ColPali"
DATASET_SPLIT = "train"

# Image Save Configuration
MAX_SAVE_IMAGES = 3

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MAX_TOKENS = 500
OPENAI_TEMPERATURE = 0.7

# MinIO Configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "le-images"
MINIO_USE_SSL = False
