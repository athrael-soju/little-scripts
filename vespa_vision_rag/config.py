import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for Vespa Vision RAG application"""
    
    # Model configuration
    MODEL_NAME = "nomic-ai/colnomic-embed-multimodal-3b"
    
    # Vespa configuration
    VESPA_APP_NAME = "visionrag"
    VESPA_TENANT_NAME = os.getenv("VESPA_TENANT_NAME", "madkimchi")
    VESPA_API_KEY = os.getenv("VESPA_TEAM_API_KEY", None)
    
    # Docker/Local Vespa configuration
    VESPA_ENDPOINT = os.getenv("VESPA_ENDPOINT", None)  # e.g., http://vespa:8080
    USE_LOCAL_VESPA = os.getenv("USE_LOCAL_VESPA", "false").lower() == "true"
    
    # Processing configuration
    MAX_QUERY_TERMS = 64
    TARGET_HITS_PER_QUERY_TENSOR = 20
    BATCH_SIZE = 4
    MAX_IMAGE_HEIGHT = 800
    RESIZED_IMAGE_HEIGHT = 640
    
    # Ranking configuration
    SECOND_PHASE_RERANK_COUNT = 100
    ADVANCED_RERANK_COUNT = 10
    
    # Query configuration
    DEFAULT_HITS = 3
    QUERY_TIMEOUT = 120
    INDEXING_TIMEOUT = 180
    
    # HNSW configuration
    HNSW_MAX_LINKS = 32
    HNSW_NEIGHBORS_TO_EXPLORE = 400
    
    # Sample PDFs configuration
    SAMPLE_PDFS: List[Dict[str, str]] = [
        {
            "title": "ConocoPhillips Sustainability Highlights - Nature (24-0976)",
            "url": "https://static.conocophillips.com/files/resources/24-0976-sustainability-highlights_nature.pdf",
        },
        {
            "title": "ConocoPhillips Managing Climate Related Risks",
            "url": "https://static.conocophillips.com/files/resources/conocophillips-2023-managing-climate-related-risks.pdf",
        },
        {
            "title": "ConocoPhillips 2023 Sustainability Report",
            "url": "https://static.conocophillips.com/files/resources/conocophillips-2023-sustainability-report.pdf",
        },
    ]
    
    # Sample queries
    SAMPLE_QUERIES: List[str] = [
        "Percentage of non-fresh water as source?",
        "Policies related to nature risk?",
        "How much of produced water is recycled?",
    ]
    
    @classmethod
    def get_processed_api_key(cls) -> str:
        """Get processed API key with proper newline handling"""
        if cls.VESPA_API_KEY is not None:
            return cls.VESPA_API_KEY.replace(r"\n", "\n")
        return None
    
    @classmethod
    def is_docker_mode(cls) -> bool:
        """Check if running in Docker mode with local Vespa"""
        return cls.USE_LOCAL_VESPA or cls.VESPA_ENDPOINT is not None
    
    @classmethod
    def get_vespa_endpoint(cls) -> str:
        """Get Vespa endpoint URL for local deployment"""
        return cls.VESPA_ENDPOINT or "http://localhost:8080"
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration settings"""
        if not cls.is_docker_mode():
            if cls.VESPA_TENANT_NAME is None:
                raise ValueError("VESPA_TENANT_NAME must be set for cloud deployment")
        
        if cls.MODEL_NAME is None:
            raise ValueError("MODEL_NAME must be set")
        
        print(f"Configuration mode: {'Docker/Local' if cls.is_docker_mode() else 'Cloud'}")
        if cls.is_docker_mode():
            print(f"Vespa endpoint: {cls.get_vespa_endpoint()}")
        else:
            print(f"Vespa tenant: {cls.VESPA_TENANT_NAME}") 