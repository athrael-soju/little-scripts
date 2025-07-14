import torch
from torch.utils.data import DataLoader
from typing import List, Union, Any
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from tqdm import tqdm

from config import Config


class ColQwen2_5Handler:
    """Handler class for ColQwen2_5 model operations"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the ColQwen2_5 handler
        
        Args:
            model_name: Name of the model to load. Defaults to Config.MODEL_NAME
        """
        self.model_name = model_name or Config.MODEL_NAME
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the ColQwen2_5 model and processor"""
        print(f"Loading model: {self.model_name}")
        
        self.model = ColQwen2_5.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        
        self.processor = ColQwen2_5_Processor.from_pretrained(self.model_name)
        self.model = self.model.eval()
        
        print("Model loaded successfully")
    
    def generate_image_embeddings(self, images: List[Image.Image], batch_size: int = None) -> List[torch.Tensor]:
        """
        Generate embeddings for a list of images
        
        Args:
            images: List of PIL images
            batch_size: Batch size for processing. Defaults to Config.BATCH_SIZE
            
        Returns:
            List of embedding tensors
        """
        batch_size = batch_size or Config.BATCH_SIZE
        embeddings = []
        
        dataloader = DataLoader(
            images,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )
        
        print(f"Processing {len(images)} images in batches of {batch_size}")
        
        for batch_doc in tqdm(dataloader, desc="Generating embeddings"):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
                embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        
        return embeddings
    
    def generate_query_embeddings(self, queries: List[str], batch_size: int = 1) -> List[torch.Tensor]:
        """
        Generate embeddings for a list of text queries
        
        Args:
            queries: List of text queries
            batch_size: Batch size for processing
            
        Returns:
            List of query embedding tensors
        """
        embeddings = []
        
        dataloader = DataLoader(
            queries,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_queries(x),
        )
        
        print(f"Processing {len(queries)} queries")
        
        for batch_query in tqdm(dataloader, desc="Generating query embeddings"):
            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
                embeddings.extend(list(torch.unbind(embeddings_query.to("cpu"))))
        
        return embeddings
    
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.model is not None and self.processor is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self.is_model_loaded():
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown"
        } 