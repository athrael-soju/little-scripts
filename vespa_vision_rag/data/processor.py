import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image

from config import Config


class DataProcessor:
    """Handler class for data processing operations"""
    
    def __init__(self):
        """Initialize the data processor"""
        pass
    
    def binarize_embedding(self, embedding: torch.Tensor) -> str:
        """
        Convert embedding tensor to binary representation
        
        Args:
            embedding: Tensor embedding to binarize
            
        Returns:
            Hex string representation of binary embedding
        """
        binary_vector = (
            np.packbits(np.where(embedding > 0, 1, 0))
            .astype(np.int8)
            .tobytes()
            .hex()
        )
        return binary_vector
    
    def create_embedding_dict(self, embeddings: List[torch.Tensor]) -> Dict[int, str]:
        """
        Create embedding dictionary with binary representations
        
        Args:
            embeddings: List of embedding tensors
            
        Returns:
            Dictionary mapping indices to binary embeddings
        """
        embedding_dict = {}
        for idx, embedding in enumerate(embeddings):
            binary_embedding = self.binarize_embedding(embedding)
            embedding_dict[idx] = binary_embedding
        
        return embedding_dict
    
    def prepare_vespa_documents(self, processed_pdfs: List[Dict[str, Any]], 
                               pdf_processor, 
                               embeddings_list: List[List[torch.Tensor]]) -> List[Dict[str, Any]]:
        """
        Prepare documents for Vespa indexing
        
        Args:
            processed_pdfs: List of processed PDF data
            pdf_processor: PDF processor instance for image operations
            embeddings_list: List of embeddings for each PDF
            
        Returns:
            List of documents formatted for Vespa
        """
        vespa_documents = []
        
        for pdf, embeddings in zip(processed_pdfs, embeddings_list):
            url = pdf["url"]
            title = pdf["title"]
            
            for page_number, (page_text, embedding, image) in enumerate(
                zip(pdf["texts"], embeddings, pdf["images"])
            ):
                # Resize and convert image to base64
                resized_image = pdf_processor.resize_image(image, Config.RESIZED_IMAGE_HEIGHT)
                base64_image = pdf_processor.image_to_base64(resized_image)
                
                # Create binary embedding dictionary
                embedding_dict = {}
                for idx, patch_embedding in enumerate(embedding):
                    binary_vector = self.binarize_embedding(patch_embedding)
                    embedding_dict[idx] = binary_vector
                
                # Create document
                document = {
                    "id": hash(url + str(page_number)),
                    "url": url,
                    "title": title,
                    "page_number": page_number,
                    "image": base64_image,
                    "text": page_text,
                    "embedding": embedding_dict,
                }
                
                vespa_documents.append(document)
        
        return vespa_documents
    
    def prepare_query_embeddings(self, query_embeddings: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Prepare query embeddings for Vespa queries
        
        Args:
            query_embeddings: List of query embedding tensors
            
        Returns:
            List of query embedding dictionaries
        """
        prepared_embeddings = []
        
        for embedding in query_embeddings:
            # Convert to float dictionary
            float_embedding = {k: v.tolist() for k, v in enumerate(embedding)}
            prepared_embeddings.append(float_embedding)
        
        return prepared_embeddings
    
    def prepare_binary_query_embeddings(self, float_embeddings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert float query embeddings to binary representations
        
        Args:
            float_embeddings: List of float embedding dictionaries
            
        Returns:
            List of binary embedding dictionaries
        """
        binary_embeddings = []
        
        for float_embedding in float_embeddings:
            binary_embedding = {}
            for k, v in float_embedding.items():
                binary_vector = (
                    np.packbits(np.where(np.array(v) > 0, 1, 0))
                    .astype(np.int8)
                    .tolist()
                )
                binary_embedding[k] = binary_vector
            
            binary_embeddings.append(binary_embedding)
        
        return binary_embeddings
    
    def get_embedding_stats(self, embeddings: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Get statistics about embeddings
        
        Args:
            embeddings: List of embedding tensors
            
        Returns:
            Dictionary with embedding statistics
        """
        if not embeddings:
            return {"count": 0}
        
        total_patches = sum(len(emb) for emb in embeddings)
        avg_patches = total_patches / len(embeddings)
        
        # Get dimensionality from first embedding
        first_patch = embeddings[0][0] if len(embeddings[0]) > 0 else None
        dimensions = first_patch.shape[0] if first_patch is not None else 0
        
        return {
            "count": len(embeddings),
            "total_patches": total_patches,
            "avg_patches_per_embedding": avg_patches,
            "dimensions": dimensions,
            "dtype": str(embeddings[0].dtype) if embeddings else "unknown"
        }
    
    def validate_embeddings(self, embeddings: List[torch.Tensor]) -> bool:
        """
        Validate that embeddings are properly formatted
        
        Args:
            embeddings: List of embedding tensors to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not embeddings:
            return False
        
        try:
            # Check that all embeddings have the same structure
            first_dims = embeddings[0][0].shape[0] if len(embeddings[0]) > 0 else 0
            
            for embedding in embeddings:
                if len(embedding) == 0:
                    continue
                
                for patch in embedding:
                    if patch.shape[0] != first_dims:
                        return False
            
            return True
            
        except Exception:
            return False 