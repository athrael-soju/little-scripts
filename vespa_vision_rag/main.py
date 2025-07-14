#!/usr/bin/env python3
"""
Vespa Vision RAG Application

A refactored version of the original PDF retrieval system using ColQWen2 (ColPali) 
with Vespa for efficient document retrieval from visual features.
"""

import asyncio
import sys
import os
from typing import List, Dict, Any, Optional

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.colqwen_handler import ColQwen2_5Handler
from pdf.processor import PDFProcessor
from search.client import VespaClient
from data.processor import DataProcessor
from utils.display import DisplayUtils


class VespaVisionRAG:
    """Main application class for Vespa Vision RAG system"""
    
    def __init__(self):
        """Initialize the application components"""
        print("Initializing Vespa Vision RAG system...")
        
        # Validate configuration
        Config.validate_config()
        
        # Initialize components
        self.model_handler = None
        self.pdf_processor = PDFProcessor()
        self.vespa_client = VespaClient()
        self.data_processor = DataProcessor()
        self.display_utils = DisplayUtils()
        
        # Data storage
        self.processed_pdfs = []
        self.vespa_documents = []
        self.query_embeddings = []
        
        print("Application initialized successfully!")
    
    def load_model(self) -> None:
        """Load the ColQwen2 model"""
        print("Loading ColQwen2 model...")
        self.model_handler = ColQwen2_5Handler()
        
        # Display model info
        model_info = self.model_handler.get_model_info()
        self.display_utils.display_stats(model_info, "Model Information")
    
    def process_pdfs(self, pdf_configs: List[Dict[str, str]] = None) -> None:
        """
        Process PDF documents
        
        Args:
            pdf_configs: List of PDF configurations. Defaults to Config.SAMPLE_PDFS
        """
        pdf_configs = pdf_configs or Config.SAMPLE_PDFS
        
        print(f"Processing {len(pdf_configs)} PDF documents...")
        self.processed_pdfs = self.pdf_processor.process_multiple_pdfs(pdf_configs)
        
        # Display processing results
        for pdf in self.processed_pdfs:
            self.display_utils.display_pdf_info(pdf)
        
        print(f"Successfully processed {len(self.processed_pdfs)} PDFs")
    
    def generate_embeddings(self) -> None:
        """Generate embeddings for all processed PDFs"""
        if not self.model_handler:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.processed_pdfs:
            raise RuntimeError("No PDFs processed. Call process_pdfs() first.")
        
        print("Generating embeddings for PDF pages...")
        
        # Generate embeddings for each PDF
        for pdf in self.processed_pdfs:
            print(f"Generating embeddings for: {pdf['title']}")
            embeddings = self.model_handler.generate_image_embeddings(pdf['images'])
            pdf['embeddings'] = embeddings
            
            # Display embedding statistics
            stats = self.data_processor.get_embedding_stats(embeddings)
            self.display_utils.display_stats(stats, f"Embeddings for {pdf['title']}")
        
        print("Embedding generation completed!")
    
    def prepare_vespa_documents(self) -> None:
        """Prepare documents for Vespa indexing"""
        if not self.processed_pdfs:
            raise RuntimeError("No PDFs processed. Call process_pdfs() first.")
        
        # Extract embeddings from processed PDFs
        embeddings_list = [pdf['embeddings'] for pdf in self.processed_pdfs]
        
        print("Preparing documents for Vespa indexing...")
        self.vespa_documents = self.data_processor.prepare_vespa_documents(
            self.processed_pdfs, self.pdf_processor, embeddings_list
        )
        
        print(f"Prepared {len(self.vespa_documents)} documents for indexing")
    
    def deploy_vespa_application(self) -> None:
        """Deploy the application to Vespa Cloud"""
        print("Deploying Vespa application...")
        self.vespa_client.deploy()
        
        # Display deployment info
        app_info = self.vespa_client.get_app_info()
        self.display_utils.display_stats(app_info, "Vespa Deployment Info")
    
    async def index_documents(self) -> None:
        """Index documents in Vespa"""
        if not self.vespa_documents:
            raise RuntimeError("No documents prepared. Call prepare_vespa_documents() first.")
        
        await self.vespa_client.index_documents(self.vespa_documents)
        print("Document indexing completed!")
    
    def prepare_query_embeddings(self, queries: List[str] = None) -> None:
        """
        Prepare query embeddings
        
        Args:
            queries: List of query strings. Defaults to Config.SAMPLE_QUERIES
        """
        if not self.model_handler:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        queries = queries or Config.SAMPLE_QUERIES
        
        print(f"Generating embeddings for {len(queries)} queries...")
        raw_embeddings = self.model_handler.generate_query_embeddings(queries)
        
        # Prepare embeddings for Vespa
        self.query_embeddings = self.data_processor.prepare_query_embeddings(raw_embeddings)
        
        print("Query embeddings prepared!")
    
    async def query_with_bm25(self, query: str = None, query_idx: int = 0) -> None:
        """
        Query using BM25 for retrieval and ColPali for reranking
        
        Args:
            query: Query string. If None, uses query from Config.SAMPLE_QUERIES
            query_idx: Index of query embedding to use
        """
        if not self.query_embeddings:
            raise RuntimeError("No query embeddings prepared. Call prepare_query_embeddings() first.")
        
        query = query or Config.SAMPLE_QUERIES[query_idx]
        query_embedding = self.query_embeddings[query_idx]
        
        print(f"Querying with BM25 ranking: '{query}'")
        
        response = await self.vespa_client.query_with_bm25(query, query_embedding)
        
        if response.is_successful():
            self.display_utils.display_query_results(query, response)
            
            # Create and display summary
            summary = self.display_utils.create_results_summary(query, response)
            self.display_utils.display_stats(summary, "Query Summary")
        else:
            print(f"Query failed: {response.json()}")
    
    async def query_with_nearest_neighbor(self, query: str = None, query_idx: int = 0) -> None:
        """
        Query using nearestNeighbor for retrieval with binary representations
        
        Args:
            query: Query string. If None, uses query from Config.SAMPLE_QUERIES
            query_idx: Index of query embedding to use
        """
        if not self.query_embeddings:
            raise RuntimeError("No query embeddings prepared. Call prepare_query_embeddings() first.")
        
        query = query or Config.SAMPLE_QUERIES[query_idx]
        float_embedding = self.query_embeddings[query_idx]
        
        # Convert to binary representation
        binary_embeddings = self.data_processor.prepare_binary_query_embeddings([float_embedding])
        binary_embedding = binary_embeddings[0]
        
        print(f"Querying with nearestNeighbor: '{query}'")
        
        response = await self.vespa_client.query_with_nearest_neighbor(
            float_embedding, binary_embedding
        )
        
        if response.is_successful():
            self.display_utils.display_query_results(query, response)
            
            # Create and display summary
            summary = self.display_utils.create_results_summary(query, response)
            self.display_utils.display_stats(summary, "Query Summary")
        else:
            print(f"Query failed: {response.json()}")
    
    async def run_complete_pipeline(self, pdf_configs: List[Dict[str, str]] = None, 
                                   queries: List[str] = None) -> None:
        """
        Run the complete pipeline from PDF processing to querying
        
        Args:
            pdf_configs: List of PDF configurations
            queries: List of queries to test
        """
        print("=== Starting Complete Pipeline ===")
        
        try:
            # Step 1: Load model
            self.load_model()
            
            # Step 2: Process PDFs
            self.process_pdfs(pdf_configs)
            
            # Step 3: Generate embeddings
            self.generate_embeddings()
            
            # Step 4: Prepare documents for Vespa
            self.prepare_vespa_documents()
            
            # Step 5: Deploy Vespa application
            self.deploy_vespa_application()
            
            # Step 6: Index documents
            await self.index_documents()
            
            # Step 7: Prepare query embeddings
            self.prepare_query_embeddings(queries)
            
            # Step 8: Run sample queries
            print("\n=== Running Sample Queries ===")
            
            queries_to_test = queries or Config.SAMPLE_QUERIES
            for i, query in enumerate(queries_to_test):
                print(f"\n--- Query {i+1}: {query} ---")
                
                # Test BM25 ranking
                print("Testing BM25 ranking...")
                await self.query_with_bm25(query, i)
                
                # Test nearestNeighbor ranking
                print("Testing nearestNeighbor ranking...")
                await self.query_with_nearest_neighbor(query, i)
            
            print("\n=== Pipeline Completed Successfully! ===")
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            raise
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        print("Cleaning up resources...")
        if self.pdf_processor:
            self.pdf_processor.cleanup()
        print("Cleanup completed!")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()


async def main():
    """Main entry point"""
    print("Vespa Vision RAG Application")
    print("=" * 50)
    
    # Create application instance
    app = VespaVisionRAG()
    
    try:
        # Run the complete pipeline
        await app.run_complete_pipeline()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main application
    asyncio.run(main()) 