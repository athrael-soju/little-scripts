import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np

from vespa.package import (
    ApplicationPackage, Schema, Document, Field, FieldSet, HNSW,
    RankProfile, Function, FirstPhaseRanking, SecondPhaseRanking
)
from vespa.deployment import VespaCloud, VespaDocker
from vespa.application import Vespa
from vespa.io import VespaResponse, VespaQueryResponse

from config import Config


class VespaClient:
    """Handler class for Vespa operations"""
    
    def __init__(self, tenant_name: str = None, app_name: str = None):
        """
        Initialize the Vespa client
        
        Args:
            tenant_name: Vespa tenant name. Defaults to Config.VESPA_TENANT_NAME
            app_name: Vespa application name. Defaults to Config.VESPA_APP_NAME
        """
        self.tenant_name = tenant_name or Config.VESPA_TENANT_NAME
        self.app_name = app_name or Config.VESPA_APP_NAME
        self.vespa_cloud = None
        self.vespa_docker = None
        self.app = None
        self.application_package = None
        self.is_docker_mode = Config.is_docker_mode()
        
        # Disable tokenizers parallelism for deployment
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self._create_application_package()
    
    def _create_schema(self) -> Schema:
        """Create the Vespa schema for PDF pages"""
        schema = Schema(
            name="pdf_page",
            document=Document(
                fields=[
                    Field(
                        name="id", 
                        type="string", 
                        indexing=["summary", "index"], 
                        match=["word"]
                    ),
                    Field(
                        name="url", 
                        type="string", 
                        indexing=["summary", "index"]
                    ),
                    Field(
                        name="title",
                        type="string",
                        indexing=["summary", "index"],
                        match=["text"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="page_number", 
                        type="int", 
                        indexing=["summary", "attribute"]
                    ),
                    Field(
                        name="image", 
                        type="raw", 
                        indexing=["summary"]
                    ),
                    Field(
                        name="text",
                        type="string",
                        indexing=["index"],
                        match=["text"],
                        index="enable-bm25",
                    ),
                    Field(
                        name="embedding",
                        type="tensor<int8>(patch{}, v[16])",
                        indexing=["attribute", "index"],
                        ann=HNSW(
                            distance_metric="hamming",
                            max_links_per_node=Config.HNSW_MAX_LINKS,
                            neighbors_to_explore_at_insert=Config.HNSW_NEIGHBORS_TO_EXPLORE,
                        ),
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["title", "text"])],
        )
        
        return schema
    
    def _create_rank_profiles(self, schema: Schema) -> None:
        """Create rank profiles for the schema"""
        
        # Default rank profile for BM25 + MaxSim
        default_profile = RankProfile(
            name="default",
            inputs=[("query(qt)", "tensor<float>(querytoken{}, v[128])")],
            functions=[
                Function(
                    name="max_sim",
                    expression="""
                        sum(
                            reduce(
                                sum(
                                    query(qt) * unpack_bits(attribute(embedding)) , v
                                ),
                                max, patch
                            ),
                            querytoken
                        )
                    """,
                ),
                Function(
                    name="bm25_score", 
                    expression="bm25(title) + bm25(text)"
                ),
            ],
            first_phase=FirstPhaseRanking(expression="bm25_score"),
            second_phase=SecondPhaseRanking(
                expression="max_sim", 
                rerank_count=Config.SECOND_PHASE_RERANK_COUNT
            ),
        )
        
        # Advanced retrieval profile with binary representations
        input_query_tensors = []
        for i in range(Config.MAX_QUERY_TERMS):
            input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))
        
        input_query_tensors.append(("query(qt)", "tensor<float>(querytoken{}, v[128])"))
        input_query_tensors.append(("query(qtb)", "tensor<int8>(querytoken{}, v[16])"))
        
        retrieval_profile = RankProfile(
            name="retrieval-and-rerank",
            inputs=input_query_tensors,
            functions=[
                Function(
                    name="max_sim",
                    expression="""
                        sum(
                            reduce(
                                sum(
                                    query(qt) * unpack_bits(attribute(embedding)) , v
                                ),
                                max, patch
                            ),
                            querytoken
                        )
                    """,
                ),
                Function(
                    name="max_sim_binary",
                    expression="""
                        sum(
                          reduce(
                            1/(1 + sum(
                                hamming(query(qtb), attribute(embedding)) ,v)
                            ),
                            max,
                            patch
                          ),
                          querytoken
                        )
                    """,
                ),
            ],
            first_phase=FirstPhaseRanking(expression="max_sim_binary"),
            second_phase=SecondPhaseRanking(
                expression="max_sim", 
                rerank_count=Config.ADVANCED_RERANK_COUNT
            ),
        )
        
        schema.add_rank_profile(default_profile)
        schema.add_rank_profile(retrieval_profile)
    
    def _create_application_package(self) -> None:
        """Create the Vespa application package"""
        schema = self._create_schema()
        self._create_rank_profiles(schema)
        
        self.application_package = ApplicationPackage(
            name=self.app_name, 
            schema=[schema]
        )
    
    def deploy(self) -> Vespa:
        """Deploy the application to Vespa (Cloud or Docker)"""
        if self.is_docker_mode:
            return self._deploy_docker()
        else:
            return self._deploy_cloud()
    
    def _deploy_cloud(self) -> Vespa:
        """Deploy to Vespa Cloud"""
        api_key = Config.get_processed_api_key()
        
        self.vespa_cloud = VespaCloud(
            tenant=self.tenant_name,
            application=self.app_name,
            key_content=api_key,
            application_package=self.application_package,
        )
        
        print(f"Deploying application '{self.app_name}' to Vespa Cloud...")
        self.app = self.vespa_cloud.deploy()
        print("Cloud deployment successful!")
        
        return self.app
    
    def _deploy_docker(self) -> Vespa:
        """Deploy to local Vespa Docker container"""
        vespa_endpoint = Config.get_vespa_endpoint()
        
        print(f"Deploying application '{self.app_name}' to local Vespa at {vespa_endpoint}...")
        
        # Check if we should connect to existing container or create new one
        try:
            # Try to connect to existing container (from docker-compose)
            self.vespa_docker = VespaDocker.from_container_name_or_id("vespa-vision-rag")
            print("Connected to existing Vespa container")
        except:
            # If no existing container, create a new one
            print("Creating new Vespa container...")
            port = 8080 if ":8080" in vespa_endpoint else 8080
            self.vespa_docker = VespaDocker(
                port=port,
                container_memory=4 * 1024 ** 3  # 4GB in bytes
            )
        
        # Deploy to the local Vespa container
        self.app = self.vespa_docker.deploy(
            application_package=self.application_package
        )
        
        print("Docker deployment successful!")
        return self.app
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents in Vespa
        
        Args:
            documents: List of documents to index
        """
        if not self.app:
            raise RuntimeError("Application not deployed. Call deploy() first.")
        
        print(f"Indexing {len(documents)} documents...")
        
        async with self.app.asyncio(connections=1, timeout=Config.INDEXING_TIMEOUT) as session:
            for doc in tqdm(documents, desc="Indexing documents"):
                try:
                    response: VespaResponse = await session.feed_data_point(
                        data_id=doc["id"], 
                        fields=doc, 
                        schema="pdf_page"
                    )
                    
                    if not response.is_successful():
                        print(f"Failed to index document {doc['id']}: {response.json()}")
                        
                except Exception as e:
                    print(f"Error indexing document {doc['id']}: {str(e)}")
    
    async def query_with_bm25(self, query: str, query_embedding: Dict[str, Any], hits: int = None) -> VespaQueryResponse:
        """
        Query using BM25 for retrieval and ColPali for reranking
        
        Args:
            query: Text query
            query_embedding: Query embedding dictionary
            hits: Number of results to return
            
        Returns:
            VespaQueryResponse
        """
        if not self.app:
            raise RuntimeError("Application not deployed. Call deploy() first.")
        
        hits = hits or Config.DEFAULT_HITS
        
        async with self.app.asyncio(connections=1, timeout=Config.QUERY_TIMEOUT) as session:
            response: VespaQueryResponse = await session.query(
                yql="select title,url,image,page_number from pdf_page where userInput(@userQuery)",
                ranking="default",
                userQuery=query,
                timeout=Config.QUERY_TIMEOUT,
                hits=hits,
                body={"input.query(qt)": query_embedding, "presentation.timing": True},
            )
            
            return response
    
    async def query_with_nearest_neighbor(self, float_query_embedding: Dict[str, Any], 
                                        binary_query_embedding: Dict[str, Any], 
                                        hits: int = None,
                                        target_hits_per_query: int = None) -> VespaQueryResponse:
        """
        Query using nearestNeighbor for retrieval with binary representations
        
        Args:
            float_query_embedding: Float query embedding dictionary
            binary_query_embedding: Binary query embedding dictionary
            hits: Number of results to return
            target_hits_per_query: Target hits per query tensor
            
        Returns:
            VespaQueryResponse
        """
        if not self.app:
            raise RuntimeError("Application not deployed. Call deploy() first.")
        
        hits = hits or Config.DEFAULT_HITS
        target_hits_per_query = target_hits_per_query or Config.TARGET_HITS_PER_QUERY_TENSOR
        
        # Build query tensors
        query_tensors = {
            "input.query(qtb)": binary_query_embedding,
            "input.query(qt)": float_query_embedding,
        }
        
        # Add individual query tensors for nearestNeighbor
        for i in range(len(binary_query_embedding)):
            query_tensors[f"input.query(rq{i})"] = binary_query_embedding[i]
        
        # Build nearestNeighbor query
        nn_queries = []
        for i in range(len(binary_query_embedding)):
            nn_queries.append(
                f"({{targetHits:{target_hits_per_query}}}nearestNeighbor(embedding,rq{i}))"
            )
        
        nn_query = " OR ".join(nn_queries)
        
        async with self.app.asyncio(connections=1, timeout=Config.QUERY_TIMEOUT) as session:
            response: VespaQueryResponse = await session.query(
                yql=f"select title, url, image, page_number from pdf_page where {nn_query}",
                ranking="retrieval-and-rerank",
                timeout=Config.QUERY_TIMEOUT,
                hits=hits,
                body={**query_tensors, "presentation.timing": True},
            )
            
            return response
    
    def get_app_info(self) -> Dict[str, Any]:
        """Get information about the deployed application"""
        if not self.app:
            return {"status": "not_deployed"}
        
        deployment_type = "docker" if self.is_docker_mode else "cloud"
        endpoint = getattr(self.app, "url", "unknown")
        
        return {
            "status": "deployed",
            "deployment_type": deployment_type,
            "tenant": self.tenant_name if not self.is_docker_mode else "local",
            "application": self.app_name,
            "endpoint": endpoint
        } 