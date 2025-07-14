import config
import stamina
from qdrant_client import QdrantClient, models


class QdrantHandler:
    """Handles all interactions with the Qdrant vector database."""

    def __init__(self, qdrant_url, collection_name, vector_size, distance_metric):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = self._get_distance_metric(distance_metric)

    def _get_distance_metric(self, distance_metric_str):
        """Maps distance metric string to Qdrant models.Distance enum."""
        metric_map = {
            "COSINE": models.Distance.COSINE,
            "EUCLID": models.Distance.EUCLID,
            "DOT": models.Distance.DOT,
        }
        return metric_map.get(distance_metric_str.upper(), models.Distance.COSINE)

    def collection_exists(self):
        """Check if the collection exists."""
        try:
            self.client.get_collection(collection_name=self.collection_name)
            return True
        except Exception:
            return False

    def create_collection(self):
        """Create Qdrant collection with binary quantization if it doesn't exist."""
        if self.collection_exists():
            return

        if config.ENABLE_RERANKING_OPTIMIZATION:
            # Create collection with multiple vector configurations (Mean Pooling and Reranking Optimization)
            self.client.create_collection(
                collection_name=self.collection_name,
                on_disk_payload=True,
                vectors_config={
                    "original": models.VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric,
                        on_disk=True,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        quantization_config=models.BinaryQuantization(
                            binary=models.BinaryQuantizationConfig(always_ram=True),
                        ),
                        hnsw_config=models.HnswConfigDiff(
                            m=0
                        ),  # HNSW turned off for original vectors
                    ),
                    "mean_pooling_columns": models.VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric,
                        on_disk=True,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        quantization_config=models.BinaryQuantization(
                            binary=models.BinaryQuantizationConfig(always_ram=True),
                        ),
                    ),
                    "mean_pooling_rows": models.VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric,
                        on_disk=True,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        quantization_config=models.BinaryQuantization(
                            binary=models.BinaryQuantizationConfig(always_ram=True),
                        ),
                    ),
                },
            )
        else:
            # Original single vector configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                on_disk_payload=True,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance_metric,
                    on_disk=True,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    quantization_config=models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(always_ram=True),
                    ),
                ),
            )

    def recreate_collection(self):
        """Deletes and recreates the collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        self.create_collection()

    @stamina.retry(on=Exception, attempts=3)
    def upsert_batch(self, batch):
        """Upload batch to Qdrant with retry mechanism."""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=False,
            )
            return True
        except Exception as e:
            raise e  # Let the pipeline handle the error

    def optimize_collection(self):
        """Optimize the collection for searching."""
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10),
        )

    def search(self, query_embedding, limit, oversampling):
        """Search for documents using a query embedding."""
        if config.ENABLE_RERANKING_OPTIMIZATION:
            return self._reranking_search(query_embedding, limit)
        else:
            return self._standard_search(query_embedding, limit, oversampling)

    def _standard_search(self, query_embedding, limit, oversampling):
        """Standard search with single vector configuration."""
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            timeout=100,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=oversampling,
                ),
            ),
        )
        return search_result

    def _reranking_search(self, query_embeddings, limit):
        """Reranking search with multiple vector configurations (Mean Pooling and Reranking Optimization)."""
        # Extract query embeddings for each vector type
        original_query = query_embeddings["original"][0]  # First query from batch
        pooled_rows_query = query_embeddings["pooled_rows"][0]
        pooled_columns_query = query_embeddings["pooled_columns"][0]

        # Create search request with prefetch strategy
        search_request = models.QueryRequest(
            query=original_query.tolist(),
            prefetch=[
                models.Prefetch(
                    query=pooled_columns_query.tolist(),
                    limit=config.RERANKING_PREFETCH_LIMIT,
                    using="mean_pooling_columns",
                ),
                models.Prefetch(
                    query=pooled_rows_query.tolist(),
                    limit=config.RERANKING_PREFETCH_LIMIT,
                    using="mean_pooling_rows",
                ),
            ],
            limit=min(limit, config.RERANKING_SEARCH_LIMIT),
            with_payload=True,
            with_vector=False,
            using="original",
        )

        # Execute reranking search
        results = self.client.query_batch_points(
            collection_name=self.collection_name, requests=[search_request]
        )

        return results[0] if results else None
