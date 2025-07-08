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
                hnsw_ef=128,  # Increased for better search quality
            ),
        )

        return search_result
