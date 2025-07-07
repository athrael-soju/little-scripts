import io
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import config
from qdrant_client.http import models
from tqdm import tqdm
from utils import Colors, colored_print


def batch_iterable(iterable, batch_size):
    """Batches an iterable into lists of a given size."""
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch


class RetrievalPipeline:
    """Orchestrates the document retrieval pipeline."""

    def __init__(self, model_handler, vector_db, minio_handler, openai_handler=None):
        self.model_handler = model_handler
        self.vector_db = vector_db
        self.minio_handler = minio_handler
        self.openai_handler = openai_handler

    def setup(self):
        """Set up the model and vector database."""
        self.model_handler.setup()
        self.vector_db.create_collection()
        if self.minio_handler:
            self.minio_handler.ensure_bucket_exists()

    def index_documents(
        self, documents_iterable, batch_size, total_docs=None, source_name="unknown"
    ):
        """Index documents from any iterable into the Qdrant collection."""
        if not self.minio_handler:
            raise ValueError(
                "Minio handler is not configured. Cannot index documents with images."
            )

        print("Indexing documents...")

        try:
            initial_points_count = self.vector_db.client.get_collection(
                self.vector_db.collection_name
            ).points_count
        except Exception:
            initial_points_count = 0

        global_doc_index = 0
        batched_docs = batch_iterable(documents_iterable, batch_size)

        pbar = (
            tqdm(total=total_docs, desc="Indexing Progress")
            if total_docs
            else tqdm(desc="Indexing Progress")
        )

        with pbar:
            for batch_id, batch_data in enumerate(batched_docs):
                images = [doc["image"] for doc in batch_data]
                image_embeddings = self.model_handler.get_image_embeddings(images)

                points = [None] * len(batch_data)
                future_to_index = {}

                with ThreadPoolExecutor() as executor:
                    for j, doc in enumerate(batch_data):
                        current_index = global_doc_index + j
                        point_id = initial_points_count + current_index

                        pil_image = doc["image"]
                        safe_source_name = "".join(
                            c if c.isalnum() else "_" for c in source_name
                        )
                        image_name = f"{safe_source_name}_idx{point_id}.png"

                        buffer = io.BytesIO()
                        pil_image.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()

                        future = executor.submit(
                            self.minio_handler.upload_image, image_name, image_bytes
                        )
                        future_to_index[future] = j

                    for future in as_completed(future_to_index):
                        j = future_to_index[future]
                        try:
                            image_url = future.result()
                            doc = batch_data[j]
                            embedding = image_embeddings[j]
                            multivector = embedding.cpu().float().numpy().tolist()

                            current_index = global_doc_index + j
                            point_id = initial_points_count + current_index

                            payload = {
                                "source": doc.get("source", source_name),
                                "dataset_index": current_index,
                                "batch_id": batch_id,
                                "has_image": True,
                                "image_url": image_url,
                            }

                            for key, value in doc.items():
                                if key != "image" and isinstance(
                                    value, (str, int, float, bool)
                                ):
                                    payload[key] = value

                            points[j] = models.PointStruct(
                                id=point_id,
                                vector=multivector,
                                payload=payload,
                            )
                        except Exception as e:
                            print(f"Error uploading image for doc index {j}: {e}")

                # Filter out any points that failed during upload
                successful_points = [p for p in points if p is not None]

                if successful_points:
                    try:
                        self.vector_db.upsert_batch(successful_points)
                    except Exception as e:
                        print(f"Error during batch upload: {e}")
                        continue

                global_doc_index += len(images)
                pbar.update(len(images))

        print("Indexing complete!")

        # Optimize collection if enabled
        if config.OPTIMIZE_COLLECTION:
            colored_print("⏳ Optimizing collection for search...", Colors.OKBLUE)
            self.vector_db.optimize_collection()
            colored_print("✅ Collection optimized successfully!", Colors.OKGREEN)

    def search(self, query_text, limit, oversampling):
        """Search for documents using a text query."""
        print(f"Searching for: '{query_text}'")
        query_embedding = self.model_handler.get_query_embedding(query_text)
        multivector_query = query_embedding[0].cpu().float().numpy().tolist()

        return self.vector_db.search(multivector_query, limit, oversampling)

    def get_images_from_results(self, search_results, max_images: int = 5) -> List[str]:
        """Retrieve image URLs from search results."""
        image_urls = []
        for point in search_results.points[:max_images]:
            try:
                if point.payload and "image_url" in point.payload:
                    image_urls.append(point.payload["image_url"])
            except Exception as e:
                print(
                    f"Warning: Failed to retrieve image URL for point {point.id}: {e}"
                )
                continue

        return image_urls

    def search_with_openai_response(
        self,
        query_text,
        limit,
        oversampling,
        max_images: int = 3,
        system_prompt: Optional[str] = None,
    ):
        """
        Search for documents and get OpenAI analysis of the results.
        This function yields the search results first, then streams the OpenAI response.
        """
        if not self.openai_handler:
            raise ValueError("OpenAI handler not configured")

        # Perform search
        search_results = self.search(query_text, limit, oversampling)

        # Yield search results immediately
        yield {"search_results": search_results, "type": "results"}

        if not search_results.points:
            yield {
                "openai_response": "No relevant images found for your query.",
                "type": "openai_response",
            }
            return

        try:
            # Get images from results
            image_urls = self.get_images_from_results(search_results, max_images)

            if not image_urls:
                yield {
                    "openai_response": "Found relevant documents but couldn't retrieve images.",
                    "type": "openai_response",
                }
                return

            # Default system prompt if none provided
            if not system_prompt:
                system_prompt = (
                    "You are a helpful AI assistant. Look at the provided images and answer the user's question naturally and simply. "
                    "If you can answer based on what you see, give a brief, direct answer. "
                    "If the images don't contain the information needed, just say 'I don't know' or 'I can't find that information in the provided context.'\n\n"
                    "Example:\n"
                    "User: 'What color is the car?'\n"
                    "You: 'The car is blue.'\n\n"
                    "User: 'What's the population of this city?'\n"
                    "You: 'I can't find that information in the provided context.'"
                )

            # Get OpenAI response stream
            response_stream = self.openai_handler.analyze_images(
                images=image_urls,
                user_prompt=query_text,
                system_prompt=system_prompt,
                stream=True,
            )

            # Yield image URLs and stream the OpenAI response
            yield {"images": image_urls, "type": "images"}
            for chunk in response_stream:
                yield {"openai_response_chunk": chunk, "type": "openai_chunk"}

        except Exception as e:
            yield {
                "openai_response": f"Error getting AI analysis: {e}",
                "type": "openai_response",
            }
