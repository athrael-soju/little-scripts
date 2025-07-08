import io
import itertools
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, List, Optional

import config
from qdrant_client.http import models
from tqdm import tqdm
from utils import Colors, colored_print


@dataclass
class ProcessingMetrics:
    """Simple metrics tracking for indexing operations."""

    start_time: float = 0
    total_docs: int = 0
    processed_docs: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    total_upload_tasks: int = 0
    completed_uploads: int = 0
    failed_uploads: int = 0

    def start(self):
        self.start_time = time.time()

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def processing_rate(self) -> float:
        elapsed = self.elapsed_time()
        return self.processed_docs / elapsed if elapsed > 0 else 0

    def print_summary(self):
        """Print a clean summary of metrics."""
        elapsed = self.elapsed_time()
        rate = self.processing_rate()

        colored_print("\n📊 Processing Summary", Colors.HEADER)
        colored_print(f"   • Documents: {self.processed_docs:,}", Colors.OKCYAN)
        colored_print(f"   • Time: {elapsed:.1f}s ({rate:.1f} docs/sec)", Colors.OKCYAN)
        colored_print(
            f"   • Batches: {self.successful_batches} successful, {self.failed_batches} failed",
            Colors.OKCYAN,
        )
        if self.total_upload_tasks > 0:
            colored_print(
                f"   • Images: {self.completed_uploads}/{self.total_upload_tasks} uploaded",
                Colors.OKCYAN,
            )


def batch_iterable(iterable, batch_size):
    """Batches an iterable into lists of a given size."""
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch


@dataclass
class ImageUploadTask:
    """Represents a MinIO upload task."""

    point_id: str
    image_data: bytes  # Still keeping this for compatibility
    image_name: str
    retries: int = 0
    max_retries: int = 3


@dataclass
class ImageProcessingTask:
    """Represents an image processing and upload task with raw PIL image."""

    point_id: str
    pil_image: Any  # PIL Image object
    image_name: str
    retries: int = 0
    max_retries: int = 3


class RetrievalPipeline:
    """Orchestrates the document retrieval pipeline."""

    def __init__(self, model_handler, vector_db, minio_handler, openai_handler=None):
        self.model_handler = model_handler
        self.vector_db = vector_db
        self.minio_handler = minio_handler
        self.openai_handler = openai_handler

        # Background upload management for optimized pipeline
        self.upload_queue = Queue()
        self.upload_results = {}  # point_id -> image_url
        self.upload_errors = {}  # point_id -> error_message
        self.upload_worker_active = False
        self.metrics = ProcessingMetrics()

    def setup(self):
        """Set up the model and vector database."""
        self.model_handler.setup()
        self.vector_db.create_collection()
        if self.minio_handler:
            self.minio_handler.ensure_bucket_exists()

    def _background_upload_worker(self):
        """Background worker to handle image processing and MinIO uploads."""
        while self.upload_worker_active or not self.upload_queue.empty():
            try:
                # Get upload task with timeout
                try:
                    task = self.upload_queue.get(timeout=1.0)
                except Exception as e:
                    colored_print(f"❌ Error getting upload task: {e}", Colors.FAIL)
                    continue

                try:
                    # Handle both PIL image processing tasks and pre-processed upload tasks
                    if isinstance(task, ImageProcessingTask):
                        # Process PIL image to bytes (moved from main thread)
                        buffer = io.BytesIO()
                        save_kwargs = {"format": config.IMAGE_FORMAT}
                        if config.IMAGE_FORMAT.upper() == "JPEG":
                            save_kwargs["quality"] = config.IMAGE_QUALITY
                        task.pil_image.save(buffer, **save_kwargs)
                        image_bytes = buffer.getvalue()

                        # Now upload to MinIO
                        image_url = self.minio_handler.upload_image(
                            task.image_name, image_bytes
                        )
                    else:
                        # Handle legacy ImageUploadTask (already processed bytes)
                        image_url = self.minio_handler.upload_image(
                            task.image_name, task.image_data
                        )

                    # Store successful result
                    self.upload_results[task.point_id] = image_url
                    self.metrics.completed_uploads += 1

                    # Update Qdrant point with image URL
                    self._update_point_with_image_url(task.point_id, image_url)

                except Exception as e:
                    # Handle upload failure
                    if task.retries < task.max_retries:
                        task.retries += 1
                        self.upload_queue.put(task)
                    else:
                        self.upload_errors[task.point_id] = str(e)
                        self.metrics.failed_uploads += 1

                self.upload_queue.task_done()

            except Exception:
                continue

    def _update_point_with_image_url(self, point_id: str, image_url: str):
        """Update a Qdrant point with the MinIO image URL."""
        try:
            # Convert string point_id back to integer for Qdrant
            point_id_int = int(point_id)

            # Update the point's payload with the image URL
            self.vector_db.client.set_payload(
                collection_name=self.vector_db.collection_name,
                payload={"image_url": image_url, "upload_pending": False},
                points=[point_id_int],
            )
        except Exception:
            pass  # Silently continue, not critical

    def get_upload_status(self):
        """Get the current status of image processing and upload tasks."""
        return {
            "total": self.metrics.total_upload_tasks,
            "completed": self.metrics.completed_uploads,
            "failed": self.metrics.failed_uploads,
            "pending": self.metrics.total_upload_tasks
            - self.metrics.completed_uploads
            - self.metrics.failed_uploads,
            "queue_size": self.upload_queue.qsize(),
        }

    def index_documents(
        self, documents_iterable, batch_size, total_docs=None, source_name="unknown"
    ):
        """Index documents with decoupled image processing and vector storage."""
        if not self.minio_handler:
            raise ValueError(
                "MinIO handler is not configured. Cannot index documents with images."
            )

        colored_print("🚀 Starting document indexing...", Colors.OKBLUE)

        # Initialize metrics
        self.metrics = ProcessingMetrics()
        self.metrics.start()
        self.metrics.total_docs = total_docs or 0

        # Start background upload worker
        self.upload_worker_active = True
        upload_thread = threading.Thread(
            target=self._background_upload_worker, daemon=True
        )
        upload_thread.start()

        try:
            initial_points_count = self.vector_db.client.get_collection(
                self.vector_db.collection_name
            ).points_count
        except Exception:
            initial_points_count = 0

        global_doc_index = 0
        batched_docs = batch_iterable(documents_iterable, batch_size)

        pbar = (
            tqdm(total=total_docs, desc="Processing", unit="docs", miniters=1)
            if total_docs
            else tqdm(desc="Processing", unit="docs", miniters=1)
        )

        with pbar:
            for batch_id, batch_data in enumerate(batched_docs):
                try:
                    # Phase 1: Get embeddings and create Qdrant points
                    images = [doc["image"] for doc in batch_data]
                    image_embeddings = self.model_handler.get_image_embeddings(images)

                    points = []
                    upload_tasks = []

                    for j, doc in enumerate(batch_data):
                        current_index = global_doc_index + j
                        point_id = initial_points_count + current_index

                        # Create Qdrant point immediately with placeholder image URL
                        embedding = image_embeddings[j]
                        multivector = embedding.cpu().float().numpy().tolist()

                        payload = {
                            "source": doc.get("source", source_name),
                            "dataset_index": current_index,
                            "batch_id": batch_id,
                            "has_image": True,
                            "image_url": None,  # Will be updated by background worker
                            "upload_pending": True,  # Flag to indicate upload is pending
                        }

                        # Add other document fields
                        for key, value in doc.items():
                            if key != "image" and isinstance(
                                value, (str, int, float, bool)
                            ):
                                payload[key] = value

                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector=multivector,
                                payload=payload,
                            )
                        )

                        # Phase 2: Prepare image processing task (TRULY ASYNC)
                        pil_image = doc["image"]
                        safe_source_name = "".join(
                            c if c.isalnum() else "_" for c in source_name
                        )
                        file_extension = (
                            "jpg" if config.IMAGE_FORMAT.upper() == "JPEG" else "png"
                        )
                        image_name = (
                            f"{safe_source_name}_idx{point_id}.{file_extension}"
                        )

                        # Queue PIL image for background processing (no conversion in main thread!)
                        processing_task = ImageProcessingTask(
                            point_id=str(point_id),
                            pil_image=pil_image,
                            image_name=image_name,
                            max_retries=config.MINIO_UPLOAD_RETRY_ATTEMPTS,
                        )
                        upload_tasks.append(processing_task)

                    # Immediately upsert to Qdrant (no waiting for MinIO)
                    self.vector_db.upsert_batch(points)
                    self.metrics.successful_batches += 1

                    # Queue MinIO uploads (non-blocking)
                    for upload_task in upload_tasks:
                        self.upload_queue.put(upload_task)
                        self.metrics.total_upload_tasks += 1

                    # Update metrics and progress
                    self.metrics.processed_docs += len(images)
                    global_doc_index += len(images)
                    pbar.update(len(images))

                except Exception as e:
                    self.metrics.failed_batches += 1
                    colored_print(
                        f"❌ Error processing batch {batch_id + 1}: {e}", Colors.FAIL
                    )
                    continue

        # Stop background worker
        self.upload_worker_active = False
        upload_thread.join(timeout=config.MINIO_UPLOAD_TIMEOUT)

        # Print summary
        self.metrics.print_summary()

        if self.metrics.failed_batches > 0:
            colored_print(
                f"⚠️  {self.metrics.failed_batches} batches failed", Colors.WARNING
            )

        colored_print("✅ Indexing complete!", Colors.OKGREEN)

        # Optimize collection if enabled
        if config.OPTIMIZE_COLLECTION:
            colored_print("⏳ Optimizing collection...", Colors.OKBLUE)
            self.vector_db.optimize_collection()
            colored_print("✅ Collection optimized!", Colors.OKGREEN)

    def search(self, query_text, limit, oversampling):
        """Search for documents using a text query."""
        start_time = time.time()
        query_embedding = self.model_handler.get_query_embedding(query_text)

        # query_embedding is a batch result from processing [query_text], so we need [0] to get the first result
        multivector_query = query_embedding[0].cpu().float().numpy().tolist()

        result = self.vector_db.search(multivector_query, limit, oversampling)

        search_time = time.time() - start_time
        colored_print(f"🔍 Search completed in {search_time:.2f}s", Colors.OKCYAN)

        return result

    def get_images_from_results(self, search_results, max_images: int = 5) -> List[str]:
        """Retrieve image URLs from search results."""
        image_urls = []
        for point in search_results.points[:max_images]:
            try:
                if point.payload and "image_url" in point.payload:
                    # Only include images that have been successfully uploaded
                    if point.payload["image_url"] is not None:
                        image_urls.append(point.payload["image_url"])
                    elif point.payload.get("upload_pending", False):
                        # Image upload is still pending, skip for now
                        continue
            except Exception:
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
                # Check if any images are still pending upload
                pending_count = sum(
                    1
                    for point in search_results.points[:max_images]
                    if point.payload and point.payload.get("upload_pending", False)
                )

                if pending_count > 0:
                    yield {
                        "openai_response": f"Found relevant documents but {pending_count} images are still uploading. Please try again in a moment.",
                        "type": "openai_response",
                    }
                else:
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
