import os
from pathlib import Path

import config
from datasets import load_dataset
from handlers.minio import MinioHandler
from handlers.model import ModelHandler
from handlers.openai import OpenAIHandler
from handlers.qdrant import QdrantHandler
from pdf2image import convert_from_path
from pypdf import PdfReader
from utils import Colors, colored_print

from .error_handling import (
    SERVICE_MANAGER,
    CriticalError,
    ErrorHandler,
    ValidationManager,
    safe_execution,
)
from .messaging import (
    InteractiveHelp,
    SetupMessages,
    StatusDisplay,
    UIMessages,
    ValidationMessages,
)
from .pipeline import RetrievalPipeline


@safe_execution(context="Pipeline setup", return_on_error=None)
def setup_pipeline(include_openai=False):
    """Initializes and returns the model_handler, vector_db, and pipeline."""
    try:
        # Initialize core services
        model_handler = ModelHandler(config.MODEL_NAME, config.PROCESSOR_NAME)
        vector_db = QdrantHandler(
            config.QDRANT_URL,
            config.COLLECTION_NAME,
            config.VECTOR_SIZE,
            config.DISTANCE_METRIC,
        )

        # Initialize MinIO handler (required service)
        minio_handler = SERVICE_MANAGER.register_service(
            "MinIO", lambda: MinioHandler()
        )

        # Initialize OpenAI handler (optional service)
        openai_handler = None
        if include_openai and config.OPENAI_API_KEY:
            try:
                openai_handler = OpenAIHandler()
                if openai_handler.is_available():
                    SERVICE_MANAGER.register_service("OpenAI", lambda: openai_handler)
                    SetupMessages.service_configured("OpenAI integration")
            except Exception as e:
                # OpenAI is optional - log but don't fail
                SetupMessages.service_failed("OpenAI", str(e))
                openai_handler = None

        pipeline = RetrievalPipeline(
            model_handler, vector_db, minio_handler, openai_handler
        )
        return model_handler, vector_db, pipeline, openai_handler
    except Exception as e:
        raise CriticalError(f"Pipeline setup failed: {e}")


@safe_execution(context="Query execution", return_on_error=False)
def ask(
    query, pipeline, vector_db, openai_handler, interactive=False, use_openai=False
):
    """Asks a question to the retrieval system."""
    # Validate query
    if not ValidationManager.validate_query(query):
        return False

    UIMessages.info(f"Searching for: '{query}'")

    try:
        # Validate collection state
        if not ValidationManager.validate_collection_state(
            vector_db, config.COLLECTION_NAME
        ):
            if interactive:
                ValidationMessages.interactive_tip_upload()
            return False

        # Ensure model is loaded for searching
        pipeline.model_handler.setup()

        # Perform search with or without OpenAI analysis
        if use_openai and openai_handler:
            UIMessages.info("Getting conversational response...")

            response_generator = pipeline.search_with_openai_response(
                query, config.SEARCH_LIMIT, config.OVERSAMPLING
            )

            # Handle generator for streaming response
            results_displayed = False
            images = []
            citation_urls = []  # Collect URLs for citations

            for response_part in response_generator:
                if response_part["type"] == "results":
                    results = response_part["search_results"]
                    if not results.points:
                        ValidationMessages.no_results_found()
                        if interactive:
                            ValidationMessages.interactive_tip_search()
                        return False

                    # In conversational mode, just collect citation data without displaying search results
                    for point in results.points:
                        if hasattr(point, "payload") and point.payload:
                            source_info = {
                                "source": point.payload.get("source", "Unknown"),
                                "page_num": point.payload.get("page_num", "Unknown"),
                                "image_url": None,
                            }
                            for key, value in point.payload.items():
                                if key.lower() == "image_url":
                                    source_info["image_url"] = value
                                    break
                            if source_info["image_url"]:
                                citation_urls.append(source_info)
                    results_displayed = True

                elif response_part["type"] == "images":
                    images = response_part["images"]

                elif response_part["type"] == "openai_chunk":
                    if results_displayed:
                        colored_print("\n🧠 Response:", Colors.HEADER)
                        colored_print("=" * 60, Colors.HEADER)
                        results_displayed = False  # Header should only be printed once

                    print(
                        f"{Colors.OKCYAN}{response_part['openai_response_chunk']}{Colors.ENDC}",
                        end="",
                        flush=True,
                    )

                elif response_part["type"] == "openai_response":
                    if results_displayed:  # If there was no stream for some reason
                        colored_print("\n🧠 Response:", Colors.HEADER)
                        colored_print("=" * 60, Colors.HEADER)
                    print(
                        f"{Colors.OKCYAN}{response_part['openai_response']}{Colors.ENDC}"
                    )

            if images:
                # print()  # Newline after streaming
                # colored_print(
                #     f"\n📸 Response based on {len(images)} retrieved images",
                #     Colors.OKCYAN,
                # )

                # Display citations with source information
                if citation_urls:
                    colored_print("\n📚 Sources:", Colors.HEADER)
                    for i, source_info in enumerate(citation_urls, 1):
                        colored_print(
                            f"  [{i}] {source_info['image_url']}", Colors.OKCYAN
                        )
                        colored_print(
                            f"      Source: {source_info['source']}, Page: {source_info['page_num']}",
                            Colors.OKCYAN,
                        )

        else:
            results = pipeline.search(query, config.SEARCH_LIMIT, config.OVERSAMPLING)
            if not results.points:
                ValidationMessages.no_results_found()
                if interactive:
                    ValidationMessages.interactive_tip_search()
                return False

            StatusDisplay.show_search_results_header(query, len(results.points))

            for i, point in enumerate(results.points):
                UIMessages.info(f"{i + 1:2d}. Document ID: {point.id}")
                UIMessages.status_item("Similarity Score", f"{point.score:.4f}")
                if hasattr(point, "payload") and point.payload:
                    UIMessages.status_item(
                        "Source", point.payload.get("source", "Unknown")
                    )
                    # Show additional metadata if available
                    for key, value in point.payload.items():
                        if key not in [
                            "source",
                            "dataset_index",
                            "batch_id",
                            "has_image",
                        ]:
                            UIMessages.status_item(key.title(), str(value))
                    print()

        return True

    except Exception as e:
        return ErrorHandler.handle_error(e, "Search operation")


@safe_execution(context="Document upload", return_on_error=False)
def upload(pipeline, file_path=None, interactive=False):
    """Uploads and indexes documents."""
    if not ValidationManager.validate_file_path(file_path):
        return False

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    documents_iterable = None
    total_docs = 0
    source_name = "unknown"

    try:
        if file_path:
            UIMessages.info(f"Loading data from file: {file_path}")
            path = Path(file_path)
            source_name = path.name

            if path.suffix.lower() == ".pdf":

                def pdf_page_generator(file_path):
                    reader = PdfReader(file_path)
                    for i in range(len(reader.pages)):
                        page_num = i + 1
                        page_image = convert_from_path(
                            file_path, first_page=page_num, last_page=page_num
                        )[0]
                        yield {
                            "image": page_image,
                            "source": source_name,
                            "page_num": page_num,
                        }

                try:
                    reader = PdfReader(file_path)
                    total_docs = len(reader.pages)
                    documents_iterable = pdf_page_generator(file_path)
                    UIMessages.info(
                        f"PDF detected with {total_docs} pages. Starting stream processing..."
                    )
                except Exception as e:
                    UIMessages.error(f"Error reading PDF: {e}")
                    return False
            else:
                # Handle text files by checking for an 'image' column later
                documents_iterable = load_dataset(
                    "text", data_files={"train": file_path}, split="train"
                )
                total_docs = len(documents_iterable)
                # This check is tricky for generic iterables, so we'll rely on the pipeline to handle it.
                if "image" not in documents_iterable.features:
                    UIMessages.error(
                        "The loaded data does not contain images and cannot be indexed by the current pipeline."
                    )
                    UIMessages.tip(
                        "This pipeline is configured for image data. Please upload a supported file type (e.g., PDF) or use the default image dataset."
                    )
                    return False

        else:
            source_name = config.DATASET_NAME
            UIMessages.info("Loading UFO dataset from Hugging Face...")
            UIMessages.tip("This may take a while for the first download")
            documents_iterable = load_dataset(
                config.DATASET_NAME, split=config.DATASET_SPLIT
            )
            total_docs = len(documents_iterable)

        # The new pipeline can handle any iterable with 'image' keys, so this check is no longer needed here.
        # We'll let the pipeline raise an error if the data is invalid.

        UIMessages.processing_start("Setting up pipeline and indexing documents...")
        pipeline.setup()
        pipeline.index_documents(
            documents_iterable,
            config.BATCH_SIZE,
            total_docs=total_docs,
            source_name=source_name,
        )
        UIMessages.processing_complete("Documents indexed successfully!")

        # Show background processing status
        upload_status = pipeline.get_upload_status()
        if upload_status["pending"] > 0:
            UIMessages.info(
                f"Background processing: {upload_status['completed']}/{upload_status['total']} completed, {upload_status['pending']} pending"
            )

        if interactive:
            UIMessages.tip(
                "You can now ask questions directly! Just type what you want to know."
            )

        return True

    except Exception as e:
        return ErrorHandler.handle_error(e, "Document upload")


@safe_execution(context="Data clearing", return_on_error=False)
def clear(vector_db, minio_handler, interactive=False):
    """Clears the vector database collection and MinIO bucket."""
    if interactive:
        UIMessages.warning(
            "This will permanently delete all indexed documents and images!"
        )
        confirmation = (
            input("Are you sure you want to continue? (y/N): ").lower().strip()
        )
        if confirmation not in ["y", "yes"]:
            UIMessages.info("Operation cancelled")
            return False

    success = True

    # Clear Qdrant collection
    try:
        UIMessages.warning("Clearing Qdrant collection...")
        vector_db.recreate_collection()
        UIMessages.success("Qdrant collection cleared and recreated")
    except Exception as e:
        UIMessages.error(f"Error clearing Qdrant collection: {e}")
        success = False

    # Clear MinIO bucket
    try:
        UIMessages.warning("Clearing MinIO bucket...")
        minio_handler.clear_bucket()
        UIMessages.success("MinIO bucket cleared")
    except Exception as e:
        UIMessages.error(f"Error clearing MinIO bucket: {e}")
        success = False

    if success:
        UIMessages.success("All data cleared successfully")
    else:
        UIMessages.warning("Some cleanup operations failed")

    return success


@safe_execution(context="Status check", return_on_error=False)
def status(vector_db, openai_handler, minio_handler, pipeline=None):
    """Shows the current status of the system."""
    StatusDisplay.show_system_status_header()

    try:
        # ================================================================
        # CONNECTION STATUS
        # ================================================================
        StatusDisplay.show_connection_status_header()

        # Check Qdrant connection
        try:
            vector_db.client.get_collections()
            UIMessages.connection_status("Qdrant", True, {"URL": config.QDRANT_URL})
        except Exception:
            UIMessages.connection_status("Qdrant", False, {"URL": config.QDRANT_URL})
            UIMessages.tip("Make sure Qdrant is running")
            return False

        # Check OpenAI connection
        if openai_handler and openai_handler.is_available():
            if openai_handler.test_connection():
                UIMessages.connection_status(
                    "OpenAI", True, {"Model": config.OPENAI_MODEL}
                )
            else:
                UIMessages.connection_status(
                    "OpenAI", False, {"Model": config.OPENAI_MODEL}
                )
        else:
            UIMessages.warning("OpenAI: Not Configured")
            UIMessages.tip("Set OPENAI_API_KEY for conversational mode")

        # Check MinIO connection
        try:
            minio_handler.ensure_bucket_exists()
            UIMessages.connection_status(
                "MinIO",
                True,
                {"Endpoint": config.MINIO_ENDPOINT, "Bucket": config.MINIO_BUCKET},
            )
        except Exception as e:
            UIMessages.connection_status(
                "MinIO", False, {"Endpoint": config.MINIO_ENDPOINT, "Error": str(e)}
            )

        # ================================================================
        # COLLECTION STATUS
        # ================================================================
        StatusDisplay.show_collection_status_header()

        # Get collection info
        try:
            collection_info = vector_db.client.get_collection(config.COLLECTION_NAME)
            collection_exists = True
            UIMessages.success("Collection: Ready")
            UIMessages.status_item("Name", config.COLLECTION_NAME)
            UIMessages.status_item("Documents", f"{collection_info.points_count:,}")
            UIMessages.status_item("Vector Size", str(config.VECTOR_SIZE))
            UIMessages.status_item_last("Distance Metric", config.DISTANCE_METRIC)
        except Exception:
            collection_exists = False
            UIMessages.warning("Collection: Not Found")
            UIMessages.status_item("Name", config.COLLECTION_NAME)
            UIMessages.status_item("Documents", "N/A")
            UIMessages.status_item("Vector Size", str(config.VECTOR_SIZE))
            UIMessages.status_item_last("Distance Metric", config.DISTANCE_METRIC)

        # ================================================================
        # BACKGROUND PROCESSING STATUS
        # ================================================================
        if pipeline:
            StatusDisplay.show_processing_status_header()

            upload_status = pipeline.get_upload_status()
            if upload_status["total"] > 0:
                UIMessages.status_item("Total Tasks", str(upload_status["total"]))
                UIMessages.status_item("Completed", str(upload_status["completed"]))

                UIMessages.status_item("Failed", str(upload_status["failed"]))

                UIMessages.status_item("Pending", str(upload_status["pending"]))

                UIMessages.status_item_last(
                    "Queue Size", str(upload_status["queue_size"])
                )

                if upload_status["pending"] > 0:
                    UIMessages.tip("Background image processing is still running")
            else:
                UIMessages.status_item_last(
                    "Processing Status", "No processing tasks in queue"
                )

        # ================================================================
        # MODEL CONFIGURATION
        # ================================================================
        StatusDisplay.show_model_config_header()
        UIMessages.status_item("ColPali Model", config.MODEL_NAME)
        UIMessages.status_item("OpenAI Model", config.OPENAI_MODEL)
        UIMessages.status_item("Search Limit", str(config.SEARCH_LIMIT))
        UIMessages.status_item_last("Background Processing", "Enabled")

        # ================================================================
        # SYSTEM SUMMARY
        # ================================================================
        StatusDisplay.show_system_summary_header()

        if collection_exists and collection_info.points_count > 0:
            UIMessages.success("Status: Ready for searches")
            InteractiveHelp.show_next_steps(True)
        elif collection_exists and collection_info.points_count == 0:
            UIMessages.warning("Status: Collection empty")
            InteractiveHelp.show_next_steps(False)
        else:
            UIMessages.warning("Status: Setup incomplete")
            UIMessages.tip("Use 'upload <file>' to create collection and add documents")

        return True

    except Exception as e:
        return ErrorHandler.handle_error(e, "Status check")
