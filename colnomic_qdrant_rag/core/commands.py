import os
import sys
from pathlib import Path

import config
from datasets import load_dataset
from handlers.minio import MinioHandler
from handlers.model import ModelHandler
from handlers.openai import OpenAIHandler
from handlers.qdrant import QdrantHandler
from pdf2image import convert_from_path
from pypdf import PdfReader
from utils import Colors, colored_print, validate_file_path, validate_query

from .pipeline import RetrievalPipeline


def setup_pipeline(include_openai=False):
    """Initializes and returns the model_handler, vector_db, and pipeline."""
    try:
        model_handler = ModelHandler(config.MODEL_NAME, config.PROCESSOR_NAME)
        vector_db = QdrantHandler(
            config.QDRANT_URL,
            config.COLLECTION_NAME,
            config.VECTOR_SIZE,
            config.DISTANCE_METRIC,
        )

        # Initialize MinIO handler
        minio_handler = None
        try:
            minio_handler = MinioHandler()
            colored_print("🔗 MinIO connection configured", Colors.OKGREEN)
        except Exception as e:
            colored_print(f"⚠️  MinIO setup failed: {e}", Colors.WARNING)
            colored_print("💡 Image upload/analysis will be disabled", Colors.OKCYAN)

        # Initialize OpenAI handler if requested and available
        openai_handler = None
        if include_openai:
            try:
                openai_handler = OpenAIHandler()
                if openai_handler.is_available():
                    colored_print("🤖 OpenAI integration enabled", Colors.OKGREEN)
                else:
                    colored_print(
                        "⚠️  OpenAI API key not found, continuing without conversational responses",
                        Colors.WARNING,
                    )
                    openai_handler = None
            except Exception as e:
                colored_print(f"⚠️  OpenAI setup failed: {e}", Colors.WARNING)
                colored_print(
                    "💡 Continuing without conversational features", Colors.OKCYAN
                )
                openai_handler = None

        pipeline = RetrievalPipeline(
            model_handler, vector_db, minio_handler, openai_handler
        )
        return model_handler, vector_db, pipeline, openai_handler
    except Exception as e:
        colored_print(f"❌ Error setting up pipeline: {e}", Colors.FAIL)
        sys.exit(1)


def ask(
    query, pipeline, vector_db, openai_handler, interactive=False, use_openai=False
):
    """Asks a question to the retrieval system."""
    if not validate_query(query):
        return False

    colored_print(f"🔍 Searching for: '{query}'", Colors.OKBLUE)

    try:
        # Check if collection exists
        if (
            not vector_db.collection_exists()
            or vector_db.client.get_collection(config.COLLECTION_NAME).points_count == 0
        ):
            colored_print("⚠️  Collection is empty or does not exist.", Colors.WARNING)
            if interactive:
                colored_print(
                    "💡 Use 'upload' to add documents first, then ask your questions directly",
                    Colors.OKCYAN,
                )
            return False

        # Ensure model is loaded for searching
        pipeline.model_handler.setup()

        # Perform search with or without OpenAI analysis
        if use_openai and openai_handler:
            colored_print("🧠 Getting conversational response...", Colors.OKCYAN)

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
                        colored_print("📭 No results found.", Colors.WARNING)
                        if interactive:
                            colored_print(
                                "💡 Tip: Try different keywords or check if documents are indexed",
                                Colors.OKCYAN,
                            )
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
                colored_print("📭 No results found.", Colors.WARNING)
                if interactive:
                    colored_print(
                        "💡 Tip: Try different keywords or check if documents are indexed",
                        Colors.OKCYAN,
                    )
                return False

            colored_print(
                f"\n✅ Found {len(results.points)} results for '{query}':",
                Colors.OKGREEN,
            )
            colored_print("=" * 60, Colors.HEADER)

            for i, point in enumerate(results.points):
                print(
                    f"{Colors.BOLD}{i + 1:2d}.{Colors.ENDC} Document ID: {Colors.OKCYAN}{point.id}{Colors.ENDC}"
                )
                print(
                    f"     Similarity Score: {Colors.OKGREEN}{point.score:.4f}{Colors.ENDC}"
                )
                if hasattr(point, "payload") and point.payload:
                    print(f"     Source: {point.payload.get('source', 'Unknown')}")
                    # Show additional metadata if available
                    for key, value in point.payload.items():
                        if key not in [
                            "source",
                            "dataset_index",
                            "batch_id",
                            "has_image",
                        ]:
                            print(f"     {key.title()}: {value}")
                    print()

        return True

    except Exception as e:
        colored_print(f"❌ Error during search: {e}", Colors.FAIL)
        return False


def upload(pipeline, file_path=None, interactive=False):
    """Uploads and indexes documents."""
    if not validate_file_path(file_path):
        return False

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    documents_iterable = None
    total_docs = 0
    source_name = "unknown"

    try:
        if file_path:
            colored_print(f"📂 Loading data from file: {file_path}", Colors.OKBLUE)
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
                    colored_print(
                        f"📄 PDF detected with {total_docs} pages. Starting stream processing...",
                        Colors.OKCYAN,
                    )
                except Exception as e:
                    colored_print(f"❌ Error reading PDF: {e}", Colors.FAIL)
                    return False
            else:
                # Handle text files by checking for an 'image' column later
                documents_iterable = load_dataset(
                    "text", data_files={"train": file_path}, split="train"
                )
                total_docs = len(documents_iterable)
                # This check is tricky for generic iterables, so we'll rely on the pipeline to handle it.
                if "image" not in documents_iterable.features:
                    colored_print(
                        "❌ The loaded data does not contain images and cannot be indexed by the current pipeline.",
                        Colors.FAIL,
                    )
                    colored_print(
                        "💡 This pipeline is configured for image data. Please upload a supported file type (e.g., PDF) or use the default image dataset.",
                        Colors.OKCYAN,
                    )
                    return False

        else:
            source_name = config.DATASET_NAME
            colored_print("🌐 Loading UFO dataset from Hugging Face...", Colors.OKBLUE)
            colored_print(
                "💡 This may take a while for the first download", Colors.OKCYAN
            )
            documents_iterable = load_dataset(
                config.DATASET_NAME, split=config.DATASET_SPLIT
            )
            total_docs = len(documents_iterable)

        # The new pipeline can handle any iterable with 'image' keys, so this check is no longer needed here.
        # We'll let the pipeline raise an error if the data is invalid.

        colored_print("🔧 Setting up pipeline and indexing documents...", Colors.OKBLUE)
        pipeline.setup()
        pipeline.index_documents(
            documents_iterable,
            config.BATCH_SIZE,
            total_docs=total_docs,
            source_name=source_name,
        )
        colored_print("🎉 Documents indexed successfully!", Colors.OKGREEN)

        if interactive:
            colored_print(
                "\n💡 You can now ask questions directly! Just type what you want to know.",
                Colors.OKCYAN,
            )

        return True

    except Exception as e:
        colored_print(f"❌ Error during upload: {e}", Colors.FAIL)
        return False


def clear(vector_db, interactive=False):
    """Clears the vector database collection."""
    if interactive:
        colored_print(
            "⚠️  This will permanently delete all indexed documents!", Colors.WARNING
        )
        confirmation = (
            input("Are you sure you want to continue? (y/N): ").lower().strip()
        )
        if confirmation not in ["y", "yes"]:
            colored_print("❌ Operation cancelled", Colors.OKCYAN)
            return False

    try:
        colored_print("🗑️  Clearing collection...", Colors.WARNING)
        vector_db.recreate_collection()
        colored_print("✅ Collection cleared and recreated", Colors.OKGREEN)
        return True

    except Exception as e:
        colored_print(f"❌ Error clearing collection: {e}", Colors.FAIL)
        return False


def status(vector_db, openai_handler, minio_handler):
    """Shows the current status of the system."""
    colored_print("📊 System Status", Colors.HEADER)
    colored_print("=" * 60, Colors.HEADER)

    try:
        # ================================================================
        # CONNECTION STATUS
        # ================================================================
        colored_print("\n🔌 Connection Status", Colors.HEADER)
        colored_print("=" * 60, Colors.HEADER)

        # Check Qdrant connection
        try:
            vector_db.client.get_collections()
            colored_print("✅ Qdrant:               Connected", Colors.OKGREEN)
            colored_print(
                f"   └─ URL:               {config.QDRANT_URL}", Colors.OKCYAN
            )
        except Exception:
            colored_print("❌ Qdrant:               Disconnected", Colors.FAIL)
            colored_print(
                f"   └─ URL:               {config.QDRANT_URL}", Colors.OKCYAN
            )
            colored_print("   └─ 💡 Make sure Qdrant is running", Colors.OKCYAN)
            return False

        # Check OpenAI connection
        if openai_handler and openai_handler.is_available():
            if openai_handler.test_connection():
                colored_print("✅ OpenAI:               Connected", Colors.OKGREEN)
                colored_print(
                    f"   └─ Model:             {config.OPENAI_MODEL}", Colors.OKCYAN
                )
            else:
                colored_print("❌ OpenAI:               Connection Failed", Colors.FAIL)
                colored_print(
                    f"   └─ Model:             {config.OPENAI_MODEL}", Colors.OKCYAN
                )
        else:
            colored_print("⚠️  OpenAI:               Not Configured", Colors.WARNING)
            colored_print(
                "   └─ 💡 Set OPENAI_API_KEY for conversational mode", Colors.OKCYAN
            )

        # Check MinIO connection
        if minio_handler:
            try:
                minio_handler.ensure_bucket_exists()
                colored_print("✅ MinIO:                Connected", Colors.OKGREEN)
                colored_print(
                    f"   └─ Endpoint:          {config.MINIO_ENDPOINT}", Colors.OKCYAN
                )
                colored_print(
                    f"   └─ Bucket:            {config.MINIO_BUCKET}", Colors.OKCYAN
                )
            except Exception as e:
                colored_print("❌ MinIO:                Connection Failed", Colors.FAIL)
                colored_print(
                    f"   └─ Endpoint:          {config.MINIO_ENDPOINT}", Colors.OKCYAN
                )
                colored_print(f"   └─ Error:             {e}", Colors.OKCYAN)
        else:
            colored_print("⚠️  MinIO:                Not Configured", Colors.WARNING)
            colored_print("   └─ 💡 Image storage will be disabled", Colors.OKCYAN)

        # ================================================================
        # COLLECTION STATUS
        # ================================================================
        colored_print("\n🗄️  Collection Status", Colors.HEADER)
        colored_print("=" * 60, Colors.HEADER)

        # Get collection info
        try:
            collection_info = vector_db.client.get_collection(config.COLLECTION_NAME)
            collection_exists = True
            colored_print("✅ Collection:           Ready", Colors.OKGREEN)
            colored_print(
                f"   ├─ Name:              {config.COLLECTION_NAME}", Colors.OKCYAN
            )
            colored_print(
                f"   ├─ Documents:         {collection_info.points_count:,}",
                Colors.OKCYAN,
            )
            colored_print(
                f"   ├─ Vector Size:       {config.VECTOR_SIZE}", Colors.OKCYAN
            )
            colored_print(
                f"   └─ Distance Metric:   {config.DISTANCE_METRIC}", Colors.OKCYAN
            )
        except Exception:
            collection_exists = False
            colored_print("❌ Collection:           Not Found", Colors.WARNING)
            colored_print(
                f"   ├─ Name:              {config.COLLECTION_NAME}", Colors.OKCYAN
            )
            colored_print("   ├─ Documents:         N/A", Colors.OKCYAN)
            colored_print(
                f"   ├─ Vector Size:       {config.VECTOR_SIZE}", Colors.OKCYAN
            )
            colored_print(
                f"   └─ Distance Metric:   {config.DISTANCE_METRIC}", Colors.OKCYAN
            )

        # ================================================================
        # MODEL CONFIGURATION
        # ================================================================
        colored_print("\n🤖 Model Configuration", Colors.HEADER)
        colored_print("=" * 60, Colors.HEADER)
        colored_print(f"├─ ColPali Model:        {config.MODEL_NAME}", Colors.OKCYAN)
        colored_print(f"├─ OpenAI Model:         {config.OPENAI_MODEL}", Colors.OKCYAN)
        colored_print(f"└─ Search Limit:         {config.SEARCH_LIMIT}", Colors.OKCYAN)

        # ================================================================
        # SYSTEM SUMMARY
        # ================================================================
        colored_print("\n📋 System Summary", Colors.HEADER)
        colored_print("=" * 60, Colors.HEADER)

        if collection_exists and collection_info.points_count > 0:
            colored_print("✅ Status:               Ready for searches", Colors.OKGREEN)
            colored_print("💡 Next Steps:", Colors.OKCYAN)
            colored_print("   └─ Use 'ask <query>' to search documents", Colors.OKCYAN)
            colored_print(
                "   └─ Use 'set-mode conversational' for AI responses", Colors.OKCYAN
            )
        elif collection_exists and collection_info.points_count == 0:
            colored_print("⚠️  Status:               Collection empty", Colors.WARNING)
            colored_print("💡 Next Steps:", Colors.OKCYAN)
            colored_print("   └─ Use 'upload <file>' to add documents", Colors.OKCYAN)
        else:
            colored_print("⚠️  Status:               Setup incomplete", Colors.WARNING)
            colored_print("💡 Next Steps:", Colors.OKCYAN)
            colored_print(
                "   └─ Use 'upload <file>' to create collection and add documents",
                Colors.OKCYAN,
            )

        return True

    except Exception as e:
        colored_print(f"❌ Error getting status: {e}", Colors.FAIL)
        colored_print("💡 Make sure Qdrant is running and accessible", Colors.OKCYAN)
        return False
