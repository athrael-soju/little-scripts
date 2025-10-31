"""
PaddleOCR-VL service service for document OCR processing.
"""

import json
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

_CONTENT_BLOCK_PATTERN = re.compile(
    r"label:\s*(\w+)\s*\n\s*bbox:\s*\[([^\]]+)\]\s*\n\s*content:\s*(.+?)(?=\n#####|$)",
    re.DOTALL,
)


@dataclass
class DocumentElement:
    """Structured representation of a single OCR element."""

    index: int
    content: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict representation compatible with the API models."""
        return {
            "index": self.index,
            "content": self.content,
            "metadata": self.metadata,
        }


class PaddleOCRVLService:
    """
    Thread-safe service for the PaddleOCR-VL pipeline.

    The pipeline is loaded lazily on first use so that application startup stays
    lightweight. A custom pipeline factory can be supplied to simplify testing.
    """

    def __init__(self, pipeline_factory: Optional[Callable[[], Any]] = None) -> None:
        self._pipeline_factory = pipeline_factory or self._default_pipeline_factory
        self._pipeline: Optional[Any] = None
        self._pipeline_lock = threading.Lock()

    def _ensure_pipeline(self) -> Any:
        """Initialise the PaddleOCR-VL pipeline on demand."""
        if self._pipeline is not None:
            return self._pipeline

        with self._pipeline_lock:
            if self._pipeline is not None:
                return self._pipeline

            logger.info("Initializing PaddleOCR-VL pipeline...")
            start_time = time.time()

            try:
                pipeline = self._pipeline_factory()
            except Exception as exc:
                logger.error(
                    "Failed to initialize PaddleOCR-VL pipeline: %s", exc, exc_info=True
                )
                raise RuntimeError(
                    f"PaddleOCR-VL initialization failed: {exc}"
                ) from exc

            self._pipeline = pipeline
            elapsed = time.time() - start_time
            logger.info(
                "PaddleOCR-VL pipeline initialized successfully in %.2fs", elapsed
            )
            logger.info("GPU support: %s", settings.use_gpu)

        return self._pipeline

    @staticmethod
    def _default_pipeline_factory() -> Any:
        """Default factory used to create a PaddleOCR-VL pipeline instance."""
        from paddleocr import PaddleOCRVL

        return PaddleOCRVL()

    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process an image file and extract document structure.

        Args:
            image_path: Path to the image file.

        Returns:
            List of OCR results with document structure.
        """
        pipeline = self._ensure_pipeline()
        logger.info("Processing image: %s", image_path)
        start_time = time.time()

        try:
            predictions = pipeline.predict(image_path)
        except Exception as exc:
            logger.error("Error processing image: %s", exc, exc_info=True)
            raise RuntimeError(f"Image processing failed: {exc}") from exc

        elements = [
            self._serialize_prediction(index, prediction)
            for index, prediction in enumerate(predictions)
        ]

        elapsed = time.time() - start_time
        logger.info(
            "Image processed successfully in %.2fs - Found %d elements",
            elapsed,
            len(elements),
        )

        return [element.to_dict() for element in elements]

    def process_image_bytes(
        self, image_bytes: bytes, filename: str = "image.jpg"
    ) -> List[Dict[str, Any]]:
        """
        Process image from bytes.

        Args:
            image_bytes: Image file bytes.
            filename: Original filename (for extension detection).
        """
        if not image_bytes:
            raise ValueError("Cannot process empty image bytes")

        with self._temporary_image(image_bytes, filename) as temp_path:
            return self.process_image(str(temp_path))

    @contextmanager
    def _temporary_image(self, image_bytes: bytes, filename: str):
        """Write bytes to a temporary file and clean it up afterwards."""
        suffix = Path(filename or "image.jpg").suffix or ".jpg"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)

        try:
            yield temp_path
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Failed to delete temporary file %s: %s", temp_path, exc)

    def _serialize_prediction(self, index: int, prediction: Any) -> DocumentElement:
        """
        Convert a PaddleOCR-VL prediction to a DocumentElement.
        """
        content = self._extract_content(prediction)
        metadata = self._extract_metadata(prediction)
        return DocumentElement(index=index, content=content, metadata=metadata)

    def _extract_content(self, result: Any) -> Dict[str, Any]:
        """
        Extract content from a PaddleOCR-VL result object.
        """
        try:
            result_str = str(result)
        except Exception as exc:
            logger.warning("Failed to convert result to string: %s", exc)
            return {"raw": repr(result)}

        content: Dict[str, Any] = {}
        matches = _CONTENT_BLOCK_PATTERN.findall(result_str)

        if matches:
            blocks = []
            for label, bbox_str, text in matches:
                bbox: List[float] = []
                for value in bbox_str.split(","):
                    cleaned = value.strip().replace("np.float32(", "").replace(")", "")
                    try:
                        bbox.append(float(cleaned))
                    except ValueError:
                        continue

                blocks.append(
                    {
                        "type": label.strip(),
                        "bbox": bbox,
                        "text": text.strip(),
                    }
                )

            if blocks:
                content["blocks"] = blocks
                content["text"] = "\n\n".join(
                    block["text"] for block in blocks if block.get("text")
                )

        if hasattr(result, "to_markdown") and callable(result.to_markdown):
            try:
                content["markdown"] = result.to_markdown()
            except Exception as exc:
                logger.debug("Failed to call to_markdown(): %s", exc)

        if content:
            return content

        return {"raw": result_str}

    def _extract_metadata(self, result: Any) -> Dict[str, Any]:
        """
        Extract metadata from a PaddleOCR-VL result object.
        """
        metadata: Dict[str, Any] = {}

        for attr in ("bbox", "confidence", "type", "label"):
            if hasattr(result, attr):
                value = getattr(result, attr)
                serialized_value = self._make_serializable(value)
                if serialized_value is not None:
                    metadata[attr] = serialized_value

        return metadata

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Convert any object to a JSON-serializable format."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, (list, tuple)):
            return [PaddleOCRVLService._make_serializable(item) for item in obj]

        if isinstance(obj, dict):
            return {
                key: PaddleOCRVLService._make_serializable(value)
                for key, value in obj.items()
                if not callable(value)
            }

        if callable(obj):
            return None

        if hasattr(obj, "__dict__"):
            try:
                return {
                    key: PaddleOCRVLService._make_serializable(value)
                    for key, value in obj.__dict__.items()
                    if not key.startswith("_") and not callable(value)
                }
            except Exception:
                pass

        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    def get_markdown_output(self, results: List[Dict[str, Any]]) -> str:
        """
        Convert results to markdown format.
        """
        lines: List[str] = ["# Document OCR Results"]

        for index, result in enumerate(results, start=1):
            lines.append("")
            lines.append(f"## Element {index}")

            content = result.get("content", {})
            if isinstance(content, dict):
                for key, value in content.items():
                    lines.append(f"**{key}**: {value}")
            else:
                lines.append(str(content))

        return "\n".join(lines)

    def is_ready(self) -> bool:
        """Check if the pipeline is initialized and ready."""
        return self._pipeline is not None

    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "initialized": self.is_ready(),
            "gpu_enabled": settings.use_gpu,
            "device": settings.device,
        }


# Global service instance
paddleocr_vl_service = PaddleOCRVLService()
