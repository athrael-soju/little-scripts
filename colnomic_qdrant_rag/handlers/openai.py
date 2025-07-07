import base64
import io
from typing import List, Optional

import config
import requests
from openai import OpenAI
from PIL import Image


class OpenAIHandler:
    """Handles OpenAI API interactions for image analysis and response generation."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI handler with API key."""
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or config.OPENAI_API_KEY"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = config.OPENAI_MODEL
        self.max_tokens = config.OPENAI_MAX_TOKENS
        self.temperature = config.OPENAI_TEMPERATURE

    def encode_image_to_base64(self, image) -> str:
        """Convert PIL Image or image bytes to base64 string."""
        if isinstance(image, Image.Image):
            # PIL Image
            buffer = io.BytesIO()
            save_kwargs = {"format": config.IMAGE_FORMAT}
            if config.IMAGE_FORMAT.upper() == "JPEG":
                save_kwargs["quality"] = config.IMAGE_QUALITY
            image.save(buffer, **save_kwargs)
            image_bytes = buffer.getvalue()
        elif hasattr(image, "read"):
            # File-like object
            image_bytes = image.read()
        elif isinstance(image, bytes):
            # Raw bytes
            image_bytes = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return base64.b64encode(image_bytes).decode("utf-8")

    def _get_image_mime_type(self) -> str:
        """Get the appropriate MIME type based on the configured image format."""
        if config.IMAGE_FORMAT.upper() == "PNG":
            return "image/png"
        elif config.IMAGE_FORMAT.upper() == "JPEG":
            return "image/jpeg"
        else:
            # Default to PNG if format is not recognized
            return "image/png"

    def download_image_from_url(self, url: str) -> bytes:
        """Download image from URL and return as bytes."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            raise ValueError(f"Failed to download image from {url}: {e}")

    def analyze_images(
        self,
        images: List,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ):
        """
        Analyze multiple images with OpenAI Vision API.

        Args:
            images: List of PIL Images, image URLs, or image bytes
            user_prompt: User's question/prompt about the images
            system_prompt: Optional system prompt to guide the AI's behavior
            stream: If True, returns a generator for streaming the response.

        Returns:
            OpenAI's response as a string or a generator if stream=True.
        """
        try:
            # Prepare messages
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Prepare content with images
            content = [{"type": "text", "text": user_prompt}]

            # Process each image
            for i, image in enumerate(images):
                try:
                    if isinstance(image, str) and (
                        image.startswith("http://") or image.startswith("https://")
                    ):
                        # URL - download it first, then encode to base64
                        image_bytes = self.download_image_from_url(image)
                        base64_image = self.encode_image_to_base64(image_bytes)
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_image_mime_type()};base64,{base64_image}"
                                },
                            }
                        )
                    else:
                        # Convert other formats (PIL, bytes) to base64
                        base64_image = self.encode_image_to_base64(image)
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{self._get_image_mime_type()};base64,{base64_image}"
                                },
                            }
                        )
                except Exception as e:
                    print(f"Warning: Failed to process image {i + 1}: {e}")
                    continue

            messages.append({"role": "user", "content": content})

            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=stream,
            )

            if stream:

                def stream_generator():
                    for chunk in response:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content

                return stream_generator()
            else:
                return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    def analyze_single_image(
        self, image, user_prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Analyze a single image with OpenAI Vision API."""
        return self.analyze_images([image], user_prompt, system_prompt)

    def is_available(self) -> bool:
        """Check if OpenAI API is available and configured."""
        try:
            return bool(self.api_key)
        except Exception:
            return False

    def test_connection(self) -> bool:
        """Test OpenAI API connection."""
        try:
            # Simple test call using the configured model
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            print(f"OpenAI connection test failed: {e}")
            return False
