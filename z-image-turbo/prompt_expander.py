"""Prompt expansion and enhancement for Z-Image-Turbo application."""

import os
import json
import re
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from pe import SYSTEM_PROMPT_TEMPLATE


@dataclass
class PromptOutput:
    """Output structure for prompt expansion results."""

    status: str
    prompt: str
    seed: Optional[int] = None
    system_prompt: Optional[str] = None
    messages: Optional[list] = None


class PromptExpander:
    """Base class for prompt expansion functionality."""

    def __init__(self, backend: str = "api"):
        """
        Initialize the prompt expander.

        Args:
            backend (str): Backend type to use for expansion
        """
        self.backend = backend

    def get_system_prompt(self, original_prompt: str = "") -> str:
        """
        Get the system prompt for enhancement.

        Args:
            original_prompt (str): Original user prompt

        Returns:
            str: System prompt for the API
        """
        if "{prompt}" in SYSTEM_PROMPT_TEMPLATE:
            return SYSTEM_PROMPT_TEMPLATE.format(prompt=original_prompt)
        return SYSTEM_PROMPT_TEMPLATE

    def extend(self, prompt: str) -> PromptOutput:
        """
        Extend/enhance a prompt. Override in subclasses.

        Args:
            prompt (str): Original prompt to enhance

        Returns:
            PromptOutput: Enhanced prompt result
        """
        raise NotImplementedError("Subclasses must implement extend()")


class APIPromptExpander(PromptExpander):
    """API-based prompt expander using OpenAI-compatible endpoints."""

    def __init__(
        self,
        backend: str = "api",
        model: str = "qwen3-max-preview",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the API prompt expander.

        Args:
            backend (str): Backend type
            model (str): Model name to use
            api_key (str, optional): API key (defaults to DASHSCOPE_API_KEY env var)
            base_url (str, optional): API base URL
        """
        super().__init__(backend)
        self.model = model

        # Default to DashScope API
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None
            print("Warning: No API key found. Prompt enhancement will be disabled.")

    def extend(self, prompt: str) -> PromptOutput:
        """
        Extend/enhance a prompt using the API.

        Args:
            prompt (str): Original prompt to enhance

        Returns:
            PromptOutput: Enhanced prompt result
        """
        if not self.client:
            return PromptOutput(
                status="error",
                prompt=prompt,
                messages=["API client not initialized. Check API key."],
            )

        system_prompt = self.get_system_prompt(prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                top_p=0.9,
            )

            content = response.choices[0].message.content

            # Try to extract JSON from the response
            enhanced_prompt = self._extract_prompt(content, prompt)

            return PromptOutput(
                status="success",
                prompt=enhanced_prompt,
                system_prompt=system_prompt,
            )

        except Exception as e:
            return PromptOutput(
                status="error",
                prompt=prompt,
                messages=[f"API error: {str(e)}"],
            )

    def _extract_prompt(self, content: str, fallback: str) -> str:
        """
        Extract enhanced prompt from API response.

        Args:
            content (str): API response content
            fallback (str): Fallback prompt if extraction fails

        Returns:
            str: Extracted or fallback prompt
        """
        # Try to extract JSON
        try:
            # Look for JSON in markdown code blocks
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                if "prompt" in data:
                    return data["prompt"]

            # Try direct JSON parsing
            data = json.loads(content)
            if "prompt" in data:
                return data["prompt"]

        except (json.JSONDecodeError, AttributeError):
            pass

        # Return content directly if it looks like a prompt
        if content and len(content) > 10:
            return content.strip()

        return fallback


def create_prompt_expander(
    backend: str = "api", model: str = "qwen3-max-preview"
) -> PromptExpander:
    """
    Factory function to create a prompt expander.

    Args:
        backend (str): Backend type ("api" supported)
        model (str): Model name to use

    Returns:
        PromptExpander: Configured prompt expander instance
    """
    if backend == "api":
        return APIPromptExpander(backend=backend, model=model)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def prompt_enhance(
    prompt: str, enable: bool, expander: Optional[PromptExpander] = None
) -> tuple[str, str]:
    """
    Enhance a prompt if enabled.

    Args:
        prompt (str): Original prompt
        enable (bool): Whether to enable enhancement
        expander (PromptExpander, optional): Expander instance to use

    Returns:
        tuple: (enhanced_prompt, status_message)
    """
    if not enable:
        return prompt, "Enhancement disabled"

    if not prompt or not prompt.strip():
        return prompt, "Empty prompt"

    if not expander:
        return prompt, "No expander available"

    result = expander.extend(prompt)

    if result.status == "success":
        return result.prompt, "Enhanced successfully"
    else:
        messages = result.messages or ["Unknown error"]
        return result.prompt, f"Enhancement failed: {messages[0]}"
