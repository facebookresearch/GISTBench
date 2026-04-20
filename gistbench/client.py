"""LLM client protocol and OpenAI-compatible backend."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Protocol

from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0


class LLMClient(Protocol):
    """Protocol for LLM inference backends."""

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send chat messages, return response text."""
        ...


class OpenAIClient:
    """OpenAI-compatible API client.

    Works with OpenAI, Azure OpenAI, vLLM, Ollama, and any server
    that implements the /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as e:
                last_err = e
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(
                    "API call failed (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, backoff,
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"API call failed after {MAX_RETRIES} retries"
        ) from last_err


def parse_json_response(text: str) -> list | dict | None:
    """Extract and parse JSON from an LLM response.

    Handles responses wrapped in ```json ... ``` blocks.
    Returns None if parsing fails.
    """
    # Try extracting from code block first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)

    # Try extracting array or object
    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if match:
        text = match.group(1)

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse JSON from LLM response")
        return None
