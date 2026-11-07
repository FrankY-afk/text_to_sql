"""Hugging Face Inference Client for Text-to-SQL generation.

Connects to Hugging Face Inference API or custom dedicated endpoints using the
official huggingface_hub.InferenceClient.

Supports:
- Qwen/Qwen2.5-Coder-32B-Instruct and other state-of-the-art open SQL models.
- Clean extraction of SQL code blocks from markdown responses.
- Resilient retry on Hugging Face 503 (model loading) responses.
- Graceful offline fallback detection when no token or network is present.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional
from huggingface_hub import InferenceClient

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class HuggingFaceInferenceError(Exception):
    """Raised when Hugging Face API inference fails."""


class HuggingFaceClient:
    """Client for generating SQL queries using Hugging Face models."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model_id = self.settings.llm_model
        self.token = self.settings.get_hf_token()
        self.endpoint_url = self.settings.hf_endpoint_url
        self.timeout = self.settings.hf_timeout_seconds
        self.temperature = self.settings.hf_temperature

        self._client: Optional[InferenceClient] = None

    def is_configured(self) -> bool:
        """Returns True if a valid Hugging Face API key/token or endpoint URL is configured."""
        return bool(self.token or self.endpoint_url)

    def get_client(self) -> InferenceClient:
        """Initializes or returns cached InferenceClient instance."""
        if self._client is None:
            self._client = InferenceClient(
                model=self.endpoint_url or self.model_id,
                api_key=self.token,
                timeout=self.timeout,
            )
        return self._client

    def generate_sql(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Generates SQL using the configured Hugging Face model.

        Args:
            prompt: User/context prompt containing schema, CTE plan, and dialect rules.
            system_instruction: System prompt framing the role.
            max_tokens: Maximum tokens in response.

        Returns:
            Extracted clean SQL string.

        Raises:
            HuggingFaceInferenceError: If inference fails or no API key is configured.
        """
        if not self.is_configured():
            raise HuggingFaceInferenceError(
                "Hugging Face API key is not configured. Set HF_API_KEY, HF_TOKEN, or TEXT2SQL_HF_API_TOKEN in your environment or .env file."
            )

        client = self.get_client()
        sys_msg = system_instruction or (
            "You are an expert SQL engineer. Your task is to generate dialect-correct, "
            "semantically accurate SQL queries based strictly on the provided CTE analytical plan, "
            "database schema, and dialect rules. Output only the SQL query inside ```sql code fences."
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                # Attempt chat completion
                completion = client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                )
                raw_response = completion.choices[0].message.content or ""
                return self.extract_sql(raw_response)
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                logger.warning("Hugging Face inference attempt %d failed: %s", attempt + 1, exc)
                # If model is loading (503), wait and retry
                if "loading" in err_str or "503" in err_str:
                    time.sleep(4 * (attempt + 1))
                    continue
                break

        raise HuggingFaceInferenceError(
            f"Failed to generate SQL from Hugging Face model '{self.model_id}': {last_error}"
        ) from last_error

    @staticmethod
    def extract_sql(response_text: str) -> str:
        """Extracts clean SQL query from model response text, stripping markdown and commentary."""
        text = response_text.strip()

        # Match ```sql ... ``` or ``` ... ```
        sql_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if sql_match:
            candidate = sql_match.group(1).strip()
            if candidate:
                return candidate.rstrip(";") + ";"

        # If starts with WITH or SELECT, find end of statement
        upper = text.upper()
        if upper.startswith("WITH ") or upper.startswith("SELECT "):
            # Find semicolon or return full text
            semi_idx = text.rfind(";")
            if semi_idx != -1:
                return text[: semi_idx + 1].strip()
            return text + ";"

        # Fallback: look for WITH ... or SELECT ... block in text
        match_stmt = re.search(r"\b(WITH\s+[\s\S]+?;|SELECT\s+[\s\S]+?;)", text, re.IGNORECASE)
        if match_stmt:
            return match_stmt.group(1).strip()

        return text

