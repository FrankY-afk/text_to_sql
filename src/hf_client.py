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

# Suppress verbose third-party HTTP transport logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


class HuggingFaceInferenceError(Exception):
    """Raised when Hugging Face API inference fails."""


class HuggingFaceClient:
    """Client for generating SQL queries using Hugging Face models."""

    _circuit_broken: bool = False
    _circuit_broken_reason: str = ""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model_id = self.settings.llm_model
        self.token = self.settings.get_hf_token()
        self.endpoint_url = self.settings.hf_endpoint_url
        self.timeout = min(self.settings.hf_timeout_seconds, 10)
        self.temperature = self.settings.hf_temperature

        self._client: Optional[InferenceClient] = None

    @classmethod
    def is_circuit_broken(cls) -> bool:
        """Returns True if the circuit breaker has been tripped by billing/auth errors."""
        return cls._circuit_broken

    @classmethod
    def reset_circuit_breaker(cls) -> None:
        """Resets the circuit breaker to allow new API connection attempts."""
        cls._circuit_broken = False
        cls._circuit_broken_reason = ""

    @classmethod
    def trip_circuit_breaker(cls, reason: str) -> None:
        """Trips the circuit breaker to fail fast and protect against multi-second hangs."""
        cls._circuit_broken = True
        cls._circuit_broken_reason = reason
        logger.debug("Hugging Face API circuit breaker tripped: %s", reason)

    def is_configured(self) -> bool:
        """Returns True if a valid Hugging Face API key/token or endpoint URL is configured and circuit is intact."""
        if self._circuit_broken:
            return False
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
        if self._circuit_broken:
            raise HuggingFaceInferenceError(
                f"Hugging Face API circuit breaker active: {self._circuit_broken_reason}. "
                "Engaging deterministic CTE planner and execution-grounded fallback."
            )

        if not self.is_configured():
            raise HuggingFaceInferenceError(
                "Hugging Face API key is not configured or circuit breaker active. "
                "Set HF_API_KEY, HF_TOKEN, or TEXT2SQL_HF_API_TOKEN in your environment or .env file."
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
                logger.debug("Hugging Face inference attempt %d failed: %s", attempt + 1, exc)

                # Permanent billing/auth/permissions/unsupported model errors -> trip circuit breaker immediately
                if any(tok in err_str for tok in (
                    "402", "payment required", "pre-paid credits",
                    "401", "unauthorized", "invalid token",
                    "403", "forbidden",
                    "400", "bad request", "model_not_supported", "not supported by any provider",
                    "404", "not found",
                )):
                    HuggingFaceClient.trip_circuit_breaker(f"Permanent model/billing/auth error: {exc}")
                    break

                # If model is loading (503), wait and retry
                if "loading" in err_str or "503" in err_str:
                    time.sleep(2 * (attempt + 1))
                    continue
                break

        raise HuggingFaceInferenceError(
            f"Failed to generate SQL from Hugging Face model '{self.model_id}': {last_error}"
        ) from last_error

    @staticmethod
    def extract_sql(response_text: str) -> str:
        """Extracts clean SQL query from model response text, stripping markdown and commentary.

        Supports:
        - SQRL agentic protocol (<answer>...</answer> and <sql>...</sql> tags)
        - DeepSeek / Qwen reasoning blocks (<think>...</think> stripping)
        - Standard markdown code fences (```sql ... ```)
        - Raw SQL statements (WITH ...; SELECT ...;)
        """
        text = response_text.strip()

        # 1. Strip reasoning blocks if present (<think>...</think>)
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

        # 2. Extract from SQRL <answer>...</answer> action blocks
        answer_match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text, re.IGNORECASE)
        if answer_match:
            candidate = answer_match.group(1).strip()
            # If answer contains code fences inside, extract from within them
            code_fence = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", candidate, re.IGNORECASE)
            if code_fence:
                candidate = code_fence.group(1).strip()
            if candidate:
                return candidate.rstrip(";") + ";"

        # 3. Extract from SQRL <sql>...</sql> exploration blocks
        sql_tag_match = re.search(r"<sql>\s*([\s\S]*?)\s*</sql>", text, re.IGNORECASE)
        if sql_tag_match:
            candidate = sql_tag_match.group(1).strip()
            code_fence = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", candidate, re.IGNORECASE)
            if code_fence:
                candidate = code_fence.group(1).strip()
            if candidate:
                return candidate.rstrip(";") + ";"

        # 4. Match ```sql ... ``` or ``` ... ```
        sql_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if sql_match:
            candidate = sql_match.group(1).strip()
            if candidate:
                return candidate.rstrip(";") + ";"

        # 5. If starts with WITH or SELECT, find end of statement
        upper = text.upper()
        if upper.startswith("WITH ") or upper.startswith("SELECT "):
            # Find semicolon or return full text
            semi_idx = text.rfind(";")
            if semi_idx != -1:
                return text[: semi_idx + 1].strip()
            return text + ";"

        # 6. Fallback: look for WITH ... or SELECT ... block in text
        match_stmt = re.search(r"\b(WITH\s+[\s\S]+?;|SELECT\s+[\s\S]+?;)", text, re.IGNORECASE)
        if match_stmt:
            return match_stmt.group(1).strip()

        return text

