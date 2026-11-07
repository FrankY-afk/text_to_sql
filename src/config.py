"""Application configuration management using Pydantic Settings.

Reads environment variables and local .env file securely, providing
type validation, sensible defaults, and masked credentials.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root path (where .env lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):   

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
        env_prefix="TEXT2SQL_",
        case_sensitive=False,
        extra="ignore",
    )
    postgres_host: str = Field(default="localhost", description="PostgreSQL host IP or domain")
    postgres_port: int = Field(default=5432, description="PostgreSQL TCP port")
    postgres_user: str = Field(default="postgres", description="Database username")
    postgres_password: Optional[SecretStr] = Field(default=None, description="Database user password")
    postgres_db: str = Field(default="lending_sample", description="Target database name")
    postgres_sslmode: str = Field(default="require", description="SSL mode (require, disable, prefer)")
    postgres_schema: str = Field(default="public", description="Default search schema")

    query_timeout_seconds: int = Field(default=60, description="Maximum execution timeout in seconds")
    max_row_limit: int = Field(default=100, description="Maximum rows returned by execute_readonly_sql")

    # Local Storage for Scratch & Ground-Truth Test Benchmarks
    local_db_path: str = Field(
        default=str(PROJECT_ROOT / "local_problem1.db"),
        description="Path to local database for local test benchmark tables",
    )

    # LLM & Hugging Face Configuration
    llm_backend: str = Field(
        default="huggingface",
        description="Selected LLM inference backend ('huggingface', 'ollama', 'mock')",
    )
    llm_model: str = Field(
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        description="Target Hugging Face model repository",
    )
    hf_api_token: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices(
            "hf_api_token",
            "hf_api_key",
            "hf_token",
            "text2sql_hf_api_token",
            "text2sql_hf_api_key",
            "text2sql_hf_token",
        ),
        description="Hugging Face API key / token (or read from HF_API_KEY / HF_TOKEN env)",
    )
    hf_endpoint_url: Optional[str] = Field(
        default=None,
        description="Optional custom Hugging Face Dedicated Inference Endpoint URL",
    )
    hf_timeout_seconds: int = Field(
        default=30,
        description="Hugging Face API request timeout in seconds",
    )
    hf_temperature: float = Field(
        default=0.0,
        description="Sampling temperature for LLM generation (0.0 for deterministic SQL)",
    )

    def get_hf_token(self) -> Optional[str]:
        """Resolves Hugging Face API key / token from settings or environment variables."""
        if self.hf_api_token:
            raw = self.hf_api_token.get_secret_value().strip()
            if raw and "placeholder" not in raw.lower() and raw.lower() not in (
                "none", "null", "your_hf_api_key_here", "your_hf_token_here"
            ):
                return raw
        for env_var in (
            "HF_API_KEY",
            "HF_TOKEN",
            "HUGGINGFACE_API_KEY",
            "HUGGINGFACEHUB_API_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "TEXT2SQL_HF_API_KEY",
            "TEXT2SQL_HF_API_TOKEN",
            "TEXT2SQL_HF_TOKEN",
        ):
            val = os.environ.get(env_var, "").strip()
            if val and "placeholder" not in val.lower() and val.lower() not in (
                "none", "null", "your_hf_api_key_here", "your_hf_token_here"
            ):
                return val
        return None


    def get_postgres_connection_dict(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "user": self.postgres_user,
            "dbname": self.postgres_db,
            "sslmode": self.postgres_sslmode,
            "connect_timeout": 15,
        }
        if self.postgres_password:
            kwargs["password"] = self.postgres_password.get_secret_value()
        return kwargs

    def get_masked_summary(self) -> Dict[str, Any]:
        return {
            "postgres_host": self.postgres_host,
            "postgres_port": self.postgres_port,
            "postgres_user": self.postgres_user,
            "postgres_db": self.postgres_db,
            "postgres_schema": self.postgres_schema,
            "postgres_sslmode": self.postgres_sslmode,
            "has_password": self.postgres_password is not None,
            "query_timeout_seconds": self.query_timeout_seconds,
            "max_row_limit": self.max_row_limit,
            "local_db_path": self.local_db_path,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    """Clears the settings cache (useful during testing)."""
    get_settings.cache_clear()
