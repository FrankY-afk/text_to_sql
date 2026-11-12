"""Application configuration management using Pydantic Settings.

Reads environment variables and local .env file securely, providing
type validation, sensible defaults, and masked credentials.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
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

    # Embeddings & Vector Store Configuration (Problem 2: Enterprise Schema Linking)
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="BGE embedding model identifier for table, column, and literal retrieval",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for embedding execution ('cpu' or 'cuda')",
    )
    chroma_db_dir: str = Field(
        default=str(PROJECT_ROOT / "chroma_db"),
        description="Persistent directory for ChromaDB vector indices",
    )
    schema_top_k_tables: int = Field(
        default=5,
        description="Default top-k candidate tables returned in Stage 1 Coarse Retrieval",
    )
    schema_top_k_columns: int = Field(
        default=8,
        description="Default top-k relevant columns returned per table in Stage 2 Retrieval",
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
            "embedding_model_name": self.embedding_model_name,
            "chroma_db_dir": self.chroma_db_dir,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    """Clears the settings cache (useful during testing)."""
    get_settings.cache_clear()


class DeterministicTokenEmbeddingFunction:
    """Lightweight, deterministic token+character-ngram Chroma EmbeddingFunction.

    Used as a fast zero-network fallback or unit-test embedder when SentenceTransformer
    model weights cannot be loaded or when sub-millisecond unit test execution is desired.
    Produces L2-normalized dense 384-d float vectors compatible with cosine similarity.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def __call__(self, input: List[str]) -> List[List[float]]:  # type: ignore[override]
        import hashlib
        import math
        import re

        vectors: List[List[float]] = []
        for text in input:
            vec = [0.0] * self.dim
            lowered = str(text).lower()
            tokens = re.findall(r"[a-z0-9]+", lowered)
            # Split camelCase / snake_case sub-tokens plus bigrams & char trigrams
            sub_tokens: List[str] = list(tokens)
            for tok in tokens:
                if len(tok) >= 3:
                    for i in range(len(tok) - 2):
                        sub_tokens.append(f"#{tok[i:i+3]}")
            for idx in range(len(tokens) - 1):
                sub_tokens.append(f"{tokens[idx]}_{tokens[idx+1]}")

            for feat in sub_tokens:
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                bucket = h % self.dim
                sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
                weight = 1.0 if not feat.startswith("#") else 0.35
                vec[bucket] += sign * weight

            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0.0:
                vec = [v / norm for v in vec]
            else:
                vec[0] = 1.0
            vectors.append(vec)
        return vectors

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "deterministic_token_hash"

    @staticmethod
    def is_legacy() -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> Dict[str, Any]:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "DeterministicTokenEmbeddingFunction":
        return DeterministicTokenEmbeddingFunction(dim=int(config.get("dim", 384)))


_bge_embedding_cache: Dict[str, Any] = {}


def get_embedding_function(
    settings: Optional[Settings] = None,
    backend: str = "auto",
) -> Any:
    """Returns a ChromaDB-compatible EmbeddingFunction using the configured BGE model.

    Args:
        settings: Optional Settings instance.
        backend: 'bge' (force BAAI/bge-small-en-v1.5), 'hash' (fast deterministic),
                 or 'auto' (try BGE local cache / download, fall back to hash if offline).
    """
    cfg = settings or get_settings()
    mode = backend.lower().strip()

    if mode == "hash":
        return DeterministicTokenEmbeddingFunction()

    cache_key = f"{cfg.embedding_model_name}:{cfg.embedding_device}"
    if cache_key in _bge_embedding_cache:
        return _bge_embedding_cache[cache_key]

    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        # Prefer local cached model first to prevent network stalls when already cached
        try:
            ef = SentenceTransformerEmbeddingFunction(
                model_name=cfg.embedding_model_name,
                device=cfg.embedding_device,
                normalize_embeddings=True,
                local_files_only=True,
            )
        except Exception:
            if mode == "bge":
                ef = SentenceTransformerEmbeddingFunction(
                    model_name=cfg.embedding_model_name,
                    device=cfg.embedding_device,
                    normalize_embeddings=True,
                )
            else:
                raise
        _bge_embedding_cache[cache_key] = ef
        return ef
    except Exception as exc:
        if mode == "bge":
            raise RuntimeError(f"Failed to load BGE embedding model '{cfg.embedding_model_name}': {exc}") from exc
        return DeterministicTokenEmbeddingFunction()


def get_chroma_client(
    persist_directory: Optional[str] = None,
    in_memory: bool = False,
    settings: Optional[Settings] = None,
) -> Any:
    """Initializes and returns a ChromaDB PersistentClient or EphemeralClient."""
    import chromadb

    if in_memory:
        return chromadb.EphemeralClient()
    cfg = settings or get_settings()
    target_dir = persist_directory or cfg.chroma_db_dir
    os.makedirs(target_dir, exist_ok=True)
    return chromadb.PersistentClient(path=target_dir)
