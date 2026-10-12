"""Core configuration, exceptions, and state definitions."""

from text2sql_engine.core.config import (
    EngineSettings,
    get_settings,
    get_embeddings,
    get_chroma_client,
    get_vector_store,
    get_llm,
    DeterministicMockCoder,
)
from text2sql_engine.core.state import SQLAgentState
from text2sql_engine.core.exceptions import (
    Text2SQLError,
    ConfigurationError,
    ModelUnavailableError,
    InferenceError,
    PlanGenerationError,
    PlanValidationError,
    DependencyValidationError,
    SQLCompilationError,
    UnsupportedDialectError,
    SecurityValidationError,
)

__all__ = [
    "EngineSettings",
    "get_settings",
    "get_embeddings",
    "get_chroma_client",
    "get_vector_store",
    "get_llm",
    "DeterministicMockCoder",
    "SQLAgentState",
    "Text2SQLError",
    "ConfigurationError",
    "ModelUnavailableError",
    "InferenceError",
    "PlanGenerationError",
    "PlanValidationError",
    "DependencyValidationError",
    "SQLCompilationError",
    "UnsupportedDialectError",
    "SecurityValidationError",
]
