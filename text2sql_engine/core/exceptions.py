"""Domain exceptions for Text-to-SQL Engine."""

from __future__ import annotations


class Text2SQLError(Exception):
    """Base exception for all Text-to-SQL engine errors."""
    pass


class ConfigurationError(Text2SQLError):
    """Raised when engine or backend settings are invalid or missing required values."""
    pass


class ModelUnavailableError(Text2SQLError):
    """Raised when the requested LLM backend or model endpoint is unreachable or unavailable."""
    pass


class InferenceError(Text2SQLError):
    """Raised when an active LLM backend fails during prompt execution or inference."""
    pass


class PlanGenerationError(Text2SQLError):
    """Raised when the planner fails to generate or parse a valid LogicPlan from model output."""
    pass


class PlanValidationError(Text2SQLError):
    """Raised when a LogicPlan violates semantic integrity rules (e.g. duplicate CTE names)."""
    pass


class DependencyValidationError(PlanValidationError):
    """Raised when CTE dependencies, ordering, or column references cannot be resolved."""
    pass


class SQLCompilationError(Text2SQLError):
    """Raised when SQL AST parsing, fragment assembly, or transpilation fails."""
    pass


class UnsupportedDialectError(SQLCompilationError):
    """Raised when an unsupported target SQL dialect is requested."""
    pass


class SecurityValidationError(SQLCompilationError):
    """Raised when SQL violates statement security constraints (e.g. DDL, DML, or multiple statements)."""
    pass


class ExecutionError(Text2SQLError):
    """Raised when query execution on the target database engine fails."""
    pass


class RetrieverError(Text2SQLError):
    """Raised when schema or value retrieval fails or receives invalid metadata."""
    pass

