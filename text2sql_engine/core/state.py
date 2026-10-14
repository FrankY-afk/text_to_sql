"""LangGraph State definition for Text-to-SQL Agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SQLAgentState(TypedDict, total=False):
    """Represents the complete state passed through the LangGraph text-to-sql pipeline."""

    # Input specifications
    user_query: str
    target_dialect: str  # duckdb, postgres, bigquery, snowflake

    # Problem 2 (Schema Retrieval & Disambiguation) outputs
    candidate_tables: List[str]
    flattened_schema: List[Dict[str, Any]]
    filtered_schema_context: str
    matched_literals: Dict[str, str]  # e.g. {"California": "state_code = 'CA'"}

    # Dialect Function RAG matches
    dialect_functions: List[Dict[str, str]]

    # Problem 1 (CTE Logic Planning & SQL Compilation) outputs
    logic_plan: Optional[Dict[str, Any]]
    generated_sql: Optional[str]

    # Validation & Database Execution outputs
    ast_valid: bool
    execution_status: str  # COMPILED, EXECUTED, FAILED, UNSUPPORTED_EXECUTION_BACKEND
    execution_result: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    failed_stage: Optional[str]  # retrieval, planning, compilation, execution
    error_category: Optional[str]  # validation, dependency, syntax, dialect, runtime, configuration
    repair_notes: Optional[str]
    retry_count: int
    max_retries: int

    # LangGraph conversation history
    messages: Annotated[Sequence[BaseMessage], add_messages]
