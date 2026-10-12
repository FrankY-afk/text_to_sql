"""Pydantic schemas for CTE Logic Planning and Dialect Function RAG."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class DialectFunctionSpec(BaseModel):
    """Specification for dialect-specific SQL functions and idioms."""

    name: str = Field(..., description="Function name or idiom (e.g., DATE_DIFF, QUALIFY, DATEDIFF)")
    dialect: str = Field(..., description="Target dialect: 'duckdb', 'postgres', 'bigquery', 'snowflake'")
    category: str = Field(..., description="Category: 'datetime', 'window', 'null_handling', 'aggregation', 'nested'")
    syntax_pattern: str = Field(..., description="Concrete usage syntax template")
    description: str = Field(..., description="Detailed description and semantics")
    notes: Optional[str] = Field(default=None, description="Important edge cases or compatibility caveats")


class CTEPlanStep(BaseModel):
    """Represents a single modular Common Table Expression (CTE) in the reasoning plan."""

    step_name: str = Field(..., description="Unique alphanumeric identifier for the CTE (e.g., filtered_orders)")
    purpose: str = Field(..., description="Human-readable purpose of this analytical step")
    source_tables: List[str] = Field(
        default_factory=list,
        description="Base tables or preceding CTEs referenced in this step",
    )
    math_logic: str = Field(
        ...,
        description="Formulas, aggregations, window math, and edge case guards (e.g., NULLIF(col, 0))",
    )
    filter_conditions: List[str] = Field(
        default_factory=list,
        description="Filter predicates applied at this step (WHERE clause clauses)",
    )
    group_by_columns: List[str] = Field(
        default_factory=list,
        description="Grouping dimensions (if aggregating)",
    )
    window_logic: Optional[str] = Field(
        default=None,
        description="Window function specifications (e.g., ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...))",
    )
    columns_produced: List[str] = Field(
        default_factory=list,
        description="List of column names or expressions produced by this CTE",
    )
    sql_fragment: Optional[str] = Field(
        default=None,
        description="Dialect-specific SQL SELECT statement defining this CTE (excluding the CTE name and AS)",
    )


class LogicPlan(BaseModel):
    """Structured multi-stage reasoning plan prior to final SQL generation."""

    query: str = Field(..., description="Original natural language query")
    target_dialect: str = Field(default="duckdb", description="Target SQL dialect: 'duckdb', 'postgres', 'bigquery', 'snowflake'")
    strategy: str = Field(..., description="High-level analytical plan and problem deconstruction")
    cte_steps: List[CTEPlanStep] = Field(
        default_factory=list,
        description="Ordered sequence of CTE steps",
    )
    final_select: str = Field(
        ...,
        description="Final SELECT projection and ordering pulling from the last CTE",
    )
    dialect_functions_used: List[DialectFunctionSpec] = Field(
        default_factory=list,
        description="List of dialect-specific functions and patterns applied in this plan",
    )
    validation_notes: Optional[str] = Field(
        default=None,
        description="Sanity check notes (e.g. division by zero protected, epoch converted to timestamp)",
    )
