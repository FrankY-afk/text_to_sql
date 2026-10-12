"""Pydantic v2 schemas for CTE Logic Planning and Dialect Function RAG."""

from __future__ import annotations

import re
from typing import List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SUPPORTED_DIALECTS: Set[str] = {
    "duckdb",
    "postgres",
    "bigquery",
    "snowflake",
    "sqlite",
}


def normalize_dialect(raw_dialect: str) -> str:
    """Normalizes and validates SQL dialect against allowlist."""
    norm = raw_dialect.lower().strip()
    if norm == "postgresql":
        norm = "postgres"
    if norm not in SUPPORTED_DIALECTS:
        raise ValueError(
            f"Unsupported SQL dialect '{raw_dialect}'. Supported dialects: {sorted(SUPPORTED_DIALECTS)}"
        )
    return norm


class DialectFunctionSpec(BaseModel):
    """Specification for dialect-specific SQL functions and idioms."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Function name or idiom (e.g., DATE_DIFF, QUALIFY, DATEDIFF)")
    dialect: str = Field(..., description="Target dialect: 'duckdb', 'postgres', 'bigquery', 'snowflake', 'sqlite'")
    category: str = Field(..., min_length=1, description="Category: 'datetime', 'window', 'null_handling', 'aggregation', 'nested'")
    syntax_pattern: str = Field(..., min_length=1, description="Concrete usage syntax template")
    description: str = Field(..., min_length=1, description="Detailed description and semantics")
    notes: Optional[str] = Field(default=None, description="Important edge cases or compatibility caveats")

    @field_validator("name", "category", "syntax_pattern", "description")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Field cannot be empty or whitespace only.")
        return clean

    @field_validator("dialect")
    @classmethod
    def validate_dialect(cls, v: str) -> str:
        return normalize_dialect(v)


class CTEPlanStep(BaseModel):
    """Represents a single modular Common Table Expression (CTE) in the reasoning plan."""

    model_config = ConfigDict(extra="forbid")

    step_name: str = Field(..., min_length=1, description="Unique identifier for the CTE (e.g., filtered_orders)")
    purpose: str = Field(..., min_length=1, description="Human-readable purpose of this analytical step")
    source_tables: List[str] = Field(
        default_factory=list,
        description="Base tables or preceding CTEs referenced in this step",
    )
    math_logic: str = Field(
        ...,
        min_length=1,
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

    @field_validator("step_name")
    @classmethod
    def validate_step_name(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("step_name cannot be empty.")
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", clean):
            raise ValueError(
                f"Invalid CTE step_name '{v}'. Must be a valid SQL identifier (alphanumeric/underscore, not starting with digit)."
            )
        return clean

    @field_validator("purpose", "math_logic")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Field cannot be empty or whitespace only.")
        return clean

    @field_validator("sql_fragment")
    @classmethod
    def validate_sql_fragment(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            clean = v.strip()
            if not clean:
                raise ValueError("sql_fragment, if specified, cannot be empty or whitespace only.")
            return clean
        return None


class LogicPlan(BaseModel):
    """Structured multi-stage reasoning plan prior to final SQL generation."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Original natural language query")
    target_dialect: str = Field(default="duckdb", description="Target SQL dialect: 'duckdb', 'postgres', 'bigquery', 'snowflake', 'sqlite'")
    strategy: str = Field(..., min_length=1, description="High-level analytical plan and problem deconstruction")
    cte_steps: List[CTEPlanStep] = Field(
        default_factory=list,
        description="Ordered sequence of CTE steps",
    )
    final_select: str = Field(
        ...,
        min_length=1,
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
    planner_backend: Optional[str] = Field(
        default=None,
        description="Backend identifier used for generation ('real_llm', 'mock', etc.)",
    )

    @field_validator("target_dialect")
    @classmethod
    def validate_dialect(cls, v: str) -> str:
        return normalize_dialect(v)

    @field_validator("query", "strategy", "final_select")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Field cannot be empty or whitespace only.")
        return clean

    @model_validator(mode="after")
    def validate_plan_integrity(self) -> LogicPlan:
        # 1. Reject duplicate CTE names
        cte_names = [step.step_name.lower() for step in self.cte_steps]
        seen = set()
        duplicates = []
        for name in cte_names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(f"Duplicate CTE names detected in plan: {duplicates}")

        # 2. Validate that final_select contains a SELECT statement
        if "select" not in self.final_select.lower():
            raise ValueError("final_select must contain a valid SELECT projection.")

        # 3. Ensure all CTE steps have a non-empty sql_fragment
        for step in self.cte_steps:
            if not step.sql_fragment or not step.sql_fragment.strip():
                raise ValueError(
                    f"CTE step '{step.step_name}' has missing or empty sql_fragment."
                )

        return self
