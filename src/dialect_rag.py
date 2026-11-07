"""Dialect Function RAG (Problem 1, Mechanism B).

Retrieves database-specific SQL function documentation, signatures, argument ordering,
and syntax caveats across PostgreSQL, BigQuery, Snowflake, DuckDB, and SQLite.

Prevents common Spider 2.0 dialect failure modes:
1. Argument order inversion (e.g., BigQuery DATE_DIFF(end, start, unit) vs DuckDB date_diff(unit, start, end)).
2. Dialect function absence (e.g., PostgreSQL has no generic DATEDIFF; requires date subtraction or AGE()).
3. Unsupported clauses (e.g., QUALIFY clause unsupported in PostgreSQL/SQLite; requires CTE with ROW_NUMBER()).
4. Safe division idioms (NULLIF(denom, 0) in Postgres/DuckDB vs IEEE_DIVIDE in BigQuery vs DIV0 in Snowflake).
5. String aggregation differences (string_agg in Postgres/DuckDB vs STRING_AGG in BigQuery vs LISTAGG in Snowflake).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, field_validator
import sqlglot
from sqlglot import exp

from src.cte_planner import SUPPORTED_DIALECTS, normalize_dialect

logger = logging.getLogger(__name__)


class DialectFunctionSpec(BaseModel):
    """Specification for dialect-specific SQL functions, idioms, and syntax rules."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1, description="Function name or idiom (e.g. 'DATE_DIFF', 'NULLIF', 'QUALIFY')")
    dialect: str = Field(..., description="Target dialect: 'postgres', 'duckdb', 'bigquery', 'snowflake', 'sqlite'")
    category: str = Field(..., min_length=1, description="Category: 'datetime', 'window', 'null_handling', 'aggregation', 'math'")
    syntax_pattern: str = Field(..., min_length=1, description="Concrete usage syntax template")
    argument_types: List[str] = Field(default_factory=list, description="Expected argument types")
    return_type: str = Field(default="", description="Function return type")
    description: str = Field(..., min_length=1, description="Detailed description and semantics")
    caveats_and_pitfalls: Optional[str] = Field(
        default=None,
        description="Important dialect differences, argument orders, or compatibility caveats",
    )
    unsupported_in_dialects: List[str] = Field(
        default_factory=list,
        description="Dialects where this function/idiom is completely unsupported or invalid",
    )

    @field_validator("dialect")
    @classmethod
    def validate_dialect(cls, v: str) -> str:
        return normalize_dialect(v)


# Comprehensive knowledge catalog covering Spider 2.0 dialects
DIALECT_CATALOG: List[DialectFunctionSpec] = [
    # --------------------------------------------------------------------------
    # Date Difference / Duration
    # --------------------------------------------------------------------------
    DialectFunctionSpec(
        name="date_subtraction_postgres",
        dialect="postgres",
        category="datetime",
        syntax_pattern="({end_date}::date - {start_date}::date)",
        argument_types=["date", "date"],
        return_type="integer (days)",
        description="Calculates day difference between two dates in PostgreSQL via date subtraction.",
        caveats_and_pitfalls=(
            "PostgreSQL does NOT have a built-in DATEDIFF() function. Subtracting two DATE values "
            "directly yields the integer difference in days: (end_date::date - start_date::date). "
            "For timestamps or intervals, use EXTRACT(DAY FROM end_ts - start_ts) or AGE(end_ts, start_ts)."
        ),
        unsupported_in_dialects=[],
    ),
    DialectFunctionSpec(
        name="date_diff_duckdb",
        dialect="duckdb",
        category="datetime",
        syntax_pattern="date_diff('{unit}', {start_date}, {end_date})",
        argument_types=["string (unit)", "date/timestamp", "date/timestamp"],
        return_type="bigint",
        description="Calculates date difference in DuckDB. Unit is first argument string ('day', 'month', 'year').",
        caveats_and_pitfalls=(
            "DuckDB requires the unit string as the FIRST argument: date_diff('day', start_date, end_date). "
            "Start date precedes end date."
        ),
        unsupported_in_dialects=[],
    ),
    DialectFunctionSpec(
        name="DATE_DIFF_bigquery",
        dialect="bigquery",
        category="datetime",
        syntax_pattern="DATE_DIFF({end_date}, {start_date}, {unit})",
        argument_types=["date", "date", "date_part"],
        return_type="int64",
        description="Calculates date difference in BigQuery.",
        caveats_and_pitfalls=(
            "BigQuery places the END DATE first and unit LAST: DATE_DIFF(end_date, start_date, DAY). "
            "Reversing end_date and start_date inverts the sign of the result!"
        ),
        unsupported_in_dialects=["postgres", "sqlite"],
    ),
    DialectFunctionSpec(
        name="DATEDIFF_snowflake",
        dialect="snowflake",
        category="datetime",
        syntax_pattern="DATEDIFF({unit}, {start_date}, {end_date})",
        argument_types=["date_part", "date/timestamp", "date/timestamp"],
        return_type="number",
        description="Calculates date difference in Snowflake.",
        caveats_and_pitfalls="Snowflake syntax: DATEDIFF(day, start_date, end_date). Unit is first, start is second, end is third.",
        unsupported_in_dialects=["postgres", "sqlite"],
    ),

    # --------------------------------------------------------------------------
    # Date Truncation (Cohorts / Monthly Grain)
    # --------------------------------------------------------------------------
    DialectFunctionSpec(
        name="date_trunc_postgres",
        dialect="postgres",
        category="datetime",
        syntax_pattern="date_trunc('{granularity}', {timestamp_col})",
        argument_types=["string ('month', 'year', 'day')", "timestamp"],
        return_type="timestamp",
        description="Truncates a timestamp to the specified precision in PostgreSQL.",
        caveats_and_pitfalls="Granularity is quoted lowercase string, e.g., date_trunc('month', created_at).",
    ),
    DialectFunctionSpec(
        name="date_trunc_duckdb",
        dialect="duckdb",
        category="datetime",
        syntax_pattern="date_trunc('{granularity}', {timestamp_col})",
        argument_types=["string ('month', 'year', 'day')", "timestamp"],
        return_type="timestamp",
        description="Truncates a timestamp to specified precision in DuckDB.",
    ),
    DialectFunctionSpec(
        name="DATE_TRUNC_bigquery",
        dialect="bigquery",
        category="datetime",
        syntax_pattern="DATE_TRUNC({date_col}, {GRANULARITY})",
        argument_types=["date/timestamp", "unquoted date_part (MONTH, YEAR)"],
        return_type="date/timestamp",
        description="Truncates date in BigQuery. Note date_col is first and part is second.",
        caveats_and_pitfalls="BigQuery requires date column FIRST, then unquoted part SECOND: DATE_TRUNC(event_date, MONTH).",
    ),
    DialectFunctionSpec(
        name="DATE_TRUNC_snowflake",
        dialect="snowflake",
        category="datetime",
        syntax_pattern="DATE_TRUNC('{granularity}', {timestamp_col})",
        argument_types=["string", "timestamp"],
        return_type="timestamp",
        description="Truncates timestamp to granularity in Snowflake.",
    ),

    # --------------------------------------------------------------------------
    # Safe Division / Zero Guard
    # --------------------------------------------------------------------------
    DialectFunctionSpec(
        name="nullif_safe_division_postgres",
        dialect="postgres",
        category="null_handling",
        syntax_pattern="{numerator} / NULLIF({denominator}, 0)",
        argument_types=["numeric", "numeric"],
        return_type="numeric",
        description="Guards against division-by-zero crashes in PostgreSQL and DuckDB using NULLIF.",
        caveats_and_pitfalls=(
            "In PostgreSQL, integer division truncates: (1 / 2) = 0. Always cast or multiply "
            "by 100.0 or 1.0 before division: ({numerator} * 100.0) / NULLIF({denominator}, 0)."
        ),
    ),
    DialectFunctionSpec(
        name="nullif_safe_division_duckdb",
        dialect="duckdb",
        category="null_handling",
        syntax_pattern="{numerator} / NULLIF({denominator}, 0)",
        argument_types=["numeric", "numeric"],
        return_type="double",
        description="Safe division guard in DuckDB.",
    ),
    DialectFunctionSpec(
        name="IEEE_DIVIDE_bigquery",
        dialect="bigquery",
        category="null_handling",
        syntax_pattern="IEEE_DIVIDE({numerator}, {denominator})",
        argument_types=["float64", "float64"],
        return_type="float64",
        description="BigQuery safe division returning NULL/NaN on zero denominator instead of erroring.",
        caveats_and_pitfalls="Can also use SAFE_DIVIDE(numerator, denominator) in GoogleSQL.",
        unsupported_in_dialects=["postgres", "duckdb", "snowflake", "sqlite"],
    ),
    DialectFunctionSpec(
        name="DIV0_snowflake",
        dialect="snowflake",
        category="null_handling",
        syntax_pattern="DIV0({numerator}, {denominator})",
        argument_types=["numeric", "numeric"],
        return_type="numeric",
        description="Snowflake division returning 0 when the denominator is zero.",
        caveats_and_pitfalls="Returns 0.0 rather than NULL.",
        unsupported_in_dialects=["postgres", "duckdb", "bigquery", "sqlite"],
    ),

    # --------------------------------------------------------------------------
    # Window Filtering & QUALIFY Clause
    # --------------------------------------------------------------------------
    DialectFunctionSpec(
        name="row_number_cte_postgres",
        dialect="postgres",
        category="window",
        syntax_pattern=(
            "ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {order_col} DESC) AS rn\n"
            "-- In downstream CTE/query: WHERE rn = 1"
        ),
        argument_types=["window spec"],
        return_type="bigint",
        description="Window rank filtering in PostgreSQL using a CTE.",
        caveats_and_pitfalls=(
            "CRITICAL: PostgreSQL does NOT support the QUALIFY clause. "
            "Attempting to use 'QUALIFY ROW_NUMBER() OVER (...) = 1' will cause syntax errors. "
            "You MUST compute ROW_NUMBER() in a CTE step, and filter 'WHERE rn = 1' in the subsequent step."
        ),
        unsupported_in_dialects=[],
    ),
    DialectFunctionSpec(
        name="qualify_clause_snowflake",
        dialect="snowflake",
        category="window",
        syntax_pattern="QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {order_col} DESC) = 1",
        argument_types=["boolean window expression"],
        return_type="filtered rows",
        description="Directly filters window function results without requiring a subquery or CTE.",
        unsupported_in_dialects=["postgres", "sqlite"],
    ),
    DialectFunctionSpec(
        name="qualify_clause_duckdb",
        dialect="duckdb",
        category="window",
        syntax_pattern="QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {order_col} DESC) = 1",
        argument_types=["boolean window expression"],
        return_type="filtered rows",
        description="DuckDB natively supports the QUALIFY clause.",
        unsupported_in_dialects=["postgres", "sqlite"],
    ),
    DialectFunctionSpec(
        name="qualify_clause_bigquery",
        dialect="bigquery",
        category="window",
        syntax_pattern="QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {order_col} DESC) = 1",
        argument_types=["boolean window expression"],
        return_type="filtered rows",
        description="BigQuery GoogleSQL natively supports QUALIFY.",
        unsupported_in_dialects=["postgres", "sqlite"],
    ),

    # --------------------------------------------------------------------------
    # String Aggregation
    # --------------------------------------------------------------------------
    DialectFunctionSpec(
        name="string_agg_postgres",
        dialect="postgres",
        category="aggregation",
        syntax_pattern="string_agg({col}, '{delimiter}' ORDER BY {order_col})",
        argument_types=["text", "string (delimiter)"],
        return_type="text",
        description="Concatenates non-null string values into a single text string with a delimiter in PostgreSQL.",
        caveats_and_pitfalls="PostgreSQL uses string_agg(col, delimiter). Optional ORDER BY goes inside the function call.",
    ),
    DialectFunctionSpec(
        name="STRING_AGG_bigquery",
        dialect="bigquery",
        category="aggregation",
        syntax_pattern="STRING_AGG({col}, '{delimiter}' ORDER BY {order_col})",
        argument_types=["string", "string"],
        return_type="string",
        description="BigQuery string concatenation aggregate.",
    ),
    DialectFunctionSpec(
        name="LISTAGG_snowflake",
        dialect="snowflake",
        category="aggregation",
        syntax_pattern="LISTAGG({col}, '{delimiter}') WITHIN GROUP (ORDER BY {order_col})",
        argument_types=["varchar", "varchar"],
        return_type="varchar",
        description="Snowflake string concatenation aggregate.",
        caveats_and_pitfalls="Snowflake requires WITHIN GROUP (ORDER BY ...): LISTAGG(col, ', ') WITHIN GROUP (ORDER BY col).",
        unsupported_in_dialects=["postgres", "bigquery", "duckdb", "sqlite"],
    ),

    # --------------------------------------------------------------------------
    # Percentile / Median Calculations
    # --------------------------------------------------------------------------
    DialectFunctionSpec(
        name="percentile_cont_postgres",
        dialect="postgres",
        category="aggregation",
        syntax_pattern="percentile_cont({fraction}) WITHIN GROUP (ORDER BY {col})",
        argument_types=["double (0.0 to 1.0)", "numeric column"],
        return_type="double precision",
        description="Continuous percentile / median in PostgreSQL.",
        caveats_and_pitfalls="PostgreSQL requires WITHIN GROUP (ORDER BY col): percentile_cont(0.5) WITHIN GROUP (ORDER BY salary).",
    ),
    DialectFunctionSpec(
        name="PERCENTILE_CONT_bigquery",
        dialect="bigquery",
        category="aggregation",
        syntax_pattern="PERCENTILE_CONT({col}, {fraction}) OVER()",
        argument_types=["numeric column", "float64 (0.0 to 1.0)"],
        return_type="float64",
        description="BigQuery continuous percentile (analytic function with OVER clause).",
    ),
    DialectFunctionSpec(
        name="PERCENTILE_CONT_snowflake",
        dialect="snowflake",
        category="aggregation",
        syntax_pattern="PERCENTILE_CONT({fraction}) WITHIN GROUP (ORDER BY {col})",
        argument_types=["float", "numeric column"],
        return_type="float",
        description="Snowflake continuous percentile aggregate.",
    ),

    # --------------------------------------------------------------------------
    # Cumulative Sum / Running Total
    # --------------------------------------------------------------------------
    DialectFunctionSpec(
        name="running_total_window",
        dialect="postgres",
        category="window",
        syntax_pattern="SUM({measure_col}) OVER (PARTITION BY {partition_col} ORDER BY {order_col} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)",
        argument_types=["numeric", "window order spec"],
        return_type="numeric",
        description="Calculates a running total across ordered rows.",
        caveats_and_pitfalls="Explicitly specify ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW to prevent duplicate ordering peer aggregation.",
    ),
]


# Map keywords and analytical concepts to search tags
OPERATION_KEYWORDS: Dict[str, Set[str]] = {
    "datetime": {
        "date", "time", "day", "days", "month", "monthly", "year", "annual", "week",
        "difference", "diff", "interval", "duration", "between", "signup", "return", "cohort",
        "returned", "truncate", "trunc", "period", "timestamp",
    },
    "null_handling": {
        "percentage", "percent", "%", "rate", "ratio", "divide", "division", "zero",
        "denominator", "numerator", "safe", "nullif", "ieee_divide", "div0",
    },
    "window": {
        "rank", "ranking", "top", "row_number", "qualify", "dense_rank", "partition",
        "lead", "lag", "running", "cumulative", "moving", "average", "window",
    },
    "aggregation": {
        "aggregate", "distinct", "unique", "string_agg", "listagg", "concat", "concatenate",
        "median", "percentile", "p50", "p90", "p95", "p99", "sum", "count",
    },
}


class DialectFunctionRAG:
    """Retrieval-Augmented Generation component for dialect-specific SQL functions and syntax."""

    def __init__(self, catalog: Optional[List[DialectFunctionSpec]] = None):
        self.catalog = catalog or list(DIALECT_CATALOG)

    def retrieve_functions(
        self,
        query_text: str,
        target_dialect: str = "postgres",
        limit: int = 5,
    ) -> List[DialectFunctionSpec]:
        """Retrieves dialect function documentation matching analytical operations in query_text.

        Args:
            query_text: User question or analytical plan description.
            target_dialect: SQL dialect of target database.
            limit: Maximum functions to return.

        Returns:
            Ranked list of relevant DialectFunctionSpec instances for the target dialect.
        """
        dialect = normalize_dialect(target_dialect)
        q_tokens = set(re.findall(r"\w+", query_text.lower()))

        # Determine relevant categories
        matched_categories: Set[str] = set()
        for cat, kw_set in OPERATION_KEYWORDS.items():
            if q_tokens.intersection(kw_set):
                matched_categories.add(cat)

        # If no specific categories matched, default to null_handling and datetime (safest baseline)
        if not matched_categories:
            matched_categories = {"null_handling", "datetime"}

        scored_specs: List[tuple[int, DialectFunctionSpec]] = []
        for spec in self.catalog:
            # 1. Must match target dialect
            if spec.dialect != dialect:
                continue

            # 2. Score based on category and token overlap
            score = 0
            if spec.category in matched_categories:
                score += 10

            spec_tokens = set(re.findall(r"\w+", f"{spec.name} {spec.description} {spec.syntax_pattern} {spec.caveats_and_pitfalls or ''}".lower()))
            overlap = len(q_tokens.intersection(spec_tokens))
            score += overlap

            if score > 0:
                scored_specs.append((score, spec))

        # Sort descending by score
        scored_specs.sort(key=lambda x: x[0], reverse=True)
        return [spec for _, spec in scored_specs[:limit]]

    def get_unsupported_functions(self, target_dialect: str) -> List[DialectFunctionSpec]:
        """Returns specifications of functions/idioms that are strictly forbidden in target_dialect."""
        dialect = normalize_dialect(target_dialect)
        forbidden: List[DialectFunctionSpec] = []
        for spec in self.catalog:
            if dialect in spec.unsupported_in_dialects:
                forbidden.append(spec)
        return forbidden

    def format_prompt_context(
        self,
        query_text: str,
        target_dialect: str = "postgres",
        limit: int = 4,
    ) -> str:
        """Formats retrieved dialect documentation into a structured prompt context block for LLM prompts.

        Args:
            query_text: User query or strategy description.
            target_dialect: SQL dialect.
            limit: Number of retrieved functions.

        Returns:
            Clean Markdown section to inject into generation prompts.
        """
        dialect = normalize_dialect(target_dialect)
        relevant_funcs = self.retrieve_functions(query_text, target_dialect=dialect, limit=limit)
        unsupported = self.get_unsupported_functions(target_dialect=dialect)

        lines: List[str] = [
            f"### Target SQL Dialect Knowledge & Rules: [{dialect.upper()}]",
            f"Adhere strictly to {dialect.upper()} SQL function syntax and argument ordering:",
        ]

        if relevant_funcs:
            lines.append("\n**Applicable Functions & Syntaxes**:")
            for f in relevant_funcs:
                lines.append(f"- **`{f.name}`** ({f.category}):")
                lines.append(f"  - *Syntax*: `{f.syntax_pattern}`")
                lines.append(f"  - *Semantics*: {f.description}")
                if f.caveats_and_pitfalls:
                    lines.append(f"  - *Caveat*: {f.caveats_and_pitfalls}")

        if unsupported:
            lines.append(f"\n**FORBIDDEN In {dialect.upper()} (Do NOT Use)**:")
            for u in unsupported:
                lines.append(f"- Do NOT use `{u.name}`: {u.caveats_and_pitfalls or u.description}")

        return "\n".join(lines)

    def validate_sql_dialect_compatibility(
        self,
        sql: str,
        target_dialect: str = "postgres",
    ) -> List[str]:
        """Validates that a generated SQL string does not use functions forbidden in the target dialect.

        Args:
            sql: Generated SQL query string.
            target_dialect: Target SQL dialect.

        Returns:
            List of violation error messages (empty if completely valid).
        """
        dialect = normalize_dialect(target_dialect)
        sql_upper = sql.upper()
        violations: List[str] = []

        # 1. QUALIFY check in PostgreSQL / SQLite
        if dialect in ("postgres", "sqlite") and re.search(r"\bQUALIFY\b", sql_upper):
            violations.append(
                f"Dialect Incompatibility: '{dialect.upper()}' does not support the QUALIFY clause. "
                "Compute the window function in a CTE and filter in the outer query."
            )

        # 2. DATEDIFF() check in PostgreSQL
        if dialect == "postgres" and re.search(r"\bDATEDIFF\s*\(", sql_upper):
            violations.append(
                "Dialect Incompatibility: PostgreSQL does not have a DATEDIFF() function. "
                "Use date subtraction `(end_date::date - start_date::date)` or AGE() instead."
            )

        # 3. IEEE_DIVIDE check in non-BigQuery dialects
        if dialect != "bigquery" and re.search(r"\bIEEE_DIVIDE\s*\(", sql_upper):
            violations.append(
                f"Dialect Incompatibility: IEEE_DIVIDE() is specific to BigQuery and unsupported in {dialect.upper()}. "
                "Use `numerator / NULLIF(denominator, 0)` instead."
            )

        # 4. DIV0 check in non-Snowflake dialects
        if dialect != "snowflake" and re.search(r"\bDIV0\s*\(", sql_upper):
            violations.append(
                f"Dialect Incompatibility: DIV0() is specific to Snowflake and unsupported in {dialect.upper()}. "
                "Use `numerator / NULLIF(denominator, 0)` instead."
            )

        return violations

