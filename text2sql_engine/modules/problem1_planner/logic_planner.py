"""LogicPlanner: Modular CTE Planning and Dialect Function RAG solving Erroneous Data Analysis."""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Set
from langchain_core.messages import SystemMessage, HumanMessage

from text2sql_engine.core.config import get_llm
from text2sql_engine.core.exceptions import (
    PlanGenerationError,
    PlanValidationError,
    DependencyValidationError,
)
from text2sql_engine.modules.problem1_planner.schemas import (
    DialectFunctionSpec,
    CTEPlanStep,
    LogicPlan,
    normalize_dialect,
)
from text2sql_engine.modules.problem1_planner.dependency_validator import CTEDependencyValidator
from text2sql_engine.modules.problem1_planner.sql_compiler import SQLCompiler


# Curated dialect knowledge base for Spider 2.0 failure modes
DIALECT_CATALOG: List[DialectFunctionSpec] = [
    # Date Difference
    DialectFunctionSpec(
        name="date_diff",
        dialect="duckdb",
        category="datetime",
        syntax_pattern="date_diff('{unit}', {start_date}, {end_date})",
        description="Calculates date difference in DuckDB. Unit is first argument string (e.g., 'day', 'month', 'week').",
        notes="DuckDB uses unit first: date_diff('day', start, end)",
    ),
    DialectFunctionSpec(
        name="DATE_DIFF",
        dialect="bigquery",
        category="datetime",
        syntax_pattern="DATE_DIFF({end_date}, {start_date}, {unit})",
        description="Calculates date difference in BigQuery. End date is first, start date second, unit third.",
        notes="BigQuery reverses date arguments compared to DuckDB: DATE_DIFF(end, start, DAY)",
    ),
    DialectFunctionSpec(
        name="DATEDIFF",
        dialect="snowflake",
        category="datetime",
        syntax_pattern="DATEDIFF({unit}, {start_date}, {end_date})",
        description="Calculates date difference in Snowflake. Unit first, then start date, then end date.",
        notes="Snowflake syntax: DATEDIFF(day, start_date, end_date)",
    ),
    DialectFunctionSpec(
        name="date_diff_postgres",
        dialect="postgres",
        category="datetime",
        syntax_pattern="({end_date}::date - {start_date}::date)",
        description="Calculates day difference between two dates in PostgreSQL via subtraction or EXTRACT.",
        notes="Postgres does not have a generic DATEDIFF; subtract dates for days or use AGE() / EXTRACT.",
    ),
    # Date Truncation
    DialectFunctionSpec(
        name="date_trunc",
        dialect="duckdb",
        category="datetime",
        syntax_pattern="date_trunc('{part}', {timestamp_col})",
        description="Truncates timestamp to specified granularity in DuckDB (e.g., 'week', 'month', 'year').",
    ),
    DialectFunctionSpec(
        name="DATE_TRUNC",
        dialect="bigquery",
        category="datetime",
        syntax_pattern="DATE_TRUNC({date_col}, {PART})",
        description="Truncates date to specified part in BigQuery. Part is unquoted (e.g. WEEK, MONTH, YEAR).",
    ),
    DialectFunctionSpec(
        name="DATE_TRUNC",
        dialect="snowflake",
        category="datetime",
        syntax_pattern="DATE_TRUNC('{part}', {timestamp_col})",
        description="Truncates timestamp to granularity in Snowflake.",
    ),
    DialectFunctionSpec(
        name="date_trunc",
        dialect="postgres",
        category="datetime",
        syntax_pattern="date_trunc('{part}', {timestamp_col})",
        description="Truncates timestamp to granularity in PostgreSQL.",
    ),
    # Safe Division
    DialectFunctionSpec(
        name="safe_divide",
        dialect="duckdb",
        category="null_handling",
        syntax_pattern="{numerator} / NULLIF({denominator}, 0)",
        description="Guards against division-by-zero errors in DuckDB using NULLIF.",
        notes="Prevents query termination when denominator is zero.",
    ),
    DialectFunctionSpec(
        name="IEEE_DIVIDE",
        dialect="bigquery",
        category="null_handling",
        syntax_pattern="IEEE_DIVIDE({numerator}, {denominator})",
        description="BigQuery safe division returning IEEE NaN or NULL instead of throwing error.",
    ),
    DialectFunctionSpec(
        name="DIV0",
        dialect="snowflake",
        category="null_handling",
        syntax_pattern="DIV0({numerator}, {denominator})",
        description="Snowflake safe division returning 0 when dividing by zero.",
    ),
    DialectFunctionSpec(
        name="safe_divide_postgres",
        dialect="postgres",
        category="null_handling",
        syntax_pattern="{numerator} / NULLIF({denominator}, 0)",
        description="PostgreSQL division-by-zero guard using NULLIF.",
    ),
    # String Aggregation
    DialectFunctionSpec(
        name="string_agg",
        dialect="duckdb",
        category="aggregation",
        syntax_pattern="string_agg({col}, '{delimiter}')",
        description="Aggregates strings into delimited list in DuckDB.",
    ),
    DialectFunctionSpec(
        name="STRING_AGG",
        dialect="bigquery",
        category="aggregation",
        syntax_pattern="STRING_AGG({col}, '{delimiter}')",
        description="Aggregates strings into delimited list in BigQuery.",
    ),
    DialectFunctionSpec(
        name="LISTAGG",
        dialect="snowflake",
        category="aggregation",
        syntax_pattern="LISTAGG({col}, '{delimiter}')",
        description="Aggregates strings into delimited list in Snowflake.",
    ),
    DialectFunctionSpec(
        name="string_agg",
        dialect="postgres",
        category="aggregation",
        syntax_pattern="string_agg({col}, '{delimiter}')",
        description="Aggregates strings into delimited list in PostgreSQL.",
    ),
    # Window Filtering / QUALIFY
    DialectFunctionSpec(
        name="QUALIFY",
        dialect="duckdb",
        category="window",
        syntax_pattern="QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {order_col} DESC) <= {k}",
        description="Filters window function results directly without an extra subquery in DuckDB.",
    ),
    DialectFunctionSpec(
        name="QUALIFY",
        dialect="bigquery",
        category="window",
        syntax_pattern="QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {order_col} DESC) <= {k}",
        description="Filters window function results directly in BigQuery.",
    ),
    DialectFunctionSpec(
        name="QUALIFY",
        dialect="snowflake",
        category="window",
        syntax_pattern="QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {order_col} DESC) <= {k}",
        description="Filters window function results directly in Snowflake.",
    ),
    DialectFunctionSpec(
        name="cte_window_filter",
        dialect="postgres",
        category="window",
        syntax_pattern="WITH ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {order_col} DESC) AS _rn FROM {table}) SELECT * FROM ranked WHERE _rn <= {k}",
        description="PostgreSQL does not support QUALIFY; must wrap in a CTE or subquery to filter window ranks.",
        notes="PostgreSQL requirement: Always use CTE for window function filtering.",
    ),
]


class DialectFunctionRAG:
    """Retrieves relevant dialect-specific function patterns and rules for query execution."""

    def __init__(self, catalog: Optional[List[DialectFunctionSpec]] = None):
        self.catalog = catalog or DIALECT_CATALOG

    def retrieve(self, user_query: str, dialect: str) -> List[DialectFunctionSpec]:
        """Retrieve relevant dialect specs matching target dialect and query intent."""
        norm_dialect = normalize_dialect(dialect)
        query_lower = user_query.lower()
        matched: List[DialectFunctionSpec] = []

        # Intent detection
        detect_datetime = any(k in query_lower for k in ["date", "day", "month", "year", "week", "diff", "interval", "duration", "age", "recent", "time", "elapsed"])
        detect_safe_div = any(k in query_lower for k in ["ratio", "percent", "average", "avg", "rate", "divide", "share", "proportion", "spend", "retention"])
        detect_window = any(k in query_lower for k in ["rank", "top", "row_number", "dense_rank", "ntile", "lead", "lag", "running", "cumulative", "cohort", "first", "latest", "most recent"])
        detect_string_agg = any(k in query_lower for k in ["list", "comma", "concat", "combine", "strings", "names of"])

        for spec in self.catalog:
            if spec.dialect != norm_dialect:
                continue

            if spec.category == "datetime" and detect_datetime:
                matched.append(spec)
            elif spec.category == "null_handling" and detect_safe_div:
                matched.append(spec)
            elif spec.category == "window" and detect_window:
                matched.append(spec)
            elif spec.category == "aggregation" and detect_string_agg:
                matched.append(spec)

        # Fallback defaults for robustness
        if not matched:
            for spec in self.catalog:
                if spec.dialect == norm_dialect and spec.category in ["null_handling", "datetime"]:
                    matched.append(spec)

        return matched


class LogicPlanner:
    """Problem 1: CTE Logic Planner & Dialect Function RAG.
    
    Deconstructs complex natural language queries into verified, modular CTE steps
    with dialect-aware function rules, robust dependency checking, and fail-closed compilation.
    """

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_llm()
        self.rag = DialectFunctionRAG()
        self.compiler = SQLCompiler()

    def generate_logic_plan(
        self,
        user_query: str,
        schema_context: str,
        dialect: str = "duckdb",
        matched_literals: Optional[Dict[str, str]] = None,
        available_base_tables: Optional[Set[str]] = None,
    ) -> LogicPlan:
        """Deconstructs the user query into a verified LogicPlan with CTE steps."""
        dialect = normalize_dialect(dialect)
        dialect_specs = self.rag.retrieve(user_query, dialect)
        dialect_notes = "\n".join(
            f"- {spec.name} ({spec.category}): {spec.syntax_pattern} -> {spec.description} (Notes: {spec.notes or 'None'})"
            for spec in dialect_specs
        )

        literals_info = ""
        if matched_literals:
            literals_info = "\nDiscovered Database Literals:\n" + "\n".join(
                f"- Entity '{k}' maps to literal filter: {v}" for k, v in matched_literals.items()
            )

        # Extract base tables from schema_context if not explicitly provided
        base_tables = set(available_base_tables) if available_base_tables is not None else set()
        for line in schema_context.splitlines():
            line_str = line.strip()
            if line_str.startswith("Table:"):
                tbl_name = line_str.split("Table:", 1)[1].strip().split()[0]
                base_tables.add(tbl_name)

        prompt = f"""You are an elite SQL Architect solving Spider 2.0 benchmark tasks.
Your goal is to build a modular Common Table Expression (CTE) Logic Plan for the user query.

TARGET DIALECT: {dialect}

DIALECT-SPECIFIC FUNCTION GUIDELINES:
{dialect_notes}
{literals_info}

SCHEMA CONTEXT:
{schema_context}

USER QUERY:
{user_query}

RULES:
1. Deconstruct complex logic into a topologically ordered sequence of CTE steps.
   - For multi-stage analytical calculations (e.g. cohort retention, cohort aggregations, multi-level rankings):
     step 1: identify cohort/base filter,
     step 2: compute cohort size / intermediate grain,
     step 3: join and compute metrics,
     step 4: final projection.
2. For any division, ALWAYS guard against zero-division using NULLIF(denominator, 0) or dialect equivalent.
3. For dates and window functions, strictly follow the target dialect syntax patterns listed above.
4. If target dialect is postgres, DO NOT USE QUALIFY; use ROW_NUMBER() in a CTE and filter in the next step.
5. Every CTE step MUST specify:
   - "step_name": valid SQL identifier (alphanumeric/underscore)
   - "purpose": concise explanation
   - "source_tables": list of upstream base tables or preceding CTEs
   - "math_logic": explicit formulas, aggregations, window operations
   - "filter_conditions": list of filter expressions
   - "group_by_columns": list of grouping columns
   - "columns_produced": explicit list of output columns
   - "sql_fragment": valid SELECT statement defining this CTE (excluding WITH and CTE name)
6. Output ONLY a valid JSON object matching this exact schema:
{{
  "query": "{user_query}",
  "target_dialect": "{dialect}",
  "strategy": "High level strategy description",
  "cte_steps": [
    {{
      "step_name": "cte_name",
      "purpose": "What this CTE computes",
      "source_tables": ["table1"],
      "math_logic": "Aggregations / Window math / NULLIF logic",
      "filter_conditions": ["col = 'val'"],
      "group_by_columns": ["col1"],
      "columns_produced": ["col1", "agg1"],
      "sql_fragment": "SELECT col1, SUM(amount) AS agg1 FROM table1 GROUP BY col1"
    }}
  ],
  "final_select": "SELECT col1, agg1 FROM cte_name ORDER BY agg1 DESC LIMIT 10",
  "validation_notes": "Division by zero protected, dialect date syntax verified"
}}
"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a principal database engineer who outputs valid JSON only."),
                HumanMessage(content=prompt),
            ])
            content = response.content.strip() if hasattr(response, "content") else str(response).strip()

            # Extract JSON block if wrapped in markdown
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
            elif not content.startswith("{") and "{" in content:
                # Find outermost JSON object
                start_idx = content.find("{")
                end_idx = content.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    content = content[start_idx : end_idx + 1]

            data = json.loads(content)
            data["dialect_functions_used"] = dialect_specs
            
            # Explicitly mark backend
            backend_id = getattr(self.llm, "backend_name", None) or ("mock" if getattr(self.llm, "is_mock", False) else "real_llm")
            data["planner_backend"] = backend_id

            # Validate against Pydantic schema
            plan = LogicPlan.model_validate(data)

            # Validate CTE dependencies, ordering, and columns with AST inspector
            CTEDependencyValidator.validate_plan(
                plan,
                available_base_tables=base_tables if base_tables else None,
            )

            return plan

        except (json.JSONDecodeError, ValueError, DependencyValidationError, PlanValidationError) as e:
            raise PlanGenerationError(f"Failed to generate valid LogicPlan for query '{user_query}': {e}") from e
        except Exception as e:
            raise PlanGenerationError(f"Inference or planning error for query '{user_query}': {e}") from e

    def compile_sql(self, logic_plan: LogicPlan, target_dialect: Optional[str] = None) -> str:
        """Compiles validated LogicPlan into target SQL dialect using SQLCompiler."""
        return self.compiler.compile(logic_plan, target_dialect=target_dialect)
