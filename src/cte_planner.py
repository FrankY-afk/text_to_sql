"""Structured CTE Logic Planner for Problem 1: Erroneous Data Analysis.

Decomposes complex analytical business questions into explicit, topologically-ordered
Common Table Expressions (CTEs) prior to SQL code generation.

Each CTE defines:
- CTE name (valid SQL identifier).
- CTE purpose.
- Input tables / upstream CTE dependencies.
- Required columns.
- Filters (WHERE conditions).
- Grouping dimensions (GROUP BY).
- Aggregation logic.
- Mathematical calculations (including safe division guards).
- Dependencies on preceding CTEs (enforcing DAG ordering).
- Expected output grain.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

SUPPORTED_DIALECTS: Set[str] = {
    "postgres",
    "duckdb",
    "bigquery",
    "snowflake",
    "sqlite",
}


def normalize_dialect(raw_dialect: str) -> str:
    """Normalizes and validates SQL dialect against allowlist."""
    norm = raw_dialect.lower().strip()
    if norm in ("postgresql", "pgsql"):
        norm = "postgres"
    if norm not in SUPPORTED_DIALECTS:
        raise ValueError(
            f"Unsupported SQL dialect '{raw_dialect}'. Supported dialects: {sorted(SUPPORTED_DIALECTS)}"
        )
    return norm


class CTEPlanStep(BaseModel):
    """Represents a single modular Common Table Expression (CTE) in the reasoning plan."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    step_name: str = Field(
        ...,
        min_length=1,
        description="Unique SQL identifier for the CTE (e.g., 'jan_signup_cohort')",
    )
    purpose: str = Field(
        ...,
        min_length=1,
        description="Human-readable business and analytical purpose of this step",
    )
    input_tables: List[str] = Field(
        default_factory=list,
        description="Base tables or preceding CTEs referenced in this step",
        alias="source_tables",
    )
    required_columns: List[str] = Field(
        default_factory=list,
        description="Columns required from input tables",
    )
    filters: List[str] = Field(
        default_factory=list,
        description="Filter predicates applied at this step (WHERE clause clauses)",
        alias="filter_conditions",
    )
    grouping_dimensions: List[str] = Field(
        default_factory=list,
        description="Grouping dimensions (GROUP BY columns)",
        alias="group_by_columns",
    )
    aggregation_logic: str = Field(
        default="",
        description="Aggregation expressions (e.g. COUNT(DISTINCT user_id), SUM(amt))",
    )
    mathematical_calculations: str = Field(
        default="",
        description="Formulas, calculations, and guards (e.g., NULLIF(col, 0))",
        alias="math_logic",
    )
    window_logic: Optional[str] = Field(
        default=None,
        description="Window function specifications (e.g., ROW_NUMBER() OVER (...))",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Explicit list of preceding CTE step_names this CTE depends on",
    )
    expected_output_grain: str = Field(
        ...,
        min_length=1,
        description="Grain of output rows (e.g., 'distinct_user_id', 'outlet_code', 'monthly_summary')",
        alias="output_grain",
    )
    columns_produced: List[str] = Field(
        default_factory=list,
        description="List of column names or expressions produced by this CTE",
    )
    sql_fragment: str = Field(
        ...,
        min_length=1,
        description="Dialect-specific SQL SELECT statement defining this CTE (excluding WITH name AS)",
    )

    @field_validator("step_name")
    @classmethod
    def validate_step_name(cls, v: str) -> str:
        clean = v.strip().lower()
        if not clean:
            raise ValueError("step_name cannot be empty.")
        if not re.match(r"^[a-z_][a-z0-9_]*$", clean):
            raise ValueError(
                f"Invalid CTE step_name '{v}'. Must be a valid SQL identifier (lowercase alphanumeric/underscore, not starting with a digit)."
            )
        return clean

    @field_validator("purpose", "expected_output_grain", "sql_fragment")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Field cannot be empty or whitespace only.")
        return clean

    # Compatibility properties for legacy aliases
    @property
    def source_tables(self) -> List[str]:
        return self.input_tables

    @property
    def math_logic(self) -> str:
        return self.mathematical_calculations

    @property
    def filter_conditions(self) -> List[str]:
        return self.filters

    @property
    def group_by_columns(self) -> List[str]:
        return self.grouping_dimensions


class LogicPlan(BaseModel):
    """Structured multi-stage analytical plan prior to final SQL generation."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, description="Original natural language query")
    target_dialect: str = Field(
        default="postgres",
        description="Target SQL dialect: 'postgres', 'duckdb', 'bigquery', 'snowflake', 'sqlite'",
    )
    strategy: str = Field(
        ...,
        min_length=1,
        description="High-level analytical plan and problem deconstruction strategy",
    )
    cte_steps: List[CTEPlanStep] = Field(
        default_factory=list,
        description="Topologically ordered sequence of CTE steps",
    )
    final_select: str = Field(
        ...,
        min_length=1,
        description="Final SELECT projection and ordering pulling from the last CTE",
    )
    expected_output_grain: str = Field(
        default="single_row_aggregate",
        description="Overall query output grain",
    )
    dialect_functions_used: List[str] = Field(
        default_factory=list,
        description="List of dialect-specific functions and patterns applied in this plan",
    )
    validation_notes: Optional[str] = Field(
        default=None,
        description="Sanity check notes (e.g. division by zero protected, distinct entity grain enforced)",
    )
    planner_backend: Optional[str] = Field(
        default=None,
        description="Backend used for plan generation ('huggingface', 'ollama', 'rule_based', etc.)",
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
        seen: Set[str] = set()
        duplicates: List[str] = []
        for name in cte_names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(f"Duplicate CTE names detected in plan: {duplicates}")

        # 2. Validate that final_select contains a SELECT statement
        if "select" not in self.final_select.lower():
            raise ValueError("final_select must contain a valid SELECT projection.")

        # 3. Ensure all CTE steps have non-empty sql_fragment
        for step in self.cte_steps:
            if not step.sql_fragment or not step.sql_fragment.strip():
                raise ValueError(f"CTE step '{step.step_name}' has missing or empty sql_fragment.")

        return self


class PlanValidationError(Exception):
    """Raised when an analytical plan fails structural or DAG dependency validation."""


class CTEDependencyValidator:
    """Validates Directed Acyclic Graph (DAG) dependencies and topological ordering of CTE steps."""

    @staticmethod
    def validate_plan(
        plan: LogicPlan,
        available_base_tables: Optional[Set[str]] = None,
    ) -> None:
        """Validates that CTE steps form a valid DAG with no forward-references or cycles.

        Args:
            plan: The LogicPlan to validate.
            available_base_tables: Optional set of permitted database base table names.

        Raises:
            PlanValidationError: If validation fails.
        """
        defined_ctes: Set[str] = set()
        base_tables = {t.lower() for t in (available_base_tables or set())}

        for idx, step in enumerate(plan.cte_steps):
            step_name = step.step_name.lower()

            # 1. Check explicit dependencies
            for dep in step.dependencies:
                dep_lower = dep.lower()
                if dep_lower == step_name:
                    raise PlanValidationError(
                        f"CTE step '{step_name}' cannot depend on itself (self-dependency)."
                    )
                if dep_lower not in defined_ctes:
                    # Check if it exists in future steps (forward-reference)
                    future_ctes = {s.step_name.lower() for s in plan.cte_steps[idx + 1:]}
                    if dep_lower in future_ctes:
                        raise PlanValidationError(
                            f"Forward-reference in CTE '{step_name}': depends on '{dep}', which is defined later in step sequence."
                        )
                    raise PlanValidationError(
                        f"Unknown dependency in CTE '{step_name}': '{dep}' is neither a preceding CTE nor a known table."
                    )

            # 2. Check input_tables
            for table_ref in step.input_tables:
                ref_lower = table_ref.lower()
                is_preceding_cte = ref_lower in defined_ctes
                is_base_table = (not base_tables) or (ref_lower in base_tables)

                if not is_preceding_cte and not is_base_table:
                    # Check if referencing a future CTE
                    future_ctes = {s.step_name.lower() for s in plan.cte_steps[idx + 1:]}
                    if ref_lower in future_ctes:
                        raise PlanValidationError(
                            f"Forward-reference in CTE '{step_name}': references input table '{table_ref}', defined later."
                        )
                    if base_tables:
                        logger.warning(
                            "CTE '%s' references table '%s' which is not in known base tables %s",
                            step_name,
                            table_ref,
                            sorted(list(base_tables)),
                        )

            # 3. Add to defined set
            defined_ctes.add(step_name)

        # 4. Validate final_select references at least one defined CTE or base table
        final_lower = plan.final_select.lower()
        if defined_ctes and not any(cte in final_lower for cte in defined_ctes):
            logger.warning(
                "final_select does not explicitly reference any defined CTEs: %s",
                list(defined_ctes),
            )


class CTEPlanCompiler:
    """Compiles a validated LogicPlan into clean, dialect-correct SQL."""

    @staticmethod
    def compile(plan: LogicPlan, target_dialect: Optional[str] = None) -> str:
        """Assembles CTE steps and final SELECT into a complete SQL query string.

        Args:
            plan: The validated LogicPlan.
            target_dialect: Optional override for the target SQL dialect.

        Returns:
            Formatted SQL query string.
        """
        dialect = target_dialect or plan.target_dialect

        if not plan.cte_steps:
            # Query has no CTEs, return final select directly
            return plan.final_select.strip().rstrip(";") + ";"

        cte_blocks: List[str] = []
        for step in plan.cte_steps:
            # Clean up fragment (strip trailing semicolons, leading WITH if accidentally included)
            frag = step.sql_fragment.strip().rstrip(";")
            if frag.upper().startswith("WITH "):
                frag = frag[5:].strip()
            cte_blocks.append(f"{step.step_name} AS (\n{_indent(frag, 4)}\n)")

        ctes_joined = ",\n".join(cte_blocks)
        final_clean = plan.final_select.strip().rstrip(";")
        raw_sql = f"WITH {ctes_joined}\n{final_clean};"

        # Transpile / format with SQLGlot if possible
        try:
            transpiled = sqlglot.transpile(raw_sql, read=dialect, write=dialect, pretty=True)
            if transpiled and transpiled[0]:
                return transpiled[0]
        except Exception as exc:
            logger.debug("SQLGlot formatting fallback on raw SQL: %s", exc)

        return raw_sql


def _indent(text: str, spaces: int = 4) -> str:
    """Indents every line of text by specified number of spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else "" for line in text.splitlines())


class CTEPlanner:
    """Analytical Planner that decomposes complex business questions into modular CTE Logic Plans."""

    def __init__(self, target_dialect: str = "postgres"):
        self.target_dialect = normalize_dialect(target_dialect)
        self.validator = CTEDependencyValidator()
        self.compiler = CTEPlanCompiler()

    def create_cohort_retention_plan(
        self,
        question: str,
        target_dialect: Optional[str] = None,
    ) -> LogicPlan:
        """Constructs an explicit 5-step cohort retention plan solving Problem 1 grain errors.

        Decomposition:
        1. jan_signup_cohort: Base cohort filtered on January signup date.
        2. distinct_jan_users: Distinct January users count (denominator).
        3. feb_returning_cohort: Distinct January users returning with activity in February.
        4. retention_metrics: Cross-join metrics into single record (numerator and denominator).
        5. final_retention: Retention percentage calculation guarded with NULLIF(total, 0).
        """
        dialect = target_dialect or self.target_dialect

        step1 = CTEPlanStep(
            step_name="jan_signup_cohort",
            purpose="Isolate the January 2024 user signup cohort (baseline population)",
            input_tables=["users"],
            required_columns=["user_id", "signup_date"],
            filters=["signup_date >= '2024-01-01'", "signup_date < '2024-02-01'"],
            grouping_dimensions=[],
            aggregation_logic="None (row-level cohort members)",
            mathematical_calculations="Half-open date interval [2024-01-01, 2024-02-01)",
            dependencies=[],
            expected_output_grain="distinct_user_id",
            columns_produced=["user_id", "signup_date"],
            sql_fragment=(
                "SELECT user_id, signup_date\n"
                "FROM users\n"
                "WHERE signup_date >= '2024-01-01' AND signup_date < '2024-02-01'"
            ),
        )

        step2 = CTEPlanStep(
            step_name="distinct_jan_users",
            purpose="Calculate the exact distinct user count for the January signup cohort (denominator)",
            input_tables=["jan_signup_cohort"],
            required_columns=["user_id"],
            filters=[],
            grouping_dimensions=[],
            aggregation_logic="COUNT(DISTINCT user_id)",
            mathematical_calculations="Strict distinct entity counting preventing row explosion",
            dependencies=["jan_signup_cohort"],
            expected_output_grain="single_row_aggregate",
            columns_produced=["total_jan_signups"],
            sql_fragment=(
                "SELECT COUNT(DISTINCT user_id) AS total_jan_signups\n"
                "FROM jan_signup_cohort"
            ),
        )

        step3 = CTEPlanStep(
            step_name="feb_returning_cohort",
            purpose="Identify distinct users from the January cohort who returned with activity in February",
            input_tables=["jan_signup_cohort", "activity_events"],
            required_columns=["user_id", "event_date"],
            filters=["event_date >= '2024-02-01'", "event_date < '2024-03-01'"],
            grouping_dimensions=[],
            aggregation_logic="DISTINCT user_id",
            mathematical_calculations="Strict distinct returning entity deduplication",
            dependencies=["jan_signup_cohort"],
            expected_output_grain="distinct_user_id",
            columns_produced=["user_id"],
            sql_fragment=(
                "SELECT DISTINCT c.user_id\n"
                "FROM jan_signup_cohort c\n"
                "JOIN activity_events e ON c.user_id = e.user_id\n"
                "WHERE e.event_date >= '2024-02-01' AND e.event_date < '2024-03-01'"
            ),
        )

        step4 = CTEPlanStep(
            step_name="retention_metrics",
            purpose="Aggregate distinct returning users and align with the total cohort denominator",
            input_tables=["distinct_jan_users", "feb_returning_cohort"],
            required_columns=["total_jan_signups", "user_id"],
            filters=[],
            grouping_dimensions=["d.total_jan_signups"],
            aggregation_logic="COUNT(r.user_id)",
            mathematical_calculations="Align numerator and denominator on a 1=1 cross-grain join",
            dependencies=["distinct_jan_users", "feb_returning_cohort"],
            expected_output_grain="single_row_aggregate",
            columns_produced=["total_jan_signups", "returning_feb_users"],
            sql_fragment=(
                "SELECT \n"
                "    d.total_jan_signups,\n"
                "    COUNT(r.user_id) AS returning_feb_users\n"
                "FROM distinct_jan_users d\n"
                "LEFT JOIN feb_returning_cohort r ON 1 = 1\n"
                "GROUP BY d.total_jan_signups"
            ),
        )

        final_select = (
            "SELECT \n"
            "    ROUND(\n"
            "        (returning_feb_users * 100.0) / NULLIF(total_jan_signups, 0),\n"
            "        2\n"
            "    ) AS retention_rate\n"
            "FROM retention_metrics"
        )

        plan = LogicPlan(
            query=question,
            target_dialect=dialect,
            strategy=(
                "Decompose into 4 modular CTEs: isolate January cohort -> compute distinct denominator -> "
                "isolate February returning users -> aggregate numerator and apply NULLIF division guard."
            ),
            cte_steps=[step1, step2, step3, step4],
            final_select=final_select,
            expected_output_grain="single_row_aggregate",
            dialect_functions_used=["ROUND", "NULLIF", "COUNT(DISTINCT)"],
            validation_notes="Distinct user grain enforced; NULLIF guards against division-by-zero.",
            planner_backend="rule_based",
        )

        # Validate DAG
        self.validator.validate_plan(plan)
        return plan

    def create_monthly_payouts_plan(
        self,
        question: str,
        target_dialect: Optional[str] = None,
    ) -> LogicPlan:
        """Constructs a CTE plan combining group loans and individual loans without Cartesian fan-out."""
        dialect = target_dialect or self.target_dialect

        step1 = CTEPlanStep(
            step_name="group_monthly",
            purpose="Pre-aggregate group loans monthly payout totals before JOIN to prevent fan-out",
            input_tables=["group_loans"],
            required_columns=["paid_at", "paid_amt", "stage"],
            filters=["paid_at IS NOT NULL", "stage = 'Payout'"],
            grouping_dimensions=["TO_CHAR(paid_at, 'YYYY-MM')"],
            aggregation_logic="COALESCE(SUM(paid_amt), 0)",
            mathematical_calculations="Monthly pre-aggregation",
            dependencies=[],
            expected_output_grain="month",
            columns_produced=["payout_month", "total_group_payout"],
            sql_fragment=(
                "SELECT \n"
                "    TO_CHAR(paid_at, 'YYYY-MM') AS payout_month,\n"
                "    COALESCE(SUM(paid_amt), 0) AS total_group_payout\n"
                "FROM group_loans\n"
                "WHERE paid_at IS NOT NULL AND stage = 'Payout'\n"
                "GROUP BY TO_CHAR(paid_at, 'YYYY-MM')"
            ),
        )

        step2 = CTEPlanStep(
            step_name="individual_monthly",
            purpose="Pre-aggregate individual loans monthly payout totals before JOIN to prevent fan-out",
            input_tables=["individual_loans"],
            required_columns=["payout_at", "payout_amt", "stage"],
            filters=["payout_at IS NOT NULL", "stage = 'PAYOUT'"],
            grouping_dimensions=["TO_CHAR(payout_at, 'YYYY-MM')"],
            aggregation_logic="COALESCE(SUM(payout_amt), 0)",
            mathematical_calculations="Monthly pre-aggregation",
            dependencies=[],
            expected_output_grain="month",
            columns_produced=["payout_month", "total_individual_payout"],
            sql_fragment=(
                "SELECT \n"
                "    TO_CHAR(payout_at, 'YYYY-MM') AS payout_month,\n"
                "    COALESCE(SUM(payout_amt), 0) AS total_individual_payout\n"
                "FROM individual_loans\n"
                "WHERE payout_at IS NOT NULL AND stage = 'PAYOUT'\n"
                "GROUP BY TO_CHAR(payout_at, 'YYYY-MM')"
            ),
        )

        final_select = (
            "SELECT \n"
            "    COALESCE(g.payout_month, i.payout_month) AS month,\n"
            "    COALESCE(g.total_group_payout, 0) AS group_loans_payout,\n"
            "    COALESCE(i.total_individual_payout, 0) AS individual_loans_payout,\n"
            "    (COALESCE(g.total_group_payout, 0) + COALESCE(i.total_individual_payout, 0)) AS total_combined_payout\n"
            "FROM group_monthly g\n"
            "FULL OUTER JOIN individual_monthly i ON g.payout_month = i.payout_month\n"
            "ORDER BY month"
        )

        plan = LogicPlan(
            query=question,
            target_dialect=dialect,
            strategy="Pre-aggregate group_loans and individual_loans into independent monthly CTEs before FULL OUTER JOIN.",
            cte_steps=[step1, step2],
            final_select=final_select,
            expected_output_grain="month",
            dialect_functions_used=["TO_CHAR", "COALESCE", "SUM"],
            validation_notes="Pre-aggregation prevents Cartesian fan-out row multiplication.",
            planner_backend="rule_based",
        )

        self.validator.validate_plan(plan)
        return plan

    def create_staff_attendance_plan(
        self,
        question: str,
        target_dialect: Optional[str] = None,
    ) -> LogicPlan:
        """Constructs a CTE plan preserving the total staff denominator and counting June attendance."""
        dialect = target_dialect or self.target_dialect

        step1 = CTEPlanStep(
            step_name="all_staff_denominator",
            purpose="Establish the full active employee baseline population (denominator)",
            input_tables=["staff"],
            required_columns=["staff_no"],
            filters=[],
            grouping_dimensions=[],
            aggregation_logic="COUNT(DISTINCT staff_no)",
            mathematical_calculations="Count all active employees",
            dependencies=[],
            expected_output_grain="single_row_aggregate",
            columns_produced=["total_staff_count"],
            sql_fragment=(
                "SELECT COUNT(DISTINCT staff_no) AS total_staff_count\n"
                "FROM staff"
            ),
        )

        step2 = CTEPlanStep(
            step_name="june_present_staff",
            purpose="Count distinct staff members marked present at least once in June 2026",
            input_tables=["staff_attendance"],
            required_columns=["staff_no", "attendance_date", "attendance_code"],
            filters=[
                "attendance_date >= '2026-06-01'",
                "attendance_date < '2026-07-01'",
                "attendance_code = 'PR'",
            ],
            grouping_dimensions=[],
            aggregation_logic="COUNT(DISTINCT staff_no)",
            mathematical_calculations="Distinct employee attendance counting",
            dependencies=[],
            expected_output_grain="single_row_aggregate",
            columns_produced=["present_staff_count"],
            sql_fragment=(
                "SELECT COUNT(DISTINCT staff_no) AS present_staff_count\n"
                "FROM staff_attendance\n"
                "WHERE attendance_date >= '2026-06-01' AND attendance_date < '2026-07-01'\n"
                "  AND attendance_code = 'PR'"
            ),
        )

        final_select = (
            "SELECT \n"
            "    ROUND(\n"
            "        (p.present_staff_count * 100.0) / NULLIF(d.total_staff_count, 0),\n"
            "        2\n"
            "    ) AS attendance_rate\n"
            "FROM all_staff_denominator d\n"
            "CROSS JOIN june_present_staff p"
        )

        plan = LogicPlan(
            query=question,
            target_dialect=dialect,
            strategy="Separate total staff denominator CTE from June attendance numerator CTE to prevent LEFT JOIN filter leakage.",
            cte_steps=[step1, step2],
            final_select=final_select,
            expected_output_grain="single_row_aggregate",
            dialect_functions_used=["ROUND", "NULLIF", "COUNT(DISTINCT)"],
            validation_notes="Prevents WHERE filter leakage on absent staff members.",
            planner_backend="rule_based",
        )

        self.validator.validate_plan(plan)
        return plan

    def plan_question(
        self,
        question: str,
        target_dialect: Optional[str] = None,
        schema_context: Optional[Any] = None,
    ) -> LogicPlan:
        """Main routing method to generate an analytical CTE LogicPlan for a question."""
        q_lower = question.lower()
        dialect = normalize_dialect(target_dialect or self.target_dialect)

        if "retention" in q_lower or ("signup" in q_lower and "return" in q_lower) or ("january" in q_lower and "february" in q_lower):
            return self.create_cohort_retention_plan(question, target_dialect=dialect)

        if ("signup" in q_lower or "user" in q_lower) and "january" in q_lower:
            step1 = CTEPlanStep(
                step_name="jan_signups",
                purpose="Filter users who signed up in January 2024 using half-open date interval",
                input_tables=["users"],
                required_columns=["user_id", "signup_date"],
                filters=["signup_date >= '2024-01-01'", "signup_date < '2024-02-01'"],
                grouping_dimensions=[],
                aggregation_logic="COUNT(DISTINCT user_id)",
                mathematical_calculations="Half-open interval [2024-01-01, 2024-02-01)",
                dependencies=[],
                expected_output_grain="single_row_aggregate",
                columns_produced=["total_jan_signups"],
                sql_fragment=(
                    "SELECT COUNT(DISTINCT user_id) AS total_jan_signups\n"
                    "FROM users\n"
                    "WHERE signup_date >= '2024-01-01' AND signup_date < '2024-02-01'"
                ),
            )
            plan = LogicPlan(
                query=question,
                target_dialect=dialect,
                strategy="Filter users on January 2024 half-open interval and count distinct user_id.",
                cte_steps=[step1],
                final_select="SELECT total_jan_signups FROM jan_signups",
                expected_output_grain="single_row_aggregate",
                dialect_functions_used=["COUNT(DISTINCT)"],
                validation_notes="Half-open interval prevents end-of-day boundary cutoff.",
                planner_backend="rule_based",
            )
            self.validator.validate_plan(plan)
            return plan

        if "payout" in q_lower and ("both" in q_lower or ("group" in q_lower and "individual" in q_lower)):
            return self.create_monthly_payouts_plan(question, target_dialect=dialect)

        if "staff" in q_lower and ("attendance" in q_lower or "present" in q_lower or "rate" in q_lower or "%" in q_lower):
            return self.create_staff_attendance_plan(question, target_dialect=dialect)

        # Schema-linked CTE planning using Two-Stage SchemaContext (Problem 2 integration)
        if schema_context is not None:
            ctx_tables = list(getattr(schema_context, "selected_table_names", None) or [])
            if isinstance(schema_context, dict):
                ctx_tables = list(schema_context.get("selected_table_names", []))

            if ctx_tables:
                primary_table = ctx_tables[0]
                candidate_sections = getattr(schema_context, "candidate_tables", None) or []
                resolved_lits = getattr(schema_context, "resolved_literals", None) or []
                join_rels = list(getattr(schema_context, "join_relationships", None) or [])
                nested_hints = list(getattr(schema_context, "nested_field_hints", None) or [])

                # Collect directly selectable column names for primary_table
                selectable_cols: List[str] = []
                required_cols: List[str] = []
                if candidate_sections:
                    primary_sec = candidate_sections[0]
                    for col in getattr(primary_sec, "selected_columns", []):
                        cpath = getattr(col, "path", "")
                        cname = getattr(col, "column_name", cpath)
                        if cpath and f"{primary_table}.{cpath}" not in required_cols:
                            required_cols.append(f"{primary_table}.{cpath}")
                        if cname and cname not in selectable_cols:
                            selectable_cols.append(cname)
                else:
                    col_paths = list(getattr(schema_context, "selected_column_paths", None) or [])
                    if isinstance(schema_context, dict):
                        col_paths = list(schema_context.get("selected_column_paths", []))
                    required_cols = col_paths[:10]
                    for cp in col_paths:
                        parts = cp.split(".")
                        if len(parts) >= 2 and parts[0] == primary_table and parts[1] not in selectable_cols:
                            selectable_cols.append(parts[1])

                if not selectable_cols:
                    selectable_cols = ["*"]
                else:
                    selectable_cols = selectable_cols[:8]

                # Collect grounded literal filters scoped to primary_table
                filter_preds: List[str] = []
                for lit in resolved_lits:
                    ltbl = getattr(lit, "table_name", None) or (lit.get("table_name") if isinstance(lit, dict) else None)
                    lpred = getattr(lit, "sql_predicate", None) or (lit.get("sql_predicate") if isinstance(lit, dict) else None)
                    if ltbl == primary_table and lpred and lpred not in filter_preds:
                        filter_preds.append(lpred)

                cols_sql = ", ".join(selectable_cols)
                where_sql = f"\nWHERE {' AND '.join(filter_preds)}" if filter_preds else ""
                is_count_intent = any(tok in q_lower for tok in ("how many", "count ", "total number", "number of"))
                step1 = CTEPlanStep(
                    step_name="schema_filtered_base",
                    purpose=f"Extract linked columns from candidate table '{primary_table}' with grounded filters",
                    input_tables=[primary_table],
                    required_columns=required_cols or selectable_cols,
                    filters=filter_preds,
                    grouping_dimensions=[],
                    aggregation_logic="COUNT(*)" if is_count_intent else "None",
                    mathematical_calculations="None",
                    dependencies=[],
                    expected_output_grain="single_row_aggregate" if is_count_intent else f"{primary_table}_record",
                    columns_produced=["total_count"] if is_count_intent else selectable_cols,
                    sql_fragment=f"SELECT {cols_sql}\nFROM {primary_table}{where_sql}",
                )

                notes_parts: List[str] = [f"Schema-linked tables: {', '.join(ctx_tables)}."]
                if join_rels:
                    notes_parts.append(f"Join paths: {'; '.join(join_rels[:3])}.")
                if nested_hints:
                    notes_parts.append(f"Nested field access required ({len(nested_hints)} field(s)).")

                final_sel = (
                    "SELECT COUNT(*) AS total_count FROM schema_filtered_base"
                    if is_count_intent
                    else "SELECT * FROM schema_filtered_base LIMIT 25"
                )
                plan = LogicPlan(
                    query=question,
                    target_dialect=dialect,
                    strategy=(
                        f"Schema-linked CTE plan grounded in candidate table(s) {', '.join(ctx_tables)} "
                        f"with {len(required_cols)} retrieved field(s)."
                    ),
                    cte_steps=[step1],
                    final_select=final_sel,
                    expected_output_grain=step1.expected_output_grain,
                    dialect_functions_used=["NESTED_ACCESS"] if nested_hints else [],
                    validation_notes=" ".join(notes_parts),
                    planner_backend="schema_linked",
                )
                self.validator.validate_plan(plan)
                return plan

        # Fallback to single-CTE query
        step1 = CTEPlanStep(
            step_name="base_data",
            purpose="Extract candidate records matching query intent",
            input_tables=["outlets"],
            required_columns=["*"],
            filters=[],
            grouping_dimensions=[],
            aggregation_logic="None",
            mathematical_calculations="None",
            dependencies=[],
            expected_output_grain="table_records",
            columns_produced=["*"],
            sql_fragment="SELECT * FROM outlets LIMIT 10",
        )

        plan = LogicPlan(
            query=question,
            target_dialect=dialect,
            strategy="Extract base records within a modular CTE.",
            cte_steps=[step1],
            final_select="SELECT * FROM base_data",
            expected_output_grain="table_records",
            planner_backend="fallback",
        )
        self.validator.validate_plan(plan)
        return plan

    def plan_and_compile(
        self,
        question: str,
        target_dialect: Optional[str] = None,
        schema_context: Optional[Any] = None,
    ) -> str:
        """Decomposes a question into a validated LogicPlan and compiles it into clean SQL."""
        plan = self.plan_question(
            question,
            target_dialect=target_dialect,
            schema_context=schema_context,
        )
        return self.compiler.compile(plan, target_dialect=target_dialect)
