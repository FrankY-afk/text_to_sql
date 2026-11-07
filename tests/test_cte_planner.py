"""Unit tests for Structured CTE Logic Planner (Problem 1, Mechanism A)."""

from __future__ import annotations

import sqlite3
import pytest
from src.config import get_settings
from src.cte_planner import (
    CTEDependencyValidator,
    CTEPlanCompiler,
    CTEPlanStep,
    CTEPlanner,
    LogicPlan,
    PlanValidationError,
)
from src.sql_validator import validate_readonly_sql


@pytest.fixture
def planner():
    return CTEPlanner(target_dialect="postgres")


def test_cte_step_valid_creation():
    """Verifies that CTEPlanStep validates required fields properly."""
    step = CTEPlanStep(
        step_name="jan_cohort",
        purpose="Filter users who signed up in January 2024",
        input_tables=["users"],
        required_columns=["user_id", "signup_date"],
        filters=["signup_date >= '2024-01-01'", "signup_date < '2024-02-01'"],
        grouping_dimensions=[],
        aggregation_logic="None",
        mathematical_calculations="Half-open date interval",
        dependencies=[],
        expected_output_grain="distinct_user_id",
        columns_produced=["user_id", "signup_date"],
        sql_fragment="SELECT user_id, signup_date FROM users WHERE signup_date >= '2024-01-01' AND signup_date < '2024-02-01'",
    )
    assert step.step_name == "jan_cohort"
    assert step.expected_output_grain == "distinct_user_id"
    assert "users" in step.input_tables
    # Test legacy aliases
    assert step.source_tables == step.input_tables
    assert step.math_logic == step.mathematical_calculations
    assert step.filter_conditions == step.filters
    assert step.group_by_columns == step.grouping_dimensions


@pytest.mark.parametrize(
    "bad_name",
    ["123bad", "bad name with spaces", "bad-hyphen", ""],
)
def test_cte_step_invalid_name_rejected(bad_name: str):
    """Verifies that invalid SQL identifiers in step_name are rejected."""
    with pytest.raises(ValueError):
        CTEPlanStep(
            step_name=bad_name,
            purpose="Testing bad name",
            expected_output_grain="test",
            sql_fragment="SELECT 1",
        )


def test_logic_plan_rejects_duplicate_cte_names():
    """Verifies that duplicate CTE names within a plan are caught and rejected."""
    step1 = CTEPlanStep(
        step_name="step_a",
        purpose="First step",
        expected_output_grain="grain_a",
        sql_fragment="SELECT 1 AS id",
    )
    step2 = CTEPlanStep(
        step_name="step_a",  # Duplicate!
        purpose="Second step with same name",
        expected_output_grain="grain_b",
        sql_fragment="SELECT 2 AS id",
    )
    with pytest.raises(ValueError, match="Duplicate CTE names"):
        LogicPlan(
            query="Test query",
            target_dialect="postgres",
            strategy="Test duplicate",
            cte_steps=[step1, step2],
            final_select="SELECT * FROM step_a",
        )


def test_logic_plan_rejects_invalid_final_select():
    """Verifies that final_select must contain a SELECT statement."""
    step1 = CTEPlanStep(
        step_name="step_a",
        purpose="First step",
        expected_output_grain="grain_a",
        sql_fragment="SELECT 1 AS id",
    )
    with pytest.raises(ValueError, match="final_select must contain a valid SELECT"):
        LogicPlan(
            query="Test query",
            target_dialect="postgres",
            strategy="Test final select",
            cte_steps=[step1],
            final_select="INSERT INTO tbl VALUES (1)",
        )


def test_dag_validator_detects_forward_reference():
    """Verifies that referencing a CTE defined later in sequence raises PlanValidationError."""
    step1 = CTEPlanStep(
        step_name="first_step",
        purpose="Should not reference second_step",
        dependencies=["second_step"],  # Forward reference!
        expected_output_grain="grain_a",
        sql_fragment="SELECT * FROM second_step",
    )
    step2 = CTEPlanStep(
        step_name="second_step",
        purpose="Defined later",
        dependencies=[],
        expected_output_grain="grain_b",
        sql_fragment="SELECT 1 AS val",
    )
    plan = LogicPlan(
        query="Test forward reference",
        strategy="Test forward reference",
        cte_steps=[step1, step2],
        final_select="SELECT * FROM second_step",
    )
    with pytest.raises(PlanValidationError, match="Forward-reference"):
        CTEDependencyValidator.validate_plan(plan)


def test_dag_validator_detects_self_dependency():
    """Verifies that a CTE step cannot depend on itself."""
    step = CTEPlanStep(
        step_name="recursive_bad",
        purpose="Self-referencing step",
        dependencies=["recursive_bad"],
        expected_output_grain="grain_a",
        sql_fragment="SELECT 1",
    )
    plan = LogicPlan(
        query="Test self dependency",
        strategy="Test self dependency",
        cte_steps=[step],
        final_select="SELECT * FROM recursive_bad",
    )
    with pytest.raises(PlanValidationError, match="cannot depend on itself"):
        CTEDependencyValidator.validate_plan(plan)


def test_cohort_retention_plan_generation_and_compilation(planner):
    """Verifies cohort retention question decomposes into a 4-step valid DAG and compiles to SQL."""
    q = "What percentage of users who signed up in January returned in February?"
    plan = planner.create_cohort_retention_plan(q, target_dialect="postgres")

    assert len(plan.cte_steps) == 4
    step_names = [s.step_name for s in plan.cte_steps]
    assert step_names == [
        "jan_signup_cohort",
        "distinct_jan_users",
        "feb_returning_cohort",
        "retention_metrics",
    ]

    # Verify dependency chain
    assert plan.cte_steps[0].dependencies == []
    assert plan.cte_steps[1].dependencies == ["jan_signup_cohort"]
    assert plan.cte_steps[2].dependencies == ["jan_signup_cohort"]
    assert set(plan.cte_steps[3].dependencies) == {"distinct_jan_users", "feb_returning_cohort"}

    # Compile to SQL
    compiled_sql = CTEPlanCompiler.compile(plan, target_dialect="postgres")
    assert "WITH" in compiled_sql
    assert "jan_signup_cohort AS (" in compiled_sql
    assert "distinct_jan_users AS (" in compiled_sql
    assert "feb_returning_cohort AS (" in compiled_sql
    assert "retention_metrics AS (" in compiled_sql
    assert "NULLIF(total_jan_signups, 0)" in compiled_sql

    # AST validation using SQLGlot
    report = validate_readonly_sql(compiled_sql, dialect="postgres")
    assert report.is_valid is True
    assert report.is_readonly is True
    assert "users" in report.tables_referenced
    assert "activity_events" in report.tables_referenced


def test_cohort_retention_plan_execution_on_local_benchmark(planner):
    """Executes the compiled CTE plan directly on the local benchmark database and checks exact accuracy."""
    q = "What percentage of users who signed up in January returned in February?"
    plan = planner.create_cohort_retention_plan(q, target_dialect="sqlite")
    compiled_sql = CTEPlanCompiler.compile(plan, target_dialect="sqlite")

    settings = get_settings()
    conn = sqlite3.connect(settings.local_db_path)
    cur = conn.cursor()
    cur.execute(compiled_sql)
    row = cur.fetchone()
    conn.close()

    assert row is not None
    retention_rate = float(row[0])
    # Benchmark ground truth is exactly 30.0%
    assert retention_rate == 30.0


def test_monthly_payouts_plan(planner):
    """Verifies multi-table monthly payouts plan compiles and separates loan tables into CTEs."""
    q = "Show monthly payouts across both group loans and individual loans"
    plan = planner.create_monthly_payouts_plan(q, target_dialect="postgres")

    assert len(plan.cte_steps) == 2
    assert plan.cte_steps[0].step_name == "group_monthly"
    assert plan.cte_steps[1].step_name == "individual_monthly"

    sql = CTEPlanCompiler.compile(plan, target_dialect="postgres")
    report = validate_readonly_sql(sql, dialect="postgres")
    assert report.is_valid is True
    assert "FULL OUTER JOIN" in sql or "FULL JOIN" in sql


def test_staff_attendance_plan(planner):
    """Verifies staff attendance plan separates denominator from June attendance numerator."""
    q = "What percentage of staff were present in June 2026?"
    plan = planner.create_staff_attendance_plan(q, target_dialect="postgres")

    assert len(plan.cte_steps) == 2
    assert plan.cte_steps[0].step_name == "all_staff_denominator"
    assert plan.cte_steps[1].step_name == "june_present_staff"

    sql = CTEPlanCompiler.compile(plan, target_dialect="postgres")
    report = validate_readonly_sql(sql, dialect="postgres")
    assert report.is_valid is True
    assert "NULLIF(d.total_staff_count, 0)" in sql

