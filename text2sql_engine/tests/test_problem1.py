"""Comprehensive Unit and Integration tests for Problem 1:
CTE Logic Planner, Dialect Function RAG, AST Dependency Validator, and SQL Compiler.
"""

from __future__ import annotations

import os
import pytest
import duckdb
import sqlglot
from pydantic import ValidationError

from text2sql_engine.core.config import (
    DeterministicMockCoder,
    get_llm,
    get_settings,
)
from text2sql_engine.core.exceptions import (
    ConfigurationError,
    DependencyValidationError,
    ModelUnavailableError,
    PlanGenerationError,
    PlanValidationError,
    SQLCompilationError,
    SecurityValidationError,
    UnsupportedDialectError,
)
from text2sql_engine.modules.problem1_planner.schemas import (
    DialectFunctionSpec,
    CTEPlanStep,
    LogicPlan,
    SUPPORTED_DIALECTS,
)
from text2sql_engine.modules.problem1_planner.dependency_validator import (
    CTEDependencyValidator,
)
from text2sql_engine.modules.problem1_planner.sql_compiler import (
    SQLCompiler,
)
from text2sql_engine.modules.problem1_planner.logic_planner import (
    LogicPlanner,
    DialectFunctionRAG,
)
from text2sql_engine.data.setup_sample_db import init_sample_database


@pytest.fixture(scope="module")
def duckdb_conn():
    """Provides an in-memory DuckDB database fixture populated with sample warehouse data."""
    con = init_sample_database()
    yield con
    con.close()


# ==============================================================================
# 1. Pydantic v2 Schema Contract & Integrity Validation Tests
# ==============================================================================

def test_pydantic_schemas_valid_construction():
    """Tests valid creation of Pydantic models with expected field values."""
    spec = DialectFunctionSpec(
        name="date_diff",
        dialect="duckdb",
        category="datetime",
        syntax_pattern="date_diff('day', start_date, end_date)",
        description="Calculates day difference between two dates",
        notes="DuckDB unit argument comes first",
    )
    assert spec.dialect == "duckdb"
    assert spec.category == "datetime"

    step = CTEPlanStep(
        step_name="monthly_cohort",
        purpose="Group transactions by customer and calculate total spend",
        source_tables=["orders"],
        math_logic="SUM(total_amount) AS total_spent, COUNT(order_id) AS order_cnt",
        filter_conditions=["order_status = 'COMPLETED'"],
        group_by_columns=["customer_id"],
        columns_produced=["customer_id", "total_spent", "order_cnt"],
        sql_fragment="SELECT customer_id, SUM(total_amount) AS total_spent, COUNT(order_id) AS order_cnt FROM orders WHERE order_status = 'COMPLETED' GROUP BY customer_id",
    )
    assert step.step_name == "monthly_cohort"

    plan = LogicPlan(
        query="Calculate total spend per customer and rank top 3",
        target_dialect="duckdb",
        strategy="1. Aggregate order totals per customer. 2. Filter top 3 customers.",
        cte_steps=[step],
        final_select="SELECT customer_id, total_spent FROM monthly_cohort ORDER BY total_spent DESC LIMIT 3",
        dialect_functions_used=[spec],
    )
    assert len(plan.cte_steps) == 1
    assert plan.target_dialect == "duckdb"


def test_schema_rejects_duplicate_cte_names():
    """Tests that duplicate CTE names are rejected with clear validation error."""
    step1 = CTEPlanStep(
        step_name="orders_cte",
        purpose="First step",
        math_logic="Direct select",
        sql_fragment="SELECT customer_id, total_amount FROM orders",
    )
    step2 = CTEPlanStep(
        step_name="orders_cte",  # Duplicate name!
        purpose="Second step with same name",
        math_logic="Direct select",
        sql_fragment="SELECT customer_id, total_amount FROM orders",
    )

    with pytest.raises(ValidationError) as exc_info:
        LogicPlan(
            query="Duplicate CTE query",
            target_dialect="duckdb",
            strategy="Test duplicate",
            cte_steps=[step1, step2],
            final_select="SELECT * FROM orders_cte",
        )
    assert "Duplicate CTE names" in str(exc_info.value)


def test_schema_rejects_unsupported_dialect():
    """Tests that unsupported dialects (e.g. 'oracle') are rejected by Pydantic."""
    with pytest.raises(ValidationError) as exc_info:
        DialectFunctionSpec(
            name="sysdate",
            dialect="oracle",  # Unsupported!
            category="datetime",
            syntax_pattern="SYSDATE",
            description="Oracle system date",
        )
    assert "Unsupported SQL dialect" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        LogicPlan(
            query="Test query",
            target_dialect="redshift",  # Unsupported!
            strategy="Strategy",
            final_select="SELECT 1",
        )
    assert "Unsupported SQL dialect" in str(exc_info.value)


def test_schema_rejects_empty_strings_and_invalid_identifiers():
    """Tests that whitespace-only strings and invalid SQL identifiers are rejected."""
    # Empty query
    with pytest.raises(ValidationError):
        LogicPlan(
            query="   ",
            target_dialect="duckdb",
            strategy="Strategy",
            final_select="SELECT 1",
        )

    # Invalid step_name (starting with digit or having special chars)
    with pytest.raises(ValidationError) as exc_info:
        CTEPlanStep(
            step_name="123_invalid_name",
            purpose="Valid purpose",
            math_logic="SUM(col)",
            sql_fragment="SELECT 1",
        )
    assert "Invalid CTE step_name" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        CTEPlanStep(
            step_name="invalid name with spaces",
            purpose="Valid purpose",
            math_logic="SUM(col)",
            sql_fragment="SELECT 1",
        )
    assert "Invalid CTE step_name" in str(exc_info.value)


def test_schema_rejects_missing_sql_fragment_in_plan():
    """Tests that missing or empty sql_fragment in a CTE step is rejected to avoid runtime failure."""
    step = CTEPlanStep(
        step_name="step1",
        purpose="Purpose",
        math_logic="Logic",
        sql_fragment=None,  # Missing!
    )
    with pytest.raises(ValidationError) as exc_info:
        LogicPlan(
            query="Query",
            target_dialect="duckdb",
            strategy="Strategy",
            cte_steps=[step],
            final_select="SELECT 1",
        )
    assert "missing or empty sql_fragment" in str(exc_info.value)


def test_schema_rejects_unexpected_fields():
    """Tests that extra unexpected fields are rejected (extra='forbid')."""
    with pytest.raises(ValidationError):
        CTEPlanStep(
            step_name="valid_step",
            purpose="Purpose",
            math_logic="Logic",
            sql_fragment="SELECT 1",
            unexpected_field="disallowed",  # Extra field!
        )


# ==============================================================================
# 2. Dialect Function RAG Tests
# ==============================================================================

def test_dialect_function_rag_retrieval():
    """Tests that DialectFunctionRAG retrieves dialect-accurate syntax patterns."""
    rag = DialectFunctionRAG()

    duckdb_specs = rag.retrieve("calculate days elapsed between order and delivery", dialect="duckdb")
    bq_specs = rag.retrieve("calculate days elapsed between order and delivery", dialect="bigquery")
    sf_specs = rag.retrieve("calculate days elapsed between order and delivery", dialect="snowflake")
    pg_specs = rag.retrieve("calculate days elapsed between order and delivery", dialect="postgres")

    duckdb_names = [s.name for s in duckdb_specs]
    bq_names = [s.name for s in bq_specs]
    sf_names = [s.name for s in sf_specs]
    pg_names = [s.name for s in pg_specs]

    assert "date_diff" in duckdb_names
    assert "DATE_DIFF" in bq_names
    assert "DATEDIFF" in sf_names
    assert "date_diff_postgres" in pg_names

    # Test safe division
    div_specs = rag.retrieve("compute completion ratio and rate with safe divide", dialect="duckdb")
    assert any(s.name == "safe_divide" for s in div_specs)

    # Test window and qualify
    window_specs = rag.retrieve("rank orders and filter top row using window", dialect="postgres")
    assert any("cte_window_filter" in s.name for s in window_specs)


# ==============================================================================
# 3. CTE Dependency & Semantic Validator Tests
# ==============================================================================

def test_dependency_validator_accepts_valid_plan():
    """Tests that a well-ordered, valid plan passes dependency validation."""
    step1 = CTEPlanStep(
        step_name="completed_orders",
        purpose="Filter completed orders",
        source_tables=["orders"],
        math_logic="None",
        columns_produced=["order_id", "customer_id", "total_amount"],
        sql_fragment="SELECT order_id, customer_id, total_amount FROM orders WHERE order_status = 'COMPLETED'",
    )
    step2 = CTEPlanStep(
        step_name="customer_aggregates",
        purpose="Aggregate spend per customer",
        source_tables=["completed_orders"],
        math_logic="SUM(total_amount)",
        columns_produced=["customer_id", "gross_spend"],
        sql_fragment="SELECT customer_id, SUM(total_amount) AS gross_spend FROM completed_orders GROUP BY customer_id",
    )
    plan = LogicPlan(
        query="Top customers",
        target_dialect="duckdb",
        strategy="Filter then aggregate",
        cte_steps=[step1, step2],
        final_select="SELECT customer_id, gross_spend FROM customer_aggregates ORDER BY gross_spend DESC LIMIT 5",
    )

    # Should pass without exception
    CTEDependencyValidator.validate_plan(plan, available_base_tables={"orders", "customers"})


def test_dependency_validator_rejects_forward_reference():
    """Tests that forward references (referencing a CTE defined later) are rejected."""
    # Step 1 references step 2 (defined later!)
    step1 = CTEPlanStep(
        step_name="step_one",
        purpose="First step",
        source_tables=["step_two"],
        math_logic="None",
        sql_fragment="SELECT customer_id FROM step_two",
    )
    step2 = CTEPlanStep(
        step_name="step_two",
        purpose="Second step",
        source_tables=["orders"],
        math_logic="None",
        sql_fragment="SELECT customer_id FROM orders",
    )
    plan = LogicPlan(
        query="Invalid forward reference plan",
        target_dialect="duckdb",
        strategy="Strategy",
        cte_steps=[step1, step2],
        final_select="SELECT customer_id FROM step_one",
    )

    with pytest.raises(DependencyValidationError) as exc_info:
        CTEDependencyValidator.validate_plan(plan, available_base_tables={"orders"})
    assert "forward reference" in str(exc_info.value)


def test_dependency_validator_rejects_recursive_self_reference():
    """Tests that self-referencing CTEs are rejected."""
    step1 = CTEPlanStep(
        step_name="infinite_loop",
        purpose="Self-referencing step",
        source_tables=["infinite_loop"],
        math_logic="None",
        sql_fragment="SELECT customer_id FROM infinite_loop",
    )
    plan = LogicPlan(
        query="Self-referencing query",
        target_dialect="duckdb",
        strategy="Strategy",
        cte_steps=[step1],
        final_select="SELECT customer_id FROM infinite_loop",
    )

    with pytest.raises(DependencyValidationError) as exc_info:
        CTEDependencyValidator.validate_plan(plan, available_base_tables={"orders"})
    assert "recursive self-reference" in str(exc_info.value)


def test_dependency_validator_rejects_unknown_table():
    """Tests that referencing a table not in schema or CTEs is rejected."""
    step1 = CTEPlanStep(
        step_name="valid_step",
        purpose="Step referencing nonexistent table",
        source_tables=["ghost_table"],
        math_logic="None",
        sql_fragment="SELECT col FROM ghost_table",
    )
    plan = LogicPlan(
        query="Unknown table query",
        target_dialect="duckdb",
        strategy="Strategy",
        cte_steps=[step1],
        final_select="SELECT col FROM valid_step",
    )

    with pytest.raises(DependencyValidationError) as exc_info:
        CTEDependencyValidator.validate_plan(plan, available_base_tables={"orders", "customers"})
    assert "unknown table or undefined CTE 'ghost_table'" in str(exc_info.value)


def test_dependency_validator_rejects_missing_column_reference():
    """Tests that downstream reference to an unavailable qualified column is rejected."""
    step1 = CTEPlanStep(
        step_name="customer_summary",
        purpose="Step producing only customer_id and total_amount",
        source_tables=["orders"],
        math_logic="None",
        columns_produced=["customer_id", "total_amount"],
        sql_fragment="SELECT customer_id, total_amount FROM orders",
    )
    step2 = CTEPlanStep(
        step_name="next_step",
        purpose="Step attempting to use customer_summary.secret_token which was not produced",
        source_tables=["customer_summary"],
        math_logic="None",
        columns_produced=["customer_id"],
        sql_fragment="SELECT customer_summary.customer_id, customer_summary.secret_token FROM customer_summary",
    )
    plan = LogicPlan(
        query="Missing column query",
        target_dialect="duckdb",
        strategy="Strategy",
        cte_steps=[step1, step2],
        final_select="SELECT customer_id FROM next_step",
    )

    with pytest.raises(DependencyValidationError) as exc_info:
        CTEDependencyValidator.validate_plan(plan, available_base_tables={"orders"})
    assert "secret_token" in str(exc_info.value)


# ==============================================================================
# 4. Dialect-Aware SQL Compiler Tests (Fail-Closed, Security, Transpilation)
# ==============================================================================

def test_sql_compiler_transpilation_across_dialects():
    """Tests SQL compilation and transpilation to DuckDB, Postgres, BigQuery, and Snowflake."""
    compiler = SQLCompiler()

    step = CTEPlanStep(
        step_name="customer_spend",
        purpose="Aggregate spend",
        source_tables=["orders"],
        math_logic="SUM(total_amount) AS total_spent",
        columns_produced=["customer_id", "total_spent"],
        sql_fragment="SELECT customer_id, SUM(total_amount) AS total_spent FROM orders GROUP BY customer_id",
    )
    plan = LogicPlan(
        query="Top spenders",
        target_dialect="duckdb",
        strategy="Strategy",
        cte_steps=[step],
        final_select="SELECT customer_id, total_spent FROM customer_spend ORDER BY total_spent DESC LIMIT 3",
    )

    # 1. DuckDB
    duckdb_sql = compiler.compile(plan, target_dialect="duckdb")
    assert "WITH customer_spend AS" in duckdb_sql
    assert sqlglot.parse_one(duckdb_sql, read="duckdb") is not None

    # 2. Postgres
    pg_sql = compiler.compile(plan, target_dialect="postgres")
    assert "customer_spend" in pg_sql
    assert sqlglot.parse_one(pg_sql, read="postgres") is not None

    # 3. BigQuery
    bq_sql = compiler.compile(plan, target_dialect="bigquery")
    assert "customer_spend" in bq_sql
    assert sqlglot.parse_one(bq_sql, read="bigquery") is not None

    # 4. Snowflake
    sf_sql = compiler.compile(plan, target_dialect="snowflake")
    assert "customer_spend" in sf_sql
    assert sqlglot.parse_one(sf_sql, read="snowflake") is not None


def test_sql_compiler_rejects_unsupported_dialect():
    """Tests that SQLCompiler fails closed when an unsupported dialect is passed."""
    compiler = SQLCompiler()
    plan = LogicPlan(
        query="Test query",
        target_dialect="duckdb",
        strategy="Strategy",
        final_select="SELECT 1",
    )

    with pytest.raises(UnsupportedDialectError) as exc_info:
        compiler.compile(plan, target_dialect="teradata")
    assert "Unsupported SQL dialect 'teradata'" in str(exc_info.value)


def test_sql_compiler_rejects_ddl_and_dml():
    """Tests that SQLCompiler strictly rejects DROP, INSERT, DELETE, and UPDATE statements."""
    compiler = SQLCompiler()

    # Attempted DROP TABLE injection
    step = CTEPlanStep(
        step_name="evil_cte",
        purpose="Attempted attack",
        math_logic="None",
        sql_fragment="DROP TABLE orders",
    )
    plan = LogicPlan(
        query="Drop table attack",
        target_dialect="duckdb",
        strategy="Strategy",
        cte_steps=[step],
        final_select="SELECT 1",
    )

    with pytest.raises(SecurityValidationError) as exc_info:
        compiler.compile(plan)
    assert "Only read-only SELECT queries are permitted" in str(exc_info.value) or "Forbidden SQL operation" in str(exc_info.value)


def test_sql_compiler_rejects_multiple_statements():
    """Tests that multi-statement queries (separated by semicolons) are rejected."""
    compiler = SQLCompiler()
    step = CTEPlanStep(
        step_name="multi_statement",
        purpose="Attempted multi statement",
        math_logic="None",
        sql_fragment="SELECT 1; SELECT 2",
    )
    plan = LogicPlan(
        query="Multi statement query",
        target_dialect="duckdb",
        strategy="Strategy",
        cte_steps=[step],
        final_select="SELECT 1",
    )

    with pytest.raises(SecurityValidationError) as exc_info:
        compiler.compile(plan)
    assert "Multiple SQL statements detected" in str(exc_info.value)


def test_sql_compiler_fails_closed_on_malformed_sql():
    """Tests that malformed SQL fails closed by raising SQLCompilationError rather than returning comments."""
    compiler = SQLCompiler()
    step = CTEPlanStep(
        step_name="broken_sql",
        purpose="Broken SQL",
        math_logic="None",
        sql_fragment="SELECT FROM WHERE ;;;",
    )
    plan = LogicPlan(
        query="Broken query",
        target_dialect="duckdb",
        strategy="Strategy",
        cte_steps=[step],
        final_select="SELECT * FROM broken_sql",
    )

    with pytest.raises(SQLCompilationError) as exc_info:
        compiler.compile(plan)
    assert "Failed to parse query" in str(exc_info.value)


def test_sql_compiler_safe_division_and_qualify_semantics():
    """Tests that safe division with NULLIF and QUALIFY rewrites are preserved."""
    compiler = SQLCompiler()

    step = CTEPlanStep(
        step_name="safe_math",
        purpose="Safe division and window ranking",
        source_tables=["orders"],
        math_logic="safe ratio",
        columns_produced=["customer_id", "ratio", "rank_num"],
        sql_fragment="SELECT customer_id, total_amount * 1.0 / NULLIF(shipping_cost, 0) AS ratio, ROW_NUMBER() OVER (ORDER BY total_amount DESC) AS rank_num FROM orders",
    )
    plan = LogicPlan(
        query="Safe math query",
        target_dialect="duckdb",
        strategy="Strategy",
        cte_steps=[step],
        final_select="SELECT customer_id, ratio FROM safe_math WHERE rank_num <= 3",
    )

    # Transpile to Postgres and verify NULLIF is preserved and parses cleanly
    pg_sql = compiler.compile(plan, target_dialect="postgres")
    assert "NULLIF" in pg_sql
    assert sqlglot.parse_one(pg_sql, read="postgres") is not None


# ==============================================================================
# 5. Integration Test: Four-Week Cohort Retention on DuckDB (Exact Verification)
# ==============================================================================

def test_four_week_cohort_retention_duckdb_execution():
    """Creates a deterministic event log and tests end-to-end execution of a 4-week cohort retention query.
    
    METRIC DEFINITION:
    - Cohort: Users grouped by their signup week (DATE_TRUNC('week', signup_date)).
    - Week 0: The signup week (week_number = 0).
    - Week N Retention: Count of distinct users from the original cohort who were active in week N,
      divided by the original cohort size (with safe division NULLIF(cohort_size, 0)).
    - Duplicate events within the same week for the same user count as 1 active user.
    - Users with no activity in a week remain in the original denominator.
    
    MANUALLY CALCULATED GROUND TRUTH (Cohort 2024-01-01, size = 5 users):
    - Week 0: 5 active users (U1, U2, U3, U4, U5) -> Retention: 5/5 = 1.0 (100.0%)
    - Week 1: 3 active users (U1, U2, U3)         -> Retention: 3/5 = 0.6 (60.0%)
    - Week 2: 2 active users (U1, U2)             -> Retention: 2/5 = 0.4 (40.0%)
    - Week 3: 3 active users (U1, U2, U3 returned!)-> Retention: 3/5 = 0.6 (60.0%)
    - Week 4: 2 active users (U1, U2)             -> Retention: 2/5 = 0.4 (40.0%)
    """
    con = duckdb.connect(":memory:")

    # 1. Seed user signups
    con.execute("""
    CREATE TABLE user_signups (
        user_id VARCHAR PRIMARY KEY,
        signup_date DATE
    );
    """)
    con.execute("""
    INSERT INTO user_signups VALUES
        ('U1', DATE '2024-01-01'),
        ('U2', DATE '2024-01-02'),
        ('U3', DATE '2024-01-03'),
        ('U4', DATE '2024-01-04'),
        ('U5', DATE '2024-01-05');
    """)

    # 2. Seed activity events (including edge cases: multiple events per user in a week, dormant users)
    con.execute("""
    CREATE TABLE activity_events (
        event_id VARCHAR PRIMARY KEY,
        user_id VARCHAR,
        event_date DATE
    );
    """)
    con.execute("""
    INSERT INTO activity_events VALUES
        -- Week 0 (2024-01-01 to 2024-01-07): U1, U2, U3, U4, U5 (5 users, duplicate events for U1)
        ('E01', 'U1', DATE '2024-01-01'),
        ('E02', 'U1', DATE '2024-01-03'),
        ('E03', 'U2', DATE '2024-01-02'),
        ('E04', 'U3', DATE '2024-01-04'),
        ('E05', 'U4', DATE '2024-01-05'),
        ('E06', 'U5', DATE '2024-01-05'),

        -- Week 1 (2024-01-08 to 2024-01-14): U1, U2, U3 (3 users, duplicate events for U2)
        ('E07', 'U1', DATE '2024-01-09'),
        ('E08', 'U2', DATE '2024-01-10'),
        ('E09', 'U2', DATE '2024-01-12'),
        ('E10', 'U3', DATE '2024-01-11'),

        -- Week 2 (2024-01-15 to 2024-01-21): U1, U2 (2 users, U3 is inactive this week)
        ('E11', 'U1', DATE '2024-01-16'),
        ('E12', 'U2', DATE '2024-01-18'),

        -- Week 3 (2024-01-22 to 2024-01-28): U1, U2, U3 (3 users, U3 returns!)
        ('E13', 'U1', DATE '2024-01-23'),
        ('E14', 'U2', DATE '2024-01-24'),
        ('E15', 'U3', DATE '2024-01-26'),

        -- Week 4 (2024-01-29 to 2024-02-04): U1, U2 (2 users, duplicate events for U1)
        ('E16', 'U1', DATE '2024-01-30'),
        ('E17', 'U1', DATE '2024-02-01'),
        ('E18', 'U2', DATE '2024-01-31');
    """)

    # 3. Construct multi-stage LogicPlan
    planner = LogicPlanner()

    step1 = CTEPlanStep(
        step_name="user_cohorts",
        purpose="Assign users to their signup week cohort",
        source_tables=["user_signups"],
        math_logic="CAST(DATE_TRUNC('week', signup_date) AS DATE) AS cohort_week",
        columns_produced=["user_id", "cohort_week"],
        sql_fragment="SELECT user_id, CAST(DATE_TRUNC('week', signup_date) AS DATE) AS cohort_week FROM user_signups",
    )
    step2 = CTEPlanStep(
        step_name="cohort_sizes",
        purpose="Calculate the total distinct user denominator per cohort",
        source_tables=["user_cohorts"],
        math_logic="COUNT(DISTINCT user_id) AS cohort_size",
        group_by_columns=["cohort_week"],
        columns_produced=["cohort_week", "cohort_size"],
        sql_fragment="SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_size FROM user_cohorts GROUP BY cohort_week",
    )
    step3 = CTEPlanStep(
        step_name="weekly_activities",
        purpose="Compute elapsed weeks from cohort signup week for each activity event",
        source_tables=["activity_events", "user_cohorts"],
        math_logic="DATE_DIFF('week', uc.cohort_week, a.event_date) AS week_number",
        filter_conditions=["a.event_date >= uc.cohort_week"],
        group_by_columns=["a.user_id", "uc.cohort_week"],
        columns_produced=["user_id", "cohort_week", "week_number"],
        sql_fragment="""SELECT 
            a.user_id, 
            uc.cohort_week, 
            CAST(DATE_DIFF('week', CAST(uc.cohort_week AS DATE), CAST(DATE_TRUNC('week', a.event_date) AS DATE)) AS INTEGER) AS week_number 
        FROM activity_events AS a 
        JOIN user_cohorts AS uc 
            ON a.user_id = uc.user_id 
        WHERE a.event_date >= uc.cohort_week 
        GROUP BY a.user_id, uc.cohort_week, DATE_TRUNC('week', a.event_date)""",
    )
    step4 = CTEPlanStep(
        step_name="cohort_retention_agg",
        purpose="Count distinct active users per cohort and week number up to 4 weeks",
        source_tables=["weekly_activities"],
        math_logic="COUNT(DISTINCT wa.user_id) AS active_users",
        filter_conditions=["wa.week_number <= 4"],
        group_by_columns=["wa.cohort_week", "wa.week_number"],
        columns_produced=["cohort_week", "week_number", "active_users"],
        sql_fragment="""SELECT 
            wa.cohort_week, 
            wa.week_number, 
            COUNT(DISTINCT wa.user_id) AS active_users 
        FROM weekly_activities AS wa 
        WHERE wa.week_number <= 4 
        GROUP BY wa.cohort_week, wa.week_number""",
    )

    plan = LogicPlan(
        query="Calculate 4-week cohort retention rates with safe division",
        target_dialect="duckdb",
        strategy="Multi-stage CTE reasoning: cohorts -> sizes -> activities -> retention aggregation",
        cte_steps=[step1, step2, step3, step4],
        final_select="""SELECT 
            cs.cohort_week, 
            cra.week_number, 
            cs.cohort_size, 
            cra.active_users, 
            ROUND(cra.active_users * 1.0 / NULLIF(cs.cohort_size, 0), 4) AS retention_rate 
        FROM cohort_retention_agg AS cra 
        JOIN cohort_sizes AS cs 
            ON cra.cohort_week = cs.cohort_week 
        ORDER BY cs.cohort_week, cra.week_number""",
    )

    # 4. Validate dependencies
    CTEDependencyValidator.validate_plan(
        plan,
        available_base_tables={"user_signups", "activity_events"},
    )

    # 5. Compile query
    compiled_sql = planner.compile_sql(plan, target_dialect="duckdb")
    assert "WITH user_cohorts AS" in compiled_sql
    assert "cohort_sizes AS" in compiled_sql
    assert "weekly_activities AS" in compiled_sql
    assert "cohort_retention_agg AS" in compiled_sql
    assert "NULLIF" in compiled_sql

    # 6. Execute against DuckDB and verify exact deterministic results
    results = con.execute(compiled_sql).fetchall()
    con.close()

    assert len(results) == 5, f"Expected 5 weeks of retention data, got {len(results)}"

    expected_retention = [
        # (week_number, cohort_size, active_users, retention_rate)
        (0, 5, 5, 1.0),
        (1, 5, 3, 0.6),
        (2, 5, 2, 0.4),
        (3, 5, 3, 0.6),
        (4, 5, 2, 0.4),
    ]

    for row, expected in zip(results, expected_retention):
        # row: (cohort_week, week_number, cohort_size, active_users, retention_rate)
        week_num = row[1]
        cohort_size = row[2]
        active_users = row[3]
        retention_rate = row[4]

        exp_week, exp_size, exp_active, exp_rate = expected
        assert week_num == exp_week, f"Week number mismatch: expected {exp_week}, got {week_num}"
        assert cohort_size == exp_size, f"Cohort size mismatch at week {week_num}: expected {exp_size}, got {cohort_size}"
        assert active_users == exp_active, f"Active users mismatch at week {week_num}: expected {exp_active}, got {active_users}"
        assert pytest.approx(retention_rate, 0.0001) == exp_rate, f"Retention rate mismatch at week {week_num}: expected {exp_rate}, got {retention_rate}"


# ==============================================================================
# 6. Backend Distinction, Mock Verification, and Error Path Tests
# ==============================================================================

def test_explicit_mock_coder_distinction():
    """Tests that the mock coder explicitly identifies itself as a mock backend."""
    mock_llm = DeterministicMockCoder()
    assert mock_llm.is_mock is True
    assert mock_llm.backend_name == "mock"

    planner = LogicPlanner(llm=mock_llm)
    schema_context = "Table: orders (order_id VARCHAR, customer_id VARCHAR, total_amount DOUBLE)\nTable: customers (customer_id VARCHAR, state_code VARCHAR)"
    plan = planner.generate_logic_plan(
        user_query="Find top 3 customers ranked by total spend with safe division",
        schema_context=schema_context,
        dialect="duckdb",
    )
    assert isinstance(plan, LogicPlan)
    assert plan.planner_backend == "mock"
    assert len(plan.cte_steps) >= 1


def test_real_model_unavailable_raises_clear_error():
    """Tests that requesting an unavailable real backend fails closed with ModelUnavailableError."""
    # When requesting Ollama with unreachable endpoint or when offline
    with pytest.raises(ModelUnavailableError) as exc_info:
        get_llm(backend="ollama")
    assert "Ollama backend requested" in str(exc_info.value) or "not reachable" in str(exc_info.value)


def test_huggingface_missing_token_raises_configuration_error():
    """Tests that requesting HuggingFace backend without token raises ConfigurationError."""
    # Temporarily remove tokens if set
    old_hf = os.environ.pop("HF_TOKEN", None)
    old_hfb = os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            get_llm(backend="huggingface")
        assert "no API token provided" in str(exc_info.value)
    finally:
        if old_hf:
            os.environ["HF_TOKEN"] = old_hf
        if old_hfb:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = old_hfb


def test_malformed_llm_output_fails_closed():
    """Tests that malformed or non-JSON output from an LLM raises PlanGenerationError without silent fallback."""
    class BrokenLLM:
        backend_name = "broken"
        is_mock = False
        def invoke(self, messages):
            class Response:
                content = "I cannot generate SQL for you because I am just an AI."
            return Response()

    planner = LogicPlanner(llm=BrokenLLM())
    with pytest.raises(PlanGenerationError) as exc_info:
        planner.generate_logic_plan(
            user_query="Calculate sales metrics",
            schema_context="Table: orders (id INT)",
            dialect="duckdb",
        )
    assert "Failed to generate valid LogicPlan" in str(exc_info.value)
