"""Unit tests for Problem 1: CTE Logic Planner, Dialect Function RAG, and SQL Compiler."""

from __future__ import annotations

import os
import pytest
import duckdb
import sqlglot

from text2sql_engine.modules.problem1_planner.schemas import (
    DialectFunctionSpec,
    CTEPlanStep,
    LogicPlan,
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


def test_pydantic_schemas_validation():
    """Tests that Pydantic models enforce schema contracts correctly."""
    spec = DialectFunctionSpec(
        name="date_diff",
        dialect="duckdb",
        category="datetime",
        syntax_pattern="date_diff('day', start_date, end_date)",
        description="Calculates day difference between two dates",
    )
    assert spec.dialect == "duckdb"
    assert spec.category == "datetime"

    step = CTEPlanStep(
        step_name="monthly_cohort",
        purpose="Group transactions by customer and calculate total spend",
        source_tables=["orders"],
        math_logic="SUM(total_amount) AS total_spent, COUNT(order_id) AS order_cnt",
        group_by_columns=["customer_id"],
        columns_produced=["customer_id", "total_spent", "order_cnt"],
        sql_fragment="SELECT customer_id, SUM(total_amount) AS total_spent FROM orders GROUP BY customer_id",
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


def test_dialect_function_rag_retrieval():
    """Tests that DialectFunctionRAG retrieves dialect-accurate functions."""
    rag = DialectFunctionRAG()

    # Test date difference across dialects
    duckdb_specs = rag.retrieve("calculate days between order and shipment", dialect="duckdb")
    bq_specs = rag.retrieve("calculate days between order and shipment", dialect="bigquery")
    sf_specs = rag.retrieve("calculate days between order and shipment", dialect="snowflake")
    pg_specs = rag.retrieve("calculate days between order and shipment", dialect="postgres")

    duckdb_func_names = [s.name for s in duckdb_specs]
    bq_func_names = [s.name for s in bq_specs]
    sf_func_names = [s.name for s in sf_specs]
    pg_func_names = [s.name for s in pg_specs]

    assert "date_diff" in duckdb_func_names
    assert "DATE_DIFF" in bq_func_names
    assert "DATEDIFF" in sf_func_names
    assert "date_diff_postgres" in pg_func_names

    # Test safe division retrieval
    div_specs = rag.retrieve("compute completion ratio and percentage", dialect="duckdb")
    div_names = [s.name for s in div_specs]
    assert "safe_divide" in div_names


def test_logic_planner_generates_modular_plan():
    """Tests that LogicPlanner decomposes queries into verified CTE steps."""
    planner = LogicPlanner()
    query = "Find the top 3 customers ranked by total spend with safe division"
    schema_context = "Table: customers (customer_id VARCHAR, customer_name VARCHAR)\nTable: orders (order_id VARCHAR, customer_id VARCHAR, total_amount DOUBLE)"

    plan = planner.generate_logic_plan(
        user_query=query,
        schema_context=schema_context,
        dialect="duckdb",
    )

    assert isinstance(plan, LogicPlan)
    assert len(plan.cte_steps) >= 1
    assert plan.target_dialect == "duckdb"
    assert "SELECT" in plan.final_select


def test_compile_sql_ast_and_transpilation():
    """Tests compilation and AST transpilation across SQL dialects (DuckDB, Postgres, BigQuery, Snowflake)."""
    planner = LogicPlanner()

    step = CTEPlanStep(
        step_name="customer_spend",
        purpose="Aggregate spend and compute rank",
        source_tables=["orders"],
        math_logic="SUM(total_amount) AS total_spent",
        sql_fragment="SELECT customer_id, SUM(total_amount) AS total_spent FROM orders GROUP BY customer_id",
    )
    plan = LogicPlan(
        query="Top spenders",
        target_dialect="duckdb",
        strategy="CTE aggregation and ranking",
        cte_steps=[step],
        final_select="SELECT customer_id, total_spent FROM customer_spend ORDER BY total_spent DESC LIMIT 3",
    )

    # 1. Compile for DuckDB
    duckdb_sql = planner.compile_sql(plan, target_dialect="duckdb")
    assert "WITH customer_spend AS" in duckdb_sql
    assert "LIMIT 3" in duckdb_sql

    # 2. Compile for PostgreSQL
    pg_sql = planner.compile_sql(plan, target_dialect="postgres")
    assert "customer_spend" in pg_sql
    # SQLGlot will parse and output valid Postgres AST
    assert sqlglot.parse_one(pg_sql, read="postgres") is not None

    # 3. Compile for BigQuery
    bq_sql = planner.compile_sql(plan, target_dialect="bigquery")
    assert "customer_spend" in bq_sql
    assert sqlglot.parse_one(bq_sql, read="bigquery") is not None

    # 4. Compile for Snowflake
    sf_sql = planner.compile_sql(plan, target_dialect="snowflake")
    assert "customer_spend" in sf_sql
    assert sqlglot.parse_one(sf_sql, read="snowflake") is not None


def test_problem1_execution_on_duckdb(duckdb_conn):
    """Executes compiled multi-stage CTE SQL directly on DuckDB sample database."""
    planner = LogicPlanner()

    # Step 1: Filter completed orders
    step1 = CTEPlanStep(
        step_name="completed_orders",
        purpose="Filter orders with COMPLETED status",
        source_tables=["orders"],
        math_logic="order_status = 'COMPLETED'",
        sql_fragment="SELECT customer_id, total_amount, shipping_cost FROM orders WHERE order_status = 'COMPLETED'",
    )
    # Step 2: Compute spend and net spend ratio
    step2 = CTEPlanStep(
        step_name="customer_aggregates",
        purpose="Compute net spend with division by zero guard",
        source_tables=["completed_orders"],
        math_logic="SUM(total_amount) as gross_spend, SUM(shipping_cost) / NULLIF(SUM(total_amount), 0) as shipping_ratio",
        sql_fragment="""SELECT 
            customer_id, 
            SUM(total_amount) AS gross_spend,
            ROUND(SUM(shipping_cost) * 1.0 / NULLIF(SUM(total_amount), 0), 4) AS shipping_ratio
        FROM completed_orders 
        GROUP BY customer_id""",
    )

    plan = LogicPlan(
        query="What is the gross spend and shipping ratio per customer for completed orders?",
        target_dialect="duckdb",
        strategy="Multi-stage CTE: 1) filter valid orders, 2) compute aggregates with safe division",
        cte_steps=[step1, step2],
        final_select="SELECT customer_id, gross_spend, shipping_ratio FROM customer_aggregates ORDER BY gross_spend DESC",
    )

    compiled_sql = planner.compile_sql(plan, target_dialect="duckdb")
    cursor_result = duckdb_conn.execute(compiled_sql).fetchall()

    assert len(cursor_result) > 0
    # C005 has 510 gross spend, C001 has 430 gross spend, C002 has 420, C003 has 90
    top_customer = cursor_result[0]
    assert top_customer[0] == "C005"
    assert top_customer[1] == 510.0
