"""Integration tests for LangGraph pipeline orchestration, bounded error recovery, and execution reporting (Phase C)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from text2sql_engine.main import Text2SQLPipeline
from text2sql_engine.modules.problem1_planner.schemas import CTEPlanStep, LogicPlan
from text2sql_engine.modules.problem1_planner.logic_planner import LogicPlanner
from text2sql_engine.core.exceptions import PlanGenerationError


@pytest.fixture(scope="module")
def shared_pipeline():
    """Initializes a shared pipeline instance to reuse the bootstrapped schema retriever across tests."""
    return Text2SQLPipeline()


def test_pipeline_duckdb_successful_execution(shared_pipeline):
    """Tests full pipeline execution on DuckDB reporting EXECUTED status and row results."""
    result = shared_pipeline.run("Find top 3 customers ranked by total spend with safe division handling", target_dialect="duckdb")

    assert result.get("execution_status") == "EXECUTED"
    assert result.get("ast_valid") is True
    assert result.get("error") is None

    rows = result.get("execution_result")
    assert rows is not None
    assert len(rows) == 3
    assert "customer_id" in rows[0]
    assert "total_spent" in rows[0]


def test_pipeline_non_duckdb_unsupported_backend_reporting(shared_pipeline):
    """Tests that BigQuery/Snowflake/Postgres queries compile and verify AST but report UNSUPPORTED_EXECUTION_BACKEND."""
    result = shared_pipeline.run("Calculate days elapsed between orders and rank top transactions", target_dialect="bigquery")

    # A compiled query must NEVER masquerade as executed
    assert result.get("execution_status") == "UNSUPPORTED_EXECUTION_BACKEND"
    assert result.get("ast_valid") is True
    assert result.get("generated_sql") is not None
    assert "DATE_DIFF" in result.get("generated_sql")
    assert "Direct execution is unsupported" in result.get("error", "")


def test_pipeline_bounded_retries_terminate_on_persistent_failure(shared_pipeline):
    """Tests that persistent planner errors route through error_handler and stop at max_retries with FAILED status."""
    mock_planner = MagicMock(spec=LogicPlanner)
    mock_planner.generate_logic_plan.side_effect = PlanGenerationError("Simulated persistent model failure")

    pipeline = Text2SQLPipeline(planner=mock_planner, retriever=shared_pipeline.retriever)
    result = pipeline.run("Any query", target_dialect="duckdb", max_retries=2)

    assert result.get("execution_status") == "FAILED"
    assert result.get("retry_count") == 2
    assert "Simulated persistent model failure" in result.get("error", "")
    assert result.get("failed_stage") == "planning"


def test_pipeline_recovers_after_transient_failure(shared_pipeline):
    """Tests that a recoverable error triggers a retry with repair context and succeeds."""
    mock_planner = MagicMock(spec=LogicPlanner)

    valid_plan = LogicPlan(
        query="Test query",
        target_dialect="duckdb",
        strategy="Test strategy",
        cte_steps=[
            CTEPlanStep(
                step_name="cust_summary",
                purpose="Summary",
                source_tables=["customers"],
                math_logic="None",
                sql_fragment="SELECT customer_id, customer_name FROM customers",
            )
        ],
        final_select="SELECT customer_id, customer_name FROM cust_summary LIMIT 2",
    )

    # First attempt fails, second attempt succeeds
    mock_planner.generate_logic_plan.side_effect = [
        PlanGenerationError("Transient syntax glitch in CTE"),
        valid_plan,
    ]
    mock_planner.compile_sql.return_value = (
        "WITH cust_summary AS (SELECT customer_id, customer_name FROM customers) "
        "SELECT customer_id, customer_name FROM cust_summary LIMIT 2"
    )

    pipeline = Text2SQLPipeline(planner=mock_planner, retriever=shared_pipeline.retriever)
    result = pipeline.run("Test recovery query", target_dialect="duckdb", max_retries=2)

    assert result.get("execution_status") == "EXECUTED"
    assert result.get("retry_count") == 1
    assert result.get("error") is None
    assert len(result.get("execution_result", [])) == 2
