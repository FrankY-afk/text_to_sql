"""Unit and integration tests for Spider 2.0 Problem 3: Documentation Grounding & Autonomous Database Exploration."""

from __future__ import annotations

import pytest
from src.database_explorer import DatabaseExplorer, ProbeType
from src.doc_rag import BUILTIN_METRICS, DocumentationRetriever
from src.evaluator import Problem3Evaluator
from src.query_executor import MCPQueryExecutor
from src.sql_generator import SQLGenerator


@pytest.fixture
def doc_retriever():
    retriever = DocumentationRetriever(in_memory=True, embedding_backend="hash")
    retriever.ensure_indexed()
    return retriever


@pytest.fixture
def explorer():
    return DatabaseExplorer()


def test_doc_rag_builtin_metrics_indexing(doc_retriever):
    """Verifies that standard business metrics are loaded into catalog and vector index."""
    assert len(doc_retriever.metrics) >= 8
    assert "npa_90_plus" in doc_retriever.metrics
    assert "gross_loan_portfolio" in doc_retriever.metrics
    assert "effective_staff_attendance" in doc_retriever.metrics
    assert "rolling_30d_churn" in doc_retriever.metrics
    assert "net_disbursed_payout" in doc_retriever.metrics


def test_doc_rag_markdown_file_chunking(doc_retriever):
    """Verifies that markdown rulebooks are split into section chunks."""
    assert len(doc_retriever.chunks) > len(doc_retriever.metrics)
    doc_titles = {c.doc_title for c in doc_retriever.chunks}
    assert any("Lending" in t for t in doc_titles)
    assert any("Attendance" in t or "Operational" in t for t in doc_titles)


def test_doc_rag_hybrid_retrieval_and_prompt_formatting(doc_retriever):
    """Verifies hybrid retrieval matches questions and formats grounded prompt block."""
    # 1. Delinquent NPA query
    res_npa = doc_retriever.retrieve_documentation("What is the overdue volume for 90+ DPD NPA loans?")
    assert len(res_npa.matched_metrics) >= 1
    assert any(m.metric_id == "npa_90_plus" for m in res_npa.matched_metrics)
    assert "dpd >= 90" in res_npa.formatted_prompt_block

    # 2. Staff Attendance query
    res_att = doc_retriever.retrieve_documentation("What is the staff attendance rate across branches?")
    assert len(res_att.matched_metrics) >= 1
    assert any(m.metric_id == "effective_staff_attendance" for m in res_att.matched_metrics)
    assert "0.5" in res_att.formatted_prompt_block


def test_database_explorer_cardinality_and_date_bounds(explorer):
    """Verifies autonomous read-only diagnostic probes on local SQLite benchmark."""
    # 1. Cardinality probe
    p_card = explorer.probe_cardinality("users", target="local")
    assert p_card.probe_type == ProbeType.ROW_COUNT_CHECK
    assert p_card.result_data["row_count"] > 0
    assert not p_card.has_anomaly

    # 2. Date boundary probe
    p_date = explorer.probe_date_bounds("users", "signup_date", target="local")
    assert p_date.probe_type == ProbeType.DATE_BOUNDARY_PROBE
    assert "min_val" in p_date.result_data
    assert p_date.result_data["min_val"] <= p_date.result_data["max_val"]


def test_database_explorer_null_and_join_multiplicity(explorer):
    """Verifies null rate and Cartesian join multiplicity checks."""
    # 1. Null check on primary key
    p_null = explorer.probe_nullability("users", "user_id", target="local")
    assert p_null.probe_type == ProbeType.NULL_CHECK
    assert p_null.result_data["null_rows"] == 0

    # 2. Join multiplicity probe
    p_join = explorer.probe_join_multiplicity(
        "users", "activity_events", "a.user_id = b.user_id", target="local"
    )
    assert p_join.probe_type == ProbeType.JOIN_CARDINALITY_PROBE
    assert p_join.result_data["joined_cnt"] >= 0


def test_mcp_problem3_tools():
    """Verifies MCP server tools for Problem 3: retrieve_documentation, get_metric_definitions, explore_database."""
    executor = MCPQueryExecutor()

    # 1. retrieve_documentation
    res_doc = executor.retrieve_documentation_sync(query="Net disbursed payout after fees", top_k=2)
    assert "matched_metrics" in res_doc
    assert len(res_doc["matched_metrics"]) >= 1

    # 2. get_metric_definitions
    metrics = executor.get_metric_definitions_sync(domain="lending")
    assert isinstance(metrics, list)
    assert len(metrics) >= 2
    assert any(m["metric_id"] == "npa_90_plus" for m in metrics)

    # 3. explore_database
    res_exp = executor.explore_database_sync(
        user_query="What was the user retention between January and February?",
        target="local",
    )
    assert "probes_run" in res_exp
    assert len(res_exp["probes_run"]) >= 1


def test_problem3_evaluator_benchmark_full_suite():
    """Verifies Problem3Evaluator executes all 10 Spider 2.0 Problem 3 test cases."""
    report = Problem3Evaluator.run_benchmark(embedding_backend="hash")
    assert report.total_cases == 10
    assert report.doc_grounding_accuracy == 100.0
    assert report.problem3_compliance_rate == 100.0
    assert report.baseline_compliance_rate == 0.0
    assert len(report.case_results) == 10


def test_sql_generator_problem3_doc_grounding_integration():
    """Verifies that SQLGenerator attaches grounded metrics and prompt block from Problem 3."""
    gen = SQLGenerator(embedding_backend="hash")
    question = "What is the total overdue amount for 90+ DPD NPA loans?"
    result = gen.generate_improved_sql(question, target="local", use_hf=False)

    assert result.doc_context is not None
    assert len(result.doc_context) > 0
    assert any(m.metric_id == "npa_90_plus" for m in result.grounded_metrics)
