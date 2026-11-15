"""Evaluation and regression test suite for Problem 2: Enterprise Schema Linking (Stage 7)."""

from __future__ import annotations

import duckdb
import pytest

from src.evaluator import (
    Problem2RetrievalEvaluator,
    RetrievalEvaluationReport,
    build_enterprise_benchmark_catalog,
)
from src.schema_context_builder import EnterpriseSchemaRetriever
from src.schema_inspector import SchemaInspector


def test_problem2_retrieval_evaluation_all_metrics():
    """Executes the full 20-table / 240+ column benchmark and verifies all 7 Stage-7 metrics."""
    report = Problem2RetrievalEvaluator.run_benchmark(embedding_backend="hash")
    assert isinstance(report, RetrievalEvaluationReport)

    # Large enterprise scale: 20 tables and 240+ columns indexed
    assert report.total_tables_indexed == 20
    assert report.total_columns_indexed >= 240
    assert report.total_cases == 6

    # All 7 Stage-7 metrics populated and within expected ranges
    assert report.mean_table_recall_at_k >= 0.80
    assert report.mean_column_recall_at_k >= 0.70
    assert 0.0 <= report.mean_table_precision_at_k <= 1.0
    assert 0.0 <= report.mean_column_precision_at_k <= 1.0
    assert report.mean_reciprocal_rank >= 0.80
    assert report.mean_token_reduction_pct >= 65.0  # Significant pruning vs 20-table dump
    assert report.mean_retrieval_latency_ms >= 0.0
    assert report.literal_resolution_accuracy == 1.0

    # Verify nested STRUCT/ARRAY & JSONB cases specifically retrieved flattened paths
    nested_case = next(c for c in report.case_results if c.case_id == "P2-02")
    assert "analytics.events" in nested_case.retrieved_tables
    assert any("event_params.key" in col for col in nested_case.retrieved_columns)
    assert any("event_params.value.int_value" in col for col in nested_case.retrieved_columns)

    json_case = next(c for c in report.case_results if c.case_id == "P2-03")
    assert "billing.payment_webhooks" in json_case.retrieved_tables
    assert any("payload.failure.reason_code" in col for col in json_case.retrieved_columns)

    # Scope discipline: disclaimer prevents overclaiming 27.6% global Spider 2.0 elimination
    assert "do not" in report.scope_disclaimer.lower()


def test_duckdb_schema_changes_refresh_and_missing_or_ambiguous_columns():
    """Tests live schema evolution, index refresh, missing columns, and ambiguous column qualification."""
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, status VARCHAR, created_at TIMESTAMP);
        CREATE TABLE refunds (refund_id INTEGER PRIMARY KEY, status VARCHAR, created_at TIMESTAMP);
        INSERT INTO orders VALUES (1, 'NY', TIMESTAMP '2026-01-01 00:00:00');
        INSERT INTO refunds VALUES (10, 'PENDING', TIMESTAMP '2026-01-02 00:00:00');
        """
    )

    inspector = SchemaInspector(target="duckdb", duckdb_conn=conn)
    retriever = EnterpriseSchemaRetriever(
        in_memory=True,
        embedding_backend="hash",
        collection_prefix="test_duck_refresh",
    )
    retriever.index_from_inspector(inspector=inspector, clear_existing=True, index_distinct_values=True)

    # Ambiguous columns ('status', 'created_at' exist in both orders and refunds) -> always table-qualified
    amb_ctx = retriever.retrieve_schema_context(
        query="Show order status and created_at",
        top_k_tables=2,
        top_k_columns_per_table=5,
        target_dialect="duckdb",
    )
    assert all("." in path for path in amb_ctx.selected_column_paths)

    # Missing column query must execute gracefully and only return real indexed columns
    missing_ctx = retriever.retrieve_schema_context(
        query="Show nonexistent_bitcoin_wallet_hash from orders",
        top_k_tables=2,
        top_k_columns_per_table=4,
        target_dialect="duckdb",
    )
    assert not any("nonexistent_bitcoin_wallet_hash" in c for c in missing_ctx.selected_column_paths)

    # Mutate live schema: add loyalty_tiers table, insert 'CA' status, and drop refunds
    conn.execute("DROP TABLE refunds;")
    conn.execute(
        """
        CREATE TABLE loyalty_tiers (
            tier_id INTEGER PRIMARY KEY,
            tier_name VARCHAR,
            discount_bps INTEGER
        );
        INSERT INTO loyalty_tiers VALUES (1, 'Platinum', 1500);
        INSERT INTO orders VALUES (2, 'CA', TIMESTAMP '2026-02-01 00:00:00');
        """
    )

    counts = retriever.refresh(inspector=inspector, dialect="duckdb")
    assert counts["tables"] == 2

    refreshed_tables = retriever.table_retriever.get_indexed_tables()
    assert not any("refunds" in t for t in refreshed_tables)
    assert any("loyalty_tiers" in t for t in refreshed_tables)

    # New categorical DB value 'CA' resolves California after refresh
    post_ctx = retriever.retrieve_schema_context(
        query="Find orders in California and loyalty discount_bps",
        top_k_tables=2,
        top_k_columns_per_table=5,
        target_dialect="duckdb",
        require_db_evidence_for_literals=True,
    )
    assert "loyalty_tiers" in post_ctx.selected_table_names
    assert any(m.database_literal == "CA" for m in post_ctx.resolved_literals)

