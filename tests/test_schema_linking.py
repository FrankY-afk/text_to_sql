"""Comprehensive tests for Problem 2: Two-Stage Schema Linking (Stages 1, 2, 4, 5, 6)."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import duckdb
import pytest

from src.column_retriever import ColumnRetriever
from src.cte_planner import CTEPlanner
from src.hf_client import HuggingFaceClient
from src.literal_resolver import LiteralResolver, format_safe_sql_predicate
from src.query_executor import MCPQueryExecutor
from src.schema_context_builder import EnterpriseSchemaRetriever, SchemaContext
from src.schema_inspector import (
    ColumnMetadata,
    ForeignKeyMetadata,
    SchemaInspector,
    TableMetadata,
    validate_sql_identifier,
)
from src.table_retriever import TableRetriever


def _make_wide_enterprise_tables(num_tables: int = 12, extra_cols_per_table: int = 20) -> list[TableMetadata]:
    """Generates a multi-schema enterprise catalog with >250 columns and similar/ambiguous names."""
    domains = [
        ("crm", "customer_accounts", "Customer accounts, state_code, and annual contract value", "account_id", ["CA", "NY", "TX"]),
        ("crm", "support_tickets", "Customer support tickets and SLA breaches", "ticket_id", ["P1", "P2"]),
        ("billing", "invoices", "Subscription invoices and total billed USD", "invoice_id", ["PAID", "OVERDUE"]),
        ("billing", "payment_refunds", "Refund transactions and chargeback reason codes", "refund_id", ["APPROVED"]),
        ("hr", "employees", "Corporate employee directory and department assignments", "employee_id", ["Engineering", "Risk"]),
        ("hr", "employee_attendance", "Daily staff attendance and leave codes", "attendance_id", ["PR", "CL"]),
        ("inventory", "warehouses", "Regional warehouses and storage capacity", "warehouse_id", ["US", "IN"]),
        ("inventory", "shipments", "Carrier freight deliveries and tracking status", "shipment_id", ["DELIVERED"]),
        ("analytics", "web_sessions", "User browsing sessions and acquisition channel", "session_id", ["mobile"]),
        ("analytics", "funnel_steps", "Conversion funnel daily user counts", "step_id", ["checkout"]),
        ("risk", "fraud_alerts", "High-risk payment anomaly alerts", "alert_id", ["HIGH"]),
        ("risk", "kyc_verifications", "Customer identity KYC review records", "kyc_id", ["VERIFIED"]),
    ]

    tables: list[TableMetadata] = []
    for schema, tname, desc, pk, samples in domains[:num_tables]:
        cols = [
            ColumnMetadata(name=pk, data_type="VARCHAR(36)", is_primary_key=True, comment=f"Primary key for {tname}"),
            # Deliberately include similar/ambiguous columns across every table
            ColumnMetadata(name="status", data_type="VARCHAR(20)", sample_values=samples, comment=f"{tname} lifecycle status"),
            ColumnMetadata(name="created_at", data_type="TIMESTAMP", comment=f"Creation timestamp for {tname}"),
            ColumnMetadata(name="updated_at", data_type="TIMESTAMP"),
        ]
        if tname == "customer_accounts":
            cols.extend(
                [
                    ColumnMetadata(name="account_name", data_type="VARCHAR(120)", comment="Customer legal name"),
                    ColumnMetadata(name="state_code", data_type="VARCHAR(4)", sample_values=["CA", "NY", "TX"], comment="Two-letter US state abbreviation"),
                    ColumnMetadata(name="annual_contract_value", data_type="NUMERIC(14,2)", comment="Annual contract revenue USD"),
                ]
            )
        elif tname == "invoices":
            cols.extend(
                [
                    ColumnMetadata(name="account_id", data_type="VARCHAR(36)", is_foreign_key=True, foreign_key_target="crm.customer_accounts.account_id"),
                    ColumnMetadata(name="invoice_amount_usd", data_type="NUMERIC(12,2)", comment="Total invoice amount USD"),
                    ColumnMetadata(name="due_date", data_type="DATE"),
                ]
            )
        for idx in range(extra_cols_per_table):
            cols.append(ColumnMetadata(name=f"attr_col_{idx:02d}", data_type="VARCHAR(64)", comment=f"Auxiliary metadata column {idx}"))

        fks = []
        if tname == "invoices":
            fks.append(
                ForeignKeyMetadata(
                    source_schema="billing",
                    source_table="invoices",
                    source_column="account_id",
                    target_schema="crm",
                    target_table="customer_accounts",
                    target_column="account_id",
                )
            )

        tables.append(
            TableMetadata(
                schema_name=schema,
                table_name=tname,
                full_name=f"{schema}.{tname}",
                description=desc,
                primary_keys=[pk],
                foreign_keys=fks,
                columns=cols,
            )
        )
    return tables


def test_stage1_table_retriever_multischema_and_refresh():
    """Tests Stage 1 top-k retrieval, similarity scores, multi-schema filtering, and schema refresh."""
    tables = _make_wide_enterprise_tables(num_tables=8, extra_cols_per_table=5)
    retriever = TableRetriever(in_memory=True, collection_name="test_tbl_stage1", embedding_backend="hash")
    count = retriever.index_tables(tables)
    assert count == 8

    # Query scoped to billing schema
    billing_hits = retriever.retrieve_tables("total invoice amount USD overdue", top_k=2, schemas=["billing"])
    assert len(billing_hits) >= 1
    assert all(hit.schema_name == "billing" for hit in billing_hits)
    assert billing_hits[0].table_name == "invoices"
    assert 0.0 <= billing_hits[0].similarity_score <= 1.0

    # Schema change: drop invoices, add tax_filings -> refresh_index updates Chroma cleanly
    updated_tables = [t for t in tables if t.table_name != "invoices"] + [
        TableMetadata(
            schema_name="billing",
            table_name="tax_filings",
            full_name="billing.tax_filings",
            description="Quarterly VAT and sales tax filing summaries",
            primary_keys=["filing_id"],
            columns=[
                ColumnMetadata(name="filing_id", data_type="VARCHAR(36)", is_primary_key=True),
                ColumnMetadata(name="vat_payable_usd", data_type="NUMERIC(12,2)"),
            ],
        )
    ]
    retriever.refresh_index(tables=updated_tables)
    refreshed_names = retriever.get_indexed_tables()
    assert "billing.invoices" not in refreshed_names
    assert "billing.tax_filings" in refreshed_names

    tax_hits = retriever.retrieve_tables("Quarterly VAT tax filing payable", top_k=2, schemas=["billing"])
    assert tax_hits[0].table_name == "tax_filings"


def test_stage2_column_retriever_prunes_wide_tables_and_preserves_keys():
    """Tests Stage 2 pruning on 250+ column enterprise schema while preserving PK/FK join keys."""
    tables = _make_wide_enterprise_tables(num_tables=12, extra_cols_per_table=22)
    total_cols = sum(len(t.columns) for t in tables)
    assert total_cols > 280

    col_ret = ColumnRetriever(in_memory=True, collection_name="test_col_stage2", embedding_backend="hash")
    indexed = col_ret.index_tables(tables)
    assert indexed == total_cols

    selected = col_ret.retrieve_columns(
        query="Find overdue invoices total amount USD for customer account",
        candidate_tables=["billing.invoices"],
        top_k_per_table=5,
        preserve_keys=True,
    )
    selected_paths = {c.path for c in selected}

    # Pruned far below all 29 columns in billing.invoices
    assert len(selected) < len(tables[2].columns)
    assert "invoice_amount_usd" in selected_paths
    # PK and FK preserved automatically for join connectivity
    assert "invoice_id" in selected_paths
    assert "account_id" in selected_paths


def test_stage4_literal_resolver_db_evidence_and_sql_safety():
    """Verifies DB-evidenced resolution ('California' -> 'CA' only when 'CA' exists), refresh, and injection guards."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE customer_accounts (account_id TEXT PRIMARY KEY, state_code TEXT)")
    conn.execute("INSERT INTO customer_accounts VALUES ('a1', 'NY'), ('a2', 'TX')")
    conn.commit()

    # Check identifier sanitizer blocks unsafe SQL strings
    assert validate_sql_identifier("state_code") == "state_code"
    with pytest.raises(ValueError):
        validate_sql_identifier("state_code; DROP TABLE users--")

    # Quote escaping in predicate formatter
    assert format_safe_sql_predicate("account_name", "O'Reilly Auto") == "account_name = 'O''Reilly Auto'"

    # Index distinct values before 'CA' exists -> 'California' must NOT resolve to 'CA'
    resolver = LiteralResolver(target="local", in_memory=True, collection_name="test_lit_stage4", embedding_backend="hash")
    vals_before = []
    cur = conn.execute("SELECT DISTINCT state_code FROM customer_accounts")
    vals_before = [r[0] for r in cur.fetchall()]
    resolver.index_column_values("customer_accounts", "state_code", vals_before)

    neg_matches = resolver.resolve_literals("Show accounts in California", require_db_evidence=True)
    assert all(m.database_literal != "CA" for m in neg_matches)

    # Now add 'CA' to DB evidence & refresh index -> 'California' resolves cleanly with evidence
    conn.execute("INSERT INTO customer_accounts VALUES ('a3', 'CA')")
    conn.commit()
    vals_after = [r[0] for r in conn.execute("SELECT DISTINCT state_code FROM customer_accounts").fetchall()]
    resolver.clear_index()
    resolver.index_column_values("customer_accounts", "state_code", vals_after)

    pos_matches = resolver.resolve_literals("Show accounts in California", require_db_evidence=True)
    ca_matches = [m for m in pos_matches if m.database_literal == "CA"]
    assert len(ca_matches) == 1
    assert ca_matches[0].sql_predicate == "state_code = 'CA'"
    assert ca_matches[0].confidence >= 0.9
    assert "CA" in ca_matches[0].evidence


def test_stage5_and_stage6_schema_context_and_generation_integration():
    """Verifies EnterpriseSchemaRetriever -> SchemaContext -> CTEPlanner -> HuggingFace prompt integration."""
    tables = _make_wide_enterprise_tables(num_tables=12, extra_cols_per_table=20)
    retriever = EnterpriseSchemaRetriever(
        in_memory=True,
        embedding_backend="hash",
        collection_prefix="test_e2e_linking",
    )
    retriever.index_tables(tables, clear_existing=True)

    question = "Show total overdue invoice amount USD for enterprise accounts in California"
    ctx = retriever.retrieve_schema_context(
        query=question,
        top_k_tables=3,
        top_k_columns_per_table=6,
        target_dialect="postgres",
        require_db_evidence_for_literals=True,
    )
    assert isinstance(ctx, SchemaContext)
    assert ctx.token_reduction_pct > 50.0  # Significant token pruning vs 12x26 column schema
    assert ctx.pruned_schema_token_estimate < ctx.full_schema_token_estimate
    assert any(m.database_literal == "CA" for m in ctx.resolved_literals)
    assert any("customer_accounts" in t for t in ctx.selected_table_names)

    # Integrate with CTEPlanner (no longer falls back to hardcoded 'outlets')
    planner = CTEPlanner()
    plan = planner.plan_question(question, target_dialect="postgres", schema_context=ctx)
    assert plan.planner_backend == "schema_linked"
    assert "outlets" not in plan.cte_steps[0].input_tables

    # Integrate with SQLGenerator + mocked Hugging Face client
    mock_hf = MagicMock(spec=HuggingFaceClient)
    mock_hf.is_configured.return_value = True
    mock_hf.model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
    mock_hf.generate_sql.return_value = (
        "WITH ca_accounts AS (\n"
        "    SELECT account_id FROM crm.customer_accounts WHERE state_code = 'CA'\n"
        ")\n"
        "SELECT COUNT(*) AS cnt FROM ca_accounts;"
    )

    from src.sql_generator import SQLGenerator

    gen = SQLGenerator(hf_client=mock_hf, schema_retriever=retriever, embedding_backend="hash")
    res = gen.generate_improved_sql(question, target="postgres", use_hf=True)
    assert res.schema_context is not None
    assert res.schema_context.token_reduction_pct > 50.0
    # Verify prompt sent to HF received pruned schema context & CA literal instead of entire schema
    sent_prompt = mock_hf.generate_sql.call_args[0][0]
    assert "Pruned Schema Context" in sent_prompt
    assert "state_code = 'CA'" in sent_prompt


def test_mcp_relationship_and_schema_linking_tools():
    """Verifies new MCP tools get_table_relationships and retrieve_schema_context preserve MCP boundaries."""
    executor = MCPQueryExecutor()
    rel = executor.get_table_relationships_sync(target="local")
    assert "users" in " ".join(rel["primary_keys"].keys())

    ctx_dict = executor.retrieve_schema_context_sync(
        user_query="Count January signups",
        target="local",
        top_k_tables=2,
        top_k_columns_per_table=4,
    )
    assert "users" in ctx_dict["selected_table_names"]
    assert ctx_dict["pruned_schema_token_estimate"] > 0


def test_real_bge_small_embedding_smoke_retrieval():
    """End-to-end smoke test verifying cached BAAI/bge-small-en-v1.5 embeddings work with Chroma."""
    tables = _make_wide_enterprise_tables(num_tables=4, extra_cols_per_table=3)
    retriever = EnterpriseSchemaRetriever(
        in_memory=True,
        embedding_backend="bge",
        collection_prefix="test_real_bge",
    )
    retriever.index_tables(tables, clear_existing=True)

    ctx = retriever.retrieve_schema_context(
        query="Which customer organizations in California generated unpaid invoices?",
        top_k_tables=2,
        top_k_columns_per_table=5,
        target_dialect="postgres",
    )
    assert "customer_accounts" in ctx.selected_table_names or "invoices" in ctx.selected_table_names
    assert any(m.database_literal == "CA" for m in ctx.resolved_literals)

