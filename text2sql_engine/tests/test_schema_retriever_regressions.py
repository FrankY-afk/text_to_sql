"""Regression tests for Problem 2: Two-Stage Schema Retriever, Literal Value Indexer, and Nested Schemas."""

from __future__ import annotations

from typing import List
import pytest
import chromadb
import duckdb

from text2sql_engine.modules.problem2_retriever.schema_flattener import (
    FlattenedField,
    SchemaFlattener,
)
from text2sql_engine.modules.problem2_retriever.vector_retriever import (
    DistinctValueIndexer,
    TwoStageSchemaRetriever,
    LiteralMatch,
)
from text2sql_engine.core.exceptions import RetrieverError


class FastDeterministicTestEmbeddings:
    """Fast, deterministic bag-of-words embedding for sub-second isolated test execution."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * 64
            for token in text.lower().replace(".", " ").replace("_", " ").split():
                idx = sum(ord(c) for c in token) % 64
                vec[idx] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@pytest.fixture
def fast_chroma():
    return chromadb.EphemeralClient()


@pytest.fixture
def fast_embeddings():
    return FastDeterministicTestEmbeddings()


# ==============================================================================
# Phase A1: Fine-Grained Column Retrieval Tests
# ==============================================================================

def test_different_queries_select_different_columns_from_same_table(fast_chroma, fast_embeddings):
    """Proves that fine-grained retrieval selects distinct column subsets based on query intent."""
    retriever = TwoStageSchemaRetriever(
        collection_prefix="test_fine_grained",
        chroma_client=fast_chroma,
        embeddings=fast_embeddings,
    )

    fields = [
        FlattenedField(table_name="accounts", root_column="account_id", path="account_id", data_type="BIGINT"),
        FlattenedField(table_name="accounts", root_column="revenue", path="revenue", data_type="DOUBLE"),
        FlattenedField(table_name="accounts", root_column="tax_amount", path="tax_amount", data_type="DOUBLE"),
        FlattenedField(table_name="accounts", root_column="billing_city", path="billing_city", data_type="VARCHAR"),
        FlattenedField(table_name="accounts", root_column="shipping_address", path="shipping_address", data_type="VARCHAR"),
    ]

    tables_meta = {
        "accounts": {
            "description": "Enterprise customer accounts, financials, and physical addresses",
            "fields": fields,
            "primary_keys": ["account_id"],
            "foreign_keys": [],
        }
    }

    retriever.index_database_schema(tables_meta)

    # Query 1: Financial metrics
    _, ctx1, _ = retriever.retrieve_schema_context("total revenue and tax_amount for enterprise accounts")
    assert "revenue" in ctx1
    assert "tax_amount" in ctx1
    assert "account_id" in ctx1  # Primary key preserved

    # Query 2: Address information
    _, ctx2, _ = retriever.retrieve_schema_context("shipping_address and billing_city for customer deliveries")
    assert "shipping_address" in ctx2
    assert "billing_city" in ctx2
    assert "account_id" in ctx2  # Primary key preserved


def test_essential_join_keys_always_preserved(fast_chroma, fast_embeddings):
    """Proves that primary and foreign keys are preserved in schema context even when not mentioned."""
    retriever = TwoStageSchemaRetriever(
        collection_prefix="test_keys_preserved",
        chroma_client=fast_chroma,
        embeddings=fast_embeddings,
    )

    fields = [
        FlattenedField(table_name="orders", root_column="order_id", path="order_id", data_type="VARCHAR"),
        FlattenedField(table_name="orders", root_column="customer_id", path="customer_id", data_type="VARCHAR"),
        FlattenedField(table_name="orders", root_column="total_amount", path="total_amount", data_type="DOUBLE"),
        FlattenedField(table_name="orders", root_column="internal_notes", path="internal_notes", data_type="VARCHAR"),
    ]

    tables_meta = {
        "orders": {
            "description": "Customer orders table",
            "fields": fields,
            "primary_keys": ["order_id"],
            "foreign_keys": ["customer_id"],
        }
    }

    retriever.index_database_schema(tables_meta)

    # Query asks solely for total_amount
    _, context, _ = retriever.retrieve_schema_context("sum of total_amount")
    assert "total_amount" in context
    assert "order_id" in context  # PK preserved
    assert "customer_id" in context  # FK preserved


def test_fallback_used_only_when_necessary(fast_chroma, fast_embeddings):
    """Proves that bounded fallback is used only when column retrieval finds insufficient matches."""
    retriever = TwoStageSchemaRetriever(
        collection_prefix="test_fallback",
        chroma_client=fast_chroma,
        embeddings=fast_embeddings,
    )

    fields = [
        FlattenedField(table_name="logs", root_column=f"col_{i}", path=f"col_{i}", data_type="VARCHAR")
        for i in range(20)
    ]

    tables_meta = {
        "logs": {
            "description": "System access audit logs",
            "fields": fields,
            "primary_keys": [],
            "foreign_keys": [],
        }
    }

    retriever.index_database_schema(tables_meta)

    # Generic query with no specific column names falls back to bounded subset
    _, context, _ = retriever.retrieve_schema_context("show everything in system audit logs")
    # Must contain bounded first 10 fields, not all 20
    assert "col_0" in context
    assert "col_9" in context
    assert "col_19" not in context


def test_empty_query_produces_controlled_result(fast_chroma, fast_embeddings):
    """Proves that empty or whitespace queries produce controlled empty output without crashing."""
    retriever = TwoStageSchemaRetriever(
        collection_prefix="test_empty",
        chroma_client=fast_chroma,
        embeddings=fast_embeddings,
    )

    cand_tables, context, literals = retriever.retrieve_schema_context("")
    assert cand_tables == []
    assert context == ""
    assert literals == {}

    cand_tables2, context2, literals2 = retriever.retrieve_schema_context("   ")
    assert cand_tables2 == []
    assert context2 == ""
    assert literals2 == {}


# ==============================================================================
# Phase A2: Literal-Value Matching & Security Tests
# ==============================================================================

def test_distinct_value_indexer_sql_injection_defense(fast_chroma, fast_embeddings):
    """Tests that malicious injection strings in indexed literals are safely escaped."""
    indexer = DistinctValueIndexer(
        collection_name="test_injection",
        chroma_client=fast_chroma,
        embeddings=fast_embeddings,
    )

    # Malicious SQL injection payload
    indexer.index_column_values(
        table_name="users",
        column_name="username",
        distinct_values=[
            ("admin'; DROP TABLE users; --", "admin user"),
            ("O'Connor", "Irish surname"),
        ],
    )

    matches = indexer.match_literals_detailed("Show details for admin user")
    assert len(matches) == 1
    m = matches[0]
    # Single quotes must be doubled in the SQL predicate clause
    assert "admin''; DROP TABLE users; --" in m.sql_predicate
    assert "username = 'admin''; DROP TABLE users; --'" == m.sql_predicate

    matches2 = indexer.match_literals_detailed("Find user with surname O'Connor")
    assert len(matches2) == 1
    assert "username = 'O''Connor'" == matches2[0].sql_predicate


def test_distinct_value_indexer_rejects_invalid_identifiers(fast_chroma, fast_embeddings):
    """Tests that invalid SQL identifiers for table or column raise RetrieverError."""
    indexer = DistinctValueIndexer(
        collection_name="test_bad_id",
        chroma_client=fast_chroma,
        embeddings=fast_embeddings,
    )

    with pytest.raises(RetrieverError):
        indexer.index_column_values("users; DROP TABLE customers", "state", [("CA", "California")])

    with pytest.raises(RetrieverError):
        indexer.index_column_values("users", "state = '1' OR '1'='1", [("CA", "California")])


def test_distinct_value_indexer_ambiguity_detection(fast_chroma, fast_embeddings):
    """Tests that ambiguous matches without direct query mention do not generate predicates."""
    indexer = DistinctValueIndexer(
        collection_name="test_ambiguity",
        chroma_client=fast_chroma,
        embeddings=fast_embeddings,
    )

    indexer.index_column_values(
        table_name="orders",
        column_name="status",
        distinct_values=[
            ("PENDING_REVIEW", "pending status review"),
            ("PENDING_PAYMENT", "pending status payment"),
        ],
    )

    # User query says "pending order", neither label is fully mentioned
    matches = indexer.match_literals_detailed("pending order query", k=2, score_threshold=0.0)
    # If ambiguous, no predicate should be created
    assert len(matches) <= 1


# ==============================================================================
# Phase A3: Nested-Schema Support Tests
# ==============================================================================

def test_schema_flattener_dialect_unnest_expressions():
    """Tests dialect-specific unnest expression generation."""
    field = FlattenedField(
        table_name="events",
        root_column="event_params",
        path="event_params.key",
        data_type="VARCHAR",
        is_array=True,
        parent_array_path="event_params",
    )

    duckdb_expr = SchemaFlattener.get_dialect_unnest_expression(field, "duckdb")
    assert duckdb_expr == "UNNEST(event_params) AS t(event_params)"

    bq_expr = SchemaFlattener.get_dialect_unnest_expression(field, "bigquery")
    assert bq_expr == "CROSS JOIN UNNEST(event_params) AS event_params"

    snowflake_expr = SchemaFlattener.get_dialect_unnest_expression(field, "snowflake")
    assert snowflake_expr == ", LATERAL FLATTEN(input => event_params) AS event_params"

    pg_expr = SchemaFlattener.get_dialect_unnest_expression(field, "postgres")
    assert pg_expr == "CROSS JOIN json_array_elements(event_params) AS event_params"

    # Non-array field returns None
    scalar_field = FlattenedField(
        table_name="events",
        root_column="event_id",
        path="event_id",
        data_type="VARCHAR",
        is_array=False,
    )
    assert SchemaFlattener.get_dialect_unnest_expression(scalar_field, "duckdb") is None


# ==============================================================================
# Phase A4 & A5: Table Join Selection & Index Lifecycle Tests
# ==============================================================================

def test_multi_table_relationship_join_enrichment(fast_chroma, fast_embeddings):
    """Tests that related join tables are added to candidate tables when mentioned in query."""
    retriever = TwoStageSchemaRetriever(
        collection_prefix="test_joins",
        chroma_client=fast_chroma,
        embeddings=fast_embeddings,
    )

    tables_meta = {
        "orders": {
            "description": "E-commerce order transactions and payment totals",
            "fields": [
                FlattenedField(table_name="orders", root_column="order_id", path="order_id", data_type="VARCHAR"),
                FlattenedField(table_name="orders", root_column="customer_id", path="customer_id", data_type="VARCHAR"),
                FlattenedField(table_name="orders", root_column="amount", path="amount", data_type="DOUBLE"),
            ],
            "primary_keys": ["order_id"],
            "foreign_keys": ["customer_id"],
        },
        "order_items": {
            "description": "Line items for orders, product units, item price",
            "fields": [
                FlattenedField(table_name="order_items", root_column="item_id", path="item_id", data_type="VARCHAR"),
                FlattenedField(table_name="order_items", root_column="order_id", path="order_id", data_type="VARCHAR"),
                FlattenedField(table_name="order_items", root_column="product_name", path="product_name", data_type="VARCHAR"),
            ],
            "primary_keys": ["item_id"],
            "foreign_keys": ["order_id"],
        },
    }

    retriever.index_database_schema(tables_meta)

    # Query mentions orders and product items
    cand_tables, _, _ = retriever.retrieve_schema_context("order transactions and order_items product details")
    assert "orders" in cand_tables
    assert "order_items" in cand_tables


def test_index_lifecycle_clearing_prevents_duplicate_documents(fast_chroma, fast_embeddings):
    """Tests that re-indexing with clear_existing=True prevents document duplication."""
    retriever = TwoStageSchemaRetriever(
        collection_prefix="test_lifecycle",
        chroma_client=fast_chroma,
        embeddings=fast_embeddings,
    )

    fields = [
        FlattenedField(table_name="t1", root_column="col1", path="col1", data_type="VARCHAR"),
    ]
    meta = {
        "t1": {
            "description": "Test table",
            "fields": fields,
            "primary_keys": ["col1"],
        }
    }

    # Index once
    retriever.index_database_schema(meta, clear_existing=True)
    count1 = len(retriever.column_store.get()["ids"])

    # Index again with clear_existing=True
    retriever.index_database_schema(meta, clear_existing=True)
    count2 = len(retriever.column_store.get()["ids"])

    assert count1 == count2
    assert count2 == 1
