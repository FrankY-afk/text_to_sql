"""Unit tests for Problem 2: Two-Stage Schema Retriever, Nested Struct/Array Flattener, and Value Indexer."""

from __future__ import annotations

import pytest
import duckdb
import chromadb

from text2sql_engine.modules.problem2_retriever.schema_flattener import (
    FlattenedField,
    SchemaFlattener,
)
from text2sql_engine.modules.problem2_retriever.vector_retriever import (
    DistinctValueIndexer,
    TwoStageSchemaRetriever,
)
from text2sql_engine.data.setup_sample_db import init_sample_database


@pytest.fixture(scope="module")
def duckdb_conn():
    """Provides an in-memory DuckDB database fixture populated with relational and nested tables."""
    con = init_sample_database()
    yield con
    con.close()


@pytest.fixture(scope="module")
def ephemeral_chroma():
    """Provides an ephemeral in-memory Chroma client for fast isolated test execution."""
    return chromadb.EphemeralClient()


def test_schema_flattener_nested_struct_and_array():
    """Tests recursive extraction of nested STRUCT and ARRAY definitions into dot paths."""
    type_str = "STRUCT(key VARCHAR, value STRUCT(string_value VARCHAR, int_value BIGINT))[]"
    fields = SchemaFlattener.parse_type_string(
        table_name="web_events",
        root_column="event_params",
        current_path="event_params",
        type_str=type_str,
    )

    path_map = {f.path: f for f in fields}

    assert "event_params.key" in path_map
    assert "event_params.value.string_value" in path_map
    assert "event_params.value.int_value" in path_map

    # Check properties
    key_field = path_map["event_params.key"]
    assert key_field.is_array is True
    assert key_field.is_nested is True
    assert key_field.data_type.upper() == "VARCHAR"
    assert "UNNEST" in (key_field.unnest_syntax or "")

    str_val_field = path_map["event_params.value.string_value"]
    assert str_val_field.is_array is True
    assert str_val_field.is_nested is True
    assert str_val_field.data_type.upper() == "VARCHAR"

    int_val_field = path_map["event_params.value.int_value"]
    assert int_val_field.data_type.upper() == "BIGINT"


def test_schema_flattener_duckdb_live_table(duckdb_conn):
    """Tests flattening directly from live DuckDB table reflection."""
    fields = SchemaFlattener.flatten_duckdb_table(duckdb_conn, "web_events")

    paths = [f.path for f in fields]
    assert "event_id" in paths
    assert "user_id" in paths
    assert "geo_info.country" in paths
    assert "geo_info.city" in paths
    assert "event_params.key" in paths
    assert "event_params.value.string_value" in paths

    card = SchemaFlattener.generate_schema_card("web_events", fields, "GA4 raw analytics stream")
    assert "Table: web_events" in card
    assert "event_params.value.string_value (VARCHAR) [ARRAY (requires UNNEST), NESTED]" in card


def test_distinct_value_indexer_literal_mapping(ephemeral_chroma):
    """Tests mapping natural language filter terms (e.g. 'California') to database literals ('state_code = 'CA'')."""
    indexer = DistinctValueIndexer(
        collection_name="test_distinct_values",
        chroma_client=ephemeral_chroma,
    )

    # Index sample categorical values
    indexer.index_column_values(
        table_name="customers",
        column_name="state_code",
        distinct_values=[
            ("CA", "California"),
            ("NY", "New York"),
            ("TX", "Texas"),
        ],
    )
    indexer.index_column_values(
        table_name="orders",
        column_name="order_status",
        distinct_values=[
            ("COMPLETED", "completed"),
            ("CANCELLED", "cancelled"),
        ],
    )

    # Test query entity disambiguation
    query1 = "Find all orders completed by users located in California"
    matched1 = indexer.match_literals(query1)

    assert "California" in matched1 or "CA" in matched1
    matched_predicates = list(matched1.values())
    assert any("state_code = 'CA'" in p for p in matched_predicates)
    assert any("order_status = 'COMPLETED'" in p for p in matched_predicates)


def test_two_stage_schema_retriever(duckdb_conn, ephemeral_chroma):
    """Tests coarse table pruning and fine-grained column extraction over multi-table schema."""
    retriever = TwoStageSchemaRetriever(
        collection_prefix="test_schema_retriever",
        chroma_client=ephemeral_chroma,
    )

    # Index all 4 tables from DuckDB
    tables_meta = {
        "customers": {
            "description": "Customer profiles, demographic location, active status",
            "fields": SchemaFlattener.flatten_duckdb_table(duckdb_conn, "customers"),
            "primary_keys": ["customer_id"],
        },
        "orders": {
            "description": "E-commerce orders, timestamps, order status, total monetary amounts",
            "fields": SchemaFlattener.flatten_duckdb_table(duckdb_conn, "orders"),
            "primary_keys": ["order_id"],
        },
        "order_items": {
            "description": "Line items per order, product categories, unit prices and quantities",
            "fields": SchemaFlattener.flatten_duckdb_table(duckdb_conn, "order_items"),
            "primary_keys": ["item_id"],
        },
        "web_events": {
            "description": "GA4 web traffic events, page views, campaign attribution and nested parameters",
            "fields": SchemaFlattener.flatten_duckdb_table(duckdb_conn, "web_events"),
            "primary_keys": ["event_id"],
        },
    }

    sample_values = {
        "customers": {
            "state_code": [("CA", "California"), ("NY", "New York"), ("TX", "Texas")]
        },
        "orders": {
            "order_status": [("COMPLETED", "Completed"), ("CANCELLED", "Cancelled")]
        },
    }

    retriever.index_database_schema(tables_meta, sample_values)

    # Test retrieval for web events question
    web_query = "What campaigns were associated with purchase events in California?"
    cand_tables, schema_context, literals = retriever.retrieve_schema_context(web_query, top_k_tables=2)

    assert "web_events" in cand_tables
    assert "event_params.value.string_value" in schema_context
    assert "geo_info.region" in schema_context
    assert "California" in literals or "CA" in literals

    # Test retrieval for e-commerce orders question
    orders_query = "Total revenue and shipping cost for completed customer orders"
    cand_tables2, schema_context2, literals2 = retriever.retrieve_schema_context(orders_query, top_k_tables=2)

    assert "orders" in cand_tables2
    assert "total_amount" in schema_context2


def test_nested_data_duckdb_execution(duckdb_conn):
    """Executes SQL against nested STRUCT and ARRAY columns on DuckDB validating real output."""
    sql = """
    SELECT 
        event_id,
        geo_info.city AS city,
        geo_info.region AS region,
        ep.key AS param_key,
        ep.value.string_value AS param_val
    FROM web_events, UNNEST(event_params) AS t(ep)
    WHERE geo_info.region = 'California'
      AND ep.key = 'campaign'
    ORDER BY event_id ASC;
    """

    results = duckdb_conn.execute(sql).fetchall()
    assert len(results) == 2  # EV01 (San Francisco) and EV03 (Los Angeles)

    # Check EV01
    ev01 = results[0]
    assert ev01[0] == "EV01"
    assert ev01[1] == "San Francisco"
    assert ev01[2] == "California"
    assert ev01[3] == "campaign"
    assert ev01[4] == "winter_sale"

    # Check EV03
    ev03 = results[1]
    assert ev03[0] == "EV03"
    assert ev03[1] == "Los Angeles"
    assert ev03[2] == "California"
    assert ev03[3] == "campaign"
    assert ev03[4] == "spring_kickoff"
