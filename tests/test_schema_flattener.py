"""Unit and DuckDB execution tests for Stage 3: Nested Schema Tree Flattener."""

from __future__ import annotations

import duckdb
import pytest

from src.schema_flattener import FlattenedField, SchemaFlattener
from src.schema_inspector import ColumnMetadata, SchemaInspector, TableMetadata


def test_flatten_bigquery_ga4_struct_array():
    """Flattens BigQuery ARRAY<STRUCT<key, value STRUCT<...>>> and preserves types/lineage."""
    table = TableMetadata(
        schema_name="analytics",
        table_name="events",
        full_name="analytics.events",
        columns=[
            ColumnMetadata(name="event_id", data_type="STRING", is_primary_key=True),
            ColumnMetadata(
                name="event_params",
                data_type="ARRAY<STRUCT<key STRING, value STRUCT<string_value STRING, int_value INT64>>>",
                comment="GA4 event parameters",
            ),
        ],
    )

    fields = SchemaFlattener.flatten_table_columns(
        table, dialect="bigquery", include_intermediate_nodes=True
    )
    paths = {f.path: f for f in fields}

    assert "event_params.key" in paths
    assert "event_params.value.string_value" in paths
    assert "event_params.value.int_value" in paths

    # Original types and lineage preserved
    assert paths["event_params.key"].data_type == "STRING"
    assert paths["event_params.value.int_value"].data_type == "INT64"
    assert paths["event_params.value.int_value"].parent_path == "event_params.value"
    assert paths["event_params.value"].parent_path == "event_params"
    assert "event_params.value" in paths["event_params"].children_paths

    # Must NOT assume flattened path inside ARRAY is directly queryable without UNNEST
    assert paths["event_params.key"].is_directly_queryable is False
    assert paths["event_params.value.int_value"].is_directly_queryable is False
    assert "UNNEST" in paths["event_params.value.int_value"].dialect_access_hint


def test_flatten_duckdb_struct_and_array_live_execution():
    """Verifies DuckDB nested STRUCT/ARRAY schema flattening and actual UNNEST SQL semantics."""
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE web_events (
            event_id VARCHAR PRIMARY KEY,
            event_params STRUCT(key VARCHAR, value STRUCT(string_value VARCHAR, int_value BIGINT))[]
        );
        INSERT INTO web_events VALUES
            ('e1', [{'key': 'engagement_time_msec', 'value': {'string_value': NULL, 'int_value': 4200}}]),
            ('e2', [{'key': 'page_title', 'value': {'string_value': 'Pricing', 'int_value': NULL}}]);
        """
    )

    inspector = SchemaInspector(target="duckdb", duckdb_conn=conn)
    meta = inspector.get_table_metadata("web_events", include_samples=False)
    flattened = SchemaFlattener.flatten_table_columns(meta, dialect="duckdb")
    by_path = {f.path: f for f in flattened}

    assert "event_params.key" in by_path
    assert "event_params.value.string_value" in by_path
    assert "event_params.value.int_value" in by_path
    assert by_path["event_params.value.int_value"].is_directly_queryable is False
    assert "UNNEST" in by_path["event_params.value.int_value"].dialect_access_hint

    # Execute actual DuckDB UNNEST query to confirm access semantics work on real DB
    rows = conn.execute(
        """
        SELECT ep.unnest.key AS param_key, ep.unnest.value.int_value AS engagement_ms
        FROM web_events, UNNEST(event_params) AS ep
        WHERE ep.unnest.key = 'engagement_time_msec'
        """
    ).fetchall()
    assert rows == [("engagement_time_msec", 4200)]


def test_flatten_postgres_jsonb_nested_objects_and_arrays():
    """Flattens JSON/JSONB objects & arrays into dot paths with PostgreSQL ->> access metadata."""
    sample_payload = {
        "customer": {"id": 101, "tier": "enterprise"},
        "items": [{"sku": "SKU-9", "qty": 2}],
    }
    fields = SchemaFlattener.flatten_json_payload(
        table_name="billing.webhooks",
        root_column="payload",
        payload=sample_payload,
        dialect="postgres",
    )
    by_path = {f.path: f for f in fields}

    assert "payload.customer.tier" in by_path
    assert "payload.items.sku" in by_path
    assert "payload.items.qty" in by_path

    tier_field = by_path["payload.customer.tier"]
    assert tier_field.parent_path == "payload.customer"
    assert tier_field.is_json is True
    assert tier_field.is_directly_queryable is False
    assert "payload->'customer'->>'tier'" in tier_field.dialect_access_hint

    sku_field = by_path["payload.items.sku"]
    assert sku_field.is_array is True
    assert "jsonb_array_elements" in sku_field.dialect_access_hint


def test_flatten_sqlite_and_snowflake_json_access_hints():
    """Ensures dialect-aware JSON extraction hints differ cleanly for SQLite and Snowflake."""
    sample = {"payment": {"currency": "USD"}}
    sq_fields = {f.path: f for f in SchemaFlattener.flatten_json_payload("orders", "meta", sample, dialect="sqlite")}
    sf_fields = {f.path: f for f in SchemaFlattener.flatten_json_payload("orders", "meta", sample, dialect="snowflake")}

    assert "json_extract(meta, '$.payment.currency')" in sq_fields["meta.payment.currency"].dialect_access_hint
    assert "meta:payment:currency::STRING" in sf_fields["meta.payment.currency"].dialect_access_hint

