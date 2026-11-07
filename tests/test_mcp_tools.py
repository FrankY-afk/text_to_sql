"""Unit and integration tests for MCP server tools, discovery, and execution."""

from __future__ import annotations

import asyncio
import pytest
from src.query_executor import MCPQueryExecutor
from tests.conftest import postgres_required


@pytest.fixture
def executor():
    return MCPQueryExecutor()


def test_mcp_tool_discovery(executor):
    """Verifies that the MCP server advertises all required tools."""
    tools = asyncio.run(executor.list_available_mcp_tools())
    tool_names = [t["name"] for t in tools]

    expected_tools = [
        "list_tables",
        "get_table_schema",
        "get_sample_rows",
        "validate_sql",
        "execute_readonly_sql",
        "resolve_categorical_literals",
    ]
    for expected in expected_tools:
        assert expected in tool_names


def test_mcp_resolve_categorical_literals_offline(executor):
    """Verifies resolve_categorical_literals MCP tool grounds abbreviations offline."""
    matches = executor.resolve_categorical_literals_sync(
        "Show outlets in Tier 1", target="local"
    )
    terms = {m["term"]: m["database_literal"] for m in matches}
    assert terms.get("tier 1") == "T1"


@postgres_required
def test_mcp_resolve_categorical_literals(executor):
    """Verifies resolve_categorical_literals MCP tool grounds natural language to database cell values."""
    matches = executor.resolve_categorical_literals_sync(
        "Show outlets in Telangana in Tier 1", target="postgres"
    )
    terms = {m["term"]: m["database_literal"] for m in matches}
    assert terms.get("tier 1") == "T1"
    assert terms.get("Telangana") == "Telangana"


def test_mcp_list_tables_local(executor):
    """Verifies list_tables returns tables for local target."""
    local_tables = executor.list_tables_sync(target="local")
    assert "users" in local_tables
    assert "activity_events" in local_tables


@postgres_required
def test_mcp_list_tables_postgres(executor):
    """Verifies list_tables returns tables for postgres target."""
    pg_tables = executor.list_tables_sync(target="postgres")
    assert "outlets" in pg_tables
    assert "group_loans" in pg_tables


def test_mcp_get_table_schema_local(executor):
    """Verifies get_table_schema returns structured schema markdown for local database."""
    schema_card = executor.get_table_schema_sync("users", target="local")
    assert "### Table: `users`" in schema_card
    assert "`user_id`" in schema_card
    assert "`signup_date`" in schema_card


@postgres_required
def test_mcp_get_table_schema_postgres(executor):
    """Verifies get_table_schema returns structured schema markdown for PostgreSQL."""
    schema_card = executor.get_table_schema_sync("outlets", target="postgres")
    assert "### Table: `outlets`" in schema_card
    assert "`outlet_code`" in schema_card
    assert "`state_name`" in schema_card


def test_mcp_get_sample_rows_clamped_local(executor):
    """Verifies get_sample_rows clamps limits to safe boundaries on local database."""
    sample = executor.get_sample_rows_sync("users", limit=2, target="local")
    assert sample["count"] == 2
    assert len(sample["rows"]) == 2


@postgres_required
def test_mcp_get_sample_rows_clamped_postgres(executor):
    """Verifies get_sample_rows clamps limits to safe boundaries on PostgreSQL."""
    sample = executor.get_sample_rows_sync("outlets", limit=2, target="postgres")
    assert sample["count"] == 2
    assert len(sample["rows"]) == 2


def test_mcp_validate_sql_tool(executor):
    """Verifies validate_sql tool detects read-only status and tables."""
    res = executor.validate_sql_sync("SELECT user_id FROM users LIMIT 1", dialect="sqlite")
    assert res["is_valid"] is True
    assert res["is_readonly"] is True
    assert "users" in res["tables_referenced"]


def test_mcp_execute_readonly_sql_blocks_write(executor):
    """Verifies execute_readonly_sql blocks destructive writes before database execution."""
    res = executor.execute_readonly_sql_sync("DROP TABLE users", target="local")
    assert res["success"] is False
    assert "Validation Rejected" in res["error"]

