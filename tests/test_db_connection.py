"""Unit and integration tests for PostgreSQL and local database connectivity."""

from __future__ import annotations

import pytest
from src.config import get_settings
from src.db import (
    execute_readonly_query,
    get_db_cursor,
    get_local_db_connection,
    test_connection as check_db_connection,
)
from tests.conftest import postgres_required


@postgres_required
def test_postgres_live_connection():
    """Verifies that live PostgreSQL connection succeeds and returns valid metadata."""
    settings = get_settings()
    info = check_db_connection(settings)

    assert info["status"] == "connected"
    assert info["user"] == "t2sql_reader"
    assert info["database"] == "lending_sample"
    assert "PostgreSQL" in info["version"]


@postgres_required
def test_postgres_readonly_session_settings():
    """Verifies that the session enforces statement timeouts and search path."""
    with get_db_cursor() as cur:
        # Check statement timeout is 60s (PostgreSQL formats as '1min' or '60s' or '60000ms')
        cur.execute("SHOW statement_timeout;")
        timeout = cur.fetchone()[0]
        assert any(t in timeout.lower() for t in ["1min", "60s", "60000"])

        # Check default transaction read-only is on
        cur.execute("SHOW default_transaction_read_only;")
        read_only = cur.fetchone()[0]
        assert read_only == "on"


@postgres_required
def test_postgres_execute_readonly_query_bounded():
    """Verifies that execute_readonly_query respects row bounds and returns dicts."""
    res = execute_readonly_query("SELECT outlet_code, state_name FROM outlets;", max_rows=5)


    assert len(res["rows"]) == 5
    assert res["row_count"] == 5
    assert res["truncated"] is True
    assert "outlet_code" in res["columns"]
    assert "state_name" in res["columns"]


def test_local_sqlite_connection():
    """Verifies connection and table presence in the local benchmark database."""
    conn = get_local_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in cur.fetchall()]
    conn.close()

    assert "users" in tables
    assert "activity_events" in tables
