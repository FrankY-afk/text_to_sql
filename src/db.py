from __future__ import annotations
import logging
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple
import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extensions import cursor as PgCursor

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when connecting to PostgreSQL fails."""


class QueryExecutionError(Exception):
    """Raised when executing a database query fails."""


def create_postgres_connection(settings: Optional[Settings] = None) -> PgConnection:
    """Creates a new PostgreSQL connection with statement timeout and read-only defaults."""
    cfg = settings or get_settings()
    conn_dict = cfg.get_postgres_connection_dict()

    try:
        conn = psycopg2.connect(**conn_dict)
        conn.autocommit = True  

        # Session safety initialization
        with conn.cursor() as cur:
            # Set search path to configured schema
            cur.execute(f"SET search_path TO {cfg.postgres_schema}, public;")
            # Apply execution timeout (e.g. 60000ms = 60s)
            timeout_ms = cfg.query_timeout_seconds * 1000
            cur.execute(f"SET statement_timeout = {timeout_ms};")
            # Enforce read-only transaction mode as an additional layer of defense
            cur.execute("SET default_transaction_read_only = 'on';")

        return conn
    except Exception as exc:
        logger.error("Failed to connect to PostgreSQL at %s:%s: %s", cfg.postgres_host, cfg.postgres_port, exc)
        raise DatabaseConnectionError(f"PostgreSQL connection failed: {exc}") from exc


@contextmanager
def get_db_cursor(settings: Optional[Settings] = None) -> Generator[PgCursor, None, None]:
    """Context manager yielding an active PostgreSQL cursor and cleanly closing the connection."""
    conn = create_postgres_connection(settings)
    try:
        with conn.cursor() as cur:
            yield cur
    finally:
        try:
            conn.close()
        except Exception:
            pass


def test_connection(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Tests the PostgreSQL connection and returns server metadata."""
    cfg = settings or get_settings()
    with get_db_cursor(cfg) as cur:
        cur.execute("SELECT current_user, current_database(), version();")
        row = cur.fetchone()
        return {
            "status": "connected",
            "user": row[0] if row else None,
            "database": row[1] if row else None,
            "version": row[2] if row else None,
            "host": cfg.postgres_host,
            "port": cfg.postgres_port,
        }


def execute_readonly_query(
    sql: str,
    params: Optional[Tuple[Any, ...]] = None,
    max_rows: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
   
    cfg = settings or get_settings()
    limit = max_rows or cfg.max_row_limit

    with get_db_cursor(cfg) as cur:
        try:
            cur.execute(sql, params)
        except Exception as exc:
            raise QueryExecutionError(f"Database error executing SQL: {exc}") from exc

        if cur.description is None:
            # Query did not produce rows (e.g., SET or empty command)
            return {"columns": [], "rows": [], "row_count": 0, "truncated": False}

        columns = [col[0] for col in cur.description]
        fetched_rows = cur.fetchmany(limit + 1)
        truncated = len(fetched_rows) > limit
        final_rows = fetched_rows[:limit]

        # Convert rows into list of dicts for clean JSON serialization
        dict_rows = [dict(zip(columns, row)) for row in final_rows]

        return {
            "columns": columns,
            "rows": dict_rows,
            "row_count": len(dict_rows),
            "truncated": truncated,
        }


def get_local_db_connection(settings: Optional[Settings] = None) -> sqlite3.Connection:
    """Connects to the local SQLite database used for local test benchmarks and scratch tables."""
    cfg = settings or get_settings()
    conn = sqlite3.connect(cfg.local_db_path)
    conn.row_factory = sqlite3.Row
    return conn
