"""Pytest configuration, shared fixtures, and database availability detectors."""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from src.config import get_settings
from scripts.seed_test_data import seed_local_benchmark_db

# Ensure local benchmark database exists before tests run
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DB = PROJECT_ROOT / "local_problem1.db"
if not LOCAL_DB.exists():
    seed_local_benchmark_db(str(LOCAL_DB))


def _check_postgres_live() -> bool:
    """Checks whether the configured PostgreSQL server is reachable with valid credentials."""
    try:
        from src.db import create_postgres_connection
        # Fast connection attempt
        conn = create_postgres_connection()
        conn.close()
        return True
    except Exception:
        return False


POSTGRES_LIVE = _check_postgres_live()

postgres_required = pytest.mark.skipif(
    not POSTGRES_LIVE,
    reason="Live PostgreSQL is not reachable or credentials are not configured in .env",
)

