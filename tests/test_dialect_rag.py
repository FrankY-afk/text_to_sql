"""Unit tests for Dialect Function RAG (Problem 1, Mechanism B)."""

from __future__ import annotations

import pytest
from src.dialect_rag import (
    DIALECT_CATALOG,
    DialectFunctionRAG,
    DialectFunctionSpec,
)


@pytest.fixture
def rag():
    return DialectFunctionRAG()


def test_catalog_has_all_required_dialects(rag):
    """Verifies that the catalog indexes functions across postgres, duckdb, bigquery, and snowflake."""
    dialects_present = {f.dialect for f in rag.catalog}
    assert "postgres" in dialects_present
    assert "duckdb" in dialects_present
    assert "bigquery" in dialects_present
    assert "snowflake" in dialects_present


def test_retrieve_date_diff_postgres(rag):
    """Verifies that PostgreSQL retrieves date subtraction and explains lack of DATEDIFF."""
    funcs = rag.retrieve_functions("Calculate days between signup and purchase", target_dialect="postgres")
    assert len(funcs) > 0
    names = [f.name for f in funcs]
    assert "date_subtraction_postgres" in names
    
    spec = next(f for f in funcs if f.name == "date_subtraction_postgres")
    assert "::date - " in spec.syntax_pattern
    assert "does NOT have a built-in DATEDIFF" in spec.caveats_and_pitfalls


def test_retrieve_date_diff_bigquery_vs_duckdb(rag):
    """Verifies BigQuery and DuckDB retrieve different argument ordering for date difference."""
    bq_funcs = rag.retrieve_functions("Find difference in days between dates", target_dialect="bigquery")
    duck_funcs = rag.retrieve_functions("Find difference in days between dates", target_dialect="duckdb")

    bq_spec = next((f for f in bq_funcs if "DATE_DIFF" in f.name), None)
    duck_spec = next((f for f in duck_funcs if "date_diff" in f.name), None)

    assert bq_spec is not None
    assert duck_spec is not None

    # BigQuery places end_date first
    assert "DATE_DIFF({end_date}, {start_date}, {unit})" in bq_spec.syntax_pattern
    # DuckDB places unit first
    assert "date_diff('{unit}', {start_date}, {end_date})" in duck_spec.syntax_pattern


def test_retrieve_safe_division_for_percentage(rag):
    """Verifies that ratio/percentage questions retrieve safe division guards."""
    funcs = rag.retrieve_functions("What percentage of users returned?", target_dialect="postgres")
    assert any("nullif" in f.name.lower() for f in funcs)


def test_unsupported_functions_in_postgres(rag):
    """Verifies that QUALIFY and DATEDIFF are recognized as unsupported in PostgreSQL."""
    unsupported = rag.get_unsupported_functions(target_dialect="postgres")
    unsupported_names = [u.name for u in unsupported]
    assert any("qualify" in name.lower() for name in unsupported_names)
    assert any("datediff" in name.lower() for name in unsupported_names)


def test_format_prompt_context(rag):
    """Verifies that prompt context renders cleanly with rules and forbidden warnings."""
    context = rag.format_prompt_context(
        "What is the user retention rate?",
        target_dialect="postgres",
    )
    assert "### Target SQL Dialect Knowledge & Rules: [POSTGRES]" in context
    assert "Applicable Functions & Syntaxes" in context
    assert "FORBIDDEN In POSTGRES" in context
    assert "QUALIFY" in context


def test_validate_sql_dialect_compatibility_detects_violations(rag):
    """Verifies that invalid dialect functions are caught by validation."""
    # 1. QUALIFY in Postgres should fail
    bad_pg_qualify = "SELECT user_id FROM users QUALIFY ROW_NUMBER() OVER (ORDER BY signup_date) = 1;"
    violations = rag.validate_sql_dialect_compatibility(bad_pg_qualify, target_dialect="postgres")
    assert len(violations) >= 1
    assert any("QUALIFY" in v for v in violations)

    # 2. QUALIFY in Snowflake should pass
    violations_sf = rag.validate_sql_dialect_compatibility(bad_pg_qualify, target_dialect="snowflake")
    assert len(violations_sf) == 0

    # 3. DATEDIFF in Postgres should fail
    bad_pg_datediff = "SELECT DATEDIFF(day, signup_date, event_date) FROM users;"
    violations_datediff = rag.validate_sql_dialect_compatibility(bad_pg_datediff, target_dialect="postgres")
    assert len(violations_datediff) >= 1
    assert any("DATEDIFF" in v for v in violations_datediff)

    # 4. Valid Postgres date subtraction should pass
    valid_pg = "SELECT (e.event_date::date - u.signup_date::date) AS days_diff FROM users u;"
    assert len(rag.validate_sql_dialect_compatibility(valid_pg, target_dialect="postgres")) == 0

