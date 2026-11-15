"""Unit tests for AST-based SQL syntax, read-only safety, and policy validation."""

from __future__ import annotations

import pytest
from src.sql_validator import validate_readonly_sql


def test_valid_select_queries_allowed():
    """Verifies that legitimate SELECT queries pass validation."""
    valid_sql = """
    WITH cohort AS (
        SELECT user_id, signup_date FROM users WHERE signup_date >= '2024-01-01'
    )
    SELECT c.user_id, COUNT(DISTINCT e.event_id) AS events_cnt
    FROM cohort c
    LEFT JOIN activity_events e ON c.user_id = e.user_id
    GROUP BY c.user_id;
    """
    report = validate_readonly_sql(valid_sql)
    assert report.is_valid is True
    assert report.is_readonly is True
    assert "users" in report.tables_referenced
    assert "activity_events" in report.tables_referenced
    assert report.has_distinct is True
    assert report.has_ctes is True
    assert "cohort" in report.cte_names


@pytest.mark.parametrize(
    "malicious_sql, expected_error_fragment",
    [
        ("DROP TABLE users;", "Statement of type 'Drop' is not permitted"),
        ("CREATE TABLE evil (id int);", "Statement of type 'Create' is not permitted"),
        ("ALTER TABLE users ADD COLUMN hacked text;", "Statement of type 'Alter' is not permitted"),
        ("DELETE FROM users WHERE user_id = 1;", "Statement of type 'Delete' is not permitted"),
        ("UPDATE users SET country = 'US';", "Statement of type 'Update' is not permitted"),
        ("INSERT INTO users VALUES (999, 'bad', '2024-01-01', 'US');", "Statement of type 'Insert' is not permitted"),
        ("TRUNCATE TABLE users;", "Statement of type 'TruncateTable' is not permitted"),
    ],
)
def test_ddl_and_dml_rejected(malicious_sql: str, expected_error_fragment: str):
    """Verifies that all destructive operations are blocked at the AST level."""
    report = validate_readonly_sql(malicious_sql)
    assert report.is_valid is False
    assert report.is_readonly is False
    assert expected_error_fragment in (report.error or "")


def test_multiple_statements_rejected():
    """Verifies that batch query injection attempts are rejected."""
    batch_sql = "SELECT * FROM users; DROP TABLE activity_events;"
    report = validate_readonly_sql(batch_sql)
    assert report.is_valid is False
    assert "Multiple SQL statements detected" in (report.error or "")


def test_dangerous_system_functions_blocked():
    """Verifies that administrative system functions are blocked."""
    func_sql = "SELECT pg_sleep(10);"
    report = validate_readonly_sql(func_sql)
    assert report.is_valid is False
    assert "forbidden system function" in (report.error or "")


def test_dialect_violations_blocked():
    """Verifies that dialect-forbidden syntax (QUALIFY on Postgres) is blocked."""
    bad_qualify = "SELECT user_id FROM users QUALIFY ROW_NUMBER() OVER (ORDER BY signup_date) = 1;"
    report = validate_readonly_sql(bad_qualify, dialect="postgres")
    assert report.is_valid is False
    assert any("QUALIFY" in v for v in report.dialect_violations)


def test_cte_forward_reference_in_sql_detected():
    """Verifies that CTE forward references inside raw SQL are detected."""
    bad_cte_sql = """
    WITH step1 AS (
        SELECT * FROM step2
    ),
    step2 AS (
        SELECT 1 AS val
    )
    SELECT * FROM step1;
    """
    report = validate_readonly_sql(bad_cte_sql, dialect="postgres")
    assert report.is_valid is False
    assert "forward-reference" in (report.error or "")
