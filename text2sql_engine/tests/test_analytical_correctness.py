"""Analytical correctness tests with deterministic DuckDB fixtures (Phase B4).

Verifies that SQL queries are not merely syntactically valid in SQLGlot,
but analytically correct regarding grain, joins, counts, ratios, cohorts,
nulls, ties, and time boundaries.
"""

from __future__ import annotations

import duckdb
import pytest

from text2sql_engine.modules.problem1_planner.sql_compiler import SQLCompiler
from text2sql_engine.modules.problem1_planner.schemas import CTEPlanStep, LogicPlan


@pytest.fixture
def analytics_db():
    """Sets up a small, deterministic in-memory DuckDB database with known expected values."""
    con = duckdb.connect(":memory:")

    # 1. Users table (Cohorts: 2 in week 2024-01-01, 1 in week 2024-01-08)
    con.execute("""
    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY,
        name VARCHAR,
        signup_date DATE
    );
    INSERT INTO users VALUES
        (1, 'Alice', DATE '2024-01-01'),
        (2, 'Bob', DATE '2024-01-01'),
        (3, 'Charlie', DATE '2024-01-08');
    """)

    # 2. Orders table (user 1 has 2 orders, user 2 has 1 order, user 3 has 1 order)
    con.execute("""
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        order_date DATE,
        total_amount DOUBLE,
        shipping_cost DOUBLE
    );
    INSERT INTO orders VALUES
        (101, 1, DATE '2024-01-02', 100.0, 10.0),
        (102, 1, DATE '2024-01-03', 200.0, 20.0),
        (103, 2, DATE '2024-01-04', 150.0, 0.0),
        (104, 3, DATE '2024-01-09', 50.0, 5.0),
        (105, 1, DATE '2024-01-10', 0.0, 0.0);   -- Zero amount order
    """)

    # 3. Order Items table (Order 101 has 2 line items - potential join explosion)
    con.execute("""
    CREATE TABLE order_items (
        item_id INTEGER PRIMARY KEY,
        order_id INTEGER,
        product_name VARCHAR,
        unit_price DOUBLE,
        quantity INTEGER
    );
    INSERT INTO order_items VALUES
        (1, 101, 'Widget A', 50.0, 1),
        (2, 101, 'Widget B', 50.0, 1),
        (3, 102, 'Gadget C', 200.0, 1),
        (4, 103, 'Gadget D', 150.0, 1),
        (5, 104, 'Widget E', 50.0, 1);
    """)

    # 4. Activity events (user 1 has multiple events on same day)
    con.execute("""
    CREATE TABLE events (
        event_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        event_type VARCHAR,
        event_time TIMESTAMP
    );
    INSERT INTO events VALUES
        (1, 1, 'view', TIMESTAMP '2024-01-02 10:00:00'),
        (2, 1, 'view', TIMESTAMP '2024-01-02 10:05:00'),
        (3, 1, 'click', TIMESTAMP '2024-01-02 10:10:00'),
        (4, 2, 'view', TIMESTAMP '2024-01-04 11:00:00');
    """)

    yield con
    con.close()


# ==============================================================================
# B4 Tests: Analytical Grain & Join Correctness
# ==============================================================================

def test_aggregation_grain_customer_vs_order(analytics_db):
    """Proves the difference between customer-level aggregation grain vs order-level grain."""
    # Incorrect query: Grouping by order_id when customer total is requested
    wrong_sql = "SELECT user_id, order_id, SUM(total_amount) AS total FROM orders GROUP BY user_id, order_id"
    wrong_rows = analytics_db.execute(wrong_sql).fetchall()
    # Returns 5 rows because grain is per-order
    assert len(wrong_rows) == 5

    # Correct query: Aggregating to customer grain
    correct_sql = "SELECT user_id, SUM(total_amount) AS total FROM orders GROUP BY user_id ORDER BY user_id"
    correct_rows = analytics_db.execute(correct_sql).fetchall()
    # Returns 3 rows: Alice (300.0), Bob (150.0), Charlie (50.0)
    assert len(correct_rows) == 3
    assert correct_rows[0] == (1, 300.0)
    assert correct_rows[1] == (2, 150.0)
    assert correct_rows[2] == (3, 50.0)


def test_join_multiplication_fanout_prevention(analytics_db):
    """Proves that joining 1-to-many without pre-aggregation inflates order amounts."""
    # Incorrect query: Directly joining orders and order_items inflates order 101
    bad_join_sql = """
    SELECT o.user_id, SUM(o.total_amount) AS inflated_total
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.user_id = 1
    GROUP BY o.user_id
    """
    bad_result = analytics_db.execute(bad_join_sql).fetchone()
    # Order 101 had 2 line items, so its 100.0 was counted twice: 100*2 + 200 = 400.0 (INCORRECT!)
    assert bad_result[1] == 400.0

    # Correct CTE query: Pre-aggregate or join safely
    compiler = SQLCompiler()
    step1 = CTEPlanStep(
        step_name="user_orders",
        purpose="Compute true customer order sums at order grain before item joins",
        source_tables=["orders"],
        math_logic="SUM(total_amount) AS total_spend",
        sql_fragment="SELECT user_id, SUM(total_amount) AS total_spend FROM orders WHERE user_id = 1 GROUP BY user_id",
    )
    plan = LogicPlan(
        query="True customer spend",
        target_dialect="duckdb",
        strategy="Aggregate orders at customer grain",
        cte_steps=[step1],
        final_select="SELECT user_id, total_spend FROM user_orders",
    )
    sql = compiler.compile(plan)
    correct_result = analytics_db.execute(sql).fetchone()
    # True total for user 1 is 300.0
    assert correct_result[1] == 300.0


def test_count_distinct_vs_count_star(analytics_db):
    """Proves that COUNT(*) counts total event occurrences while COUNT(DISTINCT user_id) measures unique users."""
    event_count_sql = "SELECT COUNT(*) FROM events WHERE event_time >= '2024-01-02' AND event_time < '2024-01-03'"
    total_events = analytics_db.execute(event_count_sql).fetchone()[0]
    # Alice has 3 events on Jan 2nd
    assert total_events == 3

    distinct_user_sql = "SELECT COUNT(DISTINCT user_id) FROM events WHERE event_time >= '2024-01-02' AND event_time < '2024-01-03'"
    unique_users = analytics_db.execute(distinct_user_sql).fetchone()[0]
    # Only Alice (1 user) was active
    assert unique_users == 1


def test_safe_division_prevents_zero_division_runtime_crash(analytics_db):
    """Proves that compiler-enforced safe division prevents runtime division-by-zero crashes on DuckDB."""
    compiler = SQLCompiler()

    # Query dividing shipping by total_amount where user 1 order 105 has total_amount = 0
    step = CTEPlanStep(
        step_name="order_ratios",
        purpose="Compute shipping ratio with possible 0 total_amount",
        source_tables=["orders"],
        math_logic="shipping_cost / total_amount",
        sql_fragment="SELECT order_id, shipping_cost / total_amount AS ratio FROM orders WHERE order_id = 105",
    )
    plan = LogicPlan(
        query="Order 105 ratio",
        target_dialect="duckdb",
        strategy="Safe division check",
        cte_steps=[step],
        final_select="SELECT order_id, ratio FROM order_ratios",
    )

    # Compiler automatically guards denominator with NULLIF(total_amount, 0)
    compiled_sql = compiler.compile(plan)
    assert "NULLIF" in compiled_sql

    result = analytics_db.execute(compiled_sql).fetchall()
    # Returns (105, None) safely instead of crashing DuckDB with division by zero
    assert len(result) == 1
    assert result[0][0] == 105
    assert result[0][1] is None


def test_deterministic_ordering_with_tie_breaker(analytics_db):
    """Proves that ranking and ordering with ties requires deterministic secondary keys."""
    # Insert tied order
    analytics_db.execute("INSERT INTO orders VALUES (106, 2, DATE '2024-01-05', 200.0, 10.0)")

    # Both order 102 and order 106 have total_amount = 200.0
    sql = "SELECT order_id, total_amount FROM orders WHERE total_amount = 200.0 ORDER BY total_amount DESC, order_id ASC"
    rows = analytics_db.execute(sql).fetchall()

    # Must be stably ordered by order_id ASC: 102 before 106
    assert rows[0][0] == 102
    assert rows[1][0] == 106


def test_time_window_boundary_semantics(analytics_db):
    """Proves exact filtering on date window boundaries [start, end)."""
    # Half-open interval [2024-01-02, 2024-01-04) excludes 2024-01-04
    sql_half_open = "SELECT order_id FROM orders WHERE order_date >= DATE '2024-01-02' AND order_date < DATE '2024-01-04' ORDER BY order_id"
    half_open_orders = [r[0] for r in analytics_db.execute(sql_half_open).fetchall()]
    assert half_open_orders == [101, 102]

    # Closed interval BETWEEN includes 2024-01-04
    sql_closed = "SELECT order_id FROM orders WHERE order_date BETWEEN DATE '2024-01-02' AND DATE '2024-01-04' ORDER BY order_id"
    closed_orders = [r[0] for r in analytics_db.execute(sql_closed).fetchall()]
    assert 103 in closed_orders


def test_cohort_retention_exact_fixture_calculation(analytics_db):
    """Proves cohort retention calculation against the deterministic fixture database."""
    compiler = SQLCompiler()

    step1 = CTEPlanStep(
        step_name="cohort_base",
        purpose="Assign cohort week to users",
        source_tables=["users"],
        math_logic="date_trunc('week', signup_date) AS cohort_week",
        sql_fragment="SELECT user_id, date_trunc('week', signup_date) AS cohort_week FROM users",
    )
    step2 = CTEPlanStep(
        step_name="cohort_sizes",
        purpose="Count users in each cohort",
        source_tables=["cohort_base"],
        math_logic="COUNT(DISTINCT user_id) AS cohort_size",
        sql_fragment="SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_size FROM cohort_base GROUP BY cohort_week",
    )
    plan = LogicPlan(
        query="Cohort sizes",
        target_dialect="duckdb",
        strategy="Calculate cohort sizes",
        cte_steps=[step1, step2],
        final_select="SELECT cohort_week, cohort_size FROM cohort_sizes ORDER BY cohort_week",
    )

    sql = compiler.compile(plan)
    rows = analytics_db.execute(sql).fetchall()

    # Week 1 (2024-01-01) has Alice & Bob (2 users)
    # Week 2 (2024-01-08) has Charlie (1 user)
    assert len(rows) == 2
    assert rows[0][1] == 2
    assert rows[1][1] == 1
