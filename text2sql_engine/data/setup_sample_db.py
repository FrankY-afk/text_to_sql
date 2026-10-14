"""Sample database generator for DuckDB containing flat relational and nested STRUCT/ARRAY tables."""

from __future__ import annotations

import os
from datetime import date
from typing import Optional
import duckdb


def init_sample_database(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """Initializes sample DuckDB database with realistic e-commerce and GA4 analytics tables."""
    if db_path:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        con = duckdb.connect(db_path)
    else:
        con = duckdb.connect(":memory:")

    # Drop existing tables if re-initializing
    con.execute("DROP TABLE IF EXISTS order_items;")
    con.execute("DROP TABLE IF EXISTS web_events;")
    con.execute("DROP TABLE IF EXISTS orders;")
    con.execute("DROP TABLE IF EXISTS customers;")

    # 1. Flat Relational Table: customers
    con.execute("""
    CREATE TABLE customers (
        customer_id VARCHAR PRIMARY KEY,
        customer_name VARCHAR,
        state_code VARCHAR,
        signup_date DATE,
        is_active BOOLEAN
    );
    """)

    con.execute("""
    INSERT INTO customers VALUES
        ('C001', 'Alice Johnson', 'CA', DATE '2023-01-15', TRUE),
        ('C002', 'Bob Smith', 'NY', DATE '2023-02-20', TRUE),
        ('C003', 'Carol Davis', 'CA', DATE '2023-03-10', TRUE),
        ('C004', 'David Wilson', 'TX', DATE '2023-04-05', FALSE),
        ('C005', 'Eva Martinez', 'CA', DATE '2023-05-18', TRUE);
    """)

    # 2. Flat Relational Table: orders
    con.execute("""
    CREATE TABLE orders (
        order_id VARCHAR PRIMARY KEY,
        customer_id VARCHAR,
        order_status VARCHAR,
        order_date DATE,
        total_amount DOUBLE,
        shipping_cost DOUBLE
    );
    """)

    con.execute("""
    INSERT INTO orders VALUES
        ('O101', 'C001', 'COMPLETED', DATE '2024-01-10', 250.00, 15.00),
        ('O102', 'C001', 'COMPLETED', DATE '2024-02-14', 180.00, 10.00),
        ('O103', 'C002', 'COMPLETED', DATE '2024-01-20', 420.00, 20.00),
        ('O104', 'C003', 'COMPLETED', DATE '2024-03-01', 90.00, 5.00),
        ('O105', 'C003', 'CANCELLED', DATE '2024-03-15', 300.00, 0.00),
        ('O106', 'C005', 'COMPLETED', DATE '2024-04-12', 510.00, 25.00);
    """)

    # 3. Flat Relational Table: order_items
    con.execute("""
    CREATE TABLE order_items (
        item_id VARCHAR PRIMARY KEY,
        order_id VARCHAR,
        product_category VARCHAR,
        quantity INTEGER,
        unit_price DOUBLE
    );
    """)

    con.execute("""
    INSERT INTO order_items VALUES
        ('I01', 'O101', 'electronics', 1, 250.00),
        ('I02', 'O102', 'apparel', 2, 90.00),
        ('I03', 'O103', 'electronics', 1, 420.00),
        ('I04', 'O104', 'books', 3, 30.00),
        ('I05', 'O105', 'furniture', 1, 300.00),
        ('I06', 'O106', 'electronics', 2, 255.00);
    """)

    # 4. Nested STRUCT and ARRAY Table: web_events (GA4 Style)
    con.execute("""
    CREATE TABLE web_events (
        event_id VARCHAR PRIMARY KEY,
        user_id VARCHAR,
        event_name VARCHAR,
        event_timestamp BIGINT,
        geo_info STRUCT(country VARCHAR, region VARCHAR, city VARCHAR),
        event_params STRUCT(key VARCHAR, value STRUCT(string_value VARCHAR, int_value BIGINT))[]
    );
    """)

    con.execute("""
    INSERT INTO web_events VALUES
        (
            'EV01',
            'C001',
            'purchase',
            1704931200000,
            {'country': 'USA', 'region': 'California', 'city': 'San Francisco'},
            [
                {'key': 'campaign', 'value': {'string_value': 'winter_sale', 'int_value': NULL}},
                {'key': 'item_count', 'value': {'string_value': NULL, 'int_value': 1}}
            ]
        ),
        (
            'EV02',
            'C002',
            'purchase',
            1705708800000,
            {'country': 'USA', 'region': 'New York', 'city': 'New York'},
            [
                {'key': 'campaign', 'value': {'string_value': 'new_year', 'int_value': NULL}},
                {'key': 'item_count', 'value': {'string_value': NULL, 'int_value': 2}}
            ]
        ),
        (
            'EV03',
            'C003',
            'purchase',
            1709251200000,
            {'country': 'USA', 'region': 'California', 'city': 'Los Angeles'},
            [
                {'key': 'campaign', 'value': {'string_value': 'spring_kickoff', 'int_value': NULL}},
                {'key': 'item_count', 'value': {'string_value': NULL, 'int_value': 3}}
            ]
        );
    """)

    return con


if __name__ == "__main__":
    from text2sql_engine.core.config import get_settings
    settings = get_settings()
    db_file = settings.duckdb_path
    con = init_sample_database(db_file)
    print(f"Sample DuckDB created at {db_file}")
    tables = con.execute("SHOW TABLES;").fetchall()
    print("Tables created:", [t[0] for t in tables])
    con.close()

