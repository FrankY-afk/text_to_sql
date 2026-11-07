"""SQL syntax, AST security, and read-only policy validator using SQLGlot.

Ensures generated SQL is syntactically valid, single-statement, strictly read-only,
and free from destructive DDL/DML, file access, administrative commands, or dialect-forbidden functions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field
import sqlglot
from sqlglot import exp

from src.cte_planner import normalize_dialect
from src.dialect_rag import DialectFunctionRAG

logger = logging.getLogger(__name__)

# Disallowed AST node types that modify data or schema
FORBIDDEN_EXPRESSION_TYPES: Set[type] = {
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
    exp.TruncateTable,
    exp.Merge,
}

# Dangerous system functions that should not be invoked by LLM-generated queries
FORBIDDEN_FUNCTIONS: Set[str] = {
    "pg_sleep",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_read_file",
    "pg_write_file",
    "dblink",
    "dblink_exec",
    "query_to_xml",
    "lo_import",
    "lo_export",
}


class SQLValidationReport(BaseModel):
    """Structured report returned after parsing and inspecting a SQL query."""

    is_valid: bool = Field(description="True if query passes all syntax and security checks")
    error: Optional[str] = Field(default=None, description="Detailed error description if validation failed")
    is_readonly: bool = Field(default=False, description="True if query contains only read-only statements")
    statement_type: Optional[str] = Field(default=None, description="Primary AST statement type (e.g., Select)")
    tables_referenced: List[str] = Field(default_factory=list, description="List of table names referenced")
    has_aggregation: bool = Field(default=False, description="True if query uses aggregate functions")
    has_distinct: bool = Field(default=False, description="True if query contains DISTINCT clauses")
    has_ctes: bool = Field(default=False, description="True if query defines Common Table Expressions (WITH clause)")
    cte_names: List[str] = Field(default_factory=list, description="List of CTE names defined in query")
    dialect_violations: List[str] = Field(default_factory=list, description="Dialect-specific syntax or function violations")


def validate_readonly_sql(sql: str, dialect: str = "postgres") -> SQLValidationReport:
    """Validates that a SQL string is syntactically sound and strictly read-only.

    Args:
        sql: Raw SQL query string.
        dialect: SQL dialect for SQLGlot parser (default: 'postgres').

    Returns:
        SQLValidationReport with validation status, error messages, and AST summary.
    """
    clean_sql = sql.strip().rstrip(";")
    if not clean_sql:
        return SQLValidationReport(is_valid=False, error="Query cannot be empty.")

    norm_dialect = normalize_dialect(dialect)

    # 1. Parse into AST statements
    try:
        statements = sqlglot.parse(clean_sql, read=norm_dialect)
    except Exception as exc:
        return SQLValidationReport(is_valid=False, error=f"SQL syntax error: {exc}")

    if not statements:
        return SQLValidationReport(is_valid=False, error="No valid SQL statements found.")

    # 2. Reject multiple statements to prevent batch injection
    if len(statements) > 1:
        return SQLValidationReport(
            is_valid=False,
            error=f"Multiple SQL statements detected ({len(statements)} statements). Only single statements are permitted.",
        )

    ast = statements[0]
    if ast is None:
        return SQLValidationReport(is_valid=False, error="Failed to parse SQL into an abstract syntax tree.")

    statement_type = ast.__class__.__name__

    # 3. Root statement must be a Select or Union
    if not isinstance(ast, (exp.Select, exp.Union)):
        return SQLValidationReport(
            is_valid=False,
            is_readonly=False,
            statement_type=statement_type,
            error=f"Statement of type '{statement_type}' is not permitted. Only SELECT queries are allowed.",
        )

    # 4. Check entire AST for forbidden DDL / DML operations
    for forbidden_type in FORBIDDEN_EXPRESSION_TYPES:
        match = ast.find(forbidden_type)
        if match is not None:
            return SQLValidationReport(
                is_valid=False,
                is_readonly=False,
                statement_type=statement_type,
                error=f"Forbidden operation '{forbidden_type.__name__}' detected inside query.",
            )

    # 5. Check for dangerous system function calls
    for func in ast.find_all(exp.Anonymous, exp.Func):
        func_name = func.name.lower() if hasattr(func, "name") and func.name else ""
        if func_name in FORBIDDEN_FUNCTIONS:
            return SQLValidationReport(
                is_valid=False,
                is_readonly=False,
                statement_type=statement_type,
                error=f"Execution of forbidden system function '{func_name}' is blocked.",
            )

    # 6. Check Dialect Function Compatibility
    rag = DialectFunctionRAG()
    dialect_violations = rag.validate_sql_dialect_compatibility(clean_sql, target_dialect=norm_dialect)
    if dialect_violations:
        return SQLValidationReport(
            is_valid=False,
            is_readonly=True,
            statement_type=statement_type,
            error="; ".join(dialect_violations),
            dialect_violations=dialect_violations,
        )

    # 7. Extract CTE information and verify CTE ordering
    cte_nodes = list(ast.find_all(exp.CTE))
    cte_names: List[str] = []
    if cte_nodes:
        defined_so_far: Set[str] = set()
        for cte in cte_nodes:
            c_alias = cte.alias_or_name.lower() if cte.alias_or_name else ""
            if c_alias:
                cte_names.append(c_alias)
                # Check for forward-reference within CTE body
                body_tables = {t.name.lower() for t in cte.find_all(exp.Table) if t.name}
                # If body_tables references another CTE not yet defined
                future_ctes = {other.alias_or_name.lower() for other in cte_nodes if other != cte and other.alias_or_name.lower() not in defined_so_far}
                leaked_forward = body_tables.intersection(future_ctes)
                if leaked_forward:
                    return SQLValidationReport(
                        is_valid=False,
                        is_readonly=True,
                        statement_type=statement_type,
                        cte_names=cte_names,
                        has_ctes=True,
                        error=f"CTE dependency forward-reference: CTE '{c_alias}' references '{list(leaked_forward)[0]}', which is defined later.",
                    )
                defined_so_far.add(c_alias)

    # 8. Extract referenced base tables
    all_tables = [t.name for t in ast.find_all(exp.Table) if t.name]
    cte_names_set = set(cte_names)
    base_tables = sorted(list({t for t in all_tables if t.lower() not in cte_names_set}))

    # 9. Check for aggregation and distinct
    has_agg = bool(list(ast.find_all(exp.AggFunc)))
    has_dist = bool(ast.find(exp.Distinct)) or any(
        getattr(f, "distinct", False) for f in ast.find_all(exp.AggFunc)
    )

    return SQLValidationReport(
        is_valid=True,
        is_readonly=True,
        statement_type=statement_type,
        tables_referenced=base_tables,
        has_aggregation=has_agg,
        has_distinct=has_dist,
        has_ctes=bool(cte_names),
        cte_names=cte_names,
        dialect_violations=dialect_violations,
    )
