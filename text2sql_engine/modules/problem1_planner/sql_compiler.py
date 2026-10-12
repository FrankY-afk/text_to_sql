"""Dialect-Aware SQL Compiler using SQLGlot AST transpilation and fail-closed validation."""

from __future__ import annotations

from typing import Optional, Set
import sqlglot
from sqlglot import exp

from text2sql_engine.core.exceptions import (
    SQLCompilationError,
    SecurityValidationError,
    UnsupportedDialectError,
)
from text2sql_engine.modules.problem1_planner.schemas import (
    SUPPORTED_DIALECTS,
    LogicPlan,
    normalize_dialect,
)


class SQLCompiler:
    """Compiles validated LogicPlan into target SQL dialect with strict validation and fail-closed semantics."""

    # Map engine dialect identifiers to SQLGlot dialect identifiers
    SQLGLOT_DIALECT_MAP = {
        "duckdb": "duckdb",
        "postgres": "postgres",
        "postgresql": "postgres",
        "bigquery": "bigquery",
        "snowflake": "snowflake",
        "sqlite": "sqlite",
    }

    # Prohibited expression types (DDL, DML, transaction control)
    FORBIDDEN_EXPRESSION_TYPES = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Command,
        exp.Commit,
        exp.Rollback,
        exp.Pragma,
    )

    def compile(
        self,
        plan: LogicPlan,
        target_dialect: Optional[str] = None,
    ) -> str:
        """Compiles and transpiles a LogicPlan into the requested SQL dialect.

        Args:
            plan: Validated LogicPlan.
            target_dialect: Optional override dialect; defaults to plan.target_dialect.

        Returns:
            Transpiled, verified SQL query string.

        Raises:
            UnsupportedDialectError: If target dialect is not supported.
            SecurityValidationError: If SQL contains forbidden DDL/DML or multiple statements.
            SQLCompilationError: If AST parsing, transpilation, or target verification fails.
        """
        raw_dialect = target_dialect or plan.target_dialect
        try:
            dialect = normalize_dialect(raw_dialect)
        except ValueError as e:
            raise UnsupportedDialectError(str(e)) from e

        sqlglot_target = self.SQLGLOT_DIALECT_MAP.get(dialect)
        if not sqlglot_target:
            raise UnsupportedDialectError(
                f"No SQLGlot target dialect mapping found for '{dialect}'. Supported dialects: {sorted(SUPPORTED_DIALECTS)}"
            )

        # 1. Validate that no individual CTE fragment or final_select contains multiple statements
        for step in plan.cte_steps:
            frag_clean = step.sql_fragment.strip().rstrip(";")
            try:
                frag_exprs = [e for e in sqlglot.parse(frag_clean, read="duckdb") if e is not None]
                if len(frag_exprs) > 1:
                    raise SecurityValidationError(
                        f"Multiple SQL statements detected in CTE '{step.step_name}'. Only a single SELECT query is permitted."
                    )
            except SecurityValidationError:
                raise
            except Exception:
                pass

        final_clean = plan.final_select.strip().rstrip(";")
        try:
            final_exprs = [e for e in sqlglot.parse(final_clean, read="duckdb") if e is not None]
            if len(final_exprs) > 1:
                raise SecurityValidationError(
                    "Multiple SQL statements detected in final_select. Only a single SELECT query is permitted."
                )
        except SecurityValidationError:
            raise
        except Exception:
            pass

        # 2. Assemble raw SQL from CTE steps and final_select
        raw_sql = self._assemble_raw_sql(plan)

        # 3. Stage 1: Parse and enforce statement safety on source SQL (DuckDB dialect)
        source_ast = self._parse_and_validate_safety(raw_sql, source_dialect="duckdb")

        # 3. Stage 2: Transpile to target dialect using SQLGlot
        try:
            transpiled_list = sqlglot.transpile(
                raw_sql,
                read="duckdb",
                write=sqlglot_target,
                pretty=True,
            )
            if not transpiled_list:
                raise SQLCompilationError(
                    f"Transpilation stage failed: SQLGlot produced empty output for dialect '{dialect}'."
                )
            transpiled_sql = transpiled_list[0].strip()
        except Exception as e:
            raise SQLCompilationError(
                f"SQLGlot transpilation failed from 'duckdb' to '{dialect}': {str(e)}"
            ) from e

        # 4. Stage 3: Parse and verify generated SQL in target dialect
        self._verify_target_sql(transpiled_sql, target_dialect=sqlglot_target)

        return transpiled_sql

    def _assemble_raw_sql(self, plan: LogicPlan) -> str:
        """Assembles WITH CTE chain and final SELECT in topological dependency order."""
        if plan.cte_steps:
            cte_definitions = []
            for step in plan.cte_steps:
                frag = step.sql_fragment.strip().rstrip(";")
                cte_definitions.append(f"{step.step_name} AS (\n  {frag}\n)")
            raw_sql = f"WITH {', '.join(cte_definitions)}\n{plan.final_select.strip().rstrip(';')}"
        else:
            raw_sql = plan.final_select.strip().rstrip(";")
        return raw_sql

    def _parse_and_validate_safety(self, sql_text: str, source_dialect: str = "duckdb") -> exp.Expression:
        """Parses source SQL and rejects multi-statements, DDL, and DML."""
        # Check for multiple statements
        try:
            parsed_expressions = sqlglot.parse(sql_text, read=source_dialect)
        except Exception as e:
            raise SQLCompilationError(f"Failed to parse query in source dialect '{source_dialect}': {e}") from e

        # Filter out empty statements
        non_empty = [expr for expr in parsed_expressions if expr is not None]
        if not non_empty:
            raise SQLCompilationError("Source query is empty or contains no valid SQL expressions.")
        if len(non_empty) > 1:
            raise SecurityValidationError(
                f"Multiple SQL statements detected ({len(non_empty)} statements). Only a single SELECT query is permitted."
            )

        ast = non_empty[0]

        # Enforce statement safety: only SELECT statements allowed
        if not isinstance(ast, exp.Select):
            raise SecurityValidationError(
                f"Only read-only SELECT queries are permitted. Detected statement type: {type(ast).__name__}."
            )

        # Inspect AST for any nested forbidden statements (e.g. injected DROP/INSERT)
        for forbidden_cls in self.FORBIDDEN_EXPRESSION_TYPES:
            if list(ast.find_all(forbidden_cls)):
                raise SecurityValidationError(
                    f"Forbidden SQL operation detected: {forbidden_cls.__name__}. Only read-only queries are permitted."
                )

        return ast

    def _verify_target_sql(self, sql_text: str, target_dialect: str) -> None:
        """Verifies that the compiled SQL is syntactically valid in the target dialect."""
        try:
            target_ast = sqlglot.parse_one(sql_text, read=target_dialect)
        except Exception as e:
            raise SQLCompilationError(
                f"Compiled query failed validation in target dialect '{target_dialect}': {e}"
            ) from e

        if not isinstance(target_ast, exp.Select):
            raise SQLCompilationError(
                f"Compiled query in dialect '{target_dialect}' parsed to unexpected expression type: {type(target_ast).__name__}."
            )
