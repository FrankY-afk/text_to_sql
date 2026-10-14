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
        source_dialect: Optional[str] = None,
    ) -> str:
        """Compiles, validates, guards safe division, and transpiles a LogicPlan into the requested SQL dialect.

        Args:
            plan: Validated LogicPlan.
            target_dialect: Target SQL dialect (defaults to plan.target_dialect).
            source_dialect: Source SQL dialect of the plan fragments (defaults to 'duckdb').

        Returns:
            Transpiled, verified SQL query string with safe division enforced.

        Raises:
            UnsupportedDialectError: If target or source dialect is not supported.
            SecurityValidationError: If SQL contains forbidden DDL/DML or multiple statements.
            SQLCompilationError: If AST parsing, safe division transformation, or target verification fails.
        """
        # Validate target dialect
        raw_target = target_dialect or plan.target_dialect
        try:
            target_norm = normalize_dialect(raw_target)
        except ValueError as e:
            raise UnsupportedDialectError(f"Unsupported target dialect '{raw_target}': {e}") from e

        sqlglot_target = self.SQLGLOT_DIALECT_MAP.get(target_norm)
        if not sqlglot_target:
            raise UnsupportedDialectError(
                f"No SQLGlot target dialect mapping found for '{target_norm}'. Supported dialects: {sorted(SUPPORTED_DIALECTS)}"
            )

        # Validate source dialect
        src_raw = source_dialect or "duckdb"
        try:
            source_norm = normalize_dialect(src_raw)
        except ValueError as e:
            raise UnsupportedDialectError(f"Unsupported source dialect '{src_raw}': {e}") from e

        sqlglot_source = self.SQLGLOT_DIALECT_MAP.get(source_norm, "duckdb")

        # 1. Validate that no individual CTE fragment or final_select contains multiple statements
        for step in plan.cte_steps:
            frag_clean = step.sql_fragment.strip().rstrip(";")
            try:
                frag_exprs = [e for e in sqlglot.parse(frag_clean, read=sqlglot_source) if e is not None]
                if len(frag_exprs) > 1:
                    raise SecurityValidationError(
                        f"Multiple SQL statements detected in CTE '{step.step_name}'. Only a single SELECT query is permitted."
                    )
            except SecurityValidationError:
                raise
            except Exception as e:
                raise SQLCompilationError(f"Failed to parse query in CTE '{step.step_name}': {e}") from e

        final_clean = plan.final_select.strip().rstrip(";")
        try:
            final_exprs = [e for e in sqlglot.parse(final_clean, read=sqlglot_source) if e is not None]
            if len(final_exprs) > 1:
                raise SecurityValidationError(
                    "Multiple SQL statements detected in final_select. Only a single SELECT query is permitted."
                )
        except SecurityValidationError:
            raise
        except Exception as e:
            raise SQLCompilationError(f"Failed to parse query in final_select: {e}") from e


        # 2. Assemble raw SQL from CTE steps and final_select
        raw_sql = self._assemble_raw_sql(plan)

        # 3. Stage 1: Parse and enforce statement safety on source SQL
        source_ast = self._parse_and_validate_safety(raw_sql, source_dialect=sqlglot_source)

        # 4. Stage 2: Enforce Safe Division (Phase B2)
        safe_ast = self.enforce_safe_division(source_ast)

        # 5. Stage 3: Transpile to target dialect using SQLGlot
        try:
            transpiled_sql = safe_ast.sql(dialect=sqlglot_target, pretty=True).strip()
            if not transpiled_sql:
                raise SQLCompilationError(
                    f"Transpilation stage failed: SQLGlot produced empty output for dialect '{target_norm}'."
                )
        except SQLCompilationError:
            raise
        except Exception as e:
            raise SQLCompilationError(
                f"SQLGlot transpilation failed from '{source_norm}' to '{target_norm}': {str(e)}"
            ) from e

        # 6. Stage 4: Parse and verify generated SQL in target dialect
        self._verify_target_sql(transpiled_sql, target_dialect=sqlglot_target)

        return transpiled_sql

    @classmethod
    def enforce_safe_division(cls, ast: exp.Expression) -> exp.Expression:
        """Enforces safe division semantics by guarding division denominators with NULLIF(..., 0).

        Rules:
        - If denominator is a non-zero numeric literal (e.g. 100, 1.0), it is preserved as safe.
        - If denominator is a literal zero (0 or 0.0), raises SQLCompilationError.
        - If denominator is already a NULLIF(..., 0), it is preserved.
        - Otherwise, denominator is wrapped in NULLIF(denominator, 0).
        """
        def _transform(node: exp.Expression) -> exp.Expression:
            if isinstance(node, (exp.Div, exp.IntDiv)):
                denom = node.expression
                if denom is None:
                    return node

                # If denominator is already NULLIF(..., 0)
                if isinstance(denom, exp.Nullif):
                    sec = denom.expression
                    if isinstance(sec, exp.Literal) and sec.this in ("0", 0, "0.0"):
                        return node

                # If denominator is a numeric literal
                if isinstance(denom, exp.Literal) and denom.is_number:
                    try:
                        val = float(denom.this)
                        if val != 0.0:
                            return node
                        else:
                            raise SQLCompilationError(
                                "Literal division by zero detected in query."
                            )
                    except ValueError:
                        pass

                # Guard denominator with NULLIF(denom, 0)
                safe_denom = exp.Nullif(this=denom.copy(), expression=exp.Literal.number(0))
                node.set("expression", safe_denom)
                return node

            return node

        return ast.transform(_transform)

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
        try:
            parsed_expressions = sqlglot.parse(sql_text, read=source_dialect)
        except Exception as e:
            raise SQLCompilationError(f"Failed to parse query in source dialect '{source_dialect}': {e}") from e

        non_empty = [expr for expr in parsed_expressions if expr is not None]
        if not non_empty:
            raise SQLCompilationError("Source query is empty or contains no valid SQL expressions.")
        if len(non_empty) > 1:
            raise SecurityValidationError(
                f"Multiple SQL statements detected ({len(non_empty)} statements). Only a single SELECT query is permitted."
            )

        ast = non_empty[0]

        if not isinstance(ast, exp.Select):
            raise SecurityValidationError(
                f"Only read-only SELECT queries are permitted. Detected statement type: {type(ast).__name__}."
            )

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
