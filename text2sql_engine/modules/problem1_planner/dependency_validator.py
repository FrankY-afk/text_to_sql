"""CTE Dependency and Semantic Validator using SQLGlot AST inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import sqlglot
from sqlglot import exp

from text2sql_engine.core.exceptions import DependencyValidationError, PlanValidationError
from text2sql_engine.modules.problem1_planner.schemas import CTEPlanStep, LogicPlan


@dataclass
class ValidationReport:
    """Detailed summary of CTE dependency and semantic verification."""

    is_valid: bool
    verified_tables: Set[str] = field(default_factory=set)
    unverified_tables: Set[str] = field(default_factory=set)
    verified_columns: Set[str] = field(default_factory=set)
    unverified_columns: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class CTEDependencyValidator:
    """Validates CTE topological ordering, dependency acyclicity, and column resolutions."""

    @classmethod
    def validate_plan(
        cls,
        plan: LogicPlan,
        available_base_tables: Optional[Set[str]] = None,
        table_columns_map: Optional[Dict[str, Set[str]]] = None,
    ) -> ValidationReport:
        """Validates the complete LogicPlan against dependency, ordering, and AST constraints.

        Args:
            plan: The LogicPlan to validate.
            available_base_tables: Optional set of allowable base tables from the database schema.
            table_columns_map: Optional mapping of base table names to sets of known column names.

        Returns:
            ValidationReport detailing verified tables, columns, and analytical notes.

        Raises:
            DependencyValidationError: If dependencies, references, or columns fail validation.
            PlanValidationError: If the plan violates structural or AST requirements.
        """
        report = ValidationReport(is_valid=False)

        if not plan.cte_steps:
            # If no CTEs, validate final_select directly
            cls._validate_final_select(
                plan.final_select,
                defined_ctes=set(),
                available_base_tables=available_base_tables,
                report=report,
            )
            report.is_valid = True
            return report

        # 1. Check CTE uniqueness
        cte_names: List[str] = []
        cte_index_map: Dict[str, int] = {}
        for idx, step in enumerate(plan.cte_steps):
            name_lower = step.step_name.lower()
            if name_lower in cte_index_map:
                raise DependencyValidationError(
                    f"Duplicate CTE step_name '{step.step_name}' detected at index {idx}."
                )
            cte_names.append(name_lower)
            cte_index_map[name_lower] = idx

        # 2. Build dependency graph and inspect AST of each CTE
        dep_graph: Dict[str, Set[str]] = {name: set() for name in cte_names}
        columns_produced_map: Dict[str, Set[str]] = {}

        for idx, step in enumerate(plan.cte_steps):
            step_name = step.step_name.lower()
            upstream_available_ctes = set(cte_names[:idx])
            downstream_ctes = set(cte_names[idx + 1:])

            # Parse SQL fragment
            try:
                ast = sqlglot.parse_one(step.sql_fragment, read="duckdb")
            except Exception as e:
                raise PlanValidationError(
                    f"CTE '{step.step_name}' contains malformed SQL fragment: {e}"
                ) from e

            if not isinstance(ast, exp.Select):
                raise PlanValidationError(
                    f"CTE '{step.step_name}' must be a SELECT statement, got {type(ast).__name__}."
                )

            # Inspect AST table references
            ast_tables = cls._extract_table_references(ast)

            # Record columns produced for column verification
            cols_prod = {c.lower() for c in step.columns_produced if c != "*"}
            columns_produced_map[step_name] = cols_prod
            for col in cols_prod:
                report.verified_columns.add(f"{step_name}.{col}")

            # Check each referenced table in AST
            for tbl in ast_tables:
                tbl_lower = tbl.lower()

                # Self-reference check
                if tbl_lower == step_name:
                    raise DependencyValidationError(
                        f"CTE '{step.step_name}' contains a recursive self-reference to '{tbl}'."
                    )

                # Forward reference check
                if tbl_lower in downstream_ctes:
                    raise DependencyValidationError(
                        f"CTE '{step.step_name}' contains forward reference to downstream CTE '{tbl}'. "
                        f"CTEs must be defined in dependency order."
                    )

                # Check if it resolves to upstream CTE or allowable base table
                if tbl_lower in upstream_available_ctes:
                    dep_graph[step_name].add(tbl_lower)
                    report.verified_tables.add(tbl_lower)
                elif available_base_tables is not None:
                    base_tables_lower = {b.lower() for b in available_base_tables}
                    if tbl_lower not in base_tables_lower:
                        raise DependencyValidationError(
                            f"CTE '{step.step_name}' references unknown table or undefined CTE '{tbl}'. "
                            f"Available upstream CTEs: {sorted(upstream_available_ctes)}, "
                            f"Available base tables: {sorted(base_tables_lower)}"
                        )
                    report.verified_tables.add(tbl_lower)
                else:
                    report.unverified_tables.add(tbl_lower)
                    report.notes.append(
                        f"Table '{tbl}' referenced in CTE '{step.step_name}' could not be verified against base catalog (no catalog provided)."
                    )

            # Check consistency with declared source_tables
            for src in step.source_tables:
                src_lower = src.lower()
                if src_lower in downstream_ctes:
                    raise DependencyValidationError(
                        f"CTE '{step.step_name}' lists downstream CTE '{src}' in source_tables (forward reference)."
                    )

            # Check column availability from upstream CTEs if qualified
            cls._validate_column_references(ast, columns_produced_map, current_cte=step.step_name, report=report)

        # 3. Detect cyclic dependencies across the CTE graph
        cls._detect_cycles(dep_graph)

        # 4. Validate final_select
        cls._validate_final_select(
            plan.final_select,
            defined_ctes=set(cte_names),
            available_base_tables=available_base_tables,
            columns_produced_map=columns_produced_map,
            report=report,
        )

        report.is_valid = True
        return report

    @classmethod
    def _extract_table_references(cls, ast: exp.Expression) -> Set[str]:
        """Extracts table and CTE relation names referenced in a query AST, ignoring aliases."""
        tables = set()
        for tbl in ast.find_all(exp.Table):
            if tbl.name:
                tables.add(tbl.name)
        return tables

    @classmethod
    def _validate_column_references(
        cls,
        ast: exp.Expression,
        columns_produced_map: Dict[str, Set[str]],
        current_cte: str,
        report: Optional[ValidationReport] = None,
    ) -> None:
        """Validates that explicit qualified column references (e.g. cte.col) exist in upstream CTE columns_produced."""
        for col_expr in ast.find_all(exp.Column):
            table_qualifier = col_expr.table.lower() if col_expr.table else None
            col_name = col_expr.name.lower() if col_expr.name else None

            if table_qualifier and table_qualifier in columns_produced_map:
                produced = columns_produced_map[table_qualifier]
                if produced and col_name not in produced:
                    raise DependencyValidationError(
                        f"In CTE '{current_cte}', qualified column reference '{table_qualifier}.{col_name}' "
                        f"was not found in columns produced by upstream CTE '{table_qualifier}': {sorted(produced)}"
                    )
                if report and col_name in produced:
                    report.verified_columns.add(f"{table_qualifier}.{col_name}")

    @classmethod
    def _detect_cycles(cls, graph: Dict[str, Set[str]]) -> None:
        """Detects cycles in directed dependency graph using DFS coloring."""
        visited: Dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited

        def dfs(node: str, path: List[str]) -> None:
            visited[node] = 1
            for neighbor in graph.get(node, set()):
                if visited.get(neighbor, 0) == 1:
                    cycle = " -> ".join(path + [neighbor])
                    raise DependencyValidationError(f"Cyclic dependency detected among CTEs: {cycle}")
                elif visited.get(neighbor, 0) == 0:
                    dfs(neighbor, path + [neighbor])
            visited[node] = 2

        for node in graph:
            if visited.get(node, 0) == 0:
                dfs(node, [node])

    @classmethod
    def _validate_final_select(
        cls,
        final_select: str,
        defined_ctes: Set[str],
        available_base_tables: Optional[Set[str]] = None,
        columns_produced_map: Optional[Dict[str, Set[str]]] = None,
        report: Optional[ValidationReport] = None,
    ) -> None:
        """Validates the final SELECT projection."""
        try:
            ast = sqlglot.parse_one(final_select, read="duckdb")
        except Exception as e:
            raise PlanValidationError(f"final_select contains malformed SQL: {e}") from e

        if not isinstance(ast, exp.Select):
            raise PlanValidationError(
                f"final_select must be a SELECT statement, got {type(ast).__name__}."
            )

        tables = cls._extract_table_references(ast)
        for tbl in tables:
            tbl_lower = tbl.lower()
            is_in_ctes = tbl_lower in defined_ctes
            is_in_base = (
                tbl_lower in {b.lower() for b in available_base_tables}
                if available_base_tables is not None
                else False
            )
            if not is_in_ctes and not is_in_base and available_base_tables is not None:
                raise DependencyValidationError(
                    f"final_select references unknown relation '{tbl}'. "
                    f"Available CTEs: {sorted(defined_ctes)}, Available base tables: {sorted(available_base_tables)}"
                )
            if report:
                if is_in_ctes or is_in_base:
                    report.verified_tables.add(tbl_lower)
                else:
                    report.unverified_tables.add(tbl_lower)

        if columns_produced_map:
            cls._validate_column_references(ast, columns_produced_map, current_cte="final_select", report=report)
