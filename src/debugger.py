"""Execution-grounded debugging loop for Problem 1: Erroneous Data Analysis.

Implements the cyclic reasoning and self-repair workflow:
User Question
    ↓
Understand Analytical Intent
    ↓
Retrieve Database Schema (MCP)
    ↓
Retrieve Dialect Documentation (DialectFunctionRAG)
    ↓
Generate Analytical CTE Plan (CTEPlanner)
    ↓
Validate Analytical Plan (CTEDependencyValidator)
    ↓
Generate SQL (Hugging Face / CTE Compiler)
    ↓
SQLGlot AST Validation (SQLValidator)
    ↓
Semantic Validation (SemanticValidator)
    ↓
Database EXPLAIN / Dry Run (MCP / Database)
    ↓
If failure → Debugger (Classify & Advise) → Regenerate / Repair
    ↓
If successful → Controlled Safe Execution (MCP)
    ↓
Evaluate Business Result
"""

from __future__ import annotations

from enum import Enum
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.cte_planner import CTEPlanCompiler, CTEPlanner, LogicPlan
from src.dialect_rag import DialectFunctionRAG
from src.hf_client import HuggingFaceClient
from src.literal_resolver import LiteralResolver
from src.query_executor import MCPQueryExecutor
from src.schema_inspector import SchemaInspector
from src.semantic_validator import (
    AnalyticalContract,
    SemanticValidationReport,
    validate_semantic_analysis,
)
from src.sql_generator import SQLGenerator
from src.sql_validator import SQLValidationReport, validate_readonly_sql

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Classification taxonomy for Text-to-SQL errors."""

    SYNTAX_ERROR = "syntax_error"
    AST_SECURITY_VIOLATION = "ast_security_violation"
    DIALECT_INCOMPATIBILITY = "dialect_incompatibility"
    CTE_DEPENDENCY_ERROR = "cte_dependency_error"
    GRAIN_MISMATCH = "grain_mismatch"
    FILTER_MISPLACEMENT = "filter_misplacement"
    CARTESIAN_FANOUT = "cartesian_fanout"
    ZERO_DIVISION_RISK = "zero_division_risk"
    PLAUSIBILITY_VIOLATION = "plausibility_violation"
    DATABASE_EXECUTION_ERROR = "database_execution_error"
    UNKNOWN = "unknown"


class AttemptRecord(BaseModel):
    """Telemetry recorded for each attempt in the debugging loop."""

    attempt_index: int
    sql: str
    is_valid_syntax: bool
    is_semantically_sound: bool
    error_category: Optional[ErrorCategory] = None
    error_message: Optional[str] = None
    repair_advice: Optional[str] = None


class DebuggedExecutionResult(BaseModel):
    """Final result produced by the execution-grounded debugging loop."""

    success: bool
    final_sql: str
    total_attempts: int
    raw_result: Optional[float] = None
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    row_count: int = 0
    contract: AnalyticalContract
    semantic_report: SemanticValidationReport
    history: List[AttemptRecord] = Field(default_factory=list)
    error: Optional[str] = None
    explanation: str = ""


class ExecutionGroundedDebugger:
    """Orchestrates validation, EXPLAIN dry runs, error classification, and conditional self-repair."""

    def __init__(
        self,
        executor: Optional[MCPQueryExecutor] = None,
        generator: Optional[SQLGenerator] = None,
        max_retries: int = 3,
        settings: Optional[Settings] = None,
    ):
        self.executor = executor or MCPQueryExecutor()
        self.generator = generator or SQLGenerator(executor=self.executor)
        self.max_retries = max_retries
        self.settings = settings or get_settings()
        self.dialect_rag = DialectFunctionRAG()

    def classify_error(
        self,
        ast_report: Optional[SQLValidationReport] = None,
        semantic_report: Optional[SemanticValidationReport] = None,
        db_error: Optional[str] = None,
    ) -> Tuple[ErrorCategory, str]:
        """Classifies an encountered failure into the ErrorCategory taxonomy."""
        # 1. AST Security / Syntax
        if ast_report and not ast_report.is_valid:
            err = ast_report.error or "Unknown AST error"
            if not ast_report.is_readonly:
                return ErrorCategory.AST_SECURITY_VIOLATION, err
            if ast_report.dialect_violations:
                return ErrorCategory.DIALECT_INCOMPATIBILITY, err
            if "forward-reference" in err.lower() or "cte" in err.lower():
                return ErrorCategory.CTE_DEPENDENCY_ERROR, err
            return ErrorCategory.SYNTAX_ERROR, err

        # 2. Semantic Analysis
        if semantic_report and not semantic_report.is_semantically_sound:
            err_joined = "; ".join(semantic_report.errors + semantic_report.warnings)
            if semantic_report.has_join_multiplication_risk:
                return ErrorCategory.CARTESIAN_FANOUT, err_joined
            if semantic_report.has_join_filter_leak:
                return ErrorCategory.FILTER_MISPLACEMENT, err_joined
            if not semantic_report.has_distinct_entity_count:
                return ErrorCategory.GRAIN_MISMATCH, err_joined
            if not semantic_report.result_bounds_valid:
                return ErrorCategory.PLAUSIBILITY_VIOLATION, err_joined
            if not semantic_report.has_safe_division:
                return ErrorCategory.ZERO_DIVISION_RISK, err_joined
            return ErrorCategory.GRAIN_MISMATCH, err_joined

        # 3. Database Execution
        if db_error:
            db_lower = db_error.lower()
            if "syntax error" in db_lower:
                return ErrorCategory.SYNTAX_ERROR, db_error
            if "does not exist" in db_lower:
                return ErrorCategory.SYNTAX_ERROR, db_error
            return ErrorCategory.DATABASE_EXECUTION_ERROR, db_error

        return ErrorCategory.UNKNOWN, "Unknown validation failure"

    def generate_repair_guidance(
        self,
        category: ErrorCategory,
        error_details: str,
        current_sql: str,
        target_dialect: str,
    ) -> str:
        """Formulates actionable repair instructions to guide query regeneration."""
        if category == ErrorCategory.FILTER_MISPLACEMENT:
            return (
                "FIX FILTER MISPLACEMENT: You placed a date/status filter in the WHERE clause on a LEFT-JOINED activity table. "
                "This inadvertently converts the LEFT JOIN into an INNER JOIN, discarding churned users from the denominator! "
                "Instead, evaluate activity conditionally inside the CTE projection: "
                "COUNT(DISTINCT CASE WHEN event_date >= ... THEN user_id ELSE NULL END), keeping the WHERE clause clean."
            )
        if category == ErrorCategory.GRAIN_MISMATCH:
            return (
                "FIX GRAIN MISMATCH: The query counts row or event occurrences (COUNT(*), COUNT(event_id)) rather than distinct entities. "
                "Repeat sessions by the same entity are distorting your counts. Use COUNT(DISTINCT entity_id) in every aggregation."
            )
        if category == ErrorCategory.CARTESIAN_FANOUT:
            return (
                "FIX CARTESIAN FAN-OUT: You joined multiple independent transactional tables directly without pre-aggregating them. "
                "Pre-aggregate each table in its own separate CTE first, then join the pre-aggregated CTE summaries on the dimension key."
            )
        if category == ErrorCategory.DIALECT_INCOMPATIBILITY:
            return (
                f"FIX DIALECT INCOMPATIBILITY in {target_dialect.upper()}: {error_details}. "
                "If using QUALIFY on PostgreSQL, rewrite to compute ROW_NUMBER() in a CTE step and filter in the subsequent SELECT."
            )
        if category == ErrorCategory.ZERO_DIVISION_RISK:
            return (
                "FIX ZERO DIVISION RISK: Wrap every division denominator with NULLIF(denominator, 0) to prevent division-by-zero crashes."
            )
        if category == ErrorCategory.PLAUSIBILITY_VIOLATION:
            return (
                f"FIX PLAUSIBILITY VIOLATION: Computed percentage is outside [0%, 100%]: {error_details}. "
                "Verify your numerator is a subset of the denominator cohort, and guard the calculation with NULLIF."
            )
        if category == ErrorCategory.AST_SECURITY_VIOLATION:
            return (
                "FIX SECURITY VIOLATION: Destructive operations (DROP, DELETE, UPDATE, ALTER, CREATE) are strictly forbidden. "
                "Emit only read-only SELECT queries."
            )

        return f"FIX SQL ERROR: {error_details}"

    def run_dry_run_explain(
        self,
        sql: str,
        target: str = "postgres",
    ) -> Tuple[bool, Optional[str]]:
        """Performs a non-executing EXPLAIN / dry-run against the target database."""
        dialect = "sqlite" if target.lower() == "local" else "postgres"
        clean_sql = sql.strip().rstrip(";")
        explain_sql = f"EXPLAIN QUERY PLAN {clean_sql}" if dialect == "sqlite" else f"EXPLAIN {clean_sql}"

        try:
            if dialect == "sqlite":
                from src.db import get_local_db_connection
                conn = get_local_db_connection(self.settings)
                cur = conn.cursor()
                cur.execute(explain_sql)
                cur.fetchall()
                conn.close()
                return True, None

            from src.db import get_db_cursor
            with get_db_cursor(self.settings) as cur:
                cur.execute(explain_sql)
                cur.fetchall()
                return True, None
        except Exception as exc:
            return False, str(exc)

    def execute_with_debugging(
        self,
        question: str,
        target: Optional[str] = None,
        use_hf: Optional[bool] = None,
    ) -> DebuggedExecutionResult:
        """Executes the full cyclic reasoning and debugging pipeline."""
        contract, detected_target = self.generator.decompose_question(question)
        final_target = target or detected_target
        dialect = "sqlite" if final_target == "local" else "postgres"

        seen_sql_hashes: Set[str] = set()
        history: List[AttemptRecord] = []

        # Initial Generation
        gen_result = self.generator.generate_improved_sql(
            question=question,
            target=final_target,
            use_hf=use_hf,
        )
        current_sql = gen_result.generated_sql

        for attempt_idx in range(1, self.max_retries + 1):
            sql_hash = hashlib.sha256(current_sql.encode("utf-8")).hexdigest()
            if sql_hash in seen_sql_hashes:
                logger.warning("Repeated failed SQL query detected in debugging loop; stopping retry loop.")
                break
            seen_sql_hashes.add(sql_hash)

            # Step 1: SQLGlot AST Syntax & Security Check
            ast_report = validate_readonly_sql(current_sql, dialect=dialect)
            if not ast_report.is_valid:
                cat, err_msg = self.classify_error(ast_report=ast_report)
                advice = self.generate_repair_guidance(cat, err_msg, current_sql, dialect)
                history.append(
                    AttemptRecord(
                        attempt_index=attempt_idx,
                        sql=current_sql,
                        is_valid_syntax=False,
                        is_semantically_sound=False,
                        error_category=cat,
                        error_message=err_msg,
                        repair_advice=advice,
                    )
                )
                # Repair: Fall back to validated CTE plan compiler
                plan = self.generator.planner.plan_question(question, target_dialect=dialect)
                current_sql = CTEPlanCompiler.compile(plan, target_dialect=dialect)
                continue

            # Step 2: Semantic Analysis
            sem_report = validate_semantic_analysis(current_sql, contract=contract, dialect=dialect)
            if not sem_report.is_semantically_sound:
                cat, err_msg = self.classify_error(semantic_report=sem_report)
                advice = self.generate_repair_guidance(cat, err_msg, current_sql, dialect)
                history.append(
                    AttemptRecord(
                        attempt_index=attempt_idx,
                        sql=current_sql,
                        is_valid_syntax=True,
                        is_semantically_sound=False,
                        error_category=cat,
                        error_message=err_msg,
                        repair_advice=advice,
                    )
                )
                plan = self.generator.planner.plan_question(question, target_dialect=dialect)
                current_sql = CTEPlanCompiler.compile(plan, target_dialect=dialect)
                continue

            # Step 3: Database EXPLAIN Dry Run
            dry_ok, dry_err = self.run_dry_run_explain(current_sql, target=final_target)
            if not dry_ok:
                cat, err_msg = self.classify_error(db_error=dry_err)
                advice = self.generate_repair_guidance(cat, err_msg, current_sql, dialect)
                history.append(
                    AttemptRecord(
                        attempt_index=attempt_idx,
                        sql=current_sql,
                        is_valid_syntax=True,
                        is_semantically_sound=True,
                        error_category=cat,
                        error_message=err_msg,
                        repair_advice=advice,
                    )
                )
                plan = self.generator.planner.plan_question(question, target_dialect=dialect)
                current_sql = CTEPlanCompiler.compile(plan, target_dialect=dialect)
                continue

            # Step 4: Controlled Read-Only Execution via MCP
            exec_res = self.executor.execute_readonly_sql_sync(current_sql, target=final_target)
            if not exec_res.get("success"):
                cat, err_msg = self.classify_error(db_error=exec_res.get("error"))
                history.append(
                    AttemptRecord(
                        attempt_index=attempt_idx,
                        sql=current_sql,
                        is_valid_syntax=True,
                        is_semantically_sound=True,
                        error_category=cat,
                        error_message=err_msg,
                        repair_advice="Execution failed in database engine.",
                    )
                )
                continue

            # Step 5: Post-execution Result Evaluation
            rows = exec_res.get("rows", [])
            cols = exec_res.get("columns", [])
            raw_numeric: Optional[float] = None
            if rows:
                first_val = list(rows[0].values())[0]
                try:
                    raw_numeric = float(first_val) if first_val is not None else None
                except (ValueError, TypeError):
                    raw_numeric = None

            # Plausibility check on numeric result
            final_sem = validate_semantic_analysis(
                current_sql,
                contract=contract,
                actual_result=raw_numeric,
                dialect=dialect,
            )

            history.append(
                AttemptRecord(
                    attempt_index=attempt_idx,
                    sql=current_sql,
                    is_valid_syntax=True,
                    is_semantically_sound=final_sem.is_semantically_sound,
                    error_category=None if final_sem.is_semantically_sound else ErrorCategory.PLAUSIBILITY_VIOLATION,
                    error_message=None if final_sem.is_semantically_sound else "; ".join(final_sem.errors),
                )
            )

            if not final_sem.is_semantically_sound:
                # Plausibility violation, repair
                continue

            # Success!
            return DebuggedExecutionResult(
                success=True,
                final_sql=current_sql,
                total_attempts=attempt_idx,
                raw_result=raw_numeric,
                rows=rows,
                columns=cols,
                row_count=len(rows),
                contract=contract,
                semantic_report=final_sem,
                history=history,
                explanation=f"Successfully verified and executed on attempt {attempt_idx}.",
            )

        # Max retries exceeded
        return DebuggedExecutionResult(
            success=False,
            final_sql=current_sql,
            total_attempts=len(history),
            contract=contract,
            semantic_report=sem_report if "sem_report" in locals() else SemanticValidationReport(is_semantically_sound=False),
            history=history,
            error="Exceeded maximum retries in debugging loop without achieving verified execution.",
            explanation="Failed after retries.",
        )
