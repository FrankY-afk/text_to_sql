"""Unit tests for the Execution-Grounded Debugging Loop (Problem 1, Phase G)."""

from __future__ import annotations

import pytest
from src.debugger import (
    ErrorCategory,
    ExecutionGroundedDebugger,
)
from src.query_executor import MCPQueryExecutor
from src.semantic_validator import AnalyticalContract, SemanticValidationReport
from src.sql_generator import SQLGenerator
from src.sql_validator import SQLValidationReport


@pytest.fixture
def debugger():
    return ExecutionGroundedDebugger()


def test_error_classification_taxonomy(debugger):
    """Verifies that failures are correctly classified into the ErrorCategory taxonomy."""
    # 1. AST Security
    ast_unsafe = SQLValidationReport(is_valid=False, is_readonly=False, error="Statement of type 'Drop' is not permitted")
    cat1, _ = debugger.classify_error(ast_report=ast_unsafe)
    assert cat1 == ErrorCategory.AST_SECURITY_VIOLATION

    # 2. Dialect Incompatibility
    ast_dialect = SQLValidationReport(
        is_valid=False,
        is_readonly=True,
        error="QUALIFY is not supported in POSTGRES",
        dialect_violations=["QUALIFY is not supported in POSTGRES"],
    )
    cat2, _ = debugger.classify_error(ast_report=ast_dialect)
    assert cat2 == ErrorCategory.DIALECT_INCOMPATIBILITY

    # 3. CTE Dependency Error
    ast_cte = SQLValidationReport(
        is_valid=False,
        is_readonly=True,
        error="CTE dependency forward-reference: CTE 'step1' references 'step2'",
    )
    cat3, _ = debugger.classify_error(ast_report=ast_cte)
    assert cat3 == ErrorCategory.CTE_DEPENDENCY_ERROR

    # 4. Filter Misplacement
    sem_leak = SemanticValidationReport(
        is_semantically_sound=False,
        has_join_filter_leak=True,
        errors=["Filter Misplacement Trap: LEFT JOIN on table has filtering predicates in WHERE"],
    )
    cat4, _ = debugger.classify_error(semantic_report=sem_leak)
    assert cat4 == ErrorCategory.FILTER_MISPLACEMENT

    # 5. Cartesian Fan-Out
    sem_fanout = SemanticValidationReport(
        is_semantically_sound=False,
        has_join_multiplication_risk=True,
        errors=["Cartesian Fan-Out / Join Multiplication Trap"],
    )
    cat5, _ = debugger.classify_error(semantic_report=sem_fanout)
    assert cat5 == ErrorCategory.CARTESIAN_FANOUT

    # 6. Grain Mismatch
    sem_grain = SemanticValidationReport(
        is_semantically_sound=False,
        has_distinct_entity_count=False,
        errors=["Grain Mismatch: Analytical contract requires distinct counting"],
    )
    cat6, _ = debugger.classify_error(semantic_report=sem_grain)
    assert cat6 == ErrorCategory.GRAIN_MISMATCH


def test_repair_guidance_generation(debugger):
    """Verifies that repair guidance produces concrete instructions for fixing logic errors."""
    advice_leak = debugger.generate_repair_guidance(
        category=ErrorCategory.FILTER_MISPLACEMENT,
        error_details="WHERE filter on LEFT JOIN",
        current_sql="SELECT ...",
        target_dialect="postgres",
    )
    assert "FIX FILTER MISPLACEMENT" in advice_leak
    assert "COUNT(DISTINCT CASE WHEN" in advice_leak

    advice_fanout = debugger.generate_repair_guidance(
        category=ErrorCategory.CARTESIAN_FANOUT,
        error_details="Joined group_loans and individual_loans directly",
        current_sql="SELECT ...",
        target_dialect="postgres",
    )
    assert "FIX CARTESIAN FAN-OUT" in advice_fanout
    assert "separate CTE first" in advice_fanout


def test_execute_with_debugging_cohort_retention(debugger):
    """Verifies that the full debugging loop executes cleanly and returns ground truth on local benchmark."""
    q = "What percentage of users who signed up in January returned in February?"
    res = debugger.execute_with_debugging(q, target="local")

    assert res.success is True
    assert res.total_attempts >= 1
    assert res.raw_result == 30.0
    assert res.semantic_report.is_semantically_sound is True
    assert len(res.rows) == 1
    assert "retention_rate" in res.columns or "30.0" in str(res.rows[0])
    assert len(res.history) >= 1
    assert res.history[0].is_valid_syntax is True


def test_debugging_loop_self_repairs_flawed_initial_query(debugger):
    """Simulates an initial flawed query and verifies the debugger detects the error and repairs it."""
    # Temporarily monkey-patch generator to emit a flawed baseline query on the first call
    original_generate = debugger.generator.generate_improved_sql
    calls = 0

    def mock_flawed_initial_generator(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Emits naive query with filter misplacement
            return debugger.generator.generate_baseline_sql("Calculate retention", target="local")
        return original_generate(*args, **kwargs)

    debugger.generator.generate_improved_sql = mock_flawed_initial_generator

    q = "What percentage of users who signed up in January returned in February?"
    res = debugger.execute_with_debugging(q, target="local")

    assert res.success is True
    assert res.total_attempts == 2  # First attempt caught error, second attempt repaired!
    assert res.raw_result == 30.0
    assert res.history[0].is_semantically_sound is False
    assert res.history[0].error_category == ErrorCategory.FILTER_MISPLACEMENT
    assert res.history[1].is_semantically_sound is True

