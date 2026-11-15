"""Expanded evaluation and regression test suite for Problem 1 (Phase H).

Verifies the 10 Spider 2.0 analytical failure modes:
1. Distinct user counts versus event counts.
2. Incorrect aggregation order.
3. Join multiplication (Cartesian fan-out).
4. Incorrect denominator selection.
5. LEFT JOIN filter mistakes.
6. Running totals.
7. Moving averages.
8. Percentile calculations.
9. Date and time-window boundaries.
10. Dialect-specific function differences.

Separately measures execution accuracy and analytical correctness.
"""

from __future__ import annotations

import pytest
from src.evaluator import EvaluationRecord, Problem1Evaluator


@pytest.fixture
def evaluator():
    return Problem1Evaluator()


def test_cohort_retention_exact_benchmark_regression(evaluator):
    """Core Problem 1 Benchmark Test Case:

    Question: 'What percentage of users who signed up in January returned in February?'
    Expected result:
    - January signup cohort: 100 distinct users
    - February returning users: 30 distinct users
    - Expected retention: 30.0%
    Baseline:
    - Incorrect result: 100.0% (Failure: counts events, converts LEFT JOIN to INNER JOIN)
    Improved approach:
    - Expected result: 30.0% (Exact Match: distinct users, NULLIF guard, pre-aggregated CTEs)
    """
    # 1. Verify Ground Truth dynamically from DB
    gt = evaluator.compute_ground_truth(target="local")
    assert gt["january_signups"] == 100
    assert gt["returning_february_users"] == 30
    assert gt["true_retention_rate"] == 30.0

    # 2. Evaluate Baseline Naive approach
    base_res = evaluator.evaluate_baseline("What percentage of users who signed up in January returned in February?", target="local")
    assert base_res.execution_success is True  # Executes without DB error
    assert base_res.is_accurate is False      # But produces wrong business result
    assert base_res.raw_result == 100.0       # 100.0% failure
    assert base_res.absolute_error == 70.0
    assert base_res.semantic_report.is_semantically_sound is False

    # 3. Evaluate Improved Semantically Guarded approach
    imp_res = evaluator.evaluate_improved("What percentage of users who signed up in January returned in February?", target="local")
    assert imp_res.execution_success is True  # Executes successfully
    assert imp_res.is_accurate is True       # Exactly matches ground truth
    assert imp_res.raw_result == 30.0        # 30.0% accurate result
    assert imp_res.absolute_error == 0.0
    assert imp_res.semantic_report.is_semantically_sound is True


def test_run_full_problem1_benchmark_records(evaluator):
    """Executes the complete 10-test analytical failure mode benchmark.

    Verifies that for every test case, the evaluation records:
    - Input question.
    - Expected analytical plan.
    - Expected SQL behaviour.
    - Actual generated SQL.
    - Validation result.
    - Execution result.
    - Expected business result.
    - Actual business result.
    - Whether the query was analytically correct.
    """
    records = evaluator.run_full_problem1_benchmark(target="local")

    assert len(records) == 10

    expected_case_ids = [f"TC-{i:02d}" for i in range(1, 11)]
    actual_case_ids = [r.case_id for r in records]
    assert actual_case_ids == expected_case_ids

    for rec in records:
        assert isinstance(rec, EvaluationRecord)
        assert rec.question
        assert rec.expected_analytical_plan
        assert rec.expected_sql_behavior
        assert rec.actual_generated_sql
        assert isinstance(rec.validation_result, dict)
        assert isinstance(rec.execution_accurate, bool)
        assert isinstance(rec.analytically_correct, bool)

    # TC-01: Cohort retention must be analytically correct and exact 30.0%
    tc01 = next(r for r in records if r.case_id == "TC-01")
    assert tc01.execution_accurate is True
    assert tc01.analytically_correct is True
    assert tc01.actual_business_result == 30.0
    assert tc01.expected_business_result == 30.0

    # TC-03: Join multiplication query must not use unaggregated raw joins
    tc03 = next(r for r in records if r.case_id == "TC-03")
    assert tc03.validation_result["has_join_multiplication_risk"] is False

    # TC-05: Staff attendance query must not have join filter leaks
    tc05 = next(r for r in records if r.case_id == "TC-05")
    assert tc05.validation_result["has_join_filter_leak"] is False

    # TC-09: January signup count must match 100 users exactly
    tc09 = next(r for r in records if r.case_id == "TC-09")
    assert tc09.execution_accurate is True
    assert tc09.actual_business_result == 100.0


def test_execution_accuracy_vs_analytical_correctness_separation(evaluator):
    """Verifies that execution accuracy and analytical correctness are measured separately."""
    # The naive baseline executes without error (Execution Accurate = True),
    # but computes an incorrect metric (Analytically Correct = False).
    base_res = evaluator.evaluate_baseline("Retention calculation", target="local")
    assert base_res.execution_success is True  # Execution succeeded
    assert base_res.is_accurate is False      # Analytical failure

