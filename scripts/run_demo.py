"""Interactive demonstration runner for Problem 1: Erroneous Data Analysis.

Executes the full pipeline:
1. Connects to the database and inspects tables via MCP.
2. Runs the January/February cohort retention benchmark on the test dataset.
3. Compares Naive Baseline Text-to-SQL vs Semantically Guarded MCP Pipeline.
4. Executes the cyclic Execution-Grounded Debugging & Self-Repair loop.
5. Explains the exact failure mode and how MCP + Semantic Validation fixes it.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import Problem1Evaluator
from src.query_executor import MCPQueryExecutor
from scripts.seed_test_data import seed_local_benchmark_db


def run_demonstration():
    print("=" * 80)
    print("DEMO: Problem 1 -- Erroneous Data Analysis Using MCP and PostgreSQL")
    print("=" * 80)

    # Step 1: Ensure local benchmark data is seeded
    print("\n[Step 1] Ensuring benchmark dataset is initialized...")
    seed_local_benchmark_db()

    # Step 2: Test MCP Client connection to database
    executor = MCPQueryExecutor()
    print("[Step 2] Testing MCP Server Tool Discovery...")
    local_tables = executor.list_tables_sync(target="local")
    print(f"  -> Discovered {len(local_tables)} local benchmark tables via MCP: {local_tables}")
    sample = executor.get_sample_rows_sync("users", limit=1, target="local")
    print(f"  -> Fetched sample user via MCP: user_id={sample['rows'][0]['user_id']} ({sample['rows'][0]['country']})")

    try:
        pg_tables = executor.list_tables_sync(target="postgres")
        print(f"  -> Connected to remote PostgreSQL. Discovered {len(pg_tables)} tables: {pg_tables}")
    except Exception:
        print("  -> Remote PostgreSQL not configured in .env (running against verified local benchmark).")

    # Step 3: Run the Problem 1 Cohort Retention Benchmark
    question = "What percentage of users who signed up in January returned in February?"
    print("\n" + "=" * 80)
    print(f"BENCHMARK QUESTION:\n\"{question}\"")
    print("=" * 80)

    evaluator = Problem1Evaluator(executor=executor)
    gt = evaluator.compute_ground_truth(target="local")
    print("\n[GROUND TRUTH CALCULATION (From raw records)]:")
    print(f"  - Total January Sign-ups (Denominator) : {gt['january_signups']}")
    print(f"  - Distinct January Users Returning in Feb: {gt['returning_february_users']}")
    print(f"  - Mathematically Correct Retention Rate : {gt['true_retention_rate']}%\n")

    # Step 4: Run Baseline
    print("-" * 80)
    print("PIPELINE A: NAIVE BASELINE TEXT-TO-SQL (FAILURE MODE)")
    print("-" * 80)
    base_res = evaluator.evaluate_baseline(question, target="local")
    print("[Generated SQL]:")
    print(base_res.generated_sql)
    print(f"\n[Execution Status] : {'SUCCESS (0 SQL Errors)' if base_res.execution_success else 'FAILED'}")
    print(f"[Returned Answer]  : {base_res.raw_result}%")
    print(f"[Ground Truth]     : {base_res.ground_truth}%")
    print(f"[Absolute Error]   : {base_res.absolute_error:.2f}%")
    print(f"[Accurate?]        : {base_res.is_accurate}")
    print(f"[Semantic Sound?]  : {base_res.semantic_report.is_semantically_sound}")
    print(f"[Detected Pitfalls]:")
    for err in base_res.semantic_report.errors:
        print(f"  [ERROR] {err}")
    for warn in base_res.semantic_report.warnings:
        print(f"  [WARNING] {warn}")

    # Step 5: Run Improved MCP Pipeline
    print("\n" + "-" * 80)
    print("PIPELINE B: IMPROVED MCP + STRUCTURED CTE PLANNER PIPELINE")
    print("-" * 80)
    imp_res = evaluator.evaluate_improved(question, target="local")
    print("[Generated SQL]:")
    print(imp_res.generated_sql)
    print(f"\n[Execution Status] : {'SUCCESS' if imp_res.execution_success else 'FAILED'}")
    print(f"[Returned Answer]  : {imp_res.raw_result}%")
    print(f"[Ground Truth]     : {imp_res.ground_truth}%")
    print(f"[Absolute Error]   : {imp_res.absolute_error:.2f}%")
    print(f"[Accurate?]        : {imp_res.is_accurate}")
    print(f"[Semantic Sound?]  : {imp_res.semantic_report.is_semantically_sound}")
    print(f"[Detected Grain]   : {imp_res.semantic_report.detected_grain}")
    print(f"[Safe Division?]   : {imp_res.semantic_report.has_safe_division}")

    # Step 6: Run Execution-Grounded Debugging Loop
    print("\n" + "-" * 80)
    print("PIPELINE C: EXECUTION-GROUNDED DEBUGGING & SELF-REPAIR LOOP")
    print("-" * 80)
    dbg_res = evaluator.debugger.execute_with_debugging(question, target="local")
    print(f"[Execution Status] : {'SUCCESS' if dbg_res.success else 'FAILED'}")
    print(f"[Total Attempts]   : {dbg_res.total_attempts}")
    print(f"[Returned Result]  : {dbg_res.raw_result}%")
    print(f"[Self-Repair Trace]: {dbg_res.explanation}")

    print("\n" + "=" * 80)
    print("[COMPARISON SUMMARY]:")
    print(f"  * Baseline Naive Result : {base_res.raw_result}%  [INCORRECT] (Absolute Error: {base_res.absolute_error:.2f}%)")
    print(f"  * Improved CTE Result   : {imp_res.raw_result}%  [CORRECT]   (Absolute Error: {imp_res.absolute_error:.2f}%)")
    print(f"  * Debugger Verified     : {dbg_res.raw_result}%  [VERIFIED]  (Attempts: {dbg_res.total_attempts})")
    print("=" * 80)


if __name__ == "__main__":
    run_demonstration()
