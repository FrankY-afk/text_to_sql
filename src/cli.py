"""Interactive Terminal CLI Console for Text-to-SQL Problem 1.

Provides an interactive developer REPL with:
- Natural Language Question evaluation (Problem 1 Erroneous Analysis diagnosis).
- Direct SQL Execution Sandbox with real-time AST security validation.
- Live PostgreSQL schema exploration.
- Rich-formatted colorized panels, SQL syntax highlighting, and result tables.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src.config import get_settings
from src.cte_planner import CTEPlanCompiler, CTEPlanner
from src.debugger import ExecutionGroundedDebugger
from src.dialect_rag import DialectFunctionRAG
from src.evaluator import (
    Problem1Evaluator,
    Problem2RetrievalEvaluator,
    run_full_problem1_benchmark,
)
from src.query_executor import MCPQueryExecutor
from src.schema_inspector import SchemaInspector
from src.semantic_validator import validate_semantic_analysis
from src.sql_generator import SQLGenerator
from src.sql_validator import validate_readonly_sql

console = Console()


class InteractiveConsole:
    """Interactive CLI REPL for Text-to-SQL Problem 1."""

    def __init__(self):
        self.settings = get_settings()
        self.executor = MCPQueryExecutor()
        self.generator = SQLGenerator(executor=self.executor)
        self.planner = CTEPlanner()
        self.dialect_rag = DialectFunctionRAG()
        self.debugger = ExecutionGroundedDebugger(executor=self.executor, generator=self.generator)
        self.evaluator = Problem1Evaluator(executor=self.executor)
        self.active_target = "postgres"  # Default database target

    def print_banner(self) -> None:
        """Displays the startup banner."""
        banner_text = (
            "[bold cyan]Text-to-SQL Problem 1: Interactive Terminal Console[/bold cyan]\n"
            "[dim]Model Context Protocol (MCP) + PostgreSQL + Semantic AST Validation[/dim]\n\n"
            f"[bold yellow]Active Database Target:[/bold yellow] [bold green]{self.active_target.upper()}[/bold green] "
            f"([dim]{'lending_sample on 35.200.180.162' if self.active_target == 'postgres' else 'local_problem1.db'}[/dim])"
        )
        console.print(Panel(banner_text, border_style="cyan", padding=(1, 2)))

    def print_menu(self) -> None:
        """Prints available preset options and command hints."""
        menu = Table(title="Interactive Commands & Presets", border_style="dim", show_header=True)
        menu.add_column("Key", style="bold yellow", width=6)
        menu.add_column("Description", style="white")
        menu.add_column("Target Database", style="dim cyan", width=18)

        menu.add_row("1", "User Retention Benchmark (Jan/Feb Cohort Grain Mismatch)", "Local Benchmark")
        menu.add_row("2", "Staff Attendance Rate (June 2026 Join Leak Trap)", "Live PostgreSQL")
        menu.add_row("3", "Group Loan Payout Conversion Rate (May 2026)", "Live PostgreSQL")
        menu.add_row("4", "Inspect Live PostgreSQL Database Schema & Tables", "Live PostgreSQL")
        menu.add_row("5", "SQL Security Sandbox (Test AST validation on custom SQL)", "Active Target")
        menu.add_row("6", f"Toggle Target Database (Current: {self.active_target.upper()})", "Switch Target")
        menu.add_row("7", "Run Full Spider 2.0 Regression Suite (10 Failure Modes)", "Local Benchmark")
        menu.add_row("8", "Run Problem 2 Enterprise Schema Linking Evaluation (20 Tables / 240+ Cols)", "In-Memory + BGE")
        menu.add_row("help", "Redisplay this menu", "-")
        menu.add_row("q", "Exit / Quit", "-")

        console.print(menu)
        console.print("[dim]Tip: You can also type ANY custom SQL query (e.g. 'SELECT * FROM outlets LIMIT 3;') directly![/dim]\n")

    def run_benchmark_suite(self) -> None:
        """Executes the full Spider 2.0 10-category regression suite and renders results."""
        console.print("\n[bold cyan]=== Running Full Problem 1 Regression Suite (10 Failure Modes) ===[/bold cyan]\n")
        records = run_full_problem1_benchmark(executor=self.executor, target="local")

        table = Table(title="Problem 1 Spider 2.0 Failure Modes Evaluation", border_style="bright_blue")
        table.add_column("Case ID", style="bold yellow", width=8)
        table.add_column("Spider 2.0 Failure Mode", style="white")
        table.add_column("Exec", style="green", width=7)
        table.add_column("Semantic", style="cyan", width=10)
        table.add_column("Computed", style="dim", width=10)
        table.add_column("Ground Truth", style="dim", width=12)
        table.add_column("Status", style="bold", width=8)

        passed_count = 0
        audited_count = 0
        for r in records:
            if r.analytically_correct:
                passed_count += 1
                status_text = "[bold green]PASS[/bold green]"
            elif r.expected_business_result is None:
                audited_count += 1
                status_text = "[cyan]AUDITED[/cyan]"
            else:
                status_text = "[bold red]FAIL[/bold red]"

            sem_sound = r.validation_result.get("is_semantically_sound", False)
            exec_str = "OK" if r.execution_accurate else ("(Schema Only)" if r.expected_business_result is None else "FAIL")

            table.add_row(
                r.case_id,
                r.failure_mode,
                exec_str,
                "SOUND" if sem_sound else "WARN",
                f"{r.actual_business_result:.1f}%" if r.actual_business_result is not None else "N/A",
                f"{r.expected_business_result:.1f}%" if r.expected_business_result is not None else "N/A",
                status_text,
            )

        console.print(table)
        console.print(
            f"\n[bold green]Suite Complete:[/bold green] [bold]{passed_count} Regression Benchmarks PASSED with Exact Ground Truth[/bold], "
            f"[bold]{audited_count} Analytical Patterns Audited[/bold] ({len(records)}/{len(records)} total).\n"
            f"[dim]As specified in benchmark rules, execution accuracy and analytical plan correctness are tracked separately.[/dim]\n"
        )

    def run_problem2_schema_evaluation(self) -> None:
        """Executes Stage 7 Enterprise Schema Linking Evaluation Suite (Problem 2)."""
        console.print("\n[bold cyan]=== Running Problem 2: Two-Stage Schema Linking Evaluation ===[/bold cyan]\n")
        report = Problem2RetrievalEvaluator.run_benchmark(embedding_backend="auto")

        summary = Table(title="Stage 7 Retrieval Metrics (Aggregated)", border_style="bright_blue")
        summary.add_column("Metric", style="bold yellow")
        summary.add_column("Value", style="bold green")
        summary.add_row("Indexed Enterprise Catalog", f"{report.total_tables_indexed} tables / {report.total_columns_indexed} fields")
        summary.add_row("Table Recall@K", f"{report.mean_table_recall_at_k:.2%}")
        summary.add_row("Column Recall@K", f"{report.mean_column_recall_at_k:.2%}")
        summary.add_row("Table Precision@K / Col Precision@K", f"{report.mean_table_precision_at_k:.2%} / {report.mean_column_precision_at_k:.2%}")
        summary.add_row("Mean Reciprocal Rank (Table MRR / Col MRR)", f"{report.mean_reciprocal_rank:.3f} / {report.mean_column_mrr:.3f}")
        summary.add_row("Mean Schema Token Reduction", f"{report.mean_token_reduction_pct:.1f}%")
        summary.add_row("Mean Retrieval Latency", f"{report.mean_retrieval_latency_ms:.2f} ms")
        summary.add_row("DB-Evidenced Literal Accuracy", f"{report.literal_resolution_accuracy:.2%}")
        console.print(summary)

        cases_tbl = Table(title="Per-Case Retrieval Breakdown", border_style="cyan")
        cases_tbl.add_column("ID", style="bold yellow", width=7)
        cases_tbl.add_column("Scenario", style="white")
        cases_tbl.add_column("Tbl R@K", style="green", width=8)
        cases_tbl.add_column("Col R@K", style="green", width=8)
        cases_tbl.add_column("MRR", style="cyan", width=6)
        cases_tbl.add_column("Token Save", style="magenta", width=11)
        cases_tbl.add_column("Latency", style="dim", width=10)
        for cr in report.case_results:
            cases_tbl.add_row(
                cr.case_id,
                cr.category,
                f"{cr.table_recall_at_k:.0%}",
                f"{cr.column_recall_at_k:.0%}",
                f"{cr.table_reciprocal_rank:.2f}",
                f"{cr.token_reduction_pct:.1f}%",
                f"{cr.retrieval_latency_ms:.1f} ms",
            )
        console.print("\n")
        console.print(cases_tbl)
        console.print(f"[dim]{report.scope_disclaimer}[/dim]\n")

    def run_problem1_analysis(self, question: str, target: Optional[str] = None) -> None:
        """Runs the complete Problem 1 Baseline vs Improved analysis and displays Rich comparison."""
        tgt = target or self.active_target
        dialect = "sqlite" if tgt.lower() == "local" else "postgres"
        console.print(f"\n[bold cyan]=== Evaluating Question on [{tgt.upper()} / {dialect.upper()}] ===[/bold cyan]")
        console.print(f"[bold italic white]\"{question}\"[/bold italic white]\n")

        contract, _ = self.generator.decompose_question(question)

        # 1. Show Analytical Contract
        contract_table = Table(title="Decomposed Analytical Contract", border_style="cyan", show_header=False)
        contract_table.add_column("Field", style="bold yellow")
        contract_table.add_column("Specification", style="white")
        contract_table.add_row("Metric Name", contract.metric_name)
        contract_table.add_row("Target Grain Entity", f"[bold green]{contract.target_entity}[/bold green]")
        contract_table.add_row("Requires DISTINCT?", str(contract.requires_distinct_entity_count))
        contract_table.add_row("Is Percentage?", str(contract.is_percentage))
        contract_table.add_row("Baseline Cohort", contract.baseline_cohort_filter or "Full Population")
        contract_table.add_row("Returning Filter", contract.returning_cohort_filter or "Active Subset")
        console.print(contract_table)

        # 1b. Show Two-Stage Schema Linking Summary (Problem 2)
        schema_ctx = self.generator.get_schema_context(question, target=tgt)
        if schema_ctx is not None and schema_ctx.candidate_tables:
            schema_tbl = Table(
                title=(
                    f"Two-Stage Schema Linking (Latency: {schema_ctx.retrieval_latency_ms:.1f}ms | "
                    f"Token Reduction: {schema_ctx.token_reduction_pct:.1f}%)"
                ),
                border_style="bright_blue",
                show_header=True,
            )
            schema_tbl.add_column("Candidate Table", style="bold yellow")
            schema_tbl.add_column("Score", style="green", width=7)
            schema_tbl.add_column("Pruned Columns", style="cyan")
            for sec in schema_ctx.candidate_tables:
                col_list = ", ".join(c.path for c in sec.selected_columns)
                schema_tbl.add_row(
                    sec.full_name,
                    f"{sec.similarity_score:.2f}",
                    f"({len(sec.selected_columns)}/{sec.total_table_columns}) {col_list}",
                )
            console.print("\n")
            console.print(schema_tbl)

        # 2. Show Structured CTE Logic Plan (Mechanism A)
        plan = self.planner.plan_question(question, target_dialect=dialect, schema_context=schema_ctx)
        cte_table = Table(title=f"Structured CTE Logic Plan ({len(plan.cte_steps)} Decomposed Steps)", border_style="cyan", show_header=True)
        cte_table.add_column("#", style="dim", width=3)
        cte_table.add_column("CTE Name", style="bold yellow")
        cte_table.add_column("Purpose", style="white")
        cte_table.add_column("Inputs", style="cyan")
        cte_table.add_column("Dependencies", style="dim magenta")
        cte_table.add_column("Output Grain", style="bold green")
        for idx, step in enumerate(plan.cte_steps, 1):
            cte_table.add_row(
                str(idx),
                step.step_name,
                step.purpose,
                ", ".join(step.input_tables),
                ", ".join(step.dependencies) if step.dependencies else "(None)",
                step.expected_output_grain,
            )
        console.print("\n")
        console.print(cte_table)

        # 3. Show Dialect Function RAG Recommendations (Mechanism B)
        rag_funcs = self.dialect_rag.retrieve_functions(question, target_dialect=dialect, limit=3)
        if rag_funcs:
            rag_table = Table(title=f"Retrieved Dialect Function Documentation [{dialect.upper()}]", border_style="magenta", show_header=True)
            rag_table.add_column("Function", style="bold yellow")
            rag_table.add_column("Category", style="cyan")
            rag_table.add_column("Syntax", style="white")
            rag_table.add_column("Dialect Rule / Caveat", style="italic")
            for rf in rag_funcs:
                rag_table.add_row(rf.name, rf.category, rf.syntax_pattern, rf.caveats_and_pitfalls or rf.description)
            console.print("\n")
            console.print(rag_table)

        # 4. Schema Grounding: Resolve Categorical Literals via MCP
        grounded_literals = self.executor.resolve_categorical_literals_sync(question, target=tgt)
        if grounded_literals:
            lit_table = Table(title="Discovered Database Literals (Schema Grounding via MCP)", border_style="magenta", show_header=True)
            lit_table.add_column("Natural Language Term", style="bold yellow")
            lit_table.add_column("Target Table & Column", style="white")
            lit_table.add_column("Database Cell Value", style="bold green")
            lit_table.add_column("Match Type", style="dim")
            for m in grounded_literals:
                lit_table.add_row(m["term"], f"{m['table_name']}.{m['column_name']}", f"'{m['database_literal']}'", m["match_type"])
            console.print("\n")
            console.print(lit_table)

        # Helper to convert numeric outputs safely
        def safe_float(val: Any) -> Optional[float]:
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        # 5. Execute Naive Baseline Pipeline
        gen_base = self.generator.generate_baseline_sql(question, target=tgt)
        res_base = self.executor.execute_readonly_sql_sync(gen_base.generated_sql, target=tgt)
        raw_base_first = list(res_base["rows"][0].values())[0] if res_base.get("rows") else None
        val_base = validate_semantic_analysis(
            gen_base.generated_sql,
            contract=contract,
            actual_result=safe_float(raw_base_first),
            dialect=dialect,
        )
        base_val = raw_base_first if raw_base_first is not None else "N/A"

        # 6. Execute Execution-Grounded Debugging Loop (Improved Pipeline)
        t0 = time.time()
        debug_res = self.debugger.execute_with_debugging(question, target=tgt)
        elapsed = (time.time() - t0) * 1000

        val_imp = debug_res.semantic_report
        imp_val = debug_res.raw_result if debug_res.raw_result is not None else "N/A"

        # 7. Self-Repair Diagnostics Trace (if retried)
        if len(debug_res.history) > 1:
            trace_table = Table(title="Execution Debugger Cyclic Self-Repair Trace", border_style="yellow")
            trace_table.add_column("Attempt", style="bold yellow", width=8)
            trace_table.add_column("Error Category", style="red")
            trace_table.add_column("Diagnosed Failure", style="white")
            trace_table.add_column("Repair Guidance Injected", style="green")
            for att in debug_res.history:
                trace_table.add_row(
                    f"#{att.attempt_index}",
                    att.error_category.value if att.error_category else "None",
                    att.error_message or "None",
                    att.repair_advice or "None",
                )
            console.print("\n")
            console.print(trace_table)

        # 8. Display SQL Side-by-Side
        console.print("\n[bold red]Pipeline A: Naive Baseline SQL (Collapse Mode):[/bold red]")
        console.print(Syntax(gen_base.generated_sql, "sql", theme="monokai", line_numbers=True))

        console.print("\n[bold green]Pipeline B: Debugged / Improved CTE SQL (Semantically Guarded):[/bold green]")
        console.print(Syntax(debug_res.final_sql, "sql", theme="monokai", line_numbers=True))

        # 9. Comparison Table or Data Table
        if contract.is_percentage:
            comp_table = Table(title="Benchmark Comparison Results", border_style="bright_blue")
            comp_table.add_column("Dimension", style="bold")
            comp_table.add_column("Naive Baseline", style="red")
            comp_table.add_column("Improved MCP (Debugged)", style="green")

            base_status = "SUCCESS (0 Errors)" if res_base.get("success") else "FAILED"
            imp_status = "SUCCESS (0 Errors)" if debug_res.success else "FAILED"

            comp_table.add_row("SQL Execution", base_status, imp_status)
            comp_table.add_row("Calculated Result", f"{base_val}%", f"[bold]{imp_val}%[/bold]")
            comp_table.add_row(
                "Semantic Status",
                "[bold red]FAILED[/bold red]" if not val_base.is_semantically_sound else "[bold green]SOUND[/bold green]",
                "[bold green]PASSED (Sound)[/bold green]" if val_imp.is_semantically_sound else "[bold red]FAILED[/bold red]",
            )
            comp_table.add_row("Counting Grain", val_base.detected_grain, val_imp.detected_grain)
            comp_table.add_row("Safe Math Guard?", str(val_base.has_safe_division), str(val_imp.has_safe_division))
            comp_table.add_row("Join Filter Leak?", str(val_base.has_join_filter_leak), str(val_imp.has_join_filter_leak))
            console.print("\n")
            console.print(comp_table)
        else:
            # Render returned tabular records
            rows = debug_res.rows
            cols = debug_res.columns
            if rows:
                data_table = Table(title=f"Query Output ({len(rows)} rows)", border_style="green", header_style="bold cyan")
                for c in cols:
                    data_table.add_column(str(c))
                for r in rows:
                    data_table.add_row(*[str(r.get(c, "")) for c in cols])
                console.print("\n")
                console.print(data_table)

        # 10. Diagnosed Pitfalls Alert
        if val_base.errors or val_base.warnings:
            pitfalls = "\n".join([f"[red][ERROR][/red] {e}" for e in val_base.errors] + [f"[yellow][WARNING][/yellow] {w}" for w in val_base.warnings])
            console.print(Panel(f"[bold red]Detected Pitfalls in Baseline Query:[/bold red]\n{pitfalls}", border_style="red"))

        console.print(f"[dim]Executed via MCP client & debugger in {elapsed:.1f}ms[/dim]\n")

    def run_direct_sql(self, sql: str) -> None:
        """Executes custom SQL through MCP with real-time AST safety and displays rich table."""
        console.print(f"\n[bold cyan]Validating & Executing via MCP on [{self.active_target.upper()}]:[/bold cyan]")
        console.print(Syntax(sql, "sql", theme="monokai"))

        # Step 1: Pre-execution AST security check
        report = validate_readonly_sql(sql)
        if not report.is_valid:
            error_msg = f"[bold red]AST SECURITY VIOLATION REJECTED:[/bold red]\n{report.error}"
            console.print(Panel(error_msg, border_style="bold red"))
            return

        # Step 2: Safe execution via MCP
        t0 = time.time()
        res = self.executor.execute_readonly_sql_sync(sql, target=self.active_target)
        elapsed = (time.time() - t0) * 1000

        if not res.get("success"):
            console.print(Panel(f"[bold red]Database Execution Error:[/bold red]\n{res.get('error')}", border_style="red"))
            return

        rows = res.get("rows", [])
        columns = res.get("columns", [])

        if not rows:
            console.print("[yellow]Query executed successfully, but returned 0 rows.[/yellow]\n")
            return

        # Step 3: Render formatted data table
        table = Table(
            title=f"Query Output ({len(rows)} rows returned in {elapsed:.1f}ms)",
            border_style="green",
            show_header=True,
            header_style="bold cyan",
        )
        for col in columns:
            table.add_column(str(col))

        for r in rows:
            table.add_row(*[str(r.get(c, "")) for c in columns])

        console.print(table)
        console.print(f"[dim]Verified read-only query executed in {elapsed:.1f}ms via MCP.[/dim]\n")

    def read_multiline_sql(self, first_line: str = "") -> str:
        """Reads multiline SQL input until a semicolon (;) is entered."""
        lines = [first_line] if first_line else []
        if lines and lines[0].strip().endswith(";"):
            return lines[0].strip()

        while True:
            try:
                line = input("...  " if lines else "SQL> ")
                if not line.strip() and lines:
                    break
                lines.append(line)
                if line.strip().endswith(";"):
                    break
            except (KeyboardInterrupt, EOFError):
                break
        return "\n".join(lines).strip()

    def inspect_tables(self) -> None:
        """Displays interactive overview of live database tables."""
        inspector = SchemaInspector(target=self.active_target)
        tables = inspector.get_table_names()

        table = Table(title=f"Accessible Tables in [{self.active_target.upper()}]", border_style="blue")
        table.add_column("#", style="dim", width=4)
        table.add_column("Table Name", style="bold white")
        table.add_column("Columns", style="cyan")

        for idx, tbl in enumerate(tables, 1):
            cols = [c["name"] for c in inspector.get_table_columns(tbl)]
            table.add_row(str(idx), tbl, ", ".join(cols[:6]) + ("..." if len(cols) > 6 else ""))

        console.print(table)
        console.print("[dim]Use command 5 to query any table with SELECT.[/dim]\n")

    def start_loop(self) -> None:
        """Main REPL loop."""
        self.print_banner()
        self.print_menu()

        while True:
            try:
                user_input = Prompt.ask(
                    f"\n[bold yellow]text2sql[/bold yellow] ([cyan]{self.active_target}[/cyan])"
                ).strip()

                if not user_input:
                    continue

                cmd = user_input.lower()

                if cmd in ("q", "quit", "exit"):
                    console.print("[bold green]Goodbye![/bold green]")
                    break

                elif cmd in ("help", "menu", "?"):
                    self.print_menu()

                elif cmd == "1":
                    # Preset 1: User Retention
                    q = "What percentage of users who signed up in January returned in February?"
                    self.run_problem1_analysis(q, target="local")

                elif cmd == "2":
                    # Preset 2: Staff Attendance
                    q = "What percentage of our staff members were marked present in June 2026?"
                    self.run_problem1_analysis(q, target="postgres")

                elif cmd == "3":
                    # Preset 3: Loan Payout
                    q = "What percentage of group loans created in May 2026 reached the Payout stage?"
                    self.run_problem1_analysis(q, target="postgres")

                elif cmd == "4":
                    self.inspect_tables()

                elif cmd == "5":
                    console.print("[bold green]Paste or type SQL query (end with ';' and press Enter):[/bold green]")
                    sql = self.read_multiline_sql()
                    if sql.strip():
                        self.run_direct_sql(sql)

                elif cmd == "6":
                    # Toggle database target
                    self.active_target = "local" if self.active_target == "postgres" else "postgres"
                    console.print(f"[bold yellow]Switched target database to:[/bold yellow] [bold green]{self.active_target.upper()}[/bold green]\n")

                elif cmd == "7":
                    self.run_benchmark_suite()

                elif cmd == "8":
                    self.run_problem2_schema_evaluation()

                elif user_input.upper().startswith("SELECT ") or user_input.upper().startswith("WITH "):
                    # User typed or pasted a multi-line SQL query directly
                    sql = self.read_multiline_sql(first_line=user_input)
                    if sql.strip():
                        self.run_direct_sql(sql)

                else:
                    # User typed a natural language question
                    self.run_problem1_analysis(user_input)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[bold green]Session ended. Goodbye![/bold green]")
                break
            except Exception as exc:
                console.print(f"[bold red]Unexpected error:[/bold red] {exc}")


def launch_cli() -> None:
    """Entry point for the interactive CLI."""
    app = InteractiveConsole()
    app.start_loop()


if __name__ == "__main__":
    launch_cli()
