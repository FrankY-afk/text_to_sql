"""Interactive Terminal CLI Console for Text-to-SQL (Spider 2.0 + MCP).

A streamlined developer REPL:
- Free-text query input (natural language or custom SQL).
- Adaptive CTE / Direct query planning with MCP schema grounding.
- Execution-grounded AST validation and self-repair.
- Outputs query results FIRST, followed by the generated SQL.
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
from src.debugger import ExecutionGroundedDebugger
from src.evaluator import run_full_problem1_benchmark
from src.literal_resolver import LiteralResolver
from src.query_executor import MCPQueryExecutor
from src.schema_inspector import SchemaInspector
from src.sql_generator import SQLGenerator
from src.sql_validator import validate_readonly_sql

console = Console()


class InteractiveConsole:
    """Streamlined CLI REPL for Text-to-SQL."""

    def __init__(self):
        self.settings = get_settings()
        self.executor = MCPQueryExecutor()
        self.generator = SQLGenerator(executor=self.executor)
        self.debugger = ExecutionGroundedDebugger(executor=self.executor, generator=self.generator)
        self.active_target = "postgres"  # Default database target

    def print_banner(self) -> None:
        """Displays a clean startup banner."""
        target_info = (
            f"{self.settings.postgres_db} on {self.settings.postgres_host}"
            if self.active_target == "postgres"
            else "local_problem1.db (SQLite)"
        )
        banner_content = (
            "[bold cyan]Text-to-SQL Assistant[/bold cyan] [dim](Spider 2.0 • MCP • Adaptive CTE • AST Self-Repair)[/dim]\n"
            f"[bold yellow]Active Database Target:[/bold yellow] [bold green]{self.active_target.upper()}[/bold green] "
            f"([dim]{target_info}[/dim])\n\n"
            "[dim]Commands: [white]'target'[/white] to switch DB | [white]'bench'[/white] to run benchmarks | "
            "[white]'tables'[/white] to list tables | [white]'exit'[/white] to quit[/dim]"
        )
        console.print(Panel(banner_content, border_style="cyan", padding=(1, 2)))

    def run_query(self, user_query: str) -> None:
        """Processes a natural language query through the adaptive debugger.

        CRITICAL REQUIREMENT:
        Outputs the result of the query FIRST, followed by the generated SQL.
        """
        tgt = self.active_target
        t0 = time.time()

        with console.status("[bold cyan]Analyzing intent, grounding schema, and generating SQL...[/bold cyan]", spinner="dots"):
            debug_res = self.debugger.execute_with_debugging(user_query, target=tgt)
            elapsed = (time.time() - t0) * 1000

        # Determine strategy type
        is_cte = "WITH " in debug_res.final_sql.upper().strip()
        strategy_label = "Adaptive CTE" if is_cte else "Adaptive Direct"

        # -------------------------------------------------------------
        # 1. DISPLAY OUTPUT OF THE QUERY FIRST
        # -------------------------------------------------------------
        console.print("\n")
        if debug_res.success:
            # If a scalar percentage / numeric aggregate was computed
            if debug_res.raw_result is not None:
                contract_name = (
                    debug_res.semantic_report.detected_grain
                    if debug_res.semantic_report
                    else "Aggregate Metric"
                )
                metric_panel = (
                    f"[bold white]Computed Result:[/bold white] [bold green]{debug_res.raw_result:.2f}%[/bold green]\n"
                    f"[dim]Metric Grain: {contract_name} | Plausibility Bound: [0%, 100%][/dim]"
                )
                console.print(Panel(metric_panel, title="[bold green]Query Output[/bold green]", border_style="green"))

            # If tabular rows returned
            rows = debug_res.rows
            columns = debug_res.columns
            if rows:
                row_label = f"{len(rows)} row" if len(rows) == 1 else f"{len(rows)} rows"
                table_title = f"Query Output ({row_label} in {elapsed:.1f}ms)"
                data_table = Table(
                    title=table_title,
                    border_style="green",
                    header_style="bold cyan",
                    show_header=True,
                )
                for col in columns:
                    data_table.add_column(str(col))
                for row in rows:
                    data_table.add_row(*[str(row.get(c, "")) for c in columns])
                console.print(data_table)
            elif debug_res.raw_result is None:
                console.print(Panel(
                    "[bold yellow]Query executed successfully, but returned 0 matching rows.[/bold yellow]",
                    title="[bold yellow]Query Output[/bold yellow]",
                    border_style="yellow",
                ))
        else:
            # Execution failure
            error_cat = debug_res.error_category.value if debug_res.error_category else "Execution Error"
            err_msg = (
                f"[bold red]Status:[/bold red] Failed after {debug_res.total_attempts} attempt(s)\n"
                f"[bold red]Category:[/bold red] {error_cat}\n"
                f"[bold yellow]Diagnostic Notes:[/bold yellow] {debug_res.explanation}"
            )
            console.print(Panel(err_msg, title="[bold red]Query Execution Failed[/bold red]", border_style="red"))

        # -------------------------------------------------------------
        # 2. DISPLAY GENERATED SQL SECOND
        # -------------------------------------------------------------
        sql_title = f"[bold cyan]Generated SQL[/bold cyan] [dim]({strategy_label})[/dim]"
        console.print("\n")
        console.print(Panel(
            Syntax(debug_res.final_sql, "sql", theme="monokai", line_numbers=True),
            title=sql_title,
            border_style="cyan",
        ))

        # -------------------------------------------------------------
        # 3. CONCISE METADATA FOOTER
        # -------------------------------------------------------------
        status_color = "green" if debug_res.success else "red"
        status_text = "Verified Read-Only" if debug_res.success else "Execution Error"
        attempts_info = f"Attempts: {debug_res.total_attempts}" if debug_res.total_attempts > 1 else "1 attempt"
        console.print(
            f"[dim]Target: {tgt.upper()} | Plan: {strategy_label} | Latency: {elapsed:.1f}ms | "
            f"{attempts_info} | [{status_color}]{status_text}[/{status_color}][/dim]\n"
        )

    def run_direct_sql(self, sql: str) -> None:
        """Executes custom SQL through MCP with real-time AST safety.

        CRITICAL REQUIREMENT:
        Outputs the result of the query FIRST, followed by the validated SQL.
        """
        # Step 1: Pre-execution AST security check
        report = validate_readonly_sql(sql)
        if not report.is_valid:
            error_msg = f"[bold red]AST SECURITY VIOLATION REJECTED:[/bold red]\n{report.error}"
            console.print(Panel(error_msg, border_style="bold red"))
            return

        # Step 2: Safe execution via MCP
        tgt = self.active_target
        t0 = time.time()
        res = self.executor.execute_readonly_sql_sync(sql, target=tgt)
        elapsed = (time.time() - t0) * 1000

        # -------------------------------------------------------------
        # 1. DISPLAY OUTPUT OF THE QUERY FIRST
        # -------------------------------------------------------------
        console.print("\n")
        if not res.get("success"):
            console.print(Panel(
                f"[bold red]Database Execution Error:[/bold red]\n{res.get('error')}",
                title="[bold red]Execution Error[/bold red]",
                border_style="red",
            ))
            return

        rows = res.get("rows", [])
        columns = res.get("columns", [])

        if rows:
            row_label = f"{len(rows)} row" if len(rows) == 1 else f"{len(rows)} rows"
            table = Table(
                title=f"Query Output ({row_label} in {elapsed:.1f}ms)",
                border_style="green",
                header_style="bold cyan",
                show_header=True,
            )
            for col in columns:
                table.add_column(str(col))
            for r in rows:
                table.add_row(*[str(r.get(c, "")) for c in columns])
            console.print(table)
        else:
            console.print(Panel(
                "[bold yellow]Query executed successfully, but returned 0 rows.[/bold yellow]",
                title="[bold yellow]Query Output[/bold yellow]",
                border_style="yellow",
            ))

        # -------------------------------------------------------------
        # 2. DISPLAY VALIDATED SQL SECOND
        # -------------------------------------------------------------
        console.print("\n")
        console.print(Panel(
            Syntax(sql, "sql", theme="monokai", line_numbers=True),
            title="[bold cyan]Validated SQL[/bold cyan] [dim](AST Verified Read-Only)[/dim]",
            border_style="cyan",
        ))

        # -------------------------------------------------------------
        # 3. METADATA FOOTER
        # -------------------------------------------------------------
        console.print(
            f"[dim]Target: {tgt.upper()} | Execution Time: {elapsed:.1f}ms | Status: [green]Verified Read-Only[/green][/dim]\n"
        )

    def read_multiline_sql(self, first_line: str = "") -> str:
        """Reads multiline SQL input until a semicolon (;) is entered."""
        lines = [first_line] if first_line else []
        if lines and lines[0].strip().endswith(";"):
            return lines[0].strip()

        while True:
            try:
                line = input("...  " if lines else "SQL> ")
                if not line.strip() and lines:
                    continue
                lines.append(line)
                if line.strip().endswith(";"):
                    break
            except (KeyboardInterrupt, EOFError):
                return ""
        return "\n".join(lines).strip()

    def inspect_tables(self) -> None:
        """Displays accessible tables for the active target database."""
        try:
            inspector = SchemaInspector(target=self.active_target)
            tables = inspector.get_table_names()
            tbl_table = Table(
                title=f"Accessible Tables in [{self.active_target.upper()}]",
                border_style="cyan",
                header_style="bold yellow",
            )
            tbl_table.add_column("#", style="dim", width=4)
            tbl_table.add_column("Table Name", style="bold green")
            for idx, tbl in enumerate(tables, 1):
                tbl_table.add_row(str(idx), tbl)
            console.print("\n")
            console.print(tbl_table)
            console.print(f"[dim]Discovered {len(tables)} tables via MCP schema inspection.[/dim]\n")
        except Exception as exc:
            console.print(f"[bold red]Failed to inspect tables:[/bold red] {exc}")

    def run_benchmark_suite(self) -> None:
        """Runs the 10 Spider 2.0 failure mode regression suite."""
        console.print("\n[bold cyan]=== Running Spider 2.0 Analytical Regression Suite (10 Failure Modes) ===[/bold cyan]\n")
        records = run_full_problem1_benchmark(executor=self.executor, target="local")

        table = Table(title="Spider 2.0 Failure Modes Evaluation", border_style="bright_blue")
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
        )

    def start_loop(self) -> None:
        """Main REPL loop."""
        self.print_banner()

        while True:
            try:
                user_input = Prompt.ask(
                    f"\n[bold yellow]Query[/bold yellow] ([cyan]{self.active_target}[/cyan])"
                ).strip()

                if not user_input:
                    continue

                cmd = user_input.lower()

                if cmd in ("q", "quit", "exit"):
                    console.print("[bold green]Goodbye![/bold green]")
                    break

                elif cmd in ("help", "?"):
                    console.print(
                        "\n[bold cyan]Command Cheatsheet:[/bold cyan]\n"
                        "  • [yellow]target[/yellow] / [yellow]switch[/yellow]  - Toggle between PostgreSQL and local SQLite\n"
                        "  • [yellow]bench[/yellow]               - Run 10-case Spider 2.0 regression benchmarks\n"
                        "  • [yellow]tables[/yellow] / [yellow]schema[/yellow]  - List accessible tables in active database\n"
                        "  • [yellow]clear[/yellow] / [yellow]cls[/yellow]        - Clear terminal screen\n"
                        "  • [yellow]exit[/yellow] / [yellow]quit[/yellow]       - Exit console\n"
                        "  • [white]<text question>[/white]      - Any natural language query (e.g. 'Show outlets in Maharashtra')\n"
                        "  • [white]<sql statement>[/white]      - Any SELECT/WITH query (validated and executed safely)\n"
                    )

                elif cmd in ("target", "switch"):
                    self.active_target = "local" if self.active_target == "postgres" else "postgres"
                    target_desc = (
                        f"{self.settings.postgres_db} on {self.settings.postgres_host}"
                        if self.active_target == "postgres"
                        else "local_problem1.db (SQLite)"
                    )
                    console.print(
                        f"[bold yellow]Switched target database to:[/bold yellow] [bold green]{self.active_target.upper()}[/bold green] "
                        f"([dim]{target_desc}[/dim])\n"
                    )

                elif cmd in ("bench", "benchmark"):
                    self.run_benchmark_suite()

                elif cmd in ("tables", "schema"):
                    self.inspect_tables()

                elif cmd in ("clear", "cls"):
                    os.system("cls" if os.name == "nt" else "clear")
                    self.print_banner()

                elif user_input.upper().startswith("SELECT ") or user_input.upper().startswith("WITH "):
                    sql = self.read_multiline_sql(first_line=user_input)
                    if sql.strip():
                        self.run_direct_sql(sql)

                else:
                    self.run_query(user_input)

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
