"""Interactive Project Explainer & Technical Q&A Console.

Walks through all 9 stages of the Enterprise Text-to-SQL architecture
and answers the top technical questions asked by mentors, interviewers, and architects.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown

# Ensure root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

console = Console()

STAGES = [
    {
        "number": 1,
        "title": "Turn Ingestion & Multi-Turn Conversational Memory",
        "files": ["src/conversational_memory.py", "src/cli.py"],
        "concept": "Stateful context tracking and follow-up query resolution.",
        "details": (
            "Analyzes user questions to determine if they are standalone or follow-ups "
            "(e.g., 'Now filter for Tier 1' or 'What about Gujarat?'). It automatically inherits "
            "candidate tables, metric calculations, and swaps dimensions without starting from scratch."
        ),
        "analogy": "Like a conversation with a human analyst who remembers what was discussed 30 seconds ago."
    },
    {
        "number": 2,
        "title": "Two-Stage Enterprise Schema Linking & Pruning",
        "files": ["src/table_retriever.py", "src/column_retriever.py", "src/schema_flattener.py"],
        "concept": "Context reduction from 20+ tables down to <400 tokens (~70% savings).",
        "details": (
            "Stage 1 (Coarse): Vector search in ChromaDB ranks top 2-4 candidate tables.\n"
            "Stage 2 (Fine): Ranks relevant columns while strictly preserving Primary & Foreign Keys.\n"
            "Stage 3 (Flattener): Unpacks nested STRUCT, ARRAY, and JSON/JSONB into queryable dot-paths."
        ),
        "analogy": "Instead of handing someone an entire library encyclopedia, you hand them only the 2 relevant pages."
    },
    {
        "number": 3,
        "title": "DB-Evidenced Categorical Literal Grounding",
        "files": ["src/literal_resolver.py"],
        "concept": "Matching fuzzy user terms to actual database storage cells.",
        "details": (
            "Natural language abbreviations ('MH', 'RJCT', 'T1') or typos ('Maharahtra') are checked "
            "against cached distinct database values. Requires actual storage evidence (require_db_evidence=True) "
            "before emitting exact SQL predicates like `state_name = 'Maharashtra'`."
        ),
        "analogy": "Spell-check and nickname translator backed by real database passport records."
    },
    {
        "number": 4,
        "title": "Relational Graph Join Path Discovery & Rulebook RAG",
        "files": ["src/schema_graph.py", "src/doc_rag.py", "src/database_explorer.py"],
        "concept": "Finding join bridges, retrieving business formulas, and live data probing.",
        "details": (
            "1. Schema Graph: In-memory bidirectional BFS finds the shortest join path across foreign keys.\n"
            "2. Doc RAG: Ingests Markdown rulebooks for industry formulas (NPA 90+ DPD, Attendance, GLP).\n"
            "3. Database Explorer: Fast diagnostic probes (<15ms) check row counts, date bounds, and join fan-out."
        ),
        "analogy": "GPS navigation between tables + consulting the company handbook + doctor's pre-op checkup."
    },
    {
        "number": 5,
        "title": "Adaptive CTE Logic Planning & DAG Validation",
        "files": ["src/cte_planner.py"],
        "concept": "Breaking complex problems into step-by-step modular recipes.",
        "details": (
            "Complexity routing divides queries into two paths:\n"
            "- Simple queries -> Adaptive Direct: Single clean SELECT statement (0 CTE overhead).\n"
            "- Complex queries -> Adaptive CTE DAG: Multi-step Common Table Expressions (cohorts, ratios).\n"
            "Strictly validates DAG integrity: no cycles and no forward-references allowed."
        ),
        "analogy": "Frying an egg directly in the pan vs. using separate prep bowls to bake a multi-tier wedding cake."
    },
    {
        "number": 6,
        "title": "Dialect-Specific Code Generation & Compiler Fallback",
        "files": ["src/sql_generator.py", "src/dialect_rag.py", "src/hf_client.py"],
        "concept": "High-accuracy code generation with 100% offline fallback certainty.",
        "details": (
            "1. Dialect RAG: Supplies exact dialect syntax (Postgres vs DuckDB date diffs, NULLIF guards).\n"
            "2. Hugging Face LLM: Calls Qwen 2.5 Coder 32B with structured context.\n"
            "3. CTE Plan Compiler: Pure Python compiler assembling the plan into verified SQL if offline."
        ),
        "analogy": "A master builder with a local dialect phrasebook, backed by an offline mechanical blueprint machine."
    },
    {
        "number": 7,
        "title": "Dual-Layer AST Security & Semantic Guardrails",
        "files": ["src/sql_validator.py", "src/semantic_validator.py"],
        "concept": "Airport security + Logic police catching syntax, security, and math traps.",
        "details": (
            "Layer 1 (AST Gate): SQLGlot blocks DDL/DML, multi-statement injection, and dangerous functions.\n"
            "Layer 2 (Semantic Gate): Enforces distinct entity counting (COUNT(DISTINCT)), flags WHERE filters "
            "on LEFT JOINs, mandates NULLIF division guards, and verifies [0%, 100%] bounds."
        ),
        "analogy": "Metal detector for weapons + immigration officer verifying travel intent."
    },
    {
        "number": 8,
        "title": "Execution-Grounded Debugger & Self-Repair Loop",
        "files": ["src/debugger.py"],
        "concept": "Cyclic self-repair with 10-category error classification and anti-looping.",
        "details": (
            "Runs a database EXPLAIN dry run. If syntax, AST, or semantic errors occur:\n"
            "1. Classifies into 1 of 10 error taxonomies (e.g., FILTER_MISPLACEMENT, CARTESIAN_FANOUT).\n"
            "2. Computes SHA-256 state hash to prevent repeating the same mistake.\n"
            "3. Formulates targeted repair guidance and retries generation (up to 3 attempts)."
        ),
        "analogy": "An F1 pit crew diagnosing the telemetry sensor and fixing the car in seconds."
    },
    {
        "number": 9,
        "title": "Safe MCP Execution & Inverted Output Presentation",
        "files": ["src/mcp_server.py", "src/query_executor.py", "src/cli.py"],
        "concept": "Read-only transaction boundaries and clean, output-first display.",
        "details": (
            "1. Execution: MCP server executes under enforced `SET TRANSACTION READ ONLY` with a 60s timeout.\n"
            "2. Presentation: Displays the computed business result/table FIRST, and generated SQL SECOND.\n"
            "3. State Update: Records the result into Conversational Memory for subsequent follow-up turns."
        ),
        "analogy": "Delivering the hot, freshly cooked meal to the table first, with the recipe card on the side."
    }
]

FAQS = [
    {
        "q": "Why do we use the Model Context Protocol (MCP) instead of connecting directly with SQLAlchemy?",
        "a": (
            "MCP decouples the AI reasoning agent from raw database infrastructure. Instead of granting "
            "an LLM unrestricted database connection strings or raw cursor execution, MCP establishes a "
            "strictly bounded sandbox with 14 discrete tools (e.g. list_tables, get_table_schema, "
            "validate_sql, execute_readonly_sql). It enforces read-only transaction boundaries, statement timeouts, "
            "and AST validation at the protocol layer, making it enterprise-secure and interoperable across any agent framework."
        )
    },
    {
        "q": "Why use Common Table Expressions (CTEs) instead of nested subqueries?",
        "a": (
            "In complex multi-stage analytics (like cohort retention or Cartesian prevention), nested subqueries "
            "become deeply indented 'onion queries' where filters in outer layers accidentally corrupt inner "
            "aggregations. CTEs decompose the problem into a topologically ordered Directed Acyclic Graph (DAG) "
            "of named steps (e.g. Step 1: signup_cohort, Step 2: returning_cohort). This enforces explicit entity grain, "
            "allows the semantic validator to verify each intermediate table, and dramatically improves readability."
        )
    },
    {
        "q": "How does the system prevent Cartesian Join Explosions (Fan-Out)?",
        "a": (
            "When joining multiple one-to-many tables (e.g. joining outlets to group_loans and individual_loans), "
            "unaggregated joins cause multiplicative row explosion, inflating financial sums into billions. "
            "The system prevents this in 3 places:\n"
            "1. Database Explorer runs a `probe_join_multiplicity` test before generation.\n"
            "2. The Schema Graph tags reverse edges as `one-to-many` to warn against fan-out.\n"
            "3. The CTE Planner enforces independent pre-aggregation CTEs before joining on parent keys."
        )
    },
    {
        "q": "How does the system generate SQL when there is NO API key or internet connection?",
        "a": (
            "The system has a built-in deterministic `CTEPlanCompiler` written in pure Python. "
            "The CTE Planner produces a typed Pydantic `LogicPlan` with modular `CTEPlanStep` blocks. "
            "If Hugging Face is unconfigured or offline, `CTEPlanCompiler.compile()` directly transforms "
            "the plan into dialect-perfect SQL using SQLGlot. This is why our 104-test pytest suite "
            "runs 100% offline with zero network latency."
        )
    },
    {
        "q": "Why is Two-Stage Schema Linking needed if modern LLMs have 128k+ context windows?",
        "a": (
            "Large context windows do NOT solve the 'needle in a haystack' problem. Dumping 25+ tables and "
            "300+ columns into a prompt confuses code models, causes hallucinated joins, and increases latency "
            "and cost. Our two-stage retriever (coarse table ranking + fine column pruning preserving PK/FK) "
            "cuts context tokens by ~70.4% (<400 tokens), yielding higher generation accuracy and faster responses."
        )
    },
    {
        "q": "What is the purpose of SHA-256 state hashing in the Debugger?",
        "a": (
            "LLMs frequently fall into repetitive repair loops—repeating the exact same faulty SQL query "
            "even after receiving an error message. The Debugger computes a SHA-256 hash of each attempted "
            "SQL string. If it detects a duplicate hash, it immediately triggers loop-breaking guidance, "
            "forcing the generator to adopt an alternative analytical strategy."
        )
    },
    {
        "q": "How does Categorical Literal Grounding prevent zero-row results?",
        "a": (
            "Users frequently use abbreviations ('MH', 'T1'), typos ('Maharahtra'), or business shorthand ('casual leave'). "
            "If an AI writes `WHERE state = 'MH'` when the database stores `'Maharashtra'`, the query returns 0 rows. "
            "LiteralResolver queries actual distinct values in the database with `require_db_evidence=True`, "
            "grounding the user's natural language into exact database cell values before writing SQL."
        )
    },
    {
        "q": "Why does the Schema Graph need both Forward and Reverse edges?",
        "a": (
            "Foreign key constraints in databases are strictly one-way (e.g. `staff` references `outlets`). "
            "However, SQL joins can travel in either direction. More importantly, to connect two child tables "
            "(e.g. `staff` and `group_loans` through parent `outlets`), the search algorithm must walk forward "
            "into `outlets` and backwards along `group_loans`'s foreign key. Reverse edges make the graph bidirectional."
        )
    },
    {
        "q": "How does the system handle business rules like NPA (90+ DPD) or Attendance that aren't in the schema?",
        "a": (
            "Via Problem 3 Documentation RAG (`src/doc_rag.py`). It indexes external company rulebooks and dbt metric "
            "definitions. When a user asks about NPA or attendance, it retrieves the authoritative mathematical formula "
            "(e.g. DPD >= 90, PR=1.0 day, HD=0.5 day) and injects it directly into the LLM prompt and CTE plan."
        )
    },
    {
        "q": "What security guarantees prevent data corruption or injection?",
        "a": (
            "Triple-layer protection:\n"
            "1. SQLGlot AST Gate rejects all DDL (DROP, ALTER), DML (INSERT, UPDATE, DELETE), and multi-statements.\n"
            "2. Database session sets `SET TRANSACTION READ ONLY` and 60-second timeouts.\n"
            "3. MCP execution tool clamps max returned rows to prevent memory exhaustion."
        )
    }
]


def display_banner():
    banner = (
        "[bold cyan]Enterprise Text-to-SQL Architecture Explainer[/bold cyan]\n"
        "[dim]Spider 2.0 • MCP • Adaptive CTE Engine • Multi-Turn Memory • Self-Repair[/dim]\n\n"
        "[bold yellow]1.[/bold yellow] Stage-by-Stage Walkthrough (1 to 9)\n"
        "[bold yellow]2.[/bold yellow] Top 10 Technical Interview & Mentor Q&A\n"
        "[bold yellow]3.[/bold yellow] Frameworks & Architecture Matrix\n"
        "[bold yellow]4.[/bold yellow] Exit"
    )
    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))


def run_stages_walkthrough():
    console.print("\n[bold green]=== 9-Stage Query Resolution Pipeline ===[/bold green]\n")
    for s in STAGES:
        content = (
            f"[bold yellow]Core Concept:[/bold yellow] {s['concept']}\n\n"
            f"[bold cyan]What Happens Under the Hood:[/bold cyan]\n{s['details']}\n\n"
            f"[bold magenta]Files Involved:[/bold magenta] [white]{', '.join(s['files'])}[/white]\n\n"
            f"[dim italic]Analogy: {s['analogy']}[/dim italic]"
        )
        title = f"Stage {s['number']}: {s['title']}"
        console.print(Panel(content, title=f"[bold green]{title}[/bold green]", border_style="green"))
        
        if s['number'] < len(STAGES):
            inp = Prompt.ask("\n[bold white]Press [Enter] for next stage (or 'q' to return to menu)[/bold white]", default="")
            if inp.lower().strip() == 'q':
                break
            console.print()


def run_faq_explorer():
    console.print("\n[bold green]=== Top 10 Technical Questions & Deep-Dive Answers ===[/bold green]\n")
    table = Table(title="Select a Question to Explore", border_style="cyan")
    table.add_column("#", style="bold yellow", width=4)
    table.add_column("Question", style="bold white")

    for i, item in enumerate(FAQS, start=1):
        table.add_row(str(i), item["q"])

    console.print(table)
    console.print("[dim]Enter a number (1-10) to read the answer, 'all' to read all, or 'q' to return.[/dim]\n")

    choice = Prompt.ask("[bold cyan]Question selection[/bold cyan]", default="1")
    if choice.lower() == 'q':
        return
    elif choice.lower() == 'all':
        for i, item in enumerate(FAQS, start=1):
            console.print(Panel(
                f"[bold white]{item['a']}[/bold white]",
                title=f"[bold green]Q{i}: {item['q']}[/bold green]",
                border_style="green",
                padding=(1, 2)
            ))
            time.sleep(0.3)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(FAQS):
                item = FAQS[idx]
                console.print(Panel(
                    f"[bold white]{item['a']}[/bold white]",
                    title=f"[bold green]Q{idx+1}: {item['q']}[/bold green]",
                    border_style="green",
                    padding=(1, 2)
                ))
            else:
                console.print("[red]Invalid index.[/red]")
        except ValueError:
            console.print("[red]Invalid input.[/red]")


def run_framework_matrix():
    console.print("\n[bold green]=== Frameworks & Responsibilities Matrix ===[/bold green]\n")
    table = Table(title="Framework Task Allocation", border_style="cyan")
    table.add_column("Framework / Library", style="bold cyan")
    table.add_column("Version", style="bold yellow")
    table.add_column("Primary Responsibility in This Project", style="white")

    matrix = [
        ("MCP (Model Context Protocol)", ">=1.0.0", "Decoupled 14-tool read-only server boundary & client executor"),
        ("LangChain & LCEL", ">=0.3.0", "Agentic orchestration, tool adapters, and structured prompt pipelines"),
        ("SQLGlot", ">=26.0.0", "AST parsing, DDL/DML security blocking, and dialect transpilation"),
        ("ChromaDB + SentenceTransformers", ">=1.5.0", "Vector embeddings for two-stage schema linking & documentation RAG"),
        ("Hugging Face Hub", ">=0.20.0", "Inference client for open-weight Qwen 2.5 Coder 32B model"),
        ("PostgreSQL (psycopg2)", ">=2.9.0", "Enterprise data warehouse target with read-only transaction guards"),
        ("DuckDB & SQLite", ">=1.0.0", "Local benchmark testing, nested STRUCT/ARRAY unnesting tests"),
        ("Pydantic", ">=2.10.0", "Strict data contract schemas (CTEPlanStep, LogicPlan, SchemaContext)"),
        ("Rich", ">=13.0.0", "Terminal REPL interface, output-first display, syntax highlighting"),
        ("Pytest", ">=8.0.0", "Continuous automated regression testing (104 tests, 100% pass)")
    ]

    for fw, ver, task in matrix:
        table.add_row(fw, ver, task)

    console.print(table)


def main():
    while True:
        display_banner()
        choice = Prompt.ask("[bold cyan]Select an option[/bold cyan]", choices=["1", "2", "3", "4"], default="1")
        if choice == "1":
            run_stages_walkthrough()
        elif choice == "2":
            run_faq_explorer()
        elif choice == "3":
            run_framework_matrix()
        elif choice == "4":
            console.print("[bold green]Thank you! Have a great day.[/bold green]")
            break
        console.print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    main()
