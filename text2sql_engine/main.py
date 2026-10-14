"""Pipeline Orchestrator: LangGraph StateGraph executing Enterprise Text-to-SQL pipeline with bounded error recovery."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional
import duckdb
from langgraph.graph import StateGraph, START, END

from text2sql_engine.core.config import get_settings
from text2sql_engine.core.state import SQLAgentState
from text2sql_engine.modules.problem1_planner.schemas import LogicPlan
from text2sql_engine.modules.problem1_planner.logic_planner import LogicPlanner
from text2sql_engine.modules.problem2_retriever.schema_flattener import SchemaFlattener
from text2sql_engine.modules.problem2_retriever.vector_retriever import (
    TwoStageSchemaRetriever,
)
from text2sql_engine.data.setup_sample_db import init_sample_database


class Text2SQLPipeline:
    """End-to-end LangGraph Orchestrator solving Spider 2.0 benchmark failure modes."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        retriever: Optional[TwoStageSchemaRetriever] = None,
        planner: Optional[LogicPlanner] = None,
    ):
        settings = get_settings()
        self.db_path = db_path or settings.duckdb_path

        # Ensure database exists
        if not os.path.exists(self.db_path):
            self.con = init_sample_database(self.db_path)
        else:
            self.con = duckdb.connect(self.db_path)

        # Initialize Retriever & Planner
        import chromadb

        self.retriever = retriever or TwoStageSchemaRetriever(
            collection_prefix="pipeline",
            chroma_client=chromadb.EphemeralClient(),
        )
        self.planner = planner or LogicPlanner()

        # Seed schema into retriever if not already populated
        self._bootstrap_schema_index()

        # Build LangGraph workflow
        self.graph = self._build_graph()

    def _bootstrap_schema_index(self) -> None:
        """Indexes sample database tables, nested types, and distinct categorical values."""
        tables = [t[0] for t in self.con.execute("SHOW TABLES;").fetchall()]
        tables_meta: Dict[str, Dict[str, Any]] = {}

        descriptions = {
            "customers": "Customer profiles, demographic location, active status, signup dates",
            "orders": "Order records, order status, total amount, shipping costs, order dates",
            "order_items": "Order line items, product category, unit price, quantity",
            "web_events": "GA4 web traffic events, page views, geo info, and nested event_params",
        }

        for tbl in tables:
            fields = SchemaFlattener.flatten_duckdb_table(self.con, tbl)
            pks = [f"{tbl[:-1]}_id"] if tbl.endswith("s") else ["id"]
            if tbl == "order_items":
                pks = ["item_id"]
            elif tbl == "web_events":
                pks = ["event_id"]

            tables_meta[tbl] = {
                "description": descriptions.get(tbl, f"Table {tbl}"),
                "fields": fields,
                "primary_keys": pks,
                "foreign_keys": ["customer_id"] if tbl in ("orders", "web_events") else (["order_id"] if tbl == "order_items" else []),
            }

        sample_values = {
            "customers": {
                "state_code": [
                    ("CA", "California"),
                    ("NY", "New York"),
                    ("TX", "Texas"),
                ]
            },
            "orders": {
                "order_status": [
                    ("COMPLETED", "completed"),
                    ("CANCELLED", "cancelled"),
                ]
            },
            "order_items": {
                "product_category": [
                    ("electronics", "electronics"),
                    ("apparel", "clothing"),
                    ("books", "books"),
                ]
            },
        }

        self.retriever.index_database_schema(tables_meta, sample_values, clear_existing=True)

    # ------------------ LangGraph Node Definitions ------------------

    def node_schema_retriever(self, state: SQLAgentState) -> Dict[str, Any]:
        """Node 1: Two-Stage Schema Retrieval and Distinct Value Disambiguation."""
        query = state.get("user_query", "")
        try:
            cand_tables, schema_context, matched_literals = self.retriever.retrieve_schema_context(query)
            return {
                "candidate_tables": cand_tables,
                "filtered_schema_context": schema_context,
                "matched_literals": matched_literals,
                "failed_stage": None,
                "error": None,
            }
        except Exception as e:
            return {
                "candidate_tables": [],
                "filtered_schema_context": "",
                "matched_literals": {},
                "failed_stage": "retrieval",
                "error_category": "retrieval",
                "error": str(e),
                "execution_status": "FAILED",
            }

    def node_logic_planner(self, state: SQLAgentState) -> Dict[str, Any]:
        """Node 2: CTE Logic Planning and Dialect Function Retrieval."""
        query = state.get("user_query", "")
        schema_ctx = state.get("filtered_schema_context", "")
        dialect = state.get("target_dialect", "duckdb")
        matched_literals = state.get("matched_literals", {})
        repair_notes = state.get("repair_notes")

        try:
            plan = self.planner.generate_logic_plan(
                user_query=query,
                schema_context=schema_ctx,
                dialect=dialect,
                matched_literals=matched_literals,
                repair_context=repair_notes,
            )

            return {
                "logic_plan": plan.model_dump(),
                "dialect_functions": [f.model_dump() for f in plan.dialect_functions_used],
                "failed_stage": None,
                "error": None,
            }
        except Exception as e:
            return {
                "failed_stage": "planning",
                "error_category": "planning",
                "error": str(e),
            }

    def node_sql_compiler(self, state: SQLAgentState) -> Dict[str, Any]:
        """Node 3: AST Compilation and Dialect Transpilation using SQLGlot."""
        plan_dict = state.get("logic_plan", {})
        dialect = state.get("target_dialect", "duckdb")

        try:
            plan = LogicPlan(**plan_dict)
            compiled_sql = self.planner.compile_sql(plan, target_dialect=dialect)
            return {
                "generated_sql": compiled_sql,
                "ast_valid": True,
                "execution_status": "COMPILED",
                "failed_stage": None,
                "error": None,
            }
        except Exception as e:
            return {
                "generated_sql": None,
                "ast_valid": False,
                "failed_stage": "compilation",
                "error_category": "compilation",
                "error": str(e),
            }

    def node_db_executor(self, state: SQLAgentState) -> Dict[str, Any]:
        """Node 4: Execute SQL against target database engine with truthful status reporting."""
        sql = state.get("generated_sql", "")
        dialect = state.get("target_dialect", "duckdb").lower().strip()

        # DuckDB engine: local execution supported
        if dialect == "duckdb":
            try:
                cursor = self.con.execute(sql)
                cols = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(cols, row)) for row in rows]
                return {
                    "execution_status": "EXECUTED",
                    "execution_result": results,
                    "failed_stage": None,
                    "error": None,
                }
            except Exception as e:
                return {
                    "execution_status": "ERROR",
                    "execution_result": None,
                    "failed_stage": "execution",
                    "error_category": "runtime",
                    "error": str(e),
                }
        else:
            # For non-DuckDB dialects (BigQuery, Snowflake, PostgreSQL), direct live execution is unsupported
            return {
                "execution_status": "UNSUPPORTED_EXECUTION_BACKEND",
                "execution_result": None,
                "error": f"Direct execution is unsupported for dialect '{dialect}'. No active remote database connection configured. SQL was compiled and AST-verified.",
            }

    def node_error_handler(self, state: SQLAgentState) -> Dict[str, Any]:
        """Node 5: Bounded recovery and error attribution node."""
        current_retries = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)
        failed_stage = state.get("failed_stage", "unknown")
        err_msg = state.get("error", "Unknown error")
        category = state.get("error_category", "general")

        # Non-recoverable errors (e.g. unsupported dialect, security injection attempt)
        non_recoverable = category in ("security", "unsupported_dialect") or "unsupported" in err_msg.lower()

        if current_retries >= max_retries or non_recoverable:
            return {
                "execution_status": "FAILED",
                "repair_notes": None,
                "error": err_msg,
            }

        # Recoverable error: increment retry count and feed error context into next planning attempt
        return {
            "retry_count": current_retries + 1,
            "repair_notes": f"Stage '{failed_stage}' failed with: {err_msg}",
            "execution_status": "RETRYING",
            "error": None,
        }

    # ------------------ Graph Edge Routers ------------------

    def _route_after_planner(self, state: SQLAgentState) -> str:
        if state.get("error") or not state.get("logic_plan"):
            return "error_handler"
        return "sql_compiler"

    def _route_after_compiler(self, state: SQLAgentState) -> str:
        if state.get("error") or not state.get("generated_sql"):
            return "error_handler"
        return "db_executor"

    def _route_after_executor(self, state: SQLAgentState) -> str:
        if state.get("execution_status") == "ERROR" or (state.get("error") and state.get("execution_status") != "UNSUPPORTED_EXECUTION_BACKEND"):
            return "error_handler"
        return END

    def _route_after_error(self, state: SQLAgentState) -> str:
        if state.get("execution_status") == "FAILED":
            return END
        return "logic_planner"


    def _build_graph(self) -> Any:
        """Constructs the LangGraph StateGraph pipeline with conditional routing and bounded error recovery."""
        builder = StateGraph(SQLAgentState)

        # Add Nodes
        builder.add_node("schema_retriever", self.node_schema_retriever)
        builder.add_node("logic_planner", self.node_logic_planner)
        builder.add_node("sql_compiler", self.node_sql_compiler)
        builder.add_node("db_executor", self.node_db_executor)
        builder.add_node("error_handler", self.node_error_handler)

        # Connect Flow
        builder.add_edge(START, "schema_retriever")
        builder.add_edge("schema_retriever", "logic_planner")

        # Conditional branch after planner
        builder.add_conditional_edges(
            "logic_planner",
            self._route_after_planner,
            {
                "sql_compiler": "sql_compiler",
                "error_handler": "error_handler",
            },
        )

        # Conditional branch after compiler
        builder.add_conditional_edges(
            "sql_compiler",
            self._route_after_compiler,
            {
                "db_executor": "db_executor",
                "error_handler": "error_handler",
            },
        )

        # Conditional branch after executor
        builder.add_conditional_edges(
            "db_executor",
            self._route_after_executor,
            {
                END: END,
                "error_handler": "error_handler",
            },
        )

        # Conditional branch after error handler (bounded retry)
        builder.add_conditional_edges(
            "error_handler",
            self._route_after_error,
            {
                "logic_planner": "logic_planner",
                END: END,
            },
        )

        return builder.compile()

    def run(self, user_query: str, target_dialect: str = "duckdb", max_retries: int = 2) -> SQLAgentState:
        """Executes the complete pipeline for a user query."""
        initial_state: SQLAgentState = {
            "user_query": user_query,
            "target_dialect": target_dialect,
            "retry_count": 0,
            "max_retries": max_retries,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state


def main():
    """Demonstrates execution of the Enterprise Text-to-SQL pipeline on sample queries."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("================================================================================")
    print("[INFO] Initializing Enterprise Text-to-SQL Engine (Spider 2.0 Methodology)")
    print("================================================================================")

    pipeline = Text2SQLPipeline()

    sample_queries = [
        (
            "Find the top 3 customers ranked by total spend with safe division handling",
            "duckdb",
        ),
        (
            "Show purchases in California and rank orders by total amount",
            "duckdb",
        ),
        (
            "What campaigns were associated with web events in California?",
            "duckdb",
        ),
        (
            "Calculate days elapsed between orders and rank top transactions",
            "bigquery",
        ),
    ]

    all_passed = True
    for idx, (query, dialect) in enumerate(sample_queries, 1):
        print(f"\n--- [Query {idx}] '{query}' (Target Dialect: {dialect.upper()}) ---")
        result = pipeline.run(query, target_dialect=dialect)

        status = result.get("execution_status", "UNKNOWN")
        print(f"[Execution Status]: {status}")
        print("[Candidate Tables Selected]:", result.get("candidate_tables"))
        print("[Discovered Database Literals]:", result.get("matched_literals"))

        if result.get("generated_sql"):
            print("\n[Compiled & Transpiled SQL]:")
            print(result.get("generated_sql"))

        if status == "EXECUTED":
            execution_result = result.get("execution_result", [])
            print(f"\n[Execution Result ({len(execution_result)} rows)]:")
            for row in execution_result[:5]:
                print(" ", row)
        elif status == "UNSUPPORTED_EXECUTION_BACKEND":
            print(f"\n[Backend Notice]: {result.get('error')}")
        elif status == "FAILED":
            all_passed = False
            print(f"\n[Execution Error]: {result.get('error')}")

    print("\n================================================================================")
    if all_passed:
        print("[SUMMARY] Pipeline run complete: all queries were compiled and handled accurately.")
    else:
        print("[SUMMARY] Pipeline run complete with errors flagged in output.")
    print("================================================================================")


if __name__ == "__main__":
    main()
