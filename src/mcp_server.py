"""Official Model Context Protocol (MCP) Server for safe database interaction.

Uses the official Python MCP SDK (MCPServer).
Exposes 5 core tools to LLM applications:
1. list_tables: Lists permitted base tables.
2. get_table_schema: Retrieves formatted table schema cards with types and keys.
3. get_sample_rows: Retrieves a small, bounded sample of records (1-5 rows).
4. validate_sql: Evaluates query AST for syntax errors, multiple statements, and write operations.
5. execute_readonly_sql: Executes approved read-only queries with timeouts and row limits.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from src.config import Settings, get_settings
from src.db import execute_readonly_query, get_local_db_connection
from src.literal_resolver import LiteralResolver
from src.schema_graph import SchemaGraph
from src.schema_inspector import SchemaInspector
from src.schema_pruner import SchemaPruner
from src.sql_validator import validate_readonly_sql
from src.doc_rag import BUILTIN_METRICS, DocumentationRetriever
from src.database_explorer import DatabaseExplorer

logger = logging.getLogger(__name__)

# Initialize official MCP Server
server = MCPServer("postgres_text2sql_server")


@server.tool(
    name="list_tables",
    description="Lists all permitted base tables in the target database ('postgres' or 'local').",
)
def list_tables(target: str = "postgres") -> List[str]:
    """Retrieves sorted list of table names in the active schema."""
    inspector = SchemaInspector(target=target)
    return inspector.get_table_names()


@server.tool(
    name="get_table_schema",
    description="Retrieves column definitions, data types, nullability, and primary keys for a table.",
)
def get_table_schema(table_name: str, target: str = "postgres") -> str:
    """Returns an LLM-friendly schema card in Markdown for the specified table."""
    inspector = SchemaInspector(target=target)
    return inspector.get_schema_card(table_name)


@server.tool(
    name="get_sample_rows",
    description="Retrieves a small, bounded sample of rows (1 to 5 rows) from a permitted table.",
)
def get_sample_rows(table_name: str, limit: int = 3, target: str = "postgres") -> Dict[str, Any]:
    """Fetches sample rows to help the LLM understand cell formats and category values."""
    inspector = SchemaInspector(target=target)
    return inspector.get_sample_rows(table_name, limit=limit)


@server.tool(
    name="validate_sql",
    description="Validates that a SQL query is single-statement, syntactically valid, and strictly read-only.",
)
def validate_sql(sql: str, dialect: str = "postgres") -> Dict[str, Any]:
    """Parses SQL AST using SQLGlot to detect DDL/DML, multi-statements, and dangerous functions."""
    report = validate_readonly_sql(sql, dialect=dialect)
    return report.model_dump()


@server.tool(
    name="execute_readonly_sql",
    description="Executes a verified read-only SQL query with a timeout and returns bounded rows.",
)
def execute_readonly_sql(
    sql: str,
    max_rows: int = 100,
    target: str = "postgres",
) -> Dict[str, Any]:
    """Validates and executes a query, returning columns, rows, and row counts.
    
    Security Guarantee:
    - Queries failing validate_readonly_sql() are rejected immediately.
    - Read-only transaction mode and statement timeouts are enforced at database level.
    """
    dialect = "sqlite" if target.lower() == "local" else "postgres"

    # Step 1: Pre-execution AST validation
    validation = validate_readonly_sql(sql, dialect=dialect)
    if not validation.is_valid:
        return {
            "success": False,
            "error": f"Validation Rejected: {validation.error}",
            "columns": [],
            "rows": [],
            "row_count": 0,
        }

    settings = get_settings()
    bounded_limit = min(max(1, max_rows), settings.max_row_limit)

    # Step 2: Safe execution
    try:
        if target.lower() == "local":
            conn = get_local_db_connection(settings)
            cur = conn.cursor()
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchmany(bounded_limit)]
            columns = list(rows[0].keys()) if rows else []
            conn.close()
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "target": "local",
            }

        # PostgreSQL execution
        result = execute_readonly_query(sql, max_rows=bounded_limit, settings=settings)
        return {
            "success": True,
            "columns": result["columns"],
            "rows": result["rows"],
            "row_count": result["row_count"],
            "truncated": result["truncated"],
            "target": "postgres",
        }
    except Exception as exc:
        logger.error("Error executing read-only SQL via MCP: %s", exc)
        return {
            "success": False,
            "error": f"Execution error: {exc}",
            "columns": [],
            "rows": [],
            "row_count": 0,
        }


@server.tool(
    name="resolve_categorical_literals",
    description="Resolves natural language terms and abbreviations (e.g. 'casual leave', 'Tier 1') to exact database cell values (e.g. 'CL', 'T1').",
)
def resolve_categorical_literals(user_query: str, target: str = "postgres") -> List[Dict[str, Any]]:
    """Grounds user mentions against actual database cell literals to prevent zero-row query collapse."""
    resolver = LiteralResolver(target=target)
    matches = resolver.resolve_literals(user_query)
    return [m.model_dump() for m in matches]


@server.tool(
    name="prune_schema",
    description="Ranks and prunes tables and columns to only relevant candidates for a query, discovering join paths and minimizing tokens (Problem 2).",
)
def prune_schema(user_query: str, target: str = "postgres", top_k_tables: int = 3) -> Dict[str, Any]:
    """Prunes enterprise schema down to minimal compact cards and active join paths."""
    pruner = SchemaPruner(target=target)
    report = pruner.prune(user_query=user_query, top_k_tables=top_k_tables)
    return report.model_dump()


@server.tool(
    name="find_join_path",
    description="Finds the shortest relational join path between two tables using foreign keys and logical relationships.",
)
def find_join_path(source_table: str, target_table: str, target: str = "postgres") -> Dict[str, Any]:
    """Discovers join paths across tables using graph traversal."""
    graph = SchemaGraph(target=target)
    path = graph.find_shortest_join_path(source_table, target_table)
    return {
        "source_table": source_table,
        "target_table": target_table,
        "found_path": path is not None,
        "path_length": len(path) if path is not None else 0,
        "join_predicates": [p.join_predicate for p in path] if path else [],
    }


@server.tool(
    name="get_schema_graph",
    description="Retrieves the complete relational knowledge graph (all tables, descriptions, primary keys, and join edges).",
)
def get_schema_graph(target: str = "postgres") -> Dict[str, Any]:
    """Returns full relational schema graph."""
    graph = SchemaGraph(target=target)
    return graph.get_full_graph_definition()


@server.tool(
    name="get_table_relationships",
    description="Retrieves schemas, primary keys, and foreign-key join relationships for tables in the target database.",
)
def get_table_relationships(table_name: Optional[str] = None, target: str = "postgres") -> Dict[str, Any]:
    """Returns schema list, primary keys, and foreign key join edges for schema linking."""
    inspector = SchemaInspector(target=target)
    schemas = inspector.get_schemas()
    target_tables = [table_name] if table_name else inspector.get_table_names()
    fks: List[Dict[str, Any]] = []
    pks: Dict[str, List[str]] = {}
    for tbl in target_tables:
        try:
            meta = inspector.get_table_metadata(tbl, include_samples=False)
            pks[meta.full_name] = meta.primary_keys
            fks.extend(fk.model_dump() for fk in meta.foreign_keys)
        except Exception as exc:
            logger.debug("Skipping metadata inspection for '%s': %s", tbl, exc)
    return {
        "target": target,
        "schemas": schemas,
        "primary_keys": pks,
        "foreign_keys": fks,
    }


_mcp_retriever_cache: Dict[str, Any] = {}


@server.tool(
    name="retrieve_schema_context",
    description="Performs Two-Stage Enterprise Schema Linking (coarse tables + fine columns + nested fields + literals) and returns pruned SchemaContext.",
)
def retrieve_schema_context(
    user_query: str,
    target: str = "postgres",
    top_k_tables: int = 5,
    top_k_columns_per_table: int = 8,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Runs Two-Stage Schema Linking via MCP using cached BGE/Chroma indices."""
    from src.schema_context_builder import EnterpriseSchemaRetriever

    cache_key = target.lower().strip()
    retriever = _mcp_retriever_cache.get(cache_key)
    inspector = SchemaInspector(target=target)
    dialect = "sqlite" if cache_key in ("local", "sqlite") else cache_key

    if retriever is None:
        retriever = EnterpriseSchemaRetriever(in_memory=True, collection_prefix=f"mcp_{cache_key}")
        retriever.index_from_inspector(inspector=inspector, clear_existing=True, index_distinct_values=False)
        _mcp_retriever_cache[cache_key] = retriever
    elif refresh:
        retriever.refresh(inspector=inspector, dialect=dialect)

    ctx = retriever.retrieve_schema_context(
        query=user_query,
        top_k_tables=top_k_tables,
        top_k_columns_per_table=top_k_columns_per_table,
        target_dialect=dialect,
        require_db_evidence_for_literals=True,
    )
    return ctx.model_dump()




_doc_retriever_instance = None


@server.tool(
    name="retrieve_documentation",
    description="Retrieves domain business rules, compliance calculation formulas, and metric specifications (Spider 2.0 Problem 3).",
)
def retrieve_documentation(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Retrieves matching business rules and formulas from the enterprise documentation knowledge base."""
    global _doc_retriever_instance
    if _doc_retriever_instance is None:
        _doc_retriever_instance = DocumentationRetriever(in_memory=True)
    res = _doc_retriever_instance.retrieve_documentation(query=query, top_k=top_k)
    return res.model_dump()


@server.tool(
    name="get_metric_definitions",
    description="Lists all standardized enterprise business metrics, formulas, required tables, and filter rules (Spider 2.0 Problem 3).",
)
def get_metric_definitions(domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns catalog of standardized business metrics, filtered optionally by domain ('lending', 'operations', 'customer', 'financial')."""
    metrics = BUILTIN_METRICS
    if domain:
        d_lower = domain.lower().strip()
        metrics = [m for m in metrics if m.domain.lower() == d_lower]
    return [m.model_dump() for m in metrics]


@server.tool(
    name="explore_database",
    description="Executes read-only exploratory diagnostic probes (cardinality, date bounds, nulls, join fan-out) before writing queries (Spider 2.0 Problem 3).",
)
def explore_database(user_query: str, target: str = "postgres") -> Dict[str, Any]:
    """Autonomous explorer probing data distribution, boundaries, and join multiplicity to ground query generation."""
    explorer = DatabaseExplorer()
    report = explorer.explore_for_query(user_query=user_query, target=target)
    return report.model_dump()

def run_standalone_server() -> None:
    """Entrypoint to run the MCP server using standard I/O (stdio) transport."""
    server.run()


if __name__ == "__main__":
    run_standalone_server()
