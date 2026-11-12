"""Query executor acting as the MCP client layer.

Dispatches requests through the MCP protocol / MCPServer interface,
handling tool discovery, argument serialization, and structured response parsing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from src.mcp_server import server

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    """Raised when an MCP tool call fails or returns an error."""


class MCPQueryExecutor:
    """Client layer communicating with the MCP Server to discover tools and execute read-only SQL."""

    def __init__(self, mcp_server=None):
        self.server = mcp_server or server

    async def list_available_mcp_tools(self) -> List[Dict[str, Any]]:
        """Discovers all tools advertised by the MCP server."""
        tools = await self.server.list_tools()
        return [{"name": t.name, "description": t.description} for t in tools]

    async def list_tables(self, target: str = "postgres") -> List[str]:
        """Calls MCP tool 'list_tables'."""
        result = await self.server.call_tool("list_tables", {"target": target})
        if result.is_error:
            raise MCPClientError(f"list_tables failed: {result.content}")
        return result.structured_content["result"]

    async def get_table_schema(self, table_name: str, target: str = "postgres") -> str:
        """Calls MCP tool 'get_table_schema'."""
        result = await self.server.call_tool("get_table_schema", {"table_name": table_name, "target": target})
        if result.is_error:
            raise MCPClientError(f"get_table_schema failed: {result.content}")
        return result.structured_content["result"]

    async def get_sample_rows(self, table_name: str, limit: int = 3, target: str = "postgres") -> Dict[str, Any]:
        """Calls MCP tool 'get_sample_rows'."""
        result = await self.server.call_tool("get_sample_rows", {"table_name": table_name, "limit": limit, "target": target})
        if result.is_error:
            raise MCPClientError(f"get_sample_rows failed: {result.content}")
        return result.structured_content["result"]

    async def validate_sql(self, sql: str, dialect: str = "postgres") -> Dict[str, Any]:
        """Calls MCP tool 'validate_sql'."""
        result = await self.server.call_tool("validate_sql", {"sql": sql, "dialect": dialect})
        if result.is_error:
            raise MCPClientError(f"validate_sql failed: {result.content}")
        return result.structured_content["result"]

    async def execute_readonly_sql(
        self,
        sql: str,
        max_rows: int = 100,
        target: str = "postgres",
    ) -> Dict[str, Any]:
        """Calls MCP tool 'execute_readonly_sql' to run a validated query safely."""
        result = await self.server.call_tool(
            "execute_readonly_sql",
            {"sql": sql, "max_rows": max_rows, "target": target},
        )
        if result.is_error:
            raise MCPClientError(f"execute_readonly_sql failed: {result.content}")
        return result.structured_content["result"]

    async def resolve_categorical_literals(self, user_query: str, target: str = "postgres") -> List[Dict[str, Any]]:
        """Calls MCP tool 'resolve_categorical_literals'."""
        result = await self.server.call_tool(
            "resolve_categorical_literals",
            {"user_query": user_query, "target": target},
        )
        if result.is_error:
            raise MCPClientError(f"resolve_categorical_literals failed: {result.content}")
        return result.structured_content["result"]

    async def get_table_relationships(
        self,
        table_name: Optional[str] = None,
        target: str = "postgres",
    ) -> Dict[str, Any]:
        """Calls MCP tool 'get_table_relationships'."""
        payload: Dict[str, Any] = {"target": target}
        if table_name is not None:
            payload["table_name"] = table_name
        result = await self.server.call_tool("get_table_relationships", payload)
        if result.is_error:
            raise MCPClientError(f"get_table_relationships failed: {result.content}")
        return result.structured_content["result"]

    async def retrieve_schema_context(
        self,
        user_query: str,
        target: str = "postgres",
        top_k_tables: int = 5,
        top_k_columns_per_table: int = 8,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Calls MCP tool 'retrieve_schema_context'."""
        result = await self.server.call_tool(
            "retrieve_schema_context",
            {
                "user_query": user_query,
                "target": target,
                "top_k_tables": top_k_tables,
                "top_k_columns_per_table": top_k_columns_per_table,
                "refresh": refresh,
            },
        )
        if result.is_error:
            raise MCPClientError(f"retrieve_schema_context failed: {result.content}")
        return result.structured_content["result"]

    # Synchronous convenience wrappers for ease of use in scripts and tests
    def list_tables_sync(self, target: str = "postgres") -> List[str]:
        return asyncio.run(self.list_tables(target=target))

    def get_table_schema_sync(self, table_name: str, target: str = "postgres") -> str:
        return asyncio.run(self.get_table_schema(table_name=table_name, target=target))

    def get_sample_rows_sync(self, table_name: str, limit: int = 3, target: str = "postgres") -> Dict[str, Any]:
        return asyncio.run(self.get_sample_rows(table_name=table_name, limit=limit, target=target))

    def validate_sql_sync(self, sql: str, dialect: str = "postgres") -> Dict[str, Any]:
        return asyncio.run(self.validate_sql(sql=sql, dialect=dialect))

    def execute_readonly_sql_sync(self, sql: str, max_rows: int = 100, target: str = "postgres") -> Dict[str, Any]:
        return asyncio.run(self.execute_readonly_sql(sql=sql, max_rows=max_rows, target=target))

    def resolve_categorical_literals_sync(self, user_query: str, target: str = "postgres") -> List[Dict[str, Any]]:
        return asyncio.run(self.resolve_categorical_literals(user_query=user_query, target=target))

    def get_table_relationships_sync(
        self,
        table_name: Optional[str] = None,
        target: str = "postgres",
    ) -> Dict[str, Any]:
        return asyncio.run(self.get_table_relationships(table_name=table_name, target=target))

    def retrieve_schema_context_sync(
        self,
        user_query: str,
        target: str = "postgres",
        top_k_tables: int = 5,
        top_k_columns_per_table: int = 8,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        return asyncio.run(
            self.retrieve_schema_context(
                user_query=user_query,
                target=target,
                top_k_tables=top_k_tables,
                top_k_columns_per_table=top_k_columns_per_table,
                refresh=refresh,
            )
        )
