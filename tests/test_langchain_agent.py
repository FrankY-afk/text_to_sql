"""Unit and integration tests for LangChain Autonomous Text-to-SQL Agent (src/langchain_agent.py)."""

import asyncio
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from src.conversational_memory import ConversationalMemory
from src.langchain_agent import (
    AgentExecutionResult,
    HuggingFaceChatModel,
    LangChainText2SQLAgent,
)
from src.query_executor import MCPQueryExecutor


def test_hf_chat_model_initialization():
    """HuggingFaceChatModel should initialize and declare proper llm_type."""
    model = HuggingFaceChatModel()
    assert model._llm_type == "huggingface_chat_model"
    assert model.model_name is not None


def test_hf_chat_model_offline_fallback():
    """When Hugging Face is not configured, model returns offline notification gracefully."""
    model = HuggingFaceChatModel()
    # Mock unconfigured client
    model.hf_client.token = None
    model.hf_client.endpoint_url = None

    res = model.invoke([HumanMessage(content="SELECT 1;")])
    assert res.content is not None
    assert "LLM inference offline" in res.content


def test_langchain_agent_tools_loading():
    """Agent should initialize all 12 MCP tools."""
    agent = LangChainText2SQLAgent()
    assert len(agent.tools) == 12
    tool_names = set(agent.tool_map.keys())
    expected_tools = {
        "list_tables",
        "get_table_schema",
        "get_sample_rows",
        "validate_sql",
        "execute_readonly_sql",
        "resolve_categorical_literals",
        "prune_schema",
        "find_join_path",
        "get_schema_graph",
        "retrieve_documentation",
        "get_metric_definitions",
        "explore_database",
    }
    assert expected_tools.issubset(tool_names)


def test_langchain_agent_execution_local_sqlite():
    """Agent should run end-to-end against local SQLite database."""
    agent = LangChainText2SQLAgent()
    result = agent.run("Show distinct user activity events in February 2024", target="local")

    assert isinstance(result, AgentExecutionResult)
    assert result.success is True
    assert len(result.sql) > 0
    assert any(tbl in result.sql.lower() for tbl in ("activity_events", "users"))
    assert len(result.steps_taken) >= 5
    assert len(result.intermediate_tool_calls) >= 3

    # Check intermediate tools called
    tools_called = [tc["tool"] for tc in result.intermediate_tool_calls]
    assert "prune_schema" in tools_called
    assert "resolve_categorical_literals" in tools_called


def test_langchain_agent_multi_turn_followup():
    """Agent should resolve multi-turn follow-up queries using ConversationalMemory."""
    mem = ConversationalMemory()
    agent = LangChainText2SQLAgent(memory=mem)

    # Turn 1: Base query
    turn1_res = agent.run("Show distinct user activity events in February 2024", target="local")
    assert turn1_res.success is True
    assert turn1_res.is_followup is False
    assert len(mem.turns) == 1

    # Turn 2: Follow-up substitution
    turn2_res = agent.run("What about in 2023?", target="local")
    assert turn2_res.success is True
    assert turn2_res.is_followup is True
    assert "2023" in turn2_res.resolved_query
    assert len(mem.turns) == 2


def test_langchain_agent_arun():
    """Agent should support async invocation via arun()."""
    agent = LangChainText2SQLAgent()
    result = asyncio.run(agent.arun("List distinct users", target="local"))
    assert isinstance(result, AgentExecutionResult)
    assert result.success is True
