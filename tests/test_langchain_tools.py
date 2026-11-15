"""Unit tests for LangChain Tool Adapters and Prompt Templates."""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool

from src.langchain_prompts import (
    SQLPlan,
    SYSTEM_PROMPT_TEXT2SQL,
    format_langchain_context,
    get_text2sql_prompt_template,
)
from src.langchain_tools import (
    get_langchain_tool_map,
    get_langchain_tools,
)
from tests.conftest import postgres_required


@pytest.fixture
def tools():
    return get_langchain_tools()


@pytest.fixture
def tool_map():
    return get_langchain_tool_map()


def test_langchain_tool_count_and_types(tools, tool_map):
    """Verifies that all 9 MCP tools are exported as valid LangChain BaseTool instances."""
    assert len(tools) == 12
    assert len(tool_map) == 12

    expected_names = {
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
    assert set(tool_map.keys()) == expected_names

    for t in tools:
        assert isinstance(t, BaseTool)
        assert t.name in expected_names
        assert t.description is not None
        assert len(t.description) > 10
        assert t.args_schema is not None


@postgres_required
def test_langchain_tool_sync_invocation(tool_map):
    """Verifies that LangChain tools execute synchronously via .invoke()."""
    # 1. list_tables returns list of table names
    res_tables = tool_map["list_tables"].invoke({"target": "postgres"})
    assert isinstance(res_tables, list)
    assert "outlets" in res_tables

    # 2. resolve_categorical_literals (fuzzy typo resolution)
    res_lit = tool_map["resolve_categorical_literals"].invoke({
        "natural_language_query": "Show branches in Maharahtra",
        "target": "postgres",
    })
    assert len(res_lit) >= 1
    assert any(m["database_literal"] == "Maharashtra" for m in res_lit)

    # 3. prune_schema (Problem 2)
    res_prune = tool_map["prune_schema"].invoke({
        "natural_language_query": "What is the staff attendance rate in June?",
        "target": "postgres",
    })
    assert "candidate_tables" in res_prune
    assert "staff_attendance" in res_prune["candidate_tables"]
    assert res_prune["token_reduction_pct"] > 0

    # 4. find_join_path (Problem 2 BFS)
    res_path = tool_map["find_join_path"].invoke({
        "source_table": "staff_attendance",
        "target_table": "outlets",
        "target": "postgres",
    })
    assert res_path["found_path"] is True
    assert res_path["path_length"] == 2


@postgres_required
@pytest.mark.anyio
async def test_langchain_tool_async_invocation(tool_map):
    """Verifies that LangChain tools execute asynchronously via .ainvoke()."""
    res = await tool_map["list_tables"].ainvoke({"target": "postgres"})
    assert isinstance(res, list)
    assert "staff" in res


def test_langchain_prompt_template_and_context_formatter():
    """Verifies LangChain ChatPromptTemplate generation and LCEL message formatting."""
    template = get_text2sql_prompt_template()
    assert template is not None

    context = format_langchain_context(
        pruned_schema_markdown="TABLE outlets (outlet_code, state_name)",
        grounded_literals=[{
            "term": "Maharahtra",
            "table_name": "outlets",
            "column_name": "state_name",
            "database_literal": "Maharashtra",
        }],
        join_paths=["staff.outlet_code = outlets.outlet_code"],
    )

    messages = template.format_messages(
        question="Show all outlets in Maharahtra",
        schema_context=context["schema_context"],
        literal_hints=context["literal_hints"],
    )

    assert len(messages) == 2
    # System message contains our guardrails against Problem 1 & 2
    system_text = messages[0].content
    assert "PROBLEM 1: ERRONEOUS DATA ANALYSIS" in system_text
    assert "PROBLEM 2: SCHEMA LINKING" in system_text
    assert "COUNT(DISTINCT" in system_text
    assert "NULLIF(" in system_text

    # Human message contains the user question and injected grounding context
    human_text = messages[1].content
    assert "Show all outlets in Maharahtra" in human_text
    assert "Maharashtra" in human_text
    assert "staff.outlet_code = outlets.outlet_code" in human_text


def test_langchain_sql_plan_schema():
    """Verifies Pydantic schema serialization for .with_structured_output(SQLPlan)."""
    plan = SQLPlan(
        metric_name="June 2026 Staff Attendance Rate",
        target_entity="staff_no",
        is_percentage=True,
        candidate_tables=["staff", "staff_attendance"],
        join_predicates=["staff_attendance.staff_no = staff.staff_no"],
        sql_query="SELECT (COUNT(DISTINCT a.staff_no) * 100.0) / NULLIF(COUNT(DISTINCT s.staff_no), 0) FROM staff s LEFT JOIN staff_attendance a ON s.staff_no = a.staff_no;",
        explanation="Strict distinct count with safe NULLIF denominator.",
    )
    plan_dict = plan.model_dump()
    assert plan_dict["metric_name"] == "June 2026 Staff Attendance Rate"
    assert plan_dict["is_percentage"] is True
    assert len(plan_dict["candidate_tables"]) == 2

def test_langchain_tool_local_invocation(tool_map):
    """Verifies that LangChain tools execute offline with local SQLite target."""
    res_tables = tool_map["list_tables"].invoke({"target": "local"})
    assert isinstance(res_tables, list)
    assert len(res_tables) > 0

    res_val = tool_map["validate_sql"].invoke({
        "sql": "SELECT COUNT(*) FROM users;",
        "target": "local",
    })
    assert res_val["is_valid"] is True
