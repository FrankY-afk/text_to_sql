"""Unit tests for Multi-Turn Conversational Follow-Up Memory (src/conversational_memory.py)."""

import pytest
from src.conversational_memory import ConversationalMemory, ConversationTurn, FollowupResolution


@pytest.fixture
def memory():
    return ConversationalMemory(max_history_turns=5)


def test_empty_memory_not_followup(memory):
    """When memory has no previous turns, any query is treated as standalone."""
    is_fu, reason = memory.is_followup_query("Show total loan disbursement for Maharashtra in 2024")
    assert not is_fu
    res = memory.resolve_followup("Show total loan disbursement for Maharashtra in 2024")
    assert not res.is_followup
    assert res.resolved_query == "Show total loan disbursement for Maharashtra in 2024"
    assert res.parent_turn_id is None


def test_drilldown_followup_resolution(memory):
    """'Now break that down by cluster' should inherit previous query and append breakdown."""
    memory.add_turn(
        user_query="Show total loan disbursement for Maharashtra in 2024",
        resolved_query="Show total loan disbursement for Maharashtra in 2024",
        generated_sql="SELECT sum(disbursed_amt) FROM group_loans JOIN outlets ON ... WHERE state = 'Maharashtra'",
        candidate_tables=["outlets", "group_loans"],
        row_count=1,
        raw_result=5000000.0,
    )

    is_fu, reason = memory.is_followup_query("Now break that down by cluster")
    assert is_fu

    res = memory.resolve_followup("Now break that down by cluster")
    assert res.is_followup
    assert res.parent_turn_id == 1
    assert "cluster_name" in res.breakdown_dimensions
    assert "broken down by cluster_name" in res.resolved_query.lower()
    assert "group_loans" in res.inherited_tables
    assert "outlets" in res.inherited_tables


def test_geographic_substitution_followup(memory):
    """'What about in Gujarat?' should replace Maharashtra with Gujarat."""
    memory.add_turn(
        user_query="Show total loan disbursement for Maharashtra in 2024",
        resolved_query="Show total loan disbursement for Maharashtra in 2024",
        generated_sql="SELECT sum(disbursed_amt) FROM group_loans JOIN outlets ...",
        candidate_tables=["outlets", "group_loans"],
        row_count=1,
    )

    res = memory.resolve_followup("What about in Gujarat?")
    assert res.is_followup
    assert res.parent_turn_id == 1
    assert "Gujarat" in res.resolved_query
    assert "Maharashtra" not in res.resolved_query
    assert "disbursement" in res.resolved_query.lower()


def test_filter_addition_followup(memory):
    """'Filter only for Tier 1 outlets' should append filter to parent query."""
    memory.add_turn(
        user_query="Show staff attendance rate for Tamil Nadu in 2024",
        resolved_query="Show staff attendance rate for Tamil Nadu in 2024",
        generated_sql="SELECT ...",
        candidate_tables=["staff", "staff_attendance", "outlets"],
        row_count=1,
    )

    res = memory.resolve_followup("Filter only for Tier 1")
    assert res.is_followup
    assert res.parent_turn_id == 1
    assert "tier" in res.new_filters
    assert res.new_filters["tier"] == "TIER 1"
    assert "filtered:" in res.resolved_query.lower() or "tier 1" in res.resolved_query.lower()


def test_elliptical_fragment_followup(memory):
    """Short fragment like 'In Maharashtra?' after a query should be detected as follow-up."""
    memory.add_turn(
        user_query="What is the average group loan disbursement per borrower?",
        resolved_query="What is the average group loan disbursement per borrower?",
        generated_sql="SELECT ...",
        candidate_tables=["group_loans"],
    )

    res = memory.resolve_followup("In Maharashtra?")
    assert res.is_followup
    assert "Maharashtra" in res.resolved_query


def test_history_summary_formatting(memory):
    """get_history_summary should format turns into clean Markdown."""
    memory.add_turn(
        user_query="Show total outlets in Gujarat",
        resolved_query="Show total outlets in Gujarat",
        generated_sql="SELECT count(*) FROM outlets WHERE state = 'Gujarat';",
        candidate_tables=["outlets"],
        row_count=15,
        success=True,
    )

    summary = memory.get_history_summary()
    assert "Conversation History" in summary
    assert "Turn #1" in summary
    assert "Show total outlets in Gujarat" in summary
    assert "outlets" in summary


def test_clear_memory(memory):
    """clear() should reset turns."""
    memory.add_turn(
        user_query="Query 1",
        resolved_query="Query 1",
        generated_sql="SELECT 1;",
    )
    assert len(memory.turns) == 1
    memory.clear()
    assert len(memory.turns) == 0
    assert memory.get_last_turn() is None


def test_max_history_enforcement():
    """Memory should truncate oldest turns when exceeding max_history_turns."""
    mem = ConversationalMemory(max_history_turns=3)
    for i in range(5):
        mem.add_turn(
            user_query=f"Query {i}",
            resolved_query=f"Query {i}",
            generated_sql=f"SELECT {i};",
        )
    assert len(mem.turns) == 3
    assert mem.turns[0].user_query == "Query 2"
    assert mem.turns[-1].user_query == "Query 4"
