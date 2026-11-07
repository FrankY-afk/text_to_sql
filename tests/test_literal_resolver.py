"""Unit tests for Categorical Literal Grounding (Problem 2 Slice)."""

from __future__ import annotations

import pytest
from src.literal_resolver import LiteralResolver
from tests.conftest import postgres_required


@pytest.fixture
def resolver():
    return LiteralResolver(target="postgres")


def test_resolve_abbreviations_and_synonyms(resolver):
    """Verifies that human phrases resolve to cryptic database codes."""
    text = "How many employees were on casual leave or sick leave in June?"
    matches = resolver.resolve_literals(text)

    resolved_map = {m.term: m.database_literal for m in matches}
    assert resolved_map.get("casual leave") == "CL"
    assert resolved_map.get("sick leave") == "SL"


def test_resolve_tier_abbreviations(resolver):
    """Verifies that 'Tier 1' resolves to 'T1' in outlets table."""
    text = "Find the total portfolio for outlets in Tier 1 and Tier 2."
    matches = resolver.resolve_literals(text)

    resolved_map = {m.term: m.database_literal for m in matches}
    assert resolved_map.get("tier 1") == "T1"
    assert resolved_map.get("tier 2") == "T2"


@postgres_required
def test_resolve_live_state_literals(resolver):
    """Verifies that live PostgreSQL states are matched case-insensitively."""
    text = "Find all outlets in telangana state."
    matches = resolver.resolve_literals(text)


    state_matches = [m for m in matches if m.column_name == "state_name"]
    assert len(state_matches) >= 1
    assert state_matches[0].database_literal == "Telangana"


def test_generate_grounding_prompt_hint(resolver):
    """Verifies that prompt hints format cleanly for downstream LLM prompts."""
    text = "List employees on casual leave."
    hint = resolver.generate_grounding_prompt_hint(text)

    assert "[Discovered Database Literals (Schema Grounding)]:" in hint
    assert "attendance_code = 'CL'" in hint
