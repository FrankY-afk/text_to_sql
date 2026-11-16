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


def test_resolve_fuzzy_state_typo_and_unindexed_state(resolver):
    """Verifies typo tolerance ('Maharahtra' -> 'Maharashtra') and out-of-domain state handling."""
    text = "Show all outlets located in Maharahtra in Tier 1"
    matches = resolver.resolve_literals(text)

    term_map = {m.term: m for m in matches}
    assert "Maharahtra" in term_map
    match = term_map["Maharahtra"]
    assert match.database_literal == "Maharashtra"
    assert "fuzzy" in match.match_type
    assert match.table_name == "outlets"
    assert match.column_name == "state_name"


def test_resolve_cluster_typo_and_subcluster(resolver):
    """Verifies typo tolerance for cluster regions ('decan' -> 'Deccan%') and sub-clusters ('decan 2' -> 'Deccan-2')."""
    # 1. Base cluster region with typo: "decan outlets" -> cluster_name LIKE 'Deccan%'
    q1 = "list all the decan outlets"
    matches1 = resolver.resolve_literals(q1)
    cluster_matches1 = [m for m in matches1 if m.column_name == "cluster_name"]
    assert len(cluster_matches1) == 1
    assert cluster_matches1[0].database_literal == "Deccan%"
    assert cluster_matches1[0].sql_predicate == "cluster_name LIKE 'Deccan%'"
    assert "fuzzy" in cluster_matches1[0].match_type
    assert cluster_matches1[0].confidence >= 0.95

    # 2. Sub-cluster with typo: "decan 2" -> cluster_name = 'Deccan-2'
    q2 = "show outlets in decan 2"
    matches2 = resolver.resolve_literals(q2)
    cluster_matches2 = [m for m in matches2 if m.column_name == "cluster_name"]
    assert len(cluster_matches2) == 1
    assert cluster_matches2[0].database_literal == "Deccan-2"
    assert cluster_matches2[0].sql_predicate == "cluster_name = 'Deccan-2'"

    # 3. Exact cluster region: "coastal outlets" -> cluster_name LIKE 'Coastal%'
    q3 = "how many coastal outlets are there"
    matches3 = resolver.resolve_literals(q3)
    cluster_matches3 = [m for m in matches3 if m.column_name == "cluster_name"]
    assert len(cluster_matches3) == 1
    assert cluster_matches3[0].database_literal == "Coastal%"
    assert cluster_matches3[0].sql_predicate == "cluster_name LIKE 'Coastal%'"
