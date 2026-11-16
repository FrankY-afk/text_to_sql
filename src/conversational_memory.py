"""Multi-Turn Conversational Follow-Up Memory for Text-to-SQL.

Maintains multi-turn context across user interactions to support:
1. Follow-up query detection (pronouns, elliptical fragments, drill-downs, pivots).
2. Contextual query resolution (inheriting candidate tables, WHERE filters, metrics).
3. Substitution vs. addition logic (e.g. "What about Gujarat?" replaces "Maharashtra",
   while "Filter only for rural" adds an outlet_type filter).
4. Formatted conversational history summaries for LLM prompt context injection.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    """Snapshot of a single interaction turn in a multi-turn analytical session."""

    turn_id: int
    user_query: str
    resolved_query: str
    target: str = "postgres"
    candidate_tables: List[str] = Field(default_factory=list)
    active_filters: Dict[str, Any] = Field(default_factory=dict)
    group_by_dimensions: List[str] = Field(default_factory=list)
    metric_name: Optional[str] = None
    generated_sql: str = ""
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    raw_result: Optional[float] = None
    success: bool = True
    timestamp: float = Field(default_factory=time.time)


class FollowupResolution(BaseModel):
    """Result of analyzing and resolving an incoming user query against conversation history."""

    is_followup: bool
    original_query: str
    resolved_query: str
    parent_turn_id: Optional[int] = None
    inherited_tables: List[str] = Field(default_factory=list)
    inherited_filters: Dict[str, Any] = Field(default_factory=dict)
    new_filters: Dict[str, Any] = Field(default_factory=dict)
    breakdown_dimensions: List[str] = Field(default_factory=list)
    resolution_explanation: str = ""


# ---------------------------------------------------------------------------
# Dimension & Entity Patterns for Heuristic Resolution
# ---------------------------------------------------------------------------

STATE_NAMES = {
    "maharashtra", "tamil nadu", "karnataka", "gujarat", "rajasthan",
    "uttar pradesh", "madhya pradesh", "bihar", "west bengal", "punjab",
    "haryana", "kerala", "andhra pradesh", "telangana", "delhi", "odisha",
}

STATUS_KEYWORDS = {
    "approved", "rejected", "pending", "disbursed", "closed", "active",
    "present", "absent", "defaulted", "churned",
}

DIMENSION_KEYWORDS = {
    "cluster": "cluster_name",
    "cluster_name": "cluster_name",
    "tier": "outlet_tier",
    "outlet tier": "outlet_tier",
    "outlet_tier": "outlet_tier",
    "outlet": "outlet_name",
    "outlet_name": "outlet_name",
    "branch": "outlet_name",
    "state": "state",
    "month": "month",
    "year": "year",
    "staff": "staff_name",
    "staff_no": "staff_no",
    "department": "department",
    "gender": "gender",
    "type": "outlet_type",
    "outlet type": "outlet_type",
}


# ---------------------------------------------------------------------------
# Conversational Memory Engine
# ---------------------------------------------------------------------------

class ConversationalMemory:
    """Manages multi-turn conversation state, follow-up detection, and query reformulation."""

    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns
        self.turns: List[ConversationTurn] = []

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Returns the most recent conversation turn, or None if history is empty."""
        return self.turns[-1] if self.turns else None

    def get_last_successful_turn(self) -> Optional[ConversationTurn]:
        """Returns the most recent successful turn."""
        for turn in reversed(self.turns):
            if turn.success and turn.generated_sql:
                return turn
        return None

    def clear(self) -> None:
        """Clears all conversation history."""
        self.turns.clear()
        logger.info("Conversational memory cleared.")

    def add_turn(
        self,
        user_query: str,
        resolved_query: str,
        generated_sql: str,
        target: str = "postgres",
        candidate_tables: Optional[List[str]] = None,
        active_filters: Optional[Dict[str, Any]] = None,
        group_by_dimensions: Optional[List[str]] = None,
        metric_name: Optional[str] = None,
        columns: Optional[List[str]] = None,
        rows: Optional[List[Dict[str, Any]]] = None,
        row_count: int = 0,
        raw_result: Optional[float] = None,
        success: bool = True,
    ) -> ConversationTurn:
        """Appends a new turn to memory, pruning oldest turns if exceeding limit."""
        turn_id = len(self.turns) + 1
        turn = ConversationTurn(
            turn_id=turn_id,
            user_query=user_query,
            resolved_query=resolved_query,
            target=target,
            candidate_tables=candidate_tables or [],
            active_filters=active_filters or self._extract_simple_filters(resolved_query),
            group_by_dimensions=group_by_dimensions or [],
            metric_name=metric_name,
            generated_sql=generated_sql,
            columns=columns or [],
            rows=rows or [],
            row_count=row_count,
            raw_result=raw_result,
            success=success,
            timestamp=time.time(),
        )
        self.turns.append(turn)

        # Enforce history limit
        if len(self.turns) > self.max_history_turns:
            self.turns = self.turns[-self.max_history_turns:]

        return turn

    def is_followup_query(self, query: str) -> Tuple[bool, str]:
        """Determines if a natural language query relies on context from previous turns.

        Returns:
            Tuple of (is_followup: bool, reason: str).
        """
        if not self.turns:
            return False, "No previous conversation history"

        q_clean = query.strip().lower()
        words = q_clean.split()

        # 1. Anaphoric Pronouns & Demonstrative References
        anaphoric_patterns = [
            r"\b(that|this|those|these|them|it|same|the same|above|previous)\b",
            r"\b(break (that|this|it) down)\b",
            r"\b(show (that|this|it))\b",
            r"\b(split (that|this|it))\b",
        ]
        for pat in anaphoric_patterns:
            if re.search(pat, q_clean):
                return True, f"Anaphoric reference detected ('{re.search(pat, q_clean).group(0)}')"

        # 2. Drill-Down / Grouping Starters
        drilldown_patterns = [
            r"^(now\s+)?(break\s+down|group\s+by|split\s+by|slice\s+by)",
            r"^(now\s+)?by\s+[a-z_]+",
            r"^(now\s+)?per\s+[a-z_]+",
            r"\bbroken\s+down\s+by\b",
        ]
        for pat in drilldown_patterns:
            if re.search(pat, q_clean):
                return True, "Drill-down / grouping specification"

        # 3. Filter Additions / Modifications
        filter_starters = [
            r"^(only\s+(for|in|with)|filter\s+(only\s+)?(by|for|where)|limit\s+to|restrict\s+to|just\s+(for|the)|also\s+(for|in|filter)|exclude\s+)",
            r"^(and\s+also|and\s+only|for\s+only)\b",
        ]
        for pat in filter_starters:
            if re.search(pat, q_clean):
                return True, "Filter modifier specification"

        # 4. Pivot / Substitution Phrases
        pivot_patterns = [
            r"^(what\s+about|how\s+about|what\s+if)\b",
            r"\b(instead\s+of|instead)\b",
            r"^(and\s+for|now\s+for|what\s+for)\b",
        ]
        for pat in pivot_patterns:
            if re.search(pat, q_clean):
                return True, "Pivot / entity substitution phrase"

        # 5. Elliptical Queries (short fragments <= 5 words starting with prepositions or dimensions)
        if len(words) <= 5:
            elliptical_starters = ("in ", "for ", "by ", "with ", "at ", "of ", "per ", "now ")
            if q_clean.startswith(elliptical_starters):
                return True, f"Short elliptical fragment ({len(words)} words)"
            # Or bare dimension / state question like "Gujarat?", "Cluster?", "Tier 1?"
            stripped = q_clean.rstrip("?").strip()
            if stripped in STATE_NAMES or stripped in DIMENSION_KEYWORDS:
                return True, f"Direct entity mention fragment ('{stripped}')"

        return False, "Standalone query"

    def resolve_followup(self, query: str, target: str = "postgres") -> FollowupResolution:
        """Resolves an incoming query, merging multi-turn context when a follow-up is detected.

        Args:
            query: The user's query string.
            target: Active database target.

        Returns:
            FollowupResolution with is_followup, resolved_query, and inherited details.
        """
        is_followup, reason = self.is_followup_query(query)
        if not is_followup:
            return FollowupResolution(
                is_followup=False,
                original_query=query,
                resolved_query=query,
                resolution_explanation="Treated as standalone new query.",
            )

        parent_turn = self.get_last_successful_turn() or self.get_last_turn()
        if not parent_turn:
            return FollowupResolution(
                is_followup=False,
                original_query=query,
                resolved_query=query,
                resolution_explanation="No prior parent turn found; executing standalone.",
            )

        q_clean = query.strip()
        q_lower = q_clean.lower()

        inherited_tables = list(parent_turn.candidate_tables)
        inherited_filters = dict(parent_turn.active_filters)
        new_filters: Dict[str, Any] = {}
        breakdown_dims: List[str] = []
        resolution_notes: List[str] = []

        # -------------------------------------------------------------
        # 1. Check for Drill-down / Group-by Additions
        # -------------------------------------------------------------
        for dim_kw, dim_col in DIMENSION_KEYWORDS.items():
            dim_match = re.search(rf"\b(by|per|group by|split by|break down by)\s+{re.escape(dim_kw)}\b", q_lower)
            if dim_match:
                breakdown_dims.append(dim_col)
                resolution_notes.append(f"Added breakdown dimension '{dim_col}'")

        # -------------------------------------------------------------
        # 2. Check for State / Geographic Substitution or Addition
        # -------------------------------------------------------------
        mentioned_state: Optional[str] = None
        for state in STATE_NAMES:
            if re.search(rf"\b{re.escape(state)}\b", q_lower):
                mentioned_state = state.title()
                break

        if mentioned_state:
            # Check if parent query had a state filter
            if "state" in inherited_filters or "state" in parent_turn.resolved_query.lower():
                old_state = inherited_filters.get("state", "previous state")
                inherited_filters.pop("state", None)
                new_filters["state"] = mentioned_state
                resolution_notes.append(f"Substituted state filter '{old_state}' -> '{mentioned_state}'")
            else:
                new_filters["state"] = mentioned_state
                resolution_notes.append(f"Added state filter '{mentioned_state}'")

        # -------------------------------------------------------------
        # 3. Check for Date / Year Mention
        # -------------------------------------------------------------
        year_match = re.search(r"\b(202[0-9])\b", q_lower)
        if year_match:
            year_val = year_match.group(1)
            if "year" in inherited_filters:
                old_year = inherited_filters.get("year")
                inherited_filters.pop("year", None)
                new_filters["year"] = year_val
                resolution_notes.append(f"Substituted year filter '{old_year}' -> '{year_val}'")
            else:
                new_filters["year"] = year_val
                resolution_notes.append(f"Added year filter '{year_val}'")

        # -------------------------------------------------------------
        # 4. Check for Tier / Status Mention
        # -------------------------------------------------------------
        tier_match = re.search(r"\b(tier[\s-]?[123]|t[123])\b", q_lower)
        if tier_match:
            tier_val = tier_match.group(1).upper()
            new_filters["tier"] = tier_val
            resolution_notes.append(f"Added tier filter '{tier_val}'")

        # -------------------------------------------------------------
        # 5. Synthesize Resolved Question
        # -------------------------------------------------------------
        base_parent_query = parent_turn.resolved_query.rstrip("?.")

        # Case A: Pivot / Substitution (e.g. "What about Gujarat?", "in 2023 instead")
        if mentioned_state and ("what about" in q_lower or "how about" in q_lower or "instead" in q_lower or len(q_clean.split()) <= 4):
            # Replace state in parent query
            pattern = re.compile(r"\b(in|for)\s+(" + "|".join(STATE_NAMES) + r")\b", re.IGNORECASE)
            if pattern.search(base_parent_query):
                resolved = pattern.sub(f"\\1 {mentioned_state}", base_parent_query)
            else:
                resolved = f"{base_parent_query} for {mentioned_state}"
        elif year_match and ("what about" in q_lower or "how about" in q_lower or "instead" in q_lower or len(q_clean.split()) <= 4):
            # Replace year in parent query
            new_yr = year_match.group(1)
            if re.search(r"\b202[0-9]\b", base_parent_query):
                resolved = re.sub(r"\b202[0-9]\b", new_yr, base_parent_query)
            else:
                resolved = f"{base_parent_query} in {new_yr}"
        # Case B: Drill-down / Breakdown (e.g. "Now break that down by cluster")
        elif breakdown_dims:
            dim_str = ", ".join(breakdown_dims)
            # Remove any trailing breakdown from parent
            clean_base = re.sub(r"\s+broken down by\s+.*$", "", base_parent_query, flags=re.IGNORECASE)
            resolved = f"{clean_base} broken down by {dim_str}"
        # Case C: Filter addition (e.g. "Filter only for rural outlets")
        elif "only" in q_lower or "filter" in q_lower or "limit" in q_lower:
            # Strip filter prefixes to isolate the criteria
            clean_filter = re.sub(r"^(now\s+)?(filter\s+(only\s+)?(by|for|where)|only\s+(for|in|with)|limit\s+to|just\s+(for|the))\s+", "", q_clean, flags=re.IGNORECASE)
            resolved = f"{base_parent_query} (filtered: {clean_filter})"
        # Case D: Generic modifier or short fragment
        else:
            clean_mod = re.sub(r"\b(that|this|it|those|these)\b", "the metric", q_clean, flags=re.IGNORECASE)
            resolved = f"{base_parent_query}, {clean_mod}"

        explanation = (
            f"Resolved follow-up turn against Turn #{parent_turn.turn_id}. "
            + ("; ".join(resolution_notes) if resolution_notes else "Merged contextual intent.")
        )

        return FollowupResolution(
            is_followup=True,
            original_query=query,
            resolved_query=resolved,
            parent_turn_id=parent_turn.turn_id,
            inherited_tables=inherited_tables,
            inherited_filters=inherited_filters,
            new_filters=new_filters,
            breakdown_dimensions=breakdown_dims,
            resolution_explanation=explanation,
        )

    def _extract_simple_filters(self, query: str) -> Dict[str, Any]:
        """Extracts common entities (state, year, tier) from a query string."""
        filters: Dict[str, Any] = {}
        q_lower = query.lower()

        for state in STATE_NAMES:
            if re.search(rf"\b{re.escape(state)}\b", q_lower):
                filters["state"] = state.title()
                break

        year_match = re.search(r"\b(202[0-9])\b", q_lower)
        if year_match:
            filters["year"] = year_match.group(1)

        tier_match = re.search(r"\b(tier[\s-]?[123]|t[123])\b", q_lower)
        if tier_match:
            filters["tier"] = tier_match.group(1).upper()

        return filters

    def get_history_summary(self, max_turns: int = 5) -> str:
        """Generates a compact Markdown history block for LLM prompt context injection."""
        if not self.turns:
            return "No previous conversation turns."

        recent = self.turns[-max_turns:]
        lines: List[str] = [f"### Conversation History ({len(recent)} recent turns):"]
        for t in recent:
            status_str = "SUCCESS" if t.success else "FAILED"
            result_str = f"{t.raw_result:.2f}%" if t.raw_result is not None else f"{t.row_count} rows"
            lines.append(f"- **Turn #{t.turn_id}** ({status_str}):")
            lines.append(f"  - User Query: \"{t.user_query}\"")
            if t.resolved_query != t.user_query:
                lines.append(f"  - Contextualized: \"{t.resolved_query}\"")
            if t.candidate_tables:
                lines.append(f"  - Tables: [{', '.join(t.candidate_tables)}]")
            if t.generated_sql:
                short_sql = t.generated_sql.strip().replace("\n", " ")[:120] + ("..." if len(t.generated_sql) > 120 else "")
                lines.append(f"  - SQL: `{short_sql}`")
            lines.append(f"  - Result: {result_str}")

        return "\n".join(lines)

    def export_history(self) -> List[Dict[str, Any]]:
        """Serializes the conversation turns for external reporting or inspection."""
        return [t.model_dump() for t in self.turns]
