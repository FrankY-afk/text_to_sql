"""Stage 4: Distinct Value Indexer & Categorical Literal Grounding (Problem 2).

Solves Entity Literal Mismatch where natural language mentions (e.g., "California",
"casual leave", "Tier 1") differ from stored cell values (e.g., 'CA', 'CL', 'T1').

Capabilities:
1. Dynamically inspects permitted categorical columns via SchemaInspector using safe
   identifier validation and parameterized SQL queries.
2. Indexes observed distinct database values into ChromaDB with BGE embeddings.
3. Enriches observed DB values with standard linguistic / ISO abbreviations ONLY when
   the underlying code is backed by live database evidence (never assumes blind mappings).
4. Returns confidence scores and database evidence provenance for every resolved literal.
5. Supports explicit index refresh when underlying table data changes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.config import Settings, get_chroma_client, get_embedding_function, get_settings
from src.db import execute_readonly_query, get_local_db_connection
from src.schema_inspector import SchemaInspector, TableMetadata, validate_sql_identifier

logger = logging.getLogger(__name__)

# Generic standard geographic / common abbreviations (used ONLY to enrich values actually observed in DB)
STANDARD_VALUE_ALIASES: Dict[str, List[str]] = {
    "CA": ["California", "Calif", "Canada"],
    "NY": ["New York"],
    "TX": ["Texas"],
    "FL": ["Florida"],
    "WA": ["Washington"],
    "IL": ["Illinois"],
    "MA": ["Massachusetts"],
    "US": ["United States", "USA", "America"],
    "USA": ["United States", "United States of America"],
    "UK": ["United Kingdom", "Britain", "Great Britain"],
    "GB": ["United Kingdom", "Great Britain"],
    "IN": ["India"],
    "DE": ["Germany"],
    "FR": ["France"],
    "JP": ["Japan"],
    "AU": ["Australia"],
    "PR": ["Present", "Attendance Present"],
    "AB": ["Absent"],
    "CL": ["Casual Leave"],
    "SL": ["Sick Leave"],
    "WO": ["Week Off", "Weekly Off"],
    "HD": ["Half Day"],
    "PH": ["Public Holiday"],
    "LWP": ["Loss of Pay"],
    "T1": ["Tier 1", "Tier-1"],
    "T2": ["Tier 2", "Tier-2"],
    "T3": ["Tier 3", "Tier-3"],
    "RJCT": ["Rejected", "Reject"],
    "APPR": ["Approved", "Approve"],
    "PEND": ["Pending"],
}

# Optional domain-specific synonym fallback (used ONLY when require_db_evidence=False for offline legacy compatibility)
COMMON_DOMAIN_SYNONYMS: Dict[str, Dict[str, str]] = {
    "staff_attendance": {
        "present": "PR",
        "pres": "PR",
        "absent": "AB",
        "casual leave": "CL",
        "sick leave": "SL",
        "week off": "WO",
        "weekly off": "WO",
        "half day": "HD",
        "public holiday": "PH",
        "loss of pay": "LWP",
    },
    "outlets": {
        "tier 1": "T1",
        "tier 2": "T2",
        "tier 3": "T3",
        "tier-1": "T1",
        "tier-2": "T2",
        "tier-3": "T3",
    },
    "group_loans": {
        "training": "Group Training",
        "group training": "Group Training",
        "payout": "Payout",
        "paid": "Payout",
        "disbursed": "Payout",
        "ready": "Ready",
        "enrolment": "Enrolment",
        "enrollment": "Enrolment",
    },
}

# Targeted categorical columns for legacy standalone postgres literal grounding
INDEXED_CATEGORICAL_COLUMNS = [
    ("outlets", "state_name"),
    ("outlets", "tier"),
    ("staff_attendance", "attendance_code"),
    ("group_loans", "stage"),
    ("group_loans", "origin_type"),
    ("staff", "position"),
]


class LiteralMatch(BaseModel):
    """Represents a grounded mapping from a natural language term to an exact database literal."""

    term: str = Field(description="The user's natural language word or phrase")
    table_name: str = Field(description="Target table where the literal belongs")
    column_name: str = Field(description="Target column in the table")
    database_literal: str = Field(description="Exact case-sensitive database cell literal")
    match_type: str = Field(description="'exact', 'case_insensitive', 'alias_semantic', 'synonym', or 'substring'")
    sql_predicate: str = Field(description="Safe SQL predicate, e.g. state = 'CA'")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Resolution confidence score in [0, 1]")
    evidence: str = Field(default="", description="Database evidence confirming the literal exists in storage")


def format_safe_sql_predicate(column_name: str, literal_value: str) -> str:
    """Safely formats `column = 'value'` with validated identifier and single-quote escaping."""
    clean_col = validate_sql_identifier(column_name)
    escaped = str(literal_value).replace("'", "''")
    return f"{clean_col} = '{escaped}'"


class LiteralResolver:
    """Distinct Value Indexer and Categorical Literal Resolver backed by DB evidence & ChromaDB."""

    def __init__(
        self,
        target: str = "postgres",
        settings: Optional[Settings] = None,
        collection_name: str = "distinct_values_v1",
        chroma_client: Any = None,
        embedding_function: Any = None,
        in_memory: bool = True,
        embedding_backend: str = "auto",
        duckdb_conn: Any = None,
    ):
        self.target = target.lower().strip()
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.duckdb_conn = duckdb_conn
        self._cache: Dict[Tuple[str, str], List[str]] = {}
        # Observed distinct values: {(table_name, column_name): [raw_val, ...]}
        self._observed_values: Dict[Tuple[str, str], List[str]] = {}
        self._custom_aliases: Dict[str, List[str]] = {}
        self._db_unavailable = False

        self._chroma_client_arg = chroma_client
        self._embedding_function_arg = embedding_function
        self._in_memory = in_memory
        self._embedding_backend = embedding_backend
        self._chroma_client: Any = None
        self._embedding_function: Any = None
        self.collection: Any = None

    def _ensure_vector_store(self) -> None:
        """Lazily initializes ChromaDB & BGE embeddings only when indexing/querying vectors."""
        if self.collection is not None:
            return
        self._chroma_client = self._chroma_client_arg or get_chroma_client(
            in_memory=self._in_memory, settings=self.settings
        )
        self._embedding_function = self._embedding_function_arg or get_embedding_function(
            settings=self.settings,
            backend=self._embedding_backend,
        )
        self.collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def register_aliases(self, literal_to_aliases: Dict[str, List[str]]) -> None:
        """Registers custom code-to-alias hints used strictly to describe observed DB values."""
        for raw_val, aliases in literal_to_aliases.items():
            existing = self._custom_aliases.setdefault(str(raw_val), [])
            for a in aliases:
                if a not in existing:
                    existing.append(a)

    def get_distinct_column_literals(
        self,
        table_name: str,
        column_name: str,
        limit: int = 50,
    ) -> List[str]:
        """Safely fetches distinct cell values using identifier validation and parameterized SQL."""
        clean_table = validate_sql_identifier(table_name)
        clean_col = validate_sql_identifier(column_name)
        clamped_limit = min(max(1, int(limit)), 500)

        cache_key = (clean_table, clean_col)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.target in ("local", "sqlite"):
            inspector = SchemaInspector(target="local", settings=self.settings)
            if clean_table not in inspector.get_table_names():
                return []
            valid_cols = {c["name"] for c in inspector.get_table_columns(clean_table)}
            if clean_col not in valid_cols:
                return []

            conn = get_local_db_connection(self.settings)
            cur = conn.cursor()
            sql = (
                f'SELECT DISTINCT "{clean_col}" FROM "{clean_table}" '
                f'WHERE "{clean_col}" IS NOT NULL ORDER BY "{clean_col}" LIMIT ?;'
            )
            cur.execute(sql, (clamped_limit,))
            values = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
            conn.close()
            self._cache[cache_key] = values
            return values

        if self.target == "duckdb":
            if self.duckdb_conn is None:
                return []
            inspector = SchemaInspector(target="duckdb", settings=self.settings, duckdb_conn=self.duckdb_conn)
            if clean_table not in inspector.get_table_names():
                return []
            valid_cols = {c["name"] for c in inspector.get_table_columns(clean_table)}
            if clean_col not in valid_cols:
                return []

            sql = (
                f'SELECT DISTINCT "{clean_col}" FROM "{clean_table}" '
                f'WHERE "{clean_col}" IS NOT NULL ORDER BY "{clean_col}" LIMIT ?;'
            )
            rows = self.duckdb_conn.execute(sql, [clamped_limit]).fetchall()
            values = [str(r[0]) for r in rows if r[0] is not None]
            self._cache[cache_key] = values
            return values

        # Live PostgreSQL
        if self._db_unavailable:
            return []
        try:
            inspector = SchemaInspector(target="postgres", settings=self.settings)
            if clean_table not in inspector.get_table_names():
                return []
            valid_cols = {c["name"] for c in inspector.get_table_columns(clean_table)}
            if clean_col not in valid_cols:
                return []

            sql = (
                f'SELECT DISTINCT "{clean_col}" FROM "{clean_table}" '
                f'WHERE "{clean_col}" IS NOT NULL ORDER BY "{clean_col}" LIMIT %s;'
            )
            res = execute_readonly_query(sql, params=(clamped_limit,), settings=self.settings)
            values = [str(r[clean_col]) for r in res.get("rows", []) if r.get(clean_col) is not None]
            self._cache[cache_key] = values
            return values
        except Exception as exc:
            logger.debug("PostgreSQL distinct literal lookup unavailable: %s", exc)
            self._db_unavailable = True
            return []

    def index_column_values(
        self,
        table_name: str,
        column_name: str,
        distinct_values: List[str],
        aliases: Optional[Dict[str, List[str]]] = None,
    ) -> int:
        """Indexes observed distinct database values into ChromaDB with optional alias enrichment."""
        clean_table = validate_sql_identifier(table_name.split(".")[-1])
        clean_col = validate_sql_identifier(column_name)
        if not distinct_values:
            return 0

        key = (clean_table, clean_col)
        self._observed_values[key] = [str(v) for v in distinct_values]
        self._cache[key] = [str(v) for v in distinct_values]

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for raw_val in distinct_values:
            val_str = str(raw_val)
            alias_list: List[str] = []
            for source_map in (STANDARD_VALUE_ALIASES, self._custom_aliases, aliases or {}):
                for a in source_map.get(val_str, []):
                    if a not in alias_list:
                        alias_list.append(a)
                for a in source_map.get(val_str.upper(), []):
                    if a not in alias_list:
                        alias_list.append(a)

            alias_text = f" | Aliases: {', '.join(alias_list)}" if alias_list else ""
            doc_text = f"Database literal '{val_str}' in {clean_table}.{clean_col}{alias_text}"
            predicate = format_safe_sql_predicate(clean_col, val_str)

            ids.append(f"{clean_table}.{clean_col}::{val_str}")
            documents.append(doc_text)
            metadatas.append(
                {
                    "table_name": clean_table,
                    "column_name": clean_col,
                    "raw_literal": val_str,
                    "aliases": "|".join(alias_list),
                    "sql_predicate": predicate,
                    "evidence": f"Observed distinct value '{val_str}' in {clean_table}.{clean_col}",
                }
            )

        if ids:
            self._ensure_vector_store()
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def build_index_from_database(
        self,
        tables: Optional[List[TableMetadata]] = None,
        limit_per_column: int = 50,
    ) -> int:
        """Dynamically indexes distinct values for text/categorical columns across tables."""
        inspector = SchemaInspector(
            target=self.target,
            settings=self.settings,
            duckdb_conn=self.duckdb_conn,
        )
        table_metas = tables or inspector.inspect_all_tables(include_samples=False)
        total_indexed = 0

        text_types = ("TEXT", "VARCHAR", "CHAR", "CHARACTER VARYING", "STRING")
        for tbl in table_metas:
            for col in tbl.columns:
                if col.is_primary_key:
                    continue
                upper_t = col.data_type.upper()
                if any(tt in upper_t for tt in text_types):
                    # Skip likely free-text names/descriptions unless explicitly categorical if short
                    vals = self.get_distinct_column_literals(tbl.table_name, col.name, limit=limit_per_column)
                    # Heuristic: categorical columns generally have concise values
                    concise_vals = [v for v in vals if 0 < len(v) <= 50]
                    if concise_vals:
                        total_indexed += self.index_column_values(tbl.table_name, col.name, concise_vals)
        return total_indexed

    def clear_index(self) -> None:
        """Clears cached literals and Chroma distinct value collection."""
        self._cache.clear()
        self._observed_values.clear()
        if self._chroma_client is not None:
            try:
                self._chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = None
            self._ensure_vector_store()

    def refresh_index(
        self,
        tables: Optional[List[TableMetadata]] = None,
        limit_per_column: int = 50,
    ) -> int:
        """Clears stale literal indexes and re-indexes distinct values from live storage."""
        self.clear_index()
        return self.build_index_from_database(tables=tables, limit_per_column=limit_per_column)

    def resolve_from_index(
        self,
        user_text: str,
        candidate_tables: Optional[List[str]] = None,
        top_k: int = 8,
        min_confidence: float = 0.78,
    ) -> List[LiteralMatch]:
        """Resolves query terms strictly against DB-observed values in the index."""
        if not user_text or not user_text.strip() or not self._observed_values:
            return []

        q_lower = user_text.lower()
        matches: List[LiteralMatch] = []
        matched_keys: Set[Tuple[str, str, str]] = set()

        allowed_tables: Optional[Set[str]] = None
        if candidate_tables:
            allowed_tables = {t.split(".")[-1] for t in candidate_tables}

        # Pass 1: Direct DB value & DB-evidenced alias matching (highest confidence)
        for (tbl, col), db_vals in self._observed_values.items():
            if allowed_tables is not None and tbl not in allowed_tables:
                continue
            for val in db_vals:
                val_lower = val.lower()
                # Exact or case-insensitive word-boundary match on stored DB value.
                # Require exact case for <=2 char codes so 'IN'/'OR'/'ID'/'AS' don't match English prepositions.
                is_short_code = len(val_lower) <= 2
                direct_hit = (
                    len(val_lower) >= 2
                    and (
                        bool(re.search(r"\b" + re.escape(val) + r"\b", user_text))
                        if is_short_code
                        else bool(re.search(r"\b" + re.escape(val_lower) + r"\b", q_lower))
                    )
                )
                if direct_hit:
                    key = (tbl, col, val)
                    if key not in matched_keys:
                        matched_keys.add(key)
                        is_exact = val in user_text
                        matches.append(
                            LiteralMatch(
                                term=val,
                                table_name=tbl,
                                column_name=col,
                                database_literal=val,
                                match_type="exact" if is_exact else "case_insensitive",
                                sql_predicate=format_safe_sql_predicate(col, val),
                                confidence=1.0 if is_exact else 0.96,
                                evidence=f"Observed '{val}' in {tbl}.{col} ({len(db_vals)} distinct values sampled)",
                            )
                        )
                    continue

                # Alias check strictly gated on DB value existence
                candidate_aliases: List[str] = []
                for source_map in (STANDARD_VALUE_ALIASES, self._custom_aliases):
                    candidate_aliases.extend(source_map.get(val, []))
                    candidate_aliases.extend(source_map.get(val.upper(), []))

                for alias in candidate_aliases:
                    if re.search(r"\b" + re.escape(alias.lower()) + r"\b", q_lower):
                        key = (tbl, col, val)
                        if key not in matched_keys:
                            matched_keys.add(key)
                            matches.append(
                                LiteralMatch(
                                    term=alias,
                                    table_name=tbl,
                                    column_name=col,
                                    database_literal=val,
                                    match_type="alias_semantic",
                                    sql_predicate=format_safe_sql_predicate(col, val),
                                    confidence=0.92,
                                    evidence=f"Alias '{alias}' matched DB-observed literal '{val}' in {tbl}.{col}",
                                )
                            )
                        break

        # Pass 2: Semantic Chroma Vector Retrieval over indexed DB values
        total_docs = sum(len(v) for v in self._observed_values.values())
        if total_docs > 0:
            query_kwargs: Dict[str, Any] = {
                "query_texts": [user_text],
                "n_results": min(top_k, total_docs),
                "include": ["metadatas", "distances"],
            }
            if allowed_tables:
                t_list = sorted(allowed_tables)
                query_kwargs["where"] = (
                    {"table_name": t_list[0]} if len(t_list) == 1 else {"table_name": {"$in": t_list}}
                )

            try:
                self._ensure_vector_store()
                res = self.collection.query(**query_kwargs)
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]
                for meta, dist in zip(metas, dists):
                    if not meta:
                        continue
                    sim = max(0.0, min(1.0, 1.0 - float(dist)))
                    if sim < min_confidence:
                        continue
                    tbl = meta["table_name"]
                    col = meta["column_name"]
                    val = meta["raw_literal"]
                    key = (tbl, col, val)
                    if key not in matched_keys:
                        matched_keys.add(key)
                        matches.append(
                            LiteralMatch(
                                term=user_text,
                                table_name=tbl,
                                column_name=col,
                                database_literal=val,
                                match_type="alias_semantic",
                                sql_predicate=meta["sql_predicate"],
                                confidence=round(sim, 4),
                                evidence=f"{meta.get('evidence', '')} (vector similarity={sim:.3f})",
                            )
                        )
            except Exception as exc:
                logger.debug("Chroma literal query skipped: %s", exc)

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def resolve_literals(
        self,
        user_text: str,
        candidate_tables: Optional[List[str]] = None,
        require_db_evidence: bool = False,
    ) -> List[LiteralMatch]:
        """Scans user text and returns grounded database literals with confidence and evidence."""
        q_lower = user_text.lower()
        matches: List[LiteralMatch] = []
        matched_keys: Set[Tuple[str, str, str]] = set()

        # 1. Check indexed DB-evidenced distinct values first
        for m in self.resolve_from_index(user_text, candidate_tables=candidate_tables):
            key = (m.table_name, m.column_name, m.database_literal)
            if key not in matched_keys:
                matched_keys.add(key)
                matches.append(m)

        if require_db_evidence:
            return matches

        # 2. Check Live Database Distinct Cell Values for configured categorical columns (backward compatibility)
        for table, col in INDEXED_CATEGORICAL_COLUMNS:
            if candidate_tables and table not in {t.split(".")[-1] for t in candidate_tables}:
                continue
            try:
                db_values = self.get_distinct_column_literals(table, col)
            except Exception:
                continue

            for val in db_values:
                val_lower = val.lower()
                if re.search(r"\b" + re.escape(val_lower) + r"\b", q_lower):
                    key = (table, col, val)
                    if key not in matched_keys:
                        matched_keys.add(key)
                        matches.append(
                            LiteralMatch(
                                term=val,
                                table_name=table,
                                column_name=col,
                                database_literal=val,
                                match_type="exact" if val in user_text else "case_insensitive",
                                sql_predicate=format_safe_sql_predicate(col, val),
                                confidence=1.0 if val in user_text else 0.95,
                                evidence=f"Observed '{val}' directly in {table}.{col}",
                            )
                        )

        # 3. Legacy Domain Synonyms fallback (when offline/unconnected for Problem 1 regression tests)
        for table, synonym_dict in COMMON_DOMAIN_SYNONYMS.items():
            if candidate_tables and table not in {t.split(".")[-1] for t in candidate_tables}:
                continue
            for phrase, literal in synonym_dict.items():
                if re.search(r"\b" + re.escape(phrase) + r"\b", q_lower):
                    col = "attendance_code" if table == "staff_attendance" else ("tier" if table == "outlets" else "stage")
                    key = (table, col, literal)
                    if key not in matched_keys:
                        matched_keys.add(key)
                        matches.append(
                            LiteralMatch(
                                term=phrase,
                                table_name=table,
                                column_name=col,
                                database_literal=literal,
                                match_type="synonym",
                                sql_predicate=format_safe_sql_predicate(col, literal),
                                confidence=0.80,
                                evidence="Domain synonym mapping (offline fallback)",
                            )
                        )

        return matches

    def generate_grounding_prompt_hint(
        self,
        user_text: str,
        candidate_tables: Optional[List[str]] = None,
        require_db_evidence: bool = False,
    ) -> str:
        """Produces a formatted prompt hint to guide LLMs with exact database literals."""
        matches = self.resolve_literals(
            user_text,
            candidate_tables=candidate_tables,
            require_db_evidence=require_db_evidence,
        )
        if not matches:
            return ""

        lines = [
            "[Discovered Database Literals (Schema Grounding)]:",
            "The user's query references domain terms that map to specific database cell literals:",
        ]
        for m in matches:
            lines.append(
                f"  - '{m.term}' -> `{m.table_name}.{m.column_name}`, "
                f"filter with: `{m.sql_predicate}` (confidence={m.confidence:.2f}, evidence: {m.evidence})"
            )
        lines.append("Use these exact literals in SQL WHERE clauses instead of inventing literal values.")
        return "\n".join(lines)
