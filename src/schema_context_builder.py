"""Stage 5: Schema Context Builder & Two-Stage Enterprise Schema Linker.

Combines:
1. Stage 1 Coarse Table Retrieval (`TableRetriever`).
2. Stage 2 Fine-Grained Column & Nested Field Retrieval (`ColumnRetriever`).
3. Stage 3 Nested Schema Tree Metadata (`SchemaFlattener` dialect access & UNNEST hints).
4. Stage 4 Distinct Value Resolution (`LiteralResolver` with DB evidence & confidence).
5. Primary Key & Foreign Key relationship preservation across selected tables.

Produces a compact `SchemaContext` object and Markdown prompt block for the
Problem 1 CTE Logic Planner and Hugging Face SQL Generator, avoiding full-schema bloat.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.column_retriever import ColumnCandidate, ColumnRetriever
from src.config import Settings, get_chroma_client, get_embedding_function, get_settings
from src.literal_resolver import LiteralMatch, LiteralResolver
from src.schema_flattener import FlattenedField
from src.schema_inspector import ForeignKeyMetadata, SchemaInspector, TableMetadata
from src.table_retriever import TableCandidate, TableRetriever

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Fast character-based token estimate (~4 chars per token for SQL/schema text)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class TableContextSection(BaseModel):
    """Compact schema representation for a single retrieved candidate table."""

    schema_name: str
    table_name: str
    full_name: str
    similarity_score: float
    description: Optional[str] = None
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: List[str] = Field(default_factory=list)
    selected_columns: List[ColumnCandidate] = Field(default_factory=list)
    total_table_columns: int = 0


class SchemaContext(BaseModel):
    """Structured, pruned schema linking output passed to CTEPlanner and SQLGenerator."""

    query: str
    target_dialect: str = "postgres"
    candidate_tables: List[TableContextSection] = Field(default_factory=list)
    selected_table_names: List[str] = Field(default_factory=list)
    selected_column_paths: List[str] = Field(default_factory=list)
    join_relationships: List[str] = Field(default_factory=list)
    nested_field_hints: List[str] = Field(default_factory=list)
    resolved_literals: List[LiteralMatch] = Field(default_factory=list)
    rendered_markdown: str = ""
    retrieval_latency_ms: float = 0.0
    full_schema_token_estimate: int = 0
    pruned_schema_token_estimate: int = 0
    token_reduction_pct: float = 0.0


class SchemaContextBuilder:
    """Assembles compact, structured schema context from two-stage retrieval outputs."""

    @staticmethod
    def estimate_full_schema_tokens(all_tables: List[TableMetadata]) -> int:
        """Estimates token cost of dumping every table and column unfiltered."""
        if not all_tables:
            return 0
        lines: List[str] = []
        for tbl in all_tables:
            lines.append(f"### Table: {tbl.full_name}")
            if tbl.description:
                lines.append(f"Description: {tbl.description}")
            for c in tbl.columns:
                fk_str = f" FK->{c.foreign_key_target}" if c.foreign_key_target else ""
                pk_str = " PK" if c.is_primary_key else ""
                sample_str = f" samples={c.sample_values[:3]}" if c.sample_values else ""
                lines.append(f"  - {c.name}: {c.data_type}{pk_str}{fk_str}{sample_str}")
        return estimate_tokens("\n".join(lines))

    @classmethod
    def build(
        cls,
        query: str,
        table_candidates: List[TableCandidate],
        column_candidates: List[ColumnCandidate],
        resolved_literals: Optional[List[LiteralMatch]] = None,
        all_tables_metadata: Optional[List[TableMetadata]] = None,
        target_dialect: str = "postgres",
        retrieval_latency_ms: float = 0.0,
    ) -> SchemaContext:
        """Combines retrieved tables, columns, PK/FKs, nested hints, and literals into SchemaContext."""
        literals = resolved_literals or []
        cols_by_table: Dict[str, List[ColumnCandidate]] = {}
        for col in column_candidates:
            cols_by_table.setdefault(col.full_table_name, []).append(col)

        sections: List[TableContextSection] = []
        join_edges: List[str] = []
        seen_edges: Set[str] = set()
        nested_hints: List[str] = []
        seen_nested: Set[str] = set()
        selected_table_names: List[str] = []
        selected_column_paths: List[str] = []

        selected_full_set = {tc.full_name for tc in table_candidates}
        selected_short_set = {tc.table_name for tc in table_candidates}

        for tc in table_candidates:
            tbl_cols = cols_by_table.get(tc.full_name, [])
            # Deterministic fallback: ensure at least PKs/first few columns if column retrieval returned 0
            if not tbl_cols and tc.metadata.columns:
                for c in tc.metadata.columns[:4]:
                    tbl_cols.append(
                        ColumnCandidate(
                            schema_name=tc.schema_name,
                            table_name=tc.table_name,
                            full_table_name=tc.full_name,
                            column_name=c.name,
                            path=c.name,
                            data_type=c.data_type,
                            similarity_score=0.0,
                            is_primary_key=c.is_primary_key,
                            is_foreign_key=c.is_foreign_key,
                            foreign_key_target=c.foreign_key_target,
                            comment=c.comment,
                            sample_values=c.sample_values[:3],
                            preserved_as_key=c.is_primary_key or c.is_foreign_key,
                        )
                    )

            for fk in tc.metadata.foreign_keys:
                edge = (
                    f"{fk.source_schema}.{fk.source_table}.{fk.source_column} "
                    f"= {fk.target_schema}.{fk.target_table}.{fk.target_column}"
                )
                if (fk.target_table in selected_short_set or f"{fk.target_schema}.{fk.target_table}" in selected_full_set) and edge not in seen_edges:
                    seen_edges.add(edge)
                    join_edges.append(edge)
                elif edge not in seen_edges:
                    seen_edges.add(edge)
                    join_edges.append(edge)

            for col in tbl_cols:
                selected_column_paths.append(f"{tc.table_name}.{col.path}")
                if col.is_nested or col.is_array or col.is_json or not col.is_directly_queryable:
                    hint_text = (
                        f"- `{tc.table_name}.{col.path}` ({col.data_type}): "
                        f"{'NOT directly selectable; ' if not col.is_directly_queryable else ''}"
                        f"requires {target_dialect.upper()} access: `{col.dialect_access_hint or col.path}`"
                    )
                    if hint_text not in seen_nested:
                        seen_nested.add(hint_text)
                        nested_hints.append(hint_text)

            selected_table_names.append(tc.table_name)
            sections.append(
                TableContextSection(
                    schema_name=tc.schema_name,
                    table_name=tc.table_name,
                    full_name=tc.full_name,
                    similarity_score=tc.similarity_score,
                    description=tc.description,
                    primary_keys=list(tc.primary_keys),
                    foreign_keys=list(tc.foreign_keys),
                    selected_columns=tbl_cols,
                    total_table_columns=len(tc.metadata.columns),
                )
            )

        # Render compact Markdown prompt context
        md_lines: List[str] = [
            f"### Pruned Schema Context (Dialect: {target_dialect.upper()})",
            f"Selected {len(sections)} candidate table(s) and {len(selected_column_paths)} relevant field(s):",
            "",
        ]
        for sec in sections:
            md_lines.append(
                f"#### Table `{sec.full_name}` (relevance={sec.similarity_score:.2f}, "
                f"showing {len(sec.selected_columns)}/{sec.total_table_columns} columns)"
            )
            if sec.description:
                md_lines.append(f"- Purpose: {sec.description}")
            md_lines.append("| Column / Path | Type | Key / Role | Sample Values |")
            md_lines.append("|---|---|---|---|")
            for col in sec.selected_columns:
                roles: List[str] = []
                if col.is_primary_key:
                    roles.append("PK")
                if col.is_foreign_key and col.foreign_key_target:
                    roles.append(f"FK->{col.foreign_key_target}")
                if col.is_array:
                    roles.append("ARRAY(UNNEST)")
                elif col.is_json:
                    roles.append("JSON")
                elif col.is_nested:
                    roles.append("STRUCT")
                role_str = ", ".join(roles) if roles else "-"
                samples_str = ", ".join(f"'{s}'" for s in col.sample_values[:3]) if col.sample_values else "-"
                md_lines.append(f"| `{col.path}` | `{col.data_type}` | {role_str} | {samples_str} |")
            md_lines.append("")

        if join_edges:
            md_lines.append("**Foreign Key Join Paths:**")
            for edge in join_edges:
                md_lines.append(f"- `{edge}`")
            md_lines.append("")

        if nested_hints:
            md_lines.append(f"**Nested Field Access Rules ({target_dialect.upper()}):**")
            md_lines.extend(nested_hints)
            md_lines.append("")

        if literals:
            md_lines.append("**Resolved Categorical Literals (DB-Evidenced):**")
            for lit in literals:
                md_lines.append(
                    f"- '{lit.term}' -> `{lit.table_name}.{lit.column_name}`: `{lit.sql_predicate}` "
                    f"(confidence={lit.confidence:.2f}, evidence={lit.evidence})"
                )
            md_lines.append("")

        rendered_md = "\n".join(md_lines).strip()
        pruned_tokens = estimate_tokens(rendered_md)
        full_tokens = (
            cls.estimate_full_schema_tokens(all_tables_metadata)
            if all_tables_metadata
            else pruned_tokens
        )
        reduction_pct = (
            round(max(0.0, (full_tokens - pruned_tokens) * 100.0 / full_tokens), 2)
            if full_tokens > 0
            else 0.0
        )

        return SchemaContext(
            query=query,
            target_dialect=target_dialect,
            candidate_tables=sections,
            selected_table_names=selected_table_names,
            selected_column_paths=selected_column_paths,
            join_relationships=join_edges,
            nested_field_hints=nested_hints,
            resolved_literals=literals,
            rendered_markdown=rendered_md,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            full_schema_token_estimate=full_tokens,
            pruned_schema_token_estimate=pruned_tokens,
            token_reduction_pct=reduction_pct,
        )


class EnterpriseSchemaRetriever:
    """Unified Two-Stage Schema Retriever coordinating Tables, Columns, Nested Fields, and Literals."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        chroma_client: Any = None,
        embedding_function: Any = None,
        in_memory: bool = True,
        embedding_backend: str = "auto",
        collection_prefix: str = "enterprise_schema",
    ):
        self.settings = settings or get_settings()
        self.client = chroma_client or get_chroma_client(in_memory=in_memory, settings=self.settings)
        self.embedding_function = embedding_function or get_embedding_function(
            settings=self.settings,
            backend=embedding_backend,
        )
        self.table_retriever = TableRetriever(
            collection_name=f"{collection_prefix}_tables",
            chroma_client=self.client,
            embedding_function=self.embedding_function,
            settings=self.settings,
        )
        self.column_retriever = ColumnRetriever(
            collection_name=f"{collection_prefix}_columns",
            chroma_client=self.client,
            embedding_function=self.embedding_function,
            settings=self.settings,
        )
        self.literal_resolver = LiteralResolver(
            collection_name=f"{collection_prefix}_values",
            chroma_client=self.client,
            embedding_function=self.embedding_function,
            settings=self.settings,
        )
        self._all_tables_metadata: List[TableMetadata] = []

    def index_tables(
        self,
        tables: List[TableMetadata],
        flattened_by_table: Optional[Dict[str, List[FlattenedField]]] = None,
        dialect: str = "postgres",
        clear_existing: bool = True,
    ) -> Dict[str, int]:
        """Indexes tables, columns/nested fields, and observed sample categorical values."""
        self._all_tables_metadata = list(tables)
        n_tables = self.table_retriever.index_tables(tables, clear_existing=clear_existing)
        n_cols = self.column_retriever.index_tables(
            tables,
            flattened_by_table=flattened_by_table,
            dialect=dialect,
            clear_existing=clear_existing,
        )

        if clear_existing:
            self.literal_resolver.clear_index()

        # Index any observed categorical sample values attached to columns
        n_vals = 0
        for tbl in tables:
            for col in tbl.columns:
                if col.sample_values and not col.is_primary_key:
                    n_vals += self.literal_resolver.index_column_values(
                        tbl.table_name, col.name, col.sample_values
                    )

        return {"tables": n_tables, "columns": n_cols, "values": n_vals}

    def index_from_inspector(
        self,
        inspector: SchemaInspector,
        schemas: Optional[List[str]] = None,
        clear_existing: bool = True,
        index_distinct_values: bool = True,
    ) -> Dict[str, int]:
        """Inspects database via SchemaInspector and indexes all tables, columns, and distinct values."""
        dialect = "sqlite" if inspector.target in ("local", "sqlite") else inspector.target
        tables = inspector.inspect_all_tables(schemas=schemas, include_samples=True)
        counts = self.index_tables(tables, dialect=dialect, clear_existing=clear_existing)

        if index_distinct_values:
            self.literal_resolver.target = inspector.target
            self.literal_resolver.duckdb_conn = inspector.duckdb_conn
            counts["values"] += self.literal_resolver.build_index_from_database(tables=tables)
        return counts

    def refresh(
        self,
        inspector: Optional[SchemaInspector] = None,
        tables: Optional[List[TableMetadata]] = None,
        schemas: Optional[List[str]] = None,
        dialect: str = "postgres",
    ) -> Dict[str, int]:
        """Refreshes all schema indices when tables, columns, or categorical values change."""
        if tables is not None:
            return self.index_tables(tables, dialect=dialect, clear_existing=True)
        if inspector is not None:
            return self.index_from_inspector(inspector=inspector, schemas=schemas, clear_existing=True)
        self.table_retriever.clear_index()
        self.column_retriever.clear_index()
        self.literal_resolver.clear_index()
        self._all_tables_metadata = []
        return {"tables": 0, "columns": 0, "values": 0}

    def retrieve_schema_context(
        self,
        query: str,
        top_k_tables: Optional[int] = None,
        top_k_columns_per_table: Optional[int] = None,
        schemas: Optional[List[str]] = None,
        target_dialect: str = "postgres",
        require_db_evidence_for_literals: bool = True,
    ) -> SchemaContext:
        """Executes end-to-end Two-Stage Retrieval and returns compact SchemaContext."""
        t0 = time.perf_counter()

        # Stage 1: Coarse Table Retrieval
        table_candidates = self.table_retriever.retrieve_tables(
            query=query,
            top_k=top_k_tables,
            schemas=schemas,
        )
        candidate_full_names = [tc.full_name for tc in table_candidates]

        # Stage 4 (early boost): Check if resolved DB literals point to a candidate table missed by Stage 1
        literals = self.literal_resolver.resolve_literals(
            query,
            candidate_tables=None,
            require_db_evidence=require_db_evidence_for_literals,
        )
        indexed_catalog = {m.full_name: m for m in self._all_tables_metadata}
        for lit in literals:
            # Only promote missed tables when DB-evidenced literal match has high confidence
            if lit.confidence < 0.85:
                continue
            # Find matching TableMetadata if not yet in candidate tables
            for full_name, meta in indexed_catalog.items():
                if meta.table_name == lit.table_name and full_name not in candidate_full_names:
                    if schemas is None or meta.schema_name in schemas:
                        candidate_full_names.append(full_name)
                        table_candidates.append(
                            TableCandidate(
                                schema_name=meta.schema_name,
                                table_name=meta.table_name,
                                full_name=meta.full_name,
                                similarity_score=round(min(1.0, lit.confidence * 0.9), 4),
                                description=meta.description,
                                column_summary=", ".join(f"{c.name} ({c.data_type})" for c in meta.columns[:10]),
                                primary_keys=list(meta.primary_keys),
                                foreign_keys=[
                                    f"{fk.source_column} -> {fk.target_schema}.{fk.target_table}.{fk.target_column}"
                                    for fk in meta.foreign_keys
                                ],
                                metadata=meta,
                            )
                        )

        # Stage 2: Fine-Grained Column & Nested Field Retrieval within candidate tables
        column_candidates = self.column_retriever.retrieve_columns(
            query=query,
            candidate_tables=candidate_full_names,
            top_k_per_table=top_k_columns_per_table,
            preserve_keys=True,
        )

        # Filter literals strictly to selected candidate tables
        selected_short = {tc.table_name for tc in table_candidates}
        scoped_literals = [m for m in literals if m.table_name in selected_short]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return SchemaContextBuilder.build(
            query=query,
            table_candidates=table_candidates,
            column_candidates=column_candidates,
            resolved_literals=scoped_literals,
            all_tables_metadata=self._all_tables_metadata,
            target_dialect=target_dialect,
            retrieval_latency_ms=elapsed_ms,
        )

