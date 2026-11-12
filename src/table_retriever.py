"""Stage 1: Coarse Table-Level Retrieval using BGE Embeddings & ChromaDB.

Identifies candidate database tables across multiple schemas given a natural-language
business question. Avoids sending full enterprise schemas (dozens/hundreds of tables)
to the LLM by retrieving only the top-k semantically relevant tables with similarity scores
and supporting dynamic schema refresh.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.config import Settings, get_chroma_client, get_embedding_function, get_settings
from src.schema_inspector import SchemaInspector, TableMetadata

logger = logging.getLogger(__name__)


class TableCandidate(BaseModel):
    """A candidate table returned by Stage 1 Coarse Table Retrieval."""

    schema_name: str
    table_name: str
    full_name: str
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Normalized cosine/hybrid similarity score in [0, 1]")
    description: Optional[str] = None
    column_summary: str = ""
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: List[str] = Field(default_factory=list)
    metadata: TableMetadata


def _tokenize_identifier(name: str) -> str:
    """Expands snake_case and camelCase identifiers into readable words for semantic search."""
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return re.sub(r"[_\-.]+", " ", s1).strip().lower()


class TableRetriever:
    """Coarse Stage-1 table retriever backed by BGE embeddings and ChromaDB."""

    def __init__(
        self,
        collection_name: str = "schema_tables_v1",
        chroma_client: Any = None,
        embedding_function: Any = None,
        in_memory: bool = False,
        settings: Optional[Settings] = None,
        embedding_backend: str = "auto",
    ):
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.client = chroma_client or get_chroma_client(in_memory=in_memory, settings=self.settings)
        self.embedding_function = embedding_function or get_embedding_function(
            settings=self.settings,
            backend=embedding_backend,
        )
        self._table_catalog: Dict[str, TableMetadata] = {}
        self._init_collection()

    def _init_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def build_table_document(meta: TableMetadata) -> str:
        """Builds a semantically rich embedding document for a table's metadata."""
        readable_table = _tokenize_identifier(meta.table_name)
        readable_schema = _tokenize_identifier(meta.schema_name)

        col_summaries: List[str] = []
        for col in meta.columns:
            col_words = _tokenize_identifier(col.name)
            parts = [f"{col.name} ({col_words}, {col.data_type})"]
            if col.comment:
                parts.append(f": {col.comment}")
            if col.sample_values:
                samples = ", ".join(col.sample_values[:3])
                parts.append(f" [samples: {samples}]")
            col_summaries.append("".join(parts))

        fk_summaries = [
            f"{fk.source_column} -> {fk.target_schema}.{fk.target_table}.{fk.target_column}"
            for fk in meta.foreign_keys
        ]

        lines = [
            f"Table: {meta.full_name} ({readable_schema} {readable_table})",
            f"Description: {meta.description}" if meta.description else f"Description: Records related to {readable_table} in {readable_schema}",
            f"Primary Keys: {', '.join(meta.primary_keys) if meta.primary_keys else 'None'}",
            f"Relationships: {'; '.join(fk_summaries) if fk_summaries else 'None'}",
            f"Columns: {'; '.join(col_summaries)}",
        ]
        return "\n".join(lines)

    def clear_index(self) -> None:
        """Deletes and recreates the underlying Chroma table collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._table_catalog.clear()
        self._init_collection()

    def index_tables(
        self,
        tables: List[TableMetadata],
        clear_existing: bool = False,
    ) -> int:
        """Indexes structured TableMetadata objects into ChromaDB."""
        if clear_existing:
            self.clear_index()

        if not tables:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for meta in tables:
            doc_id = meta.full_name
            self._table_catalog[doc_id] = meta

            col_names_str = ", ".join(c.name for c in meta.columns)
            fk_strs = [f"{fk.source_column}->{fk.target_table}.{fk.target_column}" for fk in meta.foreign_keys]

            ids.append(doc_id)
            documents.append(self.build_table_document(meta))
            metadatas.append(
                {
                    "schema_name": meta.schema_name,
                    "table_name": meta.table_name,
                    "full_name": meta.full_name,
                    "column_count": len(meta.columns),
                    "column_names": col_names_str[:500],
                    "foreign_keys": ", ".join(fk_strs)[:500],
                }
            )

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def index_from_inspector(
        self,
        inspector: SchemaInspector,
        schemas: Optional[List[str]] = None,
        clear_existing: bool = True,
        include_samples: bool = True,
    ) -> int:
        """Inspects tables via SchemaInspector (respecting MCP/DB boundaries) and indexes them."""
        tables = inspector.inspect_all_tables(schemas=schemas, include_samples=include_samples)
        return self.index_tables(tables, clear_existing=clear_existing)

    def refresh_index(
        self,
        inspector: Optional[SchemaInspector] = None,
        tables: Optional[List[TableMetadata]] = None,
        schemas: Optional[List[str]] = None,
        include_samples: bool = True,
    ) -> int:
        """Refreshes the table index when schemas change (added, altered, or dropped tables)."""
        self.clear_index()
        if tables is not None:
            return self.index_tables(tables, clear_existing=False)
        if inspector is not None:
            return self.index_from_inspector(
                inspector=inspector,
                schemas=schemas,
                clear_existing=False,
                include_samples=include_samples,
            )
        return 0

    def get_indexed_tables(self) -> List[str]:
        """Returns list of currently indexed full table names ('schema.table')."""
        return sorted(self._table_catalog.keys())

    def retrieve_tables(
        self,
        query: str,
        top_k: Optional[int] = None,
        schemas: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[TableCandidate]:
        """Retrieves the top-k relevant tables for a natural-language question with similarity scores."""
        if not query or not query.strip() or not self._table_catalog:
            return []

        k = top_k or self.settings.schema_top_k_tables
        total_available = len(self._table_catalog)
        if schemas:
            filtered_count = sum(1 for m in self._table_catalog.values() if m.schema_name in schemas)
            if filtered_count == 0:
                return []
            n_results = min(max(k * 2, k), filtered_count)
            where_filter: Optional[Dict[str, Any]] = (
                {"schema_name": schemas[0]} if len(schemas) == 1 else {"schema_name": {"$in": schemas}}
            )
        else:
            n_results = min(max(k * 2, k), total_available)
            where_filter = None

        query_kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        raw_res = self.collection.query(**query_kwargs)
        ids_list = raw_res.get("ids", [[]])[0]
        distances_list = raw_res.get("distances", [[]])[0]

        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored_candidates: List[TableCandidate] = []

        for doc_id, dist in zip(ids_list, distances_list):
            meta = self._table_catalog.get(doc_id)
            if meta is None:
                continue

            # Cosine distance in Chroma is (1 - cosine_similarity) in [0, 2]
            cosine_sim = max(0.0, min(1.0, 1.0 - float(dist)))

            # Lightweight lexical identifier tie-breaker if user explicitly mentions table/column tokens
            tbl_tokens = set(re.findall(r"[a-z0-9]+", _tokenize_identifier(meta.table_name)))
            col_tokens = set()
            for c in meta.columns:
                col_tokens.update(re.findall(r"[a-z0-9]+", _tokenize_identifier(c.name)))

            tbl_overlap = len(q_tokens & tbl_tokens) / max(len(tbl_tokens), 1)
            col_overlap = len(q_tokens & col_tokens) / max(len(q_tokens), 1)
            lexical_bonus = min(0.15, (0.10 * tbl_overlap) + (0.05 * col_overlap))
            final_score = round(min(1.0, cosine_sim + lexical_bonus), 4)

            if final_score < min_score:
                continue

            col_summary = ", ".join(f"{c.name} ({c.data_type})" for c in meta.columns[:12])
            if len(meta.columns) > 12:
                col_summary += f", ... (+{len(meta.columns) - 12} more)"

            fk_strs = [
                f"{fk.source_column} -> {fk.target_schema}.{fk.target_table}.{fk.target_column}"
                for fk in meta.foreign_keys
            ]

            scored_candidates.append(
                TableCandidate(
                    schema_name=meta.schema_name,
                    table_name=meta.table_name,
                    full_name=meta.full_name,
                    similarity_score=final_score,
                    description=meta.description,
                    column_summary=col_summary,
                    primary_keys=list(meta.primary_keys),
                    foreign_keys=fk_strs,
                    metadata=meta,
                )
            )

        scored_candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        return scored_candidates[:k]

