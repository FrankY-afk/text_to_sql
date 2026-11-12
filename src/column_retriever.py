"""Stage 2: Fine-Grained Column & Nested Field Retrieval using BGE Embeddings & ChromaDB.

Searches columns and flattened nested subfields strictly within the candidate tables
selected by Stage 1 TableRetriever. Prunes irrelevant columns from wide enterprise tables
while preserving Primary Keys and Foreign Keys required for relational joins.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from src.config import Settings, get_chroma_client, get_embedding_function, get_settings
from src.schema_flattener import FlattenedField, SchemaFlattener
from src.schema_inspector import ColumnMetadata, TableMetadata
from src.table_retriever import _tokenize_identifier

logger = logging.getLogger(__name__)


class ColumnCandidate(BaseModel):
    """A ranked column or flattened nested field returned by Stage 2 Retrieval."""

    schema_name: str
    table_name: str
    full_table_name: str
    column_name: str = Field(..., description="Physical root column name")
    path: str = Field(..., description="Column name or flattened dot-path (e.g. event_params.value.int_value)")
    data_type: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_target: Optional[str] = None
    comment: Optional[str] = None
    sample_values: List[str] = Field(default_factory=list)
    is_nested: bool = False
    is_array: bool = False
    is_json: bool = False
    is_directly_queryable: bool = True
    dialect_access_hint: Optional[str] = None
    unnest_syntax: Optional[str] = None
    preserved_as_key: bool = Field(
        default=False,
        description="True if included automatically to preserve PK/FK join connectivity",
    )


class ColumnRetriever:
    """Fine-grained Stage-2 column and subfield retriever backed by BGE & ChromaDB."""

    def __init__(
        self,
        collection_name: str = "schema_columns_v1",
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
        # Catalog keyed by f"{full_table_name}::{path}"
        self._column_catalog: Dict[str, ColumnCandidate] = {}
        self._table_to_entries: Dict[str, List[ColumnCandidate]] = {}
        self._init_collection()

    def _init_collection(self) -> None:
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def build_column_document(
        table_meta: TableMetadata,
        col: ColumnMetadata,
        flattened: Optional[FlattenedField] = None,
    ) -> str:
        """Constructs a rich embedding document for a column or nested field."""
        path = flattened.path if flattened else col.name
        dtype = flattened.data_type if flattened else col.data_type
        readable_path = _tokenize_identifier(path)
        readable_tbl = _tokenize_identifier(table_meta.table_name)

        role_parts: List[str] = []
        if col.is_primary_key:
            role_parts.append("Primary Key")
        if col.is_foreign_key and col.foreign_key_target:
            role_parts.append(f"Foreign Key referencing {col.foreign_key_target}")
        if flattened and flattened.is_nested:
            role_parts.append("Nested Field")
        if flattened and flattened.is_array:
            role_parts.append("Array Field")

        lines = [
            f"Column: {path} ({readable_path}) in table {table_meta.full_name} ({readable_tbl})",
            f"Data Type: {dtype}",
        ]
        if role_parts:
            lines.append(f"Role: {', '.join(role_parts)}")
        if col.comment or (flattened and flattened.comment):
            lines.append(f"Description: {col.comment or flattened.comment}")
        if col.sample_values:
            lines.append(f"Sample Values: {', '.join(col.sample_values[:5])}")
        return "\n".join(lines)

    def clear_index(self) -> None:
        """Clears the column Chroma collection and internal catalog."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._column_catalog.clear()
        self._table_to_entries.clear()
        self._init_collection()

    def index_tables(
        self,
        tables: List[TableMetadata],
        flattened_by_table: Optional[Dict[str, List[FlattenedField]]] = None,
        dialect: str = "postgres",
        clear_existing: bool = False,
    ) -> int:
        """Indexes all columns and flattened nested subfields for the given tables."""
        if clear_existing:
            self.clear_index()

        if not tables:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for tbl in tables:
            col_lookup: Dict[str, ColumnMetadata] = {c.name: c for c in tbl.columns}
            tbl_flattened = (flattened_by_table or {}).get(tbl.full_name) or (flattened_by_table or {}).get(tbl.table_name)

            # Auto-flatten nested STRUCT/ARRAY types if caller didn't pre-flatten
            if tbl_flattened is None:
                tbl_flattened = []
                for c in tbl.columns:
                    upper_t = c.data_type.upper()
                    if any(tok in upper_t for tok in ("STRUCT", "ARRAY", "[]", "JSON", "JSONB")):
                        parsed = SchemaFlattener.parse_type_string(
                            table_name=tbl.full_name,
                            root_column=c.name,
                            current_path=c.name,
                            type_str=c.data_type,
                            dialect=dialect,
                            comment=c.comment,
                        )
                        tbl_flattened.extend(parsed)

            flat_by_root: Dict[str, List[FlattenedField]] = {}
            for ff in tbl_flattened:
                flat_by_root.setdefault(ff.root_column, []).append(ff)

            table_entries: List[ColumnCandidate] = []

            for c in tbl.columns:
                nested_children = flat_by_root.get(c.name)
                if nested_children and (len(nested_children) > 1 or nested_children[0].is_nested):
                    for ff in nested_children:
                        entry_id = f"{tbl.full_name}::{ff.path}"
                        candidate = ColumnCandidate(
                            schema_name=tbl.schema_name,
                            table_name=tbl.table_name,
                            full_table_name=tbl.full_name,
                            column_name=c.name,
                            path=ff.path,
                            data_type=ff.data_type,
                            similarity_score=0.0,
                            is_primary_key=c.is_primary_key,
                            is_foreign_key=c.is_foreign_key,
                            foreign_key_target=c.foreign_key_target,
                            comment=ff.comment or c.comment,
                            sample_values=c.sample_values[:3],
                            is_nested=ff.is_nested,
                            is_array=ff.is_array,
                            is_json=ff.is_json,
                            is_directly_queryable=ff.is_directly_queryable,
                            dialect_access_hint=ff.dialect_access_hint,
                            unnest_syntax=ff.unnest_syntax,
                        )
                        self._column_catalog[entry_id] = candidate
                        table_entries.append(candidate)
                        ids.append(entry_id)
                        documents.append(self.build_column_document(tbl, c, flattened=ff))
                        metadatas.append(
                            {
                                "schema_name": tbl.schema_name,
                                "table_name": tbl.table_name,
                                "full_table_name": tbl.full_name,
                                "column_name": c.name,
                                "path": ff.path,
                                "is_key": bool(c.is_primary_key or c.is_foreign_key),
                            }
                        )
                else:
                    entry_id = f"{tbl.full_name}::{c.name}"
                    candidate = ColumnCandidate(
                        schema_name=tbl.schema_name,
                        table_name=tbl.table_name,
                        full_table_name=tbl.full_name,
                        column_name=c.name,
                        path=c.name,
                        data_type=c.data_type,
                        similarity_score=0.0,
                        is_primary_key=c.is_primary_key,
                        is_foreign_key=c.is_foreign_key,
                        foreign_key_target=c.foreign_key_target,
                        comment=c.comment,
                        sample_values=c.sample_values[:3],
                    )
                    self._column_catalog[entry_id] = candidate
                    table_entries.append(candidate)
                    ids.append(entry_id)
                    documents.append(self.build_column_document(tbl, c))
                    metadatas.append(
                        {
                            "schema_name": tbl.schema_name,
                            "table_name": tbl.table_name,
                            "full_table_name": tbl.full_name,
                            "column_name": c.name,
                            "path": c.name,
                            "is_key": bool(c.is_primary_key or c.is_foreign_key),
                        }
                    )

            self._table_to_entries[tbl.full_name] = table_entries

        if ids:
            # Upsert in batches to stay safe on large 1000+ column schemas
            batch_size = 200
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                self.collection.upsert(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )

        return len(ids)

    def refresh_index(
        self,
        tables: List[TableMetadata],
        flattened_by_table: Optional[Dict[str, List[FlattenedField]]] = None,
        dialect: str = "postgres",
    ) -> int:
        """Clears and rebuilds the fine-grained column index."""
        return self.index_tables(
            tables=tables,
            flattened_by_table=flattened_by_table,
            dialect=dialect,
            clear_existing=True,
        )

    def _resolve_candidate_full_names(self, candidate_tables: List[str]) -> List[str]:
        """Normalizes short or schema-qualified table names to indexed full_table_name keys."""
        resolved: List[str] = []
        for cand in candidate_tables:
            if cand in self._table_to_entries:
                resolved.append(cand)
            else:
                for full_name, entries in self._table_to_entries.items():
                    if entries and entries[0].table_name == cand and full_name not in resolved:
                        resolved.append(full_name)
        return resolved

    def retrieve_columns(
        self,
        query: str,
        candidate_tables: List[str],
        top_k_per_table: Optional[int] = None,
        top_k_total: Optional[int] = None,
        preserve_keys: bool = True,
        min_score: float = 0.0,
    ) -> List[ColumnCandidate]:
        """Ranks and retrieves relevant columns strictly within candidate_tables.

        Preserves PK and FK columns of candidate tables when `preserve_keys=True` so
        downstream CTE/SQL generation retains valid join keys without sending every column.
        """
        if not query or not query.strip() or not candidate_tables or not self._column_catalog:
            return []

        target_full_names = self._resolve_candidate_full_names(candidate_tables)
        if not target_full_names:
            return []

        k_per_tbl = top_k_per_table or self.settings.schema_top_k_columns
        candidate_pool_size = sum(len(self._table_to_entries.get(t, [])) for t in target_full_names)
        if candidate_pool_size == 0:
            return []

        n_results = min(candidate_pool_size, max(len(target_full_names) * k_per_tbl * 2, 25))
        where_filter: Dict[str, Any] = (
            {"full_table_name": target_full_names[0]}
            if len(target_full_names) == 1
            else {"full_table_name": {"$in": target_full_names}}
        )

        raw_res = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["metadatas", "distances"],
        )
        ids_list = raw_res.get("ids", [[]])[0]
        distances_list = raw_res.get("distances", [[]])[0]

        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        per_table_selected: Dict[str, List[ColumnCandidate]] = {t: [] for t in target_full_names}
        selected_ids: Set[str] = set()

        for entry_id, dist in zip(ids_list, distances_list):
            base_entry = self._column_catalog.get(entry_id)
            if base_entry is None:
                continue

            cosine_sim = max(0.0, min(1.0, 1.0 - float(dist)))
            col_tokens = set(re.findall(r"[a-z0-9]+", _tokenize_identifier(base_entry.path)))
            sample_tokens: Set[str] = set()
            for sv in base_entry.sample_values:
                sample_tokens.update(re.findall(r"[a-z0-9]+", sv.lower()))

            col_overlap = len(q_tokens & col_tokens) / max(len(col_tokens), 1)
            sample_overlap = len(q_tokens & sample_tokens) / max(len(q_tokens), 1)
            lexical_bonus = min(0.18, (0.12 * col_overlap) + (0.06 * sample_overlap))
            final_score = round(min(1.0, cosine_sim + lexical_bonus), 4)

            if final_score < min_score:
                continue

            tbl_bucket = per_table_selected.get(base_entry.full_table_name)
            if tbl_bucket is not None and len(tbl_bucket) < k_per_tbl:
                scored_copy = base_entry.model_copy(update={"similarity_score": final_score})
                tbl_bucket.append(scored_copy)
                selected_ids.add(entry_id)

        # Preserve Primary Keys and Foreign Keys for each candidate table
        combined: List[ColumnCandidate] = []
        for tbl_full in target_full_names:
            bucket = per_table_selected.get(tbl_full, [])
            bucket.sort(key=lambda c: c.similarity_score, reverse=True)
            combined.extend(bucket)

            if preserve_keys:
                for entry in self._table_to_entries.get(tbl_full, []):
                    eid = f"{entry.full_table_name}::{entry.path}"
                    if (entry.is_primary_key or entry.is_foreign_key) and eid not in selected_ids:
                        selected_ids.add(eid)
                        combined.append(
                            entry.model_copy(
                                update={"similarity_score": 0.5, "preserved_as_key": True}
                            )
                        )

        # Final deterministic ordering: non-preserved ranked descending, plus preserved keys grouped cleanly
        combined.sort(key=lambda c: (not c.preserved_as_key, c.similarity_score), reverse=True)
        if top_k_total is not None:
            # Ensure preserved keys aren't evicted before top_k_total if possible
            keys = [c for c in combined if c.preserved_as_key]
            non_keys = [c for c in combined if not c.preserved_as_key]
            remaining_slots = max(0, top_k_total - len(keys))
            combined = non_keys[:remaining_slots] + keys[:top_k_total]
            combined.sort(key=lambda c: c.similarity_score, reverse=True)

        return combined

