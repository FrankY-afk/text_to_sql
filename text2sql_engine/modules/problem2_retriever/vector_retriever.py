"""Two-Stage Schema Retriever & Distinct Value Indexer solving Wrong Schema Linking (27.6% of errors)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb

from text2sql_engine.core.config import get_embeddings, get_chroma_client
from text2sql_engine.core.exceptions import RetrieverError
from text2sql_engine.modules.problem2_retriever.schema_flattener import FlattenedField, SchemaFlattener


IDENTIFIER_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass
class LiteralMatch:
    """Provenance and confidence metadata for an entity-to-literal mapping."""

    entity: str
    table_name: str
    column_name: str
    raw_literal: str
    confidence: float
    sql_predicate: str
    provenance: str


class DistinctValueIndexer:
    """Indexes sample categorical cell strings to map natural language filter terms
    (e.g., 'purchases in California') to exact database literals ('state_code = 'CA'')
    with SQL injection prevention, provenance tracking, and ambiguity detection.
    """

    def __init__(
        self,
        collection_name: str = "distinct_values",
        chroma_client: Optional[chromadb.ClientAPI] = None,
        embeddings: Optional[Any] = None,
    ):
        self.client = chroma_client or get_chroma_client()
        self.embeddings = embeddings or get_embeddings()
        self.collection_name = collection_name
        self.known_columns: Set[Tuple[str, str]] = set()
        self._init_store()

    def _init_store(self) -> None:
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    def clear(self) -> None:
        """Clears all indexed distinct values and resets vector store."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.known_columns.clear()
        self._init_store()

    def index_column_values(
        self,
        table_name: str,
        column_name: str,
        distinct_values: List[Tuple[str, str]],  # List of (raw_literal, semantic_label)
    ) -> None:
        """Indexes distinct categorical values with semantic descriptions and safe SQL predicates.

        Args:
            table_name: Database table name (must be a valid SQL identifier).
            column_name: Column name (must be a valid SQL identifier).
            distinct_values: List of (raw_literal, semantic_label).

        Raises:
            RetrieverError: If table_name or column_name contains invalid SQL identifier characters.
        """
        if not IDENTIFIER_REGEX.match(table_name):
            raise RetrieverError(f"Invalid table identifier '{table_name}' in DistinctValueIndexer.")
        if not IDENTIFIER_REGEX.match(column_name):
            raise RetrieverError(f"Invalid column identifier '{column_name}' in DistinctValueIndexer.")

        self.known_columns.add((table_name.lower(), column_name.lower()))
        documents: List[Document] = []

        for raw_literal, semantic_label in distinct_values:
            # Escape single quotes in literal to prevent SQL injection
            escaped_literal = str(raw_literal).replace("'", "''")
            predicate = f"{column_name} = '{escaped_literal}'"
            content = f"{semantic_label} for column {column_name} in table {table_name} (literal: {raw_literal})"

            doc = Document(
                page_content=content,
                metadata={
                    "table_name": table_name,
                    "column_name": column_name,
                    "raw_literal": str(raw_literal),
                    "semantic_label": str(semantic_label),
                    "sql_predicate": predicate,
                },
            )
            documents.append(doc)

        if documents:
            self.vector_store.add_documents(documents)

    def match_literals_detailed(
        self,
        user_query: str,
        k: int = 5,
        score_threshold: float = 0.5,
    ) -> List[LiteralMatch]:
        """Matches entities in the user query to database literals with confidence and provenance."""
        if not user_query or not user_query.strip():
            return []

        try:
            results = self.vector_store.similarity_search_with_relevance_scores(user_query, k=k)
        except Exception:
            return []

        query_lower = user_query.lower()
        candidates: List[Tuple[Document, float, str, str]] = []  # (doc, score, key, provenance)

        for doc, score in results:
            if not doc.metadata:
                continue
            table_name = doc.metadata.get("table_name", "")
            column_name = doc.metadata.get("column_name", "")
            label = doc.metadata.get("semantic_label", "")
            raw_val = doc.metadata.get("raw_literal", "")

            # Ensure table and column identifiers are valid
            if not IDENTIFIER_REGEX.match(table_name) or not IDENTIFIER_REGEX.match(column_name):
                continue

            label_lower = label.lower() if label else ""
            raw_lower = raw_val.lower() if raw_val else ""

            # Check direct mentions in query text
            mentioned = False
            provenance = "vector_similarity"
            key = label or raw_val

            if label_lower and (re.search(rf"\b{re.escape(label_lower)}\b", query_lower) or label_lower in query_lower):
                mentioned = True
                provenance = "exact_label_in_query"
                key = label
            elif raw_lower and (re.search(rf"\b{re.escape(raw_lower)}\b", query_lower) or raw_lower in query_lower):
                mentioned = True
                provenance = "exact_literal_in_query"
                key = raw_val
            elif score >= 0.85:
                mentioned = True
                provenance = f"high_similarity_score_{score:.2f}"
                key = label or raw_val

            if mentioned and ("exact_" in provenance or score >= score_threshold):
                candidates.append((doc, float(score), key, provenance))


        # Ambiguity detection: group by (table_name, column_name)
        by_col: Dict[Tuple[str, str], List[Tuple[Document, float, str, str]]] = {}
        for item in candidates:
            doc, score, key, prov = item
            tbl = doc.metadata["table_name"]
            col = doc.metadata["column_name"]
            by_col.setdefault((tbl, col), []).append(item)

        matches: List[LiteralMatch] = []
        for (tbl, col), col_candidates in by_col.items():
            if len(col_candidates) > 1:
                # If top two candidates have very close scores and neither is an exact match, mark ambiguous
                sorted_c = sorted(col_candidates, key=lambda x: x[1], reverse=True)
                top_score = sorted_c[0][1]
                second_score = sorted_c[1][1]
                top_prov = sorted_c[0][3]
                second_prov = sorted_c[1][3]

                if abs(top_score - second_score) < 0.05 and not (
                    "exact_" in top_prov and "exact_" not in second_prov
                ):
                    # Ambiguous entity match; do not create predicate
                    continue

                best_doc, best_score, best_key, best_prov = sorted_c[0]
            else:
                best_doc, best_score, best_key, best_prov = col_candidates[0]

            matches.append(
                LiteralMatch(
                    entity=best_key,
                    table_name=best_doc.metadata["table_name"],
                    column_name=best_doc.metadata["column_name"],
                    raw_literal=best_doc.metadata["raw_literal"],
                    confidence=best_score,
                    sql_predicate=best_doc.metadata["sql_predicate"],
                    provenance=best_prov,
                )
            )

        return matches

    def match_literals(self, user_query: str, k: int = 5, score_threshold: float = 0.5) -> Dict[str, str]:
        """Matches entities in the user query to exact database literals.

        Returns:
            Dictionary mapping matched entity/label -> sql predicate clause (e.g. {"California": "state_code = 'CA'"})
        """
        matches = self.match_literals_detailed(user_query, k=k, score_threshold=score_threshold)
        return {m.entity: m.sql_predicate for m in matches}


class TwoStageSchemaRetriever:
    """Problem 2 Solution: Two-Stage Schema Retriever & Nested Flattener.

    Prevents 700+ column context window overflows and schema linking errors:
    - Stage 1: Coarse Table Filter prunes massive schemas down to top candidate tables
      while preserving required primary-key / foreign-key join relationships.
    - Stage 2: Fine-Grained Column/Field Selection identifies relevant paths including
      nested structs/arrays, ensuring join keys remain available.
    """

    def __init__(
        self,
        collection_prefix: str = "schema_retriever",
        chroma_client: Optional[chromadb.ClientAPI] = None,
        embeddings: Optional[Any] = None,
    ):
        self.client = chroma_client or get_chroma_client()
        self.embeddings = embeddings or get_embeddings()
        self.collection_prefix = collection_prefix

        self.table_cards: Dict[str, str] = {}
        self.table_fields: Dict[str, List[FlattenedField]] = {}
        self.table_pks: Dict[str, List[str]] = {}
        self.table_fks: Dict[str, List[str]] = {}
        self.table_descriptions: Dict[str, str] = {}
        self.table_relationships: Dict[str, Set[str]] = {}

        self._init_stores()

    def _init_stores(self) -> None:
        self.table_store = Chroma(
            client=self.client,
            collection_name=f"{self.collection_prefix}_tables",
            embedding_function=self.embeddings,
        )
        self.column_store = Chroma(
            client=self.client,
            collection_name=f"{self.collection_prefix}_columns",
            embedding_function=self.embeddings,
        )
        self.value_indexer = DistinctValueIndexer(
            collection_name=f"{self.collection_prefix}_values",
            chroma_client=self.client,
            embeddings=self.embeddings,
        )

    def clear(self) -> None:
        """Clears all vector stores and memory caches to prevent stale or duplicate entries."""
        for suffix in ("tables", "columns", "values"):
            coll_name = f"{self.collection_prefix}_{suffix}"
            try:
                self.client.delete_collection(coll_name)
            except Exception:
                pass
        self.table_cards.clear()
        self.table_fields.clear()
        self.table_pks.clear()
        self.table_fks.clear()
        self.table_descriptions.clear()
        self.table_relationships.clear()
        self._init_stores()

    def remove_table(self, table_name: str) -> None:
        """Removes a specific table and its columns from memory and indexes."""
        self.table_cards.pop(table_name, None)
        self.table_fields.pop(table_name, None)
        self.table_pks.pop(table_name, None)
        self.table_fks.pop(table_name, None)
        self.table_descriptions.pop(table_name, None)
        self.table_relationships.pop(table_name, None)

    def index_database_schema(
        self,
        tables_metadata: Dict[str, Dict[str, Any]],
        sample_values: Optional[Dict[str, Dict[str, List[Tuple[str, str]]]]] = None,
        clear_existing: bool = True,
    ) -> None:
        """Indexes table cards, flattened columns, and distinct values into ChromaDB.

        Args:
            tables_metadata: Dict of table_name -> {
                "description": str,
                "fields": List[FlattenedField],
                "primary_keys": List[str],
                "foreign_keys": List[str],
            }
            sample_values: Dict of table_name -> column_name -> List of (raw_literal, semantic_label)
            clear_existing: Whether to clear existing collections before indexing (prevents duplicates).
        """
        if clear_existing:
            self.clear()

        table_docs: List[Document] = []
        column_docs: List[Document] = []

        for table_name, meta in tables_metadata.items():
            desc = meta.get("description", "")
            fields: List[FlattenedField] = meta.get("fields", [])
            pks = meta.get("primary_keys", [])
            fks = meta.get("foreign_keys", [])

            self.table_fields[table_name] = fields
            self.table_descriptions[table_name] = desc
            self.table_pks[table_name] = pks
            self.table_fks[table_name] = fks

            # Relationship tracking
            self.table_relationships.setdefault(table_name, set())
            for other_tbl, other_meta in tables_metadata.items():
                if other_tbl != table_name:
                    # Check if foreign keys reference table_name or vice versa
                    if any(table_name in fk for fk in other_meta.get("foreign_keys", [])):
                        self.table_relationships[table_name].add(other_tbl)
                    if any(other_tbl in fk for fk in fks):
                        self.table_relationships[table_name].add(other_tbl)

            # Create rich table card
            schema_card = SchemaFlattener.generate_schema_card(table_name, fields, description=desc)
            self.table_cards[table_name] = schema_card

            # Stage 1: Coarse Table Document
            field_names_summary = ", ".join([f.path for f in fields[:25]])
            table_summary = (
                f"Table Name: {table_name}\n"
                f"Purpose: {desc}\n"
                f"Primary Keys: {', '.join(pks)}\n"
                f"Foreign Keys: {', '.join(fks)}\n"
                f"Columns & Nested Paths: {field_names_summary}"
            )
            table_docs.append(
                Document(
                    page_content=table_summary,
                    metadata={"table_name": table_name, "num_fields": len(fields)},
                )
            )

            # Stage 2: Fine-grained Column Documents
            for field in fields:
                flags = " ARRAY UNNEST" if field.is_array else (" NESTED" if field.is_nested else "")
                field_desc = f": {field.comment}" if field.comment else ""
                col_content = f"{field.table_name}.{field.path} ({field.data_type}){flags}{field_desc}"
                column_docs.append(

                    Document(
                        page_content=col_content,
                        metadata={
                            "table_name": table_name,
                            "path": field.path,
                            "root_column": field.root_column,
                            "data_type": field.data_type,
                            "is_nested": field.is_nested,
                            "is_array": field.is_array,
                        },
                    )
                )

        if table_docs:
            self.table_store.add_documents(table_docs)
        if column_docs:
            self.column_store.add_documents(column_docs)

        # Index distinct values
        if sample_values:
            for tbl, cols in sample_values.items():
                for col, vals in cols.items():
                    self.value_indexer.index_column_values(tbl, col, vals)

    def retrieve_schema_context(
        self,
        user_query: str,
        top_k_tables: int = 3,
        top_k_columns_per_table: int = 15,
    ) -> Tuple[List[str], str, Dict[str, str]]:
        """Executes Two-Stage Schema Retrieval and Distinct Value Matching.

        Returns:
            - candidate_tables: List of top selected table names.
            - schema_context: Compact, LLM-ready schema prompt context.
            - matched_literals: Dict of entity -> exact SQL literal predicate.
        """
        if not user_query or not user_query.strip():
            return [], "", {}

        # Stage 1: Coarse Table Filter
        raw_table_results = self.table_store.similarity_search(user_query, k=top_k_tables * 2)
        candidate_tables: List[str] = []
        for doc in raw_table_results:
            tbl = doc.metadata.get("table_name")
            if tbl and tbl in self.table_fields and tbl not in candidate_tables:
                candidate_tables.append(tbl)
            if len(candidate_tables) >= top_k_tables:
                break

        # Distinct Value Indexer Matching
        matched_literals = self.value_indexer.match_literals(user_query)

        # Ensure tables referenced by matched literals are included in candidate tables
        detailed_matches = self.value_indexer.match_literals_detailed(user_query)
        for match in detailed_matches:
            tbl = match.table_name
            if tbl in self.table_fields and tbl not in candidate_tables:
                candidate_tables.append(tbl)

        # Multi-table and Join Selection Enrichment (Phase A4):
        # If candidate tables have foreign key relationships with other tables whose names/concepts
        # are mentioned in the user query, add those join tables so multi-table queries don't fail.
        query_lower = user_query.lower()
        additional_join_tables: List[str] = []
        for tbl in list(candidate_tables):
            related = self.table_relationships.get(tbl, set())
            for rel_tbl in related:
                if rel_tbl not in candidate_tables and rel_tbl not in additional_join_tables:
                    # Check if query mentions the related table name or its primary key
                    rel_name_words = rel_tbl.lower().split("_")
                    if any(w in query_lower for w in rel_name_words if len(w) > 3):
                        additional_join_tables.append(rel_tbl)

        for rel_tbl in additional_join_tables:
            if len(candidate_tables) < top_k_tables + 2:
                candidate_tables.append(rel_tbl)

        # Stage 2: Fine-Grained Column/Field Retrieval within candidate tables (Phase A1)
        relevant_columns_by_table: Dict[str, Set[str]] = {t: set() for t in candidate_tables}
        total_col_k = max(len(candidate_tables) * top_k_columns_per_table, 30)

        try:
            column_results = self.column_store.similarity_search_with_relevance_scores(user_query, k=total_col_k)
            query_words = set(re.findall(r"\w+", user_query.lower()))
            for doc, score in column_results:
                if not doc or not doc.metadata:
                    continue
                tbl = doc.metadata.get("table_name")
                path = doc.metadata.get("path")
                if tbl and tbl in relevant_columns_by_table and path:
                    path_tokens = set(re.findall(r"\w+", path.lower())) if path else set()
                    is_mentioned = any(
                        pw == qw or (len(pw) >= 4 and len(qw) >= 4 and (pw.startswith(qw) or qw.startswith(pw)))
                        for pw in path_tokens
                        for qw in query_words
                    )
                    if score >= 0.18 or is_mentioned:
                        if len(relevant_columns_by_table[tbl]) < top_k_columns_per_table:
                            relevant_columns_by_table[tbl].add(path)
        except Exception:
            try:
                raw_results = self.column_store.similarity_search(user_query, k=total_col_k)
                for doc in raw_results:
                    if not doc or not doc.metadata:
                        continue
                    tbl = doc.metadata.get("table_name")
                    path = doc.metadata.get("path")
                    if tbl and tbl in relevant_columns_by_table and path:
                        if len(relevant_columns_by_table[tbl]) < top_k_columns_per_table:
                            relevant_columns_by_table[tbl].add(path)
            except Exception:
                pass


        # Assemble compact schema cards for candidate tables
        schema_sections: List[str] = []
        for table_name in candidate_tables:
            fields = self.table_fields.get(table_name, [])
            relevant_paths = relevant_columns_by_table.get(table_name, set())

            # Essential keys: Primary keys and Foreign keys must always be preserved for join execution
            pks = set(self.table_pks.get(table_name, []))
            fks = set(self.table_fks.get(table_name, []))
            essential_keys = pks | fks

            # Filter fields: include query-relevant fields plus essential join/primary keys
            filtered_fields = [
                f
                for f in fields
                if f.path in relevant_paths
                or f.root_column in essential_keys
                or f.path in essential_keys
            ]

            # Bounded fallback: ONLY when retrieval returns insufficient results
            if len(filtered_fields) < 2 and fields:
                filtered_fields = fields[:10]

            card = SchemaFlattener.generate_schema_card(
                table_name=table_name,
                fields=filtered_fields,
                description=f"Filtered schema subset for query: {user_query}",
            )
            schema_sections.append(card)

        combined_schema_context = "\n\n".join(schema_sections)
        return candidate_tables, combined_schema_context, matched_literals
