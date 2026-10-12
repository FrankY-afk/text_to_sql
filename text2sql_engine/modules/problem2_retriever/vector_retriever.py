"""Two-Stage Schema Retriever & Distinct Value Indexer solving Wrong Schema Linking (27.6% of errors)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb

from text2sql_engine.core.config import get_embeddings, get_chroma_client
from text2sql_engine.modules.problem2_retriever.schema_flattener import FlattenedField, SchemaFlattener


class DistinctValueIndexer:
    """Indexes sample categorical cell strings to map natural language filter terms
    (e.g., 'purchases in California') to exact database literals ('state_code = 'CA'').
    """

    def __init__(
        self,
        collection_name: str = "distinct_values",
        chroma_client: Optional[chromadb.ClientAPI] = None,
    ):
        self.client = chroma_client or get_chroma_client()
        self.embeddings = get_embeddings()
        self.collection_name = collection_name
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    def index_column_values(
        self,
        table_name: str,
        column_name: str,
        distinct_values: List[Tuple[str, str]],  # List of (raw_literal, semantic_label)
    ) -> None:
        """Indexes distinct categorical values with semantic descriptions.
        
        Example:
            distinct_values = [
                ("CA", "California"),
                ("NY", "New York"),
                ("TX", "Texas"),
            ]
        """
        documents: List[Document] = []
        for raw_literal, semantic_label in distinct_values:
            # We index both semantic label and raw literal for high recall
            content = f"{semantic_label} for column {column_name} in table {table_name} (literal: {raw_literal})"
            predicate = f"{column_name} = '{raw_literal}'"
            doc = Document(
                page_content=content,
                metadata={
                    "table_name": table_name,
                    "column_name": column_name,
                    "raw_literal": raw_literal,
                    "semantic_label": semantic_label,
                    "sql_predicate": predicate,
                },
            )
            documents.append(doc)

        if documents:
            self.vector_store.add_documents(documents)

    def match_literals(self, user_query: str, k: int = 3, score_threshold: float = 0.5) -> Dict[str, str]:
        """Matches entities in the user query to exact database literals.
        
        Returns:
            Dictionary mapping matched entity/label -> sql predicate clause (e.g. {"California": "state_code = 'CA'"})
        """
        results = self.vector_store.similarity_search_with_relevance_scores(user_query, k=k)
        matched_predicates: Dict[str, str] = {}

        for doc, score in results:
            label = doc.metadata.get("semantic_label", "")
            raw_val = doc.metadata.get("raw_literal", "")
            predicate = doc.metadata.get("sql_predicate", "")
            col_name = doc.metadata.get("column_name", "")

            query_lower = user_query.lower()
            label_lower = label.lower() if label else ""
            raw_lower = raw_val.lower() if raw_val else ""

            # Check if mentioned directly in user query
            mentioned = False
            if label_lower and label_lower in query_lower:
                mentioned = True
                key = label
            elif raw_lower and raw_lower in query_lower:
                mentioned = True
                key = raw_val
            elif score >= 0.85:
                mentioned = True
                key = label or raw_val

            if mentioned:
                # Avoid conflicting predicates on same column
                already_has_col = any(f"{col_name} =" in pred for pred in matched_predicates.values())
                if not already_has_col:
                    matched_predicates[key] = predicate

        return matched_predicates


class TwoStageSchemaRetriever:
    """Problem 2 Solution: Two-Stage Schema Retriever & Nested Flattener.
    
    Prevents 700+ column context window overflows and schema linking errors:
    - Stage 1: Coarse Table Filter prunes massive schemas down to top 3-5 candidate tables.
    - Stage 2: Fine-Grained Column/Field Selection identifies relevant paths including nested structs/arrays.
    """

    def __init__(
        self,
        collection_prefix: str = "schema_retriever",
        chroma_client: Optional[chromadb.ClientAPI] = None,
    ):
        self.client = chroma_client or get_chroma_client()
        self.embeddings = get_embeddings()

        self.table_store = Chroma(
            client=self.client,
            collection_name=f"{collection_prefix}_tables",
            embedding_function=self.embeddings,
        )
        self.column_store = Chroma(
            client=self.client,
            collection_name=f"{collection_prefix}_columns",
            embedding_function=self.embeddings,
        )
        self.value_indexer = DistinctValueIndexer(
            collection_name=f"{collection_prefix}_values",
            chroma_client=self.client,
        )

        self.table_cards: Dict[str, str] = {}
        self.table_fields: Dict[str, List[FlattenedField]] = {}

    def index_database_schema(
        self,
        tables_metadata: Dict[str, Dict[str, Any]],
        sample_values: Optional[Dict[str, Dict[str, List[Tuple[str, str]]]]] = None,
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
        """
        table_docs: List[Document] = []
        column_docs: List[Document] = []

        for table_name, meta in tables_metadata.items():
            desc = meta.get("description", "")
            fields: List[FlattenedField] = meta.get("fields", [])
            self.table_fields[table_name] = fields

            # Create rich table card
            schema_card = SchemaFlattener.generate_schema_card(table_name, fields, description=desc)
            self.table_cards[table_name] = schema_card

            # Stage 1: Coarse Table Document
            field_names_summary = ", ".join([f.path for f in fields[:25]])
            table_summary = (
                f"Table Name: {table_name}\n"
                f"Purpose: {desc}\n"
                f"Primary Keys: {', '.join(meta.get('primary_keys', []))}\n"
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
                col_content = f"{field.table_name}.{field.path} ({field.data_type}){flags}: {field.comment or desc}"
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
            - candidate_tables: List of top 3-5 selected table names.
            - schema_context: Compact, LLM-ready schema prompt context.
            - matched_literals: Dict of entity -> exact SQL literal predicate.
        """
        # Stage 1: Coarse Table Filter
        # Fetch slightly more to account for duplicates and deduplicate preserving order
        raw_table_results = self.table_store.similarity_search(user_query, k=top_k_tables * 2)
        candidate_tables: List[str] = []
        for doc in raw_table_results:
            tbl = doc.metadata["table_name"]
            if tbl not in candidate_tables:
                candidate_tables.append(tbl)
            if len(candidate_tables) >= top_k_tables:
                break

        # Distinct Value Indexer Matching
        matched_literals = self.value_indexer.match_literals(user_query)

        # Ensure tables referenced by matched literals are included in candidate tables
        for doc, score in self.value_indexer.vector_store.similarity_search_with_relevance_scores(user_query, k=3):
            tbl = doc.metadata.get("table_name")
            pred = doc.metadata.get("sql_predicate", "")
            if tbl and any(pred == v for v in matched_literals.values()):
                if tbl not in candidate_tables:
                    candidate_tables.append(tbl)

        # Stage 2: Fine-Grained Column/Field Retrieval within candidate tables
        column_results = self.column_store.similarity_search(user_query, k=len(candidate_tables) * top_k_columns_per_table)
        relevant_columns_by_table: Dict[str, Set[str]] = {t: set() for t in candidate_tables}

        # Assemble compact schema cards for candidate tables
        schema_sections: List[str] = []
        for table_name in candidate_tables:
            fields = self.table_fields.get(table_name, [])
            relevant_paths = relevant_columns_by_table.get(table_name, set())

            # Always include fields that were matched or top fields
            filtered_fields = [f for f in fields if f.path in relevant_paths]
            # If too few were caught by column search, include first 10 fields for completeness
            if len(filtered_fields) < 3 and fields:
                filtered_fields = fields[:12]

            card = SchemaFlattener.generate_schema_card(
                table_name=table_name,
                fields=filtered_fields,
                description=f"Filtered schema subset for query: {user_query}",
            )
            schema_sections.append(card)

        combined_schema_context = "\n\n".join(schema_sections)
        return candidate_tables, combined_schema_context, matched_literals
