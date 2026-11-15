"""Schema inspector module for extracting database metadata and schema cards.

Supports live PostgreSQL (multi-schema, PKs, FKs, comments, sample values),
local SQLite benchmarks, and embedded DuckDB analytical tables.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.db import execute_readonly_query, get_db_cursor, get_local_db_connection

logger = logging.getLogger(__name__)

VALID_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_sql_identifier(identifier: str) -> str:
    """Validates that a SQL identifier (schema, table, column) is safe against injection."""
    clean = identifier.strip()
    if not VALID_IDENTIFIER_RE.match(clean):
        raise ValueError(f"Invalid SQL identifier '{identifier}'. Only alphanumeric/underscore allowed.")
    return clean


def split_qualified_table_name(table_name: str, default_schema: str = "public") -> Tuple[str, str]:
    """Splits 'schema.table' or 'table' into validated (schema_name, table_name)."""
    if "." in table_name:
        parts = table_name.split(".", 1)
        return validate_sql_identifier(parts[0]), validate_sql_identifier(parts[1])
    return validate_sql_identifier(default_schema), validate_sql_identifier(table_name)


class ForeignKeyMetadata(BaseModel):
    """Represents a foreign key relationship between two tables."""

    source_schema: str = Field(default="public")
    source_table: str = Field(...)
    source_column: str = Field(...)
    target_schema: str = Field(default="public")
    target_table: str = Field(...)
    target_column: str = Field(...)
    constraint_name: Optional[str] = Field(default=None)


class ColumnMetadata(BaseModel):
    """Detailed metadata for a single database column."""

    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_target: Optional[str] = None
    comment: Optional[str] = None
    sample_values: List[str] = Field(default_factory=list)


class TableMetadata(BaseModel):
    """Complete structured metadata for a table across any schema."""

    schema_name: str = Field(default="public")
    table_name: str
    full_name: str
    description: Optional[str] = None
    row_count: Optional[int] = None
    columns: List[ColumnMetadata] = Field(default_factory=list)
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: List[ForeignKeyMetadata] = Field(default_factory=list)


class SchemaInspector:
    """Introspects relational database schemas and formats schema context for Text-to-SQL."""

    def __init__(
        self,
        target: str = "postgres",
        settings: Optional[Settings] = None,
        duckdb_conn: Any = None,
    ):
        """Initializes inspector for 'postgres', 'local' (SQLite), or 'duckdb'."""
        self.target = target.lower().strip()
        self.settings = settings or get_settings()
        self.duckdb_conn = duckdb_conn

    def get_schemas(self) -> List[str]:
        """Returns accessible non-system database schemas."""
        if self.target in ("local", "sqlite"):
            return ["main"]

        if self.target == "duckdb":
            if self.duckdb_conn is None:
                return ["main"]
            rows = self.duckdb_conn.execute(
                "SELECT DISTINCT table_schema FROM information_schema.tables "
                "WHERE table_schema NOT IN ('information_schema', 'pg_catalog') ORDER BY table_schema;"
            ).fetchall()
            return [r[0] for r in rows] or ["main"]

        with get_db_cursor(self.settings) as cur:
            cur.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT LIKE 'pg_%'
                  AND schema_name != 'information_schema'
                ORDER BY schema_name;
                """
            )
            return [row[0] for row in cur.fetchall()]

    def get_table_names(
        self,
        schema: Optional[str] = None,
        include_schema_prefix: bool = False,
    ) -> List[str]:
        """Returns sorted list of permitted base table names."""
        if self.target in ("local", "sqlite"):
            conn = get_local_db_connection(self.settings)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
            tables = [row["name"] for row in cur.fetchall()]
            conn.close()
            return [f"main.{t}" for t in tables] if include_schema_prefix else tables

        if self.target == "duckdb":
            if self.duckdb_conn is None:
                return []
            target_schema = validate_sql_identifier(schema or "main")
            rows = self.duckdb_conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = ? AND table_type = 'BASE TABLE' ORDER BY table_name;",
                [target_schema],
            ).fetchall()
            tables = [r[0] for r in rows]
            return [f"{target_schema}.{t}" for t in tables] if include_schema_prefix else tables

        # PostgreSQL
        target_schema = validate_sql_identifier(schema or self.settings.postgres_schema)
        with get_db_cursor(self.settings) as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """,
                (target_schema,),
            )
            tables = [row[0] for row in cur.fetchall()]
            return [f"{target_schema}.{t}" for t in tables] if include_schema_prefix else tables

    def get_foreign_keys(
        self,
        table_name: str,
        schema: Optional[str] = None,
    ) -> List[ForeignKeyMetadata]:
        """Extracts foreign key constraints originating from the specified table."""
        default_schema = "main" if self.target in ("local", "sqlite", "duckdb") else self.settings.postgres_schema
        schema_name, clean_table = split_qualified_table_name(table_name, default_schema=schema or default_schema)

        permitted = self.get_table_names(schema=schema_name)
        if clean_table not in permitted:
            raise ValueError(f"Table '{clean_table}' is not in permitted tables for schema '{schema_name}': {permitted}")

        if self.target in ("local", "sqlite"):
            conn = get_local_db_connection(self.settings)
            cur = conn.cursor()
            cur.execute(f"PRAGMA foreign_key_list('{clean_table}');")
            fks: List[ForeignKeyMetadata] = []
            for row in cur.fetchall():
                fks.append(
                    ForeignKeyMetadata(
                        source_schema="main",
                        source_table=clean_table,
                        source_column=row["from"],
                        target_schema="main",
                        target_table=row["table"],
                        target_column=row["to"],
                        constraint_name=f"fk_{clean_table}_{row['from']}",
                    )
                )
            conn.close()
            return fks

        if self.target == "duckdb":
            return []

        # PostgreSQL
        with get_db_cursor(self.settings) as cur:
            cur.execute(
                """
                SELECT
                    tc.constraint_name,
                    kcu.column_name AS source_column,
                    ccu.table_schema AS target_schema,
                    ccu.table_name AS target_table,
                    ccu.column_name AS target_column
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = %s
                  AND tc.table_name = %s
                ORDER BY kcu.ordinal_position;
                """,
                (schema_name, clean_table),
            )
            fks = []
            for cname, src_col, tgt_schema, tgt_tbl, tgt_col in cur.fetchall():
                fks.append(
                    ForeignKeyMetadata(
                        source_schema=schema_name,
                        source_table=clean_table,
                        source_column=src_col,
                        target_schema=tgt_schema,
                        target_table=tgt_tbl,
                        target_column=tgt_col,
                        constraint_name=cname,
                    )
                )
            return fks

    def get_table_columns(
        self,
        table_name: str,
        schema: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Returns detailed column specifications for a specific table."""
        default_schema = "main" if self.target in ("local", "sqlite", "duckdb") else self.settings.postgres_schema
        schema_name, clean_table = split_qualified_table_name(table_name, default_schema=schema or default_schema)

        permitted = self.get_table_names(schema=schema_name)
        if clean_table not in permitted:
            raise ValueError(f"Table '{clean_table}' is not in permitted tables: {permitted}")

        fks = self.get_foreign_keys(clean_table, schema=schema_name)
        fk_map = {fk.source_column: f"{fk.target_table}.{fk.target_column}" for fk in fks}

        if self.target in ("local", "sqlite"):
            conn = get_local_db_connection(self.settings)
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info('{clean_table}');")
            cols = []
            for row in cur.fetchall():
                cname = row["name"]
                cols.append({
                    "name": cname,
                    "data_type": row["type"].upper(),
                    "nullable": not bool(row["notnull"]),
                    "is_primary_key": bool(row["pk"]),
                    "is_foreign_key": cname in fk_map,
                    "foreign_key_target": fk_map.get(cname),
                    "comment": None,
                })
            conn.close()
            return cols

        if self.target == "duckdb":
            if self.duckdb_conn is None:
                return []
            qualified = f'"{schema_name}"."{clean_table}"' if schema_name != "main" else f'"{clean_table}"'
            rows = self.duckdb_conn.execute(f"DESCRIBE {qualified};").fetchall()
            cols = []
            for r in rows:
                cname = str(r[0])
                raw_dtype = str(r[1])
                dtype = raw_dtype if "STRUCT" in raw_dtype.upper() else raw_dtype.upper()
                cols.append({
                    "name": cname,
                    "data_type": dtype,
                    "nullable": str(r[2]).upper() == "YES",
                    "is_primary_key": str(r[3] or "").upper() == "PRI",
                    "is_foreign_key": cname in fk_map,
                    "foreign_key_target": fk_map.get(cname),
                    "comment": None,
                })
            return cols

        # PostgreSQL
        with get_db_cursor(self.settings) as cur:
            # Fetch primary keys
            cur.execute(
                """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = %s
                  AND tc.table_name = %s;
                """,
                (schema_name, clean_table),
            )
            pks = {row[0] for row in cur.fetchall()}

            # Fetch columns + optional pg_description comments
            cur.execute(
                """
                SELECT
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    pgd.description
                FROM information_schema.columns c
                LEFT JOIN pg_catalog.pg_statio_all_tables st
                  ON c.table_schema = st.schemaname
                  AND c.table_name = st.relname
                LEFT JOIN pg_catalog.pg_description pgd
                  ON pgd.objoid = st.relid
                  AND pgd.objsubid = c.ordinal_position
                WHERE c.table_schema = %s
                  AND c.table_name = %s
                ORDER BY c.ordinal_position;
                """,
                (schema_name, clean_table),
            )
            columns = []
            for col_name, data_type, is_nullable, col_comment in cur.fetchall():
                columns.append({
                    "name": col_name,
                    "data_type": data_type.upper(),
                    "nullable": is_nullable == "YES",
                    "is_primary_key": col_name in pks,
                    "is_foreign_key": col_name in fk_map,
                    "foreign_key_target": fk_map.get(col_name),
                    "comment": col_comment,
                })
            return columns

    def get_schema_card(self, table_name: str, schema: Optional[str] = None) -> str:
        """Generates an LLM-friendly schema card in Markdown format."""
        columns = self.get_table_columns(table_name, schema=schema)
        lines = [f"### Table: `{table_name}`", "| Column | Type | Nullable | Key |", "|---|---|---|---|"]
        for col in columns:
            nullable_str = "YES" if col["nullable"] else "NO"
            key_parts = []
            if col["is_primary_key"]:
                key_parts.append("PRIMARY KEY")
            if col.get("is_foreign_key") and col.get("foreign_key_target"):
                key_parts.append(f"FK -> {col['foreign_key_target']}")
            key_str = ", ".join(key_parts) if key_parts else "-"
            lines.append(f"| `{col['name']}` | `{col['data_type']}` | {nullable_str} | {key_str} |")
        return "\n".join(lines)

    def get_all_schema_cards(self) -> str:
        """Returns concatenated schema cards for all permitted tables."""
        tables = self.get_table_names()
        cards = [self.get_schema_card(t) for t in tables]
        return "\n\n".join(cards)

    def get_sample_rows(
        self,
        table_name: str,
        limit: int = 3,
        schema: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Safely fetches a bounded sample of rows for a permitted table."""
        default_schema = "main" if self.target in ("local", "sqlite", "duckdb") else self.settings.postgres_schema
        schema_name, clean_table = split_qualified_table_name(table_name, default_schema=schema or default_schema)

        permitted = self.get_table_names(schema=schema_name)
        if clean_table not in permitted:
            raise ValueError(f"Table '{clean_table}' is not permitted.")

        clamped_limit = min(max(1, limit), 5)  # Enforce 1 <= limit <= 5

        if self.target in ("local", "sqlite"):
            conn = get_local_db_connection(self.settings)
            cur = conn.cursor()
            cur.execute(f'SELECT * FROM "{clean_table}" LIMIT {clamped_limit};')
            rows = [dict(r) for r in cur.fetchall()]
            columns = list(rows[0].keys()) if rows else []
            conn.close()
            return {"columns": columns, "rows": rows, "count": len(rows)}

        if self.target == "duckdb":
            if self.duckdb_conn is None:
                return {"columns": [], "rows": [], "count": 0}
            qualified = f'"{schema_name}"."{clean_table}"' if schema_name != "main" else f'"{clean_table}"'
            cur = self.duckdb_conn.execute(f"SELECT * FROM {qualified} LIMIT {clamped_limit};")
            columns = [desc[0] for desc in cur.description] if cur.description else []
            raw_rows = cur.fetchall()
            rows = [dict(zip(columns, r)) for r in raw_rows]
            return {"columns": columns, "rows": rows, "count": len(rows)}

        # PostgreSQL
        query = f'SELECT * FROM "{schema_name}"."{clean_table}" LIMIT {clamped_limit};'
        res = execute_readonly_query(query, settings=self.settings)
        return {"columns": res["columns"], "rows": res["rows"], "count": res["row_count"]}

    def get_table_metadata(
        self,
        table_name: str,
        schema: Optional[str] = None,
        include_samples: bool = True,
    ) -> TableMetadata:
        """Extracts structured TableMetadata (columns, PKs, FKs, comments, sample values)."""
        default_schema = "main" if self.target in ("local", "sqlite", "duckdb") else self.settings.postgres_schema
        schema_name, clean_table = split_qualified_table_name(table_name, default_schema=schema or default_schema)

        raw_cols = self.get_table_columns(clean_table, schema=schema_name)
        fks = self.get_foreign_keys(clean_table, schema=schema_name)

        sample_map: Dict[str, List[str]] = {}
        if include_samples:
            try:
                sample_res = self.get_sample_rows(clean_table, limit=3, schema=schema_name)
                for row in sample_res.get("rows", []):
                    for k, v in row.items():
                        if v is not None:
                            vals = sample_map.setdefault(k, [])
                            sval = str(v)
                            if len(sval) <= 80 and sval not in vals:
                                vals.append(sval)
            except Exception as exc:
                logger.debug("Sample fetch skipped for %s.%s: %s", schema_name, clean_table, exc)

        table_comment: Optional[str] = None
        if self.target == "postgres":
            try:
                with get_db_cursor(self.settings) as cur:
                    cur.execute(
                        """
                        SELECT obj_description((%s || '.' || %s)::regclass, 'pg_class');
                        """,
                        (f'"{schema_name}"', f'"{clean_table}"'),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        table_comment = str(row[0])
            except Exception:
                pass

        col_models: List[ColumnMetadata] = []
        pks: List[str] = []
        for c in raw_cols:
            if c["is_primary_key"]:
                pks.append(c["name"])
            col_models.append(
                ColumnMetadata(
                    name=c["name"],
                    data_type=c["data_type"],
                    nullable=c["nullable"],
                    is_primary_key=c["is_primary_key"],
                    is_foreign_key=c.get("is_foreign_key", False),
                    foreign_key_target=c.get("foreign_key_target"),
                    comment=c.get("comment"),
                    sample_values=sample_map.get(c["name"], [])[:3],
                )
            )

        return TableMetadata(
            schema_name=schema_name,
            table_name=clean_table,
            full_name=f"{schema_name}.{clean_table}",
            description=table_comment,
            columns=col_models,
            primary_keys=pks,
            foreign_keys=fks,
        )

    def inspect_all_tables(
        self,
        schemas: Optional[List[str]] = None,
        include_samples: bool = True,
    ) -> List[TableMetadata]:
        """Inspects all tables across one or more database schemas."""
        target_schemas = schemas or self.get_schemas()
        all_metadata: List[TableMetadata] = []
        for s in target_schemas:
            for tbl in self.get_table_names(schema=s):
                all_metadata.append(self.get_table_metadata(tbl, schema=s, include_samples=include_samples))
        return all_metadata
