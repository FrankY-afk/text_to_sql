"""Stage 3: Nested Schema Tree Flattener (STRUCT, ARRAY, JSON/JSONB).

Recursively flattens nested schema definitions and sample JSON payloads into
dot-notation field paths (e.g., `event_params.key`, `event_params.value.string_value`,
`event_params.value.int_value`) while preserving:
- Original root and leaf data types.
- Explicit parent-child relationships (tree nodes and parent paths).
- Original full field paths.
- Dialect-aware access metadata (PostgreSQL JSON/JSONB, BigQuery STRUCT/ARRAY,
  DuckDB STRUCT/LIST/JSON, Snowflake VARIANT/FLATTEN).

IMPORTANT ARCHITECTURAL GUARANTEE:
Flattening a nested field into a dot-path does NOT imply it can be selected directly
as `SELECT event_params.value.int_value FROM events`. When `is_directly_queryable`
is False (arrays, JSON/JSONB, or repeated records), callers must emit the dialect-aware
unnesting or JSON extraction clause stored in `dialect_access_hint` / `unnest_syntax`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.cte_planner import normalize_dialect

logger = logging.getLogger(__name__)


class FlattenedField(BaseModel):
    """Represents a relational column or flattened nested subfield with full lineage."""

    table_name: str = Field(..., description="Table name (may be schema-qualified)")
    root_column: str = Field(..., description="Top-level physical column in the table")
    path: str = Field(..., description="Flattened dot-notation path, e.g. event_params.value.int_value")
    field_name: str = Field(..., description="Leaf field name")
    parent_path: Optional[str] = Field(default=None, description="Immediate parent field path (None for root columns)")
    depth: int = Field(default=0, description="Nesting depth (0 = root column, 1+ = nested)")
    data_type: str = Field(..., description="Leaf or normalized data type")
    original_type: str = Field(..., description="Original unflattened type definition from source column/node")
    is_nested: bool = Field(default=False, description="True if nested inside STRUCT, OBJECT, ARRAY, or JSON")
    is_array: bool = Field(default=False, description="True if field or an ancestor is an ARRAY / list")
    is_json: bool = Field(default=False, description="True if field resides inside a JSON / JSONB / VARIANT column")
    parent_array_path: Optional[str] = Field(default=None, description="Path to nearest enclosing ARRAY ancestor")
    is_directly_queryable: bool = Field(
        default=True,
        description="False when SQL requires UNNEST, LATERAL FLATTEN, or JSON operators before querying",
    )
    unnest_syntax: Optional[str] = Field(default=None, description="Dialect-specific UNNEST / FLATTEN clause")
    dialect_access_hint: Optional[str] = Field(
        default=None,
        description="Concrete dialect-aware SQL expression/clause required to read this field safely",
    )
    children_paths: List[str] = Field(default_factory=list, description="Direct child field paths if non-leaf")
    comment: Optional[str] = Field(default=None, description="Optional field comment or description")

    @property
    def qualified_path(self) -> str:
        return f"{self.table_name}.{self.path}"


def _split_top_level_fields(body: str) -> List[Tuple[str, str]]:
    """Splits comma-separated struct fields respecting nested (), <>, [], and quotes."""
    fields: List[Tuple[str, str]] = []
    depth_paren = 0
    depth_angle = 0
    depth_bracket = 0
    in_quote: Optional[str] = None
    current: List[str] = []

    for ch in body:
        if in_quote:
            current.append(ch)
            if ch == in_quote:
                in_quote = None
            continue

        if ch in ("'", '"', "`"):
            in_quote = ch
            current.append(ch)
        elif ch == "(":
            depth_paren += 1
            current.append(ch)
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
            current.append(ch)
        elif ch == "<":
            depth_angle += 1
            current.append(ch)
        elif ch == ">":
            depth_angle = max(0, depth_angle - 1)
            current.append(ch)
        elif ch == "[":
            depth_bracket += 1
            current.append(ch)
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
            current.append(ch)
        elif ch == "," and depth_paren == 0 and depth_angle == 0 and depth_bracket == 0:
            chunk = "".join(current).strip()
            if chunk:
                fields.append(_parse_struct_member(chunk))
            current = []
        else:
            current.append(ch)

    tail = "".join(current).strip()
    if tail:
        fields.append(_parse_struct_member(tail))
    return fields


def _parse_struct_member(chunk: str) -> Tuple[str, str]:
    """Parses 'field_name TYPE' (or quoted '`field_name` TYPE') into (name, TYPE)."""
    s = chunk.strip()
    for quote in ('"', "'", "`"):
        if s.startswith(quote):
            end_q = s.find(quote, 1)
            if end_q != -1:
                return s[1:end_q], s[end_q + 1 :].strip() or "UNKNOWN"

    parts = s.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return parts[0].strip() if parts else "unknown", "UNKNOWN"


def _infer_json_primitive_type(val: Any) -> str:
    """Infers SQL-like primitive type from a Python JSON leaf value."""
    if isinstance(val, bool):
        return "BOOLEAN"
    if isinstance(val, int):
        return "BIGINT"
    if isinstance(val, float):
        return "DOUBLE PRECISION"
    return "TEXT"


class SchemaFlattener:
    """Recursively converts STRUCT, ARRAY, and JSON schemas into dot-notation field trees."""

    @classmethod
    def get_dialect_access_metadata(
        cls,
        root_column: str,
        path: str,
        is_array: bool,
        is_json: bool,
        parent_array_path: Optional[str],
        dialect: str = "postgres",
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Computes `(is_directly_queryable, unnest_syntax, dialect_access_hint)`."""
        norm = normalize_dialect(dialect)
        sub_parts = path.split(".")[1:]  # segments after root_column

        # Flat scalar column
        if not is_array and not is_json and not sub_parts:
            return True, None, f"{root_column}"

        # Struct (non-array, non-JSON) dot navigation
        if not is_array and not is_json and sub_parts:
            if norm in ("bigquery", "duckdb", "snowflake"):
                return True, None, path
            # PostgreSQL composite types require parentheses around table/col or JSON
            return False, None, f"({root_column}).{'.'.join(sub_parts)}"

        # JSON / JSONB object navigation without enclosing array
        if is_json and not is_array:
            if not sub_parts:
                return True, None, root_column
            if norm == "postgres":
                # Build root_col->'k1'->>'leaf'
                if len(sub_parts) == 1:
                    expr = f"{root_column}->>'{sub_parts[0]}'"
                else:
                    arrows = "".join(f"->'{p}'" for p in sub_parts[:-1])
                    expr = f"{root_column}{arrows}->>'{sub_parts[-1]}'"
                return False, None, expr
            if norm == "sqlite":
                json_path = "$." + ".".join(sub_parts)
                return False, None, f"json_extract({root_column}, '{json_path}')"
            if norm == "duckdb":
                json_path = "$." + ".".join(sub_parts)
                return False, None, f"json_extract_string({root_column}, '{json_path}')"
            if norm == "bigquery":
                json_path = "$." + ".".join(sub_parts)
                return False, None, f"JSON_VALUE({root_column}, '{json_path}')"
            if norm == "snowflake":
                colon_path = ":".join(sub_parts)
                return False, None, f"{root_column}:{colon_path}::STRING"

        # Array / Repeated fields (requires UNNEST / LATERAL FLATTEN / jsonb_array_elements)
        arr_ref = parent_array_path or root_column
        alias = arr_ref.split(".")[-1] + "_item"
        rel_sub = path[len(arr_ref) :].lstrip(".")

        if norm == "postgres":
            if is_json:
                unnest = f"CROSS JOIN LATERAL jsonb_array_elements({arr_ref}::jsonb) AS {alias}"
                if rel_sub:
                    parts = rel_sub.split(".")
                    arrows = "".join(f"->'{p}'" for p in parts[:-1])
                    hint = f"{unnest} -> select {alias}{arrows}->>'{parts[-1]}'"
                else:
                    hint = f"{unnest} -> select {alias}"
                return False, unnest, hint
            unnest = f"CROSS JOIN LATERAL unnest({arr_ref}) AS {alias}"
            hint = f"{unnest} -> select {alias}" + (f".{rel_sub}" if rel_sub else "")
            return False, unnest, hint

        if norm == "bigquery":
            unnest = f"CROSS JOIN UNNEST({arr_ref}) AS {alias}"
            hint = f"{unnest} -> select {alias}" + (f".{rel_sub}" if rel_sub else "")
            return False, unnest, hint

        if norm == "duckdb":
            unnest = f"CROSS JOIN UNNEST({arr_ref}) AS _u({alias})"
            hint = f"{unnest} -> select {alias}" + (f".{rel_sub}" if rel_sub else "")
            return False, unnest, hint

        if norm == "snowflake":
            unnest = f", LATERAL FLATTEN(input => {arr_ref}) AS {alias}"
            hint = f"{unnest} -> select {alias}.value" + (f":{rel_sub.replace('.', ':')}" if rel_sub else "")
            return False, unnest, hint

        if norm == "sqlite":
            unnest = f", json_each({arr_ref}) AS {alias}"
            hint = f"{unnest} -> select json_extract({alias}.value, '$.{rel_sub}')" if rel_sub else f"{unnest} -> select {alias}.value"
            return False, unnest, hint

        return False, f"UNNEST({arr_ref}) AS {alias}", path

    @classmethod
    def parse_type_string(
        cls,
        table_name: str,
        root_column: str,
        current_path: str,
        type_str: str,
        dialect: str = "postgres",
        parent_path: Optional[str] = None,
        depth: int = 0,
        is_array: bool = False,
        is_json: bool = False,
        parent_array: Optional[str] = None,
        comment: Optional[str] = None,
        include_intermediate_nodes: bool = False,
    ) -> List[FlattenedField]:
        """Recursively parses DuckDB/BigQuery/PostgreSQL type strings into FlattenedField objects."""
        type_clean = type_str.strip()
        upper = type_clean.upper()
        field_name = current_path.split(".")[-1]

        # 1. ARRAY suffix: e.g. STRUCT(...)[] or INT[]
        if type_clean.endswith("[]"):
            inner_type = type_clean[:-2].strip()
            nodes: List[FlattenedField] = []
            children = cls.parse_type_string(
                table_name=table_name,
                root_column=root_column,
                current_path=current_path,
                type_str=inner_type,
                dialect=dialect,
                parent_path=parent_path,
                depth=depth,
                is_array=True,
                is_json=is_json,
                parent_array=current_path,
                comment=comment,
                include_intermediate_nodes=include_intermediate_nodes,
            )
            nodes.extend(children)
            return nodes

        # 2. ARRAY<...> or ARRAY(...)
        if (upper.startswith("ARRAY<") and type_clean.endswith(">")) or (
            upper.startswith("ARRAY(") and type_clean.endswith(")")
        ):
            inner_type = type_clean[6:-1].strip()
            return cls.parse_type_string(
                table_name=table_name,
                root_column=root_column,
                current_path=current_path,
                type_str=inner_type,
                dialect=dialect,
                parent_path=parent_path,
                depth=depth,
                is_array=True,
                is_json=is_json,
                parent_array=current_path,
                comment=comment,
                include_intermediate_nodes=include_intermediate_nodes,
            )

        # 3. STRUCT<...> or STRUCT(...)
        if (upper.startswith("STRUCT<") and type_clean.endswith(">")) or (
            upper.startswith("STRUCT(") and type_clean.endswith(")")
        ):
            body = type_clean[7:-1].strip()
            members = _split_top_level_fields(body)
            results: List[FlattenedField] = []
            child_paths = [f"{current_path}.{m_name}" for m_name, _ in members]

            if include_intermediate_nodes:
                direct_q, unnest, hint = cls.get_dialect_access_metadata(
                    root_column, current_path, is_array, is_json, parent_array, dialect
                )
                results.append(
                    FlattenedField(
                        table_name=table_name,
                        root_column=root_column,
                        path=current_path,
                        field_name=field_name,
                        parent_path=parent_path,
                        depth=depth,
                        data_type="STRUCT",
                        original_type=type_clean,
                        is_nested=(depth > 0 or is_array),
                        is_array=is_array,
                        is_json=is_json,
                        parent_array_path=parent_array,
                        is_directly_queryable=direct_q,
                        unnest_syntax=unnest,
                        dialect_access_hint=hint,
                        children_paths=child_paths,
                        comment=comment,
                    )
                )

            for m_name, m_type in members:
                child_path = f"{current_path}.{m_name}"
                results.extend(
                    cls.parse_type_string(
                        table_name=table_name,
                        root_column=root_column,
                        current_path=child_path,
                        type_str=m_type,
                        dialect=dialect,
                        parent_path=current_path,
                        depth=depth + 1,
                        is_array=is_array,
                        is_json=is_json,
                        parent_array=parent_array,
                        comment=comment,
                        include_intermediate_nodes=include_intermediate_nodes,
                    )
                )
            return results

        # 4. Leaf / Scalar or Opaque JSON field
        leaf_is_json = is_json or upper in ("JSON", "JSONB", "VARIANT", "OBJECT")
        leaf_is_nested = (depth > 0) or is_array or leaf_is_json
        direct_q, unnest, hint = cls.get_dialect_access_metadata(
            root_column=root_column,
            path=current_path,
            is_array=is_array,
            is_json=leaf_is_json,
            parent_array_path=parent_array,
            dialect=dialect,
        )
        display_type = f"ARRAY<{type_clean}>" if is_array and depth == 0 else type_clean

        return [
            FlattenedField(
                table_name=table_name,
                root_column=root_column,
                path=current_path,
                field_name=field_name,
                parent_path=parent_path,
                depth=depth,
                data_type=display_type,
                original_type=type_clean,
                is_nested=leaf_is_nested,
                is_array=is_array,
                is_json=leaf_is_json,
                parent_array_path=parent_array,
                is_directly_queryable=direct_q,
                unnest_syntax=unnest,
                dialect_access_hint=hint,
                comment=comment,
            )
        ]

    @classmethod
    def flatten_json_payload(
        cls,
        table_name: str,
        root_column: str,
        payload: Any,
        dialect: str = "postgres",
        original_type: str = "JSONB",
        current_path: Optional[str] = None,
        parent_path: Optional[str] = None,
        depth: int = 0,
        is_array: bool = False,
        parent_array: Optional[str] = None,
    ) -> List[FlattenedField]:
        """Infers and flattens nested dot-notation paths from a JSON/JSONB sample value or schema template."""
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                pass

        path = current_path or root_column
        fields: List[FlattenedField] = []

        if isinstance(payload, dict):
            for key, val in payload.items():
                child_path = f"{path}.{key}"
                if isinstance(val, dict):
                    fields.extend(
                        cls.flatten_json_payload(
                            table_name=table_name,
                            root_column=root_column,
                            payload=val,
                            dialect=dialect,
                            original_type=original_type,
                            current_path=child_path,
                            parent_path=path,
                            depth=depth + 1,
                            is_array=is_array,
                            parent_array=parent_array,
                        )
                    )
                elif isinstance(val, list):
                    if val and isinstance(val[0], dict):
                        fields.extend(
                            cls.flatten_json_payload(
                                table_name=table_name,
                                root_column=root_column,
                                payload=val[0],
                                dialect=dialect,
                                original_type=original_type,
                                current_path=child_path,
                                parent_path=path,
                                depth=depth + 1,
                                is_array=True,
                                parent_array=child_path,
                            )
                        )
                    else:
                        elem_type = _infer_json_primitive_type(val[0]) if val else "TEXT"
                        direct_q, unnest, hint = cls.get_dialect_access_metadata(
                            root_column, child_path, True, True, child_path, dialect
                        )
                        fields.append(
                            FlattenedField(
                                table_name=table_name,
                                root_column=root_column,
                                path=child_path,
                                field_name=key,
                                parent_path=path,
                                depth=depth + 1,
                                data_type=f"ARRAY<{elem_type}>",
                                original_type=original_type,
                                is_nested=True,
                                is_array=True,
                                is_json=True,
                                parent_array_path=child_path,
                                is_directly_queryable=direct_q,
                                unnest_syntax=unnest,
                                dialect_access_hint=hint,
                            )
                        )
                else:
                    inferred = _infer_json_primitive_type(val)
                    direct_q, unnest, hint = cls.get_dialect_access_metadata(
                        root_column, child_path, is_array, True, parent_array, dialect
                    )
                    fields.append(
                        FlattenedField(
                            table_name=table_name,
                            root_column=root_column,
                            path=child_path,
                            field_name=key,
                            parent_path=path,
                            depth=depth + 1,
                            data_type=inferred,
                            original_type=original_type,
                            is_nested=True,
                            is_array=is_array,
                            is_json=True,
                            parent_array_path=parent_array,
                            is_directly_queryable=direct_q,
                            unnest_syntax=unnest,
                            dialect_access_hint=hint,
                        )
                    )
        elif isinstance(payload, list) and payload:
            # Top-level JSON array (e.g. event_params JSONB = [{"key": "plan", "value": {"string_value": "pro"}}])
            sample_elem = payload[0]
            if isinstance(sample_elem, dict):
                # Merge keys across up to 5 array elements so sparse union keys (string_value + int_value) are all discovered!
                merged: Dict[str, Any] = {}
                for item in payload[:5]:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if k not in merged or (isinstance(v, dict) and isinstance(merged[k], dict)):
                                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                                    merged[k] = {**merged[k], **v}
                                elif v is not None:
                                    merged[k] = v
                fields.extend(
                    cls.flatten_json_payload(
                        table_name=table_name,
                        root_column=root_column,
                        payload=merged,
                        dialect=dialect,
                        original_type=original_type,
                        current_path=path,
                        parent_path=parent_path,
                        depth=depth,
                        is_array=True,
                        parent_array=path,
                    )
                )
            else:
                elem_type = _infer_json_primitive_type(sample_elem)
                direct_q, unnest, hint = cls.get_dialect_access_metadata(
                    root_column, path, True, True, path, dialect
                )
                fields.append(
                    FlattenedField(
                        table_name=table_name,
                        root_column=root_column,
                        path=path,
                        field_name=path.split(".")[-1],
                        parent_path=parent_path,
                        depth=depth,
                        data_type=f"ARRAY<{elem_type}>",
                        original_type=original_type,
                        is_nested=True,
                        is_array=True,
                        is_json=True,
                        parent_array_path=path,
                        is_directly_queryable=direct_q,
                        unnest_syntax=unnest,
                        dialect_access_hint=hint,
                    )
                )
        return fields

    @classmethod
    def flatten_table_columns(
        cls,
        table_or_name: Any,
        columns: Optional[List[Dict[str, Any]]] = None,
        sample_rows: Optional[List[Dict[str, Any]]] = None,
        dialect: str = "postgres",
        include_intermediate_nodes: bool = False,
    ) -> List[FlattenedField]:
        """Flattens all columns of a table (accepting TableMetadata or table_name + column dicts)."""
        flattened: List[FlattenedField] = []
        samples = sample_rows or []

        if columns is None and hasattr(table_or_name, "columns"):
            table_name = getattr(table_or_name, "full_name", None) or getattr(table_or_name, "table_name", "table")
            col_dicts = [
                {
                    "name": c.name,
                    "data_type": c.data_type,
                    "comment": getattr(c, "comment", None),
                }
                for c in table_or_name.columns
            ]
        else:
            table_name = str(table_or_name)
            col_dicts = columns or []

        for col in col_dicts:
            c_name = col["name"]
            c_type = col["data_type"]
            c_upper = c_type.upper()
            comment = col.get("comment")

            # Check for JSON/JSONB columns with JSON sample payloads
            if c_upper in ("JSON", "JSONB", "VARIANT", "OBJECT"):
                merged_fields: Dict[str, FlattenedField] = {}
                for row in samples:
                    raw_val = row.get(c_name)
                    if raw_val is not None:
                        inferred = cls.flatten_json_payload(
                            table_name=table_name,
                            root_column=c_name,
                            payload=raw_val,
                            dialect=dialect,
                            original_type=c_type,
                        )
                        for f in inferred:
                            merged_fields.setdefault(f.path, f)

                if merged_fields:
                    flattened.extend(merged_fields.values())
                    continue

            flattened.extend(
                cls.parse_type_string(
                    table_name=table_name,
                    root_column=c_name,
                    current_path=c_name,
                    type_str=c_type,
                    dialect=dialect,
                    comment=comment,
                    include_intermediate_nodes=include_intermediate_nodes,
                )
            )

        return flattened

