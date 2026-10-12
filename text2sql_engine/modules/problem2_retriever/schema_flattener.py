"""Schema Flattener: Recursively converts nested STRUCT, ARRAY, and JSON types to dot-notation paths."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class FlattenedField(BaseModel):
    """Represents a column or flattened nested subfield path."""

    table_name: str = Field(..., description="Name of the table")
    root_column: str = Field(..., description="Top-level column name in the database table")
    path: str = Field(..., description="Dot-notation path, e.g. event_params.value.string_value")
    data_type: str = Field(..., description="Primitive or leaf data type (e.g. VARCHAR, BIGINT)")
    is_nested: bool = Field(default=False, description="True if this field is inside a STRUCT or ARRAY")
    is_array: bool = Field(default=False, description="True if any ancestor is an ARRAY requiring UNNEST")
    parent_array_path: Optional[str] = Field(default=None, description="Path to the nearest enclosing array")
    unnest_syntax: Optional[str] = Field(default=None, description="Recommended UNNEST syntax expression")
    comment: Optional[str] = Field(default=None, description="Description or semantic comment")

    @property
    def qualified_path(self) -> str:
        """Full table-qualified dot path: table.path"""
        return f"{self.table_name}.{self.path}"


def _split_struct_fields(struct_body: str) -> List[Tuple[str, str]]:
    """Split comma-separated fields inside STRUCT(...) taking nested parentheses and quotes into account."""
    fields: List[Tuple[str, str]] = []
    depth = 0
    in_quote = False
    current: List[str] = []

    for char in struct_body:
        if char == '"' or char == "'":
            in_quote = not in_quote
            current.append(char)
        elif not in_quote:
            if char in "([<":
                depth += 1
                current.append(char)
            elif char in ")]>":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                chunk = "".join(current).strip()
                if chunk:
                    fields.append(_parse_single_field(chunk))
                current = []
            else:
                current.append(char)
        else:
            current.append(char)

    chunk = "".join(current).strip()
    if chunk:
        fields.append(_parse_single_field(chunk))

    return fields


def _parse_single_field(field_chunk: str) -> Tuple[str, str]:
    """Parse 'name TYPE' or '"name" TYPE' into (name, TYPE)."""
    field_chunk = field_chunk.strip()
    # Check if field name is quoted
    if field_chunk.startswith('"'):
        end_quote = field_chunk.find('"', 1)
        if end_quote != -1:
            name = field_chunk[1:end_quote]
            type_str = field_chunk[end_quote + 1:].strip()
            return name, type_str
    elif field_chunk.startswith("'"):
        end_quote = field_chunk.find("'", 1)
        if end_quote != -1:
            name = field_chunk[1:end_quote]
            type_str = field_chunk[end_quote + 1:].strip()
            return name, type_str

    # Unquoted name: split on first whitespace
    parts = field_chunk.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    elif len(parts) == 1:
        return parts[0].strip(), "UNKNOWN"
    return "unknown", "UNKNOWN"


class SchemaFlattener:
    """Problem 2 Component: Recursively flattens complex nested structures into dot-notation paths."""

    @classmethod
    def parse_type_string(
        cls,
        table_name: str,
        root_column: str,
        current_path: str,
        type_str: str,
        is_array: bool = False,
        parent_array: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> List[FlattenedField]:
        """Recursively parses a SQL/DuckDB data type string (STRUCT, ARRAY, etc.) into FlattenedFields."""
        type_clean = type_str.strip()

        # Check for ARRAY wrapper e.g. STRUCT(...)[] or ARRAY(STRUCT(...))
        if type_clean.endswith("[]"):
            inner_type = type_clean[:-2].strip()
            return cls.parse_type_string(
                table_name=table_name,
                root_column=root_column,
                current_path=current_path,
                type_str=inner_type,
                is_array=True,
                parent_array=current_path,
                comment=comment,
            )

        if type_clean.upper().startswith("ARRAY(") and type_clean.endswith(")"):
            inner_type = type_clean[6:-1].strip()
            return cls.parse_type_string(
                table_name=table_name,
                root_column=root_column,
                current_path=current_path,
                type_str=inner_type,
                is_array=True,
                parent_array=current_path,
                comment=comment,
            )

        # Check for STRUCT(...)
        if type_clean.upper().startswith("STRUCT(") and type_clean.endswith(")"):
            body = type_clean[7:-1].strip()
            fields = _split_struct_fields(body)
            flattened: List[FlattenedField] = []

            for field_name, sub_type in fields:
                sub_path = f"{current_path}.{field_name}" if current_path else field_name
                sub_flattened = cls.parse_type_string(
                    table_name=table_name,
                    root_column=root_column,
                    current_path=sub_path,
                    type_str=sub_type,
                    is_array=is_array,
                    parent_array=parent_array,
                    comment=comment,
                )
                flattened.extend(sub_flattened)
            return flattened

        # Leaf primitive type (VARCHAR, INT, BIGINT, TIMESTAMP, etc.)
        unnest_syntax = f"UNNEST({parent_array})" if is_array and parent_array else None
        return [
            FlattenedField(
                table_name=table_name,
                root_column=root_column,
                path=current_path,
                data_type=type_clean,
                is_nested=("." in current_path),
                is_array=is_array,
                parent_array_path=parent_array,
                unnest_syntax=unnest_syntax,
                comment=comment,
            )
        ]

    @classmethod
    def flatten_dict_schema(
        cls,
        table_name: str,
        schema_dict: Dict[str, Any],
        prefix: str = "",
        root_col: Optional[str] = None,
        is_array: bool = False,
        parent_array: Optional[str] = None,
    ) -> List[FlattenedField]:
        """Flattens a dictionary / JSON representation of nested schemas."""
        flattened: List[FlattenedField] = []

        for key, val in schema_dict.items():
            current_path = f"{prefix}.{key}" if prefix else key
            current_root = root_col or key

            if isinstance(val, dict):
                # Struct / Object
                flattened.extend(
                    cls.flatten_dict_schema(
                        table_name=table_name,
                        schema_dict=val,
                        prefix=current_path,
                        root_col=current_root,
                        is_array=is_array,
                        parent_array=parent_array,
                    )
                )
            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                # Array of Structs
                flattened.extend(
                    cls.flatten_dict_schema(
                        table_name=table_name,
                        schema_dict=val[0],
                        prefix=current_path,
                        root_col=current_root,
                        is_array=True,
                        parent_array=current_path,
                    )
                )
            elif isinstance(val, list):
                # Array of primitives
                elem_type = str(val[0]) if len(val) > 0 else "VARCHAR"
                flattened.append(
                    FlattenedField(
                        table_name=table_name,
                        root_column=current_root,
                        path=current_path,
                        data_type=f"ARRAY<{elem_type}>",
                        is_nested=True,
                        is_array=True,
                        parent_array_path=current_path,
                        unnest_syntax=f"UNNEST({current_path})",
                    )
                )
            else:
                # Leaf string type
                flattened.append(
                    FlattenedField(
                        table_name=table_name,
                        root_column=current_root,
                        path=current_path,
                        data_type=str(val),
                        is_nested=("." in current_path),
                        is_array=is_array,
                        parent_array_path=parent_array,
                        unnest_syntax=f"UNNEST({parent_array})" if is_array and parent_array else None,
                    )
                )

        return flattened

    @classmethod
    def flatten_duckdb_table(cls, con: Any, table_name: str) -> List[FlattenedField]:
        """Inspects and flattens all flat and nested columns for a DuckDB table."""
        describe_rows = con.execute(f"DESCRIBE {table_name};").fetchall()
        flattened: List[FlattenedField] = []

        for row in describe_rows:
            col_name = str(row[0])
            col_type = str(row[1])
            fields = cls.parse_type_string(
                table_name=table_name,
                root_column=col_name,
                current_path=col_name,
                type_str=col_type,
            )
            flattened.extend(fields)

        return flattened

    @classmethod
    def generate_schema_card(cls, table_name: str, fields: List[FlattenedField], description: str = "") -> str:
        """Generates an LLM-friendly schema card with dot paths and nested hints."""
        lines = [f"Table: {table_name}"]
        if description:
            lines.append(f"  Description: {description}")
        lines.append("  Columns / Fields:")

        for f in fields:
            flags = []
            if f.is_array:
                flags.append("ARRAY (requires UNNEST)")
            if f.is_nested:
                flags.append("NESTED")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            desc = f" -- {f.comment}" if f.comment else ""
            lines.append(f"    - {f.path} ({f.data_type}){flag_str}{desc}")

        return "\n".join(lines)
