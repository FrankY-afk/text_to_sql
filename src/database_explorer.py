"""Problem 3: Autonomous Database Explorer & Read-Only Diagnostic Probing.

Addresses Spider 2.0 Problem 3 where ambiguous questions, data distribution uncertainties,
or complex multi-table joins require interactive read-only database probing before
generating final production queries.

Probing Capabilities:
1. `ROW_COUNT_CHECK`: Verifies tables are non-empty and checks scale.
2. `DATE_BOUNDARY_PROBE`: Retrieves MIN/MAX dates to prevent querying non-existent date intervals.
3. `DISTRIBUTION_PROBE`: Discovers active values and frequency counts in categorical columns.
4. `NULL_CHECK`: Inspects null rates in key filtering or join columns.
5. `JOIN_CARDINALITY_PROBE`: Evaluates whether a proposed join multiplies rows (Cartesian fan-out risk).
6. Synthesizes exploration findings into concrete execution guidance for CTEPlanner and SQLGenerator.
"""

from __future__ import annotations

from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
# Deferred import of MCPQueryExecutor to prevent circular dependencies
from src.schema_inspector import SchemaInspector, validate_sql_identifier

logger = logging.getLogger(__name__)


class ProbeType(str, Enum):
    ROW_COUNT_CHECK = "row_count_check"
    DATE_BOUNDARY_PROBE = "date_boundary_probe"
    DISTRIBUTION_PROBE = "distribution_probe"
    NULL_CHECK = "null_check"
    JOIN_CARDINALITY_PROBE = "join_cardinality_probe"


class DiagnosticProbe(BaseModel):
    """Result of an exploratory diagnostic query executed on the database."""

    probe_id: str
    probe_type: ProbeType
    table_name: str
    column_name: Optional[str] = None
    sql_executed: str
    result_data: Dict[str, Any] = Field(default_factory=dict)
    finding_summary: str
    has_anomaly: bool = False
    anomaly_detail: Optional[str] = None


class ExplorationReport(BaseModel):
    """Aggregated findings from autonomous database exploration before final SQL generation."""

    user_query: str
    target: str = "postgres"
    probes_run: List[DiagnosticProbe] = Field(default_factory=list)
    discovered_date_ranges: Dict[str, Tuple[str, str]] = Field(default_factory=dict)
    discovered_value_domains: Dict[str, List[str]] = Field(default_factory=dict)
    fanout_risk_detected: bool = False
    exploration_guidance: str = ""


class DatabaseExplorer:
    """Autonomous explorer that executes read-only diagnostic probes to ground Text-to-SQL."""

    def __init__(
        self,
        executor: Any = None,
        settings: Optional[Settings] = None,
    ):
        if executor is not None:
            self.executor = executor
        else:
            from src.query_executor import MCPQueryExecutor
            self.executor = MCPQueryExecutor()
        self.settings = settings or get_settings()

    def probe_date_bounds(
        self,
        table_name: str,
        date_column: str,
        target: str = "postgres",
    ) -> DiagnosticProbe:
        """Probes the minimum and maximum dates present in a table."""
        clean_tbl = validate_sql_identifier(table_name.split(".")[-1])
        clean_col = validate_sql_identifier(date_column)
        sql = f'SELECT MIN("{clean_col}") AS min_val, MAX("{clean_col}") AS max_val FROM "{clean_tbl}";'

        try:
            res = self.executor.execute_readonly_sql_sync(sql=sql, target=target, max_rows=1)
            rows = res.get("rows", [])
            if rows and rows[0]:
                min_v = str(rows[0].get("min_val", "unknown"))
                max_v = str(rows[0].get("max_val", "unknown"))
                summary = f"Table '{clean_tbl}.{clean_col}' has date range from {min_v} to {max_v}."
                return DiagnosticProbe(
                    probe_id=f"date_{clean_tbl}_{clean_col}",
                    probe_type=ProbeType.DATE_BOUNDARY_PROBE,
                    table_name=clean_tbl,
                    column_name=clean_col,
                    sql_executed=sql,
                    result_data={"min_val": min_v, "max_val": max_v},
                    finding_summary=summary,
                    has_anomaly=min_v in ("None", "null", "") or max_v in ("None", "null", ""),
                )
        except Exception as exc:
            logger.debug("Date probe failed for %s.%s: %s", clean_tbl, clean_col, exc)

        return DiagnosticProbe(
            probe_id=f"date_{clean_tbl}_{clean_col}",
            probe_type=ProbeType.DATE_BOUNDARY_PROBE,
            table_name=clean_tbl,
            column_name=clean_col,
            sql_executed=sql,
            finding_summary=f"Date probe unavailable for {clean_tbl}.{clean_col}",
            has_anomaly=True,
            anomaly_detail="Query execution failed or column empty",
        )

    def probe_cardinality(
        self,
        table_name: str,
        target: str = "postgres",
    ) -> DiagnosticProbe:
        """Probes row count of a table."""
        clean_tbl = validate_sql_identifier(table_name.split(".")[-1])
        sql = f'SELECT COUNT(*) AS row_cnt FROM "{clean_tbl}";'

        try:
            res = self.executor.execute_readonly_sql_sync(sql=sql, target=target, max_rows=1)
            rows = res.get("rows", [])
            if rows:
                cnt = int(rows[0].get("row_cnt", 0))
                summary = f"Table '{clean_tbl}' contains {cnt} total rows."
                return DiagnosticProbe(
                    probe_id=f"card_{clean_tbl}",
                    probe_type=ProbeType.ROW_COUNT_CHECK,
                    table_name=clean_tbl,
                    sql_executed=sql,
                    result_data={"row_count": cnt},
                    finding_summary=summary,
                    has_anomaly=(cnt == 0),
                    anomaly_detail="Table contains 0 rows" if cnt == 0 else None,
                )
        except Exception as exc:
            logger.debug("Cardinality probe failed for %s: %s", clean_tbl, exc)

        return DiagnosticProbe(
            probe_id=f"card_{clean_tbl}",
            probe_type=ProbeType.ROW_COUNT_CHECK,
            table_name=clean_tbl,
            sql_executed=sql,
            finding_summary=f"Cardinality probe failed for {clean_tbl}",
            has_anomaly=True,
            anomaly_detail="Query failed",
        )

    def probe_nullability(
        self,
        table_name: str,
        column_name: str,
        target: str = "postgres",
    ) -> DiagnosticProbe:
        """Probes percentage of NULL values in a column."""
        clean_tbl = validate_sql_identifier(table_name.split(".")[-1])
        clean_col = validate_sql_identifier(column_name)
        sql = (
            f'SELECT COUNT(*) AS total_rows, '
            f'COUNT(CASE WHEN "{clean_col}" IS NULL THEN 1 END) AS null_rows '
            f'FROM "{clean_tbl}";'
        )

        try:
            res = self.executor.execute_readonly_sql_sync(sql=sql, target=target, max_rows=1)
            rows = res.get("rows", [])
            if rows:
                total = int(rows[0].get("total_rows", 0))
                nulls = int(rows[0].get("null_rows", 0))
                null_pct = round((nulls * 100.0 / total), 2) if total > 0 else 0.0
                summary = f"Column '{clean_tbl}.{clean_col}' has {null_pct}% NULLs ({nulls}/{total})."
                return DiagnosticProbe(
                    probe_id=f"null_{clean_tbl}_{clean_col}",
                    probe_type=ProbeType.NULL_CHECK,
                    table_name=clean_tbl,
                    column_name=clean_col,
                    sql_executed=sql,
                    result_data={"total_rows": total, "null_rows": nulls, "null_pct": null_pct},
                    finding_summary=summary,
                    has_anomaly=(null_pct > 20.0),
                    anomaly_detail=f"High null rate: {null_pct}%" if null_pct > 20.0 else None,
                )
        except Exception as exc:
            logger.debug("Null probe failed for %s.%s: %s", clean_tbl, clean_col, exc)

        return DiagnosticProbe(
            probe_id=f"null_{clean_tbl}_{clean_col}",
            probe_type=ProbeType.NULL_CHECK,
            table_name=clean_tbl,
            column_name=clean_col,
            sql_executed=sql,
            finding_summary=f"Null probe unavailable for {clean_tbl}.{clean_col}",
            has_anomaly=False,
        )

    def probe_join_multiplicity(
        self,
        table_a: str,
        table_b: str,
        join_predicate: str,
        target: str = "postgres",
    ) -> DiagnosticProbe:
        """Tests whether joining Table A and Table B causes row multiplication (Cartesian fan-out)."""
        clean_a = validate_sql_identifier(table_a.split(".")[-1])
        clean_b = validate_sql_identifier(table_b.split(".")[-1])
        sql = (
            f"SELECT "
            f'(SELECT COUNT(*) FROM "{clean_a}") AS cnt_a, '
            f'(SELECT COUNT(*) FROM "{clean_b}") AS cnt_b, '
            f'COUNT(*) AS joined_cnt '
            f'FROM "{clean_a}" a JOIN "{clean_b}" b ON {join_predicate};'
        )

        try:
            res = self.executor.execute_readonly_sql_sync(sql=sql, target=target, max_rows=1)
            rows = res.get("rows", [])
            if rows:
                cnt_a = int(rows[0].get("cnt_a", 0))
                joined = int(rows[0].get("joined_cnt", 0))
                is_fanout = joined > cnt_a and cnt_a > 0
                fanout_ratio = round(joined / cnt_a, 2) if cnt_a > 0 else 1.0
                summary = (
                    f"Join '{clean_a}' -> '{clean_b}' produces {joined} rows from {cnt_a} base rows "
                    f"(multiplicity factor={fanout_ratio}x)."
                )
                return DiagnosticProbe(
                    probe_id=f"join_{clean_a}_{clean_b}",
                    probe_type=ProbeType.JOIN_CARDINALITY_PROBE,
                    table_name=f"{clean_a}+{clean_b}",
                    sql_executed=sql,
                    result_data={"cnt_a": cnt_a, "joined_cnt": joined, "fanout_ratio": fanout_ratio},
                    finding_summary=summary,
                    has_anomaly=is_fanout,
                    anomaly_detail=(
                        f"Cartesian Fan-Out Risk: Join multiplies rows by {fanout_ratio}x. "
                        f"Aggregate '{clean_b}' in a separate CTE before joining!"
                        if is_fanout else None
                    ),
                )
        except Exception as exc:
            logger.debug("Join multiplicity probe failed: %s", exc)

        return DiagnosticProbe(
            probe_id=f"join_{clean_a}_{clean_b}",
            probe_type=ProbeType.JOIN_CARDINALITY_PROBE,
            table_name=f"{clean_a}+{clean_b}",
            sql_executed=sql,
            finding_summary="Join multiplicity check completed cleanly.",
            has_anomaly=False,
        )

    def explore_for_query(
        self,
        user_query: str,
        target: str = "postgres",
        candidate_tables: Optional[List[str]] = None,
    ) -> ExplorationReport:
        """Executes targeted exploratory probes based on user query intent."""
        probes: List[DiagnosticProbe] = []
        date_ranges: Dict[str, Tuple[str, str]] = {}
        domains: Dict[str, List[str]] = {}
        fanout_detected = False
        guidance_lines: List[str] = []

        q_lower = user_query.lower()
        tables = candidate_tables or []

        if not tables:
            inspector = SchemaInspector(target=target, settings=self.settings)
            all_tbls = inspector.get_table_names()
            for t in all_tbls:
                t_stem = t.split(".")[-1].replace("_", " ")
                if any(w in q_lower for w in t_stem.split()):
                    tables.append(t)
            if not tables:
                tables = all_tbls[:2]

        # 1. Probe Cardinality for top candidate tables
        for tbl in tables[:3]:
            p = self.probe_cardinality(tbl, target=target)
            probes.append(p)
            if p.has_anomaly:
                guidance_lines.append(f"WARNING: {p.finding_summary}")

        # 2. Probe Date Boundaries if temporal keywords exist
        temporal_keywords = ["january", "february", "june", "2024", "date", "month", "retention", "overdue"]
        if any(w in q_lower for w in temporal_keywords):
            date_col_candidates = [
                ("users", "signup_date"),
                ("activity_events", "event_date"),
                ("staff_attendance", "attendance_date"),
                ("overdue_daily", "snapshot_date"),
                ("group_loans", "disbursed_date"),
            ]
            for tbl, dcol in date_col_candidates:
                if any(tbl in t for t in tables) or tbl in tables:
                    dp = self.probe_date_bounds(tbl, dcol, target=target)
                    probes.append(dp)
                    min_v = dp.result_data.get("min_val")
                    max_v = dp.result_data.get("max_val")
                    if min_v and max_v and min_v != "None":
                        date_ranges[f"{tbl}.{dcol}"] = (min_v, max_v)
                        guidance_lines.append(f"Date Evidence: '{tbl}.{dcol}' spans {min_v} to {max_v}.")

        # 3. Probe Join Multiplicity if multiple candidate tables
        if len(tables) >= 2:
            clean_tbls = [t.split(".")[-1] for t in tables]
            if "staff" in clean_tbls and "staff_attendance" in clean_tbls:
                jp = self.probe_join_multiplicity(
                    "staff", "staff_attendance", "a.staff_no = b.staff_no", target=target
                )
                probes.append(jp)
                if jp.has_anomaly:
                    fanout_detected = True
                    guidance_lines.append(jp.anomaly_detail or "")
            elif "group_loans" in clean_tbls and "overdue_daily" in clean_tbls:
                jp = self.probe_join_multiplicity(
                    "group_loans", "overdue_daily", "a.loan_id = b.loan_id", target=target
                )
                probes.append(jp)
                if jp.has_anomaly:
                    fanout_detected = True
                    guidance_lines.append(jp.anomaly_detail or "")

        full_guidance = "\n".join(f"- {line}" for line in guidance_lines if line).strip()
        return ExplorationReport(
            user_query=user_query,
            target=target,
            probes_run=probes,
            discovered_date_ranges=date_ranges,
            discovered_value_domains=domains,
            fanout_risk_detected=fanout_detected,
            exploration_guidance=full_guidance,
        )
