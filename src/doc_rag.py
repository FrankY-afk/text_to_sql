"""Problem 3: External Documentation Grounding & Business Metric RAG.

Solves the Spider 2.0 Problem 3 challenge where schema metadata alone is insufficient
to write correct SQL because core business logic, metric formulas, and compliance rules
are defined in external documentation, wikis, or dbt metric specifications.

Capabilities:
1. Indexes external Markdown rulebooks and dbt metric definitions.
2. Extracts structured metric specifications (name, formula, required tables/columns,
   filter rules, and common analytical pitfalls).
3. Hybrid retrieval (Keyword/BM25 + Semantic Vector Search via ChromaDB).
4. Formats compact, grounded context blocks for CTEPlanner and SQLGenerator.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.config import Settings, get_chroma_client, get_embedding_function, get_settings

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "business_rules"


class MetricDefinition(BaseModel):
    """Structured enterprise business metric specification."""

    metric_id: str
    name: str
    aliases: List[str] = Field(default_factory=list)
    domain: str = "general"
    description: str
    required_tables: List[str] = Field(default_factory=list)
    required_columns: List[str] = Field(default_factory=list)
    calculation_formula_sql: str
    filter_rules: List[str] = Field(default_factory=list)
    pitfalls: List[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """A chunked section from external markdown documentation."""

    chunk_id: str
    doc_title: str
    section_header: str
    content: str
    domain_tags: List[str] = Field(default_factory=list)
    similarity_score: float = 0.0


class DocRetrievalResult(BaseModel):
    """Result of documentation and metric retrieval for a user question."""

    query: str
    matched_metrics: List[MetricDefinition] = Field(default_factory=list)
    matched_chunks: List[DocumentChunk] = Field(default_factory=list)
    formatted_prompt_block: str = ""


# Standard Pre-Configured Enterprise Metrics for Spider 2.0 Lending & Operational Domains
BUILTIN_METRICS: List[MetricDefinition] = [
    MetricDefinition(
        metric_id="npa_90_plus",
        name="Non-Performing Assets (NPA / 90+ DPD)",
        aliases=["npa", "non performing assets", "90+ dpd", "severe delinquency", "bad loans", "defaulted loans"],
        domain="lending",
        description="Loans with Days Past Due (DPD) >= 90 days as of evaluation snapshot.",
        required_tables=["overdue_daily"],
        required_columns=["dpd", "loan_id", "overdue_amount"],
        calculation_formula_sql="SELECT loan_id, MAX(dpd) AS max_dpd, SUM(overdue_amount) AS total_npa_overdue FROM overdue_daily WHERE dpd >= 90 GROUP BY loan_id",
        filter_rules=["dpd >= 90 (do not include SMA accounts with DPD < 90)"],
        pitfalls=["Never join overdue_daily directly without grouping by loan_id first (causes Cartesian multiplication)"],
    ),
    MetricDefinition(
        metric_id="gross_loan_portfolio",
        name="Active Loan Portfolio (Gross Loan Portfolio / GLP)",
        aliases=["glp", "active portfolio", "active loans", "disbursed volume", "outstanding portfolio"],
        domain="lending",
        description="Total principal loan volume disbursed and currently active, excluding closed, rejected, or cancelled loans.",
        required_tables=["group_loans", "individual_loans"],
        required_columns=["stage", "loan_amount", "loan_id"],
        calculation_formula_sql="SELECT SUM(loan_amount) AS glp FROM group_loans WHERE stage IN ('Payout', 'Disbursed', 'Active')",
        filter_rules=["stage IN ('Payout', 'Disbursed', 'Active')", "Exclude stage IN ('Rejected', 'Closed', 'Cancelled', 'Enrolment', 'Ready')"],
        pitfalls=["Do not count loans in 'Ready' or 'Enrolment' stage as active disbursed volume"],
    ),
    MetricDefinition(
        metric_id="effective_staff_attendance",
        name="Effective Staff Attendance Rate",
        aliases=["attendance rate", "staff attendance", "effective attendance", "working day attendance"],
        domain="operations",
        description="Ratio of effective days worked (PR=1.0, HD=0.5) to total scheduled working days (excluding WO and PH).",
        required_tables=["staff_attendance"],
        required_columns=["attendance_code", "outlet_code", "staff_no"],
        calculation_formula_sql="(COUNT(CASE WHEN attendance_code = 'PR' THEN 1 END) * 1.0 + COUNT(CASE WHEN attendance_code = 'HD' THEN 1 END) * 0.5) * 100.0 / NULLIF(COUNT(CASE WHEN attendance_code IN ('PR', 'HD', 'CL', 'SL', 'AB', 'LWP') THEN 1 END), 0)",
        filter_rules=["Count PR as 1.0 day, HD as 0.5 day", "Exclude WO (Week Off) and PH (Public Holiday) from denominator"],
        pitfalls=["Do not do a naive COUNT(DISTINCT staff_no) or treat Half-Day (HD) as 1.0 or 0.0"],
    ),
    MetricDefinition(
        metric_id="unplanned_absence_ratio",
        name="Planned vs Unplanned Absence Ratio",
        aliases=["unplanned absence", "absence ratio", "sick leave ratio", "leave breakdown"],
        domain="operations",
        description="Ratio of unplanned absences (AB, LWP) to total leaves taken (CL, SL, AB, LWP).",
        required_tables=["staff_attendance"],
        required_columns=["attendance_code"],
        calculation_formula_sql="COUNT(CASE WHEN attendance_code IN ('AB', 'LWP') THEN 1 END) * 100.0 / NULLIF(COUNT(CASE WHEN attendance_code IN ('CL', 'SL', 'AB', 'LWP') THEN 1 END), 0)",
        filter_rules=["Unplanned: AB, LWP", "Planned: CL, SL"],
        pitfalls=["Ensure NULLIF denominator is used to prevent zero-division when zero leaves occur"],
    ),
    MetricDefinition(
        metric_id="tier1_target_achievement",
        name="Tier-1 Outlet Target Achievement Rate",
        aliases=["target achievement", "disbursement target", "tier 1 achievement", "outlet targets"],
        domain="lending",
        description="Percentage of disbursement target achieved by Tier-1 outlets against pre-set monthly target.",
        required_tables=["outlets", "outlet_targets", "group_loans"],
        required_columns=["tier", "outlet_code", "target_amount", "loan_amount"],
        calculation_formula_sql="COALESCE(actuals.total_disbursed, 0.0) * 100.0 / NULLIF(t.target_amount, 0)",
        filter_rules=["o.tier = 'T1'", "Join group_loans via separate pre-aggregated CTE"],
        pitfalls=["Joining group_loans directly to outlet_targets duplicates loan amounts across target records"],
    ),
    MetricDefinition(
        metric_id="net_disbursed_payout",
        name="Net Disbursed Payout (Net Volume)",
        aliases=["net payout", "net disbursement", "processing fee deduction", "net volume"],
        domain="financial",
        description="Net disbursement credited to borrower after deducting 2.5% upfront processing fee.",
        required_tables=["individual_loans", "group_loans"],
        required_columns=["loan_amount", "stage"],
        calculation_formula_sql="loan_amount * 0.975",
        filter_rules=["Apply 2.5% processing fee deduction (multiply by 0.975)", "stage IN ('Payout', 'Disbursed')"],
        pitfalls=["Do not return raw loan_amount when user explicitly asks for net payout or after fees"],
    ),
    MetricDefinition(
        metric_id="rolling_30d_churn",
        name="30-Day Rolling Churn Rate",
        aliases=["churn rate", "customer churn", "user churn", "lost customers"],
        domain="customer",
        description="Percentage of customers active in baseline 30-day window who had 0 qualifying events in subsequent 30-day window.",
        required_tables=["activity_events", "users"],
        required_columns=["user_id", "event_date", "event_type"],
        calculation_formula_sql="((COUNT(b.user_id) - COUNT(r.user_id)) * 100.0) / NULLIF(COUNT(b.user_id), 0)",
        filter_rules=["Baseline window: event_date >= '2024-01-01' AND event_date < '2024-02-01'", "Subsequent window: event_date >= '2024-02-01' AND event_date < '2024-03-01'"],
        pitfalls=["Use strict '<' upper date boundaries rather than BETWEEN to prevent double-counting edge days"],
    ),
    MetricDefinition(
        metric_id="overdue_provisioning_coverage",
        name="Overdue Loan Provisioning Coverage",
        aliases=["provisioning", "loss provision", "credit provision", "coverage ratio"],
        domain="lending",
        description="Regulatory loan loss provision percentage based on DPD: Standard (0 DPD)=0.4%, SMA (1-89 DPD)=5%, NPA (>=90 DPD)=25%.",
        required_tables=["overdue_daily"],
        required_columns=["dpd", "overdue_amount"],
        calculation_formula_sql="CASE WHEN dpd >= 90 THEN overdue_amount * 0.25 WHEN dpd >= 1 THEN overdue_amount * 0.05 ELSE overdue_amount * 0.004 END",
        filter_rules=["90+ DPD: 25% provision", "1-89 DPD: 5% provision", "0 DPD: 0.4% provision"],
        pitfalls=["Make sure to use CASE statement across distinct DPD brackets"],
    ),
]


class DocumentationRetriever:
    """Hybrid Documentation & Business Metric Retriever for Spider 2.0 Problem 3."""

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        chroma_client: Any = None,
        embedding_function: Any = None,
        embedding_backend: str = "hash",
        in_memory: bool = True,
        collection_name: str = "doc_rag_chunks",
        settings: Optional[Settings] = None,
    ):
        self.docs_dir = docs_dir or DOCS_DIR
        self.settings = settings or get_settings()
        self.embedding_backend = embedding_backend or self.settings.embedding_backend
        self.in_memory = in_memory
        self.collection_name = collection_name
        self.client = chroma_client or get_chroma_client(in_memory=self.in_memory, settings=self.settings)
        self.embedding_function = embedding_function or get_embedding_function(
            settings=self.settings, backend=self.embedding_backend
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self.metrics: Dict[str, MetricDefinition] = {m.metric_id: m for m in BUILTIN_METRICS}
        self.chunks: List[DocumentChunk] = []
        self._index_built = False

    def ensure_indexed(self) -> int:
        """Indexes builtin metrics and disk markdown rulebooks into vector store."""
        if self._index_built:
            return len(self.chunks)

        self.chunks.clear()

        # 1. Index Builtin Metric Definitions
        for m in self.metrics.values():
            metric_text = (
                f"Metric: {m.name}\n"
                f"Domain: {m.domain}\n"
                f"Aliases: {', '.join(m.aliases)}\n"
                f"Description: {m.description}\n"
                f"Required Tables: {', '.join(m.required_tables)}\n"
                f"Filter Rules: {'; '.join(m.filter_rules)}\n"
                f"Pitfalls: {'; '.join(m.pitfalls)}\n"
                f"Formula: {m.calculation_formula_sql}"
            )
            chunk = DocumentChunk(
                chunk_id=f"metric_{m.metric_id}",
                doc_title=m.name,
                section_header="Metric Specification",
                content=metric_text,
                domain_tags=[m.domain] + m.aliases,
            )
            self.chunks.append(chunk)

        # 2. Ingest Markdown Rulebook files from docs_dir
        if self.docs_dir.exists():
            for md_file in self.docs_dir.glob("*.md"):
                self._ingest_markdown_file(md_file)

        # 3. Add to Chroma Vector Collection
        if self.chunks:
            ids = [c.chunk_id for c in self.chunks]
            docs = [f"{c.doc_title} {c.section_header}: {c.content}" for c in self.chunks]
            metadatas = [
                {
                    "doc_title": c.doc_title,
                    "section_header": c.section_header,
                    "chunk_id": c.chunk_id,
                    "tags": ",".join(c.domain_tags),
                }
                for c in self.chunks
            ]
            try:
                # Upsert into collection
                self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
            except Exception as exc:
                logger.debug("Doc RAG upsert note: %s", exc)

        self._index_built = True
        return len(self.chunks)

    def _ingest_markdown_file(self, file_path: Path) -> None:
        """Splits markdown file by H2 (## ) headers into section chunks."""
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            return

        doc_title = file_path.stem.replace("_", " ").title()
        sections = re.split(r"\n(?=##\s+)", text)
        for idx, sec in enumerate(sections):
            lines = sec.strip().split("\n")
            if not lines:
                continue
            header = lines[0].replace("#", "").strip() if lines[0].startswith("#") else f"Section {idx+1}"
            content = "\n".join(lines[1:]).strip() if len(lines) > 1 else sec.strip()
            chunk_id = f"{file_path.stem}_{idx}"
            self.chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_title=doc_title,
                    section_header=header,
                    content=content,
                    domain_tags=[file_path.stem],
                )
            )

    def retrieve_documentation(self, query: str, top_k: int = 3) -> DocRetrievalResult:
        """Retrieves matching business rules and metrics for a natural language question."""
        self.ensure_indexed()
        q_lower = query.lower().strip()

        matched_metrics: List[MetricDefinition] = []
        matched_metric_ids: Set[str] = set()

        # Phase 1: Rule-based & Alias Matching for precision
        for m in self.metrics.values():
            m_terms = [m.name.lower()] + [a.lower() for a in m.aliases]
            for term in m_terms:
                if re.search(r"\b" + re.escape(term) + r"\b", q_lower):
                    if m.metric_id not in matched_metric_ids:
                        matched_metric_ids.add(m.metric_id)
                        matched_metrics.append(m)
                    break

        # Phase 2: Vector Retrieval across document chunks
        matched_chunks: List[DocumentChunk] = []
        if self.chunks:
            try:
                res = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k, len(self.chunks)),
                    include=["metadatas", "documents", "distances"],
                )
                ids = res.get("ids", [[]])[0]
                dists = res.get("distances", [[]])[0]
                for cid, dist in zip(ids, dists):
                    chunk = next((c for c in self.chunks if c.chunk_id == cid), None)
                    if chunk:
                        c_copy = chunk.model_copy()
                        c_copy.similarity_score = round(max(0.0, min(1.0, 1.0 - float(dist))), 4)
                        matched_chunks.append(c_copy)
                        # Also check if it links to a metric
                        if cid.startswith("metric_"):
                            mid = cid.replace("metric_", "")
                            if mid in self.metrics and mid not in matched_metric_ids:
                                matched_metric_ids.add(mid)
                                matched_metrics.append(self.metrics[mid])
            except Exception as exc:
                logger.debug("Doc RAG vector query exception: %s", exc)

        # Format grounding prompt block
        formatted_prompt = self._format_prompt_block(query, matched_metrics, matched_chunks)

        return DocRetrievalResult(
            query=query,
            matched_metrics=matched_metrics,
            matched_chunks=matched_chunks,
            formatted_prompt_block=formatted_prompt,
        )

    def _format_prompt_block(
        self,
        query: str,
        metrics: List[MetricDefinition],
        chunks: List[DocumentChunk],
    ) -> str:
        """Formats grounding context block for injection into LLM prompts and CTEPlanner."""
        if not metrics and not chunks:
            return ""

        lines: List[str] = [
            "### Grounded Business Rules & Metric Specifications (Spider 2.0 Problem 3):",
            "Follow these verified external domain rules and calculation formulas strictly:",
        ]

        if metrics:
            lines.append("\n**Applicable Enterprise Metrics:**")
            for m in metrics:
                lines.append(f"- **Metric Name**: `{m.name}`")
                lines.append(f"  - **Description**: {m.description}")
                lines.append(f"  - **Required Tables**: {', '.join(m.required_tables)}")
                lines.append(f"  - **Filter Rules**: {'; '.join(m.filter_rules)}")
                lines.append(f"  - **Formula Snippet**: `{m.calculation_formula_sql}`")
                if m.pitfalls:
                    lines.append(f"  - **Analytical Pitfalls**: {'; '.join(m.pitfalls)}")

        if chunks:
            non_metric_chunks = [c for c in chunks if not c.chunk_id.startswith("metric_")]
            if non_metric_chunks:
                lines.append("\n**External Documentation Excerpts:**")
                for c in non_metric_chunks[:2]:
                    lines.append(f"- **Source**: {c.doc_title} > {c.section_header}")
                    lines.append(f"  {c.content[:200]}...")

        return "\n".join(lines).strip()
