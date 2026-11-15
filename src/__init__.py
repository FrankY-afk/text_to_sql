"""Enterprise Text-to-SQL Architecture.

Solves Spider 2.0:
- Problem 1: Erroneous Data Analysis (CTE Logic Planner, Dialect RAG, Execution-Grounded Debugger).
- Problem 2: Enterprise Schema Linking (Two-Stage BGE/Chroma Retriever, Schema Flattener, Schema Context Builder, Relational Knowledge Graph).
- Problem 3: Documentation RAG, Multi-Turn Agentic Exploration, and Multi-Hop Relational Joins.
Includes native LangChain tool adapters and LCEL prompt templates for agent orchestration.
"""

from __future__ import annotations

from src.column_retriever import ColumnCandidate, ColumnRetriever
from src.cte_planner import (
    CTEDependencyValidator,
    CTEPlanCompiler,
    CTEPlanStep,
    CTEPlanner,
    LogicPlan,
    PlanValidationError,
)
from src.debugger import (
    DebuggedExecutionResult,
    ErrorCategory,
    ExecutionGroundedDebugger,
)
from src.dialect_rag import (
    DIALECT_CATALOG,
    DialectFunctionRAG,
    DialectFunctionSpec,
)
from src.langchain_prompts import (
    SQLPlan,
    SYSTEM_PROMPT_TEXT2SQL,
    format_langchain_context,
    get_text2sql_prompt_template,
)
from src.langchain_tools import (
    get_langchain_tool_map,
    get_langchain_tools,
)
from src.schema_context_builder import (
    EnterpriseSchemaRetriever,
    SchemaContext,
    SchemaContextBuilder,
)
from src.schema_flattener import FlattenedField, SchemaFlattener
from src.schema_graph import SchemaGraph
from src.schema_pruner import SchemaPruner
from src.table_retriever import TableCandidate, TableRetriever
from src.doc_rag import (
    BUILTIN_METRICS,
    DocRetrievalResult,
    DocumentChunk,
    DocumentationRetriever,
    MetricDefinition,
)
from src.database_explorer import (
    DatabaseExplorer,
    DiagnosticProbe,
    ExplorationReport,
    ProbeType,
)
from src.evaluator import (
    Problem3CaseResult,
    Problem3EvaluationReport,
    Problem3Evaluator,
)

__version__ = "0.3.0"

__all__ = [
    # Problem 1: CTE Logic Planner & Validation
    "CTEPlanStep",
    "LogicPlan",
    "CTEPlanner",
    "CTEDependencyValidator",
    "CTEPlanCompiler",
    "PlanValidationError",
    "DialectFunctionSpec",
    "DialectFunctionRAG",
    "DIALECT_CATALOG",
    "ExecutionGroundedDebugger",
    "ErrorCategory",
    "DebuggedExecutionResult",
    # Problem 2: Two-Stage Schema Linking & Pruning
    "TableCandidate",
    "TableRetriever",
    "ColumnCandidate",
    "ColumnRetriever",
    "FlattenedField",
    "SchemaFlattener",
    "SchemaContext",
    "SchemaContextBuilder",
    "EnterpriseSchemaRetriever",
    "SchemaGraph",
    "SchemaPruner",
    # LangChain Core Integration
    "get_langchain_tools",
    "get_langchain_tool_map",
    "get_text2sql_prompt_template",
    "SQLPlan",
    "SYSTEM_PROMPT_TEXT2SQL",
    "format_langchain_context",
]
