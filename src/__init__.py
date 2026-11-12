"""Text-to-SQL Problem 1 & Problem 2 Enterprise Package.

A system built with PostgreSQL, DuckDB, Model Context Protocol (MCP), Two-Stage Schema Linking,
CTE Planning, Dialect RAG, and semantic validation to prevent erroneous data analysis (Problem 1)
and wrong schema linking (Problem 2).
"""

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
from src.schema_context_builder import (
    EnterpriseSchemaRetriever,
    SchemaContext,
    SchemaContextBuilder,
)
from src.schema_flattener import FlattenedField, SchemaFlattener
from src.table_retriever import TableCandidate, TableRetriever

__version__ = "0.2.0"

__all__ = [
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
    "TableCandidate",
    "TableRetriever",
    "ColumnCandidate",
    "ColumnRetriever",
    "FlattenedField",
    "SchemaFlattener",
    "SchemaContext",
    "SchemaContextBuilder",
    "EnterpriseSchemaRetriever",
]



