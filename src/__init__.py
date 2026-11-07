"""Text-to-SQL Problem 1: Erroneous Data Analysis Package.

A system built with PostgreSQL, Model Context Protocol (MCP), and semantic validation
to detect and prevent logically erroneous data analysis in SQL queries.
"""

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

__version__ = "0.1.0"

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
]



