"""End-to-End Autonomous LangChain Agent for Enterprise Text-to-SQL.

Integrates:
1. Custom BaseChatModel adapter wrapping HuggingFaceClient for LCEL pipelines.
2. Complete suite of 12 native LangChain BaseTool adapters (Schema linking, Join graph,
   Literal resolution, Doc RAG, Database Exploration, AST validation, Read-only execution).
3. ConversationalMemory for multi-turn follow-up resolution and state tracking.
4. Structured planning with SQLPlan and execution-grounded self-repair.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.conversational_memory import ConversationalMemory, FollowupResolution
from src.debugger import ExecutionGroundedDebugger
from src.hf_client import HuggingFaceClient, HuggingFaceInferenceError
from src.langchain_prompts import (
    SQLPlan,
    format_langchain_context,
    get_text2sql_prompt_template,
)
from src.langchain_tools import get_langchain_tool_map, get_langchain_tools
from src.query_executor import MCPQueryExecutor
from src.sql_generator import SQLGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hugging Face LangChain BaseChatModel Adapter
# ---------------------------------------------------------------------------

class HuggingFaceChatModel(BaseChatModel):
    """Custom LangChain BaseChatModel wrapping Hugging Face Inference API."""

    model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 1024
    hf_client: Optional[HuggingFaceClient] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.hf_client is None:
            self.hf_client = HuggingFaceClient()
        if hasattr(self.hf_client, "model_id") and self.hf_client.model_id:
            self.model_name = self.hf_client.model_id

    @property
    def _llm_type(self) -> str:
        return "huggingface_chat_model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronously generates a chat response using HuggingFaceClient."""
        system_instruction: Optional[str] = None
        user_prompts: List[str] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = msg.content
            elif isinstance(msg, HumanMessage):
                user_prompts.append(msg.content)
            elif isinstance(msg, AIMessage):
                user_prompts.append(f"Assistant: {msg.content}")
            else:
                user_prompts.append(f"{msg.type}: {msg.content}")

        combined_prompt = "\n\n".join(user_prompts)

        if self.hf_client and self.hf_client.is_configured():
            try:
                sql_resp = self.hf_client.generate_sql(
                    prompt=combined_prompt,
                    system_instruction=system_instruction,
                    max_tokens=self.max_tokens,
                )
                message = AIMessage(content=sql_resp)
                return ChatResult(generations=[ChatGeneration(message=message)])
            except HuggingFaceInferenceError as exc:
                logger.warning("Hugging Face chat completion failed: %s; falling back to offline", exc)

        # Offline deterministic fallback notification
        fallback_msg = AIMessage(content="-- LLM inference offline. Execution-grounded planner engaged.")
        return ChatResult(generations=[ChatGeneration(message=fallback_msg)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generates a chat response."""
        return self._generate(messages, stop=stop, run_manager=None, **kwargs)


# ---------------------------------------------------------------------------
# Structured Agent Result Model
# ---------------------------------------------------------------------------

class AgentExecutionResult(BaseModel):
    """Structured report returned by the LangChain Text-to-SQL Autonomous Agent."""

    query: str
    resolved_query: str
    is_followup: bool = False
    sql: str = ""
    success: bool = False
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)
    row_count: int = 0
    raw_result: Optional[float] = None
    error: Optional[str] = None
    steps_taken: List[str] = Field(default_factory=list)
    intermediate_tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    plan: Optional[SQLPlan] = None
    followup_context: Optional[FollowupResolution] = None


# ---------------------------------------------------------------------------
# End-to-End Autonomous LangChain Agent
# ---------------------------------------------------------------------------

class LangChainText2SQLAgent:
    """Enterprise LangChain Agent coordinating MCP tools, reasoning, and multi-turn memory."""

    def __init__(
        self,
        executor: Optional[MCPQueryExecutor] = None,
        memory: Optional[ConversationalMemory] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.executor = executor or MCPQueryExecutor()
        self.memory = memory or ConversationalMemory()
        self.generator = SQLGenerator(executor=self.executor)
        self.debugger = ExecutionGroundedDebugger(
            executor=self.executor,
            generator=self.generator,
            settings=self.settings,
        )

        # Tools and LLM
        self.tools = get_langchain_tools(executor=self.executor)
        self.tool_map = {t.name: t for t in self.tools}
        self.chat_model = HuggingFaceChatModel()
        self.prompt_template = get_text2sql_prompt_template()

    def run(self, user_query: str, target: str = "postgres") -> AgentExecutionResult:
        """Synchronously executes the end-to-end agentic workflow."""
        t0 = time.time()
        steps: List[str] = []
        tool_records: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # Step 1: Multi-Turn Conversational Memory & Follow-Up Resolution
        # -------------------------------------------------------------
        steps.append("Evaluating conversational context and follow-up intent")
        followup_res = self.memory.resolve_followup(user_query, target=target)
        active_query = followup_res.resolved_query

        if followup_res.is_followup:
            steps.append(
                f"Resolved follow-up context against Turn #{followup_res.parent_turn_id}: '{active_query}'"
            )
        else:
            steps.append("Processing standalone natural language query")

        # -------------------------------------------------------------
        # Step 2: Schema Linking & Pruning (Tool: prune_schema)
        # -------------------------------------------------------------
        steps.append("Executing Schema Linking & Column Pruning (prune_schema)")
        prune_tool = self.tool_map.get("prune_schema")
        prune_data: Dict[str, Any] = {}
        if prune_tool:
            prune_data = prune_tool.invoke({
                "natural_language_query": active_query,
                "target": target,
                "top_k_tables": 3,
            })
            tool_records.append({
                "tool": "prune_schema",
                "input": {"query": active_query, "target": target},
                "output_tables": prune_data.get("selected_tables", []),
            })

        candidate_tables: List[str] = prune_data.get("selected_tables", [])
        if not candidate_tables and followup_res.is_followup and followup_res.inherited_tables:
            candidate_tables = list(followup_res.inherited_tables)
        pruned_schema_md: str = prune_data.get("markdown_schema", "")

        # -------------------------------------------------------------
        # Step 3: Categorical Literal Grounding (Tool: resolve_categorical_literals)
        # -------------------------------------------------------------
        steps.append("Grounding categorical mentions and fixing typos (resolve_categorical_literals)")
        literal_tool = self.tool_map.get("resolve_categorical_literals")
        grounded_literals: List[Dict[str, Any]] = []
        if literal_tool:
            grounded_literals = literal_tool.invoke({
                "natural_language_query": active_query,
                "target": target,
            })
            tool_records.append({
                "tool": "resolve_categorical_literals",
                "input": {"query": active_query, "target": target},
                "resolved_count": len(grounded_literals),
            })

        # -------------------------------------------------------------
        # Step 4: Relational Graph Join Path Discovery (Tool: find_join_path)
        # -------------------------------------------------------------
        join_paths: List[str] = []
        if len(candidate_tables) >= 2:
            steps.append("Searching shortest relational join paths in knowledge graph (find_join_path)")
            join_tool = self.tool_map.get("find_join_path")
            if join_tool:
                for i in range(len(candidate_tables) - 1):
                    src_tbl = candidate_tables[i]
                    tgt_tbl = candidate_tables[i + 1]
                    join_res = join_tool.invoke({
                        "source_table": src_tbl,
                        "target_table": tgt_tbl,
                        "target": target,
                    })
                    if join_res.get("found"):
                        join_paths.append(join_res.get("join_condition", ""))
                        tool_records.append({
                            "tool": "find_join_path",
                            "input": {"source": src_tbl, "target": tgt_tbl},
                            "join": join_res.get("join_condition"),
                        })

        # -------------------------------------------------------------
        # Step 5: Metric & Documentation Grounding (Tool: get_metric_definitions)
        # -------------------------------------------------------------
        metrics_tool = self.tool_map.get("get_metric_definitions")
        metric_docs: List[Dict[str, Any]] = []
        if metrics_tool:
            metric_docs = metrics_tool.invoke({})
            tool_records.append({
                "tool": "get_metric_definitions",
                "count": len(metric_docs),
            })

        # -------------------------------------------------------------
        # Step 6: Plan Formulation & SQL Generation
        # -------------------------------------------------------------
        steps.append("Generating analytical SQL query with semantic guards")
        context_args = format_langchain_context(
            pruned_schema_markdown=pruned_schema_md,
            grounded_literals=grounded_literals,
            join_paths=join_paths,
        )

        generated_sql = ""
        # Check if remote Hugging Face endpoint is active
        if self.chat_model.hf_client and self.chat_model.hf_client.is_configured():
            try:
                lcel_chain = self.prompt_template | self.chat_model | StrOutputParser()
                generated_sql = lcel_chain.invoke({
                    "question": active_query,
                    "schema_context": context_args["schema_context"],
                    "literal_hints": context_args["literal_hints"],
                })
            except Exception as exc:
                logger.warning("LCEL generation failed: %s; falling back to debugger generator", exc)
                generated_sql = ""

        # If LLM offline or empty, utilize adaptive generator
        if not generated_sql or generated_sql.startswith("-- LLM inference offline"):
            gen_res = self.generator.generate_improved_sql(active_query, target=target)
            generated_sql = gen_res.generated_sql

        # -------------------------------------------------------------
        # Step 7: AST Validation (Tool: validate_sql)
        # -------------------------------------------------------------
        steps.append("Validating AST safety and dialect compliance (validate_sql)")
        val_tool = self.tool_map.get("validate_sql")
        val_report: Dict[str, Any] = {}
        if val_tool:
            val_report = val_tool.invoke({
                "sql": generated_sql,
                "target": target,
            })
            tool_records.append({
                "tool": "validate_sql",
                "is_valid": val_report.get("is_valid", False),
            })

        # -------------------------------------------------------------
        # Step 8: Execution & Self-Repair Loop
        # -------------------------------------------------------------
        steps.append("Executing query safely via MCP execution engine (execute_readonly_sql)")
        exec_tool = self.tool_map.get("execute_readonly_sql")
        exec_res = exec_tool.invoke({"sql": generated_sql, "target": target}) if exec_tool else {}

        # If AST validation failed or execution failed, trigger Debugger self-repair
        if not val_report.get("is_valid", True) or not exec_res.get("success", False):
            steps.append("Triggering Execution-Grounded Debugger cyclic self-repair loop")
            debug_result = self.debugger.execute_with_debugging(active_query, target=target)
            generated_sql = debug_result.final_sql
            success = debug_result.success
            rows = debug_result.rows
            columns = debug_result.columns
            row_count = debug_result.row_count
            raw_result = debug_result.raw_result
            error = debug_result.error
            steps.append(f"Self-repair concluded: {debug_result.total_attempts} attempt(s)")
        else:
            success = exec_res.get("success", True)
            rows = exec_res.get("rows", [])
            columns = exec_res.get("columns", [])
            row_count = len(rows)
            raw_result = None
            if rows:
                first_v = list(rows[0].values())[0]
                try:
                    raw_result = float(first_v) if first_v is not None else None
                except (ValueError, TypeError):
                    raw_result = None
            error = exec_res.get("error")

        elapsed_ms = (time.time() - t0) * 1000

        # -------------------------------------------------------------
        # Step 9: Record into Conversational Memory
        # -------------------------------------------------------------
        self.memory.add_turn(
            user_query=user_query,
            resolved_query=active_query,
            generated_sql=generated_sql,
            target=target,
            candidate_tables=candidate_tables,
            group_by_dimensions=followup_res.breakdown_dimensions,
            columns=columns,
            rows=rows,
            row_count=row_count,
            raw_result=raw_result,
            success=success,
        )

        return AgentExecutionResult(
            query=user_query,
            resolved_query=active_query,
            is_followup=followup_res.is_followup,
            sql=generated_sql,
            success=success,
            rows=rows,
            columns=columns,
            row_count=row_count,
            raw_result=raw_result,
            error=error,
            steps_taken=steps,
            intermediate_tool_calls=tool_records,
            latency_ms=elapsed_ms,
            followup_context=followup_res,
        )

    async def arun(self, user_query: str, target: str = "postgres") -> AgentExecutionResult:
        """Asynchronously executes the end-to-end agentic workflow in a worker thread."""
        import asyncio
        return await asyncio.to_thread(self.run, user_query, target)
