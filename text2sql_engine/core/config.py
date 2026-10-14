"""Configuration and initialization for HuggingFace embeddings, ChromaDB, and LLM backends."""

from __future__ import annotations

import os
import json
import re
import urllib.request
from typing import Any, Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from text2sql_engine.core.exceptions import (
    ConfigurationError,
    InferenceError,
    ModelUnavailableError,
)


class EngineSettings(BaseSettings):
    """Central configuration for Text-to-SQL Engine."""

    model_config = SettingsConfigDict(
        env_prefix="TEXT2SQL_",
        env_file=".env",
        extra="ignore",
    )

    # Environment mode: 'development', 'production', or 'test'
    env: str = Field(default="development", description="Runtime environment ('development', 'production', 'test')")

    # Embeddings Configuration (BAAI/bge-small-en-v1.5)
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local HuggingFace embedding model for schema retrieval",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for embedding execution ('cpu' or 'cuda')",
    )

    # ChromaDB Configuration
    chroma_db_dir: str = Field(
        default="./chroma_db",
        description="Local directory for persistent ChromaDB storage",
    )

    # LLM Backend Selection: 'auto', 'ollama', 'huggingface', 'mock'
    llm_backend: str = Field(
        default="auto",
        description="Selected LLM inference backend ('auto', 'ollama', 'huggingface', 'mock')",
    )

    # LLM / SQL Coder Models
    llm_model: str = Field(
        default="Qwen/Qwen2.5-Coder-32B-Instruct",
        description="Target Qwen coder model name (HuggingFace repo or Ollama target)",
    )
    ollama_model: str = Field(
        default="qwen2.5-coder:7b",
        description="Local Ollama model tag for offline local inference",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    hf_api_token: Optional[str] = Field(
        default=None,
        description="Optional Hugging Face Hub token (never logged)",
    )

    # Local Database Engine
    duckdb_path: str = Field(
        default="./data/warehouse.duckdb",
        description="Path to DuckDB database file",
    )


_settings: Optional[EngineSettings] = None
_embeddings_instance: Optional[HuggingFaceEmbeddings] = None


def get_settings() -> EngineSettings:
    """Get or initialize singleton engine settings."""
    global _settings
    if _settings is None:
        _settings = EngineSettings()
    return _settings


def reset_settings(new_settings: Optional[EngineSettings] = None) -> None:
    """Reset singleton settings (used in test fixtures)."""
    global _settings
    _settings = new_settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return cached HuggingFaceEmbeddings instance running locally."""
    global _embeddings_instance
    if _embeddings_instance is None:
        settings = get_settings()
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance


def get_chroma_client(persist_directory: Optional[str] = None, in_memory: bool = False) -> chromadb.ClientAPI:
    """Initialize ChromaDB client (persistent or in-memory)."""
    settings = get_settings()
    if in_memory:
        return chromadb.EphemeralClient()
    target_dir = persist_directory or settings.chroma_db_dir
    os.makedirs(target_dir, exist_ok=True)
    return chromadb.PersistentClient(path=target_dir)


def get_vector_store(
    collection_name: str,
    persist_directory: Optional[str] = None,
    in_memory: bool = False,
) -> Chroma:
    """Initialize LangChain Chroma vector store with local BAAI embeddings."""
    embeddings = get_embeddings()
    client = get_chroma_client(persist_directory=persist_directory, in_memory=in_memory)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


class DeterministicMockCoder(BaseChatModel):
    """Explicit test-only deterministic SQL coder mock.
    
    Generates realistic, schema-valid LogicPlan JSON responses for unit testing and offline
    verification without requiring network access, paid API keys, or GPU resources.
    Never misrepresents itself as actual Qwen inference.
    """

    model_name: str = "mock-deterministic-qwen-coder"
    backend_name: str = "mock"
    is_mock: bool = True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_text = messages[-1].content if messages else ""
        response_json = self._synthesize_mock_json(prompt_text)
        message = AIMessage(content=response_json)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock_deterministic_coder"

    def _synthesize_mock_json(self, prompt: str) -> str:
        # Extract user query line from prompt to avoid matching keywords in rules
        p_lower = prompt.lower()
        user_query = ""
        if "USER QUERY:" in prompt:
            user_query = prompt.split("USER QUERY:", 1)[1].split("RULES:", 1)[0].strip()
        q_lower = user_query.lower() if user_query else p_lower

        # Extract target dialect from prompt if present
        target_dialect = "duckdb"
        if "TARGET DIALECT: bigquery" in prompt or "target dialect: bigquery" in prompt.lower():
            target_dialect = "bigquery"
        elif "TARGET DIALECT: postgres" in prompt or "target dialect: postgres" in prompt.lower():
            target_dialect = "postgres"
        elif "TARGET DIALECT: snowflake" in prompt or "target dialect: snowflake" in prompt.lower():
            target_dialect = "snowflake"

        # Case: Cohort Retention (4-stage CTE plan)
        if "retention" in q_lower or "cohort" in q_lower:
            plan_data = {
                "query": "Calculate four-week cohort retention rates",
                "target_dialect": target_dialect,
                "strategy": "Deconstruct cohort retention into: 1) user signup cohorts, 2) cohort sizes, 3) user weekly activities, 4) retention aggregation with safe division.",
                "cte_steps": [
                    {
                        "step_name": "user_cohorts",
                        "purpose": "Group users into signup cohorts based on their signup week",
                        "source_tables": ["user_signups"],
                        "math_logic": "CAST(DATE_TRUNC('week', signup_date) AS DATE) AS cohort_week",
                        "filter_conditions": [],
                        "group_by_columns": [],
                        "window_logic": None,
                        "columns_produced": ["user_id", "cohort_week"],
                        "sql_fragment": "SELECT user_id, CAST(DATE_TRUNC('week', signup_date) AS DATE) AS cohort_week FROM user_signups",
                    },
                    {
                        "step_name": "cohort_sizes",
                        "purpose": "Compute the total distinct user denominator for each cohort week",
                        "source_tables": ["user_cohorts"],
                        "math_logic": "COUNT(DISTINCT user_id) AS cohort_size",
                        "filter_conditions": [],
                        "group_by_columns": ["cohort_week"],
                        "window_logic": None,
                        "columns_produced": ["cohort_week", "cohort_size"],
                        "sql_fragment": "SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_size FROM user_cohorts GROUP BY cohort_week",
                    },
                    {
                        "step_name": "weekly_activities",
                        "purpose": "Calculate activity elapsed weeks relative to cohort signup week",
                        "source_tables": ["activity_events", "user_cohorts"],
                        "math_logic": "DATE_DIFF('week', CAST(uc.cohort_week AS DATE), CAST(DATE_TRUNC('week', a.event_date) AS DATE)) AS week_number",
                        "filter_conditions": ["a.event_date >= uc.cohort_week"],
                        "group_by_columns": ["a.user_id", "uc.cohort_week"],
                        "window_logic": None,
                        "columns_produced": ["user_id", "cohort_week", "week_number"],
                        "sql_fragment": "SELECT a.user_id, uc.cohort_week, CAST(DATE_DIFF('week', CAST(uc.cohort_week AS DATE), CAST(DATE_TRUNC('week', a.event_date) AS DATE)) AS INTEGER) AS week_number FROM activity_events AS a JOIN user_cohorts AS uc ON a.user_id = uc.user_id WHERE a.event_date >= uc.cohort_week GROUP BY a.user_id, uc.cohort_week, DATE_TRUNC('week', a.event_date)",
                    },
                    {
                        "step_name": "cohort_retention_agg",
                        "purpose": "Aggregate active users by cohort week and week number up to 4 weeks",
                        "source_tables": ["weekly_activities"],
                        "math_logic": "COUNT(DISTINCT wa.user_id) AS active_users",
                        "filter_conditions": ["wa.week_number <= 4"],
                        "group_by_columns": ["wa.cohort_week", "wa.week_number"],
                        "window_logic": None,
                        "columns_produced": ["cohort_week", "week_number", "active_users"],
                        "sql_fragment": "SELECT wa.cohort_week, wa.week_number, COUNT(DISTINCT wa.user_id) AS active_users FROM weekly_activities AS wa WHERE wa.week_number <= 4 GROUP BY wa.cohort_week, wa.week_number",
                    },
                ],
                "final_select": "SELECT cs.cohort_week, cra.week_number, cs.cohort_size, cra.active_users, ROUND(cra.active_users * 1.0 / NULLIF(cs.cohort_size, 0), 4) AS retention_rate FROM cohort_retention_agg AS cra JOIN cohort_sizes AS cs ON cra.cohort_week = cs.cohort_week ORDER BY cs.cohort_week, cra.week_number",
                "validation_notes": "Safe division applied with NULLIF on cohort_size denominator; duplicate events grouped by user_id and week.",
            }
            return json.dumps(plan_data, indent=2)

        # Case: Top customers / Safe division
        if "spend" in q_lower or "division" in q_lower or "ratio" in q_lower:
            plan_data = {
                "query": "Find top 3 customers ranked by total spend with safe division handling",
                "target_dialect": target_dialect,
                "strategy": "Aggregate customer orders and calculate shipping ratio guarded against division-by-zero",
                "cte_steps": [
                    {
                        "step_name": "customer_spend",
                        "purpose": "Aggregate order revenue and compute safe shipping ratio per customer",
                        "source_tables": ["orders"],
                        "math_logic": "SUM(orders.total_amount) AS total_spent, ROUND(SUM(orders.shipping_cost) * 1.0 / NULLIF(SUM(orders.total_amount), 0), 4) AS shipping_ratio",
                        "filter_conditions": [],
                        "group_by_columns": ["orders.customer_id"],
                        "window_logic": None,
                        "columns_produced": ["customer_id", "total_spent", "shipping_ratio"],
                        "sql_fragment": "SELECT orders.customer_id, SUM(orders.total_amount) AS total_spent, ROUND(SUM(orders.shipping_cost) * 1.0 / NULLIF(SUM(orders.total_amount), 0), 4) AS shipping_ratio FROM orders GROUP BY orders.customer_id",
                    }
                ],
                "final_select": "SELECT customer_id, total_spent, shipping_ratio FROM customer_spend ORDER BY total_spent DESC LIMIT 3",
                "validation_notes": "Guarded total_amount with NULLIF(..., 0)",
            }
            return json.dumps(plan_data, indent=2)

        # Case: California Purchases Ranking
        if "california" in q_lower and "order" in q_lower:
            plan_data = {
                "query": "Show purchases in California and rank orders by total amount",
                "target_dialect": target_dialect,
                "strategy": "Join orders to customers, filter by literal state code 'CA', and compute row ranking",
                "cte_steps": [
                    {
                        "step_name": "ranked_orders",
                        "purpose": "Filter California orders and assign window ranking",
                        "source_tables": ["orders", "customers"],
                        "math_logic": "ROW_NUMBER() OVER (ORDER BY orders.total_amount DESC) AS rank_order",
                        "filter_conditions": ["customers.state_code = 'CA'"],
                        "group_by_columns": [],
                        "window_logic": "ROW_NUMBER() OVER (ORDER BY orders.total_amount DESC)",
                        "columns_produced": ["order_id", "customer_id", "total_amount", "rank_order"],
                        "sql_fragment": "SELECT orders.order_id, orders.customer_id, orders.total_amount, ROW_NUMBER() OVER (ORDER BY orders.total_amount DESC) AS rank_order FROM orders JOIN customers ON orders.customer_id = customers.customer_id WHERE customers.state_code = 'CA'",
                    }
                ],
                "final_select": "SELECT order_id, customer_id, total_amount, rank_order FROM ranked_orders WHERE rank_order <= 5 ORDER BY rank_order ASC",
                "validation_notes": "Mapped California entity to state_code = 'CA'",
            }
            return json.dumps(plan_data, indent=2)

        # Case: GA4 nested unnest
        if "web" in q_lower or "campaign" in q_lower or "event" in q_lower:
            plan_data = {
                "query": "What campaigns were associated with web events in California?",
                "target_dialect": target_dialect,
                "strategy": "Unnest GA4 event_params array and project campaign parameters for California events",
                "cte_steps": [
                    {
                        "step_name": "unnested_events",
                        "purpose": "Unnest event_params array and filter on campaign and California region",
                        "source_tables": ["web_events"],
                        "math_logic": "UNNEST(event_params) projection",
                        "filter_conditions": ["geo_info.region = 'California'", "ep.key = 'campaign'"],
                        "group_by_columns": [],
                        "window_logic": None,
                        "columns_produced": ["event_id", "city", "region", "param_key", "param_val"],
                        "sql_fragment": "SELECT event_id, geo_info.city AS city, geo_info.region AS region, ep.key AS param_key, ep.value.string_value AS param_val FROM web_events JOIN UNNEST(event_params) AS t(ep) ON TRUE WHERE geo_info.region = 'California' AND ep.key = 'campaign'",
                    }
                ],
                "final_select": "SELECT event_id, city, region, param_key, param_val FROM unnested_events ORDER BY event_id ASC",
                "validation_notes": "UNNEST array syntax with dot notation",
            }
            return json.dumps(plan_data, indent=2)

        # Case: Days elapsed between orders
        if "elapsed" in q_lower or "days between" in q_lower or "date_diff" in q_lower:
            diff_expr = (
                "DATE_DIFF(CURRENT_DATE, orders.order_date, DAY)"
                if target_dialect == "bigquery"
                else "date_diff('day', orders.order_date, CURRENT_DATE)"
            )
            plan_data = {
                "query": "Calculate days elapsed between orders and rank top transactions",
                "target_dialect": target_dialect,
                "strategy": "Calculate date difference from order_date to current date and rank top orders",
                "cte_steps": [
                    {
                        "step_name": "elapsed_orders",
                        "purpose": "Compute elapsed days per order using dialect date diff",
                        "source_tables": ["orders"],
                        "math_logic": f"{diff_expr} AS days_elapsed",
                        "filter_conditions": [],
                        "group_by_columns": [],
                        "window_logic": None,
                        "columns_produced": ["order_id", "customer_id", "total_amount", "days_elapsed"],
                        "sql_fragment": f"SELECT orders.order_id, orders.customer_id, orders.total_amount, {diff_expr} AS days_elapsed FROM orders",
                    }
                ],
                "final_select": "SELECT order_id, customer_id, total_amount, days_elapsed FROM elapsed_orders ORDER BY total_amount DESC LIMIT 5",
                "validation_notes": f"Applied {target_dialect} date difference syntax",
            }
            return json.dumps(plan_data, indent=2)

        # Default fallback plan
        plan_data = {
            "query": "Analytical query",
            "target_dialect": target_dialect,
            "strategy": "Single CTE aggregation plan",
            "cte_steps": [
                {
                    "step_name": "base_records",
                    "purpose": "Select and project base rows",
                    "source_tables": ["orders"],
                    "math_logic": "COUNT(order_id) AS total_orders",
                    "filter_conditions": [],
                    "group_by_columns": ["customer_id"],
                    "window_logic": None,
                    "columns_produced": ["customer_id", "total_orders"],
                    "sql_fragment": "SELECT customer_id, COUNT(order_id) AS total_orders FROM orders GROUP BY customer_id",
                }
            ],
            "final_select": "SELECT customer_id, total_orders FROM base_records ORDER BY total_orders DESC LIMIT 10",
            "validation_notes": "Standard aggregation",
        }
        return json.dumps(plan_data, indent=2)


class OllamaCoder(BaseChatModel):
    """Direct HTTP Ollama ChatModel client avoiding deprecated community wrappers."""

    model_name: str = "qwen2.5-coder:7b"
    base_url: str = "http://localhost:11434"
    backend_name: str = "ollama"
    is_mock: bool = False
    temperature: float = 0.0

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system" if m.type == "system" else "user", "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        url = f"{self.base_url.rstrip('/')}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                message = AIMessage(content=content)
                return ChatResult(generations=[ChatGeneration(message=message)])
        except Exception as e:
            raise InferenceError(f"Ollama inference failed at {self.base_url}: {e}") from e

    @property
    def _llm_type(self) -> str:
        return "ollama_coder"


def check_ollama_availability(base_url: str, model_name: str) -> bool:
    """Checks whether the Ollama server is reachable and has the requested model available."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", headers={"User-Agent": "text2sql"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                # Return True if server responds
                return True
    except Exception:
        return False
    return False


def get_llm(
    backend: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Explicitly initializes and returns the configured LLM backend.

    Args:
        backend: Explicit backend name override ('mock', 'ollama', 'huggingface', 'auto').
        temperature: Sampling temperature for generation.

    Returns:
        BaseChatModel instance.

    Raises:
        ModelUnavailableError: If requested real model or server is not reachable.
        ConfigurationError: If required tokens/settings are missing.
    """
    settings = get_settings()
    target_backend = (backend or settings.llm_backend).lower().strip()

    # 1. Explicit Mock Backend (Test-Only)
    if target_backend == "mock":
        return DeterministicMockCoder()

    # 2. Explicit Ollama Backend
    if target_backend == "ollama":
        if not check_ollama_availability(settings.ollama_base_url, settings.ollama_model):
            raise ModelUnavailableError(
                f"Ollama backend requested, but server at '{settings.ollama_base_url}' is not reachable. "
                f"Please start Ollama (`ollama serve`) and ensure '{settings.ollama_model}' is pulled."
            )
        return OllamaCoder(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

def is_valid_token(token: Optional[str]) -> bool:
    """Checks whether a token string is present and not a template placeholder."""
    if not token:
        return False
    t = token.strip().lower()
    if t in ("", "none", "null", "hf_placeholder_token_do_not_commit", "your_token_here") or "placeholder" in t:
        return False
    return True


def get_llm(
    backend: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Explicitly initializes and returns the configured LLM backend.

    Args:
        backend: Explicit backend name override ('mock', 'ollama', 'huggingface', 'auto').
        temperature: Sampling temperature for generation.

    Returns:
        BaseChatModel instance.

    Raises:
        ModelUnavailableError: If requested real model or server is not reachable.
        ConfigurationError: If required tokens/settings are missing.
    """
    settings = get_settings()
    target_backend = (backend or settings.llm_backend).lower().strip()

    # 1. Explicit Mock Backend (Test-Only)
    if target_backend == "mock":
        return DeterministicMockCoder()

    # 2. Explicit Ollama Backend
    if target_backend == "ollama":
        if not check_ollama_availability(settings.ollama_base_url, settings.ollama_model):
            raise ModelUnavailableError(
                f"Ollama backend requested, but server at '{settings.ollama_base_url}' is not reachable. "
                f"Please start Ollama (`ollama serve`) and ensure '{settings.ollama_model}' is pulled."
            )
        return OllamaCoder(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    # 3. Explicit Hugging Face Backend
    if target_backend == "huggingface":
        raw_token = settings.hf_api_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
        token = raw_token if is_valid_token(raw_token) else None
        if not token:
            raise ConfigurationError(
                "Hugging Face backend requested, but no API token provided. "
                "Set TEXT2SQL_HF_API_TOKEN or HF_TOKEN environment variable."
            )

        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            endpoint = HuggingFaceEndpoint(
                repo_id=settings.llm_model,
                huggingfacehub_api_token=token,
                temperature=temperature,
            )
            chat_model = ChatHuggingFace(llm=endpoint)
            chat_model.backend_name = "huggingface"
            chat_model.is_mock = False
            return chat_model
        except Exception as e:
            raise ModelUnavailableError(f"Failed to initialize Hugging Face model endpoint for '{settings.llm_model}': {e}") from e

    # 4. Auto Backend Resolution
    if target_backend == "auto":
        # Check Ollama first
        if check_ollama_availability(settings.ollama_base_url, settings.ollama_model):
            return OllamaCoder(
                model_name=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=temperature,
            )

        # Check Hugging Face token (only if not a placeholder)
        raw_token = settings.hf_api_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
        token = raw_token if is_valid_token(raw_token) else None
        if token:
            try:
                from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
                endpoint = HuggingFaceEndpoint(
                    repo_id=settings.llm_model,
                    huggingfacehub_api_token=token,
                    temperature=temperature,
                )
                chat_model = ChatHuggingFace(llm=endpoint)
                chat_model.backend_name = "huggingface"
                chat_model.is_mock = False
                return chat_model
            except Exception:
                pass

        # In production mode, fail closed if no real LLM is configured
        if settings.env == "production":
            raise ModelUnavailableError(
                "No active LLM backend available in production environment. "
                "Please configure Ollama with Qwen 2.5 Coder or set Hugging Face API token."
            )

        # In development / test mode, use the explicit test mock
        return DeterministicMockCoder()


    raise ConfigurationError(
        f"Unknown LLM backend '{target_backend}'. Allowed backends: 'auto', 'ollama', 'huggingface', 'mock'."
    )
