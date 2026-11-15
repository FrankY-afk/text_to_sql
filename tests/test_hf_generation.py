"""Unit tests for Hugging Face LLM generation integration (Problem 1, Phase E)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from src.cte_planner import CTEPlanner
from src.hf_client import HuggingFaceClient, HuggingFaceInferenceError
from src.sql_generator import SQLGenerator
from src.sql_validator import validate_readonly_sql


def test_hf_extract_sql_various_formats():
    """Verifies that SQL is cleanly extracted from various LLM response formats."""
    # 1. Standard markdown code fence
    res1 = """Here is the SQL query:
```sql
WITH jan_cohort AS (
    SELECT user_id FROM users
)
SELECT * FROM jan_cohort;
```
Hope this helps!"""
    assert HuggingFaceClient.extract_sql(res1) == "WITH jan_cohort AS (\n    SELECT user_id FROM users\n)\nSELECT * FROM jan_cohort;"

    # 2. Generic code fence without 'sql'
    res2 = """```
SELECT outlet_code FROM outlets LIMIT 5;
```"""
    assert HuggingFaceClient.extract_sql(res2) == "SELECT outlet_code FROM outlets LIMIT 5;"

    # 3. Raw SQL without code fences
    res3 = "SELECT staff_no, full_name FROM staff WHERE position = 'Officer';"
    assert HuggingFaceClient.extract_sql(res3) == "SELECT staff_no, full_name FROM staff WHERE position = 'Officer';"


def test_build_generation_prompt():
    """Verifies that the generation prompt integrates CTE plan, dialect, schema, and literals."""
    generator = SQLGenerator()
    q = "What percentage of users who signed up in January returned in February?"
    plan = generator.planner.create_cohort_retention_plan(q, target_dialect="postgres")

    prompt = generator.build_generation_prompt(
        question=q,
        target_dialect="postgres",
        plan=plan,
        schema_context="Table: users (user_id, signup_date)\nTable: activity_events (event_id, user_id, event_date)",
        literal_context="[Discovered Database Literals]: None",
        dialect_context="Target SQL Dialect Knowledge: [POSTGRES]",
    )

    assert "TARGET SQL DIALECT: POSTGRES" in prompt
    assert q in prompt
    assert "jan_signup_cohort" in prompt
    assert "distinct_jan_users" in prompt
    assert "NULLIF(denominator, 0)" in prompt


def test_generator_improved_sql_with_mock_hf():
    """Verifies that SQLGenerator can use HuggingFaceClient to generate SQL."""
    mock_hf = MagicMock(spec=HuggingFaceClient)
    mock_hf.is_configured.return_value = True
    mock_hf.model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

    simulated_sql = (
        "WITH jan_cohort AS (\n"
        "    SELECT user_id FROM users WHERE signup_date >= '2024-01-01' AND signup_date < '2024-02-01'\n"
        ")\n"
        "SELECT COUNT(DISTINCT user_id) AS total FROM jan_cohort;"
    )
    mock_hf.generate_sql.return_value = simulated_sql

    generator = SQLGenerator(hf_client=mock_hf)
    q = "What percentage of users who signed up in January returned in February?"
    res = generator.generate_improved_sql(q, target="local", use_hf=True)

    assert res.strategy == "improved"
    assert "HUGGING FACE" in res.explanation
    assert res.generated_sql == simulated_sql
    assert res.logic_plan is not None
    mock_hf.generate_sql.assert_called_once()


def test_generator_fallback_when_hf_fails():
    """Verifies that SQLGenerator falls back to the verified CTE compiler when Hugging Face is unreachable."""
    mock_hf = MagicMock(spec=HuggingFaceClient)
    mock_hf.is_configured.return_value = True
    mock_hf.model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
    mock_hf.generate_sql.side_effect = HuggingFaceInferenceError("Network unreachable (offline)")

    generator = SQLGenerator(hf_client=mock_hf)
    q = "What percentage of users who signed up in January returned in February?"
    res = generator.generate_improved_sql(q, target="local", use_hf=True)

    assert res.strategy == "improved"
    assert "IMPROVED CTE PLANNER" in res.explanation
    assert "jan_signup_cohort" in res.generated_sql
    assert "distinct_jan_users" in res.generated_sql
    assert res.logic_plan is not None


def test_hf_api_key_resolution_across_env_vars(monkeypatch):
    """Verifies that Hugging Face API key is resolved from HF_API_KEY, HF_TOKEN, or TEXT2SQL_HF_API_TOKEN."""
    from src.config import Settings, reset_settings

    # 1. Test HF_API_KEY env var
    monkeypatch.setenv("HF_API_KEY", "hf_test_api_key_123")
    settings = Settings()
    assert settings.get_hf_token() == "hf_test_api_key_123"

    # 2. Test HF_TOKEN env var
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_token_456")
    settings = Settings()
    assert settings.get_hf_token() == "hf_token_456"

    # 3. Test HUGGINGFACE_API_KEY env var
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_key_789")
    settings = Settings()
    assert settings.get_hf_token() == "hf_key_789"

    # 4. Test TEXT2SQL_HF_API_TOKEN env var
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.setenv("TEXT2SQL_HF_API_TOKEN", "hf_text2sql_token")
    settings = Settings()
    assert settings.get_hf_token() == "hf_text2sql_token"

    # 5. Verify HuggingFaceClient is_configured() returns True with resolved API key
    client = HuggingFaceClient(settings=settings)
    assert client.is_configured() is True
    assert client.token == "hf_text2sql_token"


def test_hf_placeholder_keys_are_ignored(monkeypatch):
    """Verifies that placeholder and empty tokens are safely ignored."""
    from src.config import Settings

    for bad_val in ("", "your_hf_api_key_here", "placeholder", "None", "null"):
        monkeypatch.setenv("HF_API_KEY", bad_val)
        monkeypatch.setenv("HF_TOKEN", bad_val)
        settings = Settings()
        assert settings.get_hf_token() is None

    client = HuggingFaceClient(settings=settings)
    assert client.is_configured() is False

