# Enterprise Text-to-SQL Engine (Spider 2.0 Benchmark Architecture)

An open-source, zero-cost Enterprise Text-to-SQL system built from scratch using a strict problem-by-problem engineering methodology addressing the primary real-world failure modes identified in the **Spider 2.0 benchmark**:
1. **Erroneous Data Analysis (35.5% of errors)**: Multi-stage reasoning failures, complex window functions, division-by-zero, and dialect function syntax misuse.
2. **Wrong Schema Linking (27.6% of errors)**: 700+ column context window overflows, entity literal mismatches, and nested data structures (`STRUCT`, `ARRAY`, `JSON`).

---

## 1. Zero-Cost Open-Source Stack

- **Agent Orchestration**: LangGraph (v0.2+) & LangChain (v0.3+)
- **Embeddings**: `langchain-huggingface` (`BAAI/bge-small-en-v1.5` running locally on CPU/GPU)
- **Vector Store**: Chroma DB (in-memory & persistent `./chroma_db`)
- **LLM / SQL Coder**: `Qwen/Qwen2.5-Coder-32B-Instruct` (via local Ollama `qwen2.5-coder:7b`, Hugging Face API, or built-in deterministic AST synthesizer for zero-cost offline validation)
- **SQL Parsing & AST Transpilation**: SQLGlot
- **Local Database Engine**: DuckDB / SQLite
- **Data Validation**: Pydantic v2

---

## 2. Repository Structure

```
text2sql_engine/
├── core/
│   ├── __init__.py
│   ├── config.py             # HuggingFace & ChromaDB initializations, LLM backend provider
│   └── state.py              # LangGraph SQLAgentState definition
├── modules/
│   ├── __init__.py
│   ├── problem1_planner/     # Problem 1: CTE Logic Planner & Dialect Function RAG
│   │   ├── __init__.py
│   │   ├── schemas.py        # Pydantic schemas (DialectFunctionSpec, CTEPlanStep, LogicPlan)
│   │   └── logic_planner.py  # LogicPlanner, DialectFunctionRAG, SQLGlot compiler
│   └── problem2_retriever/   # Problem 2: Two-Stage Schema Retriever & Nested Flattener
│       ├── __init__.py
│       ├── schema_flattener.py  # Recursive STRUCT / ARRAY / JSON dot-notation flattener
│       └── vector_retriever.py  # Two-Stage Retriever (Coarse Table + Fine Field) & Value Indexer
├── tests/
│   ├── __init__.py
│   ├── test_problem1.py      # Unit tests for CTE planner, Dialect RAG, AST compilation
│   └── test_problem2.py      # Unit tests for schema flattener, value indexer, DuckDB execution
├── data/
│   ├── setup_sample_db.py    # DuckDB generator with relational and nested STRUCT/ARRAY tables
│   └── warehouse.duckdb      # DuckDB database file
└── main.py                   # LangGraph StateGraph pipeline orchestrator
```

---

## 3. Problem 1: Solving Erroneous Data Analysis (35.5% of Errors)

### The Failure Mode
Standard text-to-SQL models generate monolithic queries that collapse when handling:
- Multi-stage cohort aggregations
- Complex window functions (`ROW_NUMBER`, `DENSE_RANK`, `LEAD`, `LAG`)
- Dialect function misuses (e.g. BigQuery `DATE_DIFF(end, start, DAY)` vs Snowflake `DATEDIFF(day, start, end)` vs DuckDB `date_diff('day', start, end)`)
- Division-by-zero crashes when computing ratios and percentages

### The Solution (`modules/problem1_planner/`)
1. **Pydantic Contract Planning (`schemas.py`)**:
   Deconstructs analytical intent into an explicit Directed Acyclic Graph (DAG) of Common Table Expressions (`CTEPlanStep`), each with clear inputs, outputs, math logic, and filter clauses before any SQL is written.
2. **Dialect Function RAG (`logic_planner.py`)**:
   A curated knowledge catalog indexing dialect-specific syntax rules for DuckDB, BigQuery, Snowflake, and PostgreSQL. Automatically injects verified function patterns into the prompt.
3. **Safe Math & AST Compilation**:
   Guards all divisions with `NULLIF(col, 0)`. Validates AST and transpiles queries across dialects using SQLGlot.

---

## 4. Problem 2: Solving Wrong Schema Linking (27.6% of Errors)

### The Failure Mode
Enterprise databases contain dozens of tables and 700+ columns. Models suffer from:
- Context window overflow and hallucinated column names
- Inability to reference nested `STRUCT`, `ARRAY`, and `JSON` fields (e.g. GA4 `event_params`)
- Mismatched filter literals (user queries "purchases in California", database stores `state_code = 'CA'`)

### The Solution (`modules/problem2_retriever/`)
1. **Schema Flattener (`schema_flattener.py`)**:
   Recursively parses nested `STRUCT(...)` and `ARRAY(STRUCT(...))` definitions into dot paths (e.g., `event_params.key`, `event_params.value.string_value`, `geo_info.city`) and marks whether `UNNEST` is required.
2. **Two-Stage Schema Retriever (`vector_retriever.py`)**:
   - **Stage 1 (Coarse Table Filter)**: Prunes 700+ column schemas down to the top 3–5 candidate tables using `BAAI/bge-small-en-v1.5` embeddings.
   - **Stage 2 (Fine-Grained Field Selection)**: Pulls only relevant columns and nested paths, producing compact schema cards.
3. **Distinct Value Indexer (`vector_retriever.py`)**:
   Indexes sample categorical cell literals (e.g. `("CA", "California")`, `("COMPLETED", "Completed")`), automatically mapping natural language entities directly to database filter predicates (`state_code = 'CA'`).

---

## 5. Running the Pipeline & Test Suite

### Run All Unit Tests
```bash
python -m pytest text2sql_engine/tests -v
```

### Run Pipeline End-to-End Orchestrator
```bash
python main.py
```

### Sample Output
```
--- [Query 1] 'Find the top 3 customers ranked by total spend with safe division handling' (Target Dialect: DUCKDB) ---
[Candidate Tables Selected]: ['customers', 'orders', 'order_items']
[Compiled & Transpiled SQL]:
WITH customer_spend AS (
  SELECT
    orders.customer_id,
    SUM(orders.total_amount) AS total_spent,
    ROUND(SUM(orders.shipping_cost) * 1.0 / NULLIF(SUM(orders.total_amount), 0), 4) AS shipping_ratio
  FROM orders
  GROUP BY
    orders.customer_id
)
SELECT customer_id, total_spent, shipping_ratio FROM customer_spend ORDER BY total_spent DESC LIMIT 3

[Execution Result (3 rows)]:
  {'customer_id': 'C005', 'total_spent': 510.0, 'shipping_ratio': 0.049}
  {'customer_id': 'C001', 'total_spent': 430.0, 'shipping_ratio': 0.0581}
  {'customer_id': 'C002', 'total_spent': 420.0, 'shipping_ratio': 0.0476}
```
