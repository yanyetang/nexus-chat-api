# RAG Gap Analysis & Implementation Plan

## Gaps to Close (from `project-dropship`)

### 1. Live Inventory Enrichment

After vector retrieval, re-fetch real-time stock levels, variant availability, and current prices from the supplier API and inject fresh state into the LLM prompt.

- Add a post-retrieval enrichment step in `rag.py` `get_context()` that calls `SupplierService` with the retrieved `product_id` list
- Append live stock/price data to each chunk before `build_context_block()` formats it
- Fall back gracefully if the supplier API is unreachable (use cached chunk_text only)

### 2. Similarity Threshold Gating

Filter out low-quality matches before building LLM context.

- Add a `min_score: float = 0.3` parameter to `RetrieverService.hybrid_search()`
- Drop results where RRF score is below threshold before returning
- In `rag.py`, emit the `info` SSE event (already wired) when all results are filtered out

### 3. Graceful Embedding Fallback

Keep the chat endpoint live when Cohere is unavailable.

- Wrap `EmbeddingService.embed_query()` in a try/except in `rag.py`
- On failure, fall back to keyword-only search by calling `RetrieverService` with a null embedding and skipping the semantic branch in the SQL

### 4. Explicit Vector Index

Ensure ANN index exists so queries don't degrade to sequential scans at scale.

- Add a migration/init script that runs:
  ```sql
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_product_embeddings_vector
    ON product_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
  ```
- Wire this into the app startup `lifespan` or a one-time migration script

### 5. DB-Level FTS Column Maintenance

Move tsvector computation from query-time to write-time.

- Add a `content_tsv TSVECTOR` column to `product_embeddings`
- Add a PostgreSQL trigger: `BEFORE INSERT OR UPDATE` → `to_tsvector('english', chunk_text)`
- Add a GIN index on `content_tsv`
- Update the hybrid search SQL in `retriever.py` to query the pre-computed column

---

## Strengths to Preserve

- **Multilingual embeddings** — keep `embed-multilingual-v3.0`; do not regress to the English-only model
- **Consistent hybrid RRF on both `/chat` and `/search`** — do not split these as `project-dropshi` did
- **Strict RAG grounding** in the system prompt — no general knowledge fallback
- **Typed SSE event protocol** (`sources` → `token*` → `done`) — keep source attribution
- **CI pipeline** — ruff + pyright + pytest; extend coverage to retrieval logic

---

## New Capabilities to Unlock (Python-only)

### 6. Cohere Rerank (Cross-Encoder Reranking)

Replace RRF with a cross-encoder reranker for final result ordering.

- After hybrid_search returns top-20 candidates, call `cohere.rerank()` with `model="rerank-multilingual-v3.0"`
- Return top-5 reranked results instead of RRF-fused top-5
- RRF can be kept as the candidate generation step (fetch 20), rerank applied as the final step

### 7. Metadata Filtering

Allow filtering by category, brand, or price range at retrieval time.

- Add optional `filters: dict` to `SearchRequest` and `ChatRequest`
- Extend `RetrieverService.hybrid_search()` to add `WHERE metadata->>'category' = $n` predicates dynamically

### 8. RAG Evaluation Harness (DeepEval)

Measure retrieval and generation quality using **DeepEval** as the evaluation engine.

- Add `deepeval` to `requirements-dev.txt`
- Add `tests/eval/test_rag_quality.py` with the following DeepEval metrics:
  - `ContextualPrecision` — are retrieved chunks ranked correctly relative to the query?
  - `ContextualRecall` — do retrieved chunks cover the expected answer?
  - `Faithfulness` — does the LLM response stay grounded in the retrieved context?
  - `AnswerRelevancy` — is the final answer relevant to the user's question?
- Define a golden dataset in `tests/eval/golden_dataset.json`: list of `{input, expected_output, expected_product_ids}` objects covering representative product queries
- Wire DeepEval's `@pytest.mark.deepeval` decorator and `assert_test()` so failures produce structured metric reports
- Run as a nightly CI job via a separate GitHub Actions workflow (`.github/workflows/eval.yml`), not on every push
- Emit a summary comment on PRs that touch `retriever.py`, `rag.py`, or `prompts.py` using `deepeval login` + CI integration

### 9. DSPy Training / Optimization Engine

Use **DSPy** to programmatically optimize the RAG pipeline — prompts, retrieval parameters, and chain-of-thought reasoning — rather than hand-tuning them.

**Pipeline definition** (`app/optimization/pipeline.py`):
- Model the RAG flow as a DSPy program with discrete modules:
  ```
  DSPyRAGPipeline
  ├── Retrieve      → wraps RetrieverService.hybrid_search()
  ├── Rerank        → wraps Cohere rerank (section 6)
  └── Generate      → wraps LLMService with a DSPy Signature
  ```
- Define a `Signature` for generation: `context, history, query → answer` with typed fields and docstrings DSPy uses for prompt synthesis

**Optimizer** (`app/optimization/optimize.py`):
- Use `dspy.MIPROv2` (or `BootstrapFewShot` for simpler runs) with the golden dataset from section 8 as the training set
- Use DeepEval `Faithfulness` + `AnswerRelevancy` scores as the DSPy metric function so both tools share the same ground truth
- Save the compiled (optimized) program to `artifacts/optimized_pipeline.json` via `pipeline.save()`
- Load the compiled program at app startup if the artifact exists; fall back to the un-optimized default otherwise

**Workflow**:
- Add a one-off script `scripts/run_optimization.py` that loads the golden dataset, runs the optimizer, and writes the artifact
- Run optimization manually or as a scheduled CI job (weekly) — not on every push
- Add `dspy-ai` to `requirements.txt` (runtime, since the compiled prompts are loaded at startup)

**Key constraint**: DSPy must not block the request path. Optimization runs offline; only artifact loading happens at startup.

### 10. Background Async Re-indexing

Replace synchronous blocking ingest with a background task.

- Use FastAPI `BackgroundTasks` or ARQ for long-running ingest jobs
- Return a `job_id` immediately from `POST /ingest`, expose `GET /ingest/{job_id}/status`

---

## Priority Order

| Priority | Gap / Feature                        | Effort |
| -------- | ------------------------------------ | ------ |
| P0       | Similarity threshold gating          | Low    |
| P0       | Explicit vector index migration      | Low    |
| P1       | Live inventory enrichment            | Medium |
| P1       | Graceful embedding fallback          | Low    |
| P2       | DB-level FTS column + trigger        | Medium |
| P2       | Cohere Rerank integration            | Medium |
| P3       | Metadata filtering                   | Medium |
| P3       | Background async re-indexing         | Medium |
| P4       | RAG evaluation harness (DeepEval)    | High   |
| P4       | DSPy optimization engine             | High   |

### Dependency note
DSPy optimization (section 9) **depends on** the DeepEval golden dataset and metric functions (section 8) — implement evaluation first, then wire DeepEval metrics as the DSPy training signal.
