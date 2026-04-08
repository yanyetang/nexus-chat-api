# RAG Comparison: `chatbot-api` vs `project-dropship`

_Analysis date: 2026-04-08_

## Architecture Overview

| Dimension                 | `project-dropship` (TS/Next.js)          | `chatbot-api` (Python/FastAPI)                               |
| ------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| Language                  | TypeScript                               | Python                                                       |
| Embedding model           | `embed-english-v3.0`                     | `embed-multilingual-v3.0`                                    |
| Chat search strategy      | **Vector-only**                          | **Hybrid RRF** (vector + FTS)                                |
| Search endpoint strategy  | Hybrid RRF                               | Hybrid RRF                                                   |
| Similarity threshold      | App-side filter after query              | SQL-level parameter inside query                             |
| Reranking                 | None                                     | Cohere `rerank-multilingual-v3.0` (configurable)             |
| Metadata filtering        | None                                     | Yes: category, brand, price range                            |
| Live inventory enrichment | Yes (per-product via `Promise.all`)      | Yes (batch fetch, graceful fallback)                         |
| Embedding fallback        | Silent — no keyword fallback             | Falls back to keyword-only + emits `info` SSE event          |
| Session / history         | None                                     | PostgreSQL-backed conversation history                       |
| SSE event protocol        | Raw text stream only                     | Typed: `info` → `sources` → `token*` → `done`                |
| Source attribution        | None                                     | Yes (product_id, title, score per result)                    |
| System prompt grounding   | Allows general knowledge fallback        | Strict RAG-only grounding                                    |
| FTS column                | `content_tsv` computed at **query time** | `content_tsv` pre-computed via **trigger**, GIN indexed      |
| Vector index              | Not defined in code                      | `ivfflat` ANN index bootstrapped at startup                  |
| DSPy                      | None                                     | Scaffolded (`pipeline.py`, `optimize.py`) — placeholder only |
| DeepEval                  | None                                     | Not yet implemented                                          |

---

## Key Differences Explained

### 1. Hybrid RRF in the Chat Route

`project-dropship` uses `hybridSearch` (RRF) only in the `/search` endpoint. The `/chat` route
calls `searchSimilarProducts` — vector-only. This means chat quality degrades for keyword-heavy
queries (SKUs, brand names, exact product names).

`chatbot-api` applies the same hybrid RRF pipeline (`RetrieverService.hybrid_search`) to both
`/chat` and `/search`, ensuring consistent retrieval quality across endpoints.

### 2. Multilingual Embeddings

`project-dropship` uses `embed-english-v3.0`. `chatbot-api` uses `embed-multilingual-v3.0`,
which supports 100+ languages with no accuracy regression on English. This matters for
international supplier catalogs or multilingual user bases.

### 3. DB-Level FTS Maintenance

`project-dropship` computes `to_tsvector(...)` inline at query time — expensive at scale.
`chatbot-api` maintains a `content_tsv TSVECTOR` column updated by a PostgreSQL trigger
on insert/update, with a GIN index. Queries hit the pre-computed column directly.

### 4. Retrieval Pipeline Depth

`project-dropship` pipeline: embed → vector search → threshold filter → enrich → LLM

`chatbot-api` pipeline: embed → hybrid RRF → threshold gate (with broad fallback) → rerank → enrich → LLM

Cohere rerank acts as a cross-encoder final ordering pass on top of RRF candidate generation,
which `project-dropship` has no equivalent of.

---

## Gaps Already Closed (vs `plan-ragGapAnalysis.prompt.md`)

All P0–P3 items from the gap analysis are implemented:

| #   | Gap                           | Status  | Location                                                      |
| --- | ----------------------------- | ------- | ------------------------------------------------------------- |
| 1   | Live inventory enrichment     | ✅ Done | `app/services/rag.py` — `_apply_live_enrichment()`            |
| 2   | Similarity threshold gating   | ✅ Done | `app/services/retriever.py` — `min_score` SQL param           |
| 3   | Graceful embedding fallback   | ✅ Done | `app/services/rag.py` — `get_context()` try/except            |
| 4   | Explicit vector index         | ✅ Done | `app/database.py` — `idx_product_embeddings_vector` (ivfflat) |
| 5   | DB-level FTS column + trigger | ✅ Done | `app/database.py` — `trg_product_embeddings_tsv` + GIN index  |
| 6   | Cohere Rerank integration     | ✅ Done | `app/services/rag.py` — `_maybe_rerank()`                     |
| 7   | Metadata filtering            | ✅ Done | `app/services/retriever.py` — `_build_filter_predicates()`    |

---

## Remaining Work

### P4a — RAG Evaluation Harness (DeepEval)

This is the **prerequisite** for DSPy optimization — both share the same golden dataset and
metric functions.

**What to build:**

1. **Golden dataset** — `tests/eval/golden_dataset.json`

   ```json
   [
     {
       "input": "waterproof hiking boots under $150",
       "expected_output": "...",
       "expected_product_ids": ["prod_abc", "prod_def"]
     }
   ]
   ```

   Aim for 15–25 representative queries covering edge cases (no match, multilingual, filter combos).

2. **Eval test file** — `tests/eval/test_rag_quality.py`
   - `ContextualPrecision` — are retrieved chunks ranked correctly?
   - `ContextualRecall` — do chunks cover the expected answer?
   - `Faithfulness` — does LLM output stay grounded in retrieved context?
   - `AnswerRelevancy` — is the answer relevant to the question?

3. **Nightly CI workflow** — `.github/workflows/eval.yml`
   - Runs on schedule (`cron: '0 2 * * *'`), not on every push
   - Posts a structured metric summary comment on PRs touching `retriever.py`, `rag.py`, or `prompts.py`

4. **Dependency** — add `deepeval` to `requirements-dev.txt`

---

### P4b — DSPy Optimization Engine

`app/optimization/pipeline.py` and `app/optimization/optimize.py` are scaffolded but contain
placeholders only. `DSPyRAGPipeline` is not a real DSPy program.

**What to build:**

1. Replace `DSPyRAGPipeline` with a real DSPy program:

   ```python
   class RAGSignature(dspy.Signature):
       """Grounded answer generation from retrieval context."""
       context: str = dspy.InputField()
       history: str = dspy.InputField()
       query: str = dspy.InputField()
       answer: str = dspy.OutputField()

   class DSPyRAGPipeline(dspy.Module):
       def __init__(self):
           self.generate = dspy.Predict(RAGSignature)
       def forward(self, context, history, query):
           return self.generate(context=context, history=history, query=query)
   ```

2. Replace the `return 1.0` stub metric in `optimize.py` with real DeepEval scores
   (`Faithfulness` + `AnswerRelevancy`) — this is why DeepEval must come first.

3. Wire artifact loading into `app/main.py` startup `lifespan` so optimized prompts
   are applied at runtime without blocking the request path.

4. Add `dspy-ai` to `requirements.txt` (runtime dependency).

5. Add `scripts/run_optimization.py` for one-off / scheduled optimization runs.

---

### P5 — Background Async Re-indexing

`POST /ingest` is currently synchronous and blocks until indexing completes. At scale this will
time out for large catalogs.

**What to build:**

- Use FastAPI `BackgroundTasks` (or ARQ for durability) to run ingest jobs out-of-band
- Return a `job_id` immediately from `POST /ingest`
- Expose `GET /ingest/{job_id}/status` → `{ status: "pending" | "running" | "done" | "error" }`

**Files to touch:** `app/routers/ingest.py`, `app/models/schemas.py`

---

## Suggested Priority Order

```
[Now]     P3  Background re-indexing          — independent, medium effort
[Now]     P4a DeepEval harness + golden data   — prerequisite for DSPy
[After]   P4b DSPy real pipeline + optimizer   — depends on P4a metric functions
```

DSPy optimization (P4b) **must follow** DeepEval (P4a) — the golden dataset and metric
functions from P4a become the training signal for the MIPROv2 optimizer.

---

## Strengths to Preserve

- **Multilingual embeddings** — do not regress to `embed-english-v3.0`
- **Consistent hybrid RRF on both `/chat` and `/search`** — do not split these
- **Strict RAG grounding** in the system prompt — no general knowledge fallback
- **Typed SSE event protocol** (`sources` → `token*` → `done`) — keep source attribution
- **CI pipeline** — ruff + pyright + pytest; extend coverage to retrieval logic when adding eval
