# RAG Comparison: `nexus-chat-api` vs `project-dropship`

_Analysis date: 2026-04-08 — Updated: 2026-04-11_

## Architecture Overview

| Dimension                 | `project-dropship` (TS/Next.js)          | `nexus-chat-api` (Python/FastAPI)                            |
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

`nexus-chat-api` applies the same hybrid RRF pipeline (`RetrieverService.hybrid_search`) to both
`/chat` and `/search`, ensuring consistent retrieval quality across endpoints.

### 2. Multilingual Embeddings

`project-dropship` uses `embed-english-v3.0`. `nexus-chat-api` uses `embed-multilingual-v3.0`,
which supports 100+ languages with no accuracy regression on English. This matters for
international supplier catalogs or multilingual user bases.

### 3. DB-Level FTS Maintenance

`project-dropship` computes `to_tsvector(...)` inline at query time — expensive at scale.
`nexus-chat-api` maintains a `content_tsv TSVECTOR` column updated by a PostgreSQL trigger
on insert/update, with a GIN index. Queries hit the pre-computed column directly.

### 4. Retrieval Pipeline Depth

`project-dropship` pipeline: embed → vector search → threshold filter → enrich → LLM

`nexus-chat-api` pipeline: embed → hybrid RRF → threshold gate (with broad fallback) → rerank → enrich → LLM

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

_Updated 2026-04-11 — all P4a, P4b, and P5 items are now complete._

### P4a — RAG Evaluation Harness (DeepEval) ✅

| Item | Status | Location |
|---|---|---|
| Golden dataset (9 cases, incl. multilingual) | ✅ Done | `tests/eval/golden_dataset.json` |
| Eval test file | ✅ Done | `tests/eval/test_rag_quality.py` |
| Nightly CI workflow | ✅ Done | `.github/workflows/eval.yml` |
| `deepeval` dependency | ✅ Done | `requirements-dev.txt` |

---

### P4b — DSPy Optimization Engine ✅

| Item | Status | Location |
|---|---|---|
| `DSPyRAGPipeline` — real `dspy.Module` with `RAGSignature` | ✅ Done | `app/optimization/pipeline.py` |
| Real DeepEval metric (Faithfulness + AnswerRelevancy) | ✅ Done | `app/optimization/optimize.py` |
| `OpenRouterJudge` — custom DeepEval LLM judge | ✅ Done | `app/optimization/judge.py` |
| `OptimizedPromptArtifact` dataclass + parser | ✅ Done | `app/optimization/pipeline.py` |
| Artifact wired into `app/main.py` lifespan | ✅ Done | `app/main.py` |
| Optimized instructions + demos consumed in prompts | ✅ Done | `app/utils/prompts.py` |
| `dspy-ai` runtime dependency | ✅ Done | `requirements.txt` |
| `scripts/run_optimization.py` — one-off trigger script | ✅ Done | `scripts/run_optimization.py` |

**To run optimization:**
```sh
OPENROUTER_API_KEY=<key> .venv/bin/python scripts/run_optimization.py
```
Produces `artifacts/optimized_pipeline.json` — commit this file so the app loads it at startup.

---

### P5 — Background Async Re-indexing ✅

| Item | Status | Location |
|---|---|---|
| `POST /ingest` uses `BackgroundTasks`, returns `job_id` immediately | ✅ Done | `app/routers/ingest.py` |
| `GET /ingest/{job_id}/status` endpoint | ✅ Done | `app/routers/ingest.py` |
| `IngestJobAccepted` / `IngestJobStatus` schemas | ✅ Done | `app/models/schemas.py` |

---

## No Remaining Work

All P0–P5 items from the original gap analysis are complete. The DSPy offline optimization
pipeline is the only operational step remaining — run `scripts/run_optimization.py` whenever
the golden dataset or retrieval pipeline changes significantly, then commit the updated artifact.

---

## Strengths to Preserve

- **Multilingual embeddings** — do not regress to `embed-english-v3.0`
- **Consistent hybrid RRF on both `/chat` and `/search`** — do not split these
- **Strict RAG grounding** in the system prompt — no general knowledge fallback
- **Typed SSE event protocol** (`sources` → `token*` → `done`) — keep source attribution
- **CI pipeline** — ruff + pyright + pytest; extend coverage to retrieval logic when adding eval
