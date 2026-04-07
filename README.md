# Chatbot API (Python RAG Backend)

FastAPI backend for product-grounded search and chat over a supplier catalog.

## What is Implemented

- Hybrid retrieval with vector + PostgreSQL FTS candidate generation
- Similarity threshold gating on retrieval scores
- Cohere reranking (`rerank-multilingual-v3.0`) for final ordering
- Live inventory enrichment after retrieval (fresh stock/price snapshot)
- Embedding failure fallback to keyword-only retrieval
- Async ingest jobs (`POST /ingest`) with job polling (`GET /ingest/{job_id}/status`)
- DB bootstrap for `TSVECTOR` maintenance trigger + GIN index + IVFFlat vector index
- DeepEval harness and nightly/PR workflow
- DSPy optimization scaffold with offline artifact generation and startup artifact load

## Core Workflows

### Ingest (Background)

1. Client calls `POST /ingest`
2. API returns `202` with `job_id`
3. Background task fetches supplier catalog, chunks text, embeds in batches, and upserts `product_embeddings`
4. Client polls `GET /ingest/{job_id}/status`

### Search

1. Embed query (or fallback to keyword-only retrieval when embeddings fail)
2. Run hybrid candidate retrieval (semantic + FTS) with optional metadata filters
3. Apply threshold gating
4. Return ranked JSON results

### Chat

1. Load prior session messages
2. Retrieve candidates using hybrid search with fallback and filters
3. Rerank candidates with Cohere cross-encoder
4. Enrich retrieved chunks with live supplier stock/price
5. Stream answer via SSE (`sources` -> `token*` -> `done`)
6. Persist updated chat history

## Database Bootstrap

Startup runs automatic bootstrap (enabled by default):

- Creates `product_embeddings` and `chat_sessions` if missing
- Adds `content_tsv` column on `product_embeddings`
- Creates trigger `trg_product_embeddings_tsv` to maintain `content_tsv`
- Creates GIN index on `content_tsv`
- Creates IVFFlat index on vector column:

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_product_embeddings_vector
  ON product_embeddings USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
```

Manual SQL bootstrap is also available at `scripts/init_db.sql`.

## API Endpoints

- `GET /health`
- `POST /ingest` -> `202 { job_id, status }`
- `GET /ingest/{job_id}/status` -> `{ job_id, status, indexed, total, error }`
- `GET /search?q=...&limit=...&category=...&brand=...&min_price=...&max_price=...`
- `POST /search/query`
  - Body: `{ "query": "...", "limit": 5, "filters": { ... } }`
- `POST /chat`
  - Body: `{ "session_id": "...", "message": "...", "locale": "en", "filters": { ... } }`

## Configuration

Required:

- `DATABASE_URL`
- `SUPPLIER_API_BASE_URL`
- `COHERE_API_KEY`
- `OPENROUTER_API_KEY`

Optional:

- `CHATBOT_API_KEY`
- `OPENROUTER_MODEL` (default `google/gemini-2.0-flash-001`)
- `ALLOWED_ORIGINS` (default `*`)
- `RETRIEVAL_MIN_SCORE` (default `0.3`)
- `RETRIEVAL_CANDIDATE_LIMIT` (default `20`)
- `COHERE_RERANK_ENABLED` (default `true`)
- `COHERE_RERANK_TOP_N` (default `5`)
- `DB_AUTO_BOOTSTRAP` (default `true`)

## Evaluation and Optimization

- DeepEval suite: `tests/eval/test_rag_quality.py`
- Golden dataset: `tests/eval/golden_dataset.json`
- CI workflow: `.github/workflows/eval.yml`
- Offline optimization script: `scripts/run_optimization.py`
- Runtime artifact load path: `artifacts/optimized_pipeline.json`

Run evaluation:

```bash
pytest tests/eval -m deepeval -q
```

Generate optimization artifact:

```bash
python scripts/run_optimization.py
```

## Local Development

```bash
make install
make dev
```

Quality checks:

```bash
make lint
make type-check
make test
```
