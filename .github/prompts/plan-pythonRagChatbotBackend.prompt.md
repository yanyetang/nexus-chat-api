# Plan: Python RAG Chatbot Backend (chatbot-api)

## TL;DR

Build a Python FastAPI backend in `/Projects/chatbot-api` that ingests product data from the existing supplier-api, stores Cohere multilingual embeddings in the current Supabase pgvector, and serves a hybrid-search RAG chatbot. The dropship-application gets a chat widget. The supplier-api is untouched.

## Key Decisions

- **Keep supplier-api separate** — other projects consume it
- **Use current Supabase** (project `ljlcfhltacynylgrgpnu`) until decommissioned
- **Cohere `embed-multilingual-v3.0`** for embeddings (free trial, 1024 dims, native en/fr/zh)
- **OpenRouter** for LLM chat completions
- **Supabase MCP + Context7 MCP** for IDE-assisted development

## Architecture

```
┌─────────────────────┐                         ┌──────────────────────┐
│ dropship-application │──── catalog/orders ────▶│ dropship-supplier-api│
│   (Next.js)         │                         │   (NestJS)           │
│                     │     ┌──────────────┐    │                      │
│ + Chat widget UI    │────▶│  chatbot-api │───▶│ GET /catalog/export  │
│ + /api/chat proxy   │     │  (FastAPI)   │    └──────────────────────┘
└─────────────────────┘     │              │
                            │ POST /chat   │    ┌──────────────────────┐
                            │ GET  /search │───▶│  Supabase PostgreSQL │
                            │ POST /ingest │    │  (pgvector + FTS)    │
                            └──────────────┘    └──────────────────────┘
```

---

## Phase 0: Project Setup & Tooling

### Step 0.1: Supabase MCP configuration

Create `.vscode/mcp.json` in chatbot-api workspace:

```json
{
  "servers": {
    "supabase": {
      "type": "http",
      "url": "https://mcp.supabase.com/mcp?project_ref=ljlcfhltacynylgrgpnu&features=database,docs,debugging"
    }
  }
}
```

- Scoped to project `ljlcfhltacynylgrgpnu` (the shared Supabase instance)
- Features: database (list_tables, execute_sql, apply_migration), docs, debugging
- OAuth login will be prompted on first use
- Context7 MCP is already configured globally

### Step 0.2: .env updates

Current `.env` has:

- `SUPABASE_URL` ✓
- `SUPABASE_PUBLISHABLE_KEY` — rename to `SUPABASE_ANON_KEY` (clarity, not used server-side but keep for reference)
- `SUPABASE_SECRET_KEY` — rename to `SUPABASE_SERVICE_ROLE_KEY` (standard Supabase naming)
- `DATABASE_URL` ✓ (pooled connection — used by asyncpg)
- `SUPPLIER_API_BASE_URL` ✓
- `COHERE_API_KEY` ✓
- `OPENROUTER_API_KEY` ✓

Add missing:

- `CHATBOT_API_KEY` — generate with `openssl rand -hex 32`, used by dropship-application to authenticate requests

Final `.env`:

```
# Supabase
SUPABASE_URL=https://ljlcfhltacynylgrgpnu.supabase.co
SUPABASE_ANON_KEY=<your-anon-key>
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
DATABASE_URL=<your-database-url>

# Supplier API
SUPPLIER_API_BASE_URL=https://dropship-supplier-api-dev-1de48966c7bc.herokuapp.com/

# Cohere (embeddings)
COHERE_API_KEY=<your-cohere-key>

# OpenRouter (LLM chat)
OPENROUTER_API_KEY=<your-openrouter-key>

# Service auth (dropship-application → chatbot-api)
CHATBOT_API_KEY=<generate with: openssl rand -hex 32>
```

### Step 0.3: Project scaffolding

```
chatbot-api/
├── .vscode/
│   └── mcp.json                   # Supabase MCP config
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app, CORS, lifespan (asyncpg pool)
│   ├── config.py                  # Pydantic Settings — all env vars
│   ├── database.py                # asyncpg pool management, pgvector registration
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py                # POST /chat — RAG streaming endpoint
│   │   ├── search.py              # GET /search — hybrid search
│   │   └── ingest.py              # POST /ingest — product indexing trigger
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embeddings.py          # Cohere embed-multilingual-v3.0 client
│   │   ├── llm.py                 # OpenRouter chat completions (streaming)
│   │   ├── retriever.py           # Hybrid search: pgvector + FTS + RRF
│   │   ├── rag.py                 # RAG pipeline orchestration
│   │   └── supplier.py            # httpx client for supplier-api /catalog/export
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic: ChatRequest, ChatResponse, SearchResult, etc.
│   └── utils/
│       ├── __init__.py
│       ├── chunking.py            # Product → text chunk formatting
│       └── prompts.py             # System prompt templates
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── Procfile                       # web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
└── runtime.txt                    # python-3.12.x
```

Dependencies (`requirements.txt`):

```
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
httpx>=0.28.0
asyncpg>=0.30.0
pgvector>=0.3.0
cohere>=5.0.0
pydantic-settings>=2.7.0
python-dotenv>=1.0.0
sse-starlette>=2.0.0
```

---

## Phase 1: Database & Embeddings

### Step 1.1: Enable pgvector in Supabase (_use Supabase MCP `apply_migration`_)

SQL migration:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS product_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id uuid NOT NULL UNIQUE,
    chunk_text text NOT NULL,
    embedding vector(1024) NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX product_embeddings_embedding_idx
    ON product_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX product_embeddings_fts_idx
    ON product_embeddings USING gin (to_tsvector('english', chunk_text));

CREATE TABLE IF NOT EXISTS chat_sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id text NOT NULL UNIQUE,
    messages jsonb NOT NULL DEFAULT '[]',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX chat_sessions_session_id_idx ON chat_sessions (session_id);
```

- `vector(1024)` — matches Cohere embed-multilingual-v3.0 output
- HNSW index for approximate nearest neighbor (fast cosine similarity)
- GIN index on `chunk_text` for PostgreSQL full-text search
- `product_id UNIQUE` — enables upsert on re-ingestion

### Step 1.2: Database connection module (`app/database.py`)

- Create asyncpg pool on app startup (lifespan)
- Register pgvector type with `pgvector.asyncpg.register_vector` in pool init callback
- Use `DATABASE_URL` from config
- Expose `get_pool()` dependency for routers

### Step 1.3: Cohere embeddings service (`app/services/embeddings.py`)

- Use `cohere.Client` (sync) or `cohere.AsyncClient` (async)
- Model: `embed-multilingual-v3.0` (1024 dimensions)
- **Critical**: Use `input_type="search_document"` when embedding product chunks for storage
- **Critical**: Use `input_type="search_query"` when embedding user queries for search
- Batch embed support: up to 96 texts per call (handles all ~145 products in 2 batches)
- Return `list[list[float]]`

### Step 1.4: Product ingestion pipeline (`app/routers/ingest.py` + `app/services/supplier.py`)

- `POST /ingest` — protected by `CHATBOT_API_KEY` header
- Flow:
  1. `supplier.py`: Fetch all products via `GET {SUPPLIER_API_BASE_URL}/catalog/export`
  2. `chunking.py`: Format each product as a text chunk:
     ```
     Product: {title}
     Brand: {brand} | Category: {category.title}
     Description: {description}
     Tags: {tags joined}
     Variants:
     - {variant.title}: ${price}, Size: {size}, Color: {color}, SKU: {sku}, Stock: {inventory}
     - ...
     ```
  3. `embeddings.py`: Batch embed all chunks with `input_type="search_document"`
  4. Upsert into `product_embeddings` (ON CONFLICT product_id DO UPDATE)
  5. Return `{ indexed: N, total: N }`

---

## Phase 2: Search & RAG

### Step 2.1: Hybrid retriever (`app/services/retriever.py`)

- Implements Reciprocal Rank Fusion (RRF) combining:
  - **Semantic search**: `embedding <=> query_embedding` (pgvector cosine distance)
  - **Keyword search**: `to_tsvector('english', chunk_text) @@ plainto_tsquery('english', query)`
- SQL query (from pgvector-python docs — proven pattern):
  ```sql
  WITH semantic_search AS (
      SELECT id, product_id, chunk_text, metadata,
             RANK() OVER (ORDER BY embedding <=> $1) AS rank
      FROM product_embeddings
      ORDER BY embedding <=> $1
      LIMIT 20
  ),
  keyword_search AS (
      SELECT id, product_id, chunk_text, metadata,
             RANK() OVER (ORDER BY ts_rank_cd(to_tsvector('english', chunk_text), query) DESC) AS rank
      FROM product_embeddings, plainto_tsquery('english', $2) query
      WHERE to_tsvector('english', chunk_text) @@ query
      ORDER BY ts_rank_cd(to_tsvector('english', chunk_text), query) DESC
      LIMIT 20
  )
  SELECT COALESCE(s.product_id, k.product_id) as product_id,
         COALESCE(s.chunk_text, k.chunk_text) as chunk_text,
         COALESCE(s.metadata, k.metadata) as metadata,
         COALESCE(1.0 / ($3 + s.rank), 0.0) +
         COALESCE(1.0 / ($3 + k.rank), 0.0) AS score
  FROM semantic_search s
  FULL OUTER JOIN keyword_search k ON s.id = k.id
  ORDER BY score DESC
  LIMIT $4
  ```
- Parameters: query_embedding (vector), query_text (str), k=60 (RRF constant), limit=5
- Returns: list of `SearchResult(product_id, chunk_text, metadata, score)`

### Step 2.2: Search endpoint (`app/routers/search.py`)

- `GET /search?q={query}&limit=5&locale=en`
- Embed query with `input_type="search_query"`
- Call retriever
- Return ranked results with product_id, chunk_text preview, score

### Step 2.3: LLM service (`app/services/llm.py`)

- OpenRouter API via httpx (OpenAI-compatible `/chat/completions` endpoint)
- Base URL: `https://openrouter.ai/api/v1/chat/completions`
- Model: configurable (default `openai/gpt-4o-mini` — cheap, fast)
- Streaming support: yield SSE chunks
- Headers: `Authorization: Bearer {OPENROUTER_API_KEY}`

### Step 2.4: RAG pipeline (`app/services/rag.py`)

- Input: user message, session_id, locale
- Flow:
  1. Load conversation history from `chat_sessions` (last N messages)
  2. Embed user message with Cohere (`input_type="search_query"`)
  3. Hybrid search → top 5 product chunks
  4. Build prompt:
     - System prompt (from `utils/prompts.py`) with role, capabilities, constraints
     - Retrieved product context
     - Conversation history
     - User message
  5. Stream LLM response via OpenRouter
  6. Save assistant response to `chat_sessions`
- System prompt defines:
  - Role: product discovery assistant for a dropshipping platform
  - Capabilities: find products by need, compare products, answer general Q&A
  - Constraints: only recommend products from retrieved context, cite product names, be honest when no match found
  - Locale awareness: respond in user's language when detected

### Step 2.5: Chat endpoint (`app/routers/chat.py`)

- `POST /chat` with body `{ session_id: str, message: str, locale?: str }`
- Protected by `CHATBOT_API_KEY` or open (depending on deployment)
- Returns Server-Sent Events (SSE) stream using `sse-starlette`
- Each event: `{ "type": "token", "content": "..." }` or `{ "type": "sources", "products": [...] }`
- Final event: `{ "type": "done" }`

---

## Phase 3: Frontend Integration (dropship-application)

### Step 3.1: Chat proxy route

- Create `web/src/app/api/chat/route.ts`
- Proxies POST to `{CHATBOT_API_URL}/chat` with streaming
- Adds `Authorization: Bearer {CHATBOT_API_KEY}` header
- Passes through SSE stream to client

### Step 3.2: Search proxy route (optional)

- Create `web/src/app/api/search/route.ts`
- Proxies GET to `{CHATBOT_API_URL}/search`

### Step 3.3: Chat widget component

- Create `web/src/components/chat-widget.tsx`
- Floating button (bottom-right) → expandable chat panel
- Features: message input, streaming display, product cards, session persistence (localStorage)
- Tailwind CSS matching existing emerald/slate theme
- Add to `web/src/app/layout.tsx`

### Step 3.4: Environment variables

- Add to `web/.env.local`:
  ```
  CHATBOT_API_URL=http://localhost:8000   # or Heroku URL
  CHATBOT_API_KEY=<same key as chatbot-api>
  ```

---

## Phase 4: Supplier API — No Code Changes

- `GET /catalog/export` already returns all products with variants — no modifications needed
- Server-to-server calls from chatbot-api — no CORS changes needed
- Future: add webhooks for real-time re-indexing (not MVP)

---

## Relevant Files

### chatbot-api (CREATE — all new)

| File                         | Purpose                                           |
| ---------------------------- | ------------------------------------------------- |
| `.vscode/mcp.json`           | Supabase MCP config (project-scoped)              |
| `.env`                       | Already exists — rename keys, add CHATBOT_API_KEY |
| `.env.example`               | Template without secrets                          |
| `.gitignore`                 | Python + .env patterns                            |
| `requirements.txt`           | Python dependencies                               |
| `Procfile`                   | Heroku deployment                                 |
| `runtime.txt`                | Python 3.12                                       |
| `app/main.py`                | FastAPI app, CORS, lifespan (pool init)           |
| `app/config.py`              | Pydantic Settings from .env                       |
| `app/database.py`            | asyncpg pool + pgvector registration              |
| `app/routers/chat.py`        | POST /chat — SSE streaming RAG                    |
| `app/routers/search.py`      | GET /search — hybrid search                       |
| `app/routers/ingest.py`      | POST /ingest — product indexing                   |
| `app/services/embeddings.py` | Cohere embed-multilingual-v3.0                    |
| `app/services/llm.py`        | OpenRouter streaming client                       |
| `app/services/retriever.py`  | Hybrid search (pgvector + FTS + RRF)              |
| `app/services/rag.py`        | RAG orchestration                                 |
| `app/services/supplier.py`   | Fetch products from supplier-api                  |
| `app/models/schemas.py`      | Pydantic request/response models                  |
| `app/utils/chunking.py`      | Product → text chunk formatter                    |
| `app/utils/prompts.py`       | System prompt templates                           |

### dropship-supplier-api (NO CHANGES)

### dropship-application (MODIFY — Phase 3)

| File                                 | Action                                        |
| ------------------------------------ | --------------------------------------------- |
| `web/src/app/api/chat/route.ts`      | CREATE — proxy to chatbot-api                 |
| `web/src/components/chat-widget.tsx` | CREATE — floating chat UI                     |
| `web/src/app/layout.tsx`             | MODIFY — add ChatWidget                       |
| `web/.env.local`                     | MODIFY — add CHATBOT_API_URL, CHATBOT_API_KEY |

---

## Verification

1. **pgvector setup**: Use Supabase MCP `execute_sql` → `SELECT * FROM pg_extension WHERE extname = 'vector'` → confirm enabled
2. **Ingestion**: `curl -X POST http://localhost:8000/ingest -H "Authorization: Bearer $CHATBOT_API_KEY"` → response `{ "indexed": 145, "total": 145 }`
3. **Verify embeddings**: Supabase MCP `execute_sql` → `SELECT product_id, length(chunk_text), embedding <-> embedding FROM product_embeddings LIMIT 5`
4. **Search**: `curl "http://localhost:8000/search?q=blue+cotton+t-shirt&limit=5"` → ranked product results
5. **Chat**: `curl -N -X POST http://localhost:8000/chat -d '{"session_id":"test","message":"I need a summer shirt"}' -H "Content-Type: application/json"` → SSE stream with product recommendations
6. **Multilingual**: Search with French/Chinese queries → returns relevant products (Cohere multilingual)
7. **Comparison**: Chat "compare the cheapest and most expensive hoodies" → structured comparison
8. **Frontend**: Chat widget in dropship-application → send message → streamed response with products

---

## Implementation Order & Dependencies

```
Phase 0: Setup (Steps 0.1-0.3)
    ↓
Phase 1: Database & Embeddings (Steps 1.1-1.4)
    ↓ (1.1 blocks 1.2, 1.2 blocks 1.3-1.4)
Phase 2: Search & RAG (Steps 2.1-2.5)
    ↓ (2.1 depends on 1.3, 2.4 depends on 2.1+2.3)
    ↓ Steps 2.2 and 2.3 can run in parallel
Phase 3: Frontend (Steps 3.1-3.4)
    ↓ (depends on Phase 2 being testable)
    ↓ Steps 3.1-3.3 can run in parallel
```

---

## Decisions

- **LLM**: OpenRouter → `openai/gpt-4o-mini` default (cheap, fast; can switch to Claude)
- **Embeddings**: Cohere `embed-multilingual-v3.0` (1024d, free trial, native i18n)
- **Vector store**: pgvector in current Supabase project
- **Search**: Hybrid (vector + FTS) with RRF scoring (k=60)
- **Data sync**: Manual `POST /ingest`; no webhooks for MVP
- **Supplier API**: Unchanged
- **Service auth**: Shared `CHATBOT_API_KEY` header
- **Streaming**: SSE via `sse-starlette`
- **Scope IN**: Product discovery, comparison, general Q&A
- **Scope OUT**: Order status, Shopify admin actions, multi-tenant isolation

## Further Considerations

1. **Cohere free tier**: 1,000 calls/month. Ingesting ~145 products = 2 batch calls. Each user search = 1 call. Budget ~998 search queries/month on free tier.

2. **Supabase decommission**: When it happens, create new Supabase project, update env vars (`SUPABASE_URL`, `DATABASE_URL`, keys), re-run migration SQL, re-ingest products. No code changes needed.

3. **OpenRouter model**: `gpt-4o-mini` is recommended for speed/cost. Can upgrade to `anthropic/claude-sonnet-4` or `openai/gpt-4o` for better quality by changing one config value.
