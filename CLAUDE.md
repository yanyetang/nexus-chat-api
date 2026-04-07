# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install      # Install all dependencies (prod + dev)
make dev          # Start dev server with auto-reload
make test         # Run all tests
make lint         # Run ruff linter
make type-check   # Run pyright
make format       # Auto-format with ruff
```

Run a single test file: `.venv/bin/pytest tests/test_chunking.py -q`
Run a single test: `.venv/bin/pytest tests/test_chunking.py::test_name -q`

## Architecture

FastAPI RAG backend for e-commerce product discovery via a supplier catalog. Three main endpoints: `/ingest`, `/search`, `/chat`.

**Request flow:**
- **Ingest**: Fetch catalog from Supplier API → chunk products → embed with Cohere → upsert to PostgreSQL (`product_embeddings` table with pgvector)
- **Search/Chat**: Embed query → hybrid search (vector cosine + PostgreSQL FTS) → Reciprocal Rank Fusion → top-k results → (chat only) stream LLM response via SSE

**Service layer** (`app/services/`):
- `supplier.py` — fetches product catalog from external Supplier API
- `embeddings.py` — Cohere `embed-multilingual-v3.0` (1024-d), batches up to 96 texts
- `retriever.py` — hybrid search: vector TOP 20 + FTS TOP 20, merged via RRF
- `llm.py` — streams completions from OpenRouter
- `rag.py` — orchestrates retrieval + streaming; persists sessions to `chat_sessions` table

**Database**: PostgreSQL with pgvector. Pool initialized in lifespan (`app/database.py`). Tables: `product_embeddings` (id, product_id, chunk_text, embedding, metadata jsonb) and `chat_sessions` (session_id, messages jsonb).

**Config**: Pydantic Settings (`app/config.py`). All settings come from environment variables.

## Environment Variables

```
DATABASE_URL              # PostgreSQL connection (Neon supported)
SUPPLIER_API_BASE_URL     # Base URL of the supplier catalog API
COHERE_API_KEY
OPENROUTER_API_KEY
CHATBOT_API_KEY           # Bearer token for /ingest and /chat (optional)
OPENROUTER_MODEL          # Default: google/gemini-2.0-flash-001
ALLOWED_ORIGINS           # CORS origins, default: *
```

## Key Conventions

- All I/O is async throughout (asyncpg, httpx async client, async generators for SSE)
- Linting: Ruff with rules E, F, I, UP; line length 100 (`ruff.toml`)
- Type checking: Pyright strict mode (`pyrightconfig.json`)
- Pre-commit hooks run ruff, pyright, and pytest — CI runs the same checks
- Auth is optional: if `CHATBOT_API_KEY` is unset, bearer token validation is skipped
