# Chatbot API (Python RAG Backend)

A FastAPI backend that powers product-focused chat and search for the dropship platform.

It keeps the existing supplier API unchanged, ingests product catalog data, stores vector embeddings in Supabase (pgvector), and serves:

- `POST /ingest` for indexing products
- `GET /search` for hybrid retrieval
- `POST /chat` for streaming RAG responses (SSE)

## Project Architecture

### High-Level Components

- **FastAPI application**
  - API routers for ingest, search, and chat
  - Startup lifecycle initializes async database pool
- **Supplier API integration**
  - Reads products from supplier API `GET /catalog/export`
- **Embedding service (Cohere)**
  - Uses `embed-multilingual-v3.0`
  - `search_document` for indexing, `search_query` for retrieval
- **Supabase PostgreSQL + pgvector**
  - Stores product chunks and embeddings
  - Stores chat session history
- **OpenRouter LLM service**
  - Streams chat completions for final user responses
- **Hybrid retriever**
  - Combines vector similarity and PostgreSQL full-text search with RRF scoring

### Data Stores

- `public.product_embeddings`
  - `product_id`, `chunk_text`, `embedding vector(1024)`, `metadata`, timestamps
- `public.chat_sessions`
  - `session_id`, `messages jsonb`, timestamps

## Main Workflows (Mermaid Sequence Charts)

### 1) Product Ingestion Workflow

```mermaid
sequenceDiagram
    autonumber
    participant Client as Admin/Job Trigger
    participant API as chatbot-api (FastAPI)
    participant Supplier as dropship-supplier-api
    participant Cohere as Cohere Embedding API
    participant DB as Supabase Postgres (pgvector)

    Client->>API: POST /ingest (Bearer CHATBOT_API_KEY)
    API->>Supplier: GET /catalog/export
    Supplier-->>API: Product catalog (products + variants)

    loop Batch chunks (up to 96 texts)
        API->>Cohere: embed(texts, input_type=search_document)
        Cohere-->>API: Embedding vectors (1024-d)
    end

    loop Upsert each product
        API->>DB: INSERT ... ON CONFLICT(product_id) DO UPDATE
    end

    API-->>Client: { indexed, total }
```

### 2) Chat RAG Workflow

```mermaid
sequenceDiagram
    autonumber
    participant User as Frontend User
    participant FE as dropship-application
    participant API as chatbot-api (FastAPI)
    participant Cohere as Cohere Embedding API
    participant DB as Supabase Postgres (pgvector + chat_sessions)
    participant OR as OpenRouter LLM

    User->>FE: Ask a product question
    FE->>API: POST /chat (session_id, message)

    API->>DB: Load chat_sessions by session_id
    DB-->>API: Previous messages

    API->>Cohere: embed(query, input_type=search_query)
    Cohere-->>API: Query embedding

    API->>DB: Hybrid search (vector + FTS + RRF)
    DB-->>API: Top relevant product chunks

    API->>OR: Stream completion with context + history
    OR-->>API: Token stream
    API-->>FE: SSE events (sources, token, done)

    API->>DB: Upsert chat session with new assistant reply
    DB-->>API: OK
```

### 3) Search-Only Workflow (`GET /search`)

```mermaid
sequenceDiagram
  autonumber
  participant FE as Frontend or API Client
  participant API as chatbot-api (FastAPI)
  participant Cohere as Cohere Embedding API
  participant DB as Supabase Postgres (pgvector + FTS)

  FE->>API: GET /search?q=<query>&limit=<n>
  API->>Cohere: embed(query, input_type=search_query)
  Cohere-->>API: Query embedding (1024-d)

  API->>DB: Semantic search (embedding cosine distance)
  API->>DB: Keyword search (PostgreSQL FTS)
  API->>DB: Reciprocal Rank Fusion (RRF)
  DB-->>API: Ranked product chunks + scores

  API-->>FE: JSON results [{ product_id, chunk_text, metadata, score }]
```

### 4) Failure Paths (Operational)

```mermaid
sequenceDiagram
  autonumber
  participant FE as Frontend or API Client
  participant API as chatbot-api (FastAPI)
  participant Cohere as Cohere Embedding API
  participant DB as Supabase Postgres
  participant OR as OpenRouter LLM

  FE->>API: Request (/search or /chat)

  alt Cohere timeout/error
    API-->>FE: 502 Bad Gateway (embedding provider unavailable)
  else Embedding succeeds
    API->>DB: Retrieval and/or session query
    alt DB query failure
      API-->>FE: 500 Internal Server Error
    else No matching products
      API-->>FE: 200 with empty results (/search)
      API-->>FE: SSE response with "no strong match" guidance (/chat)
    else Retrieval succeeds
      API->>OR: Chat completion request (for /chat)
      alt LLM timeout/error
        API-->>FE: SSE error event then done
      else LLM succeeds
        API-->>FE: Normal token stream or JSON response
      end
    end
  end
```

## Repository Structure

```text
chatbot-api/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models/
│   │   └── schemas.py
│   ├── routers/
│   │   ├── ingest.py
│   │   ├── search.py
│   │   └── chat.py
│   ├── services/
│   │   ├── supplier.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   ├── llm.py
│   │   └── rag.py
│   └── utils/
│       ├── chunking.py
│       └── prompts.py
├── .vscode/mcp.json
├── .env.example
├── requirements.txt
├── Procfile
└── runtime.txt
```

## Configuration

Set environment values in `.env`:

- `DATABASE_URL`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPPLIER_API_BASE_URL`
- `COHERE_API_KEY`
- `OPENROUTER_API_KEY`
- `CHATBOT_API_KEY`
- `OPENROUTER_MODEL` (optional, default `openai/gpt-4o-mini`)
- `ALLOWED_ORIGINS` (optional, default `*`)

## Run Locally

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Start API:

```bash
uvicorn app.main:app --reload
```

3. Health check:

```bash
curl http://localhost:8000/health
```

## API Endpoints

- `GET /health`
  - Service health check
- `POST /ingest`
  - Pulls supplier catalog and writes embeddings into `product_embeddings`
  - Requires `Authorization: Bearer <CHATBOT_API_KEY>` when key is configured
- `GET /search?q=...&limit=5`
  - Hybrid product retrieval
- `POST /chat`
  - SSE streaming RAG response
  - Body: `{ "session_id": "...", "message": "...", "locale": "en" }`

## Notes

- Supplier API remains independent and is not modified by this project.
- Current implementation is designed for the existing Supabase project and can be migrated later by changing environment values and re-running ingestion.
