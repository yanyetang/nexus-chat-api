# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Context

This repo is one of three services in the `dropship-nexus` project (parent dir: `../`):

- **nexus-app** — Shopify embedded frontend (Next.js)
- **nexus-catalog-api** — product catalog backend (NestJS + Prisma + PostgreSQL)
- **nexus-chat-api** — this service; RAG-powered chat backend (FastAPI + pgvector)

## Architecture

FastAPI RAG backend for e-commerce product discovery via a supplier catalog.

## MCP

Context7 is configured — use it to fetch up-to-date library documentation when implementing any critical path.

## Fixing pnpm Audit Vulnerabilities

Use `/pnpm-audit-fix` (manual skill) when audit vulnerabilities are reported.

## LLM Provider Configuration

**Rule: configure model names in `app/config.py` defaults — not in `.env`.** `.env` is for secrets (API keys) only. This ensures the model selection is consistent across all environments.

There are three separate LLM roles — each uses a different model/key:

| Role      | Config key                                            | Default                                                             | Used by                                        |
| --------- | ----------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------- |
| Chat      | `OPENROUTER_CHAT_MODEL`                               | `openai/gpt-4o-mini`                                                | Every user message (production)                |
| Optimizer | `GROQ_OPTIMIZER_MODEL` / `OPENROUTER_OPTIMIZER_MODEL` | `groq/llama-3.3-70b-versatile` (preferred) / `openrouter/openai/gpt-4o-mini` (fallback) | DSPy MIPROv2 (offline)            |
| Judge     | `OPENROUTER_JUDGE_MODEL`                              | `google/gemini-2.0-flash-001`                                       | DeepEval metrics — CI eval + DSPy optimization |


### Known provider pitfalls

- `openrouter/auto` routes to expensive models (Opus, Sonar) — drains credits fast, do not use
- Gemini free models have failed DeepEval judge scoring in testing — avoid as judge; paid Gemini (e.g. `google/gemini-2.0-flash-001`) works fine
- Groq free tier limit is 12K TPM for `llama-3.3-70b-versatile` — DSPy optimizer trials may get 429s but optimization completes with the best successful trial; Groq is never used as the DeepEval judge (TPM exhausted by 10 tests × 4 metrics — both CI eval and DSPy optimization use OpenRouter for judging)

## DeepEval

- 9 parametrized cases from `tests/eval/golden_dataset.json` + 1 standalone no-match test = 10 total
- Judge LLM uses OpenRouter (`OPENROUTER_API_KEY`) — Groq is intentionally excluded due to free-tier TPM limits failing multi-metric eval
- `CONFIDENT_API_KEY` sends results to Confident AI platform (used in CI for PR comments)
- `pytest.ini` sets `addopts = -m "not deepeval"` to keep eval tests out of normal runs — intentional
- Upload to Confident AI only happens via `deepeval test run` (CLI calls `wrap_up_test_run` after pytest); plain `pytest` never uploads regardless of `CONFIDENT_API_KEY` or `DEEPEVAL` env var
- CI command: `deepeval test run tests/eval/test_rag_quality.py -m deepeval -o "addopts="` — `-m deepeval` selects the tests, `-o "addopts="` clears the conflicting ini default
- Do NOT set `DEEPEVAL=true` with plain `pytest` — it tells `assert_test` to skip upload (expecting CLI to finalize), which then never happens


## DSPy Optimization

- Offline only — run `PYTHONPATH=. .venv/bin/python scripts/run_optimization.py` manually
- Artifact (`artifacts/optimized_pipeline.json`) is gitignored — commit it for production or it falls back to baseline prompt
- MIPROv2 `auto="light"`: 9 trials, 6 instruction candidates — takes ~5 min on Groq free tier
- The artifact injects the optimized instruction into the system prompt at startup; app falls back gracefully if absent
