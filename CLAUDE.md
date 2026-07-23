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

| Role      | Config key                                            | Default                                                                                 | Used by                                        |
| --------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Chat      | `OPENROUTER_CHAT_MODEL`                               | `openai/gpt-4o-mini`                                                                    | Every user message (production)                |
| Optimizer | `GROQ_OPTIMIZER_MODEL` / `OPENROUTER_OPTIMIZER_MODEL` | `groq/llama-3.3-70b-versatile` (preferred) / `openrouter/openai/gpt-4o-mini` (fallback) | DSPy MIPROv2 (offline)                         |
| Judge     | `OPENROUTER_JUDGE_MODEL`                              | `google/gemini-2.0-flash-001`                                                           | DeepEval metrics — CI eval + DSPy optimization |

### Known provider pitfalls

- `openrouter/auto` routes to expensive models (Opus, Sonar) — drains credits fast, do not use
- Gemini free models have failed DeepEval judge scoring in testing — avoid as judge; paid Gemini (e.g. `google/gemini-2.0-flash-001`) works fine
- Groq free tier limit is 12K TPM for `llama-3.3-70b-versatile` — DSPy optimizer trials may get 429s but optimization completes with the best successful trial; Groq is never used as the DeepEval judge (TPM exhausted by 10 tests × 4 metrics — both CI eval and DSPy optimization use OpenRouter for judging)

## DeepEval

- 9 parametrized cases from `tests/eval/golden_dataset.json` + 1 standalone no-match test = 10 total
- Judge LLM uses OpenRouter (`OPENROUTER_API_KEY`) — Groq is intentionally excluded due to free-tier TPM limits failing multi-metric eval
- `CONFIDENT_API_KEY` sends results to Confident AI; view runs at app.confident-ai.com → **Testing → Test Runs**

### How the upload works (important)

Upload to Confident AI only happens when `deepeval test run` CLI is used — it calls `wrap_up_test_run` → `post_test_run` after pytest finishes. Plain `pytest` never triggers this, regardless of `CONFIDENT_API_KEY` or `DEEPEVAL` env var being set.

- `assert_test` (used in tests) only saves results to a temp file; it does not upload
- Setting `DEEPEVAL=true` with plain `pytest` makes things worse — `assert_test` sees it and skips even the temp save, expecting the CLI to finalize (which never comes)
- Metric scores only print in CI logs when a test **fails**; passing tests are silent — check Confident AI dashboard for scores

### CI command

```
deepeval test run tests/eval/test_rag_quality.py -m deepeval -v -o "addopts=" -rs
```

- `-m deepeval` — selects eval tests (passed to pytest by the CLI's `--mark` flag)
- `-o "addopts="` — clears `addopts = -m "not deepeval"` from `pytest.ini` which would otherwise deselect all tests
- `-v -rs` — passed as raw pytest args via CLI's extra-args passthrough; keeps logs verbose with skip reasons

### Local eval run (does not upload)

```
pytest tests/eval -m deepeval -v -rs
```

Use this to run tests locally without uploading. Results stay local.

### CLI flag gotcha

Before using any `deepeval` CLI flag from docs, verify it exists in the installed version with `--help`. Context7 docs may reflect a different version. For example, `deepeval login --confident-api-key` does not exist in v3.9.7.

## Git Flow

- `main` is protected — direct pushes are rejected; all changes go through a PR
- Required status check: `ci` must pass before merge
- follow comman Branch naming convention
- The eval workflow (`.github/workflows/eval.yml`) triggers on `pull_request` only for changes to `app/services/retriever.py`, `app/services/rag.py`, or `app/utils/prompts.py` — it will NOT auto-run for workflow-only changes; use **Actions → rag-eval → Run workflow** (workflow_dispatch) to trigger manually on any branch

## Package Management

This project uses **uv**. Do not use `pip`, `python -m venv`, or direct `.venv/bin/*` paths.

- Install deps: `uv sync --group dev`
- Run anything: `uv run <command>` (e.g. `uv run pytest`, `uv run uvicorn ...`)
- Dependencies live in `pyproject.toml`; `uv.lock` is the lockfile

## DSPy Optimization

- Offline only — run `PYTHONPATH=. uv run python scripts/run_optimization.py` manually
- Artifact (`artifacts/optimized_pipeline.json`) is gitignored — commit it for production or it falls back to baseline prompt
- MIPROv2 `auto="light"`: 9 trials, 6 instruction candidates — takes ~5 min on Groq free tier
- The artifact injects the optimized instruction into the system prompt at startup; app falls back gracefully if absent
