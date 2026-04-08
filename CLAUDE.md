# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MCP

Context7 is configured — use it to fetch up-to-date library documentation when needed (e.g. DSPy, DeepEval, FastAPI, asyncpg, Cohere).

## Architecture

FastAPI RAG backend for e-commerce product discovery via a supplier catalog.

## Key Conventions

- All I/O is async throughout (asyncpg, httpx async client, async generators for SSE)
- Linting: Ruff with rules E, F, I, UP; line length 100 (`ruff.toml`)
- Type checking: Pyright strict mode (`pyrightconfig.json`)
- Pre-commit hooks run ruff, pyright, and pytest — CI runs the same checks
- Auth is optional: if `CHATBOT_API_KEY` is unset, bearer token validation is skipped
