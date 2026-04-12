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
