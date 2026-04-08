# DeepEval RAG Evaluation Harness — Implementation Summary

_Implemented: 2026-04-08_

---

## What Was Built

A meaningful RAG quality evaluation harness using DeepEval, replacing a smoke-test-only skeleton
with a fully structured eval suite that gives real signal on retrieval and generation quality.

---

## Problem with the Original Harness

The original `test_rag_quality.py` used `expected_output` as **both** `actual_output` and
`expected_output` in every `LLMTestCase`. This meant:

- All 4 metrics trivially passed (a perfect answer is always faithful to itself)
- DeepEval defaulted to OpenAI GPT as judge → crashed with no `OPENAI_API_KEY`
- Only 3 placeholder samples with no `retrieval_context` field
- Metrics were rebuilt inside a `for` loop → `GeminiModel` instantiated on every iteration

---

## What Changed

### 1. Judge Model — OpenAI → Gemini Free Tier

**Problem:** DeepEval's default judge is OpenAI GPT (requires `OPENAI_API_KEY`).

**Fix:** DeepEval `>=1.3.0` ships a built-in `GeminiModel` class. Google AI Studio has a free tier
(no credit card required). The project already uses `google/gemini-2.0-flash-001` as its primary
LLM — consistent model family.

**Setup required (one-time):**

1. Get a free key from `aistudio.google.com`
2. Add `GEMINI_API_KEY=<key>` to `.env`
3. Add `GEMINI_API_KEY` as a GitHub Actions secret

**Files changed:**

- `requirements-dev.txt` — added `google-generativeai>=0.8.0`
- `.github/workflows/eval.yml` — added `GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}` env var

---

### 2. Golden Dataset — 3 → 11 Samples

**File:** `tests/eval/golden_dataset.json`

Each sample now has 5 fields:

| Field                  | Purpose                                                                 |
| ---------------------- | ----------------------------------------------------------------------- |
| `input`                | The user query                                                          |
| `actual_output`        | Realistic LLM-style answer — grounded but with 1–2 subtle imperfections |
| `expected_output`      | The ideal fully-grounded answer                                         |
| `expected_product_ids` | List of product IDs that should appear                                  |
| `retrieval_context`    | List of strings, each formatted as `build_context_block()` produces     |

**Query coverage across 11 samples:**

| #   | Query type                                    | Products                   |
| --- | --------------------------------------------- | -------------------------- |
| 1   | Price-range (t-shirts under $25)              | tee-101, tee-204           |
| 2   | Color/variant (black running shoes)           | shoe-302, shoe-499         |
| 3   | Comparison (backpacks ~$60)                   | bag-901, bag-114           |
| 4   | Category (wireless headphones, battery life)  | hdp-201, hdp-355           |
| 5   | Brand filter (Nike sneakers)                  | snk-601, snk-677           |
| 6   | Size-specific (women's sandals size 8)        | san-410, san-502           |
| 7   | Low-stock urgency                             | tee-204, shoe-499, hdp-355 |
| 8   | Multi-attribute (blue denim jacket under $80) | jkt-303, jkt-418           |
| 9   | No-match (empty context)                      | —                          |
| 10  | French query (chaussures de course noires)    | shoe-499, shoe-302         |
| 11  | Spanish query (mochilas para viaje ~$60)      | bag-901, bag-114           |

**Why intentional imperfections in `actual_output`?**
Each `actual_output` contains 1–2 subtle divergences from the `retrieval_context`
(e.g. price mis-quoted as `$18.99` instead of `$19.99`, unsupported claim like "Both ship quickly")
so `FaithfulnessMetric` has real signal to detect rather than trivially passing.
`expected_output` is always fully grounded.

**`retrieval_context` format** — matches exactly what `build_context_block()` in
`app/utils/prompts.py` produces at runtime, so the eval reflects real pipeline output:

```
[1] Product ID: tee-101
Title: Basic Cotton Crew Neck T-Shirt
Score: 0.9120
Data:
Product: Basic Cotton Crew Neck T-Shirt
Brand: ComfortWear | Model: CW-TEE-001 | Category: Tops
Description: ...
Tags: ...
Variants:
- White / S: $19.99, Size: S, Color: White, SKU: ..., Stock: 50
...
```

---

### 3. Session-Scoped Fixtures — `tests/eval/conftest.py`

**New file** scoped only to the `tests/eval/` package — does not affect the 23 unit tests.

```python
# _require_gemini_key — autouse, session-scoped
# Skips entire eval session if GEMINI_API_KEY is unset
# (test skips gracefully instead of hard-failing)

# gemini_judge — session-scoped
# Single GeminiModel instance shared across all 11 test cases
# Avoids 11 separate auth/init calls

# deepeval_metrics — session-scoped
# 4 metrics built once with judge injected:
#   ContextualPrecisionMetric, ContextualRecallMetric,
#   FaithfulnessMetric, AnswerRelevancyMetric
```

---

### 4. Parametrized Tests — `tests/eval/test_rag_quality.py`

**Before:** single test with a `for` loop over 3 samples → one failure aborts all; metrics
rebuilt every iteration; `GeminiModel` instantiated inline.

**After:** `@pytest.mark.parametrize` over 11 samples → 11 independent pytest nodes.

Benefits:

- One sample failure does not abort the other 10
- Per-sample metric scores appear in CI logs
- `pytest --lf` (last-failed) works at sample granularity
- Fixtures injected from conftest (metrics built once per session)

---

## The 4 DeepEval Metrics — What They Measure

| Metric                | Question                                           | What it catches                                   |
| --------------------- | -------------------------------------------------- | ------------------------------------------------- |
| `ContextualPrecision` | Are the most relevant chunks ranked first?         | Reranker not working; wrong chunks ranked highest |
| `ContextualRecall`    | Do the chunks contain everything needed to answer? | Missing products; retrieval too narrow            |
| `Faithfulness`        | Does the answer stick to what the chunks say?      | LLM hallucinating facts not in context            |
| `AnswerRelevancy`     | Does the answer address what was asked?            | LLM going off-topic or being too vague            |

Each metric scores 0–1. Scores below DeepEval's default threshold (0.5) fail the test.

---

## Files Changed / Created

| File                             | Action   | Description                                                 |
| -------------------------------- | -------- | ----------------------------------------------------------- |
| `tests/eval/golden_dataset.json` | Replaced | 3 → 11 samples, added `actual_output` + `retrieval_context` |
| `tests/eval/conftest.py`         | Created  | Session-scoped Gemini key guard + judge + metrics fixtures  |
| `tests/eval/test_rag_quality.py` | Replaced | Parametrized, fixture-injected, key guard removed           |
| `requirements-dev.txt`           | Updated  | Added `google-generativeai>=0.8.0`                          |
| `.github/workflows/eval.yml`     | Updated  | Added `GEMINI_API_KEY` env var to pytest step; Python 3.12  |

---

## CI Workflow

**File:** `.github/workflows/eval.yml`

**Triggers:**

- Weekly schedule (Monday 04:00 UTC)
- Pull requests touching `retriever.py`, `rag.py`, or `prompts.py`
- Manual `workflow_dispatch`

**Behaviour without `GEMINI_API_KEY` secret:** test session skips gracefully — CI passes.
**Behaviour with key:** 11 test cases run, each reporting individual metric scores.

---

## Verification

```bash
# Confirm 11 nodes collected — no API call, no key needed
pytest tests/eval -m deepeval --collect-only

# Run without key — all 11 should skip (not fail)
pytest tests/eval -m deepeval -v

# Run with key — live eval
GEMINI_API_KEY=<key> pytest tests/eval -m deepeval -v

# Confirm 23 unit tests still pass
pytest tests/ -v --ignore=tests/eval
```

---

## What This Does NOT Cover (Future Work)

- **Live end-to-end eval** — current harness uses pre-built `actual_output` from the golden
  dataset, not real LLM generation. A future enhancement would mock retrieval but call the
  real `LLMService` to generate answers at eval time (adds OpenRouter cost per run).
- **DSPy optimization** (P4b) — depends on this harness. The golden dataset and metric
  functions here become the training signal for `dspy.MIPROv2`. Implement DSPy after
  this harness is validated in CI.
