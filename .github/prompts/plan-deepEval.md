# DeepEval RAG Evaluation Harness — Execution Plan

_Planned: 2026-04-08_

---

## Goal

Build a non-trivial DeepEval harness that can detect regressions in retrieval and grounded generation quality for this RAG backend, and run it safely in CI.

---

## Reality Check Against Current Code

The current repository state does **not** match the previously written implementation summary.

### What exists now

- `tests/eval/test_rag_quality.py` has a single loop-based test over 3 samples
- `tests/eval/golden_dataset.json` has only:
  - `input`
  - `expected_output`
  - `expected_product_ids`
- `actual_output` is currently set equal to `expected_output` in test construction
- `retrieval_context` is currently synthetic (`Expected product ids: ...`) and not realistic context blocks
- No `tests/eval/conftest.py` fixture module exists yet

### Why this is insufficient

- Faithfulness and relevancy become near-trivial when expected and actual are identical
- Minimal context makes contextual metrics low-signal
- Loop-based test structure reduces debuggability and per-case reporting
- Judge/model setup is recreated in the test body each run

---

## Target Design

### Dataset contract

Each golden sample should contain:

- `input`: user query
- `actual_output`: realistic model-style answer (possibly imperfect)
- `expected_output`: ideal grounded answer
- `expected_product_ids`: expected products to appear in grounding
- `retrieval_context`: context list formatted like runtime prompt context

### Metrics to enforce

- `ContextualPrecisionMetric`
- `ContextualRecallMetric`
- `FaithfulnessMetric`
- `AnswerRelevancyMetric`

### Test architecture

- Use `pytest.mark.parametrize` with one test node per sample
- Introduce eval-scoped fixtures in `tests/eval/conftest.py`:
  - judge model fixture
  - metric bundle fixture
  - optional key guard/skip behavior
- Use `assert_test(test_case, metrics)` per sample

### CI architecture

- Keep `.github/workflows/eval.yml` as scheduled + PR-path + manual workflow
- Ensure required secrets/env are explicit and behavior is deterministic when secrets are missing
- Publish PR summary comment with pass/fail and where to inspect metric output

---

## Implementation Plan

### Phase 1: Baseline Harness Upgrade

1. Expand `tests/eval/golden_dataset.json` from 3 samples to 10-15 representative queries
2. Add `actual_output` and `retrieval_context` fields to each sample
3. Replace loop-based test in `tests/eval/test_rag_quality.py` with parametrized tests
4. Add `tests/eval/conftest.py` to centralize judge + metric setup

Acceptance criteria:

- `pytest tests/eval -m deepeval --collect-only` shows one node per sample
- A single failing sample does not abort all sample evaluations
- Metric reasons are visible in test output/logs

### Phase 2: CI Hardening

1. Validate workflow gating logic for presence/absence of required secrets
2. Ensure workflow still runs and reports meaningful status when eval cannot execute live
3. Improve PR summary content with clear run status and metric references

Acceptance criteria:

- CI does not fail due only to missing optional secrets
- CI clearly indicates skipped vs executed evaluation states

### Phase 3: Signal Quality and Maintenance

1. Add multilingual and no-result scenarios to dataset
2. Add a small number of intentionally imperfect outputs to verify faithfulness sensitivity
3. Document update rules for dataset evolution (when retrieval/prompting changes)

Acceptance criteria:

- Faithfulness can fail on deliberate grounding errors
- Contextual metrics reflect ranking/coverage changes when retrieval is altered

---

## Decision Gates (Confirmed)

### Decision 1: Judge model strategy

Selected: Option A

Use DeepEval `GeminiModel` with `GEMINI_API_KEY`.

- Pros: native DeepEval integration, minimal wrapper code
- Cons: additional provider-specific key management

Option B (not selected): custom DeepEval model wrapper over existing runtime provider path

- Pros: potentially reuses existing provider stack
- Cons: more maintenance and more moving parts

### Decision 2: CI behavior when no judge credentials exist

Selected: Option A

Mark eval tests as skipped and pass workflow with explicit skip message when judge credentials are missing.

- Pros: contributor-friendly, no false red pipelines
- Cons: no quality signal for repos without secrets configured

Option B (not selected): hard fail when credentials are missing

- Pros: enforces strict eval readiness
- Cons: blocks external contributors and forks by default

### Decision 3: Initial evaluation mode

Selected: Option A

Use dataset-driven `actual_output` values for the first iteration.

- Pros: deterministic, low cost, fast CI
- Cons: does not execute live generation path

Option B (not selected): execute live generation in eval tests

- Pros: true end-to-end quality signal
- Cons: slower, higher cost, more flaky/network-sensitive

---

## Out of Scope For This Plan

- DSPy optimizer implementation details (separate plan; depends on this harness)
- Full online benchmarking framework beyond DeepEval nightly/PR checks

---

## Approved Baseline

Approved choices:

- Decision 1: Option A
- Decision 2: Option A
- Decision 3: Option A

This establishes a robust, deterministic DeepEval baseline that can be extended to live end-to-end evaluation in a later phase.

---

## Implementation Kickoff Checklist

1. Expand the golden dataset to 10-15 samples with `actual_output` and realistic `retrieval_context`
2. Add `tests/eval/conftest.py` with eval-scoped Gemini judge + metric fixtures
3. Refactor `tests/eval/test_rag_quality.py` to parametrized test cases
4. Update `.github/workflows/eval.yml` to show explicit skipped-vs-executed status messaging
5. Run local verification for collect-only and skip behavior without key
