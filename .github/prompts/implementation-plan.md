# Implementation Plan — Remaining Work

_Plan date: 2026-04-11_

## Current Status

| Item | Status | Notes |
|---|---|---|
| P0–P3 RAG gaps | ✅ Done | All closed per rag-comparison-and-next-steps.md |
| P4a DeepEval harness | ✅ Done | golden_dataset.json, test_rag_quality.py, eval.yml all present |
| P4b DSPy real pipeline | ❌ Incomplete | pipeline.py and optimize.py are placeholders |
| P5 Background re-indexing | ✅ Done | ingest.py already uses BackgroundTasks + job status endpoint |

---

## What Remains: P4b — DSPy Real Pipeline

### Step 1 — Replace `DSPyRAGPipeline` placeholder with a real dspy.Module

**File:** `app/optimization/pipeline.py`

Replace the current dict-based placeholder with a proper DSPy program:

```python
import dspy

class RAGSignature(dspy.Signature):
    """Grounded answer generation from retrieval context."""
    context: str = dspy.InputField()
    history: str = dspy.InputField()
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()

class DSPyRAGPipeline(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict(RAGSignature)

    def forward(self, context, history, query):
        return self.generate(context=context, history=history, query=query)
```

Keep `load_optimized_artifact()` and `get_optimized_artifact()` — they are already wired
into `app/main.py` lifespan correctly. Update `save()` to use `dspy.Module.save()`.

---

### Step 2 — Replace stub metric in `optimize.py` with real DeepEval scores

**File:** `app/optimization/optimize.py`

Replace `return 1.0` in `_metric` with real DeepEval `Faithfulness` + `AnswerRelevancy` scores:

```python
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def _metric(example, prediction, trace=None) -> float:
    test_case = LLMTestCase(
        input=example.query,
        actual_output=prediction.answer,
        retrieval_context=[example.context],
    )
    faithfulness = FaithfulnessMetric(threshold=0.5)
    relevancy = AnswerRelevancyMetric(threshold=0.5)
    faithfulness.measure(test_case)
    relevancy.measure(test_case)
    return (faithfulness.score + relevancy.score) / 2
```

Requires `OPENROUTER_API_KEY` (or another LLM judge) to be set at runtime.

---

### Step 3 — Add `scripts/run_optimization.py`

**File:** `scripts/run_optimization.py` (new file)

One-off script to run MIPROv2 optimization manually or on a schedule:

```python
import json
from pathlib import Path
from app.optimization.optimize import compile_optimized_pipeline, save_artifact

dataset = json.loads(Path("tests/eval/golden_dataset.json").read_text())
artifact = compile_optimized_pipeline(dataset)
save_artifact(artifact, Path("artifacts/optimized_pipeline.json"))
print(artifact)
```

---

### Step 4 — Fix `eval.yml` Python version

**File:** `.github/workflows/eval.yml` line 28

Change `python-version: "3.14"` → `python-version: "3.12"` (3.14 does not exist yet).

---

## What is Already Done (no action needed)

- `dspy-ai>=2.5.0` is in `requirements.txt` ✅
- `deepeval>=1.3.0` is in `requirements-dev.txt` ✅
- `load_optimized_artifact()` wired into `app/main.py` lifespan ✅
- `app/main.py` imports and calls it at startup without blocking the request path ✅
- P5 background re-indexing fully implemented in `app/routers/ingest.py` ✅

---

## Suggested Execution Order

```
1. Fix eval.yml Python version          (2 min, unblocks CI)
2. Step 1 — real DSPyRAGPipeline        (core change)
3. Step 2 — real DeepEval metric        (depends on Step 1)
4. Step 3 — scripts/run_optimization.py (final wiring)
```
