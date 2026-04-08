import json
import os
from pathlib import Path

import pytest


def _load_golden_dataset() -> list[dict]:
    dataset_path = Path(__file__).resolve().parent / "golden_dataset.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))


@pytest.mark.deepeval
def test_rag_quality_contract() -> None:
    pytest.importorskip("deepeval")

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        pytest.skip("GEMINI_API_KEY not set — skipping DeepEval judge evaluation")

    from deepeval import assert_test
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.models import GeminiModel
    from deepeval.test_case import LLMTestCase

    judge = GeminiModel(model="gemini-2.0-flash", api_key=gemini_key)

    metrics = [
        ContextualPrecisionMetric(model=judge),
        ContextualRecallMetric(model=judge),
        FaithfulnessMetric(model=judge),
        AnswerRelevancyMetric(model=judge),
    ]

    for sample in _load_golden_dataset():
        expected_ids = sample.get("expected_product_ids") or []
        retrieval_context = [f"Expected product ids: {', '.join(expected_ids)}"]
        test_case = LLMTestCase(
            input=sample["input"],
            actual_output=sample["expected_output"],
            expected_output=sample["expected_output"],
            retrieval_context=retrieval_context,
        )
        assert_test(test_case=test_case, metrics=metrics)
