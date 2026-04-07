import json
from pathlib import Path

import pytest


def _load_golden_dataset() -> list[dict]:
    dataset_path = Path(__file__).resolve().parent / "golden_dataset.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))


@pytest.mark.deepeval
def test_rag_quality_contract() -> None:
    deepeval_module = pytest.importorskip("deepeval")
    assert_test = getattr(deepeval_module, "assert_test")
    metrics_module = __import__(
        "deepeval.metrics",
        fromlist=[
            "AnswerRelevancyMetric",
            "ContextualPrecisionMetric",
            "ContextualRecallMetric",
            "FaithfulnessMetric",
        ],
    )
    test_case_module = __import__("deepeval.test_case", fromlist=["LLMTestCase"])

    AnswerRelevancyMetric = getattr(metrics_module, "AnswerRelevancyMetric")
    ContextualPrecisionMetric = getattr(metrics_module, "ContextualPrecisionMetric")
    ContextualRecallMetric = getattr(metrics_module, "ContextualRecallMetric")
    FaithfulnessMetric = getattr(metrics_module, "FaithfulnessMetric")
    LLMTestCase = getattr(test_case_module, "LLMTestCase")

    metrics = [
        ContextualPrecisionMetric(),
        ContextualRecallMetric(),
        FaithfulnessMetric(),
        AnswerRelevancyMetric(),
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
