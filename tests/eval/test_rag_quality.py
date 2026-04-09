import json
from pathlib import Path
from typing import Any

import pytest


def _load_golden_dataset() -> list[dict[str, Any]]:
    dataset_path = Path(__file__).resolve().parent / "golden_dataset.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def _sample_id(sample: dict[str, Any]) -> str:
    case_id = sample.get("case_id")
    if isinstance(case_id, str) and case_id.strip():
        return case_id.strip()
    return str(sample.get("input", "eval-case"))[:48]


@pytest.mark.deepeval
@pytest.mark.parametrize("sample", _load_golden_dataset(), ids=_sample_id)
def test_rag_quality_contract(sample: dict[str, Any], deepeval_metrics: list[Any]) -> None:
    pytest.importorskip("deepeval")

    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=sample["input"],
        actual_output=sample["actual_output"],
        expected_output=sample["expected_output"],
        retrieval_context=sample["retrieval_context"],
    )
    assert_test(test_case=test_case, metrics=deepeval_metrics)
