import os
from typing import Any

import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_gemini_key() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set; skipping DeepEval judge evaluation")


@pytest.fixture(scope="session")
def gemini_judge() -> Any:
    pytest.importorskip("deepeval")
    from deepeval.models import GeminiModel

    return GeminiModel(
        model="gemini-2.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )


@pytest.fixture(scope="session")
def deepeval_metrics(gemini_judge: Any) -> list[Any]:
    pytest.importorskip("deepeval")
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    threshold = 0.5
    return [
        ContextualPrecisionMetric(model=gemini_judge, threshold=threshold, include_reason=True),
        ContextualRecallMetric(model=gemini_judge, threshold=threshold, include_reason=True),
        FaithfulnessMetric(model=gemini_judge, threshold=threshold, include_reason=True),
        AnswerRelevancyMetric(model=gemini_judge, threshold=threshold, include_reason=True),
    ]
