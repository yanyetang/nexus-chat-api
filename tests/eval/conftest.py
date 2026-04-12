import os
from typing import Any

import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_openrouter_key() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set; skipping DeepEval judge evaluation")


@pytest.fixture(scope="session")
def openrouter_judge() -> Any:
    pytest.importorskip("deepeval")
    from app.config import get_settings
    from app.optimization.judge import OpenRouterJudge

    settings = get_settings()
    return OpenRouterJudge(
        model=settings.openrouter_judge_model, api_key=settings.openrouter_api_key
    )


@pytest.fixture(scope="session")
def deepeval_metrics(openrouter_judge: Any) -> list[Any]:
    pytest.importorskip("deepeval")
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    threshold = 0.5
    return [
        ContextualPrecisionMetric(model=openrouter_judge, threshold=threshold, include_reason=True),
        ContextualRecallMetric(model=openrouter_judge, threshold=threshold, include_reason=True),
        FaithfulnessMetric(model=openrouter_judge, threshold=threshold, include_reason=True),
        AnswerRelevancyMetric(model=openrouter_judge, threshold=threshold, include_reason=True),
    ]
