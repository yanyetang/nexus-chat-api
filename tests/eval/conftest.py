import os
from typing import Any

import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_llm_judge_key() -> None:
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip(
            "Neither GROQ_API_KEY nor OPENROUTER_API_KEY is set; skipping DeepEval judge evaluation"
        )


@pytest.fixture(scope="session")
def llm_judge() -> Any:
    pytest.importorskip("deepeval")
    from app.config import get_settings
    from app.optimization.judge import _GROQ_BASE_URL, _OPENROUTER_BASE_URL, OpenRouterJudge

    settings = get_settings()
    if settings.groq_api_key:
        return OpenRouterJudge(
            model=settings.groq_judge_model,
            api_key=settings.groq_api_key,
            base_url=_GROQ_BASE_URL,
        )
    assert settings.openrouter_api_key is not None
    return OpenRouterJudge(
        model=settings.openrouter_judge_model,
        api_key=settings.openrouter_api_key,
        base_url=_OPENROUTER_BASE_URL,
    )


@pytest.fixture(scope="session")
def deepeval_metrics(llm_judge: Any) -> list[Any]:
    pytest.importorskip("deepeval")
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    threshold = 0.5
    return [
        ContextualPrecisionMetric(model=llm_judge, threshold=threshold, include_reason=True),
        ContextualRecallMetric(model=llm_judge, threshold=threshold, include_reason=True),
        FaithfulnessMetric(model=llm_judge, threshold=threshold, include_reason=True),
        AnswerRelevancyMetric(model=llm_judge, threshold=threshold, include_reason=True),
    ]
