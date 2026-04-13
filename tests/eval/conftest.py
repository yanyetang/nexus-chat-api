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
    from app.optimization.judge import _GROQ_BASE_URL, _OPENROUTER_BASE_URL, OpenRouterJudge

    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if groq_api_key:
        return OpenRouterJudge(
            model=os.getenv("GROQ_JUDGE_MODEL", "llama-3.3-70b-versatile"),
            api_key=groq_api_key,
            base_url=_GROQ_BASE_URL,
        )
    assert openrouter_api_key is not None
    return OpenRouterJudge(
        model=os.getenv("OPENROUTER_JUDGE_MODEL", "openai/gpt-4o-mini"),
        api_key=openrouter_api_key,
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
