import os
from typing import Any

import pytest
from openai import AuthenticationError

_PLACEHOLDER_SECRET_VALUES = {
    "placeholder",
    "dummy",
    "test",
    "example",
    "your_api_key",
    "none",
    "null",
}


def _read_judge_api_key(env_var_name: str) -> str | None:
    raw_value = os.getenv(env_var_name)
    if raw_value is None:
        return None

    api_key = raw_value.strip()
    if not api_key:
        return None
    if api_key.lower() in _PLACEHOLDER_SECRET_VALUES:
        return None
    return api_key


@pytest.fixture(scope="session", autouse=True)
def _require_llm_judge_key() -> None:
    if not _read_judge_api_key("OPENROUTER_API_KEY"):
        pytest.skip(
            "OPENROUTER_API_KEY is not set or contains a placeholder; skipping DeepEval judge evaluation"
        )


@pytest.fixture(scope="session")
def llm_judge() -> Any:
    pytest.importorskip("deepeval")
    from app.config import OPENROUTER_JUDGE_MODEL_DEFAULT
    from app.optimization.judge import _OPENROUTER_BASE_URL, OpenRouterJudge

    openrouter_api_key = _read_judge_api_key("OPENROUTER_API_KEY")
    assert openrouter_api_key is not None

    judge = OpenRouterJudge(
        model=os.getenv("OPENROUTER_JUDGE_MODEL", OPENROUTER_JUDGE_MODEL_DEFAULT),
        api_key=openrouter_api_key,
        base_url=_OPENROUTER_BASE_URL,
    )

    try:
        judge.generate("Reply with OK.")
    except AuthenticationError:
        pytest.skip(
            "OpenRouter judge credentials were rejected by the provider; skipping DeepEval judge evaluation"
        )

    return judge


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
