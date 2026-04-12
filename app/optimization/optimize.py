import json
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.optimization.pipeline import DSPyRAGPipeline

_ARTIFACT_PATH = Path("artifacts/optimized_pipeline.json")
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _joined_context(example: dict[str, Any]) -> str:
    retrieval_context = example.get("retrieval_context") or []
    if isinstance(retrieval_context, list):
        return "\n\n".join(str(item) for item in retrieval_context)
    return str(retrieval_context)


def compile_optimized_pipeline(golden_dataset: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        dspy = import_module("dspy")
        teleprompt = import_module("dspy.teleprompt")
        MIPROv2 = getattr(teleprompt, "MIPROv2")
    except Exception:
        return {
            "optimizer": "dspy.MIPROv2",
            "dataset_size": len(golden_dataset),
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "dspy_not_installed",
        }

    try:
        deepeval_metrics = import_module("deepeval.metrics")
        deepeval_test_case = import_module("deepeval.test_case")
        AnswerRelevancyMetric = getattr(deepeval_metrics, "AnswerRelevancyMetric")
        FaithfulnessMetric = getattr(deepeval_metrics, "FaithfulnessMetric")
        LLMTestCase = getattr(deepeval_test_case, "LLMTestCase")
    except Exception:
        return {
            "optimizer": "dspy.MIPROv2",
            "dataset_size": len(golden_dataset),
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "deepeval_not_installed",
        }

    settings = get_settings()
    api_key = settings.openrouter_api_key
    if not api_key:
        return {
            "optimizer": "dspy.MIPROv2",
            "dataset_size": len(golden_dataset),
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "missing_openrouter_api_key",
        }

    chat_model = settings.openrouter_chat_model
    judge_model = settings.openrouter_judge_model

    dspy.configure(
        lm=dspy.LM(
            chat_model,
            api_key=api_key,
            api_base=_OPENROUTER_BASE_URL,
            cache=True,
            temperature=0.0,
            max_tokens=512,
        )
    )

    try:
        from app.optimization.judge import OpenRouterJudge
    except Exception:
        return {
            "optimizer": "dspy.MIPROv2",
            "dataset_size": len(golden_dataset),
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "judge_not_available",
        }

    judge = OpenRouterJudge(model=judge_model, api_key=api_key)

    def _metric(example: Any, prediction: Any, trace=None) -> float:  # noqa: ANN001
        del trace

        answer = getattr(prediction, "answer", "")
        test_case = LLMTestCase(
            input=str(getattr(example, "query", "")),
            actual_output=str(answer),
            expected_output=str(getattr(example, "answer", "")),
            retrieval_context=[str(getattr(example, "context", ""))],
        )

        faithfulness = FaithfulnessMetric(model=judge, threshold=0.5, include_reason=False)
        relevancy = AnswerRelevancyMetric(model=judge, threshold=0.5, include_reason=False)
        faithfulness.measure(test_case)
        relevancy.measure(test_case)
        return (float(faithfulness.score or 0.0) + float(relevancy.score or 0.0)) / 2

    example_cls = getattr(dspy, "Example")
    trainset = []
    for example in golden_dataset:
        sample = example_cls(
            context=_joined_context(example),
            history="",
            query=str(example.get("input", "")),
            answer=str(example.get("expected_output", "")),
        ).with_inputs("context", "history", "query")
        trainset.append(sample)

    program = DSPyRAGPipeline()
    teleprompter = MIPROv2(metric=_metric, auto="light")
    optimized_program = teleprompter.compile(
        program,
        trainset=trainset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
    )

    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(_ARTIFACT_PATH))

    instructions = optimized_program.generate.signature.instructions.strip()

    return {
        "optimizer": "dspy.MIPROv2",
        "dataset_size": len(golden_dataset),
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "compiled",
        "artifact": str(_ARTIFACT_PATH),
        "model": chat_model,
        "judge_model": judge_model,
        "instruction_preview": instructions[:240],
        "metric": "faithfulness_plus_answer_relevancy",
    }


def save_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
