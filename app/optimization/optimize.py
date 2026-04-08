import json
from importlib import import_module
from pathlib import Path
from typing import Any

from app.optimization.pipeline import DSPyRAGPipeline


def compile_optimized_pipeline(golden_dataset: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        dspy = import_module("dspy")
        teleprompt = import_module("dspy.teleprompt")
        MIPROv2 = getattr(teleprompt, "MIPROv2")
    except Exception:
        return {
            "optimizer": "dspy.MIPROv2",
            "dataset_size": len(golden_dataset),
            "generated_at": "offline",
            "status": "dspy_not_installed",
            "settings": {
                "retrieval_min_score": 0.3,
                "candidate_limit": 20,
                "rerank_top_n": 5,
            },
        }

    # This uses a minimal metric so MIPROv2 can run offline with a synthetic trainset.
    # Real metric wiring should map to DeepEval faithfulness/relevancy scores.
    def _metric(_: Any, __: Any, trace=None) -> float:  # noqa: ANN001
        return 1.0

    example_cls = getattr(dspy, "Example")
    trainset = []
    for example in golden_dataset:
        sample = example_cls(
            context=str(example.get("expected_product_ids", [])),
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

    artifact_path = Path("artifacts/optimized_pipeline.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(artifact_path))

    return {
        "optimizer": "dspy.MIPROv2",
        "dataset_size": len(golden_dataset),
        "generated_at": "offline",
        "status": "compiled",
        "artifact": str(artifact_path),
    }


def save_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
