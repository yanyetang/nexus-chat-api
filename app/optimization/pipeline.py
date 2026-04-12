import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import dspy
except ImportError:  # pragma: no cover - handled by runtime checks.
    dspy = None

_ARTIFACT_PATH = Path("artifacts/optimized_pipeline.json")
_OPTIMIZED_ARTIFACT: "OptimizedPromptArtifact | None" = None


@dataclass(frozen=True)
class OptimizedPromptArtifact:
    instructions: str
    demos: tuple[dict[str, str], ...]
    metadata: dict[str, Any]


def _normalize_demo(demo: Any) -> dict[str, str] | None:
    if not isinstance(demo, dict):
        return None

    normalized: dict[str, str] = {}
    for key in ("context", "history", "query", "answer"):
        value = demo.get(key)
        if value is None:
            continue
        normalized[key] = str(value)

    return normalized or None


def _parse_artifact_payload(payload: dict[str, Any]) -> OptimizedPromptArtifact:
    generate_state = payload.get("generate")
    if not isinstance(generate_state, dict):
        raise ValueError("optimized artifact is missing the generate state")

    signature_state = generate_state.get("signature")
    if not isinstance(signature_state, dict):
        raise ValueError("optimized artifact is missing the signature state")

    instructions = str(signature_state.get("instructions") or "").strip()
    if not instructions:
        raise ValueError("optimized artifact does not contain compiled instructions")

    demos_raw = generate_state.get("demos")
    demos: list[dict[str, str]] = []
    if isinstance(demos_raw, list):
        for demo in demos_raw:
            normalized = _normalize_demo(demo)
            if normalized:
                demos.append(normalized)

    raw_metadata = payload.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    return OptimizedPromptArtifact(
        instructions=instructions,
        demos=tuple(demos),
        metadata=metadata,
    )


if dspy is not None:

    class RAGSignature(dspy.Signature):  # type: ignore[misc]
        """Answer the customer query using only the retrieved catalog context and prior chat history. If the context is insufficient, say so plainly and do not invent products or details."""

        context: str = dspy.InputField(desc="Retrieved catalog context with product facts.")
        history: str = dspy.InputField(desc="Prior conversation turns, if any.")
        query: str = dspy.InputField(desc="Current customer request.")
        answer: str = dspy.OutputField(desc="Grounded answer for the customer.")

    class DSPyRAGPipeline(dspy.Module):  # type: ignore[misc]
        """DSPy program compiled offline and consumed at runtime via saved state."""

        def __init__(self) -> None:
            super().__init__()
            self.generate = dspy.Predict(RAGSignature)  # type: ignore[union-attr]

        def forward(self, context: str, history: str, query: str):
            return self.generate(context=context, history=history, query=query)

else:

    class DSPyRAGPipeline:  # type: ignore[no-redef]  # pragma: no cover
        def __init__(self) -> None:
            raise RuntimeError("dspy is required to build or load the optimized pipeline")


def load_optimized_artifact() -> OptimizedPromptArtifact | None:
    global _OPTIMIZED_ARTIFACT
    if not _ARTIFACT_PATH.exists():
        _OPTIMIZED_ARTIFACT = None
        return None

    try:
        with _ARTIFACT_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        _OPTIMIZED_ARTIFACT = None
        raise RuntimeError(f"Failed to parse optimized artifact at {_ARTIFACT_PATH}") from exc

    if not isinstance(payload, dict):
        _OPTIMIZED_ARTIFACT = None
        raise RuntimeError(f"Optimized artifact at {_ARTIFACT_PATH} must be a JSON object")

    try:
        _OPTIMIZED_ARTIFACT = _parse_artifact_payload(payload)
    except ValueError as exc:
        _OPTIMIZED_ARTIFACT = None
        raise RuntimeError(f"Optimized artifact at {_ARTIFACT_PATH} is malformed") from exc

    return _OPTIMIZED_ARTIFACT


def get_optimized_artifact() -> OptimizedPromptArtifact | None:
    return _OPTIMIZED_ARTIFACT
