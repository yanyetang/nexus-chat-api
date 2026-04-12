import json
from pathlib import Path

import pytest

from app.optimization.optimize import save_artifact
from app.optimization.pipeline import (
    OptimizedPromptArtifact,
    get_optimized_artifact,
    load_optimized_artifact,
)
from app.utils.prompts import SYSTEM_PROMPT, build_system_prompt, build_user_prompt


def test_build_system_prompt_without_artifact() -> None:
    assert build_system_prompt() == SYSTEM_PROMPT


def test_build_system_prompt_with_artifact() -> None:
    artifact = OptimizedPromptArtifact(
        instructions="Only answer from retrieved context.",
        demos=(),
        metadata={},
    )

    prompt = build_system_prompt(artifact)
    assert "Optimized answer policy:" in prompt
    assert "Only answer from retrieved context." in prompt


def test_build_user_prompt_includes_examples() -> None:
    artifact = OptimizedPromptArtifact(
        instructions="Answer briefly.",
        demos=(
            {
                "context": "Product ID: tee-101",
                "query": "Find a t-shirt",
                "answer": "Try tee-101.",
            },
        ),
        metadata={},
    )

    prompt = build_user_prompt(
        context_block="[1] Product ID: tee-101",
        message="Find a t-shirt",
        optimized_artifact=artifact,
    )

    assert "Reference examples:" in prompt
    assert "Assistant answer: Try tee-101." in prompt
    assert "User request: Find a t-shirt" in prompt


def test_load_optimized_artifact_returns_none_when_missing(monkeypatch, tmp_path: Path) -> None:
    artifact_path = tmp_path / "optimized_pipeline.json"
    monkeypatch.setattr("app.optimization.pipeline._ARTIFACT_PATH", artifact_path)

    assert load_optimized_artifact() is None
    assert get_optimized_artifact() is None


def test_load_optimized_artifact_extracts_compiled_instructions(
    monkeypatch, tmp_path: Path
) -> None:
    artifact_path = tmp_path / "optimized_pipeline.json"
    artifact_path.write_text(
        json.dumps(
            {
                "generate": {
                    "demos": [{"context": "ctx", "query": "q", "answer": "a"}],
                    "signature": {
                        "instructions": "Use only the supplied catalog context.",
                        "fields": [],
                    },
                },
                "metadata": {"dependency_versions": {"dspy": "test"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.optimization.pipeline._ARTIFACT_PATH", artifact_path)

    artifact = load_optimized_artifact()

    assert artifact is not None
    assert artifact.instructions == "Use only the supplied catalog context."
    assert artifact.demos[0]["answer"] == "a"


def test_load_optimized_artifact_rejects_malformed_payload(monkeypatch, tmp_path: Path) -> None:
    artifact_path = tmp_path / "optimized_pipeline.json"
    artifact_path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    monkeypatch.setattr("app.optimization.pipeline._ARTIFACT_PATH", artifact_path)

    with pytest.raises(RuntimeError, match="malformed"):
        load_optimized_artifact()


def test_save_artifact_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    save_artifact(
        {"status": "compiled", "artifact": "artifacts/optimized_pipeline.json"}, output_path
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "compiled"
