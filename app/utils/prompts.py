from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.optimization import OptimizedPromptArtifact

SYSTEM_PROMPT = """You are a helpful product assistant for a dropshipping catalog.
Answer customer questions about products using the catalog information provided below.
For general knowledge questions not related to products, feel free to answer from your knowledge.
Be concise and helpful.
"""


def build_system_prompt(optimized_artifact: "OptimizedPromptArtifact | None" = None) -> str:
    if optimized_artifact is None or not optimized_artifact.instructions.strip():
        return SYSTEM_PROMPT

    return (
        f"{SYSTEM_PROMPT.rstrip()}\n\n"
        "Optimized answer policy:\n"
        f"{optimized_artifact.instructions.strip()}"
    )


def build_user_prompt(
    context_block: str,
    message: str,
    optimized_artifact: "OptimizedPromptArtifact | None" = None,
) -> str:
    sections: list[str] = []

    if optimized_artifact and optimized_artifact.demos:
        examples: list[str] = []
        for index, demo in enumerate(optimized_artifact.demos[:2], start=1):
            examples.append(
                f"Example {index}\n"
                f"Context:\n{demo.get('context', '')}\n\n"
                f"User request: {demo.get('query', '')}\n"
                f"Assistant answer: {demo.get('answer', '')}"
            )
        sections.append("Reference examples:\n\n" + "\n\n".join(examples))

    sections.append(f"Use this product retrieval context:\n\n{context_block}")
    sections.append(f"User request: {message}")
    return "\n\n".join(sections)


def build_context_block(results: list[dict]) -> str:
    if not results:
        return "No relevant products were retrieved."

    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        product_id = item.get("product_id")
        title = metadata.get("title") or "Unknown"
        score = item.get("score", 0)
        chunk_text = item.get("chunk_text", "")

        lines.append(
            f"[{idx}] Product ID: {product_id}\n"
            f"Title: {title}\n"
            f"Score: {score:.4f}\n"
            f"Data:\n{chunk_text}"
        )

    return "\n\n".join(lines)
