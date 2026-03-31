SYSTEM_PROMPT = """You are a helpful product discovery assistant for a dropshipping application.

Rules:
1. Use ONLY the provided retrieval context for product recommendations and comparisons.
2. If the context does not contain enough information, say so clearly.
3. Prefer concise, practical recommendations with product names and key variant details.
4. When comparing products, include price, sizes/colors, and stock signals when available.
5. Keep answers factual and avoid claiming actions you cannot perform.
"""


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
