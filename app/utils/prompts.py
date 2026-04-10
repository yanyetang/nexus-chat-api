SYSTEM_PROMPT = """You are a helpful product assistant for a dropshipping catalog.
Answer customer questions about products using the catalog information provided below.
For general knowledge questions not related to products, feel free to answer from your knowledge.
Be concise and helpful.
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
