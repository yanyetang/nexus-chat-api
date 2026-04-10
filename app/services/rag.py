from app.config import get_settings
from app.exceptions import ExternalServiceError
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.retriever import RetrieverService
from app.services.supplier import SupplierService
from app.utils.prompts import SYSTEM_PROMPT, build_context_block


class RAGService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._embeddings = EmbeddingService()
        self._retriever = RetrieverService()
        self._supplier = SupplierService()
        self._llm = LLMService()

    @staticmethod
    def _build_live_snapshot(product: dict) -> str:
        variants = product.get("variants") or []
        if not isinstance(variants, list) or not variants:
            return "Live stock/price: unavailable"

        rows: list[str] = []
        for variant in variants[:8]:
            title = str(variant.get("title") or "Unknown").strip()
            price = str(variant.get("price") or "N/A").strip()
            inventory = variant.get("inventory")
            stock_value = inventory if inventory is not None else variant.get("inventoryQty")
            stock = str(stock_value if stock_value is not None else "N/A").strip()
            rows.append(f"- {title}: ${price} (stock: {stock})")

        return "Live stock/price:\n" + "\n".join(rows)

    async def _apply_live_enrichment(self, results: list[dict]) -> list[dict]:
        if not results:
            return results

        product_ids = [str(item.get("product_id")) for item in results if item.get("product_id")]
        if not product_ids:
            return results

        try:
            live_products = await self._supplier.fetch_products_by_ids(product_ids)
        except ExternalServiceError:
            return results

        enriched: list[dict] = []
        for item in results:
            product_id = str(item.get("product_id"))
            product = live_products.get(product_id)
            if not product:
                enriched.append(item)
                continue

            live_snapshot = self._build_live_snapshot(product)
            updated = {
                **item,
                "metadata": {
                    **(item.get("metadata") or {}),
                    "live": {
                        "title": product.get("title"),
                        "brand": product.get("brand"),
                        "category": ((product.get("category") or {}).get("title")),
                    },
                },
                "chunk_text": f"{item.get('chunk_text', '')}\n\n{live_snapshot}",
            }
            enriched.append(updated)
        return enriched

    async def _maybe_rerank(self, message: str, results: list[dict]) -> list[dict]:
        if not results or not self._settings.cohere_rerank_enabled:
            return results

        top_n = min(self._settings.cohere_rerank_top_n, len(results))
        documents = [str(item.get("chunk_text") or "") for item in results]

        try:
            reranked = await self._embeddings.rerank(
                query=message, documents=documents, top_n=top_n
            )
        except ExternalServiceError:
            return results

        if not reranked:
            return results

        output: list[dict] = []
        for item in reranked:
            idx = item.get("index")
            score = item.get("score")
            if not isinstance(idx, int) or idx < 0 or idx >= len(results):
                continue
            if not isinstance(score, float):
                continue
            selected = {**results[idx]}
            selected["retrieval_score"] = selected.get("score", 0.0)
            selected["score"] = score
            output.append(selected)

        if not output:
            return results

        filtered = [
            item for item in output if item["score"] >= self._settings.cohere_rerank_min_score
        ]
        return filtered if filtered else output

    async def get_context(
        self,
        pool,
        message: str,
        limit: int = 5,
        filters: dict | None = None,
    ) -> tuple[list[dict], str | None]:
        query_embedding: list[float] | None = None
        info_message: str | None = None

        try:
            query_vector = await self._embeddings.embed_texts([message], input_type="search_query")
            if query_vector:
                query_embedding = query_vector[0]
        except ExternalServiceError:
            info_message = "Embedding provider unavailable; using keyword-only retrieval fallback."

        results = await self._retriever.hybrid_search(
            pool=pool,
            query_text=message,
            query_embedding=query_embedding,
            limit=limit,
            min_score=self._settings.retrieval_min_score,
            filters=filters,
            candidate_limit=self._settings.retrieval_candidate_limit,
        )

        if not results:
            broad_results = await self._retriever.hybrid_search(
                pool=pool,
                query_text=message,
                query_embedding=query_embedding,
                limit=limit,
                min_score=0.0,
                filters=filters,
                candidate_limit=self._settings.retrieval_candidate_limit,
            )
            if broad_results:
                info_message = "All retrieval candidates were below the similarity threshold."
                results = broad_results

        seen: set[str] = set()
        deduped: list[dict] = []
        for item in results:
            pid = item.get("product_id")
            if pid not in seen:
                seen.add(pid)
                deduped.append(item)
        results = deduped

        reranked_results = await self._maybe_rerank(message=message, results=results)
        enriched_results = await self._apply_live_enrichment(reranked_results)
        return enriched_results, info_message

    async def stream_answer(
        self,
        history: list[dict],
        message: str,
        results: list[dict],
    ):
        context_block = build_context_block(results)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": (
                    "Use this product retrieval context:\n\n"
                    f"{context_block}\n\n"
                    f"User request: {message}"
                ),
            },
        ]

        async for token in self._llm.stream_chat(messages=messages):
            yield token
