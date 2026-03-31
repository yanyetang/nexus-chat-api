from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.retriever import RetrieverService
from app.utils.prompts import SYSTEM_PROMPT, build_context_block


class RAGService:
    def __init__(self) -> None:
        self._embeddings = EmbeddingService()
        self._retriever = RetrieverService()
        self._llm = LLMService()

    async def get_context(self, pool, message: str, limit: int = 5) -> list[dict]:
        query_vector = await self._embeddings.embed_texts([message], input_type="search_query")
        if not query_vector:
            return []
        return await self._retriever.hybrid_search(
            pool=pool,
            query_text=message,
            query_embedding=query_vector[0],
            limit=limit,
        )

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
