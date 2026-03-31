from typing import Literal

import cohere

from app.config import get_settings
from app.exceptions import ExternalServiceError

EmbeddingInputType = Literal["search_document", "search_query"]


class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = cohere.Client(api_key=settings.cohere_api_key)
        self._model = "embed-multilingual-v3.0"

    async def embed_texts(
        self,
        texts: list[str],
        input_type: EmbeddingInputType,
    ) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for i in range(0, len(texts), 96):
            batch = texts[i : i + 96]
            try:
                response = self._client.embed(
                    model=self._model,
                    input_type=input_type,
                    texts=batch,
                    embedding_types=["float"],
                )
            except Exception as exc:
                raise ExternalServiceError("Embedding provider unavailable") from exc
            vectors.extend(response.embeddings.float)

        return vectors
