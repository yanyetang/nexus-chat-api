from typing import Any, Literal

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

            embedding_payload: Any = response.embeddings
            float_vectors = getattr(embedding_payload, "float", None)
            if float_vectors is None and isinstance(embedding_payload, list):
                float_vectors = embedding_payload

            if not isinstance(float_vectors, list):
                raise ExternalServiceError("Embedding provider returned unexpected format")

            vectors.extend([list(vec) for vec in float_vectors])

        return vectors
