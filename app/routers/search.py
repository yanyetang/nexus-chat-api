from fastapi import APIRouter, HTTPException, Query

from app.database import get_pool
from app.exceptions import DatabaseOperationError, ExternalServiceError
from app.models.schemas import SearchResponse, SearchResult
from app.services.embeddings import EmbeddingService
from app.services.retriever import RetrieverService

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
async def search_products(
    q: str = Query(min_length=1, max_length=300),
    limit: int = Query(default=5, ge=1, le=20),
) -> SearchResponse:
    embeddings = EmbeddingService()
    retriever = RetrieverService()
    pool = get_pool()

    try:
        vectors = await embeddings.embed_texts([q], input_type="search_query")
    except ExternalServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not vectors:
        raise HTTPException(status_code=500, detail="Failed to embed query")

    try:
        items = await retriever.hybrid_search(
            pool=pool,
            query_text=q,
            query_embedding=vectors[0],
            limit=limit,
        )
    except DatabaseOperationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(results=[SearchResult(**item) for item in items])
