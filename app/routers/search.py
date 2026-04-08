from fastapi import APIRouter, HTTPException, Query

from app.database import get_pool
from app.exceptions import DatabaseOperationError, ExternalServiceError
from app.models.schemas import RetrievalFilters, SearchRequest, SearchResponse, SearchResult
from app.services.embeddings import EmbeddingService
from app.services.retriever import RetrieverService

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
async def search_products(
    q: str = Query(min_length=1, max_length=300),
    limit: int = Query(default=5, ge=1, le=20),
    category: str | None = Query(default=None, max_length=120),
    brand: str | None = Query(default=None, max_length=120),
    min_price: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, ge=0),
) -> SearchResponse:
    embeddings = EmbeddingService()
    retriever = RetrieverService()
    pool = get_pool()
    query_vector: list[float] | None = None

    filters = RetrievalFilters(
        category=category,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
    ).model_dump(exclude_none=True)

    try:
        vectors = await embeddings.embed_texts([q], input_type="search_query")
        if vectors:
            query_vector = vectors[0]
    except ExternalServiceError:
        query_vector = None

    if query_vector is None:
        # Keyword-only fallback keeps retrieval available if embeddings are down.
        pass

    try:
        items = await retriever.hybrid_search(
            pool=pool,
            query_text=q,
            query_embedding=query_vector,
            limit=limit,
            filters=filters,
        )
    except DatabaseOperationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(results=[SearchResult(**item) for item in items])


@router.post("/query", response_model=SearchResponse)
async def search_products_with_filters(payload: SearchRequest) -> SearchResponse:
    embeddings = EmbeddingService()
    retriever = RetrieverService()
    pool = get_pool()
    query_vector: list[float] | None = None

    try:
        vectors = await embeddings.embed_texts([payload.query], input_type="search_query")
        if vectors:
            query_vector = vectors[0]
    except ExternalServiceError:
        query_vector = None

    try:
        items = await retriever.hybrid_search(
            pool=pool,
            query_text=payload.query,
            query_embedding=query_vector,
            limit=payload.limit,
            filters=(payload.filters.model_dump(exclude_none=True) if payload.filters else None),
        )
    except DatabaseOperationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(results=[SearchResult(**item) for item in items])
