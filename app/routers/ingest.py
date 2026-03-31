import json

import asyncpg
from fastapi import APIRouter, Header, HTTPException, status

from app.config import get_settings
from app.database import get_pool
from app.exceptions import ExternalServiceError
from app.models.schemas import IngestResponse
from app.services.embeddings import EmbeddingService
from app.services.supplier import SupplierService
from app.utils.chunking import product_to_chunk

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_catalog(authorization: str | None = Header(default=None)) -> IngestResponse:
    settings = get_settings()
    if settings.chatbot_api_key:
        expected = f"Bearer {settings.chatbot_api_key}"
        if authorization != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    supplier = SupplierService()
    embeddings = EmbeddingService()
    pool = get_pool()

    try:
        products = await supplier.fetch_catalog_export()
    except ExternalServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    chunks = [product_to_chunk(product) for product in products]
    try:
        vectors = await embeddings.embed_texts(chunks, input_type="search_document")
    except ExternalServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if len(chunks) != len(vectors):
        raise HTTPException(status_code=500, detail="Embedding count mismatch")

    sql = """
    INSERT INTO product_embeddings (product_id, chunk_text, embedding, metadata, updated_at)
    VALUES ($1, $2, $3, $4, now())
    ON CONFLICT (product_id) DO UPDATE
    SET chunk_text = EXCLUDED.chunk_text,
        embedding = EXCLUDED.embedding,
        metadata = EXCLUDED.metadata,
        updated_at = now();
    """

    try:
        async with pool.acquire() as conn:
            for product, chunk, vector in zip(products, chunks, vectors, strict=False):
                product_id = product.get("id")
                if not product_id:
                    continue
                metadata = {
                    "title": product.get("title"),
                    "brand": product.get("brand"),
                    "category": (product.get("category") or {}).get("title"),
                }
                await conn.execute(sql, product_id, chunk, vector, json.dumps(metadata))
    except asyncpg.PostgresError as exc:
        raise HTTPException(status_code=500, detail="Failed to persist embeddings") from exc

    return IngestResponse(indexed=len(vectors), total=len(products))
