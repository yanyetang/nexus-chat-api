import json
from datetime import UTC, datetime
from uuid import uuid4

import asyncpg
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, status

from app.config import get_settings
from app.database import get_pool
from app.exceptions import ExternalServiceError
from app.models.schemas import IngestJobAccepted, IngestJobStatus
from app.services.embeddings import EmbeddingService
from app.services.supplier import SupplierService
from app.utils.chunking import product_to_chunk

router = APIRouter(prefix="/ingest", tags=["ingest"])

_ingest_jobs: dict[str, dict] = {}


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _run_ingest_job(job_id: str) -> None:
    supplier = SupplierService()
    embeddings = EmbeddingService()
    pool = get_pool()

    _ingest_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "indexed": 0,
        "total": 0,
        "error": None,
        "updated_at": datetime.now(UTC).isoformat(),
    }

    try:
        products = await supplier.fetch_catalog_export()
        chunks = [product_to_chunk(product) for product in products]
        vectors = await embeddings.embed_texts(chunks, input_type="search_document")

        if len(chunks) != len(vectors):
            raise ValueError("Embedding count mismatch")

        sql = """
        INSERT INTO product_embeddings (product_id, chunk_text, embedding, metadata, updated_at)
        VALUES ($1, $2, $3, $4, now())
        ON CONFLICT (product_id) DO UPDATE
        SET chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            updated_at = now();
        """

        indexed = 0
        async with pool.acquire() as conn:
            for product, chunk, vector in zip(products, chunks, vectors, strict=False):
                product_id = product.get("id")
                if not product_id:
                    continue

                variant_prices = [
                    cast_price
                    for cast_price in (
                        _to_float(variant.get("price"))
                        for variant in (product.get("variants") or [])
                    )
                    if cast_price is not None
                ]
                metadata = {
                    "title": product.get("title"),
                    "brand": product.get("brand"),
                    "category": (product.get("category") or {}).get("title"),
                    "price_min": min(variant_prices) if variant_prices else None,
                    "price_max": max(variant_prices) if variant_prices else None,
                }
                await conn.execute(sql, str(product_id), chunk, vector, json.dumps(metadata))
                indexed += 1

        _ingest_jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "indexed": indexed,
            "total": len(products),
            "error": None,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    except (ExternalServiceError, asyncpg.PostgresError, ValueError) as exc:
        _ingest_jobs[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "indexed": _ingest_jobs.get(job_id, {}).get("indexed", 0),
            "total": _ingest_jobs.get(job_id, {}).get("total", 0),
            "error": str(exc),
            "updated_at": datetime.now(UTC).isoformat(),
        }


@router.post("", response_model=IngestJobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def ingest_catalog(
    background_tasks: BackgroundTasks,
    authorization: str | None = Header(default=None),
) -> IngestJobAccepted:
    settings = get_settings()
    if settings.chatbot_api_key:
        expected = f"Bearer {settings.chatbot_api_key}"
        if authorization != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    get_pool()
    job_id = str(uuid4())
    _ingest_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "indexed": 0,
        "total": 0,
        "error": None,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    background_tasks.add_task(_run_ingest_job, job_id)
    return IngestJobAccepted(job_id=job_id, status="queued")


@router.get("/{job_id}/status", response_model=IngestJobStatus)
async def ingest_status(job_id: str) -> IngestJobStatus:
    job = _ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return IngestJobStatus(**job)
