import json

import asyncpg

from app.exceptions import DatabaseOperationError


class RetrieverService:
    async def hybrid_search(
        self,
        pool: asyncpg.Pool,
        query_text: str,
        query_embedding: list[float],
        limit: int = 5,
        k: int = 60,
    ) -> list[dict]:
        sql = """
        WITH semantic_search AS (
            SELECT
                id,
                product_id,
                chunk_text,
                metadata,
                RANK() OVER (ORDER BY embedding <=> $1) AS rank
            FROM product_embeddings
            ORDER BY embedding <=> $1
            LIMIT 20
        ),
        keyword_search AS (
            SELECT
                id,
                product_id,
                chunk_text,
                metadata,
                RANK() OVER (
                    ORDER BY ts_rank_cd(to_tsvector('english', chunk_text), query) DESC
                ) AS rank
            FROM product_embeddings, plainto_tsquery('english', $2) query
            WHERE to_tsvector('english', chunk_text) @@ query
            ORDER BY ts_rank_cd(to_tsvector('english', chunk_text), query) DESC
            LIMIT 20
        )
        SELECT
            COALESCE(s.product_id, k.product_id) AS product_id,
            COALESCE(s.chunk_text, k.chunk_text) AS chunk_text,
            COALESCE(s.metadata, k.metadata) AS metadata,
            COALESCE(1.0 / ($3 + s.rank), 0.0) +
            COALESCE(1.0 / ($3 + k.rank), 0.0) AS score
        FROM semantic_search s
        FULL OUTER JOIN keyword_search k ON s.id = k.id
        ORDER BY score DESC
        LIMIT $4
        """

        try:
            rows = await pool.fetch(sql, query_embedding, query_text, k, limit)
        except asyncpg.PostgresError as exc:
            raise DatabaseOperationError("Hybrid search query failed") from exc
        return [
            {
                "product_id": str(row["product_id"]),
                "chunk_text": row["chunk_text"],
                "metadata": json.loads(row["metadata"])
                if isinstance(row["metadata"], str)
                else (row["metadata"] or {}),
                "score": float(row["score"]),
            }
            for row in rows
        ]
