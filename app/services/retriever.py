import json
from collections.abc import Iterable

import asyncpg

from app.exceptions import DatabaseOperationError


class RetrieverService:
    @staticmethod
    def _build_filter_predicates(
        filters: dict | None,
        param_offset: int,
    ) -> tuple[list[str], list[object], int]:
        if not filters:
            return [], [], param_offset

        clauses: list[str] = []
        values: list[object] = []
        next_idx = param_offset

        category = filters.get("category")
        if isinstance(category, str) and category.strip():
            clauses.append(f"metadata->>'category' = ${next_idx}")
            values.append(category.strip())
            next_idx += 1

        brand = filters.get("brand")
        if isinstance(brand, str) and brand.strip():
            clauses.append(f"metadata->>'brand' = ${next_idx}")
            values.append(brand.strip())
            next_idx += 1

        min_price = filters.get("min_price")
        if isinstance(min_price, (int, float)):
            clauses.append(
                f"COALESCE((metadata->>'price_min')::double precision, 0) >= ${next_idx}"
            )
            values.append(float(min_price))
            next_idx += 1

        max_price = filters.get("max_price")
        if isinstance(max_price, (int, float)):
            clauses.append(
                f"COALESCE((metadata->>'price_max')::double precision, 999999999) <= ${next_idx}"
            )
            values.append(float(max_price))
            next_idx += 1

        if not clauses:
            return [], [], next_idx

        return clauses, values, next_idx

    async def hybrid_search(
        self,
        pool: asyncpg.Pool,
        query_text: str,
        query_embedding: list[float] | None,
        limit: int = 5,
        k: int = 60,
        min_score: float = 0.3,
        filters: dict | None = None,
        candidate_limit: int = 20,
    ) -> list[dict]:
        uses_semantic = bool(query_embedding)
        if uses_semantic:
            # $1=embedding  $2=candidate_limit  $3=query_text  $4=cosine_min_score  $5+=filter_values  $n=k  $n+1=limit
            filter_predicates, filter_values, next_param = self._build_filter_predicates(
                filters,
                param_offset=5,
            )
            semantic_where_predicates = [
                "(1 - (embedding <=> $1)) >= $4",
                *filter_predicates,
            ]
            semantic_filter_clause = f"WHERE {' AND '.join(semantic_where_predicates)}"
            keyword_where_predicates = [
                *filter_predicates,
                "content_tsv @@ plainto_tsquery('english', $3)",
            ]
            keyword_filter_clause = f"WHERE {' AND '.join(keyword_where_predicates)}"
            sql = f"""
            WITH semantic_search AS (
                SELECT
                    id,
                    product_id,
                    chunk_text,
                    metadata,
                    RANK() OVER (ORDER BY embedding <=> $1) AS rank
                FROM product_embeddings
                {semantic_filter_clause}
                ORDER BY embedding <=> $1
                LIMIT $2
            ),
            keyword_search AS (
                SELECT
                    id,
                    product_id,
                    chunk_text,
                    metadata,
                    RANK() OVER (
                        ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', $3)) DESC
                    ) AS rank
                FROM product_embeddings
                {keyword_filter_clause}
                ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', $3)) DESC
                LIMIT $2
            )
            SELECT
                COALESCE(s.product_id, kw.product_id) AS product_id,
                COALESCE(s.chunk_text, kw.chunk_text) AS chunk_text,
                COALESCE(s.metadata, kw.metadata) AS metadata,
                COALESCE(1.0 / (${next_param} + s.rank), 0.0) +
                COALESCE(1.0 / (${next_param} + kw.rank), 0.0) AS score
            FROM semantic_search s
            FULL OUTER JOIN keyword_search kw ON s.id = kw.id
            ORDER BY score DESC
            LIMIT ${next_param + 1}
            """
            query_params: Iterable[object] = [
                query_embedding,
                candidate_limit,
                query_text,
                min_score,
                *filter_values,
                k,
                limit,
            ]
        else:
            # $1=query_text  $2=candidate_limit  $3+=filter_values  $n=limit
            filter_predicates, filter_values, next_param = self._build_filter_predicates(
                filters,
                param_offset=3,
            )
            keyword_only_where_predicates = [
                *filter_predicates,
                "content_tsv @@ plainto_tsquery('english', $1)",
            ]
            keyword_only_filter_clause = f"WHERE {' AND '.join(keyword_only_where_predicates)}"
            sql = f"""
            WITH keyword_search AS (
                SELECT
                    product_id,
                    chunk_text,
                    metadata,
                    ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS score
                FROM product_embeddings
                {keyword_only_filter_clause}
                ORDER BY score DESC
                LIMIT $2
            )
            SELECT product_id, chunk_text, metadata, score
            FROM keyword_search
            ORDER BY score DESC
            LIMIT ${next_param}
            """
            query_params = [query_text, candidate_limit, *filter_values, limit]

        try:
            rows = await pool.fetch(sql, *query_params)
        except asyncpg.PostgresError as exc:
            raise DatabaseOperationError("Hybrid search query failed") from exc

        output = [
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
        return output
