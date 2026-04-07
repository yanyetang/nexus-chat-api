import asyncpg
from pgvector.asyncpg import register_vector

from app.config import get_settings
from app.exceptions import DatabaseOperationError

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def _bootstrap_schema(pool: asyncpg.Pool) -> None:
    setup_sql = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        """
        CREATE TABLE IF NOT EXISTS product_embeddings (
            id BIGSERIAL PRIMARY KEY,
            product_id TEXT UNIQUE NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding vector(1024) NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            content_tsv TSVECTOR,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            messages JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        "ALTER TABLE product_embeddings ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR",
        """
        CREATE OR REPLACE FUNCTION update_product_embeddings_tsv()
        RETURNS trigger AS $$
        BEGIN
            NEW.content_tsv := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """,
        """
        DROP TRIGGER IF EXISTS trg_product_embeddings_tsv ON product_embeddings
        """,
        """
        CREATE TRIGGER trg_product_embeddings_tsv
        BEFORE INSERT OR UPDATE ON product_embeddings
        FOR EACH ROW EXECUTE FUNCTION update_product_embeddings_tsv()
        """,
        """
        UPDATE product_embeddings
        SET content_tsv = to_tsvector('english', COALESCE(chunk_text, ''))
        WHERE content_tsv IS NULL
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_product_embeddings_tsv
            ON product_embeddings USING GIN (content_tsv)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_product_embeddings_product_id
            ON product_embeddings (product_id)
        """,
    ]

    try:
        async with pool.acquire() as conn:
            for statement in setup_sql:
                await conn.execute(statement)
            await conn.execute(
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_product_embeddings_vector
                ON product_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
    except asyncpg.PostgresError as exc:
        raise DatabaseOperationError("Database bootstrap failed") from exc


async def init_db() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=1,
            max_size=10,
            init=_init_connection,
        )
        if settings.db_auto_bootstrap:
            await _bootstrap_schema(_pool)
    return _pool


async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized. Call init_db first.")
    return _pool
