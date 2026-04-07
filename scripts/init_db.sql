CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS product_embeddings (
    id BIGSERIAL PRIMARY KEY,
    product_id TEXT UNIQUE NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_tsv TSVECTOR,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION update_product_embeddings_tsv()
RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_product_embeddings_tsv ON product_embeddings;

CREATE TRIGGER trg_product_embeddings_tsv
BEFORE INSERT OR UPDATE ON product_embeddings
FOR EACH ROW EXECUTE FUNCTION update_product_embeddings_tsv();

UPDATE product_embeddings
SET content_tsv = to_tsvector('english', COALESCE(chunk_text, ''))
WHERE content_tsv IS NULL;

CREATE INDEX IF NOT EXISTS idx_product_embeddings_tsv
    ON product_embeddings USING GIN (content_tsv);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_product_embeddings_vector
    ON product_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
