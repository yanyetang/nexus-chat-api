"""Tests for FastAPI routes using TestClient (no live DB/API calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_missing_query():
    resp = client.get("/search")
    assert resp.status_code == 422


def test_search_query_too_short():
    resp = client.get("/search?q=")
    assert resp.status_code == 422


def test_chat_missing_body():
    resp = client.post("/chat", json={})
    assert resp.status_code == 422


def test_chat_empty_session_id():
    resp = client.post("/chat", json={"session_id": "", "message": "hello"})
    assert resp.status_code == 422


def test_chat_empty_message():
    resp = client.post("/chat", json={"session_id": "s1", "message": ""})
    assert resp.status_code == 422


@patch("app.routers.search.EmbeddingService")
@patch("app.routers.search.RetrieverService")
@patch("app.routers.search.get_pool")
def test_search_returns_results(mock_pool, mock_retriever_cls, mock_embedding_cls):
    mock_embedding_cls.return_value.embed_texts = AsyncMock(return_value=[[0.1] * 1024])
    mock_retriever_cls.return_value.hybrid_search = AsyncMock(
        return_value=[
            {
                "product_id": "prod-1",
                "chunk_text": "Product: T-Shirt",
                "metadata": {"title": "T-Shirt"},
                "score": 0.95,
            }
        ]
    )
    mock_pool.return_value = MagicMock()

    resp = client.get("/search?q=cotton+shirt")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["product_id"] == "prod-1"


@patch("app.routers.search.EmbeddingService")
@patch("app.routers.search.get_pool")
@patch("app.routers.search.RetrieverService")
def test_search_falls_back_when_embedding_fails(mock_retriever_cls, mock_pool, mock_embedding_cls):
    from app.exceptions import ExternalServiceError

    mock_embedding_cls.return_value.embed_texts = AsyncMock(
        side_effect=ExternalServiceError("Embedding provider unavailable")
    )
    mock_retriever_cls.return_value.hybrid_search = AsyncMock(return_value=[])
    mock_pool.return_value = MagicMock()

    resp = client.get("/search?q=shirt")
    assert resp.status_code == 200
    assert resp.json() == {"results": []}


@patch("app.routers.search.EmbeddingService")
@patch("app.routers.search.RetrieverService")
@patch("app.routers.search.get_pool")
def test_search_post_query_with_filters(mock_pool, mock_retriever_cls, mock_embedding_cls):
    mock_embedding_cls.return_value.embed_texts = AsyncMock(return_value=[[0.1] * 1024])
    mock_retriever_cls.return_value.hybrid_search = AsyncMock(
        return_value=[
            {
                "product_id": "prod-2",
                "chunk_text": "Product: Hoodie",
                "metadata": {"title": "Hoodie", "brand": "Acme"},
                "score": 0.82,
            }
        ]
    )
    mock_pool.return_value = MagicMock()

    resp = client.post(
        "/search/query",
        json={
            "query": "acme hoodie",
            "limit": 5,
            "filters": {"brand": "Acme", "max_price": 99},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"][0]["product_id"] == "prod-2"
