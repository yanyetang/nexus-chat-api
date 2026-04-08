"""Tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ChatRequest,
    IngestResponse,
    RetrievalFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
)


def test_chat_request_valid():
    req = ChatRequest(session_id="abc", message="hello")
    assert req.session_id == "abc"
    assert req.locale is None


def test_chat_request_empty_message_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc", message="")


def test_chat_request_empty_session_rejected():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="", message="hello")


def test_search_result_defaults():
    result = SearchResult(product_id="123", chunk_text="A shirt", score=0.9)
    assert result.metadata == {}


def test_search_response_empty():
    resp = SearchResponse(results=[])
    assert resp.results == []


def test_ingest_response():
    resp = IngestResponse(indexed=145, total=145)
    assert resp.indexed == 145


def test_search_request_with_filters_valid():
    req = SearchRequest(
        query="running shoes",
        filters=RetrievalFilters(category="Footwear", min_price=10, max_price=120),
    )
    assert req.filters is not None
    assert req.filters.category == "Footwear"


def test_chat_request_filters_valid():
    req = ChatRequest(
        session_id="s1",
        message="show acme products",
        filters=RetrievalFilters(brand="Acme"),
    )
    assert req.filters is not None
    assert req.filters.brand == "Acme"
