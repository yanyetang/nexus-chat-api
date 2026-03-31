"""Tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from app.models.schemas import ChatRequest, IngestResponse, SearchResponse, SearchResult


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
