import json

import asyncpg
from fastapi import APIRouter, Header, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.database import get_pool
from app.exceptions import ExternalServiceError
from app.models.schemas import ChatRequest
from app.services.rag import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])


def _format_sse(data: dict) -> dict:
    return {"event": data.get("type", "message"), "data": json.dumps(data)}


@router.post("")
async def chat(
    payload: ChatRequest, authorization: str | None = Header(default=None)
) -> EventSourceResponse:
    settings = get_settings()
    if settings.chatbot_api_key:
        expected = f"Bearer {settings.chatbot_api_key}"
        if authorization != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    pool = get_pool()
    rag = RAGService()

    history: list[dict] = []
    try:
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT messages FROM chat_sessions WHERE session_id = $1",
                payload.session_id,
            )
            if existing and existing["messages"]:
                raw_messages = existing["messages"]
                if isinstance(raw_messages, str):
                    history = json.loads(raw_messages)
                elif isinstance(raw_messages, list):
                    history = raw_messages
    except (asyncpg.PostgresError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Failed to load chat session") from exc

    async def event_generator():
        response_text = ""
        try:
            results, retrieval_info = await rag.get_context(
                pool=pool,
                message=payload.message,
                filters=(
                    payload.filters.model_dump(exclude_none=True) if payload.filters else None
                ),
            )
            if retrieval_info:
                yield _format_sse({"type": "info", "content": retrieval_info})
            if results:
                sources = [
                    {
                        "product_id": item.get("product_id"),
                        "title": (item.get("metadata") or {}).get("title"),
                        "score": item.get("score"),
                    }
                    for item in results
                ]
                yield _format_sse({"type": "sources", "products": sources})
            else:
                yield _format_sse(
                    {
                        "type": "info",
                        "content": "No strong product match found in retrieval context.",
                    }
                )

            async for token in rag.stream_answer(
                history=history, message=payload.message, results=results
            ):
                response_text += token
                yield _format_sse({"type": "token", "content": token})
        except ExternalServiceError as exc:
            yield _format_sse({"type": "error", "content": str(exc)})
            yield _format_sse({"type": "done"})
            return
        except asyncpg.PostgresError:
            yield _format_sse({"type": "error", "content": "Database query failed"})
            yield _format_sse({"type": "done"})
            return

        updated_messages = [
            *history,
            {"role": "user", "content": payload.message},
            {"role": "assistant", "content": response_text},
        ]
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO chat_sessions (session_id, messages, updated_at)
                    VALUES ($1, $2, now())
                    ON CONFLICT (session_id) DO UPDATE
                    SET messages = EXCLUDED.messages,
                        updated_at = now();
                    """,
                    payload.session_id,
                    json.dumps(updated_messages),
                )
        except asyncpg.PostgresError:
            yield _format_sse({"type": "error", "content": "Failed to persist chat session"})
            yield _format_sse({"type": "done"})
            return

        yield _format_sse({"type": "done"})

    return EventSourceResponse(event_generator())
