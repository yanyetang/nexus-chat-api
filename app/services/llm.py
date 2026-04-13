import json
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings
from app.exceptions import ExternalServiceError


class LLMService:
    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.openrouter_api_key
        self._model = settings.openrouter_chat_model
        self._url = "https://openrouter.ai/api/v1/chat/completions"

    async def _try_stream(
        self, client: httpx.AsyncClient, messages: list[dict]
    ) -> AsyncIterator[str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self._model, "messages": messages, "stream": True}

        async with client.stream("POST", self._url, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    json_data = json.loads(data)
                except ValueError:
                    continue

                choices = json_data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        if not self._api_key:
            raise ExternalServiceError("OpenRouter API key not configured")

        async with httpx.AsyncClient(timeout=120) as client:
            try:
                async for token in self._try_stream(client, messages):
                    yield token
            except httpx.HTTPStatusError as exc:
                raise ExternalServiceError(
                    f"LLM provider error {exc.response.status_code}"
                ) from exc
            except httpx.HTTPError as exc:
                raise ExternalServiceError(f"LLM provider unavailable: {exc}") from exc
