from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class OpenRouterJudge(DeepEvalBaseLLM):
    def __init__(self, model: str, api_key: str, base_url: str = _OPENROUTER_BASE_URL) -> None:
        self._model = model
        api_key = api_key.strip()
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def load_model(self) -> Any:
        return self._client

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    def get_model_name(self) -> str:
        return f"OpenRouter/{self._model}"
