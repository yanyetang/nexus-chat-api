from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default model names — import these in tests instead of duplicating the strings
GROQ_JUDGE_MODEL_DEFAULT = "llama-3.3-70b-versatile"
OPENROUTER_JUDGE_MODEL_DEFAULT = "openai/gpt-4o-mini"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(validation_alias="DATABASE_URL")

    supplier_api_base_url: str = Field(validation_alias="SUPPLIER_API_BASE_URL")
    cohere_api_key: str = Field(validation_alias="COHERE_API_KEY")
    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")

    chatbot_api_key: str | None = Field(default=None, validation_alias="CHATBOT_API_KEY")
    openrouter_chat_model: str = Field(
        default="openai/gpt-4o-mini", validation_alias="OPENROUTER_CHAT_MODEL"
    )
    openrouter_judge_model: str = Field(
        default=OPENROUTER_JUDGE_MODEL_DEFAULT, validation_alias="OPENROUTER_JUDGE_MODEL"
    )
    openrouter_optimizer_model: str = Field(
        default="openrouter/openai/gpt-4o-mini",  # fallback if GROQ_API_KEY not set; Groq is preferred
        validation_alias="OPENROUTER_OPTIMIZER_MODEL",
    )
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_optimizer_model: str = Field(
        default="groq/llama-3.3-70b-versatile", validation_alias="GROQ_OPTIMIZER_MODEL"
    )
    groq_judge_model: str = Field(
        default=GROQ_JUDGE_MODEL_DEFAULT, validation_alias="GROQ_JUDGE_MODEL"
    )
    allowed_origins: str = Field(default="*", validation_alias="ALLOWED_ORIGINS")
    retrieval_min_score: float = Field(default=0.3, validation_alias="RETRIEVAL_MIN_SCORE")
    retrieval_candidate_limit: int = Field(default=20, validation_alias="RETRIEVAL_CANDIDATE_LIMIT")
    cohere_rerank_enabled: bool = Field(default=True, validation_alias="COHERE_RERANK_ENABLED")
    cohere_rerank_top_n: int = Field(default=5, validation_alias="COHERE_RERANK_TOP_N")
    cohere_rerank_min_score: float = Field(default=0.1, validation_alias="COHERE_RERANK_MIN_SCORE")
    db_auto_bootstrap: bool = Field(default=True, validation_alias="DB_AUTO_BOOTSTRAP")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
