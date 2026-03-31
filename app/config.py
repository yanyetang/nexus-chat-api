from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(validation_alias="DATABASE_URL")

    supplier_api_base_url: str = Field(validation_alias="SUPPLIER_API_BASE_URL")
    cohere_api_key: str = Field(validation_alias="COHERE_API_KEY")
    openrouter_api_key: str = Field(validation_alias="OPENROUTER_API_KEY")

    chatbot_api_key: str | None = Field(default=None, validation_alias="CHATBOT_API_KEY")
    openrouter_model: str = Field(
        default="google/gemini-2.0-flash-001", validation_alias="OPENROUTER_MODEL"
    )
    allowed_origins: str = Field(default="*", validation_alias="ALLOWED_ORIGINS")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
