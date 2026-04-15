from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """this is the env"""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str = ""
    google_api_key: str = ""
    openai_api_base: str = ""
    openai_organization: str = ""
    openai_request_timeout: float = 60.0
    openai_model: str = "gpt-4o-mini-2024-07-18"
    gemini_model: str = "gemini-2.5-flash-lite"
    openai_embedding_model: str = "text-embedding-3-small"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    google_project: str = ""
    google_location: str = ""
    google_use_vertex: bool = False

    force_hash_embeddings: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "force_hash_embeddings", "MAGI_FORCE_HASH_EMBEDDER"
        ),
    )
    vector_db_url: str = Field(
        default="",
        validation_alias=AliasChoices("vector_db_url", "DATABASE_URL"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
