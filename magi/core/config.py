from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and .env files."""

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
    provider_max_retries: int = Field(
        default=3,
        validation_alias=AliasChoices(
            "provider_max_retries", "MAGI_PROVIDER_MAX_RETRIES"
        ),
    )
    provider_retry_initial_delay: float = Field(
        default=1.0,
        validation_alias=AliasChoices(
            "provider_retry_initial_delay", "MAGI_PROVIDER_RETRY_INITIAL_DELAY"
        ),
    )
    provider_requests_per_minute: int = Field(
        default=0,
        validation_alias=AliasChoices(
            "provider_requests_per_minute", "MAGI_PROVIDER_REQUESTS_PER_MINUTE"
        ),
    )
    approve_min_citation_hit_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "approve_min_citation_hit_rate",
            "MAGI_APPROVE_MIN_CITATION_HIT_RATE",
        ),
    )
    approve_min_answer_support_score: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "approve_min_answer_support_score",
            "MAGI_APPROVE_MIN_ANSWER_SUPPORT_SCORE",
        ),
    )
    require_human_review_for_approvals: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "require_human_review_for_approvals",
            "MAGI_REQUIRE_HUMAN_REVIEW_FOR_APPROVALS",
        ),
    )
    openai_model: str = "gpt-5-mini"
    openai_fast_model: str = Field(
        default="gpt-5-mini",
        validation_alias=AliasChoices("openai_fast_model", "MAGI_OPENAI_FAST_MODEL"),
    )
    openai_strong_model: str = Field(
        default="gpt-5.2",
        validation_alias=AliasChoices(
            "openai_strong_model",
            "MAGI_OPENAI_STRONG_MODEL",
        ),
    )
    openai_high_stakes_model: str = Field(
        default="gpt-5.2",
        validation_alias=AliasChoices(
            "openai_high_stakes_model",
            "MAGI_OPENAI_HIGH_STAKES_MODEL",
        ),
    )
    enable_model_routing: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "enable_model_routing",
            "MAGI_ENABLE_MODEL_ROUTING",
        ),
    )
    enable_responder_llm: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "enable_responder_llm",
            "MAGI_ENABLE_RESPONDER_LLM",
        ),
    )
    enable_live_personas: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "enable_live_personas",
            "MAGI_ENABLE_LIVE_PERSONAS",
        ),
    )
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
    decision_trace_dir: str = Field(
        default="",
        validation_alias=AliasChoices(
            "decision_trace_dir", "MAGI_DECISION_TRACE_DIR"
        ),
    )
    run_artifact_dir: str = Field(
        default="",
        validation_alias=AliasChoices(
            "run_artifact_dir", "MAGI_RUN_ARTIFACT_DIR"
        ),
    )
    profile_dir: str = Field(
        default="",
        validation_alias=AliasChoices(
            "profile_dir", "MAGI_PROFILE_DIR"
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
