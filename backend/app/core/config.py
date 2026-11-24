import json
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

INSECURE_DEFAULT_JWT_SECRET = "dev-only-insecure-jwt-secret-key-change-in-production-32chars"
INSECURE_KNOWN_SECRETS = {
    INSECURE_DEFAULT_JWT_SECRET,
    "secret",
    "changeme",
    "your-secret-key",
    "replace-with-secure-random-secret",
}


def parse_cors_origins(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [origin.strip() for origin in stripped.split(",") if origin.strip()]
    if isinstance(value, list):
        return value
    raise ValueError(f"Invalid CORS_ORIGINS value: {value}")


CorsOriginsList = Annotated[list[str], NoDecode, BeforeValidator(parse_cors_origins)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_NAME: str = "StudyGuide API"
    APP_ENV: Literal["development", "testing", "staging", "production"] = "development"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    CORS_ORIGINS: CorsOriginsList = Field(
        default=["http://localhost:8081", "http://localhost:19006", "http://127.0.0.1:8081"]
    )

    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/studyguide"
    DB_ECHO: bool = False
    DB_POOL_SIZE: int = Field(default=5, ge=1, le=50)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0, le=100)

    REDIS_URL: str = "redis://localhost:6379/0"

    # Authentication & JWT
    JWT_SECRET_KEY: str = INSECURE_DEFAULT_JWT_SECRET
    JWT_ALGORITHM: Literal["HS256", "HS384", "HS512"] = "HS256"
    JWT_ISSUER: str = "studyguide-api"
    JWT_AUDIENCE: str = "studyguide-client"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, ge=1, le=1440)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=30, ge=1, le=365)

    # Document Storage & Upload Limits
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    LOCAL_STORAGE_PATH: str = ".storage/documents"
    S3_ENDPOINT_URL: str | None = None
    S3_ACCESS_KEY_ID: str | None = None
    S3_SECRET_ACCESS_KEY: str | None = None
    S3_BUCKET_NAME: str = "studyguide-documents"
    S3_REGION: str = "us-east-1"
    MAX_UPLOAD_SIZE_BYTES: int = Field(default=26_214_400, ge=1024, le=104_857_600)

    # Document Processing, Chunking, Embeddings & Background Queue (Phase 4)
    GEMINI_API_KEY: str | None = None
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_EMBEDDING_DIMENSION: int = Field(default=768, ge=64, le=3072)
    GEMINI_EMBEDDING_BATCH_SIZE: int = Field(default=100, ge=1, le=100)
    CHUNK_SIZE_CHARS: int = Field(default=1000, ge=100, le=8000)
    CHUNK_OVERLAP_CHARS: int = Field(default=200, ge=0, le=4000)
    RQ_QUEUE_NAME: str = "document_processing"
    PROCESSING_JOB_TIMEOUT_SECONDS: int = Field(default=300, ge=30, le=3600)
    PROCESSING_EAGER_FALLBACK: bool = False

    # In-Depth Analysis & Structured Generation (Phase 5)
    GEMINI_GENERATION_MODEL: str = "gemini-3.5-flash"
    ANALYSIS_MAX_DOCUMENTS: int = Field(default=10, ge=1, le=50)
    ANALYSIS_MAX_CONTEXT_CHARS: int = Field(default=32_000, ge=1000, le=200_000)
    ANALYSIS_TOP_K_CHUNKS: int = Field(default=24, ge=1, le=100)

    # Voice Practice & Gemini Live Gateway (Phase 3 & Phase 4)
    GEMINI_LIVE_MODEL: str = "gemini-2.5-flash-native-audio-latest"
    VOICE_ALLOW_IN_MEMORY_FALLBACK: bool = False
    VOICE_TICKET_TTL_SECONDS: int = Field(default=60, ge=10, le=300)
    VOICE_MAX_SESSION_DURATION_SECONDS: int = Field(default=600, ge=5, le=3600)
    VOICE_SESSION_WARNING_SECONDS: int = Field(default=570, ge=1, le=3599)
    VOICE_MAX_CONCURRENT_SESSIONS_PER_USER: int = Field(default=1, ge=1, le=5)
    VOICE_MAX_GLOBAL_CONCURRENT_SESSIONS: int = Field(default=3, ge=1, le=100)
    VOICE_MAX_DAILY_SESSIONS_PER_USER: int = Field(default=40, ge=1, le=100)
    VOICE_MAX_AUDIO_FRAME_BYTES: int = Field(default=65_536, ge=512, le=262_144)
    VOICE_MAX_JSON_MESSAGE_BYTES: int = Field(default=16_384, ge=256, le=65_536)
    VOICE_RAG_TOP_K_CHUNKS: int = Field(default=8, ge=1, le=25)
    VOICE_RAG_MAX_CONTEXT_CHARS: int = Field(default=12_000, ge=1000, le=64_000)
    VOICE_INTERVIEW_SILENCE_DURATION_MS: int = Field(
        default=3_500,
        ge=1000,
        le=30_000,
        description="Silence duration in ms before automatic end-of-speech in interview mode.",
    )
    VOICE_CONVERSATIONAL_SILENCE_DURATION_MS: int = Field(
        default=2_200,
        ge=500,
        le=10_000,
        description="Silence duration in ms before automatic end-of-speech in speaking/oral modes.",
    )
    VOICE_VAD_END_SENSITIVITY: Literal["low", "high"] = Field(
        default="low",
        description=(
            "End-of-speech sensitivity. 'low' tolerates natural pauses so long answers "
            "are not cut off mid-thought; 'high' ends turns faster."
        ),
    )
    VOICE_VAD_START_SENSITIVITY: Literal["low", "high"] = Field(
        default="high",
        description="Start-of-speech sensitivity. 'high' catches the first words quickly.",
    )
    VOICE_VAD_PREFIX_PADDING_MS: int = Field(
        default=300,
        ge=0,
        le=2_000,
        description="Audio kept before detected speech start so the first words are not clipped.",
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def normalize_database_url(cls, value: object) -> object:
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed.startswith("postgres://"):
                return "postgresql+psycopg://" + trimmed[len("postgres://") :]
            if trimmed.startswith("postgresql://"):
                return "postgresql+psycopg://" + trimmed[len("postgresql://") :]
            return trimmed
        return value

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if self.APP_ENV in ("production", "staging"):
            secret = self.JWT_SECRET_KEY.strip()
            if len(secret) < 32 or secret in INSECURE_KNOWN_SECRETS:
                raise ValueError(
                    "JWT_SECRET_KEY must be at least 32 characters and not use a development "
                    "default when APP_ENV is 'production' or 'staging'."
                )
        if self.STORAGE_BACKEND == "s3" and (
            not self.S3_ACCESS_KEY_ID or not self.S3_SECRET_ACCESS_KEY or not self.S3_BUCKET_NAME
        ):
            raise ValueError(
                "S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, and S3_BUCKET_NAME are required "
                "when STORAGE_BACKEND is 's3'."
            )
        if self.CHUNK_OVERLAP_CHARS >= self.CHUNK_SIZE_CHARS:
            raise ValueError("CHUNK_OVERLAP_CHARS must be strictly less than CHUNK_SIZE_CHARS.")
        if self.VOICE_SESSION_WARNING_SECONDS >= self.VOICE_MAX_SESSION_DURATION_SECONDS:
            raise ValueError(
                "VOICE_SESSION_WARNING_SECONDS must be strictly less than "
                "VOICE_MAX_SESSION_DURATION_SECONDS."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
