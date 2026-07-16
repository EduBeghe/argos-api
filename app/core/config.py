from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = Field(
        validation_alias=AliasChoices("DATABASE_URL", "ARGOS_DATABASE_URL")
    )
    service_name: str = Field(default="argos-api", validation_alias="SERVICE_NAME")
    version: str = Field(default="0.1.0", validation_alias="APP_VERSION")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    binance_base_url: str = Field(
        default="https://api.binance.com", validation_alias="BINANCE_BASE_URL"
    )
    binance_connect_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        validation_alias="BINANCE_CONNECT_TIMEOUT_SECONDS",
    )
    binance_read_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        validation_alias="BINANCE_READ_TIMEOUT_SECONDS",
    )
    binance_ingestion_limit: int = Field(
        default=12,
        ge=1,
        le=1000,
        validation_alias="BINANCE_INGESTION_LIMIT",
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def use_asyncpg_driver(cls, value: object) -> str:
        """Normalize standard PostgreSQL URLs for SQLAlchemy asyncpg usage."""

        if not isinstance(value, str):
            raise TypeError("DATABASE_URL must be a string")
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+asyncpg://", 1)
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the cached process-wide settings."""

    return Settings()
