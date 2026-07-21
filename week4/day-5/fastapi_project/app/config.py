"""Centralized application configuration.

All runtime configuration is defined here using Pydantic Settings so it can
be supplied via environment variables (or a `.env` file locally) rather than
hard-coded. This is the single source of truth for config throughout the app.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App metadata ---
    app_name: str = Field(default="Documents API")
    app_version: str = Field(default="1.0.0")
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=False)

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # --- Database ---
    database_url: str = Field(
        default="sqlite:///./data/documents.db",
        description="SQLAlchemy database URL",
    )
    database_echo: bool = Field(default=False)

    # --- CORS ---
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Logging ---
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
