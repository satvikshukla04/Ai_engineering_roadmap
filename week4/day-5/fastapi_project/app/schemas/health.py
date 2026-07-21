"""Pydantic schemas for the health and readiness endpoints."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness response — the process is up and able to respond."""

    status: str = "ok"
    app_name: str
    version: str
    environment: str


class ReadyResponse(BaseModel):
    """Readiness response — the app and its dependencies are ready to serve traffic."""

    status: str
    checks: dict[str, bool]
