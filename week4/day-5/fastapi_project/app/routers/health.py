"""Liveness (`/health`) and readiness (`/ready`) endpoints.

These are the two probes a container orchestrator (Kubernetes, ECS, etc.)
uses to decide whether to route traffic to this instance, and whether to
restart it.
"""
from fastapi import APIRouter, Depends, Response, status

from app.config import Settings, get_settings
from app.db.database import check_db_connection
from app.schemas.health import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness probe. Returns 200 as long as the process is running."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready(response: Response) -> ReadyResponse:
    """Readiness probe. Returns 200 only if all dependencies are reachable."""
    checks = {"database": check_db_connection()}
    all_ok = all(checks.values())

    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadyResponse(status="ready" if all_ok else "not_ready", checks=checks)
