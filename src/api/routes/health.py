"""Health check endpoint."""

from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness check. Deliberately has zero dependencies on the DB or
    the agent pipeline, so it stays green even if the DB file or an LLM
    provider key is temporarily unavailable -- readiness (DB/JD/etc.) is
    checked per-route where it actually matters, not here."""
    return HealthResponse(status="ok")