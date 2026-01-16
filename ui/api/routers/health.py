"""Health check router."""

from fastapi import APIRouter

from models import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Return health status of the API."""
    return {"status": "ok"}
