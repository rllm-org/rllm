"""Health check router."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health_check(request: Request):
    """Return health status of the API."""
    store = request.app.state.store
    store_type = type(store).__name__
    return {
        "status": "ok",
        "datastore": store_type,  # "SQLiteStore" or "PostgresStore"
    }
