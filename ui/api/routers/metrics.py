"""Metrics router."""

from fastapi import APIRouter, HTTPException, Request
from models import MetricsCreate, MetricsResponse

router = APIRouter(prefix="/api", tags=["metrics"])


@router.post("/metrics", response_model=MetricsResponse)
def create_metrics(request: Request, metrics: MetricsCreate):
    """Receive and store metrics from training."""
    store = request.app.state.store

    # Check session
    session = store.get_session(metrics.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    metric = store.log_metrics(metrics.session_id, metrics.step, metrics.data)
    if not metric:
        raise HTTPException(status_code=500, detail="Failed to log metrics")
    return metric


@router.get("/sessions/{session_id}/metrics", response_model=list[MetricsResponse])
def get_session_metrics(request: Request, session_id: str):
    """Get all metrics for a session."""
    store = request.app.state.store

    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return store.get_metrics(session_id)
