"""Metrics router."""

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException

from database import get_db
from models import MetricsCreate, MetricsResponse

router = APIRouter(prefix="/api", tags=["metrics"])


@router.post("/metrics", response_model=MetricsResponse)
def create_metrics(metrics: MetricsCreate):
    """Receive and store metrics from training."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if session exists
        cursor.execute("SELECT id FROM sessions WHERE id = ?", (metrics.session_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Insert metrics
        cursor.execute(
            """
            INSERT INTO metrics (session_id, step, data)
            VALUES (?, ?, ?)
            """,
            (metrics.session_id, metrics.step, json.dumps(metrics.data)),
        )
        conn.commit()
        
        # Get the created record
        metrics_id = cursor.lastrowid
        cursor.execute("SELECT * FROM metrics WHERE id = ?", (metrics_id,))
        row = cursor.fetchone()
    
    return _row_to_metrics(row)


@router.get("/sessions/{session_id}/metrics", response_model=list[MetricsResponse])
def get_session_metrics(session_id: str):
    """Get all metrics for a session."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if session exists
        cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get metrics
        cursor.execute(
            "SELECT * FROM metrics WHERE session_id = ? ORDER BY step",
            (session_id,),
        )
        rows = cursor.fetchall()
    
    return [_row_to_metrics(row) for row in rows]


def _row_to_metrics(row) -> dict:
    """Convert a database row to a metrics dict."""
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "step": row["step"],
        "data": json.loads(row["data"]),
        "created_at": row["created_at"],
    }
