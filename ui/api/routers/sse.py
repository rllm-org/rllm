"""SSE (Server-Sent Events) router for real-time metric streaming."""

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from database import get_db

router = APIRouter(prefix="/api", tags=["sse"])


# Simple in-memory state for tracking last seen metrics
# In production, this could use Redis or similar
_last_seen_metrics: dict[str, int] = {}


async def metrics_event_generator(session_id: str) -> AsyncGenerator[str, None]:
    """Generate SSE events for new metrics."""
    last_id = _last_seen_metrics.get(session_id, 0)
    
    while True:
        # Check for new metrics
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM metrics 
                WHERE session_id = ? AND id > ?
                ORDER BY id
                """,
                (session_id, last_id),
            )
            rows = cursor.fetchall()
        
        # Send new metrics as SSE events
        for row in rows:
            data = {
                "id": row["id"],
                "step": row["step"],
                "data": json.loads(row["data"]),
                "created_at": row["created_at"],
            }
            last_id = row["id"]
            _last_seen_metrics[session_id] = last_id
            
            yield f"data: {json.dumps(data)}\n\n"
        
        # Wait before polling again
        await asyncio.sleep(0.5)


@router.get("/sessions/{session_id}/metrics/stream")
async def stream_metrics(session_id: str):
    """Stream metrics for a session via SSE.
    
    Connect to this endpoint with EventSource to receive real-time metrics.
    
    Example:
        const eventSource = new EventSource('/api/sessions/abc123/metrics/stream');
        eventSource.onmessage = (event) => {
            const metrics = JSON.parse(event.data);
            console.log(metrics);
        };
    """
    return StreamingResponse(
        metrics_event_generator(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
