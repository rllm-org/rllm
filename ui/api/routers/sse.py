"""SSE (Server-Sent Events) router for real-time metric streaming."""

import asyncio
import json
from collections.abc import AsyncGenerator

from datastore.base import DataStore
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api", tags=["sse"])


# Simple in-memory state for tracking last seen metrics
# In production, this could use Redis or similar
_last_seen_metrics: dict[str, int] = {}


async def metrics_event_generator(session_id: str, store: DataStore) -> AsyncGenerator[str, None]:
    """Generate SSE events for new metrics."""
    last_id = _last_seen_metrics.get(session_id, 0)

    while True:
        # Check for new metrics using DataStore
        new_metrics = store.get_new_metrics(session_id, last_id)

        # Send new metrics as SSE events
        for metric in new_metrics:
            last_id = metric["id"]
            _last_seen_metrics[session_id] = last_id

            # Format explicitly or pass through. The router returned id, step, data, created_at.
            # Our store returns exactly these standard columns.
            yield f"data: {json.dumps(metric)}\n\n"

        # Wait before polling again
        await asyncio.sleep(0.5)


@router.get("/sessions/{session_id}/metrics/stream")
async def stream_metrics(request: Request, session_id: str):
    """Stream metrics for a session via SSE.

    Connect to this endpoint with EventSource to receive real-time metrics.

    Example:
        const eventSource = new EventSource('/api/sessions/abc123/metrics/stream');
        eventSource.onmessage = (event) => {
            const metrics = JSON.parse(event.data);
            console.log(metrics);
        };
    """
    store = request.app.state.store
    return StreamingResponse(
        metrics_event_generator(session_id, store),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
