"""Sessions router."""

import json
import uuid
from datetime import datetime, UTC

from fastapi import APIRouter, HTTPException

from database import get_db
from models import SessionCreate, SessionResponse

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
def create_session(session: SessionCreate):
    """Create a new training session."""
    session_id = str(uuid.uuid4())

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (id, project, experiment, config, source_metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                session.project,
                session.experiment,
                json.dumps(session.config) if session.config else None,
                json.dumps(session.source_metadata) if session.source_metadata else None,
            ),
        )
        conn.commit()
        
        # Fetch the created session
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
    
    return _row_to_session(row)


@router.get("", response_model=list[SessionResponse])
def list_sessions():
    """List all sessions."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
        rows = cursor.fetchall()
    
    return [_row_to_session(row) for row in rows]


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    """Get a specific session by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
    
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return _row_to_session(row)


@router.post("/{session_id}/complete", response_model=SessionResponse)
def complete_session(session_id: str):
    """Mark a session as completed."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if session exists
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update session
        now = datetime.now(UTC).isoformat()
        cursor.execute(
            """
            UPDATE sessions
            SET completed_at = ?
            WHERE id = ?
            """,
            (now, session_id),
        )
        conn.commit()
        
        # Fetch updated session
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
    
    return _row_to_session(row)


def _row_to_session(row) -> dict:
    """Convert a database row to a session dict."""
    return {
        "id": row["id"],
        "project": row["project"],
        "experiment": row["experiment"],
        "config": json.loads(row["config"]) if row["config"] else None,
        "source_metadata": json.loads(row["source_metadata"]) if row["source_metadata"] else None,
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
    }
