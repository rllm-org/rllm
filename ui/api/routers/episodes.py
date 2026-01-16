"""Episodes router - handles episode/trajectory data."""

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from database import get_db
from models import EpisodeCreate, EpisodeResponse

router = APIRouter(prefix="/api", tags=["episodes"])


@router.post("/episodes", response_model=EpisodeResponse)
def create_episode(episode: EpisodeCreate):
    """Receive and store episode data with trajectories."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Check if session exists
        cursor.execute("SELECT id FROM sessions WHERE id = ?", (episode.session_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Session not found")

        # Serialize trajectories and create full data object
        data = {
            "trajectories": [traj.model_dump() for traj in episode.trajectories],
            "info": episode.info if hasattr(episode, 'info') else {}
        }

        # Insert episode
        cursor.execute(
            """
            INSERT INTO episodes (id, session_id, step, task, is_correct, reward, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                episode.session_id,
                episode.step,
                json.dumps(episode.task),
                episode.is_correct,
                episode.reward,
                json.dumps(data),
            ),
        )
        conn.commit()

        # Get the created record
        cursor.execute("SELECT * FROM episodes WHERE id = ?", (episode.episode_id,))
        row = cursor.fetchone()

    return _row_to_episode(row)


@router.get("/episodes", response_model=list[EpisodeResponse])
def get_episodes(session_id: str = Query(..., description="Filter episodes by session ID")):
    """Query episodes by session ID."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Check if session exists
        cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get episodes for this session
        cursor.execute(
            "SELECT * FROM episodes WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        rows = cursor.fetchall()

    return [_row_to_episode(row) for row in rows]


@router.get("/episodes/{episode_id}", response_model=EpisodeResponse)
def get_episode(episode_id: str):
    """Get a single episode with full trajectory data."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        row = cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    return _row_to_episode(row)


def _row_to_episode(row) -> dict:
    """Convert a database row to an episode dict."""
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "step": row["step"],
        "task": json.loads(row["task"]),
        "is_correct": bool(row["is_correct"]),
        "reward": row["reward"],
        "data": json.loads(row["data"]),
        "created_at": row["created_at"],
    }
