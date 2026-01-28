"""Episodes router - handles episode/trajectory data."""

from fastapi import APIRouter, HTTPException, Query, Request
from models import EpisodeCreate, EpisodeResponse, EpisodeSearchResponse

router = APIRouter(prefix="/api", tags=["episodes"])


@router.post("/episodes", response_model=EpisodeResponse)
def create_episode(request: Request, episode: EpisodeCreate):
    """Receive and store episode data with trajectories."""
    store = request.app.state.store

    # Check if session exists (store handles this internally or we let FK fail?
    # SQLiteStore doesn't strictly enforce in create logic unless we added it.
    # But usually we want to return 404 if session missing.
    # The original code checked explicitly. DataStore should optionally check or we rely on logic.
    # Let's assume store handles basic validation, but SQLiteStore didn't explicitly check session existence in append_episode.
    # We can add check here if we want strictness, via store.get_session.

    session = store.get_session(episode.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Pass the full dict
    episode_data = episode.model_dump(mode="json")
    store.append_episode(episode.session_id, episode_data)

    return store.get_episode(episode.episode_id)


@router.get("/episodes", response_model=list[EpisodeResponse])
def get_episodes(request: Request, session_id: str = Query(..., description="Filter episodes by session ID")):
    """Query episodes by session ID."""
    store = request.app.state.store

    # Check session
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return store.get_episodes(session_id)


@router.get("/episodes/search", response_model=EpisodeSearchResponse)
def search_episodes(
    request: Request,
    q: str = Query(..., description="Search query"),
    session_id: str | None = Query(None, description="Optional session ID to filter results"),
    step: int | None = Query(None, description="Optional step number to filter results"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
):
    """Search episodes by text content.

    Searches through task descriptions, observations, actions, and model responses.
    PostgreSQL backend provides full-text search with stemming and relevance ranking.

    Returns:
        EpisodeSearchResponse with:
        - episodes: List of matching episodes (PostgreSQL includes rank field)
        - matched_terms: Stemmed terms (PostgreSQL) or original query terms (SQLite)
    """
    store = request.app.state.store
    return store.search_episodes(q, session_id, limit, step)


@router.get("/episodes/{episode_id}", response_model=EpisodeResponse)
def get_episode(request: Request, episode_id: str):
    """Get a single episode with full trajectory data."""
    store = request.app.state.store
    episode = store.get_episode(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode
