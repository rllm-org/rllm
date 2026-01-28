"""Sessions router."""

from fastapi import APIRouter, HTTPException, Request
from models import SessionCreate, SessionResponse

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
def create_session(request: Request, session: SessionCreate):
    """Create a new training session."""
    store = request.app.state.store
    session_id = store.create_session(project=session.project, experiment=session.experiment, config=session.config, source_metadata=session.source_metadata)
    return store.get_session(session_id)


@router.get("", response_model=list[SessionResponse])
def list_sessions(request: Request):
    """List all sessions."""
    store = request.app.state.store
    return store.get_all_sessions()


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(request: Request, session_id: str):
    """Get a specific session by ID."""
    store = request.app.state.store
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{session_id}/complete", response_model=SessionResponse)
def complete_session(request: Request, session_id: str):
    """Mark a session as completed."""
    store = request.app.state.store
    session = store.complete_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
