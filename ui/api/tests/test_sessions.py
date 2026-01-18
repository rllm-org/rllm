"""Tests for session endpoints.

TDD: Write these tests FIRST, then implement the endpoints.
"""


def test_create_session(client):
    """POST /api/sessions should create and return a session."""
    response = client.post("/api/sessions", json={"project": "test-project", "experiment": "run-1", "config": {"learning_rate": 0.001}})

    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["project"] == "test-project"
    assert data["experiment"] == "run-1"
    assert data["status"] == "running"


def test_create_session_minimal(client):
    """POST /api/sessions should work with just project and experiment."""
    response = client.post("/api/sessions", json={"project": "test-project", "experiment": "run-1"})

    assert response.status_code == 200
    data = response.json()
    assert data["config"] is None


def test_list_sessions(client):
    """GET /api/sessions should list all sessions."""
    # Create two sessions
    client.post("/api/sessions", json={"project": "project-1", "experiment": "run-1"})
    client.post("/api/sessions", json={"project": "project-2", "experiment": "run-2"})

    response = client.get("/api/sessions")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_get_session(client):
    """GET /api/sessions/{id} should return session details."""
    # Create a session
    create_response = client.post("/api/sessions", json={"project": "test-project", "experiment": "run-1"})
    session_id = create_response.json()["id"]

    # Get the session
    response = client.get(f"/api/sessions/{session_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == session_id
    assert data["project"] == "test-project"


def test_get_session_not_found(client):
    """GET /api/sessions/{id} should return 404 for unknown session."""
    response = client.get("/api/sessions/nonexistent-id")

    assert response.status_code == 404


def test_complete_session(client):
    """POST /api/sessions/{id}/complete should mark session as completed."""
    # Create a session
    create_response = client.post("/api/sessions", json={"project": "test-project", "experiment": "run-1"})
    session_id = create_response.json()["id"]

    # Complete the session
    response = client.post(f"/api/sessions/{session_id}/complete")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["completed_at"] is not None
