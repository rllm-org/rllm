"""Tests for metrics endpoints.

TDD: Write these tests FIRST, then implement the endpoints.
"""


def test_post_metrics(client):
    """POST /api/metrics should store metric data."""
    # First create a session
    session_response = client.post("/api/sessions", json={
        "project": "test-project",
        "experiment": "run-1"
    })
    session_id = session_response.json()["id"]
    
    # Post metrics
    response = client.post("/api/metrics", json={
        "session_id": session_id,
        "step": 1,
        "data": {"reward/mean": 0.5, "loss": 0.1}
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["step"] == 1
    assert data["data"]["reward/mean"] == 0.5


def test_post_metrics_invalid_session(client):
    """POST /api/metrics should return 404 for invalid session."""
    response = client.post("/api/metrics", json={
        "session_id": "nonexistent",
        "step": 1,
        "data": {"reward/mean": 0.5}
    })
    
    assert response.status_code == 404


def test_get_session_metrics(client):
    """GET /api/sessions/{id}/metrics should return all metrics for session."""
    # Create session
    session_response = client.post("/api/sessions", json={
        "project": "test-project",
        "experiment": "run-1"
    })
    session_id = session_response.json()["id"]
    
    # Post multiple metrics
    client.post("/api/metrics", json={
        "session_id": session_id,
        "step": 1,
        "data": {"reward/mean": 0.5}
    })
    client.post("/api/metrics", json={
        "session_id": session_id,
        "step": 2,
        "data": {"reward/mean": 0.7}
    })
    
    # Get metrics
    response = client.get(f"/api/sessions/{session_id}/metrics")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["step"] == 1
    assert data[1]["step"] == 2
