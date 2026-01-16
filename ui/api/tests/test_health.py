"""Tests for health check endpoint.

TDD: Write these tests FIRST, then implement the endpoint.
"""


def test_health_check_returns_ok(client):
    """GET /api/health should return status ok."""
    response = client.get("/api/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
