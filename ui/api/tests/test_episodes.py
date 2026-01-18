"""Tests for episode endpoints."""

import pytest
from database import reset_db
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_database():
    """Reset database before each test."""
    reset_db()


@pytest.fixture
def test_session():
    """Create a test session."""
    response = client.post(
        "/api/sessions",
        json={"project": "test-project", "experiment": "test-exp"},
    )
    return response.json()


def test_post_episode_stores_trajectory(test_session):
    """POST /api/episodes should store episode with trajectories."""
    episode_data = {
        "session_id": test_session["id"],
        "step": 1,
        "episode_id": "task1:0",
        "task": {"question": "What is 2+2?"},
        "is_correct": True,
        "reward": 1.0,
        "trajectories": [
            {
                "uid": "abc123",
                "reward": 1.0,
                "steps": [
                    {
                        "observation": "What is 2+2?",
                        "action": "The answer is 4",
                        "reward": 1.0,
                        "done": True,
                        "chat_completions": [
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "The answer is 4"},
                        ],
                        "model_response": "The answer is 4",
                    }
                ],
            }
        ],
    }

    response = client.post("/api/episodes", json=episode_data)
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == "task1:0"
    assert data["session_id"] == test_session["id"]
    assert data["step"] == 1
    assert data["is_correct"] is True
    assert data["reward"] == 1.0
    assert data["task"] == {"question": "What is 2+2?"}
    assert "trajectories" in data["data"]
    assert len(data["data"]["trajectories"]) == 1


def test_post_episode_requires_valid_session():
    """POST /api/episodes should fail if session doesn't exist."""
    episode_data = {
        "session_id": "nonexistent",
        "step": 1,
        "episode_id": "task1:0",
        "task": {},
        "is_correct": True,
        "trajectories": [],
    }

    response = client.post("/api/episodes", json=episode_data)
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]


def test_get_episodes_returns_all_for_session(test_session):
    """GET /api/episodes should return all episodes for a session."""
    # Create multiple episodes
    for i in range(3):
        client.post(
            "/api/episodes",
            json={
                "session_id": test_session["id"],
                "step": i,
                "episode_id": f"task{i}:0",
                "task": {"question": f"Question {i}"},
                "is_correct": i % 2 == 0,
                "reward": float(i),
                "trajectories": [],
            },
        )

    response = client.get(f"/api/episodes?session_id={test_session['id']}")
    assert response.status_code == 200

    episodes = response.json()
    assert len(episodes) == 3
    assert episodes[0]["episode_id"] == "task0:0" or episodes[0]["id"] == "task0:0"


def test_get_episodes_requires_session_id():
    """GET /api/episodes should require session_id parameter."""
    response = client.get("/api/episodes")
    assert response.status_code == 422  # Validation error


def test_get_episode_by_id_returns_full_data(test_session):
    """GET /api/episodes/{id} should return full trajectory data."""
    episode_data = {
        "session_id": test_session["id"],
        "step": 1,
        "episode_id": "detailed-episode",
        "task": {"question": "Complex task"},
        "is_correct": False,
        "reward": 0.5,
        "trajectories": [
            {
                "uid": "traj1",
                "reward": 0.5,
                "steps": [
                    {
                        "observation": "obs1",
                        "action": "act1",
                        "reward": 0.5,
                        "done": False,
                    },
                    {
                        "observation": "obs2",
                        "action": "act2",
                        "reward": 0.0,
                        "done": True,
                    },
                ],
            }
        ],
    }

    # Create episode
    client.post("/api/episodes", json=episode_data)

    # Get it back
    response = client.get("/api/episodes/detailed-episode")
    assert response.status_code == 200

    episode = response.json()
    assert episode["id"] == "detailed-episode"
    assert episode["is_correct"] is False
    assert len(episode["data"]["trajectories"]) == 1
    assert len(episode["data"]["trajectories"][0]["steps"]) == 2


def test_get_episode_not_found():
    """GET /api/episodes/{id} should return 404 for nonexistent episode."""
    response = client.get("/api/episodes/nonexistent")
    assert response.status_code == 404
    assert "Episode not found" in response.json()["detail"]
