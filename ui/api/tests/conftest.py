"""Pytest fixtures for API tests."""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """Create test client for API with proper lifespan handling."""
    with TestClient(app) as test_client:
        # Reset database for clean test state
        if hasattr(app.state, "store"):
            app.state.store.reset()
        yield test_client
