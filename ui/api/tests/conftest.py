"""Pytest fixtures for API tests."""

import pytest
from database import reset_db
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_database():
    """Reset database before each test."""
    reset_db()
    yield


@pytest.fixture
def client():
    """Create test client for API."""
    # Import here to ensure database is reset first
    from main import app

    return TestClient(app)
