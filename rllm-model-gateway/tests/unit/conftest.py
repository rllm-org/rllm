"""Shared fixtures for rllm-model-gateway unit tests."""

import pytest

from tests.helpers.mock_sglang import MockSGLangServer
from tests.helpers.mock_vllm import MockVLLMServer


@pytest.fixture
def mock_vllm():
    """Start a mock vLLM server and yield it."""
    server = MockVLLMServer(port=0)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def mock_sglang():
    """Start a mock SGLang server (native /generate) and yield it."""
    server = MockSGLangServer(port=0)
    server.start()
    yield server
    server.stop()
