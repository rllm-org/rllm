"""Unit tests for SessionManager image state — set/get/delete lifecycle."""

import asyncio

import pytest
from rllm_model_gateway.session_manager import SessionManager
from rllm_model_gateway.store.memory_store import MemoryTraceStore


@pytest.fixture
def sm():
    return SessionManager(MemoryTraceStore())


class TestSessionImages:
    def test_get_empty_before_set(self, sm):
        assert sm.get_images("nope") == []

    def test_set_then_get(self, sm):
        sm.set_images("sid1", ["data:image/png;base64,A"])
        assert sm.get_images("sid1") == ["data:image/png;base64,A"]

    def test_set_overwrites_not_appends(self, sm):
        """C3 fix — Codex CLI resends full history per turn, so set() is
        idempotent across turns rather than accumulating duplicates."""
        sm.set_images("sid1", ["data:image/png;base64,A"])
        # Simulate turn 2 request that includes turn 1 image + a new one
        sm.set_images("sid1", ["data:image/png;base64,A", "data:image/png;base64,B"])
        assert sm.get_images("sid1") == ["data:image/png;base64,A", "data:image/png;base64,B"]

    def test_set_idempotent_when_history_unchanged(self, sm):
        """If Codex resends the same 2 images in turn 3, we still see 2, not 4."""
        sm.set_images("sid1", ["A", "B"])
        sm.set_images("sid1", ["A", "B"])
        assert sm.get_images("sid1") == ["A", "B"]

    def test_set_isolates_sessions(self, sm):
        sm.set_images("sidA", ["A"])
        sm.set_images("sidB", ["B"])
        assert sm.get_images("sidA") == ["A"]
        assert sm.get_images("sidB") == ["B"]

    def test_set_stores_copy_not_reference(self, sm):
        urls = ["A", "B"]
        sm.set_images("sid1", urls)
        urls.append("C")
        assert sm.get_images("sid1") == ["A", "B"], "SessionManager should not share list mutation with caller"

    def test_delete_session_clears_images(self, sm):
        sm.set_images("sid1", ["A"])
        assert sm.get_images("sid1") == ["A"]
        asyncio.run(sm.delete_session("sid1"))
        assert sm.get_images("sid1") == []

    def test_delete_session_leaves_others_alone(self, sm):
        sm.set_images("sidA", ["A"])
        sm.set_images("sidB", ["B"])
        asyncio.run(sm.delete_session("sidA"))
        assert sm.get_images("sidA") == []
        assert sm.get_images("sidB") == ["B"]
