#!/usr/bin/env python3
"""Tests for context compaction in ProgressLoggingAgent."""

from __future__ import annotations

from typing import Any

from swe.environment import ensure_bootstrapped

ensure_bootstrapped()

from swe.agent_flow import ProgressLoggingAgent


class DummyModel:
    """Minimal model that records calls and returns a no-op assistant message."""

    def __init__(self, summary_text: str = "Summary of prior work."):
        self.query_calls: list[list[dict]] = []
        self.summarize_calls: list[list[dict]] = []
        self._summary_text = summary_text

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict:
        self.query_calls.append(list(messages))
        return {
            "role": "assistant",
            "content": "ok",
            "extra": {
                "actions": [],
                "cost": 0.0,
                "response": {
                    "usage": {"prompt_tokens": sum(len(str(m.get("content", ""))) // 4 for m in messages)}
                },
                "timestamp": 0.0,
            },
        }

    def summarize_context(
        self, messages: list[dict[str, Any]], summary_prompt: str, **kwargs
    ) -> dict[str, Any]:
        self.summarize_calls.append(list(messages))
        return {
            "role": "user",
            "content": self._summary_text,
            "extra": {"summary": True, "cost": 0.0, "response": {}, "timestamp": 0.0},
        }

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None):
        return []

    def get_template_vars(self, **kwargs) -> dict:
        return {}

    def serialize(self) -> dict:
        return {}


class DummyEnv:
    def execute(self, action, **kwargs):
        return {"returncode": 0, "output": "", "exception_info": None}

    def get_template_vars(self, **kwargs):
        return {"cwd": "/testbed"}

    def serialize(self):
        return {}


def _make_agent(compaction_enabled=False, token_trigger=100, keep_recent=2) -> tuple[ProgressLoggingAgent, DummyModel]:
    model = DummyModel()
    env = DummyEnv()
    agent = ProgressLoggingAgent(
        model=model,
        env=env,
        instance_id="test",
        agent_timeout=0,
        log_fn=lambda _: None,
        compaction_enabled=compaction_enabled,
        compaction_token_trigger=token_trigger,
        compaction_keep_recent_turns=keep_recent,
        compaction_summary_prompt="Summarize the conversation.",
        system_template="{{task}}",
        instance_template="{{task}}",
        step_limit=100,
    )
    return agent, model


def _seed_messages(n_turns: int = 8) -> list[dict]:
    """Build a realistic message history: system, task, then N (assistant, tool) pairs."""
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the bug in src/main.py."},
    ]
    for i in range(n_turns):
        messages.append({
            "role": "assistant",
            "content": f"Analyzing module_{i}.py, running tests...",
            "extra": {"actions": [{"command": f"cat module_{i}.py"}], "cost": 0.0},
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": f"<returncode>0</returncode>\n<output>content of module_{i}</output>",
        })
    return messages


def test_no_compaction_when_disabled():
    agent, model = _make_agent(compaction_enabled=False, token_trigger=10)
    agent.messages = _seed_messages(8)
    agent._last_prompt_tokens = 99999  # way over trigger

    original_len = len(agent.messages)
    agent.query()

    assert len(model.summarize_calls) == 0
    # Messages should have grown by 1 (the new assistant response)
    assert len(agent.messages) == original_len + 1


def test_no_compaction_when_under_trigger():
    agent, model = _make_agent(compaction_enabled=True, token_trigger=99999)
    agent.messages = _seed_messages(8)
    agent._last_prompt_tokens = 100  # under trigger

    original_len = len(agent.messages)
    agent.query()

    assert len(model.summarize_calls) == 0
    assert len(agent.messages) == original_len + 1


def test_compaction_fires_when_over_trigger():
    agent, model = _make_agent(compaction_enabled=True, token_trigger=100, keep_recent=2)
    agent.messages = _seed_messages(8)
    agent._last_prompt_tokens = 200  # over trigger

    original_len = len(agent.messages)
    agent.query()

    # Summary should have been called
    assert len(model.summarize_calls) == 1
    # Messages should be much shorter: system + task + compact summary + recent + new query response
    assert len(agent.messages) < original_len


def test_compaction_preserves_system_and_task():
    agent, model = _make_agent(compaction_enabled=True, token_trigger=100, keep_recent=2)
    agent.messages = _seed_messages(8)
    agent._last_prompt_tokens = 200

    agent.query()

    # First two messages should still be system and task
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[1]["role"] == "user"
    assert "Fix the bug" in agent.messages[1]["content"]


def test_compaction_puts_compact_summary_after_task():
    agent, model = _make_agent(compaction_enabled=True, token_trigger=100, keep_recent=2)
    agent.messages = _seed_messages(8)
    agent._last_prompt_tokens = 200

    agent.query()

    # After system + task, next should be the compact continuation message.
    assert agent.messages[2]["role"] == "user"
    assert agent.messages[2]["extra"].get("summary") is True
    assert agent.messages[2]["content"] == "Summary of prior work."


def test_compaction_keeps_complete_turn_pairs():
    agent, model = _make_agent(compaction_enabled=True, token_trigger=100, keep_recent=2)
    agent.messages = _seed_messages(8)
    agent._last_prompt_tokens = 200

    agent.query()

    # Messages: system, task, compact summary, [recent...], new_query_resp
    recent_and_new = agent.messages[3:]  # skip system, task, compact summary
    # Last message is the new query response
    recent = recent_and_new[:-1]
    for i, msg in enumerate(recent):
        if msg.get("role") == "tool":
            # The message before it should be assistant
            assert i > 0 and recent[i - 1].get("role") == "assistant"


def test_find_turn_boundary():
    msgs = [
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
        {"role": "tool", "content": "t2"},
        {"role": "assistant", "content": "a3"},
        {"role": "tool", "content": "t3"},
    ]
    # Last 2 turns → boundary at index of 2nd-to-last assistant = index 2
    assert ProgressLoggingAgent._find_turn_boundary(msgs, 2) == 2
    # Last 1 turn → boundary at last assistant = index 4
    assert ProgressLoggingAgent._find_turn_boundary(msgs, 1) == 4
    # Last 3 turns → boundary at first assistant = index 0
    assert ProgressLoggingAgent._find_turn_boundary(msgs, 3) == 0
    # More turns than exist → index 0
    assert ProgressLoggingAgent._find_turn_boundary(msgs, 10) == 0


def test_segments_capture_full_context():
    agent, model = _make_agent(compaction_enabled=True, token_trigger=100, keep_recent=2)
    agent.messages = _seed_messages(8)
    agent._last_prompt_tokens = 200

    agent.query()
    segments = agent.get_segments()

    assert len(segments) >= 2
    # First segment is solver (pre-compaction)
    assert segments[0]["kind"] == "solver"
    # Second is summarizer
    assert segments[1]["kind"] == "summarizer"
    # Summarizer segment should have the summary prompt as a user message near the end
    summarizer_msgs = segments[1]["messages"]
    assert any("Summarize" in m.get("content", "") for m in summarizer_msgs if m.get("role") == "user")
    # Each segment has full context — starts with system message
    for seg in segments:
        assert seg["messages"][0]["role"] == "system"


if __name__ == "__main__":
    tests = [
        test_no_compaction_when_disabled,
        test_no_compaction_when_under_trigger,
        test_compaction_fires_when_over_trigger,
        test_compaction_preserves_system_and_task,
        test_compaction_puts_compact_summary_after_task,
        test_compaction_keeps_complete_turn_pairs,
        test_find_turn_boundary,
        test_segments_capture_full_context,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed!")
