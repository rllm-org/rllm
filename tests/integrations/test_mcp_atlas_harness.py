from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from rllm.integrations.mcp_atlas.constants import DEFAULT_SERVERS, GATEWAY_API_KEY, IMAGE
from rllm.integrations.mcp_atlas.harness import MCPAtlasHarness
from rllm.types import AgentConfig, Task
from rllm.workflows.workflow import TerminationReason


class _Response:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = "response text"
        self.ok = status_code < 400

    def json(self):
        return self._data


def _task(task_id="atlas-1"):
    return Task(
        id=task_id,
        instruction="Find the answer",
        metadata={
            "TASK": task_id,
            "PROMPT": "Use the tools and report the result",
            "ENABLED_TOOLS": ["filesystem_read_file", "git_git_status"],
            "GTFA_CLAIMS": ["The answer is present"],
        },
        dataset_dir=Path("."),
    )


def _configured_harness():
    harness = MCPAtlasHarness()
    harness._manager = SimpleNamespace(harness_url="http://harness.test")
    return harness


def test_harness_request_matches_official_contract(monkeypatch):
    posted = {}
    events = [
        {"type": "message", "data": {"role": "user", "content": "Use the tools and report the result"}},
        {
            "type": "message",
            "data": {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "filesystem_read_file"}}]},
        },
        {"type": "message", "data": {"role": "tool", "tool_call_id": "1", "content": "raw tool output"}},
        {"type": "message", "data": {"role": "assistant", "content": "Exact final response"}},
    ]

    def fake_post(url, **kwargs):
        posted.update(url=url, **kwargs)
        return _Response(events)

    monkeypatch.setattr("rllm.integrations.mcp_atlas.harness.requests.post", fake_post)
    harness = _configured_harness()
    episode = harness.run(
        _task(),
        AgentConfig(
            base_url="http://gateway/sessions/s-1/v1",
            model="accounts/fireworks/models/glm-5p2",
            session_uid="s-1",
            sampling_params={"temperature": 0.2},
        ),
    )

    assert posted["url"] == "http://harness.test/v2/mcp_eval/run_agent"
    assert posted["timeout"] == 1800.0
    assert posted["json"] == {
        "task_id": "atlas-1",
        "model": "fireworks_ai/accounts/fireworks/models/glm-5p2",
        "messages": [{"role": "user", "content": "Use the tools and report the result"}],
        "enabledTools": ["filesystem_read_file", "git_git_status"],
        "image": IMAGE,
        "tags": {"task_id": "atlas-1", "rllm_session_uid": "s-1"},
        "max_turns": 256,
        "max_tool_calls": 100,
        "llm_base_url": "http://gateway/sessions/s-1/v1",
        "extra_llm_params": {"api_key": GATEWAY_API_KEY},
    }
    assert episode.artifacts["answer"] == "Exact final response"
    assert episode.artifacts["raw_conversation_history"][-1]["content"] == "Exact final response"
    assert episode.artifacts["mcp_atlas"]["tool_call_count"] == 1
    assert len(episode.artifacts["mcp_atlas"]["tool_trajectory"]) == 2
    assert episode.termination_reason is None


def test_harness_records_terminal_errors(monkeypatch):
    events = [
        {"type": "message", "data": {"role": "assistant", "content": "partial"}},
        {"type": "error", "data": {"reason": "max_turns_reached", "maxTurns": 256}},
    ]
    monkeypatch.setattr("rllm.integrations.mcp_atlas.harness.requests.post", lambda *args, **kwargs: _Response(events))

    episode = _configured_harness().run(
        _task(),
        AgentConfig(base_url="http://gateway/v1", model="model", session_uid="s-error"),
    )

    assert episode.termination_reason == TerminationReason.ERROR
    assert episode.artifacts["raw_harness_events"] == events
    assert "max_turns_reached" in episode.metadata["error"]["message"]


def test_harness_converts_request_timeout_to_resumable_episode(monkeypatch):
    def time_out(*args, **kwargs):
        raise requests.Timeout("late")

    monkeypatch.setattr("rllm.integrations.mcp_atlas.harness.requests.post", time_out)
    episode = _configured_harness().run(
        _task(),
        AgentConfig(base_url="http://gateway/v1", model="model", session_uid="s-timeout"),
    )

    assert episode.termination_reason == TerminationReason.TIMEOUT
    assert episode.artifacts["mcp_atlas"]["timed_out"] is True
    assert "1800" in episode.metadata["error"]["message"]


def test_prepare_derives_selected_servers_and_cleans_up(monkeypatch, tmp_path):
    managers = []

    class FakeManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.harness_url = "http://fake"
            self.started = False
            self.stopped = False
            managers.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def metadata(self):
            return {"required_servers": sorted(self.kwargs["required_servers"])}

    monkeypatch.setattr("rllm.integrations.mcp_atlas.harness.MCPAtlasServiceManager", FakeManager)
    harness = MCPAtlasHarness()
    context = SimpleNamespace(
        run_dir=tmp_path,
        tasks=(_task(),),
        model="model",
        concurrency=5,
        sampling_params={},
    )

    harness.prepare_eval(context)
    harness.finalize_eval(context)

    assert managers[0].started is True
    assert managers[0].stopped is True
    assert managers[0].kwargs["required_servers"] == {"filesystem", "git"}
    assert harness.eval_run_metadata()["mcp_atlas"]["required_servers"] == ["filesystem", "git"]


def test_smoke_mode_requires_explicit_subset():
    harness = MCPAtlasHarness()
    harness.preflight = "smoke"
    context = SimpleNamespace(run_dir=None, tasks=tuple(_task(str(idx)) for idx in range(500)))

    with pytest.raises(RuntimeError, match="explicitly selected"):
        harness.prepare_eval(context)


def test_default_server_filter_excludes_credentials_and_e2b():
    harness = MCPAtlasHarness()
    leftovers = harness.configure({"task_filter": "default_servers", "preflight": "smoke"})
    tasks = [
        _task("default"),
        Task(id="airtable", instruction="x", metadata={"ENABLED_TOOLS": ["airtable_list_bases"]}),
        Task(id="e2b", instruction="x", metadata={"ENABLED_TOOLS": ["e2b-server_run_code"]}),
        Task(id="mixed", instruction="x", metadata={"ENABLED_TOOLS": ["filesystem_read_file", "notion_API-get-users"]}),
    ]

    selected = harness.filter_eval_tasks(tasks)
    metadata = harness.eval_task_filter_metadata()

    assert leftovers == {}
    assert [task.id for task in selected] == ["default"]
    assert len(DEFAULT_SERVERS) == 20
    assert "e2b-server" not in DEFAULT_SERVERS
    assert metadata["name"] == "default_servers"
    assert metadata["source_task_count"] == 4
    assert metadata["selected_task_count"] == 1
    assert metadata["excluded_servers"] == ["airtable", "e2b-server", "notion"]


def test_no_credentials_filter_alias_and_invalid_filter():
    harness = MCPAtlasHarness()
    harness.configure({"task_filter": "no_credentials"})
    assert harness.task_filter == "default_servers"

    with pytest.raises(ValueError, match="task_filter"):
        MCPAtlasHarness().configure({"task_filter": "unknown"})
