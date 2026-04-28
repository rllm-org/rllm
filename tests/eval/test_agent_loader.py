"""Tests for agent loader: registry, import paths, auto-instantiation, entry-point and persistent discovery."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rllm.eval.agent_loader import (
    list_agents,
    load_agent,
    register_agent,
    unregister_agent,
)


class TestLoadAgent:
    def test_load_builtin_react_harness(self):
        agent = load_agent("react")
        assert hasattr(agent, "run") and callable(agent.run)

    def test_load_by_import_path_class_auto_instantiates(self):
        """Loading a module:ClassName path auto-instantiates the class."""
        agent = load_agent("rllm.harnesses.react:ReActHarness")
        from rllm.harnesses.react import ReActHarness

        assert isinstance(agent, ReActHarness)
        assert not isinstance(agent, type)

    def test_load_bad_import_path_raises(self):
        with pytest.raises(ImportError):
            load_agent("nonexistent.module:my_agent")

    def test_load_missing_attr_raises(self):
        with pytest.raises(AttributeError):
            load_agent("rllm.eval.types:nonexistent_attr")

    def test_load_object_without_run_raises(self):
        with pytest.raises(TypeError, match="must be an AgentFlow"):
            load_agent("rllm.eval.evaluator_loader:_EVALUATOR_REGISTRY")

    def test_load_unknown_name_raises(self):
        with pytest.raises(KeyError, match="not found"):
            load_agent("nonexistent_agent_xyz")

    def test_entry_point_discovery(self, monkeypatch):
        """Plugin agents are discoverable via entry points."""
        from rllm.harnesses.react import ReActHarness

        plugin_agent = ReActHarness()

        mock_ep = MagicMock()
        mock_ep.name = "my_plugin_agent"
        mock_ep.load.return_value = plugin_agent

        def fake_entry_points(group):
            if group == "rllm.agents":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "rllm.eval.agent_loader.entry_points",
            fake_entry_points,
        )

        agent = load_agent("my_plugin_agent")
        assert agent is plugin_agent
        mock_ep.load.assert_called_once()

    def test_entry_point_class_auto_instantiates(self, monkeypatch):
        """Plugin agents that are classes get auto-instantiated."""
        from rllm.harnesses.react import ReActHarness

        mock_ep = MagicMock()
        mock_ep.name = "my_class_plugin"
        mock_ep.load.return_value = ReActHarness

        def fake_entry_points(group):
            if group == "rllm.agents":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "rllm.eval.agent_loader.entry_points",
            fake_entry_points,
        )

        agent = load_agent("my_class_plugin")
        assert isinstance(agent, ReActHarness)


class TestListAgents:
    def test_includes_plugin_agents(self, monkeypatch):
        """Plugin agents appear in the list."""
        mock_ep = MagicMock()
        mock_ep.name = "my_plugin"
        mock_ep.value = "my_pkg.agent:my_agent"
        mock_ep.dist = MagicMock()
        mock_ep.dist.name = "my-pkg"

        def fake_entry_points(group):
            if group == "rllm.agents":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "rllm.eval.agent_loader.entry_points",
            fake_entry_points,
        )

        agents = list_agents()
        plugin = [a for a in agents if a["name"] == "my_plugin"]
        assert len(plugin) == 1
        assert plugin[0]["source"] == "plugin (my-pkg)"


class TestRegisterAgent:
    """Tests for persistent agent registration (writes to ~/.rllm/agents.json)."""

    @pytest.fixture(autouse=True)
    def _isolate_registry(self, tmp_path, monkeypatch):
        """Point the agent registry at a temp directory."""
        agents_file = str(tmp_path / "agents.json")
        monkeypatch.setattr("rllm.eval.agent_loader._USER_AGENTS_FILE", agents_file)
        monkeypatch.setattr("rllm.eval.agent_loader._RLLM_HOME", str(tmp_path))

    def test_register_string_path_and_load(self):
        register_agent("test_agent", "rllm.harnesses.react:ReActHarness")
        agent = load_agent("test_agent")
        from rllm.harnesses.react import ReActHarness

        assert isinstance(agent, ReActHarness)

    def test_register_class(self):
        from rllm.harnesses.react import ReActHarness

        register_agent("test_agent", ReActHarness)
        agent = load_agent("test_agent")
        assert isinstance(agent, ReActHarness)

    def test_register_instance(self):
        from rllm.harnesses.react import ReActHarness

        register_agent("test_agent", ReActHarness())
        agent = load_agent("test_agent")
        assert isinstance(agent, ReActHarness)

    def test_persists_to_disk(self, tmp_path):
        register_agent("test_agent", "rllm.harnesses.react:ReActHarness")
        import json

        data = json.loads((tmp_path / "agents.json").read_text())
        assert "test_agent" in data
        assert data["test_agent"]["import_path"] == "rllm.harnesses.react:ReActHarness"

    def test_appears_in_list(self):
        register_agent("test_agent", "rllm.harnesses.react:ReActHarness")
        agents = list_agents()
        registered = [a for a in agents if a["name"] == "test_agent"]
        assert len(registered) == 1
        assert registered[0]["source"] == "registered"

    def test_unregister(self):
        register_agent("test_agent", "rllm.harnesses.react:ReActHarness")
        assert unregister_agent("test_agent") is True
        with pytest.raises(KeyError):
            load_agent("test_agent")

    def test_unregister_nonexistent(self):
        assert unregister_agent("nonexistent") is False

    def test_register_bad_class_raises(self):
        with pytest.raises(TypeError, match="must be an AgentFlow"):
            register_agent("test_agent", dict)
