"""Tests for evaluator loader: registry, import paths, entry-point discovery, and catalog resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rllm.experimental.eval.evaluator_loader import (
    _EVALUATOR_REGISTRY,
    list_evaluators,
    load_evaluator,
    register_evaluator,
    resolve_evaluator_from_catalog,
    unregister_evaluator,
)
from rllm.experimental.eval.types import (
    CodeEvaluator,
    CountdownEvaluator,
    EvalOutput,
    Evaluator,
    F1Evaluator,
    MathEvaluator,
)
from rllm.types import Episode


class TestLoadEvaluator:
    def test_load_math_evaluator_by_name(self):
        evaluator = load_evaluator("math_reward_fn")
        assert isinstance(evaluator, MathEvaluator)

    def test_load_countdown_evaluator_by_name(self):
        evaluator = load_evaluator("countdown_reward_fn")
        assert isinstance(evaluator, CountdownEvaluator)

    def test_load_code_evaluator_by_name(self):
        evaluator = load_evaluator("code_reward_fn")
        assert isinstance(evaluator, CodeEvaluator)

    def test_load_f1_evaluator_by_name(self):
        evaluator = load_evaluator("f1_reward_fn")
        assert isinstance(evaluator, F1Evaluator)

    def test_load_by_import_path(self):
        evaluator = load_evaluator("rllm.experimental.eval.types:MathEvaluator")
        assert isinstance(evaluator, MathEvaluator)

    def test_load_unknown_name_raises(self):
        with pytest.raises(KeyError, match="not found"):
            load_evaluator("nonexistent_evaluator")

    def test_load_bad_import_path_raises(self):
        with pytest.raises(ImportError):
            load_evaluator("nonexistent.module:MyEvaluator")

    def test_load_object_without_evaluate_raises(self):
        # EvalOutput is a dataclass with required args, so instantiation fails with TypeError
        with pytest.raises(TypeError):
            load_evaluator("rllm.experimental.eval.types:EvalOutput")

    def test_all_registry_entries_are_evaluators(self):
        for name in _EVALUATOR_REGISTRY:
            evaluator = load_evaluator(name)
            assert isinstance(evaluator, Evaluator), f"{name} is not an Evaluator"


class TestResolveEvaluatorFromCatalog:
    def test_resolve_gsm8k(self):
        evaluator = resolve_evaluator_from_catalog("gsm8k")
        assert isinstance(evaluator, MathEvaluator)

    def test_resolve_countdown(self):
        evaluator = resolve_evaluator_from_catalog("countdown")
        assert isinstance(evaluator, CountdownEvaluator)

    def test_resolve_humaneval(self):
        evaluator = resolve_evaluator_from_catalog("humaneval")
        assert isinstance(evaluator, CodeEvaluator)

    def test_resolve_hotpotqa(self):
        evaluator = resolve_evaluator_from_catalog("hotpotqa")
        assert isinstance(evaluator, F1Evaluator)

    def test_resolve_unknown_returns_none(self):
        result = resolve_evaluator_from_catalog("unknown_benchmark")
        assert result is None


# --- Helpers for entry-point tests ---


class _DummyEvaluator:
    """A class that conforms to Evaluator protocol."""

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(reward=1.0, is_correct=True, signals=[])


class TestEntryPointDiscovery:
    def test_plugin_evaluator_discovery(self, monkeypatch):
        """Plugin evaluators are discoverable via entry points."""
        mock_ep = MagicMock()
        mock_ep.name = "my_plugin_eval"
        mock_ep.load.return_value = _DummyEvaluator

        def fake_entry_points(group):
            if group == "rllm.evaluators":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "rllm.experimental.eval.evaluator_loader.entry_points",
            fake_entry_points,
        )

        evaluator = load_evaluator("my_plugin_eval")
        assert isinstance(evaluator, _DummyEvaluator)
        mock_ep.load.assert_called_once()

    def test_plugin_evaluator_instance(self, monkeypatch):
        """Plugin evaluators that are instances (not classes) work directly."""
        instance = _DummyEvaluator()
        mock_ep = MagicMock()
        mock_ep.name = "my_instance_eval"
        mock_ep.load.return_value = instance

        def fake_entry_points(group):
            if group == "rllm.evaluators":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "rllm.experimental.eval.evaluator_loader.entry_points",
            fake_entry_points,
        )

        evaluator = load_evaluator("my_instance_eval")
        assert evaluator is instance

    def test_builtin_takes_priority_over_plugin(self, monkeypatch):
        """Built-in evaluators take priority over agenthub entries with the same name."""
        mock_ep = MagicMock()
        mock_ep.name = "math_reward_fn"
        mock_ep.load.return_value = _DummyEvaluator

        def fake_entry_points(group):
            if group == "rllm.evaluators":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "rllm.experimental.eval.evaluator_loader.entry_points",
            fake_entry_points,
        )

        evaluator = load_evaluator("math_reward_fn")
        assert isinstance(evaluator, MathEvaluator)
        mock_ep.load.assert_not_called()


class TestListEvaluators:
    def test_includes_builtin_evaluators(self):
        evaluators = list_evaluators()
        names = {e["name"] for e in evaluators}
        assert "math_reward_fn" in names
        assert "code_reward_fn" in names

    def test_builtin_source(self):
        evaluators = list_evaluators()
        for e in evaluators:
            if e["name"] == "math_reward_fn":
                assert e["source"] == "built-in"
                break

    def test_includes_plugin_evaluators(self, monkeypatch):
        """Plugin evaluators appear in the list."""
        mock_ep = MagicMock()
        mock_ep.name = "my_plugin"
        mock_ep.value = "my_pkg.evaluator:MyEvaluator"
        mock_ep.dist = MagicMock()
        mock_ep.dist.name = "my-pkg"

        def fake_entry_points(group):
            if group == "rllm.evaluators":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "rllm.experimental.eval.evaluator_loader.entry_points",
            fake_entry_points,
        )

        evaluators = list_evaluators()
        plugin = [e for e in evaluators if e["name"] == "my_plugin"]
        assert len(plugin) == 1
        assert plugin[0]["source"] == "plugin (my-pkg)"


class TestRegisterEvaluator:
    """Tests for persistent evaluator registration (writes to ~/.rllm/evaluators.json)."""

    @pytest.fixture(autouse=True)
    def _isolate_registry(self, tmp_path, monkeypatch):
        """Point the evaluator registry at a temp directory."""
        evals_file = str(tmp_path / "evaluators.json")
        monkeypatch.setattr("rllm.experimental.eval.evaluator_loader._USER_EVALUATORS_FILE", evals_file)
        monkeypatch.setattr("rllm.experimental.eval.evaluator_loader._RLLM_HOME", str(tmp_path))

    def test_register_string_path_and_load(self):
        register_evaluator("test_eval", "rllm.experimental.eval.types:MathEvaluator")
        evaluator = load_evaluator("test_eval")
        assert isinstance(evaluator, MathEvaluator)

    def test_register_class(self):
        register_evaluator("test_eval", MathEvaluator)
        evaluator = load_evaluator("test_eval")
        assert isinstance(evaluator, MathEvaluator)

    def test_register_instance(self):
        register_evaluator("test_eval", MathEvaluator())
        evaluator = load_evaluator("test_eval")
        assert isinstance(evaluator, MathEvaluator)

    def test_persists_to_disk(self, tmp_path):
        register_evaluator("test_eval", "rllm.experimental.eval.types:MathEvaluator")
        import json

        data = json.loads((tmp_path / "evaluators.json").read_text())
        assert "test_eval" in data

    def test_appears_in_list(self):
        register_evaluator("test_eval", "rllm.experimental.eval.types:MathEvaluator")
        evaluators = list_evaluators()
        registered = [e for e in evaluators if e["name"] == "test_eval"]
        assert len(registered) == 1
        assert registered[0]["source"] == "registered"

    def test_unregister(self):
        register_evaluator("test_eval", "rllm.experimental.eval.types:MathEvaluator")
        assert unregister_evaluator("test_eval") is True
        with pytest.raises(KeyError):
            load_evaluator("test_eval")

    def test_unregister_nonexistent(self):
        assert unregister_evaluator("nonexistent") is False

    def test_register_bad_class_raises(self):
        with pytest.raises(TypeError, match="must be a class with an .evaluate"):
            register_evaluator("test_eval", dict)
