"""Tests for evaluator loader: registry, import paths, entry-point discovery, and catalog resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rllm.eval.evaluator_loader import (
    _EVALUATOR_REGISTRY,
    list_evaluators,
    load_evaluator,
    register_evaluator,
    resolve_evaluator_from_catalog,
    unregister_evaluator,
)
from rllm.eval.types import EvalOutput, Evaluator
from rllm.types import Episode


def _has_evaluate(obj):
    return hasattr(obj, "evaluate") and callable(obj.evaluate)


class TestLoadEvaluator:
    """The built-in registry now points at score_fn import paths; the loader
    wraps bare functions in ``_FunctionEvaluator``. We assert the loaded
    object satisfies the Evaluator protocol rather than checking class type."""

    def test_load_math_evaluator_by_name(self):
        evaluator = load_evaluator("math_reward_fn")
        assert _has_evaluate(evaluator)

    def test_load_countdown_evaluator_by_name(self):
        evaluator = load_evaluator("countdown_reward_fn")
        assert _has_evaluate(evaluator)

    def test_load_code_evaluator_by_name(self):
        evaluator = load_evaluator("code_reward_fn")
        assert _has_evaluate(evaluator)

    def test_load_f1_evaluator_by_name(self):
        evaluator = load_evaluator("f1_reward_fn")
        assert _has_evaluate(evaluator)

    def test_load_by_import_path(self):
        evaluator = load_evaluator("rllm.eval.reward_fns.math:evaluate")
        assert _has_evaluate(evaluator)

    def test_load_unknown_name_raises(self):
        with pytest.raises(KeyError, match="not found"):
            load_evaluator("nonexistent_evaluator")

    def test_load_bad_import_path_raises(self):
        with pytest.raises(ImportError):
            load_evaluator("nonexistent.module:MyEvaluator")

    def test_all_registry_entries_are_evaluators(self):
        for name in _EVALUATOR_REGISTRY:
            evaluator = load_evaluator(name)
            assert _has_evaluate(evaluator), f"{name} is not an Evaluator"

    def test_legacy_dict_call_still_works(self):
        """Score_fns expect Task; loader wraps so dict-passing still works."""
        from rllm.types import Step, Trajectory

        evaluator = load_evaluator("math_reward_fn")
        episode = Episode(
            id="t",
            task="",
            trajectories=[
                Trajectory(
                    uid="t",
                    name="f",
                    task="",
                    steps=[Step(id="s0", input="", output=r"\boxed{4}")],
                    output=r"\boxed{4}",
                )
            ],
        )
        out = evaluator.evaluate({"ground_truth": "4"}, episode)
        assert out.reward == 1.0
        assert out.is_correct is True


class TestResolveEvaluatorFromCatalog:
    def test_resolve_gsm8k(self):
        evaluator = resolve_evaluator_from_catalog("gsm8k")
        assert _has_evaluate(evaluator)

    def test_resolve_countdown(self):
        evaluator = resolve_evaluator_from_catalog("countdown")
        assert _has_evaluate(evaluator)

    def test_resolve_humaneval(self):
        evaluator = resolve_evaluator_from_catalog("humaneval")
        assert _has_evaluate(evaluator)

    def test_resolve_hotpotqa(self):
        evaluator = resolve_evaluator_from_catalog("hotpotqa")
        assert _has_evaluate(evaluator)

    def test_resolve_unknown_returns_none(self):
        assert resolve_evaluator_from_catalog("unknown_benchmark") is None


class _DummyEvaluator:
    """A class that conforms to Evaluator protocol."""

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(reward=1.0, is_correct=True, signals=[])


class TestEntryPointDiscovery:
    def test_plugin_evaluator_discovery(self, monkeypatch):
        mock_ep = MagicMock()
        mock_ep.name = "my_plugin_eval"
        mock_ep.load.return_value = _DummyEvaluator

        def fake_entry_points(group):
            return [mock_ep] if group == "rllm.evaluators" else []

        monkeypatch.setattr("rllm.eval.evaluator_loader.entry_points", fake_entry_points)
        evaluator = load_evaluator("my_plugin_eval")
        assert isinstance(evaluator, _DummyEvaluator)

    def test_plugin_evaluator_instance(self, monkeypatch):
        instance = _DummyEvaluator()
        mock_ep = MagicMock()
        mock_ep.name = "my_instance_eval"
        mock_ep.load.return_value = instance

        def fake_entry_points(group):
            return [mock_ep] if group == "rllm.evaluators" else []

        monkeypatch.setattr("rllm.eval.evaluator_loader.entry_points", fake_entry_points)
        evaluator = load_evaluator("my_instance_eval")
        assert evaluator is instance

    def test_builtin_takes_priority_over_plugin(self, monkeypatch):
        """Built-in evaluators take priority over plugin entries with the same name."""
        mock_ep = MagicMock()
        mock_ep.name = "math_reward_fn"
        mock_ep.load.return_value = _DummyEvaluator

        def fake_entry_points(group):
            return [mock_ep] if group == "rllm.evaluators" else []

        monkeypatch.setattr("rllm.eval.evaluator_loader.entry_points", fake_entry_points)
        evaluator = load_evaluator("math_reward_fn")
        # Built-in resolves first → score_fn wrapper, not the plugin
        mock_ep.load.assert_not_called()
        assert _has_evaluate(evaluator)


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
        mock_ep = MagicMock()
        mock_ep.name = "my_plugin"
        mock_ep.value = "my_pkg.evaluator:MyEvaluator"
        mock_ep.dist = MagicMock()
        mock_ep.dist.name = "my-pkg"

        def fake_entry_points(group):
            return [mock_ep] if group == "rllm.evaluators" else []

        monkeypatch.setattr("rllm.eval.evaluator_loader.entry_points", fake_entry_points)
        evaluators = list_evaluators()
        plugin = [e for e in evaluators if e["name"] == "my_plugin"]
        assert len(plugin) == 1
        assert plugin[0]["source"] == "plugin (my-pkg)"


class TestRegisterEvaluator:
    """Tests for persistent evaluator registration (writes to ~/.rllm/evaluators.json)."""

    @pytest.fixture(autouse=True)
    def _isolate_registry(self, tmp_path, monkeypatch):
        evals_file = str(tmp_path / "evaluators.json")
        monkeypatch.setattr("rllm.eval.evaluator_loader._USER_EVALUATORS_FILE", evals_file)
        monkeypatch.setattr("rllm.eval.evaluator_loader._RLLM_HOME", str(tmp_path))

    def test_register_string_path_and_load(self):
        register_evaluator("test_eval", "rllm.eval.reward_fns.math:evaluate")
        evaluator = load_evaluator("test_eval")
        assert _has_evaluate(evaluator)

    def test_register_class(self):
        register_evaluator("test_eval", _DummyEvaluator)
        evaluator = load_evaluator("test_eval")
        assert isinstance(evaluator, _DummyEvaluator)

    def test_register_instance(self):
        register_evaluator("test_eval", _DummyEvaluator())
        evaluator = load_evaluator("test_eval")
        assert isinstance(evaluator, _DummyEvaluator)

    def test_persists_to_disk(self, tmp_path):
        register_evaluator("test_eval", "rllm.eval.reward_fns.math:evaluate")
        import json

        data = json.loads((tmp_path / "evaluators.json").read_text())
        assert "test_eval" in data

    def test_appears_in_list(self):
        register_evaluator("test_eval", "rllm.eval.reward_fns.math:evaluate")
        evaluators = list_evaluators()
        registered = [e for e in evaluators if e["name"] == "test_eval"]
        assert len(registered) == 1
        assert registered[0]["source"] == "registered"

    def test_unregister(self):
        register_evaluator("test_eval", "rllm.eval.reward_fns.math:evaluate")
        assert unregister_evaluator("test_eval") is True
        with pytest.raises(KeyError):
            load_evaluator("test_eval")

    def test_unregister_nonexistent(self):
        assert unregister_evaluator("nonexistent") is False

    def test_register_bad_class_raises(self):
        with pytest.raises(TypeError, match="must be a class with an .evaluate"):
            register_evaluator("test_eval", dict)


# Make `Evaluator` import a no-op consumer so the import line still validates
_ = Evaluator
