"""Tests for rllm eval CLI command."""

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from rllm.experimental.cli.main import cli
from rllm.experimental.eval.config import RllmConfig
from rllm.experimental.eval.types import AgentConfig, EvalOutput, Signal, Task
from rllm.types import Episode, Step, Trajectory


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
    """Set up a temporary RLLM_HOME directory."""
    rllm_home = str(tmp_path / ".rllm")
    monkeypatch.setenv("RLLM_HOME", rllm_home)
    from rllm.data.dataset import DatasetRegistry

    monkeypatch.setattr(DatasetRegistry, "_RLLM_HOME", rllm_home)
    monkeypatch.setattr(DatasetRegistry, "_REGISTRY_FILE", os.path.join(rllm_home, "datasets", "registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_DATASET_DIR", os.path.join(rllm_home, "datasets"))
    legacy_dir = str(tmp_path / "legacy_registry")
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_REGISTRY_DIR", legacy_dir)
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_REGISTRY_FILE", os.path.join(legacy_dir, "dataset_registry.json"))
    monkeypatch.setattr(DatasetRegistry, "_LEGACY_DATASET_DIR", os.path.join(legacy_dir, "datasets"))
    return rllm_home


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_dataset(tmp_rllm_home):
    """Register a small test dataset."""
    from rllm.data import DatasetRegistry

    data = [
        {"question": "What is 1+1?", "ground_truth": "2", "data_source": "test"},
        {"question": "What is 2+2?", "ground_truth": "4", "data_source": "test"},
        {"question": "What is 3+3?", "ground_truth": "6", "data_source": "test"},
    ]
    DatasetRegistry.register_dataset("test_math", data, split="test")
    return data


class _MockAgentFlow:
    """Mock AgentFlow that returns a fixed Episode."""

    def run(self, task: Task, config: AgentConfig) -> Episode:
        data = task.data if isinstance(task, Task) else task
        step = Step(input=data.get("question", ""), output="mock answer", done=True)
        return Episode(task=data, trajectories=[Trajectory(name="mock", steps=[step])], artifacts={"answer": "mock answer"})


class _MockEvaluator:
    """Mock evaluator that always returns correct."""

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(reward=1.0, is_correct=True, signals=[Signal(name="accuracy", value=1.0)])


def test_eval_missing_config(runner, tmp_rllm_home):
    """Eval without --base-url and no config should tell user to run 'rllm setup'."""
    with patch("rllm.experimental.eval.config.load_config", return_value=RllmConfig()):
        result = runner.invoke(cli, ["eval", "gsm8k"])
    assert result.exit_code != 0
    assert "rllm setup" in result.output


def test_eval_base_url_requires_model(runner, tmp_rllm_home):
    """Eval with --base-url but no --model should error."""
    result = runner.invoke(cli, ["eval", "gsm8k", "--base-url", "http://localhost:8000/v1"])
    assert result.exit_code != 0
    assert "--model is required" in result.output


def test_eval_with_proxy_mode(runner, tmp_rllm_home, mock_dataset):
    """Eval without --base-url should auto-start proxy from config."""
    config = RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-test"})
    mock_pm = MagicMock()
    mock_pm.get_proxy_url.return_value = "http://127.0.0.1:4000/v1"
    mock_pm.build_proxy_config.return_value = {"model_list": []}

    with patch("rllm.experimental.eval.config.load_config", return_value=config), patch("rllm.experimental.eval.proxy.EvalProxyManager", return_value=mock_pm), patch("rllm.experimental.cli.eval._run_eval"):
        result = runner.invoke(
            cli,
            [
                "eval",
                "test_math",
                "--agent",
                "math",
            ],
        )

    assert result.exit_code == 0
    mock_pm.start_proxy_subprocess.assert_called_once()
    mock_pm.shutdown_proxy.assert_called_once()


def test_eval_base_url_skips_proxy(runner, tmp_rllm_home, mock_dataset):
    """Eval with --base-url should not create a proxy."""
    mock_agent = _MockAgentFlow()

    with patch("rllm.experimental.eval.proxy.EvalProxyManager") as mock_pm_cls, patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=_MockEvaluator()):
        result = runner.invoke(
            cli,
            [
                "eval",
                "test_math",
                "--agent",
                "math",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "test-model",
            ],
        )

    assert result.exit_code == 0
    mock_pm_cls.assert_not_called()


def test_eval_with_mock_agent(runner, tmp_rllm_home, mock_dataset):
    """Eval with a mock agent should produce results."""
    mock_agent = _MockAgentFlow()

    with patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=_MockEvaluator()):
        result = runner.invoke(
            cli,
            [
                "eval",
                "test_math",
                "--agent",
                "math",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "test-model",
            ],
        )

    assert result.exit_code == 0
    assert "Accuracy" in result.output
    assert "100.0%" in result.output
    assert "3/3" in result.output


def test_eval_with_max_examples(runner, tmp_rllm_home, mock_dataset):
    """Eval with --max-examples should limit evaluation."""
    mock_agent = _MockAgentFlow()

    with patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=_MockEvaluator()):
        result = runner.invoke(
            cli,
            [
                "eval",
                "test_math",
                "--agent",
                "math",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "test-model",
                "--max-examples",
                "2",
            ],
        )

    assert result.exit_code == 0
    assert "2 examples" in result.output
    assert "2/2" in result.output


def test_eval_saves_results(runner, tmp_rllm_home, mock_dataset):
    """Eval should save results to a JSON file."""
    mock_agent = _MockAgentFlow()

    with patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=_MockEvaluator()):
        result = runner.invoke(
            cli,
            [
                "eval",
                "test_math",
                "--agent",
                "math",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "test-model",
            ],
        )

    assert result.exit_code == 0
    assert "Saved to" in result.output


def test_eval_with_explicit_evaluator(runner, tmp_rllm_home, mock_dataset):
    """Eval with --evaluator should use specified evaluator."""
    mock_agent = _MockAgentFlow()

    with patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.load_evaluator", return_value=_MockEvaluator()) as mock_load_eval:
        result = runner.invoke(
            cli,
            [
                "eval",
                "test_math",
                "--agent",
                "math",
                "--evaluator",
                "math_reward_fn",
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "test-model",
            ],
        )

    assert result.exit_code == 0
    mock_load_eval.assert_called_once_with("math_reward_fn")
