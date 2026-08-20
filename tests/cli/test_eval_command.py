"""Tests for rllm eval CLI command."""

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from rllm.cli.eval import _apply_agent_task_filter, _load_agent_config, _redact_config, _sanitize_endpoint
from rllm.cli.main import cli
from rllm.eval.config import RllmConfig, load_config, save_config
from rllm.eval.types import EvalOutput, Signal
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory


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
        data = task.metadata if isinstance(task, Task) else task
        step = Step(input=data.get("question", ""), output="mock answer", done=True)
        return Episode(task=data, trajectories=[Trajectory(name="mock", steps=[step])], artifacts={"answer": "mock answer"})


class _MockEvaluator:
    """Mock evaluator that always returns correct."""

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        return EvalOutput(reward=1.0, is_correct=True, signals=[Signal(name="accuracy", value=1.0)])


def _invoke_rejected_eval(runner: CliRunner, args: list[str]):
    """Invoke an invalid eval command and assert it never starts evaluation."""
    with (
        patch("rllm.eval.proxy.EvalProxyManager") as mock_pm_cls,
        patch("rllm.cli.eval._run_eval") as mock_run,
    ):
        result = runner.invoke(cli, ["eval", "test_math", *args])

    assert result.exit_code != 0
    mock_pm_cls.assert_not_called()
    mock_run.assert_not_called()
    return result


def test_eval_missing_config(runner, tmp_rllm_home):
    """Eval without --base-url and no config should tell user to run 'rllm setup'."""
    with patch("rllm.eval.config.load_config", return_value=RllmConfig()):
        result = runner.invoke(cli, ["eval", "gsm8k"])
    assert result.exit_code != 0
    assert "rllm setup" in result.output


def test_eval_base_url_requires_model(runner, tmp_rllm_home):
    """Eval with --base-url but no --model should error."""
    result = runner.invoke(cli, ["eval", "gsm8k", "--base-url", "http://localhost:8000/v1"])
    assert result.exit_code != 0
    assert "--model is required" in result.output


def test_agent_config_loading_redaction_and_endpoint_sanitization(tmp_path):
    config_path = tmp_path / "agent.json"
    config_path.write_text('{"preflight":"strict","nested":{"api_key":"secret"}}')

    config = _load_agent_config(f"@{config_path}")

    assert config["preflight"] == "strict"
    assert _redact_config(config)["nested"]["api_key"] == "<redacted>"
    assert _sanitize_endpoint("https://user:pass@example.test/v1?token=secret") == "https://example.test/v1"


def test_optional_agent_task_filter_preserves_dataset_identity():
    from rllm.data.dataset import Dataset

    class FilterAgent:
        def filter_eval_tasks(self, tasks):
            return tasks[1:]

        def eval_task_filter_metadata(self):
            return {"name": "test", "selected_task_count": 2}

    source = Dataset(data=[1, 2, 3], name="bench", split="public")
    filtered, metadata = _apply_agent_task_filter(source, FilterAgent())

    assert filtered.data == [2, 3]
    assert filtered.name == "bench"
    assert filtered.split == "public"
    assert metadata == {"name": "test", "selected_task_count": 2}


def test_eval_with_proxy_mode(runner, tmp_rllm_home, mock_dataset):
    """Eval without --base-url should auto-start proxy from config."""
    config = RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-test"})
    mock_pm = MagicMock()
    mock_pm.get_proxy_url.return_value = "http://127.0.0.1:4000/v1"
    mock_pm.build_proxy_config.return_value = {"model_list": []}

    with (
        patch("rllm.eval.config.load_config", return_value=config),
        patch("rllm.eval.proxy.EvalProxyManager", return_value=mock_pm),
        patch("rllm.cli.eval._run_eval"),
    ):
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


def test_eval_provider_override_uses_stored_key_for_run(runner, tmp_rllm_home):
    """--provider should route through that provider without changing config."""
    original = RllmConfig(
        provider="openai",
        model="gpt-5-mini",
        api_keys={"openai": "sk-openai", "fireworks": "fw-test"},
    )
    save_config(original)

    mock_pm = MagicMock()
    mock_pm.get_proxy_url.return_value = "http://127.0.0.1:4000/v1"
    mock_pm.build_proxy_config.return_value = {"model_list": []}
    model = "accounts/fireworks/models/kimi-k3"

    with (
        patch("rllm.eval.proxy.EvalProxyManager", return_value=mock_pm) as mock_pm_cls,
        patch("rllm.cli.eval._run_eval"),
    ):
        result = runner.invoke(
            cli,
            [
                "eval",
                "test_math",
                "--agent",
                "math",
                "--provider",
                "fireworks",
                "--model",
                model,
            ],
        )

    assert result.exit_code == 0, result.output
    kwargs = mock_pm_cls.call_args.kwargs
    assert kwargs["provider"] == "fireworks"
    assert kwargs["model_name"] == model
    assert kwargs["api_key"] == "fw-test"
    mock_pm.start_proxy_subprocess.assert_called_once()
    mock_pm.shutdown_proxy.assert_called_once()
    assert load_config() == original


def test_eval_provider_override_requires_model(runner, tmp_rllm_home):
    """A provider override must not reuse the configured model implicitly."""
    save_config(
        RllmConfig(
            provider="openai",
            model="gpt-5-mini",
            api_keys={"openai": "sk-openai", "fireworks": "fw-test"},
        )
    )

    result = _invoke_rejected_eval(runner, ["--provider", "fireworks"])

    assert "--model is required when --provider is provided" in result.output


def test_eval_provider_override_requires_stored_key(runner, tmp_rllm_home):
    """A provider override should fail before launch when its key is missing."""
    save_config(RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-openai"}))

    result = _invoke_rejected_eval(
        runner,
        ["--provider", "fireworks", "--model", "accounts/fireworks/models/kimi-k3"],
    )

    assert "No configuration found" in result.output


def test_eval_provider_cannot_be_used_with_base_url(runner, tmp_rllm_home):
    """Direct endpoints already define routing, so --provider is ambiguous."""
    result = _invoke_rejected_eval(
        runner,
        [
            "--provider",
            "fireworks",
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "test-model",
        ],
    )

    assert "--provider cannot be used with --base-url" in result.output


def test_eval_base_url_skips_proxy(runner, tmp_rllm_home, mock_dataset):
    """Eval with --base-url should not create a proxy."""
    mock_agent = _MockAgentFlow()

    with (
        patch("rllm.eval.proxy.EvalProxyManager") as mock_pm_cls,
        patch("rllm.eval.agent_loader.load_agent", return_value=mock_agent),
        patch("rllm.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=_MockEvaluator()),
    ):
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

    with (
        patch("rllm.eval.agent_loader.load_agent", return_value=mock_agent),
        patch("rllm.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=_MockEvaluator()),
    ):
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

    with (
        patch("rllm.eval.agent_loader.load_agent", return_value=mock_agent),
        patch("rllm.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=_MockEvaluator()),
    ):
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

    with (
        patch("rllm.eval.agent_loader.load_agent", return_value=mock_agent),
        patch("rllm.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=_MockEvaluator()),
    ):
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

    with (
        patch("rllm.eval.agent_loader.load_agent", return_value=mock_agent),
        patch("rllm.eval.evaluator_loader.load_evaluator", return_value=_MockEvaluator()) as mock_load_eval,
    ):
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


_TINKER_CKPT = "tinker://b244fa1e-941c-5d3f-b6b9-ad7bd4ce136d:train:0/sampler_weights/final"


def test_eval_tinker_checkpoint_routes_to_tinker_provider(runner, tmp_rllm_home, mock_dataset, monkeypatch):
    """A ``tinker://`` checkpoint model routes through the tinker provider even
    when the local config selects a different provider."""
    monkeypatch.setenv("TINKER_API_KEY", "tml-test-key")
    # Config points at an unrelated provider — must be overridden by the model.
    config = RllmConfig(provider="openai", model="gpt-5-mini", api_keys={"openai": "sk-test"})

    mock_pm = MagicMock()
    mock_pm.get_proxy_url.return_value = "http://127.0.0.1:4000/v1"
    mock_pm.build_proxy_config.return_value = {"model_list": []}

    with (
        patch("rllm.eval.config.load_config", return_value=config),
        patch("rllm.eval.proxy.EvalProxyManager", return_value=mock_pm) as mock_pm_cls,
        patch("rllm.cli.eval._run_eval"),
    ):
        result = runner.invoke(cli, ["eval", "test_math", "--agent", "math", "--model", _TINKER_CKPT])

    assert result.exit_code == 0, result.output
    assert "tinker://" in result.output  # informative routing line
    mock_pm_cls.assert_called_once()
    kwargs = mock_pm_cls.call_args.kwargs
    assert kwargs["provider"] == "tinker"
    assert kwargs["model_name"] == _TINKER_CKPT
    assert kwargs["api_key"] == "tml-test-key"


def test_eval_tinker_checkpoint_without_config(runner, tmp_rllm_home, mock_dataset, monkeypatch):
    """A ``tinker://`` checkpoint works even when no provider config exists."""
    monkeypatch.setenv("TINKER_API_KEY", "tml-test-key")

    mock_pm = MagicMock()
    mock_pm.get_proxy_url.return_value = "http://127.0.0.1:4000/v1"
    mock_pm.build_proxy_config.return_value = {"model_list": []}

    with (
        patch("rllm.eval.config.load_config", return_value=RllmConfig()),
        patch("rllm.eval.proxy.EvalProxyManager", return_value=mock_pm) as mock_pm_cls,
        patch("rllm.cli.eval._run_eval"),
    ):
        result = runner.invoke(cli, ["eval", "test_math", "--agent", "math", "--model", _TINKER_CKPT])

    assert result.exit_code == 0, result.output
    assert "rllm setup" not in result.output  # the is_configured() gate is bypassed
    mock_pm_cls.assert_called_once()
    assert mock_pm_cls.call_args.kwargs["provider"] == "tinker"


def test_eval_tinker_checkpoint_requires_api_key(runner, tmp_rllm_home, monkeypatch):
    """A ``tinker://`` checkpoint without TINKER_API_KEY fails with an actionable error."""
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    with (
        patch("rllm.eval.config.load_config", return_value=RllmConfig()),
        patch("rllm.eval.proxy.EvalProxyManager") as mock_pm_cls,
        patch("rllm.cli.eval._run_eval") as mock_run,
    ):
        result = runner.invoke(cli, ["eval", "test_math", "--agent", "math", "--model", _TINKER_CKPT])

    assert result.exit_code != 0
    assert "TINKER_API_KEY" in result.output
    mock_pm_cls.assert_not_called()
    mock_run.assert_not_called()
