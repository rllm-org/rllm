"""Tests for rllm train CLI command."""

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from rllm.experimental.cli.main import cli
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
def mock_train_dataset(tmp_rllm_home):
    """Register a small test dataset with train and test splits."""
    from rllm.data import DatasetRegistry

    train_data = [
        {"question": "What is 1+1?", "ground_truth": "2", "data_source": "test"},
        {"question": "What is 2+2?", "ground_truth": "4", "data_source": "test"},
        {"question": "What is 3+3?", "ground_truth": "6", "data_source": "test"},
        {"question": "What is 4+4?", "ground_truth": "8", "data_source": "test"},
    ]
    test_data = [
        {"question": "What is 5+5?", "ground_truth": "10", "data_source": "test"},
        {"question": "What is 6+6?", "ground_truth": "12", "data_source": "test"},
    ]
    DatasetRegistry.register_dataset("test_math", train_data, split="train")
    DatasetRegistry.register_dataset("test_math", test_data, split="test")
    return train_data, test_data


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


class TestBuildTrainConfig:
    """Tests for build_train_config()."""

    def test_produces_valid_dictconfig(self):
        """Config should be an OmegaConf DictConfig with expected keys."""
        from rllm.experimental.cli.train import build_train_config

        cfg = build_train_config(
            model_name="Qwen/Qwen3-8B",
            group_size=4,
            batch_size=16,
            lr=1e-5,
            lora_rank=16,
            total_epochs=2,
            total_steps=None,
            val_freq=10,
            save_freq=50,
            project="test-project",
            experiment="test-exp",
            output_dir=None,
            config_file=None,
        )

        from omegaconf import DictConfig

        assert isinstance(cfg, DictConfig)

        # Check model config
        assert cfg.model.name == "Qwen/Qwen3-8B"
        assert cfg.model.lora_rank == 16

        # Check model_name is set in rllm namespace (used by SdkWorkflowFactory proxy)
        assert cfg.rllm.model_name == "Qwen/Qwen3-8B"

        # Check training config
        assert cfg.training.group_size == 4
        assert cfg.training.learning_rate == 1e-5

        # Check rllm trainer config
        assert cfg.rllm.trainer.total_epochs == 2
        assert cfg.rllm.trainer.test_freq == 10
        assert cfg.rllm.trainer.save_freq == 50
        assert cfg.rllm.trainer.project_name == "test-project"
        assert cfg.rllm.trainer.experiment_name == "test-exp"

        # Check data config exists
        assert hasattr(cfg, "data")
        assert cfg.data.train_batch_size == 16

    def test_total_steps_overrides_epochs(self):
        """--max-steps should set total_batches and force epochs=1."""
        from rllm.experimental.cli.train import build_train_config

        cfg = build_train_config(
            model_name="Qwen/Qwen3-8B",
            group_size=8,
            batch_size=32,
            lr=2e-5,
            lora_rank=32,
            total_epochs=10,
            total_steps=100,
            val_freq=5,
            save_freq=20,
            project="test",
            experiment="test",
            output_dir=None,
            config_file=None,
        )

        assert cfg.rllm.trainer.total_batches == 100
        assert cfg.rllm.trainer.total_epochs == 1

    def test_output_dir_override(self):
        """--output should set training.default_local_dir."""
        from rllm.experimental.cli.train import build_train_config

        cfg = build_train_config(
            model_name="Qwen/Qwen3-8B",
            group_size=8,
            batch_size=32,
            lr=2e-5,
            lora_rank=32,
            total_epochs=1,
            total_steps=None,
            val_freq=5,
            save_freq=20,
            project="test",
            experiment="test",
            output_dir="/tmp/my-checkpoints",
            config_file=None,
        )

        assert cfg.training.default_local_dir == "/tmp/my-checkpoints"

    def test_config_file_merge(self, tmp_path):
        """--config file should be merged and overridable by CLI flags."""
        from rllm.experimental.cli.train import build_train_config

        config_file = tmp_path / "custom.yaml"
        config_file.write_text("model:\n  name: custom-model\ntraining:\n  learning_rate: 1e-4\n")

        cfg = build_train_config(
            model_name="Qwen/Qwen3-8B",  # CLI override should win
            group_size=8,
            batch_size=32,
            lr=2e-5,  # CLI override should win over config file's 1e-4
            lora_rank=32,
            total_epochs=1,
            total_steps=None,
            val_freq=5,
            save_freq=20,
            project="test",
            experiment="test",
            output_dir=None,
            config_file=str(config_file),
        )

        # CLI flags should win over config file
        assert cfg.model.name == "Qwen/Qwen3-8B"
        assert cfg.training.learning_rate == 2e-5

    def test_workflow_enabled(self):
        """Config should enable workflow mode."""
        from rllm.experimental.cli.train import build_train_config

        cfg = build_train_config(
            model_name="Qwen/Qwen3-8B",
            group_size=8,
            batch_size=32,
            lr=2e-5,
            lora_rank=32,
            total_epochs=1,
            total_steps=None,
            val_freq=5,
            save_freq=20,
            project="test",
            experiment="test",
            output_dir=None,
            config_file=None,
        )

        assert cfg.rllm.workflow.use_workflow is True


class TestTrainCommand:
    """Tests for the train CLI command."""

    def test_train_help(self, runner):
        """rllm train --help should show all options."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train a model" in result.output
        assert "--model" in result.output
        assert "--group-size" in result.output
        assert "--batch-size" in result.output
        assert "--lr" in result.output
        assert "--lora-rank" in result.output
        assert "--epochs" in result.output
        assert "--max-steps" in result.output
        assert "--val-freq" in result.output
        assert "--save-freq" in result.output
        assert "--train-dataset" in result.output
        assert "--val-dataset" in result.output
        assert "--agent" in result.output
        assert "--evaluator" in result.output
        assert "--config" in result.output
        assert "--ui" in result.output
        assert "--ui-url" in result.output

    def test_train_listed_in_main_help(self, runner):
        """rllm --help should list the train command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output

    def test_train_no_agent_no_catalog(self, runner, tmp_rllm_home, mock_train_dataset):
        """Train without --agent and no catalog default should error."""
        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value={"datasets": {}}):
            result = runner.invoke(cli, ["train", "unknown_benchmark"])
        assert result.exit_code != 0
        assert "No --agent specified" in result.output

    def test_train_agent_resolution_from_catalog(self, runner, tmp_rllm_home, mock_train_dataset):
        """Train should resolve agent from catalog when not specified."""
        catalog = {"datasets": {"test_math": {"default_agent": "math", "reward_fn": "math_reward_fn", "eval_split": "test"}}}
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()
        mock_trainer = MagicMock()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent) as mock_load_agent, patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=mock_evaluator), patch("rllm.experimental.unified_trainer.AgentTrainer", return_value=mock_trainer):
            result = runner.invoke(cli, ["train", "test_math", "--model", "test-model"])

        assert result.exit_code == 0
        mock_load_agent.assert_called_once_with("math")
        mock_trainer.train.assert_called_once()

    def test_train_explicit_agent_and_evaluator(self, runner, tmp_rllm_home, mock_train_dataset):
        """Train with explicit --agent and --evaluator should use them."""
        catalog = {"datasets": {"test_math": {"eval_split": "test"}}}
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()
        mock_trainer = MagicMock()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent) as mock_load_agent, patch("rllm.experimental.eval.evaluator_loader.load_evaluator", return_value=mock_evaluator) as mock_load_eval, patch("rllm.experimental.unified_trainer.AgentTrainer", return_value=mock_trainer):
            result = runner.invoke(
                cli,
                [
                    "train",
                    "test_math",
                    "--agent",
                    "custom_agent",
                    "--evaluator",
                    "custom_evaluator",
                    "--model",
                    "test-model",
                ],
            )

        assert result.exit_code == 0
        mock_load_agent.assert_called_once_with("custom_agent")
        mock_load_eval.assert_called_once_with("custom_evaluator")

    def test_train_header_display(self, runner, tmp_rllm_home, mock_train_dataset):
        """Train should display a header panel with configuration info."""
        catalog = {"datasets": {"test_math": {"default_agent": "math", "reward_fn": "math_reward_fn", "eval_split": "test"}}}
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()
        mock_trainer = MagicMock()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=mock_evaluator), patch("rllm.experimental.unified_trainer.AgentTrainer", return_value=mock_trainer):
            result = runner.invoke(
                cli,
                [
                    "train",
                    "test_math",
                    "--model",
                    "my-model",
                    "--group-size",
                    "4",
                    "--batch-size",
                    "16",
                ],
            )

        assert result.exit_code == 0
        assert "rLLM Train" in result.output
        assert "my-model" in result.output
        assert "test_math" in result.output

    def test_train_with_max_examples(self, runner, tmp_rllm_home, mock_train_dataset):
        """Train with --max-examples should limit training data."""
        catalog = {"datasets": {"test_math": {"default_agent": "math", "reward_fn": "math_reward_fn", "eval_split": "test"}}}
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()
        mock_trainer = MagicMock()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=mock_evaluator), patch("rllm.experimental.unified_trainer.AgentTrainer", return_value=mock_trainer):
            result = runner.invoke(
                cli,
                [
                    "train",
                    "test_math",
                    "--model",
                    "test-model",
                    "--max-examples",
                    "2",
                ],
            )

        assert result.exit_code == 0
        # Header should show 2 examples
        assert "2 examples" in result.output

    def test_train_passes_correct_config_to_trainer(self, runner, tmp_rllm_home, mock_train_dataset):
        """Train should construct AgentTrainer with correct parameters."""
        catalog = {"datasets": {"test_math": {"default_agent": "math", "reward_fn": "math_reward_fn", "eval_split": "test"}}}
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()
        mock_trainer = MagicMock()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=mock_evaluator), patch("rllm.experimental.unified_trainer.AgentTrainer", return_value=mock_trainer) as mock_at_cls:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "test_math",
                    "--model",
                    "test-model",
                    "--group-size",
                    "4",
                    "--lr",
                    "1e-4",
                ],
            )

        assert result.exit_code == 0
        # Verify AgentTrainer was called with correct kwargs
        call_kwargs = mock_at_cls.call_args[1]
        assert call_kwargs["backend"] == "tinker"
        assert call_kwargs["agent_flow"] is not None
        assert call_kwargs["evaluator"] is not None
        assert call_kwargs["train_dataset"] is not None
        assert call_kwargs["config"].model.name == "test-model"
        assert call_kwargs["config"].training.group_size == 4
        assert call_kwargs["config"].training.learning_rate == 1e-4

    def test_train_separate_val_dataset(self, runner, tmp_rllm_home):
        """Train with --val-dataset should use a different validation dataset."""
        from rllm.data import DatasetRegistry

        train_data = [{"question": "q1", "ground_truth": "a1", "data_source": "test"}]
        val_data = [{"question": "q2", "ground_truth": "a2", "data_source": "test"}]
        DatasetRegistry.register_dataset("train_bench", train_data, split="train")
        DatasetRegistry.register_dataset("val_bench", val_data, split="test")

        catalog = {
            "datasets": {
                "train_bench": {"default_agent": "math", "reward_fn": "math_reward_fn", "eval_split": "test"},
                "val_bench": {"eval_split": "test"},
            }
        }
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()
        mock_trainer = MagicMock()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=mock_evaluator), patch("rllm.experimental.unified_trainer.AgentTrainer", return_value=mock_trainer):
            result = runner.invoke(
                cli,
                [
                    "train",
                    "train_bench",
                    "--val-dataset",
                    "val_bench",
                    "--model",
                    "test-model",
                ],
            )

        assert result.exit_code == 0
        # Both datasets should be in header
        assert "train_bench" in result.output
        assert "val_bench" in result.output

    def test_train_no_evaluator_found(self, runner, tmp_rllm_home, mock_train_dataset):
        """Train should fail if no evaluator can be resolved."""
        catalog = {"datasets": {"test_math": {"default_agent": "math"}}}
        mock_agent = _MockAgentFlow()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=None):
            result = runner.invoke(cli, ["train", "test_math", "--model", "test-model"])

        assert result.exit_code != 0
        assert "No evaluator found" in result.output

    def test_train_default_experiment_name(self, runner, tmp_rllm_home, mock_train_dataset):
        """Experiment name should default to benchmark name."""
        catalog = {"datasets": {"test_math": {"default_agent": "math", "reward_fn": "math_reward_fn", "eval_split": "test"}}}
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()
        mock_trainer = MagicMock()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=mock_evaluator), patch("rllm.experimental.unified_trainer.AgentTrainer", return_value=mock_trainer) as mock_at_cls:
            result = runner.invoke(cli, ["train", "test_math", "--model", "test-model"])

        assert result.exit_code == 0
        call_kwargs = mock_at_cls.call_args[1]
        assert call_kwargs["config"].rllm.trainer.experiment_name == "test_math"

    def test_train_default_no_ui_logger(self, runner, tmp_rllm_home, mock_train_dataset, monkeypatch):
        """When not logged in, 'ui' should NOT be in the logger list by default."""
        monkeypatch.delenv("RLLM_API_KEY", raising=False)
        catalog = {"datasets": {"test_math": {"default_agent": "math", "reward_fn": "math_reward_fn", "eval_split": "test"}}}
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()
        mock_trainer = MagicMock()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=mock_evaluator), patch("rllm.experimental.unified_trainer.AgentTrainer", return_value=mock_trainer) as mock_at_cls:
            result = runner.invoke(cli, ["train", "test_math", "--model", "test-model"])

        assert result.exit_code == 0
        call_kwargs = mock_at_cls.call_args[1]
        loggers = list(call_kwargs["config"].rllm.trainer.logger)
        assert "ui" not in loggers

    def test_train_ui_flag_appends_ui_logger(self, runner, tmp_rllm_home, mock_train_dataset, monkeypatch):
        """--ui should append 'ui' to the logger list when RLLM_API_KEY is set."""
        monkeypatch.setenv("RLLM_API_KEY", "test-key")
        catalog = {"datasets": {"test_math": {"default_agent": "math", "reward_fn": "math_reward_fn", "eval_split": "test"}}}
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()
        mock_trainer = MagicMock()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=mock_evaluator), patch("rllm.experimental.unified_trainer.AgentTrainer", return_value=mock_trainer) as mock_at_cls:
            result = runner.invoke(cli, ["train", "test_math", "--model", "test-model", "--ui"])

        assert result.exit_code == 0
        call_kwargs = mock_at_cls.call_args[1]
        loggers = list(call_kwargs["config"].rllm.trainer.logger)
        assert "ui" in loggers

    def test_train_ui_url_implies_ui(self, runner, tmp_rllm_home, mock_train_dataset):
        """--ui-url should implicitly enable UI logging (no API key needed)."""
        catalog = {"datasets": {"test_math": {"default_agent": "math", "reward_fn": "math_reward_fn", "eval_split": "test"}}}
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()
        mock_trainer = MagicMock()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=mock_evaluator), patch("rllm.experimental.unified_trainer.AgentTrainer", return_value=mock_trainer) as mock_at_cls:
            result = runner.invoke(cli, ["train", "test_math", "--model", "test-model", "--ui-url", "http://localhost:3000"])

        assert result.exit_code == 0
        call_kwargs = mock_at_cls.call_args[1]
        loggers = list(call_kwargs["config"].rllm.trainer.logger)
        assert "ui" in loggers
        assert "Live UI" in result.output
        assert "localhost:3000" in result.output

    def test_train_ui_without_api_key_errors(self, runner, tmp_rllm_home, mock_train_dataset, monkeypatch):
        """--ui without RLLM_API_KEY and no --ui-url should error."""
        monkeypatch.delenv("RLLM_API_KEY", raising=False)
        catalog = {"datasets": {"test_math": {"default_agent": "math", "reward_fn": "math_reward_fn", "eval_split": "test"}}}
        mock_agent = _MockAgentFlow()
        mock_evaluator = _MockEvaluator()

        with patch("rllm.experimental.cli.train.load_dataset_catalog", return_value=catalog), patch("rllm.experimental.eval.agent_loader.load_agent", return_value=mock_agent), patch("rllm.experimental.eval.evaluator_loader.resolve_evaluator_from_catalog", return_value=mock_evaluator):
            result = runner.invoke(cli, ["train", "test_math", "--model", "test-model", "--ui"])

        assert result.exit_code != 0
        assert "RLLM_API_KEY" in result.output
