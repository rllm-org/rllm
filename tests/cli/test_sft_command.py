"""Tests for the `rllm sft` CLI command (resolution + dispatch, no real training)."""

import os

import pytest
from click.testing import CliRunner

from rllm.cli.main import cli


@pytest.fixture
def tmp_rllm_home(monkeypatch, tmp_path):
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


def _register_toy(name="toy-sft"):
    from rllm.data import DatasetRegistry

    rows = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]} for _ in range(3)]
    DatasetRegistry.register_dataset(name, rows, split="train")
    return name


def test_sft_registered_in_cli():
    """`sft` shows up in the top-level command list and has help."""
    result = CliRunner().invoke(cli, ["sft", "--help"])
    assert result.exit_code == 0
    assert "Fine-tune a model" in result.output


def test_sft_requires_a_source(runner, tmp_rllm_home):
    result = runner.invoke(cli, ["sft"])
    assert result.exit_code == 1
    assert "DATASET" in result.output or "train-file" in result.output


def test_sft_missing_dataset(runner, tmp_rllm_home):
    result = runner.invoke(cli, ["sft", "no-such-dataset"])
    assert result.exit_code == 1
    assert "Could not load" in result.output


def test_sft_verl_backend_dispatches_to_launcher(runner, tmp_rllm_home, monkeypatch):
    """`--backend verl` is wired: it reaches the torchrun launcher (mocked)."""
    from omegaconf import OmegaConf

    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    monkeypatch.delenv("RLLM_SFT_IN_TORCHRUN", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    # Skip the real verl/hydra config build + parquet materialization.
    monkeypatch.setattr(
        VerlSFTBackend,
        "build_config",
        lambda self: OmegaConf.create({"model": {"path": self.spec.model}, "trainer": {"default_local_dir": "/tmp/x", "n_gpus_per_node": 2}}),
    )
    monkeypatch.setattr(VerlSFTBackend, "prepare_data", lambda self: None)
    launched = {}
    monkeypatch.setattr(AgentSFTTrainer, "_launch_distributed", lambda self, be: launched.setdefault("name", be.name))

    name = _register_toy()
    result = runner.invoke(cli, ["sft", name, "--backend", "verl", "--gpus", "2"])
    assert "not wired yet" not in result.output
    assert launched.get("name") == "verl"
    assert result.exit_code == 0
