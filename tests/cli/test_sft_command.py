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


@pytest.fixture
def sft_no_train(monkeypatch):
    """Capture the dispatched SFTSpec and stub the verl backend's config build +
    data prep, so ``--config`` resolution is assertable without a real
    verl/hydra/torchrun stack (verl's own overrides->config merge is covered by
    tests/trainer test_verl_build_config_maps_spec). Returns a dict populated with
    ``'spec'`` when the command reaches dispatch."""
    from omegaconf import OmegaConf

    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    captured: dict = {}
    monkeypatch.setattr(AgentSFTTrainer, "train", lambda self: captured.setdefault("spec", self.spec))
    monkeypatch.setattr(
        VerlSFTBackend,
        "build_config",
        lambda self: OmegaConf.create({"model": {"path": self.spec.model}, "trainer": {"default_local_dir": "/tmp/x"}}),
    )
    monkeypatch.setattr(VerlSFTBackend, "prepare_data", lambda self: None)
    return captured


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


def test_dataset_import_think_tags(runner, tmp_rllm_home, tmp_path):
    """`rllm dataset import FILE --format think-tags` bridges think-tagged rows and
    registers them. Explode is ON by default: one row per assistant turn, each
    message carrying `trainable` + parts-list content.

    RED today: the `dataset` group has no `import` subcommand (exit_code != 0).
    """
    import json

    from rllm.data import DatasetRegistry

    rows = [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": '<think>\nplan A\n</think>\n\n{"cmd": "ls"}'},
                {"role": "user", "content": "out1"},
                {"role": "assistant", "content": "<think>\nplan B\n</think>\n\ndone"},
            ],
            "_task": "t1",
            "_group": "g",
            "_model": "opus",
            "_reward": 1,
        },
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<think>\nthink\n</think>\n\nhi"},
            ],
            "_task": "t2",
            "_group": "g",
            "_model": "opus",
            "_reward": 1,
        },
    ]
    f = tmp_path / "traces.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))

    result = runner.invoke(cli, ["dataset", "import", str(f), "--name", "x", "--format", "think-tags"])
    assert result.exit_code == 0, result.output

    ds = DatasetRegistry.load_dataset("x", "train")
    assert ds is not None
    total_assistant_turns = sum(1 for r in rows for m in r["messages"] if m["role"] == "assistant")
    assert len(ds) == total_assistant_turns  # explode default ON
    for row in ds.get_data():
        for m in row["messages"]:
            assert "trainable" in m
            assert isinstance(m["content"], list)


def test_sft_renderer_flag(runner, tmp_rllm_home, monkeypatch):
    """`rllm sft ... --renderer qwen3` lands in SFTSpec.overrides['data']['renderer_name'].

    RED today: there is no `--renderer` option, so Click errors with exit_code 2.
    """
    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer

    captured = {}
    monkeypatch.setattr(AgentSFTTrainer, "train", lambda self: captured.setdefault("spec", self.spec))

    name = _register_toy("renderer-toy")
    result = runner.invoke(cli, ["sft", name, "--backend", "tinker", "--renderer", "qwen3"])
    assert result.exit_code == 0, result.output
    spec = captured["spec"]
    assert spec.overrides["data"]["renderer_name"] == "qwen3"


def test_sft_logger_flag_reaches_spec(runner, tmp_rllm_home, monkeypatch):
    """`--logger wandb` resolves to ['console', 'wandb'] on the SFTSpec (UI auto-enable off)."""
    monkeypatch.delenv("RLLM_API_KEY", raising=False)
    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer

    captured = {}
    monkeypatch.setattr(AgentSFTTrainer, "train", lambda self: captured.setdefault("spec", self.spec))

    name = _register_toy("logger-toy")
    result = runner.invoke(cli, ["sft", name, "--backend", "tinker", "--logger", "wandb"])
    assert result.exit_code == 0, result.output
    assert captured["spec"].logger == ["console", "wandb"]


def test_sft_no_logger_flags_leaves_spec_none(runner, tmp_rllm_home, monkeypatch):
    """No --logger and UI auto-enable off -> spec.logger stays None (yaml default rules)."""
    monkeypatch.delenv("RLLM_API_KEY", raising=False)
    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer

    captured = {}
    monkeypatch.setattr(AgentSFTTrainer, "train", lambda self: captured.setdefault("spec", self.spec))

    name = _register_toy("nologger-toy")
    result = runner.invoke(cli, ["sft", name, "--backend", "tinker"])
    assert result.exit_code == 0, result.output
    assert captured["spec"].logger is None


def test_sft_ui_flag_appends_ui(runner, tmp_rllm_home, monkeypatch):
    """`--ui` appends 'ui' after console + any --logger values (deduped, order-preserving)."""
    monkeypatch.delenv("RLLM_API_KEY", raising=False)
    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer

    captured = {}
    monkeypatch.setattr(AgentSFTTrainer, "train", lambda self: captured.setdefault("spec", self.spec))

    name = _register_toy("ui-toy")
    result = runner.invoke(cli, ["sft", name, "--backend", "tinker", "--logger", "wandb", "--ui"])
    assert result.exit_code == 0, result.output
    assert captured["spec"].logger == ["console", "wandb", "ui"]


@pytest.mark.parametrize(
    "backend, config_text, extra_flags, expected",
    [
        pytest.param(
            "tinker",
            "model:\n  tokenizer_model: Qwen/Qwen3.6-35B-A3B\ndata:\n  renderer_name: role_colon\n",
            ["--renderer", "qwen3"],
            {("model", "tokenizer_model"): "Qwen/Qwen3.6-35B-A3B", ("data", "renderer_name"): "qwen3"},
            id="tinker-file-plus-renderer-cli-wins",
        ),
        pytest.param(
            "verl",
            "trainer:\n  n_gpus_per_node: 8\n",
            [],
            {("trainer", "n_gpus_per_node"): 8},
            id="verl-file-gpus-survives-gpus-default",
        ),
        pytest.param(
            "verl",
            "trainer:\n  n_gpus_per_node: 8\n",
            ["--gpus", "2"],
            {("trainer", "n_gpus_per_node"): 2},
            id="verl-explicit-gpus-flag-wins",
        ),
    ],
)
def test_sft_config_file_precedence(runner, tmp_rllm_home, sft_no_train, tmp_path, backend, config_text, extra_flags, expected):
    """`--config file.yaml` merges into SFTSpec.overrides; an equivalent explicit
    CLI flag beats the file (--renderer for tinker/fireworks, an explicitly passed
    --gpus for verl), while the --gpus Click default does not clobber a file value."""
    cfg = tmp_path / "override.yaml"
    cfg.write_text(config_text)
    name = _register_toy(f"cfg-prec-{backend}-{len(extra_flags)}")
    result = runner.invoke(cli, ["sft", name, "--backend", backend, "--config", str(cfg), *extra_flags])
    assert result.exit_code == 0, result.output
    overrides = sft_no_train["spec"].overrides
    for path, want in expected.items():
        node = overrides
        for key in path:
            node = node[key]
        assert node == want


def test_sft_summary_panel_shows_resolved_gpus_verl(runner, tmp_rllm_home, monkeypatch, sft_no_train, tmp_path):
    """The pre-launch panel reports cfg.trainer.n_gpus_per_node (what build_config
    resolved) rather than the raw --gpus flag. build_config here returns a fixed
    8-GPU config (the real overrides->config merge is covered by
    test_verl_build_config_maps_spec) and --gpus is 1, so the panel must show 8."""
    from omegaconf import OmegaConf

    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    monkeypatch.setattr(
        VerlSFTBackend,
        "build_config",
        lambda self: OmegaConf.create({"model": {"path": self.spec.model}, "trainer": {"default_local_dir": "/tmp/x", "n_gpus_per_node": 8}}),
    )
    name = _register_toy("gpus-panel-toy")
    result = runner.invoke(cli, ["sft", name, "--backend", "verl", "--gpus", "1"])
    assert result.exit_code == 0, result.output
    assert "8 GPUs" in result.output  # panel reads the resolved config, not --gpus
    assert "1 GPU," not in result.output
