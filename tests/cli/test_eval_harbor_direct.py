"""CLI coverage for gateway-free Harbor eval routing."""

from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from rllm.cli.main import cli
from rllm.data.dataset import Dataset
from rllm.eval.config import RllmConfig
from rllm.eval.results import EvalItem, EvalResult


def test_harbor_pair_invokes_harbor_job_without_proxy_or_gateway(tmp_path):
    task_path = tmp_path / "task"
    task_path.mkdir()
    dataset = Dataset(data=[{"id": "task", "instruction": "fix", "task_path": str(task_path)}], name="swebench-verified", split="default")
    entry = {
        "source": "harbor:swebench-verified",
        "default_agent": "harbor:openhands-sdk",
        "eval_split": "default",
        "reward_fn": "harbor_reward_fn",
    }
    config = RllmConfig(provider="openrouter", model="qwen/qwen3.8-27b", api_keys={"openrouter": "secret"})
    harbor_config = tmp_path / "harbor.json"
    harbor_config.write_text("{}")
    harbor_run = AsyncMock(
        return_value=(
            EvalResult.from_items("swebench-verified", "qwen/qwen3.8-27b", "harbor:openhands-sdk", [EvalItem(idx=0, reward=1, is_correct=True)]),
            [],
        )
    )

    with (
        patch("rllm.eval.config.load_config", return_value=config),
        patch("rllm.cli.eval.load_dataset_catalog", return_value={"datasets": {}}),
        patch("rllm.cli._pull.resolve_harbor_catalog_entry", return_value=entry),
        patch("rllm.data.DatasetRegistry.load_dataset", return_value=dataset),
        patch("rllm.integrations.harbor.eval_runner.run_harbor_eval", harbor_run),
        patch("rllm.eval.proxy.EvalProxyManager") as proxy,
        patch("rllm.eval.runner.run_dataset") as gateway_run,
    ):
        result = CliRunner().invoke(
            cli,
            [
                "eval",
                "harbor:swebench-verified",
                "--agent",
                "harbor:openhands-sdk",
                "--provider",
                "openrouter",
                "--model",
                "qwen/qwen3.8-27b",
                "--sandbox-backend",
                "modal",
                "--ak",
                "max_iterations=50",
                "--ae",
                "EXPERIMENT=direct",
                "--harbor-config",
                str(harbor_config),
                "--episodes-dir",
                str(tmp_path / "run"),
                "--output",
                str(tmp_path / "result.json"),
                "--no-save-episodes",
                "--no-ui",
            ],
        )

    assert result.exit_code == 0, result.output
    proxy.assert_not_called()
    gateway_run.assert_not_called()
    kwargs = harbor_run.await_args.kwargs
    assert kwargs["provider"] == "openrouter"
    assert kwargs["api_key"] == "secret"
    assert kwargs["agent_kwargs"] == {"max_iterations": 50}
    assert kwargs["agent_env"] == {"EXPERIMENT": "direct"}
    assert kwargs["harbor_config"] == str(harbor_config)
