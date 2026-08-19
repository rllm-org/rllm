"""CLI routing tests for Harbor's gateway-free eval runtime."""

from unittest.mock import patch

from click.testing import CliRunner

from rllm.cli.main import cli
from rllm.eval.config import RllmConfig


def test_harbor_pair_skips_proxy_and_forwards_native_options(tmp_path):
    config = RllmConfig(
        provider="openrouter",
        model="qwen/qwen3.8-27b",
        api_keys={"openrouter": "or-test"},
    )
    config_path = tmp_path / "harbor.json"
    config_path.write_text('{"job_name": "base"}')

    with (
        patch("rllm.eval.config.load_config", return_value=config),
        patch("rllm.eval.proxy.EvalProxyManager") as proxy_cls,
        patch("rllm.cli.eval._run_eval") as run_eval,
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
                "--ak",
                "max_iterations=50",
                "--ak",
                "load_skills=false",
                "--ae",
                "EXPERIMENT=direct",
                "--harbor-config",
                str(config_path),
                "--no-ui",
            ],
        )

    assert result.exit_code == 0, result.output
    proxy_cls.assert_not_called()
    kwargs = run_eval.call_args.kwargs
    assert kwargs["provider"] == "openrouter"
    assert kwargs["api_key"] == "or-test"
    assert kwargs["harbor_agent_kwargs"] == {"max_iterations": 50, "load_skills": False}
    assert kwargs["harbor_agent_env"] == {"EXPERIMENT": "direct"}
    assert kwargs["harbor_config"] == str(config_path)


def test_harbor_agent_kwarg_rejects_invalid_format():
    with patch("rllm.cli.eval._run_eval") as run_eval:
        result = CliRunner().invoke(
            cli,
            [
                "eval",
                "harbor:swebench-verified",
                "--agent",
                "harbor:openhands-sdk",
                "--base-url",
                "https://openrouter.ai/api/v1",
                "--model",
                "openrouter/qwen/qwen3.8-27b",
                "--ak",
                "missing-equals",
                "--no-ui",
            ],
        )

    assert result.exit_code != 0
    assert "Expected key=value" in result.output
    run_eval.assert_not_called()
