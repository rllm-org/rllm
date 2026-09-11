"""Focused tests for ngrok wildcard setup."""

from __future__ import annotations

import subprocess

import pytest
from click.testing import CliRunner

from rllm.cli.main import cli
from rllm.eval.config import load_tunnel_config, save_tunnel_config


@pytest.fixture(autouse=True)
def _setup(monkeypatch, tmp_path):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / ".rllm"))
    monkeypatch.setattr("rllm.gateway.tunnel.NgrokTunnel.is_available", classmethod(lambda cls: True))


def test_setup_can_reserve_wildcard_without_saving_api_key(monkeypatch):
    calls = []

    def _run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="201 Created", stderr="")

    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", _run)
    result = CliRunner().invoke(
        cli,
        ["tunnel", "setup"],
        input="1\n\n*.rllm-team.ngrok.app\ny\napi-secret\n",
    )

    assert result.exit_code == 0, result.output
    command, kwargs = calls[0]
    assert command[:4] == ["ngrok", "api", "reserved-domains", "create"]
    assert command[-2:] == ["--api-key", "api-secret"]
    assert "api-secret" not in result.output
    assert load_tunnel_config() == {"backend": "ngrok", "domain": "*.rllm-team.ngrok.app"}


def test_setup_can_use_existing_wildcard_without_api_call(monkeypatch):
    save_tunnel_config("ngrok", domain="*.existing.ngrok.app")
    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected ngrok call")))

    result = CliRunner().invoke(cli, ["tunnel", "setup"], input="1\n\n\n\n")

    assert result.exit_code == 0, result.output
    assert load_tunnel_config() == {"backend": "ngrok", "domain": "*.existing.ngrok.app"}


def test_setup_accepts_already_reserved_wildcard(monkeypatch):
    error = subprocess.CalledProcessError(1, ["ngrok"], stderr="This domain is already reserved. [ERR_NGROK_413]")
    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(error))

    result = CliRunner().invoke(cli, ["tunnel", "setup"], input="1\n\n*.existing.ngrok.app\ny\n\n")

    assert result.exit_code == 0, result.output
    assert "already reserved" in result.output
    assert load_tunnel_config() == {"backend": "ngrok", "domain": "*.existing.ngrok.app"}


def test_fixed_domain_keeps_configured_port(monkeypatch):
    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected ngrok call")))

    result = CliRunner().invoke(cli, ["tunnel", "setup"], input="1\n\ngateway.ngrok.app\n9191\n")

    assert result.exit_code == 0, result.output
    assert load_tunnel_config() == {"backend": "ngrok", "domain": "gateway.ngrok.app", "port": 9191}
