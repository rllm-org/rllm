"""Tests for one-time gateway tunnel setup."""

from __future__ import annotations

import subprocess

import pytest
from click.testing import CliRunner

from rllm.cli.main import cli
from rllm.eval.config import load_tunnel_config, save_tunnel_config


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("RLLM_HOME", str(tmp_path / ".rllm"))


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _ngrok_available(monkeypatch):
    from rllm.gateway.tunnel import NgrokTunnel

    monkeypatch.setattr(NgrokTunnel, "is_available", classmethod(lambda cls: True))


def test_setup_creates_and_saves_wildcard_without_port_or_secrets(runner, monkeypatch):
    calls: list[list[str]] = []

    def _run(command, **kwargs):
        calls.append(command)
        if command[1:4] == ["api", "reserved-domains", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout='200 OK\n{"reserved_domains": []}', stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="201 Created\n{}", stderr="")

    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", _run)

    result = runner.invoke(
        cli,
        ["tunnel", "setup"],
        input="1\n\n1\n*.rllm-team.ngrok.app\napi-secret\n",
    )

    assert result.exit_code == 0, result.output
    assert calls[0][:3] == ["ngrok", "config", "add-api-key"]
    assert calls[1][1:4] == ["api", "reserved-domains", "list"]
    assert calls[2] == [
        "ngrok",
        "api",
        "reserved-domains",
        "create",
        "--domain",
        "*.rllm-team.ngrok.app",
        "--description",
        "rLLM per-run gateway tunnels",
    ]
    assert load_tunnel_config() == {"backend": "ngrok", "domain": "*.rllm-team.ngrok.app"}
    assert "api-secret" not in result.output
    assert "own ngrok hostname and gateway port" in result.output


def test_setup_uses_existing_wildcard_without_api_call(runner, monkeypatch):
    def _unexpected_run(*args, **kwargs):
        raise AssertionError("existing wildcard setup must not call ngrok")

    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", _unexpected_run)

    result = runner.invoke(cli, ["tunnel", "setup"], input="1\n\n2\n*.existing.ngrok.app\n")

    assert result.exit_code == 0, result.output
    assert load_tunnel_config() == {"backend": "ngrok", "domain": "*.existing.ngrok.app"}


def test_setup_keeps_fixed_domain_and_port_with_warning(runner, monkeypatch):
    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))

    result = runner.invoke(cli, ["tunnel", "setup"], input="1\n\n3\ngateway.ngrok.app\n9191\n")

    assert result.exit_code == 0, result.output
    assert load_tunnel_config() == {"backend": "ngrok", "domain": "gateway.ngrok.app", "port": 9191}
    assert "only one gateway at a time" in result.output


def test_failed_wildcard_reservation_preserves_existing_config(runner, monkeypatch):
    save_tunnel_config("cloudflared", port=9090)

    def _fail(command, **kwargs):
        if command[1:4] == ["api", "reserved-domains", "list"]:
            return subprocess.CompletedProcess(command, 0, stdout='200 OK\n{"reserved_domains": []}', stderr="")
        raise subprocess.CalledProcessError(1, command, stderr="ERR_NGROK_419 plan does not support wildcards")

    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", _fail)

    result = runner.invoke(cli, ["tunnel", "setup"], input="1\n\n1\n*.new.ngrok.app\n\n")

    assert result.exit_code == 1
    assert "ERR_NGROK_419" in result.output
    assert load_tunnel_config() == {"backend": "cloudflared", "port": 9090}


def test_create_mode_is_idempotent_for_owned_wildcard(runner, monkeypatch):
    calls: list[list[str]] = []

    def _run(command, **kwargs):
        calls.append(command)
        payload = '{"reserved_domains": [{"domain": "*.existing.ngrok.app"}]}'
        return subprocess.CompletedProcess(command, 0, stdout=f"200 OK\n{payload}", stderr="")

    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", _run)

    result = runner.invoke(cli, ["tunnel", "setup"], input="1\n\n1\n*.existing.ngrok.app\n\n")

    assert result.exit_code == 0, result.output
    assert [command[1:4] for command in calls] == [["api", "reserved-domains", "list"]]
    assert "already reserved" in result.output
    assert load_tunnel_config() == {"backend": "ngrok", "domain": "*.existing.ngrok.app"}


def test_wildcard_mode_rejects_fixed_domain_before_api_call(runner, monkeypatch):
    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must validate first")))

    result = runner.invoke(cli, ["tunnel", "setup"], input="1\n\n2\nnot-a-wildcard.ngrok.app\n")

    assert result.exit_code == 1
    assert "valid DNS wildcard" in result.output


@pytest.mark.parametrize(
    "domain",
    ["*.", "*.single-label", "*.https://bad.example", "*.bad/example.com", "*.bad example.com", "*.bad:443.example.com"],
)
def test_wildcard_mode_rejects_invalid_hostnames(runner, monkeypatch, domain):
    monkeypatch.setattr("rllm.cli.tunnel.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must validate first")))

    result = runner.invoke(cli, ["tunnel", "setup"], input=f"1\n\n2\n{domain}\n")

    assert result.exit_code == 1
    assert "ngrok wildcard" in result.output


def test_tunnel_up_rejects_per_run_wildcard(runner):
    save_tunnel_config("ngrok", domain="*.rllm-team.ngrok.app")

    result = runner.invoke(cli, ["tunnel", "up"])

    assert result.exit_code == 1
    assert "created per eval/train run" in result.output
