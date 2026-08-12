"""Eval-specific tunnel selection and gateway wiring."""

from __future__ import annotations

import asyncio

import pytest

import rllm.eval.runner as runner_mod
from rllm.gateway.tunnel import ENV_TUNNEL


@pytest.fixture(autouse=True)
def _no_tunnel_override(monkeypatch):
    monkeypatch.delenv(ENV_TUNNEL, raising=False)


@pytest.mark.parametrize("override", ["ngrok", "https://gateway.example.test"])
def test_explicit_environment_override_wins(monkeypatch, override):
    monkeypatch.setenv(ENV_TUNNEL, override)

    assert runner_mod._resolve_eval_tunnel() == override


def test_configured_ngrok_wildcard_is_owned_by_eval(monkeypatch):
    import rllm.eval.config as config_mod

    monkeypatch.setattr(config_mod, "load_tunnel_config", lambda: {"backend": "ngrok", "domain": "*.rllm.example.ngrok.app"})

    assert runner_mod._resolve_eval_tunnel() == "ngrok:*.rllm.example.ngrok.app"


def test_fixed_ngrok_domain_is_not_reused_by_eval(monkeypatch):
    import rllm.eval.config as config_mod

    monkeypatch.setattr(config_mod, "load_tunnel_config", lambda: {"backend": "ngrok", "domain": "gateway.example.ngrok.app"})

    assert runner_mod._resolve_eval_tunnel() == "cloudflared"


def test_persistent_daemon_is_not_consulted(monkeypatch):
    import rllm.eval.config as config_mod
    import rllm.gateway.tunnel as tunnel_mod

    def _unexpected_daemon_lookup():
        raise AssertionError("eval tunnel resolution must not inspect the persistent daemon")

    monkeypatch.setattr(config_mod, "load_tunnel_config", lambda: {})
    monkeypatch.setattr(tunnel_mod, "live_tunnel_url", _unexpected_daemon_lookup)

    assert runner_mod._resolve_eval_tunnel() == "cloudflared"


def test_remote_eval_passes_owned_backend_with_automatic_port(monkeypatch):
    import rllm.gateway.manager as manager_mod
    import rllm.gateway.tunnel as tunnel_mod

    monkeypatch.setattr(runner_mod, "_resolve_eval_tunnel", lambda: "cloudflared")
    monkeypatch.setattr(tunnel_mod, "is_local_sandbox_backend", lambda _: False)

    seen: dict = {}

    class _Stop(Exception):
        pass

    class _Gateway:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def start(self):
            raise _Stop

    monkeypatch.setattr(manager_mod, "EvalGatewayManager", _Gateway)

    with pytest.raises(_Stop):
        asyncio.run(
            runner_mod.run_dataset(
                tasks=[],
                agent_flow=object(),
                base_url="http://127.0.0.1:1/v1",
                model="probe-model",
                sandbox_backend="modal",
            )
        )

    assert seen["tunnel"] == "cloudflared"
    assert seen["port"] is None
