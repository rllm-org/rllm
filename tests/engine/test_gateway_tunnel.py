"""Focused tests for owned tunnel isolation."""

from __future__ import annotations

from omegaconf import OmegaConf

from rllm.gateway.tunnel import create_tunnel


def _ngrok_url(tunnel) -> str:
    command = tunnel._command()
    return command[command.index("--url") + 1]


def test_ngrok_wildcard_expands_once_per_tunnel():
    first = create_tunnel("ngrok:*.rllm.example.ngrok.app", "http://127.0.0.1:9101")
    second = create_tunnel("ngrok:*.rllm.example.ngrok.app", "http://127.0.0.1:9102")
    first_url = _ngrok_url(first)

    assert first_url.startswith("https://rllm-")
    assert first_url.endswith(".rllm.example.ngrok.app")
    assert first_url != _ngrok_url(second)
    assert first_url == _ngrok_url(first)
    assert _ngrok_url(create_tunnel("ngrok:fixed.ngrok.app", "http://127.0.0.1:9090")) == "https://fixed.ngrok.app"


def test_configured_wildcard_precedes_daemon(monkeypatch):
    import rllm.eval.config as config_mod
    import rllm.gateway.tunnel as tunnel_mod

    monkeypatch.delenv(tunnel_mod.ENV_TUNNEL, raising=False)
    monkeypatch.setattr(config_mod, "load_tunnel_config", lambda: {"backend": "ngrok", "domain": "*.rllm.example.ngrok.app"})
    monkeypatch.setattr(tunnel_mod, "live_tunnel_url", lambda: (_ for _ in ()).throw(AssertionError("daemon must not win")))

    assert tunnel_mod.resolve_auto_tunnel() == ("ngrok:*.rllm.example.ngrok.app", None)


def test_environment_override_precedes_wildcard(monkeypatch):
    import rllm.eval.config as config_mod
    import rllm.gateway.tunnel as tunnel_mod

    monkeypatch.setenv(tunnel_mod.ENV_TUNNEL, "ngrok")
    monkeypatch.setattr(config_mod, "load_tunnel_config", lambda: {"backend": "ngrok", "domain": "*.rllm.example.ngrok.app"})

    assert tunnel_mod.resolve_auto_tunnel() == ("ngrok", None)


def test_default_resolution_skips_daemon(monkeypatch):
    import rllm.eval.config as config_mod
    import rllm.gateway.tunnel as tunnel_mod

    monkeypatch.delenv(tunnel_mod.ENV_TUNNEL, raising=False)
    monkeypatch.setattr(config_mod, "load_tunnel_config", lambda: {})
    monkeypatch.setattr(tunnel_mod, "live_tunnel_url", lambda: "https://daemon.example")

    value, _ = tunnel_mod.resolve_auto_tunnel()
    assert value == "cloudflared"
    assert tunnel_mod.resolve_auto_tunnel(reuse_daemon=True) == ("https://daemon.example", None)


def test_train_resolution_skips_daemon(monkeypatch):
    import rllm.gateway.tunnel as tunnel_mod
    import rllm.hooks as hooks_mod

    monkeypatch.setattr(tunnel_mod, "resolve_auto_tunnel", lambda: ("cloudflared", None))
    config = OmegaConf.create({"rllm": {"gateway": {"tunnel": None}}})

    assert hooks_mod.enable_gateway_tunnel(config).rllm.gateway.tunnel == "cloudflared"


def test_owned_tunnel_gets_free_port_unless_explicit(monkeypatch):
    import rllm.gateway.manager as manager_mod

    monkeypatch.setattr(manager_mod, "_find_free_port", lambda: 49123)
    automatic = OmegaConf.create({"rllm": {"gateway": {"port": None, "tunnel": "cloudflared"}}, "model": {"name": "probe"}})
    explicit = OmegaConf.create({"rllm": {"gateway": {"port": 9191, "tunnel": "cloudflared"}}, "model": {"name": "probe"}})
    local = OmegaConf.create({"rllm": {"gateway": {"port": None, "tunnel": None}}, "model": {"name": "probe"}})

    assert manager_mod.GatewayManager(automatic).port == 49123
    assert manager_mod.GatewayManager(explicit).port == 9191
    assert manager_mod.GatewayManager(local).port == 9090
