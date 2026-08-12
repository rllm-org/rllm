"""Tunnel backend templates and automatic train-side wiring."""

from __future__ import annotations

from omegaconf import OmegaConf

from rllm.gateway.tunnel import NgrokTunnel, create_tunnel


def test_ngrok_wildcard_expands_once_per_tunnel():
    first = create_tunnel("ngrok:*.rllm.example.ngrok.app", "http://127.0.0.1:9101")
    second = create_tunnel("ngrok:*.rllm.example.ngrok.app", "http://127.0.0.1:9102")

    assert isinstance(first, NgrokTunnel)
    assert isinstance(second, NgrokTunnel)
    first_url = first._command()[first._command().index("--url") + 1]
    second_url = second._command()[second._command().index("--url") + 1]
    assert first_url.startswith("https://rllm-")
    assert first_url.endswith(".rllm.example.ngrok.app")
    assert first_url != second_url
    assert first._command()[first._command().index("--url") + 1] == first_url


def test_fixed_ngrok_domain_is_unchanged():
    tunnel = create_tunnel("ngrok:gateway.example.ngrok.app", "http://127.0.0.1:9090")

    command = tunnel._command()
    assert command[command.index("--url") + 1] == "https://gateway.example.ngrok.app"


def test_unset_gateway_port_keeps_legacy_9090_without_auto_wiring():
    from rllm.gateway.manager import GatewayManager

    config = OmegaConf.create({"rllm": {"gateway": {"port": None}}, "model": {"name": "probe"}})

    assert GatewayManager(config).port == 9090


def test_train_auto_resolution_prefers_configured_wildcard(monkeypatch):
    import rllm.eval.config as config_mod
    import rllm.gateway.tunnel as tunnel_mod

    monkeypatch.delenv(tunnel_mod.ENV_TUNNEL, raising=False)
    monkeypatch.setattr(config_mod, "load_tunnel_config", lambda: {"backend": "ngrok", "domain": "*.rllm.example.ngrok.app"})
    monkeypatch.setattr(tunnel_mod, "live_tunnel_url", lambda: (_ for _ in ()).throw(AssertionError("daemon must not win")))

    assert tunnel_mod.resolve_auto_tunnel() == ("ngrok:*.rllm.example.ngrok.app", None)


def test_train_wildcard_gets_automatic_gateway_port(monkeypatch):
    import rllm.gateway.manager as manager_mod
    import rllm.gateway.tunnel as tunnel_mod
    import rllm.hooks as hooks_mod

    config = OmegaConf.create({"rllm": {"gateway": {"port": None, "tunnel": None}}})
    monkeypatch.setattr(tunnel_mod, "resolve_auto_tunnel", lambda: ("ngrok:*.rllm.example.ngrok.app", None))
    monkeypatch.setattr(manager_mod, "_find_free_port", lambda: 49123)

    resolved = hooks_mod.enable_gateway_tunnel(config)

    assert resolved.rllm.gateway.tunnel == "ngrok:*.rllm.example.ngrok.app"
    assert resolved.rllm.gateway.port == 49123


def test_train_wildcard_preserves_explicit_gateway_port(monkeypatch):
    import rllm.gateway.tunnel as tunnel_mod
    import rllm.hooks as hooks_mod

    config = OmegaConf.create({"rllm": {"gateway": {"port": 9191, "tunnel": None}}})
    monkeypatch.setattr(tunnel_mod, "resolve_auto_tunnel", lambda: ("ngrok:*.rllm.example.ngrok.app", None))

    resolved = hooks_mod.enable_gateway_tunnel(config)

    assert resolved.rllm.gateway.tunnel == "ngrok:*.rllm.example.ngrok.app"
    assert resolved.rllm.gateway.port == 9191


def test_train_cloudflared_default_gets_automatic_gateway_port(monkeypatch):
    import rllm.gateway.manager as manager_mod
    import rllm.gateway.tunnel as tunnel_mod
    import rllm.hooks as hooks_mod

    config = OmegaConf.create({"rllm": {"gateway": {"port": None, "tunnel": None}}})
    monkeypatch.setattr(tunnel_mod, "resolve_auto_tunnel", lambda: ("cloudflared", None))
    monkeypatch.setattr(manager_mod, "_find_free_port", lambda: 49124)

    resolved = hooks_mod.enable_gateway_tunnel(config)

    assert resolved.rllm.gateway.tunnel == "cloudflared"
    assert resolved.rllm.gateway.port == 49124


def test_train_environment_wildcard_gets_automatic_gateway_port(monkeypatch):
    import rllm.gateway.manager as manager_mod
    import rllm.gateway.tunnel as tunnel_mod
    import rllm.hooks as hooks_mod

    config = OmegaConf.create({"rllm": {"gateway": {"port": None, "tunnel": None}}})
    monkeypatch.setenv(tunnel_mod.ENV_TUNNEL, "ngrok:*.env.example.ngrok.app")
    monkeypatch.setattr(manager_mod, "_find_free_port", lambda: 49125)

    resolved = hooks_mod.enable_gateway_tunnel(config)

    assert resolved.rllm.gateway.tunnel == "ngrok:*.env.example.ngrok.app"
    assert resolved.rllm.gateway.port == 49125


def test_explicit_train_tunnel_and_port_are_unchanged():
    from rllm.hooks import enable_gateway_tunnel

    config = OmegaConf.create({"rllm": {"gateway": {"port": 9191, "tunnel": "ngrok:fixed.example.ngrok.app"}}})

    assert enable_gateway_tunnel(config) is config
