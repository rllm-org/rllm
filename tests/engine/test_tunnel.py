"""Tests for gateway tunnel backends and spec parsing (rllm/gateway/tunnel.py)."""

import pytest

from rllm.gateway.tunnel import (
    ENV_CF_TUNNEL_TOKEN,
    CloudflaredTunnel,
    CloudflareNamedTunnel,
    NgrokTunnel,
    TunnelStartError,
    create_tunnel,
    parse_tunnel,
)

UPSTREAM = "http://127.0.0.1:9090"


class TestParseTunnel:
    def test_url_passes_through(self):
        assert parse_tunnel("https://gw.example.com") == ("https://gw.example.com", None)

    def test_backend_spec_preserves_case(self):
        # Cloudflare tunnel names are case-sensitive; the spec must survive verbatim.
        assert parse_tunnel("cloudflare:rllm.example.com@MyTunnel") == (None, "cloudflare:rllm.example.com@MyTunnel")

    def test_empty(self):
        assert parse_tunnel(None) == (None, None)
        assert parse_tunnel("") == (None, None)


class TestCreateTunnel:
    def test_quick_tunnel(self):
        assert isinstance(create_tunnel("cloudflared", UPSTREAM), CloudflaredTunnel)

    def test_ngrok_domain(self):
        tnl = create_tunnel("ngrok:rllm.ngrok.dev", UPSTREAM)
        assert isinstance(tnl, NgrokTunnel)
        assert tnl.domain == "rllm.ngrok.dev"

    def test_cloudflare_token_mode(self):
        tnl = create_tunnel("cloudflare:rllm.example.com", UPSTREAM)
        assert isinstance(tnl, CloudflareNamedTunnel)
        assert tnl.hostname == "rllm.example.com"
        assert tnl.tunnel_ref is None

    def test_cloudflare_named_mode(self):
        tnl = create_tunnel("cloudflare:rllm.example.com@my-tunnel", UPSTREAM)
        assert isinstance(tnl, CloudflareNamedTunnel)
        assert tnl.hostname == "rllm.example.com"
        assert tnl.tunnel_ref == "my-tunnel"

    def test_cloudflare_backend_name_case_insensitive(self):
        tnl = create_tunnel("CLOUDFLARE:rllm.example.com@MyTunnel", UPSTREAM)
        assert isinstance(tnl, CloudflareNamedTunnel)
        assert tnl.tunnel_ref == "MyTunnel"

    def test_cloudflare_hostname_scheme_stripped(self):
        tnl = create_tunnel("cloudflare:https://rllm.example.com/", UPSTREAM)
        assert tnl.hostname == "rllm.example.com"

    def test_cloudflare_requires_hostname(self):
        with pytest.raises(ValueError, match="hostname"):
            create_tunnel("cloudflare", UPSTREAM)
        with pytest.raises(ValueError, match="hostname"):
            create_tunnel("cloudflare:", UPSTREAM)

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unsupported gateway tunnel backend"):
            create_tunnel("wireguard", UPSTREAM)


class TestCloudflareNamedTunnel:
    def test_named_command_pins_upstream(self):
        tnl = CloudflareNamedTunnel(UPSTREAM, hostname="rllm.example.com", tunnel_ref="my-tunnel")
        assert tnl._command() == ["cloudflared", "tunnel", "run", "--url", UPSTREAM, "my-tunnel"]
        assert tnl._popen_env() is None

    def test_token_command_uses_env_not_argv(self, monkeypatch):
        monkeypatch.setenv("TUNNEL_TOKEN", "tok-123")
        monkeypatch.delenv(ENV_CF_TUNNEL_TOKEN, raising=False)
        tnl = CloudflareNamedTunnel(UPSTREAM, hostname="rllm.example.com")
        # Ingress is dashboard-managed, so no --url; token stays out of argv.
        assert tnl._command() == ["cloudflared", "tunnel", "run"]
        assert tnl._popen_env()["TUNNEL_TOKEN"] == "tok-123"

    def test_rllm_token_alias_is_forwarded(self, monkeypatch):
        monkeypatch.delenv("TUNNEL_TOKEN", raising=False)
        monkeypatch.setenv(ENV_CF_TUNNEL_TOKEN, "tok-456")
        tnl = CloudflareNamedTunnel(UPSTREAM, hostname="rllm.example.com")
        assert tnl._command() == ["cloudflared", "tunnel", "run"]
        assert tnl._popen_env()["TUNNEL_TOKEN"] == "tok-456"

    def test_token_mode_without_token_fails(self, monkeypatch):
        monkeypatch.delenv("TUNNEL_TOKEN", raising=False)
        monkeypatch.delenv(ENV_CF_TUNNEL_TOKEN, raising=False)
        tnl = CloudflareNamedTunnel(UPSTREAM, hostname="rllm.example.com")
        with pytest.raises(TunnelStartError, match="TUNNEL_TOKEN"):
            tnl._command()

    def test_url_derived_from_hostname_on_edge_registration(self):
        tnl = CloudflareNamedTunnel(UPSTREAM, hostname="rllm.example.com", tunnel_ref="my-tunnel")
        assert tnl._extract_url("2026-01-01T00:00:00Z INF Registered tunnel connection connIndex=0 location=icn") == "https://rllm.example.com"
        assert tnl._extract_url("2026-01-01T00:00:00Z INF Starting tunnel tunnelID=abc") is None

    def test_fatal_hints(self):
        tnl = CloudflareNamedTunnel(UPSTREAM, hostname="rllm.example.com", tunnel_ref="my-tunnel")
        assert "token" in tnl._classify_fatal("Provided Tunnel token is not valid")
        assert "cloudflared tunnel login" in tnl._classify_fatal("Cannot determine default origin certificate path. No file cert.pem in [~/.cloudflared]")
        assert "cloudflared tunnel list" in tnl._classify_fatal("Tunnel credentials file '/x/y.json' doesn't exist")
        assert tnl._classify_fatal("some unrelated line") is None


class TestTunnelConfig:
    def test_save_and_load_cloudflare_tunnel_name(self, tmp_path, monkeypatch):
        import rllm.eval.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "_config_path", lambda: str(tmp_path / "config.json"))
        cfg_mod.save_tunnel_config("cloudflare", domain="rllm.example.com", name="my-tunnel", port=9090)
        assert cfg_mod.load_tunnel_config() == {"backend": "cloudflare", "domain": "rllm.example.com", "name": "my-tunnel", "port": 9090}
