"""Tests for EvalGatewayManager config generation and lifecycle."""

import pytest

from rllm.eval.gateway import EvalGatewayManager


class TestEvalGatewayManagerConfig:
    def test_build_config_openai(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test")
        config = gm.build_config()

        assert len(config.providers) == 1
        route = config.providers[0]
        assert route.model_name == "gpt-5-mini"
        assert route.backend_url == "https://api.openai.com/v1"
        assert route.backend_model == "gpt-5-mini"
        assert route.api_key_env == "OPENAI_API_KEY"
        # API key flowed through env (so it's never on the config object)
        import os

        assert os.environ["OPENAI_API_KEY"] == "sk-test"

    def test_build_config_disables_vllm_extensions(self):
        gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test")
        config = gm.build_config()
        assert config.add_logprobs is False
        assert config.add_return_token_ids is False
        assert config.strip_vllm_fields is False

    def test_build_config_defaults_to_shared_db(self, monkeypatch):
        """Without ``db_path``, the manager points at the shared user db."""
        monkeypatch.delenv("RLLM_GATEWAY_DB", raising=False)
        gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test")
        config = gm.build_config()
        assert config.store_worker == "sqlite"
        assert config.db_path is not None
        assert config.db_path.endswith("/.rllm/gateway/traces.db")

    def test_build_config_respects_db_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RLLM_GATEWAY_DB", str(tmp_path / "custom.db"))
        gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test")
        config = gm.build_config()
        assert config.db_path == str(tmp_path / "custom.db")

    def test_build_config_explicit_db_path_wins_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RLLM_GATEWAY_DB", str(tmp_path / "env.db"))
        explicit = str(tmp_path / "explicit.db")
        gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test", db_path=explicit)
        config = gm.build_config()
        assert config.db_path == explicit

    def test_build_config_propagates_run_id_and_metadata(self):
        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            run_id="my-run-2026",
            run_metadata={"benchmark": "gsm8k", "agent": "react"},
        )
        config = gm.build_config()
        assert config.run_id == "my-run-2026"
        assert config.run_metadata == {"benchmark": "gsm8k", "agent": "react"}

    def test_build_config_minimax(self):
        gm = EvalGatewayManager(provider="minimax", model_name="MiniMax-M2.7", api_key="mm-test")
        config = gm.build_config()
        route = config.providers[0]
        assert route.backend_url == "https://api.minimaxi.chat/v1"
        assert route.api_key_env == "MINIMAX_API_KEY"

    def test_build_config_fails_for_unknown_provider(self):
        gm = EvalGatewayManager(provider="not-a-provider", model_name="x", api_key="k")
        with pytest.raises(ValueError, match="no configured backend_url"):
            gm.build_config()

    def test_build_config_fails_for_custom_provider(self):
        # The "custom" provider exists in the registry but has no backend_url —
        # callers should bypass EvalGatewayManager and pass --base-url directly.
        gm = EvalGatewayManager(provider="custom", model_name="x", api_key="k")
        with pytest.raises(ValueError, match="no configured backend_url"):
            gm.build_config()


class TestEvalGatewayManagerProperties:
    def test_get_url_before_start(self):
        gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test", port=5555)
        assert gm.get_url() == "http://127.0.0.1:5555/v1"

    def test_repr(self):
        gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test")
        r = repr(gm)
        assert "EvalGatewayManager" in r
        assert "openai" in r
        assert "gpt-5-mini" in r

    def test_no_server_on_init(self):
        gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test")
        assert gm._server is None
        assert gm._thread is None


class TestPublicUrl:
    def test_no_public_url_keeps_loopback_bind_and_no_token(self):
        """In-process eval against docker/local: no exposure → no auth required."""
        gm = EvalGatewayManager(provider="openai", model_name="gpt-5-mini", api_key="sk-test")
        assert gm.host == "127.0.0.1"
        assert gm.public_url is None
        assert gm.inbound_auth_token is None

    def test_public_url_flips_bind_to_all_interfaces(self):
        """A public URL only routes if the gateway actually listens on a public iface."""
        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            public_url="https://abc-def.trycloudflare.com",
        )
        assert gm.host == "0.0.0.0"  # bind on all interfaces

    def test_public_url_generates_inbound_auth_token(self):
        """Public exposure → bearer token mandatory; manager generates it eagerly."""
        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            public_url="https://x.trycloudflare.com",
        )
        assert gm.inbound_auth_token is not None
        assert len(gm.inbound_auth_token) >= 32

    def test_base_url_returns_public_url_when_set(self):
        """Harnesses get the public URL — handing them 127.0.0.1 inside Modal would 404."""
        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            public_url="https://abc-def.trycloudflare.com",
        )
        assert gm.base_url == "https://abc-def.trycloudflare.com/v1"

    def test_base_url_appends_v1_if_user_omits_it(self):
        """Eval runner stamps /sessions/<sid>/v1 — base must already end in /v1."""
        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            public_url="https://abc-def.trycloudflare.com",  # no /v1
        )
        assert gm.base_url.endswith("/v1")

    def test_local_url_unchanged_by_public_override(self):
        """Readiness/diagnostics still need the bound URL even when public override is set."""
        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            host="0.0.0.0",
            port=12345,
            public_url="https://x.trycloudflare.com",
        )
        assert gm.local_url == "http://0.0.0.0:12345/v1"
        assert gm.base_url != gm.local_url

    def test_inbound_auth_token_propagates_into_gateway_config(self):
        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            public_url="https://x.trycloudflare.com",
        )
        config = gm.build_config()
        assert config.inbound_auth_token == gm.inbound_auth_token


class TestAutoTunnel:
    def test_auto_tunnel_and_public_url_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            EvalGatewayManager(
                provider="openai",
                model_name="gpt-5-mini",
                api_key="sk-test",
                public_url="https://x.trycloudflare.com",
                auto_tunnel=True,
            )

    def test_auto_tunnel_eagerly_flips_bind_and_generates_token(self):
        """At ``__init__`` we don't have a public URL yet (cloudflared
        hasn't run), but the bind interface and token generation must
        already be set so ``start()`` doesn't have to retro-fix state."""
        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            auto_tunnel=True,
        )
        assert gm.host == "0.0.0.0"
        assert gm.inbound_auth_token is not None
        # public_url is unknown until ``start()`` spawns cloudflared.
        assert gm.public_url is None

    def test_auto_tunnel_invokes_tunnel_helper_on_start(self, monkeypatch):
        """``start()`` must shell out to cloudflared after the gateway
        is ready and pin the parsed URL onto ``self.public_url``."""

        # Stub the heavy bits: never actually start uvicorn.
        captured: dict = {}

        def fake_tunnel(port, *, timeout=30.0, binary="cloudflared"):  # noqa: ARG001
            captured["port"] = port
            return ("https://stub.trycloudflare.com", _FakeProc())

        monkeypatch.setattr("rllm.eval.tunnel.start_cloudflared_tunnel", fake_tunnel)

        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            auto_tunnel=True,
        )
        gm.port = 12345
        gm._url = "http://0.0.0.0:12345/v1"
        # Drive just the tunnel-spawn branch (skip uvicorn startup).
        gm._spawn_tunnel()

        assert gm.public_url == "https://stub.trycloudflare.com"
        assert captured["port"] == 12345
        assert gm._tunnel_proc is not None

    def test_shutdown_terminates_the_tunnel_subprocess(self, monkeypatch):
        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            auto_tunnel=True,
        )
        proc = _FakeProc()
        gm._tunnel_proc = proc

        gm.shutdown()

        assert proc.terminated is True
        assert gm._tunnel_proc is None


class TestSigtermCleanup:
    """SIGTERM/atexit must close not just sandboxes but also tunnel + gateway,
    otherwise ``kill <pid>`` leaves a public URL pointing at a dying port."""

    def test_shutdown_deregisters_from_cleanup_registry(self):
        from rllm.sandbox import cleanup

        with cleanup._lock:
            cleanup._late_callbacks.clear()

        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            public_url="https://x.trycloudflare.com",
        )
        # Manually register (the real path is via ``start()`` which we can't
        # easily run hermetically without bringing up uvicorn).
        cleanup.register_late_cleanup(gm._cleanup_name, gm.shutdown)
        assert gm._cleanup_name in cleanup._late_callbacks

        gm.shutdown()  # the graceful path also deregisters

        assert gm._cleanup_name not in cleanup._late_callbacks

    def test_close_all_invokes_gateway_shutdown(self):
        """Process death triggers cleanup.close_all → reaches gateway."""
        from rllm.sandbox import cleanup

        with cleanup._lock:
            cleanup._late_callbacks.clear()

        gm = EvalGatewayManager(
            provider="openai",
            model_name="gpt-5-mini",
            api_key="sk-test",
            auto_tunnel=True,
        )
        proc = _FakeProc()
        gm._tunnel_proc = proc
        cleanup.register_late_cleanup(gm._cleanup_name, gm.shutdown)

        cleanup.close_all()  # simulates SIGTERM / atexit

        assert proc.terminated is True
        assert gm._tunnel_proc is None


class _FakeProc:
    """Minimal subprocess-like for tunnel tests."""

    def __init__(self) -> None:
        self.terminated = False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        return self.returncode or 0

    def kill(self) -> None:
        self.returncode = -9


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch):
    """Don't let API keys leak between tests via os.environ."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.delenv(key, raising=False)
