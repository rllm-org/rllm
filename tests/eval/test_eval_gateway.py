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


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch):
    """Don't let API keys leak between tests via os.environ."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.delenv(key, raising=False)
