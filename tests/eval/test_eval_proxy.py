"""Tests for EvalProxyManager config generation."""

from rllm.experimental.eval.proxy import EvalProxyManager


class TestEvalProxyManager:
    def test_build_proxy_config_openai(self):
        pm = EvalProxyManager(provider="openai", model_name="gpt-4o", api_key="sk-test")
        config = pm.build_proxy_config()

        assert "model_list" in config
        assert len(config["model_list"]) == 1

        entry = config["model_list"][0]
        assert entry["model_name"] == "gpt-4o"
        assert entry["litellm_params"]["model"] == "openai/gpt-4o"
        assert entry["litellm_params"]["api_key"] == "sk-test"

    def test_build_proxy_config_litellm_settings(self):
        pm = EvalProxyManager(provider="openai", model_name="gpt-4o-mini", api_key="sk-test")
        config = pm.build_proxy_config()

        assert config["litellm_settings"]["drop_params"] is True
        assert config["litellm_settings"]["num_retries"] == 3

    def test_get_proxy_url(self):
        pm = EvalProxyManager(provider="openai", model_name="gpt-4o", api_key="sk-test", proxy_port=5555)
        assert pm.get_proxy_url() == "http://127.0.0.1:5555/v1"

    def test_repr(self):
        pm = EvalProxyManager(provider="openai", model_name="gpt-4o", api_key="sk-test")
        r = repr(pm)
        assert "EvalProxyManager" in r
        assert "openai" in r
        assert "gpt-4o" in r

    def test_no_subprocess_on_init(self):
        pm = EvalProxyManager(provider="openai", model_name="gpt-4o", api_key="sk-test")
        assert pm._proxy_process is None

    def test_generate_matches_build(self):
        pm = EvalProxyManager(provider="openai", model_name="gpt-4o", api_key="sk-test")
        assert pm._generate_litellm_config() == pm.build_proxy_config()

    def test_custom_host_port(self):
        pm = EvalProxyManager(
            provider="openai",
            model_name="gpt-4o",
            api_key="sk-test",
            proxy_host="0.0.0.0",
            proxy_port=8080,
        )
        assert pm.proxy_host == "0.0.0.0"
        assert pm.proxy_port == 8080
        assert pm.get_proxy_url() == "http://0.0.0.0:8080/v1"

    def test_build_proxy_config_minimax_m27(self):
        """MiniMax M2.7 should route through minimax/ LiteLLM prefix."""
        pm = EvalProxyManager(provider="minimax", model_name="MiniMax-M2.7", api_key="mm-test-key")
        config = pm.build_proxy_config()

        assert "model_list" in config
        assert len(config["model_list"]) == 1

        entry = config["model_list"][0]
        assert entry["model_name"] == "MiniMax-M2.7"
        assert entry["litellm_params"]["model"] == "minimax/MiniMax-M2.7"
        assert entry["litellm_params"]["api_key"] == "mm-test-key"

    def test_build_proxy_config_minimax_highspeed(self):
        """MiniMax M2.7-highspeed should also route correctly."""
        pm = EvalProxyManager(provider="minimax", model_name="MiniMax-M2.7-highspeed", api_key="mm-key")
        config = pm.build_proxy_config()

        entry = config["model_list"][0]
        assert entry["model_name"] == "MiniMax-M2.7-highspeed"
        assert entry["litellm_params"]["model"] == "minimax/MiniMax-M2.7-highspeed"

    def test_minimax_repr(self):
        pm = EvalProxyManager(provider="minimax", model_name="MiniMax-M2.7", api_key="mm-key")
        r = repr(pm)
        assert "minimax" in r
        assert "MiniMax-M2.7" in r
