"""Tests for EvalProxyManager config generation and process isolation."""

from pathlib import Path
from unittest.mock import Mock

import rllm.eval.proxy as proxy_module
from rllm.eval.proxy import EvalProxyManager


def _mock_proxy_startup(monkeypatch):
    commands = []
    monkeypatch.delenv("RLLM_PROXY_CONFIG_DIR", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_STATE_DIR", raising=False)

    def fake_popen(cmd, **kwargs):
        commands.append(cmd)
        return Mock(pid=1000, returncode=None, poll=Mock(return_value=None), wait=Mock(return_value=0))

    monkeypatch.setattr(proxy_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(EvalProxyManager, "_wait_for_proxy", lambda self, timeout: None)
    monkeypatch.setattr(EvalProxyManager, "reload_proxy_config", lambda self, config: {"status": "ok"})
    monkeypatch.setattr(EvalProxyManager, "_start_proxy_monitor", lambda self: None)
    monkeypatch.setattr(proxy_module.atexit, "register", lambda callback: None)
    return commands


def _state_dir(command):
    return Path(command[command.index("--state-dir") + 1])


class TestEvalProxyManager:
    def test_concurrent_managers_use_unique_state_dirs(self, monkeypatch, tmp_path):
        commands = _mock_proxy_startup(monkeypatch)
        monkeypatch.chdir(tmp_path)

        first = EvalProxyManager(provider="openai", model_name="model-a", api_key="key-a", proxy_port=5001)
        second = EvalProxyManager(provider="openai", model_name="model-b", api_key="key-b", proxy_port=5002)
        first_snapshot = Path(first.start_proxy_subprocess(first.build_proxy_config()))
        second_snapshot = Path(second.start_proxy_subprocess(second.build_proxy_config()))

        first_state = _state_dir(commands[0])
        second_state = _state_dir(commands[1])
        assert first_snapshot.parent != second_snapshot.parent
        assert first_state != second_state
        assert first_state == first_snapshot.parent
        assert second_state == second_snapshot.parent
        assert first_state.parent == second_state.parent == tmp_path / ".litellm_proxy"

        first.shutdown_proxy()
        second.shutdown_proxy()
        assert first_state.exists()
        assert second_state.exists()

    def test_explicit_config_and_state_dirs_keep_exact_paths(self, monkeypatch, tmp_path):
        commands = _mock_proxy_startup(monkeypatch)
        monkeypatch.chdir(tmp_path)
        config_dir = tmp_path / "config"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        monkeypatch.setenv("RLLM_PROXY_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("LITELLM_PROXY_STATE_DIR", str(state_dir))
        manager = EvalProxyManager(provider="openai", model_name="model-a", api_key="key-a", proxy_port=5001)

        snapshot = Path(manager.start_proxy_subprocess(manager.build_proxy_config()))
        assert snapshot == config_dir / "litellm_proxy_config_autogen.yaml"
        assert _state_dir(commands[0]) == state_dir

        manager.shutdown_proxy()
        assert snapshot.exists()
        assert not (tmp_path / ".litellm_proxy").exists()

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

    def test_build_proxy_config_minimax_m3(self):
        """MiniMax M3 should route through minimax/ LiteLLM prefix."""
        pm = EvalProxyManager(provider="minimax", model_name="MiniMax-M3", api_key="mm-test-key")
        config = pm.build_proxy_config()

        assert "model_list" in config
        assert len(config["model_list"]) == 1

        entry = config["model_list"][0]
        assert entry["model_name"] == "MiniMax-M3"
        assert entry["litellm_params"]["model"] == "minimax/MiniMax-M3"
        assert entry["litellm_params"]["api_key"] == "mm-test-key"

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


class TestSamplingExtraPassthrough:
    """``drop_params`` deletes params outside litellm's per-provider allowlist,
    even ones the provider honours (fireworks_ai has no ``reasoning_effort``
    entry). Those must ride in ``extra_body``, which litellm forwards verbatim,
    or the run silently uses the model's default and still looks valid."""

    FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"

    def test_unsupported_param_moves_to_extra_body(self):
        pm = EvalProxyManager(provider="fireworks", model_name=self.FIREWORKS_MODEL, api_key="fw-key", sampling_extra={"reasoning_effort": "max"})

        params = pm.build_proxy_config()["model_list"][0]["litellm_params"]
        assert params["extra_body"] == {"reasoning_effort": "max"}

    def test_supported_param_is_not_diverted_to_extra_body(self):
        """litellm-known params must not be duplicated into ``extra_body``."""
        pm = EvalProxyManager(provider="fireworks", model_name=self.FIREWORKS_MODEL, api_key="fw-key", sampling_extra={"top_k": 20, "reasoning_effort": "max"})

        params = pm.build_proxy_config()["model_list"][0]["litellm_params"]
        assert params["extra_body"] == {"reasoning_effort": "max"}

    def test_no_extra_body_key_without_extras(self):
        """Runs that pass no extras must generate the same config as before."""
        pm = EvalProxyManager(provider="fireworks", model_name=self.FIREWORKS_MODEL, api_key="fw-key")

        assert "extra_body" not in pm.build_proxy_config()["model_list"][0]["litellm_params"]
