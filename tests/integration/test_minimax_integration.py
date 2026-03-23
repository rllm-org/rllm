"""Integration tests for MiniMax provider via LiteLLM.

These tests require a valid MINIMAX_API_KEY environment variable.
They verify that the MiniMax provider configuration generates correct
LiteLLM routing and that the API responds to basic requests.
"""

import os

import pytest

from rllm.experimental.eval.config import get_provider_info, load_config, save_config, RllmConfig
from rllm.experimental.eval.proxy import EvalProxyManager

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")

requires_minimax = pytest.mark.skipif(
    not MINIMAX_API_KEY,
    reason="MINIMAX_API_KEY env var required",
)


class TestMiniMaxIntegration:
    """Integration tests that verify MiniMax LiteLLM routing end-to-end."""

    @requires_minimax
    def test_minimax_litellm_config_generation(self):
        """Verify LiteLLM config is correctly generated for MiniMax M2.7."""
        pm = EvalProxyManager(
            provider="minimax",
            model_name="MiniMax-M2.7",
            api_key=MINIMAX_API_KEY,
        )
        config = pm.build_proxy_config()

        entry = config["model_list"][0]
        assert entry["litellm_params"]["model"] == "minimax/MiniMax-M2.7"
        assert entry["litellm_params"]["api_key"] == MINIMAX_API_KEY

    @requires_minimax
    def test_minimax_config_save_and_load(self, tmp_path, monkeypatch):
        """Verify MiniMax config persists correctly through save/load cycle."""
        monkeypatch.setenv("RLLM_HOME", str(tmp_path / ".rllm"))

        original = RllmConfig(
            provider="minimax",
            api_keys={"minimax": MINIMAX_API_KEY},
            model="MiniMax-M2.7",
        )
        save_config(original)

        loaded = load_config()
        assert loaded.provider == "minimax"
        assert loaded.model == "MiniMax-M2.7"
        assert loaded.api_key == MINIMAX_API_KEY
        assert loaded.is_configured()
        assert loaded.validate() == []

    @requires_minimax
    def test_minimax_litellm_completion(self):
        """Verify MiniMax M2.7 responds via LiteLLM completion (no proxy)."""
        import litellm

        response = litellm.completion(
            model="minimax/MiniMax-M2.7",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            api_key=MINIMAX_API_KEY,
            max_tokens=16,
        )
        assert response.choices
        assert len(response.choices[0].message.content) > 0
