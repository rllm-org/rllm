"""Integration tests for MiniMax provider via the rLLM gateway.

These tests require a valid MINIMAX_API_KEY environment variable. They
verify that the MiniMax provider configuration generates the right
gateway route and that the API responds to a real chat completion
through the gateway.
"""

import os

import httpx
import pytest

from rllm.eval.config import RllmConfig, load_config, save_config
from rllm.eval.gateway import EvalGatewayManager

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")

requires_minimax = pytest.mark.skipif(
    not MINIMAX_API_KEY,
    reason="MINIMAX_API_KEY env var required",
)


class TestMiniMaxIntegration:
    """Integration tests for MiniMax routing through the rLLM gateway."""

    @requires_minimax
    def test_minimax_gateway_config_generation(self):
        """Verify the gateway config is correctly generated for MiniMax M2.7."""
        gm = EvalGatewayManager(
            provider="minimax",
            model_name="MiniMax-M2.7",
            api_key=MINIMAX_API_KEY,
        )
        config = gm.build_config()
        assert len(config.providers) == 1
        route = config.providers[0]
        assert route.model_name == "MiniMax-M2.7"
        assert route.backend_url == "https://api.minimaxi.chat/v1"
        assert route.api_key_env == "MINIMAX_API_KEY"
        assert os.environ["MINIMAX_API_KEY"] == MINIMAX_API_KEY

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
    def test_minimax_gateway_completion(self):
        """Verify a real MiniMax completion lands through the gateway."""
        gm = EvalGatewayManager(
            provider="minimax",
            model_name="MiniMax-M2.7",
            api_key=MINIMAX_API_KEY,
        )
        url = gm.start()
        try:
            resp = httpx.post(
                f"{url}/chat/completions",
                json={
                    "model": "MiniMax-M2.7",
                    "messages": [{"role": "user", "content": "Say hello in one word."}],
                    "max_tokens": 16,
                },
                headers={"X-RLLM-Session-Id": "minimax-smoke"},
                timeout=30.0,
            )
            resp.raise_for_status()
            body = resp.json()
            assert body["choices"]
            assert len(body["choices"][0]["message"]["content"]) > 0
        finally:
            gm.shutdown()
