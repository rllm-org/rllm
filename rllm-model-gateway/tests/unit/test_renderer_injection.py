"""Cumulative-mode renderer injection into create_app.

The gateway stays free of tinker/Fireworks deps: for models prime-rl lacks, the
in-process GatewayManager builds the renderer on the rllm side and injects it.
These tests cover the gateway-side contract (accept an injected renderer; reject
one without a bridge) without importing rllm.
"""

from __future__ import annotations

import pytest
from rllm_model_gateway.models import GatewayConfig
from rllm_model_gateway.server import _renderer_has_bridge, create_app


class _FakeBridgeRenderer:
    has_bridge = True

    def bridge_to_next_turn(self, *a, **k):
        return None


class _FakeNoBridgeRenderer:
    has_bridge = False


def test_renderer_has_bridge_flag_and_default_renderer():
    assert _renderer_has_bridge(_FakeBridgeRenderer()) is True
    assert _renderer_has_bridge(_FakeNoBridgeRenderer()) is False

    class DefaultRenderer:  # prime-rl's no-bridge fallback, detected by name
        pass

    assert _renderer_has_bridge(DefaultRenderer()) is False

    class Qwen3Renderer:  # prime-rl hand-coded renderer (no has_bridge attr)
        pass

    assert _renderer_has_bridge(Qwen3Renderer()) is True


def test_create_app_accepts_injected_renderer():
    cfg = GatewayConfig(cumulative_token_mode=True)  # no model needed when injected
    app = create_app(config=cfg, renderer=_FakeBridgeRenderer())
    assert app is not None


def test_create_app_rejects_injected_renderer_without_bridge():
    cfg = GatewayConfig(cumulative_token_mode=True)
    with pytest.raises(ValueError, match="no cross-turn bridge"):
        create_app(config=cfg, renderer=_FakeNoBridgeRenderer())


def test_create_app_requires_model_when_not_injected():
    cfg = GatewayConfig(cumulative_token_mode=True)  # no model, no injection
    with pytest.raises(ValueError, match="requires 'model'"):
        create_app(config=cfg)
