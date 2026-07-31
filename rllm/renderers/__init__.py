"""Unified renderer layer.

One bridge-capable renderer interface for every backend's models. ``get_renderer``
resolves a model to a prime-rl native renderer, a wrapped tinker/Fireworks-cookbook
renderer (e.g. DeepSeek-V4), or a chat-template fallback — all exposing
``bridge_to_next_turn`` for cumulative token mode. See design/unified-renderer-layer.md.
"""

from __future__ import annotations

from .adapters import ChatTemplateAdapter, TinkerRendererAdapter
from .bridging import BridgingRendererMixin
from .registry import Backend, RendererResolution, describe, get_renderer, resolve
from .types import ParsedResponse, RenderedTokens, Renderer

__all__ = [
    "Backend",
    "BridgingRendererMixin",
    "ChatTemplateAdapter",
    "ParsedResponse",
    "RenderedTokens",
    "Renderer",
    "RendererResolution",
    "TinkerRendererAdapter",
    "describe",
    "get_renderer",
    "resolve",
]
