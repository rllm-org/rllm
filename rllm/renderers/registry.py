"""Model → renderer-backend resolution for rLLM training.

``resolve`` returns a native :class:`~rllm.renderers.types.Renderer` for a model,
choosing the backend by precedence (RL-first):

1. **prime-rl** when the model is in prime-rl's ``MODEL_RENDERER_MAP`` — the only
   backend with a real ``bridge_to_next_turn`` (drift-free multi-turn token
   forwarding), so a model present in both ecosystems is routed here.
2. **tinker-cookbook / Fireworks** for models prime-rl lacks (DeepSeek-V4-Flash,
   Gemma-4, Ministral-3, Kimi-K2.7-code, …).
3. **DefaultRenderer** (prime-rl's ``apply_chat_template`` fallback) as a last
   resort — single-turn correct, ``bridge_to_next_turn`` always ``None``.

Two explicit overrides mirror the existing config knobs:

- ``renderer_family`` (prime-rl style, e.g. ``"qwen3.5"``) forces the prime-rl
  backend — same field the model gateway already uses.
- ``renderer_name`` (tinker style, e.g. ``"qwen3_5"`` / ``"deepseek_v4"``) forces
  the tinker/Fireworks backend — same field ``TinkerEngine`` already accepts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rllm.renderers import _prime, _tinker
from rllm.renderers.types import Renderer

logger = logging.getLogger(__name__)

__all__ = ["Backend", "RendererResolution", "select_backend", "describe", "resolve"]


class Backend(str, Enum):
    PRIME = "prime"
    TINKER = "tinker"
    DEFAULT = "default"


@dataclass
class RendererResolution:
    """The routing decision for a model, without instantiating anything."""

    backend: Backend
    renderer_name: str | None = None
    renderer_family: str | None = None
    has_bridge: bool = False


def _infer_model_name(model_name: str | None, tokenizer: Any) -> str | None:
    if model_name:
        return model_name
    return getattr(tokenizer, "name_or_path", None)


def select_backend(
    model_name: str | None,
    *,
    renderer_name: str | None = None,
    renderer_family: str | None = None,
) -> RendererResolution:
    """Decide which backend serves ``model_name`` (pure — no tokenizer load)."""
    if renderer_family:
        return RendererResolution(Backend.PRIME, renderer_family=renderer_family, has_bridge=True)
    if renderer_name:
        return RendererResolution(
            Backend.TINKER, renderer_name=renderer_name, has_bridge=_tinker.BRIDGE_AVAILABLE
        )
    if _prime.prime_supports(model_name):
        return RendererResolution(Backend.PRIME, renderer_family="auto", has_bridge=True)
    tname = _tinker.tinker_renderer_name(model_name)
    if tname is not None:
        return RendererResolution(
            Backend.TINKER, renderer_name=tname, has_bridge=_tinker.BRIDGE_AVAILABLE
        )
    return RendererResolution(Backend.DEFAULT, renderer_family="auto", has_bridge=False)


def describe(
    model_name: str | None,
    *,
    renderer_name: str | None = None,
    renderer_family: str | None = None,
) -> RendererResolution:
    """Alias of :func:`select_backend`, for logging / CLI introspection."""
    return select_backend(
        model_name, renderer_name=renderer_name, renderer_family=renderer_family
    )


def resolve(
    model_name: str | None,
    tokenizer: Any,
    *,
    renderer_name: str | None = None,
    renderer_family: str | None = None,
    image_processor: Any | None = None,
    **prime_kwargs: Any,
) -> Renderer:
    """Resolve a native :class:`Renderer` for ``model_name``.

    ``prime_kwargs`` (``preserve_all_thinking``, ``tool_parser``, …) are forwarded
    to the prime-rl backend and ignored by the tinker backend.
    """
    model_name = _infer_model_name(model_name, tokenizer)
    decision = select_backend(
        model_name, renderer_name=renderer_name, renderer_family=renderer_family
    )

    if decision.backend is Backend.PRIME:
        renderer = _prime.make_prime_renderer(
            tokenizer, renderer_family=decision.renderer_family or "auto", **prime_kwargs
        )
        logger.info(
            "Renderer for %r -> prime-rl %s (bridge=%s)",
            model_name, renderer.name, renderer.has_bridge,
        )
        return renderer

    if decision.backend is Backend.TINKER:
        renderer = _tinker.make_tinker_renderer(
            decision.renderer_name, tokenizer, image_processor=image_processor
        )
        logger.info(
            "Renderer for %r -> tinker/fireworks %r (no token bridge)",
            model_name, decision.renderer_name,
        )
        return renderer

    # DEFAULT: prime-rl's apply_chat_template fallback (single-turn correct).
    if _prime.PRIME_AVAILABLE:
        renderer = _prime.make_prime_renderer(tokenizer, renderer_family="auto", **prime_kwargs)
        logger.warning(
            "No hand-coded renderer for %r; falling back to %s (no token bridge, "
            "multi-turn RL will full-re-render each turn). Pass renderer_family/"
            "renderer_name to override.",
            model_name, renderer.name,
        )
        return renderer

    raise ValueError(
        f"Could not resolve a renderer for model_name={model_name!r}: it is not in "
        "prime-rl's MODEL_RENDERER_MAP, has no tinker/Fireworks renderer, and the "
        "prime-rl 'renderers' package (needed for the DefaultRenderer fallback) is "
        "not installed. Install 'renderers', or pass renderer_name explicitly."
    )
