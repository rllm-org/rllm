"""Resolve a (model, backend) to the best available renderer.

Priority:
1. Explicit override — ``family`` (prime-rl native) or ``renderer_name`` (tinker
   style, e.g. ``"deepseek_v4"``).
2. prime-rl native renderer if the model is in its ``MODEL_RENDERER_MAP``
   (hand-tuned, parity-tested, first-class ``bridge_to_next_turn``).
3. Tinker / Fireworks-cookbook renderer wrapped by ``TinkerRendererAdapter``
   (the path for models prime-rl doesn't cover yet, e.g. DeepSeek-V4).
4. ``ChatTemplateAdapter`` fallback (logged loudly; bridge prefix-check guards).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .adapters import ChatTemplateAdapter, TinkerRendererAdapter

logger = logging.getLogger(__name__)


class Backend(str, Enum):
    VERL = "verl"
    TINKER = "tinker"
    FIREWORKS = "fireworks"


# Models prime-rl doesn't cover, served via a tinker-style renderer name. The
# renderer class is resolved through tinker_cookbook.renderers.get_renderer;
# Fireworks-cookbook renderers self-register on import (see _FW_MODULES).
_TINKER_NAME_BY_MODEL: dict[str, str] = {
    "deepseek-ai/DeepSeek-V4": "deepseek_v4",
    "deepseek-ai/DeepSeek-V4-Flash": "deepseek_v4",
}

# Candidate import paths that register Fireworks-cookbook renderers. Imported
# lazily and best-effort — only present on a box with the cookbook installed.
_FW_MODULES: tuple[str, ...] = (
    "fireworks_cookbook.renderers.deepseek_v4",
    "fireworks_training_cookbook.renderers.deepseek_v4",
)


@dataclass
class RendererResolution:
    renderer: Any
    source: str  # "prime", "tinker", "chat_template"
    name: str


def _infer_model_name(model: str | None, tokenizer: Any) -> str:
    return model or getattr(tokenizer, "name_or_path", "") or ""


def _try_prime(tokenizer: Any, family: str) -> Any | None:
    """prime-rl native renderer, or None if it falls back to DefaultRenderer."""
    try:
        from renderers import create_renderer  # type: ignore
    except ImportError:
        return None
    renderer = create_renderer(tokenizer, renderer=family)
    if type(renderer).__name__ == "DefaultRenderer":
        return None
    return renderer


def _ensure_fw_registered() -> None:
    for mod in _FW_MODULES:
        try:
            __import__(mod)
            return
        except ImportError:
            continue


def _tinker_adapter(renderer_name: str, tokenizer: Any, *, model: str = "") -> TinkerRendererAdapter:
    from tinker_cookbook import renderers as tk  # type: ignore

    try:
        inner = tk.get_renderer(renderer_name, tokenizer, model_name=model or None)
    except Exception:
        _ensure_fw_registered()
        inner = tk.get_renderer(renderer_name, tokenizer, model_name=model or None)
    return TinkerRendererAdapter(inner)


def resolve(
    model: str | None,
    tokenizer: Any,
    *,
    backend: Backend | str | None = None,
    family: str = "auto",
    renderer_name: str | None = None,
) -> RendererResolution:
    name = _infer_model_name(model, tokenizer)

    # 1a. Explicit tinker-style override.
    if renderer_name:
        return RendererResolution(_tinker_adapter(renderer_name, tokenizer, model=name), "tinker", renderer_name)

    # 1b. Explicit prime-rl family.
    if family and family != "auto":
        renderer = _try_prime(tokenizer, family)
        if renderer is not None:
            return RendererResolution(renderer, "prime", family)
        logger.warning("prime-rl family=%r did not resolve a native renderer; continuing auto-resolution.", family)

    # 2. prime-rl native by exact model match.
    renderer = _try_prime(tokenizer, "auto")
    if renderer is not None:
        return RendererResolution(renderer, "prime", type(renderer).__name__)

    # 3. Tinker / Fireworks-cookbook renderer for known models.
    tk_name = _TINKER_NAME_BY_MODEL.get(name)
    if tk_name:
        return RendererResolution(_tinker_adapter(tk_name, tokenizer, model=name), "tinker", tk_name)

    # 3b. Tinker auto-recommendation (covers models tinker_cookbook knows).
    try:
        from tinker_cookbook import model_info  # type: ignore

        rec = model_info.get_recommended_renderer_name(name)
        if rec:
            return RendererResolution(_tinker_adapter(rec, tokenizer, model=name), "tinker", rec)
    except Exception:
        pass

    # 4. Fallback.
    logger.warning(
        "No prime-rl/tinker renderer for model=%r; using ChatTemplateAdapter "
        "(chat-template parity not guaranteed; cumulative bridge is best-effort). "
        "Pass renderer_name=<tinker name> or family=<prime family> to pin one.",
        name or "<unnamed>",
    )
    return RendererResolution(ChatTemplateAdapter(tokenizer), "chat_template", "chat_template")


def get_renderer(
    model: str | None,
    tokenizer: Any,
    *,
    backend: Backend | str | None = None,
    family: str = "auto",
    renderer_name: str | None = None,
):
    """Return a renderer satisfying the canonical interface (incl. ``bridge_to_next_turn``)."""
    res = resolve(model, tokenizer, backend=backend, family=family, renderer_name=renderer_name)
    logger.info("renderer: model=%r -> %s (%s)", _infer_model_name(model, tokenizer), res.name, res.source)
    return res.renderer


def describe(model: str | None, tokenizer: Any, **kwargs) -> str:
    res = resolve(model, tokenizer, **kwargs)
    return f"{res.source}:{res.name}"
