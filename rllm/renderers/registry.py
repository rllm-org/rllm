"""Resolve a (model, backend) to the best available renderer.

Priority:
1. Explicit override — ``family`` (prime-rl native) or ``renderer_name`` (tinker
   style, e.g. ``"deepseek_v4"``).
2. prime-rl native renderer if the model is in its ``MODEL_RENDERER_MAP``
   (hand-tuned, parity-tested, first-class ``bridge_to_next_turn``).
3. Fireworks-cookbook renderer (``glm5``, ``deepseek_v4``, ...) auto-detected by
   model-family prefix and wrapped by ``TinkerRendererAdapter`` — the path for
   models prime-rl doesn't cover (e.g. GLM-5.2). No config needed.
4. ``ChatTemplateAdapter`` fallback (logged loudly; bridge prefix-check guards).

This resolution is used by both the gateway (turns 1+ bridge) and FireworksEngine
(turn-0 render), so a given model resolves to the *same* renderer on both sides.
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


# Fireworks-cookbook renderer names (training/renderer/__init__.py registers
# glm5, deepseek_v4, gemma4, kimi_k27_code, minimax_m2, ...), keyed by canonical
# HF-id prefix. The cookbook ships the renderer *implementations* but no
# model->renderer map, and tinker_cookbook's recommender doesn't know these
# models — so we own the mapping. prime-rl's exact-match map is tried first, so
# this only catches models prime-rl doesn't cover (e.g. GLM-5.2). Family-level:
# one cookbook renderer serves a whole minor series (glm5 -> GLM-5.x). Prefix
# match (not exact) so new point releases resolve without a code change; extend
# as the cookbook adds families.
_FW_COOKBOOK_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("zai-org/glm-5", "glm5"),
    ("deepseek-ai/deepseek-v4", "deepseek_v4"),
)

# Import paths that register the Fireworks cookbook's renderers (they self-register
# into tinker_cookbook's registry on import). ``training.renderer`` is the package
# name on a Fireworks box (cf. ``from training.utils import ...``). Best-effort —
# only present where the cookbook is installed.
_FW_MODULES: tuple[str, ...] = (
    "training.renderer",
    "fireworks_cookbook.renderers",
    "fireworks_training_cookbook.renderers",
)


def _cookbook_renderer_name(model_name: str) -> str | None:
    """Fireworks-cookbook renderer name for *model_name*, or None."""
    n = (model_name or "").lower()
    for prefix, rname in _FW_COOKBOOK_BY_PREFIX:
        if n.startswith(prefix):
            return rname
    return None


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

    # Import the Fireworks cookbook first so its renderers (glm5, deepseek_v4, ...)
    # are registered before we ask for one by name.
    _ensure_fw_registered()
    inner = tk.get_renderer(renderer_name, tokenizer, model_name=model or None)
    return TinkerRendererAdapter(inner)


def _try_tinker_adapter(renderer_name: str, tokenizer: Any, *, model: str = "") -> TinkerRendererAdapter | None:
    """Build a tinker/cookbook adapter, or None if the renderer can't be loaded
    (e.g. the Fireworks cookbook isn't installed on this box)."""
    try:
        return _tinker_adapter(renderer_name, tokenizer, model=model)
    except Exception as e:  # noqa: BLE001 - resolution must never hard-fail
        logger.debug("tinker/cookbook renderer %r unavailable: %s", renderer_name, e)
        return None


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

    # 3. Fireworks-cookbook renderer for families prime-rl doesn't cover (e.g. GLM-5.2).
    ck = _cookbook_renderer_name(name)
    if ck:
        adapter = _try_tinker_adapter(ck, tokenizer, model=name)
        if adapter is not None:
            return RendererResolution(adapter, "tinker", ck)
        logger.warning(
            "Model %r maps to Fireworks-cookbook renderer %r, but it could not be loaded (is the Fireworks cookbook 'training.renderer' importable here?). Falling back.",
            name,
            ck,
        )

    # 3b. Tinker auto-recommendation (covers base models tinker_cookbook knows).
    try:
        from tinker_cookbook import model_info  # type: ignore

        rec = model_info.get_recommended_renderer_name(name)
        if rec:
            adapter = _try_tinker_adapter(rec, tokenizer, model=name)
            if adapter is not None:
                return RendererResolution(adapter, "tinker", rec)
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
