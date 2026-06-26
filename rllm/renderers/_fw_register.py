"""Activate the Fireworks cookbook renderers.

``fireworks-training-cookbook`` ships renderers (``deepseek_v4``, ``gemma4``,
``kimi_k27_code``, ``glm5``, ``minimax_m2``, ``nemotron``, plus disaggregate
re-registrations of ``qwen3`` / ``deepseekv3`` / ``nemotron3`` / ``gpt_oss``)
that register themselves with ``tinker_cookbook.renderers`` **only when the
``training.renderer`` package is imported**. Nothing in rLLM imported it, so
those renderers were installed but unreachable via ``get_renderer``.

The actual import is deferred to :func:`ensure_registered` (called from the
tinker backend when it builds a renderer), so merely importing ``rllm.renderers``
does not pull in the cookbook. ``FIREWORKS_AVAILABLE`` reports importability
cheaply via ``find_spec``.
"""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger(__name__)


def _detect() -> bool:
    # Probe the top-level ``training`` package only (no import of its submodules),
    # so the availability flag stays cheap. The real import + registration happens
    # in ``ensure_registered``.
    try:
        return importlib.util.find_spec("training") is not None
    except Exception as err:  # noqa: BLE001 - parent package import can fail
        logger.debug("Fireworks cookbook not importable: %s", err)
        return False


FIREWORKS_AVAILABLE: bool = _detect()

_registered: bool = False


def ensure_registered() -> bool:
    """Import ``training.renderer`` once so its renderers register with
    tinker_cookbook. Idempotent; returns whether registration succeeded."""
    global _registered
    if _registered:
        return True
    try:
        import training.renderer  # noqa: F401  (import side effect registers renderers)

        _registered = True
    except Exception as err:  # noqa: BLE001
        logger.debug("Fireworks cookbook renderers unavailable: %s", err)
    return _registered
