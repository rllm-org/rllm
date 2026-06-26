"""Single source of truth for a run's renderer settings.

Both the gateway (turn-1+ cumulative bridge) and the trainer rollout engine
(turn-0 render + completion parse) must resolve the *same* renderer, or the
cumulative prefix contract breaks. They read renderer config from one place:
``rllm.renderer.{family,name}``. ``renderer_settings`` extracts that (with
deprecated fallbacks) so every consumer resolves identically.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def renderer_settings(cfg: Any) -> tuple[str, str | None]:
    """Return ``(family, name)`` for the run's renderer.

    Canonical source: ``rllm.renderer.family`` / ``rllm.renderer.name``. Falls
    back (deprecated, warns once-ish) to the old split keys
    ``rllm.gateway.renderer_family`` and ``rollout_engine.renderer_name`` so
    existing configs keep working. ``name`` takes precedence over ``family`` in
    ``resolve()`` (a pinned tinker/cookbook renderer beats a prime family).
    """
    rllm = _get(cfg, "rllm", {}) or {}
    rend = _get(rllm, "renderer", {}) or {}
    family = _get(rend, "family")
    name = _get(rend, "name")

    if not family or family == "auto":
        legacy_family = _get(_get(rllm, "gateway", {}) or {}, "renderer_family")
        if legacy_family and legacy_family != "auto":
            logger.warning("rllm.gateway.renderer_family is deprecated; set rllm.renderer.family instead.")
            family = legacy_family
    if name is None:
        legacy_name = _get(_get(cfg, "rollout_engine", {}) or {}, "renderer_name")
        if legacy_name is not None:
            logger.warning("rollout_engine.renderer_name is deprecated; set rllm.renderer.name instead.")
            name = legacy_name

    return (family or "auto", name)
