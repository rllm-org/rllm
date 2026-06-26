"""Canonical renderer types for rLLM.

We adopt the prime-rl ``renderers`` package's protocol as the canonical
interface — the gateway's ``TokenAccumulator`` already depends on its
``bridge_to_next_turn`` / ``render_ids`` / ``RenderedTokens`` shape. This module
just re-exports those so the rest of rLLM imports from one place.
"""

from __future__ import annotations

from renderers import (  # type: ignore
    Message,
    ParsedResponse,
    RenderedTokens,
    Renderer,
    ToolSpec,
)
from renderers.base import (  # type: ignore
    reject_assistant_in_extension,
    trim_to_turn_close,
)

__all__ = [
    "Message",
    "ParsedResponse",
    "RenderedTokens",
    "Renderer",
    "ToolSpec",
    "reject_assistant_in_extension",
    "trim_to_turn_close",
]
