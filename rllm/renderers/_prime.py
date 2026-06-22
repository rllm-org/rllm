"""prime-rl (``renderers``) backend, adapted to the native :class:`Renderer`.

prime-rl renderers already speak the native token-level contract (``render_ids`` /
``parse_response`` / ``get_stop_token_ids`` / ``bridge_to_next_turn``). This
wrapper only normalizes the boundary types — tools in (→ prime ``ToolSpec``) and
results out (→ native ``RenderedTokens`` / ``ParsedResponse`` with rLLM
``ToolCall``) — so a resolved renderer behaves identically regardless of backend.

This is the RL-preferred backend: it is the only one whose ``bridge_to_next_turn``
can return a real prefix-extension (drift-free multi-turn token forwarding).

Heavy imports are deferred: ``import rllm.renderers`` only checks that the
``renderers`` package is importable (via ``find_spec``); the package itself —
which pulls in ``transformers`` — is imported on first use.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from rllm.renderers._common import iter_tool_specs, normalize_tool_calls
from rllm.renderers.types import Message, ParsedResponse, RenderedTokens, ToolSpec

logger = logging.getLogger(__name__)

PRIME_AVAILABLE: bool = importlib.util.find_spec("renderers") is not None

_prime_module: Any | None = None
_model_renderer_map: dict[str, str] | None = None


def _prime():
    """Lazily import and cache the prime-rl ``renderers`` package."""
    global _prime_module
    if _prime_module is None:
        import renderers as _module  # type: ignore

        _prime_module = _module
    return _prime_module


def _model_map() -> dict[str, str]:
    """Lazily import and cache prime-rl's ``MODEL_RENDERER_MAP``."""
    global _model_renderer_map
    if _model_renderer_map is None:
        try:
            from renderers.base import MODEL_RENDERER_MAP  # type: ignore

            _model_renderer_map = dict(MODEL_RENDERER_MAP)
        except Exception as err:  # noqa: BLE001
            logger.debug("Could not load prime-rl MODEL_RENDERER_MAP: %s", err)
            _model_renderer_map = {}
    return _model_renderer_map


def prime_supports(model_name: str | None) -> bool:
    """True iff prime-rl has a hand-coded renderer for ``model_name``.

    Auto-detect is exact-match on the HF id (prefix matching is intentionally
    off upstream), so we check membership in ``MODEL_RENDERER_MAP`` rather than
    trusting ``create_renderer`` (which always returns a ``DefaultRenderer``
    fallback for unknown ids).
    """
    if not PRIME_AVAILABLE or not model_name:
        return False
    return model_name in _model_map()


def _to_prime_tools(tools: list[ToolSpec] | list[dict[str, Any]] | None):
    specs = iter_tool_specs(tools)
    if not specs:
        return None
    from renderers.base import ToolSpec as PrimeToolSpec  # type: ignore

    return [
        PrimeToolSpec(name=s.name, description=s.description, parameters=s.parameters)
        for s in specs
    ]


def _wrap_rendered(rt: Any) -> RenderedTokens:
    return RenderedTokens(
        token_ids=list(rt.token_ids),
        message_indices=getattr(rt, "message_indices", None),
        multi_modal_data=getattr(rt, "multi_modal_data", None),
    )


def _wrap_parsed(pr: Any) -> ParsedResponse:
    return ParsedResponse(
        content=getattr(pr, "content", "") or "",
        reasoning_content=getattr(pr, "reasoning_content", "") or "",
        tool_calls=normalize_tool_calls(getattr(pr, "tool_calls", None)),
    )


class PrimeRenderer:
    """Native-protocol wrapper over a prime-rl renderer instance."""

    backend = "prime"

    def __init__(self, renderer: Any):
        self._renderer = renderer
        self.is_default = type(renderer).__name__ == "DefaultRenderer"
        # DefaultRenderer's bridge always returns None; hand-coded renderers have
        # a real cross-turn bridge.
        self.has_bridge = not self.is_default

    @property
    def name(self) -> str:
        return type(self._renderer).__name__

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        return _wrap_rendered(
            self._renderer.render(
                messages,
                tools=_to_prime_tools(tools),
                add_generation_prompt=add_generation_prompt,
            )
        )

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return list(
            self._renderer.render_ids(
                messages,
                tools=_to_prime_tools(tools),
                add_generation_prompt=add_generation_prompt,
            )
        )

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        return _wrap_parsed(self._renderer.parse_response(token_ids))

    def get_stop_token_ids(self) -> list[int]:
        return list(self._renderer.get_stop_token_ids())

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | list[dict[str, Any]] | None = None,
    ) -> RenderedTokens | None:
        rendered = self._renderer.bridge_to_next_turn(
            previous_prompt_ids,
            previous_completion_ids,
            new_messages,
            tools=_to_prime_tools(tools),
        )
        return None if rendered is None else _wrap_rendered(rendered)


def make_prime_renderer(
    tokenizer: Any,
    *,
    renderer_family: str = "auto",
    preserve_all_thinking: bool = False,
    preserve_thinking_between_tool_calls: bool = False,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
) -> PrimeRenderer:
    """Build a :class:`PrimeRenderer` via prime-rl ``create_renderer``."""
    if not PRIME_AVAILABLE:
        raise RuntimeError(
            "prime-rl 'renderers' package is not installed. "
            "Install it with: pip install renderers"
        )
    renderer = _prime().create_renderer(
        tokenizer,
        renderer=renderer_family,
        tool_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        preserve_all_thinking=preserve_all_thinking,
        preserve_thinking_between_tool_calls=preserve_thinking_between_tool_calls,
    )
    return PrimeRenderer(renderer)
