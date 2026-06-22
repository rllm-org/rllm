"""Native rLLM renderer types and the :class:`Renderer` protocol.

rLLM speaks a single, token-level renderer interface for training (mirroring the
shape of the ``renderers`` / prime-rl package, which is the only backend with a
real cross-turn ``bridge_to_next_turn``). Concrete backends — the prime-rl
renderers and the tinker-cookbook / Fireworks renderers — are adapted to this
interface in :mod:`rllm.renderers._prime` and :mod:`rllm.renderers._tinker`.

These types are intentionally self-contained: importing this module does **not**
require either backend to be installed, so ``isinstance`` checks, type hints, and
the registry routing logic all work in a minimal environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rllm.tools.tool_base import ToolCall

__all__ = [
    "Message",
    "ToolCall",
    "ToolSpec",
    "RenderedTokens",
    "ParsedResponse",
    "Renderer",
]

# A chat message is a plain dict (OpenAI-style: role / content / tool_calls …).
# Both backends accept this shape, so we don't impose a stricter TypedDict here.
Message = dict[str, Any]


@dataclass
class ToolSpec:
    """A single callable tool, backend-agnostic.

    Mirrors both ``renderers.base.ToolSpec`` and
    ``tinker_cookbook.renderers.base.ToolSpec`` (identical fields), so it can be
    converted to either at the backend boundary.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderedTokens:
    """Result of rendering messages to token IDs.

    Field names match prime-rl's ``RenderedTokens`` so callers that read
    ``.token_ids`` (e.g. the model gateway's ``TokenAccumulator``) work against
    either the native or the prime-rl object.
    """

    token_ids: list[int]
    # Per-token attribution to its source message index (``-1`` = structural
    # scaffolding). ``None`` when the backend does not provide it.
    message_indices: list[int] | None = None
    multi_modal_data: Any | None = None


@dataclass
class ParsedResponse:
    """Structured assistant turn parsed from sampled token IDs."""

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class Renderer(Protocol):
    """Token-level renderer interface used across rLLM training.

    The contract is prime-rl's: render messages → token IDs, parse completion
    IDs → a structured turn, and (for RL multi-turn) extend a prior turn without
    re-rendering history. ``bridge_to_next_turn`` returning ``None`` is always a
    valid answer — it tells the caller "I can't prove the prefix-extension
    contract, fall back to a full re-render."
    """

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens: ...

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]: ...

    def parse_response(self, token_ids: list[int]) -> ParsedResponse: ...

    def get_stop_token_ids(self) -> list[int]: ...

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | list[dict[str, Any]] | None = None,
    ) -> RenderedTokens | None: ...
