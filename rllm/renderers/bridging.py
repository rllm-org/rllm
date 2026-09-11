"""Generic ``bridge_to_next_turn`` for any deterministic renderer.

A cumulative-mode bridge keeps the sampled tokens of the prior turn verbatim
and appends only what the new (non-assistant) messages add. We synthesize that
delta from any renderer that can render messages -> token ids: render a tiny
synthetic ``[user, assistant]`` history both closed and extended-with-new, then
diff after the assistant's close token. The delta is content-independent, so
splicing it onto the real (verbatim) prior tokens reproduces a full re-render's
prefix. A runtime prefix-check is the safety net: on any mismatch we return
``None`` so the accumulator resets instead of corrupting training.
"""

from __future__ import annotations

from typing import Any

from .types import RenderedTokens, reject_assistant_in_extension, trim_to_turn_close

# Arbitrary closed turn; only the tokens *after* the assistant close are used.
_SYN_HISTORY: tuple[dict[str, Any], ...] = (
    {"role": "user", "content": "ping"},
    {"role": "assistant", "content": "pong"},
)


def _last_index_in(ids: list[int], wanted: set[int]) -> int:
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] in wanted:
            return i
    return -1


class BridgingRendererMixin:
    """Adds a synthesized ``bridge_to_next_turn`` to a renderer.

    Concrete subclasses must provide ``render_ids(messages, *, tools,
    add_generation_prompt)`` and set ``close_token_ids`` (turn-close tokens,
    e.g. ``<|im_end|>``) and ``synthesize_close`` (the close to append if a
    truncated prior turn lacks one).
    """

    close_token_ids: set[int] = set()
    synthesize_close: int | None = None

    def render_ids(self, messages, *, tools=None, add_generation_prompt: bool = False) -> list[int]:  # noqa: D102
        raise NotImplementedError

    def _render_delta(self, new_messages, *, tools=None) -> list[int] | None:
        """Tokens the new messages add after an assistant close, incl. the next
        generation prompt. Returns ``None`` if the close can't be located."""
        syn = list(_SYN_HISTORY)
        closed = self.render_ids(syn, tools=tools, add_generation_prompt=False)
        close_idx = _last_index_in(closed, self.close_token_ids)
        if close_idx < 0:
            return None
        base = closed[: close_idx + 1]
        extended = self.render_ids(syn + list(new_messages), tools=tools, add_generation_prompt=True)
        if extended[: len(base)] != base:
            return None
        return extended[len(base) :]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages,
        *,
        tools=None,
    ) -> RenderedTokens | None:
        if not previous_prompt_ids or not new_messages or reject_assistant_in_extension(new_messages):
            return None
        anchored = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            self.close_token_ids,
            synthesize_close=self.synthesize_close,
        )
        if anchored is None:
            return None
        delta = self._render_delta(new_messages, tools=tools)
        if delta is None:
            return None
        token_ids = list(anchored) + list(delta)
        prev = list(previous_prompt_ids) + list(previous_completion_ids)
        if token_ids[: len(prev)] != prev:  # contract guard
            return None
        return RenderedTokens(token_ids=token_ids)
