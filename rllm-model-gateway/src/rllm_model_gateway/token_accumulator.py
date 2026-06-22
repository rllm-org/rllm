"""Per-session cumulative token state for drift-free multi-turn RL training.

Prevents tokenization drift (decode→re-encode producing different token IDs)
by maintaining raw token ID sequences across turns and using /v1/completions
with pre-tokenized prompts instead of re-tokenizing from text each turn.

The cross-turn "bridge" (end-of-assistant-turn + new user/tool messages +
start-of-next-generation) is delegated to the ``renderers`` package
(https://github.com/PrimeIntellect-ai/renderers), whose per-model
``bridge_to_next_turn`` guarantees the returned sequence starts byte-for-byte
with ``previous_prompt + previous_completion`` — exactly the prefix-extension
invariant the training-side merger relies on.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _messages_fingerprint(messages: list[dict[str, Any]]) -> str:
    """Compute a stable fingerprint for a message list prefix.

    Uses JSON serialization + SHA-256 to detect any content/role changes.
    """
    raw = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def extract_new_messages(messages: list[dict[str, Any]], prev_message_count: int) -> list[dict[str, Any]]:
    """Return the messages added since the last processed turn, minus assistants.

    The agent sends a cumulative message list. We already processed
    ``prev_message_count`` messages. The genuinely new messages are
    ``messages[prev_message_count:]`` — but the first of these is typically the
    assistant turn the model just sampled, which is already captured as
    completion token IDs. ``bridge_to_next_turn`` refuses assistant content
    (re-tokenizing sampled tokens would corrupt training), so we drop every
    assistant message from the slice.

    Returns an empty list if there are no new non-assistant messages.
    """
    if len(messages) <= prev_message_count:
        return []
    new = messages[prev_message_count:]
    return [m for m in new if m.get("role") != "assistant"]


class TokenAccumulator:
    """Maintains per-session cumulative token state for drift-free generation.

    Tracks the exact token IDs of the most recent turn so the next turn's
    prompt can be built by ``renderers.bridge_to_next_turn`` (concatenation +
    new-message rendering) rather than re-tokenization from text.

    Detects non-cumulative message list changes (prefix divergence) and
    automatically resets, so the training-side merger sees a segment break.
    """

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer
        self.prev_prompt_ids: list[int] = []
        self.prev_completion_ids: list[int] = []
        self.turn_count: int = 0
        self.message_count: int = 0
        self._prefix_fingerprint: str = ""
        # Per-message fingerprints of the last verified prefix, for diagnosing
        # exactly where a non-cumulative request diverged (see divergence).
        self._prefix_msg_fingerprints: list[str] = []

    @property
    def cumulative_ids(self) -> list[int]:
        """Full token sequence through the most recent completion.

        After turn N this is ``prev_prompt_ids + prev_completion_ids`` — the
        prompt the last turn was sampled from plus the tokens it produced.
        """
        return self.prev_prompt_ids + self.prev_completion_ids

    def should_rewrite(self) -> bool:
        """Return True if this session should use /v1/completions rewriting."""
        return self.turn_count > 0

    def is_cumulative(self, messages: list[dict[str, Any]]) -> bool:
        """Check if *messages* is a cumulative extension of the tracked prefix.

        Returns True if the first `message_count` messages in the new list
        match the fingerprint of the messages we already processed.
        Returns False (and the caller should reset) if:
          - The new list has fewer messages than what we've processed.
          - The prefix (first `message_count` msgs) differs from what we saw.
        """
        if self.turn_count == 0:
            return True
        if len(messages) <= self.message_count:
            return False
        prefix = messages[: self.message_count]
        return _messages_fingerprint(prefix) == self._prefix_fingerprint

    def reset(self, reason: str = "") -> None:
        """Clear all accumulated state, restarting this session's history.

        ``reason`` (optional) is logged so the cause of a cumulative-mode break
        is visible — see the call sites in ``ReverseProxy.handle``.
        """
        logger.info(
            "Resetting TokenAccumulator (was at turn %d, %d messages)%s",
            self.turn_count,
            self.message_count,
            f" — {reason}" if reason else "",
        )
        self.prev_prompt_ids = []
        self.prev_completion_ids = []
        self.turn_count = 0
        self.message_count = 0
        self._prefix_fingerprint = ""
        self._prefix_msg_fingerprints = []

    def divergence(self, messages: list[dict[str, Any]]) -> tuple[str, int]:
        """Diagnose why ``messages`` is not a cumulative extension of the stored
        prefix. Returns ``(kind, index)``:

          - ``("changed", i)``: ``messages[i]`` differs from the stored prefix's
            message ``i`` — the prefix was *mutated*, not just appended to
            (e.g. an edited/reformatted earlier message).
          - ``("shrunk", n)``: the overlap matches but ``messages`` has only ``n``
            messages (fewer than the verified prefix) — the conversation got
            *shorter*, i.e. summarization / history unwind.
          - ``("duplicate", n)``: ``messages`` is byte-identical to the verified
            prefix (same length, every message matches) — the agent re-sent an
            already-processed request, i.e. a retried / replayed sampling call.
          - ``("clean", -1)``: no divergence found.

        Only called on a reset (after ``is_cumulative`` already returned False),
        so the per-message rehash here is off the hot path.
        """
        prior = self._prefix_msg_fingerprints
        overlap = min(len(messages), len(prior))
        for i in range(overlap):
            if _messages_fingerprint([messages[i]]) != prior[i]:
                return ("changed", i)
        if len(messages) < len(prior):
            return ("shrunk", len(messages))
        if len(messages) == len(prior):
            return ("duplicate", len(messages))
        return ("clean", -1)

    def ingest_turn(self, prompt_token_ids: list[int], completion_token_ids: list[int]) -> None:
        """Record the token IDs from a completed turn.

        ``prompt_token_ids`` is the full prompt the turn was sampled from (turn
        1: the chat-rendered prompt; later turns: the bridge output we built).
        ``completion_token_ids`` is what the model produced. Together they form
        ``previous_prompt`` / ``previous_completion`` for the next bridge call.
        """
        self.prev_prompt_ids = list(prompt_token_ids)
        self.prev_completion_ids = list(completion_token_ids)
        self.turn_count += 1

    def update_prefix(self, messages: list[dict[str, Any]]) -> None:
        """Snapshot the current message list as the verified prefix."""
        self.message_count = len(messages)
        self._prefix_fingerprint = _messages_fingerprint(messages)
        self._prefix_msg_fingerprints = [_messages_fingerprint([m]) for m in messages]

    def build_next_prompt(
        self,
        new_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int] | None:
        """Construct the full prompt token IDs for the next turn.

        Delegates to ``renderers.bridge_to_next_turn``, which extends the prior
        ``prev_prompt_ids + prev_completion_ids`` with the rendered new messages
        plus the next generation prompt. Returns the full token-ID sequence, or
        ``None`` if the renderer cannot prove the prefix-extension contract (in
        which case the caller falls back to the normal chat path).
        """
        rendered = self.renderer.bridge_to_next_turn(
            self.prev_prompt_ids,
            self.prev_completion_ids,
            new_messages,
            tools=tools,
        )
        if rendered is None:
            return None
        return list(rendered.token_ids)
