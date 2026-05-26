"""Per-session cumulative token state for drift-free multi-turn RL training.

Prevents tokenization drift (decode→re-encode producing different token IDs)
by maintaining raw token ID sequences across turns and using /v1/completions
with pre-tokenized prompts instead of re-tokenizing from text each turn.
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


def compute_bridge(tokenizer: Any, user_content: str) -> list[int]:
    """Compute bridge token IDs between an assistant completion and next generation.

    Uses the tokenizer's own apply_chat_template to determine the exact tokens
    for: end-of-assistant-turn + user-turn-with-content + start-of-next-generation.

    This is template-agnostic: works for Qwen, Llama-3, Mistral, etc. as long as
    the template uses special tokens at message boundaries (preventing BPE merges).

    Args:
        tokenizer: A HuggingFace tokenizer with apply_chat_template support.
        user_content: The new user message content for this turn.

    Returns:
        List of token IDs representing the bridge between turns.
    """
    prefix_msgs = [
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": "Y"},
    ]
    full_msgs = prefix_msgs + [{"role": "user", "content": user_content}]

    prefix_ids = tokenizer.apply_chat_template(prefix_msgs, tokenize=True, continue_final_message=True)
    full_ids = tokenizer.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=True)

    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "Chat template prefix mismatch — template may not use special tokens at message boundaries. Cumulative token mode requires templates where BPE merges cannot cross message boundaries."
        )

    return full_ids[len(prefix_ids) :]


def extract_new_user_content(messages: list[dict[str, Any]], prev_message_count: int) -> str | None:
    """Extract the content of the new user message added since last turn.

    The agent sends a cumulative message list. We already processed
    `prev_message_count` messages. The new messages should be:
    [assistant_response, new_user_message] — we return the user content.

    Returns None if no new user message is found.
    """
    if len(messages) <= prev_message_count:
        return None

    new_messages = messages[prev_message_count:]
    for msg in reversed(new_messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


class TokenAccumulator:
    """Maintains per-session cumulative token state for drift-free generation.

    Tracks the exact token IDs across turns so that the next turn's prompt
    can be constructed by concatenation rather than re-tokenization from text.

    Detects non-cumulative message list changes (prefix divergence) and
    automatically resets, so the training-side merger sees a segment break.
    """

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.cumulative_ids: list[int] = []
        self.turn_count: int = 0
        self.message_count: int = 0
        self._prefix_fingerprint: str = ""

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

    def reset(self) -> None:
        """Clear all accumulated state, restarting this session's history."""
        logger.info(
            "Resetting TokenAccumulator (was at turn %d, %d messages)",
            self.turn_count,
            self.message_count,
        )
        self.cumulative_ids = []
        self.turn_count = 0
        self.message_count = 0
        self._prefix_fingerprint = ""

    def ingest_turn(self, prompt_token_ids: list[int], completion_token_ids: list[int]) -> None:
        """Record the token IDs from a completed turn.

        On the first turn, sets cumulative_ids = prompt + completion.
        On subsequent turns, appends the completion (prompt was already built by us).
        """
        if self.turn_count == 0:
            self.cumulative_ids = list(prompt_token_ids) + list(completion_token_ids)
        else:
            self.cumulative_ids.extend(completion_token_ids)
        self.turn_count += 1

    def update_prefix(self, messages: list[dict[str, Any]]) -> None:
        """Snapshot the current message list as the verified prefix."""
        self.message_count = len(messages)
        self._prefix_fingerprint = _messages_fingerprint(messages)

    def build_next_prompt(self, user_content: str) -> list[int]:
        """Construct the full prompt token IDs for the next turn.

        Appends bridge tokens (end-of-turn + user message + generation prompt)
        to the cumulative sequence.

        If the last completion token is the same as the bridge's first token
        (typically <|im_end|>), the bridge is trimmed to avoid duplication.
        This handles the common case where the model generates the end-of-turn
        token as its natural stop.
        """
        bridge = compute_bridge(self.tokenizer, user_content)
        if self.cumulative_ids and bridge and self.cumulative_ids[-1] == bridge[0]:
            bridge = bridge[1:]
        prompt_ids = self.cumulative_ids + bridge
        self.cumulative_ids = list(prompt_ids)
        return prompt_ids
