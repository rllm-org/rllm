"""Shared utilities for chat client implementations.

This module provides common utility functions used across all chat client variants.
Each client remains independent - this module only provides shared helpers to avoid
code duplication.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def merge_args(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Merge positional and keyword arguments into a single dict.

    Supports passing a single dict as the first positional argument,
    which will be merged with keyword arguments.

    Args:
        args: Positional arguments (at most one dict allowed)
        kwargs: Keyword arguments

    Returns:
        Merged dictionary of all arguments

    Raises:
        TypeError: If positional arguments are not supported format
    """
    if args:
        if len(args) == 1 and isinstance(args[0], Mapping):
            merged = dict(args[0])
            merged.update(kwargs)
            return merged
        raise TypeError("Positional arguments are not supported; pass keyword arguments.")
    return dict(kwargs)


def extract_completion_tokens(response_payload: Mapping[str, Any]) -> list[int] | None:
    """Extract completion token IDs from response payload.

    Looks for token IDs in various locations within the response:
    1. choice.output_token_ids (some providers)
    2. choice.logprobs.token_ids (vLLM and others)

    Args:
        response_payload: The response dict from the LLM API

    Returns:
        List of token IDs if found, None otherwise
    """
    choices = response_payload.get("choices") or []
    if not choices:
        return None
    choice0 = choices[0]
    output_ids = choice0.get("output_token_ids")
    if isinstance(output_ids, list):
        return [int(tok) for tok in output_ids]
    logprobs = choice0.get("logprobs")
    if isinstance(logprobs, Mapping):
        token_ids = logprobs.get("token_ids")
        if isinstance(token_ids, list):
            return [int(tok) for tok in token_ids]
    return None


def extract_usage_tokens(response_payload: Mapping[str, Any]) -> dict[str, int]:
    """Extract token usage statistics from response payload.

    Args:
        response_payload: The response dict from the LLM API

    Returns:
        Dict with prompt, completion, and total token counts
    """
    usage = response_payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt": prompt_tokens,
        "completion": completion_tokens,
        "total": total_tokens,
    }
