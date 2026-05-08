#!/usr/bin/env python3
"""SWE utilities — env cleanup and termination classification."""

from __future__ import annotations

from rllm.workflows.workflow import TerminationReason


def close_env(env, log_fn=None) -> None:
    """Stop a SWE-ReX Modal sandbox environment."""
    try:
        env.stop()
    except Exception as exc:
        if log_fn:
            log_fn(f"Env cleanup error: {type(exc).__name__}: {exc}")


_MAX_PROMPT_DIRECT_MARKERS = (
    "max_prompt_length_exceeded",
    "contextwindowexceedederror",
)
_MAX_PROMPT_LIMIT_MARKERS = (
    "prompt length plus max_tokens exceeds",
    "maximum context length",
    "maximum model length",
    "maximum input length",
    "model's context length",
)


def is_context_length_error(message: str) -> bool:
    """Return True when an API error indicates prompt/context overflow."""
    msg = message.lower()
    markers = _MAX_PROMPT_DIRECT_MARKERS + _MAX_PROMPT_LIMIT_MARKERS
    return any(marker in msg for marker in markers)


def tool_response_user_message(content: str) -> dict:
    """Format feedback as a user-visible tool response block."""
    return {
        "role": "user",
        "content": f"<tool_response>\n{content}\n</tool_response>",
    }


def build_error_details(exit_status: str) -> dict | None:
    """Convert an agent exit_status string into a structured error payload.

    Returns None for non-error statuses.
    """
    if classify_termination(exit_status) != TerminationReason.ERROR:
        return None

    if exit_status.startswith("Error: "):
        remainder = exit_status[len("Error: "):].strip()
        error_type, sep, error_message = remainder.partition(": ")
        return {
            "error_type": error_type if sep else "error",
            "error_message": error_message if sep else remainder,
            "raw_error": exit_status,
        }

    return {
        "error_type": "error",
        "error_message": exit_status,
        "raw_error": exit_status,
    }


def classify_termination(exit_status: str) -> TerminationReason:
    """Determine termination reason from the recorded agent exit status."""
    if exit_status == "Submitted":
        return TerminationReason.ENV_DONE
    if exit_status == "MaxPromptLengthExceeded":
        return TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED
    if exit_status == "MaxResponseLengthExceeded":
        return TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
    if exit_status == "Timeout":
        return TerminationReason.TIMEOUT
    if exit_status == "MaxTurnsExceeded":
        return TerminationReason.MAX_TURNS_EXCEEDED
    return TerminationReason.ERROR
