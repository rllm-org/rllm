"""Timeout configuration for harness-issued sandbox housekeeping commands."""

import os


def sandbox_control_timeout_s() -> int:
    """Read the positive integer timeout in seconds (default: 10).

    This covers reward/outcome retrieval and short verifier setup commands,
    independently of agent commands and the full verifier execution budget.
    """
    name = "RLLM_SANDBOX_CONTROL_TIMEOUT_S"
    value = os.environ.get(name, "10")
    try:
        timeout = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if timeout <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return timeout
