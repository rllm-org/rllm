"""Harbor integration utilities."""

from __future__ import annotations

import subprocess


def check_docker_available() -> bool:
    """Check if Docker is available and running.

    Returns:
        True if Docker daemon is reachable, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def is_harbor_source(catalog_entry: dict | None) -> bool:
    """Check if a catalog entry refers to a Harbor dataset."""
    if catalog_entry is None:
        return False
    return catalog_entry.get("source", "").startswith("harbor:")


def is_harbor_agent(agent_name: str | None) -> bool:
    """Check if an agent name refers to a Harbor agent."""
    if agent_name is None:
        return False
    return agent_name.startswith("harbor:")
