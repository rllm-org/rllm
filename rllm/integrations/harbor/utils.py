"""Harbor integration utilities."""

from __future__ import annotations

import platform
import subprocess


def diagnose_docker() -> tuple[bool, str, str]:
    """Check the local Docker setup and return ``(ok, reason, hint)``.

    On failure, ``reason`` is a one-line explanation suitable for the
    CLI (e.g. ``"Docker CLI not found on PATH"`` vs.
    ``"Docker daemon not running"``) and ``hint`` is a short
    actionable suggestion (platform-aware where useful). On success,
    both strings are empty.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        hint = "Install Docker Desktop from https://www.docker.com/products/docker-desktop/" if platform.system() == "Darwin" else "Install Docker from https://docs.docker.com/engine/install/"
        return (False, "Docker CLI not found on PATH", hint)
    except subprocess.TimeoutExpired:
        return (False, "Docker daemon timed out (10s)", "Try restarting Docker.")
    except OSError as e:
        return (False, f"Docker check failed: {e}", "")

    if result.returncode == 0:
        return (True, "", "")

    stderr = (result.stderr or "").strip()
    last_line = stderr.splitlines()[-1] if stderr else ""

    # Daemon-not-running is the common case — surface a platform-specific hint.
    daemon_down = "cannot connect to the docker daemon" in stderr.lower() or "is the docker daemon running" in stderr.lower()
    if daemon_down:
        if platform.system() == "Darwin":
            hint = "Start Docker Desktop (`open -a Docker`) and wait for the whale icon to settle."
        elif platform.system() == "Linux":
            hint = "Start the daemon with `sudo systemctl start docker` (or your init system's equivalent)."
        else:
            hint = "Start Docker and try again."
        return (False, "Docker daemon not running", hint)

    return (False, last_line or "Docker is not available", "")


def check_docker_available() -> bool:
    """Backward-compatible boolean wrapper around :func:`diagnose_docker`."""
    ok, _, _ = diagnose_docker()
    return ok


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
