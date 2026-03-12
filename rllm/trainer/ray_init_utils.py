"""Utilities for initializing Ray consistently.

Issue #166 reports that in some environments (notably Docker), a child process may
call `ray.init(namespace=...)` and accidentally start a fresh local Ray cluster
instead of attaching to the already-running one. This can lead to confusing
failures where named actors appear to be missing.

This module centralizes the logic for selecting Ray init parameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _ray_current_cluster_path() -> Path:
    # Default location Ray uses to store the current cluster address.
    # See Ray docs and common troubleshooting guides.
    return Path("/tmp/ray/ray_current_cluster")


def should_attach_to_existing_ray_cluster() -> bool:
    """Whether we should attempt to attach to an existing Ray cluster."""

    # Explicitly configured by the user or environment.
    if os.getenv("RAY_ADDRESS"):
        return True

    # Heuristic: if a Ray head has been started on this filesystem namespace,
    # Ray writes the address here.
    try:
        return _ray_current_cluster_path().exists()
    except Exception:
        return False


def get_ray_init_settings(config: Any | None = None) -> dict[str, Any]:
    """Build kwargs for `ray.init(...)` from config + environment.

    Notes:
    - If `config.ray_init.address` is set, we pass it through verbatim.
    - Otherwise, if we detect a running cluster (or RAY_ADDRESS is set), we use
      `address="auto"` to attach.
    - If none of the above applies, we return no `address`, so Ray will start a
      local cluster.
    """

    settings: dict[str, Any] = {}

    if config is not None and hasattr(config, "ray_init"):
        for k, v in config.ray_init.items():
            if v is not None:
                settings[k] = v

    # Prefer explicit address from config.
    if "address" in settings:
        return settings

    if should_attach_to_existing_ray_cluster():
        settings["address"] = "auto"

    return settings
