#!/usr/bin/env python3
"""Runtime bootstrap, config loading, and environment creation helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from patches import (
    apply_jinja_tojson_patch,
    apply_swerex_modal_compat_patch,
    apply_swerex_modal_minimal_patch,
    apply_swerex_remote_retry_patch,
)

_BASE_DIR = Path(__file__).resolve().parent
_SWE_BENCH_PRO_PATH = _BASE_DIR / "SWE-bench_Pro-os"
_CONFIG_PATH = _BASE_DIR / "swebench_pro.yaml"

DEFAULT_COMMAND_TIMEOUT = 120
DEFAULT_SANDBOX_TIMEOUT = 3600

_bootstrapped = False
_cached_runtime_configs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None


def ensure_bootstrapped() -> None:
    """Apply runtime bootstrap side effects exactly once."""
    global _bootstrapped
    if _bootstrapped:
        return

    if str(_SWE_BENCH_PRO_PATH) not in sys.path:
        sys.path.insert(0, str(_SWE_BENCH_PRO_PATH))

    # Apply SWE-ReX patches:
    # - Minimal patch: GCR support, /bin/bash fix
    # - Compat patch: mini-swe-agent v2 protocol compatibility
    # - Remote retry: retry logic for flaky Modal remotes
    # - Jinja patch: fix tojson Unicode escaping for LLM consumption
    apply_swerex_modal_minimal_patch()
    apply_swerex_modal_compat_patch()
    apply_swerex_remote_retry_patch()
    apply_jinja_tojson_patch()

    _bootstrapped = True


def load_runtime_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load cached mini-swe-agent runtime configs from local YAML."""
    ensure_bootstrapped()
    global _cached_runtime_configs
    if _cached_runtime_configs is None:
        with open(_CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        _cached_runtime_configs = (
            dict(config.get("agent", {})),
            dict(config.get("model", {})),
            dict(config.get("environment", {})),
        )

    # Return shallow copies so callers can safely mutate top-level keys.
    return (
        dict(_cached_runtime_configs[0]),
        dict(_cached_runtime_configs[1]),
        dict(_cached_runtime_configs[2]),
    )


def default_scripts_dir() -> str:
    """Default run_scripts directory for SWE-bench Pro grading."""
    return str(_SWE_BENCH_PRO_PATH / "run_scripts")


def create_env(
    task: dict,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
):
    """Create a Modal environment from runtime config for a task."""
    ensure_bootstrapped()
    from minisweagent.environments import get_environment

    _, _, environment_config = load_runtime_configs()
    env_config = dict(environment_config)
    env_config["environment_class"] = env_config.get("environment_class", "swerex_modal")
    env_config["image"] = task["docker_image"]
    env_config["cwd"] = task["working_dir"]
    env_config["timeout"] = command_timeout
    env_config["startup_timeout"] = float(sandbox_timeout)
    env_config["runtime_timeout"] = float(sandbox_timeout)
    env_config["deployment_timeout"] = float(sandbox_timeout)
    return get_environment(env_config)
