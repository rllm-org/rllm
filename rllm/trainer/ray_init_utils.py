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

FORWARD_PREFIXES = [
    "VLLM_",
    "SGL_",
    "SGLANG_",
    "HF_",
    "TOKENIZERS_",
    "DATASETS_",
    "TORCH_",
    "PYTORCH_",
    # Triton's compile cache must be a fresh tmpfs dir per run or concurrent
    # ranks race on it; TRITON_CACHE_DIR only helps if workers actually see it.
    "TRITON_",
    "DEEPSPEED_",
    "MEGATRON_",
    "NCCL_",
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "NV_",
    "NVIDIA_",
    "RLLM_",
]


def get_forwarded_env_vars():
    """
    Get the forwarded environment variables. The `RLLM_EXCLUDE` environment variable can be used to
    exclude specific environment variables or all variables with a specific prefix.

    Example:
    ```
    RLLM_EXCLUDE=VLLM*,CUDA*,NCCL_IB_DISABLE
    ```
    will exclude all variables with prefix `VLLM_`, `CUDA_`, and `NCCL_IB_DISABLE`.

    By default, all environment variables with prefix in `FORWARD_PREFIXES` are forwarded.
    """
    if os.environ.get("RLLM_EXCLUDE", None) is not None:
        rllm_exclude = str(os.environ.get("RLLM_EXCLUDE")).split(",")
    else:
        rllm_exclude = []

    forward_prefix = FORWARD_PREFIXES.copy()

    # RLLM_EXCLUDE is a control var read on the launching node; it matches the
    # RLLM_ prefix but must never be forwarded into workers.
    exclude_vars = {"RLLM_EXCLUDE"}
    for name in rllm_exclude:
        if "*" in name:  # denote a prefix match, e.g. "VLLM*"
            prefix = name.replace("*", "_")
            try:
                forward_prefix.remove(prefix)
            except ValueError:
                pass
        else:
            exclude_vars.add(name)

    forwarded = {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in forward_prefix) and k not in exclude_vars}
    return forwarded


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
