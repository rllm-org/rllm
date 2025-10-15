"""Main entry point for verl patches.

This module applies all verl patches using the modular patch system.
Individual patches are organized in separate files by target module.
"""

from . import (
    patch_fsdp_vllm,
    patch_torch_functional,
    patch_vllm_async_server,
    patch_vllm_rollout_spmd,
)


def setup():
    """
    Setup all verl patches.

    Called by Ray's worker_process_setup_hook to apply patches before
    the main verl code runs.
    """
    patch_vllm_async_server.register()
    patch_vllm_rollout_spmd.register()
    patch_fsdp_vllm.register()
    patch_torch_functional.register()
