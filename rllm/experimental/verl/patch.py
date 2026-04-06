"""Monkey-patches for Verl and vLLM within the rLLM unified trainer.

All patches are applied lazily (on first call) and are idempotent — calling
them multiple times is safe.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_VERL_DYNAMIC_BATCH_PATCHED = False
_VLLM_SDK_PATCHED = False


# ---------------------------------------------------------------------------
# Verl dynamic batch: sync micro-batch counts across DP ranks
# ---------------------------------------------------------------------------


def patch_verl_dynamic_batch_sync() -> None:
    """Patch ``prepare_dynamic_batch`` to sync micro-batch counts across DP ranks.

    Fixes `verl#5750 <https://github.com/verl-project/verl/issues/5750>`_:
    when ``use_dynamic_bsz=True``, each DP rank independently calculates
    ``num_micro_batches`` based on its local sequence lengths.  Different
    ranks can end up with different counts, causing NCCL collective
    operations (AllGather/ReduceScatter in FSDP) to deadlock.

    The fix defaults ``dp_group`` to ``torch.distributed.group.WORLD`` so
    that ``prepare_dynamic_batch`` performs an ``all_reduce(MAX)`` across
    ranks, forcing every rank to iterate through the same number of
    micro-batches.  This is the same approach as verl PR #5591.
    """
    global _VERL_DYNAMIC_BATCH_PATCHED
    if _VERL_DYNAMIC_BATCH_PATCHED:
        return

    import verl.utils.seqlen_balancing as sbl

    _original_prepare = sbl.prepare_dynamic_batch

    def _patched_prepare(data, max_token_len, dp_group=None, **kwargs):
        if dp_group is None:
            import torch.distributed

            if torch.distributed.is_initialized():
                dp_group = torch.distributed.group.WORLD
        return _original_prepare(data, max_token_len, dp_group=dp_group, **kwargs)

    sbl.prepare_dynamic_batch = _patched_prepare

    # Also patch the already-imported reference in dp_actor so both
    # compute_log_prob and update_policy use the patched version.
    try:
        from verl.workers.actor import dp_actor

        dp_actor.prepare_dynamic_batch = _patched_prepare
    except (ImportError, AttributeError):
        pass  # dp_actor may not be importable outside GPU workers

    _VERL_DYNAMIC_BATCH_PATCHED = True
    logger.info("Patched prepare_dynamic_batch to sync micro-batch counts across DP ranks (verl#5750)")


# ---------------------------------------------------------------------------
# vLLM SDK instrumentation
# ---------------------------------------------------------------------------


def patch_vllm_for_sdk() -> None:
    """Patch vLLM replicas to add logprob/token-id instrumentation for SDK traces.

    Creates an ``InstrumentedvLLMHttpServer`` Ray actor that loads
    ``rllm/patches/vllm_instrumentation.py`` in an isolated module namespace
    and patches ``vLLMReplica.__init__`` to use it.
    """
    global _VLLM_SDK_PATCHED
    if _VLLM_SDK_PATCHED:
        return

    import ray
    from verl.workers.rollout.vllm_rollout.vllm_async_server import (
        vLLMHttpServer,
        vLLMReplica,
    )

    @ray.remote(num_cpus=1)
    class InstrumentedvLLMHttpServer(vLLMHttpServer):
        """vLLM HTTP server with automatic vLLM instrumentation in Ray worker."""

        def __init__(self, *args, **kwargs):
            import importlib.util
            import sys
            from pathlib import Path

            instrumentation_path = Path(__file__).parent.parent.parent / "patches" / "vllm_instrumentation.py"

            spec = importlib.util.spec_from_file_location("rllm_vllm_instrumentation_isolated", str(instrumentation_path))
            vllm_instrumentation = importlib.util.module_from_spec(spec)
            sys.modules["rllm_vllm_instrumentation_isolated"] = vllm_instrumentation
            spec.loader.exec_module(vllm_instrumentation)

            vllm_instrumentation.instrument_vllm(add_response_logprobs=True)
            super().__init__(*args, **kwargs)

    _original_init = vLLMReplica.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.server_class = InstrumentedvLLMHttpServer

    vLLMReplica.__init__ = _patched_init
    _VLLM_SDK_PATCHED = True
    logger.info("Patched vLLMReplica for SDK instrumentation")
