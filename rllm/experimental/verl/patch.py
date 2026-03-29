"""Monkey-patches for Verl and vLLM within the rLLM unified trainer.

All patches are applied lazily (on first call) and are idempotent — calling
them multiple times is safe.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_VERL_ACTOR_PATCHED = False
_VLLM_SDK_PATCHED = False


# ---------------------------------------------------------------------------
# Verl actor: per-call policy loss mode override
# ---------------------------------------------------------------------------


def patch_verl_actor_for_loss_override() -> None:
    """Patch ``DataParallelPPOActor.update_policy`` to support per-call loss mode.

    When ``data.meta_info`` contains ``"policy_loss_mode_override"``, the
    actor temporarily uses that loss mode instead of the one baked into
    ``self.config.policy_loss.loss_mode``.  The original config value is
    restored after the call (even on exception).
    """
    global _VERL_ACTOR_PATCHED
    if _VERL_ACTOR_PATCHED:
        return

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    _original_update_policy = DataParallelPPOActor.update_policy

    def _patched_update_policy(self, data):
        override = data.meta_info.get("policy_loss_mode_override")
        if override is not None:
            original = self.config.policy_loss.get("loss_mode", "vanilla")
            self.config.policy_loss["loss_mode"] = override
            try:
                return _original_update_policy(self, data)
            finally:
                self.config.policy_loss["loss_mode"] = original
        return _original_update_policy(self, data)

    DataParallelPPOActor.update_policy = _patched_update_policy
    _VERL_ACTOR_PATCHED = True
    logger.info("Patched DataParallelPPOActor.update_policy for per-call loss mode override")


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
