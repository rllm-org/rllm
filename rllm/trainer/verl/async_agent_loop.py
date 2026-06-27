"""Fully-async / separated-mode rollout classes.

rLLM used to hand-roll the fully-async + partial-rollout rollout machinery here,
subclassing verl's pre-0.8 experimental ``AsyncLLMServerManager`` / ``AgentLoopWorker``.
verl 0.8.0 removed those APIs and upstreamed the very same machinery (same class
names, same partial-rollout ``generate`` loop) into
``verl.experimental.fully_async_policy.fully_async_rollouter``, rebuilt on the new
``LLMServerClient`` / ``LLMServerManager`` architecture:

* ``FullyAsyncLLMServerClient`` — ``LLMServerClient`` whose ``generate`` resumes an
  aborted (preempted) rollout when ``async_training.partial_rollout`` is enabled.
  This replaces the old ``FullyAsyncLLMServerManager(AsyncLLMServerManager)`` and is
  the object handed to ``VerlEngine`` as ``server_manager``.
* ``FullyAsyncLLMServerManager`` — ``LLMServerManager`` that owns/launches the rollout
  replicas (hybrid + standalone). In 0.8.0 the *manager* owns the servers, not the
  ``AgentLoopManager``; mint the client via ``get_client(client_cls=FullyAsyncLLMServerClient)``.
* ``FullyAsyncAgentLoopManager`` — ``AgentLoopManager`` with ``generate_sequences_single``
  / ``_select_best_worker`` (the per-worker dispatch rLLM relies on).

The dedicated ``FullyAsyncAgentLoopWorker`` is gone: partial-rollout now lives in the
*client*, so the stock ``AgentLoopWorker`` (which takes the client) is sufficient.

We simply re-export the upstream implementations so the rest of rLLM keeps a stable
import path.
"""

from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncLLMServerClient,
    FullyAsyncLLMServerManager,
)

__all__ = [
    "FullyAsyncAgentLoopManager",
    "FullyAsyncLLMServerClient",
    "FullyAsyncLLMServerManager",
]
