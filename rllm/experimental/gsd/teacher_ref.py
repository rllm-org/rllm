"""Shared holder for the frozen reference teacher sampling client.

The GSD algorithm uses a *frozen* teacher so its logprobs are stable across
training steps — unlike the live training client, which Tinker refreshes
via ``save_checkpoint_and_get_sampling_client`` after every ``optim_step``.

The trick: when the Tinker backend calls
``rollout_engine.set_sampling_client(client)`` for the first time (inside
``initialize_async``, before any ``optim_step``), that client corresponds to
the initial weights.  A reference to that object stays valid forever — the
backend later replaces ``rollout_engine.sampling_client`` with a *new*
object, but the old one continues to point at the initial weights (Tinker's
documented stale-client behaviour).

:class:`FrozenTeacherRef` captures this initial client on first access and
then hands it out to every workflow that asks.  It is shared across all
workflow instances via ``workflow_args`` (same pattern as ``HintPool``).
"""

from __future__ import annotations

import logging

import tinker

from rllm.experimental.rollout.rollout_engine import RolloutEngine

logger = logging.getLogger(__name__)


class FrozenTeacherRef:
    """Pin the initial sampling client and expose it to all workflow instances.

    Usage::

        # In the entry point:
        teacher_ref = FrozenTeacherRef()
        trainer = AgentTrainer(
            workflow_args={"teacher_ref": teacher_ref, ...},
            ...,
        )

        # In the workflow:
        class GsdWorkflow(Workflow):
            def __init__(self, ..., teacher_ref: FrozenTeacherRef | None = None, **kwargs):
                self.teacher_ref = teacher_ref

            async def _do_training(self, task, uid):
                teacher_client = self.teacher_ref.capture(self.rollout_engine)
                ...

    The first workflow to call :meth:`capture` installs the reference; all
    subsequent calls are no-ops and return the same object.  Races are safe:
    every workflow reads ``rollout_engine.sampling_client`` while it still
    points at the initial client, so concurrent ``capture()`` calls install
    the same value.
    """

    def __init__(self) -> None:
        self._client: tinker.SamplingClient | None = None

    def capture(self, rollout_engine: RolloutEngine) -> tinker.SamplingClient:
        """Capture the current sampling client if none is pinned yet.

        Returns the pinned client (either freshly captured or previously
        captured).  Raises :class:`RuntimeError` if the engine doesn't have
        a sampling client yet — which should only happen if a workflow
        runs before the Tinker backend has called
        ``rollout_engine.set_sampling_client``.
        """
        if self._client is None:
            live = getattr(rollout_engine, "sampling_client", None)
            if live is None:
                raise RuntimeError("FrozenTeacherRef.capture() called before the rollout engine has a sampling client.  This should never happen in a normal training run.")
            self._client = live
            logger.info("[FrozenTeacherRef] pinned initial sampling client as frozen teacher")
        return self._client

    @property
    def client(self) -> tinker.SamplingClient:
        """The pinned client.  Raises if :meth:`capture` has not been called."""
        if self._client is None:
            raise RuntimeError("FrozenTeacherRef.client accessed before capture() — call capture(rollout_engine) first (typically at the top of a workflow method).")
        return self._client

    @property
    def is_captured(self) -> bool:
        return self._client is not None


__all__ = ["FrozenTeacherRef"]
