"""Workflow-level utilities for On-Policy Self-Distillation (OPSD).

Provides a decorator ``opsd_postprocess`` that wraps a workflow's ``run`` method to
compute token-level reverse KL advantages using a teacher policy conditioned on
privileged information (e.g., ground-truth solutions).

Usage::

    class MyWorkflow(Workflow):
        def __init__(self, rollout_engine, **kwargs):
            super().__init__(rollout_engine, **kwargs)
            self.opsd_config = OPSDConfig(kl_penalty_coef=1.0, ...)

        @opsd_postprocess
        async def run(self, task, uid, **kwargs) -> Episode:
            ...
            step.info["teacher_messages"] = [...]  # per-step teacher messages
            ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rllm.experimental.opsd.advantage import calculate_reverse_kl_advantage

if TYPE_CHECKING:
    from rllm.agents.agent import Episode

logger = logging.getLogger(__name__)


@dataclass
class OPSDConfig:
    """Configuration for (on-policy) self-distillation."""

    kl_penalty_coef: float
    kl_discount_factor: float
    teacher_messages_key: str = "teacher_messages"
    teacher_policy_update_freq: int = -1  # -1 for always using the initial policy


@dataclass
class OPSDState:
    """Internal mutable state tracked by the OPSD decorator across ``run`` calls.

    This state is lazily attached to each workflow instance (as ``_opsd_state``) and
    is used to manage the teacher policy lifecycle.

    Attributes:
        teacher_sampling_client: The sampling client used for teacher logprob
            computation. Initialized to the first observed sampling client (i.e. the
            initial model) and optionally updated per ``teacher_policy_update_freq``.
        _last_client_id: Python ``id()`` of the last seen sampling client on the
            rollout engine. Used to detect batch boundaries — a new sampling client
            signals a new training batch.
        batch_count: Number of batch transitions observed, used for teacher update
            scheduling.
    """

    teacher_sampling_client: Any = None
    _last_client_id: int = -1
    batch_count: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _maybe_update_teacher_policy(
    state: OPSDState,
    opsd_config: OPSDConfig,
    current_client: Any,
) -> None:
    """Detect batch transitions and update the teacher sampling client if needed.

    Teacher policy update logic:
    - On the first call, the teacher is set to the current (initial) sampling client.
    - On subsequent batch transitions (detected via sampling-client identity change):
        - ``teacher_policy_update_freq == -1``: never update (always use initial policy).
        - ``teacher_policy_update_freq > 0``: update every *N* batches.
    """
    if id(current_client) == state._last_client_id:
        return  # still in the same batch — nothing to do

    state._last_client_id = id(current_client)

    if state.teacher_sampling_client is None:
        # First batch: capture the current client as the initial teacher policy.
        state.teacher_sampling_client = current_client
        logger.debug("OPSD: initialized teacher policy from initial sampling client")
    else:
        state.batch_count += 1
        freq = opsd_config.teacher_policy_update_freq
        if freq > 0 and state.batch_count % freq == 0:
            state.teacher_sampling_client = current_client
            logger.info(f"OPSD: updated teacher policy at batch {state.batch_count}")


async def _compute_opsd_advantages(
    episode: Episode,
    opsd_config: OPSDConfig,
    teacher_client: Any,
    parser: Any,
) -> None:
    """Compute reverse KL advantages for all eligible steps in *episode*.

    Iterates over trajectories and steps, computing token-level advantages for each
    step that has teacher messages stored in ``step.info[teacher_messages_key]``.
    """
    teacher_key = opsd_config.teacher_messages_key
    tasks: list = []

    for traj in episode.trajectories:
        for step in traj.steps:
            teacher_msgs = step.info.get(teacher_key)
            if teacher_msgs is not None:
                tasks.append(
                    calculate_reverse_kl_advantage(
                        step,
                        teacher_client,
                        parser,
                        opsd_config.kl_penalty_coef,
                        opsd_config.kl_discount_factor,
                        teacher_messages=teacher_msgs,
                    )
                )

    if tasks:
        await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------


def opsd_postprocess(run_method):
    """Decorator for ``Workflow.run`` that computes OPSD advantages after episode generation.

    The decorated workflow class must expose:

    - ``self.opsd_config`` (:class:`OPSDConfig`): OPSD hyperparameters and settings.
    - ``self.rollout_engine``: a rollout engine with ``sampling_client`` and
      ``chat_parser`` attributes (e.g. ``TinkerEngine``).

    Steps in the produced episode that contain teacher messages (keyed by
    ``opsd_config.teacher_messages_key``) in their ``info`` dict will have per-token
    advantages computed via reverse KL divergence against the teacher policy.

    The decorator also manages teacher policy state:

    - Initializes the teacher to the first observed sampling client (initial model).
    - Optionally updates the teacher per ``opsd_config.teacher_policy_update_freq``.
    - Skips advantage computation during validation
      (detected via ``rollout_engine.is_validation``).
    """

    @functools.wraps(run_method)
    async def wrapper(self, task, uid, **kwargs):
        episode: Episode | None = await run_method(self, task, uid, **kwargs)

        # Skip if no episode produced or during validation.
        if episode is None or self.rollout_engine.is_validation:
            return episode

        # Lazily initialise OPSD state on the workflow instance.
        if not hasattr(self, "_opsd_state"):
            if not hasattr(self, "opsd_config"):
                logger.warning("Workflow decorated with @opsd_postprocess has no self.opsd_config; falling back to default OPSDConfig.")
                self.opsd_config = OPSDConfig(
                    kl_penalty_coef=1.0,
                    kl_discount_factor=0.0,
                )
            assert hasattr(self.rollout_engine, "sampling_client"), "OPSD requires a rollout engine with a sampling_client attribute (e.g. TinkerEngine)."
            assert self.rollout_engine.chat_parser is not None, "OPSD requires a rollout engine with a non-None chat_parser."
            self._opsd_state = OPSDState()

        opsd_config: OPSDConfig = self.opsd_config
        state: OPSDState = self._opsd_state

        # Teacher policy tracking
        _maybe_update_teacher_policy(state, opsd_config, self.rollout_engine.sampling_client)

        # Compute advantages
        await _compute_opsd_advantages(
            episode,
            opsd_config,
            state.teacher_sampling_client,
            self.rollout_engine.chat_parser,
        )

        return episode

    return wrapper
