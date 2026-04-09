"""GSD Countdown workflow — pool-based hint selection instead of per-problem generation.

Subclasses :class:`GsdWorkflow` with three targeted overrides:

* :meth:`_make_student_prompt` / :meth:`_make_teacher_prompt` — countdown-specific
  prompt templates.
* :meth:`_get_hint` — select a hint from the shared :class:`HintPool` via UCB1
  instead of sampling from the live model.  Because the hint is pool-generated
  (not model-sampled), we return ``hint_step=None`` so there is no
  ``gsd_hint`` trajectory for this workflow — meta-RL on the hint generator
  does not apply when the hints don't come from the model.
* :meth:`_post_training_hook` — update the pool's EMA score for the selected
  hint, collect "hard solves", and periodically evolve the pool via LiteLLM.

All distillation / gating / IS / CE logic is inherited unchanged from the
base workflow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import tinker

from rllm.agents.agent import Step
from rllm.experimental.gsd.utils import HintPool
from rllm.experimental.gsd.workflow import GsdConfig, GsdWorkflow, RewardFn
from rllm.experimental.rollout.rollout_engine import RolloutEngine

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class GsdCountdownWorkflow(GsdWorkflow):
    """GSD workflow for countdown with pool-based hint selection."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor: ThreadPoolExecutor,
        *,
        reward_fn: RewardFn,
        gsd_config: GsdConfig | None = None,
        hint_pool: HintPool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rollout_engine=rollout_engine,
            executor=executor,
            reward_fn=reward_fn,
            gsd_config=gsd_config,
            **kwargs,
        )
        if hint_pool is None:
            raise ValueError("GsdCountdownWorkflow requires a hint_pool")
        self.hint_pool = hint_pool
        # UCB-selected hints are not model-sampled, so there is nothing
        # sensible to optimize via hint GRPO.  Force ``train_hint=False``
        # so the base workflow does not try to create a ``gsd_hint``
        # trajectory from a ``None`` step.
        if self.cfg.train_hint:
            logger.info("[GsdCountdownWorkflow] overriding cfg.train_hint=False — hint pool is not model-sampled")
            self.cfg.train_hint = False
        self._selected_hint = None  # set per-task in ``_get_hint``

    # ------------------------------------------------------------------
    # Prompt overrides
    # ------------------------------------------------------------------

    def _make_student_prompt(self, question: str) -> list[dict]:
        from rllm.experimental.test_examples.gsd.countdown.utils import (
            build_countdown_student_prompt,
        )

        return build_countdown_student_prompt(question)

    def _make_teacher_prompt(self, question: str, hint: str) -> list[dict]:
        from rllm.experimental.test_examples.gsd.countdown.utils import (
            build_countdown_teacher_prompt,
        )

        return build_countdown_teacher_prompt(question, hint)

    # ------------------------------------------------------------------
    # Hint override: select from the pool, no step for GRPO
    # ------------------------------------------------------------------

    async def _get_hint(
        self,
        question: str,
        uid: str,
        live_client: tinker.SamplingClient,
    ) -> tuple[str, Step | None]:
        if self.rollout_engine.is_validation:
            # Validation uses the best hint deterministically.
            best = self.hint_pool.get_best(n=1)
            hint_text = best[0].text if best else ""
            logger.info(f"[{uid}] val hint (best): {hint_text[:200]}")
            return hint_text, None

        selected = self.hint_pool.select()
        self._selected_hint = selected  # stashed for the post-training hook
        logger.info(f"[{uid}] selected hint (score={selected.score:.3f}, uses={selected.n_uses}): {selected.text[:200]}")
        return selected.text, None

    # ------------------------------------------------------------------
    # Post-training hook: pool update + hard solve collection + evolution
    # ------------------------------------------------------------------

    async def _post_training_hook(
        self,
        *,
        task: dict,
        uid: str,
        question: str,
        hint_text: str,
        student_steps: list[Step],
        teacher_steps: list[Step],
        R_S_avg: float,
        R_T_avg: float,
        teacher_valid: bool,
        metrics: dict[str, Any],
    ) -> None:
        selected = self._selected_hint
        if selected is None:
            return  # should only happen during validation

        self.hint_pool.update(selected, R_T_avg - R_S_avg)
        metrics["pool_size"] = float(self.hint_pool.size)
        metrics["hint_score"] = float(selected.score)

        # Collect "hard solves": tasks with low student pass rate where at
        # least one rollout found a correct answer.  These rare successes
        # on difficult problems are the most informative signal for hint
        # evolution and are stored in the shared HintPool.
        N = len(student_steps)
        threshold = self.cfg.success_reward_threshold
        n_correct_student = sum(1 for s in student_steps if s.reward >= threshold)
        student_pass_rate = n_correct_student / max(N, 1)

        if student_pass_rate < self.cfg.hard_solve_threshold:
            correct_response = None
            for s in student_steps:
                if s.reward >= threshold:
                    correct_response = s.model_response
                    break
            if correct_response is None:
                for t in teacher_steps:
                    if t.reward >= threshold:
                        correct_response = t.model_response
                        break
            if correct_response is not None:
                self.hint_pool.record_hard_solve(
                    {
                        "target": task.get("target"),
                        "nums": task.get("nums", []),
                        "response": correct_response,
                        "student_pass_rate": student_pass_rate,
                    }
                )

        # Periodic evolution (uses the pool's own _total_uses counter as
        # a global step counter shared across workflow instances).
        if self.hint_pool.should_evolve():
            logger.info(f"[{uid}] Evolving hint pool (total_uses={self.hint_pool._total_uses}, hard_solves={self.hint_pool.num_hard_solves})")
            await self.hint_pool.evolve()
