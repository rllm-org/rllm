"""GSD Countdown workflow — pool-based hint selection instead of per-problem generation.

Subclasses :class:`GsdWorkflow` to replace the LLM hint generation phase
with selection from a :class:`HintPool` of generic strategies.  The
distillation, gating, and scoring logic is inherited unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.experimental.gsd.hint_pool import HintPool
from rllm.experimental.gsd.workflow import GsdConfig, GsdWorkflow, RewardFn, _gather_with_progress
from rllm.experimental.rollout.rollout_engine import RolloutEngine

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class GsdCountdownWorkflow(GsdWorkflow):
    """GSD workflow for countdown with pool-based hint selection.

    Key differences from the base :class:`GsdWorkflow`:

    * **No per-problem hint generation** — a hint is selected from the
      :class:`HintPool` via UCB1 (zero LLM cost per task).
    * After gating, the hint's score is updated with the teacher-student
      improvement signal.
    * Challenging examples (student failed, teacher may have succeeded)
      are tracked for periodic hint evolution.
    * ``_do_validation`` uses the current best hint from the pool.
    """

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
        # Skip experience_store — countdown uses hint_pool instead
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

    # ------------------------------------------------------------------
    # Override prompt builders for countdown format
    # ------------------------------------------------------------------

    def _make_student_prompt(self, question: str) -> list[dict]:
        from rllm.experimental.test_examples.gsd.countdown_utils import build_countdown_student_prompt

        return build_countdown_student_prompt(question)

    def _make_teacher_prompt(self, question: str, hint: str) -> list[dict]:
        from rllm.experimental.test_examples.gsd.countdown_utils import build_countdown_teacher_prompt

        return build_countdown_teacher_prompt(question, hint)

    # ------------------------------------------------------------------
    # Override: training with pool-based hint selection
    # ------------------------------------------------------------------

    async def _do_training(self, task: dict, uid: str) -> Episode:
        question = task["question"]
        N = self.cfg.N

        # ---- Phase 1: Select hint from pool (no LLM call!) -----------
        selected_hint = self.hint_pool.select()
        hint_text = selected_hint.text
        logger.info(f"[{uid}] selected hint (score={selected_hint.score:.3f}, uses={selected_hint.n_uses}): {hint_text[:200]}")

        # ---- Phase 2: Dual rollouts ---------------------------------
        student_messages = self._make_student_prompt(question)
        teacher_messages = self._make_teacher_prompt(question, hint_text)

        coros: list[Awaitable[tuple[Step, float]]] = []
        coros += [self._do_rollout(task, student_messages) for _ in range(N)]
        coros += [self._do_rollout(task, teacher_messages) for _ in range(N)]
        results = await _gather_with_progress(coros, desc=f"[train:{uid}] rollouts")

        student_results = results[:N]
        teacher_results = results[N:]

        # ---- Phase 3: Gating ----------------------------------------
        student_rewards = [r for _, r in student_results]
        teacher_rewards = [r for _, r in teacher_results]
        R_S_avg = sum(student_rewards) / len(student_rewards)
        R_T_avg = sum(teacher_rewards) / len(teacher_rewards)
        teacher_valid = R_T_avg > R_S_avg

        trajectories: list[Trajectory] = []
        n_correct_teacher = sum(1 for r in teacher_rewards if r >= self.cfg.success_reward_threshold)
        n_correct_student = sum(1 for r in student_rewards if r >= self.cfg.success_reward_threshold)

        metrics: dict[str, Any] = {
            "R_S_avg": R_S_avg,
            "R_T_avg": R_T_avg,
            "teacher_valid": float(teacher_valid),
            "hint_improvement": R_T_avg - R_S_avg,
            "pool_size": self.hint_pool.size,
            "hint_score": selected_hint.score,
        }

        # ---- Phase 4: Loss routing (reuse parent's methods) ----------
        if teacher_valid:
            logger.info(f"[{uid}] Case 1 (distill): R_S={R_S_avg:.2f} R_T={R_T_avg:.2f} student={n_correct_student}/{N} teacher={n_correct_teacher}/{N}")
            await self._case1_distillation(uid, question, hint_text, student_results, teacher_results, trajectories)
        else:
            logger.info(f"[{uid}] Case 2 (GRPO fallback): R_S={R_S_avg:.2f} R_T={R_T_avg:.2f} student={n_correct_student}/{N} teacher={n_correct_teacher}/{N}")
            self._case2_grpo_fallback(student_results, trajectories)

        # ---- Phase 5: Update hint pool + collect hard solves ----------
        self.hint_pool.update(selected_hint, R_T_avg - R_S_avg)

        # Collect "hard solves": tasks with low student pass rate where at
        # least one rollout (student or teacher) found a correct answer.
        # These rare successes on hard problems are the most informative
        # signal for hint evolution.  Stored in the HintPool (shared across
        # all workflow instances).
        student_pass_rate = n_correct_student / N
        if student_pass_rate < self.cfg.hard_solve_threshold:
            correct_response = None
            for step, reward in student_results:
                if reward >= self.cfg.success_reward_threshold:
                    correct_response = step.model_response
                    break
            if correct_response is None:
                for step, reward in teacher_results:
                    if reward >= self.cfg.success_reward_threshold:
                        correct_response = step.model_response
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

        # Periodic evolution (uses _total_uses as the global step counter)
        if self.hint_pool.should_evolve():
            logger.info(f"[{uid}] Evolving hint pool (total_uses={self.hint_pool._total_uses}, hard_solves={self.hint_pool.num_hard_solves})")
            await self.hint_pool.evolve()

        is_correct = max(student_rewards + teacher_rewards) >= self.cfg.success_reward_threshold
        return self._build_episode(uid, task, trajectories, metrics, is_correct)

    # ------------------------------------------------------------------
    # Override: validation uses the best hint from the pool
    # ------------------------------------------------------------------

    async def _do_validation(self, task: dict, uid: str) -> Episode:
        question = task["question"]
        N = self.cfg.N_val

        # Use the best hint for validation
        best_hints = self.hint_pool.get_best(n=1)
        hint_text = best_hints[0].text if best_hints else ""

        student_messages = self._make_student_prompt(question)
        teacher_messages = self._make_teacher_prompt(question, hint_text)

        coros = [self._do_rollout(task, student_messages) for _ in range(N)]
        coros += [self._do_rollout(task, teacher_messages) for _ in range(N)]
        results = await _gather_with_progress(coros, desc=f"[val:{uid}] rollouts")

        student_results = results[:N]
        teacher_results = results[N:]

        student_rewards = [r for _, r in student_results]
        teacher_rewards = [r for _, r in teacher_results]
        threshold = self.cfg.success_reward_threshold

        student_correct = [r >= threshold for r in student_rewards]
        teacher_correct = [r >= threshold for r in teacher_rewards]

        metrics: dict[str, Any] = {
            "val/student_pass@1": sum(student_correct) / N,
            "val/teacher_pass@1": sum(teacher_correct) / N,
            "val/hint_improvement": sum(teacher_correct) / N - sum(student_correct) / N,
            "val/student_any_correct": float(any(student_correct)),
            "val/teacher_any_correct": float(any(teacher_correct)),
            "val/student_avg_reward": sum(student_rewards) / N,
            "val/teacher_avg_reward": sum(teacher_rewards) / N,
        }

        logger.info(f"[{uid}] val: student_pass@1={sum(student_correct)}/{N} teacher_pass@1={sum(teacher_correct)}/{N} hint_improvement={sum(teacher_correct) / N - sum(student_correct) / N:+.2f}")

        step, reward = student_results[0]
        traj = Trajectory(name="gsd_student", steps=[step], reward=reward)
        return self._build_episode(uid, task, [traj], metrics, any(student_correct))
