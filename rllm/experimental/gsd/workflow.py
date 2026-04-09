"""GsdWorkflow — Generalized Self-Distillation workflow.

Implements the GSD training loop for a single task:

1. **Self-hinting** — generate a solution-independent strategic hint.
2. **Dual rollouts** — N student (no hint) + N teacher (with hint) rollouts.
3. **Gating** — compare average rewards; decide distillation vs GRPO fallback.
4. **Loss routing** —

   * **Case 1 (teacher valid):**
     * On-policy distillation (reverse KL via ``importance_sampling``):
       student sequences scored under teacher → per-token advantages.
     * Supervised distillation (forward KL via ``cross_entropy``):
       teacher's correct sequences with Top-K soft targets.
   * **Case 2 (fallback):**
     * Student trajectories → GRPO via ``importance_sampling``.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from tqdm import tqdm

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.experimental.gsd.losses import compute_sampled_rkl_advantages, score_teacher_for_response
from rllm.experimental.gsd.prompts import (
    build_hint_prompt,
    build_student_prompt,
    build_teacher_prompt,
    extract_hint,
)
from rllm.experimental.rollout.rollout_engine import RolloutEngine
from rllm.workflows.workflow import Workflow

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Type alias for the reward function: (task_dict, response_text) → float
RewardFn = Callable[[dict, str], float]


@dataclass
class GsdConfig:
    """Configuration for the GSD workflow."""

    N: int = 4
    N_val: int = 2
    distill_topk: int = 19  # Max 19: combined datum uses K+1 columns, server limit is 20
    train_hint: bool = False
    hint_sampling_params: dict[str, Any] = field(
        default_factory=lambda: {"temperature": 0.6, "top_p": 0.9, "max_tokens": 256},
    )
    kl_coeff: float = 1.0
    kl_clip_min: float = -5.0
    kl_clip_max: float = 5.0
    success_reward_threshold: float = 0.5
    max_context_length: int = 32768

    # Experience buffer retrieval
    retrieval_k: int = 3
    # Threshold for collecting "hard solves" — tasks with student pass rate
    # below this that still have at least one correct solution.
    hard_solve_threshold: float = 0.5

    # When True, Case 2 (teacher invalid) produces NO training trajectories.
    # All gradients come exclusively from the combined IS + CE distillation loss.
    # When False (default), Case 2 falls back to standard RL on student rollouts.
    distill_only: bool = False


class GsdWorkflow(Workflow):
    """Generalized Self-Distillation workflow for single-turn math tasks.

    Each :meth:`run` call processes one problem through the full GSD loop
    and returns an :class:`Episode` with named trajectories that the
    per-role loss router dispatches to the correct loss function.

    .. important::

       GSD manages its own rollout parallelism (``N`` student + ``N`` teacher).
       The Tinker ``training.group_size`` and ``validation.group_size`` should
       both be set to **1** in the config so the engine dispatches one task at
       a time to this workflow.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor: ThreadPoolExecutor,
        *,
        reward_fn: RewardFn,
        gsd_config: GsdConfig | None = None,
        experience_store: Any | None = None,
        scoring_accumulator: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.cfg = gsd_config or GsdConfig()
        self.reward_fn = reward_fn
        self.experience_store = experience_store  # Optional EmbeddingExperienceStore
        self.scoring_accumulator = scoring_accumulator  # Optional ScoringAccumulator

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, task: dict, uid: str, **kwargs: Any) -> Episode:
        is_validation = self.rollout_engine.is_validation

        if is_validation:
            return await self._do_validation(task, uid)
        return await self._do_training(task, uid)

    async def _get_hint_text(self, question: str, uid: str) -> str:
        experiences = None
        if self.experience_store is not None:
            experiences = await self.experience_store.query(question, top_k=self.cfg.retrieval_k)
            if experiences:
                logger.info(f"[{uid}] retrieved {len(experiences)} experiences from buffer (size={self.experience_store.size})")
        hint_messages = build_hint_prompt(question, experiences=experiences)
        hint_output = await self.rollout_engine.get_model_response(
            hint_messages,
            **self.cfg.hint_sampling_params,
        )
        hint_text = extract_hint(hint_output.text or hint_output.content or "")
        logger.info(f"[{uid}] hint generated ({len(hint_text)} chars): {hint_text[:500]}")
        return hint_text

    # ------------------------------------------------------------------
    # Validation: N_val student + N_val teacher, report pass@N metrics
    # ------------------------------------------------------------------

    async def _do_validation(self, task: dict, uid: str) -> Episode:
        """Run validation rollouts and compute pass@N metrics for both roles."""
        question = task["question"]
        N = self.cfg.N_val

        # Generate hint
        hint_text = await self._get_hint_text(question, uid)

        # Dual rollouts with progress
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

        # Return a single student trajectory for the trainer pipeline
        step, reward = student_results[0]
        traj = Trajectory(name="gsd_student", steps=[step], reward=reward)
        is_correct = any(student_correct)
        return self._build_episode(uid, task, [traj], metrics, is_correct)

    # ------------------------------------------------------------------
    # Training: hint → dual rollouts → gate → score → route
    # ------------------------------------------------------------------

    async def _do_training(self, task: dict, uid: str) -> Episode:
        question = task["question"]
        N = self.cfg.N

        # ---- Phase 1: Self-hinting --------------------------------
        hint_text = await self._get_hint_text(question, uid)

        # ---- Phase 2: Dual rollouts (concurrent with progress) ----
        student_messages = self._make_student_prompt(question)
        teacher_messages = self._make_teacher_prompt(question, hint_text)

        coros: list[Awaitable[tuple[Step, float]]] = []
        coros += [self._do_rollout(task, student_messages) for _ in range(N)]
        coros += [self._do_rollout(task, teacher_messages) for _ in range(N)]
        results = await _gather_with_progress(coros, desc=f"[train:{uid}] rollouts")

        student_results = results[:N]
        teacher_results = results[N:]

        # ---- Phase 3: Gating --------------------------------------
        student_rewards = [r for _, r in student_results]
        teacher_rewards = [r for _, r in teacher_results]
        R_S_avg = sum(student_rewards) / len(student_rewards)
        R_T_avg = sum(teacher_rewards) / len(teacher_rewards)

        # Teacher is valid when R_T_avg is greater than R_S_avg, or when they are equal but the R_T_avg is >= success_reward_threshold
        teacher_valid = R_T_avg > R_S_avg or (R_T_avg == R_S_avg and R_T_avg >= self.cfg.success_reward_threshold)

        trajectories: list[Trajectory] = []
        metrics: dict[str, Any] = {
            "R_S_avg": R_S_avg,
            "R_T_avg": R_T_avg,
            "teacher_valid": float(teacher_valid),
            "hint_improvement": R_T_avg - R_S_avg,
        }

        # ---- Phase 4: Loss routing --------------------------------
        n_correct_teacher = sum(1 for r in teacher_rewards if r >= self.cfg.success_reward_threshold)
        n_correct_student = sum(1 for r in student_rewards if r >= self.cfg.success_reward_threshold)

        if teacher_valid:
            logger.info(f"[{uid}] Case 1 (distill): R_S={R_S_avg:.2f} R_T={R_T_avg:.2f} student_correct={n_correct_student}/{N} teacher_correct={n_correct_teacher}/{N}")
            await self._case1_distillation(
                uid,
                question,
                hint_text,
                student_results,
                teacher_results,
                trajectories,
            )
        else:
            if self.cfg.distill_only:
                logger.info(f"[{uid}] Case 2 (skipped, distill_only=True): R_S={R_S_avg:.2f} R_T={R_T_avg:.2f}")
            else:
                logger.info(f"[{uid}] Case 2 (GRPO fallback): R_S={R_S_avg:.2f} R_T={R_T_avg:.2f} student_correct={n_correct_student}/{N} teacher_correct={n_correct_teacher}/{N}")
                self._case2_grpo_fallback(student_results, trajectories)

        # ---- Phase 5: Update experience buffer ----------------------
        if teacher_valid and self.experience_store is not None:
            await self.experience_store.add(
                text=question,
                metadata={
                    "hint": hint_text,
                    "summary": (f"Teacher outperformed student ({R_T_avg:.2f} vs {R_S_avg:.2f}). {n_correct_teacher}/{N} teacher rollouts correct."),
                    "improvement": R_T_avg - R_S_avg,
                },
            )

        is_correct = max(student_rewards + teacher_rewards) >= self.cfg.success_reward_threshold
        return self._build_episode(uid, task, trajectories, metrics, is_correct)

    # ------------------------------------------------------------------
    # Shared rollout helper
    # ------------------------------------------------------------------

    async def _do_rollout(self, task: dict, messages: list[dict]) -> tuple[Step, float]:
        """Generate a single rollout and score it."""
        output = await self.rollout_engine.get_model_response(messages)
        response_text = output.text or output.content or ""
        reward = self.reward_fn(task, response_text)
        step = Step.from_model_output(output, messages=messages)
        step.reward = reward
        step.done = True
        return step, reward

    # ------------------------------------------------------------------
    # Case 1: Teacher valid → distillation
    # ------------------------------------------------------------------

    async def _case1_distillation(
        self,
        uid: str,
        question: str,
        hint_text: str,
        student_results: list[tuple[Step, float]],
        teacher_results: list[tuple[Step, float]],
        trajectories: list[Trajectory],
    ) -> None:
        engine = self.rollout_engine
        client = engine.sampling_client
        cfg = self.cfg

        # Tokenize prompts for scoring (use overridable prompt builders)
        teacher_prompt_ids = self._tokenize_messages(self._make_teacher_prompt(question, hint_text))
        student_prompt_ids = self._tokenize_messages(self._make_student_prompt(question))

        # (a) On-policy distillation: score student seqs under teacher
        scoring_coros = [
            score_teacher_for_response(
                client,
                teacher_prompt_ids,
                step.response_ids,
                topk=cfg.distill_topk,
                max_context_length=cfg.max_context_length,
            )
            for step, _ in student_results
        ]

        # (b) Supervised distillation: score teacher's correct seqs
        correct_teacher = [(s, r) for s, r in teacher_results if r >= cfg.success_reward_threshold]
        sup_scoring_coros = [
            score_teacher_for_response(
                client,
                teacher_prompt_ids,
                step.response_ids,
                topk=cfg.distill_topk,
                max_context_length=cfg.max_context_length,
            )
            for step, _ in correct_teacher
        ]

        # Run all scoring passes concurrently.
        # When a ScoringAccumulator is available, submit coroutines there so
        # they get batched with scoring requests from other concurrent workflows.
        all_scoring_coros = scoring_coros + sup_scoring_coros
        if not all_scoring_coros:
            all_scorings = []
        elif self.scoring_accumulator is not None:
            all_scorings = await asyncio.gather(*[self.scoring_accumulator.submit(c) for c in all_scoring_coros])
        else:
            all_scorings = await _gather_with_progress(
                all_scoring_coros,
                desc=f"[train:{uid}] teacher scoring",
            )

        on_policy_scorings = all_scorings[: len(scoring_coros)]
        sup_scorings = all_scorings[len(scoring_coros) :]

        # Process on-policy results
        for (step, reward), scoring in zip(student_results, on_policy_scorings, strict=True):
            if scoring["response_len"] == 0:
                continue
            advantages = compute_sampled_rkl_advantages(
                teacher_logprobs=scoring["sampled_logprobs"],
                student_logprobs=step.logprobs[: scoring["response_len"]],
                kl_coeff=cfg.kl_coeff,
                clip_min=cfg.kl_clip_min,
                clip_max=cfg.kl_clip_max,
            )
            step.advantage = advantages
            # Store teacher Top-K so the combined datum has CE data too
            step.metadata = step.metadata or {}
            step.metadata["teacher_topk"] = {
                "topk_ids": scoring["topk_ids"],
                "topk_logprobs": scoring["topk_logprobs"],
            }
            traj = Trajectory(name="gsd_distill", steps=[step], reward=reward)
            trajectories.append(traj)

        # Process supervised results → combined datums with CE fields only
        for (step, reward), scoring in zip(correct_teacher, sup_scorings, strict=True):
            if scoring["response_len"] == 0:
                continue
            sup_step = copy.deepcopy(step)
            sup_step.prompt_ids = student_prompt_ids
            sup_step.chat_completions = self._make_student_prompt(question)
            sup_step.model_output = None
            sup_step.advantage = None  # no IS for supervised datums
            sup_step.metadata = {
                "teacher_topk": {
                    "topk_ids": scoring["topk_ids"],
                    "topk_logprobs": scoring["topk_logprobs"],
                },
            }
            traj = Trajectory(name="gsd_distill", steps=[sup_step], reward=reward)
            trajectories.append(traj)

        n_distill = sum(1 for t in trajectories if t.name == "gsd_distill")
        logger.info(f"[{uid}] distillation: {n_distill} combined datums ({len(on_policy_scorings)} on-policy + {len(correct_teacher)} supervised)")

    # ------------------------------------------------------------------
    # Case 2: Teacher invalid → GRPO fallback
    # ------------------------------------------------------------------

    def _case2_grpo_fallback(
        self,
        student_results: list[tuple[Step, float]],
        trajectories: list[Trajectory],
    ) -> None:
        for step, reward in student_results:
            traj = Trajectory(name="gsd_student", steps=[step], reward=reward)
            trajectories.append(traj)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_student_prompt(self, question: str) -> list[dict]:
        """Build the student prompt.  Override in subclasses for task-specific prompts."""
        return build_student_prompt(question)

    def _make_teacher_prompt(self, question: str, hint: str) -> list[dict]:
        """Build the teacher prompt.  Override in subclasses for task-specific prompts."""
        return build_teacher_prompt(question, hint)

    def _tokenize_messages(self, messages: list[dict]) -> list[int]:
        """Convert OpenAI-style messages to token IDs via the engine's chat parser.

        Uses ``build_prompt()`` which goes through the Tinker renderer natively,
        avoiding the token→string→token round-trip of ``parse()`` + ``encode()``.
        """
        model_input = self.rollout_engine.chat_parser.build_prompt(messages)
        return model_input.to_ints()

    def _build_episode(
        self,
        uid: str,
        task: dict,
        trajectories: list[Trajectory],
        metrics: dict,
        is_correct: bool,
    ) -> Episode:
        return Episode(
            id=uid,
            task=task,
            is_correct=is_correct,
            trajectories=trajectories,
            metrics=metrics,
        )


# ---------------------------------------------------------------------------
# Async gather with tqdm progress bar
# ---------------------------------------------------------------------------


async def _gather_with_progress(
    coros: list[Awaitable[T]],
    desc: str = "GSD",
) -> list[T]:
    """Run coroutines concurrently, showing a tqdm progress bar as each completes."""
    results: list[T | None] = [None] * len(coros)

    # Wrap each coroutine to track its original index
    async def _indexed(idx: int, coro: Awaitable[T]) -> tuple[int, T]:
        return idx, await coro

    tasks = [_indexed(i, c) for i, c in enumerate(coros)]

    with tqdm(total=len(tasks), desc=desc, leave=False) as pbar:
        for future in asyncio.as_completed(tasks):
            idx, result = await future
            results[idx] = result
            pbar.update(1)

    return results  # type: ignore[return-value]
