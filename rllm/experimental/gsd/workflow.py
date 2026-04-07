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
    """Configuration for the GSD workflow (Phase 1 — no meta-RL, no buffer)."""

    N: int = 4
    N_val: int = 2
    distill_topk: int = 20
    train_hint: bool = False
    hint_sampling_params: dict[str, Any] = field(
        default_factory=lambda: {"temperature": 0.6, "top_p": 0.9, "max_tokens": 256},
    )
    kl_coeff: float = 1.0
    kl_clip_min: float = -5.0
    kl_clip_max: float = 5.0
    success_reward_threshold: float = 0.5
    max_context_length: int = 32768


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
        **kwargs: Any,
    ) -> None:
        super().__init__(rollout_engine=rollout_engine, executor=executor, **kwargs)
        self.cfg = gsd_config or GsdConfig()
        self.reward_fn = reward_fn

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, task: dict, uid: str, **kwargs: Any) -> Episode:
        is_validation = self.rollout_engine.is_validation

        if is_validation:
            return await self._do_validation(task, uid)
        return await self._do_training(task, uid)

    # ------------------------------------------------------------------
    # Validation: N_val student + N_val teacher, report pass@N metrics
    # ------------------------------------------------------------------

    async def _do_validation(self, task: dict, uid: str) -> Episode:
        """Run validation rollouts and compute pass@N metrics for both roles."""
        question = task["question"]
        N = self.cfg.N_val

        # Generate hint
        hint_messages = build_hint_prompt(question)
        hint_output = await self.rollout_engine.get_model_response(
            hint_messages,
            **self.cfg.hint_sampling_params,
        )
        hint_text = extract_hint(hint_output.text or hint_output.content or "")

        # Dual rollouts with progress
        student_messages = build_student_prompt(question)
        teacher_messages = build_teacher_prompt(question, hint_text)

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
        hint_messages = build_hint_prompt(question)
        hint_output = await self.rollout_engine.get_model_response(
            hint_messages,
            **self.cfg.hint_sampling_params,
        )
        hint_text = extract_hint(hint_output.text or hint_output.content or "")

        # ---- Phase 2: Dual rollouts (concurrent with progress) ----
        student_messages = build_student_prompt(question)
        teacher_messages = build_teacher_prompt(question, hint_text)

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
        teacher_valid = R_T_avg >= R_S_avg

        trajectories: list[Trajectory] = []
        metrics: dict[str, Any] = {
            "R_S_avg": R_S_avg,
            "R_T_avg": R_T_avg,
            "teacher_valid": float(teacher_valid),
            "hint_improvement": R_T_avg - R_S_avg,
        }

        # ---- Phase 4: Loss routing --------------------------------
        if teacher_valid:
            await self._case1_distillation(
                uid,
                question,
                hint_text,
                student_results,
                teacher_results,
                trajectories,
            )
        else:
            self._case2_grpo_fallback(student_results, trajectories)

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

        # Tokenize prompts for scoring
        teacher_prompt_ids = self._tokenize_messages(build_teacher_prompt(question, hint_text))
        student_prompt_ids = self._tokenize_messages(build_student_prompt(question))

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

        # Run all scoring passes concurrently with progress
        all_scoring_coros = scoring_coros + sup_scoring_coros
        if all_scoring_coros:
            all_scorings = await _gather_with_progress(
                all_scoring_coros,
                desc=f"[train:{uid}] teacher scoring",
            )
        else:
            all_scorings = []

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
            traj = Trajectory(name="gsd_distill_onpolicy", steps=[step], reward=reward)
            trajectories.append(traj)

        # Process supervised results
        for (step, reward), scoring in zip(correct_teacher, sup_scorings, strict=True):
            if scoring["response_len"] == 0:
                continue
            # Deep-copy and swap prompt to student view (hint removed)
            sup_step = copy.deepcopy(step)
            sup_step.prompt_ids = student_prompt_ids
            sup_step.chat_completions = build_student_prompt(question)
            sup_step.model_output = None  # recompute student logprobs during training
            sup_step.advantage = None
            sup_step.metadata = {
                "teacher_topk": {
                    "topk_ids": scoring["topk_ids"],
                    "topk_logprobs": scoring["topk_logprobs"],
                },
            }
            traj = Trajectory(name="gsd_distill_supervised", steps=[sup_step], reward=reward)
            trajectories.append(traj)

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
