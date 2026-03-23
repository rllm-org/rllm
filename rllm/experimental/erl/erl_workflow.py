"""ErlWorkflow — generic Experiential Reinforcement Learning workflow.

Implements the 4-phase ERL training loop (first attempt → reflection →
second attempt → internalization) as a :class:`~rllm.workflows.workflow.Workflow`
subclass that works on both the Tinker and Verl backends.

Users provide three task-specific callables (``solver_fn``,
``feedback_fn``, ``state_builder_fn``) and the workflow handles the
orchestration, gating, distillation, and cross-episode memory.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from typing import Any

from rllm.agents.agent import Episode, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.experimental.erl.updater import ErlPromptUpdater
from rllm.experimental.erl.utils import UPDATER_SYSTEM_PROMPT, default_feedback
from rllm.workflows.store import Store
from rllm.workflows.workflow import Workflow

# Type aliases for the user-supplied callables.
SolverFn = Callable[[str, dict, RolloutEngine], Awaitable[Trajectory]]
FeedbackFn = Callable[[dict, Trajectory], str]
StateBuilderFn = Callable[[str, dict, Trajectory, str], str]


class ErlWorkflow(Workflow):
    """Experiential Reinforcement Learning workflow.

    Each call to :meth:`run` executes one task through the ERL loop:

    1. **First attempt** — solver generates a trajectory with the base prompt.
    2. **Gated reflection** — if the first attempt failed, the model reflects
       on the attempt and proposes an improved prompt.
    3. **Second attempt** — solver retries with the improved prompt.
    4. **Internalization** — a copy of the second-attempt trajectory has its
       system prompt replaced with the original, teaching the model to
       reproduce the improved behavior without reflection at inference time.

    The four trajectory roles (``erl_first``, ``erl_updater``,
    ``erl_second``, ``erl_distill``) can each use a different advantage
    estimator via ``traj_group_adv_estimator_map`` on the trainer.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        *,
        # ---- task-specific callables ----
        solver_fn: SolverFn,
        state_builder_fn: StateBuilderFn,
        initial_system_prompt: str,
        feedback_fn: FeedbackFn | None = None,
        # ---- updater configuration ----
        updater_system_prompt: str = UPDATER_SYSTEM_PROMPT,
        updater_sampling_params: dict[str, Any] | None = None,
        # ---- phase control flags ----
        train_first_attempt: bool = True,
        train_second_attempt: bool = True,
        train_distilled: bool = True,
        train_updater: bool = True,
        # ---- gating / ablation ----
        success_reward_threshold: float = 1.0,
        no_memory: bool = False,
        no_reflection: bool = False,
        # ---- store (injected by the engine) ----
        store: Store | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rollout_engine=rollout_engine,
            executor=executor,
            store=store,
            **kwargs,
        )
        self.solver_fn = solver_fn
        self.feedback_fn: FeedbackFn = feedback_fn or default_feedback
        self.state_builder_fn = state_builder_fn
        self.initial_system_prompt = initial_system_prompt

        self.train_first_attempt = train_first_attempt
        self.train_second_attempt = train_second_attempt
        self.train_distilled = train_distilled
        self.train_updater = train_updater
        self.success_reward_threshold = success_reward_threshold
        self.no_memory = no_memory
        self.no_reflection = no_reflection

        self.updater = ErlPromptUpdater(
            rollout_engine=rollout_engine,
            system_prompt=updater_system_prompt,
            sampling_params=updater_sampling_params,
        )

    # ------------------------------------------------------------------
    # Main ERL loop
    # ------------------------------------------------------------------

    async def run(self, task: dict, uid: str, **kwargs: Any) -> Episode:
        self.reset(task, uid)
        is_validation = kwargs.get("is_validation", False)
        trajectories: list[Trajectory] = []

        # ---- Phase 1: First attempt --------------------------------
        first_traj = await self.solver_fn(self.initial_system_prompt, task, self.rollout_engine)
        first_traj.name = "erl_first"
        first_reward = first_traj.reward if first_traj.reward is not None else 0.0

        if self.train_first_attempt and not is_validation:
            trajectories.append(first_traj)

        if is_validation:
            trajectories.append(first_traj)
            return self._build_episode(
                uid,
                task,
                trajectories,
                metrics={"avg_reward": first_reward},
                is_correct=first_reward >= self.success_reward_threshold,
            )

        # ---- Gating: skip reflection when first attempt succeeds ----
        if first_reward >= self.success_reward_threshold:
            return self._build_episode(
                uid,
                task,
                trajectories,
                metrics={
                    "avg_first_reward": first_reward,
                    "first_success_rate": 1.0,
                },
                is_correct=True,
            )

        # ---- Phase 2: Self-reflection ------------------------------
        base_prompt = self.initial_system_prompt
        if self.store and not self.no_memory:
            stored = await self.store.get("improved_prompt")
            if stored is not None:
                base_prompt = stored

        feedback = self.feedback_fn(task, first_traj)
        state = self.state_builder_fn(base_prompt, task, first_traj, feedback)

        updater_traj: Trajectory | None = None
        if self.no_reflection:
            improved_prompt = base_prompt
        else:
            improved_prompt, updater_traj = await self.updater.propose_prompt(state, base_prompt)

        # ---- Phase 3: Second attempt -------------------------------
        second_traj = await self.solver_fn(improved_prompt, task, self.rollout_engine)
        second_traj.name = "erl_second"
        second_reward = second_traj.reward if second_traj.reward is not None else 0.0

        if self.train_second_attempt:
            trajectories.append(second_traj)

        # ---- Phase 4: Internalization (distillation) ----------------
        if self.train_distilled:
            distilled_traj = self._create_distilled_trajectory(second_traj)
            trajectories.append(distilled_traj)

        # ---- Updater trajectory (reward aligned to second attempt) --
        if updater_traj is not None and self.train_updater:
            updater_traj.reward = second_reward
            for step in updater_traj.steps:
                step.reward = second_reward
            trajectories.append(updater_traj)

        # ---- Memory update -----------------------------------------
        second_succeeded = second_reward >= self.success_reward_threshold
        if self.store and not self.no_memory and not self.no_reflection and second_succeeded:
            await self.store.set("improved_prompt", improved_prompt)

        return self._build_episode(
            uid,
            task,
            trajectories,
            metrics={
                "avg_first_reward": first_reward,
                "avg_second_reward": second_reward,
                "first_success_rate": 0.0,
                "second_success_rate": float(second_succeeded),
            },
            is_correct=second_succeeded,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_distilled_trajectory(self, second_traj: Trajectory) -> Trajectory:
        """Deep-copy second attempt and replace system prompt with the initial one."""
        distilled = copy.deepcopy(second_traj)
        distilled.name = "erl_distill"
        for step in distilled.steps:
            step.model_output = None
            if step.chat_completions and step.chat_completions[0].get("role") == "system":
                step.chat_completions[0]["content"] = self.initial_system_prompt
        return distilled

    def _build_episode(
        self,
        uid: str,
        task: dict,
        trajectories: list[Trajectory],
        metrics: dict[str, float],
        is_correct: bool,
    ) -> Episode:
        for traj in trajectories:
            self.adjust_step_rewards(traj)
            self.compute_trajectory_reward(traj)

        return Episode(
            id=uid,
            task=task,
            is_correct=is_correct,
            trajectories=trajectories,
            metrics=metrics,
        )
