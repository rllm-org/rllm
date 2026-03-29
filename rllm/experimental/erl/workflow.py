"""ErlWorkflow — generic Experiential Reinforcement Learning workflow.

Implements the 4-phase ERL training loop
(first attempt → reflection → second attempt → internalization)
as a :class:`~rllm.workflows.workflow.Workflow`
subclass that works on both the Tinker and Verl backends.

Users provide three task-specific callables (``solver_fn``,
``feedback_fn``, ``state_builder_fn``) and the workflow handles the
orchestration, gating, distillation, and cross-episode memory.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rllm.agents.agent import Episode, Trajectory
from rllm.engine.rollout.rollout_engine import RolloutEngine
from rllm.experimental.erl.updater import ErlPromptUpdater
from rllm.experimental.erl.utils import GENERIC_RETRY_INSTRUCTION, UPDATER_SYSTEM_PROMPT, default_feedback
from rllm.workflows.workflow import Workflow

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    from rllm.workflows.store import Store

# Type aliases for the user-supplied callables.
SolverFn = Callable[[str, dict, RolloutEngine], Awaitable[Trajectory]]
FeedbackFn = Callable[[dict, Trajectory], str]
StateBuilderFn = Callable[[str, dict, Trajectory, str], str]


@dataclass
class ErlConfig:
    """Configuration for the ERL workflow.

    Groups all serialisable / Hydra-friendly parameters.  Callables and
    runtime objects (``solver_fn``, ``store``, ``rollout_engine``, …) are
    passed directly to :class:`ErlWorkflow.__init__` instead.

    Example with Hydra structured configs::

        @dataclass
        class MyConfig:
            erl: ErlConfig = field(default_factory=ErlConfig)
    """

    # ---- prompt configuration ----
    initial_system_prompt: str = ""
    updater_system_prompt: str = UPDATER_SYSTEM_PROMPT
    updater_sampling_params: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.7, "top_p": 0.9})

    # ---- phase control flags ----
    train_first_attempt: bool = True
    train_second_attempt: bool = True
    train_distilled: bool = True
    train_updater: bool = True

    # ---- gating / ablation ----
    success_reward_threshold: float = 1.0
    no_memory: bool = False
    no_reflection: bool = False


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
        executor: ThreadPoolExecutor,
        *,
        solver_fn: SolverFn,
        state_builder_fn: StateBuilderFn,
        feedback_fn: FeedbackFn | None = None,
        erl_config: ErlConfig | None = None,
        store: Store | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rollout_engine=rollout_engine,
            executor=executor,
            store=store,
            **kwargs,
        )
        cfg = erl_config or ErlConfig()

        self.solver_fn = solver_fn
        self.feedback_fn: FeedbackFn = feedback_fn or default_feedback
        self.state_builder_fn = state_builder_fn
        self.initial_system_prompt = cfg.initial_system_prompt

        self.train_first_attempt = cfg.train_first_attempt
        self.train_second_attempt = cfg.train_second_attempt
        self.train_distilled = cfg.train_distilled
        self.train_updater = cfg.train_updater
        self.success_reward_threshold = cfg.success_reward_threshold
        self.no_memory = cfg.no_memory
        self.no_reflection = cfg.no_reflection

        self.updater = ErlPromptUpdater(
            rollout_engine=rollout_engine,
            system_prompt=cfg.updater_system_prompt,
            sampling_params=cfg.updater_sampling_params,
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

        # ---- Compute-gate: skip reflection if no flags need it -----
        needs_second = self.train_second_attempt or self.train_distilled
        needs_reflection = needs_second or self.train_updater
        if not needs_reflection:
            return self._build_episode(
                uid,
                task,
                trajectories,
                metrics={
                    "avg_first_reward": first_reward,
                    "first_success_rate": 0.0,
                },
                is_correct=False,
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
            # Inject failure context without the updater LLM call.
            improved_prompt = self._build_generic_retry_prompt(base_prompt, state)
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

    @staticmethod
    def _build_generic_retry_prompt(base_prompt: str, state: str) -> str:
        """Build a second-attempt prompt with failure context but without the updater LLM.

        Used when ``no_reflection=True`` to still give the solver contextual
        information about its previous failure.
        """
        return f"{base_prompt.strip()}\n\n{GENERIC_RETRY_INSTRUCTION}\n\n{state.strip()}"

    def _build_episode(
        self,
        uid: str,
        task: dict,
        trajectories: list[Trajectory],
        metrics: dict[str, float],
        is_correct: bool,
    ) -> Episode:
        # Note: we intentionally do NOT call adjust_step_rewards or
        # compute_trajectory_reward here.  The solver_fn sets
        # trajectory.reward directly, and the UnifiedTrainer's transform
        # pipeline handles reward propagation in broadcast mode.
        return Episode(
            id=uid,
            task=task,
            is_correct=is_correct,
            trajectories=trajectories,
            metrics=metrics,
        )
