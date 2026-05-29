"""Multi-pass validation framework for agent training.

Enables running multiple validation flows per task (e.g., single-episode GRPO
and multi-episode LaMer) in a single training run. Each pass produces its own
set of evaluation signals, prefixed by pass name.

The MultiPassFlow satisfies the AgentFlow protocol and concatenates trajectories
from all validation passes so that the engine's trace enrichment works correctly
(positional matching against chronologically-ordered gateway traces).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rllm.eval.types import EvalOutput, Signal
from rllm.types import AgentConfig, Episode, Task, run_agent_flow

logger = logging.getLogger(__name__)


@dataclass
class ValidationPass:
    """A single validation pass configuration.

    Attributes:
        sample_budget: Controls how many of the engine's N rollout samples
            this pass runs on. None = run on all samples (follows val_kwargs.n).
            1 = only run on the first sample (rollout_idx=0). Use sample_budget=1
            for expensive multi-episode passes where pass@k doesn't apply.
    """

    name: str
    flow: Any
    evaluator: Any
    enabled: bool = True
    sample_budget: int | None = None


@dataclass
class MultiPassConfig:
    """Configuration for multi-pass training and validation."""

    train_flow: Any
    train_evaluator: Any
    val_passes: list[ValidationPass] = field(default_factory=list)


class MultiPassFlow:
    """Routes training to one flow; validation to N configurable passes.

    During training (is_validation=False): delegates to train_flow directly.
    During validation (is_validation=True): runs each enabled validation pass
    sequentially, concatenates all trajectories, and stores per-pass artifacts
    for the MultiPassEvaluator to score.
    """

    def __init__(self, config: MultiPassConfig):
        self.config = config

    def _get_rollout_idx(self, config: AgentConfig) -> int:
        """Extract the rollout sample index from the session uid (format: task_id:idx)."""
        uid = getattr(config, "session_uid", None) or ""
        parts = uid.rsplit(":", 1)
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 0

    async def arun(self, task: Task, config: AgentConfig) -> Episode:
        if not config.is_validation:
            return await run_agent_flow(self.config.train_flow, task, config)

        rollout_idx = self._get_rollout_idx(config)
        all_trajectories = []
        pass_results: dict[str, dict] = {}

        enabled_passes = [vp for vp in self.config.val_passes if vp.enabled]
        if not enabled_passes:
            logger.warning("No validation passes enabled; returning empty episode")
            return Episode(is_correct=False, trajectories=[], artifacts={})

        for vp in enabled_passes:
            if vp.sample_budget is not None and rollout_idx >= vp.sample_budget:
                continue
            episode = await run_agent_flow(vp.flow, task, config)
            all_trajectories.extend(episode.trajectories)
            pass_data = episode.artifacts.copy() if episode.artifacts else {}
            pass_data["__is_correct__"] = episode.is_correct
            pass_results[vp.name] = pass_data

        # Primary is_correct from the first pass that actually ran
        first_pass_name = next(
            (vp.name for vp in enabled_passes if vp.name in pass_results), None
        )
        primary_correct = (
            pass_results[first_pass_name]["__is_correct__"] if first_pass_name else False
        )

        return Episode(
            is_correct=primary_correct,
            trajectories=all_trajectories,
            artifacts={"__multi_pass__": pass_results},
        )


class MultiPassEvaluator:
    """Evaluator that delegates to per-pass sub-evaluators with prefixed signals.

    For training episodes (no __multi_pass__ key): delegates to train_evaluator.
    For validation episodes: builds a minimal Episode per pass from stored
    artifacts, runs each pass's evaluator, and collects prefixed signals.
    """

    def __init__(self, config: MultiPassConfig):
        self.config = config

    def evaluate(self, task: Any, episode: Episode) -> EvalOutput:
        pass_results = episode.artifacts.get("__multi_pass__") if episode.artifacts else None

        if pass_results is None:
            return self.config.train_evaluator.evaluate(task, episode)

        signals: list[Signal] = []
        primary_reward = 0.0
        primary_correct = False
        found_primary = False

        enabled_passes = [vp for vp in self.config.val_passes if vp.enabled]

        for vp in enabled_passes:
            if vp.name not in pass_results:
                continue

            pass_data = pass_results[vp.name]
            sub_episode = Episode(
                is_correct=pass_data.get("__is_correct__", False),
                artifacts={k: v for k, v in pass_data.items() if not k.startswith("__")},
                trajectories=[],
            )
            eval_out = vp.evaluator.evaluate(task, sub_episode)

            prefix = f"{vp.name}/"
            for s in eval_out.signals:
                signals.append(Signal(name=f"{prefix}{s.name}", value=s.value))

            if not found_primary:
                primary_reward = eval_out.reward
                primary_correct = eval_out.is_correct
                found_primary = True

        return EvalOutput(
            reward=primary_reward,
            is_correct=primary_correct,
            signals=signals,
        )
