"""Native rLLM SWE GRPO on verl — Qwen3.5-4B + mini-swe-agent.

Not Harbor: the rollout is ``AgentFlowEngine`` → ``SandboxTaskHooks`` (Docker)
→ ``MiniSweAgentHarness`` (the CLI runs *inside* the task sandbox and calls
back through the rLLM model gateway) → ``ShellScriptEvaluator`` (``tests/test.sh``
writes ``/logs/verifier/reward.txt``). ``rllm.remote_runtime.enabled=false``.

Datasets are the small locally-materialized benchmarks built by
``scripts/prepare_datasets.py``; both names are overridable from the CLI::

    python recipe/qwen3_5_swe_grpo/train.py \\
        recipe.train_dataset=rllm_swesmith_small \\
        recipe.val_dataset=swebench_verified_local

Normally launched via ``train_verl.sh`` / ``smoke_test.sh``.
"""

from __future__ import annotations

import logging
import os
import time

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.harnesses.mini_swe_agent import MiniSweAgentHarness
from rllm.trainer import AgentTrainer
from rllm.trainer.algorithms.transform import _default_traj_grouping_hook
from rllm.types import AgentConfig, Task
from rllm.workflows.workflow import TerminationReason

logger = logging.getLogger(__name__)

# SWE-Master's "budget exhaustion" terminations (arXiv 2602.03411, sec. 3.4.2:
# TIMEOUT, MAX_STEPS, MAX_TOKENS). The rollout is graded on whatever the
# agent left in the repo -- the verifier runs regardless of how the agent
# stopped, which is the paper's "forced submission" -- and its reward is then
# scaled by a constant below 1. Not in the set: VERIFIER_TIMEOUT and ERROR,
# which are infrastructure and are dropped by compact_filtering instead.
BUDGET_EXHAUSTED = frozenset(
    {
        TerminationReason.MAX_TURNS_EXCEEDED,  # paper's MAX_STEPS
        TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED,  # paper's MAX_TOKENS: vLLM 400 on the cumulative prompt
        TerminationReason.AGENT_TIMEOUT,  # paper's TIMEOUT: task.toml [agent] timeout_sec
    }
)


def make_budget_scaled_grouping_hook(scale: float):
    """Return a ``traj_grouping_hook`` that applies SWE-Master's reward shaping.

    The hook runs before trajectory groups are built, so the scaled reward is
    what RLOO sees. ``is_correct`` is deliberately left alone: rejection
    sampling and the solve_all/solve_none metrics read it, and "solved but
    slowly" is still solved for those purposes.

    One caveat: the trainer runs the same hook over validation episodes, so
    ``val/reward_*`` is scaled too. ``val/accuracy`` is built from
    ``is_correct`` and is not affected.
    """
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"budget_reward_scale must be in [0, 1], got {scale}")

    def hook(episodes, transform_config, compact_filtering_config=None):
        if scale != 1.0:
            for episode in episodes:
                if episode.termination_reason not in BUDGET_EXHAUSTED:
                    continue
                for trajectory in episode.trajectories:
                    if trajectory.reward is not None:
                        trajectory.reward = trajectory.reward * scale
        return _default_traj_grouping_hook(episodes, transform_config, compact_filtering_config)

    return hook

# Episode logs are keyed by <project>/<experiment>, which is stable across runs,
# so two runs of this recipe wrote into the same train_step_N_epoch_0 directory:
# one merged 88 old episodes with 64 new ones, and the next silently *overwrote*
# the previous run's files, because the episode filename is a hash of the task
# plus the rollout index and both runs draw the same tasks in the same order.
# Post-hoc analysis then reads two runs as one. Stamp a run id so each launch
# owns its directory. Set before Hydra composes, since the config interpolates
# ``${oc.env:RLLM_RUN_ID}``; ``setdefault`` lets train_verl.sh pin the same id it
# puts on the transcript filename, and a bare ``python train.py`` still gets one.
os.environ.setdefault("RLLM_RUN_ID", time.strftime("%Y%m%d_%H%M%S"))


class StepLimitedMiniSweAgent(MiniSweAgentHarness):
    """mini-swe-agent with an explicit turn budget.

    Upstream defaults to ``agent.step_limit: 0`` (unlimited) and guards runtime
    with ``cost_limit`` instead — which is inert here, because the gateway-routed
    model has no litellm cost table and the harness sets
    ``MSWEA_COST_TRACKING=ignore_errors``. Left unbounded, the cumulative prompt
    keeps growing until vLLM rejects the turn with "maximum context length is N
    tokens", the agent's retries all fail, and the whole episode is thrown away
    as an error instead of being scored.

    ``-c`` normally *replaces* the default config rather than layering on it, so
    the builtin ``mini.yaml`` has to be named again before the override.
    """

    step_limit: int = 50

    def build_invocation(self, instruction: str, task: Task, config: AgentConfig) -> str:
        invocation = super().build_invocation(instruction, task, config)
        return invocation.replace(
            "mini-swe-agent --yolo ",
            f"mini-swe-agent --yolo -c mini.yaml -c agent.step_limit={int(self.step_limit)} ",
            1,
        )


def _load(name: str, split: str, limit: int | None, kind: str):
    # as_tasks=True roots each row at its ``task_path`` and merges the per-task
    # ``task.toml``. Without it every Task lands on ``dataset_dir="."`` and the
    # per-task verifier auto-detection fails with "No verifier configured".
    dataset = DatasetRegistry.load_dataset(name, split, as_tasks=True)
    if dataset is None:
        raise SystemExit(
            f"{kind} dataset '{name}/{split}' is not registered.\n"
            f"Build it first:  python recipe/qwen3_5_swe_grpo/scripts/prepare_datasets.py"
        )
    if limit and limit > 0 and limit < len(dataset):
        dataset = dataset.select(range(limit))
    logger.info("%s dataset %s/%s: %d tasks", kind, name, split, len(dataset))
    return dataset


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    recipe = config.recipe

    train_dataset = _load(recipe.train_dataset, recipe.train_split, recipe.get("train_limit"), "train")
    val_dataset = _load(recipe.val_dataset, recipe.val_split, recipe.get("val_limit"), "val")

    # `auto` mounts a pre-built mini-swe-agent image into the task sandbox
    # instead of running `uv tool install` on every rollout (Docker only).
    # Same mechanism as `rllm eval --agent-image`.
    agent_image = os.environ.get("RLLM_AGENT_IMAGE", recipe.agent_image)
    os.environ["RLLM_AGENT_IMAGE"] = str(agent_image)

    agent_flow = StepLimitedMiniSweAgent(step_limit=recipe.agent_step_limit)
    agent_flow.configure({"agent_image": agent_image})

    trainer = AgentTrainer(
        backend="verl",
        agent_flow=agent_flow,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        sandbox_backend=os.environ.get("SANDBOX_BACKEND", recipe.sandbox_backend),
        sandbox_concurrency=recipe.get("sandbox_concurrency"),
        # Passed through **kwargs to UnifiedTrainer (verl_launcher.py forwards them).
        traj_grouping_hook=make_budget_scaled_grouping_hook(float(recipe.budget_reward_scale)),
    )
    trainer.train()


if __name__ == "__main__":
    main()
