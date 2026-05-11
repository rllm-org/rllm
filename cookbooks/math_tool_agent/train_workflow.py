"""Train ``MathToolWorkflow`` via the unified ``AgentTrainer``.

Usage (from rllm repo root):
    python cookbooks/math_tool_agent/train_workflow.py

Or with Hydra overrides:
    python cookbooks/math_tool_agent/train_workflow.py \\
        model.name=Qwen/Qwen3-1.7B training.group_size=4

Backend selection follows ``rllm/backend=verl`` (or ``tinker``) as a Hydra
override — the underlying ``AgentTrainer`` picks the right launcher based
on ``config.rllm.backend``.

The workflow path (as opposed to the agentflow path used by ``train.py``)
goes through ``rllm.experimental.engine.UnifiedWorkflowEngine``. This is
the path that:

  - Uses ``rllm.experimental.rollout.RolloutEngine`` directly (no gateway).
  - Plumbs ``TITOCompleter`` (and therefore token-level step merging) into
    each rollout call.
  - Yields ``Step.prompt_ids + Step.response_ids`` for the transform path
    to recognize the prefix-extension property between turns.

If everything is wired correctly, the training log should show
``batch/merge_compression_ratio > 1.0`` on the first training step.
"""

import hydra
from omegaconf import DictConfig
from workflow import MathToolWorkflow

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("math500", "test")

    if train_dataset is None:
        raise RuntimeError("deepscaler_math train split not found. Run: rllm dataset pull deepscaler_math")
    if test_dataset is None:
        raise RuntimeError("math500 test split not found. Run: rllm dataset pull math500")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        workflow_class=MathToolWorkflow,
        workflow_args={
            "max_turns": config.rllm.workflow.get("max_turns", 8),
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
