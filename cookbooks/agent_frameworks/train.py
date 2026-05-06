"""Train any of the framework AgentFlows in this cookbook.

Usage (from rllm repo root):
    python cookbooks/agent_frameworks/train.py +rllm.agent_name=strands_math

Discoverable agents (registered as entry points by ``pyproject.toml``):
    - langgraph_math
    - openai_agents_math
    - smolagents_math
    - strands_math

The matching evaluator is ``math_evaluator``.
"""

import hydra
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.eval.agent_loader import load_agent
from rllm.eval.evaluator_loader import load_evaluator
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    agent_name = config.rllm.get("agent_name") or "langgraph_math"
    evaluator_name = config.rllm.get("evaluator_name") or "math_evaluator"

    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("math500", "test")

    if train_dataset is None:
        raise RuntimeError("deepscaler_math train split not found. Run: rllm dataset pull deepscaler_math")
    if test_dataset is None:
        raise RuntimeError("math500 test split not found. Run: rllm dataset pull math500")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=load_agent(agent_name),
        evaluator=load_evaluator(evaluator_name),
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
