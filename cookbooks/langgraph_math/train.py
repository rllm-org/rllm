"""Train the LangGraph math agent using the Python API.

Usage (from rllm repo root):
    python cookbooks/langgraph_math/train.py

Or with Hydra overrides:
    python cookbooks/langgraph_math/train.py model.name=Qwen/Qwen3-1.7B training.group_size=4
"""

import hydra
from evaluator import langgraph_math_evaluator
from langgraph_math import langgraph_math
from omegaconf import DictConfig

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
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=langgraph_math,
        evaluator=langgraph_math_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
