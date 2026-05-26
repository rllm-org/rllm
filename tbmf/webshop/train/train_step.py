"""Train the step-based (non-cumulative) WebShop agent.

Usage from the rllm repo root::

    python3 -m tbmf.webshop.train.train_step rllm/backend=tinker
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

try:
    from ..eval.webshop_eval import webshop_evaluator
    from ..flow.webshop_step_flow import webshop_step_flow
except (ImportError, ValueError):
    from eval.webshop_eval import webshop_evaluator
    from flow.webshop_step_flow import webshop_step_flow

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("webshop", "train")
    val_dataset = DatasetRegistry.load_dataset("webshop", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError(
            "WebShop dataset not found. Run: python3 tbmf/webshop/prepare_webshop_data.py"
        )

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=webshop_step_flow,
        evaluator=webshop_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    try:
        trainer.train()
    finally:
        try:
            try:
                from ..webshop_env import WebShopEnv
            except (ImportError, ValueError):
                from webshop_env import WebShopEnv

            WebShopEnv.shutdown_pool()
        except Exception:
            pass


if __name__ == "__main__":
    main()
