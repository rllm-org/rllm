"""Train the LaMer (Meta-RL) WebShop agent.

Usage from the rllm repo root::

    python3 -m tbmf.webshop.train.train_lamer rllm/backend=verl
"""

import hydra
from omegaconf import DictConfig

try:
    from ..eval.webshop_lamer_eval import webshop_lamer_evaluator
    from ..flow.webshop_lamer_flow import webshop_lamer_flow
    from ..webshop_env import WebShopEnv
except (ImportError, ValueError):
    from eval.webshop_lamer_eval import webshop_lamer_evaluator
    from flow.webshop_lamer_flow import webshop_lamer_flow
    from webshop_env import WebShopEnv

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("webshop", "train")
    val_dataset = DatasetRegistry.load_dataset("webshop", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("WebShop dataset not found. Run: python3 tbmf/webshop/prepare_webshop_data.py")

    try:
        trainer = AgentTrainer(
            backend=config.rllm.get("backend", "verl"),
            agent_flow=webshop_lamer_flow,
            evaluator=webshop_lamer_evaluator,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        trainer.train()
    finally:
        WebShopEnv.shutdown_pool()


if __name__ == "__main__":
    main()
