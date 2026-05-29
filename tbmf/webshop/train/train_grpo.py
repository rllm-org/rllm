"""GRPO WebShop training with multi-pass validation.

Train: single-episode GRPO rollouts.
Val pass 1 (single_episode): standard GRPO flow -> pass@1/pass@4.
Val pass 2 (multi_episode): LaMer 3-episode flow -> success_at1/2/3 (runs once per task).

Usage::

    python3 -m tbmf.webshop.train.train_grpo rllm/backend=verl
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

try:
    from ..eval.webshop_eval import webshop_evaluator
    from ..eval.webshop_lamer_eval import webshop_lamer_evaluator
    from ..flow.webshop_flow import webshop_flow
    from ..flow.webshop_lamer_flow import webshop_lamer_flow
except (ImportError, ValueError):
    from eval.webshop_eval import webshop_evaluator
    from eval.webshop_lamer_eval import webshop_lamer_evaluator
    from flow.webshop_flow import webshop_flow
    from flow.webshop_lamer_flow import webshop_lamer_flow

try:
    from .multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass
except (ImportError, ValueError):
    from multi_pass import MultiPassConfig, MultiPassEvaluator, MultiPassFlow, ValidationPass

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer


def _build_multi_pass(config: DictConfig):
    val_cfg = config.get("rllm", {}).get("validation", {}).get("passes", {})
    single_ep_enabled = val_cfg.get("single_episode", {}).get("enabled", True)
    multi_ep_enabled = val_cfg.get("multi_episode", {}).get("enabled", True)

    mp_config = MultiPassConfig(
        train_flow=webshop_flow,
        train_evaluator=webshop_evaluator,
        val_passes=[
            ValidationPass("single_episode", webshop_flow, webshop_evaluator, enabled=single_ep_enabled),
            ValidationPass("multi_episode", webshop_lamer_flow, webshop_lamer_evaluator, enabled=multi_ep_enabled, sample_budget=1),
        ],
    )
    return MultiPassFlow(mp_config), MultiPassEvaluator(mp_config)


def _shutdown_webshop_pool():
    try:
        try:
            from ..webshop_env import WebShopEnv
        except (ImportError, ValueError):
            from webshop_env import WebShopEnv
        WebShopEnv.shutdown_pool()
    except Exception:
        pass


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("webshop", "train")
    val_dataset = DatasetRegistry.load_dataset("webshop", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("WebShop dataset not found. Run: python3 tbmf/webshop/prepare_webshop_data.py")

    flow, evaluator = _build_multi_pass(config)

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=flow,
        evaluator=evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    try:
        trainer.train()
    finally:
        _shutdown_webshop_pool()


if __name__ == "__main__":
    main()
