"""GRPO MiniSweeper training with multi-pass validation.

Train: single-episode GRPO rollouts.
Val pass 1 (single_episode): standard GRPO flow -> pass@1/pass@4.
Val pass 2 (multi_episode): LaMer 3-episode flow -> success_at1/2/3 (runs once per task).

Usage::

    python3 -m tbmf.minisweeper.train.train_grpo rllm/backend=verl
"""

import hydra
from omegaconf import DictConfig

try:
    from ..eval.minisweeper_eval import minisweeper_evaluator
    from ..eval.minisweeper_lamer_eval import minisweeper_lamer_evaluator
    from ..flow.minisweeper_flow import minisweeper_flow
    from ..flow.minisweeper_lamer_flow import minisweeper_lamer_flow
except (ImportError, ValueError):
    from eval.minisweeper_eval import minisweeper_evaluator
    from eval.minisweeper_lamer_eval import minisweeper_lamer_evaluator
    from flow.minisweeper_flow import minisweeper_flow
    from flow.minisweeper_lamer_flow import minisweeper_lamer_flow

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
        train_flow=minisweeper_flow,
        train_evaluator=minisweeper_evaluator,
        val_passes=[
            ValidationPass("single_episode", minisweeper_flow, minisweeper_evaluator, enabled=single_ep_enabled),
            ValidationPass("multi_episode", minisweeper_lamer_flow, minisweeper_lamer_evaluator, enabled=multi_ep_enabled, sample_budget=1),
        ],
    )
    return MultiPassFlow(mp_config), MultiPassEvaluator(mp_config)


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("minisweeper", "train")
    val_dataset = DatasetRegistry.load_dataset("minisweeper", "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError("MiniSweeper dataset not found. Run: python3 tbmf/minisweeper/prepare_minisweeper_data.py")

    flow, evaluator = _build_multi_pass(config)

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=flow,
        evaluator=evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
