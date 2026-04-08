"""GSD countdown training example — Tinker backend.

Trains on the countdown number puzzle using Generalized Self-Distillation
with an evolving hint pool.  Unlike the math example (per-problem hints),
countdown uses a pool of generic strategies selected via UCB1 and evolved
periodically by an LLM.

Launch via ``tmp/gsd/test_gsd_countdown_tinker.sh``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from rllm.experimental.gsd.hint_pool import HintPool
from rllm.experimental.gsd.losses import build_gsd_estimator_map
from rllm.experimental.gsd.transform import gsd_transform_trajectory_groups_to_datums
from rllm.experimental.gsd.workflow import GsdConfig
from rllm.experimental.test_examples.gsd.countdown_utils import (
    COUNTDOWN_SEED_HINTS,
    countdown_reward,
    prepare_countdown_datasets,
)
from rllm.experimental.test_examples.gsd.countdown_workflow import GsdCountdownWorkflow
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.experimental.config",
    config_name="unified",
    version_base=None,
)
def main(config: DictConfig):
    # GSD manages its own N rollouts per task
    OmegaConf.update(config, "training.group_size", 1, force_add=True)
    OmegaConf.update(config, "validation.group_size", 1, force_add=True)

    train_dataset, test_dataset = prepare_countdown_datasets()

    gsd_config = GsdConfig(
        N=8,
        N_val=2,
        distill_topk=20,
        train_hint=False,
        success_reward_threshold=1.0,
        kl_coeff=1.0,
        kl_clip_min=-5.0,
        kl_clip_max=5.0,
    )

    save_dir = config.training.get("default_local_dir", "/tmp/rllm-gsd-checkpoints")
    hint_pool = HintPool(
        max_size=5,
        ema_alpha=0.3,
        ucb_c=1.0,
        evolve_every=16,
        seed_hints=COUNTDOWN_SEED_HINTS,
        save_path=f"{save_dir}/hint_pool.json",
        autosave_every=32,
    )

    trainer = AgentTrainer(
        workflow_class=GsdCountdownWorkflow,
        workflow_args={
            "reward_fn": countdown_reward,
            "gsd_config": gsd_config,
            "hint_pool": hint_pool,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
        traj_group_adv_estimator_map=build_gsd_estimator_map(train_hint=False),
        transform_fn=gsd_transform_trajectory_groups_to_datums,
    )
    trainer.train()


if __name__ == "__main__":
    main()
