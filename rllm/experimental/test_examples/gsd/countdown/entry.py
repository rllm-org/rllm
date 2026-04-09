"""GSD countdown training example — Tinker backend.

Trains on the countdown number puzzle using Generalized Self-Distillation
with an evolving hint pool.  Unlike the math example (per-problem hints),
countdown uses a pool of generic strategies selected via UCB1 and evolved
periodically by an LLM.

Launch via ``tmp/gsd/test_gsd_countdown_tinker.sh``.
"""

from __future__ import annotations

import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rllm.experimental.gsd import (
    FrozenTeacherRef,
    GsdConfig,
    HintPool,
    ScoringAccumulator,
    build_gsd_estimator_map,
    gsd_transform_trajectory_groups_to_datums,
    make_gsd_grouping_hook,
)
from rllm.experimental.test_examples.gsd.countdown.utils import (
    COUNTDOWN_SEED_HINTS,
    countdown_reward,
    prepare_countdown_datasets,
)
from rllm.experimental.test_examples.gsd.countdown.workflow import GsdCountdownWorkflow
from rllm.experimental.unified_trainer import AgentTrainer

# Fixed seed for reproducibility.  This seeds the torch / numpy / random
# RNGs used by:
#
# * ``torch.utils.data.DataLoader(shuffle=True)`` in ``TinkerBackend.get_dataloader``
#   — determines the per-epoch training-task ordering.
# * GSD helpers that use ``random`` / ``np.random`` (e.g. HintPool UCB1
#   tie-breaking, experience store sampling).
#
# Change this to vary the run; set it from ``config.training.seed`` if
# you want Hydra-driven sweeps.
SEED = 42


@hydra.main(
    config_path="pkg://rllm.experimental.config",
    config_name="unified",
    version_base=None,
)
def main(config: DictConfig):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # GSD manages its own N rollouts per task
    OmegaConf.update(config, "training.group_size", 1, force_add=True)
    OmegaConf.update(config, "validation.group_size", 1, force_add=True)

    train_dataset, test_dataset = prepare_countdown_datasets()

    gsd_config = GsdConfig(
        N=5,
        N_val=2,
        # Countdown uses pool-selected hints (not model-sampled), so hint
        # GRPO does not apply.  The workflow subclass forces this to False
        # regardless, but we set it explicitly here for clarity.
        train_hint=False,
        success_reward_threshold=0.5,
        kl_coeff=1.0,
        kl_clip_min=-2.0,
        kl_clip_max=2.0,
    )

    save_dir = config.training.get("default_local_dir", "/tmp/rllm-gsd-checkpoints")
    hint_pool = HintPool(
        max_size=5,
        ema_alpha=0.2,  # smoother score updates, less noise-driven
        ucb_c=0.2,  # exploitation-biased: improvement scores live in [-1,1],
        # so a large c lets the exploration bonus drown out real
        # score differences and the best hints rarely win
        evolve_every=16,
        seed_hints=COUNTDOWN_SEED_HINTS,
        save_path=f"{save_dir}/hint_pool.json",
        autosave_every=32,
        max_hard_solves=20,  # FIFO buffer capacity
        hard_solve_window=5,  # only feed the 5 most recent into evolution
    )

    teacher_ref = FrozenTeacherRef()
    scoring_accumulator = ScoringAccumulator(batch_interval=0.05, batch_threshold=64)

    trainer = AgentTrainer(
        workflow_class=GsdCountdownWorkflow,
        workflow_args={
            "reward_fn": countdown_reward,
            "gsd_config": gsd_config,
            "teacher_ref": teacher_ref,
            "hint_pool": hint_pool,
            "scoring_accumulator": scoring_accumulator,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
        traj_group_adv_estimator_map=build_gsd_estimator_map(train_hint=False),
        transform_fn=gsd_transform_trajectory_groups_to_datums,
        # Not strictly required since countdown has no hint trajectories,
        # but cheap and future-proof if we ever turn hint GRPO back on.
        traj_grouping_hook=make_gsd_grouping_hook(),
    )
    trainer.train()


if __name__ == "__main__":
    main()
