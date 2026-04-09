"""GSD math training example — Tinker backend.

Trains a model on DeepScaleR math problems using Generalized
Self-Distillation (new SFT-style CE + IS + hint GRPO variant).

Launch via the companion shell script ``tmp/gsd/test_gsd_math_tinker.sh``.
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
    GsdWorkflow,
    build_gsd_estimator_map,
    gsd_transform_trajectory_groups_to_datums,
    make_gsd_grouping_hook,
)
from rllm.experimental.test_examples.gsd.math.utils import (
    gsd_math_reward,
    prepare_deepscaler_datasets,
)
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

    # GSD manages its own N rollouts per task — the Tinker group_size must be 1
    # so the engine dispatches one task at a time to the workflow.
    OmegaConf.update(config, "training.group_size", 1, force_add=True)
    OmegaConf.update(config, "validation.group_size", 1, force_add=True)

    train_dataset, test_dataset = prepare_deepscaler_datasets()

    gsd_config = GsdConfig(
        N=5,
        N_val=2,
        train_hint=True,
        success_reward_threshold=0.5,
        kl_coeff=1.0,
        kl_clip_min=-5.0,
        kl_clip_max=5.0,
        retrieval_k=3,
        distill_only=True,
    )

    # save_dir = config.training.get("default_local_dir", "/tmp/rllm-gsd-checkpoints")
    # experience_store = EmbeddingExperienceStore(
    #     device="cpu",
    #     max_size=500,
    #     save_path=f"{save_dir}/experience_store.json",
    #     autosave_every=20,
    # )

    # Shared across all workflow instances — pins the initial sampling
    # client as the frozen reference teacher on first access.
    teacher_ref = FrozenTeacherRef()

    # Optional shared batching primitive for cross-task teacher logprob
    # evaluation.  Not strictly required by the new IS pipeline
    # (``compute_logprobs_async`` is cheap enough for per-task gathers),
    # but can still help at very high concurrency.
    # scoring_accumulator = ScoringAccumulator(batch_interval=0.05, batch_threshold=32)

    trainer = AgentTrainer(
        workflow_class=GsdWorkflow,
        workflow_args={
            "reward_fn": gsd_math_reward,
            "gsd_config": gsd_config,
            "teacher_ref": teacher_ref,
            # "experience_store": experience_store,
            # "scoring_accumulator": scoring_accumulator,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
        # Per-role loss dispatch via the estimator map.
        traj_group_adv_estimator_map=build_gsd_estimator_map(train_hint=gsd_config.train_hint),
        # Custom datum builder: per-role dispatch into gsd_ce / gsd_is /
        # gsd_grpo / gsd_hint.
        transform_fn=gsd_transform_trajectory_groups_to_datums,
        # Custom grouping hook: collapses per-task gsd_hint groups into
        # one cross-task group so REINFORCE's baseline has something to
        # average over.
        traj_grouping_hook=make_gsd_grouping_hook(),
    )
    trainer.train()


if __name__ == "__main__":
    main()
