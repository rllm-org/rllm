"""GSD math training example — Tinker backend.

Trains a model on DeepScaleR math problems using Generalized
Self-Distillation: hint-conditioned pseudo-teacher provides dense
token-level supervision via reverse KL (on-policy) and forward KL
(supervised) losses, with GRPO fallback when the hint doesn't help.

Launch via the companion shell script ``tmp/gsd/test_gsd_math_tinker.sh``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from rllm.experimental.gsd.experience_store import EmbeddingExperienceStore
from rllm.experimental.gsd.losses import build_gsd_estimator_map
from rllm.experimental.gsd.transform import gsd_transform_trajectory_groups_to_datums
from rllm.experimental.gsd.workflow import GsdConfig, GsdWorkflow
from rllm.experimental.test_examples.gsd.math_utils import gsd_math_reward, prepare_deepscaler_datasets
from rllm.experimental.unified_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.experimental.config",
    config_name="unified",
    version_base=None,
)
def main(config: DictConfig):
    # GSD manages its own N rollouts per task — the Tinker group_size must be 1
    # so the engine dispatches one task at a time to the workflow.
    OmegaConf.update(config, "training.group_size", 1, force_add=True)
    OmegaConf.update(config, "validation.group_size", 1, force_add=True)

    train_dataset, test_dataset = prepare_deepscaler_datasets()

    gsd_config = GsdConfig(
        N=5,
        N_val=2,
        distill_topk=20,
        train_hint=False,
        success_reward_threshold=0.5,
        kl_coeff=1.0,
        kl_clip_min=-5.0,
        kl_clip_max=5.0,
        retrieval_k=3,
    )

    experience_store = EmbeddingExperienceStore(max_size=500)

    trainer = AgentTrainer(
        workflow_class=GsdWorkflow,
        workflow_args={
            "reward_fn": gsd_math_reward,
            "gsd_config": gsd_config,
            "experience_store": experience_store,
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
