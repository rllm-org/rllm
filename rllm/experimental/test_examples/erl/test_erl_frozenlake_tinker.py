"""ERL FrozenLake training example — **Tinker backend**.

Reproduces the FrozenLake experiment from:
  Shi et al., "Experiential Reinforcement Learning", 2026.

Launch via the companion shell script ``tmp/test_erl_frozenlake_tinker.sh``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from rllm.experimental.erl import DEFAULT_ERL_ADV_ESTIMATOR_MAP, ErlConfig, ErlWorkflow
from rllm.experimental.test_examples.erl.frozen_lake_utils import (
    ERL_FROZENLAKE_SYMBOLS,
    ERL_FROZENLAKE_SYSTEM_PROMPT,
    FROZENLAKE_UPDATER_PROMPT,
    frozenlake_feedback_fn,
    frozenlake_state_builder_fn,
    make_solver_fn,
    prepare_frozenlake_datasets,
)
from rllm.experimental.unified_trainer import AgentTrainer
from rllm.workflows.store import InMemoryStore


@hydra.main(
    config_path="pkg://rllm.experimental.config",
    config_name="unified",
    version_base=None,
)
def main(config: DictConfig):
    train_dataset, test_dataset = prepare_frozenlake_datasets()

    erl_config = ErlConfig(
        initial_system_prompt=ERL_FROZENLAKE_SYSTEM_PROMPT,
        updater_system_prompt=FROZENLAKE_UPDATER_PROMPT,
        updater_sampling_params={"temperature": 0.7, "top_p": 0.9},
        train_first_attempt=True,
        train_second_attempt=False,
        train_distilled=False,
        train_updater=False,
        success_reward_threshold=1.0,
    )

    store = InMemoryStore()

    trainer = AgentTrainer(
        workflow_class=ErlWorkflow,
        workflow_args={
            "solver_fn": make_solver_fn(
                max_steps=4,
                env_max_steps=4,
                symbol_map=ERL_FROZENLAKE_SYMBOLS,
            ),
            "feedback_fn": frozenlake_feedback_fn,
            "state_builder_fn": frozenlake_state_builder_fn,
            "erl_config": erl_config,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="tinker",
        traj_group_adv_estimator_map=DEFAULT_ERL_ADV_ESTIMATOR_MAP,
        store=store,
    )
    trainer.train()


if __name__ == "__main__":
    main()
