from typing import Any

import hydra
import verifiers as vf
from omegaconf import OmegaConf

from examples.verifiers_env.workflow import VerifiersWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    # Access custom top-level config key with defaults
    vf_env_id = OmegaConf.select(config, "verifiers.env_id", default="DefaultEnv")
    vf_env_args_raw = OmegaConf.select(config, "verifiers.env_args", default=None)

    vf_env_args: dict[str, Any] = (
        OmegaConf.to_container(vf_env_args_raw, resolve=True)  # type: ignore[assignment]
        if vf_env_args_raw is not None
        else {}
    )

    # Get sampling args for verifiers rollouts
    vf_sampling_args_raw = OmegaConf.select(
        config, "verifiers.sampling_args", default=None
    )
    vf_sampling_args: dict[str, Any] | None = (
        OmegaConf.to_container(vf_sampling_args_raw, resolve=True)  # type: ignore[assignment]
        if vf_sampling_args_raw is not None
        else None
    )

    vf_env = vf.load_environment(vf_env_id, **vf_env_args)

    # Registering during train time since for some environments, creating some datasets for verifiers environments needs resources to spin up.

    if DatasetRegistry.dataset_exists(vf_env_id, "train"):
        train_dataset = DatasetRegistry.load_dataset(vf_env_id, "train")
        test_dataset = DatasetRegistry.load_dataset(vf_env_id, "test")
    else:
        vf_train_dataset = vf_env.get_dataset()
        vf_eval_dataset = vf_env.get_eval_dataset()

        train_dataset = DatasetRegistry.register_dataset(
            vf_env_id, vf_train_dataset, "train"
        )

        test_dataset = DatasetRegistry.register_dataset(
            vf_env_id, vf_eval_dataset, "test"
        )

    # Get backend from config (default: verl)
    backend = OmegaConf.select(config, "backend", default="verl")

    trainer = AgentTrainer(
        workflow_class=VerifiersWorkflow,
        workflow_args={
            "vf_env": vf_env,
            "sampling_args": vf_sampling_args,
        },
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        config=config,
        backend=backend,
    )
    trainer.train()


if __name__ == "__main__":
    main()
