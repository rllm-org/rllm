import logging

import ray
from omegaconf import DictConfig

from rllm.data import Dataset
from rllm.experimental.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.experimental.verl.verl_backend import VerlBackend
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
from rllm.trainer.verl.train_agent_ppo import TaskRunner
from rllm.workflows.workflow import Workflow

logger = logging.getLogger(__name__)


# TODO(listar2000): when later deprecating `train_agent_ppo`, need to migrate all the logic here to `WorkflowTaskRunner`
@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class WorkflowTaskRunner(TaskRunner):
    """Ray remote class for executing distributed PPO training with the unified trainer.

    Inherits worker setup logic from `rllm.trainer.verl.train_agent_ppo.TaskRunner`
    and overrides `run` to use the `UnifiedTrainer` with `VerlBackend`.
    """

    def run(self, config, workflow_class: type[Workflow], workflow_args: dict, **kwargs):  # type: ignore
        """Execute the main PPO training workflow using the unified trainer.

        Args:
            config: Training configuration
            workflow_class: Workflow class
            workflow_args: Workflow arguments
        """
        import os
        import socket
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.config import validate_config
        from verl.utils.fs import copy_to_local

        print(f"WorkflowTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config))
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)

        # Force the new EngineWorker path before worker classes are selected.
        # VerlBackend (UnifiedTrainer) requires use_legacy_worker_impl='disable';
        # this must happen before add_actor_rollout_worker() reads the value.
        legacy_mode = config.trainer.get("use_legacy_worker_impl", "auto")
        if legacy_mode != "disable":
            logger.warning(f"VerlBackend requires use_legacy_worker_impl='disable' (new EngineWorker path), got '{legacy_mode}'. Overriding to 'disable'.")
            config.trainer.use_legacy_worker_impl = "disable"

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            **config.reward_model.get("reward_kwargs", {}),
        )
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # Assemble backend-specific arguments for initializing the verl backend.
        backend_args = {
            "tokenizer": tokenizer,
            "processor": processor,
            "role_worker_mapping": self.role_worker_mapping,
            "resource_pool_manager": resource_pool_manager,
            "ray_worker_group_cls": ray_worker_group_cls,
            "reward_fn": reward_fn,
            "val_reward_fn": val_reward_fn,
        }

        trainer = None
        try:
            trainer = UnifiedTrainer(
                backend_cls=VerlBackend,
                config=config,
                workflow_class=workflow_class,
                train_dataset=None,
                val_dataset=None,
                workflow_args=workflow_args,
                backend_args=backend_args,
                **kwargs,
            )
            trainer.fit()
        except Exception as e:
            print(f"Error training Verl: {e}")
            raise e
        finally:
            if trainer is not None:
                trainer.shutdown()


class VerlTrainerLauncher(TrainerLauncher):
    """
    Verl trainer launcher that handles the necessary setup for the verl backend.
    """

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        **kwargs,
    ):
        """Initialize the VerlTrainerLauncher. The heavy lifting is done in the `run` method of the `TaskRunner` class."""
        super().__init__(config, workflow_class, train_dataset, val_dataset, workflow_args, **kwargs)

        # For Verl specifically, the datasets are not passed directly to the backend, which instead relies on the data paths
        # being set in the config. TODO(listar2000): check whether this can be deprecated in favor of a more standard approach.
        if train_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.train_files = train_dataset.get_verl_data_path()
        if val_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.val_files = val_dataset.get_verl_data_path()

    def train(self):
        if not ray.is_initialized():
            from rllm.trainer.ray_init_utils import get_ray_init_settings

            ray_init_settings = get_ray_init_settings(self.config)
            ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

        runner = WorkflowTaskRunner.remote()  # type: ignore

        ray.get(
            runner.run.remote(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                store=self.store,
                **self.kwargs,
            )
        )
