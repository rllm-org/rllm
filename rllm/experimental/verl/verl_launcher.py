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


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class VerlTaskRunner(TaskRunner):
    """Ray remote class for executing training with the unified trainer."""

    def run(self, config, workflow_class: type[Workflow], workflow_args: dict, hydra_overrides: list[str] | None = None, **kwargs):  # type: ignore
        import os
        import socket
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.trainer.ppo.utils import need_reference_policy
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.config import validate_config
        from verl.utils.fs import copy_to_local

        from rllm.experimental.verl.utils import sync_config

        print(f"VerlTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        sync_config(config, hydra_overrides=hydra_overrides)
        OmegaConf.resolve(config)
        sync_config(config, hydra_overrides=hydra_overrides)
        config.trainer.use_legacy_worker_impl = "disable"
        pprint(OmegaConf.to_container(config))

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=False,
        )

        # Download the checkpoint from HDFS to the local machine.
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        # Assemble backend-specific arguments for initializing the verl backend.
        backend_args = {
            "tokenizer": tokenizer,
            "processor": processor,
            "role_worker_mapping": self.role_worker_mapping,
            "resource_pool_manager": resource_pool_manager,
            "ray_worker_group_cls": ray_worker_group_cls,
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

        # Capture Hydra CLI overrides while we're still in the Hydra-decorated
        # process; the Ray actor below cannot read HydraConfig itself.
        try:
            from hydra.core.hydra_config import HydraConfig

            hydra_overrides = list(HydraConfig.get().overrides.task)
        except (ValueError, AttributeError, ImportError):
            hydra_overrides = []

        runner = VerlTaskRunner.remote()  # type: ignore

        ray.get(
            runner.run.remote(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                store=self.store,
                hydra_overrides=hydra_overrides,
                **self.kwargs,
            )
        )
