import os
import socket
from pprint import pprint

import ray
from omegaconf import DictConfig, OmegaConf
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local

from rllm.data import Dataset
from rllm.experimental.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.experimental.verl.verl_backend import VerlBackend
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
from rllm.workflows.workflow import Workflow


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Adapted directly from `rllm.trainer.verl.train_agent_ppo.TaskRunner`.
    """

    def run(self, config, workflow_class, workflow_args, **kwargs):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration
            workflow_class: Workflow class
            workflow_args: Workflow arguments
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Starting from verl 0.6.1, this cls has been standardized for both fsdp and megatron backends.
        ray_worker_group_cls = RayWorkerGroup
        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                # import warnings
                # warnings.warn(f"Legacy worker impl is going to be deprecated, will be removed in the future. \
                #   Please set trainer.use_legacy_worker_impl = false to switch to the new worker implementation.")
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        else:
            raise NotImplementedError

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Assemble backend-specific arguments for initializing the verl backend.
        backend_args = {
            "tokenizer": tokenizer,
            "processor": processor,
            "role_worker_mapping": role_worker_mapping,
            "resource_pool_manager": resource_pool_manager,
            "ray_worker_group_cls": ray_worker_group_cls,
            "reward_fn": reward_fn,
            "val_reward_fn": val_reward_fn,
        }

        trainer = None
        try:
            trainer = UnifiedTrainer(backend_cls=VerlBackend, config=config, workflow_class=workflow_class, train_dataset=None, val_dataset=None, workflow_args=workflow_args, backend_args=backend_args, **kwargs)
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
        workflow_class: type[Workflow],
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
            # read off all the `ray_init` settings from the config
            if self.config is not None and hasattr(self.config, "ray_init"):
                ray_init_settings = {k: v for k, v in self.config.ray_init.items() if v is not None}
            else:
                ray_init_settings = {}
            ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

        runner = TaskRunner.remote()  # type: ignore

        ray.get(
            runner.run.remote(  # type: ignore
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                **self.kwargs,
            )
        )
