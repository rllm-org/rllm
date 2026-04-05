import ray
from omegaconf import DictConfig

from rllm.data import Dataset
from rllm.experimental.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.experimental.verl.verl_backend import VerlBackend
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
from rllm.trainer.verl.train_agent_ppo import TaskRunner as _BaseTaskRunner
from rllm.workflows.workflow import Workflow


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class VerlTaskRunner(_BaseTaskRunner):
    """Ray remote class for executing training with the unified trainer."""

    def run(self, config, workflow_class: type[Workflow], workflow_args: dict, **kwargs):  # type: ignore
        import os
        import socket
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.trainer.ppo.utils import need_reference_policy
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.config import validate_config
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config))
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)

        is_separated = config.rllm.get("async_training", {}).get("enable", False)

        # Propagate rllm algorithm config to verl actor config.
        # Must happen before worker mapping creation since verl's
        # need_reference_policy() reads from the verl config.
        from rllm.experimental.verl.utils import propagate_rllm_to_verl_config
        propagate_rllm_to_verl_config(config)

        # --- Worker mapping and resource pools ---
        # Follows verl's TaskRunner.run() for colocated,
        # verl's FullyAsyncTaskRunner._initialize_components() for separated.
        if is_separated:
            from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
            from verl.trainer.ppo.utils import Role

            # Propagate rollout GPU config into actor_rollout_ref (verl convention)
            config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
            config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

            role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)

            # Trainer resource pool: all roles except Rollout
            # (Rollout servers are launched by AgentLoopManager in standalone mode)
            trainer_roles = {r: cls for r, cls in role_worker_mapping.items() if r != Role.Rollout}
            resource_pool_manager = create_resource_pool_manager(config, roles=list(trainer_roles.keys()))
        else:
            actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
            self.add_ref_policy_worker(config, actor_rollout_cls)

            trainer_roles = self.role_worker_mapping
            resource_pool_manager = self.init_resource_pool_mgr(config)

        # --- Config validation ---
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=False,
        )

        # --- Model, tokenizer, processor ---
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # --- Build backend args ---
        backend_args = {
            "tokenizer": tokenizer,
            "processor": processor,
            "role_worker_mapping": trainer_roles if is_separated else self.role_worker_mapping,
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
    """Verl trainer launcher."""

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(config, workflow_class, train_dataset, val_dataset, workflow_args, **kwargs)

        if train_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.train_files = train_dataset.get_verl_data_path()
        if val_dataset is not None and self.config is not None and hasattr(self.config, "data"):
            self.config.data.val_files = val_dataset.get_verl_data_path()

    def train(self):
        if not ray.is_initialized():
            from rllm.trainer.ray_init_utils import get_ray_init_settings

            ray_init_settings = get_ray_init_settings(self.config)
            ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

        runner = VerlTaskRunner.remote()
        ray.get(
            runner.run.remote(
                config=self.config,
                workflow_class=self.workflow_class,
                workflow_args=self.workflow_args,
                store=self.store,
                **self.kwargs,
            )
        )
