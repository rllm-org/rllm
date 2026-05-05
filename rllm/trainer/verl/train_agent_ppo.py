"""
Copyright under Agentica Project.

Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available

from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env


@hydra.main(config_path="../config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    run_ppo_agent(config)


def run_ppo_agent(config):
    # Check if Ray is not initialized
    if not ray.is_initialized():
        from rllm.trainer.ray_init_utils import get_ray_init_settings

        ray_init_settings = get_ray_init_settings(config)
        ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

    runner_cls = ray.remote(num_cpus=1)(TaskRunner)

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if is_cuda_available and config.trainer.get("profile_steps") is not None and len(config.trainer.get("profile_steps", [])) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = runner_cls.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = runner_cls.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # use new model engine implementation
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            # NOTE: In new model engine, ref policy and actor rollout are in same ActorRolloutRefWorker,
            # while in legacy model engine, ref policy is in a separate ActorRolloutRefWorker.
            if need_reference_policy(config) and not ref_in_actor:
                role = Role.ActorRolloutRef
            else:
                role = Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        # Note: sync mode validation is now handled in RolloutConfig.__post_init__
        # Always use async worker since sync mode is deprecated and rejected
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                # we don't need to specialize critic worker. Just use TrainingWorker
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            # TODO: switch this to TrainingWorker as well
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.megatron_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
                print("Using new worker implementation")
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        # Ref policy has been fused into ActorRolloutRefWorker in new model engine,
        # we don't need to add a separate ref policy worker group.
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            return

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def run(
        self,
        config,
        workflow_class=None,
        workflow_args=None,
        agent_run_func=None,
    ):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config))
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        # TODO(listar2000): in rLLM, we skip the reward worker setup and assume a simple verified reward.
        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
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

        # if config.rllm.workflow.use_workflow:
        if agent_run_func is not None:
            print("IMPORTANT: Using AgentSdkTrainer")
            from rllm.trainer.verl.agent_sdk_trainer import AgentSdkTrainer

            trainer = AgentSdkTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                agent_run_func=agent_run_func,
            )
        elif workflow_class is not None:
            workflow_args = workflow_args or {}
            if config.rllm.workflow.get("workflow_args") is not None:
                for key, value in config.rllm.workflow.get("workflow_args").items():
                    if value is not None:
                        if key in workflow_args and isinstance(workflow_args[key], dict):
                            workflow_args[key].update(value)
                        else:
                            workflow_args[key] = value

            trainer = AgentWorkflowPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                workflow_class=workflow_class,
                workflow_args=workflow_args,
            )

        else:
            raise ValueError(
                "TaskRunner.run requires one of agent_run_func, workflow_class. "
                "The legacy agent_class + env_class path (driven by AgentExecutionEngine) "
                "has been removed; port your agent to a Workflow or AgentFlow."
            )

        # Apply NCCL dynamic batch sync patch (fixes verl#5750)
        from rllm.experimental.verl.patch import patch_verl_dynamic_batch_sync

        patch_verl_dynamic_batch_sync()

        trainer.init_workers()
        try:
            trainer.fit_agent()
        finally:
            trainer.shutdown()


if __name__ == "__main__":
    main()
