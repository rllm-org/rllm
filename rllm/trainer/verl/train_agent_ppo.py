# Copyright under Agentica Project.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING, WORKFLOW_CLASS_MAPPING
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer

# Local application imports
from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available


@hydra.main(config_path="../config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    run_ppo_agent(config)


def run_ppo_agent(config):
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if is_cuda_available and config.trainer.get("profile_steps") is not None and len(config.trainer.get("profile_steps", [])) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config, workflow_class=None, workflow_args=None, agent_class=None, env_class=None, agent_args=None, env_args=None):
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
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_tokenizer
        from verl.utils.tokenizer import hf_processor

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Enable processor for multimodal models
        processor = None
        if config.get("multimodal", {}).get("enable", False):
            try:
                processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
                print(f"Multimodal processor loaded: {type(processor)}")
            except Exception as e:
                print(f"Failed to load multimodal processor: {e}")
                print("Falling back to text-only training")

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
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
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

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

        if config.rllm.workflow.use_workflow:
            if workflow_class is None:
                workflow_class = WORKFLOW_CLASS_MAPPING[config.rllm.workflow.name]
            workflow_args = workflow_args or {}
            if config.rllm.workflow.get("workflow_args") is not None:
                workflow_args.update(config.rllm.workflow.get("workflow_args"))

            trainer = AgentWorkflowPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                workflow_class=workflow_class,
                workflow_args=workflow_args,
            )

        else:
            if env_class is None:
                env_class = ENV_CLASS_MAPPING[config.rllm.env.name]
            if agent_class is None:
                agent_class = AGENT_CLASS_MAPPING[config.rllm.agent.name]

            env_args = env_args or {}
            agent_args = agent_args or {}
            if config.rllm.env.get("env_args") is not None:
                env_args.update(config.rllm.env.get("env_args"))
            if config.rllm.agent.get("agent_args") is not None:
                agent_args.update(config.rllm.agent.get("agent_args"))

            trainer = AgentPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                env_class=env_class,
                agent_class=agent_class,
                env_args=env_args,
                agent_args=agent_args,
            )

        trainer.init_workers()
        try:
            trainer.fit_agent()
        finally:
            trainer.shutdown()


if __name__ == "__main__":
    main()
