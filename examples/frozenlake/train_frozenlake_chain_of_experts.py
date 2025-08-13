import hydra
import ray
from omegaconf import DictConfig

from rllm.agents.frozenlake_multi_agent import (
    FrozenLakeProposerAgent,
    FrozenLakeExpertAgent, 
    FrozenLakeJudgeAgent,
)
from rllm.data import DatasetRegistry
from rllm.engine.multi_agent_execution_engine import (
    AgentConfig,
    AgentRole,
    ChainOfExpertsWorkflow
)
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.environments.multi_agent_env import MultiAgentEnv
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
from rllm.trainer.verl.multi_agent_ppo_trainer import (
    MultiAgentPPOTrainer,
    create_chain_of_experts_trainer
)


class FrozenLakeChainOfExpertsEnv(MultiAgentEnv):
    """
    FrozenLake environment Chain of Experts.
    Inherits from MultiAgentEnv manages context between agents.
    """
    
    def __init__(self, **kwargs):
        self.base_env = FrozenLakeEnv(**kwargs)
        super().__init__()
    
    def _create_observation(self):
        obs, _ = self.base_env.reset()
        return obs
    
    def _create_info(self):
        _, info = self.base_env.reset()
        return info
    
    def reset(self):
        observation, info = self.base_env.reset()
        self.multi_agent_context = {}
        self.agent_history = []
        return observation, info
    
    def reset_with_input(self, agent_input):
        super().reset_with_input(agent_input)
        observation, info = self.base_env.reset()
        return observation, info
    
    def step(self, action):
        return self.base_env.step(action)
    
    def render(self, *args, **kwargs):
        return self.base_env.render(*args, **kwargs)
    
    def finished(self):
        return self.base_env.finished()
    
    def success(self):
        return self.base_env.success()
    
    @classmethod
    def from_dict(cls, env_dict):
        return cls(**env_dict)
    
    def __getattr__(self, name):
        return getattr(self.base_env, name)


def create_frozenlake_chain_of_experts_agents(config):
    from rllm.engine.multi_agent_execution_engine import AgentConfig, AgentRole
    from rllm.agents.frozenlake_multi_agent import (
        FrozenLakeProposerAgent,
        FrozenLakeExpertAgent, 
        FrozenLakeJudgeAgent
    )
    
    multi_agent_section = config.get("multi_agent", {})
    
    agent_configs = [
        AgentConfig(
            agent_id="proposer",
            agent_class=FrozenLakeProposerAgent,
            agent_args={"max_steps": config.agent.max_steps},
            role=AgentRole.PROPOSER,
            temperature=0.7,
            top_p=0.9,
        ),
        AgentConfig(
            agent_id="expert", 
            agent_class=FrozenLakeExpertAgent,
            agent_args={"max_steps": config.agent.max_steps},
            role=AgentRole.SPECIALIST,
            temperature=0.5,
            top_p=0.8,
        ),
        AgentConfig(
            agent_id="judge",
            agent_class=FrozenLakeJudgeAgent,
            agent_args={"max_steps": config.agent.max_steps},
            role=AgentRole.JUDGE,
            temperature=0.3,
            top_p=0.7,
        )
    ]
    
    return agent_configs


@ray.remote(num_cpus=1)
def train_frozenlake_chain_of_experts(config, agent_class=None, env_class=None, agent_args=None, env_args=None):
    """
    Multi-agent training function that sets up all required infrastructure.
    """
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.trainer.ppo.reward import load_reward_manager

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.critic.strategy in ["fsdp", "fsdp2"]
        
        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    if env_class is None:
        env_class = FrozenLakeChainOfExpertsEnv
    if agent_class is None:
        agent_class = FrozenLakeProposerAgent

    env_args = env_args or {}
    agent_args = agent_args or {}
    if config.env.get("env_args") is not None:
        env_args.update(config.env.get("env_args"))
    if config.agent.get("agent_args") is not None:
        agent_args.update(config.agent.get("agent_args"))

    base_trainer = AgentPPOTrainer(
        config=config,
        tokenizer=tokenizer,
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

    agent_configs = create_frozenlake_chain_of_experts_agents(config)
    
    for agent_config in agent_configs:
        print(f"  - {agent_config.agent_id} ({agent_config.role.value}): {agent_config.agent_class.__name__}")

    multi_agent_config = {
        "training_mode": "final_agent",  
        "reward_aggregation": "final_agent",
    }

    trainer = create_chain_of_experts_trainer(
        base_trainer=base_trainer,
        agent_configs=agent_configs,
        multi_agent_config=multi_agent_config
    )
    
    trainer.init_workers()
    trainer.fit_multi_agent()
    
    print("Training completed!")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    """Main function that starts the Ray-based training"""
    
    print("Starting FrozenLake Chain of Experts Training")
    print("=" * 60)
    
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    ray.get(train_frozenlake_chain_of_experts.remote(
        config, 
        agent_class=FrozenLakeProposerAgent, 
        env_class=FrozenLakeChainOfExpertsEnv,
        agent_args={},
        env_args={}
    ))


if __name__ == "__main__":
    main() 