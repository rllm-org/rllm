# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray

from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING, setup_environment
from rllm.trainer.verl.agent_ppo_trainer_pipeline import PipelineAgentPPOTrainer

# Local application imports
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.reward_manager import NaiveRewardManager


@hydra.main(config_path="../config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    run_ppo_agent_async(config)


def run_ppo_agent_async(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    actor_pool_id = "actor_pool"
    rollout_pool_id = "rollout_pool"
    num_training_gpus = config.trainer.n_training_gpus_per_node
    resource_pool_spec = {
        actor_pool_id: [num_training_gpus] * config.trainer.nnodes,
        rollout_pool_id: [config.trainer.n_gpus_per_node - num_training_gpus] * config.trainer.nnodes,
    }
    mapping = {
        Role.Actor: actor_pool_id,
        Role.Rollout: rollout_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    val_reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    role_worker_mapping = {Role.Actor: ray.remote(ActorRolloutRefWorker), Role.Rollout: ray.remote(max_concurrency=512)(ActorRolloutRefWorker)}

    # Below are agent specific initialization
    env_class = ENV_CLASS_MAPPING[config.rllm.env.name]
    agent_class = AGENT_CLASS_MAPPING[config.rllm.agent.name]
    setup_environment(config)

    trainer = PipelineAgentPPOTrainer(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=RayWorkerGroup, reward_fn=reward_fn, val_reward_fn=val_reward_fn, env_class=env_class, agent_class=agent_class)

    trainer.init_workers()
    trainer.fit_agent()


if __name__ == "__main__":
    main()
