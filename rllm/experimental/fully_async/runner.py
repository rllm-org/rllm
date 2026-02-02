# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import os
import random
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_main import (
    create_resource_pool_manager,
    create_role_worker_mapping,
)
from rllm.experimental.fully_async.async_trainer import AsyncTrainer
from rllm.experimental.fully_async.fully_async_rollouter import FullyAsyncRollouter
from rllm.experimental.fully_async.message_queue import MessageQueue, MessageQueueClient
from rllm.experimental.fully_async.param_sync import ParameterSynchronizer
from rllm.experimental.fully_async.protocol import Trajectory
from rllm.experimental.fully_async.rollout_engine import RolloutExecutor
from rllm.experimental.fully_async.utils import calculate_max_concurrency
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.utils.fs import copy_to_local


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        print("[ASYNC MAIN] Creating FullyAsyncRollouter...")
        self._create_rollouter(config)

        print("[ASYNC MAIN] Creating FullyAsyncTrainer...")
        self._create_trainer(config)

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # max_queue_size
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
        ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

        print("[ASYNC MAIN] Setting up parameter synchronization...")

        param_synchronizer = ParameterSynchronizer.remote(
            config=config,
            trainer=self.components["trainer"],
            rollouter=self.components["rollouter"],
            mq=self.components["message_queue_client"],
        )
        ray.get(self.components["trainer"].set_parameter_synchronizer.remote(param_synchronizer))

        # Create rollout executor BEFORE sync_weights (so it can be paused during sync)
        self._create_rollout_executor(config)

        # Set rollout_executor and router_url on param_synchronizer BEFORE sync_weights
        ray.get(param_synchronizer.set_rollout_executor.remote(self.components["rollout_executor"]))
        ray.get(param_synchronizer.set_router_url.remote(self.router_url))

        # load checkpoint and sync parameter before doing anything
        val_before_train = config.trainer.get("val_before_train", True)
        # param_version resume from ckpt or default 0
        param_version = ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())
        ray.get(
            param_synchronizer.sync_weights.remote(
                version=param_version,
                validate=val_before_train,
                use_trainer_do_validate=config.async_training.use_trainer_do_validate,
            )
        )
        ray.get(param_synchronizer.wait_last_valid.remote())

        self.components["param_synchronizer"] = param_synchronizer
        print("[ASYNC MAIN] All components initialized successfully")
        # set router url and msg client for the parameter syncer


    def _create_rollouter(self, config) -> None:
        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping={Role.Rollout: self.components["role_worker_mapping"][Role.Rollout]},
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        server_urls = ray.get(rollouter.get_server_urls.remote())
        print("Launched server urls: ", server_urls)

        self.router_url = ray.get(rollouter.launch_router.remote(server_urls))

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = AsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _create_rollout_executor(self, config):
        import time as time_module

        from verl.utils.reward_score import default_compute_score

        # Get overlong buffer config from reward_model.reward_kwargs
        reward_kwargs = config.reward_model.get("reward_kwargs", {})
        overlong_buffer_cfg = reward_kwargs.get("overlong_buffer_cfg", {})
        overlong_buffer_enabled = overlong_buffer_cfg.get("enable", False)
        overlong_buffer_len = overlong_buffer_cfg.get("len", 0)
        overlong_penalty_factor = overlong_buffer_cfg.get("penalty_factor", 1.0)
        max_resp_len = reward_kwargs.get("max_resp_len", config.data.max_response_length)

        async def rollout_fn(client, tokenizer, **kwargs):
            start_time = time_module.time()
            param_version_start = client.cur_version

            # Extract raw_prompt from dataset (chat format: [{'content': '...', 'role': 'user'}])
            # raw_prompt is ndarray shape (1,), raw_prompt[0] is the list of message dicts
            messages = kwargs["raw_prompt"][0]
            messages = [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}] + messages
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )

            # Get sampling params from config or use defaults
            sampling_params = {
                "temperature": config.actor_rollout_ref.rollout.get("temperature", 1.0),
                "max_new_tokens": config.actor_rollout_ref.rollout.get("max_new_tokens", 8192),
                "top_p": 1.0,
                "top_k": -1,
            }

            output = await client.generate(prompt_ids, sampling_params=sampling_params)

            # Capture timing and version info
            end_time = time_module.time()
            param_version_end = client.cur_version
            processing_time = end_time - start_time

            # Extract response_ids from output_chunks (OutputWithVersion protocol)
            # OutputWithVersion has output_chunks, each OutputChunk has response_ids
            response_ids = []
            for chunk in output.output_chunks:
                response_ids.extend(chunk.response_ids)

            # Decode the response for reward calculation
            response_str = tokenizer.decode(response_ids, skip_special_tokens=False)
            if random.random() < 0.001:
                prompt_str = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                print(f"[FullyAsyncRollouter DEBUG] Prompt: {prompt_str}")
                print(f"[FullyAsyncRollouter DEBUG] Response: {response_str}")

            # Extract ground_truth and data_source from kwargs for reward calculation
            # reward_model is ndarray shape (1,), reward_model[0] is the dict with ground_truth
            # data_source is ndarray shape (1,), data_source[0] is the string
            reward_model_info = kwargs["reward_model"][0]
            ground_truth = reward_model_info.get("ground_truth", "")
            data_source = kwargs["data_source"][0]

            # Compute reward using default_compute_score (same as DAPORewardManager)
            try:
                result = default_compute_score(
                    data_source=data_source,
                    solution_str=tokenizer.decode(response_ids, skip_special_tokens=True),
                    ground_truth=ground_truth,
                )
                if isinstance(result, dict):
                    score = result["score"]
                else:
                    score = float(result)
            except Exception as e:
                print(f"[RolloutFn] Error computing reward: {e}, using default score -1.0")
                score = -1.0

            reward = score

            # Apply overlong penalty (DAPO-specific feature)
            if overlong_buffer_enabled:
                valid_response_length = len(response_ids)
                expected_len = max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward

            # Store metadata for statistics tracking
            metadata = {
                "processing_time": processing_time,
                "param_version_start": param_version_start,
                "param_version_end": param_version_end,
                "param_version": param_version_end,  # The version used for this trajectory
                "is_partial": param_version_start != param_version_end,  # Was there a param update during generation?
                "tool_calls_time": 0.0,  # Placeholder for agent-based rollouts with tool calls
            }

            return Trajectory(sequences=[output.to_sequence()], reward=reward, metadata=metadata)

        # Calculate max_concurrent_tasks for the new simplified design
        # This controls how many generation tasks run concurrently
        # Staleness bound = max_concurrent_tasks + max_queue_size
        max_concurrent_tasks = calculate_max_concurrency(config)
        # Get n (number of trajectories per datum) from config, matching fully_async_rollouter.py
        n = config.actor_rollout_ref.rollout.get("n", 1)

        rollout_executor = RolloutExecutor.remote(
            router_url=self.router_url,
            rollout_fn=rollout_fn,
            n=n,
            message_queue_client=self.components["message_queue_client"],
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            max_concurrency=max_concurrent_tasks,
            total_rollout_steps=config.rollout.total_rollout_steps,
        )

        self.components["rollout_executor"] = rollout_executor

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollout_executor_future = self.components["rollout_executor"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollout_executor_future, trainer_future]

        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            asyncio.run(self.components["message_queue_client"].clear_queue())
            print("[ASYNC MAIN] Training completed or interrupted")


@hydra.main(config_path="../config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")
    from time import time

    start_time = time()
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
