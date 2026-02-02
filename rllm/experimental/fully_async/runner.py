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
import inspect
import os
import random
import socket
import threading
import time
from pprint import pprint

import ray
from omegaconf import OmegaConf

from rllm.experimental.fully_async.async_trainer import AsyncTrainer
from rllm.experimental.fully_async.fully_async_rollouter import FullyAsyncRollouter
from rllm.experimental.fully_async.message_queue import MessageQueue, MessageQueueClient
from rllm.experimental.fully_async.param_sync import ParameterSynchronizer
from rllm.experimental.fully_async.protocol import Trajectory
from rllm.experimental.fully_async.rollout_executor import RolloutExecutor
from rllm.experimental.fully_async.utils import calculate_max_concurrency
from verl.experimental.fully_async_policy.fully_async_main import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.utils.fs import copy_to_local


def create_task_runner_with_rollout_fn(rollout_fn):
    """
    Factory function that creates a FullyAsyncTaskRunner class with a custom rollout_fn baked in.

    This allows passing a custom rollout function to the TaskRunner without modifying run_ppo.

    Args:
        rollout_fn: Async function with signature:
            async def rollout_fn(client, tokenizer, **kwargs) -> Trajectory

    Returns:
        A Ray remote class configured with the custom rollout_fn
    """

    @ray.remote(num_cpus=1)
    class ConfiguredTaskRunner(FullyAsyncTaskRunner):
        _custom_rollout_fn = staticmethod(rollout_fn)

    return ConfiguredTaskRunner


class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    # Default rollout function - should be overridden by subclasses or set via factory
    _custom_rollout_fn = None

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
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
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

    def _create_rollout_executor(self, config):
        """Override to use the custom rollout_fn instead of the default."""
        max_concurrent_tasks = calculate_max_concurrency(config)
        n = config.actor_rollout_ref.rollout.get("n", 1)

        if self._custom_rollout_fn is None:
            raise ValueError("Use create_task_runner_with_rollout_fn to pass in a rollout_fn")

        if not inspect.iscoroutinefunction(self._custom_rollout_fn):
            raise TypeError(f"rollout_fn must be an async function (defined with 'async def'), but got {type(self._custom_rollout_fn).__name__}. Only async functions are supported.")

        rollout_executor = RolloutExecutor.remote(
            router_url=self.router_url,
            rollout_fn=self._custom_rollout_fn,
            n=n,
            message_queue_client=self.components["message_queue_client"],
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            max_concurrency=max_concurrent_tasks,
            total_rollout_steps=config.rollout.total_rollout_steps,
        )

        self.components["rollout_executor"] = rollout_executor

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
        trainer_role_mapping = {role: worker_cls for role, worker_cls in self.components["role_worker_mapping"].items() if role != Role.Rollout}

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


class AsyncAgentTrainer:
    def __init__(self, config, train_dataset, val_dataset, rollout_fn):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.rollout_fn = rollout_fn

    def train(self):
        from verl.trainer.main_ppo import run_ppo

        # Ensure async training config exists
        if not hasattr(self.config, "async_training"):
            raise RuntimeError("must set async_training config")

        start_time = time.time()

        # Create a configured TaskRunner class with rollout_fn baked in
        task_runner_class = create_task_runner_with_rollout_fn(self.rollout_fn)
        run_ppo(self.config, task_runner_class=task_runner_class)

        print(f"total time: {time.time() - start_time:.2f} seconds")
