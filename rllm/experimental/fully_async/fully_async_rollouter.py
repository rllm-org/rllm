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
import functools
import multiprocessing
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat

import numpy as np
import ray
import torch
from ray import ObjectRef

from rllm.experimental.fully_async.message_queue import MessageQueueClient
from rllm.experimental.fully_async.utils import abort_async, continue_generation_async
from verl.experimental.fully_async_policy.detach_utils import RolloutSample, ValidateMetrics, prepare_single_generation_data
from verl.experimental.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger


@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(FullyAsyncRayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        self.val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, "trigger_parameter_sync_step must larger than 1"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_critic = False
        self.use_reference_policy = False
        self.use_rm = False

        print("[FullyAsyncRollouter] Creating datasets...")
        from rllm.data.dataset import DatasetRegistry
        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        dataset_name = config.async_training.dataset_name
        train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
        val_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
        if train_dataset is None:
            raise ValueError(f"Failed to load dataset '{dataset_name}' from DatasetRegistry. Run create_rllm_dataset.py first.")
        if val_dataset is None:
            print(f"[FullyAsyncRollouter] Warning: No test split for dataset '{dataset_name}', using train split for validation")
            val_dataset = train_dataset
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        if self.config.async_training.use_trainer_do_validate:
            rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
            train_gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node
            total_gpus = rollout_gpus + train_gpus
            print(f"[FullyAsyncRollouter] split before val_dataset total len: {len(val_dataset)}")
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[:rollout_gpus]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            print(f"[FullyAsyncRollouter] split after val_dataset total len: {len(val_dataset)}")
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== fully async config ====================

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.async_training.required_samples
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.current_param_version = 0
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        # we start from step 1
        self.global_steps = 1
        self.idle_start_time = None
        self.version_start_time = None

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True
        self.monitor_loop_trigger = True

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Initialize async queues
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()
        self.cancel_queue = asyncio.Queue()

        cpu_cores = multiprocessing.cpu_count()
        # cpu case use cpu_cores; io case use cpu_cores*2
        self.validate_executor = ThreadPoolExecutor(max_workers=cpu_cores)
        self.parallel_validate_and_rollout = config.async_training.get("parallel_validate_and_rollout", False)
        self.validate_task = None

    def _init_async_objects(self):
        # Initialize asyncio synchronization primitives.
        # We let asyncio.Condition create the Lock internally to ensure they share the same Event Loop.
        # This avoids 'ValueError: loop argument must agree with lock' which can occur in Ray environments
        # where the lock's captured loop (get_running_loop) differs from Condition's default loop check.
        # Explicitly passing the loop is deprecated/removed in Python 3.10+, so this reverse-initialization
        # is the most robust workaround.
        self.condition = asyncio.Condition()
        self.lock = self.condition._lock

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_required_samples(self):
        async with self.lock:
            self.max_required_samples = int(self.required_samples * (self.staleness_threshold + 1) * self.config.async_training.trigger_parameter_sync_step)
            self.total_train_steps = int(self.total_rollout_steps / (self.required_samples * self.config.async_training.trigger_parameter_sync_step))

            self.max_concurrent_samples = len(self.async_rollout_manager.server_handles) * 16
            self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples)
            self.max_queue_size = self.max_required_samples

            print(f"[FullyAsyncRollouter] required_samples : {self.required_samples} max_required_samples: {self.max_required_samples} max_queue_size: {self.max_queue_size} total_train_steps: {self.total_train_steps} total_rollout_steps: {self.total_rollout_steps} max_concurrent_samples: {self.max_concurrent_samples} ")

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def update_param_version(self, version: int, validate: bool = False, global_steps: int = 0, use_trainer_do_validate: bool = False, rollout_executor_timing: dict = None):
        """Update current parameter version"""
        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset staleness_samples
            self.staleness_samples = len(self.active_tasks) + self.cancel_queue.qsize() + await self.message_queue_client.get_queue_size()
            # Use timing metrics from RolloutExecutor if provided, otherwise compute from local timestamps
            timing_raw = rollout_executor_timing if rollout_executor_timing else {}
            idle_ratio = timing_raw.get("rollouter/idle_ratio") if timing_raw else None

            print(f"[FullyAsyncRollouter][Public][update_param_version] Parameter version updated from {old_version} to {version} ,reset staleness_samples to: {self.staleness_samples},idle_ratio: {idle_ratio}")
            need_validate = (
                (self.val_reward_fn is not None and self.config.rollout.test_freq > 0 and self.current_param_version % self.config.rollout.test_freq == 0 and self.current_param_version > 0)  # don't test here in the initial parameter sync
                or (validate and self.val_reward_fn is not None)
            )
            print(f"[FullyAsyncRollouter] need_validate: {need_validate},parallel_validate_and_rollout: {self.parallel_validate_and_rollout}")
            if not need_validate:
                data = ValidateMetrics(timing_raw=timing_raw, metrics=None, global_steps=global_steps, param_version=version)
            elif need_validate and not self.parallel_validate_and_rollout:
                print(f"[DEBUG][FullyAsyncRollouter][update_param_version] Running _validate_wrapper...")
                data = self._validate_wrapper(timing_raw, version, global_steps, use_trainer_do_validate)
                print(f"[DEBUG][FullyAsyncRollouter][update_param_version] _validate_wrapper COMPLETED")

            if not need_validate or not self.parallel_validate_and_rollout:
                print(f"[DEBUG][FullyAsyncRollouter][update_param_version] Putting validate data to MQ...")
                await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))
                print(f"[DEBUG][FullyAsyncRollouter][update_param_version] Validate data put to MQ")

            self.version_start_time = time.time()

        if need_validate and self.parallel_validate_and_rollout:
            if self.validate_task and not self.validate_task.done():
                print("[FullyAsyncRollouter] validate_task is running, wait last validate_task to finish")
                self.validate_task.get()
            self.validate_task = asyncio.create_task(self.do_validate_async(timing_raw, version, global_steps, use_trainer_do_validate))

    def _validate_wrapper(self, timing_raw: dict, version: int, global_steps: int = 0, use_trainer_do_validate: bool = False):
        val_metrics = None
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate(use_trainer_do_validate)
        data = ValidateMetrics(timing_raw=timing_raw, metrics=val_metrics, global_steps=global_steps, param_version=version)
        return data

    async def do_validate_async(
        self,
        timing_raw: dict,
        version: int,
        global_steps: int = 0,
        use_trainer_do_validate: bool = False,
    ):
        loop = asyncio.get_running_loop()

        data = await loop.run_in_executor(
            self.validate_executor,
            functools.partial(
                self._validate_wrapper,
                timing_raw=timing_raw,
                version=version,
                global_steps=global_steps,
                use_trainer_do_validate=use_trainer_do_validate,
            ),
        )
        await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_async_objects()
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        await self._init_async_rollout_manager()

    def _create_actor_rollout_classes(self):
        # only create rollout
        for role in [Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    async def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
        )

    async def get_server_urls(self) -> list[str]:
        """Get SGLang server URLs from async_rollout_manager."""
        if self.async_rollout_manager is None:
            return []

        # _server_address is set on each replica after launch_server()
        server_addresses = self.async_rollout_manager.server_addresses
        return [f"http://{addr}" for addr in server_addresses]

    def launch_router(self, urls: list[str], port: int = 30000):
        """Launch SGLang router with the given server URLs."""
        cmd = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--worker-urls",
            *urls,
            "--port",
            str(port),
            "--policy",
            "cache_aware",
            "--log-level",
            "warn",
        ]
        self.router_process = subprocess.Popen(cmd)
        self.router_url = f"http://{ray.util.get_node_ip_address()}:{port}"
        return self.router_url

    async def pause(self):
        """pause rollout"""
        print("[FullyAsyncRollouter][Public][Pause] partial rollout:", self.config.async_training.partial_rollout)
        async with self.lock:
            self.paused = True
            # Cancel all rollout tasks
            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.cancel()
                print("[FullyAsyncRollouter][Public][pause] Unfinished rollout tasks canceled")
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks, return_exceptions=True)
                self.active_tasks.clear()
                print("[FullyAsyncRollouter][Public][Pause] All active tasks completed")
            # Always clear KV cache to release GPU memory during weight synchronization,
            # regardless of partial_rollout setting.
            await self.async_rollout_manager.clear_kv_cache()
            self.monitor_loop_trigger = False

    async def abort_router(self):
        """Abort all in-flight requests on the router.

        This runs in the rollouter's event loop, so it can safely use the shared HTTP client.
        This uses /pause_generation with mode="abort" which WAITS for all requests to complete.
        """
        await abort_async(self.router_url)

    async def resume_router(self):
        """Resume generation on all workers after abort_router.

        This must be called after abort_router to allow the workers to process new requests.
        The continue_generation endpoint is fast - it just sets is_pause=False and notifies.
        """
        await continue_generation_async(self.router_url)

    async def resume(self, dependency_ref: ObjectRef = None):
        """resume rollout"""
        if dependency_ref is not None:
            ray.get(dependency_ref)
        print("[FullyAsyncRollouter][Public][Resume]")
        async with self.lock:
            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.resume()
            self.paused = False
            self.monitor_loop_trigger = True
            self.condition.notify_all()

    # async def get_statistics(self) -> dict:
    #     queue_stats = self.message_queue_client.get_statistics_sync()

    #     stats = {
    #         # monitor stats
    #         "monitor/active_tasks_size": len(self.active_tasks),
    #         "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
    #         "monitor/queue/cancel_queue_size": self.cancel_queue.qsize(),
    #         "monitor/queue/mq_queue_size": queue_stats["queue_size"],
    #         # counting stats
    #         "count/current_param_version": self.current_param_version,
    #         "count/total_generated_samples": self.total_generated_samples,
    #         "count/staleness_samples": self.staleness_samples,
    #         "count/dropped_stale_samples": self.dropped_stale_samples,
    #         # static stats
    #         "static/max_required_samples": self.max_required_samples,
    #         "static/required_samples": self.required_samples,
    #         "static/staleness_threshold": self.staleness_threshold,
    #         "static/max_queue_size": self.max_queue_size,
    #         "static/max_concurrent_samples": self.max_concurrent_samples,
    #     }

    #     return stats