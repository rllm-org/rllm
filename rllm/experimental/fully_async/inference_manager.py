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

import subprocess
import time

import httpx
import ray
from verl.experimental.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.net_utils import get_free_port


@ray.remote(num_cpus=10, max_concurrency=100)
class InferenceManager(FullyAsyncRayPPOTrainer):
    """
    Manages SGLang inference servers for async training.
    Responsible for:
    - Launching and managing SGLang worker processes
    - Launching the router for load balancing
    - Clearing KV cache during weight synchronization

    Does NOT handle:
    - Dataset loading (owned by RolloutExecutor)
    - Staleness/queue sizing (owned by RolloutExecutor)
    - Sample generation (owned by RolloutExecutor)
    - Pause/resume of generation (owned by RolloutExecutor)
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
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

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_critic = False
        self.use_reference_policy = False
        self.use_rm = False

        self._validate_config()

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[InferenceManager] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
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

    def launch_router(self, port: int = 30000, health_check_timeout: int = 120):
        """Launch SGLang router with the server URLs from async_rollout_manager.

        After launching, polls the router's /health endpoint until it responds
        (up to health_check_timeout seconds) before returning.
        """
        if self.async_rollout_manager is None:
            raise RuntimeError("async_rollout_manager not initialized. Call init_workers() first.")

        # Get server URLs from async_rollout_manager
        server_addresses = self.async_rollout_manager.server_addresses
        urls = [f"http://{addr}" for addr in server_addresses]

        # Auto-find available port
        ip = ray.util.get_node_ip_address()
        actual_port, sock = get_free_port(ip)
        sock.close()  # Release the socket, router will bind to this port

        print(f"[InferenceManager] Worker URLs: {urls}")
        print(f"[InferenceManager] Launching router on {ip}:{actual_port}")

        cmd = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--worker-urls",
            *urls,
            "--port",
            str(actual_port),
            "--policy",
            "cache_aware",
            "--log-level",
            "warn",
        ]
        self.router_process = subprocess.Popen(cmd)
        self.router_url = f"http://{ip}:{actual_port}"

        # Health check loop: wait until the router is ready
        print(f"[InferenceManager] Waiting for router to become healthy at {self.router_url}/health ...")
        start = time.time()
        while time.time() - start < health_check_timeout:
            try:
                resp = httpx.get(f"{self.router_url}/health", timeout=5.0)
                if resp.status_code == 200:
                    elapsed = time.time() - start
                    print(f"[InferenceManager] Router is healthy! (took {elapsed:.1f}s)")
                    return self.router_url
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            # Check if the router process has crashed
            if self.router_process.poll() is not None:
                raise RuntimeError(f"Router process exited with code {self.router_process.returncode} before becoming healthy")
            time.sleep(2)

        raise TimeoutError(f"Router at {self.router_url} did not become healthy within {health_check_timeout}s")

    async def clear_kv_cache(self):
        await self.async_rollout_manager.clear_kv_cache()
