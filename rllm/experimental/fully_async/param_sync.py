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

import logging
import time

import ray
from ray.util.collective import collective
from verl.utils.device import get_nccl_backend

logger = logging.getLogger(__name__)


@ray.remote
class ParameterSynchronizer:
    """
    Unified parameter synchronizer, responsible for synchronizing model parameters between actor and rollout
    Based on the mature synchronization mode implementation of one_step_off_policy
    Merges the functions of the original multiple synchronizer classes
    """

    def __init__(self, config, trainer, inference_manager, mq):
        self.config = config
        self.trainer = trainer
        self.inference_manager = inference_manager
        self.mq_client = mq
        self.actor_wg = ray.get(trainer.get_actor_wg.remote())
        self.rollout_wg = ray.get(inference_manager.get_rollout_wg.remote())

        # Basic attributes
        self.weights_info = None
        self.sync_group_initialized = False
        self.sync_group_name = "actor_rollout"
        self.validate_task = None

        # Statistics
        self.current_version = 0

        self._init_weights_info()
        self._init_sync_group()

        if self.config.async_training.checkpoint_engine.enable:
            self._init_actor_rollout_checkpoint_engine()

    def set_router_url(self, router_url):
        self.router_url = router_url

    def get_current_param_version(self) -> int:
        """Get current parameter version number"""
        return self.current_version

    def get_weights_info(self):
        """Get weights info"""
        return self.weights_info

    def _init_weights_info(self):
        self.weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(self.weights_info)

    def _init_sync_group(self):
        print("[ParameterSynchronizer] Initializing parameter synchronization group...")
        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        n_workers = len(self.actor_wg.workers + self.rollout_wg.workers)
        if self.config.trainer.device == "npu":
            master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
            master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
            self.actor_wg.create_weight_sync_group(
                master_address,
                master_port,
                0,
                n_workers,
            )
            ray.get(
                self.rollout_wg.create_weight_sync_group(
                    master_address,
                    master_port,
                    len(self.actor_wg.workers),
                    n_workers,
                )
            )
        else:
            collective.create_collective_group(
                actor_rollout_workers,
                n_workers,
                list(range(0, n_workers)),
                backend=get_nccl_backend(),
                group_name=self.sync_group_name,
            )

    def _init_actor_rollout_checkpoint_engine(self):
        ray.get(
            self.actor_wg.init_checkpoint_engine(
                rank_offset=0,
                actor_num=len(self.actor_wg.workers),
                rollout_num=len(self.rollout_wg.workers),
            )
        )
        ray.get(
            self.rollout_wg.init_checkpoint_engine(
                rank_offset=len(self.actor_wg.workers),
                actor_num=len(self.actor_wg.workers),
                rollout_num=len(self.rollout_wg.workers),
            )
        )

    def set_rollout_executor(self, rollout_executor):
        self.rollout_executor = rollout_executor

    # ------------------------------------------------------------------
    # Remote mode support
    # ------------------------------------------------------------------

    def set_remote_mode(self, sync_flag=None):
        """Enable remote mode (no local RolloutExecutor).

        In remote mode, the synchronizer pauses/resumes SGLang directly
        via HTTP and sets an ``is_syncing`` flag (via a shared Ray actor)
        that the API gateway can check to return 503 to remote clients.

        Args:
            sync_flag: A ``_SyncFlag`` Ray actor handle used to
                communicate the syncing state to the API gateway.
        """
        self._remote_mode = True
        self._sync_flag = sync_flag
        self.rollout_executor = None  # not used in remote mode

    @property
    def _is_remote_mode(self):
        return getattr(self, "_remote_mode", False)

    # ------------------------------------------------------------------
    # Weight synchronisation
    # ------------------------------------------------------------------

    def sync_weights(self, version, validate=False, global_steps=0):
        """Sync weights between trainer and rollouter, and update parameter version.

        In **local mode** (with RolloutExecutor), validation is triggered
        on the executor after weights are synced.  In **remote mode**,
        the synchronizer pauses/resumes SGLang directly and sets the
        ``is_syncing`` flag for the API gateway.

        Returns:
            dict: Timing metrics (may be empty in remote mode).
        """
        # Wait for previous validation to complete before syncing new weights
        if self.validate_task:
            print("[ParameterSynchronizer] Waiting for previous validation to complete...")
            ray.get(self.validate_task)
            self.validate_task = None

        start_time = time.time()
        self.current_version = version

        if self._is_remote_mode:
            return self._sync_weights_remote(version, start_time)
        else:
            return self._sync_weights_local(version, validate, global_steps, start_time)

    def _sync_weights_local(self, version, validate, global_steps, start_time):
        """Original local-mode sync with RolloutExecutor."""
        ray.get(self.rollout_executor.pause.remote())

        # Now safe to pause (which includes clear_kv_cache -> release_memory_occupation)
        ray.get(self.inference_manager.clear_kv_cache.remote())

        print(f"[ParameterSynchronizer] rollout paused. cost {time.time() - start_time:.2f} seconds")

        # Get timing metrics from RolloutExecutor before updating version
        rollout_executor_timing = ray.get(self.rollout_executor.update_param_version.remote(version))

        # Update staleness tracking
        ray.get(self.rollout_executor.update_staleness_tracking.remote())
        print("[ParameterSynchronizer] update_staleness_tracking completed", flush=True)

        pause_time = time.time()
        self._do_weight_sync(pause_time, start_time)

        # Resume executor (includes resume_router for SGLang generation)
        ray.get(self.rollout_executor.resume.remote())

        # Trigger validation AFTER resume so it runs with new weights
        need_validate = (self.config.rollout.test_freq > 0 and version % self.config.rollout.test_freq == 0 and version > 0) or validate
        self.validate_task = self.rollout_executor.validate.remote(version, global_steps) if need_validate else None

        return rollout_executor_timing

    def _sync_weights_remote(self, version, start_time):
        """Remote-mode sync: no RolloutExecutor, pause SGLang directly."""
        import asyncio

        from rllm.experimental.fully_async.utils import abort_async, continue_generation_async

        # Signal the API gateway that we are syncing
        sync_flag = getattr(self, "_sync_flag", None)
        if sync_flag is not None:
            ray.get(sync_flag.set.remote(True))

        # Abort in-flight SGLang requests
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(abort_async(self.router_url))
        finally:
            loop.close()

        ray.get(self.inference_manager.clear_kv_cache.remote())
        print(f"[ParameterSynchronizer] [remote] SGLang paused. cost {time.time() - start_time:.2f}s")

        pause_time = time.time()
        self._do_weight_sync(pause_time, start_time)

        # Resume SGLang generation
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(continue_generation_async(self.router_url))
        finally:
            loop.close()

        # Clear the syncing flag
        if sync_flag is not None:
            ray.get(sync_flag.set.remote(False))

        return {}  # no rollout executor timing in remote mode

    def _do_weight_sync(self, pause_time, start_time):
        """Perform the actual NCCL weight transfer (shared by both modes)."""
        rollout_name = getattr(self.config.actor_rollout_ref.rollout, "name", None)
        use_checkpoint_engine = self.config.async_training.checkpoint_engine.enable and rollout_name != "sglang"

        if use_checkpoint_engine:
            self.actor_wg.sync_rollout_weights_by_checkpoint(self.sync_group_name)
            ray.get(self.rollout_wg.sync_rollout_weights_by_checkpoint(self.sync_group_name))
        else:
            self.actor_wg.sync_rollout_weights(self.sync_group_name)
            ray.get(self.rollout_wg.sync_rollout_weights(self.sync_group_name))

        end_time = time.time()
        print(f"[ParameterSynchronizer] sync_weights success. cost {end_time - start_time:.2f}s, pause:{pause_time - start_time:.2f}s, sync:{end_time - pause_time:.2f}s")

    def wait_last_valid(self):
        print("[ParameterSynchronizer] Waiting last sync and validate...")
        start_time = time.time()
        if self.validate_task:
            ray.get(self.validate_task)
        print(f"[ParameterSynchronizer] Wait last validate cost: {time.time() - start_time:.2f} seconds")

    def rollout_executor_save_checkpoint(self, local_global_step_folder: str):
        """Trigger RolloutExecutor to save its StatefulDataLoader state.

        In remote mode this is a no-op (dataset lives on the client side).
        """
        if self._is_remote_mode:
            print(f"[ParameterSynchronizer] [remote] Skipping RolloutExecutor checkpoint (dataset on client)")
            return
        if not hasattr(self, "rollout_executor") or self.rollout_executor is None:
            raise RuntimeError("rollout_executor is not set; call set_rollout_executor() before saving checkpoint")
        print(f"[ParameterSynchronizer] Triggering RolloutExecutor checkpoint save at {local_global_step_folder} ...")
        return ray.get(self.rollout_executor.save_checkpoint.remote(local_global_step_folder))
