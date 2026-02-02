import asyncio
import time

import ray

from rllm.experimental.fully_async.client import RolloutClient
from rllm.experimental.fully_async.protocol import TrajectoryGroup


@ray.remote(num_cpus=10)
class RolloutExecutor:
    def __init__(self, router_url, rollout_fn, n, message_queue_client, config, tokenizer, processor, max_concurrency: int = 4096, total_rollout_steps: int = None):
        self.rollout_fn = rollout_fn
        self.n = n
        self.message_queue_client = message_queue_client
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.router_url = router_url
        self.total_rollout_steps = total_rollout_steps
        self.global_steps = 1

        # Use the passed max_concurrency value directly
        self.max_concurrency = max_concurrency

        self.client = RolloutClient(router_url=router_url, max_concurrency=self.max_concurrency)
        self.last_consumed = 0
        self.dataloader = self._create_dataloader()

        # Internal buffer for completed trajectories
        self.result_queue = asyncio.Queue()

        # Timing tracking for version_time, idle_ratio, active_time
        self.version_start_time = None  # Set when a new param version starts
        self.idle_start_time = None  # Set when rollout is paused
        self.current_param_version = 0
        self.is_paused = False
        self.continue_event = asyncio.Event()

        # Track active rollouts
        self.active_sample = 0
        self.enqueued_sample = 0
        self.max_staleness_samples = None  # fill in during fit()

    def _create_dataloader(self):
        """Create dataset and dataloader inside the actor."""
        from torch.utils.data import DataLoader

        from verl.trainer.main_ppo import create_rl_dataset
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            data_paths=self.config.data.train_files,
            data_config=self.config.data,
            tokenizer=self.tokenizer,
            processor=self.processor,
            is_train=True,
        )

        dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=self.config.data.get("shuffle", True),
            collate_fn=collate_fn,
            drop_last=self.config.data.get("drop_last", False),
        )
        return dataloader

    def pause(self):
        """Pause rollout and record idle start time for timing metrics."""
        self.is_paused = True
        if self.idle_start_time is None:
            self.idle_start_time = time.time()
        self.client.pause()
        print(f"[RolloutExecutor] Paused at {self.idle_start_time:.2f}")

    def resume(self):
        """Resume rollout."""
        self.is_paused = False
        self.idle_start_time = None
        self.continue_event.set()  # Unblock the main loop
        self.client.resume()
        print(f"[RolloutExecutor] Resumed")

    async def _watch_task(self, interval: float = 10.0):
        """Periodically print debug stats for monitoring staleness control."""
        while True:
            try:
                await asyncio.sleep(interval)

                # Get message queue stats
                mq_stats = await self.message_queue_client.get_statistics()

                print(f"\n{'=' * 60}")
                print(f"[RolloutExecutor][WATCH] Debug Stats @ {time.strftime('%H:%M:%S')}")
                print(f"  Staleness Control:")
                print(f"    active_sample:        {self.active_sample}")
                print(f"    enqueued_sample:      {self.enqueued_sample}")
                print(f"    total_in_flight:      {self.active_sample + self.enqueued_sample}")
                print(f"    max_staleness_samples:{self.max_staleness_samples}")
                print(f"    headroom:             {self.max_staleness_samples - (self.active_sample + self.enqueued_sample)}")
                print(f"    continue_event_set:   {self.continue_event.is_set() if self.continue_event else 'N/A'}")
                print(f"  Message Queue:")
                print(f"    mq_queue_size:        {mq_stats.get('queue_size', 'N/A')}")
                print(f"    mq_total_consumed:    {mq_stats.get('total_consumed', 'N/A')}")
                print(f"    mq_total_produced:    {mq_stats.get('total_produced', 'N/A')}")
                print(f"    mq_max_queue_size:    {mq_stats.get('max_queue_size', 'N/A')}")
                print(f"    mq_param_version:     {mq_stats.get('current_param_version', 'N/A')}")
                print(f"  Internal State:")
                print(f"    result_queue_size:    {self.result_queue.qsize()}")
                print(f"    current_param_version:{self.current_param_version}")
                print(f"    is_paused:            {self.is_paused}")
                print(f"    last_consumed:        {self.last_consumed}")
                print(f"{'=' * 60}\n")

            except asyncio.CancelledError:
                print("[RolloutExecutor][WATCH] Watch task cancelled")
                raise
            except Exception as e:
                print(f"[RolloutExecutor][WATCH] Error getting stats: {e}")

    async def _drain_results_to_mq(self):
        """Single loop that drains internal result queue to MessageQueue.

        This serializes all put_sample() calls to avoid lock contention
        at the MessageQueue actor level.
        """
        while True:
            try:
                serialized, param_version = await self.result_queue.get()
                put_succeeded = await self.message_queue_client.put_sample(serialized, param_version=param_version)
                # If drop happened, return permit to compensate for lost old sample
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[DrainLoop] Error: {e}")

    async def generate_trajectory_group(self, datum):
        """Generate n trajectories for one datum and put to internal queue."""
        # Acquire semaphore to limit concurrent rollout tasks
        # This prevents overwhelming the SGLang server with too many requests
        async with self.sema:
            try:
                trajectory_ls = await asyncio.gather(*[self.generate_trajectory(datum) for _ in range(self.n)])
                trajectory_gp = TrajectoryGroup(trajectories=trajectory_ls)
                # Serialize before putting in queue (trainer expects bytes and does cloudpickle.loads)
                serialized = ray.cloudpickle.dumps(trajectory_gp)
                # Put to internal queue (fast, non-blocking)
                await self.result_queue.put((serialized, self.client.cur_version))
                self.enqueued_sample += 1
            except Exception as e:
                import traceback

                error_msg = traceback.format_exc()
                print(f"[RolloutExecutor] Full traceback:\n{error_msg}")
                # Re-raise to make the error visible and stop the system if it keeps failing
                raise RuntimeError(f"[RolloutExecutor] Trajectory generation failed: {e}") from e
            finally:
                self.active_sample -= 1

    async def generate_trajectory(self, datum):
        try:
            return await self.rollout_fn(self.client, self.tokenizer, **datum)
        finally:
            pass

    async def fit(self):
        """Main loop."""
        # max_concurrency / n = max concurrent tasks (each task generates n requests)
        print(f"[RolloutExecutor] fit() STARTED (max_concurrency={self.max_concurrency}, n={self.n})", flush=True)

        self.continue_event = asyncio.Event()
        self.continue_event.set()
        print(f"[RolloutExecutor] continue_event set", flush=True)

        # Sync last_consumed and client version with current state
        print(f"[RolloutExecutor] Getting MQ statistics...", flush=True)
        stats = await self.message_queue_client.get_statistics()
        print(f"[RolloutExecutor] MQ stats: {stats}", flush=True)

        self.sema = asyncio.Semaphore(64)

        self.max_staleness_samples = stats["max_queue_size"]
        print(f"[RolloutExecutor] max_staleness_samples set to {self.max_staleness_samples}", flush=True)

        self.client.set_version(stats["current_param_version"])
        self.last_consumed = stats["total_consumed"]

        drain_task = asyncio.create_task(self._drain_results_to_mq())
        watch_task = asyncio.create_task(self._watch_task(interval=10.0))

        print(f"[RolloutExecutor] Dataloader type: {type(self.dataloader)}, len: {len(self.dataloader) if hasattr(self.dataloader, '__len__') else 'unknown'}", flush=True)

        try:
            iteration = 0
            while self.global_steps < self.total_rollout_steps:
                print(f"[RolloutExecutor] Starting epoch iteration {iteration}", flush=True)
                datum_count = 0
                for datum in self.dataloader:
                    datum_count += 1
                    if datum_count % 100 == 1:
                        print(f"[RolloutExecutor] Processing datum {datum_count}, global_steps={self.global_steps}/{self.total_rollout_steps}, active={self.active_sample}, enqueued={self.enqueued_sample}", flush=True)

                    await self.continue_event.wait()

                    if self.active_sample + self.enqueued_sample < self.max_staleness_samples:
                        self.active_sample += 1
                        asyncio.create_task(self.generate_trajectory_group(datum))
                        if self.active_sample + self.enqueued_sample >= self.max_staleness_samples:
                            self.idle_start_time = time.time()
                            print(f"[RolloutExecutor] Reached staleness limit, clearing continue_event", flush=True)
                            self.continue_event.clear()
                    else:
                        print("Error, shouldn't happen.")
                        self.continue_event.clear()

                    if self.global_steps >= self.total_rollout_steps:
                        print(f"[RolloutExecutor] Reached total_rollout_steps {self.total_rollout_steps}, stopping", flush=True)
                        break
                    self.global_steps += 1

                iteration += 1
                print(f"[RolloutExecutor] Completed epoch {iteration}, processed {datum_count} datums", flush=True)
        except Exception as e:
            import traceback

            print(f"[RolloutExecutor] EXCEPTION in main loop: {e}")
            print(f"[RolloutExecutor] Traceback:\n{traceback.format_exc()}")
            raise
        finally:
            # Cleanup
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
            print(f"[RolloutExecutor] fit() ENDED")

    async def update_staleness_tracking(self, consumed_since_last_sync):
        # Wait for internal result_queue to drain to MQ before syncing
        while not self.result_queue.empty():
            await asyncio.sleep(0.5)

        mq_stats = await self.message_queue_client.get_statistics()
        mq_queue_size = mq_stats.get("queue_size", 0)
        mq_total_consumed = mq_stats.get("total_consumed", "N/A")
        mq_total_produced = mq_stats.get("total_produced", "N/A")

        print(f"[RolloutExecutor] update_staleness_tracking CALLED with consumed={consumed_since_last_sync}, current enqueued_sample={self.enqueued_sample}, active_sample={self.active_sample}, mq_queue_size={mq_queue_size}, mq_total_consumed={mq_total_consumed}, mq_total_produced={mq_total_produced}", flush=True)
        self.enqueued_sample = mq_queue_size
        print(f"[RolloutExecutor] update_staleness_tracking DONE, new enqueued_sample={self.enqueued_sample}", flush=True)

    def update_param_version(self, version):
        """
        Update parameter version and compute timing metrics.

        Returns timing_raw dict with:
        - rollouter/active_time: Time actively generating (from version_start to idle_start)
        - rollouter/version_time: Total time for this version cycle
        - rollouter/idle_ratio: Fraction of time spent idle (1 - active_time/version_time)
        """
        timing_raw = {}
        idle_ratio = None

        # Compute timing metrics if we have both timestamps
        if self.idle_start_time is not None and self.version_start_time is not None:
            rollout_active_time = self.idle_start_time - self.version_start_time
            rollout_version_time = time.time() - self.version_start_time
            idle_ratio = 1 - rollout_active_time / rollout_version_time if rollout_version_time > 0 else 0

            timing_raw["rollouter/active_time"] = rollout_active_time
            timing_raw["rollouter/version_time"] = rollout_version_time
            timing_raw["rollouter/idle_ratio"] = idle_ratio

            # Reset idle_start_time after capturing metrics
            self.idle_start_time = None

        old_version = self.current_param_version
        self.current_param_version = version

        print(f"[RolloutExecutor] Parameter version updated from {old_version} to {version}, idle_ratio: {idle_ratio}")

        # Reset version_start_time for next version cycle
        self.version_start_time = time.time()

        # Update client version
        self.client.set_version(version)

        return timing_raw
