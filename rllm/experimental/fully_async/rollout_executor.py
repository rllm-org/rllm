import asyncio
import time
from collections import defaultdict

import ray
from torchdata.stateful_dataloader import StatefulDataLoader

from rllm.experimental.fully_async.client import RolloutClient
from rllm.experimental.fully_async.protocol import TrajectoryGroup
from rllm.experimental.fully_async.utils import (
    abort_async,
    calculate_rollout_global_steps,
    continue_generation_async,
    load_dataloader_checkpoint,
    save_dataloader_checkpoint,
)


@ray.remote(num_cpus=10, max_concurrency=10)
class RolloutExecutor:
    def __init__(self, router_url, rollout_fn, n, config, tokenizer, processor, max_concurrency: int = 4096, total_rollout_steps: int = None, val_rollout_fn=None):
        self.rollout_fn = rollout_fn
        # Use val_rollout_fn if provided, otherwise fall back to rollout_fn for validation
        self.val_rollout_fn = val_rollout_fn if val_rollout_fn is not None else rollout_fn
        self.n = n
        self.message_queue_client = None  # Set later via set_message_queue_client
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.router_url = router_url
        self.global_steps = 1

        # Calculate max_concurrent_rollout from config: limits total concurrent individual rollouts
        num_servers = config.rollout.n_gpus_per_node * config.rollout.nnodes
        self.max_concurrent_rollout = 128 * num_servers
        print(f"[RolloutExecutor] num_servers={num_servers}, max_concurrent_rollout={self.max_concurrent_rollout}")

        self.result_dict = defaultdict(list)

        # Use the passed max_concurrency value directly
        self.max_concurrency = max_concurrency

        max_prompt_length = int(getattr(config.data, "max_prompt_length", 0) or 0)
        max_response_length = int(getattr(config.data, "max_response_length", 0) or 0)
        max_tokens = max_prompt_length + max_response_length
        # RolloutClient defaults to 32768; only override when config provides a positive budget.
        client_kwargs = dict(router_url=router_url, tokenizer=tokenizer, max_concurrency=self.max_concurrency)
        if max_tokens > 0:
            client_kwargs["max_tokens"] = max_tokens

        self.client = RolloutClient(**client_kwargs)
        self.dataloader = self._create_dataloader()
        self.val_dataloader = self._create_val_dataloader()

        # Calculate total_rollout_steps from dataloader if not provided
        if total_rollout_steps is None:
            dataset_size = len(self.dataloader.dataset)
            total_epochs = getattr(config.trainer, "total_epochs", 1)
            self.total_rollout_steps = dataset_size * total_epochs
            print(f"[RolloutExecutor] Calculated total_rollout_steps={self.total_rollout_steps} from dataset")
        else:
            self.total_rollout_steps = total_rollout_steps

        # ==================== Staleness and queue configuration ====================
        # These are calculated here since executor owns the dataset and generation logic
        staleness_threshold = config.async_training.get("staleness_threshold", 1)
        required_samples = config.async_training.required_samples
        trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step

        self.max_required_samples = int(required_samples * (staleness_threshold + 1) * trigger_parameter_sync_step)
        self.total_train_steps = int(self.total_rollout_steps / (required_samples * trigger_parameter_sync_step))
        self.max_queue_size = self.max_required_samples

        print(f"[RolloutExecutor] required_samples={required_samples} max_required_samples={self.max_required_samples} max_queue_size={self.max_queue_size} total_train_steps={self.total_train_steps} total_rollout_steps={self.total_rollout_steps}")

        # Lock for dataloader access (async safety)
        self.dataloader_lock = asyncio.Lock()

        # Internal buffer for completed trajectories
        self.trajectory_group_queue = asyncio.Queue()

        # Timing tracking for version_time, idle_ratio, active_time
        self.version_start_time = None  # Set when a new param version starts
        self.idle_start_time = None  # Set when rollout is paused
        self.current_param_version = 0
        self.is_paused = False
        self.continue_event = asyncio.Event()

        # Track active rollouts
        self.active_sample = 0
        self.enqueued_sample = 0
        self.dropped_samples = 0
        self.max_staleness_samples = None  # fill in during fit()

        # Validation tracking
        self.is_validating = False
        self.last_val_version = None
        self.last_val_reward = None
        self.val_count = 0

    def _create_dataloader(self):
        """Load dataset from DatasetRegistry and create StatefulDataLoader."""
        from rllm.data.dataset import DatasetRegistry

        dataset_name = self.config.data.train_dataset_name
        train_split = self.config.data.train_split
        dataset = DatasetRegistry.load_dataset(dataset_name, train_split)
        if dataset is None:
            raise ValueError(f"Failed to load dataset '{dataset_name}' split '{train_split}' from DatasetRegistry")
        print(f"[RolloutExecutor] Loaded dataset '{dataset_name}' split '{train_split}' with {len(dataset)} examples")

        # Create StatefulDataLoader with shuffle=True for checkpoint resumption support
        dataloader = StatefulDataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x,  # Return batches as lists without merging
        )
        return dataloader

    def _create_val_dataloader(self):
        """Load validation dataset if configured."""
        from rllm.data.dataset import DatasetRegistry

        val_dataset_name = getattr(self.config.data, "val_dataset_name", None)
        if val_dataset_name is None:
            return None

        val_split = getattr(self.config.data, "val_split", "test")
        dataset = DatasetRegistry.load_dataset(val_dataset_name, val_split)
        if dataset is None:
            return None

        print(f"[RolloutExecutor] Loaded val dataset '{val_dataset_name}' split '{val_split}' with {len(dataset)} examples")
        from torch.utils.data import DataLoader

        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    async def validate(self, param_version: int, global_steps: int = 0):
        """Run validation and send metrics to MessageQueue."""
        if self.val_dataloader is None:
            return

        import numpy as np
        from rllm.experimental.fully_async.metric_utils import ValidateMetrics

        self.is_validating = True
        self.val_count += 1
        print(f"[RolloutExecutor] Starting validation #{self.val_count} for version {param_version} ({len(self.val_dataloader)} samples)...")
        start_time = time.time()

        # Run validation rollouts concurrently, but cap max concurrency to avoid
        # spiky load on the router / servers.
        sema = asyncio.Semaphore(1024)

        async def run_one(datum):
            async with sema:
                result = await self.val_rollout_fn(self.client, self.tokenizer, **datum)
                if isinstance(result, tuple):
                    return result[0], result[1]  # reward, metrics
                return getattr(result, "reward", 0.0), getattr(result, "metadata", {}) or {}

        results = await asyncio.gather(*[run_one(batch[0]) for batch in self.val_dataloader])

        rewards = [r for r, _ in results]

        # Aggregate all numeric metrics from user-returned metadata
        all_metadata = [m for _, m in results]
        aggregated_user_metrics = {}
        if all_metadata:
            # Collect all keys that have numeric values
            for key in all_metadata[0].keys():
                values = [m.get(key) for m in all_metadata if key in m]
                # Only aggregate numeric types
                if values and all(isinstance(v, (int, float, bool)) for v in values):
                    aggregated_user_metrics[f"val/{key}"] = float(np.mean([float(v) for v in values]))

        metrics = {
            "val/avg_reward": float(np.mean(rewards)),
            "val/num_samples": len(rewards),
            "timing_s/validation": time.time() - start_time,
            **aggregated_user_metrics,  # Include all user metrics
        }
        self.is_validating = False
        self.last_val_version = param_version
        self.last_val_reward = metrics["val/avg_reward"]
        print(f"[RolloutExecutor] Validation #{self.val_count} done: avg_reward={metrics['val/avg_reward']:.4f}, took {metrics['timing_s/validation']:.1f}s")

        # Send to MessageQueue
        data = ValidateMetrics(timing_raw={}, metrics=metrics, global_steps=global_steps, param_version=param_version)
        await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

    async def save_checkpoint(self, checkpoint_folder: str):
        """Save dataloader state to checkpoint folder."""
        await save_dataloader_checkpoint(self.dataloader, self.dataloader_lock, checkpoint_folder)

    def load_checkpoint(self) -> int:
        """Load checkpoint and set global_steps. Returns trainer_global_steps."""
        trainer_global_steps = load_dataloader_checkpoint(self.dataloader, self.config)
        if trainer_global_steps > 0:
            self.global_steps = calculate_rollout_global_steps(trainer_global_steps, self.config)
            print(f"[RolloutExecutor] Set global_steps to {self.global_steps}")
        return trainer_global_steps

    def set_message_queue_client(self, message_queue_client):
        """Set message queue client after initialization."""
        self.message_queue_client = message_queue_client

    def get_max_queue_size(self):
        """Get max queue size for MessageQueue initialization."""
        return self.max_queue_size

    def get_total_train_steps(self):
        """Get total training steps for trainer initialization."""
        return self.total_train_steps

    async def pause(self):
        """Pause rollout, abort in-flight requests, and record idle start time for timing metrics."""
        self.is_paused = True
        if self.idle_start_time is None:
            self.idle_start_time = time.time()
        self.client.pause()
        # Abort all in-flight requests on the router - waits for completion
        await abort_async(self.router_url)
        print(f"[RolloutExecutor] Paused at {self.idle_start_time:.2f}")

    async def resume(self):
        """Resume rollout and SGLang generation."""
        # Resume SGLang generation first (unblocks workers)
        await continue_generation_async(self.router_url)
        self.is_paused = False
        self.idle_start_time = None
        self.continue_event.set()  # Unblock the main loop
        self.client.resume()
        print(f"[RolloutExecutor] Resumed")

    async def _watch_task(self, interval: float = 20.0):
        """Periodically print debug stats for monitoring."""
        last_watch_time = time.time()

        while True:
            try:
                pre_sleep = time.time()
                await asyncio.sleep(interval)
                post_sleep = time.time()

                event_loop_latency = (post_sleep - pre_sleep) - interval
                watch_interval = post_sleep - last_watch_time
                last_watch_time = post_sleep

                mq_stats = await self.message_queue_client.get_statistics()
                pending_tasks = len([t for t in asyncio.all_tasks() if not t.done()])
                sema_val = self.sema._value if hasattr(self, "sema") else 0
                active_rollouts = self.max_concurrent_rollout - sema_val if isinstance(sema_val, int) else 0
                continue_set = self.continue_event.is_set() if self.continue_event else "N/A"
                val_reward = f"{self.last_val_reward:.4f}" if self.last_val_reward is not None else "N/A"

                print(
                    f"""
{"=" * 80}
[RolloutExecutor][WATCH] @ {time.strftime("%H:%M:%S")}
  EventLoop: latency={event_loop_latency:.3f}s interval={watch_interval:.1f}s pending_tasks={pending_tasks}
  Staleness: active={self.active_sample} enqueued={self.enqueued_sample} dropped={self.dropped_samples} max={self.max_staleness_samples} continue={continue_set}
  Concurrency: rollouts={active_rollouts}/{self.max_concurrent_rollout} sema={sema_val} repeat={self.n}
  MessageQueue: size={mq_stats.get("queue_size", "N/A")} max={mq_stats.get("max_queue_size", "N/A")} consumed={mq_stats.get("total_consumed", "N/A")} produced={mq_stats.get("total_produced", "N/A")}
  State: traj_q={self.trajectory_group_queue.qsize()} version={self.current_param_version} paused={self.is_paused}
  Validation: validating={self.is_validating} count={self.val_count} last_version={self.last_val_version} reward={val_reward}
{"=" * 80}
""",
                    flush=True,
                )

            except asyncio.CancelledError:
                print("[RolloutExecutor][WATCH] Watch task cancelled")
                raise
            except Exception as e:
                import traceback

                print(f"[RolloutExecutor][WATCH] Error getting stats: {e}")
                print(traceback.format_exc())

    async def _drain_results_to_mq(self):
        """Single loop that drains internal result queue to MessageQueue.

        This serializes all put_sample() calls to avoid lock contention
        at the MessageQueue actor level.
        """
        while True:
            try:
                serialized = await self.trajectory_group_queue.get()
                put_succeeded = await self.message_queue_client.put_sample(serialized)
                if not put_succeeded:
                    self.dropped_samples += 1
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[DrainLoop] Error: {e}")

    async def generate_trajectory(self, idx, datum):
        result = None
        try:
            result = await self.rollout_fn(self.client, self.tokenizer, **datum)
        except Exception as e:
            import traceback

            error_msg = traceback.format_exc()
            print(f"[RolloutExecutor] Trajectory {idx} generation failed:\n{error_msg}")
        finally:
            self.result_dict[idx].append(result)
            self.sema.release()
            if len(self.result_dict[idx]) >= self.n:
                group = TrajectoryGroup(trajectories=[res for res in self.result_dict[idx] if res is not None])
                serialized = ray.cloudpickle.dumps(group)
                await self.trajectory_group_queue.put(serialized)
                del self.result_dict[idx]
                self.active_sample -= 1
                self.enqueued_sample += 1

    async def fit(self):
        """Main loop."""
        # max_concurrency / n = max concurrent tasks (each task generates n requests)
        print(f"[RolloutExecutor] fit() STARTED (max_concurrency={self.max_concurrency}, n={self.n})", flush=True)

        self.continue_event = asyncio.Event()
        self.continue_event.set()

        # Initialize version_start_time so timing metrics work from the first sync
        self.version_start_time = time.time()

        # Sync last_consumed and client version with current state
        stats = await self.message_queue_client.get_statistics()
        print(f"[RolloutExecutor] MQ stats: {stats}", flush=True)

        # Each task is self.n rollouts
        self.sema = asyncio.Semaphore(self.max_concurrent_rollout)

        self.max_staleness_samples = stats["max_queue_size"]
        print(f"[RolloutExecutor] max_staleness_samples set to {self.max_staleness_samples}", flush=True)

        drain_task = asyncio.create_task(self._drain_results_to_mq())
        watch_task = asyncio.create_task(self._watch_task(interval=20.0))

        try:
            iteration = 0
            while self.global_steps < self.total_rollout_steps:
                print(f"[RolloutExecutor] Starting epoch iteration {iteration}", flush=True)
                datum_count = 0

                # Create iterator from dataloader (must be done after load_state_dict for proper resumption)
                async with self.dataloader_lock:
                    dataloader_iter = iter(self.dataloader)

                for batch in dataloader_iter:
                    datum = batch[0]  # batch_size=1, extract single item
                    datum_count += 1
                    if datum_count % 128 == 1:
                        print(f"[RolloutExecutor] Processing datum {datum_count}, global_steps={self.global_steps}/{self.total_rollout_steps}, active={self.active_sample}, enqueued={self.enqueued_sample}", flush=True)

                    if self.active_sample + self.enqueued_sample >= self.max_staleness_samples:
                        self.continue_event.clear()

                    for idx in range(self.n):
                        await self.continue_event.wait()  # Wait BEFORE acquiring semaphore to avoid leak
                        await self.sema.acquire()
                        if idx == 0:
                            self.active_sample += 1
                        asyncio.create_task(self.generate_trajectory(datum_count, datum))

                    if self.global_steps >= self.total_rollout_steps:
                        print(f"[RolloutExecutor] Reached total_rollout_steps {self.total_rollout_steps}, stopping", flush=True)
                        break
                    self.global_steps += 1

                iteration += 1
                print(f"[RolloutExecutor] Completed epoch {iteration}, processed {datum_count} datums", flush=True)
        except Exception as e:
            import traceback

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

    async def update_staleness_tracking(self):
        # Wait for internal result_queue to drain to MQ before syncing
        while not self.trajectory_group_queue.empty():
            await asyncio.sleep(0.5)

        mq_stats = await self.message_queue_client.get_statistics()
        mq_queue_size = mq_stats.get("queue_size", 0)
        mq_total_consumed = mq_stats.get("total_consumed", "N/A")
        mq_total_produced = mq_stats.get("total_produced", "N/A")

        print(f"[RolloutExecutor] update_staleness_tracking CALLED, current enqueued_sample={self.enqueued_sample}, active_sample={self.active_sample}, mq_queue_size={mq_queue_size}, mq_total_consumed={mq_total_consumed}, mq_total_produced={mq_total_produced}", flush=True)
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

        print(f"[RolloutExecutor] Parameter version updated from {self.current_param_version} to {version}, idle_ratio: {idle_ratio}")
        self.current_param_version = version

        # Reset version_start_time for next version cycle
        self.version_start_time = time.time()

        # Update client version
        self.client.set_version(version)

        return timing_raw