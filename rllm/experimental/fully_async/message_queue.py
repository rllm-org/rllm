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
import logging
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue = deque(maxlen=self.max_queue_size)

        self.val_queue = deque()

        try:
            if hasattr(config, "async_training") and config.async_training is not None:
                self.staleness_threshold = getattr(config.async_training, "staleness_threshold", 3)
            else:
                self.staleness_threshold = 3
        except (AttributeError, RecursionError):
            self.staleness_threshold = 3

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        print(f"[MessageQueue] initialized with max_queue_size={max_queue_size},staleness_threshold={self.staleness_threshold}")

    async def put_sample(self, sample: Any) -> bool:
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        async with self._lock:
            # If queue is full, remove the oldest sample (rarely happens)
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                self.queue.popleft()
                self.dropped_samples += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
            self.queue.append(sample)
            self.total_produced += 1

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 128 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            if is_drop:
                return False
            return True

    async def get_sample(self) -> Any | None:
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None

            # Get one sample
            data = self.queue.popleft()
            self.total_consumed += 1
            return data, len(self.queue)

    async def get_queue_size(self) -> int:
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        """Get queue statistics.

        Note: This method is intentionally lock-free to avoid blocking when
        get_sample() is waiting on an empty queue. Reading integers and len()
        is atomic enough for monitoring purposes - we may get slightly stale
        values but that's acceptable for statistics.
        """
        # No lock needed - we're just reading counters for monitoring
        # This prevents blocking when consumer is waiting in get_sample()
        return {
            "queue_size": len(self.queue),
            "total_produced": self.total_produced,
            "total_consumed": self.total_consumed,
            "dropped_samples": self.dropped_samples,
            "staleness_threshold": self.staleness_threshold,
            "max_queue_size": self.max_queue_size,
        }

    async def clear_queue(self):
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def put_validate(self, data):
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any) -> bool:
        """Put batch into queue (async)"""
        future = self.queue_actor.put_sample.remote(sample)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        return ray.get(self.queue_actor.get_validate.remote())

    def put_validate_sync(self, data: Any) -> bool:
        """Put validation data into queue (sync)"""
        return ray.get(self.queue_actor.put_validate.remote(data))

    async def get_sample(self) -> Any | None:
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    # Synchronous version of the method (deprecated)
    def put_sample_sync(self, sample: Any) -> bool:
        """Put batch into queue (sync - deprecated, use put_sample instead)"""
        return ray.get(self.queue_actor.put_sample.remote(sample))

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.queue_actor.get_sample.remote())