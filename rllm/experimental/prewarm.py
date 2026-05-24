"""Environment prewarming for agent rollouts.

Allows overlapping environment initialization with GPU training by
pre-initializing environments for the next batch while the current
batch's policy update runs on GPU.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rllm.types import Task

logger = logging.getLogger(__name__)


class PrewarmStore:
    """Thread-safe store for prewarmed environment resources.

    Keyed by rollout uid (``task_id:rollout_idx``). Flow functions pop their
    prewarmed resource at the start of execution; any leftover entries are
    cleaned up via :meth:`clear`.
    """

    def __init__(self, cleanup_fn: Callable[[Any], None] | None = None) -> None:
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cleanup_fn = cleanup_fn

    def put(self, uid: str, value: Any) -> None:
        with self._lock:
            self._store[uid] = value

    def pop(self, uid: str) -> Any | None:
        with self._lock:
            return self._store.pop(uid, None)

    def clear(self) -> None:
        with self._lock:
            if self._cleanup_fn is not None:
                for uid, value in self._store.items():
                    try:
                        self._cleanup_fn(value)
                    except Exception:
                        logger.warning("Prewarm cleanup failed for uid=%s", uid)
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


@dataclass
class PrewarmResult:
    """Tracks the outcome of a prewarm batch."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    elapsed_s: float = 0.0


async def run_prewarm(
    tasks: list[dict],
    task_ids: list[str],
    repeat_times: int,
    prewarm_fn: Callable[[Task], Any],
    store: PrewarmStore,
) -> PrewarmResult:
    """Sequentially prewarm environments for all (task, rollout_idx) pairs.

    Runs ``prewarm_fn(task_obj)`` in a thread for each slot. Results are
    stored in ``store`` keyed by uid. Failures are logged and skipped —
    the flow will fall back to inline init for missing entries.
    """
    total = len(tasks) * repeat_times
    result = PrewarmResult(total=total)
    t0 = time.perf_counter()

    for task_dict, task_id in zip(tasks, task_ids):
        for rollout_idx in range(repeat_times):
            uid = f"{task_id}:{rollout_idx}"
            task_obj = Task(
                id=str(task_id),
                instruction=str(task_dict.get("question", task_dict.get("instruction", ""))),
                metadata=task_dict,
                dataset_dir=Path("."),
            )
            try:
                value = await asyncio.to_thread(prewarm_fn, task_obj)
                store.put(uid, value)
                result.completed += 1
            except asyncio.CancelledError:
                logger.info("Prewarm cancelled at %d/%d completed", result.completed, total)
                break
            except Exception as e:
                result.failed += 1
                logger.warning("Prewarm failed for %s: %s", uid, e)
        else:
            continue
        break

    result.elapsed_s = time.perf_counter() - t0
    return result
