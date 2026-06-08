"""Backend-agnostic, stateful dataloader yielding rllm task dicts.

Replaces the per-backend dataloaders (verl's StatefulDataLoader over an
RLHFDataset, tinker's plain torch DataLoader). Yields ``list[dict]`` batches of
rllm task dicts directly, shuffles deterministically per epoch, and checkpoints
its position so training can resume mid-epoch.

In fully-async training the state reflects the dispatched cursor, so tasks that
were in flight at crash time are not re-run on resume.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rllm.data import Dataset


class StatefulTaskDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last = drop_last
        # State: next sample to yield is ``_order(epoch)[cursor]``.
        self._epoch = 0
        self._cursor = 0

    def __len__(self) -> int:
        """Batches per epoch."""
        n = len(self._dataset)
        return n // self._batch_size if self._drop_last else math.ceil(n / self._batch_size)

    @property
    def epoch(self) -> int:
        return self._epoch

    def _order(self, epoch: int) -> list[int]:
        indices = list(range(len(self._dataset)))
        if self._shuffle:
            random.Random(self._seed + epoch).shuffle(indices)
        return indices

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        """Yield ``list[dict]`` task-dict batches for one epoch."""
        order = self._order(self._epoch)
        n = len(order)
        pos = self._cursor
        while pos < n:
            end = pos + self._batch_size
            if end > n and self._drop_last:
                break
            batch = [self._dataset[i] for i in order[pos:end]]
            pos = end
            self._cursor = pos
            yield batch
        # Epoch exhausted: advance to the next epoch for the next pass.
        self._epoch += 1
        self._cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "cursor": self._cursor, "seed": self._seed}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._epoch = state["epoch"]
        self._cursor = state["cursor"]
        self._seed = state.get("seed", self._seed)
