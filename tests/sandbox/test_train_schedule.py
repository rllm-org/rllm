"""Tests for the training warm-pool schedule (rllm/sandbox/train_schedule.py).

The schedule must reproduce the live dataloader's remaining task order (GRPO
copies included) with an ``env_key`` sequence identical to what the engine
computes at rollout, and must not perturb the live loader.
"""

from pathlib import Path

import pytest

from rllm.data import Dataset, StatefulTaskDataLoader
from rllm.data.utils import interleave_tasks, task_from_row
from rllm.sandbox.snapshot import env_key_for
from rllm.sandbox.train_schedule import build_train_schedule
from rllm.types import Task

BACKEND = "daytona"  # non-docker: env_key resolves offline (no image build)


def _dict_ds(n: int) -> Dataset:
    """Dict-row dataset; two distinct envs alternating by index."""
    rows = [{"id": i, "environment": {"docker_image": f"img-{i % 2}"}} for i in range(n)]
    return Dataset(data=rows, name="t", split="train")


def _task_ds(n: int) -> Dataset:
    """Harbor-style dataset whose rows are already Task objects."""
    tasks = [Task(id=str(i), instruction="x", metadata={"environment": {"docker_image": f"img-{i % 2}"}}, dataset_dir=Path(".")) for i in range(n)]
    return Dataset(data=tasks, name="t", split="train")


def _live_env_keys(loader, *, group_size, total_epochs, backend, remaining_batches=-1):
    """The env_key sequence the live path produces, computed independently of
    build_train_schedule by walking a fresh clone of the same loader state."""
    walker = loader.clone()
    keys, emitted = [], 0
    for _epoch in range(walker.epoch, total_epochs):
        for batch in walker:
            interleaved, _ = interleave_tasks(batch, group_size)
            for item in interleaved:
                task = item if isinstance(item, Task) else task_from_row(item, str(item.get("id", "")))
                keys.append(env_key_for(task, backend))
            emitted += 1
            if 0 < remaining_batches <= emitted:
                return keys
    return keys


@pytest.mark.parametrize("make_ds", [_dict_ds, _task_ds], ids=["dict_rows", "task_rows"])
def test_env_key_sequence_matches_live(make_ds):
    """Both _as_task branches (dict rows and Task rows) match the live walk."""
    loader = StatefulTaskDataLoader(make_ds(4), 2, shuffle=False)
    schedule = build_train_schedule(loader, group_size=2, total_epochs=1)
    assert all(isinstance(t, Task) for t in schedule)
    # batches [0,1],[2,3] x group_size 2 -> rows 0,0,1,1,2,2,3,3
    expected = _live_env_keys(loader, group_size=2, total_epochs=1, backend=BACKEND)
    assert [env_key_for(t, BACKEND) for t in schedule] == expected
    assert len(schedule) == 2 * 2 * 2  # batches * batch_size * group_size


def test_build_does_not_perturb_live_loader():
    loader = StatefulTaskDataLoader(_dict_ds(10), 3, seed=7)
    it = iter(loader)
    next(it)
    next(it)  # advance 2 batches mid-epoch
    before = loader.state_dict()
    build_train_schedule(loader, group_size=4, total_epochs=3)
    assert loader.state_dict() == before


def test_resume_starts_at_live_cursor():
    """A schedule built from an advanced loader starts at the next task the live
    loop would train, and covers exactly the remaining batches."""
    ds = _dict_ds(12)
    loader = StatefulTaskDataLoader(ds, 3, seed=5)
    consumed_batches = []
    it = iter(loader)
    for _ in range(2):  # consume 2 batches of epoch 0
        consumed_batches.append(next(it))

    schedule = build_train_schedule(loader, group_size=2, total_epochs=2)

    # Reference: independent walk from the same advanced state.
    expected = _live_env_keys(loader, group_size=2, total_epochs=2, backend=BACKEND)
    assert [env_key_for(t, BACKEND) for t in schedule] == expected

    # First scheduled task == first task the resumed live loop trains.
    clone = loader.clone()
    next_batch = next(iter(clone))
    first_live, _ = interleave_tasks(next_batch, 2)
    expected_first = env_key_for(task_from_row(first_live[0], ""), BACKEND)
    assert env_key_for(schedule[0], BACKEND) == expected_first


def test_remaining_batches_truncation():
    loader = StatefulTaskDataLoader(_dict_ds(100), 4, seed=1)
    schedule = build_train_schedule(loader, group_size=8, total_epochs=10, remaining_batches=3)
    assert len(schedule) == 3 * 4 * 8  # remaining_batches * batch_size * group_size


def test_end_of_epoch_resume_no_phantom_batch():
    """A loader parked exactly at an epoch boundary starts at epoch e+1's first
    batch with no empty leading batch."""
    ds = _dict_ds(8)
    loader = StatefulTaskDataLoader(ds, 4, seed=2)
    list(loader)  # exhaust epoch 0; now epoch=1, cursor=0
    assert loader.epoch == 1

    schedule = build_train_schedule(loader, group_size=2, total_epochs=3)
    expected = _live_env_keys(loader, group_size=2, total_epochs=3, backend=BACKEND)
    assert [env_key_for(t, BACKEND) for t in schedule] == expected
    # 2 remaining epochs * 2 batches/epoch * batch_size 4 * group_size 2
    assert len(schedule) == 2 * 2 * 4 * 2
