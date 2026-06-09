"""Offline tests for the warm queue (rllm/sandbox/warm_queue.py).

No cloud backend is touched: ``get_sandbox`` is monkeypatched to return counted
sentinel sandboxes (optionally gated on an Event), so the queue's *scheduling*
— the size bound, no-early-fetch, refill-on-pop, env_key matching, the
synchronous fallback, and tail drain on shutdown — is exercised deterministically.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from rllm.sandbox.snapshot import env_key_for
from rllm.sandbox.warm_queue import WarmQueue
from rllm.types import Task


def _task(id: str, image: str) -> Task:
    return Task(
        id=id,
        instruction="do the thing",
        metadata={"environment": {"docker_image": image}, "sandbox_backend": "modal"},
        dataset_dir=Path("/nonexistent"),
        sub_dir=None,
    )


def _distinct(n: int) -> list[Task]:
    return [_task(f"t{i}", f"img-{i}:latest") for i in range(n)]


class _CountingSandbox:
    def __init__(self, task: Task) -> None:
        self.task = task
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeGetSandbox:
    """Stand-in for ``get_sandbox``: records each call and returns a sentinel.

    When ``gate`` is set, fills block on it so a test can freeze the frontier
    and inspect the cursor before any sandbox becomes ready.
    """

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate
        self.made: list[_CountingSandbox] = []

    def __call__(self, task: Task, backend: str | None, registry) -> _CountingSandbox:
        sandbox = _CountingSandbox(task)
        self.made.append(sandbox)
        if self.gate is not None:
            self.gate.wait()
        return sandbox


def _patch(monkeypatch, fake: _FakeGetSandbox) -> None:
    monkeypatch.setattr("rllm.sandbox.warm_queue.get_sandbox", fake)


def _wait_until(pred, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


def _env(task: Task) -> str:
    return env_key_for(task, "modal")


# --------------------------------------------------------------------------


def test_no_reuse_api():
    # The queue hands ownership to the consumer; guard against V1 pool regressions.
    assert not hasattr(WarmQueue, "release")
    assert not hasattr(WarmQueue, "acquire")


def test_prefetches_and_pops_matching_env(monkeypatch):
    fake = _FakeGetSandbox()
    _patch(monkeypatch, fake)
    schedule = _distinct(5)
    queue = WarmQueue(schedule, "modal", None, size=5)
    queue.start()
    try:
        for task in schedule:
            sandbox = queue.pop(task)
            assert _env(sandbox.task) == _env(task)  # prefetched for this env
            assert not sandbox.closed  # consumer owns it
        assert len(fake.made) == 5  # each env built once, no sync fallback
    finally:
        queue.shutdown()


def test_bounded_and_refills_on_pop(monkeypatch):
    gate = threading.Event()
    fake = _FakeGetSandbox(gate=gate)
    _patch(monkeypatch, fake)
    schedule = _distinct(100)
    queue = WarmQueue(schedule, "modal", None, size=20)
    queue.start()
    try:
        # Fills are gated, so the frontier freezes at `size`: only the first 20
        # tasks are ever claimed — tasks 20..99 are untouched (no early fetch).
        assert _wait_until(lambda: len(fake.made) == 20)
        time.sleep(0.05)
        assert len(fake.made) == 20
        assert queue._cursor == 20

        # Let the buffer fill, then one pop frees exactly one slot → the cursor
        # advances by exactly one (refill-on-pop).
        gate.set()
        sandbox = queue.pop(schedule[0])
        assert _env(sandbox.task) == _env(schedule[0])
        assert _wait_until(lambda: len(fake.made) == 21)
        time.sleep(0.05)
        assert queue._cursor == 21
    finally:
        gate.set()
        queue.shutdown()


def test_pop_falls_back_to_sync_create(monkeypatch):
    fake = _FakeGetSandbox()
    _patch(monkeypatch, fake)
    queue = WarmQueue([], "modal", None, size=2)  # empty schedule → nothing prefetched
    queue.start()
    try:
        orphan = _task("orphan", "img-x:latest")
        sandbox = queue.pop(orphan)
        assert sandbox.task is orphan  # created synchronously for the caller
        assert len(fake.made) == 1
    finally:
        queue.shutdown()


def test_shutdown_closes_only_unpopped_tail(monkeypatch):
    fake = _FakeGetSandbox()
    _patch(monkeypatch, fake)
    schedule = _distinct(4)
    queue = WarmQueue(schedule, "modal", None, size=4)
    queue.start()

    popped = [queue.pop(schedule[0]), queue.pop(schedule[1])]
    assert _wait_until(lambda: len(fake.made) == 4)  # all four prefetched
    queue.shutdown()

    for sandbox in fake.made:
        consumed = sandbox in popped
        assert sandbox.closed is (not consumed)  # tail closed, popped left alone


def test_shared_env_pools_across_group(monkeypatch):
    # GRPO: several schedule entries share one env_key (same image, different ids).
    fake = _FakeGetSandbox()
    _patch(monkeypatch, fake)
    group = [_task(f"g{i}", "shared:latest") for i in range(3)]
    queue = WarmQueue(group, "modal", None, size=3)
    queue.start()
    try:
        sandboxes = [queue.pop(task) for task in group]
        assert len({id(s) for s in sandboxes}) == 3  # three distinct prefetched sandboxes
        assert len(fake.made) == 3  # all from the buffer, none synchronous
        assert all(_env(s.task) == _env(group[0]) for s in sandboxes)
    finally:
        queue.shutdown()
