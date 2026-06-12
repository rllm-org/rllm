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
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive and not self.closed

    def close(self) -> None:
        self.closed = True


class _FakeGetSandbox:
    """Stand-in for ``get_sandbox``: records each call and returns a sentinel.

    When ``gate`` is set, fills block on it so a test can freeze the frontier
    and inspect the cursor before any sandbox becomes ready. ``fail_times``
    maps a task id to how many calls for it should raise before succeeding.
    """

    def __init__(self, gate: threading.Event | None = None, fail_times: dict[str, int] | None = None) -> None:
        self.gate = gate
        self.fail_times = dict(fail_times or {})
        self.calls = 0
        self.made: list[_CountingSandbox] = []

    def __call__(self, task: Task, backend: str | None, registry, install_script: str = "") -> _CountingSandbox:
        self.calls += 1
        if self.fail_times.get(task.id, 0) > 0:
            self.fail_times[task.id] -= 1
            raise RuntimeError(f"injected create failure for {task.id}")
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


# ---- liveness guard (pop never hands out a dead sandbox) -------------------


def test_pop_replaces_dead_sandbox_inline(monkeypatch):
    # The parked sandbox died (e.g. provider auto-stop); pop must close it and
    # hand the consumer a fresh replacement, without leaving a skip-credit
    # (the schedule entry WAS legitimately consumed by the original build).
    fake = _FakeGetSandbox()
    _patch(monkeypatch, fake)
    schedule = [_task("t0", "img-0:latest")]
    queue = WarmQueue(schedule, "modal", None, size=1)
    queue.start()
    try:
        assert _wait_until(lambda: len(fake.made) == 1)
        fake.made[0].alive = False

        sandbox = queue.pop(schedule[0])
        assert sandbox is not fake.made[0]  # replacement, not the corpse
        assert sandbox.is_alive()
        assert fake.made[0].closed  # corpse closed by the queue
        assert len(fake.made) == 2  # one inline replacement create
        assert not queue._credits  # replacement must not credit-skip anything
    finally:
        queue.shutdown()


def test_pop_walks_past_multiple_dead_ready_boxes(monkeypatch):
    # Several parked boxes for one env all died; pop closes each and falls
    # through to an inline create.
    fake = _FakeGetSandbox()
    _patch(monkeypatch, fake)
    group = [_task(f"g{i}", "shared:latest") for i in range(2)]
    queue = WarmQueue(group, "modal", None, size=2)
    queue.start()
    try:
        assert _wait_until(lambda: len(fake.made) == 2)
        for sandbox in fake.made:
            sandbox.alive = False

        sandbox = queue.pop(group[0])
        assert sandbox.is_alive()
        assert all(s.closed for s in fake.made[:2])
        assert len(fake.made) == 3
    finally:
        queue.shutdown()


def test_sandbox_without_is_alive_is_handed_out(monkeypatch):
    # Backends predating the protocol method are assumed alive (old behavior).
    class _LegacySandbox:
        def __init__(self, task: Task) -> None:
            self.task = task

        def close(self) -> None:
            pass

    class _LegacyFake(_FakeGetSandbox):
        def __call__(self, task: Task, backend, registry, install_script: str = ""):
            self.calls += 1
            sandbox = _LegacySandbox(task)
            self.made.append(sandbox)
            return sandbox

    fake = _LegacyFake()
    _patch(monkeypatch, fake)
    schedule = [_task("t0", "img-0:latest")]
    queue = WarmQueue(schedule, "modal", None, size=1)
    queue.start()
    try:
        assert queue.pop(schedule[0]) is fake.made[0]
        assert fake.calls == 1
    finally:
        queue.shutdown()


# ---- schedule integrity (misses and failures never disturb the cursor) ----


def test_inline_miss_credits_and_filler_skips_entry(monkeypatch):
    # size=1 keeps the filler busy on tA while tB's pop arrives early: the pop
    # self-serves inline, and the filler must later SKIP tB's schedule entry
    # instead of building a sandbox nobody will pop.
    gate = threading.Event()
    fake = _FakeGetSandbox(gate=gate)
    _patch(monkeypatch, fake)
    task_a, task_b = _task("tA", "img-a:latest"), _task("tB", "img-b:latest")
    queue = WarmQueue([task_a, task_b], "modal", None, size=1)
    queue.start()
    try:
        assert _wait_until(lambda: len(fake.made) == 1)  # filler holds tA at the gate

        popped: list = []
        consumer = threading.Thread(target=lambda: popped.append(queue.pop(task_b)))
        consumer.start()
        # The miss is recorded before the (gated) inline create runs.
        assert _wait_until(lambda: queue._credits.get(_env(task_b), 0) == 1)

        gate.set()
        consumer.join(timeout=5)
        assert popped and popped[0].task is task_b  # consumer self-served

        # Filler finishes tA, then consumes the credit: tB is skipped, never built.
        assert _wait_until(lambda: queue._cursor == 2)
        time.sleep(0.05)
        assert fake.calls == 2  # fill(tA) + inline(tB); no third build
        assert queue.pop(task_a) is fake.made[0]  # tA still served warm
    finally:
        gate.set()
        queue.shutdown()


def test_prefetch_failure_retries_once_then_succeeds(monkeypatch):
    monkeypatch.setattr("rllm.sandbox.warm_queue._PREFETCH_RETRY_BACKOFF_S", 0.01)
    fake = _FakeGetSandbox(fail_times={"t0": 1})
    _patch(monkeypatch, fake)
    schedule = [_task("t0", "img-0:latest")]
    queue = WarmQueue(schedule, "modal", None, size=1)
    queue.start()
    try:
        assert _wait_until(lambda: len(fake.made) == 1)  # retry built it
        assert fake.calls == 2
        assert queue.pop(schedule[0]) is fake.made[0]  # served warm
        assert not queue._failed.get(_env(schedule[0]))
    finally:
        queue.shutdown()


def test_prefetch_gives_up_then_pop_inline_without_credit(monkeypatch):
    # Both prefetch attempts fail: the eventual pop self-serves inline, and
    # because the schedule entry was already consumed by the failed prefetch,
    # the miss must NOT leave a skip-credit for the env.
    monkeypatch.setattr("rllm.sandbox.warm_queue._PREFETCH_RETRY_BACKOFF_S", 0.01)
    fake = _FakeGetSandbox(fail_times={"t0": 2})
    _patch(monkeypatch, fake)
    schedule = [_task("t0", "img-0:latest")]
    key = _env(schedule[0])
    queue = WarmQueue(schedule, "modal", None, size=1)
    queue.start()
    try:
        assert _wait_until(lambda: queue._failed.get(key, 0) == 1)  # gave up

        sandbox = queue.pop(schedule[0])
        assert sandbox.task is schedule[0]  # created inline for the caller
        assert fake.calls == 3  # 2 failed prefetches + 1 inline
        assert queue._failed.get(key, 0) == 0  # miss consumed the failure marker
        assert not queue._credits  # ...instead of crediting a skip
    finally:
        queue.shutdown()
