"""Unit tests for SyncCoordinator's continuous dispatch capacity."""

from __future__ import annotations

import asyncio

import pytest

from rllm.trainer.sync_coordinator import SyncCoordinator, SyncCoordinatorConfig


def _make(staleness=3.0, mini_batch=8, group_size=16, sync_step=1, max_concurrent=256):
    return SyncCoordinator(
        SyncCoordinatorConfig(
            mini_batch_size=mini_batch,
            group_size=group_size,
            staleness_threshold=staleness,
            trigger_parameter_sync_step=sync_step,
            max_concurrent_rollouts=max_concurrent,
        )
    )


def test_max_in_flight_groups_formula():
    # (1 + staleness) * trigger * mini_batch
    assert _make(staleness=3.0, mini_batch=8, sync_step=1).config.max_in_flight_groups == 32
    assert _make(staleness=0.0, mini_batch=8, sync_step=1).config.max_in_flight_groups == 8
    assert _make(staleness=1.0, mini_batch=16, sync_step=2).config.max_in_flight_groups == 64


def test_staleness_cap_blocks_dispatch():
    c = _make(staleness=3.0, mini_batch=8)  # cap = 32 groups
    for _ in range(32):
        assert c.has_capacity()
        c.on_group_dispatched()
    assert not c.has_capacity()  # staleness budget exhausted


def test_concurrency_cap_blocks_dispatch():
    # tiny concurrency budget so it binds before staleness
    c = _make(staleness=3.0, mini_batch=8, max_concurrent=2)
    # fill concurrency with 2 fake running tasks
    loop = asyncio.new_event_loop()
    try:

        async def _noop():
            await asyncio.sleep(3600)

        tasks = [loop.create_task(_noop()) for _ in range(2)]
        for t in tasks:
            c.track_task(t)
        assert not c.has_capacity()  # 2 running >= max_concurrent=2
        for t in tasks:
            t.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    finally:
        loop.close()
    # done-callbacks fired -> concurrency freed
    assert len(c._in_flight_tasks) == 0
    assert c.has_capacity()


def test_consume_reopens_capacity_without_sync():
    """The core less-bursty fix: consuming a group frees staleness budget and
    wakes generation immediately, with no weight sync in between."""

    async def scenario():
        c = _make(staleness=3.0, mini_batch=8)  # cap = 32
        for _ in range(32):
            c.on_group_dispatched()
        assert not c.has_capacity()

        # Generation parks on capacity.
        waiter = asyncio.create_task(c.wait_for_capacity())
        await asyncio.sleep(0)
        assert not waiter.done()

        # A single consume (no on_sync_complete!) must release the waiter.
        c.on_group_consumed()
        await asyncio.wait_for(waiter, timeout=1.0)
        assert c.has_capacity()

    asyncio.run(scenario())


def test_task_completion_reopens_concurrency():
    async def scenario():
        c = _make(staleness=3.0, mini_batch=8, max_concurrent=1)

        async def _job(evt: asyncio.Event):
            await evt.wait()

        gate = asyncio.Event()
        t = asyncio.create_task(_job(gate))
        c.track_task(t)
        await asyncio.sleep(0)
        assert not c.has_capacity()  # concurrency full (1/1)

        waiter = asyncio.create_task(c.wait_for_capacity())
        await asyncio.sleep(0)
        assert not waiter.done()

        gate.set()  # let the job finish -> done-callback signals capacity
        await asyncio.wait_for(waiter, timeout=1.0)
        assert c.has_capacity()

    asyncio.run(scenario())


def test_sync_only_bumps_version():
    c = _make()
    c.on_group_dispatched()
    in_flight_before = c._in_flight
    c.on_training_step_complete()
    assert c.should_sync()
    c.on_sync_complete()
    assert c.weight_version == 1
    assert c._steps_since_sync == 0
    # sync must NOT touch the staleness budget (continuous model)
    assert c._in_flight == in_flight_before


def test_stats_shape():
    c = _make()
    c.on_group_dispatched()
    s = c.stats()
    assert s["async/in_flight_groups"] == 1
    assert s["async/max_in_flight_groups"] == 32
    assert s["async/max_concurrent_rollouts"] == 256
    assert s["async/running_rollouts"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
