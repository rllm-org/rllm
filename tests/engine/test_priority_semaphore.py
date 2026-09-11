import asyncio

import pytest

from rllm.utils.priority_semaphore import EVAL_PRIORITY, TRAIN_PRIORITY, PrioritySemaphore


def test_fast_path_and_counts():
    async def _run():
        sem = PrioritySemaphore(2)
        assert sem.available == 2 and not sem.locked()
        await sem.acquire()
        await sem.acquire()
        assert sem.available == 0 and sem.locked()
        assert sem.waiting == 0
        sem.release()
        assert sem.available == 1 and not sem.locked()

    asyncio.run(_run())


def test_higher_priority_served_first():
    """A queued eval (high) waiter preempts already-queued train (low) waiters;
    ties within a priority stay FIFO."""

    async def _run():
        sem = PrioritySemaphore(1)
        order = []
        await sem.acquire(TRAIN_PRIORITY)  # hold the only permit

        async def worker(name, prio):
            await sem.acquire(prio)
            order.append(name)
            await asyncio.sleep(0)
            sem.release()

        workers = [
            asyncio.create_task(worker("train-A", TRAIN_PRIORITY)),
            asyncio.create_task(worker("train-B", TRAIN_PRIORITY)),
            asyncio.create_task(worker("eval-1", EVAL_PRIORITY)),
            asyncio.create_task(worker("train-C", TRAIN_PRIORITY)),
        ]
        await asyncio.sleep(0.05)  # let all four enqueue
        assert sem.waiting == 4
        sem.release()  # release the held permit -> cascade through the queue
        await asyncio.gather(*workers)
        return order

    order = asyncio.run(_run())
    assert order[0] == "eval-1", order
    assert order[1:] == ["train-A", "train-B", "train-C"], order


def test_slot_ctx_releases_on_exception():
    async def _run():
        sem = PrioritySemaphore(1)
        with pytest.raises(ValueError):
            async with sem.slot(EVAL_PRIORITY):
                assert sem.available == 0
                raise ValueError("boom")
        assert sem.available == 1

    asyncio.run(_run())


def test_cancel_after_grant_hands_off_permit():
    """A waiter cancelled after being granted must pass its permit to the next
    waiter, not leak it (counter must not go negative or lose a slot)."""

    async def _run():
        sem = PrioritySemaphore(1)
        await sem.acquire()  # hold permit
        granted = []

        async def waiter(name):
            await sem.acquire(TRAIN_PRIORITY)
            granted.append(name)
            sem.release()

        t1 = asyncio.create_task(waiter("w1"))
        t2 = asyncio.create_task(waiter("w2"))
        await asyncio.sleep(0.02)
        sem.release()  # grants to w1 (sets its future result)
        t1.cancel()  # cancel w1 after grant, before it resumes
        await asyncio.sleep(0.05)
        await asyncio.gather(t1, t2, return_exceptions=True)
        return granted, sem.available

    granted, available = asyncio.run(_run())
    assert granted == ["w2"], granted
    assert available == 1, available


def test_rejects_negative_initial_value():
    with pytest.raises(ValueError):
        PrioritySemaphore(-1)
