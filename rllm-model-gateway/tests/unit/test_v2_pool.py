import asyncio
import queue
from unittest.mock import AsyncMock, Mock

import pytest
from rllm_model_gateway.v2.config import GatewayConfig
from rllm_model_gateway.v2.pool import WorkerPool
from rllm_model_gateway.v2.types import GatewayError, WorkerUnavailableError


class UnusedInferenceClient:
    pass


def _pool(num_workers: int = 2) -> WorkerPool:
    return WorkerPool(
        GatewayConfig(
            admin_key="admin",
            tokenizer_model="unused",
            num_workers=num_workers,
        ),
        UnusedInferenceClient,
        {},
        lambda: None,
    )


@pytest.mark.asyncio
async def test_sessions_are_assigned_by_active_count_then_in_flight_count() -> None:
    pool = _pool(2)
    pool.call_worker = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await pool.create_session("session-1", {"session_id": "session-1"})
    await pool.create_session("session-2", {"session_id": "session-2"})
    pool._in_flight_counts = [1, 0]
    await pool.create_session("session-3", {"session_id": "session-3"})

    assert pool.owner("session-1") == 0
    assert pool.owner("session-2") == 1
    assert pool.owner("session-3") == 1
    assert pool._active_session_counts == [1, 2]


@pytest.mark.asyncio
async def test_failed_session_creation_releases_its_worker_reservation() -> None:
    pool = _pool()
    pool.call_worker = AsyncMock(side_effect=GatewayError("create failed"))  # type: ignore[method-assign]

    with pytest.raises(GatewayError, match="create failed"):
        await pool.create_session("session-1", {"session_id": "session-1"})

    with pytest.raises(GatewayError, match="was not found"):
        pool.owner("session-1")
    assert pool._active_session_counts == [0, 0]


@pytest.mark.asyncio
async def test_duplicate_session_is_rejected_before_worker_call() -> None:
    pool = _pool()
    call_worker = AsyncMock(return_value=None)
    pool.call_worker = call_worker  # type: ignore[method-assign]
    await pool.create_session("session-1", {"session_id": "session-1"})

    with pytest.raises(GatewayError) as error:
        await pool.create_session("session-1", {"session_id": "session-1"})

    assert error.value.status_code == 409
    assert call_worker.await_count == 1


@pytest.mark.asyncio
async def test_deleting_session_fails_only_its_pending_generations() -> None:
    pool = _pool()
    pool.call_worker = AsyncMock(return_value=None)  # type: ignore[method-assign]
    await pool.create_session("session-1", {"session_id": "session-1"})
    loop = asyncio.get_running_loop()
    generation = loop.create_future()
    trace_read = loop.create_future()
    other_generation = loop.create_future()
    pool._pending = {
        "generation": (0, "session-1", "generate", loop, generation),
        "trace-read": (0, "session-1", "get_session_traces", loop, trace_read),
        "other": (1, "session-2", "generate", loop, other_generation),
    }

    await pool.delete_session("session-1", {"session_id": "session-1"})
    await asyncio.sleep(0)

    assert generation.done()
    with pytest.raises(GatewayError) as error:
        await generation
    assert error.value.status_code == 410
    assert not trace_read.done()
    assert not other_generation.done()
    with pytest.raises(GatewayError, match="was not found"):
        pool.owner("session-1")

    trace_read.cancel()
    other_generation.cancel()


@pytest.mark.asyncio
async def test_fatal_worker_failure_fails_all_pending_calls_and_shuts_down_once() -> None:
    pool = _pool()
    fatal_callback = Mock()
    pool._fatal_callback = fatal_callback
    loop = asyncio.get_running_loop()
    first = loop.create_future()
    second = loop.create_future()
    pool._pending = {
        "first": (0, "session-1", "generate", loop, first),
        "second": (1, "session-2", "get_session_traces", loop, second),
    }

    pool._fail_gateway(1, "worker exited")
    await asyncio.sleep(0)

    assert pool._pending == {}
    for future in (first, second):
        with pytest.raises(WorkerUnavailableError, match="worker 1 is unavailable"):
            await future
    with pytest.raises(WorkerUnavailableError):
        pool.health()
    fatal_callback.assert_called_once_with()

    pool._fail_gateway(0, "another failure")
    fatal_callback.assert_called_once_with()


@pytest.mark.asyncio
async def test_inference_client_update_failure_makes_gateway_fatal() -> None:
    pool = _pool()
    fatal_callback = Mock()
    pool._fatal_callback = fatal_callback
    pool.call_worker = AsyncMock(  # type: ignore[method-assign]
        side_effect=[None, GatewayError("update rejected")]
    )

    with pytest.raises(GatewayError, match="update rejected"):
        await pool.update_inference_client({"weight_version": 2})

    with pytest.raises(GatewayError, match="gateway inference-client update failed") as error:
        pool.health()
    assert error.value.status_code == 503
    fatal_callback.assert_called_once_with()


@pytest.mark.asyncio
async def test_timed_out_worker_call_cleans_up_pending_and_in_flight_state() -> None:
    pool = _pool(num_workers=1)
    request_queue: queue.Queue = queue.Queue()
    pool._request_queues = [request_queue]
    pool._started = True

    with pytest.raises(GatewayError, match="timed out") as error:
        await pool.call_worker(
            0,
            "generate",
            {"request": {}},
            session_id="session-1",
            timeout_seconds=0.001,
        )

    assert error.value.status_code == 504
    assert error.value.error_type == "timeout_error"
    assert pool._pending == {}
    assert pool._in_flight_counts == [0]
    assert request_queue.qsize() == 1
