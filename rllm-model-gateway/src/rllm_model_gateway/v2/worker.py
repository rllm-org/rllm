import asyncio
import threading
import traceback
from multiprocessing.queues import Queue
from typing import Any

from rllm_model_gateway.v2.backend import build_backend
from rllm_model_gateway.v2.config import WorkerProcessConfig
from rllm_model_gateway.v2.types import GatewayError, GatewayRequest
from rllm_model_gateway.v2.service import GatewayService
from rllm_model_gateway.v2.tokenization import TokenizationService


class WorkerRuntime:
    def __init__(self, worker_id: int, config: WorkerProcessConfig) -> None:
        self.worker_id = worker_id
        tokenization = TokenizationService(config.tokenization)
        backend = build_backend(config.backend)
        self._service = GatewayService(tokenization, backend, cumulative=config.cumulative)

    async def handle(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "create_session":
            return self._service.create_session(
                payload["session_id"],
                payload.get("sampling_params"),
            )
        if operation == "generate":
            request = GatewayRequest.model_validate(payload["request"])
            output = await self._service.generate(request)
            return output.model_dump(mode="json")
        if operation == "get_session_traces":
            return self._service.get_session_traces(payload["session_id"])
        if operation == "delete_session":
            return self._service.delete_session(payload["session_id"])
        raise GatewayError(f"unknown worker operation: {operation}", 500, "server_error")

    async def close(self) -> None:
        await self._service.close()


async def _serve(worker_id: int, config_data: dict[str, Any], request_queue: Queue, response_queue: Queue) -> None:
    try:
        config = WorkerProcessConfig.model_validate(config_data)
        runtime = WorkerRuntime(worker_id, config)
    except Exception:
        response_queue.put({"type": "startup", "worker_id": worker_id, "ok": False, "error": traceback.format_exc()})
        return
    response_queue.put({"type": "startup", "worker_id": worker_id, "ok": True})
    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()
    tasks: set[asyncio.Task[None]] = set()

    async def execute(item: dict[str, Any]) -> None:
        try:
            value = await runtime.handle(item["operation"], item["payload"])
            result = {"call_id": item["call_id"], "ok": True, "value": value}
        except GatewayError as exc:
            result = {
                "call_id": item["call_id"],
                "ok": False,
                "error": {"message": str(exc), "status_code": exc.status_code, "error_type": exc.error_type},
            }
        except Exception as exc:
            result = {
                "call_id": item["call_id"],
                "ok": False,
                "error": {"message": f"worker error: {exc}", "status_code": 500, "error_type": "server_error"},
            }
        await asyncio.to_thread(response_queue.put, result)

    def schedule(item: dict[str, Any]) -> None:
        task = asyncio.create_task(execute(item))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    def pump() -> None:
        while True:
            item = request_queue.get()
            if item is None:
                loop.call_soon_threadsafe(stopping.set)
                return
            loop.call_soon_threadsafe(schedule, item)

    pump_thread = threading.Thread(target=pump, name=f"gateway-worker-{worker_id}-pump", daemon=True)
    pump_thread.start()
    await stopping.wait()
    if tasks:
        await asyncio.gather(*list(tasks), return_exceptions=True)
    await runtime.close()


def worker_main(worker_id: int, config_data: dict[str, Any], request_queue: Queue, response_queue: Queue) -> None:
    try:
        asyncio.run(_serve(worker_id, config_data, request_queue, response_queue))
    except BaseException:
        try:
            response_queue.put({"type": "fatal", "worker_id": worker_id, "error": traceback.format_exc()})
        finally:
            raise
