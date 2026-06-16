"""Gateway enforcement through ``run_dataset`` and ``AgentFlowEngine.execute_tasks``.

A real gateway in front of an in-process FastAPI mock upstream, driven by an
inline flow (not the CLI or a cookbook). Confirms the gateway overwrites a
configured key (overriding the flow's own temperature) and injects extra keys
(presence_penalty / min_p / top_k) the OpenAI SDK never sent — on both the eval
and the training-rollout paths.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request

import rllm
from rllm.types import Episode, Step, Task, Trajectory

# ---------------------------------------------------------------------------
# Tiny in-process OpenAI-compatible mock upstream that records request bodies
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = {
    "id": "chatcmpl-e2e",
    "object": "chat.completion",
    "created": 0,
    "model": "mock-model",
    # Root-level prompt_token_ids + choices[].token_ids mimic a vLLM response with
    # return_token_ids=True, so training-mode (strict) trace enrichment succeeds.
    "prompt_token_ids": [1, 2, 3],
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": r"The answer is \boxed{4}"},
            "token_ids": [4, 5, 6],
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
}


class MockUpstream:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.host = "127.0.0.1"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, 0))
            self.port = s.getsockname()[1]
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/v1/models")
        async def models():
            return {"data": [{"id": "mock-model", "object": "model"}]}

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            self.requests.append(await request.json())
            return _MOCK_RESPONSE

        self._server = uvicorn.Server(uvicorn.Config(app, host=self.host, port=self.port, log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not self._server.started:
            time.sleep(0.05)
        if not self._server.started:
            raise TimeoutError("mock upstream did not start")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


@pytest.fixture
def mock_upstream():
    server = MockUpstream()
    server.start()
    yield server
    server.stop()


# ---------------------------------------------------------------------------
# Minimal flow + evaluator. The flow sends temperature=0.9 to prove the gateway
# overrides a flow-set value.
# ---------------------------------------------------------------------------


@rllm.rollout(name="e2e")
async def e2e_flow(task: Task, config) -> Episode:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    resp = await client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": str(task.instruction)}],
        temperature=0.9,  # flow's own value — gateway must override when configured
    )
    content = resp.choices[0].message.content or ""
    return Episode(
        trajectories=[Trajectory(name="e2e", steps=[Step(chat_completions=[], model_response=content, action=content)])],
        artifacts={"answer": content},
    )


@rllm.evaluator
def e2e_eval(task, episode):  # noqa: ARG001
    return 1.0


def _tasks(n: int) -> list[Task]:
    return [Task(id=str(i), instruction="2+2?", metadata={"question": "2+2?", "ground_truth": "4"}) for i in range(n)]


# ---------------------------------------------------------------------------
# eval path: run_dataset
# ---------------------------------------------------------------------------


def test_eval_run_dataset_enforces_sampling(mock_upstream):
    from rllm.eval.runner import run_dataset

    sampling = {"temperature": 0.0, "presence_penalty": 0.7, "top_k": 20}
    result, episodes = asyncio.run(
        run_dataset(
            tasks=_tasks(2),
            agent_flow=e2e_flow,
            base_url=mock_upstream.url,
            model="mock-model",
            concurrency=2,
            evaluator=e2e_eval,
            sampling_params=sampling,
        )
    )

    assert len(mock_upstream.requests) >= 2
    for req in mock_upstream.requests:
        assert req["temperature"] == 0.0, "gateway must override the flow's temperature=0.9"
        assert req["presence_penalty"] == 0.7, "extra key must be injected at the gateway"
        assert req["top_k"] == 20, "top_k must reach upstream even though the OpenAI SDK never sent it"


# ---------------------------------------------------------------------------
# training rollout path: AgentFlowEngine.execute_tasks(is_validation=False)
# ---------------------------------------------------------------------------


def test_train_rollout_enforces_train_sampling(mock_upstream):
    from rllm.engine.agentflow_engine import AgentFlowEngine
    from rllm.gateway.manager import EvalGatewayManager

    gw = EvalGatewayManager(upstream_url=mock_upstream.url, model="mock-model")
    gw.start()
    try:
        engine = AgentFlowEngine(
            agent_flow=e2e_flow,
            evaluator=e2e_eval,
            gateway=gw,
            model="mock-model",
            n_parallel_tasks=2,
            raise_on_error=True,
            train_sampling_params={"temperature": 0.5, "min_p": 0.1},
            val_sampling_params={"temperature": 0.0},
        )
        tasks = _tasks(2)
        asyncio.run(engine.execute_tasks(tasks, task_ids=[t.id for t in tasks], is_validation=False))
        engine.shutdown()
    finally:
        gw.stop()

    assert mock_upstream.requests
    # Every forwarded request carries the train-mode params (temperature overridden, extra injected).
    assert all(req["temperature"] == 0.5 for req in mock_upstream.requests)
    assert all(req.get("min_p") == 0.1 for req in mock_upstream.requests)
