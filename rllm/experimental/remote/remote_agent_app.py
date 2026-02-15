"""
Remote Agent Application Helper.

Provides a ready-to-use FastAPI application that implements the remote agent
endpoint protocol.  Users supply a ``Workflow`` class and the helper handles
request parsing, episode serialization, and inference client setup.

The app uses ``RemoteRolloutEngine`` which calls the trainer's native
``POST /v1/model_response`` endpoint.  This preserves the full ``ModelOutput``
(including prompt_ids, completion_ids, logprobs) needed for RL training.

Usage
-----

1.  Subclass ``Workflow`` as usual (same as local training).

2.  Create the app and run it::

        from rllm.experimental.remote.remote_agent_app import create_remote_agent_app

        app = create_remote_agent_app(
            workflow_cls=MyWorkflow,
            workflow_args={...},
        )

        # Run with:  uvicorn my_module:app --host 0.0.0.0 --port 8000

3.  Configure the trainer to point at this endpoint::

        rllm:
          remote_agent:
            enabled: true
            endpoints:
              - "http://<agent-host>:8000"
            inference_api:
              host: "0.0.0.0"
              port: 8089


Protocol
--------
The app exposes a single endpoint:

    POST /generate_episode
    {
        "task":              dict,   # task specification
        "task_id":           str,    # unique task identifier
        "rollout_idx":       int,    # rollout index within the group
        "inference_api_url": str,    # trainer's inference API URL
        "is_validation":     bool,   # whether this is a validation rollout
        "config":            dict    # optional config overrides
    }

    Response:
    { "episode": <Episode.to_dict()> }
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rllm.agents.agent import Episode
from rllm.experimental.remote.remote_rollout_engine import RemoteRolloutEngine
from rllm.workflows.workflow import Workflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class GenerateEpisodeRequest(BaseModel):
    """Request body for the /generate_episode endpoint."""

    task: dict
    task_id: str
    rollout_idx: int = 0
    inference_api_url: str
    is_validation: bool = False
    config: dict = Field(default_factory=dict)


class GenerateEpisodeResponse(BaseModel):
    """Response body for the /generate_episode endpoint."""

    episode: dict


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_remote_agent_app(
    workflow_cls: type[Workflow],
    workflow_args: dict[str, Any] | None = None,
    n_parallel: int = 32,
    engine_kwargs: dict[str, Any] | None = None,
) -> FastAPI:
    """Create a FastAPI app that serves as a remote agent endpoint.

    The app uses ``RemoteRolloutEngine`` to call the trainer's native
    ``/v1/model_response`` endpoint, which returns the full ``ModelOutput``
    (with token IDs and logprobs) needed for RL training.

    Args:
        workflow_cls: The Workflow subclass to run for each task.
        workflow_args: Extra keyword arguments passed to ``workflow_cls()``.
        n_parallel: Size of the thread pool used by workflows.
        engine_kwargs: Extra keyword arguments forwarded to
            ``RemoteRolloutEngine`` (e.g. ``timeout``, ``max_retries``).

    Returns:
        A FastAPI application ready to be served with uvicorn.
    """
    workflow_args = workflow_args or {}
    engine_kwargs = engine_kwargs or {}

    app = FastAPI(title="rLLM Remote Agent", version="0.1.0")

    executor = ThreadPoolExecutor(max_workers=n_parallel)
    # Cache of rollout engines keyed by inference_api_url to avoid recreating
    _engines: dict[str, RemoteRolloutEngine] = {}

    def _get_engine(inference_api_url: str) -> RemoteRolloutEngine:
        if inference_api_url not in _engines:
            _engines[inference_api_url] = RemoteRolloutEngine(
                inference_api_url=inference_api_url,
                **engine_kwargs,
            )
        return _engines[inference_api_url]

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/generate_episode", response_model=GenerateEpisodeResponse)
    async def generate_episode(request: GenerateEpisodeRequest):
        engine = _get_engine(request.inference_api_url)
        uid = f"{request.task_id}:{request.rollout_idx}"

        workflow = workflow_cls(
            rollout_engine=engine,
            executor=executor,
            **workflow_args,
        )
        workflow.reset(task=request.task, uid=uid)

        episode: Episode = await workflow.run_with_termination_handling(
            task=request.task,
            uid=uid,
        )
        episode.id = uid
        episode.task = request.task

        return GenerateEpisodeResponse(episode=episode.to_dict())

    return app
