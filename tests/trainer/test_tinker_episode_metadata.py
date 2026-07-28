from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from rllm.trainer.tinker.tinker_backend import TinkerBackend
from rllm.types import Episode


def test_generate_episodes_preserves_data_source_for_each_rollout():
    backend = object.__new__(TinkerBackend)
    backend.rollout_engine = SimpleNamespace(set_sampling_client=Mock())
    backend.sampling_client = object()
    backend.full_config = SimpleNamespace(
        rllm=SimpleNamespace(rollout=SimpleNamespace(n=2, n_val=1))
    )
    workflow_engine = SimpleNamespace(
        execute_tasks=AsyncMock(
            return_value=[
                Episode(id="task-one:0"),
                Episode(id="task-one:1"),
            ]
        )
    )

    episodes = asyncio.run(
        backend.generate_episodes(
            [{"task_id": "task-one", "data_source": "terminal-bench-2.1"}],
            agent_workflow_engine=workflow_engine,
        )
    )

    assert [episode.info["data_source"] for episode in episodes] == [
        "terminal-bench-2.1",
        "terminal-bench-2.1",
    ]
