from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rllm.eval.episode_store import EvalEpisodeStore
from rllm.eval.results import EvalItem
from rllm.eval.runner import run_dataset
from rllm.eval.types import EvalOutput, Signal
from rllm.types import AgentConfig, Episode, Task, Trajectory
from rllm.workflows.workflow import TerminationReason


def _task(task_id: str) -> Task:
    return Task(id=task_id, instruction="prompt", metadata={}, dataset_dir=Path("."))


def _episode(task: Task, *, reward: float = 1.0, error: str | None = None) -> Episode:
    trajectory = Trajectory(uid=f"traj-{task.id}", task=task.id, output="answer", reward=reward, signals={"coverage": reward})
    return Episode(
        id=f"episode-{task.id}",
        task=task,
        trajectories=[trajectory],
        is_correct=reward == 1.0,
        termination_reason=TerminationReason.ERROR if error else None,
        metadata={"error": {"message": error}} if error else {},
    )


def test_progress_records_are_atomic_and_errors_are_rerun(tmp_path):
    store = EvalEpisodeStore(tmp_path / "run")
    store.write_progress(0, 0, 0, _episode(_task("a")))
    store.write_progress(1, 1, 0, _episode(_task("b"), reward=0.0, error="transient"))

    successful = store.load_completed_items(successful_only=True)
    all_items = store.load_completed_items(successful_only=False)

    assert [(item.idx, item.attempt, item.task_id) for item in successful] == [(0, 0, "a")]
    assert [(item.idx, item.error) for item in all_items] == [(0, None), (1, "transient")]
    assert not list(store.progress_dir.glob("*.tmp"))


@pytest.mark.parametrize(
    "reason",
    [
        TerminationReason.MODEL_ERROR,
        TerminationReason.SANDBOX_ERROR,
        TerminationReason.GRADING_ERROR,
        TerminationReason.AGENT_SETUP_TIMEOUT,
        TerminationReason.ENV_START_TIMEOUT,
        TerminationReason.VERIFIER_TIMEOUT,
    ],
)
def test_infra_terminated_rollouts_are_rerun_on_resume(tmp_path, reason):
    store = EvalEpisodeStore(tmp_path / "run")
    store.write_progress(0, 0, 0, _episode(_task("a")))
    broken = _episode(_task("b"), reward=0.0)
    broken.termination_reason = reason
    store.write_progress(1, 1, 0, broken)

    successful = store.load_completed_items(successful_only=True)

    assert [(item.idx, item.attempt, item.task_id) for item in successful] == [(0, 0, "a")]


def test_fully_resumed_run_does_not_start_agent_lifecycle():
    class Agent:
        def prepare_eval(self, context):
            raise AssertionError("completed resume must not prepare services")

    items = [
        EvalItem(idx=1, attempt=0, task_id="b", reward=0.0, is_correct=False),
        EvalItem(idx=0, attempt=0, task_id="a", reward=1.0, is_correct=True),
    ]
    result, episodes = asyncio.run(
        run_dataset(
            [_task("a"), _task("b")],
            Agent(),
            "http://unused",
            "model",
            dataset_name="dataset",
            agent_name="agent",
            resume_items=items,
        )
    )

    assert episodes == []
    assert [(item.idx, item.task_id) for item in result.items] == [(0, "a"), (1, "b")]


@pytest.mark.parametrize(
    "items",
    [
        [EvalItem(idx=0, attempt=0, task_id="wrong", reward=1.0, is_correct=True)],
        [
            EvalItem(idx=0, attempt=0, task_id="a", reward=1.0, is_correct=True),
            EvalItem(idx=0, attempt=0, task_id="a", reward=1.0, is_correct=True),
        ],
        [EvalItem(idx=0, attempt=1, task_id="a", reward=1.0, is_correct=True)],
    ],
)
def test_resume_rejects_task_id_duplicates_and_invalid_attempts(items):
    with pytest.raises(ValueError):
        asyncio.run(
            run_dataset(
                [_task("a")],
                object(),
                "http://unused",
                "model",
                attempts=1,
                resume_items=items,
            )
        )


class _Gateway:
    async def acreate_session(self, session_id, is_validation=False, sampling_params=None):
        return session_id

    def get_session_url(self, session_id, public=True):
        return f"http://gateway/sessions/{session_id}/v1"

    async def aget_traces(self, session_id):
        return []

    async def adelete_session(self, session_id):
        return 1

    async def adelete_sessions(self, session_ids):
        return len(session_ids)


class _Evaluator:
    def evaluate(self, task, episode):
        return EvalOutput(reward=1.0, is_correct=True, signals=[Signal(name="coverage", value=1.0)])


def test_lifecycle_wraps_run_and_legacy_callback_still_works(tmp_path):
    calls = []
    callback_calls = []

    class Agent:
        def prepare_eval(self, context):
            calls.append(("prepare", context.task_count, len(context.tasks), context.run_dir))

        def finalize_eval(self, context, error):
            calls.append(("finalize", error))

        def run(self, task: Task, config: AgentConfig):
            return _episode(task)

    def legacy_callback(idx, episode):
        callback_calls.append((idx, getattr(episode.task, "id", episode.task)))

    result, _episodes = asyncio.run(
        run_dataset(
            [_task("a")],
            Agent(),
            "http://unused",
            "model",
            dataset_name="dataset",
            agent_name="agent",
            gateway=_Gateway(),
            evaluator=_Evaluator(),
            run_dir=tmp_path,
            on_episode_complete=legacy_callback,
        )
    )

    assert calls == [("prepare", 1, 1, tmp_path), ("finalize", None)]
    assert len(callback_calls) == 1
    assert callback_calls[0][0] == 0
    assert result.items[0].task_id == "a"
    assert result.items[0].signals == {"coverage": 1.0}


def test_completion_callback_streams_before_slowest_task_finishes():
    async def scenario():
        release_slow = asyncio.Event()
        first_saved = asyncio.Event()
        completed = []

        class Agent:
            async def arun(self, task: Task, config: AgentConfig):
                if task.id == "slow":
                    await release_slow.wait()
                return _episode(task)

        def callback(flat_idx, task_idx, attempt, episode):
            completed.append((flat_idx, task_idx, attempt))
            first_saved.set()

        running = asyncio.create_task(
            run_dataset(
                [_task("fast"), _task("slow")],
                Agent(),
                "http://unused",
                "model",
                dataset_name="dataset",
                agent_name="agent",
                gateway=_Gateway(),
                evaluator=_Evaluator(),
                concurrency=2,
                on_episode_complete=callback,
            )
        )
        await asyncio.wait_for(first_saved.wait(), timeout=2)
        assert completed == [(0, 0, 0)]
        assert not running.done()
        release_slow.set()
        result, _episodes = await running
        assert result.total == 2
        assert completed == [(0, 0, 0), (1, 1, 0)]

    asyncio.run(scenario())
