"""Tests for :class:`rllm.runner.Runner` — verifier dispatch + override.

These tests cover the Runner's host-only paths (no Docker required):

- ``[verifier].name`` (registered reward fn)
- ``[verifier].module`` (Python module verifier)
- ``[verifier].import_path`` (bare callable)
- Auto-detect via ``tests/evaluate.py``
- ``evaluator_override`` short-circuits per-task resolution
- ``"missing"`` verifier raises a clear error

Sandbox-shell + python-hybrid paths are integration-tested elsewhere
since they need a real container.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rllm.eval.types import EvalOutput
from rllm.runner import Runner, build_dataset_evaluator
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubAgent:
    """Echoes a fixed answer into the trajectory output."""

    def __init__(self, answer: str = r"\boxed{4}"):
        self.answer = answer
        self.calls: list[Task] = []

    def run(self, task: Task, config: AgentConfig) -> Episode:
        self.calls.append(task)
        traj = Trajectory(
            uid="t",
            name="stub",
            task=task.id,
            steps=[Step(id="s0", input=str(task.instruction), output=self.answer)],
            output=self.answer,
        )
        return Episode(id=task.id, task=task.id, trajectories=[traj])


@pytest.fixture
def cfg():
    return AgentConfig(base_url="http://stub", model="stub-m", session_uid="s")


def _write_data_dataset(root: Path, *, verifier_block: str) -> Path:
    """Build a minimal data-style benchmark and return its dir."""
    bench = root / "bench"
    bench.mkdir()
    (bench / "data").mkdir()
    (bench / "data" / "test.jsonl").write_text('{"question":"x?","ground_truth":"4"}\n')
    (bench / "instruction.md.tpl").write_text("{{question}}\n")
    (bench / "dataset.toml").write_text('[dataset]\nname = "bench"\ninstruction_field = "question"\n\n' + verifier_block)
    return bench


# ---------------------------------------------------------------------------
# [verifier].name
# ---------------------------------------------------------------------------


def test_runner_dispatches_to_registered_reward_fn(tmp_path, cfg):
    bench = _write_data_dataset(tmp_path, verifier_block='[verifier]\nname = "math_reward_fn"\n')
    task = Task(id="0", instruction="x?", metadata={"ground_truth": "4"}, dataset_dir=bench)

    runner = Runner(agent_flow=_StubAgent(answer=r"\boxed{4}"))
    episode = asyncio.run(runner.run(task, cfg))

    assert episode.is_correct is True
    assert episode.trajectories[0].reward == 1.0


def test_runner_marks_wrong_answer_incorrect(tmp_path, cfg):
    bench = _write_data_dataset(tmp_path, verifier_block='[verifier]\nname = "math_reward_fn"\n')
    task = Task(id="0", instruction="x?", metadata={"ground_truth": "4"}, dataset_dir=bench)

    runner = Runner(agent_flow=_StubAgent(answer=r"\boxed{5}"))
    episode = asyncio.run(runner.run(task, cfg))

    assert episode.is_correct is False
    assert episode.trajectories[0].reward == 0.0


# ---------------------------------------------------------------------------
# [verifier].import_path
# ---------------------------------------------------------------------------


def test_runner_dispatches_to_import_path(tmp_path, cfg):
    bench = _write_data_dataset(
        tmp_path,
        verifier_block='[verifier]\nimport_path = "rllm.eval.reward_fns.math:evaluate"\n',
    )
    task = Task(id="0", instruction="x?", metadata={"ground_truth": "4"}, dataset_dir=bench)

    runner = Runner(agent_flow=_StubAgent(answer=r"\boxed{4}"))
    episode = asyncio.run(runner.run(task, cfg))

    assert episode.is_correct is True


# ---------------------------------------------------------------------------
# [verifier].module + tests/evaluate.py auto-detect
# ---------------------------------------------------------------------------


_VERIFIER_PY = """
def evaluate(task, episode):
    answer = (episode.trajectories[-1].output or "") if episode.trajectories else ""
    expected = task.metadata.get("ground_truth", "")
    return 1.0 if expected in answer else 0.0
"""


def test_runner_loads_python_module_verifier(tmp_path, cfg):
    bench = _write_data_dataset(tmp_path, verifier_block='[verifier]\nmodule = "tests.evaluate"\n')
    (bench / "tests").mkdir()
    (bench / "tests" / "evaluate.py").write_text(_VERIFIER_PY)
    task = Task(id="0", instruction="x?", metadata={"ground_truth": "4"}, dataset_dir=bench)

    runner = Runner(agent_flow=_StubAgent(answer="answer is 4"))
    episode = asyncio.run(runner.run(task, cfg))
    assert episode.is_correct is True
    assert episode.trajectories[0].reward == 1.0


def test_runner_auto_detects_tests_evaluate_py(tmp_path, cfg):
    """No [verifier] block → fall back to tests/evaluate.py at benchmark root."""
    bench = _write_data_dataset(tmp_path, verifier_block="")
    (bench / "tests").mkdir()
    (bench / "tests" / "evaluate.py").write_text(_VERIFIER_PY)
    task = Task(id="0", instruction="x?", metadata={"ground_truth": "5"}, dataset_dir=bench)

    runner = Runner(agent_flow=_StubAgent(answer="not 5 but 6"))
    episode = asyncio.run(runner.run(task, cfg))
    # Stub answered "not 5 but 6" — '5' substring matches → correct
    assert episode.is_correct is True


# ---------------------------------------------------------------------------
# evaluator_override short-circuits resolution
# ---------------------------------------------------------------------------


class _FixedEvaluator:
    """Always returns the same EvalOutput — used to verify override wins."""

    def __init__(self, output: EvalOutput):
        self._output = output

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        return self._output


def test_evaluator_override_bypasses_per_task_verifier(tmp_path, cfg):
    bench = _write_data_dataset(tmp_path, verifier_block='[verifier]\nname = "math_reward_fn"\n')
    task = Task(id="0", instruction="x?", metadata={"ground_truth": "4"}, dataset_dir=bench)

    # Per-task verifier would say wrong → 0; override says correct → 1
    override = _FixedEvaluator(EvalOutput(reward=1.0, is_correct=True))
    runner = Runner(agent_flow=_StubAgent(answer="totally wrong"), evaluator_override=override)
    episode = asyncio.run(runner.run(task, cfg))

    assert episode.is_correct is True
    assert episode.trajectories[0].reward == 1.0


# ---------------------------------------------------------------------------
# Missing verifier
# ---------------------------------------------------------------------------


def test_missing_verifier_raises(tmp_path, cfg):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "data").mkdir()
    (bench / "data" / "test.jsonl").write_text('{"question":"x?"}\n')
    # No dataset.toml verifier block; no tests/ either
    (bench / "dataset.toml").write_text('[dataset]\nname = "bench"\n')
    task = Task(id="0", instruction="x?", metadata={}, dataset_dir=bench)

    runner = Runner(agent_flow=_StubAgent())
    with pytest.raises(RuntimeError, match="No verifier configured"):
        asyncio.run(runner.run(task, cfg))


# ---------------------------------------------------------------------------
# build_dataset_evaluator (used by train CLI)
# ---------------------------------------------------------------------------


class TestBuildDatasetEvaluator:
    def test_registered_name(self, tmp_path):
        bench = _write_data_dataset(tmp_path, verifier_block='[verifier]\nname = "math_reward_fn"\n')
        ev = build_dataset_evaluator(bench)
        assert ev is not None
        assert hasattr(ev, "evaluate")

    def test_python_module(self, tmp_path):
        bench = _write_data_dataset(tmp_path, verifier_block='[verifier]\nmodule = "tests.evaluate"\n')
        (bench / "tests").mkdir()
        (bench / "tests" / "evaluate.py").write_text(_VERIFIER_PY)
        ev = build_dataset_evaluator(bench)
        assert ev is not None

    def test_sandbox_shell_returns_none(self, tmp_path):
        bench = tmp_path / "bench"
        bench.mkdir()
        (bench / "tests").mkdir()
        (bench / "tests" / "test.sh").write_text("#!/bin/sh\necho ok\n")
        (bench / "dataset.toml").write_text('[dataset]\nname = "x"\n')
        assert build_dataset_evaluator(bench) is None

    def test_missing_verifier_returns_none(self, tmp_path):
        bench = tmp_path / "bench"
        bench.mkdir()
        (bench / "dataset.toml").write_text('[dataset]\nname = "x"\n')
        assert build_dataset_evaluator(bench) is None
