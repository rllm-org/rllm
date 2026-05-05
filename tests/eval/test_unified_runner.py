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


# ---------------------------------------------------------------------------
# Image + workdir resolution — Modal must build the task's Dockerfile
# (instead of silently dropping it for the configured fallback) AND
# propagate the image's WORKDIR as the sandbox cwd, since Modal's
# ``Sandbox.exec()`` doesn't honour image WORKDIR the way ``docker
# exec`` does. Regression for hello-world coming back reward=0 on
# Modal but reward=1 on Docker.
# ---------------------------------------------------------------------------


class TestParseDockerfileWorkdir:
    def test_returns_last_workdir(self, tmp_path):
        from rllm.runner import _parse_dockerfile_workdir

        df = tmp_path / "Dockerfile"
        df.write_text("FROM ubuntu:24.04\nWORKDIR /tmp\nRUN echo hi\nWORKDIR /app\n")
        assert _parse_dockerfile_workdir(df) == "/app"

    def test_no_workdir_returns_none(self, tmp_path):
        from rllm.runner import _parse_dockerfile_workdir

        df = tmp_path / "Dockerfile"
        df.write_text("FROM ubuntu:24.04\nRUN echo hi\n")
        assert _parse_dockerfile_workdir(df) is None

    def test_ignores_comments_and_blanks(self, tmp_path):
        from rllm.runner import _parse_dockerfile_workdir

        df = tmp_path / "Dockerfile"
        df.write_text("FROM x\n# WORKDIR /commented-out\n\nWORKDIR /real\n")
        assert _parse_dockerfile_workdir(df) == "/real"

    def test_missing_file(self, tmp_path):
        from rllm.runner import _parse_dockerfile_workdir

        assert _parse_dockerfile_workdir(tmp_path / "nope") is None


class TestResolveWorkdir:
    """``_resolve_workdir`` order: explicit metadata → Dockerfile parse → None."""

    def test_explicit_metadata_wins(self, tmp_path):
        from rllm.runner import _resolve_workdir

        env = tmp_path / "environment"
        env.mkdir()
        (env / "Dockerfile").write_text("FROM x\nWORKDIR /from-dockerfile\n")
        task = Task(
            id="t",
            instruction="i",
            metadata={"environment": {"workdir": "/from-toml"}},
            dataset_dir=tmp_path,
        )
        assert _resolve_workdir(task) == "/from-toml"

    def test_dockerfile_fallback(self, tmp_path):
        from rllm.runner import _resolve_workdir

        env = tmp_path / "environment"
        env.mkdir()
        (env / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n")
        task = Task(id="t", instruction="i", metadata={}, dataset_dir=tmp_path)
        assert _resolve_workdir(task) == "/app"

    def test_no_dockerfile_no_metadata(self, tmp_path):
        from rllm.runner import _resolve_workdir

        task = Task(id="t", instruction="i", metadata={}, dataset_dir=tmp_path)
        assert _resolve_workdir(task) is None


class TestResolveImage:
    """``_resolve_image`` returns: built docker tag / modal.Image / config string."""

    def test_modal_with_dockerfile_returns_modal_image(self, tmp_path, monkeypatch):
        """Without this, Modal silently boots ``python:3.11-slim`` and ignores
        the task's Dockerfile — which broke every Harbor task on Modal."""
        from rllm.runner import _resolve_image

        env = tmp_path / "environment"
        env.mkdir()
        (env / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n")
        task = Task(id="t", instruction="i", metadata={}, dataset_dir=tmp_path)

        # Stub the modal import so the test runs without modal installed.
        sentinel = object()
        called: dict = {}

        class _StubImage:
            @staticmethod
            def from_dockerfile(path, **kwargs):
                called["path"] = path
                called.update(kwargs)
                return sentinel

        class _StubModal:
            Image = _StubImage

        import sys

        monkeypatch.setitem(sys.modules, "modal", _StubModal)
        result = _resolve_image(task, "modal")
        assert result is sentinel
        assert called["path"] == env / "Dockerfile"

    def test_modal_without_dockerfile_returns_configured(self, tmp_path):
        from rllm.runner import _resolve_image

        task = Task(
            id="t",
            instruction="i",
            metadata={"environment": {"docker_image": "alpine:3.20"}},
            dataset_dir=tmp_path,
        )
        assert _resolve_image(task, "modal") == "alpine:3.20"

    def test_non_docker_non_modal_backend_returns_configured(self, tmp_path):
        from rllm.runner import _resolve_image

        env = tmp_path / "environment"
        env.mkdir()
        (env / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        task = Task(
            id="t",
            instruction="i",
            metadata={"environment": {"docker_image": "fallback:1"}},
            dataset_dir=tmp_path,
        )
        # Local backend doesn't build the Dockerfile.
        assert _resolve_image(task, "local") == "fallback:1"


class TestCreateSandboxForTask:
    """``_create_sandbox_for_task`` must propagate the resolved workdir."""

    def test_workdir_passed_to_create_sandbox(self, tmp_path, monkeypatch):
        import rllm.runner as runner_mod

        env = tmp_path / "environment"
        env.mkdir()
        (env / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n")
        task = Task(id="t", instruction="i", metadata={}, dataset_dir=tmp_path)

        captured: dict = {}

        def _fake_create_sandbox(backend, *, name, image, **kwargs):
            captured["backend"] = backend
            captured["image"] = image
            captured["kwargs"] = kwargs
            return _FakeSandbox()

        # Stub out the sandbox factory + image build.
        from rllm.sandbox import sandboxed_flow

        monkeypatch.setattr(sandboxed_flow, "create_sandbox", _fake_create_sandbox)
        monkeypatch.setattr(runner_mod, "_resolve_image", lambda t, b: "stub:img")

        runner_mod._create_sandbox_for_task(task, "modal")
        assert captured["kwargs"].get("workdir") == "/app"
        assert captured["backend"] == "modal"

    def test_no_workdir_when_neither_dockerfile_nor_metadata(self, tmp_path, monkeypatch):
        import rllm.runner as runner_mod
        from rllm.sandbox import sandboxed_flow

        task = Task(id="t", instruction="i", metadata={}, dataset_dir=tmp_path)
        captured: dict = {}

        def _fake_create_sandbox(backend, *, name, image, **kwargs):
            captured["kwargs"] = kwargs
            return _FakeSandbox()

        monkeypatch.setattr(sandboxed_flow, "create_sandbox", _fake_create_sandbox)
        monkeypatch.setattr(runner_mod, "_resolve_image", lambda t, b: "stub:img")

        runner_mod._create_sandbox_for_task(task, "modal")
        # ``workdir`` is absent (not None) — keeps the sandbox factory's
        # default cwd path live for backends that have a sensible default.
        assert "workdir" not in captured["kwargs"]


# ---------------------------------------------------------------------------
# Sandbox lifecycle — runner must close even when agent_flow is a
# SandboxedAgentFlow, otherwise cloud sandboxes (Modal, Daytona) leak
# duration-billed containers on every task. Regression for the bug where
# ``teardown_sandbox`` is a no-op for externally-managed sandboxes and the
# runner's finally only closed on the non-SandboxedAgentFlow branch.
# ---------------------------------------------------------------------------


class _FakeSandbox:
    """Minimal Sandbox protocol stub that records close() calls."""

    def __init__(self) -> None:
        self.closed = 0

    def exec(self, command, timeout=None, user=None):  # noqa: ARG002
        return ""

    def upload_file(self, *_a, **_kw):
        pass

    def upload_dir(self, *_a, **_kw):
        pass

    def start_agent_process(self, *_a, **_kw):
        pass

    def get_endpoint(self, *_a, **_kw):
        return ("", {})

    def close(self) -> None:
        self.closed += 1


def test_runner_closes_sandbox_on_sandboxed_flow_path(monkeypatch, tmp_path):
    """SandboxedAgentFlow's ``teardown_sandbox`` is a no-op for externally-
    managed sandboxes (the runner is the manager). The runner MUST still
    close + deregister or Modal/Daytona containers leak per task."""
    import rllm.runner as runner_mod
    from rllm.sandbox import cleanup
    from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow

    fake = _FakeSandbox()
    teardown_calls: list[int] = []

    class _Flow(SandboxedAgentFlow):
        sandbox_backend = "docker"

        def run(self, task, config):
            return Episode(id=config.session_uid, task=task.id, trajectories=[])

        # Track teardown calls to confirm both code paths fire.
        def teardown_sandbox(self):  # type: ignore[override]
            teardown_calls.append(1)
            super().teardown_sandbox()

    flow = _Flow()
    flow.set_sandbox(fake)  # marks externally_managed=True
    cleanup.register(fake)  # mirrors what ``create_sandbox`` does

    monkeypatch.setattr(
        runner_mod,
        "_create_sandbox_for_task",
        lambda task, backend, agent_flow: (fake, "python:3.11-slim", backend or "docker"),
    )
    monkeypatch.setattr(runner_mod, "_setup_task_environment", lambda task, sandbox: None)

    class _NopEvaluator:
        def evaluate(self, task, episode):  # noqa: ARG002
            return EvalOutput(reward=0.0, is_correct=False, signals=[])

    runner = Runner(agent_flow=flow, sandbox_backend="docker", evaluator_override=_NopEvaluator())
    task = Task(id="t-1", instruction="hi", metadata={}, dataset_dir=tmp_path)
    config = AgentConfig(base_url="http://x", model="m", session_uid="eval-0")

    asyncio.run(runner.run(task, config))

    # Harness's teardown drops its sandbox ref.
    assert teardown_calls == [1]
    # The runner ALSO closed the sandbox — this is the regression that was
    # leaking Modal containers per task.
    assert fake.closed == 1
    # And deregistered, so SIGTERM doesn't try to double-close.
    with cleanup._lock:
        assert fake not in cleanup._active
