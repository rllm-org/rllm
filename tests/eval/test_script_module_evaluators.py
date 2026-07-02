"""Unit tests for ShellScriptEvaluator and PythonModuleEvaluator.

ShellScriptEvaluator runs a verifier inside a sandbox and reads a
reward file. We use a tiny in-memory sandbox stub to drive its file
operations and parsing logic without spinning up Docker.

PythonModuleEvaluator imports a verifier from a benchmark dir and
adapts to whichever signature the user wrote.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rllm.eval.module_evaluator import PythonModuleEvaluator, _coerce_eval_result
from rllm.eval.script_evaluator import ShellScriptEvaluator
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Step, Task, Trajectory

# ---------------------------------------------------------------------------
# In-memory sandbox stub
# ---------------------------------------------------------------------------


class _FakeSandbox:
    """Pretends to be a Sandbox by routing reads/writes through a dict.

    Files written via ``write_file()`` are visible to the
    ``test -f``/``cat`` shell commands the evaluator runs.
    """

    def __init__(self, files: dict[str, str] | None = None):
        self.files: dict[str, str] = dict(files or {})
        self.execs: list[tuple[str, str | None]] = []
        self.uploads: list[tuple[str, str]] = []

    def exec(self, cmd: str, timeout: float | None = None, user: str | None = None) -> str:
        self.execs.append((cmd, user))
        # ``test -f X && echo yes || echo no``
        if cmd.startswith("test -f "):
            path = cmd.split()[2]
            return "yes" if path in self.files else "no"
        # ``cat X``
        if cmd.startswith("cat "):
            return self.files[cmd.split()[1]]
        # mkdir / chmod / runner script — no-op
        return ""

    def upload_dir(self, src: str, dst: str) -> None:
        self.uploads.append((src, dst))


def _episode() -> Episode:
    """A minimal episode that the script evaluator can score."""
    traj = Trajectory(uid="t", name="x", task="0", steps=[Step(id="s0", input="", output="ans")], output="ans")
    return Episode(id="0", task="0", trajectories=[traj])


def _bench(tmp_path: Path) -> Path:
    bench = tmp_path / "bench"
    bench.mkdir()
    tests = bench / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/sh\necho ok\n")
    return bench


# ---------------------------------------------------------------------------
# ShellScriptEvaluator
# ---------------------------------------------------------------------------


class TestShellScriptEvaluator:
    def test_reads_reward_txt(self, tmp_path):
        bench = _bench(tmp_path)
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

        # Partial score: not correct
        sb = _FakeSandbox(files={"/logs/verifier/reward.txt": "0.75"})
        out = ShellScriptEvaluator(sandbox=sb).evaluate(task, _episode())
        assert out.reward == 0.75
        assert out.is_correct is False  # 0.75 < 1.0
        # The evaluator copies tests/ to /tests in the sandbox
        assert any(dst == "/tests" for _, dst in sb.uploads)

        # Full score: correct
        sb = _FakeSandbox(files={"/logs/verifier/reward.txt": "1.0"})
        out = ShellScriptEvaluator(sandbox=sb).evaluate(task, _episode())
        assert out.reward == 1.0
        assert out.is_correct is True

    def test_reads_reward_json(self, tmp_path):
        bench = _bench(tmp_path)
        sb = _FakeSandbox(
            files={
                "/logs/verifier/reward.json": '{"reward": 0.5, "is_correct": false, "signals": {"acc": 0.5}}',
            }
        )
        ev = ShellScriptEvaluator(sandbox=sb)
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

        out = ev.evaluate(task, _episode())
        assert out.reward == 0.5
        assert out.is_correct is False
        assert any(s.name == "acc" and s.value == 0.5 for s in out.signals)

    def test_harbor_rewards_dict_averaged(self, tmp_path):
        """Harbor-style ``{"rewards": {...}}`` — values averaged."""
        bench = _bench(tmp_path)
        sb = _FakeSandbox(
            files={"/logs/verifier/reward.json": '{"rewards": {"a": 1.0, "b": 0.0}}'},
        )
        ev = ShellScriptEvaluator(sandbox=sb)
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

        out = ev.evaluate(task, _episode())
        assert out.reward == 0.5

    def test_search_order_prefers_tmp_rllm(self, tmp_path):
        """``/tmp/rllm/reward.json`` wins over ``/logs/verifier/*``."""
        bench = _bench(tmp_path)
        sb = _FakeSandbox(
            files={
                "/tmp/rllm/reward.json": '{"reward": 1.0}',
                "/logs/verifier/reward.txt": "0.0",
            }
        )
        ev = ShellScriptEvaluator(sandbox=sb)
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

        out = ev.evaluate(task, _episode())
        assert out.reward == 1.0

    def test_no_reward_file_is_infra_error_not_score_zero(self, tmp_path):
        bench = _bench(tmp_path)
        sb = _FakeSandbox(files={})  # nothing written
        ev = ShellScriptEvaluator(sandbox=sb)
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

        # A missing reward file means the verifier produced no verdict — a verifier/infra
        # failure, distinct from a legitimate score of 0 — so it surfaces as a typed
        # grading error (was a bare RuntimeError before the error taxonomy).
        out = ev.evaluate(task, _episode())
        assert out.error == "RewardFileNotFoundError"
        assert out.is_correct is False

    def test_missing_tests_dir_short_circuits(self, tmp_path):
        bench = tmp_path / "bench-no-tests"
        bench.mkdir()
        sb = _FakeSandbox()
        ev = ShellScriptEvaluator(sandbox=sb)
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

        out = ev.evaluate(task, _episode())
        assert out.reward == 0.0
        assert "no" in out.metadata.get("error", "")
        assert out.error == "AddTestsDirError"


class TestShellScriptEvaluatorErrorTagging:
    """Grading-infra failures set EvalOutput.error (Harbor-aligned) so the engine
    routes them to an infra TerminationReason instead of a spurious reward 0."""

    def test_verifier_timeout_tagged(self, tmp_path):
        from rllm.sandbox.protocol import SandboxCommandTimeout

        bench = _bench(tmp_path)
        sb = _FakeSandbox(files={})
        base_exec = sb.exec

        def exec_with_verifier_timeout(cmd, timeout=None, user=None):
            if "/tests/test.sh" in cmd:
                raise SandboxCommandTimeout("verifier timed out")
            return base_exec(cmd, timeout=timeout, user=user)

        sb.exec = exec_with_verifier_timeout
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
        out = ShellScriptEvaluator(sandbox=sb).evaluate(task, _episode())
        assert out.error == "VerifierTimeoutError"
        assert out.reward == 0.0

    def test_no_reward_file_tagged(self, tmp_path):
        bench = _bench(tmp_path)
        sb = _FakeSandbox(files={})
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
        out = ShellScriptEvaluator(sandbox=sb).evaluate(task, _episode())
        assert out.error == "RewardFileNotFoundError"

    def test_empty_reward_file_tagged(self, tmp_path):
        bench = _bench(tmp_path)
        sb = _FakeSandbox(files={"/logs/verifier/reward.txt": ""})  # present but empty
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
        out = ShellScriptEvaluator(sandbox=sb).evaluate(task, _episode())
        assert out.error == "RewardFileEmptyError"

    def test_unparseable_reward_tagged(self, tmp_path):
        bench = _bench(tmp_path)
        sb = _FakeSandbox(files={"/logs/verifier/reward.json": "not-json{{"})
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
        out = ShellScriptEvaluator(sandbox=sb).evaluate(task, _episode())
        assert out.error == "VerifierOutputParseError"

    def test_upload_failure_tagged(self, tmp_path):
        bench = _bench(tmp_path)
        sb = _FakeSandbox(files={})

        def boom(src, dst):
            raise RuntimeError("upload failed")

        sb.upload_dir = boom
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
        out = ShellScriptEvaluator(sandbox=sb).evaluate(task, _episode())
        assert out.error == "AddTestsDirError"

    def test_legit_zero_is_not_tagged(self, tmp_path):
        """A real reward of 0 (verifier ran, wrote 0) is NOT an error."""
        bench = _bench(tmp_path)
        sb = _FakeSandbox(files={"/logs/verifier/reward.txt": "0.0"})
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)
        out = ShellScriptEvaluator(sandbox=sb).evaluate(task, _episode())
        assert out.reward == 0.0
        assert out.error is None


# ---------------------------------------------------------------------------
# PythonModuleEvaluator
# ---------------------------------------------------------------------------


def _verifier_dir(tmp_path: Path, body: str) -> Path:
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "tests").mkdir()
    (bench / "tests" / "evaluate.py").write_text(body)
    return bench


class TestPythonModuleEvaluator:
    def test_task_episode_signature(self, tmp_path):
        bench = _verifier_dir(
            tmp_path,
            """
def evaluate(task, episode):
    expected = task.metadata.get("ground_truth", "")
    answer = episode.trajectories[-1].output if episode.trajectories else ""
    return 1.0 if expected == answer else 0.0
""",
        )
        ev = PythonModuleEvaluator.from_module(bench)
        task = Task(id="0", instruction="", metadata={"ground_truth": "yes"}, dataset_dir=bench)
        traj = Trajectory(uid="t", name="x", task="0", output="yes")
        ep = Episode(trajectories=[traj])

        out = ev.evaluate(task, ep)
        assert out.reward == 1.0
        assert out.is_correct is True

    def test_metadata_trajectory_signature(self, tmp_path):
        bench = _verifier_dir(
            tmp_path,
            """
def evaluate(metadata, trajectory):
    return {"reward": 1.0 if metadata.get("ok") else 0.0, "is_correct": metadata.get("ok", False)}
""",
        )
        ev = PythonModuleEvaluator.from_module(bench)
        task = Task(id="0", instruction="", metadata={"ok": True}, dataset_dir=bench)
        ep = Episode(trajectories=[Trajectory(uid="t", name="x", task="0", output="yes")])

        out = ev.evaluate(task, ep)
        assert out.reward == 1.0
        assert out.is_correct is True

    def test_dotted_and_path_like_module_paths(self, tmp_path):
        """Both module-path forms — dotted and file-path-like — load the verifier."""
        bench = _verifier_dir(
            tmp_path,
            "def evaluate(task, episode): return 0.5\n",
        )
        task = Task(id="0", instruction="", metadata={}, dataset_dir=bench)

        ev = PythonModuleEvaluator.from_module(bench, module_path="tests.evaluate")
        assert ev.evaluate(task, Episode()).reward == 0.5

        ev = PythonModuleEvaluator.from_module(bench, module_path="tests/evaluate.py")
        assert ev.evaluate(task, Episode()).reward == 0.5

    def test_alternate_function_name(self, tmp_path):
        bench = _verifier_dir(
            tmp_path,
            "def grade(task, episode): return True\n",
        )
        ev = PythonModuleEvaluator.from_module(bench, function="grade")
        out = ev.evaluate(Task(id="0", instruction="", metadata={}, dataset_dir=bench), Episode())
        assert out.reward == 1.0
        assert out.is_correct is True

    def test_missing_file_raises(self, tmp_path):
        bench = tmp_path / "empty"
        bench.mkdir()

        with pytest.raises(FileNotFoundError):
            PythonModuleEvaluator.from_module(bench)

    def test_verifier_exception_returns_zero(self, tmp_path):
        bench = _verifier_dir(
            tmp_path,
            "def evaluate(task, episode):\n    raise RuntimeError('boom')\n",
        )
        ev = PythonModuleEvaluator.from_module(bench)
        out = ev.evaluate(Task(id="0", instruction="", metadata={}, dataset_dir=bench), Episode())
        assert out.reward == 0.0
        assert out.is_correct is False
        assert "boom" in out.metadata["error"]
        # The exception class is carried so the engine can route it to GRADING_ERROR.
        assert out.error == "RuntimeError"


# ---------------------------------------------------------------------------
# _coerce_eval_result
# ---------------------------------------------------------------------------


class TestCoerceEvalResult:
    def test_eval_output_passthrough(self):
        original = EvalOutput(reward=0.42, is_correct=True)
        assert _coerce_eval_result(original) is original

    @pytest.mark.parametrize(
        ("raw", "reward", "is_correct"),
        [
            (True, 1.0, True),
            (False, 0.0, False),
            (0.5, 0.5, False),
            (1.0, 1.0, True),
            ((0.5, True), 0.5, True),
        ],
        ids=["bool-true", "bool-false", "float-partial", "float-one", "tuple"],
    )
    def test_coercion_table(self, raw, reward, is_correct):
        out = _coerce_eval_result(raw)
        assert out.reward == reward
        assert out.is_correct is is_correct

    def test_dict_with_signals(self):
        out = _coerce_eval_result({"reward": 0.7, "is_correct": True, "signals": {"acc": 0.7}})
        assert out.reward == 0.7
        assert out.is_correct is True
        assert any(isinstance(s, Signal) and s.name == "acc" for s in out.signals)

    def test_unknown_type_returns_zero(self):
        out = _coerce_eval_result(object())
        assert out.reward == 0.0
        assert "Cannot coerce" in out.metadata["error"]
