"""ShellScriptEvaluator: run a shell verifier inside a sandbox.

Used when ``dataset.toml`` / ``task.toml`` declares ``[verifier].script``
or when ``tests/test.sh`` is auto-detected. Implements the rLLM
:class:`~rllm.types.Evaluator` protocol.

Reward contract (Harbor-compatible): the script writes to one of
``/tmp/rllm/reward.json``, ``/logs/verifier/reward.json``, or
``/logs/verifier/reward.txt``. The first existing file wins.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rllm.eval.types import EvalOutput, Signal
from rllm.sandbox.protocol import Sandbox, SandboxCommandTimeout
from rllm.types import Episode, Task

logger = logging.getLogger(__name__)


# Reward file search order (first existing file wins)
_REWARD_PATHS = [
    "/tmp/rllm/reward.json",
    "/logs/verifier/reward.json",
    "/logs/verifier/reward.txt",
]


class ShellScriptEvaluator:
    """Run a verifier script inside the sandbox, parse the reward file.

    Constructed by :func:`rllm.eval._resolution._resolve_evaluator` once
    the sandbox is alive — the evaluator carries its sandbox reference
    internally instead of fishing it out of episode artifacts.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        script_path: str = "tests/test.sh",
        verifier_user: str | None = None,
        verifier_timeout: float = 600.0,
        reward_file_override: str | None = None,
    ):
        self.sandbox = sandbox
        self.script_path = script_path  # relative to the task's directory
        self.verifier_user = verifier_user
        self.verifier_timeout = verifier_timeout
        self.reward_file_override = reward_file_override

    def evaluate(self, task: Task, episode: Episode) -> EvalOutput:
        tests_dir = task.task_dir / Path(self.script_path).parent
        script_name = Path(self.script_path).name
        if not tests_dir.is_dir():
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                error="AddTestsDirError",
                metadata={"error": f"no {tests_dir} directory"},
            )

        v_user = self.verifier_user

        # Prepare reward directories
        try:
            self.sandbox.exec("mkdir -p /tmp/rllm /logs/verifier", timeout=10, user=v_user)
        except Exception:
            pass

        # Upload to /tests/ (Harbor convention — scripts may reference /tests/*.py).
        # A failed upload means the verifier can't run — a grading-infra failure,
        # not a task score; tag it so the engine doesn't read it as reward 0.
        try:
            self.sandbox.upload_dir(str(tests_dir), "/tests")
        except Exception as e:
            logger.warning("Failed to upload tests dir for %s: %s", task.id, e)
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                error="AddTestsDirError",
                metadata={"error": f"failed to upload tests dir: {e}"},
            )

        if not (tests_dir / script_name).exists():
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                error="AddTestsDirError",
                metadata={"error": f"verifier script {script_name} not found in {tests_dir}"},
            )

        # Only ``cd`` when the task explicitly declared a workdir.
        # Otherwise the Dockerfile's WORKDIR wins — required for swesmith
        # and similar harbor task families whose verifier scripts run
        # ``git`` and ``pytest`` against ``/testbed`` (the image WORKDIR)
        # and silently collect zero tests when forced into ``/workspace``.
        workdir = task.metadata.get("workdir")
        cd_prefix = f"cd {workdir} && " if workdir else ""
        try:
            self.sandbox.exec(
                f"chmod +x /tests/{script_name} && {cd_prefix}/tests/{script_name}",
                timeout=self.verifier_timeout,
                user=v_user,
            )
        except SandboxCommandTimeout as e:
            # The verifier itself blew its time budget — a grading-infra
            # failure, NOT "agent scored 0". Tag it (Harbor's VerifierTimeoutError)
            # so the engine routes it to VERIFIER_TIMEOUT and training drops the
            # untrustworthy reward instead of training on a spurious zero.
            logger.warning("Verifier timed out after %ss for %s: %s", self.verifier_timeout, task.id, e)
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                error="VerifierTimeoutError",
                metadata={"error": f"verifier timed out after {self.verifier_timeout}s"},
            )
        except Exception as e:
            # Verifier exit != 0 is the *expected* outcome when an agent
            # didn't solve the task; the reward (read from reward.txt
            # below) carries the signal. Log at debug so a benchmark of
            # 100 unsolved tasks doesn't spam 100 multi-KB stack traces.
            logger.debug("Verifier exited non-zero for %s: %s", task.id, e)

        # Read reward (as verifier — agent may not have read access)
        reward_paths = list(_REWARD_PATHS)
        if self.reward_file_override:
            reward_paths.insert(0, self.reward_file_override)
        return _read_reward_from_sandbox(self.sandbox, reward_paths, user=v_user)


# ---------------------------------------------------------------------------
# Reward parsing helpers (extracted from rllm/tasks/task.py)
# ---------------------------------------------------------------------------


def _read_reward_from_sandbox(sandbox: Sandbox, paths: list[str], user: str | None = None) -> EvalOutput:
    """Try reading reward from the sandbox at each path in order.

    Distinguishes grading-infra failures from a legitimate ``reward=0`` by
    setting :attr:`EvalOutput.error` (Harbor-aligned names): an unparseable
    file → ``VerifierOutputParseError``, an empty file → ``RewardFileEmptyError``,
    no file at all → ``RewardFileNotFoundError``. The engine promotes these to
    ``GRADING_ERROR`` so the untrustworthy reward is filtered from training.
    """
    saw_empty = False
    for path in paths:
        try:
            check = sandbox.exec(f"test -f {path} && echo yes || echo no", timeout=10, user=user).strip()
            if check != "yes":
                continue
            raw = sandbox.exec(f"cat {path}", timeout=10, user=user).strip()
        except Exception as e:
            logger.debug("Could not read reward from %s: %s", path, e)
            continue
        if not raw:
            saw_empty = True
            continue
        try:
            if path.endswith(".txt"):
                reward = float(raw)
                return EvalOutput(reward=reward, is_correct=reward >= 1.0)
            return _parse_reward_json(raw)
        except Exception as e:
            logger.warning("Could not parse reward file %s: %s", path, e)
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                error="VerifierOutputParseError",
                metadata={"error": f"could not parse reward file {path}: {e}"},
            )

    # A missing reward file means the verifier produced no verdict — a verifier/
    # infra failure, NOT a legitimate score of 0 (a correctly-failing verifier
    # writes reward 0). Surface it as a typed grading error (Harbor-aligned
    # names) so the engine promotes it to GRADING_ERROR and the untrustworthy
    # reward is filtered, distinguishable from an agent that genuinely scored 0.
    if saw_empty:
        logger.warning("Reward file present but empty at one of: %s", paths)
        return EvalOutput(reward=0.0, is_correct=False, error="RewardFileEmptyError", metadata={"error": "reward file present but empty"})
    logger.warning("No reward file found at any of: %s", paths)
    return EvalOutput(reward=0.0, is_correct=False, error="RewardFileNotFoundError", metadata={"error": "no reward file found"})


def _parse_reward_json(raw: str) -> EvalOutput:
    """Parse a JSON reward file into an EvalOutput.

    Supports both ``{"reward": 0.5}`` and Harbor-style ``{"rewards": {...}}``.
    """
    data = json.loads(raw)

    if "reward" in data:
        reward = float(data["reward"])
    elif "rewards" in data and data["rewards"]:
        reward = sum(float(v) for v in data["rewards"].values()) / len(data["rewards"])
    else:
        reward = 0.0

    is_correct = data.get("is_correct", reward >= 1.0)

    signals: list[Signal] = []
    for key, val in data.get("signals", {}).items():
        signals.append(Signal(name=key, value=float(val)))
    for key, val in data.get("rewards", {}).items():
        if key != "reward":
            signals.append(Signal(name=key, value=float(val)))

    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=signals,
        metadata=data.get("metadata", {}),
    )
