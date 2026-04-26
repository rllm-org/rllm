"""TaskEvaluator: runs a task's tests/ scripts inside the sandbox.

Backward-compatible with Harbor's reward protocol (``/logs/verifier/reward.txt``
and ``reward.json``) while also supporting the rLLM path (``/tmp/rllm/reward.json``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rllm.experimental.eval.types import EvalOutput, Signal
from rllm.tasks.task_config import load_task
from rllm.types import Episode

logger = logging.getLogger(__name__)

# Reward file search order (first match wins)
_REWARD_PATHS = [
    "/tmp/rllm/reward.json",
    "/logs/verifier/reward.json",
    "/logs/verifier/reward.txt",
]


class TaskEvaluator:
    """Evaluator that runs a task's ``tests/`` directory inside the sandbox.

    The sandbox is retrieved from ``episode.artifacts["_sandbox"]``, which
    is set by ``EvalRunner`` for sandboxed agents.
    """

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        sandbox = episode.artifacts.get("_sandbox")
        task_path = task.get("task_path")
        if task_path is None:
            return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no task_path"})

        loaded = load_task(task_path)

        if sandbox is None:
            # No sandbox — check for pre-computed reward in artifacts
            return self._evaluate_from_artifacts(episode)

        # Prepare reward directories
        try:
            sandbox.exec("mkdir -p /tmp/rllm /logs/verifier", timeout=10)
        except Exception:
            pass

        # Upload tests/ directory
        tests_dir = loaded.path / "tests"
        if tests_dir.is_dir():
            sandbox.upload_dir(str(tests_dir), "/tmp/tests")
        else:
            return EvalOutput(reward=0.0, is_correct=False, metadata={"error": f"no tests/ directory in {loaded.path}"})

        # Find and run the test script
        test_script = self._find_test_script(tests_dir)
        if test_script is None:
            return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no test.sh or test.py found in tests/"})

        timeout = float(loaded.verifier_timeout)
        try:
            sandbox.exec(
                f"chmod +x /tmp/tests/{test_script} && cd {loaded.workdir} && /tmp/tests/{test_script}",
                timeout=timeout,
            )
        except Exception as e:
            logger.warning("Test script execution error for %s: %s", loaded.task_name, e)

        # Read reward
        reward_paths = list(_REWARD_PATHS)
        if loaded.rllm.reward_file:
            reward_paths.insert(0, loaded.rllm.reward_file)

        return self._read_reward_from_sandbox(sandbox, reward_paths)

    def _read_reward_from_sandbox(self, sandbox, paths: list[str]) -> EvalOutput:
        """Try reading reward from the sandbox at each path in order."""
        for path in paths:
            try:
                raw = sandbox.exec(f"cat {path}", timeout=10).strip()
                if not raw:
                    continue
                if path.endswith(".txt"):
                    reward = float(raw)
                    return EvalOutput(reward=reward, is_correct=reward >= 1.0)
                return _parse_reward_json(raw)
            except Exception:
                continue

        return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no reward file found"})

    def _evaluate_from_artifacts(self, episode: Episode) -> EvalOutput:
        """Fall back to pre-computed reward in episode artifacts."""
        if "harbor_reward" in episode.artifacts:
            reward = float(episode.artifacts["harbor_reward"])
            is_correct = bool(episode.artifacts.get("harbor_is_correct", reward >= 1.0))
            return EvalOutput(reward=reward, is_correct=is_correct)
        return EvalOutput(reward=0.0, is_correct=False, metadata={"error": "no sandbox and no pre-computed reward"})

    @staticmethod
    def _find_test_script(tests_dir: Path) -> str | None:
        """Find the test script name (relative to tests_dir)."""
        for name in ("test.sh", "test.py", "test.bat"):
            if (tests_dir / name).exists():
                return name
        return None


def _parse_reward_json(raw: str) -> EvalOutput:
    """Parse a JSON reward file into an EvalOutput."""
    data = json.loads(raw)

    # Handle both {"reward": 0.5} and {"rewards": {"accuracy": 0.5}}
    if "reward" in data:
        reward = float(data["reward"])
    elif "rewards" in data:
        rewards = data["rewards"]
        reward = sum(rewards.values()) / max(len(rewards), 1) if rewards else 0.0
    else:
        reward = 0.0

    is_correct = data.get("is_correct", reward >= 1.0)

    signals = []
    for key, val in data.get("signals", {}).items():
        signals.append(Signal(name=key, value=float(val)))
    # Also convert Harbor-style "rewards" dict to signals
    for key, val in data.get("rewards", {}).items():
        if key != "reward":
            signals.append(Signal(name=key, value=float(val)))

    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=signals,
        metadata=data.get("metadata", {}),
    )
